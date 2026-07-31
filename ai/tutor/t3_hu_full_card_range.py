"""History-weighted full-card physical ranges for public T3/T4 solving.

The builder accepts only a legal acting-player :class:`InfoSetKey`.  It then
constructs opponent-discard hypotheses from the canonical 54-card deck,
weights each hypothesis by a frozen behavior model's likelihood of the
opponent's observed public placements, and returns validated
:class:`JointParticle` objects.  Hidden assignments remain inside particles;
reconstructing the policy key for every particle must reproduce the original
observation exactly.

This module builds a posterior range.  It does not choose an action, evaluate
a leaf, or claim that the full-card MCCFR policy is complete.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_public_cfr import (
    Actor,
    CardRows,
    InfoSetKey,
    JointParticle,
    PrivateRecall,
    PublicHistoryEntry,
    epsilon_smoothed_likelihood,
)


ROWS = ("top", "middle", "bottom")
RANGE_MODEL = "history_weighted_full_card_discard_particles_v2"
SAMPLER = "deterministic_uniform_without_replacement_hash_priority_v1"
CONTENT_SCHEMA = "ofc_full_card_range_content/v2"
BUILD_SCHEMA = "ofc_full_card_range_build/v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_UNDEALT_BY_PHASE = {
    "t3_first": 29,
    "t3_second": 26,
    "t4_first": 23,
    "t4_second": 20,
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_snapshot(value: Any) -> Any:
    """Return the JSON value whose bytes are committed by `_canonical_sha256`."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json.loads(encoded)


def _board(rows: CardRows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _rows(board: Mapping[str, Sequence[str]]) -> CardRows:
    return tuple(tuple(sorted(board[row])) for row in ROWS)  # type: ignore[return-value]


def _recall_payload(recall: PrivateRecall) -> dict[str, Any]:
    return recall.to_canonical_dict()


@dataclass(frozen=True)
class BehaviorInfoSet:
    """Pre-action information supplied to a frozen behavior policy."""

    actor: Actor
    turn: int
    board_bb: CardRows
    board_btn: CardRows
    public_action_history: tuple[PublicHistoryEntry, ...]
    own_recall_before: PrivateRecall
    current_draw: tuple[str, ...]
    legal_action_ids: tuple[str, ...]
    fantasy_state: str | None = None

    def __post_init__(self) -> None:
        legal = tuple(sorted(str(action_id) for action_id in self.legal_action_ids))
        if not legal or len(legal) != len(set(legal)):
            raise ValueError("behavior decision needs unique legal action IDs")
        object.__setattr__(self, "legal_action_ids", legal)

    @property
    def legal_action_count(self) -> int:
        return len(self.legal_action_ids)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "actor": self.actor,
            "turn": self.turn,
            "board_bb": {row: list(cards) for row, cards in zip(ROWS, self.board_bb)},
            "board_btn": {row: list(cards) for row, cards in zip(ROWS, self.board_btn)},
            "public_action_history": [
                {
                    "turn": turn,
                    "actor": actor,
                    "placements": [[card, row] for card, row in placements],
                }
                for turn, actor, placements in self.public_action_history
            ],
            "own_recall_before": _recall_payload(self.own_recall_before),
            "current_draw": list(self.current_draw),
            "legal_action_ids": list(self.legal_action_ids),
            "fantasy_state": self.fantasy_state,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True)
class BehaviorDecision:
    """Observed action paired with the information available before that action."""

    information: BehaviorInfoSet
    observed_action_key: str

    def __post_init__(self) -> None:
        if self.observed_action_key not in self.information.legal_action_ids:
            raise ValueError("observed behavior action is not in the legal action set")

    def __getattr__(self, name: str) -> Any:
        # Preserve ergonomic read-only access for evidence and diagnostics while
        # keeping the policy input itself free of the observed action.
        return getattr(self.information, name)

    def information_digest(self) -> str:
        return self.information.digest()

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            **self.information.to_canonical_dict(),
            "observed_action_key": self.observed_action_key,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_canonical_dict())


BEHAVIOR_DISTRIBUTION_SOURCES = frozenset(
    {"model", "table", "uniform_model", "uniform_fallback"}
)


@dataclass(frozen=True)
class BehaviorDistribution:
    """Atomic policy result, including auditable coverage provenance."""

    information_digest: str
    probabilities: Mapping[str, Fraction]
    source: str
    used_fallback: bool

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(str(self.information_digest)):
            raise ValueError("behavior distribution requires a lowercase query SHA256")
        if self.source not in BEHAVIOR_DISTRIBUTION_SOURCES:
            raise ValueError(
                f"unsupported behavior distribution source: {self.source!r}"
            )
        if not isinstance(self.used_fallback, bool):
            raise TypeError("behavior used_fallback must be boolean")
        if self.used_fallback != (self.source == "uniform_fallback"):
            raise ValueError(
                "behavior used_fallback must be true exactly for uniform_fallback"
            )


class FrozenBehaviorModel(Protocol):
    """Content-addressed exact policy used only to infer a public belief range."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_sha256(self) -> str: ...

    @property
    def model_manifest(self) -> Mapping[str, Any]: ...

    def action_distribution(
        self, information: BehaviorInfoSet
    ) -> BehaviorDistribution: ...


@dataclass(frozen=True)
class UniformLegalBehaviorModel:
    """Frozen no-learning baseline; useful before a trained model is supplied."""

    model_id: str = "uniform_legal_behavior_v1"

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": self.model_id,
            "model_type": "uniform_legal",
            "promotion_eligible": False,
            "probability": "1/legal_action_count",
            "position_contract_version": POSITION_CONTRACT_VERSION,
        }

    @property
    def model_sha256(self) -> str:
        return _canonical_sha256(self.model_manifest)

    def action_distribution(
        self, information: BehaviorInfoSet
    ) -> BehaviorDistribution:
        if information.legal_action_count <= 0:
            raise ValueError("behavior decision has no legal actions")
        probability = Fraction(1, information.legal_action_count)
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=MappingProxyType(
                {action_id: probability for action_id in information.legal_action_ids}
            ),
            source="uniform_model",
            used_fallback=False,
        )


@dataclass(frozen=True)
class FrozenBehaviorTable:
    """Content-addressed table of action probabilities with uniform fallback."""

    probabilities_by_decision: Mapping[str, Mapping[str, Fraction | int | str]]
    model_id: str = "frozen_behavior_table_v1"
    uniform_fallback: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("behavior model_id must be a non-empty string")
        if not isinstance(self.promotion_eligible, bool):
            raise TypeError("behavior promotion_eligible must be boolean")
        normalized: dict[str, dict[str, Fraction]] = {}
        for decision_digest, probabilities in self.probabilities_by_decision.items():
            if not isinstance(decision_digest, str) or not _SHA256_RE.fullmatch(
                decision_digest
            ):
                raise ValueError(
                    "behavior table decision keys must be lowercase SHA256 digests"
                )
            row: dict[str, Fraction] = {}
            for action_id, raw_probability in probabilities.items():
                if isinstance(raw_probability, bool) or isinstance(raw_probability, float):
                    raise TypeError("behavior probabilities must be exact rational inputs")
                probability = Fraction(raw_probability)
                if not 0 <= probability <= 1:
                    raise ValueError("behavior probabilities must be in [0, 1]")
                row[str(action_id)] = probability
            if row and sum(row.values(), Fraction(0, 1)) != 1:
                raise ValueError("each behavior-table distribution must sum exactly to one")
            normalized[str(decision_digest)] = dict(sorted(row.items()))
        object.__setattr__(
            self,
            "probabilities_by_decision",
            MappingProxyType(
                {
                    digest: MappingProxyType(row)
                    for digest, row in sorted(normalized.items())
                }
            ),
        )

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": self.model_id,
            "model_type": "exact_probability_table",
            "uniform_fallback": self.uniform_fallback,
            "promotion_eligible": self.promotion_eligible,
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "probabilities": {
                digest: {
                    action: f"{probability.numerator}/{probability.denominator}"
                    for action, probability in probabilities.items()
                }
                for digest, probabilities in sorted(
                    self.probabilities_by_decision.items()
                )
            },
        }

    @property
    def model_sha256(self) -> str:
        return _canonical_sha256(self.model_manifest)

    def action_distribution(
        self, information: BehaviorInfoSet
    ) -> BehaviorDistribution:
        information_digest = information.digest()
        probabilities = self.probabilities_by_decision.get(
            information_digest
        )
        if probabilities is None:
            if self.uniform_fallback:
                probability = Fraction(1, information.legal_action_count)
                return BehaviorDistribution(
                    information_digest=information_digest,
                    probabilities=MappingProxyType(
                        {
                            action_id: probability
                            for action_id in information.legal_action_ids
                        }
                    ),
                    source="uniform_fallback",
                    used_fallback=True,
                )
            raise KeyError(
                f"behavior table has no row for information {information_digest}"
            )
        if set(probabilities) != set(information.legal_action_ids):
            missing = sorted(set(information.legal_action_ids) - set(probabilities))
            extra = sorted(set(probabilities) - set(information.legal_action_ids))
            raise ValueError(
                "behavior table action coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        return BehaviorDistribution(
            information_digest=information_digest,
            probabilities=probabilities,
            source="table",
            used_fallback=False,
        )

    def action_probability(self, decision: BehaviorDecision) -> Fraction:
        """Compatibility helper for callers inspecting one observed action."""
        return self.action_distribution(decision.information).probabilities[
            decision.observed_action_key
        ]


@dataclass(frozen=True)
class FullCardRange:
    observation_digest: str
    particles: tuple[JointParticle, ...]
    particle_commitments: tuple[str, ...]
    behavior_model_id: str
    behavior_model_sha256: str
    epsilon: Fraction
    evidence_normalizer: Fraction
    effective_sample_size: Fraction
    range_sha256: str
    range_content_sha256: str
    range_build_sha256: str
    metadata: Mapping[str, Any]

    @property
    def particle_count(self) -> int:
        return len(self.particles)


def _particle_commitment(particle: JointParticle) -> str:
    return _canonical_sha256(
        {
            "bb_recall": _recall_payload(particle.bb_recall),
            "btn_recall": _recall_payload(particle.btn_recall),
            "undealt_cards": list(particle.undealt_cards),
        }
    )


def _validated_behavior_identity(
    behavior_model: FrozenBehaviorModel,
) -> tuple[str, str, Mapping[str, Any]]:
    model_id = behavior_model.model_id
    model_sha256 = behavior_model.model_sha256
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("behavior model_id must be a non-empty string")
    if not isinstance(model_sha256, str) or not _SHA256_RE.fullmatch(model_sha256):
        raise ValueError("behavior model_sha256 must be a lowercase SHA256 digest")
    model_manifest = behavior_model.model_manifest
    if not isinstance(model_manifest, Mapping):
        raise TypeError("behavior model_manifest must be a mapping")
    try:
        manifest_snapshot = _canonical_snapshot(dict(model_manifest))
    except (TypeError, ValueError) as exc:
        raise TypeError("behavior model_manifest must be canonical JSON data") from exc
    if not isinstance(manifest_snapshot, dict):
        raise TypeError("behavior model_manifest must encode a JSON object")
    if manifest_snapshot.get("model_id") != model_id:
        raise ValueError("behavior model manifest model_id does not match model_id")
    if not isinstance(manifest_snapshot.get("schema"), str) or not manifest_snapshot[
        "schema"
    ]:
        raise ValueError("behavior model manifest requires a non-empty schema")
    if manifest_snapshot.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        raise ValueError("behavior model manifest requires bb_first_v1")
    computed_sha256 = _canonical_sha256(manifest_snapshot)
    if computed_sha256 != model_sha256:
        raise ValueError(
            "behavior model_sha256 does not match the canonical model_manifest"
        )
    return model_id, model_sha256, MappingProxyType(manifest_snapshot)


def _validated_behavior_distribution(
    behavior_model: FrozenBehaviorModel,
    information: BehaviorInfoSet,
) -> tuple[Mapping[str, Fraction], Mapping[str, Any]]:
    distribution = behavior_model.action_distribution(information)
    if not isinstance(distribution, BehaviorDistribution):
        raise TypeError("behavior model must return BehaviorDistribution")
    information_digest = information.digest()
    if distribution.information_digest != information_digest:
        raise ValueError("behavior distribution query digest does not match information")
    raw = distribution.probabilities
    if not isinstance(raw, Mapping):
        raise TypeError("behavior model must return an action-probability mapping")
    if set(raw) != set(information.legal_action_ids):
        missing = sorted(set(information.legal_action_ids) - set(raw))
        extra = sorted(set(raw) - set(information.legal_action_ids))
        raise ValueError(
            "behavior model action coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    probabilities: dict[str, Fraction] = {}
    for action_id in information.legal_action_ids:
        probability = raw[action_id]
        if not isinstance(probability, Fraction):
            raise TypeError("behavior model probabilities must be fractions.Fraction")
        if not 0 <= probability <= 1:
            raise ValueError("behavior model probabilities must be in [0, 1]")
        probabilities[action_id] = probability
    if sum(probabilities.values(), Fraction(0, 1)) != 1:
        raise ValueError("behavior model action probabilities must sum exactly to one")
    distribution_manifest = {
        "schema": "ofc_behavior_distribution/v1",
        "information_digest": information_digest,
        "source": distribution.source,
        "used_fallback": distribution.used_fallback,
        "probabilities": {
            action_id: f"{probability.numerator}/{probability.denominator}"
            for action_id, probability in probabilities.items()
        },
    }
    return MappingProxyType(probabilities), MappingProxyType(
        {
            "information_digest": information_digest,
            "distribution_sha256": _canonical_sha256(distribution_manifest),
            "source": distribution.source,
            "used_fallback": distribution.used_fallback,
        }
    )


def _validate_full_card_partition(
    observation: InfoSetKey,
    particle: JointParticle,
) -> None:
    """Prove that one range particle partitions the physical 54-card deck."""
    expected_undealt = EXPECTED_UNDEALT_BY_PHASE[observation.phase]
    if len(particle.undealt_cards) != expected_undealt:
        raise ValueError(
            f"phase {observation.phase!r} requires {expected_undealt} undealt cards, "
            f"got {len(particle.undealt_cards)}"
        )

    public_cards = [
        card
        for _turn, _actor, placements in observation.public_action_history
        for card, _row in placements
    ]
    private_discards = [
        card
        for recall in (particle.bb_recall, particle.btn_recall)
        for _turn, card in recall.discards_by_turn
    ]
    physical_partition = Counter(
        (*public_cards, *private_discards, *observation.current_draw, *particle.undealt_cards)
    )
    canonical_deck = Counter(ALL_CARDS)
    if physical_partition != canonical_deck:
        missing = sorted((canonical_deck - physical_partition).elements())
        extra = sorted((physical_partition - canonical_deck).elements())
        duplicates = sorted(
            card for card, count in physical_partition.items() if count > 1
        )
        raise ValueError(
            "full-card range particle does not partition the canonical 54-card deck: "
            f"missing={missing}, extra={extra}, duplicates={duplicates}"
        )


def _assignment_priority(
    assignment: tuple[str, ...], *, seed: int, observation_digest: str
) -> int:
    raw = (
        f"{int(seed)}|{observation_digest}|" + "|".join(assignment)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest(), "big")


def _select_assignments(
    pool: tuple[str, ...],
    count: int,
    *,
    max_particles: int | None,
    seed: int,
    observation_digest: str,
) -> tuple[tuple[tuple[str, ...], ...], int, bool]:
    total = math.perm(len(pool), count)
    if max_particles is None or max_particles >= total:
        return tuple(itertools.permutations(pool, count)), total, True
    if isinstance(max_particles, bool) or int(max_particles) <= 0:
        raise ValueError("max_particles must be a positive integer or None")
    limit = int(max_particles)
    # Maintain a max-heap (negative priority) containing the globally smallest
    # hash priorities.  Scanning lexicographic assignments is deterministic and
    # does not materialize the full permutation set.
    heap: list[tuple[int, tuple[str, ...]]] = []
    for assignment in itertools.permutations(pool, count):
        priority = _assignment_priority(
            assignment,
            seed=seed,
            observation_digest=observation_digest,
        )
        item = (-priority, assignment)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = tuple(
        assignment
        for _negative_priority, assignment in sorted(
            heap,
            key=lambda item: (-item[0], item[1]),
        )
    )
    return selected, total, False


def _opponent_turns(observation: InfoSetKey) -> tuple[int, ...]:
    opponent = "btn" if observation.actor == "bb" else "bb"
    return tuple(
        turn
        for turn, actor, _placements in observation.public_action_history
        if actor == opponent and turn > 0
    )


def _recall_from_discards(
    history: Sequence[PublicHistoryEntry],
    *,
    actor: Actor,
    discard_by_turn: Mapping[int, str],
) -> PrivateRecall:
    dealt: list[tuple[int, tuple[str, ...]]] = []
    discards: list[tuple[int, str]] = []
    for turn, history_actor, placements in history:
        if history_actor != actor or turn == 0:
            continue
        discard = discard_by_turn[turn]
        dealt.append((turn, tuple(card for card, _row in placements) + (discard,)))
        discards.append((turn, discard))
    return PrivateRecall(tuple(dealt), tuple(discards))


def _behavior_likelihood(
    observation: InfoSetKey,
    *,
    opponent: Actor,
    discard_by_turn: Mapping[int, str],
    behavior_model: FrozenBehaviorModel,
    behavior_cache: dict[
        str,
        tuple[BehaviorInfoSet, Mapping[str, Fraction], Mapping[str, Any]],
    ],
    epsilon: Fraction,
) -> tuple[Fraction, tuple[Mapping[str, Any], ...]]:
    boards: dict[str, dict[str, list[str]]] = {
        "bb": {row: [] for row in ROWS},
        "btn": {row: [] for row in ROWS},
    }
    history_before: list[PublicHistoryEntry] = []
    prior_dealt: list[tuple[int, tuple[str, ...]]] = []
    prior_discards: list[tuple[int, str]] = []
    likelihood = Fraction(1, 1)
    distribution_audits: list[Mapping[str, Any]] = []
    for turn, actor, placements in observation.public_action_history:
        if actor == opponent and turn > 0:
            discard = discard_by_turn[turn]
            draw = tuple(sorted((*[card for card, _row in placements], discard)))
            actor_board = _board(_rows(boards[actor]))
            legal = get_turn_actions(list(draw), actor_board)
            observed = Action(
                placements=[(card, row) for card, row in placements],
                discard=discard,
            )
            observed_id = action_key(observed)
            legal_ids = tuple(action_key(action) for action in legal)
            if observed_id not in legal_ids:
                return Fraction(0, 1), tuple(distribution_audits)
            information = BehaviorInfoSet(
                actor=opponent,
                turn=turn,
                board_bb=_rows(boards["bb"]),
                board_btn=_rows(boards["btn"]),
                public_action_history=tuple(history_before),
                own_recall_before=PrivateRecall(
                    tuple(prior_dealt), tuple(prior_discards)
                ),
                current_draw=draw,
                legal_action_ids=legal_ids,
                fantasy_state=observation.fantasy_state,
            )
            information_digest = information.digest()
            cached = behavior_cache.get(information_digest)
            if cached is None:
                probabilities, audit = _validated_behavior_distribution(
                    behavior_model, information
                )
                behavior_cache[information_digest] = (
                    information,
                    probabilities,
                    audit,
                )
            else:
                cached_information, probabilities, audit = cached
                if cached_information != information:
                    raise AssertionError("behavior information SHA256 collision")
            probability = probabilities[observed_id]
            distribution_audits.append(audit)
            likelihood *= epsilon_smoothed_likelihood(
                probability,
                epsilon=epsilon,
                action_count=len(legal),
            )
            prior_dealt.append((turn, draw))
            prior_discards.append((turn, discard))

        for card, row in placements:
            boards[actor][row].append(card)
        history_before.append((turn, actor, placements))
    return likelihood, tuple(distribution_audits)


def build_history_weighted_full_card_range(
    observation: InfoSetKey,
    behavior_model: FrozenBehaviorModel,
    *,
    epsilon: Fraction | int | str = Fraction(1, 1000),
    max_particles: int | None = 2048,
    seed: int = 0,
) -> FullCardRange:
    """Build a deterministic posterior over compatible opponent discards."""
    if not isinstance(observation, InfoSetKey):
        raise TypeError("observation must be an InfoSetKey")
    if observation.contract_version != POSITION_CONTRACT_VERSION:
        raise ValueError("full-card range requires bb_first_v1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if max_particles is not None:
        if isinstance(max_particles, bool) or not isinstance(max_particles, int):
            raise TypeError("max_particles must be a positive integer or None")
        if max_particles <= 0:
            raise ValueError("max_particles must be a positive integer or None")
    if isinstance(epsilon, bool) or isinstance(epsilon, float):
        raise TypeError("epsilon must be an exact Fraction/int/str")
    exact_epsilon = Fraction(epsilon)
    if not 0 <= exact_epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    (
        behavior_model_id,
        behavior_model_sha256,
        behavior_model_manifest,
    ) = _validated_behavior_identity(behavior_model)

    observation_digest = observation.digest()
    opponent: Actor = "btn" if observation.actor == "bb" else "bb"
    opponent_turns = _opponent_turns(observation)
    own_discards = {card for _turn, card in observation.own_recall.discards_by_turn}
    public_cards = {
        card
        for _turn, _actor, placements in observation.public_action_history
        for card, _row in placements
    }
    physically_known = public_cards | own_discards | set(observation.current_draw)
    candidate_pool = tuple(card for card in ALL_CARDS if card not in physically_known)
    assignments, total_assignments, exhaustive = _select_assignments(
        candidate_pool,
        len(opponent_turns),
        max_particles=max_particles,
        seed=seed,
        observation_digest=observation_digest,
    )

    weighted: list[tuple[JointParticle, Fraction, str]] = []
    behavior_source_counts: Counter[str] = Counter()
    behavior_query_audit: dict[str, dict[str, Any]] = {}
    behavior_cache: dict[
        str,
        tuple[BehaviorInfoSet, Mapping[str, Fraction], Mapping[str, Any]],
    ] = {}
    for assignment in assignments:
        discard_by_turn = dict(zip(opponent_turns, assignment))
        opponent_recall = _recall_from_discards(
            observation.public_action_history,
            actor=opponent,
            discard_by_turn=discard_by_turn,
        )
        likelihood, distribution_audits = _behavior_likelihood(
            observation,
            opponent=opponent,
            discard_by_turn=discard_by_turn,
            behavior_model=behavior_model,
            behavior_cache=behavior_cache,
            epsilon=exact_epsilon,
        )
        for audit in distribution_audits:
            information_digest = str(audit["information_digest"])
            source = str(audit["source"])
            used_fallback = bool(audit["used_fallback"])
            behavior_source_counts[source] += 1
            existing = behavior_query_audit.get(information_digest)
            stable = {
                "information_digest": information_digest,
                "distribution_sha256": str(audit["distribution_sha256"]),
                "source": source,
                "used_fallback": used_fallback,
            }
            if existing is None:
                behavior_query_audit[information_digest] = {
                    **stable,
                    "evaluation_count": 1,
                }
            else:
                if any(existing[key] != value for key, value in stable.items()):
                    raise ValueError(
                        "frozen behavior model returned inconsistent results for "
                        f"query {information_digest}"
                    )
                existing["evaluation_count"] += 1
        if likelihood <= 0:
            continue
        undealt = tuple(card for card in candidate_pool if card not in assignment)
        particle = JointParticle(
            bb_recall=(observation.own_recall if observation.actor == "bb" else opponent_recall),
            btn_recall=(observation.own_recall if observation.actor == "btn" else opponent_recall),
            undealt_cards=undealt,
            weight=likelihood,
        )
        _validate_full_card_partition(observation, particle)
        rebuilt = InfoSetKey.for_particle(
            particle,
            contract_version=observation.contract_version,
            actor=observation.actor,
            turn=observation.turn,
            phase=observation.phase,
            board_bb=observation.board_bb,
            board_btn=observation.board_btn,
            public_action_history=observation.public_action_history,
            current_draw=observation.current_draw,
            fantasy_state=observation.fantasy_state,
        )
        if rebuilt != observation:
            raise AssertionError("physical range particle changed the public policy key")
        commitment = _particle_commitment(particle)
        weighted.append((particle, likelihood, commitment))

    total_weight = sum((weight for _particle, weight, _commitment in weighted), Fraction(0, 1))
    if total_weight <= 0:
        raise ValueError("behavior model assigns zero posterior mass to every physical world")
    evidence_normalizer = total_weight / len(assignments)
    evidence_scope = (
        "exact_uniform_hidden_assignment_marginal_likelihood"
        if exhaustive
        else "deterministic_sample_mean_hidden_assignment_marginal_likelihood"
    )
    normalized: list[JointParticle] = []
    commitments: list[str] = []
    normalized_weights: list[Fraction] = []
    for particle, weight, commitment in sorted(weighted, key=lambda item: item[2]):
        normalized_weight = weight / total_weight
        normalized.append(
            JointParticle(
                bb_recall=particle.bb_recall,
                btn_recall=particle.btn_recall,
                undealt_cards=particle.undealt_cards,
                weight=normalized_weight,
            )
        )
        commitments.append(commitment)
        normalized_weights.append(normalized_weight)
    if len(commitments) != len(set(commitments)):
        raise AssertionError("full-card range contains duplicate physical particles")
    ess = Fraction(1, 1) / sum(
        (weight * weight for weight in normalized_weights), Fraction(0, 1)
    )
    posterior_mass = sum(normalized_weights, Fraction(0, 1))
    if posterior_mass != 1:
        raise AssertionError("normalized full-card posterior mass is not exactly one")
    content_manifest = {
        "schema": CONTENT_SCHEMA,
        "range_model": RANGE_MODEL,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "deck_size": len(ALL_CARDS),
        "physical_joker_ids": ["X1", "X2"],
        "observation_digest": observation_digest,
        "actor": observation.actor,
        "turn": observation.turn,
        "phase": observation.phase,
        "visible_joker_count": sum(
            card in ("X1", "X2") for card in observation.current_draw
        ),
        "behavior_model_id": behavior_model_id,
        "behavior_model_sha256": behavior_model_sha256,
        "epsilon": f"{exact_epsilon.numerator}/{exact_epsilon.denominator}",
        "particle_count": len(normalized),
        "expected_undealt_card_count": EXPECTED_UNDEALT_BY_PHASE[observation.phase],
        "posterior_probability_mass_exact": "1/1",
        "opponent_public_evidence_normalizer_exact": (
            f"{evidence_normalizer.numerator}/{evidence_normalizer.denominator}"
        ),
        "opponent_public_evidence_sampled_assignment_count": len(assignments),
        "opponent_public_evidence_total_assignment_count": total_assignments,
        "opponent_public_evidence_hidden_assignment_exhaustive": exhaustive,
        "opponent_public_evidence_scope": evidence_scope,
        "particles": [
            {
                "commitment": commitment,
                "weight": f"{weight.numerator}/{weight.denominator}",
            }
            for commitment, weight in zip(commitments, normalized_weights)
        ],
    }
    range_content_sha256 = _canonical_sha256(content_manifest)
    behavior_query_count = sum(behavior_source_counts.values())
    if behavior_query_count <= 0:
        raise AssertionError("full-card range evaluated no behavior-policy queries")
    if sum(
        int(audit["evaluation_count"])
        for audit in behavior_query_audit.values()
    ) != behavior_query_count:
        raise AssertionError("behavior query audit counts do not match query events")
    if len(behavior_query_audit) != len(behavior_cache):
        raise AssertionError("behavior query audit does not match memoized evaluations")
    fallback_query_count = behavior_source_counts.get("uniform_fallback", 0)
    fallback_unique_query_count = sum(
        1 for audit in behavior_query_audit.values() if audit["used_fallback"]
    )
    model_hit_rate = Fraction(
        behavior_query_count - fallback_query_count,
        behavior_query_count,
    )
    build_manifest = {
        "schema": BUILD_SCHEMA,
        "range_model": RANGE_MODEL,
        "range_content_sha256": range_content_sha256,
        "sampler": SAMPLER,
        "sampling_contract": "hash_priority_subset_then_self_normalized_posterior_v1",
        "seed": seed,
        "max_particles": max_particles,
        "candidate_pool_size": len(candidate_pool),
        "opponent_hidden_turns": list(opponent_turns),
        "candidate_assignment_count": total_assignments,
        "sampled_assignment_count": len(assignments),
        "positive_weight_particle_count": len(normalized),
        "zero_weight_assignment_count": len(assignments) - len(normalized),
        "opponent_public_evidence_normalizer_exact": (
            f"{evidence_normalizer.numerator}/{evidence_normalizer.denominator}"
        ),
        "opponent_public_evidence_scope": evidence_scope,
        "behavior_query_count": behavior_query_count,
        "behavior_unique_query_count": len(behavior_query_audit),
        "behavior_model_evaluation_count": len(behavior_cache),
        "behavior_distribution_source_counts": dict(
            sorted(behavior_source_counts.items())
        ),
        "behavior_uniform_fallback_count": fallback_query_count,
        "behavior_uniform_fallback_unique_count": fallback_unique_query_count,
        "behavior_model_hit_rate_exact": (
            f"{model_hit_rate.numerator}/{model_hit_rate.denominator}"
        ),
        "behavior_model_hit_rate": float(model_hit_rate),
        "behavior_distribution_validation_failures": 0,
        "behavior_query_audit": [
            behavior_query_audit[digest]
            for digest in sorted(behavior_query_audit)
        ],
        "exhaustive_hidden_discard_enumeration": exhaustive,
        "posterior_scope": (
            "exact_enumerated_posterior"
            if exhaustive
            else "sampled_support_self_normalized_posterior"
        ),
    }
    range_build_sha256 = _canonical_sha256(build_manifest)
    return FullCardRange(
        observation_digest=observation_digest,
        particles=tuple(normalized),
        particle_commitments=tuple(commitments),
        behavior_model_id=behavior_model_id,
        behavior_model_sha256=behavior_model_sha256,
        epsilon=exact_epsilon,
        evidence_normalizer=evidence_normalizer,
        effective_sample_size=ess,
        range_sha256=range_content_sha256,
        range_content_sha256=range_content_sha256,
        range_build_sha256=range_build_sha256,
        metadata={
            **content_manifest,
            **build_manifest,
            "content_manifest": content_manifest,
            "build_manifest": build_manifest,
            "behavior_model_manifest": _canonical_snapshot(
                dict(behavior_model_manifest)
            ),
            "range_sha256": range_content_sha256,
            "range_content_sha256": range_content_sha256,
            "range_build_sha256": range_build_sha256,
            "effective_sample_size": float(ess),
            "effective_sample_size_exact": f"{ess.numerator}/{ess.denominator}",
            "effective_sample_size_fraction": float(ess / len(normalized)),
            "opponent_hidden_turns": list(opponent_turns),
            "policy_key_contains_hidden_assignment": False,
            "history_weighted": True,
            "full_card_physical_remainder": True,
            "full_deck_partition_validated": True,
            "range_sampling_approx": not exhaustive,
            "hu_exact": False,
            "action_selected": False,
            "opponent_public_evidence_normalizer_exact": (
                f"{evidence_normalizer.numerator}/{evidence_normalizer.denominator}"
            ),
            "opponent_public_evidence_scope": evidence_scope,
        },
    )


def verify_full_card_range(
    observation: InfoSetKey,
    result: FullCardRange,
) -> Mapping[str, Any]:
    """Independently recompute the physical, posterior, and manifest invariants."""
    if not isinstance(observation, InfoSetKey):
        raise TypeError("observation must be an InfoSetKey")
    if not isinstance(result, FullCardRange):
        raise TypeError("result must be a FullCardRange")
    if result.observation_digest != observation.digest():
        raise ValueError("range observation digest does not match the supplied observation")
    if result.particle_count <= 0:
        raise ValueError("full-card range must contain at least one particle")
    if len(result.particle_commitments) != result.particle_count:
        raise ValueError("particle commitment count does not match particle count")
    if len(set(result.particle_commitments)) != result.particle_count:
        raise ValueError("full-card range commitments are not unique")
    if not isinstance(result.evidence_normalizer, Fraction):
        raise TypeError("range evidence_normalizer must be an exact Fraction")
    if result.evidence_normalizer <= 0:
        raise ValueError("range evidence_normalizer must be positive")

    weights: list[Fraction] = []
    recomputed_commitments: list[str] = []
    for particle in result.particles:
        _validate_full_card_partition(observation, particle)
        rebuilt = InfoSetKey.for_particle(
            particle,
            contract_version=observation.contract_version,
            actor=observation.actor,
            turn=observation.turn,
            phase=observation.phase,
            board_bb=observation.board_bb,
            board_btn=observation.board_btn,
            public_action_history=observation.public_action_history,
            current_draw=observation.current_draw,
            fantasy_state=observation.fantasy_state,
        )
        if rebuilt != observation:
            raise ValueError("range particle does not reconstruct the public policy key")
        if particle.weight <= 0:
            raise ValueError("normalized range particles must have positive weight")
        weights.append(particle.weight)
        recomputed_commitments.append(_particle_commitment(particle))
    if tuple(recomputed_commitments) != result.particle_commitments:
        raise ValueError("particle commitments do not match physical particle contents")
    if sum(weights, Fraction(0, 1)) != 1:
        raise ValueError("range posterior weights do not sum exactly to one")
    recomputed_ess = Fraction(1, 1) / sum(
        (weight * weight for weight in weights), Fraction(0, 1)
    )
    if recomputed_ess != result.effective_sample_size:
        raise ValueError("range effective sample size does not match particle weights")

    metadata = result.metadata
    content_manifest = metadata.get("content_manifest")
    build_manifest = metadata.get("build_manifest")
    model_manifest = metadata.get("behavior_model_manifest")
    if not isinstance(content_manifest, Mapping) or not isinstance(
        build_manifest, Mapping
    ):
        raise ValueError("range metadata must contain content/build manifests")
    if not isinstance(model_manifest, Mapping):
        raise ValueError("range metadata must contain a behavior model manifest")
    if _canonical_sha256(dict(content_manifest)) != result.range_content_sha256:
        raise ValueError("range content manifest hash mismatch")
    if _canonical_sha256(dict(build_manifest)) != result.range_build_sha256:
        raise ValueError("range build manifest hash mismatch")
    if result.range_sha256 != result.range_content_sha256:
        raise ValueError("range_sha256 must alias the content hash")
    if build_manifest.get("range_content_sha256") != result.range_content_sha256:
        raise ValueError("build manifest is not bound to the range content hash")
    if _canonical_sha256(dict(model_manifest)) != result.behavior_model_sha256:
        raise ValueError("behavior model manifest hash mismatch")
    if model_manifest.get("model_id") != result.behavior_model_id:
        raise ValueError("behavior model manifest ID mismatch")

    expected_particles = [
        {
            "commitment": commitment,
            "weight": f"{weight.numerator}/{weight.denominator}",
        }
        for commitment, weight in zip(result.particle_commitments, weights)
    ]
    semantic_content = {
        "schema": CONTENT_SCHEMA,
        "range_model": RANGE_MODEL,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "deck_size": len(ALL_CARDS),
        "physical_joker_ids": ["X1", "X2"],
        "observation_digest": observation.digest(),
        "actor": observation.actor,
        "turn": observation.turn,
        "phase": observation.phase,
        "visible_joker_count": sum(
            card in ("X1", "X2") for card in observation.current_draw
        ),
        "behavior_model_id": result.behavior_model_id,
        "behavior_model_sha256": result.behavior_model_sha256,
        "epsilon": f"{result.epsilon.numerator}/{result.epsilon.denominator}",
        "particle_count": result.particle_count,
        "expected_undealt_card_count": EXPECTED_UNDEALT_BY_PHASE[observation.phase],
        "posterior_probability_mass_exact": "1/1",
        "opponent_public_evidence_normalizer_exact": (
            f"{result.evidence_normalizer.numerator}/"
            f"{result.evidence_normalizer.denominator}"
        ),
        "opponent_public_evidence_sampled_assignment_count": build_manifest.get(
            "sampled_assignment_count"
        ),
        "opponent_public_evidence_total_assignment_count": build_manifest.get(
            "candidate_assignment_count"
        ),
        "opponent_public_evidence_hidden_assignment_exhaustive": build_manifest.get(
            "exhaustive_hidden_discard_enumeration"
        ),
        "opponent_public_evidence_scope": build_manifest.get(
            "opponent_public_evidence_scope"
        ),
        "particles": expected_particles,
    }
    if dict(content_manifest) != semantic_content:
        raise ValueError("range content manifest does not match range semantics")
    expected_evidence = (
        f"{result.evidence_normalizer.numerator}/"
        f"{result.evidence_normalizer.denominator}"
    )
    if build_manifest.get(
        "opponent_public_evidence_normalizer_exact"
    ) != expected_evidence:
        raise ValueError("range build manifest evidence normalizer mismatch")
    if metadata.get(
        "opponent_public_evidence_normalizer_exact"
    ) != expected_evidence:
        raise ValueError("range metadata evidence normalizer mismatch")
    sampled_assignments = build_manifest.get("sampled_assignment_count")
    total_assignments = build_manifest.get("candidate_assignment_count")
    exhaustive = build_manifest.get("exhaustive_hidden_discard_enumeration")
    if (
        isinstance(sampled_assignments, bool)
        or not isinstance(sampled_assignments, int)
        or sampled_assignments <= 0
        or isinstance(total_assignments, bool)
        or not isinstance(total_assignments, int)
        or total_assignments < sampled_assignments
        or not isinstance(exhaustive, bool)
    ):
        raise ValueError("range evidence assignment-count provenance is invalid")
    expected_evidence_scope = (
        "exact_uniform_hidden_assignment_marginal_likelihood"
        if exhaustive
        else "deterministic_sample_mean_hidden_assignment_marginal_likelihood"
    )
    if exhaustive is not (sampled_assignments == total_assignments):
        raise ValueError("range evidence exhaustive/count provenance is inconsistent")
    if build_manifest.get("opponent_public_evidence_scope") != expected_evidence_scope:
        raise ValueError("range build manifest evidence scope mismatch")
    if metadata.get("opponent_public_evidence_scope") != expected_evidence_scope:
        raise ValueError("range metadata evidence scope mismatch")

    raw_audit = build_manifest.get("behavior_query_audit")
    if not isinstance(raw_audit, list) or not raw_audit:
        raise ValueError("build manifest requires raw behavior query audit entries")
    seen_digests: set[str] = set()
    source_counts: Counter[str] = Counter()
    fallback_events = 0
    fallback_unique = 0
    total_events = 0
    for entry in raw_audit:
        if not isinstance(entry, Mapping):
            raise ValueError("behavior query audit entries must be mappings")
        digest = entry.get("information_digest")
        distribution_sha256 = entry.get("distribution_sha256")
        source = entry.get("source")
        used_fallback = entry.get("used_fallback")
        evaluation_count = entry.get("evaluation_count")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError("behavior query audit contains an invalid digest")
        if digest in seen_digests:
            raise ValueError("behavior query audit contains duplicate digests")
        seen_digests.add(digest)
        if not isinstance(distribution_sha256, str) or not _SHA256_RE.fullmatch(
            distribution_sha256
        ):
            raise ValueError("behavior query audit contains an invalid distribution hash")
        if source not in BEHAVIOR_DISTRIBUTION_SOURCES:
            raise ValueError("behavior query audit contains an invalid source")
        if not isinstance(used_fallback, bool) or used_fallback != (
            source == "uniform_fallback"
        ):
            raise ValueError("behavior query audit fallback provenance is inconsistent")
        if (
            isinstance(evaluation_count, bool)
            or not isinstance(evaluation_count, int)
            or evaluation_count <= 0
        ):
            raise ValueError("behavior query audit evaluation_count must be positive")
        source_counts[str(source)] += evaluation_count
        total_events += evaluation_count
        if used_fallback:
            fallback_unique += 1
            fallback_events += evaluation_count

    if total_events != build_manifest.get("behavior_query_count"):
        raise ValueError("behavior query event count does not match raw audit")
    if len(seen_digests) != build_manifest.get("behavior_unique_query_count"):
        raise ValueError("behavior unique query count does not match raw audit")
    if len(seen_digests) != build_manifest.get("behavior_model_evaluation_count"):
        raise ValueError("behavior model evaluation count does not match memoized audit")
    if dict(sorted(source_counts.items())) != build_manifest.get(
        "behavior_distribution_source_counts"
    ):
        raise ValueError("behavior source counts do not match raw audit")
    if fallback_events != build_manifest.get("behavior_uniform_fallback_count"):
        raise ValueError("behavior fallback event count does not match raw audit")
    if fallback_unique != build_manifest.get(
        "behavior_uniform_fallback_unique_count"
    ):
        raise ValueError("behavior fallback unique count does not match raw audit")
    hit_rate = Fraction(total_events - fallback_events, total_events)
    if build_manifest.get("behavior_model_hit_rate_exact") != (
        f"{hit_rate.numerator}/{hit_rate.denominator}"
    ):
        raise ValueError("behavior model hit rate does not match raw audit")
    if build_manifest.get("behavior_distribution_validation_failures") != 0:
        raise ValueError("behavior distribution validation failures must be zero")

    return MappingProxyType(
        {
            "verified": True,
            "particle_count": result.particle_count,
            "effective_sample_size_exact": (
                f"{recomputed_ess.numerator}/{recomputed_ess.denominator}"
            ),
            "behavior_query_count": total_events,
            "behavior_unique_query_count": len(seen_digests),
            "behavior_uniform_fallback_count": fallback_events,
            "range_content_sha256": result.range_content_sha256,
            "range_build_sha256": result.range_build_sha256,
            "opponent_public_evidence_normalizer_exact": expected_evidence,
        }
    )
