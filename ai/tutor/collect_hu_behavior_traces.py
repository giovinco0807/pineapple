"""Deterministic full-private HU T1/T2 behavior trace collection.

The public decision artifact contains only the acting player's
``BehaviorInfoSet`` and a commitment to the hidden root.  The commitment
preimage (complete physical deck, seat assignment, draws, actions and private
discards) is written to a separate restricted ``roots.jsonl`` sidecar.  This
separation is intentional: policy queries never receive the root preimage.

Collection is deterministic by root identity and decision identity.  Deck
generation uses a SHA-256 counter RNG and each categorical action is sampled
from an independent ``(seed_namespace, root_id, turn, actor)`` domain.  Thus a
future batched evaluator can preserve exactly the same samples.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import ai.engine.action_space as action_space_module
import ai.tutor.exact_late as exact_late_module
from ai.engine.action_space import Action, get_initial_actions, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.behavior_calibration_contract import (
    build_behavior_decision_log,
    build_behavior_decision_manifest,
    canonical_json,
    canonical_sha256,
    commit_hidden_root_trace,
    verify_behavior_decision_dataset,
    verify_behavior_decision_log,
)
from ai.tutor.exact_late import action_key, apply_action
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    BehaviorInfoSet,
    FrozenBehaviorModel,
)
from ai.tutor.t3_hu_public_cfr import CardRows, PrivateRecall, PublicHistoryEntry


COLLECTION_SCHEMA = "ofc_behavior_trace_collection/v1"
ROOT_TRACE_SCHEMA = "ofc_behavior_hidden_root_trace/v1"
ROOT_ID_SCHEMA = "ofc_behavior_root_identity/v1"
SAMPLING_SCHEMA = "ofc_behavior_exact_categorical/v1"
T0_BASELINE_SCHEMA = "ofc_t0_canonical_baseline/v1"
NATURAL_UNIFORM_SHUFFLE = "natural_uniform_shuffle"
TARGETED_JOKER_CHALLENGE = "targeted_joker_challenge"
DETERMINISTIC_JOKER_CYCLE = "deterministic_role_joker_cycle_v1"
CALLER_SUPPLIED_DECKS = "caller_supplied_full_decks_v1"
ROOT_SAMPLING_MODES = frozenset(
    {NATURAL_UNIFORM_SHUFFLE, TARGETED_JOKER_CHALLENGE}
)
ROWS = ("top", "middle", "bottom")
ACTORS = ("bb", "btn")
LOGGED_DECISIONS = ((1, "bb"), (1, "btn"), (2, "bb"), (2, "btn"))
DEAL_SPECS = (
    (0, "bb", 5),
    (0, "btn", 5),
    (1, "bb", 3),
    (1, "btn", 3),
    (2, "bb", 3),
    (2, "btn", 3),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHALLENGE_ID_RE = re.compile(
    r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*$"
)
_VALID_CARDS = frozenset(ALL_CARDS)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def t0_baseline_manifest() -> dict[str, Any]:
    """Content-address the exact deterministic, information-safe T0 policy."""
    return {
        "schema": T0_BASELINE_SCHEMA,
        "baseline_id": "canonical_lexicographic_initial_action_v1",
        "algorithm": "min(action_key(action)) over canonical legal initial actions",
        "actor_inputs": ["own_five_card_draw", "own_empty_board"],
        "opponent_private_input": False,
        "opponent_public_input": False,
        "hidden_deck_input": False,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "action_space_source_sha256": _sha256_file(action_space_module.__file__),
        "action_key_source_sha256": _sha256_file(exact_late_module.__file__),
    }


def t0_baseline_sha256() -> str:
    return canonical_sha256(t0_baseline_manifest())


@dataclass(frozen=True)
class BehaviorTraceCollectionConfig:
    """Immutable collection population contract.

    Natural IID roots and targeted Joker challenge roots cannot share a config
    or manifest.  Caller-supplied decks are accepted only in an explicitly
    named challenge population.
    """

    seed_namespace: str
    root_sampling_mode: str = NATURAL_UNIFORM_SHUFFLE
    challenge_id: str | None = None
    challenge_deck_source: str | None = None
    allow_uniform_fallback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace:
            raise ValueError("seed_namespace must be a non-empty string")
        if self.root_sampling_mode not in ROOT_SAMPLING_MODES:
            raise ValueError("unsupported root_sampling_mode")
        if type(self.allow_uniform_fallback) is not bool:
            raise TypeError("allow_uniform_fallback must be boolean")
        if self.root_sampling_mode == NATURAL_UNIFORM_SHUFFLE:
            if self.challenge_id is not None or self.challenge_deck_source is not None:
                raise ValueError("natural collection cannot carry challenge metadata")
        else:
            if not isinstance(self.challenge_id, str) or not self.challenge_id:
                raise ValueError("targeted challenge requires a non-empty challenge_id")
            if not _CHALLENGE_ID_RE.fullmatch(self.challenge_id):
                raise ValueError(
                    "challenge_id must be normalized lowercase path segments "
                    "without leading/trailing/double slashes"
                )
            if self.challenge_deck_source not in {
                DETERMINISTIC_JOKER_CYCLE,
                CALLER_SUPPLIED_DECKS,
            }:
                raise ValueError("targeted challenge requires an explicit deck source")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "seed_namespace": self.seed_namespace,
            "root_sampling_mode": self.root_sampling_mode,
            "challenge_id": self.challenge_id,
            "challenge_deck_source": self.challenge_deck_source,
            "allow_uniform_fallback": self.allow_uniform_fallback,
        }

    @classmethod
    def from_canonical_dict(cls, value: Mapping[str, Any]) -> "BehaviorTraceCollectionConfig":
        expected = {
            "seed_namespace",
            "root_sampling_mode",
            "challenge_id",
            "challenge_deck_source",
            "allow_uniform_fallback",
        }
        if set(value) != expected:
            raise ValueError("collection config keys mismatch")
        result = cls(
            seed_namespace=value["seed_namespace"],
            root_sampling_mode=value["root_sampling_mode"],
            challenge_id=value["challenge_id"],
            challenge_deck_source=value["challenge_deck_source"],
            allow_uniform_fallback=value["allow_uniform_fallback"],
        )
        if result.to_canonical_dict() != dict(value):
            raise ValueError("collection config is not canonical")
        return result


def derive_root_id(config: BehaviorTraceCollectionConfig, root_index: int) -> str:
    if isinstance(root_index, bool) or not isinstance(root_index, int) or root_index < 0:
        raise ValueError("root_index must be a non-negative integer")
    digest = canonical_sha256(
        {
            "schema": ROOT_ID_SCHEMA,
            "seed_namespace": config.seed_namespace,
            "root_sampling_mode": config.root_sampling_mode,
            "challenge_id": config.challenge_id,
            "challenge_deck_source": config.challenge_deck_source,
            "root_index": root_index,
        }
    )
    if config.root_sampling_mode == TARGETED_JOKER_CHALLENGE:
        # Calibration gates can reject challenge roots before inspecting any
        # labels while the digest still binds the full preregistered identity.
        return f"{config.challenge_id}/{digest}"
    return digest


class _Sha256CounterRng:
    """Small specified hash RNG supporting unbiased arbitrary-size integers."""

    def __init__(self, domain: Mapping[str, Any]) -> None:
        self._seed = hashlib.sha256(canonical_json(dict(domain)).encode("utf-8")).digest()
        self._request = 0

    def randbelow(self, upper: int) -> int:
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
            raise ValueError("randbelow upper bound must be a positive integer")
        if upper == 1:
            self._request += 1
            return 0
        bits = upper.bit_length()
        byte_count = (bits + 7) // 8
        request = self._request
        self._request += 1
        attempt = 0
        while True:
            output = bytearray()
            block = 0
            while len(output) < byte_count:
                output.extend(
                    hashlib.sha256(
                        self._seed
                        + request.to_bytes(8, "big")
                        + attempt.to_bytes(8, "big")
                        + block.to_bytes(4, "big")
                    ).digest()
                )
                block += 1
            value = int.from_bytes(output[:byte_count], "big") & ((1 << bits) - 1)
            if value < upper:
                return value
            attempt += 1


def _deterministic_shuffle(config: BehaviorTraceCollectionConfig, root_id: str) -> tuple[str, ...]:
    cards = list(ALL_CARDS)
    rng = _Sha256CounterRng(
        {
            "schema": "ofc_behavior_deck_shuffle/v1",
            "seed_namespace": config.seed_namespace,
            "root_id": root_id,
        }
    )
    for index in range(len(cards) - 1, 0, -1):
        other = rng.randbelow(index + 1)
        cards[index], cards[other] = cards[other], cards[index]
    return tuple(cards)


def targeted_joker_cell(root_index: int) -> tuple[int, str, int]:
    """Return the preregistered T1/T2 x role x Joker(0/1/2) challenge cell."""
    if isinstance(root_index, bool) or not isinstance(root_index, int) or root_index < 0:
        raise ValueError("root_index must be a non-negative integer")
    turn, actor = LOGGED_DECISIONS[root_index % len(LOGGED_DECISIONS)]
    joker_count = (root_index // len(LOGGED_DECISIONS)) % 3
    return turn, actor, joker_count


def _targeted_cycle_deck(deck: Sequence[str], root_index: int) -> tuple[str, ...]:
    """Force one preregistered role cell to contain exactly 0, 1 or 2 Jokers."""
    role_index = root_index % len(LOGGED_DECISIONS)
    _turn, _actor, joker_count = targeted_joker_cell(root_index)
    starts = (10, 13, 16, 19)
    target_start = starts[role_index]
    constraints: dict[int, str] = {}
    if joker_count == 2:
        constraints[target_start] = "X1"
        constraints[target_start + 1] = "X2"
    elif joker_count == 1:
        constraints[target_start] = "X1"
        # Keep X2 out of every logged draw so the targeted count is exact.
        constraints[53] = "X2"
    else:
        # Keep both physical Jokers out of all logged draws for the zero cell.
        constraints[52] = "X1"
        constraints[53] = "X2"
    remainder = iter(card for card in deck if card not in {"X1", "X2"})
    out: list[str] = []
    for index in range(54):
        out.append(constraints[index] if index in constraints else next(remainder))
    return tuple(out)


def _validated_deck(deck: Sequence[str], *, label: str) -> tuple[str, ...]:
    cards = tuple(deck)
    if len(cards) != 54 or len(set(cards)) != 54 or set(cards) != _VALID_CARDS:
        raise ValueError(f"{label} must be one exact permutation of the 54-card deck")
    return cards


@dataclass(frozen=True)
class HURootDealPlan:
    root_index: int
    root_id: str
    btn_seat: int
    deck: tuple[str, ...]
    deals: tuple[tuple[int, str, int, tuple[str, ...]], ...]
    challenge_target: tuple[int, str, int] | None = None

    def draw(self, turn: int, actor: str) -> tuple[str, ...]:
        for item_turn, item_actor, _seat, cards in self.deals:
            if (item_turn, item_actor) == (turn, actor):
                return cards
        raise KeyError((turn, actor))


def build_root_deal_plan(
    config: BehaviorTraceCollectionConfig,
    root_index: int,
    *,
    forced_deck: Sequence[str] | None = None,
) -> HURootDealPlan:
    """Generate only the physical root; no policy is evaluated here."""
    root_id = derive_root_id(config, root_index)
    if config.root_sampling_mode == NATURAL_UNIFORM_SHUFFLE:
        if forced_deck is not None:
            raise ValueError("natural_uniform_shuffle rejects forced decks")
        deck = _deterministic_shuffle(config, root_id)
    elif config.challenge_deck_source == DETERMINISTIC_JOKER_CYCLE:
        if forced_deck is not None:
            raise ValueError("deterministic Joker challenge rejects forced decks")
        deck = _targeted_cycle_deck(
            _deterministic_shuffle(config, root_id), root_index
        )
    else:
        if forced_deck is None:
            raise ValueError("caller-supplied challenge requires one deck per root")
        deck = tuple(forced_deck)
    deck = _validated_deck(deck, label=f"root {root_index} deck")
    btn_seat = root_index % 2
    role_seats = {"btn": btn_seat, "bb": 1 - btn_seat}
    offset = 0
    deals: list[tuple[int, str, int, tuple[str, ...]]] = []
    for turn, actor, count in DEAL_SPECS:
        draw = tuple(deck[offset : offset + count])
        deals.append((turn, actor, role_seats[actor], draw))
        offset += count
    return HURootDealPlan(
        root_index=root_index,
        root_id=root_id,
        btn_seat=btn_seat,
        deck=deck,
        deals=tuple(deals),
        challenge_target=(
            targeted_joker_cell(root_index)
            if config.challenge_deck_source == DETERMINISTIC_JOKER_CYCLE
            else None
        ),
    )


def sampling_contract(config: BehaviorTraceCollectionConfig) -> dict[str, Any]:
    return {
        "schema": SAMPLING_SCHEMA,
        "name": "decision_local_sha256_exact_categorical_v1",
        "root_sampling_mode": config.root_sampling_mode,
        "challenge_id": config.challenge_id,
        "challenge_deck_source": config.challenge_deck_source,
        "deck_rng": "sha256_counter_fisher_yates_unbiased_randbelow_v1",
        "decision_seed_domain": [
            "seed_namespace",
            "root_id",
            "turn",
            "actor",
        ],
        "decision_rng": "sha256_counter_unbiased_randbelow_v1",
        "action_order": "sorted_canonical_action_key",
        "probability_arithmetic": "fractions.Fraction_lcm_integer_ticket",
        "allow_uniform_fallback": config.allow_uniform_fallback,
        "t0_baseline_sha256": t0_baseline_sha256(),
        "position_contract_version": POSITION_CONTRACT_VERSION,
    }


def _model_identity(model: FrozenBehaviorModel) -> tuple[str, str]:
    model_id = model.model_id
    model_sha = model.model_sha256
    manifest = model.model_manifest
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("behavior model_id must be non-empty")
    if not isinstance(model_sha, str) or not _SHA256_RE.fullmatch(model_sha):
        raise ValueError("behavior model_sha256 must be a lowercase SHA-256")
    if not isinstance(manifest, Mapping):
        raise TypeError("behavior model_manifest must be a mapping")
    snapshot = json.loads(
        json.dumps(
            dict(manifest),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    computed = hashlib.sha256(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if snapshot.get("model_id") != model_id or computed != model_sha:
        raise ValueError("behavior model manifest identity/hash mismatch")
    if snapshot.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        raise ValueError("behavior model must bind bb_first_v1")
    return model_id, model_sha


@dataclass(frozen=True)
class PolicyEvaluation:
    information_digest: str
    probabilities: tuple[tuple[str, Fraction], ...]
    source: str
    used_fallback: bool

    def probability(self, action_id: str) -> Fraction:
        return dict(self.probabilities)[action_id]


def evaluate_behavior_query(
    behavior_model: FrozenBehaviorModel,
    information: BehaviorInfoSet,
    *,
    allow_uniform_fallback: bool = False,
) -> PolicyEvaluation:
    """Evaluate a policy query without advancing any root state."""
    result = behavior_model.action_distribution(information)
    if not isinstance(result, BehaviorDistribution):
        raise TypeError("behavior model must return BehaviorDistribution")
    if result.information_digest != information.digest():
        raise ValueError("behavior distribution information digest mismatch")
    if result.used_fallback and not allow_uniform_fallback:
        raise ValueError("uniform fallback is disabled for this collection")
    if set(result.probabilities) != set(information.legal_action_ids):
        raise ValueError("behavior distribution legal-action coverage mismatch")
    probabilities: list[tuple[str, Fraction]] = []
    for action_id in information.legal_action_ids:
        probability = result.probabilities[action_id]
        if not isinstance(probability, Fraction):
            raise TypeError("behavior distribution probabilities must be exact Fractions")
        if not Fraction(0) <= probability <= Fraction(1):
            raise ValueError("behavior probability is outside [0,1]")
        probabilities.append((action_id, probability))
    if sum((item[1] for item in probabilities), Fraction(0)) != 1:
        raise ValueError("behavior probabilities must sum exactly to one")
    return PolicyEvaluation(
        information_digest=information.digest(),
        probabilities=tuple(probabilities),
        source=result.source,
        used_fallback=result.used_fallback,
    )


def sample_policy_evaluation(
    evaluation: PolicyEvaluation,
    *,
    seed_namespace: str,
    root_id: str,
    turn: int,
    actor: str,
) -> tuple[str, Fraction]:
    """Sample one exact categorical action from a decision-local hash domain."""
    denominators = [probability.denominator for _key, probability in evaluation.probabilities]
    denominator = math.lcm(*denominators)
    integer_weights = [
        probability.numerator * (denominator // probability.denominator)
        for _key, probability in evaluation.probabilities
    ]
    if sum(integer_weights) != denominator:
        raise AssertionError("exact categorical integer weights do not sum")
    rng = _Sha256CounterRng(
        {
            "schema": "ofc_behavior_decision_seed/v1",
            "seed_namespace": seed_namespace,
            "root_id": root_id,
            "turn": turn,
            "actor": actor,
        }
    )
    ticket = rng.randbelow(denominator)
    cumulative = 0
    for (action_id, probability), weight in zip(
        evaluation.probabilities, integer_weights
    ):
        cumulative += weight
        if ticket < cumulative:
            if probability <= 0:
                raise AssertionError("zero-probability action was sampled")
            return action_id, probability
    raise AssertionError("categorical ticket was not assigned")


def _rows(board: Board) -> CardRows:
    return (
        tuple(sorted(board.top)),
        tuple(sorted(board.middle)),
        tuple(sorted(board.bottom)),
    )


def _board(rows: CardRows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _placements(action: Action) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            ((str(card), str(row)) for card, row in action.placements),
            key=lambda item: (item[1], item[0]),
        )
    )


def _action_payload(action: Action) -> dict[str, Any]:
    return {
        "placements": [[card, row] for card, row in _placements(action)],
        "discard": action.discard,
    }


def canonical_t0_action(draw: Sequence[str]) -> Action:
    cards = tuple(sorted(draw))
    if len(cards) != 5 or len(set(cards)) != 5 or not set(cards) <= _VALID_CARDS:
        raise ValueError("T0 baseline requires five unique physical cards")
    legal = get_initial_actions(list(cards), Board())
    if not legal:
        raise AssertionError("canonical action engine produced no T0 action")
    return min(legal, key=action_key)


@dataclass(frozen=True)
class HUTraceState:
    board_bb: CardRows
    board_btn: CardRows
    public_action_history: tuple[PublicHistoryEntry, ...]
    bb_recall: PrivateRecall
    btn_recall: PrivateRecall


def initialize_t0_state(bb_draw: Sequence[str], btn_draw: Sequence[str]) -> tuple[HUTraceState, Action, Action]:
    """Apply the information-safe T0 baseline BB first, then BTN."""
    bb_action = canonical_t0_action(bb_draw)
    btn_action = canonical_t0_action(btn_draw)
    board_bb = _rows(apply_action(Board(), bb_action))
    board_btn = _rows(apply_action(Board(), btn_action))
    history: tuple[PublicHistoryEntry, ...] = (
        (0, "bb", _placements(bb_action)),
        (0, "btn", _placements(btn_action)),
    )
    return (
        HUTraceState(
            board_bb=board_bb,
            board_btn=board_btn,
            public_action_history=history,
            bb_recall=PrivateRecall(),
            btn_recall=PrivateRecall(),
        ),
        bb_action,
        btn_action,
    )


def build_behavior_query(
    state: HUTraceState,
    *,
    turn: int,
    actor: str,
    current_draw: Sequence[str],
) -> BehaviorInfoSet:
    """Build only the pre-action information available to the acting role."""
    if (turn, actor) not in LOGGED_DECISIONS:
        raise ValueError("collector supports only T1/T2 bb/btn decisions")
    draw = tuple(sorted(current_draw))
    if len(draw) != 3 or len(set(draw)) != 3 or not set(draw) <= _VALID_CARDS:
        raise ValueError("regular behavior query requires three unique physical cards")
    actor_rows = state.board_bb if actor == "bb" else state.board_btn
    legal_ids = tuple(
        sorted(action_key(action) for action in get_turn_actions(list(draw), _board(actor_rows)))
    )
    return BehaviorInfoSet(
        actor=actor,  # type: ignore[arg-type]
        turn=turn,
        board_bb=state.board_bb,
        board_btn=state.board_btn,
        public_action_history=state.public_action_history,
        own_recall_before=state.bb_recall if actor == "bb" else state.btn_recall,
        current_draw=draw,
        legal_action_ids=legal_ids,
        fantasy_state=None,
    )


def apply_behavior_decision(
    state: HUTraceState,
    information: BehaviorInfoSet,
    observed_action_key: str,
) -> tuple[HUTraceState, Action]:
    """Advance physical/public state after a separately evaluated decision."""
    regenerated = build_behavior_query(
        state,
        turn=information.turn,
        actor=information.actor,
        current_draw=information.current_draw,
    )
    if regenerated != information:
        raise ValueError("behavior information does not match current progression state")
    actor_rows = state.board_bb if information.actor == "bb" else state.board_btn
    legal = {
        action_key(action): action
        for action in get_turn_actions(list(information.current_draw), _board(actor_rows))
    }
    action = legal.get(observed_action_key)
    if action is None:
        raise ValueError("sampled behavior action is not legal")
    updated_rows = _rows(apply_action(_board(actor_rows), action))
    history = state.public_action_history + (
        (information.turn, information.actor, _placements(action)),
    )
    recall = information.own_recall_before
    updated_recall = PrivateRecall(
        dealt_by_turn=recall.dealt_by_turn
        + ((information.turn, information.current_draw),),
        discards_by_turn=recall.discards_by_turn
        + ((information.turn, str(action.discard)),),
    )
    return (
        HUTraceState(
            board_bb=updated_rows if information.actor == "bb" else state.board_bb,
            board_btn=updated_rows if information.actor == "btn" else state.board_btn,
            public_action_history=history,
            bb_recall=updated_recall if information.actor == "bb" else state.bb_recall,
            btn_recall=updated_recall if information.actor == "btn" else state.btn_recall,
        ),
        action,
    )


@dataclass(frozen=True)
class _PendingDecision:
    information: BehaviorInfoSet
    action: Action
    probability: Fraction
    source: str


@dataclass(frozen=True)
class CollectedBehaviorTraces:
    records: tuple[dict[str, Any], ...]
    hidden_roots: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    config: BehaviorTraceCollectionConfig
    root_index_start: int
    root_index_stop_exclusive: int
    elapsed_runtime_ns: int

    @property
    def root_count(self) -> int:
        return self.root_index_stop_exclusive - self.root_index_start


def _history_payload(history: Sequence[PublicHistoryEntry]) -> list[dict[str, Any]]:
    return [
        {
            "turn": turn,
            "actor": actor,
            "placements": [[card, row] for card, row in placements],
        }
        for turn, actor, placements in history
    ]


def _board_payload(rows: CardRows) -> dict[str, list[str]]:
    return {row: list(cards) for row, cards in zip(ROWS, rows)}


def _action_event(
    *,
    turn: int,
    actor: str,
    seat: int,
    action: Action,
    information_digest: str | None,
    probability: Fraction,
    policy_source: str,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "actor": actor,
        "seat": seat,
        "information_digest": information_digest,
        "action_key": action_key(action),
        **_action_payload(action),
        "source_action_probability": f"{probability.numerator}/{probability.denominator}",
        "policy_source": policy_source,
    }


def _collect_one_root(
    config: BehaviorTraceCollectionConfig,
    behavior_model: FrozenBehaviorModel,
    plan: HURootDealPlan,
    *,
    model_id: str,
    model_sha256: str,
    sampling: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    state, bb_t0_action, btn_t0_action = initialize_t0_state(
        plan.draw(0, "bb"), plan.draw(0, "btn")
    )
    seats = {"btn": plan.btn_seat, "bb": 1 - plan.btn_seat}
    events = [
        _action_event(
            turn=0,
            actor="bb",
            seat=seats["bb"],
            action=bb_t0_action,
            information_digest=None,
            probability=Fraction(1),
            policy_source="content_addressed_t0_baseline",
        ),
        _action_event(
            turn=0,
            actor="btn",
            seat=seats["btn"],
            action=btn_t0_action,
            information_digest=None,
            probability=Fraction(1),
            policy_source="content_addressed_t0_baseline",
        ),
    ]
    pending: list[_PendingDecision] = []
    for turn, actor in LOGGED_DECISIONS:
        information = build_behavior_query(
            state,
            turn=turn,
            actor=actor,
            current_draw=plan.draw(turn, actor),
        )
        evaluation = evaluate_behavior_query(
            behavior_model,
            information,
            allow_uniform_fallback=config.allow_uniform_fallback,
        )
        selected_id, probability = sample_policy_evaluation(
            evaluation,
            seed_namespace=config.seed_namespace,
            root_id=plan.root_id,
            turn=turn,
            actor=actor,
        )
        state, action = apply_behavior_decision(state, information, selected_id)
        pending.append(
            _PendingDecision(
                information=information,
                action=action,
                probability=probability,
                source=evaluation.source,
            )
        )
        events.append(
            _action_event(
                turn=turn,
                actor=actor,
                seat=seats[actor],
                action=action,
                information_digest=information.digest(),
                probability=probability,
                policy_source=evaluation.source,
            )
        )

    hidden_trace: dict[str, Any] = {
        "schema": ROOT_TRACE_SCHEMA,
        "root_id": plan.root_id,
        "root_index": plan.root_index,
        "seed_namespace": config.seed_namespace,
        "root_sampling_mode": config.root_sampling_mode,
        "challenge_id": config.challenge_id,
        "challenge_deck_source": config.challenge_deck_source,
        "challenge_target": (
            {
                "turn": plan.challenge_target[0],
                "actor": plan.challenge_target[1],
                "visible_joker_count": plan.challenge_target[2],
            }
            if plan.challenge_target is not None
            else None
        ),
        "physical_deck_order": list(plan.deck),
        "seat_assignment": {"bb": seats["bb"], "btn": seats["btn"]},
        "deal_sequence": [
            {
                "turn": turn,
                "actor": actor,
                "seat": seat,
                "cards": list(cards),
            }
            for turn, actor, seat, cards in plan.deals
        ],
        "t0_baseline_sha256": t0_baseline_sha256(),
        "behavior_target_id": model_id,
        "policy_sha256": model_sha256,
        "realized_actions": events,
        "final_public_state": {
            "board_bb": _board_payload(state.board_bb),
            "board_btn": _board_payload(state.board_btn),
            "public_action_history": _history_payload(state.public_action_history),
        },
    }
    root_commitment = commit_hidden_root_trace(hidden_trace)
    records = tuple(
        build_behavior_decision_log(
            root_id=plan.root_id,
            root_commitment=root_commitment,
            seed_namespace=config.seed_namespace,
            behavior_target_id=model_id,
            policy_sha256=model_sha256,
            sampling_contract=sampling,
            information=item.information,
            observed_action_key=action_key(item.action),
            source_action_probability=item.probability,
        )
        for item in pending
    )
    return records, hidden_trace


def _root_commitment(trace: Mapping[str, Any]) -> str:
    return commit_hidden_root_trace(dict(trace))


def _manifest_stable_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"elapsed_runtime_ns", "collection_content_sha256", "manifest_sha256"}
    }


def _build_collection_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    hidden_roots: Sequence[Mapping[str, Any]],
    config: BehaviorTraceCollectionConfig,
    root_index_start: int,
    root_index_stop_exclusive: int,
    model_id: str,
    model_sha256: str,
    elapsed_runtime_ns: int,
) -> dict[str, Any]:
    behavior_manifest = build_behavior_decision_manifest(records)
    commitments = [_root_commitment(trace) for trace in hidden_roots]
    record_hashes = [str(record["record_sha256"]) for record in records]
    root_ids = [str(trace["root_id"]) for trace in hidden_roots]
    manifest: dict[str, Any] = {
        "schema": COLLECTION_SCHEMA,
        "promotion_eligible": False,
        "collection_config": config.to_canonical_dict(),
        "root_index_start": root_index_start,
        "root_index_stop_exclusive": root_index_stop_exclusive,
        "root_count": len(hidden_roots),
        "decision_count": len(records),
        "policy_query_count": len(records),
        "model_evaluation_count": len(records),
        "elapsed_runtime_ns": elapsed_runtime_ns,
        "policy": {"model_id": model_id, "model_sha256": model_sha256},
        "t0_baseline": t0_baseline_manifest(),
        "sampling_contract": sampling_contract(config),
        "behavior_decision_manifest": behavior_manifest,
        "record_order_sha256": canonical_sha256(record_hashes),
        "root_id_order_sha256": canonical_sha256(root_ids),
        "hidden_root_artifact": {
            "schema": ROOT_TRACE_SCHEMA,
            "file_contract": "separate_canonical_roots_jsonl",
            "access_class": "restricted_hidden_root_preimages",
            "preimage_in_decision_records": False,
            "root_count": len(hidden_roots),
            "root_commitment_order_sha256": canonical_sha256(commitments),
        },
        "collector_contract": {
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "action_order": "bb_first_every_turn",
            "logged_turns": [1, 2],
            "decisions_per_root": 4,
            "btn_seat_assignment": "root_index_mod_2",
            "policy_query_progression_separated": True,
            "decision_local_sampling": True,
        },
    }
    manifest["collection_content_sha256"] = canonical_sha256(
        _manifest_stable_payload(manifest)
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def collect_hu_behavior_traces(
    config: BehaviorTraceCollectionConfig,
    behavior_model: FrozenBehaviorModel,
    *,
    root_count: int,
    root_index_start: int = 0,
    forced_decks: Mapping[int, Sequence[str]] | None = None,
    existing_dataset: CollectedBehaviorTraces | None = None,
) -> CollectedBehaviorTraces:
    """Collect a contiguous deterministic root range, optionally resuming one."""
    if isinstance(root_count, bool) or not isinstance(root_count, int) or root_count <= 0:
        raise ValueError("root_count must be a positive integer")
    if (
        isinstance(root_index_start, bool)
        or not isinstance(root_index_start, int)
        or root_index_start < 0
    ):
        raise ValueError("root_index_start must be a non-negative integer")
    model_id, model_sha = _model_identity(behavior_model)
    forced = dict(forced_decks or {})
    expected_indices = set(range(root_index_start, root_index_start + root_count))
    if config.challenge_deck_source == CALLER_SUPPLIED_DECKS:
        if set(forced) != expected_indices:
            raise ValueError("caller-supplied challenge needs exactly one deck per new root")
    elif forced:
        raise ValueError("forced decks are allowed only by caller-supplied challenge config")

    prior_records: tuple[dict[str, Any], ...] = ()
    prior_roots: tuple[dict[str, Any], ...] = ()
    dataset_start = root_index_start
    prior_runtime = 0
    if existing_dataset is not None:
        if existing_dataset.config != config:
            raise ValueError("resume config does not match the existing collection")
        if existing_dataset.root_index_stop_exclusive != root_index_start:
            raise ValueError("resume range must start exactly after the existing range")
        existing_policy = existing_dataset.manifest["policy"]
        if existing_policy != {"model_id": model_id, "model_sha256": model_sha}:
            raise ValueError("resume behavior policy does not match existing collection")
        dataset_start = existing_dataset.root_index_start
        prior_records = existing_dataset.records
        prior_roots = existing_dataset.hidden_roots
        prior_runtime = existing_dataset.elapsed_runtime_ns

    existing_root_ids = {str(record["root_id"]) for record in prior_records}
    started = time.perf_counter_ns()
    new_records: list[dict[str, Any]] = []
    new_roots: list[dict[str, Any]] = []
    sample_contract = sampling_contract(config)
    for root_index in range(root_index_start, root_index_start + root_count):
        plan = build_root_deal_plan(
            config,
            root_index,
            forced_deck=forced.get(root_index),
        )
        if plan.root_id in existing_root_ids:
            raise ValueError(f"duplicate root during resume: {plan.root_id}")
        records, hidden_trace = _collect_one_root(
            config,
            behavior_model,
            plan,
            model_id=model_id,
            model_sha256=model_sha,
            sampling=sample_contract,
        )
        new_records.extend(records)
        new_roots.append(hidden_trace)
        existing_root_ids.add(plan.root_id)
    elapsed = prior_runtime + (time.perf_counter_ns() - started)
    all_records = prior_records + tuple(new_records)
    all_roots = prior_roots + tuple(new_roots)
    stop = root_index_start + root_count
    manifest = _build_collection_manifest(
        records=all_records,
        hidden_roots=all_roots,
        config=config,
        root_index_start=dataset_start,
        root_index_stop_exclusive=stop,
        model_id=model_id,
        model_sha256=model_sha,
        elapsed_runtime_ns=elapsed,
    )
    result = CollectedBehaviorTraces(
        records=all_records,
        hidden_roots=all_roots,
        manifest=manifest,
        config=config,
        root_index_start=dataset_start,
        root_index_stop_exclusive=stop,
        elapsed_runtime_ns=elapsed,
    )
    verify_collected_behavior_traces(result)
    return result


_ROOT_KEYS = {
    "schema",
    "root_id",
    "root_index",
    "seed_namespace",
    "root_sampling_mode",
    "challenge_id",
    "challenge_deck_source",
    "challenge_target",
    "physical_deck_order",
    "seat_assignment",
    "deal_sequence",
    "t0_baseline_sha256",
    "behavior_target_id",
    "policy_sha256",
    "realized_actions",
    "final_public_state",
}
_EVENT_KEYS = {
    "turn",
    "actor",
    "seat",
    "information_digest",
    "action_key",
    "placements",
    "discard",
    "source_action_probability",
    "policy_source",
}


def _require_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys mismatch")


def _event_action(event: Mapping[str, Any], legal: Mapping[str, Action]) -> Action:
    _require_keys(event, _EVENT_KEYS, label="root action event")
    key = event["action_key"]
    if not isinstance(key, str) or key not in legal:
        raise ValueError("hidden root contains an illegal action")
    action = legal[key]
    if event["placements"] != _action_payload(action)["placements"]:
        raise ValueError("hidden root placements do not match action key")
    if event["discard"] != action.discard:
        raise ValueError("hidden root discard does not match action key")
    if not isinstance(event["source_action_probability"], str):
        raise TypeError("hidden root action probability must be exact text")
    probability = Fraction(event["source_action_probability"])
    if event["source_action_probability"] != f"{probability.numerator}/{probability.denominator}":
        raise ValueError("hidden root action probability is not canonical")
    if not Fraction(0) < probability <= Fraction(1):
        raise ValueError("hidden root action probability is outside (0,1]")
    return action


def verify_hidden_root_trace(
    trace: Mapping[str, Any],
    *,
    config: BehaviorTraceCollectionConfig,
    decision_records: Sequence[Mapping[str, Any]],
    model_id: str,
    model_sha256: str,
) -> str:
    """Replay one hidden preimage and return its exact root commitment."""
    _require_keys(trace, _ROOT_KEYS, label="hidden root trace")
    if trace["schema"] != ROOT_TRACE_SCHEMA:
        raise ValueError("unsupported hidden root trace schema")
    root_index = trace["root_index"]
    if isinstance(root_index, bool) or not isinstance(root_index, int) or root_index < 0:
        raise ValueError("hidden root index is invalid")
    root_id = derive_root_id(config, root_index)
    if trace["root_id"] != root_id:
        raise ValueError("hidden root identity mismatch")
    if (
        trace["seed_namespace"] != config.seed_namespace
        or trace["root_sampling_mode"] != config.root_sampling_mode
        or trace["challenge_id"] != config.challenge_id
        or trace["challenge_deck_source"] != config.challenge_deck_source
    ):
        raise ValueError("hidden root collection contract mismatch")
    expected_challenge_target = (
        targeted_joker_cell(root_index)
        if config.challenge_deck_source == DETERMINISTIC_JOKER_CYCLE
        else None
    )
    expected_challenge_payload = (
        {
            "turn": expected_challenge_target[0],
            "actor": expected_challenge_target[1],
            "visible_joker_count": expected_challenge_target[2],
        }
        if expected_challenge_target is not None
        else None
    )
    if trace["challenge_target"] != expected_challenge_payload:
        raise ValueError("hidden root challenge target mismatch")
    deck = _validated_deck(trace["physical_deck_order"], label="hidden root deck")
    if config.root_sampling_mode == NATURAL_UNIFORM_SHUFFLE:
        if deck != _deterministic_shuffle(config, root_id):
            raise ValueError("natural root deck does not match deterministic generation")
    elif config.challenge_deck_source == DETERMINISTIC_JOKER_CYCLE:
        expected = _targeted_cycle_deck(
            _deterministic_shuffle(config, root_id), root_index
        )
        if deck != expected:
            raise ValueError("targeted Joker root deck does not match its challenge")
    btn_seat = root_index % 2
    seats = {"btn": btn_seat, "bb": 1 - btn_seat}
    if trace["seat_assignment"] != seats:
        raise ValueError("hidden root seat assignment mismatch")
    if trace["t0_baseline_sha256"] != t0_baseline_sha256():
        raise ValueError("hidden root T0 baseline hash mismatch")
    if trace["behavior_target_id"] != model_id or trace["policy_sha256"] != model_sha256:
        raise ValueError("hidden root behavior policy identity mismatch")

    deals = trace["deal_sequence"]
    if not isinstance(deals, list) or len(deals) != len(DEAL_SPECS):
        raise ValueError("hidden root deal sequence length mismatch")
    offset = 0
    draw_by_key: dict[tuple[int, str], tuple[str, ...]] = {}
    for raw, (turn, actor, count) in zip(deals, DEAL_SPECS):
        if not isinstance(raw, Mapping) or set(raw) != {"turn", "actor", "seat", "cards"}:
            raise ValueError("hidden root deal entry keys mismatch")
        expected_cards = list(deck[offset : offset + count])
        expected = {"turn": turn, "actor": actor, "seat": seats[actor], "cards": expected_cards}
        if dict(raw) != expected:
            raise ValueError("hidden root deal does not match physical deck/order")
        draw_by_key[(turn, actor)] = tuple(expected_cards)
        offset += count

    events = trace["realized_actions"]
    if not isinstance(events, list) or len(events) != 6:
        raise ValueError("hidden root must contain two T0 and four T1/T2 actions")
    state, bb_t0, btn_t0 = initialize_t0_state(
        draw_by_key[(0, "bb")], draw_by_key[(0, "btn")]
    )
    for event, actor, expected_action in zip(events[:2], ACTORS, (bb_t0, btn_t0)):
        if not isinstance(event, Mapping):
            raise TypeError("hidden root action event must be an object")
        legal = {action_key(expected_action): expected_action}
        action = _event_action(event, legal)
        if (
            event["turn"] != 0
            or event["actor"] != actor
            or event["seat"] != seats[actor]
            or event["information_digest"] is not None
            or event["policy_source"] != "content_addressed_t0_baseline"
            or event["source_action_probability"] != "1/1"
            or action_key(action) != action_key(expected_action)
        ):
            raise ValueError("hidden root T0 action contract mismatch")

    records_by_key = {(r["turn"], r["actor"]): r for r in decision_records}
    if set(records_by_key) != set(LOGGED_DECISIONS) or len(decision_records) != 4:
        raise ValueError("hidden root requires exactly four matching decision records")
    for event, (turn, actor) in zip(events[2:], LOGGED_DECISIONS):
        if not isinstance(event, Mapping):
            raise TypeError("hidden root action event must be an object")
        information = build_behavior_query(
            state,
            turn=turn,
            actor=actor,
            current_draw=draw_by_key[(turn, actor)],
        )
        actor_rows = state.board_bb if actor == "bb" else state.board_btn
        legal = {
            action_key(action): action
            for action in get_turn_actions(list(information.current_draw), _board(actor_rows))
        }
        action = _event_action(event, legal)
        record = records_by_key[(turn, actor)]
        verified = verify_behavior_decision_log(record)
        if (
            event["turn"] != turn
            or event["actor"] != actor
            or event["seat"] != seats[actor]
            or event["information_digest"] != information.digest()
            or event["information_digest"] != record["information_digest"]
            or event["action_key"] != record["observed_action_key"]
            or event["discard"] != record["observed_discard"]
            or event["source_action_probability"] != record["source_action_probability"]
            or verified.root_id != root_id
        ):
            raise ValueError("hidden root action does not match its public decision record")
        if not isinstance(event["policy_source"], str) or not event["policy_source"]:
            raise ValueError("hidden root policy source must be non-empty")
        state, _ = apply_behavior_decision(state, information, action_key(action))

    expected_final = {
        "board_bb": _board_payload(state.board_bb),
        "board_btn": _board_payload(state.board_btn),
        "public_action_history": _history_payload(state.public_action_history),
    }
    if trace["final_public_state"] != expected_final:
        raise ValueError("hidden root final public state mismatch")
    commitment = _root_commitment(trace)
    if any(record["root_commitment"] != commitment for record in decision_records):
        raise ValueError("hidden root preimage hash does not match decision commitment")
    return commitment


def verify_collected_behavior_traces(dataset: CollectedBehaviorTraces) -> dict[str, Any]:
    """Fail closed over decisions, hidden preimages, contiguous roots and wrapper."""
    manifest = dataset.manifest
    if not isinstance(manifest, Mapping):
        raise TypeError("collection manifest must be an object")
    recorded_hash = manifest.get("manifest_sha256")
    if not isinstance(recorded_hash, str) or not _SHA256_RE.fullmatch(recorded_hash):
        raise ValueError("collection manifest requires a SHA-256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if canonical_sha256(unsigned) != recorded_hash:
        raise ValueError("collection manifest SHA-256 mismatch")
    content_hash = manifest.get("collection_content_sha256")
    if content_hash != canonical_sha256(_manifest_stable_payload(manifest)):
        raise ValueError("collection content SHA-256 mismatch")
    if manifest.get("schema") != COLLECTION_SCHEMA or manifest.get("promotion_eligible") is not False:
        raise ValueError("unsupported/promotable raw collection manifest")
    config = BehaviorTraceCollectionConfig.from_canonical_dict(manifest["collection_config"])
    if config != dataset.config:
        raise ValueError("dataset config does not match collection manifest")
    start = manifest["root_index_start"]
    stop = manifest["root_index_stop_exclusive"]
    if (
        start != dataset.root_index_start
        or stop != dataset.root_index_stop_exclusive
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise ValueError("collection root index range mismatch")
    if len(dataset.hidden_roots) != stop - start or len(dataset.records) != 4 * (stop - start):
        raise ValueError("collection root/decision cardinality mismatch")
    if manifest["root_count"] != len(dataset.hidden_roots) or manifest["decision_count"] != len(dataset.records):
        raise ValueError("collection manifest counts mismatch")
    if manifest["policy_query_count"] != len(dataset.records) or manifest["model_evaluation_count"] != len(dataset.records):
        raise ValueError("collection query/evaluation counts mismatch")
    if manifest["elapsed_runtime_ns"] != dataset.elapsed_runtime_ns or (
        isinstance(dataset.elapsed_runtime_ns, bool)
        or not isinstance(dataset.elapsed_runtime_ns, int)
        or dataset.elapsed_runtime_ns < 0
    ):
        raise ValueError("collection runtime observation mismatch")
    policy = manifest["policy"]
    model_id = policy["model_id"]
    model_sha = policy["model_sha256"]
    expected_root_ids = [derive_root_id(config, index) for index in range(start, stop)]
    actual_root_ids = [str(trace["root_id"]) for trace in dataset.hidden_roots]
    if actual_root_ids != expected_root_ids or len(set(actual_root_ids)) != len(actual_root_ids):
        raise ValueError("hidden root order/identity/duplicate mismatch")
    records_by_root: dict[str, list[Mapping[str, Any]]] = {root: [] for root in expected_root_ids}
    for record in dataset.records:
        root_id = str(record["root_id"])
        if root_id not in records_by_root:
            raise ValueError("decision references a root outside the contiguous range")
        records_by_root[root_id].append(record)
    commitments = []
    for trace in dataset.hidden_roots:
        root_id = str(trace["root_id"])
        commitments.append(
            verify_hidden_root_trace(
                trace,
                config=config,
                decision_records=records_by_root[root_id],
                model_id=model_id,
                model_sha256=model_sha,
            )
        )
    behavior_manifest = verify_behavior_decision_dataset(
        dataset.records, manifest["behavior_decision_manifest"]
    )
    if manifest["record_order_sha256"] != canonical_sha256(
        [record["record_sha256"] for record in dataset.records]
    ):
        raise ValueError("decision record order commitment mismatch")
    if manifest["root_id_order_sha256"] != canonical_sha256(actual_root_ids):
        raise ValueError("root ID order commitment mismatch")
    hidden_meta = manifest["hidden_root_artifact"]
    if hidden_meta["root_commitment_order_sha256"] != canonical_sha256(commitments):
        raise ValueError("hidden root commitment order mismatch")
    if hidden_meta["preimage_in_decision_records"] is not False:
        raise ValueError("hidden root preimages cannot be declared in decision records")
    rebuilt = _build_collection_manifest(
        records=dataset.records,
        hidden_roots=dataset.hidden_roots,
        config=config,
        root_index_start=start,
        root_index_stop_exclusive=stop,
        model_id=model_id,
        model_sha256=model_sha,
        elapsed_runtime_ns=dataset.elapsed_runtime_ns,
    )
    if rebuilt != dict(manifest):
        raise ValueError("collection manifest does not match decisions/hidden roots")
    return behavior_manifest


def _default_manifest_path(decisions_path: Path) -> Path:
    return decisions_path.with_name("manifest.json")


def _default_roots_path(decisions_path: Path) -> Path:
    return decisions_path.with_name("roots.jsonl")


def _stage_atomic(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def write_behavior_trace_dataset(
    dataset: CollectedBehaviorTraces,
    decisions_path: str | Path,
    *,
    roots_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> CollectedBehaviorTraces:
    """Atomically replace each artifact, then independently read it back."""
    verify_collected_behavior_traces(dataset)
    decisions = Path(decisions_path)
    roots = Path(roots_path) if roots_path is not None else _default_roots_path(decisions)
    manifest = Path(manifest_path) if manifest_path is not None else _default_manifest_path(decisions)
    resolved = [path.resolve() for path in (decisions, roots, manifest)]
    if len(set(resolved)) != 3:
        raise ValueError("decisions, hidden roots and manifest paths must be distinct")
    payloads = {
        decisions: ("\n".join(canonical_json(record) for record in dataset.records) + "\n").encode("utf-8"),
        roots: ("\n".join(canonical_json(trace) for trace in dataset.hidden_roots) + "\n").encode("utf-8"),
        manifest: (canonical_json(dataset.manifest) + "\n").encode("utf-8"),
    }
    staged: dict[Path, Path] = {}
    try:
        for path, payload in payloads.items():
            staged[path] = _stage_atomic(path, payload)
        # The manifest is the commit marker; replace it last.
        for path in (decisions, roots, manifest):
            os.replace(staged.pop(path), path)
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)
    return read_behavior_trace_dataset(
        decisions,
        roots_path=roots,
        manifest_path=manifest,
    )


def _read_canonical_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    raw = path.read_text(encoding="utf-8")
    if not raw or not raw.endswith("\n"):
        raise ValueError(f"{path.name} must be non-empty canonical JSONL with trailing newline")
    lines = raw[:-1].split("\n")
    if any(not line for line in lines):
        raise ValueError(f"{path.name} contains an empty JSONL row")
    values: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict) or canonical_json(value) != line:
            raise ValueError(f"{path.name} contains a non-canonical JSON row")
        values.append(value)
    return tuple(values)


def read_behavior_trace_dataset(
    decisions_path: str | Path,
    *,
    roots_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> CollectedBehaviorTraces:
    decisions = Path(decisions_path)
    roots = Path(roots_path) if roots_path is not None else _default_roots_path(decisions)
    manifest_file = Path(manifest_path) if manifest_path is not None else _default_manifest_path(decisions)
    records = _read_canonical_jsonl(decisions)
    hidden_roots = _read_canonical_jsonl(roots)
    raw_manifest = manifest_file.read_text(encoding="utf-8")
    if not raw_manifest.endswith("\n") or raw_manifest.count("\n") != 1:
        raise ValueError("manifest must be one canonical JSON object with trailing newline")
    manifest = json.loads(raw_manifest[:-1])
    if not isinstance(manifest, dict) or canonical_json(manifest) != raw_manifest[:-1]:
        raise ValueError("manifest is not canonical JSON")
    config = BehaviorTraceCollectionConfig.from_canonical_dict(manifest["collection_config"])
    result = CollectedBehaviorTraces(
        records=records,
        hidden_roots=hidden_roots,
        manifest=manifest,
        config=config,
        root_index_start=manifest["root_index_start"],
        root_index_stop_exclusive=manifest["root_index_stop_exclusive"],
        elapsed_runtime_ns=manifest["elapsed_runtime_ns"],
    )
    verify_collected_behavior_traces(result)
    return result


def resume_behavior_trace_dataset(
    decisions_path: str | Path,
    behavior_model: FrozenBehaviorModel,
    *,
    additional_root_count: int,
    forced_decks: Mapping[int, Sequence[str]] | None = None,
    roots_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> CollectedBehaviorTraces:
    """Read/verify, append the exact next roots, atomically publish and read back."""
    existing = read_behavior_trace_dataset(
        decisions_path,
        roots_path=roots_path,
        manifest_path=manifest_path,
    )
    resumed = collect_hu_behavior_traces(
        existing.config,
        behavior_model,
        root_count=additional_root_count,
        root_index_start=existing.root_index_stop_exclusive,
        forced_decks=forced_decks,
        existing_dataset=existing,
    )
    return write_behavior_trace_dataset(
        resumed,
        decisions_path,
        roots_path=roots_path,
        manifest_path=manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Collect with the four frozen HU T1/T2 checkpoint specialists.

    Caller-supplied challenge decks intentionally remain API-only.  The CLI
    exposes natural IID collection and the preregistered deterministic 12-cell
    Joker challenge as separate output directories/manifests.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-count", type=int, required=True)
    parser.add_argument("--seed-namespace")
    parser.add_argument(
        "--root-sampling-mode",
        choices=(NATURAL_UNIFORM_SHUFFLE, TARGETED_JOKER_CHALLENGE),
        default=NATURAL_UNIFORM_SHUFFLE,
    )
    parser.add_argument("--challenge-id", default="m3-joker-challenge-v1")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    from ai.tutor.frozen_behavior_torch import (
        build_known_hu_policy_value_prior_dispatch,
    )

    output_dir = Path(args.output_dir).resolve()
    decisions_path = output_dir / "decisions.jsonl"
    behavior_model = build_known_hu_policy_value_prior_dispatch(args.workspace_root)
    if args.resume:
        if args.seed_namespace is not None:
            parser.error("--seed-namespace is read from the existing manifest on --resume")
        dataset = resume_behavior_trace_dataset(
            decisions_path,
            behavior_model,
            additional_root_count=args.root_count,
        )
    else:
        if not args.seed_namespace:
            parser.error("--seed-namespace is required for a new collection")
        if args.root_sampling_mode == NATURAL_UNIFORM_SHUFFLE:
            config = BehaviorTraceCollectionConfig(
                seed_namespace=args.seed_namespace,
                root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
            )
        else:
            config = BehaviorTraceCollectionConfig(
                seed_namespace=args.seed_namespace,
                root_sampling_mode=TARGETED_JOKER_CHALLENGE,
                challenge_id=args.challenge_id,
                challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
            )
        dataset = collect_hu_behavior_traces(
            config,
            behavior_model,
            root_count=args.root_count,
        )
        dataset = write_behavior_trace_dataset(dataset, decisions_path)
    print(
        canonical_json(
            {
                "schema": "ofc_behavior_trace_cli_result/v1",
                "decisions_path": str(decisions_path),
                "roots_path": str(_default_roots_path(decisions_path)),
                "manifest_path": str(_default_manifest_path(decisions_path)),
                "root_count": dataset.root_count,
                "decision_count": len(dataset.records),
                "root_sampling_mode": dataset.config.root_sampling_mode,
                "collection_content_sha256": dataset.manifest[
                    "collection_content_sha256"
                ],
                "manifest_sha256": dataset.manifest["manifest_sha256"],
            }
        )
    )
    return 0


__all__ = [
    "CALLER_SUPPLIED_DECKS",
    "COLLECTION_SCHEMA",
    "DETERMINISTIC_JOKER_CYCLE",
    "NATURAL_UNIFORM_SHUFFLE",
    "ROOT_TRACE_SCHEMA",
    "TARGETED_JOKER_CHALLENGE",
    "BehaviorTraceCollectionConfig",
    "CollectedBehaviorTraces",
    "HUTraceState",
    "PolicyEvaluation",
    "apply_behavior_decision",
    "build_behavior_query",
    "build_root_deal_plan",
    "canonical_t0_action",
    "collect_hu_behavior_traces",
    "derive_root_id",
    "evaluate_behavior_query",
    "initialize_t0_state",
    "main",
    "read_behavior_trace_dataset",
    "resume_behavior_trace_dataset",
    "sample_policy_evaluation",
    "sampling_contract",
    "t0_baseline_manifest",
    "t0_baseline_sha256",
    "targeted_joker_cell",
    "verify_collected_behavior_traces",
    "verify_hidden_root_trace",
    "write_behavior_trace_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
