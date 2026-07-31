"""Generative full-card adapter and dynamic MCCFR+ for public T3/T4 play.

This module is the physical/probability boundary between a validated
:class:`~ai.tutor.t3_hu_full_card_range.FullCardRange` and the encountered-only
dynamic external-sampling MCCFR+ loop implemented below.  Solver state can be
saved only at complete BB+BTN iteration boundaries in a content-addressed,
fail-closed checkpoint.

The probability contract is strict:

* one call to :meth:`FullCardGenerativeAdapter.sample_root_for_traversal`
  starts one traversal and samples exactly one posterior particle;
* the posterior mass is used by that categorical draw exactly once;
* the sampled physical particle is copied with ``weight == 1`` before any
  public-tree transition, making accidental posterior/chance multiplication
  fail fast;
* later three-card draws are sampled directly and uniformly from unordered
  combinations of the physical remainder.  Their probability is returned for
  audit only and is not multiplied into the sampled state;
* decisions expose only :class:`~ai.tutor.t3_hu_public_cfr.InfoSetKey` and
  stable :func:`~ai.tutor.exact_late.action_key` identities.  Hidden recalls
  and the remaining deck stay on the physical state, never in a policy key.

``X1`` and ``X2`` remain distinct physical card atoms at every boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TypeAlias

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import action_key, terminal_metrics
from ai.tutor.t3_hu_full_card_range import (
    EXPECTED_UNDEALT_BY_PHASE,
    FullCardRange,
    verify_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import (
    PendingChanceState,
    PublicTreeDecisionState,
    PublicTreeTerminalState,
    SuppliedChanceDraw,
    apply_public_tree_action,
    apply_public_tree_terminal_action,
    resolve_supplied_chance,
)


ROOT_SAMPLING_CONTRACT = "exact_fraction_sequential_conditional_v1"
DRAW_SAMPLING_CONTRACT = "uniform_unordered_three_card_combinadic_v1"
POLICY_IDENTITY_CONTRACT = "infoset_key_plus_lexical_action_key_v1"
CHECKPOINT_FORMAT = "full_card_dynamic_mccfr_checkpoint_v1"
SOLVER_STATE_FORMAT = "full_card_dynamic_mccfr_state_v1"
ADAPTER_MANIFEST_FORMAT = "full_card_dynamic_mccfr_adapter_manifest_v1"
RNG_ALGORITHM = "python_random_mt19937"
VALID_CARDS = frozenset(ALL_CARDS)
ROWS = ("top", "middle", "bottom")


class FullCardMccfrCheckpointError(ValueError):
    """A full-card checkpoint failed content or compatibility validation."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FullCardMccfrCheckpointError(
            "full-card MCCFR checkpoint contains non-canonical JSON data"
        ) from exc
    return serialized.encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_snapshot(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:  # pragma: no cover
        raise FullCardMccfrCheckpointError(
            f"{label} is not canonical JSON data"
        ) from exc


class RandomBits(Protocol):
    """Minimal deterministic entropy interface used by exact samplers."""

    def getrandbits(self, k: int) -> int: ...


def _randbelow_exact(rng: RandomBits, upper: int) -> int:
    """Return an unbiased integer in ``range(upper)`` using rejection sampling."""

    if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
        raise ValueError("upper must be a positive integer")
    if not callable(getattr(rng, "getrandbits", None)):
        raise TypeError("rng must provide getrandbits(k)")
    if upper == 1:
        return 0
    bits = (upper - 1).bit_length()
    while True:
        candidate = rng.getrandbits(bits)
        if not isinstance(candidate, int) or isinstance(candidate, bool):
            raise TypeError("rng.getrandbits(k) must return an integer")
        if candidate < 0 or candidate >= (1 << bits):
            raise ValueError("rng.getrandbits(k) returned a value outside k bits")
        if candidate < upper:
            return candidate


def sample_exact_fraction_index(
    weights: Sequence[Fraction],
    rng: RandomBits,
) -> int:
    """Sample one exact categorical index without constructing a global LCM.

    Entries are processed in their supplied order.  At entry ``i`` the method
    samples ``weights[i] / remaining_mass`` as an exact Bernoulli.  This is
    equivalent to one categorical draw and avoids converting posterior mass to
    binary floating point or materializing a potentially enormous common
    denominator.
    """

    exact = tuple(weights)
    if not exact:
        raise ValueError("exact categorical sampling requires at least one weight")
    if any(not isinstance(weight, Fraction) for weight in exact):
        raise TypeError("exact categorical weights must be fractions.Fraction")
    if any(weight <= 0 for weight in exact):
        raise ValueError("exact categorical weights must be positive")
    if sum(exact, Fraction(0, 1)) != 1:
        raise ValueError("exact categorical weights must sum exactly to one")

    remaining = Fraction(1, 1)
    for index, weight in enumerate(exact[:-1]):
        conditional = weight / remaining
        if _randbelow_exact(rng, conditional.denominator) < conditional.numerator:
            return index
        remaining -= weight
        if remaining <= 0:  # pragma: no cover - defended by exact sum/positivity
            raise AssertionError("exact categorical remaining mass became non-positive")
    return len(exact) - 1


def _unrank_combination(
    cards: tuple[str, ...],
    *,
    choose: int,
    rank: int,
) -> tuple[str, ...]:
    """Return the lexicographic ``rank``-th unordered combination."""

    total = math.comb(len(cards), choose)
    if not 0 <= rank < total:
        raise ValueError("combination rank is outside the valid range")
    selected: list[str] = []
    start = 0
    remaining_rank = rank
    for slots in range(choose, 0, -1):
        last_start = len(cards) - slots
        for index in range(start, last_start + 1):
            suffix_count = math.comb(len(cards) - index - 1, slots - 1)
            if remaining_rank < suffix_count:
                selected.append(cards[index])
                start = index + 1
                break
            remaining_rank -= suffix_count
        else:  # pragma: no cover - arithmetic defense
            raise AssertionError("combination unranking exhausted the card set")
    return tuple(selected)


@dataclass(frozen=True)
class UniformThreeCardSample:
    """One directly sampled physical draw and its unapplied chance mass."""

    cards: tuple[str, ...]
    sampled_rank: int
    combination_count: int
    probability: Fraction


def sample_uniform_three_cards(
    cards: Sequence[str],
    rng: RandomBits,
) -> UniformThreeCardSample:
    """Sample one unordered three-card combination uniformly and exactly."""

    canonical = tuple(sorted(str(card) for card in cards))
    if len(canonical) < 3:
        raise ValueError("a three-card draw requires at least three remaining cards")
    if len(canonical) != len(set(canonical)):
        raise ValueError("remaining cards contain duplicate physical atoms")
    invalid = sorted(card for card in canonical if card not in VALID_CARDS)
    if invalid:
        raise ValueError(f"remaining cards contain invalid physical cards: {invalid}")
    combination_count = math.comb(len(canonical), 3)
    sampled_rank = _randbelow_exact(rng, combination_count)
    sampled_cards = _unrank_combination(
        canonical,
        choose=3,
        rank=sampled_rank,
    )
    return UniformThreeCardSample(
        cards=sampled_cards,
        sampled_rank=sampled_rank,
        combination_count=combination_count,
        probability=Fraction(1, combination_count),
    )


@dataclass(frozen=True)
class RootPosteriorSample:
    """The sole posterior sample for one traversal.

    ``posterior_probability`` and ``particle_commitment`` are audit provenance.
    A policy/regret implementation must key only on ``state.infoset_key``.
    """

    traversal_index: int
    particle_commitment: str
    posterior_probability: Fraction
    state: PublicTreeDecisionState
    range_content_sha256: str
    range_build_sha256: str
    posterior_probability_applied_to_state: bool = False

    def __post_init__(self) -> None:
        if self.traversal_index <= 0:
            raise ValueError("traversal_index must be positive")
        if self.posterior_probability <= 0:
            raise ValueError("sampled posterior probability must be positive")
        if self.state.particle.weight != 1:
            raise ValueError("a sampled root state must carry unit particle weight")
        if self.posterior_probability_applied_to_state:
            raise ValueError("posterior mass must not be reapplied to a sampled state")


@dataclass(frozen=True)
class SampledChanceTransition:
    """One sampled post-action draw and the resulting decision state."""

    completed_phase: str
    next_phase: str
    draw: UniformThreeCardSample
    before_remaining_count: int
    after_remaining_count: int
    state: PublicTreeDecisionState
    chance_probability_applied_to_state: bool = False

    def __post_init__(self) -> None:
        if self.before_remaining_count - self.after_remaining_count != 3:
            raise ValueError("sampled chance transition must remove exactly three cards")
        if self.state.particle.weight != 1:
            raise ValueError("sampled chance state must carry unit particle weight")
        if self.chance_probability_applied_to_state:
            raise ValueError("sampled chance mass must not be reapplied to state weight")


PhysicalActionResult: TypeAlias = PendingChanceState | PublicTreeTerminalState


@dataclass(frozen=True)
class _RootEntry:
    commitment: str
    posterior_probability: Fraction
    particle: JointParticle


def _board(rows: Sequence[Sequence[str]]) -> Board:
    if len(rows) != 3:
        raise ValueError("board rows must contain top/middle/bottom")
    return Board(
        top=list(rows[0]),
        middle=list(rows[1]),
        bottom=list(rows[2]),
    )


def _unit_particle(particle: JointParticle) -> JointParticle:
    return JointParticle(
        bb_recall=particle.bb_recall,
        btn_recall=particle.btn_recall,
        undealt_cards=particle.undealt_cards,
        weight=Fraction(1, 1),
    )


def _require_unit_state_weight(
    state: PublicTreeDecisionState | PendingChanceState | PublicTreeTerminalState,
) -> None:
    if state.particle.weight != 1:
        raise ValueError(
            "generative full-card traversal requires unit particle weight; "
            "posterior/chance mass is consumed only by direct sampling"
        )


class FullCardGenerativeAdapter:
    """Validated lazy physical adapter for one public T3/T4 root range."""

    def __init__(self, observation: InfoSetKey, root_range: FullCardRange) -> None:
        if not isinstance(observation, InfoSetKey):
            raise TypeError("observation must be an InfoSetKey")
        if not isinstance(root_range, FullCardRange):
            raise TypeError("root_range must be a FullCardRange")
        if observation.contract_version != POSITION_CONTRACT_VERSION:
            raise ValueError("full-card traversal requires bb_first_v1")
        verification = verify_full_card_range(observation, root_range)

        entries = tuple(
            sorted(
                (
                    _RootEntry(commitment, particle.weight, particle)
                    for commitment, particle in zip(
                        root_range.particle_commitments,
                        root_range.particles,
                    )
                ),
                key=lambda entry: entry.commitment,
            )
        )
        if not entries:
            raise ValueError("root range must contain at least one particle")
        if len({entry.commitment for entry in entries}) != len(entries):
            raise ValueError("root range particle commitments must be unique")
        if sum(
            (entry.posterior_probability for entry in entries), Fraction(0, 1)
        ) != 1:
            raise ValueError("root posterior mass must sum exactly to one")

        self.observation = observation
        self.root_range = root_range
        self._verification = MappingProxyType(dict(verification))
        self._root_entries = entries
        self._audit_lock = threading.Lock()
        self._traversal_count = 0
        self._root_sample_count = 0
        self._root_outcome_counts: dict[str, int] = {}
        self._chance_sample_count = 0
        self._chance_samples_by_next_phase: dict[str, int] = {}

    @property
    def root_distribution(self) -> tuple[tuple[str, Fraction], ...]:
        """Stable commitment/probability pairs; never a policy identity."""

        return tuple(
            (entry.commitment, entry.posterior_probability)
            for entry in self._root_entries
        )

    @property
    def range_verification(self) -> Mapping[str, Any]:
        return self._verification

    def sample_root_for_traversal(self, rng: RandomBits) -> RootPosteriorSample:
        """Start one traversal by consuming posterior mass exactly once."""

        weights = tuple(entry.posterior_probability for entry in self._root_entries)
        selected_index = sample_exact_fraction_index(weights, rng)
        selected = self._root_entries[selected_index]
        state = PublicTreeDecisionState(
            infoset_key=self.observation,
            particle=_unit_particle(selected.particle),
        )
        if state.infoset_key != self.observation:  # pragma: no cover - constructor defense
            raise AssertionError("sampled root changed the public information key")

        with self._audit_lock:
            self._traversal_count += 1
            traversal_index = self._traversal_count
            self._root_sample_count += 1
            self._root_outcome_counts[selected.commitment] = (
                self._root_outcome_counts.get(selected.commitment, 0) + 1
            )
        return RootPosteriorSample(
            traversal_index=traversal_index,
            particle_commitment=selected.commitment,
            posterior_probability=selected.posterior_probability,
            state=state,
            range_content_sha256=self.root_range.range_content_sha256,
            range_build_sha256=self.root_range.range_build_sha256,
        )

    @staticmethod
    def information_key(state: PublicTreeDecisionState) -> InfoSetKey:
        if not isinstance(state, PublicTreeDecisionState):
            raise TypeError("decision state must be PublicTreeDecisionState")
        _require_unit_state_weight(state)
        # canonical_json reruns the forbidden-private-field defense.
        state.infoset_key.canonical_json()
        return state.infoset_key

    def legal_actions(
        self,
        state: PublicTreeDecisionState,
    ) -> tuple[tuple[str, Action], ...]:
        """Enumerate legal actions using public/own information only."""

        key = self.information_key(state)
        actor_rows = key.board_bb if key.actor == "bb" else key.board_btn
        generated = get_turn_actions(list(key.current_draw), _board(actor_rows))
        by_id: dict[str, Action] = {}
        for action in generated:
            stable_id = action_key(action)
            if stable_id in by_id:
                raise ValueError(f"legal action_key collision: {stable_id}")
            by_id[stable_id] = action
        if not by_id:
            raise ValueError("decision state has no legal actions")
        return tuple(sorted(by_id.items(), key=lambda item: item[0]))

    def apply_action_id(
        self,
        state: PublicTreeDecisionState,
        stable_action_id: str,
    ) -> PhysicalActionResult:
        """Apply one stable legal action without inspecting hidden cards."""

        stable_action_id = str(stable_action_id)
        legal = dict(self.legal_actions(state))
        action = legal.get(stable_action_id)
        if action is None:
            raise ValueError("action ID is not legal at this information set")
        if state.infoset_key.phase == "t4_second":
            terminal = apply_public_tree_terminal_action(state, action)
            _require_unit_state_weight(terminal)
            return terminal
        pending = apply_public_tree_action(state, action)
        _require_unit_state_weight(pending)
        return pending

    def sample_next_draw(
        self,
        pending: PendingChanceState,
        rng: RandomBits,
    ) -> SampledChanceTransition:
        """Directly sample the next physical draw without reach multiplication."""

        if not isinstance(pending, PendingChanceState):
            raise TypeError("pending must be a PendingChanceState")
        _require_unit_state_weight(pending)
        expected_after = EXPECTED_UNDEALT_BY_PHASE.get(pending.next_phase)
        if expected_after is None:
            raise ValueError(f"unsupported next phase: {pending.next_phase!r}")
        before = len(pending.remaining_cards)
        if before != expected_after + 3:
            raise ValueError(
                f"pending {pending.next_phase} chance requires {expected_after + 3} "
                f"remaining cards before its draw, got {before}"
            )

        sampled = sample_uniform_three_cards(pending.remaining_cards, rng)
        # A singleton supplied outcome is a transition mechanism only.  The
        # physical 1/C(n,3) mass has already been consumed by direct sampling.
        branch = resolve_supplied_chance(
            pending,
            (SuppliedChanceDraw(sampled.cards, Fraction(1, 1)),),
        )[0]
        state = branch.state
        _require_unit_state_weight(state)
        after = len(state.remaining_cards)
        if after != expected_after:
            raise AssertionError("sampled chance produced the wrong physical remainder")

        with self._audit_lock:
            self._chance_sample_count += 1
            self._chance_samples_by_next_phase[pending.next_phase] = (
                self._chance_samples_by_next_phase.get(pending.next_phase, 0) + 1
            )
        return SampledChanceTransition(
            completed_phase=pending.completed_phase,
            next_phase=pending.next_phase,
            draw=sampled,
            before_remaining_count=before,
            after_remaining_count=after,
            state=state,
        )

    @staticmethod
    def terminal_metrics_bb(
        terminal: PublicTreeTerminalState,
    ) -> Mapping[str, Any]:
        """Return canonical completed-board metrics from BB's perspective."""

        if not isinstance(terminal, PublicTreeTerminalState):
            raise TypeError("terminal must be a PublicTreeTerminalState")
        _require_unit_state_weight(terminal)
        metrics = terminal_metrics(
            _board(terminal.board_bb),
            _board(terminal.board_btn),
        )
        return MappingProxyType(dict(metrics))

    @classmethod
    def terminal_utility_bb(cls, terminal: PublicTreeTerminalState) -> float:
        return float(cls.terminal_metrics_bb(terminal)["score"])

    def sampling_audit(self) -> Mapping[str, Any]:
        """Return a stable snapshot proving one posterior sample per traversal."""

        with self._audit_lock:
            if self._root_sample_count != self._traversal_count:
                raise AssertionError("root posterior was not sampled once per traversal")
            return MappingProxyType(
                {
                    "traversals": self._traversal_count,
                    "root_posterior_samples": self._root_sample_count,
                    # Do not expose particle commitments in an artifact-facing
                    # audit.  Aggregate counts prove accounting without making
                    # a hidden world part of a policy/result identity.
                    "root_outcome_count_total": sum(
                        self._root_outcome_counts.values()
                    ),
                    "root_distinct_outcomes_sampled": len(
                        self._root_outcome_counts
                    ),
                    "future_draw_samples": self._chance_sample_count,
                    "future_draw_samples_by_next_phase": dict(
                        sorted(self._chance_samples_by_next_phase.items())
                    ),
                    "root_sampling_contract": ROOT_SAMPLING_CONTRACT,
                    "draw_sampling_contract": DRAW_SAMPLING_CONTRACT,
                    "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
                    "stable_root_order": "particle_commitment_lexical",
                    "root_posterior_sampled_once_per_traversal": True,
                    "posterior_probability_multiplied_after_sampling": False,
                    "chance_probability_multiplied_after_sampling": False,
                    "joint_particle_weight_after_sampling": "1/1",
                    "policy_key_contains_hidden_assignment": False,
                    "physical_joker_ids": ["X1", "X2"],
                    "range_content_sha256": self.root_range.range_content_sha256,
                    "range_build_sha256": self.root_range.range_build_sha256,
                    "position_contract_version": POSITION_CONTRACT_VERSION,
                }
            )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "adapter": "full_card_generative_t3_t4_v1",
                "scope": "physical_sampling_boundary_for_dynamic_mccfr",
                "root_phase": self.observation.phase,
                "root_actor": self.observation.actor,
                "root_particle_count": self.root_range.particle_count,
                "root_sampling_contract": ROOT_SAMPLING_CONTRACT,
                "draw_sampling_contract": DRAW_SAMPLING_CONTRACT,
                "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
                "root_sampling_distribution": "full_card_range_particle_weight",
                "importance_sampling": False,
                "posterior_probability_multiplied_after_sampling": False,
                "chance_probability_multiplied_after_sampling": False,
                "joint_particle_weight_used_after_sampling": False,
                "terminal_utility_source": "exact_late.terminal_metrics.score_bb",
                "terminal_utility_includes_direct_fantasyland_ev": True,
                "dynamic_mccfr_integrated": True,
                "full_card_policy_promoted": False,
                "physical_joker_ids": ["X1", "X2"],
                "range_content_sha256": self.root_range.range_content_sha256,
                "range_build_sha256": self.root_range.range_build_sha256,
                "position_contract_version": POSITION_CONTRACT_VERSION,
            }
        )


def seeded_rng(seed: int) -> random.Random:
    """Construct the adapter's current deterministic Python RNG boundary."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return random.Random(seed)


def _regret_matching_plus(
    regrets: Mapping[str, float],
    action_ids: Sequence[str],
) -> dict[str, float]:
    positive = [max(0.0, float(regrets[action_id])) for action_id in action_ids]
    total = math.fsum(positive)
    if total <= 0.0:
        probability = 1.0 / len(action_ids)
        return {action_id: probability for action_id in action_ids}
    return {
        action_id: positive[index] / total
        for index, action_id in enumerate(action_ids)
    }


def _normalized_average(
    accumulated: Mapping[str, float],
    action_ids: Sequence[str],
) -> dict[str, float]:
    total = math.fsum(float(accumulated[action_id]) for action_id in action_ids)
    if total <= 0.0:
        probability = 1.0 / len(action_ids)
        return {action_id: probability for action_id in action_ids}
    return {
        action_id: float(accumulated[action_id]) / total
        for action_id in action_ids
    }


def _stable_infoset_order(keys: Sequence[InfoSetKey]) -> tuple[InfoSetKey, ...]:
    return tuple(sorted(keys, key=lambda key: (key.digest(), key.canonical_json())))


def serialize_strategy_profile(
    profile: Mapping[InfoSetKey, Mapping[str, float]],
) -> str:
    """Serialize an encountered strategy in stable information/action order.

    The serialization contains only legal player information.  It never
    contains a physical particle commitment, opponent recall, remaining deck,
    or particle weight.
    """

    records: list[dict[str, Any]] = []
    for key in _stable_infoset_order(tuple(profile)):
        if not isinstance(key, InfoSetKey):
            raise TypeError("strategy profile keys must be InfoSetKey")
        canonical_json = key.canonical_json()
        actions = profile[key]
        if not isinstance(actions, Mapping) or not actions:
            raise ValueError("strategy profile entries must be non-empty mappings")
        action_rows: list[dict[str, Any]] = []
        probabilities: list[float] = []
        for action_id in sorted(str(raw_id) for raw_id in actions):
            probability = float(actions[action_id])
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError("strategy probabilities must be finite and non-negative")
            probabilities.append(probability)
            action_rows.append(
                {
                    "action_id": action_id,
                    "probability": probability,
                }
            )
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("strategy probabilities must sum to one")
        records.append(
            {
                "infoset_digest": key.digest(),
                "infoset": json.loads(canonical_json),
                "actions": action_rows,
            }
        )
    return json.dumps(
        {
            "schema": "ofc_full_card_public_strategy/v1",
            "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
            "records": records,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _DynamicMccfrTables:
    """Encountered-only tables with fail-closed information contracts."""

    def __init__(self, max_infosets: int) -> None:
        self.max_infosets = max_infosets
        self.action_ids: dict[InfoSetKey, tuple[str, ...]] = {}
        self.actors: dict[InfoSetKey, str] = {}
        self.regrets: dict[InfoSetKey, dict[str, float]] = {}
        self.strategy_sum: dict[InfoSetKey, dict[str, float]] = {}
        self._canonical_by_digest: dict[str, str] = {}

    def ensure(
        self,
        key: InfoSetKey,
        action_ids: Sequence[str],
    ) -> bool:
        if not isinstance(key, InfoSetKey):
            raise TypeError("dynamic MCCFR policy key must be InfoSetKey")
        supplied = tuple(str(action_id) for action_id in action_ids)
        canonical_actions = tuple(sorted(supplied))
        if not canonical_actions or len(canonical_actions) != len(set(canonical_actions)):
            raise ValueError("dynamic MCCFR needs unique non-empty action IDs")
        if supplied != canonical_actions:
            raise ValueError("dynamic MCCFR action IDs must be lexically stable")
        if any(not action_id for action_id in canonical_actions):
            raise ValueError("dynamic MCCFR action IDs must be non-empty")

        canonical_json = key.canonical_json()
        digest = key.digest()
        prior_json = self._canonical_by_digest.get(digest)
        if prior_json is not None and prior_json != canonical_json:
            raise ValueError("InfoSetKey SHA256 collision")
        prior_actions = self.action_ids.get(key)
        if prior_actions is not None:
            if prior_actions != canonical_actions:
                raise ValueError(
                    "shared InfoSetKey action-set mismatch: "
                    f"expected {prior_actions}, got {canonical_actions}"
                )
            if self.actors[key] != key.actor:
                raise ValueError("shared InfoSetKey actor mismatch")
            return False

        if len(self.action_ids) >= self.max_infosets:
            raise RuntimeError(
                "dynamic MCCFR max_infosets exceeded: "
                f"{len(self.action_ids) + 1} > {self.max_infosets}"
            )
        self._canonical_by_digest[digest] = canonical_json
        self.action_ids[key] = canonical_actions
        self.actors[key] = key.actor
        self.regrets[key] = {action_id: 0.0 for action_id in canonical_actions}
        self.strategy_sum[key] = {
            action_id: 0.0 for action_id in canonical_actions
        }
        return True

    def current_strategy(self, key: InfoSetKey) -> dict[str, float]:
        return _regret_matching_plus(self.regrets[key], self.action_ids[key])

    def average_strategy(self, key: InfoSetKey) -> dict[str, float]:
        return _normalized_average(self.strategy_sum[key], self.action_ids[key])

    @property
    def stable_keys(self) -> tuple[InfoSetKey, ...]:
        return _stable_infoset_order(tuple(self.action_ids))


@dataclass
class _MutableFullCardMccfrStats:
    traversals: int = 0
    bb_traversals: int = 0
    btn_traversals: int = 0
    root_posterior_samples: int = 0
    root_outcome_counts: dict[str, int] = field(default_factory=dict)
    future_draw_samples: int = 0
    future_draw_samples_by_next_phase: dict[str, int] = field(default_factory=dict)
    decision_visits: int = 0
    terminal_visits: int = 0
    traverser_actions_expanded: int = 0
    opponent_action_samples: int = 0
    opponent_action_cache_hits: int = 0
    strategy_sum_updates: int = 0
    infosets_created: int = 0

    def begin(self, traverser: str) -> None:
        self.traversals += 1
        if traverser == "bb":
            self.bb_traversals += 1
        elif traverser == "btn":
            self.btn_traversals += 1
        else:  # pragma: no cover - internal caller defense
            raise ValueError("traverser must be bb or btn")

    def record_root(self, commitment: str) -> None:
        self.root_posterior_samples += 1
        self.root_outcome_counts[commitment] = (
            self.root_outcome_counts.get(commitment, 0) + 1
        )

    def record_future_draw(self, next_phase: str) -> None:
        self.future_draw_samples += 1
        self.future_draw_samples_by_next_phase[next_phase] = (
            self.future_draw_samples_by_next_phase.get(next_phase, 0) + 1
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "traversals": self.traversals,
            "traversals_by_actor": {
                "bb": self.bb_traversals,
                "btn": self.btn_traversals,
            },
            "root_posterior_samples": self.root_posterior_samples,
            "root_outcome_counts": dict(sorted(self.root_outcome_counts.items())),
            "future_draw_samples": self.future_draw_samples,
            "future_draw_samples_by_next_phase": dict(
                sorted(self.future_draw_samples_by_next_phase.items())
            ),
            "decision_visits": self.decision_visits,
            "terminal_visits": self.terminal_visits,
            "traverser_actions_expanded": self.traverser_actions_expanded,
            "opponent_action_samples": self.opponent_action_samples,
            "opponent_action_cache_hits": self.opponent_action_cache_hits,
            "strategy_sum_updates": self.strategy_sum_updates,
            "infosets_created": self.infosets_created,
        }

    @classmethod
    def from_snapshot(cls, raw: Any) -> "_MutableFullCardMccfrStats":
        expected = {
            "traversals",
            "traversals_by_actor",
            "root_posterior_samples",
            "root_outcome_counts",
            "future_draw_samples",
            "future_draw_samples_by_next_phase",
            "decision_visits",
            "terminal_visits",
            "traverser_actions_expanded",
            "opponent_action_samples",
            "opponent_action_cache_hits",
            "strategy_sum_updates",
            "infosets_created",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise FullCardMccfrCheckpointError(
                "checkpoint sampling_stats schema is invalid"
            )
        by_actor = raw["traversals_by_actor"]
        root_outcomes = raw["root_outcome_counts"]
        draws_by_phase = raw["future_draw_samples_by_next_phase"]
        if not isinstance(by_actor, dict) or set(by_actor) != {"bb", "btn"}:
            raise FullCardMccfrCheckpointError(
                "checkpoint traversal actor counts are invalid"
            )
        if not isinstance(root_outcomes, dict) or any(
            not isinstance(key, str) or not key for key in root_outcomes
        ):
            raise FullCardMccfrCheckpointError(
                "checkpoint root outcome counts are invalid"
            )
        valid_next_phases = {"t3_second", "t4_first", "t4_second"}
        if not isinstance(draws_by_phase, dict) or not set(draws_by_phase).issubset(
            valid_next_phases
        ):
            raise FullCardMccfrCheckpointError(
                "checkpoint future-draw phase counts are invalid"
            )

        def count(value: Any, *, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FullCardMccfrCheckpointError(
                    f"checkpoint {label} must be a non-negative integer"
                )
            return value

        return cls(
            traversals=count(raw["traversals"], label="traversals"),
            bb_traversals=count(by_actor["bb"], label="bb traversals"),
            btn_traversals=count(by_actor["btn"], label="btn traversals"),
            root_posterior_samples=count(
                raw["root_posterior_samples"], label="root posterior samples"
            ),
            root_outcome_counts={
                key: count(value, label=f"root outcome {key!r}")
                for key, value in root_outcomes.items()
            },
            future_draw_samples=count(
                raw["future_draw_samples"], label="future draw samples"
            ),
            future_draw_samples_by_next_phase={
                key: count(value, label=f"future draw phase {key!r}")
                for key, value in draws_by_phase.items()
            },
            decision_visits=count(
                raw["decision_visits"], label="decision visits"
            ),
            terminal_visits=count(
                raw["terminal_visits"], label="terminal visits"
            ),
            traverser_actions_expanded=count(
                raw["traverser_actions_expanded"],
                label="traverser actions expanded",
            ),
            opponent_action_samples=count(
                raw["opponent_action_samples"], label="opponent action samples"
            ),
            opponent_action_cache_hits=count(
                raw["opponent_action_cache_hits"],
                label="opponent action cache hits",
            ),
            strategy_sum_updates=count(
                raw["strategy_sum_updates"], label="strategy sum updates"
            ),
            infosets_created=count(
                raw["infosets_created"], label="infosets created"
            ),
        )


@dataclass(frozen=True)
class _RestoredFullCardCheckpoint:
    completed_iterations: int
    tables: _DynamicMccfrTables
    rng_state: tuple[Any, ...]
    stats: _MutableFullCardMccfrStats
    checkpoint_sha256: str


def _float_hex(value: float, *, label: str) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FullCardMccfrCheckpointError(
            f"checkpoint {label} must be finite"
        )
    return numeric.hex()


def _float_from_hex(raw: Any, *, label: str, non_negative: bool) -> float:
    if not isinstance(raw, str):
        raise FullCardMccfrCheckpointError(
            f"checkpoint {label} must be a hexadecimal float string"
        )
    try:
        value = float.fromhex(raw)
    except ValueError as exc:
        raise FullCardMccfrCheckpointError(
            f"checkpoint {label} is not a valid hexadecimal float"
        ) from exc
    if not math.isfinite(value) or (non_negative and value < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise FullCardMccfrCheckpointError(
            f"checkpoint {label} must be {qualifier}"
        )
    return value


def _encode_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 3:
        raise FullCardMccfrCheckpointError("unexpected Python RNG state shape")
    version, internal_state, gaussian = state
    if not isinstance(version, int) or not isinstance(internal_state, tuple):
        raise FullCardMccfrCheckpointError("unexpected Python RNG state values")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in internal_state
    ):
        raise FullCardMccfrCheckpointError(
            "Python RNG internal state is not integer-only"
        )
    return {
        "algorithm": RNG_ALGORITHM,
        "version": version,
        "internal_state": list(internal_state),
        "gaussian_next": (
            None
            if gaussian is None
            else _float_hex(gaussian, label="RNG gaussian cache")
        ),
    }


def _decode_rng_state(raw: Any) -> tuple[Any, ...]:
    if not isinstance(raw, dict) or set(raw) != {
        "algorithm",
        "version",
        "internal_state",
        "gaussian_next",
    }:
        raise FullCardMccfrCheckpointError(
            "checkpoint RNG state schema is invalid"
        )
    if raw["algorithm"] != RNG_ALGORITHM:
        raise FullCardMccfrCheckpointError(
            "checkpoint RNG algorithm is incompatible"
        )
    version = raw["version"]
    internal_state = raw["internal_state"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise FullCardMccfrCheckpointError("checkpoint RNG version is invalid")
    if not isinstance(internal_state, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in internal_state
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint RNG internal state is invalid"
        )
    gaussian_raw = raw["gaussian_next"]
    gaussian = (
        None
        if gaussian_raw is None
        else _float_from_hex(
            gaussian_raw,
            label="RNG gaussian cache",
            non_negative=False,
        )
    )
    state = (version, tuple(internal_state), gaussian)
    probe = random.Random()
    try:
        probe.setstate(state)
    except (TypeError, ValueError) as exc:
        raise FullCardMccfrCheckpointError(
            "checkpoint RNG state is not accepted by this Python runtime"
        ) from exc
    return state


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FullCardMccfrCheckpointError(
                f"checkpoint JSON contains duplicate key {key!r}"
            )
        value[key] = item
    return value


_SOURCE_BINDING_PATHS = (
    "ai/tutor/t3_hu_full_card_mccfr.py",
    "ai/tutor/t3_hu_full_card_range.py",
    "ai/tutor/t3_hu_public_tree.py",
    "ai/tutor/exact_late.py",
    "ai/engine/action_space.py",
    "ai/engine/scoring.py",
    "ai/mcts/rollout_evaluator.py",
    "ai/config/fl_ev.json",
)


def _terminal_utility_binding() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    sources: dict[str, str] = {}
    for relative_path in _SOURCE_BINDING_PATHS:
        source_path = project_root / relative_path
        try:
            content = source_path.read_bytes()
        except OSError as exc:
            raise FullCardMccfrCheckpointError(
                f"cannot bind solver source {relative_path!r}: {exc}"
            ) from exc
        sources[relative_path] = hashlib.sha256(content).hexdigest()

    fl_path = project_root / "ai/config/fl_ev.json"
    try:
        fl_config = json.loads(
            fl_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except FullCardMccfrCheckpointError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise FullCardMccfrCheckpointError(
            "canonical Fantasyland EV configuration cannot be read"
        ) from exc
    if not isinstance(fl_config, dict):
        raise FullCardMccfrCheckpointError(
            "canonical Fantasyland EV configuration must be an object"
        )
    canonical_fl = _canonical_snapshot(fl_config, label="Fantasyland EV config")
    binding = {
        "terminal_utility_source": "exact_late.terminal_metrics.score_bb",
        "terminal_utility_perspective": "bb",
        "canonical_fl_ev_config": canonical_fl,
        "canonical_fl_ev_config_sha256": _content_sha256(canonical_fl),
        "source_sha256": dict(sorted(sources.items())),
    }
    binding["binding_sha256"] = _content_sha256(binding)
    return binding


def _range_behavior_binding(adapter: FullCardGenerativeAdapter) -> dict[str, Any]:
    metadata = adapter.root_range.metadata
    behavior_manifest = metadata.get("behavior_model_manifest")
    if not isinstance(behavior_manifest, Mapping):
        raise FullCardMccfrCheckpointError(
            "full-card range is missing its behavior model manifest"
        )
    behavior_snapshot = _canonical_snapshot(
        dict(behavior_manifest), label="behavior model manifest"
    )
    if _content_sha256(behavior_snapshot) != adapter.root_range.behavior_model_sha256:
        raise FullCardMccfrCheckpointError(
            "behavior model manifest does not match the range behavior hash"
        )
    if behavior_snapshot.get("model_id") != adapter.root_range.behavior_model_id:
        raise FullCardMccfrCheckpointError(
            "behavior model manifest does not match the range behavior ID"
        )

    commitments = list(adapter.root_range.particle_commitments)
    root_distribution = [
        {
            "commitment": commitment,
            "posterior_probability": (
                f"{probability.numerator}/{probability.denominator}"
            ),
        }
        for commitment, probability in adapter.root_distribution
    ]
    return {
        "observation_digest": adapter.root_range.observation_digest,
        "particle_count": adapter.root_range.particle_count,
        "particle_commitments_sha256": _content_sha256(commitments),
        "root_distribution_sha256": _content_sha256(root_distribution),
        "range_content_sha256": adapter.root_range.range_content_sha256,
        "range_build_sha256": adapter.root_range.range_build_sha256,
        "behavior_model_id": adapter.root_range.behavior_model_id,
        "behavior_model_sha256": adapter.root_range.behavior_model_sha256,
        "behavior_model_manifest": behavior_snapshot,
    }


def _adapter_manifest(adapter: FullCardGenerativeAdapter) -> dict[str, Any]:
    root = adapter.observation
    phase_order = ("t3_first", "t3_second", "t4_first", "t4_second")
    if root.phase not in phase_order:  # pragma: no cover - InfoSetKey defense
        raise FullCardMccfrCheckpointError("unsupported root phase")
    manifest = {
        "format": ADAPTER_MANIFEST_FORMAT,
        "adapter": "full_card_generative_t3_t4_v1",
        "root_infoset_canonical_json": root.canonical_json(),
        "root_infoset_sha256": root.digest(),
        "remaining_phase_sequence": list(phase_order[phase_order.index(root.phase) :]),
        "root_sampling_contract": ROOT_SAMPLING_CONTRACT,
        "draw_sampling_contract": DRAW_SAMPLING_CONTRACT,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
        "range_behavior_binding": _range_behavior_binding(adapter),
        "terminal_utility_binding": _terminal_utility_binding(),
    }
    manifest["manifest_sha256"] = _content_sha256(manifest)
    return manifest


def _solver_checkpoint_config(max_infosets: int) -> dict[str, Any]:
    return {
        "method": "full_card_dynamic_external_sampling_mccfr_plus_v1",
        "sampling_scheme": "external_sampling",
        "traverser_schedule": "bb_then_btn_each_iteration",
        "alternating_updates": True,
        "regret_matching_plus": True,
        "regret_clip_scope": "once_per_infoset_after_traversal",
        "average_strategy_estimator": "two_player_simple_opponent_node",
        "encountered_infoset_tables": True,
        "max_infosets": max_infosets,
        "rng_algorithm": RNG_ALGORITHM,
        "position_contract_version": POSITION_CONTRACT_VERSION,
    }


def _checkpoint_payload(
    *,
    adapter: FullCardGenerativeAdapter,
    completed_iterations: int,
    seed: int,
    linear_averaging: bool,
    tables: _DynamicMccfrTables,
    rng_state: tuple[Any, ...],
    stats: _MutableFullCardMccfrStats,
) -> dict[str, Any]:
    manifest = _adapter_manifest(adapter)
    return {
        "format": SOLVER_STATE_FORMAT,
        "completed_iterations": completed_iterations,
        "seed": seed,
        "linear_averaging": linear_averaging,
        "solver_config": _solver_checkpoint_config(tables.max_infosets),
        "adapter_manifest": manifest,
        "adapter_manifest_sha256": _content_sha256(manifest),
        "tables": [
            {
                "infoset_canonical_json": key.canonical_json(),
                "infoset_sha256": key.digest(),
                "actor": tables.actors[key],
                "stable_action_ids": list(tables.action_ids[key]),
                "regret_plus_hex": [
                    _float_hex(
                        tables.regrets[key][action_id],
                        label="cumulative regret",
                    )
                    for action_id in tables.action_ids[key]
                ],
                "strategy_sum_hex": [
                    _float_hex(
                        tables.strategy_sum[key][action_id],
                        label="strategy sum",
                    )
                    for action_id in tables.action_ids[key]
                ],
            }
            for key in tables.stable_keys
        ],
        "rng_state": _encode_rng_state(rng_state),
        "sampling_stats": stats.snapshot(),
    }


def _checkpoint_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_sha256": _content_sha256(payload),
        "payload": payload,
    }


def _atomic_write_checkpoint(
    path: str | os.PathLike[str], envelope: Mapping[str, Any]
) -> None:
    target = Path(path)
    if not target.name:
        raise FullCardMccfrCheckpointError("checkpoint path must name a file")
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_json_bytes(envelope) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_checkpoint_envelope(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    try:
        serialized = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise FullCardMccfrCheckpointError(
            f"cannot read full-card MCCFR checkpoint: {exc}"
        ) from exc
    try:
        envelope = json.loads(
            serialized, object_pairs_hook=_reject_duplicate_json_keys
        )
    except FullCardMccfrCheckpointError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise FullCardMccfrCheckpointError(
            "full-card MCCFR checkpoint is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "format",
        "checkpoint_sha256",
        "payload",
    }:
        raise FullCardMccfrCheckpointError(
            "checkpoint envelope schema is invalid"
        )
    if envelope["format"] != CHECKPOINT_FORMAT:
        raise FullCardMccfrCheckpointError("checkpoint format is incompatible")
    supplied_hash = envelope["checkpoint_sha256"]
    if (
        not isinstance(supplied_hash, str)
        or len(supplied_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_hash)
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint SHA-256 field is invalid"
        )
    calculated_hash = _content_sha256(envelope["payload"])
    if supplied_hash != calculated_hash:
        raise FullCardMccfrCheckpointError(
            "checkpoint content SHA-256 mismatch"
        )
    if not isinstance(envelope["payload"], dict):
        raise FullCardMccfrCheckpointError("checkpoint payload must be an object")
    return envelope["payload"], supplied_hash


def _infoset_from_canonical_json(raw: Any) -> InfoSetKey:
    if not isinstance(raw, str):
        raise FullCardMccfrCheckpointError(
            "checkpoint InfoSetKey canonical JSON must be a string"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(value, dict) or set(value) != {
            "contract_version",
            "actor",
            "turn",
            "phase",
            "board_bb",
            "board_btn",
            "public_action_history",
            "own_recall",
            "current_draw",
            "fantasy_state",
        }:
            raise ValueError("InfoSetKey object schema is invalid")
        board_bb = value["board_bb"]
        board_btn = value["board_btn"]
        history = value["public_action_history"]
        recall = value["own_recall"]
        if (
            not isinstance(board_bb, dict)
            or set(board_bb) != set(ROWS)
            or not isinstance(board_btn, dict)
            or set(board_btn) != set(ROWS)
            or not isinstance(history, list)
            or not isinstance(recall, dict)
            or set(recall) != {"dealt_by_turn", "discards_by_turn"}
        ):
            raise ValueError("InfoSetKey nested schema is invalid")
        dealt = recall["dealt_by_turn"]
        discards = recall["discards_by_turn"]
        if not isinstance(dealt, list) or not isinstance(discards, list):
            raise ValueError("InfoSetKey recall schema is invalid")
        own_recall = PrivateRecall(
            dealt_by_turn=tuple(
                (row["turn"], tuple(row["cards"]))
                for row in dealt
                if isinstance(row, dict) and set(row) == {"turn", "cards"}
            ),
            discards_by_turn=tuple(
                (row["turn"], row["card"])
                for row in discards
                if isinstance(row, dict) and set(row) == {"turn", "card"}
            ),
        )
        if len(own_recall.dealt_by_turn) != len(dealt) or len(
            own_recall.discards_by_turn
        ) != len(discards):
            raise ValueError("InfoSetKey recall entry schema is invalid")
        public_history = tuple(
            (
                row["turn"],
                row["actor"],
                tuple(tuple(placement) for placement in row["placements"]),
            )
            for row in history
            if isinstance(row, dict)
            and set(row) == {"turn", "actor", "placements"}
        )
        if len(public_history) != len(history):
            raise ValueError("InfoSetKey public history schema is invalid")
        key = InfoSetKey(
            contract_version=value["contract_version"],
            actor=value["actor"],
            turn=value["turn"],
            phase=value["phase"],
            board_bb=tuple(tuple(board_bb[row]) for row in ROWS),
            board_btn=tuple(tuple(board_btn[row]) for row in ROWS),
            public_action_history=public_history,
            own_recall=own_recall,
            current_draw=tuple(value["current_draw"]),
            fantasy_state=value["fantasy_state"],
        )
    except FullCardMccfrCheckpointError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise FullCardMccfrCheckpointError(
            "checkpoint InfoSetKey canonical JSON is invalid"
        ) from exc
    if key.canonical_json() != raw:
        raise FullCardMccfrCheckpointError(
            "checkpoint InfoSetKey is not canonically serialized"
        )
    return key


def _legal_action_ids_for_key(key: InfoSetKey) -> tuple[str, ...]:
    actor_rows = key.board_bb if key.actor == "bb" else key.board_btn
    actions = get_turn_actions(list(key.current_draw), _board(actor_rows))
    stable_ids = tuple(sorted(action_key(action) for action in actions))
    if not stable_ids or len(stable_ids) != len(set(stable_ids)):
        raise FullCardMccfrCheckpointError(
            "checkpoint InfoSetKey does not produce unique legal action IDs"
        )
    return stable_ids


def _validate_descendant_infoset(root: InfoSetKey, key: InfoSetKey) -> None:
    phase_order = {phase: index for index, phase in enumerate(
        ("t3_first", "t3_second", "t4_first", "t4_second")
    )}
    if phase_order[key.phase] < phase_order[root.phase]:
        raise FullCardMccfrCheckpointError(
            "checkpoint infoset precedes the current root tree"
        )
    root_history = root.public_action_history
    if key.public_action_history[: len(root_history)] != root_history:
        raise FullCardMccfrCheckpointError(
            "checkpoint infoset is outside the current public tree"
        )
    if key.phase == root.phase and key != root:
        raise FullCardMccfrCheckpointError(
            "checkpoint contains a different infoset at the root phase"
        )


def _restore_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: FullCardGenerativeAdapter,
    seed: int,
    linear_averaging: bool,
    max_infosets: int,
) -> _RestoredFullCardCheckpoint:
    payload, checkpoint_sha256 = _read_checkpoint_envelope(path)
    expected_payload_keys = {
        "format",
        "completed_iterations",
        "seed",
        "linear_averaging",
        "solver_config",
        "adapter_manifest",
        "adapter_manifest_sha256",
        "tables",
        "rng_state",
        "sampling_stats",
    }
    if set(payload) != expected_payload_keys:
        raise FullCardMccfrCheckpointError(
            "checkpoint payload schema is invalid"
        )
    if payload["format"] != SOLVER_STATE_FORMAT:
        raise FullCardMccfrCheckpointError(
            "checkpoint solver state format is incompatible"
        )
    completed_iterations = payload["completed_iterations"]
    if (
        isinstance(completed_iterations, bool)
        or not isinstance(completed_iterations, int)
        or completed_iterations <= 0
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint completed_iterations must be positive"
        )
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] != seed
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint seed does not match requested seed"
        )
    if (
        not isinstance(payload["linear_averaging"], bool)
        or payload["linear_averaging"] is not linear_averaging
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint linear_averaging setting does not match"
        )
    if payload["solver_config"] != _solver_checkpoint_config(max_infosets):
        raise FullCardMccfrCheckpointError(
            "checkpoint solver configuration is incompatible"
        )

    current_manifest = _adapter_manifest(adapter)
    manifest = payload["adapter_manifest"]
    manifest_hash = payload["adapter_manifest_sha256"]
    if not isinstance(manifest, dict) or not isinstance(manifest_hash, str):
        raise FullCardMccfrCheckpointError(
            "checkpoint adapter manifest is invalid"
        )
    if _content_sha256(manifest) != manifest_hash:
        raise FullCardMccfrCheckpointError(
            "checkpoint adapter manifest SHA-256 mismatch"
        )
    if (
        manifest != current_manifest
        or manifest_hash != _content_sha256(current_manifest)
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint range, behavior, tree, or scoring binding does not match"
        )

    raw_tables = payload["tables"]
    if not isinstance(raw_tables, list) or not raw_tables:
        raise FullCardMccfrCheckpointError(
            "checkpoint infoset tables must be a non-empty list"
        )
    if len(raw_tables) > max_infosets:
        raise FullCardMccfrCheckpointError(
            "checkpoint infoset count exceeds max_infosets"
        )
    tables = _DynamicMccfrTables(max_infosets)
    table_schema = {
        "infoset_canonical_json",
        "infoset_sha256",
        "actor",
        "stable_action_ids",
        "regret_plus_hex",
        "strategy_sum_hex",
    }
    prior_order: tuple[str, str] | None = None
    for raw_table in raw_tables:
        if not isinstance(raw_table, dict) or set(raw_table) != table_schema:
            raise FullCardMccfrCheckpointError(
                "checkpoint infoset table schema is invalid"
            )
        key = _infoset_from_canonical_json(raw_table["infoset_canonical_json"])
        order = (key.digest(), key.canonical_json())
        if prior_order is not None and order <= prior_order:
            raise FullCardMccfrCheckpointError(
                "checkpoint infoset table order is not canonical"
            )
        prior_order = order
        if raw_table["infoset_sha256"] != key.digest():
            raise FullCardMccfrCheckpointError(
                "checkpoint InfoSetKey digest does not match canonical JSON"
            )
        if raw_table["actor"] != key.actor:
            raise FullCardMccfrCheckpointError(
                "checkpoint infoset actor is incompatible"
            )
        _validate_descendant_infoset(adapter.observation, key)
        action_ids = _legal_action_ids_for_key(key)
        if raw_table["stable_action_ids"] != list(action_ids):
            raise FullCardMccfrCheckpointError(
                "checkpoint stable action IDs do not match the current tree"
            )
        regret_values = raw_table["regret_plus_hex"]
        strategy_values = raw_table["strategy_sum_hex"]
        if (
            not isinstance(regret_values, list)
            or not isinstance(strategy_values, list)
            or len(regret_values) != len(action_ids)
            or len(strategy_values) != len(action_ids)
        ):
            raise FullCardMccfrCheckpointError(
                "checkpoint action-vector length is incompatible"
            )
        try:
            tables.ensure(key, action_ids)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FullCardMccfrCheckpointError(
                "checkpoint infoset registry is incompatible"
            ) from exc
        tables.regrets[key] = {
            action_id: _float_from_hex(
                regret_values[index],
                label="cumulative regret",
                non_negative=True,
            )
            for index, action_id in enumerate(action_ids)
        }
        tables.strategy_sum[key] = {
            action_id: _float_from_hex(
                strategy_values[index],
                label="strategy sum",
                non_negative=True,
            )
            for index, action_id in enumerate(action_ids)
        }
    if adapter.observation not in tables.action_ids:
        raise FullCardMccfrCheckpointError(
            "checkpoint does not contain the current root infoset"
        )

    rng_state = _decode_rng_state(payload["rng_state"])
    stats = _MutableFullCardMccfrStats.from_snapshot(payload["sampling_stats"])
    expected_traversals = 2 * completed_iterations
    allowed_commitments = {
        commitment for commitment, _probability in adapter.root_distribution
    }
    if (
        stats.traversals != expected_traversals
        or stats.bb_traversals != completed_iterations
        or stats.btn_traversals != completed_iterations
        or stats.root_posterior_samples != expected_traversals
        or sum(stats.root_outcome_counts.values()) != expected_traversals
        or not set(stats.root_outcome_counts).issubset(allowed_commitments)
        or sum(stats.future_draw_samples_by_next_phase.values())
        != stats.future_draw_samples
        or stats.infosets_created != len(tables.action_ids)
    ):
        raise FullCardMccfrCheckpointError(
            "checkpoint sampling statistics are incompatible"
        )
    return _RestoredFullCardCheckpoint(
        completed_iterations=completed_iterations,
        tables=tables,
        rng_state=rng_state,
        stats=stats,
        checkpoint_sha256=checkpoint_sha256,
    )


def _sample_policy_action(
    action_ids: Sequence[str],
    strategy: Mapping[str, float],
    rng: random.Random,
) -> str:
    draw = rng.random()
    cumulative = 0.0
    last_positive: str | None = None
    for action_id in action_ids:
        probability = float(strategy[action_id])
        if probability > 0.0:
            last_positive = action_id
        cumulative += probability
        if draw < cumulative:
            return action_id
    if last_positive is None:  # pragma: no cover - regret matching defense
        raise AssertionError("policy sampler received no positive action")
    return last_positive


def _sample_cached_opponent_action(
    key: InfoSetKey,
    action_ids: Sequence[str],
    strategy: Mapping[str, float],
    rng: random.Random,
    cache: dict[InfoSetKey, str],
) -> tuple[str, bool]:
    """Sample one external pure action per InfoSetKey and traversal."""

    selected = cache.get(key)
    if selected is not None:
        if selected not in action_ids:
            raise ValueError("cached opponent action is not legal at shared InfoSetKey")
        return selected, True
    selected = _sample_policy_action(action_ids, strategy, rng)
    cache[key] = selected
    return selected, False


def _freeze_profile(
    keys: Sequence[InfoSetKey],
    rows: Mapping[InfoSetKey, Mapping[str, float]],
) -> Mapping[InfoSetKey, Mapping[str, float]]:
    return MappingProxyType(
        {
            key: MappingProxyType(
                {
                    action_id: float(rows[key][action_id])
                    for action_id in sorted(rows[key])
                }
            )
            for key in keys
        }
    )


@dataclass(frozen=True)
class FullCardExternalSamplingMccfrResult:
    iterations: int
    traversals: int
    seed: int
    encountered_infosets: int
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    current_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    cumulative_regret_plus: Mapping[InfoSetKey, Mapping[str, float]]
    average_strategy_json: str
    current_strategy_json: str
    average_strategy_sha256: str
    current_strategy_sha256: str
    sampling_stats: Mapping[str, Any]
    metadata: Mapping[str, Any]


def solve_full_card_external_sampling_mccfr(
    adapter: FullCardGenerativeAdapter,
    *,
    iterations: int,
    seed: int,
    max_infosets: int,
    linear_averaging: bool = True,
    resume_from: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> FullCardExternalSamplingMccfrResult:
    """Run encountered-only alternating external-sampling MCCFR+.

    Chance and the non-traversing player are sampled; every traverser action is
    expanded.  Root posterior mass and later physical chance mass are consumed
    by direct sampling and never multiplied into regret.  The returned policy
    covers encountered information sets only and carries no exact
    exploitability or promotion claim.  With ``resume_from``, ``iterations``
    is the number of additional complete BB+BTN iterations.  The optional
    ``checkpoint_path`` is atomically replaced with the final iteration-boundary
    state as canonical, content-addressed JSON.
    """

    if not isinstance(adapter, FullCardGenerativeAdapter):
        raise TypeError("adapter must be FullCardGenerativeAdapter")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(max_infosets, bool)
        or not isinstance(max_infosets, int)
        or max_infosets <= 0
    ):
        raise ValueError("max_infosets must be a positive integer")
    if not isinstance(linear_averaging, bool):
        raise TypeError("linear_averaging must be bool")

    before_adapter_audit = dict(adapter.sampling_audit())
    rng = seeded_rng(seed)
    if resume_from is None:
        completed_before = 0
        tables = _DynamicMccfrTables(max_infosets)
        stats = _MutableFullCardMccfrStats()
    else:
        restored = _restore_checkpoint(
            resume_from,
            adapter=adapter,
            seed=seed,
            linear_averaging=linear_averaging,
            max_infosets=max_infosets,
        )
        completed_before = restored.completed_iterations
        tables = restored.tables
        stats = restored.stats
        rng.setstate(restored.rng_state)
    completed_total = completed_before + iterations
    stats_root_samples_before_run = stats.root_posterior_samples
    stats_future_draws_before_run = stats.future_draw_samples

    def traverse(
        state: PublicTreeDecisionState,
        *,
        traverser: str,
        regret_delta: dict[InfoSetKey, dict[str, float]],
        opponent_choices: dict[InfoSetKey, str],
        average_updated: set[InfoSetKey],
        average_weight: float,
    ) -> float:
        stats.decision_visits += 1
        key = adapter.information_key(state)
        actions = adapter.legal_actions(state)
        action_ids = tuple(action_id for action_id, _action in actions)
        if tables.ensure(key, action_ids):
            stats.infosets_created += 1
        sigma = tables.current_strategy(key)

        def action_value(action_id: str) -> float:
            physical = adapter.apply_action_id(state, action_id)
            if isinstance(physical, PublicTreeTerminalState):
                stats.terminal_visits += 1
                return adapter.terminal_utility_bb(physical)
            if not isinstance(physical, PendingChanceState):  # pragma: no cover
                raise TypeError("adapter returned an unsupported physical action result")
            sampled = adapter.sample_next_draw(physical, rng)
            stats.record_future_draw(sampled.next_phase)
            return traverse(
                sampled.state,
                traverser=traverser,
                regret_delta=regret_delta,
                opponent_choices=opponent_choices,
                average_updated=average_updated,
                average_weight=average_weight,
            )

        if key.actor == traverser:
            values: dict[str, float] = {}
            for action_id in action_ids:
                stats.traverser_actions_expanded += 1
                values[action_id] = action_value(action_id)
            node_value = math.fsum(
                sigma[action_id] * values[action_id] for action_id in action_ids
            )
            entry = regret_delta.setdefault(
                key,
                {action_id: 0.0 for action_id in action_ids},
            )
            sign = 1.0 if traverser == "bb" else -1.0
            for action_id in action_ids:
                entry[action_id] += sign * (values[action_id] - node_value)
            return node_value

        # Two-player simple average estimator: on the other player's traversal,
        # visitation already samples this player's own reach.  Update at most
        # once per InfoSetKey so physical-history multiplicity adds no weight.
        if key not in average_updated:
            for action_id in action_ids:
                tables.strategy_sum[key][action_id] += (
                    average_weight * sigma[action_id]
                )
            average_updated.add(key)
            stats.strategy_sum_updates += 1

        selected_id, cache_hit = _sample_cached_opponent_action(
            key,
            action_ids,
            sigma,
            rng,
            opponent_choices,
        )
        if cache_hit:
            stats.opponent_action_cache_hits += 1
        else:
            stats.opponent_action_samples += 1
        return action_value(selected_id)

    for iteration in range(completed_before + 1, completed_total + 1):
        average_weight = float(iteration if linear_averaging else 1)
        for traverser in ("bb", "btn"):
            stats.begin(traverser)
            root = adapter.sample_root_for_traversal(rng)
            stats.record_root(root.particle_commitment)
            regret_delta: dict[InfoSetKey, dict[str, float]] = {}
            traverse(
                root.state,
                traverser=traverser,
                regret_delta=regret_delta,
                opponent_choices={},
                average_updated=set(),
                average_weight=average_weight,
            )
            # Every physical contribution was aggregated by InfoSetKey above.
            # Apply CFR+ clipping once per information set after the traversal.
            for key, entry in regret_delta.items():
                if tables.actors[key] != traverser:
                    raise AssertionError("regret delta was recorded for the wrong actor")
                for action_id in tables.action_ids[key]:
                    tables.regrets[key][action_id] = max(
                        0.0,
                        tables.regrets[key][action_id] + entry[action_id],
                    )

    after_adapter_audit = dict(adapter.sampling_audit())
    expected_additional_traversals = 2 * iterations
    expected_total_traversals = 2 * completed_total
    adapter_traversal_delta = (
        int(after_adapter_audit["traversals"])
        - int(before_adapter_audit["traversals"])
    )
    root_sample_delta = (
        int(after_adapter_audit["root_posterior_samples"])
        - int(before_adapter_audit["root_posterior_samples"])
    )
    adapter_future_draw_delta = (
        int(after_adapter_audit["future_draw_samples"])
        - int(before_adapter_audit["future_draw_samples"])
    )
    if adapter_traversal_delta != expected_additional_traversals:
        raise AssertionError("adapter traversal accounting does not equal 2 * iterations")
    if root_sample_delta != expected_additional_traversals:
        raise AssertionError("root posterior was not sampled once per traversal")
    if stats.root_posterior_samples - stats_root_samples_before_run != root_sample_delta:
        raise AssertionError("solver and adapter root sampling accounting diverged")
    if stats.future_draw_samples - stats_future_draws_before_run != adapter_future_draw_delta:
        raise AssertionError("solver and adapter future-draw accounting diverged")
    if (
        stats.traversals != expected_total_traversals
        or stats.bb_traversals != completed_total
        or stats.btn_traversals != completed_total
        or stats.root_posterior_samples != expected_total_traversals
        or sum(stats.root_outcome_counts.values()) != expected_total_traversals
        or sum(stats.future_draw_samples_by_next_phase.values())
        != stats.future_draw_samples
        or stats.infosets_created != len(tables.action_ids)
    ):
        raise AssertionError("cumulative full-card MCCFR sampling state is inconsistent")

    stable_keys = tables.stable_keys
    average_rows = {
        key: tables.average_strategy(key) for key in stable_keys
    }
    current_rows = {
        key: tables.current_strategy(key) for key in stable_keys
    }
    regret_rows = {
        key: dict(tables.regrets[key]) for key in stable_keys
    }
    average_json = serialize_strategy_profile(average_rows)
    current_json = serialize_strategy_profile(current_rows)
    average_sha256 = _sha256_text(average_json)
    current_sha256 = _sha256_text(current_json)

    sampling_stats = MappingProxyType(stats.snapshot())
    final_payload = _checkpoint_payload(
        adapter=adapter,
        completed_iterations=completed_total,
        seed=seed,
        linear_averaging=linear_averaging,
        tables=tables,
        rng_state=rng.getstate(),
        stats=stats,
    )
    final_envelope = _checkpoint_envelope(final_payload)
    final_checkpoint_sha256 = final_envelope["checkpoint_sha256"]
    if checkpoint_path is not None:
        _atomic_write_checkpoint(checkpoint_path, final_envelope)
    adapter_manifest = final_payload["adapter_manifest"]
    range_binding = adapter_manifest["range_behavior_binding"]
    terminal_binding = adapter_manifest["terminal_utility_binding"]
    metadata = MappingProxyType(
        {
            "method": "full_card_dynamic_external_sampling_mccfr_plus_v1",
            "adapter": "full_card_generative_t3_t4_v1",
            "sampling_scheme": "external_sampling",
            "traverser_schedule": "bb_then_btn_each_iteration",
            "alternating_updates": True,
            "regret_matching_plus": True,
            "regret_clip_scope": "once_per_infoset_after_traversal",
            "linear_averaging": linear_averaging,
            "average_strategy_estimator": "two_player_simple_opponent_node",
            "initial_unseen_strategy": "uniform_legal",
            "encountered_infoset_tables": True,
            "max_infosets": max_infosets,
            "iterations": completed_total,
            "traversals": expected_total_traversals,
            "seed": seed,
            "rng_algorithm": RNG_ALGORITHM,
            "stable_infoset_order": "digest_then_canonical_json",
            "stable_action_order": "lexical_action_key",
            "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
            "opponent_sample_cached_per_infoset": True,
            "future_chance_sampled_per_physical_state": True,
            "root_posterior_sampled_once_per_traversal": True,
            "posterior_probability_multiplied_after_sampling": False,
            "chance_probability_multiplied_after_sampling": False,
            "joint_particle_weight_used_after_sampling": False,
            "table_key_contains_particle_commitment": False,
            "table_key_contains_remaining_cards": False,
            "table_key_contains_particle_weight": False,
            "artifact_contains_raw_particle_world": False,
            "terminal_utility_perspective": "bb",
            "terminal_utility_source": "exact_late.terminal_metrics.score_bb",
            "full_card": True,
            "full_card_policy_promoted": False,
            "hu_exact": False,
            "runtime_integrated": False,
            "exact_exploitability_computed": False,
            "average_strategy_sha256": average_sha256,
            "current_strategy_sha256": current_sha256,
            "range_content_sha256": adapter.root_range.range_content_sha256,
            "range_build_sha256": adapter.root_range.range_build_sha256,
            "range_sampling_approx": bool(
                adapter.root_range.metadata.get("range_sampling_approx", False)
            ),
            "behavior_model_id": adapter.root_range.behavior_model_id,
            "behavior_model_sha256": adapter.root_range.behavior_model_sha256,
            "behavior_model_manifest": range_binding["behavior_model_manifest"],
            "particle_commitments_sha256": range_binding[
                "particle_commitments_sha256"
            ],
            "root_distribution_sha256": range_binding[
                "root_distribution_sha256"
            ],
            "terminal_utility_binding_sha256": terminal_binding[
                "binding_sha256"
            ],
            "canonical_fl_ev_config": terminal_binding[
                "canonical_fl_ev_config"
            ],
            "canonical_fl_ev_config_sha256": terminal_binding[
                "canonical_fl_ev_config_sha256"
            ],
            "source_sha256": terminal_binding["source_sha256"],
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_iteration": completed_total,
            "checkpoint_sha256": final_checkpoint_sha256,
        }
    )
    return FullCardExternalSamplingMccfrResult(
        iterations=completed_total,
        traversals=expected_total_traversals,
        seed=seed,
        encountered_infosets=len(stable_keys),
        average_strategy=_freeze_profile(stable_keys, average_rows),
        current_strategy=_freeze_profile(stable_keys, current_rows),
        cumulative_regret_plus=_freeze_profile(stable_keys, regret_rows),
        average_strategy_json=average_json,
        current_strategy_json=current_json,
        average_strategy_sha256=average_sha256,
        current_strategy_sha256=current_sha256,
        sampling_stats=sampling_stats,
        metadata=metadata,
    )
