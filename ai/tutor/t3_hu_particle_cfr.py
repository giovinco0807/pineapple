"""Bounded shared-information-set CFR+ prototype for public T3 particles.

This module implements the first phase-2 invariant from
``ai/docs/t3_public_cfr_plan.md`` without pretending to be the full-card T3
solver.  A caller supplies a finite set of physical particles and a terminal
BB-utility vector for every legal action in each particle.  The prototype:

* creates exactly one policy node per :class:`InfoSetKey`;
* sums chance-weighted action vectors across compatible particles *before*
  computing a counterfactual regret update; and
* applies one shared strategy to every particle in that information set.

Particle IDs, opponent private cards, undealt cards, and particle weights are
diagnostic/physical state.  They never become policy keys.  There is no
per-particle ``max``/``min`` action selection in the solve path.

The utility vectors are fixed leaves rather than a recursively traversed
public tree.  Consequently this is a deterministic correctness prototype,
not a production equilibrium solver or a full HU-exact result.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ai.tutor.exact_late import (
    PHYSICAL_BB_T4_ACTION_VECTOR_SCHEMA,
    physical_bb_t4_state_commitment,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle


ExactInput = Fraction | int | str


def _exact(value: ExactInput, *, label: str) -> Fraction:
    """Return an exact rational and reject silently inexact binary floats."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{label} must be an exact Fraction/int/str")
    try:
        return value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} is not an exact rational: {value!r}") from exc


def _frozen_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ParticleActionUtilityVector:
    """All legal BB-utility leaves for one physical particle at one node.

    ``particle_id`` is only a stable diagnostic label.  ``particle.weight`` is
    the counterfactual chance reach contributed by this physical world.
    Utilities are expressed from BB's perspective, so BB nodes maximize and
    BTN nodes minimize the aggregated vector.
    """

    particle_id: str
    particle: JointParticle
    infoset_key: InfoSetKey
    action_utilities: Mapping[str, ExactInput]

    def __post_init__(self) -> None:
        particle_id = str(self.particle_id)
        if not particle_id:
            raise ValueError("particle_id must be a non-empty diagnostic label")
        object.__setattr__(self, "particle_id", particle_id)

        if self.infoset_key.own_recall != self.particle.own_recall(self.infoset_key.actor):
            raise ValueError(
                "infoset_key own recall does not match the acting player's physical particle"
            )
        if not self.action_utilities:
            raise ValueError("a particle action vector must contain at least one action")
        if any(not isinstance(action, str) or not action for action in self.action_utilities):
            raise ValueError("action IDs must be non-empty strings")

        exact = {
            action: _exact(value, label=f"utility[{particle_id!r}][{action!r}]")
            for action, value in sorted(self.action_utilities.items())
        }
        object.__setattr__(self, "action_utilities", _frozen_mapping(exact))


def _physical_metric_fraction(value: Any, *, label: str) -> Fraction:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{label} must be a finite numeric score")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return Fraction(str(value))
    try:
        exact = value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} must be a finite numeric score") from exc
    return exact


def particle_action_vector_from_physical_t4_result(
    result: Mapping[str, Any],
    *,
    particle: JointParticle,
    infoset_key: InfoSetKey,
) -> ParticleActionUtilityVector:
    """Bind one conditioned Rust T4 vector to its exact physical infoset.

    The commitment check prevents a valid vector from being accidentally
    attached to a different public board, BB draw, or represented remaining
    range.  Selection still happens only after multiple returned vectors are
    grouped by ``InfoSetKey`` in :func:`solve_shared_infoset_particle_cfr_plus`.
    """
    if not isinstance(result, Mapping):
        raise TypeError("physical T4 result must be a mapping")
    if result.get("schema") != PHYSICAL_BB_T4_ACTION_VECTOR_SCHEMA:
        raise ValueError("physical T4 result has an unsupported schema")
    if infoset_key.phase != "t4_first" or infoset_key.turn != 4 or infoset_key.actor != "bb":
        raise ValueError("physical T4 vectors require a canonical BB t4_first infoset")
    if result.get("position_contract_version") != infoset_key.contract_version:
        raise ValueError("physical T4 result and infoset use different position contracts")
    if result.get("actor") != "bb" or result.get("is_btn") is not False:
        raise ValueError("physical T4 result is not a canonical BB vector")
    if result.get("source") != "rust_exact_physical_t4_action_vector":
        raise ValueError("physical T4 result is not from the conditioned Rust vector API")
    if result.get("selection_performed") is not False:
        raise ValueError("physical T4 result must not contain a policy selection")
    if result.get("particle_aggregation_performed") is not False:
        raise ValueError("physical T4 result must not pre-aggregate particles")
    if result.get("requires_infoset_aggregation") is not True:
        raise ValueError("physical T4 result must require infoset aggregation")
    if result.get("public_policy_safe") is not False:
        raise ValueError("physical T4 result must remain an internal physical vector")
    if "best" in result or "chosen_action" in result:
        raise ValueError("physical T4 result contains a forbidden preselected action")

    particle_id = result.get("particle_id")
    if not isinstance(particle_id, str) or not particle_id:
        raise ValueError("physical T4 result requires a non-empty opaque particle_id")
    expected_commitment = physical_bb_t4_state_commitment(
        {
            "turn": 4,
            "actor": "bb",
            "is_btn": False,
            "first_actor": "bb",
            "position_contract_version": infoset_key.contract_version,
            "particle_id": particle_id,
            "board_self": {
                row: list(cards)
                for row, cards in zip(("top", "middle", "bottom"), infoset_key.board_bb)
            },
            "board_opponent": {
                row: list(cards)
                for row, cards in zip(("top", "middle", "bottom"), infoset_key.board_btn)
            },
            "dealt_cards": list(infoset_key.current_draw),
            "remaining_cards": list(particle.undealt_cards),
        }
    )
    if result.get("physical_state_commitment") != expected_commitment:
        raise ValueError(
            "physical T4 result commitment does not match the infoset/particle range"
        )

    raw_keys = result.get("action_keys")
    metrics_by_key = result.get("metrics_by_action_key")
    if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
        raise ValueError("physical T4 result requires a complete action_keys vector")
    action_ids = tuple(str(action) for action in raw_keys)
    if any(not action for action in action_ids) or len(action_ids) != len(set(action_ids)):
        raise ValueError("physical T4 action keys must be unique non-empty strings")
    if tuple(sorted(action_ids)) != action_ids:
        raise ValueError("physical T4 action keys must use stable lexical order")
    if not isinstance(metrics_by_key, Mapping) or set(metrics_by_key) != set(action_ids):
        raise ValueError("physical T4 result has incomplete metrics_by_action_key coverage")
    for counter in ("legal_actions", "evaluated_actions", "candidate_count"):
        if int(result.get(counter) or 0) != len(action_ids):
            raise ValueError(f"physical T4 result has inconsistent {counter}")

    utilities: dict[str, Fraction] = {}
    for action in action_ids:
        metrics = metrics_by_key[action]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"physical T4 metrics for {action!r} must be a mapping")
        if metrics.get("source") != "exact_hu_response" or metrics.get(
            "opponent_response"
        ) is not True:
            raise ValueError(f"physical T4 metrics for {action!r} are not an exact response")
        utilities[action] = _physical_metric_fraction(
            metrics.get("score"),
            label=f"physical T4 score[{action!r}]",
        )
    return ParticleActionUtilityVector(
        particle_id=particle_id,
        particle=particle,
        infoset_key=infoset_key,
        action_utilities=utilities,
    )


@dataclass(frozen=True)
class SharedInfoSetNodeResult:
    """Exact audit data and strategies for one shared information-set node."""

    infoset_key: InfoSetKey
    action_ids: tuple[str, ...]
    particle_ids: tuple[str, ...]
    chance_mass: Fraction
    weighted_action_utility: Mapping[str, Fraction]
    conditional_action_utility: Mapping[str, Fraction]
    first_regret_delta: Mapping[str, Fraction]
    final_regrets: Mapping[str, Fraction]
    current_strategy: Mapping[str, Fraction]
    average_strategy: Mapping[str, Fraction]


@dataclass(frozen=True)
class SharedParticleCfrResult:
    """Result of CFR+ updates over finite, shared-InfoSetKey leaf nodes."""

    iterations: int
    nodes: Mapping[InfoSetKey, SharedInfoSetNodeResult]
    metadata: Mapping[str, Any]

    @property
    def current_strategy(self) -> Mapping[InfoSetKey, Mapping[str, Fraction]]:
        return _frozen_mapping(
            {key: node.current_strategy for key, node in self.nodes.items()}
        )

    @property
    def average_strategy(self) -> Mapping[InfoSetKey, Mapping[str, Fraction]]:
        return _frozen_mapping(
            {key: node.average_strategy for key, node in self.nodes.items()}
        )

    def strategy_for_particle(
        self,
        vector: ParticleActionUtilityVector,
        *,
        average: bool = True,
    ) -> Mapping[str, Fraction]:
        """Look up a policy through the information set, never particle ID."""
        try:
            node = self.nodes[vector.infoset_key]
        except KeyError as exc:
            raise KeyError("particle's information set was not solved") from exc
        return node.average_strategy if average else node.current_strategy


def _regret_matching_plus(
    regrets: Mapping[str, Fraction],
    action_ids: Sequence[str],
) -> dict[str, Fraction]:
    positive = {action: max(Fraction(0, 1), regrets[action]) for action in action_ids}
    total = sum(positive.values(), Fraction(0, 1))
    if total == 0:
        probability = Fraction(1, len(action_ids))
        return {action: probability for action in action_ids}
    return {action: positive[action] / total for action in action_ids}


def _normalize_strategy_sum(
    accumulated: Mapping[str, Fraction],
    action_ids: Sequence[str],
) -> dict[str, Fraction]:
    total = sum((accumulated[action] for action in action_ids), Fraction(0, 1))
    if total == 0:
        probability = Fraction(1, len(action_ids))
        return {action: probability for action in action_ids}
    return {action: accumulated[action] / total for action in action_ids}


def solve_shared_infoset_particle_cfr_plus(
    vectors: Iterable[ParticleActionUtilityVector],
    *,
    iterations: int,
    linear_averaging: bool = True,
) -> SharedParticleCfrResult:
    """Run exact CFR+ updates on shared information sets over finite particles.

    Every vector at one ``InfoSetKey`` must expose the same stable action IDs.
    For a node ``I``, the update first forms

    ``U_I(a) = sum_world chance_weight(world) * utility(world, a)``.

    Only then is the shared on-policy value and regret delta computed.  At BB
    nodes the delta maximizes BB utility; at BTN nodes it minimizes BB utility.
    The algorithm never regret-matches or chooses an action inside a particle.
    """
    if isinstance(iterations, bool) or int(iterations) <= 0:
        raise ValueError("iterations must be a positive integer")
    iterations = int(iterations)
    supplied = tuple(vectors)
    if not supplied:
        raise ValueError("at least one particle action vector is required")

    grouped: dict[InfoSetKey, list[ParticleActionUtilityVector]] = {}
    for vector in supplied:
        if not isinstance(vector, ParticleActionUtilityVector):
            raise TypeError("vectors must contain ParticleActionUtilityVector instances")
        grouped.setdefault(vector.infoset_key, []).append(vector)

    ordered_keys = tuple(sorted(grouped, key=lambda key: key.digest()))
    actions_by_key: dict[InfoSetKey, tuple[str, ...]] = {}
    weighted_by_key: dict[InfoSetKey, dict[str, Fraction]] = {}
    chance_mass_by_key: dict[InfoSetKey, Fraction] = {}
    particle_ids_by_key: dict[InfoSetKey, tuple[str, ...]] = {}

    for key in ordered_keys:
        node_vectors = grouped[key]
        particle_ids = tuple(vector.particle_id for vector in node_vectors)
        if len(particle_ids) != len(set(particle_ids)):
            raise ValueError("particle_id must be unique within one information set")
        action_ids = tuple(node_vectors[0].action_utilities)
        action_set = set(action_ids)
        for vector in node_vectors[1:]:
            if set(vector.action_utilities) != action_set:
                raise ValueError(
                    "all particles in one information set must expose identical action IDs"
                )
        chance_mass = sum(
            (vector.particle.weight for vector in node_vectors), Fraction(0, 1)
        )
        if chance_mass <= 0:
            raise ValueError("particle chance mass must be positive at every information set")
        weighted = {
            action: sum(
                (
                    vector.particle.weight * vector.action_utilities[action]
                    for vector in node_vectors
                ),
                Fraction(0, 1),
            )
            for action in action_ids
        }
        actions_by_key[key] = action_ids
        weighted_by_key[key] = weighted
        chance_mass_by_key[key] = chance_mass
        particle_ids_by_key[key] = particle_ids

    regrets = {
        key: {action: Fraction(0, 1) for action in actions_by_key[key]}
        for key in ordered_keys
    }
    strategy_sum = {
        key: {action: Fraction(0, 1) for action in actions_by_key[key]}
        for key in ordered_keys
    }
    first_delta: dict[InfoSetKey, dict[str, Fraction]] = {}

    for iteration in range(1, iterations + 1):
        for key in ordered_keys:
            action_ids = actions_by_key[key]
            strategy = _regret_matching_plus(regrets[key], action_ids)
            average_weight = Fraction(iteration if linear_averaging else 1, 1)
            for action in action_ids:
                strategy_sum[key][action] += average_weight * strategy[action]

            # This dot product is deliberately over the already aggregated
            # chance-weighted vector.  There is no per-particle strategy.
            weighted = weighted_by_key[key]
            on_policy = sum(
                (strategy[action] * weighted[action] for action in action_ids),
                Fraction(0, 1),
            )
            if key.actor == "bb":
                delta = {action: weighted[action] - on_policy for action in action_ids}
            else:
                delta = {action: on_policy - weighted[action] for action in action_ids}
            if iteration == 1:
                first_delta[key] = delta
            for action in action_ids:
                regrets[key][action] = max(
                    Fraction(0, 1), regrets[key][action] + delta[action]
                )

    nodes: dict[InfoSetKey, SharedInfoSetNodeResult] = {}
    for key in ordered_keys:
        action_ids = actions_by_key[key]
        chance_mass = chance_mass_by_key[key]
        weighted = weighted_by_key[key]
        current = _regret_matching_plus(regrets[key], action_ids)
        average = _normalize_strategy_sum(strategy_sum[key], action_ids)
        nodes[key] = SharedInfoSetNodeResult(
            infoset_key=key,
            action_ids=action_ids,
            particle_ids=particle_ids_by_key[key],
            chance_mass=chance_mass,
            weighted_action_utility=_frozen_mapping(weighted),
            conditional_action_utility=_frozen_mapping(
                {action: weighted[action] / chance_mass for action in action_ids}
            ),
            first_regret_delta=_frozen_mapping(first_delta[key]),
            final_regrets=_frozen_mapping(regrets[key]),
            current_strategy=_frozen_mapping(current),
            average_strategy=_frozen_mapping(average),
        )

    return SharedParticleCfrResult(
        iterations=iterations,
        nodes=_frozen_mapping(nodes),
        metadata=_frozen_mapping(
            {
                "method": "shared_infoset_particle_leaf_cfr_plus_prototype",
                "information_model": "finite_supplied_physical_particles",
                "strategy_fusion": False,
                "equilibrium_approx": False,
                "full_public_tree": False,
                "hu_exact": False,
                "utility_perspective": "bb",
                "chance_aggregation": "weighted_action_vector_before_regret_update",
                "position_contract_version": ordered_keys[0].contract_version,
                "physical_particle_count": len(supplied),
                "policy_node_count": len(nodes),
            }
        ),
    )
