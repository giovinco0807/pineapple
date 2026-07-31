"""Recursive CFR+ reference solver for finite reduced public trees.

This module is the deliberately bounded bridge between the one-step public
belief diagnostics in :mod:`ai.tutor.t3_hu_public_cfr` and a future full-card
T3/T4 solver.  It solves an *explicitly supplied*, finite chance/decision tree
whose decision nodes carry strict :class:`PublicTreeDecisionState` objects.

The policy identity is always ``state.infoset_key``.  Physical particles are
kept on decision states for validation and leaf evaluation, but particle IDs,
world IDs, hidden opponent cards, and chance outcome IDs never enter a policy
key.  Consequently, two physical histories with the same ``InfoSetKey`` are
forced to share one strategy.  A tree that gives those histories different
action sets is rejected before training.

``PublicTreeChanceNode`` branches are the solver's only probability source.
``JointParticle.weight`` remains provenance on the validated physical state
and is intentionally never multiplied into reach; doing both would count the
same range mass twice.

Scope
-----
This is an exact solver for the supplied reduced tree, including exact
infoset-aware best responses.  It is not a full 54-card OFC tree, is not
runtime-integrated, and does not establish that the production HU policy is
solved.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeAlias

from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_particle_cfr import (
    particle_action_vector_from_physical_t4_result,
)
from ai.tutor.t3_hu_public_cfr import Actor, InfoSetKey
from ai.tutor.t3_hu_public_tree import PublicTreeDecisionState


ExactInput = Fraction | int | str


def _exact(value: ExactInput, *, label: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{label} must be an exact Fraction/int/str")
    try:
        return value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} is not an exact rational: {value!r}") from exc


@dataclass(frozen=True)
class PublicTreeTerminalNode:
    """Terminal BB utility for one explicit physical history."""

    utility_bb: Fraction
    terminal_id: str = ""
    utility_source: str = "explicit_reduced_terminal"

    def __init__(
        self,
        utility_bb: ExactInput,
        terminal_id: str = "",
        utility_source: str = "explicit_reduced_terminal",
    ) -> None:
        object.__setattr__(self, "utility_bb", _exact(utility_bb, label="terminal utility"))
        object.__setattr__(self, "terminal_id", str(terminal_id))
        utility_source = str(utility_source)
        if not utility_source:
            raise ValueError("terminal utility_source must be non-empty")
        object.__setattr__(self, "utility_source", utility_source)


@dataclass(frozen=True)
class PublicTreeChanceBranch:
    """One named branch of an explicit chance node."""

    outcome_id: str
    probability: Fraction
    child: "PublicTreeNode"

    def __init__(
        self,
        outcome_id: str,
        probability: ExactInput,
        child: "PublicTreeNode",
    ) -> None:
        outcome_id = str(outcome_id)
        if not outcome_id:
            raise ValueError("chance outcome_id must be non-empty")
        probability = _exact(probability, label=f"chance probability {outcome_id!r}")
        if probability <= 0:
            raise ValueError("chance branch probability must be positive")
        object.__setattr__(self, "outcome_id", outcome_id)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "child", child)


@dataclass(frozen=True)
class PublicTreeChanceNode:
    """Finite explicit chance node with exact conditional probabilities."""

    branches: tuple[PublicTreeChanceBranch, ...]

    def __init__(self, branches: Iterable[PublicTreeChanceBranch]) -> None:
        supplied = tuple(branches)
        if not supplied:
            raise ValueError("chance node requires at least one branch")
        if any(not isinstance(branch, PublicTreeChanceBranch) for branch in supplied):
            raise TypeError("chance branches must be PublicTreeChanceBranch instances")
        ids = [branch.outcome_id for branch in supplied]
        if len(ids) != len(set(ids)):
            raise ValueError("chance outcome IDs must be unique within a node")
        if sum((branch.probability for branch in supplied), Fraction(0, 1)) != 1:
            raise ValueError("chance branch probabilities must sum exactly to 1")
        object.__setattr__(
            self,
            "branches",
            tuple(sorted(supplied, key=lambda branch: branch.outcome_id)),
        )


@dataclass(frozen=True)
class PublicTreeDecisionNode:
    """A physical decision history whose policy is keyed only by InfoSetKey."""

    state: PublicTreeDecisionState
    actions: tuple[tuple[str, "PublicTreeNode"], ...]

    def __init__(
        self,
        state: PublicTreeDecisionState,
        actions: Mapping[str, "PublicTreeNode"] | Iterable[tuple[str, "PublicTreeNode"]],
    ) -> None:
        if not isinstance(state, PublicTreeDecisionState):
            raise TypeError("decision state must be a PublicTreeDecisionState")
        supplied = tuple(actions.items()) if isinstance(actions, Mapping) else tuple(actions)
        if not supplied:
            raise ValueError("decision node requires at least one action")
        normalized: list[tuple[str, PublicTreeNode]] = []
        seen: set[str] = set()
        for raw_action_id, child in supplied:
            action_id = str(raw_action_id)
            if not action_id:
                raise ValueError("decision action IDs must be non-empty")
            if action_id in seen:
                raise ValueError(f"duplicate decision action ID {action_id!r}")
            seen.add(action_id)
            normalized.append((action_id, child))
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "actions", tuple(sorted(normalized)))

    @property
    def infoset_key(self) -> InfoSetKey:
        return self.state.infoset_key

    @property
    def actor(self) -> Actor:
        return self.state.infoset_key.actor

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action_id for action_id, _child in self.actions)


PublicTreeNode: TypeAlias = (
    PublicTreeTerminalNode | PublicTreeChanceNode | PublicTreeDecisionNode
)


@dataclass(frozen=True)
class _ValidatedTree:
    root: PublicTreeNode
    infoset_actions: Mapping[InfoSetKey, tuple[str, ...]]
    infoset_actors: Mapping[InfoSetKey, Actor]
    terminal_utility_sources: frozenset[str]


def _validate_tree(root: PublicTreeNode) -> _ValidatedTree:
    """Validate finiteness and the shared-information-set action contract."""
    infoset_actions: dict[InfoSetKey, tuple[str, ...]] = {}
    infoset_actors: dict[InfoSetKey, Actor] = {}
    active: set[int] = set()
    completed: set[int] = set()
    terminal_sources: set[str] = set()

    def visit(node: PublicTreeNode) -> None:
        if not isinstance(
            node,
            (PublicTreeTerminalNode, PublicTreeChanceNode, PublicTreeDecisionNode),
        ):
            raise TypeError(f"unsupported public-tree node type: {type(node).__name__}")
        node_id = id(node)
        if node_id in active:
            raise ValueError("public tree contains a cycle")
        if node_id in completed:
            return
        active.add(node_id)
        if isinstance(node, PublicTreeTerminalNode):
            terminal_sources.add(node.utility_source)
        elif isinstance(node, PublicTreeChanceNode):
            for branch in node.branches:
                visit(branch.child)
        elif isinstance(node, PublicTreeDecisionNode):
            key = node.infoset_key
            if not isinstance(key, InfoSetKey):
                raise TypeError("policy keys must be InfoSetKey instances")
            # canonical_json contains a second defense against private-world
            # fields entering the policy identity.
            key.canonical_json()
            prior_actions = infoset_actions.get(key)
            if prior_actions is not None and prior_actions != node.action_ids:
                raise ValueError(
                    "shared InfoSetKey action-set mismatch: "
                    f"expected {prior_actions}, got {node.action_ids}"
                )
            infoset_actions[key] = node.action_ids
            infoset_actors[key] = node.actor
            for _action_id, child in node.actions:
                visit(child)
        active.remove(node_id)
        completed.add(node_id)

    visit(root)
    if not infoset_actions:
        raise ValueError("public tree must contain at least one decision node")
    return _ValidatedTree(
        root=root,
        infoset_actions=infoset_actions,
        infoset_actors=infoset_actors,
        terminal_utility_sources=frozenset(terminal_sources),
    )


def public_tree_t4_decision_from_physical_result(
    result: Mapping[str, Any],
    state: PublicTreeDecisionState,
) -> PublicTreeDecisionNode:
    """Compile every conditioned T4 BB action into one recursive-tree leaf node."""
    if state.infoset_key.phase != "t4_first":
        raise ValueError("conditioned physical T4 vectors require a t4_first decision state")
    vector = particle_action_vector_from_physical_t4_result(
        result,
        particle=state.particle,
        infoset_key=state.infoset_key,
    )
    return PublicTreeDecisionNode(
        state,
        {
            action_id: PublicTreeTerminalNode(
                utility,
                terminal_id=f"{vector.particle_id}:{action_id}",
                utility_source="rust_exact_physical_t4_action_vector",
            )
            for action_id, utility in vector.action_utilities.items()
        },
    )


def _stable_infosets(
    validated: _ValidatedTree,
    actor: Actor | None = None,
) -> tuple[InfoSetKey, ...]:
    keys = (
        key
        for key in validated.infoset_actions
        if actor is None or validated.infoset_actors[key] == actor
    )
    return tuple(sorted(keys, key=lambda key: (key.digest(), key.canonical_json())))


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


def _validate_strategy_entry(
    key: InfoSetKey,
    action_ids: Sequence[str],
    strategy: Mapping[str, float],
) -> None:
    if set(strategy) != set(action_ids):
        raise ValueError(
            f"strategy action set does not match infoset {key.digest()[:12]}"
        )
    probabilities = [float(strategy[action_id]) for action_id in action_ids]
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("strategy probabilities must be finite and non-negative")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("strategy probabilities must sum to 1")


def _validate_profile(
    validated: _ValidatedTree,
    profile: Mapping[InfoSetKey, Mapping[str, float]],
    *,
    required_actor: Actor | None = None,
) -> None:
    required = set(_stable_infosets(validated, required_actor))
    missing = required - set(profile)
    if missing:
        first = min(missing, key=lambda key: key.digest())
        raise ValueError(f"strategy profile is missing infoset {first.digest()[:12]}")
    for key in required:
        _validate_strategy_entry(key, validated.infoset_actions[key], profile[key])


def _evaluate_profile_validated(
    node: PublicTreeNode,
    profile: Mapping[InfoSetKey, Mapping[str, float]],
    pure_overrides: Mapping[InfoSetKey, str] | None = None,
) -> float:
    overrides = pure_overrides or {}
    if isinstance(node, PublicTreeTerminalNode):
        return float(node.utility_bb)
    if isinstance(node, PublicTreeChanceNode):
        return math.fsum(
            float(branch.probability)
            * _evaluate_profile_validated(branch.child, profile, overrides)
            for branch in node.branches
        )
    key = node.infoset_key
    selected = overrides.get(key)
    if selected is not None:
        for action_id, child in node.actions:
            if action_id == selected:
                return _evaluate_profile_validated(child, profile, overrides)
        raise AssertionError("pure override refers to a non-existent action")
    strategy = profile[key]
    return math.fsum(
        float(strategy[action_id])
        * _evaluate_profile_validated(child, profile, overrides)
        for action_id, child in node.actions
    )


def evaluate_public_tree_profile(
    root: PublicTreeNode,
    profile: Mapping[InfoSetKey, Mapping[str, float]],
) -> float:
    """Return expected BB utility of a complete behavioral profile."""
    validated = _validate_tree(root)
    _validate_profile(validated, profile)
    return _evaluate_profile_validated(root, profile)


@dataclass(frozen=True)
class PublicTreeBestResponse:
    """Exhaustive pure-infoset response against the fixed opponent profile.

    ``exact_scope`` refers to exhaustive policy enumeration on the supplied
    finite tree.  Terminal utility evaluation remains Python ``float``
    arithmetic; callers must not reinterpret this as exact-rational OFC
    scoring.
    """

    actor: Actor
    value_bb: float
    policy: Mapping[InfoSetKey, str]
    pure_profiles_evaluated: int
    exact_scope: str
    numeric_arithmetic: str


def _exact_best_response_validated(
    validated: _ValidatedTree,
    *,
    actor: Actor,
    opponent_strategy: Mapping[InfoSetKey, Mapping[str, float]],
    max_pure_profiles: int,
) -> PublicTreeBestResponse:
    if actor not in ("bb", "btn"):
        raise ValueError("best-response actor must be 'bb' or 'btn'")
    if isinstance(max_pure_profiles, bool) or int(max_pure_profiles) <= 0:
        raise ValueError("max_pure_profiles must be a positive integer")
    _validate_profile(validated, opponent_strategy, required_actor="btn" if actor == "bb" else "bb")

    keys = _stable_infosets(validated, actor)
    action_spaces = [validated.infoset_actions[key] for key in keys]
    profile_count = math.prod(len(actions) for actions in action_spaces)
    if profile_count > int(max_pure_profiles):
        raise ValueError(
            "exact infoset best response exceeds max_pure_profiles: "
            f"{profile_count} > {int(max_pure_profiles)}"
        )

    best_value: float | None = None
    best_policy: dict[InfoSetKey, str] | None = None
    for choices in itertools.product(*action_spaces):
        policy = dict(zip(keys, choices))
        value = _evaluate_profile_validated(
            validated.root,
            opponent_strategy,
            pure_overrides=policy,
        )
        better = (
            best_value is None
            or (actor == "bb" and value > best_value)
            or (actor == "btn" and value < best_value)
        )
        if better:
            best_value = value
            best_policy = policy
    if best_value is None or best_policy is None:
        raise AssertionError("best response enumerated no pure policy")
    return PublicTreeBestResponse(
        actor=actor,
        value_bb=best_value,
        policy=best_policy,
        pure_profiles_evaluated=profile_count,
        exact_scope="exhaustive_pure_infoset_policy_enumeration",
        numeric_arithmetic="float64",
    )


def exact_public_tree_best_response(
    root: PublicTreeNode,
    *,
    actor: Actor,
    opponent_strategy: Mapping[InfoSetKey, Mapping[str, float]],
    max_pure_profiles: int = 1_000_000,
) -> PublicTreeBestResponse:
    """Enumerate all actor pure infoset policies and return the exact BR.

    Enumeration is intentionally bounded: this reference is for reduced
    trees.  Choosing one action per ``InfoSetKey`` is the critical distinction
    from an illegal per-particle/PIMC response.
    """
    return _exact_best_response_validated(
        _validate_tree(root),
        actor=actor,
        opponent_strategy=opponent_strategy,
        max_pure_profiles=max_pure_profiles,
    )


@dataclass(frozen=True)
class PublicTreeProfileMetrics:
    value_bb: float
    bb_best_response: float
    btn_best_response: float
    nash_conv: float
    exploitability: float
    bb_best_response_policy: Mapping[InfoSetKey, str]
    btn_best_response_policy: Mapping[InfoSetKey, str]


def _profile_metrics_validated(
    validated: _ValidatedTree,
    profile: Mapping[InfoSetKey, Mapping[str, float]],
    *,
    max_pure_profiles: int,
) -> PublicTreeProfileMetrics:
    _validate_profile(validated, profile)
    value = _evaluate_profile_validated(validated.root, profile)
    bb_response = _exact_best_response_validated(
        validated,
        actor="bb",
        opponent_strategy=profile,
        max_pure_profiles=max_pure_profiles,
    )
    btn_response = _exact_best_response_validated(
        validated,
        actor="btn",
        opponent_strategy=profile,
        max_pure_profiles=max_pure_profiles,
    )
    nash_conv = max(0.0, bb_response.value_bb - btn_response.value_bb)
    return PublicTreeProfileMetrics(
        value_bb=value,
        bb_best_response=bb_response.value_bb,
        btn_best_response=btn_response.value_bb,
        nash_conv=nash_conv,
        exploitability=nash_conv / 2.0,
        bb_best_response_policy=bb_response.policy,
        btn_best_response_policy=btn_response.policy,
    )


def public_tree_profile_metrics(
    root: PublicTreeNode,
    profile: Mapping[InfoSetKey, Mapping[str, float]],
    *,
    max_pure_profiles: int = 1_000_000,
) -> PublicTreeProfileMetrics:
    """Return profile value, exact infoset-aware BRs, and NashConv."""
    validated = _validate_tree(root)
    return _profile_metrics_validated(
        validated,
        profile,
        max_pure_profiles=max_pure_profiles,
    )


@dataclass(frozen=True)
class StrategyFusionDiagnostic:
    actor: Actor
    infoset_aware_value_bb: float
    illegal_per_history_value_bb: float
    strategy_fusion_advantage: float


def _perfect_information_response_value(
    node: PublicTreeNode,
    *,
    actor: Actor,
    opponent_strategy: Mapping[InfoSetKey, Mapping[str, float]],
) -> float:
    if isinstance(node, PublicTreeTerminalNode):
        return float(node.utility_bb)
    if isinstance(node, PublicTreeChanceNode):
        return math.fsum(
            float(branch.probability)
            * _perfect_information_response_value(
                branch.child,
                actor=actor,
                opponent_strategy=opponent_strategy,
            )
            for branch in node.branches
        )
    child_values = {
        action_id: _perfect_information_response_value(
            child,
            actor=actor,
            opponent_strategy=opponent_strategy,
        )
        for action_id, child in node.actions
    }
    if node.actor == actor:
        extreme = max if actor == "bb" else min
        return extreme(child_values.values())
    return math.fsum(
        float(opponent_strategy[node.infoset_key][action_id]) * child_values[action_id]
        for action_id in node.action_ids
    )


def public_tree_strategy_fusion_diagnostic(
    root: PublicTreeNode,
    *,
    actor: Actor,
    opponent_strategy: Mapping[InfoSetKey, Mapping[str, float]],
    max_pure_profiles: int = 1_000_000,
) -> StrategyFusionDiagnostic:
    """Compare legal shared-infoset BR with illegal per-history optimization."""
    validated = _validate_tree(root)
    response = _exact_best_response_validated(
        validated,
        actor=actor,
        opponent_strategy=opponent_strategy,
        max_pure_profiles=max_pure_profiles,
    )
    illegal = _perfect_information_response_value(
        root,
        actor=actor,
        opponent_strategy=opponent_strategy,
    )
    advantage = (
        illegal - response.value_bb
        if actor == "bb"
        else response.value_bb - illegal
    )
    if advantage < -1e-12:
        raise AssertionError("per-history response cannot be worse than a shared-infoset BR")
    return StrategyFusionDiagnostic(
        actor=actor,
        infoset_aware_value_bb=response.value_bb,
        illegal_per_history_value_bb=illegal,
        strategy_fusion_advantage=max(0.0, advantage),
    )


@dataclass(frozen=True)
class RecursivePublicTreeCfrResult:
    iterations: int
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    current_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    cumulative_regret_plus: Mapping[InfoSetKey, Mapping[str, float]]
    metrics: PublicTreeProfileMetrics
    exploitability_trace: tuple[tuple[int, float], ...]
    metadata: Mapping[str, Any]


def solve_recursive_public_tree_cfr_plus(
    root: PublicTreeNode,
    *,
    iterations: int,
    linear_averaging: bool = True,
    checkpoints: Sequence[int] = (),
    max_pure_profiles: int = 1_000_000,
) -> RecursivePublicTreeCfrResult:
    """Run deterministic synchronous CFR+ on a finite reduced public tree."""
    if isinstance(iterations, bool) or int(iterations) <= 0:
        raise ValueError("iterations must be a positive integer")
    iterations = int(iterations)
    validated = _validate_tree(root)
    keys = _stable_infosets(validated)
    regrets = {
        key: {action_id: 0.0 for action_id in validated.infoset_actions[key]}
        for key in keys
    }
    strategy_sum = {
        key: {action_id: 0.0 for action_id in validated.infoset_actions[key]}
        for key in keys
    }
    checkpoint_set = {
        int(point) for point in checkpoints if 0 < int(point) <= iterations
    }
    checkpoint_set.add(iterations)
    trace: list[tuple[int, float]] = []
    final_checkpoint_metrics: PublicTreeProfileMetrics | None = None

    for iteration in range(1, iterations + 1):
        # Both players use this immutable iteration snapshot.  Regret updates
        # are accumulated over all chance/physical histories and applied only
        # after the traversal, hence updates are synchronous and order-free.
        strategy = {
            key: _regret_matching_plus(
                regrets[key],
                validated.infoset_actions[key],
            )
            for key in keys
        }
        deltas = {
            key: {action_id: 0.0 for action_id in validated.infoset_actions[key]}
            for key in keys
        }
        own_reach_by_infoset: dict[InfoSetKey, float] = {}

        def traverse(
            node: PublicTreeNode,
            *,
            reach_bb: float,
            reach_btn: float,
            chance_reach: Fraction,
        ) -> float:
            if isinstance(node, PublicTreeTerminalNode):
                return float(node.utility_bb)
            if isinstance(node, PublicTreeChanceNode):
                return math.fsum(
                    float(branch.probability)
                    * traverse(
                        branch.child,
                        reach_bb=reach_bb,
                        reach_btn=reach_btn,
                        chance_reach=chance_reach * branch.probability,
                    )
                    for branch in node.branches
                )

            key = node.infoset_key
            sigma = strategy[key]
            actor_reach = reach_bb if node.actor == "bb" else reach_btn
            prior_reach = own_reach_by_infoset.get(key)
            if prior_reach is None:
                own_reach_by_infoset[key] = actor_reach
            elif not math.isclose(prior_reach, actor_reach, abs_tol=1e-12):
                raise ValueError(
                    "shared InfoSetKey violates perfect recall: actor reach differs "
                    "across physical histories"
                )

            action_values: dict[str, float] = {}
            for action_id, child in node.actions:
                if node.actor == "bb":
                    action_values[action_id] = traverse(
                        child,
                        reach_bb=reach_bb * sigma[action_id],
                        reach_btn=reach_btn,
                        chance_reach=chance_reach,
                    )
                else:
                    action_values[action_id] = traverse(
                        child,
                        reach_bb=reach_bb,
                        reach_btn=reach_btn * sigma[action_id],
                        chance_reach=chance_reach,
                    )
            node_value = math.fsum(
                sigma[action_id] * action_values[action_id]
                for action_id in node.action_ids
            )
            counterfactual_reach = float(chance_reach) * (
                reach_btn if node.actor == "bb" else reach_bb
            )
            sign = 1.0 if node.actor == "bb" else -1.0
            for action_id in node.action_ids:
                deltas[key][action_id] += (
                    counterfactual_reach
                    * sign
                    * (action_values[action_id] - node_value)
                )
            return node_value

        traverse(
            root,
            reach_bb=1.0,
            reach_btn=1.0,
            chance_reach=Fraction(1, 1),
        )

        average_weight = float(iteration if linear_averaging else 1)
        for key in keys:
            own_reach = own_reach_by_infoset.get(key, 0.0)
            for action_id in validated.infoset_actions[key]:
                strategy_sum[key][action_id] += (
                    average_weight * own_reach * strategy[key][action_id]
                )
                regrets[key][action_id] = max(
                    0.0,
                    regrets[key][action_id] + deltas[key][action_id],
                )

        if iteration in checkpoint_set:
            average = {
                key: _normalized_average(
                    strategy_sum[key],
                    validated.infoset_actions[key],
                )
                for key in keys
            }
            metrics = _profile_metrics_validated(
                validated,
                average,
                max_pure_profiles=max_pure_profiles,
            )
            trace.append((iteration, metrics.exploitability))
            if iteration == iterations:
                final_checkpoint_metrics = metrics

    average_strategy = {
        key: _normalized_average(
            strategy_sum[key],
            validated.infoset_actions[key],
        )
        for key in keys
    }
    current_strategy = {
        key: _regret_matching_plus(regrets[key], validated.infoset_actions[key])
        for key in keys
    }
    # ``iterations`` is always a checkpoint.  Reuse its exact best-response
    # calculation instead of enumerating the same pure infoset policies a
    # second time after the loop.  This matters for the canonical BB reduced
    # fixtures, where exhaustive BR enumeration dominates runtime.
    if final_checkpoint_metrics is None:  # pragma: no cover - defensive
        raise RuntimeError("final CFR checkpoint metrics were not produced")
    metrics = final_checkpoint_metrics
    if not all(math.isfinite(value) for _iteration, value in trace):
        raise RuntimeError("recursive CFR produced a non-finite exploitability trace")
    return RecursivePublicTreeCfrResult(
        iterations=iterations,
        average_strategy=average_strategy,
        current_strategy=current_strategy,
        cumulative_regret_plus={
            key: dict(regrets[key]) for key in keys
        },
        metrics=metrics,
        exploitability_trace=tuple(trace),
        metadata={
            "method": "reduced_recursive_public_tree_cfr_plus",
            "tree_scope": "finite_explicit_reduced_non_full_card",
            "recursive": True,
            "synchronous_updates": True,
            "infoset_aware_exact_best_response": True,
            "best_response_exact_scope": "exhaustive_pure_infoset_policy_enumeration",
            "numeric_utility_arithmetic": "float64",
            "chance_path_arithmetic": "exact_fraction_until_regret_multiply",
            "chance_source": "explicit_tree_branches_only",
            "joint_particle_weight_used": False,
            "terminal_utility_sources": sorted(validated.terminal_utility_sources),
            "strategy_fusion": False,
            "equilibrium_approx": True,
            "full_card": False,
            "hu_exact": False,
            "runtime_integrated": False,
            "rust_leaf_integrated": (
                "rust_exact_physical_t4_action_vector"
                in validated.terminal_utility_sources
            ),
            "position_contract_version": POSITION_CONTRACT_VERSION,
        },
    )
