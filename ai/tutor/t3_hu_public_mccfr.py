"""Tabular external-sampling MCCFR for finite public-information trees.

This module is the sampled counterpart to
:mod:`ai.tutor.t3_hu_public_tree_cfr`.  The exact recursive CFR+ solver stays
the oracle; this module consumes the same explicit reduced-tree node contract
through :class:`ExplicitPublicTreeAdapter` and performs one external-sampling
episode for BB followed by one for BTN in every iteration.

The sampling contract is deliberately strict:

* policy/regret tables are keyed only by :class:`InfoSetKey`;
* legal action IDs are the lexically stable IDs stored on the explicit tree;
* chance and the non-traversing player are sampled, while every traverser
  action is expanded;
* one sampled opponent action is cached per ``InfoSetKey`` during an episode,
  representing one external pure strategy rather than per-world responses;
* the root posterior is sampled exactly once per episode when the root is a
  chance node;
* explicit chance-branch probabilities are the only range/chance mass source.
  ``JointParticle.weight`` is provenance and is never read or multiplied.

The implementation is intentionally tabular and bounded to an explicitly
supplied finite tree.  It is a correctness bridge for a future generative
full-card adapter, not a promoted full-card HU policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import Actor, InfoSetKey
from ai.tutor.t3_hu_public_tree_cfr import (
    PublicTreeChanceBranch as CfrChanceBranch,
    PublicTreeChanceNode as CfrChanceNode,
    PublicTreeDecisionNode as CfrDecisionNode,
    PublicTreeNode,
    PublicTreeProfileMetrics,
    PublicTreeTerminalNode as CfrTerminalNode,
    RecursivePublicTreeCfrResult,
    public_tree_profile_metrics,
    solve_recursive_public_tree_cfr_plus,
)


CHECKPOINT_FORMAT = "external_sampling_public_mccfr_checkpoint_v1"
TREE_MANIFEST_FORMAT = "explicit_public_mccfr_tree_manifest_v1"
SOLVER_STATE_FORMAT = "external_sampling_public_mccfr_state_v1"
RNG_ALGORITHM = "python_random_mt19937"


class PublicMccfrCheckpointError(ValueError):
    """A checkpoint failed content, tree, configuration, or state validation."""


@dataclass(frozen=True)
class _CompiledExplicitTree:
    root: PublicTreeNode
    infoset_actions: Mapping[InfoSetKey, tuple[str, ...]]
    infoset_actors: Mapping[InfoSetKey, Actor]
    stable_infosets: tuple[InfoSetKey, ...]
    terminal_utility_sources: frozenset[str]
    physical_decision_nodes: int


class ExplicitPublicTreeAdapter:
    """Read-only adapter around the exact solver's finite tree node API.

    Sampling/training code below talks to this adapter rather than reaching
    through physical OFC states.  A future generative adapter can preserve the
    same terminal/chance/decision boundary while creating children lazily.
    """

    def __init__(self, root: PublicTreeNode) -> None:
        self.root = root

    @staticmethod
    def is_terminal(node: PublicTreeNode) -> bool:
        return isinstance(node, CfrTerminalNode)

    @staticmethod
    def terminal_utility_bb(node: PublicTreeNode) -> float:
        if not isinstance(node, CfrTerminalNode):
            raise TypeError("terminal utility requested for a non-terminal node")
        return float(node.utility_bb)

    @staticmethod
    def is_chance(node: PublicTreeNode) -> bool:
        return isinstance(node, CfrChanceNode)

    @staticmethod
    def chance_branches(node: PublicTreeNode) -> tuple[CfrChanceBranch, ...]:
        if not isinstance(node, CfrChanceNode):
            raise TypeError("chance branches requested for a non-chance node")
        return node.branches

    @staticmethod
    def is_decision(node: PublicTreeNode) -> bool:
        return isinstance(node, CfrDecisionNode)

    @staticmethod
    def decision_key(node: PublicTreeNode) -> InfoSetKey:
        if not isinstance(node, CfrDecisionNode):
            raise TypeError("information key requested for a non-decision node")
        return node.infoset_key

    @staticmethod
    def decision_actor(node: PublicTreeNode) -> Actor:
        if not isinstance(node, CfrDecisionNode):
            raise TypeError("actor requested for a non-decision node")
        return node.actor

    @staticmethod
    def decision_actions(
        node: PublicTreeNode,
    ) -> tuple[tuple[str, PublicTreeNode], ...]:
        if not isinstance(node, CfrDecisionNode):
            raise TypeError("actions requested for a non-decision node")
        return node.actions

    def compile(self) -> _CompiledExplicitTree:
        """Validate finiteness and the shared-infoset action contract."""
        infoset_actions: dict[InfoSetKey, tuple[str, ...]] = {}
        infoset_actors: dict[InfoSetKey, Actor] = {}
        terminal_sources: set[str] = set()
        active: set[int] = set()
        completed: set[int] = set()
        physical_decisions = 0

        def visit(node: PublicTreeNode) -> None:
            nonlocal physical_decisions
            if not isinstance(
                node,
                (CfrTerminalNode, CfrChanceNode, CfrDecisionNode),
            ):
                raise TypeError(
                    f"unsupported explicit public-tree node: {type(node).__name__}"
                )
            node_identity = id(node)
            if node_identity in active:
                raise ValueError("public MCCFR tree contains a cycle")
            if node_identity in completed:
                return
            active.add(node_identity)
            if isinstance(node, CfrTerminalNode):
                terminal_sources.add(node.utility_source)
            elif isinstance(node, CfrChanceNode):
                for branch in node.branches:
                    visit(branch.child)
            else:
                physical_decisions += 1
                key = node.infoset_key
                if not isinstance(key, InfoSetKey):
                    raise TypeError("public MCCFR policy keys must be InfoSetKey")
                # Also runs the key's forbidden-field serialization defense.
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
            active.remove(node_identity)
            completed.add(node_identity)

        visit(self.root)
        if not infoset_actions:
            raise ValueError("public MCCFR tree requires at least one decision node")
        stable_infosets = tuple(
            sorted(
                infoset_actions,
                key=lambda key: (key.digest(), key.canonical_json()),
            )
        )
        return _CompiledExplicitTree(
            root=self.root,
            infoset_actions=infoset_actions,
            infoset_actors=infoset_actors,
            stable_infosets=stable_infosets,
            terminal_utility_sources=frozenset(terminal_sources),
            physical_decision_nodes=physical_decisions,
        )


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
        raise PublicMccfrCheckpointError(
            "MCCFR checkpoint contains non-canonical JSON data"
        ) from exc
    return serialized.encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_number_text(value: Any) -> str:
    return f"{value.numerator}/{value.denominator}"


def _tree_node_manifest(node: PublicTreeNode) -> dict[str, Any]:
    if isinstance(node, CfrTerminalNode):
        return {
            "node_type": "terminal",
            "terminal_id": node.terminal_id,
            "utility_bb": _exact_number_text(node.utility_bb),
            "utility_source": node.utility_source,
        }
    if isinstance(node, CfrChanceNode):
        return {
            "node_type": "chance",
            "branches": [
                {
                    "outcome_id": branch.outcome_id,
                    "probability": _exact_number_text(branch.probability),
                    "child": _tree_node_manifest(branch.child),
                }
                for branch in node.branches
            ],
        }
    if isinstance(node, CfrDecisionNode):
        return {
            "node_type": "decision",
            "actor": node.actor,
            "infoset_canonical_json": node.infoset_key.canonical_json(),
            "infoset_sha256": node.infoset_key.digest(),
            "actions": [
                {
                    "action_id": action_id,
                    "child": _tree_node_manifest(child),
                }
                for action_id, child in node.actions
            ],
        }
    raise TypeError(f"unsupported explicit public-tree node: {type(node).__name__}")


def _tree_manifest(compiled: _CompiledExplicitTree) -> dict[str, Any]:
    return {
        "format": TREE_MANIFEST_FORMAT,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_decision_nodes": compiled.physical_decision_nodes,
        "unique_infosets": len(compiled.stable_infosets),
        "infosets": [
            {
                "actor": compiled.infoset_actors[key],
                "canonical_json": key.canonical_json(),
                "sha256": key.digest(),
                "stable_action_ids": list(compiled.infoset_actions[key]),
            }
            for key in compiled.stable_infosets
        ],
        "root": _tree_node_manifest(compiled.root),
    }


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


def _strategy_from_regrets(
    compiled: _CompiledExplicitTree,
    regrets: Mapping[InfoSetKey, Mapping[str, float]],
) -> dict[InfoSetKey, dict[str, float]]:
    return {
        key: _regret_matching_plus(regrets[key], compiled.infoset_actions[key])
        for key in compiled.stable_infosets
    }


def _average_from_sum(
    compiled: _CompiledExplicitTree,
    strategy_sum: Mapping[InfoSetKey, Mapping[str, float]],
) -> dict[InfoSetKey, dict[str, float]]:
    return {
        key: _normalized_average(strategy_sum[key], compiled.infoset_actions[key])
        for key in compiled.stable_infosets
    }


def _validate_profile(
    compiled: _CompiledExplicitTree,
    profile: Mapping[InfoSetKey, Mapping[str, float]],
) -> None:
    missing = set(compiled.stable_infosets) - set(profile)
    if missing:
        first = min(missing, key=lambda key: key.digest())
        raise ValueError(f"strategy profile is missing infoset {first.digest()[:12]}")
    for key in compiled.stable_infosets:
        action_ids = compiled.infoset_actions[key]
        entry = profile[key]
        if set(entry) != set(action_ids):
            raise ValueError(
                "strategy action set does not match infoset "
                f"{key.digest()[:12]}"
            )
        probabilities = [float(entry[action_id]) for action_id in action_ids]
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("strategy probabilities must be finite and non-negative")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("strategy probabilities must sum to 1")


def uniform_public_tree_strategy(
    root: PublicTreeNode,
) -> Mapping[InfoSetKey, Mapping[str, float]]:
    """Return a complete stable uniform profile for an explicit tree."""
    compiled = ExplicitPublicTreeAdapter(root).compile()
    return {
        key: {
            action_id: 1.0 / len(compiled.infoset_actions[key])
            for action_id in compiled.infoset_actions[key]
        }
        for key in compiled.stable_infosets
    }


@dataclass
class _MutableSamplingStats:
    traversals: int = 0
    bb_traversals: int = 0
    btn_traversals: int = 0
    nodes_touched: int = 0
    terminal_visits: int = 0
    chance_action_samples: int = 0
    chance_cache_hits: int = 0
    opponent_action_samples: int = 0
    opponent_cache_hits: int = 0
    root_posterior_samples: int = 0
    root_outcome_counts: dict[str, int] = field(default_factory=dict)

    def begin(self, traverser: Actor) -> None:
        self.traversals += 1
        if traverser == "bb":
            self.bb_traversals += 1
        else:
            self.btn_traversals += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "traversals": self.traversals,
            "traversals_by_actor": {
                "bb": self.bb_traversals,
                "btn": self.btn_traversals,
            },
            "nodes_touched": self.nodes_touched,
            "terminal_visits": self.terminal_visits,
            "chance_action_samples": self.chance_action_samples,
            "chance_cache_hits": self.chance_cache_hits,
            "opponent_action_samples": self.opponent_action_samples,
            "opponent_cache_hits": self.opponent_cache_hits,
            "root_posterior_samples": self.root_posterior_samples,
            "root_outcome_counts": dict(sorted(self.root_outcome_counts.items())),
        }

    @classmethod
    def from_snapshot(cls, raw: Any) -> "_MutableSamplingStats":
        if not isinstance(raw, dict) or set(raw) != {
            "traversals",
            "traversals_by_actor",
            "nodes_touched",
            "terminal_visits",
            "chance_action_samples",
            "chance_cache_hits",
            "opponent_action_samples",
            "opponent_cache_hits",
            "root_posterior_samples",
            "root_outcome_counts",
        }:
            raise PublicMccfrCheckpointError(
                "checkpoint sampling_stats schema is invalid"
            )
        by_actor = raw["traversals_by_actor"]
        outcomes = raw["root_outcome_counts"]
        if not isinstance(by_actor, dict) or set(by_actor) != {"bb", "btn"}:
            raise PublicMccfrCheckpointError(
                "checkpoint traversal actor counts are invalid"
            )
        if not isinstance(outcomes, dict) or any(
            not isinstance(outcome_id, str) or not outcome_id
            for outcome_id in outcomes
        ):
            raise PublicMccfrCheckpointError(
                "checkpoint root outcome counts are invalid"
            )

        def count(value: Any, *, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PublicMccfrCheckpointError(
                    f"checkpoint {label} must be a non-negative integer"
                )
            return value

        return cls(
            traversals=count(raw["traversals"], label="traversals"),
            bb_traversals=count(by_actor["bb"], label="bb traversals"),
            btn_traversals=count(by_actor["btn"], label="btn traversals"),
            nodes_touched=count(raw["nodes_touched"], label="nodes touched"),
            terminal_visits=count(
                raw["terminal_visits"], label="terminal visits"
            ),
            chance_action_samples=count(
                raw["chance_action_samples"], label="chance action samples"
            ),
            chance_cache_hits=count(
                raw["chance_cache_hits"], label="chance cache hits"
            ),
            opponent_action_samples=count(
                raw["opponent_action_samples"],
                label="opponent action samples",
            ),
            opponent_cache_hits=count(
                raw["opponent_cache_hits"], label="opponent cache hits"
            ),
            root_posterior_samples=count(
                raw["root_posterior_samples"],
                label="root posterior samples",
            ),
            root_outcome_counts={
                outcome_id: count(value, label=f"root outcome {outcome_id!r}")
                for outcome_id, value in outcomes.items()
            },
        )


@dataclass(frozen=True)
class _RestoredCheckpoint:
    completed_iterations: int
    regrets: dict[InfoSetKey, dict[str, float]]
    strategy_sum: dict[InfoSetKey, dict[str, float]]
    rng_state: tuple[Any, ...]
    stats: _MutableSamplingStats
    checkpoint_sha256: str


def _float_hex(value: float, *, label: str) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PublicMccfrCheckpointError(
            f"checkpoint {label} must be finite"
        )
    return numeric.hex()


def _float_from_hex(raw: Any, *, label: str, non_negative: bool) -> float:
    if not isinstance(raw, str):
        raise PublicMccfrCheckpointError(
            f"checkpoint {label} must be a hexadecimal float string"
        )
    try:
        value = float.fromhex(raw)
    except ValueError as exc:
        raise PublicMccfrCheckpointError(
            f"checkpoint {label} is not a valid hexadecimal float"
        ) from exc
    if not math.isfinite(value) or (non_negative and value < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise PublicMccfrCheckpointError(
            f"checkpoint {label} must be {qualifier}"
        )
    return value


def _encode_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 3:
        raise PublicMccfrCheckpointError("unexpected Python RNG state shape")
    version, internal_state, gaussian = state
    if not isinstance(version, int) or not isinstance(internal_state, tuple):
        raise PublicMccfrCheckpointError("unexpected Python RNG state values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in internal_state):
        raise PublicMccfrCheckpointError("Python RNG internal state is not integer-only")
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
        raise PublicMccfrCheckpointError("checkpoint RNG state schema is invalid")
    if raw["algorithm"] != RNG_ALGORITHM:
        raise PublicMccfrCheckpointError("checkpoint RNG algorithm is incompatible")
    version = raw["version"]
    internal_state = raw["internal_state"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise PublicMccfrCheckpointError("checkpoint RNG version is invalid")
    if not isinstance(internal_state, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in internal_state
    ):
        raise PublicMccfrCheckpointError(
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
        raise PublicMccfrCheckpointError(
            "checkpoint RNG state is not accepted by this Python runtime"
        ) from exc
    return state


def _solver_checkpoint_config() -> dict[str, Any]:
    return {
        "sampling_scheme": "external_sampling",
        "traverser_schedule": "bb_then_btn_each_iteration",
        "regret_matching_plus": True,
        "average_strategy_estimator": "two_player_simple_opponent_node",
        "rng_algorithm": RNG_ALGORITHM,
        "position_contract_version": POSITION_CONTRACT_VERSION,
    }


def _checkpoint_payload(
    *,
    compiled: _CompiledExplicitTree,
    completed_iterations: int,
    seed: int,
    linear_averaging: bool,
    regrets: Mapping[InfoSetKey, Mapping[str, float]],
    strategy_sum: Mapping[InfoSetKey, Mapping[str, float]],
    rng_state: tuple[Any, ...],
    stats: _MutableSamplingStats,
) -> dict[str, Any]:
    manifest = _tree_manifest(compiled)
    return {
        "format": SOLVER_STATE_FORMAT,
        "completed_iterations": completed_iterations,
        "seed": seed,
        "linear_averaging": linear_averaging,
        "solver_config": _solver_checkpoint_config(),
        "tree_manifest": manifest,
        "tree_manifest_sha256": _content_sha256(manifest),
        "tables": [
            {
                "infoset_canonical_json": key.canonical_json(),
                "infoset_sha256": key.digest(),
                "actor": compiled.infoset_actors[key],
                "stable_action_ids": list(compiled.infoset_actions[key]),
                "regret_plus_hex": [
                    _float_hex(
                        regrets[key][action_id],
                        label="cumulative regret",
                    )
                    for action_id in compiled.infoset_actions[key]
                ],
                "strategy_sum_hex": [
                    _float_hex(
                        strategy_sum[key][action_id],
                        label="strategy sum",
                    )
                    for action_id in compiled.infoset_actions[key]
                ],
            }
            for key in compiled.stable_infosets
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
    path: str | os.PathLike[str],
    envelope: Mapping[str, Any],
) -> None:
    target = Path(path)
    if not target.name:
        raise PublicMccfrCheckpointError("checkpoint path must name a file")
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_json_bytes(envelope) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PublicMccfrCheckpointError(
                f"checkpoint JSON contains duplicate key {key!r}"
            )
        value[key] = item
    return value


def _read_checkpoint_envelope(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    try:
        serialized = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicMccfrCheckpointError(
            f"cannot read MCCFR checkpoint: {exc}"
        ) from exc
    try:
        envelope = json.loads(
            serialized,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except PublicMccfrCheckpointError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PublicMccfrCheckpointError(
            "MCCFR checkpoint is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "format",
        "checkpoint_sha256",
        "payload",
    }:
        raise PublicMccfrCheckpointError("checkpoint envelope schema is invalid")
    if envelope["format"] != CHECKPOINT_FORMAT:
        raise PublicMccfrCheckpointError("checkpoint format is incompatible")
    supplied_hash = envelope["checkpoint_sha256"]
    if (
        not isinstance(supplied_hash, str)
        or len(supplied_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_hash)
    ):
        raise PublicMccfrCheckpointError("checkpoint SHA-256 field is invalid")
    calculated_hash = _content_sha256(envelope["payload"])
    if supplied_hash != calculated_hash:
        raise PublicMccfrCheckpointError("checkpoint content SHA-256 mismatch")
    if not isinstance(envelope["payload"], dict):
        raise PublicMccfrCheckpointError("checkpoint payload must be an object")
    return envelope["payload"], supplied_hash


def _restore_checkpoint(
    path: str | os.PathLike[str],
    *,
    compiled: _CompiledExplicitTree,
    seed: int,
    linear_averaging: bool,
) -> _RestoredCheckpoint:
    payload, checkpoint_sha256 = _read_checkpoint_envelope(path)
    expected_payload_keys = {
        "format",
        "completed_iterations",
        "seed",
        "linear_averaging",
        "solver_config",
        "tree_manifest",
        "tree_manifest_sha256",
        "tables",
        "rng_state",
        "sampling_stats",
    }
    if set(payload) != expected_payload_keys:
        raise PublicMccfrCheckpointError("checkpoint payload schema is invalid")
    if payload["format"] != SOLVER_STATE_FORMAT:
        raise PublicMccfrCheckpointError("checkpoint solver state format is incompatible")
    completed_iterations = payload["completed_iterations"]
    if (
        isinstance(completed_iterations, bool)
        or not isinstance(completed_iterations, int)
        or completed_iterations <= 0
    ):
        raise PublicMccfrCheckpointError(
            "checkpoint completed_iterations must be positive"
        )
    checkpoint_seed = payload["seed"]
    if (
        isinstance(checkpoint_seed, bool)
        or not isinstance(checkpoint_seed, int)
        or checkpoint_seed != seed
    ):
        raise PublicMccfrCheckpointError("checkpoint seed does not match requested seed")
    checkpoint_linear = payload["linear_averaging"]
    if (
        not isinstance(checkpoint_linear, bool)
        or checkpoint_linear is not linear_averaging
    ):
        raise PublicMccfrCheckpointError(
            "checkpoint linear_averaging setting does not match"
        )
    if payload["solver_config"] != _solver_checkpoint_config():
        raise PublicMccfrCheckpointError("checkpoint solver configuration is incompatible")

    current_manifest = _tree_manifest(compiled)
    manifest = payload["tree_manifest"]
    manifest_hash = payload["tree_manifest_sha256"]
    if not isinstance(manifest, dict) or not isinstance(manifest_hash, str):
        raise PublicMccfrCheckpointError("checkpoint tree manifest is invalid")
    if _content_sha256(manifest) != manifest_hash:
        raise PublicMccfrCheckpointError("checkpoint tree manifest SHA-256 mismatch")
    if manifest != current_manifest or manifest_hash != _content_sha256(current_manifest):
        raise PublicMccfrCheckpointError(
            "checkpoint tree manifest does not match the current tree"
        )

    tables = payload["tables"]
    if not isinstance(tables, list) or len(tables) != len(compiled.stable_infosets):
        raise PublicMccfrCheckpointError("checkpoint infoset table count is invalid")
    regrets: dict[InfoSetKey, dict[str, float]] = {}
    strategy_sum: dict[InfoSetKey, dict[str, float]] = {}
    table_keys = {
        "infoset_canonical_json",
        "infoset_sha256",
        "actor",
        "stable_action_ids",
        "regret_plus_hex",
        "strategy_sum_hex",
    }
    for index, key in enumerate(compiled.stable_infosets):
        raw_table = tables[index]
        if not isinstance(raw_table, dict) or set(raw_table) != table_keys:
            raise PublicMccfrCheckpointError(
                "checkpoint infoset table schema is invalid"
            )
        action_ids = compiled.infoset_actions[key]
        if raw_table["infoset_canonical_json"] != key.canonical_json():
            raise PublicMccfrCheckpointError(
                "checkpoint InfoSetKey canonical JSON does not match current tree"
            )
        if raw_table["infoset_sha256"] != key.digest():
            raise PublicMccfrCheckpointError(
                "checkpoint InfoSetKey digest does not match canonical JSON"
            )
        if raw_table["actor"] != compiled.infoset_actors[key]:
            raise PublicMccfrCheckpointError("checkpoint infoset actor is incompatible")
        if raw_table["stable_action_ids"] != list(action_ids):
            raise PublicMccfrCheckpointError(
                "checkpoint stable action IDs do not match current tree"
            )
        regret_values = raw_table["regret_plus_hex"]
        strategy_values = raw_table["strategy_sum_hex"]
        if (
            not isinstance(regret_values, list)
            or not isinstance(strategy_values, list)
            or len(regret_values) != len(action_ids)
            or len(strategy_values) != len(action_ids)
        ):
            raise PublicMccfrCheckpointError(
                "checkpoint action-vector length is incompatible"
            )
        regrets[key] = {
            action_id: _float_from_hex(
                regret_values[action_index],
                label="cumulative regret",
                non_negative=True,
            )
            for action_index, action_id in enumerate(action_ids)
        }
        strategy_sum[key] = {
            action_id: _float_from_hex(
                strategy_values[action_index],
                label="strategy sum",
                non_negative=True,
            )
            for action_index, action_id in enumerate(action_ids)
        }

    rng_state = _decode_rng_state(payload["rng_state"])
    stats = _MutableSamplingStats.from_snapshot(payload["sampling_stats"])
    expected_traversals = 2 * completed_iterations
    if (
        stats.traversals != expected_traversals
        or stats.bb_traversals != completed_iterations
        or stats.btn_traversals != completed_iterations
    ):
        raise PublicMccfrCheckpointError(
            "checkpoint traversal counts do not match completed iterations"
        )
    if isinstance(compiled.root, CfrChanceNode):
        expected_root_ids = {
            branch.outcome_id for branch in compiled.root.branches
        }
        if (
            stats.root_posterior_samples != expected_traversals
            or sum(stats.root_outcome_counts.values()) != expected_traversals
            or not set(stats.root_outcome_counts).issubset(expected_root_ids)
        ):
            raise PublicMccfrCheckpointError(
                "checkpoint root posterior counts are incompatible"
            )
    elif stats.root_posterior_samples or stats.root_outcome_counts:
        raise PublicMccfrCheckpointError(
            "checkpoint records a root posterior for a decision root"
        )
    return _RestoredCheckpoint(
        completed_iterations=completed_iterations,
        regrets=regrets,
        strategy_sum=strategy_sum,
        rng_state=rng_state,
        stats=stats,
        checkpoint_sha256=checkpoint_sha256,
    )


def _sample_chance_branch(
    branches: Sequence[CfrChanceBranch],
    rng: random.Random,
) -> CfrChanceBranch:
    # Sampling exact Fraction mass avoids an accidental second float-based
    # probability model at the root posterior.
    denominator = 1
    for branch in branches:
        denominator = math.lcm(denominator, branch.probability.denominator)
    weights = [
        branch.probability.numerator
        * (denominator // branch.probability.denominator)
        for branch in branches
    ]
    total = sum(weights)
    if total != denominator:
        raise AssertionError("validated chance mass no longer sums exactly to one")
    draw = rng.randrange(total)
    cumulative = 0
    for branch, weight in zip(branches, weights):
        cumulative += weight
        if draw < cumulative:
            return branch
    raise AssertionError("exact chance sampler failed to select a branch")


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
    if last_positive is None:  # pragma: no cover - profile validation defense
        raise AssertionError("policy sampler received no positive action")
    return last_positive


def _external_traverse(
    adapter: ExplicitPublicTreeAdapter,
    node: PublicTreeNode,
    *,
    traverser: Actor,
    strategy: Mapping[InfoSetKey, Mapping[str, float]],
    regret_delta: dict[InfoSetKey, dict[str, float]],
    rng: random.Random,
    chance_choices: dict[int, str],
    opponent_choices: dict[InfoSetKey, str],
    average_updated: set[InfoSetKey],
    strategy_sum: dict[InfoSetKey, dict[str, float]] | None,
    average_weight: float,
    root_chance_identity: int | None,
    stats: _MutableSamplingStats,
) -> float:
    """Return sampled BB utility and aggregate one traverser's raw deltas."""
    stats.nodes_touched += 1
    if adapter.is_terminal(node):
        stats.terminal_visits += 1
        return adapter.terminal_utility_bb(node)

    if adapter.is_chance(node):
        node_identity = id(node)
        branches = adapter.chance_branches(node)
        selected_id = chance_choices.get(node_identity)
        if selected_id is None:
            selected = _sample_chance_branch(branches, rng)
            selected_id = selected.outcome_id
            chance_choices[node_identity] = selected_id
            stats.chance_action_samples += 1
            if root_chance_identity == node_identity:
                stats.root_posterior_samples += 1
                stats.root_outcome_counts[selected_id] = (
                    stats.root_outcome_counts.get(selected_id, 0) + 1
                )
        else:
            stats.chance_cache_hits += 1
            selected = next(
                branch for branch in branches if branch.outcome_id == selected_id
            )
        return _external_traverse(
            adapter,
            selected.child,
            traverser=traverser,
            strategy=strategy,
            regret_delta=regret_delta,
            rng=rng,
            chance_choices=chance_choices,
            opponent_choices=opponent_choices,
            average_updated=average_updated,
            strategy_sum=strategy_sum,
            average_weight=average_weight,
            root_chance_identity=root_chance_identity,
            stats=stats,
        )

    if not adapter.is_decision(node):  # pragma: no cover - adapter defense
        raise TypeError(f"adapter returned unknown node type {type(node).__name__}")
    key = adapter.decision_key(node)
    actor = adapter.decision_actor(node)
    actions = adapter.decision_actions(node)
    action_ids = tuple(action_id for action_id, _child in actions)
    sigma = strategy[key]

    if actor == traverser:
        action_values: dict[str, float] = {}
        for action_id, child in actions:
            action_values[action_id] = _external_traverse(
                adapter,
                child,
                traverser=traverser,
                strategy=strategy,
                regret_delta=regret_delta,
                rng=rng,
                chance_choices=chance_choices,
                opponent_choices=opponent_choices,
                average_updated=average_updated,
                strategy_sum=strategy_sum,
                average_weight=average_weight,
                root_chance_identity=root_chance_identity,
                stats=stats,
            )
        node_value = math.fsum(
            sigma[action_id] * action_values[action_id]
            for action_id in action_ids
        )
        sign = 1.0 if traverser == "bb" else -1.0
        for action_id in action_ids:
            regret_delta[key][action_id] += (
                sign * (action_values[action_id] - node_value)
            )
        return node_value

    # Standard two-player external-sampling average: the other player's own
    # earlier actions are sampled on this pass, so visitation supplies that
    # player's own reach.  One update per key prevents physical-world count
    # from becoming an additional weight.
    if strategy_sum is not None and key not in average_updated:
        for action_id in action_ids:
            strategy_sum[key][action_id] += (
                average_weight * sigma[action_id]
            )
        average_updated.add(key)

    selected_id = opponent_choices.get(key)
    if selected_id is None:
        selected_id = _sample_policy_action(action_ids, sigma, rng)
        opponent_choices[key] = selected_id
        stats.opponent_action_samples += 1
    else:
        stats.opponent_cache_hits += 1
    child = next(child for action_id, child in actions if action_id == selected_id)
    return _external_traverse(
        adapter,
        child,
        traverser=traverser,
        strategy=strategy,
        regret_delta=regret_delta,
        rng=rng,
        chance_choices=chance_choices,
        opponent_choices=opponent_choices,
        average_updated=average_updated,
        strategy_sum=strategy_sum,
        average_weight=average_weight,
        root_chance_identity=root_chance_identity,
        stats=stats,
    )


@dataclass(frozen=True)
class ExternalSamplingRegretEstimate:
    traverser: Actor
    samples: int
    seed: int
    mean_regret_delta: Mapping[InfoSetKey, Mapping[str, float]]
    mean_sampled_value_bb: float
    sampling_stats: Mapping[str, Any]
    metadata: Mapping[str, Any]


def estimate_external_sampling_regret_deltas(
    root: PublicTreeNode,
    *,
    strategy: Mapping[InfoSetKey, Mapping[str, float]],
    traverser: Actor,
    samples: int,
    seed: int,
) -> ExternalSamplingRegretEstimate:
    """Estimate one fixed profile's raw CFR regret delta without clipping.

    This diagnostic API makes the MCCFR unbiasedness contract directly
    testable against a hand calculation or the exact CFR+ oracle's first
    iteration.  Returned deltas already use the acting player's sign: BB
    maximizes ``utility_bb`` and BTN minimizes it.
    """
    if traverser not in ("bb", "btn"):
        raise ValueError("traverser must be 'bb' or 'btn'")
    if isinstance(samples, bool) or int(samples) <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    samples = int(samples)
    seed = int(seed)
    adapter = ExplicitPublicTreeAdapter(root)
    compiled = adapter.compile()
    _validate_profile(compiled, strategy)
    traverser_keys = tuple(
        key
        for key in compiled.stable_infosets
        if compiled.infoset_actors[key] == traverser
    )
    totals = {
        key: {action_id: 0.0 for action_id in compiled.infoset_actions[key]}
        for key in traverser_keys
    }
    rng = random.Random(seed)
    stats = _MutableSamplingStats()
    sampled_values: list[float] = []
    root_chance_identity = id(root) if isinstance(root, CfrChanceNode) else None
    for _sample in range(samples):
        stats.begin(traverser)
        delta = {
            key: {action_id: 0.0 for action_id in compiled.infoset_actions[key]}
            for key in traverser_keys
        }
        sampled_values.append(
            _external_traverse(
                adapter,
                root,
                traverser=traverser,
                strategy=strategy,
                regret_delta=delta,
                rng=rng,
                chance_choices={},
                opponent_choices={},
                average_updated=set(),
                strategy_sum=None,
                average_weight=0.0,
                root_chance_identity=root_chance_identity,
                stats=stats,
            )
        )
        for key in traverser_keys:
            for action_id in compiled.infoset_actions[key]:
                totals[key][action_id] += delta[key][action_id]

    expected_root_samples = samples if root_chance_identity is not None else 0
    if stats.root_posterior_samples != expected_root_samples:
        raise AssertionError("root posterior was not sampled exactly once per episode")
    return ExternalSamplingRegretEstimate(
        traverser=traverser,
        samples=samples,
        seed=seed,
        mean_regret_delta={
            key: {
                action_id: totals[key][action_id] / samples
                for action_id in compiled.infoset_actions[key]
            }
            for key in traverser_keys
        },
        mean_sampled_value_bb=math.fsum(sampled_values) / samples,
        sampling_stats=stats.snapshot(),
        metadata={
            "method": "fixed_profile_external_sampling_regret_estimate",
            "sampling_scheme": "external_sampling",
            "samples": samples,
            "seed": seed,
            "rng_algorithm": "python_random_mt19937",
            "regret_clipping_performed": False,
            "opponent_sample_cached_per_infoset": True,
            "root_is_chance": root_chance_identity is not None,
            "root_posterior_sampled_once_per_traversal": (
                root_chance_identity is not None
            ),
            "chance_probability_multiplied_after_sampling": False,
            "joint_particle_weight_used": False,
            "strategy_fusion": False,
            "position_contract_version": POSITION_CONTRACT_VERSION,
        },
    )


@dataclass(frozen=True)
class ExternalSamplingPublicMccfrResult:
    iterations: int
    traversals: int
    seed: int
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    current_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    cumulative_regret_plus: Mapping[InfoSetKey, Mapping[str, float]]
    metrics: PublicTreeProfileMetrics | None
    exploitability_trace: tuple[tuple[int, float], ...]
    sampling_stats: Mapping[str, Any]
    metadata: Mapping[str, Any]


def solve_external_sampling_public_mccfr(
    root: PublicTreeNode,
    *,
    iterations: int,
    seed: int,
    linear_averaging: bool = True,
    checkpoints: Sequence[int] = (),
    evaluate_metrics: bool = True,
    max_pure_profiles: int = 1_000_000,
    resume_from: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> ExternalSamplingPublicMccfrResult:
    """Run alternating tabular external-sampling MCCFR+.

    One ``iteration`` is a BB regret episode followed by a BTN regret episode.
    The second episode observes the first episode's clipped CFR+ update, which
    is the standard alternating-update convention.  Average strategy updates
    happen at opponent nodes (the standard two-player simple estimator).

    When ``resume_from`` is supplied, ``iterations`` is the number of
    *additional* complete BB+BTN iterations to execute.  Metric checkpoint
    numbers remain absolute completed-iteration numbers.  ``checkpoint_path``
    atomically writes the final iteration-boundary state as content-addressed
    canonical JSON; it may be the same path as ``resume_from``.
    """
    if isinstance(iterations, bool) or int(iterations) <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(linear_averaging, bool):
        raise TypeError("linear_averaging must be bool")
    if not isinstance(evaluate_metrics, bool):
        raise TypeError("evaluate_metrics must be bool")
    iterations = int(iterations)
    seed = int(seed)
    adapter = ExplicitPublicTreeAdapter(root)
    compiled = adapter.compile()
    rng = random.Random(seed)
    if resume_from is None:
        completed_before = 0
        regrets = {
            key: {
                action_id: 0.0
                for action_id in compiled.infoset_actions[key]
            }
            for key in compiled.stable_infosets
        }
        strategy_sum = {
            key: {
                action_id: 0.0
                for action_id in compiled.infoset_actions[key]
            }
            for key in compiled.stable_infosets
        }
        stats = _MutableSamplingStats()
    else:
        restored = _restore_checkpoint(
            resume_from,
            compiled=compiled,
            seed=seed,
            linear_averaging=linear_averaging,
        )
        completed_before = restored.completed_iterations
        regrets = restored.regrets
        strategy_sum = restored.strategy_sum
        rng.setstate(restored.rng_state)
        stats = restored.stats
    completed_total = completed_before + iterations
    root_chance_identity = id(root) if isinstance(root, CfrChanceNode) else None
    checkpoint_set = (
        {
            int(point)
            for point in checkpoints
            if completed_before < int(point) <= completed_total
        }
        if evaluate_metrics
        else set()
    )
    if evaluate_metrics:
        checkpoint_set.add(completed_total)
    trace: list[tuple[int, float]] = []
    final_metrics: PublicTreeProfileMetrics | None = None

    for iteration in range(completed_before + 1, completed_total + 1):
        average_weight = float(iteration if linear_averaging else 1)
        for traverser in ("bb", "btn"):
            strategy = _strategy_from_regrets(compiled, regrets)
            traverser_keys = tuple(
                key
                for key in compiled.stable_infosets
                if compiled.infoset_actors[key] == traverser
            )
            delta = {
                key: {
                    action_id: 0.0
                    for action_id in compiled.infoset_actions[key]
                }
                for key in traverser_keys
            }
            stats.begin(traverser)
            _external_traverse(
                adapter,
                root,
                traverser=traverser,
                strategy=strategy,
                regret_delta=delta,
                rng=rng,
                chance_choices={},
                opponent_choices={},
                average_updated=set(),
                strategy_sum=strategy_sum,
                average_weight=average_weight,
                root_chance_identity=root_chance_identity,
                stats=stats,
            )
            # Aggregate every sampled physical contribution by InfoSetKey
            # before applying the single CFR+ clipping operation.
            for key in traverser_keys:
                for action_id in compiled.infoset_actions[key]:
                    regrets[key][action_id] = max(
                        0.0,
                        regrets[key][action_id] + delta[key][action_id],
                    )

        if iteration in checkpoint_set:
            average = _average_from_sum(compiled, strategy_sum)
            metrics = public_tree_profile_metrics(
                root,
                average,
                max_pure_profiles=max_pure_profiles,
            )
            trace.append((iteration, metrics.exploitability))
            if iteration == completed_total:
                final_metrics = metrics

    expected_root_samples = (
        2 * completed_total if root_chance_identity is not None else 0
    )
    if stats.root_posterior_samples != expected_root_samples:
        raise AssertionError("root posterior was not sampled exactly once per episode")
    average_strategy = _average_from_sum(compiled, strategy_sum)
    current_strategy = _strategy_from_regrets(compiled, regrets)
    final_payload = _checkpoint_payload(
        compiled=compiled,
        completed_iterations=completed_total,
        seed=seed,
        linear_averaging=linear_averaging,
        regrets=regrets,
        strategy_sum=strategy_sum,
        rng_state=rng.getstate(),
        stats=stats,
    )
    final_envelope = _checkpoint_envelope(final_payload)
    final_checkpoint_sha256 = final_envelope["checkpoint_sha256"]
    if checkpoint_path is not None:
        _atomic_write_checkpoint(checkpoint_path, final_envelope)
    return ExternalSamplingPublicMccfrResult(
        iterations=completed_total,
        traversals=2 * completed_total,
        seed=seed,
        average_strategy=average_strategy,
        current_strategy=current_strategy,
        cumulative_regret_plus={
            key: dict(regrets[key]) for key in compiled.stable_infosets
        },
        metrics=final_metrics,
        exploitability_trace=tuple(trace),
        sampling_stats=stats.snapshot(),
        metadata={
            "method": "external_sampling_public_mccfr_plus_tabular",
            "tree_scope": "finite_explicit_reduced_non_full_card",
            "adapter": "explicit_public_tree_v1",
            "sampling_scheme": "external_sampling",
            "iterations": completed_total,
            "traversals": 2 * completed_total,
            "seed": seed,
            "rng_algorithm": RNG_ALGORITHM,
            "traverser_schedule": "bb_then_btn_each_iteration",
            "alternating_updates": True,
            "regret_matching_plus": True,
            "linear_averaging": linear_averaging,
            "average_strategy_estimator": "two_player_simple_opponent_node",
            "stable_action_ids": True,
            "opponent_sample_cached_per_infoset": True,
            "chance_sample_cached_per_node": True,
            "root_is_chance": root_chance_identity is not None,
            "root_posterior_sampled_once_per_traversal": (
                root_chance_identity is not None
            ),
            "chance_probability_multiplied_after_sampling": False,
            "chance_source": "explicit_tree_branches_only",
            "joint_particle_weight_used": False,
            "terminal_utility_perspective": "bb",
            "terminal_utility_sources": sorted(
                compiled.terminal_utility_sources
            ),
            "strategy_fusion": False,
            "infoset_aware_exact_metrics": evaluate_metrics,
            "full_card": False,
            "full_card_policy_promoted": False,
            "hu_exact": False,
            "runtime_integrated": False,
            "rust_leaf_integrated": (
                "rust_exact_physical_t4_action_vector"
                in compiled.terminal_utility_sources
            ),
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_iteration": completed_total,
            "checkpoint_sha256": final_checkpoint_sha256,
        },
    )


@dataclass(frozen=True)
class ExternalSamplingOracleComparison:
    sampled: ExternalSamplingPublicMccfrResult
    oracle: RecursivePublicTreeCfrResult
    strategy_total_variation: Mapping[InfoSetKey, float]
    max_strategy_total_variation: float
    mean_strategy_total_variation: float
    value_bb_gap: float
    exploitability_gap: float
    metadata: Mapping[str, Any]


def compare_external_sampling_with_cfr_plus(
    root: PublicTreeNode,
    *,
    sampled_iterations: int,
    oracle_iterations: int,
    seed: int,
    linear_averaging: bool = True,
    max_pure_profiles: int = 1_000_000,
) -> ExternalSamplingOracleComparison:
    """Solve one reduced tree with MCCFR and the deterministic CFR+ oracle."""
    sampled = solve_external_sampling_public_mccfr(
        root,
        iterations=sampled_iterations,
        seed=seed,
        linear_averaging=linear_averaging,
        max_pure_profiles=max_pure_profiles,
    )
    oracle = solve_recursive_public_tree_cfr_plus(
        root,
        iterations=oracle_iterations,
        linear_averaging=linear_averaging,
        max_pure_profiles=max_pure_profiles,
    )
    if sampled.metrics is None:  # pragma: no cover - construction defense
        raise AssertionError("oracle comparison requires sampled exact metrics")
    compiled = ExplicitPublicTreeAdapter(root).compile()
    total_variation = {
        key: 0.5
        * math.fsum(
            abs(
                sampled.average_strategy[key][action_id]
                - oracle.average_strategy[key][action_id]
            )
            for action_id in compiled.infoset_actions[key]
        )
        for key in compiled.stable_infosets
    }
    values = tuple(total_variation.values())
    return ExternalSamplingOracleComparison(
        sampled=sampled,
        oracle=oracle,
        strategy_total_variation=total_variation,
        max_strategy_total_variation=max(values, default=0.0),
        mean_strategy_total_variation=(
            math.fsum(values) / len(values) if values else 0.0
        ),
        value_bb_gap=sampled.metrics.value_bb - oracle.metrics.value_bb,
        exploitability_gap=(
            sampled.metrics.exploitability - oracle.metrics.exploitability
        ),
        metadata={
            "method": "reduced_external_sampling_vs_recursive_cfr_plus",
            "comparison_scope": "same_explicit_reduced_tree",
            "profile_identity": "InfoSetKey_and_stable_action_id",
            "oracle": "solve_recursive_public_tree_cfr_plus",
            "sampled_seed": int(seed),
            "full_card_claim": False,
        },
    )
