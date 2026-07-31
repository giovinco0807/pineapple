"""Shared-infoset MCCFR+ over a chance mixture of full-card T3 roots.

The single-root solver in :mod:`ai.tutor.t3_hu_full_card_mccfr` is the
physical transition boundary for one observed T3 information set.  Solving
several private roots independently, however, permits a later player to use a
different policy for the same information set in each solve.  Combining those
policies afterwards is strategy fusion and cannot represent public-information
signalling.

This module adds the missing ex-ante chance super-root.  One exact prior root
mass and one conditional posterior particle are sampled for every traversal,
while *all* roots update one regret/strategy table keyed only by
``InfoSetKey``.  Root identifiers and physical particles are audit provenance;
neither is ever part of a policy identity or strategy serialization.

The solver remains an encountered-only, sampled prototype.  It deliberately
makes no exact exploitability, promotion, runtime-integration, or all-turn
claim.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import CodeType, MappingProxyType, ModuleType
from typing import Any, Mapping, Protocol, Sequence

import ai.tutor.t3_hu_full_card_mccfr as _full_card_module
import ai.tutor.exact_late as _exact_late_module
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_full_card_mccfr import (
    POLICY_IDENTITY_CONTRACT,
    RNG_ALGORITHM,
    FullCardGenerativeAdapter,
    _DynamicMccfrTables,
    _adapter_manifest as _single_root_adapter_manifest,
    _decode_rng_state as _single_root_decode_rng_state,
    _encode_rng_state as _single_root_encode_rng_state,
    _float_from_hex as _single_root_float_from_hex,
    _float_hex as _single_root_float_hex,
    _freeze_profile,
    _infoset_from_canonical_json as _single_root_infoset_from_json,
    _legal_action_ids_for_key as _single_root_legal_action_ids,
    _sample_cached_opponent_action,
    sample_exact_fraction_index,
    seeded_rng,
    serialize_strategy_profile,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey
from ai.tutor.t3_hu_public_tree import PublicTreeTerminalState
from ai.tutor.t3_hu_public_tree_cfr import (
    PublicTreeDecisionNode,
    PublicTreeTerminalNode,
)


SUPER_ROOT_SAMPLING_CONTRACT = "exact_fraction_chance_super_root_v1"
MULTI_ROOT_SOLVER_METHOD = "full_card_shared_infoset_multi_root_mccfr_plus_v1"
MULTI_ROOT_CHECKPOINT_FORMAT = "shared_multi_root_mccfr_checkpoint_v1"
MULTI_ROOT_SOLVER_STATE_FORMAT = "shared_multi_root_mccfr_state_v1"
MULTI_ROOT_ADAPTER_BINDING_FORMAT = "shared_multi_root_adapter_binding_v1"

_CANONICAL_SOLVER_EXTERNAL_GLOBALS = {
    "sample_exact_fraction_index": sample_exact_fraction_index,
    "seeded_rng": seeded_rng,
    "_sample_cached_opponent_action": _sample_cached_opponent_action,
    "serialize_strategy_profile": serialize_strategy_profile,
    "_freeze_profile": _freeze_profile,
    "_DynamicMccfrTables": _DynamicMccfrTables,
    "InfoSetKey": InfoSetKey,
}


class MultiRootMccfrCheckpointError(ValueError):
    """A multi-root checkpoint or compatibility binding failed validation."""


class MultiRootTraversalAdapter(Protocol):
    """Structural boundary used by the multi-root search.

    ``FullCardGenerativeAdapter`` satisfies this protocol.  Small explicit
    reduced-tree adapters may also satisfy it, which lets tests compare the
    sampled algorithm with the existing exact shared-infoset CFR reference.
    """

    observation: InfoSetKey

    @property
    def root_distribution(self) -> tuple[tuple[str, Fraction], ...]: ...

    def sample_root_for_traversal(self, rng: Any) -> Any: ...

    def information_key(self, state: Any) -> InfoSetKey: ...

    def legal_actions(self, state: Any) -> tuple[tuple[str, Any], ...]: ...

    def apply_action_id(self, state: Any, stable_action_id: str) -> Any: ...

    def sample_next_draw(self, pending: Any, rng: Any) -> Any: ...

    def terminal_utility_bb(self, terminal: Any) -> float: ...

    def sampling_audit(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _CanonicalReducedRootSample:
    state: PublicTreeDecisionNode


@dataclass(frozen=True)
class _CanonicalReducedTransition:
    state: PublicTreeDecisionNode
    next_phase: str


class CanonicalReducedPublicTreeAdapter:
    """Repository-owned deterministic adapter for finite-tree validation.

    Arbitrary structural adapters remain usable for non-checkpoint sampled
    solves, but cannot be soundly resumed because their instance state and
    referenced module globals are not under this module's attestation.  This
    exact-type adapter is the small canonical boundary used by deterministic
    checkpoint/replay and algorithm-validation tests.
    """

    def __init__(self, root: PublicTreeDecisionNode, commitment: str) -> None:
        if not isinstance(root, PublicTreeDecisionNode):
            raise TypeError("canonical reduced root must be a decision node")
        if not isinstance(commitment, str) or not commitment:
            raise ValueError("canonical reduced commitment must be non-empty")
        self.root = root
        self.observation = root.infoset_key
        self._commitment = commitment
        self._traversals = 0
        self._future_draws = 0

    @property
    def root_distribution(self) -> tuple[tuple[str, Fraction], ...]:
        return ((self._commitment, Fraction(1, 1)),)

    @property
    def checkpoint_binding_manifest(self) -> Mapping[str, Any]:
        infoset_actions: dict[InfoSetKey, tuple[str, ...]] = {}

        def encode(node: Any) -> dict[str, Any]:
            if isinstance(node, PublicTreeTerminalNode):
                return {
                    "kind": "terminal",
                    "utility_bb_exact": _fraction_text(node.utility_bb),
                    "terminal_id": node.terminal_id,
                    "utility_source": node.utility_source,
                }
            if not isinstance(node, PublicTreeDecisionNode):
                raise MultiRootMccfrCheckpointError(
                    "canonical reduced checkpoint tree permits only direct "
                    "decision and terminal nodes"
                )
            key = node.infoset_key
            prior = infoset_actions.get(key)
            if prior is not None and prior != node.action_ids:
                raise MultiRootMccfrCheckpointError(
                    "canonical reduced shared infoset action mismatch"
                )
            infoset_actions[key] = node.action_ids
            return {
                "kind": "decision",
                "infoset_sha256": key.digest(),
                "actions": [
                    {"action_id": action_id, "child": encode(child)}
                    for action_id, child in node.actions
                ],
            }

        tree = encode(self.root)
        rows = [
            {
                "infoset_canonical_json": key.canonical_json(),
                "infoset_sha256": key.digest(),
                "actor": key.actor,
                "stable_action_ids": list(infoset_actions[key]),
            }
            for key in sorted(
                infoset_actions,
                key=lambda value: (value.digest(), value.canonical_json()),
            )
        ]
        return MappingProxyType(
            {
                "schema": "canonical_reduced_public_tree_checkpoint_binding_v1",
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "tree_sha256": _checkpoint_content_sha256(tree),
                "infoset_actions": rows,
            }
        )

    def sample_root_for_traversal(self, _rng: Any) -> _CanonicalReducedRootSample:
        self._traversals += 1
        return _CanonicalReducedRootSample(self.root)

    @staticmethod
    def information_key(state: PublicTreeDecisionNode) -> InfoSetKey:
        if not isinstance(state, PublicTreeDecisionNode):
            raise TypeError("canonical reduced state must be a decision node")
        return state.infoset_key

    @staticmethod
    def legal_actions(
        state: PublicTreeDecisionNode,
    ) -> tuple[tuple[str, Any], ...]:
        if not isinstance(state, PublicTreeDecisionNode):
            raise TypeError("canonical reduced state must be a decision node")
        return tuple((action_id, child) for action_id, child in state.actions)

    @staticmethod
    def apply_action_id(state: PublicTreeDecisionNode, stable_action_id: str) -> Any:
        return dict(state.actions)[stable_action_id]

    @staticmethod
    def is_terminal_result(value: Any) -> bool:
        return isinstance(value, PublicTreeTerminalNode)

    def sample_next_draw(
        self,
        pending: PublicTreeDecisionNode,
        _rng: Any,
    ) -> _CanonicalReducedTransition:
        if not isinstance(pending, PublicTreeDecisionNode):
            raise TypeError(
                "canonical reduced direct transition requires a decision node"
            )
        self._future_draws += 1
        return _CanonicalReducedTransition(
            pending,
            pending.infoset_key.phase,
        )

    @staticmethod
    def terminal_utility_bb(terminal: PublicTreeTerminalNode) -> float:
        if not isinstance(terminal, PublicTreeTerminalNode):
            raise TypeError("canonical reduced terminal must be a terminal node")
        return float(terminal.utility_bb)

    def sampling_audit(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "traversals": self._traversals,
                "root_posterior_samples": self._traversals,
                "future_draw_samples": self._future_draws,
            }
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _root_id_sha256(root_id: str) -> str:
    return _sha256_text(f"ofc-multi-root-id-v1\x00{root_id}")


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _checkpoint_json_bytes(value: Any) -> bytes:
    try:
        return _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint contains non-canonical JSON data"
        ) from exc


def _checkpoint_content_sha256(value: Any) -> str:
    return hashlib.sha256(_checkpoint_json_bytes(value)).hexdigest()


def _checkpoint_snapshot(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_checkpoint_json_bytes(value).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:  # pragma: no cover
        raise MultiRootMccfrCheckpointError(
            f"{label} is not canonical JSON data"
        ) from exc


def _reject_duplicate_checkpoint_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MultiRootMccfrCheckpointError(
                f"checkpoint JSON contains duplicate key {key!r}"
            )
        value[key] = item
    return value


def _reject_nonfinite_json_constant(raw: str) -> Any:
    raise MultiRootMccfrCheckpointError(
        f"checkpoint JSON contains forbidden non-finite constant {raw!r}"
    )


def _checkpoint_path(
    raw_path: str | os.PathLike[str],
    *,
    for_read: bool,
) -> Path:
    try:
        path_text = os.fspath(raw_path)
    except TypeError as exc:
        raise MultiRootMccfrCheckpointError(
            "checkpoint path must be str or os.PathLike"
        ) from exc
    if not isinstance(path_text, str) or not path_text:
        raise MultiRootMccfrCheckpointError(
            "checkpoint path must be a non-empty text path"
        )
    candidate = Path(path_text)
    if any(part == ".." for part in candidate.parts):
        raise MultiRootMccfrCheckpointError(
            "checkpoint path must not contain parent traversal"
        )
    if candidate.name in ("", ".", ".."):
        raise MultiRootMccfrCheckpointError("checkpoint path must name a file")

    if for_read:
        if candidate.is_symlink():
            raise MultiRootMccfrCheckpointError(
                "checkpoint path must not be a symbolic link"
            )
        if not candidate.is_file():
            raise MultiRootMccfrCheckpointError(
                "checkpoint path must name an existing regular file"
            )
    else:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink():
            raise MultiRootMccfrCheckpointError(
                "checkpoint target must not be a symbolic link"
            )
        if candidate.exists() and not candidate.is_file():
            raise MultiRootMccfrCheckpointError(
                "checkpoint target must be a regular file"
            )
    # Existing symlinked ancestors make the atomic-replace destination depend
    # on external filesystem state.  Reject them rather than following them.
    for ancestor in (candidate.parent, *candidate.parent.parents):
        if ancestor.is_symlink():
            raise MultiRootMccfrCheckpointError(
                "checkpoint path must not traverse a symbolic-link directory"
            )
    return candidate


_MULTI_ROOT_SOURCE_BINDING_PATHS = (
    "ai/tutor/t3_hu_multi_root_mccfr.py",
    "ai/tutor/t3_hu_full_card_mccfr.py",
    "ai/tutor/t3_hu_full_card_range.py",
    "ai/tutor/t3_hu_public_cfr.py",
    "ai/tutor/t3_hu_public_tree.py",
    "ai/tutor/exact_late.py",
    "ai/engine/action_space.py",
    "ai/engine/encoding.py",
    "ai/engine/game_engine.py",
    "ai/engine/scoring.py",
    "ai/engine/turn_order.py",
    "ai/mcts/rollout_evaluator.py",
    "ai/config/fl_ev.json",
)


def _multi_root_source_binding() -> dict[str, Any]:
    runtime_drift = sorted(
        name
        for name, canonical in _CANONICAL_SOLVER_EXTERNAL_GLOBALS.items()
        if globals().get(name) is not canonical
    )
    if runtime_drift:
        raise MultiRootMccfrCheckpointError(
            "multi-root solver live runtime globals were overridden: "
            f"{runtime_drift}"
        )
    project_root = Path(__file__).resolve().parents[2]
    source_sha256: dict[str, str] = {}
    for relative_path in _MULTI_ROOT_SOURCE_BINDING_PATHS:
        try:
            content = (project_root / relative_path).read_bytes()
        except OSError as exc:
            raise MultiRootMccfrCheckpointError(
                f"cannot bind solver source {relative_path!r}: {exc}"
            ) from exc
        source_sha256[relative_path] = hashlib.sha256(content).hexdigest()
    binding = {
        "schema": "shared_multi_root_solver_source_binding_v1",
        "solver_method": MULTI_ROOT_SOLVER_METHOD,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "super_root_sampling_contract": SUPER_ROOT_SAMPLING_CONTRACT,
        "stable_action_identity": "exact_late.action_key_lexical_v1",
        "rules_contract": {
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "deck_size": 54,
            "physical_joker_ids": ["X1", "X2"],
            "terminal_utility_perspective": "bb",
            "production_terminal_utility_source": (
                "exact_late.terminal_metrics.score_bb"
            ),
        },
        "source_sha256": dict(sorted(source_sha256.items())),
        "live_runtime_semantic_binding": _solver_runtime_semantic_binding(),
    }
    binding["binding_sha256"] = _checkpoint_content_sha256(binding)
    return binding


def _parse_checkpoint_infoset(raw: Any) -> InfoSetKey:
    try:
        return _single_root_infoset_from_json(raw)
    except (TypeError, ValueError) as exc:
        raise MultiRootMccfrCheckpointError(
            "checkpoint InfoSetKey canonical JSON is invalid"
        ) from exc


def _generic_adapter_manifest(adapter: MultiRootTraversalAdapter) -> dict[str, Any]:
    raw = getattr(adapter, "checkpoint_binding_manifest", None)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping):
        raise MultiRootMccfrCheckpointError(
            "non-standard root adapters require checkpoint_binding_manifest"
        )
    snapshot = _checkpoint_snapshot(
        dict(raw), label="generic adapter checkpoint binding manifest"
    )
    if not isinstance(snapshot, dict):
        raise MultiRootMccfrCheckpointError(
            "generic adapter checkpoint binding manifest must be an object"
        )
    schema = snapshot.get("schema")
    rows = snapshot.get("infoset_actions")
    if not isinstance(schema, str) or not schema:
        raise MultiRootMccfrCheckpointError(
            "generic adapter checkpoint binding requires a schema"
        )
    if not isinstance(rows, list) or not rows:
        raise MultiRootMccfrCheckpointError(
            "generic adapter checkpoint binding requires infoset_actions"
        )
    implementation_sha256 = snapshot.get("implementation_sha256")
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in implementation_sha256
        )
    ):
        raise MultiRootMccfrCheckpointError(
            "generic adapter checkpoint binding requires implementation_sha256"
        )
    module = sys.modules.get(type(adapter).__module__)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not isinstance(module_file, str) or not module_file:
        raise MultiRootMccfrCheckpointError(
            "generic adapter implementation module has no bindable source file"
        )
    try:
        actual_implementation_sha256 = hashlib.sha256(
            Path(module_file).read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise MultiRootMccfrCheckpointError(
            "generic adapter implementation source cannot be read"
        ) from exc
    if implementation_sha256 != actual_implementation_sha256:
        raise MultiRootMccfrCheckpointError(
            "generic adapter implementation_sha256 does not match live source"
        )
    row_schema = {
        "infoset_canonical_json",
        "infoset_sha256",
        "actor",
        "stable_action_ids",
    }
    prior_order: tuple[str, str] | None = None
    contains_root = False
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_schema:
            raise MultiRootMccfrCheckpointError(
                "generic adapter infoset action row schema is invalid"
            )
        key = _parse_checkpoint_infoset(row["infoset_canonical_json"])
        order = (key.digest(), key.canonical_json())
        if prior_order is not None and order <= prior_order:
            raise MultiRootMccfrCheckpointError(
                "generic adapter infoset action rows are not canonically ordered"
            )
        prior_order = order
        if row["infoset_sha256"] != key.digest() or row["actor"] != key.actor:
            raise MultiRootMccfrCheckpointError(
                "generic adapter infoset identity binding is invalid"
            )
        action_ids = row["stable_action_ids"]
        if (
            not isinstance(action_ids, list)
            or not action_ids
            or any(not isinstance(action_id, str) or not action_id for action_id in action_ids)
            or action_ids != sorted(action_ids)
            or len(action_ids) != len(set(action_ids))
        ):
            raise MultiRootMccfrCheckpointError(
                "generic adapter stable action binding is invalid"
            )
        contains_root = contains_root or key == adapter.observation
    if not contains_root:
        raise MultiRootMccfrCheckpointError(
            "generic adapter action binding does not contain its root InfoSetKey"
        )
    return snapshot


_GENERIC_RUNTIME_MEMBERS = (
    "root_distribution",
    "sample_root_for_traversal",
    "information_key",
    "legal_actions",
    "apply_action_id",
    "sample_next_draw",
    "terminal_utility_bb",
    "sampling_audit",
    "is_terminal_result",
    "checkpoint_binding_manifest",
)


def _python_code_constant_binding(value: Any) -> Any:
    if value is None:
        return {"type": "none"}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MultiRootMccfrCheckpointError(
                "generic runtime code contains a non-finite constant"
            )
        return {"type": "float", "value": value.hex()}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise MultiRootMccfrCheckpointError(
                "generic runtime code contains a non-finite complex constant"
            )
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_python_code_constant_binding(item) for item in value],
        }
    if isinstance(value, frozenset):
        items = [_python_code_constant_binding(item) for item in value]
        return {
            "type": "frozenset",
            "items": sorted(items, key=_canonical_json),
        }
    if isinstance(value, CodeType):
        return {"type": "code", "value": _python_code_binding(value)}
    raise MultiRootMccfrCheckpointError(
        "generic runtime code contains an unsupported constant"
    )


def _python_code_binding(code: CodeType) -> dict[str, Any]:
    return {
        "name": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "exceptiontable_hex": getattr(code, "co_exceptiontable", b"").hex(),
        "constants": [
            _python_code_constant_binding(value) for value in code.co_consts
        ],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _python_function_binding(function: Any) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        raise MultiRootMccfrCheckpointError(
            "generic runtime member has no bindable Python code"
        )
    closure = getattr(function, "__closure__", None) or ()
    if len(closure) != len(code.co_freevars):
        raise MultiRootMccfrCheckpointError(
            "generic runtime member closure shape is invalid"
        )
    closure_rows = []
    for name, cell in zip(code.co_freevars, closure):
        try:
            value = cell.cell_contents
        except ValueError as exc:
            raise MultiRootMccfrCheckpointError(
                "generic runtime member has an empty closure cell"
            ) from exc
        if isinstance(value, type):
            bound_value = {
                "type": "python_type",
                "module": value.__module__,
                "qualname": value.__qualname__,
            }
        else:
            try:
                bound_value = _python_code_constant_binding(value)
            except MultiRootMccfrCheckpointError as exc:
                try:
                    bound_value = _runtime_data_binding(value)
                except MultiRootMccfrCheckpointError:
                    raise MultiRootMccfrCheckpointError(
                        "runtime member closure value is not checkpoint-bindable"
                    ) from exc
        closure_rows.append({"name": name, "value": bound_value})
    defaults = getattr(function, "__defaults__", None)
    keyword_defaults = getattr(function, "__kwdefaults__", None)
    if keyword_defaults is not None and (
        not isinstance(keyword_defaults, dict)
        or any(not isinstance(key, str) for key in keyword_defaults)
    ):
        raise MultiRootMccfrCheckpointError(
            "generic runtime keyword defaults are not checkpoint-bindable"
        )

    def bind_runtime_value(value: Any) -> Any:
        try:
            return _python_code_constant_binding(value)
        except MultiRootMccfrCheckpointError as exc:
            try:
                return _runtime_data_binding(value)
            except MultiRootMccfrCheckpointError:
                raise MultiRootMccfrCheckpointError(
                    "runtime function value is not checkpoint-bindable"
                ) from exc

    return {
        "code": _python_code_binding(code),
        "defaults": bind_runtime_value(defaults),
        "keyword_defaults": {
            key: bind_runtime_value(keyword_defaults[key])
            for key in sorted(keyword_defaults or {})
        },
        "closure": closure_rows,
    }


def _runtime_data_binding(value: Any, *, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MultiRootMccfrCheckpointError(
                "runtime semantic graph contains non-finite data"
            )
        return {"type": "float", "value": value.hex()}
    if isinstance(value, Fraction):
        return {"type": "fraction", "value": _fraction_text(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if inspect.isfunction(value):
        return {
            "type": "python_function",
            "identity": _runtime_function_key(value),
            "binding": _python_function_binding(value),
        }
    if isinstance(value, type):
        return {
            "type": "python_type",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if isinstance(value, ModuleType):
        module_file = getattr(value, "__file__", None)
        source_sha256 = None
        if isinstance(module_file, str) and value.__name__.startswith("ai."):
            try:
                source_sha256 = hashlib.sha256(
                    Path(module_file).read_bytes()
                ).hexdigest()
            except OSError as exc:
                raise MultiRootMccfrCheckpointError(
                    "runtime semantic module source cannot be read"
                ) from exc
        return {
            "type": "python_module",
            "name": value.__name__,
            "source_sha256": source_sha256,
        }
    if type(value).__module__ == "dataclasses":
        return {
            "type": "dataclasses_runtime_sentinel",
            "class": type(value).__qualname__,
            "repr": repr(value),
        }
    identity = id(value)
    if identity in seen:
        raise MultiRootMccfrCheckpointError(
            "runtime semantic graph contains cyclic mutable data"
        )
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            rows = [
                {
                    "key": _runtime_data_binding(key, seen=seen),
                    "value": _runtime_data_binding(item, seen=seen),
                }
                for key, item in value.items()
            ]
            return {"type": "mapping", "items": sorted(rows, key=_canonical_json)}
        if isinstance(value, (tuple, list)):
            return {
                "type": "tuple" if isinstance(value, tuple) else "list",
                "items": [
                    _runtime_data_binding(item, seen=seen) for item in value
                ],
            }
        if isinstance(value, (set, frozenset)):
            items = [_runtime_data_binding(item, seen=seen) for item in value]
            return {"type": "set", "items": sorted(items, key=_canonical_json)}
    finally:
        seen.remove(identity)
    raise MultiRootMccfrCheckpointError(
        "runtime semantic graph contains unsupported mutable data"
    )


def _nested_code_global_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_nested_code_global_names(value))
    return names


def _runtime_function_key(function: Any) -> str:
    return (
        f"{getattr(function, '__module__', '')}:"
        f"{getattr(function, '__qualname__', '')}"
    )


def _runtime_semantic_graph(
    roots: Mapping[str, Any],
    *,
    data_roots: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pending = [(label, function) for label, function in sorted(roots.items())]
    root_keys: dict[str, str] = {}
    functions: dict[str, dict[str, Any]] = {}
    classes: dict[str, dict[str, Any]] = {}
    data: dict[str, Any] = {
        label: _runtime_data_binding(value)
        for label, value in sorted((data_roots or {}).items())
    }
    seen_objects: set[int] = set()

    while pending:
        label, function = pending.pop(0)
        function = getattr(function, "__func__", function)
        if not inspect.isfunction(function):
            raise MultiRootMccfrCheckpointError(
                f"runtime semantic root {label!r} is not a Python function"
            )
        key = _runtime_function_key(function)
        if label in roots:
            root_keys[label] = key
        if id(function) in seen_objects:
            continue
        seen_objects.add(id(function))
        binding = _python_function_binding(function)
        prior = functions.get(key)
        if prior is not None and prior != binding:
            raise MultiRootMccfrCheckpointError(
                "runtime semantic graph has ambiguous callable identity"
            )
        functions[key] = binding

        module_name = str(getattr(function, "__module__", ""))
        recurse = module_name == __name__ or module_name.startswith("ai.")
        if not recurse:
            continue
        function_globals = getattr(function, "__globals__", {})
        for name in sorted(_nested_code_global_names(function.__code__)):
            if name not in function_globals:
                continue
            value = function_globals[name]
            dependency_label = f"{module_name}:{name}"
            if inspect.isfunction(value):
                pending.append((dependency_label, value))
            elif isinstance(value, type) and value.__module__.startswith("ai."):
                class_key = f"{value.__module__}:{value.__qualname__}"
                if class_key not in classes:
                    method_keys: dict[str, str] = {}
                    class_data: dict[str, Any] = {}
                    for member_name, descriptor in sorted(vars(value).items()):
                        candidate = descriptor
                        if isinstance(candidate, property):
                            candidate = candidate.fget
                        elif isinstance(candidate, (staticmethod, classmethod)):
                            candidate = candidate.__func__
                        if inspect.isfunction(candidate):
                            method_keys[member_name] = _runtime_function_key(candidate)
                            pending.append(
                                (f"{class_key}.{member_name}", candidate)
                            )
                        elif (
                            not member_name.startswith("__")
                            and isinstance(
                                candidate,
                                (type(None), bool, int, float, str, bytes,
                                 Fraction, tuple, list, set, frozenset, Mapping),
                            )
                        ):
                            class_data[member_name] = _runtime_data_binding(candidate)
                    classes[class_key] = {
                        "methods": dict(sorted(method_keys.items())),
                        "data": class_data,
                    }
            elif isinstance(
                value,
                (type(None), bool, int, float, str, bytes, Fraction,
                 tuple, list, set, frozenset, Mapping),
            ):
                data[dependency_label] = _runtime_data_binding(value)

    graph = {
        "schema": "python_live_runtime_semantic_graph_v1",
        "roots": dict(sorted(root_keys.items())),
        "functions": [
            {"identity": key, "binding": functions[key]}
            for key in sorted(functions)
        ],
        "classes": [
            {"identity": key, "binding": classes[key]}
            for key in sorted(classes)
        ],
        "data": [
            {"identity": key, "binding": data[key]} for key in sorted(data)
        ],
    }
    graph["binding_sha256"] = _checkpoint_content_sha256(graph)
    return graph


def _generic_runtime_callable_binding(
    adapter: MultiRootTraversalAdapter,
) -> list[dict[str, str]]:
    instance_namespace = getattr(adapter, "__dict__", {})
    if not isinstance(instance_namespace, Mapping):
        instance_namespace = {}
    instance_overrides = sorted(
        member
        for member in _GENERIC_RUNTIME_MEMBERS
        if member in instance_namespace
    )
    if instance_overrides:
        raise MultiRootMccfrCheckpointError(
            "generic checkpoint adapter has instance-overridden runtime members: "
            f"{instance_overrides}"
        )
    rows: list[dict[str, str]] = []
    for member in _GENERIC_RUNTIME_MEMBERS:
        try:
            descriptor = inspect.getattr_static(type(adapter), member)
        except AttributeError as exc:
            raise MultiRootMccfrCheckpointError(
                f"generic checkpoint adapter is missing runtime member {member!r}"
            ) from exc
        if isinstance(descriptor, property):
            function = descriptor.fget
        elif isinstance(descriptor, (staticmethod, classmethod)):
            function = descriptor.__func__
        else:
            function = descriptor
        function = getattr(function, "__func__", function)
        function_binding = _python_function_binding(function)
        rows.append(
            {
                "member": member,
                "module": str(getattr(function, "__module__", "")),
                "qualname": str(getattr(function, "__qualname__", "")),
                "code_sha256": _checkpoint_content_sha256(function_binding),
            }
        )
    return rows


_FULL_CARD_CHECKPOINT_METHODS = (
    "root_distribution",
    "sample_root_for_traversal",
    "information_key",
    "legal_actions",
    "apply_action_id",
    "sample_next_draw",
    "terminal_metrics_bb",
    "terminal_utility_bb",
    "sampling_audit",
)
_CANONICAL_FULL_CARD_METHODS = {
    method: (
        inspect.getattr_static(FullCardGenerativeAdapter, method).fget
        if isinstance(
            inspect.getattr_static(FullCardGenerativeAdapter, method), property
        )
        else getattr(
            inspect.getattr_static(FullCardGenerativeAdapter, method),
            "__func__",
            inspect.getattr_static(FullCardGenerativeAdapter, method),
        )
    )
    for method in _FULL_CARD_CHECKPOINT_METHODS
}
_CANONICAL_FULL_CARD_GLOBALS = {
    "full_card.terminal_metrics": (
        _full_card_module,
        "terminal_metrics",
        _full_card_module.terminal_metrics,
    ),
    "full_card.get_turn_actions": (
        _full_card_module,
        "get_turn_actions",
        _full_card_module.get_turn_actions,
    ),
    "exact_late._score_against_complete_opponent": (
        _exact_late_module,
        "_score_against_complete_opponent",
        _exact_late_module._score_against_complete_opponent,
    ),
}

_REDUCED_CHECKPOINT_METHODS = (
    "root_distribution",
    "checkpoint_binding_manifest",
    "sample_root_for_traversal",
    "information_key",
    "legal_actions",
    "apply_action_id",
    "is_terminal_result",
    "sample_next_draw",
    "terminal_utility_bb",
    "sampling_audit",
)
_CANONICAL_REDUCED_METHODS = {
    method: (
        inspect.getattr_static(CanonicalReducedPublicTreeAdapter, method).fget
        if isinstance(
            inspect.getattr_static(CanonicalReducedPublicTreeAdapter, method),
            property,
        )
        else getattr(
            inspect.getattr_static(CanonicalReducedPublicTreeAdapter, method),
            "__func__",
            inspect.getattr_static(CanonicalReducedPublicTreeAdapter, method),
        )
    )
    for method in _REDUCED_CHECKPOINT_METHODS
}
_CANONICAL_REDUCED_GLOBALS = {
    "PublicTreeDecisionNode": PublicTreeDecisionNode,
    "PublicTreeTerminalNode": PublicTreeTerminalNode,
    "_CanonicalReducedRootSample": _CanonicalReducedRootSample,
    "_CanonicalReducedTransition": _CanonicalReducedTransition,
}


def _full_card_runtime_semantic_binding() -> dict[str, Any]:
    roots = {
        f"FullCardGenerativeAdapter.{name}": function
        for name, function in _CANONICAL_FULL_CARD_METHODS.items()
    }
    roots.update(
        {
            "full_card.adapter_manifest": _single_root_adapter_manifest,
            "full_card.get_turn_actions": _full_card_module.get_turn_actions,
            "full_card.terminal_metrics": _full_card_module.terminal_metrics,
        }
    )
    return _runtime_semantic_graph(
        roots,
        data_roots={
            "full_card.EXPECTED_UNDEALT_BY_PHASE": (
                _full_card_module.EXPECTED_UNDEALT_BY_PHASE
            ),
            "exact_late.FL_TYPE_BY_CARD_COUNT": (
                _exact_late_module.FL_TYPE_BY_CARD_COUNT
            ),
            "exact_late.RolloutEvaluator.FL_EV": (
                _exact_late_module.RolloutEvaluator.FL_EV
            ),
        },
    )


def _reduced_runtime_semantic_binding() -> dict[str, Any]:
    return _runtime_semantic_graph(
        {
            f"CanonicalReducedPublicTreeAdapter.{name}": function
            for name, function in _CANONICAL_REDUCED_METHODS.items()
        }
    )


def _adapter_checkpoint_binding(entry: "MultiRootChanceEntry") -> dict[str, Any]:
    adapter = entry.adapter
    root_distribution = [
        {
            "particle_commitment_sha256": _sha256_text(
                f"ofc-particle-commitment-v1\x00{commitment}"
            ),
            "posterior_probability_exact": _fraction_text(probability),
        }
        for commitment, probability in adapter.root_distribution
    ]
    common = {
        "format": MULTI_ROOT_ADAPTER_BINDING_FORMAT,
        "root_id_sha256": entry.root_id_sha256,
        "prior_mass_exact": _fraction_text(entry.prior_mass),
        "adapter_class_module": type(adapter).__module__,
        "adapter_class_qualname": type(adapter).__qualname__,
        "root_infoset_canonical_json": adapter.observation.canonical_json(),
        "root_infoset_sha256": adapter.observation.digest(),
        "root_distribution": root_distribution,
        "root_distribution_sha256": _checkpoint_content_sha256(root_distribution),
    }
    if type(adapter) is FullCardGenerativeAdapter:
        shadowed = sorted(
            method for method in _FULL_CARD_CHECKPOINT_METHODS if method in adapter.__dict__
        )
        class_drift = sorted(
            method
            for method, canonical in _CANONICAL_FULL_CARD_METHODS.items()
            if (
                inspect.getattr_static(type(adapter), method).fget
                if isinstance(
                    inspect.getattr_static(type(adapter), method), property
                )
                else getattr(
                    inspect.getattr_static(type(adapter), method),
                    "__func__",
                    inspect.getattr_static(type(adapter), method),
                )
            )
            is not canonical
        )
        global_drift = sorted(
            label
            for label, (module, name, canonical) in (
                _CANONICAL_FULL_CARD_GLOBALS.items()
            )
            if getattr(module, name) is not canonical
        )
        if shadowed or class_drift or global_drift:
            raise MultiRootMccfrCheckpointError(
                "full-card checkpoint adapter has overridden methods: "
                f"instance={shadowed}, class={class_drift}, globals={global_drift}"
            )
        try:
            full_manifest = _single_root_adapter_manifest(adapter)
        except (TypeError, ValueError) as exc:
            raise MultiRootMccfrCheckpointError(
                "full-card adapter checkpoint binding is invalid"
            ) from exc
        common.update(
            {
                "adapter_kind": "full_card_generative_t3_t4_v1",
                "full_card_adapter_manifest": full_manifest,
                "full_card_adapter_manifest_sha256": (
                    _checkpoint_content_sha256(full_manifest)
                ),
                "range_content_sha256": adapter.root_range.range_content_sha256,
                "range_build_sha256": adapter.root_range.range_build_sha256,
                "live_runtime_semantic_binding": (
                    _full_card_runtime_semantic_binding()
                ),
            }
        )
    elif type(adapter) is CanonicalReducedPublicTreeAdapter:
        instance_namespace = getattr(adapter, "__dict__", {})
        shadowed = sorted(
            method
            for method in _REDUCED_CHECKPOINT_METHODS
            if method in instance_namespace
        )
        class_drift = sorted(
            method
            for method, canonical in _CANONICAL_REDUCED_METHODS.items()
            if (
                inspect.getattr_static(type(adapter), method).fget
                if isinstance(
                    inspect.getattr_static(type(adapter), method), property
                )
                else getattr(
                    inspect.getattr_static(type(adapter), method),
                    "__func__",
                    inspect.getattr_static(type(adapter), method),
                )
            )
            is not canonical
        )
        global_drift = sorted(
            name
            for name, canonical in _CANONICAL_REDUCED_GLOBALS.items()
            if globals().get(name) is not canonical
        )
        if shadowed or class_drift or global_drift:
            raise MultiRootMccfrCheckpointError(
                "canonical reduced checkpoint adapter has overridden semantics: "
                f"instance={shadowed}, class={class_drift}, globals={global_drift}"
            )
        reduced_manifest = _generic_adapter_manifest(adapter)
        common.update(
            {
                "adapter_kind": "canonical_reduced_public_tree_v1",
                "canonical_reduced_manifest": reduced_manifest,
                "canonical_reduced_manifest_sha256": (
                    _checkpoint_content_sha256(reduced_manifest)
                ),
                "live_runtime_semantic_binding": (
                    _reduced_runtime_semantic_binding()
                ),
                "range_content_sha256": None,
                "range_build_sha256": None,
            }
        )
    else:
        raise MultiRootMccfrCheckpointError(
            "checkpoint/resume is disabled for arbitrary generic adapters; "
            "use FullCardGenerativeAdapter or the repository-owned canonical "
            "reduced adapter"
        )
    common["binding_sha256"] = _checkpoint_content_sha256(common)
    return common


def _checkpoint_adapter_bindings(
    entries: Sequence["MultiRootChanceEntry"],
) -> tuple[dict[str, Any], ...]:
    return tuple(_adapter_checkpoint_binding(entry) for entry in entries)


def _root_prior_manifest(
    entries: Sequence["MultiRootChanceEntry"],
) -> dict[str, Any]:
    return {
        "schema": "ofc_multi_root_exact_prior/v1",
        "sampling_contract": SUPER_ROOT_SAMPLING_CONTRACT,
        "roots": [
            {
                "root_id_sha256": entry.root_id_sha256,
                "prior_mass_exact": _fraction_text(entry.prior_mass),
                "observation_sha256": entry.adapter.observation.digest(),
                "conditional_particle_count": len(entry.adapter.root_distribution),
                "range_content_sha256": getattr(
                    getattr(entry.adapter, "root_range", None),
                    "range_content_sha256",
                    None,
                ),
                "range_build_sha256": getattr(
                    getattr(entry.adapter, "root_range", None),
                    "range_build_sha256",
                    None,
                ),
            }
            for entry in entries
        ],
    }


def _checkpoint_solver_config(max_infosets: int) -> dict[str, Any]:
    return {
        "method": MULTI_ROOT_SOLVER_METHOD,
        "sampling_scheme": "alternating_external_sampling",
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


@dataclass(frozen=True)
class MultiRootChanceEntry:
    """One conditional root adapter and its exact ex-ante chance mass."""

    root_id: str
    adapter: MultiRootTraversalAdapter
    prior_mass: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.root_id, str) or not self.root_id.strip():
            raise ValueError("root_id must be a non-empty string")
        if self.root_id != self.root_id.strip():
            raise ValueError("root_id must not contain surrounding whitespace")
        if not isinstance(self.prior_mass, Fraction):
            raise TypeError("root prior mass must be fractions.Fraction")
        if self.prior_mass <= 0:
            raise ValueError("root prior mass must be positive")
        if self.adapter is None:
            raise TypeError("root adapter must not be None")

    @property
    def root_id_sha256(self) -> str:
        """Opaque audit identity; it is never a policy-table component."""

        return _root_id_sha256(self.root_id)


def _adapter_public_context(observation: InfoSetKey) -> tuple[Any, ...]:
    """Return root fields that must be common before private information."""

    return (
        observation.contract_version,
        observation.actor,
        observation.turn,
        observation.phase,
        observation.board_bb,
        observation.board_btn,
        observation.public_action_history,
        observation.fantasy_state,
    )


def _validate_adapter(entry: MultiRootChanceEntry) -> InfoSetKey:
    adapter = entry.adapter
    observation = getattr(adapter, "observation", None)
    if not isinstance(observation, InfoSetKey):
        raise TypeError("each root adapter must expose an InfoSetKey observation")
    if observation.contract_version != POSITION_CONTRACT_VERSION:
        raise ValueError("multi-root MCCFR requires bb_first_v1")
    # Re-run InfoSetKey's private-field serialization defense.
    observation.canonical_json()

    for method_name in (
        "sample_root_for_traversal",
        "information_key",
        "legal_actions",
        "apply_action_id",
        "sample_next_draw",
        "terminal_utility_bb",
        "sampling_audit",
    ):
        if not callable(getattr(adapter, method_name, None)):
            raise TypeError(f"root adapter must provide {method_name}()")

    distribution = getattr(adapter, "root_distribution", None)
    if not isinstance(distribution, tuple) or not distribution:
        raise TypeError("root adapter must expose a non-empty root_distribution tuple")
    commitments: list[str] = []
    conditional_masses: list[Fraction] = []
    for row in distribution:
        if not isinstance(row, tuple) or len(row) != 2:
            raise TypeError("root_distribution rows must be (commitment, Fraction)")
        commitment, probability = row
        if not isinstance(commitment, str) or not commitment:
            raise ValueError("root particle commitments must be non-empty strings")
        if not isinstance(probability, Fraction):
            raise TypeError("root posterior probabilities must be fractions.Fraction")
        if probability <= 0:
            raise ValueError("root posterior probabilities must be positive")
        commitments.append(commitment)
        conditional_masses.append(probability)
    if len(commitments) != len(set(commitments)):
        raise ValueError("root adapter particle commitments must be unique")
    if sum(conditional_masses, Fraction(0, 1)) != 1:
        raise ValueError("root adapter posterior probabilities must sum exactly to one")

    audit = adapter.sampling_audit()
    if not isinstance(audit, Mapping):
        raise TypeError("root adapter sampling_audit() must return a mapping")
    for key in ("traversals", "root_posterior_samples", "future_draw_samples"):
        value = audit.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"root adapter audit {key!r} must be a non-negative int")
    return observation


def _validated_entries(
    entries: Sequence[MultiRootChanceEntry],
) -> tuple[MultiRootChanceEntry, ...]:
    supplied = tuple(entries)
    if len(supplied) < 2:
        raise ValueError("shared multi-root MCCFR requires at least two roots")
    if any(not isinstance(entry, MultiRootChanceEntry) for entry in supplied):
        raise TypeError("roots must contain MultiRootChanceEntry values")
    if len({entry.root_id for entry in supplied}) != len(supplied):
        raise ValueError("root_id values must be unique")
    if len({id(entry.adapter) for entry in supplied}) != len(supplied):
        raise ValueError("each chance root must own a distinct adapter instance")
    ordered = tuple(sorted(supplied, key=lambda entry: entry.root_id))
    if sum((entry.prior_mass for entry in ordered), Fraction(0, 1)) != 1:
        raise ValueError("multi-root prior masses must sum exactly to one")

    observations = tuple(_validate_adapter(entry) for entry in ordered)
    expected_context = _adapter_public_context(observations[0])
    for observation in observations[1:]:
        if _adapter_public_context(observation) != expected_context:
            raise ValueError(
                "root adapters must share actor/phase and public root context; "
                "only acting-player private recall/current draw may differ"
            )
    return ordered


def _is_terminal(adapter: MultiRootTraversalAdapter, value: Any) -> bool:
    if isinstance(value, PublicTreeTerminalState):
        return True
    predicate = getattr(adapter, "is_terminal_result", None)
    if predicate is None:
        return False
    result = predicate(value)
    if not isinstance(result, bool):
        raise TypeError("adapter is_terminal_result() must return bool")
    return result


@dataclass
class _MutableMultiRootStats:
    traversals: int = 0
    bb_traversals: int = 0
    btn_traversals: int = 0
    super_root_samples: int = 0
    conditional_root_posterior_samples: int = 0
    root_samples_by_index: dict[int, int] = field(default_factory=dict)
    future_draw_samples: int = 0
    future_draw_samples_by_next_phase: dict[str, int] = field(default_factory=dict)
    decision_visits: int = 0
    traverser_decision_visits: int = 0
    terminal_visits: int = 0
    traverser_actions_expanded: int = 0
    opponent_action_samples: int = 0
    opponent_action_cache_hits: int = 0
    strategy_sum_updates: int = 0
    infosets_created: int = 0

    def begin(self, traverser: str, root_index: int) -> None:
        self.traversals += 1
        self.super_root_samples += 1
        self.conditional_root_posterior_samples += 1
        self.root_samples_by_index[root_index] = (
            self.root_samples_by_index.get(root_index, 0) + 1
        )
        if traverser == "bb":
            self.bb_traversals += 1
        elif traverser == "btn":
            self.btn_traversals += 1
        else:  # pragma: no cover - internal defense
            raise ValueError("traverser must be bb or btn")

    def record_future_draw(self, next_phase: str) -> None:
        self.future_draw_samples += 1
        phase = str(next_phase)
        self.future_draw_samples_by_next_phase[phase] = (
            self.future_draw_samples_by_next_phase.get(phase, 0) + 1
        )

    def snapshot(
        self,
        entries: Sequence[MultiRootChanceEntry],
    ) -> Mapping[str, Any]:
        root_counts = {
            entries[index].root_id_sha256: self.root_samples_by_index.get(index, 0)
            for index in range(len(entries))
        }
        return MappingProxyType(
            {
                "traversals": self.traversals,
                "traversals_by_actor": {
                    "bb": self.bb_traversals,
                    "btn": self.btn_traversals,
                },
                "super_root_samples": self.super_root_samples,
                "conditional_root_posterior_samples": (
                    self.conditional_root_posterior_samples
                ),
                "root_samples_by_opaque_id": dict(sorted(root_counts.items())),
                "distinct_root_adapters_sampled": sum(
                    count > 0 for count in root_counts.values()
                ),
                "future_draw_samples": self.future_draw_samples,
                "future_draw_samples_by_next_phase": dict(
                    sorted(self.future_draw_samples_by_next_phase.items())
                ),
                "decision_visits": self.decision_visits,
                "traverser_decision_visits": self.traverser_decision_visits,
                "terminal_visits": self.terminal_visits,
                "traverser_actions_expanded": self.traverser_actions_expanded,
                "opponent_action_samples": self.opponent_action_samples,
                "opponent_action_cache_hits": self.opponent_action_cache_hits,
                "strategy_sum_updates": self.strategy_sum_updates,
                "infosets_created": self.infosets_created,
            }
        )

    @classmethod
    def from_snapshot(
        cls,
        raw: Any,
        entries: Sequence[MultiRootChanceEntry],
    ) -> "_MutableMultiRootStats":
        expected = {
            "traversals",
            "traversals_by_actor",
            "super_root_samples",
            "conditional_root_posterior_samples",
            "root_samples_by_opaque_id",
            "distinct_root_adapters_sampled",
            "future_draw_samples",
            "future_draw_samples_by_next_phase",
            "decision_visits",
            "traverser_decision_visits",
            "terminal_visits",
            "traverser_actions_expanded",
            "opponent_action_samples",
            "opponent_action_cache_hits",
            "strategy_sum_updates",
            "infosets_created",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise MultiRootMccfrCheckpointError(
                "checkpoint sampling statistics schema is invalid"
            )

        def count(value: Any, *, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MultiRootMccfrCheckpointError(
                    f"checkpoint {label} must be a non-negative integer"
                )
            return value

        traversals_by_actor = raw["traversals_by_actor"]
        if not isinstance(traversals_by_actor, dict) or set(
            traversals_by_actor
        ) != {"bb", "btn"}:
            raise MultiRootMccfrCheckpointError(
                "checkpoint traversals_by_actor schema is invalid"
            )
        opaque_counts = raw["root_samples_by_opaque_id"]
        expected_ids = tuple(entry.root_id_sha256 for entry in entries)
        if not isinstance(opaque_counts, dict) or set(opaque_counts) != set(
            expected_ids
        ):
            raise MultiRootMccfrCheckpointError(
                "checkpoint root sample identities do not match current roots"
            )
        root_counts = {
            index: count(
                opaque_counts[opaque_id],
                label=f"root sample count {opaque_id}",
            )
            for index, opaque_id in enumerate(expected_ids)
        }
        phase_counts = raw["future_draw_samples_by_next_phase"]
        if not isinstance(phase_counts, dict) or any(
            not isinstance(phase, str) or not phase
            for phase in phase_counts
        ):
            raise MultiRootMccfrCheckpointError(
                "checkpoint future draw phase counts are invalid"
            )
        future_by_phase = {
            phase: count(value, label=f"future draw count {phase!r}")
            for phase, value in phase_counts.items()
        }
        distinct = count(
            raw["distinct_root_adapters_sampled"],
            label="distinct root adapter count",
        )
        if distinct != sum(value > 0 for value in root_counts.values()):
            raise MultiRootMccfrCheckpointError(
                "checkpoint distinct root adapter count is inconsistent"
            )
        return cls(
            traversals=count(raw["traversals"], label="traversals"),
            bb_traversals=count(
                traversals_by_actor["bb"], label="BB traversals"
            ),
            btn_traversals=count(
                traversals_by_actor["btn"], label="BTN traversals"
            ),
            super_root_samples=count(
                raw["super_root_samples"], label="super-root samples"
            ),
            conditional_root_posterior_samples=count(
                raw["conditional_root_posterior_samples"],
                label="conditional root posterior samples",
            ),
            root_samples_by_index=root_counts,
            future_draw_samples=count(
                raw["future_draw_samples"], label="future draw samples"
            ),
            future_draw_samples_by_next_phase=future_by_phase,
            decision_visits=count(
                raw["decision_visits"], label="decision visits"
            ),
            traverser_decision_visits=count(
                raw["traverser_decision_visits"],
                label="traverser decision visits",
            ),
            terminal_visits=count(
                raw["terminal_visits"], label="terminal visits"
            ),
            traverser_actions_expanded=count(
                raw["traverser_actions_expanded"],
                label="traverser actions expanded",
            ),
            opponent_action_samples=count(
                raw["opponent_action_samples"],
                label="opponent action samples",
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
class MultiRootExternalSamplingMccfrResult:
    iterations: int
    traversals: int
    seed: int
    root_count: int
    encountered_infosets: int
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    current_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    cumulative_regret_plus: Mapping[InfoSetKey, Mapping[str, float]]
    infoset_root_support_count: Mapping[InfoSetKey, int]
    shared_across_roots_infosets: tuple[InfoSetKey, ...]
    average_strategy_json: str
    current_strategy_json: str
    average_strategy_sha256: str
    current_strategy_sha256: str
    sampling_stats: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedMultiRootCheckpointSnapshot:
    """Live-verified immutable view of one persisted solver checkpoint.

    The ordinary solver result deliberately exposes only root-support counts.
    Teacher-label provenance additionally needs the per-root visit counters
    stored in the checkpoint.  This view is produced only after the checkpoint
    has been restored against the current roots, adapters, ranges, sources and
    runtime semantics, and after every reconstructed result table has been
    compared with the supplied in-memory result.
    """

    checkpoint_sha256: str
    checkpoint_file_sha256: str
    completed_iterations: int
    seed: int
    solver_config_sha256: str
    source_binding_sha256: str
    prior_manifest_sha256: str
    average_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    current_strategy: Mapping[InfoSetKey, Mapping[str, float]]
    cumulative_regret_plus: Mapping[InfoSetKey, Mapping[str, float]]
    sampling_stats: Mapping[str, Any]
    infoset_root_support_opaque_ids: Mapping[InfoSetKey, tuple[str, ...]]
    infoset_root_visit_counts_by_opaque_id: Mapping[
        InfoSetKey, Mapping[str, int]
    ]


def _freeze_checkpoint_snapshot_value(value: Any) -> Any:
    """Recursively freeze JSON-like checkpoint evidence containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_checkpoint_snapshot_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_checkpoint_snapshot_value(item) for item in value)
    return value


def _cumulative_stats_are_consistent(
    stats: _MutableMultiRootStats,
    *,
    completed_iterations: int,
    infoset_count: int,
) -> bool:
    expected_traversals = 2 * completed_iterations
    opponent_visits = (
        stats.opponent_action_samples + stats.opponent_action_cache_hits
    )
    action_evaluations = stats.traverser_actions_expanded + opponent_visits
    return (
        stats.traversals == expected_traversals
        and stats.bb_traversals == completed_iterations
        and stats.btn_traversals == completed_iterations
        and stats.super_root_samples == expected_traversals
        and stats.conditional_root_posterior_samples == expected_traversals
        and sum(stats.root_samples_by_index.values()) == expected_traversals
        and sum(stats.future_draw_samples_by_next_phase.values())
        == stats.future_draw_samples
        and stats.infosets_created == infoset_count
        and stats.decision_visits
        == stats.traverser_decision_visits + opponent_visits
        and stats.traverser_actions_expanded >= stats.traverser_decision_visits
        and stats.terminal_visits + stats.future_draw_samples
        == action_evaluations
        and stats.strategy_sum_updates <= opponent_visits
        and stats.decision_visits >= expected_traversals
    )


@dataclass(frozen=True)
class _RestoredMultiRootCheckpoint:
    completed_iterations: int
    tables: _DynamicMccfrTables
    rng_state: tuple[Any, ...]
    stats: _MutableMultiRootStats
    infoset_root_sources: dict[InfoSetKey, set[int]]
    infoset_root_visit_counts: dict[InfoSetKey, dict[int, int]]
    checkpoint_sha256: str


def _infoset_root_visits_are_consistent(
    *,
    stable_keys: Sequence[InfoSetKey],
    entries: Sequence[MultiRootChanceEntry],
    stats: _MutableMultiRootStats,
    infoset_root_sources: Mapping[InfoSetKey, set[int]],
    infoset_root_visit_counts: Mapping[InfoSetKey, Mapping[int, int]],
) -> bool:
    key_set = set(stable_keys)
    if (
        set(infoset_root_sources) != key_set
        or set(infoset_root_visit_counts) != key_set
    ):
        return False
    total_visits = 0
    visits_by_root = {index: 0 for index in range(len(entries))}
    for key in stable_keys:
        counts = infoset_root_visit_counts[key]
        if (
            not counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or not isinstance(root_index, int)
                or not 0 <= root_index < len(entries)
                for root_index, count in counts.items()
            )
            or infoset_root_sources[key] != set(counts)
        ):
            return False
        for root_index, count in counts.items():
            total_visits += count
            visits_by_root[root_index] += count
    if total_visits != stats.decision_visits:
        return False
    for root_index, entry in enumerate(entries):
        root_samples = stats.root_samples_by_index.get(root_index, 0)
        root_key_visits = infoset_root_visit_counts.get(
            entry.adapter.observation, {}
        ).get(root_index, 0)
        if root_key_visits != root_samples or visits_by_root[root_index] < root_samples:
            return False
    return True


def _checkpoint_float_hex(value: float, *, label: str) -> str:
    try:
        return _single_root_float_hex(value, label=label)
    except (TypeError, ValueError) as exc:
        raise MultiRootMccfrCheckpointError(
            f"checkpoint {label} must be finite"
        ) from exc


def _checkpoint_float_from_hex(
    raw: Any,
    *,
    label: str,
) -> float:
    try:
        return _single_root_float_from_hex(
            raw,
            label=label,
            non_negative=True,
        )
    except (TypeError, ValueError) as exc:
        raise MultiRootMccfrCheckpointError(
            f"checkpoint {label} must be a finite non-negative hex float"
        ) from exc


def _checkpoint_encode_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    try:
        return _single_root_encode_rng_state(state)
    except (TypeError, ValueError) as exc:
        raise MultiRootMccfrCheckpointError(
            "cannot encode multi-root checkpoint RNG state"
        ) from exc


def _checkpoint_decode_rng_state(raw: Any) -> tuple[Any, ...]:
    try:
        return _single_root_decode_rng_state(raw)
    except (TypeError, ValueError) as exc:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint RNG state is invalid"
        ) from exc


def _validate_descendant_for_root(root: InfoSetKey, key: InfoSetKey) -> None:
    phase_order = {
        phase: index
        for index, phase in enumerate(
            ("t3_first", "t3_second", "t4_first", "t4_second")
        )
    }
    if phase_order[key.phase] < phase_order[root.phase]:
        raise MultiRootMccfrCheckpointError(
            "checkpoint infoset precedes its claimed root"
        )
    if key.public_action_history[: len(root.public_action_history)] != (
        root.public_action_history
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint infoset is outside its claimed public root tree"
        )
    if key.phase == root.phase and key != root:
        raise MultiRootMccfrCheckpointError(
            "checkpoint root-phase infoset does not equal its claimed root"
        )


def _reduced_action_map(binding: Mapping[str, Any]) -> dict[InfoSetKey, tuple[str, ...]]:
    manifest = binding.get("canonical_reduced_manifest")
    if not isinstance(manifest, dict):
        raise MultiRootMccfrCheckpointError(
            "canonical reduced adapter binding manifest is unavailable"
        )
    result: dict[InfoSetKey, tuple[str, ...]] = {}
    for row in manifest["infoset_actions"]:
        key = _parse_checkpoint_infoset(row["infoset_canonical_json"])
        result[key] = tuple(row["stable_action_ids"])
    return result


def _expected_action_ids_for_support(
    key: InfoSetKey,
    support: set[int],
    *,
    entries: Sequence[MultiRootChanceEntry],
    adapter_bindings: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if not support:
        raise MultiRootMccfrCheckpointError(
            "checkpoint infoset must have non-empty root support"
        )
    expected: tuple[str, ...] | None = None
    for root_index in sorted(support):
        if not 0 <= root_index < len(entries):
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset root support index is invalid"
            )
        _validate_descendant_for_root(entries[root_index].adapter.observation, key)
        binding = adapter_bindings[root_index]
        if binding["adapter_kind"] == "full_card_generative_t3_t4_v1":
            try:
                action_ids = _single_root_legal_action_ids(key)
            except (TypeError, ValueError) as exc:
                raise MultiRootMccfrCheckpointError(
                    "checkpoint full-card infoset action contract is invalid"
                ) from exc
        elif binding["adapter_kind"] == "canonical_reduced_public_tree_v1":
            action_ids = _reduced_action_map(binding).get(key)
            if action_ids is None:
                raise MultiRootMccfrCheckpointError(
                    "checkpoint infoset is absent from canonical reduced adapter "
                    "action binding"
                )
        else:  # pragma: no cover - constructed binding defense
            raise MultiRootMccfrCheckpointError(
                "checkpoint adapter kind is unsupported"
            )
        if expected is None:
            expected = action_ids
        elif expected != action_ids:
            raise MultiRootMccfrCheckpointError(
                "shared checkpoint infoset has incompatible root action contracts"
            )
    if expected is None:  # pragma: no cover - non-empty support defense
        raise AssertionError("checkpoint action validation saw no support roots")
    return expected


def _multi_root_checkpoint_payload(
    *,
    entries: Sequence[MultiRootChanceEntry],
    completed_iterations: int,
    seed: int,
    linear_averaging: bool,
    tables: _DynamicMccfrTables,
    rng_state: tuple[Any, ...],
    stats: _MutableMultiRootStats,
    infoset_root_sources: Mapping[InfoSetKey, set[int]],
    infoset_root_visit_counts: Mapping[InfoSetKey, Mapping[int, int]],
    prior_manifest: Mapping[str, Any],
    adapter_bindings: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if completed_iterations <= 0:
        raise MultiRootMccfrCheckpointError(
            "checkpoint requires at least one completed iteration"
        )
    opaque_ids = tuple(entry.root_id_sha256 for entry in entries)
    if not _infoset_root_visits_are_consistent(
        stable_keys=tables.stable_keys,
        entries=entries,
        stats=stats,
        infoset_root_sources=infoset_root_sources,
        infoset_root_visit_counts=infoset_root_visit_counts,
    ):
        raise MultiRootMccfrCheckpointError(
            "live infoset root-support visit accounting is inconsistent"
        )
    table_rows: list[dict[str, Any]] = []
    for key in tables.stable_keys:
        support = set(infoset_root_sources.get(key, set()))
        expected_actions = _expected_action_ids_for_support(
            key,
            support,
            entries=entries,
            adapter_bindings=adapter_bindings,
        )
        if expected_actions != tables.action_ids[key]:
            raise MultiRootMccfrCheckpointError(
                "live infoset actions do not match checkpoint adapter binding"
            )
        table_rows.append(
            {
                "infoset_canonical_json": key.canonical_json(),
                "infoset_sha256": key.digest(),
                "actor": tables.actors[key],
                "stable_action_ids": list(tables.action_ids[key]),
                "root_support_opaque_ids": sorted(
                    opaque_ids[index] for index in support
                ),
                "root_visit_counts_by_opaque_id": {
                    opaque_ids[index]: infoset_root_visit_counts[key][index]
                    for index in sorted(support, key=lambda value: opaque_ids[value])
                },
                "regret_plus_hex": [
                    _checkpoint_float_hex(
                        tables.regrets[key][action_id],
                        label="cumulative regret",
                    )
                    for action_id in tables.action_ids[key]
                ],
                "strategy_sum_hex": [
                    _checkpoint_float_hex(
                        tables.strategy_sum[key][action_id],
                        label="strategy sum",
                    )
                    for action_id in tables.action_ids[key]
                ],
            }
        )
    if not table_rows:
        raise MultiRootMccfrCheckpointError(
            "checkpoint requires at least one encountered infoset"
        )
    adapter_binding_rows = list(adapter_bindings)
    prior_snapshot = _checkpoint_snapshot(
        dict(prior_manifest), label="exact prior manifest"
    )
    source_snapshot = _checkpoint_snapshot(
        dict(source_binding), label="solver source binding"
    )
    encoded_rng_state = _checkpoint_encode_rng_state(rng_state)
    sampling_snapshot = dict(stats.snapshot(entries))
    runtime_state_binding_sha256 = _checkpoint_content_sha256(
        {
            "completed_iterations": completed_iterations,
            "tables": table_rows,
            "rng_state": encoded_rng_state,
            "sampling_stats": sampling_snapshot,
        }
    )
    return {
        "format": MULTI_ROOT_SOLVER_STATE_FORMAT,
        "completed_iterations": completed_iterations,
        "seed": seed,
        "linear_averaging": linear_averaging,
        "solver_config": _checkpoint_solver_config(tables.max_infosets),
        "prior_manifest": prior_snapshot,
        "prior_manifest_sha256": _checkpoint_content_sha256(prior_snapshot),
        "adapter_bindings": adapter_binding_rows,
        "adapter_bindings_sha256": _checkpoint_content_sha256(
            adapter_binding_rows
        ),
        "source_binding": source_snapshot,
        "source_binding_sha256": _checkpoint_content_sha256(source_snapshot),
        "tables": table_rows,
        "rng_state": encoded_rng_state,
        "sampling_stats": sampling_snapshot,
        "runtime_state_binding_sha256": runtime_state_binding_sha256,
    }


def _multi_root_checkpoint_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": MULTI_ROOT_CHECKPOINT_FORMAT,
        "checkpoint_sha256": _checkpoint_content_sha256(payload),
        "payload": payload,
    }


def _atomic_write_multi_root_checkpoint(
    path: str | os.PathLike[str],
    envelope: Mapping[str, Any],
) -> None:
    target = _checkpoint_path(path, for_read=False)
    serialized = _checkpoint_json_bytes(envelope) + b"\n"
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


def _read_multi_root_checkpoint(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    source = _checkpoint_path(path, for_read=True)
    try:
        serialized = source.read_bytes()
        text = serialized.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise MultiRootMccfrCheckpointError(
            f"cannot read multi-root checkpoint: {exc}"
        ) from exc
    try:
        envelope = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_checkpoint_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except MultiRootMccfrCheckpointError:
        raise
    except json.JSONDecodeError as exc:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint is not valid canonical UTF-8 JSON"
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "format",
        "checkpoint_sha256",
        "payload",
    }:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint envelope schema is invalid"
        )
    if envelope["format"] != MULTI_ROOT_CHECKPOINT_FORMAT:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint format is incompatible"
        )
    supplied_hash = envelope["checkpoint_sha256"]
    if (
        not isinstance(supplied_hash, str)
        or len(supplied_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_hash)
    ):
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint SHA-256 field is invalid"
        )
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint payload must be an object"
        )
    if supplied_hash != _checkpoint_content_sha256(payload):
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint content SHA-256 mismatch"
        )
    canonical_bytes = _checkpoint_json_bytes(envelope) + b"\n"
    if serialized != canonical_bytes:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint file is not canonically serialized"
        )
    return payload, supplied_hash


def _restore_multi_root_checkpoint(
    path: str | os.PathLike[str],
    *,
    entries: Sequence[MultiRootChanceEntry],
    seed: int,
    linear_averaging: bool,
    max_infosets: int,
    prior_manifest: Mapping[str, Any],
    adapter_bindings: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
    expected_checkpoint_sha256: str | None,
) -> _RestoredMultiRootCheckpoint:
    if (
        not isinstance(expected_checkpoint_sha256, str)
        or len(expected_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_checkpoint_sha256
        )
    ):
        raise MultiRootMccfrCheckpointError(
            "resume requires an external expected_checkpoint_sha256 binding"
        )
    payload, checkpoint_sha256 = _read_multi_root_checkpoint(path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise MultiRootMccfrCheckpointError(
            "checkpoint does not match external expected_checkpoint_sha256"
        )
    expected_payload_keys = {
        "format",
        "completed_iterations",
        "seed",
        "linear_averaging",
        "solver_config",
        "prior_manifest",
        "prior_manifest_sha256",
        "adapter_bindings",
        "adapter_bindings_sha256",
        "source_binding",
        "source_binding_sha256",
        "tables",
        "rng_state",
        "sampling_stats",
        "runtime_state_binding_sha256",
    }
    if set(payload) != expected_payload_keys:
        raise MultiRootMccfrCheckpointError(
            "multi-root checkpoint payload schema is invalid"
        )
    if payload["format"] != MULTI_ROOT_SOLVER_STATE_FORMAT:
        raise MultiRootMccfrCheckpointError(
            "multi-root solver state format is incompatible"
        )
    completed_iterations = payload["completed_iterations"]
    if (
        isinstance(completed_iterations, bool)
        or not isinstance(completed_iterations, int)
        or completed_iterations <= 0
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint completed_iterations must be positive"
        )
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] != seed
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint seed does not match requested seed"
        )
    if (
        not isinstance(payload["linear_averaging"], bool)
        or payload["linear_averaging"] is not linear_averaging
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint linear_averaging setting does not match"
        )
    if payload["solver_config"] != _checkpoint_solver_config(max_infosets):
        raise MultiRootMccfrCheckpointError(
            "checkpoint solver configuration is incompatible"
        )
    expected_runtime_state_binding = _checkpoint_content_sha256(
        {
            "completed_iterations": completed_iterations,
            "tables": payload["tables"],
            "rng_state": payload["rng_state"],
            "sampling_stats": payload["sampling_stats"],
        }
    )
    if payload["runtime_state_binding_sha256"] != (
        expected_runtime_state_binding
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint runtime state binding SHA-256 mismatch"
        )

    current_prior = _checkpoint_snapshot(
        dict(prior_manifest), label="current exact prior manifest"
    )
    if (
        payload["prior_manifest_sha256"]
        != _checkpoint_content_sha256(payload["prior_manifest"])
        or payload["prior_manifest"] != current_prior
        or payload["prior_manifest_sha256"]
        != _checkpoint_content_sha256(current_prior)
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint exact prior or root/range identity does not match"
        )
    current_bindings = list(adapter_bindings)
    if (
        payload["adapter_bindings_sha256"]
        != _checkpoint_content_sha256(payload["adapter_bindings"])
        or payload["adapter_bindings"] != current_bindings
        or payload["adapter_bindings_sha256"]
        != _checkpoint_content_sha256(current_bindings)
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint adapter/range/action/terminal binding does not match"
        )
    current_source = _checkpoint_snapshot(
        dict(source_binding), label="current solver source binding"
    )
    if (
        payload["source_binding_sha256"]
        != _checkpoint_content_sha256(payload["source_binding"])
        or payload["source_binding"] != current_source
        or payload["source_binding_sha256"]
        != _checkpoint_content_sha256(current_source)
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint solver/source/action/scoring/rules binding does not match"
        )

    raw_tables = payload["tables"]
    if not isinstance(raw_tables, list) or not raw_tables:
        raise MultiRootMccfrCheckpointError(
            "checkpoint infoset tables must be a non-empty list"
        )
    if len(raw_tables) > max_infosets:
        raise MultiRootMccfrCheckpointError(
            "checkpoint infoset count exceeds max_infosets"
        )
    tables = _DynamicMccfrTables(max_infosets)
    infoset_root_sources: dict[InfoSetKey, set[int]] = {}
    infoset_root_visit_counts: dict[InfoSetKey, dict[int, int]] = {}
    table_schema = {
        "infoset_canonical_json",
        "infoset_sha256",
        "actor",
        "stable_action_ids",
        "root_support_opaque_ids",
        "root_visit_counts_by_opaque_id",
        "regret_plus_hex",
        "strategy_sum_hex",
    }
    opaque_to_index = {
        entry.root_id_sha256: index for index, entry in enumerate(entries)
    }
    prior_order: tuple[str, str] | None = None
    for raw_table in raw_tables:
        if not isinstance(raw_table, dict) or set(raw_table) != table_schema:
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset table schema is invalid"
            )
        key = _parse_checkpoint_infoset(raw_table["infoset_canonical_json"])
        order = (key.digest(), key.canonical_json())
        if prior_order is not None and order <= prior_order:
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset table order is not canonical"
            )
        prior_order = order
        if raw_table["infoset_sha256"] != key.digest():
            raise MultiRootMccfrCheckpointError(
                "checkpoint InfoSetKey digest does not match canonical JSON"
            )
        if raw_table["actor"] != key.actor:
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset actor is incompatible"
            )
        support_ids = raw_table["root_support_opaque_ids"]
        if (
            not isinstance(support_ids, list)
            or not support_ids
            or any(not isinstance(value, str) for value in support_ids)
            or support_ids != sorted(support_ids)
            or len(support_ids) != len(set(support_ids))
            or not set(support_ids).issubset(opaque_to_index)
        ):
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset root support is invalid"
            )
        support = {opaque_to_index[value] for value in support_ids}
        raw_visit_counts = raw_table["root_visit_counts_by_opaque_id"]
        if (
            not isinstance(raw_visit_counts, dict)
            or set(raw_visit_counts) != set(support_ids)
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for count in raw_visit_counts.values()
            )
        ):
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset root support visit counts are invalid"
            )
        visit_counts = {
            opaque_to_index[opaque_id]: raw_visit_counts[opaque_id]
            for opaque_id in support_ids
        }
        action_ids = _expected_action_ids_for_support(
            key,
            support,
            entries=entries,
            adapter_bindings=adapter_bindings,
        )
        if raw_table["stable_action_ids"] != list(action_ids):
            raise MultiRootMccfrCheckpointError(
                "checkpoint stable action IDs do not match adapter contract"
            )
        regret_values = raw_table["regret_plus_hex"]
        strategy_values = raw_table["strategy_sum_hex"]
        if (
            not isinstance(regret_values, list)
            or not isinstance(strategy_values, list)
            or len(regret_values) != len(action_ids)
            or len(strategy_values) != len(action_ids)
        ):
            raise MultiRootMccfrCheckpointError(
                "checkpoint action-vector length is incompatible"
            )
        try:
            tables.ensure(key, action_ids)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise MultiRootMccfrCheckpointError(
                "checkpoint infoset registry is incompatible"
            ) from exc
        tables.regrets[key] = {
            action_id: _checkpoint_float_from_hex(
                regret_values[index], label="cumulative regret"
            )
            for index, action_id in enumerate(action_ids)
        }
        tables.strategy_sum[key] = {
            action_id: _checkpoint_float_from_hex(
                strategy_values[index], label="strategy sum"
            )
            for index, action_id in enumerate(action_ids)
        }
        infoset_root_sources[key] = support
        infoset_root_visit_counts[key] = visit_counts

    stats = _MutableMultiRootStats.from_snapshot(
        payload["sampling_stats"], entries
    )
    if not _cumulative_stats_are_consistent(
        stats,
        completed_iterations=completed_iterations,
        infoset_count=len(tables.action_ids),
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint cumulative sampling statistics are inconsistent"
        )
    if not _infoset_root_visits_are_consistent(
        stable_keys=tables.stable_keys,
        entries=entries,
        stats=stats,
        infoset_root_sources=infoset_root_sources,
        infoset_root_visit_counts=infoset_root_visit_counts,
    ):
        raise MultiRootMccfrCheckpointError(
            "checkpoint infoset root-support visit accounting is inconsistent"
        )
    for root_index, sample_count in stats.root_samples_by_index.items():
        observation = entries[root_index].adapter.observation
        if sample_count > 0:
            if observation not in tables.action_ids:
                raise MultiRootMccfrCheckpointError(
                    "sampled checkpoint root observation is absent from tables"
                )
            if root_index not in infoset_root_sources[observation]:
                raise MultiRootMccfrCheckpointError(
                    "sampled checkpoint root lacks its root-support evidence"
                )
    for key, support in infoset_root_sources.items():
        for root_index in support:
            if stats.root_samples_by_index.get(root_index, 0) <= 0:
                raise MultiRootMccfrCheckpointError(
                    "infoset support claims a root that was never sampled"
                )
        if key.phase == entries[0].adapter.observation.phase:
            expected_root_support = {
                index
                for index, entry in enumerate(entries)
                if entry.adapter.observation == key
                and stats.root_samples_by_index.get(index, 0) > 0
            }
            if support != expected_root_support:
                raise MultiRootMccfrCheckpointError(
                    "root-phase infoset support is not derivable from sampled roots"
                )

    return _RestoredMultiRootCheckpoint(
        completed_iterations=completed_iterations,
        tables=tables,
        rng_state=_checkpoint_decode_rng_state(payload["rng_state"]),
        stats=stats,
        infoset_root_sources=infoset_root_sources,
        infoset_root_visit_counts=infoset_root_visit_counts,
        checkpoint_sha256=checkpoint_sha256,
    )


def solve_multi_root_external_sampling_mccfr(
    roots: Sequence[MultiRootChanceEntry],
    *,
    iterations: int,
    seed: int,
    max_infosets: int,
    linear_averaging: bool = True,
    resume_from: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> MultiRootExternalSamplingMccfrResult:
    """Solve one exact chance mixture with a single shared policy table.

    Every complete iteration performs a BB traversal followed by a BTN
    traversal.  Each traversal samples exactly one super-root entry, then asks
    that entry's adapter to sample exactly one conditional posterior particle.
    The selected root probability and posterior probability are consumed by
    those direct categorical draws and are not multiplied into regret.

    ``iterations`` is the number of additional complete BB+BTN iterations
    when ``resume_from`` is supplied.  Resume also requires the checkpoint
    digest from an external trusted record via ``expected_checkpoint_sha256``;
    the self-declared digest inside the file is not an authentication anchor.
    Checkpoints are written only after a complete iteration, using canonical
    content-addressed JSON and atomic replacement.  Checkpoint support does
    not create a promotion path: output remains sampled evidence, not an exact
    exploitability certificate.  Arbitrary structural adapters may be used
    only without checkpointing; resumable runs accept exact-type canonical
    FullCard or repository-owned reduced-tree adapters.
    """

    entries = _validated_entries(roots)
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
    if resume_from is None and expected_checkpoint_sha256 is not None:
        raise ValueError(
            "expected_checkpoint_sha256 is valid only with resume_from"
        )

    prior_masses = tuple(entry.prior_mass for entry in entries)
    prior_manifest = _root_prior_manifest(entries)
    checkpoint_contract_required = (
        resume_from is not None or checkpoint_path is not None
    )
    if checkpoint_path is not None:
        _checkpoint_path(checkpoint_path, for_read=False)
    adapter_bindings: tuple[dict[str, Any], ...] | None = None
    source_binding: dict[str, Any] | None = None
    if checkpoint_contract_required:
        # Validate all content/source/terminal bindings before spending a long
        # run.  Non-standard adapters must explicitly provide their own tree
        # and action contract.
        adapter_bindings = _checkpoint_adapter_bindings(entries)
        source_binding = _multi_root_source_binding()

    rng = seeded_rng(seed)
    if resume_from is None:
        completed_before = 0
        tables = _DynamicMccfrTables(max_infosets)
        stats = _MutableMultiRootStats()
        infoset_root_sources: dict[InfoSetKey, set[int]] = {}
        infoset_root_visit_counts: dict[InfoSetKey, dict[int, int]] = {}
    else:
        if adapter_bindings is None or source_binding is None:  # pragma: no cover
            raise AssertionError("resume requires initialized checkpoint bindings")
        restored = _restore_multi_root_checkpoint(
            resume_from,
            entries=entries,
            seed=seed,
            linear_averaging=linear_averaging,
            max_infosets=max_infosets,
            prior_manifest=prior_manifest,
            adapter_bindings=adapter_bindings,
            source_binding=source_binding,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        completed_before = restored.completed_iterations
        tables = restored.tables
        stats = restored.stats
        infoset_root_sources = restored.infoset_root_sources
        infoset_root_visit_counts = restored.infoset_root_visit_counts
        rng.setstate(restored.rng_state)
    completed_total = completed_before + iterations
    root_samples_before_run = dict(stats.root_samples_by_index)
    future_draws_before_run = stats.future_draw_samples
    audits_before = tuple(dict(entry.adapter.sampling_audit()) for entry in entries)

    def traverse(
        adapter: MultiRootTraversalAdapter,
        state: Any,
        *,
        root_index: int,
        traverser: str,
        regret_delta: dict[InfoSetKey, dict[str, float]],
        opponent_choices: dict[InfoSetKey, str],
        average_updated: set[InfoSetKey],
        average_weight: float,
    ) -> float:
        stats.decision_visits += 1
        key = adapter.information_key(state)
        if not isinstance(key, InfoSetKey):
            raise TypeError("multi-root policy identity must be InfoSetKey")
        key.canonical_json()
        infoset_root_sources.setdefault(key, set()).add(root_index)
        root_visits = infoset_root_visit_counts.setdefault(key, {})
        root_visits[root_index] = root_visits.get(root_index, 0) + 1

        actions = adapter.legal_actions(state)
        if not isinstance(actions, tuple) or not actions:
            raise TypeError("adapter legal_actions() must return a non-empty tuple")
        action_ids = tuple(str(action_id) for action_id, _action in actions)
        if tables.ensure(key, action_ids):
            stats.infosets_created += 1
        sigma = tables.current_strategy(key)

        def action_value(action_id: str) -> float:
            physical = adapter.apply_action_id(state, action_id)
            if _is_terminal(adapter, physical):
                stats.terminal_visits += 1
                utility = float(adapter.terminal_utility_bb(physical))
                if not math.isfinite(utility):
                    raise ValueError("terminal utility must be finite")
                return utility
            sampled = adapter.sample_next_draw(physical, rng)
            next_state = getattr(sampled, "state", None)
            if next_state is None:
                raise TypeError("sample_next_draw() must return an object with state")
            next_phase = getattr(sampled, "next_phase", "adapter_transition")
            stats.record_future_draw(str(next_phase))
            return traverse(
                adapter,
                next_state,
                root_index=root_index,
                traverser=traverser,
                regret_delta=regret_delta,
                opponent_choices=opponent_choices,
                average_updated=average_updated,
                average_weight=average_weight,
            )

        if key.actor == traverser:
            stats.traverser_decision_visits += 1
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

        # On the other player's traversal, sampled visitation already carries
        # this player's reach.  Update once per InfoSetKey so multiple physical
        # histories do not manufacture extra average-strategy weight.
        if key not in average_updated:
            for action_id in action_ids:
                tables.strategy_sum[key][action_id] += average_weight * sigma[action_id]
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
            root_index = sample_exact_fraction_index(prior_masses, rng)
            selected = entries[root_index]
            root_sample = selected.adapter.sample_root_for_traversal(rng)
            root_state = getattr(root_sample, "state", None)
            if root_state is None:
                raise TypeError(
                    "sample_root_for_traversal() must return an object with state"
                )
            root_key = selected.adapter.information_key(root_state)
            if root_key != selected.adapter.observation:
                raise ValueError("root adapter sampled a state with a different InfoSetKey")
            stats.begin(traverser, root_index)
            regret_delta: dict[InfoSetKey, dict[str, float]] = {}
            traverse(
                selected.adapter,
                root_state,
                root_index=root_index,
                traverser=traverser,
                regret_delta=regret_delta,
                opponent_choices={},
                average_updated=set(),
                average_weight=average_weight,
            )
            # Aggregate all physical contributions by shared InfoSetKey, then
            # clip CFR+ once at the traversal boundary.
            for key, delta in regret_delta.items():
                if tables.actors[key] != traverser:
                    raise AssertionError("regret delta was recorded for the wrong actor")
                for action_id in tables.action_ids[key]:
                    tables.regrets[key][action_id] = max(
                        0.0,
                        tables.regrets[key][action_id] + delta[action_id],
                    )

    expected_additional_traversals = 2 * iterations
    expected_traversals = 2 * completed_total
    audits_after = tuple(dict(entry.adapter.sampling_audit()) for entry in entries)
    adapter_future_draw_delta = 0
    for index, (before, after) in enumerate(zip(audits_before, audits_after)):
        expected_root_samples = stats.root_samples_by_index.get(
            index, 0
        ) - root_samples_before_run.get(index, 0)
        traversal_delta = int(after["traversals"]) - int(before["traversals"])
        posterior_delta = int(after["root_posterior_samples"]) - int(
            before["root_posterior_samples"]
        )
        future_delta = int(after["future_draw_samples"]) - int(
            before["future_draw_samples"]
        )
        if traversal_delta != expected_root_samples:
            raise AssertionError("adapter traversal count diverged from super-root samples")
        if posterior_delta != expected_root_samples:
            raise AssertionError(
                "adapter posterior count diverged from conditional root samples"
            )
        if future_delta < 0:
            raise AssertionError("adapter future-draw audit moved backwards")
        adapter_future_draw_delta += future_delta
    if adapter_future_draw_delta != (
        stats.future_draw_samples - future_draws_before_run
    ):
        raise AssertionError("solver and adapters disagree on future-draw samples")
    if not _cumulative_stats_are_consistent(
        stats,
        completed_iterations=completed_total,
        infoset_count=len(tables.action_ids),
    ):
        raise AssertionError("multi-root MCCFR sampling state is inconsistent")
    if not _infoset_root_visits_are_consistent(
        stable_keys=tables.stable_keys,
        entries=entries,
        stats=stats,
        infoset_root_sources=infoset_root_sources,
        infoset_root_visit_counts=infoset_root_visit_counts,
    ):
        raise AssertionError(
            "multi-root MCCFR root-support visit accounting is inconsistent"
        )
    if sum(
        stats.root_samples_by_index.get(index, 0)
        - root_samples_before_run.get(index, 0)
        for index in range(len(entries))
    ) != expected_additional_traversals:
        raise AssertionError("additional super-root sample accounting is inconsistent")

    stable_keys = tables.stable_keys
    average_rows = {key: tables.average_strategy(key) for key in stable_keys}
    current_rows = {key: tables.current_strategy(key) for key in stable_keys}
    regret_rows = {key: dict(tables.regrets[key]) for key in stable_keys}
    support_rows = MappingProxyType(
        {key: len(infoset_root_sources.get(key, set())) for key in stable_keys}
    )
    shared_keys = tuple(key for key in stable_keys if support_rows[key] > 1)
    average_json = serialize_strategy_profile(average_rows)
    current_json = serialize_strategy_profile(current_rows)
    average_sha256 = _sha256_text(average_json)
    current_sha256 = _sha256_text(current_json)

    prior_manifest_sha256 = _sha256_text(_canonical_json(prior_manifest))
    all_full_card = all(
        isinstance(entry.adapter, FullCardGenerativeAdapter) for entry in entries
    )
    final_checkpoint_sha256: str | None = None
    checkpoint_source_binding_sha256: str | None = None
    if checkpoint_contract_required:
        if adapter_bindings is None or source_binding is None:  # pragma: no cover
            raise AssertionError("checkpoint bindings were not initialized")
        final_payload = _multi_root_checkpoint_payload(
            entries=entries,
            completed_iterations=completed_total,
            seed=seed,
            linear_averaging=linear_averaging,
            tables=tables,
            rng_state=rng.getstate(),
            stats=stats,
            infoset_root_sources=infoset_root_sources,
            infoset_root_visit_counts=infoset_root_visit_counts,
            prior_manifest=prior_manifest,
            adapter_bindings=adapter_bindings,
            source_binding=source_binding,
        )
        final_envelope = _multi_root_checkpoint_envelope(final_payload)
        final_checkpoint_sha256 = final_envelope["checkpoint_sha256"]
        checkpoint_source_binding_sha256 = source_binding["binding_sha256"]
        if checkpoint_path is not None:
            _atomic_write_multi_root_checkpoint(checkpoint_path, final_envelope)
    metadata = MappingProxyType(
        {
            "method": MULTI_ROOT_SOLVER_METHOD,
            "sampling_scheme": "alternating_external_sampling",
            "chance_super_root": True,
            "super_root_sampling_contract": SUPER_ROOT_SAMPLING_CONTRACT,
            "root_prior_manifest": prior_manifest,
            "root_prior_manifest_sha256": prior_manifest_sha256,
            "root_prior_mass_exact": [
                _fraction_text(entry.prior_mass) for entry in entries
            ],
            "root_prior_normalized_exact": True,
            "root_count": len(entries),
            "traverser_schedule": "bb_then_btn_each_iteration",
            "alternating_updates": True,
            "regret_matching_plus": True,
            "regret_clip_scope": "once_per_infoset_after_traversal",
            "linear_averaging": linear_averaging,
            "average_strategy_estimator": "two_player_simple_opponent_node",
            "opponent_sample_cached_per_infoset": True,
            "root_prior_sampled_once_per_traversal": True,
            "conditional_posterior_sampled_once_per_traversal": True,
            "root_probability_multiplied_after_sampling": False,
            "posterior_probability_multiplied_after_sampling": False,
            "chance_probability_multiplied_after_sampling": False,
            "joint_particle_weight_used_after_sampling": False,
            "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
            "policy_table_shared_across_all_roots": True,
            "table_key_type": "InfoSetKey",
            "table_key_contains_root_id": False,
            "table_key_contains_private_type_id": False,
            "table_key_contains_particle_commitment": False,
            "table_key_contains_remaining_cards": False,
            "strategy_serialization_contains_root_id": False,
            "strategy_serialization_contains_hidden_particle": False,
            "strategy_fusion": False,
            "independent_per_root_solve": False,
            "shared_across_roots_infoset_count": len(shared_keys),
            "compatible_full_card_adapters": all_full_card,
            "full_card": all_full_card,
            "full_card_policy_promoted": False,
            "promotion_eligible": False,
            "runtime_integrated": False,
            "hu_exact": False,
            "exact_exploitability_computed": False,
            "policy_scope": "supplied_root_set_online_tabular_solve",
            "checkpoint_artifact_purpose": [
                "algorithm_validation",
                "online_root_solve_resume",
            ],
            "global_unseen_state_policy_claim": False,
            "root_id_relabeling_is_generalization_evidence": False,
            "root_disjoint_holdout_profile_reuse_valid": False,
            "unseen_root_strength_evaluation_requires": (
                "online_solve_or_distilled_generalizing_policy"
            ),
            "strength_eval_requires_separate_solver_and_payoff_seeds": True,
            "scope_note": (
                "encountered-only sampled tabular search over the supplied roots; "
                "no unseen-root generalization, promotion, or exact exploitability "
                "claim"
            ),
            "iterations": completed_total,
            "traversals": expected_traversals,
            "seed": seed,
            "max_infosets": max_infosets,
            "average_strategy_sha256": average_sha256,
            "current_strategy_sha256": current_sha256,
            "checkpoint_supported": True,
            "checkpoint_enabled": checkpoint_contract_required,
            "checkpoint_format": MULTI_ROOT_CHECKPOINT_FORMAT,
            "checkpoint_iteration": completed_total,
            "checkpoint_sha256": final_checkpoint_sha256,
            "checkpoint_atomic_replace": True,
            "checkpoint_canonical_json": True,
            "checkpoint_iteration_boundary_only": True,
            "checkpoint_resume_requires_external_sha256": True,
            "checkpoint_arbitrary_generic_adapter_supported": False,
            "checkpoint_adapter_exact_type_allowlist": [
                "FullCardGenerativeAdapter",
                "CanonicalReducedPublicTreeAdapter",
            ],
            "checkpoint_live_python_runtime_semantics_bound": True,
            "checkpoint_native_or_stdlib_runtime_attested": False,
            "checkpoint_stats_promotion_evidence": False,
            "checkpoint_non_derived_stats_trusted_for_promotion": False,
            "checkpoint_source_binding_sha256": (
                checkpoint_source_binding_sha256
            ),
            "position_contract_version": POSITION_CONTRACT_VERSION,
        }
    )
    return MultiRootExternalSamplingMccfrResult(
        iterations=completed_total,
        traversals=expected_traversals,
        seed=seed,
        root_count=len(entries),
        encountered_infosets=len(stable_keys),
        average_strategy=_freeze_profile(stable_keys, average_rows),
        current_strategy=_freeze_profile(stable_keys, current_rows),
        cumulative_regret_plus=_freeze_profile(stable_keys, regret_rows),
        infoset_root_support_count=support_rows,
        shared_across_roots_infosets=shared_keys,
        average_strategy_json=average_json,
        current_strategy_json=current_json,
        average_strategy_sha256=average_sha256,
        current_strategy_sha256=current_sha256,
        sampling_stats=stats.snapshot(entries),
        metadata=metadata,
    )


def verify_multi_root_checkpoint_against_result(
    roots: Sequence[MultiRootChanceEntry],
    result: MultiRootExternalSamplingMccfrResult,
    *,
    checkpoint_path: str | os.PathLike[str],
    expected_checkpoint_sha256: str,
) -> VerifiedMultiRootCheckpointSnapshot:
    """Restore a persisted checkpoint and prove exact result equivalence.

    ``expected_checkpoint_sha256`` is intentionally external to the checkpoint
    file.  A coordinated edit and self-rehash of the file is therefore not an
    authentication anchor.  The function also reads the file before and after
    restoration and rejects a concurrent replacement.

    This is a structural/content verification API, not solver reexecution and
    not promotion evidence.  Its principal additional output is authoritative
    per-infoset, per-root visit accounting for downstream teacher generation.
    """

    if type(result) is not MultiRootExternalSamplingMccfrResult:
        raise TypeError(
            "result must be an exact MultiRootExternalSamplingMccfrResult"
        )
    if (
        not isinstance(expected_checkpoint_sha256, str)
        or len(expected_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_checkpoint_sha256
        )
    ):
        raise MultiRootMccfrCheckpointError(
            "expected checkpoint SHA256 must be a lowercase digest"
        )

    entries = _validated_entries(roots)
    metadata = result.metadata
    if not isinstance(metadata, Mapping):
        raise MultiRootMccfrCheckpointError("solver result metadata is invalid")
    if metadata.get("checkpoint_enabled") is not True:
        raise MultiRootMccfrCheckpointError(
            "verified checkpoint evidence requires checkpoint_enabled=true"
        )
    if metadata.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise MultiRootMccfrCheckpointError(
            "result checkpoint SHA256 does not match the external digest"
        )
    if result.seed != metadata.get("seed"):
        raise MultiRootMccfrCheckpointError("solver result seed metadata mismatch")
    max_infosets = metadata.get("max_infosets")
    if (
        isinstance(max_infosets, bool)
        or not isinstance(max_infosets, int)
        or max_infosets <= 0
    ):
        raise MultiRootMccfrCheckpointError(
            "solver result max_infosets metadata is invalid"
        )
    linear_averaging = metadata.get("linear_averaging")
    if not isinstance(linear_averaging, bool):
        raise MultiRootMccfrCheckpointError(
            "solver result linear_averaging metadata is invalid"
        )

    path = _checkpoint_path(checkpoint_path, for_read=True)
    try:
        before_bytes = path.read_bytes()
    except OSError as exc:  # pragma: no cover - path was validated above
        raise MultiRootMccfrCheckpointError(
            f"cannot read checkpoint snapshot: {exc}"
        ) from exc
    payload, parsed_checkpoint_sha256 = _read_multi_root_checkpoint(path)
    if parsed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise MultiRootMccfrCheckpointError(
            "checkpoint payload does not match the external digest"
        )

    prior_manifest = _root_prior_manifest(entries)
    adapter_bindings = _checkpoint_adapter_bindings(entries)
    source_binding = _multi_root_source_binding()
    restored = _restore_multi_root_checkpoint(
        path,
        entries=entries,
        seed=result.seed,
        linear_averaging=linear_averaging,
        max_infosets=max_infosets,
        prior_manifest=prior_manifest,
        adapter_bindings=adapter_bindings,
        source_binding=source_binding,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    try:
        after_bytes = path.read_bytes()
    except OSError as exc:  # pragma: no cover - path was validated above
        raise MultiRootMccfrCheckpointError(
            f"cannot reread checkpoint snapshot: {exc}"
        ) from exc
    if before_bytes != after_bytes:
        raise MultiRootMccfrCheckpointError(
            "checkpoint changed during verified snapshot restoration"
        )

    stable_keys = restored.tables.stable_keys
    average_rows = {
        key: restored.tables.average_strategy(key) for key in stable_keys
    }
    current_rows = {
        key: restored.tables.current_strategy(key) for key in stable_keys
    }
    regret_rows = {
        key: dict(restored.tables.regrets[key]) for key in stable_keys
    }
    reconstructed_average = _freeze_profile(stable_keys, average_rows)
    reconstructed_current = _freeze_profile(stable_keys, current_rows)
    reconstructed_regret = _freeze_profile(stable_keys, regret_rows)
    reconstructed_stats = restored.stats.snapshot(entries)
    reconstructed_support_count = MappingProxyType(
        {
            key: len(restored.infoset_root_sources[key])
            for key in stable_keys
        }
    )
    reconstructed_shared = tuple(
        key for key in stable_keys if reconstructed_support_count[key] > 1
    )
    reconstructed_average_json = serialize_strategy_profile(
        reconstructed_average
    )
    reconstructed_current_json = serialize_strategy_profile(
        reconstructed_current
    )

    equality_checks = (
        (result.iterations, restored.completed_iterations, "iterations"),
        (result.traversals, 2 * restored.completed_iterations, "traversals"),
        (result.root_count, len(entries), "root count"),
        (result.encountered_infosets, len(stable_keys), "infoset count"),
        (result.average_strategy, reconstructed_average, "average strategy"),
        (result.current_strategy, reconstructed_current, "current strategy"),
        (
            result.cumulative_regret_plus,
            reconstructed_regret,
            "cumulative regret",
        ),
        (
            result.infoset_root_support_count,
            reconstructed_support_count,
            "root support count",
        ),
        (
            result.shared_across_roots_infosets,
            reconstructed_shared,
            "shared infosets",
        ),
        (result.sampling_stats, reconstructed_stats, "sampling statistics"),
        (
            result.average_strategy_json,
            reconstructed_average_json,
            "average strategy serialization",
        ),
        (
            result.current_strategy_json,
            reconstructed_current_json,
            "current strategy serialization",
        ),
        (
            result.average_strategy_sha256,
            _sha256_text(reconstructed_average_json),
            "average strategy SHA256",
        ),
        (
            result.current_strategy_sha256,
            _sha256_text(reconstructed_current_json),
            "current strategy SHA256",
        ),
    )
    for actual, expected, label in equality_checks:
        if actual != expected:
            raise MultiRootMccfrCheckpointError(
                f"checkpoint/result {label} mismatch"
            )

    metadata_checks = {
        "iterations": restored.completed_iterations,
        "traversals": 2 * restored.completed_iterations,
        "root_count": len(entries),
        "average_strategy_sha256": _sha256_text(
            reconstructed_average_json
        ),
        "current_strategy_sha256": _sha256_text(reconstructed_current_json),
        "checkpoint_iteration": restored.completed_iterations,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_source_binding_sha256": payload["source_binding"][
            "binding_sha256"
        ],
    }
    for field_name, expected in metadata_checks.items():
        if metadata.get(field_name) != expected:
            raise MultiRootMccfrCheckpointError(
                f"checkpoint/result metadata {field_name} mismatch"
            )

    opaque_ids = tuple(entry.root_id_sha256 for entry in entries)
    support_rows: dict[InfoSetKey, tuple[str, ...]] = {}
    visit_rows: dict[InfoSetKey, Mapping[str, int]] = {}
    for key in stable_keys:
        support_indices = restored.infoset_root_sources[key]
        support_rows[key] = tuple(
            sorted(opaque_ids[index] for index in support_indices)
        )
        visit_rows[key] = MappingProxyType(
            {
                opaque_ids[index]: restored.infoset_root_visit_counts[key][
                    index
                ]
                for index in sorted(
                    restored.infoset_root_visit_counts[key],
                    key=lambda value: opaque_ids[value],
                )
            }
        )

    return VerifiedMultiRootCheckpointSnapshot(
        checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_file_sha256=hashlib.sha256(after_bytes).hexdigest(),
        completed_iterations=restored.completed_iterations,
        seed=result.seed,
        solver_config_sha256=_checkpoint_content_sha256(
            payload["solver_config"]
        ),
        source_binding_sha256=payload["source_binding"]["binding_sha256"],
        prior_manifest_sha256=payload["prior_manifest_sha256"],
        average_strategy=reconstructed_average,
        current_strategy=reconstructed_current,
        cumulative_regret_plus=reconstructed_regret,
        sampling_stats=_freeze_checkpoint_snapshot_value(reconstructed_stats),
        infoset_root_support_opaque_ids=MappingProxyType(support_rows),
        infoset_root_visit_counts_by_opaque_id=MappingProxyType(visit_rows),
    )


def _solver_runtime_semantic_binding() -> dict[str, Any]:
    return _runtime_semantic_graph(
        {
            "solve_multi_root_external_sampling_mccfr": (
                solve_multi_root_external_sampling_mccfr
            ),
            "sample_exact_fraction_index": sample_exact_fraction_index,
            "seeded_rng": seeded_rng,
            "sample_cached_opponent_action": _sample_cached_opponent_action,
        },
        data_roots={
            "POSITION_CONTRACT_VERSION": POSITION_CONTRACT_VERSION,
            "SUPER_ROOT_SAMPLING_CONTRACT": SUPER_ROOT_SAMPLING_CONTRACT,
            "POLICY_IDENTITY_CONTRACT": POLICY_IDENTITY_CONTRACT,
        },
    )


def _total_variation(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    if set(first) != set(second):
        raise ValueError("strategy action sets do not match")
    return 0.5 * math.fsum(
        abs(float(first[action_id]) - float(second[action_id]))
        for action_id in first
    )


@dataclass(frozen=True)
class IndependentRootFusionDiagnostic:
    common_infosets: tuple[InfoSetKey, ...]
    pairwise_total_variation: Mapping[InfoSetKey, float]
    prior_mixture_vs_shared_total_variation: Mapping[InfoSetKey, float]
    max_independent_pairwise_total_variation: float
    max_prior_mixture_vs_shared_total_variation: float
    strategy_fusion_error_detected: bool
    metadata: Mapping[str, Any]


def diagnose_independent_root_strategy_fusion(
    independent_profiles: Sequence[Mapping[InfoSetKey, Mapping[str, float]]],
    shared_profile: Mapping[InfoSetKey, Mapping[str, float]],
    *,
    prior_masses: Sequence[Fraction],
    detection_tolerance: float = 1e-6,
) -> IndependentRootFusionDiagnostic:
    """Expose hidden-root-conditioned policy disagreement after separate solves.

    Pairwise disagreement at one common ``InfoSetKey`` proves that the
    independent profiles cannot be selected using legal player information.
    The prior-mixture comparison is a diagnostic only: signalling changes
    posterior reach, so it is explicitly *not* an exploitability calculation.
    """

    profiles = tuple(independent_profiles)
    exact_priors = tuple(prior_masses)
    if len(profiles) < 2:
        raise ValueError("fusion diagnosis requires at least two independent profiles")
    if len(exact_priors) != len(profiles):
        raise ValueError("prior_masses length must match independent profiles")
    if any(not isinstance(prior, Fraction) for prior in exact_priors):
        raise TypeError("diagnostic prior masses must be fractions.Fraction")
    if any(prior <= 0 for prior in exact_priors):
        raise ValueError("diagnostic prior masses must be positive")
    if sum(exact_priors, Fraction(0, 1)) != 1:
        raise ValueError("diagnostic prior masses must sum exactly to one")
    if not isinstance(detection_tolerance, (int, float)) or isinstance(
        detection_tolerance, bool
    ):
        raise TypeError("detection_tolerance must be a finite non-negative number")
    tolerance = float(detection_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("detection_tolerance must be finite and non-negative")

    common = set(shared_profile)
    for profile in profiles:
        common &= set(profile)
    stable_common = tuple(
        sorted(common, key=lambda key: (key.digest(), key.canonical_json()))
    )
    if not stable_common:
        raise ValueError("independent and shared profiles have no common InfoSetKey")

    pairwise_rows: dict[InfoSetKey, float] = {}
    mixture_rows: dict[InfoSetKey, float] = {}
    for key in stable_common:
        if not isinstance(key, InfoSetKey):
            raise TypeError("fusion diagnostic policy keys must be InfoSetKey")
        action_ids = set(shared_profile[key])
        if not action_ids:
            raise ValueError("fusion diagnostic strategies must be non-empty")
        for profile in profiles:
            if set(profile[key]) != action_ids:
                raise ValueError("common InfoSetKey action sets do not match")
        pairwise_rows[key] = max(
            _total_variation(profiles[left][key], profiles[right][key])
            for left in range(len(profiles))
            for right in range(left + 1, len(profiles))
        )
        mixture = {
            action_id: math.fsum(
                float(exact_priors[index]) * float(profiles[index][key][action_id])
                for index in range(len(profiles))
            )
            for action_id in action_ids
        }
        mixture_rows[key] = _total_variation(mixture, shared_profile[key])

    max_pairwise = max(pairwise_rows.values(), default=0.0)
    max_mixture_gap = max(mixture_rows.values(), default=0.0)
    return IndependentRootFusionDiagnostic(
        common_infosets=stable_common,
        pairwise_total_variation=MappingProxyType(pairwise_rows),
        prior_mixture_vs_shared_total_variation=MappingProxyType(mixture_rows),
        max_independent_pairwise_total_variation=max_pairwise,
        max_prior_mixture_vs_shared_total_variation=max_mixture_gap,
        strategy_fusion_error_detected=max_pairwise > tolerance,
        metadata=MappingProxyType(
            {
                "method": "independent_root_strategy_fusion_diagnostic_v1",
                "policy_identity": "InfoSetKey",
                "pairwise_difference_means_hidden_root_conditioning": True,
                "prior_mixture_is_exploitability": False,
                "exact_exploitability_computed": False,
                "promotion_eligible": False,
                "detection_tolerance": tolerance,
            }
        ),
    )
