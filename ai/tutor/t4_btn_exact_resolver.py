"""Exact, information-safe terminal resolver for the T4 BTN decision.

At ``t4_second`` the BB board is already complete and BTN has no future
chance or opponent decision to model.  Every legal placement can therefore be
enumerated and scored directly from BTN's information set.  This module is an
opt-in algorithm component; it does not change serving or promote a policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import ai.engine.action_space as _action_space
import ai.engine.encoding as _encoding
import ai.engine.turn_order as _turn_order
import ai.tutor.exact_late as _exact_late
import ai.tutor.t3_hu_multi_root_mccfr as _multi_root
import ai.tutor.t3_hu_public_cfr as _public_cfr
import ai.tutor.t3_t4_infoset_encoder as _infoset_encoder
from ai.engine.encoding import Board
from ai.tutor.t3_hu_public_cfr import InfoSetKey


T4_BTN_EXACT_RESOLVE_SCHEMA = "ofc_t4_btn_exact_resolve/v1"
T4_BTN_EXACT_METHOD = "t4_btn_exact_terminal_enumeration_v1"
_SOURCE_BINDING_SCHEMA = "ofc_t4_btn_exact_source_binding/v2"
_RUNTIME_SEMANTIC_BINDING_SCHEMA = "python_live_runtime_semantic_graph_v1"


class T4BtnExactResolveError(ValueError):
    """The T4 BTN exact input, result, or runtime binding failed closed."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise T4BtnExactResolveError("value is not canonical JSON data") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError(f"not a regular file: {resolved}")
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise T4BtnExactResolveError(
            f"required source file is unavailable: {path}"
        ) from exc


def _source_paths() -> Mapping[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "t4_btn_exact_resolver": Path(__file__).resolve(),
        "exact_terminal_scoring": root / "ai" / "tutor" / "exact_late.py",
        "action_space": root / "ai" / "engine" / "action_space.py",
        "canonical_encoding": root / "ai" / "engine" / "encoding.py",
        "canonical_turn_order": root / "ai" / "engine" / "turn_order.py",
        "canonical_game_evaluator": root / "ai" / "engine" / "game_engine.py",
        "canonical_joker_scoring": root / "ai" / "engine" / "scoring.py",
        "fantasyland_utility": root / "ai" / "mcts" / "rollout_evaluator.py",
        "fantasyland_utility_config": root / "ai" / "config" / "fl_ev.json",
        "infoset_contract": root / "ai" / "tutor" / "t3_hu_public_cfr.py",
        "infoset_action_contract": root / "ai" / "tutor" / "t3_t4_infoset_encoder.py",
        "runtime_semantic_graph": root / "ai" / "tutor" / "t3_hu_multi_root_mccfr.py",
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise T4BtnExactResolveError(f"{label} must be a lowercase SHA256")
    return value


def _live_source_sha256s() -> dict[str, str]:
    paths = _source_paths()
    expected_names = {
        "t4_btn_exact_resolver",
        "exact_terminal_scoring",
        "action_space",
        "canonical_encoding",
        "canonical_turn_order",
        "canonical_game_evaluator",
        "canonical_joker_scoring",
        "fantasyland_utility",
        "fantasyland_utility_config",
        "infoset_contract",
        "infoset_action_contract",
        "runtime_semantic_graph",
    }
    if not isinstance(paths, Mapping) or set(paths) != expected_names:
        actual_names = set(paths) if isinstance(paths, Mapping) else set()
        raise T4BtnExactResolveError(
            "canonical source path set drifted: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    if any(not isinstance(path, Path) for path in paths.values()):
        raise T4BtnExactResolveError("canonical source paths must be pathlib.Path values")
    return {
        name: _file_sha256(path)
        for name, path in sorted(paths.items())
    }


_CANONICAL_MODULE_ALIASES = (
    ("_action_space", _action_space),
    ("_encoding", _encoding),
    ("_turn_order", _turn_order),
    ("_exact_late", _exact_late),
    ("_multi_root", _multi_root),
    ("_public_cfr", _public_cfr),
    ("_infoset_encoder", _infoset_encoder),
)


_CANONICAL_EXTERNALS = (
    ("exact_late.apply_action", _exact_late, "apply_action", _exact_late.apply_action),
    ("exact_late.action_key", _exact_late, "action_key", _exact_late.action_key),
    (
        "exact_late.terminal_metrics",
        _exact_late,
        "terminal_metrics",
        _exact_late.terminal_metrics,
    ),
    (
        "action_space.get_action_from_semantic_index_if_valid",
        _action_space,
        "get_action_from_semantic_index_if_valid",
        _action_space.get_action_from_semantic_index_if_valid,
    ),
    (
        "infoset_encoder.semantic_action_ids",
        _infoset_encoder,
        "semantic_action_ids",
        _infoset_encoder.semantic_action_ids,
    ),
    (
        "infoset_encoder.legal_action_mask",
        _infoset_encoder,
        "legal_action_mask",
        _infoset_encoder.legal_action_mask,
    ),
    ("encoding.Board", _encoding, "Board", _encoding.Board),
    ("public_cfr.InfoSetKey", _public_cfr, "InfoSetKey", _public_cfr.InfoSetKey),
    (
        "turn_order.POSITION_CONTRACT_VERSION",
        _turn_order,
        "POSITION_CONTRACT_VERSION",
        _turn_order.POSITION_CONTRACT_VERSION,
    ),
    (
        "multi_root._canonical_json",
        _multi_root,
        "_canonical_json",
        _multi_root._canonical_json,
    ),
    (
        "multi_root._checkpoint_content_sha256",
        _multi_root,
        "_checkpoint_content_sha256",
        _multi_root._checkpoint_content_sha256,
    ),
    (
        "multi_root._python_code_constant_binding",
        _multi_root,
        "_python_code_constant_binding",
        _multi_root._python_code_constant_binding,
    ),
    (
        "multi_root._python_code_binding",
        _multi_root,
        "_python_code_binding",
        _multi_root._python_code_binding,
    ),
    (
        "multi_root._python_function_binding",
        _multi_root,
        "_python_function_binding",
        _multi_root._python_function_binding,
    ),
    (
        "multi_root._runtime_data_binding",
        _multi_root,
        "_runtime_data_binding",
        _multi_root._runtime_data_binding,
    ),
    (
        "multi_root._nested_code_global_names",
        _multi_root,
        "_nested_code_global_names",
        _multi_root._nested_code_global_names,
    ),
    (
        "multi_root._runtime_function_key",
        _multi_root,
        "_runtime_function_key",
        _multi_root._runtime_function_key,
    ),
    (
        "multi_root._runtime_semantic_graph",
        _multi_root,
        "_runtime_semantic_graph",
        _multi_root._runtime_semantic_graph,
    ),
)
_CANONICAL_LOCAL_HELPERS: tuple[tuple[str, object], ...] = ()


def _require_runtime_identities(
    *,
    expected_module_aliases: tuple[tuple[str, object], ...],
    expected_externals: tuple[tuple[str, object, str, object], ...],
    expected_local_helpers: tuple[tuple[str, object], ...],
    expected_runtime_binding_sha256: str,
    expected_source_sha256s: Mapping[str, str],
) -> None:
    for name, expected in expected_module_aliases:
        if globals().get(name) is not expected:
            raise T4BtnExactResolveError(f"canonical module alias drifted: {name}")
    for label, module, attribute, expected in expected_externals:
        if getattr(module, attribute) is not expected:
            raise T4BtnExactResolveError(f"canonical runtime callable drifted: {label}")
    for name, expected in expected_local_helpers:
        if globals().get(name) is not expected:
            raise T4BtnExactResolveError(f"resolver runtime helper drifted: {name}")
    live_runtime = _verify_runtime_semantic_binding(
        _live_runtime_semantic_binding()
    )
    if live_runtime["binding_sha256"] != expected_runtime_binding_sha256:
        raise T4BtnExactResolveError("live runtime semantic binding drifted")
    live_sources = _live_source_sha256s()
    if live_sources != dict(expected_source_sha256s):
        changed = sorted(
            name
            for name in set(live_sources) | set(expected_source_sha256s)
            if live_sources.get(name) != expected_source_sha256s.get(name)
        )
        raise T4BtnExactResolveError(
            f"live canonical source files drifted: {changed}"
        )


def _board(rows: tuple[tuple[str, ...], ...]) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _validate_infoset(information: InfoSetKey) -> None:
    if not isinstance(information, InfoSetKey):
        raise TypeError("information must be an InfoSetKey")
    information.canonical_json()
    if (
        information.phase != "t4_second"
        or information.actor != "btn"
        or information.turn != 4
    ):
        raise T4BtnExactResolveError(
            "exact terminal bypass requires BTN t4_second at turn 4"
        )
    if information.fantasy_state is not None:
        raise T4BtnExactResolveError("standard resolver requires fantasy_state=None")


def _metrics_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "score",
        "raw_score",
        "royalty",
        "bust",
        "fl_any",
        "fl_type",
        "fl_card_count",
    }
    if set(metrics) != expected:
        raise T4BtnExactResolveError("canonical terminal metric schema drifted")
    numeric: dict[str, float] = {}
    for name in ("score", "raw_score", "royalty"):
        value = float(metrics[name])
        if not math.isfinite(value):
            raise T4BtnExactResolveError("terminal metric is non-finite")
        numeric[name] = 0.0 if value == 0.0 else value
    fl_card_count = metrics["fl_card_count"]
    if isinstance(fl_card_count, bool) or not isinstance(fl_card_count, int):
        raise T4BtnExactResolveError("fl_card_count must be an integer")
    fl_type = metrics["fl_type"]
    if fl_type is not None and not isinstance(fl_type, str):
        raise T4BtnExactResolveError("fl_type must be a string or null")
    return {
        **numeric,
        "bust": bool(metrics["bust"]),
        "fl_any": bool(metrics["fl_any"]),
        "fl_type": fl_type,
        "fl_card_count": fl_card_count,
    }


def _compute_exact_tables(
    information: InfoSetKey,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, Any]],
    tuple[str, ...],
    str,
    dict[str, float],
]:
    action_ids = _infoset_encoder.semantic_action_ids(information)
    mask = tuple(bool(value) for value in _infoset_encoder.legal_action_mask(information))
    if len(action_ids) != 27 or len(mask) != 27:
        raise T4BtnExactResolveError("T4 action contract must contain 27 semantic slots")
    if tuple(action_id is not None for action_id in action_ids) != mask:
        raise T4BtnExactResolveError("legal action mask and IDs disagree")

    btn_board = _board(information.board_btn)
    bb_board = _board(information.board_bb)
    utility_by_action: dict[str, float] = {}
    metrics_by_action: dict[str, dict[str, Any]] = {}
    for index, action_id in enumerate(action_ids):
        if action_id is None:
            continue
        action = _action_space.get_action_from_semantic_index_if_valid(
            index,
            list(information.current_draw),
            btn_board,
        )
        if action is None:
            raise T4BtnExactResolveError("semantic legal action cannot be reconstructed")
        reconstructed_id = _exact_late.action_key(action)
        if reconstructed_id != action_id:
            raise T4BtnExactResolveError("semantic action identity drifted")
        final_btn = _exact_late.apply_action(btn_board, action)
        metrics = _metrics_payload(_exact_late.terminal_metrics(final_btn, bb_board))
        utility_by_action[action_id] = float(metrics["score"])
        metrics_by_action[action_id] = metrics

    utility_by_action = dict(sorted(utility_by_action.items()))
    metrics_by_action = {
        action_id: metrics_by_action[action_id]
        for action_id in utility_by_action
    }
    if not utility_by_action:
        raise T4BtnExactResolveError("T4 BTN has no legal terminal action")
    best_value = max(utility_by_action.values())
    optimal = tuple(
        action_id
        for action_id, value in utility_by_action.items()
        if value == best_value
    )
    selected = optimal[0]
    policy = {
        action_id: (1.0 if action_id == selected else 0.0)
        for action_id in utility_by_action
    }
    return utility_by_action, metrics_by_action, optimal, selected, policy


def _live_runtime_semantic_binding() -> dict[str, Any]:
    """Bind the executable Python graph and mutable live scoring data."""

    roots = {
        "resolver._board": _board,
        "resolver._validate_infoset": _validate_infoset,
        "resolver._metrics_payload": _metrics_payload,
        "resolver._compute_exact_tables": _compute_exact_tables,
        "exact_late.action_key": _exact_late.action_key,
        "exact_late.apply_action": _exact_late.apply_action,
        "exact_late.terminal_metrics": _exact_late.terminal_metrics,
        "action_space.get_action_from_semantic_index_if_valid": (
            _action_space.get_action_from_semantic_index_if_valid
        ),
        "infoset_encoder.semantic_action_ids": _infoset_encoder.semantic_action_ids,
        "infoset_encoder.legal_action_mask": _infoset_encoder.legal_action_mask,
    }
    data_roots = {
        "exact_late.FL_TYPE_BY_CARD_COUNT": _exact_late.FL_TYPE_BY_CARD_COUNT,
        "exact_late.RolloutEvaluator.FL_EV": _exact_late.RolloutEvaluator.FL_EV,
        "infoset_encoder.ACTION_SEMANTICS_SHA256": (
            _infoset_encoder.ACTION_SEMANTICS_SHA256
        ),
        "infoset_encoder.INFOSET_ENCODER_MANIFEST_SHA256": (
            _infoset_encoder.INFOSET_ENCODER_MANIFEST_SHA256
        ),
    }
    try:
        return _multi_root._runtime_semantic_graph(
            roots,
            data_roots=data_roots,
        )
    except _multi_root.MultiRootMccfrCheckpointError as exc:
        raise T4BtnExactResolveError(
            "live runtime semantic graph could not be bound"
        ) from exc


def _verify_runtime_semantic_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T4BtnExactResolveError("runtime semantic binding must be a mapping")
    snapshot = json.loads(_canonical_json(dict(value)))
    expected_fields = {
        "schema",
        "roots",
        "functions",
        "classes",
        "data",
        "binding_sha256",
    }
    if set(snapshot) != expected_fields:
        raise T4BtnExactResolveError("runtime semantic binding fields drifted")
    if snapshot["schema"] != _RUNTIME_SEMANTIC_BINDING_SCHEMA:
        raise T4BtnExactResolveError("runtime semantic binding schema drifted")
    for field in ("roots", "functions", "classes", "data"):
        expected_type = dict if field == "roots" else list
        if not isinstance(snapshot[field], expected_type):
            raise T4BtnExactResolveError(
                f"runtime semantic binding {field} has invalid type"
            )
    declared = _require_sha256(
        snapshot["binding_sha256"],
        label="runtime semantic binding SHA256",
    )
    unsigned = dict(snapshot)
    del unsigned["binding_sha256"]
    if _canonical_sha256(unsigned) != declared:
        raise T4BtnExactResolveError("runtime semantic binding SHA256 mismatch")
    return snapshot


def _verify_source_binding(value: Any) -> dict[str, Any]:
    """Independently validate exact source fields against live files/runtime."""

    if not isinstance(value, Mapping):
        raise T4BtnExactResolveError("source binding must be a mapping")
    snapshot = json.loads(_canonical_json(dict(value)))
    expected_fields = {
        "schema",
        "files",
        "infoset_encoder_manifest_sha256",
        "action_semantics_sha256",
        "runtime_semantic_binding_schema",
        "runtime_semantic_binding_sha256",
        "binding_sha256",
    }
    if set(snapshot) != expected_fields:
        raise T4BtnExactResolveError("source binding fields drifted")
    if snapshot["schema"] != _SOURCE_BINDING_SCHEMA:
        raise T4BtnExactResolveError("source binding schema drifted")
    files = snapshot["files"]
    if not isinstance(files, dict):
        raise T4BtnExactResolveError("source binding files must be a mapping")
    live_files = _live_source_sha256s()
    if set(files) != set(live_files):
        raise T4BtnExactResolveError("source binding file set drifted")
    for name in sorted(live_files):
        declared = _require_sha256(
            files[name],
            label=f"source binding files[{name!r}]",
        )
        if declared != live_files[name]:
            raise T4BtnExactResolveError(
                f"source binding file hash drifted: {name}"
            )
    if snapshot["infoset_encoder_manifest_sha256"] != (
        _infoset_encoder.INFOSET_ENCODER_MANIFEST_SHA256
    ):
        raise T4BtnExactResolveError("infoset encoder binding drifted")
    _require_sha256(
        snapshot["infoset_encoder_manifest_sha256"],
        label="infoset encoder manifest SHA256",
    )
    if snapshot["action_semantics_sha256"] != (
        _infoset_encoder.ACTION_SEMANTICS_SHA256
    ):
        raise T4BtnExactResolveError("action semantics binding drifted")
    _require_sha256(
        snapshot["action_semantics_sha256"],
        label="action semantics SHA256",
    )
    live_runtime = _verify_runtime_semantic_binding(
        _live_runtime_semantic_binding()
    )
    if snapshot["runtime_semantic_binding_schema"] != live_runtime["schema"]:
        raise T4BtnExactResolveError("source runtime binding schema drifted")
    if snapshot["runtime_semantic_binding_sha256"] != live_runtime["binding_sha256"]:
        raise T4BtnExactResolveError("source runtime semantic binding drifted")
    _require_sha256(
        snapshot["runtime_semantic_binding_sha256"],
        label="source runtime semantic binding SHA256",
    )
    declared_binding = _require_sha256(
        snapshot["binding_sha256"],
        label="source binding SHA256",
    )
    unsigned = dict(snapshot)
    del unsigned["binding_sha256"]
    if _canonical_sha256(unsigned) != declared_binding:
        raise T4BtnExactResolveError("source binding SHA256 mismatch")
    return snapshot


def _live_source_binding() -> dict[str, Any]:
    runtime_binding = _verify_runtime_semantic_binding(
        _live_runtime_semantic_binding()
    )
    payload = {
        "schema": _SOURCE_BINDING_SCHEMA,
        "files": _live_source_sha256s(),
        "infoset_encoder_manifest_sha256": (
            _infoset_encoder.INFOSET_ENCODER_MANIFEST_SHA256
        ),
        "action_semantics_sha256": _infoset_encoder.ACTION_SEMANTICS_SHA256,
        "runtime_semantic_binding_schema": runtime_binding["schema"],
        "runtime_semantic_binding_sha256": runtime_binding["binding_sha256"],
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return _verify_source_binding(payload)


def _table_binding(
    utility_by_action: Mapping[str, float],
    metrics_by_action: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for action_id in sorted(utility_by_action):
        metrics = metrics_by_action[action_id]
        rows.append(
            {
                "action_id": action_id,
                "utility_hex": float(utility_by_action[action_id]).hex(),
                "metrics": {
                    "score_hex": float(metrics["score"]).hex(),
                    "raw_score_hex": float(metrics["raw_score"]).hex(),
                    "royalty_hex": float(metrics["royalty"]).hex(),
                    "bust": metrics["bust"],
                    "fl_any": metrics["fl_any"],
                    "fl_type": metrics["fl_type"],
                    "fl_card_count": metrics["fl_card_count"],
                },
            }
        )
    return {
        "legal_action_count": len(rows),
        "ordered_action_rows": rows,
        "table_sha256": _canonical_sha256(rows),
    }


def _policy_binding(policy: Mapping[str, float]) -> dict[str, Any]:
    rows = [
        {"action_id": action_id, "probability_hex": float(policy[action_id]).hex()}
        for action_id in sorted(policy)
    ]
    return {
        "ordered_action_probabilities": rows,
        "policy_sha256": _canonical_sha256(rows),
    }


def _build_manifest(
    information: InfoSetKey,
    utility_by_action: Mapping[str, float],
    metrics_by_action: Mapping[str, Mapping[str, Any]],
    optimal_action_ids: tuple[str, ...],
    selected_action_id: str,
    policy: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema": T4_BTN_EXACT_RESOLVE_SCHEMA,
        "method": T4_BTN_EXACT_METHOD,
        "infoset_sha256": information.digest(),
        "phase": "t4_second",
        "actor": "btn",
        "turn": 4,
        "utility_perspective": "btn",
        "terminal_opponent_board_complete": True,
        "all_legal_actions_enumerated": True,
        "chance_sampling_used": False,
        "policy_sampling_used": False,
        "opponent_hidden_cards_used": False,
        "exact_terminal_utility": True,
        "tie_break": "canonical_action_id_ascending",
        "optimal_action_ids": list(optimal_action_ids),
        "selected_action_id": selected_action_id,
        "action_value_binding": _table_binding(
            utility_by_action,
            metrics_by_action,
        ),
        "policy_binding": _policy_binding(policy),
        "source_binding": _live_source_binding(),
        "algorithm_component_ready": True,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "global_unseen_state_policy_claim": False,
        "serving_changed": False,
    }


@dataclass(frozen=True, slots=True)
class T4BtnExactResolution:
    """One content-bound exact terminal policy for an actual BTN InfoSetKey."""

    information: InfoSetKey
    utility_by_action_id: Mapping[str, float] = field(repr=False)
    terminal_metrics_by_action_id: Mapping[str, Mapping[str, Any]] = field(repr=False)
    optimal_action_ids: tuple[str, ...]
    selected_action_id: str
    action_probabilities: Mapping[str, float]
    manifest_json: str = field(repr=False)
    manifest_sha256: str


def _strict_float_table(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise T4BtnExactResolveError(f"{label} must be a mapping")
    snapshot: dict[str, float] = {}
    for action_id, number in value.items():
        if type(action_id) is not str:
            raise T4BtnExactResolveError(f"{label} action IDs must be strings")
        if type(number) is not float or not math.isfinite(number):
            raise T4BtnExactResolveError(
                f"{label}[{action_id!r}] must be a finite strict float"
            )
        snapshot[action_id] = number
    return snapshot


def _strict_terminal_metrics_table(
    value: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise T4BtnExactResolveError("terminal metrics table must be a mapping")
    expected_fields = {
        "score",
        "raw_score",
        "royalty",
        "bust",
        "fl_any",
        "fl_type",
        "fl_card_count",
    }
    snapshot: dict[str, dict[str, Any]] = {}
    for action_id, raw_row in value.items():
        if type(action_id) is not str:
            raise T4BtnExactResolveError(
                "terminal metrics table action IDs must be strings"
            )
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_fields:
            raise T4BtnExactResolveError(
                f"terminal metrics row fields drifted: {action_id}"
            )
        row = dict(raw_row)
        for name in ("score", "raw_score", "royalty"):
            number = row[name]
            if type(number) is not float or not math.isfinite(number):
                raise T4BtnExactResolveError(
                    f"terminal metrics {action_id}.{name} must be a finite strict float"
                )
        for name in ("bust", "fl_any"):
            if type(row[name]) is not bool:
                raise T4BtnExactResolveError(
                    f"terminal metrics {action_id}.{name} must be a strict bool"
                )
        if row["fl_type"] is not None and type(row["fl_type"]) is not str:
            raise T4BtnExactResolveError(
                f"terminal metrics {action_id}.fl_type has invalid type"
            )
        if type(row["fl_card_count"]) is not int:
            raise T4BtnExactResolveError(
                f"terminal metrics {action_id}.fl_card_count must be a strict integer"
            )
        snapshot[action_id] = row
    return snapshot


def resolve_t4_second_btn_exact(information: InfoSetKey) -> T4BtnExactResolution:
    """Enumerate and score every legal BTN terminal placement exactly."""

    _validate_infoset(information)
    utility, metrics, optimal, selected, policy = _compute_exact_tables(information)
    manifest = _build_manifest(
        information,
        utility,
        metrics,
        optimal,
        selected,
        policy,
    )
    manifest_json = _canonical_json(manifest)
    result = T4BtnExactResolution(
        information=information,
        utility_by_action_id=MappingProxyType(dict(utility)),
        terminal_metrics_by_action_id=MappingProxyType(
            {
                action_id: MappingProxyType(dict(row))
                for action_id, row in metrics.items()
            }
        ),
        optimal_action_ids=optimal,
        selected_action_id=selected,
        action_probabilities=MappingProxyType(dict(policy)),
        manifest_json=manifest_json,
        manifest_sha256=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )
    _VERIFY_IMPL(information, result)
    return result


def verify_t4_second_btn_exact(
    information: InfoSetKey,
    result: T4BtnExactResolution,
    *,
    expected_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Freshly recompute the terminal solve and verify every result binding."""

    _validate_infoset(information)
    if not isinstance(result, T4BtnExactResolution):
        raise TypeError("result must be T4BtnExactResolution")
    if result.information != information:
        raise T4BtnExactResolveError("result infoset does not match verification input")
    if type(result.manifest_json) is not str:
        raise T4BtnExactResolveError("result manifest must be a strict string")
    _require_sha256(result.manifest_sha256, label="result manifest SHA256")
    try:
        manifest = json.loads(result.manifest_json)
    except json.JSONDecodeError as exc:
        raise T4BtnExactResolveError("result manifest is not JSON") from exc
    if _canonical_json(manifest) != result.manifest_json:
        raise T4BtnExactResolveError("result manifest is not canonical JSON")
    computed_manifest_sha256 = hashlib.sha256(
        result.manifest_json.encode("utf-8")
    ).hexdigest()
    if computed_manifest_sha256 != result.manifest_sha256:
        raise T4BtnExactResolveError("result manifest SHA256 mismatch")
    if expected_manifest_sha256 is not None:
        if (
            not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_manifest_sha256)
        ):
            raise T4BtnExactResolveError("expected manifest SHA256 is invalid")
        if result.manifest_sha256 != expected_manifest_sha256:
            raise T4BtnExactResolveError(
                "result does not match externally trusted manifest SHA256"
            )
    _verify_source_binding(manifest.get("source_binding"))

    utility, metrics, optimal, selected, policy = _compute_exact_tables(information)
    actual_utility = _strict_float_table(
        result.utility_by_action_id,
        label="terminal utility table",
    )
    if actual_utility != utility:
        raise T4BtnExactResolveError("terminal utility table mismatch")
    actual_metrics = _strict_terminal_metrics_table(
        result.terminal_metrics_by_action_id
    )
    if actual_metrics != metrics:
        raise T4BtnExactResolveError("terminal metrics table mismatch")
    if (
        type(result.optimal_action_ids) is not tuple
        or any(type(action_id) is not str for action_id in result.optimal_action_ids)
    ):
        raise T4BtnExactResolveError("optimal action IDs must be a tuple of strings")
    if result.optimal_action_ids != optimal:
        raise T4BtnExactResolveError("optimal action set mismatch")
    if type(result.selected_action_id) is not str:
        raise T4BtnExactResolveError("selected action ID must be a strict string")
    if result.selected_action_id != selected:
        raise T4BtnExactResolveError("selected action mismatch")
    actual_policy = _strict_float_table(
        result.action_probabilities,
        label="exact terminal policy",
    )
    if actual_policy != policy:
        raise T4BtnExactResolveError("exact terminal policy mismatch")
    expected_manifest = _build_manifest(
        information,
        utility,
        metrics,
        optimal,
        selected,
        policy,
    )
    if manifest != expected_manifest:
        raise T4BtnExactResolveError("result manifest content mismatch")

    # Independent truth-boundary assertions do not rely on the manifest builder.
    fixed_truth = {
        "schema": T4_BTN_EXACT_RESOLVE_SCHEMA,
        "method": T4_BTN_EXACT_METHOD,
        "phase": "t4_second",
        "actor": "btn",
        "turn": 4,
        "utility_perspective": "btn",
        "terminal_opponent_board_complete": True,
        "all_legal_actions_enumerated": True,
        "chance_sampling_used": False,
        "policy_sampling_used": False,
        "opponent_hidden_cards_used": False,
        "exact_terminal_utility": True,
        "algorithm_component_ready": True,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "global_unseen_state_policy_claim": False,
        "serving_changed": False,
    }
    for name, expected in fixed_truth.items():
        if manifest.get(name) != expected:
            raise T4BtnExactResolveError(f"fixed manifest truth drifted: {name}")
    if manifest.get("infoset_sha256") != information.digest():
        raise T4BtnExactResolveError("manifest infoset binding mismatch")
    return MappingProxyType(
        {
            "verified": True,
            "manifest_sha256": result.manifest_sha256,
            "infoset_sha256": information.digest(),
            "legal_action_count": len(utility),
            "optimal_action_count": len(optimal),
            "selected_action_id": selected,
            "exact_terminal_utility": True,
            "promotion_eligible": False,
            "runtime_integrated": False,
        }
    )


_RESOLVE_IMPL = resolve_t4_second_btn_exact
_VERIFY_IMPL = verify_t4_second_btn_exact
_CANONICAL_LOCAL_HELPERS = (
    ("_canonical_json", _canonical_json),
    ("_canonical_sha256", _canonical_sha256),
    ("_file_sha256", _file_sha256),
    ("_source_paths", _source_paths),
    ("_require_sha256", _require_sha256),
    ("_live_source_sha256s", _live_source_sha256s),
    ("_board", _board),
    ("_validate_infoset", _validate_infoset),
    ("_metrics_payload", _metrics_payload),
    ("_compute_exact_tables", _compute_exact_tables),
    ("_live_runtime_semantic_binding", _live_runtime_semantic_binding),
    ("_verify_runtime_semantic_binding", _verify_runtime_semantic_binding),
    ("_verify_source_binding", _verify_source_binding),
    ("_live_source_binding", _live_source_binding),
    ("_table_binding", _table_binding),
    ("_policy_binding", _policy_binding),
    ("_build_manifest", _build_manifest),
    ("_strict_float_table", _strict_float_table),
    ("_strict_terminal_metrics_table", _strict_terminal_metrics_table),
    ("_RESOLVE_IMPL", _RESOLVE_IMPL),
    ("_VERIFY_IMPL", _VERIFY_IMPL),
)
_CANONICAL_SOURCE_SHA256S = MappingProxyType(_live_source_sha256s())
_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256 = _verify_runtime_semantic_binding(
    _live_runtime_semantic_binding()
)["binding_sha256"]
_CANONICAL_RUNTIME_GUARD = _require_runtime_identities


def _guard_entrypoint(function: Callable[..., Any]) -> Callable[..., Any]:
    """Close over immutable import-time attestations so guard aliases cannot bypass."""

    canonical_guard = _require_runtime_identities
    canonical_module_aliases = _CANONICAL_MODULE_ALIASES
    canonical_externals = _CANONICAL_EXTERNALS
    canonical_local_helpers = _CANONICAL_LOCAL_HELPERS
    canonical_runtime_sha256 = _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256
    canonical_source_sha256s = _CANONICAL_SOURCE_SHA256S

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if (
            globals().get("_CANONICAL_RUNTIME_GUARD") is not canonical_guard
            or globals().get("_require_runtime_identities") is not canonical_guard
        ):
            raise T4BtnExactResolveError("canonical runtime guard alias drifted")
        if globals().get("_CANONICAL_MODULE_ALIASES") is not canonical_module_aliases:
            raise T4BtnExactResolveError("canonical module alias set drifted")
        for name, expected in canonical_module_aliases:
            if globals().get(name) is not expected:
                raise T4BtnExactResolveError(
                    f"canonical module alias drifted: {name}"
                )
        if globals().get("_CANONICAL_EXTERNALS") is not canonical_externals:
            raise T4BtnExactResolveError("canonical external identity set drifted")
        if globals().get("_CANONICAL_LOCAL_HELPERS") is not canonical_local_helpers:
            raise T4BtnExactResolveError("canonical helper identity set drifted")
        if (
            globals().get("_CANONICAL_SOURCE_SHA256S")
            is not canonical_source_sha256s
        ):
            raise T4BtnExactResolveError("canonical source snapshot alias drifted")
        if globals().get(
            "_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256"
        ) != canonical_runtime_sha256:
            raise T4BtnExactResolveError("canonical runtime snapshot alias drifted")
        canonical_guard(
            expected_module_aliases=canonical_module_aliases,
            expected_externals=canonical_externals,
            expected_local_helpers=canonical_local_helpers,
            expected_runtime_binding_sha256=canonical_runtime_sha256,
            expected_source_sha256s=canonical_source_sha256s,
        )
        return function(*args, **kwargs)

    return guarded


resolve_t4_second_btn_exact = _guard_entrypoint(_RESOLVE_IMPL)
verify_t4_second_btn_exact = _guard_entrypoint(_VERIFY_IMPL)


__all__ = [
    "T4_BTN_EXACT_METHOD",
    "T4_BTN_EXACT_RESOLVE_SCHEMA",
    "T4BtnExactResolution",
    "T4BtnExactResolveError",
    "resolve_t4_second_btn_exact",
    "verify_t4_second_btn_exact",
]
