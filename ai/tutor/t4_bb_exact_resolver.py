"""Exact uniform-deal resolver for the T4 BB decision.

At ``t4_first`` BB places the final two cards while BTN still has one hidden
3-card draw.  Under the declared uniform exchangeable restart belief (BTN's
final draw is uniform over the 26 cards unseen from BB's information set),
every legal BB placement can be scored exactly: each candidate final board is
evaluated against every possible BTN draw with BTN playing its exact terminal
best response.  This mirrors ``t4_btn_exact_resolver`` and the regular-track
M3.0 first-seat component.

This is exact under the declared belief only.  It is not a Bayes posterior
conditioned on BTN behavior, not a Nash proof, and not a policy for unseen
states.  It is an opt-in algorithm component; it does not change serving or
promote a policy.
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


T4_BB_EXACT_RESOLVE_SCHEMA = "ofc_t4_bb_exact_resolve/v1"
T4_BB_EXACT_METHOD = "t4_bb_exact_uniform_deal_response_v1"
T4_BB_BELIEF_MODEL = "uniform_exchangeable_restart_v1"
T4_BB_OPPONENT_RESPONSE_MODEL = "exact_t4_best_response_v1"
_SOURCE_BINDING_SCHEMA = "ofc_t4_bb_exact_source_binding/v1"
_RUNTIME_SEMANTIC_BINDING_SCHEMA = "python_live_runtime_semantic_graph_v1"


class T4BbExactResolveError(ValueError):
    """The T4 BB exact input, result, or runtime binding failed closed."""


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
        raise T4BbExactResolveError("value is not canonical JSON data") from exc


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
        raise T4BbExactResolveError(
            f"required source file is unavailable: {path}"
        ) from exc


def _source_paths() -> Mapping[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "t4_bb_exact_resolver": Path(__file__).resolve(),
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
        raise T4BbExactResolveError(f"{label} must be a lowercase SHA256")
    return value


def _live_source_sha256s() -> dict[str, str]:
    paths = _source_paths()
    expected_names = {
        "t4_bb_exact_resolver",
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
        raise T4BbExactResolveError(
            "canonical source path set drifted: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    if any(not isinstance(path, Path) for path in paths.values()):
        raise T4BbExactResolveError("canonical source paths must be pathlib.Path values")
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
        "exact_late.best_t4_completion",
        _exact_late,
        "best_t4_completion",
        _exact_late.best_t4_completion,
    ),
    (
        "exact_late.exact_t4_opponent_response_distribution",
        _exact_late,
        "exact_t4_opponent_response_distribution",
        _exact_late.exact_t4_opponent_response_distribution,
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
            raise T4BbExactResolveError(f"canonical module alias drifted: {name}")
    for label, module, attribute, expected in expected_externals:
        if getattr(module, attribute) is not expected:
            raise T4BbExactResolveError(f"canonical runtime callable drifted: {label}")
    for name, expected in expected_local_helpers:
        if globals().get(name) is not expected:
            raise T4BbExactResolveError(f"resolver runtime helper drifted: {name}")
    live_runtime = _verify_runtime_semantic_binding(
        _live_runtime_semantic_binding()
    )
    if live_runtime["binding_sha256"] != expected_runtime_binding_sha256:
        raise T4BbExactResolveError("live runtime semantic binding drifted")
    live_sources = _live_source_sha256s()
    if live_sources != dict(expected_source_sha256s):
        changed = sorted(
            name
            for name in set(live_sources) | set(expected_source_sha256s)
            if live_sources.get(name) != expected_source_sha256s.get(name)
        )
        raise T4BbExactResolveError(
            f"live canonical source files drifted: {changed}"
        )


def _board(rows: tuple[tuple[str, ...], ...]) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _validate_infoset(information: InfoSetKey) -> None:
    if not isinstance(information, InfoSetKey):
        raise TypeError("information must be an InfoSetKey")
    information.canonical_json()
    if (
        information.phase != "t4_first"
        or information.actor != "bb"
        or information.turn != 4
    ):
        raise T4BbExactResolveError(
            "exact uniform-deal bypass requires BB t4_first at turn 4"
        )
    if information.fantasy_state is not None:
        raise T4BbExactResolveError("standard resolver requires fantasy_state=None")


_EXPECTED_DISTRIBUTION_FIELDS = frozenset(
    {
        "score",
        "ev",
        "raw_score",
        "royalty",
        "bust_rate",
        "fl_rate",
        "fl_type_rates",
        "samples",
        "source",
        "enumerated_draws",
        "remaining_deck_size",
        "start_turn",
        "opponent_response",
    }
)
_FL_TYPE_RATE_FIELDS = ("qq", "kk", "aa", "trips")


def _distribution_payload(
    distribution: Mapping[str, Any],
    *,
    expected_remaining_deck_size: int,
) -> dict[str, Any]:
    if not isinstance(distribution, Mapping):
        raise T4BbExactResolveError("opponent-response distribution must be a mapping")
    if set(distribution) != _EXPECTED_DISTRIBUTION_FIELDS:
        raise T4BbExactResolveError(
            "canonical opponent-response distribution schema drifted"
        )
    if distribution["source"] != "exact_hu_response":
        raise T4BbExactResolveError("distribution source must be exact_hu_response")
    if distribution["start_turn"] != 4 or distribution["opponent_response"] is not True:
        raise T4BbExactResolveError("distribution is not a T4 opponent-response result")
    remaining = distribution["remaining_deck_size"]
    if (
        isinstance(remaining, bool)
        or not isinstance(remaining, int)
        or remaining != expected_remaining_deck_size
    ):
        raise T4BbExactResolveError(
            "remaining deck size drifted from the BB information set: "
            f"got {remaining!r}, expected {expected_remaining_deck_size}"
        )
    draws = distribution["enumerated_draws"]
    expected_draws = math.comb(expected_remaining_deck_size, 3)
    if (
        isinstance(draws, bool)
        or not isinstance(draws, int)
        or draws != expected_draws
        or distribution["samples"] != draws
    ):
        raise T4BbExactResolveError(
            "opponent draw enumeration is not exhaustive: "
            f"got {draws!r}, expected {expected_draws}"
        )
    numeric: dict[str, float] = {}
    for name in ("score", "raw_score", "royalty", "bust_rate", "fl_rate"):
        value = float(distribution[name])
        if not math.isfinite(value):
            raise T4BbExactResolveError(f"distribution metric {name} is non-finite")
        numeric[name] = 0.0 if value == 0.0 else value
    if float(distribution["ev"]) != float(distribution["score"]):
        raise T4BbExactResolveError("distribution ev/score aliases disagree")
    raw_rates = distribution["fl_type_rates"]
    if not isinstance(raw_rates, Mapping) or set(raw_rates) != set(
        _FL_TYPE_RATE_FIELDS
    ):
        raise T4BbExactResolveError("fl_type_rates schema drifted")
    fl_type_rates: dict[str, float] = {}
    for name in _FL_TYPE_RATE_FIELDS:
        value = float(raw_rates[name])
        if not math.isfinite(value):
            raise T4BbExactResolveError(f"fl_type_rates[{name!r}] is non-finite")
        fl_type_rates[name] = 0.0 if value == 0.0 else value
    return {
        **numeric,
        "fl_type_rates": fl_type_rates,
        "enumerated_draws": draws,
        "remaining_deck_size": remaining,
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
        raise T4BbExactResolveError("T4 action contract must contain 27 semantic slots")
    if tuple(action_id is not None for action_id in action_ids) != mask:
        raise T4BbExactResolveError("legal action mask and IDs disagree")

    bb_board = _board(information.board_bb)
    btn_board = _board(information.board_btn)
    own_discards = tuple(
        card for _turn, card in information.own_recall.discards_by_turn
    )
    if len(own_discards) != 3 or len(set(own_discards)) != 3:
        raise T4BbExactResolveError(
            "BB t4_first recall must contain exactly the three prior hidden discards"
        )
    draw_cards = tuple(information.current_draw)
    expected_remaining = (
        len(_encoding.ALL_CARDS) - 13 - 11 - len(own_discards) - 1
    )

    utility_by_action: dict[str, float] = {}
    metrics_by_action: dict[str, dict[str, Any]] = {}
    for index, action_id in enumerate(action_ids):
        if action_id is None:
            continue
        action = _action_space.get_action_from_semantic_index_if_valid(
            index,
            list(draw_cards),
            bb_board,
        )
        if action is None:
            raise T4BbExactResolveError("semantic legal action cannot be reconstructed")
        reconstructed_id = _exact_late.action_key(action)
        if reconstructed_id != action_id:
            raise T4BbExactResolveError("semantic action identity drifted")
        discard = action.discard
        placed = tuple(card for card, _row in action.placements)
        if (
            type(discard) is not str
            or len(placed) != 2
            or set(placed) | {discard} != set(draw_cards)
        ):
            raise T4BbExactResolveError(
                "T4 BB action must place two draw cards and discard the third"
            )
        final_bb = _exact_late.apply_action(bb_board, action)
        distribution = _exact_late.exact_t4_opponent_response_distribution(
            final_bb,
            btn_board,
            exclude=own_discards + (discard,),
        )
        payload = _distribution_payload(
            distribution,
            expected_remaining_deck_size=expected_remaining,
        )
        utility_by_action[action_id] = float(payload["score"])
        metrics_by_action[action_id] = payload

    utility_by_action = dict(sorted(utility_by_action.items()))
    metrics_by_action = {
        action_id: metrics_by_action[action_id]
        for action_id in utility_by_action
    }
    if not utility_by_action:
        raise T4BbExactResolveError("T4 BB has no legal action")
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
        "resolver._distribution_payload": _distribution_payload,
        "resolver._compute_exact_tables": _compute_exact_tables,
        "exact_late.action_key": _exact_late.action_key,
        "exact_late.apply_action": _exact_late.apply_action,
        "exact_late.terminal_metrics": _exact_late.terminal_metrics,
        "exact_late.best_t4_completion": _exact_late.best_t4_completion,
        "exact_late.exact_t4_opponent_response_distribution": (
            _exact_late.exact_t4_opponent_response_distribution
        ),
        "action_space.get_action_from_semantic_index_if_valid": (
            _action_space.get_action_from_semantic_index_if_valid
        ),
        "infoset_encoder.semantic_action_ids": _infoset_encoder.semantic_action_ids,
        "infoset_encoder.legal_action_mask": _infoset_encoder.legal_action_mask,
    }
    data_roots = {
        "encoding.ALL_CARDS": list(_encoding.ALL_CARDS),
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
        raise T4BbExactResolveError(
            "live runtime semantic graph could not be bound"
        ) from exc


def _verify_runtime_semantic_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T4BbExactResolveError("runtime semantic binding must be a mapping")
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
        raise T4BbExactResolveError("runtime semantic binding fields drifted")
    if snapshot["schema"] != _RUNTIME_SEMANTIC_BINDING_SCHEMA:
        raise T4BbExactResolveError("runtime semantic binding schema drifted")
    for field_name in ("roots", "functions", "classes", "data"):
        expected_type = dict if field_name == "roots" else list
        if not isinstance(snapshot[field_name], expected_type):
            raise T4BbExactResolveError(
                f"runtime semantic binding {field_name} has invalid type"
            )
    declared = _require_sha256(
        snapshot["binding_sha256"],
        label="runtime semantic binding SHA256",
    )
    unsigned = dict(snapshot)
    del unsigned["binding_sha256"]
    if _canonical_sha256(unsigned) != declared:
        raise T4BbExactResolveError("runtime semantic binding SHA256 mismatch")
    return snapshot


def _verify_source_binding(value: Any) -> dict[str, Any]:
    """Independently validate exact source fields against live files/runtime."""

    if not isinstance(value, Mapping):
        raise T4BbExactResolveError("source binding must be a mapping")
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
        raise T4BbExactResolveError("source binding fields drifted")
    if snapshot["schema"] != _SOURCE_BINDING_SCHEMA:
        raise T4BbExactResolveError("source binding schema drifted")
    files = snapshot["files"]
    if not isinstance(files, dict):
        raise T4BbExactResolveError("source binding files must be a mapping")
    live_files = _live_source_sha256s()
    if set(files) != set(live_files):
        raise T4BbExactResolveError("source binding file set drifted")
    for name in sorted(live_files):
        declared = _require_sha256(
            files[name],
            label=f"source binding files[{name!r}]",
        )
        if declared != live_files[name]:
            raise T4BbExactResolveError(
                f"source binding file hash drifted: {name}"
            )
    if snapshot["infoset_encoder_manifest_sha256"] != (
        _infoset_encoder.INFOSET_ENCODER_MANIFEST_SHA256
    ):
        raise T4BbExactResolveError("infoset encoder binding drifted")
    _require_sha256(
        snapshot["infoset_encoder_manifest_sha256"],
        label="infoset encoder manifest SHA256",
    )
    if snapshot["action_semantics_sha256"] != (
        _infoset_encoder.ACTION_SEMANTICS_SHA256
    ):
        raise T4BbExactResolveError("action semantics binding drifted")
    _require_sha256(
        snapshot["action_semantics_sha256"],
        label="action semantics SHA256",
    )
    live_runtime = _verify_runtime_semantic_binding(
        _live_runtime_semantic_binding()
    )
    if snapshot["runtime_semantic_binding_schema"] != live_runtime["schema"]:
        raise T4BbExactResolveError("source runtime binding schema drifted")
    if snapshot["runtime_semantic_binding_sha256"] != live_runtime["binding_sha256"]:
        raise T4BbExactResolveError("source runtime semantic binding drifted")
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
        raise T4BbExactResolveError("source binding SHA256 mismatch")
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
                    "bust_rate_hex": float(metrics["bust_rate"]).hex(),
                    "fl_rate_hex": float(metrics["fl_rate"]).hex(),
                    "fl_type_rates_hex": {
                        name: float(metrics["fl_type_rates"][name]).hex()
                        for name in _FL_TYPE_RATE_FIELDS
                    },
                    "enumerated_draws": metrics["enumerated_draws"],
                    "remaining_deck_size": metrics["remaining_deck_size"],
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
        "schema": T4_BB_EXACT_RESOLVE_SCHEMA,
        "method": T4_BB_EXACT_METHOD,
        "infoset_sha256": information.digest(),
        "phase": "t4_first",
        "actor": "bb",
        "turn": 4,
        "utility_perspective": "bb",
        "belief_model": T4_BB_BELIEF_MODEL,
        "opponent_response_model": T4_BB_OPPONENT_RESPONSE_MODEL,
        "all_legal_actions_enumerated": True,
        "all_opponent_draws_enumerated": True,
        "chance_sampling_used": False,
        "policy_sampling_used": False,
        "opponent_hidden_cards_used": False,
        "exact_under_declared_belief": True,
        "bayes_posterior_used": False,
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
class T4BbExactResolution:
    """One content-bound exact uniform-deal policy for an actual BB InfoSetKey."""

    information: InfoSetKey
    utility_by_action_id: Mapping[str, float] = field(repr=False)
    distribution_by_action_id: Mapping[str, Mapping[str, Any]] = field(repr=False)
    optimal_action_ids: tuple[str, ...]
    selected_action_id: str
    action_probabilities: Mapping[str, float]
    manifest_json: str = field(repr=False)
    manifest_sha256: str


def _strict_float_table(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise T4BbExactResolveError(f"{label} must be a mapping")
    snapshot: dict[str, float] = {}
    for action_id, number in value.items():
        if type(action_id) is not str:
            raise T4BbExactResolveError(f"{label} action IDs must be strings")
        if type(number) is not float or not math.isfinite(number):
            raise T4BbExactResolveError(
                f"{label}[{action_id!r}] must be a finite strict float"
            )
        snapshot[action_id] = number
    return snapshot


def _strict_distribution_table(
    value: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise T4BbExactResolveError("distribution table must be a mapping")
    expected_fields = {
        "score",
        "raw_score",
        "royalty",
        "bust_rate",
        "fl_rate",
        "fl_type_rates",
        "enumerated_draws",
        "remaining_deck_size",
    }
    snapshot: dict[str, dict[str, Any]] = {}
    for action_id, raw_row in value.items():
        if type(action_id) is not str:
            raise T4BbExactResolveError(
                "distribution table action IDs must be strings"
            )
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_fields:
            raise T4BbExactResolveError(
                f"distribution row fields drifted: {action_id}"
            )
        row = dict(raw_row)
        for name in ("score", "raw_score", "royalty", "bust_rate", "fl_rate"):
            number = row[name]
            if type(number) is not float or not math.isfinite(number):
                raise T4BbExactResolveError(
                    f"distribution {action_id}.{name} must be a finite strict float"
                )
        raw_rates = row["fl_type_rates"]
        if not isinstance(raw_rates, Mapping) or set(raw_rates) != set(
            _FL_TYPE_RATE_FIELDS
        ):
            raise T4BbExactResolveError(
                f"distribution {action_id}.fl_type_rates fields drifted"
            )
        rates = dict(raw_rates)
        for name in _FL_TYPE_RATE_FIELDS:
            number = rates[name]
            if type(number) is not float or not math.isfinite(number):
                raise T4BbExactResolveError(
                    f"distribution {action_id}.fl_type_rates[{name!r}] "
                    "must be a finite strict float"
                )
        row["fl_type_rates"] = rates
        for name in ("enumerated_draws", "remaining_deck_size"):
            if isinstance(row[name], bool) or type(row[name]) is not int:
                raise T4BbExactResolveError(
                    f"distribution {action_id}.{name} must be a strict integer"
                )
        snapshot[action_id] = row
    return snapshot


def resolve_t4_first_bb_exact(information: InfoSetKey) -> T4BbExactResolution:
    """Enumerate and score every legal BB T4 placement under the declared belief."""

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
    result = T4BbExactResolution(
        information=information,
        utility_by_action_id=MappingProxyType(dict(utility)),
        distribution_by_action_id=MappingProxyType(
            {
                action_id: MappingProxyType(
                    {
                        **{
                            name: row[name]
                            for name in row
                            if name != "fl_type_rates"
                        },
                        "fl_type_rates": MappingProxyType(
                            dict(row["fl_type_rates"])
                        ),
                    }
                )
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


def verify_t4_first_bb_exact(
    information: InfoSetKey,
    result: T4BbExactResolution,
    *,
    expected_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Freshly recompute the uniform-deal solve and verify every result binding."""

    _validate_infoset(information)
    if not isinstance(result, T4BbExactResolution):
        raise TypeError("result must be T4BbExactResolution")
    if result.information != information:
        raise T4BbExactResolveError("result infoset does not match verification input")
    if type(result.manifest_json) is not str:
        raise T4BbExactResolveError("result manifest must be a strict string")
    _require_sha256(result.manifest_sha256, label="result manifest SHA256")
    try:
        manifest = json.loads(result.manifest_json)
    except json.JSONDecodeError as exc:
        raise T4BbExactResolveError("result manifest is not JSON") from exc
    if _canonical_json(manifest) != result.manifest_json:
        raise T4BbExactResolveError("result manifest is not canonical JSON")
    computed_manifest_sha256 = hashlib.sha256(
        result.manifest_json.encode("utf-8")
    ).hexdigest()
    if computed_manifest_sha256 != result.manifest_sha256:
        raise T4BbExactResolveError("result manifest SHA256 mismatch")
    if expected_manifest_sha256 is not None:
        if (
            not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_manifest_sha256)
        ):
            raise T4BbExactResolveError("expected manifest SHA256 is invalid")
        if result.manifest_sha256 != expected_manifest_sha256:
            raise T4BbExactResolveError(
                "result does not match externally trusted manifest SHA256"
            )
    _verify_source_binding(manifest.get("source_binding"))

    utility, metrics, optimal, selected, policy = _compute_exact_tables(information)
    actual_utility = _strict_float_table(
        result.utility_by_action_id,
        label="uniform-deal utility table",
    )
    if actual_utility != utility:
        raise T4BbExactResolveError("uniform-deal utility table mismatch")
    actual_metrics = _strict_distribution_table(
        {
            action_id: {
                **{name: row[name] for name in row if name != "fl_type_rates"},
                "fl_type_rates": dict(row["fl_type_rates"]),
            }
            for action_id, row in result.distribution_by_action_id.items()
        }
    )
    if actual_metrics != metrics:
        raise T4BbExactResolveError("uniform-deal distribution table mismatch")
    if (
        type(result.optimal_action_ids) is not tuple
        or any(type(action_id) is not str for action_id in result.optimal_action_ids)
    ):
        raise T4BbExactResolveError("optimal action IDs must be a tuple of strings")
    if result.optimal_action_ids != optimal:
        raise T4BbExactResolveError("optimal action set mismatch")
    if type(result.selected_action_id) is not str:
        raise T4BbExactResolveError("selected action ID must be a strict string")
    if result.selected_action_id != selected:
        raise T4BbExactResolveError("selected action mismatch")
    actual_policy = _strict_float_table(
        result.action_probabilities,
        label="exact uniform-deal policy",
    )
    if actual_policy != policy:
        raise T4BbExactResolveError("exact uniform-deal policy mismatch")
    expected_manifest = _build_manifest(
        information,
        utility,
        metrics,
        optimal,
        selected,
        policy,
    )
    if manifest != expected_manifest:
        raise T4BbExactResolveError("result manifest content mismatch")

    # Independent truth-boundary assertions do not rely on the manifest builder.
    fixed_truth = {
        "schema": T4_BB_EXACT_RESOLVE_SCHEMA,
        "method": T4_BB_EXACT_METHOD,
        "phase": "t4_first",
        "actor": "bb",
        "turn": 4,
        "utility_perspective": "bb",
        "belief_model": T4_BB_BELIEF_MODEL,
        "opponent_response_model": T4_BB_OPPONENT_RESPONSE_MODEL,
        "all_legal_actions_enumerated": True,
        "all_opponent_draws_enumerated": True,
        "chance_sampling_used": False,
        "policy_sampling_used": False,
        "opponent_hidden_cards_used": False,
        "exact_under_declared_belief": True,
        "bayes_posterior_used": False,
        "algorithm_component_ready": True,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "global_unseen_state_policy_claim": False,
        "serving_changed": False,
    }
    for name, expected in fixed_truth.items():
        if manifest.get(name) != expected:
            raise T4BbExactResolveError(f"fixed manifest truth drifted: {name}")
    if manifest.get("infoset_sha256") != information.digest():
        raise T4BbExactResolveError("manifest infoset binding mismatch")
    return MappingProxyType(
        {
            "verified": True,
            "manifest_sha256": result.manifest_sha256,
            "infoset_sha256": information.digest(),
            "legal_action_count": len(utility),
            "optimal_action_count": len(optimal),
            "selected_action_id": selected,
            "exact_under_declared_belief": True,
            "promotion_eligible": False,
            "runtime_integrated": False,
        }
    )


_RESOLVE_IMPL = resolve_t4_first_bb_exact
_VERIFY_IMPL = verify_t4_first_bb_exact
_CANONICAL_LOCAL_HELPERS = (
    ("_canonical_json", _canonical_json),
    ("_canonical_sha256", _canonical_sha256),
    ("_file_sha256", _file_sha256),
    ("_source_paths", _source_paths),
    ("_require_sha256", _require_sha256),
    ("_live_source_sha256s", _live_source_sha256s),
    ("_board", _board),
    ("_validate_infoset", _validate_infoset),
    ("_distribution_payload", _distribution_payload),
    ("_compute_exact_tables", _compute_exact_tables),
    ("_live_runtime_semantic_binding", _live_runtime_semantic_binding),
    ("_verify_runtime_semantic_binding", _verify_runtime_semantic_binding),
    ("_verify_source_binding", _verify_source_binding),
    ("_live_source_binding", _live_source_binding),
    ("_table_binding", _table_binding),
    ("_policy_binding", _policy_binding),
    ("_build_manifest", _build_manifest),
    ("_strict_float_table", _strict_float_table),
    ("_strict_distribution_table", _strict_distribution_table),
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
            raise T4BbExactResolveError("canonical runtime guard alias drifted")
        if globals().get("_CANONICAL_MODULE_ALIASES") is not canonical_module_aliases:
            raise T4BbExactResolveError("canonical module alias set drifted")
        for name, expected in canonical_module_aliases:
            if globals().get(name) is not expected:
                raise T4BbExactResolveError(
                    f"canonical module alias drifted: {name}"
                )
        if globals().get("_CANONICAL_EXTERNALS") is not canonical_externals:
            raise T4BbExactResolveError("canonical external identity set drifted")
        if globals().get("_CANONICAL_LOCAL_HELPERS") is not canonical_local_helpers:
            raise T4BbExactResolveError("canonical helper identity set drifted")
        if (
            globals().get("_CANONICAL_SOURCE_SHA256S")
            is not canonical_source_sha256s
        ):
            raise T4BbExactResolveError("canonical source snapshot alias drifted")
        if globals().get(
            "_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256"
        ) != canonical_runtime_sha256:
            raise T4BbExactResolveError("canonical runtime snapshot alias drifted")
        canonical_guard(
            expected_module_aliases=canonical_module_aliases,
            expected_externals=canonical_externals,
            expected_local_helpers=canonical_local_helpers,
            expected_runtime_binding_sha256=canonical_runtime_sha256,
            expected_source_sha256s=canonical_source_sha256s,
        )
        return function(*args, **kwargs)

    return guarded


resolve_t4_first_bb_exact = _guard_entrypoint(_RESOLVE_IMPL)
verify_t4_first_bb_exact = _guard_entrypoint(_VERIFY_IMPL)


__all__ = [
    "T4_BB_BELIEF_MODEL",
    "T4_BB_EXACT_METHOD",
    "T4_BB_EXACT_RESOLVE_SCHEMA",
    "T4_BB_OPPONENT_RESPONSE_MODEL",
    "T4BbExactResolution",
    "T4BbExactResolveError",
    "resolve_t4_first_bb_exact",
    "verify_t4_first_bb_exact",
]
