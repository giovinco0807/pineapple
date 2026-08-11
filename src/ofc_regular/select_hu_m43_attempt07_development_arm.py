"""Fail-closed development-arm selector for M4.3 Attempt07.

The selector consumes exactly 100 development teacher rows and compares the
four frozen R/V arms using only their locked action's disjoint A128 raw paired
deltas.  It never fits a model, selects a threshold, authorizes the future
audit, proves runtime trajectory cancellation, or changes ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import ACTION_KEY_SCHEMA, ActionKey, action_key
from .action_space import generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_ID,
    ATTEMPT06_FROZEN_MODEL_SHA256,
)
from .hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_ARMS,
    M43_ATTEMPT07_PLAN_SHA256,
    M43_ATTEMPT07_PROFILES,
    enumerate_attempt07_seed_schedules,
    validate_attempt07_plan,
)
from .hu_m43_attempt07_teacher import (
    ATTEMPT07_ARM_SPECS,
    ATTEMPT07_ASSESSMENT_SAMPLES,
    ATTEMPT07_RERANK_SAMPLES,
    ATTEMPT07_SCREEN_SAMPLES,
    ATTEMPT07_SHORTLIST_K,
    ATTEMPT07_SOLVER_ID,
    ATTEMPT07_TEACHER_SCHEMA,
    ATTEMPT07_TOP_K,
    ATTEMPT07_VETO_SAMPLES,
)
from .hu_m4_t1_teacher import M4_PAIRED_DELTA_SUMMARY_SCHEMA
from .hu_m4_teacher_contract import T1_SECOND_LIVE_SCHEDULE
from .run_hu_m43_attempt07_development import (
    ATTEMPT07_PROVENANCE_SCHEMA,
    ATTEMPT07_SHARD_ROW_SCHEMA,
    _canonical_sha256 as _runner_canonical_sha256,
    _fixed_contract as _runner_fixed_contract,
    _root_input_sha256 as _runner_root_input_sha256,
    _validate_provenance as _validate_runner_provenance,
)


ATTEMPT07_DEVELOPMENT_ROW_SCHEMA = (
    ATTEMPT07_SHARD_ROW_SCHEMA
)
ATTEMPT07_DEVELOPMENT_SELECTION_SCHEMA = (
    "hu_m43_attempt07_development_arm_selection_v1"
)
_GO_STATUS = "go_write_separate_winner_freeze_only"
_NO_GO_STATUS = "no_go_close_attempt07_development"
_ROOT_COUNT = 100
_ARM_KEY_MAP = {
    teacher_name: teacher_name.lower()
    for teacher_name, _rerank, _veto in ATTEMPT07_ARM_SPECS
}
_SHA256_CHARS = frozenset("0123456789abcdef")
_PAIR_FIELDS = (
    "mean",
    "standard_error",
    "std",
    "min",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "max",
    "lt0_rate",
    "le_neg6_rate",
    "le_neg12_rate",
    "le_neg20_rate",
)
_PAIR_SUMMARY_KEYS = frozenset({"schema", "count", *_PAIR_FIELDS})
_OUTER_ROW_KEYS = frozenset(
    {
        "schema",
        "root_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
        "teacher",
    }
)
_TEACHER_KEYS = frozenset(
    {
        "status",
        "schema",
        "solver_id",
        "street",
        "seat",
        "to_act_order",
        "observation_fingerprint",
        "policy_observation",
        "action_key_schema",
        "frozen_candidate_generator",
        "legal_action_mapping",
        "legal_actions",
        "baseline_action_key",
        "baseline_original_legal_index",
        "learned_top8_original_legal_indices",
        "learned_top8_action_keys",
        "proposal_mapping",
        "shortlist_proposal_positions",
        "shortlist_original_legal_indices",
        "shortlist_action_keys",
        "shortlist_mapping",
        "screen",
        "rerank",
        "veto",
        "arm_order",
        "arms",
        "assessment",
        "belief_digests",
        "rng_key_digests",
        "sample_independence",
        "root_selection_lock",
        "search_config",
        "continuation_policy",
        "live_schedule",
        "child_information_set_count",
        "teacher_value_status",
        "runtime_gate_allowed",
        "profile_activation_allowed",
        "current_profile_resolved",
        "development_only",
    }
)
_LEGAL_ACTION_ROW_KEYS = frozenset(
    {
        "original_legal_index",
        "action_key",
        "model_rank_score",
        "model_rank_disagreement",
        "in_learned_top8",
        "is_explicit_baseline",
    }
)
_PHASE_ACTION_ROW_KEYS = frozenset(
    {
        "phase_position",
        "action_key",
        "mean",
        "standard_error",
        "is_explicit_baseline",
        "paired_delta_vs_baseline",
        "raw_paired_deltas_vs_baseline",
        "raw_paired_deltas_sha256",
        "action_values_sha256",
    }
)
_ARM_ROW_KEYS = frozenset(
    {
        "rerank_prefix",
        "veto_prefix",
        "rerank_winner_action_key",
        "rerank_winner_position",
        "rerank_winner_is_baseline",
        "veto_action_position",
        "veto_paired_delta_vs_baseline",
        "veto_raw_paired_deltas_vs_baseline",
        "veto_raw_paired_deltas_sha256",
        "veto_checks",
        "veto_pass",
        "selected_action_key",
        "override_fired",
        "exact_baseline_fallback",
        "fallback_reason",
        "second_best_promotion_allowed",
    }
)
_FORBIDDEN_HIDDEN_KEYS = frozenset(
    {
        "deck",
        "deck_tail",
        "draw_order",
        "draw_pile",
        "future_cards",
        "future_rollouts",
        "hidden_cards",
        "opponent_discard_cards",
        "opponent_private_discards",
        "remaining_cards",
        "remaining_deck",
        "replay_truth",
        "replay_world",
        "true_dead_cards",
        "true_opponent_private_discards",
        "world",
        "world_state",
    }
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _close(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _raw_digest(values: Sequence[float]) -> str:
    return hashlib.sha256(
        json.dumps(
            [float(value) for value in values],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _ordered_key_digest(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()


def _set_key_digest(tokens: Sequence[str]) -> str:
    keys = [ActionKey.from_token(token) for token in tokens]
    if len(set(keys)) != len(keys):
        raise ValueError("ActionKey set contains duplicates")
    ordered = [key.to_token() for key in sorted(keys, key=ActionKey.sort_key)]
    return _ordered_key_digest(ordered)


def _read_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        data.decode("utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Attempt07 input row {line_number} must be a mapping"
            )
        rows.append(payload)
    if len(rows) != _ROOT_COUNT:
        raise ValueError("Attempt07 development input must contain exactly 100 rows")
    return rows


def _read_plan_bytes(data: bytes) -> dict[str, Any]:
    if hashlib.sha256(data).hexdigest() != M43_ATTEMPT07_PLAN_SHA256:
        raise ValueError("Attempt07 authoritative plan SHA-256 changed")
    payload = json.loads(data.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Attempt07 plan must be a mapping")
    validate_attempt07_plan(payload)
    return payload


def _validate_mapping_payload(
    value: Any,
    *,
    expected_tokens: Sequence[str],
    label: str,
    allow_extra: bool = False,
) -> None:
    payload = _mapping(value, label)
    mapping_keys = {
        "action_count",
        "action_keys",
        "action_set_digest",
        "action_order_digest",
    }
    if (allow_extra and not mapping_keys.issubset(payload)) or (
        not allow_extra and set(payload) != mapping_keys
    ):
        raise ValueError(f"{label} fields changed")
    tokens = [str(token) for token in _sequence(payload.get("action_keys"), label)]
    for token in tokens:
        ActionKey.from_token(token)
    if (
        payload.get("action_count") != len(expected_tokens)
        or tokens != list(expected_tokens)
        or payload.get("action_order_digest") != _ordered_key_digest(tokens)
        or payload.get("action_set_digest") != _set_key_digest(tokens)
    ):
        raise ValueError(f"{label} mapping changed")


def _summary_from_raw(
    raw_value: Any, *, sample_count: int, label: str
) -> tuple[list[float], dict[str, float], dict[str, float]]:
    raw = _sequence(raw_value, f"{label}.raw")
    if len(raw) != sample_count:
        raise ValueError(f"{label} raw paired delta count changed")
    values = [_finite(value, f"{label}.raw[{index}]") for index, value in enumerate(raw)]
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if sample_count > 1 else 0.0
    summary = {
        "mean": float(np.mean(array)),
        "standard_error": std / math.sqrt(sample_count),
        "std": std,
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01, method="linear")),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "p25": float(np.quantile(array, 0.25, method="linear")),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p75": float(np.quantile(array, 0.75, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "p99": float(np.quantile(array, 0.99, method="linear")),
        "max": float(np.max(array)),
        "lt0_rate": float(np.mean(array < 0.0)),
        "le_neg6_rate": float(np.mean(array <= -6.0)),
        "le_neg12_rate": float(np.mean(array <= -12.0)),
        "le_neg20_rate": float(np.mean(array <= -20.0)),
    }
    losses = {
        "p95": max(0.0, -summary["p05"]),
        "p99": max(0.0, -summary["p01"]),
        "max": max(0.0, -summary["min"]),
    }
    return values, summary, losses


def _validate_recorded_summary(
    value: Any,
    *,
    expected: Mapping[str, float],
    sample_count: int,
    label: str,
) -> None:
    summary = _mapping(value, label)
    if (
        set(summary) != _PAIR_SUMMARY_KEYS
        or summary.get("schema") != M4_PAIRED_DELTA_SUMMARY_SCHEMA
        or summary.get("count") != sample_count
    ):
        raise ValueError(f"{label} schema or count changed")
    for field in _PAIR_FIELDS:
        if not _close(summary.get(field), expected[field]):
            raise ValueError(f"{label}.{field} disagrees with raw paired deltas")


def _validate_phase_actions(
    value: Any,
    *,
    expected_tokens: Sequence[str],
    baseline_token: str,
    sample_count: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    rows = _sequence(value, f"{label}.actions")
    if len(rows) != len(expected_tokens):
        raise ValueError(f"{label} action count changed")
    by_key: dict[str, dict[str, Any]] = {}
    baseline_mean: float | None = None
    parsed: list[tuple[dict[str, Any], list[float], dict[str, float]]] = []
    for position, (raw_row, expected_token) in enumerate(
        zip(rows, expected_tokens, strict=True)
    ):
        row = dict(_mapping(raw_row, f"{label}.actions[{position}]"))
        if set(row) != _PHASE_ACTION_ROW_KEYS:
            raise ValueError(f"{label} action row fields changed")
        token = row.get("action_key")
        if (
            row.get("phase_position") != position
            or token != expected_token
            or not isinstance(token, str)
        ):
            raise ValueError(f"{label} action position/key changed")
        ActionKey.from_token(token)
        if row.get("is_explicit_baseline") is not (token == baseline_token):
            raise ValueError(f"{label} baseline flag changed")
        mean = _finite(row.get("mean"), f"{label}.{position}.mean")
        standard_error = _finite(
            row.get("standard_error"), f"{label}.{position}.standard_error"
        )
        if standard_error < 0.0:
            raise ValueError(f"{label} standard error is negative")
        raw, summary, _losses = _summary_from_raw(
            row.get("raw_paired_deltas_vs_baseline"),
            sample_count=sample_count,
            label=f"{label}.{position}",
        )
        if row.get("raw_paired_deltas_sha256") != _raw_digest(raw):
            raise ValueError(f"{label} raw paired delta digest changed")
        if not _is_sha256(row.get("action_values_sha256")):
            raise ValueError(f"{label} action-values digest is invalid")
        _validate_recorded_summary(
            row.get("paired_delta_vs_baseline"),
            expected=summary,
            sample_count=sample_count,
            label=f"{label}.{position}.summary",
        )
        if token == baseline_token:
            baseline_mean = mean
            if any(value != 0.0 for value in raw):
                raise ValueError(f"{label} baseline raw paired deltas are not zero")
        parsed.append((row, raw, summary))
        if token in by_key:
            raise ValueError(f"{label} contains duplicate ActionKeys")
        by_key[token] = row
    if baseline_mean is None:
        raise ValueError(f"{label} lacks the explicit baseline")
    for row, _raw, summary in parsed:
        if not _close(float(row["mean"]) - baseline_mean, summary["mean"]):
            raise ValueError(f"{label} action mean disagrees with paired raw mean")
    return by_key


def _validate_hidden_information(row: Mapping[str, Any]) -> None:
    violations: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.lower()
                child_path = f"{path}.{key}"
                if normalized in _FORBIDDEN_HIDDEN_KEYS:
                    violations.append(child_path)
                is_explicit_disabled_lcb_guard = (
                    normalized == "teacher_ev_or_lcb_runtime_gate_allowed"
                    and child is False
                )
                if (
                    "lcb" in normalized or "lower_confidence_bound" in normalized
                ) and not is_explicit_disabled_lcb_guard:
                    violations.append(child_path)
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(row, "row")
    if violations:
        raise ValueError(
            "Attempt07 row contains forbidden hidden/runtime-gate fields: "
            + ",".join(sorted(set(violations)))
        )


def _phase_winner_token(
    rows: Sequence[Any], *, baseline_token: str, baseline_preferred: bool
) -> tuple[int, str]:
    scored: list[tuple[float, ActionKey, int, str]] = []
    for position, value in enumerate(rows):
        row = _mapping(value, f"phase.actions[{position}]")
        token = str(row.get("action_key"))
        key = ActionKey.from_token(token)
        scored.append((_finite(row.get("mean"), "phase.mean"), key, position, token))
    best = max(value[0] for value in scored)
    tied = [value for value in scored if value[0] == best]
    if baseline_preferred:
        for _mean, _key, position, token in tied:
            if token == baseline_token:
                return position, token
    chosen = min(tied, key=lambda value: value[1].sort_key())
    return chosen[2], chosen[3]


def _validate_provenance(
    value: Any,
    *,
    root_index: int,
    profile: str,
    hand_seed: int,
    expected_seeds: Mapping[str, int],
    observation: ActorObservation,
    baseline_token: str,
) -> Mapping[str, Any]:
    """Reconstruct and validate the producer-owned immutable provenance."""

    provenance = _mapping(value, f"root[{root_index}].provenance")
    if provenance.get("schema") != ATTEMPT07_PROVENANCE_SCHEMA:
        raise ValueError(f"Attempt07 provenance schema changed at root {root_index}")
    run_id = provenance.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Attempt07 provenance run_id missing at root {root_index}")
    batch = provenance.get("batch_child_selectors")
    native_threads = provenance.get("native_batch_threads")
    if batch is not True:
        raise ValueError(
            f"Attempt07 provenance requires batch_child_selectors at root {root_index}"
        )
    if type(native_threads) is not int or native_threads != 4:
        raise ValueError(
            f"Attempt07 provenance native_batch_threads is not 4 at root {root_index}"
        )
    fixed_contract = _runner_fixed_contract(
        root_index=root_index,
        root_profile=profile,
        seeds=expected_seeds,
        run_id=run_id,
        plan_sha256=M43_ATTEMPT07_PLAN_SHA256,
        ai_profiles_sha256=AI_PROFILES_SHA256,
        model_sha256=ATTEMPT06_FROZEN_MODEL_SHA256,
        batch_child_selectors=batch,
        native_batch_threads=native_threads,
    )
    config_sha256 = _runner_canonical_sha256(fixed_contract)
    root_input_sha256 = _runner_root_input_sha256(
        root_index=root_index,
        hand_seed=hand_seed,
        root_profile=profile,
        observation=observation,
        baseline_action_key=baseline_token,
    )
    _validate_runner_provenance(
        provenance,
        fixed_contract=fixed_contract,
        config_sha256=config_sha256,
        root_input_sha256=root_input_sha256,
    )
    for key in (
        "current_profile_resolved",
        "opponent_private_discard_input_allowed",
        "teacher_values_are_realized_match_ev",
        "teacher_ev_or_lcb_runtime_gate_allowed",
        "fit_allowed",
        "threshold_selection_allowed",
        "runtime_activation_allowed",
        "full_replacement_enabled",
    ):
        if provenance.get(key) is not False:
            raise ValueError(
                f"Attempt07 provenance guard {key} changed at root {root_index}"
            )
    if provenance.get("development_only") is not True:
        raise ValueError(
            f"Attempt07 provenance development_only changed at root {root_index}"
        )
    return provenance


def _validate_teacher_identity(
    teacher: Mapping[str, Any], *, root_index: int
) -> None:
    if set(teacher) != _TEACHER_KEYS:
        raise ValueError(f"Attempt07 teacher fields changed at root {root_index}")
    expected = {
        "status": "ok",
        "schema": ATTEMPT07_TEACHER_SCHEMA,
        "solver_id": ATTEMPT07_SOLVER_ID,
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "action_key_schema": ACTION_KEY_SCHEMA,
        "teacher_value_status": "diagnostic_not_match_EV",
        "runtime_gate_allowed": False,
        "profile_activation_allowed": False,
        "current_profile_resolved": False,
        "development_only": True,
    }
    for key, value in expected.items():
        if teacher.get(key) != value or type(teacher.get(key)) is not type(value):
            raise ValueError(
                f"Attempt07 teacher identity {key} changed at root {root_index}"
            )
    frozen = _mapping(
        teacher.get("frozen_candidate_generator"), "frozen_candidate_generator"
    )
    expected_frozen = {
        "family": "lambda_rank",
        "model_id": ATTEMPT06_FROZEN_MODEL_ID,
        "artifact_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "purpose": "candidate_generation_only",
        "runtime_authorized": False,
        "profile_runtime_feature": False,
    }
    if (
        dict(frozen) != expected_frozen
        or frozen.get("runtime_authorized") is not False
        or frozen.get("profile_runtime_feature") is not False
    ):
        raise ValueError(
            f"Attempt07 candidate-generator lineage changed at root {root_index}"
        )


def _validate_legal_and_proposal_mappings(
    row: Mapping[str, Any], teacher: Mapping[str, Any], *, root_index: int
) -> tuple[ActorObservation, str, list[str], list[str], list[str]]:
    observation_payload = _mapping(
        row.get("policy_observation"), f"root[{root_index}].policy_observation"
    )
    observation = ActorObservation.from_dict(observation_payload)
    if (
        observation.to_dict() != observation_payload
        or observation.street != "T1"
        or observation.seat != "second"
        or observation.to_act_order != "second"
        or teacher.get("policy_observation") != observation_payload
        or teacher.get("observation_fingerprint") != observation.fingerprint()
    ):
        raise ValueError(f"Attempt07 observation mapping changed at root {root_index}")
    generated = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    legal_tokens = [action_key(action).to_token() for action in generated]
    _validate_mapping_payload(
        teacher.get("legal_action_mapping"),
        expected_tokens=legal_tokens,
        label=f"root[{root_index}].legal_action_mapping",
    )
    legal_rows = _sequence(
        teacher.get("legal_actions"), f"root[{root_index}].legal_actions"
    )
    if len(legal_rows) != len(legal_tokens):
        raise ValueError(f"Attempt07 legal action rows changed at root {root_index}")
    baseline_token = teacher.get("baseline_action_key")
    if not isinstance(baseline_token, str):
        raise ValueError(f"Attempt07 baseline ActionKey missing at root {root_index}")
    ActionKey.from_token(baseline_token)
    baseline_index = _strict_int(
        teacher.get("baseline_original_legal_index"), "baseline_original_legal_index"
    )
    if (
        not 0 <= baseline_index < len(legal_tokens)
        or legal_tokens[baseline_index] != baseline_token
        or row.get("baseline_action_key") != baseline_token
    ):
        raise ValueError(f"Attempt07 baseline mapping changed at root {root_index}")
    ranked_nonbaseline: list[tuple[float, ActionKey, int, str]] = []
    for position, (raw_action, token) in enumerate(
        zip(legal_rows, legal_tokens, strict=True)
    ):
        action_row = _mapping(raw_action, f"legal_actions[{position}]")
        if set(action_row) != _LEGAL_ACTION_ROW_KEYS:
            raise ValueError(f"Attempt07 legal row fields changed at root {root_index}")
        score = _finite(action_row.get("model_rank_score"), "model_rank_score")
        disagreement = _finite(
            action_row.get("model_rank_disagreement"), "model_rank_disagreement"
        )
        if (
            action_row.get("original_legal_index") != position
            or action_row.get("action_key") != token
            or disagreement < 0.0
            or action_row.get("is_explicit_baseline")
            is not (position == baseline_index)
        ):
            raise ValueError(f"Attempt07 legal row mapping changed at root {root_index}")
        if position != baseline_index:
            ranked_nonbaseline.append(
                (score, ActionKey.from_token(token), position, token)
            )
    expected_ranked = sorted(
        ranked_nonbaseline, key=lambda item: (-item[0], item[1].sort_key())
    )[:ATTEMPT07_TOP_K]
    expected_top_indices = [item[2] for item in expected_ranked]
    expected_top_tokens = [item[3] for item in expected_ranked]
    if (
        teacher.get("learned_top8_original_legal_indices") != expected_top_indices
        or teacher.get("learned_top8_action_keys") != expected_top_tokens
    ):
        raise ValueError(f"Attempt07 top8 mapping changed at root {root_index}")
    top_set = set(expected_top_tokens)
    for raw_action, token in zip(legal_rows, legal_tokens, strict=True):
        if _mapping(raw_action, "legal action").get(
            "in_learned_top8"
        ) is not (token in top_set):
            raise ValueError(f"Attempt07 top8 flag changed at root {root_index}")
    proposal_tokens = [*expected_top_tokens, baseline_token]
    _validate_mapping_payload(
        teacher.get("proposal_mapping"),
        expected_tokens=proposal_tokens,
        label=f"root[{root_index}].proposal_mapping",
    )
    return observation, baseline_token, legal_tokens, expected_top_tokens, proposal_tokens


def _validate_search_phases(
    teacher: Mapping[str, Any],
    *,
    root_index: int,
    baseline_token: str,
    legal_tokens: Sequence[str],
    proposal_tokens: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    screen = _mapping(teacher.get("screen"), f"root[{root_index}].screen")
    if set(screen) != {
        "action_count",
        "action_keys",
        "action_set_digest",
        "action_order_digest",
        "sample_count",
        "actions",
    }:
        raise ValueError(f"Attempt07 S8 fields changed at root {root_index}")
    _validate_mapping_payload(
        screen,
        expected_tokens=proposal_tokens,
        label=f"root[{root_index}].screen",
        allow_extra=True,
    )
    if screen.get("sample_count") != ATTEMPT07_SCREEN_SAMPLES:
        raise ValueError(f"Attempt07 S8 sample count changed at root {root_index}")
    screen_by_key = _validate_phase_actions(
        screen.get("actions"),
        expected_tokens=proposal_tokens,
        baseline_token=baseline_token,
        sample_count=ATTEMPT07_SCREEN_SAMPLES,
        label=f"root[{root_index}].screen",
    )
    ranked_screen = sorted(
        proposal_tokens[:-1],
        key=lambda token: (
            -float(screen_by_key[token]["mean"]),
            ActionKey.from_token(token).sort_key(),
        ),
    )[:ATTEMPT07_SHORTLIST_K]
    shortlist_positions = [proposal_tokens.index(token) for token in ranked_screen]
    shortlist_legal_indices = [legal_tokens.index(token) for token in ranked_screen]
    if (
        teacher.get("shortlist_proposal_positions") != shortlist_positions
        or teacher.get("shortlist_original_legal_indices")
        != shortlist_legal_indices
        or teacher.get("shortlist_action_keys") != ranked_screen
    ):
        raise ValueError(f"Attempt07 K3 mapping changed at root {root_index}")
    _validate_mapping_payload(
        teacher.get("shortlist_mapping"),
        expected_tokens=ranked_screen,
        label=f"root[{root_index}].shortlist_mapping",
    )

    rerank_tokens = [*ranked_screen, baseline_token]
    rerank = _mapping(teacher.get("rerank"), f"root[{root_index}].rerank")
    if set(rerank) != {
        "action_count",
        "action_keys",
        "action_set_digest",
        "action_order_digest",
        "sample_count",
        "prefix_semantics",
        "baseline_preferred_on_exact_best_tie",
        "prefixes",
    }:
        raise ValueError(f"Attempt07 rerank fields changed at root {root_index}")
    _validate_mapping_payload(
        rerank,
        expected_tokens=rerank_tokens,
        label=f"root[{root_index}].rerank",
        allow_extra=True,
    )
    if (
        rerank.get("sample_count") != ATTEMPT07_RERANK_SAMPLES
        or rerank.get("prefix_semantics")
        != "R32_is_first_32_rows_of_same_R64_batch"
        or rerank.get("baseline_preferred_on_exact_best_tie") is not True
    ):
        raise ValueError(f"Attempt07 rerank contract changed at root {root_index}")
    rerank_prefixes = _mapping(rerank.get("prefixes"), "rerank.prefixes")
    if set(rerank_prefixes) != {"R32", "R64"}:
        raise ValueError(f"Attempt07 rerank prefixes changed at root {root_index}")
    rerank_rows: dict[str, Sequence[Any]] = {}
    rerank_winners: dict[str, tuple[int, str]] = {}
    for name, count in (("R32", 32), ("R64", 64)):
        prefix = _mapping(rerank_prefixes.get(name), f"rerank.{name}")
        if set(prefix) != {
            "sample_count",
            "winner_position",
            "winner_action_key",
            "winner_is_explicit_baseline",
            "actions",
        } or prefix.get("sample_count") != count:
            raise ValueError(f"Attempt07 {name} fields changed at root {root_index}")
        rows = _sequence(prefix.get("actions"), f"rerank.{name}.actions")
        _validate_phase_actions(
            rows,
            expected_tokens=rerank_tokens,
            baseline_token=baseline_token,
            sample_count=count,
            label=f"root[{root_index}].rerank.{name}",
        )
        winner = _phase_winner_token(
            rows, baseline_token=baseline_token, baseline_preferred=True
        )
        if (
            prefix.get("winner_position") != winner[0]
            or prefix.get("winner_action_key") != winner[1]
            or prefix.get("winner_is_explicit_baseline")
            is not (winner[1] == baseline_token)
        ):
            raise ValueError(f"Attempt07 {name} winner changed at root {root_index}")
        rerank_rows[name] = rows
        rerank_winners[name] = winner
    for position in range(len(rerank_tokens)):
        r32_raw = _mapping(rerank_rows["R32"][position], "R32 row").get(
            "raw_paired_deltas_vs_baseline"
        )
        r64_raw = _sequence(
            _mapping(rerank_rows["R64"][position], "R64 row").get(
                "raw_paired_deltas_vs_baseline"
            ),
            "R64 raw",
        )
        if list(r32_raw) != list(r64_raw[:32]):
            raise ValueError(f"Attempt07 R32 is not an R64 prefix at root {root_index}")

    veto_tokens = list(
        dict.fromkeys(
            winner[1]
            for winner in (rerank_winners["R32"], rerank_winners["R64"])
            if winner[1] != baseline_token
        )
    )
    veto_tokens.append(baseline_token)
    veto = _mapping(teacher.get("veto"), f"root[{root_index}].veto")
    if set(veto) != {
        "action_count",
        "action_keys",
        "action_set_digest",
        "action_order_digest",
        "sample_count",
        "scope",
        "locked_nonbaseline_rerank_positions",
        "prefix_semantics",
        "thresholds",
        "prefixes",
    }:
        raise ValueError(f"Attempt07 veto fields changed at root {root_index}")
    _validate_mapping_payload(
        veto,
        expected_tokens=veto_tokens,
        label=f"root[{root_index}].veto",
        allow_extra=True,
    )
    if (
        veto.get("sample_count") != ATTEMPT07_VETO_SAMPLES
        or veto.get("scope")
        != "unique_nonbaseline_R32_R64_winners_plus_explicit_baseline"
        or veto.get("prefix_semantics")
        != "V64_is_first_64_rows_of_same_V128_batch"
        or veto.get("thresholds")
        != {
            "mean_strictly_greater_than": 0.0,
            "p05_at_least": -25.0,
            "p01_at_least": -40.0,
            "min_at_least": -50.0,
        }
    ):
        raise ValueError(f"Attempt07 veto contract changed at root {root_index}")
    expected_locked_positions = [
        winner[0]
        for winner in (rerank_winners["R32"], rerank_winners["R64"])
        if winner[1] != baseline_token
    ]
    expected_locked_positions = list(dict.fromkeys(expected_locked_positions))
    if veto.get("locked_nonbaseline_rerank_positions") != expected_locked_positions:
        raise ValueError(f"Attempt07 veto scope changed at root {root_index}")
    veto_prefixes = _mapping(veto.get("prefixes"), "veto.prefixes")
    if set(veto_prefixes) != {"V64", "V128"}:
        raise ValueError(f"Attempt07 veto prefixes changed at root {root_index}")
    veto_rows: dict[str, Sequence[Any]] = {}
    veto_by_prefix: dict[str, dict[str, dict[str, Any]]] = {}
    for name, count in (("V64", 64), ("V128", 128)):
        prefix = _mapping(veto_prefixes.get(name), f"veto.{name}")
        if set(prefix) != {"sample_count", "actions"} or prefix.get(
            "sample_count"
        ) != count:
            raise ValueError(f"Attempt07 {name} fields changed at root {root_index}")
        rows = _sequence(prefix.get("actions"), f"veto.{name}.actions")
        veto_by_prefix[name] = _validate_phase_actions(
            rows,
            expected_tokens=veto_tokens,
            baseline_token=baseline_token,
            sample_count=count,
            label=f"root[{root_index}].veto.{name}",
        )
        veto_rows[name] = rows
    for position in range(len(veto_tokens)):
        v64_raw = _mapping(veto_rows["V64"][position], "V64 row").get(
            "raw_paired_deltas_vs_baseline"
        )
        v128_raw = _sequence(
            _mapping(veto_rows["V128"][position], "V128 row").get(
                "raw_paired_deltas_vs_baseline"
            ),
            "V128 raw",
        )
        if list(v64_raw) != list(v128_raw[:64]):
            raise ValueError(f"Attempt07 V64 is not a V128 prefix at root {root_index}")

    assessment = _mapping(
        teacher.get("assessment"), f"root[{root_index}].assessment"
    )
    if set(assessment) != {
        "action_count",
        "action_keys",
        "action_set_digest",
        "action_order_digest",
        "sample_count",
        "scope",
        "diagnostics_only",
        "can_rerank_or_gate",
        "decision_frozen_before_namespace_open",
        "sample_best_action_key",
        "actions",
    }:
        raise ValueError(f"Attempt07 A128 fields changed at root {root_index}")
    _validate_mapping_payload(
        assessment,
        expected_tokens=proposal_tokens,
        label=f"root[{root_index}].assessment",
        allow_extra=True,
    )
    if (
        assessment.get("sample_count") != ATTEMPT07_ASSESSMENT_SAMPLES
        or assessment.get("scope") != "all_top8_plus_explicit_baseline"
        or assessment.get("diagnostics_only") is not True
        or assessment.get("can_rerank_or_gate") is not False
        or assessment.get("decision_frozen_before_namespace_open") is not True
    ):
        raise ValueError(f"Attempt07 A128 contract changed at root {root_index}")
    assessment_rows = _sequence(assessment.get("actions"), "assessment.actions")
    assessment_by_key = _validate_phase_actions(
        assessment_rows,
        expected_tokens=proposal_tokens,
        baseline_token=baseline_token,
        sample_count=ATTEMPT07_ASSESSMENT_SAMPLES,
        label=f"root[{root_index}].assessment",
    )
    _best_position, best_token = _phase_winner_token(
        assessment_rows, baseline_token=baseline_token, baseline_preferred=False
    )
    if assessment.get("sample_best_action_key") != best_token:
        raise ValueError(f"Attempt07 A128 best diagnostic changed at root {root_index}")

    return veto_by_prefix, assessment_by_key


def _validate_arm_rows(
    teacher: Mapping[str, Any],
    *,
    root_index: int,
    baseline_token: str,
    veto_by_prefix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    assessment_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_teacher_order = [name for name, _rerank, _veto in ATTEMPT07_ARM_SPECS]
    if teacher.get("arm_order") != expected_teacher_order:
        raise ValueError(f"Attempt07 arm order changed at root {root_index}")
    arms = _mapping(teacher.get("arms"), f"root[{root_index}].arms")
    if set(arms) != set(expected_teacher_order):
        raise ValueError(f"Attempt07 arm set changed at root {root_index}")
    result: dict[str, dict[str, Any]] = {}
    rerank_prefixes = _mapping(
        _mapping(teacher.get("rerank"), "rerank").get("prefixes"),
        "rerank.prefixes",
    )
    veto = _mapping(teacher.get("veto"), "veto")
    veto_tokens = [
        str(token) for token in _sequence(veto.get("action_keys"), "veto.action_keys")
    ]
    for teacher_name, rerank_count, veto_count in ATTEMPT07_ARM_SPECS:
        arm = _mapping(arms.get(teacher_name), f"arms.{teacher_name}")
        if set(arm) != _ARM_ROW_KEYS:
            raise ValueError(f"Attempt07 {teacher_name} fields changed at root {root_index}")
        rerank_name = f"R{rerank_count}"
        veto_name = f"V{veto_count}"
        rerank = _mapping(rerank_prefixes.get(rerank_name), rerank_name)
        winner_token = str(rerank.get("winner_action_key"))
        winner_position = _strict_int(
            rerank.get("winner_position"), f"{teacher_name}.winner_position"
        )
        winner_is_baseline = winner_token == baseline_token
        if winner_token not in veto_tokens:
            raise ValueError(
                f"Attempt07 {teacher_name} winner absent from veto at root {root_index}"
            )
        veto_position = veto_tokens.index(winner_token)
        veto_row = _mapping(
            veto_by_prefix[veto_name][winner_token], f"{teacher_name}.veto_row"
        )
        veto_raw, veto_summary, _veto_losses = _summary_from_raw(
            veto_row.get("raw_paired_deltas_vs_baseline"),
            sample_count=veto_count,
            label=f"root[{root_index}].{teacher_name}.veto",
        )
        checks = {
            "mean_gt_0": veto_summary["mean"] > 0.0,
            "p05_ge_neg25": veto_summary["p05"] >= -25.0,
            "p01_ge_neg40": veto_summary["p01"] >= -40.0,
            "min_ge_neg50": veto_summary["min"] >= -50.0,
        }
        veto_pass = not winner_is_baseline and all(checks.values())
        fired = veto_pass
        selected_token = winner_token if fired else baseline_token
        if winner_is_baseline:
            fallback_reason = "rerank_winner_is_explicit_baseline"
        elif not veto_pass:
            fallback_reason = "paired_safety_veto_failed"
        else:
            fallback_reason = None
        expected = {
            "rerank_prefix": rerank_name,
            "veto_prefix": veto_name,
            "rerank_winner_action_key": winner_token,
            "rerank_winner_position": winner_position,
            "rerank_winner_is_baseline": winner_is_baseline,
            "veto_action_position": veto_position,
            "veto_checks": checks,
            "veto_pass": veto_pass,
            "selected_action_key": selected_token,
            "override_fired": fired,
            "exact_baseline_fallback": not fired,
            "fallback_reason": fallback_reason,
            "second_best_promotion_allowed": False,
        }
        for key, expected_value in expected.items():
            if arm.get(key) != expected_value or type(arm.get(key)) is not type(
                expected_value
            ):
                raise ValueError(
                    f"Attempt07 {teacher_name}.{key} changed at root {root_index}"
                )
        if (
            arm.get("veto_raw_paired_deltas_vs_baseline") != veto_raw
            or arm.get("veto_raw_paired_deltas_sha256") != _raw_digest(veto_raw)
        ):
            raise ValueError(
                f"Attempt07 {teacher_name} veto raw binding changed at root {root_index}"
            )
        _validate_recorded_summary(
            arm.get("veto_paired_delta_vs_baseline"),
            expected=veto_summary,
            sample_count=veto_count,
            label=f"root[{root_index}].{teacher_name}.veto_summary",
        )

        # This is the only value source used by the development comparison.
        # The V64/V128 summary above is validated but never copied into metrics.
        try:
            assessment_row = assessment_by_key[selected_token]
        except KeyError as exc:
            raise ValueError(
                f"Attempt07 {teacher_name} selected action is absent exactly once "
                f"from A128 at root {root_index}"
            ) from exc
        assessment_raw, assessment_summary, losses = _summary_from_raw(
            assessment_row.get("raw_paired_deltas_vs_baseline"),
            sample_count=ATTEMPT07_ASSESSMENT_SAMPLES,
            label=f"root[{root_index}].{teacher_name}.assessment",
        )
        if not fired:
            if selected_token != baseline_token or any(
                value != 0.0 for value in assessment_raw
            ):
                raise ValueError(
                    f"Attempt07 nonfire lacks exact baseline action fallback at "
                    f"root {root_index} arm {teacher_name}"
                )
        result[_ARM_KEY_MAP[teacher_name]] = {
            "root_index": root_index,
            "fired": fired,
            "selected_action_key": selected_token,
            "assessment_mean": assessment_summary["mean"],
            "assessment_losses": losses,
        }
    return result


def _validate_rng_and_config(
    teacher: Mapping[str, Any],
    *,
    root_index: int,
    expected_seeds: Mapping[str, int],
    provenance: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    expected_counts = {
        "screen_s8": 8,
        "rerank_r64": 64,
        "veto_v128": 128,
        "assessment_a128": 128,
    }
    raw_rng = _mapping(teacher.get("rng_key_digests"), "rng_key_digests")
    if set(raw_rng) != set(expected_counts):
        raise ValueError(f"Attempt07 RNG phases changed at root {root_index}")
    all_rng: list[str] = []
    per_phase: list[set[str]] = []
    for phase, count in expected_counts.items():
        values = [
            str(value)
            for value in _sequence(raw_rng.get(phase), f"rng_key_digests.{phase}")
        ]
        if (
            len(values) != count
            or len(set(values)) != count
            or not all(_is_sha256(value) for value in values)
        ):
            raise ValueError(
                f"Attempt07 {phase} RNG count/digest changed at root {root_index}"
            )
        all_rng.extend(values)
        per_phase.append(set(values))
    for index, left in enumerate(per_phase):
        for right in per_phase[index + 1 :]:
            if not left.isdisjoint(right):
                raise ValueError(
                    f"Attempt07 RNG domains overlap within root {root_index}"
                )
    belief = _mapping(teacher.get("belief_digests"), "belief_digests")
    if set(belief) != set(expected_counts) or not all(
        _is_sha256(belief.get(phase)) for phase in expected_counts
    ):
        raise ValueError(f"Attempt07 belief digests changed at root {root_index}")
    belief_values = [str(belief[phase]) for phase in expected_counts]
    if len(set(belief_values)) != len(belief_values):
        raise ValueError(f"Attempt07 belief domains overlap at root {root_index}")
    if (
        teacher.get("sample_independence")
        != "pairwise_disjoint_S8_R64_V128_A128_particle_rng_keys"
        or teacher.get("root_selection_lock")
        != (
            "top8_fixed_before_S8_then_top3_fixed_before_R64_then_arms_frozen_"
            "after_V128_before_diagnostic_A128"
        )
    ):
        raise ValueError(f"Attempt07 RNG/selection lock changed at root {root_index}")
    config = _mapping(teacher.get("search_config"), "search_config")
    fingerprint = str(teacher.get("observation_fingerprint"))
    expected_teacher_run_id = (
        f"{provenance.get('run_id')}:root={root_index}:"
        f"seed={_mapping(provenance.get('seeds'), 'provenance.seeds').get('hand')}:"
        f"obs={fingerprint}"
    )
    expected_config = {
        "learned_nonbaseline_top_k": 8,
        "baseline_added_exactly_once": True,
        "screen_samples": 8,
        "shortlist_nonbaseline_k": 3,
        "rerank_samples": 64,
        "veto_samples": 128,
        "assessment_samples": 128,
        "screen_seed": expected_seeds["screen"],
        "rerank_seed": expected_seeds["rerank"],
        "veto_seed": expected_seeds["veto"],
        "assessment_seed": expected_seeds["assessment"],
        "child_policy_seed": expected_seeds["child"],
        "run_id": expected_teacher_run_id,
        "model_and_screen_tie_break": "ActionKey",
        "rerank_tie_break": "explicit_baseline_then_ActionKey",
    }
    if set(config) != {*expected_config, "batch_child_selectors"}:
        raise ValueError(f"Attempt07 search config fields changed at root {root_index}")
    for key, expected in expected_config.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError(
                f"Attempt07 search config {key} changed at root {root_index}"
            )
    if config.get("batch_child_selectors") is not True:
        raise ValueError(
            f"Attempt07 batch_child_selectors is not enabled at root {root_index}"
        )
    continuation = _mapping(teacher.get("continuation_policy"), "continuation")
    if continuation != {
        "t2_policy_id": "stage9f_p2",
        "t2_resolution": "explicit_profile_never_current",
        "t2_runtime_status": "p2_fixed",
        "t3_selector": "m3_rust_evaluate_t3",
        "t3_candidate_samples": 1,
        "t3_evaluation_samples": 1,
        "t3_downstream_samples": 1,
        "hypothetical_t4_selector": "m3_rust_evaluate_t4",
        "hypothetical_t4_candidate_samples": 1,
        "hypothetical_t4_evaluation_samples": 1,
        "real_live_t4_exact_unchanged": True,
    }:
        raise ValueError(f"Attempt07 continuation changed at root {root_index}")
    expected_schedule = [
        {"seat": step.seat, "street": step.street, "draw_offset": step.draw_offset}
        for step in T1_SECOND_LIVE_SCHEDULE
    ]
    if teacher.get("live_schedule") != expected_schedule:
        raise ValueError(f"Attempt07 live schedule changed at root {root_index}")
    child_count = teacher.get("child_information_set_count")
    if type(child_count) is not int or child_count < 0:
        raise ValueError(f"Attempt07 child infoset count changed at root {root_index}")
    return all_rng, belief_values


def _validate_row(
    row: Mapping[str, Any],
    *,
    root_index: int,
    expected_seeds: Mapping[str, int],
) -> dict[str, Any]:
    if set(row) != _OUTER_ROW_KEYS:
        raise ValueError(f"Attempt07 outer row fields changed at root {root_index}")
    profile = M43_ATTEMPT07_PROFILES[root_index % len(M43_ATTEMPT07_PROFILES)]
    hand_seed = expected_seeds["hand"]
    if (
        row.get("schema") != ATTEMPT07_DEVELOPMENT_ROW_SCHEMA
        or row.get("root_index") != root_index
        or type(row.get("root_index")) is not int
        or row.get("hand_seed") != hand_seed
        or type(row.get("hand_seed")) is not int
        or row.get("root_profile") != profile
    ):
        raise ValueError(f"Attempt07 root identity changed at root {root_index}")
    _validate_hidden_information(row)
    teacher = _mapping(row.get("teacher"), f"root[{root_index}].teacher")
    _validate_teacher_identity(teacher, root_index=root_index)
    observation, baseline, legal_tokens, _top8, proposal_tokens = (
        _validate_legal_and_proposal_mappings(row, teacher, root_index=root_index)
    )
    provenance = _validate_provenance(
        row.get("provenance"),
        root_index=root_index,
        profile=profile,
        hand_seed=hand_seed,
        expected_seeds=expected_seeds,
        observation=observation,
        baseline_token=baseline,
    )
    veto_by_prefix, assessment_by_key = _validate_search_phases(
        teacher,
        root_index=root_index,
        baseline_token=baseline,
        legal_tokens=legal_tokens,
        proposal_tokens=proposal_tokens,
    )
    arm_rows = _validate_arm_rows(
        teacher,
        root_index=root_index,
        baseline_token=baseline,
        veto_by_prefix=veto_by_prefix,
        assessment_by_key=assessment_by_key,
    )
    rng_keys, belief_digests = _validate_rng_and_config(
        teacher,
        root_index=root_index,
        expected_seeds=expected_seeds,
        provenance=provenance,
    )
    return {
        "root_index": root_index,
        "profile": profile,
        "hand_seed": hand_seed,
        "fingerprint": observation.fingerprint(),
        "baseline_action_key": baseline,
        "rng_keys": rng_keys,
        "belief_digests": belief_digests,
        "config_sha256": provenance["config_sha256"],
        "arms": arm_rows,
    }


def _profile_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    states = len(rows)
    fired = [row for row in rows if row["fired"]]
    state_deltas = [float(row["assessment_mean"]) for row in rows]
    fired_deltas = [float(row["assessment_mean"]) for row in fired]
    false_positives = sum(value <= 0.0 for value in fired_deltas)
    return {
        "states": states,
        "fires": len(fired),
        "mean_delta_per_state": sum(state_deltas) / states if states else None,
        "mean_delta_per_fire": (
            sum(fired_deltas) / len(fired_deltas) if fired_deltas else None
        ),
        "false_positive_fires": false_positives,
        "false_positive_rate_per_fire": (
            false_positives / len(fired_deltas) if fired_deltas else None
        ),
        "maximum_per_fired_root_loss": {
            name: max(float(row["assessment_losses"][name]) for row in fired)
            if fired
            else None
            for name in ("p95", "p99", "max")
        },
    }


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def _arm_report(
    *,
    arm_name: str,
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    gates_plan = _mapping(plan.get("development_go_no_go"), "development gates")
    overall = _profile_metrics(rows)
    by_profile = {
        profile: _profile_metrics(
            [row for row in rows if row["profile"] == profile]
        )
        for profile in M43_ATTEMPT07_PROFILES
    }
    profile_fires = {
        profile: int(by_profile[profile]["fires"])
        for profile in M43_ATTEMPT07_PROFILES
    }
    tails = overall["maximum_per_fired_root_loss"]
    gates = [
        _gate(
            "fires_total",
            int(overall["fires"]) >= int(gates_plan["fires_total_min"]),
            overall["fires"],
            f">= {gates_plan['fires_total_min']}",
        ),
        _gate(
            "fires_each_profile",
            all(
                count >= int(gates_plan["fires_each_profile_min"])
                for count in profile_fires.values()
            ),
            profile_fires,
            f"each >= {gates_plan['fires_each_profile_min']}",
        ),
        _gate(
            "mean_delta_per_state",
            float(overall["mean_delta_per_state"])
            > float(gates_plan["mean_delta_per_state_strictly_greater_than"]),
            overall["mean_delta_per_state"],
            f"> {gates_plan['mean_delta_per_state_strictly_greater_than']}",
        ),
        _gate(
            "mean_delta_per_fire",
            overall["mean_delta_per_fire"] is not None
            and float(overall["mean_delta_per_fire"])
            > float(gates_plan["mean_delta_per_fire_strictly_greater_than"]),
            overall["mean_delta_per_fire"],
            f"> {gates_plan['mean_delta_per_fire_strictly_greater_than']}",
        ),
        _gate(
            "false_positive_rate_per_fire",
            overall["false_positive_rate_per_fire"] is not None
            and float(overall["false_positive_rate_per_fire"])
            <= float(gates_plan["false_positive_rate_per_fire_max"]),
            overall["false_positive_rate_per_fire"],
            f"<= {gates_plan['false_positive_rate_per_fire_max']}",
        ),
        _gate(
            "maximum_per_fired_root_p95_loss",
            tails["p95"] is not None
            and float(tails["p95"]) <= float(gates_plan["override_loss_p95_max"]),
            tails["p95"],
            f"<= {gates_plan['override_loss_p95_max']}",
        ),
        _gate(
            "maximum_per_fired_root_p99_loss",
            tails["p99"] is not None
            and float(tails["p99"]) <= float(gates_plan["override_loss_p99_max"]),
            tails["p99"],
            f"<= {gates_plan['override_loss_p99_max']}",
        ),
        _gate(
            "maximum_per_fired_root_max_loss",
            tails["max"] is not None
            and float(tails["max"]) <= float(gates_plan["override_loss_max"]),
            tails["max"],
            f"<= {gates_plan['override_loss_max']}",
        ),
        _gate("action_mapping_violation_count", True, 0, "= 0"),
        _gate("rng_domain_violation_count", True, 0, "= 0"),
        _gate("hidden_information_violation_count", True, 0, "= 0"),
        _gate(
            "nonfire_exact_baseline_action_fallback",
            True,
            True,
            "required for every nonfire",
        ),
    ]
    arm_specs = {
        str(spec["name"]): _mapping(spec, "finite arm")
        for spec in _sequence(plan.get("finite_arms"), "finite_arms")
    }
    cost = _strict_int(
        arm_specs[arm_name].get("total_action_futures_per_root"),
        f"{arm_name}.cost",
    )
    return {
        "arm_name": arm_name,
        "eligible": all(bool(gate["passed"]) for gate in gates),
        "total_action_futures_per_root": cost,
        "metrics": {
            "overall": overall,
            "by_profile": by_profile,
            "fired_root_diagnostics": [
                {
                    "root_index": int(row["root_index"]),
                    "profile": str(row["profile"]),
                    "selected_action_key": str(row["selected_action_key"]),
                    "a128_paired_delta_mean": float(row["assessment_mean"]),
                    "a128_per_root_loss": dict(row["assessment_losses"]),
                    "false_positive": float(row["assessment_mean"]) <= 0.0,
                }
                for row in rows
                if row["fired"]
            ],
        },
        "gates": gates,
    }


def aggregate_attempt07_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
) -> dict[str, Any]:
    """Validate and aggregate exactly 100 Attempt07 development rows."""

    validate_attempt07_plan(plan)
    if source_plan_sha256 != M43_ATTEMPT07_PLAN_SHA256:
        raise ValueError("Attempt07 source plan SHA-256 changed")
    if not _is_sha256(source_input_sha256):
        raise ValueError("Attempt07 source input SHA-256 is invalid")
    if len(rows) != _ROOT_COUNT:
        raise ValueError("Attempt07 development requires exactly 100 rows")
    by_index: dict[int, Mapping[str, Any]] = {}
    for raw_row in rows:
        row = _mapping(raw_row, "Attempt07 row")
        index = _strict_int(row.get("root_index"), "root_index")
        if index in by_index:
            raise ValueError(f"Attempt07 duplicate root_index: {index}")
        by_index[index] = row
    if set(by_index) != set(range(_ROOT_COUNT)):
        raise ValueError("Attempt07 root_index set must be exactly 0..99")

    schedules = enumerate_attempt07_seed_schedules(plan, population="development")
    validated: list[dict[str, Any]] = []
    for root_index in range(_ROOT_COUNT):
        expected_seeds = {
            domain: values[root_index] for domain, values in schedules.items()
        }
        validated.append(
            _validate_row(
                by_index[root_index],
                root_index=root_index,
                expected_seeds=expected_seeds,
            )
        )
    fingerprints = [str(row["fingerprint"]) for row in validated]
    hand_seeds = [int(row["hand_seed"]) for row in validated]
    if len(set(fingerprints)) != _ROOT_COUNT:
        raise ValueError("Attempt07 development fingerprints are not unique")
    if len(set(hand_seeds)) != _ROOT_COUNT:
        raise ValueError("Attempt07 development hand seeds are not unique")
    profiles = Counter(str(row["profile"]) for row in validated)
    expected_profiles = Counter({profile: 20 for profile in M43_ATTEMPT07_PROFILES})
    if profiles != expected_profiles:
        raise ValueError("Attempt07 development population is not 20/profile")
    all_rng = [digest for row in validated for digest in row["rng_keys"]]
    if len(all_rng) != len(set(all_rng)):
        raise ValueError("Attempt07 particle RNG keys are not globally unique")
    all_belief = [digest for row in validated for digest in row["belief_digests"]]
    if len(all_belief) != len(set(all_belief)):
        raise ValueError("Attempt07 belief digests are not globally unique")

    arm_rows: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in M43_ATTEMPT07_ARMS
    }
    for row in validated:
        for arm in M43_ATTEMPT07_ARMS:
            arm_rows[arm].append(
                {
                    **dict(row["arms"][arm]),
                    "profile": row["profile"],
                }
            )
    reports = {
        arm: _arm_report(arm_name=arm, rows=arm_rows[arm], plan=plan)
        for arm in M43_ATTEMPT07_ARMS
    }
    eligible = [reports[arm] for arm in M43_ATTEMPT07_ARMS if reports[arm]["eligible"]]

    def winner_key(report: Mapping[str, Any]) -> tuple[Any, ...]:
        overall = _mapping(
            _mapping(report.get("metrics"), "metrics").get("overall"), "overall"
        )
        tails = _mapping(
            overall.get("maximum_per_fired_root_loss"), "maximum losses"
        )
        return (
            -float(overall["mean_delta_per_state"]),
            float(overall["false_positive_rate_per_fire"]),
            float(tails["p95"]),
            float(tails["p99"]),
            float(tails["max"]),
            int(report["total_action_futures_per_root"]),
            str(report["arm_name"]),
        )

    winner = min(eligible, key=winner_key) if eligible else None
    winner_name = str(winner["arm_name"]) if winner is not None else None
    root_identities = [
        {
            "root_index": row["root_index"],
            "profile": row["profile"],
            "hand_seed": row["hand_seed"],
            "observation_fingerprint": row["fingerprint"],
            "baseline_action_key": row["baseline_action_key"],
            "config_sha256": row["config_sha256"],
        }
        for row in validated
    ]
    return {
        "schema": ATTEMPT07_DEVELOPMENT_SELECTION_SCHEMA,
        "status": _GO_STATUS if winner_name is not None else _NO_GO_STATUS,
        "decision": "go" if winner_name is not None else "no_go",
        "decision_scope": (
            "write_separate_immutable_winner_freeze_before_future_audit_authorization"
            if winner_name is not None
            else "close_attempt07_development_without_fit_threshold_or_audit"
        ),
        "selected_arm": winner_name,
        "winner": {
            "arm_name": winner_name,
            "selection_key": list(winner_key(winner)),
        }
        if winner is not None
        else None,
        "source": {
            "input_jsonl_sha256": source_input_sha256,
            "plan_sha256": source_plan_sha256,
            "selector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "root_identity_sha256": _canonical_sha256(root_identities),
        },
        "development_population": {
            "roots": _ROOT_COUNT,
            "profiles": list(M43_ATTEMPT07_PROFILES),
            "profile_counts": dict(profiles),
            "unique_observation_fingerprints": len(set(fingerprints)),
            "unique_hand_seeds": len(set(hand_seeds)),
        },
        "arms": reports,
        "selection_contract": {
            "eligible_arms_only": True,
            "lexicographic_order": list(
                _mapping(plan.get("winner_selection"), "winner_selection")[
                    "lexicographic_order"
                ]
            ),
            "teacher_arm_key_mapping": dict(_ARM_KEY_MAP),
            "threshold_reselection_performed": False,
        },
        "integrity": {
            "action_mapping_violation_count": 0,
            "rng_domain_violation_count": 0,
            "hidden_information_violation_count": 0,
            "nonfire_exact_baseline_action_fallback_verified": True,
            "nonfire_complete_trajectory_cancellation_verified": False,
            "nonfire_complete_trajectory_acceptance_deferred": True,
        },
        "science_boundary": {
            "assessment_source": "disjoint_A128_raw_selected_action_vs_baseline_only",
            "veto_values_used_as_development_quality_metrics": False,
            "teacher_values_are_realized_match_ev": False,
            "fit_performed": False,
            "threshold_selected": False,
            "gate_reselected": False,
            "future_audit_authorized": False,
            "future_audit_opened": False,
            "runtime_trajectory_cancellation_claimed": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
    }


def select_attempt07_development_arm(
    *, input_path: str | Path, plan_path: str | Path
) -> dict[str, Any]:
    """Load, hash, validate, and aggregate a development100 JSONL file."""

    plan_bytes = Path(plan_path).read_bytes()
    plan = _read_plan_bytes(plan_bytes)
    input_bytes = Path(input_path).read_bytes()
    rows = _read_jsonl_bytes(input_bytes)
    return aggregate_attempt07_development_rows(
        rows,
        plan=plan,
        source_input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        source_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
    )


def write_attempt07_development_selection(
    path: str | Path, report: Mapping[str, Any]
) -> None:
    """Create one immutable selector report without replacing prior output."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite Attempt07 development selection: {destination}"
        )
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite Attempt07 development selection: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one frozen Attempt07 development arm without fit or thresholding"
    )
    parser.add_argument("--input", required=True, type=Path, help="development100 JSONL")
    parser.add_argument("--plan", required=True, type=Path, help="frozen Attempt07 plan")
    parser.add_argument("--output", required=True, type=Path, help="new report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = select_attempt07_development_arm(
        input_path=args.input, plan_path=args.plan
    )
    write_attempt07_development_selection(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "selected_arm": report["selected_arm"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
