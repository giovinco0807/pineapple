"""One-shot, no-fit search-quality audit for M4.3 Attempt06.

The local audit-output consumption marker is deliberately opened and
validated before the merge receipt, merged teacher rows, or plan.  This
module only aggregates the already locked c8 choice against its independent
e128 paired evaluation.  It never fits a model, chooses a threshold, changes
``current``, or authorizes fresh-200 work.
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

from .action_key import ActionKey, action_key
from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_contract import (
    M43_ATTEMPT06_LAMBDA_SHA256,
    M43_ATTEMPT06_PLAN_SHA256,
    M43_ATTEMPT06_PROFILES,
    M43_ATTEMPT06_SEED_STRIDE,
    validate_attempt06_plan,
)
from .hu_m43_attempt06_spot import (
    AUDIT_OUTPUT_CONSUMPTION_SCHEMA,
    EXPECTED_SHARDS,
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
    RECEIVE_MERGE_SCHEMA,
    ROOT_REMATERIALIZATION_MODE,
)
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_BASELINE_PROFILE,
    ATTEMPT06_CANDIDATE_SAMPLES,
    ATTEMPT06_CANDIDATE_SEED_START,
    ATTEMPT06_CHILD_SEED_START,
    ATTEMPT06_EVALUATION_SAMPLES,
    ATTEMPT06_EVALUATION_SEED_START,
    ATTEMPT06_FROZEN_MODEL_ID,
    ATTEMPT06_FROZEN_MODEL_SHA256,
    ATTEMPT06_HAND_SEED_START,
    ATTEMPT06_ROOT_GENERATION_POLICY,
    ATTEMPT06_ROOT_PROVENANCE_SCHEMA,
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_SOLVER_ID,
    ATTEMPT06_T2_POLICY_ID,
    ATTEMPT06_TEACHER_SCHEMA,
    ATTEMPT06_TOP_K,
    _validate_root_provenance,
)
from .hu_m4_t1_teacher import M4_PAIRED_DELTA_SUMMARY_SCHEMA
from .hu_m4_teacher_contract import T1_SECOND_LIVE_SCHEDULE


ATTEMPT06_SEARCH_QUALITY_AUDIT_SCHEMA = (
    "hu_m43_attempt06_search_quality_audit_v1"
)
_MERGE_STATUS = "merged_fifty_fresh_rows_without_fit_or_threshold_selection"
_GO_STATUS = "go_create_separate_freeze_only"
_NO_GO_STATUS = "no_go_close_attempt06"
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
_ROW_KEYS = frozenset(
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
        "action_key_schema",
        "frozen_candidate_generator",
        "legal_action_count",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "legal_action_keys",
        "legal_actions",
        "learned_top8_action_keys",
        "learned_top8_order_digest",
        "candidate_action_count",
        "candidate_action_set_digest",
        "candidate_action_order_digest",
        "evaluation_action_count",
        "evaluation_action_keys",
        "evaluation_action_order_digest",
        "evaluation_scope",
        "baseline_action_key",
        "baseline_original_legal_index",
        "selected_action_key",
        "selected_action_original_legal_index",
        "selected_action_candidate_position",
        "override_fired",
        "selection_score_gap",
        "selected_action_evaluation_mean",
        "selected_action_evaluation_standard_error",
        "selected_action_paired_evaluation_delta_vs_baseline",
        "selected_action_paired_evaluation_deltas_vs_baseline",
        "selected_action_paired_evaluation_deltas_sha256",
        "selected_action_paired_evaluation_override_loss",
        "selected_action_state_mean_loss_diagnostic",
        "audit_aggregation_contract",
        "evaluation_sample_best_action_key",
        "candidate_belief_digest",
        "evaluation_belief_digest",
        "candidate_rng_key_digests",
        "evaluation_rng_key_digests",
        "sample_independence",
        "root_selection_lock",
        "search_config",
        "continuation_policy",
        "live_schedule",
        "actions",
        "child_information_set_count",
        "teacher_value_status",
        "runtime_gate_allowed",
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
_CANDIDATE_ACTION_ROW_KEYS = frozenset(
    {
        "candidate_position",
        "original_legal_index",
        "learned_nonbaseline_rank",
        "action_key",
        "placements",
        "discards",
        "model_rank_score",
        "model_rank_disagreement",
        "candidate_mean",
        "candidate_standard_error",
        "evaluation_mean",
        "evaluation_standard_error",
        "paired_evaluation_delta_vs_baseline",
        "paired_evaluation_override_loss",
        "paired_evaluation_state_mean_loss_diagnostic",
        "evaluation_scope",
        "is_explicit_baseline",
        "selected_by_c8",
        "evaluation_sample_best",
    }
)
_FROZEN_GENERATOR_KEYS = frozenset(
    {
        "family",
        "model_id",
        "artifact_sha256",
        "purpose",
        "runtime_authorized",
        "profile_runtime_feature",
    }
)
_SEARCH_CONFIG_KEYS = frozenset(
    {
        "learned_nonbaseline_top_k",
        "baseline_added_exactly_once",
        "candidate_samples",
        "evaluation_samples",
        "evaluation_action_scope",
        "candidate_seed",
        "evaluation_seed",
        "child_policy_seed",
        "run_id",
        "batch_child_selectors",
        "candidate_tie_break",
        "search_tie_break",
    }
)
_PAIR_SUMMARY_KEYS = frozenset({"schema", "count", *_PAIR_FIELDS})
_LOSS_KEYS = frozenset({"p95", "p99", "max"})


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
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


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _close(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _ordered_key_digest(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()


def _set_key_digest(tokens: Sequence[str]) -> str:
    keys = [ActionKey.from_token(token) for token in tokens]
    if len(set(keys)) != len(keys):
        raise ValueError("ActionKey set contains duplicates")
    ordered = [key.to_token() for key in sorted(keys, key=ActionKey.sort_key)]
    return _ordered_key_digest(ordered)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_mapping_bytes(data: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(data.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _validate_consumption_marker_bytes(data: bytes) -> dict[str, Any]:
    marker = _read_mapping_bytes(
        data, "Attempt06 audit-output consumption marker"
    )
    if (
        marker.get("schema") != AUDIT_OUTPUT_CONSUMPTION_SCHEMA
        or marker.get("status")
        != "consumed_before_any_result_teacher_or_root_read"
        or marker.get("current_profile_mutated") is not False
        or marker.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt06 audit-output consumption marker changed")
    return marker


def _read_rows_bytes(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        data.decode("utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Attempt06 merged row {line_number} must be a mapping"
            )
        rows.append(payload)
    if len(rows) != EXPECTED_SHARDS:
        raise ValueError("Attempt06 merged teacher must contain exactly fifty rows")
    return rows


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    marker: Mapping[str, Any],
    marker_sha256: str,
) -> None:
    expected_profile_counts = {profile: 10 for profile in M43_ATTEMPT06_PROFILES}
    if (
        receipt.get("schema") != RECEIVE_MERGE_SCHEMA
        or receipt.get("status") != _MERGE_STATUS
        or receipt.get("roots") != EXPECTED_SHARDS
        or receipt.get("shards") != EXPECTED_SHARDS
        or receipt.get("roots_per_shard") != 1
        or receipt.get("profile_counts") != expected_profile_counts
        or receipt.get("consumption_marker_file_sha256") != marker_sha256
        or receipt.get("consumption_marker_run_name") != marker.get("run_name")
    ):
        raise ValueError("Attempt06 merge receipt identity changed")
    if not isinstance(marker.get("run_name"), str) or not marker["run_name"]:
        raise ValueError("Attempt06 consumption marker lacks a run_name")
    if not _is_sha256(receipt.get("merged_teacher_sha256")):
        raise ValueError("Attempt06 merge receipt lacks merged teacher SHA-256")
    shard_hashes = _sequence(
        receipt.get("shard_audit_sha256"), "receipt.shard_audit_sha256"
    )
    if len(shard_hashes) != EXPECTED_SHARDS or not all(
        _is_sha256(value) for value in shard_hashes
    ):
        raise ValueError("Attempt06 merge receipt lacks fifty shard audit hashes")
    for key in (
        "teacher_values_are_realized_match_ev",
        "fit_performed",
        "threshold_selected",
        "go_no_go_computed",
        "current_profile_mutated",
        "runtime_policy_activated",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"Attempt06 merge receipt changed boundary: {key}")


def _validate_row_identity(row: Mapping[str, Any], root_index: int) -> None:
    if set(row) != _ROW_KEYS:
        raise ValueError(f"Attempt06 outer row fields changed at root {root_index}")
    profile = M43_ATTEMPT06_PROFILES[root_index % len(M43_ATTEMPT06_PROFILES)]
    hand_seed = ATTEMPT06_HAND_SEED_START + M43_ATTEMPT06_SEED_STRIDE * root_index
    teacher = _mapping(row.get("teacher"), f"row[{root_index}].teacher")
    if set(teacher) != _TEACHER_KEYS:
        raise ValueError(f"Attempt06 teacher fields changed at root {root_index}")
    provenance = _mapping(
        row.get("provenance"), f"row[{root_index}].provenance"
    )
    _validate_root_provenance(
        provenance,
        root_profile=M43_ATTEMPT06_PROFILES[
            root_index % len(M43_ATTEMPT06_PROFILES)
        ],
        root_index=root_index,
        teacher_output=True,
    )
    if (
        row.get("schema") != ATTEMPT06_SHARD_ROW_SCHEMA
        or row.get("root_index") != root_index
        or row.get("hand_seed") != hand_seed
        or row.get("root_profile") != profile
        or teacher.get("schema") != ATTEMPT06_TEACHER_SCHEMA
        or teacher.get("status") != "ok"
        or teacher.get("solver_id") != ATTEMPT06_SOLVER_ID
        or teacher.get("street") != "T1"
        or teacher.get("seat") != "second"
        or teacher.get("to_act_order") != "second"
        or teacher.get("teacher_value_status") != "diagnostic_not_match_EV"
        or teacher.get("runtime_gate_allowed") is not False
        or row.get("baseline_action_key") != teacher.get("baseline_action_key")
    ):
        raise ValueError(f"Attempt06 merged root identity changed at {root_index}")
    continuation = _mapping(
        teacher.get("continuation_policy"),
        f"row[{root_index}].teacher.continuation_policy",
    )
    if continuation != {
        "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
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
        raise ValueError(
            f"Attempt06 continuation policy changed at root {root_index}"
        )
    expected_live_schedule = [
        {
            "seat": step.seat,
            "street": step.street,
            "draw_offset": step.draw_offset,
        }
        for step in T1_SECOND_LIVE_SCHEDULE
    ]
    if teacher.get("live_schedule") != expected_live_schedule:
        raise ValueError(f"Attempt06 live schedule changed at root {root_index}")
    run_name = provenance.get("run_name")
    expected_provenance = {
        "schema": ATTEMPT06_ROOT_PROVENANCE_SCHEMA,
        "root_index": root_index,
        "root_profile": profile,
        "root_policy_seed_base": ATTEMPT06_HAND_SEED_START,
        "root_generation_policy": ATTEMPT06_ROOT_GENERATION_POLICY,
        "baseline_profile": ATTEMPT06_BASELINE_PROFILE,
        "plan_sha256": M43_ATTEMPT06_PLAN_SHA256,
        "candidate_model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_value_status": "diagnostic_not_match_EV",
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError(f"Attempt06 provenance closure changed at root {root_index}")
    if (
        not isinstance(run_name, str)
        or not run_name
        or provenance.get("run_id") != f"{run_name}:shard={root_index}"
    ):
        raise ValueError(f"Attempt06 provenance run identity changed at {root_index}")
    for key in (
        "schedule_sha256",
        "package_manifest_sha256",
        "global_consumption_marker_sha256",
        "root_consumption_claim_sha256",
        "source_sha256",
        "startup_sha256",
        "status_sha256",
        "source_closure_sha256",
        "input_sha256",
        "config_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            raise ValueError(
                f"Attempt06 provenance {key} is invalid at root {root_index}"
            )
    if (
        provenance.get("fresh_audit_retry_or_alternate_sample_allowed") is not False
        or provenance.get("deterministic_claim_recovery_allowed") is not True
        or provenance.get("deterministic_claim_recovery_mode")
        != ROOT_REMATERIALIZATION_MODE
    ):
        raise ValueError(
            f"Attempt06 provenance recovery boundary changed at root {root_index}"
        )


def _validate_pair_summary(
    value: Any,
    *,
    root_index: int,
    raw_deltas: Any | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    summary = _mapping(value, f"root[{root_index}].selected paired summary")
    if (
        set(summary) != _PAIR_SUMMARY_KEYS
        or
        summary.get("schema") != M4_PAIRED_DELTA_SUMMARY_SCHEMA
        or summary.get("count") != ATTEMPT06_EVALUATION_SAMPLES
    ):
        raise ValueError(f"Attempt06 paired e128 summary changed at root {root_index}")
    values = {
        name: _finite(summary.get(name), f"root[{root_index}].paired.{name}")
        for name in _PAIR_FIELDS
    }
    if values["standard_error"] < 0.0 or values["std"] < 0.0:
        raise ValueError(f"Attempt06 paired dispersion is negative at root {root_index}")
    if not _close(
        values["standard_error"],
        values["std"] / math.sqrt(ATTEMPT06_EVALUATION_SAMPLES),
    ):
        raise ValueError(
            f"Attempt06 paired standard error is inconsistent at root {root_index}"
        )
    ordered = (
        values["min"],
        values["p01"],
        values["p05"],
        values["p25"],
        values["p50"],
        values["p75"],
        values["p95"],
        values["p99"],
        values["max"],
    )
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"Attempt06 paired quantiles are not ordered at root {root_index}")
    if not values["min"] <= values["mean"] <= values["max"]:
        raise ValueError(f"Attempt06 paired mean is outside range at root {root_index}")
    for name in ("lt0_rate", "le_neg6_rate", "le_neg12_rate", "le_neg20_rate"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"Attempt06 paired rate is invalid at root {root_index}")
    if not (
        values["le_neg20_rate"]
        <= values["le_neg12_rate"]
        <= values["le_neg6_rate"]
        <= values["lt0_rate"]
    ):
        raise ValueError(
            f"Attempt06 paired loss rates are not nested at root {root_index}"
        )
    if raw_deltas is not None:
        raw = _sequence(raw_deltas, f"root[{root_index}].paired raw deltas")
        if len(raw) != ATTEMPT06_EVALUATION_SAMPLES:
            raise ValueError(
                f"Attempt06 paired raw e128 count changed at root {root_index}"
            )
        array = np.asarray(
            [
                _finite(item, f"root[{root_index}].paired_raw[{index}]")
                for index, item in enumerate(raw)
            ],
            dtype=np.float64,
        )
        expected_std = float(np.std(array, ddof=1))
        expected = {
            "mean": float(np.mean(array)),
            "standard_error": expected_std / math.sqrt(array.size),
            "std": expected_std,
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
        for name, expected_value in expected.items():
            if not _close(values[name], expected_value):
                raise ValueError(
                    f"Attempt06 paired raw/summary {name} mismatch at "
                    f"root {root_index}"
                )
    losses = {
        "p95": max(0.0, -values["p05"]),
        "p99": max(0.0, -values["p01"]),
        "max": max(0.0, -values["min"]),
    }
    return values, losses


def _action_mapping_ok(
    row: Mapping[str, Any], teacher: Mapping[str, Any], root_index: int
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    try:
        observation = ActorObservation.from_dict(
            _mapping(row.get("policy_observation"), "policy_observation")
        )
        check(observation.to_dict() == row.get("policy_observation"), "observation")
        check(
            teacher.get("observation_fingerprint") == observation.fingerprint(),
            "observation_fingerprint",
        )
        generated = generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
        check(
            teacher.get("action_key_schema") == "regular_ofc_action_key_v1",
            "action_key_schema",
        )
        generated_tokens = [action_key(action).to_token() for action in generated]
        legal_tokens = [
            str(value)
            for value in _sequence(teacher.get("legal_action_keys"), "legal keys")
        ]
        for token in legal_tokens:
            ActionKey.from_token(token)
        check(len(set(legal_tokens)) == len(legal_tokens), "legal_unique")
        check(teacher.get("legal_action_count") == len(legal_tokens), "legal_count")
        check(legal_tokens == generated_tokens, "complete_legal_mapping")
        check(
            teacher.get("legal_action_order_digest")
            == _ordered_key_digest(legal_tokens),
            "legal_order_digest",
        )
        check(
            teacher.get("legal_action_set_digest") == _set_key_digest(legal_tokens),
            "legal_set_digest",
        )

        baseline = str(teacher.get("baseline_action_key"))
        selected = str(teacher.get("selected_action_key"))
        ActionKey.from_token(baseline)
        ActionKey.from_token(selected)
        baseline_index = int(teacher.get("baseline_original_legal_index"))
        selected_index = int(teacher.get("selected_action_original_legal_index"))
        check(0 <= baseline_index < len(legal_tokens), "baseline_index_range")
        check(0 <= selected_index < len(legal_tokens), "selected_index_range")
        if 0 <= baseline_index < len(legal_tokens):
            check(legal_tokens[baseline_index] == baseline, "baseline_index_key")
        if 0 <= selected_index < len(legal_tokens):
            check(legal_tokens[selected_index] == selected, "selected_index_key")

        top8 = [
            str(value)
            for value in _sequence(
                teacher.get("learned_top8_action_keys"), "learned top8"
            )
        ]
        check(len(top8) == ATTEMPT06_TOP_K, "top8_count")
        check(len(set(top8)) == ATTEMPT06_TOP_K, "top8_unique")
        check(baseline not in top8, "top8_excludes_baseline")
        check(set(top8).issubset(set(legal_tokens)), "top8_legal")
        check(
            teacher.get("learned_top8_order_digest") == _ordered_key_digest(top8),
            "top8_order_digest",
        )

        legal_rows = _sequence(teacher.get("legal_actions"), "legal actions")
        check(len(legal_rows) == len(legal_tokens), "legal_rows_count")
        legal_rank: list[tuple[float, ActionKey, str]] = []
        for index, (payload, token) in enumerate(
            zip(legal_rows, legal_tokens, strict=False)
        ):
            action_row = _mapping(payload, f"legal action {index}")
            check(set(action_row) == _LEGAL_ACTION_ROW_KEYS, "legal_row_fields")
            check(action_row.get("original_legal_index") == index, "legal_row_index")
            check(action_row.get("action_key") == token, "legal_row_key")
            score = _finite(action_row.get("model_rank_score"), "model rank score")
            disagreement = _finite(
                action_row.get("model_rank_disagreement"), "model rank disagreement"
            )
            check(disagreement >= 0.0, "model_rank_disagreement")
            check(
                action_row.get("is_explicit_baseline") is (token == baseline),
                "legal_baseline_flag",
            )
            check(
                action_row.get("in_learned_top8") is (token in set(top8)),
                "legal_top8_flag",
            )
            if token != baseline:
                legal_rank.append((score, ActionKey.from_token(token), token))
        expected_top8 = [
            token
            for _, _, token in sorted(
                legal_rank, key=lambda item: (-item[0], item[1].sort_key())
            )[:ATTEMPT06_TOP_K]
        ]
        check(top8 == expected_top8, "top8_rank_order")

        actions = _sequence(teacher.get("actions"), "candidate actions")
        check(teacher.get("candidate_action_count") == 9, "candidate_count_field")
        check(len(actions) == 9, "candidate_rows_count")
        candidate_tokens = [
            str(_mapping(value, "candidate action").get("action_key"))
            for value in actions
        ]
        check(candidate_tokens == [*top8, baseline], "candidate_order")
        check(len(set(candidate_tokens)) == 9, "candidate_unique")
        check(
            teacher.get("candidate_action_order_digest")
            == _ordered_key_digest(candidate_tokens),
            "candidate_order_digest",
        )
        check(
            teacher.get("candidate_action_set_digest")
            == _set_key_digest(candidate_tokens),
            "candidate_set_digest",
        )
        selection_rows: list[tuple[float, ActionKey, int]] = []
        legal_score_by_token = {
            str(_mapping(payload, "legal action").get("action_key")): (
                _finite(
                    _mapping(payload, "legal action").get("model_rank_score"),
                    "legal model rank score",
                ),
                _finite(
                    _mapping(payload, "legal action").get(
                        "model_rank_disagreement"
                    ),
                    "legal model rank disagreement",
                ),
            )
            for payload in legal_rows
        }
        for position, payload in enumerate(actions):
            action_row = _mapping(payload, f"candidate action {position}")
            check(
                set(action_row) == _CANDIDATE_ACTION_ROW_KEYS,
                "candidate_row_fields",
            )
            token = candidate_tokens[position]
            check(action_row.get("candidate_position") == position, "candidate_position")
            check(
                action_row.get("learned_nonbaseline_rank")
                == (position + 1 if position < ATTEMPT06_TOP_K else None),
                "candidate_learned_rank",
            )
            original_index = int(action_row.get("original_legal_index"))
            check(0 <= original_index < len(generated), "candidate_original_range")
            if 0 <= original_index < len(generated):
                generated_action = generated[original_index]
                check(
                    action_key(generated_action).to_token() == token,
                    "candidate_original_key",
                )
                check(
                    action_row.get("placements")
                    == [list(value) for value in generated_action.placements],
                    "candidate_placements",
                )
                check(
                    action_row.get("discards") == list(generated_action.discards),
                    "candidate_discards",
                )
            expected_model_score, expected_disagreement = legal_score_by_token[token]
            check(
                _close(action_row.get("model_rank_score"), expected_model_score),
                "candidate_model_rank_score",
            )
            check(
                _close(
                    action_row.get("model_rank_disagreement"),
                    expected_disagreement,
                ),
                "candidate_model_rank_disagreement",
            )
            standard_error = _finite(
                action_row.get("candidate_standard_error"),
                "candidate standard error",
            )
            check(standard_error >= 0.0, "candidate_standard_error")
            selection_rows.append(
                (
                    _finite(action_row.get("candidate_mean"), "candidate mean"),
                    ActionKey.from_token(token),
                    position,
                )
            )
            check(
                action_row.get("is_explicit_baseline") is (position == 8),
                "candidate_baseline_flag",
            )
        selected_position = int(teacher.get("selected_action_candidate_position"))
        selection_ranking = sorted(
            selection_rows, key=lambda item: (-item[0], item[1].sort_key())
        )
        expected_selected = selection_ranking[0][2]
        check(selected_position == expected_selected, "selected_c8_argmax")
        check(candidate_tokens[selected_position] == selected, "selected_candidate_key")
        check(
            sum(
                _mapping(value, "candidate action").get("selected_by_c8") is True
                for value in actions
            )
            == 1,
            "selected_flag_count",
        )
        check(
            _mapping(actions[selected_position], "selected row").get("selected_by_c8")
            is True,
            "selected_flag_position",
        )
        for position, payload in enumerate(actions):
            selected_flag = _mapping(payload, "candidate action").get(
                "selected_by_c8"
            )
            check(
                isinstance(selected_flag, bool)
                and selected_flag is (position == selected_position),
                "selected_flag_value",
            )
        expected_selection_gap = (
            selection_ranking[0][0] - selection_ranking[1][0]
        )
        check(
            _close(
                _finite(
                    teacher.get("selection_score_gap"),
                    "selection score gap",
                ),
                expected_selection_gap,
            ),
            "selection_score_gap",
        )
        child_count = teacher.get("child_information_set_count")
        check(
            isinstance(child_count, int)
            and not isinstance(child_count, bool)
            and child_count >= 0,
            "child_information_set_count",
        )
        evaluated_positions = list(dict.fromkeys((selected_position, 8)))
        evaluated_tokens = [candidate_tokens[position] for position in evaluated_positions]
        check(
            teacher.get("evaluation_action_count") == len(evaluated_positions),
            "evaluation_action_count",
        )
        check(
            teacher.get("evaluation_action_keys") == evaluated_tokens,
            "evaluation_action_keys",
        )
        check(
            teacher.get("evaluation_action_order_digest")
            == _ordered_key_digest(evaluated_tokens),
            "evaluation_action_order_digest",
        )
        check(
            teacher.get("evaluation_scope")
            == "c8_locked_action_plus_explicit_baseline_only",
            "evaluation_scope",
        )
        evaluation_rows: list[tuple[float, ActionKey, int]] = []
        for position, payload in enumerate(actions):
            action_row = _mapping(payload, f"candidate action {position}")
            if position not in evaluated_positions:
                for name in (
                    "evaluation_mean",
                    "evaluation_standard_error",
                    "paired_evaluation_delta_vs_baseline",
                    "paired_evaluation_override_loss",
                    "paired_evaluation_state_mean_loss_diagnostic",
                    "evaluation_sample_best",
                ):
                    check(action_row.get(name) is None, f"unevaluated_{name}_null")
                check(
                    action_row.get("evaluation_scope")
                    == "not_evaluated_after_c8_lock",
                    "unevaluated_scope",
                )
                continue
            evaluation_mean = _finite(
                action_row.get("evaluation_mean"), "evaluation mean"
            )
            evaluation_se = _finite(
                action_row.get("evaluation_standard_error"),
                "evaluation standard error",
            )
            check(evaluation_se >= 0.0, "evaluation_standard_error")
            check(
                action_row.get("evaluation_scope")
                == "locked_action_and_explicit_baseline",
                "evaluated_scope",
            )
            check(
                isinstance(action_row.get("evaluation_sample_best"), bool),
                "evaluated_best_flag_type",
            )
            evaluated_pair, evaluated_loss = _validate_pair_summary(
                action_row.get("paired_evaluation_delta_vs_baseline"),
                root_index=root_index,
            )
            recorded_loss = _mapping(
                action_row.get("paired_evaluation_override_loss"),
                "evaluated action loss",
            )
            check(set(recorded_loss) == _LOSS_KEYS, "evaluated_loss_fields")
            check(
                all(
                    _close(recorded_loss.get(name), evaluated_loss[name])
                    for name in evaluated_loss
                ),
                "evaluated_loss_binding",
            )
            check(
                _close(
                    action_row.get(
                        "paired_evaluation_state_mean_loss_diagnostic"
                    ),
                    max(0.0, -evaluated_pair["mean"]),
                ),
                "evaluated_mean_loss_binding",
            )
            if position == 8:
                check(
                    all(_close(evaluated_pair[name], 0.0) for name in _PAIR_FIELDS),
                    "baseline_pair_zero",
                )
                check(
                    all(_close(value, 0.0) for value in evaluated_loss.values()),
                    "baseline_loss_zero",
                )
            evaluation_rows.append(
                (evaluation_mean, ActionKey.from_token(candidate_tokens[position]), position)
            )
        expected_eval_best = sorted(
            evaluation_rows, key=lambda item: (-item[0], item[1].sort_key())
        )[0][2]
        check(
            teacher.get("evaluation_sample_best_action_key")
            == candidate_tokens[expected_eval_best],
            "evaluation_best_key",
        )
        check(
            sum(
                _mapping(actions[position], "candidate action").get(
                    "evaluation_sample_best"
                )
                is True
                for position in evaluated_positions
            )
            == 1,
            "evaluation_best_flag_count",
        )
        check(
            _mapping(actions[expected_eval_best], "evaluation best").get(
                "evaluation_sample_best"
            )
            is True,
            "evaluation_best_flag_position",
        )
        check(teacher.get("override_fired") is (selected != baseline), "fire_flag")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        failures.append(f"exception:{type(exc).__name__}")
    return not failures, sorted(set(failures))


def _rng_domain_ok(
    row: Mapping[str, Any], teacher: Mapping[str, Any], root_index: int
) -> tuple[bool, list[str], list[str], list[str]]:
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    candidate_keys: list[str] = []
    evaluation_keys: list[str] = []
    try:
        provenance = _mapping(row.get("provenance"), "provenance")
        candidate_keys = [
            str(value)
            for value in _sequence(
                teacher.get("candidate_rng_key_digests"), "candidate RNG keys"
            )
        ]
        evaluation_keys = [
            str(value)
            for value in _sequence(
                teacher.get("evaluation_rng_key_digests"), "evaluation RNG keys"
            )
        ]
        check(len(candidate_keys) == ATTEMPT06_CANDIDATE_SAMPLES, "candidate_count")
        check(len(evaluation_keys) == ATTEMPT06_EVALUATION_SAMPLES, "evaluation_count")
        check(all(_is_sha256(value) for value in candidate_keys), "candidate_digest")
        check(all(_is_sha256(value) for value in evaluation_keys), "evaluation_digest")
        check(len(set(candidate_keys)) == len(candidate_keys), "candidate_unique")
        check(len(set(evaluation_keys)) == len(evaluation_keys), "evaluation_unique")
        check(set(candidate_keys).isdisjoint(evaluation_keys), "candidate_eval_disjoint")
        check(_is_sha256(teacher.get("candidate_belief_digest")), "candidate_belief")
        check(_is_sha256(teacher.get("evaluation_belief_digest")), "evaluation_belief")
        check(
            teacher.get("candidate_belief_digest")
            != teacher.get("evaluation_belief_digest"),
            "belief_disjoint",
        )
        check(
            teacher.get("sample_independence") == "disjoint_particle_rng_keys",
            "sample_independence",
        )
        check(
            teacher.get("root_selection_lock")
            == "top8_fixed_before_sampling_then_c8_locked_before_e128",
            "root_selection_lock",
        )
        config = _mapping(teacher.get("search_config"), "search_config")
        check(set(config) == _SEARCH_CONFIG_KEYS, "config_fields")
        fingerprint = str(teacher.get("observation_fingerprint"))
        expected_run_id = (
            f"{provenance.get('run_id')}:root={root_index}:"
            f"seed={row.get('hand_seed')}:obs={fingerprint}"
        )
        expected = {
            "learned_nonbaseline_top_k": ATTEMPT06_TOP_K,
            "baseline_added_exactly_once": True,
            "candidate_samples": ATTEMPT06_CANDIDATE_SAMPLES,
            "evaluation_samples": ATTEMPT06_EVALUATION_SAMPLES,
            "evaluation_action_scope": (
                "locked_action_plus_explicit_baseline_only"
            ),
            "candidate_seed": ATTEMPT06_CANDIDATE_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            "evaluation_seed": ATTEMPT06_EVALUATION_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            "child_policy_seed": ATTEMPT06_CHILD_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            "run_id": expected_run_id,
            "batch_child_selectors": True,
            "candidate_tie_break": "ActionKey",
            "search_tie_break": "ActionKey",
        }
        for key, value in expected.items():
            check(config.get(key) == value, f"config_{key}")
        observation = ActorObservation.from_dict(
            _mapping(row.get("policy_observation"), "policy_observation")
        )
        expected_candidate_batch = sample_hidden_card_particles(
            observation,
            base_seed=ATTEMPT06_CANDIDATE_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            run_id=f"{expected_run_id}:candidate_selection",
            sample_count=ATTEMPT06_CANDIDATE_SAMPLES,
        )
        expected_evaluation_batch = sample_hidden_card_particles(
            observation,
            base_seed=ATTEMPT06_EVALUATION_SEED_START
            + M43_ATTEMPT06_SEED_STRIDE * root_index,
            run_id=f"{expected_run_id}:locked_evaluation",
            sample_count=ATTEMPT06_EVALUATION_SAMPLES,
        )
        expected_candidate_batch.validate_against(observation)
        expected_evaluation_batch.validate_against(observation)
        expected_candidate_keys = [
            particle.rng_key_digest
            for particle in expected_candidate_batch.particles
        ]
        expected_evaluation_keys = [
            particle.rng_key_digest
            for particle in expected_evaluation_batch.particles
        ]
        check(candidate_keys == expected_candidate_keys, "candidate_key_derivation")
        check(
            evaluation_keys == expected_evaluation_keys,
            "evaluation_key_derivation",
        )
        check(
            teacher.get("candidate_belief_digest")
            == expected_candidate_batch.digest(),
            "candidate_belief_derivation",
        )
        check(
            teacher.get("evaluation_belief_digest")
            == expected_evaluation_batch.digest(),
            "evaluation_belief_derivation",
        )
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"exception:{type(exc).__name__}")
    return not failures, sorted(set(failures)), candidate_keys, evaluation_keys


def _hidden_information_ok(
    row: Mapping[str, Any], teacher: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.lower()
                child_path = f"{path}.{key}"
                if normalized in _FORBIDDEN_HIDDEN_KEYS:
                    failures.append(child_path)
                if "lcb" in normalized or "lower_confidence_bound" in normalized:
                    failures.append(child_path)
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    try:
        observation = ActorObservation.from_dict(
            _mapping(row.get("policy_observation"), "policy_observation")
        )
        if observation.seat != "second" or observation.to_act_order != "second":
            failures.append("policy_observation.action_order")
        if observation.street != "T1":
            failures.append("policy_observation.street")
        provenance = _mapping(row.get("provenance"), "provenance")
        if provenance.get("opponent_private_discard_input_allowed") is not False:
            failures.append("provenance.opponent_private_discard_input_allowed")
        frozen = _mapping(
            teacher.get("frozen_candidate_generator"), "frozen candidate generator"
        )
        if (
            set(frozen) != _FROZEN_GENERATOR_KEYS
            or frozen.get("family") != "lambda_rank"
            or frozen.get("model_id") != ATTEMPT06_FROZEN_MODEL_ID
            or frozen.get("artifact_sha256") != M43_ATTEMPT06_LAMBDA_SHA256
            or frozen.get("purpose") != "candidate_generation_only"
            or frozen.get("runtime_authorized") is not False
            or frozen.get("profile_runtime_feature") is not False
        ):
            failures.append("teacher.frozen_candidate_generator")
        if teacher.get("runtime_gate_allowed") is not False:
            failures.append("teacher.runtime_gate_allowed")
        visit(row, "row")
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"exception:{type(exc).__name__}")
    return not failures, sorted(set(failures))


def _validate_selected_binding(
    teacher: Mapping[str, Any],
    *,
    root_index: int,
    paired: Mapping[str, float],
    losses: Mapping[str, float],
) -> None:
    selected_position = teacher.get("selected_action_candidate_position")
    actions = _sequence(teacher.get("actions"), "teacher.actions")
    if (
        isinstance(selected_position, bool)
        or not isinstance(selected_position, int)
        or not 0 <= selected_position < len(actions)
    ):
        raise ValueError(f"Attempt06 selected position is invalid at root {root_index}")
    selected_row = _mapping(actions[selected_position], "selected action row")
    nested_pair = _mapping(
        selected_row.get("paired_evaluation_delta_vs_baseline"),
        "selected nested pair",
    )
    nested_loss = _mapping(
        selected_row.get("paired_evaluation_override_loss"),
        "selected nested loss",
    )
    top_loss = _mapping(
        teacher.get("selected_action_paired_evaluation_override_loss"),
        "selected top loss",
    )
    top_pair = _mapping(
        teacher.get("selected_action_paired_evaluation_delta_vs_baseline"),
        "selected top pair",
    )
    if set(nested_loss) != _LOSS_KEYS or set(top_loss) != _LOSS_KEYS:
        raise ValueError(
            f"Attempt06 selected loss fields changed at root {root_index}"
        )
    baseline_row = _mapping(actions[8], "baseline action row")
    selected_evaluation_mean = _finite(
        selected_row.get("evaluation_mean"), "selected evaluation mean"
    )
    baseline_evaluation_mean = _finite(
        baseline_row.get("evaluation_mean"), "baseline evaluation mean"
    )
    raw_deltas = list(
        _sequence(
            teacher.get(
                "selected_action_paired_evaluation_deltas_vs_baseline"
            ),
            "selected raw paired deltas",
        )
    )
    raw_digest = hashlib.sha256(
        json.dumps(
            raw_deltas,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if (
        selected_row.get("action_key") != teacher.get("selected_action_key")
        or top_pair != nested_pair
        or top_loss != nested_loss
        or not all(_close(top_pair.get(name), paired[name]) for name in _PAIR_FIELDS)
        or not all(_close(top_loss.get(name), losses[name]) for name in losses)
        or not _close(
            teacher.get("selected_action_state_mean_loss_diagnostic"),
            max(0.0, -paired["mean"]),
        )
        or not _close(
            selected_row.get("paired_evaluation_state_mean_loss_diagnostic"),
            max(0.0, -paired["mean"]),
        )
        or not _close(
            teacher.get("selected_action_evaluation_mean"),
            selected_row.get("evaluation_mean"),
        )
        or not _close(
            teacher.get("selected_action_evaluation_standard_error"),
            selected_row.get("evaluation_standard_error"),
        )
        or not _close(
            selected_evaluation_mean - baseline_evaluation_mean,
            paired["mean"],
        )
        or teacher.get("selected_action_paired_evaluation_deltas_sha256")
        != raw_digest
    ):
        raise ValueError(
            f"Attempt06 selected action/e128 binding changed at root {root_index}"
        )
    contract = _mapping(
        teacher.get("audit_aggregation_contract"), "audit aggregation contract"
    )
    if contract != {
        "fire_definition": "selected_action_key_differs_from_explicit_baseline",
        "per_root_tail_source": "selected_action_paired_evaluation_override_loss",
        "aggregate_over_fires": {
            "p95": "maximum_of_per_root_e128_p95",
            "p99": "maximum_of_per_root_e128_p99",
            "max": "maximum_of_per_root_e128_max",
        },
        "quantile_method": "numpy_linear",
        "state_mean_loss_is_non_gate_diagnostic": True,
    }:
        raise ValueError(
            f"Attempt06 aggregation contract changed at root {root_index}"
        )


def _profile_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    states = len(rows)
    fired = [row for row in rows if row["fired"]]
    deltas = [float(row["mean"]) for row in rows]
    fired_deltas = [float(row["mean"]) for row in fired]
    false_positives = sum(value <= 0.0 for value in fired_deltas)
    return {
        "states": states,
        "fires": len(fired),
        "mean_delta_per_state": sum(deltas) / states if states else None,
        "mean_delta_per_fire": (
            sum(fired_deltas) / len(fired_deltas) if fired_deltas else None
        ),
        "false_positive_fires": false_positives,
        "false_positive_rate_per_fire": (
            false_positives / len(fired_deltas) if fired_deltas else None
        ),
        "metric_b_maximum_per_fired_root_loss": {
            name: max(float(row["losses"][name]) for row in fired)
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


def validate_attempt06_teacher_row(
    row: Mapping[str, Any],
    *,
    root_index: int,
    expected_run_name: str | None = None,
    expected_schedule_sha256: str | None = None,
    expected_package_manifest_sha256: str | None = None,
    expected_global_consumption_marker_sha256: str | None = None,
    expected_root_consumption_claim_sha256: str | None = None,
    expected_input_sha256: str | None = None,
    expected_source_sha256: str | None = None,
    expected_startup_sha256: str | None = None,
    expected_status_sha256: str | None = None,
    expected_source_closure_sha256: str | None = None,
) -> None:
    """Strictly validate one received row for the Spot receive boundary.

    The one-shot aggregator records action/RNG/hidden mismatches as No-Go gate
    violations.  The earlier receive boundary cannot safely retain such a row,
    so this public entry point promotes every mismatch to a fail-closed error.
    """

    _validate_row_identity(row, root_index)
    teacher = _mapping(row.get("teacher"), f"root[{root_index}].teacher")
    provenance = _mapping(
        row.get("provenance"), f"root[{root_index}].provenance"
    )
    expected = {
        "run_name": expected_run_name,
        "schedule_sha256": expected_schedule_sha256,
        "package_manifest_sha256": expected_package_manifest_sha256,
        "global_consumption_marker_sha256": (
            expected_global_consumption_marker_sha256
        ),
        "root_consumption_claim_sha256": expected_root_consumption_claim_sha256,
        "input_sha256": expected_input_sha256,
        "source_sha256": expected_source_sha256,
        "startup_sha256": expected_startup_sha256,
        "status_sha256": expected_status_sha256,
        "source_closure_sha256": expected_source_closure_sha256,
    }
    for key, value in expected.items():
        if value is not None and provenance.get(key) != value:
            raise ValueError(
                f"Attempt06 received teacher provenance {key} mismatch at "
                f"root {root_index}"
            )

    baseline = teacher.get("baseline_action_key")
    selected = teacher.get("selected_action_key")
    if not isinstance(baseline, str) or not isinstance(selected, str):
        raise ValueError(
            f"Attempt06 baseline/selected ActionKey missing at root {root_index}"
        )
    ActionKey.from_token(baseline)
    ActionKey.from_token(selected)
    fired = selected != baseline
    if teacher.get("override_fired") is not fired:
        raise ValueError(f"Attempt06 override fire flag changed at root {root_index}")
    paired, losses = _validate_pair_summary(
        teacher.get("selected_action_paired_evaluation_delta_vs_baseline"),
        root_index=root_index,
        raw_deltas=teacher.get(
            "selected_action_paired_evaluation_deltas_vs_baseline"
        ),
    )
    _validate_selected_binding(
        teacher,
        root_index=root_index,
        paired=paired,
        losses=losses,
    )
    if not fired and (
        any(not _close(paired[name], 0.0) for name in _PAIR_FIELDS)
        or any(not _close(value, 0.0) for value in losses.values())
    ):
        raise ValueError(
            f"Attempt06 non-fire lacks exact counterfactual cancellation at "
            f"root {root_index}"
        )
    action_ok, action_reasons = _action_mapping_ok(row, teacher, root_index)
    rng_ok, rng_reasons, _candidate_rng, _evaluation_rng = _rng_domain_ok(
        row, teacher, root_index
    )
    hidden_ok, hidden_reasons = _hidden_information_ok(row, teacher)
    failures = {
        "action_mapping": action_reasons if not action_ok else [],
        "rng_domain": rng_reasons if not rng_ok else [],
        "hidden_information": hidden_reasons if not hidden_ok else [],
    }
    if any(failures.values()):
        raise ValueError(
            "Attempt06 received teacher row failed strict integrity: "
            + json.dumps(failures, sort_keys=True, separators=(",", ":"))
        )


def audit_attempt06_search_quality(
    *,
    merged_path: str | Path,
    merge_receipt_path: str | Path,
    consumption_marker_path: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    """Validate and aggregate the one-shot merged50 result without writes.

    Keep the marker load as the first operation.  In particular, do not even
    hash or test existence of the result paths before this call returns.
    """

    marker_bytes = Path(consumption_marker_path).read_bytes()
    marker = _validate_consumption_marker_bytes(marker_bytes)

    merged_file = Path(merged_path)
    receipt_file = Path(merge_receipt_path)
    plan_file = Path(plan_path)
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()

    plan_bytes = plan_file.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != M43_ATTEMPT06_PLAN_SHA256:
        raise ValueError("Attempt06 authoritative plan SHA-256 changed")
    plan = _read_mapping_bytes(plan_bytes, "Attempt06 authoritative plan")
    validate_attempt06_plan(plan)
    gates_plan = _mapping(
        plan.get("search_quality_go_no_go"), "search quality plan"
    )
    receipt_bytes = receipt_file.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    receipt = _read_mapping_bytes(receipt_bytes, "Attempt06 merge receipt")
    _validate_receipt(receipt, marker=marker, marker_sha256=marker_sha256)
    merged_bytes = merged_file.read_bytes()
    merged_sha256 = hashlib.sha256(merged_bytes).hexdigest()
    if receipt.get("merged_teacher_sha256") != merged_sha256:
        raise ValueError("Attempt06 merged teacher SHA-256 differs from receipt")

    rows = _read_rows_bytes(merged_bytes)
    if [row.get("root_index") for row in rows] != list(range(EXPECTED_SHARDS)):
        raise ValueError("Attempt06 merged roots are not ordered exactly 0..49")

    metric_rows: list[dict[str, Any]] = []
    action_violations: list[dict[str, Any]] = []
    rng_violations: list[dict[str, Any]] = []
    hidden_violations: list[dict[str, Any]] = []
    all_candidate_rng: list[str] = []
    all_evaluation_rng: list[str] = []
    provenance_domains: dict[str, list[str]] = {
        "schedule_sha256": [],
        "package_manifest_sha256": [],
        "global_consumption_marker_sha256": [],
    }
    identities: list[dict[str, Any]] = []
    for root_index, row in enumerate(rows):
        _validate_row_identity(row, root_index)
        teacher = _mapping(row["teacher"], f"root[{root_index}].teacher")
        baseline = teacher.get("baseline_action_key")
        selected = teacher.get("selected_action_key")
        if not isinstance(baseline, str) or not isinstance(selected, str):
            raise ValueError(
                f"Attempt06 baseline/selected ActionKey missing at root {root_index}"
            )
        ActionKey.from_token(baseline)
        ActionKey.from_token(selected)
        fired = selected != baseline
        if teacher.get("override_fired") is not fired:
            raise ValueError(f"Attempt06 override fire flag changed at root {root_index}")

        paired, losses = _validate_pair_summary(
            teacher.get("selected_action_paired_evaluation_delta_vs_baseline"),
            root_index=root_index,
            raw_deltas=teacher.get(
                "selected_action_paired_evaluation_deltas_vs_baseline"
            ),
        )
        _validate_selected_binding(
            teacher,
            root_index=root_index,
            paired=paired,
            losses=losses,
        )
        if not fired and (
            any(not _close(paired[name], 0.0) for name in _PAIR_FIELDS)
            or any(not _close(value, 0.0) for value in losses.values())
        ):
            raise ValueError(
                f"Attempt06 non-fire lacks exact counterfactual cancellation at {root_index}"
            )

        action_ok, action_reasons = _action_mapping_ok(row, teacher, root_index)
        if not action_ok:
            action_violations.append(
                {"root_index": root_index, "reasons": action_reasons}
            )
        rng_ok, rng_reasons, candidate_rng, evaluation_rng = _rng_domain_ok(
            row, teacher, root_index
        )
        if not rng_ok:
            rng_violations.append(
                {"root_index": root_index, "reasons": rng_reasons}
            )
        all_candidate_rng.extend(candidate_rng)
        all_evaluation_rng.extend(evaluation_rng)
        hidden_ok, hidden_reasons = _hidden_information_ok(row, teacher)
        if not hidden_ok:
            hidden_violations.append(
                {"root_index": root_index, "reasons": hidden_reasons}
            )
        provenance = _mapping(row["provenance"], "provenance")
        if provenance.get("run_name") != marker.get("run_name"):
            raise ValueError(
                f"Attempt06 root run differs from local consumption marker at "
                f"root {root_index}"
            )
        for key in provenance_domains:
            provenance_domains[key].append(str(provenance[key]))
        profile = str(row["root_profile"])
        metric_rows.append(
            {
                "root_index": root_index,
                "profile": profile,
                "fired": fired,
                "mean": paired["mean"],
                "losses": losses,
            }
        )
        identities.append(
            {
                "root_index": root_index,
                "profile": profile,
                "observation_fingerprint": teacher.get("observation_fingerprint"),
                "baseline_action_key": baseline,
                "selected_action_key": selected,
            }
        )

    cross_rng_reasons: list[str] = []
    if len(all_candidate_rng) != len(set(all_candidate_rng)):
        cross_rng_reasons.append("candidate_rng_not_globally_unique")
    if len(all_evaluation_rng) != len(set(all_evaluation_rng)):
        cross_rng_reasons.append("evaluation_rng_not_globally_unique")
    if not set(all_candidate_rng).isdisjoint(all_evaluation_rng):
        cross_rng_reasons.append("candidate_evaluation_rng_overlap_across_roots")
    for key, values in provenance_domains.items():
        if len(set(values)) != 1:
            cross_rng_reasons.append(f"{key}_not_constant_across_roots")
    if cross_rng_reasons:
        rng_violations.append(
            {"root_index": None, "reasons": sorted(cross_rng_reasons)}
        )

    profile_counts = Counter(str(row["profile"]) for row in metric_rows)
    if profile_counts != Counter({profile: 10 for profile in M43_ATTEMPT06_PROFILES}):
        raise ValueError("Attempt06 profile balance changed after merge")
    overall = _profile_metrics(metric_rows)
    profiles = {
        profile: _profile_metrics(
            [row for row in metric_rows if row["profile"] == profile]
        )
        for profile in M43_ATTEMPT06_PROFILES
    }
    fires = [row for row in metric_rows if row["fired"]]
    mean_losses = np.asarray(
        [max(0.0, -float(row["mean"])) for row in fires], dtype=np.float64
    )
    diagnostic = {
        "classification": "diagnostic_only_not_gate",
        "source": "max(0,-paired_e128_mean)_per_fired_root",
        "quantile_method": "numpy_linear",
        "p95": float(np.quantile(mean_losses, 0.95)) if fires else None,
        "p99": float(np.quantile(mean_losses, 0.99)) if fires else None,
        "max": float(np.max(mean_losses)) if fires else None,
    }

    profile_fire_counts = {
        profile: int(profiles[profile]["fires"])
        for profile in M43_ATTEMPT06_PROFILES
    }
    b_tail = overall["metric_b_maximum_per_fired_root_loss"]
    gates = [
        _gate(
            "fires_total",
            overall["fires"] >= int(gates_plan["fires_total_min"]),
            overall["fires"],
            f">= {gates_plan['fires_total_min']}",
        ),
        _gate(
            "fires_each_profile",
            all(
                count >= int(gates_plan["fires_each_profile_min"])
                for count in profile_fire_counts.values()
            ),
            profile_fire_counts,
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
            "metric_b_p95_loss",
            b_tail["p95"] is not None
            and float(b_tail["p95"])
            <= float(gates_plan["override_loss_p95_max"]),
            b_tail["p95"],
            f"maximum fired-root max(0,-p05) <= {gates_plan['override_loss_p95_max']}",
        ),
        _gate(
            "metric_b_p99_loss",
            b_tail["p99"] is not None
            and float(b_tail["p99"])
            <= float(gates_plan["override_loss_p99_max"]),
            b_tail["p99"],
            f"maximum fired-root max(0,-p01) <= {gates_plan['override_loss_p99_max']}",
        ),
        _gate(
            "metric_b_max_loss",
            b_tail["max"] is not None
            and float(b_tail["max"])
            <= float(gates_plan["override_loss_max"]),
            b_tail["max"],
            f"maximum fired-root max(0,-min) <= {gates_plan['override_loss_max']}",
        ),
        _gate(
            "action_mapping_violation_count",
            len(action_violations)
            <= int(gates_plan["action_mapping_violation_count_max"]),
            len(action_violations),
            f"<= {gates_plan['action_mapping_violation_count_max']}",
        ),
        _gate(
            "rng_domain_violation_count",
            len(rng_violations)
            <= int(gates_plan["rng_domain_violation_count_max"]),
            len(rng_violations),
            f"<= {gates_plan['rng_domain_violation_count_max']}",
        ),
        _gate(
            "hidden_information_violation_count",
            len(hidden_violations)
            <= int(gates_plan["hidden_information_violation_count_max"]),
            len(hidden_violations),
            f"<= {gates_plan['hidden_information_violation_count_max']}",
        ),
    ]
    all_pass = all(bool(gate["passed"]) for gate in gates)
    return {
        "schema": ATTEMPT06_SEARCH_QUALITY_AUDIT_SCHEMA,
        "status": _GO_STATUS if all_pass else _NO_GO_STATUS,
        "decision": "go" if all_pass else "no_go",
        "decision_scope": (
            "create_separate_freeze_before_any_fresh_200_root_fit_or_distillation"
            if all_pass
            else "close_attempt06_without_fit_threshold_retry_or_spot_expansion"
        ),
        "source": {
            "plan_sha256": M43_ATTEMPT06_PLAN_SHA256,
            "merge_receipt_sha256": receipt_sha256,
            "merged_teacher_sha256": merged_sha256,
            "consumption_marker_sha256": marker_sha256,
            "consumption_marker_run_name": marker["run_name"],
            "shard_audit_sha256": list(receipt["shard_audit_sha256"]),
            "root_identity_sha256": _canonical_sha256(identities),
        },
        "audit_population": {
            "states": EXPECTED_SHARDS,
            "profiles": list(M43_ATTEMPT06_PROFILES),
            "profile_counts": dict(profile_counts),
            "paired_evaluation_futures_per_root": ATTEMPT06_EVALUATION_SAMPLES,
            "candidate_selection_futures_per_root": ATTEMPT06_CANDIDATE_SAMPLES,
        },
        "metrics": {
            "overall": overall,
            "by_profile": profiles,
            "fired_root_diagnostics": [
                {
                    "root_index": row["root_index"],
                    "profile": row["profile"],
                    "independent_e128_paired_delta_mean": row["mean"],
                    "metric_b_per_root_loss": row["losses"],
                    "false_positive": float(row["mean"]) <= 0.0,
                }
                for row in fires
            ],
            "paired_mean_cross_fire_loss": diagnostic,
            "tail_gate_semantics": {
                "variant": "B_per_fired_root_e128_tail_then_max_across_fires",
                "p95_loss": "max_over_fires(max(0,-root_p05))",
                "p99_loss": "max_over_fires(max(0,-root_p01))",
                "max_loss": "max_over_fires(max(0,-root_min))",
            },
        },
        "integrity": {
            "action_mapping_violation_count": len(action_violations),
            "rng_domain_violation_count": len(rng_violations),
            "hidden_information_violation_count": len(hidden_violations),
            "action_mapping_violations": action_violations,
            "rng_domain_violations": rng_violations,
            "hidden_information_violations": hidden_violations,
            "nonfire_counterfactual_cancellation_verified": True,
        },
        "gates": gates,
        "all_gates_passed": all_pass,
        "science_boundary": {
            "teacher_values_are_realized_match_ev": False,
            "fit_performed": False,
            "model_changed": False,
            "threshold_selected": False,
            "gate_reselected": False,
            "retry_same_seeds_allowed": False,
            "fresh_200_root_fit_or_distillation_authorized": False,
            "acceptance_holdout_opened": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement_enabled": False,
        },
    }


def write_attempt06_search_quality_report(
    path: str | Path, report: Mapping[str, Any]
) -> None:
    """Create the immutable audit report once; never replace an old report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Attempt06 report: {destination}")
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Attempt06 report was concurrently created: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--merge-receipt", type=Path, required=True)
    parser.add_argument("--consumption-marker", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_attempt06_search_quality(
        merged_path=args.merged,
        merge_receipt_path=args.merge_receipt,
        consumption_marker_path=args.consumption_marker,
        plan_path=args.plan,
    )
    write_attempt06_search_quality_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT06_SEARCH_QUALITY_AUDIT_SCHEMA",
    "audit_attempt06_search_quality",
    "main",
    "validate_attempt06_teacher_row",
    "write_attempt06_search_quality_report",
]
