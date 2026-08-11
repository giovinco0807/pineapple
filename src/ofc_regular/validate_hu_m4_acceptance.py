"""Fail-closed acceptance decision for the M4 T1 second-seat policy.

This module consumes only completed JSON artifacts.  It never runs a policy,
relabels a holdout, changes a threshold, resolves ``current``, or interprets a
teacher score as played-hand EV.  A statistically underpowered evaluation is
a completed ``No-Go`` decision, not a validator failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_pilot_contract import (
    M43_DATA_CONTRACT_SCHEMA,
    M43_FREEZE_SCHEMA,
    M43_LOCKED_RECEIPT_SCHEMA,
    canonical_manifest_sha256,
    validate_locked_holdout_receipt,
    validate_model_threshold_freeze,
)
from .evaluate_hu_m4_population import summarize_hu_m4_population_records
from .merge_hu_m4_population_shards import (
    M4_POPULATION_MERGE_SCHEMA,
    REQUIRED_ACTIVATION_GUARDS,
)


M4_ACCEPTANCE_CONFIG_SCHEMA = "hu_m4_t1_acceptance_config_v1"
M4_ACCEPTANCE_STATUS_SCHEMA = "hu_m4_t1_acceptance_status_v1"
M4_POPULATION_EVALUATION_SCHEMA = "hu_m4_t1_population_evaluation_v1"
M4_DATA_AUDIT_SCHEMA = "hu_m4_t1_second_data_audit_v1"
M4_TRAINING_MANIFEST_SCHEMA = "hu_m4_t1_joint_training_manifest_v2"
M4_LEGACY_TRAINING_MANIFEST_SCHEMA = "hu_m4_t1_joint_training_manifest_v1"
M43_ACTION_SCORE_MODE = "baseline_paired_delta_risk_ensemble_v3"
M43_CROSS_FIT_SCHEMA = "hu_m4_identity_group_paired_delta_cross_fit_v3"
M43_CONSUMPTION_MARKER_SCHEMA = "hu_m43_locked_holdout_consumption_marker_v1"
M43_POPULATION_PLAN_SCHEMA = "hu_m43_population_acceptance_plan_v1"
M43_POPULATION_SPOT_MANIFEST_SCHEMA = "hu_m4_population_spot_manifest_v1"
M43_POPULATION_SPOT_RECEIPT_SCHEMA = "hu_m4_population_spot_receipt_v1"

REQUIRED_OPPONENTS = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "random_exact_final",
)

FIXED_GATES: dict[str, Any] = {
    "invalid_counterfactuals_max": 0,
    "nonfire_cancellation_mismatches_max": 0,
    "nonfire_nonzero_deltas_max": 0,
    "nonfire_cancellation_unknown_max": 0,
    "valid_overrides_min": 300,
    "realized_gain_per_override_ci95_low_min_exclusive": 0.0,
    "paired_delta_ev_per_hand_ci95_low_min_exclusive": 0.0,
    "second_seat_delta_ev_per_hand_ci95_low_min_exclusive": 0.0,
    "first_seat_delta_exact": 0.0,
    "false_positive_override_rate_max": 0.30,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "each_opponent_delta_ev_per_hand_min": -0.005,
    "each_opponent_delta_ev_per_hand_ci95_low_min": -0.02,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(payload)
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _path(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _same_number(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1e-12)
    )


def _hash_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        return None
    return normalized


def _manifest_runtime_hashes(manifest: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Read an explicit artifact lock without guessing from a model path."""

    containers = [
        manifest,
        _mapping(manifest.get("runtime_lock")),
        _mapping(manifest.get("runtime_artifact")),
        _mapping(manifest.get("model_artifact")),
        _mapping(_mapping(manifest.get("artifacts")).get("runtime")),
    ]
    candidate_keys = (
        "candidate_model_sha256",
        "action_value_model_sha256",
        "model_sha256",
        "sha256",
    )
    safety_keys = ("safety_model_sha256", "model_sha256", "sha256")
    for container in containers:
        candidate = next(
            (_hash_string(container.get(key)) for key in candidate_keys if _hash_string(container.get(key))),
            None,
        )
        safety = next(
            (_hash_string(container.get(key)) for key in safety_keys if _hash_string(container.get(key))),
            None,
        )
        if candidate is not None or safety is not None:
            return candidate, safety
    return None, None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _population_summary_for_comparison(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(summary)
    normalized.pop("runtime_config", None)
    normalized.pop("shard_merge", None)
    normalized.pop("records", None)
    normalized["elapsed_seconds"] = None
    normalized["records_output"] = None
    return normalized


def _is_m43_manifest(training_manifest: Mapping[str, Any]) -> bool:
    return (
        _path(training_manifest, "action_score_formula", "mode")
        == M43_ACTION_SCORE_MODE
        or _path(training_manifest, "training_config", "action_score_mode")
        == M43_ACTION_SCORE_MODE
    )


def _m43_lifecycle_artifact_chain(
    *,
    evaluation: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    data_contract: Mapping[str, Any] | None,
    freeze_manifest: Mapping[str, Any] | None,
    locked_receipt: Mapping[str, Any] | None,
    consumption_marker: Mapping[str, Any] | None,
    population_plan: Mapping[str, Any] | None,
    population_records: Sequence[Mapping[str, Any]] | None,
    population_merge_manifest: Mapping[str, Any] | None,
    population_spot_receipt: Mapping[str, Any] | None,
    population_run_manifest: Mapping[str, Any] | None,
    source_hashes: Mapping[str, str],
) -> tuple[bool, dict[str, Any]]:
    """Validate the M4.3 pre-freeze -> one-shot -> population hash chain.

    The lifecycle validators intentionally use canonical content digests for
    the contract/freeze relationship.  This final acceptance layer adds the
    byte-level input hashes, the training-manifest file binding stored in the
    freeze, the one-shot marker bytes stored in the receipt, and the runtime
    model/threshold binding carried by the fresh population artifact.
    """

    observed: dict[str, Any] = {
        "required_artifacts_present": all(
            isinstance(value, Mapping)
            for value in (
                data_contract,
                freeze_manifest,
                locked_receipt,
                consumption_marker,
                population_plan,
                population_merge_manifest,
                population_spot_receipt,
                population_run_manifest,
            )
        )
        and isinstance(population_records, Sequence)
        and not isinstance(population_records, (str, bytes)),
        "source_sha256": {
            name: source_hashes.get(name)
            for name in (
                "evaluation",
                "data_audit",
                "training_manifest",
                "m43_data_contract",
                "m43_freeze_manifest",
                "m43_locked_receipt",
                "m43_consumption_marker",
                "m43_population_plan",
                "m43_population_records",
                "m43_population_merge_manifest",
                "m43_population_spot_receipt",
                "m43_population_run_manifest",
            )
        },
    }
    if not observed["required_artifacts_present"]:
        observed["error"] = "missing M4.3 lifecycle artifact"
        return False, observed
    assert data_contract is not None
    assert freeze_manifest is not None
    assert locked_receipt is not None
    assert consumption_marker is not None
    assert population_plan is not None
    assert population_records is not None
    assert population_merge_manifest is not None
    assert population_spot_receipt is not None
    assert population_run_manifest is not None

    valid_source_hashes = all(
        _hash_string(source_hashes.get(name)) is not None
        for name in observed["source_sha256"]
    )
    observed["all_input_file_hashes_present"] = valid_source_hashes
    try:
        validate_model_threshold_freeze(
            freeze_manifest, data_contract=data_contract
        )
        validate_locked_holdout_receipt(
            locked_receipt,
            freeze_manifest=freeze_manifest,
            data_contract=data_contract,
        )
    except (KeyError, TypeError, ValueError) as exc:
        observed["error"] = f"{type(exc).__name__}: {exc}"
        return False, observed

    runtime = _mapping(evaluation.get("runtime_config"))
    training_source_sha = _hash_string(source_hashes.get("training_manifest"))
    marker_source_sha = _hash_string(source_hashes.get("m43_consumption_marker"))
    evaluation_source_sha = _hash_string(source_hashes.get("evaluation"))
    population_plan_source_sha = _hash_string(
        source_hashes.get("m43_population_plan")
    )
    records_source_sha = _hash_string(
        source_hashes.get("m43_population_records")
    )
    merge_source_sha = _hash_string(
        source_hashes.get("m43_population_merge_manifest")
    )
    spot_receipt_source_sha = _hash_string(
        source_hashes.get("m43_population_spot_receipt")
    )
    run_manifest_source_sha = _hash_string(
        source_hashes.get("m43_population_run_manifest")
    )
    freeze_canonical_sha = canonical_manifest_sha256(freeze_manifest)
    base_audit_sha = _canonical_sha256(data_audit)
    model_sha = _hash_string(freeze_manifest.get("model_sha256"))
    runtime_candidate_sha = _hash_string(runtime.get("candidate_model_sha256"))
    runtime_safety_sha = _hash_string(runtime.get("safety_model_sha256"))
    frozen_threshold = _number(freeze_manifest.get("frozen_threshold"))
    runtime_threshold = _number(runtime.get("safety_threshold"))

    marker_checks = {
        "schema": consumption_marker.get("schema")
        == M43_CONSUMPTION_MARKER_SCHEMA,
        "status": consumption_marker.get("status")
        == "claimed_before_locked_content_read",
        "freeze": consumption_marker.get("freeze_manifest_sha256")
        == freeze_canonical_sha,
        "contract": consumption_marker.get("data_contract_sha256")
        == data_contract.get("contract_sha256"),
        "model": consumption_marker.get("model_sha256") == model_sha,
        "locked_identity": consumption_marker.get("locked_identity_sha256")
        == _path(data_contract, "splits", "locked_holdout", "identity_sha256"),
        "locked_shards": consumption_marker.get(
            "locked_teacher_shards_sha256"
        )
        == _path(
            data_contract,
            "teacher_shards",
            "splits",
            "locked_holdout",
            "ordered_shards_sha256",
        ),
        "pass_count": _integer(consumption_marker.get("evaluation_pass_count"))
        == 1,
        "receipt_byte_hash": marker_source_sha
        == _hash_string(locked_receipt.get("consumption_marker_sha256")),
    }
    binding = _mapping(training_manifest.get("m43_data_contract"))
    contract_checks = {
        "schema": data_contract.get("schema") == M43_DATA_CONTRACT_SCHEMA,
        "base_audit_content": base_audit_sha
        == data_contract.get("base_audit_sha256"),
        "training_contract": binding.get("contract_sha256")
        == data_contract.get("contract_sha256"),
        "training_plan": binding.get("plan_sha256")
        == data_contract.get("plan_sha256"),
        "training_base_audit": binding.get("base_audit_sha256")
        == data_contract.get("base_audit_sha256"),
        "training_prior_freshness": binding.get("prior_freshness_audit_sha256")
        == _path(data_contract, "prior_exclusions", "audit_sha256"),
        "training_teacher_shards": binding.get(
            "teacher_shards_all_splits_sha256"
        )
        == data_contract.get("teacher_shards_all_splits_sha256"),
    }
    chain_checks = {
        "training_action_score_mode": _path(
            training_manifest, "action_score_formula", "mode"
        )
        == M43_ACTION_SCORE_MODE
        and _path(training_manifest, "training_config", "action_score_mode")
        == M43_ACTION_SCORE_MODE,
        "population_action_score_mode": runtime.get("action_score_mode")
        == M43_ACTION_SCORE_MODE,
        "positive_bounded_pilot_authorization": training_manifest.get(
            "promotion_status"
        )
        == "candidate_for_realized_ev_evaluation"
        and _path(training_manifest, "calibration", "status") == "go"
        and (_integer(_path(training_manifest, "calibration", "selected_metrics", "fires")) or 0)
        >= 10
        and (_integer(_path(training_manifest, "calibration", "constraints", "minimum_fires")) or 0)
        >= 10
        and freeze_manifest.get("calibration_status") == "go"
        and freeze_manifest.get("safety_enabled") is True,
        "training_manifest_file_bound_by_freeze": training_source_sha
        == _hash_string(freeze_manifest.get("training_manifest_sha256")),
        "freeze_canonical_bound_by_receipt": locked_receipt.get(
            "freeze_manifest_sha256"
        )
        == freeze_canonical_sha,
        "population_model_bound_to_freeze": model_sha is not None
        and runtime_candidate_sha == model_sha
        and runtime_safety_sha == model_sha,
        "population_threshold_bound_to_freeze": frozen_threshold is not None
        and runtime_threshold is not None
        and _same_number(frozen_threshold, runtime_threshold),
        "population_artifact_file_hashed": evaluation_source_sha is not None,
        "single_locked_pass": _integer(
            locked_receipt.get("evaluation_pass_count")
        )
        == 1,
        "fresh_population_still_required": locked_receipt.get(
            "requires_fresh_population_acceptance"
        )
        is True,
        "minimum_override_gate_not_lowered": _integer(
            locked_receipt.get("minimum_population_valid_overrides")
        )
        is not None
        and int(locked_receipt["minimum_population_valid_overrides"]) >= 300,
    }
    plan_guards = _mapping(population_plan.get("activation_guards"))
    planned_seed = _integer(population_plan.get("seed")) or 0
    planned_stride = _integer(population_plan.get("seed_stride")) or 0
    planned_count = _integer(
        population_plan.get("paired_seeds_per_opponent")
    ) or 0
    planned_seeds = {
        planned_seed + index * planned_stride for index in range(planned_count)
    }
    excluded_schedules = _sequence(
        _path(population_plan, "freshness", "excluded_population_schedules")
    )
    excluded_schedule_valid = bool(excluded_schedules)
    excluded_seed_overlap: set[int] = set()
    for raw_schedule in excluded_schedules:
        schedule = _mapping(raw_schedule)
        excluded_seed = _integer(schedule.get("seed")) or 0
        excluded_stride = _integer(schedule.get("seed_stride")) or 0
        excluded_count = _integer(schedule.get("paired_seeds")) or 0
        if excluded_seed <= 0 or excluded_stride <= 0 or excluded_count <= 0:
            excluded_schedule_valid = False
            continue
        excluded_seed_overlap.update(
            planned_seeds
            & {
                excluded_seed + index * excluded_stride
                for index in range(excluded_count)
            }
        )
    expected_shards = _integer(population_plan.get("shards")) or 0
    paired_seeds_per_shard = _integer(
        population_plan.get("paired_seeds_per_shard")
    ) or 0
    plan_checks = {
        "schema": population_plan.get("schema") == M43_POPULATION_PLAN_SCHEMA,
        "status": population_plan.get("status")
        == "frozen_before_population_evaluation",
        "opponents": population_plan.get("opponents") == list(REQUIRED_OPPONENTS),
        "paired_seat_swap": population_plan.get("paired_seat_swap") is True,
        "seed": _integer(population_plan.get("seed"))
        == _integer(evaluation.get("seed")),
        "seed_stride": _integer(population_plan.get("seed_stride"))
        == _integer(evaluation.get("seed_stride")),
        "paired_seeds": _integer(
            population_plan.get("paired_seeds_per_opponent")
        )
        == _integer(evaluation.get("paired_seeds_per_opponent")),
        "minimum_override_gate": (
            _integer(population_plan.get("minimum_valid_overrides")) or 0
        )
        >= 300,
        "optional_stopping_forbidden": _path(
            population_plan,
            "sizing_rule",
            "optional_stopping_or_posthoc_extension_allowed",
        )
        is False,
        "activation_guards": set(plan_guards) == REQUIRED_ACTIVATION_GUARDS
        and all(plan_guards[key] is False for key in REQUIRED_ACTIVATION_GUARDS),
        "complete_shard_grid": expected_shards > 0
        and paired_seeds_per_shard > 0
        and expected_shards * paired_seeds_per_shard == planned_count,
        "prior_population_seed_disjoint": excluded_schedule_valid
        and not excluded_seed_overlap,
        "runtime_plan_byte_hash": population_plan_source_sha is not None
        and runtime.get("population_plan_sha256") == population_plan_source_sha,
    }

    try:
        recomputed = summarize_hu_m4_population_records(
            population_records,
            opponents=REQUIRED_OPPONENTS,
            paired_seeds=planned_count,
            seed=planned_seed,
            seed_stride=planned_stride,
            elapsed_seconds=None,
        )
        records_content_valid = True
        recompute_error = None
    except (KeyError, TypeError, ValueError) as exc:
        recomputed = {}
        records_content_valid = False
        recompute_error = f"{type(exc).__name__}: {exc}"
    evaluation_matches_records = records_content_valid and (
        _canonical_sha256(_population_summary_for_comparison(evaluation))
        == _canonical_sha256(_population_summary_for_comparison(recomputed))
    )

    merge_shards = _sequence(population_merge_manifest.get("shards"))
    evaluation_merge = _mapping(evaluation.get("shard_merge"))
    merge_shards_by_index: dict[int, Mapping[str, Any]] = {}
    merge_shard_provenance_valid = len(merge_shards) == expected_shards
    for raw_shard in merge_shards:
        shard = _mapping(raw_shard)
        offset = _integer(shard.get("offset"))
        shard_index = (
            offset // paired_seeds_per_shard
            if offset is not None and paired_seeds_per_shard > 0
            else -1
        )
        shard_valid = (
            offset is not None
            and offset == shard_index * paired_seeds_per_shard
            and 0 <= shard_index < expected_shards
            and shard_index not in merge_shards_by_index
            and _integer(shard.get("seed"))
            == planned_seed + offset * planned_stride
            and _integer(shard.get("paired_seeds"))
            == paired_seeds_per_shard
            and _integer(shard.get("records"))
            == paired_seeds_per_shard * 2 * len(REQUIRED_OPPONENTS)
            and _hash_string(shard.get("evaluation_sha256")) is not None
            and _hash_string(shard.get("records_sha256")) is not None
        )
        merge_shard_provenance_valid = (
            merge_shard_provenance_valid and shard_valid
        )
        if shard_valid:
            merge_shards_by_index[shard_index] = shard
    merge_checks = {
        "schema": population_merge_manifest.get("schema")
        == M4_POPULATION_MERGE_SCHEMA,
        "status": population_merge_manifest.get("status")
        == "complete_content_verified",
        "plan": population_merge_manifest.get("population_plan_sha256")
        == population_plan_source_sha,
        "schedule": _integer(population_merge_manifest.get("seed"))
        == planned_seed
        and _integer(population_merge_manifest.get("seed_stride"))
        == planned_stride
        and _integer(
            population_merge_manifest.get("paired_seeds_per_opponent")
        )
        == planned_count,
        "opponents": population_merge_manifest.get("opponents")
        == list(REQUIRED_OPPONENTS),
        "records_count": _integer(
            population_merge_manifest.get("merged_records")
        )
        == len(population_records),
        "records_byte_hash": population_merge_manifest.get(
            "merged_records_sha256"
        )
        == records_source_sha,
        "evaluation_byte_hash": population_merge_manifest.get(
            "evaluation_sha256"
        )
        == evaluation_source_sha,
        "shards": merge_shard_provenance_valid
        and set(merge_shards_by_index) == set(range(expected_shards)),
        "current_profile": population_merge_manifest.get(
            "current_profile_used"
        )
        is False,
        "cluster_recompute": population_merge_manifest.get(
            "metrics_recomputed_from_merged_seed_clusters"
        )
        is True,
        "evaluation_binding": evaluation_merge.get("schema")
        == M4_POPULATION_MERGE_SCHEMA
        and evaluation_merge.get("population_plan_sha256")
        == population_plan_source_sha
        and _integer(evaluation_merge.get("shards")) == expected_shards
        and evaluation_merge.get(
            "metrics_recomputed_from_merged_seed_clusters"
        )
        is True,
    }

    run_population = _mapping(population_run_manifest.get("population_plan"))
    run_runtime = _mapping(population_run_manifest.get("runtime"))
    run_shards = _mapping(population_run_manifest.get("shards"))
    run_source = _mapping(population_run_manifest.get("source"))
    run_startup = _mapping(population_run_manifest.get("startup"))
    run_compute = _mapping(population_run_manifest.get("compute"))
    run_checkpoint = _mapping(population_run_manifest.get("checkpoint"))
    launch_preflight = _mapping(
        population_run_manifest.get("launch_preflight")
    )
    run_checks = {
        "schema": population_run_manifest.get("schema")
        == M43_POPULATION_SPOT_MANIFEST_SCHEMA,
        "no_activation": population_run_manifest.get("no_runtime_activation")
        is True
        and population_run_manifest.get("current_profile_mutated") is False,
        "plan_hash": run_population.get("sha256")
        == population_plan_source_sha,
        "plan_schedule": _integer(run_population.get("paired_seeds"))
        == planned_count
        and _integer(run_population.get("seed")) == planned_seed
        and _integer(run_population.get("seed_stride")) == planned_stride
        and _integer(run_population.get("shards")) == expected_shards
        and _integer(run_population.get("paired_seeds_per_shard"))
        == paired_seeds_per_shard,
        "shard_manifest": _integer(run_shards.get("count")) == expected_shards
        and _hash_string(run_shards.get("sha256")) is not None,
        "immutable_source": _hash_string(run_source.get("sha256")) is not None
        and (_integer(run_source.get("bytes")) or 0) > 0
        and _hash_string(run_startup.get("sha256")) is not None,
        "spot_execution": run_compute.get("provisioning_model") == "SPOT",
        "checkpoint_contract": run_checkpoint.get("unit")
        == "completed_shard"
        and run_checkpoint.get("retry") == "deterministic_full_shard"
        and run_checkpoint.get("resume_missing_shards_only") is True
        and run_checkpoint.get("done_commit_last") is True,
        "model": run_runtime.get("model_sha256") == model_sha,
        "threshold": _same_number(
            run_runtime.get("frozen_threshold"), frozen_threshold
        ),
        "training": run_runtime.get("training_manifest_sha256")
        == training_source_sha,
        "contract": run_runtime.get("data_contract_sha256")
        == data_contract.get("contract_sha256"),
        "freeze": run_runtime.get("freeze_manifest_sha256")
        == _hash_string(source_hashes.get("m43_freeze_manifest")),
        "locked_receipt": run_runtime.get("locked_receipt_sha256")
        == _hash_string(source_hashes.get("m43_locked_receipt")),
        "consumption_marker": run_runtime.get("consumption_marker_sha256")
        == marker_source_sha,
        "current_profile": run_runtime.get("current_profile_used") is False,
        "launch_preflight": launch_preflight.get("schema")
        == "hu_m43_population_launch_preflight_v1"
        and launch_preflight.get("status") == "pass"
        and _integer(launch_preflight.get("teacher_overlap_count")) == 0
        and _integer(
            launch_preflight.get("prior_population_overlap_count")
        )
        == 0,
    }

    spot_shards = _sequence(population_spot_receipt.get("shards"))
    spot_shard_ids = {
        _integer(_mapping(row).get("shard")) for row in spot_shards
    }
    spot_shard_provenance_valid = len(spot_shards) == expected_shards
    for raw_shard in spot_shards:
        shard = _mapping(raw_shard)
        shard_index = _integer(shard.get("shard"))
        merged_shard = merge_shards_by_index.get(
            shard_index if shard_index is not None else -1, {}
        )
        spot_shard_provenance_valid = spot_shard_provenance_valid and (
            shard_index is not None
            and 0 <= shard_index < expected_shards
            and _hash_string(shard.get("done_sha256")) is not None
            and shard.get("evaluation_sha256")
            == merged_shard.get("evaluation_sha256")
            and shard.get("records_sha256")
            == merged_shard.get("records_sha256")
        )
    spot_checks = {
        "schema": population_spot_receipt.get("schema")
        == M43_POPULATION_SPOT_RECEIPT_SCHEMA,
        "status": population_spot_receipt.get("status")
        == "verified_and_merged",
        "run_identity": population_spot_receipt.get("run_name")
        == population_run_manifest.get("run_name"),
        "run_manifest": population_spot_receipt.get("run_manifest_sha256")
        == run_manifest_source_sha,
        "plan": population_spot_receipt.get("population_plan_sha256")
        == population_plan_source_sha,
        "model": population_spot_receipt.get("model_sha256") == model_sha,
        "evaluation": population_spot_receipt.get("evaluation_sha256")
        == evaluation_source_sha,
        "records": population_spot_receipt.get("records_sha256")
        == records_source_sha,
        "merge_manifest": population_spot_receipt.get(
            "merge_manifest_sha256"
        )
        == merge_source_sha,
        "schedule": _integer(
            population_spot_receipt.get("paired_seeds_per_opponent")
        )
        == planned_count,
        "metrics": _integer(population_spot_receipt.get("valid_overrides"))
        == _integer(_path(evaluation, "population", "all_seats", "overrides"))
        and _integer(
            population_spot_receipt.get("invalid_counterfactuals")
        )
        == _integer(evaluation.get("invalid_counterfactuals"))
        and _integer(
            population_spot_receipt.get("nonfire_cancellation_mismatches")
        )
        == _integer(evaluation.get("nonfire_cancellation_mismatches")),
        "shards": spot_shard_provenance_valid
        and spot_shard_ids == set(range(expected_shards)),
        "no_activation": population_spot_receipt.get(
            "current_profile_mutated"
        )
        is False
        and population_spot_receipt.get("no_runtime_activation") is True,
        "receipt_file_hashed": spot_receipt_source_sha is not None,
    }
    population_content_checks = {
        "records_content_valid": records_content_valid,
        "evaluation_recomputed_from_records": evaluation_matches_records,
        "merge_manifest": all(merge_checks.values()),
        "run_manifest": all(run_checks.values()),
        "spot_receipt": all(spot_checks.values()),
    }
    observed.update(
        {
            "freeze_schema": freeze_manifest.get("schema"),
            "receipt_schema": locked_receipt.get("schema"),
            "freeze_canonical_sha256": freeze_canonical_sha,
            "base_audit_canonical_sha256": base_audit_sha,
            "marker_checks": marker_checks,
            "contract_checks": contract_checks,
            "chain_checks": chain_checks,
            "plan_checks": plan_checks,
            "population_content_checks": population_content_checks,
            "population_recompute_error": recompute_error,
            "merge_checks": merge_checks,
            "run_checks": run_checks,
            "spot_checks": spot_checks,
        }
    )
    return (
        valid_source_hashes
        and freeze_manifest.get("schema") == M43_FREEZE_SCHEMA
        and locked_receipt.get("schema") == M43_LOCKED_RECEIPT_SCHEMA
        and all(marker_checks.values())
        and all(contract_checks.values())
        and all(chain_checks.values())
        and all(plan_checks.values())
        and all(population_content_checks.values()),
        observed,
    )


def _gate(
    name: str,
    passed: bool,
    *,
    observed: Any,
    requirement: str,
    category: str = "promotion",
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
        "category": category,
    }


def validate_hu_m4_acceptance(
    evaluation: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    *,
    m43_data_contract: Mapping[str, Any] | None = None,
    m43_freeze_manifest: Mapping[str, Any] | None = None,
    m43_locked_receipt: Mapping[str, Any] | None = None,
    m43_consumption_marker: Mapping[str, Any] | None = None,
    m43_population_plan: Mapping[str, Any] | None = None,
    m43_population_records: Sequence[Mapping[str, Any]] | None = None,
    m43_population_merge_manifest: Mapping[str, Any] | None = None,
    m43_population_spot_receipt: Mapping[str, Any] | None = None,
    m43_population_run_manifest: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return immutable gate configuration and a completed Go/No-Go status."""

    source_hashes = dict(source_hashes or {})
    gates: list[dict[str, Any]] = []
    is_m43 = (
        _is_m43_manifest(training_manifest)
        or _path(evaluation, "runtime_config", "action_score_mode")
        == M43_ACTION_SCORE_MODE
    )
    lifecycle_artifacts_present = any(
        value is not None
        for value in (
            m43_data_contract,
            m43_freeze_manifest,
            m43_locked_receipt,
            m43_consumption_marker,
            m43_population_plan,
            m43_population_records,
            m43_population_merge_manifest,
            m43_population_spot_receipt,
            m43_population_run_manifest,
        )
    )

    gates.extend(
        [
            _gate(
                "evaluation_schema",
                evaluation.get("schema") == M4_POPULATION_EVALUATION_SCHEMA,
                observed=evaluation.get("schema"),
                requirement=f"== {M4_POPULATION_EVALUATION_SCHEMA}",
                category="provenance",
            ),
            _gate(
                "data_audit_schema_and_status",
                data_audit.get("schema") == M4_DATA_AUDIT_SCHEMA
                and data_audit.get("status") == "pass",
                observed={"schema": data_audit.get("schema"), "status": data_audit.get("status")},
                requirement=f"schema == {M4_DATA_AUDIT_SCHEMA} and status == pass",
                category="provenance",
            ),
            _gate(
                "training_manifest_schema",
                training_manifest.get("schema")
                in {
                    M4_TRAINING_MANIFEST_SCHEMA,
                    M4_LEGACY_TRAINING_MANIFEST_SCHEMA,
                },
                observed=training_manifest.get("schema"),
                requirement=(
                    f"== {M4_TRAINING_MANIFEST_SCHEMA}, or explicit legacy "
                    f"{M4_LEGACY_TRAINING_MANIFEST_SCHEMA}"
                ),
                category="provenance",
            ),
        ]
    )

    if is_m43:
        lifecycle_passed, lifecycle_observed = _m43_lifecycle_artifact_chain(
            evaluation=evaluation,
            data_audit=data_audit,
            training_manifest=training_manifest,
            data_contract=m43_data_contract,
            freeze_manifest=m43_freeze_manifest,
            locked_receipt=m43_locked_receipt,
            consumption_marker=m43_consumption_marker,
            population_plan=m43_population_plan,
            population_records=m43_population_records,
            population_merge_manifest=m43_population_merge_manifest,
            population_spot_receipt=m43_population_spot_receipt,
            population_run_manifest=m43_population_run_manifest,
            source_hashes=source_hashes,
        )
    else:
        lifecycle_passed = not lifecycle_artifacts_present
        lifecycle_observed = {
            "mode": "legacy_m4_m41_m42",
            "m43_artifacts_present": lifecycle_artifacts_present,
        }
    gates.append(
        _gate(
            "m43_lifecycle_artifact_chain",
            lifecycle_passed,
            observed=lifecycle_observed,
            requirement=(
                "M4.3 requires content-valid data-contract/freeze/one-shot receipt, "
                "byte-bound consumption marker and fresh population artifact; legacy "
                "modes require no M4.3 lifecycle artifacts"
            ),
            category="provenance",
        )
    )

    audit_gates = _mapping(data_audit.get("gates"))
    audit_required_true = (
        "hidden_discard_safe",
        "all_legal_actions_mapped",
        "candidate_evaluation_rng_disjoint",
        "within_split_shard_paths_unique",
        "within_split_hand_seeds_unique",
        "within_split_fingerprints_unique",
        "seed_ranges_disjoint",
        "fingerprints_disjoint",
    )
    gates.append(
        _gate(
            "data_correctness_audit",
            all(audit_gates.get(key) is True for key in audit_required_true),
            observed={key: audit_gates.get(key) for key in audit_required_true},
            requirement="all correctness flags are true",
            category="provenance",
        )
    )

    teacher_hand_seeds: set[int] = set()
    for split in _mapping(data_audit.get("splits")).values():
        for shard in _sequence(_mapping(split).get("shards")):
            for seed in _sequence(_mapping(shard).get("hand_seeds")):
                parsed_seed = _integer(seed)
                if parsed_seed is not None:
                    teacher_hand_seeds.add(parsed_seed)
    opponent_rows = _mapping(evaluation.get("by_opponent"))
    parsed_opponent_seed_counts = [
        _integer(_mapping(row).get("paired_seeds")) for row in opponent_rows.values()
    ]
    opponent_seed_counts = {
        count for count in parsed_opponent_seed_counts if count is not None
    }
    evaluation_seed = _integer(evaluation.get("seed")) or 0
    evaluation_seed_stride = _integer(evaluation.get("seed_stride")) or 0
    evaluation_seed_count = (
        next(iter(opponent_seed_counts)) if len(opponent_seed_counts) == 1 else 0
    )
    evaluation_hand_seeds = {
        evaluation_seed + index * evaluation_seed_stride
        for index in range(max(evaluation_seed_count, 0))
    }
    evaluation_teacher_seed_overlap = teacher_hand_seeds & evaluation_hand_seeds
    fresh_seed_contract_complete = (
        bool(teacher_hand_seeds)
        and evaluation_seed > 0
        and evaluation_seed_stride > 0
        and evaluation_seed_count > 0
        and len(opponent_rows) == len(REQUIRED_OPPONENTS)
        and all(count is not None for count in parsed_opponent_seed_counts)
        and len(opponent_seed_counts) == 1
    )
    gates.append(
        _gate(
            "fresh_evaluation_seed_disjoint",
            fresh_seed_contract_complete and not evaluation_teacher_seed_overlap,
            observed={
                "contract_complete": fresh_seed_contract_complete,
                "teacher_hand_seed_count": len(teacher_hand_seeds),
                "evaluation_hand_seed_count": len(evaluation_hand_seeds),
                "overlap_count": len(evaluation_teacher_seed_overlap),
            },
            requirement=(
                "explicit positive seed/stride and common paired-seed count, with "
                "zero overlap against every teacher split hand seed"
            ),
            category="provenance",
        )
    )

    runtime = _mapping(evaluation.get("runtime_config"))
    artifact_contract = {
        "promotion_artifact_contract": runtime.get("promotion_artifact_contract"),
        "diagnostic_legacy": runtime.get("diagnostic_legacy"),
    }
    gates.append(
        _gate(
            "single_joint_artifact_contract",
            artifact_contract["promotion_artifact_contract"] is True
            and artifact_contract["diagnostic_legacy"] is False,
            observed=artifact_contract,
            requirement="single calibrated joint artifact and diagnostic_legacy == false",
            category="provenance",
        )
    )
    current_flags = {
        "evaluation_current_profile_used": runtime.get("current_profile_used"),
        "audit_current_profile_resolved": audit_gates.get("current_profile_resolved"),
    }
    if is_m43:
        current_flags.update(
            {
                "freeze_current_profile_resolved": _mapping(
                    m43_freeze_manifest
                ).get("current_profile_resolved"),
                "receipt_current_profile_resolved": _mapping(
                    m43_locked_receipt
                ).get("current_profile_resolved"),
            }
        )
    gates.append(
        _gate(
            "current_profile_not_used",
            all(value is False for value in current_flags.values()),
            observed=current_flags,
            requirement="all current flags are false",
            category="provenance",
        )
    )
    calibration = _mapping(training_manifest.get("calibration"))
    raw_calibration_partition = training_manifest.get("calibration_partition")
    has_calibration_partition = raw_calibration_partition is not None
    calibration_partition = _mapping(raw_calibration_partition)

    holdout_flags = {
        "audit_holdout_threshold_search_allowed": audit_gates.get(
            "holdout_threshold_search_allowed"
        ),
        "manifest_threshold_adaptation_after_calibration": training_manifest.get(
            "threshold_adaptation_after_calibration"
        ),
        "manifest_locked_holdout_used_for_threshold_or_training": training_manifest.get(
            "locked_holdout_used_for_threshold_or_training"
        ),
    }
    manifest_schema = training_manifest.get("schema")
    if is_m43:
        holdout_flags.update(
            {
                "manifest_locked_status_changed_after_freeze": _mapping(
                    training_manifest.get("locked_holdout")
                )
                != {"status": "not_evaluated_pre_freeze"},
                "partition_locked_labels_used_for_roles": calibration_partition.get(
                    "locked_holdout_labels_used_for_role_assignment"
                ),
                "partition_locked_labels_opened_by_trainer": calibration_partition.get(
                    "locked_holdout_labels_opened_by_trainer"
                ),
                "calibration_locked_holdout_used": calibration.get(
                    "locked_holdout_used"
                ),
                "freeze_locked_selection_or_threshold": _mapping(
                    m43_freeze_manifest
                ).get("locked_holdout_used_for_training_selection_or_threshold"),
                "receipt_threshold_search": _mapping(m43_locked_receipt).get(
                    "threshold_search_performed"
                ),
                "receipt_model_selection": _mapping(m43_locked_receipt).get(
                    "model_selection_performed"
                ),
                "receipt_feature_selection": _mapping(m43_locked_receipt).get(
                    "feature_selection_performed"
                ),
            }
        )
    elif manifest_schema == M4_TRAINING_MANIFEST_SCHEMA and has_calibration_partition:
        holdout_flags.update(
            {
                "partition_locked_holdout_used": calibration_partition.get(
                    "locked_holdout_used"
                ),
                "calibration_locked_holdout_used": calibration.get(
                    "locked_holdout_used"
                ),
            }
        )
    gates.append(
        _gate(
            "locked_holdout_not_researched",
            all(value is False for value in holdout_flags.values()),
            observed=holdout_flags,
            requirement="all holdout adaptation/search flags are false",
            category="provenance",
        )
    )
    teacher_flags = {
        "teacher_value_runtime_gate": training_manifest.get("teacher_value_runtime_gate"),
        "threshold_source": calibration.get("threshold_source"),
    }

    if is_m43:
        overlap_keys = (
            "seed_value_count",
            "observation_fingerprint_count",
            "row_hash_count",
        )
        cross_fit = _mapping(training_manifest.get("cross_fit"))
        fold_assignment = _mapping(cross_fit.get("fold_assignment"))
        oof_coverage = _mapping(cross_fit.get("oof_coverage"))
        predictor_lineage = _mapping(cross_fit.get("predictor_lineage"))
        runtime_ensemble = _mapping(cross_fit.get("runtime_ensemble"))
        safety_examples = _mapping(cross_fit.get("safety_examples"))
        split_role_overlap = _mapping(cross_fit.get("split_role_overlap"))
        train_calibration_overlap = _mapping(
            split_role_overlap.get("train_oof__calibration")
        )
        sealed_locked_overlap = _mapping(
            split_role_overlap.get("train_oof__locked_holdout")
        )
        fit_lock_overlap = _mapping(calibration.get("fit_lock_overlap"))
        oof_train_calibration_overlap = _mapping(
            calibration.get("oof_train_calibration_overlap")
        )
        training_sources = _mapping(
            calibration.get("safety_estimator_training_sources")
        )
        train_oof_source = _mapping(training_sources.get("train_oof"))
        calibration_fit_source = _mapping(
            training_sources.get("calibration_safety_fit")
        )
        calibration_lock_source = _mapping(
            training_sources.get("calibration_threshold_lock")
        )
        locked_holdout_source = _mapping(training_sources.get("locked_holdout"))
        role_flags = {
            "safety_fit_used_for_estimator_fit": _path(
                calibration, "safety_fit", "used_for_estimator_fit"
            ),
            "safety_fit_used_for_threshold_sweep": _path(
                calibration, "safety_fit", "used_for_threshold_sweep"
            ),
            "threshold_lock_used_for_estimator_fit": _path(
                calibration, "threshold_lock", "used_for_estimator_fit"
            ),
            "threshold_lock_used_for_threshold_sweep": _path(
                calibration, "threshold_lock", "used_for_threshold_sweep"
            ),
        }
        source_role_flags = {
            "train_oof_used_for_estimator_fit": train_oof_source.get(
                "used_for_estimator_fit"
            ),
            "train_oof_used_for_threshold_sweep": train_oof_source.get(
                "used_for_threshold_sweep"
            ),
            "calibration_safety_fit_used_for_estimator_fit": (
                calibration_fit_source.get("used_for_estimator_fit")
            ),
            "calibration_safety_fit_used_for_threshold_sweep": (
                calibration_fit_source.get("used_for_threshold_sweep")
            ),
            "calibration_threshold_lock_used_for_estimator_fit": (
                calibration_lock_source.get("used_for_estimator_fit")
            ),
            "calibration_threshold_lock_used_for_threshold_sweep": (
                calibration_lock_source.get("used_for_threshold_sweep")
            ),
            "locked_holdout_used_for_estimator_fit": locked_holdout_source.get(
                "used_for_estimator_fit"
            ),
            "locked_holdout_used_for_threshold_sweep": locked_holdout_source.get(
                "used_for_threshold_sweep"
            ),
        }
        expected_role_flags = {
            "safety_fit_used_for_estimator_fit": True,
            "safety_fit_used_for_threshold_sweep": False,
            "threshold_lock_used_for_estimator_fit": False,
            "threshold_lock_used_for_threshold_sweep": True,
        }
        expected_source_role_flags = {
            "train_oof_used_for_estimator_fit": True,
            "train_oof_used_for_threshold_sweep": False,
            "calibration_safety_fit_used_for_estimator_fit": True,
            "calibration_safety_fit_used_for_threshold_sweep": False,
            "calibration_threshold_lock_used_for_estimator_fit": False,
            "calibration_threshold_lock_used_for_threshold_sweep": True,
            "locked_holdout_used_for_estimator_fit": False,
            "locked_holdout_used_for_threshold_sweep": False,
        }
        cross_fit_folds = _integer(cross_fit.get("folds")) or 0
        fold_audits = _sequence(predictor_lineage.get("fold_audits"))
        observed_outer_folds: set[int] = set()
        inner_fold_audits: list[dict[str, Any]] = []
        inner_fold_lineage_valid = len(fold_audits) == cross_fit_folds
        for raw_audit in fold_audits:
            audit = _mapping(raw_audit)
            fold = _integer(audit.get("fold"))
            audit_overlap = _mapping(
                audit.get("train__validation_identity_overlap")
            )
            inner = _mapping(audit.get("oof_safety_inner_ensemble"))
            inner_assignment = _mapping(inner.get("fold_assignment"))
            row_valid = (
                fold is not None
                and 0 <= fold < cross_fit_folds
                and fold not in observed_outer_folds
                and (_integer(audit.get("training_samples")) or 0) > 0
                and (_integer(audit.get("validation_samples")) or 0) > 0
                and set(audit_overlap) == set(overlap_keys)
                and all(
                    _integer(audit_overlap.get(key)) == 0
                    for key in overlap_keys
                )
                and audit.get(
                    "validation_identity_excluded_from_all_paired_heads"
                )
                is True
                and _integer(audit.get("identity_leakage_count")) == 0
                and _integer(inner.get("fold_count")) == cross_fit_folds
                and inner.get("aggregation")
                == "mean_plus_cross_fold_disagreement"
                and inner.get("matches_runtime_fold_count") is True
                and _integer(inner_assignment.get("folds"))
                == cross_fit_folds
                and _integer(inner_assignment.get("identity_leakage_count"))
                == 0
            )
            if fold is not None:
                observed_outer_folds.add(fold)
            inner_fold_lineage_valid = inner_fold_lineage_valid and row_valid
            inner_fold_audits.append(
                {
                    "fold": fold,
                    "inner_fold_count": inner.get("fold_count"),
                    "inner_assignment_folds": inner_assignment.get("folds"),
                    "inner_identity_leakage_count": inner_assignment.get(
                        "identity_leakage_count"
                    ),
                    "valid": row_valid,
                }
            )
        inner_fold_lineage_valid = (
            inner_fold_lineage_valid
            and observed_outer_folds == set(range(cross_fit_folds))
        )
        calibration_provenance_observed = {
            "mode": "m43_paired_delta_pre_freeze_one_shot",
            "partition_schema": calibration_partition.get("schema"),
            "partition_status": calibration_partition.get("status"),
            "partition_overlap": calibration_partition.get("overlap"),
            "cross_fit_schema": cross_fit.get("schema"),
            "cross_fit_status": cross_fit.get("status"),
            "cross_fit_folds": cross_fit.get("folds"),
            "cross_fit_action_score_mode": cross_fit.get("action_score_mode"),
            "fold_assignment_identity_leakage_count": fold_assignment.get(
                "identity_leakage_count"
            ),
            "oof_coverage": dict(oof_coverage),
            "predictor_lineage": dict(predictor_lineage),
            "runtime_ensemble": dict(runtime_ensemble),
            "inner_fold_lineage_valid": inner_fold_lineage_valid,
            "inner_fold_audits": inner_fold_audits,
            "train_oof_calibration_overlap": dict(train_calibration_overlap),
            "train_oof_locked_contract": dict(sealed_locked_overlap),
            "fit_lock_overlap": dict(fit_lock_overlap),
            "oof_train_calibration_overlap": dict(
                oof_train_calibration_overlap
            ),
            "safety_estimator_source": calibration.get(
                "safety_estimator_source"
            ),
            "threshold_source": calibration.get("threshold_source"),
            "threshold_selection_source": calibration.get(
                "threshold_selection_source"
            ),
            "role_flags": role_flags,
            "source_role_flags": source_role_flags,
        }
        calibration_provenance_valid = (
            manifest_schema == M4_TRAINING_MANIFEST_SCHEMA
            and calibration_partition.get("schema")
            == "hu_m43_trainer_data_contract_binding_v1"
            and calibration_partition.get("status") == "pass"
            and _integer(calibration_partition.get("overlap")) == 0
            and calibration_partition.get("legacy_hash_partition_used") is False
            and calibration_partition.get(
                "locked_holdout_labels_used_for_role_assignment"
            )
            is False
            and calibration_partition.get(
                "locked_holdout_labels_opened_by_trainer"
            )
            is False
            and cross_fit.get("schema") == M43_CROSS_FIT_SCHEMA
            and cross_fit.get("status") == "pass"
            and cross_fit_folds >= 2
            and cross_fit.get("action_score_mode") == M43_ACTION_SCORE_MODE
            and _integer(fold_assignment.get("identity_leakage_count")) == 0
            and oof_coverage.get("paired_head_prediction_counts") == [1]
            and oof_coverage.get("each_sample_predicted_exactly_once") is True
            and predictor_lineage.get("schema")
            == "hu_m4_paired_outer_fold_predictor_lineage_v1"
            and _integer(predictor_lineage.get("identity_leakage_count")) == 0
            and predictor_lineage.get(
                "all_outer_validation_identities_excluded"
            )
            is True
            and runtime_ensemble.get("source")
            == "stored_crossfit_fold_estimators"
            and _integer(runtime_ensemble.get("fold_count"))
            == cross_fit_folds
            and _integer(runtime_ensemble.get("oof_safety_fold_count"))
            == cross_fit_folds
            and runtime_ensemble.get("full_refit_used_at_runtime") is False
            and runtime_ensemble.get(
                "oof_to_full_refit_distribution_shift"
            )
            is False
            and runtime_ensemble.get("oof_safety_aggregation_semantics_match")
            is True
            and runtime_ensemble.get("baseline_action_score_exact_zero") is True
            and inner_fold_lineage_valid
            and safety_examples.get("source") == "train_oof_only"
            and safety_examples.get("used_for_threshold_lock") is False
            and safety_examples.get("used_for_locked_holdout") is False
            and cross_fit.get("locked_holdout_used") is False
            and cross_fit.get("threshold_lock_used") is False
            and all(
                _integer(train_calibration_overlap.get(key)) == 0
                for key in overlap_keys
            )
            and sealed_locked_overlap
            == {
                "status": "sealed_contract_only_not_opened_by_trainer",
                "identity_overlap_not_computed_from_locked_labels": True,
            }
            and all(
                _integer(fit_lock_overlap.get(key)) == 0
                for key in overlap_keys
            )
            and all(
                _integer(oof_train_calibration_overlap.get(key)) == 0
                for key in overlap_keys
            )
            and role_flags == expected_role_flags
            and source_role_flags == expected_source_role_flags
            and calibration.get("safety_estimator_source")
            == "train_oof_plus_safety_fit_subset"
            and calibration.get("threshold_source")
            == "threshold_lock_subset_only"
            and calibration.get("threshold_selection_source")
            == "calibration.threshold_lock"
            and calibration.get("safety_calibrator_sources")
            == ["train_oof", "calibration.safety_fit"]
            and calibration.get("runtime_inputs_exclude_teacher_values_and_teacher_lcb")
            is True
        )
    elif has_calibration_partition:
        partition_overlap = _mapping(calibration_partition.get("overlap"))
        calibration_overlap = _mapping(calibration.get("fit_lock_overlap"))
        overlap_keys = (
            "seed_value_count",
            "observation_fingerprint_count",
            "row_hash_count",
        )
        role_flags = {
            "safety_fit_used_for_estimator_fit": _path(
                calibration, "safety_fit", "used_for_estimator_fit"
            ),
            "safety_fit_used_for_threshold_sweep": _path(
                calibration, "safety_fit", "used_for_threshold_sweep"
            ),
            "threshold_lock_used_for_estimator_fit": _path(
                calibration, "threshold_lock", "used_for_estimator_fit"
            ),
            "threshold_lock_used_for_threshold_sweep": _path(
                calibration, "threshold_lock", "used_for_threshold_sweep"
            ),
        }
        safety_estimator_source = calibration.get("safety_estimator_source")
        oof_provenance_observed: dict[str, Any] | None = None
        if safety_estimator_source == "safety_fit_subset_only":
            safety_estimator_source_valid = True
        elif safety_estimator_source == "train_oof_plus_safety_fit_subset":
            cross_fit = _mapping(training_manifest.get("cross_fit"))
            fold_assignment = _mapping(cross_fit.get("fold_assignment"))
            oof_coverage = _mapping(cross_fit.get("oof_coverage"))
            predictor_lineage = _mapping(cross_fit.get("predictor_lineage"))
            safety_examples = _mapping(cross_fit.get("safety_examples"))
            split_role_overlap = _mapping(cross_fit.get("split_role_overlap"))
            train_calibration_overlap = _mapping(
                split_role_overlap.get("train_oof__calibration")
            )
            train_holdout_overlap = _mapping(
                split_role_overlap.get("train_oof__locked_holdout")
            )
            training_sources = _mapping(
                calibration.get("safety_estimator_training_sources")
            )
            train_oof_source = _mapping(training_sources.get("train_oof"))
            calibration_fit_source = _mapping(
                training_sources.get("calibration_safety_fit")
            )
            calibration_lock_source = _mapping(
                training_sources.get("calibration_threshold_lock")
            )
            locked_holdout_source = _mapping(training_sources.get("locked_holdout"))
            oof_train_calibration_overlap = _mapping(
                calibration.get("oof_train_calibration_overlap")
            )
            source_role_flags = {
                "train_oof_used_for_estimator_fit": train_oof_source.get(
                    "used_for_estimator_fit"
                ),
                "train_oof_used_for_threshold_sweep": train_oof_source.get(
                    "used_for_threshold_sweep"
                ),
                "calibration_safety_fit_used_for_estimator_fit": (
                    calibration_fit_source.get("used_for_estimator_fit")
                ),
                "calibration_safety_fit_used_for_threshold_sweep": (
                    calibration_fit_source.get("used_for_threshold_sweep")
                ),
                "calibration_threshold_lock_used_for_estimator_fit": (
                    calibration_lock_source.get("used_for_estimator_fit")
                ),
                "calibration_threshold_lock_used_for_threshold_sweep": (
                    calibration_lock_source.get("used_for_threshold_sweep")
                ),
                "locked_holdout_used_for_estimator_fit": locked_holdout_source.get(
                    "used_for_estimator_fit"
                ),
                "locked_holdout_used_for_threshold_sweep": locked_holdout_source.get(
                    "used_for_threshold_sweep"
                ),
            }
            oof_provenance_observed = {
                "cross_fit_schema": cross_fit.get("schema"),
                "cross_fit_status": cross_fit.get("status"),
                "cross_fit_folds": cross_fit.get("folds"),
                "cross_fit_action_score_mode": cross_fit.get("action_score_mode"),
                "fold_assignment_identity_leakage_count": fold_assignment.get(
                    "identity_leakage_count"
                ),
                "oof_coverage": dict(oof_coverage),
                "predictor_lineage_schema": predictor_lineage.get("schema"),
                "nested_cross_fit": predictor_lineage.get("nested_cross_fit"),
                "predictor_identity_leakage_count": predictor_lineage.get(
                    "identity_leakage_count"
                ),
                "all_outer_validation_identities_excluded": predictor_lineage.get(
                    "all_outer_validation_identities_excluded"
                ),
                "safety_examples_used_for_threshold_lock": safety_examples.get(
                    "used_for_threshold_lock"
                ),
                "safety_examples_used_for_locked_holdout": safety_examples.get(
                    "used_for_locked_holdout"
                ),
                "safety_examples_source": safety_examples.get("source"),
                "paired_common_future_delta_contract": audit_gates.get(
                    "paired_common_future_delta_contract"
                ),
                "oof_train_calibration_overlap": dict(
                    oof_train_calibration_overlap
                ),
                "train_oof_calibration_overlap": dict(train_calibration_overlap),
                "train_oof_locked_holdout_overlap": dict(train_holdout_overlap),
                "source_role_flags": source_role_flags,
            }
            safety_estimator_source_valid = (
                cross_fit.get("schema")
                == "hu_m4_identity_group_nested_cross_fit_v2"
                and cross_fit.get("status") == "pass"
                and _integer(cross_fit.get("folds")) >= 2
                and cross_fit.get("action_score_mode")
                == "negative_regret_ranker_v2"
                and _integer(fold_assignment.get("identity_leakage_count")) == 0
                and oof_coverage.get("base_head_prediction_counts") == [1]
                and oof_coverage.get("meta_rank_prediction_counts") == [1]
                and oof_coverage.get("uncertainty_prediction_counts") == [1]
                and oof_coverage.get("each_sample_predicted_exactly_once") is True
                and predictor_lineage.get("schema")
                == "hu_m4_outer_fold_predictor_lineage_v1"
                and predictor_lineage.get("nested_cross_fit") is True
                and _integer(predictor_lineage.get("identity_leakage_count")) == 0
                and predictor_lineage.get(
                    "all_outer_validation_identities_excluded"
                )
                is True
                and safety_examples.get("used_for_threshold_lock") is False
                and safety_examples.get("used_for_locked_holdout") is False
                and safety_examples.get("source") == "train_oof_only"
                and cross_fit.get("locked_holdout_used") is False
                and cross_fit.get("threshold_lock_used") is False
                and all(
                    _integer(oof_train_calibration_overlap.get(key)) == 0
                    for key in overlap_keys
                )
                and all(
                    _integer(train_calibration_overlap.get(key)) == 0
                    for key in overlap_keys
                )
                and all(
                    _integer(train_holdout_overlap.get(key)) == 0
                    for key in overlap_keys
                )
                and source_role_flags
                == {
                    "train_oof_used_for_estimator_fit": True,
                    "train_oof_used_for_threshold_sweep": False,
                    "calibration_safety_fit_used_for_estimator_fit": True,
                    "calibration_safety_fit_used_for_threshold_sweep": False,
                    "calibration_threshold_lock_used_for_estimator_fit": False,
                    "calibration_threshold_lock_used_for_threshold_sweep": True,
                    "locked_holdout_used_for_estimator_fit": False,
                    "locked_holdout_used_for_threshold_sweep": False,
                }
                and audit_gates.get("paired_common_future_delta_contract") is True
            )
        else:
            safety_estimator_source_valid = False
        calibration_provenance_observed = {
            "mode": "disjoint_safety_fit_threshold_lock",
            "partition_schema": calibration_partition.get("schema"),
            "partition_status": calibration_partition.get("status"),
            "partition_overlap": dict(partition_overlap),
            "calibration_overlap": dict(calibration_overlap),
            "safety_estimator_source": safety_estimator_source,
            "threshold_source": calibration.get("threshold_source"),
            "role_flags": role_flags,
        }
        if oof_provenance_observed is not None:
            calibration_provenance_observed["train_oof_provenance"] = (
                oof_provenance_observed
            )
        calibration_provenance_valid = (
            calibration_partition.get("schema")
            == "hu_m4_safety_calibration_partition_v1"
            and calibration_partition.get("status") == "pass"
            and all(_integer(partition_overlap.get(key)) == 0 for key in overlap_keys)
            and all(_integer(calibration_overlap.get(key)) == 0 for key in overlap_keys)
            and safety_estimator_source_valid
            and calibration.get("threshold_source")
            == "threshold_lock_subset_only"
            and role_flags
            == {
                "safety_fit_used_for_estimator_fit": True,
                "safety_fit_used_for_threshold_sweep": False,
                "threshold_lock_used_for_estimator_fit": False,
                "threshold_lock_used_for_threshold_sweep": True,
            }
        )
    elif (
        manifest_schema == M4_LEGACY_TRAINING_MANIFEST_SCHEMA
        and not has_calibration_partition
    ):
        # Compatibility for the already-issued v1 manifest.  Once a partition
        # field is present, none of these legacy rules can mask a malformed or
        # partially populated M4.1 partition.
        calibration_provenance_observed = {
            "mode": "legacy_single_calibration_file",
            "threshold_source": calibration.get("threshold_source"),
        }
        calibration_provenance_valid = (
            calibration.get("threshold_source") == "calibration_file_only"
            and calibration.get("safety_estimator_source") is None
        )
    else:
        # A v2 file cannot omit its partition, and a v1 file cannot acquire a
        # partial/new partition while retaining legacy acceptance semantics.
        calibration_provenance_observed = {
            "mode": "schema_partition_contract_mismatch",
            "manifest_schema": manifest_schema,
            "partition_present": has_calibration_partition,
            "threshold_source": calibration.get("threshold_source"),
        }
        calibration_provenance_valid = False
    gates.append(
        _gate(
            "safety_calibration_provenance",
            calibration_provenance_valid,
            observed=calibration_provenance_observed,
            requirement=(
                "legacy calibration_file_only, or a passing disjoint v1 partition "
                "with zero fit/lock overlap and exclusive estimator/threshold roles; "
                "train OOF augmentation additionally requires leak-free nested cross-fit"
            ),
            category="provenance",
        )
    )
    gates.append(
        _gate(
            "teacher_metrics_excluded_from_promotion",
            teacher_flags["teacher_value_runtime_gate"] is False
            and calibration_provenance_valid,
            observed=teacher_flags,
            requirement=(
                "teacher runtime gate false and safety calibration provenance valid"
            ),
            category="provenance",
        )
    )

    manifest_threshold = _path(training_manifest, "calibration", "selected_threshold")
    locked_threshold = (
        _mapping(m43_freeze_manifest).get("frozen_threshold")
        if is_m43
        else _path(
            training_manifest, "locked_holdout", "frozen_safety_threshold"
        )
    )
    runtime_threshold = runtime.get("safety_threshold")
    gates.append(
        _gate(
            "frozen_threshold_match",
            _same_number(manifest_threshold, runtime_threshold)
            and _same_number(manifest_threshold, locked_threshold),
            observed={
                "manifest_selected": manifest_threshold,
                "locked_holdout_frozen": locked_threshold,
                "evaluation_runtime": runtime_threshold,
            },
            requirement="manifest selected == locked holdout frozen == runtime threshold",
            category="provenance",
        )
    )

    runtime_candidate_hash = _hash_string(runtime.get("candidate_model_sha256"))
    runtime_safety_hash = _hash_string(runtime.get("safety_model_sha256"))
    manifest_candidate_hash, manifest_safety_hash = _manifest_runtime_hashes(training_manifest)
    artifact_hash_match = (
        runtime_candidate_hash is not None
        and runtime_safety_hash is not None
        and manifest_candidate_hash is not None
        and manifest_safety_hash is not None
        and runtime_candidate_hash == manifest_candidate_hash
        and runtime_safety_hash == manifest_safety_hash
        and (
            not is_m43
            or (
                runtime_candidate_hash
                == _hash_string(_mapping(m43_freeze_manifest).get("model_sha256"))
                and runtime_safety_hash
                == _hash_string(_mapping(m43_locked_receipt).get("model_sha256"))
            )
        )
    )
    gates.append(
        _gate(
            "runtime_artifact_hash_match",
            artifact_hash_match,
            observed={
                "manifest_candidate": manifest_candidate_hash,
                "manifest_safety": manifest_safety_hash,
                "runtime_candidate": runtime_candidate_hash,
                "runtime_safety": runtime_safety_hash,
            },
            requirement="explicit manifest candidate/safety SHA-256 values match runtime SHA-256 values",
            category="provenance",
        )
    )

    population = _mapping(evaluation.get("population"))
    all_seats = _mapping(population.get("all_seats"))
    by_seat = _mapping(population.get("by_seat"))
    first = _mapping(by_seat.get("first"))
    second = _mapping(by_seat.get("second"))
    paired = _mapping(population.get("paired_seat_swap"))
    gates.append(
        _gate(
            "counterfactual_and_cluster_contract",
            evaluation.get("promotion_eligible_counterfactual_contract") is True
            and evaluation.get("primary_metrics_exclude_invalid_records") is True
            and paired.get("ci_independence_unit")
            == "hand_seed_cluster_after_opponent_average",
            observed={
                "promotion_eligible": evaluation.get(
                    "promotion_eligible_counterfactual_contract"
                ),
                "invalid_excluded": evaluation.get(
                    "primary_metrics_exclude_invalid_records"
                ),
                "ci_unit": paired.get("ci_independence_unit"),
            },
            requirement="valid counterfactuals only and seed-clustered population CI",
            category="provenance",
        )
    )

    invalid = _integer(evaluation.get("invalid_counterfactuals"))
    gates.append(
        _gate(
            "invalid_counterfactuals",
            invalid == 0 and _integer(all_seats.get("invalid_counterfactuals")) == 0,
            observed={"top_level": invalid, "population": all_seats.get("invalid_counterfactuals")},
            requirement="== 0",
        )
    )
    cancellation = {
        "mismatches": all_seats.get("nonfire_cancellation_mismatches"),
        "nonzero": all_seats.get("nonfire_nonzero_deltas"),
        "unknown": all_seats.get("nonfire_cancellation_unknown"),
        "top_level_mismatches": evaluation.get("nonfire_cancellation_mismatches"),
    }
    gates.append(
        _gate(
            "nonfire_counterfactual_cancellation",
            all(_integer(value) == 0 for value in cancellation.values()),
            observed=cancellation,
            requirement="mismatch, nonzero and unknown counts all == 0",
        )
    )

    overrides = _integer(all_seats.get("overrides"))
    gates.append(
        _gate(
            "minimum_valid_overrides",
            overrides is not None and invalid == 0 and overrides >= 300,
            observed=overrides,
            requirement=">= 300 valid overrides",
        )
    )
    gain_ci_low = _number(_path(all_seats, "realized_gain_per_override", "ci95_low"))
    gates.append(
        _gate(
            "realized_gain_per_override_ci95_low",
            gain_ci_low is not None and gain_ci_low > 0.0,
            observed=gain_ci_low,
            requirement="> 0",
        )
    )
    paired_ci_low = _number(_path(paired, "delta_ev_per_hand", "ci95_low"))
    gates.append(
        _gate(
            "paired_seat_swap_delta_ci95_low",
            paired_ci_low is not None and paired_ci_low > 0.0,
            observed=paired_ci_low,
            requirement="> 0",
        )
    )
    second_ci_low = _number(_path(second, "delta_ev_per_hand", "ci95_low"))
    gates.append(
        _gate(
            "second_seat_delta_ci95_low",
            second_ci_low is not None and second_ci_low > 0.0,
            observed=second_ci_low,
            requirement="> 0",
        )
    )

    first_delta = _mapping(first.get("delta_ev_per_hand"))
    first_exact = all(
        _same_number(first_delta.get(key), 0.0) for key in ("mean", "ci95_low", "ci95_high")
    ) and _integer(first.get("overrides")) == 0
    gates.append(
        _gate(
            "first_seat_exact_cancellation",
            first_exact,
            observed={
                "mean": first_delta.get("mean"),
                "ci95_low": first_delta.get("ci95_low"),
                "ci95_high": first_delta.get("ci95_high"),
                "overrides": first.get("overrides"),
            },
            requirement="delta mean/CI exactly 0 and overrides == 0",
        )
    )

    false_positive_rate = _number(all_seats.get("false_positive_override_rate"))
    gates.append(
        _gate(
            "false_positive_override_rate",
            false_positive_rate is not None and false_positive_rate <= 0.30,
            observed=false_positive_rate,
            requirement="<= 0.30",
        )
    )
    loss_tail = _mapping(all_seats.get("override_loss_tail"))
    tail_observed = {name: _number(loss_tail.get(name)) for name in ("p95", "p99", "max")}
    gates.append(
        _gate(
            "override_loss_tail",
            tail_observed["p95"] is not None
            and tail_observed["p95"] <= 25.0
            and tail_observed["p99"] is not None
            and tail_observed["p99"] <= 40.0
            and tail_observed["max"] is not None
            and tail_observed["max"] <= 50.0,
            observed=tail_observed,
            requirement="p95 <= 25, p99 <= 40, max <= 50",
        )
    )

    opponent_rows = _mapping(evaluation.get("by_opponent"))
    declared_opponents = evaluation.get("opponents")
    declared_set = set(declared_opponents) if isinstance(declared_opponents, list) else set()
    population_complete = declared_set == set(REQUIRED_OPPONENTS) == set(opponent_rows)
    gates.append(
        _gate(
            "required_opponent_population",
            population_complete,
            observed={"declared": sorted(declared_set), "summaries": sorted(opponent_rows)},
            requirement=f"exactly {list(REQUIRED_OPPONENTS)}",
        )
    )

    opponent_observed: dict[str, Any] = {}
    opponent_pass = population_complete
    opponent_means: dict[str, float] = {}
    for opponent in REQUIRED_OPPONENTS:
        delta = _mapping(_path(opponent_rows, opponent, "seat_swap", "delta_ev_per_hand"))
        mean = _number(delta.get("mean"))
        ci_low = _number(delta.get("ci95_low"))
        opponent_observed[opponent] = {"mean": mean, "ci95_low": ci_low}
        if mean is not None:
            opponent_means[opponent] = mean
        opponent_pass = opponent_pass and mean is not None and mean >= -0.005
        opponent_pass = opponent_pass and ci_low is not None and ci_low >= -0.02
    gates.append(
        _gate(
            "each_opponent_robustness",
            opponent_pass,
            observed=opponent_observed,
            requirement="each point >= -0.005 and each CI95 low >= -0.02",
        )
    )

    worst = _mapping(evaluation.get("worst_case_opponent"))
    worst_delta = _mapping(worst.get("delta_ev_per_hand"))
    computed_worst = min(opponent_means, key=opponent_means.get) if len(opponent_means) == len(REQUIRED_OPPONENTS) else None
    worst_mean = _number(worst_delta.get("mean"))
    worst_ci_low = _number(worst_delta.get("ci95_low"))
    worst_consistent = (
        computed_worst is not None
        and worst.get("opponent") == computed_worst
        and _same_number(worst_mean, opponent_observed[computed_worst]["mean"])
        and _same_number(worst_ci_low, opponent_observed[computed_worst]["ci95_low"])
    )
    gates.append(
        _gate(
            "population_worst_case_robustness",
            worst_consistent
            and worst_mean is not None
            and worst_mean >= -0.005
            and worst_ci_low is not None
            and worst_ci_low >= -0.02,
            observed={
                "reported_opponent": worst.get("opponent"),
                "computed_opponent": computed_worst,
                "mean": worst_mean,
                "ci95_low": worst_ci_low,
            },
            requirement="consistent population minimum with point >= -0.005 and CI95 low >= -0.02",
        )
    )

    passed = all(gate["passed"] for gate in gates)
    config = {
        "schema": M4_ACCEPTANCE_CONFIG_SCHEMA,
        "decision_scope": "m4_t1_second_selective_override_only",
        "fixed_gates": dict(FIXED_GATES),
        "required_opponents": list(REQUIRED_OPPONENTS),
        "threshold_selection": "calibration_only_locked_before_population_evaluation",
        "teacher_metrics_for_promotion": False,
        "top1_accuracy_for_promotion": False,
        "current_profile_change_allowed": False,
        "lifecycle_contract": (
            "m43_pre_freeze_one_shot_hash_chain"
            if is_m43
            else "legacy_m4_m41_m42"
        ),
        "source_sha256": source_hashes,
    }
    failed_names = [gate["name"] for gate in gates if not gate["passed"]]
    status = {
        "schema": M4_ACCEPTANCE_STATUS_SCHEMA,
        "status": "complete_go" if passed else "complete_no_go",
        "promotion_decision": "go" if passed else "no_go",
        "scope": "m4_t1_second_selective_override_only",
        "gates_passed": len(gates) - len(failed_names),
        "gates_total": len(gates),
        "failed_gates": failed_names,
        "gates": gates,
        "source_sha256": source_hashes,
        "teacher_metrics_used_for_promotion": False,
        "top1_accuracy_used_for_promotion": False,
        "current_profile_changed": False,
        "m43_lifecycle_required": is_m43,
        "insufficient_sample_policy": "complete_no_go",
    }
    return config, status


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--m43-data-contract", type=Path)
    parser.add_argument("--m43-freeze-manifest", type=Path)
    parser.add_argument("--m43-locked-receipt", type=Path)
    parser.add_argument("--m43-consumption-marker", type=Path)
    parser.add_argument("--m43-population-plan", type=Path)
    parser.add_argument("--m43-population-records", type=Path)
    parser.add_argument("--m43-population-merge-manifest", type=Path)
    parser.add_argument("--m43-population-spot-receipt", type=Path)
    parser.add_argument("--m43-population-run-manifest", type=Path)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    args = parser.parse_args(argv)
    lifecycle = (
        args.m43_data_contract,
        args.m43_freeze_manifest,
        args.m43_locked_receipt,
        args.m43_consumption_marker,
        args.m43_population_plan,
        args.m43_population_records,
        args.m43_population_merge_manifest,
        args.m43_population_spot_receipt,
        args.m43_population_run_manifest,
    )
    if any(path is not None for path in lifecycle) and not all(
        path is not None for path in lifecycle
    ):
        parser.error("M4.3 lifecycle inputs must be supplied together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = {
        "evaluation": args.evaluation,
        "data_audit": args.data_audit,
        "training_manifest": args.training_manifest,
    }
    optional_inputs = {
        "m43_data_contract": args.m43_data_contract,
        "m43_freeze_manifest": args.m43_freeze_manifest,
        "m43_locked_receipt": args.m43_locked_receipt,
        "m43_consumption_marker": args.m43_consumption_marker,
        "m43_population_plan": args.m43_population_plan,
        "m43_population_records": args.m43_population_records,
        "m43_population_merge_manifest": args.m43_population_merge_manifest,
        "m43_population_spot_receipt": args.m43_population_spot_receipt,
        "m43_population_run_manifest": args.m43_population_run_manifest,
    }
    inputs.update(
        {name: path for name, path in optional_inputs.items() if path is not None}
    )
    resolved_inputs = {name: path.resolve() for name, path in inputs.items()}
    outputs = {args.config_output.resolve(), args.status_output.resolve()}
    if len(set(resolved_inputs.values())) != len(resolved_inputs):
        raise ValueError("all acceptance input artifacts must be distinct")
    if len(outputs) != 2 or outputs & set(resolved_inputs.values()):
        raise ValueError("config/status outputs must be distinct from each other and inputs")
    payloads: dict[str, Any] = {
        name: (
            _load_jsonl(path)
            if name == "m43_population_records"
            else _load_json(path)
        )
        for name, path in resolved_inputs.items()
    }
    hashes = {name: _sha256(path) for name, path in resolved_inputs.items()}
    config, status = validate_hu_m4_acceptance(
        payloads["evaluation"],
        payloads["data_audit"],
        payloads["training_manifest"],
        m43_data_contract=payloads.get("m43_data_contract"),
        m43_freeze_manifest=payloads.get("m43_freeze_manifest"),
        m43_locked_receipt=payloads.get("m43_locked_receipt"),
        m43_consumption_marker=payloads.get("m43_consumption_marker"),
        m43_population_plan=payloads.get("m43_population_plan"),
        m43_population_records=payloads.get("m43_population_records"),
        m43_population_merge_manifest=payloads.get(
            "m43_population_merge_manifest"
        ),
        m43_population_spot_receipt=payloads.get(
            "m43_population_spot_receipt"
        ),
        m43_population_run_manifest=payloads.get(
            "m43_population_run_manifest"
        ),
        source_hashes=hashes,
    )
    _write_json_atomic(args.config_output, config)
    _write_json_atomic(args.status_output, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    # A completed No-Go is a valid output, so both decisions exit successfully.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIXED_GATES",
    "M4_ACCEPTANCE_CONFIG_SCHEMA",
    "M4_ACCEPTANCE_STATUS_SCHEMA",
    "REQUIRED_OPPONENTS",
    "validate_hu_m4_acceptance",
]
