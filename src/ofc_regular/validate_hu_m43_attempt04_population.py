"""Fail-closed Attempt04 v6 population preflight and acceptance.

Only runtime artifacts, hash-only lifecycle receipts, population records, and
transport manifests are accepted here.  Teacher/calibration/locked JSONL is
never an input.  Final summaries are recomputed from played-hand records and
the fixed M4 realized-EV gates are reused without threshold search.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_hu_m4_population import (
    M4_OPPONENT_PROFILES,
    summarize_hu_m4_population_records,
)
from .evaluate_hu_m43_attempt04_locked import validate_attempt04_locked_receipt
from .hu_m43_attempt04_runtime import (
    M43_ATTEMPT04_LOCKED_MARKER_SCHEMA,
    M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA,
    file_sha256,
    load_bound_attempt04_v6_model,
    read_json_mapping,
    self_digest,
    validate_attempt04_runtime_artifact_files,
    validate_attempt04_runtime_freeze,
)
from .hu_m43_joint_model_v6 import (
    HU_M43_V6_ACTION_SCORE_MODE,
    HU_M43_V6_ARTIFACT_SCHEMA,
    HU_M43_V6_MODEL_SCHEMA,
    HU_M43_V6_PROPOSAL_SCHEMA,
    HU_M43_V6_SAFETY_FEATURE_DIM,
    HU_M43_V6_SAFETY_FEATURE_SCHEMA,
    HuM43JointModelV6,
)
from .merge_hu_m4_population_shards import M4_POPULATION_MERGE_SCHEMA
from .validate_hu_m43_attempt02_acceptance import (
    FIXED_GATES,
    evaluate_attempt02_population_gates,
)


ATTEMPT04_POPULATION_PLAN_SCHEMA = "hu_m43_population_acceptance_plan_v1"
ATTEMPT04_POPULATION_PREFLIGHT_SCHEMA = (
    "hu_m43_attempt04_population_launch_preflight_v1"
)
ATTEMPT04_POPULATION_MANIFEST_SCHEMA = (
    "hu_m43_attempt04_population_spot_manifest_v1"
)
ATTEMPT04_POPULATION_RECEIPT_SCHEMA = (
    "hu_m43_attempt04_population_spot_receipt_v1"
)
ATTEMPT04_POPULATION_ACCEPTANCE_CONFIG_SCHEMA = (
    "hu_m43_attempt04_population_acceptance_config_v1"
)
ATTEMPT04_POPULATION_ACCEPTANCE_STATUS_SCHEMA = (
    "hu_m43_attempt04_population_acceptance_status_v1"
)
ATTEMPT04_POPULATION_SEED = 14_106_071_901
ATTEMPT04_POPULATION_SEED_STRIDE = 1_000_003
ATTEMPT04_POPULATION_SEEDS = 1000
ATTEMPT04_POPULATION_SHARDS = 20
ATTEMPT04_SEEDS_PER_SHARD = 50
EXPECTED_OPPONENTS = tuple(M4_OPPONENT_PROFILES)

_EVALUATION_CONTRACT = {
    "candidate_and_baseline_same_hand_seed": True,
    "candidate_and_baseline_same_physical_seat": True,
    "candidate_and_baseline_same_opponent_policy_seed": True,
    "first_and_second_seat_metrics": True,
    "nonfire_full_trajectory_digest_cancellation": True,
    "invalid_counterfactuals_excluded": True,
    "false_positive_override_definition": "realized_delta_le_zero",
    "confidence_interval_unit": "hand_seed_cluster_after_opponent_average",
    "teacher_values_are_realized_match_ev": False,
    "threshold_reselection_allowed": False,
}
_SPOT_CONTRACT = {
    "small_shards": True,
    "checkpoint_unit": "completed_shard",
    "heartbeat_required": True,
    "resume_missing_shards_only": True,
    "done_commit_last": True,
    "merge_recomputes_metrics_from_records": True,
}
_ACTIVATION_GUARDS = {
    "current_profile_changed": False,
    "runtime_policy_activated": False,
    "full_replacement_enabled": False,
    "threshold_changed_after_lock": False,
}
_REQUIRED_ATTEMPT04_TEACHER_SCHEDULES = {
    ("precal_holdout", 13_106_071_901, 1_000_003, 300),
    ("calibration_safety_fit", 13_506_071_901, 1_000_003, 100),
    ("calibration_threshold_lock", 13_606_071_901, 1_000_003, 100),
    ("locked_holdout", 13_806_071_901, 1_000_003, 200),
}
_REQUIRED_ATTEMPT04_RNG_SCHEDULES = {
    (f"{role}.{kind}", base + offset, 1_000_003, roots)
    for role, base, roots in (
        ("precal_holdout", 106_071_901, 300),
        ("calibration_safety_fit", 506_071_901, 100),
        ("calibration_threshold_lock", 606_071_901, 100),
        ("locked_holdout", 806_071_901, 200),
    )
    for kind, offset in (
        ("candidate", 20_000_000_000),
        ("evaluation", 21_000_000_000),
        ("child_policy", 22_000_000_000),
    )
}
_REQUIRED_PRIOR_POPULATION_SCHEDULE = (
    "M4.3-attempt03-final-population",
    12_106_071_901,
    1_000_003,
    1000,
)


def load_and_validate_attempt04_population_plan(
    source: str | Path | Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    plan = (
        dict(source)
        if isinstance(source, Mapping)
        else read_json_mapping(source, "Attempt04 population plan")
    )
    if (
        plan.get("schema") != ATTEMPT04_POPULATION_PLAN_SCHEMA
        or plan.get("status") != "frozen_before_population_evaluation"
        or plan.get("policy_attempt") != "attempt04_v6"
        or plan.get("fixed_baseline_profile") != "stage18_p1"
        or plan.get("opponents") != list(EXPECTED_OPPONENTS)
        or plan.get("paired_seat_swap") is not True
    ):
        raise ValueError("Attempt04 population plan identity changed")
    runtime = _mapping(plan.get("runtime_contract"), "runtime_contract")
    expected_runtime = {
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "model_artifact_schema": HU_M43_V6_ARTIFACT_SCHEMA,
        "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
        "safety_feature_dim": HU_M43_V6_SAFETY_FEATURE_DIM,
        "runtime_teacher_inputs": False,
        "runtime_lcb_gate": False,
        "all_four_runtime_binding_required": True,
    }
    if runtime != expected_runtime:
        raise ValueError("Attempt04 population runtime contract changed")
    attempt04_binding = _mapping(plan.get("attempt04_plan"), "attempt04_plan")
    for label, value in (
        ("Attempt04 plan SHA", attempt04_binding.get("file_sha256")),
    ):
        _sha256(value, label)
    if repo_root is not None:
        root = Path(repo_root).resolve()
        token = attempt04_binding.get("path")
        if not isinstance(token, str) or not token:
            raise ValueError("Attempt04 plan path is missing")
        if file_sha256(root / token) != attempt04_binding.get("file_sha256"):
            raise ValueError("Attempt04 plan file hash changed")

    schedule = (
        _integer(plan.get("paired_seeds_per_opponent"), "paired seeds"),
        _integer(plan.get("seed"), "population seed"),
        _integer(plan.get("seed_stride"), "population seed stride"),
        _integer(plan.get("shards"), "population shards"),
        _integer(plan.get("paired_seeds_per_shard"), "seeds per shard"),
    )
    if schedule != (
        ATTEMPT04_POPULATION_SEEDS,
        ATTEMPT04_POPULATION_SEED,
        ATTEMPT04_POPULATION_SEED_STRIDE,
        ATTEMPT04_POPULATION_SHARDS,
        ATTEMPT04_SEEDS_PER_SHARD,
    ):
        raise ValueError("Attempt04 population seed/power plan changed")
    if (
        plan.get("candidate_records") != 8000
        or plan.get("baseline_records") != 8000
        or plan.get("terminal_trace_hands") != 16000
        or plan.get("second_seat_override_opportunities") != 4000
        or plan.get("minimum_valid_overrides") != 300
        or not _same_number(plan.get("minimum_population_fire_rate_needed"), 0.075)
    ):
        raise ValueError("Attempt04 population sample-size contract changed")
    sizing = _mapping(plan.get("sizing_rule"), "sizing_rule")
    if sizing != {
        "formula": "paired_seeds_required = ceil(minimum_valid_overrides / (opponents * independent_fire_rate_lower_bound))",
        "fixed_before_realized_population_labels": True,
        "optional_stopping_or_posthoc_extension_allowed": False,
        "insufficient_overrides_decision": "complete_no_go",
    }:
        raise ValueError("Attempt04 population sizing contract changed")
    if _mapping(plan.get("evaluation_contract"), "evaluation_contract") != _EVALUATION_CONTRACT:
        raise ValueError("Attempt04 population counterfactual contract changed")
    if _mapping(plan.get("spot_execution"), "spot_execution") != _SPOT_CONTRACT:
        raise ValueError("Attempt04 population Spot contract changed")
    if _mapping(plan.get("activation_guards"), "activation_guards") != _ACTIVATION_GUARDS:
        raise ValueError("Attempt04 population activation guards changed")
    declared_gates = _mapping(plan.get("fixed_acceptance_gates"), "fixed gates")
    if set(declared_gates) != set(FIXED_GATES) or any(
        not _same_number(declared_gates.get(name), expected)
        for name, expected in FIXED_GATES.items()
    ):
        raise ValueError("Attempt04 population gates changed")

    planned = _seed_grid(schedule[1], schedule[2], schedule[0])
    freshness = _mapping(plan.get("freshness"), "freshness")
    if (
        freshness.get("exclude_all_m4_m41_m42_m43_teacher_hand_seeds") is not True
        or freshness.get("exclude_prior_population_smoke_and_frozen_schedules")
        is not True
        or freshness.get("planned_overlap_count_at_freeze") != 0
        or freshness.get("acceptance_validator_rechecks_teacher_seed_overlap")
        is not True
    ):
        raise ValueError("Attempt04 population freshness guards changed")
    prior_count, prior_rows = _validate_excluded_schedules(
        planned,
        freshness.get("excluded_population_schedules"),
        count_field="paired_seeds",
    )
    if _REQUIRED_PRIOR_POPULATION_SCHEDULE not in prior_rows:
        raise ValueError("Attempt04 population excludes no full Attempt03 schedule")
    teacher_count, teacher_rows = _validate_excluded_schedules(
        planned,
        freshness.get("excluded_teacher_schedules"),
        count_field="roots",
        split_field=True,
    )
    attempt04_rows = {
        (split, seed, stride, count)
        for milestone, split, seed, stride, count in teacher_rows
        if milestone == "M4.3-attempt04"
    }
    if not _REQUIRED_ATTEMPT04_TEACHER_SCHEDULES <= attempt04_rows:
        raise ValueError("Attempt04 population teacher exclusions are incomplete")
    rng_count, rng_rows = _validate_excluded_schedules(
        planned,
        freshness.get("excluded_rng_schedules"),
        count_field="roots",
        split_field=True,
    )
    attempt04_rng_rows = {
        (split, seed, stride, count)
        for milestone, split, seed, stride, count in rng_rows
        if milestone == "M4.3-attempt04"
    }
    if not _REQUIRED_ATTEMPT04_RNG_SCHEDULES <= attempt04_rng_rows:
        raise ValueError("Attempt04 population RNG exclusions are incomplete")
    plan["_freshness_counts"] = {
        "population": len(planned),
        "prior_population": prior_count,
        "teacher": teacher_count,
        "teacher_rng": rng_count,
    }
    return plan


def validate_attempt04_locked_lifecycle(
    *,
    marker: Mapping[str, Any],
    receipt: Mapping[str, Any],
    runtime_freeze: Mapping[str, Any],
    runtime_freeze_file_sha256: str,
    population_plan_file_sha256: str,
    final_training_manifest_file_sha256: str,
    threshold_lock_file_sha256: str,
    attempt04_plan_file_sha256: str,
    marker_path: str | Path,
) -> None:
    if marker.get("schema") != M43_ATTEMPT04_LOCKED_MARKER_SCHEMA:
        raise ValueError("Attempt04 locked marker schema mismatch")
    if marker.get("marker_sha256") != self_digest(marker, "marker_sha256"):
        raise ValueError("Attempt04 locked marker self digest mismatch")
    locked = _mapping(runtime_freeze.get("locked200"), "locked200")
    frozen_model = _mapping(runtime_freeze.get("model"), "model")
    if (
        marker.get("status") != "claimed_before_locked200_content_read"
        or marker.get("runtime_freeze_sha256")
        != runtime_freeze.get("freeze_sha256")
        or marker.get("runtime_freeze_file_sha256")
        != _sha256(runtime_freeze_file_sha256, "runtime freeze file SHA")
        or marker.get("population_plan_file_sha256")
        != _sha256(population_plan_file_sha256, "population plan file SHA")
        or marker.get("model_sha256") != frozen_model.get("file_sha256")
        or marker.get("final_training_manifest_file_sha256")
        != _sha256(
            final_training_manifest_file_sha256, "final training manifest SHA"
        )
        or marker.get("threshold_lock_file_sha256")
        != _sha256(threshold_lock_file_sha256, "threshold lock SHA")
        or marker.get("attempt04_plan_file_sha256")
        != _sha256(attempt04_plan_file_sha256, "Attempt04 plan SHA")
        or marker.get("locked200_identity_sha256")
        != locked.get("identity_sha256")
        or marker.get("locked200_records") != 200
        or marker.get("evaluation_pass_count") != 1
        or marker.get("claim_is_consuming_even_on_crash") is not True
    ):
        raise ValueError("Attempt04 locked marker binding changed")
    validate_attempt04_locked_receipt(
        receipt,
        runtime_freeze=runtime_freeze,
        marker=marker,
    )
    marker_source = Path(marker_path).resolve()
    if (
        receipt.get("schema") != M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA
        or receipt.get("population_launch_allowed") is not True
        or receipt.get("status")
        != "go_locked200_population_launch_eligible"
        or receipt.get("promotion_status")
        != "eligible_to_launch_fresh_population"
        or Path(str(receipt.get("consumption_marker_resolved_path", ""))).resolve()
        != marker_source
        or receipt.get("consumption_marker_file_sha256")
        != file_sha256(marker_source)
        or receipt.get("runtime_freeze_file_sha256")
        != runtime_freeze_file_sha256
        or receipt.get("population_plan_file_sha256")
        != population_plan_file_sha256
        or receipt.get("final_training_manifest_file_sha256")
        != final_training_manifest_file_sha256
        or receipt.get("threshold_lock_file_sha256")
        != threshold_lock_file_sha256
        or receipt.get("attempt04_plan_file_sha256")
        != attempt04_plan_file_sha256
        or receipt.get("frozen_threshold")
        != frozen_model.get("safety_threshold")
        or receipt.get("minimum_population_valid_overrides") != 300
    ):
        raise ValueError("Attempt04 locked Go receipt binding changed")


def build_attempt04_population_launch_preflight(
    *,
    model_path: str | Path,
    final_training_manifest_path: str | Path,
    threshold_lock_path: str | Path,
    runtime_freeze_path: str | Path,
    attempt04_plan_path: str | Path,
    locked_receipt_path: str | Path,
    consumption_marker_path: str | Path,
    population_plan_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = {
        "model": Path(model_path).resolve(),
        "training": Path(final_training_manifest_path).resolve(),
        "threshold": Path(threshold_lock_path).resolve(),
        "runtime_freeze": Path(runtime_freeze_path).resolve(),
        "attempt04_plan": Path(attempt04_plan_path).resolve(),
        "locked_receipt": Path(locked_receipt_path).resolve(),
        "marker": Path(consumption_marker_path).resolve(),
        "population_plan": Path(population_plan_path).resolve(),
    }
    if len(set(paths.values())) != len(paths) or not all(
        path.is_file() for path in paths.values()
    ):
        raise ValueError("Attempt04 population preflight inputs are missing or aliased")
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    plan = load_and_validate_attempt04_population_plan(
        paths["population_plan"], repo_root=root
    )
    runtime_freeze = read_json_mapping(paths["runtime_freeze"], "runtime freeze")
    training = read_json_mapping(paths["training"], "final training manifest")
    threshold = read_json_mapping(paths["threshold"], "threshold lock")
    validate_attempt04_runtime_freeze(
        runtime_freeze,
        final_training_manifest=training,
        threshold_lock=threshold,
    )
    source_hashes = validate_attempt04_runtime_artifact_files(
        runtime_freeze,
        model_path=paths["model"],
        final_training_manifest_path=paths["training"],
        threshold_lock_path=paths["threshold"],
        attempt04_plan_path=paths["attempt04_plan"],
        population_plan_path=paths["population_plan"],
        v6_implementation_path=root / "src/ofc_regular/hu_m43_joint_model_v6.py",
    )
    if any(hashes[name] != source_hashes[name] for name in (
        "model", "training", "threshold", "attempt04_plan", "population_plan"
    )):
        raise ValueError("Attempt04 preflight source hash audit changed")
    frozen_model = _mapping(runtime_freeze.get("model"), "freeze.model")
    frozen_training = _mapping(
        runtime_freeze.get("final_training_manifest"), "freeze.training"
    )
    frozen_threshold = _mapping(runtime_freeze.get("threshold_lock"), "freeze.threshold")
    frozen_attempt04 = _mapping(runtime_freeze.get("attempt04_plan"), "freeze.attempt04")
    frozen_population = _mapping(
        runtime_freeze.get("population_plan"), "freeze.population"
    )
    expected_hashes = {
        "model": frozen_model.get("file_sha256"),
        "training": frozen_training.get("file_sha256"),
        "threshold": frozen_threshold.get("file_sha256"),
        "attempt04_plan": frozen_attempt04.get("file_sha256"),
        "population_plan": frozen_population.get("file_sha256"),
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"Attempt04 {name} bytes disagree with runtime freeze")
    plan_attempt04 = _mapping(plan.get("attempt04_plan"), "plan.attempt04_plan")
    if hashes["attempt04_plan"] != plan_attempt04.get("file_sha256"):
        raise ValueError("Attempt04 population plan parent hash changed")
    locked = _mapping(runtime_freeze.get("locked200"), "freeze.locked200")
    marker_token = locked.get("global_consumption_marker")
    if not isinstance(marker_token, str) or not marker_token:
        raise ValueError("Attempt04 canonical marker binding is missing")
    expected_marker = (
        Path(marker_token)
        if Path(marker_token).is_absolute()
        else root / marker_token
    ).resolve()
    if paths["marker"] != expected_marker:
        raise ValueError("Attempt04 marker is not the canonical global marker")
    marker = read_json_mapping(paths["marker"], "locked200 marker")
    locked_receipt = read_json_mapping(paths["locked_receipt"], "locked200 receipt")
    validate_attempt04_locked_lifecycle(
        marker=marker,
        receipt=locked_receipt,
        runtime_freeze=runtime_freeze,
        runtime_freeze_file_sha256=hashes["runtime_freeze"],
        population_plan_file_sha256=hashes["population_plan"],
        final_training_manifest_file_sha256=hashes["training"],
        threshold_lock_file_sha256=hashes["threshold"],
        attempt04_plan_file_sha256=hashes["attempt04_plan"],
        marker_path=paths["marker"],
    )
    model = load_bound_attempt04_v6_model(
        paths["model"],
        expected_sha256=hashes["model"],
        runtime_freeze=runtime_freeze,
        final_training_manifest_path=paths["training"],
        threshold_lock_path=paths["threshold"],
    )
    if not isinstance(model, HuM43JointModelV6):
        raise TypeError("Attempt04 population requires a v6 model")
    counts = plan["_freshness_counts"]
    result: dict[str, Any] = {
        "schema": ATTEMPT04_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass_population_launch_authorized",
        "model_sha256": hashes["model"],
        "final_training_manifest_file_sha256": hashes["training"],
        "threshold_lock_file_sha256": hashes["threshold"],
        "runtime_freeze_file_sha256": hashes["runtime_freeze"],
        "attempt04_plan_file_sha256": hashes["attempt04_plan"],
        "v6_implementation_file_sha256": source_hashes["implementation"],
        "locked_receipt_file_sha256": hashes["locked_receipt"],
        "consumption_marker_file_sha256": hashes["marker"],
        "population_plan_file_sha256": hashes["population_plan"],
        "runtime_freeze_sha256": runtime_freeze["freeze_sha256"],
        "locked_receipt_sha256": locked_receipt["receipt_sha256"],
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "model_schema": model.schema,
        "model_id": model.model_id,
        "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
        "action_score_mode": model.action_score_mode,
        "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
        "safety_feature_dim": HU_M43_V6_SAFETY_FEATURE_DIM,
        "frozen_threshold": float(model.safety_threshold),
        "population_hand_seeds_checked": counts["population"],
        "teacher_schedule_seeds_checked": counts["teacher"],
        "teacher_rng_seeds_checked": counts["teacher_rng"],
        "prior_population_seeds_checked": counts["prior_population"],
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "canonical_global_marker_verified": True,
        "locked200_go_required_and_verified": True,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    result["preflight_sha256"] = self_digest(result, "preflight_sha256")
    return result


def evaluate_attempt04_population_gates(
    evaluation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if evaluation.get("opponents") != list(EXPECTED_OPPONENTS):
        raise ValueError("Attempt04 population opponent order changed")
    return evaluate_attempt02_population_gates(evaluation)


def validate_attempt04_population_acceptance(
    *,
    evaluation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    population_plan: Mapping[str, Any],
    merge_manifest: Mapping[str, Any],
    spot_receipt: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    lifecycle_preflight: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_and_validate_attempt04_population_plan(population_plan)
    plan.pop("_freshness_counts", None)
    required_hashes = (
        "population_plan",
        "records",
        "evaluation",
        "merge_manifest",
        "spot_receipt",
        "run_manifest",
        "lifecycle_preflight",
    )
    hashes = {
        name: _sha256(source_hashes.get(name), f"{name} SHA-256")
        for name in required_hashes
    }
    plan_sha = hashes["population_plan"]
    if lifecycle_preflight.get("population_plan_file_sha256") != plan_sha:
        raise ValueError("Attempt04 preflight/population plan hash mismatch")
    if lifecycle_preflight.get("preflight_sha256") != self_digest(
        lifecycle_preflight, "preflight_sha256"
    ):
        raise ValueError("Attempt04 population preflight self digest mismatch")
    recomputed = summarize_hu_m4_population_records(
        records,
        opponents=EXPECTED_OPPONENTS,
        paired_seeds=ATTEMPT04_POPULATION_SEEDS,
        seed=ATTEMPT04_POPULATION_SEED,
        seed_stride=ATTEMPT04_POPULATION_SEED_STRIDE,
    )
    content_match = _canonical_summary(evaluation) == _canonical_summary(recomputed)
    runtime = _mapping(evaluation.get("runtime_config"), "runtime_config")
    runtime_match = (
        runtime.get("candidate_model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and runtime.get("safety_model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and runtime.get("model_schema") == HU_M43_V6_MODEL_SCHEMA
        and runtime.get("model_id") == lifecycle_preflight.get("model_id")
        and runtime.get("action_score_mode") == HU_M43_V6_ACTION_SCORE_MODE
        and runtime.get("runtime_binding_verified") is True
        and runtime.get("freeze_manifest_sha256")
        == lifecycle_preflight.get("runtime_freeze_file_sha256")
        and runtime.get("training_manifest_sha256")
        == lifecycle_preflight.get("final_training_manifest_file_sha256")
        and runtime.get("threshold_lock_sha256")
        == lifecycle_preflight.get("threshold_lock_file_sha256")
        and runtime.get("population_plan_sha256") == plan_sha
        and runtime.get("sharded_evaluation") is True
        and runtime.get("shard_count") == ATTEMPT04_POPULATION_SHARDS
        and runtime.get("final_metrics_recomputed_from_merged_records") is True
        and runtime.get("current_profile_used") is False
        and runtime.get("promotion_artifact_contract") is True
        and runtime.get("diagnostic_legacy") is False
        and runtime.get("safety_enabled") is True
        and _same_number(
            runtime.get("safety_threshold"),
            lifecycle_preflight.get("frozen_threshold"),
        )
        and all(row.get("runtime_binding_verified") is True for row in records)
    )
    lifecycle_match = (
        lifecycle_preflight.get("schema")
        == ATTEMPT04_POPULATION_PREFLIGHT_SCHEMA
        and lifecycle_preflight.get("status")
        == "pass_population_launch_authorized"
        and lifecycle_preflight.get("teacher_overlap_count") == 0
        and lifecycle_preflight.get("prior_population_overlap_count") == 0
        and lifecycle_preflight.get("canonical_global_marker_verified") is True
        and lifecycle_preflight.get("locked200_go_required_and_verified") is True
        and lifecycle_preflight.get("teacher_calibration_locked_content_packaged")
        is False
        and lifecycle_preflight.get("current_profile_mutated") is False
        and lifecycle_preflight.get("no_runtime_activation") is True
    )
    shard_match = _validate_transport_shards(
        plan=plan,
        merge_manifest=merge_manifest,
        spot_receipt=spot_receipt,
    )
    merge_match = (
        merge_manifest.get("schema") == M4_POPULATION_MERGE_SCHEMA
        and merge_manifest.get("status") == "complete_content_verified"
        and merge_manifest.get("population_plan_sha256") == plan_sha
        and merge_manifest.get("merged_records_sha256") == hashes["records"]
        and merge_manifest.get("evaluation_sha256") == hashes["evaluation"]
        and merge_manifest.get("paired_seeds_per_opponent")
        == ATTEMPT04_POPULATION_SEEDS
        and merge_manifest.get("opponents") == list(EXPECTED_OPPONENTS)
        and merge_manifest.get("merged_records") == 8000
        and merge_manifest.get("current_profile_used") is False
        and merge_manifest.get("metrics_recomputed_from_merged_seed_clusters")
        is True
        and shard_match
    )
    manifest_runtime = _mapping(run_manifest.get("runtime"), "run runtime")
    manifest_plan = _mapping(run_manifest.get("population_plan"), "run plan")
    manifest_shards = _mapping(run_manifest.get("shards"), "run shards")
    manifest_checkpoint = _mapping(run_manifest.get("checkpoint"), "checkpoint")
    manifest_compute = _mapping(run_manifest.get("compute"), "compute")
    source_boundary = _mapping(run_manifest.get("source_boundary"), "source boundary")
    run_match = (
        run_manifest.get("schema") == ATTEMPT04_POPULATION_MANIFEST_SCHEMA
        and run_manifest.get("status") == "spot_population_complete"
        and manifest_plan.get("sha256") == plan_sha
        and manifest_plan.get("paired_seeds") == ATTEMPT04_POPULATION_SEEDS
        and manifest_plan.get("seed") == ATTEMPT04_POPULATION_SEED
        and manifest_plan.get("seed_stride") == ATTEMPT04_POPULATION_SEED_STRIDE
        and manifest_plan.get("shards") == ATTEMPT04_POPULATION_SHARDS
        and manifest_plan.get("paired_seeds_per_shard")
        == ATTEMPT04_SEEDS_PER_SHARD
        and manifest_shards.get("count") == ATTEMPT04_POPULATION_SHARDS
        and _is_sha256(manifest_shards.get("sha256"))
        and manifest_checkpoint
        == {
            "unit": "completed_shard",
            "retry": "deterministic_full_shard",
            "resume_missing_shards_only": True,
            "done_commit_last": True,
        }
        and manifest_compute.get("provisioning_model") == "SPOT"
        and manifest_compute.get("instance_termination_action") == "DELETE"
        and manifest_runtime.get("model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and manifest_runtime.get("training_manifest_sha256")
        == lifecycle_preflight.get("final_training_manifest_file_sha256")
        and manifest_runtime.get("threshold_lock_sha256")
        == lifecycle_preflight.get("threshold_lock_file_sha256")
        and manifest_runtime.get("freeze_manifest_sha256")
        == lifecycle_preflight.get("runtime_freeze_file_sha256")
        and manifest_runtime.get("locked_receipt_sha256")
        == lifecycle_preflight.get("locked_receipt_file_sha256")
        and manifest_runtime.get("consumption_marker_sha256")
        == lifecycle_preflight.get("consumption_marker_file_sha256")
        and manifest_runtime.get("model_schema") == HU_M43_V6_MODEL_SCHEMA
        and manifest_runtime.get("model_id") == lifecycle_preflight.get("model_id")
        and manifest_runtime.get("action_score_mode")
        == HU_M43_V6_ACTION_SCORE_MODE
        and manifest_runtime.get("runtime_teacher_inputs") is False
        and manifest_runtime.get("current_profile_used") is False
        and _same_number(
            manifest_runtime.get("frozen_threshold"),
            lifecycle_preflight.get("frozen_threshold"),
        )
        and run_manifest.get("launch_preflight_sha256")
        == hashes["lifecycle_preflight"]
        and source_boundary
        == {
            "teacher_jsonl_packaged": False,
            "calibration_jsonl_packaged": False,
            "locked_jsonl_packaged": False,
            "teacher_valued_receipts_packaged": False,
            "current_profile_artifact_packaged": False,
            "runtime_artifacts_only": True,
        }
        and run_manifest.get("no_runtime_activation") is True
        and run_manifest.get("current_profile_mutated") is False
    )
    receipt_match = (
        spot_receipt.get("schema") == ATTEMPT04_POPULATION_RECEIPT_SCHEMA
        and spot_receipt.get("status") == "verified_and_merged"
        and spot_receipt.get("run_manifest_sha256") == hashes["run_manifest"]
        and spot_receipt.get("population_plan_sha256") == plan_sha
        and spot_receipt.get("model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and spot_receipt.get("model_schema") == HU_M43_V6_MODEL_SCHEMA
        and spot_receipt.get("action_score_mode") == HU_M43_V6_ACTION_SCORE_MODE
        and spot_receipt.get("evaluation_sha256") == hashes["evaluation"]
        and spot_receipt.get("records_sha256") == hashes["records"]
        and spot_receipt.get("merge_manifest_sha256") == hashes["merge_manifest"]
        and spot_receipt.get("valid_overrides")
        == recomputed["population"]["all_seats"]["overrides"]
        and spot_receipt.get("paired_seeds_per_opponent")
        == ATTEMPT04_POPULATION_SEEDS
        and spot_receipt.get("invalid_counterfactuals")
        == recomputed["invalid_counterfactuals"]
        and spot_receipt.get("nonfire_cancellation_mismatches")
        == recomputed["nonfire_cancellation_mismatches"]
        and spot_receipt.get("teacher_calibration_locked_content_received")
        is False
        and spot_receipt.get("current_profile_mutated") is False
        and spot_receipt.get("no_runtime_activation") is True
        and shard_match
    )
    gates = [
        _gate(
            "attempt04_lifecycle_hash_chain",
            lifecycle_match and merge_match and run_match and receipt_match,
            {
                "lifecycle": lifecycle_match,
                "merge_manifest": merge_match,
                "run_manifest": run_match,
                "spot_receipt": receipt_match,
            },
            "bound v6 lifecycle, Spot transport, merge, and receipt all pass",
        ),
        _gate(
            "population_content_recomputed",
            content_match,
            content_match,
            "stored summary equals played-hand JSONL recomputation",
        ),
        _gate(
            "v6_all_four_runtime_binding",
            runtime_match,
            dict(runtime),
            "model, runtime freeze, final manifest, threshold lock bound; current unused",
        ),
        *evaluate_attempt04_population_gates(recomputed),
    ]
    passed = sum(gate["passed"] is True for gate in gates)
    decision = "complete_go" if passed == len(gates) else "complete_no_go"
    config = {
        "schema": ATTEMPT04_POPULATION_ACCEPTANCE_CONFIG_SCHEMA,
        "status": "frozen_evaluated_once",
        "policy_attempt": "attempt04_v6",
        "population_plan_sha256": plan_sha,
        "run_manifest_sha256": hashes["run_manifest"],
        "evaluation_sha256": hashes["evaluation"],
        "records_sha256": hashes["records"],
        "merge_manifest_sha256": hashes["merge_manifest"],
        "model_sha256": lifecycle_preflight.get("model_sha256"),
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "frozen_threshold": lifecycle_preflight.get("frozen_threshold"),
        "fixed_acceptance_gates": dict(FIXED_GATES),
        "threshold_reselection_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    status = {
        "schema": ATTEMPT04_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
        "status": decision,
        "passed_gates": passed,
        "total_gates": len(gates),
        "gates": gates,
        "promotion_eligible": decision == "complete_go",
        "population_plan_sha256": plan_sha,
        "run_manifest_sha256": hashes["run_manifest"],
        "evaluation_sha256": hashes["evaluation"],
        "records_sha256": hashes["records"],
        "merge_manifest_sha256": hashes["merge_manifest"],
        "teacher_values_reported_as_realized_match_ev": False,
        "threshold_reselection_performed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    return config, status


def _validate_transport_shards(
    *,
    plan: Mapping[str, Any],
    merge_manifest: Mapping[str, Any],
    spot_receipt: Mapping[str, Any],
) -> bool:
    merge_shards = merge_manifest.get("shards")
    receipt_shards = spot_receipt.get("shards")
    if (
        not isinstance(merge_shards, Sequence)
        or isinstance(merge_shards, (str, bytes))
        or not isinstance(receipt_shards, Sequence)
        or isinstance(receipt_shards, (str, bytes))
        or len(merge_shards) != ATTEMPT04_POPULATION_SHARDS
        or len(receipt_shards) != ATTEMPT04_POPULATION_SHARDS
    ):
        return False
    merged_by_offset = {
        row.get("offset"): row for row in merge_shards if isinstance(row, Mapping)
    }
    received_by_shard = {
        row.get("shard"): row for row in receipt_shards if isinstance(row, Mapping)
    }
    if (
        len(merged_by_offset) != ATTEMPT04_POPULATION_SHARDS
        or len(received_by_shard) != ATTEMPT04_POPULATION_SHARDS
    ):
        return False
    for shard in range(ATTEMPT04_POPULATION_SHARDS):
        offset = shard * ATTEMPT04_SEEDS_PER_SHARD
        merged = merged_by_offset.get(offset)
        received = received_by_shard.get(shard)
        expected_seed = ATTEMPT04_POPULATION_SEED + offset * ATTEMPT04_POPULATION_SEED_STRIDE
        if (
            not isinstance(merged, Mapping)
            or not isinstance(received, Mapping)
            or merged.get("seed") != expected_seed
            or merged.get("paired_seeds") != ATTEMPT04_SEEDS_PER_SHARD
            or merged.get("records") != 400
            or merged.get("evaluation_sha256")
            != received.get("evaluation_sha256")
            or merged.get("records_sha256") != received.get("records_sha256")
            or not _is_sha256(received.get("done_sha256"))
        ):
            return False
    return True


def _validate_excluded_schedules(
    planned: set[int],
    value: Any,
    *,
    count_field: str,
    split_field: bool = False,
) -> tuple[int, set[Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("Attempt04 population exclusion schedules are missing")
    result: set[int] = set()
    identities: set[Any] = set()
    for row in value:
        item = _mapping(row, "excluded schedule")
        milestone = item.get("milestone")
        if not isinstance(milestone, str) or not milestone:
            raise ValueError("Attempt04 excluded schedule milestone is missing")
        seed = _integer(item.get("seed"), "excluded schedule seed")
        stride = _integer(item.get("seed_stride"), "excluded schedule stride")
        count = _integer(item.get(count_field), f"excluded schedule {count_field}")
        schedule = _seed_grid(seed, stride, count)
        if result & schedule:
            raise ValueError("Attempt04 population exclusion schedule duplicated")
        result.update(schedule)
        if split_field:
            split = item.get("split")
            if not isinstance(split, str) or not split:
                raise ValueError("Attempt04 excluded teacher split is missing")
            identities.add((milestone, split, seed, stride, count))
        else:
            identities.add((milestone, seed, stride, count))
    if planned & result:
        raise ValueError("Attempt04 population seed overlaps an excluded schedule")
    return len(result), identities


def _seed_grid(seed: int, stride: int, count: int) -> set[int]:
    return {seed + index * stride for index in range(count)}


def _canonical_summary(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    normalized.pop("runtime_config", None)
    normalized.pop("shard_merge", None)
    normalized.pop("records", None)
    normalized["elapsed_seconds"] = None
    normalized["records_output"] = None
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"population record {line_number} is not an object")
            result.append(row)
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _is_sha256(value: Any) -> bool:
    try:
        _sha256(value, "SHA-256")
    except ValueError:
        return False
    return True


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--final-training-manifest", type=Path, required=True)
    preflight.add_argument("--threshold-lock", type=Path, required=True)
    preflight.add_argument("--runtime-freeze", type=Path, required=True)
    preflight.add_argument("--attempt04-plan", type=Path, required=True)
    preflight.add_argument("--locked-receipt", type=Path, required=True)
    preflight.add_argument("--consumption-marker", type=Path, required=True)
    preflight.add_argument("--population-plan", type=Path, required=True)
    preflight.add_argument("--repo-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    accept = subparsers.add_parser("accept")
    for name in (
        "evaluation",
        "records",
        "population-plan",
        "merge-manifest",
        "spot-receipt",
        "run-manifest",
        "lifecycle-preflight",
        "config-output",
        "status-output",
    ):
        accept.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        result = build_attempt04_population_launch_preflight(
            model_path=args.model,
            final_training_manifest_path=args.final_training_manifest,
            threshold_lock_path=args.threshold_lock,
            runtime_freeze_path=args.runtime_freeze,
            attempt04_plan_path=args.attempt04_plan,
            locked_receipt_path=args.locked_receipt,
            consumption_marker_path=args.consumption_marker,
            population_plan_path=args.population_plan,
            repo_root=args.repo_root,
        )
        if args.output is not None:
            _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    paths = {
        "evaluation": args.evaluation,
        "records": args.records,
        "population_plan": args.population_plan,
        "merge_manifest": args.merge_manifest,
        "spot_receipt": args.spot_receipt,
        "run_manifest": args.run_manifest,
        "lifecycle_preflight": args.lifecycle_preflight,
    }
    payloads = {
        name: read_json_mapping(path, name)
        for name, path in paths.items()
        if name != "records"
    }
    records = _read_jsonl(paths["records"])
    config, status = validate_attempt04_population_acceptance(
        evaluation=payloads["evaluation"],
        records=records,
        population_plan=payloads["population_plan"],
        merge_manifest=payloads["merge_manifest"],
        spot_receipt=payloads["spot_receipt"],
        run_manifest=payloads["run_manifest"],
        lifecycle_preflight=payloads["lifecycle_preflight"],
        source_hashes={name: file_sha256(path) for name, path in paths.items()},
    )
    _write_json(args.config_output, config)
    _write_json(args.status_output, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status["status"] == "complete_go" else 2


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT04_POPULATION_ACCEPTANCE_CONFIG_SCHEMA",
    "ATTEMPT04_POPULATION_ACCEPTANCE_STATUS_SCHEMA",
    "ATTEMPT04_POPULATION_MANIFEST_SCHEMA",
    "ATTEMPT04_POPULATION_PLAN_SCHEMA",
    "ATTEMPT04_POPULATION_PREFLIGHT_SCHEMA",
    "ATTEMPT04_POPULATION_RECEIPT_SCHEMA",
    "ATTEMPT04_POPULATION_SEED",
    "ATTEMPT04_POPULATION_SEED_STRIDE",
    "build_attempt04_population_launch_preflight",
    "evaluate_attempt04_population_gates",
    "load_and_validate_attempt04_population_plan",
    "validate_attempt04_locked_lifecycle",
    "validate_attempt04_population_acceptance",
]
