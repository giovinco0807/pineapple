"""Fail-closed Attempt03 v5 population preflight and realized-EV gates.

No teacher, calibration, or locked-holdout JSONL is accepted by this module.
The launch preflight consumes only the frozen v5 runtime artifact and immutable
lifecycle receipts.  Final acceptance recomputes the paired seat-swap summary
from played-hand records and applies the same fixed four-opponent gates as the
frozen Attempt02 population contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_hu_m4_population import (
    M4_OPPONENT_PROFILES,
    M4_POPULATION_EVALUATION_SCHEMA,
    summarize_hu_m4_population_records,
)
from .hu_m43_attempt03_runtime import (
    M43_ATTEMPT03_LOCKED_MARKER_SCHEMA,
    M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT03_RUNTIME_FREEZE_SCHEMA,
    file_sha256,
    load_bound_attempt03_v5_model,
    read_json_mapping,
    self_digest,
    validate_attempt03_precalibration_receipt,
    validate_attempt03_runtime_freeze,
)
from .hu_m43_joint_model_v5 import (
    HU_M43_V5_ACTION_SCORE_MODE,
    HU_M43_V5_ARTIFACT_SCHEMA,
    HU_M43_V5_MODEL_SCHEMA,
    HU_M43_V5_PROPOSAL_SCHEMA,
    HuM43JointModelV5,
)
from .hu_m43_attempt03_training import load_attempt03_training_freeze
from .validate_hu_m43_attempt02_acceptance import (
    FIXED_GATES,
    evaluate_attempt02_population_gates,
)


ATTEMPT03_POPULATION_PLAN_SCHEMA = "hu_m43_population_acceptance_plan_v1"
ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA = (
    "hu_m43_attempt03_population_launch_preflight_v1"
)
ATTEMPT03_POPULATION_MANIFEST_SCHEMA = (
    "hu_m43_attempt03_population_spot_manifest_v1"
)
ATTEMPT03_POPULATION_RECEIPT_SCHEMA = (
    "hu_m43_attempt03_population_spot_receipt_v1"
)
ATTEMPT03_POPULATION_ACCEPTANCE_CONFIG_SCHEMA = (
    "hu_m43_attempt03_population_acceptance_config_v1"
)
ATTEMPT03_POPULATION_ACCEPTANCE_STATUS_SCHEMA = (
    "hu_m43_attempt03_population_acceptance_status_v1"
)
ATTEMPT03_POPULATION_SEED = 12_106_071_901
ATTEMPT03_POPULATION_SEED_STRIDE = 1_000_003
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


def load_and_validate_attempt03_population_plan(
    source: str | Path | Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    plan = (
        dict(source)
        if isinstance(source, Mapping)
        else read_json_mapping(source, "Attempt03 population plan")
    )
    if (
        plan.get("schema") != ATTEMPT03_POPULATION_PLAN_SCHEMA
        or plan.get("status") != "frozen_before_population_evaluation"
        or plan.get("policy_attempt") != "attempt03_v5"
        or plan.get("fixed_baseline_profile") != "stage18_p1"
        or plan.get("opponents") != list(EXPECTED_OPPONENTS)
        or plan.get("paired_seat_swap") is not True
    ):
        raise ValueError("Attempt03 population plan identity changed")
    runtime = _mapping(plan.get("runtime_contract"), "runtime_contract")
    if runtime != {
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "model_artifact_schema": HU_M43_V5_ARTIFACT_SCHEMA,
        "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
        "action_score_mode": HU_M43_V5_ACTION_SCORE_MODE,
        "runtime_teacher_inputs": False,
        "runtime_lcb_gate": False,
    }:
        raise ValueError("Attempt03 population runtime contract changed")
    parent = _mapping(plan.get("attempt03_plan"), "attempt03_plan")
    prereg = _mapping(
        plan.get("attempt03_model_freeze"), "attempt03_model_freeze"
    )
    for label, value in (
        ("Attempt03 plan SHA", parent.get("file_sha256")),
        ("Attempt03 model-freeze SHA", prereg.get("file_sha256")),
    ):
        _sha256(value, label)
    if repo_root is not None:
        root = Path(repo_root).resolve()
        for label, binding in (("Attempt03 plan", parent), ("model freeze", prereg)):
            token = binding.get("path")
            if not isinstance(token, str) or not token:
                raise ValueError(f"{label} path is missing")
            if file_sha256(root / token) != binding.get("file_sha256"):
                raise ValueError(f"{label} file hash changed")

    schedule = (
        _integer(plan.get("paired_seeds_per_opponent"), "paired seeds"),
        _integer(plan.get("seed"), "population seed"),
        _integer(plan.get("seed_stride"), "population seed stride"),
        _integer(plan.get("shards"), "population shards"),
        _integer(plan.get("paired_seeds_per_shard"), "seeds per shard"),
    )
    if schedule != (
        1000,
        ATTEMPT03_POPULATION_SEED,
        ATTEMPT03_POPULATION_SEED_STRIDE,
        20,
        50,
    ):
        raise ValueError("Attempt03 population seed/power plan changed")
    if (
        plan.get("candidate_records") != 8000
        or plan.get("baseline_records") != 8000
        or plan.get("terminal_trace_hands") != 16000
        or plan.get("second_seat_override_opportunities") != 4000
        or plan.get("minimum_valid_overrides") != 300
        or not _same_number(plan.get("minimum_population_fire_rate_needed"), 0.075)
    ):
        raise ValueError("Attempt03 population sample-size contract changed")
    sizing = _mapping(plan.get("sizing_rule"), "sizing_rule")
    if sizing != {
        "formula": "paired_seeds_required = ceil(minimum_valid_overrides / (opponents * independent_fire_rate_lower_bound))",
        "fixed_before_realized_population_labels": True,
        "optional_stopping_or_posthoc_extension_allowed": False,
        "insufficient_overrides_decision": "complete_no_go",
    }:
        raise ValueError("Attempt03 population sizing contract changed")
    if _mapping(plan.get("evaluation_contract"), "evaluation_contract") != _EVALUATION_CONTRACT:
        raise ValueError("Attempt03 population counterfactual contract changed")
    if _mapping(plan.get("spot_execution"), "spot_execution") != _SPOT_CONTRACT:
        raise ValueError("Attempt03 population Spot contract changed")
    if _mapping(plan.get("activation_guards"), "activation_guards") != _ACTIVATION_GUARDS:
        raise ValueError("Attempt03 population activation guards changed")
    declared_gates = _mapping(plan.get("fixed_acceptance_gates"), "fixed gates")
    if set(declared_gates) != set(FIXED_GATES) or any(
        not _same_number(declared_gates.get(name), expected)
        for name, expected in FIXED_GATES.items()
    ):
        raise ValueError("Attempt03 population gates changed")

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
        raise ValueError("Attempt03 population freshness guards changed")
    prior_count = _validate_excluded_schedules(
        planned,
        freshness.get("excluded_population_schedules"),
        count_field="paired_seeds",
        required_identities={
            ("M4.3-attempt02-frozen", 6_106_071_901, 1_000_003, 1000),
        },
    )
    teacher_count = _validate_excluded_schedules(
        planned,
        freshness.get("excluded_teacher_schedules"),
        count_field="roots",
        split_field=True,
        required_identities={
            ("M4.3-attempt03", "train.fit", 9_106_071_901, 1_000_003, 500),
            (
                "M4.3-attempt03",
                "train.precal_holdout",
                9_506_071_901,
                1_000_003,
                200,
            ),
        },
    )
    plan["_freshness_counts"] = {
        "population": len(planned),
        "prior_population": prior_count,
        "teacher": teacher_count,
    }
    return plan


def validate_attempt03_locked_lifecycle(
    *,
    marker: Mapping[str, Any],
    receipt: Mapping[str, Any],
    runtime_freeze: Mapping[str, Any],
    runtime_freeze_file_sha256: str,
    population_plan_file_sha256: str,
    model_freeze_file_sha256: str,
    training_freeze_file_sha256: str,
    precalibration_receipt_file_sha256: str,
    marker_path: str | Path,
) -> None:
    """Validate the hash-only one-shot receipt expected before population."""

    if marker.get("schema") != M43_ATTEMPT03_LOCKED_MARKER_SCHEMA:
        raise ValueError("Attempt03 locked marker schema mismatch")
    if marker.get("marker_sha256") != self_digest(marker, "marker_sha256"):
        raise ValueError("Attempt03 locked marker self digest mismatch")
    locked = _mapping(runtime_freeze.get("inherited_locked"), "inherited_locked")
    frozen_model = _mapping(runtime_freeze.get("model"), "model")
    if (
        marker.get("status")
        != "claimed_before_inherited_locked_content_read"
        or marker.get("runtime_freeze_sha256")
        != runtime_freeze.get("freeze_sha256")
        or marker.get("runtime_freeze_file_sha256")
        != _sha256(runtime_freeze_file_sha256, "runtime freeze file SHA")
        or marker.get("population_plan_file_sha256")
        != _sha256(population_plan_file_sha256, "population plan file SHA")
        or marker.get("model_sha256") != frozen_model.get("file_sha256")
        or marker.get("model_freeze_file_sha256")
        != _sha256(model_freeze_file_sha256, "model freeze file SHA")
        or marker.get("training_freeze_file_sha256")
        != _sha256(training_freeze_file_sha256, "training freeze file SHA")
        or marker.get("precalibration_receipt_file_sha256")
        != _sha256(
            precalibration_receipt_file_sha256,
            "pre-calibration receipt file SHA",
        )
        or marker.get("locked_identity_sha256") != locked.get("identity_sha256")
        or marker.get("evaluation_pass_count") != 1
        or marker.get("claim_is_consuming_even_on_crash") is not True
    ):
        raise ValueError("Attempt03 locked marker binding changed")

    if receipt.get("schema") != M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA:
        raise ValueError("Attempt03 locked receipt schema mismatch")
    if receipt.get("receipt_sha256") != self_digest(receipt, "receipt_sha256"):
        raise ValueError("Attempt03 locked receipt self digest mismatch")
    marker_source = Path(marker_path).resolve()
    if (
        receipt.get("status")
        != "evaluated_once_diagnostic_only_no_activation"
        or receipt.get("runtime_freeze_sha256")
        != runtime_freeze.get("freeze_sha256")
        or receipt.get("runtime_freeze_file_sha256")
        != runtime_freeze_file_sha256
        or receipt.get("population_plan_file_sha256")
        != population_plan_file_sha256
        or receipt.get("model_sha256") != frozen_model.get("file_sha256")
        or receipt.get("model_freeze_file_sha256")
        != model_freeze_file_sha256
        or receipt.get("training_freeze_file_sha256")
        != training_freeze_file_sha256
        or receipt.get("precalibration_receipt_file_sha256")
        != precalibration_receipt_file_sha256
        or receipt.get("frozen_threshold")
        != frozen_model.get("safety_threshold")
        or receipt.get("locked_identity_sha256") != locked.get("identity_sha256")
        or Path(str(receipt.get("consumption_marker_resolved_path", ""))).resolve()
        != marker_source
        or receipt.get("consumption_marker_file_sha256")
        != file_sha256(marker_source)
        or receipt.get("consumption_marker_canonical_sha256")
        != marker.get("marker_sha256")
        or receipt.get("evaluation_pass_count") != 1
        or receipt.get("requires_fresh_population_acceptance") is not True
        or receipt.get("minimum_population_valid_overrides") != 300
    ):
        raise ValueError("Attempt03 locked receipt binding changed")
    false_fields = (
        "threshold_search_performed",
        "threshold_reselection_performed",
        "model_selection_performed",
        "feature_selection_performed",
        "current_profile_resolved",
        "current_profile_changed",
        "runtime_policy_activated",
        "policy_promoted",
    )
    if any(receipt.get(field) is not False for field in false_fields):
        raise ValueError("Attempt03 locked receipt contains an unsafe flag")


def build_attempt03_population_launch_preflight(
    *,
    model_path: str | Path,
    final_training_manifest_path: str | Path,
    runtime_freeze_path: str | Path,
    training_freeze_path: str | Path,
    precalibration_receipt_path: str | Path,
    model_freeze_path: str | Path,
    attempt03_plan_path: str | Path,
    locked_receipt_path: str | Path,
    consumption_marker_path: str | Path,
    population_plan_path: str | Path,
) -> dict[str, Any]:
    paths = {
        "model": Path(model_path).resolve(),
        "training": Path(final_training_manifest_path).resolve(),
        "runtime_freeze": Path(runtime_freeze_path).resolve(),
        "training_freeze": Path(training_freeze_path).resolve(),
        "precalibration": Path(precalibration_receipt_path).resolve(),
        "model_freeze": Path(model_freeze_path).resolve(),
        "attempt03_plan": Path(attempt03_plan_path).resolve(),
        "locked_receipt": Path(locked_receipt_path).resolve(),
        "marker": Path(consumption_marker_path).resolve(),
        "population_plan": Path(population_plan_path).resolve(),
    }
    if len(set(paths.values())) != len(paths) or not all(
        path.is_file() for path in paths.values()
    ):
        raise ValueError("Attempt03 population lifecycle inputs are missing or aliased")
    root = paths["attempt03_plan"].parents[1]
    plan = load_and_validate_attempt03_population_plan(
        paths["population_plan"], repo_root=root
    )
    runtime_freeze = read_json_mapping(paths["runtime_freeze"], "runtime freeze")
    training = read_json_mapping(paths["training"], "final training manifest")
    validate_attempt03_runtime_freeze(
        runtime_freeze, final_training_manifest=training
    )
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    frozen_model = _mapping(runtime_freeze.get("model"), "runtime freeze model")
    frozen_training = _mapping(
        runtime_freeze.get("training_manifest"), "runtime freeze training"
    )
    frozen_executable = _mapping(
        runtime_freeze.get("executable_model_freeze"),
        "runtime freeze executable model freeze",
    )
    frozen_training_pipeline = _mapping(
        runtime_freeze.get("training_pipeline_freeze"),
        "runtime freeze training pipeline freeze",
    )
    frozen_plan = _mapping(
        runtime_freeze.get("attempt03_plan"), "runtime freeze Attempt03 plan"
    )
    expected_hashes = {
        "model": frozen_model.get("file_sha256"),
        "training": frozen_training.get("file_sha256"),
        "model_freeze": frozen_executable.get("file_sha256"),
        "training_freeze": frozen_training_pipeline.get("file_sha256"),
        "attempt03_plan": frozen_plan.get("file_sha256"),
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"Attempt03 {name} bytes disagree with runtime freeze")
    load_attempt03_training_freeze(
        paths["training_freeze"],
        repo_root=root,
        expected_model_freeze_path=paths["model_freeze"],
    )
    if training.get("training_freeze_file_sha256") != hashes["training_freeze"]:
        raise ValueError("Attempt03 final manifest/training freeze hash changed")
    precalibration = read_json_mapping(
        paths["precalibration"], "pre-calibration receipt"
    )
    validate_attempt03_precalibration_receipt(precalibration)
    if precalibration.get("receipt_sha256") != training.get(
        "precalibration_receipt_sha256"
    ):
        raise ValueError("Attempt03 pre-calibration receipt binding changed")
    prereg_binding = _mapping(plan.get("attempt03_model_freeze"), "model freeze")
    if hashes["model_freeze"] != prereg_binding.get("file_sha256"):
        raise ValueError("Attempt03 population plan/pre-registration hash changed")
    locked = _mapping(runtime_freeze.get("inherited_locked"), "inherited_locked")
    expected_marker = Path(
        root / str(locked.get("global_consumption_marker", ""))
    ).resolve()
    if paths["marker"] != expected_marker:
        raise ValueError("Attempt03 marker is not the canonical global marker")
    marker = read_json_mapping(paths["marker"], "locked marker")
    locked_receipt = read_json_mapping(paths["locked_receipt"], "locked receipt")
    validate_attempt03_locked_lifecycle(
        marker=marker,
        receipt=locked_receipt,
        runtime_freeze=runtime_freeze,
        runtime_freeze_file_sha256=hashes["runtime_freeze"],
        population_plan_file_sha256=hashes["population_plan"],
        model_freeze_file_sha256=hashes["model_freeze"],
        training_freeze_file_sha256=hashes["training_freeze"],
        precalibration_receipt_file_sha256=hashes["precalibration"],
        marker_path=paths["marker"],
    )
    model = load_bound_attempt03_v5_model(
        paths["model"],
        expected_sha256=hashes["model"],
        runtime_freeze=runtime_freeze,
        final_training_manifest_path=paths["training"],
    )
    if not isinstance(model, HuM43JointModelV5):
        raise TypeError("Attempt03 population requires a v5 model")
    counts = plan["_freshness_counts"]
    return {
        "schema": ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "model_sha256": hashes["model"],
        "final_training_manifest_file_sha256": hashes["training"],
        "runtime_freeze_file_sha256": hashes["runtime_freeze"],
        "training_freeze_file_sha256": hashes["training_freeze"],
        "precalibration_receipt_file_sha256": hashes["precalibration"],
        "model_freeze_file_sha256": hashes["model_freeze"],
        "attempt03_plan_file_sha256": hashes["attempt03_plan"],
        "locked_receipt_file_sha256": hashes["locked_receipt"],
        "consumption_marker_file_sha256": hashes["marker"],
        "population_plan_file_sha256": hashes["population_plan"],
        "runtime_freeze_sha256": runtime_freeze["freeze_sha256"],
        "locked_receipt_sha256": locked_receipt["receipt_sha256"],
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "model_schema": model.schema,
        "model_id": model.model_id,
        "action_score_mode": model.action_score_mode,
        "frozen_threshold": float(model.safety_threshold),
        "population_hand_seeds_checked": counts["population"],
        "teacher_schedule_seeds_checked": counts["teacher"],
        "prior_population_seeds_checked": counts["prior_population"],
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "canonical_global_marker_verified": True,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }


def evaluate_attempt03_population_gates(
    evaluation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply the frozen realized-EV gates; no threshold search is performed."""

    if evaluation.get("opponents") != list(EXPECTED_OPPONENTS):
        raise ValueError("Attempt03 population opponent order changed")
    return evaluate_attempt02_population_gates(evaluation)


def validate_attempt03_population_acceptance(
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
    """Recompute merged labels and return immutable Go/No-Go artifacts."""

    plan = load_and_validate_attempt03_population_plan(population_plan)
    plan.pop("_freshness_counts", None)
    hashes = {
        name: _sha256(source_hashes.get(name), f"{name} SHA-256")
        for name in (
            "population_plan",
            "records",
            "evaluation",
            "merge_manifest",
            "run_manifest",
        )
    }
    plan_sha = hashes["population_plan"]
    if lifecycle_preflight.get("population_plan_file_sha256") != plan_sha:
        raise ValueError("Attempt03 preflight/population plan hash mismatch")
    recomputed = summarize_hu_m4_population_records(
        records,
        opponents=EXPECTED_OPPONENTS,
        paired_seeds=int(plan["paired_seeds_per_opponent"]),
        seed=int(plan["seed"]),
        seed_stride=int(plan["seed_stride"]),
    )
    content_match = _canonical_summary(evaluation) == _canonical_summary(recomputed)
    runtime = _mapping(evaluation.get("runtime_config"), "runtime_config")
    runtime_match = (
        runtime.get("candidate_model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and runtime.get("safety_model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and runtime.get("model_schema") == HU_M43_V5_MODEL_SCHEMA
        and runtime.get("model_id") == lifecycle_preflight.get("model_id")
        and runtime.get("action_score_mode") == HU_M43_V5_ACTION_SCORE_MODE
        and runtime.get("runtime_binding_verified") is True
        and runtime.get("freeze_manifest_sha256")
        == lifecycle_preflight.get("runtime_freeze_file_sha256")
        and runtime.get("training_manifest_sha256")
        == lifecycle_preflight.get("final_training_manifest_file_sha256")
        and runtime.get("population_plan_sha256") == plan_sha
        and runtime.get("sharded_evaluation") is True
        and runtime.get("shard_count") == 20
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
        == ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA
        and lifecycle_preflight.get("status") == "pass"
        and lifecycle_preflight.get("teacher_overlap_count") == 0
        and lifecycle_preflight.get("prior_population_overlap_count") == 0
        and lifecycle_preflight.get("canonical_global_marker_verified") is True
        and _is_sha256(
            lifecycle_preflight.get("training_freeze_file_sha256")
        )
        and lifecycle_preflight.get("teacher_calibration_locked_content_packaged")
        is False
        and lifecycle_preflight.get("current_profile_mutated") is False
        and lifecycle_preflight.get("no_runtime_activation") is True
    )
    merge_shards = merge_manifest.get("shards")
    receipt_shards = spot_receipt.get("shards")
    shard_match = (
        isinstance(merge_shards, Sequence)
        and not isinstance(merge_shards, (str, bytes))
        and isinstance(receipt_shards, Sequence)
        and not isinstance(receipt_shards, (str, bytes))
        and len(merge_shards) == 20
        and len(receipt_shards) == 20
    )
    if shard_match:
        merged_by_offset = {
            row.get("offset"): row
            for row in merge_shards
            if isinstance(row, Mapping)
        }
        received_by_shard = {
            row.get("shard"): row
            for row in receipt_shards
            if isinstance(row, Mapping)
        }
        shard_match = (
            len(merged_by_offset) == 20 and len(received_by_shard) == 20
        )
        if shard_match:
            for shard in range(20):
                offset = shard * 50
                merged = merged_by_offset.get(offset)
                received = received_by_shard.get(shard)
                expected_seed = int(plan["seed"]) + offset * int(
                    plan["seed_stride"]
                )
                if (
                    not isinstance(merged, Mapping)
                    or not isinstance(received, Mapping)
                    or merged.get("seed") != expected_seed
                    or merged.get("paired_seeds") != 50
                    or merged.get("records") != 400
                    or merged.get("evaluation_sha256")
                    != received.get("evaluation_sha256")
                    or merged.get("records_sha256")
                    != received.get("records_sha256")
                    or not _is_sha256(received.get("done_sha256"))
                ):
                    shard_match = False
                    break
    merge_match = (
        merge_manifest.get("schema") == "hu_m4_population_shard_merge_v1"
        and merge_manifest.get("status") == "complete_content_verified"
        and merge_manifest.get("population_plan_sha256") == plan_sha
        and merge_manifest.get("merged_records_sha256") == hashes["records"]
        and merge_manifest.get("evaluation_sha256") == hashes["evaluation"]
        and merge_manifest.get("paired_seeds_per_opponent") == 1000
        and merge_manifest.get("opponents") == list(EXPECTED_OPPONENTS)
        and merge_manifest.get("merged_records") == 8000
        and merge_manifest.get("current_profile_used") is False
        and merge_manifest.get("metrics_recomputed_from_merged_seed_clusters")
        is True
        and shard_match
    )
    manifest_runtime = _mapping(run_manifest.get("runtime"), "run runtime")
    manifest_plan = _mapping(
        run_manifest.get("population_plan"), "run population plan"
    )
    manifest_shards = _mapping(run_manifest.get("shards"), "run shards")
    manifest_checkpoint = _mapping(
        run_manifest.get("checkpoint"), "run checkpoint"
    )
    manifest_compute = _mapping(run_manifest.get("compute"), "run compute")
    source_boundary = _mapping(
        run_manifest.get("source_boundary"), "run source boundary"
    )
    run_match = (
        run_manifest.get("schema") == ATTEMPT03_POPULATION_MANIFEST_SCHEMA
        and manifest_plan.get("sha256") == plan_sha
        and manifest_plan.get("paired_seeds") == 1000
        and manifest_plan.get("seed") == int(plan["seed"])
        and manifest_plan.get("seed_stride") == int(plan["seed_stride"])
        and manifest_plan.get("shards") == 20
        and manifest_plan.get("paired_seeds_per_shard") == 50
        and manifest_shards.get("count") == 20
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
        and manifest_runtime.get("training_freeze_sha256")
        == lifecycle_preflight.get("training_freeze_file_sha256")
        and manifest_runtime.get("freeze_manifest_sha256")
        == lifecycle_preflight.get("runtime_freeze_file_sha256")
        and manifest_runtime.get("locked_receipt_sha256")
        == lifecycle_preflight.get("locked_receipt_file_sha256")
        and manifest_runtime.get("consumption_marker_sha256")
        == lifecycle_preflight.get("consumption_marker_file_sha256")
        and manifest_runtime.get("model_schema") == HU_M43_V5_MODEL_SCHEMA
        and manifest_runtime.get("model_id") == lifecycle_preflight.get("model_id")
        and manifest_runtime.get("action_score_mode")
        == HU_M43_V5_ACTION_SCORE_MODE
        and manifest_runtime.get("runtime_teacher_inputs") is False
        and manifest_runtime.get("current_profile_used") is False
        and _same_number(
            manifest_runtime.get("frozen_threshold"),
            lifecycle_preflight.get("frozen_threshold"),
        )
        and run_manifest.get("launch_preflight") == lifecycle_preflight
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
        spot_receipt.get("schema") == ATTEMPT03_POPULATION_RECEIPT_SCHEMA
        and spot_receipt.get("status") == "verified_and_merged"
        and spot_receipt.get("run_manifest_sha256") == hashes["run_manifest"]
        and spot_receipt.get("population_plan_sha256") == plan_sha
        and spot_receipt.get("model_sha256")
        == lifecycle_preflight.get("model_sha256")
        and spot_receipt.get("model_schema") == HU_M43_V5_MODEL_SCHEMA
        and spot_receipt.get("action_score_mode")
        == HU_M43_V5_ACTION_SCORE_MODE
        and spot_receipt.get("evaluation_sha256") == hashes["evaluation"]
        and spot_receipt.get("records_sha256") == hashes["records"]
        and spot_receipt.get("merge_manifest_sha256")
        == hashes["merge_manifest"]
        and spot_receipt.get("valid_overrides")
        == recomputed["population"]["all_seats"]["overrides"]
        and spot_receipt.get("paired_seeds_per_opponent") == 1000
        and spot_receipt.get("invalid_counterfactuals")
        == recomputed["invalid_counterfactuals"]
        and spot_receipt.get("nonfire_cancellation_mismatches")
        == recomputed["nonfire_cancellation_mismatches"]
        and shard_match
        and spot_receipt.get("teacher_calibration_locked_content_received")
        is False
        and spot_receipt.get("current_profile_mutated") is False
        and spot_receipt.get("no_runtime_activation") is True
    )
    gates = [
        _gate(
            "attempt03_lifecycle_hash_chain",
            lifecycle_match and merge_match and run_match and receipt_match,
            {
                "lifecycle": lifecycle_match,
                "merge_manifest": merge_match,
                "run_manifest": run_match,
                "spot_receipt": receipt_match,
            },
            "bound v5 lifecycle, Spot transport, merge, and receipt all pass",
        ),
        _gate(
            "population_content_recomputed",
            content_match,
            content_match,
            "stored summary equals played-hand JSONL recomputation",
        ),
        _gate(
            "v5_runtime_binding",
            runtime_match,
            dict(runtime),
            "bound Attempt03 v5 model/freeze/final manifest and current unused",
        ),
        *evaluate_attempt03_population_gates(recomputed),
    ]
    passed = sum(gate["passed"] is True for gate in gates)
    decision = "complete_go" if passed == len(gates) else "complete_no_go"
    config = {
        "schema": ATTEMPT03_POPULATION_ACCEPTANCE_CONFIG_SCHEMA,
        "status": "frozen_evaluated_once",
        "policy_attempt": "attempt03_v5",
        "population_plan_sha256": plan_sha,
        "run_manifest_sha256": hashes["run_manifest"],
        "evaluation_sha256": hashes["evaluation"],
        "records_sha256": hashes["records"],
        "merge_manifest_sha256": hashes["merge_manifest"],
        "model_sha256": lifecycle_preflight.get("model_sha256"),
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V5_ACTION_SCORE_MODE,
        "frozen_threshold": lifecycle_preflight.get("frozen_threshold"),
        "fixed_acceptance_gates": dict(FIXED_GATES),
        "threshold_reselection_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    status = {
        "schema": ATTEMPT03_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--final-training-manifest", type=Path, required=True)
    preflight.add_argument("--runtime-freeze", type=Path, required=True)
    preflight.add_argument("--training-freeze", type=Path, required=True)
    preflight.add_argument("--precalibration-receipt", type=Path, required=True)
    preflight.add_argument("--model-freeze", type=Path, required=True)
    preflight.add_argument("--attempt03-plan", type=Path, required=True)
    preflight.add_argument("--locked-receipt", type=Path, required=True)
    preflight.add_argument("--consumption-marker", type=Path, required=True)
    preflight.add_argument("--population-plan", type=Path, required=True)
    accept = subparsers.add_parser("accept")
    accept.add_argument("--evaluation", type=Path, required=True)
    accept.add_argument("--records", type=Path, required=True)
    accept.add_argument("--population-plan", type=Path, required=True)
    accept.add_argument("--merge-manifest", type=Path, required=True)
    accept.add_argument("--spot-receipt", type=Path, required=True)
    accept.add_argument("--run-manifest", type=Path, required=True)
    accept.add_argument("--lifecycle-preflight", type=Path, required=True)
    accept.add_argument("--config-output", type=Path)
    accept.add_argument("--status-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        result = build_attempt03_population_launch_preflight(
            model_path=args.model,
            final_training_manifest_path=args.final_training_manifest,
            runtime_freeze_path=args.runtime_freeze,
            training_freeze_path=args.training_freeze,
            precalibration_receipt_path=args.precalibration_receipt,
            model_freeze_path=args.model_freeze,
            attempt03_plan_path=args.attempt03_plan,
            locked_receipt_path=args.locked_receipt,
            consumption_marker_path=args.consumption_marker,
            population_plan_path=args.population_plan,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    evaluation = read_json_mapping(args.evaluation, "population evaluation")
    records = _read_jsonl(args.records)
    plan = read_json_mapping(args.population_plan, "population plan")
    merge = read_json_mapping(args.merge_manifest, "merge manifest")
    spot_receipt = read_json_mapping(args.spot_receipt, "Spot receipt")
    run_manifest = read_json_mapping(args.run_manifest, "run manifest")
    preflight = read_json_mapping(args.lifecycle_preflight, "lifecycle preflight")
    config, status = validate_attempt03_population_acceptance(
        evaluation=evaluation,
        records=records,
        population_plan=plan,
        merge_manifest=merge,
        spot_receipt=spot_receipt,
        run_manifest=run_manifest,
        lifecycle_preflight=preflight,
        source_hashes={
            "population_plan": file_sha256(args.population_plan),
            "records": file_sha256(args.records),
            "evaluation": file_sha256(args.evaluation),
            "merge_manifest": file_sha256(args.merge_manifest),
            "run_manifest": file_sha256(args.run_manifest),
        },
    )
    if args.config_output is not None:
        _write_json(args.config_output, config)
    if args.status_output is not None:
        _write_json(args.status_output, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status["status"] == "complete_go" else 2


def _validate_excluded_schedules(
    planned: set[int],
    raw_schedules: Any,
    *,
    count_field: str,
    required_identities: set[tuple[Any, ...]],
    split_field: bool = False,
) -> int:
    if not isinstance(raw_schedules, Sequence) or isinstance(
        raw_schedules, (str, bytes)
    ) or not raw_schedules:
        raise ValueError("Attempt03 population exclusion schedules are missing")
    identities: set[tuple[Any, ...]] = set()
    checked = 0
    for index, raw in enumerate(raw_schedules):
        schedule = _mapping(raw, f"excluded schedule {index}")
        seed = _integer(schedule.get("seed"), f"schedule {index} seed")
        stride = _integer(schedule.get("seed_stride"), f"schedule {index} stride")
        count = _integer(schedule.get(count_field), f"schedule {index} count")
        if split_field:
            identity = (
                str(schedule.get("milestone")),
                str(schedule.get("split")),
                seed,
                stride,
                count,
            )
        else:
            identity = (str(schedule.get("milestone")), seed, stride, count)
        if identity in identities:
            raise ValueError("Attempt03 population exclusion schedule duplicated")
        identities.add(identity)
        excluded = _seed_grid(seed, stride, count)
        checked += len(excluded)
        if planned & excluded:
            raise ValueError("Attempt03 population seed overlaps an excluded schedule")
    if not required_identities.issubset(identities):
        raise ValueError("Attempt03 population critical exclusions are missing")
    return checked


def _seed_grid(seed: int, stride: int, count: int) -> set[int]:
    return {seed + index * stride for index in range(count)}


def _canonical_summary(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    normalized.pop("runtime_config", None)
    normalized.pop("shard_merge", None)
    normalized.pop("records", None)
    normalized["records_output"] = None
    normalized["elapsed_seconds"] = None
    return json.dumps(
        normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"population record {line_number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1.0e-12)
    )


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT03_POPULATION_ACCEPTANCE_CONFIG_SCHEMA",
    "ATTEMPT03_POPULATION_ACCEPTANCE_STATUS_SCHEMA",
    "ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA",
    "ATTEMPT03_POPULATION_SEED",
    "ATTEMPT03_POPULATION_SEED_STRIDE",
    "build_attempt03_population_launch_preflight",
    "evaluate_attempt03_population_gates",
    "load_and_validate_attempt03_population_plan",
    "validate_attempt03_locked_lifecycle",
    "validate_attempt03_population_acceptance",
]
