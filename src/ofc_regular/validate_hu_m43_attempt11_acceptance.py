"""Freeze, preflight, and accept the Attempt11 distilled population policy.

All acceptance metrics are recomputed from fresh played-hand records.  Search
teacher estimates remain diagnostics and neither audit50 nor population data
can refit a head, tune a threshold, or change ``current``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_hu_m4_population import summarize_hu_m4_population_records
from .hu_m43_attempt11_distilled_model import (
    BoundHuM43Attempt11DistilledModel,
    HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE,
    HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA,
    HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA,
    HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA,
    HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
    HuM43Attempt11DistilledModel,
)
from .hu_m43_attempt11_distilled_runtime import (
    ATTEMPT11_BOUND_EXECUTION_MODULES,
    ATTEMPT11_DISTILLED_RUNTIME_REGISTRY_PATHS,
    ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT11_GCP_IMAGE_ID,
    ATTEMPT11_GCP_IMAGE_NAME,
    ATTEMPT11_GCP_IMAGE_SELF_LINK,
    ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256,
    load_and_validate_distilled_runtime_manifest,
    validate_distilled_runtime_dependencies,
    validate_distilled_runtime_extracted_tree,
    validate_frozen_execution_modules,
    validate_distilled_runtime_source_archive,
)
from .hu_m43_attempt11_contract import (
    ATTEMPT11_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT11_PLAN_SHA256,
)
from .select_hu_m43_attempt11_audit50 import (
    ATTEMPT11_AUDIT50_DECISION_SCHEMA,
    ATTEMPT11_AUDIT50_RECEIPT_KEYS,
    ATTEMPT11_AUDIT50_RECEIPT_SCHEMA,
)
from .merge_hu_m4_population_shards import M4_POPULATION_MERGE_SCHEMA
from .train_hu_m43_attempt11_distilled import (
    ATTEMPT11_DISTILLATION_CONFIG_SHA256,
    ATTEMPT11_DISTILLATION_FULL_ITERATIONS,
    ATTEMPT11_DISTILLATION_FULL_STATES,
    ATTEMPT11_DISTILLED_TRAINING_MANIFEST_SCHEMA,
    ATTEMPT11_PROPOSAL_ROWS_MAX,
)
from .validate_hu_m43_attempt02_acceptance import (
    evaluate_attempt02_population_gates,
)


ATTEMPT11_POPULATION_PLAN_SCHEMA = (
    "hu_m43_population_acceptance_plan_v1"
)
ATTEMPT11_POPULATION_PREFLIGHT_SCHEMA = (
    "hu_m43_attempt11_population_launch_preflight_v1"
)
ATTEMPT11_RUNTIME_FREEZE_SCHEMA = "hu_m43_attempt11_distilled_runtime_freeze_v1"
ATTEMPT11_POPULATION_ACCEPTANCE_STATUS_SCHEMA = (
    "hu_m43_attempt11_population_acceptance_status_v1"
)
ATTEMPT11_AUDIT50_SELECTOR_RECEIPT_SCHEMA = ATTEMPT11_AUDIT50_RECEIPT_SCHEMA
ATTEMPT11_OPPONENTS = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "random_exact_final",
)
ATTEMPT11_POPULATION_SEED = 190_108_071_901
ATTEMPT11_POPULATION_SEED_STRIDE = 1_000_003
ATTEMPT11_POPULATION_SEEDS = 1000
ATTEMPT11_POPULATION_SHARDS = 20
ATTEMPT11_POPULATION_SEEDS_PER_SHARD = 50
ATTEMPT11_BASELINE_PROFILE = "stage19_p0"
ATTEMPT11_FIXED_GATES = {
    "invalid_counterfactuals_max": 0,
    "nonfire_cancellation_mismatches_max": 0,
    "nonfire_cancellation_unknown_max": 0,
    "valid_overrides_min": 300,
    "realized_gain_per_override_ci95_low_exclusive": 0.0,
    "paired_seat_swap_delta_ci95_low_exclusive": 0.0,
    "second_seat_delta_ci95_low_exclusive": 0.0,
    "first_seat_delta_exact": 0.0,
    "false_positive_override_rate_max": 0.3,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "each_opponent_delta_ev_per_hand_min": -0.005,
    "each_opponent_delta_ci95_low_min": -0.02,
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
ATTEMPT11_SEED_REGISTRY_SOURCES = ATTEMPT11_DISTILLED_RUNTIME_REGISTRY_PATHS


def load_and_validate_attempt11_population_plan(
    source: str | Path | Mapping[str, Any]
) -> dict[str, Any]:
    plan = dict(source) if isinstance(source, Mapping) else _load_mapping(source, "plan")
    if (
        plan.get("schema") != ATTEMPT11_POPULATION_PLAN_SCHEMA
        or plan.get("status") != "frozen_before_population_evaluation"
        or plan.get("policy_attempt") != "attempt11_distilled_v1"
        or plan.get("fixed_baseline_profile") != ATTEMPT11_BASELINE_PROFILE
        or plan.get("opponents") != list(ATTEMPT11_OPPONENTS)
        or plan.get("paired_seat_swap") is not True
    ):
        raise ValueError("Attempt11 population plan identity changed")
    if plan.get("baseline_chain") != [
        "stage19_p0",
        "stage18_p1",
        "stage9f_p2",
        "stage7_m5_r10",
        "exact_t4",
    ]:
        raise ValueError("Attempt11 population baseline chain changed")
    runtime = _mapping(plan.get("runtime_contract"), "runtime_contract")
    expected_runtime = {
        "model_schema": HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
        "artifact_schema": HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA,
        "feature_schema": HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA,
        "head_schema": HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA,
        "action_score_mode": HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE,
        "fixed_safe_probability_threshold": 0.5,
        "fixed_fold_votes_min": 4,
        "folds": 5,
        "runtime_teacher_inputs": False,
        "runtime_teacher_ev_lcb_gate": False,
        "runtime_opponent_private_discard_input": False,
    }
    if runtime != expected_runtime:
        raise ValueError("Attempt11 population runtime contract changed")
    schedule = (
        _integer(plan.get("paired_seeds_per_opponent"), "paired seeds"),
        _integer(plan.get("seed"), "seed"),
        _integer(plan.get("seed_stride"), "seed stride"),
        _integer(plan.get("shards"), "shards"),
        _integer(plan.get("paired_seeds_per_shard"), "seeds per shard"),
    )
    if schedule != (
        ATTEMPT11_POPULATION_SEEDS,
        ATTEMPT11_POPULATION_SEED,
        ATTEMPT11_POPULATION_SEED_STRIDE,
        ATTEMPT11_POPULATION_SHARDS,
        ATTEMPT11_POPULATION_SEEDS_PER_SHARD,
    ):
        raise ValueError("Attempt11 population schedule changed")
    if (
        plan.get("candidate_records") != 8000
        or plan.get("baseline_records") != 8000
        or plan.get("terminal_trace_hands") != 16000
        or plan.get("second_seat_override_opportunities") != 4000
        or plan.get("minimum_valid_overrides") != 300
    ):
        raise ValueError("Attempt11 population power contract changed")
    sizing = _mapping(plan.get("sizing_rule"), "sizing rule")
    if sizing != {
        "formula": "paired_seeds_required = ceil(minimum_valid_overrides / (opponents * independent_fire_rate_lower_bound))",
        "fixed_before_realized_population_labels": True,
        "optional_stopping_or_posthoc_extension_allowed": False,
        "insufficient_overrides_decision": "complete_no_go",
    }:
        raise ValueError("Attempt11 population optional-stopping contract changed")
    evaluation_contract = _mapping(
        plan.get("evaluation_contract"), "evaluation contract"
    )
    if evaluation_contract != {
        "candidate_and_baseline_same_hand_seed": True,
        "candidate_and_baseline_same_physical_seat": True,
        "candidate_and_baseline_same_opponent_policy_seed": True,
        "first_and_second_seat_metrics": True,
        "first_seat_candidate_is_exact_baseline_delegate": True,
        "nonfire_full_trajectory_digest_cancellation": True,
        "invalid_counterfactuals_excluded": True,
        "false_positive_override_definition": "realized_delta_le_zero",
        "confidence_interval_unit": "hand_seed_cluster_after_opponent_average",
        "teacher_values_are_realized_match_ev": False,
        "threshold_reselection_allowed": False,
    }:
        raise ValueError("Attempt11 population counterfactual contract changed")
    spot = _mapping(plan.get("spot_execution"), "Spot execution")
    if spot != {
        "machine_type": "c4-standard-4",
        "provisioning_model": "SPOT",
        "small_shards": True,
        "checkpoint_unit": "completed_shard",
        "heartbeat_required": True,
        "immutable_package_only_first": True,
        "shard_zero_canary_required_before_fanout": True,
        "fanout_shards": "1..19_after_verified_shard0_DONE",
        "resume_missing_shards_only": True,
        "done_commit_last": True,
        "merge_recomputes_metrics_from_records": True,
    }:
        raise ValueError("Attempt11 population Spot execution contract changed")
    if _mapping(plan.get("fixed_acceptance_gates"), "gates") != ATTEMPT11_FIXED_GATES:
        raise ValueError("Attempt11 fixed acceptance gates changed")
    guards = _mapping(plan.get("activation_guards"), "activation guards")
    expected_guard_names = {
            "current_profile_changed",
            "runtime_policy_activated",
            "full_replacement_enabled",
            "threshold_changed_after_lock",
    }
    if set(guards) != expected_guard_names or any(
        value is not False for value in guards.values()
    ):
        raise ValueError("Attempt11 population activation guard changed")
    if _mapping(
        plan.get("post_acceptance_activation"), "post-acceptance activation"
    ) != {
        "population_complete_go_required": True,
        "activation_mode_if_go": "explicit_opt_in_only",
        "automatic_activation_allowed": False,
        "current_profile_change_allowed": False,
    }:
        raise ValueError("Attempt11 post-acceptance activation contract changed")
    planned = {
        ATTEMPT11_POPULATION_SEED + index * ATTEMPT11_POPULATION_SEED_STRIDE
        for index in range(ATTEMPT11_POPULATION_SEEDS)
    }
    freshness = _mapping(plan.get("freshness"), "freshness")
    for name in (
        "exclude_all_m4_m41_m42_m43_teacher_hand_seeds",
        "exclude_attempt11_preflight_development_and_audit_namespaces",
        "exclude_prior_population_smoke_and_frozen_schedules",
        "acceptance_validator_rechecks_all_declared_seed_schedules",
    ):
        if freshness.get(name) is not True:
            raise ValueError("Attempt11 population freshness contract changed")
    if (
        freshness.get("planned_overlap_count_at_freeze") != 0
        or freshness.get("alternate_seed_after_result_allowed") is not False
    ):
        raise ValueError("Attempt11 population seed-selection contract changed")
    registry_sources = tuple(
        str(item)
        for item in _sequence(
            freshness.get("excluded_schedule_registry_sources"),
            "seed registry sources",
        )
    )
    if registry_sources != ATTEMPT11_SEED_REGISTRY_SOURCES:
        raise ValueError("Attempt11 seed registry source list changed")
    registry_seeds, registry_rows = _load_seed_registry(registry_sources)
    registry_overlap = planned & registry_seeds
    if registry_overlap:
        raise ValueError("Attempt11 population seed grid overlaps hashed registry")
    registry_digest = hashlib.sha256(
        _canonical(registry_rows).encode("utf-8")
    ).hexdigest()
    if freshness.get("excluded_schedule_registry_sha256") != registry_digest:
        raise ValueError("Attempt11 hashed seed registry snapshot changed")
    plan["_freshness_registry"] = {
        "schema": "hu_m43_attempt11_seed_registry_snapshot_v1",
        "source_count": len(registry_rows),
        "seed_count": len(registry_seeds),
        "planned_overlap_count": 0,
        "sha256": registry_digest,
        "sources": registry_rows,
    }
    prior: set[int] = set()
    for row in _sequence(
        freshness.get("excluded_population_schedules"), "excluded schedules"
    ):
        schedule_row = _mapping(row, "excluded schedule")
        seed = _integer(schedule_row.get("seed"), "excluded seed")
        stride = _integer(schedule_row.get("seed_stride"), "excluded stride")
        count = _integer(schedule_row.get("paired_seeds"), "excluded count")
        prior.update(seed + index * stride for index in range(count))
    teacher: set[int] = set()
    for raw_base in _sequence(
        freshness.get("excluded_attempt11_namespace_bases"), "namespace bases"
    ):
        base = _integer(raw_base, "namespace base")
        teacher.update(
            base + index * ATTEMPT11_POPULATION_SEED_STRIDE for index in range(250)
        )
    plan["_freshness_counts"] = {
        "teacher_overlap_count": len(planned & teacher),
        "prior_population_overlap_count": len(planned & prior),
    }
    if any(plan["_freshness_counts"].values()):
        raise ValueError("Attempt11 population seed grid overlaps prior evidence")
    return plan


def _load_seed_registry(
    relative_sources: Sequence[str],
) -> tuple[set[int], list[dict[str, Any]]]:
    """Hash and enumerate every declared historical seed schedule.

    Registry paths are repository-relative and fixed by contract.  The
    recursive extractor deliberately over-approximates ``*_seed_base`` grids;
    proving disjointness against a superset is safer than trusting a hand-kept
    inline exclusion list.
    """

    seeds: set[int] = set()
    rows: list[dict[str, Any]] = []
    root = _REPO_ROOT.resolve()
    for relative in relative_sources:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Attempt11 seed registry path escapes repository")
        source = (root / candidate).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("Attempt11 seed registry path escapes repository") from exc
        if not source.is_file():
            raise ValueError(f"Attempt11 seed registry source missing: {relative}")
        payload = _load_mapping(source, f"seed registry {relative}")
        source_seeds = _extract_declared_seed_schedules(payload)
        seeds.update(source_seeds)
        rows.append(
            {
                "path": candidate.as_posix(),
                "sha256": _file_sha256(source),
                "seed_count": len(source_seeds),
                "seed_set_sha256": hashlib.sha256(
                    _canonical(sorted(source_seeds)).encode("utf-8")
                ).hexdigest(),
            }
        )
    return seeds, rows


def _extract_declared_seed_schedules(payload: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()

    def visit(value: Any, inherited_stride: int = ATTEMPT11_POPULATION_SEED_STRIDE) -> None:
        if isinstance(value, Mapping):
            local = dict(value)
            raw_stride = local.get("seed_stride", inherited_stride)
            stride = (
                raw_stride
                if isinstance(raw_stride, int) and not isinstance(raw_stride, bool)
                and raw_stride > 0
                else inherited_stride
            )
            counts = [
                local.get(name)
                for name in (
                    "paired_seeds_per_opponent",
                    "paired_seeds",
                    "roots",
                    "count",
                    "candidate_evaluation_child_seed_count",
                )
            ]
            count = next(
                (
                    int(item)
                    for item in counts
                    if isinstance(item, int) and not isinstance(item, bool) and item > 0
                ),
                1,
            )
            explicit_indices = local.get("allowed_source_root_indices")
            if isinstance(explicit_indices, Sequence) and not isinstance(
                explicit_indices, (str, bytes, bytearray)
            ):
                valid_indices = [
                    item
                    for item in explicit_indices
                    if isinstance(item, int) and not isinstance(item, bool) and item >= 0
                ]
                if valid_indices:
                    count = max(count, max(valid_indices) + 1)
            range_max = 0
            for item in local.values():
                if isinstance(item, str) and ".." in item and "inclusive" in item:
                    digits = [int(token) for token in item.replace("..", " ").split() if token.isdigit()]
                    if digits:
                        range_max = max(range_max, max(digits) + 1)
            for key, raw_seed in local.items():
                if not (
                    key == "seed"
                    or key == "seed_start"
                    or key.endswith("_seed_start")
                    or key.endswith("_seed_base")
                ):
                    continue
                if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
                    continue
                schedule_count = count
                if key.endswith("_seed_base"):
                    # A missing/changed textual index declaration must not
                    # shrink the exclusion proof.
                    schedule_count = max(schedule_count, range_max, 1000)
                result.update(
                    raw_seed + index * stride for index in range(schedule_count)
                )
            for child in local.values():
                visit(child, stride)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                visit(child, inherited_stride)

    visit(payload)
    return result


def freeze_attempt11_distilled_runtime(
    *,
    source_model_path: str | Path,
    training_manifest_path: str | Path,
    audit_decision_path: str | Path,
    audit_selector_receipt_path: str | Path,
    runtime_source_archive_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_dependency_root: str | Path,
    output_model_path: str | Path,
    output_runtime_freeze_path: str | Path,
) -> dict[str, Any]:
    """Enable only an audit-Go distilled artifact; never register a profile."""

    source_sha = _file_sha256(source_model_path)
    training_sha = _file_sha256(training_manifest_path)
    audit_sha = _file_sha256(audit_decision_path)
    audit_receipt_sha = _file_sha256(audit_selector_receipt_path)
    runtime_source_archive_sha = _file_sha256(runtime_source_archive_path)
    runtime_source_manifest_sha = _file_sha256(runtime_source_manifest_path)
    training = _load_mapping(training_manifest_path, "training manifest")
    audit = _load_mapping(audit_decision_path, "audit50 decision")
    audit_receipt = _load_mapping(
        audit_selector_receipt_path, "audit50 selector receipt"
    )
    _validate_training_manifest(training, expected_model_sha256=source_sha)
    runtime_source = validate_distilled_runtime_source_archive(
        archive_path=runtime_source_archive_path,
        manifest=runtime_source_manifest_path,
    )
    runtime_file_set = _mapping(runtime_source.get("file_set"), "runtime file set")
    runtime_semantic = _mapping(
        runtime_source.get("semantic_closure"), "runtime semantic closure"
    )
    runtime_external = _mapping(
        runtime_semantic.get("external_runtime"), "runtime external identity"
    )
    runtime_dependencies = validate_distilled_runtime_dependencies(
        runtime_dependency_root
    )
    _validate_runtime_dependency_semantics(runtime_semantic, runtime_dependencies)
    training_source = _mapping(training.get("source"), "training source")
    expected_runtime_bindings = {
        "runtime_source_archive_sha256": runtime_source_archive_sha,
        "runtime_source_manifest_sha256": runtime_source_manifest_sha,
        "runtime_source_closure_sha256": runtime_file_set["sha256"],
        "runtime_semantic_closure_sha256": runtime_semantic["sha256"],
        "runtime_requirements_sha256": ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": runtime_dependencies[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": runtime_dependencies[
            "source_native_manifest_sha256"
        ],
        "runtime_dependency_closure_sha256": runtime_dependencies["sha256"],
    }
    if any(
        training_source.get(name) != value
        for name, value in expected_runtime_bindings.items()
    ):
        raise ValueError("Attempt11 training/runtime source freeze mismatch")
    if (
        runtime_external.get("requirements_sha256")
        != ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256
        or runtime_external.get("runtime_fingerprint_sha256")
        != ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256
    ):
        raise ValueError("Attempt11 external runtime identity changed")
    audit_science = _mapping(audit.get("science_boundary"), "audit50 science")
    if (
        audit.get("schema") != ATTEMPT11_AUDIT50_DECISION_SCHEMA
        or audit.get("status") != "go_attempt11_audit50_search_quality"
        or audit.get("decision") != "go"
        or audit.get("search_freeze_authorized") is not True
        or audit.get("selected_arm") is not None
        or audit.get("selected_threshold") is not None
        or audit_science.get("fit_performed") is not False
        or audit_science.get("threshold_selected") is not False
        or audit_science.get("runtime_policy_activated") is not False
        or audit_science.get("current_profile_mutated") is not False
        or audit_science.get("teacher_values_are_realized_match_ev") is not False
    ):
        raise ValueError("Attempt11 runtime freeze requires the one-shot audit50 Go")
    audit_gates = _sequence(audit.get("gates"), "audit50 gates")
    audit_contract = _mapping(
        audit.get("decision_contract"), "audit50 decision contract"
    )
    if (
        not audit_gates
        or not all(
            isinstance(gate, Mapping) and gate.get("passed") is True
            for gate in audit_gates
        )
        or audit_contract.get("gate_evaluation_count") != 1
        or audit_contract.get("single_frozen_search_architecture") is not True
        or audit_contract.get("arm_selection_performed") is not False
        or audit_contract.get("threshold_selection_performed") is not False
        or audit_contract.get("all_gates_required") is not True
    ):
        raise ValueError("Attempt11 audit50 Go contract changed")
    _validate_attempt11_audit50_receipt_for_freeze(
        audit_receipt, audit_decision_sha256=audit_sha
    )
    source_model = HuM43Attempt11DistilledModel.load(
        source_model_path, expected_sha256=source_sha
    )
    if source_model.safety_enabled or source_model.winner_frozen:
        raise ValueError("Attempt11 source model is already runtime frozen")
    source_fit = _mapping(source_model.manifest, "source model manifest")
    if any(
        source_fit.get(name) != value
        for name, value in {
            "training_data": "attempt11_development200_only",
            "audit50_fit_rows": 0,
            "threshold_sweep_performed": False,
            "identity_grouped_folds": 5,
            "fit_mode": "full",
            "effective_iterations": ATTEMPT11_DISTILLATION_FULL_ITERATIONS,
            "training_states": ATTEMPT11_DISTILLATION_FULL_STATES,
            "distillation_config_sha256": ATTEMPT11_DISTILLATION_CONFIG_SHA256,
        }.items()
    ):
        raise ValueError("Attempt11 source model is not the frozen full-fit artifact")
    frozen_model = dataclasses.replace(
        source_model,
        safety_enabled=True,
        winner_frozen=True,
        manifest={
            **dict(source_model.manifest),
            "audit50_decision_sha256": audit_sha,
            "audit50_selector_receipt_sha256": audit_receipt_sha,
            "source_training_model_sha256": source_sha,
            "training_manifest_sha256": training_sha,
            **expected_runtime_bindings,
            "threshold_sweep_performed": False,
            "audit50_fit_rows": 0,
        },
    )
    final_sha = frozen_model.save(output_model_path)
    freeze = {
        "schema": ATTEMPT11_RUNTIME_FREEZE_SCHEMA,
        "status": "frozen_after_development200_fit_and_audit50_go",
        "model_sha256": final_sha,
        "model_id": frozen_model.model_id,
        "model_schema": frozen_model.schema,
        "artifact_schema": frozen_model.artifact_schema,
        "feature_schema": frozen_model.feature_schema,
        "head_schema": frozen_model.head_schema,
        "action_score_mode": frozen_model.action_score_mode,
        "source_training_model_sha256": source_sha,
        "training_manifest_sha256": training_sha,
        **expected_runtime_bindings,
        "gcp_image": {
            "name": ATTEMPT11_GCP_IMAGE_NAME,
            "id": ATTEMPT11_GCP_IMAGE_ID,
            "self_link": ATTEMPT11_GCP_IMAGE_SELF_LINK,
        },
        "audit50_decision_sha256": audit_sha,
        "audit50_selector_receipt_sha256": audit_receipt_sha,
        "safety_enabled": True,
        "winner_frozen": True,
        "fixed_safe_probability_threshold": 0.5,
        "fixed_fold_votes_min": 4,
        "tail_upper_limits": [25.0, 40.0, 50.0],
        "audit50_fit_rows": 0,
        "threshold_reselection_performed": False,
        "teacher_ev_or_lcb_runtime_gate": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    _write_new_json(output_runtime_freeze_path, freeze)
    return freeze


def _validate_attempt11_audit50_receipt_for_freeze(
    receipt: Mapping[str, Any], *, audit_decision_sha256: str
) -> None:
    if (
        set(receipt) != ATTEMPT11_AUDIT50_RECEIPT_KEYS
        or receipt.get("schema") != ATTEMPT11_AUDIT50_SELECTOR_RECEIPT_SCHEMA
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("decision_sha256") != audit_decision_sha256
        or receipt.get("decision") != "go"
        or receipt.get("search_freeze_authorized") is not True
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
        or receipt.get("future_audit_authorized") is not False
        or receipt.get("audit_rows_used_for_fit") is not False
        or receipt.get("fit_performed") is not False
        or receipt.get("threshold_selected") is not False
        or receipt.get("current_profile_mutated") is not False
        or receipt.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt11 audit50 selector receipt is not freeze authorization")


def load_bound_attempt11_distilled_model(
    path: str | Path,
    *,
    expected_sha256: str,
    runtime_freeze: str | Path | Mapping[str, Any],
    training_manifest_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
) -> BoundHuM43Attempt11DistilledModel:
    # Binding is a source-provenance capability, not merely an artifact-hash
    # check.  Keep this attestation inside the sole production bound loader so
    # direct library callers cannot bypass the frozen execution tree checks.
    execution_attestation = validate_frozen_execution_modules(
        extracted_root=runtime_source_root,
        manifest=runtime_source_manifest_path,
        module_names=ATTEMPT11_BOUND_EXECUTION_MODULES,
    )
    actual_sha = _file_sha256(path)
    if actual_sha != _sha256(expected_sha256, "model SHA"):
        raise ValueError("Attempt11 bound model SHA mismatch")
    freeze = (
        dict(runtime_freeze)
        if isinstance(runtime_freeze, Mapping)
        else _load_mapping(runtime_freeze, "runtime freeze")
    )
    training = _load_mapping(training_manifest_path, "training manifest")
    training_sha = _file_sha256(training_manifest_path)
    runtime_manifest_sha = _file_sha256(runtime_source_manifest_path)
    runtime_source = validate_distilled_runtime_extracted_tree(
        extracted_root=runtime_source_root,
        manifest=runtime_source_manifest_path,
    )
    runtime_file_set = _mapping(runtime_source.get("file_set"), "runtime file set")
    runtime_semantic = _mapping(
        runtime_source.get("semantic_closure"), "runtime semantic closure"
    )
    runtime_dependencies = validate_distilled_runtime_dependencies(
        runtime_dependency_root
    )
    _validate_runtime_dependency_semantics(runtime_semantic, runtime_dependencies)
    if (
        freeze.get("schema") != ATTEMPT11_RUNTIME_FREEZE_SCHEMA
        or freeze.get("status")
        != "frozen_after_development200_fit_and_audit50_go"
        or freeze.get("model_sha256") != actual_sha
        or freeze.get("training_manifest_sha256") != training_sha
        or freeze.get("safety_enabled") is not True
        or freeze.get("winner_frozen") is not True
        or freeze.get("fixed_safe_probability_threshold") != 0.5
        or freeze.get("fixed_fold_votes_min") != 4
        or freeze.get("threshold_reselection_performed") is not False
        or freeze.get("current_profile_mutated") is not False
        or freeze.get("runtime_source_manifest_sha256") != runtime_manifest_sha
        or freeze.get("runtime_source_closure_sha256")
        != runtime_file_set.get("sha256")
        or freeze.get("runtime_semantic_closure_sha256")
        != runtime_semantic.get("sha256")
        or freeze.get("runtime_source_archive_sha256")
        != runtime_source.get("archive", {}).get("sha256")
        or freeze.get("runtime_requirements_sha256")
        != ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256
        or freeze.get("runtime_fingerprint_sha256")
        != ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or freeze.get("source_model_manifest_sha256")
        != runtime_dependencies["source_model_manifest_sha256"]
        or freeze.get("source_native_manifest_sha256")
        != runtime_dependencies["source_native_manifest_sha256"]
        or freeze.get("runtime_dependency_closure_sha256")
        != runtime_dependencies["sha256"]
        or freeze.get("gcp_image")
        != {
            "name": ATTEMPT11_GCP_IMAGE_NAME,
            "id": ATTEMPT11_GCP_IMAGE_ID,
            "self_link": ATTEMPT11_GCP_IMAGE_SELF_LINK,
        }
    ):
        raise ValueError("Attempt11 runtime freeze contract changed")
    _sha256(
        freeze.get("audit50_selector_receipt_sha256"),
        "audit50 selector receipt SHA",
    )
    _validate_training_manifest(
        training,
        expected_model_sha256=freeze.get("source_training_model_sha256"),
    )
    if training.get("model_id") != freeze.get("model_id"):
        raise ValueError("Attempt11 training manifest/runtime freeze mismatch")
    training_source = _mapping(training.get("source"), "training source")
    for name in (
        "runtime_source_archive_sha256",
        "runtime_source_manifest_sha256",
        "runtime_source_closure_sha256",
        "runtime_semantic_closure_sha256",
        "runtime_requirements_sha256",
        "runtime_fingerprint_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "runtime_dependency_closure_sha256",
    ):
        if training_source.get(name) != freeze.get(name):
            raise ValueError("Attempt11 runtime source hash chain changed")
    model = HuM43Attempt11DistilledModel.load(path, expected_sha256=actual_sha)
    embedded_manifest = _mapping(model.manifest, "frozen model manifest")
    if any(
        embedded_manifest.get(name) != value
        for name, value in {
            "fit_mode": "full",
            "effective_iterations": ATTEMPT11_DISTILLATION_FULL_ITERATIONS,
            "training_states": ATTEMPT11_DISTILLATION_FULL_STATES,
            "identity_grouped_folds": 5,
            "distillation_config_sha256": ATTEMPT11_DISTILLATION_CONFIG_SHA256,
        }.items()
    ):
        raise ValueError("Attempt11 frozen model full-fit contract changed")
    for name in (
        "training_manifest_sha256",
        "runtime_source_archive_sha256",
        "runtime_source_manifest_sha256",
        "runtime_source_closure_sha256",
        "runtime_semantic_closure_sha256",
        "runtime_requirements_sha256",
        "runtime_fingerprint_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "runtime_dependency_closure_sha256",
    ):
        if embedded_manifest.get(name) != freeze.get(name):
            raise ValueError("Attempt11 frozen model semantic binding changed")
    if (
        model.safety_enabled is not True
        or model.winner_frozen is not True
        or model.model_id != freeze.get("model_id")
        or model.schema != freeze.get("model_schema")
        or model.artifact_schema != freeze.get("artifact_schema")
        or model.feature_schema != freeze.get("feature_schema")
        or model.head_schema != freeze.get("head_schema")
        or model.action_score_mode != freeze.get("action_score_mode")
        or model.safety_threshold != 0.5
        or model.minimum_fold_votes != 4
    ):
        raise ValueError("Attempt11 frozen model payload disagrees with freeze")
    return BoundHuM43Attempt11DistilledModel(model, execution_attestation)


def build_attempt11_population_preflight(
    *,
    model_path: str | Path,
    training_manifest_path: str | Path,
    runtime_freeze_path: str | Path,
    population_plan_path: str | Path,
    runtime_source_archive_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
) -> dict[str, Any]:
    plan = load_and_validate_attempt11_population_plan(population_plan_path)
    counts = _mapping(plan.pop("_freshness_counts"), "freshness counts")
    registry = _mapping(plan.pop("_freshness_registry"), "freshness registry")
    runtime_source = validate_distilled_runtime_source_archive(
        archive_path=runtime_source_archive_path,
        manifest=runtime_source_manifest_path,
    )
    model_sha = _file_sha256(model_path)
    model = load_bound_attempt11_distilled_model(
        model_path,
        expected_sha256=model_sha,
        runtime_freeze=runtime_freeze_path,
        training_manifest_path=training_manifest_path,
        runtime_source_manifest_path=runtime_source_manifest_path,
        runtime_source_root=runtime_source_root,
        runtime_dependency_root=runtime_dependency_root,
    )
    return {
        "schema": ATTEMPT11_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "model_schema": model.schema,
        "artifact_schema": model.artifact_schema,
        "feature_schema": model.feature_schema,
        "head_schema": model.head_schema,
        "action_score_mode": model.action_score_mode,
        "baseline_profile": ATTEMPT11_BASELINE_PROFILE,
        "runtime_binding_verified": True,
        "model_sha256": model_sha,
        "model_id": model.model_id,
        "training_manifest_sha256": _file_sha256(training_manifest_path),
        "runtime_freeze_sha256": _file_sha256(runtime_freeze_path),
        "runtime_source_archive_sha256": _file_sha256(runtime_source_archive_path),
        "runtime_source_manifest_sha256": _file_sha256(runtime_source_manifest_path),
        "runtime_source_closure_sha256": runtime_source["file_set"]["sha256"],
        "runtime_semantic_closure_sha256": runtime_source["semantic_closure"][
            "sha256"
        ],
        "runtime_requirements_sha256": ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": model.manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": model.manifest[
            "source_native_manifest_sha256"
        ],
        "runtime_dependency_closure_sha256": model.manifest[
            "runtime_dependency_closure_sha256"
        ],
        "population_plan_file_sha256": _file_sha256(population_plan_path),
        "fixed_safe_probability_threshold": model.safety_threshold,
        "fixed_fold_votes_min": model.minimum_fold_votes,
        "teacher_overlap_count": counts["teacher_overlap_count"],
        "prior_population_overlap_count": counts["prior_population_overlap_count"],
        "seed_registry_sha256": registry["sha256"],
        "seed_registry_source_count": registry["source_count"],
        "seed_registry_seed_count": registry["seed_count"],
        "seed_registry_sources": registry["sources"],
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }


def validate_attempt11_population_acceptance(
    *,
    evaluation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    population_plan: Mapping[str, Any],
    merge_manifest: Mapping[str, Any],
    model_path: str | Path,
    training_manifest_path: str | Path,
    runtime_freeze_path: str | Path,
    runtime_source_archive_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    plan_path = Path(source_hashes["population_plan_path"])
    plan_file_sha = _file_sha256(plan_path)
    if _sha256(source_hashes["population_plan"], "population plan SHA") != plan_file_sha:
        raise ValueError("Attempt11 population plan path/hash mismatch")
    file_plan = _load_mapping(plan_path, "population plan file")
    if _canonical(file_plan) != _canonical(population_plan):
        raise ValueError("Attempt11 population plan mapping/file mismatch")
    for name in ("records", "evaluation", "merge_manifest"):
        _sha256(source_hashes[name], f"{name} SHA")
    plan = load_and_validate_attempt11_population_plan(population_plan)
    plan.pop("_freshness_counts", None)
    plan.pop("_freshness_registry", None)
    preflight = build_attempt11_population_preflight(
        model_path=model_path,
        training_manifest_path=training_manifest_path,
        runtime_freeze_path=runtime_freeze_path,
        population_plan_path=plan_path,
        runtime_source_archive_path=runtime_source_archive_path,
        runtime_source_manifest_path=runtime_source_manifest_path,
        runtime_source_root=runtime_source_root,
        runtime_dependency_root=runtime_dependency_root,
    )
    recomputed = summarize_hu_m4_population_records(
        records,
        opponents=ATTEMPT11_OPPONENTS,
        paired_seeds=ATTEMPT11_POPULATION_SEEDS,
        seed=ATTEMPT11_POPULATION_SEED,
        seed_stride=ATTEMPT11_POPULATION_SEED_STRIDE,
        baseline_profile=ATTEMPT11_BASELINE_PROFILE,
    )
    content_match = _canonical(_summary_for_comparison(evaluation)) == _canonical(
        _summary_for_comparison(recomputed)
    )
    plan_sha = source_hashes["population_plan"]
    runtime = _mapping(evaluation.get("runtime_config"), "runtime_config")
    runtime_match = (
        runtime.get("candidate_model_sha256") == preflight["model_sha256"]
        and runtime.get("safety_model_sha256") == preflight["model_sha256"]
        and runtime.get("model_schema") == HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA
        and runtime.get("artifact_schema") == HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA
        and runtime.get("feature_schema") == HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA
        and runtime.get("head_schema") == HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA
        and runtime.get("model_id") == preflight["model_id"]
        and runtime.get("action_score_mode")
        == HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE
        and runtime.get("baseline_profile") == ATTEMPT11_BASELINE_PROFILE
        and runtime.get("runtime_binding_verified") is True
        and runtime.get("freeze_manifest_sha256")
        == preflight["runtime_freeze_sha256"]
        and runtime.get("training_manifest_sha256")
        == preflight["training_manifest_sha256"]
        and runtime.get("runtime_source_manifest_sha256")
        == preflight["runtime_source_manifest_sha256"]
        and runtime.get("runtime_source_closure_sha256")
        == preflight["runtime_source_closure_sha256"]
        and runtime.get("runtime_semantic_closure_sha256")
        == preflight["runtime_semantic_closure_sha256"]
        and runtime.get("runtime_requirements_sha256")
        == ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256
        and runtime.get("runtime_fingerprint_sha256")
        == ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        and runtime.get("source_model_manifest_sha256")
        == preflight["source_model_manifest_sha256"]
        and runtime.get("source_native_manifest_sha256")
        == preflight["source_native_manifest_sha256"]
        and runtime.get("runtime_dependency_closure_sha256")
        == preflight["runtime_dependency_closure_sha256"]
        and runtime.get("seed_registry_sha256")
        == preflight["seed_registry_sha256"]
        and runtime.get("current_profile_used") is False
        and runtime.get("promotion_artifact_contract") is True
        and runtime.get("diagnostic_legacy") is False
        and runtime.get("safety_enabled") is True
        and _same_number(runtime.get("safety_threshold"), 0.5)
        and runtime.get("population_plan_sha256") == plan_sha
        and runtime.get("sharded_evaluation") is True
        and runtime.get("shard_count") == ATTEMPT11_POPULATION_SHARDS
        and runtime.get("final_metrics_recomputed_from_merged_records") is True
        and all(row.get("runtime_binding_verified") is True for row in records)
    )
    merge_match = (
        merge_manifest.get("schema") == M4_POPULATION_MERGE_SCHEMA
        and merge_manifest.get("status") == "complete_content_verified"
        and merge_manifest.get("population_plan_sha256") == plan_sha
        and merge_manifest.get("seed") == ATTEMPT11_POPULATION_SEED
        and merge_manifest.get("seed_stride") == ATTEMPT11_POPULATION_SEED_STRIDE
        and merge_manifest.get("paired_seeds_per_opponent")
        == ATTEMPT11_POPULATION_SEEDS
        and merge_manifest.get("opponents") == list(ATTEMPT11_OPPONENTS)
        and merge_manifest.get("merged_records") == 8000
        and merge_manifest.get("current_profile_used") is False
        and merge_manifest.get("metrics_recomputed_from_merged_seed_clusters") is True
        and isinstance(merge_manifest.get("shards"), Sequence)
        and len(merge_manifest.get("shards")) == ATTEMPT11_POPULATION_SHARDS
    )
    gates = [
        _gate(
            "runtime_artifact_hash_chain",
            runtime_match,
            runtime_match,
            "bound distilled artifact, training manifest, and runtime freeze",
        ),
        _gate(
            "population_content_recomputed",
            content_match,
            content_match,
            "stored summary equals complete played-hand JSONL recomputation",
        ),
        _gate(
            "strict_shard_merge",
            merge_match,
            merge_match,
            "fixed 20x50 merge and population plan hash",
        ),
        *evaluate_attempt02_population_gates(recomputed),
    ]
    passed = sum(gate["passed"] is True for gate in gates)
    decision = "complete_go" if passed == len(gates) else "complete_no_go"
    return {
        "schema": ATTEMPT11_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
        "status": decision,
        "passed_gates": passed,
        "total_gates": len(gates),
        "gates": gates,
        "promotion_eligible": decision == "complete_go",
        "explicit_opt_in_authorized": decision == "complete_go",
        "automatic_activation_authorized": False,
        "population_plan_sha256": plan_sha,
        "records_sha256": source_hashes["records"],
        "evaluation_sha256": source_hashes["evaluation"],
        "merge_manifest_sha256": source_hashes["merge_manifest"],
        "model_sha256": preflight["model_sha256"],
        "baseline_profile": ATTEMPT11_BASELINE_PROFILE,
        "teacher_values_reported_as_realized_match_ev": False,
        "threshold_reselection_performed": False,
        "audit50_rows_used_for_fit": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }


def _summary_for_comparison(summary: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    normalized.pop("runtime_config", None)
    normalized.pop("shard_merge", None)
    normalized.pop("records", None)
    normalized["records_output"] = None
    normalized["elapsed_seconds"] = None
    return normalized


def _validate_training_manifest(
    manifest: Mapping[str, Any], *, expected_model_sha256: Any
) -> None:
    expected_sha = _sha256(expected_model_sha256, "source training model SHA")
    runtime = _mapping(manifest.get("runtime"), "training runtime")
    science = _mapping(manifest.get("science_boundary"), "training science boundary")
    source = _mapping(manifest.get("source"), "training source")
    diagnostics = _mapping(manifest.get("diagnostics"), "training diagnostics")
    fit = _mapping(manifest.get("fit_contract"), "training fit contract")
    for name in (
        "development_jsonl_sha256",
        "development_decision_sha256",
        "development_selector_receipt_sha256",
        "development_pass_freeze_sha256",
        "distillation_config_sha256",
        "candidate_model_sha256",
        "development_plan_sha256",
        "development_package_manifest_sha256",
        "development_source_package_sha256",
        "development_root_identity_sha256",
        "runtime_source_archive_sha256",
        "runtime_source_manifest_sha256",
        "runtime_source_closure_sha256",
        "runtime_semantic_closure_sha256",
        "runtime_requirements_sha256",
        "runtime_fingerprint_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "runtime_dependency_closure_sha256",
    ):
        _sha256(source.get(name), f"training source {name}")
    if (
        source.get("development_plan_sha256") != M43_ATTEMPT11_PLAN_SHA256
        or source.get("candidate_model_sha256")
        != ATTEMPT11_LAMBDA_MODEL_SHA256
        or source.get("runtime_requirements_sha256")
        != ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256
        or source.get("runtime_fingerprint_sha256")
        != ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or source.get("distillation_config_sha256")
        != ATTEMPT11_DISTILLATION_CONFIG_SHA256
        or fit
        != {
            "fit_mode": "full",
            "effective_iterations": ATTEMPT11_DISTILLATION_FULL_ITERATIONS,
            "states": ATTEMPT11_DISTILLATION_FULL_STATES,
            "folds": 5,
        }
        or diagnostics.get("states") != ATTEMPT11_DISTILLATION_FULL_STATES
        or not _valid_variable_training_diagnostics(diagnostics)
        or diagnostics.get("fold_counts")
        != {str(index): 40 for index in range(5)}
    ):
        raise ValueError("Attempt11 training runtime semantic binding changed")
    if (
        manifest.get("schema") != ATTEMPT11_DISTILLED_TRAINING_MANIFEST_SCHEMA
        or manifest.get("status") != "fit_complete_runtime_disabled"
        or manifest.get("model_schema") != HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA
        or manifest.get("feature_schema") != HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA
        or manifest.get("head_schema") != HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA
        or manifest.get("action_score_mode")
        != HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE
        or manifest.get("model_sha256") != expected_sha
        or runtime.get("safety_enabled") is not False
        or runtime.get("winner_frozen") is not False
        or runtime.get("activation_allowed") is not False
        or runtime.get("current_profile_mutated") is not False
        or runtime.get("runtime_source_frozen") is not True
        or runtime.get("runtime_requirements_sha256")
        != ATTEMPT11_RUNTIME_REQUIREMENTS_SHA256
        or runtime.get("runtime_fingerprint_sha256")
        != ATTEMPT11_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or science.get("teacher_values_are_realized_match_ev") is not False
        or science.get("audit50_fit_rows") != 0
        or science.get("threshold_sweep_performed") is not False
        or science.get("top1_accuracy_is_acceptance_gate") is not False
    ):
        raise ValueError("Attempt11 source training manifest contract changed")


def _valid_variable_training_diagnostics(diagnostics: Mapping[str, Any]) -> bool:
    """Bind every LightGBM state group to 0..12 candidates plus baseline."""

    raw_groups = diagnostics.get("group_sizes")
    if (
        not isinstance(raw_groups, list)
        or len(raw_groups) != ATTEMPT11_DISTILLATION_FULL_STATES
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= ATTEMPT11_PROPOSAL_ROWS_MAX
            for value in raw_groups
        )
    ):
        return False
    candidate_counts = [value - 1 for value in raw_groups]
    expected_histogram = {
        str(candidate_count): candidate_counts.count(candidate_count)
        for candidate_count in range(ATTEMPT11_PROPOSAL_ROWS_MAX)
    }
    return (
        diagnostics.get("rows") == sum(raw_groups)
        and diagnostics.get("candidate_count_min") == min(candidate_counts)
        and diagnostics.get("candidate_count_max") == max(candidate_counts)
        and diagnostics.get("candidate_count_histogram") == expected_histogram
    )


def _validate_runtime_dependency_semantics(
    runtime_semantic: Mapping[str, Any], dependencies: Mapping[str, Any]
) -> None:
    teacher = _mapping(runtime_semantic.get("teacher_contract"), "teacher contract")
    if teacher != {
        "schema": "hu_m43_attempt11_teacher_lineage_v1",
        "plan_sha256": M43_ATTEMPT11_PLAN_SHA256,
        "candidate_model_sha256": ATTEMPT11_LAMBDA_MODEL_SHA256,
    }:
        raise ValueError("Attempt11 runtime teacher semantic binding changed")
    expected = {
        "schema": dependencies["schema"],
        "source_model_manifest_sha256": dependencies[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": dependencies[
            "source_native_manifest_sha256"
        ],
        "model_count": dependencies["model_count"],
        "binary_count": dependencies["binary_count"],
    }
    if _mapping(
        runtime_semantic.get("runtime_dependencies"), "runtime dependencies"
    ) != expected:
        raise ValueError("Attempt11 runtime dependency semantic binding changed")


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"population record {number} is not an object")
            rows.append(value)
    return rows


def _load_mapping(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _mapping(value, label)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt11 {label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"Attempt11 {label} must be a sequence")
    return list(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Attempt11 {label} must be an integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Attempt11 {label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"Attempt11 {label} is invalid")
    return normalized


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_number(left: Any, right: float) -> bool:
    if isinstance(left, bool):
        return False
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--source-model", required=True, type=Path)
    freeze.add_argument("--training-manifest", required=True, type=Path)
    freeze.add_argument("--audit-decision", required=True, type=Path)
    freeze.add_argument("--audit-selector-receipt", required=True, type=Path)
    freeze.add_argument("--runtime-source-archive", required=True, type=Path)
    freeze.add_argument("--runtime-source-manifest", required=True, type=Path)
    freeze.add_argument("--runtime-dependency-root", required=True, type=Path)
    freeze.add_argument("--output-model", required=True, type=Path)
    freeze.add_argument("--output-runtime-freeze", required=True, type=Path)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--model", required=True, type=Path)
    preflight.add_argument("--training-manifest", required=True, type=Path)
    preflight.add_argument("--runtime-freeze", required=True, type=Path)
    preflight.add_argument("--population-plan", required=True, type=Path)
    preflight.add_argument("--runtime-source-archive", required=True, type=Path)
    preflight.add_argument("--runtime-source-manifest", required=True, type=Path)
    preflight.add_argument("--runtime-source-root", required=True, type=Path)
    preflight.add_argument("--runtime-dependency-root", required=True, type=Path)
    final = commands.add_parser("finalize")
    final.add_argument("--population-plan", required=True, type=Path)
    final.add_argument("--records", required=True, type=Path)
    final.add_argument("--evaluation", required=True, type=Path)
    final.add_argument("--merge-manifest", required=True, type=Path)
    final.add_argument("--model", required=True, type=Path)
    final.add_argument("--training-manifest", required=True, type=Path)
    final.add_argument("--runtime-freeze", required=True, type=Path)
    final.add_argument("--runtime-source-archive", required=True, type=Path)
    final.add_argument("--runtime-source-manifest", required=True, type=Path)
    final.add_argument("--runtime-source-root", required=True, type=Path)
    final.add_argument("--runtime-dependency-root", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_source_root = getattr(args, "runtime_source_root", None)
    if runtime_source_root is not None:
        validate_frozen_execution_modules(
            extracted_root=runtime_source_root,
            manifest=args.runtime_source_manifest,
            module_names=(
                "ofc_regular.hu_m43_attempt11_distilled_runtime",
                "ofc_regular.hu_m43_attempt11_distilled_model",
                "ofc_regular.train_hu_m43_attempt11_distilled",
                "ofc_regular.evaluate_hu_m4_population",
                "ofc_regular.merge_hu_m4_population_shards",
                "ofc_regular.validate_hu_m43_attempt11_acceptance",
            ),
        )
    if args.command == "freeze":
        result = freeze_attempt11_distilled_runtime(
            source_model_path=args.source_model,
            training_manifest_path=args.training_manifest,
            audit_decision_path=args.audit_decision,
            audit_selector_receipt_path=args.audit_selector_receipt,
            runtime_source_archive_path=args.runtime_source_archive,
            runtime_source_manifest_path=args.runtime_source_manifest,
            runtime_dependency_root=args.runtime_dependency_root,
            output_model_path=args.output_model,
            output_runtime_freeze_path=args.output_runtime_freeze,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = build_attempt11_population_preflight(
            model_path=args.model,
            training_manifest_path=args.training_manifest,
            runtime_freeze_path=args.runtime_freeze,
            population_plan_path=args.population_plan,
            runtime_source_archive_path=args.runtime_source_archive,
            runtime_source_manifest_path=args.runtime_source_manifest,
            runtime_source_root=args.runtime_source_root,
            runtime_dependency_root=args.runtime_dependency_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    plan = _load_mapping(args.population_plan, "population plan")
    records = _read_jsonl(args.records)
    evaluation = _load_mapping(args.evaluation, "evaluation")
    merge = _load_mapping(args.merge_manifest, "merge manifest")
    status = validate_attempt11_population_acceptance(
        evaluation=evaluation,
        records=records,
        population_plan=plan,
        merge_manifest=merge,
        model_path=args.model,
        training_manifest_path=args.training_manifest,
        runtime_freeze_path=args.runtime_freeze,
        runtime_source_archive_path=args.runtime_source_archive,
        runtime_source_manifest_path=args.runtime_source_manifest,
        runtime_source_root=args.runtime_source_root,
        runtime_dependency_root=args.runtime_dependency_root,
        source_hashes={
            "population_plan": _file_sha256(args.population_plan),
            "population_plan_path": str(args.population_plan),
            "records": _file_sha256(args.records),
            "evaluation": _file_sha256(args.evaluation),
            "merge_manifest": _file_sha256(args.merge_manifest),
        },
    )
    _write_new_json(args.output, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["status"] == "complete_go" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT11_POPULATION_ACCEPTANCE_STATUS_SCHEMA",
    "ATTEMPT11_POPULATION_PREFLIGHT_SCHEMA",
    "ATTEMPT11_RUNTIME_FREEZE_SCHEMA",
    "build_attempt11_population_preflight",
    "freeze_attempt11_distilled_runtime",
    "load_and_validate_attempt11_population_plan",
    "load_bound_attempt11_distilled_model",
    "validate_attempt11_population_acceptance",
]
