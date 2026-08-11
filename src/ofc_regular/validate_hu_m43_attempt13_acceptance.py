"""Freeze and accept the opt-in Attempt13 T1-second population policy.

Search values are never interpreted as match EV here.  A complete Go is
issued only after the fixed fresh population grid is rebuilt from played-hand
records and every runtime/source hash is re-opened.
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
from .freeze_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA,
    ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS,
)
from .hu_m43_attempt13_contract import (
    ATTEMPT13_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT13_PLAN_SHA256,
    load_and_validate_attempt13_plan,
)
from .hu_m43_attempt13_distilled_model import (
    BoundHuM43Attempt13DistilledModel,
    HU_M43_ATTEMPT13_DISTILLED_ACTION_SCORE_MODE,
    HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
    HU_M43_ATTEMPT13_DISTILLED_FEATURE_SCHEMA,
    HU_M43_ATTEMPT13_DISTILLED_HEAD_SCHEMA,
    HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
    HuM43Attempt13DistilledModel,
)
from .hu_m43_attempt13_distilled_runtime import (
    ATTEMPT13_BOUND_EXECUTION_MODULES,
    ATTEMPT13_DISTILLED_RUNTIME_REGISTRY_PATHS,
    ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT13_GCP_IMAGE_ID,
    ATTEMPT13_GCP_IMAGE_NAME,
    ATTEMPT13_GCP_IMAGE_SELF_LINK,
    ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
    validate_distilled_runtime_dependencies,
    validate_distilled_runtime_extracted_tree,
    validate_distilled_runtime_source_archive,
    validate_frozen_execution_modules,
)
from .merge_hu_m4_population_shards import M4_POPULATION_MERGE_SCHEMA
from .select_hu_m43_attempt13_audit50 import (
    ATTEMPT13_AUDIT50_DECISION_SCHEMA,
    ATTEMPT13_AUDIT50_RECEIPT_KEYS,
    ATTEMPT13_AUDIT50_RECEIPT_SCHEMA,
)
from .select_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
)
from .train_hu_m43_attempt13_distilled import (
    ATTEMPT13_DISTILLATION_CONFIG_SHA256,
    ATTEMPT13_DISTILLATION_FULL_ITERATIONS,
    ATTEMPT13_DISTILLATION_FULL_STATES,
    ATTEMPT13_DISTILLED_TRAINING_MANIFEST_SCHEMA,
    ATTEMPT13_PROPOSAL_ROWS_MAX,
)
from .validate_hu_m43_attempt02_acceptance import evaluate_attempt02_population_gates


ATTEMPT13_POPULATION_PLAN_SCHEMA = "hu_m43_population_acceptance_plan_v1"
ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA = (
    "hu_m43_attempt13_population_launch_preflight_v1"
)
ATTEMPT13_RUNTIME_FREEZE_SCHEMA = "hu_m43_attempt13_distilled_runtime_freeze_v1"
ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA = (
    "hu_m43_attempt13_population_acceptance_status_v1"
)
ATTEMPT13_AUDIT50_SELECTOR_RECEIPT_SCHEMA = ATTEMPT13_AUDIT50_RECEIPT_SCHEMA
ATTEMPT13_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA = ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA
ATTEMPT13_DEVELOPMENT_PASS_FREEZE_SCHEMA = ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA
ATTEMPT13_PROFILE_ID = "stage20_m4_attempt13"
ATTEMPT13_BASELINE_PROFILE = "stage19_p0"
ATTEMPT13_OPPONENTS = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "random_exact_final",
)
ATTEMPT13_POPULATION_SEED = 250_108_071_901
ATTEMPT13_POPULATION_SEED_STRIDE = 1_000_003
ATTEMPT13_POPULATION_SEEDS = 1000
ATTEMPT13_POPULATION_SHARDS = 20
ATTEMPT13_POPULATION_SEEDS_PER_SHARD = 50
ATTEMPT13_POPULATION_PLAN_SHA256 = (
    "bf05a9477e18049883926537b4fb15c7e3ecf350c212072aed7c5b981d593a17"
)
ATTEMPT13_POPULATION_NAMESPACE_BASES = (
    250_108_071_901,
    251_108_071_901,
    252_108_071_901,
    253_108_071_901,
    254_108_071_901,
    255_108_071_901,
    256_108_071_901,
)
ATTEMPT13_EXCLUDED_NAMESPACE_BASES = (
    230_108_071_901,
    231_108_071_901,
    232_108_071_901,
    233_108_071_901,
    234_108_071_901,
    235_108_071_901,
    236_108_071_901,
    240_108_071_901,
    241_108_071_901,
    242_108_071_901,
    243_108_071_901,
    244_108_071_901,
    245_108_071_901,
    246_108_071_901,
)
ATTEMPT13_FIXED_GATES = {
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
ATTEMPT13_SEED_REGISTRY_SOURCES = ATTEMPT13_DISTILLED_RUNTIME_REGISTRY_PATHS
ATTEMPT13_SEARCH_PLAN_REGISTRY_PATH = "configs/hu_joint_policy_m43_attempt13.json"
_REQUIRED_REGISTRY_SOURCES = frozenset(
    {
        ATTEMPT13_SEARCH_PLAN_REGISTRY_PATH,
        "configs/hu_joint_policy_m43_attempt11_population.json",
        "configs/hu_joint_policy_m43_attempt12.json",
    }
)


def load_and_validate_attempt13_population_plan(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        plan = dict(source)
    else:
        if _file_sha256(source) != ATTEMPT13_POPULATION_PLAN_SHA256:
            raise ValueError("Attempt13 frozen population plan SHA changed")
        plan = _load_mapping(source, "population plan")
    if (
        plan.get("schema") != ATTEMPT13_POPULATION_PLAN_SCHEMA
        or plan.get("milestone") != "M4.3-attempt13-final-population"
        or plan.get("status") != "frozen_before_population_evaluation"
        or plan.get("purpose")
        != "fresh_realized_match_ev_acceptance_for_attempt13_distilled_t1_second_selector"
        or plan.get("policy_attempt") != "attempt13_distilled_v1"
        or plan.get("profile_id") != ATTEMPT13_PROFILE_ID
        or plan.get("search_plan_sha256") != M43_ATTEMPT13_PLAN_SHA256
        or plan.get("fixed_baseline_profile") != ATTEMPT13_BASELINE_PROFILE
        or plan.get("opponents") != list(ATTEMPT13_OPPONENTS)
        or plan.get("paired_seat_swap") is not True
    ):
        raise ValueError("Attempt13 population plan identity changed")
    if plan.get("baseline_chain") != [
        "stage19_p0",
        "stage18_p1",
        "stage9f_p2",
        "stage7_m5_r10",
        "exact_t4",
    ]:
        raise ValueError("Attempt13 population baseline chain changed")
    runtime = _mapping(plan.get("runtime_contract"), "runtime contract")
    if runtime != {
        "model_schema": HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
        "artifact_schema": HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
        "feature_schema": HU_M43_ATTEMPT13_DISTILLED_FEATURE_SCHEMA,
        "head_schema": HU_M43_ATTEMPT13_DISTILLED_HEAD_SCHEMA,
        "action_score_mode": HU_M43_ATTEMPT13_DISTILLED_ACTION_SCORE_MODE,
        "candidate_nonbaseline_count_range": [0, 26],
        "lightgbm_group_size_range": [1, 27],
        "complete_legal_action_set_rebuilt": True,
        "candidate_padding_allowed": False,
        "candidate_duplication_allowed": False,
        "baseline_appended_exactly_once": True,
        "fixed_safe_probability_threshold": 0.5,
        "fixed_fold_votes_min": 4,
        "folds": 5,
        "runtime_teacher_inputs": False,
        "runtime_teacher_ev_lcb_gate": False,
        "runtime_opponent_private_discard_input": False,
        "runtime_opponent_profile_feature": False,
    }:
        raise ValueError("Attempt13 population runtime contract changed")
    schedule = (
        _integer(plan.get("paired_seeds_per_opponent"), "paired seeds"),
        _integer(plan.get("seed"), "seed"),
        _integer(plan.get("seed_stride"), "seed stride"),
        _integer(plan.get("shards"), "shards"),
        _integer(plan.get("paired_seeds_per_shard"), "seeds per shard"),
    )
    if schedule != (
        ATTEMPT13_POPULATION_SEEDS,
        ATTEMPT13_POPULATION_SEED,
        ATTEMPT13_POPULATION_SEED_STRIDE,
        ATTEMPT13_POPULATION_SHARDS,
        ATTEMPT13_POPULATION_SEEDS_PER_SHARD,
    ):
        raise ValueError("Attempt13 population schedule changed")
    if (
        plan.get("candidate_records") != 8000
        or plan.get("baseline_records") != 8000
        or plan.get("terminal_trace_hands") != 16000
        or plan.get("second_seat_override_opportunities") != 4000
        or plan.get("minimum_valid_overrides") != 300
        or not _same_number(plan.get("minimum_population_fire_rate_needed"), 0.075)
    ):
        raise ValueError("Attempt13 population power contract changed")
    namespace = _mapping(plan.get("population_rng_namespaces"), "RNG namespaces")
    if namespace != {
        "hand_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[0],
        "rerank_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[1],
        "veto_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[2],
        "stress_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[3],
        "confirmation_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[4],
        "evaluation_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[5],
        "child_policy_seed_base": ATTEMPT13_POPULATION_NAMESPACE_BASES[6],
        "paired_seed_indices": "0..999_inclusive",
        "per_seed_formula": "namespace_seed_base + seed_stride * paired_seed_index",
        "candidate_and_baseline_common_randomness": True,
        "candidate_selection_and_evaluation_rng_independent": True,
    }:
        raise ValueError("Attempt13 population RNG namespace contract changed")
    if _mapping(plan.get("sizing_rule"), "sizing rule") != {
        "formula": "paired_seeds_required = ceil(minimum_valid_overrides / (opponents * independent_fire_rate_lower_bound))",
        "fixed_before_realized_population_labels": True,
        "optional_stopping_or_posthoc_extension_allowed": False,
        "insufficient_overrides_decision": "complete_no_go",
    }:
        raise ValueError("Attempt13 optional-stopping contract changed")
    if _mapping(plan.get("evaluation_contract"), "evaluation contract") != {
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
        raise ValueError("Attempt13 counterfactual contract changed")
    if _mapping(plan.get("fixed_acceptance_gates"), "acceptance gates") != ATTEMPT13_FIXED_GATES:
        raise ValueError("Attempt13 fixed acceptance gates changed")
    if _mapping(plan.get("spot_execution"), "Spot contract") != {
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
        raise ValueError("Attempt13 Spot execution contract changed")
    guards = _mapping(plan.get("activation_guards"), "activation guards")
    if set(guards) != {
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "threshold_changed_after_lock",
    } or any(value is not False for value in guards.values()):
        raise ValueError("Attempt13 activation guard changed")
    if _mapping(plan.get("post_acceptance_activation"), "post acceptance") != {
        "population_complete_go_required": True,
        "activation_mode_if_go": "explicit_opt_in_only",
        "automatic_activation_allowed": False,
        "current_profile_change_allowed": False,
    }:
        raise ValueError("Attempt13 post-acceptance activation changed")
    _validate_attempt13_freshness(plan)
    return plan


def _validate_attempt13_freshness(plan: dict[str, Any]) -> None:
    freshness = _mapping(plan.get("freshness"), "freshness")
    for name in (
        "exclude_all_m4_m41_m42_m43_teacher_hand_seeds",
        "exclude_attempt13_preflight_development_and_audit_namespaces",
        "exclude_prior_population_smoke_and_frozen_schedules",
        "acceptance_validator_rechecks_all_declared_seed_schedules",
    ):
        if freshness.get(name) is not True:
            raise ValueError("Attempt13 population freshness contract changed")
    if (
        freshness.get("planned_overlap_count_at_freeze") != 0
        or freshness.get("alternate_seed_after_result_allowed") is not False
    ):
        raise ValueError("Attempt13 population seed selection changed")
    sources = tuple(
        str(value)
        for value in _sequence(
            freshness.get("excluded_schedule_registry_sources"),
            "seed registry sources",
        )
    )
    if sources != ATTEMPT13_SEED_REGISTRY_SOURCES:
        raise ValueError("Attempt13 seed registry source list changed")
    if not _REQUIRED_REGISTRY_SOURCES.issubset(sources):
        raise ValueError("Attempt13 required seed registry source is missing")
    registry_seeds, registry_rows = _load_seed_registry(sources)
    all_reserved = {
        base + index * ATTEMPT13_POPULATION_SEED_STRIDE
        for base in ATTEMPT13_POPULATION_NAMESPACE_BASES
        for index in range(ATTEMPT13_POPULATION_SEEDS)
    }
    if all_reserved & registry_seeds:
        raise ValueError("Attempt13 population namespace overlaps hashed registry")
    digest = hashlib.sha256(_canonical(registry_rows).encode("utf-8")).hexdigest()
    if freshness.get("excluded_schedule_registry_sha256") != digest:
        raise ValueError("Attempt13 hashed seed registry snapshot changed")
    excluded_bases = _sequence(
        freshness.get("excluded_attempt13_namespace_bases"), "namespace bases"
    )
    if excluded_bases != list(ATTEMPT13_EXCLUDED_NAMESPACE_BASES):
        raise ValueError("Attempt13 excluded namespace contract changed")
    prior: set[int] = set()
    prior_rows = _sequence(
        freshness.get("excluded_population_schedules"), "prior populations"
    )
    for row in prior_rows:
        schedule = _mapping(row, "prior population")
        base = _integer(schedule.get("seed"), "prior seed")
        stride = _integer(schedule.get("seed_stride"), "prior stride")
        count = _integer(schedule.get("paired_seeds"), "prior paired seeds")
        prior.update(base + index * stride for index in range(count))
    required_prior = {160_108_071_901, 180_108_071_901, 190_108_071_901, 220_108_071_901}
    declared_prior = {
        row.get("seed") for row in prior_rows if isinstance(row, Mapping)
    }
    if not required_prior.issubset(declared_prior) or all_reserved & prior:
        raise ValueError("Attempt13 prior population exclusion changed")
    excluded = {
        base + index * ATTEMPT13_POPULATION_SEED_STRIDE
        for base in ATTEMPT13_EXCLUDED_NAMESPACE_BASES
        for index in range(250)
    }
    if all_reserved & excluded:
        raise ValueError("Attempt13 population overlaps development evidence")
    plan["_freshness_counts"] = {
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "all_reserved_registry_overlap_count": 0,
    }
    plan["_freshness_registry"] = {
        "schema": "hu_m43_attempt13_seed_registry_snapshot_v1",
        "source_count": len(registry_rows),
        "seed_count": len(registry_seeds),
        "planned_overlap_count": 0,
        "sha256": digest,
        "sources": registry_rows,
    }


def _load_seed_registry(
    relative_sources: Sequence[str],
) -> tuple[set[int], list[dict[str, Any]]]:
    seeds: set[int] = set()
    rows: list[dict[str, Any]] = []
    root = _REPO_ROOT.resolve()
    for relative in relative_sources:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Attempt13 seed registry path escapes repository")
        source = (root / candidate).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("Attempt13 seed registry path escapes repository") from exc
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Attempt13 seed registry source missing: {relative}")
        payload = _load_mapping(source, f"seed registry {relative}")
        source_seeds = _extract_declared_seed_schedules(payload)
        reservation_count = 0
        if relative == ATTEMPT13_SEARCH_PLAN_REGISTRY_PATH:
            search_plan = load_and_validate_attempt13_plan(source)
            reservation = _mapping(
                search_plan.get("population_seed_reservation"),
                "population reservation",
            )
            expected_keys = (
                "hand_seed_base",
                "rerank_seed_base",
                "veto_seed_base",
                "stress_seed_base",
                "confirmation_seed_base",
                "evaluation_seed_base",
                "child_policy_seed_base",
            )
            if (
                tuple(reservation.get(key) for key in expected_keys)
                != ATTEMPT13_POPULATION_NAMESPACE_BASES
                or reservation.get("seed_stride") != ATTEMPT13_POPULATION_SEED_STRIDE
                or reservation.get("paired_seed_count") != ATTEMPT13_POPULATION_SEEDS
                or reservation.get("paired_seat_swap_required") is not True
                or reservation.get("authorized") is not False
                or reservation.get("content_opened") is not False
            ):
                raise ValueError("Attempt13 reserved population namespace changed")
            reserved = {
                base + index * ATTEMPT13_POPULATION_SEED_STRIDE
                for base in ATTEMPT13_POPULATION_NAMESPACE_BASES
                for index in range(ATTEMPT13_POPULATION_SEEDS)
            }
            if not reserved.issubset(source_seeds):
                raise ValueError("Attempt13 population reservation was not indexed")
            source_seeds.difference_update(reserved)
            reservation_count = len(reserved)
        seeds.update(source_seeds)
        row = {
            "path": candidate.as_posix(),
            "sha256": _file_sha256(source),
            "seed_count": len(source_seeds),
            "seed_set_sha256": hashlib.sha256(
                _canonical(sorted(source_seeds)).encode("utf-8")
            ).hexdigest(),
        }
        if reservation_count:
            row["authorized_self_reservation_seed_count"] = reservation_count
        rows.append(row)
    return seeds, rows


def _extract_declared_seed_schedules(payload: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()

    def visit(value: Any, inherited_stride: int = ATTEMPT13_POPULATION_SEED_STRIDE) -> None:
        if isinstance(value, Mapping):
            local = dict(value)
            raw_stride = local.get("seed_stride", inherited_stride)
            stride = (
                raw_stride
                if isinstance(raw_stride, int)
                and not isinstance(raw_stride, bool)
                and raw_stride > 0
                else inherited_stride
            )
            counts = [
                local.get(name)
                for name in (
                    "paired_seeds_per_opponent",
                    "paired_seeds",
                    "paired_seed_count",
                    "roots",
                    "count",
                    "candidate_evaluation_child_seed_count",
                )
            ]
            count = next(
                (
                    int(item)
                    for item in counts
                    if isinstance(item, int)
                    and not isinstance(item, bool)
                    and item > 0
                ),
                1,
            )
            explicit = local.get("allowed_source_root_indices")
            if isinstance(explicit, Sequence) and not isinstance(
                explicit, (str, bytes, bytearray)
            ):
                indices = [
                    item
                    for item in explicit
                    if isinstance(item, int)
                    and not isinstance(item, bool)
                    and item >= 0
                ]
                if indices:
                    count = max(count, max(indices) + 1)
            range_max = 0
            for item in local.values():
                if isinstance(item, str) and ".." in item and "inclusive" in item:
                    digits = [
                        int(token)
                        for token in item.replace("..", " ").split()
                        if token.isdigit()
                    ]
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
                schedule_count = max(count, range_max, 1000) if key.endswith("_seed_base") else count
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


def freeze_attempt13_distilled_runtime(
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
    """Enable only an Audit50-Go artifact; no profile is registered here."""

    source_sha = _file_sha256(source_model_path)
    training_sha = _file_sha256(training_manifest_path)
    audit_sha = _file_sha256(audit_decision_path)
    audit_receipt_sha = _file_sha256(audit_selector_receipt_path)
    runtime_archive_sha = _file_sha256(runtime_source_archive_path)
    runtime_manifest_sha = _file_sha256(runtime_source_manifest_path)
    training = _load_mapping(training_manifest_path, "training manifest")
    audit = _load_mapping(audit_decision_path, "Audit50 decision")
    audit_receipt = _load_mapping(
        audit_selector_receipt_path, "Audit50 selector receipt"
    )
    _validate_training_manifest(training, expected_model_sha256=source_sha)
    runtime_source = validate_distilled_runtime_source_archive(
        archive_path=runtime_source_archive_path,
        manifest=runtime_source_manifest_path,
    )
    file_set = _mapping(runtime_source.get("file_set"), "runtime file set")
    semantic = _mapping(runtime_source.get("semantic_closure"), "runtime semantic")
    external = _mapping(semantic.get("external_runtime"), "external runtime")
    dependencies = validate_distilled_runtime_dependencies(runtime_dependency_root)
    _validate_runtime_dependency_semantics(semantic, dependencies)
    expected_bindings = {
        "runtime_source_archive_sha256": runtime_archive_sha,
        "runtime_source_manifest_sha256": runtime_manifest_sha,
        "runtime_source_closure_sha256": file_set["sha256"],
        "runtime_semantic_closure_sha256": semantic["sha256"],
        "runtime_requirements_sha256": ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": dependencies["source_model_manifest_sha256"],
        "source_native_manifest_sha256": dependencies["source_native_manifest_sha256"],
        "runtime_dependency_closure_sha256": dependencies["sha256"],
    }
    training_source = _mapping(training.get("source"), "training source")
    if any(training_source.get(name) != value for name, value in expected_bindings.items()):
        raise ValueError("Attempt13 training/runtime source freeze mismatch")
    if (
        external.get("requirements_sha256") != ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256
        or external.get("runtime_fingerprint_sha256")
        != ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256
    ):
        raise ValueError("Attempt13 external runtime identity changed")
    _validate_attempt13_audit50_go_authorization(
        audit,
        audit_receipt,
        audit_decision_sha256=audit_sha,
    )
    source_model = HuM43Attempt13DistilledModel.load(
        source_model_path, expected_sha256=source_sha
    )
    if source_model.safety_enabled or source_model.winner_frozen:
        raise ValueError("Attempt13 source model is already runtime frozen")
    source_fit = _mapping(source_model.manifest, "source model manifest")
    expected_fit = {
        "training_data": "attempt13_development200_only",
        "audit50_fit_rows": 0,
        "threshold_sweep_performed": False,
        "identity_grouped_folds": 5,
        "fit_mode": "full",
        "effective_iterations": ATTEMPT13_DISTILLATION_FULL_ITERATIONS,
        "training_states": ATTEMPT13_DISTILLATION_FULL_STATES,
        "distillation_config_sha256": ATTEMPT13_DISTILLATION_CONFIG_SHA256,
    }
    if any(source_fit.get(name) != value for name, value in expected_fit.items()):
        raise ValueError("Attempt13 source model is not the frozen full fit")
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
            **expected_bindings,
            "threshold_sweep_performed": False,
            "audit50_fit_rows": 0,
        },
    )
    final_sha = frozen_model.save(output_model_path)
    freeze = {
        "schema": ATTEMPT13_RUNTIME_FREEZE_SCHEMA,
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
        **expected_bindings,
        "gcp_image": {
            "name": ATTEMPT13_GCP_IMAGE_NAME,
            "id": ATTEMPT13_GCP_IMAGE_ID,
            "self_link": ATTEMPT13_GCP_IMAGE_SELF_LINK,
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


def _validate_attempt13_audit50_receipt_for_freeze(
    receipt: Mapping[str, Any], *, audit_decision_sha256: str
) -> None:
    if (
        set(receipt) != ATTEMPT13_AUDIT50_RECEIPT_KEYS
        or receipt.get("schema") != ATTEMPT13_AUDIT50_SELECTOR_RECEIPT_SCHEMA
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
        raise ValueError("Attempt13 Audit50 receipt is not freeze authorization")


def _validate_attempt13_audit50_go_authorization(
    audit: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    audit_decision_sha256: str,
) -> None:
    _sha256(audit_decision_sha256, "Audit50 decision SHA")
    science = _mapping(audit.get("science_boundary"), "Audit50 science")
    if (
        audit.get("schema") != ATTEMPT13_AUDIT50_DECISION_SCHEMA
        or audit.get("status") != "go_attempt13_audit50_search_quality"
        or audit.get("decision") != "go"
        or audit.get("search_freeze_authorized") is not True
        or audit.get("selected_arm") is not None
        or audit.get("selected_threshold") is not None
        or science.get("fit_performed") is not False
        or science.get("threshold_selected") is not False
        or science.get("runtime_policy_activated") is not False
        or science.get("current_profile_mutated") is not False
        or science.get("teacher_values_are_realized_match_ev") is not False
    ):
        raise ValueError("Attempt13 runtime freeze requires one-shot Audit50 Go")
    gates = _sequence(audit.get("gates"), "Audit50 gates")
    contract = _mapping(audit.get("decision_contract"), "Audit50 contract")
    if (
        not gates
        or not all(isinstance(gate, Mapping) and gate.get("passed") is True for gate in gates)
        or contract.get("gate_evaluation_count") != 1
        or contract.get("single_frozen_search_architecture") is not True
        or contract.get("arm_selection_performed") is not False
        or contract.get("threshold_selection_performed") is not False
        or contract.get("all_gates_required") is not True
    ):
        raise ValueError("Attempt13 Audit50 Go contract changed")
    _validate_attempt13_audit50_receipt_for_freeze(
        receipt, audit_decision_sha256=audit_decision_sha256
    )


def load_bound_attempt13_distilled_model(
    path: str | Path,
    *,
    expected_sha256: str,
    runtime_freeze: str | Path | Mapping[str, Any],
    training_manifest_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
) -> BoundHuM43Attempt13DistilledModel:
    """Issue the sole bound Attempt13 runtime capability after full closure checks."""

    execution_attestation = validate_frozen_execution_modules(
        extracted_root=runtime_source_root,
        manifest=runtime_source_manifest_path,
        module_names=ATTEMPT13_BOUND_EXECUTION_MODULES,
    )
    actual_sha = _file_sha256(path)
    if actual_sha != _sha256(expected_sha256, "model SHA"):
        raise ValueError("Attempt13 bound model SHA mismatch")
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
    file_set = _mapping(runtime_source.get("file_set"), "runtime file set")
    semantic = _mapping(runtime_source.get("semantic_closure"), "runtime semantic")
    dependencies = validate_distilled_runtime_dependencies(runtime_dependency_root)
    _validate_runtime_dependency_semantics(semantic, dependencies)
    expected_freeze = {
        "schema": ATTEMPT13_RUNTIME_FREEZE_SCHEMA,
        "status": "frozen_after_development200_fit_and_audit50_go",
        "model_sha256": actual_sha,
        "training_manifest_sha256": training_sha,
        "safety_enabled": True,
        "winner_frozen": True,
        "fixed_safe_probability_threshold": 0.5,
        "fixed_fold_votes_min": 4,
        "threshold_reselection_performed": False,
        "current_profile_mutated": False,
        "runtime_source_manifest_sha256": runtime_manifest_sha,
        "runtime_source_closure_sha256": file_set.get("sha256"),
        "runtime_semantic_closure_sha256": semantic.get("sha256"),
        "runtime_source_archive_sha256": runtime_source.get("archive", {}).get("sha256"),
        "runtime_requirements_sha256": ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": dependencies["source_model_manifest_sha256"],
        "source_native_manifest_sha256": dependencies["source_native_manifest_sha256"],
        "runtime_dependency_closure_sha256": dependencies["sha256"],
        "gcp_image": {
            "name": ATTEMPT13_GCP_IMAGE_NAME,
            "id": ATTEMPT13_GCP_IMAGE_ID,
            "self_link": ATTEMPT13_GCP_IMAGE_SELF_LINK,
        },
    }
    if any(freeze.get(name) != value for name, value in expected_freeze.items()):
        raise ValueError("Attempt13 runtime freeze contract changed")
    _sha256(freeze.get("audit50_decision_sha256"), "Audit50 decision SHA")
    _sha256(freeze.get("audit50_selector_receipt_sha256"), "Audit50 receipt SHA")
    _validate_training_manifest(
        training, expected_model_sha256=freeze.get("source_training_model_sha256")
    )
    if training.get("model_id") != freeze.get("model_id"):
        raise ValueError("Attempt13 training/runtime freeze model mismatch")
    training_source = _mapping(training.get("source"), "training source")
    binding_names = (
        "runtime_source_archive_sha256",
        "runtime_source_manifest_sha256",
        "runtime_source_closure_sha256",
        "runtime_semantic_closure_sha256",
        "runtime_requirements_sha256",
        "runtime_fingerprint_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "runtime_dependency_closure_sha256",
    )
    if any(training_source.get(name) != freeze.get(name) for name in binding_names):
        raise ValueError("Attempt13 runtime source hash chain changed")
    model = HuM43Attempt13DistilledModel.load(path, expected_sha256=actual_sha)
    embedded = _mapping(model.manifest, "frozen model manifest")
    expected_fit = {
        "fit_mode": "full",
        "effective_iterations": ATTEMPT13_DISTILLATION_FULL_ITERATIONS,
        "training_states": ATTEMPT13_DISTILLATION_FULL_STATES,
        "identity_grouped_folds": 5,
        "distillation_config_sha256": ATTEMPT13_DISTILLATION_CONFIG_SHA256,
    }
    if any(embedded.get(name) != value for name, value in expected_fit.items()):
        raise ValueError("Attempt13 frozen model full-fit contract changed")
    embedded_names = (
        "audit50_decision_sha256",
        "audit50_selector_receipt_sha256",
        "training_manifest_sha256",
        *binding_names,
    )
    if any(embedded.get(name) != freeze.get(name) for name in embedded_names):
        raise ValueError("Attempt13 frozen model semantic binding changed")
    if (
        model.safety_enabled is not True
        or model.winner_frozen is not True
        or model.model_id != freeze.get("model_id")
        or model.schema != freeze.get("model_schema")
        or model.artifact_schema != freeze.get("artifact_schema")
        or model.feature_schema != freeze.get("feature_schema")
        or model.head_schema != freeze.get("head_schema")
        or model.action_score_mode != freeze.get("action_score_mode")
        or not _same_number(model.safety_threshold, 0.5)
        or model.minimum_fold_votes != 4
    ):
        raise ValueError("Attempt13 frozen model payload disagrees with freeze")
    return BoundHuM43Attempt13DistilledModel(model, execution_attestation)


def _validate_attempt13_population_provenance_bundle(
    *,
    training_manifest: Mapping[str, Any],
    runtime_freeze: Mapping[str, Any],
    development_decision_path: str | Path,
    development_selector_receipt_path: str | Path,
    development_pass_freeze_path: str | Path,
    audit_decision_path: str | Path,
    audit_selector_receipt_path: str | Path,
) -> dict[str, str]:
    """Re-open all five Go artifacts; hashes alone are not authorization."""

    training_source = _mapping(training_manifest.get("source"), "training source")
    paths: dict[str, str | Path] = {
        "development_decision_sha256": development_decision_path,
        "development_selector_receipt_sha256": development_selector_receipt_path,
        "development_pass_freeze_sha256": development_pass_freeze_path,
        "audit50_decision_sha256": audit_decision_path,
        "audit50_selector_receipt_sha256": audit_selector_receipt_path,
    }
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    for name in (
        "development_decision_sha256",
        "development_selector_receipt_sha256",
        "development_pass_freeze_sha256",
    ):
        if training_source.get(name) != hashes[name]:
            raise ValueError("Attempt13 development provenance bytes changed")
    for name in ("audit50_decision_sha256", "audit50_selector_receipt_sha256"):
        if runtime_freeze.get(name) != hashes[name]:
            raise ValueError("Attempt13 Audit50 provenance bytes changed")

    decision = _load_mapping(development_decision_path, "development decision")
    receipt = _load_mapping(
        development_selector_receipt_path, "development selector receipt"
    )
    pass_freeze = _load_mapping(development_pass_freeze_path, "development freeze")
    if (
        decision.get("schema") != ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA
        or decision.get("decision") != "go"
        or decision.get("search_freeze_authorized") is not True
        or receipt.get("schema") != ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("decision_sha256") != hashes["development_decision_sha256"]
        or receipt.get("decision") != "go"
        or receipt.get("search_freeze_authorized") is not True
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
        or receipt.get("fit_performed") is not False
        or receipt.get("threshold_selected") is not False
        or receipt.get("current_profile_mutated") is not False
        or receipt.get("runtime_policy_activated") is not False
        or pass_freeze.get("schema") != ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA
        or pass_freeze.get("status") != ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS
        or pass_freeze.get("decision") != "go"
        or pass_freeze.get("future_audit_authorized") is not False
        or pass_freeze.get("fit_performed") is not False
        or pass_freeze.get("threshold_selected") is not False
        or pass_freeze.get("current_profile_mutated") is not False
        or pass_freeze.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt13 development provenance is not a complete Go")
    bindings = _mapping(pass_freeze.get("bindings"), "development freeze bindings")
    if (
        bindings.get("plan_sha256") != M43_ATTEMPT13_PLAN_SHA256
        or bindings.get("selector_decision_sha256")
        != hashes["development_decision_sha256"]
        or bindings.get("selector_receipt_sha256")
        != hashes["development_selector_receipt_sha256"]
        or bindings.get("merged_input_sha256")
        != training_source.get("development_jsonl_sha256")
    ):
        raise ValueError("Attempt13 development freeze binding changed")
    audit = _load_mapping(audit_decision_path, "Audit50 decision")
    audit_receipt = _load_mapping(audit_selector_receipt_path, "Audit50 receipt")
    _validate_attempt13_audit50_go_authorization(
        audit,
        audit_receipt,
        audit_decision_sha256=hashes["audit50_decision_sha256"],
    )
    return hashes


def build_attempt13_population_preflight(
    *,
    model_path: str | Path,
    training_manifest_path: str | Path,
    runtime_freeze_path: str | Path,
    population_plan_path: str | Path,
    runtime_source_archive_path: str | Path,
    runtime_source_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
    development_decision_path: str | Path | None = None,
    development_selector_receipt_path: str | Path | None = None,
    development_pass_freeze_path: str | Path | None = None,
    audit_decision_path: str | Path | None = None,
    audit_selector_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    plan = load_and_validate_attempt13_population_plan(population_plan_path)
    counts = _mapping(plan.pop("_freshness_counts"), "freshness counts")
    registry = _mapping(plan.pop("_freshness_registry"), "freshness registry")
    runtime_source = validate_distilled_runtime_source_archive(
        archive_path=runtime_source_archive_path,
        manifest=runtime_source_manifest_path,
    )
    model_sha = _file_sha256(model_path)
    model = load_bound_attempt13_distilled_model(
        model_path,
        expected_sha256=model_sha,
        runtime_freeze=runtime_freeze_path,
        training_manifest_path=training_manifest_path,
        runtime_source_manifest_path=runtime_source_manifest_path,
        runtime_source_root=runtime_source_root,
        runtime_dependency_root=runtime_dependency_root,
    )
    training = _load_mapping(training_manifest_path, "training manifest")
    training_source = _mapping(training.get("source"), "training source")
    runtime_freeze = _load_mapping(runtime_freeze_path, "runtime freeze")
    provenance_paths = (
        development_decision_path,
        development_selector_receipt_path,
        development_pass_freeze_path,
        audit_decision_path,
        audit_selector_receipt_path,
    )
    if all(path is not None for path in provenance_paths):
        provenance = _validate_attempt13_population_provenance_bundle(
            training_manifest=training,
            runtime_freeze=runtime_freeze,
            development_decision_path=development_decision_path,
            development_selector_receipt_path=development_selector_receipt_path,
            development_pass_freeze_path=development_pass_freeze_path,
            audit_decision_path=audit_decision_path,
            audit_selector_receipt_path=audit_selector_receipt_path,
        )
    elif any(path is not None for path in provenance_paths):
        raise ValueError("Attempt13 provenance bundle must be complete")
    else:
        provenance = {
            name: _sha256(source.get(name), name)
            for name, source in (
                ("development_decision_sha256", training_source),
                ("development_selector_receipt_sha256", training_source),
                ("development_pass_freeze_sha256", training_source),
                ("audit50_decision_sha256", runtime_freeze),
                ("audit50_selector_receipt_sha256", runtime_freeze),
            )
        }
    return {
        "schema": ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "profile_id": ATTEMPT13_PROFILE_ID,
        "search_plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "model_schema": model.schema,
        "artifact_schema": model.artifact_schema,
        "feature_schema": model.feature_schema,
        "head_schema": model.head_schema,
        "action_score_mode": model.action_score_mode,
        "baseline_profile": ATTEMPT13_BASELINE_PROFILE,
        "opponents": list(ATTEMPT13_OPPONENTS),
        "runtime_binding_verified": True,
        "model_sha256": model_sha,
        "model_id": model.model_id,
        "training_manifest_sha256": _file_sha256(training_manifest_path),
        "runtime_freeze_sha256": _file_sha256(runtime_freeze_path),
        "runtime_source_archive_sha256": _file_sha256(runtime_source_archive_path),
        "runtime_source_manifest_sha256": _file_sha256(runtime_source_manifest_path),
        "runtime_source_closure_sha256": runtime_source["file_set"]["sha256"],
        "runtime_semantic_closure_sha256": runtime_source["semantic_closure"]["sha256"],
        "runtime_requirements_sha256": ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": model.manifest["source_model_manifest_sha256"],
        "source_native_manifest_sha256": model.manifest["source_native_manifest_sha256"],
        "runtime_dependency_closure_sha256": model.manifest[
            "runtime_dependency_closure_sha256"
        ],
        "population_plan_file_sha256": _file_sha256(population_plan_path),
        "population_namespace_bases": list(ATTEMPT13_POPULATION_NAMESPACE_BASES),
        "fixed_safe_probability_threshold": model.safety_threshold,
        "fixed_fold_votes_min": model.minimum_fold_votes,
        "teacher_overlap_count": counts["teacher_overlap_count"],
        "prior_population_overlap_count": counts["prior_population_overlap_count"],
        "all_reserved_registry_overlap_count": counts[
            "all_reserved_registry_overlap_count"
        ],
        "seed_registry_sha256": registry["sha256"],
        "seed_registry_source_count": registry["source_count"],
        "seed_registry_seed_count": registry["seed_count"],
        "seed_registry_sources": registry["sources"],
        **provenance,
        "development200_full_fit_bound": True,
        "audit50_one_shot_go_bound": True,
        "audit50_fit_rows": 0,
        "threshold_reselection_performed": False,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }


def validate_attempt13_population_acceptance(
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
        raise ValueError("Attempt13 population plan path/hash mismatch")
    file_plan = _load_mapping(plan_path, "population plan file")
    if _canonical(file_plan) != _canonical(population_plan):
        raise ValueError("Attempt13 population plan mapping/file mismatch")
    for name in ("records", "evaluation", "merge_manifest"):
        _sha256(source_hashes[name], f"{name} SHA")
    plan = load_and_validate_attempt13_population_plan(population_plan)
    plan.pop("_freshness_counts", None)
    plan.pop("_freshness_registry", None)
    preflight = build_attempt13_population_preflight(
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
        opponents=ATTEMPT13_OPPONENTS,
        paired_seeds=ATTEMPT13_POPULATION_SEEDS,
        seed=ATTEMPT13_POPULATION_SEED,
        seed_stride=ATTEMPT13_POPULATION_SEED_STRIDE,
        baseline_profile=ATTEMPT13_BASELINE_PROFILE,
    )
    content_match = _canonical(_summary_for_comparison(evaluation)) == _canonical(
        _summary_for_comparison(recomputed)
    )
    plan_sha = source_hashes["population_plan"]
    runtime = _mapping(evaluation.get("runtime_config"), "runtime config")
    runtime_match = (
        runtime.get("candidate_model_sha256") == preflight["model_sha256"]
        and runtime.get("safety_model_sha256") == preflight["model_sha256"]
        and runtime.get("model_schema") == HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA
        and runtime.get("artifact_schema") == HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA
        and runtime.get("feature_schema") == HU_M43_ATTEMPT13_DISTILLED_FEATURE_SCHEMA
        and runtime.get("head_schema") == HU_M43_ATTEMPT13_DISTILLED_HEAD_SCHEMA
        and runtime.get("model_id") == preflight["model_id"]
        and runtime.get("action_score_mode") == HU_M43_ATTEMPT13_DISTILLED_ACTION_SCORE_MODE
        and runtime.get("profile_id") == ATTEMPT13_PROFILE_ID
        and runtime.get("baseline_profile") == ATTEMPT13_BASELINE_PROFILE
        and runtime.get("opponents") == list(ATTEMPT13_OPPONENTS)
        and runtime.get("runtime_binding_verified") is True
        and runtime.get("freeze_manifest_sha256") == preflight["runtime_freeze_sha256"]
        and runtime.get("training_manifest_sha256") == preflight["training_manifest_sha256"]
        and runtime.get("runtime_source_manifest_sha256")
        == preflight["runtime_source_manifest_sha256"]
        and runtime.get("runtime_source_closure_sha256")
        == preflight["runtime_source_closure_sha256"]
        and runtime.get("runtime_semantic_closure_sha256")
        == preflight["runtime_semantic_closure_sha256"]
        and runtime.get("runtime_requirements_sha256")
        == ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256
        and runtime.get("runtime_fingerprint_sha256")
        == ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        and runtime.get("source_model_manifest_sha256")
        == preflight["source_model_manifest_sha256"]
        and runtime.get("source_native_manifest_sha256")
        == preflight["source_native_manifest_sha256"]
        and runtime.get("runtime_dependency_closure_sha256")
        == preflight["runtime_dependency_closure_sha256"]
        and runtime.get("seed_registry_sha256") == preflight["seed_registry_sha256"]
        and runtime.get("population_namespace_bases")
        == list(ATTEMPT13_POPULATION_NAMESPACE_BASES)
        and runtime.get("current_profile_used") is False
        and runtime.get("promotion_artifact_contract") is True
        and runtime.get("diagnostic_legacy") is False
        and runtime.get("safety_enabled") is True
        and _same_number(runtime.get("safety_threshold"), 0.5)
        and runtime.get("population_plan_sha256") == plan_sha
        and runtime.get("sharded_evaluation") is True
        and runtime.get("shard_count") == ATTEMPT13_POPULATION_SHARDS
        and runtime.get("final_metrics_recomputed_from_merged_records") is True
        and all(row.get("runtime_binding_verified") is True for row in records)
    )
    merge_match = (
        merge_manifest.get("schema") == M4_POPULATION_MERGE_SCHEMA
        and merge_manifest.get("status") == "complete_content_verified"
        and merge_manifest.get("population_plan_sha256") == plan_sha
        and merge_manifest.get("seed") == ATTEMPT13_POPULATION_SEED
        and merge_manifest.get("seed_stride") == ATTEMPT13_POPULATION_SEED_STRIDE
        and merge_manifest.get("paired_seeds_per_opponent") == ATTEMPT13_POPULATION_SEEDS
        and merge_manifest.get("opponents") == list(ATTEMPT13_OPPONENTS)
        and merge_manifest.get("merged_records") == 8000
        and merge_manifest.get("current_profile_used") is False
        and merge_manifest.get("metrics_recomputed_from_merged_seed_clusters") is True
        and isinstance(merge_manifest.get("shards"), Sequence)
        and not isinstance(merge_manifest.get("shards"), (str, bytes, bytearray))
        and len(merge_manifest.get("shards")) == ATTEMPT13_POPULATION_SHARDS
    )
    gates = [
        _gate(
            "runtime_artifact_hash_chain",
            runtime_match,
            runtime_match,
            "bound Attempt13 artifact, training, freeze, source, and seed registry",
        ),
        _gate(
            "population_content_recomputed",
            content_match,
            content_match,
            "stored summary equals complete played-hand record recomputation",
        ),
        _gate(
            "strict_shard_merge",
            merge_match,
            merge_match,
            "fixed 20x50 merge and frozen population plan hash",
        ),
        *evaluate_attempt02_population_gates(recomputed),
    ]
    passed = sum(gate["passed"] is True for gate in gates)
    decision = "complete_go" if passed == len(gates) else "complete_no_go"
    return {
        "schema": ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
        "status": decision,
        "passed_gates": passed,
        "total_gates": len(gates),
        "gates": gates,
        "profile_id": ATTEMPT13_PROFILE_ID,
        "promotion_eligible": decision == "complete_go",
        "explicit_opt_in_authorized": decision == "complete_go",
        "automatic_activation_authorized": False,
        "population_plan_sha256": plan_sha,
        "search_plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "records_sha256": source_hashes["records"],
        "evaluation_sha256": source_hashes["evaluation"],
        "merge_manifest_sha256": source_hashes["merge_manifest"],
        "model_sha256": preflight["model_sha256"],
        "baseline_profile": ATTEMPT13_BASELINE_PROFILE,
        "opponents": list(ATTEMPT13_OPPONENTS),
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
    science = _mapping(manifest.get("science_boundary"), "science boundary")
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
        source.get("development_plan_sha256") != M43_ATTEMPT13_PLAN_SHA256
        or source.get("candidate_model_sha256") != ATTEMPT13_LAMBDA_MODEL_SHA256
        or source.get("runtime_requirements_sha256")
        != ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256
        or source.get("runtime_fingerprint_sha256")
        != ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or source.get("distillation_config_sha256")
        != ATTEMPT13_DISTILLATION_CONFIG_SHA256
        or fit
        != {
            "fit_mode": "full",
            "effective_iterations": ATTEMPT13_DISTILLATION_FULL_ITERATIONS,
            "states": ATTEMPT13_DISTILLATION_FULL_STATES,
            "folds": 5,
        }
        or diagnostics.get("states") != ATTEMPT13_DISTILLATION_FULL_STATES
        or not _valid_variable_training_diagnostics(diagnostics)
        or diagnostics.get("fold_counts") != {str(index): 40 for index in range(5)}
    ):
        raise ValueError("Attempt13 training runtime semantic binding changed")
    if (
        manifest.get("schema") != ATTEMPT13_DISTILLED_TRAINING_MANIFEST_SCHEMA
        or manifest.get("status") != "fit_complete_runtime_disabled"
        or manifest.get("model_schema") != HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA
        or manifest.get("feature_schema") != HU_M43_ATTEMPT13_DISTILLED_FEATURE_SCHEMA
        or manifest.get("head_schema") != HU_M43_ATTEMPT13_DISTILLED_HEAD_SCHEMA
        or manifest.get("action_score_mode")
        != HU_M43_ATTEMPT13_DISTILLED_ACTION_SCORE_MODE
        or manifest.get("model_sha256") != expected_sha
        or runtime.get("safety_enabled") is not False
        or runtime.get("winner_frozen") is not False
        or runtime.get("activation_allowed") is not False
        or runtime.get("current_profile_mutated") is not False
        or runtime.get("runtime_source_frozen") is not True
        or runtime.get("runtime_requirements_sha256")
        != ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256
        or runtime.get("runtime_fingerprint_sha256")
        != ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or science.get("teacher_values_are_realized_match_ev") is not False
        or science.get("audit50_fit_rows") != 0
        or science.get("threshold_sweep_performed") is not False
        or science.get("top1_accuracy_is_acceptance_gate") is not False
    ):
        raise ValueError("Attempt13 source training manifest contract changed")


def _valid_variable_training_diagnostics(diagnostics: Mapping[str, Any]) -> bool:
    raw_groups = diagnostics.get("group_sizes")
    if (
        not isinstance(raw_groups, list)
        or len(raw_groups) != ATTEMPT13_DISTILLATION_FULL_STATES
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= ATTEMPT13_PROPOSAL_ROWS_MAX
            for value in raw_groups
        )
    ):
        return False
    candidate_counts = [value - 1 for value in raw_groups]
    expected_histogram = {
        str(count): candidate_counts.count(count)
        for count in range(ATTEMPT13_PROPOSAL_ROWS_MAX)
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
        "schema": "hu_m43_attempt13_teacher_lineage_v1",
        "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "candidate_model_sha256": ATTEMPT13_LAMBDA_MODEL_SHA256,
    }:
        raise ValueError("Attempt13 runtime teacher semantic binding changed")
    expected = {
        "schema": dependencies["schema"],
        "source_model_manifest_sha256": dependencies["source_model_manifest_sha256"],
        "source_native_manifest_sha256": dependencies["source_native_manifest_sha256"],
        "model_count": dependencies["model_count"],
        "binary_count": dependencies["binary_count"],
    }
    if _mapping(runtime_semantic.get("runtime_dependencies"), "runtime dependencies") != expected:
        raise ValueError("Attempt13 runtime dependency semantic binding changed")


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
                raise ValueError(f"Attempt13 population record {number} is not an object")
            rows.append(value)
    return rows


def _load_mapping(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _mapping(value, label)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt13 {label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"Attempt13 {label} must be a sequence")
    return list(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Attempt13 {label} must be an integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Attempt13 {label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"Attempt13 {label} is invalid")
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
    freeze = commands.add_parser("freeze-runtime")
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
    for flag in (
        "model",
        "training-manifest",
        "runtime-freeze",
        "population-plan",
        "runtime-source-archive",
        "runtime-source-manifest",
        "runtime-source-root",
        "runtime-dependency-root",
        "development-decision",
        "development-selector-receipt",
        "development-pass-freeze",
        "audit-decision",
        "audit-selector-receipt",
    ):
        preflight.add_argument(f"--{flag}", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)

    final = commands.add_parser("finalize")
    for flag in (
        "evaluation",
        "records",
        "merge-manifest",
        "population-plan",
        "model",
        "training-manifest",
        "runtime-freeze",
        "runtime-source-archive",
        "runtime-source-manifest",
        "runtime-source-root",
        "runtime-dependency-root",
    ):
        final.add_argument(f"--{flag}", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze-runtime":
        result = freeze_attempt13_distilled_runtime(
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
        result = build_attempt13_population_preflight(
            model_path=args.model,
            training_manifest_path=args.training_manifest,
            runtime_freeze_path=args.runtime_freeze,
            population_plan_path=args.population_plan,
            runtime_source_archive_path=args.runtime_source_archive,
            runtime_source_manifest_path=args.runtime_source_manifest,
            runtime_source_root=args.runtime_source_root,
            runtime_dependency_root=args.runtime_dependency_root,
            development_decision_path=args.development_decision,
            development_selector_receipt_path=args.development_selector_receipt,
            development_pass_freeze_path=args.development_pass_freeze,
            audit_decision_path=args.audit_decision,
            audit_selector_receipt_path=args.audit_selector_receipt,
        )
        _write_new_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    evaluation = _load_mapping(args.evaluation, "evaluation")
    records = _read_jsonl(args.records)
    plan = _load_mapping(args.population_plan, "population plan")
    merge = _load_mapping(args.merge_manifest, "merge manifest")
    result = validate_attempt13_population_acceptance(
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
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete_go" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT13_BASELINE_PROFILE",
    "ATTEMPT13_FIXED_GATES",
    "ATTEMPT13_OPPONENTS",
    "ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA",
    "ATTEMPT13_POPULATION_NAMESPACE_BASES",
    "ATTEMPT13_POPULATION_PLAN_SCHEMA",
    "ATTEMPT13_POPULATION_PLAN_SHA256",
    "ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA",
    "ATTEMPT13_POPULATION_SEED",
    "ATTEMPT13_POPULATION_SEEDS",
    "ATTEMPT13_POPULATION_SEED_STRIDE",
    "ATTEMPT13_PROFILE_ID",
    "ATTEMPT13_RUNTIME_FREEZE_SCHEMA",
    "build_attempt13_population_preflight",
    "freeze_attempt13_distilled_runtime",
    "load_and_validate_attempt13_population_plan",
    "load_bound_attempt13_distilled_model",
    "validate_attempt13_population_acceptance",
]
