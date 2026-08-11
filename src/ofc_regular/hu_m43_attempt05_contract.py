"""Frozen lifecycle and RNG-domain contract for the M4.3 Attempt05 pilot.

Attempt05 is a new experiment.  Attempt04 development evidence may freeze the
family, but it is never reused as a fresh gate.  This module deliberately does
not generate data, open inherited holdouts, change ``current``, or activate a
runtime policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


M43_ATTEMPT05_PLAN_SCHEMA = "hu_m43_attempt05_plan_v1"
M43_ATTEMPT05_STATUS_SCHEMA = "hu_joint_policy_m43_attempt05_status_v1"
M43_ATTEMPT05_SEED_STRIDE = 1_000_003
M43_ATTEMPT05_ROOTS_PER_SHARD = 10
M43_ATTEMPT05_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
M43_ATTEMPT05_THRESHOLD_GRID = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
)
AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

_DEVELOPMENT_ROLES: dict[str, tuple[int, int, int, int, int]] = {
    "pilot_train": (200, 17_106_071_901, 23_106_071_901, 24_106_071_901, 25_106_071_901),
    "pilot_audit": (50, 17_306_071_901, 23_306_071_901, 24_306_071_901, 25_306_071_901),
    "expand_train": (600, 17_506_071_901, 23_506_071_901, 24_506_071_901, 25_506_071_901),
    "final_audit": (150, 17_706_071_901, 23_706_071_901, 24_706_071_901, 25_706_071_901),
}
_ACCEPTANCE_ROLES: dict[str, tuple[int, int, int, int, int]] = {
    "precal_holdout": (300, 13_106_071_901, 20_106_071_901, 21_106_071_901, 22_106_071_901),
    "calibration_safety_fit": (100, 13_506_071_901, 20_506_071_901, 21_506_071_901, 22_506_071_901),
    "calibration_threshold_lock": (100, 13_606_071_901, 20_606_071_901, 21_606_071_901, 22_606_071_901),
    "locked_holdout": (200, 13_806_071_901, 20_806_071_901, 21_806_071_901, 22_806_071_901),
}

_EXCLUDED_HAND = (
    ("m4.correctness_smoke", 2_026_071_401, 1),
    ("m4.train_a", 3_026_072_001, 32),
    ("m4.train_b", 4_026_072_001, 32),
    ("m4.calibration", 5_026_072_001, 24),
    ("m4.locked_holdout", 6_026_072_001, 24),
    ("m41.correctness_smoke", 9_026_072_001, 1),
    ("m41.train_a", 10_026_072_001, 25),
    ("m41.train_b", 20_026_072_001, 25),
    ("m41.calibration", 30_026_072_001, 30),
    ("m41.locked_holdout", 40_026_072_001, 20),
    ("m42.correctness_smoke", 50_026_072_001, 1),
    ("m42.train", 1_706_071_901, 20),
    ("m42.calibration", 1_806_071_901, 10),
    ("m42.locked_holdout", 1_906_071_901, 10),
    ("m43_attempt01.train", 2_306_071_901, 100),
    ("m43_attempt01.calibration", 2_506_071_901, 60),
    ("m43_attempt01.locked_holdout", 2_706_071_901, 40),
    ("m43_attempt02.train", 7_106_071_901, 200),
    ("m43_attempt02.calibration", 7_506_071_901, 100),
    ("m43_attempt03.train_fit", 9_106_071_901, 500),
    ("m43_attempt03.precal_holdout", 9_506_071_901, 200),
)
_EXCLUDED_RNG = (
    ("m43_attempt02.train.candidate", 7_906_071_901, 20),
    ("m43_attempt02.train.evaluation", 8_306_071_901, 20),
    ("m43_attempt02.train.child", 8_706_071_901, 20),
    ("m43_attempt02.calibration.candidate", 8_106_071_901, 10),
    ("m43_attempt02.calibration.evaluation", 8_506_071_901, 10),
    ("m43_attempt02.calibration.child", 8_906_071_901, 10),
    ("m43_attempt03.train_fit.candidate", 9_906_071_901, 50),
    ("m43_attempt03.train_fit.evaluation", 10_706_071_901, 50),
    ("m43_attempt03.train_fit.child", 11_506_071_901, 50),
    ("m43_attempt03.precal.candidate", 10_306_071_901, 20),
    ("m43_attempt03.precal.evaluation", 11_106_071_901, 20),
    ("m43_attempt03.precal.child", 11_906_071_901, 20),
)
_EXCLUDED_POPULATION = (
    ("m4.population", 7_026_072_001, 2),
    ("m41.population", 50_026_072_001, 2),
    ("m42.population", 2_126_071_901, 2),
    ("m43_attempt02.population", 6_106_071_901, 1000),
    ("m43_attempt03.population", 12_106_071_901, 1000),
)


def load_and_validate_attempt05_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("Attempt05 plan must be a mapping")
    validate_attempt05_plan(plan)
    return plan


def load_and_validate_attempt05_status(path: str | Path) -> dict[str, Any]:
    status = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(status, dict):
        raise ValueError("Attempt05 status must be a mapping")
    validate_attempt05_status(status)
    return status


def validate_attempt05_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != M43_ATTEMPT05_PLAN_SCHEMA:
        raise ValueError("unsupported Attempt05 plan schema")
    if (plan.get("milestone"), plan.get("status")) != (
        "M4.3-attempt05",
        "frozen_pre_pilot_generation",
    ):
        raise ValueError("Attempt05 plan is not frozen pre-pilot")
    if plan.get("fixed_baseline_profile") != "stage18_p1":
        raise ValueError("Attempt05 baseline changed")
    continuation = _mapping(plan.get("fixed_continuation"), "fixed_continuation")
    if dict(continuation) != {
        "t2_profile": "stage9f_p2",
        "t3_selector": "m3_rust_evaluate_t3",
        "t4_selector": "m3_rust_evaluate_t4",
    }:
        raise ValueError("Attempt05 fixed continuation changed")
    if tuple(plan.get("profiles", ())) != M43_ATTEMPT05_PROFILES:
        raise ValueError("Attempt05 five-profile population changed")

    old = _mapping(plan.get("prior_attempt04_development"), "prior Attempt04 dev900")
    expected_old = {
        "classification": "development_only_already_consumed_not_a_fresh_gate",
        "roots": 900,
        "eligible_count": 0,
        "allowed_use": "freeze_attempt05_family_before_new_pilot_only",
        "acceptance_or_generalization_claim_allowed": False,
        "threshold_reselection_allowed": False,
    }
    for key, expected in expected_old.items():
        if old.get(key) != expected:
            raise ValueError(f"Attempt04 dev900 boundary changed: {key}")

    budget = _mapping(plan.get("budget"), "budget")
    expected_budget = {
        "fresh_development_roots": 1000,
        "fresh_development_shards": 100,
        "roots_per_shard": M43_ATTEMPT05_ROOTS_PER_SHARD,
        "profiles": 5,
        "roots_per_profile": 200,
        "pilot_roots_before_expand": 250,
        "expand_roots_after_pilot_go_only": 750,
        "inherited_sealed_acceptance_roots": 700,
        "spot_script_overhaul_authorized": False,
        "large_scale_authorized": False,
    }
    for key, expected in expected_budget.items():
        if budget.get(key) != expected:
            raise ValueError(f"Attempt05 budget changed: {key}")

    planned = []
    planned.extend(
        _validate_roles(
            plan.get("development_roles"),
            expected=_DEVELOPMENT_ROLES,
            section="development_roles",
        )
    )
    planned.extend(
        _validate_roles(
            plan.get("inherited_attempt04_acceptance_roles"),
            expected=_ACCEPTANCE_ROLES,
            section="inherited_attempt04_acceptance_roles",
            require_unopened=True,
        )
    )
    _validate_search_and_lifecycle(plan)

    population = _mapping(plan.get("population_acceptance"), "population_acceptance")
    expected_population = {
        "classification": "inherited_unopened",
        "paired_seat_swap": True,
        "opponents": 4,
        "paired_seeds_per_opponent": 1000,
        "shards": 20,
        "paired_seeds_per_shard": 50,
        "seed_start": 14_106_071_901,
        "seed_stride": M43_ATTEMPT05_SEED_STRIDE,
        "open_after": "attempt05_locked_holdout_go",
        "threshold_reselection_allowed": False,
    }
    if dict(population) != expected_population:
        raise ValueError("Attempt05 population14.106b contract changed")
    planned.append(
        (
            "population_acceptance",
            _schedule(14_106_071_901, 1000),
        )
    )
    _validate_freshness(plan, planned)
    _validate_guards(plan)


def validate_attempt05_status(status: Mapping[str, Any]) -> None:
    if status.get("schema") != M43_ATTEMPT05_STATUS_SCHEMA:
        raise ValueError("unsupported Attempt05 status schema")
    if (
        status.get("milestone"),
        status.get("attempt"),
        status.get("status"),
        status.get("decision"),
    ) != (
        "M4.3",
        "attempt05",
        "complete_no_go_architecture_one_shot",
        "no_go_no_fresh_audit_or_spot",
    ):
        raise ValueError("Attempt05 status overstates execution")

    plan = _mapping(status.get("plan"), "plan")
    if dict(plan) != {
        "path": "configs/hu_joint_policy_m43_attempt05.json",
        "schema": M43_ATTEMPT05_PLAN_SCHEMA,
        "classification": "historical_frozen_pre_pilot_contract",
    }:
        raise ValueError("Attempt05 status plan boundary changed")

    architecture = _mapping(
        status.get("architecture_one_shot"), "architecture_one_shot"
    )
    expected_architecture = {
        "classification": "old_dev900_development_only_consumed_not_fresh",
        "report_path": (
            "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/"
            "architecture_comparison.json"
        ),
        "report_sha256": (
            "c8c8f364632ec85c600572239a387bedcc0ccbf82d3ef1f87147233ac765553f"
        ),
        "report_schema": "hu_m43_attempt05_old_dev900_architecture_comparison_v1",
        "report_status": "development_no_go_no_new_audit_authorized",
        "states": 900,
        "folds": 5,
        "profiles": 5,
        "selected_family": None,
        "winner_is_runtime_frozen": False,
        "threshold_sweep_performed": False,
        "fresh_generalization_claim": False,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"Attempt05 architecture No-Go changed: {key}")
    if dict(_mapping(architecture.get("raw_gate"), "raw_gate")) != {
        "overall_positive_rate_min": 0.4,
        "each_profile_positive_rate_min": 0.3,
    }:
        raise ValueError("Attempt05 architecture raw gate changed")
    families = _mapping(architecture.get("families"), "families")
    expected_families = {
        "lambda_rank": (False, 0.35333333333333333, 0.31666666666666665),
        "deepsets": (False, 0.21, 0.17777777777777778),
    }
    if tuple(families) != tuple(expected_families):
        raise ValueError("Attempt05 architecture family set changed")
    for family, expected in expected_families.items():
        result = _mapping(families.get(family), f"families.{family}")
        actual = (
            result.get("raw_gate_pass"),
            result.get("raw_selected_positive_rate"),
            result.get("minimum_profile_positive_rate"),
        )
        if actual != expected:
            raise ValueError(f"Attempt05 {family} raw No-Go changed")
        if not float(result.get("raw_selected_mean_delta", 0.0)) < 0.0:
            raise ValueError(f"Attempt05 {family} mean delta no longer records No-Go")

    trust = _mapping(status.get("postrun_trust_audit"), "postrun_trust_audit")
    expected_trust = {
        "path": (
            "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/"
            "postrun_trust_audit.json"
        ),
        "sha256": (
            "0414b3bbc7b8e678911c66cd038d73528f2a03c015e53b120017e116eb35512f"
        ),
        "schema": "hu_m43_attempt05_postrun_trust_audit_v1",
        "status": "pass_current_trust_boundary_no_reselection",
        "one_shot_wall_seconds": 447.158,
        "architecture_report_sha256_matches": True,
        "fresh_or_locked_data_opened": False,
        "new_audit_authorized": False,
        "spot_started": False,
        "current_profile_mutated": False,
    }
    for key, expected in expected_trust.items():
        if trust.get(key) != expected:
            raise ValueError(f"Attempt05 postrun trust boundary changed: {key}")
    loader = _mapping(
        trust.get("current_loader_revalidation"), "current_loader_revalidation"
    )
    if dict(loader) != {
        "states": 900,
        "unique_observation_identities": 900,
        "identity_leakage": 0,
        "t1_second_only": True,
        "canonical_policy_observation": True,
        "complete_legal_action_key_set": True,
        "baseline_action_key_index_binding": True,
        "source_manifest_equal_to_one_shot": True,
    }:
        raise ValueError("Attempt05 hardened loader revalidation changed")
    source_timing = _mapping(trust.get("source_timing"), "source_timing")
    for key in (
        "comparison_reexecuted",
        "model_weights_recomputed_after_hardening",
        "architecture_selection_recomputed_after_hardening",
    ):
        if source_timing.get(key) is not False:
            raise ValueError("Attempt05 post-hardening recomputation was overstated")
    if source_timing.get(
        "comparison_process_started_before_final_trust_hardening_landed"
    ) is not True:
        raise ValueError("Attempt05 postrun source timing boundary changed")
    candidates = _mapping(trust.get("candidate_artifacts"), "candidate_artifacts")
    expected_candidate_hashes = {
        "lambda_rank": "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3",
        "deepsets": "c05b7469abe8d9ac8b4ccaf9def9f24bed4d6fc4670aab95d096baa702696b1b",
    }
    if tuple(candidates) != tuple(expected_candidate_hashes):
        raise ValueError("Attempt05 audited candidate family set changed")
    for family, expected_hash in expected_candidate_hashes.items():
        candidate = _mapping(candidates.get(family), f"candidate_artifacts.{family}")
        if candidate.get("sha256") != expected_hash or dict(candidate) != {
            "sha256": expected_hash,
            "two_independent_loads_equal": True,
            "runtime_enabled": False,
            "winner_frozen": False,
        }:
            raise ValueError(f"Attempt05 {family} artifact trust boundary changed")

    timing = _mapping(
        status.get("search_feasibility_boundary"), "search_feasibility_boundary"
    )
    if (
        timing.get("classification"),
        timing.get("exact_all_legal_actions_wall_seconds"),
        timing.get("mc1_wall_seconds"),
        timing.get("observed_wall_time_ratio_exact_all_over_mc1"),
        timing.get("exact_all_scaled_generation_authorized"),
        timing.get("mc1_quality_or_acceptance_claim_allowed"),
    ) != (
        "local_timing_diagnostic_only_not_policy_quality_evidence",
        400.711,
        2.509,
        159.709446,
        False,
        False,
    ):
        raise ValueError("Attempt05 exact-all/MC1 timing boundary changed")

    roles = _mapping(status.get("development_roles"), "development_roles")
    if dict(roles) != {
        "pilot_train": "not_started_closed_by_architecture_no_go",
        "pilot_audit": "not_started_unopened",
        "expand_train": "not_started_unopened",
        "final_audit": "not_started_unopened",
    }:
        raise ValueError("Attempt05 development lifecycle status changed")
    inherited = _mapping(
        status.get("inherited_attempt04_acceptance_roles"),
        "inherited_attempt04_acceptance_roles",
    )
    if dict(inherited) != {
        "precal_holdout": "unopened",
        "calibration_safety_fit": "unopened",
        "calibration_threshold_lock": "unopened",
        "locked_holdout": "unopened",
    } or status.get("population_acceptance") != "unopened":
        raise ValueError("Attempt05 unopened acceptance boundary changed")
    spot = _mapping(status.get("spot_execution"), "spot_execution")
    if set(spot) != {
        "scripts_overhauled",
        "package_created",
        "instances_created",
        "run_started",
    } or any(value is not False for value in spot.values()):
        raise ValueError("Attempt05 status claims Spot execution")
    attempt06 = _mapping(status.get("attempt06_boundary"), "attempt06_boundary")
    if dict(attempt06) != {
        "top4_diagnostic_status": "complete_design_evidence_only",
        "candidate_set_diagnostic_status": (
            "complete_metric_corrected_design_evidence_only"
        ),
        "attempt06_plan": {
            "path": "configs/hu_joint_policy_m43_attempt06.json",
            "schema": "hu_m43_attempt06_corrected_pre_fresh_plan_v3",
            "sha256": (
                "4844fb970780c04ff093eb43b1672e403f006515c47b287e6abdbea17867f5b8"
            ),
        },
        "plan_frozen": True,
        "selected_search_design": {
            "candidate_generator": "lambda_rank_artifact_design_only",
            "learned_nonbaseline_top_k": 8,
            "candidate_selection_samples": 8,
            "independent_evaluation_samples": 128,
        },
        "runtime_model_selected": False,
        "fresh_generation_authorized": False,
        "spot_authorized": False,
        "runtime_authorized": False,
    }:
        raise ValueError("Attempt05-to-Attempt06 handoff boundary changed")
    if status.get("postmortem") != "docs/hu_joint_policy_m43_attempt05_postmortem.md":
        raise ValueError("Attempt05 postmortem binding changed")
    guards = _mapping(status.get("guards"), "guards")
    if set(guards) != {
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "teacher_lcb_runtime_gate_enabled",
        "threshold_reselected_on_development",
        "fresh_development_opened",
        "acceptance_holdout_opened",
        "population_evaluation_started",
        "large_scale_authorized",
    } or any(value is not False for value in guards.values()):
        raise ValueError("Attempt05 status activation guard changed")
    _validate_hash_audit(status)


def _validate_roles(
    raw: Any,
    *,
    expected: Mapping[str, tuple[int, int, int, int, int]],
    section: str,
    require_unopened: bool = False,
) -> list[tuple[str, set[int]]]:
    roles = _mapping(raw, section)
    if tuple(roles) != tuple(expected):
        raise ValueError(f"{section} role set/order changed")
    schedules: list[tuple[str, set[int]]] = []
    for role, values in expected.items():
        roots, hand, candidate, evaluation, child = values
        spec = _mapping(roles.get(role), f"{section}.{role}")
        expected_fields = {
            "roots": roots,
            "shards": roots // M43_ATTEMPT05_ROOTS_PER_SHARD,
            "roots_per_profile": roots // len(M43_ATTEMPT05_PROFILES),
            "seed_start": hand,
            "candidate_seed_start": candidate,
            "evaluation_seed_start": evaluation,
            "child_policy_seed_start": child,
        }
        for key, value in expected_fields.items():
            if spec.get(key) != value:
                raise ValueError(f"{section}.{role} changed: {key}")
        if require_unopened and spec.get("classification") != "inherited_unopened":
            raise ValueError(f"{section}.{role} is no longer unopened")
        shards = roots // M43_ATTEMPT05_ROOTS_PER_SHARD
        schedules.append((f"{section}.{role}.hand", _schedule(hand, roots)))
        schedules.append((f"{section}.{role}.candidate", _schedule(candidate, shards)))
        schedules.append((f"{section}.{role}.evaluation", _schedule(evaluation, shards)))
        schedules.append((f"{section}.{role}.child", _schedule(child, shards)))
    return schedules


def _validate_search_and_lifecycle(plan: Mapping[str, Any]) -> None:
    search = _mapping(plan.get("teacher_search"), "teacher_search")
    expected_search = {
        "candidate_top_k": 4,
        "search_evaluation_samples_candidates": [4, 8],
        "final_teacher_label_evaluation_samples": 128,
        "common_random_futures": True,
        "candidate_selection_and_final_evaluation_rng_disjoint": True,
        "candidate_seed_namespace": "hu-m43-attempt05-candidate-v1",
        "evaluation_seed_namespace": "hu-m43-attempt05-evaluation-v1",
        "child_policy_seed_namespace": "hu-m43-attempt05-child-v1",
        "namespace_domain_separation": "role_shard",
        "action_mapping_must_match_exactly": True,
        "illegal_action_masking_required": True,
    }
    if dict(search) != expected_search:
        raise ValueError("Attempt05 topK4/e4-e8/e128 search contract changed")
    lifecycle = _mapping(plan.get("development_lifecycle"), "development_lifecycle")
    expected_lifecycle = {
        "role_order": list(_DEVELOPMENT_ROLES),
        "pilot_train_fit_only": True,
        "pilot_audit_one_shot_only": True,
        "expand_requires_pilot_audit_go": True,
        "search_evaluation_setting_frozen_before_expand": True,
        "final_audit_one_shot_only": True,
        "final_audit_fit_allowed": False,
        "attempt04_acceptance_roles_open_only_after_final_audit_go": True,
        "fresh_data_required_after_model_feature_search_or_gate_change": True,
    }
    if dict(lifecycle) != expected_lifecycle:
        raise ValueError("Attempt05 lifecycle boundary was weakened")
    threshold = _mapping(plan.get("threshold_contract"), "threshold_contract")
    if tuple(threshold.get("grid", ())) != M43_ATTEMPT05_THRESHOLD_GRID:
        raise ValueError("Attempt05 threshold grid changed")
    for key in (
        "fixed_before_fresh_attempt05_generation",
    ):
        if threshold.get(key) is not True:
            raise ValueError("Attempt05 threshold freeze weakened")
    for key in (
        "audit_or_locked_threshold_research_allowed",
        "teacher_lcb_as_runtime_gate_allowed",
    ):
        if threshold.get(key) is not False:
            raise ValueError("Attempt05 threshold boundary weakened")


def _validate_freshness(
    plan: Mapping[str, Any], planned: Sequence[tuple[str, set[int]]]
) -> None:
    freshness = _mapping(plan.get("freshness"), "freshness")
    if freshness.get("seed_stride") != M43_ATTEMPT05_SEED_STRIDE:
        raise ValueError("Attempt05 seed stride changed")
    if freshness.get("planned_schedule_overlap_count_at_freeze") != 0:
        raise ValueError("Attempt05 freeze claims a schedule overlap")
    for key in (
        "reject_hand_seed_or_fingerprint_overlap",
        "audit_all_hand_candidate_evaluation_child_and_population_namespaces",
    ):
        if freshness.get(key) is not True:
            raise ValueError("Attempt05 freshness guard weakened")
    manifest = (
        ("excluded_hand_schedules", _EXCLUDED_HAND),
        ("excluded_rng_schedules", _EXCLUDED_RNG),
        ("excluded_population_schedules", _EXCLUDED_POPULATION),
    )
    excluded: list[tuple[str, set[int]]] = []
    for key, expected in manifest:
        rows = freshness.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"Attempt05 {key} missing")
        actual = tuple(
            (row.get("name"), row.get("seed_start"), row.get("count"))
            for row in rows
            if isinstance(row, Mapping)
        )
        if actual != expected:
            raise ValueError(f"Attempt05 {key} changed")
        excluded.extend(
            (f"{key}.{name}", _schedule(seed, count))
            for name, seed, count in expected
        )
    _reject_pairwise_overlap(planned, "Attempt05 planned schedules")
    for planned_name, planned_seeds in planned:
        for excluded_name, excluded_seeds in excluded:
            overlap = planned_seeds.intersection(excluded_seeds)
            if overlap:
                raise ValueError(
                    f"Attempt05 seed collision: {planned_name} vs {excluded_name}"
                )


def _validate_guards(plan: Mapping[str, Any]) -> None:
    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    if any(value is not False for value in guards.values()):
        raise ValueError("Attempt05 activation guard changed")
    _validate_hash_audit(plan)


def _validate_hash_audit(payload: Mapping[str, Any]) -> None:
    audit = _mapping(payload.get("baseline_hash_audit"), "baseline_hash_audit")
    if audit.get("policy_registry_path") != "src/ofc_regular/ai_profiles.py":
        raise ValueError("Attempt05 ai_profiles path changed")
    if audit.get("policy_registry_expected_sha256") != AI_PROFILES_SHA256:
        raise ValueError("Attempt05 ai_profiles hash changed")
    if audit.get("current_mapping_changed") is not False:
        raise ValueError("Attempt05 current mapping changed")


def _schedule(seed_start: int, count: int) -> set[int]:
    return {
        int(seed_start) + M43_ATTEMPT05_SEED_STRIDE * index
        for index in range(int(count))
    }


def _reject_pairwise_overlap(
    schedules: Sequence[tuple[str, set[int]]], label: str
) -> None:
    for index, (left_name, left) in enumerate(schedules):
        for right_name, right in schedules[index + 1 :]:
            if left.intersection(right):
                raise ValueError(f"{label} overlap: {left_name} vs {right_name}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


__all__ = [
    "AI_PROFILES_SHA256",
    "M43_ATTEMPT05_PLAN_SCHEMA",
    "M43_ATTEMPT05_PROFILES",
    "M43_ATTEMPT05_SEED_STRIDE",
    "M43_ATTEMPT05_STATUS_SCHEMA",
    "M43_ATTEMPT05_THRESHOLD_GRID",
    "load_and_validate_attempt05_plan",
    "load_and_validate_attempt05_status",
    "validate_attempt05_plan",
    "validate_attempt05_status",
]
