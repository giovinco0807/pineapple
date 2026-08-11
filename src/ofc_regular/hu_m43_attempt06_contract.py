"""Frozen corrected pre-fresh contract for M4.3 Attempt06.

Attempt06 is limited to one fresh 50-root search-quality audit.  It does not
fit, distill, select a threshold, open acceptance data, start Spot, register a
runtime model, or change ``current``.  A Go permits only creation of a separate
freeze before any fresh 200-root fit/distillation work.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


M43_ATTEMPT06_PLAN_SCHEMA = "hu_m43_attempt06_corrected_pre_fresh_plan_v3"
M43_ATTEMPT06_STATUS_SCHEMA = "hu_m43_attempt06_status_v3"
M43_ATTEMPT06_PLAN_SHA256 = (
    "4844fb970780c04ff093eb43b1672e403f006515c47b287e6abdbea17867f5b8"
)
M43_ATTEMPT06_SEED_STRIDE = 1_000_003
M43_ATTEMPT06_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
M43_ATTEMPT06_LAMBDA_SHA256 = (
    "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
)
AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

_PROFILE_COUNTS = {
    "stage19_p0": (180, 109, 103),
    "stage9f_p2": (180, 111, 107),
    "stage7_m5_r10": (180, 107, 105),
    "stage3_baseline": (180, 115, 108),
    "random_exact_final": (180, 123, 118),
}
_SEED_STARTS = {
    "hand": 17_306_071_901,
    "candidate": 23_306_071_901,
    "evaluation": 24_306_071_901,
    "child": 25_306_071_901,
}


def load_and_validate_attempt06_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("Attempt06 plan must be a mapping")
    validate_attempt06_plan(plan)
    return plan


def load_and_validate_attempt06_status(path: str | Path) -> dict[str, Any]:
    status = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(status, dict):
        raise ValueError("Attempt06 status must be a mapping")
    validate_attempt06_status(status)
    return status


def validate_attempt06_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != M43_ATTEMPT06_PLAN_SCHEMA:
        raise ValueError("unsupported Attempt06 plan schema")
    if (plan.get("milestone"), plan.get("status")) != (
        "M4.3-attempt06",
        "frozen_preflight",
    ):
        raise ValueError("Attempt06 is not frozen at preflight")
    _validate_scope(plan)
    _validate_design_evidence(plan)
    _validate_metric_correction(plan)
    _validate_search_and_continuation(plan)
    _validate_schedule_transfer(plan)
    _validate_one_shot_and_go_no_go(plan)
    _validate_pre_spot(plan)
    _validate_hash_and_guards(plan, guard_key="activation_guards")


def validate_attempt06_status(status: Mapping[str, Any]) -> None:
    if status.get("schema") != M43_ATTEMPT06_STATUS_SCHEMA:
        raise ValueError("unsupported Attempt06 status schema")
    if (
        status.get("milestone"),
        status.get("attempt"),
        status.get("status"),
        status.get("decision"),
    ) != (
        "M4.3",
        "attempt06",
        "frozen_preflight",
        "corrected_rank8_search_quality_audit_only",
    ):
        raise ValueError("Attempt06 status overstates preflight")
    if dict(_mapping(status.get("plan"), "status.plan")) != {
        "path": "configs/hu_joint_policy_m43_attempt06.json",
        "schema": M43_ATTEMPT06_PLAN_SCHEMA,
        "sha256": M43_ATTEMPT06_PLAN_SHA256,
    }:
        raise ValueError("Attempt06 status plan binding changed")
    frozen = _mapping(status.get("frozen_design"), "frozen_design")
    expected_frozen = {
        "family": "lambda_rank",
        "artifact_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
        "teacher_schema": "hu_m43_attempt06_t1_second_top8_c8_e128_teacher_v3",
        "learned_nonbaseline_top_k": 8,
        "candidate_selection_samples": 8,
        "independent_evaluation_samples": 128,
        "independent_evaluation_raw_paired_deltas_retained": True,
        "independent_evaluation_raw_count": 128,
        "raw_summary_recomputed_with": "numpy_linear",
        "independent_evaluation_action_scope": (
            "locked_action_plus_explicit_baseline_only"
        ),
        "native_batch_threads": 4,
        "batch_child_selectors_required": True,
        "config_sha256_cross_bind_required": [
            "teacher_row_provenance",
            "checkpoint",
            "heartbeat",
            "generator_summary",
            "done",
        ],
        "rank8_conditional_recall_overall_min": 0.95,
        "rank8_conditional_recall_each_profile_min": 0.93,
        "observed_overall": 541 / 565,
        "observed_min_profile": 108 / 115,
        "tail_gate_semantics": (
            "per_fired_root_e128_p05_p01_min_then_maximum_across_fires"
        ),
        "paired_mean_cross_fire_quantiles_are_gate": False,
        "runtime_model_frozen": False,
        "runtime_enabled": False,
    }
    if dict(frozen) != expected_frozen:
        raise ValueError("Attempt06 frozen design changed")
    transfer = _mapping(status.get("seed_transfer"), "seed_transfer")
    if dict(transfer) != {
        "source": "attempt05.development_roles.pilot_audit",
        "source_started": False,
        "exclusive_owner": "attempt06.search_quality_audit",
        "roots": 50,
        "shards": 50,
        "roots_per_shard": 1,
        "seed_granularity": "one_hand_candidate_evaluation_child_tuple_per_root",
        "per_root_seed_tuple_count": 50,
        "profiles": 5,
        "roots_per_profile": 10,
        "hand_seed_start": _SEED_STARTS["hand"],
        "candidate_seed_start": _SEED_STARTS["candidate"],
        "evaluation_seed_start": _SEED_STARTS["evaluation"],
        "child_policy_seed_start": _SEED_STARTS["child"],
    }:
        raise ValueError("Attempt06 status seed transfer changed")
    preflight = _mapping(status.get("preflight"), "preflight")
    expected_preflight = {
        "local_correctness": "not_started",
        "determinism": "not_started",
        "scalar_batch_exact_parity": "not_started",
        "latency_profile": "not_started",
    }
    if dict(preflight) != expected_preflight:
        raise ValueError("Attempt06 status prematurely passes preflight")
    audit = _mapping(
        status.get("one_shot_search_quality_audit"),
        "one_shot_search_quality_audit",
    )
    if dict(audit) != {
        "status": "not_started",
        "consumption_marker_claimed": False,
        "fresh_content_opened": False,
        "fit_performed": False,
        "threshold_selected": False,
        "result": None,
    }:
        raise ValueError("Attempt06 status prematurely opens one-shot audit")
    spot = _mapping(status.get("spot_execution"), "spot_execution")
    if set(spot) != {
        "authorized",
        "scripts_created",
        "package_created",
        "instances_created",
        "run_started",
    } or any(value is not False for value in spot.values()):
        raise ValueError("Attempt06 status claims Spot execution")
    if dict(_mapping(status.get("continuation_boundary"), "continuation_boundary")) != {
        "actual_t2_policy_id": "stage9f_p2",
        "hypothetical_t4_mode": "counter_mc_1",
        "real_live_t4_exact_unchanged": True,
    }:
        raise ValueError("Attempt06 status continuation boundary changed")
    _validate_hash_and_guards(status, guard_key="guards")


def enumerate_attempt06_seed_schedules(
    plan: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Enumerate all four 50-root namespaces from the frozen transfer."""

    transfer = _mapping(plan.get("audit_schedule_transfer"), "audit_schedule_transfer")
    roots = int(transfer.get("roots", -1))
    stride = int(transfer.get("seed_stride", -1))
    starts = {
        "hand": transfer.get("hand_seed_start"),
        "candidate": transfer.get("candidate_seed_start"),
        "evaluation": transfer.get("evaluation_seed_start"),
        "child": transfer.get("child_policy_seed_start"),
    }
    if roots != 50 or stride != M43_ATTEMPT06_SEED_STRIDE:
        raise ValueError("Attempt06 seed enumeration boundary changed")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in starts.values()):
        raise ValueError("Attempt06 seed starts must be integers")
    return {
        name: tuple(int(start) + stride * index for index in range(roots))
        for name, start in starts.items()
    }


def _validate_scope(plan: Mapping[str, Any]) -> None:
    scope = _mapping(plan.get("scope"), "scope")
    if dict(scope) != {
        "street": "T1",
        "seat": "second",
        "first_seat_behavior": "unchanged_stage18_p1_baseline",
        "fixed_baseline_profile": "stage18_p1",
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
    }:
        raise ValueError("Attempt06 scope changed")
    boundary = _mapping(plan.get("attempt05_boundary"), "attempt05_boundary")
    if boundary.get("status") != "complete_no_go_architecture_one_shot" or any(
        boundary.get(key) is not False
        for key in (
            "attempt05_promotion_allowed",
            "attempt05_candidate_artifact_runtime_authorized",
            "teacher_values_are_realized_match_ev",
        )
    ):
        raise ValueError("Attempt06 lost the Attempt05 No-Go boundary")


def _validate_design_evidence(plan: Mapping[str, Any]) -> None:
    evidence = _mapping(plan.get("design_evidence"), "design_evidence")
    architecture = _mapping(evidence.get("architecture_report"), "architecture_report")
    if architecture.get("sha256") != (
        "c8c8f364632ec85c600572239a387bedcc0ccbf82d3ef1f87147233ac765553f"
    ) or architecture.get("selected_family") is not None:
        raise ValueError("Attempt06 architecture evidence changed")
    diagnostic = _mapping(
        evidence.get("candidate_set_diagnostic"), "candidate_set_diagnostic"
    )
    if (
        diagnostic.get("sha256"),
        diagnostic.get("original_status"),
        diagnostic.get("original_absolute_gate_invalidated_by_metric_correction"),
        diagnostic.get("fresh_generalization_claim_allowed"),
    ) != (
        "37c612600c5cf33424e487adf2bdff5db785f893fad92da096c39a294420360f",
        "development_no_go_no_candidate_set_meets_coverage_gate",
        True,
        False,
    ):
        raise ValueError("Attempt06 candidate-set evidence changed")
    generator = _mapping(evidence.get("candidate_generator"), "candidate_generator")
    if generator.get("family") != "lambda_rank" or generator.get(
        "artifact_sha256"
    ) != M43_ATTEMPT06_LAMBDA_SHA256:
        raise ValueError("Attempt06 LambdaRank artifact changed")
    if generator.get("frozen_for_attempt06_candidate_generation") is not True or any(
        generator.get(key) is not False
        for key in ("runtime_model_frozen", "runtime_enabled")
    ):
        raise ValueError("Attempt06 candidate artifact was promoted to runtime")


def _validate_metric_correction(plan: Mapping[str, Any]) -> None:
    correction = _mapping(plan.get("coverage_metric_correction"), "metric correction")
    if correction.get("classification") != (
        "pre_fresh_denominator_correction_not_threshold_retune"
    ):
        raise ValueError("Attempt06 metric correction classification changed")
    invalid = _mapping(correction.get("invalid_absolute_gate"), "invalid gate")
    if (
        invalid.get("overall_min"),
        invalid.get("each_profile_min"),
        invalid.get("all_legal_opportunity_states"),
        invalid.get("all_states"),
        invalid.get("overall_gate_reachable"),
        invalid.get("each_profile_gate_reachable"),
    ) != (0.7, 0.6, 565, 900, False, False):
        raise ValueError("Attempt06 invalid absolute gate boundary changed")
    _require_close(
        invalid.get("overall_mathematical_ceiling"),
        565 / 900,
        "overall opportunity ceiling",
    )
    _require_close(
        invalid.get("minimum_profile_mathematical_ceiling"),
        107 / 180,
        "profile opportunity ceiling",
    )
    metric = _mapping(correction.get("corrected_metric"), "corrected_metric")
    if (
        metric.get("name"),
        metric.get("overall_min"),
        metric.get("each_profile_min"),
        metric.get("used_to_freeze_candidate_set_only"),
        metric.get("fresh_search_quality_audit_gate"),
    ) != (
        "rank8_conditional_positive_recall",
        0.95,
        0.93,
        True,
        False,
    ):
        raise ValueError("Attempt06 corrected conditional-recall gate changed")
    observed = _mapping(correction.get("observed_multiple_use_dev900"), "observed")
    overall = _mapping(observed.get("overall"), "observed.overall")
    if (
        overall.get("states"),
        overall.get("opportunity_states"),
        overall.get("rank8_covered_opportunity_states"),
    ) != (900, 565, 541):
        raise ValueError("Attempt06 overall rank8 counts changed")
    _require_close(overall.get("absolute_coverage"), 541 / 900, "absolute coverage")
    _require_close(overall.get("conditional_recall"), 541 / 565, "conditional recall")
    profiles = _mapping(observed.get("profile"), "observed.profile")
    if tuple(profiles) != M43_ATTEMPT06_PROFILES:
        raise ValueError("Attempt06 metric profile set/order changed")
    recalls = []
    for profile, (states, opportunities, covered) in _PROFILE_COUNTS.items():
        row = _mapping(profiles.get(profile), f"observed.profile.{profile}")
        if (
            row.get("states"),
            row.get("opportunity_states"),
            row.get("rank8_covered_opportunity_states"),
        ) != (states, opportunities, covered):
            raise ValueError(f"Attempt06 {profile} rank8 counts changed")
        expected = covered / opportunities
        _require_close(row.get("conditional_recall"), expected, f"{profile} recall")
        recalls.append(expected)
    if 541 / 565 < 0.95 or min(recalls) < 0.93:
        raise ValueError("Attempt06 frozen rank8 evidence fails corrected gate")
    if (
        observed.get("overall_gate_pass"),
        observed.get("each_profile_gate_pass"),
        observed.get("minimum_profile"),
        observed.get("acceptance_or_runtime_claim_allowed"),
    ) != (True, True, "stage3_baseline", False):
        raise ValueError("Attempt06 corrected metric claim boundary changed")


def _validate_search_and_continuation(plan: Mapping[str, Any]) -> None:
    search = _mapping(plan.get("search_contract"), "search_contract")
    expected_search = {
        "teacher_schema": "hu_m43_attempt06_t1_second_top8_c8_e128_teacher_v3",
        "action_key_schema": "regular_ofc_action_key_v1",
        "candidate_generator": "lambda_rank_strict_action_scores",
        "learned_nonbaseline_top_k": 8,
        "baseline_added_exactly_once": True,
        "complete_legal_action_set_required": True,
        "illegal_action_masking_required": True,
        "candidate_selection_samples": 8,
        "independent_evaluation_samples": 128,
        "independent_evaluation_raw_paired_deltas_retained": True,
        "independent_evaluation_raw_count": 128,
        "raw_summary_recomputed_with": "numpy_linear",
        "independent_evaluation_action_scope": (
            "locked_action_plus_explicit_baseline_only"
        ),
        "unselected_candidate_evaluation_fields": "null",
        "evaluation_pair_best_is_diagnostic_only": True,
        "candidate_selected_before_evaluation_opened": True,
        "evaluation_may_rerank": False,
        "common_random_futures_across_actions": True,
        "candidate_evaluation_rng_disjoint": True,
        "candidate_ties": "ActionKey",
        "search_ties": "ActionKey",
        "profile_runtime_feature_allowed": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_ev_or_lcb_runtime_gate_allowed": False,
    }
    if dict(search) != expected_search:
        raise ValueError("Attempt06 top8/c8/e128 search contract changed")
    continuation = _mapping(plan.get("fixed_continuation"), "fixed_continuation")
    expected_continuation = {
        "t2_policy_id": "stage9f_p2",
        "t2_resolution": "explicit_profile_never_current",
        "t3_selector": "m3_rust_evaluate_t3",
        "t3_candidate_samples": 1,
        "t3_evaluation_samples": 1,
        "t3_downstream_t3_samples": 1,
        "hypothetical_t1_search_direct_t4_mode": "counter_mc_1",
        "hypothetical_t1_search_direct_t4_candidate_samples": 1,
        "hypothetical_t1_search_direct_t4_evaluation_samples": 1,
        "hypothetical_nested_t4_mode": "counter_mc_1",
        "hypothetical_nested_t4_samples": 1,
        "real_live_t4_selector": "exact_solver",
        "real_live_t4_exact_unchanged": True,
    }
    if dict(continuation) != expected_continuation:
        raise ValueError("Attempt06 continuation boundary changed")


def _validate_schedule_transfer(plan: Mapping[str, Any]) -> None:
    transfer = _mapping(plan.get("audit_schedule_transfer"), "audit schedule")
    expected = {
        "source_plan": "configs/hu_joint_policy_m43_attempt05.json",
        "source_role": "development_roles.pilot_audit",
        "source_role_started": False,
        "source_role_consumed": False,
        "exclusive_owner_after_freeze": "M4.3-attempt06.search_quality_audit",
        "attempt05_reuse_after_transfer_allowed": False,
        "roots": 50,
        "shards": 50,
        "roots_per_shard": 1,
        "shard_assignment": (
            "one_root_per_shard_for_checkpoint_heartbeat_and_preemption_resume"
        ),
        "profiles": list(M43_ATTEMPT06_PROFILES),
        "roots_per_profile": 10,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "seed_stride": M43_ATTEMPT06_SEED_STRIDE,
        "hand_seed_start": _SEED_STARTS["hand"],
        "candidate_seed_start": _SEED_STARTS["candidate"],
        "evaluation_seed_start": _SEED_STARTS["evaluation"],
        "child_policy_seed_start": _SEED_STARTS["child"],
        "seed_granularity": "one_hand_candidate_evaluation_child_tuple_per_root",
        "candidate_evaluation_child_seed_count": 50,
        "per_root_seed_formula": (
            "namespace_seed_start + seed_stride * root_index for root_index 0..49"
        ),
        "candidate_seed_namespace": "hu-m43-attempt06-candidate-v1",
        "evaluation_seed_namespace": "hu-m43-attempt06-evaluation-v1",
        "child_policy_seed_namespace": "hu-m43-attempt06-child-v1",
        "numeric_seed_sets_pairwise_disjoint": True,
    }
    if dict(transfer) != expected:
        raise ValueError("Attempt06 transferred 50-root seed schedule changed")
    schedules = enumerate_attempt06_seed_schedules(plan)
    if any(len(values) != 50 or len(set(values)) != 50 for values in schedules.values()):
        raise ValueError("Attempt06 per-root seed namespace is not unique")
    _reject_pairwise_overlap(schedules)
    assigned = [M43_ATTEMPT06_PROFILES[index % 5] for index in range(50)]
    if any(assigned.count(profile) != 10 for profile in M43_ATTEMPT06_PROFILES):
        raise ValueError("Attempt06 profile assignment is not 5x10 balanced")


def _validate_one_shot_and_go_no_go(plan: Mapping[str, Any]) -> None:
    audit = _mapping(
        plan.get("one_shot_search_quality_audit"), "one_shot_search_quality_audit"
    )
    expected_audit = {
        "classification": "fresh_one_shot_search_quality_only",
        "fit_allowed": False,
        "model_change_allowed": False,
        "threshold_selection_allowed": False,
        "gate_reselection_allowed": False,
        "retry_same_seeds_after_open_allowed": False,
        "one_shot_consumption_marker_required_before_content_read": True,
        "teacher_values_reported_as_realized_match_ev": False,
        "larger_generation_or_acceptance_authorized_by_go": False,
        "go_authorizes_only": (
            "create_a_separate_freeze_for_fresh_200_root_fit_and_distillation"
        ),
        "fresh_200_root_fit_or_distillation_authorized_by_this_plan": False,
        "no_go_action": (
            "close_attempt06_without_fit_threshold_retry_or_spot_expansion"
        ),
    }
    if dict(audit) != expected_audit:
        raise ValueError("Attempt06 one-shot audit role changed")
    gates = _mapping(plan.get("search_quality_go_no_go"), "search quality gates")
    expected_gates = {
        "all_gates_required": True,
        "fires_total_min": 10,
        "fires_each_profile_min": 1,
        "mean_delta_per_state_strictly_greater_than": 0.0,
        "mean_delta_per_fire_strictly_greater_than": 0.0,
        "false_positive_definition": (
            "fired_action_independent_e128_paired_delta_vs_baseline_lte_zero"
        ),
        "false_positive_rate_per_fire_max": 0.5,
        "tail_gate_semantics": {
            "variant": (
                "B_per_fired_root_independent_e128_paired_future_tail_then_"
                "max_across_fires"
            ),
            "paired_future_count_per_fired_root": 128,
            "within_root_quantile_method": "numpy_linear",
            "per_fired_root_metrics": {
                "p95_loss": (
                    "max(0,-p05(independent_e128_paired_future_deltas))"
                ),
                "p99_loss": (
                    "max(0,-p01(independent_e128_paired_future_deltas))"
                ),
                "max_loss": (
                    "max(0,-min(independent_e128_paired_future_deltas))"
                ),
            },
            "audit_gate_aggregation": {
                "p95_loss": "maximum_per_fired_root_p95_loss",
                "p99_loss": "maximum_per_fired_root_p99_loss",
                "max_loss": "maximum_per_fired_root_max_loss",
            },
            "paired_mean_cross_fire_tail": {
                "per_fired_root_loss": (
                    "max(0,-independent_e128_paired_delta_mean)"
                ),
                "aggregation": "numpy_quantile_across_fired_roots",
                "classification": "diagnostic_only_not_gate",
            },
            "aligned_with": (
                "Attempt04_locked_selected_action_paired_p05_p01_min_maxima"
            ),
        },
        "override_loss_p95_max": 25.0,
        "override_loss_p99_max": 40.0,
        "override_loss_max": 50.0,
        "action_mapping_violation_count_max": 0,
        "rng_domain_violation_count_max": 0,
        "hidden_information_violation_count_max": 0,
        "go_action": (
            "create_separate_freeze_before_authorizing_fresh_200_root_fit_or_distillation"
        ),
        "go_directly_authorizes_fresh_200_root_fit_or_distillation": False,
        "no_go_action": (
            "close_attempt06_without_fit_threshold_retry_or_spot_expansion"
        ),
    }
    if dict(gates) != expected_gates:
        raise ValueError("Attempt06 frozen search-quality Go/No-Go changed")


def _validate_pre_spot(plan: Mapping[str, Any]) -> None:
    preflight = _mapping(plan.get("pre_spot_requirements"), "pre_spot_requirements")
    if tuple(preflight.get("required_order", ())) != (
        "local_correctness",
        "determinism",
        "scalar_batch_exact_parity",
        "latency_profile",
    ):
        raise ValueError("Attempt06 pre-Spot order changed")
    if preflight.get("spot_may_start_only_after_all_pass") is not True or preflight.get(
        "spot_authorized_at_freeze"
    ) is not False:
        raise ValueError("Attempt06 Spot was prematurely authorized")
    parity = str(preflight.get("scalar_batch_exact_parity", ""))
    if "mc1_value_need_not_equal_exact_value" not in parity:
        raise ValueError("Attempt06 parity boundary conflates MC1 with exact value")
    expected_preflight = {
        "required_order": [
            "local_correctness",
            "determinism",
            "scalar_batch_exact_parity",
            "latency_profile",
        ],
        "local_correctness": (
            "focused contract infoset mapping and fail_closed tests pass"
        ),
        "determinism": "repeat run digests identical for fixed inputs and seeds",
        "scalar_batch_exact_parity": (
            "scalar_batch_same_budget_parity_and_bounded_exact_reference_mapping_"
            "parity; mc1_value_need_not_equal_exact_value"
        ),
        "latency_profile": (
            "record top8_c8_e128 and bounded exact reference timings before Spot "
            "decision"
        ),
        "native_batch_threads": 4,
        "batch_child_selectors_required": True,
        "config_sha256_contract": "sha256_of_canonical_attempt06_fixed_contract",
        "config_sha256_cross_bind_required": [
            "teacher_row_provenance",
            "checkpoint",
            "heartbeat",
            "generator_summary",
            "done",
        ],
        "spot_may_start_only_after_all_pass": True,
        "spot_authorized_at_freeze": False,
        "spot_preemption_recovery": {
            "classification": (
                "claim_preserving_deterministic_rematerialization_not_fresh_"
                "audit_retry"
            ),
            "allowed_only_when_root_claim_is_byte_identical": True,
            "allowed_only_when_immutable_root_object_is_absent": True,
            "same_run_root_hand_seed_profile_required": True,
            "same_deterministic_seed_path_required": True,
            "same_frozen_source_startup_config_closure_required": True,
            "claim_bound_hashes": [
                "source_sha256",
                "startup_sha256",
                "source_closure_sha256",
                "manifest_sha256",
                "schedule_sha256",
                "plan_sha256",
                "status_sha256",
                "model_sha256",
                "source_model_manifest_sha256",
                "source_native_manifest_sha256",
            ],
            "alternate_sample_allowed": False,
            "fresh_audit_retry_allowed": False,
            "post_result_retry_allowed": False,
            "mismatch_action": "fail_closed_without_materializing_root",
        },
    }
    if dict(preflight) != expected_preflight:
        raise ValueError("Attempt06 pre-Spot native/config closure changed")


def _validate_hash_and_guards(payload: Mapping[str, Any], *, guard_key: str) -> None:
    audit = _mapping(payload.get("baseline_hash_audit"), "baseline_hash_audit")
    if dict(audit) != {
        "policy_registry_path": "src/ofc_regular/ai_profiles.py",
        "policy_registry_expected_sha256": AI_PROFILES_SHA256,
        "current_mapping_changed": False,
    }:
        raise ValueError("Attempt06 current profile hash boundary changed")
    guards = _mapping(payload.get(guard_key), guard_key)
    expected_keys = {
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "fresh_audit_started",
        "spot_run_started",
        "fit_started",
        "threshold_selection_started",
        "acceptance_holdout_opened",
        "population_evaluation_started",
        "large_scale_authorized",
    }
    if set(guards) != expected_keys or any(value is not False for value in guards.values()):
        raise ValueError("Attempt06 activation guard changed")


def _reject_pairwise_overlap(schedules: Mapping[str, Sequence[int]]) -> None:
    items = list(schedules.items())
    for index, (left_name, left) in enumerate(items):
        left_set = set(left)
        for right_name, right in items[index + 1 :]:
            if left_set.intersection(right):
                raise ValueError(
                    f"Attempt06 seed overlap: {left_name} vs {right_name}"
                )


def _require_close(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(f"Attempt06 {label} changed")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


__all__ = [
    "AI_PROFILES_SHA256",
    "M43_ATTEMPT06_LAMBDA_SHA256",
    "M43_ATTEMPT06_PLAN_SCHEMA",
    "M43_ATTEMPT06_PLAN_SHA256",
    "M43_ATTEMPT06_PROFILES",
    "M43_ATTEMPT06_SEED_STRIDE",
    "M43_ATTEMPT06_STATUS_SCHEMA",
    "enumerate_attempt06_seed_schedules",
    "load_and_validate_attempt06_plan",
    "load_and_validate_attempt06_status",
    "validate_attempt06_plan",
    "validate_attempt06_status",
]
