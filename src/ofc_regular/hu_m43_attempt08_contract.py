"""Frozen pre-development contract for M4.3 Attempt08.

Attempt08 contains one precommitted tail-safe search arm over 200 new balanced
T1-second roots.  This module validates the plan, its seed separation, its
fixed compute accounting, and the immutable artifacts on which it depends.
It does not authorize generation, the reserved audit, fitting, runtime use, or
any change to ``current``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


M43_ATTEMPT08_PLAN_SCHEMA = "hu_m43_attempt08_development_plan_v1"
M43_ATTEMPT08_PLAN_SHA256 = (
    "f0b8d7c2af40aeeb9b1a1c4e446237afb70ba04713b3d4bce555f26e4617ba51"
)
M43_ATTEMPT08_SEED_STRIDE = 1_000_003
M43_ATTEMPT08_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)

ATTEMPT08_TOP_K = 8
ATTEMPT08_RERANK_SAMPLES = 128
ATTEMPT08_SHORTLIST_K = 4
ATTEMPT08_VETO_SAMPLES = 256
ATTEMPT08_STRESS_SAMPLES = 512
ATTEMPT08_ASSESSMENT_SAMPLES = 256
ATTEMPT08_RISK_LIMITS = (22.0, 36.0, 45.0)
ATTEMPT08_VETO_MIN_MEAN = 0.0
ATTEMPT08_VETO_MIN_P05 = -22.0
ATTEMPT08_VETO_MIN_P01 = -36.0
ATTEMPT08_VETO_MIN_VALUE = -45.0
ATTEMPT08_STRESS_MIN_VALUE = -45.0

ATTEMPT07_CLOSEOUT_SHA256 = (
    "cf9da0b0da80f66e97eab70bb99d8658a01c3052e068b128f2d7318d6ad323fb"
)
ATTEMPT07_SELECTION_SHA256 = (
    "53b797756747cbb71f89251020e7fdc86dabce75cd298507dc59c0764ed6373a"
)
ATTEMPT08_LAMBDA_MODEL_SHA256 = (
    "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
)
AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

_SEED_BASES = {
    "hand": 80_108_071_901,
    "rerank": 81_108_071_901,
    "veto": 82_108_071_901,
    "stress": 83_108_071_901,
    "assessment": 84_108_071_901,
    "child": 85_108_071_901,
}
_ATTEMPT06_SEED_BASES = {
    "hand": 17_306_071_901,
    "candidate": 23_306_071_901,
    "evaluation": 24_306_071_901,
    "child": 25_306_071_901,
}
_ATTEMPT07_SEED_BASES = {
    "hand": 60_106_071_901,
    "screen": 61_106_071_901,
    "rerank": 62_106_071_901,
    "veto": 63_106_071_901,
    "assessment": 64_106_071_901,
    "child": 65_106_071_901,
}
_ATTEMPT07_PREFLIGHT_SEED_BASES = {
    "screen": 71_106_071_901,
    "rerank": 72_106_071_901,
    "veto": 73_106_071_901,
    "assessment": 74_106_071_901,
    "child": 75_106_071_901,
}

_EXPECTED_ATTEMPT07_BOUNDARY = {
    "status": "complete_no_go_development",
    "closeout": {
        "path": "configs/hu_joint_policy_m43_attempt07_closeout.json",
        "sha256": ATTEMPT07_CLOSEOUT_SHA256,
    },
    "selection": {
        "path": (
            "outputs/hu_joint_policy/m43_attempt07_development/"
            "regular-hu-m43-attempt07-development100-20260714-150046/"
            "merged/development_arm_selection.json"
        ),
        "sha256": ATTEMPT07_SELECTION_SHA256,
    },
    "data_use": "development_postmortem_design_only",
    "threshold_or_seed_retry_allowed": False,
    "development_root_or_seed_reuse_allowed": False,
    "future_audit50_reuse_allowed": False,
    "promotion_allowed": False,
}
_EXPECTED_SCOPE = {
    "street": "T1",
    "seat": "second",
    "fixed_baseline_profile": "stage18_p1",
    "first_seat_behavior": "unchanged_stage18_p1_baseline",
    "teacher_values_are_realized_match_ev": False,
    "teacher_ev_or_lcb_runtime_gate_allowed": False,
    "current_profile_mutated": False,
    "runtime_policy_activated": False,
    "full_replacement_enabled": False,
}
_EXPECTED_DEVELOPMENT = {
    "classification": "new_balanced_development_only",
    "roots": 200,
    "root_index_first": 0,
    "root_index_last": 199,
    "profiles": 5,
    "roots_per_profile": 40,
    "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
    "search_arm_count": 1,
    "fresh_generalization_claim_allowed": False,
    "fit_allowed": False,
    "threshold_selection_allowed": False,
    "runtime_activation_allowed": False,
}
_EXPECTED_DEVELOPMENT_EXECUTION = {
    "shard_layout": "one_root_per_shard",
    "batch_child_selectors_required": True,
    "native_batch_threads": 4,
    "opening_lookahead_samples": 0,
    "root_generation_policy": "attempt08_explicit_mod5_per_root_policy_seed_v1",
    "root_policy_seed_formula": (
        "profile_policy_seed(per_root_hand_seed,root_profile,seat)"
    ),
    "baseline_policy_seed_formula": (
        "profile_policy_seed(per_root_hand_seed,stage18_p1,second)"
    ),
    "continuation_policy_seed_formula": "child_seed_plus_0_first_plus_1_second",
    "correctness_determinism_parity_and_latency_preflight_required": True,
    "checkpoint_required": True,
    "heartbeat_required": True,
    "deterministic_same_contract_recompute_after_preemption_allowed": True,
    "alternate_root_or_seed_after_result_allowed": False,
    "current_profile_resolved": False,
}
_EXPECTED_FUTURE_AUDIT = {
    "classification": "declared_disjoint_one_shot_not_authorized",
    "roots": 50,
    "root_index_first": 200,
    "root_index_last": 249,
    "profiles": 5,
    "roots_per_profile": 10,
    "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
    "development_decision_freeze_required_before_authorization": True,
    "authorized": False,
    "started": False,
    "content_opened": False,
    "fit_allowed": False,
    "threshold_selection_allowed": False,
    "runtime_activation_allowed": False,
}
_EXPECTED_SEARCH = {
    "action_key_schema": "regular_ofc_action_key_v1",
    "candidate_generator": "frozen_lambda_rank_strict_action_scores",
    "candidate_generator_artifact_path": (
        "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/"
        "lambda_rank_candidate.pkl"
    ),
    "candidate_generator_artifact_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
    "learned_nonbaseline_top_k": ATTEMPT08_TOP_K,
    "baseline_added_exactly_once": True,
    "complete_legal_action_set_required": True,
    "illegal_action_masking_required": True,
    "candidate_order": "model_score_desc_then_ActionKey_with_baseline_last",
    "lambda_risk_predictions": {
        "source": "frozen_raw_downside_p95_p99_max_heads",
        "risk_score": (
            "max(downside_p95/22.0,downside_p99/36.0,downside_max/45.0)"
        ),
        "lower_is_safer": True,
        "conformal_upper_or_gain_gate_allowed": False,
        "may_directly_fire": False,
    },
    "rerank": {
        "symbol": "R128",
        "rng_domain": "rerank",
        "samples": ATTEMPT08_RERANK_SAMPLES,
        "action_scope": "rank8_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "nonbaseline_ranking": "paired_delta_mean_desc_then_ActionKey",
        "explicit_baseline_is_reference_not_nonbaseline_rank": True,
        "assessment_input_allowed": False,
    },
    "shortlist": {
        "symbol": "K4",
        "nonbaseline_actions": ATTEMPT08_SHORTLIST_K,
        "rerank_top_actions": 3,
        "risk_reserve": {
            "candidate_R_positions_inclusive": [4, 8],
            "selection": "minimum_lambda_risk_score_then_ActionKey",
            "score": (
                "max(downside_p95/22.0,downside_p99/36.0,downside_max/45.0)"
            ),
        },
        "selection": (
            "R_top3_plus_one_lambda_predicted_risk_reserve_from_"
            "R_ranks4_through8"
        ),
        "final_nonbaseline_order": "original_R_order",
        "baseline_added_exactly_once_after_shortlist": True,
        "assessment_input_allowed": False,
    },
    "veto": {
        "symbol": "V256",
        "rng_domain": "veto",
        "samples": ATTEMPT08_VETO_SAMPLES,
        "action_scope": "K4_nonbaseline_plus_explicit_baseline",
        "action_order": "original_R_order_for_K4_then_explicit_baseline",
        "common_random_futures_across_actions": True,
        "quantile_method": "numpy_linear",
        "selection": (
            "first_nonbaseline_action_in_original_R_order_passing_all_"
            "eligibility_checks"
        ),
        "eligibility": {
            "paired_delta_mean_strictly_greater_than": ATTEMPT08_VETO_MIN_MEAN,
            "paired_delta_p05_min": ATTEMPT08_VETO_MIN_P05,
            "paired_delta_p01_min": ATTEMPT08_VETO_MIN_P01,
            "paired_delta_min_min": ATTEMPT08_VETO_MIN_VALUE,
        },
        "may_maximize_veto_mean": False,
        "no_eligible_action": "exact_baseline_fallback",
        "assessment_input_allowed": False,
    },
    "stress": {
        "symbol": "X512",
        "rng_domain": "stress",
        "samples": ATTEMPT08_STRESS_SAMPLES,
        "execution_condition": "V256_selected_nonbaseline_action",
        "action_scope": "locked_V256_action_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "eligibility": {
            "paired_delta_min_min": ATTEMPT08_STRESS_MIN_VALUE,
        },
        "cancel_only": True,
        "may_promote_rerank_or_switch_action": False,
        "failed_stress_action": "exact_baseline_fallback",
        "skipped_when_veto_output_is_baseline": True,
        "assessment_input_allowed": False,
    },
    "assessment": {
        "symbol": "A256",
        "rng_domain": "assessment",
        "samples": ATTEMPT08_ASSESSMENT_SAMPLES,
        "execution_condition": "final_output_is_nonbaseline_after_X512",
        "action_scope": "locked_final_output_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "raw_paired_deltas_retained": True,
        "quantile_method": "numpy_linear",
        "diagnostics_only": True,
        "may_rerank_veto_promote_or_change_root_output": False,
        "skipped_when_final_output_is_baseline": True,
        "use": "development_go_no_go_diagnostics_only_not_realized_match_ev",
    },
    "rng_domains": ["rerank", "veto", "stress", "assessment"],
    "rng_domains_pairwise_disjoint": True,
    "candidate_selection_and_assessment_rng_independent": True,
    "opponent_private_discard_input_allowed": False,
    "opponent_profile_runtime_feature_allowed": False,
}
_EXPECTED_COST = {
    "rerank_action_futures_per_root": 1_152,
    "veto_action_futures_per_root": 1_280,
    "stress_action_futures_per_provisional_fire": 1_024,
    "assessment_action_futures_per_final_fire": 512,
    "prefire_action_futures_per_root": 2_432,
    "veto_nonfire_action_futures_per_root": 2_432,
    "stress_cancelled_nonfire_action_futures_per_root": 3_456,
    "full_final_fire_action_futures_per_root": 3_968,
}
_EXPECTED_GATES = {
    "all_gates_required": True,
    "assessment_source": (
        "disjoint_A256_locked_final_nonbaseline_output_vs_explicit_baseline"
    ),
    "fires_total_min": 40,
    "fires_each_profile_min": 3,
    "mean_delta_per_state_formula": (
        "sum_of_fired_root_A256_paired_means_divided_by_all_200_roots_"
        "with_nonfires_zero"
    ),
    "mean_delta_per_state_strictly_greater_than": 0.0,
    "mean_delta_per_fire_formula": "mean_of_fired_root_A256_paired_means",
    "mean_delta_per_fire_strictly_greater_than": 0.0,
    "false_positive_definition": (
        "final_fired_action_A256_paired_delta_mean_vs_baseline_lte_zero"
    ),
    "false_positive_rate_per_fire_max": 0.4,
    "tail_semantics": (
        "for_each_final_fired_root_compute_A256_loss_p95=max(0,-p05),"
        "loss_p99=max(0,-p01),loss_max=max(0,-min),then_take_strict_"
        "maximum_across_fired_roots_for_each_metric"
    ),
    "quantile_method": "numpy_linear",
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "action_mapping_violation_count_max": 0,
    "rng_domain_violation_count_max": 0,
    "hidden_information_violation_count_max": 0,
    "risk_reserve_contract_violation_count_max": 0,
    "locked_action_change_violation_count_max": 0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "nonfire_complete_trajectory_acceptance_deferred": True,
    "teacher_values_reported_as_realized_match_ev": False,
}
_EXPECTED_DECISION = {
    "candidate_set": "frozen_single_arm_only",
    "arm_count": 1,
    "arm_comparison_or_winner_selection_allowed": False,
    "development_threshold_reselection_allowed": False,
    "profile_exclusion_allowed": False,
    "alternate_seed_or_result_retry_allowed": False,
    "all_gates_pass_action": (
        "write_separate_development_pass_freeze_before_future_audit_"
        "authorization"
    ),
    "any_gate_fail_action": "close_attempt08_development_no_go",
    "future_audit_directly_authorized": False,
}
_EXPECTED_SEEDS = {
    "seed_stride": M43_ATTEMPT08_SEED_STRIDE,
    "hand_seed_base": _SEED_BASES["hand"],
    "rerank_seed_base": _SEED_BASES["rerank"],
    "veto_seed_base": _SEED_BASES["veto"],
    "stress_seed_base": _SEED_BASES["stress"],
    "assessment_seed_base": _SEED_BASES["assessment"],
    "child_policy_seed_base": _SEED_BASES["child"],
    "development_root_indices": "0..199_inclusive",
    "future_audit_root_indices": "200..249_inclusive",
    "per_root_formula": "namespace_seed_base + seed_stride * root_index",
    "all_twelve_development_audit_namespace_schedules_pairwise_disjoint": True,
    "attempt06_indices_0_through_49_numeric_seed_sets_disjoint": True,
    "attempt07_indices_0_through_149_numeric_seed_sets_disjoint": True,
    "attempt07_preflight_indices_0_through_2_numeric_seed_sets_disjoint": True,
    "attempt06_or_attempt07_seed_reuse_allowed": False,
}
_EXPECTED_CONTINUATION = {
    "t2_policy_id": "stage9f_p2",
    "t2_resolution": "explicit_profile_never_current",
    "t3_selector": "m3_rust_evaluate_t3",
    "t3_candidate_samples": 1,
    "t3_evaluation_samples": 1,
    "t3_downstream_samples": 1,
    "hypothetical_t4_selector": "m3_rust_evaluate_t4",
    "hypothetical_t4_candidate_samples": 1,
    "hypothetical_t4_evaluation_samples": 1,
    "real_live_t4_selector": "exact_solver",
    "real_live_t4_exact_unchanged": True,
}
_EXPECTED_BASELINE_HASH = {
    "policy_registry_path": "src/ofc_regular/ai_profiles.py",
    "policy_registry_expected_sha256": AI_PROFILES_SHA256,
    "current_mapping_changed": False,
}
_EXPECTED_GUARDS = {
    "development_generation_authorized": False,
    "development_generation_started": False,
    "future_audit_authorized": False,
    "future_audit_started": False,
    "fit_started": False,
    "threshold_selection_started": False,
    "runtime_policy_activated": False,
    "current_profile_changed": False,
    "full_replacement_enabled": False,
    "large_scale_authorized": False,
}


def load_and_validate_attempt08_plan(path: str | Path) -> dict[str, Any]:
    """Load the canonical JSON and fail closed on any contract drift."""

    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("Attempt08 plan must be a mapping")
    validate_attempt08_plan(plan)
    return plan


def validate_attempt08_plan(plan: Mapping[str, Any]) -> None:
    """Validate every frozen pre-development field and derived invariant."""

    expected_top_keys = {
        "schema",
        "milestone",
        "status_date",
        "status",
        "classification",
        "attempt07_boundary",
        "scope",
        "profiles",
        "development_population",
        "development_execution",
        "future_audit_population",
        "search_protocol",
        "single_arm_cost",
        "development_go_no_go",
        "decision_contract",
        "seed_contract",
        "fixed_continuation",
        "baseline_hash_audit",
        "activation_guards",
    }
    if set(plan) != expected_top_keys:
        raise ValueError("Attempt08 plan top-level fields changed")
    _require_exact(plan.get("schema"), M43_ATTEMPT08_PLAN_SCHEMA, "plan.schema")
    _require_exact(plan.get("milestone"), "M4.3-attempt08", "plan.milestone")
    _require_exact(plan.get("status_date"), "2026-07-14", "plan.status_date")
    _require_exact(plan.get("status"), "frozen_pre_development", "plan.status")
    _require_exact(
        plan.get("classification"),
        "single_arm_tail_safe_search_development_only",
        "plan.classification",
    )
    _require_exact(
        plan.get("attempt07_boundary"),
        _EXPECTED_ATTEMPT07_BOUNDARY,
        "attempt07_boundary",
    )
    _require_exact(plan.get("scope"), _EXPECTED_SCOPE, "scope")
    _require_exact(plan.get("profiles"), list(M43_ATTEMPT08_PROFILES), "profiles")
    _require_exact(
        plan.get("development_population"),
        _EXPECTED_DEVELOPMENT,
        "development_population",
    )
    _require_exact(
        plan.get("development_execution"),
        _EXPECTED_DEVELOPMENT_EXECUTION,
        "development_execution",
    )
    _require_exact(
        plan.get("future_audit_population"),
        _EXPECTED_FUTURE_AUDIT,
        "future_audit_population",
    )
    _require_exact(plan.get("search_protocol"), _EXPECTED_SEARCH, "search_protocol")
    _require_exact(plan.get("single_arm_cost"), _EXPECTED_COST, "single_arm_cost")
    _require_exact(
        plan.get("development_go_no_go"),
        _EXPECTED_GATES,
        "development_go_no_go",
    )
    _require_exact(
        plan.get("decision_contract"), _EXPECTED_DECISION, "decision_contract"
    )
    _require_exact(plan.get("seed_contract"), _EXPECTED_SEEDS, "seed_contract")
    _require_exact(
        plan.get("fixed_continuation"),
        _EXPECTED_CONTINUATION,
        "fixed_continuation",
    )
    _require_exact(
        plan.get("baseline_hash_audit"),
        _EXPECTED_BASELINE_HASH,
        "baseline_hash_audit",
    )
    _require_exact(
        plan.get("activation_guards"), _EXPECTED_GUARDS, "activation_guards"
    )
    _validate_single_arm_cost(plan)
    _validate_profile_balance(plan)
    validate_attempt08_seed_freshness(plan)


def validate_attempt08_artifact_bindings(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> None:
    """Verify the immutable Attempt07, Lambda, and policy-registry bytes."""

    validate_attempt08_plan(plan)
    root = Path(repository_root).resolve()
    boundary = _mapping(plan.get("attempt07_boundary"), "attempt07_boundary")
    closeout = _mapping(boundary.get("closeout"), "attempt07_boundary.closeout")
    selection = _mapping(boundary.get("selection"), "attempt07_boundary.selection")
    search = _mapping(plan.get("search_protocol"), "search_protocol")
    baseline = _mapping(plan.get("baseline_hash_audit"), "baseline_hash_audit")
    bindings = (
        (closeout["path"], closeout["sha256"], "Attempt07 closeout"),
        (selection["path"], selection["sha256"], "Attempt07 selection"),
        (
            search["candidate_generator_artifact_path"],
            search["candidate_generator_artifact_sha256"],
            "Attempt08 Lambda model",
        ),
        (
            baseline["policy_registry_path"],
            baseline["policy_registry_expected_sha256"],
            "AI profile registry",
        ),
    )
    for relative_path, expected_sha256, label in bindings:
        target = (root / str(relative_path)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes repository root") from exc
        if not target.is_file():
            raise ValueError(f"{label} artifact is missing: {relative_path}")
        actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{label} SHA-256 changed")


def enumerate_attempt08_seed_schedules(
    plan: Mapping[str, Any], *, population: str
) -> dict[str, tuple[int, ...]]:
    """Derive the six namespace schedules for one declared population."""

    seed_contract = _mapping(plan.get("seed_contract"), "seed_contract")
    stride = _strict_int(seed_contract.get("seed_stride"), "seed_stride")
    if population == "development":
        spec = _mapping(plan.get("development_population"), population)
    elif population == "future_audit":
        spec = _mapping(plan.get("future_audit_population"), population)
    else:
        raise ValueError("population must be development or future_audit")
    first = _strict_int(spec.get("root_index_first"), f"{population}.first")
    last = _strict_int(spec.get("root_index_last"), f"{population}.last")
    roots = _strict_int(spec.get("roots"), f"{population}.roots")
    if last - first + 1 != roots:
        raise ValueError(f"{population} root-index range does not match roots")
    key_by_domain = {
        "hand": "hand_seed_base",
        "rerank": "rerank_seed_base",
        "veto": "veto_seed_base",
        "stress": "stress_seed_base",
        "assessment": "assessment_seed_base",
        "child": "child_policy_seed_base",
    }
    return {
        domain: tuple(
            _strict_int(seed_contract.get(key), key) + stride * root_index
            for root_index in range(first, last + 1)
        )
        for domain, key in key_by_domain.items()
    }


def enumerate_attempt06_known_seed_schedules() -> dict[str, tuple[int, ...]]:
    """Derive the four immutable Attempt06 schedules at indices 0..49."""

    return {
        domain: tuple(
            seed_base + M43_ATTEMPT08_SEED_STRIDE * root_index
            for root_index in range(50)
        )
        for domain, seed_base in _ATTEMPT06_SEED_BASES.items()
    }


def enumerate_attempt07_known_seed_schedules() -> dict[str, tuple[int, ...]]:
    """Derive the six immutable Attempt07 schedules at indices 0..149."""

    return {
        domain: tuple(
            seed_base + M43_ATTEMPT08_SEED_STRIDE * root_index
            for root_index in range(150)
        )
        for domain, seed_base in _ATTEMPT07_SEED_BASES.items()
    }


def enumerate_attempt07_preflight_known_seed_schedules() -> dict[str, tuple[int, ...]]:
    """Derive the five opened Attempt07 preflight schedules at indices 0..2."""

    return {
        domain: tuple(
            seed_base + M43_ATTEMPT08_SEED_STRIDE * root_index
            for root_index in range(3)
        )
        for domain, seed_base in _ATTEMPT07_PREFLIGHT_SEED_BASES.items()
    }


def validate_attempt08_seed_freshness(plan: Mapping[str, Any]) -> None:
    """Prove all 12 new slices are disjoint and avoid Attempts 06 and 07."""

    declared: dict[str, tuple[int, ...]] = {}
    for population, expected_count in (("development", 200), ("future_audit", 50)):
        schedules = enumerate_attempt08_seed_schedules(plan, population=population)
        for domain, values in schedules.items():
            if len(values) != expected_count or len(set(values)) != expected_count:
                raise ValueError(f"Attempt08 {population}.{domain} is not unique")
            declared[f"{population}.{domain}"] = values
    _reject_pairwise_overlap(declared, label="Attempt08 declared seed schedules")

    for prior_label, prior_schedules in (
        ("Attempt06", enumerate_attempt06_known_seed_schedules()),
        ("Attempt07", enumerate_attempt07_known_seed_schedules()),
        (
            "Attempt07 preflight",
            enumerate_attempt07_preflight_known_seed_schedules(),
        ),
    ):
        for new_name, new_values in declared.items():
            new_set = set(new_values)
            for old_name, old_values in prior_schedules.items():
                if new_set.intersection(old_values):
                    raise ValueError(
                        f"Attempt08 {new_name} overlaps {prior_label} {old_name}"
                    )


def _validate_single_arm_cost(plan: Mapping[str, Any]) -> None:
    search = _mapping(plan.get("search_protocol"), "search_protocol")
    rerank = _mapping(search.get("rerank"), "search_protocol.rerank")
    veto = _mapping(search.get("veto"), "search_protocol.veto")
    stress = _mapping(search.get("stress"), "search_protocol.stress")
    assessment = _mapping(search.get("assessment"), "search_protocol.assessment")
    r_cost = 9 * _strict_int(rerank.get("samples"), "rerank.samples")
    v_cost = 5 * _strict_int(veto.get("samples"), "veto.samples")
    x_cost = 2 * _strict_int(stress.get("samples"), "stress.samples")
    a_cost = 2 * _strict_int(assessment.get("samples"), "assessment.samples")
    expected = {
        "rerank_action_futures_per_root": r_cost,
        "veto_action_futures_per_root": v_cost,
        "stress_action_futures_per_provisional_fire": x_cost,
        "assessment_action_futures_per_final_fire": a_cost,
        "prefire_action_futures_per_root": r_cost + v_cost,
        "veto_nonfire_action_futures_per_root": r_cost + v_cost,
        "stress_cancelled_nonfire_action_futures_per_root": r_cost + v_cost + x_cost,
        "full_final_fire_action_futures_per_root": r_cost + v_cost + x_cost + a_cost,
    }
    actual = plan.get("single_arm_cost")
    _require_exact(actual, expected, "derived single_arm_cost")


def _validate_profile_balance(plan: Mapping[str, Any]) -> None:
    profiles = tuple(plan.get("profiles", ()))
    if profiles != M43_ATTEMPT08_PROFILES:
        raise ValueError("Attempt08 profile order changed")
    for section, expected_per_profile in (
        ("development_population", 40),
        ("future_audit_population", 10),
    ):
        spec = _mapping(plan.get(section), section)
        first = _strict_int(spec.get("root_index_first"), f"{section}.first")
        last = _strict_int(spec.get("root_index_last"), f"{section}.last")
        assigned = [profiles[index % len(profiles)] for index in range(first, last + 1)]
        if any(assigned.count(profile) != expected_per_profile for profile in profiles):
            raise ValueError(f"Attempt08 {section} is not profile-balanced")


def _reject_pairwise_overlap(
    schedules: Mapping[str, Sequence[int]], *, label: str
) -> None:
    names = tuple(schedules)
    for index, left_name in enumerate(names):
        left = set(schedules[left_name])
        for right_name in names[index + 1 :]:
            if left.intersection(schedules[right_name]):
                raise ValueError(f"{label} overlap: {left_name} vs {right_name}")


def _require_exact(actual: Any, expected: Any, path: str) -> None:
    """Deep equality with exact JSON leaf types and no extra mapping keys."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{path} changed")
        if set(actual) != set(expected):
            raise ValueError(f"{path} fields changed")
        for key, expected_value in expected.items():
            _require_exact(actual[key], expected_value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} changed")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _require_exact(actual_value, expected_value, f"{path}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} changed")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


__all__ = [
    "AI_PROFILES_SHA256",
    "ATTEMPT07_CLOSEOUT_SHA256",
    "ATTEMPT07_SELECTION_SHA256",
    "ATTEMPT08_ASSESSMENT_SAMPLES",
    "ATTEMPT08_LAMBDA_MODEL_SHA256",
    "ATTEMPT08_RERANK_SAMPLES",
    "ATTEMPT08_RISK_LIMITS",
    "ATTEMPT08_SHORTLIST_K",
    "ATTEMPT08_STRESS_MIN_VALUE",
    "ATTEMPT08_STRESS_SAMPLES",
    "ATTEMPT08_TOP_K",
    "ATTEMPT08_VETO_MIN_MEAN",
    "ATTEMPT08_VETO_MIN_P01",
    "ATTEMPT08_VETO_MIN_P05",
    "ATTEMPT08_VETO_MIN_VALUE",
    "ATTEMPT08_VETO_SAMPLES",
    "M43_ATTEMPT08_PLAN_SCHEMA",
    "M43_ATTEMPT08_PLAN_SHA256",
    "M43_ATTEMPT08_PROFILES",
    "M43_ATTEMPT08_SEED_STRIDE",
    "enumerate_attempt06_known_seed_schedules",
    "enumerate_attempt07_known_seed_schedules",
    "enumerate_attempt07_preflight_known_seed_schedules",
    "enumerate_attempt08_seed_schedules",
    "load_and_validate_attempt08_plan",
    "validate_attempt08_artifact_bindings",
    "validate_attempt08_plan",
    "validate_attempt08_seed_freshness",
]
