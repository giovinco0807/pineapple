"""Frozen development-only contract for M4.3 Attempt07.

Attempt07 compares exactly four predeclared search arms on 100 new balanced
development roots.  It does not authorize the declared future audit, fitting,
threshold selection, runtime activation, or a change to ``current``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


M43_ATTEMPT07_PLAN_SCHEMA = "hu_m43_attempt07_development_plan_v1"
M43_ATTEMPT07_STATUS_SCHEMA = "hu_m43_attempt07_status_v1"
M43_ATTEMPT07_PLAN_SHA256 = (
    "8bf15f8d109a2e441e1c240e533c56df42a7e65d2eefd3b2750d1bd63758b189"
)
M43_ATTEMPT07_SEED_STRIDE = 1_000_003
M43_ATTEMPT07_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
M43_ATTEMPT07_ARMS = (
    "r32_v64",
    "r64_v64",
    "r32_v128",
    "r64_v128",
)
AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

_SEED_BASES = {
    "hand": 60_106_071_901,
    "screen": 61_106_071_901,
    "rerank": 62_106_071_901,
    "veto": 63_106_071_901,
    "assessment": 64_106_071_901,
    "child": 65_106_071_901,
}
_ATTEMPT06_SEED_BASES = {
    "hand": 17_306_071_901,
    "candidate": 23_306_071_901,
    "evaluation": 24_306_071_901,
    "child": 25_306_071_901,
}

_EXPECTED_SCOPE = {
    "street": "T1",
    "seat": "second",
    "fixed_baseline_profile": "stage18_p1",
    "first_seat_behavior": "unchanged_stage18_p1_baseline",
    "attempt06_status": "complete_no_go_search_quality",
    "attempt06_data_use": "development_postmortem_only",
    "attempt06_threshold_or_seed_retry_allowed": False,
    "teacher_values_are_realized_match_ev": False,
    "teacher_ev_or_lcb_runtime_gate_allowed": False,
    "current_profile_mutated": False,
    "runtime_policy_activated": False,
    "full_replacement_enabled": False,
}
_EXPECTED_DEVELOPMENT = {
    "classification": "new_balanced_development_only",
    "roots": 100,
    "root_index_first": 0,
    "root_index_last": 99,
    "profiles": 5,
    "roots_per_profile": 20,
    "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
    "candidate_arm_count": 4,
    "winner_count_max": 1,
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
    "root_generation_policy": "attempt07_explicit_mod5_per_root_policy_seed_v1",
    "root_policy_seed_formula": (
        "profile_policy_seed(per_root_hand_seed,root_profile,seat)"
    ),
    "baseline_policy_seed_formula": (
        "profile_policy_seed(per_root_hand_seed,stage18_p1,second)"
    ),
    "continuation_policy_seed_formula": "child_seed_plus_0_first_plus_1_second",
    "checkpoint_required": True,
    "heartbeat_required": True,
    "deterministic_same_contract_recompute_after_preemption_allowed": True,
    "alternate_root_or_seed_after_result_allowed": False,
    "current_profile_resolved": False,
}
_EXPECTED_FUTURE_AUDIT = {
    "classification": "declared_disjoint_one_shot_not_authorized",
    "roots": 50,
    "root_index_first": 100,
    "root_index_last": 149,
    "profiles": 5,
    "roots_per_profile": 10,
    "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
    "winner_freeze_required_before_authorization": True,
    "authorized": False,
    "started": False,
    "content_opened": False,
    "fit_allowed": False,
    "threshold_selection_allowed": False,
    "runtime_activation_allowed": False,
}
_EXPECTED_SEARCH = {
    "action_key_schema": "regular_ofc_action_key_v1",
    "candidate_generator": "lambda_rank_strict_action_scores",
    "candidate_generator_artifact_sha256": (
        "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
    ),
    "learned_nonbaseline_top_k": 8,
    "baseline_added_exactly_once": True,
    "complete_legal_action_set_required": True,
    "illegal_action_masking_required": True,
    "candidate_order": "model_score_desc_then_ActionKey_with_baseline_last",
    "screen": {
        "symbol": "S8",
        "rng_domain": "screen",
        "samples": 8,
        "action_scope": "rank8_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "may_directly_fire": False,
    },
    "shortlist": {
        "symbol": "K3",
        "nonbaseline_actions": 3,
        "selection": (
            "top_three_unique_legal_nonbaseline_actions_by_S8_mean_then_ActionKey"
        ),
        "baseline_added_exactly_once_after_shortlist": True,
        "assessment_input_allowed": False,
    },
    "rerank": {
        "rng_domain": "rerank",
        "budgets": [32, 64],
        "nested_prefix": "R32_is_exact_prefix_of_R64",
        "action_scope": "K3_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "winner_tie_break": "explicit_baseline_then_ActionKey",
        "assessment_input_allowed": False,
    },
    "veto": {
        "rng_domain": "veto",
        "budgets": [64, 128],
        "nested_prefix": "V64_is_exact_prefix_of_V128",
        "action_scope": "locked_rerank_winner_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "quantile_method": "numpy_linear",
        "eligibility": {
            "paired_delta_mean_strictly_greater_than": 0.0,
            "paired_delta_p05_min": -25.0,
            "paired_delta_p01_min": -40.0,
            "paired_delta_min_min": -50.0,
        },
        "ineligible_action": "exact_baseline_fallback",
        "assessment_input_allowed": False,
    },
    "assessment": {
        "symbol": "A128",
        "rng_domain": "assessment",
        "samples": 128,
        "action_scope": "all_rank8_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "raw_paired_deltas_retained": True,
        "quantile_method": "numpy_linear",
        "may_rerank_or_change_arm_output": False,
        "use": "development_arm_comparison_only_not_realized_match_ev",
    },
    "rng_domains_pairwise_disjoint": True,
    "opponent_private_discard_input_allowed": False,
    "opponent_profile_runtime_feature_allowed": False,
}
_EXPECTED_ARMS = [
    {
        "name": "r32_v64",
        "rerank_samples": 32,
        "veto_samples": 64,
        "screen_action_futures_per_root": 72,
        "variable_action_futures_per_root": 256,
        "assessment_action_futures_per_root": 1152,
        "total_action_futures_per_root": 1480,
    },
    {
        "name": "r64_v64",
        "rerank_samples": 64,
        "veto_samples": 64,
        "screen_action_futures_per_root": 72,
        "variable_action_futures_per_root": 384,
        "assessment_action_futures_per_root": 1152,
        "total_action_futures_per_root": 1608,
    },
    {
        "name": "r32_v128",
        "rerank_samples": 32,
        "veto_samples": 128,
        "screen_action_futures_per_root": 72,
        "variable_action_futures_per_root": 384,
        "assessment_action_futures_per_root": 1152,
        "total_action_futures_per_root": 1608,
    },
    {
        "name": "r64_v128",
        "rerank_samples": 64,
        "veto_samples": 128,
        "screen_action_futures_per_root": 72,
        "variable_action_futures_per_root": 512,
        "assessment_action_futures_per_root": 1152,
        "total_action_futures_per_root": 1736,
    },
]
_EXPECTED_GATES = {
    "all_gates_required_for_arm_eligibility": True,
    "assessment_source": "disjoint_A128_locked_arm_output_vs_explicit_baseline",
    "fires_total_min": 20,
    "fires_each_profile_min": 2,
    "mean_delta_per_state_strictly_greater_than": 0.0,
    "mean_delta_per_fire_strictly_greater_than": 0.0,
    "false_positive_definition": (
        "fired_action_A128_paired_delta_mean_vs_baseline_lte_zero"
    ),
    "false_positive_rate_per_fire_max": 0.5,
    "tail_semantics": (
        "for_each_fired_root_compute_A128_loss_from_p05_p01_min_then_take_"
        "maximum_across_fires"
    ),
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "action_mapping_violation_count_max": 0,
    "rng_domain_violation_count_max": 0,
    "hidden_information_violation_count_max": 0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "nonfire_complete_trajectory_acceptance_deferred": True,
    "teacher_values_reported_as_realized_match_ev": False,
}
_EXPECTED_WINNER_SELECTION = {
    "candidate_set": "eligible_arms_only",
    "winner_count": 1,
    "lexicographic_order": [
        "mean_delta_per_state_desc",
        "false_positive_rate_per_fire_asc",
        "tail_tuple_p95_p99_max_asc",
        "total_action_futures_per_root_asc",
        "fixed_arm_name_asc",
    ],
    "no_eligible_arm_action": "close_attempt07_development_no_go",
    "winner_action": (
        "write_separate_winner_freeze_before_future_audit_authorization"
    ),
    "development_threshold_reselection_allowed": False,
    "future_audit_directly_authorized": False,
}
_EXPECTED_SEEDS = {
    "seed_stride": M43_ATTEMPT07_SEED_STRIDE,
    "hand_seed_base": _SEED_BASES["hand"],
    "screen_seed_base": _SEED_BASES["screen"],
    "rerank_seed_base": _SEED_BASES["rerank"],
    "veto_seed_base": _SEED_BASES["veto"],
    "assessment_seed_base": _SEED_BASES["assessment"],
    "child_policy_seed_base": _SEED_BASES["child"],
    "development_root_indices": "0..99_inclusive",
    "future_audit_root_indices": "100..149_inclusive",
    "per_root_formula": "namespace_seed_base + seed_stride * root_index",
    "all_six_declared_numeric_seed_sets_pairwise_disjoint": True,
    "attempt06_indices_0_through_49_numeric_seed_sets_disjoint": True,
    "attempt06_seed_reuse_allowed": False,
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
    "development_generation_started": False,
    "winner_frozen": False,
    "future_audit_authorized": False,
    "future_audit_started": False,
    "fit_started": False,
    "threshold_selection_started": False,
    "runtime_policy_activated": False,
    "current_profile_changed": False,
    "full_replacement_enabled": False,
    "large_scale_authorized": False,
}


def load_and_validate_attempt07_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("Attempt07 plan must be a mapping")
    validate_attempt07_plan(plan)
    return plan


def load_and_validate_attempt07_status(path: str | Path) -> dict[str, Any]:
    status = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(status, dict):
        raise ValueError("Attempt07 status must be a mapping")
    validate_attempt07_status(status)
    return status


def validate_attempt07_plan(plan: Mapping[str, Any]) -> None:
    """Fail closed unless every development boundary is exactly frozen."""

    expected_top_keys = {
        "schema",
        "milestone",
        "status_date",
        "status",
        "classification",
        "scope",
        "profiles",
        "development_population",
        "development_execution",
        "future_audit_population",
        "search_protocol",
        "finite_arms",
        "development_go_no_go",
        "winner_selection",
        "seed_contract",
        "fixed_continuation",
        "baseline_hash_audit",
        "activation_guards",
    }
    if set(plan) != expected_top_keys:
        raise ValueError("Attempt07 plan top-level fields changed")
    _require_exact(plan.get("schema"), M43_ATTEMPT07_PLAN_SCHEMA, "plan.schema")
    _require_exact(plan.get("milestone"), "M4.3-attempt07", "plan.milestone")
    _require_exact(plan.get("status_date"), "2026-07-14", "plan.status_date")
    _require_exact(plan.get("status"), "frozen_pre_development", "plan.status")
    _require_exact(
        plan.get("classification"),
        "finite_search_architecture_development_only",
        "plan.classification",
    )
    _require_exact(plan.get("scope"), _EXPECTED_SCOPE, "plan.scope")
    _require_exact(plan.get("profiles"), list(M43_ATTEMPT07_PROFILES), "profiles")
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
    _require_exact(plan.get("finite_arms"), _EXPECTED_ARMS, "finite_arms")
    _require_exact(
        plan.get("development_go_no_go"), _EXPECTED_GATES, "development_go_no_go"
    )
    _require_exact(
        plan.get("winner_selection"),
        _EXPECTED_WINNER_SELECTION,
        "winner_selection",
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
    _validate_arm_costs(plan)
    validate_attempt07_seed_freshness(plan)
    _validate_profile_balance(plan)


def validate_attempt07_status(status: Mapping[str, Any]) -> None:
    """Validate that Attempt07 remains development-only and unopened."""

    expected = {
        "schema": M43_ATTEMPT07_STATUS_SCHEMA,
        "milestone": "M4.3",
        "attempt": "attempt07",
        "status_date": "2026-07-14",
        "status": "frozen_pre_development",
        "decision": "finite_four_arm_development_only",
        "plan": {
            "path": "configs/hu_joint_policy_m43_attempt07.json",
            "schema": M43_ATTEMPT07_PLAN_SCHEMA,
            "sha256": M43_ATTEMPT07_PLAN_SHA256,
        },
        "attempt06_boundary": {
            "status": "complete_no_go_search_quality",
            "closeout_path": "configs/hu_joint_policy_m43_attempt06_closeout.json",
            "data_use": "development_postmortem_only",
            "threshold_or_seed_retry_allowed": False,
            "promotion_allowed": False,
        },
        "frozen_development_protocol": {
            "roots": 100,
            "roots_per_profile": 20,
            "candidate_set": "rank8_plus_explicit_baseline",
            "screen": "S8",
            "shortlist": "K3_nonbaseline_plus_explicit_baseline",
            "nested_rerank_budgets": [32, 64],
            "nested_veto_budgets": [64, 128],
            "assessment": "disjoint_A128_all_rank8_plus_explicit_baseline",
            "shard_layout": "one_root_per_shard",
            "batch_child_selectors_required": True,
            "native_batch_threads": 4,
            "finite_arms": list(M43_ATTEMPT07_ARMS),
            "winner": None,
            "generation_status": "not_started",
        },
        "future_audit": {
            "classification": "declared_disjoint_one_shot_not_authorized",
            "roots": 50,
            "roots_per_profile": 10,
            "root_index_first": 100,
            "root_index_last": 149,
            "winner_freeze_required": True,
            "authorized": False,
            "started": False,
            "content_opened": False,
        },
        "preflight": {
            "local_correctness": "not_started",
            "determinism": "not_started",
            "scalar_batch_exact_parity": "not_started",
            "latency_profile": "not_started",
        },
        "execution": {
            "local_pilot_started": False,
            "spot_authorized": False,
            "spot_started": False,
            "fit_performed": False,
            "threshold_selected": False,
            "runtime_model_frozen": False,
            "runtime_enabled": False,
        },
        "baseline_hash_audit": _EXPECTED_BASELINE_HASH,
        "guards": _EXPECTED_GUARDS,
        "next_action": (
            "implement_and_preflight_attempt07_teacher_without_opening_future_audit"
        ),
    }
    _require_exact(status, expected, "Attempt07 status")


def enumerate_attempt07_seed_schedules(
    plan: Mapping[str, Any], *, population: str
) -> dict[str, tuple[int, ...]]:
    """Derive one seed sequence per namespace for a declared population."""

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
        "screen": "screen_seed_base",
        "rerank": "rerank_seed_base",
        "veto": "veto_seed_base",
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
    """Derive the four immutable Attempt06 0..49 schedules."""

    return {
        domain: tuple(
            seed_base + M43_ATTEMPT07_SEED_STRIDE * root_index
            for root_index in range(50)
        )
        for domain, seed_base in _ATTEMPT06_SEED_BASES.items()
    }


def validate_attempt07_seed_freshness(plan: Mapping[str, Any]) -> None:
    """Prove all 12 declared slices are disjoint and avoid Attempt06."""

    declared: dict[str, tuple[int, ...]] = {}
    for population in ("development", "future_audit"):
        schedules = enumerate_attempt07_seed_schedules(plan, population=population)
        for domain, values in schedules.items():
            expected_count = 100 if population == "development" else 50
            if len(values) != expected_count or len(set(values)) != expected_count:
                raise ValueError(f"Attempt07 {population}.{domain} is not unique")
            declared[f"{population}.{domain}"] = values
    _reject_pairwise_overlap(declared, label="Attempt07 declared seed schedules")

    attempt06 = enumerate_attempt06_known_seed_schedules()
    for new_name, new_values in declared.items():
        new_set = set(new_values)
        for old_name, old_values in attempt06.items():
            if new_set.intersection(old_values):
                raise ValueError(
                    f"Attempt07 {new_name} overlaps Attempt06 {old_name}"
                )


def _validate_arm_costs(plan: Mapping[str, Any]) -> None:
    arms = plan.get("finite_arms")
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        raise ValueError("finite_arms must be a sequence")
    for raw_arm in arms:
        arm = _mapping(raw_arm, "finite_arm")
        rerank = _strict_int(arm.get("rerank_samples"), "rerank_samples")
        veto = _strict_int(arm.get("veto_samples"), "veto_samples")
        expected_screen = 9 * 8
        expected_variable = 4 * rerank + 2 * veto
        expected_assessment = 9 * 128
        expected_total = expected_screen + expected_variable + expected_assessment
        actual = (
            arm.get("screen_action_futures_per_root"),
            arm.get("variable_action_futures_per_root"),
            arm.get("assessment_action_futures_per_root"),
            arm.get("total_action_futures_per_root"),
        )
        expected = (
            expected_screen,
            expected_variable,
            expected_assessment,
            expected_total,
        )
        if actual != expected or any(type(value) is not int for value in actual):
            raise ValueError(f"Attempt07 arm cost changed: {arm.get('name')}")


def _validate_profile_balance(plan: Mapping[str, Any]) -> None:
    profiles = tuple(plan.get("profiles", ()))
    if profiles != M43_ATTEMPT07_PROFILES:
        raise ValueError("Attempt07 profile order changed")
    for section, expected_per_profile in (
        ("development_population", 20),
        ("future_audit_population", 10),
    ):
        spec = _mapping(plan.get(section), section)
        first = _strict_int(spec.get("root_index_first"), f"{section}.first")
        last = _strict_int(spec.get("root_index_last"), f"{section}.last")
        assigned = [profiles[index % len(profiles)] for index in range(first, last + 1)]
        if any(assigned.count(profile) != expected_per_profile for profile in profiles):
            raise ValueError(f"Attempt07 {section} is not profile-balanced")


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
