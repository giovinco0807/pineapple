from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt08_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt08.json"


def _raw_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_attempt08_freezes_single_r128_k4_v256_x512_a256_arm() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    search = plan["search_protocol"]
    assert plan["status"] == "frozen_pre_development"
    assert plan["development_population"]["search_arm_count"] == 1
    assert search["learned_nonbaseline_top_k"] == 8
    assert search["rerank"] == {
        "symbol": "R128",
        "rng_domain": "rerank",
        "samples": 128,
        "action_scope": "rank8_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "nonbaseline_ranking": "paired_delta_mean_desc_then_ActionKey",
        "explicit_baseline_is_reference_not_nonbaseline_rank": True,
        "assessment_input_allowed": False,
    }
    assert search["shortlist"]["symbol"] == "K4"
    assert search["shortlist"]["nonbaseline_actions"] == 4
    assert search["shortlist"]["rerank_top_actions"] == 3
    assert search["shortlist"]["risk_reserve"] == {
        "candidate_R_positions_inclusive": [4, 8],
        "selection": "minimum_lambda_risk_score_then_ActionKey",
        "score": "max(downside_p95/22.0,downside_p99/36.0,downside_max/45.0)",
    }
    assert search["shortlist"]["final_nonbaseline_order"] == "original_R_order"


def test_attempt08_lambda_risk_is_raw_candidate_only_and_hash_bound() -> None:
    search = contract.load_and_validate_attempt08_plan(PLAN)["search_protocol"]
    assert search["candidate_generator_artifact_sha256"] == (
        contract.ATTEMPT08_LAMBDA_MODEL_SHA256
    )
    assert search["lambda_risk_predictions"] == {
        "source": "frozen_raw_downside_p95_p99_max_heads",
        "risk_score": (
            "max(downside_p95/22.0,downside_p99/36.0,downside_max/45.0)"
        ),
        "lower_is_safer": True,
        "conformal_upper_or_gain_gate_allowed": False,
        "may_directly_fire": False,
    }


def test_attempt08_veto_selects_first_r_ordered_safe_action() -> None:
    veto = contract.load_and_validate_attempt08_plan(PLAN)["search_protocol"][
        "veto"
    ]
    assert veto["symbol"] == "V256"
    assert veto["samples"] == 256
    assert veto["action_scope"] == "K4_nonbaseline_plus_explicit_baseline"
    assert veto["action_order"] == "original_R_order_for_K4_then_explicit_baseline"
    assert veto["selection"] == (
        "first_nonbaseline_action_in_original_R_order_passing_all_"
        "eligibility_checks"
    )
    assert veto["eligibility"] == {
        "paired_delta_mean_strictly_greater_than": 0.0,
        "paired_delta_p05_min": -22.0,
        "paired_delta_p01_min": -36.0,
        "paired_delta_min_min": -45.0,
    }
    assert veto["may_maximize_veto_mean"] is False
    assert veto["no_eligible_action"] == "exact_baseline_fallback"


def test_attempt08_stress_is_locked_cancel_only_and_assessment_is_diagnostic() -> None:
    search = contract.load_and_validate_attempt08_plan(PLAN)["search_protocol"]
    stress = search["stress"]
    assert stress["symbol"] == "X512"
    assert stress["samples"] == 512
    assert stress["action_scope"] == "locked_V256_action_plus_explicit_baseline"
    assert stress["eligibility"] == {"paired_delta_min_min": -45.0}
    assert stress["cancel_only"] is True
    assert stress["may_promote_rerank_or_switch_action"] is False
    assert stress["failed_stress_action"] == "exact_baseline_fallback"
    assessment = search["assessment"]
    assert assessment["symbol"] == "A256"
    assert assessment["samples"] == 256
    assert assessment["action_scope"] == "locked_final_output_plus_explicit_baseline"
    assert assessment["diagnostics_only"] is True
    assert assessment["may_rerank_veto_promote_or_change_root_output"] is False
    assert assessment["raw_paired_deltas_retained"] is True


def test_attempt08_rng_domains_and_fixed_compute_accounting_are_exact() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    search = plan["search_protocol"]
    assert search["rng_domains"] == ["rerank", "veto", "stress", "assessment"]
    assert search["rng_domains_pairwise_disjoint"] is True
    assert search["candidate_selection_and_assessment_rng_independent"] is True
    assert plan["single_arm_cost"] == {
        "rerank_action_futures_per_root": 1152,
        "veto_action_futures_per_root": 1280,
        "stress_action_futures_per_provisional_fire": 1024,
        "assessment_action_futures_per_final_fire": 512,
        "prefire_action_futures_per_root": 2432,
        "veto_nonfire_action_futures_per_root": 2432,
        "stress_cancelled_nonfire_action_futures_per_root": 3456,
        "full_final_fire_action_futures_per_root": 3968,
    }


def test_attempt08_development_gates_use_strict_per_fired_root_a256_tails() -> None:
    gates = contract.load_and_validate_attempt08_plan(PLAN)["development_go_no_go"]
    assert gates["fires_total_min"] == 40
    assert gates["fires_each_profile_min"] == 3
    assert gates["mean_delta_per_state_strictly_greater_than"] == 0.0
    assert gates["mean_delta_per_fire_strictly_greater_than"] == 0.0
    assert gates["false_positive_rate_per_fire_max"] == 0.4
    assert gates["tail_semantics"] == (
        "for_each_final_fired_root_compute_A256_loss_p95=max(0,-p05),"
        "loss_p99=max(0,-p01),loss_max=max(0,-min),then_take_strict_"
        "maximum_across_fired_roots_for_each_metric"
    )
    assert gates["quantile_method"] == "numpy_linear"
    assert (
        gates["override_loss_p95_max"],
        gates["override_loss_p99_max"],
        gates["override_loss_max"],
    ) == (25.0, 40.0, 50.0)
    for name in (
        "action_mapping_violation_count_max",
        "rng_domain_violation_count_max",
        "hidden_information_violation_count_max",
        "risk_reserve_contract_violation_count_max",
        "locked_action_change_violation_count_max",
    ):
        assert gates[name] == 0
    assert gates["nonfire_exact_baseline_action_fallback_required"] is True
    assert gates["nonfire_complete_trajectory_acceptance_deferred"] is True


def test_attempt08_population_is_balanced_and_future_audit_is_closed() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    profiles = tuple(plan["profiles"])
    development = [profiles[index % 5] for index in range(200)]
    future_audit = [profiles[index % 5] for index in range(200, 250)]
    assert {profile: development.count(profile) for profile in profiles} == {
        profile: 40 for profile in profiles
    }
    assert {profile: future_audit.count(profile) for profile in profiles} == {
        profile: 10 for profile in profiles
    }
    future = plan["future_audit_population"]
    assert (future["root_index_first"], future["root_index_last"]) == (200, 249)
    assert future["authorized"] is False
    assert future["started"] is False
    assert future["content_opened"] is False


def test_attempt08_all_generation_audit_fit_runtime_and_current_guards_are_false() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    assert all(value is False for value in plan["activation_guards"].values())
    assert plan["scope"]["current_profile_mutated"] is False
    assert plan["scope"]["runtime_policy_activated"] is False
    assert plan["scope"]["full_replacement_enabled"] is False
    assert plan["baseline_hash_audit"]["current_mapping_changed"] is False
    decision = plan["decision_contract"]
    assert decision["arm_comparison_or_winner_selection_allowed"] is False
    assert decision["development_threshold_reselection_allowed"] is False
    assert decision["future_audit_directly_authorized"] is False


def test_attempt08_all_twelve_new_seed_schedules_are_pairwise_disjoint() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    schedules: dict[str, tuple[int, ...]] = {}
    for population, expected_count in (("development", 200), ("future_audit", 50)):
        derived = contract.enumerate_attempt08_seed_schedules(
            plan, population=population
        )
        assert set(derived) == {
            "hand",
            "rerank",
            "veto",
            "stress",
            "assessment",
            "child",
        }
        for domain, values in derived.items():
            assert len(values) == expected_count
            assert len(set(values)) == expected_count
            schedules[f"{population}.{domain}"] = values
    assert len(schedules) == 12
    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right]), (left, right)

    stride = contract.M43_ATTEMPT08_SEED_STRIDE
    assert schedules["development.hand"] == tuple(
        80_108_071_901 + stride * index for index in range(200)
    )
    assert schedules["future_audit.hand"] == tuple(
        80_108_071_901 + stride * index for index in range(200, 250)
    )


def test_attempt08_seeds_avoid_all_opened_attempt06_and_attempt07_schedules() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    prior_groups = (
        contract.enumerate_attempt06_known_seed_schedules(),
        contract.enumerate_attempt07_known_seed_schedules(),
        contract.enumerate_attempt07_preflight_known_seed_schedules(),
    )
    assert [set(group) for group in prior_groups] == [
        {"hand", "candidate", "evaluation", "child"},
        {"hand", "screen", "rerank", "veto", "assessment", "child"},
        {"screen", "rerank", "veto", "assessment", "child"},
    ]
    assert all(len(values) == 3 for values in prior_groups[2].values())
    assert prior_groups[2]["screen"][0] == 71_106_071_901
    assert prior_groups[2]["child"][0] == 75_106_071_901
    for population in ("development", "future_audit"):
        current = contract.enumerate_attempt08_seed_schedules(
            plan, population=population
        )
        for new_values in current.values():
            for prior in prior_groups:
                for old_values in prior.values():
                    assert set(new_values).isdisjoint(old_values)


def test_attempt08_seed_freshness_rejects_new_namespace_collision() -> None:
    plan = _raw_plan()
    plan["seed_contract"]["rerank_seed_base"] = plan["seed_contract"][
        "hand_seed_base"
    ]
    with pytest.raises(ValueError, match="seed schedules overlap"):
        contract.validate_attempt08_seed_freshness(plan)


@pytest.mark.parametrize(
    "old_base,prior_label",
    [
        (17_306_071_901, "Attempt06"),
        (60_106_071_901, "Attempt07"),
        (71_106_071_901, "Attempt07 preflight"),
    ],
)
def test_attempt08_seed_freshness_rejects_prior_schedule_collision(
    old_base: int, prior_label: str
) -> None:
    plan = _raw_plan()
    plan["seed_contract"]["hand_seed_base"] = old_base
    with pytest.raises(ValueError, match=prior_label):
        contract.validate_attempt08_seed_freshness(plan)


def test_attempt08_artifact_and_plan_hashes_are_bound_to_actual_bytes() -> None:
    plan = contract.load_and_validate_attempt08_plan(PLAN)
    contract.validate_attempt08_artifact_bindings(plan, repository_root=ROOT)
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == (
        contract.M43_ATTEMPT08_PLAN_SHA256
    )
    boundary = plan["attempt07_boundary"]
    assert boundary["closeout"]["sha256"] == contract.ATTEMPT07_CLOSEOUT_SHA256
    assert boundary["selection"]["sha256"] == contract.ATTEMPT07_SELECTION_SHA256
    assert hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest() == contract.AI_PROFILES_SHA256


def test_attempt08_continuation_is_explicit_stage9f_m3_mc1_and_live_t4_exact() -> None:
    continuation = contract.load_and_validate_attempt08_plan(PLAN)[
        "fixed_continuation"
    ]
    assert continuation == {
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


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["attempt07_boundary"]["closeout"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["search_protocol"].__setitem__(
            "learned_nonbaseline_top_k", 7
        ),
        lambda value: value["search_protocol"]["lambda_risk_predictions"].__setitem__(
            "risk_score", "max(downside_p95/25.0,downside_p99/40.0,downside_max/50.0)"
        ),
        lambda value: value["search_protocol"]["shortlist"]["risk_reserve"].__setitem__(
            "selection", "minimum_lambda_risk_score_then_original_R_rank_then_ActionKey"
        ),
        lambda value: value["search_protocol"]["shortlist"].__setitem__(
            "final_nonbaseline_order", "lambda_risk_order"
        ),
        lambda value: value["search_protocol"]["veto"].__setitem__(
            "selection", "maximum_veto_mean"
        ),
        lambda value: value["search_protocol"]["veto"]["eligibility"].__setitem__(
            "paired_delta_p05_min", -25.0
        ),
        lambda value: value["search_protocol"]["stress"].__setitem__(
            "cancel_only", False
        ),
        lambda value: value["search_protocol"]["stress"]["eligibility"].__setitem__(
            "paired_delta_min_min", -50.0
        ),
        lambda value: value["search_protocol"]["assessment"].__setitem__(
            "diagnostics_only", False
        ),
        lambda value: value["single_arm_cost"].__setitem__(
            "full_final_fire_action_futures_per_root", 3456
        ),
        lambda value: value["development_population"].__setitem__("roots", 199),
        lambda value: value["development_go_no_go"].__setitem__(
            "fires_total_min", 39
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "false_positive_rate_per_fire_max", 0.5
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "override_loss_max", 55.0
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "nonfire_exact_baseline_action_fallback_required", False
        ),
        lambda value: value["future_audit_population"].__setitem__(
            "authorized", True
        ),
        lambda value: value["activation_guards"].__setitem__(
            "development_generation_authorized", True
        ),
        lambda value: value["activation_guards"].__setitem__(
            "current_profile_changed", True
        ),
        lambda value: value["scope"].__setitem__("current_profile_mutated", 0),
        lambda value: value.__setitem__("unexpected", "field"),
    ],
)
def test_attempt08_plan_rejects_adversarial_mutation(mutator) -> None:
    plan = copy.deepcopy(_raw_plan())
    mutator(plan)
    with pytest.raises(ValueError):
        contract.validate_attempt08_plan(plan)
