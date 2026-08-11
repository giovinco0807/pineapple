from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt07_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
STATUS = ROOT / "configs" / "hu_joint_policy_m43_attempt07_status.json"


def _raw_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _raw_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8"))


def test_attempt07_freezes_four_arm_development_protocol() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    search = plan["search_protocol"]
    assert plan["development_population"]["roots"] == 100
    assert plan["development_population"]["roots_per_profile"] == 20
    assert search["learned_nonbaseline_top_k"] == 8
    assert search["screen"]["symbol"] == "S8"
    assert search["shortlist"]["symbol"] == "K3"
    assert search["rerank"]["budgets"] == [32, 64]
    assert search["rerank"]["nested_prefix"] == "R32_is_exact_prefix_of_R64"
    assert search["rerank"]["winner_tie_break"] == (
        "explicit_baseline_then_ActionKey"
    )
    assert search["veto"]["budgets"] == [64, 128]
    assert search["veto"]["nested_prefix"] == "V64_is_exact_prefix_of_V128"
    assert search["assessment"] == {
        "symbol": "A128",
        "rng_domain": "assessment",
        "samples": 128,
        "action_scope": "all_rank8_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "raw_paired_deltas_retained": True,
        "quantile_method": "numpy_linear",
        "may_rerank_or_change_arm_output": False,
        "use": "development_arm_comparison_only_not_realized_match_ev",
    }
    assert tuple(arm["name"] for arm in plan["finite_arms"]) == (
        contract.M43_ATTEMPT07_ARMS
    )


def test_attempt07_veto_and_development_gates_are_exact() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    assert plan["search_protocol"]["veto"]["eligibility"] == {
        "paired_delta_mean_strictly_greater_than": 0.0,
        "paired_delta_p05_min": -25.0,
        "paired_delta_p01_min": -40.0,
        "paired_delta_min_min": -50.0,
    }
    gates = plan["development_go_no_go"]
    assert gates["fires_total_min"] == 20
    assert gates["fires_each_profile_min"] == 2
    assert gates["mean_delta_per_state_strictly_greater_than"] == 0.0
    assert gates["mean_delta_per_fire_strictly_greater_than"] == 0.0
    assert gates["false_positive_rate_per_fire_max"] == 0.5
    assert (
        gates["override_loss_p95_max"],
        gates["override_loss_p99_max"],
        gates["override_loss_max"],
    ) == (25.0, 40.0, 50.0)
    assert gates["action_mapping_violation_count_max"] == 0
    assert gates["rng_domain_violation_count_max"] == 0
    assert gates["hidden_information_violation_count_max"] == 0
    assert gates["nonfire_exact_baseline_action_fallback_required"] is True
    assert gates["nonfire_complete_trajectory_acceptance_deferred"] is True


def test_attempt07_development_execution_is_one_root_batched_native4() -> None:
    execution = contract.load_and_validate_attempt07_plan(PLAN)[
        "development_execution"
    ]
    assert execution == {
        "shard_layout": "one_root_per_shard",
        "batch_child_selectors_required": True,
        "native_batch_threads": 4,
        "opening_lookahead_samples": 0,
        "root_generation_policy": (
            "attempt07_explicit_mod5_per_root_policy_seed_v1"
        ),
        "root_policy_seed_formula": (
            "profile_policy_seed(per_root_hand_seed,root_profile,seat)"
        ),
        "baseline_policy_seed_formula": (
            "profile_policy_seed(per_root_hand_seed,stage18_p1,second)"
        ),
        "continuation_policy_seed_formula": (
            "child_seed_plus_0_first_plus_1_second"
        ),
        "checkpoint_required": True,
        "heartbeat_required": True,
        "deterministic_same_contract_recompute_after_preemption_allowed": True,
        "alternate_root_or_seed_after_result_allowed": False,
        "current_profile_resolved": False,
    }


def test_attempt07_winner_selection_is_finite_and_lexicographic() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    selection = plan["winner_selection"]
    assert selection["candidate_set"] == "eligible_arms_only"
    assert selection["winner_count"] == 1
    assert selection["lexicographic_order"] == [
        "mean_delta_per_state_desc",
        "false_positive_rate_per_fire_asc",
        "tail_tuple_p95_p99_max_asc",
        "total_action_futures_per_root_asc",
        "fixed_arm_name_asc",
    ]
    assert selection["development_threshold_reselection_allowed"] is False
    assert selection["future_audit_directly_authorized"] is False


def test_attempt07_continuation_is_explicit_stage9f_m3_mc1_and_live_t4_exact() -> None:
    continuation = contract.load_and_validate_attempt07_plan(PLAN)[
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


def test_attempt07_arm_costs_match_fixed_action_scopes() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    assert {
        arm["name"]: (
            arm["variable_action_futures_per_root"],
            arm["total_action_futures_per_root"],
        )
        for arm in plan["finite_arms"]
    } == {
        "r32_v64": (256, 1480),
        "r64_v64": (384, 1608),
        "r32_v128": (384, 1608),
        "r64_v128": (512, 1736),
    }


def test_attempt07_all_declared_seed_slices_are_pairwise_disjoint() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    schedules: dict[str, tuple[int, ...]] = {}
    for population, expected_count in (("development", 100), ("future_audit", 50)):
        derived = contract.enumerate_attempt07_seed_schedules(
            plan, population=population
        )
        assert set(derived) == {
            "hand",
            "screen",
            "rerank",
            "veto",
            "assessment",
            "child",
        }
        for domain, values in derived.items():
            assert len(values) == expected_count
            assert len(set(values)) == expected_count
            schedules[f"{population}.{domain}"] = values

    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right]), (left, right)

    stride = contract.M43_ATTEMPT07_SEED_STRIDE
    assert schedules["development.hand"] == tuple(
        60_106_071_901 + stride * index for index in range(100)
    )
    assert schedules["future_audit.hand"] == tuple(
        60_106_071_901 + stride * index for index in range(100, 150)
    )


def test_attempt07_seed_slices_do_not_overlap_attempt06_known_schedules() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    attempt06 = contract.enumerate_attempt06_known_seed_schedules()
    assert set(attempt06) == {"hand", "candidate", "evaluation", "child"}
    for population in ("development", "future_audit"):
        attempt07 = contract.enumerate_attempt07_seed_schedules(
            plan, population=population
        )
        for new_values in attempt07.values():
            for old_values in attempt06.values():
                assert set(new_values).isdisjoint(old_values)


def test_attempt07_seed_freshness_rejects_cross_namespace_collision() -> None:
    plan = _raw_plan()
    plan["seed_contract"]["screen_seed_base"] = plan["seed_contract"][
        "hand_seed_base"
    ]
    with pytest.raises(ValueError, match="seed schedules overlap"):
        contract.validate_attempt07_seed_freshness(plan)


def test_attempt07_populations_are_balanced_and_future_audit_is_unopened() -> None:
    plan = contract.load_and_validate_attempt07_plan(PLAN)
    profiles = tuple(plan["profiles"])
    dev = [profiles[index % 5] for index in range(100)]
    audit = [profiles[index % 5] for index in range(100, 150)]
    assert {profile: dev.count(profile) for profile in profiles} == {
        profile: 20 for profile in profiles
    }
    assert {profile: audit.count(profile) for profile in profiles} == {
        profile: 10 for profile in profiles
    }
    future = plan["future_audit_population"]
    assert future["classification"] == "declared_disjoint_one_shot_not_authorized"
    assert future["winner_freeze_required_before_authorization"] is True
    assert future["authorized"] is False
    assert future["started"] is False
    assert future["content_opened"] is False


def test_attempt07_status_keeps_fit_runtime_current_and_audit_closed() -> None:
    status = contract.load_and_validate_attempt07_status(STATUS)
    assert status["status"] == "frozen_pre_development"
    assert status["frozen_development_protocol"]["winner"] is None
    assert status["future_audit"]["authorized"] is False
    assert all(value is False for value in status["execution"].values())
    assert all(value is False for value in status["guards"].values())
    assert status["baseline_hash_audit"]["current_mapping_changed"] is False


def test_attempt07_plan_hash_and_ai_profiles_hash_are_bound() -> None:
    contract.load_and_validate_attempt07_plan(PLAN)
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == (
        contract.M43_ATTEMPT07_PLAN_SHA256
    )
    status = contract.load_and_validate_attempt07_status(STATUS)
    assert status["plan"]["sha256"] == contract.M43_ATTEMPT07_PLAN_SHA256
    assert hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest() == contract.AI_PROFILES_SHA256


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["search_protocol"].__setitem__(
            "learned_nonbaseline_top_k", 7
        ),
        lambda value: value["search_protocol"]["screen"].__setitem__("samples", 16),
        lambda value: value["search_protocol"]["shortlist"].__setitem__(
            "nonbaseline_actions", 4
        ),
        lambda value: value["search_protocol"]["rerank"].__setitem__(
            "budgets", [32, 96]
        ),
        lambda value: value["search_protocol"]["veto"]["eligibility"].__setitem__(
            "paired_delta_p05_min", -30.0
        ),
        lambda value: value["search_protocol"]["assessment"].__setitem__(
            "may_rerank_or_change_arm_output", True
        ),
        lambda value: value["development_execution"].__setitem__(
            "native_batch_threads", 8
        ),
        lambda value: value["finite_arms"][0].__setitem__("rerank_samples", 16),
        lambda value: value["development_go_no_go"].__setitem__(
            "fires_total_min", 19
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "mean_delta_per_state_strictly_greater_than", -0.1
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "override_loss_p99_max", 45.0
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "nonfire_exact_baseline_action_fallback_required", False
        ),
        lambda value: value["winner_selection"]["lexicographic_order"].reverse(),
        lambda value: value["seed_contract"].__setitem__(
            "hand_seed_base", 17_306_071_901
        ),
        lambda value: value["future_audit_population"].__setitem__(
            "authorized", True
        ),
        lambda value: value["activation_guards"].__setitem__(
            "current_profile_changed", True
        ),
        lambda value: value["scope"].__setitem__("current_profile_mutated", 0),
        lambda value: value.__setitem__("unexpected", "field"),
    ],
)
def test_attempt07_plan_rejects_adversarial_mutation(mutator) -> None:
    plan = copy.deepcopy(_raw_plan())
    mutator(plan)
    with pytest.raises(ValueError):
        contract.validate_attempt07_plan(plan)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["frozen_development_protocol"].__setitem__(
            "winner", "r32_v64"
        ),
        lambda value: value["future_audit"].__setitem__("authorized", True),
        lambda value: value["execution"].__setitem__("fit_performed", True),
        lambda value: value["guards"].__setitem__("runtime_policy_activated", True),
        lambda value: value["baseline_hash_audit"].__setitem__(
            "current_mapping_changed", True
        ),
        lambda value: value["plan"].__setitem__("sha256", "0" * 64),
        lambda value: value.__setitem__("unexpected", False),
    ],
)
def test_attempt07_status_rejects_claimed_execution_or_drift(mutator) -> None:
    status = copy.deepcopy(_raw_status())
    mutator(status)
    with pytest.raises(ValueError):
        contract.validate_attempt07_status(status)
