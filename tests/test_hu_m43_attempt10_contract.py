from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt10_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt10.json"


def _raw_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_attempt10_plan_is_hash_bound_and_frozen_pre_preflight() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == (
        contract.M43_ATTEMPT10_PLAN_SHA256
    )
    assert plan["schema"] == contract.M43_ATTEMPT10_PLAN_SCHEMA
    assert plan["status_date"] == "2026-07-15"
    assert plan["status"] == "frozen_pre_preflight"
    assert plan["development_population"]["search_arm_count"] == 1


def test_attempt10_binds_closed_attempt09_no_go_decision_and_receipt() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    boundary = plan["attempt09_boundary"]
    assert boundary["status"] == "complete_no_go_development"
    assert boundary["decision"] == {
        "path": (
            "outputs/hu_joint_policy/m43_attempt09_development/"
            "regular-hu-m43-attempt09-development200-20260715-052644/"
            "selector/decision.json"
        ),
        "sha256": contract.ATTEMPT09_DECISION_SHA256,
        "decision": "no_go",
        "status": "no_go_close_attempt09_development",
    }
    assert boundary["decision_receipt"]["sha256"] == (
        contract.ATTEMPT09_DECISION_RECEIPT_SHA256
    )
    assert boundary["decision_receipt"]["decision_sha256"] == (
        contract.ATTEMPT09_DECISION_SHA256
    )
    contract.validate_attempt10_artifact_bindings(plan, repository_root=ROOT)


def test_attempt10_freezes_top12_r128_and_unique_k8() -> None:
    search = contract.load_and_validate_attempt10_plan(PLAN)["search_protocol"]
    assert search["learned_nonbaseline_top_k"] == 12
    assert search["candidate_generator_artifact_sha256"] == (
        contract.ATTEMPT10_LAMBDA_MODEL_SHA256
    )
    assert search["rerank"] == {
        "symbol": "R128",
        "rng_domain": "rerank",
        "samples": 128,
        "action_scope": "rank12_plus_explicit_baseline",
        "common_random_futures_across_actions": True,
        "nonbaseline_ranking": "paired_delta_mean_desc_then_ActionKey",
        "explicit_baseline_is_reference_not_nonbaseline_rank": True,
        "evaluation_input_allowed": False,
    }
    assert search["shortlist"]["symbol"] == "K8"
    assert search["shortlist"]["nonbaseline_actions"] == 8
    assert search["shortlist"]["rerank_top_actions"] == 4
    assert search["shortlist"]["risk_reserve"] == {
        "candidate_R_positions_inclusive": [5, 12],
        "count": 4,
        "selection": (
            "four_minimum_lambda_risk_score_then_ActionKey_without_replacement"
        ),
        "score": (
            "max(downside_p95/22.0,downside_p99/36.0,downside_max/45.0)"
        ),
    }
    assert search["shortlist"]["final_nonbaseline_order"] == "original_R_order"


def test_attempt10_v256_is_the_frozen_coarse_k8_filter() -> None:
    veto = contract.load_and_validate_attempt10_plan(PLAN)["search_protocol"][
        "veto"
    ]
    assert veto["symbol"] == "V256"
    assert veto["samples"] == 256
    assert veto["eligibility"] == {
        "paired_delta_mean_strictly_greater_than": 0.0,
        "paired_delta_p05_min": -25.0,
        "paired_delta_p01_min": -40.0,
        "paired_delta_min_min": -50.0,
    }
    assert veto["retention"] == (
        "all_passing_nonbaseline_actions_in_original_R_order"
    )
    assert veto["may_select_final_action"] is False
    assert veto["may_rerank_candidates"] is False
    assert veto["empty_retained_set"] == "exact_baseline_fallback"


def test_attempt10_x1024_c512_strict_filter_and_e256_is_independent() -> None:
    search = contract.load_and_validate_attempt10_plan(PLAN)["search_protocol"]
    stress = search["stress"]
    assert stress["symbol"] == "X1024"
    assert stress["samples"] == 1024
    assert stress["action_scope"] == (
        "all_V256_retained_nonbaseline_actions_plus_explicit_baseline"
    )
    strict = {
        "paired_delta_mean_strictly_greater_than": 0.0,
        "paired_delta_p05_min": -22.0,
        "paired_delta_p01_min": -36.0,
        "paired_delta_min_min": -45.0,
    }
    assert stress["eligibility"] == strict
    assert stress["retention"] == (
        "all_passing_nonbaseline_actions_in_original_R_order"
    )
    assert stress["may_select_final_action"] is False

    confirmation = search["confirmation"]
    assert confirmation["symbol"] == "C512"
    assert confirmation["samples"] == 512
    assert confirmation["action_scope"] == (
        "all_X1024_retained_nonbaseline_actions_plus_explicit_baseline"
    )
    assert confirmation["eligibility"] == strict
    assert confirmation["selection"] == (
        "minimum_normalized_tail_risk_then_paired_delta_mean_desc_then_original_R_position_then_ActionKey"
    )
    assert confirmation["selection_risk_score"] == (
        "max(max(0,-paired_delta_p05)/22.0,max(0,-paired_delta_p01)/36.0,"
        "max(0,-paired_delta_min)/45.0)"
    )
    assert confirmation["empty_retained_set"] == "exact_baseline_fallback"

    evaluation = search["evaluation"]
    assert evaluation["symbol"] == "E256"
    assert evaluation["samples"] == 256
    assert evaluation["diagnostics_only"] is True
    assert evaluation["may_rerank_veto_confirm_promote_or_change_root_output"] is False
    assert search["rng_domains"] == [
        "rerank",
        "veto",
        "stress",
        "confirmation",
        "evaluation",
    ]
    assert search["rng_domains_pairwise_disjoint"] is True
    assert search["candidate_selection_and_evaluation_rng_independent"] is True


def test_attempt10_variable_candidate_compute_ceiling_is_exact() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    assert plan["single_arm_cost"] == {
        "rerank_action_futures_per_root": 1664,
        "veto_action_futures_per_root": 2304,
        "fixed_prefilter_action_futures_per_root": 3968,
        "stress_action_futures_formula": "(V_retained_nonbaseline_count+1)*1024",
        "stress_action_futures_min_when_opened": 2048,
        "stress_action_futures_max_when_opened": 9216,
        "confirmation_action_futures_formula": "(X_retained_nonbaseline_count+1)*512",
        "confirmation_action_futures_min_when_opened": 1024,
        "confirmation_action_futures_max_when_opened": 4608,
        "evaluation_action_futures_per_final_fire": 512,
        "maximum_full_fire_action_futures_per_root": 18304,
    }


def test_attempt10_go_no_go_uses_only_disjoint_e256_and_same_fixed_gates() -> None:
    gates = contract.load_and_validate_attempt10_plan(PLAN)["development_go_no_go"]
    assert gates["assessment_source"] == (
        "disjoint_E256_locked_final_nonbaseline_output_vs_explicit_baseline"
    )
    assert gates["fires_total_min"] == 40
    assert gates["fires_each_profile_min"] == 3
    assert gates["mean_delta_per_state_strictly_greater_than"] == 0.0
    assert gates["mean_delta_per_fire_strictly_greater_than"] == 0.0
    assert gates["false_positive_rate_per_fire_max"] == 0.4
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
        "retained_order_violation_count_max",
        "phase_filter_violation_count_max",
        "locked_action_change_violation_count_max",
    ):
        assert gates[name] == 0
    assert gates["nonfire_exact_baseline_action_fallback_required"] is True
    assert gates["teacher_values_reported_as_realized_match_ev"] is False


def test_attempt10_public_infoset_and_all_activation_guards_are_frozen() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    scope = plan["scope"]
    assert scope["public_information_set_only"] is True
    assert scope["opponent_private_discard_input_allowed"] is False
    assert scope["opponent_profile_runtime_feature_allowed"] is False
    assert scope["current_profile_mutated"] is False
    assert scope["runtime_policy_activated"] is False
    assert all(value is False for value in plan["activation_guards"].values())
    assert plan["baseline_hash_audit"]["current_mapping_changed"] is False


def test_attempt10_development_and_reserved_audit_are_balanced_and_closed() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    profiles = tuple(plan["profiles"])
    development = [profiles[index % 5] for index in range(200)]
    audit = [profiles[index % 5] for index in range(200, 250)]
    assert {profile: development.count(profile) for profile in profiles} == {
        profile: 40 for profile in profiles
    }
    assert {profile: audit.count(profile) for profile in profiles} == {
        profile: 10 for profile in profiles
    }
    assert plan["future_audit_population"]["authorized"] is False
    assert plan["future_audit_population"]["content_opened"] is False
    assert plan["preflight_population"]["authorized"] is False


def test_attempt10_all_21_declared_schedules_are_pairwise_disjoint() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    schedules: dict[str, tuple[int, ...]] = {}
    for population, count in (("development", 200), ("future_audit", 50)):
        derived = contract.enumerate_attempt10_seed_schedules(
            plan, population=population
        )
        assert set(derived) == {
            "hand",
            "rerank",
            "veto",
            "stress",
            "confirmation",
            "evaluation",
            "child",
        }
        assert all(len(values) == count for values in derived.values())
        schedules.update(
            {f"{population}.{domain}": values for domain, values in derived.items()}
        )
    preflight = contract.enumerate_attempt10_seed_schedules(
        plan, population="preflight"
    )
    assert all(len(values) == 3 for values in preflight.values())
    schedules.update(
        {f"preflight.{domain}": values for domain, values in preflight.items()}
    )
    assert len(schedules) == 21
    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right]), (left, right)

    stride = contract.M43_ATTEMPT10_SEED_STRIDE
    assert schedules["development.hand"] == tuple(
        120_108_071_901 + stride * index for index in range(200)
    )
    assert schedules["future_audit.hand"] == tuple(
        120_108_071_901 + stride * index for index in range(200, 250)
    )
    assert schedules["preflight.hand"] == tuple(
        130_108_071_901 + stride * index for index in range(3)
    )


def test_attempt10_schedules_are_disjoint_from_known_attempt06_through09() -> None:
    plan = contract.load_and_validate_attempt10_plan(PLAN)
    known = contract.enumerate_known_seed_schedules()
    assert any(name.startswith("Attempt06.") for name in known)
    assert any(name.startswith("Attempt07 preflight.") for name in known)
    assert any(name.startswith("Attempt08.") for name in known)
    assert any(name.startswith("Attempt08 preflight.") for name in known)
    assert any(name.startswith("Attempt09.") for name in known)
    assert any(name.startswith("Attempt09 preflight.") for name in known)
    for population in ("development", "future_audit", "preflight"):
        current = contract.enumerate_attempt10_seed_schedules(
            plan, population=population
        )
        for new_values in current.values():
            for old_values in known.values():
                assert set(new_values).isdisjoint(old_values)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda value: value["seed_contract"].__setitem__(
                "rerank_seed_base", value["seed_contract"]["hand_seed_base"]
            ),
            "declared seed schedules overlap",
        ),
        (
            lambda value: value["seed_contract"].__setitem__(
                "hand_seed_base", 80_108_071_901
            ),
            "Attempt08.hand",
        ),
        (
            lambda value: value["seed_contract"].__setitem__(
                "hand_seed_base", 100_108_071_901
            ),
            "Attempt09.hand",
        ),
        (
            lambda value: value["preflight_seed_contract"].__setitem__(
                "hand_seed_base", value["seed_contract"]["hand_seed_base"]
            ),
            "declared seed schedules overlap",
        ),
    ],
)
def test_attempt10_seed_freshness_rejects_collision(mutator, match: str) -> None:
    plan = _raw_plan()
    mutator(plan)
    with pytest.raises(ValueError, match=match):
        contract.validate_attempt10_seed_freshness(plan)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("status", "preflight_authorized"),
        lambda value: value["attempt09_boundary"]["decision"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["search_protocol"]["veto"].__setitem__(
            "retention", "first_passing_only"
        ),
        lambda value: value["search_protocol"]["stress"].__setitem__(
            "action_scope", "locked_candidate_plus_baseline"
        ),
        lambda value: value["search_protocol"]["confirmation"][
            "eligibility"
        ].__setitem__("paired_delta_min_min", -50.0),
        lambda value: value["search_protocol"]["evaluation"].__setitem__(
            "diagnostics_only", False
        ),
        lambda value: value["development_go_no_go"].__setitem__(
            "fires_total_min", 39
        ),
        lambda value: value["activation_guards"].__setitem__(
            "preflight_generation_authorized", True
        ),
        lambda value: value["scope"].__setitem__(
            "opponent_private_discard_input_allowed", True
        ),
        lambda value: value.__setitem__("unexpected", True),
    ],
)
def test_attempt10_plan_rejects_adversarial_mutation(mutator) -> None:
    plan = copy.deepcopy(_raw_plan())
    mutator(plan)
    with pytest.raises(ValueError):
        contract.validate_attempt10_plan(plan)
