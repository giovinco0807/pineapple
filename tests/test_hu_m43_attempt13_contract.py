from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt13_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt13.json"
CLOSEOUT = ROOT / "configs" / "hu_joint_policy_m43_attempt12_closeout.json"


def _raw_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _raw_closeout() -> dict:
    return json.loads(CLOSEOUT.read_text(encoding="utf-8"))


def test_attempt13_plan_and_attempt12_closeout_are_hash_bound() -> None:
    plan = contract.load_and_validate_attempt13_plan(PLAN)
    closeout = contract.load_and_validate_attempt12_closeout(CLOSEOUT)

    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == (
        contract.M43_ATTEMPT13_PLAN_SHA256
    )
    assert hashlib.sha256(CLOSEOUT.read_bytes()).hexdigest() == (
        contract.M43_ATTEMPT13_CLOSEOUT_SHA256
    )
    assert plan["schema"] == contract.M43_ATTEMPT13_PLAN_SCHEMA
    assert plan["status"] == "frozen_pre_preflight"
    assert closeout["schema"] == contract.M43_ATTEMPT13_CLOSEOUT_SCHEMA
    assert closeout["status"] == "consumed_architecture_diagnostic_only"


def test_attempt12_closeout_binds_no_go_artifacts_and_forbids_promotion() -> None:
    closeout = contract.load_and_validate_attempt12_closeout(CLOSEOUT)
    artifacts = closeout["immutable_artifacts"]
    assert artifacts["decision"]["sha256"] == contract.ATTEMPT12_DECISION_SHA256
    assert artifacts["decision_receipt"]["sha256"] == (
        contract.ATTEMPT12_DECISION_RECEIPT_SHA256
    )
    assert artifacts["merged_teacher"]["sha256"] == (
        contract.ATTEMPT12_MERGED_TEACHER_SHA256
    )
    assert artifacts["receive_receipt"]["sha256"] == (
        contract.ATTEMPT12_RECEIVE_RECEIPT_SHA256
    )
    assert closeout["frozen_result"]["decision"] == "no_go"
    assert closeout["frozen_result"]["fires_total"] == 35
    assert closeout["frozen_result"]["failed_gates"] == [
        "fires_total",
        "fires_each_profile",
        "maximum_per_fired_root_max_loss",
    ]
    science = closeout["science_boundary"]
    assert science["fresh_claim"] is False
    assert science["fresh_generalization_evidence"] is False
    assert science["attempt12_future_audit_authorized"] is False
    assert science["attempt12_fit_authorized"] is False
    assert science["attempt12_runtime_authorized"] is False
    assert science["attempt12_population_authorized"] is False
    assert science["attempt12_current_profile_mutated"] is False


def test_attempt13_validates_every_frozen_artifact_binding() -> None:
    plan = contract.load_and_validate_attempt13_plan(PLAN)
    contract.validate_attempt13_artifact_bindings(plan, repository_root=ROOT)


def test_attempt13_search_teacher_has_one_relaxed_v_gate_and_one_pooled_gate() -> None:
    search = contract.load_and_validate_attempt13_plan(PLAN)["search_protocol"]
    assert search["rerank"]["symbol"] == "R128"
    assert search["rerank"]["samples"] == contract.ATTEMPT13_RERANK_SAMPLES
    assert search["shortlist"]["symbol"] == "K=min(8,n)"
    assert search["shortlist"]["nonbaseline_actions_max"] == (
        contract.ATTEMPT13_SHORTLIST_MAX
    )

    veto = search["veto"]
    assert veto["symbol"] == "V256"
    assert veto["eligibility"] == {
        "paired_delta_mean_strictly_greater_than": 0,
        "normalized_p95_p99_tail_risk_definition": (
            "max(max(0,-p05)/25.0,max(0,-p01)/40.0)"
        ),
        "normalized_p95_p99_tail_risk_max": 1.05,
        "equivalent_p05_min": -26.25,
        "equivalent_p01_min": -42,
    }
    assert veto["raw_minimum_use"] == (
        "diagnostic_only_never_filter_select_or_gate"
    )

    assert search["stress"]["symbol"] == "X1024"
    assert search["confirmation"]["symbol"] == "C1024"
    assert search["stress"]["filtering_allowed"] is False
    assert search["confirmation"]["filtering_allowed"] is False
    pooled = search["pooled_decision"]
    assert pooled["symbol"] == "P2048"
    assert pooled["source_phases"] == ["X1024", "C1024"]
    assert pooled["eligibility"]["normalized_p95_p99_tail_risk_max"] == 1
    assert pooled["eligibility"]["q001_min"] == -50
    assert pooled["eligibility"]["es01_min"] == -40
    assert pooled["eligibility"]["es01_definition"] == (
        "mean_lowest_ceil_0.01_times_2048_samples_equal_21"
    )
    assert pooled["raw_minimum_use"] == (
        "diagnostic_only_never_filter_select_or_gate"
    )
    assert pooled["selection_risk_score"] == (
        "max(max(0,-p05)/25.0,max(0,-p01)/40.0,max(0,-q001)/50.0,"
        "max(0,-es01)/40.0)"
    )


def test_attempt13_e512_is_opened_only_after_the_action_is_locked() -> None:
    search = contract.load_and_validate_attempt13_plan(PLAN)["search_protocol"]
    evaluation = search["evaluation"]
    assert evaluation["symbol"] == "E512"
    assert evaluation["samples"] == contract.ATTEMPT13_EVALUATION_SAMPLES
    assert evaluation["diagnostics_only"] is True
    assert evaluation["decision_frozen_before_namespace_open"] is True
    assert (
        evaluation["may_rerank_veto_confirm_promote_or_change_root_output"]
        is False
    )
    assert search["rng_domains_pairwise_disjoint"] is True
    assert search["candidate_selection_and_evaluation_rng_independent"] is True
    assert search["hidden_information_input_allowed"] is False


def test_attempt13_development_gates_are_not_weakened_after_attempt12() -> None:
    gates = contract.load_and_validate_attempt13_plan(PLAN)["development_go_no_go"]
    assert gates["fires_total_min"] == 40
    assert gates["fires_each_profile_min"] == 3
    assert gates["mean_delta_per_state_strictly_greater_than"] == 0
    assert gates["mean_delta_per_fire_strictly_greater_than"] == 0
    assert gates["false_positive_rate_per_fire_max"] == 0.4
    assert (
        gates["override_loss_p95_max"],
        gates["override_loss_p99_max"],
        gates["override_loss_max"],
    ) == (25, 40, 50)
    for name in (
        "action_mapping_violation_count_max",
        "rng_domain_violation_count_max",
        "hidden_information_violation_count_max",
        "risk_reserve_contract_violation_count_max",
        "retained_order_violation_count_max",
        "phase_filter_violation_count_max",
        "pooled_phase_violation_count_max",
        "extreme_tail_statistic_violation_count_max",
        "locked_action_change_violation_count_max",
    ):
        assert gates[name] == 0
    assert gates["nonfire_exact_baseline_action_fallback_required"] is True
    assert gates["raw_minimum_search_gate_allowed"] is False
    assert gates["teacher_values_reported_as_realized_match_ev"] is False


def test_attempt13_development_and_audit_are_profile_balanced_and_closed() -> None:
    plan = contract.load_and_validate_attempt13_plan(PLAN)
    profiles = tuple(plan["profiles"])
    development = [profiles[index % 5] for index in range(200)]
    audit = [profiles[index % 5] for index in range(200, 250)]
    assert {profile: development.count(profile) for profile in profiles} == {
        profile: 40 for profile in profiles
    }
    assert {profile: audit.count(profile) for profile in profiles} == {
        profile: 10 for profile in profiles
    }
    assert plan["preflight_population"]["authorized"] is False
    assert plan["future_audit_population"]["authorized"] is False
    assert plan["population_seed_reservation"]["authorized"] is False
    assert all(value is False for value in plan["activation_guards"].values())
    assert plan["baseline_hash_audit"]["current_mapping_changed"] is False


def test_attempt13_all_28_declared_seed_schedules_are_pairwise_disjoint() -> None:
    plan = contract.load_and_validate_attempt13_plan(PLAN)
    schedules: dict[str, tuple[int, ...]] = {}
    expected_counts = {
        "development": 200,
        "future_audit": 50,
        "preflight": 5,
        "population_reserved": 1000,
    }
    for population, expected in expected_counts.items():
        derived = contract.enumerate_attempt13_seed_schedules(
            plan, population=population
        )
        assert set(derived) == set(contract.ATTEMPT13_SEED_BASES)
        assert all(len(values) == expected for values in derived.values())
        schedules.update(
            {f"{population}.{domain}": values for domain, values in derived.items()}
        )
    assert len(schedules) == 28
    for (left_name, left), (right_name, right) in combinations(schedules.items(), 2):
        assert set(left).isdisjoint(right), f"{left_name} overlaps {right_name}"


def test_attempt13_declared_schedules_do_not_overlap_any_prior_registry() -> None:
    plan = contract.load_and_validate_attempt13_plan(PLAN)
    prior = contract.enumerate_attempt13_prior_seed_schedules()
    assert "Attempt12.development.hand" in prior
    assert "Attempt12.future_audit.evaluation" in prior
    assert "Attempt12.preflight.child" in prior
    assert "Attempt12.population_reserved" in prior

    for population in (
        "development",
        "future_audit",
        "preflight",
        "population_reserved",
    ):
        for domain, values in contract.enumerate_attempt13_seed_schedules(
            plan, population=population
        ).items():
            for prior_name, prior_values in prior.items():
                assert set(values).isdisjoint(prior_values), (
                    f"{population}.{domain} overlaps {prior_name}"
                )


def test_attempt13_seed_formulas_are_pinned_to_the_new_bases() -> None:
    plan = contract.load_and_validate_attempt13_plan(PLAN)
    development = contract.enumerate_attempt13_seed_schedules(
        plan, population="development"
    )
    audit = contract.enumerate_attempt13_seed_schedules(
        plan, population="future_audit"
    )
    preflight = contract.enumerate_attempt13_seed_schedules(
        plan, population="preflight"
    )
    population = contract.enumerate_attempt13_seed_schedules(
        plan, population="population_reserved"
    )
    stride = contract.M43_ATTEMPT13_SEED_STRIDE
    for domain, base in contract.ATTEMPT13_SEED_BASES.items():
        assert development[domain][0] == base
        assert development[domain][-1] == base + stride * 199
        assert audit[domain][0] == base + stride * 200
        assert audit[domain][-1] == base + stride * 249
    for domain, base in contract.ATTEMPT13_PREFLIGHT_SEED_BASES.items():
        assert preflight[domain] == tuple(base + stride * index for index in range(5))
    for domain, base in contract.ATTEMPT13_POPULATION_SEED_BASES.items():
        assert population[domain][0] == base
        assert population[domain][-1] == base + stride * 999


def test_attempt13_contract_rejects_tail_gate_or_closeout_drift() -> None:
    plan = _raw_plan()
    plan["search_protocol"]["pooled_decision"]["eligibility"]["q001_min"] = -54
    with pytest.raises(ValueError, match="plan changed|pooled"):
        contract.validate_attempt13_plan(plan)

    closeout = _raw_closeout()
    closeout["science_boundary"]["attempt12_runtime_authorized"] = True
    with pytest.raises(ValueError, match="runtime_authorized"):
        contract.validate_attempt12_closeout(closeout)


def test_attempt13_seed_validator_rejects_reallocated_prior_namespace() -> None:
    plan = _raw_plan()
    plan["preflight_seed_contract"]["hand_seed_base"] = (
        plan["seed_contract"]["hand_seed_base"]
    )
    with pytest.raises(ValueError, match="overlap"):
        contract.validate_attempt13_seed_freshness(plan)


def test_attempt13_byte_loader_rejects_reformatted_or_mutated_plan(
    tmp_path: Path,
) -> None:
    mutated = copy.deepcopy(_raw_plan())
    mutated["activation_guards"]["preflight_generation_authorized"] = True
    path = tmp_path / "attempt13.json"
    path.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        contract.load_and_validate_attempt13_plan(path)
