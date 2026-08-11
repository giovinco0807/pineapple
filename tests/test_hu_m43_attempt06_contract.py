from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt06_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt06.json"
STATUS = ROOT / "configs" / "hu_joint_policy_m43_attempt06_status.json"
ATTEMPT05_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt05.json"
ATTEMPT05_STATUS = ROOT / "configs" / "hu_joint_policy_m43_attempt05_status.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt06_freezes_corrected_rank8_pre_fresh_contract() -> None:
    plan = contract.load_and_validate_attempt06_plan(PLAN)
    assert plan["status"] == "frozen_preflight"
    correction = plan["coverage_metric_correction"]
    invalid = correction["invalid_absolute_gate"]
    assert invalid["overall_min"] == pytest.approx(0.70)
    assert invalid["overall_mathematical_ceiling"] == pytest.approx(565 / 900)
    assert invalid["overall_mathematical_ceiling"] < invalid["overall_min"]
    assert invalid["minimum_profile_mathematical_ceiling"] == pytest.approx(
        107 / 180
    )
    assert invalid["each_profile_gate_reachable"] is False

    metric = correction["corrected_metric"]
    assert metric["name"] == "rank8_conditional_positive_recall"
    assert metric["overall_min"] == pytest.approx(0.95)
    assert metric["each_profile_min"] == pytest.approx(0.93)
    assert metric["fresh_search_quality_audit_gate"] is False
    observed = correction["observed_multiple_use_dev900"]
    assert observed["overall"]["conditional_recall"] == pytest.approx(541 / 565)
    profile_recalls = [
        row["rank8_covered_opportunity_states"] / row["opportunity_states"]
        for row in observed["profile"].values()
    ]
    assert min(profile_recalls) == pytest.approx(108 / 115)
    assert min(profile_recalls) >= metric["each_profile_min"]
    assert observed["acceptance_or_runtime_claim_allowed"] is False


def test_attempt06_binds_lambda_artifact_and_multiple_use_design_evidence() -> None:
    plan = contract.load_and_validate_attempt06_plan(PLAN)
    evidence = plan["design_evidence"]
    architecture = evidence["architecture_report"]
    diagnostic = evidence["candidate_set_diagnostic"]
    generator = evidence["candidate_generator"]
    assert _sha256(ROOT / architecture["path"]) == architecture["sha256"]
    assert _sha256(ROOT / diagnostic["path"]) == diagnostic["sha256"]
    assert _sha256(ROOT / generator["artifact_path"]) == generator["artifact_sha256"]
    assert generator["artifact_sha256"] == contract.M43_ATTEMPT06_LAMBDA_SHA256
    assert architecture["selected_family"] is None
    assert diagnostic["fresh_generalization_claim_allowed"] is False
    assert generator["runtime_model_frozen"] is False
    assert generator["runtime_enabled"] is False


def test_attempt06_freezes_top8_c8_e128_and_continuation_boundary() -> None:
    plan = contract.load_and_validate_attempt06_plan(PLAN)
    search = plan["search_contract"]
    assert search["teacher_schema"] == (
        "hu_m43_attempt06_t1_second_top8_c8_e128_teacher_v3"
    )
    assert search["learned_nonbaseline_top_k"] == 8
    assert search["candidate_selection_samples"] == 8
    assert search["independent_evaluation_samples"] == 128
    assert search["independent_evaluation_raw_paired_deltas_retained"] is True
    assert search["independent_evaluation_raw_count"] == 128
    assert search["raw_summary_recomputed_with"] == "numpy_linear"
    assert search["independent_evaluation_action_scope"] == (
        "locked_action_plus_explicit_baseline_only"
    )
    assert search["unselected_candidate_evaluation_fields"] == "null"
    assert search["evaluation_pair_best_is_diagnostic_only"] is True
    assert search["candidate_selected_before_evaluation_opened"] is True
    assert search["evaluation_may_rerank"] is False
    assert search["opponent_private_discard_input_allowed"] is False

    pre_spot = plan["pre_spot_requirements"]
    assert pre_spot["native_batch_threads"] == 4
    assert pre_spot["batch_child_selectors_required"] is True
    assert pre_spot["config_sha256_contract"] == (
        "sha256_of_canonical_attempt06_fixed_contract"
    )
    assert pre_spot["config_sha256_cross_bind_required"] == [
        "teacher_row_provenance",
        "checkpoint",
        "heartbeat",
        "generator_summary",
        "done",
    ]

    continuation = plan["fixed_continuation"]
    assert continuation["t2_policy_id"] == "stage9f_p2"
    assert continuation["t2_resolution"] == "explicit_profile_never_current"
    assert continuation["hypothetical_t1_search_direct_t4_mode"] == "counter_mc_1"
    assert continuation["hypothetical_nested_t4_mode"] == "counter_mc_1"
    assert continuation["real_live_t4_selector"] == "exact_solver"
    assert continuation["real_live_t4_exact_unchanged"] is True


def test_attempt06_transfers_all_four_per_root_seed_namespaces_without_overlap() -> None:
    plan = contract.load_and_validate_attempt06_plan(PLAN)
    attempt05 = json.loads(ATTEMPT05_PLAN.read_text(encoding="utf-8"))
    source = attempt05["development_roles"]["pilot_audit"]
    transfer = plan["audit_schedule_transfer"]
    assert transfer["source_role"] == "development_roles.pilot_audit"
    assert transfer["roots"] == source["roots"] == 50
    assert transfer["roots_per_profile"] == source["roots_per_profile"] == 10
    assert transfer["hand_seed_start"] == source["seed_start"] == 17306071901
    assert transfer["candidate_seed_start"] == source["candidate_seed_start"]
    assert transfer["evaluation_seed_start"] == source["evaluation_seed_start"]
    assert transfer["child_policy_seed_start"] == source["child_policy_seed_start"]
    assert transfer["source_role_started"] is False
    assert transfer["source_role_consumed"] is False
    assert transfer["attempt05_reuse_after_transfer_allowed"] is False

    # Operational correction: one root is the checkpoint/heartbeat/preemption unit.
    assert transfer["shards"] == 50
    assert transfer["roots_per_shard"] == 1
    assert transfer["candidate_evaluation_child_seed_count"] == 50
    schedules = contract.enumerate_attempt06_seed_schedules(plan)
    assert set(schedules) == {"hand", "candidate", "evaluation", "child"}
    starts = {
        "hand": 17306071901,
        "candidate": 23306071901,
        "evaluation": 24306071901,
        "child": 25306071901,
    }
    for namespace, values in schedules.items():
        assert len(values) == len(set(values)) == 50
        assert values == tuple(
            starts[namespace] + contract.M43_ATTEMPT06_SEED_STRIDE * index
            for index in range(50)
        )
    for left_index, (left_name, left) in enumerate(schedules.items()):
        for right_name, right in list(schedules.items())[left_index + 1 :]:
            assert set(left).isdisjoint(right), f"{left_name} overlaps {right_name}"
    assert len(set().union(*(set(values) for values in schedules.values()))) == 200

    profiles = tuple(transfer["profiles"])
    assigned = [profiles[index % len(profiles)] for index in range(50)]
    assert {profile: assigned.count(profile) for profile in profiles} == {
        profile: 10 for profile in profiles
    }
    attempt05_status = json.loads(ATTEMPT05_STATUS.read_text(encoding="utf-8"))
    assert attempt05_status["development_roles"]["pilot_audit"].startswith(
        "not_started"
    )


def test_attempt06_one_shot_go_only_allows_a_separate_future_freeze() -> None:
    plan = contract.load_and_validate_attempt06_plan(PLAN)
    audit = plan["one_shot_search_quality_audit"]
    assert audit["fit_allowed"] is False
    assert audit["threshold_selection_allowed"] is False
    assert audit["larger_generation_or_acceptance_authorized_by_go"] is False
    assert audit["fresh_200_root_fit_or_distillation_authorized_by_this_plan"] is False
    assert audit["go_authorizes_only"] == (
        "create_a_separate_freeze_for_fresh_200_root_fit_and_distillation"
    )

    gates = plan["search_quality_go_no_go"]
    assert gates["fires_total_min"] == 10
    assert gates["fires_each_profile_min"] == 1
    assert gates["mean_delta_per_state_strictly_greater_than"] == 0.0
    assert gates["mean_delta_per_fire_strictly_greater_than"] == 0.0
    assert gates["false_positive_rate_per_fire_max"] == pytest.approx(0.50)
    tail = gates["tail_gate_semantics"]
    assert tail["variant"] == (
        "B_per_fired_root_independent_e128_paired_future_tail_then_max_across_fires"
    )
    assert tail["paired_future_count_per_fired_root"] == 128
    assert tail["per_fired_root_metrics"] == {
        "p95_loss": "max(0,-p05(independent_e128_paired_future_deltas))",
        "p99_loss": "max(0,-p01(independent_e128_paired_future_deltas))",
        "max_loss": "max(0,-min(independent_e128_paired_future_deltas))",
    }
    assert tail["audit_gate_aggregation"] == {
        "p95_loss": "maximum_per_fired_root_p95_loss",
        "p99_loss": "maximum_per_fired_root_p99_loss",
        "max_loss": "maximum_per_fired_root_max_loss",
    }
    assert tail["paired_mean_cross_fire_tail"]["classification"] == (
        "diagnostic_only_not_gate"
    )
    assert (
        gates["override_loss_p95_max"],
        gates["override_loss_p99_max"],
        gates["override_loss_max"],
    ) == (25.0, 40.0, 50.0)
    assert gates["action_mapping_violation_count_max"] == 0
    assert gates["rng_domain_violation_count_max"] == 0
    assert gates["hidden_information_violation_count_max"] == 0
    assert gates["go_directly_authorizes_fresh_200_root_fit_or_distillation"] is False


def test_attempt06_status_is_frozen_preflight_with_spot_and_runtime_off() -> None:
    status = contract.load_and_validate_attempt06_status(STATUS)
    assert status["status"] == "frozen_preflight"
    assert _sha256(ROOT / status["plan"]["path"]) == status["plan"]["sha256"]
    assert status["plan"]["sha256"] == contract.M43_ATTEMPT06_PLAN_SHA256
    assert set(status["preflight"].values()) == {"not_started"}
    assert status["seed_transfer"]["per_root_seed_tuple_count"] == 50
    assert status["one_shot_search_quality_audit"]["status"] == "not_started"
    assert all(value is False for value in status["spot_execution"].values())
    assert all(value is False for value in status["guards"].values())
    assert status["frozen_design"]["runtime_model_frozen"] is False
    assert status["frozen_design"]["runtime_enabled"] is False
    assert status["frozen_design"]["tail_gate_semantics"] == (
        "per_fired_root_e128_p05_p01_min_then_maximum_across_fires"
    )
    assert status["frozen_design"]["paired_mean_cross_fire_quantiles_are_gate"] is False


def test_attempt06_contract_binds_unchanged_ai_profiles_registry() -> None:
    contract.load_and_validate_attempt06_plan(PLAN)
    assert _sha256(ROOT / "src" / "ofc_regular" / "ai_profiles.py") == (
        contract.AI_PROFILES_SHA256
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["coverage_metric_correction"]["corrected_metric"].__setitem__(
                "overall_min", 0.94
            ),
            "conditional-recall gate",
        ),
        (
            lambda value: value["search_contract"].__setitem__(
                "learned_nonbaseline_top_k", 4
            ),
            "top8/c8/e128",
        ),
        (
            lambda value: value["fixed_continuation"].__setitem__(
                "real_live_t4_exact_unchanged", False
            ),
            "continuation boundary",
        ),
        (
            lambda value: value["audit_schedule_transfer"].__setitem__("shards", 5),
            "50-root seed schedule",
        ),
        (
            lambda value: value["one_shot_search_quality_audit"].__setitem__(
                "fit_allowed", True
            ),
            "one-shot audit role",
        ),
        (
            lambda value: value["search_quality_go_no_go"].__setitem__(
                "go_directly_authorizes_fresh_200_root_fit_or_distillation", True
            ),
            "Go/No-Go",
        ),
        (
            lambda value: value["search_quality_go_no_go"][
                "tail_gate_semantics"
            ]["audit_gate_aggregation"].__setitem__(
                "p95_loss", "numpy_p95_across_fired_root_means"
            ),
            "Go/No-Go",
        ),
        (
            lambda value: value["pre_spot_requirements"].__setitem__(
                "spot_authorized_at_freeze", True
            ),
            "Spot",
        ),
    ],
)
def test_attempt06_plan_rejects_frozen_boundary_drift(mutator, message: str) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    mutator(plan)
    with pytest.raises(ValueError, match=message):
        contract.validate_attempt06_plan(plan)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["preflight"].__setitem__(
                "local_correctness", "passed"
            ),
            "preflight",
        ),
        (
            lambda value: value["spot_execution"].__setitem__("authorized", True),
            "Spot",
        ),
        (
            lambda value: value["guards"].__setitem__(
                "runtime_policy_activated", True
            ),
            "activation guard",
        ),
    ],
)
def test_attempt06_status_rejects_premature_progress(mutator, message: str) -> None:
    status = copy.deepcopy(json.loads(STATUS.read_text(encoding="utf-8")))
    mutator(status)
    with pytest.raises(ValueError, match=message):
        contract.validate_attempt06_status(status)
