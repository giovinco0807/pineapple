from __future__ import annotations

from copy import deepcopy

import pytest

import ofc_regular.validate_hu_m43_attempt04_population as validator
from ofc_regular.hu_m43_attempt04_runtime import self_digest
from ofc_regular.hu_m43_joint_model_v6 import (
    HU_M43_V6_ACTION_SCORE_MODE,
    HU_M43_V6_ARTIFACT_SCHEMA,
    HU_M43_V6_MODEL_SCHEMA,
    HU_M43_V6_PROPOSAL_SCHEMA,
    HU_M43_V6_SAFETY_FEATURE_DIM,
    HU_M43_V6_SAFETY_FEATURE_SCHEMA,
)
from ofc_regular.validate_hu_m43_attempt02_acceptance import FIXED_GATES
from ofc_regular.validate_hu_m43_attempt04_population import (
    ATTEMPT04_POPULATION_MANIFEST_SCHEMA,
    ATTEMPT04_POPULATION_PREFLIGHT_SCHEMA,
    ATTEMPT04_POPULATION_RECEIPT_SCHEMA,
    ATTEMPT04_POPULATION_SEED,
    ATTEMPT04_POPULATION_SEED_STRIDE,
    evaluate_attempt04_population_gates,
    load_and_validate_attempt04_population_plan,
    validate_attempt04_population_acceptance,
)


def _schedule(milestone, split, seed, roots):
    return {
        "milestone": milestone,
        "split": split,
        "seed": seed,
        "seed_stride": 1_000_003,
        "roots": roots,
    }


def _plan():
    roles = (
        ("precal_holdout", 106_071_901, 300),
        ("calibration_safety_fit", 506_071_901, 100),
        ("calibration_threshold_lock", 606_071_901, 100),
        ("locked_holdout", 806_071_901, 200),
    )
    teacher = [
        _schedule("M4.3-attempt04", role, 13_000_000_000 + base, roots)
        for role, base, roots in roles
    ]
    rng = [
        _schedule(
            "M4.3-attempt04",
            f"{role}.{kind}",
            major + base,
            roots,
        )
        for role, base, roots in roles
        for kind, major in (
            ("candidate", 20_000_000_000),
            ("evaluation", 21_000_000_000),
            ("child_policy", 22_000_000_000),
        )
    ]
    return {
        "schema": "hu_m43_population_acceptance_plan_v1",
        "milestone": "M4.3-attempt04-final-population",
        "status": "frozen_before_population_evaluation",
        "policy_attempt": "attempt04_v6",
        "attempt04_plan": {"path": "configs/attempt04.json", "file_sha256": "1" * 64},
        "runtime_contract": {
            "model_schema": HU_M43_V6_MODEL_SCHEMA,
            "model_artifact_schema": HU_M43_V6_ARTIFACT_SCHEMA,
            "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
            "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
            "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
            "safety_feature_dim": HU_M43_V6_SAFETY_FEATURE_DIM,
            "runtime_teacher_inputs": False,
            "runtime_lcb_gate": False,
            "all_four_runtime_binding_required": True,
        },
        "fixed_baseline_profile": "stage18_p1",
        "opponents": list(validator.EXPECTED_OPPONENTS),
        "paired_seat_swap": True,
        "paired_seeds_per_opponent": 1000,
        "seed": ATTEMPT04_POPULATION_SEED,
        "seed_stride": ATTEMPT04_POPULATION_SEED_STRIDE,
        "shards": 20,
        "paired_seeds_per_shard": 50,
        "candidate_records": 8000,
        "baseline_records": 8000,
        "terminal_trace_hands": 16000,
        "second_seat_override_opportunities": 4000,
        "minimum_valid_overrides": 300,
        "minimum_population_fire_rate_needed": 0.075,
        "sizing_rule": {
            "formula": "paired_seeds_required = ceil(minimum_valid_overrides / (opponents * independent_fire_rate_lower_bound))",
            "fixed_before_realized_population_labels": True,
            "optional_stopping_or_posthoc_extension_allowed": False,
            "insufficient_overrides_decision": "complete_no_go",
        },
        "freshness": {
            "exclude_all_m4_m41_m42_m43_teacher_hand_seeds": True,
            "exclude_prior_population_smoke_and_frozen_schedules": True,
            "planned_overlap_count_at_freeze": 0,
            "acceptance_validator_rechecks_teacher_seed_overlap": True,
            "excluded_population_schedules": [
                {
                    "milestone": "M4.3-attempt03-final-population",
                    "seed": 12_106_071_901,
                    "seed_stride": 1_000_003,
                    "paired_seeds": 1000,
                }
            ],
            "excluded_teacher_schedules": teacher,
            "excluded_rng_schedules": rng,
        },
        "evaluation_contract": {
            "candidate_and_baseline_same_hand_seed": True,
            "candidate_and_baseline_same_physical_seat": True,
            "candidate_and_baseline_same_opponent_policy_seed": True,
            "first_and_second_seat_metrics": True,
            "nonfire_full_trajectory_digest_cancellation": True,
            "invalid_counterfactuals_excluded": True,
            "false_positive_override_definition": "realized_delta_le_zero",
            "confidence_interval_unit": "hand_seed_cluster_after_opponent_average",
            "teacher_values_are_realized_match_ev": False,
            "threshold_reselection_allowed": False,
        },
        "fixed_acceptance_gates": dict(FIXED_GATES),
        "spot_execution": {
            "small_shards": True,
            "checkpoint_unit": "completed_shard",
            "heartbeat_required": True,
            "resume_missing_shards_only": True,
            "done_commit_last": True,
            "merge_recomputes_metrics_from_records": True,
        },
        "activation_guards": {
            "current_profile_changed": False,
            "runtime_policy_activated": False,
            "full_replacement_enabled": False,
            "threshold_changed_after_lock": False,
        },
    }


def test_attempt04_population_plan_freezes_power_runtime_and_all_rng_exclusions():
    plan = load_and_validate_attempt04_population_plan(_plan())
    assert plan["seed"] == 14_106_071_901
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["_freshness_counts"] == {
        "population": 1000,
        "prior_population": 1000,
        "teacher": 700,
        "teacher_rng": 2100,
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda plan: plan.__setitem__("seed", 13_106_071_901), "seed/power"),
        (
            lambda plan: plan["runtime_contract"].__setitem__("safety_feature_dim", 1),
            "runtime contract",
        ),
        (
            lambda plan: plan["freshness"]["excluded_rng_schedules"].pop(),
            "RNG exclusions",
        ),
        (
            lambda plan: plan["fixed_acceptance_gates"].__setitem__(
                "valid_overrides_min", 299
            ),
            "gates changed",
        ),
    ],
)
def test_attempt04_population_plan_mutations_fail_closed(mutator, message):
    plan = _plan()
    mutator(plan)
    with pytest.raises(ValueError, match=message):
        load_and_validate_attempt04_population_plan(plan)


def _acceptance_inputs(monkeypatch):
    plan = _plan()
    hashes = {
        "population_plan": "a" * 64,
        "records": "b" * 64,
        "evaluation": "c" * 64,
        "merge_manifest": "d" * 64,
        "spot_receipt": "e" * 64,
        "run_manifest": "f" * 64,
        "lifecycle_preflight": "9" * 64,
    }
    preflight = {
        "schema": ATTEMPT04_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass_population_launch_authorized",
        "model_sha256": "1" * 64,
        "final_training_manifest_file_sha256": "2" * 64,
        "threshold_lock_file_sha256": "3" * 64,
        "runtime_freeze_file_sha256": "4" * 64,
        "locked_receipt_file_sha256": "5" * 64,
        "consumption_marker_file_sha256": "6" * 64,
        "population_plan_file_sha256": hashes["population_plan"],
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "model_id": "attempt04-v6-test",
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "frozen_threshold": 0.7,
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "canonical_global_marker_verified": True,
        "locked200_go_required_and_verified": True,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    preflight["preflight_sha256"] = self_digest(preflight, "preflight_sha256")
    runtime = {
        "candidate_model_sha256": preflight["model_sha256"],
        "safety_model_sha256": preflight["model_sha256"],
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "model_id": preflight["model_id"],
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": preflight["runtime_freeze_file_sha256"],
        "training_manifest_sha256": preflight[
            "final_training_manifest_file_sha256"
        ],
        "threshold_lock_sha256": preflight["threshold_lock_file_sha256"],
        "population_plan_sha256": hashes["population_plan"],
        "sharded_evaluation": True,
        "shard_count": 20,
        "final_metrics_recomputed_from_merged_records": True,
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "safety_enabled": True,
        "safety_threshold": 0.7,
    }
    summary = {
        "schema": "hu_m4_t1_population_evaluation_v1",
        "opponents": list(validator.EXPECTED_OPPONENTS),
        "paired_seat_swap": True,
        "paired_seeds_per_opponent": 1000,
        "seed": ATTEMPT04_POPULATION_SEED,
        "seed_stride": ATTEMPT04_POPULATION_SEED_STRIDE,
        "population": {"all_seats": {"overrides": 350}},
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "runtime_config": runtime,
    }
    monkeypatch.setattr(
        validator,
        "summarize_hu_m4_population_records",
        lambda *_args, **_kwargs: deepcopy(summary),
    )
    monkeypatch.setattr(
        validator,
        "evaluate_attempt04_population_gates",
        lambda _evaluation: [
            {
                "name": "fixed_realized_gates_fixture",
                "passed": True,
                "observed": True,
                "requirement": "all fixed gates pass",
            }
        ],
    )
    merge_shards = []
    receipt_shards = []
    for shard in range(20):
        offset = shard * 50
        evaluation_sha = f"{shard + 20:064x}"
        records_sha = f"{shard + 40:064x}"
        merge_shards.append(
            {
                "offset": offset,
                "seed": ATTEMPT04_POPULATION_SEED
                + offset * ATTEMPT04_POPULATION_SEED_STRIDE,
                "paired_seeds": 50,
                "records": 400,
                "evaluation_sha256": evaluation_sha,
                "records_sha256": records_sha,
            }
        )
        receipt_shards.append(
            {
                "shard": shard,
                "done_sha256": f"{shard + 60:064x}",
                "evaluation_sha256": evaluation_sha,
                "records_sha256": records_sha,
            }
        )
    merge = {
        "schema": "hu_m4_population_shard_merge_v1",
        "status": "complete_content_verified",
        "population_plan_sha256": hashes["population_plan"],
        "merged_records_sha256": hashes["records"],
        "evaluation_sha256": hashes["evaluation"],
        "paired_seeds_per_opponent": 1000,
        "opponents": list(validator.EXPECTED_OPPONENTS),
        "merged_records": 8000,
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
        "shards": merge_shards,
    }
    run = {
        "schema": ATTEMPT04_POPULATION_MANIFEST_SCHEMA,
        "status": "spot_population_complete",
        "population_plan": {
            "sha256": hashes["population_plan"],
            "paired_seeds": 1000,
            "seed": ATTEMPT04_POPULATION_SEED,
            "seed_stride": ATTEMPT04_POPULATION_SEED_STRIDE,
            "shards": 20,
            "paired_seeds_per_shard": 50,
        },
        "shards": {"count": 20, "sha256": "7" * 64},
        "checkpoint": {
            "unit": "completed_shard",
            "retry": "deterministic_full_shard",
            "resume_missing_shards_only": True,
            "done_commit_last": True,
        },
        "compute": {
            "provisioning_model": "SPOT",
            "instance_termination_action": "DELETE",
        },
        "runtime": {
            "model_sha256": preflight["model_sha256"],
            "training_manifest_sha256": preflight[
                "final_training_manifest_file_sha256"
            ],
            "threshold_lock_sha256": preflight["threshold_lock_file_sha256"],
            "freeze_manifest_sha256": preflight["runtime_freeze_file_sha256"],
            "locked_receipt_sha256": preflight["locked_receipt_file_sha256"],
            "consumption_marker_sha256": preflight[
                "consumption_marker_file_sha256"
            ],
            "model_schema": HU_M43_V6_MODEL_SCHEMA,
            "model_id": preflight["model_id"],
            "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
            "runtime_teacher_inputs": False,
            "current_profile_used": False,
            "frozen_threshold": 0.7,
        },
        "launch_preflight_sha256": hashes["lifecycle_preflight"],
        "source_boundary": {
            "teacher_jsonl_packaged": False,
            "calibration_jsonl_packaged": False,
            "locked_jsonl_packaged": False,
            "teacher_valued_receipts_packaged": False,
            "current_profile_artifact_packaged": False,
            "runtime_artifacts_only": True,
        },
        "no_runtime_activation": True,
        "current_profile_mutated": False,
    }
    receipt = {
        "schema": ATTEMPT04_POPULATION_RECEIPT_SCHEMA,
        "status": "verified_and_merged",
        "run_manifest_sha256": hashes["run_manifest"],
        "population_plan_sha256": hashes["population_plan"],
        "model_sha256": preflight["model_sha256"],
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "evaluation_sha256": hashes["evaluation"],
        "records_sha256": hashes["records"],
        "merge_manifest_sha256": hashes["merge_manifest"],
        "valid_overrides": 350,
        "paired_seeds_per_opponent": 1000,
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "shards": receipt_shards,
        "teacher_calibration_locked_content_received": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    return {
        "evaluation": summary,
        "records": [{"runtime_binding_verified": True}],
        "population_plan": plan,
        "merge_manifest": merge,
        "spot_receipt": receipt,
        "run_manifest": run,
        "lifecycle_preflight": preflight,
        "source_hashes": hashes,
    }


def test_attempt04_population_acceptance_requires_full_v6_lifecycle(monkeypatch):
    inputs = _acceptance_inputs(monkeypatch)
    config, status = validate_attempt04_population_acceptance(**inputs)
    assert status["status"] == "complete_go"
    assert status["passed_gates"] == status["total_gates"] == 4
    assert config["policy_attempt"] == "attempt04_v6"
    changed = deepcopy(inputs)
    changed["evaluation"]["runtime_config"]["threshold_lock_sha256"] = "0" * 64
    _config, status = validate_attempt04_population_acceptance(**changed)
    assert status["status"] == "complete_no_go"
    binding = next(gate for gate in status["gates"] if gate["name"] == "v6_all_four_runtime_binding")
    assert binding["passed"] is False


def test_attempt04_gate_adapter_rejects_opponent_substitution():
    with pytest.raises(ValueError, match="opponent order"):
        evaluate_attempt04_population_gates({"opponents": ["current"]})
