from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular.evaluate_hu_m4_population import (
    M4_HAND_RECORD_SCHEMA,
    summarize_hu_m4_population_records,
)
from ofc_regular.hu_m43_joint_model_v4 import (
    HU_M43_V4_ACTION_SCORE_MODE,
    HU_M43_V4_MODEL_SCHEMA,
)
from ofc_regular.merge_hu_m4_population_shards import M4_POPULATION_MERGE_SCHEMA
from ofc_regular.validate_hu_m43_attempt02_acceptance import (
    ATTEMPT02_POPULATION_MANIFEST_SCHEMA,
    ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA,
    ATTEMPT02_POPULATION_RECEIPT_SCHEMA,
    load_and_validate_attempt02_population_plan,
    validate_attempt02_population_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_population.json"
ATTEMPT02_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt02.json"
OPPONENTS = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "random_exact_final",
)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _records(*, fired_second_seeds: int = 1000) -> list[dict[str, object]]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for opponent in OPPONENTS:
        for index in range(1000):
            seed = plan["seed"] + index * plan["seed_stride"]
            for seat in ("first", "second"):
                fired = seat == "second" and index < fired_second_seeds
                delta = 1.0 if fired else 0.0
                baseline_digest = _digest(f"baseline:{opponent}:{seed}:{seat}")
                candidate_digest = (
                    _digest(f"candidate:{opponent}:{seed}:{seat}")
                    if fired
                    else baseline_digest
                )
                records.append(
                    {
                        "schema": M4_HAND_RECORD_SCHEMA,
                        "opponent": opponent,
                        "seed": seed,
                        "seat": seat,
                        "hero_policy_seed": seed * 4 + (1 if seat == "second" else 0),
                        "opponent_policy_seed": seed * 4 + (0 if seat == "second" else 1),
                        "candidate_score": delta,
                        "baseline_score": 0.0,
                        "delta": delta,
                        "override_log_valid": True,
                        "override_log_reason": None,
                        "override_fired": fired,
                        "t1_decision_count": 1,
                        "nonfire_reason": None if fired else "safety_gate",
                        "safety_probability": 0.9 if fired else 0.1,
                        "predicted_delta": 1.0,
                        "runtime_binding_verified": True,
                        "candidate_gameplay_digest": candidate_digest,
                        "baseline_gameplay_digest": baseline_digest,
                        "nonfire_cancellation_valid": None if fired else True,
                        "counterfactual_basis": (
                            "same_seed_physical_seat_opponent_policy_seeds_v1"
                        ),
                    }
                )
    return records


def _inputs(*, fired_second_seeds: int = 1000) -> dict[str, object]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    records = _records(fired_second_seeds=fired_second_seeds)
    evaluation = summarize_hu_m4_population_records(
        records,
        opponents=OPPONENTS,
        paired_seeds=1000,
        seed=plan["seed"],
        seed_stride=plan["seed_stride"],
    )
    lifecycle = {
        "schema": ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "model_sha256": "1" * 64,
        "training_manifest_sha256": "2" * 64,
        "data_contract_file_sha256": "3" * 64,
        "freeze_manifest_file_sha256": "4" * 64,
        "locked_receipt_file_sha256": "5" * 64,
        "consumption_marker_file_sha256": "6" * 64,
        "population_plan_file_sha256": "a" * 64,
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "model_id": "attempt02-v4-test",
        "action_score_mode": HU_M43_V4_ACTION_SCORE_MODE,
        "frozen_threshold": 0.5,
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "canonical_global_marker_verified": True,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    evaluation["runtime_config"] = {
        "candidate_model_sha256": lifecycle["model_sha256"],
        "safety_model_sha256": lifecycle["model_sha256"],
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V4_ACTION_SCORE_MODE,
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": lifecycle["freeze_manifest_file_sha256"],
        "training_manifest_sha256": lifecycle["training_manifest_sha256"],
        "population_plan_sha256": "a" * 64,
        "sharded_evaluation": True,
        "shard_count": 20,
        "final_metrics_recomputed_from_merged_records": True,
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "safety_enabled": True,
        "safety_threshold": lifecycle["frozen_threshold"],
        "model_id": lifecycle["model_id"],
    }
    source_hashes = {
        "population_plan": "a" * 64,
        "records": "b" * 64,
        "evaluation": "c" * 64,
        "merge_manifest": "d" * 64,
        "run_manifest": "e" * 64,
    }
    merge_shards = []
    receipt_shards = []
    for shard in range(20):
        evaluation_shard_sha = _digest(f"evaluation-shard:{shard}")
        records_shard_sha = _digest(f"records-shard:{shard}")
        merge_shards.append(
            {
                "offset": shard * 50,
                "seed": plan["seed"] + shard * 50 * plan["seed_stride"],
                "paired_seeds": 50,
                "evaluation_sha256": evaluation_shard_sha,
                "records_sha256": records_shard_sha,
                "records": 400,
            }
        )
        receipt_shards.append(
            {
                "shard": shard,
                "done_sha256": _digest(f"done-shard:{shard}"),
                "evaluation_sha256": evaluation_shard_sha,
                "records_sha256": records_shard_sha,
            }
        )
    merge = {
        "schema": M4_POPULATION_MERGE_SCHEMA,
        "status": "complete_content_verified",
        "population_plan_sha256": source_hashes["population_plan"],
        "merged_records_sha256": source_hashes["records"],
        "evaluation_sha256": source_hashes["evaluation"],
        "paired_seeds_per_opponent": 1000,
        "opponents": list(OPPONENTS),
        "merged_records": 8000,
        "current_profile_used": False,
        "shards": merge_shards,
        "metrics_recomputed_from_merged_seed_clusters": True,
    }
    run_manifest = {
        "schema": ATTEMPT02_POPULATION_MANIFEST_SCHEMA,
        "population_plan": {
            "sha256": source_hashes["population_plan"],
            "paired_seeds": 1000,
            "seed": plan["seed"],
            "seed_stride": plan["seed_stride"],
            "shards": 20,
            "paired_seeds_per_shard": 50,
        },
        "shards": {"count": 20, "sha256": _digest("shard-spec")},
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
            "model_sha256": lifecycle["model_sha256"],
            "model_schema": HU_M43_V4_MODEL_SCHEMA,
            "model_id": lifecycle["model_id"],
            "action_score_mode": HU_M43_V4_ACTION_SCORE_MODE,
            "training_manifest_sha256": lifecycle["training_manifest_sha256"],
            "data_contract_sha256": lifecycle["data_contract_file_sha256"],
            "freeze_manifest_sha256": lifecycle["freeze_manifest_file_sha256"],
            "locked_receipt_sha256": lifecycle["locked_receipt_file_sha256"],
            "consumption_marker_sha256": lifecycle[
                "consumption_marker_file_sha256"
            ],
            "runtime_teacher_inputs": False,
            "current_profile_used": False,
            "frozen_threshold": lifecycle["frozen_threshold"],
        },
        "launch_preflight": dict(lifecycle),
        "source_boundary": {
            "teacher_jsonl_packaged": False,
            "calibration_jsonl_packaged": False,
            "locked_jsonl_packaged": False,
            "current_profile_artifact_packaged": False,
            "lifecycle_hashes_and_receipts_only": True,
        },
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    spot_receipt = {
        "schema": ATTEMPT02_POPULATION_RECEIPT_SCHEMA,
        "status": "verified_and_merged",
        "run_manifest_sha256": source_hashes["run_manifest"],
        "population_plan_sha256": source_hashes["population_plan"],
        "model_sha256": lifecycle["model_sha256"],
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V4_ACTION_SCORE_MODE,
        "evaluation_sha256": source_hashes["evaluation"],
        "records_sha256": source_hashes["records"],
        "merge_manifest_sha256": source_hashes["merge_manifest"],
        "valid_overrides": evaluation["population"]["all_seats"]["overrides"],
        "paired_seeds_per_opponent": 1000,
        "invalid_counterfactuals": evaluation["invalid_counterfactuals"],
        "nonfire_cancellation_mismatches": evaluation[
            "nonfire_cancellation_mismatches"
        ],
        "shards": receipt_shards,
        "teacher_calibration_locked_content_received": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    return {
        "evaluation": evaluation,
        "records": records,
        "population_plan": plan,
        "merge_manifest": merge,
        "spot_receipt": spot_receipt,
        "run_manifest": run_manifest,
        "lifecycle_preflight": lifecycle,
        "source_hashes": source_hashes,
    }


def _validate(inputs: dict[str, object]):
    return validate_attempt02_population_acceptance(**inputs)  # type: ignore[arg-type]


def test_frozen_attempt02_population_plan_has_exact_power_and_gates() -> None:
    plan = load_and_validate_attempt02_population_plan(PLAN)
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["minimum_valid_overrides"] == 300
    assert plan["runtime_contract"]["model_schema"] == HU_M43_V4_MODEL_SCHEMA
    assert plan["attempt02_training_plan"]["file_sha256"] == hashlib.sha256(
        ATTEMPT02_PLAN.read_bytes()
    ).hexdigest()
    assert plan["_freshness_counts"] == {
        "population": 1000,
        "teacher_schedule": 755,
        "prior_population": 6,
    }


def test_fresh_population_seed_overlap_is_rejected_before_launch() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["freshness"]["excluded_teacher_schedules"][3]["seed"] = plan["seed"]
    with pytest.raises(ValueError, match="population/teacher seed overlap"):
        load_and_validate_attempt02_population_plan(plan)


def test_all_m4_through_attempt02_teacher_schedules_are_required() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["freshness"]["excluded_teacher_schedules"].pop(0)
    with pytest.raises(ValueError, match="teacher seed exclusions changed"):
        load_and_validate_attempt02_population_plan(plan)

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["freshness"]["excluded_teacher_schedules"].append(
        copy.deepcopy(plan["freshness"]["excluded_teacher_schedules"][0])
    )
    with pytest.raises(ValueError, match="duplicate teacher seed schedule"):
        load_and_validate_attempt02_population_plan(plan)


def test_complete_realized_population_is_go() -> None:
    config, status = _validate(_inputs())
    assert config["policy_attempt"] == "attempt02_v4"
    assert config["threshold_reselection_allowed"] is False
    assert status["decision"] == "complete_go"
    assert status["valid_overrides"] == 4000
    assert status["gates_passed"] == status["gates_total"]


def test_underpowered_population_is_completed_no_go() -> None:
    _config, status = _validate(_inputs(fired_second_seeds=74))
    assert status["decision"] == "complete_no_go"
    minimum = next(
        gate for gate in status["gates"] if gate["name"] == "minimum_valid_overrides"
    )
    assert minimum["observed"] == 296
    assert minimum["passed"] is False


def test_v4_runtime_or_hash_tamper_is_no_go() -> None:
    inputs = _inputs()
    inputs["evaluation"]["runtime_config"]["action_score_mode"] = "v3"  # type: ignore[index]
    _config, status = _validate(inputs)
    runtime = next(
        gate for gate in status["gates"] if gate["name"] == "v4_runtime_binding"
    )
    assert runtime["passed"] is False
    assert status["decision"] == "complete_no_go"

    inputs = _inputs()
    inputs["spot_receipt"]["records_sha256"] = "f" * 64  # type: ignore[index]
    _config, status = _validate(inputs)
    lifecycle = next(
        gate for gate in status["gates"] if gate["name"] == "lifecycle_hash_chain"
    )
    assert lifecycle["passed"] is False


def test_shard_provenance_or_done_last_contract_tamper_is_no_go() -> None:
    inputs = _inputs()
    inputs["spot_receipt"]["shards"][7]["records_sha256"] = "f" * 64  # type: ignore[index]
    _config, status = _validate(inputs)
    lifecycle = next(
        gate for gate in status["gates"] if gate["name"] == "lifecycle_hash_chain"
    )
    assert lifecycle["passed"] is False

    inputs = _inputs()
    inputs["run_manifest"]["checkpoint"]["done_commit_last"] = False  # type: ignore[index]
    _config, status = _validate(inputs)
    lifecycle = next(
        gate for gate in status["gates"] if gate["name"] == "lifecycle_hash_chain"
    )
    assert lifecycle["passed"] is False


def test_forged_nonfire_record_is_rejected_by_content_recomputation() -> None:
    inputs = _inputs()
    row = inputs["records"][0]  # type: ignore[index]
    row["candidate_gameplay_digest"] = _digest("forged")
    with pytest.raises(ValueError, match="nonfire cancellation claim"):
        _validate(inputs)
