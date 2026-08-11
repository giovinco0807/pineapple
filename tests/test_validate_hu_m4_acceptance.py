from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular.evaluate_hu_m4_population import (
    M4_HAND_RECORD_SCHEMA,
    summarize_hu_m4_population_records,
)

from ofc_regular.validate_hu_m4_acceptance import (
    M4_ACCEPTANCE_CONFIG_SCHEMA,
    M4_ACCEPTANCE_STATUS_SCHEMA,
    REQUIRED_OPPONENTS,
    main,
    validate_hu_m4_acceptance,
)
from ofc_regular.hu_m43_pilot_contract import (
    M43_DATA_CONTRACT_SCHEMA,
    M43_FREEZE_SCHEMA,
    M43_LOCKED_RECEIPT_SCHEMA,
    canonical_manifest_sha256,
)


_HASH = "a" * 64


def _ci(mean: float, low: float, high: float, n: int = 1000) -> dict[str, float | int]:
    return {
        "n": n,
        "mean": mean,
        "std_error": (high - low) / 3.92,
        "ci95_low": low,
        "ci95_high": high,
    }


def _seat_summary(*, first: bool = False) -> dict[str, object]:
    if first:
        return {
            "hands": 4000,
            "delta_ev_per_hand": _ci(0.0, 0.0, 0.0, 4000),
            "overrides": 0,
            "realized_gain_per_override": _ci(0.0, 0.0, 0.0, 0),
            "false_positive_override_rate": None,
            "override_loss_tail": {"n": 0, "p95": None, "p99": None, "max": None},
            "nonfire_cancellation_mismatches": 0,
            "nonfire_nonzero_deltas": 0,
            "nonfire_cancellation_unknown": 0,
            "invalid_counterfactuals": 0,
        }
    return {
        "hands": 4000,
        "delta_ev_per_hand": _ci(0.04, 0.01, 0.07, 4000),
        "overrides": 350,
        "realized_gain_per_override": _ci(0.46, 0.12, 0.80, 350),
        "false_positive_override_rate": 0.25,
        "override_loss_tail": {"n": 350, "p95": 20.0, "p99": 35.0, "max": 48.0},
        "nonfire_cancellation_mismatches": 0,
        "nonfire_nonzero_deltas": 0,
        "nonfire_cancellation_unknown": 0,
        "invalid_counterfactuals": 0,
    }


def _evaluation() -> dict[str, object]:
    means = {
        "stage19_p0": (0.02, -0.01),
        "stage9f_p2": (0.03, 0.00),
        "stage7_m5_r10": (0.00, -0.015),
        "random_exact_final": (0.05, 0.02),
    }
    by_opponent = {
        name: {
            "paired_seeds": 2,
            "seat_swap": {"delta_ev_per_hand": _ci(mean, low, mean + 0.03)}
        }
        for name, (mean, low) in means.items()
    }
    return {
        "schema": "hu_m4_t1_population_evaluation_v1",
        "seed": 10_000,
        "seed_stride": 101,
        "paired_seeds_per_opponent": 2,
        "opponents": list(REQUIRED_OPPONENTS),
        "by_opponent": by_opponent,
        "population": {
            "all_seats": _seat_summary(),
            "by_seat": {
                "first": _seat_summary(first=True),
                "second": _seat_summary(),
            },
            "paired_seat_swap": {
                "delta_ev_per_hand": _ci(0.02, 0.005, 0.035, 4000),
                "ci_independence_unit": "hand_seed_cluster_after_opponent_average",
            },
        },
        "worst_case_opponent": {
            "opponent": "stage7_m5_r10",
            "delta_ev_per_hand": _ci(0.00, -0.015, 0.03),
        },
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "promotion_eligible_counterfactual_contract": True,
        "primary_metrics_exclude_invalid_records": True,
        "runtime_config": {
            "safety_threshold": 0.8,
            "candidate_model_sha256": _HASH,
            "safety_model_sha256": _HASH,
            "current_profile_used": False,
            "promotion_artifact_contract": True,
            "diagnostic_legacy": False,
        },
        # These deliberately bad diagnostics must not affect promotion.
        "teacher_ev": -999999.0,
        "top1_accuracy": 0.0,
    }


def _audit() -> dict[str, object]:
    return {
        "schema": "hu_m4_t1_second_data_audit_v1",
        "status": "pass",
        "gates": {
            "hidden_discard_safe": True,
            "all_legal_actions_mapped": True,
            "candidate_evaluation_rng_disjoint": True,
            "within_split_shard_paths_unique": True,
            "within_split_hand_seeds_unique": True,
            "within_split_fingerprints_unique": True,
            "seed_ranges_disjoint": True,
            "fingerprints_disjoint": True,
            "holdout_threshold_search_allowed": False,
            "current_profile_resolved": False,
            "paired_common_future_delta_contract": True,
        },
        "splits": {
            "train": {"shards": [{"hand_seeds": [1, 2]}]},
            "calibration": {"shards": [{"hand_seeds": [3, 4]}]},
            "locked_holdout": {"shards": [{"hand_seeds": [5, 6]}]},
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "schema": "hu_m4_t1_joint_training_manifest_v1",
        "teacher_value_runtime_gate": False,
        "threshold_adaptation_after_calibration": False,
        "locked_holdout_used_for_threshold_or_training": False,
        "calibration": {
            "selected_threshold": 0.8,
            "threshold_source": "calibration_file_only",
        },
        "locked_holdout": {"frozen_safety_threshold": 0.8},
        "runtime_lock": {
            "candidate_model_sha256": _HASH,
            "safety_model_sha256": _HASH,
        },
    }


def _m41_manifest() -> dict[str, object]:
    manifest = _manifest()
    manifest["schema"] = "hu_m4_t1_joint_training_manifest_v2"
    manifest["calibration_partition"] = {
        "schema": "hu_m4_safety_calibration_partition_v1",
        "status": "pass",
        "overlap": {
            "seed_value_count": 0,
            "observation_fingerprint_count": 0,
            "row_hash_count": 0,
        },
        "locked_holdout_used": False,
    }
    manifest["calibration"] = {
        "selected_threshold": 0.8,
        "threshold_source": "threshold_lock_subset_only",
        "safety_estimator_source": "safety_fit_subset_only",
        "locked_holdout_used": False,
        "fit_lock_overlap": {
            "seed_value_count": 0,
            "observation_fingerprint_count": 0,
            "row_hash_count": 0,
        },
        "safety_fit": {
            "used_for_estimator_fit": True,
            "used_for_threshold_sweep": False,
        },
        "threshold_lock": {
            "used_for_estimator_fit": False,
            "used_for_threshold_sweep": True,
        },
    }
    return manifest


def _m42_manifest() -> dict[str, object]:
    manifest = _m41_manifest()
    calibration = manifest["calibration"]  # type: ignore[index]
    calibration["safety_estimator_source"] = (  # type: ignore[index]
        "train_oof_plus_safety_fit_subset"
    )
    calibration["safety_estimator_training_sources"] = {  # type: ignore[index]
        "train_oof": {
            "used_for_estimator_fit": True,
            "used_for_threshold_sweep": False,
        },
        "calibration_safety_fit": {
            "used_for_estimator_fit": True,
            "used_for_threshold_sweep": False,
        },
        "calibration_threshold_lock": {
            "used_for_estimator_fit": False,
            "used_for_threshold_sweep": True,
        },
        "locked_holdout": {
            "used_for_estimator_fit": False,
            "used_for_threshold_sweep": False,
        },
    }
    calibration["oof_train_calibration_overlap"] = {  # type: ignore[index]
        "seed_value_count": 0,
        "observation_fingerprint_count": 0,
        "row_hash_count": 0,
    }
    manifest["cross_fit"] = {
        "schema": "hu_m4_identity_group_nested_cross_fit_v2",
        "status": "pass",
        "folds": 5,
        "action_score_mode": "negative_regret_ranker_v2",
        "fold_assignment": {"identity_leakage_count": 0},
        "oof_coverage": {
            "base_head_prediction_counts": [1],
            "meta_rank_prediction_counts": [1],
            "uncertainty_prediction_counts": [1],
            "each_sample_predicted_exactly_once": True,
        },
        "predictor_lineage": {
            "schema": "hu_m4_outer_fold_predictor_lineage_v1",
            "nested_cross_fit": True,
            "identity_leakage_count": 0,
            "all_outer_validation_identities_excluded": True,
        },
        "safety_examples": {
            "source": "train_oof_only",
            "used_for_threshold_lock": False,
            "used_for_locked_holdout": False,
        },
        "split_role_overlap": {
            "train_oof__calibration": {
                "seed_value_count": 0,
                "observation_fingerprint_count": 0,
                "row_hash_count": 0,
            },
            "train_oof__locked_holdout": {
                "seed_value_count": 0,
                "observation_fingerprint_count": 0,
                "row_hash_count": 0,
            },
        },
        "locked_holdout_used": False,
        "threshold_lock_used": False,
    }
    return manifest


def _canonical_sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _m43_inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
    audit = _audit()
    plan_sha = "1" * 64
    prior_sha = "2" * 64
    teacher_sha = "3" * 64
    locked_identity_sha = "4" * 64
    locked_shards_sha = "5" * 64
    unsigned_contract: dict[str, object] = {
        "schema": M43_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_data_sealed_for_model_freeze",
        "plan_sha256": plan_sha,
        "base_audit": audit,
        "base_audit_sha256": _canonical_sha(audit),
        "prior_exclusions": {"audit_sha256": prior_sha},
        "teacher_shards_all_splits_sha256": teacher_sha,
        "splits": {
            "locked_holdout": {"identity_sha256": locked_identity_sha}
        },
        "teacher_shards": {
            "splits": {
                "locked_holdout": {
                    "ordered_shards_sha256": locked_shards_sha
                }
            }
        },
    }
    data_contract = dict(unsigned_contract)
    data_contract["contract_sha256"] = _canonical_sha(unsigned_contract)

    manifest = _m42_manifest()
    manifest["action_score_formula"] = {
        "mode": "baseline_paired_delta_risk_ensemble_v3",
        "baseline_action_score_exact_zero": True,
        "runtime_model": "stored_crossfit_fold_ensemble",
        "teacher_value_runtime_input": False,
        "teacher_lcb_runtime_gate": False,
    }
    manifest["training_config"] = {
        "action_score_mode": "baseline_paired_delta_risk_ensemble_v3"
    }
    manifest["calibration_partition"] = {
        "schema": "hu_m43_trainer_data_contract_binding_v1",
        "status": "pass",
        "overlap": 0,
        "legacy_hash_partition_used": False,
        "locked_holdout_labels_used_for_role_assignment": False,
        "locked_holdout_labels_opened_by_trainer": False,
    }
    manifest["calibration"].update(  # type: ignore[union-attr]
        {
            "status": "go",
            "selected_metrics": {"fires": 10},
            "constraints": {"minimum_fires": 10},
            "threshold_selection_source": "calibration.threshold_lock",
            "safety_calibrator_sources": [
                "train_oof",
                "calibration.safety_fit",
            ],
            "runtime_inputs_exclude_teacher_values_and_teacher_lcb": True,
        }
    )
    manifest["cross_fit"] = {
        "schema": "hu_m4_identity_group_paired_delta_cross_fit_v3",
        "status": "pass",
        "folds": 5,
        "action_score_mode": "baseline_paired_delta_risk_ensemble_v3",
        "fold_assignment": {"identity_leakage_count": 0},
        "oof_coverage": {
            "paired_head_prediction_counts": [1],
            "each_sample_predicted_exactly_once": True,
        },
        "predictor_lineage": {
            "schema": "hu_m4_paired_outer_fold_predictor_lineage_v1",
            "identity_leakage_count": 0,
            "all_outer_validation_identities_excluded": True,
            "fold_audits": [
                {
                    "fold": fold,
                    "training_samples": 80,
                    "validation_samples": 20,
                    "train__validation_identity_overlap": {
                        "seed_value_count": 0,
                        "observation_fingerprint_count": 0,
                        "row_hash_count": 0,
                    },
                    "validation_identity_excluded_from_all_paired_heads": True,
                    "identity_leakage_count": 0,
                    "oof_safety_inner_ensemble": {
                        "fold_count": 5,
                        "aggregation": "mean_plus_cross_fold_disagreement",
                        "matches_runtime_fold_count": True,
                        "fold_assignment": {
                            "folds": 5,
                            "identity_leakage_count": 0,
                        },
                    },
                }
                for fold in range(5)
            ],
        },
        "runtime_ensemble": {
            "source": "stored_crossfit_fold_estimators",
            "fold_count": 5,
            "full_refit_used_at_runtime": False,
            "oof_to_full_refit_distribution_shift": False,
            "oof_safety_aggregation_semantics_match": True,
            "oof_safety_fold_count": 5,
            "baseline_action_score_exact_zero": True,
        },
        "safety_examples": {
            "source": "train_oof_only",
            "used_for_threshold_lock": False,
            "used_for_locked_holdout": False,
        },
        "split_role_overlap": {
            "train_oof__calibration": {
                "seed_value_count": 0,
                "observation_fingerprint_count": 0,
                "row_hash_count": 0,
            },
            "train_oof__locked_holdout": {
                "status": "sealed_contract_only_not_opened_by_trainer",
                "identity_overlap_not_computed_from_locked_labels": True,
            },
        },
        "locked_holdout_used": False,
        "threshold_lock_used": False,
    }
    manifest["locked_holdout"] = {"status": "not_evaluated_pre_freeze"}
    manifest["promotion_status"] = "candidate_for_realized_ev_evaluation"
    manifest["inputs"] = {"train": [], "calibration": []}
    manifest["m43_data_contract"] = {
        "contract_sha256": data_contract["contract_sha256"],
        "plan_sha256": plan_sha,
        "prior_freshness_audit_sha256": prior_sha,
        "base_audit_sha256": data_contract["base_audit_sha256"],
        "teacher_shards_all_splits_sha256": teacher_sha,
    }

    training_manifest_file_sha = "6" * 64
    marker_file_sha = "7" * 64
    freeze: dict[str, object] = {
        "schema": M43_FREEZE_SCHEMA,
        "status": "model_and_threshold_frozen_locked_unopened",
        "plan_sha256": plan_sha,
        "data_contract_sha256": data_contract["contract_sha256"],
        "data_contract_resolved_path": "C:/m43/data_contract.json",
        "locked_consumption_marker_resolved_path": (
            "C:/m43/M43_LOCKED_CONSUMED.json"
        ),
        "prior_freshness_audit_sha256": prior_sha,
        "base_audit_sha256": data_contract["base_audit_sha256"],
        "teacher_shards_all_splits_sha256": teacher_sha,
        "training_manifest_sha256": training_manifest_file_sha,
        "model_sha256": _HASH,
        "model_id": "m43-test",
        "frozen_threshold": 0.8,
        "safety_enabled": True,
        "calibration_status": "go",
        "ranker_training_sources": ["train"],
        "ranker_crossfit_source": "train",
        "safety_calibrator_sources": [
            "train_oof",
            "calibration.safety_fit",
        ],
        "threshold_selection_source": "calibration.threshold_lock",
        "model_selection_sources": ["train"],
        "locked_holdout_used_for_training_selection_or_threshold": False,
        "model_or_threshold_locked_label_access_count_at_freeze": 0,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "pilot_can_promote_policy": False,
        "fresh_population_acceptance_required": True,
        "minimum_population_valid_overrides": 300,
    }
    freeze_sha = canonical_manifest_sha256(freeze)
    receipt: dict[str, object] = {
        "schema": M43_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "freeze_manifest_sha256": freeze_sha,
        "data_contract_sha256": data_contract["contract_sha256"],
        "model_sha256": _HASH,
        "frozen_threshold": 0.8,
        "locked_identity_sha256": locked_identity_sha,
        "locked_teacher_shards_sha256": locked_shards_sha,
        "consumption_marker_sha256": marker_file_sha,
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": 300,
        "teacher_value_status": "diagnostic_not_realized_match_ev",
    }
    marker: dict[str, object] = {
        "schema": "hu_m43_locked_holdout_consumption_marker_v1",
        "status": "claimed_before_locked_content_read",
        "freeze_manifest_sha256": freeze_sha,
        "data_contract_sha256": data_contract["contract_sha256"],
        "model_sha256": _HASH,
        "locked_identity_sha256": locked_identity_sha,
        "locked_teacher_shards_sha256": locked_shards_sha,
        "evaluation_pass_count": 1,
    }
    population_plan: dict[str, object] = {
        "schema": "hu_m43_population_acceptance_plan_v1",
        "status": "frozen_before_population_evaluation",
        "opponents": list(REQUIRED_OPPONENTS),
        "paired_seat_swap": True,
        "paired_seeds_per_opponent": 75,
        "seed": 10_000,
        "seed_stride": 101,
        "shards": 3,
        "paired_seeds_per_shard": 25,
        "minimum_valid_overrides": 300,
        "sizing_rule": {
            "optional_stopping_or_posthoc_extension_allowed": False
        },
        "freshness": {
            "excluded_population_schedules": [
                {"seed": 20_000, "seed_stride": 101, "paired_seeds": 2}
            ]
        },
        "activation_guards": {
            "current_profile_changed": False,
            "runtime_policy_activated": False,
            "full_replacement_enabled": False,
            "threshold_changed_after_lock": False,
        },
    }
    source_hashes = {
        "evaluation": "8" * 64,
        "data_audit": "9" * 64,
        "training_manifest": training_manifest_file_sha,
        "m43_data_contract": "b" * 64,
        "m43_freeze_manifest": "c" * 64,
        "m43_locked_receipt": "d" * 64,
        "m43_consumption_marker": marker_file_sha,
        "m43_population_plan": "e" * 64,
        "m43_population_records": "a" * 64,
        "m43_population_merge_manifest": "b" * 64,
        "m43_population_spot_receipt": "c" * 64,
        "m43_population_run_manifest": "d" * 64,
    }
    return (
        manifest,
        data_contract,
        freeze,
        receipt,
        marker,
        population_plan,
        source_hashes,
    )


def _m43_population_records(
    plan: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed = int(plan["seed"])
    stride = int(plan["seed_stride"])
    count = int(plan["paired_seeds_per_opponent"])
    for opponent in REQUIRED_OPPONENTS:
        for index in range(count):
            hand_seed = seed + index * stride
            for seat in ("first", "second"):
                fired = seat == "second"
                delta = 1.0 if fired else 0.0
                rows.append(
                    {
                        "schema": M4_HAND_RECORD_SCHEMA,
                        "opponent": opponent,
                        "seed": hand_seed,
                        "seat": seat,
                        "hero_policy_seed": hand_seed * 4 + int(fired),
                        "opponent_policy_seed": hand_seed * 4 + int(not fired),
                        "candidate_score": delta,
                        "baseline_score": 0.0,
                        "delta": delta,
                        "override_log_valid": True,
                        "override_fired": fired,
                        "candidate_gameplay_digest": (
                            ("b" if fired else "a") * 64
                        ),
                        "baseline_gameplay_digest": "a" * 64,
                        "nonfire_cancellation_valid": None if fired else True,
                        "counterfactual_basis": (
                            "same_seed_physical_seat_opponent_policy_seeds_v1"
                        ),
                    }
                )
    return rows


def _m43_complete_inputs() -> dict[str, object]:
    manifest, contract, freeze, receipt, marker, plan, source_hashes = _m43_inputs()
    records = _m43_population_records(plan)
    evaluation = summarize_hu_m4_population_records(
        records,
        opponents=REQUIRED_OPPONENTS,
        paired_seeds=int(plan["paired_seeds_per_opponent"]),
        seed=int(plan["seed"]),
        seed_stride=int(plan["seed_stride"]),
    )
    evaluation["runtime_config"] = {
        "safety_threshold": 0.8,
        "candidate_model_sha256": _HASH,
        "safety_model_sha256": _HASH,
        "action_score_mode": "baseline_paired_delta_risk_ensemble_v3",
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "population_plan_sha256": source_hashes["m43_population_plan"],
    }
    evaluation["shard_merge"] = {
        "schema": "hu_m4_population_shard_merge_v1",
        "population_plan_sha256": source_hashes["m43_population_plan"],
        "shards": int(plan["shards"]),
        "metrics_recomputed_from_merged_seed_clusters": True,
    }
    merge_manifest = {
        "schema": "hu_m4_population_shard_merge_v1",
        "status": "complete_content_verified",
        "population_plan_sha256": source_hashes["m43_population_plan"],
        "seed": plan["seed"],
        "seed_stride": plan["seed_stride"],
        "paired_seeds_per_opponent": plan["paired_seeds_per_opponent"],
        "opponents": list(REQUIRED_OPPONENTS),
        "shards": [
            {
                "offset": index * 25,
                "seed": int(plan["seed"])
                + index * 25 * int(plan["seed_stride"]),
                "paired_seeds": 25,
                "records": 25 * 2 * len(REQUIRED_OPPONENTS),
                "evaluation_sha256": f"{index + 1:x}" * 64,
                "records_sha256": f"{index + 4:x}" * 64,
            }
            for index in range(3)
        ],
        "merged_records": len(records),
        "merged_records_sha256": source_hashes["m43_population_records"],
        "evaluation_sha256": source_hashes["evaluation"],
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
    }
    run_manifest = {
        "schema": "hu_m4_population_spot_manifest_v1",
        "run_name": "m43-test",
        "population_plan": {
            "sha256": source_hashes["m43_population_plan"],
            "paired_seeds": plan["paired_seeds_per_opponent"],
            "seed": plan["seed"],
            "seed_stride": plan["seed_stride"],
            "shards": plan["shards"],
            "paired_seeds_per_shard": plan["paired_seeds_per_shard"],
        },
        "source": {"sha256": "1" * 64, "bytes": 1234},
        "startup": {"sha256": "2" * 64},
        "shards": {"count": plan["shards"], "sha256": "3" * 64},
        "compute": {"provisioning_model": "SPOT"},
        "checkpoint": {
            "unit": "completed_shard",
            "retry": "deterministic_full_shard",
            "resume_missing_shards_only": True,
            "done_commit_last": True,
        },
        "runtime": {
            "model_sha256": _HASH,
            "frozen_threshold": 0.8,
            "training_manifest_sha256": source_hashes["training_manifest"],
            "data_contract_sha256": contract["contract_sha256"],
            "freeze_manifest_sha256": source_hashes["m43_freeze_manifest"],
            "locked_receipt_sha256": source_hashes["m43_locked_receipt"],
            "consumption_marker_sha256": source_hashes[
                "m43_consumption_marker"
            ],
            "current_profile_used": False,
        },
        "launch_preflight": {
            "schema": "hu_m43_population_launch_preflight_v1",
            "status": "pass",
            "teacher_overlap_count": 0,
            "prior_population_overlap_count": 0,
        },
        "no_runtime_activation": True,
        "current_profile_mutated": False,
    }
    overrides = evaluation["population"]["all_seats"]["overrides"]
    spot_receipt = {
        "schema": "hu_m4_population_spot_receipt_v1",
        "status": "verified_and_merged",
        "run_name": "m43-test",
        "run_manifest_sha256": source_hashes["m43_population_run_manifest"],
        "population_plan_sha256": source_hashes["m43_population_plan"],
        "model_sha256": _HASH,
        "evaluation_sha256": source_hashes["evaluation"],
        "records_sha256": source_hashes["m43_population_records"],
        "merge_manifest_sha256": source_hashes[
            "m43_population_merge_manifest"
        ],
        "paired_seeds_per_opponent": plan["paired_seeds_per_opponent"],
        "valid_overrides": overrides,
        "invalid_counterfactuals": evaluation["invalid_counterfactuals"],
        "nonfire_cancellation_mismatches": evaluation[
            "nonfire_cancellation_mismatches"
        ],
        "shards": [
            {
                "shard": index,
                "done_sha256": f"{index + 7:x}" * 64,
                "evaluation_sha256": f"{index + 1:x}" * 64,
                "records_sha256": f"{index + 4:x}" * 64,
            }
            for index in range(int(plan["shards"]))
        ],
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    return {
        "manifest": manifest,
        "contract": contract,
        "freeze": freeze,
        "receipt": receipt,
        "marker": marker,
        "plan": plan,
        "evaluation": evaluation,
        "records": records,
        "merge_manifest": merge_manifest,
        "spot_receipt": spot_receipt,
        "run_manifest": run_manifest,
        "source_hashes": source_hashes,
    }


def _validate_m43(inputs: dict[str, object]):
    return validate_hu_m4_acceptance(
        inputs["evaluation"],  # type: ignore[arg-type]
        _audit(),
        inputs["manifest"],  # type: ignore[arg-type]
        m43_data_contract=inputs["contract"],  # type: ignore[arg-type]
        m43_freeze_manifest=inputs["freeze"],  # type: ignore[arg-type]
        m43_locked_receipt=inputs["receipt"],  # type: ignore[arg-type]
        m43_consumption_marker=inputs["marker"],  # type: ignore[arg-type]
        m43_population_plan=inputs["plan"],  # type: ignore[arg-type]
        m43_population_records=inputs["records"],  # type: ignore[arg-type]
        m43_population_merge_manifest=inputs["merge_manifest"],  # type: ignore[arg-type]
        m43_population_spot_receipt=inputs["spot_receipt"],  # type: ignore[arg-type]
        m43_population_run_manifest=inputs["run_manifest"],  # type: ignore[arg-type]
        source_hashes=inputs["source_hashes"],  # type: ignore[arg-type]
    )


def _gate(status: dict[str, object], name: str) -> dict[str, object]:
    return next(row for row in status["gates"] if row["name"] == name)  # type: ignore[index]


def test_complete_go_uses_only_realized_population_gates() -> None:
    config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), _manifest(), source_hashes={"evaluation": "b" * 64}
    )

    assert config["schema"] == M4_ACCEPTANCE_CONFIG_SCHEMA
    assert config["teacher_metrics_for_promotion"] is False
    assert status["schema"] == M4_ACCEPTANCE_STATUS_SCHEMA
    assert status["status"] == "complete_go"
    assert status["promotion_decision"] == "go"
    assert status["failed_gates"] == []
    assert status["teacher_metrics_used_for_promotion"] is False
    assert status["top1_accuracy_used_for_promotion"] is False


def test_m41_disjoint_safety_fit_and_threshold_lock_is_accepted() -> None:
    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), _m41_manifest()
    )

    assert status["status"] == "complete_go"
    provenance = _gate(status, "safety_calibration_provenance")
    assert provenance["passed"] is True
    assert provenance["observed"]["mode"] == "disjoint_safety_fit_threshold_lock"


def test_m42_leak_free_train_oof_safety_augmentation_is_accepted() -> None:
    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), _m42_manifest()
    )

    assert status["status"] == "complete_go"
    provenance = _gate(status, "safety_calibration_provenance")
    assert provenance["passed"] is True
    assert provenance["observed"]["safety_estimator_source"] == (
        "train_oof_plus_safety_fit_subset"
    )
    assert provenance["observed"]["train_oof_provenance"][
        "predictor_identity_leakage_count"
    ] == 0


def test_m43_requires_and_accepts_complete_one_shot_population_hash_chain() -> None:
    config, status = _validate_m43(_m43_complete_inputs())

    assert status["status"] == "complete_go"
    assert status["m43_lifecycle_required"] is True
    assert config["lifecycle_contract"] == "m43_pre_freeze_one_shot_hash_chain"
    assert _gate(status, "m43_lifecycle_artifact_chain")["passed"] is True
    provenance = _gate(status, "safety_calibration_provenance")
    assert provenance["passed"] is True
    assert provenance["observed"]["mode"] == (
        "m43_paired_delta_pre_freeze_one_shot"
    )


def test_m43_manifest_without_lifecycle_artifacts_is_fail_closed_no_go() -> None:
    manifest, _contract, _freeze, _receipt, _marker, _plan, _hashes = _m43_inputs()

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), manifest
    )

    assert status["status"] == "complete_no_go"
    assert _gate(status, "m43_lifecycle_artifact_chain")["passed"] is False


@pytest.mark.parametrize(
    ("target_name", "path", "tampered"),
    [
        ("source_hashes", ("training_manifest",), "e" * 64),
        ("source_hashes", ("m43_consumption_marker",), "e" * 64),
        ("source_hashes", ("m43_population_plan",), "f" * 64),
        ("freeze", ("training_manifest_sha256",), "e" * 64),
        ("receipt", ("threshold_search_performed",), True),
        ("receipt", ("freeze_manifest_sha256",), "e" * 64),
        ("marker", ("freeze_manifest_sha256",), "e" * 64),
        ("marker", ("evaluation_pass_count",), 2),
        ("contract", ("base_audit_sha256",), "e" * 64),
        ("manifest", ("action_score_formula", "mode"), "negative_regret_ranker_v2"),
        ("manifest", ("calibration", "selected_metrics", "fires"), 9),
        ("plan", ("seed",), 10_001),
        (
            ("plan"),
            ("freshness", "excluded_population_schedules"),
            [{"seed": 10_000, "seed_stride": 101, "paired_seeds": 1}],
        ),
        ("evaluation", ("runtime_config", "candidate_model_sha256"), "e" * 64),
        ("evaluation", ("runtime_config", "safety_threshold"), 0.81),
        ("evaluation", ("runtime_config", "population_plan_sha256"), "f" * 64),
        ("evaluation", ("runtime_config", "action_score_mode"), "negative_regret_ranker_v2"),
        ("merge_manifest", ("evaluation_sha256",), "f" * 64),
        ("spot_receipt", ("records_sha256",), "f" * 64),
        ("run_manifest", ("runtime", "model_sha256"), "f" * 64),
    ],
)
def test_m43_malformed_or_unbound_artifact_is_no_go(
    target_name: str, path: tuple[str, ...], tampered: object
) -> None:
    inputs = _m43_complete_inputs()
    targets: dict[str, dict[str, object]] = {
        name: inputs[name]  # type: ignore[dict-item]
        for name in (
            "evaluation",
            "manifest",
            "contract",
            "freeze",
            "receipt",
            "marker",
            "plan",
            "merge_manifest",
            "spot_receipt",
            "run_manifest",
            "source_hashes",
        )
    }
    target = targets[target_name]
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = tampered

    _config, status = _validate_m43(inputs)

    assert status["status"] == "complete_no_go"
    assert _gate(status, "m43_lifecycle_artifact_chain")["passed"] is False


@pytest.mark.parametrize(
    ("path", "tampered"),
    [
        (("cross_fit", "schema"), "hu_m4_identity_group_nested_cross_fit_v2"),
        (("cross_fit", "oof_coverage", "paired_head_prediction_counts"), [0, 1]),
        (("cross_fit", "runtime_ensemble", "full_refit_used_at_runtime"), True),
        (("cross_fit", "runtime_ensemble", "baseline_action_score_exact_zero"), False),
        (("cross_fit", "runtime_ensemble", "oof_safety_fold_count"), 4),
        (("cross_fit", "predictor_lineage", "fold_audits"), []),
        (
            (
                "cross_fit",
                "predictor_lineage",
                "fold_audits",
                0,
                "oof_safety_inner_ensemble",
                "fold_count",
            ),
            4,
        ),
        (
            (
                "cross_fit",
                "split_role_overlap",
                "train_oof__locked_holdout",
                "identity_overlap_not_computed_from_locked_labels",
            ),
            False,
        ),
        (("calibration_partition", "locked_holdout_labels_opened_by_trainer"), True),
        (("calibration", "threshold_selection_source"), "locked_holdout"),
    ],
)
def test_m43_crossfit_or_calibration_tampering_is_no_go(
    path: tuple[str, ...], tampered: object
) -> None:
    inputs = _m43_complete_inputs()
    target = inputs["manifest"]
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = tampered

    _config, status = _validate_m43(inputs)

    assert status["status"] == "complete_no_go"
    assert _gate(status, "safety_calibration_provenance")["passed"] is False


def test_m43_rejects_hand_authored_summary_and_forged_nonfire_record() -> None:
    inputs = _m43_complete_inputs()
    evaluation = inputs["evaluation"]
    evaluation["population"]["by_seat"]["second"]["delta_ev_per_hand"][  # type: ignore[index]
        "mean"
    ] = 99.0
    _config, status = _validate_m43(inputs)
    assert status["status"] == "complete_no_go"
    lifecycle = _gate(status, "m43_lifecycle_artifact_chain")
    assert lifecycle["passed"] is False
    assert lifecycle["observed"]["population_content_checks"][  # type: ignore[index]
        "evaluation_recomputed_from_records"
    ] is False

    inputs = _m43_complete_inputs()
    nonfire = next(
        row
        for row in inputs["records"]  # type: ignore[union-attr]
        if row["override_fired"] is False
    )
    nonfire["candidate_gameplay_digest"] = "f" * 64
    assert nonfire["nonfire_cancellation_valid"] is True
    _config, status = _validate_m43(inputs)
    assert status["status"] == "complete_no_go"
    lifecycle = _gate(status, "m43_lifecycle_artifact_chain")
    assert lifecycle["observed"]["population_content_checks"][  # type: ignore[index]
        "records_content_valid"
    ] is False


@pytest.mark.parametrize(
    ("path", "tampered"),
    [
        (("cross_fit", "status"), "failed"),
        (("cross_fit", "folds"), 1),
        (("cross_fit", "action_score_mode"), "legacy_fixed_composition_v1"),
        (("cross_fit", "fold_assignment", "identity_leakage_count"), 1),
        (("cross_fit", "oof_coverage", "base_head_prediction_counts"), [0, 1]),
        (("cross_fit", "oof_coverage", "each_sample_predicted_exactly_once"), False),
        (("cross_fit", "predictor_lineage", "nested_cross_fit"), False),
        (("cross_fit", "predictor_lineage", "identity_leakage_count"), 1),
        (
            (
                "cross_fit",
                "predictor_lineage",
                "all_outer_validation_identities_excluded",
            ),
            False,
        ),
        (("cross_fit", "safety_examples", "used_for_threshold_lock"), True),
        (("cross_fit", "safety_examples", "source"), "train_in_sample"),
        (
            (
                "cross_fit",
                "split_role_overlap",
                "train_oof__locked_holdout",
                "row_hash_count",
            ),
            1,
        ),
        (
            (
                "calibration",
                "oof_train_calibration_overlap",
                "observation_fingerprint_count",
            ),
            1,
        ),
        (
            (
                "calibration",
                "safety_estimator_training_sources",
                "locked_holdout",
                "used_for_estimator_fit",
            ),
            True,
        ),
    ],
)
def test_m42_oof_safety_provenance_tampering_is_no_go(
    path: tuple[str, ...], tampered: object
) -> None:
    manifest = _m42_manifest()
    target = manifest
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = tampered

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), manifest
    )

    assert status["status"] == "complete_no_go"
    assert _gate(status, "safety_calibration_provenance")["passed"] is False


def test_population_teacher_seed_overlap_is_a_provenance_no_go() -> None:
    evaluation = _evaluation()
    evaluation["seed"] = 1

    _config, status = validate_hu_m4_acceptance(
        evaluation, _audit(), _m42_manifest()
    )

    gate = _gate(status, "fresh_evaluation_seed_disjoint")
    assert gate["passed"] is False
    assert gate["observed"]["overlap_count"] == 1
    assert status["status"] == "complete_no_go"


@pytest.mark.parametrize(
    "gate_name",
    [
        "within_split_shard_paths_unique",
        "within_split_hand_seeds_unique",
        "within_split_fingerprints_unique",
    ],
)
def test_within_split_uniqueness_tampering_is_no_go(gate_name: str) -> None:
    audit = _audit()
    audit["gates"][gate_name] = False  # type: ignore[index]

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), audit, _m41_manifest()
    )

    assert status["status"] == "complete_no_go"
    correctness = _gate(status, "data_correctness_audit")
    assert correctness["passed"] is False
    assert correctness["observed"][gate_name] is False


def test_legacy_audit_without_within_split_proof_is_compatible_no_go() -> None:
    audit = _audit()
    for gate_name in (
        "within_split_shard_paths_unique",
        "within_split_hand_seeds_unique",
        "within_split_fingerprints_unique",
    ):
        audit["gates"].pop(gate_name)  # type: ignore[index]

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), audit, _manifest()
    )

    # The old schema remains readable, but absence of the new uniqueness proof
    # cannot promote a candidate.
    assert _gate(status, "data_audit_schema_and_status")["passed"] is True
    assert _gate(status, "data_correctness_audit")["passed"] is False
    assert status["status"] == "complete_no_go"


@pytest.mark.parametrize(
    ("path", "tampered"),
    [
        (("calibration_partition", "status"), "no_go_insufficient_samples"),
        (("calibration_partition", "overlap", "seed_value_count"), 1),
        (("calibration", "fit_lock_overlap", "row_hash_count"), 1),
        (("calibration", "safety_estimator_source"), "calibration_file_only"),
        (("calibration", "threshold_source"), "calibration_file_only"),
        (("calibration", "safety_fit", "used_for_threshold_sweep"), True),
        (("calibration", "threshold_lock", "used_for_estimator_fit"), True),
        (("calibration_partition", "locked_holdout_used"), True),
        (("calibration", "locked_holdout_used"), True),
    ],
)
def test_m41_calibration_provenance_tampering_is_no_go(
    path: tuple[str, ...], tampered: object
) -> None:
    manifest = _m41_manifest()
    target = manifest
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = tampered

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), manifest
    )

    assert status["status"] == "complete_no_go"
    if path[-1] == "locked_holdout_used":
        assert _gate(status, "locked_holdout_not_researched")["passed"] is False
    else:
        assert _gate(status, "safety_calibration_provenance")["passed"] is False


def test_partial_m41_partition_cannot_fall_back_to_legacy_acceptance() -> None:
    manifest = _manifest()
    manifest["calibration_partition"] = {}

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), manifest
    )

    assert status["status"] == "complete_no_go"
    assert _gate(status, "safety_calibration_provenance")["passed"] is False


def test_v2_manifest_without_partition_cannot_fall_back_to_legacy() -> None:
    manifest = _manifest()
    manifest["schema"] = "hu_m4_t1_joint_training_manifest_v2"

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), manifest
    )

    assert _gate(status, "training_manifest_schema")["passed"] is True
    provenance = _gate(status, "safety_calibration_provenance")
    assert provenance["passed"] is False
    assert provenance["observed"]["mode"] == "schema_partition_contract_mismatch"
    assert status["status"] == "complete_no_go"


def test_insufficient_fires_is_a_completed_no_go_not_an_error() -> None:
    evaluation = _evaluation()
    evaluation["population"]["all_seats"]["overrides"] = 299  # type: ignore[index]

    _config, status = validate_hu_m4_acceptance(
        evaluation, _audit(), _manifest()
    )

    assert status["status"] == "complete_no_go"
    assert status["promotion_decision"] == "no_go"
    assert status["insufficient_sample_policy"] == "complete_no_go"
    assert _gate(status, "minimum_valid_overrides")["passed"] is False


def test_counterfactual_tail_and_seat_gates_fail_closed() -> None:
    evaluation = _evaluation()
    all_seats = evaluation["population"]["all_seats"]  # type: ignore[index]
    all_seats["nonfire_cancellation_unknown"] = 1
    all_seats["false_positive_override_rate"] = 0.31
    all_seats["override_loss_tail"] = {"p95": 25.01, "p99": 40.01, "max": 50.01}
    evaluation["population"]["by_seat"]["first"]["delta_ev_per_hand"] = _ci(  # type: ignore[index]
        0.001, 0.0, 0.002
    )

    _config, status = validate_hu_m4_acceptance(
        evaluation, _audit(), _manifest()
    )

    assert status["status"] == "complete_no_go"
    assert _gate(status, "nonfire_counterfactual_cancellation")["passed"] is False
    assert _gate(status, "false_positive_override_rate")["passed"] is False
    assert _gate(status, "override_loss_tail")["passed"] is False
    assert _gate(status, "first_seat_exact_cancellation")["passed"] is False


def test_runtime_lock_holdout_and_current_must_match_exactly() -> None:
    evaluation = _evaluation()
    audit = _audit()
    manifest = _manifest()
    evaluation["runtime_config"]["safety_threshold"] = 0.81  # type: ignore[index]
    evaluation["runtime_config"]["candidate_model_sha256"] = "c" * 64  # type: ignore[index]
    evaluation["runtime_config"]["current_profile_used"] = True  # type: ignore[index]
    manifest["locked_holdout_used_for_threshold_or_training"] = True

    _config, status = validate_hu_m4_acceptance(evaluation, audit, manifest)

    assert status["status"] == "complete_no_go"
    assert _gate(status, "frozen_threshold_match")["passed"] is False
    assert _gate(status, "runtime_artifact_hash_match")["passed"] is False
    assert _gate(status, "current_profile_not_used")["passed"] is False
    assert _gate(status, "locked_holdout_not_researched")["passed"] is False


def test_each_opponent_and_reported_worst_case_are_both_checked() -> None:
    evaluation = _evaluation()
    evaluation["by_opponent"]["stage19_p0"]["seat_swap"]["delta_ev_per_hand"] = _ci(  # type: ignore[index]
        -0.006, -0.019, 0.007
    )

    _config, status = validate_hu_m4_acceptance(
        evaluation, _audit(), _manifest()
    )

    assert _gate(status, "each_opponent_robustness")["passed"] is False
    assert _gate(status, "population_worst_case_robustness")["passed"] is False


def test_cli_writes_locked_config_and_complete_no_go(tmp_path: Path) -> None:
    evaluation = _evaluation()
    evaluation["population"]["all_seats"]["overrides"] = 10  # type: ignore[index]
    sources = {
        "evaluation": evaluation,
        "data_audit": _audit(),
        "training_manifest": _manifest(),
    }
    paths: dict[str, Path] = {}
    for name, payload in sources.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    config_output = tmp_path / "acceptance_config.json"
    status_output = tmp_path / "acceptance_status.json"

    exit_code = main(
        [
            "--evaluation",
            str(paths["evaluation"]),
            "--data-audit",
            str(paths["data_audit"]),
            "--training-manifest",
            str(paths["training_manifest"]),
            "--config-output",
            str(config_output),
            "--status-output",
            str(status_output),
        ]
    )

    assert exit_code == 0
    config = json.loads(config_output.read_text(encoding="utf-8"))
    status = json.loads(status_output.read_text(encoding="utf-8"))
    assert config["source_sha256"].keys() == sources.keys()
    assert config["fixed_gates"]["valid_overrides_min"] == 300
    assert status["status"] == "complete_no_go"


def test_m43_cli_hashes_and_binds_every_lifecycle_file(tmp_path: Path) -> None:
    inputs = _m43_complete_inputs()
    manifest = inputs["manifest"]
    contract = inputs["contract"]
    freeze = inputs["freeze"]
    locked_receipt = inputs["receipt"]
    marker = inputs["marker"]
    plan = inputs["plan"]
    evaluation = inputs["evaluation"]
    records = inputs["records"]
    merge_manifest = inputs["merge_manifest"]
    spot_receipt = inputs["spot_receipt"]
    run_manifest = inputs["run_manifest"]
    audit = _audit()
    paths = {
        "evaluation": tmp_path / "evaluation.json",
        "data_audit": tmp_path / "data_audit.json",
        "training_manifest": tmp_path / "training_manifest.json",
        "m43_data_contract": tmp_path / "data_contract.json",
        "m43_freeze_manifest": tmp_path / "freeze.json",
        "m43_locked_receipt": tmp_path / "locked_receipt.json",
        "m43_consumption_marker": tmp_path / "M43_LOCKED_CONSUMED.json",
        "m43_population_plan": tmp_path / "population_plan.json",
        "m43_population_records": tmp_path / "records.jsonl",
        "m43_population_merge_manifest": tmp_path / "merge_manifest.json",
        "m43_population_spot_receipt": tmp_path / "spot_receipt.json",
        "m43_population_run_manifest": tmp_path / "run_manifest.json",
    }
    paths["training_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    freeze["training_manifest_sha256"] = hashlib.sha256(
        paths["training_manifest"].read_bytes()
    ).hexdigest()
    freeze_sha = canonical_manifest_sha256(freeze)
    locked_receipt["freeze_manifest_sha256"] = freeze_sha
    marker["freeze_manifest_sha256"] = freeze_sha
    paths["m43_consumption_marker"].write_text(
        json.dumps(marker), encoding="utf-8"
    )
    locked_receipt["consumption_marker_sha256"] = hashlib.sha256(
        paths["m43_consumption_marker"].read_bytes()
    ).hexdigest()
    paths["m43_population_plan"].write_text(json.dumps(plan), encoding="utf-8")
    plan_sha = hashlib.sha256(paths["m43_population_plan"].read_bytes()).hexdigest()
    evaluation["runtime_config"]["population_plan_sha256"] = plan_sha  # type: ignore[index]
    evaluation["shard_merge"]["population_plan_sha256"] = plan_sha  # type: ignore[index]
    paths["m43_population_records"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    for name, payload in (
        ("evaluation", evaluation),
        ("data_audit", audit),
        ("m43_data_contract", contract),
        ("m43_freeze_manifest", freeze),
        ("m43_locked_receipt", locked_receipt),
    ):
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    evaluation_sha = hashlib.sha256(paths["evaluation"].read_bytes()).hexdigest()
    records_sha = hashlib.sha256(
        paths["m43_population_records"].read_bytes()
    ).hexdigest()
    merge_manifest.update(  # type: ignore[union-attr]
        {
            "population_plan_sha256": plan_sha,
            "evaluation_sha256": evaluation_sha,
            "merged_records_sha256": records_sha,
        }
    )
    paths["m43_population_merge_manifest"].write_text(
        json.dumps(merge_manifest), encoding="utf-8"
    )
    merge_sha = hashlib.sha256(
        paths["m43_population_merge_manifest"].read_bytes()
    ).hexdigest()
    run_manifest["population_plan"]["sha256"] = plan_sha  # type: ignore[index]
    run_manifest["runtime"].update(  # type: ignore[union-attr]
        {
            "training_manifest_sha256": hashlib.sha256(
                paths["training_manifest"].read_bytes()
            ).hexdigest(),
            "freeze_manifest_sha256": hashlib.sha256(
                paths["m43_freeze_manifest"].read_bytes()
            ).hexdigest(),
            "locked_receipt_sha256": hashlib.sha256(
                paths["m43_locked_receipt"].read_bytes()
            ).hexdigest(),
            "consumption_marker_sha256": hashlib.sha256(
                paths["m43_consumption_marker"].read_bytes()
            ).hexdigest(),
        }
    )
    paths["m43_population_run_manifest"].write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )
    run_sha = hashlib.sha256(
        paths["m43_population_run_manifest"].read_bytes()
    ).hexdigest()
    spot_receipt.update(  # type: ignore[union-attr]
        {
            "run_manifest_sha256": run_sha,
            "population_plan_sha256": plan_sha,
            "evaluation_sha256": evaluation_sha,
            "records_sha256": records_sha,
            "merge_manifest_sha256": merge_sha,
        }
    )
    paths["m43_population_spot_receipt"].write_text(
        json.dumps(spot_receipt), encoding="utf-8"
    )
    config_output = tmp_path / "acceptance_config.json"
    status_output = tmp_path / "acceptance_status.json"

    exit_code = main(
        [
            "--evaluation",
            str(paths["evaluation"]),
            "--data-audit",
            str(paths["data_audit"]),
            "--training-manifest",
            str(paths["training_manifest"]),
            "--m43-data-contract",
            str(paths["m43_data_contract"]),
            "--m43-freeze-manifest",
            str(paths["m43_freeze_manifest"]),
            "--m43-locked-receipt",
            str(paths["m43_locked_receipt"]),
            "--m43-consumption-marker",
            str(paths["m43_consumption_marker"]),
            "--m43-population-plan",
            str(paths["m43_population_plan"]),
            "--m43-population-records",
            str(paths["m43_population_records"]),
            "--m43-population-merge-manifest",
            str(paths["m43_population_merge_manifest"]),
            "--m43-population-spot-receipt",
            str(paths["m43_population_spot_receipt"]),
            "--m43-population-run-manifest",
            str(paths["m43_population_run_manifest"]),
            "--config-output",
            str(config_output),
            "--status-output",
            str(status_output),
        ]
    )

    assert exit_code == 0
    config = json.loads(config_output.read_text(encoding="utf-8"))
    status = json.loads(status_output.read_text(encoding="utf-8"))
    assert status["status"] == "complete_go"
    assert _gate(status, "m43_lifecycle_artifact_chain")["passed"] is True
    assert set(config["source_sha256"]) == set(paths)


def test_missing_manifest_artifact_hash_is_no_go() -> None:
    manifest = deepcopy(_manifest())
    manifest.pop("runtime_lock")

    _config, status = validate_hu_m4_acceptance(
        _evaluation(), _audit(), manifest
    )

    assert status["status"] == "complete_no_go"
    assert _gate(status, "runtime_artifact_hash_match")["passed"] is False
