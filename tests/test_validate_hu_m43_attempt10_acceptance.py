from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ofc_regular.validate_hu_m43_attempt10_acceptance as acceptance
from ofc_regular.hu_m43_attempt10_distilled_model import (
    HU_M43_ATTEMPT10_DISTILLED_ACTION_SCORE_MODE,
    HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA,
    HU_M43_ATTEMPT10_DISTILLED_FEATURE_SCHEMA,
    HU_M43_ATTEMPT10_DISTILLED_HEAD_SCHEMA,
    HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt10_distilled_model,
)
from ofc_regular.hu_m43_attempt10_distilled_runtime import (
    ATTEMPT10_BOUND_EXECUTION_MODULES,
    ATTEMPT10_DISTILLED_RUNTIME_DEPENDENCY_SCHEMA,
    FrozenExecutionModulesAttestation,
)
from ofc_regular.train_hu_m43_attempt10_distilled import (
    ATTEMPT10_DISTILLATION_CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt10_population.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attestation() -> FrozenExecutionModulesAttestation:
    value = object.__new__(FrozenExecutionModulesAttestation)
    object.__setattr__(value, "extracted_root", "unit-test-frozen-tree")
    object.__setattr__(
        value,
        "module_paths",
        tuple(
            (name, f"src/{name.replace('.', '/')}.py")
            for name in ATTEMPT10_BOUND_EXECUTION_MODULES
        ),
    )
    return value


def test_attempt10_population_plan_is_fresh_fixed_and_science_safe() -> None:
    plan = acceptance.load_and_validate_attempt10_population_plan(PLAN)
    assert plan["seed"] == 180_108_071_901
    assert plan["seed_stride"] == 1_000_003
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["_freshness_counts"] == {
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
    }
    assert plan["_freshness_registry"]["planned_overlap_count"] == 0
    assert plan["evaluation_contract"]["teacher_values_are_realized_match_ev"] is False
    assert plan["evaluation_contract"]["nonfire_full_trajectory_digest_cancellation"] is True
    assert plan["activation_guards"] == {
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
        "threshold_changed_after_lock": False,
    }
    assert plan["post_acceptance_activation"] == {
        "population_complete_go_required": True,
        "activation_mode_if_go": "explicit_opt_in_only",
        "automatic_activation_allowed": False,
        "current_profile_change_allowed": False,
    }


def test_attempt10_population_plan_rejects_registry_or_seed_drift() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(plan)
    changed["freshness"]["excluded_schedule_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hashed seed registry snapshot"):
        acceptance.load_and_validate_attempt10_population_plan(changed)
    changed = copy.deepcopy(plan)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="schedule changed"):
        acceptance.load_and_validate_attempt10_population_plan(changed)


def test_training_manifest_validation_matches_attempt10_trainer_contract() -> None:
    arbitrary = "a" * 64
    source = {
        name: arbitrary
        for name in (
            "development_jsonl_sha256",
            "development_decision_sha256",
            "development_selector_receipt_sha256",
            "development_pass_freeze_sha256",
            "development_package_manifest_sha256",
            "development_source_package_sha256",
            "development_root_identity_sha256",
            "runtime_source_archive_sha256",
            "runtime_source_manifest_sha256",
            "runtime_source_closure_sha256",
            "runtime_semantic_closure_sha256",
            "source_model_manifest_sha256",
            "source_native_manifest_sha256",
            "runtime_dependency_closure_sha256",
        )
    }
    source.update(
        {
            "distillation_config_sha256": ATTEMPT10_DISTILLATION_CONFIG_SHA256,
            "candidate_model_sha256": acceptance.ATTEMPT10_LAMBDA_MODEL_SHA256,
            "development_plan_sha256": acceptance.M43_ATTEMPT10_PLAN_SHA256,
            "runtime_requirements_sha256": acceptance.ATTEMPT10_RUNTIME_REQUIREMENTS_SHA256,
            "runtime_fingerprint_sha256": acceptance.ATTEMPT10_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        }
    )
    manifest = {
        "schema": acceptance.ATTEMPT10_DISTILLED_TRAINING_MANIFEST_SCHEMA,
        "status": "fit_complete_runtime_disabled",
        "model_schema": HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
        "feature_schema": HU_M43_ATTEMPT10_DISTILLED_FEATURE_SCHEMA,
        "head_schema": HU_M43_ATTEMPT10_DISTILLED_HEAD_SCHEMA,
        "action_score_mode": HU_M43_ATTEMPT10_DISTILLED_ACTION_SCORE_MODE,
        "model_sha256": "b" * 64,
        "source": source,
        "diagnostics": {
            "states": 200,
            "rows": 2600,
            "fold_counts": {str(index): 40 for index in range(5)},
        },
        "fit_contract": {
            "fit_mode": "full",
            "effective_iterations": 180,
            "states": 200,
            "folds": 5,
        },
        "runtime": {
            "safety_enabled": False,
            "winner_frozen": False,
            "activation_allowed": False,
            "current_profile_mutated": False,
            "runtime_source_frozen": True,
            "runtime_requirements_sha256": acceptance.ATTEMPT10_RUNTIME_REQUIREMENTS_SHA256,
            "runtime_fingerprint_sha256": acceptance.ATTEMPT10_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        },
        "science_boundary": {
            "teacher_values_are_realized_match_ev": False,
            "audit50_fit_rows": 0,
            "threshold_sweep_performed": False,
            "top1_accuracy_is_acceptance_gate": False,
        },
    }
    acceptance._validate_training_manifest(
        manifest, expected_model_sha256=manifest["model_sha256"]
    )
    changed = copy.deepcopy(manifest)
    changed["science_boundary"]["teacher_values_are_realized_match_ev"] = True
    with pytest.raises(ValueError, match="training manifest contract"):
        acceptance._validate_training_manifest(
            changed, expected_model_sha256=changed["model_sha256"]
        )


def test_runtime_freeze_requires_exact_audit50_no_fit_receipt() -> None:
    decision_sha = "a" * 64
    receipt = {
        "schema": acceptance.ATTEMPT10_AUDIT50_SELECTOR_RECEIPT_SCHEMA,
        "status": "single_frozen_gate_evaluation_complete",
        "run_name": "regular-hu-m43-attempt10-audit50-unit",
        "decision_sha256": decision_sha,
        "decision": "go",
        "search_freeze_authorized": True,
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "future_audit_authorized": False,
        "audit_rows_used_for_fit": False,
        "fit_performed": False,
        "threshold_selected": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
    }
    acceptance._validate_attempt10_audit50_receipt_for_freeze(
        receipt, audit_decision_sha256=decision_sha
    )
    missing = dict(receipt)
    missing.pop("audit_rows_used_for_fit")
    with pytest.raises(ValueError, match="not freeze authorization"):
        acceptance._validate_attempt10_audit50_receipt_for_freeze(
            missing, audit_decision_sha256=decision_sha
        )
    extra = {**receipt, "unexpected": False}
    with pytest.raises(ValueError, match="not freeze authorization"):
        acceptance._validate_attempt10_audit50_receipt_for_freeze(
            extra, audit_decision_sha256=decision_sha
        )


def test_bound_loader_issues_only_semantically_bound_attempt10_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"frozen-attempt10-model")
    model_sha = _sha(model_path)
    training_path = tmp_path / "training.json"
    runtime_manifest_path = tmp_path / "runtime.json"
    runtime_manifest_path.write_text("{}\n", encoding="utf-8")
    runtime_manifest_sha = _sha(runtime_manifest_path)
    bindings = {
        "runtime_source_archive_sha256": "1" * 64,
        "runtime_source_manifest_sha256": runtime_manifest_sha,
        "runtime_source_closure_sha256": "2" * 64,
        "runtime_semantic_closure_sha256": "3" * 64,
        "runtime_requirements_sha256": acceptance.ATTEMPT10_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": acceptance.ATTEMPT10_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": "4" * 64,
        "source_native_manifest_sha256": "5" * 64,
        "runtime_dependency_closure_sha256": "6" * 64,
    }
    training = {
        "model_id": "attempt10-bound-test",
        "source": dict(bindings),
    }
    training_path.write_text(json.dumps(training) + "\n", encoding="utf-8")
    training_sha = _sha(training_path)
    embedded = {
        "fit_mode": "full",
        "effective_iterations": 180,
        "training_states": 200,
        "identity_grouped_folds": 5,
        "distillation_config_sha256": ATTEMPT10_DISTILLATION_CONFIG_SHA256,
        "training_manifest_sha256": training_sha,
        **bindings,
    }
    model = SimpleNamespace(
        safety_enabled=True,
        winner_frozen=True,
        model_id="attempt10-bound-test",
        schema=HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
        artifact_schema=HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA,
        feature_schema=HU_M43_ATTEMPT10_DISTILLED_FEATURE_SCHEMA,
        head_schema=HU_M43_ATTEMPT10_DISTILLED_HEAD_SCHEMA,
        action_score_mode=HU_M43_ATTEMPT10_DISTILLED_ACTION_SCORE_MODE,
        safety_threshold=0.5,
        minimum_fold_votes=4,
        manifest=embedded,
    )
    dependencies = {
        "schema": ATTEMPT10_DISTILLED_RUNTIME_DEPENDENCY_SCHEMA,
        "source_model_manifest_sha256": bindings["source_model_manifest_sha256"],
        "source_native_manifest_sha256": bindings["source_native_manifest_sha256"],
        "model_count": 11,
        "binary_count": 2,
        "sha256": bindings["runtime_dependency_closure_sha256"],
    }
    runtime = {
        "archive": {"sha256": bindings["runtime_source_archive_sha256"]},
        "file_set": {"sha256": bindings["runtime_source_closure_sha256"]},
        "semantic_closure": {
            "sha256": bindings["runtime_semantic_closure_sha256"],
            "teacher_contract": {
                "schema": "hu_m43_attempt10_teacher_lineage_v1",
                "plan_sha256": acceptance.M43_ATTEMPT10_PLAN_SHA256,
                "candidate_model_sha256": acceptance.ATTEMPT10_LAMBDA_MODEL_SHA256,
            },
            "runtime_dependencies": {
                key: dependencies[key]
                for key in (
                    "schema",
                    "source_model_manifest_sha256",
                    "source_native_manifest_sha256",
                    "model_count",
                    "binary_count",
                )
            },
        },
    }
    freeze = {
        "schema": acceptance.ATTEMPT10_RUNTIME_FREEZE_SCHEMA,
        "status": "frozen_after_development200_fit_and_audit50_go",
        "model_sha256": model_sha,
        "model_id": model.model_id,
        "model_schema": model.schema,
        "artifact_schema": model.artifact_schema,
        "feature_schema": model.feature_schema,
        "head_schema": model.head_schema,
        "action_score_mode": model.action_score_mode,
        "source_training_model_sha256": "7" * 64,
        "training_manifest_sha256": training_sha,
        "safety_enabled": True,
        "winner_frozen": True,
        "fixed_safe_probability_threshold": 0.5,
        "fixed_fold_votes_min": 4,
        "threshold_reselection_performed": False,
        "current_profile_mutated": False,
        "audit50_selector_receipt_sha256": "8" * 64,
        "gcp_image": {
            "name": acceptance.ATTEMPT10_GCP_IMAGE_NAME,
            "id": acceptance.ATTEMPT10_GCP_IMAGE_ID,
            "self_link": acceptance.ATTEMPT10_GCP_IMAGE_SELF_LINK,
        },
        **bindings,
    }
    monkeypatch.setattr(
        acceptance, "validate_frozen_execution_modules", lambda **_kwargs: _attestation()
    )
    monkeypatch.setattr(
        acceptance, "validate_distilled_runtime_extracted_tree", lambda **_kwargs: runtime
    )
    monkeypatch.setattr(
        acceptance, "validate_distilled_runtime_dependencies", lambda _root: dependencies
    )
    monkeypatch.setattr(acceptance, "_validate_training_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        acceptance.HuM43Attempt10DistilledModel,
        "load",
        staticmethod(lambda *_args, **_kwargs: model),
    )
    bound = acceptance.load_bound_attempt10_distilled_model(
        model_path,
        expected_sha256=model_sha,
        runtime_freeze=freeze,
        training_manifest_path=training_path,
        runtime_source_manifest_path=runtime_manifest_path,
        runtime_source_root=tmp_path / "tree",
        runtime_dependency_root=tmp_path / "dependencies",
    )
    assert is_bound_attempt10_distilled_model(bound)
    assert bound.runtime_binding_verified is True
    with pytest.raises(ValueError, match="bound model SHA mismatch"):
        acceptance.load_bound_attempt10_distilled_model(
            model_path,
            expected_sha256="0" * 64,
            runtime_freeze=freeze,
            training_manifest_path=training_path,
            runtime_source_manifest_path=runtime_manifest_path,
            runtime_source_root=tmp_path / "tree",
            runtime_dependency_root=tmp_path / "dependencies",
        )


def test_final_acceptance_uses_only_recomputed_realized_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan_sha = _sha(PLAN)
    preflight = {
        "model_sha256": "1" * 64,
        "model_id": "attempt10-population-test",
        "runtime_freeze_sha256": "2" * 64,
        "training_manifest_sha256": "3" * 64,
        "runtime_source_manifest_sha256": "4" * 64,
        "runtime_source_closure_sha256": "5" * 64,
        "runtime_semantic_closure_sha256": "6" * 64,
        "source_model_manifest_sha256": "7" * 64,
        "source_native_manifest_sha256": "8" * 64,
        "runtime_dependency_closure_sha256": "9" * 64,
        "seed_registry_sha256": acceptance.load_and_validate_attempt10_population_plan(
            PLAN
        )["_freshness_registry"]["sha256"],
    }
    base = {
        "schema": "hu_m4_t1_population_evaluation_v1",
        "evaluation_basis": "fresh_same_seed_candidate_vs_stage19_p0_counterfactual_v1",
        "records_output": None,
        "elapsed_seconds": None,
        "nonfire_cancellation_mismatches": 0,
    }
    runtime = {
        "candidate_model_sha256": preflight["model_sha256"],
        "safety_model_sha256": preflight["model_sha256"],
        "model_schema": HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
        "artifact_schema": HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA,
        "feature_schema": HU_M43_ATTEMPT10_DISTILLED_FEATURE_SCHEMA,
        "head_schema": HU_M43_ATTEMPT10_DISTILLED_HEAD_SCHEMA,
        "model_id": preflight["model_id"],
        "action_score_mode": HU_M43_ATTEMPT10_DISTILLED_ACTION_SCORE_MODE,
        "baseline_profile": "stage19_p0",
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": preflight["runtime_freeze_sha256"],
        "training_manifest_sha256": preflight["training_manifest_sha256"],
        "runtime_source_manifest_sha256": preflight["runtime_source_manifest_sha256"],
        "runtime_source_closure_sha256": preflight["runtime_source_closure_sha256"],
        "runtime_semantic_closure_sha256": preflight["runtime_semantic_closure_sha256"],
        "runtime_requirements_sha256": acceptance.ATTEMPT10_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": acceptance.ATTEMPT10_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": preflight["source_model_manifest_sha256"],
        "source_native_manifest_sha256": preflight["source_native_manifest_sha256"],
        "runtime_dependency_closure_sha256": preflight["runtime_dependency_closure_sha256"],
        "seed_registry_sha256": preflight["seed_registry_sha256"],
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "safety_enabled": True,
        "safety_threshold": 0.5,
        "population_plan_sha256": plan_sha,
        "sharded_evaluation": True,
        "shard_count": 20,
        "final_metrics_recomputed_from_merged_records": True,
    }
    evaluation = {**base, "runtime_config": runtime}
    merge = {
        "schema": acceptance.M4_POPULATION_MERGE_SCHEMA,
        "status": "complete_content_verified",
        "population_plan_sha256": plan_sha,
        "seed": acceptance.ATTEMPT10_POPULATION_SEED,
        "seed_stride": acceptance.ATTEMPT10_POPULATION_SEED_STRIDE,
        "paired_seeds_per_opponent": 1000,
        "opponents": list(acceptance.ATTEMPT10_OPPONENTS),
        "merged_records": 8000,
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
        "shards": [{} for _ in range(20)],
    }
    monkeypatch.setattr(
        acceptance, "build_attempt10_population_preflight", lambda **_kwargs: preflight
    )
    monkeypatch.setattr(
        acceptance, "summarize_hu_m4_population_records", lambda *_args, **_kwargs: dict(base)
    )
    monkeypatch.setattr(
        acceptance,
        "evaluate_attempt02_population_gates",
        lambda summary: [
            {
                "name": "nonfire_cancellation_mismatches",
                "passed": summary["nonfire_cancellation_mismatches"] == 0,
                "observed": summary["nonfire_cancellation_mismatches"],
                "requirement": "= 0",
            }
        ],
    )
    kwargs = {
        "evaluation": evaluation,
        "records": [{"runtime_binding_verified": True}],
        "population_plan": plan,
        "merge_manifest": merge,
        "model_path": "unused-model",
        "training_manifest_path": "unused-training",
        "runtime_freeze_path": "unused-freeze",
        "runtime_source_archive_path": "unused-archive",
        "runtime_source_manifest_path": "unused-manifest",
        "runtime_source_root": "unused-tree",
        "runtime_dependency_root": "unused-dependencies",
        "source_hashes": {
            "population_plan": plan_sha,
            "population_plan_path": str(PLAN),
            "records": "a" * 64,
            "evaluation": "b" * 64,
            "merge_manifest": "c" * 64,
        },
    }
    status = acceptance.validate_attempt10_population_acceptance(**kwargs)
    assert status["status"] == "complete_go"
    assert status["promotion_eligible"] is True
    assert status["explicit_opt_in_authorized"] is True
    assert status["automatic_activation_authorized"] is False
    assert status["teacher_values_reported_as_realized_match_ev"] is False
    assert status["current_profile_mutated"] is False
    assert status["runtime_policy_activated"] is False

    changed = copy.deepcopy(evaluation)
    changed["runtime_config"]["runtime_binding_verified"] = False
    rejected = acceptance.validate_attempt10_population_acceptance(
        **{**kwargs, "evaluation": changed}
    )
    assert rejected["status"] == "complete_no_go"
    assert rejected["promotion_eligible"] is False
    assert rejected["explicit_opt_in_authorized"] is False
