from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ofc_regular import validate_hu_m43_attempt13_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt13_population.json"


def test_attempt13_population_plan_is_byte_frozen_fresh_and_balanced() -> None:
    plan = acceptance.load_and_validate_attempt13_population_plan(PLAN)
    assert plan["search_plan_sha256"] == acceptance.M43_ATTEMPT13_PLAN_SHA256
    assert plan["opponents"] == list(acceptance.ATTEMPT13_OPPONENTS)
    assert plan["paired_seat_swap"] is True
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] * plan["paired_seeds_per_shard"] == 1000
    assert plan["candidate_records"] == plan["baseline_records"] == 8000
    assert plan["terminal_trace_hands"] == 16000
    assert plan["_freshness_counts"] == {
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "all_reserved_registry_overlap_count": 0,
    }
    assert plan["_freshness_registry"]["source_count"] == len(
        acceptance.ATTEMPT13_SEED_REGISTRY_SOURCES
    )


def test_attempt13_population_plan_fails_closed_on_hash_or_contract_drift(
    tmp_path: Path,
) -> None:
    changed = json.loads(PLAN.read_text(encoding="utf-8"))
    changed["fixed_acceptance_gates"]["override_loss_p99_max"] = 41
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="plan SHA changed"):
        acceptance.load_and_validate_attempt13_population_plan(changed_path)
    with pytest.raises(ValueError, match="fixed acceptance gates changed"):
        acceptance.load_and_validate_attempt13_population_plan(changed)


def test_attempt13_all_reserved_namespaces_are_outside_hashed_registry() -> None:
    plan = acceptance.load_and_validate_attempt13_population_plan(PLAN)
    registry_seeds, _rows = acceptance._load_seed_registry(
        acceptance.ATTEMPT13_SEED_REGISTRY_SOURCES
    )
    reserved = {
        base + index * acceptance.ATTEMPT13_POPULATION_SEED_STRIDE
        for base in acceptance.ATTEMPT13_POPULATION_NAMESPACE_BASES
        for index in range(acceptance.ATTEMPT13_POPULATION_SEEDS)
    }
    assert reserved.isdisjoint(registry_seeds)
    assert plan["_freshness_registry"]["planned_overlap_count"] == 0


def test_attempt13_audit_go_requires_one_shot_all_pass_receipt() -> None:
    decision_sha = "a" * 64
    audit = {
        "schema": acceptance.ATTEMPT13_AUDIT50_DECISION_SCHEMA,
        "status": "go_attempt13_audit50_search_quality",
        "decision": "go",
        "search_freeze_authorized": True,
        "selected_arm": None,
        "selected_threshold": None,
        "science_boundary": {
            "fit_performed": False,
            "threshold_selected": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "teacher_values_are_realized_match_ev": False,
        },
        "gates": [{"name": "all", "passed": True}],
        "decision_contract": {
            "gate_evaluation_count": 1,
            "single_frozen_search_architecture": True,
            "arm_selection_performed": False,
            "threshold_selection_performed": False,
            "all_gates_required": True,
        },
    }
    receipt = {
        "schema": acceptance.ATTEMPT13_AUDIT50_SELECTOR_RECEIPT_SCHEMA,
        "status": "single_frozen_gate_evaluation_complete",
        "decision_sha256": decision_sha,
        "decision": "go",
        "search_freeze_authorized": True,
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "future_audit_authorized": False,
        "audit_rows_used_for_fit": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "run_name": "attempt13-audit50",
    }
    acceptance._validate_attempt13_audit50_go_authorization(
        audit, receipt, audit_decision_sha256=decision_sha
    )
    changed = copy.deepcopy(audit)
    changed["gates"][0]["passed"] = False
    with pytest.raises(ValueError, match="Go contract changed"):
        acceptance._validate_attempt13_audit50_go_authorization(
            changed, receipt, audit_decision_sha256=decision_sha
        )


def test_attempt13_population_preflight_emits_complete_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "model.pkl",
            "training.json",
            "freeze.json",
            "plan.json",
            "runtime.zip",
            "runtime_manifest.json",
        )
    }
    dev = {
        "development_decision_sha256": "1" * 64,
        "development_selector_receipt_sha256": "2" * 64,
        "development_pass_freeze_sha256": "3" * 64,
    }
    audit = {
        "audit50_decision_sha256": "4" * 64,
        "audit50_selector_receipt_sha256": "5" * 64,
    }
    paths["model.pkl"].write_bytes(b"model")
    paths["training.json"].write_text(json.dumps({"source": dev}), encoding="utf-8")
    paths["freeze.json"].write_text(json.dumps(audit), encoding="utf-8")
    for name in ("plan.json", "runtime.zip", "runtime_manifest.json"):
        paths[name].write_bytes(name.encode("ascii"))
    monkeypatch.setattr(
        acceptance,
        "load_and_validate_attempt13_population_plan",
        lambda _: {
            "_freshness_counts": {
                "teacher_overlap_count": 0,
                "prior_population_overlap_count": 0,
                "all_reserved_registry_overlap_count": 0,
            },
            "_freshness_registry": {
                "sha256": "6" * 64,
                "source_count": 39,
                "seed_count": 1,
                "sources": ["registry"],
            },
        },
    )
    monkeypatch.setattr(
        acceptance,
        "validate_distilled_runtime_source_archive",
        lambda **_: {
            "file_set": {"sha256": "7" * 64},
            "semantic_closure": {"sha256": "8" * 64},
        },
    )
    monkeypatch.setattr(
        acceptance,
        "load_bound_attempt13_distilled_model",
        lambda *_, **__: SimpleNamespace(
            schema=acceptance.HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
            artifact_schema=acceptance.HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
            feature_schema=acceptance.HU_M43_ATTEMPT13_DISTILLED_FEATURE_SCHEMA,
            head_schema=acceptance.HU_M43_ATTEMPT13_DISTILLED_HEAD_SCHEMA,
            action_score_mode=acceptance.HU_M43_ATTEMPT13_DISTILLED_ACTION_SCORE_MODE,
            model_id="attempt13-test",
            safety_threshold=0.5,
            minimum_fold_votes=4,
            manifest={
                "source_model_manifest_sha256": "9" * 64,
                "source_native_manifest_sha256": "a" * 64,
                "runtime_dependency_closure_sha256": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_validate_attempt13_population_provenance_bundle",
        lambda **_: {**dev, **audit},
    )
    receipt = acceptance.build_attempt13_population_preflight(
        model_path=paths["model.pkl"],
        training_manifest_path=paths["training.json"],
        runtime_freeze_path=paths["freeze.json"],
        population_plan_path=paths["plan.json"],
        runtime_source_archive_path=paths["runtime.zip"],
        runtime_source_manifest_path=paths["runtime_manifest.json"],
        runtime_source_root=tmp_path,
        runtime_dependency_root=tmp_path,
        development_decision_path=tmp_path / "development_decision.json",
        development_selector_receipt_path=tmp_path / "development_receipt.json",
        development_pass_freeze_path=tmp_path / "development_freeze.json",
        audit_decision_path=tmp_path / "audit_decision.json",
        audit_selector_receipt_path=tmp_path / "audit_receipt.json",
    )
    assert receipt["profile_id"] == "stage20_m4_attempt13"
    assert receipt["opponents"] == list(acceptance.ATTEMPT13_OPPONENTS)
    assert receipt["population_namespace_bases"] == list(
        acceptance.ATTEMPT13_POPULATION_NAMESPACE_BASES
    )
    assert receipt["runtime_binding_verified"] is True
    assert receipt["audit50_fit_rows"] == 0
    assert receipt["current_profile_mutated"] is False


def test_attempt13_variable_training_groups_cover_zero_to_26_candidates() -> None:
    groups = [1 + index % 27 for index in range(200)]
    counts = [value - 1 for value in groups]
    diagnostics = {
        "rows": sum(groups),
        "group_sizes": groups,
        "candidate_count_min": min(counts),
        "candidate_count_max": max(counts),
        "candidate_count_histogram": {
            str(value): counts.count(value) for value in range(27)
        },
    }
    assert acceptance._valid_variable_training_diagnostics(diagnostics)
    diagnostics["group_sizes"] = groups[:-1] + [28]
    assert not acceptance._valid_variable_training_diagnostics(diagnostics)


def test_attempt13_complete_go_is_issued_only_after_recomputed_population_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_sha = "a" * 64
    preflight = {
        "model_sha256": model_sha,
        "model_id": "attempt13-model",
        "runtime_freeze_sha256": "b" * 64,
        "training_manifest_sha256": "c" * 64,
        "runtime_source_manifest_sha256": "d" * 64,
        "runtime_source_closure_sha256": "e" * 64,
        "runtime_semantic_closure_sha256": "f" * 64,
        "source_model_manifest_sha256": "1" * 64,
        "source_native_manifest_sha256": "2" * 64,
        "runtime_dependency_closure_sha256": "3" * 64,
        "seed_registry_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        acceptance, "build_attempt13_population_preflight", lambda **_: preflight
    )
    recomputed = {
        "schema": "hu_m4_t1_population_evaluation_v1",
        "records_output": None,
        "elapsed_seconds": None,
    }
    monkeypatch.setattr(
        acceptance, "summarize_hu_m4_population_records", lambda *_, **__: recomputed
    )
    monkeypatch.setattr(
        acceptance,
        "evaluate_attempt02_population_gates",
        lambda _: [
            {
                "name": "realized_population",
                "passed": True,
                "observed": True,
                "requirement": "fixed gates",
            }
        ],
    )
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    runtime = {
        "candidate_model_sha256": model_sha,
        "safety_model_sha256": model_sha,
        "model_schema": acceptance.HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
        "artifact_schema": acceptance.HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
        "feature_schema": acceptance.HU_M43_ATTEMPT13_DISTILLED_FEATURE_SCHEMA,
        "head_schema": acceptance.HU_M43_ATTEMPT13_DISTILLED_HEAD_SCHEMA,
        "model_id": "attempt13-model",
        "action_score_mode": acceptance.HU_M43_ATTEMPT13_DISTILLED_ACTION_SCORE_MODE,
        "profile_id": acceptance.ATTEMPT13_PROFILE_ID,
        "baseline_profile": acceptance.ATTEMPT13_BASELINE_PROFILE,
        "opponents": list(acceptance.ATTEMPT13_OPPONENTS),
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": "b" * 64,
        "training_manifest_sha256": "c" * 64,
        "runtime_source_manifest_sha256": "d" * 64,
        "runtime_source_closure_sha256": "e" * 64,
        "runtime_semantic_closure_sha256": "f" * 64,
        "runtime_requirements_sha256": acceptance.ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": acceptance.ATTEMPT13_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": "1" * 64,
        "source_native_manifest_sha256": "2" * 64,
        "runtime_dependency_closure_sha256": "3" * 64,
        "seed_registry_sha256": "4" * 64,
        "population_namespace_bases": list(
            acceptance.ATTEMPT13_POPULATION_NAMESPACE_BASES
        ),
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
    evaluation = {**recomputed, "runtime_config": runtime}
    merge = {
        "schema": acceptance.M4_POPULATION_MERGE_SCHEMA,
        "status": "complete_content_verified",
        "population_plan_sha256": plan_sha,
        "seed": acceptance.ATTEMPT13_POPULATION_SEED,
        "seed_stride": acceptance.ATTEMPT13_POPULATION_SEED_STRIDE,
        "paired_seeds_per_opponent": acceptance.ATTEMPT13_POPULATION_SEEDS,
        "opponents": list(acceptance.ATTEMPT13_OPPONENTS),
        "merged_records": 8000,
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
        "shards": [{} for _ in range(20)],
    }
    kwargs = {
        "evaluation": evaluation,
        "records": [{"runtime_binding_verified": True}],
        "population_plan": plan,
        "merge_manifest": merge,
        "model_path": "model.pkl",
        "training_manifest_path": "training.json",
        "runtime_freeze_path": "freeze.json",
        "runtime_source_archive_path": "runtime.zip",
        "runtime_source_manifest_path": "runtime.json",
        "runtime_source_root": ".",
        "runtime_dependency_root": ".",
        "source_hashes": {
            "population_plan": plan_sha,
            "population_plan_path": str(PLAN),
            "records": "5" * 64,
            "evaluation": "6" * 64,
            "merge_manifest": "7" * 64,
        },
    }
    status = acceptance.validate_attempt13_population_acceptance(**kwargs)
    assert status["status"] == "complete_go"
    assert status["profile_id"] == "stage20_m4_attempt13"
    assert status["explicit_opt_in_authorized"] is True
    assert status["automatic_activation_authorized"] is False
    unsafe = copy.deepcopy(evaluation)
    unsafe["runtime_config"]["current_profile_used"] = True
    kwargs["evaluation"] = unsafe
    status = acceptance.validate_attempt13_population_acceptance(**kwargs)
    assert status["status"] == "complete_no_go"
    assert status["explicit_opt_in_authorized"] is False
