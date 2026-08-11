from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import ofc_regular.hu_m43_attempt08_distilled_runtime as distilled_runtime
import ofc_regular.validate_hu_m43_attempt08_acceptance as acceptance
from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput, HuM43Attempt05Model
from ofc_regular.hu_m43_attempt08_distilled_model import (
    Attempt08DistilledFoldPredictor,
    HuM43Attempt08DistilledModel,
)
from ofc_regular.hu_m43_attempt08_teacher import ATTEMPT08_FROZEN_MODEL_ID
from ofc_regular.hu_m43_attempt08_distilled_runtime import (
    build_distilled_runtime_source_freeze,
    extract_distilled_runtime_source_archive,
    validate_distilled_runtime_dependencies,
)
from ofc_regular.hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from ofc_regular.hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
)
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.train_hu_m43_attempt08_distilled import (
    ATTEMPT08_DISTILLATION_CONFIG_SHA256,
    ATTEMPT08_DISTILLED_TRAINING_MANIFEST_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt08_population.json"
DEPENDENCY_ROOT = (
    ROOT
    / "outputs/gcp_runs"
    / "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228"
    / "package_src"
)


class _CandidateFold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        count = len(runtime_sample["actions"])
        return Attempt05FoldOutput(
            rank_score=np.arange(count, dtype=np.float64),
            gain_probability=np.full(count, 0.5),
            downside_p95=np.full(count, 2.0),
            downside_p99=np.full(count, 4.0),
            downside_max=np.full(count, 6.0),
        )


class _Regression:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features):
        return np.full(np.asarray(features).shape[0], self.value)


class _Probability:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        positive = np.full(np.asarray(features).shape[0], 0.9)
        return np.column_stack((1.0 - positive, positive))


class _BaselinePolicy:
    def choose_action_observation(self, *_args, **_kwargs):
        raise AssertionError("binding tests must not play a hand")


def _source_model() -> HuM43Attempt08DistilledModel:
    candidate = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT08_FROZEN_MODEL_ID,
    )
    folds = tuple(
        Attempt08DistilledFoldPredictor(
            ranker=_Regression(1.0),
            delta_head=_Regression(1.0),
            safe_head=_Probability(),
            tail_p95_head=_Regression(2.0),
            tail_p99_head=_Regression(4.0),
            tail_max_head=_Regression(6.0),
            fold_index=index,
        )
        for index in range(5)
    )
    return HuM43Attempt08DistilledModel(
        candidate_generator=candidate,
        fold_predictors=folds,
        model_id="attempt08-distilled-acceptance-test",
        manifest={
            "training_data": "attempt08_development200_only",
            "audit50_fit_rows": 0,
            "threshold_sweep_performed": False,
            "identity_grouped_folds": 5,
            "fit_mode": "full",
            "effective_iterations": 180,
            "training_states": 200,
            "distillation_config_sha256": ATTEMPT08_DISTILLATION_CONFIG_SHA256,
        },
    )


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_artifacts(tmp_path: Path):
    dependencies = validate_distilled_runtime_dependencies(DEPENDENCY_ROOT)
    runtime_archive = tmp_path / "runtime_source.zip"
    runtime_manifest = tmp_path / "runtime_source.json"
    runtime_tree = tmp_path / "runtime_tree"
    runtime = build_distilled_runtime_source_freeze(
        source_root=ROOT,
        output_archive=runtime_archive,
        output_manifest=runtime_manifest,
    )
    extract_distilled_runtime_source_archive(
        archive_path=runtime_archive,
        manifest=runtime_manifest,
        output_root=runtime_tree,
    )
    source = tmp_path / "source.pkl"
    source_sha = _source_model().save(source)
    training = tmp_path / "training.json"
    _json(
        training,
        {
            "schema": ATTEMPT08_DISTILLED_TRAINING_MANIFEST_SCHEMA,
            "status": "fit_complete_runtime_disabled",
            "model_schema": _source_model().schema,
            "feature_schema": _source_model().feature_schema,
            "head_schema": _source_model().head_schema,
            "action_score_mode": _source_model().action_score_mode,
            "model_id": _source_model().model_id,
            "model_sha256": source_sha,
            "source": {
                "development_jsonl_sha256": "1" * 64,
                "development_decision_sha256": "2" * 64,
                "development_selector_receipt_sha256": "3" * 64,
                "development_pass_freeze_sha256": "4" * 64,
                "distillation_config_sha256": ATTEMPT08_DISTILLATION_CONFIG_SHA256,
                "candidate_model_sha256": "6" * 64,
                "development_package_manifest_sha256": "7" * 64,
                "development_source_zip_sha256": "8" * 64,
                "development_runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
                "development_runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
                "runtime_source_archive_sha256": runtime["archive"]["sha256"],
                "runtime_source_manifest_sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
                "runtime_source_closure_sha256": runtime["file_set"]["sha256"],
                "runtime_semantic_closure_sha256": runtime["semantic_closure"]["sha256"],
                "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
                "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
                "source_model_manifest_sha256": dependencies["source_model_manifest_sha256"],
                "source_native_manifest_sha256": dependencies["source_native_manifest_sha256"],
                "runtime_dependency_closure_sha256": dependencies["sha256"],
            },
            "diagnostics": {
                "states": 200,
                "rows": 1800,
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
                "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
                "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
            },
            "science_boundary": {
                "teacher_values_are_realized_match_ev": False,
                "audit50_fit_rows": 0,
                "threshold_sweep_performed": False,
                "top1_accuracy_is_acceptance_gate": False,
            },
        },
    )
    audit = tmp_path / "audit.json"
    _json(
        audit,
        {
            "schema": acceptance.ATTEMPT08_AUDIT50_DECISION_SCHEMA,
            "status": "go_write_separate_distillation_freeze_only",
            "decision": "go",
            "fit_performed": False,
            "threshold_selected": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "gates": [{"name": "audit", "passed": True}],
            "decision_contract": {
                "gate_evaluation_count": 1,
                "threshold_search_on_audit_performed": False,
                "fit_on_audit_performed": False,
                "audit_rows_used_for_fit": False,
                "realized_population_acceptance_still_required": True,
            },
        },
    )
    audit_receipt = tmp_path / "audit_receipt.json"
    _json(
        audit_receipt,
        {
            "schema": acceptance.ATTEMPT08_AUDIT50_SELECTOR_RECEIPT_SCHEMA,
            "status": "single_frozen_audit_gate_evaluation_complete",
            "decision_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
            "decision": "go",
            "go_action": "authorize_distillation_from_development200_only",
            "gate_evaluation_count": 1,
            "selector_executed": True,
            "audit_rows_used_for_fit": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    final = tmp_path / "final.pkl"
    freeze = tmp_path / "freeze.json"
    acceptance.freeze_attempt08_distilled_runtime(
        source_model_path=source,
        training_manifest_path=training,
        audit_decision_path=audit,
        audit_selector_receipt_path=audit_receipt,
        runtime_source_archive_path=runtime_archive,
        runtime_source_manifest_path=runtime_manifest,
        runtime_dependency_root=DEPENDENCY_ROOT,
        output_model_path=final,
        output_runtime_freeze_path=freeze,
    )
    return final, training, freeze, runtime_archive, runtime_manifest, runtime_tree


def _permit_unit_process_binding(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Stub only the process-location check after a dedicated bypass test."""

    calls: list[tuple[str, ...]] = []

    def _attest(*, extracted_root, manifest, module_names):
        del manifest
        names = tuple(module_names)
        calls.append(names)
        return Path(extracted_root).resolve(), {
            name: f"src/{name.replace('.', '/')}.py" for name in names
        }

    monkeypatch.setattr(
        distilled_runtime, "_validate_frozen_execution_module_paths", _attest
    )
    return calls


def test_direct_bound_loaders_reject_live_modules_against_decoy_tree(
    tmp_path: Path,
) -> None:
    model, training, freeze, _archive, runtime_manifest, runtime_tree = (
        _frozen_artifacts(tmp_path)
    )
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    binding = {
        "expected_sha256": model_sha,
        "freeze_manifest": freeze,
        "training_manifest_path": training,
        "runtime_source_manifest_path": runtime_manifest,
        "runtime_source_root": runtime_tree,
        "runtime_dependency_root": DEPENDENCY_ROOT,
    }
    with pytest.raises(ValueError, match="imported outside frozen tree"):
        load_hu_m43_joint_action_model(model, **binding)

    policy = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _BaselinePolicy(),
        action_value_model_path=model,
        safety_model_path=model,
        safety_probability_threshold=0.5,
        expected_joint_artifact_sha256=model_sha,
        freeze_manifest_path=freeze,
        training_manifest_path=training,
        runtime_source_manifest_path=runtime_manifest,
        runtime_source_root=runtime_tree,
        runtime_dependency_root=DEPENDENCY_ROOT,
    )
    assert policy.action_value_model is None
    assert policy.safety_model is None
    assert policy.runtime_binding_verified is False
    assert policy.model_load_failures == (
        "frozen_joint_artifact_binding_failed:ValueError",
    )


def test_attempt08_population_plan_and_runtime_preflight_are_exact(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = acceptance.load_and_validate_attempt08_population_plan(PLAN)
    assert plan["fixed_baseline_profile"] == "stage19_p0"
    assert plan["_freshness_counts"] == {
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
    }
    model, training, freeze, runtime_archive, runtime_manifest, runtime_tree = _frozen_artifacts(tmp_path)
    attestation_calls = _permit_unit_process_binding(monkeypatch)
    monkeypatch.setattr(acceptance, "_REPO_ROOT", runtime_tree)
    frozen_plan = runtime_tree / "configs/hu_joint_policy_m43_attempt08_population.json"
    preflight = acceptance.build_attempt08_population_preflight(
        model_path=model,
        training_manifest_path=training,
        runtime_freeze_path=freeze,
        population_plan_path=frozen_plan,
        runtime_source_archive_path=runtime_archive,
        runtime_source_manifest_path=runtime_manifest,
        runtime_source_root=runtime_tree,
        runtime_dependency_root=DEPENDENCY_ROOT,
    )
    assert preflight["status"] == "pass"
    assert preflight["runtime_binding_verified"] is True
    assert preflight["baseline_profile"] == "stage19_p0"
    assert preflight["fixed_safe_probability_threshold"] == 0.5
    assert preflight["fixed_fold_votes_min"] == 4
    assert preflight["teacher_calibration_locked_content_packaged"] is False
    assert attestation_calls
    raw = load_hu_m43_joint_action_model(model)
    assert raw.schema == _source_model().schema
    assert raw.runtime_binding_verified is False
    with pytest.raises(ValueError, match="frozen source manifest"):
        load_hu_m43_joint_action_model(
            model,
            expected_sha256=preflight["model_sha256"],
            freeze_manifest=freeze,
            training_manifest_path=training,
        )
    bound = load_hu_m43_joint_action_model(
        model,
        expected_sha256=preflight["model_sha256"],
        freeze_manifest=freeze,
        training_manifest_path=training,
        runtime_source_manifest_path=runtime_manifest,
        runtime_source_root=runtime_tree,
        runtime_dependency_root=DEPENDENCY_ROOT,
    )
    assert bound.safety_enabled is True
    assert bound.winner_frozen is True
    assert bound.runtime_binding_verified is True
    tamper = runtime_tree / "src" / "runtime_tamper.py"
    tamper.write_text("raise RuntimeError('tamper')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extracted runtime tree"):
        load_hu_m43_joint_action_model(
            model,
            expected_sha256=preflight["model_sha256"],
            freeze_manifest=freeze,
            training_manifest_path=training,
            runtime_source_manifest_path=runtime_manifest,
            runtime_source_root=runtime_tree,
            runtime_dependency_root=DEPENDENCY_ROOT,
        )


def test_attempt08_seed_registry_is_path_safe_and_content_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    escaped = json.loads(json.dumps(plan))
    escaped["freshness"]["excluded_schedule_registry_sources"][0] = "../../attacker.json"
    with pytest.raises(ValueError, match="registry source list"):
        acceptance.load_and_validate_attempt08_population_plan(escaped)

    registry_root = tmp_path / "registry"
    for relative in acceptance.ATTEMPT08_SEED_REGISTRY_SOURCES:
        source = ROOT / relative
        target = registry_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    first = registry_root / acceptance.ATTEMPT08_SEED_REGISTRY_SOURCES[0]
    changed = json.loads(first.read_text(encoding="utf-8"))
    changed["seed"] = acceptance.ATTEMPT08_POPULATION_SEED
    first.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(acceptance, "_REPO_ROOT", registry_root)
    with pytest.raises(ValueError, match="hashed registry|overlaps hashed registry"):
        acceptance.load_and_validate_attempt08_population_plan(plan)


def test_attempt08_seed_registry_digest_is_frozen_in_plan() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["freshness"]["excluded_schedule_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hashed seed registry snapshot"):
        acceptance.load_and_validate_attempt08_population_plan(plan)


def test_runtime_freeze_rejects_smoke_or_posthoc_training_contract(
    tmp_path: Path,
) -> None:
    _model, training, _freeze, *_rest = _frozen_artifacts(tmp_path)
    original = json.loads(training.read_text(encoding="utf-8"))
    for section, key, value in (
        ("fit_contract", "fit_mode", "smoke"),
        ("fit_contract", "effective_iterations", 24),
        ("fit_contract", "states", 199),
        ("source", "distillation_config_sha256", "0" * 64),
    ):
        payload = copy.deepcopy(original)
        payload[section][key] = value
        with pytest.raises(ValueError, match="semantic binding"):
            acceptance._validate_training_manifest(
                payload, expected_model_sha256=payload["model_sha256"]
            )


def test_attempt08_final_acceptance_uses_recomputed_population_only(
    tmp_path, monkeypatch
) -> None:
    model, training, freeze, runtime_archive, runtime_manifest, runtime_tree = _frozen_artifacts(tmp_path)
    attestation_calls = _permit_unit_process_binding(monkeypatch)
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    training_sha = hashlib.sha256(training.read_bytes()).hexdigest()
    freeze_sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    base_summary = {
        "schema": "hu_m4_t1_population_evaluation_v1",
        "evaluation_basis": "fresh_same_seed_candidate_vs_stage19_p0_counterfactual_v1",
        "records_output": None,
        "elapsed_seconds": None,
    }
    runtime = {
        "candidate_model_sha256": model_sha,
        "safety_model_sha256": model_sha,
        "model_schema": _source_model().schema,
        "artifact_schema": _source_model().artifact_schema,
        "feature_schema": _source_model().feature_schema,
        "head_schema": _source_model().head_schema,
        "model_id": _source_model().model_id,
        "action_score_mode": _source_model().action_score_mode,
        "baseline_profile": "stage19_p0",
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": freeze_sha,
        "training_manifest_sha256": training_sha,
        "runtime_source_manifest_sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
        "runtime_source_closure_sha256": json.loads(runtime_manifest.read_text())["file_set"]["sha256"],
        "runtime_semantic_closure_sha256": json.loads(runtime_manifest.read_text())["semantic_closure"]["sha256"],
        "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": validate_distilled_runtime_dependencies(DEPENDENCY_ROOT)["source_model_manifest_sha256"],
        "source_native_manifest_sha256": validate_distilled_runtime_dependencies(DEPENDENCY_ROOT)["source_native_manifest_sha256"],
        "runtime_dependency_closure_sha256": validate_distilled_runtime_dependencies(DEPENDENCY_ROOT)["sha256"],
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "safety_enabled": True,
        "safety_threshold": 0.5,
        "population_plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "sharded_evaluation": True,
        "shard_count": 20,
        "final_metrics_recomputed_from_merged_records": True,
        "seed_registry_sha256": acceptance.load_and_validate_attempt08_population_plan(PLAN)["_freshness_registry"]["sha256"],
    }
    evaluation = {**base_summary, "runtime_config": runtime}
    monkeypatch.setattr(
        acceptance, "summarize_hu_m4_population_records", lambda *args, **kwargs: dict(base_summary)
    )
    monkeypatch.setattr(
        acceptance,
        "evaluate_attempt02_population_gates",
        lambda _summary: [
            {"name": "fixed_realized_gates", "passed": True, "observed": True, "requirement": "all"}
        ],
    )
    records = [{"runtime_binding_verified": True}]
    merge = {
        "schema": acceptance.M4_POPULATION_MERGE_SCHEMA,
        "status": "complete_content_verified",
        "population_plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "seed": acceptance.ATTEMPT08_POPULATION_SEED,
        "seed_stride": acceptance.ATTEMPT08_POPULATION_SEED_STRIDE,
        "paired_seeds_per_opponent": 1000,
        "opponents": list(acceptance.ATTEMPT08_OPPONENTS),
        "merged_records": 8000,
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
        "shards": [{} for _ in range(20)],
    }
    status = acceptance.validate_attempt08_population_acceptance(
        evaluation=evaluation,
        records=records,
        population_plan=plan,
        merge_manifest=merge,
        model_path=model,
        training_manifest_path=training,
        runtime_freeze_path=freeze,
        runtime_source_archive_path=runtime_archive,
        runtime_source_manifest_path=runtime_manifest,
        runtime_source_root=runtime_tree,
        runtime_dependency_root=DEPENDENCY_ROOT,
        source_hashes={
            "population_plan": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            "population_plan_path": str(PLAN),
            "records": "1" * 64,
            "evaluation": "2" * 64,
            "merge_manifest": "3" * 64,
        },
    )
    assert status["status"] == "complete_go"
    assert status["promotion_eligible"] is True
    assert status["teacher_values_reported_as_realized_match_ev"] is False
    assert status["current_profile_mutated"] is False
    assert attestation_calls

    changed = dict(evaluation)
    changed["runtime_config"] = {**runtime, "baseline_profile": "stage18_p1"}
    rejected = acceptance.validate_attempt08_population_acceptance(
        evaluation=changed,
        records=records,
        population_plan=plan,
        merge_manifest=merge,
        model_path=model,
        training_manifest_path=training,
        runtime_freeze_path=freeze,
        runtime_source_archive_path=runtime_archive,
        runtime_source_manifest_path=runtime_manifest,
        runtime_source_root=runtime_tree,
        runtime_dependency_root=DEPENDENCY_ROOT,
        source_hashes={
            "population_plan": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            "population_plan_path": str(PLAN),
            "records": "1" * 64,
            "evaluation": "2" * 64,
            "merge_manifest": "3" * 64,
        },
    )
    assert rejected["status"] == "complete_no_go"
