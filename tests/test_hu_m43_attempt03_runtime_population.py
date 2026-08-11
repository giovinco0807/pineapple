from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ofc_regular.evaluate_hu_m43_attempt03_locked_holdout as locked_evaluator
import ofc_regular.evaluate_hu_m4_population as population_evaluator
import ofc_regular.validate_hu_m43_attempt03_population as population_validator
from ofc_regular.assemble_hu_m43_attempt03_model import (
    build_attempt03_model_threshold_freeze,
)

from ofc_regular.hu_m43_attempt03_runtime import (
    M43_ATTEMPT03_LOCKED_MARKER_SCHEMA,
    M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT03_RUNTIME_FREEZE_SCHEMA,
    M43_ATTEMPT03_RUNTIME_FREEZE_STATUS,
    file_sha256,
    self_digest,
)
from ofc_regular.hu_m43_attempt03_training import (
    Attempt03TrainingConfig,
    M43_ATTEMPT03_FINAL_MANIFEST_SCHEMA,
    M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA,
    M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA,
    M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA,
    M43_ATTEMPT03_TRAINING_SOURCE_PATHS,
    M43_ATTEMPT03_THRESHOLD_REPORT_SCHEMA,
    M43_ATTEMPT03_WORKER_SOURCE_PATHS,
)
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m43_joint_model_v5 import (
    HU_M43_V5_ACTION_SCORE_MODE,
    HU_M43_V5_ARTIFACT_SCHEMA,
    HU_M43_V5_MODEL_SCHEMA,
    HU_M43_V5_PROPOSAL_SCHEMA,
    HuM43JointModelV5,
)
from ofc_regular.hu_m43_pilot_contract import canonical_manifest_sha256
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM
from ofc_regular.validate_hu_m43_attempt02_acceptance import FIXED_GATES
from ofc_regular.validate_hu_m43_attempt03_population import (
    ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA,
    ATTEMPT03_POPULATION_MANIFEST_SCHEMA,
    ATTEMPT03_POPULATION_RECEIPT_SCHEMA,
    ATTEMPT03_POPULATION_SEED,
    ATTEMPT03_POPULATION_SEED_STRIDE,
    build_attempt03_population_launch_preflight,
    evaluate_attempt03_population_gates,
    load_and_validate_attempt03_population_plan,
    validate_attempt03_locked_lifecycle,
    validate_attempt03_population_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "hu_joint_policy_m43_attempt03.json"
MODEL_FREEZE_PATH = (
    ROOT / "configs" / "hu_joint_policy_m43_attempt03_model_freeze.json"
)
POPULATION_PLAN_PATH = (
    ROOT / "configs" / "hu_joint_policy_m43_attempt03_population.json"
)


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


class _Stage18:
    def predict_sample(self, sample):
        return np.arange(len(sample["actions"]), dtype=np.float64)


def _fold(index: int) -> PairedDeltaRiskFoldEstimator:
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_ConstantRegression(1.0),
        positive_gain_estimator=_ConstantRegression(0.75),
        downside_p95_estimator=_ConstantRegression(5.0),
        downside_p99_estimator=_ConstantRegression(10.0),
        downside_max_estimator=_ConstantRegression(20.0),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _model(*, threshold: float = 0.5) -> HuM43JointModelV5:
    return HuM43JointModelV5(
        paired_fold_estimators=tuple(_fold(index) for index in range(5)),
        stage18_scorer=_Stage18(),
        meta_ranker=_ConstantRegression(0.25),
        safety_estimator=ConstantProbabilityEstimator(0.8),
        safety_threshold=threshold,
        safety_enabled=True,
        model_id="attempt03-v5-runtime-fixture",
    )


def _precalibration_receipt() -> dict:
    payload = {
        "schema": M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA,
        "status": "go_precalibration",
        "promotion_status": "eligible_to_open_sealed_calibration",
        "candidate_model_sha256": "1" * 64,
        "fit_manifest_sha256": "2" * 64,
        "precalibration_consumption_marker_sha256": "3" * 64,
        "precalibration_report": {"status": "go"},
        "sealed_calibration_open_allowed": True,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "algorithm_or_gate_change_after_result_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    payload["receipt_sha256"] = canonical_manifest_sha256(payload)
    return payload


def _final_manifest(
    model_sha: str,
    receipt_sha: str,
    threshold: float,
    training_freeze_sha: str,
) -> dict:
    payload = {
        "schema": M43_ATTEMPT03_FINAL_MANIFEST_SCHEMA,
        "status": "candidate_ready_for_freeze",
        "promotion_status": "candidate_ready_for_freeze",
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
        "model_sha256": model_sha,
        "model_freeze_file_sha256": file_sha256(MODEL_FREEZE_PATH),
        "training_freeze_file_sha256": training_freeze_sha,
        "precalibration_receipt_sha256": receipt_sha,
        "safety_fit": {"status": "fit_threshold_unselected"},
        "threshold_selection": {
            "schema": M43_ATTEMPT03_THRESHOLD_REPORT_SCHEMA,
            "status": "go",
            "selected_threshold": threshold,
            "safety_enabled": True,
            "threshold_adaptation_after_selection": False,
            "inherited_locked_opened": False,
            "runtime_teacher_inputs": False,
        },
        "locked_holdout": {
            "status": "not_evaluated_pre_freeze",
            "opened": False,
        },
        "runtime_teacher_inputs": False,
        "teacher_value_status": "diagnostic_only",
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    payload["manifest_sha256"] = canonical_manifest_sha256(payload)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _training_freeze(tmp_path: Path) -> Path:
    files = {
        relative: file_sha256(ROOT / relative)
        for relative in M43_ATTEMPT03_TRAINING_SOURCE_PATHS
    }
    workers = {
        relative: files[relative]
        for relative in M43_ATTEMPT03_WORKER_SOURCE_PATHS
    }
    payload = {
        "schema": M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA,
        "milestone": "M4.3-attempt03",
        "status": (
            "frozen_after_one_structural_train_fit_canary_before_"
            "row_valued_design_or_any_holdout_open"
        ),
        "frozen_at": "2026-07-14T00:00:00Z",
        "decision_boundary": {
            "cloud_teacher_rows_may_exist": True,
            "attempt03_fit_rows_received_locally": 1,
            "attempt03_fit_rows_structurally_inspected": 1,
            "attempt03_fit_jsonl_content_parse_count": 1,
            "attempt03_fit_row_valued_labels_or_metrics_used_for_design": 0,
            "precalibration_rows_opened": 0,
            "sealed_calibration_rows_opened": 0,
            "inherited_locked_rows_opened": 0,
            "source_or_config_selected_from_row_values": False,
        },
        "parent_model_freeze": {
            "path": str(MODEL_FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "file_sha256": file_sha256(MODEL_FREEZE_PATH),
        },
        "training_config": {
            "schema": M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA,
            "canonical_sha256": canonical_manifest_sha256(
                Attempt03TrainingConfig().to_manifest()
            ),
        },
        "executable_sources": {
            "files": files,
            "files_sha256": canonical_manifest_sha256(files),
            "worker_paths": list(M43_ATTEMPT03_WORKER_SOURCE_PATHS),
            "worker_files_sha256": canonical_manifest_sha256(workers),
        },
        "lifecycle": {
            "teacher_row_values_in_freeze": False,
            "holdout_path_in_freeze": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
        },
    }
    payload["freeze_sha256"] = canonical_manifest_sha256(payload)
    path = tmp_path / "training-freeze.json"
    _write_json(path, payload)
    return path


def _runtime_artifacts(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    _model().save(model_path)
    model_sha = file_sha256(model_path)
    precal = _precalibration_receipt()
    precal_path = tmp_path / "precal.json"
    _write_json(precal_path, precal)
    training_freeze_path = _training_freeze(tmp_path)
    final = _final_manifest(
        model_sha,
        precal["receipt_sha256"],
        0.5,
        file_sha256(training_freeze_path),
    )
    final_path = tmp_path / "final.json"
    _write_json(final_path, final)
    freeze = build_attempt03_model_threshold_freeze(
        model_path=model_path,
        training_manifest_path=final_path,
        model_freeze_path=MODEL_FREEZE_PATH,
        training_freeze_path=training_freeze_path,
        attempt03_plan_path=PLAN_PATH,
        repo_root=ROOT,
    )
    freeze_path = tmp_path / "runtime-freeze.json"
    _write_json(freeze_path, freeze)
    return model_path, model_sha, precal_path, final_path, freeze_path, freeze


def test_attempt03_population_plan_is_fresh_fixed_and_opt_in():
    plan = load_and_validate_attempt03_population_plan(
        POPULATION_PLAN_PATH, repo_root=ROOT
    )
    assert plan["policy_attempt"] == "attempt03_v5"
    assert plan["opponents"] == [
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "random_exact_final",
    ]
    assert plan["seed"] == ATTEMPT03_POPULATION_SEED
    assert plan["seed_stride"] == ATTEMPT03_POPULATION_SEED_STRIDE
    assert plan["minimum_valid_overrides"] == 300
    assert plan["fixed_acceptance_gates"] == FIXED_GATES
    assert plan["activation_guards"] == {
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
        "threshold_changed_after_lock": False,
    }
    assert plan["_freshness_counts"]["population"] == 1000
    assert plan["_freshness_counts"]["teacher"] >= 700


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fixed_baseline_profile", "current", "identity"),
        ("seed", 9_106_071_901, "seed overlaps|seed/power"),
        ("minimum_valid_overrides", 299, "sample-size"),
    ),
)
def test_attempt03_population_plan_mutations_fail_closed(field, value, message):
    plan = json.loads(POPULATION_PLAN_PATH.read_text(encoding="utf-8"))
    plan[field] = value
    with pytest.raises(ValueError, match=message):
        load_and_validate_attempt03_population_plan(plan)


def test_v5_dispatch_loader_verifies_runtime_freeze_and_final_manifest(tmp_path):
    model_path, model_sha, _precal, final_path, freeze_path, _freeze = (
        _runtime_artifacts(tmp_path)
    )
    loaded = load_hu_m43_joint_action_model(
        model_path,
        expected_sha256=model_sha,
        freeze_manifest=freeze_path,
        training_manifest_path=final_path,
    )
    assert isinstance(loaded, HuM43JointModelV5)
    assert loaded.safety_enabled is True
    assert loaded.safety_threshold == 0.5
    assert loaded.action_score_mode == HU_M43_V5_ACTION_SCORE_MODE

    with pytest.raises(ValueError, match="requires expected SHA"):
        load_hu_m43_joint_action_model(
            model_path,
            expected_sha256=model_sha,
        )
    changed = json.loads(final_path.read_text(encoding="utf-8"))
    changed["threshold_selection"]["selected_threshold"] = 0.6
    _write_json(final_path, changed)
    with pytest.raises(ValueError, match="self digest|bytes changed"):
        load_hu_m43_joint_action_model(
            model_path,
            expected_sha256=model_sha,
            freeze_manifest=freeze_path,
            training_manifest_path=final_path,
        )


def test_population_cli_rejects_unbound_v5_before_playing_hands(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "placeholder.pkl"
    model_path.write_bytes(b"not-opened-by-fixture")
    monkeypatch.setattr(population_evaluator, "load_model_bundle", lambda *_a, **_k: {})
    monkeypatch.setattr(
        population_evaluator,
        "load_hu_m43_joint_action_model",
        lambda *_a, **_k: SimpleNamespace(schema=HU_M43_V5_MODEL_SCHEMA),
    )
    with pytest.raises(ValueError, match="requires the expected model SHA"):
        population_evaluator.main(
            ["--model", str(model_path), "--paired-seeds", "1"]
        )


def test_population_cli_accepts_explicitly_bound_v5_without_current(
    tmp_path, monkeypatch
):
    model_path, model_sha, _precal, final_path, freeze_path, _freeze = (
        _runtime_artifacts(tmp_path)
    )
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(population_evaluator, "load_model_bundle", lambda *_a, **_k: {})
    monkeypatch.setattr(
        population_evaluator,
        "evaluate_hu_m4_population",
        lambda **_kwargs: {"schema": "bound-v5-fixture"},
    )
    assert population_evaluator.main(
        [
            "--model", str(model_path),
            "--expected-model-sha256", model_sha,
            "--freeze-manifest", str(freeze_path),
            "--training-manifest", str(final_path),
            "--paired-seeds", "1",
            "--output", str(output),
        ]
    ) == 0
    runtime = json.loads(output.read_text(encoding="utf-8"))["runtime_config"]
    assert runtime["model_schema"] == HU_M43_V5_MODEL_SCHEMA
    assert runtime["action_score_mode"] == HU_M43_V5_ACTION_SCORE_MODE
    assert runtime["runtime_binding_verified"] is True
    assert runtime["candidate_model_sha256"] == model_sha
    assert runtime["safety_model_sha256"] == model_sha
    assert runtime["current_profile_used"] is False
    assert runtime["promotion_artifact_contract"] is True
    assert runtime["safety_enabled"] is True


def test_runtime_freeze_binds_every_future_population_input_without_locked_read(
    tmp_path,
):
    model_path, model_sha, precal_path, final_path, _freeze_path, freeze = (
        _runtime_artifacts(tmp_path)
    )
    assert freeze["schema"] == M43_ATTEMPT03_RUNTIME_FREEZE_SCHEMA
    assert freeze["status"] == M43_ATTEMPT03_RUNTIME_FREEZE_STATUS
    assert freeze["model"]["file_sha256"] == model_sha
    assert freeze["training_manifest"]["file_sha256"] == file_sha256(final_path)
    assert freeze["executable_model_freeze"]["file_sha256"] == file_sha256(
        MODEL_FREEZE_PATH
    )
    assert freeze["training_pipeline_freeze"]["file_sha256"] == file_sha256(
        freeze["training_pipeline_freeze"]["path"]
    )
    assert freeze["attempt03_plan"]["file_sha256"] == file_sha256(PLAN_PATH)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    precal = json.loads(precal_path.read_text(encoding="utf-8"))
    assert final["precalibration_receipt_sha256"] == precal["receipt_sha256"]
    assert freeze["inherited_locked"]["content_opened"] is False
    assert freeze["current_profile_mutated"] is False
    assert model_path.is_file()


def test_locked_lifecycle_hash_only_contract_and_unsafe_flag_rejection(tmp_path):
    _model_path, _sha, precal_path, _final, freeze_path, freeze = _runtime_artifacts(
        tmp_path
    )
    training_freeze_path = Path(freeze["training_pipeline_freeze"]["path"])
    marker_path = tmp_path / "M43_LOCKED_CONSUMED.json"
    population_sha = file_sha256(POPULATION_PLAN_PATH)
    marker = {
        "schema": M43_ATTEMPT03_LOCKED_MARKER_SCHEMA,
        "status": "claimed_before_inherited_locked_content_read",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": file_sha256(freeze_path),
        "population_plan_file_sha256": population_sha,
        "model_sha256": freeze["model"]["file_sha256"],
        "model_freeze_file_sha256": file_sha256(MODEL_FREEZE_PATH),
        "training_freeze_file_sha256": file_sha256(training_freeze_path),
        "precalibration_receipt_file_sha256": file_sha256(precal_path),
        "locked_identity_sha256": freeze["inherited_locked"]["identity_sha256"],
        "evaluation_pass_count": 1,
        "claim_is_consuming_even_on_crash": True,
    }
    marker["marker_sha256"] = canonical_manifest_sha256(marker)
    _write_json(marker_path, marker)
    receipt = {
        "schema": M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": file_sha256(freeze_path),
        "population_plan_file_sha256": population_sha,
        "model_sha256": freeze["model"]["file_sha256"],
        "model_freeze_file_sha256": file_sha256(MODEL_FREEZE_PATH),
        "training_freeze_file_sha256": file_sha256(training_freeze_path),
        "precalibration_receipt_file_sha256": file_sha256(precal_path),
        "frozen_threshold": freeze["model"]["safety_threshold"],
        "locked_identity_sha256": freeze["inherited_locked"]["identity_sha256"],
        "consumption_marker_resolved_path": str(marker_path.resolve()),
        "consumption_marker_file_sha256": file_sha256(marker_path),
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "evaluation_pass_count": 1,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": 300,
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
    }
    receipt["receipt_sha256"] = canonical_manifest_sha256(receipt)
    validate_attempt03_locked_lifecycle(
        marker=marker,
        receipt=receipt,
        runtime_freeze=freeze,
        runtime_freeze_file_sha256=file_sha256(freeze_path),
        population_plan_file_sha256=population_sha,
        model_freeze_file_sha256=file_sha256(MODEL_FREEZE_PATH),
        training_freeze_file_sha256=file_sha256(training_freeze_path),
        precalibration_receipt_file_sha256=file_sha256(precal_path),
        marker_path=marker_path,
    )
    unsafe = deepcopy(receipt)
    unsafe["threshold_reselection_performed"] = True
    unsafe["receipt_sha256"] = self_digest(unsafe, "receipt_sha256")
    with pytest.raises(ValueError, match="unsafe flag"):
        validate_attempt03_locked_lifecycle(
            marker=marker,
            receipt=unsafe,
            runtime_freeze=freeze,
            runtime_freeze_file_sha256=file_sha256(freeze_path),
            population_plan_file_sha256=population_sha,
            model_freeze_file_sha256=file_sha256(MODEL_FREEZE_PATH),
            training_freeze_file_sha256=file_sha256(training_freeze_path),
            precalibration_receipt_file_sha256=file_sha256(precal_path),
            marker_path=marker_path,
        )


def test_population_preflight_loads_only_hash_receipts_and_bound_v5(tmp_path):
    model_path, model_sha, precal_path, final_path, freeze_path, freeze = (
        _runtime_artifacts(tmp_path)
    )
    marker_path = tmp_path / "canonical" / "M43_LOCKED_CONSUMED.json"
    freeze["inherited_locked"]["global_consumption_marker"] = str(
        marker_path.resolve()
    )
    freeze["freeze_sha256"] = self_digest(freeze, "freeze_sha256")
    _write_json(freeze_path, freeze)
    marker = {
        "schema": M43_ATTEMPT03_LOCKED_MARKER_SCHEMA,
        "status": "claimed_before_inherited_locked_content_read",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": file_sha256(freeze_path),
        "population_plan_file_sha256": file_sha256(POPULATION_PLAN_PATH),
        "model_sha256": model_sha,
        "model_freeze_file_sha256": file_sha256(MODEL_FREEZE_PATH),
        "training_freeze_file_sha256": file_sha256(
            freeze["training_pipeline_freeze"]["path"]
        ),
        "precalibration_receipt_file_sha256": file_sha256(precal_path),
        "locked_identity_sha256": freeze["inherited_locked"]["identity_sha256"],
        "evaluation_pass_count": 1,
        "claim_is_consuming_even_on_crash": True,
    }
    marker["marker_sha256"] = canonical_manifest_sha256(marker)
    marker_path.parent.mkdir(parents=True)
    _write_json(marker_path, marker)
    receipt = {
        "schema": M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": file_sha256(freeze_path),
        "population_plan_file_sha256": file_sha256(POPULATION_PLAN_PATH),
        "model_sha256": model_sha,
        "model_freeze_file_sha256": file_sha256(MODEL_FREEZE_PATH),
        "training_freeze_file_sha256": file_sha256(
            freeze["training_pipeline_freeze"]["path"]
        ),
        "precalibration_receipt_file_sha256": file_sha256(precal_path),
        "frozen_threshold": 0.5,
        "locked_identity_sha256": freeze["inherited_locked"]["identity_sha256"],
        "consumption_marker_resolved_path": str(marker_path.resolve()),
        "consumption_marker_file_sha256": file_sha256(marker_path),
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "evaluation_pass_count": 1,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": 300,
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
    }
    receipt["receipt_sha256"] = canonical_manifest_sha256(receipt)
    receipt_path = tmp_path / "locked-receipt.json"
    _write_json(receipt_path, receipt)

    result = build_attempt03_population_launch_preflight(
        model_path=model_path,
        final_training_manifest_path=final_path,
        runtime_freeze_path=freeze_path,
        training_freeze_path=freeze["training_pipeline_freeze"]["path"],
        precalibration_receipt_path=precal_path,
        model_freeze_path=MODEL_FREEZE_PATH,
        attempt03_plan_path=PLAN_PATH,
        locked_receipt_path=receipt_path,
        consumption_marker_path=marker_path,
        population_plan_path=POPULATION_PLAN_PATH,
    )
    assert result["schema"] == ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA
    assert result["status"] == "pass"
    assert result["model_schema"] == HU_M43_V5_MODEL_SCHEMA
    assert result["model_sha256"] == model_sha
    assert result["action_score_mode"] == HU_M43_V5_ACTION_SCORE_MODE
    assert result["canonical_global_marker_verified"] is True
    assert result["teacher_calibration_locked_content_packaged"] is False
    assert result["current_profile_mutated"] is False
    assert result["no_runtime_activation"] is True


def test_one_shot_locked_evaluator_claims_marker_before_content_touch(
    tmp_path, monkeypatch
):
    model_path, model_sha, precal_path, final_path, freeze_path, freeze = (
        _runtime_artifacts(tmp_path)
    )
    locked_path = tmp_path / "locked.jsonl"
    locked_path.write_text("sealed-fixture\n", encoding="utf-8")
    marker_path = tmp_path / "global" / "M43_LOCKED_CONSUMED.json"
    receipt_path = tmp_path / "locked-receipt.json"
    freeze["inherited_locked"].update(
        {
            "path": str(locked_path.resolve()),
            "records": 40,
            "file_sha256": file_sha256(locked_path),
            "identity_sha256": "a" * 64,
            "global_consumption_marker": str(marker_path.resolve()),
            "content_opened": False,
            "model_evaluation_count": 0,
        }
    )
    freeze["freeze_sha256"] = self_digest(freeze, "freeze_sha256")
    _write_json(freeze_path, freeze)
    monkeypatch.setattr(locked_evaluator, "_find_consumption_markers", lambda *_: [])

    def audit_after_claim(path, *, expected):
        assert path == locked_path.resolve()
        assert marker_path.is_file()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["status"] == "claimed_before_inherited_locked_content_read"
        assert marker["model_sha256"] == model_sha
        return [object()], {
            "records": 40,
            "file_sha256": expected["file_sha256"],
            "identity_sha256": expected["identity_sha256"],
        }

    monkeypatch.setattr(locked_evaluator, "_audit_and_read_locked", audit_after_claim)
    monkeypatch.setattr(
        locked_evaluator,
        "_evaluate_v5_fixed_threshold",
        lambda _model, _samples: {
            "threshold_search_performed": False,
            "teacher_value_status": "diagnostic_only_not_realized_match_ev_not_runtime_gate",
        },
    )
    receipt = locked_evaluator.evaluate_attempt03_locked_holdout_once(
        model_path=model_path,
        final_training_manifest_path=final_path,
        runtime_freeze_path=freeze_path,
        training_freeze_path=freeze["training_pipeline_freeze"]["path"],
        precalibration_receipt_path=precal_path,
        model_freeze_path=MODEL_FREEZE_PATH,
        attempt03_plan_path=PLAN_PATH,
        population_plan_path=POPULATION_PLAN_PATH,
        repo_root=ROOT,
        receipt_path=receipt_path,
    )
    assert receipt["status"] == "evaluated_once_diagnostic_only_no_activation"
    assert receipt["population_plan_file_sha256"] == file_sha256(
        POPULATION_PLAN_PATH
    )
    assert receipt["threshold_search_performed"] is False
    assert receipt["current_profile_resolved"] is False
    assert receipt["runtime_policy_activated"] is False
    assert marker_path.is_file() and receipt_path.is_file()

    with pytest.raises(FileExistsError, match="already consumed"):
        locked_evaluator.evaluate_attempt03_locked_holdout_once(
            model_path=model_path,
            final_training_manifest_path=final_path,
            runtime_freeze_path=freeze_path,
            training_freeze_path=freeze["training_pipeline_freeze"]["path"],
            precalibration_receipt_path=precal_path,
            model_freeze_path=MODEL_FREEZE_PATH,
            attempt03_plan_path=PLAN_PATH,
            population_plan_path=POPULATION_PLAN_PATH,
            repo_root=ROOT,
            receipt_path=tmp_path / "second-receipt.json",
        )


def test_final_acceptance_binds_spot_merge_receipt_and_runtime(monkeypatch):
    plan = json.loads(POPULATION_PLAN_PATH.read_text(encoding="utf-8"))
    plan_sha = file_sha256(POPULATION_PLAN_PATH)
    digests = {
        "records": "1" * 64,
        "evaluation": "2" * 64,
        "merge_manifest": "3" * 64,
        "run_manifest": "4" * 64,
    }
    preflight = {
        "schema": ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "model_sha256": "5" * 64,
        "model_id": "attempt03-v5-final-acceptance-fixture",
        "frozen_threshold": 0.5,
        "final_training_manifest_file_sha256": "6" * 64,
        "training_freeze_file_sha256": "7" * 64,
        "runtime_freeze_file_sha256": "8" * 64,
        "locked_receipt_file_sha256": "9" * 64,
        "consumption_marker_file_sha256": "a" * 64,
        "population_plan_file_sha256": plan_sha,
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "canonical_global_marker_verified": True,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    runtime = {
        "candidate_model_sha256": preflight["model_sha256"],
        "safety_model_sha256": preflight["model_sha256"],
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "model_id": preflight["model_id"],
        "action_score_mode": HU_M43_V5_ACTION_SCORE_MODE,
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": preflight["runtime_freeze_file_sha256"],
        "training_manifest_sha256": preflight[
            "final_training_manifest_file_sha256"
        ],
        "population_plan_sha256": plan_sha,
        "sharded_evaluation": True,
        "shard_count": 20,
        "final_metrics_recomputed_from_merged_records": True,
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "safety_enabled": True,
        "safety_threshold": 0.5,
    }
    summary = {
        "schema": "hu_m4_t1_population_evaluation_v1",
        "opponents": list(population_validator.EXPECTED_OPPONENTS),
        "paired_seat_swap": True,
        "paired_seeds_per_opponent": 1000,
        "seed": plan["seed"],
        "seed_stride": plan["seed_stride"],
        "population": {"all_seats": {"overrides": 350}},
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "runtime_config": runtime,
    }
    monkeypatch.setattr(
        population_validator,
        "summarize_hu_m4_population_records",
        lambda *_args, **_kwargs: deepcopy(summary),
    )
    monkeypatch.setattr(
        population_validator,
        "evaluate_attempt03_population_gates",
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
                "seed": plan["seed"] + offset * plan["seed_stride"],
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
        "population_plan_sha256": plan_sha,
        "merged_records_sha256": digests["records"],
        "evaluation_sha256": digests["evaluation"],
        "paired_seeds_per_opponent": 1000,
        "opponents": list(population_validator.EXPECTED_OPPONENTS),
        "merged_records": 8000,
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
        "shards": merge_shards,
    }
    run = {
        "schema": ATTEMPT03_POPULATION_MANIFEST_SCHEMA,
        "population_plan": {
            "sha256": plan_sha,
            "paired_seeds": 1000,
            "seed": plan["seed"],
            "seed_stride": plan["seed_stride"],
            "shards": 20,
            "paired_seeds_per_shard": 50,
        },
        "shards": {"count": 20, "sha256": "b" * 64},
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
            "training_freeze_sha256": preflight[
                "training_freeze_file_sha256"
            ],
            "freeze_manifest_sha256": preflight[
                "runtime_freeze_file_sha256"
            ],
            "locked_receipt_sha256": preflight[
                "locked_receipt_file_sha256"
            ],
            "consumption_marker_sha256": preflight[
                "consumption_marker_file_sha256"
            ],
            "model_schema": HU_M43_V5_MODEL_SCHEMA,
            "model_id": preflight["model_id"],
            "action_score_mode": HU_M43_V5_ACTION_SCORE_MODE,
            "runtime_teacher_inputs": False,
            "current_profile_used": False,
            "frozen_threshold": 0.5,
        },
        "launch_preflight": preflight,
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
        "schema": ATTEMPT03_POPULATION_RECEIPT_SCHEMA,
        "status": "verified_and_merged",
        "run_manifest_sha256": digests["run_manifest"],
        "population_plan_sha256": plan_sha,
        "model_sha256": preflight["model_sha256"],
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V5_ACTION_SCORE_MODE,
        "evaluation_sha256": digests["evaluation"],
        "records_sha256": digests["records"],
        "merge_manifest_sha256": digests["merge_manifest"],
        "valid_overrides": 350,
        "paired_seeds_per_opponent": 1000,
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "shards": receipt_shards,
        "teacher_calibration_locked_content_received": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    config, status = validate_attempt03_population_acceptance(
        evaluation=summary,
        records=[{"runtime_binding_verified": True}],
        population_plan=plan,
        merge_manifest=merge,
        spot_receipt=receipt,
        run_manifest=run,
        lifecycle_preflight=preflight,
        source_hashes={"population_plan": plan_sha, **digests},
    )
    assert status["status"] == "complete_go"
    assert status["passed_gates"] == status["total_gates"] == 4
    assert config["current_profile_mutated"] is False
    assert config["runtime_policy_activated"] is False


def test_attempt03_gate_adapter_rejects_any_population_substitution():
    with pytest.raises(ValueError, match="opponent order"):
        evaluate_attempt03_population_gates({"opponents": ["current"]})


def test_preflight_schema_is_attempt03_specific():
    assert ATTEMPT03_POPULATION_PREFLIGHT_SCHEMA == (
        "hu_m43_attempt03_population_launch_preflight_v1"
    )
    assert hashlib.sha256(POPULATION_PLAN_PATH.read_bytes()).hexdigest()
