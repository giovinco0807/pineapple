from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ofc_regular.hu_m43_attempt04_runtime import (
    M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA,
    M43_ATTEMPT04_LOCKED_BINDING_SCHEMA,
    M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA,
    M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA,
    build_attempt04_runtime_freeze,
    file_sha256,
    load_bound_attempt04_v6_model,
    self_digest,
    validate_attempt04_runtime_artifact_files,
    validate_attempt04_runtime_freeze,
    write_attempt04_runtime_freeze,
)
from ofc_regular.hu_m43_joint_model_v6 import (
    HU_M43_V6_ACTION_SCORE_MODE,
    HU_M43_V6_ARTIFACT_SCHEMA,
    HU_M43_V6_MODEL_SCHEMA,
    HU_M43_V6_PROPOSAL_SCHEMA,
    HU_M43_V6_SAFETY_FEATURE_DIM,
    HU_M43_V6_SAFETY_FEATURE_SCHEMA,
    HuM43JointModelV6,
)
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


def _fold(index: int) -> PairedDeltaRiskFoldEstimator:
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_ConstantRegression(1.0 + index),
        positive_gain_estimator=_ConstantRegression(0.8),
        downside_p95_estimator=_ConstantRegression(5.0),
        downside_p99_estimator=_ConstantRegression(10.0),
        downside_max_estimator=_ConstantRegression(15.0),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _model() -> HuM43JointModelV6:
    return HuM43JointModelV6(
        paired_fold_estimators=tuple(_fold(index) for index in range(5)),
        safety_estimator=ConstantProbabilityEstimator(0.8),
        safety_threshold=0.7,
        safety_enabled=True,
        tail_cushions=(1.0, 2.0, 3.0),
        model_id="attempt04-runtime-test",
    )


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifacts(tmp_path):
    model_path = tmp_path / "model.pkl"
    model_sha = _model().save(model_path)
    threshold_path = tmp_path / "threshold.json"
    threshold = {
        "schema": M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA,
        "status": "fixed_before_locked200_open",
        "model_sha256": model_sha,
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
        "safety_feature_dim": HU_M43_V6_SAFETY_FEATURE_DIM,
        "safety_enabled": True,
        "threshold_grid": [0.6, 0.7, 0.8],
        "selected_threshold": 0.7,
        "threshold_selection_method": "fixed_grid_on_sealed_calibration_v1",
        "threshold_reselection_after_freeze_allowed": False,
        "model_reselection_after_freeze_allowed": False,
        "feature_reselection_after_freeze_allowed": False,
        "locked200_opened": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    threshold["lock_sha256"] = self_digest(threshold, "lock_sha256")
    _write_json(threshold_path, threshold)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema": M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA,
        "status": "candidate_ready_for_runtime_freeze",
        "promotion_status": "candidate_ready_for_runtime_freeze",
        "model_sha256": model_sha,
        "model_schema": HU_M43_V6_MODEL_SCHEMA,
        "artifact_schema": HU_M43_V6_ARTIFACT_SCHEMA,
        "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
        "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
        "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
        "safety_feature_dim": HU_M43_V6_SAFETY_FEATURE_DIM,
        "selected_threshold": 0.7,
        "threshold_lock_file_sha256": file_sha256(threshold_path),
        "threshold_lock_sha256": threshold["lock_sha256"],
        "runtime_teacher_inputs": False,
        "threshold_reselection_after_freeze_allowed": False,
        "model_reselection_after_freeze_allowed": False,
        "feature_reselection_after_freeze_allowed": False,
        "locked200_opened": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    manifest["manifest_sha256"] = self_digest(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    plan_path = tmp_path / "attempt04-plan.json"
    population_path = tmp_path / "population-plan.json"
    _write_json(plan_path, {"schema": "attempt04-test-plan"})
    _write_json(population_path, {"schema": "attempt04-test-population"})
    locked = {
        "schema": M43_ATTEMPT04_LOCKED_BINDING_SCHEMA,
        "status": "sealed_unopened",
        "records": 200,
        "identity_sha256": "a" * 64,
        "ordered_shards": [
            {
                "path": "sealed/locked200.jsonl",
                "file_sha256": "b" * 64,
                "records": 200,
                "bytes": 1234,
            }
        ],
        "global_consumption_marker": (
            "outputs/attempt04/M43_ATTEMPT04_LOCKED200_CONSUMED.json"
        ),
        "content_opened": False,
        "model_evaluation_count": 0,
    }
    freeze = build_attempt04_runtime_freeze(
        model_path=model_path,
        final_training_manifest_path=manifest_path,
        threshold_lock_path=threshold_path,
        attempt04_plan_path=plan_path,
        population_plan_path=population_path,
        locked200_binding=locked,
    )
    return model_path, model_sha, manifest_path, threshold_path, freeze


def test_attempt04_runtime_freeze_builds_and_loads_exact_v6(tmp_path):
    model_path, model_sha, manifest_path, threshold_path, freeze = _artifacts(tmp_path)
    assert freeze["schema"] == M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA
    assert freeze["model"]["safety_threshold"] == 0.7
    assert freeze["locked200"]["records"] == 200
    loaded = load_bound_attempt04_v6_model(
        model_path,
        expected_sha256=model_sha,
        runtime_freeze=freeze,
        final_training_manifest_path=manifest_path,
        threshold_lock_path=threshold_path,
    )
    assert loaded.model_id == "attempt04-runtime-test"
    assert loaded.safety_threshold == 0.7


def test_attempt04_generic_dispatch_loads_actual_all_four_binding(tmp_path):
    model_path, model_sha, manifest_path, threshold_path, freeze = _artifacts(tmp_path)
    freeze_path = tmp_path / "runtime-freeze.json"
    _write_json(freeze_path, freeze)
    loaded = load_hu_m43_joint_action_model(
        model_path,
        expected_sha256=model_sha,
        freeze_manifest=freeze_path,
        training_manifest_path=manifest_path,
        threshold_lock_path=threshold_path,
    )
    assert isinstance(loaded, HuM43JointModelV6)
    assert loaded.model_id == "attempt04-runtime-test"
    with pytest.raises(ValueError, match="threshold lock together"):
        load_hu_m43_joint_action_model(
            model_path,
            expected_sha256=model_sha,
            freeze_manifest=freeze_path,
            training_manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["model"].__setitem__("safety_threshold", 0.8), "self digest"),
        (lambda value: value["model"].__setitem__("schema", "v5"), "model identity"),
        (lambda value: value["locked200"].__setitem__("records", 199), "locked200"),
        (lambda value: value.__setitem__("current_profile_mutated", True), "guard"),
    ],
)
def test_attempt04_runtime_freeze_tampering_fails_closed(tmp_path, mutator, message):
    *_unused, freeze = _artifacts(tmp_path)
    changed = deepcopy(freeze)
    mutator(changed)
    if message != "self digest":
        changed["freeze_sha256"] = self_digest(changed, "freeze_sha256")
    with pytest.raises(ValueError, match=message):
        validate_attempt04_runtime_freeze(changed)


def test_attempt04_loader_rejects_threshold_lock_bytes_changed(tmp_path):
    model_path, model_sha, manifest_path, threshold_path, freeze = _artifacts(tmp_path)
    threshold_path.write_text(threshold_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes disagree"):
        load_bound_attempt04_v6_model(
            model_path,
            expected_sha256=model_sha,
            runtime_freeze=freeze,
            final_training_manifest_path=manifest_path,
            threshold_lock_path=threshold_path,
        )


@pytest.mark.parametrize(
    ("source_name", "message"),
    [
        ("model", "model bytes disagree"),
        ("training", "training bytes disagree"),
        ("threshold", "threshold bytes disagree"),
        ("attempt04_plan", "attempt04_plan bytes disagree"),
        ("population_plan", "population_plan bytes disagree"),
    ],
)
def test_attempt04_runtime_source_hash_mismatch_fails_closed(
    tmp_path, source_name, message
):
    model_path, _model_sha, manifest_path, threshold_path, freeze = _artifacts(tmp_path)
    sources = {
        "model": model_path,
        "training": manifest_path,
        "threshold": threshold_path,
        "attempt04_plan": Path(freeze["attempt04_plan"]["path"]),
        "population_plan": Path(freeze["population_plan"]["path"]),
    }
    source = sources[source_name]
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match=message):
        validate_attempt04_runtime_artifact_files(
            freeze,
            model_path=model_path,
            final_training_manifest_path=manifest_path,
            threshold_lock_path=threshold_path,
            attempt04_plan_path=sources["attempt04_plan"],
            population_plan_path=sources["population_plan"],
        )


def test_attempt04_lifecycle_keeps_profile_registry_byte_identical():
    root = Path(__file__).resolve().parents[1]
    assert file_sha256(root / "src/ofc_regular/ai_profiles.py") == (
        "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    )


def test_attempt04_runtime_freeze_write_is_create_new(tmp_path):
    *_unused, freeze = _artifacts(tmp_path)
    destination = tmp_path / "runtime-freeze.json"
    write_attempt04_runtime_freeze(destination, freeze)
    with pytest.raises(FileExistsError):
        write_attempt04_runtime_freeze(destination, freeze)
