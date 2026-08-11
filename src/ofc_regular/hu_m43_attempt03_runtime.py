"""Fail-closed loader contract for the opt-in Attempt03 v5 candidate.

The canonical model/threshold freeze is produced by
``assemble_hu_m43_attempt03_model freeze-candidate`` after calibration Go and
before inherited-lock access.  This module validates and consumes that exact
freeze; it does not introduce another freeze format and never resolves or
changes ``current``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .assemble_hu_m43_attempt03_model import (
    M43_ATTEMPT03_MODEL_THRESHOLD_FREEZE_SCHEMA,
)
from .hu_m43_attempt03_training import (
    M43_ATTEMPT03_FINAL_MANIFEST_SCHEMA,
    M43_ATTEMPT03_FIXED_THRESHOLDS,
    M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA,
    M43_ATTEMPT03_THRESHOLD_REPORT_SCHEMA,
)
from .hu_m43_joint_model_v5 import (
    HU_M43_V5_ACTION_SCORE_MODE,
    HU_M43_V5_ARTIFACT_SCHEMA,
    HU_M43_V5_MODEL_SCHEMA,
    HU_M43_V5_PROPOSAL_SCHEMA,
    HuM43JointModelV5,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256


M43_ATTEMPT03_RUNTIME_FREEZE_SCHEMA = (
    M43_ATTEMPT03_MODEL_THRESHOLD_FREEZE_SCHEMA
)
M43_ATTEMPT03_RUNTIME_FREEZE_STATUS = "frozen_before_inherited_locked_open"
M43_ATTEMPT03_LOCKED_MARKER_SCHEMA = (
    "hu_m43_attempt03_locked_holdout_consumption_marker_v1"
)
M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA = (
    "hu_m43_attempt03_locked_holdout_receipt_v1"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_mapping(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def self_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return canonical_manifest_sha256(unsigned)


def validate_attempt03_final_training_manifest(
    manifest: Mapping[str, Any], *, model_sha256: str
) -> float:
    expected_model_sha = _sha256_text(model_sha256, "model_sha256")
    if manifest.get("schema") != M43_ATTEMPT03_FINAL_MANIFEST_SCHEMA:
        raise ValueError("Attempt03 final training manifest schema mismatch")
    if manifest.get("manifest_sha256") != self_digest(
        manifest, "manifest_sha256"
    ):
        raise ValueError("Attempt03 final training manifest self digest mismatch")
    if (
        manifest.get("status") != "candidate_ready_for_freeze"
        or manifest.get("promotion_status") != "candidate_ready_for_freeze"
        or manifest.get("model_schema") != HU_M43_V5_MODEL_SCHEMA
        or manifest.get("proposal_schema") != HU_M43_V5_PROPOSAL_SCHEMA
        or manifest.get("model_sha256") != expected_model_sha
    ):
        raise ValueError("Attempt03 final training manifest model identity mismatch")
    threshold = _mapping(
        manifest.get("threshold_selection"), "threshold_selection"
    )
    if (
        threshold.get("schema") != M43_ATTEMPT03_THRESHOLD_REPORT_SCHEMA
        or threshold.get("status") != "go"
        or threshold.get("safety_enabled") is not True
        or threshold.get("threshold_adaptation_after_selection") is not False
        or threshold.get("inherited_locked_opened") is not False
        or threshold.get("runtime_teacher_inputs") is not False
    ):
        raise ValueError("Attempt03 threshold selection is not a frozen Go")
    selected = _finite_probability(
        threshold.get("selected_threshold"), "selected_threshold"
    )
    if not any(
        math.isclose(selected, float(value), rel_tol=0.0, abs_tol=1.0e-12)
        for value in M43_ATTEMPT03_FIXED_THRESHOLDS
    ):
        raise ValueError("Attempt03 selected threshold is outside the frozen grid")
    if manifest.get("locked_holdout") != {
        "status": "not_evaluated_pre_freeze",
        "opened": False,
    }:
        raise ValueError("Attempt03 final manifest opened the inherited lock")
    if any(
        manifest.get(field) is not False
        for field in (
            "runtime_teacher_inputs",
            "current_profile_mutated",
            "runtime_policy_activated",
            "full_replacement",
        )
    ):
        raise ValueError("Attempt03 final manifest violates runtime guards")
    _sha256_text(
        manifest.get("training_freeze_file_sha256"),
        "training_freeze_file_sha256",
    )
    return selected


def validate_attempt03_precalibration_receipt(
    receipt: Mapping[str, Any]
) -> None:
    if receipt.get("schema") != M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA:
        raise ValueError("Attempt03 pre-calibration receipt schema mismatch")
    if receipt.get("receipt_sha256") != self_digest(receipt, "receipt_sha256"):
        raise ValueError("Attempt03 pre-calibration receipt self digest mismatch")
    if (
        receipt.get("status") != "go_precalibration"
        or receipt.get("promotion_status")
        != "eligible_to_open_sealed_calibration"
        or receipt.get("sealed_calibration_open_allowed") is not True
        or receipt.get("sealed_calibration_opened") is not False
        or receipt.get("inherited_locked_opened") is not False
        or receipt.get("algorithm_or_gate_change_after_result_allowed") is not False
        or receipt.get("current_profile_mutated") is not False
        or receipt.get("runtime_policy_activated") is not False
        or receipt.get("full_replacement") is not False
    ):
        raise ValueError("Attempt03 pre-calibration receipt is not an unopened Go")
    for field in (
        "candidate_model_sha256",
        "fit_manifest_sha256",
        "precalibration_consumption_marker_sha256",
    ):
        _sha256_text(receipt.get(field), field)


def validate_attempt03_runtime_freeze(
    freeze: Mapping[str, Any],
    *,
    final_training_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate the canonical post-calibration model/threshold freeze."""

    if (
        freeze.get("schema") != M43_ATTEMPT03_RUNTIME_FREEZE_SCHEMA
        or freeze.get("status") != M43_ATTEMPT03_RUNTIME_FREEZE_STATUS
    ):
        raise ValueError("Attempt03 model-threshold freeze identity mismatch")
    if freeze.get("freeze_sha256") != self_digest(freeze, "freeze_sha256"):
        raise ValueError("Attempt03 model-threshold freeze self digest mismatch")
    model = _mapping(freeze.get("model"), "model")
    training = _mapping(freeze.get("training_manifest"), "training_manifest")
    executable = _mapping(
        freeze.get("executable_model_freeze"), "executable_model_freeze"
    )
    training_pipeline = _mapping(
        freeze.get("training_pipeline_freeze"), "training_pipeline_freeze"
    )
    plan = _mapping(freeze.get("attempt03_plan"), "attempt03_plan")
    locked = _mapping(freeze.get("inherited_locked"), "inherited_locked")
    if (
        model.get("schema") != HU_M43_V5_MODEL_SCHEMA
        or model.get("safety_enabled") is not True
        or not isinstance(model.get("model_id"), str)
        or not model.get("model_id")
    ):
        raise ValueError("Attempt03 frozen model identity changed")
    _sha256_text(model.get("file_sha256"), "model.file_sha256")
    threshold = _finite_probability(
        model.get("safety_threshold"), "model.safety_threshold"
    )
    for value, label in (
        (training.get("file_sha256"), "training_manifest.file_sha256"),
        (training.get("canonical_sha256"), "training_manifest.canonical_sha256"),
        (executable.get("file_sha256"), "executable_model_freeze.file_sha256"),
        (
            executable.get("v5_implementation_sha256"),
            "executable_model_freeze.v5_implementation_sha256",
        ),
        (
            training_pipeline.get("file_sha256"),
            "training_pipeline_freeze.file_sha256",
        ),
        (plan.get("file_sha256"), "attempt03_plan.file_sha256"),
        (locked.get("file_sha256"), "inherited_locked.file_sha256"),
        (locked.get("identity_sha256"), "inherited_locked.identity_sha256"),
    ):
        _sha256_text(value, label)
    _positive_integer(locked.get("records"), "inherited_locked.records")
    if (
        locked.get("content_opened") is not False
        or locked.get("model_evaluation_count") != 0
        or not isinstance(locked.get("global_consumption_marker"), str)
        or not locked.get("global_consumption_marker")
    ):
        raise ValueError("Attempt03 frozen inherited-lock binding changed")
    if any(
        freeze.get(field) is not False
        for field in (
            "threshold_reselection_after_freeze_allowed",
            "model_reselection_after_freeze_allowed",
            "current_profile_mutated",
            "runtime_policy_activated",
            "full_replacement",
        )
    ):
        raise ValueError("Attempt03 model-threshold freeze guard changed")
    if final_training_manifest is not None:
        selected = validate_attempt03_final_training_manifest(
            final_training_manifest,
            model_sha256=str(model["file_sha256"]),
        )
        if (
            final_training_manifest.get("manifest_sha256")
            != training.get("canonical_sha256")
            or final_training_manifest.get("training_freeze_file_sha256")
            != training_pipeline.get("file_sha256")
            or not math.isclose(
                selected, threshold, rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            raise ValueError("Attempt03 freeze/final training manifest mismatch")


def load_bound_attempt03_v5_model(
    model_path: str | Path,
    *,
    expected_sha256: str,
    runtime_freeze: str | Path | Mapping[str, Any],
    final_training_manifest_path: str | Path,
) -> HuM43JointModelV5:
    """Load v5 only when model, threshold, freeze, and manifest all agree."""

    source = Path(model_path)
    expected = _sha256_text(expected_sha256, "expected_sha256")
    if file_sha256(source) != expected:
        raise ValueError("Attempt03 v5 artifact SHA-256 mismatch")
    freeze = (
        dict(runtime_freeze)
        if isinstance(runtime_freeze, Mapping)
        else read_json_mapping(runtime_freeze, "Attempt03 model-threshold freeze")
    )
    training_path = Path(final_training_manifest_path)
    training = read_json_mapping(training_path, "Attempt03 final manifest")
    validate_attempt03_runtime_freeze(freeze, final_training_manifest=training)
    frozen_model = _mapping(freeze.get("model"), "model")
    frozen_training = _mapping(
        freeze.get("training_manifest"), "training_manifest"
    )
    if expected != frozen_model.get("file_sha256"):
        raise ValueError("Attempt03 expected model SHA disagrees with freeze")
    if file_sha256(training_path) != frozen_training.get("file_sha256"):
        raise ValueError("Attempt03 final training manifest bytes changed")
    model = HuM43JointModelV5.load(source, expected_sha256=expected)
    if (
        model.model_id != frozen_model.get("model_id")
        or model.action_score_mode != HU_M43_V5_ACTION_SCORE_MODE
        or model.safety_enabled is not True
        or model.safety_estimator is None
        or not math.isclose(
            float(model.safety_threshold),
            float(frozen_model["safety_threshold"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("Attempt03 loaded v5 model disagrees with freeze")
    return model


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


__all__ = [
    "M43_ATTEMPT03_LOCKED_MARKER_SCHEMA",
    "M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA",
    "M43_ATTEMPT03_RUNTIME_FREEZE_SCHEMA",
    "M43_ATTEMPT03_RUNTIME_FREEZE_STATUS",
    "file_sha256",
    "load_bound_attempt03_v5_model",
    "read_json_mapping",
    "self_digest",
    "validate_attempt03_final_training_manifest",
    "validate_attempt03_precalibration_receipt",
    "validate_attempt03_runtime_freeze",
]
