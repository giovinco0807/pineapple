"""Fail-closed runtime-freeze contract for the opt-in Attempt04 v6 model.

The freeze is created only after the model and absolute safety threshold are
fixed and before any locked200 byte is inspected.  Loading requires the model,
final training manifest, threshold lock, and runtime freeze to agree byte for
byte.  Nothing in this module resolves ``current`` or activates a profile.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_joint_model_v6 import (
    HU_M43_V6_ACTION_SCORE_MODE,
    HU_M43_V6_ARTIFACT_SCHEMA,
    HU_M43_V6_MODEL_SCHEMA,
    HU_M43_V6_PROPOSAL_SCHEMA,
    HU_M43_V6_SAFETY_FEATURE_DIM,
    HU_M43_V6_SAFETY_FEATURE_SCHEMA,
    HuM43JointModelV6,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256


M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA = (
    "hu_m43_attempt04_v6_final_training_manifest_v1"
)
M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA = "hu_m43_attempt04_v6_threshold_lock_v1"
M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA = (
    "hu_m43_attempt04_v6_model_threshold_runtime_freeze_v1"
)
M43_ATTEMPT04_RUNTIME_FREEZE_STATUS = "frozen_before_locked200_open"
M43_ATTEMPT04_LOCKED_BINDING_SCHEMA = "hu_m43_attempt04_locked200_binding_v1"
M43_ATTEMPT04_LOCKED_MARKER_SCHEMA = (
    "hu_m43_attempt04_locked200_consumption_marker_v1"
)
M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA = "hu_m43_attempt04_locked200_receipt_v1"
M43_ATTEMPT04_LOCKED_RECORDS = 200
M43_ATTEMPT04_TAIL_LIMITS = (25.0, 40.0, 50.0)

_FALSE_GUARDS = (
    "threshold_reselection_after_freeze_allowed",
    "model_reselection_after_freeze_allowed",
    "feature_reselection_after_freeze_allowed",
    "locked200_opened",
    "current_profile_resolved",
    "current_profile_mutated",
    "runtime_policy_activated",
    "full_replacement",
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


def validate_attempt04_threshold_lock(
    lock: Mapping[str, Any], *, model_sha256: str
) -> float:
    expected_model_sha = _sha256_text(model_sha256, "model_sha256")
    if lock.get("schema") != M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA:
        raise ValueError("Attempt04 threshold-lock schema mismatch")
    if lock.get("lock_sha256") != self_digest(lock, "lock_sha256"):
        raise ValueError("Attempt04 threshold-lock self digest mismatch")
    if (
        lock.get("status") != "fixed_before_locked200_open"
        or lock.get("model_sha256") != expected_model_sha
        or lock.get("model_schema") != HU_M43_V6_MODEL_SCHEMA
        or lock.get("proposal_schema") != HU_M43_V6_PROPOSAL_SCHEMA
        or lock.get("action_score_mode") != HU_M43_V6_ACTION_SCORE_MODE
        or lock.get("safety_feature_schema")
        != HU_M43_V6_SAFETY_FEATURE_SCHEMA
        or lock.get("safety_feature_dim") != HU_M43_V6_SAFETY_FEATURE_DIM
        or lock.get("safety_enabled") is not True
    ):
        raise ValueError("Attempt04 threshold-lock model identity mismatch")
    grid_value = lock.get("threshold_grid")
    if not isinstance(grid_value, Sequence) or isinstance(
        grid_value, (str, bytes)
    ):
        raise ValueError("Attempt04 threshold grid must be an array")
    grid = tuple(
        _finite_probability(value, "threshold_grid value") for value in grid_value
    )
    if not grid or tuple(sorted(set(grid))) != grid:
        raise ValueError("Attempt04 threshold grid must be unique and increasing")
    selected = _finite_probability(
        lock.get("selected_threshold"), "selected_threshold"
    )
    if not any(_same_number(selected, value) for value in grid):
        raise ValueError("Attempt04 selected threshold is outside its frozen grid")
    if lock.get("threshold_selection_method") not in {
        "fixed_grid_on_sealed_calibration_v1",
        "fixed_preregistered_absolute_threshold_v1",
    }:
        raise ValueError("Attempt04 threshold selection method changed")
    if any(lock.get(field) is not False for field in _FALSE_GUARDS):
        raise ValueError("Attempt04 threshold-lock lifecycle guard changed")
    return selected


def validate_attempt04_final_training_manifest(
    manifest: Mapping[str, Any],
    *,
    model_sha256: str,
    threshold_lock_file_sha256: str,
    threshold_lock: Mapping[str, Any],
) -> float:
    expected_model_sha = _sha256_text(model_sha256, "model_sha256")
    expected_lock_file_sha = _sha256_text(
        threshold_lock_file_sha256, "threshold_lock_file_sha256"
    )
    if manifest.get("schema") != M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA:
        raise ValueError("Attempt04 final training manifest schema mismatch")
    if manifest.get("manifest_sha256") != self_digest(
        manifest, "manifest_sha256"
    ):
        raise ValueError("Attempt04 final training manifest self digest mismatch")
    if (
        manifest.get("status") != "candidate_ready_for_runtime_freeze"
        or manifest.get("promotion_status")
        != "candidate_ready_for_runtime_freeze"
        or manifest.get("model_sha256") != expected_model_sha
        or manifest.get("model_schema") != HU_M43_V6_MODEL_SCHEMA
        or manifest.get("artifact_schema") != HU_M43_V6_ARTIFACT_SCHEMA
        or manifest.get("proposal_schema") != HU_M43_V6_PROPOSAL_SCHEMA
        or manifest.get("action_score_mode") != HU_M43_V6_ACTION_SCORE_MODE
        or manifest.get("safety_feature_schema")
        != HU_M43_V6_SAFETY_FEATURE_SCHEMA
        or manifest.get("safety_feature_dim") != HU_M43_V6_SAFETY_FEATURE_DIM
        or manifest.get("threshold_lock_file_sha256")
        != expected_lock_file_sha
        or manifest.get("threshold_lock_sha256")
        != threshold_lock.get("lock_sha256")
        or manifest.get("runtime_teacher_inputs") is not False
    ):
        raise ValueError("Attempt04 final training manifest identity mismatch")
    selected = validate_attempt04_threshold_lock(
        threshold_lock, model_sha256=expected_model_sha
    )
    if not _same_number(manifest.get("selected_threshold"), selected):
        raise ValueError("Attempt04 manifest/threshold-lock selection mismatch")
    if any(manifest.get(field) is not False for field in _FALSE_GUARDS):
        raise ValueError("Attempt04 final manifest lifecycle guard changed")
    return selected


def validate_attempt04_locked_binding(binding: Mapping[str, Any]) -> None:
    """Validate metadata only; this function never resolves or stats shards."""

    if (
        binding.get("schema") != M43_ATTEMPT04_LOCKED_BINDING_SCHEMA
        or binding.get("status") != "sealed_unopened"
        or binding.get("records") != M43_ATTEMPT04_LOCKED_RECORDS
        or binding.get("content_opened") is not False
        or binding.get("model_evaluation_count") != 0
    ):
        raise ValueError("Attempt04 locked200 binding identity mismatch")
    _sha256_text(binding.get("identity_sha256"), "locked200.identity_sha256")
    marker = binding.get("global_consumption_marker")
    if (
        not isinstance(marker, str)
        or not marker
        or Path(marker).name != "M43_ATTEMPT04_LOCKED200_CONSUMED.json"
    ):
        raise ValueError("Attempt04 locked200 marker path is missing")
    shards = binding.get("ordered_shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("Attempt04 locked200 ordered_shards must be an array")
    if not shards:
        raise ValueError("Attempt04 locked200 requires at least one shard")
    paths: set[str] = set()
    records = 0
    for index, row in enumerate(shards):
        shard = _mapping(row, f"locked200.ordered_shards[{index}]")
        path = shard.get("path")
        if not isinstance(path, str) or not path or path in paths:
            raise ValueError("Attempt04 locked200 shard path is missing or duplicated")
        paths.add(path)
        _sha256_text(shard.get("file_sha256"), "locked200 shard file_sha256")
        records += _positive_integer(shard.get("records"), "locked200 shard records")
        _positive_integer(shard.get("bytes"), "locked200 shard bytes")
    if records != M43_ATTEMPT04_LOCKED_RECORDS:
        raise ValueError("Attempt04 locked200 shard record total changed")


def validate_attempt04_runtime_freeze(
    freeze: Mapping[str, Any],
    *,
    final_training_manifest: Mapping[str, Any] | None = None,
    threshold_lock: Mapping[str, Any] | None = None,
) -> None:
    if (
        freeze.get("schema") != M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA
        or freeze.get("status") != M43_ATTEMPT04_RUNTIME_FREEZE_STATUS
    ):
        raise ValueError("Attempt04 runtime-freeze identity mismatch")
    if freeze.get("freeze_sha256") != self_digest(freeze, "freeze_sha256"):
        raise ValueError("Attempt04 runtime-freeze self digest mismatch")
    model = _mapping(freeze.get("model"), "freeze.model")
    training = _mapping(
        freeze.get("final_training_manifest"), "freeze.final_training_manifest"
    )
    threshold = _mapping(freeze.get("threshold_lock"), "freeze.threshold_lock")
    implementation = _mapping(
        freeze.get("v6_implementation"), "freeze.v6_implementation"
    )
    attempt04_plan = _mapping(freeze.get("attempt04_plan"), "freeze.attempt04_plan")
    population_plan = _mapping(
        freeze.get("population_plan"), "freeze.population_plan"
    )
    locked = _mapping(freeze.get("locked200"), "freeze.locked200")
    if (
        model.get("schema") != HU_M43_V6_MODEL_SCHEMA
        or model.get("artifact_schema") != HU_M43_V6_ARTIFACT_SCHEMA
        or model.get("proposal_schema") != HU_M43_V6_PROPOSAL_SCHEMA
        or model.get("action_score_mode") != HU_M43_V6_ACTION_SCORE_MODE
        or model.get("safety_feature_schema")
        != HU_M43_V6_SAFETY_FEATURE_SCHEMA
        or model.get("safety_feature_dim") != HU_M43_V6_SAFETY_FEATURE_DIM
        or model.get("safety_enabled") is not True
        or not isinstance(model.get("model_id"), str)
        or not model.get("model_id")
    ):
        raise ValueError("Attempt04 frozen model identity changed")
    _sha256_text(model.get("file_sha256"), "model.file_sha256")
    _finite_probability(model.get("safety_threshold"), "model.safety_threshold")
    cushions = _finite_nonnegative_triplet(
        model.get("tail_cushions"), "model.tail_cushions"
    )
    limits = _finite_positive_triplet(model.get("tail_limits"), "model.tail_limits")
    if tuple(limits) != M43_ATTEMPT04_TAIL_LIMITS:
        raise ValueError("Attempt04 frozen tail limits changed")
    if any(value < 0.0 for value in cushions):  # defensive clarity
        raise ValueError("Attempt04 frozen tail cushions changed")
    for value, label in (
        (training.get("file_sha256"), "final_training_manifest.file_sha256"),
        (training.get("canonical_sha256"), "final_training_manifest.canonical_sha256"),
        (threshold.get("file_sha256"), "threshold_lock.file_sha256"),
        (threshold.get("canonical_sha256"), "threshold_lock.canonical_sha256"),
        (implementation.get("file_sha256"), "v6_implementation.file_sha256"),
        (attempt04_plan.get("file_sha256"), "attempt04_plan.file_sha256"),
        (population_plan.get("file_sha256"), "population_plan.file_sha256"),
    ):
        _sha256_text(value, label)
    if implementation.get("module") != "src/ofc_regular/hu_m43_joint_model_v6.py":
        raise ValueError("Attempt04 v6 implementation path changed")
    validate_attempt04_locked_binding(locked)
    if any(freeze.get(field) is not False for field in _FALSE_GUARDS):
        raise ValueError("Attempt04 runtime-freeze lifecycle guard changed")
    if (final_training_manifest is None) != (threshold_lock is None):
        raise ValueError("Attempt04 manifest and threshold lock must validate together")
    if final_training_manifest is not None and threshold_lock is not None:
        selected = validate_attempt04_final_training_manifest(
            final_training_manifest,
            model_sha256=str(model["file_sha256"]),
            threshold_lock_file_sha256=str(threshold["file_sha256"]),
            threshold_lock=threshold_lock,
        )
        if (
            final_training_manifest.get("manifest_sha256")
            != training.get("canonical_sha256")
            or threshold_lock.get("lock_sha256")
            != threshold.get("canonical_sha256")
            or not _same_number(selected, model.get("safety_threshold"))
        ):
            raise ValueError("Attempt04 freeze/manifest/threshold-lock mismatch")


def build_attempt04_runtime_freeze(
    *,
    model_path: str | Path,
    final_training_manifest_path: str | Path,
    threshold_lock_path: str | Path,
    attempt04_plan_path: str | Path,
    population_plan_path: str | Path,
    locked200_binding: Mapping[str, Any],
    v6_implementation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build metadata-only freeze without touching any locked200 shard."""

    model_source = Path(model_path).resolve()
    training_source = Path(final_training_manifest_path).resolve()
    threshold_source = Path(threshold_lock_path).resolve()
    attempt04_source = Path(attempt04_plan_path).resolve()
    population_source = Path(population_plan_path).resolve()
    implementation_source = (
        Path(v6_implementation_path).resolve()
        if v6_implementation_path is not None
        else Path(__file__).with_name("hu_m43_joint_model_v6.py").resolve()
    )
    sources = {
        model_source,
        training_source,
        threshold_source,
        attempt04_source,
        population_source,
        implementation_source,
    }
    if len(sources) != 6 or not all(path.is_file() for path in sources):
        raise ValueError("Attempt04 runtime-freeze inputs are missing or aliased")
    model_sha = file_sha256(model_source)
    lock = read_json_mapping(threshold_source, "Attempt04 threshold lock")
    training = read_json_mapping(training_source, "Attempt04 final manifest")
    selected = validate_attempt04_final_training_manifest(
        training,
        model_sha256=model_sha,
        threshold_lock_file_sha256=file_sha256(threshold_source),
        threshold_lock=lock,
    )
    model = HuM43JointModelV6.load(model_source, expected_sha256=model_sha)
    if (
        not model.safety_enabled
        or model.safety_estimator is None
        or not _same_number(model.safety_threshold, selected)
        or model.action_score_mode != HU_M43_V6_ACTION_SCORE_MODE
    ):
        raise ValueError("Attempt04 v6 model disagrees with final threshold lock")
    validate_attempt04_locked_binding(locked200_binding)
    freeze: dict[str, Any] = {
        "schema": M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA,
        "status": M43_ATTEMPT04_RUNTIME_FREEZE_STATUS,
        "model": {
            "path": str(model_source),
            "file_sha256": model_sha,
            "model_id": model.model_id,
            "schema": HU_M43_V6_MODEL_SCHEMA,
            "artifact_schema": HU_M43_V6_ARTIFACT_SCHEMA,
            "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
            "action_score_mode": HU_M43_V6_ACTION_SCORE_MODE,
            "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
            "safety_feature_dim": HU_M43_V6_SAFETY_FEATURE_DIM,
            "safety_enabled": True,
            "safety_threshold": float(model.safety_threshold),
            "tail_cushions": [float(value) for value in model.tail_cushions],
            "tail_limits": [float(value) for value in model.tail_limits],
        },
        "final_training_manifest": {
            "path": str(training_source),
            "file_sha256": file_sha256(training_source),
            "canonical_sha256": training["manifest_sha256"],
        },
        "threshold_lock": {
            "path": str(threshold_source),
            "file_sha256": file_sha256(threshold_source),
            "canonical_sha256": lock["lock_sha256"],
        },
        "attempt04_plan": {
            "path": str(attempt04_source),
            "file_sha256": file_sha256(attempt04_source),
        },
        "population_plan": {
            "path": str(population_source),
            "file_sha256": file_sha256(population_source),
        },
        "v6_implementation": {
            "module": "src/ofc_regular/hu_m43_joint_model_v6.py",
            "file_sha256": file_sha256(implementation_source),
        },
        "locked200": dict(locked200_binding),
        **{field: False for field in _FALSE_GUARDS},
    }
    freeze["freeze_sha256"] = self_digest(freeze, "freeze_sha256")
    validate_attempt04_runtime_freeze(
        freeze,
        final_training_manifest=training,
        threshold_lock=lock,
    )
    return freeze


def load_bound_attempt04_v6_model(
    model_path: str | Path,
    *,
    expected_sha256: str,
    runtime_freeze: str | Path | Mapping[str, Any],
    final_training_manifest_path: str | Path,
    threshold_lock_path: str | Path,
) -> HuM43JointModelV6:
    source = Path(model_path).resolve()
    expected = _sha256_text(expected_sha256, "expected_sha256")
    if file_sha256(source) != expected:
        raise ValueError("Attempt04 v6 artifact SHA-256 mismatch")
    freeze = (
        dict(runtime_freeze)
        if isinstance(runtime_freeze, Mapping)
        else read_json_mapping(runtime_freeze, "Attempt04 runtime freeze")
    )
    training_source = Path(final_training_manifest_path).resolve()
    threshold_source = Path(threshold_lock_path).resolve()
    training = read_json_mapping(training_source, "Attempt04 final manifest")
    threshold = read_json_mapping(threshold_source, "Attempt04 threshold lock")
    validate_attempt04_runtime_freeze(
        freeze,
        final_training_manifest=training,
        threshold_lock=threshold,
    )
    frozen_model = _mapping(freeze.get("model"), "freeze.model")
    frozen_training = _mapping(
        freeze.get("final_training_manifest"), "freeze.final_training_manifest"
    )
    frozen_threshold = _mapping(freeze.get("threshold_lock"), "freeze.threshold_lock")
    if (
        expected != frozen_model.get("file_sha256")
        or file_sha256(training_source) != frozen_training.get("file_sha256")
        or file_sha256(threshold_source) != frozen_threshold.get("file_sha256")
    ):
        raise ValueError("Attempt04 loader artifact bytes disagree with freeze")
    model = HuM43JointModelV6.load(source, expected_sha256=expected)
    if (
        model.model_id != frozen_model.get("model_id")
        or model.action_score_mode != HU_M43_V6_ACTION_SCORE_MODE
        or model.safety_enabled is not True
        or model.safety_estimator is None
        or not _same_number(
            model.safety_threshold, frozen_model.get("safety_threshold")
        )
        or tuple(float(value) for value in model.tail_cushions)
        != tuple(float(value) for value in frozen_model.get("tail_cushions", ()))
        or tuple(float(value) for value in model.tail_limits)
        != tuple(float(value) for value in frozen_model.get("tail_limits", ()))
    ):
        raise ValueError("Attempt04 loaded v6 model disagrees with freeze")
    return model


def validate_attempt04_runtime_artifact_files(
    freeze: Mapping[str, Any],
    *,
    model_path: str | Path,
    final_training_manifest_path: str | Path,
    threshold_lock_path: str | Path,
    attempt04_plan_path: str | Path,
    population_plan_path: str | Path,
    v6_implementation_path: str | Path | None = None,
) -> dict[str, str]:
    """Verify every non-locked source byte bound by the runtime freeze."""

    sources = {
        "model": Path(model_path).resolve(),
        "training": Path(final_training_manifest_path).resolve(),
        "threshold": Path(threshold_lock_path).resolve(),
        "attempt04_plan": Path(attempt04_plan_path).resolve(),
        "population_plan": Path(population_plan_path).resolve(),
        "implementation": (
            Path(v6_implementation_path).resolve()
            if v6_implementation_path is not None
            else Path(__file__).with_name("hu_m43_joint_model_v6.py").resolve()
        ),
    }
    if len(set(sources.values())) != len(sources) or not all(
        path.is_file() for path in sources.values()
    ):
        raise ValueError("Attempt04 runtime source files are missing or aliased")
    training = read_json_mapping(sources["training"], "Attempt04 final manifest")
    threshold = read_json_mapping(sources["threshold"], "Attempt04 threshold lock")
    validate_attempt04_runtime_freeze(
        freeze,
        final_training_manifest=training,
        threshold_lock=threshold,
    )
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    expected = {
        "model": _mapping(freeze.get("model"), "freeze.model").get("file_sha256"),
        "training": _mapping(
            freeze.get("final_training_manifest"), "freeze.final_training_manifest"
        ).get("file_sha256"),
        "threshold": _mapping(
            freeze.get("threshold_lock"), "freeze.threshold_lock"
        ).get("file_sha256"),
        "attempt04_plan": _mapping(
            freeze.get("attempt04_plan"), "freeze.attempt04_plan"
        ).get("file_sha256"),
        "population_plan": _mapping(
            freeze.get("population_plan"), "freeze.population_plan"
        ).get("file_sha256"),
        "implementation": _mapping(
            freeze.get("v6_implementation"), "freeze.v6_implementation"
        ).get("file_sha256"),
    }
    for name, actual in hashes.items():
        if actual != expected[name]:
            raise ValueError(f"Attempt04 {name} bytes disagree with runtime freeze")
    return hashes


def write_attempt04_runtime_freeze(
    path: str | Path, freeze: Mapping[str, Any]
) -> None:
    validate_attempt04_runtime_freeze(freeze)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(freeze, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def _finite_nonnegative_triplet(value: Any, label: str) -> tuple[float, float, float]:
    result = _finite_triplet(value, label)
    if any(number < 0.0 for number in result):
        raise ValueError(f"{label} must be non-negative")
    return result


def _finite_positive_triplet(value: Any, label: str) -> tuple[float, float, float]:
    result = _finite_triplet(value, label)
    if any(number <= 0.0 for number in result):
        raise ValueError(f"{label} must be positive")
    return result


def _finite_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain three finite values")
    try:
        result = tuple(float(number) for number in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain three finite values") from exc
    if len(result) != 3 or not all(math.isfinite(number) for number in result):
        raise ValueError(f"{label} must contain three finite values")
    return result  # type: ignore[return-value]


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


__all__ = [
    "M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA",
    "M43_ATTEMPT04_LOCKED_BINDING_SCHEMA",
    "M43_ATTEMPT04_LOCKED_MARKER_SCHEMA",
    "M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA",
    "M43_ATTEMPT04_LOCKED_RECORDS",
    "M43_ATTEMPT04_RUNTIME_FREEZE_SCHEMA",
    "M43_ATTEMPT04_RUNTIME_FREEZE_STATUS",
    "M43_ATTEMPT04_TAIL_LIMITS",
    "M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA",
    "build_attempt04_runtime_freeze",
    "file_sha256",
    "load_bound_attempt04_v6_model",
    "read_json_mapping",
    "self_digest",
    "validate_attempt04_final_training_manifest",
    "validate_attempt04_locked_binding",
    "validate_attempt04_runtime_freeze",
    "validate_attempt04_runtime_artifact_files",
    "validate_attempt04_threshold_lock",
    "write_attempt04_runtime_freeze",
]
