"""Cloud-safe, resumable fold artifacts for the M4.3 joint model.

Only train/calibration rows and a redacted immutable projection are permitted in
the worker package.  The sealed local data contract and plan are validated while
the projection is prepared and again by the local assembler; neither source file
is required or accepted by a fold worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn

from .hu_m43_pilot_contract import canonical_manifest_sha256
from .hu_m4_joint_model import (
    PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    PairedDeltaRiskFoldEstimator,
)
from .train_hu_m4_joint_model import (
    M43FoldJobDefinition,
    M43FoldJobSpec,
    PreparedTeacherSample,
    _fit_paired_delta_risk_fold,
    _load_m43_role_binding,
    _m43_sample_identity_sha256,
    _normalize_input_paths,
    _sample_membership_hash,
    _sha256,
    build_m43_fold_training_plan,
    prepare_teacher_samples,
    read_teacher_jsonl,
    validate_disjoint_splits,
)


M43_FOLD_CLOUD_CONTRACT_SCHEMA = "hu_m43_fold_cloud_contract_v1"
M43_FOLD_JOB_MANIFEST_SCHEMA = "hu_m43_fold_job_manifest_v1"
M43_FOLD_JOB_DONE_SCHEMA = "hu_m43_fold_job_done_v1"
M43_FOLD_ESTIMATOR_ARTIFACT_SCHEMA = "hu_m43_fold_estimator_artifact_v1"
M43_FOLD_ASSEMBLY_SCHEMA = "hu_m43_fold_assembly_v1"
M43_LOCAL_ABORT_RECEIPT_SCHEMA = "hu_m43_local_training_abort_receipt_v1"
M43_LOCAL_REBIND_MANIFEST_SCHEMA = "hu_m43_local_sealed_rebind_manifest_v1"
M43_FOLD_COUNT = 5
M43_FOLD_JOB_COUNT = M43_FOLD_COUNT * (M43_FOLD_COUNT + 1)
M43_FROZEN_DEPENDENCIES = {
    "numpy": "2.2.6",
    "scikit_learn": "1.8.0",
}
M43_FROZEN_PROCESS_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

FOLD_HYPERPARAMETER_KEYS = (
    "cross_fit_folds",
    "iterations",
    "max_leaf_nodes",
    "learning_rate",
    "seed",
    "paired_se_floor",
    "paired_huber_alpha",
    "downside_quantile",
    "positive_gain_score_weight",
    "downside_risk_score_weight",
    "ensemble_disagreement_score_weight",
)
TRAINING_HYPERPARAMETER_KEYS = FOLD_HYPERPARAMETER_KEYS + (
    "action_score_mode",
    "model_id",
    "near_best_margin",
    "minimum_safe_teacher_gain",
    "l2_regularization",
    "safety_calibrator_c",
    "safety_fit_ratio",
    "safety_split_seed",
    "minimum_safety_fit_samples",
    "minimum_threshold_lock_samples",
    "thresholds",
    "minimum_calibration_fires",
    "maximum_false_positive_rate",
    "maximum_p95_loss",
    "maximum_p99_loss",
    "maximum_max_loss",
)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _optional_integer_matches(
    value: Any, expected: int | None, label: str
) -> bool:
    if expected is None:
        return value is None
    return _integer(value, label) == expected


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _reject_sensitive_strings(value: Any, *, label: str = "cloud contract") -> None:
    """Reject any sealed-split name/path accidentally copied into cloud state."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if "locked" in str(key).lower():
                raise ValueError(f"{label} contains a forbidden key")
            _reject_sensitive_strings(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_strings(child, label=label)
    elif isinstance(value, str) and "locked" in value.lower():
        raise ValueError(f"{label} contains a forbidden string")


def _verify_frozen_dependencies(value: Any) -> dict[str, str]:
    dependencies = dict(_mapping(value, "dependencies"))
    if dependencies != M43_FROZEN_DEPENDENCIES:
        raise ValueError("M4.3 frozen dependency versions changed")
    actual = {"numpy": np.__version__, "scikit_learn": sklearn.__version__}
    if actual != M43_FROZEN_DEPENDENCIES:
        raise RuntimeError(
            "M4.3 runtime dependency versions disagree with the frozen run: "
            f"expected={M43_FROZEN_DEPENDENCIES} actual={actual}"
        )
    return dependencies


def _verify_frozen_process_environment(value: Any) -> dict[str, str]:
    declared = dict(_mapping(value, "process_environment"))
    if declared != M43_FROZEN_PROCESS_ENVIRONMENT:
        raise ValueError("M4.3 frozen process environment changed")
    actual = {key: os.environ.get(key) for key in declared}
    if actual != M43_FROZEN_PROCESS_ENVIRONMENT:
        raise RuntimeError(
            "M4.3 process environment disagrees with the frozen run: "
            f"expected={M43_FROZEN_PROCESS_ENVIRONMENT} actual={actual}"
        )
    return declared


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_entries(
    paths: Sequence[Path], rows_by_path: Sequence[Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    if len(paths) != len(rows_by_path) or not paths:
        raise ValueError("input shard path/row lists disagree")
    return [
        {
            "index": index,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }
        for index, (path, rows) in enumerate(zip(paths, rows_by_path, strict=True))
    ]


def _fold_hyperparameters(values: Mapping[str, Any]) -> dict[str, Any]:
    if not set(FOLD_HYPERPARAMETER_KEYS) <= set(values):
        raise ValueError("M4.3 fold hyperparameters are incomplete")
    result = {key: values[key] for key in FOLD_HYPERPARAMETER_KEYS}
    if _integer(result["cross_fit_folds"], "cross_fit_folds", minimum=2) != 5:
        raise ValueError("distributed M4.3 requires exactly five folds")
    _integer(result["iterations"], "iterations", minimum=1)
    _integer(result["max_leaf_nodes"], "max_leaf_nodes", minimum=2)
    _integer(result["seed"], "seed", minimum=0)
    for key in (
        "learning_rate",
        "paired_se_floor",
        "paired_huber_alpha",
        "downside_quantile",
        "positive_gain_score_weight",
        "downside_risk_score_weight",
        "ensemble_disagreement_score_weight",
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        result[key] = float(value)
    if result["learning_rate"] <= 0.0 or result["paired_se_floor"] <= 0.0:
        raise ValueError("positive fold hyperparameters must be positive")
    return result


def _training_hyperparameters(values: Mapping[str, Any]) -> dict[str, Any]:
    if set(values) != set(TRAINING_HYPERPARAMETER_KEYS):
        raise ValueError("M4.3 training hyperparameter key set changed")
    result = dict(values)
    result.update(_fold_hyperparameters(values))
    if result["action_score_mode"] != PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
        raise ValueError("M4.3 distributed action-score mode changed")
    if not isinstance(result["model_id"], str) or not result["model_id"]:
        raise ValueError("model_id must be non-empty")
    for key in (
        "near_best_margin",
        "minimum_safe_teacher_gain",
        "l2_regularization",
        "safety_calibrator_c",
        "safety_fit_ratio",
        "maximum_false_positive_rate",
        "maximum_p95_loss",
        "maximum_p99_loss",
        "maximum_max_loss",
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        result[key] = float(value)
    for key in (
        "safety_split_seed",
        "minimum_safety_fit_samples",
        "minimum_threshold_lock_samples",
        "minimum_calibration_fires",
    ):
        result[key] = _integer(result[key], key, minimum=1)
    thresholds = _sequence(result["thresholds"], "thresholds")
    normalized_thresholds: list[float] = []
    for value in thresholds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("thresholds must be numeric")
        parsed = float(value)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError("thresholds must be inside [0, 1]")
        normalized_thresholds.append(parsed)
    if not normalized_thresholds or normalized_thresholds != sorted(
        set(normalized_thresholds)
    ):
        raise ValueError("thresholds must be sorted and unique")
    result["thresholds"] = normalized_thresholds
    if not 0.0 < result["safety_fit_ratio"] < 1.0:
        raise ValueError("safety_fit_ratio must be inside (0, 1)")
    if result["safety_calibrator_c"] <= 0.0:
        raise ValueError("safety_calibrator_c must be positive")
    return result


def _validate_predeclared_training_receipt(
    path: Path,
    *,
    expected_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a new distributed attempt to the safe aborted local attempt.

    The receipt is intentionally consumed only while building or locally
    verifying the redacted cloud projection.  Its contents are never copied
    into worker-visible state.
    """

    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(receipt, dict):
        raise ValueError("M4.3 predeclared training receipt must be an object")
    if receipt.get("schema") != M43_LOCAL_ABORT_RECEIPT_SCHEMA:
        raise ValueError("M4.3 predeclared training receipt schema mismatch")
    if receipt.get("status") != "aborted_before_model_or_manifest_write":
        raise ValueError("M4.3 predeclared training receipt status mismatch")
    expected_flags = {
        "model_written": False,
        "training_manifest_written": False,
        "locked_holdout_path_passed_to_trainer": False,
        "locked_holdout_opened_by_trainer": False,
        "locked_holdout_consumed": False,
        "safe_to_start_new_pre_holdout_training_attempt": True,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    for key, expected in expected_flags.items():
        if receipt.get(key) is not expected:
            raise ValueError(
                f"M4.3 predeclared training receipt unsafe flag: {key}"
            )
    declared = _training_hyperparameters(
        _mapping(
            receipt.get(
                "predeclared_training_configuration_for_equivalent_attempt"
            ),
            "predeclared training configuration",
        )
    )
    expected = _training_hyperparameters(expected_hyperparameters)
    if declared != expected:
        raise ValueError(
            "M4.3 distributed hyperparameters disagree with the predeclared "
            "aborted-attempt configuration"
        )
    return receipt


def _read_inputs(
    train_paths: Sequence[Path], calibration_paths: Sequence[Path]
) -> tuple[
    dict[str, list[list[dict[str, Any]]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[PreparedTeacherSample]],
]:
    raw_shards = {
        "train": [read_teacher_jsonl(path) for path in train_paths],
        "calibration": [read_teacher_jsonl(path) for path in calibration_paths],
    }
    raw = {
        name: [row for shard in shards for row in shard]
        for name, shards in raw_shards.items()
    }
    prepared = {name: prepare_teacher_samples(rows) for name, rows in raw.items()}
    validate_disjoint_splits(prepared["train"], prepared["calibration"], ())
    return raw_shards, raw, prepared


def build_fold_cloud_contract(
    *,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    data_contract_path: str | Path,
    plan_path: str | Path,
    repo_root: str | Path,
    predeclared_receipt_path: str | Path,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate local sealed inputs and return their path-free cloud projection."""

    train_paths = _normalize_input_paths(train_path, split="train")
    calibration_paths = _normalize_input_paths(
        calibration_path, split="calibration"
    )
    raw_shards, raw, prepared = _read_inputs(train_paths, calibration_paths)
    if len(raw["train"]) != 100 or len(raw["calibration"]) != 60:
        raise ValueError("M4.3 cloud projection requires exactly 100/60 rows")
    data_contract_source = Path(data_contract_path).resolve()
    plan_source = Path(plan_path).resolve()
    predeclared_receipt_source = Path(predeclared_receipt_path).resolve()
    contract = json.loads(data_contract_source.read_text(encoding="utf-8-sig"))
    if not isinstance(contract, dict):
        raise ValueError("M4.3 data contract must be an object")
    # This is the only operation that receives the full contract.  Its return
    # value is deliberately discarded because it contains local path metadata.
    _load_m43_role_binding(
        data_contract_source,
        plan_path=plan_source,
        repo_root=repo_root,
        train_paths=train_paths,
        calibration_paths=calibration_paths,
        raw=raw,
        prepared=prepared,
    )
    training_hyperparameters = _training_hyperparameters(hyperparameters)
    _validate_predeclared_training_receipt(
        predeclared_receipt_source,
        expected_hyperparameters=training_hyperparameters,
    )
    fold_hyperparameters = _fold_hyperparameters(training_hyperparameters)
    fold_plan = build_m43_fold_training_plan(
        prepared["train"],
        cross_fit_folds=int(fold_hyperparameters["cross_fit_folds"]),
        seed=int(fold_hyperparameters["seed"]),
    )
    jobs = [definition.spec.to_manifest() for definition in fold_plan.jobs]
    job_spec_sha256 = [definition.spec.sha256 for definition in fold_plan.jobs]
    inputs = {
        "train": _input_entries(train_paths, raw_shards["train"]),
        "calibration": _input_entries(
            calibration_paths, raw_shards["calibration"]
        ),
    }
    input_bundle_sha256 = canonical_manifest_sha256({"inputs": inputs})
    fold_plan_payload = {
        "outer_folds": M43_FOLD_COUNT,
        "inner_folds_per_outer": M43_FOLD_COUNT,
        "total_jobs": M43_FOLD_JOB_COUNT,
        "train_identity_sha256": _m43_sample_identity_sha256(
            fold_plan.ordered_samples
        ),
        "jobs": [
            {**job, "job_spec_sha256": digest}
            for job, digest in zip(jobs, job_spec_sha256, strict=True)
        ],
    }
    fold_plan_payload["fold_plan_sha256"] = canonical_manifest_sha256(
        {key: value for key, value in fold_plan_payload.items() if key != "fold_plan_sha256"}
    )
    unsigned = {
        "schema": M43_FOLD_CLOUD_CONTRACT_SCHEMA,
        "status": "frozen_cloud_safe",
        "predeclared_training_receipt_sha256": _sha256(
            predeclared_receipt_source
        ),
        "inputs": inputs,
        "input_bundle_sha256": input_bundle_sha256,
        "fold_plan": fold_plan_payload,
        "hyperparameters": training_hyperparameters,
        "dependencies": dict(M43_FROZEN_DEPENDENCIES),
        "process_environment": dict(M43_FROZEN_PROCESS_ENVIRONMENT),
        "cloud_safe": True,
    }
    _reject_sensitive_strings(unsigned)
    return {**unsigned, "contract_sha256": canonical_manifest_sha256(unsigned)}


def write_fold_cloud_contract(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    payload = build_fold_cloud_contract(**kwargs)
    _atomic_json(Path(path), payload)
    return payload


def build_local_rebind_manifest(
    *,
    run_name: str,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    data_contract_path: str | Path,
    plan_path: str | Path,
    repo_root: str | Path,
    predeclared_receipt_path: str | Path,
    fold_cloud_contract_path: str | Path,
    run_manifest_path: str | Path,
    source_archive_path: str | Path,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the local-only binding omitted from every cloud artifact.

    This sidecar is deliberately created only after the immutable worker run
    manifest and source archive exist.  Full sealed-contract and plan hashes
    live here and in the later trusted external training/freeze manifests, not
    in the worker projection or model pickle.
    """

    if not run_name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in run_name
    ):
        raise ValueError("local rebind run_name contains unsafe characters")
    normalized = _training_hyperparameters(hyperparameters)
    expected_projection = build_fold_cloud_contract(
        train_path=train_path,
        calibration_path=calibration_path,
        data_contract_path=data_contract_path,
        plan_path=plan_path,
        repo_root=repo_root,
        predeclared_receipt_path=predeclared_receipt_path,
        hyperparameters=normalized,
    )
    projection_path = Path(fold_cloud_contract_path).resolve()
    actual_projection = load_fold_cloud_contract(projection_path)
    if actual_projection != expected_projection:
        raise ValueError("local rebind cloud projection differs from sealed inputs")
    run_path = Path(run_manifest_path).resolve()
    run_manifest = json.loads(run_path.read_text(encoding="utf-8-sig"))
    if not isinstance(run_manifest, dict):
        raise ValueError("local rebind run manifest must be an object")
    if (
        run_manifest.get("schema") != "hu_m43_fold_spot_run_manifest_v1"
        or run_manifest.get("status") != "frozen"
        or run_manifest.get("run_name") != run_name
        or run_manifest.get("current_profile_mutated") is not False
        or run_manifest.get("no_runtime_activation") is not True
    ):
        raise ValueError("local rebind run lifecycle identity is invalid")
    source_path = Path(source_archive_path).resolve()
    source_binding = _mapping(run_manifest.get("source"), "run.source")
    cloud_binding = _mapping(
        run_manifest.get("cloud_contract"), "run.cloud_contract"
    )
    if source_binding.get("sha256") != _sha256(source_path):
        raise ValueError("local rebind source archive hash mismatch")
    if (
        cloud_binding.get("file_sha256") != _sha256(projection_path)
        or cloud_binding.get("contract_sha256")
        != actual_projection.get("contract_sha256")
        or run_manifest.get("training_config") != normalized
        or run_manifest.get("training_config_sha256")
        != canonical_manifest_sha256(normalized)
        or _mapping(run_manifest.get("inputs"), "run.inputs").get(
            "input_bundle_sha256"
        )
        != actual_projection.get("input_bundle_sha256")
    ):
        raise ValueError("local rebind run/projection binding mismatch")
    data_contract_source = Path(data_contract_path).resolve()
    plan_source = Path(plan_path).resolve()
    receipt_source = Path(predeclared_receipt_path).resolve()
    full_contract = json.loads(
        data_contract_source.read_text(encoding="utf-8-sig")
    )
    if not isinstance(full_contract, dict):
        raise ValueError("local rebind full data contract must be an object")
    canonical_contract_sha = _require_sha256(
        full_contract.get("contract_sha256"),
        "local rebind data contract canonical SHA-256",
    )
    unsigned = {
        "schema": M43_LOCAL_REBIND_MANIFEST_SCHEMA,
        "status": "frozen_local_only",
        "run_name": run_name,
        "sealed_data_contract": {
            "file_sha256": _sha256(data_contract_source),
            "canonical_sha256": canonical_contract_sha,
        },
        "full_plan": {"file_sha256": _sha256(plan_source)},
        "predeclared_training_receipt_sha256": _sha256(receipt_source),
        "inputs": actual_projection.get("inputs"),
        "input_bundle_sha256": actual_projection.get("input_bundle_sha256"),
        "cloud_projection": {
            "file_sha256": _sha256(projection_path),
            "contract_sha256": actual_projection.get("contract_sha256"),
        },
        "run_manifest": {"file_sha256": _sha256(run_path)},
        "source_archive": {"file_sha256": _sha256(source_path)},
        "training_config_sha256": canonical_manifest_sha256(normalized),
        "cloud_upload_forbidden": True,
        "model_pickle_embedding_forbidden": True,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    return {
        **unsigned,
        "rebind_sha256": canonical_manifest_sha256(unsigned),
    }


def write_local_rebind_manifest(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_local_rebind_manifest(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def load_local_rebind_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("local sealed rebind manifest must be an object")
    unsigned = dict(payload)
    declared = _require_sha256(
        unsigned.pop("rebind_sha256", None), "local rebind SHA-256"
    )
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("local sealed rebind manifest digest mismatch")
    if (
        payload.get("schema") != M43_LOCAL_REBIND_MANIFEST_SCHEMA
        or payload.get("status") != "frozen_local_only"
        or payload.get("cloud_upload_forbidden") is not True
        or payload.get("model_pickle_embedding_forbidden") is not True
        or payload.get("current_profile_mutated") is not False
        or payload.get("no_runtime_activation") is not True
    ):
        raise ValueError("local sealed rebind lifecycle contract is invalid")
    return payload


def verify_local_rebind_manifest(
    *,
    local_rebind_manifest_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    actual = load_local_rebind_manifest(local_rebind_manifest_path)
    expected = build_local_rebind_manifest(**kwargs)
    if actual != expected:
        raise ValueError("local sealed rebind manifest is stale or mismatched")
    return actual


def load_fold_cloud_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("fold cloud contract must be an object")
    digest = _require_sha256(payload.get("contract_sha256"), "contract_sha256")
    unsigned = dict(payload)
    unsigned.pop("contract_sha256", None)
    if canonical_manifest_sha256(unsigned) != digest:
        raise ValueError("fold cloud contract canonical SHA-256 mismatch")
    if payload.get("schema") != M43_FOLD_CLOUD_CONTRACT_SCHEMA:
        raise ValueError("fold cloud contract schema mismatch")
    expected_keys = {
        "schema",
        "status",
        "predeclared_training_receipt_sha256",
        "inputs",
        "input_bundle_sha256",
        "fold_plan",
        "hyperparameters",
        "dependencies",
        "process_environment",
        "cloud_safe",
        "contract_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("fold cloud contract key set changed")
    if payload.get("status") != "frozen_cloud_safe" or payload.get("cloud_safe") is not True:
        raise ValueError("fold cloud contract is not frozen cloud-safe")
    _require_sha256(
        payload.get("predeclared_training_receipt_sha256"),
        "predeclared_training_receipt_sha256",
    )
    _reject_sensitive_strings(payload)
    _training_hyperparameters(
        _mapping(payload.get("hyperparameters"), "hyperparameters")
    )
    _verify_frozen_dependencies(payload.get("dependencies"))
    if payload.get("process_environment") != M43_FROZEN_PROCESS_ENVIRONMENT:
        raise ValueError("fold cloud process environment changed")
    fold_plan = _mapping(payload.get("fold_plan"), "fold_plan")
    if (
        _integer(fold_plan.get("outer_folds"), "outer_folds") != M43_FOLD_COUNT
        or _integer(fold_plan.get("inner_folds_per_outer"), "inner_folds")
        != M43_FOLD_COUNT
        or _integer(fold_plan.get("total_jobs"), "total_jobs")
        != M43_FOLD_JOB_COUNT
    ):
        raise ValueError("fold cloud contract does not describe exact 30-job coverage")
    jobs = _sequence(fold_plan.get("jobs"), "fold_plan.jobs")
    if len(jobs) != M43_FOLD_JOB_COUNT:
        raise ValueError("fold cloud contract job list is incomplete")
    unsigned_plan = dict(fold_plan)
    declared_plan_sha = _require_sha256(
        unsigned_plan.pop("fold_plan_sha256", None), "fold_plan_sha256"
    )
    if canonical_manifest_sha256(unsigned_plan) != declared_plan_sha:
        raise ValueError("fold cloud plan SHA-256 mismatch")
    return payload


def _verify_cloud_inputs(
    contract: Mapping[str, Any],
    *,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
) -> tuple[list[PreparedTeacherSample], M43FoldTrainingPlan]:
    train_paths = _normalize_input_paths(train_path, split="train")
    calibration_paths = _normalize_input_paths(
        calibration_path, split="calibration"
    )
    raw_shards, raw, prepared = _read_inputs(train_paths, calibration_paths)
    actual_inputs = {
        "train": _input_entries(train_paths, raw_shards["train"]),
        "calibration": _input_entries(
            calibration_paths, raw_shards["calibration"]
        ),
    }
    if actual_inputs != contract.get("inputs"):
        raise ValueError("fold worker input bytes/rows disagree with cloud contract")
    actual_bundle_sha = canonical_manifest_sha256({"inputs": actual_inputs})
    if actual_bundle_sha != contract.get("input_bundle_sha256"):
        raise ValueError("fold worker input bundle SHA-256 mismatch")
    if len(raw["train"]) != 100 or len(raw["calibration"]) != 60:
        raise ValueError("fold worker requires exactly 100/60 rows")
    hyperparameters = _fold_hyperparameters(
        _mapping(contract.get("hyperparameters"), "hyperparameters")
    )
    plan = build_m43_fold_training_plan(
        prepared["train"],
        cross_fit_folds=int(hyperparameters["cross_fit_folds"]),
        seed=int(hyperparameters["seed"]),
    )
    declared_fold_plan = _mapping(contract.get("fold_plan"), "fold_plan")
    if declared_fold_plan.get("train_identity_sha256") != (
        _m43_sample_identity_sha256(plan.ordered_samples)
    ):
        raise ValueError("fold worker train identity digest mismatch")
    declared_jobs = _sequence(declared_fold_plan.get("jobs"), "fold_plan.jobs")
    expected_jobs = [
        {**job.spec.to_manifest(), "job_spec_sha256": job.spec.sha256}
        for job in plan.jobs
    ]
    if list(declared_jobs) != expected_jobs:
        raise ValueError("fold worker recomputed job grid disagrees with contract")
    return prepared["calibration"], plan


def _job_files(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "estimator.pkl",
        output_dir / "job_manifest.json",
        output_dir / "DONE.json",
    )


def _validate_existing_done(
    output_dir: Path,
    *,
    expected_definition: M43FoldJobDefinition,
    expected_cloud_contract_sha256: str,
    expected_run_name: str,
    expected_source_sha256: str,
    expected_run_manifest_sha256: str,
    expected_input_bundle_sha256: str,
) -> dict[str, Any]:
    artifact_path, manifest_path, done_path = _job_files(output_dir)
    if not done_path.is_file():
        raise ValueError("completed fold output has no DONE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    done = json.loads(done_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(done, dict):
        raise ValueError("fold manifest/DONE must be objects")
    spec = expected_definition.spec
    if (
        manifest.get("schema") != M43_FOLD_JOB_MANIFEST_SCHEMA
        or manifest.get("status") != "pass"
        or done.get("schema") != M43_FOLD_JOB_DONE_SCHEMA
        or done.get("status") != "complete"
        or _integer(manifest.get("job_index"), "job_index") != spec.job_index
        or _integer(done.get("job_index"), "DONE.job_index") != spec.job_index
        or manifest.get("run_name") != expected_run_name
        or done.get("run_name") != expected_run_name
        or manifest.get("source_sha256") != expected_source_sha256
        or done.get("source_sha256") != expected_source_sha256
        or manifest.get("run_manifest_sha256")
        != expected_run_manifest_sha256
        or done.get("run_manifest_sha256") != expected_run_manifest_sha256
        or manifest.get("cloud_contract_sha256")
        != expected_cloud_contract_sha256
        or done.get("cloud_contract_sha256") != expected_cloud_contract_sha256
        or manifest.get("input_bundle_sha256")
        != expected_input_bundle_sha256
        or done.get("input_bundle_sha256") != expected_input_bundle_sha256
        or manifest.get("job_kind") != spec.kind
        or done.get("job_kind") != spec.kind
        or _integer(manifest.get("outer_fold"), "manifest.outer_fold")
        != spec.outer_fold
        or _integer(done.get("outer_fold"), "DONE.outer_fold")
        != spec.outer_fold
        or not _optional_integer_matches(
            manifest.get("inner_fold"), spec.inner_fold, "manifest.inner_fold"
        )
        or not _optional_integer_matches(
            done.get("inner_fold"), spec.inner_fold, "DONE.inner_fold"
        )
        or manifest.get("job_spec") != spec.to_manifest()
        or not isinstance(manifest.get("job_spec"), Mapping)
        or canonical_manifest_sha256(manifest["job_spec"]) != spec.sha256
        or manifest.get("job_spec_sha256") != spec.sha256
        or done.get("job_spec_sha256") != spec.sha256
        or manifest.get("artifact_sha256") != _sha256(artifact_path)
        or done.get("artifact_sha256") != manifest.get("artifact_sha256")
        or done.get("job_manifest_sha256") != _sha256(manifest_path)
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("no_runtime_activation") is not True
        or done.get("current_profile_mutated") is not False
        or done.get("no_runtime_activation") is not True
    ):
        raise ValueError("existing fold DONE hash/identity chain is invalid")
    return done


def run_fold_job(
    *,
    job_index: int,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    fold_cloud_contract_path: str | Path,
    output_dir: str | Path,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    input_bundle_sha256: str,
    job_spec_sha256: str,
) -> dict[str, Any]:
    """Fit one deterministic estimator and publish DONE last, idempotently."""

    job_index = _integer(job_index, "job_index", minimum=0)
    if job_index >= M43_FOLD_JOB_COUNT:
        raise ValueError("job_index is outside exact M4.3 fold grid")
    source_sha256 = _require_sha256(source_sha256, "source_sha256")
    run_manifest_sha256 = _require_sha256(
        run_manifest_sha256, "run_manifest_sha256"
    )
    input_bundle_sha256 = _require_sha256(
        input_bundle_sha256, "input_bundle_sha256"
    )
    job_spec_sha256 = _require_sha256(job_spec_sha256, "job_spec_sha256")
    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_name):
        raise ValueError("run_name contains unsafe characters")
    cloud_contract_source = Path(fold_cloud_contract_path).resolve()
    cloud_contract = load_fold_cloud_contract(cloud_contract_source)
    _verify_frozen_process_environment(
        cloud_contract.get("process_environment")
    )
    cloud_contract_file_sha = _sha256(cloud_contract_source)
    if input_bundle_sha256 != cloud_contract.get("input_bundle_sha256"):
        raise ValueError("runner input bundle digest disagrees with cloud contract")
    _calibration, plan = _verify_cloud_inputs(
        cloud_contract,
        train_path=train_path,
        calibration_path=calibration_path,
    )
    definition = plan.jobs[job_index]
    if definition.spec.job_index != job_index or definition.spec.sha256 != job_spec_sha256:
        raise ValueError("runner job spec digest disagrees with deterministic plan")
    destination = Path(output_dir)
    artifact_path, manifest_path, done_path = _job_files(destination)
    if done_path.exists():
        return _validate_existing_done(
            destination,
            expected_definition=definition,
            expected_cloud_contract_sha256=cloud_contract_file_sha,
            expected_run_name=run_name,
            expected_source_sha256=source_sha256,
            expected_run_manifest_sha256=run_manifest_sha256,
            expected_input_bundle_sha256=input_bundle_sha256,
        )
    hyperparameters = _fold_hyperparameters(
        _mapping(cloud_contract.get("hyperparameters"), "hyperparameters")
    )
    estimator = _fit_paired_delta_risk_fold(
        definition.fit_samples,
        fold_index=definition.spec.estimator_fold_index,
        paired_se_floor=float(hyperparameters["paired_se_floor"]),
        huber_alpha=float(hyperparameters["paired_huber_alpha"]),
        downside_quantile=float(hyperparameters["downside_quantile"]),
        iterations=int(hyperparameters["iterations"]),
        max_leaf_nodes=int(hyperparameters["max_leaf_nodes"]),
        learning_rate=float(hyperparameters["learning_rate"]),
        seed=definition.spec.estimator_seed,
    )
    artifact_payload = {
        "schema": M43_FOLD_ESTIMATOR_ARTIFACT_SCHEMA,
        "job_spec_sha256": definition.spec.sha256,
        "estimator": estimator,
    }
    _atomic_pickle(artifact_path, artifact_payload)
    artifact_sha = _sha256(artifact_path)
    manifest = {
        "schema": M43_FOLD_JOB_MANIFEST_SCHEMA,
        "status": "pass",
        "run_name": run_name,
        "job_index": job_index,
        "job_kind": definition.spec.kind,
        "outer_fold": definition.spec.outer_fold,
        "inner_fold": definition.spec.inner_fold,
        "job_spec_sha256": definition.spec.sha256,
        "job_spec": definition.spec.to_manifest(),
        "source_sha256": source_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "cloud_contract_sha256": cloud_contract_file_sha,
        "input_bundle_sha256": input_bundle_sha256,
        "artifact_sha256": artifact_sha,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    _reject_sensitive_strings(manifest, label="fold job manifest")
    _atomic_json(manifest_path, manifest)
    done = {
        "schema": M43_FOLD_JOB_DONE_SCHEMA,
        "status": "complete",
        "run_name": run_name,
        "job_index": job_index,
        "job_kind": definition.spec.kind,
        "outer_fold": definition.spec.outer_fold,
        "inner_fold": definition.spec.inner_fold,
        "job_spec_sha256": definition.spec.sha256,
        "source_sha256": source_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "cloud_contract_sha256": cloud_contract_file_sha,
        "input_bundle_sha256": input_bundle_sha256,
        "artifact_sha256": artifact_sha,
        "job_manifest_sha256": _sha256(manifest_path),
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    _reject_sensitive_strings(done, label="fold job DONE")
    _atomic_json(done_path, done)
    return done


@dataclass
class M43FoldArtifactProvider:
    estimators: dict[int, PairedDeltaRiskFoldEstimator]
    expected_specs: dict[int, M43FoldJobSpec]
    assembly_receipt: dict[str, Any]
    consumed: set[int]

    def __call__(
        self,
        spec: M43FoldJobSpec,
        fit_samples: Sequence[PreparedTeacherSample],
    ) -> PairedDeltaRiskFoldEstimator:
        if spec.job_index in self.consumed:
            raise ValueError(f"M4.3 fold job consumed twice: {spec.job_index}")
        expected = self.expected_specs.get(spec.job_index)
        if expected is None or expected.to_manifest() != spec.to_manifest():
            raise ValueError("M4.3 assembler job spec changed during assembly")
        if _m43_sample_identity_sha256(fit_samples) != spec.fit_identity_sha256:
            raise ValueError("M4.3 assembler fit identities changed")
        estimator = self.estimators.get(spec.job_index)
        if estimator is None:
            raise ValueError(f"M4.3 estimator is missing: {spec.job_index}")
        self.consumed.add(spec.job_index)
        return estimator

    def assert_complete(self) -> None:
        expected = set(range(M43_FOLD_JOB_COUNT))
        if self.consumed != expected:
            raise ValueError(
                "M4.3 assembler did not consume exact 30-job coverage: "
                f"missing={sorted(expected - self.consumed)} "
                f"extra={sorted(self.consumed - expected)}"
            )


def _load_job_artifact(
    job_dir: Path,
    *,
    expected_definition: M43FoldJobDefinition,
    cloud_contract_sha256: str,
    input_bundle_sha256: str,
) -> tuple[PairedDeltaRiskFoldEstimator, dict[str, Any]]:
    artifact_path, manifest_path, done_path = _job_files(job_dir)
    for path in (artifact_path, manifest_path, done_path):
        if not path.is_file():
            raise ValueError(f"M4.3 fold job file missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    done = json.loads(done_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(done, dict):
        raise ValueError("M4.3 fold manifest/DONE must be objects")
    spec = expected_definition.spec
    artifact_sha = _sha256(artifact_path)
    manifest_sha = _sha256(manifest_path)
    shared_checks = (
        manifest.get("schema") == M43_FOLD_JOB_MANIFEST_SCHEMA
        and manifest.get("status") == "pass"
        and done.get("schema") == M43_FOLD_JOB_DONE_SCHEMA
        and done.get("status") == "complete"
        and _integer(manifest.get("job_index"), "job manifest index")
        == spec.job_index
        and _integer(done.get("job_index"), "DONE job index") == spec.job_index
        and manifest.get("job_kind") == spec.kind
        and done.get("job_kind") == spec.kind
        and _integer(manifest.get("outer_fold"), "manifest outer fold")
        == spec.outer_fold
        and _integer(done.get("outer_fold"), "DONE outer fold") == spec.outer_fold
        and _optional_integer_matches(
            manifest.get("inner_fold"), spec.inner_fold, "manifest inner fold"
        )
        and _optional_integer_matches(
            done.get("inner_fold"), spec.inner_fold, "DONE inner fold"
        )
        and manifest.get("job_spec") == spec.to_manifest()
        and isinstance(manifest.get("job_spec"), Mapping)
        and canonical_manifest_sha256(manifest["job_spec"]) == spec.sha256
        and manifest.get("job_spec_sha256") == spec.sha256
        and done.get("job_spec_sha256") == spec.sha256
        and manifest.get("cloud_contract_sha256") == cloud_contract_sha256
        and done.get("cloud_contract_sha256") == cloud_contract_sha256
        and manifest.get("input_bundle_sha256") == input_bundle_sha256
        and done.get("input_bundle_sha256") == input_bundle_sha256
        and manifest.get("artifact_sha256") == artifact_sha
        and done.get("artifact_sha256") == artifact_sha
        and done.get("job_manifest_sha256") == manifest_sha
        and manifest.get("current_profile_mutated") is False
        and manifest.get("no_runtime_activation") is True
        and done.get("current_profile_mutated") is False
        and done.get("no_runtime_activation") is True
    )
    if not shared_checks:
        raise ValueError(f"M4.3 fold job hash/spec chain invalid: {spec.job_index}")
    for key in ("run_name", "source_sha256", "run_manifest_sha256"):
        if manifest.get(key) != done.get(key):
            raise ValueError(f"M4.3 fold manifest/DONE mismatch: {key}")
    _require_sha256(manifest.get("source_sha256"), "job source_sha256")
    _require_sha256(
        manifest.get("run_manifest_sha256"), "job run_manifest_sha256"
    )
    encoded = artifact_path.read_bytes()
    payload = pickle.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("M4.3 fold estimator artifact must be an object")
    if (
        payload.get("schema") != M43_FOLD_ESTIMATOR_ARTIFACT_SCHEMA
        or payload.get("job_spec_sha256") != spec.sha256
    ):
        raise ValueError("M4.3 fold estimator artifact header mismatch")
    estimator = payload.get("estimator")
    if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
        raise TypeError("M4.3 fold artifact estimator type mismatch")
    estimator.__post_init__()
    if estimator.fold_index != spec.estimator_fold_index:
        raise ValueError("M4.3 fold artifact estimator index mismatch")
    return estimator, {
        "job_index": spec.job_index,
        "artifact_sha256": artifact_sha,
        "job_manifest_sha256": manifest_sha,
        "job_spec_sha256": spec.sha256,
        "run_name": manifest.get("run_name"),
        "source_sha256": manifest.get("source_sha256"),
        "run_manifest_sha256": manifest.get("run_manifest_sha256"),
    }


def load_fold_artifact_provider(
    *,
    artifacts_dir: str | Path,
    fold_cloud_contract_path: str | Path,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
) -> M43FoldArtifactProvider:
    root = Path(artifacts_dir).resolve()
    cloud_contract_source = Path(fold_cloud_contract_path).resolve()
    contract = load_fold_cloud_contract(cloud_contract_source)
    _calibration, plan = _verify_cloud_inputs(
        contract,
        train_path=train_path,
        calibration_path=calibration_path,
    )
    expected_names = {f"job-{index:02d}" for index in range(M43_FOLD_JOB_COUNT)}
    actual_names = {
        path.name for path in root.glob("job-*") if path.is_dir()
    }
    if actual_names != expected_names:
        raise ValueError(
            "M4.3 fold artifact directory coverage mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    cloud_contract_file_sha = _sha256(cloud_contract_source)
    input_bundle_sha = _require_sha256(
        contract.get("input_bundle_sha256"), "input_bundle_sha256"
    )
    estimators: dict[int, PairedDeltaRiskFoldEstimator] = {}
    job_receipts: list[dict[str, Any]] = []
    for definition in plan.jobs:
        estimator, receipt = _load_job_artifact(
            root / f"job-{definition.spec.job_index:02d}",
            expected_definition=definition,
            cloud_contract_sha256=cloud_contract_file_sha,
            input_bundle_sha256=input_bundle_sha,
        )
        estimators[definition.spec.job_index] = estimator
        job_receipts.append(receipt)
    run_names = {str(row["run_name"]) for row in job_receipts}
    source_hashes = {str(row["source_sha256"]) for row in job_receipts}
    run_manifest_hashes = {
        str(row["run_manifest_sha256"]) for row in job_receipts
    }
    if len(run_names) != 1 or len(source_hashes) != 1 or len(run_manifest_hashes) != 1:
        raise ValueError("M4.3 fold jobs do not share one immutable run identity")
    assembly = {
        "schema": M43_FOLD_ASSEMBLY_SCHEMA,
        "status": "pass",
        "job_count": M43_FOLD_JOB_COUNT,
        "run_name": next(iter(run_names)),
        "cloud_contract_sha256": cloud_contract_file_sha,
        "source_sha256": next(iter(source_hashes)),
        "run_manifest_sha256": next(iter(run_manifest_hashes)),
        "input_bundle_sha256": input_bundle_sha,
        "jobs": [
            {
                "job_index": row["job_index"],
                "artifact_sha256": row["artifact_sha256"],
                "job_manifest_sha256": row["job_manifest_sha256"],
                "job_spec_sha256": row["job_spec_sha256"],
            }
            for row in job_receipts
        ],
        "exact_outer_inner_coverage": True,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    return M43FoldArtifactProvider(
        estimators=estimators,
        expected_specs={job.spec.job_index: job.spec for job in plan.jobs},
        assembly_receipt=assembly,
        consumed=set(),
    )


def verify_fold_cloud_contract_against_local(
    *,
    fold_cloud_contract_path: str | Path,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    data_contract_path: str | Path,
    plan_path: str | Path,
    repo_root: str | Path,
    predeclared_receipt_path: str | Path,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    actual = load_fold_cloud_contract(fold_cloud_contract_path)
    expected = build_fold_cloud_contract(
        train_path=train_path,
        calibration_path=calibration_path,
        data_contract_path=data_contract_path,
        plan_path=plan_path,
        repo_root=repo_root,
        predeclared_receipt_path=predeclared_receipt_path,
        hyperparameters=hyperparameters,
    )
    if actual != expected:
        raise ValueError(
            "fold cloud contract disagrees with local sealed inputs or training config"
        )
    return actual


__all__ = [
    "FOLD_HYPERPARAMETER_KEYS",
    "M43FoldArtifactProvider",
    "M43_FOLD_ASSEMBLY_SCHEMA",
    "M43_FOLD_CLOUD_CONTRACT_SCHEMA",
    "M43_FOLD_COUNT",
    "M43_FOLD_JOB_COUNT",
    "M43_FOLD_JOB_DONE_SCHEMA",
    "M43_FOLD_JOB_MANIFEST_SCHEMA",
    "M43_FROZEN_DEPENDENCIES",
    "M43_FROZEN_PROCESS_ENVIRONMENT",
    "M43_LOCAL_REBIND_MANIFEST_SCHEMA",
    "TRAINING_HYPERPARAMETER_KEYS",
    "build_fold_cloud_contract",
    "build_local_rebind_manifest",
    "load_fold_artifact_provider",
    "load_fold_cloud_contract",
    "load_local_rebind_manifest",
    "run_fold_job",
    "verify_fold_cloud_contract_against_local",
    "verify_local_rebind_manifest",
    "write_local_rebind_manifest",
    "write_fold_cloud_contract",
]
