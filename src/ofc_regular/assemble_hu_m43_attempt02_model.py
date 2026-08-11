"""Assemble the M4.3 Attempt02 v4 model on the trusted local host.

The exact 30 train-only fold artifacts are verified and consumed before the
pre-calibration gate is evaluated.  A failed gate writes a durable No-Go
receipt without opening either the fresh calibration shards or the Attempt02
data contract.  Only a passing gate unlocks the sealed 50/50 role binding.
No inherited-holdout argument exists in this module or its CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hu_m43_attempt02_contract import (
    M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
    M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
    M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
)
from .hu_m43_attempt02_fold_training import (
    Attempt02V4TrainingConfig,
    load_attempt02_fold_artifact_provider,
    load_attempt02_fold_cloud_contract,
)
from .hu_m43_joint_model_v4 import (
    HU_M43_V4_TEACHER_STATUS,
    V4SafetyFitResult,
    V4ThresholdSelectionResult,
    build_v4_training_manifest,
    fit_v4_nested_crossfit,
    fit_v4_safety_calibrator,
    select_v4_threshold,
    write_v4_training_manifest,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256
from .train_hu_m4_joint_model import (
    PreparedTeacherSample,
    _normalize_input_paths,
    prepare_teacher_samples,
    read_teacher_jsonl,
)


M43_ATTEMPT02_ASSEMBLY_DECISION_SCHEMA = (
    "hu_m43_attempt02_v4_assembly_decision_v1"
)
M43_ATTEMPT02_CALIBRATION_BINDING_SCHEMA = (
    "hu_m43_attempt02_v4_calibration_binding_v1"
)

_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_DATA_CONTRACT_KEYS = {
    "schema",
    "status",
    "plan_sha256",
    "preflight",
    "fresh_generation",
    "exclusion_union",
    "fresh_splits",
    "calibration_partition",
    "teacher_shards",
    "teacher_shards_all_fresh_splits_sha256",
    "inherited_locked",
    "global_locked_consumption",
    "freshness",
    "runtime",
    "contract_sha256",
}
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Attempt02CalibrationRoles:
    safety_fit: tuple[PreparedTeacherSample, ...]
    threshold_lock: tuple[PreparedTeacherSample, ...]
    audit: Mapping[str, Any]


def assemble_attempt02_v4_from_fold_artifacts(
    *,
    fold_artifacts_dir: str | Path,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    attempt02_data_contract_path: str | Path,
    repo_root: str | Path,
    fold_cloud_contract_path: str | Path,
    output_model: str | Path,
    manifest_output: str | Path,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify, gate, calibrate if allowed, and publish the opt-in candidate."""

    model_path = Path(output_model).resolve()
    manifest_path = Path(manifest_output).resolve()
    if model_path == manifest_path:
        raise ValueError("Attempt02 model and manifest outputs must differ")
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("Attempt02 assembler refuses to overwrite outputs")

    cloud_contract_path = Path(fold_cloud_contract_path).resolve()
    cloud_contract = load_attempt02_fold_cloud_contract(cloud_contract_path)
    config = Attempt02V4TrainingConfig.from_manifest(
        cloud_contract.get("training_config")
    )
    provider = load_attempt02_fold_artifact_provider(
        artifacts_dir=fold_artifacts_dir,
        fold_cloud_contract_path=cloud_contract_path,
        train_path=train_path,
        expected_run_name=run_name,
        expected_source_sha256=source_sha256,
        expected_run_manifest_sha256=run_manifest_sha256,
    )
    crossfit = fit_v4_nested_crossfit(
        provider.training_samples,
        cross_fit_folds=5,
        seed=config.fold_seed,
        model_id=config.model_id,
        worker_config=config.worker,
        gate_config=config.precalibration_gate,
        fold_estimator_provider=provider,
    )
    provider.assert_complete()
    _validate_attempt02_crossfit(crossfit, config=config)
    assembly = {
        **provider.assembly_receipt,
        "status": "verified_consumed_exactly_once",
    }

    if crossfit.precalibration_report.get("status") != "go":
        receipt = _precalibration_no_go_receipt(
            crossfit_report=crossfit.report,
            assembly=assembly,
            cloud_contract=cloud_contract,
        )
        _write_json_exclusive(manifest_path, receipt)
        return receipt

    # This is the first point at which either the data-contract bytes or fresh
    # calibration JSONL are touched.  Keep it below the pre-calibration return.
    roles = load_attempt02_calibration_roles(
        data_contract_path=attempt02_data_contract_path,
        train_path=train_path,
        calibration_path=calibration_path,
        repo_root=repo_root,
    )
    safety_fit = fit_v4_safety_calibrator(
        crossfit.model,
        crossfit.safety_dataset,
        roles.safety_fit,
        safety_calibrator_c=config.safety_calibrator_c,
        seed=config.safety_seed,
        maximum_p95_loss=config.maximum_p95_loss,
        maximum_p99_loss=config.maximum_p99_loss,
        maximum_max_loss=config.maximum_max_loss,
    )
    threshold = select_v4_threshold(
        safety_fit,
        roles.threshold_lock,
        thresholds=config.thresholds,
        minimum_fires=config.minimum_fires,
        maximum_false_positive_rate=config.maximum_false_positive_rate,
        maximum_p95_loss=config.maximum_p95_loss,
        maximum_p99_loss=config.maximum_p99_loss,
        maximum_max_loss=config.maximum_max_loss,
    )
    return _publish_model_and_manifest(
        model_path=model_path,
        manifest_path=manifest_path,
        config=config,
        cloud_contract=cloud_contract,
        assembly=assembly,
        crossfit=crossfit,
        safety_fit=safety_fit,
        threshold=threshold,
        calibration_audit=roles.audit,
    )


def load_attempt02_calibration_roles(
    *,
    data_contract_path: str | Path,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    repo_root: str | Path,
) -> Attempt02CalibrationRoles:
    """Strictly validate the sealed Attempt02 contract and return 50/50 roles."""

    contract_path = Path(data_contract_path).resolve()
    contract = _read_mapping(contract_path, "Attempt02 data contract")
    if set(contract) != _DATA_CONTRACT_KEYS:
        raise ValueError("Attempt02 data contract key set changed")
    unsigned = dict(contract)
    declared_contract_sha = _require_sha256(
        unsigned.pop("contract_sha256", None), "Attempt02 contract SHA-256"
    )
    if canonical_manifest_sha256(unsigned) != declared_contract_sha:
        raise ValueError("Attempt02 data contract digest mismatch")
    if (
        contract.get("schema") != M43_ATTEMPT02_DATA_CONTRACT_SCHEMA
        or contract.get("status")
        != "pass_fresh_train_calibration_sealed_inherited_locked_unopened"
    ):
        raise ValueError("Attempt02 data contract lifecycle changed")
    runtime = _mapping(contract.get("runtime"), "Attempt02 runtime")
    if runtime != {
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "policy_activated": False,
        "full_replacement": False,
        "large_scale_authorized": False,
    }:
        raise ValueError("Attempt02 data contract runtime guards changed")
    inherited = _mapping(
        contract.get("inherited_locked"), "Attempt02 inherited metadata"
    )
    if (
        inherited.get("classification") != "inherited_unopened"
        or inherited.get("content_parse_count") != 0
        or inherited.get("model_evaluation_count") != 0
        or inherited.get("structural_byte_hash_audit_only") is not True
    ):
        raise ValueError("Attempt02 inherited holdout was opened before freeze")
    global_consumption = _mapping(
        contract.get("global_locked_consumption"),
        "Attempt02 global consumption",
    )
    if (
        global_consumption.get("matching_marker_count") != 0
        or global_consumption.get("status") != "unconsumed_preflight"
    ):
        raise ValueError("Attempt02 inherited holdout is no longer unconsumed")
    freshness = _mapping(contract.get("freshness"), "Attempt02 freshness")
    if set(freshness) != {
        "train_calibration_identity_overlap",
        "exclusion_hand_seed_overlap",
        "exclusion_observation_fingerprint_overlap",
        "inherited_locked_hand_seed_overlap",
        "inherited_locked_observation_fingerprint_overlap",
    } or any(value != 0 for value in freshness.values()):
        raise ValueError("Attempt02 freshness evidence changed")

    root = Path(repo_root).resolve()
    train_paths = _normalize_input_paths(train_path, split="train")
    calibration_paths = _normalize_input_paths(
        calibration_path, split="calibration"
    )
    rows_by_split = {
        "train": [read_teacher_jsonl(path) for path in train_paths],
        "calibration": [read_teacher_jsonl(path) for path in calibration_paths],
    }
    actual_teacher_shards = {
        "schema": M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
        "splits": {
            "train": _ordered_shard_binding(
                train_paths, rows_by_split["train"], root=root, split="train"
            ),
            "calibration": _ordered_shard_binding(
                calibration_paths,
                rows_by_split["calibration"],
                root=root,
                split="calibration",
            ),
        },
    }
    actual_teacher_shards["all_fresh_splits_sha256"] = (
        canonical_manifest_sha256(actual_teacher_shards)
    )
    declared_teacher_shards = _mapping(
        contract.get("teacher_shards"), "Attempt02 teacher shards"
    )
    if dict(declared_teacher_shards) != actual_teacher_shards:
        raise ValueError("Attempt02 fresh teacher shard binding changed")
    if contract.get("teacher_shards_all_fresh_splits_sha256") != (
        actual_teacher_shards["all_fresh_splits_sha256"]
    ):
        raise ValueError("Attempt02 teacher shard aggregate digest changed")

    raw_train = [row for shard in rows_by_split["train"] for row in shard]
    raw_calibration = [
        row for shard in rows_by_split["calibration"] for row in shard
    ]
    if len(raw_train) != 200 or len(raw_calibration) != 100:
        raise ValueError("Attempt02 fresh split counts changed")
    fresh_splits = _mapping(contract.get("fresh_splits"), "fresh_splits")
    if set(fresh_splits) != {"train", "calibration"}:
        raise ValueError("Attempt02 fresh split key set changed")
    for split, rows in (("train", raw_train), ("calibration", raw_calibration)):
        declared_split = _mapping(fresh_splits.get(split), f"fresh_splits.{split}")
        identities = {_row_identity(row, split) for row in rows}
        profile_counts = Counter(
            str(
                _mapping(row.get("provenance"), f"{split} provenance").get(
                    "root_profile", ""
                )
            )
            for row in rows
        )
        expected_profiles = {
            profile: (40 if split == "train" else 20) for profile in _PROFILES
        }
        if (
            declared_split.get("records") != len(rows)
            or declared_split.get("shards")
            != (20 if split == "train" else 10)
            or declared_split.get("identity_sha256") != _identity_digest(identities)
            or dict(declared_split.get("profile_counts", {}))
            != expected_profiles
            or dict(profile_counts) != expected_profiles
            or declared_split.get("audited_paired_delta_records") != len(rows)
        ):
            raise ValueError(f"Attempt02 {split} identity binding changed")

    prepared = prepare_teacher_samples(raw_calibration)
    by_identity: dict[tuple[int, str], PreparedTeacherSample] = {}
    profile_by_identity: dict[tuple[int, str], str] = {}
    for row, sample in zip(raw_calibration, prepared, strict=True):
        identity = _row_identity(row, "calibration")
        if identity[1] != sample.observation_fingerprint:
            raise ValueError("Attempt02 calibration observation fingerprint changed")
        if identity in by_identity:
            raise ValueError("Attempt02 calibration identity is duplicated")
        provenance = _mapping(row.get("provenance"), "calibration provenance")
        profile = str(provenance.get("root_profile", ""))
        if profile not in _PROFILES:
            raise ValueError("Attempt02 calibration root profile changed")
        by_identity[identity] = sample
        profile_by_identity[identity] = profile

    partition = _mapping(
        contract.get("calibration_partition"), "calibration_partition"
    )
    if (
        partition.get("schema") != M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA
        or partition.get("method") != "profile_stratified_identity_hash_v1"
        or partition.get("overlap") != 0
        or partition.get("inherited_locked_used") is not False
        or set(partition)
        != {
            "schema",
            "method",
            "safety_fit",
            "threshold_lock",
            "overlap",
            "inherited_locked_used",
        }
    ):
        raise ValueError("Attempt02 calibration partition changed")
    role_values: dict[str, tuple[PreparedTeacherSample, ...]] = {}
    role_identities: dict[str, set[tuple[int, str]]] = {}
    role_audit: dict[str, Any] = {}
    for role in ("safety_fit", "threshold_lock"):
        declared = _mapping(partition.get(role), f"calibration_partition.{role}")
        if set(declared) != {
            "records",
            "identity_sha256",
            "profile_counts",
            "identities",
        }:
            raise ValueError(f"Attempt02 {role} key set changed")
        raw_identities = declared.get("identities")
        if not isinstance(raw_identities, list):
            raise ValueError(f"Attempt02 {role} identities must be a list")
        ordered = tuple(_declared_identity(value, role) for value in raw_identities)
        identity_set = set(ordered)
        counts = Counter(profile_by_identity.get(identity) for identity in ordered)
        expected_counts = {profile: 10 for profile in _PROFILES}
        if (
            len(ordered) != 50
            or len(identity_set) != 50
            or declared.get("records") != 50
            or declared.get("identity_sha256") != _identity_digest(identity_set)
            or dict(declared.get("profile_counts", {})) != expected_counts
            or dict(counts) != expected_counts
            or not identity_set <= set(by_identity)
        ):
            raise ValueError(f"Attempt02 {role} identity binding changed")
        role_values[role] = tuple(by_identity[identity] for identity in ordered)
        role_identities[role] = identity_set
        role_audit[role] = {
            "records": 50,
            "identity_sha256": declared["identity_sha256"],
            "profile_counts": expected_counts,
        }
    if role_identities["safety_fit"] & role_identities["threshold_lock"]:
        raise ValueError("Attempt02 calibration roles overlap")
    if role_identities["safety_fit"] | role_identities["threshold_lock"] != set(
        by_identity
    ):
        raise ValueError("Attempt02 calibration roles do not cover fresh calibration")
    train_identities = {_row_identity(row, "train") for row in raw_train}
    if train_identities & set(by_identity):
        raise ValueError("Attempt02 train/calibration identities overlap")
    audit = {
        "schema": M43_ATTEMPT02_CALIBRATION_BINDING_SCHEMA,
        "data_contract_file_sha256": _file_sha256(contract_path),
        "data_contract_sha256": declared_contract_sha,
        "teacher_shards_all_fresh_splits_sha256": actual_teacher_shards[
            "all_fresh_splits_sha256"
        ],
        "roles": role_audit,
        "train_calibration_identity_overlap": 0,
        "inherited_holdout_input_accepted": False,
        "inherited_holdout_content_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return Attempt02CalibrationRoles(
        safety_fit=role_values["safety_fit"],
        threshold_lock=role_values["threshold_lock"],
        audit=audit,
    )


def _precalibration_no_go_receipt(
    *,
    crossfit_report: Mapping[str, Any],
    assembly: Mapping[str, Any],
    cloud_contract: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema": M43_ATTEMPT02_ASSEMBLY_DECISION_SCHEMA,
        "status": "no_go_precalibration",
        "promotion_status": "no_go_precalibration",
        "crossfit": dict(crossfit_report),
        "fold_assembly": dict(assembly),
        "cloud_contract_sha256": cloud_contract["contract_sha256"],
        "training_config_sha256": cloud_contract["training_config_sha256"],
        "calibration": {
            "data_contract_opened": False,
            "fresh_rows_opened": False,
            "safety_fit_performed": False,
            "threshold_selection_performed": False,
        },
        "inherited_holdout_input_accepted": False,
        "inherited_holdout_content_opened": False,
        "model_written": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
        "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
    }
    return {**unsigned, "receipt_sha256": canonical_manifest_sha256(unsigned)}


def _validate_attempt02_crossfit(
    crossfit: Any, *, config: Attempt02V4TrainingConfig
) -> None:
    report = _mapping(crossfit.report, "Attempt02 v4 crossfit report")
    precalibration = _mapping(
        crossfit.precalibration_report, "Attempt02 v4 pre-calibration report"
    )
    expected_gate_sha = canonical_manifest_sha256(
        config.precalibration_gate.to_manifest()
    )
    if (
        report.get("states") != 200
        or report.get("folds") != 5
        or report.get("fold_jobs") != 30
        or report.get("exact_outer_inner_job_grid") is not True
        or report.get("baseline_training_rows_included") is not False
        or report.get("calibration_opened") is not False
        or precalibration.get("config_sha256") != expected_gate_sha
        or _mapping(precalibration.get("metrics"), "pre-calibration metrics").get(
            "states"
        )
        != 200
        or precalibration.get("calibration_opened") is not False
        or precalibration.get("status") not in {"go", "no_go"}
    ):
        raise ValueError("Attempt02 v4 nested OOF/pre-calibration contract changed")


def _publish_model_and_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    config: Attempt02V4TrainingConfig,
    cloud_contract: Mapping[str, Any],
    assembly: Mapping[str, Any],
    crossfit: Any,
    safety_fit: V4SafetyFitResult,
    threshold: V4ThresholdSelectionResult,
    calibration_audit: Mapping[str, Any],
) -> dict[str, Any]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging = manifest_path.parent / f".m43-attempt02-v4-{os.getpid()}"
    if staging.exists():
        raise FileExistsError("Attempt02 assembler staging already exists")
    staging.mkdir()
    staged_model = staging / "model.pkl"
    staged_manifest = staging / "training_manifest.json"
    model_published = False
    try:
        model_manifest = {
            "schema": "hu_m43_attempt02_v4_model_binding_v1",
            "training_config_schema": config.schema,
            "cloud_contract_sha256": cloud_contract["contract_sha256"],
            "training_config_sha256": cloud_contract["training_config_sha256"],
            "fold_assembly_sha256": canonical_manifest_sha256(assembly),
            "data_contract_sha256": calibration_audit["data_contract_sha256"],
            "calibration_binding_schema": calibration_audit["schema"],
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
            "runtime_teacher_inputs": False,
        }
        frozen_model = replace(threshold.model, manifest=model_manifest)
        frozen_threshold = replace(threshold, model=frozen_model)
        model_sha = frozen_model.save(staged_model)
        manifest = build_v4_training_manifest(
            crossfit,
            safety_fit,
            frozen_threshold,
            model_sha256=model_sha,
        )
        manifest.pop("manifest_sha256", None)
        manifest.update(
            {
                "milestone": "M4.3-attempt02",
                "fold_assembly": dict(assembly),
                "cloud_contract": {
                    "contract_sha256": cloud_contract["contract_sha256"],
                    "training_config_sha256": cloud_contract[
                        "training_config_sha256"
                    ],
                    "fresh_train_only": True,
                },
                "attempt02_data_contract": {
                    "schema": M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
                    "file_sha256": calibration_audit[
                        "data_contract_file_sha256"
                    ],
                    "contract_sha256": calibration_audit[
                        "data_contract_sha256"
                    ],
                },
                "calibration_binding": dict(calibration_audit),
                "calibration_opened_after_precalibration_go": True,
                "model_artifact": {
                    "sha256": model_sha,
                    "model_id": frozen_model.model_id,
                    "safety_enabled": frozen_model.safety_enabled,
                    "safety_threshold": frozen_model.safety_threshold,
                },
                "inherited_holdout_input_accepted": False,
                "inherited_holdout_content_opened": False,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
                "full_replacement": False,
            }
        )
        manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
        write_v4_training_manifest(staged_manifest, manifest)
        if _file_sha256(staged_model) != model_sha:
            raise ValueError("Attempt02 staged model hash changed")
        os.replace(staged_model, model_path)
        model_published = True
        os.replace(staged_manifest, manifest_path)
        return manifest
    except Exception:
        if model_published and model_path.exists() and not manifest_path.exists():
            model_path.unlink()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _ordered_shard_binding(
    paths: Sequence[Path],
    rows_by_path: Sequence[Sequence[Mapping[str, Any]]],
    *,
    root: Path,
    split: str,
) -> dict[str, Any]:
    if len(paths) != len(rows_by_path) or not paths:
        raise ValueError(f"Attempt02 {split} shard list is invalid")
    shards: list[dict[str, Any]] = []
    all_rows: list[Mapping[str, Any]] = []
    for index, (path, rows) in enumerate(zip(paths, rows_by_path, strict=True)):
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"Attempt02 {split} shard contains another split")
        identities = [_row_identity(row, split) for row in rows]
        shards.append(
            {
                "index": index,
                "path": _path_token(path.resolve(), root),
                "bytes": path.stat().st_size,
                "records": len(rows),
                "file_sha256": _file_sha256(path),
                "canonical_rows_sha256": _canonical_rows_sha256(rows),
                "identity_sha256": _identity_digest(identities),
            }
        )
        all_rows.extend(rows)
    unsigned = {
        "ordered_shards": shards,
        "records": len(all_rows),
        "canonical_rows_sha256": _canonical_rows_sha256(all_rows),
    }
    return {
        **unsigned,
        "ordered_shards_sha256": canonical_manifest_sha256(unsigned),
    }


def _row_identity(row: Mapping[str, Any], label: str) -> tuple[int, str]:
    seed = row.get("hand_seed")
    fingerprint = row.get("observation_fingerprint")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Attempt02 {label} hand_seed is invalid")
    return int(seed), _require_sha256(fingerprint, f"{label} fingerprint")


def _declared_identity(value: Any, label: str) -> tuple[int, str]:
    row = _mapping(value, f"{label} identity")
    if set(row) != {"hand_seed", "observation_fingerprint"}:
        raise ValueError(f"Attempt02 {label} identity key set changed")
    return _row_identity(row, label)


def _identity_digest(values: Iterable[tuple[int, str]]) -> str:
    encoded = "".join(
        f"{seed}\t{fingerprint}\n" for seed, fingerprint in sorted(values)
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _path_token(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-artifacts-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument(
        "--calibration", type=Path, action="append", required=True
    )
    parser.add_argument("--attempt02-data-contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fold-cloud-contract", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = assemble_attempt02_v4_from_fold_artifacts(
        fold_artifacts_dir=args.fold_artifacts_dir,
        train_path=args.train,
        calibration_path=args.calibration,
        attempt02_data_contract_path=args.attempt02_data_contract,
        repo_root=args.repo_root,
        fold_cloud_contract_path=args.fold_cloud_contract,
        output_model=args.output_model,
        manifest_output=args.manifest_output,
        run_name=args.run_name,
        source_sha256=args.source_sha256,
        run_manifest_sha256=args.run_manifest_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Attempt02CalibrationRoles",
    "M43_ATTEMPT02_ASSEMBLY_DECISION_SCHEMA",
    "assemble_attempt02_v4_from_fold_artifacts",
    "load_attempt02_calibration_roles",
    "main",
    "parse_args",
]
