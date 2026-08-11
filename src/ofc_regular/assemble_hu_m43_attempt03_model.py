"""Assemble, one-shot pre-calibrate, and calibrate Attempt03 v5.

Subcommands intentionally have disjoint arguments.  ``assemble-fit`` cannot
name a holdout; ``evaluate-precal`` cannot name inherited calibration; and
``calibrate`` refuses to parse calibration until a bound pre-cal Go receipt
has been validated.  No subcommand accepts the inherited locked holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .assemble_hu_m43_attempt02_model import load_attempt02_calibration_roles
from .hu_m43_attempt03_contract import (
    _identity_digest,
    _row_identity,
    load_and_validate_attempt03_plan,
)
from .hu_m43_attempt03_training import (
    M43_ATTEMPT03_FIT_MANIFEST_SCHEMA,
    M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA,
    Attempt03TrainingConfig,
    build_attempt03_final_manifest,
    build_attempt03_fit_manifest,
    build_attempt03_precalibration_receipt,
    claim_attempt03_precalibration,
    evaluate_attempt03_precalibration,
    fit_attempt03_base_oof_runtime_meta,
    fit_attempt03_safety_calibrator,
    load_attempt03_fit_bundle,
    load_attempt03_fold_artifact_provider,
    load_attempt03_fold_cloud_contract,
    load_attempt03_model_freeze,
    load_attempt03_training_freeze,
    save_attempt03_fit_bundle,
    select_attempt03_threshold,
)
from .hu_m43_joint_model_v5 import HuM43JointModelV5
from .hu_m43_pilot_contract import (
    build_ordered_teacher_shard_binding,
    canonical_manifest_sha256,
)
from .hu_turn3_model import load_hu_action_value_model
from .train_hu_m4_joint_model import prepare_teacher_samples, read_teacher_jsonl


M43_ATTEMPT03_PRECAL_OPEN_AUTHORIZATION_SCHEMA = (
    "hu_m43_attempt03_precal_open_authorization_v1"
)
M43_ATTEMPT03_MODEL_THRESHOLD_FREEZE_SCHEMA = (
    "hu_m43_attempt03_v5_model_threshold_freeze_v1"
)
M43_ATTEMPT03_CALIBRATION_MARKER_SCHEMA = (
    "hu_m43_attempt03_sealed_calibration_consumption_v1"
)


def assemble_attempt03_fit_candidate(
    *,
    fold_artifacts_dir: str | Path,
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
    fold_cloud_contract_path: str | Path,
    model_freeze_path: str | Path,
    training_freeze_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    """Assemble fit-only artifacts; this signature cannot accept holdouts."""

    root = Path(repo_root).resolve()
    freeze_path = Path(model_freeze_path).resolve()
    load_attempt03_model_freeze(freeze_path, repo_root=root)
    training_freeze_path = Path(training_freeze_path).resolve()
    load_attempt03_training_freeze(
        training_freeze_path,
        repo_root=root,
        expected_model_freeze_path=freeze_path,
    )
    contract_path = Path(fold_cloud_contract_path).resolve()
    contract = load_attempt03_fold_cloud_contract(contract_path, repo_root=root)
    if (
        contract["model_freeze"]["file_sha256"] != _sha(freeze_path)
        or contract["training_freeze"]["file_sha256"]
        != _sha(training_freeze_path)
    ):
        raise ValueError("Attempt03 fold contract/model freeze hash changed")
    provider = load_attempt03_fold_artifact_provider(
        artifacts_dir=fold_artifacts_dir,
        fold_cloud_contract_path=contract_path,
        inherited_train_path=inherited_train_path,
        fresh_train_fit_path=fresh_train_fit_path,
        expected_run_name=run_name,
        expected_source_sha256=source_sha256,
        expected_run_manifest_sha256=run_manifest_sha256,
    )
    stage18 = load_hu_action_value_model(
        root / "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl"
    )
    config = Attempt03TrainingConfig.from_manifest(contract["training_config"])
    result = fit_attempt03_base_oof_runtime_meta(
        provider.training_samples,
        stage18_scorer=stage18,
        config=config,
        fold_estimator_provider=provider,
    )
    provider.assert_complete()
    bound_manifest = {
        **dict(result.model.manifest),
        "fold_cloud_contract_sha256": contract["contract_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "model_freeze_file_sha256": _sha(freeze_path),
        "training_freeze_file_sha256": _sha(training_freeze_path),
        "fold_assembly_sha256": canonical_manifest_sha256(
            provider.assembly_receipt
        ),
        "precalibration_opened": False,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    result = replace(
        result,
        model=replace(result.model, manifest=bound_manifest),
    )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Attempt03 fit output is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{os.getpid()}.fit"
    if staging.exists():
        raise FileExistsError("Attempt03 fit staging exists")
    staging.mkdir()
    try:
        model_path = staging / "candidate_model.pkl"
        bundle_path = staging / "fit_bundle.pkl"
        manifest_path = staging / "fit_manifest.json"
        model_sha = result.model.save(model_path)
        bundle_sha = save_attempt03_fit_bundle(
            bundle_path,
            result,
            training_config_sha256=contract["training_config_sha256"],
            model_freeze_file_sha256=_sha(freeze_path),
            training_freeze_file_sha256=_sha(training_freeze_path),
        )
        manifest = build_attempt03_fit_manifest(
            result=result,
            model_sha256=model_sha,
            fit_bundle_sha256=bundle_sha,
            fold_assembly=provider.assembly_receipt,
            fold_cloud_contract_sha256=contract["contract_sha256"],
            training_config_sha256=contract["training_config_sha256"],
            model_freeze_file_sha256=_sha(freeze_path),
            training_freeze_file_sha256=_sha(training_freeze_path),
        )
        _write_json_exclusive(manifest_path, manifest)
        os.replace(staging, destination)
        return {
            "status": manifest["status"],
            "model": str(destination / "candidate_model.pkl"),
            "model_sha256": model_sha,
            "fit_bundle": str(destination / "fit_bundle.pkl"),
            "fit_bundle_sha256": bundle_sha,
            "fit_manifest": str(destination / "fit_manifest.json"),
            "fit_manifest_file_sha256": _sha(destination / "fit_manifest.json"),
            "precalibration_opened": False,
            "sealed_calibration_opened": False,
            "inherited_locked_opened": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_attempt03_precal_open_authorization(
    *,
    candidate_model_path: str | Path,
    fit_bundle_path: str | Path,
    fit_manifest_path: str | Path,
    fold_cloud_contract_path: str | Path,
    model_freeze_path: str | Path,
    training_freeze_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Verify the completed fit without locating or reading pre-cal data.

    Cross-fit metrics and teacher-valued rows in the fit bundle are not read
    into the authorization decision.  Eligibility is purely the frozen
    schema/hash/lifecycle chain established before fresh teacher rows existed.
    """

    root = Path(repo_root).resolve()
    freeze_path = Path(model_freeze_path).resolve()
    freeze = load_attempt03_model_freeze(freeze_path, repo_root=root)
    training_freeze_path = Path(training_freeze_path).resolve()
    load_attempt03_training_freeze(
        training_freeze_path,
        repo_root=root,
        expected_model_freeze_path=freeze_path,
    )
    contract_path = Path(fold_cloud_contract_path).resolve()
    contract = load_attempt03_fold_cloud_contract(contract_path, repo_root=root)
    fit_manifest_path = Path(fit_manifest_path).resolve()
    fit_manifest = _load_fit_manifest(fit_manifest_path)
    candidate_path = Path(candidate_model_path).resolve()
    bundle_path = Path(fit_bundle_path).resolve()
    if (
        fit_manifest["model_sha256"] != _sha(candidate_path)
        or fit_manifest["fit_bundle_sha256"] != _sha(bundle_path)
        or fit_manifest["fold_cloud_contract_sha256"]
        != contract["contract_sha256"]
        or fit_manifest["training_config_sha256"]
        != contract["training_config_sha256"]
        or fit_manifest["model_freeze_file_sha256"] != _sha(freeze_path)
        or fit_manifest["training_freeze_file_sha256"]
        != _sha(training_freeze_path)
        or contract["model_freeze"]["file_sha256"] != _sha(freeze_path)
        or contract["training_freeze"]["file_sha256"]
        != _sha(training_freeze_path)
    ):
        raise ValueError("Attempt03 pre-cal open fit/hash chain changed")
    model = HuM43JointModelV5.load(
        candidate_path, expected_sha256=fit_manifest["model_sha256"]
    )
    if model.safety_enabled or model.safety_estimator is not None:
        raise ValueError("Attempt03 pre-cal open candidate is not fail-closed")
    # Validate the bundle envelope/type/hash, but deliberately do not inspect
    # any teacher-valued row or metric when making this authorization.
    load_attempt03_fit_bundle(
        bundle_path, expected_sha256=fit_manifest["fit_bundle_sha256"]
    )
    implementation = _mapping(freeze["implementation"], "implementation")
    authorization = {
        "schema": M43_ATTEMPT03_PRECAL_OPEN_AUTHORIZATION_SCHEMA,
        "status": "authorized_to_create_receiver_open_claim",
        "decision_basis": "frozen_schema_and_hash_chain_only",
        "row_valued_metric_used_for_authorization": False,
        "precalibration_path_received": False,
        "precalibration_content_read": False,
        # The receiver accepts only this authorization file.  These resolved
        # paths let it re-hash every fit artifact itself before it creates the
        # irreversible pre-cal open claim; a digest copied into an
        # authorization is never trusted on its own.
        "candidate_model_path": str(candidate_path),
        "candidate_model_sha256": fit_manifest["model_sha256"],
        "fit_bundle_path": str(bundle_path),
        "fit_bundle_sha256": fit_manifest["fit_bundle_sha256"],
        "fit_manifest_path": str(fit_manifest_path),
        "fit_manifest_file_sha256": _sha(fit_manifest_path),
        "fit_manifest_canonical_sha256": fit_manifest["manifest_sha256"],
        "fold_cloud_contract_path": str(contract_path),
        "fold_cloud_contract_file_sha256": _sha(contract_path),
        "fold_cloud_contract_canonical_sha256": contract["contract_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "model_freeze_path": str(freeze_path),
        "model_freeze_file_sha256": _sha(freeze_path),
        "training_freeze_path": str(training_freeze_path),
        "training_freeze_file_sha256": _sha(training_freeze_path),
        "v5_implementation_sha256": implementation["file_sha256"],
        "stage18_model_sha256": implementation["stage18_model"]["file_sha256"],
        "fit_manifest_status_required": "fit_candidate_precalibration_unopened",
        "candidate_safety_enabled": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    authorization["authorization_sha256"] = canonical_manifest_sha256(
        authorization
    )
    return authorization


def write_attempt03_precal_open_authorization(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    payload = build_attempt03_precal_open_authorization(**kwargs)
    _write_json_exclusive(Path(path), payload)
    return payload


def _load_attempt02_sealed_calibration_declaration(
    path: str | Path,
) -> dict[str, Any]:
    """Read only the signed Attempt02 metadata that identifies calibration.

    This helper deliberately has no calibration-path argument.  It therefore
    cannot stat, hash, or parse the sealed JSONL while preparing the one-shot
    consumption claim.
    """

    source = Path(path).resolve()
    contract = _read_mapping(source, "Attempt02 data contract")
    _validate_self_digest(contract, "contract_sha256", "Attempt02 data contract")
    if (
        contract.get("schema") != "hu_m43_attempt02_data_contract_v1"
        or contract.get("status")
        != "pass_fresh_train_calibration_sealed_inherited_locked_unopened"
    ):
        raise ValueError("Attempt03 sealed calibration contract lifecycle changed")
    fresh = _mapping(contract.get("fresh_splits"), "Attempt02 fresh_splits")
    calibration = _mapping(
        fresh.get("calibration"), "Attempt02 fresh_splits.calibration"
    )
    identity_sha256 = _require_sha(
        calibration.get("identity_sha256"),
        "Attempt02 calibration identity_sha256",
    )
    if calibration.get("records") != 100:
        raise ValueError("Attempt02 sealed calibration record declaration changed")
    partition = _mapping(
        contract.get("calibration_partition"), "Attempt02 calibration_partition"
    )
    if (
        partition.get("schema")
        != "hu_m43_attempt02_calibration_partition_v1"
        or partition.get("overlap") != 0
        or partition.get("inherited_locked_used") is not False
    ):
        raise ValueError("Attempt02 sealed calibration partition changed")
    roles: dict[str, dict[str, Any]] = {}
    declared_union: set[tuple[int, str]] = set()
    for role in ("safety_fit", "threshold_lock"):
        value = _mapping(partition.get(role), f"Attempt02 calibration {role}")
        role_identity = _require_sha(
            value.get("identity_sha256"),
            f"Attempt02 calibration {role} identity_sha256",
        )
        raw_identities = value.get("identities")
        if value.get("records") != 50 or not isinstance(raw_identities, list):
            raise ValueError(f"Attempt02 calibration {role} declaration changed")
        identities: set[tuple[int, str]] = set()
        for raw in raw_identities:
            item = _mapping(raw, f"Attempt02 calibration {role} identity")
            hand_seed = item.get("hand_seed")
            fingerprint = item.get("observation_fingerprint")
            if (
                not isinstance(hand_seed, int)
                or isinstance(hand_seed, bool)
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError(
                    f"Attempt02 calibration {role} identity declaration changed"
                )
            identities.add((hand_seed, fingerprint))
        if len(identities) != 50 or _identity_digest(identities) != role_identity:
            raise ValueError(f"Attempt02 calibration {role} identity digest changed")
        if declared_union & identities:
            raise ValueError("Attempt02 calibration role identities overlap")
        declared_union.update(identities)
        roles[role] = {"records": 50, "identity_sha256": role_identity}
    if len(declared_union) != 100 or _identity_digest(declared_union) != identity_sha256:
        raise ValueError("Attempt02 calibration union identity digest changed")
    return {
        "data_contract_path": str(source),
        "data_contract_file_sha256": _sha(source),
        "data_contract_sha256": contract["contract_sha256"],
        "records": 100,
        "identity_sha256": identity_sha256,
        "roles": roles,
    }


def canonical_attempt03_calibration_marker_path(
    *, repo_root: str | Path, calibration_identity_sha256: str
) -> Path:
    identity = _require_sha(
        calibration_identity_sha256, "Attempt03 calibration identity_sha256"
    )
    return (
        Path(repo_root).resolve()
        / "outputs"
        / "hu_joint_policy"
        / "m43_attempt03_calibration_consumption"
        / identity
        / "M43_ATTEMPT03_SEALED_CALIBRATION_CONSUMED.json"
    )


def claim_attempt03_sealed_calibration(
    marker_path: str | Path,
    *,
    repo_root: str | Path,
    candidate_model_sha256: str,
    fit_bundle_sha256: str,
    fit_manifest_file_sha256: str,
    fit_manifest_canonical_sha256: str,
    precalibration_receipt_file_sha256: str,
    precalibration_receipt_sha256: str,
    attempt02_data_contract_file_sha256: str,
    attempt02_data_contract_sha256: str,
    calibration_identity_sha256: str,
    calibration_roles: Mapping[str, Any],
    sealed_calibration_lexical_path: str,
) -> dict[str, Any]:
    """Irreversibly claim the one allowed sealed-calibration consumption."""

    expected = canonical_attempt03_calibration_marker_path(
        repo_root=repo_root,
        calibration_identity_sha256=calibration_identity_sha256,
    )
    requested = Path(marker_path).resolve()
    if requested != expected:
        raise ValueError(
            "Attempt03 calibration marker must be the canonical identity-bound path"
        )
    payload: dict[str, Any] = {
        "schema": M43_ATTEMPT03_CALIBRATION_MARKER_SCHEMA,
        "status": "consumed_before_any_sealed_calibration_stat_hash_or_read",
        "candidate_model_sha256": _require_sha(
            candidate_model_sha256, "candidate model SHA-256"
        ),
        "fit_bundle_sha256": _require_sha(fit_bundle_sha256, "fit bundle SHA-256"),
        "fit_manifest_file_sha256": _require_sha(
            fit_manifest_file_sha256, "fit manifest file SHA-256"
        ),
        "fit_manifest_canonical_sha256": _require_sha(
            fit_manifest_canonical_sha256, "fit manifest canonical SHA-256"
        ),
        "precalibration_receipt_file_sha256": _require_sha(
            precalibration_receipt_file_sha256,
            "pre-calibration receipt file SHA-256",
        ),
        "precalibration_receipt_sha256": _require_sha(
            precalibration_receipt_sha256,
            "pre-calibration receipt canonical SHA-256",
        ),
        "attempt02_data_contract_file_sha256": _require_sha(
            attempt02_data_contract_file_sha256,
            "Attempt02 data contract file SHA-256",
        ),
        "attempt02_data_contract_sha256": _require_sha(
            attempt02_data_contract_sha256,
            "Attempt02 data contract canonical SHA-256",
        ),
        "sealed_calibration_identity_sha256": _require_sha(
            calibration_identity_sha256,
            "sealed calibration identity SHA-256",
        ),
        "calibration_roles": dict(calibration_roles),
        # Lexical normalization does not stat the sealed path.  Its bytes and
        # declared file hash are intentionally checked only after this marker
        # has been durably created.
        "sealed_calibration_lexical_path": sealed_calibration_lexical_path,
        "evaluation_pass_count": 1,
        "claim_is_consuming_even_on_crash": True,
        "sealed_calibration_stat_before_marker": False,
        "sealed_calibration_hash_before_marker": False,
        "sealed_calibration_read_before_marker": False,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["marker_sha256"] = canonical_manifest_sha256(payload)
    _write_json_exclusive(requested, payload)
    return payload


def evaluate_attempt03_precalibration_once(
    *,
    candidate_model_path: str | Path,
    fit_bundle_path: str | Path,
    fit_manifest_path: str | Path,
    one_shot_receive_receipt_path: str | Path,
    data_contract_path: str | Path,
    model_freeze_path: str | Path,
    training_freeze_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Claim the global marker before the first model-evaluation parse."""

    root = Path(repo_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Attempt03 pre-cal evaluation output is immutable")
    freeze_path = Path(model_freeze_path).resolve()
    load_attempt03_model_freeze(freeze_path, repo_root=root)
    training_freeze_path = Path(training_freeze_path).resolve()
    load_attempt03_training_freeze(
        training_freeze_path,
        repo_root=root,
        expected_model_freeze_path=freeze_path,
    )
    fit_manifest_path = Path(fit_manifest_path).resolve()
    fit_manifest = _load_fit_manifest(fit_manifest_path)
    candidate_path = Path(candidate_model_path).resolve()
    bundle_path = Path(fit_bundle_path).resolve()
    if (
        fit_manifest["model_sha256"] != _sha(candidate_path)
        or fit_manifest["fit_bundle_sha256"] != _sha(bundle_path)
        or fit_manifest["model_freeze_file_sha256"] != _sha(freeze_path)
        or fit_manifest["training_freeze_file_sha256"]
        != _sha(training_freeze_path)
    ):
        raise ValueError("Attempt03 fit candidate binding changed")
    model = HuM43JointModelV5.load(
        candidate_path, expected_sha256=fit_manifest["model_sha256"]
    )
    if model.safety_enabled or model.safety_estimator is not None:
        raise ValueError("Attempt03 pre-cal candidate must be fail-closed")
    fit_result = load_attempt03_fit_bundle(
        bundle_path, expected_sha256=fit_manifest["fit_bundle_sha256"]
    )
    receive_path = Path(one_shot_receive_receipt_path).resolve()
    receive_file_sha256 = _sha(receive_path)
    receive = _load_precal_receive_receipt(receive_path)
    if _sha(receive_path) != receive_file_sha256:
        raise ValueError("Attempt03 pre-cal receive receipt changed while loading")
    contract_path = Path(data_contract_path).resolve()
    contract_file_sha256 = _sha(contract_path)
    contract = _load_attempt03_data_contract(contract_path)
    if _sha(contract_path) != contract_file_sha256:
        raise ValueError("Attempt03 data contract changed while loading")
    if (
        _resolve_repo_bound_path(
            receive["data_contract"], root, "pre-cal receive data contract"
        )
        != contract_path
        or receive["data_contract_sha256"] != contract["contract_sha256"]
        or receive["model_freeze_file_sha256"] != _sha(freeze_path)
        or receive["training_freeze_file_sha256"]
        != _sha(training_freeze_path)
        or receive["candidate_model_sha256"] != fit_manifest["model_sha256"]
        or receive["fit_bundle_sha256"] != fit_manifest["fit_bundle_sha256"]
        or receive["fit_manifest_file_sha256"] != _sha(fit_manifest_path)
    ):
        raise ValueError("Attempt03 pre-cal receive binding changed")
    downstream = _mapping(
        receive.get("downstream_model_evaluation"),
        "downstream_model_evaluation",
    )
    marker_relative = str(contract["precal_holdout"]["consumption_marker"])
    if (
        downstream.get("global_consumption_marker") != marker_relative
        or downstream.get(
            "global_consumption_marker_must_be_created_exclusively_before_parse"
        )
        is not True
        or downstream.get("ready_for_exactly_one_bound_model_evaluation") is not True
        or downstream.get("model_evaluation_count") != 0
    ):
        raise ValueError("Attempt03 one-shot evaluation boundary changed")
    marker_path = (root / marker_relative).resolve()
    if root not in marker_path.parents:
        raise ValueError("Attempt03 global consumption marker escapes repo root")
    (
        raw_path,
        raw_sha256,
        precal_shard_paths,
        declared_precal_shard_binding,
    ) = _validate_precal_receive_data_binding(
        receive=receive,
        contract=contract,
        repo_root=root,
    )
    # Irreversible boundary: no JSONL line has been parsed above this point.
    marker = claim_attempt03_precalibration(
        marker_path,
        precal_identity_sha256=contract["precal_holdout"]["identity_sha256"],
        candidate_model_sha256=fit_manifest["model_sha256"],
        fit_manifest_sha256=_sha(fit_manifest_path),
        data_contract_sha256=contract["contract_sha256"],
        model_freeze_file_sha256=_sha(freeze_path),
        training_freeze_file_sha256=_sha(training_freeze_path),
    )
    rows = _validate_precal_content_binding(
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        shard_paths=precal_shard_paths,
        declared_shard_binding=declared_precal_shard_binding,
        repo_root=root,
    )
    if len(rows) != 200 or any(row.get("split") != "train" for row in rows):
        raise ValueError("Attempt03 pre-cal row count/split changed after claim")
    identities = {_row_identity(row, "Attempt03 pre-cal") for row in rows}
    if (
        len(identities) != 200
        or _identity_digest(identities)
        != contract["precal_holdout"]["identity_sha256"]
    ):
        raise ValueError("Attempt03 pre-cal identity changed after claim")
    profiles = Counter(
        str(_mapping(row.get("provenance"), "pre-cal provenance").get("root_profile"))
        for row in rows
    )
    samples = prepare_teacher_samples(rows)
    if any(
        sample.observation_fingerprint != row["observation_fingerprint"]
        for sample, row in zip(samples, rows, strict=True)
    ):
        raise ValueError("Attempt03 pre-cal observation fingerprint changed")
    report = evaluate_attempt03_precalibration(
        model,
        samples,
        fit_seed_values=fit_result.safety_dataset.seed_values,
        fit_observation_fingerprints=(
            fit_result.safety_dataset.observation_fingerprints
        ),
        profile_counts=profiles,
    )
    if (
        _sha(receive_path) != receive_file_sha256
        or _sha(contract_path) != contract_file_sha256
    ):
        raise ValueError("Attempt03 pre-cal hash-chain artifact changed during evaluation")
    report = {
        **report,
        "global_consumption_marker": str(marker_path),
        "global_consumption_marker_sha256": _sha(marker_path),
        "one_shot_receive_receipt_file_sha256": receive_file_sha256,
        "one_shot_receive_receipt_sha256": receive["receipt_sha256"],
        "data_contract_file_sha256": contract_file_sha256,
        "data_contract_sha256": contract["contract_sha256"],
    }
    receipt = build_attempt03_precalibration_receipt(
        report=report,
        marker=marker,
        candidate_model_sha256=fit_manifest["model_sha256"],
        fit_manifest_sha256=_sha(fit_manifest_path),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{os.getpid()}.precal"
    if staging.exists():
        raise FileExistsError("Attempt03 pre-cal staging exists")
    staging.mkdir()
    try:
        _write_json_exclusive(staging / "precalibration_report.json", report)
        _write_json_exclusive(staging / "precalibration_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": receipt["status"],
        "promotion_status": receipt["promotion_status"],
        "receipt": str(destination / "precalibration_receipt.json"),
        "receipt_file_sha256": _sha(destination / "precalibration_receipt.json"),
        "global_consumption_marker": str(marker_path),
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def calibrate_attempt03_after_precal_go(
    *,
    candidate_model_path: str | Path,
    fit_bundle_path: str | Path,
    fit_manifest_path: str | Path,
    precalibration_receipt_path: str | Path,
    attempt02_data_contract_path: str | Path,
    attempt02_train_path: str | Path,
    sealed_calibration_path: str | Path,
    calibration_consumption_marker_path: str | Path,
    model_freeze_path: str | Path,
    training_freeze_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """The sole path that may parse the inherited sealed calibration."""

    root = Path(repo_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Attempt03 calibration output is immutable")
    model_freeze_path = Path(model_freeze_path).resolve()
    load_attempt03_model_freeze(model_freeze_path, repo_root=root)
    training_freeze_path = Path(training_freeze_path).resolve()
    load_attempt03_training_freeze(
        training_freeze_path,
        repo_root=root,
        expected_model_freeze_path=model_freeze_path,
    )
    precal_path = Path(precalibration_receipt_path).resolve()
    precal = _read_mapping(precal_path, "Attempt03 pre-cal receipt")
    _validate_self_digest(precal, "receipt_sha256", "pre-cal receipt")
    if (
        precal.get("schema") != M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA
        or precal.get("status") != "go_precalibration"
        or precal.get("promotion_status")
        != "eligible_to_open_sealed_calibration"
        or precal.get("sealed_calibration_open_allowed") is not True
        or precal.get("sealed_calibration_opened") is not False
        or precal.get("inherited_locked_opened") is not False
    ):
        raise ValueError("Attempt03 sealed calibration requires bound pre-cal Go")
    fit_manifest_path = Path(fit_manifest_path).resolve()
    fit_manifest = _load_fit_manifest(fit_manifest_path)
    candidate_path = Path(candidate_model_path).resolve()
    bundle_path = Path(fit_bundle_path).resolve()
    if (
        precal.get("candidate_model_sha256") != _sha(candidate_path)
        or precal.get("fit_manifest_sha256") != _sha(fit_manifest_path)
        or fit_manifest["fit_bundle_sha256"] != _sha(bundle_path)
        or fit_manifest["model_freeze_file_sha256"] != _sha(model_freeze_path)
        or fit_manifest["training_freeze_file_sha256"]
        != _sha(training_freeze_path)
    ):
        raise ValueError("Attempt03 pre-cal/fitted candidate lineage changed")
    candidate = HuM43JointModelV5.load(
        candidate_path, expected_sha256=fit_manifest["model_sha256"]
    )
    fit_result = load_attempt03_fit_bundle(
        bundle_path, expected_sha256=fit_manifest["fit_bundle_sha256"]
    )
    declaration = _load_attempt02_sealed_calibration_declaration(
        attempt02_data_contract_path
    )
    sealed_lexical_path = os.path.abspath(os.fspath(sealed_calibration_path))
    # Irreversible boundary.  Neither Path.resolve/stat, a byte hash, nor a
    # JSONL read of sealed_calibration_path occurs above this call.  CreateNew
    # means a crash consumes the one allowed calibration use and every retry
    # fails closed before it can inspect the sealed file.
    calibration_marker = claim_attempt03_sealed_calibration(
        calibration_consumption_marker_path,
        repo_root=root,
        candidate_model_sha256=fit_manifest["model_sha256"],
        fit_bundle_sha256=fit_manifest["fit_bundle_sha256"],
        fit_manifest_file_sha256=_sha(fit_manifest_path),
        fit_manifest_canonical_sha256=fit_manifest["manifest_sha256"],
        precalibration_receipt_file_sha256=_sha(precal_path),
        precalibration_receipt_sha256=precal["receipt_sha256"],
        attempt02_data_contract_file_sha256=declaration[
            "data_contract_file_sha256"
        ],
        attempt02_data_contract_sha256=declaration["data_contract_sha256"],
        calibration_identity_sha256=declaration["identity_sha256"],
        calibration_roles=declaration["roles"],
        sealed_calibration_lexical_path=sealed_lexical_path,
    )
    calibration_marker_path = Path(calibration_consumption_marker_path).resolve()
    calibration_marker_file_sha256 = _sha(calibration_marker_path)
    # No sealed calibration JSONL is parsed until every Go/artifact binding is
    # validated and the exclusive identity-bound marker exists durably.
    roles = load_attempt02_calibration_roles(
        data_contract_path=attempt02_data_contract_path,
        train_path=attempt02_train_path,
        calibration_path=sealed_calibration_path,
        repo_root=repo_root,
    )
    config = Attempt03TrainingConfig()
    safety = fit_attempt03_safety_calibrator(
        candidate,
        fit_result.safety_dataset,
        roles.safety_fit,
        config=config,
    )
    threshold = select_attempt03_threshold(
        safety, roles.threshold_lock, config=config
    )
    if _sha(calibration_marker_path) != calibration_marker_file_sha256:
        raise ValueError("Attempt03 calibration consumption marker changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{os.getpid()}.cal"
    if staging.exists():
        raise FileExistsError("Attempt03 calibration staging exists")
    staging.mkdir()
    try:
        model_sha: str | None = None
        if threshold.report["status"] == "go":
            final_binding = {
                **dict(threshold.model.manifest),
                "status": "candidate_ready_for_freeze",
                "precalibration_receipt_file_sha256": _sha(precal_path),
                "calibration_binding": dict(roles.audit),
                "calibration_consumption_marker": str(calibration_marker_path),
                "calibration_consumption_marker_file_sha256": (
                    calibration_marker_file_sha256
                ),
                "calibration_consumption_marker_sha256": calibration_marker[
                    "marker_sha256"
                ],
                "model_freeze_file_sha256": _sha(model_freeze_path),
                "training_freeze_file_sha256": _sha(training_freeze_path),
                "locked_holdout_opened": False,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
                "full_replacement": False,
            }
            final_model = threshold.model.with_frozen_safety(
                threshold.model.safety_estimator,
                threshold=threshold.model.safety_threshold,
                enabled=True,
                manifest=final_binding,
            )
            threshold = replace(threshold, model=final_model)
            model_sha = final_model.save(staging / "model.pkl")
        manifest = build_attempt03_final_manifest(
            threshold=threshold,
            safety_fit=safety,
            precalibration_receipt=precal,
            model_sha256=model_sha,
        )
        manifest["calibration_binding"] = dict(roles.audit)
        manifest["calibration_consumption_marker"] = {
            "path": str(calibration_marker_path),
            "file_sha256": calibration_marker_file_sha256,
            "canonical_sha256": calibration_marker["marker_sha256"],
            "sealed_calibration_identity_sha256": declaration[
                "identity_sha256"
            ],
            "claim_is_consuming_even_on_crash": True,
        }
        manifest["fit_manifest_file_sha256"] = _sha(fit_manifest_path)
        manifest["model_freeze_file_sha256"] = _sha(model_freeze_path)
        manifest["training_freeze_file_sha256"] = _sha(training_freeze_path)
        manifest["manifest_sha256"] = canonical_manifest_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        _write_json_exclusive(staging / "training_manifest.json", manifest)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": manifest["status"],
        "promotion_status": manifest["promotion_status"],
        "model": str(destination / "model.pkl") if model_sha else None,
        "model_sha256": model_sha,
        "manifest": str(destination / "training_manifest.json"),
        "calibration_consumption_marker": str(calibration_marker_path),
        "calibration_consumption_marker_file_sha256": (
            calibration_marker_file_sha256
        ),
        "sealed_calibration_opened_after_precal_go": True,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }


def build_attempt03_model_threshold_freeze(
    *,
    model_path: str | Path,
    training_manifest_path: str | Path,
    model_freeze_path: str | Path,
    training_freeze_path: str | Path,
    attempt03_plan_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Freeze a Go model/threshold before any locked-holdout access."""

    root = Path(repo_root).resolve()
    executable_freeze_path = Path(model_freeze_path).resolve()
    executable_freeze = load_attempt03_model_freeze(
        executable_freeze_path, repo_root=root
    )
    training_freeze_path = Path(training_freeze_path).resolve()
    load_attempt03_training_freeze(
        training_freeze_path,
        repo_root=root,
        expected_model_freeze_path=executable_freeze_path,
    )
    plan_path = Path(attempt03_plan_path).resolve()
    plan = load_and_validate_attempt03_plan(plan_path)
    manifest_path = Path(training_manifest_path).resolve()
    manifest = _read_mapping(manifest_path, "Attempt03 final training manifest")
    _validate_self_digest(manifest, "manifest_sha256", "final training manifest")
    model_path = Path(model_path).resolve()
    if (
        manifest.get("schema") != "hu_m43_attempt03_v5_final_training_manifest_v1"
        or manifest.get("status") != "candidate_ready_for_freeze"
        or manifest.get("promotion_status") != "candidate_ready_for_freeze"
        or manifest.get("model_sha256") != _sha(model_path)
        or manifest.get("model_freeze_file_sha256")
        != _sha(executable_freeze_path)
        or manifest.get("training_freeze_file_sha256")
        != _sha(training_freeze_path)
        or manifest.get("locked_holdout")
        != {"status": "not_evaluated_pre_freeze", "opened": False}
        or any(manifest.get(key) is not False for key in (
            "current_profile_mutated",
            "runtime_policy_activated",
            "full_replacement",
        ))
    ):
        raise ValueError("Attempt03 final manifest is not freeze-eligible")
    model = HuM43JointModelV5.load(
        model_path, expected_sha256=manifest["model_sha256"]
    )
    if (
        not model.safety_enabled
        or model.safety_estimator is None
        or model.safety_threshold
        not in Attempt03TrainingConfig().thresholds
    ):
        raise ValueError("Attempt03 final model safety/threshold is not frozen")
    locked = _mapping(plan.get("inherited_locked"), "inherited_locked")
    if (
        locked.get("classification") != "inherited_unopened"
        or locked.get("content_open_before_frozen_model_and_threshold_allowed")
        is not False
    ):
        raise ValueError("Attempt03 inherited locked lifecycle changed")
    freeze = {
        "schema": M43_ATTEMPT03_MODEL_THRESHOLD_FREEZE_SCHEMA,
        "status": "frozen_before_inherited_locked_open",
        "model": {
            "path": str(model_path),
            "file_sha256": manifest["model_sha256"],
            "schema": model.schema,
            "model_id": model.model_id,
            "safety_enabled": True,
            "safety_threshold": model.safety_threshold,
        },
        "training_manifest": {
            "path": str(manifest_path),
            "file_sha256": _sha(manifest_path),
            "canonical_sha256": manifest["manifest_sha256"],
        },
        "executable_model_freeze": {
            "path": str(executable_freeze_path),
            "file_sha256": _sha(executable_freeze_path),
            "v5_implementation_sha256": executable_freeze["implementation"][
                "file_sha256"
            ],
        },
        "training_pipeline_freeze": {
            "path": str(training_freeze_path),
            "file_sha256": _sha(training_freeze_path),
        },
        "attempt03_plan": {
            "path": str(plan_path),
            "file_sha256": _sha(plan_path),
        },
        "inherited_locked": {
            "path": locked["path"],
            "records": locked["records"],
            "file_sha256": locked["file_sha256"],
            "identity_sha256": locked["identity_sha256"],
            "global_consumption_marker": locked["global_consumption_marker"],
            "content_opened": False,
            "model_evaluation_count": 0,
        },
        "threshold_reselection_after_freeze_allowed": False,
        "model_reselection_after_freeze_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    freeze["freeze_sha256"] = canonical_manifest_sha256(freeze)
    return freeze


def write_attempt03_model_threshold_freeze(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    payload = build_attempt03_model_threshold_freeze(**kwargs)
    _write_json_exclusive(Path(path), payload)
    return payload


def _load_fit_manifest(path: Path) -> dict[str, Any]:
    payload = _read_mapping(path, "Attempt03 fit manifest")
    _validate_self_digest(payload, "manifest_sha256", "fit manifest")
    if (
        payload.get("schema") != M43_ATTEMPT03_FIT_MANIFEST_SCHEMA
        or payload.get("status") != "fit_candidate_precalibration_unopened"
        or payload.get("promotion_status")
        != "not_eligible_precalibration_unopened"
        or payload.get("precalibration") != {"opened": False, "evaluated": False}
        or payload.get("sealed_calibration")
        != {"opened": False, "evaluated": False}
        or payload.get("inherited_locked") != {"opened": False, "evaluated": False}
        or any(payload.get(key) is not False for key in (
            "current_profile_mutated",
            "runtime_policy_activated",
            "full_replacement",
        ))
    ):
        raise ValueError("Attempt03 fit manifest lifecycle changed")
    return payload


def _load_precal_receive_receipt(path: str | Path) -> dict[str, Any]:
    payload = _read_mapping(Path(path), "Attempt03 pre-cal receive receipt")
    _validate_self_digest(
        payload,
        "receipt_sha256",
        "Attempt03 pre-cal receive receipt",
    )
    if (
        payload.get("schema")
        != "hu_m43_attempt03_teacher_precal_receive_receipt_v1"
        or payload.get("status")
        != "verified_structural_precal_open_after_frozen_claim"
        or payload.get("verified_shards") != 20
        or payload.get("verified_roots") != 200
        or any(payload.get(key) is not False for key in (
            "sealed_calibration_opened",
            "inherited_locked_opened",
            "current_profile_mutated",
            "runtime_policy_activated",
        ))
    ):
        raise ValueError("Attempt03 pre-cal receive lifecycle changed")
    for key in (
        "manifest_sha256",
        "schedule_sha256",
        "model_freeze_file_sha256",
        "training_freeze_file_sha256",
        "precal_open_authorization_file_sha256",
        "precal_open_authorization_sha256",
        "candidate_model_sha256",
        "fit_bundle_sha256",
        "fit_manifest_file_sha256",
        "fold_cloud_contract_file_sha256",
        "open_claim_file_sha256",
        "data_contract_sha256",
    ):
        _require_sha(payload.get(key), f"pre-cal receive {key}")
    if _sha(Path(str(payload["open_claim_path"]))) != payload["open_claim_file_sha256"]:
        raise ValueError("Attempt03 pre-cal open claim changed")
    authorization_path = Path(
        str(payload.get("precal_open_authorization_path"))
    ).resolve()
    if (
        _sha(authorization_path)
        != payload["precal_open_authorization_file_sha256"]
    ):
        raise ValueError("Attempt03 pre-cal open authorization changed")
    return payload


def _validate_precal_receive_data_binding(
    *,
    receive: Mapping[str, Any],
    contract: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, str, tuple[Path, ...], dict[str, Any]]:
    """Validate byte/path metadata without parsing a pre-cal JSONL row."""

    root = repo_root.resolve()
    teacher_shards = _mapping(
        contract.get("teacher_shards"), "Attempt03 teacher shards"
    )
    if set(teacher_shards) != {
        "schema",
        "roles",
        "all_fresh_roles_sha256",
    } or teacher_shards.get("schema") != (
        "hu_m43_attempt03_ordered_teacher_shards_v1"
    ):
        raise ValueError("Attempt03 teacher shard schema changed")
    _validate_self_digest(
        teacher_shards,
        "all_fresh_roles_sha256",
        "Attempt03 teacher shards",
    )
    if contract.get("teacher_shards_all_fresh_roles_sha256") != (
        teacher_shards["all_fresh_roles_sha256"]
    ):
        raise ValueError("Attempt03 teacher shard aggregate digest changed")
    roles = _mapping(teacher_shards.get("roles"), "Attempt03 teacher shard roles")
    if set(roles) != {"train.fit", "train.precal_holdout"}:
        raise ValueError("Attempt03 teacher shard role set changed")
    declared = dict(
        _mapping(
            roles.get("train.precal_holdout"),
            "Attempt03 pre-cal teacher shard binding",
        )
    )
    if set(declared) != {
        "ordered_shards",
        "records",
        "canonical_rows_sha256",
        "ordered_shards_sha256",
    }:
        raise ValueError("Attempt03 pre-cal teacher shard binding schema changed")
    _validate_self_digest(
        declared,
        "ordered_shards_sha256",
        "Attempt03 pre-cal teacher shard binding",
    )
    _require_sha(
        declared.get("canonical_rows_sha256"),
        "Attempt03 pre-cal canonical rows SHA-256",
    )
    ordered = declared.get("ordered_shards")
    if (
        not isinstance(ordered, list)
        or len(ordered) != 20
        or not _is_exact_integer(declared.get("records"), 200)
    ):
        raise ValueError("Attempt03 pre-cal ordered shard count changed")
    precal_manifest = _mapping(
        contract.get("precal_holdout"), "Attempt03 pre-cal manifest"
    )
    if (
        not _is_exact_integer(precal_manifest.get("records"), 200)
        or not _is_exact_integer(precal_manifest.get("shards"), 20)
        or precal_manifest.get("identity_sha256") is None
    ):
        raise ValueError("Attempt03 pre-cal manifest/shard binding changed")
    _require_sha(
        precal_manifest.get("identity_sha256"),
        "Attempt03 pre-cal identity SHA-256",
    )

    raw_spec = _mapping(
        receive.get("fresh_precal_holdout"), "fresh_precal_holdout"
    )
    if set(raw_spec) != {"path", "rows", "sha256", "shards"}:
        raise ValueError("Attempt03 received pre-cal artifact schema changed")
    raw_path = _resolve_repo_bound_path(
        raw_spec.get("path"), root, "received merged pre-cal"
    )
    raw_sha256 = _require_sha(
        raw_spec.get("sha256"), "received merged pre-cal SHA-256"
    )
    receipt_shards = raw_spec.get("shards")
    if (
        not _is_exact_integer(raw_spec.get("rows"), 200)
        or not isinstance(receipt_shards, list)
        or len(receipt_shards) != 20
        or not raw_path.is_file()
        or _sha(raw_path) != raw_sha256
    ):
        raise ValueError("Attempt03 received pre-cal artifact changed")

    shard_paths: list[Path] = []
    receipt_keys = {
        "shard",
        "logical_split",
        "split_shard",
        "rows",
        "sha256",
        "path",
    }
    contract_keys = {
        "index",
        "path",
        "bytes",
        "records",
        "file_sha256",
        "canonical_rows_sha256",
        "identity_sha256",
    }
    for index, (receipt_value, contract_value) in enumerate(
        zip(receipt_shards, ordered, strict=True)
    ):
        receipt_shard = _mapping(
            receipt_value, f"Attempt03 pre-cal receive shard {index}"
        )
        contract_shard = _mapping(
            contract_value, f"Attempt03 pre-cal contract shard {index}"
        )
        if set(receipt_shard) != receipt_keys or set(contract_shard) != contract_keys:
            raise ValueError("Attempt03 pre-cal shard record schema changed")
        receipt_sha = _require_sha(
            receipt_shard.get("sha256"),
            f"Attempt03 pre-cal receive shard {index} SHA-256",
        )
        contract_sha = _require_sha(
            contract_shard.get("file_sha256"),
            f"Attempt03 pre-cal contract shard {index} SHA-256",
        )
        _require_sha(
            contract_shard.get("canonical_rows_sha256"),
            f"Attempt03 pre-cal contract shard {index} canonical rows SHA-256",
        )
        _require_sha(
            contract_shard.get("identity_sha256"),
            f"Attempt03 pre-cal contract shard {index} identity SHA-256",
        )
        receipt_path = _resolve_repo_bound_path(
            receipt_shard.get("path"),
            root,
            f"Attempt03 pre-cal receive shard {index}",
        )
        contract_path = _resolve_repo_bound_path(
            contract_shard.get("path"),
            root,
            f"Attempt03 pre-cal contract shard {index}",
        )
        if (
            not _is_exact_integer(receipt_shard.get("shard"), 50 + index)
            or receipt_shard.get("logical_split") != "train.precal_holdout"
            or not _is_exact_integer(receipt_shard.get("split_shard"), index)
            or not _is_exact_integer(receipt_shard.get("rows"), 10)
            or not _is_exact_integer(contract_shard.get("index"), index)
            or not _is_exact_integer(contract_shard.get("records"), 10)
            or receipt_sha != contract_sha
            or receipt_path != contract_path
            or not receipt_path.is_file()
            or not _is_exact_integer(
                contract_shard.get("bytes"), receipt_path.stat().st_size
            )
            or _sha(receipt_path) != contract_sha
        ):
            raise ValueError(f"Attempt03 pre-cal shard {index} binding changed")
        shard_paths.append(receipt_path)
    return raw_path, raw_sha256, tuple(shard_paths), declared


def _validate_precal_content_binding(
    *,
    raw_path: Path,
    raw_sha256: str,
    shard_paths: Sequence[Path],
    declared_shard_binding: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """After the durable claim, recompute exact shard and merged row bindings."""

    actual = build_ordered_teacher_shard_binding(
        shard_paths,
        repo_root=repo_root,
        expected_split="train",
    )
    if actual != dict(declared_shard_binding):
        raise ValueError("Attempt03 pre-cal ordered shard content binding changed")
    if _sha(raw_path) != raw_sha256:
        raise ValueError("Attempt03 merged pre-cal changed before content parse")
    rows = read_teacher_jsonl(raw_path)
    if (
        _sha(raw_path) != raw_sha256
        or _canonical_rows_sha256(rows)
        != declared_shard_binding["canonical_rows_sha256"]
    ):
        raise ValueError("Attempt03 merged pre-cal canonical rows changed")
    return rows


def _load_attempt03_data_contract(path: Path) -> dict[str, Any]:
    payload = _read_mapping(path, "Attempt03 data contract")
    _validate_self_digest(payload, "contract_sha256", "Attempt03 data contract")
    if (
        payload.get("schema") != "hu_m43_attempt03_data_contract_v1"
        or payload.get("status")
        != "pass_fresh_fit_and_one_shot_precal_sealed_calibration_locked_unopened"
        or payload.get("fit", {}).get("records") != 700
        or payload.get("precal_holdout", {}).get("records") != 200
        or payload.get("precal_holdout", {}).get("model_evaluation_count") != 0
        or payload.get("sealed_calibration", {}).get("jsonl_content_parse_count") != 0
        or payload.get("inherited_locked", {}).get("jsonl_content_parse_count") != 0
    ):
        raise ValueError("Attempt03 data contract lifecycle changed")
    return payload


def _validate_self_digest(
    payload: Mapping[str, Any], field: str, label: str
) -> None:
    declared = _require_sha(payload.get(field), f"{label}.{field}")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError(f"{label} digest changed")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _is_exact_integer(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _resolve_repo_bound_path(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    source = Path(value)
    resolved = (source if source.is_absolute() else root / source).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes repo root") from exc
    return resolved


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("assemble-fit")
    fit.add_argument("--fold-artifacts-dir", required=True)
    fit.add_argument("--inherited-train", required=True)
    fit.add_argument("--fresh-train-fit", required=True)
    fit.add_argument("--fold-cloud-contract", required=True)
    fit.add_argument("--model-freeze", required=True)
    fit.add_argument("--training-freeze", required=True)
    fit.add_argument("--repo-root", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--run-name", required=True)
    fit.add_argument("--source-sha256", required=True)
    fit.add_argument("--run-manifest-sha256", required=True)

    precal = sub.add_parser("evaluate-precal")
    precal.add_argument("--candidate-model", required=True)
    precal.add_argument("--fit-bundle", required=True)
    precal.add_argument("--fit-manifest", required=True)
    precal.add_argument("--one-shot-receive-receipt", required=True)
    precal.add_argument("--data-contract", required=True)
    precal.add_argument("--model-freeze", required=True)
    precal.add_argument("--training-freeze", required=True)
    precal.add_argument("--repo-root", required=True)
    precal.add_argument("--output-dir", required=True)

    authorize = sub.add_parser("authorize-precal-open")
    authorize.add_argument("--candidate-model", required=True)
    authorize.add_argument("--fit-bundle", required=True)
    authorize.add_argument("--fit-manifest", required=True)
    authorize.add_argument("--fold-cloud-contract", required=True)
    authorize.add_argument("--model-freeze", required=True)
    authorize.add_argument("--training-freeze", required=True)
    authorize.add_argument("--repo-root", required=True)
    authorize.add_argument("--output", required=True)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--candidate-model", required=True)
    calibrate.add_argument("--fit-bundle", required=True)
    calibrate.add_argument("--fit-manifest", required=True)
    calibrate.add_argument("--precalibration-receipt", required=True)
    calibrate.add_argument("--attempt02-data-contract", required=True)
    calibrate.add_argument("--attempt02-train", required=True)
    calibrate.add_argument("--sealed-calibration", required=True)
    calibrate.add_argument("--calibration-consumption-marker", required=True)
    calibrate.add_argument("--model-freeze", required=True)
    calibrate.add_argument("--training-freeze", required=True)
    calibrate.add_argument("--repo-root", required=True)
    calibrate.add_argument("--output-dir", required=True)

    freeze = sub.add_parser("freeze-candidate")
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--training-manifest", required=True)
    freeze.add_argument("--model-freeze", required=True)
    freeze.add_argument("--training-freeze", required=True)
    freeze.add_argument("--attempt03-plan", required=True)
    freeze.add_argument("--repo-root", required=True)
    freeze.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "assemble-fit":
        result = assemble_attempt03_fit_candidate(
            fold_artifacts_dir=args.fold_artifacts_dir,
            inherited_train_path=args.inherited_train,
            fresh_train_fit_path=args.fresh_train_fit,
            fold_cloud_contract_path=args.fold_cloud_contract,
            model_freeze_path=args.model_freeze,
            training_freeze_path=args.training_freeze,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            run_name=args.run_name,
            source_sha256=args.source_sha256,
            run_manifest_sha256=args.run_manifest_sha256,
        )
    elif args.command == "evaluate-precal":
        result = evaluate_attempt03_precalibration_once(
            candidate_model_path=args.candidate_model,
            fit_bundle_path=args.fit_bundle,
            fit_manifest_path=args.fit_manifest,
            one_shot_receive_receipt_path=args.one_shot_receive_receipt,
            data_contract_path=args.data_contract,
            model_freeze_path=args.model_freeze,
            training_freeze_path=args.training_freeze,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
        )
    elif args.command == "authorize-precal-open":
        result = write_attempt03_precal_open_authorization(
            args.output,
            candidate_model_path=args.candidate_model,
            fit_bundle_path=args.fit_bundle,
            fit_manifest_path=args.fit_manifest,
            fold_cloud_contract_path=args.fold_cloud_contract,
            model_freeze_path=args.model_freeze,
            training_freeze_path=args.training_freeze,
            repo_root=args.repo_root,
        )
    elif args.command == "calibrate":
        result = calibrate_attempt03_after_precal_go(
            candidate_model_path=args.candidate_model,
            fit_bundle_path=args.fit_bundle,
            fit_manifest_path=args.fit_manifest,
            precalibration_receipt_path=args.precalibration_receipt,
            attempt02_data_contract_path=args.attempt02_data_contract,
            attempt02_train_path=args.attempt02_train,
            sealed_calibration_path=args.sealed_calibration,
            calibration_consumption_marker_path=(
                args.calibration_consumption_marker
            ),
            model_freeze_path=args.model_freeze,
            training_freeze_path=args.training_freeze,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
        )
    else:
        result = write_attempt03_model_threshold_freeze(
            args.output,
            model_path=args.model,
            training_manifest_path=args.training_manifest,
            model_freeze_path=args.model_freeze,
            training_freeze_path=args.training_freeze,
            attempt03_plan_path=args.attempt03_plan,
            repo_root=args.repo_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "assemble_attempt03_fit_candidate",
    "build_attempt03_precal_open_authorization",
    "build_attempt03_model_threshold_freeze",
    "calibrate_attempt03_after_precal_go",
    "canonical_attempt03_calibration_marker_path",
    "claim_attempt03_sealed_calibration",
    "evaluate_attempt03_precalibration_once",
    "main",
    "parse_args",
    "write_attempt03_precal_open_authorization",
    "write_attempt03_model_threshold_freeze",
]
