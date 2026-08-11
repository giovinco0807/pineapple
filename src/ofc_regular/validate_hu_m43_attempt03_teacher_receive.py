"""Fail-closed validation for M4.3 Attempt03 Spot teacher artifacts.

The PowerShell receiver deliberately addresses every GCS object by its exact
URI.  This module validates the downloaded immutable closure and each shard
before it is admitted to either the fit-only or one-shot pre-calibration
output.  It has no cloud access and never resolves the ``current`` profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit_hu_m4_t1_data import read_and_audit_shard
from .hu_m43_attempt03_contract import load_and_validate_attempt03_plan


_ROLES = ("train.fit", "train.precal_holdout")
_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_ROLE_START = {"train.fit": 0, "train.precal_holdout": 50}
_ROOTS_PER_SHARD = 10
_AMENDED_FREEZE_SHA256 = (
    "8aed10b143172c9d3b2a98fca199e4fbe4324f5956406fea51c2d17fcb614b9d"
)
_ORIGINAL_FREEZE_SHA256 = (
    "ce47b111d923be163075f135164599415c5446cc71e9bd46eda8af4b93eb7e2f"
)
_FREEZE_LINEAGE_SHA256 = (
    "b93999e230c4e94c4e62eedad1f15c6076b19f7a010eeeaf9a815305cda95276"
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: str | Path, location: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON mapping")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_self_digest(
    value: Mapping[str, Any], *, field: str, location: str
) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{location} {field} is not a SHA-256 digest")
    actual = _canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if claimed != actual:
        raise ValueError(f"{location} {field} mismatch")
    return claimed


def _resolve_under_root(root: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} path is missing")
    path = Path(value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{location} escapes the repository root")
    return path


def validate_attempt03_model_freeze_lineage(
    *,
    repo_root: str | Path,
    model_freeze_path: str | Path,
    original_freeze_path: str | Path,
    freeze_lineage_path: str | Path,
) -> dict[str, Any]:
    """Recompute the amendment's semantic and pre-first-row evidence."""

    root = Path(repo_root).resolve()
    amended_path = Path(model_freeze_path).resolve()
    original_path = Path(original_freeze_path).resolve()
    lineage_path = Path(freeze_lineage_path).resolve()
    for path, label in (
        (amended_path, "amended freeze"),
        (original_path, "original freeze"),
        (lineage_path, "freeze lineage"),
    ):
        if path != root and root not in path.parents:
            raise ValueError(f"Attempt03 {label} escapes repository root")
    if _file_sha256(amended_path) != _AMENDED_FREEZE_SHA256:
        raise ValueError("Attempt03 amended model freeze SHA changed")
    if _file_sha256(original_path) != _ORIGINAL_FREEZE_SHA256:
        raise ValueError("Attempt03 original model freeze SHA changed")
    if _file_sha256(lineage_path) != _FREEZE_LINEAGE_SHA256:
        raise ValueError("Attempt03 model freeze lineage SHA changed")

    lineage = _read_mapping(lineage_path, "Attempt03 freeze lineage")
    original = _read_mapping(original_path, "Attempt03 original freeze")
    amended = _read_mapping(amended_path, "Attempt03 amended freeze")
    if (
        lineage.get("schema") != "hu_m43_attempt03_model_freeze_lineage_v1"
        or lineage.get("status")
        != "pass_administrative_amendment_precedes_first_teacher_row"
    ):
        raise ValueError("Attempt03 model freeze lineage status changed")
    original_binding = _mapping(lineage.get("original_freeze"), "original freeze")
    amended_binding = _mapping(lineage.get("amended_freeze"), "amended freeze")
    if (
        (root / str(original_binding.get("path"))).resolve() != original_path
        or original_binding.get("file_sha256") != _ORIGINAL_FREEZE_SHA256
        or (root / str(amended_binding.get("path"))).resolve() != amended_path
        or amended_binding.get("file_sha256") != _AMENDED_FREEZE_SHA256
    ):
        raise ValueError("Attempt03 model freeze lineage path/hash changed")

    projection = _mapping(lineage.get("semantic_projection"), "semantic projection")
    fields = projection.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Attempt03 semantic projection fields changed")
    try:
        original_projection = {field: original[field] for field in fields}
        amended_projection = {field: amended[field] for field in fields}
    except (KeyError, TypeError) as exc:
        raise ValueError("Attempt03 semantic projection is incomplete") from exc
    original_projection_sha = _canonical_sha256(original_projection)
    amended_projection_sha = _canonical_sha256(amended_projection)
    changed_keys = sorted(
        key
        for key in set(original) | set(amended)
        if original.get(key) != amended.get(key)
    )
    if (
        original_projection != amended_projection
        or original_projection_sha != projection.get("original_sha256")
        or amended_projection_sha != projection.get("amended_sha256")
        or projection.get("equal") is not True
        or changed_keys != amended_binding.get("changed_top_level_keys_only")
        or original.get("parent_plan", {}).get("file_sha256")
        != projection.get("parent_plan_file_sha256_original")
        or amended.get("parent_plan", {}).get("file_sha256")
        != projection.get("parent_plan_file_sha256_amended")
    ):
        raise ValueError("Attempt03 model freeze semantic projection changed")

    boundary = _mapping(
        lineage.get("first_teacher_row_boundary"), "first teacher row boundary"
    )
    evidence_path = (root / str(boundary.get("evidence_path"))).resolve()
    if root not in evidence_path.parents:
        raise ValueError("Attempt03 first-row evidence escapes repository root")
    if _file_sha256(evidence_path) != boundary.get("evidence_file_sha256"):
        raise ValueError("Attempt03 first-row evidence SHA changed")
    checkpoint = _read_mapping(evidence_path, "Attempt03 first-row checkpoint")
    if checkpoint.get("completed_roots") != boundary.get("completed_roots") or (
        boundary.get("completed_roots") != 1
    ):
        raise ValueError("Attempt03 first-row checkpoint boundary changed")
    amended_time = datetime.fromisoformat(
        str(amended_binding.get("filesystem_last_write_time_evidence"))
    )
    declared_first_time = datetime.fromisoformat(str(boundary.get("completed_at")))
    observed_first_time = datetime.fromtimestamp(
        float(checkpoint["updated_unix_seconds"]), tz=timezone.utc
    )
    if amended_time.tzinfo is None or declared_first_time.tzinfo is None:
        raise ValueError("Attempt03 freeze lineage timestamps lost timezone")
    if abs((declared_first_time - observed_first_time).total_seconds()) > 0.001:
        raise ValueError("Attempt03 first-row timestamp evidence changed")
    recomputed_seconds = (declared_first_time - amended_time).total_seconds()
    if (
        recomputed_seconds <= 0.0
        or abs(
            recomputed_seconds
            - float(boundary.get("amended_freeze_precedes_first_row_seconds"))
        )
        > 1e-6
    ):
        raise ValueError("Attempt03 amendment no longer precedes first row")
    holdout = _mapping(
        lineage.get("holdout_boundary_at_amendment"), "holdout boundary"
    )
    if any(value != 0 for value in holdout.values()) or any(
        lineage.get(key) is not False
        for key in (
            "current_profile_mutated",
            "runtime_policy_activated",
            "full_replacement_enabled",
        )
    ):
        raise ValueError("Attempt03 freeze lineage lifecycle changed")
    return {
        "amended_freeze_file_sha256": _AMENDED_FREEZE_SHA256,
        "original_freeze_file_sha256": _ORIGINAL_FREEZE_SHA256,
        "freeze_lineage_file_sha256": _FREEZE_LINEAGE_SHA256,
        "semantic_projection_sha256": amended_projection_sha,
        "amendment_precedes_first_row_seconds": recomputed_seconds,
    }


def validate_precal_open_authorization(
    *,
    repo_root: str | Path,
    authorization_path: str | Path,
    model_freeze_path: str | Path,
    original_freeze_path: str | Path,
    freeze_lineage_path: str | Path,
) -> dict[str, Any]:
    """Validate authorization and re-hash every named fit artifact.

    This function has no pre-calibration input and cannot address a teacher
    result shard.  It is safe to run before the receiver's exclusive claim.
    """

    root = Path(repo_root).resolve()
    authorization_source = Path(authorization_path).resolve()
    if root not in authorization_source.parents:
        raise ValueError("Attempt03 pre-cal authorization escapes repository root")
    lineage = validate_attempt03_model_freeze_lineage(
        repo_root=root,
        model_freeze_path=model_freeze_path,
        original_freeze_path=original_freeze_path,
        freeze_lineage_path=freeze_lineage_path,
    )
    authorization = _read_mapping(
        authorization_source, "Attempt03 pre-cal open authorization"
    )
    expected_keys = {
        "schema",
        "status",
        "decision_basis",
        "row_valued_metric_used_for_authorization",
        "precalibration_path_received",
        "precalibration_content_read",
        "candidate_model_path",
        "candidate_model_sha256",
        "fit_bundle_path",
        "fit_bundle_sha256",
        "fit_manifest_path",
        "fit_manifest_file_sha256",
        "fit_manifest_canonical_sha256",
        "fold_cloud_contract_path",
        "fold_cloud_contract_file_sha256",
        "fold_cloud_contract_canonical_sha256",
        "training_config_sha256",
        "model_freeze_path",
        "model_freeze_file_sha256",
        "training_freeze_path",
        "training_freeze_file_sha256",
        "v5_implementation_sha256",
        "stage18_model_sha256",
        "fit_manifest_status_required",
        "candidate_safety_enabled",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement",
        "authorization_sha256",
    }
    if set(authorization) != expected_keys:
        raise ValueError("Attempt03 pre-cal authorization key set changed")
    authorization_sha = _validate_self_digest(
        authorization,
        field="authorization_sha256",
        location="Attempt03 pre-cal authorization",
    )
    if (
        authorization.get("schema")
        != "hu_m43_attempt03_precal_open_authorization_v1"
        or authorization.get("status")
        != "authorized_to_create_receiver_open_claim"
        or authorization.get("decision_basis")
        != "frozen_schema_and_hash_chain_only"
        or authorization.get("fit_manifest_status_required")
        != "fit_candidate_precalibration_unopened"
        or authorization.get("row_valued_metric_used_for_authorization") is not False
        or authorization.get("precalibration_path_received") is not False
        or authorization.get("precalibration_content_read") is not False
        or authorization.get("candidate_safety_enabled") is not False
        or any(
            authorization.get(key) is not False
            for key in (
                "current_profile_mutated",
                "runtime_policy_activated",
                "full_replacement",
            )
        )
    ):
        raise ValueError("Attempt03 pre-cal authorization lifecycle changed")

    artifact_paths = {
        key: _resolve_under_root(root, authorization.get(key), key)
        for key in (
            "candidate_model_path",
            "fit_bundle_path",
            "fit_manifest_path",
            "fold_cloud_contract_path",
            "model_freeze_path",
            "training_freeze_path",
        )
    }
    if artifact_paths["model_freeze_path"] != Path(model_freeze_path).resolve():
        raise ValueError("Attempt03 authorization names a different model freeze")
    hash_bindings = {
        "candidate_model_path": "candidate_model_sha256",
        "fit_bundle_path": "fit_bundle_sha256",
        "fit_manifest_path": "fit_manifest_file_sha256",
        "fold_cloud_contract_path": "fold_cloud_contract_file_sha256",
        "model_freeze_path": "model_freeze_file_sha256",
        "training_freeze_path": "training_freeze_file_sha256",
    }
    for path_key, hash_key in hash_bindings.items():
        if _file_sha256(artifact_paths[path_key]) != authorization.get(hash_key):
            raise ValueError(f"Attempt03 authorization actual hash changed: {path_key}")
    if authorization.get("model_freeze_file_sha256") != _AMENDED_FREEZE_SHA256:
        raise ValueError("Attempt03 authorization is not bound to amended freeze")

    fit_manifest = _read_mapping(
        artifact_paths["fit_manifest_path"], "Attempt03 fit manifest"
    )
    fit_manifest_sha = _validate_self_digest(
        fit_manifest, field="manifest_sha256", location="Attempt03 fit manifest"
    )
    contract = _read_mapping(
        artifact_paths["fold_cloud_contract_path"], "Attempt03 fold contract"
    )
    contract_sha = _validate_self_digest(
        contract, field="contract_sha256", location="Attempt03 fold contract"
    )
    training_freeze = _read_mapping(
        artifact_paths["training_freeze_path"], "Attempt03 training freeze"
    )
    training_freeze_sha = _validate_self_digest(
        training_freeze,
        field="freeze_sha256",
        location="Attempt03 training freeze",
    )
    if (
        fit_manifest.get("schema") != "hu_m43_attempt03_v5_fit_manifest_v1"
        or fit_manifest.get("status") != "fit_candidate_precalibration_unopened"
        or fit_manifest.get("model_sha256")
        != authorization.get("candidate_model_sha256")
        or fit_manifest.get("fit_bundle_sha256")
        != authorization.get("fit_bundle_sha256")
        or fit_manifest_sha != authorization.get("fit_manifest_canonical_sha256")
        or fit_manifest.get("fold_cloud_contract_sha256") != contract_sha
        or fit_manifest.get("training_config_sha256")
        != authorization.get("training_config_sha256")
        or fit_manifest.get("model_freeze_file_sha256") != _AMENDED_FREEZE_SHA256
        or fit_manifest.get("training_freeze_file_sha256")
        != authorization.get("training_freeze_file_sha256")
    ):
        raise ValueError("Attempt03 authorization fit-manifest chain changed")
    if (
        contract.get("schema")
        != "hu_m43_attempt03_v5_fold_cloud_contract_v1"
        or contract.get("status") != "frozen_fit700_only"
        or contract_sha
        != authorization.get("fold_cloud_contract_canonical_sha256")
        or contract.get("training_config_sha256")
        != authorization.get("training_config_sha256")
        or contract.get("model_freeze", {}).get("file_sha256")
        != _AMENDED_FREEZE_SHA256
        or contract.get("training_freeze", {}).get("file_sha256")
        != authorization.get("training_freeze_file_sha256")
    ):
        raise ValueError("Attempt03 authorization fold-contract chain changed")
    if (
        training_freeze.get("schema")
        != "hu_m43_attempt03_training_pipeline_freeze_v1"
        or training_freeze.get("status")
        != (
            "frozen_after_one_structural_train_fit_canary_before_"
            "row_valued_design_or_any_holdout_open"
        )
        or training_freeze.get("parent_model_freeze", {}).get("file_sha256")
        != _AMENDED_FREEZE_SHA256
        or training_freeze_sha != training_freeze.get("freeze_sha256")
    ):
        raise ValueError("Attempt03 authorization training-freeze chain changed")
    if training_freeze.get("decision_boundary") != {
        "cloud_teacher_rows_may_exist": True,
        "attempt03_fit_rows_received_locally": 1,
        "attempt03_fit_rows_structurally_inspected": 1,
        "attempt03_fit_jsonl_content_parse_count": 1,
        "attempt03_fit_row_valued_labels_or_metrics_used_for_design": 0,
        "precalibration_rows_opened": 0,
        "sealed_calibration_rows_opened": 0,
        "inherited_locked_rows_opened": 0,
        "source_or_config_selected_from_row_values": False,
    }:
        raise ValueError("Attempt03 training-freeze decision boundary changed")
    return {
        "schema": "hu_m43_attempt03_precal_open_authorization_audit_v1",
        "status": "pass_without_precalibration_access",
        "authorization_file_sha256": _file_sha256(authorization_source),
        "authorization_sha256": authorization_sha,
        "candidate_model_sha256": authorization["candidate_model_sha256"],
        "fit_bundle_sha256": authorization["fit_bundle_sha256"],
        "fit_manifest_file_sha256": authorization["fit_manifest_file_sha256"],
        "fold_cloud_contract_file_sha256": authorization[
            "fold_cloud_contract_file_sha256"
        ],
        "training_freeze_file_sha256": authorization[
            "training_freeze_file_sha256"
        ],
        "freeze_lineage": lineage,
        "precalibration_path_received": False,
        "precalibration_content_read": False,
        "current_profile_resolved": False,
    }


def load_and_validate_schedule(
    *, plan_path: str | Path, schedule_path: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate all public schedule metadata without opening result objects."""

    plan = load_and_validate_attempt03_plan(plan_path)
    specs: list[dict[str, Any]] = []
    with Path(schedule_path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"schedule line {line_number} is not a mapping")
            specs.append(value)
    if len(specs) != 70:
        raise ValueError("Attempt03 schedule must contain exactly 70 shards")

    output_prefixes: set[str] = set()
    for global_shard, spec in enumerate(specs):
        role = "train.fit" if global_shard < 50 else "train.precal_holdout"
        split_shard = global_shard - _ROLE_START[role]
        role_plan = _mapping(plan["fresh_splits"][role], role)
        expected = {
            "schema": "hu_m43_attempt03_teacher_shard_v1",
            "shard": global_shard,
            "logical_split": role,
            "split": "train",
            "split_shard": split_shard,
            "roots": _ROOTS_PER_SHARD,
            "seed_start": int(role_plan["seed_start"])
            + split_shard
            * _ROOTS_PER_SHARD
            * int(role_plan["seed_stride"]),
            "seed_stride": int(role_plan["seed_stride"]),
            "candidate_seed": int(role_plan["candidate_seed_start"])
            + split_shard * int(role_plan["seed_stride"]),
            "evaluation_seed": int(role_plan["evaluation_seed_start"])
            + split_shard * int(role_plan["seed_stride"]),
            "child_policy_seed": int(role_plan["child_policy_seed_start"])
            + split_shard * int(role_plan["seed_stride"]),
            "candidate_samples": 2,
            "evaluation_samples": 64,
        }
        for key, wanted in expected.items():
            if spec.get(key) != wanted:
                raise ValueError(
                    f"Attempt03 shard {global_shard} schedule mismatch: {key}"
                )
        expected_slug = "train_fit" if role == "train.fit" else "precal_holdout"
        expected_prefix = (
            f"{expected_slug}_shard_{split_shard:03d}_roots10_"
            f"seed{expected['seed_start']}"
        )
        if spec.get("output_prefix") != expected_prefix:
            raise ValueError(
                f"Attempt03 shard {global_shard} output prefix mismatch"
            )
        prefix = str(spec["output_prefix"])
        if prefix in output_prefixes:
            raise ValueError("Attempt03 schedule contains duplicate output prefixes")
        output_prefixes.add(prefix)
        quota = _mapping(
            spec.get("profile_quota_per_shard"),
            f"shard {global_shard} profile quota",
        )
        if dict(quota) != {profile: 2 for profile in _PROFILES}:
            raise ValueError(
                f"Attempt03 shard {global_shard} profile quota changed"
            )
    return plan, specs


def validate_closure(
    *,
    plan_path: str | Path,
    manifest_path: str | Path,
    schedule_path: str | Path,
    source_path: str | Path,
    startup_path: str | Path,
    model_manifest_path: str | Path,
    native_manifest_path: str | Path,
    run_name: str,
    project_id: str,
    bucket: str,
) -> dict[str, Any]:
    plan, specs = load_and_validate_schedule(
        plan_path=plan_path, schedule_path=schedule_path
    )
    manifest = _read_mapping(manifest_path, "Attempt03 cloud manifest")
    expected_identity = {
        "schema": "hu_m43_attempt03_teacher_spot_manifest_v1",
        "run_name": run_name,
        "project_id": project_id,
        "bucket": bucket,
        "milestone": "M4.3-attempt03",
        "purpose": "fresh_train_fit_and_sealed_one_shot_precal_teacher",
        "total_roots": 700,
        "total_shards": 70,
        "roots_per_shard": 10,
        "record_split": "train",
        "candidate_samples": 2,
        "evaluation_samples": 64,
        "spot": True,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "sealed_calibration_uploaded": False,
        "locked_uploaded": False,
    }
    for key, wanted in expected_identity.items():
        if manifest.get(key) != wanted:
            raise ValueError(f"Attempt03 cloud manifest mismatch: {key}")
    if dict(_mapping(manifest.get("logical_split_roots"), "split roots")) != {
        "train.fit": 500,
        "train.precal_holdout": 200,
    }:
        raise ValueError("Attempt03 logical split roots changed")
    if manifest.get("plan_file_sha256") != _file_sha256(plan_path):
        raise ValueError("Attempt03 plan hash binding changed")
    closure_paths = {
        "schedule_sha256": schedule_path,
        "source_sha256": source_path,
        "startup_sha256": startup_path,
        "model_manifest_sha256": model_manifest_path,
        "native_manifest_sha256": native_manifest_path,
    }
    closure_hashes: dict[str, str] = {}
    for field, path in closure_paths.items():
        actual = _file_sha256(path)
        if manifest.get(field) != actual:
            raise ValueError(f"Attempt03 immutable closure hash mismatch: {field}")
        closure_hashes[field] = actual
    inherited = _mapping(
        manifest.get("inherited_teacher_closure"), "inherited closure"
    )
    if dict(inherited) != {
        "source_run": "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201",
        "algorithm_unchanged": True,
    }:
        raise ValueError("Attempt03 inherited teacher closure changed")
    if len(specs) != int(plan["budget"]["fresh_shards"]):
        raise ValueError("Attempt03 plan/schedule shard count mismatch")
    return {
        "schema": "hu_m43_attempt03_teacher_closure_audit_v1",
        "status": "pass",
        "run_name": run_name,
        "manifest_sha256": _file_sha256(manifest_path),
        **closure_hashes,
        "total_shards": len(specs),
        "train_fit_shards": 50,
        "precal_shards": 20,
        "precal_result_content_opened": False,
        "current_profile_resolved": False,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"teacher row {line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"teacher row {line_number} is not a mapping")
            rows.append(value)
    return rows


def _validate_summary(
    summary: Mapping[str, Any], *, spec: Mapping[str, Any], run_name: str
) -> None:
    if summary.get("schema") != "hu_m4_t1_second_shard_v1":
        raise ValueError("Attempt03 generator summary schema changed")
    if summary.get("status") != "complete":
        raise ValueError("Attempt03 generator summary is not complete")
    if summary.get("current_profile_resolved") is not False:
        raise ValueError("Attempt03 generator resolved current profile")
    if summary.get("teacher_value_status") != "diagnostic_not_match_EV":
        raise ValueError("Attempt03 teacher value status changed")
    if summary.get("roots") != spec["roots"]:
        raise ValueError("Attempt03 generator root count changed")
    config = _mapping(summary.get("config"), "generator config")
    expected = {
        "baseline_profile": "stage18_p1",
        "t2_profile": "stage9f_p2",
        "batch_child_selectors": True,
        "native_batch_threads": 4,
        "opening_lookahead_samples": 0,
        "roots": spec["roots"],
        "seed_start": spec["seed_start"],
        "seed_stride": spec["seed_stride"],
        "candidate_seed": spec["candidate_seed"],
        "evaluation_seed": spec["evaluation_seed"],
        "child_policy_seed": spec["child_policy_seed"],
        "candidate_samples": 2,
        "evaluation_samples": 64,
        "split": "train",
        "run_id": f"{run_name}:shard={spec['shard']}",
        "root_profile": "stage3_baseline",
        "root_profiles": list(_PROFILES),
        "root_profile_weights": [1.0] * 5,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(f"Attempt03 generator config mismatch: {key}")
    population = _mapping(
        summary.get("root_population_manifest"), "root population manifest"
    )
    if population.get("schema") != "weighted_quota_seeded_shuffle_v1":
        raise ValueError("Attempt03 root population schedule schema changed")
    profile_rows = population.get("profiles")
    if not isinstance(profile_rows, list) or len(profile_rows) != 5:
        raise ValueError("Attempt03 root population summary is incomplete")
    observed: dict[str, tuple[int, int]] = {}
    for row in profile_rows:
        entry = _mapping(row, "root population profile")
        observed[str(entry.get("profile"))] = (
            int(entry.get("target_count", -1)),
            int(entry.get("completed_count", -1)),
        )
    if observed != {profile: (2, 2) for profile in _PROFILES}:
        raise ValueError("Attempt03 per-shard root profile quota changed")


def validate_shard(
    *,
    plan_path: str | Path,
    manifest_path: str | Path,
    schedule_path: str | Path,
    shard: int,
    teacher_path: str | Path,
    done_path: str | Path,
    checkpoint_path: str | Path,
    heartbeat_path: str | Path,
    generator_summary_path: str | Path,
    run_name: str,
) -> dict[str, Any]:
    _, specs = load_and_validate_schedule(
        plan_path=plan_path, schedule_path=schedule_path
    )
    if shard < 0 or shard >= len(specs):
        raise ValueError("Attempt03 shard index is outside the schedule")
    spec = specs[shard]
    manifest = _read_mapping(manifest_path, "Attempt03 cloud manifest")
    if manifest.get("run_name") != run_name:
        raise ValueError("Attempt03 run identity changed")
    done = _read_mapping(done_path, "Attempt03 DONE")
    # Attempt03 deliberately reuses the byte-pinned Attempt02 teacher worker,
    # so its DONE schema remains Attempt02 while every hash and schedule field
    # is rebound to the Attempt03 run.
    expected_done = {
        "schema": "hu_m43_attempt02_teacher_done_v1",
        "status": "complete",
        "run_name": run_name,
        "shard": shard,
        "split": "train",
        "roots": spec["roots"],
        "output_prefix": spec["output_prefix"],
        "manifest_sha256": _file_sha256(manifest_path),
        "shards_manifest_sha256": _file_sha256(schedule_path),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "model_manifest_sha256": manifest["model_manifest_sha256"],
        "native_manifest_sha256": manifest["native_manifest_sha256"],
        "output_sha256": _file_sha256(teacher_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "heartbeat_sha256": _file_sha256(heartbeat_path),
    }
    for key, wanted in expected_done.items():
        if done.get(key) != wanted:
            raise ValueError(f"Attempt03 shard {shard} DONE mismatch: {key}")

    summaries = [
        _read_mapping(checkpoint_path, "Attempt03 checkpoint"),
        _read_mapping(heartbeat_path, "Attempt03 heartbeat"),
        _read_mapping(generator_summary_path, "Attempt03 generator summary"),
    ]
    for summary in summaries:
        _validate_summary(summary, spec=spec, run_name=run_name)
        if summary.get("output_sha256") != expected_done["output_sha256"]:
            raise ValueError("Attempt03 summary/output hash mismatch")
    if summaries[0] != summaries[1] or summaries[0] != summaries[2]:
        raise ValueError("Attempt03 completion summaries disagree")

    structural = read_and_audit_shard(
        teacher_path, expected_split="train", require_paired_delta=True
    )
    if int(structural.get("records", -1)) != spec["roots"]:
        raise ValueError("Attempt03 teacher structural row count changed")
    if int(structural.get("paired_delta_records", -1)) != spec["roots"]:
        raise ValueError("Attempt03 teacher paired-delta coverage is incomplete")
    rows = _read_jsonl(teacher_path)
    if len(rows) != spec["roots"]:
        raise ValueError("Attempt03 teacher line count changed")
    expected_seeds = {
        spec["seed_start"] + offset * spec["seed_stride"]
        for offset in range(spec["roots"])
    }
    observed_seeds = {int(row.get("hand_seed", -1)) for row in rows}
    if observed_seeds != expected_seeds:
        raise ValueError("Attempt03 teacher hand-seed schedule changed")
    profiles: Counter[str] = Counter()
    for row in rows:
        if (
            row.get("schema") != "hu_m4_t1_second_training_sample_v2"
            or row.get("split") != "train"
            or row.get("seat") != "second"
            or row.get("street") != "T1"
            or row.get("to_act_order") != "second"
        ):
            raise ValueError("Attempt03 teacher row role/schema changed")
        provenance = _mapping(row.get("provenance"), "teacher provenance")
        if provenance.get("current_profile_resolved") is not False:
            raise ValueError("Attempt03 teacher row resolved current profile")
        if (
            provenance.get("baseline_profile") != "stage18_p1"
            or provenance.get("t2_profile") != "stage9f_p2"
            or provenance.get("native_batch_threads") != 4
        ):
            raise ValueError("Attempt03 teacher row continuation changed")
        profile = str(provenance.get("root_profile", ""))
        if profile not in _PROFILES:
            raise ValueError("Attempt03 teacher row has an unknown root profile")
        profiles[profile] += 1
        search = _mapping(row.get("search_config"), "teacher search config")
        expected_search = {
            "candidate_samples": 2,
            "evaluation_samples": 64,
            "candidate_seed": spec["candidate_seed"],
            "evaluation_seed": spec["evaluation_seed"],
            "child_policy_seed": spec["child_policy_seed"],
        }
        for key, wanted in expected_search.items():
            if search.get(key) != wanted:
                raise ValueError(f"Attempt03 teacher row search mismatch: {key}")
    if dict(profiles) != {profile: 2 for profile in _PROFILES}:
        raise ValueError("Attempt03 teacher per-shard profile quota changed")
    identity_rows = sorted(
        (int(row["hand_seed"]), str(row["observation_fingerprint"]))
        for row in rows
    )
    identity_sha256 = hashlib.sha256(
        json.dumps(identity_rows, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "hu_m43_attempt03_teacher_shard_audit_v1",
        "status": "pass",
        "run_name": run_name,
        "shard": shard,
        "logical_split": spec["logical_split"],
        "split_shard": spec["split_shard"],
        "roots": len(rows),
        "output_prefix": spec["output_prefix"],
        "output_sha256": expected_done["output_sha256"],
        "checkpoint_sha256": expected_done["checkpoint_sha256"],
        "heartbeat_sha256": expected_done["heartbeat_sha256"],
        "generator_summary_sha256": _file_sha256(generator_summary_path),
        "identity_sha256": identity_sha256,
        "profile_counts": dict(sorted(profiles.items())),
        "candidate_samples": 2,
        "evaluation_samples": 64,
        "current_profile_resolved": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    closure = subparsers.add_parser("validate-closure")
    closure.add_argument("--plan", required=True)
    closure.add_argument("--manifest", required=True)
    closure.add_argument("--schedule", required=True)
    closure.add_argument("--source", required=True)
    closure.add_argument("--startup", required=True)
    closure.add_argument("--model-manifest", required=True)
    closure.add_argument("--native-manifest", required=True)
    closure.add_argument("--run-name", required=True)
    closure.add_argument("--project-id", required=True)
    closure.add_argument("--bucket", required=True)
    shard = subparsers.add_parser("validate-shard")
    shard.add_argument("--plan", required=True)
    shard.add_argument("--manifest", required=True)
    shard.add_argument("--schedule", required=True)
    shard.add_argument("--shard", required=True, type=int)
    shard.add_argument("--teacher", required=True)
    shard.add_argument("--done", required=True)
    shard.add_argument("--checkpoint", required=True)
    shard.add_argument("--heartbeat", required=True)
    shard.add_argument("--generator-summary", required=True)
    shard.add_argument("--run-name", required=True)
    authorization = subparsers.add_parser("validate-precal-open")
    authorization.add_argument("--repo-root", required=True)
    authorization.add_argument("--authorization", required=True)
    authorization.add_argument("--model-freeze", required=True)
    authorization.add_argument("--original-freeze", required=True)
    authorization.add_argument("--freeze-lineage", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "validate-closure":
        payload = validate_closure(
            plan_path=args.plan,
            manifest_path=args.manifest,
            schedule_path=args.schedule,
            source_path=args.source,
            startup_path=args.startup,
            model_manifest_path=args.model_manifest,
            native_manifest_path=args.native_manifest,
            run_name=args.run_name,
            project_id=args.project_id,
            bucket=args.bucket,
        )
    elif args.command == "validate-shard":
        payload = validate_shard(
            plan_path=args.plan,
            manifest_path=args.manifest,
            schedule_path=args.schedule,
            shard=args.shard,
            teacher_path=args.teacher,
            done_path=args.done,
            checkpoint_path=args.checkpoint,
            heartbeat_path=args.heartbeat,
            generator_summary_path=args.generator_summary,
            run_name=args.run_name,
        )
    else:
        payload = validate_precal_open_authorization(
            repo_root=args.repo_root,
            authorization_path=args.authorization,
            model_freeze_path=args.model_freeze,
            original_freeze_path=args.original_freeze,
            freeze_lineage_path=args.freeze_lineage,
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover
    main()
