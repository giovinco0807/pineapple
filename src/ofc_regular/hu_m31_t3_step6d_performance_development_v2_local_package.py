"""Immutable local package gate for the T3 performance-development v2 probe.

This module is intentionally unable to launch or mutate cloud resources.  It
rehashes the accepted Candidate02 inputs, builds two source-isolated tail role
manifests, and publishes a self-validating local directory.  The published
startup file is a fail-closed marker rather than a cloud startup program; a
later, separately reviewed slice must replace that boundary before execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_full100_plan as full100
from . import hu_m31_t3_step6d_performance_development_v2_contract as contract_v1
from . import hu_m31_t3_step6d_performance_development_v2_preflight as preflight
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as selector


PACKAGE_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_local_package_v2"
)
PACKAGE_READY_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_local_package_ready_v2"
)
SOURCE_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_source_manifest_v2"
)
RUNTIME_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_runtime_manifest_v2"
)
TOOLING_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_tooling_manifest_v1"
)
TAIL_SEED_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_tail_seed_schedule_v2"
)
PACKAGE_STATUS = "package_ready_local_not_authorized"

MANIFEST_NAME = "MANIFEST.json"
READY_NAME = "PACKAGE_READY.json"
SOURCE_MANIFEST_PATH = "manifests/source.json"
RUNTIME_MANIFEST_PATH = "manifests/runtime.json"
TOOLING_MANIFEST_PATH = "manifests/tooling.json"
REFERENCE_PACKAGE_PATH = (
    "payload/source/native/reference/release/libofc_hu_m3_engine.so"
)
CANDIDATE_PACKAGE_PATH = (
    "payload/source/native/candidate/release/libofc_hu_m3_engine.so"
)
FEATURE_PACKAGE_PATH = (
    "payload/source/target/release/libofc_stage3_feature_encoder.so"
)
PLAN_PACKAGE_PATH = "payload/source/frozen/full100_plan.json"
SUMMARY_PACKAGE_PATH = "payload/source/frozen/accepted_summary.json"
VALIDATION_PACKAGE_PATH = "payload/source/frozen/accepted_validation.json"
DRY_RECEIPT_PACKAGE_PATH = "payload/source/frozen/live_dry_run_receipt.json"
TAIL_SELECTION_MANIFEST_PACKAGE_PATH = (
    "payload/source/frozen/candidate02_tail_v2_selection.json"
)
ROOT_PACKAGE_DIR = "payload/source/frozen/full100_roots"
TAIL_SEED_PACKAGE_PATH = "payload/runtime/tail_seed_schedule.json"
STARTUP_PACKAGE_PATH = "payload/runtime/startup_perfdev_v2_local_only.sh"
ROLE_PACKAGE_PATHS = {
    "candidate": "payload/runtime/roles/candidate.json",
    "reference": "payload/runtime/roles/reference.json",
}

FEATURE_LIBRARY_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
MAX_RECEIPT_AGE_SECONDS = 300
MAX_FUTURE_SKEW_SECONDS = 30
REARM2_STARTUP_SHA256 = (
    "9a2fab31412bac1a9a422a2194a93035b262598c9ac1cb17c27b9963481f728d"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = contract_v1.DEFAULT_FULL100_PLAN_PATH
DEFAULT_ACCEPTED_SUMMARY_PATH = contract_v1.DEFAULT_ACCEPTED_MERGE_DIR / "summary.json"
DEFAULT_ACCEPTED_VALIDATION_PATH = (
    contract_v1.DEFAULT_ACCEPTED_MERGE_DIR / "validation.json"
)
DEFAULT_TAIL_SELECTION_MANIFEST_PATH = (
    contract_v1.DEFAULT_TAIL_SELECTION_MANIFEST_PATH
)
DEFAULT_ROOT_DIR = full100.DEFAULT_ROOT_DIR
DEFAULT_CANDIDATE_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "candidate02_frozen/"
    "libofc_hu_m3_engine_candidate02_4050e04b22d7943d.so"
)
DEFAULT_REFERENCE_PATH = (
    _REPO_ROOT
    / "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/"
    "package_src/native/release/libofc_hu_m3_engine.so"
)
DEFAULT_FEATURE_PATH = (
    _REPO_ROOT
    / "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/"
    "package_src/target/release/libofc_stage3_feature_encoder.so"
)
DEFAULT_DRY_RECEIPT_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/"
    "perfdev_v2_live_preflight_20260722_001/dry_run_receipt.json"
)
DEFAULT_REARM2_ROOT_DIR = Path(
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-rearm2-20260718-001/"
    "package/package_src/frozen/full100_roots"
)

_CONFIG_PATHS = (
    "configs/hu_joint_policy_m31_t3_step6d_contract.json",
    "configs/hu_m43_attempt08_runtime_requirements.txt",
)
_MODEL_PATHS = (
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
)

_RUN_NAME = re.compile(
    r"regular-hu-m31-c02-perfdev-v2-[a-z0-9][a-z0-9-]{7,47}"
)
_FORBIDDEN_PATH_TERMS = (
    "performance_lock",
    "performance-lock",
    "rearm2",
    "lock-r2",
    "recovery-v3",
)
_STARTUP_BYTES = (
    b"#!/bin/sh\n"
    b"set -eu\n"
    b"echo 'perfdev-v2 local package is not cloud executable or launch authorized' >&2\n"
    b"exit 97\n"
)


@dataclass(frozen=True)
class _AuditedInputs:
    inventory: dict[str, Any]
    paths: dict[str, Path]
    plan: dict[str, Any]
    summary: dict[str, Any]
    validation: dict[str, Any]
    receipt: dict[str, Any] | None
    roots: tuple[dict[str, Any], ...]
    root_records: tuple[dict[str, Any], ...]
    rearm2_root_records: tuple[dict[str, Any], ...]
    tail_seed_schedule: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    return contract_v1.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return contract_v1.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    return contract_v1.sha256_file(path)


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _reject_link_components(path: Path, *, include_leaf: bool = True) -> None:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    if not parts:
        raise ValueError("unsafe empty path")
    current = Path(parts[0])
    limit = len(parts) if include_leaf else max(1, len(parts) - 1)
    for part in parts[1:limit]:
        current /= part
        if current.exists() and _is_reparse_or_link(current):
            raise ValueError(f"symlink or reparse-point path is forbidden: {current}")


def _safe_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    _reject_link_components(raw)
    target = raw.resolve(strict=True)
    if not target.is_file() or _is_reparse_or_link(target):
        raise ValueError(f"{label} is missing or unsafe")
    return target


def _safe_directory(path: str | Path, label: str) -> Path:
    raw = Path(path)
    _reject_link_components(raw)
    target = raw.resolve(strict=True)
    if not target.is_dir() or _is_reparse_or_link(target):
        raise ValueError(f"{label} is missing or unsafe")
    return target


def _read_canonical(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    target = _safe_file(path, label)
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return target, value


def _entry(path: Path, package_path: str) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "package_path": package_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_receipt_against_inputs(
    value: Mapping[str, Any],
    *,
    plan_path: Path,
    summary_path: Path,
    validation_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    payload = dict(value)
    frozen_contract = contract_v1.validate_contract(
        payload.get("contract", {}),
        full100_plan_path=plan_path,
        accepted_summary_path=summary_path,
        accepted_validation_path=validation_path,
        tail_selection_manifest_path=selection_path,
    )
    image_record = payload.get("image_preflight")
    runtime_record = payload.get("runtime_preflight")
    if not isinstance(image_record, Mapping) or not isinstance(runtime_record, Mapping):
        raise ValueError("dry-run receipt preflight records are missing")
    image = preflight.build_image_preflight(image_record.get("observation", {}))
    runtime = preflight.build_runtime_preflight(runtime_record.get("observation", {}))
    expected = preflight.build_dry_run_receipt_unchecked(
        contract=frozen_contract,
        image_preflight=image,
        runtime_preflight=runtime,
    )
    if payload != expected:
        raise ValueError("performance-development v2 dry-run receipt changed")
    return payload


def _receipt_unix_seconds(receipt: Mapping[str, Any]) -> int:
    observed = receipt["runtime_preflight"]["observation"]["observed_at_utc"]
    if not isinstance(observed, str) or not observed.endswith("Z"):
        raise ValueError("dry-run receipt timestamp changed")
    try:
        parsed = datetime.fromisoformat(observed[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("dry-run receipt timestamp is invalid") from exc
    return int(parsed.timestamp())


def _check_receipt_freshness(
    receipt: Mapping[str, Any], *, now_unix_seconds: int | None
) -> tuple[int, int]:
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise ValueError("freshness timestamp changed")
    observed = _receipt_unix_seconds(receipt)
    age = now - observed
    if age > MAX_RECEIPT_AGE_SECONDS or age < -MAX_FUTURE_SKEW_SECONDS:
        raise ValueError("live dry-run receipt is stale or future-dated")
    return now, age


def _seed_values(
    bases: Mapping[str, int], *, stride: int, indices: Iterable[int]
) -> set[int]:
    return {
        int(base) + stride * int(index)
        for index in indices
        for base in bases.values()
    }


def _build_tail_seed_schedule() -> dict[str, Any]:
    development = runner.candidate02_seed_contract()
    if development.get("seed_set_sha256") != contract_v1.DEVELOPMENT_SEED_SET_SHA256:
        raise ValueError("development all-100 seed digest changed")
    rearm2 = runner.candidate02_performance_lock_recovery_v3_seed_contract()
    if rearm2.get("seed_set_sha256") != contract_v1.REARM2_SEED_SET_SHA256:
        raise ValueError("rearm2 seed digest changed")
    stride = int(development["seed_stride"])
    dev_all = _seed_values(
        development["namespace_bases"],
        stride=stride,
        indices=runner.CONTRACT_HAND_INDICES,
    )
    rearm_all = _seed_values(
        rearm2["namespace_bases"],
        stride=int(rearm2["seed_stride"]),
        indices=runner.CONTRACT_HAND_INDICES,
    )
    if (
        len(dev_all) != 600
        or len(rearm_all) != 600
        or dev_all & rearm_all
        or canonical_sha256(sorted(dev_all))
        != contract_v1.DEVELOPMENT_SEED_SET_SHA256
        or runner.contract_canonical_sha256(sorted(rearm_all))
        != contract_v1.REARM2_SEED_SET_SHA256
    ):
        raise ValueError("development/rearm2 seed overlap is not zero")
    rows = [
        {
            "hand_index": index,
            "seeds": runner.candidate02_seed_values(index),
        }
        for index in contract_v1.TAIL_HAND_INDICES
    ]
    return {
        "schema": TAIL_SEED_SCHEMA,
        "hand_indices": list(contract_v1.TAIL_HAND_INDICES),
        "namespace_bases": dict(development["namespace_bases"]),
        "seed_stride": stride,
        "rows": rows,
        "tail_seed_count": len(rows) * len(development["namespace_bases"]),
        "tail_seed_set_sha256": canonical_sha256(
            sorted(seed for row in rows for seed in row["seeds"].values())
        ),
        "all100_seed_set_sha256": contract_v1.DEVELOPMENT_SEED_SET_SHA256,
        "rearm2_seed_set_sha256": contract_v1.REARM2_SEED_SET_SHA256,
        "rearm2_seed_overlap_count": 0,
        "all_values_unique": True,
        "training_eligible": False,
    }


def _validate_tail_seed_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = _build_tail_seed_schedule()
    payload = dict(value)
    if payload != expected:
        raise ValueError("tail seed schedule changed")
    return payload


def _rearm2_root_overlap(
    development_file_hashes: Iterable[str] | None = None,
    rearm2_file_hashes: Iterable[str] | None = None,
) -> dict[str, Any]:
    development = {
        contract_v1.DEVELOPMENT_ROOT_SET_SHA256,
        contract_v1.DEVELOPMENT_ROOT_TOPOLOGY_SHA256,
    }
    rearm2 = {
        contract_v1.REARM2_ROOT_MATERIALIZATION_SHA256,
        contract_v1.REARM2_ROOT_SEAL_SHA256,
        contract_v1.REARM2_AGGREGATE_ROOT_SHA256,
        contract_v1.REARM2_ROOT_TOPOLOGY_SHA256,
    }
    overlap = development & rearm2
    if overlap:
        raise ValueError("development/rearm2 root identity overlap is not zero")
    individual_compared = (
        development_file_hashes is not None and rearm2_file_hashes is not None
    )
    individual_overlap_count: int | None = None
    if individual_compared:
        dev_files = list(development_file_hashes or ())
        rearm_files = list(rearm2_file_hashes or ())
        individual_overlap = set(dev_files) & set(rearm_files)
        if (
            len(dev_files) != 100
            or len(rearm_files) != 100
            or len(set(dev_files)) != 100
            or len(set(rearm_files)) != 100
            or canonical_sha256(dev_files)
            != contract_v1.DEVELOPMENT_ROOT_SET_SHA256
            or canonical_sha256(rearm_files)
            != contract_v1.REARM2_AGGREGATE_ROOT_SHA256
            or individual_overlap
        ):
            raise ValueError("development/rearm2 individual root overlap is not zero")
        individual_overlap_count = 0
    return {
        "comparison_scope": "aggregate_topology_identity_tokens",
        "development_token_count": len(development),
        "rearm2_token_count": len(rearm2),
        "overlap_count": 0,
        "individual_rearm2_root_hashes_compared": individual_compared,
        "individual_file_count_each": 100 if individual_compared else 0,
        "individual_file_hash_overlap_count": individual_overlap_count,
    }


def _forbidden_namespace_or_path(value: str) -> bool:
    folded = value.replace("\\", "/").casefold()
    return any(term in folded for term in _FORBIDDEN_PATH_TERMS)


def _audit_inputs(
    *,
    candidate_library: str | Path,
    reference_library: str | Path,
    feature_encoder: str | Path,
    full100_plan_path: str | Path,
    accepted_summary_path: str | Path,
    accepted_validation_path: str | Path,
    tail_selection_manifest_path: str | Path,
    root_dir: str | Path,
    rearm2_root_dir: str | Path,
    dry_run_receipt_path: str | Path | None,
) -> _AuditedInputs:
    candidate = _safe_file(candidate_library, "candidate library")
    reference = _safe_file(reference_library, "reference library")
    feature = _safe_file(feature_encoder, "feature encoder")
    plan_path, plan_raw = _read_canonical(full100_plan_path, "full100 plan")
    summary_path, summary = _read_canonical(
        accepted_summary_path, "accepted full100 summary"
    )
    validation_path, validation = _read_canonical(
        accepted_validation_path, "accepted full100 validation"
    )
    selection_path, selection = _read_canonical(
        tail_selection_manifest_path, "candidate02 tail-v2 selection manifest"
    )
    contract_v1.validate_tail_selection_manifest(selection_path)
    root_target = _safe_directory(root_dir, "frozen all-100 root directory")
    rearm2_root_target = _safe_directory(
        rearm2_root_dir, "rearm2 denylist root directory"
    )

    expected_hashes = {
        "candidate": contract_v1.CANDIDATE_LIBRARY_SHA256,
        "reference": contract_v1.REFERENCE_LIBRARY_SHA256,
        "feature": FEATURE_LIBRARY_SHA256,
        "plan": contract_v1.FULL100_PLAN_SHA256,
        "summary": contract_v1.ACCEPTED_SUMMARY_SHA256,
        "validation": contract_v1.ACCEPTED_VALIDATION_SHA256,
        "selection": contract_v1.TAIL_SELECTION_MANIFEST_SHA256,
    }
    observed_hashes = {
        "candidate": sha256_file(candidate),
        "reference": sha256_file(reference),
        "feature": sha256_file(feature),
        "plan": sha256_file(plan_path),
        "summary": sha256_file(summary_path),
        "validation": sha256_file(validation_path),
        "selection": sha256_file(selection_path),
    }
    if observed_hashes != expected_hashes or candidate == reference:
        raise ValueError("accepted binary or result artifact digest changed")

    plan = full100.validate_full100_plan(plan_raw)
    frozen_contract = contract_v1.build_contract(
        full100_plan_path=plan_path,
        accepted_summary_path=summary_path,
        accepted_validation_path=validation_path,
        tail_selection_manifest_path=selection_path,
    )
    if plan["run_contract_digest"] != contract_v1.FULL_RUN_CONTRACT_DIGEST:
        raise ValueError("accepted full100 run contract changed")
    roots = tuple(selector.load_frozen_roots(root_target))
    topology = selector.topology_rows(roots)
    root_records: list[dict[str, Any]] = []
    for index in runner.CONTRACT_HAND_INDICES:
        path = _safe_file(root_target / f"hand_{index:03d}.json", "frozen root")
        root_records.append(
            _entry(path, f"{ROOT_PACKAGE_DIR}/hand_{index:03d}.json")
        )
    expected_rearm2_names = [
        f"hand_{index:03d}.json" for index in runner.CONTRACT_HAND_INDICES
    ]
    observed_rearm2_names = sorted(
        path.name for path in rearm2_root_target.glob("hand_*.json")
    )
    if observed_rearm2_names != expected_rearm2_names:
        raise ValueError("rearm2 denylist requires exactly roots 000..099")
    rearm2_root_records: list[dict[str, Any]] = []
    for index in runner.CONTRACT_HAND_INDICES:
        path = _safe_file(
            rearm2_root_target / f"hand_{index:03d}.json", "rearm2 denylist root"
        )
        rearm2_root_records.append(
            {
                "hand_index": index,
                "source_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    aggregate = canonical_sha256([canonical_sha256(root) for root in roots])
    topology_digest = canonical_sha256(topology)
    if (
        aggregate != contract_v1.DEVELOPMENT_ROOT_SET_SHA256
        or topology_digest != contract_v1.DEVELOPMENT_ROOT_TOPOLOGY_SHA256
        or plan["root_set"]["all100_root_sha256"] != aggregate
        or plan["root_set"]["topology_sha256"] != topology_digest
    ):
        raise ValueError("all-100 root aggregate or topology digest changed")

    receipt: dict[str, Any] | None = None
    receipt_path: Path | None = None
    if dry_run_receipt_path is not None:
        receipt_path, raw_receipt = _read_canonical(
            dry_run_receipt_path, "live dry-run receipt"
        )
        receipt = _validate_receipt_against_inputs(
            raw_receipt,
            plan_path=plan_path,
            summary_path=summary_path,
            validation_path=validation_path,
            selection_path=selection_path,
        )

    seed_schedule = _build_tail_seed_schedule()
    root_overlap = _rearm2_root_overlap(
        (record["sha256"] for record in root_records),
        (record["sha256"] for record in rearm2_root_records),
    )
    paths = {
        "candidate": candidate,
        "reference": reference,
        "feature": feature,
        "plan": plan_path,
        "summary": summary_path,
        "validation": validation_path,
        "selection": selection_path,
        "roots": root_target,
        "rearm2_roots": rearm2_root_target,
    }
    if receipt_path is not None:
        paths["receipt"] = receipt_path
    primary = {
        "candidate": _entry(candidate, CANDIDATE_PACKAGE_PATH),
        "reference": _entry(reference, REFERENCE_PACKAGE_PATH),
        "feature": _entry(feature, FEATURE_PACKAGE_PATH),
        "plan": _entry(plan_path, PLAN_PACKAGE_PATH),
        "summary": _entry(summary_path, SUMMARY_PACKAGE_PATH),
        "validation": _entry(validation_path, VALIDATION_PACKAGE_PATH),
        "selection": _entry(
            selection_path, TAIL_SELECTION_MANIFEST_PACKAGE_PATH
        ),
    }
    if receipt_path is not None:
        primary["receipt"] = _entry(receipt_path, DRY_RECEIPT_PACKAGE_PATH)
    inventory = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "status": "source_inventory_rehashed_local_only",
        "accepted_contract_sha256": canonical_sha256(frozen_contract),
        "full100_plan_run_contract_digest": contract_v1.FULL_RUN_CONTRACT_DIGEST,
        "run_contract_digest": contract_v1.TAIL_RUN_CONTRACT_DIGEST,
        "selection_manifest_sha256": contract_v1.TAIL_SELECTION_MANIFEST_SHA256,
        "selection_manifest_bytes": contract_v1.TAIL_SELECTION_MANIFEST_BYTES,
        "primary": primary,
        "root_directory": str(root_target),
        "roots": root_records,
        "root_count": len(root_records),
        "root_bytes": sum(record["bytes"] for record in root_records),
        "root_set_sha256": aggregate,
        "root_topology_sha256": topology_digest,
        "rearm2_denylist_roots": {
            "root_directory": str(rearm2_root_target),
            "root_count": len(rearm2_root_records),
            "root_bytes": sum(record["bytes"] for record in rearm2_root_records),
            "ordered_root_sha256": contract_v1.REARM2_AGGREGATE_ROOT_SHA256,
            "records": rearm2_root_records,
        },
        "tail_seed_schedule_sha256": canonical_sha256(seed_schedule),
        "all100_seed_set_sha256": contract_v1.DEVELOPMENT_SEED_SET_SHA256,
        "rearm2_overlap": {
            "roots": root_overlap,
            "seeds": {"compared_count_each": 600, "overlap_count": 0},
        },
        "source_files_rehashed_now": True,
        "source_paths_are_current_candidates_not_historical_caller_path_claims": True,
        "cloud_mutated": False,
        "cloud_executable": False,
        "launch_authorized": False,
    }
    return _AuditedInputs(
        inventory=inventory,
        paths=paths,
        plan=plan,
        summary=summary,
        validation=validation,
        receipt=receipt,
        roots=roots,
        root_records=tuple(root_records),
        rearm2_root_records=tuple(rearm2_root_records),
        tail_seed_schedule=seed_schedule,
    )


def audit_source_inventory(
    *,
    candidate_library: str | Path = DEFAULT_CANDIDATE_PATH,
    reference_library: str | Path = DEFAULT_REFERENCE_PATH,
    feature_encoder: str | Path = DEFAULT_FEATURE_PATH,
    full100_plan_path: str | Path = DEFAULT_PLAN_PATH,
    accepted_summary_path: str | Path = DEFAULT_ACCEPTED_SUMMARY_PATH,
    accepted_validation_path: str | Path = DEFAULT_ACCEPTED_VALIDATION_PATH,
    tail_selection_manifest_path: str | Path = DEFAULT_TAIL_SELECTION_MANIFEST_PATH,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    rearm2_root_dir: str | Path = DEFAULT_REARM2_ROOT_DIR,
    dry_run_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rehash real source candidates without creating a package."""

    return _audit_inputs(
        candidate_library=candidate_library,
        reference_library=reference_library,
        feature_encoder=feature_encoder,
        full100_plan_path=full100_plan_path,
        accepted_summary_path=accepted_summary_path,
        accepted_validation_path=accepted_validation_path,
        tail_selection_manifest_path=tail_selection_manifest_path,
        root_dir=root_dir,
        rearm2_root_dir=rearm2_root_dir,
        dry_run_receipt_path=dry_run_receipt_path,
    ).inventory


def _normalize_package_relative(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "\\"))
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in value
    ):
        raise ValueError(f"unsafe package relative path: {value!r}")
    return pure.as_posix()


def _default_tooling_sources(repository_root: Path) -> list[tuple[str, Path]]:
    source_root = _safe_directory(repository_root / "src/ofc_regular", "source tree")
    values: list[tuple[str, Path]] = []
    for path in sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix()):
        safe = _safe_file(path, "Python tooling source")
        relative = safe.relative_to(repository_root).as_posix()
        values.append((relative, safe))
    for relative in (*_CONFIG_PATHS, *_MODEL_PATHS):
        values.append((relative, _safe_file(repository_root / relative, "runtime tooling")))
    return values


def _tooling_specs(
    *,
    repository_root: Path,
    tooling_sources: Sequence[tuple[str, str | Path]] | None,
    fixture_only: bool,
) -> list[tuple[str, Path]]:
    if tooling_sources is not None and not fixture_only:
        raise ValueError("custom tooling sources are fixture-only")
    raw = (
        _default_tooling_sources(repository_root)
        if tooling_sources is None
        else [(relative, _safe_file(path, "fixture tooling source")) for relative, path in tooling_sources]
    )
    seen: set[str] = set()
    result: list[tuple[str, Path]] = []
    for relative, source in raw:
        checked = _normalize_package_relative(str(relative))
        package_relative = f"payload/tooling/{checked}"
        if package_relative in seen:
            raise ValueError(f"duplicate package relative path: {package_relative}")
        seen.add(package_relative)
        result.append((package_relative, source))
    if not result:
        raise ValueError("tooling source set must not be empty")
    return sorted(result, key=lambda row: row[0])


def _write_bytes_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable package entry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_once(path: Path, value: Any) -> None:
    _write_bytes_once(path, canonical_bytes(value))


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    _safe_file(source, "package source")
    if destination.exists():
        raise FileExistsError(f"duplicate package relative path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    observed = digest.hexdigest()
    if observed != expected_sha256 or count != expected_bytes:
        raise ValueError("source changed while package entry was copied")
    if sha256_file(destination) != expected_sha256 or destination.stat().st_size != count:
        raise ValueError("packaged bytes differ from source bytes")
    return {"sha256": observed, "bytes": count}


def _file_record(path: Path, *, group: str) -> dict[str, Any]:
    return {"group": group, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _manifest_entries(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"path": path, "sha256": record["sha256"], "bytes": record["bytes"]}
        for path, record in sorted(records.items())
    ]


def _aggregate_entries(records: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_sha256(_manifest_entries(records))


def _package_hash_denylist() -> set[str]:
    return {
        contract_v1.REARM2_PACKAGE_MANIFEST_SHA256,
        contract_v1.REARM2_PACKAGE_READY_SHA256,
        contract_v1.REARM2_PACKAGE_SOURCE_SHA256,
        contract_v1.REARM2_PACKAGE_SMOKE_SHA256,
        REARM2_STARTUP_SHA256,
    }


def _assert_no_rearm2_package_hashes(hashes: Iterable[str]) -> None:
    overlap = set(hashes) & _package_hash_denylist()
    if overlap:
        raise ValueError("rearm2 package identity overlap is not zero")


def _expected_role_manifests() -> dict[str, dict[str, Any]]:
    tail_run_contract = contract_v1.build_tail_run_contract()
    return {
        role: runner.build_shard_manifest(
            run_contract=tail_run_contract,
            source_role=role,
            work_hand_indices=contract_v1.TAIL_HAND_INDICES,
        )
        for role in contract_v1.SOURCE_ROLES
    }


def _safe_output_destination(output_parent: str | Path, run_name: str) -> tuple[Path, Path]:
    if _RUN_NAME.fullmatch(run_name) is None or _forbidden_namespace_or_path(run_name):
        raise ValueError("performance-development v2 run name is not fresh")
    parent_raw = Path(output_parent)
    _reject_link_components(parent_raw, include_leaf=False)
    if parent_raw.exists() and _is_reparse_or_link(parent_raw):
        raise ValueError(f"symlink or reparse-point path is forbidden: {parent_raw}")
    parent_raw.mkdir(parents=True, exist_ok=True)
    _reject_link_components(parent_raw)
    parent = parent_raw.resolve(strict=True)
    if _is_reparse_or_link(parent) or _forbidden_namespace_or_path(str(parent)):
        raise ValueError("package output path overlaps rearm2/lock namespace")
    destination = parent / run_name
    if destination.exists() or _is_reparse_or_link(destination):
        raise FileExistsError("local package destination is immutable")
    return parent, destination


def _publish_no_replace(stage: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("local package destination is immutable")
    try:
        os.rename(stage, destination)
    except FileExistsError as exc:
        raise FileExistsError("local package destination is immutable") from exc


def package_local(
    *,
    output_parent: str | Path,
    run_name: str,
    dry_run_receipt_path: str | Path,
    candidate_library: str | Path = DEFAULT_CANDIDATE_PATH,
    reference_library: str | Path = DEFAULT_REFERENCE_PATH,
    feature_encoder: str | Path = DEFAULT_FEATURE_PATH,
    full100_plan_path: str | Path = DEFAULT_PLAN_PATH,
    accepted_summary_path: str | Path = DEFAULT_ACCEPTED_SUMMARY_PATH,
    accepted_validation_path: str | Path = DEFAULT_ACCEPTED_VALIDATION_PATH,
    tail_selection_manifest_path: str | Path = DEFAULT_TAIL_SELECTION_MANIFEST_PATH,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    rearm2_root_dir: str | Path = DEFAULT_REARM2_ROOT_DIR,
    repository_root: str | Path = _REPO_ROOT,
    now_unix_seconds: int | None = None,
    tooling_sources: Sequence[tuple[str, str | Path]] | None = None,
    fixture_only: bool = False,
) -> dict[str, Any]:
    """Create and validate one immutable local, non-executable package."""

    parent, destination = _safe_output_destination(output_parent, run_name)
    audited = _audit_inputs(
        candidate_library=candidate_library,
        reference_library=reference_library,
        feature_encoder=feature_encoder,
        full100_plan_path=full100_plan_path,
        accepted_summary_path=accepted_summary_path,
        accepted_validation_path=accepted_validation_path,
        tail_selection_manifest_path=tail_selection_manifest_path,
        root_dir=root_dir,
        rearm2_root_dir=rearm2_root_dir,
        dry_run_receipt_path=dry_run_receipt_path,
    )
    if audited.receipt is None:
        raise ValueError("live dry-run receipt is required")
    runtime_namespace = audited.receipt["runtime_preflight"]["observation"]["namespace"]
    if runtime_namespace["run_name"] != run_name:
        raise ValueError("package run name differs from live dry-run namespace")
    if _forbidden_namespace_or_path(str(runtime_namespace)):
        raise ValueError("runtime namespace overlaps rearm2/lock namespace")
    freshness_now, receipt_age = _check_receipt_freshness(
        audited.receipt, now_unix_seconds=now_unix_seconds
    )
    repository = _safe_directory(repository_root, "repository root")
    tooling = _tooling_specs(
        repository_root=repository,
        tooling_sources=tooling_sources,
        fixture_only=fixture_only,
    )
    role_manifests = _expected_role_manifests()

    stage = parent / f".{run_name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    if stage.exists():
        raise FileExistsError("fresh staging directory collision")
    stage.mkdir()
    records: dict[str, dict[str, Any]] = {}
    try:
        primary_by_package = {
            record["package_path"]: record
            for record in audited.inventory["primary"].values()
        }
        roots_by_package = {
            record["package_path"]: record for record in audited.root_records
        }
        all_source_records = {**primary_by_package, **roots_by_package}
        if len(all_source_records) != len(primary_by_package) + len(roots_by_package):
            raise ValueError("duplicate package relative path in source inputs")
        for relative, record in sorted(all_source_records.items()):
            source = Path(record["source_path"])
            copied = _copy_verified(
                source,
                stage / relative,
                expected_sha256=record["sha256"],
                expected_bytes=record["bytes"],
            )
            records[relative] = {"group": "source", **copied}

        _write_json_once(stage / TAIL_SEED_PACKAGE_PATH, audited.tail_seed_schedule)
        records[TAIL_SEED_PACKAGE_PATH] = _file_record(
            stage / TAIL_SEED_PACKAGE_PATH, group="runtime"
        )
        _write_bytes_once(stage / STARTUP_PACKAGE_PATH, _STARTUP_BYTES)
        records[STARTUP_PACKAGE_PATH] = _file_record(
            stage / STARTUP_PACKAGE_PATH, group="runtime"
        )
        if records[STARTUP_PACKAGE_PATH]["sha256"] == REARM2_STARTUP_SHA256:
            raise ValueError("startup bytes reuse the rearm2 package startup")
        role_records: dict[str, dict[str, Any]] = {}
        for role, value in role_manifests.items():
            relative = ROLE_PACKAGE_PATHS[role]
            _write_json_once(stage / relative, value)
            records[relative] = _file_record(stage / relative, group="runtime")
            role_records[role] = {
                "path": relative,
                "sha256": records[relative]["sha256"],
                "bytes": records[relative]["bytes"],
                "work_hand_indices": list(contract_v1.TAIL_HAND_INDICES),
            }

        tooling_records: dict[str, dict[str, Any]] = {}
        occupied = set(records)
        for relative, source in tooling:
            if relative in occupied:
                raise ValueError(f"duplicate package relative path: {relative}")
            occupied.add(relative)
            expected_sha = sha256_file(source)
            expected_bytes = source.stat().st_size
            copied = _copy_verified(
                source,
                stage / relative,
                expected_sha256=expected_sha,
                expected_bytes=expected_bytes,
            )
            records[relative] = {"group": "tooling", **copied}
            tooling_records[relative] = {
                "source_path": str(source),
                **copied,
            }

        source_manifest = dict(audited.inventory)
        source_manifest.update(
            {
                "status": "source_and_packaged_bytes_rehashed_equal",
                "package_entry_sha256": _aggregate_entries(
                    {k: v for k, v in records.items() if v["group"] == "source"}
                ),
                "postpackage_source_rehash_required": True,
            }
        )
        runtime_manifest = {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "status": PACKAGE_STATUS,
            "run_name": run_name,
            "run_contract_digest": contract_v1.TAIL_RUN_CONTRACT_DIGEST,
            "full100_plan_run_contract_digest": (
                contract_v1.FULL_RUN_CONTRACT_DIGEST
            ),
            "dry_run_receipt_sha256": audited.inventory["primary"]["receipt"]["sha256"],
            "freshness_checked_at_unix_seconds": freshness_now,
            "receipt_age_seconds_at_packaging": receipt_age,
            "maximum_receipt_age_seconds": MAX_RECEIPT_AGE_SECONDS,
            "image": audited.receipt["image_preflight"]["observation"],
            "runtime_namespace": runtime_namespace,
            "allocation": {
                "machine_type": contract_v1.MACHINE_TYPE,
                "source_roles": list(contract_v1.SOURCE_ROLES),
                "instances": 2,
                "workers_per_source_process": 1,
                "rayon_threads_per_worker": 16,
            },
            "tail_seed_schedule": {
                "path": TAIL_SEED_PACKAGE_PATH,
                "sha256": records[TAIL_SEED_PACKAGE_PATH]["sha256"],
            },
            "tail_selection_manifest": {
                "path": TAIL_SELECTION_MANIFEST_PACKAGE_PATH,
                "sha256": records[TAIL_SELECTION_MANIFEST_PACKAGE_PATH]["sha256"],
                "bytes": records[TAIL_SELECTION_MANIFEST_PACKAGE_PATH]["bytes"],
            },
            "startup": {
                "path": STARTUP_PACKAGE_PATH,
                "sha256": records[STARTUP_PACKAGE_PATH]["sha256"],
                "behavior": "fail_closed_local_package_only_exit_97",
            },
            "roles": role_records,
            "rearm2_overlap": {
                "roots": audited.inventory["rearm2_overlap"]["roots"],
                "seeds": audited.inventory["rearm2_overlap"]["seeds"],
                "run_namespace_overlap_count": 0,
            },
            "cloud_executable": False,
            "launch_authorized": False,
            "cloud_mutated": False,
            "gcloud_invoked": False,
        }
        tooling_manifest = {
            "schema": TOOLING_MANIFEST_SCHEMA,
            "status": "tooling_bytes_rehashed_and_packaged",
            "fixture_only": fixture_only,
            "entries": tooling_records,
            "entry_count": len(tooling_records),
            "entries_sha256": _aggregate_entries(tooling_records),
            "cloud_executable": False,
            "launch_authorized": False,
        }
        for relative, value in (
            (SOURCE_MANIFEST_PATH, source_manifest),
            (RUNTIME_MANIFEST_PATH, runtime_manifest),
            (TOOLING_MANIFEST_PATH, tooling_manifest),
        ):
            _write_json_once(stage / relative, value)
            records[relative] = _file_record(stage / relative, group="manifest")

        # Rehash every mutable source after all package copies are complete.
        for record in audited.inventory["primary"].values():
            if (
                sha256_file(record["source_path"]) != record["sha256"]
                or Path(record["source_path"]).stat().st_size != record["bytes"]
            ):
                raise ValueError("primary source changed after package copy")
        for record in audited.root_records:
            if sha256_file(record["source_path"]) != record["sha256"]:
                raise ValueError("root source changed after package copy")
        for record in audited.rearm2_root_records:
            if sha256_file(record["source_path"]) != record["sha256"]:
                raise ValueError("rearm2 denylist root changed after package copy")
        for record in tooling_records.values():
            if sha256_file(record["source_path"]) != record["sha256"]:
                raise ValueError("tooling source changed after package copy")

        _assert_no_rearm2_package_hashes(record["sha256"] for record in records.values())
        package_manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": PACKAGE_STATUS,
            "run_name": run_name,
            "run_contract_digest": contract_v1.TAIL_RUN_CONTRACT_DIGEST,
            "full100_plan_run_contract_digest": (
                contract_v1.FULL_RUN_CONTRACT_DIGEST
            ),
            "selection_manifest_sha256": (
                contract_v1.TAIL_SELECTION_MANIFEST_SHA256
            ),
            "source_manifest": {
                "path": SOURCE_MANIFEST_PATH,
                "sha256": records[SOURCE_MANIFEST_PATH]["sha256"],
            },
            "runtime_manifest": {
                "path": RUNTIME_MANIFEST_PATH,
                "sha256": records[RUNTIME_MANIFEST_PATH]["sha256"],
            },
            "tooling_manifest": {
                "path": TOOLING_MANIFEST_PATH,
                "sha256": records[TOOLING_MANIFEST_PATH]["sha256"],
            },
            "entries": dict(sorted(records.items())),
            "entry_count": len(records),
            "entries_sha256": _aggregate_entries(records),
            "rearm2_package_hash_overlap_count": 0,
            "source_bytes_equal_packaged_bytes": True,
            "postpackage_validation_passed_before_publish": True,
            "cloud_executable": False,
            "launch_authorized": False,
            "cloud_mutated": False,
            "gcloud_invoked": False,
            "current_profile_changed": False,
        }
        _write_json_once(stage / MANIFEST_NAME, package_manifest)
        manifest_sha = sha256_file(stage / MANIFEST_NAME)
        _assert_no_rearm2_package_hashes((manifest_sha,))
        ready = {
            "schema": PACKAGE_READY_SCHEMA,
            "status": PACKAGE_STATUS,
            "run_name": run_name,
            "run_contract_digest": contract_v1.TAIL_RUN_CONTRACT_DIGEST,
            "selection_manifest_sha256": (
                contract_v1.TAIL_SELECTION_MANIFEST_SHA256
            ),
            "package_manifest_sha256": manifest_sha,
            "source_manifest_sha256": records[SOURCE_MANIFEST_PATH]["sha256"],
            "runtime_manifest_sha256": records[RUNTIME_MANIFEST_PATH]["sha256"],
            "tooling_manifest_sha256": records[TOOLING_MANIFEST_PATH]["sha256"],
            "all_entries_postpackage_rehashed": True,
            "source_bytes_equal_packaged_bytes": True,
            "aggregate_root_digest_revalidated": True,
            "aggregate_topology_digest_revalidated": True,
            "seed_digest_revalidated": True,
            "rearm2_overlap_count": 0,
            "cloud_executable": False,
            "launch_authorized": False,
            "cloud_mutated": False,
            "gcloud_invoked": False,
            "current_profile_changed": False,
        }
        _write_json_once(stage / READY_NAME, ready)
        _assert_no_rearm2_package_hashes((sha256_file(stage / READY_NAME),))

        validate_local_package(stage, _expected_run_name=run_name)
        _publish_no_replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_local_package(destination)


def _all_package_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in directory.rglob("*"):
        _reject_link_components(path)
        if _is_reparse_or_link(path):
            raise ValueError("package contains a symlink or reparse point")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if relative in files:
                raise ValueError("package contains duplicate relative paths")
            files[relative] = path
    return files


def validate_local_package(
    run_dir: str | Path, *, _expected_run_name: str | None = None
) -> dict[str, Any]:
    """Rehash and semantically validate a local package without side effects."""

    directory = _safe_directory(run_dir, "local package")
    expected_run_name = directory.name if _expected_run_name is None else _expected_run_name
    files = _all_package_files(directory)
    manifest_path, manifest = _read_canonical(directory / MANIFEST_NAME, "package manifest")
    ready_path, ready = _read_canonical(directory / READY_NAME, "package ready receipt")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or ready.get("schema") != PACKAGE_READY_SCHEMA
        or ready.get("status") != PACKAGE_STATUS
        or manifest.get("run_name") != expected_run_name
        or ready.get("run_name") != expected_run_name
        or manifest.get("run_contract_digest")
        != contract_v1.TAIL_RUN_CONTRACT_DIGEST
        or ready.get("run_contract_digest")
        != contract_v1.TAIL_RUN_CONTRACT_DIGEST
        or manifest.get("full100_plan_run_contract_digest")
        != contract_v1.FULL_RUN_CONTRACT_DIGEST
        or manifest.get("selection_manifest_sha256")
        != contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        or ready.get("selection_manifest_sha256")
        != contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        or _RUN_NAME.fullmatch(expected_run_name) is None
        or _forbidden_namespace_or_path(str(directory))
        or manifest.get("postpackage_validation_passed_before_publish") is not True
        or ready.get("all_entries_postpackage_rehashed") is not True
        or ready.get("source_bytes_equal_packaged_bytes") is not True
        or ready.get("aggregate_root_digest_revalidated") is not True
        or ready.get("aggregate_topology_digest_revalidated") is not True
        or ready.get("seed_digest_revalidated") is not True
    ):
        raise ValueError("local package identity or status changed")
    if any(
        record.get(field) is not False
        for record in (manifest, ready)
        for field in ("cloud_executable", "launch_authorized", "cloud_mutated", "gcloud_invoked", "current_profile_changed")
    ):
        raise ValueError("local package crossed the non-executable boundary")
    entries = manifest.get("entries")
    if not isinstance(entries, Mapping) or manifest.get("entry_count") != len(entries):
        raise ValueError("package entry manifest changed")
    expected_files = set(entries) | {MANIFEST_NAME, READY_NAME}
    if set(files) != expected_files:
        raise ValueError("package has missing, unknown, or duplicate entries")
    normalized_records: dict[str, dict[str, Any]] = {}
    for relative, raw_record in entries.items():
        if _normalize_package_relative(str(relative)) != relative or not isinstance(raw_record, Mapping):
            raise ValueError("package entry record changed")
        record = dict(raw_record)
        if set(record) != {"group", "sha256", "bytes"}:
            raise ValueError("package entry record fields changed")
        path = files[relative]
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"package entry bytes changed: {relative}")
        normalized_records[relative] = record
    if (
        manifest.get("entries_sha256") != _aggregate_entries(normalized_records)
        or ready.get("package_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("package aggregate or manifest digest changed")

    source_path, source_manifest = _read_canonical(
        directory / SOURCE_MANIFEST_PATH, "source manifest"
    )
    runtime_path, runtime_manifest = _read_canonical(
        directory / RUNTIME_MANIFEST_PATH, "runtime manifest"
    )
    tooling_path, tooling_manifest = _read_canonical(
        directory / TOOLING_MANIFEST_PATH, "tooling manifest"
    )
    for key, path, value, schema in (
        ("source_manifest", source_path, source_manifest, SOURCE_MANIFEST_SCHEMA),
        ("runtime_manifest", runtime_path, runtime_manifest, RUNTIME_MANIFEST_SCHEMA),
        ("tooling_manifest", tooling_path, tooling_manifest, TOOLING_MANIFEST_SCHEMA),
    ):
        if value.get("schema") != schema or manifest[key]["sha256"] != sha256_file(path):
            raise ValueError(f"{key} hash or schema changed")
        if ready[f"{key}_sha256"] != sha256_file(path):
            raise ValueError(f"ready receipt {key} hash changed")

    source_primary = source_manifest.get("primary")
    source_roots = source_manifest.get("roots")
    if not isinstance(source_primary, Mapping) or not isinstance(source_roots, list):
        raise ValueError("source manifest entry inventory changed")
    source_inventory_records = [*source_primary.values(), *source_roots]
    if len(source_inventory_records) != 108:
        raise ValueError("source manifest entry count changed")
    source_package_records: dict[str, dict[str, Any]] = {}
    for raw in source_inventory_records:
        if not isinstance(raw, Mapping):
            raise ValueError("source manifest entry changed")
        package_relative = raw.get("package_path")
        if (
            not isinstance(package_relative, str)
            or package_relative in source_package_records
            or package_relative not in normalized_records
            or raw.get("sha256") != normalized_records[package_relative]["sha256"]
            or raw.get("bytes") != normalized_records[package_relative]["bytes"]
            or normalized_records[package_relative]["group"] != "source"
        ):
            raise ValueError("source and packaged byte records differ")
        source_package_records[package_relative] = normalized_records[package_relative]
    if (
        set(source_package_records)
        != {
            relative
            for relative, record in normalized_records.items()
            if record["group"] == "source"
        }
        or source_manifest.get("package_entry_sha256")
        != _aggregate_entries(source_package_records)
    ):
        raise ValueError("source package aggregate changed")
    rearm2_roots = source_manifest.get("rearm2_denylist_roots")
    if not isinstance(rearm2_roots, Mapping) or not isinstance(
        rearm2_roots.get("records"), list
    ):
        raise ValueError("rearm2 root denylist provenance changed")
    rearm2_hashes = [record.get("sha256") for record in rearm2_roots["records"]]
    development_hashes = [record.get("sha256") for record in source_roots]
    root_overlap = source_manifest.get("rearm2_overlap", {}).get("roots", {})
    if (
        rearm2_roots.get("root_count") != 100
        or len(rearm2_hashes) != 100
        or len(set(rearm2_hashes)) != 100
        or canonical_sha256(rearm2_hashes)
        != contract_v1.REARM2_AGGREGATE_ROOT_SHA256
        or set(rearm2_hashes) & set(development_hashes)
        or root_overlap.get("individual_rearm2_root_hashes_compared") is not True
        or root_overlap.get("individual_file_count_each") != 100
        or root_overlap.get("individual_file_hash_overlap_count") != 0
    ):
        raise ValueError("rearm2 individual root overlap proof changed")

    plan_path, plan_raw = _read_canonical(directory / PLAN_PACKAGE_PATH, "packaged plan")
    summary_path, _ = _read_canonical(directory / SUMMARY_PACKAGE_PATH, "packaged summary")
    validation_path, _ = _read_canonical(
        directory / VALIDATION_PACKAGE_PATH, "packaged validation"
    )
    selection_path, _ = _read_canonical(
        directory / TAIL_SELECTION_MANIFEST_PACKAGE_PATH,
        "packaged candidate02 tail-v2 selection manifest",
    )
    contract_v1.validate_tail_selection_manifest(selection_path)
    receipt_path, receipt_raw = _read_canonical(
        directory / DRY_RECEIPT_PACKAGE_PATH, "packaged dry-run receipt"
    )
    plan = full100.validate_full100_plan(plan_raw)
    if (
        sha256_file(plan_path) != contract_v1.FULL100_PLAN_SHA256
        or sha256_file(summary_path) != contract_v1.ACCEPTED_SUMMARY_SHA256
        or sha256_file(validation_path) != contract_v1.ACCEPTED_VALIDATION_SHA256
        or sha256_file(selection_path)
        != contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        or selection_path.stat().st_size != contract_v1.TAIL_SELECTION_MANIFEST_BYTES
    ):
        raise ValueError("packaged accepted evidence digest changed")
    receipt = _validate_receipt_against_inputs(
        receipt_raw,
        plan_path=plan_path,
        summary_path=summary_path,
        validation_path=validation_path,
        selection_path=selection_path,
    )
    namespace = receipt["runtime_preflight"]["observation"]["namespace"]
    if namespace["run_name"] != expected_run_name or runtime_manifest.get("runtime_namespace") != namespace:
        raise ValueError("packaged runtime namespace changed")
    if (
        runtime_manifest.get("status") != PACKAGE_STATUS
        or runtime_manifest.get("run_contract_digest")
        != contract_v1.TAIL_RUN_CONTRACT_DIGEST
        or runtime_manifest.get("full100_plan_run_contract_digest")
        != contract_v1.FULL_RUN_CONTRACT_DIGEST
        or runtime_manifest.get("tail_selection_manifest")
        != {
            "path": TAIL_SELECTION_MANIFEST_PACKAGE_PATH,
            "sha256": contract_v1.TAIL_SELECTION_MANIFEST_SHA256,
            "bytes": contract_v1.TAIL_SELECTION_MANIFEST_BYTES,
        }
        or runtime_manifest.get("dry_run_receipt_sha256") != sha256_file(receipt_path)
        or runtime_manifest.get("tail_seed_schedule", {}).get("sha256")
        != normalized_records[TAIL_SEED_PACKAGE_PATH]["sha256"]
        or runtime_manifest.get("startup", {}).get("sha256")
        != normalized_records[STARTUP_PACKAGE_PATH]["sha256"]
        or runtime_manifest.get("startup", {}).get("behavior")
        != "fail_closed_local_package_only_exit_97"
        or any(
            runtime_manifest.get(field) is not False
            for field in ("cloud_executable", "launch_authorized", "cloud_mutated", "gcloud_invoked")
        )
    ):
        raise ValueError("runtime manifest boundary changed")
    checked_at = runtime_manifest.get("freshness_checked_at_unix_seconds")
    age = runtime_manifest.get("receipt_age_seconds_at_packaging")
    if (
        isinstance(checked_at, bool)
        or not isinstance(checked_at, int)
        or age != checked_at - _receipt_unix_seconds(receipt)
        or age > MAX_RECEIPT_AGE_SECONDS
        or age < -MAX_FUTURE_SKEW_SECONDS
    ):
        raise ValueError("packaged receipt freshness proof changed")

    packaged_roots = tuple(selector.load_frozen_roots(directory / ROOT_PACKAGE_DIR))
    if (
        canonical_sha256([canonical_sha256(root) for root in packaged_roots])
        != contract_v1.DEVELOPMENT_ROOT_SET_SHA256
        or canonical_sha256(selector.topology_rows(packaged_roots))
        != contract_v1.DEVELOPMENT_ROOT_TOPOLOGY_SHA256
        or source_manifest.get("root_set_sha256")
        != contract_v1.DEVELOPMENT_ROOT_SET_SHA256
        or source_manifest.get("root_topology_sha256")
        != contract_v1.DEVELOPMENT_ROOT_TOPOLOGY_SHA256
    ):
        raise ValueError("packaged root aggregate or topology changed")
    _validate_tail_seed_schedule(
        _read_canonical(directory / TAIL_SEED_PACKAGE_PATH, "tail seed schedule")[1]
    )
    if (directory / STARTUP_PACKAGE_PATH).read_bytes() != _STARTUP_BYTES:
        raise ValueError("fail-closed local startup changed")
    role_manifests = _expected_role_manifests()
    for role, expected in role_manifests.items():
        observed = _read_canonical(directory / ROLE_PACKAGE_PATHS[role], f"{role} role manifest")[1]
        if runner.validate_shard_manifest(observed) != expected:
            raise ValueError(f"{role} tail role manifest changed")

    tooling_entries = tooling_manifest.get("entries")
    if not isinstance(tooling_entries, Mapping) or tooling_manifest.get("entry_count") != len(tooling_entries):
        raise ValueError("tooling manifest entries changed")
    packaged_tooling = {
        relative: normalized_records[relative]
        for relative in normalized_records
        if normalized_records[relative]["group"] == "tooling"
    }
    if (
        set(tooling_entries) != set(packaged_tooling)
        or tooling_manifest.get("entries_sha256") != _aggregate_entries(tooling_entries)
    ):
        raise ValueError("tooling manifest aggregate changed")
    for relative, record in tooling_entries.items():
        if record.get("sha256") != packaged_tooling[relative]["sha256"] or record.get("bytes") != packaged_tooling[relative]["bytes"]:
            raise ValueError("tooling manifest/package bytes differ")

    all_hashes = [record["sha256"] for record in normalized_records.values()]
    all_hashes.extend((sha256_file(manifest_path), sha256_file(ready_path)))
    _assert_no_rearm2_package_hashes(all_hashes)
    if (
        manifest.get("rearm2_package_hash_overlap_count") != 0
        or runtime_manifest.get("rearm2_overlap", {}).get("run_namespace_overlap_count") != 0
        or runtime_manifest.get("rearm2_overlap", {}).get("seeds", {}).get("overlap_count") != 0
        or runtime_manifest.get("rearm2_overlap", {}).get("roots", {}).get("overlap_count") != 0
        or ready.get("rearm2_overlap_count") != 0
    ):
        raise ValueError("rearm2 overlap proof changed")
    return {
        "schema": PACKAGE_READY_SCHEMA,
        "status": PACKAGE_STATUS,
        "run_name": expected_run_name,
        "package_path": str(directory),
        "package_manifest_sha256": sha256_file(manifest_path),
        "package_ready_sha256": sha256_file(ready_path),
        "entry_count": len(normalized_records),
        "root_count": len(packaged_roots),
        "run_contract_digest": contract_v1.TAIL_RUN_CONTRACT_DIGEST,
        "selection_manifest_sha256": contract_v1.TAIL_SELECTION_MANIFEST_SHA256,
        "tail_hand_indices": list(contract_v1.TAIL_HAND_INDICES),
        "source_roles": list(contract_v1.SOURCE_ROLES),
        "source_bytes_equal_packaged_bytes": True,
        "rearm2_overlap_count": 0,
        "cloud_executable": False,
        "launch_authorized": False,
        "cloud_mutated": False,
        "gcloud_invoked": False,
        "current_profile_changed": False,
    }


__all__ = [
    "DEFAULT_ACCEPTED_SUMMARY_PATH",
    "DEFAULT_ACCEPTED_VALIDATION_PATH",
    "DEFAULT_CANDIDATE_PATH",
    "DEFAULT_DRY_RECEIPT_PATH",
    "DEFAULT_FEATURE_PATH",
    "DEFAULT_PLAN_PATH",
    "DEFAULT_REFERENCE_PATH",
    "DEFAULT_REARM2_ROOT_DIR",
    "DEFAULT_ROOT_DIR",
    "DEFAULT_TAIL_SELECTION_MANIFEST_PATH",
    "PACKAGE_READY_SCHEMA",
    "PACKAGE_SCHEMA",
    "PACKAGE_STATUS",
    "TAIL_SELECTION_MANIFEST_PACKAGE_PATH",
    "audit_source_inventory",
    "canonical_bytes",
    "canonical_sha256",
    "package_local",
    "sha256_file",
    "validate_local_package",
]
