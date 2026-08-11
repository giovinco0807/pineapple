"""Local-only lifecycle for the bounded rearm2 diagnostic canaries.

This module deliberately has no cloud launcher.  It packages the frozen
diagnostic roots, verifies a startup identity before any root member is read,
records deterministic write-once uploads and heartbeats, publishes ``DONE``
last, and receives an exact stage result into an immutable directory.

The package is diagnostic infrastructure evidence only.  Nothing produced by
this module is performance-lock, quality, training, promotion, or runtime
activation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan


PACKAGE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_package_v1"
PACKAGE_STATUS = "immutable_diagnostic_local_package_ready_no_cloud_path"
PACKAGE_READY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_local_package_ready_v1"
)
JOB_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_job_v1"
STAGE_OPEN_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_stage_open_v2"
UPLOAD_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_upload_v1"
HEARTBEAT_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_heartbeat_v1"
DONE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_done_v1"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_receive_v1"
AGGREGATE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_aggregate_v1"

MANIFEST_NAME = "manifest.json"
READY_NAME = "PACKAGE_READY.json"
SOURCE_NAME = "diagnostic_source_v1.zip"
STARTUP_NAME = "diagnostic_startup_verify_v1.py"
LOCAL_STAGE_OPEN_NAME = "LOCAL_STAGE_OPEN.json"
DONE_NAME = "DONE.json"
RECEIPT_NAME = "receive_receipt.json"
AGGREGATE_NAME = "aggregate.json"

_SHA256_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_KEY_PARTS = (
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
)

_STARTUP_SOURCE = r'''from __future__ import annotations
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PACKAGE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_package_v1"
PACKAGE_STATUS = "immutable_diagnostic_local_package_ready_no_cloud_path"
JOB_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_job_v1"
STAGE_OPEN_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_stage_open_v2"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_local_receive_v1"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("startup control must be an object")
    return value, hashlib.sha256(raw).hexdigest()

def canonical_sha(value):
    raw = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def fail(message):
    raise ValueError("diagnostic startup pre-content rejection: " + message)

source, manifest_path, stage_open_path, job_path = map(Path, sys.argv[1:5])
manifest, manifest_sha = read(manifest_path)
stage_open, stage_open_sha = read(stage_open_path)
job, job_sha = read(job_path)
if manifest.get("schema") != PACKAGE_SCHEMA or manifest.get("status") != PACKAGE_STATUS:
    fail("package identity")
if any(manifest.get(k) is not False for k in (
    "cloud_capable", "gcloud_callable", "claim_write_authorized",
    "authorization_write_authorized", "performance_lock_evidence",
    "quality_evidence", "training_eligible", "promotion_evidence",
    "current_profile_changed", "runtime_policy_activated",
)):
    fail("forbidden package capability")
src = manifest.get("source")
if not isinstance(src, dict) or src.get("name") != source.name or src.get("sha256") != sha(source):
    fail("source identity")
entries = src.get("entries")
if not isinstance(entries, list):
    fail("source entries")
expected_names = [row.get("path") for row in entries if isinstance(row, dict)]
if len(expected_names) != len(entries) or len(set(expected_names)) != len(entries):
    fail("source entry identities")
with zipfile.ZipFile(source) as archive:
    infos = archive.infolist()
    if [info.filename for info in infos] != expected_names:
        fail("source central directory")
    if [info.file_size for info in infos] != [row.get("bytes") for row in entries]:
        fail("source entry sizes")
if stage_open.get("schema") != STAGE_OPEN_SCHEMA:
    fail("stage-open schema")
if stage_open.get("package_manifest_sha256") != manifest_sha:
    fail("stage-open package")
if any(stage_open.get(k) is not False for k in (
    "cloud_launch_authorized", "gcloud_invocation_authorized",
    "spot_claim_written", "authorization_written",
    "performance_lock_evidence", "quality_evidence", "training_eligible",
    "promotion_evidence", "current_profile_changed",
)):
    fail("stage-open capability")
if job.get("schema") != JOB_SCHEMA:
    fail("job schema")
stage_id = stage_open.get("stage_id")
job_id = job.get("job_id")
stages = manifest.get("stages")
if not isinstance(stages, list):
    fail("manifest stages")
stage = next((row for row in stages if isinstance(row, dict) and row.get("stage_id") == stage_id), None)
if stage is None or stage_open.get("selected_job_ids") != stage.get("selected_job_ids"):
    fail("stage identity")
prerequisite_id = stage.get("prerequisite_stage_id")
prerequisite = stage_open.get("prerequisite_receive_receipt")
prerequisite_sha = stage_open.get("prerequisite_receive_receipt_sha256")
if prerequisite_id is None:
    if prerequisite is not None or prerequisite_sha is not None:
        fail("unexpected prerequisite receipt")
else:
    prerequisite_stage = next(
        (
            row
            for row in stages
            if isinstance(row, dict) and row.get("stage_id") == prerequisite_id
        ),
        None,
    )
    if not isinstance(prerequisite, dict) or prerequisite_stage is None:
        fail("missing prerequisite receipt")
    prerequisite_jobs = prerequisite.get("job_records")
    expected_prerequisite_ids = prerequisite_stage.get("selected_job_ids")
    if (
        prerequisite.get("schema") != RECEIVE_SCHEMA
        or prerequisite.get("status")
        != "diagnostic_local_stage_received_and_validated"
        or prerequisite.get("package_manifest_sha256") != manifest_sha
        or prerequisite.get("stage_id") != prerequisite_id
        or prerequisite.get("run_name") != prerequisite_stage.get("run_name")
        or prerequisite.get("selected_job_ids") != expected_prerequisite_ids
        or not isinstance(prerequisite_jobs, list)
        or [row.get("job_id") for row in prerequisite_jobs if isinstance(row, dict)]
        != expected_prerequisite_ids
        or prerequisite.get("job_record_aggregate_sha256")
        != canonical_sha(prerequisite_jobs)
        or prerequisite_sha != canonical_sha(prerequisite)
        or prerequisite.get("diagnostic_only") is not True
        or any(
            prerequisite.get(key) is not False
            for key in (
                "cloud_receive_performed",
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        fail("invalid prerequisite receipt")
    expected_roles = prerequisite_stage.get("source_roles")
    if [row.get("source_role") for row in prerequisite_jobs] != expected_roles:
        fail("prerequisite receipt role order")
jobs = stage.get("jobs")
if not isinstance(jobs, list):
    fail("stage jobs")
record = next((row for row in jobs if isinstance(row, dict) and row.get("job_id") == job_id), None)
if record is None or record.get("sha256") != job_sha:
    fail("job identity")
if job_id not in stage_open.get("selected_job_ids", []):
    fail("job outside exact stage set")
if job.get("stage_id") != stage_id or job.get("run_name") != stage.get("run_name"):
    fail("job stage binding")
if any(job.get(k) is not False for k in (
    "cloud_capable", "performance_lock_evidence", "quality_evidence",
    "training_eligible", "promotion_evidence", "current_profile_changed",
)):
    fail("job capability")
print("diag_local_v2|" + str(stage_id) + "|" + str(job_id))
'''


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _read_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {source}")
    raw = source.read_bytes()
    if canonical_bytes(json.loads(raw)) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _write_once(path: Path, value: Mapping[str, Any] | bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable diagnostic artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(raw)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys changed")


def _reject_hidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"forbidden hidden field in diagnostic payload: {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _stage_by_id(contract: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in contract["stages"]:
        if stage["stage_id"] == stage_id:
            return dict(stage)
    raise ValueError(f"unknown diagnostic stage: {stage_id}")


def _job_path(package_dir: Path, stage_id: str, job_id: str) -> Path:
    return package_dir / "jobs" / stage_id / f"{job_id}.json"


def _source_entries(
    contract: Mapping[str, Any],
    root_directory: Path,
) -> tuple[list[tuple[str, bytes]], list[dict[str, Any]]]:
    contract_raw = canonical_bytes(contract)
    content: list[tuple[str, bytes]] = [
        ("control/diagnostic_contract.json", contract_raw)
    ]
    seen: set[int] = set()
    for stage in contract["stages"]:
        for record in stage["root_records"]:
            index = int(record["hand_index"])
            if index in seen:
                continue
            seen.add(index)
            root_path = root_directory / f"hand_{index:03d}.json"
            if root_path.is_symlink() or not root_path.is_file():
                raise FileNotFoundError(f"diagnostic root is missing or unsafe: {root_path}")
            raw = root_path.read_bytes()
            if (
                hashlib.sha256(raw).hexdigest() != record["sha256"]
                or len(raw) != record["size_bytes"]
            ):
                raise ValueError(f"diagnostic development root changed: {index}")
            content.append((f"roots/hand_{index:03d}.json", raw))
    content.sort(key=lambda row: row[0])
    records = [
        {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for name, raw in content
    ]
    return content, records


def _write_deterministic_zip(path: Path, entries: Sequence[tuple[str, bytes]]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable diagnostic source exists: {path}")
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)


def _job_manifest(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    stage: Mapping[str, Any],
    job_id: str,
    source_role: str,
) -> dict[str, Any]:
    roots = [
        {
            "hand_index": int(record["hand_index"]),
            "archive_path": f"roots/hand_{int(record['hand_index']):03d}.json",
            "sha256": record["sha256"],
            "bytes": record["size_bytes"],
        }
        for record in stage["root_records"]
    ]
    value = {
        "schema": JOB_SCHEMA,
        "status": "diagnostic_local_job_frozen_no_cloud_path",
        "diagnostic_contract_sha256": contract_sha256,
        "development_run_contract_digest": contract["development_root_source"][
            "run_contract_digest"
        ],
        "stage_id": stage["stage_id"],
        "run_name": stage["run_name"],
        "job_id": job_id,
        "source_role": source_role,
        "work_hand_indices": list(stage["hand_indices"]),
        "root_records": roots,
        "cloud_capable": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    validate_job_manifest(value, contract=contract)
    return value


def validate_job_manifest(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(value)
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "diagnostic_contract_sha256",
            "development_run_contract_digest",
            "stage_id",
            "run_name",
            "job_id",
            "source_role",
            "work_hand_indices",
            "root_records",
            "cloud_capable",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
            "current_profile_changed",
        },
        "diagnostic job manifest",
    )
    stage = _stage_by_id(contract, str(payload.get("stage_id")))
    try:
        offset = stage["selected_job_ids"].index(payload.get("job_id"))
    except ValueError as exc:
        raise ValueError("diagnostic job is outside the exact stage set") from exc
    expected_roots = [
        {
            "hand_index": int(record["hand_index"]),
            "archive_path": f"roots/hand_{int(record['hand_index']):03d}.json",
            "sha256": record["sha256"],
            "bytes": record["size_bytes"],
        }
        for record in stage["root_records"]
    ]
    if (
        payload.get("schema") != JOB_SCHEMA
        or payload.get("status") != "diagnostic_local_job_frozen_no_cloud_path"
        or payload.get("diagnostic_contract_sha256") != canonical_sha256(contract)
        or payload.get("development_run_contract_digest")
        != contract["development_root_source"]["run_contract_digest"]
        or payload.get("run_name") != stage["run_name"]
        or payload.get("source_role") != stage["source_roles"][offset]
        or payload.get("work_hand_indices") != stage["hand_indices"]
        or payload.get("root_records") != expected_roots
        or any(
            payload.get(field) is not False
            for field in (
                "cloud_capable",
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("diagnostic job manifest changed")
    return payload


def build_local_package(
    *,
    output_dir: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
    root_directory: str | Path = plan.DEFAULT_DEVELOPMENT_ROOTS,
) -> dict[str, Any]:
    """Build an immutable package with no launch, claim, or auth API."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError("diagnostic local package destination is immutable")
    contract = plan.validate_frozen_contract(contract_path)
    roots = Path(root_directory).resolve()
    expected_roots = plan.DEFAULT_DEVELOPMENT_ROOTS.resolve()
    if roots != expected_roots:
        raise ValueError("diagnostic local package requires frozen development roots")
    stage = target.parent / f".{target.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError("stale diagnostic package staging directory exists")
    stage.mkdir(parents=True)
    try:
        content, source_records = _source_entries(contract, roots)
        source = stage / SOURCE_NAME
        _write_deterministic_zip(source, content)
        startup = stage / STARTUP_NAME
        _write_once(startup, _STARTUP_SOURCE.encode("utf-8"))
        contract_sha = canonical_sha256(contract)
        stage_records: list[dict[str, Any]] = []
        for raw_stage in contract["stages"]:
            jobs: list[dict[str, Any]] = []
            for job_id, source_role in zip(
                raw_stage["selected_job_ids"],
                raw_stage["source_roles"],
                strict=True,
            ):
                job = _job_manifest(
                    contract=contract,
                    contract_sha256=contract_sha,
                    stage=raw_stage,
                    job_id=job_id,
                    source_role=source_role,
                )
                relative = f"jobs/{raw_stage['stage_id']}/{job_id}.json"
                path = stage / relative
                _write_once(path, job)
                jobs.append(
                    {
                        "job_id": job_id,
                        "source_role": source_role,
                        "path": relative,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
            stage_records.append(
                {
                    "stage_id": raw_stage["stage_id"],
                    "run_name": raw_stage["run_name"],
                    "reserved_claim_name": raw_stage["claim_name"],
                    "reserved_result_name": raw_stage["result_name"],
                    "selected_job_ids": list(raw_stage["selected_job_ids"]),
                    "source_roles": list(raw_stage["source_roles"]),
                    "hand_indices": list(raw_stage["hand_indices"]),
                    "root_record_digest": raw_stage["root_record_digest"],
                    "prerequisite_stage_id": (
                        None
                        if raw_stage["stage_id"] == plan.STAGE1_ID
                        else plan.STAGE1_ID
                    ),
                    "jobs": jobs,
                }
            )
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": PACKAGE_STATUS,
            "diagnostic_contract_sha256": contract_sha,
            "development_root_classification": contract["development_root_source"][
                "classification"
            ],
            "production_rearm2_read_only_anchor": dict(
                contract["rearm2_production_anchor"]
            ),
            "source": {
                "name": SOURCE_NAME,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "entries": source_records,
            },
            "startup": {
                "name": STARTUP_NAME,
                "sha256": sha256_file(startup),
                "bytes": startup.stat().st_size,
                "pre_content_root_reads": 0,
            },
            "stages": stage_records,
            "cloud_capable": False,
            "gcloud_callable": False,
            "claim_write_authorized": False,
            "authorization_write_authorized": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
            "runtime_policy_activated": False,
        }
        _write_once(stage / MANIFEST_NAME, manifest)
        ready = {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "diagnostic_local_package_complete_no_cloud_path",
            "package_manifest_sha256": sha256_file(stage / MANIFEST_NAME),
            "source_sha256": sha256_file(source),
            "startup_sha256": sha256_file(startup),
            "stage_job_counts": [len(row["jobs"]) for row in stage_records],
            "cloud_capable": False,
            "gcloud_callable": False,
            "claim_written": False,
            "authorization_written": False,
            "current_profile_changed": False,
        }
        _write_once(stage / READY_NAME, ready)
        validate_local_package(stage, contract_path=contract_path)
        stage.rename(target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_local_package(target, contract_path=contract_path)


def _expected_package_files(manifest: Mapping[str, Any]) -> set[str]:
    files = {MANIFEST_NAME, READY_NAME, SOURCE_NAME, STARTUP_NAME}
    for stage in manifest["stages"]:
        files.update(record["path"] for record in stage["jobs"])
    return files


def validate_local_package(
    package_dir: str | Path,
    *,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    contract = plan.validate_frozen_contract(contract_path)
    manifest = _read_object(root / MANIFEST_NAME, "diagnostic package manifest")
    ready = _read_object(root / READY_NAME, "diagnostic package ready marker")
    _exact_keys(
        manifest,
        {
            "schema",
            "status",
            "diagnostic_contract_sha256",
            "development_root_classification",
            "production_rearm2_read_only_anchor",
            "source",
            "startup",
            "stages",
            "cloud_capable",
            "gcloud_callable",
            "claim_write_authorized",
            "authorization_write_authorized",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
            "current_profile_changed",
            "runtime_policy_activated",
        },
        "diagnostic package manifest",
    )
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("diagnostic_contract_sha256") != canonical_sha256(contract)
        or manifest.get("development_root_classification")
        != contract["development_root_source"]["classification"]
        or manifest.get("production_rearm2_read_only_anchor")
        != contract["rearm2_production_anchor"]
        or any(
            manifest.get(field) is not False
            for field in (
                "cloud_capable",
                "gcloud_callable",
                "claim_write_authorized",
                "authorization_write_authorized",
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("diagnostic local package identity changed")
    source = manifest.get("source")
    startup = manifest.get("startup")
    stages = manifest.get("stages")
    if not isinstance(source, Mapping) or not isinstance(startup, Mapping) or not isinstance(stages, list):
        raise ValueError("diagnostic local package structure changed")
    if (
        source.get("name") != SOURCE_NAME
        or source.get("sha256") != sha256_file(root / SOURCE_NAME)
        or source.get("bytes") != (root / SOURCE_NAME).stat().st_size
        or startup
        != {
            "name": STARTUP_NAME,
            "sha256": sha256_file(root / STARTUP_NAME),
            "bytes": (root / STARTUP_NAME).stat().st_size,
            "pre_content_root_reads": 0,
        }
    ):
        raise ValueError("diagnostic package source/startup changed")
    expected_stage_ids = [row["stage_id"] for row in contract["stages"]]
    if [row.get("stage_id") for row in stages if isinstance(row, Mapping)] != expected_stage_ids:
        raise ValueError("diagnostic package stages changed")
    for stage_record, contract_stage in zip(stages, contract["stages"], strict=True):
        if (
            stage_record.get("run_name") != contract_stage["run_name"]
            or stage_record.get("reserved_claim_name") != contract_stage["claim_name"]
            or stage_record.get("reserved_result_name") != contract_stage["result_name"]
            or stage_record.get("selected_job_ids") != contract_stage["selected_job_ids"]
            or stage_record.get("source_roles") != contract_stage["source_roles"]
            or stage_record.get("hand_indices") != contract_stage["hand_indices"]
            or stage_record.get("root_record_digest")
            != contract_stage["root_record_digest"]
            or stage_record.get("prerequisite_stage_id")
            != (None if contract_stage["stage_id"] == plan.STAGE1_ID else plan.STAGE1_ID)
        ):
            raise ValueError("diagnostic package stage changed")
        jobs = stage_record.get("jobs")
        if not isinstance(jobs, list) or [row.get("job_id") for row in jobs] != contract_stage["selected_job_ids"]:
            raise ValueError("diagnostic package exact job set changed")
        for job_record in jobs:
            path = root / str(job_record["path"])
            job = validate_job_manifest(
                _read_object(path, "diagnostic packaged job"),
                contract=contract,
            )
            if (
                job_record.get("source_role") != job["source_role"]
                or job_record.get("sha256") != sha256_file(path)
                or job_record.get("bytes") != path.stat().st_size
            ):
                raise ValueError("diagnostic package job record changed")
    entries = source.get("entries")
    if not isinstance(entries, list):
        raise ValueError("diagnostic source entries changed")
    expected_entry_meta: list[dict[str, Any]] = [
        {
            "path": "control/diagnostic_contract.json",
            "sha256": canonical_sha256(contract),
            "bytes": len(canonical_bytes(contract)),
        }
    ]
    seen_root_indices: set[int] = set()
    for contract_stage in contract["stages"]:
        for record in contract_stage["root_records"]:
            index = int(record["hand_index"])
            if index in seen_root_indices:
                continue
            seen_root_indices.add(index)
            expected_entry_meta.append(
                {
                    "path": f"roots/hand_{index:03d}.json",
                    "sha256": record["sha256"],
                    "bytes": record["size_bytes"],
                }
            )
    expected_entry_meta.sort(key=lambda row: row["path"])
    if entries != expected_entry_meta:
        raise ValueError("diagnostic source entries escaped the frozen root set")
    with zipfile.ZipFile(root / SOURCE_NAME) as archive:
        infos = archive.infolist()
        if [item.filename for item in infos] != [row.get("path") for row in entries]:
            raise ValueError("diagnostic source member set changed")
        for info, record in zip(infos, entries, strict=True):
            raw = archive.read(info.filename)
            if (
                info.file_size != record.get("bytes")
                or hashlib.sha256(raw).hexdigest() != record.get("sha256")
            ):
                raise ValueError("diagnostic source member changed")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != _expected_package_files(manifest):
        raise ValueError("diagnostic local package file set changed")
    if (
        ready
        != {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "diagnostic_local_package_complete_no_cloud_path",
            "package_manifest_sha256": sha256_file(root / MANIFEST_NAME),
            "source_sha256": sha256_file(root / SOURCE_NAME),
            "startup_sha256": sha256_file(root / STARTUP_NAME),
            "stage_job_counts": [len(row["jobs"]) for row in stages],
            "cloud_capable": False,
            "gcloud_callable": False,
            "claim_written": False,
            "authorization_written": False,
            "current_profile_changed": False,
        }
    ):
        raise ValueError("diagnostic package ready marker changed")
    return manifest


def _manifest_stage(manifest: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    for row in manifest["stages"]:
        if row["stage_id"] == stage_id:
            return dict(row)
    raise ValueError(f"diagnostic package has no stage: {stage_id}")


def _validate_embedded_prerequisite_receipt(
    *,
    package_dir: Path,
    manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    receipt: Any,
) -> dict[str, Any] | None:
    prerequisite_id = stage["prerequisite_stage_id"]
    if prerequisite_id is None:
        if receipt is not None:
            raise ValueError("stage1 prerequisite receipt must be absent")
        return None
    if not isinstance(receipt, Mapping):
        raise ValueError("stage2 embedded prerequisite receipt is missing")
    value = dict(receipt)
    _exact_keys(
        value,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "job_records",
            "job_record_aggregate_sha256",
            "diagnostic_only",
            "cloud_receive_performed",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
            "current_profile_changed",
        },
        "diagnostic embedded prerequisite receipt",
    )
    prerequisite_stage = _manifest_stage(manifest, prerequisite_id)
    records = value.get("job_records")
    if not isinstance(records, list) or len(records) != len(
        prerequisite_stage["selected_job_ids"]
    ):
        raise ValueError("stage2 embedded prerequisite job records changed")
    for record, job_id, source_role in zip(
        records,
        prerequisite_stage["selected_job_ids"],
        prerequisite_stage["source_roles"],
        strict=True,
    ):
        if not isinstance(record, Mapping):
            raise ValueError("stage2 embedded prerequisite job record is invalid")
        _exact_keys(
            record,
            {
                "job_id",
                "source_role",
                "done_path",
                "done_sha256",
                "upload_aggregate_sha256",
                "heartbeat_aggregate_sha256",
            },
            "diagnostic embedded prerequisite job record",
        )
        if (
            record.get("job_id") != job_id
            or record.get("source_role") != source_role
            or record.get("done_path") != f"jobs/{job_id}/{DONE_NAME}"
            or any(
                not _is_sha256(record.get(field))
                for field in (
                    "done_sha256",
                    "upload_aggregate_sha256",
                    "heartbeat_aggregate_sha256",
                )
            )
        ):
            raise ValueError("stage2 embedded prerequisite job binding changed")
    if (
        value.get("schema") != RECEIVE_SCHEMA
        or value.get("status") != "diagnostic_local_stage_received_and_validated"
        or value.get("package_manifest_sha256")
        != sha256_file(package_dir / MANIFEST_NAME)
        or value.get("stage_id") != prerequisite_id
        or value.get("run_name") != prerequisite_stage["run_name"]
        or value.get("selected_job_ids") != prerequisite_stage["selected_job_ids"]
        or value.get("job_record_aggregate_sha256")
        != canonical_sha256(records)
        or value.get("diagnostic_only") is not True
        or any(
            value.get(field) is not False
            for field in (
                "cloud_receive_performed",
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("stage2 embedded prerequisite receipt changed")
    return value


def open_local_stage(
    *,
    package_dir: str | Path,
    stage_id: str,
    output_path: str | Path,
    prerequisite_receive_dir: str | Path | None = None,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    manifest = validate_local_package(package, contract_path=contract_path)
    stage = _manifest_stage(manifest, stage_id)
    prerequisite_sha: str | None = None
    prerequisite_receipt: dict[str, Any] | None = None
    expected_prerequisite = stage["prerequisite_stage_id"]
    if expected_prerequisite is None:
        if prerequisite_receive_dir is not None:
            raise ValueError("stage1 must not bind a prerequisite receipt")
    else:
        if prerequisite_receive_dir is None:
            raise ValueError("stage2 requires the validated stage1 receive receipt")
        receipt = validate_received_stage(
            package_dir=package,
            receive_dir=prerequisite_receive_dir,
            stage_id=expected_prerequisite,
            contract_path=contract_path,
        )
        prerequisite_receipt = _validate_embedded_prerequisite_receipt(
            package_dir=package,
            manifest=manifest,
            stage=stage,
            receipt=receipt,
        )
        prerequisite_sha = sha256_file(Path(prerequisite_receive_dir) / RECEIPT_NAME)
        if receipt["stage_id"] != expected_prerequisite:
            raise ValueError("diagnostic stage prerequisite changed")
    value = {
        "schema": STAGE_OPEN_SCHEMA,
        "status": "local_stage_open_for_offline_lifecycle_validation_only",
        "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "selected_job_ids": list(stage["selected_job_ids"]),
        "prerequisite_stage_id": expected_prerequisite,
        "prerequisite_receive_receipt": prerequisite_receipt,
        "prerequisite_receive_receipt_sha256": prerequisite_sha,
        "cloud_launch_authorized": False,
        "gcloud_invocation_authorized": False,
        "spot_claim_written": False,
        "authorization_written": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    _write_once(Path(output_path), value)
    return validate_stage_open(
        package_dir=package,
        stage_open_path=output_path,
        contract_path=contract_path,
    )


def validate_stage_open(
    *,
    package_dir: str | Path,
    stage_open_path: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    manifest = validate_local_package(package, contract_path=contract_path)
    value = _read_object(stage_open_path, "diagnostic local stage-open token")
    stage = _manifest_stage(manifest, str(value.get("stage_id")))
    prerequisite_receipt = _validate_embedded_prerequisite_receipt(
        package_dir=package,
        manifest=manifest,
        stage=stage,
        receipt=value.get("prerequisite_receive_receipt"),
    )
    expected = {
        "schema": STAGE_OPEN_SCHEMA,
        "status": "local_stage_open_for_offline_lifecycle_validation_only",
        "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "stage_id": stage["stage_id"],
        "run_name": stage["run_name"],
        "selected_job_ids": stage["selected_job_ids"],
        "prerequisite_stage_id": stage["prerequisite_stage_id"],
        "prerequisite_receive_receipt": prerequisite_receipt,
        "prerequisite_receive_receipt_sha256": value.get(
            "prerequisite_receive_receipt_sha256"
        ),
        "cloud_launch_authorized": False,
        "gcloud_invocation_authorized": False,
        "spot_claim_written": False,
        "authorization_written": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    if value != expected:
        raise ValueError("diagnostic local stage-open token changed")
    if stage["prerequisite_stage_id"] is None:
        if (
            value["prerequisite_receive_receipt"] is not None
            or value["prerequisite_receive_receipt_sha256"] is not None
        ):
            raise ValueError("stage1 prerequisite receipt must be absent")
    elif (
        not _is_sha256(value["prerequisite_receive_receipt_sha256"])
        or value["prerequisite_receive_receipt_sha256"]
        != canonical_sha256(prerequisite_receipt)
    ):
        raise ValueError("stage2 prerequisite receipt hash changed")
    return value


def run_startup_precontent(
    *,
    package_dir: str | Path,
    stage_open_path: str | Path,
    stage_id: str,
    job_id: str,
    source_override: str | Path | None = None,
    manifest_override: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the standalone verifier; it has no ``ofc_regular`` or cloud import."""

    package = Path(package_dir).resolve()
    source = Path(source_override) if source_override else package / SOURCE_NAME
    manifest = Path(manifest_override) if manifest_override else package / MANIFEST_NAME
    job = _job_path(package, stage_id, job_id)
    return subprocess.run(
        [
            sys.executable,
            str(package / STARTUP_NAME),
            str(source),
            str(manifest),
            str(stage_open_path),
            str(job),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def initialize_local_job(
    *,
    package_dir: str | Path,
    stage_open_path: str | Path,
    stage_id: str,
    job_id: str,
    output_dir: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    manifest = validate_local_package(package, contract_path=contract_path)
    opened = validate_stage_open(
        package_dir=package,
        stage_open_path=stage_open_path,
        contract_path=contract_path,
    )
    stage = _manifest_stage(manifest, stage_id)
    if opened["stage_id"] != stage_id or job_id not in stage["selected_job_ids"]:
        raise ValueError("diagnostic local job is outside the opened stage")
    startup = run_startup_precontent(
        package_dir=package,
        stage_open_path=stage_open_path,
        stage_id=stage_id,
        job_id=job_id,
    )
    if startup.returncode != 0:
        raise ValueError(f"diagnostic startup pre-content failed: {startup.stderr}")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(_job_path(package, stage_id, job_id), target / "job_manifest.json")
    shutil.copyfile(stage_open_path, target / LOCAL_STAGE_OPEN_NAME)
    (target / "uploads").mkdir()
    (target / "heartbeats").mkdir()
    return _read_object(target / "job_manifest.json", "initialized diagnostic job")


def _load_job_context(
    *,
    package_dir: Path,
    output_dir: Path,
    contract_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = validate_local_package(package_dir, contract_path=contract_path)
    contract = plan.validate_frozen_contract(contract_path)
    job = validate_job_manifest(
        _read_object(output_dir / "job_manifest.json", "diagnostic output job"),
        contract=contract,
    )
    opened = validate_stage_open(
        package_dir=package_dir,
        stage_open_path=output_dir / LOCAL_STAGE_OPEN_NAME,
        contract_path=contract_path,
    )
    if opened["stage_id"] != job["stage_id"] or job["job_id"] not in opened["selected_job_ids"]:
        raise ValueError("diagnostic output job/open binding changed")
    return manifest, job, opened


def _upload_path(output_dir: Path, index: int) -> Path:
    return output_dir / "uploads" / f"hand_{index:03d}.json"


def _heartbeat_path(output_dir: Path, sequence: int) -> Path:
    return output_dir / "heartbeats" / f"{sequence:06d}.json"


def record_local_upload(
    *,
    package_dir: str | Path,
    output_dir: str | Path,
    hand_index: int,
    payload: Mapping[str, Any],
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    target = Path(output_dir).resolve()
    _manifest, job, _opened = _load_job_context(
        package_dir=package,
        output_dir=target,
        contract_path=contract_path,
    )
    if (target / DONE_NAME).exists():
        raise FileExistsError("diagnostic DONE already exists")
    if type(hand_index) is not int or hand_index not in job["work_hand_indices"]:
        raise ValueError("diagnostic upload hand is outside the exact job set")
    _reject_hidden(payload)
    root_record = next(
        row for row in job["root_records"] if row["hand_index"] == hand_index
    )
    value = {
        "schema": UPLOAD_SCHEMA,
        "status": "diagnostic_local_hand_uploaded_write_once",
        "stage_id": job["stage_id"],
        "run_name": job["run_name"],
        "job_id": job["job_id"],
        "source_role": job["source_role"],
        "hand_index": hand_index,
        "root_sha256": root_record["sha256"],
        "payload": dict(payload),
        "payload_sha256": canonical_sha256(payload),
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    upload = _upload_path(target, hand_index)
    existing_indices = [
        index
        for index in job["work_hand_indices"]
        if _upload_path(target, index).is_file()
    ]
    if upload.exists():
        if _read_object(upload, "existing diagnostic upload") != value:
            raise ValueError("diagnostic resume upload differs from immutable artifact")
    else:
        next_pending = next(
            (
                index
                for index in job["work_hand_indices"]
                if index not in existing_indices
            ),
            None,
        )
        if hand_index != next_pending:
            raise ValueError(
                "diagnostic uploads must follow the frozen work-hand order"
            )
        _write_once(upload, value)
    completed = [
        index for index in job["work_hand_indices"] if _upload_path(target, index).is_file()
    ]
    sequence = completed.index(hand_index) + 1
    # A resumed call for an already-recorded hand validates the corresponding
    # deterministic heartbeat instead of creating an extra heartbeat.
    heartbeat = {
        "schema": HEARTBEAT_SCHEMA,
        "status": "diagnostic_local_progress_write_once",
        "stage_id": job["stage_id"],
        "run_name": job["run_name"],
        "job_id": job["job_id"],
        "source_role": job["source_role"],
        "sequence": sequence,
        "completed_hand_indices": completed[:sequence],
        "pending_hand_indices": [
            index for index in job["work_hand_indices"] if index not in completed[:sequence]
        ],
        "latest_upload_sha256": sha256_file(upload),
        "diagnostic_only": True,
        "cloud_upload_performed": False,
        "current_profile_changed": False,
    }
    heartbeat_path = _heartbeat_path(target, sequence)
    if heartbeat_path.exists():
        if _read_object(heartbeat_path, "existing diagnostic heartbeat") != heartbeat:
            raise ValueError("diagnostic resume heartbeat differs from immutable artifact")
    else:
        _write_once(heartbeat_path, heartbeat)
    return value


def _upload_records(output_dir: Path, job: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in job["work_hand_indices"]:
        path = _upload_path(output_dir, index)
        if not path.is_file():
            raise ValueError("diagnostic job upload set is incomplete")
        value = _read_object(path, "diagnostic upload")
        root_record = next(
            row for row in job["root_records"] if row["hand_index"] == index
        )
        if (
            value.get("schema") != UPLOAD_SCHEMA
            or value.get("stage_id") != job["stage_id"]
            or value.get("run_name") != job["run_name"]
            or value.get("job_id") != job["job_id"]
            or value.get("source_role") != job["source_role"]
            or value.get("hand_index") != index
            or value.get("root_sha256") != root_record["sha256"]
            or value.get("payload_sha256") != canonical_sha256(value.get("payload"))
            or value.get("diagnostic_only") is not True
            or any(
                value.get(field) is not False
                for field in (
                    "performance_lock_evidence",
                    "quality_evidence",
                    "training_eligible",
                    "promotion_evidence",
                    "current_profile_changed",
                )
            )
        ):
            raise ValueError("diagnostic upload changed")
        _reject_hidden(value["payload"])
        records.append(
            {
                "hand_index": index,
                "path": f"uploads/hand_{index:03d}.json",
                "sha256": sha256_file(path),
            }
        )
    return records


def _heartbeat_records(output_dir: Path, job: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    work = job["work_hand_indices"]
    for sequence, index in enumerate(work, 1):
        path = _heartbeat_path(output_dir, sequence)
        value = _read_object(path, "diagnostic heartbeat")
        upload = _upload_path(output_dir, index)
        expected = {
            "schema": HEARTBEAT_SCHEMA,
            "status": "diagnostic_local_progress_write_once",
            "stage_id": job["stage_id"],
            "run_name": job["run_name"],
            "job_id": job["job_id"],
            "source_role": job["source_role"],
            "sequence": sequence,
            "completed_hand_indices": work[:sequence],
            "pending_hand_indices": work[sequence:],
            "latest_upload_sha256": sha256_file(upload),
            "diagnostic_only": True,
            "cloud_upload_performed": False,
            "current_profile_changed": False,
        }
        if value != expected:
            raise ValueError("diagnostic heartbeat changed")
        records.append(
            {
                "sequence": sequence,
                "path": f"heartbeats/{sequence:06d}.json",
                "sha256": sha256_file(path),
            }
        )
    return records


def complete_local_job(
    *,
    package_dir: str | Path,
    output_dir: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    target = Path(output_dir).resolve()
    _manifest, job, opened = _load_job_context(
        package_dir=package,
        output_dir=target,
        contract_path=contract_path,
    )
    uploads = _upload_records(target, job)
    heartbeats = _heartbeat_records(target, job)
    value = {
        "schema": DONE_SCHEMA,
        "status": "diagnostic_local_job_complete_done_published_last",
        "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "stage_open_sha256": sha256_file(target / LOCAL_STAGE_OPEN_NAME),
        "job_manifest_sha256": sha256_file(target / "job_manifest.json"),
        "stage_id": job["stage_id"],
        "run_name": job["run_name"],
        "job_id": job["job_id"],
        "source_role": job["source_role"],
        "work_hand_indices": job["work_hand_indices"],
        "completed_hand_indices": job["work_hand_indices"],
        "upload_records": uploads,
        "heartbeat_records": heartbeats,
        "upload_aggregate_sha256": canonical_sha256(uploads),
        "heartbeat_aggregate_sha256": canonical_sha256(heartbeats),
        "prerequisite_receive_receipt_sha256": opened[
            "prerequisite_receive_receipt_sha256"
        ],
        "diagnostic_only": True,
        "cloud_execution_performed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    done = target / DONE_NAME
    if done.exists():
        if _read_object(done, "existing diagnostic DONE") != value:
            raise ValueError("diagnostic DONE differs on resume")
    else:
        _write_once(done, value)
    return validate_completed_local_job(
        package_dir=package,
        output_dir=target,
        contract_path=contract_path,
    )


def _expected_job_files(job: Mapping[str, Any], *, complete: bool) -> set[str]:
    files = {"job_manifest.json", LOCAL_STAGE_OPEN_NAME}
    for sequence, index in enumerate(job["work_hand_indices"], 1):
        files.add(f"uploads/hand_{index:03d}.json")
        files.add(f"heartbeats/{sequence:06d}.json")
    if complete:
        files.add(DONE_NAME)
    return files


def validate_completed_local_job(
    *,
    package_dir: str | Path,
    output_dir: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    target = Path(output_dir).resolve()
    _manifest, job, opened = _load_job_context(
        package_dir=package,
        output_dir=target,
        contract_path=contract_path,
    )
    uploads = _upload_records(target, job)
    heartbeats = _heartbeat_records(target, job)
    expected = {
        "schema": DONE_SCHEMA,
        "status": "diagnostic_local_job_complete_done_published_last",
        "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "stage_open_sha256": sha256_file(target / LOCAL_STAGE_OPEN_NAME),
        "job_manifest_sha256": sha256_file(target / "job_manifest.json"),
        "stage_id": job["stage_id"],
        "run_name": job["run_name"],
        "job_id": job["job_id"],
        "source_role": job["source_role"],
        "work_hand_indices": job["work_hand_indices"],
        "completed_hand_indices": job["work_hand_indices"],
        "upload_records": uploads,
        "heartbeat_records": heartbeats,
        "upload_aggregate_sha256": canonical_sha256(uploads),
        "heartbeat_aggregate_sha256": canonical_sha256(heartbeats),
        "prerequisite_receive_receipt_sha256": opened[
            "prerequisite_receive_receipt_sha256"
        ],
        "diagnostic_only": True,
        "cloud_execution_performed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    done = _read_object(target / DONE_NAME, "diagnostic DONE")
    if done != expected:
        raise ValueError("diagnostic completed job changed")
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    if actual != _expected_job_files(job, complete=True):
        raise ValueError("diagnostic completed job file set changed")
    return done


def receive_stage(
    *,
    package_dir: str | Path,
    stage_id: str,
    job_outputs: Mapping[str, str | Path],
    destination: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    manifest = validate_local_package(package, contract_path=contract_path)
    stage = _manifest_stage(manifest, stage_id)
    if list(job_outputs) != stage["selected_job_ids"]:
        raise ValueError("diagnostic receive requires the exact ordered stage job set")
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError("diagnostic receive destination is immutable")
    staging = target.parent / f".{target.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError("stale diagnostic receive staging exists")
    staging.mkdir(parents=True)
    try:
        records: list[dict[str, Any]] = []
        for job_id in stage["selected_job_ids"]:
            source = Path(job_outputs[job_id]).resolve()
            done = validate_completed_local_job(
                package_dir=package,
                output_dir=source,
                contract_path=contract_path,
            )
            if done["job_id"] != job_id or done["stage_id"] != stage_id:
                raise ValueError("diagnostic receive job identity changed")
            destination_job = staging / "jobs" / job_id
            shutil.copytree(source, destination_job)
            records.append(
                {
                    "job_id": job_id,
                    "source_role": done["source_role"],
                    "done_path": f"jobs/{job_id}/{DONE_NAME}",
                    "done_sha256": sha256_file(source / DONE_NAME),
                    "upload_aggregate_sha256": done["upload_aggregate_sha256"],
                    "heartbeat_aggregate_sha256": done["heartbeat_aggregate_sha256"],
                }
            )
        receipt = {
            "schema": RECEIVE_SCHEMA,
            "status": "diagnostic_local_stage_received_and_validated",
            "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
            "stage_id": stage_id,
            "run_name": stage["run_name"],
            "selected_job_ids": stage["selected_job_ids"],
            "job_records": records,
            "job_record_aggregate_sha256": canonical_sha256(records),
            "diagnostic_only": True,
            "cloud_receive_performed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
        _write_once(staging / RECEIPT_NAME, receipt)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_received_stage(
        package_dir=package,
        receive_dir=target,
        stage_id=stage_id,
        contract_path=contract_path,
    )


def validate_received_stage(
    *,
    package_dir: str | Path,
    receive_dir: str | Path,
    stage_id: str,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    target = Path(receive_dir).resolve()
    manifest = validate_local_package(package, contract_path=contract_path)
    stage = _manifest_stage(manifest, stage_id)
    records: list[dict[str, Any]] = []
    for job_id in stage["selected_job_ids"]:
        job_dir = target / "jobs" / job_id
        done = validate_completed_local_job(
            package_dir=package,
            output_dir=job_dir,
            contract_path=contract_path,
        )
        if done["job_id"] != job_id or done["stage_id"] != stage_id:
            raise ValueError("diagnostic received job identity changed")
        records.append(
            {
                "job_id": job_id,
                "source_role": done["source_role"],
                "done_path": f"jobs/{job_id}/{DONE_NAME}",
                "done_sha256": sha256_file(job_dir / DONE_NAME),
                "upload_aggregate_sha256": done["upload_aggregate_sha256"],
                "heartbeat_aggregate_sha256": done["heartbeat_aggregate_sha256"],
            }
        )
    expected = {
        "schema": RECEIVE_SCHEMA,
        "status": "diagnostic_local_stage_received_and_validated",
        "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "selected_job_ids": stage["selected_job_ids"],
        "job_records": records,
        "job_record_aggregate_sha256": canonical_sha256(records),
        "diagnostic_only": True,
        "cloud_receive_performed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    receipt = _read_object(target / RECEIPT_NAME, "diagnostic receive receipt")
    if receipt != expected:
        raise ValueError("diagnostic receive receipt changed")
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    expected_files = {RECEIPT_NAME}
    for job_id in stage["selected_job_ids"]:
        job = _read_object(
            target / "jobs" / job_id / "job_manifest.json",
            "received diagnostic job",
        )
        expected_files.update(
            f"jobs/{job_id}/{relative}"
            for relative in _expected_job_files(job, complete=True)
        )
    if actual != expected_files:
        raise ValueError("diagnostic receive file set changed")
    return receipt


def aggregate_received_stages(
    *,
    package_dir: str | Path,
    received_stages: Mapping[str, str | Path],
    output_path: str | Path | None = None,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    manifest = validate_local_package(package, contract_path=contract_path)
    expected_stage_ids = [row["stage_id"] for row in manifest["stages"]]
    if list(received_stages) != expected_stage_ids:
        raise ValueError("diagnostic aggregate requires both exact stages in order")
    stage_records: list[dict[str, Any]] = []
    semantic_uploads: list[dict[str, Any]] = []
    for stage_id in expected_stage_ids:
        receive_dir = Path(received_stages[stage_id]).resolve()
        receipt = validate_received_stage(
            package_dir=package,
            receive_dir=receive_dir,
            stage_id=stage_id,
            contract_path=contract_path,
        )
        stage_records.append(
            {
                "stage_id": stage_id,
                "run_name": receipt["run_name"],
                "receipt_sha256": sha256_file(receive_dir / RECEIPT_NAME),
                "job_record_aggregate_sha256": receipt[
                    "job_record_aggregate_sha256"
                ],
            }
        )
        for job_record in receipt["job_records"]:
            job_id = job_record["job_id"]
            job_dir = receive_dir / "jobs" / job_id
            job = _read_object(job_dir / "job_manifest.json", "aggregate job")
            for index in job["work_hand_indices"]:
                upload = _read_object(
                    _upload_path(job_dir, index),
                    "aggregate diagnostic upload",
                )
                semantic_uploads.append(
                    {
                        "stage_id": stage_id,
                        "job_id": job_id,
                        "source_role": job["source_role"],
                        "hand_index": index,
                        "root_sha256": upload["root_sha256"],
                        "payload_sha256": upload["payload_sha256"],
                    }
                )
    value = {
        "schema": AGGREGATE_SCHEMA,
        "status": "diagnostic_local_lifecycle_aggregate_complete",
        "package_manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "stage_records": stage_records,
        "semantic_uploads": semantic_uploads,
        "semantic_upload_aggregate_sha256": canonical_sha256(semantic_uploads),
        "diagnostic_only": True,
        "cloud_execution_performed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    if output_path is not None:
        _write_once(Path(output_path), value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate-package")
    validate.add_argument("--package-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = build_local_package(output_dir=args.output_dir)
    else:
        result = validate_local_package(args.package_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_SCHEMA",
    "DONE_SCHEMA",
    "HEARTBEAT_SCHEMA",
    "PACKAGE_SCHEMA",
    "RECEIVE_SCHEMA",
    "STAGE_OPEN_SCHEMA",
    "UPLOAD_SCHEMA",
    "aggregate_received_stages",
    "build_local_package",
    "complete_local_job",
    "initialize_local_job",
    "open_local_stage",
    "record_local_upload",
    "run_startup_precontent",
    "validate_completed_local_job",
    "validate_local_package",
    "validate_received_stage",
    "validate_stage_open",
]
