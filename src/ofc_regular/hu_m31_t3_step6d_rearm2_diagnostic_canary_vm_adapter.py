"""Fail-closed, local-only VM transport preview for the rearm2 canaries.

This module is intentionally incapable of launching anything.  It translates
the accepted diagnostic package into an exact one-job or two-job transport
preview, validates simulated remote lifecycle objects, and audits the boundary
that a later cloud implementation must satisfy.

There is deliberately no gcloud/subprocess call, claim writer, launch
authorization writer, startup shell generator, object uploader, or VM create
API in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import ActionKey
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_local as local
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan


ADAPTER_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_adapter_v1"
PREVIEW_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_preview_v1"
STAGE_TOKEN_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_stage_token_v1"
REMOTE_MANIFEST_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_remote_manifest_v1"
UPLOAD_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_upload_v1"
HEARTBEAT_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_heartbeat_v1"
DONE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_done_v1"
FAILURE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_failure_v1"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_receive_v1"
SNAPSHOT_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_vm_snapshot_v1"

DEFAULT_PROJECT = "ofc-solver-485418"
DEFAULT_BUCKET = "pokerhu-ofc-solver-485418-training"
DEFAULT_ZONE = "asia-northeast1-b"
MACHINE_TYPE = "c4-standard-16"
SPOT_PRICE_CEILING_USD_PER_VM_HOUR = 0.57
MAX_RUNTIME_SECONDS_PER_VM = 4200
WATCHDOG_SECONDS_PER_VM = 3900
HEARTBEAT_INTERVAL_SECONDS = 60
MAX_ATTEMPTS_PER_JOB = 2
DIAGNOSTIC_COMPUTE_CAP_USD = 4.0

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_CODES = frozenset(
    {
        "startup_identity_rejected",
        "watchdog_timeout",
        "solver_nonzero_exit",
        "upload_validation_rejected",
        "unexpected_runtime_error",
    }
)


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
    return local.sha256_file(path)


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase sha256")
    return value


def _safe_identifier(value: str, label: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}")
    return value


def _manifest_stage(manifest: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in manifest["stages"]:
        if stage["stage_id"] == stage_id:
            return dict(stage)
    raise ValueError(f"unknown diagnostic stage: {stage_id}")


def _receipt_binding(
    *,
    package_dir: Path,
    stage_id: str,
    prerequisite_receive_dir: str | Path | None,
    contract_path: str | Path,
) -> dict[str, Any] | None:
    if stage_id == plan.STAGE1_ID:
        if prerequisite_receive_dir is not None:
            raise ValueError("stage1 must not bind a stage1 receive receipt")
        return None
    if stage_id != plan.STAGE2_ID:
        raise ValueError("adapter permits only the frozen stage1 or stage2")
    if prerequisite_receive_dir is None:
        raise ValueError("stage2 requires an actual validated stage1 receive directory")
    receive_dir = Path(prerequisite_receive_dir).resolve()
    receipt = local.validate_received_stage(
        package_dir=package_dir,
        receive_dir=receive_dir,
        stage_id=plan.STAGE1_ID,
        contract_path=contract_path,
    )
    receipt_path = receive_dir / local.RECEIPT_NAME
    return {
        "stage_id": receipt["stage_id"],
        "run_name": receipt["run_name"],
        "selected_job_ids": receipt["selected_job_ids"],
        "package_manifest_sha256": receipt["package_manifest_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "job_record_aggregate_sha256": receipt["job_record_aggregate_sha256"],
    }


def _cost_guard(vm_count: int, observed_spot_price: float | None) -> dict[str, Any]:
    if type(vm_count) is not int or vm_count not in (1, 2):
        raise ValueError("diagnostic stage VM count must be exactly one or two")
    if observed_spot_price is not None and (
        isinstance(observed_spot_price, bool)
        or not isinstance(observed_spot_price, (int, float))
        or observed_spot_price <= 0
        or observed_spot_price > SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    ):
        raise ValueError("observed Spot price exceeds the frozen planning ceiling")
    initial = (
        vm_count
        * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        * MAX_RUNTIME_SECONDS_PER_VM
        / 3600
    )
    all_attempts = initial * MAX_ATTEMPTS_PER_JOB
    if all_attempts > DIAGNOSTIC_COMPUTE_CAP_USD:
        raise ValueError("diagnostic stage exceeds its compute cap")
    return {
        "currency": "USD",
        "planning_price_source": (
            "caller_supplied_read_only_observation"
            if observed_spot_price is not None
            else "frozen_ceiling_only_no_live_price_query"
        ),
        "observed_spot_price_usd_per_vm_hour": observed_spot_price,
        "spot_price_ceiling_usd_per_vm_hour": SPOT_PRICE_CEILING_USD_PER_VM_HOUR,
        "max_runtime_seconds_per_vm": MAX_RUNTIME_SECONDS_PER_VM,
        "watchdog_seconds_per_vm": WATCHDOG_SECONDS_PER_VM,
        "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
        "vm_count": vm_count,
        "initial_estimated_max_compute_usd": initial,
        "all_attempts_estimated_max_compute_usd": all_attempts,
        "diagnostic_compute_cap_usd": DIAGNOSTIC_COMPUTE_CAP_USD,
    }


def _stage_token(
    *,
    package_sha: str,
    stage: Mapping[str, Any],
    prerequisite: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": STAGE_TOKEN_SCHEMA,
        "status": "dry_run_identity_only_cloud_not_authorized",
        "package_manifest_sha256": package_sha,
        "stage_id": stage["stage_id"],
        "run_name": stage["run_name"],
        "selected_job_ids": stage["selected_job_ids"],
        "prerequisite_stage1_receive": prerequisite,
        "cloud_launch_authorized": False,
        "gcloud_invocation_authorized": False,
        "claim_written": False,
        "authorization_written": False,
        "diagnostic_only": True,
    }


def _job_record(
    *,
    package_dir: Path,
    package_manifest: Mapping[str, Any],
    stage: Mapping[str, Any],
    job_record: Mapping[str, Any],
    stage_token: Mapping[str, Any],
    project: str,
    bucket: str,
    zone: str,
    prefix: str,
    cost_guard_sha: str,
) -> dict[str, Any]:
    job_id = str(job_record["job_id"])
    job = json.loads((package_dir / job_record["path"]).read_text(encoding="utf-8"))
    output_prefix = f"{prefix}/results/jobs/{job_id}"
    progress_prefix = f"{prefix}/progress/jobs/{job_id}"
    attempt_id = f"{stage['run_name']}|{job_id}|attempt-0"
    instance_name = _safe_identifier(
        f"r2diag-{1 if stage['stage_id'] == plan.STAGE1_ID else 2}-"
        f"{'c' if job['source_role'] == 'candidate' else 'r'}-"
        f"{job_id.rsplit('-', 1)[-1]}-a0",
        "instance name",
    )
    upload_uris = [
        f"{output_prefix}/uploads/hand_{index:03d}.json"
        for index in job["work_hand_indices"]
    ]
    heartbeat_uris = [
        f"{progress_prefix}/heartbeats/{sequence:06d}.json"
        for sequence in range(1, len(job["work_hand_indices"]) + 1)
    ]
    metadata = {
        "ADAPTER_SCHEMA": ADAPTER_SCHEMA,
        "ATTEMPT_ID": attempt_id,
        "ATTEMPT_INDEX": "0",
        "BUCKET": bucket,
        "CLOUD_LAUNCH_AUTHORIZED": "0",
        "COST_GUARD_SHA256": cost_guard_sha,
        "DIAGNOSTIC_ONLY": "1",
        "DONE_URI": f"{output_prefix}/DONE.json",
        "FAILURE_URI": f"{output_prefix}/FAILURE.json",
        "GCLOUD_INVOCATION_AUTHORIZED": "0",
        "HEARTBEAT_INTERVAL_SECONDS": str(HEARTBEAT_INTERVAL_SECONDS),
        "HEARTBEAT_PREFIX": f"{progress_prefix}/heartbeats",
        "INSTANCE_NAME": instance_name,
        "JOB_ID": job_id,
        "JOB_MANIFEST_SHA256": job_record["sha256"],
        "JOB_MANIFEST_URI": f"{prefix}/source/{job_record['path']}",
        "MACHINE_TYPE": MACHINE_TYPE,
        "MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS_PER_VM),
        "PACKAGE_MANIFEST_SHA256": sha256_file(package_dir / local.MANIFEST_NAME),
        "PACKAGE_MANIFEST_URI": f"{prefix}/source/{local.MANIFEST_NAME}",
        "PRESERVE_VM_ON_FAILURE": "1",
        "PROJECT_ID": project,
        "PUBLISH_DONE_LAST": "1",
        "RESULT_PREFIX": output_prefix,
        "RUN_NAME": stage["run_name"],
        "SELF_DELETE_ON_SUCCESS": "1",
        "SOURCE_ROLE": job["source_role"],
        "SOURCE_SHA256": package_manifest["source"]["sha256"],
        "SOURCE_URI": f"{prefix}/source/{local.SOURCE_NAME}",
        "STAGE_ID": stage["stage_id"],
        "STAGE_TOKEN_SHA256": canonical_sha256(stage_token),
        "STAGE_TOKEN_URI": f"{prefix}/control/stage_token.json",
        "UPLOAD_PREFIX": f"{output_prefix}/uploads",
        "WATCHDOG_SECONDS": str(WATCHDOG_SECONDS_PER_VM),
        "ZONE": zone,
    }
    return {
        "job_id": job_id,
        "source_role": job["source_role"],
        "instance_name": instance_name,
        "attempt_id": attempt_id,
        "work_hand_indices": job["work_hand_indices"],
        "root_records": job["root_records"],
        "metadata": metadata,
        "metadata_sha256": canonical_sha256(metadata),
        "control_objects": [
            {
                "uri": f"{prefix}/source/{job_record['path']}",
                "sha256": job_record["sha256"],
                "bytes": job_record["bytes"],
            }
        ],
        "upload_uris": upload_uris,
        "heartbeat_uris": heartbeat_uris,
        "done_uri": f"{output_prefix}/DONE.json",
        "failure_uri": f"{output_prefix}/FAILURE.json",
        "success_policy": {
            "publish_done_last": True,
            "self_delete_requested_after_done_validation": True,
            "preserve_vm": False,
        },
        "failure_policy": {
            "publish_done": False,
            "self_delete_requested": False,
            "preserve_vm_for_diagnosis": True,
            "bounded_shutdown_required": True,
            "shutdown_deadline_seconds": MAX_RUNTIME_SECONDS_PER_VM,
        },
        "resume_policy": {
            "same_attempt_identity_only": True,
            "ordered_upload_heartbeat_prefix_only": True,
            "same_content_is_idempotent": True,
            "different_content_collision_is_fatal": True,
            "max_attempts": MAX_ATTEMPTS_PER_JOB,
        },
    }


def build_preview(
    *,
    package_dir: str | Path,
    stage_id: str,
    prerequisite_receive_dir: str | Path | None = None,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zone: str = DEFAULT_ZONE,
    observed_spot_price: float | None = None,
    _skip_validation: bool = False,
) -> dict[str, Any]:
    """Build a deterministic in-memory preview; no files or cloud state change."""

    if (
        project != DEFAULT_PROJECT
        or bucket != DEFAULT_BUCKET
        or zone != DEFAULT_ZONE
        or _SAFE_ID.fullmatch(project) is None
        or _SAFE_BUCKET.fullmatch(bucket) is None
    ):
        raise ValueError("diagnostic target escaped the frozen project/bucket/zone")
    package = Path(package_dir).resolve()
    manifest = local.validate_local_package(package, contract_path=contract_path)
    stage = _manifest_stage(manifest, stage_id)
    expected_jobs = (
        list(plan.STAGE1_JOB_IDS)
        if stage_id == plan.STAGE1_ID
        else list(plan.STAGE2_JOB_IDS)
        if stage_id == plan.STAGE2_ID
        else None
    )
    if expected_jobs is None or stage["selected_job_ids"] != expected_jobs:
        raise ValueError("adapter stage escaped the exact frozen job set")
    prerequisite = _receipt_binding(
        package_dir=package,
        stage_id=stage_id,
        prerequisite_receive_dir=prerequisite_receive_dir,
        contract_path=contract_path,
    )
    package_sha = sha256_file(package / local.MANIFEST_NAME)
    token = _stage_token(
        package_sha=package_sha,
        stage=stage,
        prerequisite=prerequisite,
    )
    guard = _cost_guard(len(expected_jobs), observed_spot_price)
    prefix = (
        f"gs://{bucket}/hu-m31-r2diag-vm-v1/{stage['run_name']}/"
        f"{package_sha[:16]}"
    )
    jobs = [
        _job_record(
            package_dir=package,
            package_manifest=manifest,
            stage=stage,
            job_record=job_record,
            stage_token=token,
            project=project,
            bucket=bucket,
            zone=zone,
            prefix=prefix,
            cost_guard_sha=canonical_sha256(guard),
        )
        for job_record in stage["jobs"]
    ]
    control = [
        {
            "uri": f"{prefix}/source/{local.MANIFEST_NAME}",
            "sha256": package_sha,
            "bytes": (package / local.MANIFEST_NAME).stat().st_size,
        },
        {
            "uri": f"{prefix}/source/{local.READY_NAME}",
            "sha256": sha256_file(package / local.READY_NAME),
            "bytes": (package / local.READY_NAME).stat().st_size,
        },
        {
            "uri": f"{prefix}/source/{local.SOURCE_NAME}",
            "sha256": sha256_file(package / local.SOURCE_NAME),
            "bytes": (package / local.SOURCE_NAME).stat().st_size,
        },
        {
            "uri": f"{prefix}/source/{local.STARTUP_NAME}",
            "sha256": sha256_file(package / local.STARTUP_NAME),
            "bytes": (package / local.STARTUP_NAME).stat().st_size,
        },
        {
            "uri": f"{prefix}/control/stage_token.json",
            "sha256": canonical_sha256(token),
            "bytes": len(canonical_bytes(token)),
        },
    ]
    for job in jobs:
        control.extend(job["control_objects"])
    remote_manifest = {
        "schema": REMOTE_MANIFEST_SCHEMA,
        "prefix": prefix,
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "freshness_preflight": {
            "entire_stage_prefix_must_be_empty": True,
            "conditional_create_generation_match": 0,
            "unknown_object_is_fatal": True,
            "permission_or_query_error_is_fatal": True,
            "performed_by_this_dry_run_module": False,
        },
        "control_objects": control,
        "job_prefixes": [
            {
                "job_id": job["job_id"],
                "upload_uris": job["upload_uris"],
                "heartbeat_uris": job["heartbeat_uris"],
                "done_uri": job["done_uri"],
                "failure_uri": job["failure_uri"],
            }
            for job in jobs
        ],
        "receive_uri": f"{prefix}/received/{stage['reserved_result_name']}",
    }
    value = {
        "schema": PREVIEW_SCHEMA,
        "status": "local_dry_run_preview_cloud_not_authorized",
        "adapter_schema": ADAPTER_SCHEMA,
        "package_manifest_sha256": package_sha,
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "selected_job_ids": expected_jobs,
        "vm_count": len(expected_jobs),
        "stage_token": token,
        "stage_token_sha256": canonical_sha256(token),
        "remote_manifest": remote_manifest,
        "remote_manifest_sha256": canonical_sha256(remote_manifest),
        "jobs": jobs,
        "cost_guard": guard,
        "cost_guard_sha256": canonical_sha256(guard),
        "capabilities": {
            "cloud_launch_authorized": False,
            "gcloud_invocation_authorized": False,
            "subprocess_invocation_authorized": False,
            "claim_write_authorized": False,
            "launch_authorization_write_authorized": False,
            "object_write_authorized": False,
            "vm_create_authorized": False,
            "production_all20_launcher_reused": False,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
    }
    if _skip_validation:
        return value
    return validate_preview(
        value,
        package_dir=package,
        prerequisite_receive_dir=prerequisite_receive_dir,
        contract_path=contract_path,
    )


def validate_preview(
    value: Mapping[str, Any],
    *,
    package_dir: str | Path,
    prerequisite_receive_dir: str | Path | None = None,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    preview = dict(value)
    _exact(
        preview,
        {
            "schema",
            "status",
            "adapter_schema",
            "package_manifest_sha256",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "vm_count",
            "stage_token",
            "stage_token_sha256",
            "remote_manifest",
            "remote_manifest_sha256",
            "jobs",
            "cost_guard",
            "cost_guard_sha256",
            "capabilities",
        },
        "diagnostic VM preview",
    )
    guard = preview.get("cost_guard")
    jobs = preview.get("jobs")
    if not isinstance(guard, Mapping) or not isinstance(jobs, list) or not jobs:
        raise ValueError("diagnostic VM preview structure changed")
    metadata = jobs[0].get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("diagnostic VM metadata changed")
    expected = build_preview(
        package_dir=package_dir,
        stage_id=str(preview.get("stage_id")),
        prerequisite_receive_dir=prerequisite_receive_dir,
        contract_path=contract_path,
        project=str(metadata.get("PROJECT_ID")),
        bucket=str(metadata.get("BUCKET")),
        zone=str(metadata.get("ZONE")),
        observed_spot_price=guard.get("observed_spot_price_usd_per_vm_hour"),
        _skip_validation=True,
    )
    if preview != expected:
        raise ValueError("diagnostic VM preview changed")
    uris: list[str] = [
        row["uri"] for row in preview["remote_manifest"]["control_objects"]
    ]
    for job in preview["jobs"]:
        uris.extend(job["upload_uris"])
        uris.extend(job["heartbeat_uris"])
        uris.extend((job["done_uri"], job["failure_uri"]))
    uris.append(preview["remote_manifest"]["receive_uri"])
    if len(uris) != len(set(uris)):
        raise ValueError("diagnostic remote object URI collision")
    return preview


def _validate_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("remote solver result must be an object")
    result = dict(value)
    _exact(
        result,
        {"schema", "action_key", "q_milli", "portable_sha256"},
        "remote solver result",
    )
    if (
        result["schema"] != "diagnostic_t3_result_v1"
        or not isinstance(result["action_key"], str)
        or type(result["q_milli"]) is not int
        or not -1_000_000 <= result["q_milli"] <= 1_000_000
    ):
        raise ValueError("remote solver result value changed")
    try:
        if ActionKey.from_token(result["action_key"]).to_token() != result["action_key"]:
            raise ValueError("non-canonical ActionKey")
    except (TypeError, ValueError) as exc:
        raise ValueError("remote solver result ActionKey changed") from exc
    _sha(result["portable_sha256"], "portable result digest")
    return result


def build_upload(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    sequence: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if job is None or type(sequence) is not int or not 1 <= sequence <= len(job["work_hand_indices"]):
        raise ValueError("remote upload is outside the exact job sequence")
    checked_result = _validate_result(result)
    index = job["work_hand_indices"][sequence - 1]
    root = job["root_records"][sequence - 1]
    return {
        "schema": UPLOAD_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": job_id,
        "source_role": job["source_role"],
        "attempt_id": job["attempt_id"],
        "preview_sha256": canonical_sha256(preview),
        "sequence": sequence,
        "hand_index": index,
        "root_sha256": root["sha256"],
        "result": checked_result,
        "result_sha256": canonical_sha256(checked_result),
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def build_heartbeat(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    uploads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if job is None or not uploads or len(uploads) > len(job["work_hand_indices"]):
        raise ValueError("remote heartbeat upload prefix is invalid")
    checked: list[dict[str, Any]] = []
    for sequence, upload in enumerate(uploads, 1):
        expected = build_upload(
            preview,
            job_id=job_id,
            sequence=sequence,
            result=upload.get("result", {}),
        )
        if dict(upload) != expected:
            raise ValueError("remote upload changed")
        checked.append(expected)
    sequence = len(checked)
    return {
        "schema": HEARTBEAT_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": job_id,
        "attempt_id": job["attempt_id"],
        "preview_sha256": canonical_sha256(preview),
        "sequence": sequence,
        "completed_hand_indices": job["work_hand_indices"][:sequence],
        "pending_hand_indices": job["work_hand_indices"][sequence:],
        "latest_upload_sha256": canonical_sha256(checked[-1]),
        "diagnostic_only": True,
    }


def build_done(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    uploads: Sequence[Mapping[str, Any]],
    heartbeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if job is None or len(uploads) != len(job["work_hand_indices"]) or len(heartbeats) != len(uploads):
        raise ValueError("DONE requires every ordered upload and heartbeat")
    checked_uploads: list[dict[str, Any]] = []
    checked_heartbeats: list[dict[str, Any]] = []
    for sequence, upload in enumerate(uploads, 1):
        expected_upload = build_upload(
            preview, job_id=job_id, sequence=sequence, result=upload.get("result", {})
        )
        if dict(upload) != expected_upload:
            raise ValueError("remote upload changed before DONE")
        checked_uploads.append(expected_upload)
        expected_heartbeat = build_heartbeat(
            preview, job_id=job_id, uploads=checked_uploads
        )
        if dict(heartbeats[sequence - 1]) != expected_heartbeat:
            raise ValueError("remote heartbeat changed before DONE")
        checked_heartbeats.append(expected_heartbeat)
    return {
        "schema": DONE_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": job_id,
        "attempt_id": job["attempt_id"],
        "preview_sha256": canonical_sha256(preview),
        "completed_hand_indices": job["work_hand_indices"],
        "upload_sha256s": [canonical_sha256(row) for row in checked_uploads],
        "heartbeat_sha256s": [canonical_sha256(row) for row in checked_heartbeats],
        "done_published_last": True,
        "self_delete_requested": True,
        "preserve_vm_for_diagnosis": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def build_failure(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    failure_code: str,
    completed_sequences: int,
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if (
        job is None
        or failure_code not in _FAILURE_CODES
        or type(completed_sequences) is not int
        or not 0 <= completed_sequences < len(job["work_hand_indices"])
    ):
        raise ValueError("remote failure record changed")
    return {
        "schema": FAILURE_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": job_id,
        "attempt_id": job["attempt_id"],
        "preview_sha256": canonical_sha256(preview),
        "failure_code": failure_code,
        "completed_sequences": completed_sequences,
        "done_published": False,
        "self_delete_requested": False,
        "preserve_vm_for_diagnosis": True,
        "bounded_shutdown_required": True,
        "shutdown_deadline_seconds": MAX_RUNTIME_SECONDS_PER_VM,
        "watchdog_seconds": WATCHDOG_SECONDS_PER_VM,
        "diagnostic_only": True,
    }


def _validate_done_record(
    preview: Mapping[str, Any],
    job: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    done = dict(value)
    _exact(
        done,
        {
            "schema",
            "stage_id",
            "run_name",
            "job_id",
            "attempt_id",
            "preview_sha256",
            "completed_hand_indices",
            "upload_sha256s",
            "heartbeat_sha256s",
            "done_published_last",
            "self_delete_requested",
            "preserve_vm_for_diagnosis",
            "diagnostic_only",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
        },
        "remote DONE",
    )
    count = len(job["work_hand_indices"])
    if (
        done["schema"] != DONE_SCHEMA
        or done["stage_id"] != preview["stage_id"]
        or done["run_name"] != preview["run_name"]
        or done["job_id"] != job["job_id"]
        or done["attempt_id"] != job["attempt_id"]
        or done["preview_sha256"] != canonical_sha256(preview)
        or done["completed_hand_indices"] != job["work_hand_indices"]
        or not isinstance(done["upload_sha256s"], list)
        or len(done["upload_sha256s"]) != count
        or not all(_SHA256.fullmatch(str(item)) for item in done["upload_sha256s"])
        or not isinstance(done["heartbeat_sha256s"], list)
        or len(done["heartbeat_sha256s"]) != count
        or not all(
            _SHA256.fullmatch(str(item)) for item in done["heartbeat_sha256s"]
        )
        or done["done_published_last"] is not True
        or done["self_delete_requested"] is not True
        or done["preserve_vm_for_diagnosis"] is not False
        or done["diagnostic_only"] is not True
        or any(
            done[field] is not False
            for field in (
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
            )
        )
    ):
        raise ValueError("remote DONE identity or safety flags changed")
    return done


def build_receive(
    preview: Mapping[str, Any],
    *,
    done_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [row.get("job_id") for row in done_records] != preview["selected_job_ids"]:
        raise ValueError("receive requires the exact ordered stage job set")
    checked_done: list[dict[str, Any]] = []
    for job, done in zip(preview["jobs"], done_records, strict=True):
        checked_done.append(_validate_done_record(preview, job, done))
    return {
        "schema": RECEIVE_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "selected_job_ids": preview["selected_job_ids"],
        "preview_sha256": canonical_sha256(preview),
        "done_sha256s": [canonical_sha256(row) for row in checked_done],
        "prerequisite_stage1_receive": preview["stage_token"][
            "prerequisite_stage1_receive"
        ],
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def validate_snapshot(
    preview: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an offline inventory; it never queries or writes remote state."""

    value = dict(snapshot)
    _exact(
        value,
        {
            "schema",
            "preview_sha256",
            "stage_id",
            "run_name",
            "objects",
            "cloud_query_performed",
            "cloud_write_performed",
        },
        "remote snapshot",
    )
    if (
        value["schema"] != SNAPSHOT_SCHEMA
        or value["preview_sha256"] != canonical_sha256(preview)
        or value["stage_id"] != preview["stage_id"]
        or value["run_name"] != preview["run_name"]
        or value["cloud_query_performed"] is not False
        or value["cloud_write_performed"] is not False
        or not isinstance(value["objects"], list)
    ):
        raise ValueError("remote snapshot identity changed")
    allowed: set[str] = set()
    for job in preview["jobs"]:
        allowed.update(job["upload_uris"])
        allowed.update(job["heartbeat_uris"])
        allowed.update((job["done_uri"], job["failure_uri"]))
    allowed.add(preview["remote_manifest"]["receive_uri"])
    by_uri: dict[str, dict[str, Any]] = {}
    for raw in value["objects"]:
        if not isinstance(raw, Mapping):
            raise ValueError("remote object record must be an object")
        record = dict(raw)
        _exact(record, {"uri", "generation", "content_sha256", "content"}, "remote object")
        if (
            record["uri"] not in allowed
            or record["uri"] in by_uri
            or type(record["generation"]) is not int
            or record["generation"] <= 0
            or record["content_sha256"] != canonical_sha256(record["content"])
        ):
            raise ValueError("remote object collision or stale identity")
        if not isinstance(record["content"], Mapping):
            raise ValueError("remote object content must be an object")
        by_uri[record["uri"]] = record
    states: list[dict[str, Any]] = []
    done_rows: list[dict[str, Any]] = []
    for job in preview["jobs"]:
        uploads: list[dict[str, Any]] = []
        heartbeats: list[dict[str, Any]] = []
        for sequence, (upload_uri, heartbeat_uri) in enumerate(
            zip(job["upload_uris"], job["heartbeat_uris"], strict=True), 1
        ):
            upload_record = by_uri.get(upload_uri)
            heartbeat_record = by_uri.get(heartbeat_uri)
            if (upload_record is None) != (heartbeat_record is None):
                raise ValueError("upload and heartbeat prefixes diverged")
            if upload_record is None:
                if any(
                    uri in by_uri
                    for uri in (
                        job["upload_uris"][sequence:]
                        + job["heartbeat_uris"][sequence:]
                    )
                ):
                    raise ValueError("remote uploads are not a contiguous prefix")
                break
            upload = build_upload(
                preview,
                job_id=job["job_id"],
                sequence=sequence,
                result=upload_record["content"].get("result", {}),
            )
            if upload_record["content"] != upload:
                raise ValueError("remote upload content changed")
            uploads.append(upload)
            heartbeat = build_heartbeat(
                preview, job_id=job["job_id"], uploads=uploads
            )
            if heartbeat_record["content"] != heartbeat:
                raise ValueError("remote heartbeat content changed")
            heartbeats.append(heartbeat)
        done_record = by_uri.get(job["done_uri"])
        failure_record = by_uri.get(job["failure_uri"])
        if done_record and failure_record:
            raise ValueError("job cannot publish both DONE and FAILURE")
        if done_record:
            expected_done = build_done(
                preview,
                job_id=job["job_id"],
                uploads=uploads,
                heartbeats=heartbeats,
            )
            if done_record["content"] != expected_done:
                raise ValueError("DONE was stale, changed, or published early")
            done_rows.append(expected_done)
            state = "success_self_delete_requested"
        elif failure_record:
            content = failure_record["content"]
            expected_failure = build_failure(
                preview,
                job_id=job["job_id"],
                failure_code=content.get("failure_code", ""),
                completed_sequences=len(uploads),
            )
            if content != expected_failure:
                raise ValueError("failure preservation record changed")
            state = "failure_vm_preserved"
        else:
            state = "clean" if not uploads else "resumable_same_attempt"
        states.append(
            {
                "job_id": job["job_id"],
                "completed_sequences": len(uploads),
                "state": state,
            }
        )
    receive_record = by_uri.get(preview["remote_manifest"]["receive_uri"])
    if receive_record:
        expected_receive = build_receive(preview, done_records=done_rows)
        if receive_record["content"] != expected_receive:
            raise ValueError("remote receive record changed or was published early")
    return {
        "schema": SNAPSHOT_SCHEMA,
        "preview_sha256": canonical_sha256(preview),
        "job_states": states,
        "receive_validated": receive_record is not None,
        "diagnostic_only": True,
    }


def audit(
    *,
    package_dir: str | Path,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    manifest = local.validate_local_package(package_dir, contract_path=contract_path)
    counts = [len(stage["selected_job_ids"]) for stage in manifest["stages"]]
    if counts != [1, 2]:
        raise ValueError("diagnostic package no longer has exact 1/2 job stages")
    combined_all_attempts = sum(
        _cost_guard(count, None)["all_attempts_estimated_max_compute_usd"]
        for count in counts
    )
    if combined_all_attempts > DIAGNOSTIC_COMPUTE_CAP_USD:
        raise ValueError("combined diagnostic canaries exceed cost cap")
    return {
        "schema": ADAPTER_SCHEMA,
        "status": "local_preview_and_validation_ready_cloud_not_authorized",
        "stage_job_counts": counts,
        "combined_all_attempts_estimated_max_compute_usd": combined_all_attempts,
        "diagnostic_compute_cap_usd": DIAGNOSTIC_COMPUTE_CAP_USD,
        "stage2_requires_actual_validated_stage1_receive": True,
        "remote_result_exact_key_allowlist": True,
        "opaque_result_payload_permitted": False,
        "gcloud_callable": False,
        "subprocess_callable": False,
        "claim_or_authorization_writer_present": False,
        "startup_shell_generator_present": False,
        "remote_collision_state_queried": False,
        "launch_ready": False,
        "production_all20_launcher_changed": False,
        "current_profile_changed": False,
    }


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--package-dir", type=Path, required=True)
    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("--package-dir", type=Path, required=True)
    preview_parser.add_argument("--stage-id", required=True)
    preview_parser.add_argument("--prerequisite-receive-dir", type=Path)
    preview_parser.add_argument("--project", default=DEFAULT_PROJECT)
    preview_parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    preview_parser.add_argument("--zone", default=DEFAULT_ZONE)
    preview_parser.add_argument("--observed-spot-price", type=float)
    validate_parser = commands.add_parser("validate-preview")
    validate_parser.add_argument("--package-dir", type=Path, required=True)
    validate_parser.add_argument("--preview", type=Path, required=True)
    validate_parser.add_argument("--prerequisite-receive-dir", type=Path)
    snapshot_parser = commands.add_parser("validate-snapshot")
    snapshot_parser.add_argument("--package-dir", type=Path, required=True)
    snapshot_parser.add_argument("--preview", type=Path, required=True)
    snapshot_parser.add_argument("--snapshot", type=Path, required=True)
    snapshot_parser.add_argument("--prerequisite-receive-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = audit(package_dir=args.package_dir)
    elif args.command == "preview":
        result = build_preview(
            package_dir=args.package_dir,
            stage_id=args.stage_id,
            prerequisite_receive_dir=args.prerequisite_receive_dir,
            project=args.project,
            bucket=args.bucket,
            zone=args.zone,
            observed_spot_price=args.observed_spot_price,
        )
    elif args.command == "validate-preview":
        result = validate_preview(
            _read_object(args.preview),
            package_dir=args.package_dir,
            prerequisite_receive_dir=args.prerequisite_receive_dir,
        )
    else:
        preview = validate_preview(
            _read_object(args.preview),
            package_dir=args.package_dir,
            prerequisite_receive_dir=args.prerequisite_receive_dir,
        )
        result = validate_snapshot(preview, _read_object(args.snapshot))
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_SCHEMA",
    "DONE_SCHEMA",
    "FAILURE_SCHEMA",
    "HEARTBEAT_SCHEMA",
    "PREVIEW_SCHEMA",
    "RECEIVE_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "UPLOAD_SCHEMA",
    "audit",
    "build_done",
    "build_failure",
    "build_heartbeat",
    "build_preview",
    "build_receive",
    "build_upload",
    "canonical_bytes",
    "canonical_sha256",
    "validate_preview",
    "validate_snapshot",
]
