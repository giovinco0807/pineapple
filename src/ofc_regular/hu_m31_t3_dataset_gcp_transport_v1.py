"""Cloud-neutral Spot transport contract for the 9,000-pair M3.1 dataset.

The first smoke shard is already complete.  This module schedules the exact
remaining 359 immutable 25-pair shards in 45 waves of at most eight C4 VMs.
It models two attempts, pair-granular create-only checkpoints and heartbeats,
terminal cleanup/absence evidence, and source-replayed receive into local
shard directories.  It contains no GCP client and never launches resources.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_portable_worker_v1 as portable
from . import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as gce_contract
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as c4_contract


TRANSPORT_PLAN_SCHEMA = "hu_m31_t3_dataset_gcp_transport_plan_v1"
ATTEMPT_LEDGER_SCHEMA = "hu_m31_t3_dataset_gcp_attempt_ledger_v1"
RESUME_PLAN_SCHEMA = "hu_m31_t3_dataset_gcp_resume_plan_v1"
WAVE_REQUEST_SCHEMA = "hu_m31_t3_dataset_gcp_wave_request_v1"
CHECKPOINT_SCHEMA = "hu_m31_t3_dataset_gcp_checkpoint_manifest_v1"
HEARTBEAT_SCHEMA = "hu_m31_t3_dataset_gcp_heartbeat_v1"
LIFECYCLE_SCHEMA = "hu_m31_t3_dataset_gcp_lifecycle_receipt_v1"
RECEIVE_SCHEMA = "hu_m31_t3_dataset_gcp_receive_receipt_v1"

PROJECT = gce_contract.PROJECT
REGION = gce_contract.REGION
ZONE = gce_contract.ZONE
MACHINE_TYPE = c4_contract.MACHINE_TYPE
VCPUS_PER_VM = c4_contract.VCPUS_PER_VM
MAX_CONCURRENT_VMS = 8
MAX_ATTEMPTS_PER_SHARD = 2
ATTEMPT_IDS = ("a00", "a01")
SHARD_PAIR_COUNT = dataset.SHARD_PAIR_COUNT
MAX_HEARTBEAT_SEQUENCE = 2_000
TOTAL_SHARD_COUNT = 360
CLOUD_SHARD_COUNT = 359
WAVE_COUNT = 45

_SHA = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[1-9][0-9]*$")
_RUN = re.compile(r"^[a-z][a-z0-9-]{2,62}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
_SHARD = re.compile(
    r"^(?:train|safety-fit|threshold-lock|diagnostic-holdout)-[0-9]{4}$"
)

CONTENT_KINDS = (
    "dataset_plan",
    "fresh_quality_gate",
    "smoke_gate",
    "portable_authorization",
    "smoke_shard_archive",
    "runtime_archive",
    "wheelhouse_archive",
    "candidate_library",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_dataset_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical M3.1 JSON") from exc
    if not isinstance(value, dict) or raw != dataset.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical M3.1 JSON")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return source


def _content_record(kind: str, path: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "source_path": str(path),
        "filename": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _safe_tar_inventory(
    path: Path, *, expected_root: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != expected_root
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise ValueError("archive contains an unsafe entry")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("archive contains an unsupported entry")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("archive member cannot be read")
                raw = stream.read()
                result.append(
                    {
                        "path": PurePosixPath(*name.parts[1:]).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("archive is not a safe tar archive") from exc
    paths = [row["path"] for row in result]
    if not result or len(paths) != len(set(paths)):
        raise ValueError("archive inventory is empty or duplicated")
    return sorted(result, key=lambda row: row["path"])


def _safe_zip_inventory(
    path: Path, *, expected_root: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != expected_root
                ):
                    raise ValueError("zip contains an unsafe entry")
                if info.is_dir():
                    continue
                raw = archive.read(info)
                result.append(
                    {
                        "path": PurePosixPath(*name.parts[1:]).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("wheelhouse is not a safe ZIP archive") from exc
    paths = [row["path"] for row in result]
    if not result or len(paths) != len(set(paths)):
        raise ValueError("wheelhouse inventory is empty or duplicated")
    return sorted(result, key=lambda row: row["path"])


def _smoke_archive_inventory(path: Path) -> list[dict[str, Any]]:
    return _safe_tar_inventory(path, expected_root="smoke_shard")


def _wave_rows(shards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    waves: list[dict[str, Any]] = []
    for wave_index, start in enumerate(range(0, len(shards), 8)):
        ids = [str(row["shard_id"]) for row in shards[start : start + 8]]
        waves.append(
            {
                "wave_index": wave_index,
                "shard_ids": ids,
                "shard_count": len(ids),
                "max_vm_count": len(ids),
                "prerequisite_wave_indices": list(range(wave_index)),
                "requires_prior_waves_complete": wave_index > 0,
                "requires_owned_compute_absent": True,
                "requires_worker_iam_removed_before_receive": True,
            }
        )
    return waves


def _attempt(
    *,
    execution_identity_sha256: str,
    shard_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    compact = shard_id.replace("-", "")[:28]
    instance_id = (
        f"ofc-m31-ds-{execution_identity_sha256[:10]}-"
        f"{compact}-{attempt_id}"
    )[:63].rstrip("-")
    prefix = (
        f"m31-dataset/{execution_identity_sha256}/shards/"
        f"{shard_id}/{attempt_id}"
    )
    return {
        "attempt_id": attempt_id,
        "instance_id": instance_id,
        "object_prefix": prefix,
        "checkpoint_object_format": f"{prefix}/checkpoints/%06d.json",
        "heartbeat_object_format": f"{prefix}/heartbeats/%06d.json",
        "file_object_prefix": f"{prefix}/files/",
    }


def build_transport_plan(
    *,
    run_name: str,
    bucket: str,
    dataset_plan_path: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_gate_path: str | Path,
    portable_authorization_path: str | Path,
    smoke_shard_directory: str | Path,
    smoke_shard_archive_path: str | Path,
    runtime_archive_path: str | Path,
    wheelhouse_archive_path: str | Path,
    candidate_library_path: str | Path,
) -> dict[str, Any]:
    if _RUN.fullmatch(run_name) is None or _BUCKET.fullmatch(bucket) is None:
        raise ValueError("dataset transport run name or bucket is invalid")
    plan_path = _plain_file(dataset_plan_path, "dataset plan")
    plan = dataset.validate_dataset_plan(
        _read_dataset_canonical(plan_path, "M3.1 dataset plan")
    )
    fresh_path = _plain_file(fresh_quality_gate_path, "fresh-quality gate")
    smoke_gate_source = _plain_file(smoke_gate_path, "dataset smoke gate")
    portable_path = _plain_file(
        portable_authorization_path, "portable fanout authorization"
    )
    portable_value = portable._read_canonical(  # type: ignore[attr-defined]
        portable_path, "portable fanout authorization"
    )
    expected_portable = portable.build_portable_fanout_authorization(
        plan=plan,
        fresh_quality_gate_path=fresh_path,
        smoke_gate_receipt_path=smoke_gate_source,
        smoke_shard_directory=smoke_shard_directory,
    )
    if portable_value != expected_portable:
        raise PermissionError(
            "portable fanout authorization differs from controller source replay"
        )
    smoke_archive = _plain_file(smoke_shard_archive_path, "smoke shard archive")
    runtime_archive = _plain_file(runtime_archive_path, "runtime archive")
    wheelhouse_archive = _plain_file(
        wheelhouse_archive_path, "wheelhouse archive"
    )
    library = _plain_file(candidate_library_path, "Candidate02 library")
    if sha256_file(library) != dataset.ACCEPTED_CANDIDATE_LIBRARY_SHA256:
        raise PermissionError("dataset Candidate02 library identity changed")
    smoke_inventory = _smoke_archive_inventory(smoke_archive)
    if smoke_inventory != portable_value["smoke_shard_inventory"]:
        raise PermissionError("smoke archive differs from replayed smoke shard")
    runtime_inventory = _safe_tar_inventory(
        runtime_archive, expected_root="runtime"
    )
    wheelhouse_inventory = _safe_zip_inventory(
        wheelhouse_archive, expected_root="wheelhouse"
    )
    content_paths = (
        plan_path,
        fresh_path,
        smoke_gate_source,
        portable_path,
        smoke_archive,
        runtime_archive,
        wheelhouse_archive,
        library,
    )
    content = [
        _content_record(kind, path)
        for kind, path in zip(CONTENT_KINDS, content_paths, strict=True)
    ]
    cloud_shards = [
        deepcopy(dict(row))
        for row in plan["shards"]
        if row["shard_id"] != dataset.SMOKE_SHARD_ID
    ]
    if len(cloud_shards) != CLOUD_SHARD_COUNT:
        raise RuntimeError("post-smoke cloud shard grid changed")
    source_identity = {
        "dataset_plan_sha256": dataset.canonical_sha256(plan),
        "portable_authorization_sha256": portable_value[
            "authorization_sha256"
        ],
        "content_sha256": canonical_sha256(content),
    }
    execution_identity = canonical_sha256(
        {
            "run_name": run_name,
            "bucket": bucket,
            "source_identity": source_identity,
        }
    )
    jobs = []
    for ordinal, descriptor in enumerate(cloud_shards):
        shard_id = str(descriptor["shard_id"])
        jobs.append(
            {
                "shard_id": shard_id,
                "ordinal": ordinal,
                "wave_index": ordinal // 8,
                "split": descriptor["split"],
                "paired_hand_count": SHARD_PAIR_COUNT,
                "global_pair_indices": list(
                    descriptor["global_pair_indices"]
                ),
                "attempts": [
                    _attempt(
                        execution_identity_sha256=execution_identity,
                        shard_id=shard_id,
                        attempt_id=attempt_id,
                    )
                    for attempt_id in ATTEMPT_IDS
                ],
            }
        )
    core = {
        "schema": TRANSPORT_PLAN_SCHEMA,
        "status": "qualified_ready_cloud_not_started",
        "run_name": run_name,
        "execution_identity_sha256": execution_identity,
        "project": PROJECT,
        "region": REGION,
        "zone": ZONE,
        "bucket": bucket,
        "machine_contract": {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": VCPUS_PER_VM,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "processes_per_vm": 1,
            "rayon_threads": 16,
            "provisioning_model": "SPOT",
            "small_shard_pair_count": SHARD_PAIR_COUNT,
        },
        "source_paths": {
            "smoke_shard_directory": str(Path(smoke_shard_directory).resolve())
        },
        "source_identity": source_identity,
        "content_sources": content,
        "smoke_archive_inventory_sha256": canonical_sha256(smoke_inventory),
        "runtime_archive_inventory_sha256": canonical_sha256(
            runtime_inventory
        ),
        "wheelhouse_inventory_sha256": canonical_sha256(
            wheelhouse_inventory
        ),
        "total_dataset_shard_count": TOTAL_SHARD_COUNT,
        "precompleted_smoke_shard_id": dataset.SMOKE_SHARD_ID,
        "cloud_shard_count": len(jobs),
        "wave_count": WAVE_COUNT,
        "waves": _wave_rows(cloud_shards),
        "jobs": jobs,
        "checkpoint_contract": {
            "pair_granularity": 1,
            "maximum_sequence": SHARD_PAIR_COUNT,
            "create_only_objects": True,
            "heartbeat_create_only_sequence": True,
            "checkpoint_published_after_files": True,
            "final_checkpoint_contains_shard_done": True,
            "receiver_replays_every_file": True,
        },
        "retry_contract": {
            "maximum_attempts_per_shard": MAX_ATTEMPTS_PER_SHARD,
            "attempt_ids": list(ATTEMPT_IDS),
            "earliest_incomplete_wave_only": True,
            "checkpoint_required_before_retry_when_available": True,
        },
        "cleanup_contract": {
            "exact_owned_instance_delete_only": True,
            "wildcard_delete_allowed": False,
            "vm_and_boot_disk_absence_required": True,
            "worker_iam_removed_before_receive": True,
            "gcs_evidence_deleted": False,
        },
        "quality_and_smoke_source_replayed": True,
        "cloud_launch_authorized": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    return {**core, "plan_sha256": canonical_sha256(core)}


def validate_transport_plan(
    value: Mapping[str, Any], *, replay_sources: bool = True
) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    if plan.get("plan_sha256") != _self_digest(plan, "plan_sha256"):
        raise ValueError("dataset transport plan digest changed")
    if (
        plan.get("schema") != TRANSPORT_PLAN_SCHEMA
        or plan.get("status") != "qualified_ready_cloud_not_started"
        or plan.get("project") != PROJECT
        or plan.get("region") != REGION
        or plan.get("zone") != ZONE
        or plan.get("machine_contract")
        != {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": VCPUS_PER_VM,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "processes_per_vm": 1,
            "rayon_threads": 16,
            "provisioning_model": "SPOT",
            "small_shard_pair_count": SHARD_PAIR_COUNT,
        }
        or plan.get("total_dataset_shard_count") != TOTAL_SHARD_COUNT
        or plan.get("precompleted_smoke_shard_id") != dataset.SMOKE_SHARD_ID
        or plan.get("cloud_shard_count") != CLOUD_SHARD_COUNT
        or plan.get("wave_count") != WAVE_COUNT
        or plan.get("quality_and_smoke_source_replayed") is not True
        or plan.get("cloud_launch_authorized") is not True
        or plan.get("cloud_execution_started") is not False
        or plan.get("training_eligible") is not False
        or plan.get("promotion_evidence") is not False
        or plan.get("teacher_values_are_realized_match_ev") is not False
        or plan.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset transport plan boundary changed")
    jobs = plan.get("jobs")
    waves = plan.get("waves")
    content = plan.get("content_sources")
    if (
        not isinstance(jobs, list)
        or len(jobs) != CLOUD_SHARD_COUNT
        or not isinstance(waves, list)
        or len(waves) != WAVE_COUNT
        or not isinstance(content, list)
        or [row.get("kind") for row in content] != list(CONTENT_KINDS)
    ):
        raise ValueError("dataset transport plan grid changed")
    ids = [row.get("shard_id") for row in jobs]
    expected_ids = [
        row["shard_id"]
        for row in dataset.build_dataset_plan()["shards"]
        if row["shard_id"] != dataset.SMOKE_SHARD_ID
    ]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError("dataset transport shard order changed")
    instance_ids = [
        attempt["instance_id"]
        for job in jobs
        for attempt in job.get("attempts", [])
    ]
    object_prefixes = [
        attempt["object_prefix"]
        for job in jobs
        for attempt in job.get("attempts", [])
    ]
    if (
        len(instance_ids) != CLOUD_SHARD_COUNT * MAX_ATTEMPTS_PER_SHARD
        or len(set(instance_ids)) != len(instance_ids)
        or len(set(object_prefixes)) != len(object_prefixes)
    ):
        raise ValueError("dataset transport attempt identities collide")
    for ordinal, job in enumerate(jobs):
        expected_attempts = [
            _attempt(
                execution_identity_sha256=plan["execution_identity_sha256"],
                shard_id=job["shard_id"],
                attempt_id=attempt_id,
            )
            for attempt_id in ATTEMPT_IDS
        ]
        if (
            job.get("ordinal") != ordinal
            or job.get("wave_index") != ordinal // 8
            or job.get("paired_hand_count") != SHARD_PAIR_COUNT
            or job.get("attempts") != expected_attempts
        ):
            raise ValueError("dataset transport job binding changed")
    for index, wave in enumerate(waves):
        expected = ids[index * 8 : (index + 1) * 8]
        if (
            wave.get("wave_index") != index
            or wave.get("shard_ids") != expected
            or wave.get("shard_count") != len(expected)
            or wave.get("max_vm_count") != len(expected)
            or len(expected) > MAX_CONCURRENT_VMS
            or wave.get("prerequisite_wave_indices") != list(range(index))
            or wave.get("requires_prior_waves_complete") != (index > 0)
            or wave.get("requires_owned_compute_absent") is not True
            or wave.get("requires_worker_iam_removed_before_receive") is not True
        ):
            raise ValueError("dataset transport wave binding changed")
    if replay_sources:
        by_kind = {row["kind"]: row for row in content}
        for row in content:
            path = _plain_file(row["source_path"], f"{row['kind']} source")
            if (
                path.name != row["filename"]
                or sha256_file(path) != row["sha256"]
                or path.stat().st_size != row["bytes"]
            ):
                raise ValueError("dataset transport content source changed")
        rebuilt = build_transport_plan(
            run_name=plan["run_name"],
            bucket=plan["bucket"],
            dataset_plan_path=by_kind["dataset_plan"]["source_path"],
            fresh_quality_gate_path=by_kind["fresh_quality_gate"][
                "source_path"
            ],
            smoke_gate_path=by_kind["smoke_gate"]["source_path"],
            portable_authorization_path=by_kind["portable_authorization"][
                "source_path"
            ],
            smoke_shard_directory=plan["source_paths"][
                "smoke_shard_directory"
            ],
            smoke_shard_archive_path=by_kind["smoke_shard_archive"][
                "source_path"
            ],
            runtime_archive_path=by_kind["runtime_archive"]["source_path"],
            wheelhouse_archive_path=by_kind["wheelhouse_archive"][
                "source_path"
            ],
            candidate_library_path=by_kind["candidate_library"][
                "source_path"
            ],
        )
        if rebuilt != plan:
            raise ValueError("dataset transport plan differs from source replay")
    return plan


def _job_by_id(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["shard_id"]: row for row in plan["jobs"]}


def _read_lifecycle(path: str | Path) -> dict[str, Any]:
    return _read_canonical(path, "dataset lifecycle receipt")


def _ledger_from_receipts(
    checked: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len({row["receipt_sha256"] for row in receipts}) != len(receipts):
        raise ValueError("dataset lifecycle receipt is duplicated")
    history: dict[str, list[dict[str, Any]]] = {
        job["shard_id"]: [] for job in checked["jobs"]
    }
    for receipt in receipts:
        for row in receipt["attempt_rows"]:
            history[row["shard_id"]].append(deepcopy(row))
    rows = []
    for job in checked["jobs"]:
        shard_id = job["shard_id"]
        attempts = sorted(
            history[shard_id],
            key=lambda row: ATTEMPT_IDS.index(row["attempt_id"]),
        )
        ids = [row["attempt_id"] for row in attempts]
        if ids != list(ATTEMPT_IDS[: len(ids)]):
            raise ValueError("dataset shard attempt order changed")
        if len(attempts) > MAX_ATTEMPTS_PER_SHARD:
            raise ValueError("dataset shard exceeded retry limit")
        if any(
            later["completed_pair_count"] < earlier["completed_pair_count"]
            for earlier, later in zip(attempts, attempts[1:])
        ):
            raise ValueError("dataset shard checkpoint regressed across attempts")
        complete = bool(attempts and attempts[-1]["status"] == "ready")
        if any(row["status"] == "ready" for row in attempts[:-1]):
            raise ValueError("dataset shard retried after completion")
        completed_pair_count = (
            attempts[-1]["completed_pair_count"] if attempts else 0
        )
        rows.append(
            {
                "shard_id": shard_id,
                "wave_index": job["wave_index"],
                "attempts": attempts,
                "attempt_count": len(attempts),
                "completed_pair_count": completed_pair_count,
                "status": (
                    "complete"
                    if complete
                    else (
                        "exhausted"
                        if len(attempts) == MAX_ATTEMPTS_PER_SHARD
                        else "pending"
                    )
                ),
            }
        )
    core = {
        "schema": ATTEMPT_LEDGER_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "lifecycle_receipt_sha256": [
            row["receipt_sha256"] for row in receipts
        ],
        "shards": rows,
        "complete_shard_count": sum(row["status"] == "complete" for row in rows),
        "pending_shard_count": sum(row["status"] == "pending" for row in rows),
        "exhausted_shard_count": sum(
            row["status"] == "exhausted" for row in rows
        ),
        "cloud_execution_started": bool(receipts),
        "current_profile_changed": False,
    }
    return {**core, "ledger_sha256": canonical_sha256(core)}


def build_attempt_ledger(
    plan: Mapping[str, Any],
    *,
    lifecycle_receipt_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    accepted: list[dict[str, Any]] = []
    for path in lifecycle_receipt_paths:
        before = _ledger_from_receipts(checked, accepted)
        resume = build_resume_plan(checked, before)
        if resume["status"] != "wave_ready":
            raise PermissionError(
                "dataset lifecycle receipt exists without an authorized wave"
            )
        request = build_wave_request(checked, before, resume)
        receipt = validate_lifecycle_receipt(
            _read_lifecycle(path),
            plan=checked,
            ledger=before,
            resume=resume,
            wave_request=request,
        )
        accepted.append(receipt)
    return _ledger_from_receipts(checked, accepted)


def validate_attempt_ledger(
    plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    ledger = deepcopy(dict(value))
    if ledger.get("ledger_sha256") != _self_digest(ledger, "ledger_sha256"):
        raise ValueError("dataset attempt ledger digest changed")
    paths = ledger.get("lifecycle_receipt_sha256")
    if (
        ledger.get("schema") != ATTEMPT_LEDGER_SCHEMA
        or ledger.get("plan_sha256") != checked["plan_sha256"]
        or not isinstance(paths, list)
        or ledger.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset attempt ledger boundary changed")
    rows = ledger.get("shards")
    if not isinstance(rows, list) or len(rows) != CLOUD_SHARD_COUNT:
        raise ValueError("dataset attempt ledger shard grid changed")
    if [row.get("shard_id") for row in rows] != [
        row["shard_id"] for row in checked["jobs"]
    ]:
        raise ValueError("dataset attempt ledger order changed")
    if (
        ledger["complete_shard_count"]
        != sum(row.get("status") == "complete" for row in rows)
        or ledger["pending_shard_count"]
        != sum(row.get("status") == "pending" for row in rows)
        or ledger["exhausted_shard_count"]
        != sum(row.get("status") == "exhausted" for row in rows)
    ):
        raise ValueError("dataset attempt ledger totals changed")
    return ledger


def build_resume_plan(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    state = validate_attempt_ledger(checked, ledger)
    exhausted = [
        row["shard_id"]
        for row in state["shards"]
        if row["status"] == "exhausted"
    ]
    incomplete_waves = [
        wave
        for wave in checked["waves"]
        if any(
            next(
                row
                for row in state["shards"]
                if row["shard_id"] == shard_id
            )["status"]
            != "complete"
            for shard_id in wave["shard_ids"]
        )
    ]
    wave = incomplete_waves[0] if incomplete_waves else None
    selected = []
    if wave is not None and not exhausted:
        ledger_by_id = {row["shard_id"]: row for row in state["shards"]}
        jobs = _job_by_id(checked)
        for shard_id in wave["shard_ids"]:
            row = ledger_by_id[shard_id]
            if row["status"] == "complete":
                continue
            attempt = jobs[shard_id]["attempts"][row["attempt_count"]]
            selected.append(
                {
                    "shard_id": shard_id,
                    "attempt_id": attempt["attempt_id"],
                    "instance_id": attempt["instance_id"],
                    "object_prefix": attempt["object_prefix"],
                    "checkpoint_object_format": attempt[
                        "checkpoint_object_format"
                    ],
                    "heartbeat_object_format": attempt[
                        "heartbeat_object_format"
                    ],
                    "file_object_prefix": attempt["file_object_prefix"],
                    "resume_completed_pair_count": row[
                        "completed_pair_count"
                    ],
                    "previous_attempt": (
                        deepcopy(row["attempts"][-1])
                        if row["attempts"]
                        else None
                    ),
                }
            )
    status = (
        "complete"
        if wave is None
        else ("no_go_attempts_exhausted" if exhausted else "wave_ready")
    )
    core = {
        "schema": RESUME_PLAN_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "ledger_sha256": state["ledger_sha256"],
        "status": status,
        "resume_wave_index": wave["wave_index"] if wave is not None else None,
        "selected_attempts": selected,
        "selected_count": len(selected),
        "exhausted_shard_ids": exhausted,
        "earliest_incomplete_wave_only": True,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "resume_sha256": canonical_sha256(core)}


def validate_resume_plan(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_resume_plan(plan, ledger)
    observed = deepcopy(dict(value))
    if observed != expected:
        raise ValueError("dataset resume plan differs from source replay")
    return observed


def build_wave_request(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    state = validate_attempt_ledger(checked, ledger)
    resumed = validate_resume_plan(checked, state, resume)
    if resumed["status"] != "wave_ready" or not resumed["selected_attempts"]:
        raise PermissionError("dataset resume state does not authorize a wave")
    core = {
        "schema": WAVE_REQUEST_SCHEMA,
        "status": "qualified_wave_request_cloud_not_started",
        "plan_sha256": checked["plan_sha256"],
        "ledger_sha256": state["ledger_sha256"],
        "resume_sha256": resumed["resume_sha256"],
        "wave_index": resumed["resume_wave_index"],
        "selected_attempts": deepcopy(resumed["selected_attempts"]),
        "selected_count": resumed["selected_count"],
        "machine_type": MACHINE_TYPE,
        "max_vm_count": MAX_CONCURRENT_VMS,
        "quality_and_smoke_source_replayed": True,
        "cloud_launch_authorized": True,
        "cloud_execution_started": False,
        "current_profile_changed": False,
    }
    return {**core, "request_sha256": canonical_sha256(core)}


def validate_wave_request(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_wave_request(plan, ledger, resume)
    observed = deepcopy(dict(value))
    if observed != expected or len(observed["selected_attempts"]) > 8:
        raise ValueError("dataset wave request differs from source replay")
    return observed


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset remote file path is not a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or not path.parts
    ):
        raise ValueError("dataset remote file path is unsafe")
    return value


def _allowed_shard_file(relative: str) -> bool:
    return bool(
        relative == "DATASET_AUTHORIZATION.json"
        or relative == "SHARD_DONE.json"
        or re.fullmatch(r"pairs/pair_[0-9]{6}\.json", relative)
        or re.fullmatch(r"evidence/pair_[0-9]{6}\.json", relative)
    )


def validate_checkpoint_manifest(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=False)
    manifest = deepcopy(dict(value))
    job = _job_by_id(checked).get(shard_id)
    attempt = (
        next(
            (
                row
                for row in job["attempts"]
                if row["attempt_id"] == attempt_id
            ),
            None,
        )
        if job is not None
        else None
    )
    if attempt is None:
        raise ValueError("dataset checkpoint attempt is unknown")
    declared = manifest.get("checkpoint_sha256")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("dataset checkpoint files are missing")
    normalized = []
    paths: list[str] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "object_name",
            "sha256",
            "bytes",
        }:
            raise ValueError("dataset checkpoint file record changed")
        relative = _safe_relative(row["relative_path"])
        if (
            not _allowed_shard_file(relative)
            or row["object_name"]
            != f"{attempt['file_object_prefix']}{relative}"
            or _SHA.fullmatch(str(row["sha256"])) is None
            or not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] <= 0
        ):
            raise ValueError("dataset checkpoint file record is invalid")
        normalized.append(deepcopy(dict(row)))
        paths.append(relative)
    completed = manifest.get("completed_pair_count")
    complete = manifest.get("complete")
    expected_count = 1 + 2 * int(completed) + (1 if complete else 0)
    if (
        manifest.get("schema") != CHECKPOINT_SCHEMA
        or declared != _self_digest(manifest, "checkpoint_sha256")
        or manifest.get("plan_sha256") != checked["plan_sha256"]
        or manifest.get("shard_id") != shard_id
        or manifest.get("attempt_id") != attempt_id
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or not 0 <= completed <= SHARD_PAIR_COUNT
        or manifest.get("sequence") != completed
        or not isinstance(complete, bool)
        or complete != (completed == SHARD_PAIR_COUNT)
        or len(files) != expected_count
        or len(paths) != len(set(paths))
        or "DATASET_AUTHORIZATION.json" not in paths
        or ("SHARD_DONE.json" in paths) != complete
        or manifest.get("checkpoint_published_after_files") is not True
        or manifest.get("create_only") is not True
        or manifest.get("teacher_values_are_realized_match_ev") is not False
        or manifest.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset checkpoint manifest contract changed")
    return manifest


def validate_heartbeat(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=False)
    heartbeat = deepcopy(dict(value))
    completed = heartbeat.get("completed_pair_count")
    if (
        heartbeat.get("schema") != HEARTBEAT_SCHEMA
        or heartbeat.get("heartbeat_sha256")
        != _self_digest(heartbeat, "heartbeat_sha256")
        or heartbeat.get("plan_sha256") != checked["plan_sha256"]
        or heartbeat.get("shard_id") != shard_id
        or heartbeat.get("attempt_id") != attempt_id
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or not 0 <= completed <= SHARD_PAIR_COUNT
        or not isinstance(heartbeat.get("sequence"), int)
        or isinstance(heartbeat.get("sequence"), bool)
        or not 0 <= heartbeat["sequence"] <= MAX_HEARTBEAT_SEQUENCE
        or heartbeat.get("create_only") is not True
        or heartbeat.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset heartbeat contract changed")
    return heartbeat


def validate_lifecycle_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any] | None = None,
    resume: Mapping[str, Any] | None = None,
    wave_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    receipt = deepcopy(dict(value))
    if receipt.get("receipt_sha256") != _self_digest(
        receipt, "receipt_sha256"
    ):
        raise ValueError("dataset lifecycle receipt digest changed")
    rows = receipt.get("attempt_rows")
    if not isinstance(rows, list) or not rows or len(rows) > 8:
        raise ValueError("dataset lifecycle attempt rows changed")
    jobs = _job_by_id(checked)
    seen: set[str] = set()
    for row in rows:
        shard_id = row.get("shard_id")
        job = jobs.get(shard_id)
        attempt = next(
            (
                item
                for item in job["attempts"]
                if item["attempt_id"] == row.get("attempt_id")
            ),
            None,
        ) if job is not None else None
        checkpoint = row.get("checkpoint")
        heartbeat = row.get("heartbeat")
        if (
            job is None
            or attempt is None
            or shard_id in seen
            or row.get("instance_id") != attempt["instance_id"]
            or row.get("status") not in {"ready", "checkpointed", "failed"}
            or not isinstance(row.get("completed_pair_count"), int)
            or not 0 <= row["completed_pair_count"] <= SHARD_PAIR_COUNT
            or (
                row["status"] == "ready"
                and row["completed_pair_count"] != SHARD_PAIR_COUNT
            )
            or row.get("owned_compute_absent") is not True
            or row.get("worker_iam_removed") is not True
        ):
            raise ValueError("dataset lifecycle attempt row changed")
        seen.add(shard_id)
        if checkpoint is not None:
            if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
                "object",
                "generation",
                "sha256",
                "bytes",
                "manifest",
                "file_objects",
            }:
                raise ValueError("dataset lifecycle checkpoint record changed")
            manifest = validate_checkpoint_manifest(
                checkpoint["manifest"],
                plan=checked,
                shard_id=shard_id,
                attempt_id=row["attempt_id"],
            )
            if (
                checkpoint["object"]
                != attempt["checkpoint_object_format"]
                % row["completed_pair_count"]
                or checkpoint["sha256"] != canonical_sha256(manifest)
                or checkpoint["bytes"] != len(canonical_bytes(manifest))
                or _GENERATION.fullmatch(str(checkpoint["generation"])) is None
                or row["completed_pair_count"]
                != manifest["completed_pair_count"]
                or len(checkpoint["file_objects"]) != len(manifest["files"])
            ):
                raise ValueError("dataset lifecycle checkpoint binding changed")
        elif row["completed_pair_count"] != 0 or row["status"] != "failed":
            raise ValueError("dataset lifecycle lost its checkpoint")
        if heartbeat is not None:
            if not isinstance(heartbeat, Mapping) or set(heartbeat) != {
                "object",
                "generation",
                "sha256",
                "bytes",
                "value",
            }:
                raise ValueError("dataset lifecycle heartbeat record changed")
            checked_heartbeat = validate_heartbeat(
                heartbeat["value"],
                plan=checked,
                shard_id=shard_id,
                attempt_id=row["attempt_id"],
            )
            if (
                heartbeat["object"]
                != attempt["heartbeat_object_format"]
                % checked_heartbeat["sequence"]
                or heartbeat["sha256"]
                != canonical_sha256(checked_heartbeat)
                or heartbeat["bytes"] != len(canonical_bytes(checked_heartbeat))
                or _GENERATION.fullmatch(str(heartbeat["generation"])) is None
            ):
                raise ValueError("dataset lifecycle heartbeat binding changed")
    if any(
        item is None for item in (ledger, resume, wave_request)
    ) and not all(item is None for item in (ledger, resume, wave_request)):
        raise ValueError("dataset lifecycle context must be all present or absent")
    if ledger is not None:
        assert resume is not None and wave_request is not None
        checked_ledger = validate_attempt_ledger(checked, ledger)
        checked_resume = validate_resume_plan(
            checked, checked_ledger, resume
        )
        checked_request = validate_wave_request(
            checked,
            checked_ledger,
            checked_resume,
            wave_request,
        )
        expected = [
            (row["shard_id"], row["attempt_id"], row["instance_id"])
            for row in checked_request["selected_attempts"]
        ]
        observed = [
            (row["shard_id"], row["attempt_id"], row["instance_id"])
            for row in rows
        ]
        if (
            observed != expected
            or receipt.get("ledger_sha256")
            != checked_ledger["ledger_sha256"]
            or receipt.get("resume_sha256")
            != checked_resume["resume_sha256"]
            or receipt.get("request_sha256")
            != checked_request["request_sha256"]
            or receipt.get("wave_index") != checked_request["wave_index"]
            or any(
                row["completed_pair_count"]
                < selected["resume_completed_pair_count"]
                for row, selected in zip(
                    rows,
                    checked_request["selected_attempts"],
                    strict=True,
                )
            )
        ):
            raise ValueError("dataset lifecycle wave authorization changed")
    if (
        receipt.get("schema") != LIFECYCLE_SCHEMA
        or receipt.get("plan_sha256") != checked["plan_sha256"]
        or _SHA.fullmatch(str(receipt.get("ledger_sha256", ""))) is None
        or _SHA.fullmatch(str(receipt.get("resume_sha256", ""))) is None
        or _SHA.fullmatch(str(receipt.get("request_sha256", ""))) is None
        or not isinstance(receipt.get("wave_index"), int)
        or isinstance(receipt.get("wave_index"), bool)
        or not 0 <= receipt["wave_index"] < WAVE_COUNT
        or receipt.get("selected_attempt_count") != len(rows)
        or receipt.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or receipt.get("create_only_gcs") is not True
        or receipt.get("exact_owned_cleanup") is not True
        or receipt.get("owned_vm_disk_absent") is not True
        or receipt.get("worker_iam_removed_before_receive") is not True
        or receipt.get("receiver_handoff_ready") is not True
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_resource_touched") is not False
        or receipt.get("gcs_evidence_deleted") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset lifecycle receipt boundary changed")
    return receipt


def _copy_create_only(source: Path, destination: Path, expected_sha: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("received dataset object is missing or unsafe")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("received dataset object hash changed")
    if destination.exists() or destination.is_symlink():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.read_bytes() != raw
        ):
            raise FileExistsError("local dataset checkpoint conflicts")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def receive_wave(
    *,
    plan: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    object_root: str | Path,
    local_root: str | Path,
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    lifecycle = validate_lifecycle_receipt(
        lifecycle_receipt, plan=checked
    )
    remote = Path(object_root).resolve()
    local = Path(local_root).resolve()
    if not remote.is_dir() or remote.is_symlink():
        raise ValueError("dataset receiver object root is missing or unsafe")
    received = []
    for row in lifecycle["attempt_rows"]:
        checkpoint = row["checkpoint"]
        if checkpoint is None:
            received.append(
                {
                    "shard_id": row["shard_id"],
                    "attempt_id": row["attempt_id"],
                    "status": "failed_without_checkpoint",
                    "completed_pair_count": 0,
                    "file_count": 0,
                    "done_sha256": None,
                }
            )
            continue
        manifest = checkpoint["manifest"]
        shard_directory = local / row["shard_id"]
        for file_record, object_record in zip(
            manifest["files"], checkpoint["file_objects"], strict=True
        ):
            if (
                object_record.get("object") != file_record["object_name"]
                or object_record.get("sha256") != file_record["sha256"]
                or object_record.get("bytes") != file_record["bytes"]
                or _GENERATION.fullmatch(
                    str(object_record.get("generation", ""))
                )
                is None
            ):
                raise ValueError("dataset receive file-object binding changed")
            object_path = remote.joinpath(
                *PurePosixPath(object_record["object"]).parts
            )
            _copy_create_only(
                object_path,
                shard_directory.joinpath(
                    *PurePosixPath(file_record["relative_path"]).parts
                ),
                file_record["sha256"],
            )
        resume = dataset.inspect_shard_resume(
            plan=dataset.build_dataset_plan(),
            shard_id=row["shard_id"],
            shard_directory=shard_directory,
        )
        expected_complete = row["status"] == "ready"
        if (
            resume["completed_pair_count"] != row["completed_pair_count"]
            or resume["already_complete"] != expected_complete
            or (not expected_complete and not resume["safe_to_resume"])
        ):
            raise ValueError("received dataset shard resume state changed")
        done_sha = (
            sha256_file(shard_directory / "SHARD_DONE.json")
            if expected_complete
            else None
        )
        received.append(
            {
                "shard_id": row["shard_id"],
                "attempt_id": row["attempt_id"],
                "status": (
                    "complete_source_replayed"
                    if expected_complete
                    else "partial_safe_to_resume"
                ),
                "completed_pair_count": row["completed_pair_count"],
                "file_count": len(manifest["files"]),
                "done_sha256": done_sha,
            }
        )
    core = {
        "schema": RECEIVE_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
        "rows": received,
        "received_count": len(received),
        "complete_count": sum(
            row["status"] == "complete_source_replayed" for row in received
        ),
        "partial_count": sum(
            row["status"] == "partial_safe_to_resume" for row in received
        ),
        "failed_count": sum(
            row["status"] == "failed_without_checkpoint" for row in received
        ),
        "source_replayed": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def write_transport_plan(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    plan = build_transport_plan(**kwargs)
    target = Path(path)
    _write_once(target, plan)
    stored = _read_canonical(target, "dataset transport plan")
    return validate_transport_plan(stored, replay_sources=True)


__all__ = [
    "ATTEMPT_LEDGER_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CLOUD_SHARD_COUNT",
    "HEARTBEAT_SCHEMA",
    "LIFECYCLE_SCHEMA",
    "MAX_CONCURRENT_VMS",
    "MAX_HEARTBEAT_SEQUENCE",
    "RECEIVE_SCHEMA",
    "RESUME_PLAN_SCHEMA",
    "TRANSPORT_PLAN_SCHEMA",
    "WAVE_COUNT",
    "WAVE_REQUEST_SCHEMA",
    "build_attempt_ledger",
    "build_resume_plan",
    "build_transport_plan",
    "build_wave_request",
    "canonical_bytes",
    "canonical_sha256",
    "receive_wave",
    "sha256_file",
    "validate_attempt_ledger",
    "validate_checkpoint_manifest",
    "validate_heartbeat",
    "validate_lifecycle_receipt",
    "validate_resume_plan",
    "validate_transport_plan",
    "validate_wave_request",
    "write_transport_plan",
]
