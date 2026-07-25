"""Dedicated bounded Spot lifecycle for Candidate02 full100 performance work.

This lifecycle consumes the frozen local full100 plan and schedules exactly
twenty source-isolated jobs: ten Candidate02 shards and ten reference shards,
with ten paired-hand roots in every shard.  It is intentionally separate from
the byte-frozen Step6d tail lifecycle.

Packaging never invokes gcloud.  Authorization is a distinct immutable local
action.  Initial launch is all-or-nothing after quota/collision preflight, and
at most one attempt-1 resume is allowed for the exact validated incomplete job
set.  Receive validates every runner artifact and the complete hash chain but
does not merge, train, promote, or resolve ``current``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_full100_plan as full_plan
from . import hu_m31_t3_step6d_spot_v2 as tail_spot
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as selector


PACKAGE_SCHEMA = "hu_m31_t3_step6d_full100_spot_package_v1"
PACKAGE_READY_SCHEMA = "hu_m31_t3_step6d_full100_package_ready_v1"
AUTHORIZATION_SCHEMA = "hu_m31_t3_step6d_full100_launch_authorization_v1"
LAUNCH_PREFLIGHT_SCHEMA = "hu_m31_t3_step6d_full100_launch_preflight_v1"
LAUNCH_CLAIM_SCHEMA = "hu_m31_t3_step6d_full100_launch_claim_v1"
LAUNCH_RESULT_SCHEMA = "hu_m31_t3_step6d_full100_launch_result_v1"
RESUME_PREFLIGHT_SCHEMA = "hu_m31_t3_step6d_full100_resume_preflight_v1"
RESUME_CLAIM_SCHEMA = "hu_m31_t3_step6d_full100_resume_claim_v1"
RESUME_RESULT_SCHEMA = "hu_m31_t3_step6d_full100_resume_result_v1"
STATUS_SCHEMA = "hu_m31_t3_step6d_full100_cloud_status_v1"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_full100_receive_v1"
HEARTBEAT_SCHEMA = "hu_m31_t3_step6d_full100_heartbeat_v1"
COST_GUARD_SCHEMA = "hu_m31_t3_step6d_full100_cost_guard_v1"

SOURCE_NAME = "ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip"
STARTUP_NAME = "startup_hu_m31_t3_step6d_full100_v1.sh"
MANIFEST_NAME = "manifest.json"
READY_NAME = "PACKAGE_READY.json"
AUTHORIZATION_NAME = "launch_authorization.json"
LAUNCH_CLAIM_NAME = "launch_claim.json"
LAUNCH_RESULT_NAME = "launch_result.json"
RESUME_CLAIM_NAME = "resume_claim.json"
RESUME_RESULT_NAME = "resume_result.json"
PLAN_PACKAGE_PATH = "frozen/full100_plan.json"
ROOT_PACKAGE_DIR = "frozen/full100_roots"

DEFAULT_PROJECT = tail_spot.DEFAULT_PROJECT
DEFAULT_BUCKET = tail_spot.DEFAULT_BUCKET
DEFAULT_REGION = tail_spot.DEFAULT_REGION
DEFAULT_ZONES = tuple(tail_spot.DEFAULT_ZONES)
EXPECTED_MACHINE_TYPE = tail_spot.EXPECTED_MACHINE_TYPE
EXPECTED_IMAGE_NAME = tail_spot.EXPECTED_IMAGE_NAME
EXPECTED_IMAGE_ID = tail_spot.EXPECTED_IMAGE_ID
EXPECTED_IMAGE_SELF_LINK = tail_spot.EXPECTED_IMAGE_SELF_LINK
REFERENCE_PACKAGE_PATH = tail_spot.REFERENCE_PACKAGE_PATH
CANDIDATE_PACKAGE_PATH = tail_spot.CANDIDATE_PACKAGE_PATH
FEATURE_PACKAGE_PATH = tail_spot.FEATURE_PACKAGE_PATH
EXPECTED_REFERENCE_RELATIVE = tail_spot.EXPECTED_REFERENCE_RELATIVE
EXPECTED_FEATURE_RELATIVE = tail_spot.EXPECTED_FEATURE_RELATIVE
EXPECTED_FEATURE_SHA256 = tail_spot.EXPECTED_FEATURE_ENCODER_SHA256

SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
MAX_LOGICAL_JOBS = 20
MAX_CONCURRENT_VMS = 20
SHARDS_PER_ROLE = 10
HANDS_PER_SHARD = 10
VCPUS_PER_VM = 16
PROCESS_COUNT = 1
RAYON_THREADS = 16
HEARTBEAT_INTERVAL_SECONDS = 60
SPOT_PRICE_CEILING_USD_PER_VM_HOUR = 0.50
MAX_RUNTIME_SECONDS_PER_VM = 4200
INTERNAL_WATCHDOG_SECONDS_PER_VM = 3900
MAX_ATTEMPTS_PER_JOB = 2
MAX_CUMULATIVE_VM_JOBS = MAX_LOGICAL_JOBS * MAX_ATTEMPTS_PER_JOB
INITIAL_ESTIMATED_MAX_COMPUTE_USD = (
    MAX_LOGICAL_JOBS
    * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    * MAX_RUNTIME_SECONDS_PER_VM
    / 3600
)
ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD = (
    MAX_CUMULATIVE_VM_JOBS
    * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    * MAX_RUNTIME_SECONDS_PER_VM
    / 3600
)
PHASE_COMPUTE_CAP_USD = 25.0
M31_TOTAL_COMPUTE_CAP_USD = 500.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_PACKAGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "source_name",
        "source_sha256",
        "source_bytes",
        "startup_name",
        "startup_sha256",
        "plan_package_path",
        "plan_sha256",
        "tail_qualification",
        "run_contract",
        "run_contract_digest",
        "accepted_reference",
        "accepted_candidate",
        "feature_encoder",
        "image",
        "allocation",
        "launch_target",
        "cost_guard",
        "schedule",
        "job_manifests",
        "source_entries",
        "source_entry_count",
        "checkpoint",
        "heartbeat",
        "spot_execution_authorized",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "gcloud_invoked",
    }
)
_READY_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "plan_sha256",
        "run_contract_digest",
        "logical_job_count",
        "gcloud_invoked",
        "spot_vm_started",
        "current_profile_changed",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "plan_sha256",
        "tail_summary_sha256",
        "tail_validation_sha256",
        "run_contract_digest",
        "launch_target",
        "cost_guard",
        "authorized_job_ids",
        "logical_job_count",
        "performance_development_only",
        "spot_execution_authorized",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "authorized_unix_seconds",
    }
)
_JOB_RECORD_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "shard_index",
        "work_hand_indices",
        "path",
        "output_prefix",
        "sha256",
        "bytes",
    }
)
_LAUNCH_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "selected_job_ids",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "plan_sha256",
        "run_contract_digest",
        "launch_target",
        "cost_guard_sha256",
        "preflight_sha256",
        "claimed_unix_seconds",
        "crash_reuse_authorized",
    }
)
_RESUME_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "attempt_index",
        "selected_job_ids",
        "initial_launch_claim_sha256",
        "initial_launch_result_sha256",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "plan_sha256",
        "run_contract_digest",
        "launch_target",
        "cost_guard_sha256",
        "preflight_sha256",
        "claimed_unix_seconds",
        "third_attempt_authorized",
    }
)
_RECEIVE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "launch_claim_sha256",
        "launch_result_sha256",
        "resume_claim_sha256",
        "resume_result_sha256",
        "plan_sha256",
        "tail_summary_sha256",
        "tail_validation_sha256",
        "run_contract_digest",
        "launch_target",
        "work_hand_indices",
        "source_roles",
        "logical_job_count",
        "paired_hand_count",
        "root_count",
        "jobs",
        "candidate_done_paths",
        "reference_done_paths",
        "source_isolation_validated",
        "root_pairing_validated",
        "merge_executed",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)
_RECEIVE_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "shard_index",
        "work_hand_indices",
        "done_path",
        "done_sha256",
        "root_artifacts",
        "source_hand_artifacts",
        "run_contract_digest",
    }
)
_RECEIVE_ARTIFACT_KEYS = frozenset({"hand_index", "path", "sha256"})


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys changed "
            f"(missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)})"
        )


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Any, *, raw: bool = False) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable full100 artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value if raw else canonical_bytes(value)
    if not isinstance(payload, bytes):
        raise TypeError("raw full100 artifact must be bytes")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable full100 artifact exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_launch_target() -> dict[str, Any]:
    return {
        "project": DEFAULT_PROJECT,
        "bucket": DEFAULT_BUCKET,
        "region": DEFAULT_REGION,
        "zones": list(DEFAULT_ZONES),
        "self_delete": True,
    }


def build_cost_guard() -> dict[str, Any]:
    if not (
        INITIAL_ESTIMATED_MAX_COMPUTE_USD
        < ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        <= PHASE_COMPUTE_CAP_USD
        < M31_TOTAL_COMPUTE_CAP_USD
    ):
        raise AssertionError("full100 cost caps no longer nest")
    return {
        "schema": COST_GUARD_SCHEMA,
        "currency": "USD",
        "spot_price_ceiling_usd_per_vm_hour": SPOT_PRICE_CEILING_USD_PER_VM_HOUR,
        "max_runtime_seconds_per_vm": MAX_RUNTIME_SECONDS_PER_VM,
        "internal_watchdog_seconds_per_vm": INTERNAL_WATCHDOG_SECONDS_PER_VM,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "max_logical_jobs": MAX_LOGICAL_JOBS,
        "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
        "max_cumulative_vm_jobs": MAX_CUMULATIVE_VM_JOBS,
        "initial_estimated_max_compute_usd": INITIAL_ESTIMATED_MAX_COMPUTE_USD,
        "all_attempts_estimated_max_compute_usd": (
            ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "phase_compute_cap_usd": PHASE_COMPUTE_CAP_USD,
        "m31_total_compute_cap_usd": M31_TOTAL_COMPUTE_CAP_USD,
    }


def validate_cost_guard(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != build_cost_guard():
        raise ValueError("full100 cost guard changed")
    return dict(value)


def authorized_job_ids(plan: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    if plan is None:
        return tuple(
            f"{role}-shard-{index:02d}"
            for role in SOURCE_ROLES
            for index in range(SHARDS_PER_ROLE)
        )
    checked = full_plan.validate_full100_plan(plan)
    values = tuple(str(row["job_id"]) for row in checked["jobs"])
    if values != authorized_job_ids():
        raise ValueError("full100 plan job order changed")
    return values


def _bounded_jobs(values: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(values)
    allowed = authorized_job_ids()
    if (
        not selected
        or len(selected) > MAX_CONCURRENT_VMS
        or len(selected) != len(set(selected))
        or any(value not in allowed for value in selected)
        or tuple(value for value in allowed if value in set(selected)) != selected
    ):
        raise ValueError("full100 jobs must be unique and in frozen plan order")
    return selected


def _job_manifest(plan: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    manifest = runner.build_shard_manifest(
        run_contract=plan["run_contract"],
        source_role=str(record["source_role"]),
        work_hand_indices=list(record["work_hand_indices"]),
    )
    if canonical_sha256(manifest) != record["shard_manifest_sha256"]:
        raise ValueError("full100 plan shard-manifest digest changed")
    return manifest


def _copy_package_sources(
    *,
    repository_root: Path,
    package_root: Path,
    reference_library: Path,
    candidate_library: Path,
    feature_encoder: Path,
    plan_path: Path,
    root_dir: Path,
) -> dict[str, dict[str, Any]]:
    entries = tail_spot._copy_source_tree(
        repository_root=repository_root,
        package_root=package_root,
        reference_library=reference_library,
        candidate_library=candidate_library,
        feature_encoder=feature_encoder,
    )

    def copy(relative: str, source: Path) -> None:
        entries[relative] = tail_spot._copy_file(source, package_root / relative)

    copy(PLAN_PACKAGE_PATH, plan_path)
    for index in runner.CONTRACT_HAND_INDICES:
        copy(
            f"{ROOT_PACKAGE_DIR}/hand_{index:03d}.json",
            root_dir / f"hand_{index:03d}.json",
        )
    return dict(sorted(entries.items()))


def _job_records(plan: Mapping[str, Any], stage: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for plan_record in plan["jobs"]:
        manifest = _job_manifest(plan, plan_record)
        identifier = str(plan_record["job_id"])
        relative = f"jobs/{identifier}.json"
        path = stage / relative
        _write_once(path, manifest)
        records.append(
            {
                "job_id": identifier,
                "source_role": plan_record["source_role"],
                "shard_index": plan_record["shard_index"],
                "work_hand_indices": list(plan_record["work_hand_indices"]),
                "path": relative,
                "output_prefix": f"jobs/{identifier}",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def package_full100(
    *,
    output_dir: str | Path,
    run_name: str,
    candidate_library: str | Path,
    candidate_sha256: str = full_plan.CANDIDATE_LIBRARY_SHA256,
    reference_library: str | Path | None = None,
    reference_sha256: str = full_plan.REFERENCE_LIBRARY_SHA256,
    feature_encoder: str | Path | None = None,
    plan_path: str | Path = (
        full_plan.DEFAULT_ROOT_DIR.parent.parent / "full100_plan_v1.json"
    ),
    root_dir: str | Path = full_plan.DEFAULT_ROOT_DIR,
    tail_summary_path: str | Path = full_plan.DEFAULT_TAIL_MERGE_DIR / "summary.json",
    tail_validation_path: str | Path = (
        full_plan.DEFAULT_TAIL_MERGE_DIR / "validation.json"
    ),
    repository_root: str | Path = _REPO_ROOT,
    startup_script: str | Path | None = None,
) -> dict[str, Any]:
    """Create an immutable local package without invoking gcloud."""

    if _SAFE_RUN.fullmatch(run_name) is None:
        raise ValueError("full100 run name is not a safe bounded identity")
    root = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("full100 package destination is immutable")
    plan_target = Path(plan_path).resolve()
    stored_plan = full_plan.validate_full100_plan(
        _read_canonical(plan_target, "full100 plan")
    )
    rebuilt_plan = full_plan.build_full100_plan(
        root_dir=root_dir,
        tail_summary_path=tail_summary_path,
        tail_validation_path=tail_validation_path,
    )
    if (
        stored_plan != rebuilt_plan
        or sha256_file(plan_target) != full_plan.FULL100_PLAN_SHA256
    ):
        raise ValueError("full100 plan does not match frozen source evidence")
    candidate_path = tail_spot._validate_binary(
        candidate_library, candidate_sha256, "full100 candidate library"
    )
    reference_path = tail_spot._validate_binary(
        reference_library or root / EXPECTED_REFERENCE_RELATIVE,
        reference_sha256,
        "full100 reference library",
    )
    feature_path = tail_spot._validate_binary(
        feature_encoder or root / EXPECTED_FEATURE_RELATIVE,
        EXPECTED_FEATURE_SHA256,
        "full100 feature encoder",
    )
    if (
        candidate_sha256 != full_plan.CANDIDATE_LIBRARY_SHA256
        or reference_sha256 != full_plan.REFERENCE_LIBRARY_SHA256
        or candidate_path == reference_path
    ):
        raise ValueError("full100 accepted binary identity changed")
    startup = Path(startup_script or root / "scripts" / STARTUP_NAME).resolve()
    if not startup.is_file() or startup.is_symlink():
        raise FileNotFoundError(f"full100 startup script is missing: {startup}")
    root_target = Path(root_dir).resolve()

    stage = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if stage.exists():
        raise FileExistsError("stale full100 package staging exists")
    stage.mkdir(parents=True)
    try:
        package_root = stage / "package_src"
        entries = _copy_package_sources(
            repository_root=root,
            package_root=package_root,
            reference_library=reference_path,
            candidate_library=candidate_path,
            feature_encoder=feature_path,
            plan_path=plan_target,
            root_dir=root_target,
        )
        source_path = stage / SOURCE_NAME
        tail_spot._zip_tree(package_root, source_path)
        shutil.copy2(startup, stage / STARTUP_NAME)
        jobs = _job_records(stored_plan, stage)
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": "immutable_full100_package_ready_not_authorized",
            "run_name": run_name,
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source_path),
            "source_bytes": source_path.stat().st_size,
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(stage / STARTUP_NAME),
            "plan_package_path": PLAN_PACKAGE_PATH,
            "plan_sha256": full_plan.FULL100_PLAN_SHA256,
            "tail_qualification": dict(stored_plan["tail_qualification"]),
            "run_contract": dict(stored_plan["run_contract"]),
            "run_contract_digest": stored_plan["run_contract_digest"],
            "accepted_reference": {
                "package_path": REFERENCE_PACKAGE_PATH,
                "sha256": reference_sha256,
            },
            "accepted_candidate": {
                "package_path": CANDIDATE_PACKAGE_PATH,
                "sha256": candidate_sha256,
            },
            "feature_encoder": {
                "package_path": FEATURE_PACKAGE_PATH,
                "sha256": EXPECTED_FEATURE_SHA256,
            },
            "image": {
                "project": "debian-cloud",
                "name": EXPECTED_IMAGE_NAME,
                "id": EXPECTED_IMAGE_ID,
                "self_link": EXPECTED_IMAGE_SELF_LINK,
            },
            "allocation": {
                "machine_type": EXPECTED_MACHINE_TYPE,
                "process_count": PROCESS_COUNT,
                "rayon_threads_per_process": RAYON_THREADS,
                "omp_threads": 1,
                "m3_batch_threads": 1,
            },
            "launch_target": build_launch_target(),
            "cost_guard": build_cost_guard(),
            "schedule": {
                "scope": full_plan.PLAN_SCOPE,
                "contract_hand_indices": list(runner.CONTRACT_HAND_INDICES),
                "source_roles": list(SOURCE_ROLES),
                "shards_per_role": SHARDS_PER_ROLE,
                "hands_per_shard": HANDS_PER_SHARD,
                "logical_job_count": MAX_LOGICAL_JOBS,
                "mapping": "one_source_ten_hand_shard_per_vm",
                "max_concurrent_vms": MAX_CONCURRENT_VMS,
            },
            "job_manifests": jobs,
            "source_entries": entries,
            "source_entry_count": len(entries),
            "checkpoint": {
                "unit": "completed_source_hand",
                "upload_after_each_hand": True,
                "immutable_write_once": True,
                "resume_verifies_all_artifacts": True,
                "done_uploaded_last": True,
            },
            "heartbeat": {
                "schema": HEARTBEAT_SCHEMA,
                "required": True,
                "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                "remote_scope": "progress_only",
            },
            "spot_execution_authorized": False,
            "performance_lock_authorized": False,
            "quality_pilot_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
            "gcloud_invoked": False,
        }
        _write_once(stage / MANIFEST_NAME, manifest)
        ready = {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "immutable_local_full100_package_complete",
            "run_name": run_name,
            "package_manifest_sha256": sha256_file(stage / MANIFEST_NAME),
            "source_sha256": manifest["source_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "plan_sha256": full_plan.FULL100_PLAN_SHA256,
            "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
            "logical_job_count": MAX_LOGICAL_JOBS,
            "gcloud_invoked": False,
            "spot_vm_started": False,
            "current_profile_changed": False,
        }
        _write_once(stage / READY_NAME, ready)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_package(destination)


def _validate_source_archive(
    source_path: Path, entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    tail_spot._validate_zip_entries(source_path, entries)
    with zipfile.ZipFile(source_path) as archive:
        plan_raw = archive.read(PLAN_PACKAGE_PATH)
        plan_value = json.loads(plan_raw.decode("utf-8"))
        if (
            not isinstance(plan_value, dict)
            or plan_raw != canonical_bytes(plan_value)
            or full_plan.validate_full100_plan(plan_value) != plan_value
            or hashlib.sha256(plan_raw).hexdigest() != full_plan.FULL100_PLAN_SHA256
        ):
            raise ValueError("full100 packaged plan changed")
        roots: list[dict[str, Any]] = []
        for index in runner.CONTRACT_HAND_INDICES:
            relative = f"{ROOT_PACKAGE_DIR}/hand_{index:03d}.json"
            raw = archive.read(relative)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or raw != canonical_bytes(value):
                raise ValueError(f"full100 packaged root {index} is not canonical")
            runner._validate_candidate02_root_artifact(
                value, expected_row=runner.candidate02_schedule_row(index)
            )
            roots.append(value)
        digest = selector.canonical_sha256(
            [selector.canonical_sha256(root) for root in roots]
        )
        if digest != selector.ALL100_ROOT_SHA256:
            raise ValueError("full100 packaged root-set digest changed")
        selector.topology_rows(roots)
    return plan_value


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = _read_canonical(target / MANIFEST_NAME, "full100 package manifest")
    ready = _read_canonical(target / READY_NAME, "full100 package ready")
    _exact_keys(manifest, _PACKAGE_KEYS, "full100 package manifest")
    _exact_keys(ready, _READY_KEYS, "full100 package ready")
    if _SAFE_RUN.fullmatch(str(manifest.get("run_name", ""))) is None:
        raise ValueError("full100 package run name changed")
    source = target / SOURCE_NAME
    startup = target / STARTUP_NAME
    if (
        source.is_symlink()
        or startup.is_symlink()
        or not source.is_file()
        or not startup.is_file()
    ):
        raise ValueError("full100 package source/startup is missing or unsafe")
    entries = manifest.get("source_entries")
    if not isinstance(entries, Mapping):
        raise ValueError("full100 source entries are missing")
    packaged_plan = _validate_source_archive(source, entries)
    contract = runner.validate_run_contract(manifest["run_contract"])
    plan_jobs = packaged_plan["jobs"]
    jobs = manifest.get("job_manifests")
    if not isinstance(jobs, list) or len(jobs) != MAX_LOGICAL_JOBS:
        raise ValueError("full100 package requires exactly twenty jobs")
    expected_ids = authorized_job_ids(packaged_plan)
    for raw, plan_record, expected_id in zip(
        jobs, plan_jobs, expected_ids, strict=True
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("full100 job record is not an object")
        _exact_keys(raw, _JOB_RECORD_KEYS, "full100 job record")
        path = target / str(raw["path"])
        if (
            raw.get("job_id") != expected_id
            or raw.get("source_role") != plan_record["source_role"]
            or raw.get("shard_index") != plan_record["shard_index"]
            or raw.get("work_hand_indices") != plan_record["work_hand_indices"]
            or raw.get("path") != f"jobs/{expected_id}.json"
            or raw.get("output_prefix") != f"jobs/{expected_id}"
            or path.is_symlink()
            or not path.is_file()
            or raw.get("sha256") != sha256_file(path)
            or raw.get("bytes") != path.stat().st_size
        ):
            raise ValueError("full100 job record changed")
        expected_manifest = _job_manifest(packaged_plan, plan_record)
        if _read_canonical(path, f"full100 job {expected_id}") != expected_manifest:
            raise ValueError("full100 job manifest changed")

    expected_false = (
        "spot_execution_authorized",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "gcloud_invoked",
    )
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status") != "immutable_full100_package_ready_not_authorized"
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("source_bytes") != source.stat().st_size
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("startup_sha256") != sha256_file(startup)
        or manifest.get("plan_package_path") != PLAN_PACKAGE_PATH
        or manifest.get("plan_sha256") != full_plan.FULL100_PLAN_SHA256
        or manifest.get("tail_qualification") != packaged_plan["tail_qualification"]
        or contract != packaged_plan["run_contract"]
        or runner.contract_variant(contract) != runner.CANDIDATE02_VARIANT
        or manifest.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or canonical_sha256(contract) != full_plan.FULL_RUN_CONTRACT_DIGEST
        or manifest.get("accepted_reference")
        != {
            "package_path": REFERENCE_PACKAGE_PATH,
            "sha256": full_plan.REFERENCE_LIBRARY_SHA256,
        }
        or manifest.get("accepted_candidate")
        != {
            "package_path": CANDIDATE_PACKAGE_PATH,
            "sha256": full_plan.CANDIDATE_LIBRARY_SHA256,
        }
        or manifest.get("feature_encoder")
        != {
            "package_path": FEATURE_PACKAGE_PATH,
            "sha256": EXPECTED_FEATURE_SHA256,
        }
        or manifest.get("image")
        != {
            "project": "debian-cloud",
            "name": EXPECTED_IMAGE_NAME,
            "id": EXPECTED_IMAGE_ID,
            "self_link": EXPECTED_IMAGE_SELF_LINK,
        }
        or manifest.get("allocation")
        != {
            "machine_type": EXPECTED_MACHINE_TYPE,
            "process_count": PROCESS_COUNT,
            "rayon_threads_per_process": RAYON_THREADS,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        }
        or manifest.get("launch_target") != build_launch_target()
        or validate_cost_guard(manifest.get("cost_guard")) != build_cost_guard()
        or manifest.get("schedule")
        != {
            "scope": full_plan.PLAN_SCOPE,
            "contract_hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "source_roles": list(SOURCE_ROLES),
            "shards_per_role": SHARDS_PER_ROLE,
            "hands_per_shard": HANDS_PER_SHARD,
            "logical_job_count": MAX_LOGICAL_JOBS,
            "mapping": "one_source_ten_hand_shard_per_vm",
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
        }
        or manifest.get("source_entry_count") != len(entries)
        or manifest.get("checkpoint")
        != {
            "unit": "completed_source_hand",
            "upload_after_each_hand": True,
            "immutable_write_once": True,
            "resume_verifies_all_artifacts": True,
            "done_uploaded_last": True,
        }
        or manifest.get("heartbeat")
        != {
            "schema": HEARTBEAT_SCHEMA,
            "required": True,
            "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "remote_scope": "progress_only",
        }
        or any(manifest.get(field) is not False for field in expected_false)
    ):
        raise ValueError("full100 package boundary changed")
    expected_ready = {
        "schema": PACKAGE_READY_SCHEMA,
        "status": "immutable_local_full100_package_complete",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": full_plan.FULL100_PLAN_SHA256,
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "logical_job_count": MAX_LOGICAL_JOBS,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
    }
    if ready != expected_ready:
        raise ValueError("full100 package-ready boundary changed")
    return manifest


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    qualification = manifest["tail_qualification"]
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_full100_performance_development_spot_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "tail_summary_sha256": qualification["summary_sha256"],
        "tail_validation_sha256": qualification["validation_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "authorized_job_ids": list(authorized_job_ids()),
        "logical_job_count": MAX_LOGICAL_JOBS,
        "performance_development_only": True,
        "spot_execution_authorized": True,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "authorized_unix_seconds": time.time(),
    }
    _write_once(target / AUTHORIZATION_NAME, authorization)
    validate_launch_authorization(target)
    return authorization


def _valid_timestamp(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def validate_launch_authorization(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    authorization = _read_canonical(
        target / AUTHORIZATION_NAME, "full100 launch authorization"
    )
    _exact_keys(authorization, _AUTHORIZATION_KEYS, "full100 authorization")
    qualification = manifest["tail_qualification"]
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status")
        != "explicit_full100_performance_development_spot_authorization"
        or authorization.get("run_name") != manifest["run_name"]
        or authorization.get("package_manifest_sha256")
        != sha256_file(target / MANIFEST_NAME)
        or authorization.get("source_sha256") != manifest["source_sha256"]
        or authorization.get("startup_sha256") != manifest["startup_sha256"]
        or authorization.get("plan_sha256") != full_plan.FULL100_PLAN_SHA256
        or authorization.get("tail_summary_sha256") != qualification["summary_sha256"]
        or authorization.get("tail_validation_sha256")
        != qualification["validation_sha256"]
        or authorization.get("run_contract_digest")
        != full_plan.FULL_RUN_CONTRACT_DIGEST
        or authorization.get("launch_target") != build_launch_target()
        or validate_cost_guard(authorization.get("cost_guard")) != build_cost_guard()
        or authorization.get("authorized_job_ids") != list(authorized_job_ids())
        or authorization.get("logical_job_count") != MAX_LOGICAL_JOBS
        or authorization.get("performance_development_only") is not True
        or authorization.get("spot_execution_authorized") is not True
        or any(
            authorization.get(field) is not False
            for field in (
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
        or not _valid_timestamp(authorization.get("authorized_unix_seconds"))
    ):
        raise ValueError("full100 launch authorization changed")
    return manifest, authorization


def _run_prefix(manifest: Mapping[str, Any], *, project: str, bucket: str) -> str:
    if (
        manifest.get("launch_target") != build_launch_target()
        or project != DEFAULT_PROJECT
        or bucket != DEFAULT_BUCKET
    ):
        raise ValueError("full100 remote target differs from authorization")
    return f"gs://{bucket}/runs/{manifest['run_name']}/full100"


def _checked_query(command: Sequence[str], *, label: str, timeout: int) -> str:
    result = tail_spot._subprocess_run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to query full100 {label} ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _json_query(command: Sequence[str], *, label: str, timeout: int) -> Any:
    raw = _checked_query(command, label=label, timeout=timeout)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"full100 {label} returned invalid JSON") from exc


def _quota_snapshot(*, project: str, required_vcpus: int) -> dict[str, Any]:
    region = _json_query(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            DEFAULT_REGION,
            "--project",
            project,
            "--format=json",
        ],
        label="regional quotas",
        timeout=120,
    )
    if not isinstance(region, Mapping) or not isinstance(region.get("quotas"), list):
        raise RuntimeError("full100 regional quota payload changed")
    result: dict[str, Any] = {}
    for metric in ("CPUS", "PREEMPTIBLE_CPUS"):
        matches = [
            row
            for row in region["quotas"]
            if isinstance(row, Mapping) and row.get("metric") == metric
        ]
        if len(matches) != 1:
            raise RuntimeError(f"full100 {metric} quota is missing")
        limit = matches[0].get("limit")
        usage = matches[0].get("usage")
        if (
            isinstance(limit, bool)
            or isinstance(usage, bool)
            or not isinstance(limit, (int, float))
            or not isinstance(usage, (int, float))
            or not math.isfinite(float(limit))
            or not math.isfinite(float(usage))
            or limit < 0
            or usage < 0
        ):
            raise RuntimeError(f"full100 {metric} quota is invalid")
        available = limit - usage
        if available < required_vcpus:
            raise RuntimeError(
                f"full100 {metric} quota insufficient: "
                f"need {required_vcpus}, available {available}"
            )
        result[metric] = {
            "limit": limit,
            "usage": usage,
            "available": available,
        }
    return result


def _instance_name(manifest: Mapping[str, Any], identifier: str, attempt: int) -> str:
    if identifier not in authorized_job_ids() or attempt not in (0, 1):
        raise ValueError("full100 instance mapping changed")
    suffix = "" if attempt == 0 else "-a01"
    return (
        f"{manifest['run_name']}-j{authorized_job_ids().index(identifier):02d}{suffix}"
    )


def _job_record_by_id(
    manifest: Mapping[str, Any], identifier: str
) -> Mapping[str, Any]:
    matches = [
        row for row in manifest["job_manifests"] if row.get("job_id") == identifier
    ]
    if len(matches) != 1:
        raise ValueError(f"full100 job record is not unique: {identifier}")
    return matches[0]


def _object_exists(uri: str, *, project: str) -> bool:
    result = tail_spot._subprocess_run(
        [
            "gcloud",
            "storage",
            "objects",
            "describe",
            uri,
            "--project",
            project,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if result.returncode == 0:
        return True
    if re.search(
        r"(?i)not found|does not exist|no urls matched|404",
        f"{result.stdout}\n{result.stderr}",
    ):
        return False
    raise RuntimeError(f"could not prove GCS object state: {uri}")


def _instance_rows(
    *, manifest: Mapping[str, Any], project: str
) -> list[Mapping[str, Any]]:
    values = _json_query(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            f"--filter=name~'^{manifest['run_name']}-j[01][0-9](-a01)?$'",
            "--format=json(name,zone,status)",
        ],
        label="instance identities",
        timeout=120,
    )
    if not isinstance(values, list) or any(
        not isinstance(value, Mapping) for value in values
    ):
        raise RuntimeError("full100 instance-list payload changed")
    return list(values)


def preflight_launch(
    *,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    jobs = _bounded_jobs(selected)
    if jobs != authorized_job_ids():
        raise ValueError("full100 initial launch requires exactly all twenty jobs")
    prefix = _run_prefix(manifest, project=project, bucket=bucket)
    required_vcpus = len(jobs) * VCPUS_PER_VM
    quota = _quota_snapshot(project=project, required_vcpus=required_vcpus)
    expected_names = [_instance_name(manifest, job, 0) for job in jobs]
    existing = {
        str(row.get("name"))
        for row in _instance_rows(manifest=manifest, project=project)
    } & set(expected_names)
    if existing:
        raise FileExistsError(
            "full100 initial instances already exist: " + ",".join(sorted(existing))
        )
    done_uris = [f"{prefix}/results/jobs/{identifier}/DONE.json" for identifier in jobs]
    existing_done = [
        identifier
        for identifier, uri in zip(jobs, done_uris, strict=True)
        if _object_exists(uri, project=project)
    ]
    if existing_done:
        raise FileExistsError(
            "full100 DONE objects already exist: " + ",".join(existing_done)
        )
    return {
        "schema": LAUNCH_PREFLIGHT_SCHEMA,
        "status": "full100_quota_and_collision_checks_passed",
        "checked_unix_seconds": time.time(),
        "region": DEFAULT_REGION,
        "selected_job_ids": list(jobs),
        "selected_instance_names": expected_names,
        "selected_done_uris": done_uris,
        "required_vcpus": required_vcpus,
        "quota": quota,
        "instances_absent": True,
        "done_objects_absent": True,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "selected_estimated_max_compute_usd": (INITIAL_ESTIMATED_MAX_COMPUTE_USD),
        "phase_compute_cap_usd": PHASE_COMPUTE_CAP_USD,
    }


def _validate_launch_preflight(
    value: Any, *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("full100 launch preflight must be an object")
    payload = dict(value)
    jobs = authorized_job_ids()
    expected_names = [_instance_name(manifest, job, 0) for job in jobs]
    prefix = _run_prefix(manifest, project=DEFAULT_PROJECT, bucket=DEFAULT_BUCKET)
    expected_done = [f"{prefix}/results/jobs/{job}/DONE.json" for job in jobs]
    quota = payload.get("quota")
    quota_ok = isinstance(quota, Mapping) and set(quota) == {
        "CPUS",
        "PREEMPTIBLE_CPUS",
    }
    if quota_ok:
        for metric in quota:
            record = quota[metric]
            quota_ok = (
                isinstance(record, Mapping)
                and set(record) == {"limit", "usage", "available"}
                and record["available"] == record["limit"] - record["usage"]
                and record["available"] >= MAX_LOGICAL_JOBS * VCPUS_PER_VM
            )
            if not quota_ok:
                break
    if (
        payload.get("schema") != LAUNCH_PREFLIGHT_SCHEMA
        or payload.get("status") != "full100_quota_and_collision_checks_passed"
        or not _valid_timestamp(payload.get("checked_unix_seconds"))
        or payload.get("region") != DEFAULT_REGION
        or payload.get("selected_job_ids") != list(jobs)
        or payload.get("selected_instance_names") != expected_names
        or payload.get("selected_done_uris") != expected_done
        or payload.get("required_vcpus") != MAX_LOGICAL_JOBS * VCPUS_PER_VM
        or not quota_ok
        or payload.get("instances_absent") is not True
        or payload.get("done_objects_absent") is not True
        or payload.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or payload.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or payload.get("selected_estimated_max_compute_usd")
        != INITIAL_ESTIMATED_MAX_COMPUTE_USD
        or payload.get("phase_compute_cap_usd") != PHASE_COMPUTE_CAP_USD
    ):
        raise ValueError("full100 launch preflight changed")
    return payload


def _publish_package(
    *, target: Path, manifest: Mapping[str, Any], project: str, bucket: str
) -> str:
    prefix = _run_prefix(manifest, project=project, bucket=bucket)
    sources = [
        (target / MANIFEST_NAME, f"{prefix}/{MANIFEST_NAME}"),
        (target / READY_NAME, f"{prefix}/source/{READY_NAME}"),
        (target / SOURCE_NAME, f"{prefix}/source/{SOURCE_NAME}"),
        (target / STARTUP_NAME, f"{prefix}/source/{STARTUP_NAME}"),
        (target / AUTHORIZATION_NAME, f"{prefix}/source/{AUTHORIZATION_NAME}"),
    ]
    sources.extend(
        (target / row["path"], f"{prefix}/source/{row['path']}")
        for row in manifest["job_manifests"]
    )
    for source, uri in sources:
        tail_spot._publish_once(source, uri, project=project)
    return prefix


def _acquire_launch_claim(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    claim = {
        "schema": LAUNCH_CLAIM_SCHEMA,
        "status": ("exclusive_full100_launch_claim_acquired_before_remote_mutation"),
        "run_name": manifest["run_name"],
        "selected_job_ids": list(authorized_job_ids()),
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "launch_authorization_sha256": sha256_file(target / AUTHORIZATION_NAME),
        "plan_sha256": full_plan.FULL100_PLAN_SHA256,
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "launch_target": build_launch_target(),
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "crash_reuse_authorized": False,
    }
    _write_once(target / LAUNCH_CLAIM_NAME, claim)
    return claim


def _create_instance(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    prefix: str,
    identifier: str,
    project: str,
    zone: str,
    attempt_index: int,
    claim_uri: str,
    claim_sha256: str,
) -> dict[str, Any]:
    record = _job_record_by_id(manifest, identifier)
    instance = _instance_name(manifest, identifier, attempt_index)
    initial_claim = target / LAUNCH_CLAIM_NAME
    initial_result = target / LAUNCH_RESULT_NAME
    metadata = ",".join(
        (
            f"PROJECT_ID={project}",
            f"BUCKET={DEFAULT_BUCKET}",
            f"RUN_NAME={manifest['run_name']}",
            f"JOB_ID={identifier}",
            f"INSTANCE_NAME={instance}",
            f"ZONE={zone}",
            f"SOURCE_URI={prefix}/source/{SOURCE_NAME}",
            f"SOURCE_SHA256={manifest['source_sha256']}",
            f"MANIFEST_URI={prefix}/{MANIFEST_NAME}",
            f"MANIFEST_SHA256={sha256_file(target / MANIFEST_NAME)}",
            f"JOB_MANIFEST_URI={prefix}/source/{record['path']}",
            f"JOB_MANIFEST_SHA256={record['sha256']}",
            f"AUTHORIZATION_URI={prefix}/source/{AUTHORIZATION_NAME}",
            f"AUTHORIZATION_SHA256={sha256_file(target / AUTHORIZATION_NAME)}",
            f"ATTEMPT_INDEX={attempt_index}",
            f"ATTEMPT_CLAIM_URI={claim_uri}",
            f"ATTEMPT_CLAIM_SHA256={claim_sha256}",
            "INITIAL_LAUNCH_CLAIM_SHA256="
            + (sha256_file(initial_claim) if initial_claim.is_file() else "none"),
            "INITIAL_LAUNCH_RESULT_SHA256="
            + (sha256_file(initial_result) if initial_result.is_file() else "none"),
            f"RESULT_PREFIX={prefix}/results/jobs/{identifier}",
            f"PROGRESS_PREFIX={prefix}/progress/jobs/{identifier}",
            f"COST_GUARD_SHA256={canonical_sha256(manifest['cost_guard'])}",
            f"MAX_RUNTIME_SECONDS={INTERNAL_WATCHDOG_SECONDS_PER_VM}",
            "SELF_DELETE=1",
        )
    )
    tail_spot._run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance,
            "--project",
            project,
            "--zone",
            zone,
            "--machine-type",
            EXPECTED_MACHINE_TYPE,
            "--provisioning-model=SPOT",
            "--instance-termination-action=DELETE",
            f"--max-run-duration={MAX_RUNTIME_SECONDS_PER_VM}s",
            "--maintenance-policy=TERMINATE",
            "--no-restart-on-failure",
            "--image-project=debian-cloud",
            f"--image={EXPECTED_IMAGE_NAME}",
            "--boot-disk-size=50GB",
            "--boot-disk-type=hyperdisk-balanced",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            f"--metadata={metadata}",
            f"--metadata-from-file=startup-script={target / STARTUP_NAME}",
            "--quiet",
        ],
        timeout=600,
    )
    return {
        "job_id": identifier,
        "source_role": record["source_role"],
        "shard_index": record["shard_index"],
        "work_hand_indices": record["work_hand_indices"],
        "instance": instance,
        "zone": zone,
        "attempt_index": attempt_index,
        "status": "created",
    }


def _cleanup_instances(
    *,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    project: str,
    attempt_index: int,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for identifier in selected:
        ordinal = authorized_job_ids().index(identifier)
        instance = _instance_name(manifest, identifier, attempt_index)
        zone = DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)]
        result = tail_spot._subprocess_run(
            [
                "gcloud",
                "compute",
                "instances",
                "delete",
                instance,
                "--zone",
                zone,
                "--project",
                project,
                "--quiet",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        absent = result.returncode == 0 or re.search(
            r"(?i)not found|does not exist|404",
            f"{result.stdout}\n{result.stderr}",
        )
        outcomes.append(
            {
                "job_id": identifier,
                "attempt_index": attempt_index,
                "instance": instance,
                "zone": zone,
                "delete_returncode": result.returncode,
                "absence_proven": bool(absent),
                "compute_stopped_or_absent": bool(absent),
            }
        )
    return outcomes


def _launch_selected(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    selected: Sequence[str],
    prefix: str,
    project: str,
    attempt_index: int,
    claim_uri: str,
    claim_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    created: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def create(identifier: str) -> dict[str, Any]:
        ordinal = authorized_job_ids().index(identifier)
        return _create_instance(
            target=target,
            manifest=manifest,
            authorization=authorization,
            prefix=prefix,
            identifier=identifier,
            project=project,
            zone=DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)],
            attempt_index=attempt_index,
            claim_uri=claim_uri,
            claim_sha256=claim_sha256,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT_VMS, len(selected))
    ) as pool:
        futures = {
            pool.submit(create, identifier): identifier for identifier in selected
        }
        for future in concurrent.futures.as_completed(futures):
            identifier = futures[future]
            try:
                created.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "job_id": identifier,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    order = {identifier: index for index, identifier in enumerate(selected)}
    created.sort(key=lambda row: order[row["job_id"]])
    failures.sort(key=lambda row: order[row["job_id"]])
    return created, failures


def launch_jobs(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch_authorization(target)
    if (target / LAUNCH_CLAIM_NAME).exists() or (target / LAUNCH_RESULT_NAME).exists():
        raise FileExistsError("full100 initial launch is already claimed")
    selected = authorized_job_ids()
    preflight = preflight_launch(
        manifest=manifest,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    preflight = _validate_launch_preflight(preflight, manifest=manifest)
    claim = _acquire_launch_claim(
        target=target,
        manifest=manifest,
        authorization=authorization,
        preflight=preflight,
    )
    prefix = _publish_package(
        target=target, manifest=manifest, project=project, bucket=bucket
    )
    claim_uri = f"{prefix}/control/{LAUNCH_CLAIM_NAME}"
    tail_spot._publish_once(target / LAUNCH_CLAIM_NAME, claim_uri, project=project)
    created, failures = _launch_selected(
        target=target,
        manifest=manifest,
        authorization=authorization,
        selected=selected,
        prefix=prefix,
        project=project,
        attempt_index=0,
        claim_uri=claim_uri,
        claim_sha256=sha256_file(target / LAUNCH_CLAIM_NAME),
    )
    cleanup = (
        _cleanup_instances(
            manifest=manifest,
            selected=selected,
            project=project,
            attempt_index=0,
        )
        if failures
        else []
    )
    result = {
        "schema": LAUNCH_RESULT_SCHEMA,
        "status": (
            "full100_jobs_created"
            if not failures
            else "full100_create_failed_cleanup_attempted"
        ),
        "run_name": manifest["run_name"],
        "selected_job_ids": list(selected),
        "created": created,
        "failures": failures,
        "cleanup": cleanup,
        "cleanup_compute_stopped_or_absent": (
            True
            if not failures
            else all(row["compute_stopped_or_absent"] for row in cleanup)
        ),
        "logical_job_count": len(created),
        "max_logical_jobs": MAX_LOGICAL_JOBS,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "machine_type": EXPECTED_MACHINE_TYPE,
        "process_count": PROCESS_COUNT,
        "rayon_threads_per_process": RAYON_THREADS,
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "production_fanout_authorized": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    _write_once(target / LAUNCH_RESULT_NAME, result)
    tail_spot._publish_once(
        target / LAUNCH_RESULT_NAME,
        f"{prefix}/control/{LAUNCH_RESULT_NAME}",
        project=project,
    )
    if failures:
        raise RuntimeError(
            "full100 instance creation failed; cleanup attempted and result persisted"
        )
    return result


def validate_launch_chain(
    *, run_dir: str | Path, manifest: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    checked_manifest = validate_package(target) if manifest is None else dict(manifest)
    claim = _read_canonical(target / LAUNCH_CLAIM_NAME, "full100 launch claim")
    result = _read_canonical(target / LAUNCH_RESULT_NAME, "full100 launch result")
    _exact_keys(claim, _LAUNCH_CLAIM_KEYS, "full100 launch claim")
    selected = authorized_job_ids()
    preflight = _validate_launch_preflight(
        result.get("preflight"), manifest=checked_manifest
    )
    expected_created = []
    for identifier in selected:
        record = _job_record_by_id(checked_manifest, identifier)
        ordinal = selected.index(identifier)
        expected_created.append(
            {
                "job_id": identifier,
                "source_role": record["source_role"],
                "shard_index": record["shard_index"],
                "work_hand_indices": record["work_hand_indices"],
                "instance": _instance_name(checked_manifest, identifier, 0),
                "zone": DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)],
                "attempt_index": 0,
                "status": "created",
            }
        )
    if (
        claim.get("schema") != LAUNCH_CLAIM_SCHEMA
        or claim.get("status")
        != "exclusive_full100_launch_claim_acquired_before_remote_mutation"
        or claim.get("run_name") != checked_manifest["run_name"]
        or claim.get("selected_job_ids") != list(selected)
        or claim.get("package_manifest_sha256") != sha256_file(target / MANIFEST_NAME)
        or claim.get("launch_authorization_sha256")
        != sha256_file(target / AUTHORIZATION_NAME)
        or claim.get("plan_sha256") != full_plan.FULL100_PLAN_SHA256
        or claim.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or claim.get("launch_target") != build_launch_target()
        or claim.get("cost_guard_sha256")
        != canonical_sha256(checked_manifest["cost_guard"])
        or claim.get("preflight_sha256") != canonical_sha256(preflight)
        or not _valid_timestamp(claim.get("claimed_unix_seconds"))
        or claim.get("crash_reuse_authorized") is not False
        or result.get("schema") != LAUNCH_RESULT_SCHEMA
        or result.get("status") != "full100_jobs_created"
        or result.get("run_name") != checked_manifest["run_name"]
        or result.get("selected_job_ids") != list(selected)
        or result.get("created") != expected_created
        or result.get("failures") != []
        or result.get("cleanup") != []
        or result.get("cleanup_compute_stopped_or_absent") is not True
        or result.get("logical_job_count") != MAX_LOGICAL_JOBS
        or result.get("max_logical_jobs") != MAX_LOGICAL_JOBS
        or result.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or result.get("machine_type") != EXPECTED_MACHINE_TYPE
        or result.get("process_count") != PROCESS_COUNT
        or result.get("rayon_threads_per_process") != RAYON_THREADS
        or result.get("run_contract_digest") != checked_manifest["run_contract_digest"]
        or result.get("launch_target") != build_launch_target()
        or result.get("cost_guard") != build_cost_guard()
        or result.get("preflight") != preflight
        or result.get("launch_claim_sha256") != sha256_file(target / LAUNCH_CLAIM_NAME)
        or any(
            result.get(field) is not False
            for field in (
                "production_fanout_authorized",
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("full100 launch hash chain changed")
    return claim, result


def _expected_received_relatives(
    source_role: str, work_hand_indices: Sequence[int]
) -> tuple[str, ...]:
    values = [
        "DONE.json",
        "run_contract.json",
        "shard_manifest.json",
    ]
    values.extend(f"roots/hand_{index:03d}.json" for index in work_hand_indices)
    values.extend(
        f"hands/{source_role}/hand_{index:03d}.json" for index in work_hand_indices
    )
    return tuple(sorted(values))


def _validate_received_job(
    *, job_dir: Path, record: Mapping[str, Any], package_root: Path
) -> dict[str, Any]:
    identifier = str(record["job_id"])
    job_manifest_path = package_root / str(record["path"])
    shard_manifest = runner.validate_shard_manifest(
        _read_canonical(job_manifest_path, f"full100 packaged job {identifier}")
    )
    role = str(shard_manifest["source_role"])
    work = list(shard_manifest["work_hand_indices"])
    actual = tuple(
        sorted(
            path.relative_to(job_dir).as_posix()
            for path in job_dir.rglob("*")
            if path.is_file()
        )
    )
    if actual != _expected_received_relatives(role, work):
        raise ValueError(f"full100 received file set changed: {identifier}")
    if (
        _read_canonical(job_dir / "run_contract.json", "received run contract")
        != shard_manifest["run_contract"]
        or _read_canonical(job_dir / "shard_manifest.json", "received shard manifest")
        != shard_manifest
    ):
        raise ValueError(f"full100 received contract chain changed: {identifier}")
    done = runner.validate_completed_output(job_dir)
    if (
        done.get("schema") != runner.CANDIDATE02_DONE_SCHEMA
        or done.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or done.get("source_role") != role
        or done.get("work_hand_indices") != work
        or done.get("completed_hand_indices") != work
        or done.get("artifact_count") != 2 * HANDS_PER_SHARD
        or done.get("native_library_sha256")
        != shard_manifest["run_contract"][f"{role}_library_sha256"]
        or any(
            done.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError(f"full100 received DONE binding changed: {identifier}")
    roots = [
        {
            "hand_index": index,
            "path": f"jobs/{identifier}/roots/hand_{index:03d}.json",
            "sha256": sha256_file(job_dir / "roots" / f"hand_{index:03d}.json"),
        }
        for index in work
    ]
    hands = [
        {
            "hand_index": index,
            "path": f"jobs/{identifier}/hands/{role}/hand_{index:03d}.json",
            "sha256": sha256_file(job_dir / "hands" / role / f"hand_{index:03d}.json"),
        }
        for index in work
    ]
    return {
        "job_id": identifier,
        "source_role": role,
        "shard_index": record["shard_index"],
        "work_hand_indices": work,
        "done_path": f"jobs/{identifier}/DONE.json",
        "done_sha256": sha256_file(job_dir / "DONE.json"),
        "root_artifacts": roots,
        "source_hand_artifacts": hands,
        "run_contract_digest": shard_manifest["run_contract_digest"],
    }


def preflight_resume(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, _authorization = validate_launch_authorization(target)
    validate_launch_chain(run_dir=target, manifest=manifest)
    if (target / RESUME_CLAIM_NAME).exists() or (target / RESUME_RESULT_NAME).exists():
        raise FileExistsError("full100 attempt-1 resume is already claimed")
    selected_jobs = _bounded_jobs(selected)
    prefix = _run_prefix(manifest, project=project, bucket=bucket)
    instances = _instance_rows(manifest=manifest, project=project)
    relevant = {
        _instance_name(manifest, job, attempt)
        for job in authorized_job_ids()
        for attempt in (0, 1)
    }
    active_initial = [
        str(row.get("name"))
        for row in instances
        if row.get("name") in relevant
        and not str(row.get("name")).endswith("-a01")
        and row.get("status") != "TERMINATED"
    ]
    existing_resume = [
        str(row.get("name"))
        for row in instances
        if row.get("name") in relevant and str(row.get("name")).endswith("-a01")
    ]
    if active_initial or existing_resume:
        raise FileExistsError("full100 resume instance eligibility failed")

    completed: list[str] = []
    incomplete: list[str] = []
    with tempfile.TemporaryDirectory(prefix="full100-resume-") as temporary:
        base = Path(temporary)
        for record in manifest["job_manifests"]:
            identifier = str(record["job_id"])
            done_uri = f"{prefix}/results/jobs/{identifier}/DONE.json"
            if not _object_exists(done_uri, project=project):
                incomplete.append(identifier)
                continue
            job_dir = base / identifier
            job_dir.mkdir()
            tail_spot._run(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    f"{prefix}/results/jobs/{identifier}",
                    str(job_dir),
                    "--project",
                    project,
                ],
                timeout=7200,
            )
            _validate_received_job(job_dir=job_dir, record=record, package_root=target)
            completed.append(identifier)
    if not incomplete:
        raise ValueError("full100 resume has no incomplete jobs")
    if tuple(incomplete) != selected_jobs:
        raise ValueError("full100 resume must select exact validated incomplete set")
    quota = _quota_snapshot(
        project=project, required_vcpus=len(selected_jobs) * VCPUS_PER_VM
    )
    return {
        "schema": RESUME_PREFLIGHT_SCHEMA,
        "status": "full100_attempt1_checks_passed",
        "checked_unix_seconds": time.time(),
        "region": DEFAULT_REGION,
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "selected_initial_instance_names": [
            _instance_name(manifest, job, 0) for job in selected_jobs
        ],
        "selected_resume_instance_names": [
            _instance_name(manifest, job, 1) for job in selected_jobs
        ],
        "required_vcpus": len(selected_jobs) * VCPUS_PER_VM,
        "quota": quota,
        "no_active_initial_instances": True,
        "resume_instance_names_absent": True,
        "validated_completed_job_ids": completed,
        "all_incomplete_jobs_selected": True,
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "max_cumulative_vm_jobs": MAX_CUMULATIVE_VM_JOBS,
        "all_attempts_estimated_max_compute_usd": (
            ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "phase_compute_cap_usd": PHASE_COMPUTE_CAP_USD,
    }


def _validate_resume_preflight(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("full100 resume preflight must be an object")
    payload = dict(value)
    selected_jobs = _bounded_jobs(selected)
    completed = [job for job in authorized_job_ids() if job not in set(selected_jobs)]
    quota = payload.get("quota")
    quota_ok = isinstance(quota, Mapping) and set(quota) == {
        "CPUS",
        "PREEMPTIBLE_CPUS",
    }
    if quota_ok:
        quota_ok = all(
            isinstance(row, Mapping)
            and row.get("available") == row.get("limit") - row.get("usage")
            and row.get("available") >= len(selected_jobs) * VCPUS_PER_VM
            for row in quota.values()
        )
    if (
        payload.get("schema") != RESUME_PREFLIGHT_SCHEMA
        or payload.get("status") != "full100_attempt1_checks_passed"
        or not _valid_timestamp(payload.get("checked_unix_seconds"))
        or payload.get("region") != DEFAULT_REGION
        or payload.get("attempt_index") != 1
        or payload.get("selected_job_ids") != list(selected_jobs)
        or payload.get("selected_initial_instance_names")
        != [_instance_name(manifest, job, 0) for job in selected_jobs]
        or payload.get("selected_resume_instance_names")
        != [_instance_name(manifest, job, 1) for job in selected_jobs]
        or payload.get("required_vcpus") != len(selected_jobs) * VCPUS_PER_VM
        or not quota_ok
        or payload.get("no_active_initial_instances") is not True
        or payload.get("resume_instance_names_absent") is not True
        or payload.get("validated_completed_job_ids") != completed
        or payload.get("all_incomplete_jobs_selected") is not True
        or payload.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or payload.get("max_cumulative_vm_jobs") != MAX_CUMULATIVE_VM_JOBS
        or payload.get("all_attempts_estimated_max_compute_usd")
        != ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        or payload.get("phase_compute_cap_usd") != PHASE_COMPUTE_CAP_USD
    ):
        raise ValueError("full100 resume preflight changed")
    return payload


def resume_jobs(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch_authorization(target)
    if (target / RESUME_CLAIM_NAME).exists() or (target / RESUME_RESULT_NAME).exists():
        raise FileExistsError("full100 attempt-1 resume is already claimed")
    preflight = preflight_resume(
        run_dir=target,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    selected_jobs = _bounded_jobs(selected)
    preflight = _validate_resume_preflight(
        preflight, manifest=manifest, selected=selected_jobs
    )
    claim = {
        "schema": RESUME_CLAIM_SCHEMA,
        "status": ("exclusive_full100_attempt1_claim_acquired_before_remote_mutation"),
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "initial_launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "initial_launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "launch_authorization_sha256": sha256_file(target / AUTHORIZATION_NAME),
        "plan_sha256": full_plan.FULL100_PLAN_SHA256,
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "launch_target": build_launch_target(),
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "third_attempt_authorized": False,
    }
    _write_once(target / RESUME_CLAIM_NAME, claim)
    prefix = _run_prefix(manifest, project=project, bucket=bucket)
    claim_uri = f"{prefix}/resume/{RESUME_CLAIM_NAME}"
    tail_spot._publish_once(target / RESUME_CLAIM_NAME, claim_uri, project=project)
    created, failures = _launch_selected(
        target=target,
        manifest=manifest,
        authorization=authorization,
        selected=selected_jobs,
        prefix=prefix,
        project=project,
        attempt_index=1,
        claim_uri=claim_uri,
        claim_sha256=sha256_file(target / RESUME_CLAIM_NAME),
    )
    cleanup = (
        _cleanup_instances(
            manifest=manifest,
            selected=selected_jobs,
            project=project,
            attempt_index=1,
        )
        if failures
        else []
    )
    result = {
        "schema": RESUME_RESULT_SCHEMA,
        "status": (
            "full100_resume_jobs_created"
            if not failures
            else "full100_resume_create_failed_cleanup_attempted"
        ),
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "created": created,
        "failures": failures,
        "cleanup": cleanup,
        "cleanup_compute_stopped_or_absent": (
            True
            if not failures
            else all(row["compute_stopped_or_absent"] for row in cleanup)
        ),
        "logical_job_count": len(created),
        "max_resume_jobs": MAX_LOGICAL_JOBS,
        "max_cumulative_vm_jobs": MAX_CUMULATIVE_VM_JOBS,
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "resume_claim_sha256": sha256_file(target / RESUME_CLAIM_NAME),
        "initial_launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "initial_launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "third_attempt_authorized": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    _write_once(target / RESUME_RESULT_NAME, result)
    tail_spot._publish_once(
        target / RESUME_RESULT_NAME,
        f"{prefix}/resume/{RESUME_RESULT_NAME}",
        project=project,
    )
    if failures:
        raise RuntimeError(
            "full100 attempt-1 creation failed; cleanup attempted and persisted"
        )
    return result


def validate_resume_chain(
    *, run_dir: str | Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    target = Path(run_dir).resolve()
    claim_path = target / RESUME_CLAIM_NAME
    result_path = target / RESUME_RESULT_NAME
    if not claim_path.exists() and not result_path.exists():
        return None
    if not claim_path.is_file() or not result_path.is_file():
        raise ValueError("full100 resume hash chain is incomplete")
    claim = _read_canonical(claim_path, "full100 resume claim")
    result = _read_canonical(result_path, "full100 resume result")
    _exact_keys(claim, _RESUME_CLAIM_KEYS, "full100 resume claim")
    selected_raw = claim.get("selected_job_ids")
    if not isinstance(selected_raw, list):
        raise ValueError("full100 resume selected jobs changed")
    selected = _bounded_jobs(selected_raw)
    preflight = _validate_resume_preflight(
        result.get("preflight"), manifest=manifest, selected=selected
    )
    expected_created = []
    for identifier in selected:
        record = _job_record_by_id(manifest, identifier)
        ordinal = authorized_job_ids().index(identifier)
        expected_created.append(
            {
                "job_id": identifier,
                "source_role": record["source_role"],
                "shard_index": record["shard_index"],
                "work_hand_indices": record["work_hand_indices"],
                "instance": _instance_name(manifest, identifier, 1),
                "zone": DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)],
                "attempt_index": 1,
                "status": "created",
            }
        )
    if (
        claim.get("schema") != RESUME_CLAIM_SCHEMA
        or claim.get("status")
        != "exclusive_full100_attempt1_claim_acquired_before_remote_mutation"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("attempt_index") != 1
        or claim.get("selected_job_ids") != list(selected)
        or claim.get("initial_launch_claim_sha256")
        != sha256_file(target / LAUNCH_CLAIM_NAME)
        or claim.get("initial_launch_result_sha256")
        != sha256_file(target / LAUNCH_RESULT_NAME)
        or claim.get("package_manifest_sha256") != sha256_file(target / MANIFEST_NAME)
        or claim.get("launch_authorization_sha256")
        != sha256_file(target / AUTHORIZATION_NAME)
        or claim.get("plan_sha256") != full_plan.FULL100_PLAN_SHA256
        or claim.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or claim.get("launch_target") != build_launch_target()
        or claim.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or claim.get("preflight_sha256") != canonical_sha256(preflight)
        or not _valid_timestamp(claim.get("claimed_unix_seconds"))
        or claim.get("third_attempt_authorized") is not False
        or result.get("schema") != RESUME_RESULT_SCHEMA
        or result.get("status") != "full100_resume_jobs_created"
        or result.get("run_name") != manifest["run_name"]
        or result.get("attempt_index") != 1
        or result.get("selected_job_ids") != list(selected)
        or result.get("created") != expected_created
        or result.get("failures") != []
        or result.get("cleanup") != []
        or result.get("cleanup_compute_stopped_or_absent") is not True
        or result.get("logical_job_count") != len(selected)
        or result.get("max_resume_jobs") != MAX_LOGICAL_JOBS
        or result.get("max_cumulative_vm_jobs") != MAX_CUMULATIVE_VM_JOBS
        or result.get("run_contract_digest") != manifest["run_contract_digest"]
        or result.get("launch_target") != build_launch_target()
        or result.get("cost_guard") != build_cost_guard()
        or result.get("preflight") != preflight
        or result.get("resume_claim_sha256") != sha256_file(claim_path)
        or result.get("initial_launch_claim_sha256")
        != sha256_file(target / LAUNCH_CLAIM_NAME)
        or result.get("initial_launch_result_sha256")
        != sha256_file(target / LAUNCH_RESULT_NAME)
        or result.get("third_attempt_authorized") is not False
        or any(
            result.get(field) is not False
            for field in (
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("full100 resume hash chain changed")
    return claim, result


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    manifest, _authorization = validate_launch_authorization(run_dir)
    prefix = _run_prefix(manifest, project=project, bucket=bucket)
    listing = tail_spot._subprocess_run(
        ["gcloud", "storage", "ls", "--recursive", prefix, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError("failed to inspect full100 GCS prefix")
    objects = listing.stdout.splitlines()
    instances = _instance_rows(manifest=manifest, project=project)
    expected_names = {
        _instance_name(manifest, job, attempt)
        for job in authorized_job_ids()
        for attempt in (0, 1)
    }
    selected_instances = [row for row in instances if row.get("name") in expected_names]
    if len({row.get("name") for row in selected_instances}) != len(selected_instances):
        raise ValueError("full100 instance name duplicated across zones")
    by_name = {row.get("name"): row for row in selected_instances}
    jobs: dict[str, Any] = {}
    for identifier in authorized_job_ids():
        result_fragment = f"/results/jobs/{identifier}/"
        progress_fragment = f"/progress/jobs/{identifier}/"
        initial = by_name.get(_instance_name(manifest, identifier, 0))
        resumed = by_name.get(_instance_name(manifest, identifier, 1))
        jobs[identifier] = {
            "done_present_unvalidated": any(
                uri.endswith(f"/results/jobs/{identifier}/DONE.json") for uri in objects
            ),
            "heartbeat_present": any(
                uri.endswith(f"/progress/jobs/{identifier}/heartbeat.json")
                for uri in objects
            ),
            "progress_object_count": sum(progress_fragment in uri for uri in objects),
            "result_object_count": sum(result_fragment in uri for uri in objects),
            "instance": (
                None
                if initial is None
                else {
                    "name": initial.get("name"),
                    "status": initial.get("status"),
                    "zone": str(initial.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
            "resume_instance": (
                None
                if resumed is None
                else {
                    "name": resumed.get("name"),
                    "status": resumed.get("status"),
                    "zone": str(resumed.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
        }
    done_count = sum(row["done_present_unvalidated"] for row in jobs.values())
    return {
        "schema": STATUS_SCHEMA,
        "status": (
            "all_full100_done_present_receive_to_validate"
            if done_count == MAX_LOGICAL_JOBS
            else "incomplete"
        ),
        "run_name": manifest["run_name"],
        "done_present_count": done_count,
        "all_done_present_unvalidated": done_count == MAX_LOGICAL_JOBS,
        "jobs": jobs,
        "logical_job_count": MAX_LOGICAL_JOBS,
        "run_contract_digest": manifest["run_contract_digest"],
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def _build_receive_receipt(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    resume_present: bool,
) -> dict[str, Any]:
    candidate_done = [
        row["done_path"] for row in jobs if row["source_role"] == "candidate"
    ]
    reference_done = [
        row["done_path"] for row in jobs if row["source_role"] == "reference"
    ]
    return {
        "schema": RECEIVE_SCHEMA,
        "status": "exact_full100_source_shards_received_and_validated",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "launch_authorization_sha256": sha256_file(target / AUTHORIZATION_NAME),
        "launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "resume_claim_sha256": (
            sha256_file(target / RESUME_CLAIM_NAME) if resume_present else None
        ),
        "resume_result_sha256": (
            sha256_file(target / RESUME_RESULT_NAME) if resume_present else None
        ),
        "plan_sha256": full_plan.FULL100_PLAN_SHA256,
        "tail_summary_sha256": full_plan.TAIL_SUMMARY_SHA256,
        "tail_validation_sha256": full_plan.TAIL_VALIDATION_SHA256,
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "launch_target": build_launch_target(),
        "work_hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "source_roles": list(SOURCE_ROLES),
        "logical_job_count": MAX_LOGICAL_JOBS,
        "paired_hand_count": 100,
        "root_count": 200,
        "jobs": list(jobs),
        "candidate_done_paths": candidate_done,
        "reference_done_paths": reference_done,
        "source_isolation_validated": True,
        "root_pairing_validated": True,
        "merge_executed": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }


def validate_receive_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    _exact_keys(payload, _RECEIVE_KEYS, "full100 receive receipt")
    jobs = payload.get("jobs")
    expected_ids = list(authorized_job_ids())
    if not isinstance(jobs, list) or len(jobs) != MAX_LOGICAL_JOBS:
        raise ValueError("full100 receive jobs changed")
    plan = full_plan.build_full100_plan()
    plan_by_id = {row["job_id"]: row for row in plan["jobs"]}
    root_by_role: dict[tuple[str, int], str] = {}
    role_coverage = {role: [] for role in SOURCE_ROLES}
    source_artifact_keys: set[tuple[str, int]] = set()
    for expected_id, raw in zip(expected_ids, jobs, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("full100 receive job is not an object")
        _exact_keys(raw, _RECEIVE_JOB_KEYS, "full100 receive job")
        expected = plan_by_id[expected_id]
        role = expected["source_role"]
        shard_index = expected["shard_index"]
        work = list(expected["work_hand_indices"])
        roots = raw.get("root_artifacts")
        source_hands = raw.get("source_hand_artifacts")
        if (
            raw.get("job_id") != expected_id
            or raw.get("source_role") != role
            or raw.get("shard_index") != shard_index
            or raw.get("work_hand_indices") != work
            or raw.get("done_path") != f"jobs/{expected_id}/DONE.json"
            or not _is_sha256(raw.get("done_sha256"))
            or raw.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
            or not isinstance(roots, list)
            or not isinstance(source_hands, list)
            or len(roots) != HANDS_PER_SHARD
            or len(source_hands) != HANDS_PER_SHARD
        ):
            raise ValueError("full100 receive job mapping changed")
        for artifact_type, records in (
            ("roots", roots),
            ("hands", source_hands),
        ):
            observed_hands: list[int] = []
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("full100 receive artifact is not an object")
                _exact_keys(record, _RECEIVE_ARTIFACT_KEYS, "full100 receive artifact")
                hand_index = record.get("hand_index")
                expected_path = (
                    f"jobs/{expected_id}/roots/hand_{hand_index:03d}.json"
                    if artifact_type == "roots"
                    else (
                        f"jobs/{expected_id}/hands/{role}/"
                        f"hand_{hand_index:03d}.json"
                    )
                )
                if (
                    isinstance(hand_index, bool)
                    or not isinstance(hand_index, int)
                    or hand_index not in work
                    or record.get("path") != expected_path
                    or not _is_sha256(record.get("sha256"))
                ):
                    raise ValueError("full100 receive artifact mapping changed")
                observed_hands.append(hand_index)
                if artifact_type == "roots":
                    key = (role, hand_index)
                    if key in root_by_role:
                        raise ValueError("full100 receive duplicate root record")
                    root_by_role[key] = str(record["sha256"])
                else:
                    key = (role, hand_index)
                    if key in source_artifact_keys:
                        raise ValueError("full100 receive duplicate source-hand record")
                    source_artifact_keys.add(key)
            if observed_hands != work:
                raise ValueError("full100 receive artifact order/coverage changed")
        role_coverage[role].extend(work)
    if (
        any(
            sorted(role_coverage[role]) != list(runner.CONTRACT_HAND_INDICES)
            for role in SOURCE_ROLES
        )
        or len(root_by_role) != 200
        or len(source_artifact_keys) != 200
        or any(
            root_by_role[("candidate", index)] != root_by_role[("reference", index)]
            for index in runner.CONTRACT_HAND_INDICES
        )
    ):
        raise ValueError("full100 receive role/root coverage changed")
    candidate_root_digest = selector.canonical_sha256(
        [root_by_role[("candidate", index)] for index in runner.CONTRACT_HAND_INDICES]
    )
    if candidate_root_digest != selector.ALL100_ROOT_SHA256:
        raise ValueError("full100 receive frozen all100 root digest changed")
    resume_claim = payload.get("resume_claim_sha256")
    resume_result = payload.get("resume_result_sha256")
    resume_ok = (resume_claim is None and resume_result is None) or (
        _is_sha256(resume_claim) and _is_sha256(resume_result)
    )
    if (
        payload.get("schema") != RECEIVE_SCHEMA
        or payload.get("status") != "exact_full100_source_shards_received_and_validated"
        or _SAFE_RUN.fullmatch(str(payload.get("run_name", ""))) is None
        or any(
            not _is_sha256(payload.get(field))
            for field in (
                "package_manifest_sha256",
                "launch_authorization_sha256",
                "launch_claim_sha256",
                "launch_result_sha256",
            )
        )
        or not resume_ok
        or payload.get("plan_sha256") != full_plan.FULL100_PLAN_SHA256
        or payload.get("tail_summary_sha256") != full_plan.TAIL_SUMMARY_SHA256
        or payload.get("tail_validation_sha256") != full_plan.TAIL_VALIDATION_SHA256
        or payload.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or payload.get("launch_target") != build_launch_target()
        or payload.get("work_hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or payload.get("source_roles") != list(SOURCE_ROLES)
        or payload.get("logical_job_count") != MAX_LOGICAL_JOBS
        or payload.get("paired_hand_count") != 100
        or payload.get("root_count") != 200
        or payload.get("candidate_done_paths")
        != [
            f"jobs/{identifier}/DONE.json"
            for identifier in expected_ids[:SHARDS_PER_ROLE]
        ]
        or payload.get("reference_done_paths")
        != [
            f"jobs/{identifier}/DONE.json"
            for identifier in expected_ids[SHARDS_PER_ROLE:]
        ]
        or payload.get("source_isolation_validated") is not True
        or payload.get("root_pairing_validated") is not True
        or payload.get("merge_executed") is not False
        or any(
            payload.get(field) is not False
            for field in (
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("full100 receive receipt changed")
    return payload


def receive_jobs(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, _authorization = validate_launch_authorization(target)
    validate_launch_chain(run_dir=target, manifest=manifest)
    resume_chain = validate_resume_chain(run_dir=target, manifest=manifest)
    prefix = _run_prefix(manifest, project=project, bucket=bucket)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("full100 receive destination is immutable")
    stage = destination.with_name(f".{destination.name}.{os.getpid()}.receiving")
    if stage.exists():
        raise FileExistsError("stale full100 receive staging exists")
    stage.mkdir(parents=True)
    try:
        evidence = (
            MANIFEST_NAME,
            READY_NAME,
            SOURCE_NAME,
            STARTUP_NAME,
            AUTHORIZATION_NAME,
            LAUNCH_CLAIM_NAME,
            LAUNCH_RESULT_NAME,
        )
        for name in evidence:
            source = target / name
            _write_once(stage / name, source.read_bytes(), raw=True)
        if resume_chain is not None:
            for name in (RESUME_CLAIM_NAME, RESUME_RESULT_NAME):
                _write_once(stage / name, (target / name).read_bytes(), raw=True)
        for record in manifest["job_manifests"]:
            _write_once(
                stage / str(record["path"]),
                (target / str(record["path"])).read_bytes(),
                raw=True,
            )

        def download(record: Mapping[str, Any]) -> dict[str, Any]:
            identifier = str(record["job_id"])
            job_dir = stage / "jobs" / identifier
            job_dir.mkdir(parents=True)
            tail_spot._run(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    f"{prefix}/results/jobs/{identifier}",
                    str(job_dir),
                    "--project",
                    project,
                ],
                timeout=7200,
            )
            return _validate_received_job(
                job_dir=job_dir, record=record, package_root=stage
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(download, manifest["job_manifests"]))
        if [row["job_id"] for row in jobs] != list(authorized_job_ids()):
            raise ValueError("full100 received job order changed")
        if any(
            row["run_contract_digest"] != full_plan.FULL_RUN_CONTRACT_DIGEST
            for row in jobs
        ):
            raise ValueError("full100 received contract digest changed")
        roots: dict[tuple[str, int], str] = {}
        for row in jobs:
            for root_record in row["root_artifacts"]:
                roots[(row["source_role"], root_record["hand_index"])] = root_record[
                    "sha256"
                ]
        if len(roots) != 200 or any(
            roots[("candidate", index)] != roots[("reference", index)]
            for index in runner.CONTRACT_HAND_INDICES
        ):
            raise ValueError("full100 candidate/reference root pairing changed")
        receipt = validate_receive_receipt(
            _build_receive_receipt(
                target=target,
                manifest=manifest,
                jobs=jobs,
                resume_present=resume_chain is not None,
            )
        )
        _write_once(stage / "receive_receipt.json", receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_received_directory(
    receive_dir: str | Path,
    expected_run_name: str | None = None,
) -> dict[str, Any]:
    """Independently replay the complete local receive and artifact chain."""

    target = Path(receive_dir).resolve()
    manifest = validate_package(target)
    if expected_run_name is not None and manifest["run_name"] != expected_run_name:
        raise ValueError("full100 received run name differs from expectation")
    _manifest, _authorization = validate_launch_authorization(target)
    validate_launch_chain(run_dir=target, manifest=manifest)
    resume_chain = validate_resume_chain(run_dir=target, manifest=manifest)
    receipt = validate_receive_receipt(
        _read_canonical(target / "receive_receipt.json", "full100 receive receipt")
    )
    if receipt["run_name"] != manifest["run_name"]:
        raise ValueError("full100 receipt/package run name changed")
    resume_expected = resume_chain is not None
    if (receipt["resume_claim_sha256"] is not None) != resume_expected or (
        receipt["resume_result_sha256"] is not None
    ) != resume_expected:
        raise ValueError("full100 receipt resume-chain presence changed")
    jobs: list[dict[str, Any]] = []
    for record in manifest["job_manifests"]:
        identifier = str(record["job_id"])
        jobs.append(
            _validate_received_job(
                job_dir=target / "jobs" / identifier,
                record=record,
                package_root=target,
            )
        )
    rebuilt = validate_receive_receipt(
        _build_receive_receipt(
            target=target,
            manifest=manifest,
            jobs=jobs,
            resume_present=resume_expected,
        )
    )
    if receipt != rebuilt:
        raise ValueError("full100 receive receipt does not match source artifacts")
    candidate_done = [
        str((target / relative).resolve())
        for relative in receipt["candidate_done_paths"]
    ]
    reference_done = [
        str((target / relative).resolve())
        for relative in receipt["reference_done_paths"]
    ]
    if (
        len(candidate_done) != SHARDS_PER_ROLE
        or len(reference_done) != SHARDS_PER_ROLE
        or any(not Path(path).is_file() for path in [*candidate_done, *reference_done])
    ):
        raise ValueError("full100 receive DONE path coverage changed")
    return {
        "schema": "hu_m31_t3_step6d_full100_received_directory_validation_v1",
        "status": "exact_full100_receive_directory_revalidated",
        "run_name": manifest["run_name"],
        "receive_receipt_sha256": sha256_file(target / "receive_receipt.json"),
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "candidate_done_paths": candidate_done,
        "reference_done_paths": reference_done,
        "paired_hand_count": 100,
        "root_count": 200,
        "source_isolation_validated": True,
        "root_pairing_validated": True,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def _parse_jobs(value: str) -> tuple[str, ...]:
    if value == "all":
        return authorized_job_ids()
    return _bounded_jobs(token.strip() for token in value.split(",") if token.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    package = commands.add_parser("package")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument("--candidate-library", type=Path, required=True)
    package.add_argument(
        "--candidate-sha256", default=full_plan.CANDIDATE_LIBRARY_SHA256
    )
    package.add_argument("--reference-library", type=Path)
    package.add_argument(
        "--reference-sha256", default=full_plan.REFERENCE_LIBRARY_SHA256
    )
    package.add_argument("--feature-encoder", type=Path)
    package.add_argument("--plan-path", type=Path)
    package.add_argument("--root-dir", type=Path, default=full_plan.DEFAULT_ROOT_DIR)
    package.add_argument("--tail-summary", type=Path)
    package.add_argument("--tail-validation", type=Path)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--startup-script", type=Path)

    for name in ("validate-package", "authorize", "validate-authorization"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--project", default=DEFAULT_PROJECT)
    launch.add_argument("--bucket", default=DEFAULT_BUCKET)
    resume = commands.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--jobs", required=True)
    resume.add_argument("--project", default=DEFAULT_PROJECT)
    resume.add_argument("--bucket", default=DEFAULT_BUCKET)
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--project", default=DEFAULT_PROJECT)
    status.add_argument("--bucket", default=DEFAULT_BUCKET)
    receive = commands.add_parser("receive")
    receive.add_argument("--run-dir", type=Path, required=True)
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--project", default=DEFAULT_PROJECT)
    receive.add_argument("--bucket", default=DEFAULT_BUCKET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        kwargs: dict[str, Any] = {
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "candidate_library": args.candidate_library,
            "candidate_sha256": args.candidate_sha256,
            "reference_library": args.reference_library,
            "reference_sha256": args.reference_sha256,
            "feature_encoder": args.feature_encoder,
            "root_dir": args.root_dir,
            "repository_root": args.repository_root,
            "startup_script": args.startup_script,
        }
        if args.plan_path is not None:
            kwargs["plan_path"] = args.plan_path
        if args.tail_summary is not None:
            kwargs["tail_summary_path"] = args.tail_summary
        if args.tail_validation is not None:
            kwargs["tail_validation_path"] = args.tail_validation
        result: Any = package_full100(**kwargs)
    elif args.command == "validate-package":
        result = validate_package(args.run_dir)
    elif args.command == "authorize":
        result = authorize_launch(args.run_dir)
    elif args.command == "validate-authorization":
        manifest, authorization = validate_launch_authorization(args.run_dir)
        result = {
            "manifest_sha256": canonical_sha256(manifest),
            "authorization": authorization,
        }
    elif args.command == "launch":
        result = launch_jobs(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket
        )
    elif args.command == "resume":
        result = resume_jobs(
            run_dir=args.run_dir,
            selected=_parse_jobs(args.jobs),
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "status":
        result = cloud_status(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket
        )
    else:
        result = receive_jobs(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            project=args.project,
            bucket=args.bucket,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD",
    "MAX_CONCURRENT_VMS",
    "MAX_LOGICAL_JOBS",
    "PHASE_COMPUTE_CAP_USD",
    "authorize_launch",
    "build_cost_guard",
    "cloud_status",
    "launch_jobs",
    "main",
    "package_full100",
    "preflight_launch",
    "preflight_resume",
    "receive_jobs",
    "resume_jobs",
    "validate_cost_guard",
    "validate_launch_authorization",
    "validate_launch_chain",
    "validate_package",
    "validate_receive_receipt",
    "validate_received_directory",
    "validate_resume_chain",
]
