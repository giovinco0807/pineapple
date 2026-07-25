"""Immutable tail-only Spot lifecycle for the Step 6d source-isolated v2 runner.

The lifecycle deliberately stops at packaging unless authorization and launch are
requested as separate operations.  It schedules exactly one source and one hand
per logical job: candidate/reference x the ten frozen tail hands.  Every package,
job, checkpoint, and DONE document is bound to one shared runner contract digest.

Nothing in this module trains a model, changes a named profile, resolves
``current``, reopens Step 6c, or authorizes wider performance/quality fanout.
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
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .hu_m31_t3_step6a_spot import (
    DEFAULT_BUCKET,
    DEFAULT_PROJECT,
    EXPECTED_IMAGE_ID,
    EXPECTED_IMAGE_NAME,
    EXPECTED_IMAGE_SELF_LINK,
    EXPECTED_MACHINE_TYPE,
    _MODEL_PATHS,
    _copy_file,
    _publish_once,
    _run,
    _subprocess_run,
    _zip_tree,
    canonical_bytes,
    sha256_file,
)
from . import run_hu_m31_t3_step6d_performance_v2 as _runner_v2
from .run_hu_m31_t3_step6d_performance import REFERENCE_NATIVE_LIBRARY_SHA256


SPOT_PACKAGE_SCHEMA = "hu_m31_t3_step6d_spot_package_v2"
PACKAGE_READY_SCHEMA = "hu_m31_t3_step6d_spot_package_ready_v2"
LAUNCH_AUTHORIZATION_SCHEMA = "hu_m31_t3_step6d_spot_launch_authorization_v2"
LAUNCH_CLAIM_SCHEMA = "hu_m31_t3_step6d_spot_launch_claim_v1"
LAUNCH_RESULT_SCHEMA = "hu_m31_t3_step6d_spot_launch_v2"
LAUNCH_PREFLIGHT_SCHEMA = "hu_m31_t3_step6d_spot_preflight_v2"
RESUME_CLAIM_SCHEMA = "hu_m31_t3_step6d_spot_resume_claim_v1"
RESUME_RESULT_SCHEMA = "hu_m31_t3_step6d_spot_resume_result_v1"
RESUME_PREFLIGHT_SCHEMA = "hu_m31_t3_step6d_spot_resume_preflight_v1"
COST_GUARD_SCHEMA = "hu_m31_t3_step6d_spot_cost_guard_v1"
CLOUD_STATUS_SCHEMA = "hu_m31_t3_step6d_spot_status_v2"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_spot_receive_v2"
HEARTBEAT_SCHEMA = "hu_m31_t3_step6d_spot_heartbeat_v2"
JOB_MANIFEST_SCHEMA = _runner_v2.SHARD_MANIFEST_SCHEMA

SOURCE_NAME = "ofc_regular_hu_m31_t3_step6d_v2_source.zip"
STARTUP_NAME = "startup_hu_m31_t3_step6d_v2.sh"
PACKAGE_MANIFEST_NAME = "manifest.json"
PACKAGE_READY_NAME = "PACKAGE_READY.json"
AUTHORIZATION_NAME = "launch_authorization.json"
LAUNCH_CLAIM_NAME = "launch_claim.json"
LAUNCH_RESULT_NAME = "launch_result.json"
RESUME_CLAIM_NAME = "resume_claim.json"
RESUME_RESULT_NAME = "resume_result.json"

SOURCE_ROLES = _runner_v2.SOURCE_ROLES
TAIL_HAND_INDICES = _runner_v2.TAIL_HAND_INDICES
CONTRACT_HAND_INDICES = _runner_v2.CONTRACT_HAND_INDICES
MAX_LOGICAL_JOBS = 20
# These two aliases describe the legacy Candidate01/Candidate02-v1 contract
# only.  Runtime lifecycle decisions are derived from each validated run
# contract so a new immutable tail selection cannot be silently interpreted as
# the legacy tail.
AUTHORIZED_JOB_COUNT = len(SOURCE_ROLES) * len(TAIL_HAND_INDICES)
if AUTHORIZED_JOB_COUNT != MAX_LOGICAL_JOBS:
    raise RuntimeError("Step 6d v2 runner tail no longer maps to exactly 20 jobs")
PROCESS_COUNT = _runner_v2.ALLOCATION["workers"]
RAYON_THREADS_PER_PROCESS = _runner_v2.ALLOCATION["rayon_threads_per_worker"]
if (PROCESS_COUNT, RAYON_THREADS_PER_PROCESS) != (1, 16):
    raise RuntimeError("Step 6d v2 Spot allocation must remain exactly one process x16")
if EXPECTED_MACHINE_TYPE != "c4-standard-16":
    raise RuntimeError("Step 6d v2 Spot machine must remain c4-standard-16")
DEFAULT_ZONES = ("asia-northeast1-b", "asia-northeast1-c")
DEFAULT_REGION = "asia-northeast1"
HEARTBEAT_INTERVAL_SECONDS = 60
VCPUS_PER_VM = 16
SPOT_PRICE_CEILING_USD_PER_VM_HOUR = 0.50
MAX_RUNTIME_SECONDS_PER_VM = 3300
INTERNAL_WATCHDOG_SECONDS_PER_VM = 3000
ALL_20_ESTIMATED_MAX_COMPUTE_USD = (
    MAX_LOGICAL_JOBS
    * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    * MAX_RUNTIME_SECONDS_PER_VM
    / 3600
)
HARD_TAIL_COMPUTE_CAP_USD = 20.0
M31_TOTAL_COMPUTE_CAP_USD = 500.0
MAX_ATTEMPTS_PER_JOB = 2
MAX_CUMULATIVE_VM_JOBS = MAX_LOGICAL_JOBS * MAX_ATTEMPTS_PER_JOB
ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD = (
    MAX_CUMULATIVE_VM_JOBS
    * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    * MAX_RUNTIME_SECONDS_PER_VM
    / 3600
)
if VCPUS_PER_VM != RAYON_THREADS_PER_PROCESS:
    raise RuntimeError("Step 6d v2 quota allocation no longer matches one x16 VM")

EXPECTED_REFERENCE_RELATIVE = (
    "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/"
    "package_src/native/release/libofc_hu_m3_engine.so"
)
EXPECTED_FEATURE_RELATIVE = (
    "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/"
    "package_src/target/release/libofc_stage3_feature_encoder.so"
)
EXPECTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
REFERENCE_PACKAGE_PATH = "native/reference/release/libofc_hu_m3_engine.so"
CANDIDATE_PACKAGE_PATH = "native/candidate/release/libofc_hu_m3_engine.so"
FEATURE_PACKAGE_PATH = "target/release/libofc_stage3_feature_encoder.so"
CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH = (
    "configs/hu_joint_policy_m31_t3_candidate02_tail_v2_selection.json"
)
CANDIDATE02_TAIL_V2_SELECTION_BYTES = 1523

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_PATHS = (
    "configs/hu_joint_policy_m31_t3_step6d_contract.json",
    "configs/hu_m43_attempt08_runtime_requirements.txt",
)
_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "src/ofc_regular/__init__.py",
        "src/ofc_regular/run_hu_m31_t3_step6d_performance.py",
        "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py",
        "configs/hu_joint_policy_m31_t3_step6d_contract.json",
        "configs/hu_m43_attempt08_runtime_requirements.txt",
        REFERENCE_PACKAGE_PATH,
        CANDIDATE_PACKAGE_PATH,
        FEATURE_PACKAGE_PATH,
        *_MODEL_PATHS,
    }
)

_JOB_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "run_contract",
        "run_contract_digest",
        "source_role",
        "work_hand_indices",
    }
)
_PACKAGE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "source_name",
        "source_sha256",
        "source_bytes",
        "startup_name",
        "startup_sha256",
        "run_contract",
        "run_contract_digest",
        "accepted_reference",
        "accepted_candidate",
        "feature_encoder",
        "image",
        "allocation",
        "launch_target",
        "cost_guard",
        "tail_schedule",
        "job_manifests",
        "source_entries",
        "source_entry_count",
        "checkpoint",
        "heartbeat",
        "spot_execution_authorized",
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "gcloud_invoked",
    }
)
_PACKAGE_READY_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "source_sha256",
        "startup_sha256",
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
        "run_contract_digest",
        "launch_target",
        "cost_guard",
        "authorized_job_ids",
        "logical_job_count",
        "tail_only",
        "spot_execution_authorized",
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "authorized_unix_seconds",
    }
)
_LAUNCH_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "selected_job_ids",
        "run_contract_digest",
        "launch_target",
        "cost_guard_sha256",
        "preflight_sha256",
        "claimed_unix_seconds",
        "crash_reuse_authorized",
    }
)
_LAUNCH_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "status",
        "checked_unix_seconds",
        "region",
        "selected_job_ids",
        "selected_instance_names",
        "selected_done_uris",
        "required_vcpus",
        "quota",
        "instances_absent",
        "done_objects_absent",
        "cost_guard_sha256",
        "selected_estimated_max_compute_usd",
        "hard_tail_compute_cap_usd",
    }
)
_LAUNCH_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "selected_job_ids",
        "created",
        "logical_job_count",
        "max_logical_jobs",
        "machine_type",
        "process_count",
        "rayon_threads_per_process",
        "run_contract_digest",
        "launch_target",
        "cost_guard",
        "preflight",
        "launch_claim_sha256",
        "failures",
        "cleanup",
        "cleanup_absence_proven",
        "cleanup_compute_stopped_or_absent",
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
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
        "run_contract_digest",
        "launch_target",
        "cost_guard_sha256",
        "preflight_sha256",
        "claimed_unix_seconds",
        "third_attempt_authorized",
    }
)
_RESUME_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "status",
        "checked_unix_seconds",
        "region",
        "attempt_index",
        "selected_job_ids",
        "selected_initial_instance_names",
        "selected_resume_instance_names",
        "selected_done_uris",
        "required_vcpus",
        "quota",
        "no_active_initial_instances",
        "resume_instance_names_absent",
        "done_objects_absent",
        "validated_completed_job_ids",
        "all_incomplete_jobs_selected",
        "cost_guard_sha256",
        "max_cumulative_vm_jobs",
        "all_attempts_estimated_max_compute_usd",
        "hard_tail_compute_cap_usd",
    }
)
_RESUME_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "attempt_index",
        "selected_job_ids",
        "created",
        "logical_job_count",
        "max_resume_jobs",
        "max_cumulative_vm_jobs",
        "run_contract_digest",
        "launch_target",
        "cost_guard",
        "preflight",
        "resume_claim_sha256",
        "initial_launch_claim_sha256",
        "initial_launch_result_sha256",
        "failures",
        "cleanup",
        "cleanup_absence_proven",
        "cleanup_compute_stopped_or_absent",
        "third_attempt_authorized",
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
    }
)
_RECEIVE_RECEIPT_KEYS = frozenset(
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
        "run_contract_digest",
        "launch_target",
        "result_prefix",
        "work_hand_indices",
        "source_roles",
        "logical_job_count",
        "jobs",
        "candidate_done_paths",
        "reference_done_paths",
        "source_isolation_validated",
        "merge_executed",
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)


def _runner_module():
    return _runner_v2


def _contract_variant_choices() -> tuple[str, ...]:
    values = [
        _runner_v2.CANDIDATE01_VARIANT,
        _runner_v2.CANDIDATE02_VARIANT,
    ]
    tail_v2 = getattr(_runner_v2, "CANDIDATE02_TAIL_V2_VARIANT", None)
    if isinstance(tail_v2, str):
        values.append(tail_v2)
    return tuple(values)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} keys changed (missing={missing}, extra={extra})")


def _write_once(path: Path, value: Any, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if raw else canonical_bytes(value)
    if not isinstance(payload, bytes):
        raise TypeError("raw Step 6d v2 artifact must be bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable Step 6d v2 artifact exists: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} must be a canonical JSON object")
    return value


def _validate_linux_x86_64_elf(path: Path, label: str) -> None:
    with path.open("rb") as handle:
        header = handle.read(20)
    if (
        len(header) != 20
        or header[:4] != b"\x7fELF"
        or header[4] != 2
        or header[5] != 1
        or header[16:18] != b"\x03\x00"
        or header[18:20] != b"\x3e\x00"
    ):
        raise ValueError(f"{label} must be a Linux x86_64 shared library")


def _validate_binary(path: str | Path, expected_sha256: str, label: str) -> Path:
    candidate = Path(path).resolve()
    expected = str(expected_sha256).casefold()
    if not _is_sha256(expected):
        raise ValueError(f"{label} SHA-256 must be an explicit lowercase digest")
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"{label} is missing or unsafe: {candidate}")
    if sha256_file(candidate) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    _validate_linux_x86_64_elf(candidate, label)
    return candidate


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def build_launch_target() -> dict[str, Any]:
    """Return the fixed cloud target authorized for this tail-only run."""

    return {
        "project": DEFAULT_PROJECT,
        "bucket": DEFAULT_BUCKET,
        "region": DEFAULT_REGION,
        "zones": list(DEFAULT_ZONES),
        "self_delete": True,
    }


def build_cost_guard() -> dict[str, Any]:
    """Return the immutable Step 6d tail compute-spend boundary."""

    estimated = (
        MAX_LOGICAL_JOBS
        * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        * MAX_RUNTIME_SECONDS_PER_VM
        / 3600
    )
    if estimated != ALL_20_ESTIMATED_MAX_COMPUTE_USD:
        raise AssertionError("Step 6d v2 all-20 cost estimate changed")
    all_attempts = (
        MAX_CUMULATIVE_VM_JOBS
        * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        * MAX_RUNTIME_SECONDS_PER_VM
        / 3600
    )
    if all_attempts != ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD:
        raise AssertionError("Step 6d v2 all-attempt cost estimate changed")
    if not (
        estimated
        <= all_attempts
        <= HARD_TAIL_COMPUTE_CAP_USD
        <= M31_TOTAL_COMPUTE_CAP_USD
    ):
        raise AssertionError("Step 6d v2 cost caps no longer nest inside M3.1")
    return {
        "schema": COST_GUARD_SCHEMA,
        "currency": "USD",
        "spot_price_ceiling_usd_per_vm_hour": (SPOT_PRICE_CEILING_USD_PER_VM_HOUR),
        "max_runtime_seconds_per_vm": MAX_RUNTIME_SECONDS_PER_VM,
        "internal_watchdog_seconds_per_vm": INTERNAL_WATCHDOG_SECONDS_PER_VM,
        "max_logical_jobs": MAX_LOGICAL_JOBS,
        "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
        "max_cumulative_vm_jobs": MAX_CUMULATIVE_VM_JOBS,
        "all_20_estimated_max_compute_usd": (ALL_20_ESTIMATED_MAX_COMPUTE_USD),
        "all_attempts_estimated_max_compute_usd": (
            ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "hard_tail_compute_cap_usd": HARD_TAIL_COMPUTE_CAP_USD,
        "m31_total_compute_cap_usd": M31_TOTAL_COMPUTE_CAP_USD,
    }


def validate_cost_guard(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != build_cost_guard():
        raise ValueError("Step 6d v2 cost guard changed")
    return dict(value)


def _estimated_max_compute_usd(logical_job_count: int) -> float:
    if (
        isinstance(logical_job_count, bool)
        or not isinstance(logical_job_count, int)
        or not 1 <= logical_job_count <= MAX_LOGICAL_JOBS
    ):
        raise ValueError("Step 6d v2 cost estimate requires 1-20 logical jobs")
    estimate = (
        logical_job_count
        * SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        * MAX_RUNTIME_SECONDS_PER_VM
        / 3600
    )
    if estimate > HARD_TAIL_COMPUTE_CAP_USD:
        raise ValueError("Step 6d v2 selected jobs exceed the hard tail cost cap")
    return estimate


def _contract_tail_hand_indices(
    run_contract: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Return the exact tail frozen by a validated runner contract.

    ``None`` intentionally retains the public legacy helper behavior used by
    Candidate01 and Candidate02-v1 artifacts.  All package/cloud/receive paths
    pass their concrete contract.
    """

    if run_contract is None:
        return tuple(TAIL_HAND_INDICES)
    runner = _runner_module()
    contract = runner.validate_run_contract(dict(run_contract))
    raw_tail = contract.get("tail_hand_indices")
    contract_indices = contract.get("contract_hand_indices")
    if (
        not isinstance(raw_tail, list)
        or len(raw_tail) * len(SOURCE_ROLES) != MAX_LOGICAL_JOBS
        or len(set(raw_tail)) != len(raw_tail)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in contract_indices
            for index in raw_tail
        )
    ):
        raise ValueError("Step 6d v2 validated contract has an invalid tail")
    return tuple(raw_tail)


def _logical_job_count(run_contract: Mapping[str, Any] | None = None) -> int:
    count = len(SOURCE_ROLES) * len(_contract_tail_hand_indices(run_contract))
    if count != MAX_LOGICAL_JOBS:
        raise ValueError("Step 6d v2 run contract must map to exactly 20 jobs")
    return count


def job_id(
    source_role: str,
    hand_index: int,
    run_contract: Mapping[str, Any] | None = None,
) -> str:
    if (
        source_role not in SOURCE_ROLES
        or hand_index not in _contract_tail_hand_indices(run_contract)
    ):
        raise ValueError("Step 6d v2 job must select one frozen role/hand pair")
    return f"{source_role}-hand-{hand_index:03d}"


def authorized_job_ids(
    run_contract: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    tail = _contract_tail_hand_indices(run_contract)
    return tuple(
        job_id(role, index, run_contract) for role in SOURCE_ROLES for index in tail
    )


def build_job_manifest(
    *, run_contract: Mapping[str, Any], source_role: str, hand_index: int
) -> dict[str, Any]:
    runner = _runner_module()
    contract = runner.validate_run_contract(dict(run_contract))
    digest = runner.canonical_sha256(contract)
    manifest = {
        "schema": JOB_MANIFEST_SCHEMA,
        "run_contract": contract,
        "run_contract_digest": digest,
        "source_role": source_role,
        "work_hand_indices": [hand_index],
    }
    return validate_job_manifest(manifest)


def validate_job_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Step 6d v2 job manifest must be an object")
    payload = dict(value)
    _require_exact_keys(payload, _JOB_MANIFEST_KEYS, "Step 6d v2 job manifest")
    runner = _runner_module()
    contract = runner.validate_run_contract(payload["run_contract"])
    tail = _contract_tail_hand_indices(contract)
    role = payload.get("source_role")
    indices = payload.get("work_hand_indices")
    if (
        payload.get("schema") != JOB_MANIFEST_SCHEMA
        or payload.get("run_contract_digest") != runner.canonical_sha256(contract)
        or role not in SOURCE_ROLES
        or not isinstance(indices, list)
        or len(indices) != 1
        or isinstance(indices[0], bool)
        or indices[0] not in tail
    ):
        raise ValueError("Step 6d v2 job manifest boundary changed")
    contract_indices = contract.get("contract_hand_indices")
    if (
        contract_indices != list(CONTRACT_HAND_INDICES)
        or indices[0] not in contract_indices
    ):
        raise ValueError("Step 6d v2 job escapes the shared 0..99 contract")
    return payload


def build_job_manifests(run_contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    runner = _runner_module()
    contract = runner.validate_run_contract(dict(run_contract))
    tail = _contract_tail_hand_indices(contract)
    manifests = [
        build_job_manifest(
            run_contract=contract,
            source_role=role,
            hand_index=index,
        )
        for role in SOURCE_ROLES
        for index in tail
    ]
    if len(manifests) != _logical_job_count(contract):
        raise AssertionError("internal Step 6d v2 logical-job count changed")
    return manifests


def _job_record(value: Mapping[str, Any]) -> dict[str, Any]:
    role = str(value["source_role"])
    hand_index = int(value["work_hand_indices"][0])
    identifier = job_id(role, hand_index, value["run_contract"])
    return {
        "job_id": identifier,
        "source_role": role,
        "work_hand_indices": [hand_index],
        "path": f"jobs/{identifier}.json",
        "output_prefix": f"jobs/{identifier}",
    }


def _build_shared_contract(
    *,
    candidate_sha256: str,
    reference_sha256: str,
    contract_variant: str = _runner_v2.CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    runner = _runner_module()
    return runner.build_run_contract(
        candidate_library_sha256=candidate_sha256,
        reference_library_sha256=reference_sha256,
        workers=PROCESS_COUNT,
        rayon_threads_per_worker=RAYON_THREADS_PER_PROCESS,
        variant=contract_variant,
    )


def _copy_source_tree(
    *,
    repository_root: Path,
    package_root: Path,
    reference_library: Path,
    candidate_library: Path,
    feature_encoder: Path,
    selection_manifest: Path | None = None,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}

    def copy(relative: str, source: Path | None = None) -> None:
        origin = source if source is not None else repository_root / relative
        entries[relative] = _copy_file(origin, package_root / relative)

    source_root = repository_root / "src/ofc_regular"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Step 6d v2 source tree is missing: {source_root}")
    for source in sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = source.relative_to(repository_root).as_posix()
        copy(relative, source)
    for relative in (*_CONFIG_PATHS, *_MODEL_PATHS):
        copy(relative)
    if selection_manifest is not None:
        copy(CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH, selection_manifest)
    copy(REFERENCE_PACKAGE_PATH, reference_library)
    copy(CANDIDATE_PACKAGE_PATH, candidate_library)
    copy(FEATURE_PACKAGE_PATH, feature_encoder)
    return dict(sorted(entries.items()))


def package_step6d_v2(
    *,
    output_dir: str | Path,
    run_name: str,
    candidate_library: str | Path,
    candidate_sha256: str,
    reference_library: str | Path | None = None,
    reference_sha256: str = REFERENCE_NATIVE_LIBRARY_SHA256,
    feature_encoder: str | Path | None = None,
    repository_root: str | Path = _REPO_ROOT,
    startup_script: str | Path | None = None,
    contract_variant: str = _runner_v2.CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    """Freeze a local package only; this function never invokes gcloud."""

    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6d v2 run name is not a safe bounded GCP identity")
    root = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6d v2 package destination is immutable")
    reference_path = _validate_binary(
        reference_library or root / EXPECTED_REFERENCE_RELATIVE,
        reference_sha256,
        "Step 6d v2 reference library",
    )
    if reference_sha256.casefold() != REFERENCE_NATIVE_LIBRARY_SHA256:
        raise ValueError("Step 6d v2 reference is not the accepted Step 6c binary")
    candidate_path = _validate_binary(
        candidate_library,
        candidate_sha256,
        "Step 6d v2 candidate library",
    )
    if (
        candidate_path == reference_path
        or candidate_sha256.casefold() == reference_sha256.casefold()
    ):
        raise ValueError("Step 6d v2 candidate and reference must be distinct binaries")
    feature_path = _validate_binary(
        feature_encoder or root / EXPECTED_FEATURE_RELATIVE,
        EXPECTED_FEATURE_ENCODER_SHA256,
        "Step 6d v2 feature encoder",
    )
    startup = Path(startup_script or root / "scripts" / STARTUP_NAME).resolve()
    if not startup.is_file() or startup.is_symlink():
        raise FileNotFoundError(f"Step 6d v2 startup script is missing: {startup}")

    contract = _build_shared_contract(
        candidate_sha256=candidate_sha256.casefold(),
        reference_sha256=reference_sha256.casefold(),
        contract_variant=contract_variant,
    )
    runner = _runner_module()
    contract = runner.validate_run_contract(contract)
    contract_digest = runner.canonical_sha256(contract)
    tail_indices = _contract_tail_hand_indices(contract)
    logical_job_count = _logical_job_count(contract)
    tail_v2_variant = getattr(runner, "CANDIDATE02_TAIL_V2_VARIANT", None)
    is_tail_v2 = runner.contract_variant(contract) == tail_v2_variant
    selection_path: Path | None = None
    if is_tail_v2:
        selection_path = (root / CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH).resolve()
        expected_selection_sha256 = getattr(
            runner,
            "CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256",
            None,
        )
        if (
            not selection_path.is_file()
            or selection_path.is_symlink()
            or not _is_sha256(expected_selection_sha256)
            or sha256_file(selection_path) != expected_selection_sha256
            or contract.get("selection_manifest_sha256") != expected_selection_sha256
        ):
            raise ValueError(
                "Step 6d v2 Candidate02 tail-v2 selection manifest changed"
            )
    if (
        contract.get("candidate_library_sha256") != candidate_sha256.casefold()
        or contract.get("reference_library_sha256") != reference_sha256.casefold()
        or contract.get("contract_hand_indices") != list(CONTRACT_HAND_INDICES)
        or contract.get("allocation")
        != {
            "workers": PROCESS_COUNT,
            "rayon_threads_per_worker": RAYON_THREADS_PER_PROCESS,
        }
    ):
        raise ValueError("Step 6d v2 shared runner contract changed")
    job_values = build_job_manifests(contract)

    stage = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if stage.exists():
        raise FileExistsError(f"stale Step 6d v2 package staging exists: {stage}")
    stage.mkdir(parents=True)
    try:
        package_root = stage / "package_src"
        entries = _copy_source_tree(
            repository_root=root,
            package_root=package_root,
            reference_library=reference_path,
            candidate_library=candidate_path,
            feature_encoder=feature_path,
            selection_manifest=selection_path,
        )
        source_path = stage / SOURCE_NAME
        _zip_tree(package_root, source_path)
        shutil.copy2(startup, stage / STARTUP_NAME)
        job_records: list[dict[str, Any]] = []
        for value in job_values:
            record = _job_record(value)
            path = stage / record["path"]
            _write_once(path, value)
            record.update({"sha256": sha256_file(path), "bytes": path.stat().st_size})
            job_records.append(record)
        manifest = {
            "schema": SPOT_PACKAGE_SCHEMA,
            "status": "immutable_package_ready_not_authorized",
            "run_name": run_name,
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source_path),
            "source_bytes": source_path.stat().st_size,
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(stage / STARTUP_NAME),
            "run_contract": contract,
            "run_contract_digest": contract_digest,
            "accepted_reference": {
                "package_path": REFERENCE_PACKAGE_PATH,
                "sha256": reference_sha256.casefold(),
            },
            "accepted_candidate": {
                "package_path": CANDIDATE_PACKAGE_PATH,
                "sha256": candidate_sha256.casefold(),
            },
            "feature_encoder": {
                "package_path": FEATURE_PACKAGE_PATH,
                "sha256": EXPECTED_FEATURE_ENCODER_SHA256,
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
                "rayon_threads_per_process": RAYON_THREADS_PER_PROCESS,
                "omp_threads": 1,
                "m3_batch_threads": 1,
            },
            "launch_target": build_launch_target(),
            "cost_guard": build_cost_guard(),
            "tail_schedule": {
                "contract_hand_indices": list(CONTRACT_HAND_INDICES),
                "work_hand_indices": list(tail_indices),
                "source_roles": list(SOURCE_ROLES),
                "mapping": "one_source_hand_per_vm",
                "logical_job_count": logical_job_count,
                "max_logical_jobs": MAX_LOGICAL_JOBS,
            },
            "job_manifests": job_records,
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
            "production_fanout_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
            "gcloud_invoked": False,
        }
        _write_once(stage / PACKAGE_MANIFEST_NAME, manifest)
        ready = {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "immutable_local_package_complete",
            "run_name": run_name,
            "package_manifest_sha256": sha256_file(stage / PACKAGE_MANIFEST_NAME),
            "source_sha256": manifest["source_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "run_contract_digest": contract_digest,
            "logical_job_count": logical_job_count,
            "gcloud_invoked": False,
            "spot_vm_started": False,
            "current_profile_changed": False,
        }
        _write_once(stage / PACKAGE_READY_NAME, ready)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_package(destination)


def _validate_zip_entries(
    source_path: Path, entries: Mapping[str, Mapping[str, Any]]
) -> None:
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("Step 6d v2 source entry manifest is absent")
    with zipfile.ZipFile(source_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (
            len(names) != len(set(names))
            or set(names) != set(entries)
            or any(
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
                for name in names
            )
        ):
            raise ValueError("Step 6d v2 source zip file set changed")
        for info in infos:
            expected = entries.get(info.filename)
            if (
                not isinstance(expected, Mapping)
                or set(expected) != {"sha256", "bytes"}
                or not _is_sha256(expected.get("sha256"))
                or isinstance(expected.get("bytes"), bool)
                or not isinstance(expected.get("bytes"), int)
                or expected["bytes"] < 0
                or info.is_dir()
                or ((info.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError(f"unsafe Step 6d v2 source entry: {info.filename}")
            data = archive.read(info)
            if (
                len(data) != expected["bytes"]
                or hashlib.sha256(data).hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Step 6d v2 source entry changed: {info.filename}")


def _expected_job_records(run_contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_job_record(value) for value in build_job_manifests(run_contract)]


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest_path = target / PACKAGE_MANIFEST_NAME
    ready_path = target / PACKAGE_READY_NAME
    source_path = target / SOURCE_NAME
    startup_path = target / STARTUP_NAME
    if any(
        not path.is_file()
        for path in (manifest_path, ready_path, source_path, startup_path)
    ):
        raise ValueError("Step 6d v2 package is incomplete")
    manifest = _load_canonical(manifest_path, "Step 6d v2 package manifest")
    ready = _load_canonical(ready_path, "Step 6d v2 package ready marker")
    _require_exact_keys(manifest, _PACKAGE_MANIFEST_KEYS, "Step 6d v2 package manifest")
    _require_exact_keys(ready, _PACKAGE_READY_KEYS, "Step 6d v2 package ready marker")
    runner = _runner_module()
    contract = runner.validate_run_contract(manifest.get("run_contract"))
    contract_digest = runner.canonical_sha256(contract)
    tail_indices = _contract_tail_hand_indices(contract)
    logical_job_count = _logical_job_count(contract)
    reference = manifest.get("accepted_reference")
    candidate = manifest.get("accepted_candidate")
    feature = manifest.get("feature_encoder")
    image = manifest.get("image")
    allocation = manifest.get("allocation")
    launch_target = manifest.get("launch_target")
    cost_guard = manifest.get("cost_guard")
    tail = manifest.get("tail_schedule")
    checkpoint = manifest.get("checkpoint")
    heartbeat = manifest.get("heartbeat")
    if (
        manifest.get("schema") != SPOT_PACKAGE_SCHEMA
        or manifest.get("status") != "immutable_package_ready_not_authorized"
        or not _SAFE_RUN.fullmatch(str(manifest.get("run_name", "")))
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("run_contract_digest") != contract_digest
        or contract.get("contract_hand_indices") != list(CONTRACT_HAND_INDICES)
        or not isinstance(reference, Mapping)
        or set(reference) != {"package_path", "sha256"}
        or reference.get("package_path") != REFERENCE_PACKAGE_PATH
        or reference.get("sha256") != REFERENCE_NATIVE_LIBRARY_SHA256
        or contract.get("reference_library_sha256") != reference.get("sha256")
        or not isinstance(candidate, Mapping)
        or set(candidate) != {"package_path", "sha256"}
        or candidate.get("package_path") != CANDIDATE_PACKAGE_PATH
        or not _is_sha256(candidate.get("sha256"))
        or candidate.get("sha256") == reference.get("sha256")
        or contract.get("candidate_library_sha256") != candidate.get("sha256")
        or feature
        != {
            "package_path": FEATURE_PACKAGE_PATH,
            "sha256": EXPECTED_FEATURE_ENCODER_SHA256,
        }
        or image
        != {
            "project": "debian-cloud",
            "name": EXPECTED_IMAGE_NAME,
            "id": EXPECTED_IMAGE_ID,
            "self_link": EXPECTED_IMAGE_SELF_LINK,
        }
        or allocation
        != {
            "machine_type": EXPECTED_MACHINE_TYPE,
            "process_count": PROCESS_COUNT,
            "rayon_threads_per_process": RAYON_THREADS_PER_PROCESS,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        }
        or launch_target != build_launch_target()
        or cost_guard != build_cost_guard()
        or contract.get("allocation")
        != {
            "workers": PROCESS_COUNT,
            "rayon_threads_per_worker": RAYON_THREADS_PER_PROCESS,
        }
        or tail
        != {
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "work_hand_indices": list(tail_indices),
            "source_roles": list(SOURCE_ROLES),
            "mapping": "one_source_hand_per_vm",
            "logical_job_count": logical_job_count,
            "max_logical_jobs": MAX_LOGICAL_JOBS,
        }
        or checkpoint
        != {
            "unit": "completed_source_hand",
            "upload_after_each_hand": True,
            "immutable_write_once": True,
            "resume_verifies_all_artifacts": True,
            "done_uploaded_last": True,
        }
        or heartbeat
        != {
            "schema": HEARTBEAT_SCHEMA,
            "required": True,
            "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "remote_scope": "progress_only",
        }
        or manifest.get("spot_execution_authorized") is not False
        or manifest.get("production_fanout_authorized") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("current_profile_changed") is not False
        or manifest.get("named_profile_added") is not False
        or manifest.get("runtime_policy_activated") is not False
        or manifest.get("m31_complete") is not False
        or manifest.get("gcloud_invoked") is not False
    ):
        raise ValueError("Step 6d v2 package semantic boundary changed")
    if (
        manifest.get("source_sha256") != sha256_file(source_path)
        or manifest.get("source_bytes") != source_path.stat().st_size
        or manifest.get("startup_sha256") != sha256_file(startup_path)
    ):
        raise ValueError("Step 6d v2 package bytes changed")
    entries = manifest.get("source_entries")
    is_tail_v2 = runner.contract_variant(contract) == getattr(
        runner, "CANDIDATE02_TAIL_V2_VARIANT", None
    )
    required_source_paths = set(_REQUIRED_SOURCE_PATHS)
    if is_tail_v2:
        required_source_paths.add(CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH)
    if (
        not isinstance(entries, Mapping)
        or manifest.get("source_entry_count") != len(entries)
        or not required_source_paths.issubset(entries)
    ):
        raise ValueError("Step 6d v2 source entry boundary changed")
    if is_tail_v2:
        expected_selection_sha256 = getattr(
            runner,
            "CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256",
            None,
        )
        selection_entry = entries.get(CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH)
        if (
            contract.get("selection_manifest_sha256") != expected_selection_sha256
            or not isinstance(selection_entry, Mapping)
            or selection_entry.get("sha256") != expected_selection_sha256
            or selection_entry.get("bytes") != CANDIDATE02_TAIL_V2_SELECTION_BYTES
        ):
            raise ValueError("Step 6d v2 Candidate02 tail-v2 selection binding changed")
    for binding in (reference, candidate, feature):
        entry = entries.get(binding["package_path"])
        if not isinstance(entry, Mapping) or entry.get("sha256") != binding["sha256"]:
            raise ValueError("Step 6d v2 packaged native binding changed")
    _validate_zip_entries(source_path, entries)

    records = manifest.get("job_manifests")
    if not isinstance(records, list) or len(records) != logical_job_count:
        raise ValueError("Step 6d v2 logical-job count changed")
    expected_base = _expected_job_records(contract)
    validated_ids: list[str] = []
    for record, expected in zip(records, expected_base, strict=True):
        if (
            not isinstance(record, Mapping)
            or set(record) != set(expected) | {"sha256", "bytes"}
            or any(record.get(key) != value for key, value in expected.items())
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
        ):
            raise ValueError("Step 6d v2 job record changed")
        job_path = (target / record["path"]).resolve()
        if not job_path.is_relative_to(target) or not job_path.is_file():
            raise ValueError("Step 6d v2 job path is unsafe or missing")
        value = validate_job_manifest(
            _load_canonical(job_path, f"Step 6d v2 job {record['job_id']}")
        )
        if (
            value["run_contract_digest"] != contract_digest
            or value["run_contract"] != contract
            or sha256_file(job_path) != record["sha256"]
            or job_path.stat().st_size != record["bytes"]
        ):
            raise ValueError("Step 6d v2 job manifest bytes changed")
        validated_ids.append(str(record["job_id"]))
    if tuple(validated_ids) != authorized_job_ids(contract):
        raise ValueError("Step 6d v2 role/hand job ordering changed")

    if ready != {
        "schema": PACKAGE_READY_SCHEMA,
        "status": "immutable_local_package_complete",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "run_contract_digest": contract_digest,
        "logical_job_count": logical_job_count,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
    }:
        raise ValueError("Step 6d v2 package-ready chain changed")
    return manifest


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    contract = manifest["run_contract"]
    allowed = authorized_job_ids(contract)
    authorization = {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "explicit_tail_spot_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / PACKAGE_MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "authorized_job_ids": list(allowed),
        "logical_job_count": _logical_job_count(contract),
        "tail_only": True,
        "spot_execution_authorized": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "authorized_unix_seconds": time.time(),
    }
    _write_once(target / AUTHORIZATION_NAME, authorization)
    validate_launch(target)
    return authorization


def validate_launch(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    contract = manifest["run_contract"]
    allowed = authorized_job_ids(contract)
    authorization = _load_canonical(
        target / AUTHORIZATION_NAME, "Step 6d v2 launch authorization"
    )
    _require_exact_keys(
        authorization, _AUTHORIZATION_KEYS, "Step 6d v2 launch authorization"
    )
    timestamp = authorization.get("authorized_unix_seconds")
    if (
        authorization.get("schema") != LAUNCH_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "explicit_tail_spot_authorization"
        or authorization.get("run_name") != manifest["run_name"]
        or authorization.get("package_manifest_sha256")
        != sha256_file(target / PACKAGE_MANIFEST_NAME)
        or authorization.get("source_sha256") != manifest["source_sha256"]
        or authorization.get("startup_sha256") != manifest["startup_sha256"]
        or authorization.get("run_contract_digest") != manifest["run_contract_digest"]
        or authorization.get("launch_target") != manifest["launch_target"]
        or authorization.get("launch_target") != build_launch_target()
        or authorization.get("cost_guard") != manifest["cost_guard"]
        or authorization.get("cost_guard") != build_cost_guard()
        or authorization.get("authorized_job_ids") != list(allowed)
        or authorization.get("logical_job_count") != _logical_job_count(contract)
        or authorization.get("tail_only") is not True
        or authorization.get("spot_execution_authorized") is not True
        or authorization.get("production_fanout_authorized") is not False
        or authorization.get("training_eligible") is not False
        or authorization.get("current_profile_changed") is not False
        or authorization.get("named_profile_added") is not False
        or authorization.get("runtime_policy_activated") is not False
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or timestamp <= 0
    ):
        raise ValueError("Step 6d v2 launch authorization changed")
    return manifest, authorization


def _authorized_run_prefix(
    manifest: Mapping[str, Any], *, project: str, bucket: str
) -> str:
    target = build_launch_target()
    if (
        manifest.get("launch_target") != target
        or project != target["project"]
        or bucket != target["bucket"]
    ):
        raise ValueError("Step 6d v2 remote target differs from authorization")
    return f"gs://{target['bucket']}/runs/{manifest['run_name']}"


def _bounded_jobs(
    values: Iterable[str],
    run_contract: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    selected = tuple(values)
    allowed = authorized_job_ids(run_contract)
    if (
        not selected
        or len(selected) > MAX_LOGICAL_JOBS
        or len(set(selected)) != len(selected)
        or any(value not in allowed for value in selected)
    ):
        raise ValueError(
            "Step 6d v2 launch requires 1-20 unique frozen candidate/reference tail jobs"
        )
    return selected


def _parse_jobs(
    value: str,
    run_contract: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if value == "all":
        return authorized_job_ids(run_contract)
    try:
        return _bounded_jobs(
            (token.strip() for token in value.split(",") if token.strip()),
            run_contract,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _checked_query(command: Sequence[str], *, label: str, timeout: int) -> str:
    result = _subprocess_run(
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
            f"failed to query Step 6d v2 {label} ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _json_query(command: Sequence[str], *, label: str, timeout: int) -> Any:
    raw = _checked_query(command, label=label, timeout=timeout)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Step 6d v2 {label} returned invalid JSON") from exc


def _selected_instance_names(
    manifest: Mapping[str, Any], selected: Sequence[str]
) -> tuple[str, ...]:
    allowed = authorized_job_ids(manifest["run_contract"])
    return tuple(
        f"{manifest['run_name']}-j{allowed.index(identifier):02d}"
        for identifier in selected
    )


def _quota_number(value: Any, *, metric: str, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Step 6d v2 {metric} quota {field} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"Step 6d v2 {metric} quota {field} is invalid")
    return int(number) if number.is_integer() else number


def preflight_launch(
    *,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    project: str,
    bucket: str,
) -> dict[str, Any]:
    """Prove quota and collision absence before any remote mutation."""

    selected_jobs = _bounded_jobs(selected, manifest["run_contract"])
    target = build_launch_target()
    if (
        manifest.get("launch_target") != target
        or project != target["project"]
        or bucket != target["bucket"]
    ):
        raise ValueError("Step 6d v2 launch target differs from authorization")
    guard = validate_cost_guard(manifest.get("cost_guard"))
    required_vcpus = len(selected_jobs) * VCPUS_PER_VM
    selected_estimate = _estimated_max_compute_usd(len(selected_jobs))

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
        label=f"{DEFAULT_REGION} quotas",
        timeout=120,
    )
    if not isinstance(region, Mapping) or not isinstance(region.get("quotas"), list):
        raise RuntimeError("Step 6d v2 regional quota payload changed")
    quota_records: dict[str, dict[str, int | float]] = {}
    for metric in ("CPUS", "PREEMPTIBLE_CPUS"):
        matches = [
            value
            for value in region["quotas"]
            if isinstance(value, Mapping) and value.get("metric") == metric
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Step 6d v2 {metric} regional quota is absent")
        limit = _quota_number(matches[0].get("limit"), metric=metric, field="limit")
        usage = _quota_number(matches[0].get("usage"), metric=metric, field="usage")
        available = limit - usage
        quota_records[metric] = {
            "limit": limit,
            "usage": usage,
            "available": available,
        }
    insufficient = [
        metric
        for metric, record in quota_records.items()
        if record["available"] < required_vcpus
    ]
    if insufficient:
        raise RuntimeError(
            "Step 6d v2 quota insufficient for "
            f"{required_vcpus} vCPUs: {','.join(insufficient)}"
        )

    instance_names = _selected_instance_names(manifest, selected_jobs)
    instances = _json_query(
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
        label="selected instance names",
        timeout=120,
    )
    if not isinstance(instances, list) or any(
        not isinstance(value, Mapping) for value in instances
    ):
        raise RuntimeError("Step 6d v2 instance-list payload changed")
    existing_names = sorted(
        {
            str(value.get("name"))
            for value in instances
            if value.get("name") in instance_names
        }
    )
    if existing_names:
        raise FileExistsError(
            "Step 6d v2 selected instances already exist: " + ",".join(existing_names)
        )

    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    done_uris = tuple(
        f"{prefix}/results/jobs/{identifier}/DONE.json" for identifier in selected_jobs
    )
    existing_done: list[str] = []
    unknown_done: list[str] = []
    for identifier, done_uri in zip(selected_jobs, done_uris, strict=True):
        result = _subprocess_run(
            [
                "gcloud",
                "storage",
                "objects",
                "describe",
                done_uri,
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
            existing_done.append(identifier)
            continue
        error = f"{result.stdout}\n{result.stderr}"
        if (
            re.search(r"(?i)not found|does not exist|no urls matched|404", error)
            is None
        ):
            unknown_done.append(identifier)
    if unknown_done:
        raise RuntimeError(
            "failed to prove Step 6d v2 DONE absence: " + ",".join(unknown_done)
        )
    if existing_done:
        raise FileExistsError(
            "Step 6d v2 selected DONE objects already exist: " + ",".join(existing_done)
        )

    return {
        "schema": LAUNCH_PREFLIGHT_SCHEMA,
        "status": "quota_and_collision_checks_passed",
        "checked_unix_seconds": time.time(),
        "region": DEFAULT_REGION,
        "selected_job_ids": list(selected_jobs),
        "selected_instance_names": list(instance_names),
        "selected_done_uris": list(done_uris),
        "required_vcpus": required_vcpus,
        "quota": quota_records,
        "instances_absent": True,
        "done_objects_absent": True,
        "cost_guard_sha256": canonical_sha256(guard),
        "selected_estimated_max_compute_usd": selected_estimate,
        "hard_tail_compute_cap_usd": guard["hard_tail_compute_cap_usd"],
    }


def preflight_resume(
    *,
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    project: str,
    bucket: str,
) -> dict[str, Any]:
    """Prove one bounded attempt-1 retry is safe before claiming it."""

    target = Path(run_dir).resolve()
    selected_jobs = _bounded_jobs(selected, manifest["run_contract"])
    prefix = _authorized_run_prefix(manifest, project=project, bucket=bucket)
    guard = validate_cost_guard(manifest.get("cost_guard"))
    allowed = authorized_job_ids(manifest["run_contract"])

    all_initial_names = tuple(
        f"{manifest['run_name']}-j{ordinal:02d}"
        for ordinal in range(_logical_job_count(manifest["run_contract"]))
    )
    all_resume_names = tuple(f"{name}-a01" for name in all_initial_names)
    instances = _json_query(
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
        label="resume instance eligibility",
        timeout=120,
    )
    if not isinstance(instances, list) or any(
        not isinstance(value, Mapping) for value in instances
    ):
        raise RuntimeError("Step 6d v2 resume instance payload changed")
    target_zones = set(manifest["launch_target"]["zones"])
    relevant_names = set(all_initial_names) | set(all_resume_names)
    if any(
        value.get("name") in relevant_names
        and str(value.get("zone", "")).rsplit("/", 1)[-1] not in target_zones
        for value in instances
    ):
        raise ValueError("Step 6d v2 resume instance escaped authorized zones")
    active_initial = sorted(
        str(value.get("name"))
        for value in instances
        if value.get("name") in all_initial_names
        and value.get("status") != "TERMINATED"
    )
    existing_resume = sorted(
        str(value.get("name"))
        for value in instances
        if value.get("name") in all_resume_names
    )
    if active_initial:
        raise FileExistsError(
            "Step 6d v2 resume has active initial instances: "
            + ",".join(active_initial)
        )
    if existing_resume:
        raise FileExistsError(
            "Step 6d v2 resume instance names already exist: "
            + ",".join(existing_resume)
        )

    incomplete_jobs: list[str] = []
    completed_jobs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="step6d-v2-resume-") as temporary:
        temporary_root = Path(temporary)
        for record in manifest["job_manifests"]:
            identifier = str(record["job_id"])
            done_uri = f"{prefix}/results/jobs/{identifier}/DONE.json"
            result = _subprocess_run(
                [
                    "gcloud",
                    "storage",
                    "objects",
                    "describe",
                    done_uri,
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
            if result.returncode != 0:
                if _instance_not_found(result.stdout, result.stderr):
                    incomplete_jobs.append(identifier)
                    continue
                raise RuntimeError(
                    "failed to classify Step 6d v2 resume DONE for " + identifier
                )
            job_dir = temporary_root / identifier
            job_dir.mkdir()
            _run(
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
            _validate_received_job(
                job_dir=job_dir,
                record=record,
                package_root=target,
            )
            completed_jobs.append(identifier)
    if not incomplete_jobs:
        raise ValueError("Step 6d v2 resume has no incomplete jobs")
    if tuple(incomplete_jobs) != selected_jobs:
        raise ValueError(
            "Step 6d v2 resume must select exactly all validated incomplete jobs"
        )

    required_vcpus = len(selected_jobs) * VCPUS_PER_VM
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
        label=f"{DEFAULT_REGION} resume quotas",
        timeout=120,
    )
    if not isinstance(region, Mapping) or not isinstance(region.get("quotas"), list):
        raise RuntimeError("Step 6d v2 resume regional quota payload changed")
    quota_records: dict[str, dict[str, int | float]] = {}
    for metric in ("CPUS", "PREEMPTIBLE_CPUS"):
        matches = [
            value
            for value in region["quotas"]
            if isinstance(value, Mapping) and value.get("metric") == metric
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Step 6d v2 resume {metric} quota is absent")
        limit = _quota_number(matches[0].get("limit"), metric=metric, field="limit")
        usage = _quota_number(matches[0].get("usage"), metric=metric, field="usage")
        quota_records[metric] = {
            "limit": limit,
            "usage": usage,
            "available": limit - usage,
        }
    insufficient = [
        metric
        for metric, record in quota_records.items()
        if record["available"] < required_vcpus
    ]
    if insufficient:
        raise RuntimeError(
            "Step 6d v2 resume quota insufficient for "
            f"{required_vcpus} vCPUs: {','.join(insufficient)}"
        )

    initial_names = tuple(
        f"{manifest['run_name']}-j{allowed.index(identifier):02d}"
        for identifier in selected_jobs
    )
    resume_names = tuple(f"{name}-a01" for name in initial_names)
    done_uris = tuple(
        f"{prefix}/results/jobs/{identifier}/DONE.json" for identifier in selected_jobs
    )
    return {
        "schema": RESUME_PREFLIGHT_SCHEMA,
        "status": "attempt1_quota_done_and_instance_checks_passed",
        "checked_unix_seconds": time.time(),
        "region": DEFAULT_REGION,
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "selected_initial_instance_names": list(initial_names),
        "selected_resume_instance_names": list(resume_names),
        "selected_done_uris": list(done_uris),
        "required_vcpus": required_vcpus,
        "quota": quota_records,
        "no_active_initial_instances": True,
        "resume_instance_names_absent": True,
        "done_objects_absent": True,
        "validated_completed_job_ids": completed_jobs,
        "all_incomplete_jobs_selected": True,
        "cost_guard_sha256": canonical_sha256(guard),
        "max_cumulative_vm_jobs": MAX_CUMULATIVE_VM_JOBS,
        "all_attempts_estimated_max_compute_usd": (
            ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "hard_tail_compute_cap_usd": HARD_TAIL_COMPUTE_CAP_USD,
    }


def _acquire_launch_claim(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically and permanently reserve this run before remote mutation."""

    claim = {
        "schema": LAUNCH_CLAIM_SCHEMA,
        "status": "exclusive_claim_acquired_before_remote_mutation",
        "run_name": manifest["run_name"],
        "selected_job_ids": list(selected),
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "crash_reuse_authorized": False,
    }
    path = target / LAUNCH_CLAIM_NAME
    _write_once(path, claim)
    if _load_canonical(path, "Step 6d v2 launch claim") != claim:
        raise ValueError("Step 6d v2 launch claim changed after acquisition")
    return claim


def _publish_package(
    *, target: Path, manifest: Mapping[str, Any], project: str, bucket: str
) -> str:
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    sources = [
        (target / PACKAGE_MANIFEST_NAME, f"{prefix}/{PACKAGE_MANIFEST_NAME}"),
        (target / PACKAGE_READY_NAME, f"{prefix}/source/{PACKAGE_READY_NAME}"),
        (target / SOURCE_NAME, f"{prefix}/source/{SOURCE_NAME}"),
        (target / STARTUP_NAME, f"{prefix}/source/{STARTUP_NAME}"),
        (target / AUTHORIZATION_NAME, f"{prefix}/source/{AUTHORIZATION_NAME}"),
    ]
    sources.extend(
        (target / record["path"], f"{prefix}/source/{record['path']}")
        for record in manifest["job_manifests"]
    )
    for source, uri in sources:
        _publish_once(source, uri, project=project)
    return prefix


def _job_record_by_id(
    manifest: Mapping[str, Any], identifier: str
) -> Mapping[str, Any]:
    matches = [
        record
        for record in manifest["job_manifests"]
        if record.get("job_id") == identifier
    ]
    if len(matches) != 1:
        raise ValueError(f"Step 6d v2 job record is not unique: {identifier}")
    return matches[0]


def _create_instance(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    prefix: str,
    identifier: str,
    project: str,
    zone: str,
    attempt_index: int = 0,
    attempt_claim_uri: str | None = None,
    attempt_claim_sha256: str | None = None,
) -> dict[str, Any]:
    if attempt_index not in (0, 1):
        raise ValueError("Step 6d v2 VM attempt must be 0 or 1")
    record = _job_record_by_id(manifest, identifier)
    ordinal = authorized_job_ids(manifest["run_contract"]).index(identifier)
    suffix = "" if attempt_index == 0 else "-a01"
    instance = f"{manifest['run_name']}-j{ordinal:02d}{suffix}"
    bucket_from_prefix = prefix.split("/")[2]
    cost_guard = validate_cost_guard(manifest["cost_guard"])
    authorization_uri = f"{prefix}/source/{AUTHORIZATION_NAME}"
    authorization_sha256 = sha256_file(target / AUTHORIZATION_NAME)
    expected_claim_uri = (
        f"{prefix}/control/{LAUNCH_CLAIM_NAME}"
        if attempt_index == 0
        else f"{prefix}/resume/{RESUME_CLAIM_NAME}"
    )
    if attempt_claim_uri != expected_claim_uri or not _is_sha256(attempt_claim_sha256):
        raise ValueError("Step 6d v2 attempt claim digest changed")
    initial_claim_path = target / LAUNCH_CLAIM_NAME
    initial_result_path = target / LAUNCH_RESULT_NAME
    initial_claim_sha256 = (
        sha256_file(initial_claim_path) if initial_claim_path.is_file() else "none"
    )
    initial_result_sha256 = (
        sha256_file(initial_result_path) if initial_result_path.is_file() else "none"
    )
    metadata = ",".join(
        (
            f"PROJECT_ID={project}",
            f"BUCKET={bucket_from_prefix}",
            f"RUN_NAME={manifest['run_name']}",
            f"JOB_ID={identifier}",
            f"SOURCE_URI={prefix}/source/{SOURCE_NAME}",
            f"SOURCE_SHA256={manifest['source_sha256']}",
            f"MANIFEST_URI={prefix}/{PACKAGE_MANIFEST_NAME}",
            f"MANIFEST_SHA256={authorization['package_manifest_sha256']}",
            f"JOB_MANIFEST_URI={prefix}/source/{record['path']}",
            f"JOB_MANIFEST_SHA256={record['sha256']}",
            f"AUTHORIZATION_URI={authorization_uri}",
            f"AUTHORIZATION_SHA256={authorization_sha256}",
            f"ATTEMPT_INDEX={attempt_index}",
            f"ATTEMPT_CLAIM_URI={attempt_claim_uri}",
            f"ATTEMPT_CLAIM_SHA256={attempt_claim_sha256}",
            f"INITIAL_LAUNCH_CLAIM_SHA256={initial_claim_sha256}",
            f"INITIAL_LAUNCH_RESULT_SHA256={initial_result_sha256}",
            f"COST_GUARD_SHA256={canonical_sha256(cost_guard)}",
            "MAX_RUNTIME_SECONDS=" f"{cost_guard['internal_watchdog_seconds_per_vm']}",
            "SELF_DELETE=1",
        )
    )
    _run(
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
        "work_hand_indices": record["work_hand_indices"],
        "instance": instance,
        "zone": zone,
        "attempt_index": attempt_index,
        "status": "created",
    }


def _instance_not_found(stdout: str, stderr: str) -> bool:
    return (
        re.search(
            r"(?i)not found|does not exist|was not found|no urls matched|404",
            f"{stdout}\n{stderr}",
        )
        is not None
    )


def _probe_instance(*, instance: str, zone: str, project: str):
    return _cleanup_command(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            instance,
            "--zone",
            zone,
            "--project",
            project,
            "--format=json(name,status,zone)",
        ],
        timeout=120,
    )


def _cleanup_command(
    command: Sequence[str], *, timeout: int
) -> subprocess.CompletedProcess[str]:
    try:
        return _subprocess_run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # cleanup must persist ambiguity instead of escaping
        return subprocess.CompletedProcess(
            list(command),
            -1,
            "",
            f"{type(exc).__name__}: {exc}",
        )


def _cleanup_selected_instances(
    *,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    project: str,
    zones: Sequence[str],
    attempt_index: int = 0,
) -> list[dict[str, Any]]:
    """Probe and terminate every selected VM, including ambiguous creates."""

    outcomes: list[dict[str, Any]] = []
    allowed = authorized_job_ids(manifest["run_contract"])
    for identifier in selected:
        ordinal = allowed.index(identifier)
        suffix = "" if attempt_index == 0 else "-a01"
        instance = f"{manifest['run_name']}-j{ordinal:02d}{suffix}"
        zone = zones[ordinal % len(zones)]
        probe = _probe_instance(instance=instance, zone=zone, project=project)
        if probe.returncode != 0 and _instance_not_found(probe.stdout, probe.stderr):
            outcomes.append(
                {
                    "job_id": identifier,
                    "attempt_index": attempt_index,
                    "instance": instance,
                    "zone": zone,
                    "probe_status": "absent",
                    "status": "absent",
                    "absence_proven": True,
                    "compute_stopped_or_absent": True,
                }
            )
            continue
        probe_status = "exists" if probe.returncode == 0 else "absence_unproven"
        delete = _cleanup_command(
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
            timeout=300,
        )
        stop_returncode: int | None = None
        if delete.returncode != 0:
            stop = _cleanup_command(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "stop",
                    instance,
                    "--zone",
                    zone,
                    "--project",
                    project,
                    "--quiet",
                ],
                timeout=300,
            )
            stop_returncode = stop.returncode
        final_probe = _probe_instance(instance=instance, zone=zone, project=project)
        absence_proven = final_probe.returncode != 0 and _instance_not_found(
            final_probe.stdout, final_probe.stderr
        )
        final_status = ""
        if final_probe.returncode == 0:
            try:
                final_payload = json.loads(final_probe.stdout)
                if isinstance(final_payload, Mapping):
                    final_status = str(final_payload.get("status", ""))
            except json.JSONDecodeError:
                final_status = ""
        compute_stopped = absence_proven or final_status == "TERMINATED"
        if absence_proven:
            status = "deleted" if delete.returncode == 0 else "absent_after_cleanup"
        elif final_status == "TERMINATED":
            status = "stopped_after_delete_failure"
        else:
            status = "cleanup_failed_absence_unproven"
        outcomes.append(
            {
                "job_id": identifier,
                "attempt_index": attempt_index,
                "instance": instance,
                "zone": zone,
                "probe_status": probe_status,
                "delete_returncode": delete.returncode,
                "stop_returncode": stop_returncode,
                "final_instance_status": final_status or None,
                "status": status,
                "absence_proven": absence_proven,
                "compute_stopped_or_absent": compute_stopped,
            }
        )
    return outcomes


def launch_jobs(
    *,
    run_dir: str | Path,
    jobs: Iterable[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zones: Sequence[str] = DEFAULT_ZONES,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch(target)
    selected = _bounded_jobs(jobs, manifest["run_contract"])
    allowed = authorized_job_ids(manifest["run_contract"])
    if selected != allowed:
        raise ValueError("Step 6d v2 initial launch requires exactly all 20 jobs")
    if tuple(zones) != DEFAULT_ZONES:
        raise ValueError("Step 6d v2 launch must use both authorized zones in order")
    if (target / LAUNCH_RESULT_NAME).exists():
        raise FileExistsError("Step 6d v2 launch result already exists")
    if (
        manifest["launch_target"] != authorization["launch_target"]
        or manifest["cost_guard"] != authorization["cost_guard"]
    ):
        raise ValueError("Step 6d v2 authorization launch boundary changed")
    preflight = preflight_launch(
        manifest=manifest,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    image = json.loads(
        _run(
            [
                "gcloud",
                "compute",
                "images",
                "describe",
                EXPECTED_IMAGE_NAME,
                "--project",
                "debian-cloud",
                "--format=json",
            ],
            timeout=120,
        ).stdout
    )
    if (
        str(image.get("id")) != EXPECTED_IMAGE_ID
        or image.get("selfLink") != EXPECTED_IMAGE_SELF_LINK
    ):
        raise ValueError("Step 6d v2 immutable Debian image changed")
    claim = _acquire_launch_claim(
        target=target,
        manifest=manifest,
        selected=selected,
        preflight=preflight,
    )
    prefix = _authorized_run_prefix(manifest, project=project, bucket=bucket)
    claim_uri = f"{prefix}/control/{LAUNCH_CLAIM_NAME}"
    _publish_once(target / LAUNCH_CLAIM_NAME, claim_uri, project=project)
    claim_sha256 = sha256_file(target / LAUNCH_CLAIM_NAME)
    prefix = _publish_package(
        target=target, manifest=manifest, project=project, bucket=bucket
    )
    created: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {
            pool.submit(
                _create_instance,
                target=target,
                manifest=manifest,
                authorization=authorization,
                prefix=prefix,
                identifier=identifier,
                project=project,
                zone=zones[allowed.index(identifier) % len(zones)],
                attempt_index=0,
                attempt_claim_uri=claim_uri,
                attempt_claim_sha256=claim_sha256,
            ): identifier
            for identifier in selected
        }
        for future in concurrent.futures.as_completed(futures):
            identifier = futures[future]
            try:
                created.append(future.result())
            except Exception as exc:  # preserve all concurrent outcomes first
                failures.append(
                    {
                        "job_id": identifier,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    order = {identifier: index for index, identifier in enumerate(selected)}
    created.sort(key=lambda value: order[str(value["job_id"])])
    failures.sort(key=lambda value: order[value["job_id"]])
    result = {
        "schema": LAUNCH_RESULT_SCHEMA,
        "status": (
            "selected_tail_jobs_created"
            if not failures
            else "create_failed_cleanup_attempted"
        ),
        "run_name": manifest["run_name"],
        "selected_job_ids": list(selected),
        "created": created,
        "logical_job_count": len(created),
        "max_logical_jobs": MAX_LOGICAL_JOBS,
        "machine_type": EXPECTED_MACHINE_TYPE,
        "process_count": PROCESS_COUNT,
        "rayon_threads_per_process": RAYON_THREADS_PER_PROCESS,
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "launch_claim_sha256": canonical_sha256(claim),
        "failures": failures,
        "cleanup": [],
        "cleanup_absence_proven": True,
        "cleanup_compute_stopped_or_absent": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    if failures:
        cleanup = _cleanup_selected_instances(
            manifest=manifest,
            selected=selected,
            project=project,
            zones=zones,
        )
        result["cleanup"] = cleanup
        result["cleanup_absence_proven"] = all(
            value["absence_proven"] for value in cleanup
        )
        result["cleanup_compute_stopped_or_absent"] = all(
            value["compute_stopped_or_absent"] for value in cleanup
        )
        if not result["cleanup_compute_stopped_or_absent"]:
            result["status"] = "create_failed_cleanup_unproven"
    _write_once(target / LAUNCH_RESULT_NAME, result)
    if failures:
        raise RuntimeError(
            "Step 6d v2 instance creation failed; cleanup attempted and launch "
            f"result recorded at {target / LAUNCH_RESULT_NAME}"
        )
    return result


def _acquire_resume_claim(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    initial_claim_path = target / LAUNCH_CLAIM_NAME
    initial_result_path = target / LAUNCH_RESULT_NAME
    claim = {
        "schema": RESUME_CLAIM_SCHEMA,
        "status": "exclusive_attempt1_claim_acquired_before_remote_mutation",
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": list(selected),
        "initial_launch_claim_sha256": sha256_file(initial_claim_path),
        "initial_launch_result_sha256": sha256_file(initial_result_path),
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "third_attempt_authorized": False,
    }
    path = target / RESUME_CLAIM_NAME
    _write_once(path, claim)
    if _load_canonical(path, "Step 6d v2 resume claim") != claim:
        raise ValueError("Step 6d v2 resume claim changed after acquisition")
    return claim


def resume_jobs(
    *,
    run_dir: str | Path,
    jobs: Iterable[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zones: Sequence[str] = DEFAULT_ZONES,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch(target)
    selected = _bounded_jobs(jobs, manifest["run_contract"])
    allowed = authorized_job_ids(manifest["run_contract"])
    if tuple(zones) != DEFAULT_ZONES:
        raise ValueError("Step 6d v2 resume must use both authorized zones in order")
    if (target / RESUME_CLAIM_NAME).exists() or (target / RESUME_RESULT_NAME).exists():
        raise FileExistsError("Step 6d v2 attempt1 resume is already consumed")
    validate_receive_launch_chain(run_dir=target, manifest=manifest)
    preflight = preflight_resume(
        run_dir=target,
        manifest=manifest,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    image = json.loads(
        _run(
            [
                "gcloud",
                "compute",
                "images",
                "describe",
                EXPECTED_IMAGE_NAME,
                "--project",
                "debian-cloud",
                "--format=json",
            ],
            timeout=120,
        ).stdout
    )
    if (
        str(image.get("id")) != EXPECTED_IMAGE_ID
        or image.get("selfLink") != EXPECTED_IMAGE_SELF_LINK
    ):
        raise ValueError("Step 6d v2 resume immutable Debian image changed")
    claim = _acquire_resume_claim(
        target=target,
        manifest=manifest,
        selected=selected,
        preflight=preflight,
    )
    prefix = _authorized_run_prefix(manifest, project=project, bucket=bucket)
    claim_uri = f"{prefix}/resume/{RESUME_CLAIM_NAME}"
    _publish_once(target / RESUME_CLAIM_NAME, claim_uri, project=project)
    claim_sha256 = sha256_file(target / RESUME_CLAIM_NAME)
    created: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {
            pool.submit(
                _create_instance,
                target=target,
                manifest=manifest,
                authorization=authorization,
                prefix=prefix,
                identifier=identifier,
                project=project,
                zone=zones[allowed.index(identifier) % len(zones)],
                attempt_index=1,
                attempt_claim_uri=claim_uri,
                attempt_claim_sha256=claim_sha256,
            ): identifier
            for identifier in selected
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
    created.sort(key=lambda value: order[str(value["job_id"])])
    failures.sort(key=lambda value: order[value["job_id"]])
    result = {
        "schema": RESUME_RESULT_SCHEMA,
        "status": (
            "selected_resume_jobs_created"
            if not failures
            else "resume_create_failed_cleanup_attempted"
        ),
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": list(selected),
        "created": created,
        "logical_job_count": len(created),
        "max_resume_jobs": MAX_LOGICAL_JOBS,
        "max_cumulative_vm_jobs": MAX_CUMULATIVE_VM_JOBS,
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "resume_claim_sha256": canonical_sha256(claim),
        "initial_launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "initial_launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "failures": failures,
        "cleanup": [],
        "cleanup_absence_proven": True,
        "cleanup_compute_stopped_or_absent": True,
        "third_attempt_authorized": False,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    if failures:
        cleanup = _cleanup_selected_instances(
            manifest=manifest,
            selected=selected,
            project=project,
            zones=zones,
            attempt_index=1,
        )
        result["cleanup"] = cleanup
        result["cleanup_absence_proven"] = all(
            value["absence_proven"] for value in cleanup
        )
        result["cleanup_compute_stopped_or_absent"] = all(
            value["compute_stopped_or_absent"] for value in cleanup
        )
        if not result["cleanup_compute_stopped_or_absent"]:
            result["status"] = "resume_create_failed_cleanup_unproven"
    _write_once(target / RESUME_RESULT_NAME, result)
    if failures:
        raise RuntimeError(
            "Step 6d v2 resume instance creation failed; cleanup attempted and "
            f"result recorded at {target / RESUME_RESULT_NAME}"
        )
    return result


def _valid_timestamp(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def _validate_all20_preflight(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Step 6d v2 launch preflight must be an object")
    payload = dict(value)
    _require_exact_keys(payload, _LAUNCH_PREFLIGHT_KEYS, "Step 6d v2 launch preflight")
    selected = list(authorized_job_ids(manifest["run_contract"]))
    prefix = _authorized_run_prefix(
        manifest, project=DEFAULT_PROJECT, bucket=DEFAULT_BUCKET
    )
    expected_names = list(_selected_instance_names(manifest, selected))
    expected_done = [
        f"{prefix}/results/jobs/{identifier}/DONE.json" for identifier in selected
    ]
    quota = payload.get("quota")
    quota_valid = isinstance(quota, Mapping) and set(quota) == {
        "CPUS",
        "PREEMPTIBLE_CPUS",
    }
    if quota_valid:
        for metric in ("CPUS", "PREEMPTIBLE_CPUS"):
            record = quota[metric]
            if not isinstance(record, Mapping) or set(record) != {
                "limit",
                "usage",
                "available",
            }:
                quota_valid = False
                break
            limit = record.get("limit")
            usage = record.get("usage")
            available = record.get("available")
            if (
                any(isinstance(item, bool) for item in (limit, usage, available))
                or any(
                    not isinstance(item, (int, float))
                    for item in (limit, usage, available)
                )
                or any(
                    not math.isfinite(float(item)) for item in (limit, usage, available)
                )
                or limit < 0
                or usage < 0
                or available != limit - usage
                or available < MAX_LOGICAL_JOBS * VCPUS_PER_VM
            ):
                quota_valid = False
                break
    if (
        payload.get("schema") != LAUNCH_PREFLIGHT_SCHEMA
        or payload.get("status") != "quota_and_collision_checks_passed"
        or not _valid_timestamp(payload.get("checked_unix_seconds"))
        or payload.get("region") != DEFAULT_REGION
        or payload.get("selected_job_ids") != selected
        or payload.get("selected_instance_names") != expected_names
        or payload.get("selected_done_uris") != expected_done
        or payload.get("required_vcpus") != MAX_LOGICAL_JOBS * VCPUS_PER_VM
        or not quota_valid
        or payload.get("instances_absent") is not True
        or payload.get("done_objects_absent") is not True
        or payload.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or payload.get("selected_estimated_max_compute_usd")
        != ALL_20_ESTIMATED_MAX_COMPUTE_USD
        or payload.get("hard_tail_compute_cap_usd") != HARD_TAIL_COMPUTE_CAP_USD
    ):
        raise ValueError("Step 6d v2 all-20 launch preflight changed")
    return payload


def validate_receive_launch_chain(
    *, run_dir: str | Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the exact successful all-20 claim/result chain before receive."""

    target = Path(run_dir).resolve()
    claim_path = target / LAUNCH_CLAIM_NAME
    result_path = target / LAUNCH_RESULT_NAME
    if not claim_path.is_file() or not result_path.is_file():
        raise ValueError("Step 6d v2 receive launch chain is incomplete")
    claim = _load_canonical(claim_path, "Step 6d v2 launch claim")
    result = _load_canonical(result_path, "Step 6d v2 launch result")
    _require_exact_keys(claim, _LAUNCH_CLAIM_KEYS, "Step 6d v2 launch claim")
    _require_exact_keys(result, _LAUNCH_RESULT_KEYS, "Step 6d v2 launch result")
    selected = list(authorized_job_ids(manifest["run_contract"]))
    preflight = _validate_all20_preflight(result.get("preflight"), manifest=manifest)
    claim_sha256 = sha256_file(claim_path)
    expected_created = []
    for ordinal, identifier in enumerate(selected):
        record = _job_record_by_id(manifest, identifier)
        expected_created.append(
            {
                "job_id": identifier,
                "source_role": record["source_role"],
                "work_hand_indices": record["work_hand_indices"],
                "instance": f"{manifest['run_name']}-j{ordinal:02d}",
                "zone": DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)],
                "attempt_index": 0,
                "status": "created",
            }
        )
    expected_by_id = {value["job_id"]: value for value in expected_created}
    created = result.get("created")
    failures = result.get("failures")
    cleanup = result.get("cleanup")
    created_valid = (
        isinstance(created, list)
        and all(isinstance(value, Mapping) for value in created)
        and len({value.get("job_id") for value in created}) == len(created)
        and all(
            value.get("job_id") in expected_by_id
            and dict(value) == expected_by_id[value["job_id"]]
            for value in created
        )
    )
    failure_valid = (
        isinstance(failures, list)
        and all(
            isinstance(value, Mapping)
            and set(value) == {"job_id", "error_type", "error"}
            and value.get("job_id") in selected
            and isinstance(value.get("error_type"), str)
            and bool(value.get("error_type"))
            and isinstance(value.get("error"), str)
            and bool(value.get("error"))
            for value in failures
        )
        and len({value.get("job_id") for value in failures}) == len(failures)
    )
    outcome_ids = (
        {value.get("job_id") for value in created}
        | {value.get("job_id") for value in failures}
        if created_valid and failure_valid
        else set()
    )
    success = result.get("status") == "selected_tail_jobs_created"
    if success:
        outcome_valid = created == expected_created and failures == [] and cleanup == []
    else:
        cleanup_valid = (
            isinstance(cleanup, list)
            and len(cleanup) == _logical_job_count(manifest["run_contract"])
            and {value.get("job_id") for value in cleanup if isinstance(value, Mapping)}
            == set(selected)
            and all(
                isinstance(value, Mapping)
                and isinstance(value.get("absence_proven"), bool)
                and isinstance(value.get("compute_stopped_or_absent"), bool)
                for value in cleanup
            )
        )
        outcome_valid = (
            result.get("status")
            in {"create_failed_cleanup_attempted", "create_failed_cleanup_unproven"}
            and bool(failures)
            and outcome_ids == set(selected)
            and cleanup_valid
            and result.get("cleanup_absence_proven")
            == all(value["absence_proven"] for value in cleanup)
            and result.get("cleanup_compute_stopped_or_absent")
            == all(value["compute_stopped_or_absent"] for value in cleanup)
            and (result.get("status") == "create_failed_cleanup_unproven")
            == (not result.get("cleanup_compute_stopped_or_absent"))
        )
    if (
        claim.get("schema") != LAUNCH_CLAIM_SCHEMA
        or claim.get("status") != "exclusive_claim_acquired_before_remote_mutation"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("selected_job_ids") != selected
        or claim.get("run_contract_digest") != manifest["run_contract_digest"]
        or claim.get("launch_target") != manifest["launch_target"]
        or claim.get("launch_target") != build_launch_target()
        or claim.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or claim.get("preflight_sha256") != canonical_sha256(preflight)
        or not _valid_timestamp(claim.get("claimed_unix_seconds"))
        or claim.get("crash_reuse_authorized") is not False
        or result.get("schema") != LAUNCH_RESULT_SCHEMA
        or result.get("run_name") != manifest["run_name"]
        or result.get("selected_job_ids") != selected
        or not created_valid
        or not failure_valid
        or not outcome_valid
        or result.get("logical_job_count") != len(created)
        or result.get("max_logical_jobs") != MAX_LOGICAL_JOBS
        or result.get("machine_type") != EXPECTED_MACHINE_TYPE
        or result.get("process_count") != PROCESS_COUNT
        or result.get("rayon_threads_per_process") != RAYON_THREADS_PER_PROCESS
        or result.get("run_contract_digest") != manifest["run_contract_digest"]
        or result.get("launch_target") != manifest["launch_target"]
        or result.get("cost_guard") != manifest["cost_guard"]
        or result.get("preflight") != preflight
        or result.get("launch_claim_sha256") != claim_sha256
        or any(
            result.get(field) is not False
            for field in (
                "production_fanout_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Step 6d v2 all-20 receive launch chain changed")
    return claim, result


def _validate_resume_preflight(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any], selected: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Step 6d v2 resume preflight must be an object")
    payload = dict(value)
    _require_exact_keys(payload, _RESUME_PREFLIGHT_KEYS, "Step 6d v2 resume preflight")
    allowed = authorized_job_ids(manifest["run_contract"])
    if tuple(
        identifier for identifier in allowed if identifier in set(selected)
    ) != tuple(selected):
        raise ValueError("Step 6d v2 resume job ordering changed")
    completed = [
        identifier for identifier in allowed if identifier not in set(selected)
    ]
    initial_names = [
        f"{manifest['run_name']}-j{allowed.index(identifier):02d}"
        for identifier in selected
    ]
    resume_names = [f"{name}-a01" for name in initial_names]
    prefix = _authorized_run_prefix(
        manifest, project=DEFAULT_PROJECT, bucket=DEFAULT_BUCKET
    )
    done_uris = [
        f"{prefix}/results/jobs/{identifier}/DONE.json" for identifier in selected
    ]
    quota = payload.get("quota")
    quota_valid = isinstance(quota, Mapping) and set(quota) == {
        "CPUS",
        "PREEMPTIBLE_CPUS",
    }
    if quota_valid:
        for metric in ("CPUS", "PREEMPTIBLE_CPUS"):
            record = quota[metric]
            if not isinstance(record, Mapping) or set(record) != {
                "limit",
                "usage",
                "available",
            }:
                quota_valid = False
                break
            limit, usage, available = (
                record.get("limit"),
                record.get("usage"),
                record.get("available"),
            )
            if (
                any(isinstance(item, bool) for item in (limit, usage, available))
                or any(
                    not isinstance(item, (int, float))
                    for item in (limit, usage, available)
                )
                or available != limit - usage
                or available < len(selected) * VCPUS_PER_VM
            ):
                quota_valid = False
                break
    if (
        payload.get("schema") != RESUME_PREFLIGHT_SCHEMA
        or payload.get("status") != "attempt1_quota_done_and_instance_checks_passed"
        or not _valid_timestamp(payload.get("checked_unix_seconds"))
        or payload.get("region") != DEFAULT_REGION
        or payload.get("attempt_index") != 1
        or payload.get("selected_job_ids") != list(selected)
        or payload.get("selected_initial_instance_names") != initial_names
        or payload.get("selected_resume_instance_names") != resume_names
        or payload.get("selected_done_uris") != done_uris
        or payload.get("required_vcpus") != len(selected) * VCPUS_PER_VM
        or not quota_valid
        or payload.get("no_active_initial_instances") is not True
        or payload.get("resume_instance_names_absent") is not True
        or payload.get("done_objects_absent") is not True
        or payload.get("validated_completed_job_ids") != completed
        or payload.get("all_incomplete_jobs_selected") is not True
        or payload.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or payload.get("max_cumulative_vm_jobs") != MAX_CUMULATIVE_VM_JOBS
        or payload.get("all_attempts_estimated_max_compute_usd")
        != ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        or payload.get("hard_tail_compute_cap_usd") != HARD_TAIL_COMPUTE_CAP_USD
    ):
        raise ValueError("Step 6d v2 resume preflight changed")
    return payload


def validate_receive_resume_chain(
    *, run_dir: str | Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    target = Path(run_dir).resolve()
    claim_path = target / RESUME_CLAIM_NAME
    result_path = target / RESUME_RESULT_NAME
    if not claim_path.exists() and not result_path.exists():
        return None
    if not claim_path.is_file() or not result_path.is_file():
        raise ValueError("Step 6d v2 resume chain is incomplete")
    claim = _load_canonical(claim_path, "Step 6d v2 resume claim")
    result = _load_canonical(result_path, "Step 6d v2 resume result")
    _require_exact_keys(claim, _RESUME_CLAIM_KEYS, "Step 6d v2 resume claim")
    _require_exact_keys(result, _RESUME_RESULT_KEYS, "Step 6d v2 resume result")
    raw_selected = claim.get("selected_job_ids")
    if not isinstance(raw_selected, list):
        raise ValueError("Step 6d v2 resume selected jobs changed")
    selected = _bounded_jobs(raw_selected, manifest["run_contract"])
    preflight = _validate_resume_preflight(
        result.get("preflight"), manifest=manifest, selected=selected
    )
    expected_created = []
    for identifier in selected:
        ordinal = authorized_job_ids(manifest["run_contract"]).index(identifier)
        record = _job_record_by_id(manifest, identifier)
        expected_created.append(
            {
                "job_id": identifier,
                "source_role": record["source_role"],
                "work_hand_indices": record["work_hand_indices"],
                "instance": f"{manifest['run_name']}-j{ordinal:02d}-a01",
                "zone": DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)],
                "attempt_index": 1,
                "status": "created",
            }
        )
    created = result.get("created")
    failures = result.get("failures")
    cleanup = result.get("cleanup")
    expected_by_id = {value["job_id"]: value for value in expected_created}
    created_valid = (
        isinstance(created, list)
        and all(isinstance(value, Mapping) for value in created)
        and all(
            value.get("job_id") in expected_by_id
            and dict(value) == expected_by_id[value["job_id"]]
            for value in created
        )
        and len({value.get("job_id") for value in created}) == len(created)
    )
    failure_valid = (
        isinstance(failures, list)
        and all(
            isinstance(value, Mapping)
            and set(value) == {"job_id", "error_type", "error"}
            and value.get("job_id") in selected
            for value in failures
        )
        and len({value.get("job_id") for value in failures}) == len(failures)
    )
    success = result.get("status") == "selected_resume_jobs_created"
    if success:
        outcome_valid = created == expected_created and failures == [] and cleanup == []
    else:
        cleanup_valid = (
            isinstance(cleanup, list)
            and len(cleanup) == len(selected)
            and {value.get("job_id") for value in cleanup if isinstance(value, Mapping)}
            == set(selected)
        )
        outcome_ids = (
            {value.get("job_id") for value in created}
            | {value.get("job_id") for value in failures}
            if created_valid and failure_valid
            else set()
        )
        outcome_valid = (
            result.get("status")
            in {
                "resume_create_failed_cleanup_attempted",
                "resume_create_failed_cleanup_unproven",
            }
            and bool(failures)
            and outcome_ids == set(selected)
            and cleanup_valid
            and result.get("cleanup_absence_proven")
            == all(bool(value.get("absence_proven")) for value in cleanup)
            and result.get("cleanup_compute_stopped_or_absent")
            == all(bool(value.get("compute_stopped_or_absent")) for value in cleanup)
        )
    initial_claim_sha = sha256_file(target / LAUNCH_CLAIM_NAME)
    initial_result_sha = sha256_file(target / LAUNCH_RESULT_NAME)
    resume_claim_sha = sha256_file(claim_path)
    if (
        claim.get("schema") != RESUME_CLAIM_SCHEMA
        or claim.get("status")
        != "exclusive_attempt1_claim_acquired_before_remote_mutation"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("attempt_index") != 1
        or claim.get("selected_job_ids") != list(selected)
        or claim.get("initial_launch_claim_sha256") != initial_claim_sha
        or claim.get("initial_launch_result_sha256") != initial_result_sha
        or claim.get("run_contract_digest") != manifest["run_contract_digest"]
        or claim.get("launch_target") != manifest["launch_target"]
        or claim.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or claim.get("preflight_sha256") != canonical_sha256(preflight)
        or not _valid_timestamp(claim.get("claimed_unix_seconds"))
        or claim.get("third_attempt_authorized") is not False
        or result.get("schema") != RESUME_RESULT_SCHEMA
        or result.get("run_name") != manifest["run_name"]
        or result.get("attempt_index") != 1
        or result.get("selected_job_ids") != list(selected)
        or not created_valid
        or not failure_valid
        or not outcome_valid
        or result.get("logical_job_count") != len(created)
        or result.get("max_resume_jobs") != MAX_LOGICAL_JOBS
        or result.get("max_cumulative_vm_jobs") != MAX_CUMULATIVE_VM_JOBS
        or result.get("run_contract_digest") != manifest["run_contract_digest"]
        or result.get("launch_target") != manifest["launch_target"]
        or result.get("cost_guard") != manifest["cost_guard"]
        or result.get("preflight") != preflight
        or result.get("resume_claim_sha256") != resume_claim_sha
        or result.get("initial_launch_claim_sha256") != initial_claim_sha
        or result.get("initial_launch_result_sha256") != initial_result_sha
        or result.get("third_attempt_authorized") is not False
        or any(
            result.get(field) is not False
            for field in (
                "production_fanout_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Step 6d v2 receive resume chain changed")
    return claim, result


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    manifest, _authorization = validate_launch(run_dir)
    allowed = authorized_job_ids(manifest["run_contract"])
    logical_job_count = _logical_job_count(manifest["run_contract"])
    prefix = _authorized_run_prefix(manifest, project=project, bucket=bucket)
    listing = _subprocess_run(
        ["gcloud", "storage", "ls", "--recursive", prefix, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError("failed to inspect Step 6d v2 GCS prefix")
    objects = listing.stdout.splitlines()
    instances_result = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            f"--filter=name~'^{manifest['run_name']}-j[01][0-9](-a01)?$'",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if instances_result.returncode != 0:
        raise RuntimeError("failed to inspect Step 6d v2 instances")
    instances = (
        json.loads(instances_result.stdout) if instances_result.stdout.strip() else []
    )
    if not isinstance(instances, list) or any(
        not isinstance(row, Mapping) for row in instances
    ):
        raise RuntimeError("Step 6d v2 instance status payload changed")
    expected_names = {
        name
        for ordinal in range(logical_job_count)
        for name in (
            f"{manifest['run_name']}-j{ordinal:02d}",
            f"{manifest['run_name']}-j{ordinal:02d}-a01",
        )
    }
    selected_rows = [row for row in instances if row.get("name") in expected_names]
    names = [str(row.get("name")) for row in selected_rows]
    if len(names) != len(set(names)):
        raise ValueError("Step 6d v2 instance name is duplicated across zones")
    target_zones = set(manifest["launch_target"]["zones"])
    if any(
        str(row.get("zone", "")).rsplit("/", 1)[-1] not in target_zones
        for row in selected_rows
    ):
        raise ValueError("Step 6d v2 instance escaped the authorized zones")
    by_name = {row.get("name"): row for row in selected_rows}
    jobs: dict[str, Any] = {}
    for ordinal, identifier in enumerate(allowed):
        result_prefix = f"/results/jobs/{identifier}/"
        progress_prefix = f"/progress/jobs/{identifier}/"
        vm = by_name.get(f"{manifest['run_name']}-j{ordinal:02d}")
        resume_vm = by_name.get(f"{manifest['run_name']}-j{ordinal:02d}-a01")
        jobs[identifier] = {
            "done_present_unvalidated": any(
                uri.endswith(f"/results/jobs/{identifier}/DONE.json") for uri in objects
            ),
            "heartbeat_present": any(
                uri.endswith(f"/progress/jobs/{identifier}/heartbeat.json")
                for uri in objects
            ),
            "progress_object_count": sum(progress_prefix in uri for uri in objects),
            "result_object_count": sum(result_prefix in uri for uri in objects),
            "instance": (
                None
                if vm is None
                else {
                    "name": vm.get("name"),
                    "status": vm.get("status"),
                    "zone": str(vm.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
            "resume_instance": (
                None
                if resume_vm is None
                else {
                    "name": resume_vm.get("name"),
                    "status": resume_vm.get("status"),
                    "zone": str(resume_vm.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
        }
    done_count = sum(row["done_present_unvalidated"] for row in jobs.values())
    return {
        "schema": CLOUD_STATUS_SCHEMA,
        "status": (
            "all_done_present_receive_to_validate"
            if done_count == logical_job_count
            else "incomplete"
        ),
        "run_name": manifest["run_name"],
        "done_present_count": done_count,
        "all_done_present_unvalidated": done_count == logical_job_count,
        "jobs": jobs,
        "logical_job_count": logical_job_count,
        "run_contract_digest": manifest["run_contract_digest"],
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def _expected_received_relatives(source_role: str, hand_index: int) -> tuple[str, ...]:
    return (
        "DONE.json",
        f"hands/{source_role}/hand_{hand_index:03d}.json",
        f"roots/hand_{hand_index:03d}.json",
        "run_contract.json",
        "shard_manifest.json",
    )


def _validate_received_job(
    *, job_dir: Path, record: Mapping[str, Any], package_root: Path
) -> dict[str, Any]:
    identifier = str(record["job_id"])
    manifest_path = package_root / record["path"]
    shard_manifest = validate_job_manifest(
        _load_canonical(manifest_path, f"Step 6d v2 received job {identifier}")
    )
    role = str(shard_manifest["source_role"])
    hand_index = int(shard_manifest["work_hand_indices"][0])
    actual = tuple(
        sorted(
            path.relative_to(job_dir).as_posix()
            for path in job_dir.rglob("*")
            if path.is_file()
        )
    )
    expected = _expected_received_relatives(role, hand_index)
    if actual != expected:
        raise ValueError(f"Step 6d v2 received file set changed: {identifier}")
    runner = _runner_module()
    stored_contract = _load_canonical(
        job_dir / "run_contract.json", f"Step 6d v2 run contract {identifier}"
    )
    stored_manifest = _load_canonical(
        job_dir / "shard_manifest.json", f"Step 6d v2 shard manifest {identifier}"
    )
    root_path = job_dir / "roots" / f"hand_{hand_index:03d}.json"
    hand_path = job_dir / "hands" / role / f"hand_{hand_index:03d}.json"
    _load_canonical(root_path, f"Step 6d v2 root {identifier}")
    _load_canonical(hand_path, f"Step 6d v2 source hand {identifier}")
    done = _load_canonical(job_dir / "DONE.json", f"Step 6d v2 DONE {identifier}")
    if (
        stored_contract != shard_manifest["run_contract"]
        or stored_manifest != shard_manifest
    ):
        raise ValueError(f"Step 6d v2 received contract chain changed: {identifier}")
    validated_done = runner.validate_completed_output(job_dir)
    if (
        validated_done != done
        or done.get("schema") != runner._done_schema(stored_contract)
        or done.get("run_contract_digest") != shard_manifest["run_contract_digest"]
        or done.get("source_role") != role
        or done.get("work_hand_indices") != [hand_index]
        or done.get("completed_hand_indices") != [hand_index]
        or done.get("reference_library_sha256")
        != stored_contract["reference_library_sha256"]
        or done.get("candidate_library_sha256")
        != stored_contract["candidate_library_sha256"]
        or done.get("native_library_sha256")
        != stored_contract[f"{role}_library_sha256"]
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
        raise ValueError(f"Step 6d v2 received DONE binding changed: {identifier}")
    return {
        "job_id": identifier,
        "source_role": role,
        "work_hand_indices": [hand_index],
        "done_path": f"jobs/{identifier}/DONE.json",
        "done_sha256": sha256_file(job_dir / "DONE.json"),
        "source_hand_path": f"jobs/{identifier}/hands/{role}/hand_{hand_index:03d}.json",
        "source_hand_sha256": sha256_file(hand_path),
        "root_sha256": sha256_file(root_path),
        "run_contract_digest": shard_manifest["run_contract_digest"],
    }


def validate_receive_receipt(
    value: Mapping[str, Any],
    *,
    run_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Step 6d v2 receive receipt must be an object")
    payload = dict(value)
    _require_exact_keys(payload, _RECEIVE_RECEIPT_KEYS, "Step 6d v2 receive receipt")
    run_name = str(payload.get("run_name", ""))
    target = build_launch_target()
    result_prefix = f"gs://{target['bucket']}/runs/{run_name}/results"
    jobs = payload.get("jobs")
    # Legacy standalone receipt validation remains byte-compatible.  New tail
    # variants must be checked with their validated package contract.
    if run_contract is None and payload.get("work_hand_indices") != list(
        TAIL_HAND_INDICES
    ):
        raise ValueError("Step 6d v2 non-legacy receipt requires its run contract")
    contract = (
        None
        if run_contract is None
        else _runner_module().validate_run_contract(dict(run_contract))
    )
    tail_indices = _contract_tail_hand_indices(contract)
    logical_job_count = _logical_job_count(contract)
    expected_ids = list(authorized_job_ids(contract))
    resume_claim_sha = payload.get("resume_claim_sha256")
    resume_result_sha = payload.get("resume_result_sha256")
    resume_hashes_valid = (resume_claim_sha is None and resume_result_sha is None) or (
        _is_sha256(resume_claim_sha) and _is_sha256(resume_result_sha)
    )
    if (
        payload.get("schema") != RECEIVE_SCHEMA
        or payload.get("status") != "exact_tail_source_jobs_received_and_validated"
        or _SAFE_RUN.fullmatch(run_name) is None
        or not _is_sha256(payload.get("package_manifest_sha256"))
        or not _is_sha256(payload.get("launch_authorization_sha256"))
        or not _is_sha256(payload.get("launch_claim_sha256"))
        or not _is_sha256(payload.get("launch_result_sha256"))
        or not resume_hashes_valid
        or not _is_sha256(payload.get("run_contract_digest"))
        or (
            contract is not None
            and payload.get("run_contract_digest")
            != _runner_module().canonical_sha256(contract)
        )
        or payload.get("launch_target") != target
        or payload.get("result_prefix") != result_prefix
        or payload.get("work_hand_indices") != list(tail_indices)
        or payload.get("source_roles") != list(SOURCE_ROLES)
        or payload.get("logical_job_count") != logical_job_count
        or not isinstance(jobs, list)
        or len(jobs) != logical_job_count
        or any(not isinstance(row, Mapping) for row in jobs)
        or [row.get("job_id") for row in jobs] != expected_ids
        or payload.get("candidate_done_paths")
        != [
            f"jobs/{identifier}/DONE.json"
            for identifier in expected_ids[: len(tail_indices)]
        ]
        or payload.get("reference_done_paths")
        != [
            f"jobs/{identifier}/DONE.json"
            for identifier in expected_ids[len(tail_indices) :]
        ]
        or payload.get("source_isolation_validated") is not True
        or payload.get("merge_executed") is not False
        or any(
            payload.get(field) is not False
            for field in (
                "production_fanout_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Step 6d v2 receive receipt changed")
    return payload


def build_receive_receipt(
    *,
    manifest: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    package_manifest_sha256: str,
    launch_authorization_sha256: str,
    launch_claim_sha256: str,
    launch_result_sha256: str,
    resume_claim_sha256: str | None = None,
    resume_result_sha256: str | None = None,
) -> dict[str, Any]:
    target = build_launch_target()
    if manifest.get("launch_target") != target:
        raise ValueError("Step 6d v2 receive target differs from package")
    runner = _runner_module()
    contract = runner.validate_run_contract(dict(manifest["run_contract"]))
    if manifest.get("run_contract_digest") != runner.canonical_sha256(contract):
        raise ValueError("Step 6d v2 receive contract differs from package")
    tail_indices = _contract_tail_hand_indices(contract)
    candidate_done = [
        row["done_path"] for row in jobs if row["source_role"] == "candidate"
    ]
    reference_done = [
        row["done_path"] for row in jobs if row["source_role"] == "reference"
    ]
    return validate_receive_receipt(
        {
            "schema": RECEIVE_SCHEMA,
            "status": "exact_tail_source_jobs_received_and_validated",
            "run_name": manifest["run_name"],
            "package_manifest_sha256": package_manifest_sha256,
            "launch_authorization_sha256": launch_authorization_sha256,
            "launch_claim_sha256": launch_claim_sha256,
            "launch_result_sha256": launch_result_sha256,
            "resume_claim_sha256": resume_claim_sha256,
            "resume_result_sha256": resume_result_sha256,
            "run_contract_digest": manifest["run_contract_digest"],
            "launch_target": target,
            "result_prefix": (
                f"gs://{target['bucket']}/runs/{manifest['run_name']}/results"
            ),
            "work_hand_indices": list(tail_indices),
            "source_roles": list(SOURCE_ROLES),
            "logical_job_count": len(jobs),
            "jobs": list(jobs),
            "candidate_done_paths": candidate_done,
            "reference_done_paths": reference_done,
            "source_isolation_validated": True,
            "merge_executed": False,
            "production_fanout_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
        },
        run_contract=contract,
    )


def receive_jobs(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    manifest, _authorization = validate_launch(source)
    prefix = _authorized_run_prefix(manifest, project=project, bucket=bucket)
    validate_receive_launch_chain(run_dir=source, manifest=manifest)
    resume_chain = validate_receive_resume_chain(run_dir=source, manifest=manifest)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6d v2 receive destination is immutable")
    stage = destination.with_name(f".{destination.name}.{os.getpid()}.receiving")
    if stage.exists():
        raise FileExistsError(f"stale Step 6d v2 receive staging exists: {stage}")
    stage.mkdir(parents=True)
    try:
        _write_once(
            stage / "package_manifest.json",
            (source / PACKAGE_MANIFEST_NAME).read_bytes(),
            raw=True,
        )
        _write_once(
            stage / AUTHORIZATION_NAME,
            (source / AUTHORIZATION_NAME).read_bytes(),
            raw=True,
        )
        _write_once(
            stage / PACKAGE_READY_NAME,
            (source / PACKAGE_READY_NAME).read_bytes(),
            raw=True,
        )
        for evidence_name in (LAUNCH_CLAIM_NAME, LAUNCH_RESULT_NAME):
            _write_once(
                stage / evidence_name,
                (source / evidence_name).read_bytes(),
                raw=True,
            )
        if resume_chain is not None:
            for evidence_name in (RESUME_CLAIM_NAME, RESUME_RESULT_NAME):
                _write_once(
                    stage / evidence_name,
                    (source / evidence_name).read_bytes(),
                    raw=True,
                )
        for record in manifest["job_manifests"]:
            _write_once(
                stage / record["path"],
                (source / record["path"]).read_bytes(),
                raw=True,
            )

        def download(record: Mapping[str, Any]) -> dict[str, Any]:
            identifier = str(record["job_id"])
            job_dir = stage / "jobs" / identifier
            job_dir.mkdir(parents=True)
            _run(
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
                job_dir=job_dir,
                record=record,
                package_root=stage,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(download, manifest["job_manifests"]))
        if [row["job_id"] for row in jobs] != list(
            authorized_job_ids(manifest["run_contract"])
        ):
            raise ValueError("Step 6d v2 received job ordering changed")
        if any(
            row["run_contract_digest"] != manifest["run_contract_digest"]
            for row in jobs
        ):
            raise ValueError("Step 6d v2 received shared contract changed")
        receipt = build_receive_receipt(
            manifest=manifest,
            jobs=jobs,
            package_manifest_sha256=sha256_file(source / PACKAGE_MANIFEST_NAME),
            launch_authorization_sha256=sha256_file(source / AUTHORIZATION_NAME),
            launch_claim_sha256=sha256_file(source / LAUNCH_CLAIM_NAME),
            launch_result_sha256=sha256_file(source / LAUNCH_RESULT_NAME),
            resume_claim_sha256=(
                sha256_file(source / RESUME_CLAIM_NAME)
                if resume_chain is not None
                else None
            ),
            resume_result_sha256=(
                sha256_file(source / RESUME_RESULT_NAME)
                if resume_chain is not None
                else None
            ),
        )
        _write_once(stage / "receive_receipt.json", receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    package = commands.add_parser("package")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument("--candidate-library", type=Path, required=True)
    package.add_argument("--candidate-sha256", required=True)
    package.add_argument("--reference-library", type=Path)
    package.add_argument("--reference-sha256", default=REFERENCE_NATIVE_LIBRARY_SHA256)
    package.add_argument("--feature-encoder", type=Path)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--startup-script", type=Path)
    package.add_argument(
        "--contract-variant",
        choices=_contract_variant_choices(),
        default=_runner_v2.CANDIDATE01_VARIANT,
    )

    validate = commands.add_parser("validate-package")
    validate.add_argument("--run-dir", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    validate_authorization = commands.add_parser("validate-launch")
    validate_authorization.add_argument("--run-dir", type=Path, required=True)

    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--jobs", required=True)
    launch.add_argument("--project", default=DEFAULT_PROJECT)
    launch.add_argument("--bucket", default=DEFAULT_BUCKET)
    launch.add_argument("--zone", action="append", choices=DEFAULT_ZONES)

    resume = commands.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--jobs", required=True)
    resume.add_argument("--project", default=DEFAULT_PROJECT)
    resume.add_argument("--bucket", default=DEFAULT_BUCKET)
    resume.add_argument("--zone", action="append", choices=DEFAULT_ZONES)

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
        result: Any = package_step6d_v2(
            output_dir=args.output_dir,
            run_name=args.run_name,
            candidate_library=args.candidate_library,
            candidate_sha256=args.candidate_sha256,
            reference_library=args.reference_library,
            reference_sha256=args.reference_sha256,
            feature_encoder=args.feature_encoder,
            repository_root=args.repository_root,
            startup_script=args.startup_script,
            contract_variant=args.contract_variant,
        )
    elif args.command == "validate-package":
        result = validate_package(args.run_dir)
    elif args.command == "authorize":
        result = authorize_launch(args.run_dir)
    elif args.command == "validate-launch":
        manifest, authorization = validate_launch(args.run_dir)
        result = {
            "schema": LAUNCH_AUTHORIZATION_SCHEMA,
            "status": "valid",
            "run_name": manifest["run_name"],
            "run_contract_digest": manifest["run_contract_digest"],
            "authorization_sha256": sha256_file(
                Path(args.run_dir) / AUTHORIZATION_NAME
            ),
            "logical_job_count": authorization["logical_job_count"],
        }
    elif args.command == "launch":
        launch_manifest = validate_package(args.run_dir)
        result = launch_jobs(
            run_dir=args.run_dir,
            jobs=_parse_jobs(
                args.jobs,
                launch_manifest["run_contract"],
            ),
            project=args.project,
            bucket=args.bucket,
            zones=tuple(args.zone) if args.zone else DEFAULT_ZONES,
        )
    elif args.command == "resume":
        resume_manifest = validate_package(args.run_dir)
        result = resume_jobs(
            run_dir=args.run_dir,
            jobs=_parse_jobs(
                args.jobs,
                resume_manifest["run_contract"],
            ),
            project=args.project,
            bucket=args.bucket,
            zones=tuple(args.zone) if args.zone else DEFAULT_ZONES,
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
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_JOB_COUNT",
    "COST_GUARD_SCHEMA",
    "HEARTBEAT_SCHEMA",
    "JOB_MANIFEST_SCHEMA",
    "MAX_LOGICAL_JOBS",
    "PROCESS_COUNT",
    "RAYON_THREADS_PER_PROCESS",
    "SPOT_PACKAGE_SCHEMA",
    "SOURCE_ROLES",
    "TAIL_HAND_INDICES",
    "authorize_launch",
    "authorized_job_ids",
    "build_cost_guard",
    "build_job_manifest",
    "build_job_manifests",
    "build_launch_target",
    "build_receive_receipt",
    "canonical_sha256",
    "cloud_status",
    "job_id",
    "launch_jobs",
    "main",
    "package_step6d_v2",
    "preflight_launch",
    "preflight_resume",
    "receive_jobs",
    "resume_jobs",
    "validate_job_manifest",
    "validate_cost_guard",
    "validate_launch",
    "validate_package",
    "validate_receive_receipt",
]
