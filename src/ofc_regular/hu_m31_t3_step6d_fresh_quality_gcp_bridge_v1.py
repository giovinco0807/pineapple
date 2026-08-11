"""Production-safe lifecycle bridge for the M3.1 fresh-quality transport.

The accepted full100 wave-v2 controller is intentionally *not* fed a forged
full100 plan.  Its scientific plan is fixed to twenty candidate/reference jobs
in 8+8+4 waves, while fresh quality has fifteen independent jobs in 8+7 waves.
This module reuses the proven lifecycle invariants that are genuinely common:

* one immutable run identity and content tree;
* at most eight ``c4-standard-16`` workers at once;
* one VM per job, two named attempts maximum, and earliest-wave ordering;
* append-only full100-v2 controller journals;
* result publication with create-only object generations;
* exact-owned cleanup, absence readback, and worker-IAM removal before receive;
* source-replayed result validation and a write-once quality gate.

There is deliberately no cloud client in this module.  A provider adapter may
execute a wave request, but the adapter's evidence is accepted only through
``apply_lifecycle_receipt`` and ``receive_ready_jobs``.  This keeps credentials
out of immutable artifacts and makes planning/receiving independently
testable.  No function changes ``current`` or any named AI profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate
from . import hu_m31_t3_step6d_fresh_quality_transport_v1 as transport
from . import hu_m31_t3_step6d_fresh_quality_v1 as quality
from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as full100_wave


PLAN_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_plan_v1"
LEDGER_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_attempt_ledger_v1"
RESUME_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_resume_plan_v1"
WAVE_REQUEST_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_wave_request_v1"
LIFECYCLE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_lifecycle_receipt_v1"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_receive_receipt_v1"
FINAL_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_final_receipt_v1"

RUN_NAME_PREFIX = "regular-hu-m31-t3-fqv1-"
MACHINE_TYPE = full100_wave.MACHINE_TYPE
VCPUS_PER_VM = full100_wave.VCPUS_PER_VM
MAX_CONCURRENT_VMS = full100_wave.MAX_CONCURRENT_VMS
MAX_ATTEMPTS_PER_JOB = 2
ATTEMPT_IDS = ("a00", "a01")
WAVE_JOB_IDS = transport.WAVE_JOB_IDS
WAVE_JOB_COUNTS = transport.WAVE_JOB_COUNTS

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SALT = re.compile(r"^[0-9a-f]{32}$")
_RUN = re.compile(r"^[a-z][a-z0-9-]{2,62}$")
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_ZONE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
_GENERATION = re.compile(r"^[1-9][0-9]*$")
_UTC = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)

_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_identity_sha256",
        "source_paths",
        "source_file_sha256",
        "launch_manifest",
        "launch_manifest_sha256",
        "performance_authorization",
        "project",
        "region",
        "zone",
        "bucket",
        "machine_contract",
        "waves",
        "jobs",
        "artifact_contract",
        "reused_safety_contract",
        "cloud_launch_authorized",
        "cloud_execution_started",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
        "plan_sha256",
    }
)
_PLAN_SOURCE_KEYS = frozenset(
    {"staging_directory", "launch_manifest", "performance_receipt", "profile_registry"}
)
_FILE_SHA_KEYS = frozenset(
    {"launch_manifest", "performance_receipt", "profile_registry"}
)
_MACHINE_KEYS = frozenset(
    {
        "machine_type",
        "vcpus_per_vm",
        "max_concurrent_vms",
        "processes_per_vm",
        "rayon_threads",
        "provisioning_model",
    }
)
_WAVE_KEYS = frozenset(
    {
        "wave_index",
        "job_ids",
        "job_count",
        "prerequisite_wave_indices",
        "max_vm_count",
        "requires_prior_wave_accepted",
        "requires_owned_compute_absent",
        "requires_worker_iam_removed_before_receive",
    }
)
_JOB_KEYS = frozenset(
    {
        "job_id",
        "phase",
        "wave_index",
        "ordinal",
        "job_manifest_sha256",
        "result_path",
        "attempts",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "attempt_id",
        "instance_id",
        "artifact_prefix",
        "result_object",
        "done_object",
        "task_object_prefix",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "content_prefix",
        "claim_prefix",
        "result_prefix",
        "create_only",
        "object_generation_required",
        "done_published_last",
        "task_objects_required_for_source_replay",
    }
)
_REUSE_KEYS = frozenset(
    {
        "controller_event_schema",
        "controller_version",
        "journal_hash_chain_required",
        "two_attempt_limit",
        "earliest_incomplete_wave_only",
        "exact_owned_cleanup_required",
        "absence_readback_required",
        "worker_iam_removal_required",
        "full100_science_plan_reused",
        "reason_full100_science_plan_not_reused",
    }
)

_LEDGER_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "run_name",
        "execution_identity_sha256",
        "transitions",
        "accepted_jobs",
        "ledger_sha256",
    }
)
_TRANSITION_KEYS = frozenset(
    {
        "sequence",
        "previous_transition_sha256",
        "wave_index",
        "resume_sha256",
        "lifecycle_receipt_sha256",
        "receive_receipt_sha256",
        "attempt_rows",
        "owned_compute_absent",
        "worker_iam_removed",
        "transition_sha256",
    }
)
_ATTEMPT_ROW_KEYS = frozenset(
    {
        "job_id",
        "attempt_id",
        "instance_id",
        "status",
        "result_object",
        "done_object",
        "task_objects",
    }
)
_OBJECT_KEYS = frozenset({"name", "generation", "sha256", "bytes"})
_ACCEPTED_KEYS = frozenset(
    {
        "job_id",
        "attempt_id",
        "result_path",
        "result_sha256",
        "done_sha256",
        "task_record_aggregate_sha256",
        "receive_receipt_sha256",
    }
)
_RESUME_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "ledger_sha256",
        "run_name",
        "execution_identity_sha256",
        "resume_wave_index",
        "selected_attempts",
        "all_jobs_accepted",
        "attempts_exhausted",
        "owned_compute_quiescent",
        "cloud_launch_authorized",
        "resume_sha256",
    }
)
_SELECTED_KEYS = frozenset(
    {"job_id", "phase", "wave_index", "attempt_id", "instance_id", "artifact_prefix"}
)
_WAVE_REQUEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "ledger_sha256",
        "resume_sha256",
        "run_name",
        "execution_identity_sha256",
        "project",
        "region",
        "zone",
        "bucket",
        "machine_contract",
        "content_prefix",
        "selected_attempts",
        "expected_result_objects",
        "required_lifecycle",
        "cloud_launch_authorized",
        "current_profile_changed",
        "request_sha256",
    }
)
_LIFECYCLE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "ledger_sha256",
        "resume_sha256",
        "request_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "attempt_rows",
        "launch_create_only",
        "one_vm_per_job",
        "unlisted_vm_created",
        "wildcard_delete_used",
        "unrelated_resource_touched",
        "owned_compute_absent",
        "worker_iam_removed",
        "receiver_handoff_ready",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_RECEIVE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "input_ledger_sha256",
        "resume_sha256",
        "lifecycle_receipt_sha256",
        "wave_index",
        "accepted_records",
        "accepted_job_count",
        "failed_job_count",
        "output_ledger",
        "output_ledger_sha256",
        "source_replay_complete",
        "owned_compute_absent",
        "worker_iam_removed",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_FINAL_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "plan_sha256",
        "ledger_sha256",
        "accepted_job_count",
        "accepted_job_ids",
        "lifecycle_transition_sha256s",
        "quality_merge",
        "quality_merge_sha256",
        "quality_gate",
        "quality_gate_sha256",
        "quality_pilot_passed",
        "data_pilot_25_paired_authorized",
        "full_9000_paired_fanout_authorized",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "receipt_sha256",
    }
)


class RemoteObjectReader(Protocol):
    """Read an exact GCS object without list or mutation authority."""

    def read_object(self, *, bucket: str, object_name: str) -> Mapping[str, Any]: ...


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
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"required file is missing or unsafe: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields changed: missing={missing}, extra={extra}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is not a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} is not a safe relative path")
    return value


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw not in (canonical_bytes(value), canonical_bytes(value) + b"\n"):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _file_from_record(
    staging: Path, record: Mapping[str, Any], label: str
) -> Path:
    _exact_keys(record, frozenset({"path", "sha256", "bytes"}), label)
    relative = _safe_relative(record.get("path"), f"{label} path")
    target = staging.joinpath(*PurePosixPath(relative).parts).resolve()
    if (
        staging not in target.parents
        or target.is_symlink()
        or not target.is_file()
        or sha256_file(target) != _sha(record.get("sha256"), f"{label} SHA-256")
        or target.stat().st_size != record.get("bytes")
    ):
        raise ValueError(f"{label} file binding changed")
    return target


def _instance_id(identity: str, wave_index: int, ordinal: int, attempt_id: str) -> str:
    # 31 characters; safely below GCE's 63-character maximum.
    return f"fq-{identity[:10]}-w{wave_index}-j{ordinal:02d}-{attempt_id}"


def _attempt(
    *, run_name: str, identity: str, wave_index: int, ordinal: int, job_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    prefix = f"runs/{run_name}/fresh-quality/jobs/{job_id}/attempts/{attempt_id}"
    return {
        "attempt_id": attempt_id,
        "instance_id": _instance_id(identity, wave_index, ordinal, attempt_id),
        "artifact_prefix": prefix,
        "result_object": f"{prefix}/result.json",
        "done_object": f"{prefix}/DONE.json",
        "task_object_prefix": f"{prefix}/tasks/",
    }


def _plan_digest(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("plan_sha256", None)
    return canonical_sha256(payload)


def build_gcp_plan(
    *,
    launch_manifest_path: str | Path,
    staging_directory: str | Path,
    performance_receipt_path: str | Path,
    profile_registry_path: str | Path,
    run_name: str,
    identity_salt: str,
    project: str,
    region: str,
    zone: str,
    bucket: str,
) -> dict[str, Any]:
    """Replay local science and freeze an 8+7 cloud lifecycle plan."""

    if (
        not isinstance(run_name, str)
        or _RUN.fullmatch(run_name) is None
        or not run_name.startswith(RUN_NAME_PREFIX)
    ):
        raise ValueError("fresh-quality run name is invalid")
    if not isinstance(identity_salt, str) or _SALT.fullmatch(identity_salt) is None:
        raise ValueError("identity salt must be 16 random bytes rendered as lowercase hex")
    if not isinstance(project, str) or _PROJECT.fullmatch(project) is None:
        raise ValueError("GCP project changed")
    if not isinstance(zone, str) or _ZONE.fullmatch(zone) is None:
        raise ValueError("GCP zone changed")
    if not isinstance(region, str) or not zone.startswith(f"{region}-"):
        raise ValueError("GCP region/zone binding changed")
    if not isinstance(bucket, str) or _BUCKET.fullmatch(bucket) is None:
        raise ValueError("GCS bucket changed")

    staging = Path(staging_directory).resolve()
    launch_path = Path(launch_manifest_path).resolve()
    performance_path = Path(performance_receipt_path).resolve()
    profile_path = Path(profile_registry_path).resolve()
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("staging directory is missing or unsafe")
    launch = transport.validate_local_launch_manifest(
        _read_canonical(launch_path, "fresh-quality launch manifest"),
        staging_directory=staging,
    )
    if launch["run_name"] != run_name:
        raise ValueError("launch manifest run name changed")
    profile_sha = sha256_file(profile_path)
    if (
        profile_sha != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or launch["profile_registry_sha256"] != profile_sha
    ):
        raise PermissionError("current profile registry differs from the frozen pin")
    package_path = _file_from_record(
        staging, launch["quality_package"], "quality package"
    )
    with tempfile.TemporaryDirectory(prefix="ofc-fq-plan-replay-") as temporary:
        package = transport.extract_and_validate_package(
            archive_path=package_path,
            expected_archive_sha256=launch["quality_package"]["sha256"],
            extraction_directory=Path(temporary) / "package",
        )
        quality.validate_plan_authorization(
            package["plan"], performance_receipt_path=performance_path
        )
        performance_authorization = deepcopy(
            package["plan"]["authorizing_performance_receipt"]
        )

    identity_basis = {
        "run_name": run_name,
        "launch_manifest_sha256": canonical_sha256(launch),
        "performance_receipt_sha256": sha256_file(performance_path),
        "profile_registry_sha256": profile_sha,
        "project": project,
        "region": region,
        "zone": zone,
        "bucket": bucket,
        "identity_salt": identity_salt,
    }
    identity = canonical_sha256(identity_basis)
    launch_jobs = {row["job_id"]: row for row in launch["jobs"]}
    jobs: list[dict[str, Any]] = []
    ordinal = 0
    for wave_index, job_ids in enumerate(WAVE_JOB_IDS):
        for job_id in job_ids:
            row = launch_jobs[job_id]
            jobs.append(
                {
                    "job_id": job_id,
                    "phase": row["phase"],
                    "wave_index": wave_index,
                    "ordinal": ordinal,
                    "job_manifest_sha256": row["job_manifest_sha256"],
                    "result_path": row["result_path"],
                    "attempts": [
                        _attempt(
                            run_name=run_name,
                            identity=identity,
                            wave_index=wave_index,
                            ordinal=ordinal,
                            job_id=job_id,
                            attempt_id=attempt_id,
                        )
                        for attempt_id in ATTEMPT_IDS
                    ],
                }
            )
            ordinal += 1
    source_paths = {
        "staging_directory": str(staging),
        "launch_manifest": str(launch_path),
        "performance_receipt": str(performance_path),
        "profile_registry": str(profile_path),
    }
    source_sha = {
        "launch_manifest": sha256_file(launch_path),
        "performance_receipt": sha256_file(performance_path),
        "profile_registry": profile_sha,
    }
    core = {
        "schema": PLAN_SCHEMA,
        "status": "immutable_quality_cloud_plan_ready_not_authorized",
        "run_name": run_name,
        "execution_identity_sha256": identity,
        "source_paths": source_paths,
        "source_file_sha256": source_sha,
        "launch_manifest": deepcopy(launch),
        "launch_manifest_sha256": canonical_sha256(launch),
        "performance_authorization": performance_authorization,
        "project": project,
        "region": region,
        "zone": zone,
        "bucket": bucket,
        "machine_contract": {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": VCPUS_PER_VM,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "processes_per_vm": 1,
            "rayon_threads": 16,
            "provisioning_model": "SPOT",
        },
        "waves": [
            {
                "wave_index": index,
                "job_ids": list(ids),
                "job_count": len(ids),
                "prerequisite_wave_indices": list(range(index)),
                "max_vm_count": len(ids),
                "requires_prior_wave_accepted": index > 0,
                "requires_owned_compute_absent": True,
                "requires_worker_iam_removed_before_receive": True,
            }
            for index, ids in enumerate(WAVE_JOB_IDS)
        ],
        "jobs": jobs,
        "artifact_contract": {
            "content_prefix": f"runs/{run_name}/fresh-quality/content/",
            "claim_prefix": f"runs/{run_name}/fresh-quality/control/claims/",
            "result_prefix": f"runs/{run_name}/fresh-quality/jobs/",
            "create_only": True,
            "object_generation_required": True,
            "done_published_last": True,
            "task_objects_required_for_source_replay": True,
        },
        "reused_safety_contract": {
            "controller_event_schema": controller_v2.EVENT_SCHEMA,
            "controller_version": controller_v2.CONTROLLER_VERSION,
            "journal_hash_chain_required": True,
            "two_attempt_limit": True,
            "earliest_incomplete_wave_only": True,
            "exact_owned_cleanup_required": True,
            "absence_readback_required": True,
            "worker_iam_removal_required": True,
            "full100_science_plan_reused": False,
            "reason_full100_science_plan_not_reused": (
                "full100_is_20_candidate_reference_jobs_8_8_4_quality_is_15_jobs_8_7"
            ),
        },
        "cloud_launch_authorized": False,
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    plan = {**core, "plan_sha256": canonical_sha256(core)}
    return validate_gcp_plan(plan, replay_sources=True)


def validate_gcp_plan(
    value: Mapping[str, Any], *, replay_sources: bool
) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    _exact_keys(plan, _PLAN_KEYS, "fresh-quality GCP plan")
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise ValueError("fresh-quality GCP plan digest changed")
    _sha(plan.get("execution_identity_sha256"), "execution identity")
    _exact_keys(plan.get("source_paths", {}), _PLAN_SOURCE_KEYS, "plan source paths")
    _exact_keys(
        plan.get("source_file_sha256", {}), _FILE_SHA_KEYS, "plan source hashes"
    )
    _exact_keys(plan.get("machine_contract", {}), _MACHINE_KEYS, "machine contract")
    _exact_keys(plan.get("artifact_contract", {}), _ARTIFACT_KEYS, "artifact contract")
    _exact_keys(
        plan.get("reused_safety_contract", {}), _REUSE_KEYS, "reused safety contract"
    )
    launch = plan.get("launch_manifest")
    jobs = plan.get("jobs")
    waves = plan.get("waves")
    if not isinstance(launch, Mapping) or not isinstance(jobs, list) or not isinstance(waves, list):
        raise ValueError("fresh-quality GCP plan collections are missing")
    for wave in waves:
        if not isinstance(wave, Mapping):
            raise ValueError("fresh-quality GCP wave is not an object")
        _exact_keys(wave, _WAVE_KEYS, "fresh-quality GCP wave")
    launch_by_id = {row["job_id"]: row for row in launch["jobs"]}
    expected_job_ids = [job_id for ids in WAVE_JOB_IDS for job_id in ids]
    seen_instances: set[str] = set()
    for expected_ordinal, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            raise ValueError("fresh-quality GCP job is not an object")
        _exact_keys(job, _JOB_KEYS, "fresh-quality GCP job")
        attempts = job.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != MAX_ATTEMPTS_PER_JOB:
            raise ValueError("fresh-quality job attempt coverage changed")
        launch_row = launch_by_id.get(job.get("job_id"))
        expected_wave = 0 if expected_ordinal < WAVE_JOB_COUNTS[0] else 1
        if (
            job.get("ordinal") != expected_ordinal
            or job.get("job_id") != expected_job_ids[expected_ordinal]
            or job.get("wave_index") != expected_wave
            or launch_row is None
            or job.get("phase") != launch_row["phase"]
            or job.get("job_manifest_sha256") != launch_row["job_manifest_sha256"]
            or job.get("result_path") != launch_row["result_path"]
        ):
            raise ValueError("fresh-quality GCP job mapping changed")
        for expected_attempt, attempt in zip(ATTEMPT_IDS, attempts, strict=True):
            if not isinstance(attempt, Mapping):
                raise ValueError("fresh-quality GCP attempt is not an object")
            _exact_keys(attempt, _ATTEMPT_KEYS, "fresh-quality GCP attempt")
            if (
                attempt.get("attempt_id") != expected_attempt
                or attempt.get("instance_id") in seen_instances
                or attempt.get("result_object")
                != f"{attempt.get('artifact_prefix')}/result.json"
                or attempt.get("done_object")
                != f"{attempt.get('artifact_prefix')}/DONE.json"
                or attempt.get("task_object_prefix")
                != f"{attempt.get('artifact_prefix')}/tasks/"
            ):
                raise ValueError("fresh-quality GCP attempt identity changed")
            seen_instances.add(str(attempt["instance_id"]))
    machine = plan["machine_contract"]
    artifact = plan["artifact_contract"]
    reused = plan["reused_safety_contract"]
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "immutable_quality_cloud_plan_ready_not_authorized"
        or not str(plan.get("run_name", "")).startswith(RUN_NAME_PREFIX)
        or plan.get("launch_manifest_sha256") != canonical_sha256(launch)
        or launch.get("run_name") != plan.get("run_name")
        or [wave["job_ids"] for wave in waves] != [list(ids) for ids in WAVE_JOB_IDS]
        or [wave["job_count"] for wave in waves] != list(WAVE_JOB_COUNTS)
        or any(wave["max_vm_count"] > MAX_CONCURRENT_VMS for wave in waves)
        or machine
        != {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": VCPUS_PER_VM,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "processes_per_vm": 1,
            "rayon_threads": 16,
            "provisioning_model": "SPOT",
        }
        or artifact.get("create_only") is not True
        or artifact.get("object_generation_required") is not True
        or artifact.get("done_published_last") is not True
        or artifact.get("task_objects_required_for_source_replay") is not True
        or reused.get("controller_event_schema") != controller_v2.EVENT_SCHEMA
        or reused.get("controller_version") != controller_v2.CONTROLLER_VERSION
        or reused.get("full100_science_plan_reused") is not False
        or any(
            plan.get(field) is not False
            for field in (
                "cloud_launch_authorized",
                "cloud_execution_started",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("fresh-quality GCP plan safety boundary changed")
    authorization = plan.get("performance_authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("schema") != quality.PERFORMANCE_RECEIPT_SCHEMA
        or authorization.get("status") != "qualified"
        or authorization.get("decision") != quality.QUALIFIED_DECISION
        or _SHA.fullmatch(str(authorization.get("receipt_sha256", ""))) is None
        or authorization.get("performance_lock_qualified") is not True
        or authorization.get("quality_pilot_authorized") is not True
        or authorization.get("performance_lock_finalized") is not True
        or authorization.get("one_shot_lock_consumed") is not True
        or authorization.get("current_profile_changed") is not False
    ):
        raise PermissionError("qualified v4 performance receipt is not bound")
    if replay_sources is not True:
        raise PermissionError("fresh-quality cloud plan validation requires source replay")
    paths = {key: Path(str(raw)).resolve() for key, raw in plan["source_paths"].items()}
    if paths["staging_directory"].is_symlink() or not paths["staging_directory"].is_dir():
        raise ValueError("staging source path changed")
    for field in _FILE_SHA_KEYS:
        source_path = paths[field]
        if sha256_file(source_path) != plan["source_file_sha256"][field]:
            raise ValueError(f"fresh-quality source changed: {field}")
    replayed_launch = transport.validate_local_launch_manifest(
        _read_canonical(paths["launch_manifest"], "fresh-quality launch manifest"),
        staging_directory=paths["staging_directory"],
    )
    if replayed_launch != launch:
        raise ValueError("embedded fresh-quality launch differs from source replay")
    profile_sha = sha256_file(paths["profile_registry"])
    if (
        profile_sha != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or profile_sha != launch["profile_registry_sha256"]
    ):
        raise PermissionError("profile registry changed after cloud plan freeze")
    package_path = _file_from_record(
        paths["staging_directory"], launch["quality_package"], "quality package"
    )
    with tempfile.TemporaryDirectory(prefix="ofc-fq-plan-validate-") as temporary:
        package = transport.extract_and_validate_package(
            archive_path=package_path,
            expected_archive_sha256=launch["quality_package"]["sha256"],
            extraction_directory=Path(temporary) / "package",
        )
        quality.validate_plan_authorization(
            package["plan"], performance_receipt_path=paths["performance_receipt"]
        )
        if package["plan"]["authorizing_performance_receipt"] != authorization:
            raise PermissionError("performance authorization summary changed")
    return plan


def write_gcp_plan(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    plan = build_gcp_plan(**kwargs)
    _write_once(path, plan)
    stored = _read_canonical(path, "stored fresh-quality GCP plan")
    if validate_gcp_plan(stored, replay_sources=True) != plan:
        raise ValueError("stored fresh-quality GCP plan differs from source replay")
    return plan


def _ledger_digest(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("ledger_sha256", None)
    return canonical_sha256(payload)


def initial_attempt_ledger(plan: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_gcp_plan(plan, replay_sources=True)
    core = {
        "schema": LEDGER_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "run_name": checked["run_name"],
        "execution_identity_sha256": checked["execution_identity_sha256"],
        "transitions": [],
        "accepted_jobs": [],
    }
    return {**core, "ledger_sha256": canonical_sha256(core)}


def validate_attempt_ledger(
    plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    ledger = deepcopy(dict(value))
    _exact_keys(ledger, _LEDGER_KEYS, "fresh-quality attempt ledger")
    if ledger.get("ledger_sha256") != _ledger_digest(ledger):
        raise ValueError("fresh-quality attempt ledger digest changed")
    transitions = ledger.get("transitions")
    accepted = ledger.get("accepted_jobs")
    if not isinstance(transitions, list) or not isinstance(accepted, list):
        raise ValueError("fresh-quality attempt ledger collections are missing")
    previous: str | None = None
    seen_attempts: set[tuple[str, str]] = set()
    jobs_by_id = {row["job_id"]: row for row in checked_plan["jobs"]}
    attempt_counts: dict[str, int] = {}
    accepted_transition_jobs: dict[str, str] = {}
    receive_identities: set[str] = set()
    for sequence, transition in enumerate(transitions, start=1):
        if not isinstance(transition, Mapping):
            raise ValueError("fresh-quality transition is not an object")
        _exact_keys(transition, _TRANSITION_KEYS, "fresh-quality transition")
        payload = deepcopy(dict(transition))
        digest = payload.pop("transition_sha256", None)
        rows = transition.get("attempt_rows")
        if (
            digest != canonical_sha256(payload)
            or transition.get("sequence") != sequence
            or transition.get("previous_transition_sha256") != previous
            or not isinstance(rows, list)
            or transition.get("owned_compute_absent") is not True
            or transition.get("worker_iam_removed") is not True
        ):
            raise ValueError("fresh-quality transition chain changed")
        _sha(transition.get("resume_sha256"), "transition resume")
        _sha(transition.get("lifecycle_receipt_sha256"), "transition lifecycle")
        receive_identity = _sha(
            transition.get("receive_receipt_sha256"), "transition receive"
        )
        receive_identities.add(receive_identity)
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("fresh-quality attempt row is not an object")
            _exact_keys(row, _ATTEMPT_ROW_KEYS, "fresh-quality attempt row")
            key = (str(row.get("job_id")), str(row.get("attempt_id")))
            job = jobs_by_id.get(key[0])
            attempt_index = attempt_counts.get(key[0], 0)
            expected_attempt = (
                None
                if job is None or attempt_index >= len(job["attempts"])
                else job["attempts"][attempt_index]
            )
            if (
                key in seen_attempts
                or job is None
                or expected_attempt is None
                or key[1] != expected_attempt["attempt_id"]
                or row.get("instance_id") != expected_attempt["instance_id"]
                or row.get("status") not in {"failed", "ready", "accepted"}
            ):
                raise ValueError("fresh-quality attempt history is duplicated or invalid")
            if row["status"] == "failed":
                if (
                    row.get("result_object") is not None
                    or row.get("done_object") is not None
                    or row.get("task_objects") != []
                ):
                    raise ValueError("failed fresh-quality attempt has artifacts")
            else:
                _validate_object_record(row.get("result_object", {}), "ledger result")
                _validate_object_record(row.get("done_object", {}), "ledger DONE")
                tasks = row.get("task_objects")
                if not isinstance(tasks, list) or not tasks:
                    raise ValueError("ready/accepted ledger attempt lacks tasks")
                for task in tasks:
                    _validate_object_record(task, "ledger task")
            if row["status"] == "accepted":
                accepted_transition_jobs[key[0]] = receive_identity
            seen_attempts.add(key)
            attempt_counts[key[0]] = attempt_index + 1
        previous = str(digest)
    accepted_ids: set[str] = set()
    for row in accepted:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality accepted row is not an object")
        _exact_keys(row, _ACCEPTED_KEYS, "fresh-quality accepted row")
        job_id = str(row.get("job_id"))
        if job_id in accepted_ids:
            raise ValueError("fresh-quality accepted job is duplicated")
        result_path = Path(str(row.get("result_path")))
        if (
            result_path.is_symlink()
            or not result_path.is_file()
            or sha256_file(result_path) != row.get("result_sha256")
            or row.get("receive_receipt_sha256") not in receive_identities
            or accepted_transition_jobs.get(job_id)
            != row.get("receive_receipt_sha256")
        ):
            raise ValueError("fresh-quality accepted result evidence changed")
        _sha(row.get("done_sha256"), "accepted DONE")
        _sha(row.get("task_record_aggregate_sha256"), "accepted tasks")
        accepted_ids.add(job_id)
    expected_ids = {job["job_id"] for job in checked_plan["jobs"]}
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("plan_sha256") != checked_plan["plan_sha256"]
        or ledger.get("run_name") != checked_plan["run_name"]
        or ledger.get("execution_identity_sha256")
        != checked_plan["execution_identity_sha256"]
        or not accepted_ids.issubset(expected_ids)
        or [row["job_id"] for row in accepted]
        != [row["job_id"] for row in checked_plan["jobs"] if row["job_id"] in accepted_ids]
    ):
        raise ValueError("fresh-quality attempt ledger binding changed")
    return ledger


def _attempt_history(ledger: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for transition in ledger["transitions"]:
        for row in transition["attempt_rows"]:
            result.setdefault(row["job_id"], []).append(deepcopy(dict(row)))
    return result


def build_resume_plan(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    accepted = {row["job_id"] for row in checked_ledger["accepted_jobs"]}
    all_ids = [job["job_id"] for job in checked_plan["jobs"]]
    history = _attempt_history(checked_ledger)
    selected: list[dict[str, Any]] = []
    exhausted: list[str] = []
    resume_wave: int | None = None
    for wave in checked_plan["waves"]:
        pending = [job_id for job_id in wave["job_ids"] if job_id not in accepted]
        if not pending:
            continue
        resume_wave = int(wave["wave_index"])
        jobs = {job["job_id"]: job for job in checked_plan["jobs"]}
        for job_id in pending:
            attempts = history.get(job_id, [])
            if attempts and attempts[-1]["status"] in {"ready", "accepted"}:
                raise PermissionError(
                    "ready quality result must be received before another launch"
                )
            attempt_index = len(attempts)
            if attempt_index >= MAX_ATTEMPTS_PER_JOB:
                exhausted.append(job_id)
                continue
            job = jobs[job_id]
            attempt = job["attempts"][attempt_index]
            selected.append(
                {
                    "job_id": job_id,
                    "phase": job["phase"],
                    "wave_index": resume_wave,
                    "attempt_id": attempt["attempt_id"],
                    "instance_id": attempt["instance_id"],
                    "artifact_prefix": attempt["artifact_prefix"],
                }
            )
        break
    all_accepted = len(accepted) == len(all_ids)
    status = (
        "all_15_quality_jobs_accepted"
        if all_accepted
        else (
            "quality_attempt_limit_exhausted_no_go"
            if exhausted
            else f"wave_{resume_wave}_ready_not_authorized"
        )
    )
    core = {
        "schema": RESUME_SCHEMA,
        "status": status,
        "plan_sha256": checked_plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "run_name": checked_plan["run_name"],
        "execution_identity_sha256": checked_plan["execution_identity_sha256"],
        "resume_wave_index": resume_wave,
        "selected_attempts": selected,
        "all_jobs_accepted": all_accepted,
        "attempts_exhausted": exhausted,
        "owned_compute_quiescent": True,
        "cloud_launch_authorized": False,
    }
    resume = {**core, "resume_sha256": canonical_sha256(core)}
    return validate_resume_plan(checked_plan, checked_ledger, resume)


def validate_resume_plan(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    resume = deepcopy(dict(value))
    _exact_keys(resume, _RESUME_KEYS, "fresh-quality resume plan")
    payload = deepcopy(resume)
    digest = payload.pop("resume_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("fresh-quality resume plan digest changed")
    selected = resume.get("selected_attempts")
    if not isinstance(selected, list):
        raise ValueError("fresh-quality selected attempts are missing")
    for row in selected:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality selected attempt is not an object")
        _exact_keys(row, _SELECTED_KEYS, "fresh-quality selected attempt")
    if len(selected) > MAX_CONCURRENT_VMS:
        raise PermissionError("fresh-quality resume exceeds live C4 quota")
    # Rebuild from the already validated immutable inputs.  This makes any
    # selected-attempt or earliest-wave drift fail closed.
    # Avoid recursion: compare with a locally reconstructed core.
    accepted = {row["job_id"] for row in checked_ledger["accepted_jobs"]}
    history = _attempt_history(checked_ledger)
    expected_selected: list[dict[str, Any]] = []
    expected_wave: int | None = None
    exhausted: list[str] = []
    jobs = {job["job_id"]: job for job in checked_plan["jobs"]}
    for wave in checked_plan["waves"]:
        pending = [job_id for job_id in wave["job_ids"] if job_id not in accepted]
        if not pending:
            continue
        expected_wave = wave["wave_index"]
        for job_id in pending:
            attempts = history.get(job_id, [])
            if attempts and attempts[-1]["status"] in {"ready", "accepted"}:
                raise PermissionError("ready quality result is awaiting receive")
            index = len(attempts)
            if index >= MAX_ATTEMPTS_PER_JOB:
                exhausted.append(job_id)
                continue
            job = jobs[job_id]
            attempt = job["attempts"][index]
            expected_selected.append(
                {
                    "job_id": job_id,
                    "phase": job["phase"],
                    "wave_index": expected_wave,
                    "attempt_id": attempt["attempt_id"],
                    "instance_id": attempt["instance_id"],
                    "artifact_prefix": attempt["artifact_prefix"],
                }
            )
        break
    all_accepted = len(accepted) == len(jobs)
    expected_status = (
        "all_15_quality_jobs_accepted"
        if all_accepted
        else (
            "quality_attempt_limit_exhausted_no_go"
            if exhausted
            else f"wave_{expected_wave}_ready_not_authorized"
        )
    )
    if (
        resume.get("schema") != RESUME_SCHEMA
        or resume.get("status") != expected_status
        or resume.get("plan_sha256") != checked_plan["plan_sha256"]
        or resume.get("ledger_sha256") != checked_ledger["ledger_sha256"]
        or resume.get("run_name") != checked_plan["run_name"]
        or resume.get("execution_identity_sha256")
        != checked_plan["execution_identity_sha256"]
        or resume.get("resume_wave_index") != expected_wave
        or selected != expected_selected
        or resume.get("all_jobs_accepted") is not all_accepted
        or resume.get("attempts_exhausted") != exhausted
        or resume.get("owned_compute_quiescent") is not True
        or resume.get("cloud_launch_authorized") is not False
    ):
        raise ValueError("fresh-quality resume plan changed")
    return resume


def build_wave_request(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    checked_resume = validate_resume_plan(checked_plan, checked_ledger, resume)
    if (
        checked_resume["all_jobs_accepted"]
        or checked_resume["attempts_exhausted"]
        or not checked_resume["selected_attempts"]
    ):
        raise PermissionError("fresh-quality resume plan has no launchable wave")
    jobs = {job["job_id"]: job for job in checked_plan["jobs"]}
    expected_objects = []
    for selected in checked_resume["selected_attempts"]:
        job = jobs[selected["job_id"]]
        attempt = next(
            row for row in job["attempts"] if row["attempt_id"] == selected["attempt_id"]
        )
        expected_objects.append(
            {
                "job_id": selected["job_id"],
                "result_object": attempt["result_object"],
                "done_object": attempt["done_object"],
                "task_object_prefix": attempt["task_object_prefix"],
            }
        )
    core = {
        "schema": WAVE_REQUEST_SCHEMA,
        "status": "exact_wave_request_ready_external_authorization_required",
        "plan_sha256": checked_plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "resume_sha256": checked_resume["resume_sha256"],
        "run_name": checked_plan["run_name"],
        "execution_identity_sha256": checked_plan["execution_identity_sha256"],
        "project": checked_plan["project"],
        "region": checked_plan["region"],
        "zone": checked_plan["zone"],
        "bucket": checked_plan["bucket"],
        "machine_contract": deepcopy(checked_plan["machine_contract"]),
        "content_prefix": checked_plan["artifact_contract"]["content_prefix"],
        "selected_attempts": deepcopy(checked_resume["selected_attempts"]),
        "expected_result_objects": expected_objects,
        "required_lifecycle": {
            "one_vm_per_job": True,
            "create_only_results": True,
            "done_last": True,
            "exact_owned_cleanup": True,
            "absence_readback": True,
            "worker_iam_removed_before_receive": True,
            "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
        },
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }
    request = {**core, "request_sha256": canonical_sha256(core)}
    return validate_wave_request(checked_plan, checked_ledger, checked_resume, request)


def validate_wave_request(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    request = deepcopy(dict(value))
    _exact_keys(request, _WAVE_REQUEST_KEYS, "fresh-quality wave request")
    payload = deepcopy(request)
    digest = payload.pop("request_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("fresh-quality wave request digest changed")
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    checked_resume = validate_resume_plan(checked_plan, checked_ledger, resume)
    selected = checked_resume["selected_attempts"]
    if (
        request.get("schema") != WAVE_REQUEST_SCHEMA
        or request.get("status")
        != "exact_wave_request_ready_external_authorization_required"
        or request.get("plan_sha256") != checked_plan["plan_sha256"]
        or request.get("ledger_sha256") != checked_ledger["ledger_sha256"]
        or request.get("resume_sha256") != checked_resume["resume_sha256"]
        or request.get("run_name") != checked_plan["run_name"]
        or request.get("execution_identity_sha256")
        != checked_plan["execution_identity_sha256"]
        or request.get("project") != checked_plan["project"]
        or request.get("region") != checked_plan["region"]
        or request.get("zone") != checked_plan["zone"]
        or request.get("bucket") != checked_plan["bucket"]
        or request.get("machine_contract") != checked_plan["machine_contract"]
        or request.get("content_prefix")
        != checked_plan["artifact_contract"]["content_prefix"]
        or request.get("selected_attempts") != selected
        or request.get("cloud_launch_authorized") is not False
        or request.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality wave request binding changed")
    if len(selected) > MAX_CONCURRENT_VMS:
        raise PermissionError("fresh-quality wave request exceeds quota")
    return request


def open_controller_journal(
    directory: str | Path, *, wave_request: Mapping[str, Any], create: bool = True
) -> controller_v2.ControllerJournal:
    """Use the proven full100-v2 append-only journal with a quality context."""

    _sha(wave_request.get("request_sha256"), "wave request")
    return controller_v2.ControllerJournal(
        directory,
        context_sha256=wave_request["request_sha256"],
        create=create,
    )


def _validate_object_record(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    record = deepcopy(dict(value))
    _exact_keys(record, _OBJECT_KEYS, label)
    _safe_relative(record.get("name"), f"{label} name")
    if (
        not isinstance(record.get("generation"), str)
        or _GENERATION.fullmatch(record["generation"]) is None
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or record["bytes"] <= 0
    ):
        raise ValueError(f"{label} generation/size changed")
    _sha(record.get("sha256"), f"{label} SHA-256")
    return record


def validate_lifecycle_receipt(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    request: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    checked_resume = validate_resume_plan(checked_plan, checked_ledger, resume)
    checked_request = validate_wave_request(
        checked_plan, checked_ledger, checked_resume, request
    )
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _LIFECYCLE_KEYS, "fresh-quality lifecycle receipt")
    payload = deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("fresh-quality lifecycle receipt digest changed")
    rows = receipt.get("attempt_rows")
    if not isinstance(rows, list) or len(rows) != len(checked_resume["selected_attempts"]):
        raise ValueError("fresh-quality lifecycle result cardinality changed")
    selected = {
        (row["job_id"], row["attempt_id"]): row
        for row in checked_resume["selected_attempts"]
    }
    expected_objects = {
        row["job_id"]: row for row in checked_request["expected_result_objects"]
    }
    job_phase = {row["job_id"]: row["phase"] for row in checked_plan["jobs"]}
    seen: set[tuple[str, str]] = set()
    for row, expected_selected in zip(
        rows, checked_resume["selected_attempts"], strict=True
    ):
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality lifecycle attempt row is not an object")
        _exact_keys(row, _ATTEMPT_ROW_KEYS, "fresh-quality lifecycle attempt row")
        key = (str(row.get("job_id")), str(row.get("attempt_id")))
        selected_row = selected.get(key)
        if (
            key in seen
            or selected_row is None
            or row.get("job_id") != expected_selected["job_id"]
            or row.get("attempt_id") != expected_selected["attempt_id"]
            or row.get("instance_id") != selected_row["instance_id"]
        ):
            raise ValueError("fresh-quality lifecycle attempt escaped selected wave")
        seen.add(key)
        status = row.get("status")
        if status == "failed":
            if any(row.get(field) is not None for field in ("result_object", "done_object")) or row.get("task_objects") != []:
                raise ValueError("failed fresh-quality attempt carries result objects")
        elif status == "ready":
            result = _validate_object_record(row.get("result_object", {}), "result object")
            done = _validate_object_record(row.get("done_object", {}), "DONE object")
            tasks = row.get("task_objects")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("ready fresh-quality attempt lacks task objects")
            checked_tasks = [
                _validate_object_record(item, "task object") for item in tasks
            ]
            expected_task_count = (
                10 if job_phase[row["job_id"]] == quality.PRIMARY_PHASE else 2
            )
            expected = expected_objects[row["job_id"]]
            if (
                result["name"] != expected["result_object"]
                or done["name"] != expected["done_object"]
                or any(
                    not item["name"].startswith(expected["task_object_prefix"])
                    for item in checked_tasks
                )
                or len({item["name"] for item in checked_tasks}) != len(checked_tasks)
                or len(checked_tasks) != expected_task_count
            ):
                raise ValueError("fresh-quality lifecycle object topology changed")
            row["result_object"] = result
            row["done_object"] = done
            row["task_objects"] = checked_tasks
        else:
            raise ValueError("fresh-quality lifecycle attempt status changed")
    if (
        receipt.get("schema") != LIFECYCLE_SCHEMA
        or receipt.get("status")
        != "exact_wave_terminal_cleanup_and_receiver_handoff_ready"
        or receipt.get("plan_sha256") != checked_plan["plan_sha256"]
        or receipt.get("ledger_sha256") != checked_ledger["ledger_sha256"]
        or receipt.get("resume_sha256") != checked_resume["resume_sha256"]
        or receipt.get("request_sha256") != checked_request["request_sha256"]
        or receipt.get("run_name") != checked_plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != checked_plan["execution_identity_sha256"]
        or receipt.get("wave_index") != checked_resume["resume_wave_index"]
        or receipt.get("launch_create_only") is not True
        or receipt.get("one_vm_per_job") is not True
        or receipt.get("unlisted_vm_created") != 0
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_resource_touched") is not False
        or receipt.get("owned_compute_absent") is not True
        or receipt.get("worker_iam_removed") is not True
        or receipt.get("receiver_handoff_ready") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise PermissionError("fresh-quality lifecycle cleanup boundary changed")
    receipt["attempt_rows"] = rows
    return receipt


def _append_transition(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    receive_receipt_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    accepted_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    transitions = deepcopy(list(ledger["transitions"]))
    previous = None if not transitions else transitions[-1]["transition_sha256"]
    transition_core = {
        "sequence": len(transitions) + 1,
        "previous_transition_sha256": previous,
        "wave_index": resume["resume_wave_index"],
        "resume_sha256": resume["resume_sha256"],
        "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
        "receive_receipt_sha256": receive_receipt_sha256,
        "attempt_rows": [deepcopy(dict(row)) for row in rows],
        "owned_compute_absent": True,
        "worker_iam_removed": True,
    }
    transition = {
        **transition_core,
        "transition_sha256": canonical_sha256(transition_core),
    }
    transitions.append(transition)
    accepted = deepcopy(list(ledger["accepted_jobs"]))
    accepted.extend(deepcopy(list(accepted_records)))
    core = {
        "schema": LEDGER_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "transitions": transitions,
        "accepted_jobs": accepted,
    }
    output = {**core, "ledger_sha256": canonical_sha256(core)}
    return validate_attempt_ledger(plan, output)


def _remote_payload(
    reader: RemoteObjectReader, *, bucket: str, record: Mapping[str, Any]
) -> bytes:
    observed = dict(reader.read_object(bucket=bucket, object_name=record["name"]))
    if set(observed) != {"name", "generation", "sha256", "bytes", "payload"}:
        raise ValueError("remote object readback fields changed")
    payload = observed.get("payload")
    if not isinstance(payload, bytes):
        raise ValueError("remote object payload is not bytes")
    if (
        {key: observed[key] for key in _OBJECT_KEYS} != dict(record)
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
        or len(payload) != record["bytes"]
    ):
        raise ValueError("remote object generation/hash/size changed")
    return payload


def receive_ready_jobs(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    request: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    reader: RemoteObjectReader,
    output_directory: str | Path,
    accepted_results_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Receive one terminal wave after cleanup and source-replay every ready job."""

    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    checked_resume = validate_resume_plan(checked_plan, checked_ledger, resume)
    checked_request = validate_wave_request(
        checked_plan, checked_ledger, checked_resume, request
    )
    lifecycle = validate_lifecycle_receipt(
        checked_plan,
        checked_ledger,
        checked_resume,
        checked_request,
        lifecycle_receipt,
    )
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("fresh-quality receiver output is create-only")
    output.mkdir(parents=True, exist_ok=False)
    accepted_root = (
        output / "accepted-results"
        if accepted_results_directory is None
        else Path(accepted_results_directory).resolve()
    )
    if accepted_root.exists():
        if accepted_root.is_symlink() or not accepted_root.is_dir():
            raise ValueError("accepted fresh-quality result root is unsafe")
    else:
        accepted_root.mkdir(parents=True, exist_ok=False)
    staging = Path(checked_plan["source_paths"]["staging_directory"])
    package_archive = _file_from_record(
        staging, checked_plan["launch_manifest"]["quality_package"], "quality package"
    )
    package = transport.extract_and_validate_package(
        archive_path=package_archive,
        expected_archive_sha256=checked_plan["launch_manifest"]["quality_package"][
            "sha256"
        ],
        extraction_directory=output / "package",
    )
    jobs = {job["job_id"]: job for job in package["jobs"]}
    accepted_records: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    receipt_sha_placeholder = "0" * 64
    try:
        for row in lifecycle["attempt_rows"]:
            transition = deepcopy(dict(row))
            if row["status"] == "failed":
                transition_rows.append(transition)
                continue
            job_id = row["job_id"]
            job_output = output / "jobs" / job_id
            job_output.mkdir(parents=True, exist_ok=False)
            result_payload = _remote_payload(
                reader, bucket=checked_plan["bucket"], record=row["result_object"]
            )
            done_payload = _remote_payload(
                reader, bucket=checked_plan["bucket"], record=row["done_object"]
            )
            result_value = json.loads(result_payload.decode("ascii"))
            done_value = json.loads(done_payload.decode("ascii"))
            if (
                result_payload != canonical_bytes(result_value)
                or done_payload != canonical_bytes(done_value)
                or not isinstance(result_value, dict)
                or not isinstance(done_value, dict)
            ):
                raise ValueError("remote fresh-quality JSON is not canonical")
            result_path = job_output.joinpath(
                *PurePosixPath(jobs[job_id]["result_path"]).parts
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_bytes(result_payload)
            for task_record in row["task_objects"]:
                task_payload = _remote_payload(
                    reader, bucket=checked_plan["bucket"], record=task_record
                )
                name = PurePosixPath(task_record["name"]).name
                if re.fullmatch(r"root_[0-9]{3}\.json", name) is None:
                    raise ValueError("remote task object name changed")
                task_path = job_output / "tasks" / name
                task_path.parent.mkdir(parents=True, exist_ok=True)
                task_path.write_bytes(task_payload)
            (job_output / "DONE.json").write_bytes(done_payload)
            selected_job = transport.validate_selected_job(
                package=package,
                job_id=job_id,
                expected_job_manifest_sha256=next(
                    item["job_manifest_sha256"]
                    for item in checked_plan["jobs"]
                    if item["job_id"] == job_id
                ),
            )
            lookup, _root_artifacts = transport._root_lookup(
                Path(package["package_root"]), selected_job, package["plan"]
            )
            validated_done = transport._validate_done(
                done_value,
                output_root=job_output,
                package=package,
                job=selected_job,
                library_sha256=checked_plan["launch_manifest"]["candidate_library"][
                    "sha256"
                ],
                lookup=lookup,
            )
            accepted_result = accepted_root / f"{job_id}.json"
            if accepted_result.exists():
                raise FileExistsError("accepted fresh-quality result already exists")
            with accepted_result.open("xb") as stream:
                stream.write(result_path.read_bytes())
            accepted_records.append(
                {
                    "job_id": job_id,
                    "attempt_id": row["attempt_id"],
                    "result_path": str(accepted_result),
                    "result_sha256": sha256_file(accepted_result),
                    "done_sha256": canonical_sha256(validated_done),
                    "task_record_aggregate_sha256": validated_done[
                        "task_record_aggregate_sha256"
                    ],
                    "receive_receipt_sha256": receipt_sha_placeholder,
                }
            )
            transition["status"] = "accepted"
            transition_rows.append(transition)
        failed_count = sum(row["status"] == "failed" for row in transition_rows)
        receipt_core_without_ledger = {
            "schema": RECEIVE_SCHEMA,
            "status": "wave_received_source_replayed_after_cleanup",
            "plan_sha256": checked_plan["plan_sha256"],
            "input_ledger_sha256": checked_ledger["ledger_sha256"],
            "resume_sha256": checked_resume["resume_sha256"],
            "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
            "wave_index": checked_resume["resume_wave_index"],
            "accepted_records": accepted_records,
            "accepted_job_count": len(accepted_records),
            "failed_job_count": failed_count,
            "source_replay_complete": True,
            "owned_compute_absent": True,
            "worker_iam_removed": True,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        # The receive receipt digest is included in accepted records and the
        # ledger.  Bind it to an invariant preimage without a circular hash.
        receive_identity = canonical_sha256(receipt_core_without_ledger)
        for record in accepted_records:
            record["receive_receipt_sha256"] = receive_identity
        output_ledger = _append_transition(
            plan=checked_plan,
            ledger=checked_ledger,
            resume=checked_resume,
            lifecycle=lifecycle,
            receive_receipt_sha256=receive_identity,
            rows=transition_rows,
            accepted_records=accepted_records,
        )
        receipt_core = {
            **receipt_core_without_ledger,
            "accepted_records": accepted_records,
            "output_ledger": output_ledger,
            "output_ledger_sha256": output_ledger["ledger_sha256"],
        }
        receipt = {**receipt_core, "receipt_sha256": canonical_sha256(receipt_core)}
        _write_once(output / "receive_receipt.json", receipt)
        return validate_receive_receipt(
            checked_plan,
            checked_ledger,
            checked_resume,
            lifecycle,
            receipt,
        )
    except Exception:
        # Preserve partial receiver evidence for forensics.  It cannot contain
        # a receipt and therefore cannot be mistaken for an accepted wave.
        raise


def validate_receive_receipt(
    plan: Mapping[str, Any],
    input_ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_input = validate_attempt_ledger(checked_plan, input_ledger)
    checked_resume = validate_resume_plan(checked_plan, checked_input, resume)
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _RECEIVE_KEYS, "fresh-quality receive receipt")
    payload = deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("fresh-quality receive receipt digest changed")
    accepted = receipt.get("accepted_records")
    if not isinstance(accepted, list):
        raise ValueError("fresh-quality receive accepted records are missing")
    for row in accepted:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality accepted record is not an object")
        _exact_keys(row, _ACCEPTED_KEYS, "fresh-quality accepted record")
    identity_records = deepcopy(accepted)
    for row in identity_records:
        row["receive_receipt_sha256"] = "0" * 64
    identity_core = {
        key: deepcopy(receipt[key])
        for key in (
            "schema",
            "status",
            "plan_sha256",
            "input_ledger_sha256",
            "resume_sha256",
            "lifecycle_receipt_sha256",
            "wave_index",
            "accepted_records",
            "accepted_job_count",
            "failed_job_count",
            "source_replay_complete",
            "owned_compute_absent",
            "worker_iam_removed",
            "cloud_mutated",
            "current_profile_changed",
        )
    }
    identity_core["accepted_records"] = identity_records
    receive_identity = canonical_sha256(identity_core)
    if any(
        row["receive_receipt_sha256"] != receive_identity for row in accepted
    ):
        raise ValueError("fresh-quality receive identity changed")
    output_ledger = validate_attempt_ledger(
        checked_plan, receipt.get("output_ledger", {})
    )
    if (
        receipt.get("schema") != RECEIVE_SCHEMA
        or receipt.get("status") != "wave_received_source_replayed_after_cleanup"
        or receipt.get("plan_sha256") != checked_plan["plan_sha256"]
        or receipt.get("input_ledger_sha256") != checked_input["ledger_sha256"]
        or receipt.get("resume_sha256") != checked_resume["resume_sha256"]
        or receipt.get("lifecycle_receipt_sha256") != lifecycle["receipt_sha256"]
        or receipt.get("wave_index") != checked_resume["resume_wave_index"]
        or receipt.get("accepted_job_count") != len(accepted)
        or receipt.get("accepted_job_count") + receipt.get("failed_job_count")
        != len(checked_resume["selected_attempts"])
        or receipt.get("output_ledger_sha256") != output_ledger["ledger_sha256"]
        or receipt.get("source_replay_complete") is not True
        or receipt.get("owned_compute_absent") is not True
        or receipt.get("worker_iam_removed") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality receive receipt boundary changed")
    return receipt


def build_final_receipt(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    accepted_results_directory: str | Path,
) -> dict[str, Any]:
    checked_plan = validate_gcp_plan(plan, replay_sources=True)
    checked_ledger = validate_attempt_ledger(checked_plan, ledger)
    resume = build_resume_plan(checked_plan, checked_ledger)
    if resume["all_jobs_accepted"] is not True:
        raise PermissionError("all fifteen fresh-quality jobs are not accepted")
    accepted_ids = [row["job_id"] for row in checked_ledger["accepted_jobs"]]
    expected_ids = [job["job_id"] for job in checked_plan["jobs"]]
    if accepted_ids != expected_ids:
        raise ValueError("fresh-quality accepted job order changed")
    staging = Path(checked_plan["source_paths"]["staging_directory"])
    package_archive = _file_from_record(
        staging, checked_plan["launch_manifest"]["quality_package"], "quality package"
    )
    with tempfile.TemporaryDirectory(prefix="ofc-fq-final-") as temporary:
        package = transport.extract_and_validate_package(
            archive_path=package_archive,
            expected_archive_sha256=checked_plan["launch_manifest"]["quality_package"][
                "sha256"
            ],
            extraction_directory=Path(temporary) / "package",
        )
        merge = quality_gate.build_fresh_quality_merge(
            plan_path=Path(package["package_root"]) / "control" / "plan.json",
            materialization_path=Path(package["package_root"])
            / "control"
            / "materialization.json",
            root_seal_path=Path(package["package_root"])
            / "control"
            / "root_seal.json",
            results_directory=Path(accepted_results_directory).resolve(),
            performance_receipt_path=checked_plan["source_paths"][
                "performance_receipt"
            ],
        )
        gate = quality_gate.build_fresh_quality_gate(
            merge=merge, replay_sources=True
        )
    passed = gate["quality_pilot_passed"] is True
    core = {
        "schema": FINAL_SCHEMA,
        "status": "qualified" if passed else "no_go",
        "decision": (
            "fresh_quality_passed_open_25_paired_data_pilot_only"
            if passed
            else "fresh_quality_failed_no_data_fanout"
        ),
        "plan_sha256": checked_plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "accepted_job_count": len(accepted_ids),
        "accepted_job_ids": accepted_ids,
        "lifecycle_transition_sha256s": [
            row["transition_sha256"] for row in checked_ledger["transitions"]
        ],
        "quality_merge": merge,
        "quality_merge_sha256": canonical_sha256(merge),
        "quality_gate": gate,
        "quality_gate_sha256": canonical_sha256(gate),
        "quality_pilot_passed": passed,
        "data_pilot_25_paired_authorized": passed,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_final_receipt(receipt)


def validate_final_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _FINAL_KEYS, "fresh-quality final receipt")
    payload = deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("fresh-quality final receipt digest changed")
    merge = receipt.get("quality_merge")
    gate = receipt.get("quality_gate")
    if not isinstance(merge, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("fresh-quality final merge/gate is missing")
    passed = gate.get("quality_pilot_passed") is True
    if (
        receipt.get("schema") != FINAL_SCHEMA
        or receipt.get("status") != ("qualified" if passed else "no_go")
        or receipt.get("quality_merge_sha256") != canonical_sha256(merge)
        or receipt.get("quality_gate_sha256") != canonical_sha256(gate)
        or gate.get("merge_sha256") != canonical_sha256(merge)
        or receipt.get("accepted_job_count") != 15
        or receipt.get("accepted_job_ids")
        != [job_id for ids in WAVE_JOB_IDS for job_id in ids]
        or receipt.get("quality_pilot_passed") is not passed
        or receipt.get("data_pilot_25_paired_authorized") is not passed
        or receipt.get("full_9000_paired_fanout_authorized") is not False
        or any(
            receipt.get(field) is not False
            for field in (
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("fresh-quality final receipt boundary changed")
    return receipt


def write_final_receipt(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    receipt = build_final_receipt(**kwargs)
    _write_once(path, receipt)
    stored = _read_canonical(path, "stored fresh-quality final receipt")
    if validate_final_receipt(stored) != receipt:
        raise ValueError("stored fresh-quality final receipt changed")
    return receipt


__all__ = [
    "ATTEMPT_IDS",
    "FINAL_SCHEMA",
    "LEDGER_SCHEMA",
    "LIFECYCLE_SCHEMA",
    "MAX_ATTEMPTS_PER_JOB",
    "MAX_CONCURRENT_VMS",
    "PLAN_SCHEMA",
    "RECEIVE_SCHEMA",
    "RESUME_SCHEMA",
    "RUN_NAME_PREFIX",
    "WAVE_JOB_COUNTS",
    "WAVE_JOB_IDS",
    "WAVE_REQUEST_SCHEMA",
    "RemoteObjectReader",
    "build_final_receipt",
    "build_gcp_plan",
    "build_resume_plan",
    "build_wave_request",
    "canonical_bytes",
    "canonical_sha256",
    "initial_attempt_ledger",
    "open_controller_journal",
    "receive_ready_jobs",
    "validate_attempt_ledger",
    "validate_final_receipt",
    "validate_gcp_plan",
    "validate_lifecycle_receipt",
    "validate_receive_receipt",
    "validate_resume_plan",
    "validate_wave_request",
    "write_final_receipt",
    "write_gcp_plan",
]
