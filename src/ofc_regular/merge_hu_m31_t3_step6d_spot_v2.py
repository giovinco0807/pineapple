"""Bridge an immutable Step 6d Spot receive into the existing v2 merger.

The receive receipt is treated as an untrusted index.  This module revalidates
its package/authorization chain, resolves every relative DONE beneath the
receive root, proves exact candidate/reference tail coverage, and independently
validates each completed runner directory before invoking the scientific merger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_spot_v2 as spot
from . import merge_hu_m31_t3_step6d_candidate02_performance as candidate02_merger
from . import (
    merge_hu_m31_t3_step6d_candidate02_tail_v2 as candidate02_tail_v2_merger,
)
from . import merge_hu_m31_t3_step6d_performance_v2 as merger
from . import run_hu_m31_t3_step6d_performance_v2 as runner


_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "launch_target",
        "result_prefix",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "launch_claim_sha256",
        "launch_result_sha256",
        "resume_claim_sha256",
        "resume_result_sha256",
        "run_contract_digest",
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
_PREFLIGHT_KEYS = frozenset(
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
_CREATED_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "work_hand_indices",
        "instance",
        "zone",
        "attempt_index",
        "status",
    }
)
_RECEIPT_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "work_hand_indices",
        "done_path",
        "done_sha256",
        "source_hand_path",
        "source_hand_sha256",
        "root_sha256",
        "run_contract_digest",
    }
)
_PACKAGE_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "work_hand_indices",
        "path",
        "output_prefix",
        "sha256",
        "bytes",
    }
)
_EXPECTED_RECEIVED_FILENAMES = frozenset(
    {"DONE.json", "run_contract.json", "shard_manifest.json"}
)
_CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH = (
    spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH
)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} keys changed (missing={missing}, extra={extra})")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.casefold()
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    original = Path(path)
    if original.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = original.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != runner.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _safe_relative_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a portable relative path")
    fragment = PurePosixPath(relative)
    if fragment.is_absolute() or ".." in fragment.parts or "." in fragment.parts:
        raise ValueError(f"{label} escapes the receive root")
    current = root
    for part in fragment.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink")
    resolved = current.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the receive root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is missing")
    return resolved


def _contract_tail_hand_indices(contract: Mapping[str, Any]) -> tuple[int, ...]:
    indices = tuple(spot._contract_tail_hand_indices(contract))
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Step 6d received contract tail indices changed")
    return indices


def _contract_authorized_job_ids(contract: Mapping[str, Any]) -> tuple[str, ...]:
    expected = tuple(spot.authorized_job_ids(contract))
    tail = _contract_tail_hand_indices(contract)
    independently_derived = tuple(
        spot.job_id(role, index, contract)
        for role in runner.SOURCE_ROLES
        for index in tail
    )
    if expected != independently_derived:
        raise ValueError("Step 6d received authorized job mapping changed")
    return expected


def _validate_package_chain(
    receive_root: Path, receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]]:
    package_path = receive_root / "package_manifest.json"
    authorization_path = receive_root / spot.AUTHORIZATION_NAME
    ready_path = receive_root / spot.PACKAGE_READY_NAME
    package = _read_canonical(package_path, "Step 6d received package manifest")
    authorization = _read_canonical(
        authorization_path, "Step 6d received launch authorization"
    )
    ready = _read_canonical(ready_path, "Step 6d received package-ready marker")
    _require_exact_keys(
        package, spot._PACKAGE_MANIFEST_KEYS, "Step 6d received package manifest"
    )
    _require_exact_keys(
        authorization,
        spot._AUTHORIZATION_KEYS,
        "Step 6d received launch authorization",
    )
    _require_exact_keys(
        ready, spot._PACKAGE_READY_KEYS, "Step 6d received package-ready marker"
    )

    contract = runner.validate_run_contract(
        _mapping(package.get("run_contract"), "Step 6d package run contract")
    )
    contract_digest = runner.canonical_sha256(contract)
    run_name = receipt.get("run_name")
    package_sha = _sha256(package_path)
    authorization_sha = _sha256(authorization_path)
    expected_ids = list(_contract_authorized_job_ids(contract))
    tail_indices = _contract_tail_hand_indices(contract)
    logical_job_count = len(expected_ids)
    forbidden = (
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    )
    if (
        package.get("schema") != spot.SPOT_PACKAGE_SCHEMA
        or package.get("status") != "immutable_package_ready_not_authorized"
        or package.get("run_name") != run_name
        or package.get("run_contract_digest") != contract_digest
        or receipt.get("package_manifest_sha256") != package_sha
        or authorization.get("schema") != spot.LAUNCH_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "explicit_tail_spot_authorization"
        or authorization.get("run_name") != run_name
        or authorization.get("package_manifest_sha256") != package_sha
        or authorization.get("run_contract_digest") != contract_digest
        or authorization.get("authorized_job_ids") != expected_ids
        or authorization.get("logical_job_count") != logical_job_count
        or authorization.get("tail_only") is not True
        or authorization.get("spot_execution_authorized") is not True
        or isinstance(authorization.get("authorized_unix_seconds"), bool)
        or not isinstance(authorization.get("authorized_unix_seconds"), (int, float))
        or authorization["authorized_unix_seconds"] <= 0
        or receipt.get("launch_authorization_sha256") != authorization_sha
        or any(package.get(field) is not False for field in forbidden)
        or any(authorization.get(field) is not False for field in forbidden)
    ):
        raise ValueError("Step 6d received package/authorization chain changed")
    if (
        authorization.get("source_sha256") != package.get("source_sha256")
        or authorization.get("startup_sha256") != package.get("startup_sha256")
        or authorization.get("launch_target") != package.get("launch_target")
        or authorization.get("cost_guard") != package.get("cost_guard")
    ):
        raise ValueError("Step 6d received package source/launch binding changed")
    if ready != {
        "schema": spot.PACKAGE_READY_SCHEMA,
        "status": "immutable_local_package_complete",
        "run_name": run_name,
        "package_manifest_sha256": package_sha,
        "source_sha256": package["source_sha256"],
        "startup_sha256": package["startup_sha256"],
        "run_contract_digest": contract_digest,
        "logical_job_count": logical_job_count,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
    }:
        raise ValueError("Step 6d received package-ready chain changed")

    reference = _mapping(package.get("accepted_reference"), "accepted reference")
    candidate = _mapping(package.get("accepted_candidate"), "accepted candidate")
    entries = _mapping(package.get("source_entries"), "package source entries")
    if (
        set(reference) != {"package_path", "sha256"}
        or set(candidate) != {"package_path", "sha256"}
        or reference.get("package_path") != spot.REFERENCE_PACKAGE_PATH
        or candidate.get("package_path") != spot.CANDIDATE_PACKAGE_PATH
        or reference.get("sha256") != spot.REFERENCE_NATIVE_LIBRARY_SHA256
        or reference.get("sha256") != contract["reference_library_sha256"]
        or candidate.get("sha256") != contract["candidate_library_sha256"]
        or reference.get("sha256") == candidate.get("sha256")
        or reference.get("package_path") == candidate.get("package_path")
        or package.get("source_entry_count") != len(entries)
    ):
        raise ValueError("Step 6d received source binary binding changed")
    for binding in (reference, candidate):
        entry = _mapping(
            entries.get(binding["package_path"]), "bound native source entry"
        )
        if (
            set(entry) != {"sha256", "bytes"}
            or entry.get("sha256") != binding["sha256"]
            or isinstance(entry.get("bytes"), bool)
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] <= 0
        ):
            raise ValueError("Step 6d received native source entry changed")

    tail = _mapping(package.get("tail_schedule"), "Step 6d package tail schedule")
    if (
        tail.get("contract_hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or tail.get("work_hand_indices") != list(tail_indices)
        or tail.get("source_roles") != list(runner.SOURCE_ROLES)
        or tail.get("logical_job_count") != logical_job_count
        or tail.get("max_logical_jobs") != spot.MAX_LOGICAL_JOBS
        or tail.get("mapping") != "one_source_hand_per_vm"
    ):
        raise ValueError("Step 6d received tail schedule changed")

    records = _array(package.get("job_manifests"), "package job manifests")
    if len(records) != logical_job_count:
        raise ValueError("Step 6d package job count changed")
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for expected_id, raw in zip(expected_ids, records, strict=True):
        record = _mapping(raw, "package job manifest record")
        _require_exact_keys(record, _PACKAGE_JOB_KEYS, "package job manifest record")
        identifier = record.get("job_id")
        if identifier != expected_id or identifier in records_by_id:
            raise ValueError("Step 6d package job identity/order changed")
        records_by_id[str(identifier)] = record
    return contract, package, records_by_id


def _positive_finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _validate_launch_evidence(
    receive_root: Path,
    *,
    receipt: Mapping[str, Any],
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    claim_path = receive_root / spot.LAUNCH_CLAIM_NAME
    result_path = receive_root / spot.LAUNCH_RESULT_NAME
    claim = _read_canonical(claim_path, "Step 6d v2 launch claim")
    result = _read_canonical(result_path, "Step 6d v2 launch result")
    _require_exact_keys(claim, _LAUNCH_CLAIM_KEYS, "Step 6d v2 launch claim")
    _require_exact_keys(result, _LAUNCH_RESULT_KEYS, "Step 6d v2 launch result")
    expected_ids = list(_contract_authorized_job_ids(contract))
    logical_job_count = len(expected_ids)
    run_name = str(receipt["run_name"])
    contract_digest = str(receipt["run_contract_digest"])
    target = spot.build_launch_target()
    cost_guard = spot.build_cost_guard()
    cost_guard_sha = runner.canonical_sha256(cost_guard)
    if (
        not _is_sha256(receipt.get("launch_claim_sha256"))
        or not _is_sha256(receipt.get("launch_result_sha256"))
        or receipt["launch_claim_sha256"] != _sha256(claim_path)
        or receipt["launch_result_sha256"] != _sha256(result_path)
        or claim.get("schema") != spot.LAUNCH_CLAIM_SCHEMA
        or claim.get("status") != "exclusive_claim_acquired_before_remote_mutation"
        or claim.get("run_name") != run_name
        or claim.get("selected_job_ids") != expected_ids
        or claim.get("run_contract_digest") != contract_digest
        or claim.get("launch_target") != target
        or claim.get("launch_target") != package.get("launch_target")
        or claim.get("cost_guard_sha256") != cost_guard_sha
        or claim.get("crash_reuse_authorized") is not False
    ):
        raise ValueError("Step 6d v2 launch claim chain changed")
    _positive_finite_number(
        claim.get("claimed_unix_seconds"), "Step 6d v2 claim timestamp"
    )

    preflight = _mapping(result.get("preflight"), "Step 6d v2 launch preflight")
    _require_exact_keys(preflight, _PREFLIGHT_KEYS, "Step 6d v2 launch preflight")
    expected_instances = [
        f"{run_name}-j{index:02d}" for index in range(logical_job_count)
    ]
    expected_done_uris = [
        f"{receipt['result_prefix']}/jobs/{identifier}/DONE.json"
        for identifier in expected_ids
    ]
    if (
        preflight.get("schema") != spot.LAUNCH_PREFLIGHT_SCHEMA
        or preflight.get("status") != "quota_and_collision_checks_passed"
        or preflight.get("region") != spot.DEFAULT_REGION
        or preflight.get("selected_job_ids") != expected_ids
        or preflight.get("selected_instance_names") != expected_instances
        or preflight.get("selected_done_uris") != expected_done_uris
        or preflight.get("required_vcpus") != logical_job_count * spot.VCPUS_PER_VM
        or preflight.get("instances_absent") is not True
        or preflight.get("done_objects_absent") is not True
        or preflight.get("cost_guard_sha256") != cost_guard_sha
        or preflight.get("selected_estimated_max_compute_usd")
        != cost_guard["all_20_estimated_max_compute_usd"]
        or preflight.get("hard_tail_compute_cap_usd")
        != cost_guard["hard_tail_compute_cap_usd"]
        or claim.get("preflight_sha256") != runner.canonical_sha256(preflight)
    ):
        raise ValueError("Step 6d v2 launch preflight chain changed")
    _positive_finite_number(
        preflight.get("checked_unix_seconds"), "Step 6d v2 preflight timestamp"
    )
    quota = _mapping(preflight.get("quota"), "Step 6d v2 preflight quota")
    if set(quota) != {"CPUS", "PREEMPTIBLE_CPUS"}:
        raise ValueError("Step 6d v2 preflight quota metrics changed")
    for metric, raw_record in quota.items():
        record = _mapping(raw_record, f"Step 6d v2 {metric} quota")
        if set(record) != {"limit", "usage", "available"}:
            raise ValueError(f"Step 6d v2 {metric} quota fields changed")
        values = (record.get("limit"), record.get("usage"), record.get("available"))
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError(f"Step 6d v2 {metric} quota values changed")
        if (
            record["available"] != record["limit"] - record["usage"]
            or record["available"] < preflight["required_vcpus"]
        ):
            raise ValueError(f"Step 6d v2 {metric} quota proof changed")

    if (
        result.get("schema") != spot.LAUNCH_RESULT_SCHEMA
        or result.get("status") != "selected_tail_jobs_created"
        or result.get("run_name") != run_name
        or result.get("selected_job_ids") != expected_ids
        or result.get("logical_job_count") != logical_job_count
        or result.get("max_logical_jobs") != spot.MAX_LOGICAL_JOBS
        or result.get("machine_type") != spot.EXPECTED_MACHINE_TYPE
        or result.get("process_count") != spot.PROCESS_COUNT
        or result.get("rayon_threads_per_process") != spot.RAYON_THREADS_PER_PROCESS
        or result.get("run_contract_digest") != contract_digest
        or result.get("launch_target") != target
        or result.get("cost_guard") != cost_guard
        or result.get("launch_claim_sha256") != runner.canonical_sha256(claim)
        or result.get("failures") != []
        or result.get("cleanup") != []
        or result.get("cleanup_absence_proven") is not True
        or result.get("cleanup_compute_stopped_or_absent") is not True
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
        raise ValueError("Step 6d v2 launch result chain changed")
    created = _array(result.get("created"), "Step 6d v2 created jobs")
    if len(created) != logical_job_count:
        raise ValueError("Step 6d v2 created job count changed")
    for ordinal, (identifier, raw_created) in enumerate(
        zip(expected_ids, created, strict=True)
    ):
        created_job = _mapping(raw_created, "Step 6d v2 created job")
        _require_exact_keys(created_job, _CREATED_JOB_KEYS, "Step 6d v2 created job")
        role = str(identifier).split("-hand-", 1)[0]
        hand_index = int(str(identifier).rsplit("-", 1)[-1])
        if created_job != {
            "job_id": identifier,
            "source_role": role,
            "work_hand_indices": [hand_index],
            "instance": expected_instances[ordinal],
            "zone": spot.DEFAULT_ZONES[ordinal % len(spot.DEFAULT_ZONES)],
            "attempt_index": 0,
            "status": "created",
        }:
            raise ValueError("Step 6d v2 created job binding changed")

    resume_claim_sha = receipt.get("resume_claim_sha256")
    resume_result_sha = receipt.get("resume_result_sha256")
    if resume_claim_sha is None and resume_result_sha is None:
        if (
            (receive_root / spot.RESUME_CLAIM_NAME).exists()
            or (receive_root / spot.RESUME_RESULT_NAME).exists()
            or spot.validate_receive_resume_chain(
                run_dir=receive_root, manifest=package
            )
            is not None
        ):
            raise ValueError("Step 6d v2 unbound resume evidence exists")
        return
    if not (_is_sha256(resume_claim_sha) and _is_sha256(resume_result_sha)):
        raise ValueError("Step 6d v2 resume receipt hashes must be an exact pair")
    resume_claim_path = receive_root / spot.RESUME_CLAIM_NAME
    resume_result_path = receive_root / spot.RESUME_RESULT_NAME
    _read_canonical(resume_claim_path, "Step 6d v2 resume claim")
    _read_canonical(resume_result_path, "Step 6d v2 resume result")
    resume_chain = spot.validate_receive_resume_chain(
        run_dir=receive_root, manifest=package
    )
    if (
        _sha256(resume_claim_path) != resume_claim_sha
        or _sha256(resume_result_path) != resume_result_sha
        or resume_chain is None
    ):
        raise ValueError("Step 6d v2 resume evidence chain changed")
    resume_claim, resume_result = resume_chain
    selected = _array(
        resume_claim.get("selected_job_ids"), "Step 6d v2 resume selected jobs"
    )
    expected_ids = list(_contract_authorized_job_ids(contract))
    selected_set = set(selected)
    preflight = _mapping(resume_result.get("preflight"), "Step 6d v2 resume preflight")
    if (
        not selected
        or len(selected_set) != len(selected)
        or selected != [value for value in expected_ids if value in selected_set]
        or resume_result.get("selected_job_ids") != selected
        or resume_claim.get("initial_launch_claim_sha256")
        != receipt["launch_claim_sha256"]
        or resume_claim.get("initial_launch_result_sha256")
        != receipt["launch_result_sha256"]
        or resume_result.get("initial_launch_claim_sha256")
        != receipt["launch_claim_sha256"]
        or resume_result.get("initial_launch_result_sha256")
        != receipt["launch_result_sha256"]
        or preflight.get("validated_completed_job_ids")
        != [value for value in expected_ids if value not in selected_set]
        or preflight.get("all_incomplete_jobs_selected") is not True
    ):
        raise ValueError("Step 6d v2 resume completeness chain changed")


def _validate_receipt_header(
    receive_root: Path,
    *,
    expected_run_name: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Mapping[str, Any]],
]:
    receipt = _read_canonical(
        receive_root / "receive_receipt.json", "Step 6d v2 receive receipt"
    )
    _require_exact_keys(receipt, _RECEIPT_KEYS, "Step 6d v2 receive receipt")
    run_name = receipt.get("run_name")
    forbidden = (
        "merge_executed",
        "production_fanout_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    )
    resume_claim_sha = receipt.get("resume_claim_sha256")
    resume_result_sha = receipt.get("resume_result_sha256")
    resume_hashes_valid = (resume_claim_sha is None and resume_result_sha is None) or (
        _is_sha256(resume_claim_sha) and _is_sha256(resume_result_sha)
    )
    if (
        receipt.get("schema") != spot.RECEIVE_SCHEMA
        or receipt.get("status") != "exact_tail_source_jobs_received_and_validated"
        or not isinstance(run_name, str)
        or spot._SAFE_RUN.fullmatch(run_name) is None
        or (expected_run_name is not None and run_name != expected_run_name)
        or not _is_sha256(receipt.get("package_manifest_sha256"))
        or not _is_sha256(receipt.get("launch_authorization_sha256"))
        or not _is_sha256(receipt.get("launch_claim_sha256"))
        or not _is_sha256(receipt.get("launch_result_sha256"))
        or not resume_hashes_valid
        or not _is_sha256(receipt.get("run_contract_digest"))
        or receipt.get("source_roles") != list(runner.SOURCE_ROLES)
        or receipt.get("source_isolation_validated") is not True
        or any(receipt.get(field) is not False for field in forbidden)
    ):
        raise ValueError("Step 6d v2 receive receipt boundary changed")
    if (
        expected_run_name is not None
        and spot._SAFE_RUN.fullmatch(expected_run_name) is None
    ):
        raise ValueError("expected Step 6d v2 run name is unsafe")
    jobs = _array(receipt.get("jobs"), "Step 6d v2 receipt jobs")
    candidate_paths = _array(
        receipt.get("candidate_done_paths"), "candidate DONE receipt paths"
    )
    reference_paths = _array(
        receipt.get("reference_done_paths"), "reference DONE receipt paths"
    )

    contract, package, records = _validate_package_chain(receive_root, receipt)
    tail_indices = _contract_tail_hand_indices(contract)
    logical_job_count = len(_contract_authorized_job_ids(contract))
    if (
        receipt.get("work_hand_indices") != list(tail_indices)
        or receipt.get("logical_job_count") != logical_job_count
        or len(jobs) != logical_job_count
        or len(candidate_paths) != len(tail_indices)
        or len(reference_paths) != len(tail_indices)
    ):
        raise ValueError("Step 6d v2 receipt tail cardinality changed")
    if (
        receipt["run_contract_digest"] != runner.canonical_sha256(contract)
        or package["run_contract"] != contract
        or receipt.get("launch_target") != package.get("launch_target")
        or receipt.get("launch_target") != spot.build_launch_target()
        or receipt.get("result_prefix")
        != (
            f"gs://{spot.build_launch_target()['bucket']}/runs/"
            f"{receipt['run_name']}/results"
        )
    ):
        raise ValueError("Step 6d v2 receipt shared contract changed")
    _validate_launch_evidence(
        receive_root,
        receipt=receipt,
        package=package,
        contract=contract,
    )
    return receipt, contract, package, records


def _validate_package_semantics(package: Mapping[str, Any]) -> None:
    """Validate package semantics that remain provable after source removal."""

    feature = _mapping(package.get("feature_encoder"), "feature encoder binding")
    image = _mapping(package.get("image"), "package image binding")
    allocation = _mapping(package.get("allocation"), "package allocation")
    checkpoint = _mapping(package.get("checkpoint"), "package checkpoint policy")
    heartbeat = _mapping(package.get("heartbeat"), "package heartbeat policy")
    entries = _mapping(package.get("source_entries"), "package source entries")
    if (
        package.get("source_name") != spot.SOURCE_NAME
        or not _is_sha256(package.get("source_sha256"))
        or isinstance(package.get("source_bytes"), bool)
        or not isinstance(package.get("source_bytes"), int)
        or package["source_bytes"] <= 0
        or package.get("startup_name") != spot.STARTUP_NAME
        or not _is_sha256(package.get("startup_sha256"))
        or feature
        != {
            "package_path": spot.FEATURE_PACKAGE_PATH,
            "sha256": spot.EXPECTED_FEATURE_ENCODER_SHA256,
        }
        or image
        != {
            "project": "debian-cloud",
            "name": spot.EXPECTED_IMAGE_NAME,
            "id": spot.EXPECTED_IMAGE_ID,
            "self_link": spot.EXPECTED_IMAGE_SELF_LINK,
        }
        or allocation
        != {
            "machine_type": spot.EXPECTED_MACHINE_TYPE,
            "process_count": spot.PROCESS_COUNT,
            "rayon_threads_per_process": spot.RAYON_THREADS_PER_PROCESS,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        }
        or package.get("launch_target") != spot.build_launch_target()
        or package.get("cost_guard") != spot.build_cost_guard()
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
            "schema": spot.HEARTBEAT_SCHEMA,
            "required": True,
            "interval_seconds": spot.HEARTBEAT_INTERVAL_SECONDS,
            "remote_scope": "progress_only",
        }
        or package.get("spot_execution_authorized") is not False
        or package.get("m31_complete") is not False
        or package.get("gcloud_invoked") is not False
        or not spot._REQUIRED_SOURCE_PATHS.issubset(entries)
    ):
        raise ValueError("Step 6d v2 received package semantic boundary changed")
    for name, raw in entries.items():
        path = PurePosixPath(name) if isinstance(name, str) else None
        entry = _mapping(raw, "package source entry")
        if (
            path is None
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or set(entry) != {"sha256", "bytes"}
            or not _is_sha256(entry.get("sha256"))
            or isinstance(entry.get("bytes"), bool)
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] < 0
        ):
            raise ValueError("Step 6d v2 received source entry changed")

    contract = runner.validate_run_contract(
        _mapping(package.get("run_contract"), "package semantic run contract")
    )
    if runner.contract_variant(contract) == runner.CANDIDATE02_TAIL_V2_VARIANT:
        selection = _mapping(
            entries.get(_CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH),
            "candidate02 tail-v2 selection manifest source entry",
        )
        if (
            contract.get("selection_manifest_sha256")
            != runner.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
            or set(selection) != {"sha256", "bytes"}
            or selection.get("sha256")
            != runner.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
            or isinstance(selection.get("bytes"), bool)
            or not isinstance(selection.get("bytes"), int)
            or selection["bytes"] != spot.CANDIDATE02_TAIL_V2_SELECTION_BYTES
        ):
            raise ValueError(
                "candidate02 tail-v2 selection manifest package binding changed"
            )


def _expected_job_tree(source_role: str, hand_index: int) -> tuple[set[str], set[str]]:
    files = {
        "DONE.json",
        "run_contract.json",
        "shard_manifest.json",
        f"roots/hand_{hand_index:03d}.json",
        f"hands/{source_role}/hand_{hand_index:03d}.json",
    }
    directories = {"roots", "hands", f"hands/{source_role}"}
    return files, directories


def _validate_job_tree(
    *,
    receive_root: Path,
    receipt_job: Mapping[str, Any],
    package_record: Mapping[str, Any],
    contract: Mapping[str, Any],
    source_role: str,
    hand_index: int,
    job_id: str,
) -> tuple[Path, Path, Path]:
    expected_manifest_relative = f"jobs/{job_id}.json"
    expected_prefix = f"jobs/{job_id}"
    if (
        package_record.get("job_id") != job_id
        or package_record.get("source_role") != source_role
        or package_record.get("work_hand_indices") != [hand_index]
        or package_record.get("path") != expected_manifest_relative
        or package_record.get("output_prefix") != expected_prefix
        or not _is_sha256(package_record.get("sha256"))
        or isinstance(package_record.get("bytes"), bool)
        or not isinstance(package_record.get("bytes"), int)
        or package_record["bytes"] <= 0
    ):
        raise ValueError(f"Step 6d v2 package job binding changed: {job_id}")
    package_job_path = _safe_relative_file(
        receive_root,
        expected_manifest_relative,
        f"Step 6d v2 package job manifest {job_id}",
    )
    package_job = spot.validate_job_manifest(
        _read_canonical(package_job_path, f"Step 6d v2 package job {job_id}")
    )
    if (
        _sha256(package_job_path) != package_record["sha256"]
        or package_job_path.stat().st_size != package_record["bytes"]
        or package_job.get("run_contract") != contract
        or package_job.get("run_contract_digest") != runner.canonical_sha256(contract)
        or package_job.get("source_role") != source_role
        or package_job.get("work_hand_indices") != [hand_index]
    ):
        raise ValueError(f"Step 6d v2 package job bytes changed: {job_id}")

    expected_done = f"jobs/{job_id}/DONE.json"
    expected_hand = f"jobs/{job_id}/hands/{source_role}/hand_{hand_index:03d}.json"
    expected_root = f"jobs/{job_id}/roots/hand_{hand_index:03d}.json"
    if (
        receipt_job.get("done_path") != expected_done
        or receipt_job.get("source_hand_path") != expected_hand
        or not _is_sha256(receipt_job.get("done_sha256"))
        or not _is_sha256(receipt_job.get("source_hand_sha256"))
        or not _is_sha256(receipt_job.get("root_sha256"))
    ):
        raise ValueError(f"Step 6d v2 receipt artifact binding changed: {job_id}")
    done_path = _safe_relative_file(
        receive_root, receipt_job["done_path"], f"Step 6d v2 DONE {job_id}"
    )
    hand_path = _safe_relative_file(
        receive_root,
        receipt_job["source_hand_path"],
        f"Step 6d v2 source hand {job_id}",
    )
    root_path = _safe_relative_file(
        receive_root, expected_root, f"Step 6d v2 root {job_id}"
    )
    job_dir = done_path.parent
    if job_dir != (receive_root / expected_prefix).resolve() or job_dir.is_symlink():
        raise ValueError(f"Step 6d v2 receipt job directory changed: {job_id}")

    expected_files, expected_directories = _expected_job_tree(source_role, hand_index)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for entry in job_dir.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"Step 6d v2 receipt job traverses a symlink: {job_id}")
        relative = entry.relative_to(job_dir).as_posix()
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"Step 6d v2 receipt job has an unsafe entry: {job_id}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(f"Step 6d v2 received file set changed: {job_id}")

    stored_contract = runner.validate_run_contract(
        _read_canonical(job_dir / "run_contract.json", f"run contract {job_id}")
    )
    stored_manifest = runner.validate_shard_manifest(
        _read_canonical(job_dir / "shard_manifest.json", f"shard manifest {job_id}")
    )
    _read_canonical(root_path, f"Step 6d v2 root {job_id}")
    _read_canonical(hand_path, f"Step 6d v2 source hand {job_id}")
    stored_done = _read_canonical(done_path, f"Step 6d v2 DONE {job_id}")
    if (
        stored_contract != contract
        or stored_manifest != package_job
        or stored_manifest.get("run_contract") != contract
        or stored_manifest.get("source_role") != source_role
        or stored_manifest.get("work_hand_indices") != [hand_index]
    ):
        raise ValueError(f"Step 6d v2 received contract chain changed: {job_id}")
    validated_done = runner.validate_completed_output(job_dir)
    variant = runner.contract_variant(contract)
    expected_done_schema = {
        runner.CANDIDATE01_VARIANT: runner.DONE_SCHEMA,
        runner.CANDIDATE02_VARIANT: runner.CANDIDATE02_DONE_SCHEMA,
        runner.CANDIDATE02_TAIL_V2_VARIANT: runner.CANDIDATE02_TAIL_V2_DONE_SCHEMA,
    }.get(variant)
    if expected_done_schema is None:
        raise ValueError("Step 6d v2 received DONE variant changed")
    if (
        validated_done != stored_done
        or stored_done.get("schema") != expected_done_schema
        or stored_done.get("status") != "complete_source_isolated_shard"
        or stored_done.get("run_contract_digest") != runner.canonical_sha256(contract)
        or stored_done.get("source_role") != source_role
        or stored_done.get("work_hand_indices") != [hand_index]
        or stored_done.get("completed_hand_indices") != [hand_index]
        or stored_done.get("reference_library_sha256")
        != contract["reference_library_sha256"]
        or stored_done.get("candidate_library_sha256")
        != contract["candidate_library_sha256"]
        or stored_done.get("native_library_sha256")
        != contract[f"{source_role}_library_sha256"]
        or any(
            stored_done.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError(f"Step 6d v2 received DONE binding changed: {job_id}")
    if (
        _sha256(done_path) != receipt_job["done_sha256"]
        or _sha256(hand_path) != receipt_job["source_hand_sha256"]
        or _sha256(root_path) != receipt_job["root_sha256"]
    ):
        raise ValueError(f"Step 6d v2 receipt artifact bytes changed: {job_id}")
    return done_path, hand_path, root_path


def _validate_complete_receive_tree(
    receive_root: Path,
    *,
    job_ids: Sequence[str],
    include_resume: bool,
) -> None:
    expected_files = {
        "receive_receipt.json",
        "package_manifest.json",
        spot.AUTHORIZATION_NAME,
        spot.PACKAGE_READY_NAME,
        spot.LAUNCH_CLAIM_NAME,
        spot.LAUNCH_RESULT_NAME,
    }
    if include_resume:
        expected_files.update({spot.RESUME_CLAIM_NAME, spot.RESUME_RESULT_NAME})
    expected_directories = {"jobs"}
    for identifier in job_ids:
        role, hand_text = identifier.rsplit("-hand-", 1)
        hand_index = int(hand_text)
        expected_files.add(f"jobs/{identifier}.json")
        job_files, job_directories = _expected_job_tree(role, hand_index)
        expected_files.update(f"jobs/{identifier}/{value}" for value in job_files)
        expected_directories.add(f"jobs/{identifier}")
        expected_directories.update(
            f"jobs/{identifier}/{value}" for value in job_directories
        )
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for entry in receive_root.rglob("*"):
        if entry.is_symlink():
            raise ValueError("Step 6d v2 receive tree contains a symlink")
        relative = entry.relative_to(receive_root).as_posix()
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError("Step 6d v2 receive tree contains an unsafe entry")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("Step 6d v2 immutable receive tree changed")


def resolve_received_done_paths(
    receive_dir: str | Path,
    *,
    expected_run_name: str | None = None,
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Resolve and revalidate all receipt-bound DONE paths beneath a receive."""

    original_root = Path(receive_dir)
    if original_root.is_symlink():
        raise ValueError("Step 6d v2 receive root must not be a symlink")
    receive_root = original_root.resolve()
    if not receive_root.is_dir():
        raise ValueError(f"Step 6d v2 receive directory is missing: {receive_root}")
    receipt, contract, package, records = _validate_receipt_header(
        receive_root, expected_run_name=expected_run_name
    )
    _validate_package_semantics(package)

    tail_indices = _contract_tail_hand_indices(contract)
    expected_ids = list(_contract_authorized_job_ids(contract))
    expected_pairs = [
        (role, index) for role in runner.SOURCE_ROLES for index in tail_indices
    ]
    candidate_done: list[Path] = []
    reference_done: list[Path] = []
    derived_candidate_paths: list[str] = []
    derived_reference_paths: list[str] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    seen_done_paths: set[Path] = set()
    seen_hand_paths: set[Path] = set()
    seen_root_paths: set[Path] = set()
    receipt_jobs = _array(receipt["jobs"], "Step 6d v2 receipt jobs")
    for expected_id, expected_pair, raw in zip(
        expected_ids, expected_pairs, receipt_jobs, strict=True
    ):
        receipt_job = _mapping(raw, "Step 6d v2 receipt job")
        _require_exact_keys(receipt_job, _RECEIPT_JOB_KEYS, "Step 6d v2 receipt job")
        source_role, hand_index = expected_pair
        if (
            receipt_job.get("job_id") != expected_id
            or receipt_job.get("source_role") != source_role
            or receipt_job.get("work_hand_indices") != [hand_index]
            or receipt_job.get("run_contract_digest")
            != runner.canonical_sha256(contract)
            or expected_id in seen_ids
            or expected_pair in seen_pairs
        ):
            raise ValueError("Step 6d v2 receipt role/hand uniqueness changed")
        done_path, hand_path, root_path = _validate_job_tree(
            receive_root=receive_root,
            receipt_job=receipt_job,
            package_record=records[expected_id],
            contract=contract,
            source_role=source_role,
            hand_index=hand_index,
            job_id=expected_id,
        )
        if (
            done_path in seen_done_paths
            or hand_path in seen_hand_paths
            or root_path in seen_root_paths
        ):
            raise ValueError("Step 6d v2 receipt artifact uniqueness changed")
        seen_ids.add(expected_id)
        seen_pairs.add(expected_pair)
        seen_done_paths.add(done_path)
        seen_hand_paths.add(hand_path)
        seen_root_paths.add(root_path)
        if source_role == "candidate":
            candidate_done.append(done_path)
            derived_candidate_paths.append(str(receipt_job["done_path"]))
        else:
            reference_done.append(done_path)
            derived_reference_paths.append(str(receipt_job["done_path"]))

    if (
        list(seen_ids) == []
        or seen_ids != set(expected_ids)
        or seen_pairs != set(expected_pairs)
        or receipt["candidate_done_paths"] != derived_candidate_paths
        or receipt["reference_done_paths"] != derived_reference_paths
        or len(candidate_done) != len(tail_indices)
        or len(reference_done) != len(tail_indices)
    ):
        raise ValueError("Step 6d v2 receipt exact tail coverage changed")
    _validate_complete_receive_tree(
        receive_root,
        job_ids=expected_ids,
        include_resume=receipt["resume_claim_sha256"] is not None,
    )
    return candidate_done, reference_done, receipt


def merge_received_spot_v2(
    *,
    receive_dir: str | Path,
    summary_output_path: str | Path,
    validation_output_path: str | Path,
    scope: str = "auto",
    expected_run_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an immutable Spot receive, then invoke the existing merger."""

    receive_root = Path(receive_dir).resolve()
    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    for output in (summary_output, validation_output):
        if output == receive_root or receive_root in output.parents:
            raise ValueError(
                "Step 6d v2 merge outputs must be outside the receive tree"
            )
    candidate_done, reference_done, _receipt = resolve_received_done_paths(
        receive_root, expected_run_name=expected_run_name
    )
    contract = runner.validate_run_contract(
        _read_canonical(
            candidate_done[0].parent / "run_contract.json",
            "Step 6d v2 received merge contract",
        )
    )
    variant = runner.contract_variant(contract)
    if variant == runner.CANDIDATE02_TAIL_V2_VARIANT:
        if scope not in {
            "auto",
            candidate02_tail_v2_merger.TAIL_DIAGNOSTIC_SCOPE,
        }:
            raise ValueError(
                "candidate02 tail-v2 Spot receive requires its dedicated tail scope"
            )
        return candidate02_tail_v2_merger.merge_and_validate_candidate02_tail_v2(
            candidate_done_paths=candidate_done,
            reference_done_paths=reference_done,
            summary_output_path=summary_output,
            validation_output_path=validation_output,
        )
    if variant == runner.CANDIDATE02_VARIANT:
        if scope not in {"auto", candidate02_merger.TAIL_DIAGNOSTIC_SCOPE}:
            raise ValueError("candidate02 Spot receive requires candidate02 tail scope")
        return candidate02_merger.merge_and_validate_candidate02_performance(
            candidate_done_paths=candidate_done,
            reference_done_paths=reference_done,
            summary_output_path=summary_output,
            validation_output_path=validation_output,
        )
    if scope in {
        candidate02_merger.TAIL_DIAGNOSTIC_SCOPE,
        candidate02_tail_v2_merger.TAIL_DIAGNOSTIC_SCOPE,
    }:
        raise ValueError("candidate01 Spot receive cannot use a candidate02 tail scope")
    return merger.merge_and_validate_performance_v2(
        candidate_done_paths=candidate_done,
        reference_done_paths=reference_done,
        summary_output_path=summary_output,
        validation_output_path=validation_output,
        scope=scope,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receive-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--expected-run-name")
    parser.add_argument(
        "--scope",
        choices=(
            "auto",
            merger.TAIL_DIAGNOSTIC_SCOPE,
            merger.FULL_PERFORMANCE_SCOPE,
            candidate02_merger.TAIL_DIAGNOSTIC_SCOPE,
            candidate02_tail_v2_merger.TAIL_DIAGNOSTIC_SCOPE,
        ),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, validation = merge_received_spot_v2(
        receive_dir=args.receive_dir,
        summary_output_path=args.summary_output,
        validation_output_path=args.validation_output,
        scope=args.scope,
        expected_run_name=args.expected_run_name,
    )
    print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "main",
    "merge_received_spot_v2",
    "resolve_received_done_paths",
]
