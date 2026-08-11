"""Parallel-20 cloud transport for the immutable M3.1 v1 dataset.

This module deliberately does *not* define a new scientific dataset.  The
dataset plan, 360-shard grid, seed/split schedule, 25-pair shard executor,
portable worker, and final merge remain the v1 implementations.  Only the
post-smoke cloud delivery schedule changes from 8 to 20 C4 Spot VMs.

Every v2 plan source-replays a complete v1 transport plan and records exact
source-module hashes.  Thus a v2 worker still invokes
``hu_m31_t3_dataset_portable_worker_v1`` and produces the same shard bytes a
v1 worker would produce for the same shard directory and immutable inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_executor_v1 as executor_v1
from . import hu_m31_t3_dataset_gcp_transport_v1 as transport_v1
from . import hu_m31_t3_dataset_portable_worker_v1 as worker_v1


TRANSPORT_PLAN_SCHEMA = "hu_m31_t3_dataset_gcp_transport_plan_v2"
ATTEMPT_LEDGER_SCHEMA = "hu_m31_t3_dataset_gcp_attempt_ledger_v2"
RESUME_PLAN_SCHEMA = "hu_m31_t3_dataset_gcp_resume_plan_v2"
WAVE_REQUEST_SCHEMA = "hu_m31_t3_dataset_gcp_wave_request_v2"
LIFECYCLE_SCHEMA = "hu_m31_t3_dataset_gcp_lifecycle_receipt_v2"
RECEIVE_SCHEMA = "hu_m31_t3_dataset_gcp_receive_receipt_v2"
RELOCATION_RECEIPT_SCHEMA = "hu_m31_t3_dataset_source_ext4_relocation_v2"

MAX_CONCURRENT_VMS = 20
VCPUS_PER_VM = transport_v1.VCPUS_PER_VM
MAX_CONCURRENT_VCPUS = MAX_CONCURRENT_VMS * VCPUS_PER_VM
MAX_ATTEMPTS_PER_SHARD = transport_v1.MAX_ATTEMPTS_PER_SHARD
ATTEMPT_IDS = transport_v1.ATTEMPT_IDS
TOTAL_SHARD_COUNT = transport_v1.TOTAL_SHARD_COUNT
CLOUD_SHARD_COUNT = transport_v1.CLOUD_SHARD_COUNT
SHARD_PAIR_COUNT = transport_v1.SHARD_PAIR_COUNT
WAVE_COUNT = 18
FIRST_WAVE_COUNTS = tuple([20] * 17 + [19])
MIN_OAUTH_TTL_SECONDS = 45 * 60

_SHA = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[1-9][0-9]*$")
_EXT4_FILESYSTEM_NAMES = frozenset({"ext2/ext3", "ext4"})
_SOURCE_KINDS = transport_v1.CONTENT_KINDS


def canonical_bytes(value: Any) -> bytes:
    return transport_v1.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return transport_v1.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    return transport_v1.sha256_file(path)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    raw = canonical_bytes(value)
    if target.exists() or target.is_symlink():
        if (
            not target.is_file()
            or target.is_symlink()
            or target.read_bytes() != raw
        ):
            raise FileExistsError(f"immutable artifact conflicts: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _module_sha256(module: Any) -> str:
    source = Path(module.__file__).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("M3.1 v1 source module is missing or unsafe")
    return sha256_file(source)


def _scientific_lock(v1_plan: Mapping[str, Any]) -> dict[str, Any]:
    plan = dataset.build_dataset_plan()
    plan_bytes = dataset.canonical_bytes(plan)
    if (
        dataset.canonical_sha256(plan) != dataset.EXPECTED_DATASET_PLAN_SHA256
        or len(plan["shards"]) != TOTAL_SHARD_COUNT
        or any(row["paired_hand_count"] != SHARD_PAIR_COUNT for row in plan["shards"])
    ):
        raise RuntimeError("M3.1 v1 scientific dataset contract changed")
    return {
        "dataset_plan_schema": plan["schema"],
        "dataset_plan_sha256": dataset.canonical_sha256(plan),
        "dataset_plan_bytes_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "dataset_contract_module_sha256": _module_sha256(dataset),
        "dataset_executor_module_sha256": _module_sha256(executor_v1),
        "portable_worker_module_sha256": _module_sha256(worker_v1),
        "transport_v1_module_sha256": _module_sha256(transport_v1),
        "transport_v1_plan_sha256": v1_plan["plan_sha256"],
        "transport_v1_schema": transport_v1.TRANSPORT_PLAN_SCHEMA,
        "shard_executor": (
            "ofc_regular.hu_m31_t3_dataset_executor_v1.run_dataset_shard"
        ),
        "portable_worker": (
            "ofc_regular.hu_m31_t3_dataset_portable_worker_v1."
            "run_portable_dataset_shard"
        ),
        "merge_implementation": (
            "ofc_regular.hu_m31_t3_dataset_contract_v1.build_merge_manifest"
        ),
        "same_shard_bytes_as_v1": True,
        "same_merge_bytes_as_v1": True,
    }


def _wave_rows(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    waves: list[dict[str, Any]] = []
    for wave_index, start in enumerate(range(0, len(jobs), MAX_CONCURRENT_VMS)):
        ids = [str(row["shard_id"]) for row in jobs[start : start + MAX_CONCURRENT_VMS]]
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
                "one_vm_per_shard": True,
            }
        )
    return waves


def _v1_build_kwargs(plan: Mapping[str, Any]) -> dict[str, Any]:
    by_kind = {row["kind"]: row for row in plan["content_sources"]}
    return {
        "run_name": plan["run_name"],
        "bucket": plan["bucket"],
        "dataset_plan_path": by_kind["dataset_plan"]["source_path"],
        "fresh_quality_gate_path": by_kind["fresh_quality_gate"]["source_path"],
        "smoke_gate_path": by_kind["smoke_gate"]["source_path"],
        "portable_authorization_path": by_kind["portable_authorization"][
            "source_path"
        ],
        "smoke_shard_directory": plan["source_paths"]["smoke_shard_directory"],
        "smoke_shard_archive_path": by_kind["smoke_shard_archive"]["source_path"],
        "runtime_archive_path": by_kind["runtime_archive"]["source_path"],
        "wheelhouse_archive_path": by_kind["wheelhouse_archive"]["source_path"],
        "candidate_library_path": by_kind["candidate_library"]["source_path"],
    }


def _validate_relocation_binding(
    receipt: Mapping[str, Any],
    *,
    v1_plan: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_source_relocation_receipt(receipt)
    expected = {
        row["kind"]: (str(Path(row["source_path"]).resolve()), row["sha256"], row["bytes"])
        for row in v1_plan["content_sources"]
    }
    observed = {
        row["kind"]: (
            row["destination_path"],
            row["sha256"],
            row["bytes"],
        )
        for row in checked["files"]
    }
    if observed != expected:
        raise ValueError("v2 transport inputs differ from ext4 relocation receipt")
    smoke = checked["smoke_shard_directory"]
    if smoke["destination_path"] != str(
        Path(v1_plan["source_paths"]["smoke_shard_directory"]).resolve()
    ):
        raise ValueError("v2 smoke shard differs from ext4 relocation receipt")
    return checked


def build_transport_plan(
    *,
    source_relocation_receipt: Mapping[str, Any],
    **v1_kwargs: Any,
) -> dict[str, Any]:
    """Build a parallel schedule while source-replaying all v1 science."""

    v1_plan = transport_v1.build_transport_plan(**v1_kwargs)
    relocation = _validate_relocation_binding(
        source_relocation_receipt, v1_plan=v1_plan
    )
    jobs = []
    for ordinal, source in enumerate(v1_plan["jobs"]):
        row = deepcopy(dict(source))
        row["wave_index"] = ordinal // MAX_CONCURRENT_VMS
        jobs.append(row)
    waves = _wave_rows(jobs)
    core = {
        **{
            key: deepcopy(value)
            for key, value in v1_plan.items()
            if key not in {"schema", "plan_sha256", "jobs", "waves", "wave_count"}
        },
        "schema": TRANSPORT_PLAN_SCHEMA,
        "status": "qualified_parallel20_ready_cloud_not_started",
        "machine_contract": {
            **deepcopy(v1_plan["machine_contract"]),
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "max_concurrent_vcpus": MAX_CONCURRENT_VCPUS,
            "one_vm_per_shard": True,
        },
        "wave_count": WAVE_COUNT,
        "waves": waves,
        "jobs": jobs,
        "parallel_transport_contract": {
            "version": 2,
            "cloud_fanout_only_changed": True,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "max_concurrent_vcpus": MAX_CONCURRENT_VCPUS,
            "expected_wave_shard_counts": list(FIRST_WAVE_COUNTS),
            "worker_service_account_pool_size": MAX_CONCURRENT_VMS,
            "oauth_minimum_ttl_seconds": MIN_OAUTH_TTL_SECONDS,
            "live_c4_spot_inventory_headroom_required": True,
            "partial_launch_cleanup_required": True,
        },
        "scientific_v1_lock": _scientific_lock(v1_plan),
        "source_ext4_relocation_receipt_sha256": relocation["receipt_sha256"],
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    return {**core, "plan_sha256": canonical_sha256(core)}


def validate_transport_plan(
    value: Mapping[str, Any], *, replay_sources: bool = True
) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    if plan.get("plan_sha256") != _self_digest(plan, "plan_sha256"):
        raise ValueError("parallel20 transport plan digest changed")
    jobs = plan.get("jobs")
    waves = plan.get("waves")
    if (
        plan.get("schema") != TRANSPORT_PLAN_SCHEMA
        or plan.get("status")
        != "qualified_parallel20_ready_cloud_not_started"
        or plan.get("cloud_shard_count") != CLOUD_SHARD_COUNT
        or plan.get("total_dataset_shard_count") != TOTAL_SHARD_COUNT
        or plan.get("wave_count") != WAVE_COUNT
        or not isinstance(jobs, list)
        or len(jobs) != CLOUD_SHARD_COUNT
        or not isinstance(waves, list)
        or len(waves) != WAVE_COUNT
        or plan.get("machine_contract", {}).get("max_concurrent_vms")
        != MAX_CONCURRENT_VMS
        or plan.get("machine_contract", {}).get("max_concurrent_vcpus")
        != MAX_CONCURRENT_VCPUS
        or plan.get("machine_contract", {}).get("one_vm_per_shard") is not True
        or plan.get("parallel_transport_contract", {}).get(
            "cloud_fanout_only_changed"
        )
        is not True
        or plan.get("cloud_execution_started") is not False
        or plan.get("training_eligible") is not False
        or plan.get("promotion_evidence") is not False
        or plan.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 transport boundary changed")
    expected_ids = [
        row["shard_id"]
        for row in dataset.build_dataset_plan()["shards"]
        if row["shard_id"] != dataset.SMOKE_SHARD_ID
    ]
    if [row.get("shard_id") for row in jobs] != expected_ids:
        raise ValueError("parallel20 shard science/order changed")
    for ordinal, job in enumerate(jobs):
        if (
            job.get("ordinal") != ordinal
            or job.get("wave_index") != ordinal // MAX_CONCURRENT_VMS
            or job.get("paired_hand_count") != SHARD_PAIR_COUNT
            or len(job.get("attempts", [])) != MAX_ATTEMPTS_PER_SHARD
        ):
            raise ValueError("parallel20 job binding changed")
    if [row.get("shard_count") for row in waves] != list(FIRST_WAVE_COUNTS):
        raise ValueError("parallel20 wave cardinality changed")
    for index, wave in enumerate(waves):
        expected = expected_ids[
            index * MAX_CONCURRENT_VMS : (index + 1) * MAX_CONCURRENT_VMS
        ]
        if wave != _wave_rows(jobs)[index] or wave["shard_ids"] != expected:
            raise ValueError("parallel20 wave binding changed")
    if replay_sources:
        v1_plan = transport_v1.build_transport_plan(**_v1_build_kwargs(plan))
        lock = _scientific_lock(v1_plan)
        if plan.get("scientific_v1_lock") != lock:
            raise ValueError("parallel20 v1 scientific source lock changed")
        v1_jobs = deepcopy(v1_plan["jobs"])
        normalized = deepcopy(jobs)
        for row in normalized:
            row["wave_index"] = row["ordinal"] // transport_v1.MAX_CONCURRENT_VMS
        if normalized != v1_jobs:
            raise ValueError("parallel20 shard descriptors differ from v1")
    return plan


def _job_by_id(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["shard_id"]: row for row in plan["jobs"]}


def _ledger_from_receipts(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    history: dict[str, list[dict[str, Any]]] = {
        row["shard_id"]: [] for row in plan["jobs"]
    }
    receipt_hashes: list[str] = []
    for raw in receipts:
        receipt = validate_lifecycle_receipt(raw, plan=plan)
        if receipt["receipt_sha256"] in receipt_hashes:
            raise ValueError("parallel20 lifecycle receipt is duplicated")
        receipt_hashes.append(receipt["receipt_sha256"])
        for row in receipt["attempt_rows"]:
            history[row["shard_id"]].append(deepcopy(row))
    rows = []
    for job in plan["jobs"]:
        attempts = history[job["shard_id"]]
        ids = [row["attempt_id"] for row in attempts]
        if ids != list(ATTEMPT_IDS[: len(ids)]):
            raise ValueError("parallel20 attempt order changed")
        if any(
            later["completed_pair_count"] < earlier["completed_pair_count"]
            for earlier, later in zip(attempts, attempts[1:])
        ):
            raise ValueError("parallel20 checkpoint regressed")
        complete = bool(attempts and attempts[-1]["status"] == "ready")
        if any(row["status"] == "ready" for row in attempts[:-1]):
            raise ValueError("parallel20 completed shard was retried")
        count = attempts[-1]["completed_pair_count"] if attempts else 0
        rows.append(
            {
                "shard_id": job["shard_id"],
                "wave_index": job["wave_index"],
                "attempts": attempts,
                "attempt_count": len(attempts),
                "completed_pair_count": count,
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
        "plan_sha256": plan["plan_sha256"],
        "lifecycle_receipt_sha256": receipt_hashes,
        "shards": rows,
        "complete_shard_count": sum(row["status"] == "complete" for row in rows),
        "pending_shard_count": sum(row["status"] == "pending" for row in rows),
        "exhausted_shard_count": sum(row["status"] == "exhausted" for row in rows),
        "cloud_execution_started": bool(receipts),
        "current_profile_changed": False,
    }
    return {**core, "ledger_sha256": canonical_sha256(core)}


def build_attempt_ledger(
    plan: Mapping[str, Any],
    *,
    lifecycle_receipts: Sequence[Mapping[str, Any]] = (),
    lifecycle_receipt_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    values = [deepcopy(dict(row)) for row in lifecycle_receipts]
    values.extend(
        _read_canonical(path, "parallel20 lifecycle receipt")
        for path in lifecycle_receipt_paths
    )
    accepted: list[dict[str, Any]] = []
    for value in values:
        ledger = _ledger_from_receipts(checked, accepted)
        resume = build_resume_plan(checked, ledger)
        request = build_wave_request(checked, ledger, resume)
        accepted.append(
            validate_lifecycle_receipt(
                value,
                plan=checked,
                ledger=ledger,
                resume=resume,
                wave_request=request,
            )
        )
    return _ledger_from_receipts(checked, accepted)


def validate_attempt_ledger(
    plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    observed = deepcopy(dict(value))
    if observed.get("ledger_sha256") != _self_digest(observed, "ledger_sha256"):
        raise ValueError("parallel20 ledger digest changed")
    rows = observed.get("shards")
    if (
        observed.get("schema") != ATTEMPT_LEDGER_SCHEMA
        or observed.get("plan_sha256") != checked["plan_sha256"]
        or not isinstance(rows, list)
        or len(rows) != CLOUD_SHARD_COUNT
        or observed.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 ledger boundary changed")
    return observed


def build_resume_plan(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    state = validate_attempt_ledger(checked, ledger)
    exhausted = [
        row["shard_id"] for row in state["shards"] if row["status"] == "exhausted"
    ]
    by_id = {row["shard_id"]: row for row in state["shards"]}
    wave = next(
        (
            candidate
            for candidate in checked["waves"]
            if any(by_id[shard_id]["status"] != "complete" for shard_id in candidate["shard_ids"])
        ),
        None,
    )
    jobs = _job_by_id(checked)
    selected: list[dict[str, Any]] = []
    if wave is not None and not exhausted:
        for shard_id in wave["shard_ids"]:
            row = by_id[shard_id]
            if row["status"] == "complete":
                continue
            attempt = jobs[shard_id]["attempts"][row["attempt_count"]]
            selected.append(
                {
                    **deepcopy(attempt),
                    "shard_id": shard_id,
                    "resume_completed_pair_count": row["completed_pair_count"],
                    "previous_attempt": (
                        deepcopy(row["attempts"][-1]) if row["attempts"] else None
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
        "one_vm_per_shard": True,
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
        raise ValueError("parallel20 resume plan differs from source replay")
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
        raise PermissionError("parallel20 resume does not authorize a wave")
    core = {
        "schema": WAVE_REQUEST_SCHEMA,
        "status": "qualified_parallel20_wave_cloud_not_started",
        "plan_sha256": checked["plan_sha256"],
        "scientific_v1_plan_sha256": checked["scientific_v1_lock"][
            "transport_v1_plan_sha256"
        ],
        "ledger_sha256": state["ledger_sha256"],
        "resume_sha256": resumed["resume_sha256"],
        "wave_index": resumed["resume_wave_index"],
        "selected_attempts": deepcopy(resumed["selected_attempts"]),
        "selected_count": resumed["selected_count"],
        "machine_type": checked["machine_contract"]["machine_type"],
        "max_vm_count": MAX_CONCURRENT_VMS,
        "required_vcpus": resumed["selected_count"] * VCPUS_PER_VM,
        "one_vm_per_shard": True,
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
    if (
        observed != expected
        or not 1 <= observed["selected_count"] <= MAX_CONCURRENT_VMS
        or observed["required_vcpus"]
        != observed["selected_count"] * VCPUS_PER_VM
    ):
        raise ValueError("parallel20 wave request differs from source replay")
    return observed


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
    rows = receipt.get("attempt_rows")
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != LIFECYCLE_SCHEMA
        or receipt.get("plan_sha256") != checked["plan_sha256"]
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_CONCURRENT_VMS
        or receipt.get("selected_attempt_count") != len(rows)
        or receipt.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or receipt.get("one_vm_per_shard") is not True
        or receipt.get("exact_owned_cleanup") is not True
        or receipt.get("owned_vm_disk_absent") is not True
        or receipt.get("worker_iam_removed_before_receive") is not True
        or receipt.get("partial_launch_cleanup_complete") is not True
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_resource_touched") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 lifecycle receipt boundary changed")
    jobs = _job_by_id(checked)
    v1_plan = transport_v1.build_transport_plan(**_v1_build_kwargs(checked))
    observed_ids: set[str] = set()
    for row in rows:
        shard_id = row.get("shard_id")
        job = jobs.get(str(shard_id))
        attempt = next(
            (
                item
                for item in job["attempts"]
                if item["attempt_id"] == row.get("attempt_id")
            ),
            None,
        ) if job is not None else None
        if (
            job is None
            or attempt is None
            or shard_id in observed_ids
            or row.get("instance_id") != attempt["instance_id"]
            or row.get("status") not in {"ready", "checkpointed", "failed"}
            or not isinstance(row.get("completed_pair_count"), int)
            or not 0 <= row["completed_pair_count"] <= SHARD_PAIR_COUNT
            or (row["status"] == "ready" and row["completed_pair_count"] != SHARD_PAIR_COUNT)
            or row.get("owned_compute_absent") is not True
            or row.get("worker_iam_removed") is not True
        ):
            raise ValueError("parallel20 lifecycle attempt row changed")
        observed_ids.add(str(shard_id))
        checkpoint = row.get("checkpoint")
        heartbeat = row.get("heartbeat")
        if checkpoint is None:
            if row["status"] != "failed" or row["completed_pair_count"] != 0:
                raise ValueError("parallel20 lifecycle lost its checkpoint")
        else:
            if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
                "object",
                "generation",
                "sha256",
                "bytes",
                "manifest",
                "file_objects",
            }:
                raise ValueError("parallel20 checkpoint record changed")
            manifest = transport_v1.validate_checkpoint_manifest(
                checkpoint["manifest"],
                plan=v1_plan,
                shard_id=str(shard_id),
                attempt_id=row["attempt_id"],
            )
            if (
                checkpoint["object"]
                != attempt["checkpoint_object_format"]
                % row["completed_pair_count"]
                or checkpoint["sha256"] != canonical_sha256(manifest)
                or checkpoint["bytes"] != len(canonical_bytes(manifest))
                or _GENERATION.fullmatch(str(checkpoint["generation"])) is None
                or manifest["completed_pair_count"]
                != row["completed_pair_count"]
                or not isinstance(checkpoint["file_objects"], list)
                or len(checkpoint["file_objects"]) != len(manifest["files"])
            ):
                raise ValueError("parallel20 checkpoint binding changed")
        if heartbeat is not None:
            if not isinstance(heartbeat, Mapping) or set(heartbeat) != {
                "object",
                "generation",
                "sha256",
                "bytes",
                "value",
            }:
                raise ValueError("parallel20 heartbeat record changed")
            checked_heartbeat = transport_v1.validate_heartbeat(
                heartbeat["value"],
                plan=v1_plan,
                shard_id=str(shard_id),
                attempt_id=row["attempt_id"],
            )
            if (
                heartbeat["object"]
                != attempt["heartbeat_object_format"]
                % checked_heartbeat["sequence"]
                or heartbeat["sha256"] != canonical_sha256(checked_heartbeat)
                or heartbeat["bytes"] != len(canonical_bytes(checked_heartbeat))
                or _GENERATION.fullmatch(str(heartbeat["generation"])) is None
            ):
                raise ValueError("parallel20 heartbeat binding changed")
    if any(item is None for item in (ledger, resume, wave_request)) and not all(
        item is None for item in (ledger, resume, wave_request)
    ):
        raise ValueError("parallel20 lifecycle context must be all present or absent")
    if ledger is not None:
        assert resume is not None and wave_request is not None
        request = validate_wave_request(checked, ledger, resume, wave_request)
        expected = [
            (row["shard_id"], row["attempt_id"], row["instance_id"])
            for row in request["selected_attempts"]
        ]
        observed = [
            (row["shard_id"], row["attempt_id"], row["instance_id"]) for row in rows
        ]
        if (
            observed != expected
            or receipt.get("ledger_sha256") != ledger["ledger_sha256"]
            or receipt.get("resume_sha256") != resume["resume_sha256"]
            or receipt.get("request_sha256") != request["request_sha256"]
            or receipt.get("wave_index") != request["wave_index"]
        ):
            raise ValueError("parallel20 lifecycle authorization changed")
    return receipt


def build_lifecycle_receipt(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    attempt_rows: Sequence[Mapping[str, Any]],
    partial_launch_cleanup_complete: bool = True,
) -> dict[str, Any]:
    checked = validate_transport_plan(plan, replay_sources=True)
    request = validate_wave_request(checked, ledger, resume, wave_request)
    core = {
        "schema": LIFECYCLE_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "request_sha256": request["request_sha256"],
        "wave_index": request["wave_index"],
        "selected_attempt_count": len(attempt_rows),
        "attempt_rows": [deepcopy(dict(row)) for row in attempt_rows],
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "one_vm_per_shard": True,
        "create_only_gcs": True,
        "exact_owned_cleanup": True,
        "owned_vm_disk_absent": True,
        "worker_iam_removed_before_receive": True,
        "partial_launch_cleanup_complete": partial_launch_cleanup_complete,
        "receiver_handoff_ready": True,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "gcs_evidence_deleted": False,
        "current_profile_changed": False,
    }
    result = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_lifecycle_receipt(
        result,
        plan=checked,
        ledger=ledger,
        resume=resume,
        wave_request=request,
    )


def _copy_create_only(
    source: Path, destination: Path, *, sha256: str, byte_count: int
) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("parallel20 received object is missing or unsafe")
    raw = source.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != sha256
        or len(raw) != byte_count
    ):
        raise ValueError("parallel20 received object bytes changed")
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise FileExistsError("parallel20 local shard checkpoint conflicts")
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
    """Receive exact v1 shard bytes after v2 compute/IAM cleanup."""

    checked = validate_transport_plan(plan, replay_sources=True)
    lifecycle = validate_lifecycle_receipt(lifecycle_receipt, plan=checked)
    remote = Path(object_root).resolve()
    local = Path(local_root).resolve()
    if remote.is_symlink() or not remote.is_dir():
        raise ValueError("parallel20 object mirror is missing or unsafe")
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
                raise ValueError("parallel20 file-object binding changed")
            object_path = remote.joinpath(
                *PurePosixPath(object_record["object"]).parts
            )
            destination = shard_directory.joinpath(
                *PurePosixPath(file_record["relative_path"]).parts
            )
            _copy_create_only(
                object_path,
                destination,
                sha256=file_record["sha256"],
                byte_count=file_record["bytes"],
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
            raise ValueError("parallel20 received shard resume state changed")
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
        "scientific_v1_plan_sha256": checked["scientific_v1_lock"][
            "transport_v1_plan_sha256"
        ],
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
        "source_replayed_v1_shard_bytes": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _filesystem_type(path: Path) -> str:
    if os.name == "nt":
        raise OSError("ext4 relocation must run inside Linux/WSL")
    result = subprocess.run(
        ["stat", "-f", "-c", "%T", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _copy_file_create_only(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise ValueError("relocation source file is missing or unsafe")
    source_sha = sha256_file(source)
    source_bytes = source.stat().st_size
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != source_sha
            or destination.stat().st_size != source_bytes
        ):
            raise FileExistsError("relocation destination conflicts")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            if sha256_file(temporary) != source_sha:
                raise IOError("relocation copy hash changed")
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "source_path": str(source.resolve()),
        "destination_path": str(destination.resolve()),
        "sha256": source_sha,
        "bytes": source_bytes,
        "create_only_or_identical_reuse": True,
    }


def relocate_sources_to_ext4(
    *,
    source_files: Mapping[str, str | Path],
    smoke_shard_directory: str | Path,
    destination_root: str | Path,
) -> dict[str, Any]:
    """Copy immutable v1 inputs to ext4 and emit a source-replay receipt."""

    if set(source_files) != set(_SOURCE_KINDS):
        raise ValueError("relocation source kind set changed")
    destination = Path(destination_root).resolve()
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise ValueError("ext4 relocation root is unsafe")
    destination.mkdir(parents=True, exist_ok=True)
    filesystem = _filesystem_type(destination)
    if filesystem not in _EXT4_FILESYSTEM_NAMES:
        raise ValueError("relocation destination is not ext4")
    rows = []
    for kind in _SOURCE_KINDS:
        source = Path(source_files[kind]).resolve()
        copied = _copy_file_create_only(
            source, destination / "content" / kind / source.name
        )
        rows.append({"kind": kind, **copied})
    smoke_source = Path(smoke_shard_directory).resolve()
    if smoke_source.is_symlink() or not smoke_source.is_dir():
        raise ValueError("smoke shard source directory is unsafe")
    smoke_destination = destination / "smoke_shard"
    files = []
    for source in sorted(path for path in smoke_source.rglob("*") if path.is_file()):
        if source.is_symlink():
            raise ValueError("smoke shard source contains a symlink")
        relative = source.relative_to(smoke_source)
        copied = _copy_file_create_only(source, smoke_destination / relative)
        files.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": copied["sha256"],
                "bytes": copied["bytes"],
            }
        )
    core = {
        "schema": RELOCATION_RECEIPT_SCHEMA,
        "status": "immutable_v1_sources_relocated_to_ext4",
        "destination_root": str(destination),
        "destination_filesystem": filesystem,
        "files": rows,
        "smoke_shard_directory": {
            "source_path": str(smoke_source),
            "destination_path": str(smoke_destination.resolve()),
            "inventory": files,
            "inventory_sha256": canonical_sha256(files),
        },
        "file_count": len(rows) + len(files),
        "all_bytes_hash_replayed": True,
        "create_only_or_identical_reuse": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_source_relocation_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    rows = receipt.get("files")
    smoke = receipt.get("smoke_shard_directory")
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != RELOCATION_RECEIPT_SCHEMA
        or receipt.get("status")
        != "immutable_v1_sources_relocated_to_ext4"
        or receipt.get("destination_filesystem") not in _EXT4_FILESYSTEM_NAMES
        or not isinstance(rows, list)
        or [row.get("kind") for row in rows] != list(_SOURCE_KINDS)
        or not isinstance(smoke, Mapping)
        or receipt.get("all_bytes_hash_replayed") is not True
        or receipt.get("create_only_or_identical_reuse") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("ext4 relocation receipt boundary changed")
    destination_root = Path(str(receipt["destination_root"]))
    if (
        destination_root.is_symlink()
        or not destination_root.is_dir()
        or _filesystem_type(destination_root)
        != receipt["destination_filesystem"]
    ):
        raise ValueError("ext4 relocation filesystem readback changed")
    for row in rows:
        source = Path(row["source_path"])
        destination = Path(row["destination_path"])
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != row["sha256"]
            or destination.stat().st_size != row["bytes"]
            or _SHA.fullmatch(str(row["sha256"])) is None
            or source.resolve() == destination.resolve()
        ):
            raise ValueError("ext4 relocated file changed")
    inventory = smoke.get("inventory")
    if (
        not isinstance(inventory, list)
        or smoke.get("inventory_sha256") != canonical_sha256(inventory)
    ):
        raise ValueError("ext4 smoke inventory changed")
    smoke_root = Path(str(smoke["destination_path"]))
    actual_inventory = []
    for path in sorted(item for item in smoke_root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError("ext4 relocated smoke shard contains a symlink")
        actual_inventory.append(
            {
                "relative_path": path.relative_to(smoke_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if actual_inventory != inventory:
        raise ValueError("ext4 relocated smoke shard inventory changed")
    for row in inventory:
        path = smoke_root.joinpath(*Path(row["relative_path"]).parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            raise ValueError("ext4 relocated smoke shard changed")
    if receipt.get("file_count") != len(rows) + len(inventory):
        raise ValueError("ext4 relocation file count changed")
    return receipt


# The v2 cloud layer intentionally invokes the exact v1 byte producer.
run_portable_dataset_shard = worker_v1.run_portable_dataset_shard
build_merge_manifest = dataset.build_merge_manifest


__all__ = [
    "ATTEMPT_LEDGER_SCHEMA",
    "CLOUD_SHARD_COUNT",
    "LIFECYCLE_SCHEMA",
    "MAX_CONCURRENT_VCPUS",
    "MAX_CONCURRENT_VMS",
    "MIN_OAUTH_TTL_SECONDS",
    "RELOCATION_RECEIPT_SCHEMA",
    "RECEIVE_SCHEMA",
    "RESUME_PLAN_SCHEMA",
    "TOTAL_SHARD_COUNT",
    "TRANSPORT_PLAN_SCHEMA",
    "VCPUS_PER_VM",
    "WAVE_COUNT",
    "WAVE_REQUEST_SCHEMA",
    "build_attempt_ledger",
    "build_lifecycle_receipt",
    "build_merge_manifest",
    "build_resume_plan",
    "build_transport_plan",
    "build_wave_request",
    "canonical_bytes",
    "canonical_sha256",
    "relocate_sources_to_ext4",
    "receive_wave",
    "run_portable_dataset_shard",
    "sha256_file",
    "validate_attempt_ledger",
    "validate_lifecycle_receipt",
    "validate_resume_plan",
    "validate_source_relocation_receipt",
    "validate_transport_plan",
    "validate_wave_request",
]
