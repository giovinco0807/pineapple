"""Cloud-neutral, create-only launch preflight for M3.1 dataset v1.

The preflight can be written before the fresh-quality closeout exists.  In
that state it records one explicit blocker: the source-replayed closeout and
25-pair smoke binding.  Re-running it after that immutable binding exists
validates the prepared controller and emits the exact ``run-next`` argv.

No function in this module opens a cloud adapter or changes an AI profile.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_gcp_controller_v1 as controller
from . import hu_m31_t3_dataset_gcp_transport_v1 as transport
from . import hu_m31_t3_dataset_source_binding_v1 as source_binding
from . import hu_m31_t3_dataset_supervisor_v1 as supervisor
from . import hu_m31_t3_dataset_transport_selection_v1 as selection


PREFLIGHT_SCHEMA = "hu_m31_t3_dataset_launch_preflight_v1"
PENDING_STATUS = "blocked_only_on_fresh_quality_closeout_source_binding"
READY_STATUS = "ready_for_one_supervised_wave_cloud_not_opened"


def canonical_bytes(value: Any) -> bytes:
    return controller.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return controller.canonical_sha256(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return value


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _selection_receipt(path: Path) -> dict[str, Any]:
    receipt = selection.validate_selection_receipt(
        _read_canonical(path, "M3.1 v1 transport selection")
    )
    evidence = _absolute(receipt["evidence_source"], "selection evidence")
    if (
        evidence.is_symlink()
        or not evidence.is_file()
        or _file_sha256(evidence) != receipt["evidence_sha256"]
    ):
        raise ValueError("M3.1 transport selection evidence changed")
    source = supervisor.build_source_receipt()
    observed = {row["relative_path"]: row["observed_sha256"] for row in source["files"]}
    if any(
        observed.get(relative) != digest
        for relative, digest in receipt["source_hashes"].items()
    ):
        raise ValueError("M3.1 transport selection source hash changed")
    return receipt


def _run_next_argv(
    *,
    controller_root: Path,
    supervisor_root: Path,
    source_binding_path: Path,
    dataset_run_name: str,
) -> list[str]:
    return [
        "python",
        "scripts/run_hu_m31_t3_dataset_supervisor_v1.py",
        "run-next",
        "--controller-root",
        str(controller_root),
        "--supervisor-root",
        str(supervisor_root),
        "--confirm-run-name",
        dataset_run_name,
        "--source-binding",
        str(source_binding_path),
        "--confirm-total-cost-cap-usd",
        supervisor.CONFIRM_TOTAL_COST_CAP_USD,
        "--confirm-spot-rate-guard-usd-per-vm-hour",
        supervisor.CONFIRM_SPOT_RATE_GUARD_USD_PER_VM_HOUR,
        "--max-poll-attempts",
        str(supervisor.MAX_POLL_ATTEMPTS),
        "--poll-interval-seconds",
        str(supervisor.DEFAULT_POLL_INTERVAL_SECONDS),
        "--allow-cloud-mutation",
    ]


def build_launch_preflight(
    *,
    dataset_run_name: str,
    selection_receipt_path: str | Path,
    planned_source_binding_path: str | Path,
    planned_controller_root: str | Path,
    planned_supervisor_root: str | Path,
) -> dict[str, Any]:
    if not isinstance(dataset_run_name, str) or not dataset_run_name:
        raise ValueError("M3.1 dataset run name is invalid")
    selected_path = _absolute(selection_receipt_path, "selection receipt")
    selected = _selection_receipt(selected_path)
    binding_path = _absolute(planned_source_binding_path, "planned source binding")
    controller_root = _absolute(planned_controller_root, "planned controller root")
    supervisor_root = _absolute(planned_supervisor_root, "planned supervisor root")

    plan = dataset.validate_dataset_plan(dataset.build_dataset_plan())
    if (
        plan["paired_hand_count"] != 9_000
        or plan["root_count"] != 18_000
        or plan["shard_contract"]["shard_count"] != 360
        or plan["shard_contract"]["paired_hands_per_shard"] != 25
        or plan["smoke_gate"]["paired_hand_count"] != 25
        or len(plan["shards"]) != 360
        or transport.CLOUD_SHARD_COUNT != 359
        or transport.MAX_CONCURRENT_VMS != 8
        or transport.WAVE_COUNT != 45
        or supervisor.MAX_SUPERVISED_LIFECYCLES != 90
    ):
        raise ValueError("M3.1 dataset launch boundary changed")

    blockers: list[str] = []
    binding_sha256: str | None = None
    controller_status: dict[str, Any] | None = None
    if binding_path.exists() or binding_path.is_symlink():
        bound = source_binding.validate_source_binding_file(
            binding_path,
            expected_dataset_run_name=dataset_run_name,
            expected_controller_root=controller_root,
        )
        binding_sha256 = bound["binding_sha256"]
        controller_status = controller.controller_status(controller_root)
        if (
            controller_status["resume_status"] != "wave_ready"
            or controller_status["accepted_lifecycle_count"] != 0
            or controller_status["complete_shard_count"] != 0
            or controller_status["exhausted_shard_count"] != 0
            or controller_status["selected_count"] != 8
        ):
            raise PermissionError(
                "M3.1 prepared controller is not at the exact first cloud wave"
            )
    else:
        blockers.append(
            "fresh_quality_closeout_and_25_pair_smoke_source_binding_missing"
        )

    ready = not blockers
    source_receipt = supervisor.build_source_receipt()
    run_next = _run_next_argv(
        controller_root=controller_root,
        supervisor_root=supervisor_root,
        source_binding_path=binding_path,
        dataset_run_name=dataset_run_name,
    )
    core = {
        "schema": PREFLIGHT_SCHEMA,
        "status": READY_STATUS if ready else PENDING_STATUS,
        "dataset_run_name": dataset_run_name,
        "selection_receipt_path": str(selected_path),
        "selection_receipt_file_sha256": _file_sha256(selected_path),
        "selection_receipt_sha256": selected["receipt_sha256"],
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_binding_path": str(binding_path),
        "source_binding_sha256": binding_sha256,
        "planned_controller_root": str(controller_root),
        "planned_supervisor_root": str(supervisor_root),
        "controller_status": controller_status,
        "scientific_boundary": {
            "paired_hand_count": 9_000,
            "root_count": 18_000,
            "paired_hands_per_shard": 25,
            "total_shard_count": 360,
            "completed_local_smoke_shard_count": 1,
            "cloud_shard_count": 359,
            "dataset_plan_sha256": canonical_sha256(plan),
        },
        "transport_boundary": {
            "machine_type": transport.MACHINE_TYPE,
            "maximum_concurrent_vms": 8,
            "maximum_concurrent_vcpus": 128,
            "wave_count": 45,
            "maximum_accepted_lifecycles": 90,
            "minimum_oauth_ttl_seconds": supervisor.MIN_OAUTH_TTL_SECONDS,
            "one_wave_only": True,
            "create_only": True,
            "exact_vm_disk_iam_cleanup_required": True,
            "source_replayed_receive_required": True,
        },
        "cost_boundary": source_receipt["cost_contract"],
        "merge_boundary": {
            "finalize_requires_complete_cloud_shard_count": 359,
            "finalize_requires_exhausted_shard_count": 0,
            "exact_360_shard_map_required": True,
            "duplicate_or_gap_allowed": False,
            "training_eligible_before_merge": False,
        },
        "storage_boundary": {
            "production_temp_staging_source_replay_filesystem": (
                "native_ext4_under_/home/wner"
            ),
            "mnt_d_allowed_usage": (
                "final_receipts_and_final_artifacts_create_only_only"
            ),
            "production_temp_on_mnt_d_allowed": False,
            "production_staging_on_mnt_d_allowed": False,
            "production_source_replay_on_mnt_d_allowed": False,
            "wsl_vhd_physical_backing": "D:",
        },
        "blocking_reasons": blockers,
        "ready_to_run_next": ready,
        "run_next_argv": run_next,
        "cloud_opened": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_launch_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    digest = receipt.pop("receipt_sha256", None)
    if (
        digest != canonical_sha256(receipt)
        or receipt.get("schema") != PREFLIGHT_SCHEMA
        or receipt.get("status") not in {PENDING_STATUS, READY_STATUS}
        or receipt.get("ready_to_run_next")
        is not (receipt.get("status") == READY_STATUS)
        or bool(receipt.get("blocking_reasons")) is receipt.get("ready_to_run_next")
        or receipt.get("cloud_opened") is not False
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("M3.1 dataset launch preflight changed")
    return {**receipt, "receipt_sha256": digest}


def write_launch_preflight(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    target = _absolute(output_path, "launch preflight output")
    receipt = build_launch_preflight(**kwargs)
    raw = canonical_bytes(receipt)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise FileExistsError(f"immutable launch preflight conflicts: {target}")
    else:
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
    return validate_launch_preflight(
        _read_canonical(target, "M3.1 dataset launch preflight")
    )


__all__ = [
    "PENDING_STATUS",
    "PREFLIGHT_SCHEMA",
    "READY_STATUS",
    "build_launch_preflight",
    "canonical_bytes",
    "canonical_sha256",
    "validate_launch_preflight",
    "write_launch_preflight",
]
