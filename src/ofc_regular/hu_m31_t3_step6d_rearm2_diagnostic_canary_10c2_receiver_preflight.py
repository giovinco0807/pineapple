"""Local-only 10c.2 receiver and read-only cloud-preflight contract.

This module deliberately has no GCP SDK, HTTP, subprocess, authentication,
claim, authorization, or VM-create surface.  It consumes an injected
read-only object backend, verifies the exact runner-tree inventory, writes a
fresh local materialization, and calls the independent Step 6d runner
validator.  The preflight helpers only validate externally supplied
observations; they never fetch or invent quota or Spot price data.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1 as direct_transport,
)
from . import run_hu_m31_t3_step6d_performance_v2 as runner


INVENTORY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_remote_inventory_v1"
)
MATERIALIZATION_RESULT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_materialization_result_v1"
)
VM_ABSENCE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_vm_absence_observation_v1"
)
CONTROLLER_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_controller_receipt_v1"
)
PREFLIGHT_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_read_only_preflight_plan_v1"
)
PREFIX_OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_prefix_observation_v1"
)
CAPACITY_OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_capacity_observation_v1"
)
PRICE_OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_spot_price_observation_v1"
)
INSTANCE_OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_instance_observation_v1"
)
OUTER_PACKAGE_MANIFEST_SCHEMA = direct_transport.OUTER_MANIFEST_SCHEMA
DIRECT_STAGE_IDENTITY_SCHEMA = direct_transport.DIRECT_STAGE_IDENTITY_SCHEMA
PREFLIGHT_RESULT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_read_only_preflight_result_v1"
)

MACHINE_TYPE = "c4-standard-16"
VCPU_PER_VM = 16
SPOT_PRICE_CEILING_USD_PER_VM_HOUR = (
    adapter.legacy_vm.SPOT_PRICE_CEILING_USD_PER_VM_HOUR
)
DIRECT_V1_NAMESPACE = "hu-m31-r2diag-direct-v1"
CAPACITY_PROVIDER_METRIC_MAPPING = (
    "floor(min("
    "region.quotas[C4_CPUS].limit-usage,"
    "region.quotas[PREEMPTIBLE_CPUS].limit-usage,"
    "project.quotas[CPUS_ALL_REGIONS].limit-usage"
    "))"
)
REQUIRED_WORKER_OAUTH_SCOPE = (
    "https://www.googleapis.com/auth/cloud-platform"
)
PREFIX_OBSERVATION_MAX_AGE_SECONDS = 300
CAPACITY_OBSERVATION_MAX_AGE_SECONDS = 300
PRICE_OBSERVATION_MAX_AGE_SECONDS = 3600
INSTANCE_OBSERVATION_MAX_AGE_SECONDS = 300
DIRECT_API_ALLOWLIST = (
    "compute.instances.get",
    "compute.instances.list",
    "storage.objects.get",
    "storage.objects.list",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")


class ReadOnlyObjectBackend(Protocol):
    """The receiver intentionally exposes no write method."""

    backend_id: str
    fixture_only: bool

    def list_prefix(self, prefix: str) -> Sequence[Mapping[str, Any]]:
        """Return generation-bound metadata for every object below ``prefix``."""

    def read_bytes(self, uri: str, generation: int) -> bytes:
        """Read the exact immutable generation returned by ``list_prefix``."""


def canonical_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed")


def _strict_bool(value: Any, expected: bool | None, label: str) -> bool:
    if type(value) is not bool or (expected is not None and value is not expected):
        raise ValueError(f"{label} must be a strict boolean")
    return value


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{label} must be an exact bounded integer")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero SHA-256")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{label} is not a finite nonnegative number")
    return float(value)


def _safe_relative(path: Any) -> str:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith(("/", "\\"))
        or "\\" in path
        or ".." in Path(path).parts
        or Path(path).is_absolute()
    ):
        raise ValueError("remote inventory path escaped the materialization root")
    return path


def _reject_scientific_evidence(value: Any, path: str = "$") -> None:
    forbidden = (
        "opponent_private_discard",
        "hidden_truth",
        "teacher_ev",
        "q_value",
        "action_value",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden):
                raise ValueError(f"forbidden scientific field at {path}.{key}")
            _reject_scientific_evidence(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_scientific_evidence(child, f"{path}[{index}]")


def build_outer_package_manifest(
    *,
    package_dir: str | Path,
    offline_wheel_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return direct_transport.build_outer_package_manifest(
        package_dir=package_dir,
        offline_wheel_record=offline_wheel_record,
    )


def validate_outer_package_manifest(
    value: Mapping[str, Any], *, preview: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = direct_transport.validate_outer_package_manifest(value)
    if (
        manifest["inner_package_manifest_sha256"]
        != preview["package_manifest_sha256"]
    ):
        raise ValueError("outer package lost exact inner package binding")
    return manifest


def build_direct_stage_identity(
    preview: Mapping[str, Any],
    *,
    outer_package_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    outer = validate_outer_package_manifest(
        outer_package_manifest, preview=preview
    )
    return direct_transport.build_direct_stage_identity(
        preview=preview,
        outer_manifest=outer,
    )


def validate_direct_stage_identity(
    value: Mapping[str, Any],
    *,
    preview: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_direct_stage_identity(
        preview, outer_package_manifest=outer_package_manifest
    )
    identity = dict(value)
    if identity != expected:
        raise ValueError("direct stage identity changed")
    return expected


def build_direct_v1_remote_layout(
    preview: Mapping[str, Any],
    *,
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Remap legacy preview URIs into collision-separated direct-v1 prefixes."""

    if (outer_package_manifest is None) != (direct_stage_identity is None):
        raise ValueError(
            "outer package and direct stage identity must be supplied together"
        )
    real_eligible = outer_package_manifest is not None
    if real_eligible:
        outer = validate_outer_package_manifest(
            outer_package_manifest, preview=preview
        )
        direct_identity = validate_direct_stage_identity(
            direct_stage_identity,
            preview=preview,
            outer_package_manifest=outer,
        )
        canonical_layout = direct_transport.build_direct_v1_remote_layout(
            preview=preview,
            outer_manifest=outer,
        )
        if canonical_layout["direct_stage_identity"] != direct_identity:
            raise ValueError("transport direct stage identity changed")
        package_sha = canonical_layout["package_inventory"][
            "outer_package_identity_sha256"
        ]
        stage_sha = canonical_layout["direct_stage_identity_sha256"]
        source_objects = canonical_layout["package_inventory"]["records"]
    else:
        outer = None
        direct_identity = None
        package_sha = _sha(
            preview["package_manifest_sha256"], "package manifest sha256"
        )
        stage_sha = _sha(
            preview["stage_identity_sha256"], "stage identity sha256"
        )
        source_objects = preview["package_files"]
        canonical_layout = None
    run_name = preview["run_name"]
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("direct-v1 run name changed")
    base = f"gs://{adapter.DEFAULT_BUCKET}/{DIRECT_V1_NAMESPACE}"
    package_prefix = f"{base}/packages/{package_sha}"
    stage_prefix = f"{base}/stages/{run_name}/{stage_sha}"
    result_prefix = f"{stage_prefix}/results"
    attempt_control_prefix = (
        f"{stage_prefix}/control/attempt-{preview['attempt_index']}"
    )
    package_objects = [
        {
            "path": row["path"],
            "uri": row.get("uri", f"{package_prefix}/{row['path']}"),
            "legacy_preview_uri": next(
                (
                    legacy["uri"]
                    for legacy in preview["remote_manifest"][
                        "control_objects"
                    ]
                    if legacy["path"] == row["path"]
                ),
                None,
            ),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in source_objects
    ]
    stage_tag = (
        "s1"
        if preview["stage_id"].startswith("stage1_")
        else "s2" if preview["stage_id"].startswith("stage2_") else None
    )
    if stage_tag is None:
        raise ValueError("direct-v1 stage tag escaped fixed diagnostic stages")
    job_layouts = [
        {
            "job_id": job["job_id"],
            "instance_name": direct_transport.deterministic_instance_name(
                stage_id=preview["stage_id"],
                job_id=job["job_id"],
                attempt_index=preview["attempt_index"],
                preview_stage_identity_sha256=preview[
                    "stage_identity_sha256"
                ],
            ),
            "result_prefix": f"{result_prefix}/jobs/{job['job_id']}",
            "tree_prefix": f"{result_prefix}/jobs/{job['job_id']}/tree",
            "done_envelope_uri": (
                f"{result_prefix}/jobs/{job['job_id']}/DONE.envelope.json"
            ),
            "legacy_result_prefix": job["tree_prefix"].rsplit("/tree", 1)[0],
            "legacy_tree_prefix": job["tree_prefix"],
        }
        for job in preview["jobs"]
    ]
    if any(
        _SAFE_NAME.fullmatch(row["instance_name"]) is None
        or len(row["instance_name"]) > 63
        for row in job_layouts
    ) or len({row["instance_name"] for row in job_layouts}) != len(job_layouts):
        raise ValueError("direct-v1 expected instance names collide or are unsafe")
    if canonical_layout is not None and (
        canonical_layout["base_prefix"] != base
        or canonical_layout["package_prefix"] != package_prefix
        or canonical_layout["stage_prefix"] != stage_prefix
        or canonical_layout["result_prefix"] != result_prefix
        or canonical_layout["attempt_control_prefix"]
        != attempt_control_prefix
        or [
            (row["job_id"], row["tree_prefix"])
            for row in canonical_layout["jobs"]
        ]
        != [(row["job_id"], row["tree_prefix"]) for row in job_layouts]
    ):
        raise ValueError("receiver layout diverged from canonical transport")
    layout = {
        "schema": "hu_m31_t3_step6d_rearm2_diagnostic_10c2_direct_v1_layout_v1",
        "base_prefix": base,
        "package_prefix": package_prefix,
        "inner_package_manifest_sha256": preview[
            "package_manifest_sha256"
        ],
        "outer_package_identity_sha256": (
            outer["outer_package_identity_sha256"] if outer else None
        ),
        "package_objects": package_objects,
        "package_objects_sha256": canonical_sha256(package_objects),
        "stage_prefix": stage_prefix,
        "preview_stage_identity_sha256": preview[
            "stage_identity_sha256"
        ],
        "direct_stage_identity_sha256": (
            direct_identity["direct_stage_identity_sha256"]
            if direct_identity
            else None
        ),
        "result_prefix": result_prefix,
        "attempt_control_prefix": attempt_control_prefix,
        "receive_uri": f"{result_prefix}/received/{preview['stage_id']}.json",
        "job_layouts": job_layouts,
        "job_layouts_sha256": canonical_sha256(job_layouts),
        "package_and_stage_prefix_disjoint": True,
        "legacy_preview_uris_are_identity_provenance_only": True,
        "outer_package_manifest_bound": real_eligible,
        "direct_stage_identity_bound": real_eligible,
        "real_read_only_preflight_contract_eligible": real_eligible,
        "legacy_local_contract_only": not real_eligible,
        "remote_read_performed": False,
        "remote_write_performed": False,
        "cloud_mutation_performed": False,
    }
    if (
        package_prefix.startswith(stage_prefix + "/")
        or stage_prefix.startswith(package_prefix + "/")
        or len({row["uri"] for row in package_objects})
        != len(package_objects)
    ):
        raise ValueError("direct-v1 package/stage namespace collision")
    return layout


def build_remote_object_inventory(
    preview: Mapping[str, Any],
    *,
    receive: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every received runner tree to an exact path/URI/hash/byte list."""

    checked_receive = adapter.validate_receive(receive, preview=preview)
    direct_layout = build_direct_v1_remote_layout(
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    jobs: list[dict[str, Any]] = []
    all_uris: list[str] = []
    for done in checked_receive["done_records"]:
        materialization = adapter.build_materialization_manifest(
            preview, done_record=done
        )
        direct_job = next(
            row
            for row in direct_layout["job_layouts"]
            if row["job_id"] == done["job_id"]
        )
        objects: list[dict[str, Any]] = []
        for position, raw in enumerate(materialization["objects"]):
            row = dict(raw)
            _exact(row, {"path", "uri", "sha256", "bytes"}, "tree object")
            path = _safe_relative(row["path"])
            legacy_uri = row["uri"]
            uri = f"{direct_job['tree_prefix']}/{path}"
            if (
                not isinstance(legacy_uri, str)
                or legacy_uri
                != f"{direct_job['legacy_tree_prefix']}/{path}"
            ):
                raise ValueError("legacy preview URI escaped the exact tree prefix")
            objects.append(
                {
                    "position": position,
                    "path": path,
                    "uri": uri,
                    "legacy_preview_uri": legacy_uri,
                    "sha256": _sha(row["sha256"], "tree object sha256"),
                    "bytes": _strict_int(
                        row["bytes"], "tree object bytes", minimum=1
                    ),
                }
            )
            all_uris.append(uri)
        jobs.append(
            {
                "job_id": materialization["job_id"],
                "source_role": materialization["source_role"],
                "runner_job_manifest_sha256": materialization[
                    "runner_job_manifest_sha256"
                ],
                "tree_prefix": direct_job["tree_prefix"],
                "legacy_preview_tree_prefix": direct_job[
                    "legacy_tree_prefix"
                ],
                "objects": objects,
                "objects_sha256": canonical_sha256(objects),
                "object_count": len(objects),
            }
        )
    if [row["job_id"] for row in jobs] != preview["selected_job_ids"]:
        raise ValueError("remote inventory job order changed")
    if len(all_uris) != len(set(all_uris)):
        raise ValueError("remote inventory contains a URI collision")
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "status": "exact_remote_runner_tree_inventory_ready_for_read_only_receive",
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "direct_v1_layout_sha256": canonical_sha256(direct_layout),
        "selected_job_ids": list(preview["selected_job_ids"]),
        "receive_sha256": canonical_sha256(checked_receive),
        "jobs": jobs,
        "jobs_sha256": canonical_sha256(jobs),
        "object_count": len(all_uris),
        "all_uris_unique": True,
        "exact_prefix_inventory_required": True,
        "download_bytes_and_hash_required": True,
        "fresh_local_materialization_required": True,
        "runner_validate_completed_output_required": True,
        "identity_envelope_is_not_content_proof": True,
        "remote_read_performed": False,
        "remote_write_performed": False,
        "cloud_mutation_performed": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    _reject_scientific_evidence(inventory)
    return inventory


def validate_remote_object_inventory(
    value: Mapping[str, Any],
    *,
    preview: Mapping[str, Any],
    receive: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = build_remote_object_inventory(
        preview,
        receive=receive,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    if dict(value) != expected:
        raise ValueError("remote object inventory changed")
    return expected


def _validate_backend(backend: ReadOnlyObjectBackend) -> tuple[str, bool]:
    backend_id = getattr(backend, "backend_id", None)
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or len(backend_id) > 128
    ):
        raise ValueError("read-only backend identity changed")
    fixture_only = _strict_bool(
        getattr(backend, "fixture_only", None), None, "backend fixture_only"
    )
    return backend_id, fixture_only


def _listed_object_record(
    value: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(value)
    _exact(
        row,
        {
            "uri",
            "generation",
            "metageneration",
            "bytes",
            "sha256",
            "crc32c",
            "etag",
        },
        "generation-bound object listing",
    )
    if (
        row["uri"] != expected["uri"]
        or row["bytes"] != expected["bytes"]
        or row["sha256"] != expected["sha256"]
    ):
        raise ValueError("listed object metadata changed expected content identity")
    _strict_int(row["generation"], "object generation", minimum=1)
    _strict_int(row["metageneration"], "object metageneration", minimum=1)
    _strict_int(row["bytes"], "listed object bytes", minimum=1)
    _sha(row["sha256"], "listed object sha256")
    for field in ("crc32c", "etag"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"listed object {field} is missing")
    return row


def _fsync_directory(path: Path) -> None:
    """Durably flush a real directory without silently skipping Windows."""

    if not path.is_dir() or path.is_symlink():
        raise ValueError("directory fsync target must be an existing real directory")
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        flush_file_buffers = kernel32.FlushFileBuffers
        flush_file_buffers.argtypes = [wintypes.HANDLE]
        flush_file_buffers.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        generic_write = 0x40000000
        share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
        open_existing = 3
        file_flag_backup_semantics = 0x02000000
        handle = create_file(
            str(path.resolve(strict=True)),
            generic_write,
            share_read_write_delete,
            None,
            open_existing,
            file_flag_backup_semantics,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        flush_error: BaseException | None = None
        try:
            if not flush_file_buffers(handle):
                flush_error = ctypes.WinError(ctypes.get_last_error())
        finally:
            if not close_handle(handle) and flush_error is None:
                flush_error = ctypes.WinError(ctypes.get_last_error())
        if flush_error is not None:
            raise flush_error
        return

    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "receiver directory fsync is implemented only for Windows and Linux"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    """Flush every created directory entry from leaves through ``root``."""

    directories: list[Path] = []
    for current, names, _ in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            raise ValueError("receiver materialization contains a directory symlink")
        directories.append(current_path)
        for name in names:
            child = current_path / name
            if child.is_symlink():
                raise ValueError("receiver materialization contains a directory symlink")
    for directory in reversed(directories):
        _fsync_directory(directory)


def _finalize_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically move ``source`` while refusing to replace ``destination``."""

    source_parent = source.parent.resolve(strict=True)
    destination_parent = destination.parent.resolve(strict=True)
    if source_parent != destination_parent:
        raise ValueError("receiver finalization must remain in one parent directory")

    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
        move_file_ex.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
        ]
        move_file_ex.restype = wintypes.BOOL
        movefile_write_through = 0x00000008
        if move_file_ex(
            str(source.resolve(strict=True)),
            str(destination_parent / destination.name),
            movefile_write_through,
        ):
            return
        error_code = ctypes.get_last_error()
        if error_code in {80, 183}:
            raise FileExistsError(
                error_code,
                "receiver destination appeared during no-replace finalization",
                str(destination),
            )
        raise ctypes.WinError(error_code)

    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "receiver no-replace finalization is implemented only for Windows and Linux"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe fallback"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    ctypes.set_errno(0)
    status = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if status == 0:
        return
    error_code = ctypes.get_errno()
    if error_code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_code,
            "receiver destination appeared during no-replace finalization",
            str(destination),
        )
    if error_code in {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }:
        raise RuntimeError(
            "renameat2(RENAME_NOREPLACE) is unsupported; refusing unsafe fallback"
        ) from OSError(error_code, os.strerror(error_code))
    raise OSError(error_code, os.strerror(error_code), str(destination))


def _remove_owned_hidden_staging(staging: Path, parent: Path) -> None:
    """Remove only a staging tree this receiver created, then flush its parent."""

    if staging.is_symlink():
        staging.unlink()
    elif staging.exists():
        if not staging.is_dir():
            staging.unlink()
        else:
            shutil.rmtree(staging)
    if staging.exists() or staging.is_symlink():
        raise RuntimeError("receiver failed to remove its hidden staging artifact")
    _fsync_directory(parent)


def materialize_and_validate_received_stage(
    preview: Mapping[str, Any],
    *,
    receive: Mapping[str, Any],
    destination_root: str | Path,
    backend: ReadOnlyObjectBackend,
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Download, verify, freshly materialize, and independently validate a stage."""

    inventory = build_remote_object_inventory(
        preview,
        receive=receive,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    backend_id, fixture_only = _validate_backend(backend)
    destination = Path(destination_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("receiver destination must be fresh and absent")
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("receiver destination parent must be an existing real directory")
    staging = parent / (
        f".{destination.name}.10c2-"
        f"{canonical_sha256(inventory)[:12]}.staging"
    )
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("receiver hidden staging destination is not fresh")

    # Fail before the first local write on unknown/missing remote objects or
    # corrupted content.  This also makes a retry select a new fresh root.
    downloaded: list[
        tuple[str, list[tuple[dict[str, Any], dict[str, Any], bytes]]]
    ] = []
    for job in inventory["jobs"]:
        listed_raw = list(backend.list_prefix(job["tree_prefix"]))
        if any(not isinstance(row, Mapping) for row in listed_raw):
            raise TypeError("read-only backend listing must return metadata records")
        listed_by_uri = {row.get("uri"): row for row in listed_raw}
        if (
            len(listed_by_uri) != len(listed_raw)
            or set(listed_by_uri)
            != {row["uri"] for row in job["objects"]}
        ):
            raise ValueError("remote tree prefix inventory has missing or unknown objects")
        job_bytes: list[
            tuple[dict[str, Any], dict[str, Any], bytes]
        ] = []
        for row in job["objects"]:
            listed = _listed_object_record(
                listed_by_uri[row["uri"]], expected=row
            )
            raw = backend.read_bytes(row["uri"], listed["generation"])
            if not isinstance(raw, bytes):
                raise TypeError("read-only backend must return exact bytes")
            if len(raw) != row["bytes"]:
                raise ValueError("downloaded remote object byte length changed")
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("downloaded remote object SHA-256 changed")
            job_bytes.append((row, listed, raw))
        downloaded.append((job["job_id"], job_bytes))

    validated_jobs: list[dict[str, Any]] = []
    staging_owned = False
    try:
        staging.mkdir()
        staging_owned = True
        _fsync_directory(parent)
        for job_id, job_bytes in downloaded:
            job_dir = staging / "jobs" / job_id
            job_dir.mkdir(parents=True)
            for row, _, raw in job_bytes:
                target = job_dir / row["path"]
                resolved_parent = target.parent.resolve()
                job_root = job_dir.resolve()
                if (
                    resolved_parent != job_root
                    and job_root not in resolved_parent.parents
                ):
                    raise ValueError("receiver write escaped the fresh job root")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                checked = target.read_bytes()
                if (
                    len(checked) != row["bytes"]
                    or hashlib.sha256(checked).hexdigest() != row["sha256"]
                ):
                    raise ValueError(
                        "fresh local materialization changed object bytes"
                    )
            validated_done = dict(runner.validate_completed_output(job_dir))
            listed_records = [listed for _, listed, _ in job_bytes]
            validated_jobs.append(
                {
                    "job_id": job_id,
                    "relative_directory": f"jobs/{job_id}",
                    "object_count": len(job_bytes),
                    "objects_sha256": canonical_sha256(
                        [row for row, _, _ in job_bytes]
                    ),
                    "listed_generations_sha256": canonical_sha256(
                        listed_records
                    ),
                    "runner_done_sha256": canonical_sha256(validated_done),
                    "runner_validate_completed_output_called": True,
                }
            )
        _fsync_directory_tree(staging)
        _fsync_directory(parent)
        _finalize_directory_noreplace(staging, destination)
        staging_owned = False
        _fsync_directory(destination)
        _fsync_directory(parent)
    except BaseException:
        if staging_owned:
            _remove_owned_hidden_staging(staging, parent)
        raise

    result = {
        "schema": MATERIALIZATION_RESULT_SCHEMA,
        "status": "all_runner_trees_downloaded_hashed_freshly_materialized_and_validated",
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "selected_job_ids": list(preview["selected_job_ids"]),
        "receive_sha256": inventory["receive_sha256"],
        "inventory_sha256": canonical_sha256(inventory),
        "backend_id": backend_id,
        "backend_fixture_only": fixture_only,
        "generation_bound_reads_performed": True,
        "external_cloud_read_performed": False,
        "signed_external_query_receipt_present": False,
        "contract_or_fake_evidence_only": True,
        "validated_jobs": validated_jobs,
        "validated_jobs_sha256": canonical_sha256(validated_jobs),
        "download_bytes_and_hash_verified": True,
        "exact_prefix_inventory_verified": True,
        "fresh_local_materialization_performed": True,
        "exclusive_hidden_staging_performed": True,
        "file_fsync_performed": True,
        "directory_fsync_performed": True,
        "atomic_rename_performed": True,
        "destination_noreplace_finalize_performed": True,
        "staging_artifact_remaining": False,
        "runner_validate_completed_output_performed": True,
        "remote_write_performed": False,
        "cloud_mutation_performed": False,
        "claim_created": False,
        "authorization_created": False,
        "vm_created": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    _reject_scientific_evidence(result)
    return result


def validate_materialization_result(
    value: Mapping[str, Any],
    *,
    preview: Mapping[str, Any],
    receive: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(value)
    _exact(
        result,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_identity_sha256",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "receive_sha256",
            "inventory_sha256",
            "backend_id",
            "backend_fixture_only",
            "generation_bound_reads_performed",
            "external_cloud_read_performed",
            "signed_external_query_receipt_present",
            "contract_or_fake_evidence_only",
            "validated_jobs",
            "validated_jobs_sha256",
            "download_bytes_and_hash_verified",
            "exact_prefix_inventory_verified",
            "fresh_local_materialization_performed",
            "exclusive_hidden_staging_performed",
            "file_fsync_performed",
            "directory_fsync_performed",
            "atomic_rename_performed",
            "destination_noreplace_finalize_performed",
            "staging_artifact_remaining",
            "runner_validate_completed_output_performed",
            "remote_write_performed",
            "cloud_mutation_performed",
            "claim_created",
            "authorization_created",
            "vm_created",
            "diagnostic_only",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
        },
        "materialization result",
    )
    inventory = build_remote_object_inventory(
        preview,
        receive=receive,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    if (
        result["schema"] != MATERIALIZATION_RESULT_SCHEMA
        or result["status"]
        != "all_runner_trees_downloaded_hashed_freshly_materialized_and_validated"
        or result["package_manifest_sha256"] != preview["package_manifest_sha256"]
        or result["stage_identity_sha256"] != preview["stage_identity_sha256"]
        or result["stage_id"] != preview["stage_id"]
        or result["run_name"] != preview["run_name"]
        or result["selected_job_ids"] != preview["selected_job_ids"]
        or result["receive_sha256"] != inventory["receive_sha256"]
        or result["inventory_sha256"] != canonical_sha256(inventory)
        or result["validated_jobs_sha256"]
        != canonical_sha256(result["validated_jobs"])
        or [row.get("job_id") for row in result["validated_jobs"]]
        != preview["selected_job_ids"]
    ):
        raise ValueError("materialization result identity changed")
    for field in (
        "download_bytes_and_hash_verified",
        "exact_prefix_inventory_verified",
        "fresh_local_materialization_performed",
        "exclusive_hidden_staging_performed",
        "file_fsync_performed",
        "directory_fsync_performed",
        "atomic_rename_performed",
        "destination_noreplace_finalize_performed",
        "runner_validate_completed_output_performed",
        "diagnostic_only",
    ):
        _strict_bool(result[field], True, f"materialization {field}")
    for field in (
        "remote_write_performed",
        "cloud_mutation_performed",
        "external_cloud_read_performed",
        "signed_external_query_receipt_present",
        "staging_artifact_remaining",
        "claim_created",
        "authorization_created",
        "vm_created",
        "performance_lock_evidence",
        "quality_evidence",
        "training_eligible",
        "promotion_evidence",
    ):
        _strict_bool(result[field], False, f"materialization {field}")
    for field in (
        "generation_bound_reads_performed",
        "contract_or_fake_evidence_only",
    ):
        _strict_bool(result[field], True, f"materialization {field}")
    _strict_bool(
        result["backend_fixture_only"], None, "materialization backend fixture"
    )
    for row in result["validated_jobs"]:
        _exact(
            row,
            {
                "job_id",
                "relative_directory",
                "object_count",
                "objects_sha256",
                "listed_generations_sha256",
                "runner_done_sha256",
                "runner_validate_completed_output_called",
            },
            "validated materialization job",
        )
        _strict_bool(
            row["runner_validate_completed_output_called"],
            True,
            "runner validator call",
        )
        _strict_int(row["object_count"], "validated object count", minimum=1)
        _sha(row["objects_sha256"], "validated objects")
        _sha(row["listed_generations_sha256"], "listed generations")
        _sha(row["runner_done_sha256"], "validated runner DONE")
        inventory_job = next(
            item
            for item in inventory["jobs"]
            if item["job_id"] == row["job_id"]
        )
        if (
            row["relative_directory"] != f"jobs/{row['job_id']}"
            or row["object_count"] != inventory_job["object_count"]
            or row["objects_sha256"]
            != canonical_sha256(inventory_job["objects"])
        ):
            raise ValueError("validated materialization job inventory changed")
    _reject_scientific_evidence(result)
    return result


def build_vm_absence_observation(
    preview: Mapping[str, Any],
    *,
    instance_names_by_job: Mapping[str, str],
    present_instance_names: Sequence[str],
    observation_source: str,
    query_performed: bool,
    fixture_only: bool,
) -> dict[str, Any]:
    """Build an observation; it may report present VMs but cannot hide them."""

    if set(instance_names_by_job) != set(preview["selected_job_ids"]):
        raise ValueError("VM absence observation must bind every exact job")
    _strict_bool(query_performed, None, "VM absence query")
    _strict_bool(fixture_only, None, "VM absence fixture")
    if fixture_only and query_performed:
        raise ValueError("fixture VM absence cannot claim a cloud query")
    if (
        not isinstance(observation_source, str)
        or not observation_source
        or len(observation_source) > 128
    ):
        raise ValueError("VM absence observation source changed")
    names: list[str] = []
    instances: list[dict[str, Any]] = []
    for job_id in preview["selected_job_ids"]:
        name = instance_names_by_job[job_id]
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise ValueError("VM instance name is not safe")
        names.append(name)
        instances.append(
            {
                "job_id": job_id,
                "instance_name": name,
                "present": name in present_instance_names,
            }
        )
    if len(names) != len(set(names)):
        raise ValueError("VM instance names collide")
    if (
        any(not isinstance(name, str) for name in present_instance_names)
        or len(present_instance_names) != len(set(present_instance_names))
        or not set(present_instance_names).issubset(names)
    ):
        raise ValueError("VM absence observation contains an unknown instance")
    return {
        "schema": VM_ABSENCE_SCHEMA,
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "project": adapter.DEFAULT_PROJECT,
        "zone": adapter.DEFAULT_ZONE,
        "observation_source": observation_source,
        "query_performed": query_performed,
        "fixture_only": fixture_only,
        "instances": instances,
        "instances_sha256": canonical_sha256(instances),
        "present_instance_names": list(present_instance_names),
        "all_expected_instances_absent": not present_instance_names,
        "cloud_mutation_performed": False,
        "vm_delete_performed": False,
        "vm_create_performed": False,
    }


def _validate_vm_absence(
    value: Mapping[str, Any], *, preview: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(value)
    _exact(
        row,
        {
            "schema",
            "stage_identity_sha256",
            "stage_id",
            "run_name",
            "project",
            "zone",
            "observation_source",
            "query_performed",
            "fixture_only",
            "instances",
            "instances_sha256",
            "present_instance_names",
            "all_expected_instances_absent",
            "cloud_mutation_performed",
            "vm_delete_performed",
            "vm_create_performed",
        },
        "VM absence observation",
    )
    if (
        row["schema"] != VM_ABSENCE_SCHEMA
        or row["stage_identity_sha256"] != preview["stage_identity_sha256"]
        or row["stage_id"] != preview["stage_id"]
        or row["run_name"] != preview["run_name"]
        or row["project"] != adapter.DEFAULT_PROJECT
        or row["zone"] != adapter.DEFAULT_ZONE
        or row["instances_sha256"] != canonical_sha256(row["instances"])
        or [item.get("job_id") for item in row["instances"]]
        != preview["selected_job_ids"]
    ):
        raise ValueError("VM absence observation identity changed")
    instance_names: list[str] = []
    _strict_bool(row["query_performed"], None, "VM absence query")
    _strict_bool(row["fixture_only"], None, "VM absence fixture")
    if row["fixture_only"] and row["query_performed"]:
        raise ValueError("fixture VM absence claimed an external query")
    present = []
    for item in row["instances"]:
        _exact(item, {"job_id", "instance_name", "present"}, "VM state")
        name = item["instance_name"]
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise ValueError("VM absence instance name is not safe")
        instance_names.append(name)
        _strict_bool(item["present"], None, "VM present")
        if item["present"]:
            present.append(name)
    if len(instance_names) != len(set(instance_names)):
        raise ValueError("VM absence instance names collide")
    if row["present_instance_names"] != present:
        raise ValueError("VM present inventory changed")
    _strict_bool(
        row["all_expected_instances_absent"],
        not present,
        "all VMs absent",
    )
    for field in (
        "cloud_mutation_performed",
        "vm_delete_performed",
        "vm_create_performed",
    ):
        _strict_bool(row[field], False, f"VM absence {field}")
    return row


def build_controller_receipt(
    preview: Mapping[str, Any],
    *,
    receive: Mapping[str, Any],
    materialization_result: Mapping[str, Any],
    vm_absence_observation: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Issue a receipt only after content validation and exact VM absence."""

    adapter.validate_receive(receive, preview=preview)
    materialized = validate_materialization_result(
        materialization_result,
        preview=preview,
        receive=receive,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    absence = _validate_vm_absence(vm_absence_observation, preview=preview)
    if absence["present_instance_names"] or not absence[
        "all_expected_instances_absent"
    ]:
        raise ValueError("controller receipt requires every worker VM absent")
    fixture_only = (
        materialized["backend_fixture_only"] or absence["fixture_only"]
    )
    # This module has no signed provider-query transcript validator.  Caller
    # booleans can never be promoted to real cloud evidence here.
    external_read_only_evidence = False
    receipt = {
        "schema": CONTROLLER_RECEIPT_SCHEMA,
        "status": (
            "external_read_only_receive_validated_all_vms_absent"
            if external_read_only_evidence
            else "local_contract_validated_all_vms_absent_not_cloud_evidence"
        ),
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "selected_job_ids": list(preview["selected_job_ids"]),
        "receive_sha256": canonical_sha256(receive),
        "materialization_result_sha256": canonical_sha256(materialized),
        "vm_absence_observation_sha256": canonical_sha256(absence),
        "runner_content_validated": True,
        "all_worker_vms_absent": True,
        "fixture_only": fixture_only,
        "external_read_only_evidence": external_read_only_evidence,
        "cloud_mutation_performed": False,
        "claim_created": False,
        "authorization_created": False,
        "vm_created": False,
        "launch_authorized": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    _reject_scientific_evidence(receipt)
    return receipt


def build_read_only_preflight_plan(
    preview: Mapping[str, Any],
    *,
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
    spot_price_ceiling_usd_per_vm_hour: float = (
        SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    ),
) -> dict[str, Any]:
    ceiling = _number(
        spot_price_ceiling_usd_per_vm_hour, "Spot price ceiling"
    )
    if ceiling <= 0:
        raise ValueError("Spot price ceiling must be positive")
    if preview["attempt_index"] != 0:
        raise ValueError(
            "10c.2 fresh-prefix preflight is only valid before initial attempt0"
        )
    vm_count = _strict_int(preview["vm_count"], "preview VM count", minimum=1)
    direct_layout = build_direct_v1_remote_layout(
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    if direct_stage_identity is not None:
        checked_direct_identity = validate_direct_stage_identity(
            direct_stage_identity,
            preview=preview,
            outer_package_manifest=outer_package_manifest,
        )
        expected_instances = [
            {
                "job_id": row["job_id"],
                "attempt_index": row["attempt_index"],
                "instance_name": row["instance_name"],
            }
            for row in checked_direct_identity["inputs"]["attempt_layout"]
        ]
        expected_attempts = {
            row["attempt_index"] for row in expected_instances
        }
        expected_jobs = {job["job_id"] for job in preview["jobs"]}
        if (
            expected_attempts != {0, 1}
            or {
                row["job_id"]
                for row in expected_instances
                if row["attempt_index"] == 0
            }
            != expected_jobs
            or {
                row["job_id"]
                for row in expected_instances
                if row["attempt_index"] == 1
            }
            != expected_jobs
        ):
            raise ValueError(
                "canonical direct identity lost attempt0/attempt1 instance coverage"
            )
    else:
        expected_instances = [
            {
                "job_id": row["job_id"],
                "attempt_index": preview["attempt_index"],
                "instance_name": row["instance_name"],
            }
            for row in direct_layout["job_layouts"]
        ]
    expected_package_objects = [
        {
            "uri": row["uri"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in direct_layout["package_objects"]
    ]
    if (
        len(expected_package_objects) < preview["package_file_count"]
        or any(
            row["uri"] == direct_layout["package_prefix"]
            or not row["uri"].startswith(
                direct_layout["package_prefix"] + "/"
            )
            for row in expected_package_objects
        )
    ):
        raise ValueError("content-addressed package prefix inventory changed")
    requirements = {
        "prefix": {
            "base_prefix": direct_layout["base_prefix"],
            "package_prefix": direct_layout["package_prefix"],
            "expected_package_objects": expected_package_objects,
            "expected_package_objects_sha256": canonical_sha256(
                expected_package_objects
            ),
            "allowed_package_inventory_states": [
                "exact_empty_provisioning_required",
                "exact_expected_subset_provisioning_required",
                "exact_complete_immutable_reuse",
            ],
            "partial_unknown_or_mismatch_is_fatal": True,
            "stage_prefix": direct_layout["stage_prefix"],
            "stage_prefix_must_be_exactly_empty": True,
            "result_prefix": direct_layout["result_prefix"],
            "attempt_control_prefix": direct_layout[
                "attempt_control_prefix"
            ],
            "package_and_stage_prefix_disjoint": True,
            "legacy_preview_uris_are_identity_provenance_only": True,
            "unknown_object_is_fatal": True,
            "observation_max_age_seconds": (
                PREFIX_OBSERVATION_MAX_AGE_SECONDS
            ),
        },
        "capacity": {
            "project": adapter.DEFAULT_PROJECT,
            "region": adapter.DEFAULT_ZONE.rsplit("-", 1)[0],
            "zone": adapter.DEFAULT_ZONE,
            "machine_type": MACHINE_TYPE,
            "provisioning_model": "SPOT",
            "vm_count": vm_count,
            "vcpu_per_vm": VCPU_PER_VM,
            "required_vcpu": vm_count * VCPU_PER_VM,
            "provider_metric_mapping_must_be_supplied_by_external_observer": True,
            "required_provider_metric_mapping": (
                CAPACITY_PROVIDER_METRIC_MAPPING
            ),
            "required_provider_metrics": [
                "region.quotas[C4_CPUS]",
                "region.quotas[PREEMPTIBLE_CPUS]",
                "project.quotas[CPUS_ALL_REGIONS]",
            ],
            "project_global_quota_headroom_required": True,
            "observation_max_age_seconds": (
                CAPACITY_OBSERVATION_MAX_AGE_SECONDS
            ),
        },
        "instances": {
            "project": adapter.DEFAULT_PROJECT,
            "zone": adapter.DEFAULT_ZONE,
            "expected_instances": expected_instances,
            "required_attempt_indices": sorted(
                {row["attempt_index"] for row in expected_instances}
            ),
            "expected_instance_names_must_all_be_absent": True,
            "unknown_instance_record_is_fatal": True,
            "observation_max_age_seconds": (
                INSTANCE_OBSERVATION_MAX_AGE_SECONDS
            ),
        },
        "spot_price": {
            "region": adapter.DEFAULT_ZONE.rsplit("-", 1)[0],
            "machine_type": MACHINE_TYPE,
            "provisioning_model": "SPOT",
            "currency": "USD",
            "unit": "vm_hour",
            "ceiling_usd_per_vm_hour": ceiling,
            "authoritative_external_observation_required": True,
            "official_source_url_required": True,
            "official_source_domain": "cloud.google.com",
            "sku_effective_time_required": True,
            "retrieval_freshness_and_provider_lag_are_separate": True,
            "observation_max_age_seconds": PRICE_OBSERVATION_MAX_AGE_SECONDS,
        },
        "launch_permissions": {
            "compute_instances_delete_permission_required": True,
            "worker_oauth_scope_required": REQUIRED_WORKER_OAUTH_SCOPE,
            "get_only_observation_does_not_prove_launch_permissions": True,
            "missing_permission_or_scope_evidence_is_launch_no_go": True,
        },
    }
    plan = {
        "schema": PREFLIGHT_PLAN_SCHEMA,
        "status": (
            "external_read_only_observations_required_fail_closed"
            if direct_layout["real_read_only_preflight_contract_eligible"]
            else "legacy_local_contract_only_outer_bindings_required"
        ),
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "outer_package_identity_sha256": direct_layout[
            "outer_package_identity_sha256"
        ],
        "direct_stage_identity_sha256": direct_layout[
            "direct_stage_identity_sha256"
        ],
        "real_read_only_preflight_contract_eligible": direct_layout[
            "real_read_only_preflight_contract_eligible"
        ],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "requirements": requirements,
        "requirements_sha256": canonical_sha256(requirements),
        "observation_order": [
            "exact_content_addressed_package_inventory",
            "exact_empty_direct_v1_stage_prefix",
            "exact_expected_instance_names_absent",
            "launch_capacity",
            "authoritative_spot_price",
            "launch_permission_readiness_separate_from_read_only_observation",
        ],
        "cloud_query_performed": False,
        "cloud_mutation_performed": False,
        "claim_created": False,
        "authorization_created": False,
        "vm_created": False,
        "launch_authorized": False,
        "launch_ready": False,
        "package_provisioning_authorized": False,
        "price_invented_or_cached": False,
        "diagnostic_only": True,
    }
    _reject_scientific_evidence(plan)
    return plan


def validate_read_only_preflight_plan(
    value: Mapping[str, Any],
    *,
    preview: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(value)
    _exact(
        plan,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_identity_sha256",
            "outer_package_identity_sha256",
            "direct_stage_identity_sha256",
            "real_read_only_preflight_contract_eligible",
            "stage_id",
            "run_name",
            "requirements",
            "requirements_sha256",
            "observation_order",
            "cloud_query_performed",
            "cloud_mutation_performed",
            "claim_created",
            "authorization_created",
            "vm_created",
            "launch_authorized",
            "launch_ready",
            "package_provisioning_authorized",
            "price_invented_or_cached",
            "diagnostic_only",
        },
        "read-only preflight plan",
    )
    requirements = plan.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("read-only preflight requirements are missing")
    spot = requirements.get("spot_price")
    if not isinstance(spot, Mapping):
        raise ValueError("read-only preflight Spot requirement is missing")
    expected = build_read_only_preflight_plan(
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
        spot_price_ceiling_usd_per_vm_hour=spot.get(
            "ceiling_usd_per_vm_hour"
        ),
    )
    if plan != expected:
        raise ValueError("read-only preflight plan changed")
    return expected


def build_prefix_observation(
    plan: Mapping[str, Any],
    *,
    package_objects: Sequence[Mapping[str, Any]],
    stage_object_uris: Sequence[str],
    observation_source: str,
    source_identity: str,
    observed_at_unix_seconds: int,
    query_performed: bool,
    fixture_only: bool,
) -> dict[str, Any]:
    _strict_bool(query_performed, None, "prefix query")
    _strict_bool(fixture_only, None, "prefix fixture")
    if fixture_only and query_performed:
        raise ValueError("fixture prefix cannot claim an external query")
    _strict_int(
        observed_at_unix_seconds,
        "prefix observed_at",
        minimum=0,
    )
    if (
        not isinstance(source_identity, str)
        or not source_identity
        or len(source_identity) > 256
    ):
        raise ValueError("prefix source identity changed")
    checked_package: list[dict[str, Any]] = []
    for raw in package_objects:
        row = dict(raw)
        _exact(
            row,
            {
                "uri",
                "generation",
                "metageneration",
                "sha256",
                "bytes",
                "crc32c",
                "etag",
            },
            "package prefix object",
        )
        if not isinstance(row["uri"], str):
            raise ValueError("package prefix object URI changed")
        checked_package.append(
            {
                "uri": row["uri"],
                "generation": _strict_int(
                    row["generation"], "package object generation", minimum=1
                ),
                "metageneration": _strict_int(
                    row["metageneration"],
                    "package object metageneration",
                    minimum=1,
                ),
                "sha256": _sha(row["sha256"], "package object sha256"),
                "bytes": _strict_int(
                    row["bytes"], "package object bytes", minimum=1
                ),
                "crc32c": row["crc32c"],
                "etag": row["etag"],
            }
        )
        if not row["crc32c"] or not row["etag"]:
            raise ValueError("package object crc32c/etag is missing")
    if any(not isinstance(uri, str) for uri in stage_object_uris) or len(
        stage_object_uris
    ) != len(set(stage_object_uris)):
        raise ValueError("stage prefix observation URI inventory changed")
    requirement = plan["requirements"]["prefix"]
    expected_objects = requirement["expected_package_objects"]
    checked_identities_unordered = [
        {
            "uri": row["uri"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in checked_package
    ]
    observed_by_uri: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for checked_row, identity in zip(
        checked_package, checked_identities_unordered, strict=True
    ):
        uri = identity["uri"]
        if uri in observed_by_uri:
            raise ValueError("package prefix inventory contains a duplicate URI")
        observed_by_uri[uri] = (checked_row, identity)
    expected_by_uri = {row["uri"]: row for row in expected_objects}
    if len(expected_by_uri) != len(expected_objects):
        raise ValueError("expected package manifest contains duplicate URIs")
    unknown_uris = set(observed_by_uri) - set(expected_by_uri)
    mismatched_uris = {
        uri
        for uri, (_, identity) in observed_by_uri.items()
        if uri in expected_by_uri and identity != expected_by_uri[uri]
    }
    if unknown_uris or mismatched_uris:
        raise ValueError("package prefix contains an unknown or mismatched object")
    # GCS returns lexicographic order, while the immutable outer manifest order
    # is the content identity.  Any exact sparse subset is normalized back to
    # manifest order before hashes and missing-object lists are derived.
    checked_package = [
        observed_by_uri[row["uri"]][0]
        for row in expected_objects
        if row["uri"] in observed_by_uri
    ]
    checked_identities = [
        observed_by_uri[row["uri"]][1]
        for row in expected_objects
        if row["uri"] in observed_by_uri
    ]
    observed_uris = [row["uri"] for row in checked_identities]
    expected_subset = [
        row for row in expected_objects if row["uri"] in observed_by_uri
    ]
    if checked_identities == expected_objects:
        package_state = "exact_complete_immutable_reuse"
    elif not checked_package:
        package_state = "exact_empty_provisioning_required"
    elif checked_identities == expected_subset:
        package_state = "exact_expected_subset_provisioning_required"
    else:  # pragma: no cover - guarded by exact URI/content checks above.
        raise ValueError("package prefix subset normalization failed")
    missing_objects = [
        row for row in expected_objects if row["uri"] not in observed_uris
    ]
    return {
        "schema": PREFIX_OBSERVATION_SCHEMA,
        "base_prefix": requirement["base_prefix"],
        "package_prefix": requirement["package_prefix"],
        "stage_prefix": requirement["stage_prefix"],
        "result_prefix": requirement["result_prefix"],
        "attempt_control_prefix": requirement["attempt_control_prefix"],
        "observation_source": observation_source,
        "source_identity": source_identity,
        "observed_at_unix_seconds": observed_at_unix_seconds,
        "valid_until_unix_seconds": (
            observed_at_unix_seconds
            + requirement["observation_max_age_seconds"]
        ),
        "query_performed": query_performed,
        "fixture_only": fixture_only,
        "package_objects": checked_package,
        "package_objects_sha256": canonical_sha256(checked_package),
        "package_object_identities": checked_identities,
        "package_object_identities_sha256": canonical_sha256(
            checked_identities
        ),
        "package_inventory_state": package_state,
        "package_inventory_acceptable": package_state
        in requirement["allowed_package_inventory_states"],
        "package_provisioning_required": (
            package_state
            in {
                "exact_empty_provisioning_required",
                "exact_expected_subset_provisioning_required",
            }
        ),
        "missing_package_objects": missing_objects,
        "missing_package_objects_sha256": canonical_sha256(missing_objects),
        "stage_object_uris": list(stage_object_uris),
        "stage_object_uris_sha256": canonical_sha256(
            list(stage_object_uris)
        ),
        "stage_prefix_exactly_empty": len(stage_object_uris) == 0,
        "cloud_mutation_performed": False,
    }


def build_instance_observation(
    plan: Mapping[str, Any],
    *,
    observed_instances: Sequence[Mapping[str, Any]],
    observation_source: str,
    source_identity: str,
    observed_at_unix_seconds: int,
    query_performed: bool,
    fixture_only: bool,
) -> dict[str, Any]:
    _strict_bool(query_performed, None, "instance query")
    _strict_bool(fixture_only, None, "instance fixture")
    if fixture_only and query_performed:
        raise ValueError("fixture instance observation cannot claim a cloud query")
    _strict_int(
        observed_at_unix_seconds, "instance observed_at", minimum=0
    )
    if (
        not isinstance(source_identity, str)
        or not source_identity
        or len(source_identity) > 256
    ):
        raise ValueError("instance source identity changed")
    requirement = plan["requirements"]["instances"]
    checked: list[dict[str, Any]] = []
    for raw in observed_instances:
        row = dict(raw)
        _exact(
            row,
            {"instance_name", "zone", "status"},
            "observed instance",
        )
        if (
            not isinstance(row["instance_name"], str)
            or _SAFE_NAME.fullmatch(row["instance_name"]) is None
            or row["zone"] != requirement["zone"]
            or not isinstance(row["status"], str)
            or not row["status"]
        ):
            raise ValueError("observed instance identity changed")
        checked.append(row)
    names = [row["instance_name"] for row in checked]
    if len(names) != len(set(names)):
        raise ValueError("observed instance inventory contains duplicates")
    expected_names = [
        row["instance_name"] for row in requirement["expected_instances"]
    ]
    collisions = [name for name in expected_names if name in names]
    return {
        "schema": INSTANCE_OBSERVATION_SCHEMA,
        "project": requirement["project"],
        "zone": requirement["zone"],
        "expected_instance_names": expected_names,
        "observation_source": observation_source,
        "source_identity": source_identity,
        "observed_at_unix_seconds": observed_at_unix_seconds,
        "valid_until_unix_seconds": (
            observed_at_unix_seconds
            + requirement["observation_max_age_seconds"]
        ),
        "query_performed": query_performed,
        "fixture_only": fixture_only,
        "observed_instances": checked,
        "observed_instances_sha256": canonical_sha256(checked),
        "colliding_expected_instance_names": collisions,
        "all_expected_instance_names_absent": not collisions,
        "cloud_mutation_performed": False,
    }


def build_capacity_observation(
    plan: Mapping[str, Any],
    *,
    available_vcpu: int | None,
    provider_metric_mapping: str | None,
    observation_source: str,
    source_identity: str,
    observed_at_unix_seconds: int,
    query_performed: bool,
    fixture_only: bool,
) -> dict[str, Any]:
    _strict_bool(query_performed, None, "capacity query")
    _strict_bool(fixture_only, None, "capacity fixture")
    if fixture_only and query_performed:
        raise ValueError("fixture capacity cannot claim an external query")
    _strict_int(
        observed_at_unix_seconds,
        "capacity observed_at",
        minimum=0,
    )
    if (
        not isinstance(source_identity, str)
        or not source_identity
        or len(source_identity) > 256
    ):
        raise ValueError("capacity source identity changed")
    if available_vcpu is not None:
        _strict_int(available_vcpu, "available Spot vCPU", minimum=0)
    if provider_metric_mapping is not None and (
        not isinstance(provider_metric_mapping, str)
        or not provider_metric_mapping
    ):
        raise ValueError("provider capacity metric mapping changed")
    requirement = plan["requirements"]["capacity"]
    if (
        provider_metric_mapping is not None
        and provider_metric_mapping
        != requirement["required_provider_metric_mapping"]
    ):
        raise ValueError(
            "provider capacity mapping omitted regional or global quota"
        )
    authoritative = (
        query_performed
        and not fixture_only
        and available_vcpu is not None
        and provider_metric_mapping is not None
    )
    return {
        "schema": CAPACITY_OBSERVATION_SCHEMA,
        "project": requirement["project"],
        "region": requirement["region"],
        "zone": requirement["zone"],
        "machine_type": requirement["machine_type"],
        "provisioning_model": requirement["provisioning_model"],
        "required_vcpu": requirement["required_vcpu"],
        "available_vcpu": available_vcpu,
        "provider_metric_mapping": provider_metric_mapping,
        "observation_source": observation_source,
        "source_identity": source_identity,
        "observed_at_unix_seconds": observed_at_unix_seconds,
        "valid_until_unix_seconds": (
            observed_at_unix_seconds
            + requirement["observation_max_age_seconds"]
        ),
        "query_performed": query_performed,
        "fixture_only": fixture_only,
        "authoritative_external_observation": authoritative,
        "capacity_sufficient": (
            authoritative and available_vcpu >= requirement["required_vcpu"]
        ),
        "cloud_mutation_performed": False,
    }


def build_spot_price_observation(
    plan: Mapping[str, Any],
    *,
    observed_price_usd_per_vm_hour: float | None,
    observation_source: str,
    source_identity: str,
    official_source_url: str | None,
    observed_at_unix_seconds: int,
    sku_effective_at_unix_seconds: int | None,
    query_performed: bool,
    fixture_only: bool,
) -> dict[str, Any]:
    _strict_bool(query_performed, None, "price query")
    _strict_bool(fixture_only, None, "price fixture")
    if fixture_only and query_performed:
        raise ValueError("fixture price cannot claim an external query")
    _strict_int(observed_at_unix_seconds, "price observed_at", minimum=0)
    if (
        not isinstance(source_identity, str)
        or not source_identity
        or len(source_identity) > 256
    ):
        raise ValueError("price source identity changed")
    if official_source_url is not None and (
        not isinstance(official_source_url, str)
        or not official_source_url.startswith("https://cloud.google.com/")
    ):
        raise ValueError("Spot price source URL is not an official URL")
    if sku_effective_at_unix_seconds is not None:
        _strict_int(
            sku_effective_at_unix_seconds,
            "SKU effective time",
            minimum=0,
            maximum=observed_at_unix_seconds,
        )
    if observed_price_usd_per_vm_hour is not None:
        _number(observed_price_usd_per_vm_hour, "observed Spot price")
    requirement = plan["requirements"]["spot_price"]
    authoritative = (
        query_performed
        and not fixture_only
        and observed_price_usd_per_vm_hour is not None
        and official_source_url is not None
        and sku_effective_at_unix_seconds is not None
    )
    return {
        "schema": PRICE_OBSERVATION_SCHEMA,
        "region": requirement["region"],
        "machine_type": requirement["machine_type"],
        "provisioning_model": requirement["provisioning_model"],
        "currency": requirement["currency"],
        "unit": requirement["unit"],
        "ceiling_usd_per_vm_hour": requirement[
            "ceiling_usd_per_vm_hour"
        ],
        "observed_price_usd_per_vm_hour": observed_price_usd_per_vm_hour,
        "observation_source": observation_source,
        "source_identity": source_identity,
        "official_source_url": official_source_url,
        "observed_at_unix_seconds": observed_at_unix_seconds,
        "retrieved_at_unix_seconds": observed_at_unix_seconds,
        "sku_effective_at_unix_seconds": sku_effective_at_unix_seconds,
        "provider_effective_lag_seconds": (
            observed_at_unix_seconds - sku_effective_at_unix_seconds
            if sku_effective_at_unix_seconds is not None
            else None
        ),
        "valid_until_unix_seconds": (
            observed_at_unix_seconds
            + requirement["observation_max_age_seconds"]
        ),
        "query_performed": query_performed,
        "fixture_only": fixture_only,
        "authoritative_external_observation": authoritative,
        "within_ceiling": (
            authoritative
            and observed_price_usd_per_vm_hour
            <= requirement["ceiling_usd_per_vm_hour"]
        ),
        "cloud_mutation_performed": False,
    }


def evaluate_read_only_preflight(
    plan: Mapping[str, Any],
    *,
    preview: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
    prefix_observation: Mapping[str, Any],
    instance_observation: Mapping[str, Any],
    capacity_observation: Mapping[str, Any],
    price_observation: Mapping[str, Any],
    evaluation_unix_seconds: int,
) -> dict[str, Any]:
    """Evaluate observations without turning a pass into launch authorization."""

    prefix = dict(prefix_observation)
    instances = dict(instance_observation)
    capacity = dict(capacity_observation)
    price = dict(price_observation)
    _strict_int(
        evaluation_unix_seconds, "preflight evaluation time", minimum=0
    )
    validate_read_only_preflight_plan(
        plan,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    _exact(
        plan,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_identity_sha256",
            "outer_package_identity_sha256",
            "direct_stage_identity_sha256",
            "real_read_only_preflight_contract_eligible",
            "stage_id",
            "run_name",
            "requirements",
            "requirements_sha256",
            "observation_order",
            "cloud_query_performed",
            "cloud_mutation_performed",
            "claim_created",
            "authorization_created",
            "vm_created",
            "launch_authorized",
            "launch_ready",
            "package_provisioning_authorized",
            "price_invented_or_cached",
            "diagnostic_only",
        },
        "read-only preflight plan",
    )
    if (
        plan["schema"] != PREFLIGHT_PLAN_SCHEMA
        or plan["status"]
        not in {
            "external_read_only_observations_required_fail_closed",
            "legacy_local_contract_only_outer_bindings_required",
        }
        or plan["requirements_sha256"]
        != canonical_sha256(plan["requirements"])
    ):
        raise ValueError("read-only preflight plan identity changed")
    for field in ("cloud_query_performed",):
        _strict_bool(plan[field], False, f"preflight plan {field}")
    for field in (
        "cloud_mutation_performed",
        "claim_created",
        "authorization_created",
        "vm_created",
        "launch_authorized",
        "launch_ready",
        "package_provisioning_authorized",
        "price_invented_or_cached",
    ):
        _strict_bool(plan[field], False, f"preflight plan {field}")
    _strict_bool(plan["diagnostic_only"], True, "preflight diagnostic")
    _strict_bool(
        plan["real_read_only_preflight_contract_eligible"],
        outer_package_manifest is not None
        and direct_stage_identity is not None,
        "real preflight eligibility",
    )
    _exact(
        prefix,
        {
            "schema",
            "base_prefix",
            "package_prefix",
            "stage_prefix",
            "result_prefix",
            "attempt_control_prefix",
            "observation_source",
            "source_identity",
            "observed_at_unix_seconds",
            "valid_until_unix_seconds",
            "query_performed",
            "fixture_only",
            "package_objects",
            "package_objects_sha256",
            "package_object_identities",
            "package_object_identities_sha256",
            "package_inventory_state",
            "package_inventory_acceptable",
            "package_provisioning_required",
            "missing_package_objects",
            "missing_package_objects_sha256",
            "stage_object_uris",
            "stage_object_uris_sha256",
            "stage_prefix_exactly_empty",
            "cloud_mutation_performed",
        },
        "prefix observation",
    )
    _exact(
        instances,
        {
            "schema",
            "project",
            "zone",
            "expected_instance_names",
            "observation_source",
            "source_identity",
            "observed_at_unix_seconds",
            "valid_until_unix_seconds",
            "query_performed",
            "fixture_only",
            "observed_instances",
            "observed_instances_sha256",
            "colliding_expected_instance_names",
            "all_expected_instance_names_absent",
            "cloud_mutation_performed",
        },
        "instance observation",
    )
    _exact(
        capacity,
        {
            "schema",
            "project",
            "region",
            "zone",
            "machine_type",
            "provisioning_model",
            "required_vcpu",
            "available_vcpu",
            "provider_metric_mapping",
            "observation_source",
            "source_identity",
            "observed_at_unix_seconds",
            "valid_until_unix_seconds",
            "query_performed",
            "fixture_only",
            "authoritative_external_observation",
            "capacity_sufficient",
            "cloud_mutation_performed",
        },
        "capacity observation",
    )
    _exact(
        price,
        {
            "schema",
            "region",
            "machine_type",
            "provisioning_model",
            "currency",
            "unit",
            "ceiling_usd_per_vm_hour",
            "observed_price_usd_per_vm_hour",
            "observation_source",
            "source_identity",
            "official_source_url",
            "observed_at_unix_seconds",
            "retrieved_at_unix_seconds",
            "sku_effective_at_unix_seconds",
            "provider_effective_lag_seconds",
            "valid_until_unix_seconds",
            "query_performed",
            "fixture_only",
            "authoritative_external_observation",
            "within_ceiling",
            "cloud_mutation_performed",
        },
        "Spot price observation",
    )
    requirements = plan["requirements"]
    if (
        prefix["schema"] != PREFIX_OBSERVATION_SCHEMA
        or any(
            prefix[field] != requirements["prefix"][field]
            for field in (
                "base_prefix",
                "package_prefix",
                "stage_prefix",
                "result_prefix",
                "attempt_control_prefix",
            )
        )
        or prefix["package_objects_sha256"]
        != canonical_sha256(prefix["package_objects"])
        or prefix["package_object_identities_sha256"]
        != canonical_sha256(prefix["package_object_identities"])
        or prefix["stage_object_uris_sha256"]
        != canonical_sha256(prefix["stage_object_uris"])
        or prefix["stage_prefix_exactly_empty"]
        != (len(prefix["stage_object_uris"]) == 0)
    ):
        raise ValueError("prefix observation identity changed")
    reconstructed_identities: list[dict[str, Any]] = []
    for row in prefix["package_objects"]:
        _exact(
            row,
            {
                "uri",
                "generation",
                "metageneration",
                "sha256",
                "bytes",
                "crc32c",
                "etag",
            },
            "observed package object",
        )
        if not row["uri"].startswith(prefix["package_prefix"] + "/"):
            raise ValueError("observed package object escaped package prefix")
        _strict_int(
            row["generation"], "observed package generation", minimum=1
        )
        _strict_int(
            row["metageneration"],
            "observed package metageneration",
            minimum=1,
        )
        _sha(row["sha256"], "observed package object sha256")
        _strict_int(row["bytes"], "observed package object bytes", minimum=1)
        if any(
            not isinstance(row[field], str) or not row[field]
            for field in ("crc32c", "etag")
        ):
            raise ValueError("observed package crc32c/etag is missing")
        reconstructed_identities.append(
            {
                "uri": row["uri"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
        )
    if prefix["package_object_identities"] != reconstructed_identities:
        raise ValueError("observed package content identities changed")
    if any(
        not uri.startswith(prefix["stage_prefix"] + "/")
        for uri in prefix["stage_object_uris"]
    ):
        raise ValueError("observed stage object escaped stage prefix")
    expected_objects = requirements["prefix"]["expected_package_objects"]
    observed_uris = [
        row["uri"] for row in prefix["package_object_identities"]
    ]
    expected_subset = [
        row for row in expected_objects if row["uri"] in observed_uris
    ]
    expected_package_state = (
        "exact_complete_immutable_reuse"
        if prefix["package_object_identities"] == expected_objects
        else (
            "exact_empty_provisioning_required"
            if not prefix["package_object_identities"]
            else (
                "exact_expected_subset_provisioning_required"
                if prefix["package_object_identities"] == expected_subset
                else "invalid_partial_unknown_or_mismatch"
            )
        )
    )
    if prefix["package_inventory_state"] != expected_package_state:
        raise ValueError("package inventory state changed")
    _strict_bool(
        prefix["package_inventory_acceptable"],
        expected_package_state
        in requirements["prefix"]["allowed_package_inventory_states"],
        "package inventory acceptable",
    )
    _strict_bool(
        prefix["package_provisioning_required"],
        expected_package_state
        in {
            "exact_empty_provisioning_required",
            "exact_expected_subset_provisioning_required",
        },
        "package provisioning required",
    )
    expected_missing = [
        row for row in expected_objects if row["uri"] not in observed_uris
    ]
    if (
        prefix["missing_package_objects"] != expected_missing
        or prefix["missing_package_objects_sha256"]
        != canonical_sha256(expected_missing)
    ):
        raise ValueError("missing package object inventory changed")
    _strict_bool(
        prefix["stage_prefix_exactly_empty"],
        not prefix["stage_object_uris"],
        "stage prefix empty",
    )
    instance_requirement = requirements["instances"]
    expected_instance_names = [
        row["instance_name"]
        for row in instance_requirement["expected_instances"]
    ]
    if (
        instances["schema"] != INSTANCE_OBSERVATION_SCHEMA
        or instances["project"] != instance_requirement["project"]
        or instances["zone"] != instance_requirement["zone"]
        or instances["expected_instance_names"] != expected_instance_names
        or instances["observed_instances_sha256"]
        != canonical_sha256(instances["observed_instances"])
    ):
        raise ValueError("instance observation identity changed")
    observed_names: list[str] = []
    for row in instances["observed_instances"]:
        _exact(
            row,
            {"instance_name", "zone", "status"},
            "observed instance",
        )
        if (
            not isinstance(row["instance_name"], str)
            or _SAFE_NAME.fullmatch(row["instance_name"]) is None
            or row["zone"] != instance_requirement["zone"]
            or not isinstance(row["status"], str)
            or not row["status"]
        ):
            raise ValueError("observed instance record changed")
        observed_names.append(row["instance_name"])
    if len(observed_names) != len(set(observed_names)):
        raise ValueError("observed instance inventory contains duplicates")
    unknown_instances = set(observed_names) - set(expected_instance_names)
    if unknown_instances:
        raise ValueError("instance observation contains an unknown instance")
    expected_collisions = [
        name for name in expected_instance_names if name in observed_names
    ]
    if (
        instances["colliding_expected_instance_names"]
        != expected_collisions
    ):
        raise ValueError("instance collision inventory changed")
    _strict_bool(
        instances["all_expected_instance_names_absent"],
        not expected_collisions,
        "all expected instance names absent",
    )
    expected_capacity = requirements["capacity"]
    if (
        capacity["schema"] != CAPACITY_OBSERVATION_SCHEMA
        or any(
            capacity[field] != expected_capacity[field]
            for field in (
                "project",
                "region",
                "zone",
                "machine_type",
                "provisioning_model",
                "required_vcpu",
            )
        )
    ):
        raise ValueError("capacity observation identity changed")
    available_vcpu = capacity["available_vcpu"]
    if available_vcpu is not None:
        _strict_int(available_vcpu, "available Spot vCPU", minimum=0)
    metric_mapping = capacity["provider_metric_mapping"]
    if metric_mapping is not None and (
        not isinstance(metric_mapping, str) or not metric_mapping
    ):
        raise ValueError("capacity provider metric mapping changed")
    if (
        metric_mapping is not None
        and metric_mapping
        != expected_capacity["required_provider_metric_mapping"]
    ):
        raise ValueError(
            "capacity observation omitted regional or global quota headroom"
        )
    capacity_authoritative = (
        capacity["query_performed"]
        and not capacity["fixture_only"]
        and available_vcpu is not None
        and metric_mapping is not None
    )
    _strict_bool(
        capacity["authoritative_external_observation"],
        capacity_authoritative,
        "capacity authoritative observation",
    )
    _strict_bool(
        capacity["capacity_sufficient"],
        capacity_authoritative
        and available_vcpu >= expected_capacity["required_vcpu"],
        "capacity sufficient",
    )
    expected_price = requirements["spot_price"]
    if (
        price["schema"] != PRICE_OBSERVATION_SCHEMA
        or any(
            price[field] != expected_price[field]
            for field in (
                "region",
                "machine_type",
                "provisioning_model",
                "currency",
                "unit",
                "ceiling_usd_per_vm_hour",
            )
        )
    ):
        raise ValueError("Spot price observation identity changed")
    observed_price = price["observed_price_usd_per_vm_hour"]
    if observed_price is not None:
        _number(observed_price, "observed Spot price")
    official_source_url = price["official_source_url"]
    if official_source_url is not None and (
        not isinstance(official_source_url, str)
        or not official_source_url.startswith("https://cloud.google.com/")
    ):
        raise ValueError("Spot price observation lost official URL binding")
    if price["retrieved_at_unix_seconds"] != price["observed_at_unix_seconds"]:
        raise ValueError("Spot price retrieval time changed")
    sku_effective = price["sku_effective_at_unix_seconds"]
    if sku_effective is not None:
        _strict_int(
            sku_effective,
            "SKU effective time",
            minimum=0,
            maximum=price["retrieved_at_unix_seconds"],
        )
    expected_provider_lag = (
        price["retrieved_at_unix_seconds"] - sku_effective
        if sku_effective is not None
        else None
    )
    if price["provider_effective_lag_seconds"] != expected_provider_lag:
        raise ValueError("Spot price provider effective lag changed")
    price_authoritative = (
        price["query_performed"]
        and not price["fixture_only"]
        and observed_price is not None
        and official_source_url is not None
        and sku_effective is not None
    )
    _strict_bool(
        price["authoritative_external_observation"],
        price_authoritative,
        "price authoritative observation",
    )
    _strict_bool(
        price["within_ceiling"],
        price_authoritative
        and observed_price <= expected_price["ceiling_usd_per_vm_hour"],
        "price within ceiling",
    )
    for label, observation in (
        ("prefix", prefix),
        ("instances", instances),
        ("capacity", capacity),
        ("price", price),
    ):
        _strict_bool(
            observation["query_performed"], None, f"{label} query"
        )
        _strict_bool(observation["fixture_only"], None, f"{label} fixture")
        _strict_bool(
            observation["cloud_mutation_performed"],
            False,
            f"{label} mutation",
        )
        if observation["fixture_only"] and observation["query_performed"]:
            raise ValueError(f"{label} fixture claimed an external query")
        observed_at = _strict_int(
            observation["observed_at_unix_seconds"],
            f"{label} observed_at",
            minimum=0,
        )
        valid_until = _strict_int(
            observation["valid_until_unix_seconds"],
            f"{label} valid_until",
            minimum=observed_at,
        )
        max_age = requirements[
            "prefix" if label == "prefix" else (
                "instances" if label == "instances" else (
                    "capacity" if label == "capacity" else "spot_price"
                )
            )
        ]["observation_max_age_seconds"]
        if valid_until != observed_at + max_age:
            raise ValueError(f"{label} freshness window changed")
        if (
            not isinstance(observation["source_identity"], str)
            or not observation["source_identity"]
        ):
            raise ValueError(f"{label} source identity changed")

    failures: list[str] = []
    if not plan["real_read_only_preflight_contract_eligible"]:
        failures.append("outer_package_and_direct_stage_identity_missing")
    if not prefix["query_performed"] or prefix["fixture_only"]:
        failures.append("prefix_external_observations_missing")
    else:
        if not prefix["package_inventory_acceptable"]:
            failures.append("content_addressed_package_inventory_invalid")
        if not prefix["stage_prefix_exactly_empty"]:
            failures.append("direct_v1_stage_prefix_not_empty")
    if not instances["query_performed"] or instances["fixture_only"]:
        failures.append("instance_collision_external_observation_missing")
    elif not instances["all_expected_instance_names_absent"]:
        failures.append("expected_instance_name_collision")
    if (
        not capacity["authoritative_external_observation"]
        or not capacity["query_performed"]
        or capacity["fixture_only"]
        or capacity["available_vcpu"] is None
        or capacity["provider_metric_mapping"] is None
    ):
        failures.append("launch_capacity_external_observation_missing")
    elif not capacity["capacity_sufficient"]:
        failures.append("launch_capacity_insufficient")
    if (
        not price["authoritative_external_observation"]
        or not price["query_performed"]
        or price["fixture_only"]
        or price["observed_price_usd_per_vm_hour"] is None
    ):
        failures.append("spot_price_external_observation_missing")
    elif not price["within_ceiling"]:
        failures.append("spot_price_above_ceiling")
    for label, observation in (
        ("prefix", prefix),
        ("instance", instances),
        ("capacity", capacity),
        ("spot_price", price),
    ):
        if not (
            observation["observed_at_unix_seconds"]
            <= evaluation_unix_seconds
            <= observation["valid_until_unix_seconds"]
        ):
            failures.append(f"{label}_observation_stale_or_from_future")
    observation_contract_passed = not failures
    if "signed_external_query_receipt_missing" not in failures:
        failures.append("signed_external_query_receipt_missing")
    observations = {
        "prefix": prefix,
        "instances": instances,
        "capacity": capacity,
        "spot_price": price,
    }
    result = {
        "schema": PREFLIGHT_RESULT_SCHEMA,
        "status": (
            "local_observation_contract_passed_external_receipt_required"
            if observation_contract_passed
            else "no_go_read_only_preflight"
        ),
        "package_manifest_sha256": plan["package_manifest_sha256"],
        "stage_identity_sha256": plan["stage_identity_sha256"],
        "stage_id": plan["stage_id"],
        "run_name": plan["run_name"],
        "plan_sha256": canonical_sha256(plan),
        "evaluation_unix_seconds": evaluation_unix_seconds,
        "observations": observations,
        "observations_sha256": canonical_sha256(observations),
        "failures": failures,
        "observation_contract_passed": observation_contract_passed,
        # This generic receiver consumes caller-provided observations and can
        # validate their shape only.  A concrete network collector is the sole
        # component allowed to promote this separate read-only gate.
        "read_only_observation_passed": False,
        "preflight_passed": False,
        "package_provisioning_required": prefix[
            "package_provisioning_required"
        ],
        "all_observations_external_and_read_only": False,
        "caller_reported_all_queries_performed": all(
            row["query_performed"] for row in observations.values()
        ),
        "signed_external_query_receipt_present": False,
        "external_read_only_evidence": False,
        "launch_permission_ready": False,
        "launch_permission_failures": [
            "compute.instances.delete_permission_not_observed_by_get_only_preflight",
            "worker_oauth_scope_not_observed_before_vm_creation",
        ],
        "cloud_query_performed": False,
        "cloud_mutation_performed": False,
        "claim_created": False,
        "authorization_created": False,
        "vm_created": False,
        "launch_authorized": False,
        "launch_ready": False,
        "package_provisioning_authorized": False,
        "price_invented_or_cached": False,
        "diagnostic_only": True,
    }
    _reject_scientific_evidence(result)
    return result


def validate_read_only_preflight_result(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    preview: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(value)
    _exact(
        result,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_identity_sha256",
            "stage_id",
            "run_name",
            "plan_sha256",
            "evaluation_unix_seconds",
            "observations",
            "observations_sha256",
            "failures",
            "observation_contract_passed",
            "read_only_observation_passed",
            "preflight_passed",
            "package_provisioning_required",
            "all_observations_external_and_read_only",
            "caller_reported_all_queries_performed",
            "signed_external_query_receipt_present",
            "external_read_only_evidence",
            "launch_permission_ready",
            "launch_permission_failures",
            "cloud_query_performed",
            "cloud_mutation_performed",
            "claim_created",
            "authorization_created",
            "vm_created",
            "launch_authorized",
            "launch_ready",
            "package_provisioning_authorized",
            "price_invented_or_cached",
            "diagnostic_only",
        },
        "read-only preflight result",
    )
    observations = result.get("observations")
    if not isinstance(observations, Mapping):
        raise ValueError("read-only preflight observations are missing")
    expected = evaluate_read_only_preflight(
        plan,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
        prefix_observation=observations.get("prefix", {}),
        instance_observation=observations.get("instances", {}),
        capacity_observation=observations.get("capacity", {}),
        price_observation=observations.get("spot_price", {}),
        evaluation_unix_seconds=result.get("evaluation_unix_seconds"),
    )
    if result != expected:
        raise ValueError("read-only preflight result changed")
    return expected


__all__ = [
    "CAPACITY_PROVIDER_METRIC_MAPPING",
    "CAPACITY_OBSERVATION_SCHEMA",
    "CAPACITY_OBSERVATION_MAX_AGE_SECONDS",
    "CONTROLLER_RECEIPT_SCHEMA",
    "DIRECT_STAGE_IDENTITY_SCHEMA",
    "DIRECT_V1_NAMESPACE",
    "INVENTORY_SCHEMA",
    "INSTANCE_OBSERVATION_MAX_AGE_SECONDS",
    "INSTANCE_OBSERVATION_SCHEMA",
    "MATERIALIZATION_RESULT_SCHEMA",
    "OUTER_PACKAGE_MANIFEST_SCHEMA",
    "PREFLIGHT_PLAN_SCHEMA",
    "PREFLIGHT_RESULT_SCHEMA",
    "PREFIX_OBSERVATION_SCHEMA",
    "PREFIX_OBSERVATION_MAX_AGE_SECONDS",
    "PRICE_OBSERVATION_SCHEMA",
    "PRICE_OBSERVATION_MAX_AGE_SECONDS",
    "REQUIRED_WORKER_OAUTH_SCOPE",
    "ReadOnlyObjectBackend",
    "SPOT_PRICE_CEILING_USD_PER_VM_HOUR",
    "VM_ABSENCE_SCHEMA",
    "build_capacity_observation",
    "build_controller_receipt",
    "build_direct_stage_identity",
    "build_direct_v1_remote_layout",
    "build_instance_observation",
    "build_outer_package_manifest",
    "build_prefix_observation",
    "build_read_only_preflight_plan",
    "build_remote_object_inventory",
    "build_spot_price_observation",
    "build_vm_absence_observation",
    "canonical_bytes",
    "canonical_sha256",
    "evaluate_read_only_preflight",
    "materialize_and_validate_received_stage",
    "validate_materialization_result",
    "validate_direct_stage_identity",
    "validate_outer_package_manifest",
    "validate_read_only_preflight_plan",
    "validate_read_only_preflight_result",
    "validate_remote_object_inventory",
]
