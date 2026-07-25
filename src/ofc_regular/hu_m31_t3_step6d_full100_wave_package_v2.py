"""Immutable outer package for the full-100 8+8+4 wave transport.

The legacy full-100 package is used only as a frozen scientific payload.  Its
launcher and cloud authorization are deliberately not reused.  This module
binds that payload, the v2 wave plan, the v2 startup script, and all twenty
source-isolated job manifests into a content-addressed outer package.

No function in this module invokes a cloud API or changes an AI profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import ctypes
from ctypes.util import find_library
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from . import hu_m31_t3_step6d_full100_spot_v1 as scientific
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 as perf_cloud


OUTER_MANIFEST_SCHEMA = "hu_m31_t3_step6d_full100_wave_outer_package_v2"
OUTER_READY_SCHEMA = "hu_m31_t3_step6d_full100_wave_outer_package_ready_v2"
JOB_BOOTSTRAP_SCHEMA = "hu_m31_t3_step6d_full100_wave_job_bootstrap_v2"

MANIFEST_NAME = "outer_manifest.json"
READY_NAME = "OUTER_PACKAGE_READY.json"
WAVE_PLAN_PATH = "control/wave_plan.json"
SOURCE_PATH = "content/science/source.zip"
SCIENTIFIC_MANIFEST_PATH = "content/science/manifest.json"
WHEELHOUSE_PATH = "content/wheelhouse/wheelhouse.zip"
WHEELHOUSE_MANIFEST_PATH = "content/wheelhouse/wheelhouse_manifest.json"
STARTUP_ROOT = "content/startup"
STARTUP_PATH = (
    f"{STARTUP_ROOT}/"
    f"{PurePosixPath(science_registry.DEVELOPMENT_STARTUP_RELATIVE_PATH).name}"
)
JOB_PATH_TEMPLATE = "content/jobs/{job_id}.json"
CONTENT_PREFIX_ROOT = "hu-m31-t3/full100-wave-v2/content"


def startup_content_path(startup_sha256: str) -> str:
    """Staged object path for the startup script of this lineage.

    Each startup script re-derives the object name it was staged under and
    aborts if it differs, so a registered lineage must be staged under its own
    name. The lineage table lives in the science registry; resolving through it
    means a newly registered startup script cannot be staged under another
    lineage's name by omission. Unregistered digests — synthetic scripts in
    tests — keep the development path they have always used.
    """

    registered = science_registry.startup_relative_paths_by_sha256()
    relative = registered.get(startup_sha256)
    if relative is None:
        return STARTUP_PATH
    return f"{STARTUP_ROOT}/{PurePosixPath(relative).name}"


_SHA = re.compile(r"^[0-9a-f]{64}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]$")
_SERVICE_ACCOUNT = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,61}[a-z0-9]"
    r"\.iam\.gserviceaccount\.com$"
)

_MANIFEST_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "full100_plan_sha256", "run_contract_digest",
        "scientific_lineage", "runtime_binding", "entries", "entry_count",
        "wheelhouse_binding", "expected_startup_sha256",
        "content_payload_sha256", "content_prefix", "content_create_only",
        "legacy_launcher_authorized", "cloud_launch_authorized",
        "cloud_started", "performance_lock_authorized",
        "quality_pilot_authorized", "training_eligible",
        "current_profile_changed", "named_profile_added",
        "runtime_policy_activated", "manifest_sha256",
    }
)
_LINEAGE_KEYS = frozenset(
    {
        "legacy_run_name", "legacy_manifest_sha256", "legacy_ready_sha256",
        "legacy_manifest_bytes", "source_sha256", "source_bytes", "startup_ignored",
        "launch_authorization_ignored", "launcher_reuse_forbidden",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "relative_path", "object_name", "kind", "sha256", "bytes",
        "job_id", "source_role", "shard_index", "work_hand_indices",
    }
)
_READY_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "manifest_sha256", "wave_plan_sha256", "content_payload_sha256",
        "entry_count", "cloud_started", "current_profile_changed",
    }
)
_BOOTSTRAP_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "attempt_ledger_sha256", "resume_sha256",
        "observed_transition_digest", "wave_index", "job_id", "source_role",
        "attempt_id", "instance_name", "artifact_prefix", "bucket",
        "content_prefix", "outer_manifest_sha256", "content_payload_sha256",
        "scientific_source", "scientific_manifest", "wheelhouse",
        "wheelhouse_manifest", "startup", "wave_plan", "job_manifest",
        "prelaunch_authorization_sha256", "worker_principal",
        "one_vm_one_job_one_role", "additional_create_authorized",
        "hidden_truth_exposed", "bootstrap_sha256",
    }
)
_OBJECT_BINDING_KEYS = frozenset({"object_name", "sha256", "bytes"})
_WHEELHOUSE_BINDING_KEYS = frozenset(
    {
        "archive_sha256", "archive_bytes", "manifest_sha256",
        "manifest_bytes", "requirements_sha256", "offline_install_only",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_link_components(path: Path, *, label: str) -> None:
    absolute = _lexical_absolute(path)
    chain = [absolute, *absolute.parents]
    for component in reversed(chain):
        if not component.exists():
            continue
        try:
            component.lstat()
        except OSError as exc:
            raise ValueError(f"{label} component cannot be inspected") from exc
        if _is_link_or_junction(component):
            raise ValueError(f"{label} contains a symlink or junction")


def _safe_existing_file(path: str | Path, label: str) -> Path:
    lexical = _lexical_absolute(path)
    _assert_no_link_components(lexical, label=label)
    try:
        mode = lexical.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is missing or unsafe")
    return lexical.resolve(strict=True)


def _safe_existing_directory(path: str | Path, label: str) -> Path:
    lexical = _lexical_absolute(path)
    _assert_no_link_components(lexical, label=label)
    try:
        mode = lexical.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is missing or unsafe")
    return lexical.resolve(strict=True)


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    safe = _safe_existing_file(path, label)
    raw = safe.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_scientific_package_manifest(
    path: Path,
    label: str,
    descriptor: science_registry.ScienceDescriptor,
) -> dict[str, Any]:
    safe = _safe_existing_file(path, label)
    raw = safe.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or descriptor.package_canonical_bytes(value) != raw
    ):
        raise ValueError(f"{label} is not canonical package JSON")
    return descriptor.validate_package_manifest_facade(value)


def _safe_file(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("outer package path is unsafe")
    resolved_root = _safe_existing_directory(root, "outer package owned root")
    candidate = resolved_root.joinpath(*relative.split("/"))
    _assert_no_link_components(candidate, label="outer package path")
    resolved = _safe_existing_file(candidate, f"outer package file {relative}")
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("outer package path escapes the owned root") from exc
    return resolved


def _source_record(
    relative_path: str,
    kind: str,
    sha256: str,
    size: int,
    *,
    job_id: str | None = None,
    source_role: str | None = None,
    shard_index: int | None = None,
    work_hand_indices: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "object_name": "",  # filled after the content digest is frozen
        "kind": kind,
        "sha256": _require_sha(sha256, f"{kind} SHA-256"),
        "bytes": _require_positive_int(size, f"{kind} bytes"),
        "job_id": job_id,
        "source_role": source_role,
        "shard_index": shard_index,
        "work_hand_indices": deepcopy(work_hand_indices),
    }


def _payload_records(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in _ENTRY_KEYS if key != "object_name"}
        for row in entries
    ]


def _bind_object_names(
    entries: list[dict[str, Any]], content_prefix: str
) -> list[dict[str, Any]]:
    result = deepcopy(entries)
    for row in result:
        row["object_name"] = f"{content_prefix}/{row['relative_path']}"
    return result


def build_outer_manifest(
    *,
    scientific_package_dir: str | Path,
    startup_script: str | Path,
    wheelhouse_archive: str | Path,
    wheelhouse_manifest: str | Path,
    expected_startup_sha256: str,
    wave_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a local, non-authorizing outer content manifest."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    science_descriptor = science_registry.descriptor_for_plan(
        plan["full100_plan"]
    )
    expected_startup = _require_sha(
        expected_startup_sha256, "expected wave v2 startup"
    )
    science_root = _safe_existing_directory(
        scientific_package_dir, "legacy scientific package"
    )
    science = science_descriptor.validate_package(science_root)
    startup = _safe_existing_file(startup_script, "full100 wave v2 startup")
    if sha256_file(startup) != expected_startup:
        raise ValueError("full100 wave v2 startup does not match expected hash")
    wheelhouse = _safe_existing_file(
        wheelhouse_archive, "offline wheelhouse archive"
    )
    wheel_manifest_path = _safe_existing_file(
        wheelhouse_manifest, "offline wheelhouse manifest"
    )
    wheel_manifest = perf_cloud._read_canonical(
        wheel_manifest_path, "offline wheelhouse manifest"
    )
    perf_cloud._validate_wheelhouse_archive(wheelhouse, wheel_manifest)
    requirements_sha = _require_sha(
        wheel_manifest.get("requirements_sha256"),
        "offline wheelhouse requirements",
    )

    source = _safe_file(science_root, str(science["source_name"]))
    science_manifest_path = _safe_file(
        science_root, science_descriptor.package_manifest_name
    )
    science_ready_path = _safe_file(
        science_root, science_descriptor.package_ready_name
    )
    if (
        science["source_sha256"] != sha256_file(source)
        or science["source_bytes"] != source.stat().st_size
        or plan["runtime_binding"]["package_sha256"] != science["source_sha256"]
        or science["plan_sha256"] != plan["full100_plan_sha256"]
        or science["run_contract_digest"] != plan["run_contract_digest"]
    ):
        raise ValueError("legacy scientific payload does not match the wave plan")

    entries = [
        _source_record(
            SOURCE_PATH,
            "scientific_source_archive",
            science["source_sha256"],
            science["source_bytes"],
        ),
        _source_record(
            SCIENTIFIC_MANIFEST_PATH,
            "scientific_source_manifest",
            sha256_file(science_manifest_path),
            science_manifest_path.stat().st_size,
        ),
        _source_record(
            WHEELHOUSE_PATH,
            "offline_wheelhouse_archive",
            sha256_file(wheelhouse),
            wheelhouse.stat().st_size,
        ),
        _source_record(
            WHEELHOUSE_MANIFEST_PATH,
            "offline_wheelhouse_manifest",
            sha256_file(wheel_manifest_path),
            wheel_manifest_path.stat().st_size,
        ),
        _source_record(
            startup_content_path(expected_startup),
            "wave_v2_startup",
            sha256_file(startup),
            startup.stat().st_size,
        ),
        _source_record(
            WAVE_PLAN_PATH,
            "wave_plan",
            hashlib.sha256(wave_v2.canonical_bytes(plan)).hexdigest(),
            len(wave_v2.canonical_bytes(plan)),
        ),
    ]
    science_jobs = {row["job_id"]: row for row in science["job_manifests"]}
    plan_jobs = {row["job_id"]: row for row in plan["full100_plan"]["jobs"]}
    for job_id in plan["coverage"]["job_ids"]:
        record = science_jobs.get(job_id)
        frozen = plan_jobs.get(job_id)
        if record is None or frozen is None:
            raise ValueError("scientific job manifest coverage changed")
        path = _safe_file(science_root, str(record["path"]))
        if (
            record["source_role"] != frozen["source_role"]
            or record["shard_index"] != frozen["shard_index"]
            or record["work_hand_indices"] != frozen["work_hand_indices"]
            or record["sha256"] != sha256_file(path)
            or record["bytes"] != path.stat().st_size
        ):
            raise ValueError("scientific job manifest changed")
        entries.append(
            _source_record(
                JOB_PATH_TEMPLATE.format(job_id=job_id),
                "job_manifest",
                record["sha256"],
                record["bytes"],
                job_id=job_id,
                source_role=record["source_role"],
                shard_index=record["shard_index"],
                work_hand_indices=list(record["work_hand_indices"]),
            )
        )

    payload_sha = canonical_sha256(_payload_records(entries))
    content_prefix = f"{CONTENT_PREFIX_ROOT}/{payload_sha}"
    entries = _bind_object_names(entries, content_prefix)
    lineage = {
        "legacy_run_name": science["run_name"],
        "legacy_manifest_sha256": sha256_file(science_manifest_path),
        "legacy_ready_sha256": sha256_file(science_ready_path),
        "legacy_manifest_bytes": science_manifest_path.stat().st_size,
        "source_sha256": science["source_sha256"],
        "source_bytes": science["source_bytes"],
        "startup_ignored": True,
        "launch_authorization_ignored": True,
        "launcher_reuse_forbidden": True,
    }
    value: dict[str, Any] = {
        "schema": OUTER_MANIFEST_SCHEMA,
        "status": "immutable_outer_content_ready_cloud_not_authorized",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "full100_plan_sha256": plan["full100_plan_sha256"],
        "run_contract_digest": plan["run_contract_digest"],
        "scientific_lineage": lineage,
        "runtime_binding": deepcopy(plan["runtime_binding"]),
        "wheelhouse_binding": {
            "archive_sha256": sha256_file(wheelhouse),
            "archive_bytes": wheelhouse.stat().st_size,
            "manifest_sha256": sha256_file(wheel_manifest_path),
            "manifest_bytes": wheel_manifest_path.stat().st_size,
            "requirements_sha256": requirements_sha,
            "offline_install_only": True,
        },
        "expected_startup_sha256": expected_startup,
        "entries": entries,
        "entry_count": len(entries),
        "content_payload_sha256": payload_sha,
        "content_prefix": content_prefix,
        "content_create_only": True,
        "legacy_launcher_authorized": False,
        "cloud_launch_authorized": False,
        "cloud_started": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return validate_outer_manifest(
        plan, value, expected_startup_sha256=expected_startup
    )


def validate_outer_manifest(
    wave_plan: Mapping[str, Any], value: Mapping[str, Any], *,
    expected_startup_sha256: str,
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    expected_startup = _require_sha(
        expected_startup_sha256, "expected wave v2 startup"
    )
    if not isinstance(value, Mapping):
        raise ValueError("outer manifest must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _MANIFEST_KEYS, "outer manifest")
    digest = payload.pop("manifest_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("outer manifest digest changed")
    payload["manifest_sha256"] = digest
    lineage = payload.get("scientific_lineage")
    entries = payload.get("entries")
    if not isinstance(lineage, Mapping) or not isinstance(entries, list):
        raise ValueError("outer manifest nested records are missing")
    _exact_keys(lineage, _LINEAGE_KEYS, "scientific lineage")
    _require_sha(lineage["legacy_manifest_sha256"], "legacy manifest")
    _require_sha(lineage["legacy_ready_sha256"], "legacy ready")
    _require_sha(lineage["source_sha256"], "scientific source")
    _require_positive_int(lineage["source_bytes"], "scientific source bytes")
    _require_positive_int(
        lineage["legacy_manifest_bytes"], "legacy manifest bytes"
    )
    wheelhouse_binding = payload.get("wheelhouse_binding")
    if not isinstance(wheelhouse_binding, Mapping):
        raise ValueError("offline wheelhouse binding is missing")
    _exact_keys(
        wheelhouse_binding,
        _WHEELHOUSE_BINDING_KEYS,
        "offline wheelhouse binding",
    )
    for field in ("archive_sha256", "manifest_sha256", "requirements_sha256"):
        _require_sha(wheelhouse_binding[field], f"wheelhouse {field}")
    for field in ("archive_bytes", "manifest_bytes"):
        _require_positive_int(wheelhouse_binding[field], f"wheelhouse {field}")
    if wheelhouse_binding["offline_install_only"] is not True:
        raise ValueError("offline wheelhouse execution boundary changed")
    if any(lineage[field] is not True for field in (
        "startup_ignored", "launch_authorization_ignored", "launcher_reuse_forbidden"
    )):
        raise ValueError("legacy launcher boundary changed")
    expected_paths = [
        SOURCE_PATH,
        SCIENTIFIC_MANIFEST_PATH,
        WHEELHOUSE_PATH,
        WHEELHOUSE_MANIFEST_PATH,
        startup_content_path(expected_startup),
        WAVE_PLAN_PATH,
    ] + [
        JOB_PATH_TEMPLATE.format(job_id=job)
        for job in plan["coverage"]["job_ids"]
    ]
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("outer content entry is not an object")
        _exact_keys(row, _ENTRY_KEYS, "outer content entry")
        _require_sha(row["sha256"], "outer content entry")
        _require_positive_int(row["bytes"], "outer content entry bytes")
        if row["work_hand_indices"] is not None and not isinstance(
            row["work_hand_indices"], list
        ):
            raise ValueError("outer content work-hand mapping changed")
    content_sha = canonical_sha256(_payload_records(entries))
    content_prefix = f"{CONTENT_PREFIX_ROOT}/{content_sha}"
    if (
        payload["schema"] != OUTER_MANIFEST_SCHEMA
        or payload["status"] != "immutable_outer_content_ready_cloud_not_authorized"
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"] != plan["execution_identity_sha256"]
        or payload["wave_plan_sha256"] != plan["schedule_sha256"]
        or payload["full100_plan_sha256"] != plan["full100_plan_sha256"]
        or payload["run_contract_digest"] != plan["run_contract_digest"]
        or payload["runtime_binding"] != plan["runtime_binding"]
        or payload["expected_startup_sha256"] != expected_startup
        or lineage["source_sha256"] != plan["runtime_binding"]["package_sha256"]
        or payload["entry_count"] != 26
        or len(entries) != 26
        or [row["relative_path"] for row in entries] != expected_paths
        or len({row["relative_path"] for row in entries}) != len(entries)
        or payload["content_payload_sha256"] != content_sha
        or payload["content_prefix"] != content_prefix
        or any(row["object_name"] != f"{content_prefix}/{row['relative_path']}" for row in entries)
        or payload["content_create_only"] is not True
        or any(payload[field] is not False for field in (
            "legacy_launcher_authorized", "cloud_launch_authorized", "cloud_started",
            "performance_lock_authorized", "quality_pilot_authorized",
            "training_eligible", "current_profile_changed", "named_profile_added",
            "runtime_policy_activated",
        ))
    ):
        raise ValueError("outer package boundary changed")
    expected_jobs = list(plan["coverage"]["job_ids"])
    role_by_job = {
        row["job_id"]: row["source_role"]
        for row in plan["full100_plan"]["jobs"]
    }
    plan_job_by_id = {
        row["job_id"]: row for row in plan["full100_plan"]["jobs"]
    }
    job_entries = entries[6:]
    if (
        entries[0]["sha256"] != lineage["source_sha256"]
        or entries[0]["bytes"] != lineage["source_bytes"]
        or entries[1]["sha256"] != lineage["legacy_manifest_sha256"]
        or entries[1]["bytes"] != lineage["legacy_manifest_bytes"]
        or entries[2]["sha256"] != wheelhouse_binding["archive_sha256"]
        or entries[2]["bytes"] != wheelhouse_binding["archive_bytes"]
        or entries[3]["sha256"] != wheelhouse_binding["manifest_sha256"]
        or entries[3]["bytes"] != wheelhouse_binding["manifest_bytes"]
        or entries[4]["sha256"] != expected_startup
        or entries[5]["sha256"] != hashlib.sha256(
            wave_v2.canonical_bytes(plan)
        ).hexdigest()
        or entries[5]["bytes"] != len(wave_v2.canonical_bytes(plan))
        or [row["job_id"] for row in job_entries] != expected_jobs
        or [row["source_role"] for row in job_entries]
        != [role_by_job[job] for job in expected_jobs]
        or any(row["kind"] != "job_manifest" for row in job_entries)
        or any(
            row["sha256"]
            != plan_job_by_id[row["job_id"]]["shard_manifest_sha256"]
            or row["source_role"]
            != plan_job_by_id[row["job_id"]]["source_role"]
            or row["shard_index"]
            != plan_job_by_id[row["job_id"]]["shard_index"]
            or row["work_hand_indices"]
            != plan_job_by_id[row["job_id"]]["work_hand_indices"]
            for row in job_entries
        )
        or any(
            row["job_id"] is not None
            or row["source_role"] is not None
            or row["shard_index"] is not None
            or row["work_hand_indices"] is not None
            for row in entries[:6]
        )
        or [row["kind"] for row in entries[:6]]
        != [
            "scientific_source_archive",
            "scientific_source_manifest",
            "offline_wheelhouse_archive",
            "offline_wheelhouse_manifest",
            "wave_v2_startup",
            "wave_plan",
        ]
    ):
        raise ValueError("outer package job mapping changed")
    return payload


def _fsync_directory(path: Path) -> None:
    directory = _safe_existing_directory(path, "directory fsync target")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        flush_file_buffers = kernel32.FlushFileBuffers
        flush_file_buffers.argtypes = [ctypes.c_void_p]
        flush_file_buffers.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        handle = create_file(
            str(directory),
            0x40000000,  # GENERIC_WRITE; required to flush directory metadata.
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,  # OPEN_EXISTING
            0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            raise OSError(ctypes.get_last_error(), "directory open for fsync failed")
        try:
            if not flush_file_buffers(handle):
                raise OSError(
                    ctypes.get_last_error(), "directory FlushFileBuffers failed"
                )
        finally:
            close_handle(handle)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> Path:
    target = _lexical_absolute(path)
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            raise ValueError("directory root is unavailable")
        cursor = cursor.parent
    _safe_existing_directory(cursor, "existing directory ancestor")
    for directory in reversed(missing):
        directory.mkdir()
        _assert_no_link_components(directory, label="created package directory")
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    return _safe_existing_directory(target, "package directory")


def _write_once(path: Path, raw: bytes) -> None:
    _ensure_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        raise


def _publish_no_replace(stage: Path, destination: Path) -> None:
    if destination.exists() or _is_link_or_junction(destination):
        raise FileExistsError("outer package destination is immutable")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        # No MOVEFILE_REPLACE_EXISTING: this is create-only.  WRITE_THROUGH
        # provides durable directory metadata before the call returns.
        if not move(str(stage), str(destination), 0x00000008):
            error = ctypes.get_last_error()
            if error in {80, 183}:  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
                raise FileExistsError("outer package destination is immutable")
            raise OSError(error, "create-only outer package publish failed")
        return
    libc_name = find_library("c")
    if not libc_name:
        raise RuntimeError("no create-only rename primitive is available")
    libc = ctypes.CDLL(libc_name, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(stage),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {17}:  # EEXIST
            raise FileExistsError("outer package destination is immutable")
        raise OSError(error, "create-only outer package publish failed")


def _expected_tree(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    files = {
        *(row["relative_path"] for row in manifest["entries"]),
        MANIFEST_NAME,
        READY_NAME,
    }
    directories = {"."}
    for relative in files:
        parent = Path(relative).parent
        while str(parent) not in {"", "."}:
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _validate_exact_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    safe_root = _safe_existing_directory(root, "outer package root")
    expected_files, expected_directories = _expected_tree(manifest)
    observed_files: set[str] = set()
    observed_directories: set[str] = {"."}
    for current, directory_names, file_names in os.walk(
        safe_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        _assert_no_link_components(current_path, label="outer package tree")
        for name in directory_names:
            child = current_path / name
            if _is_link_or_junction(child):
                raise ValueError("outer package tree contains a symlink or junction")
            observed_directories.add(child.relative_to(safe_root).as_posix())
        for name in file_names:
            child = current_path / name
            if _is_link_or_junction(child) or not stat.S_ISREG(child.lstat().st_mode):
                raise ValueError("outer package tree contains an unsafe file")
            observed_files.add(child.relative_to(safe_root).as_posix())
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("outer package exact tree has extra or missing paths")


def materialize_outer_package(
    *,
    output_dir: str | Path,
    scientific_package_dir: str | Path,
    startup_script: str | Path,
    wheelhouse_archive: str | Path,
    wheelhouse_manifest: str | Path,
    expected_startup_sha256: str,
    wave_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the outer package once; existing destinations fail closed."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    science_descriptor = science_registry.descriptor_for_plan(
        plan["full100_plan"]
    )
    manifest = build_outer_manifest(
        scientific_package_dir=scientific_package_dir,
        startup_script=startup_script,
        wheelhouse_archive=wheelhouse_archive,
        wheelhouse_manifest=wheelhouse_manifest,
        expected_startup_sha256=expected_startup_sha256,
        wave_plan=plan,
    )
    destination = _lexical_absolute(output_dir)
    if destination.exists() or _is_link_or_junction(destination):
        raise FileExistsError("outer package destination is immutable")
    _ensure_directory(destination.parent)
    _assert_no_link_components(destination.parent, label="outer package parent")
    stage = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    )
    science_root = _safe_existing_directory(
        scientific_package_dir, "legacy scientific package"
    )
    startup = _safe_existing_file(startup_script, "full100 wave v2 startup")
    wheelhouse = _safe_existing_file(
        wheelhouse_archive, "offline wheelhouse archive"
    )
    wheel_manifest_path = _safe_existing_file(
        wheelhouse_manifest, "offline wheelhouse manifest"
    )
    science_manifest = science_descriptor.validate_package(science_root)
    sources: dict[str, Path | bytes] = {
        SOURCE_PATH: _safe_file(science_root, science_manifest["source_name"]),
        SCIENTIFIC_MANIFEST_PATH: _safe_file(
            science_root, science_descriptor.package_manifest_name
        ),
        WHEELHOUSE_PATH: wheelhouse,
        WHEELHOUSE_MANIFEST_PATH: wheel_manifest_path,
        startup_content_path(manifest["expected_startup_sha256"]): startup,
        WAVE_PLAN_PATH: wave_v2.canonical_bytes(plan),
    }
    for record in science_manifest["job_manifests"]:
        sources[JOB_PATH_TEMPLATE.format(job_id=record["job_id"])] = _safe_file(
            science_root, record["path"]
        )
    stage_created = False
    published = False
    validated: dict[str, Any] | None = None
    try:
        stage.mkdir()
        stage_created = True
        _fsync_directory(stage)
        _fsync_directory(stage.parent)
        for row in manifest["entries"]:
            target = stage.joinpath(*row["relative_path"].split("/"))
            source = sources[row["relative_path"]]
            raw = source if isinstance(source, bytes) else Path(source).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["bytes"]:
                raise ValueError("outer package source changed while materializing")
            _write_once(target, raw)
        _write_once(stage / MANIFEST_NAME, canonical_bytes(manifest))
        ready = {
            "schema": OUTER_READY_SCHEMA,
            "status": "immutable_outer_package_complete_cloud_not_authorized",
            "run_name": plan["run_name"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "wave_plan_sha256": plan["schedule_sha256"],
            "content_payload_sha256": manifest["content_payload_sha256"],
            "entry_count": manifest["entry_count"],
            "cloud_started": False,
            "current_profile_changed": False,
        }
        _write_once(stage / READY_NAME, canonical_bytes(ready))
        validate_outer_package(
            stage,
            plan,
            expected_startup_sha256=expected_startup_sha256,
        )
        _fsync_directory(stage)
        _fsync_directory(stage.parent)
        _publish_no_replace(stage, destination)
        published = True
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        validated = validate_outer_package(
            destination,
            plan,
            expected_startup_sha256=expected_startup_sha256,
        )
    except BaseException:
        if stage_created and stage.exists() and not _is_link_or_junction(stage):
            shutil.rmtree(stage, ignore_errors=False)
            _fsync_directory(stage.parent)
        if published and destination.exists() and not _is_link_or_junction(destination):
            shutil.rmtree(destination, ignore_errors=False)
            _fsync_directory(destination.parent)
        raise
    if validated is None:  # defensive: every successful publish is validated.
        raise RuntimeError("outer package publish completed without validation")
    return validated


def validate_outer_package(
    package_dir: str | Path, wave_plan: Mapping[str, Any], *,
    expected_startup_sha256: str,
) -> dict[str, Any]:
    root = _safe_existing_directory(package_dir, "outer package root")
    plan = wave_v2.validate_wave_plan(wave_plan)
    science_descriptor = science_registry.descriptor_for_plan(
        plan["full100_plan"]
    )
    manifest = validate_outer_manifest(
        plan,
        _read_canonical(_safe_file(root, MANIFEST_NAME), "outer manifest"),
        expected_startup_sha256=expected_startup_sha256,
    )
    ready = _read_canonical(
        _safe_file(root, READY_NAME), "outer package ready"
    )
    _exact_keys(ready, _READY_KEYS, "outer package ready")
    _validate_exact_tree(root, manifest)
    for row in manifest["entries"]:
        path = _safe_file(root, row["relative_path"])
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("outer package content changed")
    expected_ready = {
        "schema": OUTER_READY_SCHEMA,
        "status": "immutable_outer_package_complete_cloud_not_authorized",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "content_payload_sha256": manifest["content_payload_sha256"],
        "entry_count": manifest["entry_count"],
        "cloud_started": False,
        "current_profile_changed": False,
    }
    if ready != expected_ready:
        raise ValueError("outer package ready boundary changed")
    scientific_manifest = _read_scientific_package_manifest(
        _safe_file(root, SCIENTIFIC_MANIFEST_PATH),
        "outer package scientific manifest",
        science_descriptor,
    )
    scientific_jobs = scientific_manifest.get("job_manifests")
    frozen_jobs = plan["full100_plan"]["jobs"]
    if (
        scientific_manifest.get("source_sha256")
        != manifest["scientific_lineage"]["source_sha256"]
        or scientific_manifest.get("source_bytes")
        != manifest["scientific_lineage"]["source_bytes"]
        or scientific_manifest.get("plan_sha256") != plan["full100_plan_sha256"]
        or scientific_manifest.get("run_contract_digest")
        != plan["run_contract_digest"]
        or not isinstance(scientific_jobs, list)
        or len(scientific_jobs) != len(frozen_jobs)
    ):
        raise ValueError("outer package scientific lineage changed")
    entry_by_job = {
        row["job_id"]: row for row in manifest["entries"][6:]
    }
    for record, frozen in zip(scientific_jobs, frozen_jobs, strict=True):
        entry = entry_by_job[frozen["job_id"]]
        output_prefix_valid = isinstance(record, Mapping) and (
            (
                record.get("output_prefix")
                == f"jobs/{frozen['job_id']}"
            )
            if science_descriptor.legacy_development_identity
            else "output_prefix" not in record
        )
        if (
            not isinstance(record, Mapping)
            or record.get("job_id") != frozen["job_id"]
            or record.get("source_role") != frozen["source_role"]
            or record.get("shard_index") != frozen["shard_index"]
            or record.get("work_hand_indices") != frozen["work_hand_indices"]
            or record.get("path") != f"jobs/{frozen['job_id']}.json"
            or not output_prefix_valid
            or record.get("sha256") != frozen["shard_manifest_sha256"]
            or record.get("sha256") != entry["sha256"]
            or record.get("bytes") != entry["bytes"]
        ):
            raise ValueError("outer package scientific job lineage changed")
    wheel_manifest_path = _safe_file(root, WHEELHOUSE_MANIFEST_PATH)
    wheel_manifest = perf_cloud._read_canonical(
        wheel_manifest_path, "outer package wheelhouse manifest"
    )
    perf_cloud._validate_wheelhouse_archive(
        _safe_file(root, WHEELHOUSE_PATH), wheel_manifest
    )
    if (
        wheel_manifest.get("requirements_sha256")
        != manifest["wheelhouse_binding"]["requirements_sha256"]
    ):
        raise ValueError("outer package wheelhouse requirements binding changed")
    return manifest


def _entry_by_kind(
    manifest: Mapping[str, Any], kind: str, *, job_id: str | None = None
) -> dict[str, Any]:
    matches = [
        row for row in manifest["entries"]
        if row["kind"] == kind and (job_id is None or row["job_id"] == job_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"outer package has ambiguous {kind} content")
    return matches[0]


def _object_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"object_name": row["object_name"], "sha256": row["sha256"], "bytes": row["bytes"]}


def build_job_bootstrap_metadata(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    job_id: str,
    bucket: str,
    worker_principal: str,
    prelaunch_authorization_sha256: str,
    expected_startup_sha256: str,
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    manifest = validate_outer_manifest(
        plan,
        outer_manifest,
        expected_startup_sha256=expected_startup_sha256,
    )
    selected = [row for row in resume["selected_attempts"] if row["job_id"] == job_id]
    if len(selected) != 1 or resume["all_jobs_complete"] is True:
        raise ValueError("job is not selected by the current resume plan")
    if not isinstance(bucket, str) or _BUCKET.fullmatch(bucket) is None:
        raise ValueError("bucket name is unsafe")
    if not isinstance(worker_principal, str) or _SERVICE_ACCOUNT.fullmatch(worker_principal) is None:
        raise ValueError("worker service account is unsafe")
    auth_sha = _require_sha(prelaunch_authorization_sha256, "prelaunch authorization")
    attempt = selected[0]
    value: dict[str, Any] = {
        "schema": JOB_BOOTSTRAP_SCHEMA,
        "status": "single_job_bootstrap_bound_to_prelaunch_authorization",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "observed_transition_digest": resume["observed_transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "job_id": job_id,
        "source_role": attempt["source_role"],
        "attempt_id": attempt["attempt_id"],
        "instance_name": attempt["instance_id"],
        "artifact_prefix": attempt["artifact_prefix"],
        "bucket": bucket,
        "content_prefix": manifest["content_prefix"],
        "outer_manifest_sha256": manifest["manifest_sha256"],
        "content_payload_sha256": manifest["content_payload_sha256"],
        "scientific_source": _object_binding(_entry_by_kind(manifest, "scientific_source_archive")),
        "scientific_manifest": _object_binding(_entry_by_kind(manifest, "scientific_source_manifest")),
        "wheelhouse": _object_binding(_entry_by_kind(manifest, "offline_wheelhouse_archive")),
        "wheelhouse_manifest": _object_binding(_entry_by_kind(manifest, "offline_wheelhouse_manifest")),
        "startup": _object_binding(_entry_by_kind(manifest, "wave_v2_startup")),
        "wave_plan": _object_binding(_entry_by_kind(manifest, "wave_plan")),
        "job_manifest": _object_binding(_entry_by_kind(manifest, "job_manifest", job_id=job_id)),
        "prelaunch_authorization_sha256": auth_sha,
        "worker_principal": worker_principal,
        "one_vm_one_job_one_role": True,
        "additional_create_authorized": False,
        "hidden_truth_exposed": False,
    }
    value["bootstrap_sha256"] = canonical_sha256(value)
    return validate_job_bootstrap_metadata(
        plan,
        ledger,
        resume,
        manifest,
        value,
        expected_startup_sha256=expected_startup_sha256,
    )


def validate_job_bootstrap_metadata(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    expected_startup_sha256: str,
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    manifest = validate_outer_manifest(
        plan,
        outer_manifest,
        expected_startup_sha256=expected_startup_sha256,
    )
    if not isinstance(value, Mapping):
        raise ValueError("job bootstrap metadata must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _BOOTSTRAP_KEYS, "job bootstrap metadata")
    digest = payload.pop("bootstrap_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("job bootstrap metadata digest changed")
    payload["bootstrap_sha256"] = digest
    for field in (
        "scientific_source",
        "scientific_manifest",
        "wheelhouse",
        "wheelhouse_manifest",
        "startup",
        "wave_plan",
        "job_manifest",
    ):
        binding = payload.get(field)
        if not isinstance(binding, Mapping):
            raise ValueError("job bootstrap object binding is missing")
        _exact_keys(binding, _OBJECT_BINDING_KEYS, "job bootstrap object binding")
        _require_sha(binding["sha256"], "job bootstrap object")
        _require_positive_int(binding["bytes"], "job bootstrap object bytes")
    selected = [row for row in resume["selected_attempts"] if row["job_id"] == payload.get("job_id")]
    if len(selected) != 1:
        raise ValueError("job bootstrap is outside the selected wave")
    row = selected[0]
    expected_bindings = {
        "scientific_source": _object_binding(_entry_by_kind(manifest, "scientific_source_archive")),
        "scientific_manifest": _object_binding(_entry_by_kind(manifest, "scientific_source_manifest")),
        "wheelhouse": _object_binding(_entry_by_kind(manifest, "offline_wheelhouse_archive")),
        "wheelhouse_manifest": _object_binding(_entry_by_kind(manifest, "offline_wheelhouse_manifest")),
        "startup": _object_binding(_entry_by_kind(manifest, "wave_v2_startup")),
        "wave_plan": _object_binding(_entry_by_kind(manifest, "wave_plan")),
        "job_manifest": _object_binding(_entry_by_kind(manifest, "job_manifest", job_id=row["job_id"])),
    }
    if (
        payload["schema"] != JOB_BOOTSTRAP_SCHEMA
        or payload["status"] != "single_job_bootstrap_bound_to_prelaunch_authorization"
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"] != plan["execution_identity_sha256"]
        or payload["wave_plan_sha256"] != plan["schedule_sha256"]
        or payload["attempt_ledger_sha256"] != ledger["ledger_sha256"]
        or payload["resume_sha256"] != resume["resume_sha256"]
        or payload["observed_transition_digest"] != resume["observed_transition_digest"]
        or payload["wave_index"] != resume["resume_wave_index"]
        or payload["source_role"] != row["source_role"]
        or payload["attempt_id"] != row["attempt_id"]
        or payload["instance_name"] != row["instance_id"]
        or payload["artifact_prefix"] != row["artifact_prefix"]
        or not isinstance(payload["bucket"], str)
        or _BUCKET.fullmatch(payload["bucket"]) is None
        or payload["content_prefix"] != manifest["content_prefix"]
        or payload["outer_manifest_sha256"] != manifest["manifest_sha256"]
        or payload["content_payload_sha256"] != manifest["content_payload_sha256"]
        or any(payload[field] != binding for field, binding in expected_bindings.items())
        or _SHA.fullmatch(str(payload["prelaunch_authorization_sha256"])) is None
        or _SERVICE_ACCOUNT.fullmatch(str(payload["worker_principal"])) is None
        or payload["one_vm_one_job_one_role"] is not True
        or payload["additional_create_authorized"] is not False
        or payload["hidden_truth_exposed"] is not False
    ):
        raise ValueError("job bootstrap boundary changed")
    return payload


__all__ = [
    "JOB_BOOTSTRAP_SCHEMA", "MANIFEST_NAME", "OUTER_MANIFEST_SCHEMA",
    "OUTER_READY_SCHEMA", "READY_NAME", "build_job_bootstrap_metadata",
    "build_outer_manifest", "canonical_bytes", "canonical_sha256",
    "materialize_outer_package", "sha256_file",
    "validate_job_bootstrap_metadata", "validate_outer_manifest",
    "validate_outer_package",
]
