"""Fail-closed GCP lifecycle for the frozen HU RL C4 formal benchmark.

The formal benchmark harness deliberately contains no cloud lifecycle code.  This
module is the external controller boundary.  It stages one byte-locked package,
creates exactly one owned C4 Spot VM, derives the existing external attestation
from authenticated read-only GCE responses, collects and validates the formal
result, and deletes only the measured owned instance.

No function in this module mutates an AI profile.  Cloud mutations require a
separate short-lived authorization receipt and a raw UUIDv4 nonce.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import ssl
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

from ofc_regular.hu_rl_c4_formal_benchmark import (
    EXTERNAL_GCE_OBSERVATION_SOURCE,
    HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
    MAX_CANONICAL_INPUT_BYTES,
    TARGET_BOOT_DISK_TYPE,
    TARGET_MACHINE_TYPE,
    TARGET_NIC_TYPE,
    TARGET_PROJECT_ID,
    TARGET_PROVISIONING_MODEL,
    TARGET_VCPU_COUNT,
    TARGET_ZONE,
    HuRlC4FormalBenchmarkError,
    build_c4_machine_attestation,
    canonical_c4_json,
    load_canonical_c4_json,
    validate_c4_external_gce_observation,
    validate_c4_formal_benchmark_manifest,
    validate_c4_formal_benchmark_result,
)


PLAN_SCHEMA: Final = "regular_ofc_hu_rl_c4_gcp_lifecycle_plan_v1"
AUTHORIZATION_SCHEMA: Final = "regular_ofc_hu_rl_c4_gcp_operation_authorization_v1"
LAUNCH_RECEIPT_SCHEMA: Final = "regular_ofc_hu_rl_c4_gcp_launch_receipt_v1"
COLLECTION_RECEIPT_SCHEMA: Final = "regular_ofc_hu_rl_c4_gcp_collection_receipt_v1"
CLEANUP_RECEIPT_SCHEMA: Final = "regular_ofc_hu_rl_c4_gcp_cleanup_receipt_v1"
WORKER_STATUS_SCHEMA: Final = "regular_ofc_hu_rl_c4_gcp_worker_status_v1"

PROJECT: Final = TARGET_PROJECT_ID
ZONE: Final = TARGET_ZONE
REGION: Final = "asia-northeast1"
# A label-generation fleet outgrows any single region's Spot CPU quota, so the
# same run may be sharded across several. These are the zones that verifiably
# offer the c4 machine family and hyperdisk-balanced, and whose regions carry
# the auto-mode VPC's "default" subnet. ZONE stays first and stays the default
# for every caller, so nothing moves unless a caller names another zone.
APPROVED_ZONES: Final = (
    ZONE,
    "us-east1-b",
    "us-west1-a",
    "us-east4-a",
    "europe-west4-a",
)
MACHINE_TYPE: Final = TARGET_MACHINE_TYPE
INSTANCE_COUNT: Final = 1
BOOT_DISK_TYPE: Final = TARGET_BOOT_DISK_TYPE
BOOT_DISK_INTERFACE: Final = "NVME"
BOOT_DISK_SIZE_GB: Final = 20
NIC_TYPE: Final = TARGET_NIC_TYPE
PROVISIONING_MODEL: Final = TARGET_PROVISIONING_MODEL
MAX_RUN_DURATION_SECONDS: Final = 4_500
WORKER_WATCHDOG_SECONDS: Final = 4_200
NETWORK_NAME: Final = "default"
SUBNETWORK_NAME: Final = "default"
STORAGE_SCOPE: Final = "https://www.googleapis.com/auth/devstorage.read_write"
SNAPSHOT_TIMESTAMP: Final = "20260722T000000Z"

OPERATION_POLL_ATTEMPTS: Final = 120
OPERATION_POLL_INTERVAL_SECONDS: Final = 2.0
INSTANCE_POLL_ATTEMPTS: Final = 90
INSTANCE_POLL_INTERVAL_SECONDS: Final = 2.0
RESULT_POLL_ATTEMPTS: Final = 420
RESULT_POLL_INTERVAL_SECONDS: Final = 10.0
HTTP_TIMEOUT_SECONDS: Final = 60
# Reads only. A long poll can span minutes, so a single dropped or throttled
# response should not end an operation that is otherwise healthy.
_READ_RETRY_ATTEMPTS: Final = 4
_READ_RETRY_BACKOFF_SECONDS: Final = 2.0
_RETRYABLE_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
UPLOAD_TIMEOUT_SECONDS: Final = 600

_RUN_NAME_RE = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")
_INSTANCE_RE = _RUN_NAME_RE
_SERVICE_ACCOUNT_RE = re.compile(
    r"[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com"
)
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class HuRlC4GcpLifecycleError(RuntimeError):
    """The lifecycle contract or measured cloud state is unsafe or invalid."""


class ResponseLostError(HuRlC4GcpLifecycleError):
    """A mutating request may have reached GCP but its response was lost."""


@dataclasses.dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


class LifecycleTransport(Protocol):
    """Minimal cloud boundary used by the lifecycle and by local fakes."""

    def get_object_metadata(self, *, bucket: str, object_name: str) -> Mapping[str, Any] | None: ...

    def get_object_bytes(self, *, bucket: str, object_name: str, generation: str | None = None) -> bytes | None: ...

    def put_object_new(
        self, *, bucket: str, object_name: str, payload: bytes, content_type: str
    ) -> Mapping[str, Any]: ...

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None: ...

    def create_instance(
        self, *, instance_spec: Mapping[str, Any], request_id: str
    ) -> Mapping[str, Any]: ...

    def get_zone_operation(self, *, operation_name: str) -> Mapping[str, Any]: ...

    def get_machine_type(self, *, machine_type: str) -> Mapping[str, Any]: ...

    def get_disk(self, *, disk_name: str) -> Mapping[str, Any]: ...

    def delete_instance(
        self, *, instance_name: str, request_id: str
    ) -> Mapping[str, Any]: ...


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise HuRlC4GcpLifecycleError("lifecycle artifact is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = dict(value)
    copied[field] = None
    return canonical_sha256(copied)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_uuid4(value: str, label: str) -> uuid.UUID:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HuRlC4GcpLifecycleError(f"{label} must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise HuRlC4GcpLifecycleError(f"{label} must be a canonical UUIDv4")
    return parsed


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_tar_inventory(archive_path: Path, expected_root: str) -> dict[str, str]:
    files: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise HuRlC4GcpLifecycleError("package archive contains an unsafe path")
                if path.parts[0] != expected_root:
                    raise HuRlC4GcpLifecycleError("package archive root changed")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise HuRlC4GcpLifecycleError("package archive contains a non-regular entry")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise HuRlC4GcpLifecycleError("package archive entry type is unsupported")
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise HuRlC4GcpLifecycleError("package archive file could not be read")
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                if relative in files:
                    raise HuRlC4GcpLifecycleError("package archive contains a duplicate file")
                files[relative] = digest.hexdigest()
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, HuRlC4GcpLifecycleError):
            raise
        raise HuRlC4GcpLifecycleError("package archive is unreadable") from exc
    return files


def _directory_inventory(package_root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            raise HuRlC4GcpLifecycleError("package directory contains a symlink")
        if path.is_file():
            relative = path.relative_to(package_root).as_posix()
            files[relative] = sha256_file(path)
    return files


def verify_frozen_package(
    *, package_root: Path, archive_path: Path, expected_archive_sha256: str
) -> dict[str, Any]:
    """Verify the local directory, tarball, formal manifest, sources, and wheel."""

    if not _is_sha256(expected_archive_sha256):
        raise HuRlC4GcpLifecycleError("expected package SHA-256 is invalid")
    if not package_root.is_dir() or not archive_path.is_file():
        raise HuRlC4GcpLifecycleError("package root or archive is missing")
    actual_archive_sha256 = sha256_file(archive_path)
    if actual_archive_sha256 != expected_archive_sha256:
        raise HuRlC4GcpLifecycleError("package archive SHA-256 mismatch")
    manifest_path = package_root / "hu_rl_c4_formal_benchmark_manifest.json"
    manifest = load_canonical_c4_json(manifest_path, context="C4 package manifest")
    validate_c4_formal_benchmark_manifest(manifest)

    directory_files = _directory_inventory(package_root)
    archive_files = _safe_tar_inventory(archive_path, package_root.name)
    if archive_files != directory_files:
        raise HuRlC4GcpLifecycleError("package archive and directory inventories differ")

    expected_sources = {
        **manifest["benchmark_provenance"]["source_sha256"],
        **manifest["harness_source_sha256"],
    }
    for relative, expected in expected_sources.items():
        if directory_files.get(relative) != expected:
            raise HuRlC4GcpLifecycleError(f"package source binding changed: {relative}")
    wheel_name = manifest["benchmark_provenance"]["wheel_filename"]
    wheel_relative = f"wheelhouse/{wheel_name}"
    if directory_files.get(wheel_relative) != manifest["benchmark_provenance"]["wheel_sha256"]:
        raise HuRlC4GcpLifecycleError("package native wheel binding changed")
    wheel_path = package_root / wheel_relative
    extension_name = manifest["benchmark_provenance"]["native_extension_filename"]
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            candidates = [name for name in wheel.namelist() if PurePosixPath(name).name == extension_name]
            if len(candidates) != 1:
                raise HuRlC4GcpLifecycleError("native extension wheel member changed")
            extension_sha = hashlib.sha256(wheel.read(candidates[0])).hexdigest()
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        raise HuRlC4GcpLifecycleError("native wheel is unreadable") from exc
    if extension_sha != manifest["benchmark_provenance"]["native_extension_sha256"]:
        raise HuRlC4GcpLifecycleError("native extension SHA-256 changed")

    return {
        "archive_filename": archive_path.name,
        "archive_sha256": actual_archive_sha256,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_root": package_root.name,
        "file_count": len(directory_files),
        "files_sha256": canonical_sha256(directory_files),
        "manifest_sha256": manifest["manifest_sha256"],
        "native_wheel_sha256": manifest["benchmark_provenance"]["wheel_sha256"],
    }


def _validate_image(source_image_self_link: str, image_id: str, guest_os_features: Sequence[str]) -> None:
    if (
        not source_image_self_link.startswith("https://www.googleapis.com/compute/v1/projects/")
        or "/global/images/" not in source_image_self_link
    ):
        raise HuRlC4GcpLifecycleError("source image self link is invalid")
    if re.fullmatch(r"[1-9][0-9]*", image_id) is None:
        raise HuRlC4GcpLifecycleError("source image id is invalid")
    if not guest_os_features or any(not isinstance(item, str) or not item for item in guest_os_features):
        raise HuRlC4GcpLifecycleError("source image guest OS features are invalid")
    if "GVNIC" not in guest_os_features:
        raise HuRlC4GcpLifecycleError("source image lacks GVNIC support")


def _instance_name(run_name: str, identity_sha256: str) -> str:
    suffix = identity_sha256[:10]
    prefix = run_name[: 63 - len(suffix) - 1].rstrip("-")
    name = f"{prefix}-{suffix}"
    if _INSTANCE_RE.fullmatch(name) is None:
        raise HuRlC4GcpLifecycleError("derived instance name is invalid")
    return name


def build_execution_plan(
    *,
    run_name: str,
    package_root: Path,
    archive_path: Path,
    expected_archive_sha256: str,
    bucket: str,
    worker_service_account: str,
    source_image_self_link: str,
    image_id: str,
    guest_os_features: Sequence[str],
) -> dict[str, Any]:
    """Build one immutable exact-one-VM plan without cloud calls."""

    if _RUN_NAME_RE.fullmatch(run_name) is None:
        raise HuRlC4GcpLifecycleError("run name is invalid")
    if _BUCKET_RE.fullmatch(bucket) is None:
        raise HuRlC4GcpLifecycleError("bucket name is invalid")
    if _SERVICE_ACCOUNT_RE.fullmatch(worker_service_account) is None:
        raise HuRlC4GcpLifecycleError("worker service account is invalid")
    _validate_image(source_image_self_link, image_id, guest_os_features)
    package = verify_frozen_package(
        package_root=package_root,
        archive_path=archive_path,
        expected_archive_sha256=expected_archive_sha256,
    )
    identity_core = {
        "schema": PLAN_SCHEMA,
        "run_name": run_name,
        "project": PROJECT,
        "zone": ZONE,
        "machine_type": MACHINE_TYPE,
        "bucket": bucket,
        "worker_service_account": worker_service_account,
        "package": package,
        "image": {
            "source_image_self_link": source_image_self_link,
            "image_id": image_id,
            "guest_os_features": sorted(set(guest_os_features)),
        },
    }
    identity_sha256 = canonical_sha256(identity_core)
    instance_name = _instance_name(run_name, identity_sha256)
    prefix = f"hu-rl-c4/{run_name}/{identity_sha256}/"
    owner_label = f"rlc-{identity_sha256[:28]}"
    plan_label = identity_sha256[:32]
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "exact_one_c4_spot_plan_ready",
        "run_name": run_name,
        "project": PROJECT,
        "region": REGION,
        "zone": ZONE,
        "machine_type": MACHINE_TYPE,
        "instance_count": INSTANCE_COUNT,
        "instance_name": instance_name,
        "owner_label": owner_label,
        "plan_label": plan_label,
        "bucket": bucket,
        "objects": {
            "prefix": prefix,
            "package": prefix + "input/package.tar.gz",
            "attestation": prefix + "control/machine_attestation.json",
            "result": prefix + "result/hu_rl_c4_formal_benchmark_result.json",
            "worker_status": prefix + "result/worker_status.json",
        },
        "worker_service_account": worker_service_account,
        "package": package,
        "image": identity_core["image"],
        "network": {
            "network": f"global/networks/{NETWORK_NAME}",
            "subnetwork": f"regions/{REGION}/subnetworks/{SUBNETWORK_NAME}",
            "nic_type": NIC_TYPE,
            "external_ip": False,
        },
        "vm": {
            "provisioning_model": PROVISIONING_MODEL,
            "preemptible": True,
            "automatic_restart": False,
            "on_host_maintenance": "TERMINATE",
            "instance_termination_action": "DELETE",
            "max_run_duration_seconds": MAX_RUN_DURATION_SECONDS,
            "worker_watchdog_seconds": WORKER_WATCHDOG_SECONDS,
            "deletion_protection": False,
            "boot_disk_type": BOOT_DISK_TYPE,
            "boot_disk_interface": BOOT_DISK_INTERFACE,
            "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
            "boot_disk_auto_delete": True,
        },
        "formal_manifest_sha256": package["manifest_sha256"],
        "plan_identity_sha256": identity_sha256,
        "current_profile_changed": False,
        "execution_plan_sha256": None,
    }
    plan["execution_plan_sha256"] = _self_digest(plan, "execution_plan_sha256")
    validate_execution_plan(plan)
    return plan


def validate_execution_plan(plan: Mapping[str, Any]) -> None:
    required = {
        "schema", "status", "run_name", "project", "region", "zone", "machine_type",
        "instance_count", "instance_name", "owner_label", "plan_label", "bucket", "objects",
        "worker_service_account", "package", "image", "network", "vm",
        "formal_manifest_sha256", "plan_identity_sha256", "current_profile_changed",
        "execution_plan_sha256",
    }
    if set(plan) != required:
        raise HuRlC4GcpLifecycleError("execution plan fields changed")
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["status"] != "exact_one_c4_spot_plan_ready"
        or plan["project"] != PROJECT
        or plan["region"] != REGION
        or plan["zone"] != ZONE
        or plan["machine_type"] != MACHINE_TYPE
        or plan["instance_count"] != 1
        or plan["current_profile_changed"] is not False
    ):
        raise HuRlC4GcpLifecycleError("execution plan fixed target changed")
    if _RUN_NAME_RE.fullmatch(plan["run_name"]) is None or _INSTANCE_RE.fullmatch(plan["instance_name"]) is None:
        raise HuRlC4GcpLifecycleError("execution plan name is invalid")
    if _BUCKET_RE.fullmatch(plan["bucket"]) is None or _SERVICE_ACCOUNT_RE.fullmatch(plan["worker_service_account"]) is None:
        raise HuRlC4GcpLifecycleError("execution plan cloud identity is invalid")
    if not _is_sha256(plan["plan_identity_sha256"]) or plan["plan_label"] != plan["plan_identity_sha256"][:32]:
        raise HuRlC4GcpLifecycleError("execution plan identity is invalid")
    if plan["owner_label"] != f"rlc-{plan['plan_identity_sha256'][:28]}":
        raise HuRlC4GcpLifecycleError("execution plan owner label changed")
    objects = plan["objects"]
    expected_prefix = f"hu-rl-c4/{plan['run_name']}/{plan['plan_identity_sha256']}/"
    if objects != {
        "prefix": expected_prefix,
        "package": expected_prefix + "input/package.tar.gz",
        "attestation": expected_prefix + "control/machine_attestation.json",
        "result": expected_prefix + "result/hu_rl_c4_formal_benchmark_result.json",
        "worker_status": expected_prefix + "result/worker_status.json",
    }:
        raise HuRlC4GcpLifecycleError("execution plan object namespace changed")
    package = plan["package"]
    if (
        set(package) != {
            "archive_filename", "archive_sha256", "archive_size_bytes", "archive_root", "file_count",
            "files_sha256", "manifest_sha256", "native_wheel_sha256",
        }
        or not _is_sha256(package["archive_sha256"])
        or not _is_sha256(package["files_sha256"])
        or not _is_sha256(package["manifest_sha256"])
        or not _is_sha256(package["native_wheel_sha256"])
        or type(package["archive_size_bytes"]) is not int
        or package["archive_size_bytes"] <= 0
        or type(package["file_count"]) is not int
        or package["file_count"] <= 0
        or plan["formal_manifest_sha256"] != package["manifest_sha256"]
    ):
        raise HuRlC4GcpLifecycleError("execution plan package binding is invalid")
    image = plan["image"]
    if set(image) != {"source_image_self_link", "image_id", "guest_os_features"}:
        raise HuRlC4GcpLifecycleError("execution plan image fields changed")
    _validate_image(image["source_image_self_link"], image["image_id"], image["guest_os_features"])
    if plan["network"] != {
        "network": f"global/networks/{NETWORK_NAME}",
        "subnetwork": f"regions/{REGION}/subnetworks/{SUBNETWORK_NAME}",
        "nic_type": NIC_TYPE,
        "external_ip": False,
    }:
        raise HuRlC4GcpLifecycleError("execution plan network changed")
    if plan["vm"] != {
        "provisioning_model": PROVISIONING_MODEL,
        "preemptible": True,
        "automatic_restart": False,
        "on_host_maintenance": "TERMINATE",
        "instance_termination_action": "DELETE",
        "max_run_duration_seconds": MAX_RUN_DURATION_SECONDS,
        "worker_watchdog_seconds": WORKER_WATCHDOG_SECONDS,
        "deletion_protection": False,
        "boot_disk_type": BOOT_DISK_TYPE,
        "boot_disk_interface": BOOT_DISK_INTERFACE,
        "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
        "boot_disk_auto_delete": True,
    }:
        raise HuRlC4GcpLifecycleError("execution plan VM contract changed")
    if not _is_sha256(plan["execution_plan_sha256"]) or plan["execution_plan_sha256"] != _self_digest(plan, "execution_plan_sha256"):
        raise HuRlC4GcpLifecycleError("execution plan digest mismatch")


def build_operation_authorization(
    *, plan: Mapping[str, Any], operation: str, raw_nonce: str, now_unix_seconds: int, ttl_seconds: int = 7_200
) -> dict[str, Any]:
    validate_execution_plan(plan)
    if operation not in {"launch", "cleanup"}:
        raise HuRlC4GcpLifecycleError("authorization operation is invalid")
    _require_uuid4(raw_nonce, "operation nonce")
    if type(now_unix_seconds) is not int or type(ttl_seconds) is not int or ttl_seconds < 300 or ttl_seconds > 7_200:
        raise HuRlC4GcpLifecycleError("authorization lifetime is invalid")
    auth: dict[str, Any] = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_exact_one_vm_operation_authorized",
        "operation": operation,
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "instance_name": plan["instance_name"],
        "approved_exact_instance_count": 1,
        "approved_machine_type": MACHINE_TYPE,
        "approved_provisioning_model": PROVISIONING_MODEL,
        "nonce_sha256": hashlib.sha256(raw_nonce.encode("ascii")).hexdigest(),
        "issued_unix_seconds": now_unix_seconds,
        "expires_unix_seconds": now_unix_seconds + ttl_seconds,
        "wildcard_mutation_authorized": False,
        "current_profile_changed": False,
        "authorization_sha256": None,
    }
    auth["authorization_sha256"] = _self_digest(auth, "authorization_sha256")
    validate_operation_authorization(
        auth, plan=plan, operation=operation, raw_nonce=raw_nonce, now_unix_seconds=now_unix_seconds
    )
    return auth


def validate_operation_authorization(
    auth: Mapping[str, Any], *, plan: Mapping[str, Any], operation: str, raw_nonce: str, now_unix_seconds: int
) -> None:
    validate_execution_plan(plan)
    _require_uuid4(raw_nonce, "operation nonce")
    required = {
        "schema", "status", "operation", "execution_plan_sha256", "instance_name",
        "approved_exact_instance_count", "approved_machine_type", "approved_provisioning_model",
        "nonce_sha256", "issued_unix_seconds", "expires_unix_seconds",
        "wildcard_mutation_authorized", "current_profile_changed", "authorization_sha256",
    }
    if set(auth) != required:
        raise HuRlC4GcpLifecycleError("authorization fields changed")
    if (
        auth["schema"] != AUTHORIZATION_SCHEMA
        or auth["status"] != "explicit_exact_one_vm_operation_authorized"
        or auth["operation"] != operation
        or auth["execution_plan_sha256"] != plan["execution_plan_sha256"]
        or auth["instance_name"] != plan["instance_name"]
        or auth["approved_exact_instance_count"] != 1
        or auth["approved_machine_type"] != MACHINE_TYPE
        or auth["approved_provisioning_model"] != PROVISIONING_MODEL
        or auth["wildcard_mutation_authorized"] is not False
        or auth["current_profile_changed"] is not False
        or auth["nonce_sha256"] != hashlib.sha256(raw_nonce.encode("ascii")).hexdigest()
        or type(auth["issued_unix_seconds"]) is not int
        or type(auth["expires_unix_seconds"]) is not int
        or not (auth["issued_unix_seconds"] <= now_unix_seconds < auth["expires_unix_seconds"])
    ):
        raise HuRlC4GcpLifecycleError("authorization is invalid, expired, or belongs to another plan")
    if not _is_sha256(auth["authorization_sha256"]) or auth["authorization_sha256"] != _self_digest(auth, "authorization_sha256"):
        raise HuRlC4GcpLifecycleError("authorization digest mismatch")


def worker_startup_script() -> str:
    """Return the controller-owned bootstrap; the frozen benchmark script is unchanged."""

    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/var/lib/ofc-hu-rl-c4
mkdir -p \"$ROOT\"
( sleep {WORKER_WATCHDOG_SECONDS}; systemctl poweroff --force ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT

cat >\"$ROOT/gcs.py\" <<'PY'
import json, pathlib, sys, urllib.error, urllib.parse, urllib.request
META='http://metadata.google.internal/computeMetadata/v1/'
HDR={{'Metadata-Flavor':'Google'}}
def meta(path):
    return urllib.request.urlopen(urllib.request.Request(META+path,headers=HDR),timeout=30).read().decode()
def token():
    return json.loads(meta('instance/service-accounts/default/token'))['access_token']
def url(bucket,obj,media=False,generation=None):
    base='https://storage.googleapis.com/storage/v1/b/'+urllib.parse.quote(bucket,safe='')+'/o/'+urllib.parse.quote(obj,safe='')
    query=[]
    if media: query.append('alt=media')
    if generation: query.append('generation='+urllib.parse.quote(generation,safe=''))
    return base+('?'+'&'.join(query) if query else '')
def get(bucket,obj,path,generation=None):
    req=urllib.request.Request(url(bucket,obj,True,generation),headers={{'Authorization':'Bearer '+token()}})
    try: data=urllib.request.urlopen(req,timeout=300).read()
    except urllib.error.HTTPError as e:
        if e.code==404: return 44
        raise
    pathlib.Path(path).write_bytes(data); return 0
def put(bucket,obj,path):
    data=pathlib.Path(path).read_bytes()
    u='https://storage.googleapis.com/upload/storage/v1/b/'+urllib.parse.quote(bucket,safe='')+'/o?uploadType=media&ifGenerationMatch=0&name='+urllib.parse.quote(obj,safe='')
    req=urllib.request.Request(u,data=data,method='POST',headers={{'Authorization':'Bearer '+token(),'Content-Type':'application/json'}})
    urllib.request.urlopen(req,timeout=300).read(); return 0
if sys.argv[1]=='meta': print(meta('instance/attributes/'+sys.argv[2])); raise SystemExit(0)
if sys.argv[1]=='get': raise SystemExit(get(*sys.argv[2:]))
if sys.argv[1]=='put': raise SystemExit(put(*sys.argv[2:]))
raise SystemExit(64)
PY

meta() {{ python3 \"$ROOT/gcs.py\" meta \"$1\"; }}
BUCKET=$(meta c4-bucket)
PACKAGE_OBJECT=$(meta c4-package-object)
PACKAGE_GENERATION=$(meta c4-package-generation)
PACKAGE_SHA256=$(meta c4-package-sha256)
PACKAGE_ROOT_NAME=$(meta c4-package-root)
ATTESTATION_OBJECT=$(meta c4-attestation-object)
RESULT_OBJECT=$(meta c4-result-object)
STATUS_OBJECT=$(meta c4-status-object)
PLAN_SHA256=$(meta c4-plan-sha256)

python3 \"$ROOT/gcs.py\" get \"$BUCKET\" \"$PACKAGE_OBJECT\" \"$ROOT/package.tar.gz\" \"$PACKAGE_GENERATION\"
printf '%s  %s\n' \"$PACKAGE_SHA256\" \"$ROOT/package.tar.gz\" | sha256sum -c -
python3 - \"$ROOT/package.tar.gz\" \"$ROOT\" \"$PACKAGE_ROOT_NAME\" <<'PY'
import pathlib, sys, tarfile
archive, destination, expected_root = sys.argv[1:]
with tarfile.open(archive, 'r:gz') as tf:
    for member in tf.getmembers():
        p=pathlib.PurePosixPath(member.name)
        if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=expected_root:
            raise SystemExit('unsafe package archive path')
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit('unsafe package archive entry')
    tf.extractall(destination)
PY

cat >/etc/apt/sources.list <<'EOF'
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/{SNAPSHOT_TIMESTAMP} bookworm main
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/{SNAPSHOT_TIMESTAMP} bookworm-security main
EOF
apt-get -o Acquire::Check-Valid-Until=false update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv ca-certificates
python3 -m venv \"$ROOT/venv\"
\"$ROOT/venv/bin/python\" -m pip install --no-index --find-links \"$ROOT/$PACKAGE_ROOT_NAME/wheelhouse\" numpy ofc-hu-rl-engine-native

ATT_PATH=\"$ROOT/machine_attestation.json\"
for _ in $(seq 1 360); do
  if python3 \"$ROOT/gcs.py\" get \"$BUCKET\" \"$ATTESTATION_OBJECT\" \"$ATT_PATH\"; then break; fi
  sleep 5
done
[[ -s \"$ATT_PATH\" ]] || {{ echo 'attestation was not published' >&2; exit 1; }}

PACKAGE_ROOT=\"$ROOT/$PACKAGE_ROOT_NAME\"
RESULT_PATH=\"$ROOT/hu_rl_c4_formal_benchmark_result.json\"
export PATH=\"$ROOT/venv/bin:$PATH\"
export OFC_HU_RL_C4_PACKAGE_ROOT=\"$PACKAGE_ROOT\"
export OFC_HU_RL_C4_MACHINE_ATTESTATION=\"$ATT_PATH\"
export OFC_HU_RL_C4_RESULT=\"$RESULT_PATH\"
set +e
bash \"$PACKAGE_ROOT/scripts/startup_hu_rl_c4_formal_benchmark.sh\"
BENCHMARK_RC=$?
set -e
RESULT_UPLOADED=false
if [[ -s \"$RESULT_PATH\" ]]; then
  python3 \"$ROOT/gcs.py\" put \"$BUCKET\" \"$RESULT_OBJECT\" \"$RESULT_PATH\"
  RESULT_UPLOADED=true
fi
python3 - \"$ROOT/worker_status.json\" \"$PLAN_SHA256\" \"$BENCHMARK_RC\" \"$RESULT_UPLOADED\" <<'PY'
import hashlib,json,pathlib,sys
path,plan,rc,uploaded=sys.argv[1:]
value={{'schema':'{WORKER_STATUS_SCHEMA}','execution_plan_sha256':plan,'benchmark_exit_code':int(rc),'result_uploaded':uploaded=='true','worker_status_sha256':None}}
copy=dict(value); copy['worker_status_sha256']=None
value['worker_status_sha256']=hashlib.sha256(json.dumps(copy,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()
pathlib.Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='ascii')
PY
python3 \"$ROOT/gcs.py\" put \"$BUCKET\" \"$STATUS_OBJECT\" \"$ROOT/worker_status.json\"
sync
systemctl poweroff
exit \"$BENCHMARK_RC\"
"""


def build_instance_spec(plan: Mapping[str, Any], *, package_generation: str) -> dict[str, Any]:
    validate_execution_plan(plan)
    if not isinstance(package_generation, str) or re.fullmatch(r"[1-9][0-9]*", package_generation) is None:
        raise HuRlC4GcpLifecycleError("package object generation is invalid")
    metadata = {
        "c4-bucket": plan["bucket"],
        "c4-package-object": plan["objects"]["package"],
        "c4-package-generation": package_generation,
        "c4-package-sha256": plan["package"]["archive_sha256"],
        "c4-package-root": plan["package"]["archive_root"],
        "c4-attestation-object": plan["objects"]["attestation"],
        "c4-result-object": plan["objects"]["result"],
        "c4-status-object": plan["objects"]["worker_status"],
        "c4-plan-sha256": plan["execution_plan_sha256"],
        "startup-script": worker_startup_script(),
    }
    return {
        "name": plan["instance_name"],
        "machineType": f"zones/{ZONE}/machineTypes/{MACHINE_TYPE}",
        "labels": {"ofc-owner": plan["owner_label"], "ofc-plan": plan["plan_label"], "ofc-role": "rlc-formal"},
        "deletionProtection": False,
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "interface": BOOT_DISK_INTERFACE,
                "initializeParams": {
                    "sourceImage": plan["image"]["source_image_self_link"],
                    "diskType": f"zones/{ZONE}/diskTypes/{BOOT_DISK_TYPE}",
                    "diskSizeGb": str(BOOT_DISK_SIZE_GB),
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": plan["network"]["network"],
                "subnetwork": plan["network"]["subnetwork"],
                "nicType": NIC_TYPE,
            }
        ],
        "scheduling": {
            "provisioningModel": PROVISIONING_MODEL,
            "preemptible": True,
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": str(MAX_RUN_DURATION_SECONDS)},
        },
        "serviceAccounts": [{"email": plan["worker_service_account"], "scopes": [STORAGE_SCOPE]}],
        "metadata": {"items": [{"key": key, "value": value} for key, value in sorted(metadata.items())]},
    }


def _metadata_dict(instance: Mapping[str, Any]) -> dict[str, str]:
    items = instance.get("metadata", {}).get("items", [])
    if not isinstance(items, list):
        raise HuRlC4GcpLifecycleError("instance metadata is invalid")
    result: dict[str, str] = {}
    for row in items:
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str) or not isinstance(row.get("value"), str):
            raise HuRlC4GcpLifecycleError("instance metadata row is invalid")
        if row["key"] in result:
            raise HuRlC4GcpLifecycleError("instance metadata contains duplicate keys")
        result[row["key"]] = row["value"]
    return result


def validate_owned_instance(
    instance: Mapping[str, Any], *, plan: Mapping[str, Any], package_generation: str, require_running: bool
) -> None:
    validate_execution_plan(plan)
    expected = build_instance_spec(plan, package_generation=package_generation)
    measured_machine_type = instance.get("machineType")
    expected_machine_suffix = f"/projects/{PROJECT}/zones/{ZONE}/machineTypes/{MACHINE_TYPE}"
    if (
        instance.get("name") != plan["instance_name"]
        or not isinstance(measured_machine_type, str)
        or not measured_machine_type.endswith(expected_machine_suffix)
        or instance.get("labels") != expected["labels"]
        or instance.get("deletionProtection") is not False
        or (require_running and instance.get("status") != "RUNNING")
    ):
        raise HuRlC4GcpLifecycleError("instance identity, ownership, or state changed")
    scheduling = instance.get("scheduling", {})
    if any(scheduling.get(k) != v for k, v in expected["scheduling"].items()):
        raise HuRlC4GcpLifecycleError("instance Spot/TTL scheduling changed")
    interfaces = instance.get("networkInterfaces")
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        raise HuRlC4GcpLifecycleError("instance network interface count changed")
    nic = interfaces[0]
    if nic.get("nicType") != NIC_TYPE or nic.get("accessConfigs") not in (None, []):
        raise HuRlC4GcpLifecycleError("instance NIC or external IP contract changed")
    disks = instance.get("disks")
    if not isinstance(disks, list) or len(disks) != 1:
        raise HuRlC4GcpLifecycleError("instance disk count changed")
    disk = disks[0]
    if disk.get("boot") is not True or disk.get("autoDelete") is not True or disk.get("interface") != BOOT_DISK_INTERFACE:
        raise HuRlC4GcpLifecycleError("instance boot disk attachment changed")
    service_accounts = instance.get("serviceAccounts")
    if not isinstance(service_accounts, list) or len(service_accounts) != 1:
        raise HuRlC4GcpLifecycleError("instance service account count changed")
    if service_accounts[0].get("email") != plan["worker_service_account"] or service_accounts[0].get("scopes") != [STORAGE_SCOPE]:
        raise HuRlC4GcpLifecycleError("instance worker identity changed")
    measured_metadata = _metadata_dict(instance)
    expected_metadata = {row["key"]: row["value"] for row in expected["metadata"]["items"]}
    if measured_metadata != expected_metadata:
        raise HuRlC4GcpLifecycleError("instance startup metadata changed")


def _wait_operation(
    transport: LifecycleTransport,
    *,
    initial: Mapping[str, Any],
    operation_type: str,
    target_name: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    name = initial.get("name")
    if not isinstance(name, str) or _INSTANCE_RE.fullmatch(name) is None:
        raise HuRlC4GcpLifecycleError("GCE operation name is invalid")
    current = dict(initial)
    for attempt in range(OPERATION_POLL_ATTEMPTS):
        if (
            current.get("name") != name
            or current.get("operationType") != operation_type
            or current.get("targetLink", "").rsplit("/", 1)[-1] != target_name
        ):
            raise HuRlC4GcpLifecycleError("GCE operation identity changed")
        if current.get("status") == "DONE":
            if current.get("error"):
                raise HuRlC4GcpLifecycleError(f"GCE {operation_type} operation failed")
            return current
        if attempt + 1 < OPERATION_POLL_ATTEMPTS:
            sleep(OPERATION_POLL_INTERVAL_SECONDS)
            current = dict(transport.get_zone_operation(operation_name=name))
    raise HuRlC4GcpLifecycleError(f"GCE {operation_type} operation timed out")


def _wait_instance_running(
    transport: LifecycleTransport,
    *,
    plan: Mapping[str, Any],
    package_generation: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    for attempt in range(INSTANCE_POLL_ATTEMPTS):
        instance = transport.get_instance(instance_name=plan["instance_name"])
        if instance is not None:
            if instance.get("status") == "RUNNING":
                validate_owned_instance(instance, plan=plan, package_generation=package_generation, require_running=True)
                return dict(instance)
            if instance.get("status") in {"TERMINATED", "SUSPENDED"}:
                raise HuRlC4GcpLifecycleError("owned instance terminated before attestation")
        if attempt + 1 < INSTANCE_POLL_ATTEMPTS:
            sleep(INSTANCE_POLL_INTERVAL_SECONDS)
    raise HuRlC4GcpLifecycleError("owned instance did not reach RUNNING within bound")


def build_external_observation(
    *,
    plan: Mapping[str, Any],
    instance: Mapping[str, Any],
    machine_type: Mapping[str, Any],
    disk: Mapping[str, Any],
    package_generation: str,
    controller_principal_sha256: str,
    observed_at_utc: str,
) -> dict[str, Any]:
    """Derive the frozen observation schema only from external GCE GET payloads."""

    validate_owned_instance(instance, plan=plan, package_generation=package_generation, require_running=True)
    if not _is_sha256(controller_principal_sha256):
        raise HuRlC4GcpLifecycleError("controller principal digest is invalid")
    if type(machine_type.get("guestCpus")) is not int or machine_type.get("guestCpus") != TARGET_VCPU_COUNT:
        raise HuRlC4GcpLifecycleError("machine type vCPU count changed")
    attached = instance["disks"][0]
    disk_source = attached.get("source")
    if not isinstance(disk_source, str) or disk_source.rsplit("/", 1)[-1] != disk.get("name"):
        raise HuRlC4GcpLifecycleError("attached disk identity changed")
    expected_disk_type_suffix = f"/projects/{PROJECT}/zones/{ZONE}/diskTypes/{BOOT_DISK_TYPE}"
    if not isinstance(disk.get("type"), str) or not disk["type"].endswith(expected_disk_type_suffix):
        raise HuRlC4GcpLifecycleError("boot disk type changed")
    if disk.get("sourceImage") != plan["image"]["source_image_self_link"] or str(disk.get("sourceImageId")) != plan["image"]["image_id"]:
        raise HuRlC4GcpLifecycleError("boot disk source image changed")
    machine_uri = instance.get("machineType")
    if not isinstance(machine_uri, str) or not machine_uri.endswith(f"/projects/{PROJECT}/zones/{ZONE}/machineTypes/{MACHINE_TYPE}"):
        raise HuRlC4GcpLifecycleError("instance machine type URI changed")
    observation: dict[str, Any] = {
        "schema": HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
        "source": EXTERNAL_GCE_OBSERVATION_SOURCE,
        "provider": "gcp",
        "project_id": PROJECT,
        "zone": ZONE,
        "instance_name": instance["name"],
        "instance_id": str(instance.get("id")),
        "status": instance["status"],
        "machine_type": MACHINE_TYPE,
        "machine_type_uri": machine_uri,
        "vcpu_count": machine_type["guestCpus"],
        "cpu_platform": instance.get("cpuPlatform"),
        "provisioning_model": instance["scheduling"]["provisioningModel"],
        "preemptible": instance["scheduling"]["preemptible"],
        "deletion_protection": instance["deletionProtection"],
        "nic_type": instance["networkInterfaces"][0]["nicType"],
        "boot_disk_type": BOOT_DISK_TYPE,
        "image_self_link": disk["sourceImage"],
        "image_id": str(disk["sourceImageId"]),
        "observed_at_utc": observed_at_utc,
        "controller_principal_sha256": controller_principal_sha256,
        "observation_sha256": None,
    }
    observation["observation_sha256"] = _self_digest(observation, "observation_sha256")
    validate_c4_external_gce_observation(observation)
    return observation


def _ensure_object_bytes(
    transport: LifecycleTransport,
    *,
    bucket: str,
    object_name: str,
    payload: bytes,
    content_type: str,
) -> dict[str, Any]:
    metadata = transport.get_object_metadata(bucket=bucket, object_name=object_name)
    if metadata is None:
        try:
            created = dict(
                transport.put_object_new(
                    bucket=bucket, object_name=object_name, payload=payload, content_type=content_type
                )
            )
        except ResponseLostError:
            metadata = transport.get_object_metadata(bucket=bucket, object_name=object_name)
            if metadata is None:
                raise HuRlC4GcpLifecycleError("object create response was lost and object is absent")
        else:
            metadata = created
    remote = transport.get_object_bytes(
        bucket=bucket, object_name=object_name, generation=str(metadata.get("generation"))
    )
    if remote != payload:
        raise HuRlC4GcpLifecycleError("existing or created object bytes differ")
    if metadata.get("name") != object_name or re.fullmatch(r"[1-9][0-9]*", str(metadata.get("generation"))) is None:
        raise HuRlC4GcpLifecycleError("GCS object identity is invalid")
    return dict(metadata)


def _load_manifest_from_package(package_root: Path) -> dict[str, Any]:
    manifest = load_canonical_c4_json(
        package_root / "hu_rl_c4_formal_benchmark_manifest.json", context="C4 package manifest"
    )
    validate_c4_formal_benchmark_manifest(manifest)
    return manifest


def launch_or_recover(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    raw_nonce: str,
    package_root: Path,
    archive_path: Path,
    controller_principal_sha256: str,
    transport: LifecycleTransport,
    now_unix_seconds: int,
    observed_at_utc: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Idempotently launch/recover exactly one VM and publish its attestation."""

    validate_operation_authorization(
        authorization, plan=plan, operation="launch", raw_nonce=raw_nonce, now_unix_seconds=now_unix_seconds
    )
    package = verify_frozen_package(
        package_root=package_root,
        archive_path=archive_path,
        expected_archive_sha256=plan["package"]["archive_sha256"],
    )
    if package != plan["package"]:
        raise HuRlC4GcpLifecycleError("local package no longer matches execution plan")
    manifest = _load_manifest_from_package(package_root)
    if manifest["manifest_sha256"] != plan["formal_manifest_sha256"]:
        raise HuRlC4GcpLifecycleError("formal manifest no longer matches execution plan")

    for object_key in ("attestation", "result", "worker_status"):
        if transport.get_object_metadata(bucket=plan["bucket"], object_name=plan["objects"][object_key]) is not None:
            if object_key != "attestation":
                raise HuRlC4GcpLifecycleError("result namespace was already used")
    archive_bytes = archive_path.read_bytes()
    package_object = _ensure_object_bytes(
        transport,
        bucket=plan["bucket"],
        object_name=plan["objects"]["package"],
        payload=archive_bytes,
        content_type="application/gzip",
    )
    package_generation = str(package_object["generation"])
    expected_spec = build_instance_spec(plan, package_generation=package_generation)

    instance = transport.get_instance(instance_name=plan["instance_name"])
    create_operation: dict[str, Any] | None = None
    reconciled_after_response_loss = False
    if instance is None:
        try:
            initial = transport.create_instance(instance_spec=expected_spec, request_id=raw_nonce)
        except ResponseLostError:
            reconciled_after_response_loss = True
            instance = transport.get_instance(instance_name=plan["instance_name"])
            if instance is None:
                raise HuRlC4GcpLifecycleError("instance create response was lost and exact instance is absent")
        else:
            create_operation = _wait_operation(
                transport,
                initial=initial,
                operation_type="insert",
                target_name=plan["instance_name"],
                sleep=sleep,
            )
    else:
        reconciled_after_response_loss = True
    if instance is not None:
        validate_owned_instance(
            instance,
            plan=plan,
            package_generation=package_generation,
            require_running=instance.get("status") == "RUNNING",
        )
    running = _wait_instance_running(
        transport, plan=plan, package_generation=package_generation, sleep=sleep
    )
    disk_source = running["disks"][0].get("source")
    if not isinstance(disk_source, str):
        raise HuRlC4GcpLifecycleError("running instance has no boot disk source")
    machine = transport.get_machine_type(machine_type=MACHINE_TYPE)
    disk = transport.get_disk(disk_name=disk_source.rsplit("/", 1)[-1])
    observation = build_external_observation(
        plan=plan,
        instance=running,
        machine_type=machine,
        disk=disk,
        package_generation=package_generation,
        controller_principal_sha256=controller_principal_sha256,
        observed_at_utc=observed_at_utc or _utc_now(),
    )
    attestation = build_c4_machine_attestation(
        manifest_sha256=manifest["manifest_sha256"], external_observation=observation
    )
    attestation_payload = canonical_bytes(attestation)
    attestation_object = _ensure_object_bytes(
        transport,
        bucket=plan["bucket"],
        object_name=plan["objects"]["attestation"],
        payload=attestation_payload,
        content_type="application/json",
    )
    receipt: dict[str, Any] = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "status": "exact_one_owned_c4_spot_running_and_attested",
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "authorization_sha256": authorization["authorization_sha256"],
        "instance_name": plan["instance_name"],
        "instance_id": str(running["id"]),
        "instance_count": 1,
        "owner_label": plan["owner_label"],
        "plan_label": plan["plan_label"],
        "package_object": {
            "name": plan["objects"]["package"],
            "generation": package_generation,
            "sha256": plan["package"]["archive_sha256"],
        },
        "attestation_object": {
            "name": plan["objects"]["attestation"],
            "generation": str(attestation_object["generation"]),
            "sha256": hashlib.sha256(attestation_payload).hexdigest(),
        },
        "external_observation": observation,
        "machine_attestation": attestation,
        "create_operation": create_operation,
        "reconciled_after_response_loss": reconciled_after_response_loss,
        "external_ip_present": False,
        "current_profile_changed": False,
        "launch_receipt_sha256": None,
    }
    receipt["launch_receipt_sha256"] = _self_digest(receipt, "launch_receipt_sha256")
    validate_launch_receipt(receipt, plan=plan)
    return receipt


def validate_launch_receipt(receipt: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    validate_execution_plan(plan)
    required = {
        "schema", "status", "execution_plan_sha256", "authorization_sha256", "instance_name",
        "instance_id", "instance_count", "owner_label", "plan_label", "package_object",
        "attestation_object", "external_observation", "machine_attestation", "create_operation",
        "reconciled_after_response_loss", "external_ip_present", "current_profile_changed",
        "launch_receipt_sha256",
    }
    if set(receipt) != required:
        raise HuRlC4GcpLifecycleError("launch receipt fields changed")
    if (
        receipt["schema"] != LAUNCH_RECEIPT_SCHEMA
        or receipt["status"] != "exact_one_owned_c4_spot_running_and_attested"
        or receipt["execution_plan_sha256"] != plan["execution_plan_sha256"]
        or receipt["instance_name"] != plan["instance_name"]
        or receipt["instance_count"] != 1
        or receipt["owner_label"] != plan["owner_label"]
        or receipt["plan_label"] != plan["plan_label"]
        or receipt["external_ip_present"] is not False
        or receipt["current_profile_changed"] is not False
        or type(receipt["reconciled_after_response_loss"]) is not bool
        or re.fullmatch(r"[1-9][0-9]*", receipt["instance_id"]) is None
    ):
        raise HuRlC4GcpLifecycleError("launch receipt identity changed")
    if receipt["package_object"] != {
        "name": plan["objects"]["package"],
        "generation": receipt["package_object"].get("generation"),
        "sha256": plan["package"]["archive_sha256"],
    } or re.fullmatch(r"[1-9][0-9]*", str(receipt["package_object"].get("generation"))) is None:
        raise HuRlC4GcpLifecycleError("launch package object binding changed")
    observation = receipt["external_observation"]
    validate_c4_external_gce_observation(observation)
    if observation["instance_id"] != receipt["instance_id"] or observation["instance_name"] != receipt["instance_name"]:
        raise HuRlC4GcpLifecycleError("launch observation instance changed")
    expected_attestation = build_c4_machine_attestation(
        manifest_sha256=plan["formal_manifest_sha256"], external_observation=observation
    )
    if receipt["machine_attestation"] != expected_attestation:
        raise HuRlC4GcpLifecycleError("launch attestation changed")
    attestation_payload = canonical_bytes(expected_attestation)
    if receipt["attestation_object"] != {
        "name": plan["objects"]["attestation"],
        "generation": receipt["attestation_object"].get("generation"),
        "sha256": hashlib.sha256(attestation_payload).hexdigest(),
    } or re.fullmatch(r"[1-9][0-9]*", str(receipt["attestation_object"].get("generation"))) is None:
        raise HuRlC4GcpLifecycleError("launch attestation object binding changed")
    if not _is_sha256(receipt["authorization_sha256"]):
        raise HuRlC4GcpLifecycleError("launch authorization digest is invalid")
    if not _is_sha256(receipt["launch_receipt_sha256"]) or receipt["launch_receipt_sha256"] != _self_digest(receipt, "launch_receipt_sha256"):
        raise HuRlC4GcpLifecycleError("launch receipt digest mismatch")


def _decode_canonical_document(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_CANONICAL_INPUT_BYTES:
        raise HuRlC4GcpLifecycleError(f"{label} byte size is invalid")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlC4GcpLifecycleError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise HuRlC4GcpLifecycleError(f"{label} is not canonical byte-locked JSON")
    return value


def _validate_worker_status(value: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema", "execution_plan_sha256", "benchmark_exit_code", "result_uploaded", "worker_status_sha256"
    }:
        raise HuRlC4GcpLifecycleError("worker status fields changed")
    if (
        value["schema"] != WORKER_STATUS_SCHEMA
        or value["execution_plan_sha256"] != plan["execution_plan_sha256"]
        or type(value["benchmark_exit_code"]) is not int
        or value["benchmark_exit_code"] not in {0, 1, 2}
        or value["result_uploaded"] is not True
        or not _is_sha256(value["worker_status_sha256"])
        or value["worker_status_sha256"] != _self_digest(value, "worker_status_sha256")
    ):
        raise HuRlC4GcpLifecycleError("worker status is invalid")


def collect_result(
    *,
    plan: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    package_root: Path,
    transport: LifecycleTransport,
    sleep: Callable[[float], None] = time.sleep,
    result_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Poll, retrieve, validate, and bind a formal pass or formal no-go result."""

    validate_launch_receipt(launch_receipt, plan=plan)
    manifest = _load_manifest_from_package(package_root)
    if manifest["manifest_sha256"] != plan["formal_manifest_sha256"]:
        raise HuRlC4GcpLifecycleError("collection manifest differs from plan")
    result_raw: bytes | None = None
    status_raw: bytes | None = None
    for attempt in range(RESULT_POLL_ATTEMPTS):
        result_raw = transport.get_object_bytes(bucket=plan["bucket"], object_name=plan["objects"]["result"])
        status_raw = transport.get_object_bytes(bucket=plan["bucket"], object_name=plan["objects"]["worker_status"])
        if result_raw is not None and status_raw is not None:
            break
        instance = transport.get_instance(instance_name=plan["instance_name"])
        if instance is None and result_raw is None:
            raise HuRlC4GcpLifecycleError("owned instance disappeared without a result")
        if attempt + 1 < RESULT_POLL_ATTEMPTS:
            sleep(RESULT_POLL_INTERVAL_SECONDS)
    if result_raw is None or status_raw is None:
        raise HuRlC4GcpLifecycleError("formal result collection timed out")
    result = _decode_canonical_document(result_raw, "formal C4 result")
    status = _decode_canonical_document(status_raw, "C4 worker status")
    _validate_worker_status(status, plan=plan)
    if result_validator is None:
        validate_c4_formal_benchmark_result(result, manifest=manifest)
    else:
        result_validator(result, manifest)
    if result.get("manifest_sha256") != plan["formal_manifest_sha256"]:
        raise HuRlC4GcpLifecycleError("formal result belongs to another manifest")
    if result.get("machine_attestation") != launch_receipt["machine_attestation"]:
        raise HuRlC4GcpLifecycleError("formal result attestation differs from launch")
    overall_pass = result.get("gates", {}).get("overall_pass")
    if type(overall_pass) is not bool:
        raise HuRlC4GcpLifecycleError("formal result overall gate is invalid")
    expected_exit = 0 if overall_pass else 2
    if status["benchmark_exit_code"] != expected_exit:
        raise HuRlC4GcpLifecycleError("worker exit code disagrees with formal gate")
    receipt: dict[str, Any] = {
        "schema": COLLECTION_RECEIPT_SCHEMA,
        "status": "formal_result_validated_pass" if overall_pass else "formal_result_validated_no_go",
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "launch_receipt_sha256": launch_receipt["launch_receipt_sha256"],
        "instance_name": plan["instance_name"],
        "instance_id": launch_receipt["instance_id"],
        "result_object": plan["objects"]["result"],
        "result_bytes_sha256": hashlib.sha256(result_raw).hexdigest(),
        "formal_result_sha256": result.get("result_sha256"),
        "formal_gate_pass": overall_pass,
        "worker_status_sha256": status["worker_status_sha256"],
        "scientific_result_validated": True,
        "current_profile_changed": False,
        "collection_receipt_sha256": None,
    }
    receipt["collection_receipt_sha256"] = _self_digest(receipt, "collection_receipt_sha256")
    validate_collection_receipt(receipt, plan=plan, launch_receipt=launch_receipt)
    return receipt, result_raw


def validate_collection_receipt(
    receipt: Mapping[str, Any], *, plan: Mapping[str, Any], launch_receipt: Mapping[str, Any]
) -> None:
    validate_launch_receipt(launch_receipt, plan=plan)
    required = {
        "schema", "status", "execution_plan_sha256", "launch_receipt_sha256", "instance_name",
        "instance_id", "result_object", "result_bytes_sha256", "formal_result_sha256",
        "formal_gate_pass", "worker_status_sha256", "scientific_result_validated",
        "current_profile_changed", "collection_receipt_sha256",
    }
    if set(receipt) != required:
        raise HuRlC4GcpLifecycleError("collection receipt fields changed")
    expected_status = "formal_result_validated_pass" if receipt.get("formal_gate_pass") is True else "formal_result_validated_no_go"
    if (
        receipt["schema"] != COLLECTION_RECEIPT_SCHEMA
        or receipt["status"] != expected_status
        or receipt["execution_plan_sha256"] != plan["execution_plan_sha256"]
        or receipt["launch_receipt_sha256"] != launch_receipt["launch_receipt_sha256"]
        or receipt["instance_name"] != plan["instance_name"]
        or receipt["instance_id"] != launch_receipt["instance_id"]
        or receipt["result_object"] != plan["objects"]["result"]
        or type(receipt["formal_gate_pass"]) is not bool
        or receipt["scientific_result_validated"] is not True
        or receipt["current_profile_changed"] is not False
        or not all(_is_sha256(receipt[field]) for field in ("result_bytes_sha256", "formal_result_sha256", "worker_status_sha256"))
    ):
        raise HuRlC4GcpLifecycleError("collection receipt is invalid")
    if not _is_sha256(receipt["collection_receipt_sha256"]) or receipt["collection_receipt_sha256"] != _self_digest(receipt, "collection_receipt_sha256"):
        raise HuRlC4GcpLifecycleError("collection receipt digest mismatch")


def cleanup_owned_instance(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    raw_nonce: str,
    launch_receipt: Mapping[str, Any] | None,
    collection_receipt: Mapping[str, Any] | None,
    explicit_abort: bool,
    package_generation: str,
    transport: LifecycleTransport,
    now_unix_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Delete exactly one measured owned instance; never delete by prefix/wildcard."""

    validate_operation_authorization(
        authorization, plan=plan, operation="cleanup", raw_nonce=raw_nonce, now_unix_seconds=now_unix_seconds
    )
    if launch_receipt is not None:
        validate_launch_receipt(launch_receipt, plan=plan)
    if collection_receipt is not None:
        if launch_receipt is None:
            raise HuRlC4GcpLifecycleError("collection cleanup requires a launch receipt")
        validate_collection_receipt(collection_receipt, plan=plan, launch_receipt=launch_receipt)
    if collection_receipt is None and explicit_abort is not True:
        raise HuRlC4GcpLifecycleError("cleanup requires validated collection or explicit abort")
    instance = transport.get_instance(instance_name=plan["instance_name"])
    delete_operation: dict[str, Any] | None = None
    instance_was_present = instance is not None
    if instance is not None:
        validate_owned_instance(
            instance, plan=plan, package_generation=package_generation, require_running=False
        )
        if launch_receipt is not None and str(instance.get("id")) != launch_receipt["instance_id"]:
            raise HuRlC4GcpLifecycleError("cleanup instance id differs from launch")
        try:
            initial = transport.delete_instance(instance_name=plan["instance_name"], request_id=raw_nonce)
        except ResponseLostError:
            remaining = transport.get_instance(instance_name=plan["instance_name"])
            if remaining is not None:
                raise HuRlC4GcpLifecycleError("delete response was lost and owned instance remains")
        else:
            delete_operation = _wait_operation(
                transport,
                initial=initial,
                operation_type="delete",
                target_name=plan["instance_name"],
                sleep=sleep,
            )
    if transport.get_instance(instance_name=plan["instance_name"]) is not None:
        raise HuRlC4GcpLifecycleError("owned instance is still present after cleanup")
    receipt: dict[str, Any] = {
        "schema": CLEANUP_RECEIPT_SCHEMA,
        "status": "exact_owned_instance_cleanup_complete",
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "authorization_sha256": authorization["authorization_sha256"],
        "instance_name": plan["instance_name"],
        "instance_count_targeted": 1,
        "instance_was_present": instance_was_present,
        "instance_confirmed_absent": True,
        "collection_receipt_sha256": collection_receipt["collection_receipt_sha256"] if collection_receipt else None,
        "operator_abort": collection_receipt is None,
        "scientific_result_claimed": collection_receipt is not None,
        "delete_operation": delete_operation,
        "wildcard_delete_used": False,
        "gcs_evidence_deleted": False,
        "current_profile_changed": False,
        "cleanup_receipt_sha256": None,
    }
    receipt["cleanup_receipt_sha256"] = _self_digest(receipt, "cleanup_receipt_sha256")
    return receipt


class GcpRestTransport:
    """Narrow REST adapter.  The access token is accepted only in memory."""

    def __init__(
        self,
        *,
        access_token: str,
        project: str = PROJECT,
        zone: str = ZONE,
        request: Callable[[str, str, Mapping[str, str], bytes | None, int], HttpResponse] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        if not access_token or any(ch.isspace() for ch in access_token):
            raise HuRlC4GcpLifecycleError("GCP access token is invalid")
        if project != PROJECT or zone not in APPROVED_ZONES:
            raise HuRlC4GcpLifecycleError("GCP REST target changed")
        self._token = access_token
        self.project = project
        self.zone = zone
        self._request = request or _stdlib_request
        self._sleep = sleep
        # Optional: supplies a replacement access token when one expires
        # mid-operation. Absent, a 401 stays fatal exactly as before.
        self._token_provider = token_provider
        self._token_refreshed = False

    def _call(
        self, method: str, url: str, *, payload: bytes | None = None, timeout: int = HTTP_TIMEOUT_SECONDS,
        content_type: str = "application/json", allow_404: bool = False
    ) -> HttpResponse | None:
        """Issue one REST call, retrying only where a retry cannot change state.

        Reads are idempotent, so a lost or throttled response is retried here.
        Mutating calls are not: ``create_instance`` and ``delete_instance``
        carry a GCE ``requestId`` and convert a lost response into
        ``ResponseLostError`` so the caller can reconcile deliberately, which is
        a decision this layer must not make for them.

        A 401 is retried once after refreshing the token, because a long poll
        can outlive an access token; without that the whole operation fails
        partway through with no way to tell it from a real credential problem.
        """

        idempotent = method == "GET"
        attempt = 0
        while True:
            headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
            if payload is not None:
                headers["Content-Type"] = content_type
            try:
                response = self._request(method, url, headers, payload, timeout)
            except (TimeoutError, OSError) as exc:
                if not idempotent or attempt >= _READ_RETRY_ATTEMPTS:
                    raise
                attempt += 1
                self._sleep(_READ_RETRY_BACKOFF_SECONDS * attempt)
                continue
            if response.status == 401 and self._refresh_token():
                continue
            if (
                idempotent
                and response.status in _RETRYABLE_STATUSES
                and attempt < _READ_RETRY_ATTEMPTS
            ):
                attempt += 1
                self._sleep(_READ_RETRY_BACKOFF_SECONDS * attempt)
                continue
            if allow_404 and response.status == 404:
                return None
            if response.status < 200 or response.status >= 300:
                raise HuRlC4GcpLifecycleError(f"GCP REST {method} failed with HTTP {response.status}")
            return response

    def _refresh_token(self) -> bool:
        """Replace an expired access token, at most once per call."""

        if self._token_provider is None or self._token_refreshed:
            return False
        token = self._token_provider()
        if not isinstance(token, str) or not token:
            return False
        self._token = token
        self._token_refreshed = True
        return True

    @staticmethod
    def _json(response: HttpResponse | None, label: str) -> dict[str, Any]:
        if response is None:
            raise HuRlC4GcpLifecycleError(f"{label} response is absent")
        try:
            value = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HuRlC4GcpLifecycleError(f"{label} response is not JSON") from exc
        if not isinstance(value, dict):
            raise HuRlC4GcpLifecycleError(f"{label} response is not an object")
        return value

    def _gcs_metadata_url(self, bucket: str, object_name: str) -> str:
        return (
            "https://storage.googleapis.com/storage/v1/b/"
            + urllib.parse.quote(bucket, safe="")
            + "/o/"
            + urllib.parse.quote(object_name, safe="")
        )

    def get_object_metadata(self, *, bucket: str, object_name: str) -> Mapping[str, Any] | None:
        response = self._call("GET", self._gcs_metadata_url(bucket, object_name), allow_404=True)
        return None if response is None else self._json(response, "GCS object metadata")

    def get_object_bytes(self, *, bucket: str, object_name: str, generation: str | None = None) -> bytes | None:
        query = {"alt": "media"}
        if generation is not None:
            query["generation"] = generation
        response = self._call(
            "GET", self._gcs_metadata_url(bucket, object_name) + "?" + urllib.parse.urlencode(query), allow_404=True,
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
        return None if response is None else response.body

    def put_object_new(
        self, *, bucket: str, object_name: str, payload: bytes, content_type: str
    ) -> Mapping[str, Any]:
        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            + urllib.parse.quote(bucket, safe="")
            + "/o?"
            + urllib.parse.urlencode({"uploadType": "media", "ifGenerationMatch": "0", "name": object_name})
        )
        try:
            response = self._call("POST", url, payload=payload, timeout=UPLOAD_TIMEOUT_SECONDS, content_type=content_type)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ResponseLostError("GCS object create response was lost") from exc
        return self._json(response, "GCS object create")

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None:
        url = f"https://compute.googleapis.com/compute/v1/projects/{self.project}/zones/{self.zone}/instances/{instance_name}"
        response = self._call("GET", url, allow_404=True)
        return None if response is None else self._json(response, "GCE instance get")

    def create_instance(self, *, instance_spec: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        url = f"https://compute.googleapis.com/compute/v1/projects/{self.project}/zones/{self.zone}/instances?requestId={urllib.parse.quote(request_id)}"
        try:
            response = self._call("POST", url, payload=canonical_bytes(instance_spec).rstrip(b"\n"), timeout=HTTP_TIMEOUT_SECONDS)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ResponseLostError("GCE instance insert response was lost") from exc
        return self._json(response, "GCE instance insert")

    def get_zone_operation(self, *, operation_name: str) -> Mapping[str, Any]:
        url = f"https://compute.googleapis.com/compute/v1/projects/{self.project}/zones/{self.zone}/operations/{operation_name}"
        return self._json(self._call("GET", url), "GCE zone operation")

    def get_machine_type(self, *, machine_type: str) -> Mapping[str, Any]:
        url = f"https://compute.googleapis.com/compute/v1/projects/{self.project}/zones/{self.zone}/machineTypes/{machine_type}"
        return self._json(self._call("GET", url), "GCE machine type")

    def get_disk(self, *, disk_name: str) -> Mapping[str, Any]:
        url = f"https://compute.googleapis.com/compute/v1/projects/{self.project}/zones/{self.zone}/disks/{disk_name}"
        return self._json(self._call("GET", url), "GCE disk")

    def delete_instance(self, *, instance_name: str, request_id: str) -> Mapping[str, Any]:
        url = f"https://compute.googleapis.com/compute/v1/projects/{self.project}/zones/{self.zone}/instances/{instance_name}?requestId={urllib.parse.quote(request_id)}"
        try:
            response = self._call("DELETE", url, timeout=HTTP_TIMEOUT_SECONDS)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ResponseLostError("GCE instance delete response was lost") from exc
        return self._json(response, "GCE instance delete")


def _stdlib_request(
    method: str, url: str, headers: Mapping[str, str], payload: bytes | None, timeout: int
) -> HttpResponse:
    request = urllib.request.Request(url, data=payload, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
            return HttpResponse(response.status, response.read(), dict(response.headers.items()))
    except urllib.error.HTTPError as error:
        return HttpResponse(error.code, error.read(), dict(error.headers.items()))


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))


def load_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    return _decode_canonical_document(raw, label)


__all__ = [
    "AUTHORIZATION_SCHEMA", "BOOT_DISK_INTERFACE", "BOOT_DISK_SIZE_GB", "BOOT_DISK_TYPE",
    "CLEANUP_RECEIPT_SCHEMA", "COLLECTION_RECEIPT_SCHEMA", "GcpRestTransport",
    "HuRlC4GcpLifecycleError", "INSTANCE_COUNT", "LAUNCH_RECEIPT_SCHEMA", "MAX_RUN_DURATION_SECONDS",
    "NIC_TYPE", "PLAN_SCHEMA", "PROVISIONING_MODEL", "ResponseLostError", "WORKER_STATUS_SCHEMA",
    "build_execution_plan", "build_external_observation", "build_instance_spec",
    "build_operation_authorization", "canonical_bytes", "canonical_sha256", "cleanup_owned_instance",
    "collect_result", "launch_or_recover", "load_json", "sha256_file", "validate_collection_receipt",
    "validate_execution_plan", "validate_launch_receipt", "validate_operation_authorization",
    "validate_owned_instance", "verify_frozen_package", "worker_startup_script", "write_json_once",
]
