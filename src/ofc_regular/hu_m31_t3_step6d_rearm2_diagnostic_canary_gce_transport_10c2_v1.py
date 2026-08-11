"""Versioned direct-GCE transport contract for the Step 6d diagnostic worker.

The default artifact produced by this module is deliberately **not** launch
ready.  It describes and validates the exact metadata, content-addressed outer
package, REST operations, retry layout, and lifecycle which a later authorized
preflight must exercise.  Building or locally validating the contract performs
no metadata-server, object-store, Compute API, VM, authorization, or claim
operation.

The runtime HTTP surface is injectable.  A real caller must first provide
controller records accepted by an external trust verifier; malformed or
untrusted records fail before the first HTTP request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence


CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_gce_transport_10c2_v1"
)
OUTER_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_direct_outer_manifest_v1"
)
OUTER_IDENTITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_direct_outer_identity_v1"
)
DIRECT_STAGE_IDENTITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_direct_stage_identity_v1"
)
METADATA_BINDING_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_gce_metadata_binding_v1"
)
REST_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_gce_rest_plan_v1"
)
LOCAL_PREFLIGHT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_local_preflight_v1"
)
AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_controller_authorization_v1"
)
CLAIM_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_worker_claim_v1"
RSA_PUBLIC_KEY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_controller_rsa_public_key_v1"
)
RSA_SIGNATURE_ALGORITHM = "RSASSA-PKCS1-v1_5-SHA256"
LIFECYCLE_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_worker_lifecycle_plan_v1"
)

DIRECT_NAMESPACE = "hu-m31-r2diag-direct-v1"
MACHINE_TYPE = "c4-standard-16"
ZONE = "asia-northeast1-b"
PROJECT = "ofc-solver-485418"
BUCKET = "pokerhu-ofc-solver-485418-training"
WORKER_SERVICE_ACCOUNT = (
    "ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com"
)
REQUIRED_WORKER_OAUTH_SCOPE = (
    "https://www.googleapis.com/auth/cloud-platform"
)
IMAGE_PROJECT = "debian-cloud"
IMAGE_NAME = "debian-12-bookworm-v20260609"
IMAGE_ID = "1449487925682397051"
IMAGE_SELF_LINK = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "debian-12-bookworm-v20260609"
)
MAX_ATTEMPTS = 2
MAX_FAILURE_SHUTDOWN_SECONDS = 120
AUTHORIZED_WORKER_SHUTDOWN_ATTEMPTED_EXIT_CODE = 86
STAGE1_ID = "stage1_lifecycle_one_candidate_vm"
STAGE2_ID = "stage2_candidate_reference_pair"
STAGE1_RUN_NAME = "regular-hu-m31-r2diag-s1-20260718-001"
STAGE2_RUN_NAME = "regular-hu-m31-r2diag-s2-20260718-001"
STAGE1_JOB_IDS = ("candidate-shard-00",)
STAGE2_JOB_IDS = ("candidate-shard-01", "reference-shard-01")
STAGE1_HAND_INDICES = (0, 10, 13, 43, 49, 62, 66, 81, 82, 99)
STAGE2_HAND_INDICES = (5, 6, 35, 39, 47, 53, 76, 83, 87, 89)
BOUNDED_SHUTDOWN_MARKER_PATH = Path(
    "/run/ofc-m31-step11-shutdown-requested"
)
EXPECTED_NUMPY_REQUIREMENT = "numpy==2.2.6"
EXPECTED_NUMPY_WHEEL_FILENAME = (
    "numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)
EXPECTED_NUMPY_WHEEL_SHA256 = (
    "ba10f8411898fc418a521833e014a77d3ca01c15b0c6cdcce6a0d2897e6dbbdf"
)
EXPECTED_NUMPY_WHEEL_BYTES = 16_821_570
INNER_MANIFEST_NAME = "manifest.json"
INNER_SOURCE_NAME = "hu_m31_t3_step6d_rearm2_diagnostic_worker_v1.zip"
INNER_STARTUP_NAME = (
    "startup_hu_m31_t3_step6d_rearm2_diagnostic_canary_v1.sh"
)
INNER_VERIFIER_NAME = (
    "verify_hu_m31_t3_step6d_rearm2_diagnostic_canary_v1.py"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_RELATIVE = (
    "scripts/"
    "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_v1.sh"
)
TRANSPORT_RELATIVE = (
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1.py"
)
DIRECT_SUPPORT_RELATIVES = (
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_plan.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_HEX = re.compile(r"^[0-9a-f]+$")
_GCE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_FORBIDDEN_FIELD_PARTS = (
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
    "q_milli",
    "action_key",
)
_SUBPROCESS_SHUTDOWN_ATTEMPTED = False


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rsa_public_key_record(
    *, modulus_hex: str, exponent: int = 65_537
) -> dict[str, Any]:
    """Build the canonical, public-only controller trust anchor.

    The worker implements the small RFC 8017 verification operation directly
    with Python's standard library.  There is deliberately no private-key
    parser, signer, shared secret, or third-party runtime dependency here.
    """

    if (
        not isinstance(modulus_hex, str)
        or _LOWER_HEX.fullmatch(modulus_hex) is None
        or len(modulus_hex) % 2
    ):
        raise ValueError("RSA modulus must be even-length lowercase hexadecimal")
    modulus = int(modulus_hex, 16)
    if not 2_048 <= modulus.bit_length() <= 4_096 or modulus % 2 == 0:
        raise ValueError("RSA modulus must be an odd 2048-4096 bit integer")
    checked_exponent = _strict_int(
        exponent, "RSA public exponent", minimum=3, maximum=(1 << 31) - 1
    )
    if checked_exponent % 2 == 0:
        raise ValueError("RSA public exponent must be odd")
    identity = {
        "algorithm": RSA_SIGNATURE_ALGORITHM,
        "exponent": checked_exponent,
        "modulus_hex": modulus_hex,
    }
    return {
        "schema": RSA_PUBLIC_KEY_SCHEMA,
        **identity,
        "key_id": canonical_sha256(identity),
        "private_key_present": False,
    }


def validate_rsa_public_key_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(value)
    _exact(
        record,
        {
            "schema",
            "algorithm",
            "exponent",
            "modulus_hex",
            "key_id",
            "private_key_present",
        },
        "controller RSA public key",
    )
    expected = build_rsa_public_key_record(
        modulus_hex=record["modulus_hex"], exponent=record["exponent"]
    )
    if record != expected:
        raise ValueError("controller RSA public-key identity changed")
    return record


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
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


def _strict_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} must be {expected!r}")
    return value


def _safe_relative(value: Any) -> str:
    posix = PurePosixPath(value) if isinstance(value, str) else None
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or posix is None
        or posix.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
        or posix.as_posix() != value
    ):
        raise ValueError("outer package path escaped its root")
    return value


def _reject_hidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if any(part in key for part in _FORBIDDEN_FIELD_PARTS):
                raise ValueError(f"forbidden transport field at {path}.{raw_key}")
            _reject_hidden(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _reject_unapproved_network_locations(
    value: Any, path: str = "$"
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_unapproved_network_locations(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unapproved_network_locations(child, f"{path}[{index}]")
    elif isinstance(value, str) and "://" in value:
        allowed = (
            value.startswith(f"gs://{BUCKET}/"),
            value.startswith(
                "http://metadata.google.internal/computeMetadata/v1"
            ),
            value.startswith("https://storage.googleapis.com/"),
            value.startswith(
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{PROJECT}/"
            ),
            value == IMAGE_SELF_LINK,
        )
        if not any(allowed):
            raise ValueError(f"network location escaped fixed contract at {path}")


def _lazy_modules() -> tuple[Any, Any, Any]:
    from . import (
        hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as package,
    )
    from . import (
        hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
    )
    from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan

    return package, adapter, plan


def _runtime_adapter_module() -> Any:
    """Load only the envelope adapter needed after worker extraction."""

    from . import (
        hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
        as adapter,
    )

    return adapter


def deterministic_instance_name(
    *,
    stage_id: str,
    job_id: str,
    attempt_index: int,
    preview_stage_identity_sha256: str,
) -> str:
    _strict_int(
        attempt_index, "attempt index", minimum=0, maximum=MAX_ATTEMPTS - 1
    )
    stage_tag = (
        "s1"
        if stage_id == "stage1_lifecycle_one_candidate_vm"
        else "s2"
        if stage_id == "stage2_candidate_reference_pair"
        else None
    )
    if stage_tag is None:
        raise ValueError("instance name escaped the two diagnostic stages")
    slug = job_id.replace("-shard-", "-")
    name = (
        f"r2d-10c2-{stage_tag}-{slug}-a{attempt_index}-"
        f"{_sha(preview_stage_identity_sha256, 'preview stage identity')[:8]}"
    )
    if _GCE_NAME.fullmatch(name) is None or len(name) > 63:
        raise ValueError("deterministic instance name is not GCE-safe")
    return name


def build_offline_wheel_record(path: str | Path) -> dict[str, Any]:
    wheel = Path(path).resolve()
    if (
        wheel.name != EXPECTED_NUMPY_WHEEL_FILENAME
        or not wheel.is_file()
        or wheel.is_symlink()
        or wheel.stat().st_size != EXPECTED_NUMPY_WHEEL_BYTES
        or sha256_file(wheel) != EXPECTED_NUMPY_WHEEL_SHA256
    ):
        raise ValueError("offline numpy wheel identity changed")
    return {
        "path": f"wheels/{EXPECTED_NUMPY_WHEEL_FILENAME}",
        "uri_suffix": f"wheels/{EXPECTED_NUMPY_WHEEL_FILENAME}",
        "sha256": EXPECTED_NUMPY_WHEEL_SHA256,
        "bytes": EXPECTED_NUMPY_WHEEL_BYTES,
        "mode": "0644",
        "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
    }


def _file_record(
    path: Path,
    *,
    relative: str,
    kind: str,
    mode: str,
) -> dict[str, Any]:
    relative = _safe_relative(relative)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"outer package source is not a regular file: {relative}")
    return {
        "path": relative,
        "uri_suffix": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "mode": mode,
        "kind": kind,
    }


def build_outer_package_manifest(
    *,
    package_dir: str | Path,
    offline_wheel_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package_module, _adapter, _plan = _lazy_modules()
    package = Path(package_dir).resolve()
    inner = package_module.validate_package(package)
    inner_manifest_sha = sha256_file(package / package_module.MANIFEST_NAME)
    records: list[dict[str, Any]] = []
    for relative in (
        package_module.MANIFEST_NAME,
        package_module.READY_NAME,
        package_module.SOURCE_NAME,
        package_module.STARTUP_NAME,
        package_module.VERIFIER_NAME,
        "jobs/candidate-shard-00.json",
        "jobs/candidate-shard-01.json",
        "jobs/reference-shard-01.json",
    ):
        records.append(
            _file_record(
                package / relative,
                relative=f"inner/{relative}",
                kind="inner_worker_package",
                mode="0755" if relative == package_module.STARTUP_NAME else "0644",
            )
        )
    records.extend(
        (
            _file_record(
                _REPO_ROOT / BOOTSTRAP_RELATIVE,
                relative=BOOTSTRAP_RELATIVE,
                kind="direct_bootstrap_entrypoint",
                mode="0755",
            ),
            _file_record(
                _REPO_ROOT / TRANSPORT_RELATIVE,
                relative=TRANSPORT_RELATIVE,
                kind="direct_transport_runtime",
                mode="0644",
            ),
        )
    )
    for relative in DIRECT_SUPPORT_RELATIVES:
        records.append(
            _file_record(
                _REPO_ROOT / relative,
                relative=relative,
                kind="direct_transport_support",
                mode="0644",
            )
        )
    wheel_complete = offline_wheel_record is not None
    if offline_wheel_record is not None:
        wheel = dict(offline_wheel_record)
        _exact(
            wheel,
            {"path", "uri_suffix", "sha256", "bytes", "mode", "kind"},
            "offline wheel record",
        )
        if (
            wheel["path"] != f"wheels/{EXPECTED_NUMPY_WHEEL_FILENAME}"
            or wheel["uri_suffix"] != wheel["path"]
            or wheel["sha256"] != EXPECTED_NUMPY_WHEEL_SHA256
            or wheel["bytes"] != EXPECTED_NUMPY_WHEEL_BYTES
            or wheel["mode"] != "0644"
            or wheel["kind"]
            != "offline_numpy_cp311_manylinux_x86_64_wheel"
        ):
            raise ValueError("offline wheel record is not the pinned artifact")
        records.append(wheel)
    if len({row["path"] for row in records}) != len(records):
        raise ValueError("outer package paths collided")
    records = sorted(records, key=lambda row: row["path"])
    records_sha = canonical_sha256(records)
    identity = {
        "schema": OUTER_IDENTITY_SCHEMA,
        "inner_package_manifest_sha256": inner_manifest_sha,
        "inner_source_sha256": inner["source_sha256"],
        "objects_sha256": records_sha,
        "object_count": len(records),
        "offline_dependency_bundle_complete": wheel_complete,
    }
    outer_id = canonical_sha256(identity)
    manifest = {
        "schema": OUTER_MANIFEST_SCHEMA,
        "status": (
            "outer_payload_complete_external_preflight_required"
            if wheel_complete
            else "outer_payload_missing_offline_wheel_not_executable"
        ),
        "inner_package_manifest_sha256": inner_manifest_sha,
        "objects": records,
        "objects_sha256": records_sha,
        "outer_package_identity_sha256": outer_id,
        "complete_for_direct_v1": wheel_complete,
    }
    _reject_hidden(manifest)
    return manifest


def validate_outer_package_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    _exact(
        manifest,
        {
            "schema",
            "status",
            "inner_package_manifest_sha256",
            "objects",
            "objects_sha256",
            "outer_package_identity_sha256",
            "complete_for_direct_v1",
        },
        "outer package manifest",
    )
    records = manifest["objects"]
    expected_without_wheel = 10 + len(DIRECT_SUPPORT_RELATIVES)
    if not isinstance(records, list) or len(records) not in (
        expected_without_wheel,
        expected_without_wheel + 1,
    ):
        raise ValueError("outer package object count changed")
    checked: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("outer package object must be a mapping")
        row = dict(raw)
        _exact(
            row,
            {"path", "uri_suffix", "sha256", "bytes", "mode", "kind"},
            "outer package object",
        )
        if (
            _safe_relative(row["path"]) != row["path"]
            or row["uri_suffix"] != row["path"]
            or _sha(row["sha256"], "outer object sha256") != row["sha256"]
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or row["mode"] not in ("0644", "0755")
            or not isinstance(row["kind"], str)
            or not row["kind"]
        ):
            raise ValueError("outer package object identity changed")
        checked.append(row)
    if checked != sorted(checked, key=lambda row: row["path"]) or len(
        {row["path"] for row in checked}
    ) != len(checked):
        raise ValueError("outer package object order or uniqueness changed")
    wheel = [
        row
        for row in checked
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    ]
    complete = len(wheel) == 1
    if wheel and (
        wheel[0]["path"] != f"wheels/{EXPECTED_NUMPY_WHEEL_FILENAME}"
        or wheel[0]["sha256"] != EXPECTED_NUMPY_WHEEL_SHA256
        or wheel[0]["bytes"] != EXPECTED_NUMPY_WHEEL_BYTES
    ):
        raise ValueError("outer package wheel identity changed")
    if (
        manifest["schema"] != OUTER_MANIFEST_SCHEMA
        or _sha(
            manifest["inner_package_manifest_sha256"],
            "inner package manifest sha256",
        )
        != manifest["inner_package_manifest_sha256"]
        or manifest["objects_sha256"] != canonical_sha256(checked)
        or type(manifest["complete_for_direct_v1"]) is not bool
        or manifest["complete_for_direct_v1"] is not complete
        or manifest["status"]
        != (
            "outer_payload_complete_external_preflight_required"
            if complete
            else "outer_payload_missing_offline_wheel_not_executable"
        )
    ):
        raise ValueError("outer package manifest binding changed")
    identity = {
        "schema": OUTER_IDENTITY_SCHEMA,
        "inner_package_manifest_sha256": manifest[
            "inner_package_manifest_sha256"
        ],
        "inner_source_sha256": next(
            row["sha256"]
            for row in checked
            if row["path"]
            == "inner/hu_m31_t3_step6d_rearm2_diagnostic_worker_v1.zip"
        ),
        "objects_sha256": manifest["objects_sha256"],
        "object_count": len(checked),
        "offline_dependency_bundle_complete": complete,
    }
    if manifest["outer_package_identity_sha256"] != canonical_sha256(identity):
        raise ValueError("outer package identity changed")
    _reject_hidden(manifest)
    return manifest


def build_outer_package_inventory(
    outer_manifest: Mapping[str, Any],
    *,
    bucket: str = BUCKET,
) -> dict[str, Any]:
    manifest = validate_outer_package_manifest(outer_manifest)
    outer_id = manifest["outer_package_identity_sha256"]
    base = f"gs://{bucket}/{DIRECT_NAMESPACE}"
    package_prefix = f"{base}/packages/{outer_id}"
    manifest_raw = canonical_bytes(manifest)
    records = [
        {
            "path": "outer-manifest.json",
            "uri": f"{package_prefix}/outer-manifest.json",
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "bytes": len(manifest_raw),
            "mode": "0644",
            "kind": "outer_package_manifest",
        }
    ]
    records.extend(
        {
            "path": row["path"],
            "uri": f"{package_prefix}/{row['uri_suffix']}",
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "mode": row["mode"],
            "kind": row["kind"],
        }
        for row in manifest["objects"]
    )
    return {
        "base_prefix": base,
        "package_prefix": package_prefix,
        "outer_package_identity_sha256": outer_id,
        "outer_package_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "records": records,
        "records_sha256": canonical_sha256(records),
        "source_provisioning_allowed_states": [
            "exact_empty",
            "exact_contiguous_or_sparse_subset",
            "exact_complete",
        ],
        "unknown_or_mismatched_object_is_fatal": True,
        "missing_objects_use_generation_match_zero": True,
        "readback_required_after_create": True,
        "complete_before_launch_required": True,
    }


def build_direct_stage_identity(
    *,
    preview: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    outer = validate_outer_package_manifest(outer_manifest)
    outer_manifest_sha = canonical_sha256(outer)
    inner_stage_sha = _sha(
        preview["stage_identity_sha256"], "preview stage identity sha256"
    )
    job_layout = [
        {
            "job_id": job["job_id"],
            "source_role": job["source_role"],
            "runner_job_manifest_sha256": job["runner_job_manifest"]["sha256"],
            "work_hand_indices": job["work_hand_indices"],
        }
        for job in preview["jobs"]
    ]
    attempt_layout = [
        {
            "job_id": job["job_id"],
            "attempt_index": attempt,
            "instance_name": deterministic_instance_name(
                stage_id=preview["stage_id"],
                job_id=job["job_id"],
                attempt_index=attempt,
                preview_stage_identity_sha256=inner_stage_sha,
            ),
        }
        for attempt in range(MAX_ATTEMPTS)
        for job in preview["jobs"]
    ]
    inputs = {
        "inner_preview_stage_identity_sha256": inner_stage_sha,
        "outer_package_manifest_sha256": outer_manifest_sha,
        "outer_package_identity_sha256": outer[
            "outer_package_identity_sha256"
        ],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_layout": job_layout,
        "attempt_layout": attempt_layout,
        "max_attempts": MAX_ATTEMPTS,
        "machine_type": MACHINE_TYPE,
        "zone": ZONE,
        "image_project": IMAGE_PROJECT,
        "image_name": IMAGE_NAME,
        "image_id": IMAGE_ID,
        "image_self_link": IMAGE_SELF_LINK,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "api_allowlist": [
            "gce_metadata_identity_read",
            "gce_metadata_service_account_token_read",
            "gcs_json_objects_get_generation_pinned",
            "gcs_json_objects_insert_if_generation_match_zero",
            "gcs_json_objects_readback",
            "compute_v1_instances_delete_self",
        ],
        "gcloud_dependency": False,
        "retry_invariant_result_identity": True,
    }
    identity = {
        "schema": DIRECT_STAGE_IDENTITY_SCHEMA,
        "outer_package_identity_sha256": outer[
            "outer_package_identity_sha256"
        ],
        "preview_stage_identity_sha256": inner_stage_sha,
        "inputs": inputs,
        "inputs_sha256": canonical_sha256(inputs),
    }
    return {
        **identity,
        "direct_stage_identity_sha256": canonical_sha256(identity),
    }


def build_direct_v1_remote_layout(
    *,
    preview: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    package_inventory = build_outer_package_inventory(outer_manifest)
    identity = build_direct_stage_identity(
        preview=preview, outer_manifest=outer_manifest
    )
    stage_sha = identity["direct_stage_identity_sha256"]
    stage_prefix = (
        f"{package_inventory['base_prefix']}/stages/{preview['run_name']}/"
        f"{stage_sha}"
    )
    result_prefix = f"{stage_prefix}/results"
    attempt_control = (
        f"{stage_prefix}/control/attempt-{preview['attempt_index']}"
    )
    jobs = []
    for job in preview["jobs"]:
        job_prefix = f"{result_prefix}/jobs/{job['job_id']}"
        tree_prefix = f"{job_prefix}/tree"
        jobs.append(
            {
                "job_id": job["job_id"],
                "result_prefix": job_prefix,
                "tree_prefix": tree_prefix,
                "upload_uris": [
                    f"{job_prefix}/uploads/hand_{index:03d}.json"
                    for index in job["work_hand_indices"]
                ],
                "heartbeat_uris": [
                    f"{result_prefix}/progress/jobs/{job['job_id']}/"
                    f"heartbeats/{sequence:06d}.json"
                    for sequence in range(1, len(job["work_hand_indices"]) + 1)
                ],
                "done_uri": f"{job_prefix}/DONE.envelope.json",
                "tree_object_uris": [
                    f"{tree_prefix}/{row['path']}"
                    for row in job["tree_object_manifest"]
                ],
            }
        )
    return {
        "base_prefix": package_inventory["base_prefix"],
        "package_prefix": package_inventory["package_prefix"],
        "package_inventory": package_inventory,
        "stage_prefix": stage_prefix,
        "result_prefix": result_prefix,
        "attempt_control_prefix": attempt_control,
        "receive_uri": f"{result_prefix}/received/{preview['stage_id']}.json",
        "direct_stage_identity": identity,
        "direct_stage_identity_sha256": stage_sha,
        "jobs": jobs,
        "package_and_stage_prefix_disjoint": True,
        "stage_prefix_must_be_exactly_empty_before_attempt0": True,
        "attempt_control_prefix_must_be_exactly_empty": True,
        "result_identity_retry_invariant": True,
    }


def _gs_parts(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("object URI must use gs://")
    bucket, separator, name = uri[5:].partition("/")
    if (
        not separator
        or not bucket
        or not name
        or any(ord(character) < 32 for character in uri)
    ):
        raise ValueError("object URI is incomplete")
    return bucket, name


def _gcs_media_get_url(uri: str, *, generation: str = "{generation}") -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?alt=media&generation={generation}"
    )


def _gcs_metadata_get_url(uri: str) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?fields=bucket,name,generation,size,"
        "md5Hash,crc32c,metadata"
    )


def _gcs_upload_url(uri: str) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o?uploadType=media&"
        f"ifGenerationMatch=0&name={urllib.parse.quote(name, safe='')}"
    )


def _rebase_tree_rows(
    *,
    adapter_job: Mapping[str, Any],
    direct_job: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for source, uri in zip(
        adapter_job["tree_object_manifest"],
        direct_job["tree_object_uris"],
        strict=True,
    ):
        rows.append(
            {
                "path": source["path"],
                "uri": uri,
                "identity_source": source["identity_source"],
                "sha256": source["sha256"],
                "bytes": source["bytes"],
            }
        )
    return rows


def build_lifecycle_plan(
    *,
    preview: Mapping[str, Any],
    direct_layout: Mapping[str, Any],
    job_id: str,
) -> dict[str, Any]:
    adapter_job = next(
        (row for row in preview["jobs"] if row["job_id"] == job_id), None
    )
    direct_job = next(
        (row for row in direct_layout["jobs"] if row["job_id"] == job_id), None
    )
    if adapter_job is None or direct_job is None:
        raise ValueError("lifecycle escaped exact diagnostic job")
    tree_rows = _rebase_tree_rows(
        adapter_job=adapter_job, direct_job=direct_job
    )
    operations: list[dict[str, Any]] = []
    for row in tree_rows[:2]:
        operations.append(
            {
                "operation": "conditional_create_tree_control",
                "uri": row["uri"],
                "path": row["path"],
                "if_generation_match": 0,
                "readback_required": True,
            }
        )
    for sequence, hand_index in enumerate(
        adapter_job["work_hand_indices"], 1
    ):
        root_row = tree_rows[2 + (sequence - 1) * 2]
        hand_row = tree_rows[3 + (sequence - 1) * 2]
        for operation, row in (
            ("conditional_create_tree_root", root_row),
            ("conditional_create_tree_hand", hand_row),
        ):
            operations.append(
                {
                    "operation": operation,
                    "sequence": sequence,
                    "hand_index": hand_index,
                    "uri": row["uri"],
                    "path": row["path"],
                    "if_generation_match": 0,
                    "readback_required": True,
                }
            )
        operations.extend(
            (
                {
                    "operation": "conditional_create_artifact_envelope",
                    "sequence": sequence,
                    "hand_index": hand_index,
                    "uri": direct_job["upload_uris"][sequence - 1],
                    "if_generation_match": 0,
                    "readback_required": True,
                },
                {
                    "operation": "conditional_create_heartbeat",
                    "sequence": sequence,
                    "hand_index": hand_index,
                    "uri": direct_job["heartbeat_uris"][sequence - 1],
                    "if_generation_match": 0,
                    "readback_required": True,
                    "requires_artifact_envelope_readback": True,
                },
            )
        )
    operations.extend(
        (
            {
                "operation": "conditional_create_runner_done_tree",
                "uri": tree_rows[-1]["uri"],
                "path": "DONE.json",
                "if_generation_match": 0,
                "readback_required": True,
                "requires_runner_validation": True,
                "requires_all_heartbeats_readback": True,
            },
            {
                "operation": "conditional_create_done_envelope",
                "uri": direct_job["done_uri"],
                "if_generation_match": 0,
                "readback_required": True,
                "requires_runner_done_tree_readback": True,
            },
            {
                "operation": "request_compute_delete_self",
                "requires_done_envelope_readback": True,
                "instance_delete_only": True,
            },
        )
    )
    return {
        "schema": LIFECYCLE_PLAN_SCHEMA,
        "job_id": job_id,
        "ordered_success_operations": operations,
        "ordered_success_operations_sha256": canonical_sha256(operations),
        "resume_policy": {
            "existing_identical_object": "readback_and_continue",
            "missing_object": "conditional_create_generation_match_zero",
            "different_or_unknown_object": "fatal",
            "result_uris_retry_invariant": True,
        },
        "failure_policy": {
            "publish_done": False,
            "request_self_delete": False,
            "preserve_completed_objects": True,
            "bounded_local_shutdown_required": True,
            "shutdown_deadline_seconds": MAX_FAILURE_SHUTDOWN_SECONDS,
        },
        "post_done_delete_failure_policy": {
            "preserve_done_and_completed_objects": True,
            "bounded_local_shutdown_required": True,
            "shutdown_deadline_seconds": MAX_FAILURE_SHUTDOWN_SECONDS,
            "done_semantics_remain_successful": True,
        },
        "transport_fixture_only": False,
        "scientific_payload_present": False,
    }


def _build_rest_plan(
    *,
    layout: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    instance_name: str,
) -> dict[str, Any]:
    package_records = layout["package_inventory"]["records"]
    return {
        "schema": REST_PLAN_SCHEMA,
        "metadata": {
            "root": "http://metadata.google.internal/computeMetadata/v1",
            "required_header": {"Metadata-Flavor": "Google"},
            "identity_paths": [
                "/project/project-id",
                "/instance/id",
                "/instance/name",
                "/instance/zone",
                "/instance/service-accounts/default/email",
                "/instance/service-accounts/default/scopes",
            ],
            "token_path": "/instance/service-accounts/default/token",
        },
        "package_downloads": [
            {
                "uri": row["uri"],
                "generation_pinned_url_template": _gcs_media_get_url(row["uri"]),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "fresh_exclusive_path": row["path"],
            }
            for row in package_records
        ],
        "object_metadata_get_url_templates": [
            {
                "uri": row["uri"],
                "url": _gcs_metadata_get_url(row["uri"]),
            }
            for row in package_records
        ],
        "conditional_uploads": [
            {
                "operation": operation["operation"],
                "uri": operation["uri"],
                "upload_url": _gcs_upload_url(operation["uri"]),
                "readback_metadata_url": _gcs_metadata_get_url(operation["uri"]),
                "readback_media_url_template": _gcs_media_get_url(
                    operation["uri"]
                ),
                "if_generation_match": 0,
            }
            for operation in lifecycle["ordered_success_operations"]
            if "uri" in operation
        ],
        "self_delete": {
            "method": "DELETE",
            "url": (
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{PROJECT}/zones/{ZONE}/instances/{instance_name}"
            ),
            "requires_done_readback": True,
        },
        "oauth_header_format": "Authorization: Bearer <metadata-token>",
        "stdlib_urllib_transport": True,
        "gcloud_dependency": False,
        "network_operation_performed": False,
    }


def _outer_object_sha(
    outer_manifest: Mapping[str, Any], relative: str
) -> str:
    matches = [
        row["sha256"]
        for row in outer_manifest["objects"]
        if row["path"] == relative
    ]
    if len(matches) != 1:
        raise ValueError(f"outer package lost required object {relative}")
    return matches[0]


def _build_worker_invocation(
    *,
    preview: Mapping[str, Any],
    adapter_job: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = adapter_job["job_id"]
    role = adapter_job["source_role"]
    if role not in ("candidate", "reference"):
        raise ValueError("worker source role changed")
    library_relative = (
        "native/candidate/release/libofc_hu_m3_engine.so"
        if role == "candidate"
        else "native/reference/release/libofc_hu_m3_engine.so"
    )
    return {
        "precontent_startup_argv": [
            "bash",
            f"inner/{INNER_STARTUP_NAME}",
            f"inner/{INNER_VERIFIER_NAME}",
            f"inner/{INNER_SOURCE_NAME}",
            f"inner/{INNER_MANIFEST_NAME}",
            f"inner/jobs/{job_id}.json",
            "work/output",
        ],
        "environment": {
            "OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256": _outer_object_sha(
                outer_manifest, f"inner/{INNER_SOURCE_NAME}"
            ),
            "OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256": preview[
                "package_manifest_sha256"
            ],
            "OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256": adapter_job[
                "runner_job_manifest"
            ]["sha256"],
            "OFC_DIAGNOSTIC_EXPECTED_JOB_ID": job_id,
            "OFC_DIAGNOSTIC_EXPECTED_STAGE_ID": preview["stage_id"],
            "OFC_DIAGNOSTIC_PRECONTENT_ONLY": "1",
            "OFC_DIAGNOSTIC_POISON_ROOT_READS": "1",
        },
        "direct_runner_argv": [
            "work/venv/bin/python",
            "-m",
            "ofc_regular.run_hu_m31_t3_step6d_performance_v2",
            "--repository-root",
            "work/extracted",
            "--output-dir",
            "work/output",
            "--shard-manifest",
            f"inner/jobs/{job_id}.json",
            "--library",
            f"work/extracted/{library_relative}",
        ],
        "direct_runner_environment": {
            "PYTHONPATH": "work/extracted/src",
            "RAYON_NUM_THREADS": "16",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OFC_HU_M3_BATCH_THREADS": "1",
        },
        "fresh_output_required": True,
        "attempt1_materializes_only_validated_remote_prefix": True,
        "offline_pip_install_argv": [
            "work/venv/bin/python",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--find-links",
            "wheels",
            "--requirement",
            (
                "work/extracted/configs/"
                "hu_m31_t3_step6d_rearm2_diagnostic_runtime_requirements_v1.txt"
            ),
        ],
        "network_pip_forbidden": True,
        "safe_allowlist_extraction_required": True,
        "zip_extractall_forbidden": True,
    }


def _build_authorization_contract(
    controller_public_key_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    public_key = (
        None
        if controller_public_key_record is None
        else validate_rsa_public_key_record(controller_public_key_record)
    )
    return {
        "controller_authorization_schema": AUTHORIZATION_SCHEMA,
        "worker_claim_schema": CLAIM_SCHEMA,
        "signature_algorithm": RSA_SIGNATURE_ALGORITHM,
        "controller_public_key_schema": RSA_PUBLIC_KEY_SCHEMA,
        "controller_public_key_sha256": (
            None if public_key is None else canonical_sha256(public_key)
        ),
        "controller_key_id": (
            None if public_key is None else public_key["key_id"]
        ),
        "trusted_verifier_required": True,
        "controller_private_key_embedded": False,
        "worker_shared_signing_secret_present": False,
        "writer_included": False,
        "authorization_embedded": False,
        "claim_embedded": False,
    }


def build_job_contract(
    *,
    package_dir: str | Path,
    stage_id: str,
    job_id: str,
    attempt_index: int = 0,
    offline_wheel_record: Mapping[str, Any] | None = None,
    controller_public_key_record: Mapping[str, Any] | None = None,
    prior_preview: Mapping[str, Any] | None = None,
    prior_snapshot: Mapping[str, Any] | None = None,
    prerequisite_stage1_preview: Mapping[str, Any] | None = None,
    prerequisite_stage1_receive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _package_module, adapter, _plan = _lazy_modules()
    preview = adapter.build_preview(
        package_dir=package_dir,
        stage_id=stage_id,
        attempt_index=attempt_index,
        prior_preview=prior_preview,
        prior_snapshot=prior_snapshot,
        prerequisite_stage1_preview=prerequisite_stage1_preview,
        prerequisite_stage1_receive=prerequisite_stage1_receive,
    )
    adapter_job = next(
        (row for row in preview["jobs"] if row["job_id"] == job_id), None
    )
    if adapter_job is None:
        raise ValueError("contract escaped selected stage jobs")
    outer = build_outer_package_manifest(
        package_dir=package_dir, offline_wheel_record=offline_wheel_record
    )
    layout = build_direct_v1_remote_layout(
        preview=preview, outer_manifest=outer
    )
    direct_job = next(
        (row for row in layout["jobs"] if row["job_id"] == job_id), None
    )
    if direct_job is None:
        raise AssertionError("direct layout lost selected job")
    direct_stage_sha = layout["direct_stage_identity_sha256"]
    instance_name = deterministic_instance_name(
        stage_id=stage_id,
        job_id=job_id,
        attempt_index=attempt_index,
        preview_stage_identity_sha256=preview["stage_identity_sha256"],
    )
    lifecycle = build_lifecycle_plan(
        preview=preview, direct_layout=layout, job_id=job_id
    )
    invocation = _build_worker_invocation(
        preview=preview,
        adapter_job=adapter_job,
        outer_manifest=outer,
    )
    binding = {
        "schema": METADATA_BINDING_SCHEMA,
        "project": PROJECT,
        "zone": ZONE,
        "instance_name": instance_name,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "stage_id": stage_id,
        "run_name": preview["run_name"],
        "job_id": job_id,
        "source_role": adapter_job["source_role"],
        "attempt_index": attempt_index,
        "max_attempts": MAX_ATTEMPTS,
        "inner_preview_stage_identity_sha256": preview[
            "stage_identity_sha256"
        ],
        "direct_stage_identity_sha256": direct_stage_sha,
        "outer_package_identity_sha256": outer[
            "outer_package_identity_sha256"
        ],
        "outer_package_manifest_sha256": canonical_sha256(outer),
        "runner_job_manifest_sha256": adapter_job["runner_job_manifest"][
            "sha256"
        ],
        "package_prefix": layout["package_prefix"],
        "stage_prefix": layout["stage_prefix"],
        "result_prefix": layout["result_prefix"],
        "attempt_control_prefix": layout["attempt_control_prefix"],
        "job_result_layout": direct_job,
        "worker_invocation": invocation,
        "image": {
            "project": IMAGE_PROJECT,
            "name": IMAGE_NAME,
            "id": IMAGE_ID,
            "self_link": IMAGE_SELF_LINK,
            "family_resolution_permitted": False,
        },
    }
    binding_sha = canonical_sha256(binding)
    binding_uri = (
        f"{layout['attempt_control_prefix']}/bootstrap/{job_id}.json"
    )
    metadata_values = {
        "ofc-direct-binding-uri": binding_uri,
        "ofc-direct-binding-sha256": binding_sha,
        "ofc-direct-stage-sha256": direct_stage_sha,
        "ofc-outer-package-sha256": outer[
            "outer_package_identity_sha256"
        ],
        "ofc-stage-id": stage_id,
        "ofc-job-id": job_id,
        "ofc-attempt-index": str(attempt_index),
        "ofc-instance-name": instance_name,
    }
    rest_plan = _build_rest_plan(
        layout=layout, lifecycle=lifecycle, instance_name=instance_name
    )
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "local_contract_external_preflight_and_authorization_required",
        "outer_package_manifest": outer,
        "outer_package_manifest_sha256": canonical_sha256(outer),
        "adapter_preview": preview,
        "adapter_preview_sha256": canonical_sha256(preview),
        "direct_stage_identity": layout["direct_stage_identity"],
        "direct_stage_identity_sha256": direct_stage_sha,
        "remote_layout": layout,
        "metadata_binding": binding,
        "metadata_binding_sha256": binding_sha,
        "metadata_values": metadata_values,
        "metadata_values_sha256": canonical_sha256(metadata_values),
        "rest_plan": rest_plan,
        "rest_plan_sha256": canonical_sha256(rest_plan),
        "lifecycle_plan": lifecycle,
        "lifecycle_plan_sha256": canonical_sha256(lifecycle),
        "source_provisioning": {
            "allowed_observed_states": [
                "exact_empty",
                "exact_subset",
                "exact_complete",
            ],
            "empty_or_subset_requires_separate_provision_authorization": True,
            "missing_only_generation_match_zero": True,
            "each_create_requires_readback": True,
            "unknown_or_mismatch_is_fatal": True,
            "complete_required_before_worker_launch": True,
            "observed_state": None,
        },
        "authorization_contract": _build_authorization_contract(
            controller_public_key_record
        ),
        "capabilities": {
            "metadata_rest_runtime_implemented": True,
            "gcs_generation_pinned_download_implemented": True,
            "gcs_generation_match_zero_upload_implemented": True,
            "gcs_readback_implemented": True,
            "compute_self_delete_implemented": True,
            "bounded_failure_shutdown_implemented": True,
            "gcloud_dependency": False,
            "external_preflight_passed": False,
            "authorization_present": False,
            "claim_present": False,
            "cloud_executable": False,
            "launch_ready": False,
            "cloud_launch_authorized": False,
            "remote_write_authorized": False,
            "vm_create_authorized": False,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
    }
    _reject_hidden(contract)
    return contract


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpClient(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None = None,
        timeout_seconds: int = 30,
    ) -> HttpResponse: ...


class ControllerTrustVerifier(Protocol):
    key_id: str
    public_key_sha256: str

    def verify(
        self,
        *,
        record_type: str,
        payload: bytes,
        signature: str,
    ) -> bool: ...


class ShutdownRequester(Protocol):
    def request_shutdown(self, *, deadline_seconds: int) -> None: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class UrllibHttpClient:
    """Contract-bound REST client with no redirects or proxy inheritance."""

    def __init__(self, *, contract: Mapping[str, Any]):
        checked = validate_job_contract(contract)
        rest = checked["rest_plan"]
        metadata_root = rest["metadata"]["root"].rstrip("/")
        self._metadata_urls = {
            metadata_root + path
            for path in rest["metadata"]["identity_paths"]
        }
        self._metadata_urls.add(metadata_root + rest["metadata"]["token_path"])
        self._metadata_urls.update(
            f"{metadata_root}/instance/attributes/{key}"
            for key in checked["metadata_values"]
        )
        self._package_media_templates = {
            row["generation_pinned_url_template"]
            for row in rest["package_downloads"]
        }
        self._api_exact_routes: dict[tuple[str, str], str] = {}
        for row in rest["object_metadata_get_url_templates"]:
            self._api_exact_routes[("GET", row["url"])] = "api_get"
        for row in rest["conditional_uploads"]:
            self._api_exact_routes[("POST", row["upload_url"])] = "api_post"
            self._api_exact_routes[
                ("GET", row["readback_metadata_url"])
            ] = "api_get"
            self._package_media_templates.add(
                row["readback_media_url_template"]
            )
        self._api_exact_routes[
            ("DELETE", rest["self_delete"]["url"])
        ] = "api_delete"
        no_proxy = urllib.request.ProxyHandler({})
        self._metadata_opener = urllib.request.build_opener(
            no_proxy, _NoRedirectHandler()
        )
        self._api_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler()
        )

    @staticmethod
    def _matches_positive_generation_template(
        url: str, template: str
    ) -> bool:
        prefix, marker, suffix = template.partition("{generation}")
        if marker != "{generation}" or not url.startswith(prefix) or not url.endswith(
            suffix
        ):
            return False
        end = len(url) - len(suffix) if suffix else len(url)
        generation = url[len(prefix) : end]
        return generation.isdigit() and int(generation) > 0

    def _classify_request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> str:
        if method not in ("GET", "POST", "DELETE") or not isinstance(url, str):
            raise ValueError("HTTP method or URL escaped the fixed contract")
        if not isinstance(headers, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or "\r" in key + value
            or "\n" in key + value
            for key, value in headers.items()
        ):
            raise ValueError("HTTP headers are malformed")
        normalized = {key.casefold(): value for key, value in headers.items()}
        if len(normalized) != len(headers):
            raise ValueError("duplicate case-insensitive HTTP header")
        if method == "GET" and url in self._metadata_urls:
            if (
                dict(headers) != {"Metadata-Flavor": "Google"}
                or body is not None
            ):
                raise ValueError("metadata request headers or body changed")
            return "metadata"
        route = self._api_exact_routes.get((method, url))
        if route is None and method == "GET" and any(
            self._matches_positive_generation_template(url, template)
            for template in self._package_media_templates
        ):
            route = "api_get"
        if route is None:
            raise ValueError("HTTP request escaped exact metadata/GCS/Compute URLs")
        authorization = normalized.get("authorization")
        if (
            not isinstance(authorization, str)
            or not authorization.startswith("Bearer ")
            or len(authorization) <= len("Bearer ")
        ):
            raise ValueError("Google API request lost bearer authorization")
        if route == "api_post":
            if (
                set(normalized) != {"authorization", "content-type"}
                or normalized["content-type"] != "application/octet-stream"
                or not isinstance(body, bytes)
            ):
                raise ValueError("GCS upload headers or body changed")
        elif set(normalized) != {"authorization"} or body is not None:
            raise ValueError("Google API read/delete headers or body changed")
        return route

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None = None,
        timeout_seconds: int = 30,
    ) -> HttpResponse:
        route = self._classify_request(
            method=method, url=url, headers=headers, body=body
        )
        _strict_int(
            timeout_seconds,
            "HTTP timeout",
            minimum=1,
            maximum=MAX_FAILURE_SHUTDOWN_SECONDS,
        )
        request = urllib.request.Request(
            url=url, data=body, headers=dict(headers), method=method
        )
        opener = (
            self._metadata_opener if route == "metadata" else self._api_opener
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status=int(response.status),
                    headers={key: value for key, value in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            return HttpResponse(
                status=int(error.code),
                headers={
                    key: value
                    for key, value in (
                        error.headers.items() if error.headers else []
                    )
                },
                body=error.read(),
            )


class SubprocessShutdownRequester:
    """Linux failure shutdown.  Never used by contract/local-preflight APIs."""

    def __init__(self, *, marker_path: str | Path | None = None) -> None:
        self.shutdown_attempted = False
        self.marker_path = (
            BOUNDED_SHUTDOWN_MARKER_PATH
            if marker_path is None
            else Path(marker_path)
        )

    def request_shutdown(self, *, deadline_seconds: int) -> None:
        global _SUBPROCESS_SHUTDOWN_ATTEMPTED
        if self.shutdown_attempted:
            raise RuntimeError("bounded failure shutdown was already attempted")
        self.shutdown_attempted = True
        _SUBPROCESS_SHUTDOWN_ATTEMPTED = True
        _strict_int(
            deadline_seconds,
            "failure shutdown deadline",
            minimum=1,
            maximum=MAX_FAILURE_SHUTDOWN_SECONDS,
        )
        marker = self.marker_path
        if marker.exists() or marker.is_symlink():
            return
        if not marker.parent.is_dir() or marker.parent.is_symlink():
            raise RuntimeError("bounded failure shutdown marker parent changed")
        try:
            with marker.open("xb") as handle:
                handle.write(b"shutdown-requested\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            return
        completed = subprocess.run(
            ["/sbin/shutdown", "-h", "now"],
            check=False,
            capture_output=True,
            timeout=deadline_seconds,
        )
        if completed.returncode != 0:
            if (
                marker.is_file()
                and not marker.is_symlink()
                and marker.read_bytes() == b"shutdown-requested\n"
            ):
                marker.unlink()
            raise RuntimeError("bounded failure shutdown request was rejected")


@dataclass(frozen=True)
class RuntimeApproval:
    contract_sha256: str
    authorization_sha256: str
    claim_sha256: str
    package_generations: Mapping[str, int]


def validate_controller_approval(
    *,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    verifier: ControllerTrustVerifier,
    now_unix_seconds: int,
) -> RuntimeApproval:
    """Validate externally issued records before any HTTP client is touched."""

    contract = validate_job_contract(contract)
    contract_sha = canonical_sha256(contract)
    trust = contract["authorization_contract"]
    key_id = trust["controller_key_id"]
    public_key_sha = trust["controller_public_key_sha256"]
    if (
        trust["signature_algorithm"] != RSA_SIGNATURE_ALGORITHM
        or not isinstance(key_id, str)
        or _sha(key_id, "controller key id") != key_id
        or not isinstance(public_key_sha, str)
        or _sha(public_key_sha, "controller public key sha256")
        != public_key_sha
        or getattr(verifier, "key_id", None) != key_id
        or getattr(verifier, "public_key_sha256", None) != public_key_sha
    ):
        raise ValueError("controller public trust anchor is not contract-bound")
    auth = dict(authorization)
    _exact(
        auth,
        {
            "schema",
            "contract_sha256",
            "metadata_binding_sha256",
            "direct_stage_identity_sha256",
            "job_id",
            "attempt_index",
            "instance_name",
            "controller_key_id",
            "external_preflight_receipt_sha256",
            "allowed_operations",
            "issued_unix_seconds",
            "expires_unix_seconds",
            "nonce",
            "signature",
        },
        "controller authorization",
    )
    binding = contract["metadata_binding"]
    expected_operations = [
        "metadata_identity_read",
        "metadata_token_read",
        "generation_pinned_package_download",
        "generation_match_zero_result_upload",
        "result_readback",
        "compute_delete_self_after_done",
        "bounded_safety_shutdown_on_any_worker_failure",
    ]
    if (
        auth["schema"] != AUTHORIZATION_SCHEMA
        or auth["contract_sha256"] != contract_sha
        or auth["metadata_binding_sha256"]
        != contract["metadata_binding_sha256"]
        or auth["direct_stage_identity_sha256"]
        != contract["direct_stage_identity_sha256"]
        or auth["job_id"] != binding["job_id"]
        or auth["instance_name"] != binding["instance_name"]
        or auth["controller_key_id"] != key_id
        or auth["allowed_operations"] != expected_operations
        or _sha(
            auth["external_preflight_receipt_sha256"],
            "external preflight receipt",
        )
        != auth["external_preflight_receipt_sha256"]
        or _sha(auth["nonce"], "authorization nonce") != auth["nonce"]
        or not isinstance(auth["signature"], str)
        or not auth["signature"]
    ):
        raise ValueError("controller authorization identity changed")
    _strict_int(
        auth["attempt_index"],
        "authorization attempt",
        minimum=binding["attempt_index"],
        maximum=binding["attempt_index"],
    )
    issued = _strict_int(
        auth["issued_unix_seconds"], "authorization issued", minimum=1
    )
    expires = _strict_int(
        auth["expires_unix_seconds"], "authorization expiry", minimum=issued + 1
    )
    _strict_int(now_unix_seconds, "current time", minimum=issued, maximum=expires)
    unsigned_auth = {key: value for key, value in auth.items() if key != "signature"}
    if not verifier.verify(
        record_type="authorization",
        payload=canonical_bytes(unsigned_auth),
        signature=auth["signature"],
    ):
        raise ValueError("controller authorization trust verification failed")
    auth_sha = canonical_sha256(auth)

    claimed = dict(claim)
    _exact(
        claimed,
        {
            "schema",
            "authorization_sha256",
            "contract_sha256",
            "project",
            "project_number",
            "zone",
            "instance_name",
            "instance_id",
            "worker_service_account",
            "controller_key_id",
            "stage_id",
            "job_id",
            "attempt_index",
            "package_generations",
            "nonce",
            "signature",
        },
        "worker claim",
    )
    generations = claimed["package_generations"]
    expected_uris = [
        row["uri"]
        for row in contract["remote_layout"]["package_inventory"]["records"]
    ]
    if not isinstance(generations, Mapping) or set(generations) != set(
        expected_uris
    ):
        raise ValueError("worker claim package generations are incomplete")
    checked_generations: dict[str, int] = {}
    for uri in expected_uris:
        checked_generations[uri] = _strict_int(
            generations[uri], "package generation", minimum=1
        )
    if (
        claimed["schema"] != CLAIM_SCHEMA
        or claimed["authorization_sha256"] != auth_sha
        or claimed["contract_sha256"] != contract_sha
        or claimed["project"] != PROJECT
        or not isinstance(claimed["project_number"], str)
        or not claimed["project_number"].isdigit()
        or claimed["zone"] != ZONE
        or claimed["instance_name"] != binding["instance_name"]
        or not isinstance(claimed["instance_id"], str)
        or not claimed["instance_id"].isdigit()
        or claimed["worker_service_account"] != WORKER_SERVICE_ACCOUNT
        or claimed["controller_key_id"] != key_id
        or claimed["stage_id"] != binding["stage_id"]
        or claimed["job_id"] != binding["job_id"]
        or _sha(claimed["nonce"], "claim nonce") != claimed["nonce"]
        or not isinstance(claimed["signature"], str)
        or not claimed["signature"]
    ):
        raise ValueError("worker claim identity changed")
    _strict_int(
        claimed["attempt_index"],
        "claim attempt",
        minimum=binding["attempt_index"],
        maximum=binding["attempt_index"],
    )
    unsigned_claim = {
        key: value for key, value in claimed.items() if key != "signature"
    }
    if not verifier.verify(
        record_type="claim",
        payload=canonical_bytes(unsigned_claim),
        signature=claimed["signature"],
    ):
        raise ValueError("worker claim trust verification failed")
    return RuntimeApproval(
        contract_sha256=contract_sha,
        authorization_sha256=auth_sha,
        claim_sha256=canonical_sha256(claimed),
        package_generations=checked_generations,
    )


def _expect_http(
    response: HttpResponse, *, statuses: set[int], label: str
) -> HttpResponse:
    if type(response.status) is not int or response.status not in statuses:
        raise RuntimeError(f"{label} returned HTTP {response.status!r}")
    if not isinstance(response.body, bytes):
        raise RuntimeError(f"{label} returned non-bytes body")
    return response


def _metadata_get(client: HttpClient, path: str) -> bytes:
    response = client.request(
        method="GET",
        url=(
            "http://metadata.google.internal/computeMetadata/v1/"
            + path.lstrip("/")
        ),
        headers={"Metadata-Flavor": "Google"},
        timeout_seconds=10,
    )
    response = _expect_http(response, statuses={200}, label="metadata read")
    if response.headers.get("Metadata-Flavor") != "Google":
        raise RuntimeError("metadata response flavor changed")
    return response.body


def read_metadata_identity_and_token(
    *,
    client: HttpClient,
    contract: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Read exact VM identity and a short-lived OAuth token after approval."""

    binding = contract["metadata_binding"]
    expected = {
        "project/project-id": PROJECT,
        "instance/id": claim["instance_id"],
        "instance/name": binding["instance_name"],
        "instance/zone": (
            f"projects/{claim['project_number']}/zones/{ZONE}"
        ),
        "instance/service-accounts/default/email": WORKER_SERVICE_ACCOUNT,
        "instance/service-accounts/default/scopes": REQUIRED_WORKER_OAUTH_SCOPE,
    }
    observed: dict[str, str] = {}
    for path, wanted in expected.items():
        value = _metadata_get(client, path).decode("utf-8")
        if path == "instance/service-accounts/default/scopes":
            scope_lines = value.splitlines()
            if scope_lines != [wanted]:
                raise RuntimeError(f"metadata identity mismatch at {path}")
            value = scope_lines[0]
        elif value != wanted:
            raise RuntimeError(f"metadata identity mismatch at {path}")
        observed[path] = value
    for key, wanted in contract["metadata_values"].items():
        value = _metadata_get(
            client, f"instance/attributes/{urllib.parse.quote(key, safe='')}"
        ).decode("utf-8")
        if value != wanted:
            raise RuntimeError(f"custom metadata mismatch at {key}")
    token_raw = _metadata_get(
        client, "instance/service-accounts/default/token"
    )
    try:
        token = json.loads(token_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("metadata OAuth token is not JSON") from error
    if (
        not isinstance(token, dict)
        or set(token) != {"access_token", "expires_in", "token_type"}
        or not isinstance(token["access_token"], str)
        or not token["access_token"]
        or token["token_type"] != "Bearer"
        or type(token["expires_in"]) is not int
        or token["expires_in"] < 60
    ):
        raise RuntimeError("metadata OAuth token shape changed")
    return {
        "identity": observed,
        "access_token": token["access_token"],
        "expires_in": token["expires_in"],
    }


def generation_pinned_download(
    *,
    client: HttpClient,
    access_token: str,
    record: Mapping[str, Any],
    generation: int,
    destination: str | Path,
) -> dict[str, Any]:
    generation = _strict_int(
        generation, "download generation", minimum=1
    )
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("download destination must be fresh")
    raw = _expect_http(
        client.request(
            method="GET",
            url=_gcs_media_get_url(
                record["uri"], generation=str(generation)
            ),
            headers={"Authorization": f"Bearer {access_token}"},
        ),
        statuses={200},
        label="generation-pinned download",
    ).body
    if (
        len(raw) != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("generation-pinned download identity changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "uri": record["uri"],
        "generation": generation,
        "sha256": record["sha256"],
        "bytes": record["bytes"],
    }


def conditional_create_and_readback(
    *,
    client: HttpClient,
    access_token: str,
    uri: str,
    content: bytes,
) -> dict[str, Any]:
    """Create with ifGenerationMatch=0, or accept byte-identical prior content."""

    digest = hashlib.sha256(content).hexdigest()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream",
    }
    response = client.request(
        method="POST",
        url=_gcs_upload_url(uri),
        headers=headers,
        body=content,
    )
    created = response.status in (200, 201)
    if created:
        try:
            metadata = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("conditional create response is not JSON") from error
    elif response.status == 412:
        metadata_response = _expect_http(
            client.request(
                method="GET",
                url=_gcs_metadata_get_url(uri),
                headers={"Authorization": f"Bearer {access_token}"},
            ),
            statuses={200},
            label="existing object metadata readback",
        )
        try:
            metadata = json.loads(metadata_response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("existing object metadata is not JSON") from error
    else:
        raise RuntimeError(f"conditional create returned HTTP {response.status}")
    generation_raw = metadata.get("generation") if isinstance(metadata, dict) else None
    if (
        not isinstance(generation_raw, str)
        or not generation_raw.isdigit()
        or int(generation_raw) <= 0
    ):
        raise RuntimeError("object generation is missing")
    generation = int(generation_raw)
    readback = _expect_http(
        client.request(
            method="GET",
            url=_gcs_media_get_url(uri, generation=str(generation)),
            headers={"Authorization": f"Bearer {access_token}"},
        ),
        statuses={200},
        label="object media readback",
    ).body
    if readback != content or hashlib.sha256(readback).hexdigest() != digest:
        raise FileExistsError("generation-match object differs from expected bytes")
    return {
        "uri": uri,
        "generation": generation,
        "created": created,
        "sha256": digest,
        "bytes": len(content),
    }


def request_compute_self_delete(
    *,
    client: HttpClient,
    access_token: str,
    contract: Mapping[str, Any],
    done_readback: Mapping[str, Any],
) -> dict[str, Any]:
    expected_done_uri = next(
        row["done_uri"]
        for row in contract["remote_layout"]["jobs"]
        if row["job_id"] == contract["metadata_binding"]["job_id"]
    )
    if (
        done_readback.get("uri") != expected_done_uri
        or done_readback.get("sha256") is None
        or done_readback.get("generation") is None
    ):
        raise ValueError("self-delete requires exact DONE readback")
    response = _expect_http(
        client.request(
            method="DELETE",
            url=contract["rest_plan"]["self_delete"]["url"],
            headers={"Authorization": f"Bearer {access_token}"},
        ),
        statuses={200, 202},
        label="Compute self-delete",
    )
    return {
        "instance_name": contract["metadata_binding"]["instance_name"],
        "done_uri": expected_done_uri,
        "delete_requested": True,
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
    }


def request_bounded_failure_shutdown(
    *,
    requester: ShutdownRequester,
    deadline_seconds: int,
    done_published: bool,
) -> dict[str, Any]:
    if type(done_published) is not bool:
        raise ValueError("DONE publication state must be a strict boolean")
    deadline = _strict_int(
        deadline_seconds,
        "failure shutdown deadline",
        minimum=1,
        maximum=MAX_FAILURE_SHUTDOWN_SECONDS,
    )
    requester.request_shutdown(deadline_seconds=deadline)
    return {
        "done_published": done_published,
        "self_delete_requested": False,
        "preserve_completed_objects": True,
        "shutdown_requested": True,
        "shutdown_deadline_seconds": deadline,
        "reason": (
            "post_done_self_delete_failure"
            if done_published
            else "worker_failure_before_done"
        ),
    }


def safe_extract_worker_source(
    *,
    source_zip: str | Path,
    inner_manifest: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, Any]:
    """Rehash an allowlisted ZIP into a fresh tree using exclusive writes."""

    target_root = Path(destination)
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError("safe extraction destination must be fresh")
    expected = inner_manifest.get("source_entries")
    if not isinstance(expected, Mapping) or len(expected) != 60:
        raise ValueError("inner source allowlist changed")
    source = Path(source_zip)
    if not source.is_file() or source.is_symlink():
        raise ValueError("worker source archive is not a regular file")
    staged: list[tuple[str, bytes, int]] = []
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or {row.filename for row in infos} != set(
            expected
        ):
            raise ValueError("worker source archive allowlist changed")
        for info in infos:
            relative = _safe_relative(info.filename)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or unix_mode not in (0, 0o100000):
                raise ValueError("worker source archive contains unsafe member")
            raw = archive.read(info)
            record = expected[relative]
            if (
                not isinstance(record, Mapping)
                or record.get("bytes") != len(raw)
                or record.get("sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("worker source member identity changed")
            staged.append((relative, raw, info.external_attr))
    target_root.mkdir()
    try:
        for relative, raw, external_attr in staged:
            destination_path = target_root / PurePosixPath(relative)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with destination_path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            if (external_attr >> 16) & 0o111:
                destination_path.chmod(0o755)
        records = [
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            for relative, raw, _mode in staged
        ]
        return {
            "file_count": len(records),
            "records_sha256": canonical_sha256(records),
            "fresh_exclusive_write": True,
            "zip_extractall_used": False,
        }
    except BaseException:
        if target_root.is_dir() and not target_root.is_symlink():
            shutil.rmtree(target_root)
        raise


def download_outer_package(
    *,
    client: HttpClient,
    access_token: str,
    contract: Mapping[str, Any],
    approval: RuntimeApproval,
    destination: str | Path,
) -> dict[str, Any]:
    if approval.contract_sha256 != canonical_sha256(contract):
        raise ValueError("runtime approval is bound to another contract")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("outer package destination must be fresh")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("outer package parent must be an existing real directory")
    target.mkdir()
    records = contract["remote_layout"]["package_inventory"]["records"]
    downloaded: list[dict[str, Any]] = []
    try:
        for record in records:
            destination_path = target / PurePosixPath(record["path"])
            downloaded.append(
                generation_pinned_download(
                    client=client,
                    access_token=access_token,
                    record=record,
                    generation=approval.package_generations[record["uri"]],
                    destination=destination_path,
                )
            )
        manifest_path = target / "outer-manifest.json"
        loaded_outer = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded_outer != contract["outer_package_manifest"]:
            raise ValueError("downloaded outer manifest changed")
        validate_outer_package_manifest(loaded_outer)
        return {
            "destination": str(target),
            "object_count": len(downloaded),
            "downloaded_sha256": canonical_sha256(downloaded),
            "generation_pinned": True,
            "fresh_exclusive_write": True,
        }
    except BaseException:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        raise


def _artifact_file(
    path: Path,
    *,
    relative: str,
    source_role: str,
    hand_index: int,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("runner artifact is missing or not regular")
    return {
        "source_role": source_role,
        "hand_index": hand_index,
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def publish_validated_runner_output(
    *,
    client: HttpClient,
    access_token: str,
    contract: Mapping[str, Any],
    approval: RuntimeApproval,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and publish one exact runner tree in adapter envelope order."""

    if approval.contract_sha256 != canonical_sha256(contract):
        raise ValueError("runtime approval is bound to another contract")
    adapter = _runtime_adapter_module()
    output = Path(output_dir)
    validate_completed_output_with_worker_venv(output)
    preview = contract["adapter_preview"]
    job_id = contract["metadata_binding"]["job_id"]
    adapter_job = next(row for row in preview["jobs"] if row["job_id"] == job_id)
    direct_job = next(
        row
        for row in contract["remote_layout"]["jobs"]
        if row["job_id"] == job_id
    )
    direct_tree = {
        row["path"]: uri
        for row, uri in zip(
            adapter_job["tree_object_manifest"],
            direct_job["tree_object_uris"],
            strict=True,
        )
    }
    readbacks: list[dict[str, Any]] = []
    for relative in ("run_contract.json", "shard_manifest.json"):
        readbacks.append(
            conditional_create_and_readback(
                client=client,
                access_token=access_token,
                uri=direct_tree[relative],
                content=(output / relative).read_bytes(),
            )
        )
    uploads: list[dict[str, Any]] = []
    heartbeats: list[dict[str, Any]] = []
    role = adapter_job["source_role"]
    for sequence, hand_index in enumerate(
        adapter_job["work_hand_indices"], 1
    ):
        root_relative = f"roots/hand_{hand_index:03d}.json"
        hand_relative = f"hands/{role}/hand_{hand_index:03d}.json"
        root_file = _artifact_file(
            output / root_relative,
            relative=root_relative,
            source_role=role,
            hand_index=hand_index,
        )
        hand_file = _artifact_file(
            output / hand_relative,
            relative=hand_relative,
            source_role=role,
            hand_index=hand_index,
        )
        upload = adapter.build_artifact_upload(
            preview,
            job_id=job_id,
            sequence=sequence,
            root_file=root_file,
            source_hand_file=hand_file,
            transport_fixture_only=False,
        )
        for file_record in (root_file, hand_file):
            readbacks.append(
                conditional_create_and_readback(
                    client=client,
                    access_token=access_token,
                    uri=direct_tree[file_record["path"]],
                    content=(output / file_record["path"]).read_bytes(),
                )
            )
        readbacks.append(
            conditional_create_and_readback(
                client=client,
                access_token=access_token,
                uri=direct_job["upload_uris"][sequence - 1],
                content=adapter.canonical_bytes(upload),
            )
        )
        uploads.append(upload)
        heartbeat = adapter.build_heartbeat(
            preview, job_id=job_id, uploads=uploads
        )
        readbacks.append(
            conditional_create_and_readback(
                client=client,
                access_token=access_token,
                uri=direct_job["heartbeat_uris"][sequence - 1],
                content=adapter.canonical_bytes(heartbeat),
            )
        )
        heartbeats.append(heartbeat)
    runner_done = {
        "path": "DONE.json",
        "sha256": sha256_file(output / "DONE.json"),
        "bytes": (output / "DONE.json").stat().st_size,
    }
    done = adapter.build_done(
        preview,
        job_id=job_id,
        uploads=uploads,
        heartbeats=heartbeats,
        runner_done_file=runner_done,
    )
    readbacks.append(
        conditional_create_and_readback(
            client=client,
            access_token=access_token,
            uri=direct_tree["DONE.json"],
            content=(output / "DONE.json").read_bytes(),
        )
    )
    done_readback = conditional_create_and_readback(
        client=client,
        access_token=access_token,
        uri=direct_job["done_uri"],
        content=adapter.canonical_bytes(done),
    )
    readbacks.append(done_readback)
    return {
        "job_id": job_id,
        "done": done,
        "done_readback": done_readback,
        "object_readbacks": readbacks,
        "object_readbacks_sha256": canonical_sha256(readbacks),
        "runner_validate_completed_output_called": True,
        "done_published_last": True,
    }


def validate_job_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    required = {
        "schema",
        "status",
        "outer_package_manifest",
        "outer_package_manifest_sha256",
        "adapter_preview",
        "adapter_preview_sha256",
        "direct_stage_identity",
        "direct_stage_identity_sha256",
        "remote_layout",
        "metadata_binding",
        "metadata_binding_sha256",
        "metadata_values",
        "metadata_values_sha256",
        "rest_plan",
        "rest_plan_sha256",
        "lifecycle_plan",
        "lifecycle_plan_sha256",
        "source_provisioning",
        "authorization_contract",
        "capabilities",
    }
    _exact(contract, required, "10c2 job contract")
    if (
        contract["schema"] != CONTRACT_SCHEMA
        or contract["status"]
        != "local_contract_external_preflight_and_authorization_required"
    ):
        raise ValueError("10c2 contract schema or status changed")
    outer = validate_outer_package_manifest(contract["outer_package_manifest"])
    for field, payload in (
        ("outer_package_manifest_sha256", outer),
        ("adapter_preview_sha256", contract["adapter_preview"]),
        ("metadata_binding_sha256", contract["metadata_binding"]),
        ("metadata_values_sha256", contract["metadata_values"]),
        ("rest_plan_sha256", contract["rest_plan"]),
        ("lifecycle_plan_sha256", contract["lifecycle_plan"]),
    ):
        if contract[field] != canonical_sha256(payload):
            raise ValueError(f"{field} changed")
    preview = contract["adapter_preview"]
    if not isinstance(preview, Mapping):
        raise ValueError("adapter preview is not a mapping")
    expected_stage = {
        STAGE1_ID: (
            STAGE1_RUN_NAME,
            list(STAGE1_JOB_IDS),
            list(STAGE1_HAND_INDICES),
        ),
        STAGE2_ID: (
            STAGE2_RUN_NAME,
            list(STAGE2_JOB_IDS),
            list(STAGE2_HAND_INDICES),
        ),
    }.get(preview.get("stage_id"))
    if expected_stage is None:
        raise ValueError("adapter preview escaped fixed diagnostic stages")
    expected_run_name, expected_job_ids, expected_hand_indices = expected_stage
    if (
        preview.get("run_name") != expected_run_name
        or preview.get("selected_job_ids") != expected_job_ids
        or preview.get("attempt_index") not in (0, 1)
        or preview.get("max_attempts") != MAX_ATTEMPTS
        or not isinstance(preview.get("jobs"), list)
        or [row.get("job_id") for row in preview["jobs"]] != expected_job_ids
        or any(
            row.get("work_hand_indices") != expected_hand_indices
            for row in preview["jobs"]
        )
    ):
        raise ValueError("adapter preview fixed stage layout changed")
    expected_layout = build_direct_v1_remote_layout(
        preview=preview, outer_manifest=outer
    )
    if contract["remote_layout"] != expected_layout:
        raise ValueError("remote layout or nested URL changed")
    layout = expected_layout
    direct_identity = layout["direct_stage_identity"]
    direct_sha = layout["direct_stage_identity_sha256"]
    if (
        contract["direct_stage_identity"] != direct_identity
        or contract["direct_stage_identity_sha256"] != direct_sha
    ):
        raise ValueError("direct stage identity changed")
    binding = contract["metadata_binding"]
    adapter_job = next(
        (
            row
            for row in preview["jobs"]
            if row["job_id"] == binding.get("job_id")
        ),
        None,
    )
    direct_job = next(
        (
            row
            for row in layout["jobs"]
            if row["job_id"] == binding.get("job_id")
        ),
        None,
    )
    if adapter_job is None or direct_job is None:
        raise ValueError("metadata binding escaped fixed job")
    expected_instance = deterministic_instance_name(
        stage_id=preview["stage_id"],
        job_id=adapter_job["job_id"],
        attempt_index=preview["attempt_index"],
        preview_stage_identity_sha256=preview["stage_identity_sha256"],
    )
    expected_binding = {
        "schema": METADATA_BINDING_SCHEMA,
        "project": PROJECT,
        "zone": ZONE,
        "instance_name": expected_instance,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": adapter_job["job_id"],
        "source_role": adapter_job["source_role"],
        "attempt_index": preview["attempt_index"],
        "max_attempts": MAX_ATTEMPTS,
        "inner_preview_stage_identity_sha256": preview[
            "stage_identity_sha256"
        ],
        "direct_stage_identity_sha256": direct_sha,
        "outer_package_identity_sha256": outer[
            "outer_package_identity_sha256"
        ],
        "outer_package_manifest_sha256": canonical_sha256(outer),
        "runner_job_manifest_sha256": adapter_job["runner_job_manifest"][
            "sha256"
        ],
        "package_prefix": layout["package_prefix"],
        "stage_prefix": layout["stage_prefix"],
        "result_prefix": layout["result_prefix"],
        "attempt_control_prefix": layout["attempt_control_prefix"],
        "job_result_layout": direct_job,
        "worker_invocation": _build_worker_invocation(
            preview=preview,
            adapter_job=adapter_job,
            outer_manifest=outer,
        ),
        "image": {
            "project": IMAGE_PROJECT,
            "name": IMAGE_NAME,
            "id": IMAGE_ID,
            "self_link": IMAGE_SELF_LINK,
            "family_resolution_permitted": False,
        },
    }
    if binding != expected_binding:
        raise ValueError("metadata binding or nested layout changed")
    expected_metadata_values = {
        "ofc-direct-binding-uri": (
            f"{layout['attempt_control_prefix']}/bootstrap/"
            f"{adapter_job['job_id']}.json"
        ),
        "ofc-direct-binding-sha256": canonical_sha256(expected_binding),
        "ofc-direct-stage-sha256": direct_sha,
        "ofc-outer-package-sha256": outer[
            "outer_package_identity_sha256"
        ],
        "ofc-stage-id": preview["stage_id"],
        "ofc-job-id": adapter_job["job_id"],
        "ofc-attempt-index": str(preview["attempt_index"]),
        "ofc-instance-name": expected_instance,
    }
    if contract["metadata_values"] != expected_metadata_values:
        raise ValueError("metadata values or binding URI changed")
    expected_lifecycle = build_lifecycle_plan(
        preview=preview,
        direct_layout=layout,
        job_id=adapter_job["job_id"],
    )
    if contract["lifecycle_plan"] != expected_lifecycle:
        raise ValueError("worker lifecycle or result URI changed")
    expected_rest = _build_rest_plan(
        layout=layout,
        lifecycle=expected_lifecycle,
        instance_name=expected_instance,
    )
    if contract["rest_plan"] != expected_rest:
        raise ValueError("REST plan or fixed endpoint changed")
    expected_source_provisioning = {
        "allowed_observed_states": [
            "exact_empty",
            "exact_subset",
            "exact_complete",
        ],
        "empty_or_subset_requires_separate_provision_authorization": True,
        "missing_only_generation_match_zero": True,
        "each_create_requires_readback": True,
        "unknown_or_mismatch_is_fatal": True,
        "complete_required_before_worker_launch": True,
        "observed_state": None,
    }
    if contract["source_provisioning"] != expected_source_provisioning:
        raise ValueError("source provisioning contract changed")
    trust = contract["authorization_contract"]
    if not isinstance(trust, Mapping):
        raise ValueError("authorization contract is not a mapping")
    configured_key = trust.get("controller_public_key_sha256")
    configured_key_id = trust.get("controller_key_id")
    if (configured_key is None) != (configured_key_id is None):
        raise ValueError("controller public-key binding is partial")
    if configured_key is not None:
        _sha(configured_key, "controller public key sha256")
        _sha(configured_key_id, "controller key id")
    expected_trust = _build_authorization_contract(None)
    expected_trust["controller_public_key_sha256"] = configured_key
    expected_trust["controller_key_id"] = configured_key_id
    if trust != expected_trust:
        raise ValueError("authorization or asymmetric trust contract changed")
    expected_capabilities = {
        "metadata_rest_runtime_implemented": True,
        "gcs_generation_pinned_download_implemented": True,
        "gcs_generation_match_zero_upload_implemented": True,
        "gcs_readback_implemented": True,
        "compute_self_delete_implemented": True,
        "bounded_failure_shutdown_implemented": True,
        "gcloud_dependency": False,
        "external_preflight_passed": False,
        "authorization_present": False,
        "claim_present": False,
        "cloud_executable": False,
        "launch_ready": False,
        "cloud_launch_authorized": False,
        "remote_write_authorized": False,
        "vm_create_authorized": False,
        "current_profile_changed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    if contract["capabilities"] != expected_capabilities:
        raise ValueError("capability contract changed")
    _reject_hidden(contract)
    _reject_unapproved_network_locations(contract)
    return contract


class RsaSha256ControllerTrustVerifier:
    """Public-only RFC 8017 RSASSA-PKCS1-v1_5 SHA-256 verifier."""

    _SHA256_DIGEST_INFO_PREFIX = bytes.fromhex(
        "3031300d060960864801650304020105000420"
    )

    def __init__(self, public_key_record: Mapping[str, Any]):
        checked = validate_rsa_public_key_record(public_key_record)
        self.key_id = checked["key_id"]
        self.public_key_sha256 = canonical_sha256(checked)
        self._modulus = int(checked["modulus_hex"], 16)
        self._exponent = checked["exponent"]
        self._modulus_bytes = (self._modulus.bit_length() + 7) // 8

    def verify(
        self,
        *,
        record_type: str,
        payload: bytes,
        signature: str,
    ) -> bool:
        if record_type not in ("authorization", "claim"):
            return False
        if not isinstance(payload, bytes) or not isinstance(signature, str):
            return False
        if re.fullmatch(r"[A-Za-z0-9_-]+", signature) is None:
            return False
        try:
            padding = "=" * (-len(signature) % 4)
            signature_raw = base64.b64decode(
                signature + padding,
                altchars=b"-_",
                validate=True,
            )
        except (ValueError, TypeError):
            return False
        if (
            len(signature_raw) != self._modulus_bytes
            or "=" in signature
            or not signature
        ):
            return False
        signature_integer = int.from_bytes(signature_raw, "big")
        if signature_integer >= self._modulus:
            return False
        encoded = pow(
            signature_integer,
            self._exponent,
            self._modulus,
        ).to_bytes(self._modulus_bytes, "big")
        digest_info = self._SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(
            record_type.encode("ascii") + b"\0" + payload
        ).digest()
        padding_length = self._modulus_bytes - len(digest_info) - 3
        if padding_length < 8:
            return False
        expected = (
            b"\x00\x01"
            + b"\xff" * padding_length
            + b"\x00"
            + digest_info
        )
        return secrets.compare_digest(encoded, expected)


def validate_completed_output_with_worker_venv(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate output with the same offline venv used by the worker."""

    output = Path(output_dir).resolve()
    work = output.parent
    extracted = work / "extracted"
    venv = work / "venv"
    venv_bin = venv / "bin"
    venv_python = venv_bin / "python"
    pyvenv_config = venv / "pyvenv.cfg"
    if (
        output.name != "output"
        or not output.is_dir()
        or output.is_symlink()
        or not extracted.is_dir()
        or extracted.is_symlink()
        or not venv.is_dir()
        or venv.is_symlink()
        or not venv_bin.is_dir()
        or venv_bin.is_symlink()
        or not pyvenv_config.is_file()
        or pyvenv_config.is_symlink()
        or not venv_python.is_file()
    ):
        raise ValueError("worker validation layout changed")
    config_raw = pyvenv_config.read_bytes()
    if (
        not 1 <= len(config_raw) <= 65_536
        or b"home = " not in config_raw
        or b"include-system-site-packages = false" not in config_raw
    ):
        raise ValueError("worker venv configuration changed")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extracted / "src")
    completed = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import pathlib,sys;"
                "expected=pathlib.Path(sys.argv[2]).resolve();"
                "actual=pathlib.Path(sys.prefix).resolve();"
                "assert actual==expected and sys.prefix!=sys.base_prefix;"
                "from ofc_regular import "
                "run_hu_m31_t3_step6d_performance_v2 as runner;"
                "runner.validate_completed_output(sys.argv[1])"
            ),
            str(output),
            str(venv),
        ],
        cwd=extracted,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError("worker-venv output validation failed")
    return {
        "output_dir": str(output),
        "worker_venv_python": str(venv_python),
        "worker_venv_prefix": str(venv),
        "pythonpath": str(extracted / "src"),
        "returncode": 0,
        "venv_prefix_verified_in_subprocess": True,
        "runner_validate_completed_output_performed": True,
        "parent_interpreter_numpy_required": False,
    }


def run_downloaded_worker(
    *,
    contract: Mapping[str, Any],
    outer_root: str | Path,
    work_root: str | Path,
) -> Path:
    """Run the exact worker from a downloaded complete outer package."""

    validate_job_contract(contract)
    if contract["outer_package_manifest"]["complete_for_direct_v1"] is not True:
        raise RuntimeError("full worker execution requires the pinned offline wheel")
    outer = Path(outer_root)
    work = Path(work_root)
    if work.exists() or work.is_symlink():
        raise FileExistsError("worker work root must be fresh")
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    inner = outer / "inner"
    binding = contract["metadata_binding"]
    invocation = binding["worker_invocation"]
    manifest_path = outer / invocation["precontent_startup_argv"][4]
    manifest_raw = manifest_path.read_bytes()
    inner_manifest = json.loads(manifest_raw)
    if (
        not isinstance(inner_manifest, dict)
        or canonical_bytes(inner_manifest) + b"\n" != manifest_raw
        or hashlib.sha256(manifest_raw).hexdigest()
        != contract["outer_package_manifest"][
            "inner_package_manifest_sha256"
        ]
    ):
        raise ValueError("downloaded inner manifest changed")
    job_id = binding["job_id"]
    job_record = next(
        row for row in inner_manifest["job_manifests"] if row["job_id"] == job_id
    )
    environment = dict(os.environ)
    environment.update(invocation["environment"])
    precontent = invocation["precontent_startup_argv"]
    subprocess.run(
        precontent,
        cwd=outer,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    extracted = work / "extracted"
    safe_extract_worker_source(
        source_zip=outer / invocation["precontent_startup_argv"][3],
        inner_manifest=inner_manifest,
        destination=extracted,
    )
    import ofc_regular

    extracted_package = str(extracted / "src" / "ofc_regular")
    if extracted_package not in ofc_regular.__path__:
        ofc_regular.__path__.append(extracted_package)
    output = work / "output"
    roots = output / "roots"
    roots.mkdir(parents=True)
    for index in job_record["work_hand_indices"]:
        relative = (
            "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
            f"tail_reselection_v2/roots/hand_{index:03d}.json"
        )
        raw = (extracted / relative).read_bytes()
        with (roots / f"hand_{index:03d}.json").open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    subprocess.run(
        ["python3", "-m", "venv", str(work / "venv")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    venv_python = work / "venv" / "bin" / "python"
    wheel = outer / "wheels" / EXPECTED_NUMPY_WHEEL_FILENAME
    if (
        not wheel.is_file()
        or wheel.stat().st_size != EXPECTED_NUMPY_WHEEL_BYTES
        or sha256_file(wheel) != EXPECTED_NUMPY_WHEEL_SHA256
    ):
        raise ValueError("downloaded offline wheel changed")
    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    runner_argv = list(invocation["direct_runner_argv"])
    runner_argv[0] = str(venv_python)
    replacements = {
        "work/extracted": str(extracted),
        "work/output": str(output),
        f"inner/jobs/{job_id}.json": str(inner / "jobs" / f"{job_id}.json"),
        (
            "work/extracted/native/candidate/release/"
            "libofc_hu_m3_engine.so"
        ): str(
            extracted
            / "native"
            / "candidate"
            / "release"
            / "libofc_hu_m3_engine.so"
        ),
        (
            "work/extracted/native/reference/release/"
            "libofc_hu_m3_engine.so"
        ): str(
            extracted
            / "native"
            / "reference"
            / "release"
            / "libofc_hu_m3_engine.so"
        ),
    }
    runner_argv = [replacements.get(value, value) for value in runner_argv]
    runner_environment = dict(environment)
    runner_environment.update(
        invocation["direct_runner_environment"]
    )
    runner_environment["PYTHONPATH"] = str(extracted / "src")
    subprocess.run(
        runner_argv,
        cwd=outer,
        env=runner_environment,
        check=True,
        text=True,
        encoding="utf-8",
    )
    validate_completed_output_with_worker_venv(output)
    return output


def execute_authorized_worker(
    *,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    verifier: ControllerTrustVerifier,
    client: HttpClient,
    shutdown_requester: ShutdownRequester,
    fresh_root: str | Path,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    """Run the full direct worker lifecycle after external approval."""

    done_readback: dict[str, Any] | None = None
    try:
        checked_contract = validate_job_contract(contract)
        if (
            checked_contract["outer_package_manifest"][
                "complete_for_direct_v1"
            ]
            is not True
        ):
            raise RuntimeError(
                "authorized worker requires complete outer package before network"
            )
        approval = validate_controller_approval(
            contract=checked_contract,
            authorization=authorization,
            claim=claim,
            verifier=verifier,
            now_unix_seconds=(
                int(time.time())
                if now_unix_seconds is None
                else now_unix_seconds
            ),
        )
        metadata = read_metadata_identity_and_token(
            client=client, contract=checked_contract, claim=claim
        )
        token = metadata["access_token"]
        root = Path(fresh_root)
        if root.exists() or root.is_symlink():
            raise FileExistsError("authorized worker root must be fresh")
        if not root.parent.is_dir() or root.parent.is_symlink():
            raise ValueError(
                "authorized worker parent must be an existing directory"
            )
        root.mkdir()
        download_outer_package(
            client=client,
            access_token=token,
            contract=checked_contract,
            approval=approval,
            destination=root / "outer",
        )
        output = run_downloaded_worker(
            contract=checked_contract,
            outer_root=root / "outer",
            work_root=root / "work",
        )
        publish_metadata = read_metadata_identity_and_token(
            client=client, contract=checked_contract, claim=claim
        )
        if publish_metadata["identity"] != metadata["identity"]:
            raise RuntimeError("metadata identity changed before publish")
        published = publish_validated_runner_output(
            client=client,
            access_token=publish_metadata["access_token"],
            contract=checked_contract,
            approval=approval,
            output_dir=output,
        )
        done_readback = published["done_readback"]
        delete_metadata = read_metadata_identity_and_token(
            client=client, contract=checked_contract, claim=claim
        )
        if delete_metadata["identity"] != metadata["identity"]:
            raise RuntimeError("metadata identity changed before self-delete")
        deletion = request_compute_self_delete(
            client=client,
            access_token=delete_metadata["access_token"],
            contract=checked_contract,
            done_readback=done_readback,
        )
        return {
            "status": "done_readback_then_self_delete_requested",
            "approval": {
                "contract_sha256": approval.contract_sha256,
                "authorization_sha256": approval.authorization_sha256,
                "claim_sha256": approval.claim_sha256,
            },
            "metadata_identity": metadata["identity"],
            "metadata_token_refreshes": {
                "initial_download": True,
                "before_publish": True,
                "before_self_delete": True,
                "access_token_recorded": False,
            },
            "published": published,
            "self_delete": deletion,
        }
    except BaseException as worker_error:
        try:
            shutdown_report = request_bounded_failure_shutdown(
                requester=shutdown_requester,
                deadline_seconds=MAX_FAILURE_SHUTDOWN_SECONDS,
                done_published=done_readback is not None,
            )
        except BaseException as shutdown_error:
            raise BaseExceptionGroup(
                "worker failed and bounded safety shutdown also failed",
                [worker_error, shutdown_error],
            ) from None
        if hasattr(worker_error, "add_note"):
            worker_error.add_note(
                "10c2 bounded safety shutdown: "
                + canonical_bytes(shutdown_report).decode("ascii")
            )
        raise


def local_preflight(
    *,
    contract: Mapping[str, Any],
    package_mirror: str | Path,
    fresh_work_root: str | Path,
    offline_wheel_mirror: str | Path | None = None,
    offline_install_smoke: bool = False,
) -> dict[str, Any]:
    """Exercise local copy/hash/safe-extract/precontent without cloud access."""

    checked = validate_job_contract(contract)
    if type(offline_install_smoke) is not bool:
        raise ValueError("offline install smoke must be a strict boolean")
    package = Path(package_mirror).resolve()
    package_module, _adapter, _plan = _lazy_modules()
    package_module.validate_package(package)
    root = Path(fresh_work_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("local preflight root must be fresh")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("local preflight parent must exist")
    root.mkdir()
    outer = root / "outer"
    outer.mkdir()
    copied: list[dict[str, Any]] = []
    manifest_raw = canonical_bytes(checked["outer_package_manifest"])
    (outer / "outer-manifest.json").write_bytes(manifest_raw)
    for record in checked["outer_package_manifest"]["objects"]:
        relative = record["path"]
        if relative.startswith("inner/"):
            source = package / relative.removeprefix("inner/")
        elif relative == f"wheels/{EXPECTED_NUMPY_WHEEL_FILENAME}":
            if offline_wheel_mirror is None:
                raise ValueError("outer contract requires offline wheel mirror")
            source = Path(offline_wheel_mirror).resolve()
        else:
            source = _REPO_ROOT / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != record["bytes"]
            or sha256_file(source) != record["sha256"]
        ):
            raise ValueError(f"local outer mirror changed: {relative}")
        destination = outer / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            raw = source.read_bytes()
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        copied.append(
            {
                "path": relative,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
        )
    inner_manifest = package_module.validate_package(outer / "inner")
    extraction = safe_extract_worker_source(
        source_zip=outer / "inner" / package_module.SOURCE_NAME,
        inner_manifest=inner_manifest,
        destination=root / "extracted",
    )
    binding = checked["metadata_binding"]
    environment = dict(os.environ)
    environment.update(binding["worker_invocation"]["environment"])
    if os.name == "nt":
        invocation = binding["worker_invocation"]
        completed = subprocess.run(
            [
                *package_module._bash_prefix(),
                "-c",
                (
                    'export OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256="$1" '
                    'OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256="$2" '
                    'OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256="$3" '
                    'OFC_DIAGNOSTIC_EXPECTED_JOB_ID="$4" '
                    'OFC_DIAGNOSTIC_EXPECTED_STAGE_ID="$5" '
                    "OFC_DIAGNOSTIC_PRECONTENT_ONLY=1 "
                    "OFC_DIAGNOSTIC_POISON_ROOT_READS=1; "
                    'exec bash "$6" "$7" "$8" "$9" "${10}" "${11}"'
                ),
                "ofc-10c2-local-precontent",
                invocation["environment"][
                    "OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256"
                ],
                invocation["environment"][
                    "OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256"
                ],
                invocation["environment"][
                    "OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256"
                ],
                invocation["environment"]["OFC_DIAGNOSTIC_EXPECTED_JOB_ID"],
                invocation["environment"]["OFC_DIAGNOSTIC_EXPECTED_STAGE_ID"],
                package_module._bash_path(
                    outer / invocation["precontent_startup_argv"][1]
                ),
                package_module._bash_path(
                    outer / invocation["precontent_startup_argv"][2]
                ),
                package_module._bash_path(
                    outer / invocation["precontent_startup_argv"][3]
                ),
                package_module._bash_path(
                    outer / invocation["precontent_startup_argv"][4]
                ),
                package_module._bash_path(
                    outer / invocation["precontent_startup_argv"][5]
                ),
                package_module._bash_path(root / "unused-precontent-output"),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    else:
        completed = subprocess.run(
            binding["worker_invocation"]["precontent_startup_argv"],
            cwd=outer,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    offline_installed = False
    if offline_install_smoke:
        if not checked["outer_package_manifest"]["complete_for_direct_v1"]:
            raise RuntimeError("offline install smoke requires complete wheel bundle")
        subprocess.run(
            ["python3", "-m", "venv", str(root / "dependency-smoke-venv")],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(root / "dependency-smoke-venv" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                str(outer / "wheels" / EXPECTED_NUMPY_WHEEL_FILENAME),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        offline_installed = True
    return {
        "schema": LOCAL_PREFLIGHT_SCHEMA,
        "status": "local_precontent_and_safe_extract_pass_cloud_not_authorized",
        "contract_sha256": canonical_sha256(checked),
        "copied_object_count": len(copied),
        "copied_objects_sha256": canonical_sha256(copied),
        "safe_extraction": extraction,
        "precontent_startup_returncode": completed.returncode,
        "offline_install_smoke_performed": offline_installed,
        "network_operation_performed": False,
        "cloud_write_performed": False,
        "claim_or_authorization_written": False,
        "cloud_executable": False,
        "launch_ready": False,
    }


def _load_canonical_json(path: str | Path, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ValueError(f"{label} must use exact canonical JSON bytes")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-stage1-contract")
    build.add_argument("--package-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--offline-wheel", type=Path)
    build.add_argument("--controller-public-key", type=Path)
    local = commands.add_parser("local-preflight")
    local.add_argument("--contract", type=Path, required=True)
    local.add_argument("--package-mirror", type=Path, required=True)
    local.add_argument("--fresh-work-root", type=Path, required=True)
    local.add_argument("--offline-wheel-mirror", type=Path)
    local.add_argument("--offline-install-smoke", action="store_true")
    authorized = commands.add_parser("authorized-worker")
    authorized.add_argument("--contract", type=Path, required=True)
    authorized.add_argument("--authorization", type=Path, required=True)
    authorized.add_argument("--claim", type=Path, required=True)
    authorized.add_argument("--controller-public-key", type=Path, required=True)
    authorized.add_argument("--fresh-root", type=Path, required=True)
    return parser


def _run_authorized_worker_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Cover pre-entry parsing/validation failures with one bounded shutdown."""

    if os.environ.get("OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER") != "1":
        raise RuntimeError("authorized-worker is not explicitly enabled")
    shutdown_requester = SubprocessShutdownRequester()
    try:
        contract = _load_canonical_json(args.contract, "10c2 contract")
        authorization = _load_canonical_json(
            args.authorization, "controller authorization"
        )
        claim = _load_canonical_json(args.claim, "worker claim")
        key_path = args.controller_public_key
        if (
            not key_path.is_file()
            or key_path.is_symlink()
            or key_path.stat().st_size < 512
            or key_path.stat().st_size > 16_384
        ):
            raise ValueError("controller public-key file identity changed")
        public_key_record = validate_rsa_public_key_record(
            _load_canonical_json(key_path, "controller public key")
        )
        verifier = RsaSha256ControllerTrustVerifier(public_key_record)
        trust = contract.get("authorization_contract", {})
        if (
            trust.get("controller_public_key_sha256")
            != verifier.public_key_sha256
            or trust.get("controller_key_id") != verifier.key_id
        ):
            raise ValueError("controller public key is not pinned by contract")
        return execute_authorized_worker(
            contract=contract,
            authorization=authorization,
            claim=claim,
            verifier=verifier,
            client=UrllibHttpClient(contract=contract),
            shutdown_requester=shutdown_requester,
            fresh_root=args.fresh_root,
        )
    except BaseException as worker_error:
        if not shutdown_requester.shutdown_attempted:
            try:
                shutdown_report = request_bounded_failure_shutdown(
                    requester=shutdown_requester,
                    deadline_seconds=MAX_FAILURE_SHUTDOWN_SECONDS,
                    done_published=False,
                )
            except BaseException as shutdown_error:
                raise BaseExceptionGroup(
                    "authorized-worker pre-entry failed and bounded safety "
                    "shutdown also failed",
                    [worker_error, shutdown_error],
                ) from None
            if hasattr(worker_error, "add_note"):
                worker_error.add_note(
                    "10c2 pre-entry bounded safety shutdown: "
                    + canonical_bytes(shutdown_report).decode("ascii")
                )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-stage1-contract":
        _package, _adapter, plan = _lazy_modules()
        wheel_record = (
            build_offline_wheel_record(args.offline_wheel)
            if args.offline_wheel is not None
            else None
        )
        public_key_record = (
            validate_rsa_public_key_record(
                _load_canonical_json(
                    args.controller_public_key, "controller public key"
                )
            )
            if args.controller_public_key is not None
            else None
        )
        value = build_job_contract(
            package_dir=args.package_dir,
            stage_id=plan.STAGE1_ID,
            job_id=plan.STAGE1_JOB_IDS[0],
            offline_wheel_record=wheel_record,
            controller_public_key_record=public_key_record,
        )
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("contract output must be fresh")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        return 0
    if args.command == "local-preflight":
        contract = _load_canonical_json(args.contract, "10c2 contract")
        value = local_preflight(
            contract=contract,
            package_mirror=args.package_mirror,
            fresh_work_root=args.fresh_work_root,
            offline_wheel_mirror=args.offline_wheel_mirror,
            offline_install_smoke=args.offline_install_smoke,
        )
    else:
        value = _run_authorized_worker_cli(args)
    print(json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


def _module_entrypoint() -> int:
    """Tell the bootstrap shell when this process already attempted shutdown."""

    try:
        return main()
    except BaseException as error:
        if not _SUBPROCESS_SHUTDOWN_ATTEMPTED:
            raise
        traceback.print_exception(error)
        return AUTHORIZED_WORKER_SHUTDOWN_ATTEMPTED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(_module_entrypoint())


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZED_WORKER_SHUTDOWN_ATTEMPTED_EXIT_CODE",
    "BOUNDED_SHUTDOWN_MARKER_PATH",
    "CLAIM_SCHEMA",
    "CONTRACT_SCHEMA",
    "DIRECT_STAGE_IDENTITY_SCHEMA",
    "EXPECTED_NUMPY_WHEEL_BYTES",
    "EXPECTED_NUMPY_WHEEL_FILENAME",
    "EXPECTED_NUMPY_WHEEL_SHA256",
    "REQUIRED_WORKER_OAUTH_SCOPE",
    "RSA_PUBLIC_KEY_SCHEMA",
    "RSA_SIGNATURE_ALGORITHM",
    "RsaSha256ControllerTrustVerifier",
    "HttpClient",
    "HttpResponse",
    "RuntimeApproval",
    "UrllibHttpClient",
    "build_direct_stage_identity",
    "build_direct_v1_remote_layout",
    "build_job_contract",
    "build_lifecycle_plan",
    "build_offline_wheel_record",
    "build_rsa_public_key_record",
    "build_outer_package_inventory",
    "build_outer_package_manifest",
    "conditional_create_and_readback",
    "deterministic_instance_name",
    "download_outer_package",
    "execute_authorized_worker",
    "generation_pinned_download",
    "local_preflight",
    "publish_validated_runner_output",
    "read_metadata_identity_and_token",
    "request_bounded_failure_shutdown",
    "request_compute_self_delete",
    "safe_extract_worker_source",
    "validate_controller_approval",
    "validate_job_contract",
    "validate_outer_package_manifest",
    "validate_rsa_public_key_record",
]
