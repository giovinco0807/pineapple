"""Strict injectable GCE Phase-B adapter for one full100 wave-v2 bundle.

The module has no live-cloud entry point.  A caller must inject an HTTP
requester and a launch-bundle validator closure that revalidates every upstream
evidence object.  The four modes (create, read-status, delete, absence) expose
disjoint HTTP method sets.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.parse
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


GCE_CREATE_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gce_create_receipt_v2"
GCE_STATUS_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gce_status_receipt_v2"
GCE_DELETE_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gce_delete_receipt_v2"
GCE_DELETE_RECONCILE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_gce_delete_reconcile_receipt_v2"
)
GCE_ABSENCE_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gce_absence_receipt_v2"

PROJECT = "ofc-solver-485418"
REGION = "asia-northeast1"
ZONE = "asia-northeast1-b"
MACHINE_TYPE = "c4-standard-16"
BOOT_DISK_TYPE = "hyperdisk-balanced"
BOOT_DISK_INTERFACE = "NVME"
BOOT_DISK_SIZE_GB = 20
NETWORK = "default"
SUBNETWORK = "default"
NIC_TYPE = "GVNIC"
OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VM_TTL_SECONDS = 4_500
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
OPERATION_POLL_ATTEMPTS = 60
OPERATION_POLL_INTERVAL_SECONDS = 2
INSTANCE_READBACK_ATTEMPTS = 20
INSTANCE_READBACK_INTERVAL_SECONDS = 1
ABSENCE_POLL_ATTEMPTS = 12
ABSENCE_POLL_INTERVAL_SECONDS = 5

_MODES = frozenset(
    {
        "create", "reconcile-create", "read-status", "delete",
        "reconcile-delete", "absence",
    }
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_OPERATION = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_PROVIDER_ID = re.compile(r"^[1-9][0-9]*$")
_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}"
    r"\.iam\.gserviceaccount\.com$"
)
_IMAGE_LINK = re.compile(
    r"^https://www\.googleapis\.com/compute/v1/projects/"
    r"debian-cloud/global/images/[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)
_UTC_SECONDS = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]
BundleValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
Sleeper = Callable[[float], None]

_CREATE_ROW_KEYS = frozenset(
    {
        "job_id", "source_role", "attempt_id", "instance_name",
        "service_account", "provider_instance_id", "provider_boot_disk_id",
        "request_id", "operation_name", "operation_id", "spec_sha256",
        "bootstrap_sha256", "labels", "observed_status",
        "recovered_after_insert_failure",
    }
)
_CREATE_KEYS = frozenset(
    {
        "schema", "status", "bundle_sha256", "run_name",
        "execution_identity_sha256", "project", "zone", "wave_index",
        "active_image_self_link", "active_image_identity_sha256",
        "startup_sha256", "expected_instance_count", "created_instance_count",
        "rows", "create_complete", "one_shot_consumed",
        "existing_same_name_accepted", "request_id_reuse_authorized",
        "unlisted_create_authorized", "additional_create_authorized",
        "observed_at_utc", "current_profile_changed", "receipt_sha256",
    }
)
_STATUS_ROW_KEYS = frozenset(
    {
        "instance_name", "provider_instance_id", "provider_boot_disk_id",
        "status", "spec_sha256",
    }
)
_STATUS_KEYS = frozenset(
    {
        "schema", "status", "bundle_sha256", "create_receipt_sha256",
        "project", "zone", "wave_index", "rows", "instance_count",
        "all_specs_exact", "read_only", "observed_at_utc",
        "current_profile_changed", "receipt_sha256",
    }
)
_DELETE_ROW_KEYS = frozenset(
    {
        "instance_name", "provider_instance_id", "provider_boot_disk_id",
        "request_id", "operation_name", "operation_id", "already_absent",
        "recovered_after_transport_ambiguity",
    }
)
_ORPHAN_DISK_DELETE_ROW_KEYS = frozenset(
    {
        "instance_name", "provider_boot_disk_id", "request_id",
        "operation_name", "operation_id", "already_absent",
        "recovered_after_transport_ambiguity",
    }
)
_DELETE_KEYS = frozenset(
    {
        "schema", "status", "bundle_sha256", "create_receipt_sha256",
        "project", "zone", "wave_index", "owned_instance_count",
        "selected_instance_count", "scanned_selected_names",
        "all_selected_names_scanned", "delete_operation_count",
        "already_absent_count", "rows", "orphan_boot_disk_count",
        "orphan_boot_disk_rows",
        "exact_provider_ids_verified", "exact_labels_verified",
        "wildcard_delete_used", "request_id_reuse_authorized",
        "additional_create_authorized", "observed_at_utc",
        "current_profile_changed", "receipt_sha256",
    }
)
_ABSENCE_KEYS = frozenset(
    {
        "schema", "status", "bundle_sha256", "create_receipt_sha256",
        "delete_receipt_sha256", "project", "zone", "wave_index",
        "checked_instance_count", "absent_instance_names",
        "absent_boot_disk_names", "all_instances_absent",
        "all_boot_disks_absent", "read_only", "observed_at_utc",
        "additional_create_authorized", "current_profile_changed",
        "receipt_sha256",
    }
)
_DELETE_RECONCILE_ROW_KEYS = frozenset(
    {
        "instance_name", "instance_absent", "boot_disk_absent",
        "exact_orphan_boot_disk_owned", "provider_boot_disk_id",
    }
)
_DELETE_RECONCILE_KEYS = frozenset(
    {
        "schema", "status", "bundle_sha256", "create_receipt_sha256",
        "project", "zone", "wave_index", "rows", "selected_instance_count",
        "all_instances_absent", "all_boot_disks_absent",
        "orphan_boot_disk_count", "orphan_cleanup_required",
        "recovered_delete_receipt", "read_only", "observed_at_utc",
        "additional_create_authorized", "current_profile_changed",
        "receipt_sha256",
    }
)


class GceCreateIncompleteError(RuntimeError):
    def __init__(
        self, message: str, *, partial_receipt: Mapping[str, Any] | None
    ) -> None:
        super().__init__(message)
        self.partial_receipt = (
            None if partial_receipt is None else deepcopy(dict(partial_receipt))
        )


class GcePhaseBTransportError(RuntimeError):
    """The provider may have committed a GCE mutation without a response."""


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


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _require_provider_id(value: Any, label: str) -> str:
    rendered = str(value)
    if isinstance(value, bool) or _PROVIDER_ID.fullmatch(rendered) is None:
        raise RuntimeError(f"{label} provider id changed")
    return rendered


def _require_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not UTC")
    return value


def _utc_timestamp(value: Any, label: str) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not UTC")
    return parsed.timestamp()


def _require_request_id(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUIDv4")
    return value


def _seal(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(core))
    if "receipt_sha256" in payload:
        raise ValueError("GCE receipt was already sealed")
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _pop_receipt(
    value: Mapping[str, Any], *, keys: frozenset[str], label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, keys, label)
    digest = payload.pop("receipt_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{label} digest changed")
    return payload, _require_sha(digest, f"{label} digest")


def _validate_bundle_via_callback(
    value: Mapping[str, Any], validator: BundleValidator
) -> dict[str, Any]:
    if not callable(validator):
        raise PermissionError(
            "GCE Phase B requires an upstream-evidence launch-bundle validator"
        )
    if not isinstance(value, Mapping):
        raise ValueError("launch bundle must be an object")
    validated = validator(deepcopy(dict(value)))
    if not isinstance(validated, Mapping):
        raise ValueError("launch-bundle validator returned a non-object")
    bundle = deepcopy(dict(validated))
    if bundle != dict(value):
        raise ValueError("launch-bundle validator did not validate the supplied bundle")
    if (
        bundle.get("schema") != bundle_v2.LAUNCH_BUNDLE_SCHEMA
        or bundle.get("status") != bundle_v2.LAUNCH_BUNDLE_STATUS
        or bundle.get("project_id") != PROJECT
        or bundle.get("zone") != ZONE
        or bundle.get("cloud_create_authorized") is not True
        or bundle.get("exact_selected_create_authorized") is not True
        or bundle.get("one_shot") is not True
        or bundle.get("reuse_authorized") is not False
        or bundle.get("additional_create_authorized") is not False
        or bundle.get("unlisted_instance_create_authorized") is not False
        or bundle.get("cloud_started") is not False
    ):
        raise PermissionError("validated launch bundle is not an unused one-shot create right")
    digest = bundle.get("bundle_sha256")
    if digest != bundle_v2.canonical_sha256(
        {key: val for key, val in bundle.items() if key != "bundle_sha256"}
    ):
        raise ValueError("launch bundle digest changed after upstream validation")
    inventory = bundle.get("bootstrap_inventory")
    count = bundle.get("selected_vm_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= 8
        or not isinstance(inventory, list)
        or len(inventory) != count
    ):
        raise ValueError("validated launch bundle inventory changed")
    return bundle


def _labels(bundle: Mapping[str, Any], row: Mapping[str, Any], image_sha: str) -> dict[str, str]:
    return {
        "ofc-owner": "hu-m31-t3-f100wv2",
        "ofc-run": str(bundle["execution_identity_sha256"])[:32],
        "ofc-wave": f"w{int(bundle['wave_index']):02d}",
        "ofc-bundle": str(bundle["bundle_sha256"])[:32],
        "ofc-job": hashlib.sha256(str(row["job_id"]).encode("ascii")).hexdigest()[:32],
        "ofc-role": str(row["source_role"]),
        "ofc-image": image_sha[:32],
    }


def _metadata(
    row: Mapping[str, Any], *, startup_text: str
) -> dict[str, str]:
    bootstrap = row.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("launch inventory bootstrap is missing")
    bootstrap_payload = deepcopy(dict(bootstrap))
    digest = bootstrap_payload.pop("bootstrap_sha256", None)
    if digest != package_v2.canonical_sha256(bootstrap_payload):
        raise ValueError("launch inventory bootstrap digest changed")
    if row.get("bootstrap_sha256") != digest:
        raise ValueError("launch inventory bootstrap binding changed")
    encoded = base64.b64encode(package_v2.canonical_bytes(bootstrap)).decode("ascii")
    return {"job-bootstrap-b64": encoded, "startup-script": startup_text}


def _instance_body(
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    image_self_link: str,
    image_identity_sha256: str,
    startup_text: str,
) -> dict[str, Any]:
    name = row.get("instance_id")
    account = row.get("service_account")
    if (
        not isinstance(name, str)
        or _INSTANCE.fullmatch(name) is None
        or not isinstance(account, str)
        or _SERVICE_ACCOUNT.fullmatch(account) is None
        or row.get("source_role") not in {"candidate", "reference"}
    ):
        raise ValueError("launch inventory VM identity changed")
    metadata = _metadata(row, startup_text=startup_text)
    return {
        "name": name,
        "machineType": (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
            f"{ZONE}/machineTypes/{MACHINE_TYPE}"
        ),
        "deletionProtection": False,
        "canIpForward": False,
        "labels": _labels(bundle, row, image_identity_sha256),
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "maxRunDuration": {"seconds": str(VM_TTL_SECONDS), "nanos": 0},
        },
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "type": "PERSISTENT",
                "interface": BOOT_DISK_INTERFACE,
                "deviceName": name,
                "initializeParams": {
                    "sourceImage": image_self_link,
                    "diskSizeGb": str(BOOT_DISK_SIZE_GB),
                    "diskName": name,
                    "labels": _labels(bundle, row, image_identity_sha256),
                    "diskType": (
                        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                        f"zones/{ZONE}/diskTypes/{BOOT_DISK_TYPE}"
                    ),
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": (
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"global/networks/{NETWORK}"
                ),
                "subnetwork": (
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"regions/{REGION}/subnetworks/{SUBNETWORK}"
                ),
                "nicType": NIC_TYPE,
                "accessConfigs": [],
            }
        ],
        "serviceAccounts": [{"email": account, "scopes": [OAUTH_SCOPE]}],
        "metadata": {
            "items": [
                {"key": key, "value": value}
                for key, value in sorted(metadata.items())
            ]
        },
    }


def _metadata_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError("GCE instance metadata is missing")
    items = value.get("items")
    if not isinstance(items, list) or len(items) != 2:
        raise RuntimeError("GCE instance metadata cardinality changed")
    result: dict[str, str] = {}
    for raw in items:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"key", "value"}
            or not isinstance(raw.get("key"), str)
            or not isinstance(raw.get("value"), str)
            or raw["key"] in result
        ):
            raise RuntimeError("GCE instance metadata changed")
        result[raw["key"]] = raw["value"]
    if set(result) != {"job-bootstrap-b64", "startup-script"}:
        raise RuntimeError("GCE instance metadata keys changed")
    return result


def _request_ids(
    value: Mapping[str, Any], *, names: Sequence[str], forbidden: set[str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("GCE requestId mapping does not exactly cover inventory")
    result = {
        name: _require_request_id(value[name], f"GCE requestId for {name}")
        for name in names
    }
    if len(set(result.values())) != len(result) or set(result.values()) & forbidden:
        raise ValueError("GCE requestId is duplicated or reused")
    return result


class GceWavePhaseBAdapter:
    """One-shot mode-separated GCE REST boundary."""

    def __init__(
        self,
        *,
        mode: str,
        launch_bundle: Mapping[str, Any],
        launch_bundle_validator: BundleValidator,
        active_image_self_link: str,
        active_image_identity_sha256: str,
        expected_image_digest: str,
        startup_script_bytes: bytes,
        expected_startup_sha256: str = bundle_v2.EXPECTED_STARTUP_SHA256,
        requester: HttpRequester,
        create_receipt: Mapping[str, Any] | None = None,
        delete_receipt: Mapping[str, Any] | None = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if mode not in _MODES:
            raise ValueError("GCE Phase B adapter mode is invalid")
        self.mode = mode
        self._bundle = _validate_bundle_via_callback(
            launch_bundle, launch_bundle_validator
        )
        if (
            not isinstance(active_image_self_link, str)
            or _IMAGE_LINK.fullmatch(active_image_self_link) is None
        ):
            raise ValueError("active image selfLink escaped the pinned Debian image scope")
        self._image_link = active_image_self_link
        self._image_sha = _require_sha(
            active_image_identity_sha256, "active image identity"
        )
        if expected_image_digest != f"sha256:{self._image_sha}":
            raise ValueError("active image identity does not match expected wave digest")
        if not isinstance(startup_script_bytes, bytes) or not startup_script_bytes:
            raise ValueError("startup script bytes are missing")
        startup_sha = hashlib.sha256(startup_script_bytes).hexdigest()
        expected_startup = _require_sha(
            expected_startup_sha256, "expected startup script"
        )
        if startup_sha != expected_startup:
            raise ValueError("startup script bytes do not match the validated frozen hash")
        try:
            self._startup_text = startup_script_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("startup script is not exact UTF-8") from exc
        if self._startup_text.encode("utf-8") != startup_script_bytes or "\x00" in self._startup_text:
            raise ValueError("startup script bytes changed during UTF-8 decoding")
        self._startup_sha = startup_sha
        if not callable(requester):
            raise ValueError("GCE Phase B requires an injected HTTP requester")
        if not callable(sleeper):
            raise ValueError("GCE Phase B sleeper is invalid")
        self._requester = requester
        self._sleep = sleeper
        self._used = False
        self._http_count = 0

        inventory = self._bundle["bootstrap_inventory"]
        selected_names = self._bundle.get("selected_instance_ids")
        selected_jobs = self._bundle.get("selected_job_ids")
        selected_roles = self._bundle.get("selected_source_roles")
        selected_attempts = self._bundle.get("selected_attempt_ids")
        selected_accounts = self._bundle.get("selected_service_accounts")
        if any(
            not isinstance(value, list)
            for value in (
                selected_names, selected_jobs, selected_roles,
                selected_attempts, selected_accounts,
            )
        ):
            raise ValueError("launch bundle selected inventory vectors are missing")
        expected_vectors = {
            "instance_id": selected_names,
            "job_id": selected_jobs,
            "source_role": selected_roles,
            "attempt_id": selected_attempts,
            "service_account": selected_accounts,
        }
        for key, expected in expected_vectors.items():
            if [row.get(key) for row in inventory] != expected:
                raise ValueError("launch bundle inventory vectors changed")
        # A wave intentionally reuses the attempt ordinal (for example ``a00``)
        # across different jobs and has many rows for each source role.  Those
        # are attributes of a selected job, not globally unique identities.
        # Keep the actual one-VM/one-job/one-account boundary strict and also
        # reject duplicate composite rows without rejecting a normal wave.
        unique_vectors = (selected_names, selected_jobs, selected_accounts)
        job_attempts = [
            (row.get("job_id"), row.get("attempt_id")) for row in inventory
        ]
        selected_identities = [
            (
                row.get("instance_id"),
                row.get("job_id"),
                row.get("source_role"),
                row.get("attempt_id"),
                row.get("service_account"),
            )
            for row in inventory
        ]
        if (
            any(len(set(values)) != len(values) for values in unique_vectors)
            or len(set(job_attempts)) != len(job_attempts)
            or len(set(selected_identities)) != len(selected_identities)
        ):
            raise ValueError("launch bundle selected inventory is duplicated")
        self._specs: list[dict[str, Any]] = []
        for row in inventory:
            body = _instance_body(
                bundle=self._bundle,
                row=row,
                image_self_link=self._image_link,
                image_identity_sha256=self._image_sha,
                startup_text=self._startup_text,
            )
            bootstrap = row["bootstrap"]
            if (
                bootstrap.get("job_id") != row["job_id"]
                or bootstrap.get("source_role") != row["source_role"]
                or bootstrap.get("attempt_id") != row["attempt_id"]
                or bootstrap.get("instance_name") != row["instance_id"]
                or bootstrap.get("worker_principal") != row["service_account"]
            ):
                raise ValueError("launch inventory bootstrap escaped selected identity")
            self._specs.append(
                {
                    "inventory": deepcopy(dict(row)),
                    "body": body,
                    "spec_sha256": canonical_sha256(body),
                }
            )
        self._spec_by_name = {
            spec["body"]["name"]: spec for spec in self._specs
        }

        self._create_receipt: dict[str, Any] | None = None
        self._delete_receipt: dict[str, Any] | None = None
        if mode == "create":
            if create_receipt is not None or delete_receipt is not None:
                raise PermissionError("create mode cannot accept a consumed create receipt")
        elif mode == "reconcile-create":
            if delete_receipt is not None:
                raise PermissionError("reconcile-create cannot accept delete evidence")
            if create_receipt is not None:
                self._create_receipt = self.validate_create_receipt(create_receipt)
        else:
            if create_receipt is None:
                raise PermissionError(f"{mode} mode requires an exact create receipt")
            self._create_receipt = self.validate_create_receipt(create_receipt)
            if mode == "absence":
                if delete_receipt is None:
                    raise PermissionError("absence mode requires an exact delete receipt")
                self._delete_receipt = self.validate_delete_receipt(delete_receipt)
            elif delete_receipt is not None:
                raise PermissionError(f"{mode} mode cannot accept a delete receipt")

    @property
    def expected_instance_bodies(self) -> list[dict[str, Any]]:
        return [deepcopy(spec["body"]) for spec in self._specs]

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        token = os.environ.get(TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            character.isspace() for character in token
        ):
            raise PermissionError(f"Bearer token must be supplied only through {TOKEN_ENV}")
        result = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if content_type is not None:
            result["Content-Type"] = content_type
        return result

    def _http(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        allowed_statuses: Sequence[int] = (200,),
        timeout_seconds: int = 60,
    ) -> HttpResponse:
        allowed = {
            "create": {"GET", "POST"},
            "reconcile-create": {"GET"},
            "read-status": {"GET"},
            "delete": {"GET", "DELETE"},
            "reconcile-delete": {"GET"},
            "absence": {"GET"},
        }[self.mode]
        if method not in allowed:
            raise PermissionError("HTTP method escaped the fixed GCE Phase B mode")
        fixed_prefix = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
        )
        if not url.startswith(fixed_prefix):
            raise PermissionError("GCE request escaped the fixed project and zone")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 600
        ):
            raise ValueError("GCE request timeout escaped the fixed bound")
        # Header/token validation is a local precondition.  It must complete
        # before entering the transport ambiguity boundary; otherwise a
        # missing token could be mistaken for a request whose response was
        # lost after the provider committed the mutation.
        headers = self._headers(content_type=content_type)
        self._http_count += 1
        try:
            response = self._requester(
                method,
                url,
                headers,
                body,
                timeout_seconds,
            )
        except Exception as exc:
            raise GcePhaseBTransportError(
                f"GCE {method} transport failed without a provider response"
            ) from exc
        if not isinstance(response, HttpResponse):
            raise TypeError("GCE requester returned a non-HttpResponse")
        if response.status in {409, 412}:
            raise RuntimeError(
                f"GCE {method} rejected collision/precondition status {response.status}"
            )
        if response.status not in allowed_statuses:
            raise RuntimeError(f"GCE {method} failed with status {response.status}")
        return response

    @staticmethod
    def _json(response: HttpResponse, label: str) -> dict[str, Any]:
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} response was not JSON") from exc
        if not isinstance(value, dict) or value.get("nextPageToken") is not None:
            raise RuntimeError(f"{label} response shape changed")
        return value

    @staticmethod
    def _instance_url(name: str) -> str:
        return (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
            f"{ZONE}/instances/{urllib.parse.quote(name, safe='')}"
        )

    @staticmethod
    def _disk_url(name: str) -> str:
        return (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
            f"{ZONE}/disks/{urllib.parse.quote(name, safe='')}"
        )

    def _operation(
        self,
        initial: Mapping[str, Any],
        *,
        operation_type: str,
        instance_name: str,
        resource_collection: str = "instances",
    ) -> dict[str, Any]:
        if resource_collection not in {"instances", "disks"}:
            raise ValueError("GCE operation resource collection changed")
        expected_target = (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
            f"{resource_collection}/{instance_name}"
        )

        def checked(value: Mapping[str, Any], *, final: bool) -> dict[str, Any]:
            if not isinstance(value, Mapping):
                raise RuntimeError("GCE operation is not an object")
            result = deepcopy(dict(value))
            name = result.get("name")
            status = result.get("status")
            if (
                not isinstance(name, str)
                or _OPERATION.fullmatch(name) is None
                or status not in {"PENDING", "RUNNING", "DONE"}
                or result.get("operationType") != operation_type
                or result.get("targetLink") != expected_target
                or result.get("error") is not None
                or result.get("httpErrorStatusCode") is not None
            ):
                raise RuntimeError("GCE operation identity changed or failed")
            _require_provider_id(result.get("id"), "GCE operation")
            if final:
                if status != "DONE":
                    raise RuntimeError("GCE operation is not complete")
                _require_provider_id(result.get("targetId"), "GCE operation target")
            return result

        current = checked(initial, final=initial.get("status") == "DONE")
        operation_name = current["name"]
        operation_id = _require_provider_id(current["id"], "GCE operation")
        if current["status"] != "DONE":
            url = (
                f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
                f"{ZONE}/operations/{urllib.parse.quote(operation_name, safe='')}"
            )
            for attempt in range(OPERATION_POLL_ATTEMPTS):
                current = checked(
                    self._json(self._http(method="GET", url=url), "GCE operation poll"),
                    final=False,
                )
                if current["name"] != operation_name or str(current["id"]) != operation_id:
                    raise RuntimeError("GCE operation changed while polling")
                if current["status"] == "DONE":
                    current = checked(current, final=True)
                    break
                if attempt + 1 < OPERATION_POLL_ATTEMPTS:
                    self._sleep(OPERATION_POLL_INTERVAL_SECONDS)
            else:
                raise TimeoutError("GCE operation did not complete within fixed bound")
        return current

    def _read_exact_disk(
        self,
        spec: Mapping[str, Any],
        *,
        expected_provider_id: str | None = None,
        expected_attached: bool,
        allow_absent: bool,
    ) -> dict[str, Any] | None:
        body = spec["body"]
        name = body["name"]
        response = self._http(
            method="GET",
            url=self._disk_url(name),
            allowed_statuses=(200, 404),
        )
        if response.status == 404:
            if allow_absent:
                return None
            raise RuntimeError("expected exact owned GCE boot disk is absent")
        disk = self._json(response, "GCE boot disk readback")
        disk_id = _require_provider_id(disk.get("id"), "GCE boot disk")
        if expected_provider_id is not None and disk_id != expected_provider_id:
            raise RuntimeError("GCE boot disk provider id changed")
        disk_self_link = (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
            f"disks/{name}"
        )
        instance_self_link = (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
            f"instances/{name}"
        )
        initialize = body["disks"][0]["initializeParams"]
        expected_users = [instance_self_link] if expected_attached else []
        if (
            disk.get("name") != name
            or disk.get("selfLink") != disk_self_link
            or disk.get("type") != initialize["diskType"]
            or str(disk.get("sizeGb")) != str(BOOT_DISK_SIZE_GB)
            or disk.get("sourceImage") != self._image_link
            or disk.get("labels") != initialize["labels"]
            or disk.get("status") not in {"CREATING", "RESTORING", "READY"}
            or disk.get("users", []) != expected_users
        ):
            raise RuntimeError("GCE boot disk/image/ownership specification drifted")
        return {"provider_boot_disk_id": disk_id}

    def _read_exact_instance(
        self,
        spec: Mapping[str, Any],
        *,
        expected_provider_id: str | None = None,
        allow_absent: bool,
    ) -> dict[str, Any] | None:
        body = spec["body"]
        name = body["name"]
        response = self._http(
            method="GET",
            url=self._instance_url(name),
            allowed_statuses=(200, 404),
        )
        if response.status == 404:
            if allow_absent:
                return None
            raise RuntimeError("expected exact owned GCE instance is absent")
        instance = self._json(response, "GCE instance readback")
        provider_id = _require_provider_id(instance.get("id"), "GCE instance")
        if expected_provider_id is not None and provider_id != expected_provider_id:
            raise RuntimeError("GCE instance provider id changed")
        expected_self_link = (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
            f"instances/{name}"
        )
        scheduling = instance.get("scheduling")
        disks = instance.get("disks")
        networks = instance.get("networkInterfaces")
        accounts = instance.get("serviceAccounts")
        status = instance.get("status")
        allowed_statuses = {
            "PROVISIONING", "STAGING", "RUNNING", "STOPPING", "SUSPENDING",
            "SUSPENDED", "REPAIRING", "TERMINATED",
        }
        if (
            instance.get("name") != name
            or instance.get("selfLink") != expected_self_link
            or instance.get("machineType") != body["machineType"]
            or instance.get("deletionProtection", False) is not False
            or instance.get("canIpForward", False) is not False
            or instance.get("labels") != body["labels"]
            or status not in allowed_statuses
            or not isinstance(scheduling, Mapping)
            or scheduling.get("provisioningModel") != "SPOT"
            or scheduling.get("instanceTerminationAction") != "DELETE"
            or scheduling.get("automaticRestart") is not False
            or scheduling.get("onHostMaintenance") != "TERMINATE"
            or not isinstance(scheduling.get("maxRunDuration"), Mapping)
            or scheduling["maxRunDuration"].get("seconds") != str(VM_TTL_SECONDS)
            or scheduling["maxRunDuration"].get("nanos", 0) != 0
            or not isinstance(disks, list)
            or len(disks) != 1
            or not isinstance(networks, list)
            or len(networks) != 1
            or not isinstance(accounts, list)
            or len(accounts) != 1
        ):
            raise RuntimeError("GCE instance specification drifted")
        disk_ref = disks[0]
        disk_self_link = (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
            f"disks/{name}"
        )
        if (
            not isinstance(disk_ref, Mapping)
            or disk_ref.get("boot") is not True
            or disk_ref.get("autoDelete") is not True
            or disk_ref.get("interface") != BOOT_DISK_INTERFACE
            or disk_ref.get("deviceName") != name
            or disk_ref.get("source") != disk_self_link
        ):
            raise RuntimeError("GCE boot disk attachment drifted")
        network = networks[0]
        if (
            not isinstance(network, Mapping)
            or network.get("network") != body["networkInterfaces"][0]["network"]
            or network.get("subnetwork")
            != body["networkInterfaces"][0]["subnetwork"]
            or network.get("nicType") != NIC_TYPE
            or network.get("accessConfigs", []) != []
        ):
            raise RuntimeError("GCE private network specification drifted")
        account = accounts[0]
        if (
            not isinstance(account, Mapping)
            or account.get("email") != body["serviceAccounts"][0]["email"]
            or account.get("scopes") != [OAUTH_SCOPE]
        ):
            raise RuntimeError("GCE worker service account drifted")
        expected_metadata = {
            item["key"]: item["value"] for item in body["metadata"]["items"]
        }
        if _metadata_map(instance.get("metadata")) != expected_metadata:
            raise RuntimeError("GCE bootstrap/startup metadata drifted")

        exact_disk = self._read_exact_disk(
            spec,
            expected_attached=True,
            allow_absent=False,
        )
        assert exact_disk is not None
        return {
            "provider_instance_id": provider_id,
            "provider_boot_disk_id": exact_disk["provider_boot_disk_id"],
            "status": status,
        }

    def _read_absence(self, spec: Mapping[str, Any]) -> tuple[bool, bool]:
        name = spec["body"]["name"]
        instance = self._http(
            method="GET",
            url=self._instance_url(name),
            allowed_statuses=(200, 404),
        )
        disk = self._http(
            method="GET",
            url=self._disk_url(name),
            allowed_statuses=(200, 404),
        )
        return instance.status == 404, disk.status == 404

    def _confirm_created_instance(
        self, spec: Mapping[str, Any]
    ) -> dict[str, Any]:
        last_error: RuntimeError | None = None
        for attempt in range(INSTANCE_READBACK_ATTEMPTS):
            try:
                observed = self._read_exact_instance(spec, allow_absent=True)
            except RuntimeError as exc:
                last_error = exc
                observed = None
            if observed is not None:
                return observed
            if attempt + 1 < INSTANCE_READBACK_ATTEMPTS:
                self._sleep(INSTANCE_READBACK_INTERVAL_SECONDS)
        if last_error is not None:
            raise last_error
        raise TimeoutError("GCE inserted instance/disk readback did not become visible")

    def _create_core(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        observed_at_utc: str,
    ) -> dict[str, Any]:
        checked_rows = self._validate_create_rows(rows)
        complete = len(checked_rows) == len(self._specs) and not any(
            row["recovered_after_insert_failure"] for row in checked_rows
        )
        observed = _require_utc(observed_at_utc, "GCE create time")
        if _utc_timestamp(observed, "GCE create time") < _utc_timestamp(
            self._bundle.get("current_time_utc"), "launch bundle current time"
        ):
            raise ValueError("GCE create receipt predates launch-bundle validation")
        core = {
            "schema": GCE_CREATE_RECEIPT_SCHEMA,
            "status": (
                "exact_selected_gce_create_complete"
                if complete
                else "partial_exact_owned_gce_create"
            ),
            "bundle_sha256": self._bundle["bundle_sha256"],
            "run_name": self._bundle["run_name"],
            "execution_identity_sha256": self._bundle["execution_identity_sha256"],
            "project": PROJECT,
            "zone": ZONE,
            "wave_index": self._bundle["wave_index"],
            "active_image_self_link": self._image_link,
            "active_image_identity_sha256": self._image_sha,
            "startup_sha256": self._startup_sha,
            "expected_instance_count": len(self._specs),
            "created_instance_count": len(checked_rows),
            "rows": checked_rows,
            "create_complete": complete,
            "one_shot_consumed": True,
            "existing_same_name_accepted": False,
            "request_id_reuse_authorized": False,
            "unlisted_create_authorized": False,
            "additional_create_authorized": False,
            "observed_at_utc": observed,
            "current_profile_changed": False,
        }
        return core

    def _validate_create_rows(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("GCE create rows are missing")
        result: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        seen_provider_ids: set[str] = set()
        seen_requests: set[str] = set()
        order = {spec["body"]["name"]: i for i, spec in enumerate(self._specs)}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("GCE create row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(row, _CREATE_ROW_KEYS, "GCE create row")
            name = row.get("instance_name")
            spec = self._spec_by_name.get(name)
            inventory = None if spec is None else spec["inventory"]
            if (
                spec is None
                or name in seen_names
                or row.get("job_id") != inventory["job_id"]
                or row.get("source_role") != inventory["source_role"]
                or row.get("attempt_id") != inventory["attempt_id"]
                or row.get("service_account") != inventory["service_account"]
                or row.get("spec_sha256") != spec["spec_sha256"]
                or row.get("bootstrap_sha256") != inventory["bootstrap_sha256"]
                or row.get("labels") != spec["body"]["labels"]
                or row.get("observed_status")
                not in {"PROVISIONING", "STAGING", "RUNNING", "STOPPING", "TERMINATED"}
            ):
                raise ValueError("GCE create row escaped exact selected inventory")
            instance_id = _require_provider_id(
                row.get("provider_instance_id"), "GCE create instance"
            )
            disk_id = _require_provider_id(
                row.get("provider_boot_disk_id"), "GCE create boot disk"
            )
            request_id = _require_request_id(row.get("request_id"), "GCE create requestId")
            recovered = row.get("recovered_after_insert_failure")
            if recovered is True:
                if row.get("operation_name") is not None or row.get("operation_id") is not None:
                    raise ValueError("recovered GCE create row cannot trust an operation")
            elif recovered is False:
                _require_provider_id(row.get("operation_id"), "GCE create operation")
                if (
                    not isinstance(row.get("operation_name"), str)
                    or _OPERATION.fullmatch(row["operation_name"]) is None
                ):
                    raise ValueError("GCE create operation name changed")
            else:
                raise ValueError("GCE create recovery flag changed")
            if (
                instance_id in seen_provider_ids
                or disk_id in seen_provider_ids
                or request_id in seen_requests
            ):
                raise ValueError("GCE create row provider identity is duplicated")
            seen_names.add(name)
            seen_provider_ids.update({instance_id, disk_id})
            seen_requests.add(request_id)
            row["provider_instance_id"] = instance_id
            row["provider_boot_disk_id"] = disk_id
            if row["operation_id"] is not None:
                row["operation_id"] = str(row["operation_id"])
            result.append(row)
        if [order[row["instance_name"]] for row in result] != sorted(
            order[row["instance_name"]] for row in result
        ):
            raise ValueError("GCE create row order changed")
        return result

    def validate_create_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload, digest = _pop_receipt(
            value, keys=_CREATE_KEYS, label="GCE create receipt"
        )
        rows = self._validate_create_rows(payload.get("rows"))
        complete = len(rows) == len(self._specs) and not any(
            row["recovered_after_insert_failure"] for row in rows
        )
        if (
            payload.get("schema") != GCE_CREATE_RECEIPT_SCHEMA
            or payload.get("status")
            != ("exact_selected_gce_create_complete" if complete else "partial_exact_owned_gce_create")
            or payload.get("bundle_sha256") != self._bundle["bundle_sha256"]
            or payload.get("run_name") != self._bundle["run_name"]
            or payload.get("execution_identity_sha256")
            != self._bundle["execution_identity_sha256"]
            or payload.get("project") != PROJECT
            or payload.get("zone") != ZONE
            or payload.get("wave_index") != self._bundle["wave_index"]
            or payload.get("active_image_self_link") != self._image_link
            or payload.get("active_image_identity_sha256") != self._image_sha
            or payload.get("startup_sha256") != self._startup_sha
            or payload.get("expected_instance_count") != len(self._specs)
            or payload.get("created_instance_count") != len(rows)
            or payload.get("create_complete") is not complete
            or payload.get("one_shot_consumed") is not True
            or payload.get("existing_same_name_accepted") is not False
            or payload.get("request_id_reuse_authorized") is not False
            or payload.get("unlisted_create_authorized") is not False
            or payload.get("additional_create_authorized") is not False
            or payload.get("current_profile_changed") is not False
        ):
            raise ValueError("GCE create receipt contract changed")
        _require_utc(payload.get("observed_at_utc"), "GCE create time")
        if _utc_timestamp(payload["observed_at_utc"], "GCE create time") < _utc_timestamp(
            self._bundle.get("current_time_utc"), "launch bundle current time"
        ):
            raise ValueError("GCE create receipt predates launch-bundle validation")
        payload["rows"] = rows
        return {**payload, "receipt_sha256": digest}

    def create_selected(
        self,
        *,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
        prior_create_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.mode != "create" or self._used:
            raise PermissionError("GCE create mode is unavailable or already consumed")
        if prior_create_receipt is not None:
            raise PermissionError("GCE create is forbidden after any create receipt")
        self._used = True
        names = [spec["body"]["name"] for spec in self._specs]
        requests = _request_ids(request_ids, names=names, forbidden=set())
        for spec in self._specs:
            instance_absent, disk_absent = self._read_absence(spec)
            if not instance_absent or not disk_absent:
                raise FileExistsError(
                    "selected GCE instance or boot disk already exists; same name is never adopted"
                )
        created: list[dict[str, Any]] = []
        try:
            for spec in self._specs:
                body = spec["body"]
                name = body["name"]
                url = (
                    f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
                    f"{ZONE}/instances?requestId={urllib.parse.quote(requests[name], safe='')}"
                )
                response = self._http(
                    method="POST",
                    url=url,
                    body=canonical_bytes(body),
                    content_type="application/json",
                )
                operation = self._operation(
                    self._json(response, "GCE insert"),
                    operation_type="insert",
                    instance_name=name,
                )
                observed = self._confirm_created_instance(spec)
                target_id = _require_provider_id(
                    operation.get("targetId"), "GCE insert target"
                )
                if target_id != observed["provider_instance_id"]:
                    raise RuntimeError("GCE insert target provider id changed at readback")
                inventory = spec["inventory"]
                created.append(
                    {
                        "job_id": inventory["job_id"],
                        "source_role": inventory["source_role"],
                        "attempt_id": inventory["attempt_id"],
                        "instance_name": name,
                        "service_account": inventory["service_account"],
                        "provider_instance_id": observed["provider_instance_id"],
                        "provider_boot_disk_id": observed["provider_boot_disk_id"],
                        "request_id": requests[name],
                        "operation_name": operation["name"],
                        "operation_id": str(operation["id"]),
                        "spec_sha256": spec["spec_sha256"],
                        "bootstrap_sha256": inventory["bootstrap_sha256"],
                        "labels": deepcopy(body["labels"]),
                        "observed_status": observed["status"],
                        "recovered_after_insert_failure": False,
                    }
                )
        except Exception as exc:
            collision = "collision/precondition status" in str(exc)
            if not collision:
                created_names = {row["instance_name"] for row in created}
                for spec in self._specs:
                    body = spec["body"]
                    name = body["name"]
                    if name in created_names:
                        continue
                    try:
                        observed = self._confirm_created_instance(spec)
                    except Exception:
                        observed = None
                    if observed is None:
                        continue
                    inventory = spec["inventory"]
                    created.append(
                        {
                            "job_id": inventory["job_id"],
                            "source_role": inventory["source_role"],
                            "attempt_id": inventory["attempt_id"],
                            "instance_name": name,
                            "service_account": inventory["service_account"],
                            "provider_instance_id": observed["provider_instance_id"],
                            "provider_boot_disk_id": observed["provider_boot_disk_id"],
                            "request_id": requests[name],
                            "operation_name": None,
                            "operation_id": None,
                            "spec_sha256": spec["spec_sha256"],
                            "bootstrap_sha256": inventory["bootstrap_sha256"],
                            "labels": deepcopy(body["labels"]),
                            "observed_status": observed["status"],
                            "recovered_after_insert_failure": True,
                        }
                    )
            partial = self.validate_create_receipt(
                _seal(self._create_core(rows=created, observed_at_utc=observed_at_utc))
            )
            raise GceCreateIncompleteError(
                "GCE selected create did not complete", partial_receipt=partial
            ) from exc
        return self.validate_create_receipt(
            _seal(self._create_core(rows=created, observed_at_utc=observed_at_utc))
        )

    def reconcile_create(
        self,
        *,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
    ) -> dict[str, Any]:
        """POST-free exact scan of every selected name after an interrupted create."""

        if self.mode != "reconcile-create" or self._used:
            raise PermissionError("GCE reconcile-create mode is unavailable or consumed")
        self._used = True
        names = [spec["body"]["name"] for spec in self._specs]
        requests = _request_ids(request_ids, names=names, forbidden=set())
        known = {
            row["instance_name"]: row
            for row in ([] if self._create_receipt is None else self._create_receipt["rows"])
        }
        rows: list[dict[str, Any]] = []
        for spec in self._specs:
            inventory = spec["inventory"]
            name = spec["body"]["name"]
            previous = known.get(name)
            if previous is not None and previous["request_id"] != requests[name]:
                raise ValueError("GCE reconcile requestId differs from durable intent")
            observed = self._read_exact_instance(
                spec,
                expected_provider_id=(
                    None if previous is None else previous["provider_instance_id"]
                ),
                allow_absent=True,
            )
            if observed is None:
                if previous is not None:
                    raise RuntimeError("known GCE create disappeared during reconciliation")
                continue
            if (
                previous is not None
                and observed["provider_boot_disk_id"]
                != previous["provider_boot_disk_id"]
            ):
                raise RuntimeError("known GCE boot disk changed during reconciliation")
            rows.append(
                {
                    "job_id": inventory["job_id"],
                    "source_role": inventory["source_role"],
                    "attempt_id": inventory["attempt_id"],
                    "instance_name": name,
                    "service_account": inventory["service_account"],
                    "provider_instance_id": observed["provider_instance_id"],
                    "provider_boot_disk_id": observed["provider_boot_disk_id"],
                    "request_id": requests[name],
                    "operation_name": (
                        None if previous is None else previous["operation_name"]
                    ),
                    "operation_id": (
                        None if previous is None else previous["operation_id"]
                    ),
                    "spec_sha256": spec["spec_sha256"],
                    "bootstrap_sha256": inventory["bootstrap_sha256"],
                    "labels": deepcopy(spec["body"]["labels"]),
                    "observed_status": observed["status"],
                    "recovered_after_insert_failure": (
                        True if previous is None else previous[
                            "recovered_after_insert_failure"
                        ]
                    ),
                }
            )
        return self.validate_create_receipt(
            _seal(self._create_core(rows=rows, observed_at_utc=observed_at_utc))
        )

    def validate_status_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self._create_receipt is None:
            raise PermissionError("GCE status validation requires a create receipt")
        payload, digest = _pop_receipt(
            value, keys=_STATUS_KEYS, label="GCE status receipt"
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(self._create_receipt["rows"]):
            raise ValueError("GCE status receipt cardinality changed")
        checked: list[dict[str, Any]] = []
        for created, raw in zip(self._create_receipt["rows"], rows, strict=True):
            if not isinstance(raw, Mapping):
                raise ValueError("GCE status row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(row, _STATUS_ROW_KEYS, "GCE status row")
            if (
                row.get("instance_name") != created["instance_name"]
                or row.get("provider_instance_id") != created["provider_instance_id"]
                or row.get("provider_boot_disk_id") != created["provider_boot_disk_id"]
                or row.get("spec_sha256") != created["spec_sha256"]
                or row.get("status")
                not in {
                    "PROVISIONING", "STAGING", "RUNNING", "STOPPING",
                    "TERMINATED", "ABSENT",
                }
            ):
                raise ValueError("GCE status row identity changed")
            checked.append(row)
        if (
            payload.get("schema") != GCE_STATUS_RECEIPT_SCHEMA
            or payload.get("status") != "exact_owned_gce_status_readback"
            or payload.get("bundle_sha256") != self._bundle["bundle_sha256"]
            or payload.get("create_receipt_sha256")
            != self._create_receipt["receipt_sha256"]
            or payload.get("project") != PROJECT
            or payload.get("zone") != ZONE
            or payload.get("wave_index") != self._bundle["wave_index"]
            or payload.get("instance_count") != len(checked)
            or payload.get("all_specs_exact") is not True
            or payload.get("read_only") is not True
            or payload.get("current_profile_changed") is not False
        ):
            raise ValueError("GCE status receipt contract changed")
        _require_utc(payload.get("observed_at_utc"), "GCE status time")
        if payload["observed_at_utc"] < self._create_receipt["observed_at_utc"]:
            raise ValueError("GCE status receipt predates create receipt")
        payload["rows"] = checked
        return {**payload, "receipt_sha256": digest}

    def read_status(self, *, observed_at_utc: str) -> dict[str, Any]:
        if self.mode != "read-status" or self._used or self._create_receipt is None:
            raise PermissionError("GCE read-status mode is unavailable or consumed")
        self._used = True
        rows: list[dict[str, Any]] = []
        for created in self._create_receipt["rows"]:
            spec = self._spec_by_name[created["instance_name"]]
            instance_absent, disk_absent = self._read_absence(spec)
            if instance_absent != disk_absent:
                raise RuntimeError(
                    "GCE preemption readback found only instance or boot disk absent"
                )
            if instance_absent:
                rows.append(
                    {
                        "instance_name": created["instance_name"],
                        "provider_instance_id": created["provider_instance_id"],
                        "provider_boot_disk_id": created["provider_boot_disk_id"],
                        "status": "ABSENT",
                        "spec_sha256": created["spec_sha256"],
                    }
                )
                continue
            observed = self._read_exact_instance(
                spec,
                expected_provider_id=created["provider_instance_id"],
                allow_absent=False,
            )
            assert observed is not None
            if observed["provider_boot_disk_id"] != created["provider_boot_disk_id"]:
                raise RuntimeError("GCE status boot disk provider id changed")
            rows.append(
                {
                    "instance_name": created["instance_name"],
                    "provider_instance_id": observed["provider_instance_id"],
                    "provider_boot_disk_id": observed["provider_boot_disk_id"],
                    "status": observed["status"],
                    "spec_sha256": created["spec_sha256"],
                }
            )
        core = {
            "schema": GCE_STATUS_RECEIPT_SCHEMA,
            "status": "exact_owned_gce_status_readback",
            "bundle_sha256": self._bundle["bundle_sha256"],
            "create_receipt_sha256": self._create_receipt["receipt_sha256"],
            "project": PROJECT,
            "zone": ZONE,
            "wave_index": self._bundle["wave_index"],
            "rows": rows,
            "instance_count": len(rows),
            "all_specs_exact": True,
            "read_only": True,
            "observed_at_utc": _require_utc(observed_at_utc, "GCE status time"),
            "current_profile_changed": False,
        }
        return self.validate_status_receipt(_seal(core))

    def _validate_delete_rows(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if self._create_receipt is None:
            raise PermissionError("GCE delete rows require a create receipt")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("GCE delete rows are missing")
        if len(rows) != len(self._create_receipt["rows"]):
            raise ValueError("GCE delete rows do not exactly cover owned creates")
        result: list[dict[str, Any]] = []
        request_ids: set[str] = set()
        operation_ids: set[str] = set()
        create_requests = {row["request_id"] for row in self._create_receipt["rows"]}
        for created, raw in zip(self._create_receipt["rows"], rows, strict=True):
            if not isinstance(raw, Mapping):
                raise ValueError("GCE delete row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(row, _DELETE_ROW_KEYS, "GCE delete row")
            request_id = _require_request_id(row.get("request_id"), "GCE delete requestId")
            if (
                row.get("instance_name") != created["instance_name"]
                or row.get("provider_instance_id") != created["provider_instance_id"]
                or row.get("provider_boot_disk_id") != created["provider_boot_disk_id"]
                or request_id in request_ids
                or request_id in create_requests
            ):
                raise ValueError("GCE delete row escaped or reused owned identity")
            recovered = row.get("recovered_after_transport_ambiguity")
            if recovered is True:
                if (
                    row.get("already_absent") is not False
                    or row.get("operation_name") is not None
                    or row.get("operation_id") is not None
                ):
                    raise ValueError("recovered GCE instance delete row changed")
            elif recovered is False and row.get("already_absent") is True:
                if row.get("operation_name") is not None or row.get("operation_id") is not None:
                    raise ValueError("already-absent GCE delete row has an operation")
            elif recovered is False and row.get("already_absent") is False:
                if (
                    not isinstance(row.get("operation_name"), str)
                    or _OPERATION.fullmatch(row["operation_name"]) is None
                ):
                    raise ValueError("GCE delete operation name changed")
                operation_id = _require_provider_id(
                    row.get("operation_id"), "GCE delete operation"
                )
                if operation_id in operation_ids:
                    raise ValueError("GCE delete operation was duplicated")
                operation_ids.add(operation_id)
                row["operation_id"] = operation_id
            else:
                raise ValueError("GCE delete absence flag changed")
            request_ids.add(request_id)
            result.append(row)
        return result

    def _validate_orphan_disk_delete_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        instance_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("GCE orphan boot disk delete rows are missing")
        selected_order = {
            spec["body"]["name"]: index for index, spec in enumerate(self._specs)
        }
        instance_by_name = {row["instance_name"]: row for row in instance_rows}
        result: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        seen_disk_ids: set[str] = set()
        seen_operation_ids: set[str] = set()
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("GCE orphan boot disk delete row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(
                row, _ORPHAN_DISK_DELETE_ROW_KEYS,
                "GCE orphan boot disk delete row",
            )
            name = row.get("instance_name")
            request_id = _require_request_id(
                row.get("request_id"), "GCE orphan boot disk delete requestId"
            )
            disk_id = _require_provider_id(
                row.get("provider_boot_disk_id"), "GCE orphan boot disk delete"
            )
            instance_row = instance_by_name.get(name)
            if (
                name not in selected_order
                or name in seen_names
                or disk_id in seen_disk_ids
                or (
                    instance_row is not None
                    and (
                        not (
                            instance_row["already_absent"] is True
                            or instance_row[
                                "recovered_after_transport_ambiguity"
                            ] is True
                        )
                        or instance_row["request_id"] == request_id
                        or instance_row["provider_boot_disk_id"] != disk_id
                    )
                )
            ):
                raise ValueError("GCE orphan boot disk escaped selected ownership")
            recovered = row.get("recovered_after_transport_ambiguity")
            already_absent = row.get("already_absent")
            if recovered is True:
                if (
                    already_absent is not False
                    or row.get("operation_name") is not None
                    or row.get("operation_id") is not None
                ):
                    raise ValueError("recovered GCE orphan delete row changed")
            elif recovered is False and already_absent is True:
                if row.get("operation_name") is not None or row.get("operation_id") is not None:
                    raise ValueError("already-absent orphan disk row has an operation")
            elif recovered is False and already_absent is False:
                if (
                    not isinstance(row.get("operation_name"), str)
                    or _OPERATION.fullmatch(row["operation_name"]) is None
                ):
                    raise ValueError("GCE orphan disk delete operation name changed")
                operation_id = _require_provider_id(
                    row.get("operation_id"), "GCE orphan disk delete operation"
                )
                if operation_id in seen_operation_ids:
                    raise ValueError("GCE orphan disk delete operation was duplicated")
                seen_operation_ids.add(operation_id)
                row["operation_id"] = operation_id
            else:
                raise ValueError("GCE orphan disk delete outcome changed")
            row["provider_boot_disk_id"] = disk_id
            seen_names.add(name)
            seen_disk_ids.add(disk_id)
            result.append(row)
        if [selected_order[row["instance_name"]] for row in result] != sorted(
            selected_order[row["instance_name"]] for row in result
        ):
            raise ValueError("GCE orphan boot disk delete row order changed")
        return result

    def validate_delete_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self._create_receipt is None:
            raise PermissionError("GCE delete validation requires a create receipt")
        payload, digest = _pop_receipt(
            value, keys=_DELETE_KEYS, label="GCE delete receipt"
        )
        rows = self._validate_delete_rows(payload.get("rows"))
        orphan_rows = self._validate_orphan_disk_delete_rows(
            payload.get("orphan_boot_disk_rows"), instance_rows=rows
        )
        deleted = sum(row["already_absent"] is False for row in rows) + sum(
            row["already_absent"] is False for row in orphan_rows
        )
        absent = sum(row["already_absent"] is True for row in rows) + sum(
            row["already_absent"] is True for row in orphan_rows
        )
        selected_names = [spec["body"]["name"] for spec in self._specs]
        if (
            payload.get("schema") != GCE_DELETE_RECEIPT_SCHEMA
            or payload.get("status") != "exact_owned_gce_delete_operations_complete"
            or payload.get("bundle_sha256") != self._bundle["bundle_sha256"]
            or payload.get("create_receipt_sha256")
            != self._create_receipt["receipt_sha256"]
            or payload.get("project") != PROJECT
            or payload.get("zone") != ZONE
            or payload.get("wave_index") != self._bundle["wave_index"]
            or payload.get("owned_instance_count") != len(rows)
            or payload.get("selected_instance_count") != len(selected_names)
            or payload.get("scanned_selected_names") != selected_names
            or payload.get("all_selected_names_scanned") is not True
            or payload.get("delete_operation_count") != deleted
            or payload.get("already_absent_count") != absent
            or payload.get("orphan_boot_disk_count") != len(orphan_rows)
            or payload.get("exact_provider_ids_verified") is not True
            or payload.get("exact_labels_verified") is not True
            or payload.get("wildcard_delete_used") is not False
            or payload.get("request_id_reuse_authorized") is not False
            or payload.get("additional_create_authorized") is not False
            or payload.get("current_profile_changed") is not False
        ):
            raise ValueError("GCE delete receipt contract changed")
        _require_utc(payload.get("observed_at_utc"), "GCE delete time")
        if payload["observed_at_utc"] < self._create_receipt["observed_at_utc"]:
            raise ValueError("GCE delete receipt predates create receipt")
        payload["rows"] = rows
        payload["orphan_boot_disk_rows"] = orphan_rows
        return {**payload, "receipt_sha256": digest}

    def delete_owned(
        self,
        *,
        request_ids: Mapping[str, Any],
        orphan_disk_request_ids: Mapping[str, Any] | None = None,
        observed_at_utc: str,
    ) -> dict[str, Any]:
        if self.mode != "delete" or self._used or self._create_receipt is None:
            raise PermissionError("GCE delete mode is unavailable or consumed")
        self._used = True
        names = [spec["body"]["name"] for spec in self._specs]
        forbidden = {row["request_id"] for row in self._create_receipt["rows"]}
        requests = _request_ids(request_ids, names=names, forbidden=forbidden)
        orphan_requests = (
            None
            if orphan_disk_request_ids is None
            else _request_ids(
                orphan_disk_request_ids,
                names=names,
                forbidden=forbidden | set(requests.values()),
            )
        )
        created_by_name = {
            row["instance_name"]: row for row in self._create_receipt["rows"]
        }
        rows: list[dict[str, Any]] = []
        orphan_rows: list[dict[str, Any]] = []
        for spec in self._specs:
            name = spec["body"]["name"]
            created = created_by_name.get(name)
            instance_row_added = False
            instance_absent, disk_absent = self._read_absence(spec)
            if not instance_absent and disk_absent:
                raise RuntimeError(
                    "GCE selected instance exists without its exact boot disk"
                )
            if not instance_absent:
                if created is None:
                    raise RuntimeError(
                        "GCE selected instance exists outside the exact create receipt"
                    )
                observed = self._read_exact_instance(
                    spec,
                    expected_provider_id=created["provider_instance_id"],
                    allow_absent=False,
                )
                assert observed is not None
                if observed["provider_boot_disk_id"] != created["provider_boot_disk_id"]:
                    raise RuntimeError("GCE delete boot disk provider id changed")
                recovered_instance = False
                try:
                    response = self._http(
                        method="DELETE",
                        url=(
                            f"{self._instance_url(name)}?requestId="
                            f"{urllib.parse.quote(requests[name], safe='')}"
                        ),
                        allowed_statuses=(200, 404),
                    )
                except GcePhaseBTransportError:
                    final_instance_absent, final_disk_absent = self._read_absence(spec)
                    if not final_instance_absent:
                        raise
                    operation_name = None
                    operation_id = None
                    already_absent = False
                    recovered_instance = True
                    disk_absent = final_disk_absent
                else:
                    if response.status == 404:
                        operation_name = None
                        operation_id = None
                        already_absent = True
                    else:
                        operation = self._operation(
                            self._json(response, "GCE delete"),
                            operation_type="delete",
                            instance_name=name,
                        )
                        if _require_provider_id(
                            operation.get("targetId"), "GCE delete target"
                        ) != created["provider_instance_id"]:
                            raise RuntimeError("GCE delete target provider id changed")
                        operation_name = operation["name"]
                        operation_id = str(operation["id"])
                        already_absent = False
                rows.append(
                    {
                        "instance_name": name,
                        "provider_instance_id": created["provider_instance_id"],
                        "provider_boot_disk_id": created["provider_boot_disk_id"],
                        "request_id": requests[name],
                        "operation_name": operation_name,
                        "operation_id": operation_id,
                        "already_absent": already_absent,
                        "recovered_after_transport_ambiguity": recovered_instance,
                    }
                )
                instance_row_added = True
                if not recovered_instance or disk_absent:
                    continue
            if disk_absent:
                if created is not None:
                    rows.append(
                        {
                            "instance_name": name,
                            "provider_instance_id": created["provider_instance_id"],
                            "provider_boot_disk_id": created["provider_boot_disk_id"],
                            "request_id": requests[name],
                            "operation_name": None,
                            "operation_id": None,
                            "already_absent": True,
                            "recovered_after_transport_ambiguity": False,
                        }
                    )
                continue

            exact_disk = self._read_exact_disk(
                spec,
                expected_provider_id=(
                    None if created is None else created["provider_boot_disk_id"]
                ),
                expected_attached=False,
                allow_absent=False,
            )
            assert exact_disk is not None
            disk_id = exact_disk["provider_boot_disk_id"]
            if created is not None and not instance_row_added:
                rows.append(
                    {
                        "instance_name": name,
                        "provider_instance_id": created["provider_instance_id"],
                        "provider_boot_disk_id": disk_id,
                        "request_id": requests[name],
                        "operation_name": None,
                        "operation_id": None,
                        "already_absent": True,
                        "recovered_after_transport_ambiguity": False,
                    }
                )
            if orphan_requests is None:
                raise PermissionError(
                    "orphan boot disk cleanup requires distinct fixed requestIds"
                )
            recovered = False
            try:
                response = self._http(
                    method="DELETE",
                    url=(
                        f"{self._disk_url(name)}?requestId="
                        f"{urllib.parse.quote(orphan_requests[name], safe='')}"
                    ),
                    allowed_statuses=(200, 404),
                )
            except GcePhaseBTransportError:
                final_instance_absent, final_disk_absent = self._read_absence(spec)
                if not final_instance_absent or not final_disk_absent:
                    raise
                operation_name = None
                operation_id = None
                already_absent = False
                recovered = True
            else:
                if response.status == 404:
                    operation_name = None
                    operation_id = None
                    already_absent = True
                else:
                    operation = self._operation(
                        self._json(response, "GCE orphan boot disk delete"),
                        operation_type="delete",
                        instance_name=name,
                        resource_collection="disks",
                    )
                    if _require_provider_id(
                        operation.get("targetId"), "GCE orphan disk delete target"
                    ) != disk_id:
                        raise RuntimeError(
                            "GCE orphan disk delete target provider id changed"
                        )
                    operation_name = operation["name"]
                    operation_id = str(operation["id"])
                    already_absent = False
            orphan_rows.append(
                {
                    "instance_name": name,
                    "provider_boot_disk_id": disk_id,
                    "request_id": orphan_requests[name],
                    "operation_name": operation_name,
                    "operation_id": operation_id,
                    "already_absent": already_absent,
                    "recovered_after_transport_ambiguity": recovered,
                }
            )
        core = {
            "schema": GCE_DELETE_RECEIPT_SCHEMA,
            "status": "exact_owned_gce_delete_operations_complete",
            "bundle_sha256": self._bundle["bundle_sha256"],
            "create_receipt_sha256": self._create_receipt["receipt_sha256"],
            "project": PROJECT,
            "zone": ZONE,
            "wave_index": self._bundle["wave_index"],
            "owned_instance_count": len(rows),
            "selected_instance_count": len(names),
            "scanned_selected_names": names,
            "all_selected_names_scanned": True,
            "delete_operation_count": (
                sum(row["already_absent"] is False for row in rows)
                + sum(row["already_absent"] is False for row in orphan_rows)
            ),
            "already_absent_count": (
                sum(row["already_absent"] is True for row in rows)
                + sum(row["already_absent"] is True for row in orphan_rows)
            ),
            "rows": rows,
            "orphan_boot_disk_count": len(orphan_rows),
            "orphan_boot_disk_rows": orphan_rows,
            "exact_provider_ids_verified": True,
            "exact_labels_verified": True,
            "wildcard_delete_used": False,
            "request_id_reuse_authorized": False,
            "additional_create_authorized": False,
            "observed_at_utc": _require_utc(observed_at_utc, "GCE delete time"),
            "current_profile_changed": False,
        }
        return self.validate_delete_receipt(_seal(core))

    def validate_delete_reconcile_receipt(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self._create_receipt is None:
            raise PermissionError("GCE delete reconciliation requires a create receipt")
        payload, digest = _pop_receipt(
            value,
            keys=_DELETE_RECONCILE_KEYS,
            label="GCE delete reconcile receipt",
        )
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(self._specs):
            raise ValueError("GCE delete reconcile row cardinality changed")
        created_by_name = {
            row["instance_name"]: row for row in self._create_receipt["rows"]
        }
        rows: list[dict[str, Any]] = []
        orphan_count = 0
        for spec, raw in zip(self._specs, raw_rows, strict=True):
            if not isinstance(raw, Mapping):
                raise ValueError("GCE delete reconcile row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(row, _DELETE_RECONCILE_ROW_KEYS, "GCE delete reconcile row")
            name = spec["body"]["name"]
            orphan = row.get("exact_orphan_boot_disk_owned")
            disk_id = row.get("provider_boot_disk_id")
            if (
                row.get("instance_name") != name
                or row.get("instance_absent") is not True
                or type(row.get("boot_disk_absent")) is not bool
                or type(orphan) is not bool
                or orphan is row["boot_disk_absent"]
            ):
                raise ValueError("GCE delete reconcile row state changed")
            if orphan:
                normalized_id = _require_provider_id(
                    disk_id, "GCE reconciled orphan boot disk"
                )
                created = created_by_name.get(name)
                if created is not None and normalized_id != created["provider_boot_disk_id"]:
                    raise ValueError("GCE reconciled orphan boot disk identity changed")
                row["provider_boot_disk_id"] = normalized_id
                orphan_count += 1
            elif disk_id is not None:
                raise ValueError("absent GCE disk has a provider identity")
            rows.append(row)
        all_disks_absent = orphan_count == 0
        recovered_raw = payload.get("recovered_delete_receipt")
        recovered = (
            None
            if recovered_raw is None
            else self.validate_delete_receipt(recovered_raw)
        )
        if (
            payload.get("schema") != GCE_DELETE_RECONCILE_RECEIPT_SCHEMA
            or payload.get("status")
            != (
                "all_selected_resources_absent_after_transport_ambiguity"
                if all_disks_absent
                else "exact_owned_orphan_boot_disk_cleanup_required"
            )
            or payload.get("bundle_sha256") != self._bundle["bundle_sha256"]
            or payload.get("create_receipt_sha256")
            != self._create_receipt["receipt_sha256"]
            or payload.get("project") != PROJECT
            or payload.get("zone") != ZONE
            or payload.get("wave_index") != self._bundle["wave_index"]
            or payload.get("selected_instance_count") != len(self._specs)
            or payload.get("all_instances_absent") is not True
            or payload.get("all_boot_disks_absent") is not all_disks_absent
            or payload.get("orphan_boot_disk_count") != orphan_count
            or payload.get("orphan_cleanup_required") is not (not all_disks_absent)
            or (recovered is None) is not (not all_disks_absent)
            or payload.get("read_only") is not True
            or payload.get("additional_create_authorized") is not False
            or payload.get("current_profile_changed") is not False
        ):
            raise ValueError("GCE delete reconcile receipt contract changed")
        _require_utc(payload.get("observed_at_utc"), "GCE delete reconcile time")
        payload["rows"] = rows
        payload["recovered_delete_receipt"] = recovered
        return {**payload, "receipt_sha256": digest}

    def reconcile_delete(
        self,
        *,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
    ) -> dict[str, Any]:
        """GET-only classification after an instance DELETE response was lost."""

        if (
            self.mode != "reconcile-delete"
            or self._used
            or self._create_receipt is None
        ):
            raise PermissionError("GCE reconcile-delete mode is unavailable or consumed")
        self._used = True
        names = [spec["body"]["name"] for spec in self._specs]
        forbidden = {row["request_id"] for row in self._create_receipt["rows"]}
        requests = _request_ids(request_ids, names=names, forbidden=forbidden)
        created_by_name = {
            row["instance_name"]: row for row in self._create_receipt["rows"]
        }
        rows: list[dict[str, Any]] = []
        for spec in self._specs:
            name = spec["body"]["name"]
            instance_absent, disk_absent = self._read_absence(spec)
            if not instance_absent:
                raise RuntimeError(
                    "GCE instance DELETE ambiguity is unresolved; instance remains"
                )
            if disk_absent:
                rows.append(
                    {
                        "instance_name": name,
                        "instance_absent": True,
                        "boot_disk_absent": True,
                        "exact_orphan_boot_disk_owned": False,
                        "provider_boot_disk_id": None,
                    }
                )
                continue
            created = created_by_name.get(name)
            disk = self._read_exact_disk(
                spec,
                expected_provider_id=(
                    None if created is None else created["provider_boot_disk_id"]
                ),
                expected_attached=False,
                allow_absent=False,
            )
            assert disk is not None
            rows.append(
                {
                    "instance_name": name,
                    "instance_absent": True,
                    "boot_disk_absent": False,
                    "exact_orphan_boot_disk_owned": True,
                    "provider_boot_disk_id": disk["provider_boot_disk_id"],
                }
            )
        orphan_count = sum(row["exact_orphan_boot_disk_owned"] for row in rows)
        recovered_delete: dict[str, Any] | None = None
        if orphan_count == 0:
            delete_rows = [
                {
                    "instance_name": created["instance_name"],
                    "provider_instance_id": created["provider_instance_id"],
                    "provider_boot_disk_id": created["provider_boot_disk_id"],
                    "request_id": requests[created["instance_name"]],
                    "operation_name": None,
                    "operation_id": None,
                    "already_absent": True,
                    "recovered_after_transport_ambiguity": False,
                }
                for created in self._create_receipt["rows"]
            ]
            recovered_delete = self.validate_delete_receipt(
                _seal(
                    {
                        "schema": GCE_DELETE_RECEIPT_SCHEMA,
                        "status": "exact_owned_gce_delete_operations_complete",
                        "bundle_sha256": self._bundle["bundle_sha256"],
                        "create_receipt_sha256": self._create_receipt["receipt_sha256"],
                        "project": PROJECT,
                        "zone": ZONE,
                        "wave_index": self._bundle["wave_index"],
                        "owned_instance_count": len(delete_rows),
                        "selected_instance_count": len(names),
                        "scanned_selected_names": names,
                        "all_selected_names_scanned": True,
                        "delete_operation_count": 0,
                        "already_absent_count": len(delete_rows),
                        "rows": delete_rows,
                        "orphan_boot_disk_count": 0,
                        "orphan_boot_disk_rows": [],
                        "exact_provider_ids_verified": True,
                        "exact_labels_verified": True,
                        "wildcard_delete_used": False,
                        "request_id_reuse_authorized": False,
                        "additional_create_authorized": False,
                        "observed_at_utc": _require_utc(
                            observed_at_utc, "GCE delete reconcile time"
                        ),
                        "current_profile_changed": False,
                    }
                )
            )
        core = {
            "schema": GCE_DELETE_RECONCILE_RECEIPT_SCHEMA,
            "status": (
                "all_selected_resources_absent_after_transport_ambiguity"
                if orphan_count == 0
                else "exact_owned_orphan_boot_disk_cleanup_required"
            ),
            "bundle_sha256": self._bundle["bundle_sha256"],
            "create_receipt_sha256": self._create_receipt["receipt_sha256"],
            "project": PROJECT,
            "zone": ZONE,
            "wave_index": self._bundle["wave_index"],
            "rows": rows,
            "selected_instance_count": len(names),
            "all_instances_absent": True,
            "all_boot_disks_absent": orphan_count == 0,
            "orphan_boot_disk_count": orphan_count,
            "orphan_cleanup_required": orphan_count > 0,
            "recovered_delete_receipt": recovered_delete,
            "read_only": True,
            "observed_at_utc": _require_utc(
                observed_at_utc, "GCE delete reconcile time"
            ),
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
        return self.validate_delete_reconcile_receipt(_seal(core))

    def validate_absence_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self._create_receipt is None or self._delete_receipt is None:
            raise PermissionError("GCE absence validation requires create/delete receipts")
        payload, digest = _pop_receipt(
            value, keys=_ABSENCE_KEYS, label="GCE absence receipt"
        )
        expected_names = [spec["body"]["name"] for spec in self._specs]
        if (
            payload.get("schema") != GCE_ABSENCE_RECEIPT_SCHEMA
            or payload.get("status") != "exact_owned_gce_instances_and_disks_absent"
            or payload.get("bundle_sha256") != self._bundle["bundle_sha256"]
            or payload.get("create_receipt_sha256")
            != self._create_receipt["receipt_sha256"]
            or payload.get("delete_receipt_sha256")
            != self._delete_receipt["receipt_sha256"]
            or payload.get("project") != PROJECT
            or payload.get("zone") != ZONE
            or payload.get("wave_index") != self._bundle["wave_index"]
            or payload.get("checked_instance_count") != len(expected_names)
            or payload.get("absent_instance_names") != expected_names
            or payload.get("absent_boot_disk_names") != expected_names
            or payload.get("all_instances_absent") is not True
            or payload.get("all_boot_disks_absent") is not True
            or payload.get("read_only") is not True
            or payload.get("additional_create_authorized") is not False
            or payload.get("current_profile_changed") is not False
        ):
            raise ValueError("GCE absence receipt contract changed")
        _require_utc(payload.get("observed_at_utc"), "GCE absence time")
        if payload["observed_at_utc"] < self._delete_receipt["observed_at_utc"]:
            raise ValueError("GCE absence receipt predates delete receipt")
        return {**payload, "receipt_sha256": digest}

    def verify_absence(self, *, observed_at_utc: str) -> dict[str, Any]:
        if (
            self.mode != "absence"
            or self._used
            or self._create_receipt is None
            or self._delete_receipt is None
        ):
            raise PermissionError("GCE absence mode is unavailable or consumed")
        self._used = True
        names = [spec["body"]["name"] for spec in self._specs]
        absent_names: list[str] = []
        for attempt in range(ABSENCE_POLL_ATTEMPTS):
            absent_names = []
            for name in names:
                spec = self._spec_by_name[name]
                instance_absent, disk_absent = self._read_absence(spec)
                if instance_absent and disk_absent:
                    absent_names.append(name)
            if absent_names == names:
                break
            if attempt + 1 < ABSENCE_POLL_ATTEMPTS:
                self._sleep(ABSENCE_POLL_INTERVAL_SECONDS)
        if absent_names != names:
            raise RuntimeError("GCE instance or boot disk absence is not confirmed")
        core = {
            "schema": GCE_ABSENCE_RECEIPT_SCHEMA,
            "status": "exact_owned_gce_instances_and_disks_absent",
            "bundle_sha256": self._bundle["bundle_sha256"],
            "create_receipt_sha256": self._create_receipt["receipt_sha256"],
            "delete_receipt_sha256": self._delete_receipt["receipt_sha256"],
            "project": PROJECT,
            "zone": ZONE,
            "wave_index": self._bundle["wave_index"],
            "checked_instance_count": len(names),
            "absent_instance_names": names,
            "absent_boot_disk_names": list(names),
            "all_instances_absent": True,
            "all_boot_disks_absent": True,
            "read_only": True,
            "observed_at_utc": _require_utc(observed_at_utc, "GCE absence time"),
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
        return self.validate_absence_receipt(_seal(core))


def validate_create_receipt(
    adapter: GceWavePhaseBAdapter, value: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(adapter, GceWavePhaseBAdapter):
        raise TypeError("GCE create receipt validator requires a Phase-B adapter")
    return adapter.validate_create_receipt(value)


def validate_status_receipt(
    adapter: GceWavePhaseBAdapter, value: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(adapter, GceWavePhaseBAdapter):
        raise TypeError("GCE status receipt validator requires a Phase-B adapter")
    return adapter.validate_status_receipt(value)


def validate_delete_receipt(
    adapter: GceWavePhaseBAdapter, value: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(adapter, GceWavePhaseBAdapter):
        raise TypeError("GCE delete receipt validator requires a Phase-B adapter")
    return adapter.validate_delete_receipt(value)


def validate_delete_reconcile_receipt(
    adapter: GceWavePhaseBAdapter, value: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(adapter, GceWavePhaseBAdapter):
        raise TypeError("GCE delete reconcile validator requires a Phase-B adapter")
    return adapter.validate_delete_reconcile_receipt(value)


def validate_absence_receipt(
    adapter: GceWavePhaseBAdapter, value: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(adapter, GceWavePhaseBAdapter):
        raise TypeError("GCE absence receipt validator requires a Phase-B adapter")
    return adapter.validate_absence_receipt(value)


__all__ = [
    "BOOT_DISK_INTERFACE", "BOOT_DISK_SIZE_GB", "BOOT_DISK_TYPE",
    "GCE_ABSENCE_RECEIPT_SCHEMA", "GCE_CREATE_RECEIPT_SCHEMA",
    "GCE_DELETE_RECEIPT_SCHEMA", "GCE_DELETE_RECONCILE_RECEIPT_SCHEMA",
    "GCE_STATUS_RECEIPT_SCHEMA", "GceCreateIncompleteError",
    "GcePhaseBTransportError", "GceWavePhaseBAdapter", "HttpResponse",
    "MACHINE_TYPE", "NETWORK", "NIC_TYPE", "OAUTH_SCOPE", "PROJECT",
    "REGION", "SUBNETWORK", "TOKEN_ENV", "VM_TTL_SECONDS", "ZONE",
    "canonical_bytes", "canonical_sha256", "validate_absence_receipt",
    "validate_create_receipt", "validate_delete_receipt",
    "validate_delete_reconcile_receipt",
    "validate_status_receipt",
]
