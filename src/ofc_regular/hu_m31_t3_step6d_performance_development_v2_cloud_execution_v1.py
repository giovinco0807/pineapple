"""Fail-closed two-VM cloud boundary for T3 perfdev-v2.

The package/controller and mode-separated GCS/GCE HTTP adapter live here, but
importing them performs no network operation.  Launch mutations require a
fresh read-only receipt, immutable package, explicit authorization, and raw
one-shot nonce.  Successful-pair cleanup requires validated artifacts;
partial-launch cleanup is a separate owned-instance-only recovery contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_performance_development_v2_contract as contract_v1
from . import hu_m31_t3_step6d_performance_development_v2_local_package as local
from . import hu_m31_t3_step6d_performance_development_v2_preflight as preflight
from . import run_hu_m31_t3_step6d_performance_v2 as runner


CLOUD_PACKAGE_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_tail_v2_cloud_package_v1"
CLOUD_READY_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_tail_v2_cloud_package_ready_v1"
EXECUTION_PLAN_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_tail_v2_execution_plan_v1"
AUTHORIZATION_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_launch_authorization_v1"
LAUNCH_RECEIPT_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_launch_receipt_v1"
ROLE_RESULT_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_tail_v2_role_result_manifest_v1"
COLLECTION_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_tail_v2_collection_receipt_v1"
CLEANUP_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_cleanup_receipt_v1"
PARTIAL_CLEANUP_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_partial_cleanup_receipt_v1"
OWNED_LAUNCH_FAILURE_CLOSEOUT_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_owned_launch_failure_closeout_v1"
)
ZERO_CREATED_CLOSEOUT_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_zero_created_diagnostic_v1"
)
CONTENT_STAGE_AUTHORIZATION_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_content_stage_authorization_v1"
CONTENT_STAGE_RECEIPT_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_content_stage_receipt_v1"
LAUNCH_OVERLAY_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_fresh_launch_overlay_v1"
WORKER_IAM_READBACK_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_readback_receipt_v1"
)

CLOUD_PACKAGE_STATUS = "cloud_executable_package_ready_not_authorized"
SOURCE_NAME = "perfdev_v2_source.zip"
WHEELHOUSE_NAME = "perfdev_v2_wheelhouse.zip"
WHEELHOUSE_MANIFEST_NAME = "wheelhouse_manifest.json"
STARTUP_NAME = "startup_perfdev_v2_cloud_v1.sh"
PLAN_NAME = "execution_plan.json"
MANIFEST_NAME = "MANIFEST.json"
READY_NAME = "PACKAGE_READY.json"

SOURCE_ROLES = ("candidate", "reference")
LEGACY_MIXED_DIAGNOSTIC_HAND_INDICES = (2, 6, 7, 9, 13, 20, 21, 29, 33, 50)
TAIL_HAND_INDICES = tuple(contract_v1.TAIL_HAND_INDICES)
TAIL_HEAVY_HAND_INDICES = tuple(contract_v1.TAIL_HEAVY_HAND_INDICES)
TAIL_RANDOM_HAND_INDICES = tuple(contract_v1.TAIL_RANDOM_HAND_INDICES)
TAIL_RUN_CONTRACT_SCHEMA = contract_v1.TAIL_RUN_CONTRACT_SCHEMA
TAIL_RUN_CONTRACT_VARIANT = contract_v1.TAIL_RUN_CONTRACT_VARIANT
TAIL_RUN_CONTRACT_DIGEST = contract_v1.TAIL_RUN_CONTRACT_DIGEST
TAIL_SELECTION_MANIFEST_SHA256 = contract_v1.TAIL_SELECTION_MANIFEST_SHA256
MACHINE_TYPE = "c4-standard-16"
BOOT_DISK_TYPE = "hyperdisk-balanced"
BOOT_DISK_INTERFACE = "NVME"
BOOT_DISK_SIZE_GB = 20
NETWORK_NIC_TYPE = "GVNIC"
DELETE_CONFIRMATION_ATTEMPTS = 12
DELETE_CONFIRMATION_INTERVAL_SECONDS = 5
ZERO_CREATED_CONFIRMATION_ATTEMPTS = 12
ZERO_CREATED_CONFIRMATION_INTERVAL_SECONDS = 5
ZERO_CREATED_FAILURE_STAGES = (
    "overlay_failed",
    "authorization_failed",
    "prelaunch_aborted",
    "namespace_preflight_failed",
    "control_publish_failed",
    "before_first_insert_aborted",
)
GCE_ZONE_OPERATION_ATTEMPTS = 60
GCE_ZONE_OPERATION_INTERVAL_SECONDS = 2
GCE_INSTANCE_CONFIRMATION_ATTEMPTS = 20
GCE_INSTANCE_CONFIRMATION_INTERVAL_SECONDS = 1
WORKER_PROCESSES = 1
RAYON_THREADS = 16
VM_COUNT = 2
HEARTBEAT_SECONDS = 30
MAX_VM_RUNTIME_SECONDS = 4_200
VM_TTL_SECONDS = 4_500
MAX_ATTEMPTS_PER_ROLE = 1
MAX_AUTHORIZATION_LIFETIME_SECONDS = 180
WORKER_IAM_EXPIRY_OFFSET_SECONDS = 5_400
MAX_WORKER_IAM_WINDOW_SECONDS = 5_400
MIN_WORKER_IAM_WINDOW_SECONDS = VM_TTL_SECONDS + 300
MAX_LIVE_RECEIPT_AGE_SECONDS = local.MAX_RECEIPT_AGE_SECONDS
SPOT_PRICE_CEILING_USD_PER_VM_HOUR = contract_v1.MAX_SPOT_PRICE_USD_PER_VM_HOUR
MAX_TOTAL_COMPUTE_USD = "1.40"
FEATURE_ENCODER_SHA256 = local.FEATURE_LIBRARY_SHA256
RUNTIME_REQUIREMENTS_RELATIVE = "payload/tooling/configs/hu_m43_attempt08_runtime_requirements.txt"
WHEELHOUSE_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1"
WORKER_SERVICE_ACCOUNT = (
    "ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com"
)
WORKER_BUCKET = "pokerhu-ofc-solver-485418-training"
WORKER_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
NETWORK_NAME = "default"
SUBNETWORK_NAME = "default"
NAT_ROUTER_NAME = "ofc-t3-nat-router-asia-northeast1"
NAT_NAME = "ofc-t3-nat-asia-northeast1"
WORKER_READER_ROLES = frozenset(
    {
        "roles/storage.objectViewer",
        "roles/storage.objectAdmin",
        "projects/ofc-solver-485418/roles/ofcM31T3ObjectReaderV1",
    }
)
WORKER_CREATOR_ROLES = frozenset(
    {
        "roles/storage.objectCreator",
        "roles/storage.objectAdmin",
        "projects/ofc-solver-485418/roles/ofcM31T3ResultCreatorV1",
    }
)
WORKER_READER_ROLE = "projects/ofc-solver-485418/roles/ofcM31T3ObjectReaderV1"
WORKER_CREATOR_ROLE = "projects/ofc-solver-485418/roles/ofcM31T3ResultCreatorV1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTUP_PATH = (
    _REPO_ROOT
    / "scripts/startup_hu_m31_t3_step6d_performance_development_v2_cloud_v1.sh"
)
_SHA = re.compile(r"[0-9a-f]{64}")
_RUN = re.compile(r"regular-hu-m31-c02-perfdev-v2-[a-z0-9][a-z0-9-]{7,47}")
_IDENTITY = re.compile(r"perfdev-v2-[a-z0-9][a-z0-9-]{7,47}")
_INSTANCE = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")
_OWNER = re.compile(r"pdv2-[0-9a-f]{20}")
_WHEEL_NAME = re.compile(r"[A-Za-z0-9_.+!-]+\.whl")
_FORBIDDEN_STARTUP_HASHES = frozenset(
    {
        local.REARM2_STARTUP_SHA256,
        contract_v1.REARM2_PACKAGE_SOURCE_SHA256,
        contract_v1.REARM2_PACKAGE_SMOKE_SHA256,
    }
)


class CloudTransport(Protocol):
    """Narrow interface implemented by fakes now and a reviewed GCP adapter later."""

    def inspect_namespace(
        self,
        *,
        run_name: str,
        identity_namespace: str,
        result_prefix: str,
        instance_names: Sequence[str],
    ) -> Mapping[str, Any]: ...

    def put_if_absent(self, *, object_name: str, payload: bytes) -> Mapping[str, Any]: ...

    def copy_if_absent(
        self,
        *,
        source_object: str,
        destination_object: str,
        expected_sha256: str,
        expected_bytes: int,
    ) -> Mapping[str, Any]: ...

    def create_instance(self, *, specification: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_object(self, *, object_name: str) -> bytes | None: ...

    def list_objects(self, *, prefix: str) -> Sequence[str]: ...

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None: ...

    def delete_instance_exact(
        self, *, instance_name: str, ownership_label: str, execution_plan_sha256: str
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


def _stdlib_http_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: int,
) -> HttpResponse:
    request = urllib.request.Request(
        url=url, data=body, headers=dict(headers), method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return HttpResponse(
                status=int(response.status),
                body=response.read(),
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            body=exc.read(),
            headers=dict(exc.headers.items()) if exc.headers else {},
        )


class GcpHttpTransport:
    """Mode-separated real GCS/GCE REST adapter.

    The bearer token is read only from ``GOOGLE_OAUTH_ACCESS_TOKEN`` for each
    request.  It is never stored on the instance, returned, or interpolated
    into an error.  A launch-mode instance cannot exist without a validated
    authorization+raw nonce; cleanup mode cannot exist without a validated
    complete collection receipt.
    """

    TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
    _MODES = frozenset(
        {
            "stage", "launch", "receive", "cleanup", "partial-cleanup",
            "launch-failure-cleanup", "zero-created-closeout",
        }
    )

    def __init__(
        self,
        *,
        project: str,
        zone: str,
        bucket: str,
        mode: str,
        execution_plan: Mapping[str, Any],
        authorization: Mapping[str, Any] | None = None,
        raw_one_shot_nonce: str | None = None,
        now_unix_seconds: int | None = None,
        collection_receipt: Mapping[str, Any] | None = None,
        partial_launch_receipt: Mapping[str, Any] | None = None,
        owned_launch_receipt: Mapping[str, Any] | None = None,
        worker_iam_readback: Mapping[str, Any] | None = None,
        requester: HttpRequester | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError("real cloud adapter mode is invalid")
        self._plan = validate_execution_plan(
            execution_plan, require_fresh_receipt=False
        )
        if (
            project != self._plan["project"]
            or zone != self._plan["zone"]
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,62}", bucket)
        ):
            raise ValueError("real adapter target differs from execution plan")
        self.project = project
        self.zone = zone
        self.bucket = bucket
        self.mode = mode
        self._requester = requester or _stdlib_http_request
        self._clock = clock or time.time
        if not callable(self._clock):
            raise ValueError("real cloud adapter clock is invalid")
        self._sleep = sleep or time.sleep
        if not callable(self._sleep):
            raise ValueError("real cloud adapter sleep function is invalid")
        self._authorization: dict[str, Any] | None = None
        self._stage_authorization: dict[str, Any] | None = None
        self._collection: dict[str, Any] | None = None
        self._partial_launch: dict[str, Any] | None = None
        self._owned_launch: dict[str, Any] | None = None
        self._zero_created_iam_readback: dict[str, Any] | None = None
        if mode == "stage":
            if authorization is None or raw_one_shot_nonce is None:
                raise PermissionError(
                    "content stage adapter requires authorization and raw one-shot nonce"
                )
            self._stage_authorization = _validate_content_stage_authorization(
                authorization,
                execution_plan=self._plan,
                raw_nonce=raw_one_shot_nonce,
                now_unix_seconds=now_unix_seconds,
            )
        elif mode == "launch":
            if authorization is None or raw_one_shot_nonce is None:
                raise PermissionError(
                    "launch adapter requires authorization and raw one-shot nonce"
                )
            self._authorization = validate_launch_authorization(
                authorization,
                execution_plan=self._plan,
                raw_nonce=raw_one_shot_nonce,
                now_unix_seconds=now_unix_seconds,
            )
        elif mode == "cleanup":
            if collection_receipt is None:
                raise PermissionError(
                    "cleanup adapter requires a complete validated collection receipt"
                )
            checked = _validate_collection_receipt(collection_receipt)
            if checked["execution_plan_sha256"] != canonical_sha256(self._plan):
                raise PermissionError("cleanup receipt belongs to another plan")
            self._collection = checked
        elif mode == "partial-cleanup":
            if partial_launch_receipt is None:
                raise PermissionError(
                    "partial cleanup adapter requires a validated partial launch receipt"
                )
            checked_partial = _validate_partial_launch_receipt(
                partial_launch_receipt, execution_plan=self._plan
            )
            self._partial_launch = checked_partial
        elif mode == "launch-failure-cleanup":
            if owned_launch_receipt is None:
                raise PermissionError(
                    "launch-failure cleanup requires an exact full launch receipt"
                )
            self._owned_launch = _validate_full_launch_receipt(
                owned_launch_receipt, execution_plan=self._plan
            )
        elif mode == "zero-created-closeout":
            if worker_iam_readback is None or raw_one_shot_nonce is None:
                raise PermissionError(
                    "zero-created closeout requires worker IAM readback and raw launch nonce"
                )
            try:
                parsed_nonce = uuid.UUID(raw_one_shot_nonce)
            except ValueError as exc:
                raise ValueError("one-shot nonce is invalid") from exc
            if parsed_nonce.version != 4 or str(parsed_nonce) != raw_one_shot_nonce:
                raise ValueError("one-shot nonce is invalid")
            self._zero_created_iam_readback = validate_worker_iam_readback(
                worker_iam_readback,
                execution_plan=self._plan,
                expected_one_shot_nonce_sha256=hashlib.sha256(
                    raw_one_shot_nonce.encode("ascii")
                ).hexdigest(),
                now_unix_seconds=None,
            )
        elif (
            authorization is not None
            or raw_one_shot_nonce is not None
            or collection_receipt is not None
            or partial_launch_receipt is not None
            or owned_launch_receipt is not None
            or worker_iam_readback is not None
        ):
            raise ValueError("receive adapter does not accept mutation capabilities")

    def _token_headers(self, *, content_type: str | None = None) -> dict[str, str]:
        token = os.environ.get(self.TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            character.isspace() for character in token
        ):
            raise PermissionError(
                f"Bearer token must be supplied only through {self.TOKEN_ENV}"
            )
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def _require_mutation_authorization_live(self) -> int:
        if self.mode == "stage" and self._stage_authorization is not None:
            authorization = self._stage_authorization
            label = "content stage"
        elif self.mode == "launch" and self._authorization is not None:
            authorization = self._authorization
            label = "launch"
        else:
            raise PermissionError(
                "time-bounded cloud mutation requires stage or launch authorization"
            )
        try:
            now = int(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise PermissionError("cloud mutation clock is invalid") from None
        if not (
            authorization["authorized_unix_seconds"]
            <= now
            <= authorization["expires_unix_seconds"]
        ):
            raise PermissionError(
                f"{label} authorization expired before cloud mutation"
            )
        return now

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
        if method not in {"GET", "POST", "DELETE"}:
            raise ValueError("HTTP method escaped fixed adapter surface")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 600
        ):
            raise ValueError("cloud HTTP timeout escaped fixed 600-second maximum")
        response = self._requester(
            method,
            url,
            self._token_headers(content_type=content_type),
            body,
            timeout_seconds,
        )
        if response.status not in allowed_statuses:
            # Provider response bodies can echo request data.  Do not place
            # them in exceptions or receipts.
            raise RuntimeError(
                f"cloud HTTP {method} failed with status {response.status}"
            )
        return response

    @staticmethod
    def _json(response: HttpResponse, label: str) -> dict[str, Any]:
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} response was not JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} response was not an object")
        return value

    def _gcs_list(self, prefix: str) -> list[str]:
        if not prefix.startswith(self._plan["result_prefix"]):
            raise ValueError("GCS list prefix escaped frozen run namespace")
        query = urllib.parse.urlencode(
            {"prefix": prefix, "maxResults": "1000", "fields": "items/name,nextPageToken"}
        )
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o?{query}"
        )
        payload = self._json(self._http(method="GET", url=url), "GCS list")
        if payload.get("nextPageToken") is not None:
            raise RuntimeError("GCS namespace exceeded one fixed 1000-object page")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("GCS list items changed")
        names: list[str] = []
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise RuntimeError("GCS list item changed")
            name = item["name"]
            if not name.startswith(prefix):
                raise RuntimeError("GCS provider returned an out-of-prefix object")
            names.append(name)
        if len(names) != len(set(names)):
            raise RuntimeError("GCS provider returned duplicate objects")
        return sorted(names)

    def inspect_namespace(
        self,
        *,
        run_name: str,
        identity_namespace: str,
        result_prefix: str,
        instance_names: Sequence[str],
    ) -> Mapping[str, Any]:
        if self.mode != "launch" or self._authorization is None:
            raise PermissionError("namespace launch preflight requires launch mode")
        expected_names = [row["instance_name"] for row in self._plan["instances"]]
        if (
            run_name != self._plan["run_name"]
            or identity_namespace != self._plan["identity_namespace"]
            or result_prefix != self._plan["result_prefix"]
            or list(instance_names) != expected_names
        ):
            raise ValueError("namespace inspection escaped execution plan")
        counts: dict[str, int] = {}
        for name in expected_names:
            encoded = urllib.parse.quote(name, safe="")
            url = (
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{self.project}/zones/{self.zone}/instances/{encoded}"
            )
            response = self._http(method="GET", url=url, allowed_statuses=(200, 404))
            counts[name] = 0 if response.status == 404 else 1
        objects = self._gcs_list(result_prefix)
        router_url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/regions/{self._plan['region']}/routers/{NAT_ROUTER_NAME}"
        )
        router = self._json(self._http(method="GET", url=router_url), "Cloud NAT router")
        nats = router.get("nats")
        if not isinstance(nats, list):
            raise RuntimeError("Cloud NAT configuration is missing")
        matching_nats = [value for value in nats if isinstance(value, Mapping) and value.get("name") == NAT_NAME]
        nat = matching_nats[0] if len(matching_nats) == 1 else {}
        expected_network_suffix = f"/projects/{self.project}/global/networks/{NETWORK_NAME}"
        network_path = {
            "read_only": True,
            "router_name": router.get("name"),
            "router_region": self._plan["region"],
            "router_network_exact": isinstance(router.get("network"), str) and router["network"].endswith(expected_network_suffix),
            "nat_name": nat.get("name"),
            "nat_count_with_name": len(matching_nats),
            "nat_ip_allocate_option": nat.get("natIpAllocateOption"),
            "source_subnetwork_ip_ranges_to_nat": nat.get("sourceSubnetworkIpRangesToNat"),
            "external_ipv4_on_vm": False,
            "path_ready": (
                router.get("name") == NAT_ROUTER_NAME
                and isinstance(router.get("network"), str)
                and router["network"].endswith(expected_network_suffix)
                and len(matching_nats) == 1
                and nat.get("natIpAllocateOption") == "AUTO_ONLY"
                and nat.get("sourceSubnetworkIpRangesToNat") == "ALL_SUBNETWORKS_ALL_IP_RANGES"
            ),
        }
        iam_url = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/iam"
            "?optionsRequestedPolicyVersion=3"
        )
        iam = self._json(self._http(method="GET", url=iam_url), "bucket IAM")
        bindings = iam.get("bindings", [])
        if not isinstance(bindings, list):
            raise RuntimeError("bucket IAM bindings changed")
        member = f"serviceAccount:{WORKER_SERVICE_ACCOUNT}"
        reader_prefix_expression = (
            'resource.name.startsWith("projects/_/buckets/'
            f'{self.bucket}/objects/{result_prefix}control/")'
        )
        creator_prefix_expression = (
            'resource.name.startsWith("projects/_/buckets/'
            f'{self.bucket}/objects/{result_prefix}")'
        )
        viewer = False
        creator = False
        exact_count = 0
        reader_count = 0
        creator_count = 0
        excess_count = 0
        expiries: list[int] = []
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise RuntimeError("bucket IAM binding changed")
            members = binding.get("members", [])
            condition = binding.get("condition")
            if member not in members:
                continue
            if not isinstance(condition, Mapping):
                excess_count += 1
                continue
            expression = condition.get("expression")
            role = binding.get("role")
            if not isinstance(expression, str):
                excess_count += 1
                continue
            reader_pattern = re.fullmatch(
                re.escape(reader_prefix_expression)
                + r' && request\.time < timestamp\("([^"\r\n]+)"\)',
                expression,
            )
            creator_pattern = re.fullmatch(
                re.escape(creator_prefix_expression)
                + r' && request\.time < timestamp\("([^"\r\n]+)"\)',
                expression,
            )
            reader_match = role in WORKER_READER_ROLES and reader_pattern is not None
            creator_match = role in WORKER_CREATOR_ROLES and creator_pattern is not None
            if reader_match or creator_match:
                exact_count += 1
                reader_count += int(reader_match)
                creator_count += int(creator_match)
                stamp = (reader_pattern or creator_pattern).group(1)  # type: ignore[union-attr]
                try:
                    expiries.append(
                        int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp())
                    )
                except ValueError as exc:
                    raise RuntimeError("bucket IAM expiry timestamp changed") from exc
            else:
                excess_count += 1
            viewer = viewer or reader_match
            creator = creator or creator_match
        return {
            "read_only": True,
            "cloud_mutated": False,
            "run_name_collision_count": 1 if objects else 0,
            "identity_collision_count": 1 if objects else 0,
            "result_prefix_object_count": len(objects),
            "instance_collision_counts": counts,
            "worker_iam": {
                "read_only": True,
                "service_account": WORKER_SERVICE_ACCOUNT,
                "bucket": self.bucket,
                "required_reader_prefix": f"{result_prefix}control/",
                "required_creator_prefix": result_prefix,
                "exact_conditional_binding_count": exact_count,
                "object_get_allowed": viewer,
                "object_create_allowed": creator,
                "object_list_required": False,
                "reader_binding_count": reader_count,
                "creator_binding_count": creator_count,
                "excess_worker_binding_count": excess_count,
                "iam_expiry_unix_seconds": expiries[0] if expiries and len(set(expiries)) == 1 else None,
                "exact_prefix_condition": reader_count == 1 and creator_count == 1,
                "all_required_permissions_present": viewer and creator and reader_count == 1 and creator_count == 1 and excess_count == 0 and len(set(expiries)) == 1,
            },
            "network_path": network_path,
        }

    def put_if_absent(self, *, object_name: str, payload: bytes) -> Mapping[str, Any]:
        if self.mode == "launch" and self._authorization is not None:
            prefix = f"{self._plan['result_prefix']}control/"
        elif self.mode == "stage" and self._stage_authorization is not None:
            prefix = self._plan["content_staging"]["object_prefix"]
        else:
            raise PermissionError("GCS mutation requires an authorized stage/launch mode")
        if not object_name.startswith(prefix) or "/../" in object_name:
            raise ValueError("GCS write escaped authorized immutable prefix")
        query = urllib.parse.urlencode(
            {"uploadType": "media", "ifGenerationMatch": "0", "name": object_name}
        )
        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o?{query}"
        )
        timeout_seconds = 600 if self.mode == "stage" else 60
        self._require_mutation_authorization_live()
        response = self._http(
            method="POST",
            url=url,
            body=payload,
            content_type="application/octet-stream",
            allowed_statuses=(200, 412),
            timeout_seconds=timeout_seconds,
        )
        digest = hashlib.sha256(payload).hexdigest()
        if response.status == 412:
            return {
                "created": False,
                "object_name": object_name,
                "sha256": digest,
                "bytes": len(payload),
                "generation": "0",
            }
        value = self._json(response, "GCS upload")
        if value.get("name") != object_name or not str(value.get("generation", "")).isdigit():
            raise RuntimeError("GCS upload identity changed")
        return {
            "created": True,
            "object_name": object_name,
            "sha256": digest,
            "bytes": len(payload),
            "generation": str(value["generation"]),
        }

    def copy_if_absent(
        self,
        *,
        source_object: str,
        destination_object: str,
        expected_sha256: str,
        expected_bytes: int,
    ) -> Mapping[str, Any]:
        if self.mode != "launch" or self._authorization is None:
            raise PermissionError("GCS rewrite requires authorized launch mode")
        source_prefix = self._plan["content_staging"]["object_prefix"]
        destination_prefix = f"{self._plan['result_prefix']}control/"
        if (
            not source_object.startswith(source_prefix)
            or not destination_object.startswith(destination_prefix)
            or _SHA.fullmatch(expected_sha256) is None
            or not _plain_int(expected_bytes)
            or expected_bytes <= 0
        ):
            raise ValueError("GCS rewrite escaped staged-to-control boundary")
        source_encoded = urllib.parse.quote(source_object, safe="")
        destination_encoded = urllib.parse.quote(destination_object, safe="")
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o/{source_encoded}/rewriteTo/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o/{destination_encoded}"
            "?ifGenerationMatch=0"
        )
        self._require_mutation_authorization_live()
        response = self._http(
            method="POST", url=url, body=b"", content_type="application/json", allowed_statuses=(200, 412)
        )
        if response.status == 412:
            return {"created": False, "source_object": source_object, "object_name": destination_object, "sha256": expected_sha256, "bytes": expected_bytes, "generation": "0"}
        value = self._json(response, "GCS rewrite")
        resource = value.get("resource")
        if value.get("done") is not True or not isinstance(resource, Mapping) or resource.get("name") != destination_object or not str(resource.get("generation", "")).isdigit() or int(resource.get("size", -1)) != expected_bytes:
            raise RuntimeError("GCS rewrite identity changed")
        return {"created": True, "source_object": source_object, "object_name": destination_object, "sha256": expected_sha256, "bytes": expected_bytes, "generation": str(resource["generation"])}

    def create_instance(self, *, specification: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.mode != "launch" or self._authorization is None:
            raise PermissionError("GCE mutation requires authorized launch mode")
        spec = dict(specification)
        row = next(
            (item for item in self._plan["instances"] if item["instance_name"] == spec.get("name")),
            None,
        )
        if (
            row is None
            or spec.get("project") != self.project
            or spec.get("zone") != self.zone
            or spec.get("machine_type") != MACHINE_TYPE
            or spec.get("provisioning_model") != "SPOT"
            or spec.get("automatic_restart") is not False
            or spec.get("on_host_maintenance") != "TERMINATE"
            or spec.get("max_run_duration_seconds") != VM_TTL_SECONDS
            or spec.get("deletion_protection") is not False
            or spec.get("image") != self._plan["image"]["selfLink"]
            or spec.get("boot_disk_type") != BOOT_DISK_TYPE
            or spec.get("boot_disk_interface") != BOOT_DISK_INTERFACE
            or spec.get("boot_disk_size_gb") != BOOT_DISK_SIZE_GB
            or spec.get("network_nic_type") != NETWORK_NIC_TYPE
            or spec.get("labels", {}).get("ofc-owner") != row["ownership_label"]
            or spec.get("labels", {}).get("ofc-plan") != canonical_sha256(self._plan)[:32]
            or spec.get("metadata", {}).get("STARTUP_SHA256") != self._plan["startup"]["sha256"]
        ):
            raise ValueError("GCE specification escaped frozen authorized VM")
        metadata = spec["metadata"]
        if not isinstance(metadata.get("startup-script"), str) or hashlib.sha256(metadata["startup-script"].encode("utf-8")).hexdigest() != self._plan["startup"]["sha256"]:
            raise ValueError("GCE startup bytes differ from execution plan")
        body = {
            "name": spec["name"],
            "machineType": f"zones/{self.zone}/machineTypes/{MACHINE_TYPE}",
            "deletionProtection": False,
            "labels": spec["labels"],
            "scheduling": {
                "provisioningModel": "SPOT",
                "instanceTerminationAction": "DELETE",
                "automaticRestart": False,
                "onHostMaintenance": "TERMINATE",
                "maxRunDuration": {"seconds": str(VM_TTL_SECONDS)},
            },
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "type": "PERSISTENT",
                    "interface": BOOT_DISK_INTERFACE,
                    "initializeParams": {
                        "sourceImage": self._plan["image"]["selfLink"],
                        "diskSizeGb": str(BOOT_DISK_SIZE_GB),
                        "diskType": (
                            f"zones/{self.zone}/diskTypes/{BOOT_DISK_TYPE}"
                        ),
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "network": f"global/networks/{NETWORK_NAME}",
                    "subnetwork": f"regions/{self._plan['region']}/subnetworks/{SUBNETWORK_NAME}",
                    "nicType": NETWORK_NIC_TYPE,
                    # Deliberately no accessConfigs: no external IPv4 address.
                }
            ],
            "serviceAccounts": [
                {
                    "email": WORKER_SERVICE_ACCOUNT,
                    "scopes": [WORKER_OAUTH_SCOPE],
                }
            ],
            "metadata": {
                "items": [
                    {"key": key, "value": str(value)}
                    for key, value in sorted(metadata.items())
                ]
            },
        }
        url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/instances"
        )
        self._require_mutation_authorization_live()
        response = self._http(
            method="POST",
            url=url,
            body=canonical_bytes(body),
            content_type="application/json",
            allowed_statuses=(200,),
        )
        operation = self._json(response, "GCE insert")
        operation_name = operation.get("name")
        expected_target = (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/instances/{spec['name']}"
        )
        if (
            not isinstance(operation_name, str)
            or _INSTANCE.fullmatch(operation_name) is None
            or operation.get("status") not in {"PENDING", "RUNNING", "DONE"}
            or operation.get("operationType") != "insert"
            or operation.get("targetLink") != expected_target
            or operation.get("error") is not None
            or operation.get("httpErrorStatusCode") is not None
        ):
            raise RuntimeError("GCE insert operation identity changed")
        operation_url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/operations/"
            f"{urllib.parse.quote(operation_name, safe='')}"
        )
        completed = operation if operation["status"] == "DONE" else None
        for attempt in range(GCE_ZONE_OPERATION_ATTEMPTS):
            if completed is not None:
                break
            polled = self._json(
                self._http(method="GET", url=operation_url),
                "GCE insert operation poll",
            )
            if (
                polled.get("name") != operation_name
                or polled.get("status") not in {"PENDING", "RUNNING", "DONE"}
                or polled.get("operationType") != "insert"
                or polled.get("targetLink") != expected_target
                or polled.get("error") is not None
                or polled.get("httpErrorStatusCode") is not None
            ):
                raise RuntimeError("GCE insert operation poll changed or failed")
            if polled["status"] == "DONE":
                completed = polled
                break
            if attempt + 1 < GCE_ZONE_OPERATION_ATTEMPTS:
                self._sleep(GCE_ZONE_OPERATION_INTERVAL_SECONDS)
        if completed is None:
            raise TimeoutError("GCE insert operation did not complete within bound")

        instance_url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/instances/"
            f"{urllib.parse.quote(spec['name'], safe='')}"
        )
        measured: dict[str, Any] | None = None
        for attempt in range(GCE_INSTANCE_CONFIRMATION_ATTEMPTS):
            instance_response = self._http(
                method="GET", url=instance_url, allowed_statuses=(200, 404)
            )
            if instance_response.status == 200:
                measured = self._json(instance_response, "GCE inserted instance")
                break
            if attempt + 1 < GCE_INSTANCE_CONFIRMATION_ATTEMPTS:
                self._sleep(GCE_INSTANCE_CONFIRMATION_INTERVAL_SECONDS)
        measured_labels = measured.get("labels") if measured is not None else None
        if (
            measured is None
            or measured.get("name") != spec["name"]
            or not isinstance(measured_labels, Mapping)
            or measured_labels.get("ofc-owner") != row["ownership_label"]
            or measured_labels.get("ofc-plan")
            != canonical_sha256(self._plan)[:32]
        ):
            raise RuntimeError("GCE inserted instance was not measured as exact owned")
        return {
            "created": True,
            "name": spec["name"],
            "status": "PROVISIONING",
            "ownership_label": row["ownership_label"],
            "execution_plan_sha256": canonical_sha256(self._plan),
        }

    def get_object(self, *, object_name: str) -> bytes | None:
        if self.mode not in {"receive", "cleanup", "zero-created-closeout"}:
            raise PermissionError(
                "GCS result read requires receive/cleanup/zero-created mode"
            )
        if not object_name.startswith(self._plan["result_prefix"]):
            raise ValueError("GCS object read escaped frozen result namespace")
        encoded = urllib.parse.quote(object_name, safe="")
        url = (
            "https://storage.googleapis.com/download/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o/{encoded}?alt=media"
        )
        response = self._http(method="GET", url=url, allowed_statuses=(200, 404))
        return None if response.status == 404 else response.body

    def list_objects(self, *, prefix: str) -> Sequence[str]:
        if self.mode not in {"receive", "cleanup", "zero-created-closeout"}:
            raise PermissionError(
                "GCS result list requires receive/cleanup/zero-created mode"
            )
        return self._gcs_list(prefix)

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None:
        if self.mode == "cleanup" and self._collection is not None:
            rows = self._collection["instance_ownership"]
            plan_digest = self._collection["execution_plan_sha256"]
        elif self.mode == "partial-cleanup" and self._partial_launch is not None:
            rows = [
                {
                    "instance_name": item["name"],
                    "ownership_label": item["ownership_label"],
                }
                for item in self._partial_launch["created_instances"]
            ]
            plan_digest = self._partial_launch["execution_plan_sha256"]
        elif self.mode == "launch-failure-cleanup" and self._owned_launch is not None:
            rows = [
                {
                    "instance_name": item["name"],
                    "ownership_label": item["ownership_label"],
                }
                for item in self._owned_launch["created_instances"]
            ]
            plan_digest = self._owned_launch["execution_plan_sha256"]
        elif (
            self.mode == "zero-created-closeout"
            and self._zero_created_iam_readback is not None
        ):
            rows = self._plan["instances"]
            plan_digest = canonical_sha256(self._plan)
        else:
            raise PermissionError("instance cleanup read requires an owned cleanup mode")
        row = next((item for item in rows if item["instance_name"] == instance_name), None)
        if row is None:
            raise ValueError("instance read escaped validated cleanup pair")
        url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/instances/"
            f"{urllib.parse.quote(instance_name, safe='')}"
        )
        response = self._http(method="GET", url=url, allowed_statuses=(200, 404))
        if response.status == 404:
            return None
        value = self._json(response, "GCE instance get")
        labels = value.get("labels")
        if not isinstance(labels, Mapping):
            raise RuntimeError("GCE instance labels are missing")
        actual_name = value.get("name")
        actual_owner = labels.get("ofc-owner")
        actual_plan_label = labels.get("ofc-plan")
        if (
            actual_name != instance_name
            or actual_owner != row["ownership_label"]
            or actual_plan_label != plan_digest[:32]
        ):
            raise RuntimeError("GCE instance measured ownership labels changed")
        return {
            "name": actual_name,
            "ownership_label": actual_owner,
            "execution_plan_sha256": plan_digest,
        }

    def delete_instance_exact(
        self, *, instance_name: str, ownership_label: str, execution_plan_sha256: str
    ) -> Mapping[str, Any]:
        if self.mode == "cleanup" and self._collection is not None:
            rows = self._collection["instance_ownership"]
            plan_digest = self._collection["execution_plan_sha256"]
        elif self.mode == "partial-cleanup" and self._partial_launch is not None:
            rows = [
                {
                    "instance_name": item["name"],
                    "ownership_label": item["ownership_label"],
                }
                for item in self._partial_launch["created_instances"]
            ]
            plan_digest = self._partial_launch["execution_plan_sha256"]
        elif self.mode == "launch-failure-cleanup" and self._owned_launch is not None:
            rows = [
                {
                    "instance_name": item["name"],
                    "ownership_label": item["ownership_label"],
                }
                for item in self._owned_launch["created_instances"]
            ]
            plan_digest = self._owned_launch["execution_plan_sha256"]
        else:
            raise PermissionError("GCE delete requires validated owned cleanup mode")
        row = next((item for item in rows if item["instance_name"] == instance_name), None)
        if row is None or row["ownership_label"] != ownership_label or execution_plan_sha256 != plan_digest:
            raise PermissionError("GCE delete target escaped validated owned pair")
        url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/instances/"
            f"{urllib.parse.quote(instance_name, safe='')}"
        )
        response = self._http(method="DELETE", url=url, allowed_statuses=(200,))
        operation = self._json(response, "GCE delete")
        operation_name = operation.get("name")
        expected_target = (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/instances/{instance_name}"
        )
        if (
            not isinstance(operation_name, str)
            or _INSTANCE.fullmatch(operation_name) is None
            or operation.get("status") not in {"PENDING", "RUNNING", "DONE"}
            or operation.get("operationType") != "delete"
            or operation.get("targetLink") != expected_target
            or operation.get("error") is not None
            or operation.get("httpErrorStatusCode") is not None
        ):
            raise RuntimeError("GCE delete operation identity changed")
        operation_url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/operations/"
            f"{urllib.parse.quote(operation_name, safe='')}"
        )
        completed = operation if operation["status"] == "DONE" else None
        for attempt in range(GCE_ZONE_OPERATION_ATTEMPTS):
            if completed is not None:
                break
            polled = self._json(
                self._http(method="GET", url=operation_url),
                "GCE delete operation poll",
            )
            if (
                polled.get("name") != operation_name
                or polled.get("status") not in {"PENDING", "RUNNING", "DONE"}
                or polled.get("operationType") != "delete"
                or polled.get("targetLink") != expected_target
                or polled.get("error") is not None
                or polled.get("httpErrorStatusCode") is not None
            ):
                raise RuntimeError("GCE delete operation poll changed or failed")
            if polled["status"] == "DONE":
                completed = polled
                break
            if attempt + 1 < GCE_ZONE_OPERATION_ATTEMPTS:
                self._sleep(GCE_ZONE_OPERATION_INTERVAL_SECONDS)
        if completed is None:
            raise TimeoutError("GCE delete operation did not complete within bound")
        return {
            "instance_name": instance_name,
            "deleted": True,
            "ownership_label": ownership_label,
        }

@dataclass(frozen=True)
class _PackageView:
    directory: Path
    manifest: dict[str, Any]
    ready: dict[str, Any]
    plan: dict[str, Any]
    plan_sha256: str
    source_sha256: str
    startup_sha256: str
    package_manifest_sha256: str


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _parse_canonical_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _validate_tail_v2_run_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = runner.validate_run_contract(value)
    if (
        contract.get("schema") != TAIL_RUN_CONTRACT_SCHEMA
        or runner.contract_variant(contract) != TAIL_RUN_CONTRACT_VARIANT
        or canonical_sha256(contract) != TAIL_RUN_CONTRACT_DIGEST
        or contract.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or contract.get("selection_manifest_sha256")
        != TAIL_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("runtime run contract is not the frozen Candidate02 tail-v2 contract")
    return contract


def _validate_tail_v2_role_manifest(
    value: Mapping[str, Any], *, source_role: str
) -> dict[str, Any]:
    manifest = runner.validate_shard_manifest(value)
    contract = _validate_tail_v2_run_contract(manifest["run_contract"])
    if (
        source_role not in SOURCE_ROLES
        or manifest.get("source_role") != source_role
        or manifest.get("work_hand_indices") != list(TAIL_HAND_INDICES)
        or manifest.get("run_contract_digest") != TAIL_RUN_CONTRACT_DIGEST
        or manifest.get("run_contract") != contract
    ):
        raise ValueError("runtime shard manifest is not the exact Candidate02 tail-v2 role")
    return manifest


def _validate_tail_v2_runtime_artifacts(
    values: Mapping[str, bytes], *, source_role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        contract_raw = values["run_contract.json"]
        manifest_raw = values["shard_manifest.json"]
    except KeyError as exc:
        raise ValueError("runtime tail-v2 contract artifacts are incomplete") from exc
    contract = _validate_tail_v2_run_contract(
        _parse_canonical_bytes(contract_raw, "runtime run contract")
    )
    manifest = _validate_tail_v2_role_manifest(
        _parse_canonical_bytes(manifest_raw, "runtime shard manifest"),
        source_role=source_role,
    )
    if manifest["run_contract"] != contract:
        raise ValueError("runtime contract and shard manifest differ")
    return contract, manifest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    _write_once(path, canonical_bytes(dict(value)))


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _receipt_unix_seconds(receipt: Mapping[str, Any]) -> int:
    try:
        stamp = receipt["runtime_preflight"]["observation"]["observed_at_utc"]
        return int(datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("embedded live receipt timestamp changed") from exc


def _require_fresh_receipt(receipt: Mapping[str, Any], *, now_unix_seconds: int) -> int:
    observed = _receipt_unix_seconds(receipt)
    age = now_unix_seconds - observed
    if age < -local.MAX_FUTURE_SKEW_SECONDS or age > MAX_LIVE_RECEIPT_AGE_SECONDS:
        raise ValueError("live GET-only preflight receipt is stale or future-dated")
    return age


def _safe_package_files(directory: Path) -> list[tuple[str, Path]]:
    root = directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("local package directory is unsafe")
    rows: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink() or bool(
            getattr(path.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("local package contains a link or reparse point")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
                raise ValueError("unsafe local package relative path")
            rows.append((relative, path))
    if not rows:
        raise ValueError("local package is empty")
    return rows


def _entry_records(rows: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [
        {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for relative, path in rows
    ]


def _zip_local_package(rows: Sequence[tuple[str, Path]], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("source archive destination is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, source in rows:
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, source.read_bytes())


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        if line.count("==") != 1 or any(token in line for token in (";", " --hash", " @ ")):
            raise ValueError("runtime requirement is not one exact name==version pin")
        name, version = line.split("==", 1)
        normalized = _normalized_distribution(name)
        if not normalized or not version or normalized in result:
            raise ValueError("runtime requirement set contains an invalid duplicate")
        result[normalized] = version
    if not result:
        raise ValueError("runtime requirement lock is empty")
    return result


def _wheel_distribution_identity(path: Path) -> tuple[str, str, tuple[str, ...]]:
    try:
        with zipfile.ZipFile(path, "r") as wheel:
            metadata_names = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
                and len(PurePosixPath(name).parts) == 2
            ]
            if len(metadata_names) != 1:
                raise ValueError("wheel must contain exactly one dist-info METADATA")
            raw = wheel.read(metadata_names[0]).decode("utf-8")
            wheel_names = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/WHEEL")
                and len(PurePosixPath(name).parts) == 2
            ]
            if len(wheel_names) != 1:
                raise ValueError("wheel must contain exactly one dist-info WHEEL")
            wheel_metadata = wheel.read(wheel_names[0]).decode("utf-8")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, KeyError) as exc:
        raise ValueError(f"invalid wheel archive: {path.name}") from exc
    name_values = [line[6:].strip() for line in raw.splitlines() if line.startswith("Name: ")]
    version_values = [line[9:].strip() for line in raw.splitlines() if line.startswith("Version: ")]
    if len(name_values) != 1 or len(version_values) != 1:
        raise ValueError("wheel METADATA Name/Version changed")
    tags = tuple(
        sorted(line[5:].strip() for line in wheel_metadata.splitlines() if line.startswith("Tag: "))
    )
    if not tags:
        raise ValueError("wheel compatibility tags are missing")
    for tag in tags:
        parts = tag.split("-")
        if len(parts) != 3:
            raise ValueError("wheel compatibility tag is invalid")
        python_tag, abi_tag, platform_tag = parts
        pure = python_tag == "py3" and abi_tag == "none" and platform_tag == "any"
        linux = (
            python_tag in {"cp311", "py3"}
            and abi_tag in ({"cp311"} if python_tag == "cp311" else {"none"})
            and (
                platform_tag == "linux_x86_64"
                or (platform_tag.startswith("manylinux") and platform_tag.endswith("_x86_64"))
            )
        )
        if not (pure or linux):
            raise ValueError(f"wheel is not CPython 3.11 Linux x86_64 compatible: {tag}")
    return _normalized_distribution(name_values[0]), version_values[0], tags


def _audit_wheelhouse(
    wheelhouse_dir: str | Path, *, requirements_path: Path
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    root = Path(wheelhouse_dir).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime wheelhouse is missing or unsafe")
    locked = _locked_requirements(requirements_path)
    rows: list[tuple[str, Path]] = []
    entries: list[dict[str, Any]] = []
    found: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or _WHEEL_NAME.fullmatch(path.name) is None:
            raise ValueError("wheelhouse must contain wheel files only")
        name, version, tags = _wheel_distribution_identity(path)
        if name in found:
            raise ValueError("wheelhouse contains duplicate distributions")
        if locked.get(name) != version:
            raise ValueError(f"wheelhouse distribution is not exactly locked: {name}=={version}")
        found[name] = version
        rows.append((path.name, path))
        entries.append(
            {
                "filename": path.name,
                "distribution": name,
                "version": version,
                "tags": list(tags),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if found != locked:
        missing = sorted(set(locked) - set(found))
        raise ValueError(f"wheelhouse does not exactly close runtime requirements: {missing}")
    manifest = {
        "schema": WHEELHOUSE_SCHEMA,
        "status": "complete_hash_pinned_offline_wheelhouse",
        "requirements_sha256": sha256_file(requirements_path),
        "python_abi": "cp311",
        "target_os": "linux",
        "target_architecture": "x86_64",
        "network_install_allowed": False,
        "entries": entries,
        "entry_count": len(entries),
        "entries_sha256": canonical_sha256(entries),
    }
    return manifest, rows


def _zip_wheelhouse(rows: Sequence[tuple[str, Path]], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("wheelhouse archive destination is immutable")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, source in rows:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, source.read_bytes())


def _validate_wheelhouse_archive(
    archive_path: Path, manifest: Mapping[str, Any]
) -> None:
    entries = manifest.get("entries")
    if (
        manifest.get("schema") != WHEELHOUSE_SCHEMA
        or manifest.get("status") != "complete_hash_pinned_offline_wheelhouse"
        or manifest.get("python_abi") != "cp311"
        or manifest.get("target_os") != "linux"
        or manifest.get("target_architecture") != "x86_64"
        or manifest.get("network_install_allowed") is not False
        or not isinstance(entries, list)
        or manifest.get("entry_count") != len(entries)
        or manifest.get("entries_sha256") != canonical_sha256(entries)
    ):
        raise ValueError("offline wheelhouse manifest changed")
    expected = {row.get("filename"): row for row in entries if isinstance(row, Mapping)}
    if len(expected) != len(entries):
        raise ValueError("offline wheelhouse entries are not unique")
    with zipfile.ZipFile(archive_path, "r") as archive, tempfile.TemporaryDirectory(
        prefix="ofc-wheel-validate-"
    ) as temporary:
        if archive.namelist() != sorted(expected):
            raise ValueError("offline wheelhouse archive topology changed")
        for name, record in expected.items():
            if _WHEEL_NAME.fullmatch(str(name)) is None:
                raise ValueError("unsafe wheelhouse filename")
            raw = archive.read(name)
            if len(raw) != record.get("bytes") or hashlib.sha256(raw).hexdigest() != record.get("sha256"):
                raise ValueError("offline wheelhouse archive entry changed")
            wheel_path = Path(temporary) / str(name)
            wheel_path.write_bytes(raw)
            distribution, version, tags = _wheel_distribution_identity(wheel_path)
            if (
                record.get("distribution") != distribution
                or record.get("version") != version
                or record.get("tags") != list(tags)
            ):
                raise ValueError("offline wheelhouse semantic identity changed")


def _extract_source_archive(
    archive_path: Path, destination: Path, expected_entries: Sequence[Mapping[str, Any]]
) -> None:
    expected = {str(row["path"]): dict(row) for row in expected_entries}
    if len(expected) != len(expected_entries):
        raise ValueError("source archive entry paths are not unique")
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names != sorted(expected) or len(names) != len(set(names)):
            raise ValueError("source archive topology changed")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
                raise ValueError("unsafe source archive path")
            info = archive.getinfo(name)
            if info.is_dir() or info.file_size != expected[name]["bytes"]:
                raise ValueError("source archive entry size changed")
            raw = archive.read(info)
            if hashlib.sha256(raw).hexdigest() != expected[name]["sha256"]:
                raise ValueError("source archive entry hash changed")
            target = destination / Path(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_once(target, raw)


def _validate_source_archive_without_extraction(
    archive_path: Path,
    *,
    plan: Mapping[str, Any],
) -> None:
    """Rehash every member while parsing only the small semantic manifests."""

    base_identity = plan["base_local_package"]
    expected = {str(row["path"]): dict(row) for row in base_identity["entries"]}
    selected_names = {
        local.MANIFEST_NAME,
        local.READY_NAME,
        local.SOURCE_MANIFEST_PATH,
        local.RUNTIME_MANIFEST_PATH,
        local.TOOLING_MANIFEST_PATH,
        local.DRY_RECEIPT_PACKAGE_PATH,
        local.TAIL_SELECTION_MANIFEST_PACKAGE_PATH,
        *local.ROLE_PACKAGE_PATHS.values(),
    }
    selected: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names != sorted(expected) or len(names) != len(set(names)):
            raise ValueError("source archive topology changed")
        for name in names:
            info = archive.getinfo(name)
            raw = archive.read(info)
            record = expected[name]
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("source archive member bytes changed")
            if name in selected_names:
                selected[name] = raw
    if set(selected) != selected_names:
        raise ValueError("source archive semantic manifests are incomplete")

    def parse(name: str) -> dict[str, Any]:
        try:
            value = json.loads(selected[name].decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("source archive semantic manifest is invalid") from exc
        if not isinstance(value, dict) or selected[name] != canonical_bytes(value):
            raise ValueError("source archive semantic manifest is not canonical")
        return value

    manifest = parse(local.MANIFEST_NAME)
    ready = parse(local.READY_NAME)
    runtime = parse(local.RUNTIME_MANIFEST_PATH)
    receipt = parse(local.DRY_RECEIPT_PACKAGE_PATH)
    selection = parse(local.TAIL_SELECTION_MANIFEST_PACKAGE_PATH)
    role_manifests = {
        role: _validate_tail_v2_role_manifest(parse(path), source_role=role)
        for role, path in local.ROLE_PACKAGE_PATHS.items()
    }
    preflight.validate_dry_run_receipt(receipt)
    if (
        hashlib.sha256(selected[local.MANIFEST_NAME]).hexdigest()
        != base_identity["manifest_sha256"]
        or hashlib.sha256(selected[local.READY_NAME]).hexdigest()
        != base_identity["ready_sha256"]
        or manifest.get("run_name") != plan["run_name"]
        or ready.get("run_name") != plan["run_name"]
        or manifest.get("cloud_executable") is not False
        or manifest.get("launch_authorized") is not False
        or ready.get("cloud_executable") is not False
        or ready.get("launch_authorized") is not False
        or ready.get("package_manifest_sha256")
        != hashlib.sha256(selected[local.MANIFEST_NAME]).hexdigest()
        or runtime.get("run_contract_digest") != TAIL_RUN_CONTRACT_DIGEST
        or hashlib.sha256(
            selected[local.TAIL_SELECTION_MANIFEST_PACKAGE_PATH]
        ).hexdigest()
        != TAIL_SELECTION_MANIFEST_SHA256
        or selection.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or selection.get("heavy_hand_indices") != list(TAIL_HEAVY_HAND_INDICES)
        or selection.get("random_hand_indices") != list(TAIL_RANDOM_HAND_INDICES)
        or any(
            role_manifests[role].get("work_hand_indices")
            != list(TAIL_HAND_INDICES)
            for role in SOURCE_ROLES
        )
        or runtime.get("allocation")
        != {
            "machine_type": MACHINE_TYPE,
            "source_roles": list(SOURCE_ROLES),
            "instances": VM_COUNT,
            "workers_per_source_process": WORKER_PROCESSES,
            "rayon_threads_per_worker": RAYON_THREADS,
        }
        or hashlib.sha256(selected[local.DRY_RECEIPT_PACKAGE_PATH]).hexdigest()
        != plan["live_preflight"]["receipt_sha256"]
        or _receipt_unix_seconds(receipt)
        != plan["live_preflight"]["observed_unix_seconds"]
    ):
        raise ValueError("source archive semantic binding changed")


def _identity_from_local_package(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_canonical(directory / local.MANIFEST_NAME, "local package manifest")
    validated = local.validate_local_package(
        directory, _expected_run_name=manifest.get("run_name")
    )
    ready = _read_canonical(directory / local.READY_NAME, "local package ready receipt")
    runtime = _read_canonical(directory / local.RUNTIME_MANIFEST_PATH, "local runtime manifest")
    selection_path = directory / local.TAIL_SELECTION_MANIFEST_PACKAGE_PATH
    selection = _read_canonical(selection_path, "tail-v2 selection manifest")
    role_manifests = {
        role: _validate_tail_v2_role_manifest(
            _read_canonical(directory / path, f"{role} role manifest"),
            source_role=role,
        )
        for role, path in local.ROLE_PACKAGE_PATHS.items()
    }
    if (
        validated.get("cloud_executable") is not False
        or validated.get("launch_authorized") is not False
        or manifest.get("cloud_executable") is not False
        or ready.get("launch_authorized") is not False
        or runtime.get("run_contract_digest") != TAIL_RUN_CONTRACT_DIGEST
        or sha256_file(selection_path) != TAIL_SELECTION_MANIFEST_SHA256
        or selection.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or selection.get("heavy_hand_indices") != list(TAIL_HEAVY_HAND_INDICES)
        or selection.get("random_hand_indices") != list(TAIL_RANDOM_HAND_INDICES)
        or any(
            role_manifests[role].get("work_hand_indices")
            != list(TAIL_HAND_INDICES)
            for role in SOURCE_ROLES
        )
        or runtime.get("allocation")
        != {
            "machine_type": MACHINE_TYPE,
            "source_roles": list(SOURCE_ROLES),
            "instances": VM_COUNT,
            "workers_per_source_process": WORKER_PROCESSES,
            "rayon_threads_per_worker": RAYON_THREADS,
        }
    ):
        raise ValueError("local package execution boundary changed")
    tooling = _read_canonical(directory / local.TOOLING_MANIFEST_PATH, "local tooling manifest")
    if tooling.get("fixture_only") is not False:
        raise ValueError("fixture-only local package cannot become cloud executable")
    receipt = _read_canonical(directory / local.DRY_RECEIPT_PACKAGE_PATH, "embedded live receipt")
    return manifest, runtime, receipt


def _instance_specs(run_name: str, identity: str) -> list[dict[str, Any]]:
    token = canonical_sha256({"run_name": run_name, "identity_namespace": identity})
    owner = f"pdv2-{token[:20]}"
    stem = f"m31-pdv2-{token[:12]}"
    values = []
    for role, suffix in (("candidate", "cand"), ("reference", "ref")):
        name = f"{stem}-{suffix}"
        if _INSTANCE.fullmatch(name) is None or _OWNER.fullmatch(owner) is None:
            raise ValueError("derived instance identity is invalid")
        values.append({"source_role": role, "instance_name": name, "ownership_label": owner})
    return values


def _plan_keys() -> set[str]:
    return {
        "schema", "status", "run_name", "identity_namespace", "result_prefix",
        "project", "region", "zone", "image", "image_identity_sha256",
        "base_local_package",
        "source_archive", "runtime_wheelhouse", "startup", "run_contract_digest",
        "run_contract_schema", "run_contract_variant",
        "selection_manifest_sha256", "source_roles", "tail_hand_indices",
        "heavy_hand_indices", "random_hand_indices", "instances", "allocation", "limits", "heartbeat",
        "checkpoint", "resume", "result_contract", "live_preflight",
        "roadmap_amendment",
        "worker_identity", "feature_encoder",
        "content_staging",
        "network",
        "cloud_executable", "launch_authorized", "cloud_mutated",
        "current_profile_changed",
    }


def build_cloud_executable_package(
    *,
    local_package_dir: str | Path,
    output_parent: str | Path,
    wheelhouse_dir: str | Path,
    startup_path: str | Path = DEFAULT_STARTUP_PATH,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    """Build an immutable executable package without authorizing or mutating cloud."""

    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    if not _plain_int(now):
        raise ValueError("packaging time must be an integer")
    base = Path(local_package_dir).resolve(strict=True)
    local_manifest, runtime, receipt = _identity_from_local_package(base)
    receipt_age = _require_fresh_receipt(receipt, now_unix_seconds=now)
    namespace = receipt["runtime_preflight"]["observation"]["namespace"]
    run_name = namespace["run_name"]
    identity = namespace["identity_namespace"]
    result_prefix = namespace["result_prefix"]
    if _RUN.fullmatch(run_name) is None or _IDENTITY.fullmatch(identity) is None:
        raise ValueError("fresh namespace identity changed")

    startup = Path(startup_path).resolve(strict=True)
    if startup.is_symlink() or not startup.is_file():
        raise ValueError("startup source is missing or unsafe")
    startup_sha = sha256_file(startup)
    if startup_sha in _FORBIDDEN_STARTUP_HASHES:
        raise ValueError("denylisted startup bytes cannot be reused")
    startup_raw = startup.read_bytes()
    if not startup_raw.startswith(b"#!/usr/bin/env bash\n") or b"set -Eeuo pipefail" not in startup_raw:
        raise ValueError("fresh startup fail-closed header changed")

    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    destination = parent / run_name
    if destination.exists():
        raise FileExistsError("cloud package destination is immutable")
    stage = parent / f".{run_name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    stage.mkdir()
    try:
        rows = _safe_package_files(base)
        source_entries = _entry_records(rows)
        requirements_path = base / RUNTIME_REQUIREMENTS_RELATIVE
        wheelhouse_manifest, wheel_rows = _audit_wheelhouse(
            wheelhouse_dir, requirements_path=requirements_path
        )
        _zip_local_package(rows, stage / SOURCE_NAME)
        _zip_wheelhouse(wheel_rows, stage / WHEELHOUSE_NAME)
        write_json_once(stage / WHEELHOUSE_MANIFEST_NAME, wheelhouse_manifest)
        _write_once(stage / STARTUP_NAME, startup_raw)
        source_sha = sha256_file(stage / SOURCE_NAME)
        wheelhouse_sha = sha256_file(stage / WHEELHOUSE_NAME)
        wheelhouse_manifest_sha = sha256_file(stage / WHEELHOUSE_MANIFEST_NAME)
        content_identity = canonical_sha256(
            {
                "source_archive_sha256": source_sha,
                "wheelhouse_archive_sha256": wheelhouse_sha,
                "wheelhouse_manifest_sha256": wheelhouse_manifest_sha,
                "startup_sha256": startup_sha,
            }
        )
        local_manifest_sha = sha256_file(base / local.MANIFEST_NAME)
        local_ready_sha = sha256_file(base / local.READY_NAME)
        plan = {
            "schema": EXECUTION_PLAN_SCHEMA,
            "status": CLOUD_PACKAGE_STATUS,
            "run_name": run_name,
            "identity_namespace": identity,
            "result_prefix": result_prefix,
            "project": receipt["runtime_preflight"]["observation"]["project"],
            "region": receipt["runtime_preflight"]["observation"]["region"],
            "zone": receipt["runtime_preflight"]["observation"]["zone"],
            "image": receipt["image_preflight"]["observation"],
            "image_identity_sha256": receipt["image_preflight"][
                "image_identity_sha256"
            ],
            "base_local_package": {
                "manifest_sha256": local_manifest_sha,
                "ready_sha256": local_ready_sha,
                "entry_count": len(source_entries),
                "entries": source_entries,
                "entries_sha256": canonical_sha256(source_entries),
            },
            "source_archive": {"path": SOURCE_NAME, "sha256": source_sha, "bytes": (stage / SOURCE_NAME).stat().st_size},
            "runtime_wheelhouse": {
                "path": WHEELHOUSE_NAME,
                "sha256": wheelhouse_sha,
                "bytes": (stage / WHEELHOUSE_NAME).stat().st_size,
                "manifest_path": WHEELHOUSE_MANIFEST_NAME,
                "manifest_sha256": wheelhouse_manifest_sha,
                "requirements_sha256": wheelhouse_manifest["requirements_sha256"],
                "offline_install_only": True,
            },
            "startup": {"path": STARTUP_NAME, "sha256": startup_sha, "bytes": len(startup_raw), "denylist_overlap_count": 0},
            "run_contract_digest": TAIL_RUN_CONTRACT_DIGEST,
            "run_contract_schema": TAIL_RUN_CONTRACT_SCHEMA,
            "run_contract_variant": TAIL_RUN_CONTRACT_VARIANT,
            "selection_manifest_sha256": TAIL_SELECTION_MANIFEST_SHA256,
            "source_roles": list(SOURCE_ROLES),
            "tail_hand_indices": list(TAIL_HAND_INDICES),
            "heavy_hand_indices": list(TAIL_HEAVY_HAND_INDICES),
            "random_hand_indices": list(TAIL_RANDOM_HAND_INDICES),
            "roadmap_amendment": {
                "schema": "hu_m31_t3_step6d_perfdev_v2_tail_v2_roadmap_amendment_v1",
                "reason": "old_profile_tail_is_mixed_geometry_and_not_tail_qualification",
                "superseded_mixed_diagnostic_hand_indices": list(
                    LEGACY_MIXED_DIAGNOSTIC_HAND_INDICES
                ),
                "qualification_hand_indices": list(TAIL_HAND_INDICES),
                "heavy_21x21_hand_indices": list(TAIL_HEAVY_HAND_INDICES),
                "random_hand_indices": list(TAIL_RANDOM_HAND_INDICES),
                "old_results_reused": False,
                "fresh_rerun_required": True,
            },
            "instances": _instance_specs(run_name, identity),
            "content_staging": {
                "identity_sha256": content_identity,
                "object_prefix": f"hu-m31-t3/perfdev-v2-content/{content_identity}/",
                "prestage_required_before_fresh_launch_overlay": True,
                "server_side_copy_into_run_control_required": True,
            },
            "worker_identity": {
                "service_account": WORKER_SERVICE_ACCOUNT,
                "oauth_scope": WORKER_OAUTH_SCOPE,
                "iam_preflight_required_before_launch": True,
            },
            "network": {
                "network_name": NETWORK_NAME,
                "subnetwork_name": SUBNETWORK_NAME,
                "external_ipv4": False,
                "private_ip_google_access_required": False,
                "nat_router_name": NAT_ROUTER_NAME,
                "nat_name": NAT_NAME,
                "nat_ip_allocate_option": "AUTO_ONLY",
                "source_subnetwork_ip_ranges_to_nat": "ALL_SUBNETWORKS_ALL_IP_RANGES",
                "launch_time_nat_get_required": True,
            },
            "feature_encoder": {
                "source_path": local.FEATURE_PACKAGE_PATH,
                "runtime_path": "payload/tooling/target/release/libofc_stage3_feature_encoder.so",
                "sha256": FEATURE_ENCODER_SHA256,
                "copy_not_symlink": True,
                "import_and_load_probe_required": True,
            },
            "allocation": {
                "machine_type": MACHINE_TYPE,
                "vm_count": VM_COUNT,
                "worker_processes_per_vm": WORKER_PROCESSES,
                "rayon_threads_per_process": RAYON_THREADS,
                "spot": True,
                "boot_disk_type": BOOT_DISK_TYPE,
                "boot_disk_interface": BOOT_DISK_INTERFACE,
                "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
                "network_nic_type": NETWORK_NIC_TYPE,
            },
            "limits": {"spot_price_ceiling_usd_per_vm_hour": SPOT_PRICE_CEILING_USD_PER_VM_HOUR, "maximum_total_compute_usd": MAX_TOTAL_COMPUTE_USD, "maximum_vm_runtime_seconds": MAX_VM_RUNTIME_SECONDS, "vm_ttl_seconds": VM_TTL_SECONDS},
            "heartbeat": {"interval_seconds": HEARTBEAT_SECONDS, "required_before_success": True, "immutable_objects": True},
            "checkpoint": {"enabled": True, "interval_seconds": HEARTBEAT_SECONDS, "per_hand_immutable": True, "done_published_last": True},
            "resume": {"enabled": False, "maximum_attempts_per_role": MAX_ATTEMPTS_PER_ROLE, "validated_archive_required": True, "fresh_one_shot_nonce_required": True, "reason": "attempt1_controller_not_yet_implemented"},
            "result_contract": {"manifest_schema": ROLE_RESULT_SCHEMA, "partial_result_is_success": False, "artifact_hash_required": True, "runner_validation_required": True, "cleanup_after_validation_only": True},
            "live_preflight": {"receipt_sha256": sha256_file(base / local.DRY_RECEIPT_PACKAGE_PATH), "observed_unix_seconds": _receipt_unix_seconds(receipt), "age_seconds_at_packaging": receipt_age, "maximum_age_seconds": MAX_LIVE_RECEIPT_AGE_SECONDS, "read_only": True, "collision_counts": {"run_name": 0, "identity": 0, "result_prefix": 0}},
            "cloud_executable": True,
            "launch_authorized": False,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        write_json_once(stage / PLAN_NAME, plan)
        plan_sha = sha256_file(stage / PLAN_NAME)
        package_entries = {
            SOURCE_NAME: {"sha256": source_sha, "bytes": (stage / SOURCE_NAME).stat().st_size},
            WHEELHOUSE_NAME: {"sha256": wheelhouse_sha, "bytes": (stage / WHEELHOUSE_NAME).stat().st_size},
            WHEELHOUSE_MANIFEST_NAME: {"sha256": wheelhouse_manifest_sha, "bytes": (stage / WHEELHOUSE_MANIFEST_NAME).stat().st_size},
            STARTUP_NAME: {"sha256": startup_sha, "bytes": len(startup_raw)},
            PLAN_NAME: {"sha256": plan_sha, "bytes": (stage / PLAN_NAME).stat().st_size},
        }
        manifest = {
            "schema": CLOUD_PACKAGE_SCHEMA,
            "status": CLOUD_PACKAGE_STATUS,
            "run_name": run_name,
            "entries": package_entries,
            "entry_count": len(package_entries),
            "entries_sha256": canonical_sha256([{"path": key, **value} for key, value in sorted(package_entries.items())]),
            "base_local_manifest_sha256": local_manifest_sha,
            "base_local_ready_sha256": local_ready_sha,
            "execution_plan_sha256": plan_sha,
            "source_archive_sha256": source_sha,
            "wheelhouse_archive_sha256": wheelhouse_sha,
            "wheelhouse_manifest_sha256": wheelhouse_manifest_sha,
            "startup_sha256": startup_sha,
            "postpackage_validation_required": True,
            "cloud_executable": True,
            "launch_authorized": False,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        write_json_once(stage / MANIFEST_NAME, manifest)
        ready = {
            "schema": CLOUD_READY_SCHEMA,
            "status": CLOUD_PACKAGE_STATUS,
            "run_name": run_name,
            "package_manifest_sha256": sha256_file(stage / MANIFEST_NAME),
            "execution_plan_sha256": plan_sha,
            "source_archive_sha256": source_sha,
            "wheelhouse_archive_sha256": wheelhouse_sha,
            "wheelhouse_manifest_sha256": wheelhouse_manifest_sha,
            "startup_sha256": startup_sha,
            "postpackage_validation_passed": True,
            "cloud_executable": True,
            "launch_authorized": False,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        write_json_once(stage / READY_NAME, ready)
        validate_cloud_package(stage, now_unix_seconds=now)
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_cloud_package(destination, now_unix_seconds=now)


def validate_execution_plan(
    value: Mapping[str, Any],
    *,
    embedded_local_package: str | Path | None = None,
    require_fresh_receipt: bool = True,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    plan = dict(value)
    if set(plan) != _plan_keys():
        raise ValueError("execution plan fields changed")
    run_name = plan.get("run_name")
    identity = plan.get("identity_namespace")
    prefix = plan.get("result_prefix")
    instances = plan.get("instances")
    source = plan.get("source_archive")
    wheelhouse = plan.get("runtime_wheelhouse")
    content_staging = plan.get("content_staging")
    startup = plan.get("startup")
    base_identity = plan.get("base_local_package")
    image = plan.get("image")
    if not isinstance(image, Mapping):
        raise ValueError("execution plan image observation is missing")
    preflight.validate_image_observation(image)
    expected_image_identity = contract_v1.canonical_sha256(
        {
            "project": image["project"],
            "name": image["name"],
            "id": image["id"],
            "selfLink": image["selfLink"],
            "status": image["status"],
            "deprecation": image["deprecation"],
            "guest_os_features": image["guest_os_features"],
        }
    )
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("status") != CLOUD_PACKAGE_STATUS
        or not isinstance(run_name, str) or _RUN.fullmatch(run_name) is None
        or not isinstance(identity, str) or _IDENTITY.fullmatch(identity) is None
        or prefix != f"hu-m31-t3/perfdev-v2/{run_name}/"
        or plan.get("run_contract_digest") != TAIL_RUN_CONTRACT_DIGEST
        or plan.get("run_contract_schema") != TAIL_RUN_CONTRACT_SCHEMA
        or plan.get("run_contract_variant") != TAIL_RUN_CONTRACT_VARIANT
        or plan.get("selection_manifest_sha256")
        != TAIL_SELECTION_MANIFEST_SHA256
        or plan.get("source_roles") != list(SOURCE_ROLES)
        or plan.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or plan.get("heavy_hand_indices") != list(TAIL_HEAVY_HAND_INDICES)
        or plan.get("random_hand_indices") != list(TAIL_RANDOM_HAND_INDICES)
        or plan.get("roadmap_amendment")
        != {
            "schema": "hu_m31_t3_step6d_perfdev_v2_tail_v2_roadmap_amendment_v1",
            "reason": "old_profile_tail_is_mixed_geometry_and_not_tail_qualification",
            "superseded_mixed_diagnostic_hand_indices": list(
                LEGACY_MIXED_DIAGNOSTIC_HAND_INDICES
            ),
            "qualification_hand_indices": list(TAIL_HAND_INDICES),
            "heavy_21x21_hand_indices": list(TAIL_HEAVY_HAND_INDICES),
            "random_hand_indices": list(TAIL_RANDOM_HAND_INDICES),
            "old_results_reused": False,
            "fresh_rerun_required": True,
        }
        or plan.get("image_identity_sha256") != expected_image_identity
        or plan.get("worker_identity") != {"service_account": WORKER_SERVICE_ACCOUNT, "oauth_scope": WORKER_OAUTH_SCOPE, "iam_preflight_required_before_launch": True}
        or plan.get("network") != {"network_name": NETWORK_NAME, "subnetwork_name": SUBNETWORK_NAME, "external_ipv4": False, "private_ip_google_access_required": False, "nat_router_name": NAT_ROUTER_NAME, "nat_name": NAT_NAME, "nat_ip_allocate_option": "AUTO_ONLY", "source_subnetwork_ip_ranges_to_nat": "ALL_SUBNETWORKS_ALL_IP_RANGES", "launch_time_nat_get_required": True}
        or plan.get("feature_encoder") != {"source_path": local.FEATURE_PACKAGE_PATH, "runtime_path": "payload/tooling/target/release/libofc_stage3_feature_encoder.so", "sha256": FEATURE_ENCODER_SHA256, "copy_not_symlink": True, "import_and_load_probe_required": True}
        or plan.get("allocation") != {
            "machine_type": MACHINE_TYPE,
            "vm_count": VM_COUNT,
            "worker_processes_per_vm": WORKER_PROCESSES,
            "rayon_threads_per_process": RAYON_THREADS,
            "spot": True,
            "boot_disk_type": BOOT_DISK_TYPE,
            "boot_disk_interface": BOOT_DISK_INTERFACE,
            "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
            "network_nic_type": NETWORK_NIC_TYPE,
        }
        or plan.get("limits") != {"spot_price_ceiling_usd_per_vm_hour": SPOT_PRICE_CEILING_USD_PER_VM_HOUR, "maximum_total_compute_usd": MAX_TOTAL_COMPUTE_USD, "maximum_vm_runtime_seconds": MAX_VM_RUNTIME_SECONDS, "vm_ttl_seconds": VM_TTL_SECONDS}
        or plan.get("heartbeat") != {"interval_seconds": HEARTBEAT_SECONDS, "required_before_success": True, "immutable_objects": True}
        or plan.get("checkpoint") != {"enabled": True, "interval_seconds": HEARTBEAT_SECONDS, "per_hand_immutable": True, "done_published_last": True}
        or plan.get("resume") != {"enabled": False, "maximum_attempts_per_role": MAX_ATTEMPTS_PER_ROLE, "validated_archive_required": True, "fresh_one_shot_nonce_required": True, "reason": "attempt1_controller_not_yet_implemented"}
        or plan.get("result_contract") != {"manifest_schema": ROLE_RESULT_SCHEMA, "partial_result_is_success": False, "artifact_hash_required": True, "runner_validation_required": True, "cleanup_after_validation_only": True}
        or any(plan.get(field) is not expected for field, expected in (("cloud_executable", True), ("launch_authorized", False), ("cloud_mutated", False), ("current_profile_changed", False)))
    ):
        raise ValueError("execution plan frozen boundary changed")
    if not isinstance(instances, list) or instances != _instance_specs(run_name, identity):
        raise ValueError("execution plan escaped exact candidate/reference VM pair")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256", "bytes"} or source.get("path") != SOURCE_NAME or _SHA.fullmatch(str(source.get("sha256"))) is None or not _plain_int(source.get("bytes")) or source["bytes"] <= 0:
        raise ValueError("source archive identity changed")
    if not isinstance(wheelhouse, Mapping) or set(wheelhouse) != {"path", "sha256", "bytes", "manifest_path", "manifest_sha256", "requirements_sha256", "offline_install_only"} or wheelhouse.get("path") != WHEELHOUSE_NAME or wheelhouse.get("manifest_path") != WHEELHOUSE_MANIFEST_NAME or _SHA.fullmatch(str(wheelhouse.get("sha256"))) is None or _SHA.fullmatch(str(wheelhouse.get("manifest_sha256"))) is None or _SHA.fullmatch(str(wheelhouse.get("requirements_sha256"))) is None or not _plain_int(wheelhouse.get("bytes")) or wheelhouse["bytes"] <= 0 or wheelhouse.get("offline_install_only") is not True:
        raise ValueError("offline runtime wheelhouse identity changed")
    expected_content_identity = canonical_sha256(
        {
            "source_archive_sha256": source["sha256"],
            "wheelhouse_archive_sha256": wheelhouse["sha256"],
            "wheelhouse_manifest_sha256": wheelhouse["manifest_sha256"],
            "startup_sha256": startup["sha256"] if isinstance(startup, Mapping) else None,
        }
    )
    if content_staging != {
        "identity_sha256": expected_content_identity,
        "object_prefix": f"hu-m31-t3/perfdev-v2-content/{expected_content_identity}/",
        "prestage_required_before_fresh_launch_overlay": True,
        "server_side_copy_into_run_control_required": True,
    }:
        raise ValueError("content-addressed staging contract changed")
    if not isinstance(startup, Mapping) or set(startup) != {"path", "sha256", "bytes", "denylist_overlap_count"} or startup.get("path") != STARTUP_NAME or _SHA.fullmatch(str(startup.get("sha256"))) is None or startup.get("sha256") in _FORBIDDEN_STARTUP_HASHES or startup.get("denylist_overlap_count") != 0 or not _plain_int(startup.get("bytes")) or startup["bytes"] <= 0:
        raise ValueError("startup identity changed")
    if not isinstance(base_identity, Mapping) or set(base_identity) != {"manifest_sha256", "ready_sha256", "entry_count", "entries", "entries_sha256"} or _SHA.fullmatch(str(base_identity.get("manifest_sha256"))) is None or _SHA.fullmatch(str(base_identity.get("ready_sha256"))) is None or not isinstance(base_identity.get("entries"), list) or base_identity.get("entry_count") != len(base_identity["entries"]) or base_identity.get("entries_sha256") != canonical_sha256(base_identity["entries"]):
        raise ValueError("embedded local package identity changed")
    live_preflight = plan.get("live_preflight")
    if not isinstance(live_preflight, Mapping) or live_preflight.get("read_only") is not True or live_preflight.get("collision_counts") != {"run_name": 0, "identity": 0, "result_prefix": 0} or live_preflight.get("maximum_age_seconds") != MAX_LIVE_RECEIPT_AGE_SECONDS or _SHA.fullmatch(str(live_preflight.get("receipt_sha256"))) is None or not _plain_int(live_preflight.get("observed_unix_seconds")):
        raise ValueError("live preflight binding changed")
    if require_fresh_receipt:
        now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
        age = now - live_preflight["observed_unix_seconds"]
        if age < -local.MAX_FUTURE_SKEW_SECONDS or age > MAX_LIVE_RECEIPT_AGE_SECONDS:
            raise ValueError("execution plan live receipt is stale or future-dated")
    if embedded_local_package is not None:
        base = Path(embedded_local_package)
        local_manifest, _runtime, receipt = _identity_from_local_package(base)
        rows = _safe_package_files(base)
        records = _entry_records(rows)
        if (
            sha256_file(base / local.MANIFEST_NAME) != base_identity["manifest_sha256"]
            or sha256_file(base / local.READY_NAME) != base_identity["ready_sha256"]
            or records != base_identity["entries"]
            or sha256_file(base / local.DRY_RECEIPT_PACKAGE_PATH) != live_preflight["receipt_sha256"]
            or _receipt_unix_seconds(receipt) != live_preflight["observed_unix_seconds"]
            or local_manifest.get("run_name") != run_name
        ):
            raise ValueError("embedded local package bytes differ from execution plan")
    return plan


def _load_cloud_package(
    directory: str | Path, *, now_unix_seconds: int | None, require_fresh_receipt: bool
) -> _PackageView:
    root = Path(directory).resolve(strict=True)
    manifest = _read_canonical(root / MANIFEST_NAME, "cloud package manifest")
    ready = _read_canonical(root / READY_NAME, "cloud package ready receipt")
    plan = _read_canonical(root / PLAN_NAME, "execution plan")
    expected_entries = {
        SOURCE_NAME,
        WHEELHOUSE_NAME,
        WHEELHOUSE_MANIFEST_NAME,
        STARTUP_NAME,
        PLAN_NAME,
    }
    entries = manifest.get("entries")
    if (
        manifest.get("schema") != CLOUD_PACKAGE_SCHEMA
        or manifest.get("status") != CLOUD_PACKAGE_STATUS
        or ready.get("schema") != CLOUD_READY_SCHEMA
        or ready.get("status") != CLOUD_PACKAGE_STATUS
        or not isinstance(entries, Mapping)
        or set(entries) != expected_entries
        or manifest.get("entry_count") != len(expected_entries)
        or any(manifest.get(field) is not expected for field, expected in (("cloud_executable", True), ("launch_authorized", False), ("cloud_mutated", False), ("current_profile_changed", False)))
        or any(ready.get(field) is not expected for field, expected in (("cloud_executable", True), ("launch_authorized", False), ("cloud_mutated", False), ("current_profile_changed", False)))
    ):
        raise ValueError("cloud package boundary changed")
    normalized: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_entries):
        record = entries[relative]
        path = root / relative
        if not isinstance(record, Mapping) or set(record) != {"sha256", "bytes"} or not path.is_file() or path.is_symlink() or sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            raise ValueError("cloud package entry changed")
        normalized[relative] = dict(record)
    aggregate = canonical_sha256([{"path": key, **value} for key, value in sorted(normalized.items())])
    manifest_sha = sha256_file(root / MANIFEST_NAME)
    plan_sha = sha256_file(root / PLAN_NAME)
    if (
        manifest.get("entries_sha256") != aggregate
        or manifest.get("execution_plan_sha256") != plan_sha
        or manifest.get("source_archive_sha256") != normalized[SOURCE_NAME]["sha256"]
        or manifest.get("wheelhouse_archive_sha256") != normalized[WHEELHOUSE_NAME]["sha256"]
        or manifest.get("wheelhouse_manifest_sha256") != normalized[WHEELHOUSE_MANIFEST_NAME]["sha256"]
        or manifest.get("startup_sha256") != normalized[STARTUP_NAME]["sha256"]
        or ready.get("package_manifest_sha256") != manifest_sha
        or ready.get("execution_plan_sha256") != plan_sha
        or ready.get("source_archive_sha256") != normalized[SOURCE_NAME]["sha256"]
        or ready.get("wheelhouse_archive_sha256") != normalized[WHEELHOUSE_NAME]["sha256"]
        or ready.get("wheelhouse_manifest_sha256") != normalized[WHEELHOUSE_MANIFEST_NAME]["sha256"]
        or ready.get("startup_sha256") != normalized[STARTUP_NAME]["sha256"]
        or ready.get("postpackage_validation_passed") is not True
    ):
        raise ValueError("cloud package aggregate identity changed")
    wheelhouse_manifest = _read_canonical(
        root / WHEELHOUSE_MANIFEST_NAME, "wheelhouse manifest"
    )
    if (
        plan.get("runtime_wheelhouse", {}).get("sha256")
        != normalized[WHEELHOUSE_NAME]["sha256"]
        or plan.get("runtime_wheelhouse", {}).get("manifest_sha256")
        != normalized[WHEELHOUSE_MANIFEST_NAME]["sha256"]
        or plan.get("runtime_wheelhouse", {}).get("requirements_sha256")
        != wheelhouse_manifest.get("requirements_sha256")
    ):
        raise ValueError("execution plan/wheelhouse binding changed")
    _validate_wheelhouse_archive(root / WHEELHOUSE_NAME, wheelhouse_manifest)
    base_identity = plan.get("base_local_package")
    if not isinstance(base_identity, Mapping) or not isinstance(base_identity.get("entries"), list):
        raise ValueError("source entry manifest is missing")
    _validate_source_archive_without_extraction(root / SOURCE_NAME, plan=plan)
    validate_execution_plan(
        plan,
        require_fresh_receipt=require_fresh_receipt,
        now_unix_seconds=now_unix_seconds,
    )
    if (
        manifest.get("base_local_manifest_sha256") != plan["base_local_package"]["manifest_sha256"]
        or manifest.get("base_local_ready_sha256") != plan["base_local_package"]["ready_sha256"]
        or manifest.get("run_name") != plan["run_name"]
        or ready.get("run_name") != plan["run_name"]
    ):
        raise ValueError("cloud/local package binding changed")
    return _PackageView(root, manifest, ready, plan, plan_sha, normalized[SOURCE_NAME]["sha256"], normalized[STARTUP_NAME]["sha256"], manifest_sha)


def validate_cloud_package(
    directory: str | Path,
    *,
    now_unix_seconds: int | None = None,
    require_fresh_receipt: bool = True,
) -> dict[str, Any]:
    view = _load_cloud_package(directory, now_unix_seconds=now_unix_seconds, require_fresh_receipt=require_fresh_receipt)
    return {
        "schema": CLOUD_READY_SCHEMA,
        "status": CLOUD_PACKAGE_STATUS,
        "run_name": view.plan["run_name"],
        "package_manifest_sha256": view.package_manifest_sha256,
        "execution_plan_sha256": view.plan_sha256,
        "source_archive_sha256": view.source_sha256,
        "wheelhouse_archive_sha256": view.plan["runtime_wheelhouse"]["sha256"],
        "wheelhouse_manifest_sha256": view.plan["runtime_wheelhouse"]["manifest_sha256"],
        "startup_sha256": view.startup_sha256,
        "instance_names": [row["instance_name"] for row in view.plan["instances"]],
        "source_roles": list(SOURCE_ROLES),
        "run_contract_digest": TAIL_RUN_CONTRACT_DIGEST,
        "run_contract_schema": TAIL_RUN_CONTRACT_SCHEMA,
        "run_contract_variant": TAIL_RUN_CONTRACT_VARIANT,
        "selection_manifest_sha256": TAIL_SELECTION_MANIFEST_SHA256,
        "tail_hand_indices": list(TAIL_HAND_INDICES),
        "heavy_hand_indices": list(TAIL_HEAVY_HAND_INDICES),
        "random_hand_indices": list(TAIL_RANDOM_HAND_INDICES),
        "cloud_executable": True,
        "launch_authorized": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }


def build_content_stage_authorization(
    *,
    cloud_package_dir: str | Path,
    explicit_stage_authorized: bool,
    one_shot_nonce: str,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    if explicit_stage_authorized is not True:
        raise PermissionError("explicit content-stage authorization is required")
    parsed = uuid.UUID(one_shot_nonce)
    if parsed.version != 4 or str(parsed) != one_shot_nonce:
        raise ValueError("content-stage nonce must be a canonical UUIDv4")
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    view = _load_cloud_package(
        cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False
    )
    unsigned = {
        "schema": CONTENT_STAGE_AUTHORIZATION_SCHEMA,
        "status": "explicit_content_stage_authorized_once",
        "run_name": view.plan["run_name"],
        "package_manifest_sha256": view.package_manifest_sha256,
        "execution_plan_sha256": view.plan_sha256,
        "content_prefix": view.plan["content_staging"]["object_prefix"],
        "one_shot_nonce_sha256": hashlib.sha256(one_shot_nonce.encode("ascii")).hexdigest(),
        "authorized_unix_seconds": now,
        "expires_unix_seconds": now + 900,
        "explicit_stage_authorized": True,
        "vm_launch_authorized": False,
        "current_profile_changed": False,
    }
    return {**unsigned, "authorization_content_sha256": canonical_sha256(unsigned)}


def _validate_content_stage_authorization(
    value: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    raw_nonce: str,
    now_unix_seconds: int | None,
) -> dict[str, Any]:
    auth = dict(value)
    digest = auth.pop("authorization_content_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    if digest != canonical_sha256(auth):
        raise ValueError("content-stage authorization hash changed")
    if (
        auth.get("schema") != CONTENT_STAGE_AUTHORIZATION_SCHEMA
        or auth.get("status") != "explicit_content_stage_authorized_once"
        or auth.get("run_name") != plan["run_name"]
        or auth.get("execution_plan_sha256") != canonical_sha256(plan)
        or auth.get("content_prefix") != plan["content_staging"]["object_prefix"]
        or auth.get("one_shot_nonce_sha256") != hashlib.sha256(raw_nonce.encode("ascii")).hexdigest()
        or auth.get("explicit_stage_authorized") is not True
        or auth.get("vm_launch_authorized") is not False
        or auth.get("current_profile_changed") is not False
        or not _plain_int(auth.get("authorized_unix_seconds"))
        or not _plain_int(auth.get("expires_unix_seconds"))
        or auth["expires_unix_seconds"] - auth["authorized_unix_seconds"] != 900
    ):
        raise ValueError("content-stage authorization escaped immutable package")
    if now_unix_seconds is not None and not (
        auth["authorized_unix_seconds"]
        <= now_unix_seconds
        <= auth["expires_unix_seconds"]
    ):
        raise PermissionError("content-stage authorization expired")
    return {**auth, "authorization_content_sha256": digest}


def stage_content(
    *,
    cloud_package_dir: str | Path,
    authorization: Mapping[str, Any],
    raw_one_shot_nonce: str,
    transport: CloudTransport,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    view = _load_cloud_package(
        cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False
    )
    auth = _validate_content_stage_authorization(
        authorization,
        execution_plan=view.plan,
        raw_nonce=raw_one_shot_nonce,
        now_unix_seconds=now,
    )
    if auth["package_manifest_sha256"] != view.package_manifest_sha256:
        raise ValueError("content-stage package manifest hash changed")
    prefix = view.plan["content_staging"]["object_prefix"]
    objects = []
    for relative in (SOURCE_NAME, WHEELHOUSE_NAME, WHEELHOUSE_MANIFEST_NAME):
        objects.append(
            _put_new(
                transport,
                f"{prefix}{relative}",
                (view.directory / relative).read_bytes(),
            )
        )
    unsigned = {
        "schema": CONTENT_STAGE_RECEIPT_SCHEMA,
        "status": "content_addressed_package_staged_complete",
        "run_name": view.plan["run_name"],
        "package_manifest_sha256": view.package_manifest_sha256,
        "execution_plan_sha256": view.plan_sha256,
        "content_prefix": prefix,
        "authorization_sha256": canonical_sha256(auth),
        "objects": objects,
        "object_count": len(objects),
        "all_objects_created_without_collision": True,
        "vm_started": False,
        "result_prefix_mutated": False,
        "current_profile_changed": False,
    }
    return _receipt_with_digest(unsigned)


def _validate_content_stage_receipt(
    value: Mapping[str, Any], *, view: _PackageView
) -> dict[str, Any]:
    receipt = dict(value)
    digest = receipt.pop("receipt_content_sha256", None)
    objects = receipt.get("objects")
    expected_prefix = view.plan["content_staging"]["object_prefix"]
    expected = {
        f"{expected_prefix}{SOURCE_NAME}": (view.source_sha256, (view.directory / SOURCE_NAME).stat().st_size),
        f"{expected_prefix}{WHEELHOUSE_NAME}": (view.plan["runtime_wheelhouse"]["sha256"], (view.directory / WHEELHOUSE_NAME).stat().st_size),
        f"{expected_prefix}{WHEELHOUSE_MANIFEST_NAME}": (view.plan["runtime_wheelhouse"]["manifest_sha256"], (view.directory / WHEELHOUSE_MANIFEST_NAME).stat().st_size),
    }
    if (
        digest != canonical_sha256(receipt)
        or receipt.get("schema") != CONTENT_STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "content_addressed_package_staged_complete"
        or receipt.get("run_name") != view.plan["run_name"]
        or receipt.get("package_manifest_sha256") != view.package_manifest_sha256
        or receipt.get("execution_plan_sha256") != view.plan_sha256
        or receipt.get("content_prefix") != expected_prefix
        or not isinstance(objects, list)
        or receipt.get("object_count") != len(expected)
        or receipt.get("all_objects_created_without_collision") is not True
        or receipt.get("vm_started") is not False
        or receipt.get("result_prefix_mutated") is not False
    ):
        raise ValueError("content-stage receipt is incomplete or changed")
    observed = {row.get("object_name"): row for row in objects if isinstance(row, Mapping)}
    if set(observed) != set(expected):
        raise ValueError("content-stage object topology changed")
    for name, (digest_value, size) in expected.items():
        row = observed[name]
        if row.get("created") is not True or row.get("sha256") != digest_value or row.get("bytes") != size:
            raise ValueError("content-stage object identity changed")
    return {**receipt, "receipt_content_sha256": digest}


def validate_worker_iam_readback(
    value: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    expected_one_shot_nonce_sha256: str,
    now_unix_seconds: int | None,
) -> dict[str, Any]:
    if _SHA.fullmatch(str(expected_one_shot_nonce_sha256)) is None:
        raise ValueError("expected worker IAM one-shot nonce SHA-256 changed")
    receipt = dict(value)
    digest = receipt.pop("receipt_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    launch = receipt.get("launch_preflight")
    project_zero = receipt.get("project_worker_zero_readback")
    expected_keys = {
        "schema", "status", "iam_plan_sha256", "execution_plan_sha256",
        "one_shot_nonce_sha256", "prepare_receipt_sha256",
        "install_receipt_sha256", "role_readbacks",
        "project_worker_zero_readback", "bucket_policy_fingerprint_sha256",
        "bucket_policy_etag_sha256", "unrelated_policy_fingerprint_sha256",
        "exact_binding_count", "observed_unix_seconds", "launch_preflight",
        "cloud_mutation_performed", "current_profile_changed",
    }
    launch_keys = {
        "read_only", "service_account", "bucket", "required_reader_prefix",
        "required_creator_prefix", "exact_conditional_binding_count",
        "reader_binding_count", "creator_binding_count",
        "excess_worker_binding_count", "iam_expiry_unix_seconds",
        "object_get_allowed", "object_create_allowed", "object_list_required",
        "exact_prefix_condition", "all_required_permissions_present",
    }
    if (
        digest != canonical_sha256(receipt)
        or set(receipt) != expected_keys
        or receipt.get("schema") != WORKER_IAM_READBACK_SCHEMA
        or receipt.get("status")
        != "launch_preflight_exact_two_bindings_validated"
        or receipt.get("execution_plan_sha256") != canonical_sha256(plan)
        or receipt.get("one_shot_nonce_sha256")
        != expected_one_shot_nonce_sha256
        or any(
            _SHA.fullmatch(str(receipt.get(field))) is None
            for field in (
                "iam_plan_sha256", "prepare_receipt_sha256",
                "install_receipt_sha256", "bucket_policy_fingerprint_sha256",
                "bucket_policy_etag_sha256",
                "unrelated_policy_fingerprint_sha256",
            )
        )
        or receipt.get("exact_binding_count") != 2
        or not _plain_int(receipt.get("observed_unix_seconds"))
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
        or not isinstance(launch, Mapping)
        or set(launch) != launch_keys
        or launch.get("read_only") is not True
        or launch.get("service_account") != WORKER_SERVICE_ACCOUNT
        or launch.get("bucket") != WORKER_BUCKET
        or launch.get("required_reader_prefix")
        != f"{plan['result_prefix']}control/"
        or launch.get("required_creator_prefix") != plan["result_prefix"]
        or launch.get("exact_conditional_binding_count") != 2
        or launch.get("reader_binding_count") != 1
        or launch.get("creator_binding_count") != 1
        or launch.get("excess_worker_binding_count") != 0
        or not _plain_int(launch.get("iam_expiry_unix_seconds"))
        or launch.get("object_get_allowed") is not True
        or launch.get("object_create_allowed") is not True
        or launch.get("object_list_required") is not False
        or launch.get("exact_prefix_condition") is not True
        or launch.get("all_required_permissions_present") is not True
        or not (
            MIN_WORKER_IAM_WINDOW_SECONDS
            <= launch["iam_expiry_unix_seconds"] - receipt["observed_unix_seconds"]
            <= MAX_WORKER_IAM_WINDOW_SECONDS
        )
    ):
        raise ValueError("worker IAM lifecycle readback changed or is incomplete")
    # Reuse the producer's structural validators without importing it while this
    # module is itself being imported (the IAM module imports this controller).
    from . import (  # pylint: disable=import-outside-toplevel
        hu_m31_t3_step6d_performance_development_v2_worker_iam as worker_iam,
    )

    worker_iam._validate_role_readbacks(receipt["role_readbacks"])
    worker_iam._validate_project_worker_zero_readback(project_zero)
    if now_unix_seconds is not None and not (
        0 <= now_unix_seconds - receipt["observed_unix_seconds"] <= MAX_LIVE_RECEIPT_AGE_SECONDS
    ):
        raise PermissionError("worker IAM lifecycle readback is stale or future-dated")
    return {**receipt, "receipt_sha256": digest}


def build_fresh_launch_overlay(
    *,
    cloud_package_dir: str | Path,
    content_stage_receipt: Mapping[str, Any],
    worker_iam_readback: Mapping[str, Any],
    expected_one_shot_nonce_sha256: str,
    fresh_dry_run_receipt_path: str | Path,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    """Bind a fresh lightweight GET receipt after heavy content staging."""

    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    view = _load_cloud_package(
        cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False
    )
    staged = _validate_content_stage_receipt(content_stage_receipt, view=view)
    iam_readback = validate_worker_iam_readback(
        worker_iam_readback,
        execution_plan=view.plan,
        expected_one_shot_nonce_sha256=expected_one_shot_nonce_sha256,
        now_unix_seconds=now,
    )
    receipt = _read_canonical(fresh_dry_run_receipt_path, "fresh launch dry-run receipt")
    preflight.validate_dry_run_receipt(receipt)
    _require_fresh_receipt(receipt, now_unix_seconds=now)
    namespace = receipt["runtime_preflight"]["observation"]["namespace"]
    if (
        namespace["run_name"] != view.plan["run_name"]
        or namespace["identity_namespace"] != view.plan["identity_namespace"]
        or namespace["result_prefix"] != view.plan["result_prefix"]
    ):
        raise ValueError("fresh launch overlay namespace differs from package")
    unsigned = {
        "schema": LAUNCH_OVERLAY_SCHEMA,
        "status": "fresh_launch_overlay_ready_not_authorized",
        "run_name": view.plan["run_name"],
        "package_manifest_sha256": view.package_manifest_sha256,
        "execution_plan_sha256": view.plan_sha256,
        "content_stage_receipt_sha256": canonical_sha256(staged),
        "content_prefix": staged["content_prefix"],
        "fresh_dry_run_receipt": receipt,
        "fresh_dry_run_receipt_sha256": canonical_sha256(receipt),
        "observed_unix_seconds": _receipt_unix_seconds(receipt),
        "expires_unix_seconds": _receipt_unix_seconds(receipt) + MAX_LIVE_RECEIPT_AGE_SECONDS,
        "overlay_built_unix_seconds": now,
        "worker_iam_expires_unix_seconds": iam_readback["launch_preflight"]["iam_expiry_unix_seconds"],
        "worker_iam_minimum_window_seconds": MIN_WORKER_IAM_WINDOW_SECONDS,
        "worker_iam_maximum_window_seconds": MAX_WORKER_IAM_WINDOW_SECONDS,
        "worker_iam_readback": iam_readback,
        "worker_iam_readback_receipt_sha256": iam_readback["receipt_sha256"],
        "worker_iam_execution_plan_sha256": iam_readback[
            "execution_plan_sha256"
        ],
        "worker_iam_one_shot_nonce_sha256": iam_readback[
            "one_shot_nonce_sha256"
        ],
        "worker_iam_observed_unix_seconds": iam_readback[
            "observed_unix_seconds"
        ],
        "heavy_content_upload_after_observation": False,
        "server_side_copy_only_after_observation": True,
        "launch_authorized": False,
        "cloud_mutated": False,
    }
    return {**unsigned, "overlay_content_sha256": canonical_sha256(unsigned)}


def _validate_launch_overlay(
    value: Mapping[str, Any], *, view: _PackageView, now_unix_seconds: int | None
) -> dict[str, Any]:
    overlay = dict(value)
    digest = overlay.pop("overlay_content_sha256", None)
    receipt = overlay.get("fresh_dry_run_receipt")
    if (
        digest != canonical_sha256(overlay)
        or overlay.get("schema") != LAUNCH_OVERLAY_SCHEMA
        or overlay.get("status") != "fresh_launch_overlay_ready_not_authorized"
        or overlay.get("run_name") != view.plan["run_name"]
        or overlay.get("package_manifest_sha256") != view.package_manifest_sha256
        or overlay.get("execution_plan_sha256") != view.plan_sha256
        or overlay.get("content_prefix") != view.plan["content_staging"]["object_prefix"]
        or not isinstance(receipt, Mapping)
        or overlay.get("fresh_dry_run_receipt_sha256") != canonical_sha256(receipt)
        or overlay.get("observed_unix_seconds") != _receipt_unix_seconds(receipt)
        or overlay.get("expires_unix_seconds") != _receipt_unix_seconds(receipt) + MAX_LIVE_RECEIPT_AGE_SECONDS
        or not _plain_int(overlay.get("overlay_built_unix_seconds"))
        or not isinstance(overlay.get("worker_iam_readback"), Mapping)
        or overlay.get("worker_iam_readback_receipt_sha256")
        != overlay["worker_iam_readback"].get("receipt_sha256")
        or overlay.get("worker_iam_execution_plan_sha256")
        != view.plan_sha256
        or overlay.get("worker_iam_execution_plan_sha256")
        != overlay["worker_iam_readback"].get("execution_plan_sha256")
        or overlay.get("worker_iam_one_shot_nonce_sha256")
        != overlay["worker_iam_readback"].get("one_shot_nonce_sha256")
        or overlay.get("worker_iam_observed_unix_seconds")
        != overlay["worker_iam_readback"].get("observed_unix_seconds")
        or overlay.get("worker_iam_expires_unix_seconds")
        != overlay["worker_iam_readback"].get("launch_preflight", {}).get(
            "iam_expiry_unix_seconds"
        )
        or overlay.get("worker_iam_minimum_window_seconds") != MIN_WORKER_IAM_WINDOW_SECONDS
        or overlay.get("worker_iam_maximum_window_seconds") != MAX_WORKER_IAM_WINDOW_SECONDS
        or not (
            MIN_WORKER_IAM_WINDOW_SECONDS
            <= overlay["worker_iam_expires_unix_seconds"] - overlay["overlay_built_unix_seconds"]
            <= MAX_WORKER_IAM_WINDOW_SECONDS
        )
        or overlay.get("heavy_content_upload_after_observation") is not False
        or overlay.get("server_side_copy_only_after_observation") is not True
        or overlay.get("launch_authorized") is not False
        or overlay.get("cloud_mutated") is not False
    ):
        raise ValueError("fresh launch overlay changed")
    preflight.validate_dry_run_receipt(receipt)
    validate_worker_iam_readback(
        overlay["worker_iam_readback"],
        execution_plan=view.plan,
        expected_one_shot_nonce_sha256=overlay["worker_iam_readback"].get(
            "one_shot_nonce_sha256"
        ),
        now_unix_seconds=now_unix_seconds,
    )
    namespace = receipt["runtime_preflight"]["observation"]["namespace"]
    if namespace != {
        "run_name": view.plan["run_name"],
        "identity_namespace": view.plan["identity_namespace"],
        "result_prefix": view.plan["result_prefix"],
        "run_name_collision_count": 0,
        "identity_collision_count": 0,
        "result_prefix_collision_count": 0,
        "inventory_read_only": True,
    }:
        raise ValueError("fresh launch overlay namespace is not empty and exact")
    if now_unix_seconds is not None and not (
        overlay["observed_unix_seconds"] - local.MAX_FUTURE_SKEW_SECONDS
        <= now_unix_seconds
        <= overlay["expires_unix_seconds"]
    ):
        raise PermissionError("fresh launch overlay expired before VM creation")
    return {**overlay, "overlay_content_sha256": digest}


def build_launch_authorization(
    *,
    cloud_package_dir: str | Path,
    launch_overlay: Mapping[str, Any],
    explicit_launch_authorized: bool,
    one_shot_nonce: str,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    if explicit_launch_authorized is not True:
        raise PermissionError("explicit launch authorization flag is required")
    parsed = uuid.UUID(one_shot_nonce)
    if parsed.version != 4 or str(parsed) != one_shot_nonce:
        raise ValueError("one-shot nonce must be a canonical UUIDv4")
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    view = _load_cloud_package(cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False)
    overlay = _validate_launch_overlay(
        launch_overlay, view=view, now_unix_seconds=now
    )
    if overlay["worker_iam_readback"]["one_shot_nonce_sha256"] != hashlib.sha256(
        one_shot_nonce.encode("ascii")
    ).hexdigest():
        raise ValueError("worker IAM readback belongs to another launch nonce")
    receipt_deadline = overlay["expires_unix_seconds"]
    expires = min(now + MAX_AUTHORIZATION_LIFETIME_SECONDS, receipt_deadline)
    if expires <= now:
        raise ValueError("live receipt leaves no authorization lifetime")
    unsigned = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_pair_launch_authorized_once",
        "run_name": view.plan["run_name"],
        "identity_namespace": view.plan["identity_namespace"],
        "result_prefix": view.plan["result_prefix"],
        "package_manifest_sha256": view.package_manifest_sha256,
        "execution_plan_sha256": view.plan_sha256,
        "source_archive_sha256": view.source_sha256,
        "wheelhouse_archive_sha256": view.plan["runtime_wheelhouse"]["sha256"],
        "wheelhouse_manifest_sha256": view.plan["runtime_wheelhouse"]["manifest_sha256"],
        "startup_sha256": view.startup_sha256,
        "launch_overlay_sha256": canonical_sha256(overlay),
        "content_stage_receipt_sha256": overlay["content_stage_receipt_sha256"],
        "worker_iam_readback_receipt_sha256": overlay[
            "worker_iam_readback_receipt_sha256"
        ],
        "authorized_roles": list(SOURCE_ROLES),
        "authorized_instance_names": [row["instance_name"] for row in view.plan["instances"]],
        "attempt_index": 0,
        "resume_archives": {role: None for role in SOURCE_ROLES},
        "one_shot_nonce_sha256": hashlib.sha256(one_shot_nonce.encode("ascii")).hexdigest(),
        "authorized_unix_seconds": now,
        "expires_unix_seconds": expires,
        "explicit_launch_authorized": True,
        "one_shot": True,
        "tail_only": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    return {**unsigned, "authorization_content_sha256": canonical_sha256(unsigned)}


def _validate_launch_authorization(
    value: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    raw_nonce: str | None,
    now_unix_seconds: int | None,
    allow_runtime_nonce_hash_only: bool,
) -> dict[str, Any]:
    authorization = dict(value)
    content_digest = authorization.pop("authorization_content_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    if content_digest != canonical_sha256(authorization):
        raise ValueError("launch authorization content hash changed")
    expected_keys = {
        "schema", "status", "run_name", "identity_namespace", "result_prefix",
        "package_manifest_sha256", "execution_plan_sha256", "source_archive_sha256",
        "wheelhouse_archive_sha256", "wheelhouse_manifest_sha256", "startup_sha256", "authorized_roles", "authorized_instance_names",
        "launch_overlay_sha256", "content_stage_receipt_sha256",
        "worker_iam_readback_receipt_sha256",
        "attempt_index", "resume_archives", "one_shot_nonce_sha256",
        "authorized_unix_seconds", "expires_unix_seconds",
        "explicit_launch_authorized", "one_shot", "tail_only",
        "production_fanout_authorized", "training_eligible", "current_profile_changed",
    }
    if set(authorization) != expected_keys:
        raise ValueError("launch authorization fields changed")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "explicit_pair_launch_authorized_once"
        or authorization.get("run_name") != plan["run_name"]
        or authorization.get("identity_namespace") != plan["identity_namespace"]
        or authorization.get("result_prefix") != plan["result_prefix"]
        or authorization.get("execution_plan_sha256") != canonical_sha256(plan)
        or authorization.get("source_archive_sha256") != plan["source_archive"]["sha256"]
        or authorization.get("wheelhouse_archive_sha256") != plan["runtime_wheelhouse"]["sha256"]
        or authorization.get("wheelhouse_manifest_sha256") != plan["runtime_wheelhouse"]["manifest_sha256"]
        or authorization.get("startup_sha256") != plan["startup"]["sha256"]
        or _SHA.fullmatch(str(authorization.get("launch_overlay_sha256"))) is None
        or _SHA.fullmatch(str(authorization.get("content_stage_receipt_sha256"))) is None
        or _SHA.fullmatch(
            str(authorization.get("worker_iam_readback_receipt_sha256"))
        )
        is None
        or authorization.get("authorized_roles") != list(SOURCE_ROLES)
        or authorization.get("authorized_instance_names") != [row["instance_name"] for row in plan["instances"]]
        or authorization.get("attempt_index") != 0
        or authorization.get("resume_archives") != {role: None for role in SOURCE_ROLES}
        or _SHA.fullmatch(str(authorization.get("one_shot_nonce_sha256"))) is None
        or any(authorization.get(field) is not expected for field, expected in (("explicit_launch_authorized", True), ("one_shot", True), ("tail_only", True), ("production_fanout_authorized", False), ("training_eligible", False), ("current_profile_changed", False)))
        or not _plain_int(authorization.get("authorized_unix_seconds"))
        or not _plain_int(authorization.get("expires_unix_seconds"))
        or authorization["expires_unix_seconds"] <= authorization["authorized_unix_seconds"]
        or authorization["expires_unix_seconds"] - authorization["authorized_unix_seconds"] > MAX_AUTHORIZATION_LIFETIME_SECONDS
    ):
        raise ValueError("launch authorization escaped frozen pair")
    if raw_nonce is None:
        if not allow_runtime_nonce_hash_only:
            raise ValueError("raw one-shot nonce is required")
    else:
        try:
            parsed = uuid.UUID(raw_nonce)
        except ValueError as exc:
            raise ValueError("one-shot nonce is invalid") from exc
        if parsed.version != 4 or str(parsed) != raw_nonce or hashlib.sha256(raw_nonce.encode("ascii")).hexdigest() != authorization["one_shot_nonce_sha256"]:
            raise ValueError("one-shot nonce does not match authorization")
    if now_unix_seconds is not None:
        if now_unix_seconds < authorization["authorized_unix_seconds"] or now_unix_seconds > authorization["expires_unix_seconds"]:
            raise PermissionError("launch authorization is not currently valid")
    return {**authorization, "authorization_content_sha256": content_digest}


def validate_launch_authorization(
    value: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    raw_nonce: str,
    now_unix_seconds: int | None,
) -> dict[str, Any]:
    """Controller-side validation always requires the raw one-shot nonce."""

    return _validate_launch_authorization(
        value,
        execution_plan=execution_plan,
        raw_nonce=raw_nonce,
        now_unix_seconds=now_unix_seconds,
        allow_runtime_nonce_hash_only=False,
    )


def validate_runtime_binding(
    execution_plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    auth = _validate_launch_authorization(
        authorization,
        execution_plan=plan,
        raw_nonce=None,
        now_unix_seconds=None,
        allow_runtime_nonce_hash_only=True,
    )
    role = expected.get("source_role")
    instance = next((row for row in plan["instances"] if row["source_role"] == role), None)
    bucket = expected.get("bucket")
    if not isinstance(bucket, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,62}", bucket) is None:
        raise ValueError("runtime bucket binding changed")
    control_prefix = f"{plan['result_prefix']}control/"
    if not isinstance(instance, Mapping) or dict(expected) != {
        "project_id": plan["project"],
        "bucket": bucket,
        "run_name": plan["run_name"],
        "identity_namespace": plan["identity_namespace"],
        "result_prefix": plan["result_prefix"],
        "source_role": role,
        "instance_name": instance["instance_name"],
        "source_object": f"{control_prefix}{SOURCE_NAME}",
        "source_sha256": plan["source_archive"]["sha256"],
        "wheelhouse_object": f"{control_prefix}{WHEELHOUSE_NAME}",
        "wheelhouse_sha256": plan["runtime_wheelhouse"]["sha256"],
        "wheelhouse_manifest_sha256": plan["runtime_wheelhouse"]["manifest_sha256"],
        "plan_object": f"{control_prefix}{PLAN_NAME}",
        "plan_sha256": canonical_sha256(plan),
        "authorization_object": f"{control_prefix}launch_authorization.json",
        "authorization_sha256": canonical_sha256(auth),
        "authorization_nonce_sha256": auth["one_shot_nonce_sha256"],
        "startup_sha256": plan["startup"]["sha256"],
        "ownership_label": instance["ownership_label"],
        "attempt_index": auth["attempt_index"],
        "resume_object": "none",
        "resume_sha256": "none",
        "max_runtime_seconds": MAX_VM_RUNTIME_SECONDS,
        "heartbeat_seconds": HEARTBEAT_SECONDS,
    }:
        raise ValueError("runtime metadata binding changed")
    return dict(expected)


def _validate_namespace_observation(value: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    observation = dict(value)
    expected_names = [row["instance_name"] for row in plan["instances"]]
    worker_iam = observation.get("worker_iam")
    if set(observation) != {"read_only", "cloud_mutated", "run_name_collision_count", "identity_collision_count", "result_prefix_object_count", "instance_collision_counts", "worker_iam", "network_path"} or observation.get("read_only") is not True or observation.get("cloud_mutated") is not False or observation.get("run_name_collision_count") != 0 or observation.get("identity_collision_count") != 0 or observation.get("result_prefix_object_count") != 0 or observation.get("instance_collision_counts") != {name: 0 for name in expected_names}:
        raise ValueError("launch-time namespace collision or observation drift")
    if observation.get("network_path") != {
        "read_only": True,
        "router_name": NAT_ROUTER_NAME,
        "router_region": plan["region"],
        "router_network_exact": True,
        "nat_name": NAT_NAME,
        "nat_count_with_name": 1,
        "nat_ip_allocate_option": "AUTO_ONLY",
        "source_subnetwork_ip_ranges_to_nat": "ALL_SUBNETWORKS_ALL_IP_RANGES",
        "external_ipv4_on_vm": False,
        "path_ready": True,
    }:
        raise PermissionError("launch-time Cloud NAT path GET preflight failed")
    if (
        not isinstance(worker_iam, Mapping)
        or set(worker_iam)
        != {
            "read_only",
            "service_account",
            "bucket",
            "required_reader_prefix",
            "required_creator_prefix",
            "exact_conditional_binding_count",
            "reader_binding_count",
            "creator_binding_count",
            "excess_worker_binding_count",
            "iam_expiry_unix_seconds",
            "object_get_allowed",
            "object_create_allowed",
            "object_list_required",
            "exact_prefix_condition",
            "all_required_permissions_present",
        }
        or worker_iam.get("read_only") is not True
        or worker_iam.get("service_account") != WORKER_SERVICE_ACCOUNT
        or worker_iam.get("required_reader_prefix") != f"{plan['result_prefix']}control/"
        or worker_iam.get("required_creator_prefix") != plan["result_prefix"]
        or not isinstance(worker_iam.get("bucket"), str)
        or not _plain_int(worker_iam.get("exact_conditional_binding_count"))
        or worker_iam.get("exact_conditional_binding_count") != 2
        or worker_iam.get("reader_binding_count") != 1
        or worker_iam.get("creator_binding_count") != 1
        or worker_iam.get("excess_worker_binding_count") != 0
        or not _plain_int(worker_iam.get("iam_expiry_unix_seconds"))
        or worker_iam.get("object_get_allowed") is not True
        or worker_iam.get("object_create_allowed") is not True
        or worker_iam.get("object_list_required") is not False
        or worker_iam.get("exact_prefix_condition") is not True
        or worker_iam.get("all_required_permissions_present") is not True
    ):
        raise PermissionError("worker IAM GET-only preflight is not satisfied")
    return observation


def _put_new(transport: CloudTransport, object_name: str, payload: bytes) -> dict[str, Any]:
    receipt = dict(transport.put_if_absent(object_name=object_name, payload=payload))
    if set(receipt) != {"created", "object_name", "sha256", "bytes", "generation"} or receipt.get("created") is not True or receipt.get("object_name") != object_name or receipt.get("sha256") != hashlib.sha256(payload).hexdigest() or receipt.get("bytes") != len(payload) or not isinstance(receipt.get("generation"), str) or not receipt["generation"].isdigit():
        raise RuntimeError(f"immutable object publish failed or collided: {object_name}")
    return receipt


def _copy_new(
    transport: CloudTransport,
    *,
    source_object: str,
    destination_object: str,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    receipt = dict(
        transport.copy_if_absent(
            source_object=source_object,
            destination_object=destination_object,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
    )
    if (
        set(receipt)
        != {
            "created",
            "source_object",
            "object_name",
            "sha256",
            "bytes",
            "generation",
        }
        or receipt.get("created") is not True
        or receipt.get("source_object") != source_object
        or receipt.get("object_name") != destination_object
        or receipt.get("sha256") != expected_sha256
        or receipt.get("bytes") != expected_bytes
        or not isinstance(receipt.get("generation"), str)
        or not receipt["generation"].isdigit()
    ):
        raise RuntimeError("immutable staged-to-control copy failed or collided")
    return receipt


def _instance_spec(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    role_row: Mapping[str, Any],
    *,
    source_object: str,
    wheelhouse_object: str,
    plan_object: str,
    authorization_object: str,
) -> dict[str, Any]:
    return {
        "name": role_row["instance_name"],
        "project": plan["project"],
        "zone": plan["zone"],
        "machine_type": MACHINE_TYPE,
        "provisioning_model": "SPOT",
        "automatic_restart": False,
        "on_host_maintenance": "TERMINATE",
        "max_run_duration_seconds": VM_TTL_SECONDS,
        "deletion_protection": False,
        "image": plan["image"]["selfLink"],
        "boot_disk_type": BOOT_DISK_TYPE,
        "boot_disk_interface": BOOT_DISK_INTERFACE,
        "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
        "network_nic_type": NETWORK_NIC_TYPE,
        "labels": {"ofc-owner": role_row["ownership_label"], "ofc-plan": canonical_sha256(plan)[:32], "ofc-role": role_row["source_role"]},
        "metadata": {
            "startup-script": None,
            "PROJECT_ID": plan["project"], "BUCKET": None,
            "RUN_NAME": plan["run_name"], "IDENTITY_NAMESPACE": plan["identity_namespace"],
            "SOURCE_ROLE": role_row["source_role"], "RESULT_PREFIX": plan["result_prefix"],
            "SOURCE_OBJECT": source_object, "SOURCE_SHA256": plan["source_archive"]["sha256"],
            "WHEELHOUSE_OBJECT": wheelhouse_object,
            "WHEELHOUSE_SHA256": plan["runtime_wheelhouse"]["sha256"],
            "WHEELHOUSE_MANIFEST_SHA256": plan["runtime_wheelhouse"]["manifest_sha256"],
            "PLAN_OBJECT": plan_object, "PLAN_SHA256": canonical_sha256(plan),
            "AUTHORIZATION_OBJECT": authorization_object,
            "AUTHORIZATION_SHA256": canonical_sha256(authorization),
            "AUTHORIZATION_NONCE_SHA256": authorization["one_shot_nonce_sha256"],
            "STARTUP_SHA256": plan["startup"]["sha256"],
            "OWNERSHIP_LABEL": role_row["ownership_label"], "INSTANCE_NAME": role_row["instance_name"],
            "ATTEMPT_INDEX": "0", "RESUME_OBJECT": "none", "RESUME_SHA256": "none",
            "MAX_RUNTIME_SECONDS": str(MAX_VM_RUNTIME_SECONDS), "HEARTBEAT_SECONDS": str(HEARTBEAT_SECONDS),
        },
    }


def _receipt_with_digest(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_content_sha256": canonical_sha256(value)}


def launch_pair(
    *,
    cloud_package_dir: str | Path,
    content_stage_receipt: Mapping[str, Any],
    launch_overlay: Mapping[str, Any],
    authorization: Mapping[str, Any],
    raw_one_shot_nonce: str,
    bucket: str,
    transport: CloudTransport,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    """Launch exactly two owned VMs through an injected transport."""

    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    view = _load_cloud_package(cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False)
    staged = _validate_content_stage_receipt(content_stage_receipt, view=view)
    overlay = _validate_launch_overlay(
        launch_overlay, view=view, now_unix_seconds=now
    )
    if overlay["content_stage_receipt_sha256"] != canonical_sha256(staged):
        raise ValueError("fresh overlay/content-stage receipt binding changed")
    auth = validate_launch_authorization(
        authorization,
        execution_plan=view.plan,
        raw_nonce=raw_one_shot_nonce,
        now_unix_seconds=now,
    )
    if auth["package_manifest_sha256"] != view.package_manifest_sha256:
        raise ValueError("authorization package manifest hash changed")
    if (
        auth["launch_overlay_sha256"] != canonical_sha256(overlay)
        or auth["content_stage_receipt_sha256"] != canonical_sha256(staged)
        or auth["worker_iam_readback_receipt_sha256"]
        != overlay["worker_iam_readback_receipt_sha256"]
    ):
        raise ValueError("authorization fresh overlay binding changed")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,62}", bucket):
        raise ValueError("bucket name is invalid")
    names = [row["instance_name"] for row in view.plan["instances"]]
    observation = transport.inspect_namespace(run_name=view.plan["run_name"], identity_namespace=view.plan["identity_namespace"], result_prefix=view.plan["result_prefix"], instance_names=names)
    checked_observation = _validate_namespace_observation(observation, view.plan)
    if checked_observation["worker_iam"]["bucket"] != bucket:
        raise PermissionError("worker IAM observation belongs to another bucket")
    iam_expiry = checked_observation["worker_iam"]["iam_expiry_unix_seconds"]
    if (
        iam_expiry != overlay["worker_iam_expires_unix_seconds"]
        or iam_expiry - now < MIN_WORKER_IAM_WINDOW_SECONDS
        or iam_expiry - now > MAX_WORKER_IAM_WINDOW_SECONDS
    ):
        raise PermissionError("worker IAM expiry is not exact or TTL-safe")
    prefix = view.plan["result_prefix"]
    nonce_hash = auth["one_shot_nonce_sha256"]
    claim_object = f"{prefix}control/nonce-{nonce_hash}.json"
    source_object = f"{prefix}control/{SOURCE_NAME}"
    wheelhouse_object = f"{prefix}control/{WHEELHOUSE_NAME}"
    wheelhouse_manifest_object = f"{prefix}control/{WHEELHOUSE_MANIFEST_NAME}"
    plan_object = f"{prefix}control/{PLAN_NAME}"
    auth_object = f"{prefix}control/launch_authorization.json"
    manifest_object = f"{prefix}control/{MANIFEST_NAME}"
    publish_receipts: list[dict[str, Any]] = []
    claim = {"schema": "hu_m31_t3_step6d_perfdev_v2_nonce_claim_v1", "run_name": view.plan["run_name"], "one_shot_nonce_sha256": nonce_hash, "execution_plan_sha256": view.plan_sha256, "claimed_unix_seconds": now}
    publish_receipts.append(_put_new(transport, claim_object, canonical_bytes(claim)))
    content_prefix = staged["content_prefix"]
    publish_receipts.append(
        _copy_new(
            transport,
            source_object=f"{content_prefix}{SOURCE_NAME}",
            destination_object=source_object,
            expected_sha256=view.source_sha256,
            expected_bytes=(view.directory / SOURCE_NAME).stat().st_size,
        )
    )
    publish_receipts.append(
        _copy_new(
            transport,
            source_object=f"{content_prefix}{WHEELHOUSE_NAME}",
            destination_object=wheelhouse_object,
            expected_sha256=view.plan["runtime_wheelhouse"]["sha256"],
            expected_bytes=(view.directory / WHEELHOUSE_NAME).stat().st_size,
        )
    )
    publish_receipts.append(
        _copy_new(
            transport,
            source_object=f"{content_prefix}{WHEELHOUSE_MANIFEST_NAME}",
            destination_object=wheelhouse_manifest_object,
            expected_sha256=view.plan["runtime_wheelhouse"]["manifest_sha256"],
            expected_bytes=(view.directory / WHEELHOUSE_MANIFEST_NAME).stat().st_size,
        )
    )
    publish_receipts.append(_put_new(transport, plan_object, canonical_bytes(view.plan)))
    publish_receipts.append(_put_new(transport, auth_object, canonical_bytes(auth)))
    publish_receipts.append(_put_new(transport, manifest_object, (view.directory / MANIFEST_NAME).read_bytes()))
    created: list[dict[str, Any]] = []
    failure: str | None = None
    for role_row in view.plan["instances"]:
        spec = _instance_spec(
            view.plan,
            auth,
            role_row,
            source_object=source_object,
            wheelhouse_object=wheelhouse_object,
            plan_object=plan_object,
            authorization_object=auth_object,
        )
        spec["metadata"]["startup-script"] = (view.directory / STARTUP_NAME).read_text(encoding="utf-8")
        spec["metadata"]["BUCKET"] = bucket
        try:
            response = dict(transport.create_instance(specification=spec))
            if set(response) != {"created", "name", "status", "ownership_label", "execution_plan_sha256"} or response.get("created") is not True or response.get("name") != role_row["instance_name"] or response.get("status") not in {"PROVISIONING", "STAGING", "RUNNING"} or response.get("ownership_label") != role_row["ownership_label"] or response.get("execution_plan_sha256") != view.plan_sha256:
                raise RuntimeError("created instance identity did not match frozen plan")
            created.append(response)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
    all_started = len(created) == VM_COUNT and failure is None
    unsigned = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "status": "launched_exact_candidate_reference_pair" if all_started else "partial_launch_failure_not_success",
        "run_name": view.plan["run_name"],
        "execution_plan_sha256": view.plan_sha256,
        "authorization_sha256": canonical_sha256(auth),
        "nonce_claim_object": claim_object,
        "publish_receipts": publish_receipts,
        "created_instances": created,
        "expected_instance_names": names,
        "failure": failure,
        "all_instances_started": all_started,
        "partial_result_is_success": False,
        "cleanup_authorized": False,
        "current_profile_changed": False,
    }
    return _receipt_with_digest(unsigned)


def _validate_partial_launch_receipt(
    value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(value)
    digest = receipt.pop("receipt_content_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    created = receipt.get("created_instances")
    expected_names = [row["instance_name"] for row in plan["instances"]]
    if (
        digest != canonical_sha256(receipt)
        or receipt.get("schema") != LAUNCH_RECEIPT_SCHEMA
        or receipt.get("status") != "partial_launch_failure_not_success"
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_plan_sha256") != canonical_sha256(plan)
        or not isinstance(created, list)
        or len(created) < 1
        or len(created) >= VM_COUNT
        or receipt.get("all_instances_started") is not False
        or receipt.get("partial_result_is_success") is not False
        or receipt.get("cleanup_authorized") is not False
    ):
        raise ValueError("receipt is not an owned partial launch eligible for recovery")
    allowed = {row["instance_name"]: row for row in plan["instances"]}
    seen: set[str] = set()
    for row in created:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "created",
                "name",
                "status",
                "ownership_label",
                "execution_plan_sha256",
            }
            or row.get("created") is not True
            or row.get("name") not in allowed
            or row.get("name") in seen
            or row.get("ownership_label")
            != allowed[row["name"]]["ownership_label"]
            or row.get("execution_plan_sha256") != canonical_sha256(plan)
        ):
            raise ValueError("partial launch created-instance ownership changed")
        seen.add(row["name"])
    if receipt.get("expected_instance_names") != expected_names:
        raise ValueError("partial launch expected pair identity changed")
    return {**receipt, "receipt_content_sha256": digest}


def _validate_full_launch_receipt(
    value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(value)
    digest = receipt.pop("receipt_content_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    created = receipt.get("created_instances")
    expected_names = [row["instance_name"] for row in plan["instances"]]
    expected_fields = {
        "schema", "status", "run_name", "execution_plan_sha256",
        "authorization_sha256", "nonce_claim_object", "publish_receipts",
        "created_instances", "expected_instance_names", "failure",
        "all_instances_started", "partial_result_is_success",
        "cleanup_authorized", "current_profile_changed",
    }
    if (
        digest != canonical_sha256(receipt)
        or set(receipt) != expected_fields
        or receipt.get("schema") != LAUNCH_RECEIPT_SCHEMA
        or receipt.get("status") != "launched_exact_candidate_reference_pair"
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_plan_sha256") != canonical_sha256(plan)
        or _SHA.fullmatch(str(receipt.get("authorization_sha256"))) is None
        or receipt.get("nonce_claim_object") is None
        or not str(receipt["nonce_claim_object"]).startswith(
            f"{plan['result_prefix']}control/nonce-"
        )
        or not isinstance(receipt.get("publish_receipts"), list)
        or len(receipt["publish_receipts"]) != 7
        or any(not isinstance(row, Mapping) for row in receipt["publish_receipts"])
        or not isinstance(created, list)
        or len(created) != VM_COUNT
        or receipt.get("expected_instance_names") != expected_names
        or receipt.get("failure") is not None
        or receipt.get("all_instances_started") is not True
        or receipt.get("partial_result_is_success") is not False
        or receipt.get("cleanup_authorized") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("receipt is not an exact owned full pair launch")
    allowed = {row["instance_name"]: row for row in plan["instances"]}
    if [row.get("name") if isinstance(row, Mapping) else None for row in created] != expected_names:
        raise ValueError("full launch created-instance order or pair changed")
    for row in created:
        if (
            set(row)
            != {
                "created", "name", "status", "ownership_label",
                "execution_plan_sha256",
            }
            or row.get("created") is not True
            or row.get("status") not in {"PROVISIONING", "STAGING", "RUNNING"}
            or row.get("ownership_label")
            != allowed[row["name"]]["ownership_label"]
            or row.get("execution_plan_sha256") != canonical_sha256(plan)
        ):
            raise ValueError("full launch created-instance ownership changed")
    return {**receipt, "receipt_content_sha256": digest}


def cleanup_partial_launch(
    *,
    partial_launch_receipt: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    transport: CloudTransport,
) -> dict[str, Any]:
    """Delete only VMs confirmed created by a failed pair launch.

    This recovery does not require scientific artifacts because the pair never
    became a valid run.  It cannot delete the uncreated peer or any name absent
    from the signed launch receipt.
    """

    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    receipt = _validate_partial_launch_receipt(
        partial_launch_receipt, execution_plan=plan
    )
    deleted: list[dict[str, Any]] = []
    for row in receipt["created_instances"]:
        name = row["name"]
        owner = row["ownership_label"]
        observed = transport.get_instance(instance_name=name)
        if observed is None:
            deleted.append({"instance_name": name, "status": "already_absent"})
            continue
        if (
            observed.get("name") != name
            or observed.get("ownership_label") != owner
            or observed.get("execution_plan_sha256") != canonical_sha256(plan)
        ):
            raise ValueError("partial cleanup target ownership changed")
        result = dict(
            transport.delete_instance_exact(
                instance_name=name,
                ownership_label=owner,
                execution_plan_sha256=canonical_sha256(plan),
            )
        )
        if result != {
            "instance_name": name,
            "deleted": True,
            "ownership_label": owner,
        }:
            raise RuntimeError("partial owned instance delete did not confirm")
        deleted.append(
            {"instance_name": name, "status": "deleted_exact_partial_owned_instance"}
        )
    unsigned = {
        "schema": PARTIAL_CLEANUP_SCHEMA,
        "status": "partial_launch_owned_cleanup_complete",
        "run_name": plan["run_name"],
        "execution_plan_sha256": canonical_sha256(plan),
        "deleted": deleted,
        "created_instance_count": len(receipt["created_instances"]),
        "uncreated_peer_delete_attempted": False,
        "wildcard_delete_used": False,
        "unrelated_instance_touched": False,
        "scientific_result_claimed": False,
        "current_profile_changed": False,
    }
    return _receipt_with_digest(unsigned)


def _zero_created_control_object_names(
    *, plan: Mapping[str, Any], launch_nonce_sha256: str
) -> list[str]:
    prefix = f"{plan['result_prefix']}control/"
    return [
        f"{prefix}nonce-{launch_nonce_sha256}.json",
        f"{prefix}{SOURCE_NAME}",
        f"{prefix}{WHEELHOUSE_NAME}",
        f"{prefix}{WHEELHOUSE_MANIFEST_NAME}",
        f"{prefix}{PLAN_NAME}",
        f"{prefix}launch_authorization.json",
        f"{prefix}{MANIFEST_NAME}",
    ]


def _canonical_json_object_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def closeout_zero_created_launch_boundary(
    *,
    cloud_package_dir: str | Path,
    worker_iam_readback: Mapping[str, Any],
    raw_launch_nonce: str,
    launch_authorization: Mapping[str, Any] | None,
    raw_closeout_nonce: str,
    failure_stage: str,
    controller_insert_attempt_count: int,
    explicit_operator_abort: bool,
    transport: CloudTransport,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Record a read-only diagnostic at a declared pre-insert boundary.

    This self-contained receipt is deliberately *not* terminal IAM cleanup
    evidence: exact GCE operation history is unavailable here, so current
    absence cannot prove that a response-lost insert was never accepted.  The
    controller must declare that it made zero insert attempts, and any measured
    planned instance fails closed into the existing partial/full recovery
    paths.  Stale IAM bindings use the separate post-expiry exact2 cleanup.
    """

    if explicit_operator_abort is not True:
        raise PermissionError("explicit zero-created operator-abort is required")
    if controller_insert_attempt_count != 0 or isinstance(
        controller_insert_attempt_count, bool
    ):
        raise ValueError("zero-created diagnostic requires zero declared inserts")
    if failure_stage not in ZERO_CREATED_FAILURE_STAGES:
        raise ValueError("zero-created failure stage is invalid")
    try:
        launch_nonce = uuid.UUID(raw_launch_nonce)
        closeout_nonce = uuid.UUID(raw_closeout_nonce)
    except ValueError as exc:
        raise ValueError("zero-created nonces must be canonical UUIDv4 values") from exc
    if (
        launch_nonce.version != 4
        or str(launch_nonce) != raw_launch_nonce
        or closeout_nonce.version != 4
        or str(closeout_nonce) != raw_closeout_nonce
    ):
        raise ValueError("zero-created nonces must be canonical UUIDv4 values")

    view = _load_cloud_package(
        cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False
    )
    launch_nonce_sha = hashlib.sha256(raw_launch_nonce.encode("ascii")).hexdigest()
    iam_readback = validate_worker_iam_readback(
        worker_iam_readback,
        execution_plan=view.plan,
        expected_one_shot_nonce_sha256=launch_nonce_sha,
        now_unix_seconds=None,
    )
    checked_authorization: dict[str, Any] | None = None
    if launch_authorization is not None:
        checked_authorization = validate_launch_authorization(
            launch_authorization,
            execution_plan=view.plan,
            raw_nonce=raw_launch_nonce,
            now_unix_seconds=None,
        )

    listed_names = list(transport.list_objects(prefix=view.plan["result_prefix"]))
    if len(listed_names) != len(set(listed_names)):
        raise ValueError("zero-created result namespace contains duplicate objects")
    allowed_order = _zero_created_control_object_names(
        plan=view.plan, launch_nonce_sha256=launch_nonce_sha
    )
    expected_prefix = allowed_order[: len(listed_names)]
    if len(listed_names) > len(allowed_order) or set(listed_names) != set(
        expected_prefix
    ):
        raise ValueError(
            "zero-created result namespace is not an exact launch-control prefix"
        )
    # GCS LIST order is not a creation-order contract.  After proving the set
    # is an exact prefix, normalize to the frozen publication order.
    observed_names = expected_prefix
    object_count = len(observed_names)
    if failure_stage in {"overlay_failed", "authorization_failed"}:
        if checked_authorization is not None or object_count != 0:
            raise ValueError("pre-authorization zero-created topology changed")
    elif failure_stage in {"prelaunch_aborted", "namespace_preflight_failed"}:
        if checked_authorization is None or object_count != 0:
            raise ValueError("prelaunch zero-created topology changed")
    elif failure_stage == "control_publish_failed":
        if checked_authorization is None or not 0 <= object_count < len(allowed_order):
            raise ValueError("partial control-publish topology changed")
    elif (
        checked_authorization is None
        or object_count != len(allowed_order)
    ):
        raise ValueError("before-first-insert control topology changed")

    records: list[dict[str, Any]] = []
    for object_name in observed_names:
        raw = transport.get_object(object_name=object_name)
        if raw is None:
            raise ValueError("zero-created control object disappeared during readback")
        relative = object_name.removeprefix(
            f"{view.plan['result_prefix']}control/"
        )
        if relative == SOURCE_NAME:
            expected = view.plan["source_archive"]["sha256"]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("zero-created source archive identity changed")
        elif relative == WHEELHOUSE_NAME:
            expected = view.plan["runtime_wheelhouse"]["sha256"]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("zero-created wheelhouse identity changed")
        elif relative == WHEELHOUSE_MANIFEST_NAME:
            expected = view.plan["runtime_wheelhouse"]["manifest_sha256"]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("zero-created wheelhouse manifest identity changed")
        elif relative == PLAN_NAME:
            if raw != canonical_bytes(view.plan):
                raise ValueError("zero-created execution plan bytes changed")
        elif relative == "launch_authorization.json":
            if checked_authorization is None or raw != canonical_bytes(
                checked_authorization
            ):
                raise ValueError("zero-created launch authorization bytes changed")
        elif relative == MANIFEST_NAME:
            if raw != (view.directory / MANIFEST_NAME).read_bytes():
                raise ValueError("zero-created cloud manifest bytes changed")
        else:
            claim = _canonical_json_object_bytes(raw, "zero-created nonce claim")
            if (
                set(claim)
                != {
                    "schema", "run_name", "one_shot_nonce_sha256",
                    "execution_plan_sha256", "claimed_unix_seconds",
                }
                or claim.get("schema")
                != "hu_m31_t3_step6d_perfdev_v2_nonce_claim_v1"
                or claim.get("run_name") != view.plan["run_name"]
                or claim.get("one_shot_nonce_sha256") != launch_nonce_sha
                or claim.get("execution_plan_sha256") != view.plan_sha256
                or not _plain_int(claim.get("claimed_unix_seconds"))
            ):
                raise ValueError("zero-created nonce claim identity changed")
        records.append(
            {
                "object_name": object_name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )

    planned = [
        {
            "source_role": row["source_role"],
            "instance_name": row["instance_name"],
            "ownership_label": row["ownership_label"],
        }
        for row in view.plan["instances"]
    ]
    zero_readbacks: list[dict[str, Any]] = []
    sleeper = time.sleep if sleep is None else sleep
    if not callable(sleeper):
        raise ValueError("zero-created confirmation sleep is invalid")
    for attempt in range(ZERO_CREATED_CONFIRMATION_ATTEMPTS):
        absent: list[str] = []
        for row in planned:
            observed = transport.get_instance(instance_name=row["instance_name"])
            if observed is not None:
                raise ValueError(
                    "planned instance exists; zero-created closeout is forbidden"
                )
            absent.append(row["instance_name"])
        zero_readbacks.append(
            {"attempt_index": attempt, "absent_instance_names": absent}
        )
        if attempt + 1 < ZERO_CREATED_CONFIRMATION_ATTEMPTS:
            sleeper(ZERO_CREATED_CONFIRMATION_INTERVAL_SECONDS)
    final_names = list(transport.list_objects(prefix=view.plan["result_prefix"]))
    if len(final_names) != len(set(final_names)) or set(final_names) != set(
        observed_names
    ):
        raise ValueError("zero-created result namespace changed during confirmation")

    unsigned = {
        "schema": ZERO_CREATED_CLOSEOUT_SCHEMA,
        "status": "operator_aborted_preinsert_diagnostic_readback_complete",
        "run_name": view.plan["run_name"],
        "execution_plan_sha256": view.plan_sha256,
        "worker_iam_readback_receipt_sha256": iam_readback["receipt_sha256"],
        "launch_nonce_sha256": launch_nonce_sha,
        "closeout_nonce_sha256": hashlib.sha256(
            raw_closeout_nonce.encode("ascii")
        ).hexdigest(),
        "launch_authorization_sha256": (
            canonical_sha256(checked_authorization)
            if checked_authorization is not None
            else None
        ),
        "failure_stage": failure_stage,
        "controller_declared_insert_attempt_count": 0,
        "insert_attempt_count_independently_verified": False,
        "insert_operation_identity": None,
        "historical_zero_created_claimed": False,
        "diagnostic_only": True,
        "iam_cleanup_authorized": False,
        "planned_instances": planned,
        "owned_instance_count": 0,
        "all_planned_instances_absent": True,
        "zero_created_readback_attempt_count": len(zero_readbacks),
        "zero_created_readbacks": zero_readbacks,
        "zero_created_readbacks_sha256": canonical_sha256(zero_readbacks),
        "prior_result_prefix_object_count": len(records),
        "prior_control_objects": records,
        "prior_control_objects_sha256": canonical_sha256(records),
        "prior_control_publish_prefix_length": len(records),
        "namespace_list_readback_count": 2,
        "namespace_stable_during_confirmation": True,
        "scientific_namespace_object_count": 0,
        "operator_abort": True,
        "automatic_closeout": False,
        "scientific_result_claimed": False,
        "artifact_validation_claimed": False,
        "training_eligible": False,
        "closeout_cloud_mutation_performed": False,
        "instance_delete_attempt_count": 0,
        "wildcard_delete_used": False,
        "unrelated_instance_touched": False,
        "current_profile_changed": False,
    }
    return _receipt_with_digest(unsigned)


def _validate_zero_created_closeout_receipt(
    value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(value)
    digest = receipt.pop("receipt_content_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    expected_instances = [
        {
            "source_role": row["source_role"],
            "instance_name": row["instance_name"],
            "ownership_label": row["ownership_label"],
        }
        for row in plan["instances"]
    ]
    launch_nonce_sha = receipt.get("launch_nonce_sha256")
    allowed_order = (
        _zero_created_control_object_names(
            plan=plan, launch_nonce_sha256=str(launch_nonce_sha)
        )
        if _SHA.fullmatch(str(launch_nonce_sha)) is not None
        else []
    )
    records = receipt.get("prior_control_objects")
    readbacks = receipt.get("zero_created_readbacks")
    if (
        digest != canonical_sha256(receipt)
        or set(receipt)
        != {
            "schema", "status", "run_name", "execution_plan_sha256",
            "worker_iam_readback_receipt_sha256", "launch_nonce_sha256",
            "closeout_nonce_sha256", "launch_authorization_sha256",
            "failure_stage", "controller_declared_insert_attempt_count",
            "insert_attempt_count_independently_verified",
            "insert_operation_identity", "historical_zero_created_claimed",
            "diagnostic_only", "iam_cleanup_authorized",
            "planned_instances", "owned_instance_count",
            "all_planned_instances_absent",
            "zero_created_readback_attempt_count", "zero_created_readbacks",
            "zero_created_readbacks_sha256", "prior_result_prefix_object_count",
            "prior_control_objects", "prior_control_objects_sha256",
            "prior_control_publish_prefix_length", "namespace_list_readback_count",
            "namespace_stable_during_confirmation",
            "scientific_namespace_object_count", "operator_abort",
            "automatic_closeout", "scientific_result_claimed",
            "artifact_validation_claimed", "training_eligible",
            "closeout_cloud_mutation_performed", "instance_delete_attempt_count",
            "wildcard_delete_used", "unrelated_instance_touched",
            "current_profile_changed",
        }
        or receipt.get("schema") != ZERO_CREATED_CLOSEOUT_SCHEMA
        or receipt.get("status")
        != "operator_aborted_preinsert_diagnostic_readback_complete"
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_plan_sha256") != canonical_sha256(plan)
        or any(
            _SHA.fullmatch(str(receipt.get(field))) is None
            for field in (
                "worker_iam_readback_receipt_sha256", "launch_nonce_sha256",
                "closeout_nonce_sha256", "zero_created_readbacks_sha256",
                "prior_control_objects_sha256",
            )
        )
        or receipt.get("failure_stage") not in ZERO_CREATED_FAILURE_STAGES
        or receipt.get("controller_declared_insert_attempt_count") != 0
        or receipt.get("insert_attempt_count_independently_verified") is not False
        or receipt.get("insert_operation_identity") is not None
        or receipt.get("historical_zero_created_claimed") is not False
        or receipt.get("diagnostic_only") is not True
        or receipt.get("iam_cleanup_authorized") is not False
        or receipt.get("planned_instances") != expected_instances
        or receipt.get("owned_instance_count") != 0
        or receipt.get("all_planned_instances_absent") is not True
        or not isinstance(readbacks, list)
        or len(readbacks) != ZERO_CREATED_CONFIRMATION_ATTEMPTS
        or receipt.get("zero_created_readback_attempt_count") != len(readbacks)
        or receipt.get("zero_created_readbacks_sha256")
        != canonical_sha256(readbacks)
        or not isinstance(records, list)
        or receipt.get("prior_result_prefix_object_count") != len(records)
        or receipt.get("prior_control_objects_sha256") != canonical_sha256(records)
        or receipt.get("prior_control_publish_prefix_length") != len(records)
        or receipt.get("namespace_list_readback_count") != 2
        or receipt.get("namespace_stable_during_confirmation") is not True
        or receipt.get("scientific_namespace_object_count") != 0
        or receipt.get("operator_abort") is not True
        or receipt.get("automatic_closeout") is not False
        or receipt.get("scientific_result_claimed") is not False
        or receipt.get("artifact_validation_claimed") is not False
        or receipt.get("training_eligible") is not False
        or receipt.get("closeout_cloud_mutation_performed") is not False
        or receipt.get("instance_delete_attempt_count") != 0
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_instance_touched") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("zero-created closeout receipt changed")

    expected_names = [row["instance_name"] for row in expected_instances]
    for attempt, row in enumerate(readbacks):
        if row != {
            "attempt_index": attempt,
            "absent_instance_names": expected_names,
        }:
            raise ValueError("zero-created instance readback topology changed")
    if len(records) > len(allowed_order) or any(
        not isinstance(row, Mapping)
        or set(row) != {"object_name", "sha256", "bytes"}
        or row.get("object_name") != allowed_order[index]
        or _SHA.fullmatch(str(row.get("sha256"))) is None
        or not _plain_int(row.get("bytes"))
        or row["bytes"] <= 0
        for index, row in enumerate(records)
    ):
        raise ValueError("zero-created control object topology changed")

    count = len(records)
    stage = receipt["failure_stage"]
    auth_sha = receipt.get("launch_authorization_sha256")
    if stage in {"overlay_failed", "authorization_failed"}:
        valid_stage = count == 0 and auth_sha is None
    elif stage in {"prelaunch_aborted", "namespace_preflight_failed"}:
        valid_stage = count == 0 and _SHA.fullmatch(str(auth_sha)) is not None
    elif stage == "control_publish_failed":
        valid_stage = (
            0 <= count < len(allowed_order)
            and _SHA.fullmatch(str(auth_sha)) is not None
        )
    else:
        valid_stage = (
            count == len(allowed_order)
            and _SHA.fullmatch(str(auth_sha)) is not None
        )
    if not valid_stage:
        raise ValueError("zero-created failure-stage topology changed")
    expected_hashes = {
        SOURCE_NAME: plan["source_archive"]["sha256"],
        WHEELHOUSE_NAME: plan["runtime_wheelhouse"]["sha256"],
        WHEELHOUSE_MANIFEST_NAME: plan["runtime_wheelhouse"]["manifest_sha256"],
        PLAN_NAME: canonical_sha256(plan),
        "launch_authorization.json": auth_sha,
    }
    for row in records:
        relative = str(row["object_name"]).removeprefix(
            f"{plan['result_prefix']}control/"
        )
        expected_hash = expected_hashes.get(relative)
        if expected_hash is not None and row["sha256"] != expected_hash:
            raise ValueError("zero-created control object identity changed")
    return {**receipt, "receipt_content_sha256": digest}


def closeout_failed_owned_launch_pair(
    *,
    launch_receipt: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    raw_cleanup_nonce: str,
    explicit_operator_abort: bool,
    transport: CloudTransport,
) -> dict[str, Any]:
    """Operator-abort an exact launched pair without claiming a result.

    This is deliberately separate from collection cleanup and is never invoked
    automatically.  It exists only for the case where both inserts succeeded
    but startup/runner failed before a complete scientific collection existed.
    """

    if explicit_operator_abort is not True:
        raise PermissionError("explicit operator-abort authorization is required")
    try:
        parsed = uuid.UUID(raw_cleanup_nonce)
    except ValueError as exc:
        raise ValueError("cleanup nonce must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw_cleanup_nonce:
        raise ValueError("cleanup nonce must be a canonical UUIDv4")
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    launch = _validate_full_launch_receipt(
        launch_receipt, execution_plan=plan
    )
    deleted: list[dict[str, Any]] = []
    for row in launch["created_instances"]:
        name = row["name"]
        owner = row["ownership_label"]
        observed = transport.get_instance(instance_name=name)
        if observed is None:
            deleted.append(
                {
                    "instance_name": name,
                    "ownership_label": owner,
                    "status": "already_absent",
                }
            )
            continue
        if (
            not isinstance(observed, Mapping)
            or observed.get("name") != name
            or observed.get("ownership_label") != owner
            or observed.get("execution_plan_sha256") != canonical_sha256(plan)
        ):
            raise ValueError("operator-abort cleanup target ownership changed")
        result = dict(
            transport.delete_instance_exact(
                instance_name=name,
                ownership_label=owner,
                execution_plan_sha256=canonical_sha256(plan),
            )
        )
        if result != {
            "instance_name": name,
            "deleted": True,
            "ownership_label": owner,
        }:
            raise RuntimeError("operator-abort exact owned deletion did not confirm")
        confirmed_absent = False
        for attempt in range(DELETE_CONFIRMATION_ATTEMPTS):
            remaining = transport.get_instance(instance_name=name)
            if remaining is None:
                confirmed_absent = True
                break
            if (
                not isinstance(remaining, Mapping)
                or remaining.get("name") != name
                or remaining.get("ownership_label") != owner
                or remaining.get("execution_plan_sha256")
                != canonical_sha256(plan)
            ):
                raise ValueError(
                    "operator-abort post-delete target ownership changed"
                )
            if attempt + 1 < DELETE_CONFIRMATION_ATTEMPTS:
                time.sleep(DELETE_CONFIRMATION_INTERVAL_SECONDS)
        if not confirmed_absent:
            raise TimeoutError(
                "operator-abort delete was not confirmed absent within bound"
            )
        deleted.append(
            {
                "instance_name": name,
                "ownership_label": owner,
                "status": "deleted_exact_owned_launched_instance_confirmed_absent",
            }
        )
    unsigned = {
        "schema": OWNED_LAUNCH_FAILURE_CLOSEOUT_SCHEMA,
        "status": "operator_aborted_owned_launched_pair_cleanup_complete",
        "run_name": plan["run_name"],
        "execution_plan_sha256": canonical_sha256(plan),
        "launch_receipt_sha256": launch["receipt_content_sha256"],
        "cleanup_nonce_sha256": hashlib.sha256(
            raw_cleanup_nonce.encode("ascii")
        ).hexdigest(),
        "deleted": deleted,
        "owned_instance_count": VM_COUNT,
        "all_owned_instances_deleted": True,
        "operator_abort": True,
        "automatic_closeout": False,
        "scientific_result_claimed": False,
        "artifact_validation_claimed": False,
        "training_eligible": False,
        "wildcard_delete_used": False,
        "unrelated_instance_touched": False,
        "current_profile_changed": False,
    }
    return _receipt_with_digest(unsigned)


def _validate_owned_launch_failure_closeout_receipt(
    value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(value)
    digest = receipt.pop("receipt_content_sha256", None)
    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    expected_names = [row["instance_name"] for row in plan["instances"]]
    deleted = receipt.get("deleted")
    if (
        digest != canonical_sha256(receipt)
        or set(receipt)
        != {
            "schema", "status", "run_name", "execution_plan_sha256",
            "launch_receipt_sha256", "cleanup_nonce_sha256", "deleted",
            "owned_instance_count", "all_owned_instances_deleted",
            "operator_abort", "automatic_closeout", "scientific_result_claimed",
            "artifact_validation_claimed", "training_eligible",
            "wildcard_delete_used", "unrelated_instance_touched",
            "current_profile_changed",
        }
        or receipt.get("schema") != OWNED_LAUNCH_FAILURE_CLOSEOUT_SCHEMA
        or receipt.get("status")
        != "operator_aborted_owned_launched_pair_cleanup_complete"
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_plan_sha256") != canonical_sha256(plan)
        or _SHA.fullmatch(str(receipt.get("launch_receipt_sha256"))) is None
        or _SHA.fullmatch(str(receipt.get("cleanup_nonce_sha256"))) is None
        or not isinstance(deleted, list)
        or len(deleted) != VM_COUNT
        or [row.get("instance_name") if isinstance(row, Mapping) else None for row in deleted]
        != expected_names
        or receipt.get("owned_instance_count") != VM_COUNT
        or receipt.get("all_owned_instances_deleted") is not True
        or receipt.get("operator_abort") is not True
        or receipt.get("automatic_closeout") is not False
        or receipt.get("scientific_result_claimed") is not False
        or receipt.get("artifact_validation_claimed") is not False
        or receipt.get("training_eligible") is not False
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_instance_touched") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("owned launched-pair failure closeout receipt changed")
    owners = {row["instance_name"]: row["ownership_label"] for row in plan["instances"]}
    for row in deleted:
        if (
            set(row) != {"instance_name", "ownership_label", "status"}
            or row.get("ownership_label") != owners[row["instance_name"]]
            or row.get("status")
            not in {
                "already_absent",
                "deleted_exact_owned_launched_instance_confirmed_absent",
            }
        ):
            raise ValueError("owned launched-pair failure closeout topology changed")
    return {**receipt, "receipt_content_sha256": digest}


def _required_artifact_paths(role: str) -> list[str]:
    paths = ["run_contract.json", "shard_manifest.json", "DONE.json"]
    paths.extend(f"roots/hand_{index:03d}.json" for index in TAIL_HAND_INDICES)
    paths.extend(f"hands/{role}/hand_{index:03d}.json" for index in TAIL_HAND_INDICES)
    return sorted(paths)


def _strict_role_result_manifest(raw: bytes, *, role: str, view: _PackageView, authorization: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("role result manifest is invalid JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError("role result manifest is not canonical")
    expected_keys = {"schema", "status", "run_name", "source_role", "instance_name", "attempt_index", "source_sha256", "execution_plan_sha256", "authorization_sha256", "authorization_nonce_sha256", "startup_sha256", "ownership_label", "work_hand_indices", "artifact_count", "artifacts", "artifact_manifest_sha256", "heartbeat_count", "runner_validation_passed", "partial_result"}
    instance = next(row for row in view.plan["instances"] if row["source_role"] == role)
    artifacts = value.get("artifacts")
    if set(value) != expected_keys or value.get("schema") != ROLE_RESULT_SCHEMA or value.get("status") != "complete_validated_role_result" or value.get("run_name") != view.plan["run_name"] or value.get("source_role") != role or value.get("instance_name") != instance["instance_name"] or value.get("attempt_index") != authorization["attempt_index"] or value.get("source_sha256") != view.source_sha256 or value.get("execution_plan_sha256") != view.plan_sha256 or value.get("authorization_sha256") != canonical_sha256(authorization) or value.get("authorization_nonce_sha256") != authorization["one_shot_nonce_sha256"] or value.get("startup_sha256") != view.startup_sha256 or value.get("ownership_label") != instance["ownership_label"] or value.get("work_hand_indices") != list(TAIL_HAND_INDICES) or not isinstance(artifacts, list) or value.get("artifact_count") != len(artifacts) or value.get("artifact_manifest_sha256") != canonical_sha256(artifacts) or not _plain_int(value.get("heartbeat_count")) or value["heartbeat_count"] < 1 or value.get("runner_validation_passed") is not True or value.get("partial_result") is not False:
        raise ValueError("role result manifest escaped complete frozen result")
    paths = [row.get("path") if isinstance(row, Mapping) else None for row in artifacts]
    if paths != _required_artifact_paths(role) or any(not isinstance(row, Mapping) or set(row) != {"path", "sha256", "bytes"} or _SHA.fullmatch(str(row.get("sha256"))) is None or not _plain_int(row.get("bytes")) or row["bytes"] <= 0 for row in artifacts):
        raise ValueError("role result artifact topology changed or is incomplete")
    return value


def collect_pair(
    *,
    cloud_package_dir: str | Path,
    authorization: Mapping[str, Any],
    transport: CloudTransport,
    completed_output_validator: Callable[[str | Path], Mapping[str, Any]] = runner.validate_completed_output,
) -> dict[str, Any]:
    """Receive both roles; a missing/tampered/partial role never returns success."""

    view = _load_cloud_package(cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False)
    auth = _validate_launch_authorization(
        authorization,
        execution_plan=view.plan,
        raw_nonce=None,
        now_unix_seconds=None,
        allow_runtime_nonce_hash_only=True,
    )
    role_receipts: list[dict[str, Any]] = []
    role_bytes: dict[str, dict[str, bytes]] = {}
    for role in SOURCE_ROLES:
        base_prefix = f"{view.plan['result_prefix']}results/{role}/"
        manifest_object = f"{base_prefix}RESULT_MANIFEST.json"
        raw_manifest = transport.get_object(object_name=manifest_object)
        if raw_manifest is None:
            raise ValueError(f"{role} result manifest is missing; partial result is not success")
        manifest = _strict_role_result_manifest(raw_manifest, role=role, view=view, authorization=auth)
        expected_objects = [f"{base_prefix}{row['path']}" for row in manifest["artifacts"]] + [manifest_object]
        observed_objects = sorted(transport.list_objects(prefix=base_prefix))
        if observed_objects != sorted(expected_objects):
            raise ValueError(f"{role} result object topology changed")
        heartbeats = sorted(transport.list_objects(prefix=f"{view.plan['result_prefix']}heartbeats/{role}/"))
        progress = sorted(transport.list_objects(prefix=f"{view.plan['result_prefix']}progress/{role}/"))
        if len(heartbeats) < 1 or not any(name.endswith("run_contract.json") for name in progress) or not any(name.endswith("shard_manifest.json") for name in progress):
            raise ValueError(f"{role} heartbeat/checkpoint evidence is incomplete")
        values: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory(prefix=f"ofc-perfdev-v2-{role}-") as temporary:
            output = Path(temporary)
            for row in manifest["artifacts"]:
                object_name = f"{base_prefix}{row['path']}"
                raw = transport.get_object(object_name=object_name)
                if raw is None or len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError(f"{role} artifact bytes changed: {row['path']}")
                path = output / Path(*PurePosixPath(row["path"]).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_once(path, raw)
                values[row["path"]] = raw
            _validate_tail_v2_runtime_artifacts(values, source_role=role)
            completed_output_validator(output)
        role_bytes[role] = values
        role_receipts.append({"source_role": role, "result_manifest_object": manifest_object, "result_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(), "artifact_count": len(manifest["artifacts"]), "heartbeat_count": len(heartbeats), "checkpoint_object_count": len(progress), "runner_validation_passed": True})
    for index in TAIL_HAND_INDICES:
        path = f"roots/hand_{index:03d}.json"
        if role_bytes["candidate"][path] != role_bytes["reference"][path]:
            raise ValueError("candidate/reference root bytes differ")
    if role_bytes["candidate"]["run_contract.json"] != role_bytes["reference"]["run_contract.json"]:
        raise ValueError("candidate/reference run contract bytes differ")
    unsigned = {
        "schema": COLLECTION_SCHEMA,
        "status": "complete_candidate_reference_pair_validated",
        "run_name": view.plan["run_name"],
        "execution_plan_sha256": view.plan_sha256,
        "authorization_sha256": canonical_sha256(auth),
        "run_contract_digest": TAIL_RUN_CONTRACT_DIGEST,
        "run_contract_schema": TAIL_RUN_CONTRACT_SCHEMA,
        "run_contract_variant": TAIL_RUN_CONTRACT_VARIANT,
        "selection_manifest_sha256": TAIL_SELECTION_MANIFEST_SHA256,
        "tail_hand_indices": list(TAIL_HAND_INDICES),
        "roles": role_receipts,
        "instance_ownership": view.plan["instances"],
        "portable_pair_complete": True,
        "artifact_validation_passed": True,
        "partial_result": False,
        "cleanup_authorized": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    return _receipt_with_digest(unsigned)


def _validate_collection_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    digest = receipt.pop("receipt_content_sha256", None)
    if digest != canonical_sha256(receipt) or receipt.get("schema") != COLLECTION_SCHEMA or receipt.get("status") != "complete_candidate_reference_pair_validated" or receipt.get("run_contract_digest") != TAIL_RUN_CONTRACT_DIGEST or receipt.get("run_contract_schema") != TAIL_RUN_CONTRACT_SCHEMA or receipt.get("run_contract_variant") != TAIL_RUN_CONTRACT_VARIANT or receipt.get("selection_manifest_sha256") != TAIL_SELECTION_MANIFEST_SHA256 or receipt.get("tail_hand_indices") != list(TAIL_HAND_INDICES) or receipt.get("portable_pair_complete") is not True or receipt.get("artifact_validation_passed") is not True or receipt.get("partial_result") is not False or receipt.get("cleanup_authorized") is not True or receipt.get("roles") is None or [row.get("source_role") for row in receipt["roles"]] != list(SOURCE_ROLES) or receipt.get("instance_ownership") is None or [row.get("source_role") for row in receipt["instance_ownership"]] != list(SOURCE_ROLES):
        raise ValueError("collection receipt is not a validated complete pair")
    return {**receipt, "receipt_content_sha256": digest}


def cleanup_collected_pair(
    *, collection_receipt: Mapping[str, Any], transport: CloudTransport
) -> dict[str, Any]:
    """Delete only the two exact owned names after complete artifact validation."""

    receipt = _validate_collection_receipt(collection_receipt)
    deleted: list[dict[str, Any]] = []
    for row in receipt["instance_ownership"]:
        name = row["instance_name"]
        owner = row["ownership_label"]
        observed = transport.get_instance(instance_name=name)
        if observed is None:
            deleted.append({"instance_name": name, "status": "already_absent"})
            continue
        if not isinstance(observed, Mapping) or observed.get("name") != name or observed.get("ownership_label") != owner or observed.get("execution_plan_sha256") != receipt["execution_plan_sha256"]:
            raise ValueError("cleanup target ownership does not match validated pair")
        result = dict(transport.delete_instance_exact(instance_name=name, ownership_label=owner, execution_plan_sha256=receipt["execution_plan_sha256"]))
        if result != {"instance_name": name, "deleted": True, "ownership_label": owner}:
            raise RuntimeError("exact owned instance deletion did not confirm")
        deleted.append({"instance_name": name, "status": "deleted_exact_owned_instance"})
    unsigned = {"schema": CLEANUP_SCHEMA, "status": "exact_owned_pair_cleanup_complete", "run_name": receipt["run_name"], "execution_plan_sha256": receipt["execution_plan_sha256"], "deleted": deleted, "wildcard_delete_used": False, "unrelated_instance_touched": False, "artifact_validation_preceded_cleanup": True, "current_profile_changed": False}
    return _receipt_with_digest(unsigned)


def status_pair(
    *, execution_plan: Mapping[str, Any], transport: CloudTransport
) -> dict[str, Any]:
    """Read-only compact status; it never interprets one completed role as pass."""

    plan = validate_execution_plan(execution_plan, require_fresh_receipt=False)
    roles: list[dict[str, Any]] = []
    for role in SOURCE_ROLES:
        result_prefix = f"{plan['result_prefix']}results/{role}/"
        result_objects = sorted(transport.list_objects(prefix=result_prefix))
        heartbeat_objects = sorted(
            transport.list_objects(prefix=f"{plan['result_prefix']}heartbeats/{role}/")
        )
        progress_objects = sorted(
            transport.list_objects(prefix=f"{plan['result_prefix']}progress/{role}/")
        )
        manifest_name = f"{result_prefix}RESULT_MANIFEST.json"
        roles.append(
            {
                "source_role": role,
                "result_object_count": len(result_objects),
                "heartbeat_object_count": len(heartbeat_objects),
                "progress_object_count": len(progress_objects),
                "result_manifest_present": manifest_name in result_objects,
            }
        )
    complete = all(row["result_manifest_present"] for row in roles)
    return {
        "schema": "hu_m31_t3_step6d_perfdev_v2_status_v1",
        "status": "both_role_manifests_present_unvalidated" if complete else "incomplete_not_success",
        "run_name": plan["run_name"],
        "execution_plan_sha256": canonical_sha256(plan),
        "roles": roles,
        "both_role_manifests_present": complete,
        "artifact_validation_passed": False,
        "partial_result_is_success": False,
        "cloud_mutated": False,
    }


def _load_json_file(path: str | Path, label: str) -> dict[str, Any]:
    return _read_canonical(path, label)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    package = sub.add_parser("package")
    package.add_argument("--local-package", type=Path, required=True)
    package.add_argument("--wheelhouse", type=Path, required=True)
    package.add_argument("--output-parent", type=Path, required=True)
    package.add_argument("--receipt-output", type=Path, required=True)
    package.add_argument("--now-unix-seconds", type=int)

    authorize = sub.add_parser("authorize")
    authorize.add_argument("--cloud-package", type=Path, required=True)
    authorize.add_argument("--launch-overlay", type=Path, required=True)
    authorize.add_argument("--nonce", required=True)
    authorize.add_argument(
        "--authorize",
        required=True,
        choices=("AUTHORIZE_T3_PERFDEV_V2_EXACT_TWO_VM_TAIL",),
    )
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--now-unix-seconds", type=int)

    stage_authorize = sub.add_parser("authorize-stage")
    stage_authorize.add_argument("--cloud-package", type=Path, required=True)
    stage_authorize.add_argument("--nonce", required=True)
    stage_authorize.add_argument(
        "--authorize",
        required=True,
        choices=("AUTHORIZE_T3_PERFDEV_V2_CONTENT_STAGE",),
    )
    stage_authorize.add_argument("--output", type=Path, required=True)
    stage_authorize.add_argument("--now-unix-seconds", type=int)

    stage = sub.add_parser("stage")
    stage.add_argument("--cloud-package", type=Path, required=True)
    stage.add_argument("--authorization", type=Path, required=True)
    stage.add_argument("--nonce", required=True)
    stage.add_argument("--bucket", required=True)
    stage.add_argument(
        "--authorize",
        required=True,
        choices=("EXECUTE_T3_PERFDEV_V2_CONTENT_STAGE",),
    )
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--now-unix-seconds", type=int)
    stage.add_argument("--dry-run", action="store_true")

    overlay = sub.add_parser("overlay")
    overlay.add_argument("--cloud-package", type=Path, required=True)
    overlay.add_argument("--content-stage-receipt", type=Path, required=True)
    overlay.add_argument("--worker-iam-readback", type=Path, required=True)
    overlay.add_argument("--nonce-sha256", required=True)
    overlay.add_argument("--fresh-dry-run-receipt", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    overlay.add_argument("--now-unix-seconds", type=int)

    launch = sub.add_parser("launch")
    launch.add_argument("--cloud-package", type=Path, required=True)
    launch.add_argument("--authorization", type=Path, required=True)
    launch.add_argument("--content-stage-receipt", type=Path, required=True)
    launch.add_argument("--launch-overlay", type=Path, required=True)
    launch.add_argument("--worker-iam-readback", type=Path, required=True)
    launch.add_argument("--nonce", required=True)
    launch.add_argument("--bucket", required=True)
    launch.add_argument(
        "--authorize",
        required=True,
        choices=("EXECUTE_T3_PERFDEV_V2_EXACT_TWO_VM_TAIL",),
    )
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--now-unix-seconds", type=int)
    launch.add_argument("--dry-run", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--cloud-package", type=Path, required=True)
    status.add_argument("--bucket", required=True)
    status.add_argument("--output", type=Path, required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--cloud-package", type=Path, required=True)
    collect.add_argument("--authorization", type=Path, required=True)
    collect.add_argument("--bucket", required=True)
    collect.add_argument("--output", type=Path, required=True)

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--cloud-package", type=Path, required=True)
    cleanup.add_argument("--collection", type=Path, required=True)
    cleanup.add_argument("--cleanup-nonce", required=True)
    cleanup.add_argument("--bucket", required=True)
    cleanup.add_argument(
        "--authorize",
        required=True,
        choices=("DELETE_ONLY_VALIDATED_OWNED_T3_PERFDEV_V2_PAIR",),
    )
    cleanup.add_argument("--output", type=Path, required=True)
    cleanup.add_argument("--dry-run", action="store_true")

    partial_cleanup = sub.add_parser("cleanup-partial")
    partial_cleanup.add_argument("--cloud-package", type=Path, required=True)
    partial_cleanup.add_argument("--launch-receipt", type=Path, required=True)
    partial_cleanup.add_argument("--cleanup-nonce", required=True)
    partial_cleanup.add_argument("--bucket", required=True)
    partial_cleanup.add_argument(
        "--authorize",
        required=True,
        choices=("DELETE_ONLY_PARTIAL_LAUNCH_OWNED_VM",),
    )
    partial_cleanup.add_argument("--output", type=Path, required=True)
    partial_cleanup.add_argument("--dry-run", action="store_true")

    launched_failure_cleanup = sub.add_parser("cleanup-launched-failure")
    launched_failure_cleanup.add_argument(
        "--cloud-package", type=Path, required=True
    )
    launched_failure_cleanup.add_argument(
        "--launch-receipt", type=Path, required=True
    )
    launched_failure_cleanup.add_argument("--cleanup-nonce", required=True)
    launched_failure_cleanup.add_argument("--bucket", required=True)
    launched_failure_cleanup.add_argument(
        "--authorize",
        required=True,
        choices=(
            "OPERATOR_ABORT_DELETE_ONLY_EXACT_OWNED_LAUNCHED_PAIR_NO_SCIENTIFIC_RESULT",
        ),
    )
    launched_failure_cleanup.add_argument("--output", type=Path, required=True)
    launched_failure_cleanup.add_argument("--dry-run", action="store_true")

    zero_created_closeout = sub.add_parser("closeout-zero-created")
    zero_created_closeout.add_argument(
        "--cloud-package", type=Path, required=True
    )
    zero_created_closeout.add_argument(
        "--worker-iam-readback", type=Path, required=True
    )
    zero_created_closeout.add_argument(
        "--launch-authorization", type=Path
    )
    zero_created_closeout.add_argument("--nonce", required=True)
    zero_created_closeout.add_argument("--closeout-nonce", required=True)
    zero_created_closeout.add_argument(
        "--failure-stage", required=True, choices=ZERO_CREATED_FAILURE_STAGES
    )
    zero_created_closeout.add_argument(
        "--controller-insert-attempt-count", required=True, type=int, choices=(0,)
    )
    zero_created_closeout.add_argument("--bucket", required=True)
    zero_created_closeout.add_argument(
        "--authorize",
        required=True,
        choices=(
            "OPERATOR_ABORT_RECORD_PREINSERT_DIAGNOSTIC_NO_IAM_CLEANUP",
        ),
    )
    zero_created_closeout.add_argument("--output", type=Path, required=True)
    return parser


TransportFactory = Callable[..., CloudTransport]


def main(
    argv: Sequence[str] | None = None,
    *,
    transport_factory: TransportFactory | None = None,
) -> int:
    args = _cli_parser().parse_args(argv)
    factory: TransportFactory = transport_factory or GcpHttpTransport
    if args.command == "package":
        receipt = build_cloud_executable_package(
            local_package_dir=args.local_package,
            wheelhouse_dir=args.wheelhouse,
            output_parent=args.output_parent,
            now_unix_seconds=args.now_unix_seconds,
        )
        write_json_once(args.receipt_output, receipt)
        return 0
    if args.command == "authorize-stage":
        authorization = build_content_stage_authorization(
            cloud_package_dir=args.cloud_package,
            explicit_stage_authorized=args.authorize
            == "AUTHORIZE_T3_PERFDEV_V2_CONTENT_STAGE",
            one_shot_nonce=args.nonce,
            now_unix_seconds=args.now_unix_seconds,
        )
        write_json_once(args.output, authorization)
        return 0
    if args.command == "authorize":
        overlay = _load_json_file(args.launch_overlay, "fresh launch overlay")
        authorization = build_launch_authorization(
            cloud_package_dir=args.cloud_package,
            launch_overlay=overlay,
            explicit_launch_authorized=args.authorize
            == "AUTHORIZE_T3_PERFDEV_V2_EXACT_TWO_VM_TAIL",
            one_shot_nonce=args.nonce,
            now_unix_seconds=args.now_unix_seconds,
        )
        write_json_once(args.output, authorization)
        return 0
    if args.command == "overlay":
        staged = _load_json_file(args.content_stage_receipt, "content-stage receipt")
        iam_readback = _load_json_file(args.worker_iam_readback, "worker IAM readback")
        overlay = build_fresh_launch_overlay(
            cloud_package_dir=args.cloud_package,
            content_stage_receipt=staged,
            worker_iam_readback=iam_readback,
            expected_one_shot_nonce_sha256=args.nonce_sha256,
            fresh_dry_run_receipt_path=args.fresh_dry_run_receipt,
            now_unix_seconds=args.now_unix_seconds,
        )
        write_json_once(args.output, overlay)
        return 0

    view = _load_cloud_package(
        args.cloud_package,
        now_unix_seconds=getattr(args, "now_unix_seconds", None),
        require_fresh_receipt=False,
    )
    common = {
        "project": view.plan["project"],
        "zone": view.plan["zone"],
        "bucket": args.bucket,
        "execution_plan": view.plan,
    }
    if args.command == "closeout-zero-created":
        iam_readback = _load_json_file(
            args.worker_iam_readback, "worker IAM readback"
        )
        launch_authorization = (
            _load_json_file(args.launch_authorization, "launch authorization")
            if args.launch_authorization is not None
            else None
        )
        transport = factory(
            **common,
            mode="zero-created-closeout",
            worker_iam_readback=iam_readback,
            raw_one_shot_nonce=args.nonce,
        )
        receipt = closeout_zero_created_launch_boundary(
            cloud_package_dir=args.cloud_package,
            worker_iam_readback=iam_readback,
            raw_launch_nonce=args.nonce,
            launch_authorization=launch_authorization,
            raw_closeout_nonce=args.closeout_nonce,
            failure_stage=args.failure_stage,
            controller_insert_attempt_count=args.controller_insert_attempt_count,
            explicit_operator_abort=(
                args.authorize
                == "OPERATOR_ABORT_RECORD_PREINSERT_DIAGNOSTIC_NO_IAM_CLEANUP"
            ),
            transport=transport,
        )
        write_json_once(args.output, receipt)
        return 0
    if args.command == "stage":
        authorization = _load_json_file(args.authorization, "content-stage authorization")
        if args.dry_run:
            checked = _validate_content_stage_authorization(
                authorization,
                execution_plan=view.plan,
                raw_nonce=args.nonce,
                now_unix_seconds=args.now_unix_seconds,
            )
            receipt = {
                "schema": "hu_m31_t3_step6d_perfdev_v2_stage_dry_run_v1",
                "status": "validated_no_cloud_transport_constructed",
                "run_name": view.plan["run_name"],
                "authorization_sha256": canonical_sha256(checked),
                "cloud_mutation_count": 0,
                "content_stage_executed": False,
            }
        else:
            transport = factory(
                **common,
                mode="stage",
                authorization=authorization,
                raw_one_shot_nonce=args.nonce,
                now_unix_seconds=args.now_unix_seconds,
            )
            receipt = stage_content(
                cloud_package_dir=args.cloud_package,
                authorization=authorization,
                raw_one_shot_nonce=args.nonce,
                transport=transport,
                now_unix_seconds=args.now_unix_seconds,
            )
        write_json_once(args.output, receipt)
        return 0
    if args.command == "launch":
        authorization = _load_json_file(args.authorization, "launch authorization")
        staged = _load_json_file(args.content_stage_receipt, "content-stage receipt")
        overlay = _load_json_file(args.launch_overlay, "fresh launch overlay")
        iam_readback = _load_json_file(args.worker_iam_readback, "worker IAM readback")
        checked_iam_readback = validate_worker_iam_readback(
            iam_readback,
            execution_plan=view.plan,
            expected_one_shot_nonce_sha256=hashlib.sha256(
                args.nonce.encode("ascii")
            ).hexdigest(),
            now_unix_seconds=args.now_unix_seconds,
        )
        if overlay.get("worker_iam_readback_receipt_sha256") != (
            checked_iam_readback["receipt_sha256"]
        ):
            raise ValueError("launch CLI IAM readback differs from overlay")
        if args.dry_run:
            checked_staged = _validate_content_stage_receipt(staged, view=view)
            _validate_launch_overlay(
                overlay, view=view, now_unix_seconds=args.now_unix_seconds
            )
            if overlay["content_stage_receipt_sha256"] != canonical_sha256(
                checked_staged
            ):
                raise ValueError("dry-run overlay/content stage binding changed")
            checked = validate_launch_authorization(
                authorization,
                execution_plan=view.plan,
                raw_nonce=args.nonce,
                now_unix_seconds=args.now_unix_seconds,
            )
            receipt = {
                "schema": "hu_m31_t3_step6d_perfdev_v2_launch_dry_run_v1",
                "status": "validated_no_cloud_transport_constructed",
                "run_name": view.plan["run_name"],
                "execution_plan_sha256": view.plan_sha256,
                "authorization_sha256": canonical_sha256(checked),
                "cloud_mutation_count": 0,
                "launch_executed": False,
            }
        else:
            transport = factory(
                **common,
                mode="launch",
                authorization=authorization,
                raw_one_shot_nonce=args.nonce,
                now_unix_seconds=args.now_unix_seconds,
            )
            receipt = launch_pair(
                cloud_package_dir=args.cloud_package,
                content_stage_receipt=staged,
                launch_overlay=overlay,
                authorization=authorization,
                raw_one_shot_nonce=args.nonce,
                bucket=args.bucket,
                transport=transport,
                now_unix_seconds=args.now_unix_seconds,
            )
        write_json_once(args.output, receipt)
        return 0
    if args.command == "status":
        transport = factory(**common, mode="receive")
        write_json_once(
            args.output, status_pair(execution_plan=view.plan, transport=transport)
        )
        return 0
    if args.command == "collect":
        authorization = _load_json_file(args.authorization, "launch authorization")
        transport = factory(**common, mode="receive")
        write_json_once(
            args.output,
            collect_pair(
                cloud_package_dir=args.cloud_package,
                authorization=authorization,
                transport=transport,
            ),
        )
        return 0
    if args.command in {
        "cleanup", "cleanup-partial", "cleanup-launched-failure"
    }:
        parsed = uuid.UUID(args.cleanup_nonce)
        if parsed.version != 4 or str(parsed) != args.cleanup_nonce:
            raise ValueError("cleanup nonce must be a canonical UUIDv4")
        if args.command == "cleanup":
            checked: dict[str, Any] = _validate_collection_receipt(
                _load_json_file(args.collection, "collection receipt")
            )
            run_name = checked["run_name"]
            plan_digest = checked["execution_plan_sha256"]
        elif args.command == "cleanup-partial":
            checked = _validate_partial_launch_receipt(
                _load_json_file(args.launch_receipt, "partial launch receipt"),
                execution_plan=view.plan,
            )
            run_name = checked["run_name"]
            plan_digest = checked["execution_plan_sha256"]
        else:
            checked = _validate_full_launch_receipt(
                _load_json_file(args.launch_receipt, "full launch receipt"),
                execution_plan=view.plan,
            )
            run_name = checked["run_name"]
            plan_digest = checked["execution_plan_sha256"]
        if args.dry_run:
            receipt = {
                "schema": "hu_m31_t3_step6d_perfdev_v2_cleanup_dry_run_v1",
                "status": "validated_no_cloud_transport_constructed",
                "run_name": run_name,
                "execution_plan_sha256": plan_digest,
                "cleanup_nonce_sha256": hashlib.sha256(
                    args.cleanup_nonce.encode("ascii")
                ).hexdigest(),
                "cloud_mutation_count": 0,
                "cleanup_executed": False,
                "operator_abort": args.command == "cleanup-launched-failure",
                "scientific_result_claimed": False,
            }
        else:
            if args.command == "cleanup":
                transport = factory(
                    **common,
                    mode="cleanup",
                    collection_receipt=checked,
                )
                receipt = cleanup_collected_pair(
                    collection_receipt=checked, transport=transport
                )
            elif args.command == "cleanup-partial":
                transport = factory(
                    **common,
                    mode="partial-cleanup",
                    partial_launch_receipt=checked,
                )
                receipt = cleanup_partial_launch(
                    partial_launch_receipt=checked,
                    execution_plan=view.plan,
                    transport=transport,
                )
            else:
                transport = factory(
                    **common,
                    mode="launch-failure-cleanup",
                    owned_launch_receipt=checked,
                )
                receipt = closeout_failed_owned_launch_pair(
                    launch_receipt=checked,
                    execution_plan=view.plan,
                    raw_cleanup_nonce=args.cleanup_nonce,
                    explicit_operator_abort=True,
                    transport=transport,
                )
            if args.command != "cleanup-launched-failure":
                receipt["cleanup_nonce_sha256"] = hashlib.sha256(
                    args.cleanup_nonce.encode("ascii")
                ).hexdigest()
                receipt["receipt_content_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_content_sha256"
                    }
                )
        write_json_once(args.output, receipt)
        return 0
    raise AssertionError("unreachable command")


__all__ = [
    "AUTHORIZATION_SCHEMA", "CLEANUP_SCHEMA", "CLOUD_PACKAGE_SCHEMA",
    "CLOUD_READY_SCHEMA", "COLLECTION_SCHEMA", "CloudTransport",
    "EXECUTION_PLAN_SCHEMA", "LAUNCH_RECEIPT_SCHEMA", "ROLE_RESULT_SCHEMA",
    "OWNED_LAUNCH_FAILURE_CLOSEOUT_SCHEMA", "ZERO_CREATED_CLOSEOUT_SCHEMA",
    "GcpHttpTransport", "HttpResponse", "build_cloud_executable_package", "build_launch_authorization",
    "canonical_bytes", "canonical_sha256", "cleanup_collected_pair", "cleanup_partial_launch",
    "closeout_failed_owned_launch_pair", "closeout_zero_created_launch_boundary",
    "collect_pair", "launch_pair", "sha256_file", "validate_cloud_package",
    "validate_execution_plan", "validate_launch_authorization",
    "status_pair", "validate_runtime_binding", "write_json_once", "main",
]
