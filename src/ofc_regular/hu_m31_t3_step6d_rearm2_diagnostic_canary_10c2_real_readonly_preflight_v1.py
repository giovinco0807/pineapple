"""Versioned, fail-closed Google Cloud read-only preflight backend.

This module is the network-facing half of the 10c.2 preflight contract.  It
does not contain an HTTP implementation or an ambient credential lookup.
Instead, callers must inject both an HTTPS transport and an access-token
source.  This keeps tests offline and makes the evidence path independent of
``gcloud`` and subprocess output.

Only a fixed set of Google JSON API resources can be queried, and only with
GET or HEAD.  The collector itself uses GET exclusively.  Receipts never
contain authorization headers, access tokens, raw pagination tokens, or
response bodies.  They bind those requests and responses by SHA-256 while the
bundle contains the normalized facts needed by the existing receiver
preflight.

Even a completely successful observation bundle is deliberately incapable of
authorizing a launch.  It is read-only evidence, not a claim, authorization,
or VM-create surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_receiver_preflight as receiver,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport_contract,
)


BACKEND_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_real_readonly_preflight_backend_v1"
)
QUERY_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_https_query_receipt_v1"
)
TRANSCRIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_https_transcript_v1"
)
BUNDLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_real_readonly_observation_bundle_v1"
)
ACTUAL_RUN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_actual_readonly_run_v1"
)
FAILURE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_10c2_readonly_query_failure_v1"
)

DEFAULT_PROJECT = adapter.DEFAULT_PROJECT
DEFAULT_BUCKET = adapter.DEFAULT_BUCKET
DEFAULT_ZONE = adapter.DEFAULT_ZONE
DEFAULT_REGION = DEFAULT_ZONE.rsplit("-", 1)[0]
MACHINE_TYPE = receiver.MACHINE_TYPE
IMAGE_PROJECT = transport_contract.IMAGE_PROJECT
IMAGE_NAME = transport_contract.IMAGE_NAME
IMAGE_ID = transport_contract.IMAGE_ID
IMAGE_SELF_LINK = transport_contract.IMAGE_SELF_LINK
WORKER_SERVICE_ACCOUNT = transport_contract.WORKER_SERVICE_ACCOUNT
COMPUTE_ENGINE_SERVICE_ID = "6F81-5844-456A"
OFFICIAL_PRICE_SOURCE_URL = "https://cloud.google.com/spot-vms/pricing"
READ_ONLY_METHODS = frozenset({"GET", "HEAD"})
COLLECTOR_METHOD = "GET"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_PAGES = 100
GCS_PAGE_SIZE = 1000
BILLING_PAGE_SIZE = 5000
CPU_QUOTA_METRIC = "C4_CPUS"
SPOT_QUOTA_METRIC = "PREEMPTIBLE_CPUS"
GLOBAL_CPU_QUOTA_ID = "CPUS-ALL-REGIONS-per-project"
GLOBAL_CPU_QUOTA_METRIC = "compute.googleapis.com/cpus_all_regions"
GLOBAL_CPU_QUOTA_LOCATION = "global"
C4_CPU_QUOTA_ID = "CPUS-PER-VM-FAMILY-per-project-region"
C4_CPU_QUOTA_METRIC = "compute.googleapis.com/cpus_per_vm_family"
C4_CPU_QUOTA_VM_FAMILY = "C4"
GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS = {
    "compute_aggregated_instances": "instances",
    "compute_aggregated_reservations": "reservations",
    "compute_aggregated_node_groups": "nodeGroups",
    "compute_aggregated_future_reservations": "futureReservations",
}
GLOBAL_CPU_EMPTY_INVENTORY_KINDS = {
    "compute_aggregated_instances": "compute#instanceAggregatedList",
    "compute_aggregated_reservations": "compute#reservationAggregatedList",
    "compute_aggregated_node_groups": "compute#nodeGroupAggregatedList",
    "compute_aggregated_future_reservations": (
        "compute#futureReservationsAggregatedListResponse"
    ),
}
COMPUTE_AGGREGATED_MAX_RESULTS = 500
STORAGE_OBJECT_VIEWER_ROLE = "roles/storage.objectViewer"
STORAGE_OBJECT_CREATOR_ROLE = "roles/storage.objectCreator"
STORAGE_VIEW_CONDITION_TITLE = "hu-m31-r2diag-direct-v1-object-read"
STORAGE_CREATE_CONDITION_TITLE = "hu-m31-r2diag-direct-v1-stage-object-create"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_DIGITS = re.compile(r"^[0-9]+$")
_STANDARD_MACHINE_TYPE = re.compile(
    r"^(?P<family>[a-z][a-z0-9]*)-"
    r"standard-(?P<vcpu>[1-9][0-9]{0,3})$"
)
_INSTANCE_STATUSES = frozenset(
    {
        "PROVISIONING",
        "STAGING",
        "RUNNING",
        "STOPPING",
        "SUSPENDING",
        "SUSPENDED",
        "REPAIRING",
        "TERMINATED",
    }
)
_ALLOWED_HOSTS = frozenset(
    {
        "storage.googleapis.com",
        "compute.googleapis.com",
        "cloudresourcemanager.googleapis.com",
        "iam.googleapis.com",
        "cloudbilling.googleapis.com",
        "cloudquotas.googleapis.com",
    }
)


class AccessTokenSource(Protocol):
    """Injected credential source; returned token is never recorded."""

    def access_token(self) -> str:
        """Return one OAuth bearer token without logging it."""


@dataclass(frozen=True)
class HttpResponse:
    """Minimal response type accepted from an injected HTTPS transport."""

    status_code: int
    body: bytes
    headers: Mapping[str, str]


class ReadOnlyHttpsTransport(Protocol):
    """An injectable transport with explicit provenance."""

    backend_id: str
    fixture_only: bool
    external_cloud_read_performed: bool

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> HttpResponse:
        """Perform one HTTPS request."""


class ReadOnlyPreflightError(RuntimeError):
    """Fail-closed query or normalization error with secret-free evidence."""

    def __init__(self, code: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(code)
        self.code = code
        self.evidence = _json_copy(evidence)


class EnvironmentAccessTokenSource:
    """Read a caller-provisioned token from one environment variable.

    The source never invokes gcloud, ADC helpers, subprocesses, or a metadata
    server.  A caller may populate the environment variable with a token
    obtained outside this evidence path.
    """

    def __init__(self, environment_variable: str = "GOOGLE_OAUTH_ACCESS_TOKEN") -> None:
        if (
            not isinstance(environment_variable, str)
            or not environment_variable
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", environment_variable)
            is None
        ):
            raise ValueError("token environment-variable name is invalid")
        self._environment_variable = environment_variable

    def access_token(self) -> str:
        value = os.environ.get(self._environment_variable)
        if value is None:
            raise ValueError("required access-token environment variable is absent")
        return _validate_token(value)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class StdlibGoogleJsonReadOnlyTransport:
    """Executable stdlib HTTPS transport with redirects disabled."""

    backend_id = "stdlib-urllib-google-json-readonly-v1"
    fixture_only = False
    external_cloud_read_performed = True

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> HttpResponse:
        if method not in READ_ONLY_METHODS:
            raise ValueError("stdlib transport permits only GET/HEAD")
        if urlsplit(url).scheme != "https":
            raise ValueError("stdlib transport requires HTTPS")
        request = urllib.request.Request(
            url=url,
            headers=dict(headers),
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                return HttpResponse(
                    status_code=int(response.status),
                    body=raw,
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            return HttpResponse(
                status_code=int(exc.code),
                body=raw,
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Do not include the exception text: a custom transport stack may
            # place request headers in it.
            raise OSError("stdlib read-only HTTPS request failed") from None


@dataclass(frozen=True)
class _Target:
    project: str
    bucket: str
    region: str
    zone: str
    machine_type: str
    package_object_prefix: str
    stage_object_prefix: str
    expected_instance_names: tuple[str, ...]
    service_account_email: str
    image_project: str
    image_name: str
    image_id: str
    image_self_link: str


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


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_bytes(value))


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
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _finite_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite number")
    return result


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("JSON response contains a duplicate key")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON response contains non-finite constant {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("response is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("response JSON must be an object")
    return dict(parsed)


def _validate_token(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 8192
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError("access token source returned an invalid token")
    return value


def _gs_parts(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("preflight prefix is not a gs:// URI")
    remainder = uri[5:]
    bucket, separator, object_name = remainder.partition("/")
    if (
        separator != "/"
        or bucket != DEFAULT_BUCKET
        or not object_name
        or object_name.startswith("/")
        or object_name.endswith("/")
    ):
        raise ValueError("preflight prefix escaped the fixed bucket")
    return bucket, object_name + "/"


def _storage_prefix_condition(*, title: str, bucket: str, prefix: str) -> dict[str, str]:
    if (
        not isinstance(title, str)
        or not title
        or not isinstance(bucket, str)
        or bucket != DEFAULT_BUCKET
        or not isinstance(prefix, str)
        or not prefix
        or not prefix.endswith("/")
        or prefix.startswith("/")
        or "\\" in prefix
    ):
        raise ValueError("storage IAM condition inputs escaped the fixed namespace")
    resource_prefix = f"projects/_/buckets/{bucket}/objects/{prefix}"
    return {
        "title": title,
        "expression": f'resource.name.startsWith("{resource_prefix}")',
    }


def _target_from_plan(
    plan: Mapping[str, Any],
    preview: Mapping[str, Any],
    *,
    outer_package_manifest: Mapping[str, Any],
    direct_stage_identity: Mapping[str, Any],
) -> _Target:
    receiver.validate_read_only_preflight_plan(
        plan,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    direct = receiver.validate_direct_stage_identity(
        direct_stage_identity,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
    )
    transport_inputs = direct["inputs"]
    if (
        transport_inputs["worker_service_account"] != WORKER_SERVICE_ACCOUNT
        or transport_inputs["image_project"] != IMAGE_PROJECT
        or transport_inputs["image_name"] != IMAGE_NAME
        or transport_inputs["image_id"] != IMAGE_ID
        or transport_inputs["image_self_link"] != IMAGE_SELF_LINK
        or transport_inputs["zone"] != transport_contract.ZONE
        or transport_inputs["machine_type"] != transport_contract.MACHINE_TYPE
        or plan["real_read_only_preflight_contract_eligible"] is not True
    ):
        raise ValueError(
            "preflight plan is not bound to the exact direct-GCE service "
            "account/image/zone/machine identity"
        )
    requirements = plan["requirements"]
    capacity = requirements["capacity"]
    if (
        capacity["project"] != DEFAULT_PROJECT
        or capacity["region"] != DEFAULT_REGION
        or capacity["zone"] != DEFAULT_ZONE
        or capacity["machine_type"] != MACHINE_TYPE
    ):
        raise ValueError("preflight target escaped fixed project/region/zone/machine")
    layout = receiver.build_direct_v1_remote_layout(
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct,
    )
    package_bucket, package_prefix = _gs_parts(layout["package_prefix"])
    stage_bucket, stage_prefix = _gs_parts(layout["stage_prefix"])
    if package_bucket != stage_bucket:
        raise ValueError("package and stage bucket identities differ")
    current_instance_names = tuple(
        row["instance_name"] for row in layout["job_layouts"]
    )
    attempt_layout = transport_inputs["attempt_layout"]
    if not isinstance(attempt_layout, list):
        raise ValueError("canonical attempt instance layout is missing")
    transport_attempt_names = tuple(
        row["instance_name"] for row in attempt_layout
    )
    expected_jobs = {row["job_id"] for row in transport_inputs["job_layout"]}
    by_attempt = {
        attempt: {
            row["job_id"]
            for row in attempt_layout
            if row["attempt_index"] == attempt
        }
        for attempt in (0, 1)
    }
    current_attempt_names = tuple(
        row["instance_name"]
        for row in attempt_layout
        if row["attempt_index"] == preview["attempt_index"]
    )
    planned_instance_names = tuple(
        row["instance_name"]
        for row in plan["requirements"]["instances"]["expected_instances"]
    )
    if (
        not transport_attempt_names
        or len(transport_attempt_names) != len(set(transport_attempt_names))
        or any(
            _SAFE_NAME.fullmatch(name) is None
            for name in transport_attempt_names
        )
        or by_attempt != {0: expected_jobs, 1: expected_jobs}
        or current_instance_names != current_attempt_names
        or planned_instance_names != transport_attempt_names
    ):
        raise ValueError(
            "attempt0/attempt1 expected instance identities changed"
        )
    return _Target(
        project=capacity["project"],
        bucket=package_bucket,
        region=capacity["region"],
        zone=capacity["zone"],
        machine_type=capacity["machine_type"],
        package_object_prefix=package_prefix,
        stage_object_prefix=stage_prefix,
        expected_instance_names=transport_attempt_names,
        service_account_email=transport_inputs["worker_service_account"],
        image_project=transport_inputs["image_project"],
        image_name=transport_inputs["image_name"],
        image_id=transport_inputs["image_id"],
        image_self_link=transport_inputs["image_self_link"],
    )


def _base_urls(target: _Target) -> dict[str, str]:
    service_account = target.service_account_email
    values = {
        "crm_project": (
            f"https://cloudresourcemanager.googleapis.com/v1/projects/"
            f"{quote(target.project, safe='')}"
        ),
        "compute_project": (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{quote(target.project, safe='')}"
        ),
        "cloud_quotas_global_cpu": (
            f"https://cloudquotas.googleapis.com/v1/projects/"
            f"{quote(target.project, safe='')}/locations/global/services/"
            f"compute.googleapis.com/quotaInfos/{GLOBAL_CPU_QUOTA_ID}"
        ),
        "cloud_quotas_c4_cpu": (
            f"https://cloudquotas.googleapis.com/v1/projects/"
            f"{quote(target.project, safe='')}/locations/global/services/"
            f"compute.googleapis.com/quotaInfos/{C4_CPU_QUOTA_ID}"
        ),
        "gcs_bucket": (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{quote(target.bucket, safe='')}"
        ),
        "gcs_bucket_iam": (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{quote(target.bucket, safe='')}/iam"
        ),
        "compute_image": (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{quote(target.image_project, safe='')}/global/images/"
            f"{quote(target.image_name, safe='')}"
        ),
        "compute_machine": (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{quote(target.project, safe='')}/zones/{quote(target.zone, safe='')}/"
            f"machineTypes/{quote(target.machine_type, safe='')}"
        ),
        "compute_region": (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{quote(target.project, safe='')}/regions/{quote(target.region, safe='')}"
        ),
        "gcs_package_list": (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{quote(target.bucket, safe='')}/o"
        ),
        "gcs_stage_list": (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{quote(target.bucket, safe='')}/o"
        ),
        "billing_skus": (
            f"https://cloudbilling.googleapis.com/v1/services/"
            f"{COMPUTE_ENGINE_SERVICE_ID}/skus"
        ),
    }
    for endpoint_id, resource_collection in (
        GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS.items()
    ):
        values[endpoint_id] = (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{quote(target.project, safe='')}/aggregated/"
            f"{resource_collection}"
        )
    for name in target.expected_instance_names:
        values[f"compute_instance:{name}"] = (
            f"https://compute.googleapis.com/compute/v1/projects/"
            f"{quote(target.project, safe='')}/zones/{quote(target.zone, safe='')}/"
            f"instances/{quote(name, safe='')}"
        )
    values["service_account"] = (
        f"https://iam.googleapis.com/v1/projects/{quote(target.project, safe='')}/"
        f"serviceAccounts/{quote(service_account, safe='')}"
    )
    return values


def _query_for_endpoint(
    endpoint_id: str,
    target: _Target,
    *,
    page_token: str | None,
    retrieved_at_unix_seconds: int | None = None,
) -> list[tuple[str, str]]:
    if endpoint_id == "gcs_bucket_iam":
        if page_token is not None:
            raise ValueError("non-paginated endpoint received a page token")
        return [("optionsRequestedPolicyVersion", "3")]
    if endpoint_id in {"gcs_package_list", "gcs_stage_list"}:
        prefix = (
            target.package_object_prefix
            if endpoint_id == "gcs_package_list"
            else target.stage_object_prefix
        )
        result = [
            ("pageSize", str(GCS_PAGE_SIZE)),
            ("prefix", prefix),
            ("projection", "full"),
        ]
        if page_token is not None:
            result.append(("pageToken", page_token))
        return result
    if endpoint_id == "billing_skus":
        result = [
            ("currencyCode", "USD"),
            ("pageSize", str(BILLING_PAGE_SIZE)),
        ]
        if page_token is not None:
            result.append(("pageToken", page_token))
        return result
    if endpoint_id in GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS:
        if page_token is not None:
            raise ValueError("empty global CPU inventory must fit on one page")
        return [
            ("maxResults", str(COMPUTE_AGGREGATED_MAX_RESULTS)),
            ("returnPartialSuccess", "false"),
            ("includeAllScopes", "true"),
        ]
    if page_token is not None:
        raise ValueError("non-paginated endpoint received a page token")
    return []


def _request_url(
    endpoint_id: str,
    target: _Target,
    *,
    page_token: str | None = None,
    retrieved_at_unix_seconds: int | None = None,
) -> str:
    bases = _base_urls(target)
    if endpoint_id not in bases:
        raise ValueError("endpoint is not in the fixed read-only allowlist")
    query = _query_for_endpoint(
        endpoint_id,
        target,
        page_token=page_token,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
    )
    return bases[endpoint_id] + (f"?{urlencode(query)}" if query else "")


def _sanitized_url(url: str) -> str:
    parts = urlsplit(url)
    query: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "pageToken":
            value = f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
        query.append((key, value))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), "")
    )


def validate_allowed_request(
    *,
    method: str,
    url: str,
    endpoint_id: str,
    target: _Target,
    page_token: str | None = None,
    retrieved_at_unix_seconds: int | None = None,
) -> None:
    """Reject any method, host, path, or query outside the frozen allowlist."""

    if method not in READ_ONLY_METHODS:
        raise ValueError("only HTTPS GET/HEAD are allowed")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in _ALLOWED_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.port not in (None, 443)
        or parts.fragment
    ):
        raise ValueError("request escaped the fixed HTTPS endpoint allowlist")
    expected = _request_url(
        endpoint_id,
        target,
        page_token=page_token,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
    )
    if url != expected:
        raise ValueError("request URL or query changed from the exact allowlist")


def _request_identity(method: str, url: str) -> dict[str, Any]:
    return {
        "method": method,
        "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        "request_body_present": False,
        "authorization_header_present": True,
        "authorization_header_recorded": False,
        "access_token_recorded": False,
    }


def _receipt(
    *,
    endpoint_id: str,
    method: str,
    url: str,
    page_index: int,
    request_page_token: str | None,
    status_code: int | None,
    response_body: bytes | None,
    response_next_page_token: str | None,
    retrieved_at_unix_seconds: int,
    valid_until_unix_seconds: int,
    outcome: str,
) -> dict[str, Any]:
    request_identity = _request_identity(method, url)
    value = {
        "schema": QUERY_RECEIPT_SCHEMA,
        "endpoint_id": endpoint_id,
        "method": method,
        "sanitized_url": _sanitized_url(url),
        "request_sha256": canonical_sha256(request_identity),
        "page_index": page_index,
        "request_page_token_present": request_page_token is not None,
        "request_page_token_sha256": (
            hashlib.sha256(request_page_token.encode("utf-8")).hexdigest()
            if request_page_token is not None
            else None
        ),
        "status_code": status_code,
        "response_sha256": (
            hashlib.sha256(response_body).hexdigest()
            if response_body is not None
            else None
        ),
        "response_bytes": len(response_body) if response_body is not None else 0,
        "response_next_page_token_present": response_next_page_token is not None,
        "response_next_page_token_sha256": (
            hashlib.sha256(response_next_page_token.encode("utf-8")).hexdigest()
            if response_next_page_token is not None
            else None
        ),
        "page_tokens_exhausted": response_next_page_token is None,
        "retrieved_at_unix_seconds": retrieved_at_unix_seconds,
        "valid_until_unix_seconds": valid_until_unix_seconds,
        "outcome": outcome,
        "request_body_present": False,
        "authorization_header_recorded": False,
        "access_token_recorded": False,
        "cloud_mutation_performed": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _failure(
    code: str,
    *,
    endpoint_id: str,
    receipt: Mapping[str, Any],
    prior_receipts: Sequence[Mapping[str, Any]] = (),
) -> ReadOnlyPreflightError:
    clean_receipts = [
        {
            key: child
            for key, child in value.items()
            if key != "_next_page_token"
        }
        for value in [*prior_receipts, receipt]
    ]
    evidence = {
        "schema": FAILURE_SCHEMA,
        "code": code,
        "endpoint_id": endpoint_id,
        "receipts": clean_receipts,
        "cloud_mutation_performed": False,
        "launch_authorized": False,
        "launch_ready": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return ReadOnlyPreflightError(code, evidence)


def _issue_json(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    retrieved_at_unix_seconds: int,
    valid_until_unix_seconds: int,
    expected_status: int,
    page_index: int = 0,
    page_token: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    url = _request_url(
        endpoint_id,
        target,
        page_token=page_token,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
    )
    validate_allowed_request(
        method=COLLECTOR_METHOD,
        url=url,
        endpoint_id=endpoint_id,
        target=target,
        page_token=page_token,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
    )
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    try:
        response = transport.request(
            method=COLLECTOR_METHOD,
            url=url,
            headers=headers,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except (TimeoutError, OSError):
        receipt = _receipt(
            endpoint_id=endpoint_id,
            method=COLLECTOR_METHOD,
            url=url,
            page_index=page_index,
            request_page_token=page_token,
            status_code=None,
            response_body=None,
            response_next_page_token=None,
            retrieved_at_unix_seconds=retrieved_at_unix_seconds,
            valid_until_unix_seconds=valid_until_unix_seconds,
            outcome="transport_failure",
        )
        raise _failure(
            "readonly_https_transport_failure",
            endpoint_id=endpoint_id,
            receipt=receipt,
        ) from None
    if (
        type(response.status_code) is not int
        or not isinstance(response.body, bytes)
        or not isinstance(response.headers, Mapping)
        or len(response.body) > MAX_RESPONSE_BYTES
    ):
        raise ValueError("injected HTTPS transport returned an invalid response")
    if response.status_code != expected_status:
        receipt = _receipt(
            endpoint_id=endpoint_id,
            method=COLLECTOR_METHOD,
            url=url,
            page_index=page_index,
            request_page_token=page_token,
            status_code=response.status_code,
            response_body=response.body,
            response_next_page_token=None,
            retrieved_at_unix_seconds=retrieved_at_unix_seconds,
            valid_until_unix_seconds=valid_until_unix_seconds,
            outcome="unexpected_http_status",
        )
        raise _failure(
            "readonly_https_unexpected_status",
            endpoint_id=endpoint_id,
            receipt=receipt,
        )
    try:
        parsed = dict(_strict_json(response.body))
    except ValueError:
        receipt = _receipt(
            endpoint_id=endpoint_id,
            method=COLLECTOR_METHOD,
            url=url,
            page_index=page_index,
            request_page_token=page_token,
            status_code=response.status_code,
            response_body=response.body,
            response_next_page_token=None,
            retrieved_at_unix_seconds=retrieved_at_unix_seconds,
            valid_until_unix_seconds=valid_until_unix_seconds,
            outcome="invalid_json",
        )
        raise _failure(
            "readonly_https_invalid_json",
            endpoint_id=endpoint_id,
            receipt=receipt,
        ) from None
    raw_next = parsed.get("nextPageToken")
    # Google JSON APIs may serialize the protobuf default as an empty string
    # on the terminal page.  It is semantically identical to an omitted token.
    if raw_next == "":
        raw_next = None
    if raw_next is not None and (
        not isinstance(raw_next, str) or len(raw_next) > 8192
    ):
        receipt = _receipt(
            endpoint_id=endpoint_id,
            method=COLLECTOR_METHOD,
            url=url,
            page_index=page_index,
            request_page_token=page_token,
            status_code=response.status_code,
            response_body=response.body,
            response_next_page_token=None,
            retrieved_at_unix_seconds=retrieved_at_unix_seconds,
            valid_until_unix_seconds=valid_until_unix_seconds,
            outcome="invalid_pagination_token",
        )
        raise _failure(
            "readonly_https_invalid_pagination",
            endpoint_id=endpoint_id,
            receipt=receipt,
        )
    parsed.pop("nextPageToken", None)
    receipt = _receipt(
        endpoint_id=endpoint_id,
        method=COLLECTOR_METHOD,
        url=url,
        page_index=page_index,
        request_page_token=page_token,
        status_code=response.status_code,
        response_body=response.body,
        response_next_page_token=raw_next,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=valid_until_unix_seconds,
        outcome="expected_http_status_and_strict_json",
    )
    return parsed, {**receipt, "_next_page_token": raw_next}


def _transcript(
    *,
    endpoint_id: str,
    receipts: Sequence[Mapping[str, Any]],
    normalized_observation: Any,
) -> dict[str, Any]:
    clean_receipts = [
        {key: child for key, child in receipt.items() if key != "_next_page_token"}
        for receipt in receipts
    ]
    if (
        not clean_receipts
        or clean_receipts[-1]["page_tokens_exhausted"] is not True
    ):
        raise ValueError("paginated transcript did not exhaust page tokens")
    value = {
        "schema": TRANSCRIPT_SCHEMA,
        "endpoint_id": endpoint_id,
        "pages": clean_receipts,
        "page_count": len(clean_receipts),
        "page_tokens_exhausted": True,
        "normalized_observation_sha256": canonical_sha256(normalized_observation),
        "cloud_mutation_performed": False,
    }
    value["transcript_sha256"] = canonical_sha256(value)
    return value


def _single_query(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    retrieved_at_unix_seconds: int,
    valid_until_unix_seconds: int,
    expected_status: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, receipt = _issue_json(
        transport=transport,
        access_token=access_token,
        endpoint_id=endpoint_id,
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=valid_until_unix_seconds,
        expected_status=expected_status,
    )
    if receipt["_next_page_token"] is not None:
        raise _failure(
            "unexpected_pagination_on_single_resource",
            endpoint_id=endpoint_id,
            receipt=receipt,
        )
    transcript = _transcript(
        endpoint_id=endpoint_id,
        receipts=[receipt],
        normalized_observation=payload,
    )
    return payload, transcript


def _paginated_query(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    collection_field: str,
    retrieved_at_unix_seconds: int,
    valid_until_unix_seconds: int,
) -> tuple[list[Any], dict[str, Any]]:
    values: list[Any] = []
    receipts: list[dict[str, Any]] = []
    token: str | None = None
    seen_token_hashes: set[str] = set()
    for page_index in range(MAX_PAGES):
        try:
            payload, receipt = _issue_json(
                transport=transport,
                access_token=access_token,
                endpoint_id=endpoint_id,
                target=target,
                retrieved_at_unix_seconds=retrieved_at_unix_seconds,
                valid_until_unix_seconds=valid_until_unix_seconds,
                expected_status=200,
                page_index=page_index,
                page_token=token,
            )
        except ReadOnlyPreflightError as exc:
            if receipts:
                evidence = dict(exc.evidence)
                evidence["receipts"] = [
                    *[
                        {
                            key: child
                            for key, child in prior.items()
                            if key != "_next_page_token"
                        }
                        for prior in receipts
                    ],
                    *evidence["receipts"],
                ]
                evidence["evidence_sha256"] = canonical_sha256(
                    {
                        key: child
                        for key, child in evidence.items()
                        if key != "evidence_sha256"
                    }
                )
                raise ReadOnlyPreflightError(exc.code, evidence) from None
            raise
        page_values = payload.get(collection_field, [])
        if not isinstance(page_values, list):
            raise _failure(
                "readonly_https_collection_field_invalid",
                endpoint_id=endpoint_id,
                receipt=receipt,
                prior_receipts=receipts,
            )
        values.extend(page_values)
        receipts.append(receipt)
        next_token = receipt["_next_page_token"]
        if next_token is None:
            return values, _transcript(
                endpoint_id=endpoint_id,
                receipts=receipts,
                normalized_observation=values,
            )
        token_hash = hashlib.sha256(next_token.encode("utf-8")).hexdigest()
        if token_hash in seen_token_hashes:
            raise _failure(
                "readonly_https_pagination_token_repeated",
                endpoint_id=endpoint_id,
                receipt=receipt,
                prior_receipts=receipts[:-1],
            )
        seen_token_hashes.add(token_hash)
        token = next_token
    raise _failure(
        "readonly_https_pagination_not_exhausted",
        endpoint_id=endpoint_id,
        receipt=receipts[-1],
        prior_receipts=receipts[:-1],
    )


def _project_facts(
    crm: Mapping[str, Any], compute: Mapping[str, Any], target: _Target
) -> dict[str, Any]:
    project_id = crm.get("projectId")
    project_number = crm.get("projectNumber")
    if (
        project_id != target.project
        or not isinstance(project_number, str)
        or _DIGITS.fullmatch(project_number) is None
        or crm.get("lifecycleState") != "ACTIVE"
    ):
        raise ValueError("Cloud Resource Manager project facts changed")
    compute_id = compute.get("name")
    if compute_id != target.project or not isinstance(compute.get("id"), str):
        raise ValueError("Compute project facts changed")
    return {
        "project_id": project_id,
        "project_number": project_number,
        "lifecycle_state": "ACTIVE",
        "compute_project_name": compute_id,
        "compute_project_numeric_id": compute["id"],
    }


def _instance_inventory_record(
    value: Mapping[str, Any],
    *,
    scope: str,
    target: _Target,
) -> dict[str, Any]:
    zone = scope.split("/", 1)[1]
    zone_link = (
        f"https://www.googleapis.com/compute/v1/projects/{target.project}/"
        f"zones/{zone}"
    )
    name = value.get("name")
    instance_id = value.get("id")
    machine_type = value.get("machineType")
    status = value.get("status")
    self_link = value.get("selfLink")
    scheduling = value.get("scheduling")
    if (
        value.get("kind") != "compute#instance"
        or not isinstance(name, str)
        or _SAFE_NAME.fullmatch(name) is None
        or not isinstance(instance_id, str)
        or _DIGITS.fullmatch(instance_id) is None
        or value.get("zone") != zone_link
        or not isinstance(machine_type, str)
        or status not in _INSTANCE_STATUSES
        or self_link
        != f"{zone_link}/instances/{name}"
        or not isinstance(scheduling, Mapping)
        or value.get("reservationConsumptionInfo") is not None
    ):
        raise ValueError("Compute aggregated instance identity changed")
    machine_prefix = f"{zone_link}/machineTypes/"
    if not machine_type.startswith(machine_prefix):
        raise ValueError("Compute aggregated instance machine type changed")
    machine_name = machine_type[len(machine_prefix) :]
    match = _STANDARD_MACHINE_TYPE.fullmatch(machine_name)
    if match is None:
        raise ValueError(
            "Compute instance machine type cannot be conservatively sized"
        )
    provisioning_model = scheduling.get("provisioningModel")
    preemptible = scheduling.get("preemptible")
    if (
        provisioning_model not in {"STANDARD", "SPOT"}
        or type(preemptible) is not bool
        or (provisioning_model == "SPOT") != preemptible
    ):
        raise ValueError("Compute instance provisioning model changed")
    vcpu = int(match.group("vcpu"))
    family = match.group("family")
    region = zone.rsplit("-", 1)[0]
    return {
        "id": instance_id,
        "name": name,
        "zone": zone,
        "region": region,
        "machine_type": machine_name,
        "machine_family": family,
        "vcpu_upper_bound": vcpu,
        "status": status,
        "provisioning_model": provisioning_model,
        "preemptible": preemptible,
        "target_region": region == target.region,
        "target_region_c4": (
            region == target.region
            and family.upper() == C4_CPU_QUOTA_VM_FAMILY
        ),
    }


def _empty_compute_inventory_facts(
    value: Mapping[str, Any],
    *,
    endpoint_id: str,
    target: _Target,
) -> dict[str, Any]:
    resource_collection = GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS.get(endpoint_id)
    if resource_collection is None:
        raise ValueError("unknown global CPU inventory endpoint")
    expected_kind = GLOBAL_CPU_EMPTY_INVENTORY_KINDS.get(endpoint_id)
    expected_top_fields = {"kind", "id", "items", "selfLink"}
    if resource_collection == "futureReservations":
        expected_top_fields.add("etag")
    expected_id = f"projects/{target.project}/aggregated/{resource_collection}"
    expected_self_link = (
        f"https://www.googleapis.com/compute/v1/projects/{target.project}/"
        f"aggregated/{resource_collection}"
    )
    actual_top_fields = set(value)
    if (
        actual_top_fields
        not in (expected_top_fields, expected_top_fields | {"unreachables"})
        or value.get("kind") != expected_kind
        or value.get("id") != expected_id
        or value.get("selfLink") != expected_self_link
        or (
            "etag" in expected_top_fields
            and (
                not isinstance(value.get("etag"), str)
                or not value["etag"]
            )
        )
    ):
        raise ValueError("Compute aggregated inventory identity changed")
    unreachables = value.get("unreachables", [])
    if unreachables != []:
        raise ValueError("Compute aggregated inventory has unreachable scopes")
    items = value.get("items")
    if not isinstance(items, Mapping) or not items:
        raise ValueError("Compute aggregated inventory scopes are missing")
    warning_count = 0
    nonempty_scope_count = 0
    instance_records: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    seen_instance_names: set[tuple[str, str]] = set()
    for scope, scope_value in items.items():
        if (
            not isinstance(scope, str)
            or (
                scope != "global"
                and re.fullmatch(r"(?:zones|regions)/[a-z0-9-]+", scope) is None
            )
            or not isinstance(scope_value, Mapping)
        ):
            raise ValueError("Compute aggregated empty scope shape changed")
        if set(scope_value) == {"instances"}:
            rows = scope_value.get("instances")
            if (
                resource_collection != "instances"
                or not isinstance(rows, list)
                or not rows
            ):
                raise ValueError(
                    "unsupported nonempty Compute inventory changed"
                )
            nonempty_scope_count += 1
            for raw in rows:
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        "Compute aggregated instance is not an object"
                    )
                record = _instance_inventory_record(
                    raw,
                    scope=scope,
                    target=target,
                )
                name_key = (record["zone"], record["name"])
                if (
                    record["id"] in seen_instance_ids
                    or name_key in seen_instance_names
                ):
                    raise ValueError(
                        "Compute aggregated instance is duplicated"
                    )
                seen_instance_ids.add(record["id"])
                seen_instance_names.add(name_key)
                instance_records.append(record)
            continue
        if (
            set(scope_value) != {"warning"}
            or not isinstance(scope_value.get("warning"), Mapping)
        ):
            raise ValueError("Compute aggregated empty scope shape changed")
        warning = scope_value["warning"]
        data = warning.get("data")
        if (
            warning.get("code") != "NO_RESULTS_ON_PAGE"
            or not isinstance(warning.get("message"), str)
            or not isinstance(data, list)
            or data != [{"key": "scope", "value": scope}]
        ):
            raise ValueError("Compute aggregated empty-scope warning changed")
        warning_count += 1
    return {
        "endpoint_id": endpoint_id,
        "resource_collection": resource_collection,
        "project": target.project,
        "scope_count": len(items),
        "empty_scope_warning_count": warning_count,
        "nonempty_scope_count": nonempty_scope_count,
        "resource_count": len(instance_records),
        "global_vcpu_usage_upper_bound": sum(
            row["vcpu_upper_bound"] for row in instance_records
        ),
        "target_region_c4_vcpu_usage_upper_bound": sum(
            row["vcpu_upper_bound"]
            for row in instance_records
            if row["target_region_c4"]
        ),
        "target_region_spot_vcpu_observed": sum(
            row["vcpu_upper_bound"]
            for row in instance_records
            if row["target_region"] and row["preemptible"]
        ),
        "resource_identities_sha256": canonical_sha256(instance_records),
        "unreachable_scope_count": 0,
        "page_tokens_exhausted": True,
        "scope_names": sorted(items),
    }


def _global_cpu_quota_facts(
    quota_info: Mapping[str, Any],
    inventory_facts: Sequence[Mapping[str, Any]],
    *,
    target: _Target,
    project_number: str,
) -> dict[str, Any]:
    expected_name = (
        f"projects/{project_number}/locations/global/services/"
        f"compute.googleapis.com/quotaInfos/{GLOBAL_CPU_QUOTA_ID}"
    )
    if (
        quota_info.get("name") != expected_name
        or quota_info.get("quotaId") != GLOBAL_CPU_QUOTA_ID
        or quota_info.get("metric") != GLOBAL_CPU_QUOTA_METRIC
        or quota_info.get("service") != "compute.googleapis.com"
        or quota_info.get("isPrecise") is not True
        or quota_info.get("containerType") != "PROJECT"
        or quota_info.get("metricUnit") != "1"
        or quota_info.get("dimensions") not in (None, [])
    ):
        raise ValueError("Cloud Quotas global CPU facts changed")
    dimensions_infos = quota_info.get("dimensionsInfos")
    if not isinstance(dimensions_infos, list) or len(dimensions_infos) != 1:
        raise ValueError("Cloud Quotas global CPU dimensions changed")
    dimensions_info = dimensions_infos[0]
    if (
        not isinstance(dimensions_info, Mapping)
        or dimensions_info.get("dimensions") not in (None, {})
        or dimensions_info.get("applicableLocations")
        != [GLOBAL_CPU_QUOTA_LOCATION]
        or not isinstance(dimensions_info.get("details"), Mapping)
    ):
        raise ValueError("Cloud Quotas global CPU dimension facts changed")
    limit = _positive_cloud_quota_value(
        dimensions_info["details"].get("value"),
        GLOBAL_CPU_QUOTA_ID,
    )
    if (
        not isinstance(inventory_facts, Sequence)
        or len(inventory_facts) != len(GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS)
    ):
        raise ValueError("global CPU empty-inventory proof is incomplete")
    seen_endpoints: set[str] = set()
    expected_scopes: list[str] | None = None
    public_inventory_facts: list[dict[str, Any]] = []
    instance_inventory_facts: Mapping[str, Any] | None = None
    for facts in inventory_facts:
        scope_names = facts.get("scope_names")
        resource_count = facts.get("resource_count")
        nonempty_scope_count = facts.get("nonempty_scope_count")
        empty_scope_warning_count = facts.get(
            "empty_scope_warning_count"
        )
        resource_identities_sha256 = facts.get(
            "resource_identities_sha256"
        )
        global_vcpu_usage = facts.get("global_vcpu_usage_upper_bound")
        target_c4_usage = facts.get(
            "target_region_c4_vcpu_usage_upper_bound"
        )
        target_spot_usage = facts.get(
            "target_region_spot_vcpu_observed"
        )
        if (
            not isinstance(facts, Mapping)
            or facts.get("endpoint_id") not in GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS
            or facts.get("resource_collection")
            != GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS[facts["endpoint_id"]]
            or not isinstance(resource_count, int)
            or resource_count < 0
            or not isinstance(nonempty_scope_count, int)
            or nonempty_scope_count < 0
            or not isinstance(empty_scope_warning_count, int)
            or empty_scope_warning_count < 0
            or not isinstance(resource_identities_sha256, str)
            or _SHA256.fullmatch(resource_identities_sha256) is None
            or not isinstance(global_vcpu_usage, int)
            or global_vcpu_usage < 0
            or not isinstance(target_c4_usage, int)
            or target_c4_usage < 0
            or not isinstance(target_spot_usage, int)
            or target_spot_usage < 0
            or facts.get("unreachable_scope_count") != 0
            or facts.get("page_tokens_exhausted") is not True
            or not isinstance(facts.get("scope_count"), int)
            or facts["scope_count"] <= 0
            or facts["scope_count"]
            != empty_scope_warning_count + nonempty_scope_count
            or facts["endpoint_id"] in seen_endpoints
            or not isinstance(scope_names, list)
            or len(scope_names) != facts["scope_count"]
            or any(not isinstance(scope, str) for scope in scope_names)
        ):
            raise ValueError("global CPU inventory proof changed")
        if facts["resource_collection"] == "instances":
            instance_inventory_facts = facts
        elif (
            resource_count != 0
            or nonempty_scope_count != 0
            or global_vcpu_usage != 0
            or target_c4_usage != 0
            or target_spot_usage != 0
        ):
            raise ValueError(
                "unsupported non-instance Compute inventory is nonempty"
            )
        if expected_scopes is None:
            expected_scopes = list(scope_names)
        elif scope_names != expected_scopes:
            raise ValueError("global CPU inventory scope sets differ")
        seen_endpoints.add(facts["endpoint_id"])
    if seen_endpoints != set(GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS):
        raise ValueError("global CPU empty-inventory endpoints are incomplete")
    if instance_inventory_facts is None:
        raise ValueError("global CPU instance inventory is missing")
    required_scopes = {
        "global",
        f"regions/{target.region}",
        f"zones/{target.zone}",
    }
    if expected_scopes is None or not required_scopes.issubset(expected_scopes):
        raise ValueError("global CPU inventory omitted required scopes")
    for facts in inventory_facts:
        public = {
            key: child for key, child in facts.items() if key != "scope_names"
        }
        public["scope_names_sha256"] = canonical_sha256(facts["scope_names"])
        public_inventory_facts.append(public)
    usage = Decimal(
        instance_inventory_facts["global_vcpu_usage_upper_bound"]
    )
    available = max(Decimal(0), limit - usage)
    return {
        "quota_id": GLOBAL_CPU_QUOTA_ID,
        "metric": GLOBAL_CPU_QUOTA_METRIC,
        "location": GLOBAL_CPU_QUOTA_LOCATION,
        "limit": str(limit),
        "usage": str(usage),
        "available": str(available),
        "usage_kind": "conservative_upper_bound",
        "available_kind": "conservative_lower_bound",
        "limit_source": "cloudquotas.googleapis.com/v1/quotaInfos.get",
        "usage_source": (
            "compute.googleapis.com/compute/v1/aggregated:"
            "conservative-standard-instance-vcpu-upper-bound-with-empty-"
            "reservations-nodeGroups-futureReservations"
        ),
        "usage_inventory_proof": public_inventory_facts,
    }


def _positive_cloud_quota_value(value: Any, label: str) -> Decimal:
    if (
        not isinstance(value, str)
        or _DIGITS.fullmatch(value) is None
        or int(value) <= 0
    ):
        raise ValueError(f"{label} value is not a positive string int64")
    return Decimal(value)


def _regional_c4_quota_facts(
    quota_info: Mapping[str, Any],
    *,
    target: _Target,
    project_number: str,
    usage_inventory_proof: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_name = (
        f"projects/{project_number}/locations/global/services/"
        f"compute.googleapis.com/quotaInfos/{C4_CPU_QUOTA_ID}"
    )
    if (
        quota_info.get("name") != expected_name
        or quota_info.get("quotaId") != C4_CPU_QUOTA_ID
        or quota_info.get("metric") != C4_CPU_QUOTA_METRIC
        or quota_info.get("service") != "compute.googleapis.com"
        or quota_info.get("isPrecise") is not True
        or quota_info.get("containerType") != "PROJECT"
        or quota_info.get("metricUnit") != "1"
        or quota_info.get("dimensions") != ["region", "vm_family"]
    ):
        raise ValueError("Cloud Quotas regional C4 CPU facts changed")
    dimensions_infos = quota_info.get("dimensionsInfos")
    if not isinstance(dimensions_infos, list):
        raise ValueError("Cloud Quotas regional C4 dimensions are missing")
    matches = [
        row
        for row in dimensions_infos
        if isinstance(row, Mapping)
        and row.get("dimensions")
        == {"region": target.region, "vm_family": C4_CPU_QUOTA_VM_FAMILY}
    ]
    if len(matches) != 1:
        raise ValueError("Cloud Quotas regional C4 dimension is not unique")
    match = matches[0]
    if (
        match.get("applicableLocations") != [target.region]
        or not isinstance(match.get("details"), Mapping)
    ):
        raise ValueError("Cloud Quotas regional C4 dimension facts changed")
    limit = _positive_cloud_quota_value(
        match["details"].get("value"),
        C4_CPU_QUOTA_ID,
    )
    if (
        not isinstance(usage_inventory_proof, Sequence)
        or len(usage_inventory_proof)
        != len(GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS)
    ):
        raise ValueError("regional C4 usage proof is incomplete")
    instance_proofs = [
        row
        for row in usage_inventory_proof
        if isinstance(row, Mapping)
        and row.get("resource_collection") == "instances"
    ]
    if len(instance_proofs) != 1:
        raise ValueError("regional C4 instance proof is not unique")
    usage_value = instance_proofs[0].get(
        "target_region_c4_vcpu_usage_upper_bound"
    )
    target_spot_usage = instance_proofs[0].get(
        "target_region_spot_vcpu_observed"
    )
    if (
        not isinstance(usage_value, int)
        or usage_value < 0
        or not isinstance(target_spot_usage, int)
        or target_spot_usage < 0
    ):
        raise ValueError("regional C4 instance proof changed")
    usage = Decimal(usage_value)
    available = max(Decimal(0), limit - usage)
    proof_sha256 = canonical_sha256(usage_inventory_proof)
    return {
        "quota_id": C4_CPU_QUOTA_ID,
        "provider_metric": CPU_QUOTA_METRIC,
        "metric": C4_CPU_QUOTA_METRIC,
        "dimensions": {
            "region": target.region,
            "vm_family": C4_CPU_QUOTA_VM_FAMILY,
        },
        "limit": str(limit),
        "usage": str(usage),
        "available": str(available),
        "usage_kind": "conservative_upper_bound",
        "available_kind": "conservative_lower_bound",
        "limit_source": "cloudquotas.googleapis.com/v1/quotaInfos.get",
        "usage_source": (
            "compute.googleapis.com/compute/v1/aggregated:"
            "conservative-target-region-c4-instance-vcpu-upper-bound-with-"
            "empty-reservations-nodeGroups-futureReservations"
        ),
        "usage_inventory_proof_sha256": proof_sha256,
        "target_region_spot_vcpu_observed": target_spot_usage,
    }


def _bucket_facts(
    bucket: Mapping[str, Any],
    iam: Mapping[str, Any],
    *,
    target: _Target,
    project_number: str,
    service_account_email: str,
    expected_worker_storage_bindings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if (
        bucket.get("name") != target.bucket
        or str(bucket.get("projectNumber")) != project_number
        or not isinstance(bucket.get("location"), str)
        or not bucket["location"]
        or not isinstance(bucket.get("metageneration"), str)
        or _DIGITS.fullmatch(bucket["metageneration"]) is None
    ):
        raise ValueError("GCS bucket facts changed")
    bindings = iam.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("GCS IAM policy bindings are missing")
    if iam.get("version") != 3:
        raise ValueError("GCS IAM policy version must be exactly 3 for conditions")
    member = f"serviceAccount:{service_account_email}"
    roles: list[str] = []
    if expected_worker_storage_bindings is None:
        expected_conditions = {
            STORAGE_OBJECT_VIEWER_ROLE: _storage_prefix_condition(
                title=STORAGE_VIEW_CONDITION_TITLE,
                bucket=target.bucket,
                prefix=f"{transport_contract.DIRECT_NAMESPACE}/",
            ),
            STORAGE_OBJECT_CREATOR_ROLE: _storage_prefix_condition(
                title=STORAGE_CREATE_CONDITION_TITLE,
                bucket=target.bucket,
                prefix=target.stage_object_prefix,
            ),
        }
    else:
        expected_conditions: dict[str, dict[str, Any]] = {}
        expected_resource = f"projects/_/buckets/{target.bucket}"
        if len(expected_worker_storage_bindings) != 2:
            raise ValueError("Step11 worker storage binding count changed")
        for raw_binding in expected_worker_storage_bindings:
            if not isinstance(raw_binding, Mapping):
                raise ValueError("Step11 worker storage binding is not an object")
            role = raw_binding.get("role")
            condition = raw_binding.get("condition")
            if (
                raw_binding.get("resource") != expected_resource
                or raw_binding.get("member") != member
                or not isinstance(role, str)
                or role in expected_conditions
                or not isinstance(condition, Mapping)
                or set(condition) != {"title", "expression"}
            ):
                raise ValueError("Step11 worker storage binding changed")
            expected_conditions[role] = dict(condition)
        expected_roles = {
            f"projects/{target.project}/roles/ofcM31T3ObjectReaderV1",
            f"projects/{target.project}/roles/ofcM31T3ResultCreatorV1",
        }
        if set(expected_conditions) != expected_roles:
            raise ValueError("Step11 worker custom storage roles changed")
    qualifying_conditioned_bindings: list[dict[str, Any]] = []
    targeted_binding_violations: list[str] = []
    normalized_bindings: list[dict[str, Any]] = []
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise ValueError("GCS IAM binding is not an object")
        role = raw.get("role")
        members = raw.get("members")
        if not isinstance(role, str) or not isinstance(members, list) or any(
            not isinstance(value, str) for value in members
        ):
            raise ValueError("GCS IAM binding fields changed")
        condition = raw.get("condition")
        if condition is not None and not isinstance(condition, Mapping):
            raise ValueError("GCS IAM binding condition is not an object")
        normalized = {
            "role": role,
            "members": sorted(members),
            "condition": dict(condition) if condition is not None else None,
        }
        normalized_bindings.append(normalized)
        if member in members:
            roles.append(role)
            expected_condition = expected_conditions.get(role)
            if (
                expected_condition is None
                or members != [member]
                or condition is None
                or set(condition) != {"title", "expression"}
                or dict(condition) != expected_condition
            ):
                targeted_binding_violations.append(role)
                continue
            qualifying_conditioned_bindings.append(
                {
                    "role": role,
                    "member": member,
                    "condition": expected_condition,
                }
            )
    qualifying_roles = [
        row["role"] for row in qualifying_conditioned_bindings
    ]
    if (
        targeted_binding_violations
        or sorted(qualifying_roles) != sorted(expected_conditions)
        or len(qualifying_roles) != len(set(qualifying_roles))
    ):
        raise ValueError(
            "dedicated diagnostic service account lacks the two exact conditioned "
            "least-privilege storage bindings or has a broader bucket binding"
        )
    return {
        "bucket": target.bucket,
        "project_number": project_number,
        "location": bucket["location"],
        "location_type": bucket.get("locationType"),
        "storage_class": bucket.get("storageClass"),
        "metageneration": bucket["metageneration"],
        "uniform_bucket_level_access_enabled": bool(
            (
                bucket.get("iamConfiguration", {})
                .get("uniformBucketLevelAccess", {})
                .get("enabled", False)
            )
        ),
        "iam_policy_version": iam.get("version", 1),
        "iam_policy_etag": iam.get("etag"),
        "iam_bindings_sha256": canonical_sha256(
            sorted(
                normalized_bindings,
                key=lambda row: (
                    row["role"],
                    row["members"],
                    canonical_sha256(row["condition"]),
                ),
            )
        ),
        "service_account_member": member,
        "service_account_roles": sorted(set(roles)),
        "service_account_storage_access_sufficient": True,
        "required_worker_storage_permissions": [
            "storage.objects.get",
            "storage.objects.create",
        ],
        "qualifying_conditioned_bindings": sorted(
            qualifying_conditioned_bindings,
            key=lambda row: row["role"],
        ),
    }


def _service_account_facts(
    value: Mapping[str, Any], *, expected_email: str, expected_project: str
) -> dict[str, Any]:
    name = value.get("name")
    unique_id = value.get("uniqueId")
    disabled = value.get("disabled", False)
    if (
        value.get("email") != expected_email
        or value.get("projectId") != expected_project
        or disabled is not False
        or not isinstance(name, str)
        or name != f"projects/{expected_project}/serviceAccounts/{expected_email}"
        or not isinstance(unique_id, str)
        or _DIGITS.fullmatch(unique_id) is None
        or value.get("oauth2ClientId") != unique_id
    ):
        raise ValueError("dedicated diagnostic service-account facts changed")
    return {
        "name": name,
        "email": expected_email,
        "project_id": expected_project,
        "unique_id": unique_id,
        "disabled": False,
        "oauth2_client_id": unique_id,
    }


def _image_facts(
    value: Mapping[str, Any], target: _Target
) -> dict[str, Any]:
    deprecated = value.get("deprecated")
    deprecation_state = "ACTIVE"
    replacement: str | None = None
    if deprecated is not None:
        if not isinstance(deprecated, Mapping):
            raise ValueError("pinned Compute image deprecation facts changed")
        deprecation_state = deprecated.get("state")
        replacement = deprecated.get("replacement")
        replacement_prefix = (
            f"https://www.googleapis.com/compute/v1/projects/"
            f"{target.image_project}/global/images/"
        )
        if (
            deprecation_state != "DEPRECATED"
            or not isinstance(replacement, str)
            or not replacement.startswith(replacement_prefix)
            or replacement == target.image_self_link
        ):
            raise ValueError("pinned Compute image is no longer launchable")
    if (
        value.get("name") != target.image_name
        or str(value.get("id")) != target.image_id
        or value.get("status") != "READY"
        or value.get("selfLink") != target.image_self_link
    ):
        raise ValueError("pinned Compute image facts changed")
    return {
        "project": target.image_project,
        "name": target.image_name,
        "id": value["id"],
        "status": "READY",
        "self_link": value["selfLink"],
        "architecture": value.get("architecture"),
        "deprecated": deprecation_state == "DEPRECATED",
        "deprecation_state": deprecation_state,
        "replacement": replacement,
        "launch_warning_expected": deprecation_state == "DEPRECATED",
    }


def _machine_facts(value: Mapping[str, Any], target: _Target) -> dict[str, Any]:
    expected_suffix = (
        f"/projects/{target.project}/zones/{target.zone}/machineTypes/"
        f"{target.machine_type}"
    )
    zone_value = value.get("zone")
    accepted_zone_values = {
        target.zone,
        (
            f"https://www.googleapis.com/compute/v1/projects/"
            f"{target.project}/zones/{target.zone}"
        ),
    }
    if (
        value.get("name") != target.machine_type
        or value.get("guestCpus") != 16
        or value.get("memoryMb") != 61440
        or not str(value.get("selfLink", "")).endswith(expected_suffix)
        or zone_value not in accepted_zone_values
    ):
        raise ValueError("c4-standard-16 machine facts changed")
    return {
        "name": target.machine_type,
        "zone": target.zone,
        "guest_cpus": 16,
        "memory_mb": 61440,
        "memory_gib": "60",
        "self_link": value["selfLink"],
    }


def _quota_facts(
    value: Mapping[str, Any],
    target: _Target,
    *,
    project_facts: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        value.get("name") != target.region
        or not str(value.get("selfLink", "")).endswith(
            f"/projects/{target.project}/regions/{target.region}"
        )
    ):
        raise ValueError("Compute region facts changed")
    quotas = value.get("quotas")
    if not isinstance(quotas, list):
        raise ValueError("Compute regional quotas are missing")
    found: dict[str, tuple[Decimal, Decimal]] = {}
    for raw in quotas:
        if not isinstance(raw, Mapping):
            raise ValueError("Compute quota entry is not an object")
        metric = raw.get("metric")
        if metric != SPOT_QUOTA_METRIC:
            continue
        if metric in found:
            raise ValueError("Compute quota metric is duplicated")
        limit = _finite_decimal(raw.get("limit"), f"{metric} limit")
        usage = _finite_decimal(raw.get("usage"), f"{metric} usage")
        if limit < 0 or usage < 0:
            raise ValueError("Compute quota cannot be negative")
        found[metric] = (limit, usage)
    if set(found) != {SPOT_QUOTA_METRIC}:
        raise ValueError("required Spot regional quota metric is missing")
    project_c4 = project_facts.get("c4_cpu_quota")
    if not isinstance(project_c4, Mapping):
        raise ValueError("regional C4 CPU quota facts are missing")
    observed_target_spot = project_c4.get(
        "target_region_spot_vcpu_observed"
    )
    if (
        not isinstance(observed_target_spot, int)
        or observed_target_spot < 0
        or found[SPOT_QUOTA_METRIC][1] < observed_target_spot
    ):
        raise ValueError(
            "regional Spot quota usage is below observed Spot instances"
        )
    if (
        project_c4.get("provider_metric") != CPU_QUOTA_METRIC
        or project_c4.get("metric") != C4_CPU_QUOTA_METRIC
        or project_c4.get("dimensions")
        != {"region": target.region, "vm_family": C4_CPU_QUOTA_VM_FAMILY}
    ):
        raise ValueError("regional C4 CPU quota facts are missing")
    c4_limit = _finite_decimal(project_c4.get("limit"), "regional C4 CPU limit")
    c4_usage = _finite_decimal(project_c4.get("usage"), "regional C4 CPU usage")
    c4_available = _finite_decimal(
        project_c4.get("available"), "regional C4 CPU available"
    )
    if (
        c4_limit <= 0
        or c4_usage < 0
        or c4_available != max(Decimal(0), c4_limit - c4_usage)
    ):
        raise ValueError("regional C4 CPU quota facts changed")
    found[CPU_QUOTA_METRIC] = (c4_limit, c4_usage)
    available_by_metric = {
        metric: max(Decimal(0), limit - usage)
        for metric, (limit, usage) in found.items()
    }
    project_global = project_facts.get("global_cpu_quota")
    if (
        not isinstance(project_global, Mapping)
        or project_global.get("metric") != GLOBAL_CPU_QUOTA_METRIC
    ):
        raise ValueError("project global CPU quota facts are missing")
    global_available = _finite_decimal(
        project_global.get("available"), "global CPU available"
    )
    available = min(*available_by_metric.values(), global_available)
    available_int = math.floor(available)
    return {
        "region": target.region,
        "quota_metrics": {
            metric: {
                "limit": str(found[metric][0]),
                "usage": str(found[metric][1]),
                "available": str(available_by_metric[metric]),
                **(
                    {
                        "quota_id": project_c4["quota_id"],
                        "provider_metric": project_c4["provider_metric"],
                        "cloud_quotas_metric": project_c4["metric"],
                        "dimensions": project_c4["dimensions"],
                        "limit_source": project_c4["limit_source"],
                        "usage_source": project_c4["usage_source"],
                        "usage_kind": project_c4["usage_kind"],
                        "available_kind": project_c4["available_kind"],
                        "usage_inventory_proof_sha256": project_c4[
                            "usage_inventory_proof_sha256"
                        ],
                    }
                    if metric == CPU_QUOTA_METRIC
                    else {}
                ),
            }
            for metric in sorted(found)
        },
        "project_global_quota_metric": {
            "quota_id": project_global["quota_id"],
            "metric": GLOBAL_CPU_QUOTA_METRIC,
            "location": project_global["location"],
            "limit": project_global["limit"],
            "usage": project_global["usage"],
            "available": project_global["available"],
            "limit_source": project_global["limit_source"],
            "usage_source": project_global["usage_source"],
            "usage_kind": project_global["usage_kind"],
            "available_kind": project_global["available_kind"],
            "usage_inventory_proof": project_global[
                "usage_inventory_proof"
            ],
        },
        "available_vcpu": available_int,
        "provider_metric_mapping": receiver.CAPACITY_PROVIDER_METRIC_MAPPING,
    }


def _gcs_objects(
    values: Sequence[Any],
    *,
    target: _Target,
    object_prefix: str,
    require_sha256_metadata: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    receiver_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError("GCS object-list item is not an object")
        name = raw.get("name")
        generation = raw.get("generation")
        metageneration = raw.get("metageneration")
        size = raw.get("size")
        if (
            raw.get("bucket") != target.bucket
            or not isinstance(name, str)
            or not name.startswith(object_prefix)
            or name == object_prefix.rstrip("/")
            or name in seen_names
            or not isinstance(generation, str)
            or _DIGITS.fullmatch(generation) is None
            or int(generation) <= 0
            or not isinstance(metageneration, str)
            or _DIGITS.fullmatch(metageneration) is None
            or int(metageneration) <= 0
            or not isinstance(size, str)
            or _DIGITS.fullmatch(size) is None
        ):
            raise ValueError("GCS object-list generation metadata changed")
        seen_names.add(name)
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("GCS object custom metadata is not an object")
        sha256 = metadata.get("sha256")
        if require_sha256_metadata:
            sha256 = _sha(sha256, "GCS object metadata sha256")
        elif sha256 is not None:
            sha256 = _sha(sha256, "GCS object metadata sha256")
        row = {
            "uri": f"gs://{target.bucket}/{name}",
            "name": name,
            "generation": int(generation),
            "metageneration": int(metageneration),
            "bytes": int(size),
            "sha256_metadata": sha256,
            "etag": raw.get("etag"),
            "updated": raw.get("updated"),
            "md5_hash": raw.get("md5Hash"),
            "crc32c": raw.get("crc32c"),
        }
        inventory.append(row)
        if require_sha256_metadata:
            if not isinstance(row["crc32c"], str) or not row["crc32c"]:
                raise ValueError("GCS package object crc32c is missing")
            if not isinstance(row["etag"], str) or not row["etag"]:
                raise ValueError("GCS package object etag is missing")
            receiver_rows.append(
                {
                    "uri": row["uri"],
                    "generation": row["generation"],
                    "metageneration": row["metageneration"],
                    "sha256": sha256,
                    "bytes": row["bytes"],
                    "crc32c": row["crc32c"],
                    "etag": row["etag"],
                }
            )
    inventory.sort(key=lambda row: row["uri"])
    receiver_rows.sort(key=lambda row: row["uri"])
    return inventory, receiver_rows


def _unit_price(pricing_expression: Mapping[str, Any], expected_unit: str) -> Decimal:
    if (
        pricing_expression.get("usageUnit") != expected_unit
        or _finite_decimal(
            pricing_expression.get("displayQuantity", 1), "display quantity"
        )
        != Decimal(1)
    ):
        raise ValueError("Billing SKU pricing unit changed")
    rates = pricing_expression.get("tieredRates")
    if not isinstance(rates, list) or len(rates) != 1:
        raise ValueError("Billing SKU must have exactly one zero-based rate")
    rate = rates[0]
    if (
        not isinstance(rate, Mapping)
        or _finite_decimal(rate.get("startUsageAmount", 0), "tier start")
        != Decimal(0)
        or not isinstance(rate.get("unitPrice"), Mapping)
    ):
        raise ValueError("Billing SKU tier changed")
    price = rate["unitPrice"]
    if price.get("currencyCode") != "USD":
        raise ValueError("Billing SKU currency changed")
    units = _finite_decimal(price.get("units", 0), "unit-price units")
    nanos = _finite_decimal(price.get("nanos", 0), "unit-price nanos")
    if units < 0 or nanos < 0 or nanos >= Decimal(1_000_000_000):
        raise ValueError("Billing SKU price is out of range")
    return units + nanos / Decimal(1_000_000_000)


def _price_facts(
    skus: Sequence[Any],
    *,
    target: _Target,
    machine: Mapping[str, Any],
    retrieved_at_unix_seconds: int,
) -> dict[str, Any]:
    matches: dict[
        str, list[tuple[Mapping[str, Any], Decimal, int]]
    ] = {
        "CPU": [],
        "RAM": [],
    }
    for raw in skus:
        if not isinstance(raw, Mapping):
            raise ValueError("Billing SKU is not an object")
        category = raw.get("category")
        if not isinstance(category, Mapping):
            continue
        resource_group = category.get("resourceGroup")
        service_regions = raw.get("serviceRegions")
        if (
            resource_group not in matches
            or category.get("serviceDisplayName") != "Compute Engine"
            or category.get("resourceFamily") != "Compute"
            or category.get("usageType") != "Preemptible"
            or not isinstance(service_regions, list)
            or service_regions != [target.region]
        ):
            continue
        description = raw.get("description")
        if (
            not isinstance(description, str)
            or re.search(r"\bc4\b", description.casefold()) is None
        ):
            continue
        lowered = description.casefold()
        if resource_group == "CPU" and "core" not in lowered:
            continue
        if resource_group == "RAM" and not any(
            token in lowered for token in ("ram", "memory")
        ):
            continue
        pricing = raw.get("pricingInfo")
        if not isinstance(pricing, list) or not pricing:
            raise ValueError("Billing SKU pricingInfo time-series is missing")
        applicable: list[tuple[int, Mapping[str, Any]]] = []
        seen_effective: set[int] = set()
        for info in pricing:
            if not isinstance(info, Mapping) or not isinstance(
                info.get("pricingExpression"), Mapping
            ):
                raise ValueError("Billing SKU pricing info changed")
            effective = info.get("effectiveTime")
            if not isinstance(effective, str) or not effective.endswith("Z"):
                raise ValueError("Billing SKU effectiveTime is missing")
            try:
                parsed_effective = datetime.fromisoformat(
                    effective[:-1] + "+00:00"
                )
                effective_unix = int(parsed_effective.timestamp())
            except (OverflowError, ValueError) as exc:
                raise ValueError("Billing SKU effectiveTime is invalid") from exc
            if effective_unix < 0 or effective_unix in seen_effective:
                raise ValueError("Billing SKU effectiveTime is invalid or duplicated")
            seen_effective.add(effective_unix)
            if effective_unix <= retrieved_at_unix_seconds:
                applicable.append((effective_unix, info))
        if not applicable:
            raise ValueError("Billing SKU has no applicable price at retrieval time")
        effective_unix, info = max(applicable, key=lambda row: row[0])
        expected_unit = "h" if resource_group == "CPU" else "GiBy.h"
        matches[resource_group].append(
            (
                raw,
                _unit_price(info["pricingExpression"], expected_unit),
                effective_unix,
            )
        )
    if len(matches["CPU"]) != 1 or len(matches["RAM"]) != 1:
        raise ValueError("exact Tokyo C4 preemptible CPU/RAM SKUs are ambiguous")
    cpu_sku, cpu_unit, cpu_effective = matches["CPU"][0]
    ram_sku, ram_unit, ram_effective = matches["RAM"][0]
    vcpu = Decimal(machine["guest_cpus"])
    memory_gib = Decimal(machine["memory_mb"]) / Decimal(1024)
    total = vcpu * cpu_unit + memory_gib * ram_unit

    def sku_fact(
        raw: Mapping[str, Any],
        unit_price: Decimal,
        expected_group: str,
        effective_unix: int,
    ) -> dict[str, Any]:
        name = raw.get("name")
        sku_id = raw.get("skuId")
        if (
            not isinstance(name, str)
            or not name.startswith(
                f"services/{COMPUTE_ENGINE_SERVICE_ID}/skus/"
            )
            or not isinstance(sku_id, str)
            or not sku_id
        ):
            raise ValueError("Billing SKU identity changed")
        return {
            "name": name,
            "sku_id": sku_id,
            "description": raw["description"],
            "resource_group": expected_group,
            "usage_type": "Preemptible",
            "service_region": target.region,
            "service_regions": [target.region],
            "unit": "h" if expected_group == "CPU" else "GiBy.h",
            "unit_price_usd": format(unit_price, "f"),
            "effective_at_unix_seconds": effective_unix,
            "catalog_source_url": (
                "https://cloudbilling.googleapis.com/v1/services/"
                f"{COMPUTE_ENGINE_SERVICE_ID}/skus"
            ),
        }

    return {
        "service_id": COMPUTE_ENGINE_SERVICE_ID,
        "region": target.region,
        "machine_type": target.machine_type,
        "provisioning_model": "SPOT",
        "currency": "USD",
        "vcpu": int(vcpu),
        "memory_gib": format(memory_gib, "f"),
        "cpu_sku": sku_fact(cpu_sku, cpu_unit, "CPU", cpu_effective),
        "ram_sku": sku_fact(ram_sku, ram_unit, "RAM", ram_effective),
        "formula": "vcpu*cpu_spot_usd_per_core_hour+memory_gib*ram_spot_usd_per_giby_hour",
        "hourly_price_usd_decimal": format(total, "f"),
        "hourly_price_usd_per_vm": float(total),
        "sku_effective_at_unix_seconds": max(cpu_effective, ram_effective),
        "official_source_url": OFFICIAL_PRICE_SOURCE_URL,
        "catalog_source_url": (
            "https://cloudbilling.googleapis.com/v1/services/"
            f"{COMPUTE_ENGINE_SERVICE_ID}/skus"
        ),
    }


def _transport_provenance(
    transport: ReadOnlyHttpsTransport,
) -> tuple[str, bool, bool, bool]:
    backend_id = getattr(transport, "backend_id", None)
    fixture_only = getattr(transport, "fixture_only", None)
    external = getattr(transport, "external_cloud_read_performed", None)
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or len(backend_id) > 128
    ):
        raise ValueError("HTTPS transport backend_id changed")
    _strict_bool(fixture_only, None, "transport fixture_only")
    _strict_bool(external, None, "transport external query")
    if fixture_only and external:
        raise ValueError("fixture transport cannot claim an external cloud query")
    concrete_stdlib = type(transport) is StdlibGoogleJsonReadOnlyTransport
    if concrete_stdlib and (
        backend_id != StdlibGoogleJsonReadOnlyTransport.backend_id
        or fixture_only is not False
        or external is not True
    ):
        raise ValueError("concrete stdlib transport provenance changed")
    return backend_id, fixture_only, external, concrete_stdlib


def _credential_provenance(token_source: AccessTokenSource) -> dict[str, Any]:
    """Describe only the collector boundary, never guess token acquisition."""

    environment_source = type(token_source) is EnvironmentAccessTokenSource
    return {
        "collector_token_source": (
            "environment_access_token_source_v1"
            if environment_source
            else "injected_access_token_source"
        ),
        "credential_supplied_to_collector_externally": True,
        "credential_acquisition_method": "external_unknown",
        "credential_acquisition_subprocess_used": None,
        "credential_acquisition_gcloud_used": None,
        "collector_subprocess_used": False,
        "collector_gcloud_used": False,
    }


def collect_read_only_observations(
    *,
    plan: Mapping[str, Any],
    preview: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any],
    direct_stage_identity: Mapping[str, Any],
    transport: ReadOnlyHttpsTransport,
    token_source: AccessTokenSource,
    retrieved_at_unix_seconds: int,
    expected_worker_storage_bindings: Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Collect hash-bound facts without creating launch authority."""

    _strict_int(
        retrieved_at_unix_seconds, "retrieved_at_unix_seconds", minimum=0
    )
    (
        backend_id,
        fixture_only,
        external,
        concrete_stdlib,
    ) = _transport_provenance(transport)
    target = _target_from_plan(
        plan,
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    token = _validate_token(token_source.access_token())
    capacity_valid_until = (
        retrieved_at_unix_seconds
        + receiver.CAPACITY_OBSERVATION_MAX_AGE_SECONDS
    )
    prefix_valid_until = (
        retrieved_at_unix_seconds + receiver.PREFIX_OBSERVATION_MAX_AGE_SECONDS
    )
    price_valid_until = (
        retrieved_at_unix_seconds + receiver.PRICE_OBSERVATION_MAX_AGE_SECONDS
    )
    transcripts: list[dict[str, Any]] = []

    crm, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="crm_project",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    compute_project, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="compute_project",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    project_facts = _project_facts(crm, compute_project, target)
    quota_info, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="cloud_quotas_global_cpu",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    c4_quota_info, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="cloud_quotas_c4_cpu",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    global_inventory_facts: list[dict[str, Any]] = []
    for inventory_endpoint in GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS:
        inventory, transcript = _single_query(
            transport=transport,
            access_token=token,
            endpoint_id=inventory_endpoint,
            target=target,
            retrieved_at_unix_seconds=retrieved_at_unix_seconds,
            valid_until_unix_seconds=capacity_valid_until,
        )
        transcripts.append(transcript)
        global_inventory_facts.append(
            _empty_compute_inventory_facts(
                inventory,
                endpoint_id=inventory_endpoint,
                target=target,
            )
        )
    project_facts["global_cpu_quota"] = _global_cpu_quota_facts(
        quota_info,
        global_inventory_facts,
        target=target,
        project_number=project_facts["project_number"],
    )
    project_facts["c4_cpu_quota"] = _regional_c4_quota_facts(
        c4_quota_info,
        target=target,
        project_number=project_facts["project_number"],
        usage_inventory_proof=project_facts["global_cpu_quota"][
            "usage_inventory_proof"
        ],
    )
    service_account_email = target.service_account_email

    bucket, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="gcs_bucket",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=prefix_valid_until,
    )
    transcripts.append(transcript)
    bucket_iam, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="gcs_bucket_iam",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=prefix_valid_until,
    )
    transcripts.append(transcript)
    service_account, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="service_account",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    bucket_facts = _bucket_facts(
        bucket,
        bucket_iam,
        target=target,
        project_number=project_facts["project_number"],
        service_account_email=service_account_email,
        expected_worker_storage_bindings=expected_worker_storage_bindings,
    )
    service_account_facts = _service_account_facts(
        service_account,
        expected_email=service_account_email,
        expected_project=target.project,
    )

    image, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="compute_image",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    machine, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="compute_machine",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    region, transcript = _single_query(
        transport=transport,
        access_token=token,
        endpoint_id="compute_region",
        target=target,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=capacity_valid_until,
    )
    transcripts.append(transcript)
    image_facts = _image_facts(image, target)
    machine_facts = _machine_facts(machine, target)
    quota_facts = _quota_facts(
        region,
        target,
        project_facts=project_facts,
    )

    instance_facts: list[dict[str, Any]] = []
    for instance_name in target.expected_instance_names:
        payload, transcript = _single_query(
            transport=transport,
            access_token=token,
            endpoint_id=f"compute_instance:{instance_name}",
            target=target,
            retrieved_at_unix_seconds=retrieved_at_unix_seconds,
            valid_until_unix_seconds=capacity_valid_until,
            expected_status=404,
        )
        transcripts.append(transcript)
        error = payload.get("error")
        if not isinstance(error, Mapping) or error.get("code") != 404:
            raise ValueError("expected instance GET404 response changed")
        instance_facts.append(
            {
                "instance_name": instance_name,
                "zone": target.zone,
                "http_status": 404,
                "absent": True,
                "error_status": error.get("status"),
            }
        )

    package_items, transcript = _paginated_query(
        transport=transport,
        access_token=token,
        endpoint_id="gcs_package_list",
        target=target,
        collection_field="items",
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=prefix_valid_until,
    )
    transcripts.append(transcript)
    stage_items, transcript = _paginated_query(
        transport=transport,
        access_token=token,
        endpoint_id="gcs_stage_list",
        target=target,
        collection_field="items",
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=prefix_valid_until,
    )
    transcripts.append(transcript)
    package_inventory, package_receiver_rows = _gcs_objects(
        package_items,
        target=target,
        object_prefix=target.package_object_prefix,
        require_sha256_metadata=True,
    )
    stage_inventory, _ = _gcs_objects(
        stage_items,
        target=target,
        object_prefix=target.stage_object_prefix,
        require_sha256_metadata=False,
    )
    expected_package_rows = plan["requirements"]["prefix"][
        "expected_package_objects"
    ]
    package_by_uri = {row["uri"]: row for row in package_receiver_rows}
    if len(package_by_uri) != len(package_receiver_rows):
        raise ValueError("GCS package listing contains duplicate URIs")
    expected_by_uri = {row["uri"]: row for row in expected_package_rows}
    if len(expected_by_uri) != len(expected_package_rows):
        raise ValueError("expected package manifest contains duplicate URIs")
    if set(package_by_uri) - set(expected_by_uri):
        raise ValueError("GCS package prefix contains an unknown object")
    for uri, observed in package_by_uri.items():
        expected = expected_by_uri[uri]
        if (
            observed["sha256"] != expected["sha256"]
            or observed["bytes"] != expected["bytes"]
        ):
            raise ValueError("GCS package prefix contains a mismatched object")
    # Exact sparse subsets are restart-safe.  Normalize every accepted subset
    # to immutable outer-manifest order rather than GCS lexicographic order.
    package_receiver_rows = [
        package_by_uri[row["uri"]]
        for row in expected_package_rows
        if row["uri"] in package_by_uri
    ]

    sku_items, transcript = _paginated_query(
        transport=transport,
        access_token=token,
        endpoint_id="billing_skus",
        target=target,
        collection_field="skus",
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
        valid_until_unix_seconds=price_valid_until,
    )
    transcripts.append(transcript)
    price_facts = _price_facts(
        sku_items,
        target=target,
        machine=machine_facts,
        retrieved_at_unix_seconds=retrieved_at_unix_seconds,
    )

    source_identity = f"sha256:{canonical_sha256(transcripts)}"
    # Caller-provided booleans are descriptive only.  Only the exact executable
    # stdlib transport type may turn network observations into gate evidence.
    query_performed = concrete_stdlib
    prefix_observation = receiver.build_prefix_observation(
        plan,
        package_objects=package_receiver_rows,
        stage_object_uris=[row["uri"] for row in stage_inventory],
        observation_source=f"google-json-api-readonly:{backend_id}",
        source_identity=source_identity,
        observed_at_unix_seconds=retrieved_at_unix_seconds,
        query_performed=query_performed,
        fixture_only=fixture_only,
    )
    instance_observation = receiver.build_instance_observation(
        plan,
        observed_instances=[],
        observation_source=f"google-compute-instance-get404:{backend_id}",
        source_identity=source_identity,
        observed_at_unix_seconds=retrieved_at_unix_seconds,
        query_performed=query_performed,
        fixture_only=fixture_only,
    )
    capacity_observation = receiver.build_capacity_observation(
        plan,
        available_vcpu=quota_facts["available_vcpu"],
        provider_metric_mapping=quota_facts["provider_metric_mapping"],
        observation_source=f"google-compute-region-readonly:{backend_id}",
        source_identity=source_identity,
        observed_at_unix_seconds=retrieved_at_unix_seconds,
        query_performed=query_performed,
        fixture_only=fixture_only,
    )
    price_observation = receiver.build_spot_price_observation(
        plan,
        observed_price_usd_per_vm_hour=price_facts[
            "hourly_price_usd_per_vm"
        ],
        observation_source=f"cloud-billing-catalog-readonly:{backend_id}",
        source_identity=source_identity,
        official_source_url=price_facts["official_source_url"],
        observed_at_unix_seconds=retrieved_at_unix_seconds,
        sku_effective_at_unix_seconds=price_facts[
            "sku_effective_at_unix_seconds"
        ],
        query_performed=query_performed,
        fixture_only=fixture_only,
    )
    facts = {
        "project": project_facts,
        "bucket_and_iam": bucket_facts,
        "service_account": service_account_facts,
        "image": image_facts,
        "machine": machine_facts,
        "regional_capacity": quota_facts,
        "expected_instance_absence": instance_facts,
        "package_inventory": package_inventory,
        "stage_inventory": stage_inventory,
        "spot_price": price_facts,
    }
    receiver_observations = {
        "prefix": prefix_observation,
        "instances": instance_observation,
        "capacity": capacity_observation,
        "spot_price": price_observation,
    }
    receiver_contract_result = receiver.evaluate_read_only_preflight(
        plan,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
        prefix_observation=prefix_observation,
        instance_observation=instance_observation,
        capacity_observation=capacity_observation,
        price_observation=price_observation,
        evaluation_unix_seconds=retrieved_at_unix_seconds,
    )
    read_only_observation_passed = (
        concrete_stdlib
        and receiver_contract_result["observation_contract_passed"] is True
        and all(
            transcript["page_tokens_exhausted"] is True
            for transcript in transcripts
        )
    )
    launch_permission_readiness = {
        "compute_instances_delete_permission_verified": False,
        "worker_oauth_scope_sufficiency_verified": False,
        "launch_permission_ready": False,
        "unverified_requirements": [
            "compute.instances.delete",
            receiver.REQUIRED_WORKER_OAUTH_SCOPE,
        ],
        "reason": (
            "GET-only preflight cannot prove a future VM service-account "
            "permission or OAuth access scope"
        ),
    }
    credential_provenance = _credential_provenance(token_source)
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "status": (
            "read_only_observation_passed_launch_still_unauthorized"
            if read_only_observation_passed
            else "read_only_observation_contract_collected_non_authoritative"
        ),
        "backend_schema": BACKEND_SCHEMA,
        "transport_backend_id": backend_id,
        "fixture_only": fixture_only,
        # This field is authoritative, not a copy of caller-declared
        # provenance attributes.
        "external_cloud_read_performed": concrete_stdlib,
        "concrete_stdlib_transport_used": concrete_stdlib,
        "read_only_observation_passed": read_only_observation_passed,
        "project": target.project,
        "bucket": target.bucket,
        "region": target.region,
        "zone": target.zone,
        "machine_type": target.machine_type,
        "package_manifest_sha256": plan["package_manifest_sha256"],
        "stage_identity_sha256": plan["stage_identity_sha256"],
        "plan_sha256": receiver.canonical_sha256(plan),
        "outer_package_identity_sha256": plan[
            "outer_package_identity_sha256"
        ],
        "direct_stage_identity_sha256": plan[
            "direct_stage_identity_sha256"
        ],
        "transport_contract_schema": transport_contract.CONTRACT_SCHEMA,
        "transport_outer_package_identity_sha256": outer_package_manifest[
            "outer_package_identity_sha256"
        ],
        "transport_stage_identity_sha256": direct_stage_identity[
            "direct_stage_identity_sha256"
        ],
        "retrieved_at_unix_seconds": retrieved_at_unix_seconds,
        "valid_until_unix_seconds": min(
            prefix_valid_until, capacity_valid_until, price_valid_until
        ),
        "http_methods_used": [COLLECTOR_METHOD],
        "allowed_http_methods": sorted(READ_ONLY_METHODS),
        "page_tokens_exhausted": all(
            transcript["page_tokens_exhausted"] for transcript in transcripts
        ),
        "transcripts": transcripts,
        "transcripts_sha256": canonical_sha256(transcripts),
        "facts": facts,
        "facts_sha256": canonical_sha256(facts),
        "receiver_observations": receiver_observations,
        "receiver_observations_sha256": receiver.canonical_sha256(
            receiver_observations
        ),
        "receiver_contract_result": receiver_contract_result,
        "receiver_contract_result_sha256": receiver.canonical_sha256(
            receiver_contract_result
        ),
        "launch_permission_readiness": launch_permission_readiness,
        "launch_permission_readiness_sha256": canonical_sha256(
            launch_permission_readiness
        ),
        "credential_provenance": credential_provenance,
        "credential_provenance_sha256": canonical_sha256(
            credential_provenance
        ),
        "access_token_recorded": False,
        "authorization_header_recorded": False,
        "request_body_present": False,
        "collector_subprocess_used": False,
        "collector_gcloud_used": False,
        "cloud_mutation_performed": False,
        "claim_created": False,
        "authorization_created": False,
        "vm_created": False,
        "launch_authorized": False,
        "launch_ready": False,
        "diagnostic_only": True,
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


def validate_observation_bundle(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    preview: Mapping[str, Any],
    outer_package_manifest: Mapping[str, Any],
    direct_stage_identity: Mapping[str, Any],
    evaluation_unix_seconds: int,
) -> dict[str, Any]:
    """Validate hashes, freshness, provenance, and receiver compatibility."""

    bundle = dict(value)
    _strict_int(evaluation_unix_seconds, "evaluation time", minimum=0)
    expected_bundle_sha = bundle.pop("bundle_sha256", None)
    _sha(expected_bundle_sha, "bundle sha256")
    if canonical_sha256(bundle) != expected_bundle_sha:
        raise ValueError("read-only observation bundle hash changed")
    bundle["bundle_sha256"] = expected_bundle_sha
    _exact(
        bundle,
        {
            "schema",
            "status",
            "backend_schema",
            "transport_backend_id",
            "fixture_only",
            "external_cloud_read_performed",
            "concrete_stdlib_transport_used",
            "read_only_observation_passed",
            "project",
            "bucket",
            "region",
            "zone",
            "machine_type",
            "package_manifest_sha256",
            "stage_identity_sha256",
            "plan_sha256",
            "outer_package_identity_sha256",
            "direct_stage_identity_sha256",
            "transport_contract_schema",
            "transport_outer_package_identity_sha256",
            "transport_stage_identity_sha256",
            "retrieved_at_unix_seconds",
            "valid_until_unix_seconds",
            "http_methods_used",
            "allowed_http_methods",
            "page_tokens_exhausted",
            "transcripts",
            "transcripts_sha256",
            "facts",
            "facts_sha256",
            "receiver_observations",
            "receiver_observations_sha256",
            "receiver_contract_result",
            "receiver_contract_result_sha256",
            "launch_permission_readiness",
            "launch_permission_readiness_sha256",
            "credential_provenance",
            "credential_provenance_sha256",
            "access_token_recorded",
            "authorization_header_recorded",
            "request_body_present",
            "collector_subprocess_used",
            "collector_gcloud_used",
            "cloud_mutation_performed",
            "claim_created",
            "authorization_created",
            "vm_created",
            "launch_authorized",
            "launch_ready",
            "diagnostic_only",
            "bundle_sha256",
        },
        "read-only observation bundle",
    )
    target = _target_from_plan(
        plan,
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    if (
        bundle["schema"] != BUNDLE_SCHEMA
        or bundle["backend_schema"] != BACKEND_SCHEMA
        or bundle["project"] != target.project
        or bundle["bucket"] != target.bucket
        or bundle["region"] != target.region
        or bundle["zone"] != target.zone
        or bundle["machine_type"] != target.machine_type
        or bundle["package_manifest_sha256"] != plan["package_manifest_sha256"]
        or bundle["stage_identity_sha256"] != plan["stage_identity_sha256"]
        or bundle["plan_sha256"] != receiver.canonical_sha256(plan)
        or bundle["outer_package_identity_sha256"]
        != plan["outer_package_identity_sha256"]
        or bundle["direct_stage_identity_sha256"]
        != plan["direct_stage_identity_sha256"]
        or bundle["transport_contract_schema"]
        != transport_contract.CONTRACT_SCHEMA
        or bundle["transport_outer_package_identity_sha256"]
        != outer_package_manifest["outer_package_identity_sha256"]
        or bundle["transport_stage_identity_sha256"]
        != direct_stage_identity["direct_stage_identity_sha256"]
        or bundle["http_methods_used"] != [COLLECTOR_METHOD]
        or bundle["allowed_http_methods"] != sorted(READ_ONLY_METHODS)
        or bundle["transcripts_sha256"]
        != canonical_sha256(bundle["transcripts"])
        or bundle["facts_sha256"] != canonical_sha256(bundle["facts"])
        or bundle["receiver_observations_sha256"]
        != receiver.canonical_sha256(bundle["receiver_observations"])
        or bundle["receiver_contract_result_sha256"]
        != receiver.canonical_sha256(bundle["receiver_contract_result"])
        or bundle["launch_permission_readiness_sha256"]
        != canonical_sha256(bundle["launch_permission_readiness"])
        or bundle["credential_provenance_sha256"]
        != canonical_sha256(bundle["credential_provenance"])
    ):
        raise ValueError("read-only observation bundle identity changed")
    for field in (
        "access_token_recorded",
        "authorization_header_recorded",
        "request_body_present",
        "collector_subprocess_used",
        "collector_gcloud_used",
        "cloud_mutation_performed",
        "claim_created",
        "authorization_created",
        "vm_created",
        "launch_authorized",
        "launch_ready",
    ):
        _strict_bool(bundle[field], False, field)
    _strict_bool(bundle["diagnostic_only"], True, "diagnostic_only")
    _strict_bool(
        bundle["concrete_stdlib_transport_used"],
        None,
        "concrete stdlib transport",
    )
    _strict_bool(
        bundle["read_only_observation_passed"],
        None,
        "read-only observation passed",
    )
    _strict_bool(bundle["fixture_only"], None, "fixture_only")
    _strict_bool(
        bundle["external_cloud_read_performed"],
        None,
        "external_cloud_read_performed",
    )
    if bundle["fixture_only"] and bundle["external_cloud_read_performed"]:
        raise ValueError("fixture bundle claimed an external cloud read")
    if bundle["concrete_stdlib_transport_used"] and (
        bundle["transport_backend_id"]
        != StdlibGoogleJsonReadOnlyTransport.backend_id
        or bundle["fixture_only"] is not False
        or bundle["external_cloud_read_performed"] is not True
    ):
        raise ValueError("concrete stdlib bundle provenance changed")
    transcripts = bundle["transcripts"]
    if not isinstance(transcripts, list) or not transcripts:
        raise ValueError("read-only transcripts are missing")
    for transcript in transcripts:
        if not isinstance(transcript, Mapping):
            raise ValueError("read-only transcript is not an object")
        transcript_copy = dict(transcript)
        transcript_sha = transcript_copy.pop("transcript_sha256", None)
        _sha(transcript_sha, "transcript sha256")
        if (
            transcript.get("schema") != TRANSCRIPT_SCHEMA
            or canonical_sha256(transcript_copy) != transcript_sha
            or transcript.get("page_tokens_exhausted") is not True
        ):
            raise ValueError("read-only transcript hash or pagination changed")
        pages = transcript.get("pages")
        if (
            not isinstance(pages, list)
            or not pages
            or transcript.get("page_count") != len(pages)
            or pages[-1].get("page_tokens_exhausted") is not True
        ):
            raise ValueError("read-only transcript pages changed")
        for page in pages:
            page_copy = dict(page)
            receipt_sha = page_copy.pop("receipt_sha256", None)
            _sha(receipt_sha, "query receipt sha256")
            if (
                page.get("schema") != QUERY_RECEIPT_SCHEMA
                or page.get("method") != COLLECTOR_METHOD
                or canonical_sha256(page_copy) != receipt_sha
                or page.get("authorization_header_recorded") is not False
                or page.get("access_token_recorded") is not False
                or page.get("request_body_present") is not False
                or page.get("cloud_mutation_performed") is not False
            ):
                raise ValueError("read-only query receipt changed")
    if bundle["page_tokens_exhausted"] is not True:
        raise ValueError("read-only pagination is incomplete")
    retrieved = _strict_int(
        bundle["retrieved_at_unix_seconds"], "bundle retrieved_at", minimum=0
    )
    valid_until = _strict_int(
        bundle["valid_until_unix_seconds"],
        "bundle valid_until",
        minimum=retrieved,
    )
    if not retrieved <= evaluation_unix_seconds <= valid_until:
        raise ValueError("read-only observation bundle is stale or from the future")
    observations = bundle["receiver_observations"]
    stored_result = receiver.evaluate_read_only_preflight(
        plan,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
        prefix_observation=observations["prefix"],
        instance_observation=observations["instances"],
        capacity_observation=observations["capacity"],
        price_observation=observations["spot_price"],
        evaluation_unix_seconds=retrieved,
    )
    if bundle["receiver_contract_result"] != stored_result:
        raise ValueError("stored receiver contract result changed")
    result = receiver.evaluate_read_only_preflight(
        plan,
        preview=preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
        prefix_observation=observations["prefix"],
        instance_observation=observations["instances"],
        capacity_observation=observations["capacity"],
        price_observation=observations["spot_price"],
        evaluation_unix_seconds=evaluation_unix_seconds,
    )
    expected_read_only_pass = (
        bundle["concrete_stdlib_transport_used"]
        and stored_result["observation_contract_passed"] is True
        and bundle["page_tokens_exhausted"] is True
    )
    _strict_bool(
        bundle["read_only_observation_passed"],
        expected_read_only_pass,
        "read-only observation gate",
    )
    expected_status = (
        "read_only_observation_passed_launch_still_unauthorized"
        if expected_read_only_pass
        else "read_only_observation_contract_collected_non_authoritative"
    )
    if bundle["status"] != expected_status:
        raise ValueError("read-only observation status changed")
    launch_permissions = bundle["launch_permission_readiness"]
    _exact(
        launch_permissions,
        {
            "compute_instances_delete_permission_verified",
            "worker_oauth_scope_sufficiency_verified",
            "launch_permission_ready",
            "unverified_requirements",
            "reason",
        },
        "launch permission readiness",
    )
    for field in (
        "compute_instances_delete_permission_verified",
        "worker_oauth_scope_sufficiency_verified",
        "launch_permission_ready",
    ):
        _strict_bool(launch_permissions[field], False, field)
    if launch_permissions["unverified_requirements"] != [
        "compute.instances.delete",
        receiver.REQUIRED_WORKER_OAUTH_SCOPE,
    ]:
        raise ValueError("launch permission requirements changed")
    credential = bundle["credential_provenance"]
    _exact(
        credential,
        {
            "collector_token_source",
            "credential_supplied_to_collector_externally",
            "credential_acquisition_method",
            "credential_acquisition_subprocess_used",
            "credential_acquisition_gcloud_used",
            "collector_subprocess_used",
            "collector_gcloud_used",
        },
        "credential provenance",
    )
    _strict_bool(
        credential["credential_supplied_to_collector_externally"],
        True,
        "credential supplied externally",
    )
    for field in ("collector_subprocess_used", "collector_gcloud_used"):
        _strict_bool(credential[field], False, field)
    if (
        credential["credential_acquisition_method"] != "external_unknown"
        or credential["credential_acquisition_subprocess_used"] is not None
        or credential["credential_acquisition_gcloud_used"] is not None
    ):
        raise ValueError("credential acquisition provenance was guessed")
    if (
        result["launch_authorized"] is not False
        or result["launch_ready"] is not False
        or result["cloud_mutation_performed"] is not False
    ):
        raise ValueError("receiver preflight escaped the read-only boundary")
    return _json_copy(bundle)


def _load_json_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    source = os.fspath(path)
    with open(source, "rb") as handle:
        raw = handle.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError(f"{label} JSON file is too large")
    try:
        return dict(_strict_json(raw))
    except ValueError as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _exclusive_write_json(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> None:
    target = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(target)
    if not parent or not os.path.isdir(parent):
        raise ValueError("read-only preflight output parent must already exist")
    raw = canonical_bytes(value) + b"\n"
    with open(target, "xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _failure_artifact(
    *,
    stage_id: str,
    failure_code: str,
    evidence: Mapping[str, Any] | None,
    retrieved_at_unix_seconds: int,
    credential_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    failure = {
        "schema": ACTUAL_RUN_SCHEMA,
        "status": "no_go_read_only_preflight",
        "stage_id": stage_id,
        "retrieved_at_unix_seconds": retrieved_at_unix_seconds,
        "failure_code": failure_code,
        "failure_evidence": dict(evidence) if evidence is not None else None,
        "read_only_observation_passed": False,
        "launch_permission_ready": False,
        "credential_provenance": dict(credential_provenance),
        "access_token_recorded": False,
        "authorization_header_recorded": False,
        "collector_subprocess_used": False,
        "collector_gcloud_used": False,
        "cloud_mutation_performed": False,
        "claim_created": False,
        "authorization_created": False,
        "vm_created": False,
        "launch_authorized": False,
        "launch_ready": False,
        "diagnostic_only": True,
    }
    failure["artifact_sha256"] = canonical_sha256(failure)
    return failure


def run_actual_read_only_preflight(
    *,
    package_dir: str | os.PathLike[str],
    pinned_wheel_path: str | os.PathLike[str],
    stage_id: str,
    output_path: str | os.PathLike[str],
    token_source: AccessTokenSource | None = None,
    transport: ReadOnlyHttpsTransport | None = None,
    retrieved_at_unix_seconds: int | None = None,
    prerequisite_stage1_preview: Mapping[str, Any] | None = None,
    prerequisite_stage1_receive: Mapping[str, Any] | None = None,
    expected_worker_storage_bindings: Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Execute the actual read-only path and exclusively create one receipt.

    Supplying ``StdlibGoogleJsonReadOnlyTransport`` and
    ``EnvironmentAccessTokenSource`` is the real path.  Injected fakes use the
    same construction in tests but remain non-authoritative through their
    provenance flags.
    """

    observed_at = (
        int(time.time())
        if retrieved_at_unix_seconds is None
        else _strict_int(
            retrieved_at_unix_seconds,
            "retrieved_at_unix_seconds",
            minimum=0,
        )
    )
    active_transport = transport or StdlibGoogleJsonReadOnlyTransport()
    active_token_source = token_source or EnvironmentAccessTokenSource()
    credential_provenance = _credential_provenance(active_token_source)
    try:
        if type(active_transport) is not StdlibGoogleJsonReadOnlyTransport:
            raise ValueError(
                "actual path requires concrete stdlib GET-only transport"
            )
        preview = adapter.build_preview(
            package_dir=package_dir,
            stage_id=stage_id,
            prerequisite_stage1_preview=prerequisite_stage1_preview,
            prerequisite_stage1_receive=prerequisite_stage1_receive,
        )
        wheel_record = transport_contract.build_offline_wheel_record(
            pinned_wheel_path
        )
        outer = receiver.build_outer_package_manifest(
            package_dir=package_dir,
            offline_wheel_record=wheel_record,
        )
        direct = receiver.build_direct_stage_identity(
            preview,
            outer_package_manifest=outer,
        )
        preflight_plan = receiver.build_read_only_preflight_plan(
            preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
        )
        bundle = collect_read_only_observations(
            plan=preflight_plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            transport=active_transport,
            token_source=active_token_source,
            retrieved_at_unix_seconds=observed_at,
            expected_worker_storage_bindings=(
                expected_worker_storage_bindings
            ),
        )
        validate_observation_bundle(
            bundle,
            plan=preflight_plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            evaluation_unix_seconds=observed_at,
        )
        observations = bundle["receiver_observations"]
        result = receiver.evaluate_read_only_preflight(
            preflight_plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            prefix_observation=observations["prefix"],
            instance_observation=observations["instances"],
            capacity_observation=observations["capacity"],
            price_observation=observations["spot_price"],
            evaluation_unix_seconds=observed_at,
        )
        artifact = {
            "schema": ACTUAL_RUN_SCHEMA,
            "status": (
                "pass_read_only_preflight_launch_still_unauthorized"
                if bundle["read_only_observation_passed"]
                else "no_go_read_only_preflight"
            ),
            "stage_id": stage_id,
            "retrieved_at_unix_seconds": observed_at,
            "preview_identity": {
                "package_manifest_sha256": preview[
                    "package_manifest_sha256"
                ],
                "stage_identity_sha256": preview["stage_identity_sha256"],
                "stage_id": preview["stage_id"],
                "run_name": preview["run_name"],
            },
            "outer_package_manifest": outer,
            "direct_stage_identity": direct,
            "read_only_preflight_plan": preflight_plan,
            "observation_bundle": bundle,
            "receiver_preflight_result": result,
            "read_only_observation_passed": bundle[
                "read_only_observation_passed"
            ],
            "launch_permission_ready": False,
            "credential_provenance": credential_provenance,
            "access_token_recorded": False,
            "authorization_header_recorded": False,
            "collector_subprocess_used": False,
            "collector_gcloud_used": False,
            "cloud_mutation_performed": False,
            "claim_created": False,
            "authorization_created": False,
            "vm_created": False,
            "launch_authorized": False,
            "launch_ready": False,
            "diagnostic_only": True,
        }
        artifact["artifact_sha256"] = canonical_sha256(artifact)
    except ReadOnlyPreflightError as exc:
        artifact = _failure_artifact(
            stage_id=stage_id,
            failure_code=exc.code,
            evidence=exc.evidence,
            retrieved_at_unix_seconds=observed_at,
            credential_provenance=credential_provenance,
        )
    except (ValueError, OSError):
        # Do not serialize exception text; injected transports and credential
        # sources are untrusted and may place secrets in exception messages.
        artifact = _failure_artifact(
            stage_id=stage_id,
            failure_code="local_or_normalization_preflight_failure",
            evidence=None,
            retrieved_at_unix_seconds=observed_at,
            credential_provenance=credential_provenance,
        )
    _exclusive_write_json(output_path, artifact)
    return _json_copy(artifact)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 10c.2 Google JSON API read-only preflight. "
            "This command cannot authorize or launch a VM."
        )
    )
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--pinned-wheel", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--token-env",
        default="GOOGLE_OAUTH_ACCESS_TOKEN",
        help=(
            "Environment variable containing a token obtained outside this "
            "process; its value is never written."
        ),
    )
    parser.add_argument("--retrieved-at-unix-seconds", type=int)
    parser.add_argument("--prerequisite-stage1-preview")
    parser.add_argument("--prerequisite-stage1-receive")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    prerequisite_preview = (
        _load_json_object(
            args.prerequisite_stage1_preview, "prerequisite stage1 preview"
        )
        if args.prerequisite_stage1_preview
        else None
    )
    prerequisite_receive = (
        _load_json_object(
            args.prerequisite_stage1_receive, "prerequisite stage1 receive"
        )
        if args.prerequisite_stage1_receive
        else None
    )
    artifact = run_actual_read_only_preflight(
        package_dir=args.package_dir,
        pinned_wheel_path=args.pinned_wheel,
        stage_id=args.stage_id,
        output_path=args.output,
        token_source=EnvironmentAccessTokenSource(args.token_env),
        transport=StdlibGoogleJsonReadOnlyTransport(),
        retrieved_at_unix_seconds=args.retrieved_at_unix_seconds,
        prerequisite_stage1_preview=prerequisite_preview,
        prerequisite_stage1_receive=prerequisite_receive,
    )
    summary = {
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
        "output": os.path.abspath(args.output),
        "launch_authorized": False,
        "launch_ready": False,
    }
    sys.stdout.buffer.write(canonical_bytes(summary) + b"\n")
    return (
        0
        if artifact["status"]
        == "pass_read_only_preflight_launch_still_unauthorized"
        else 2
    )


__all__ = [
    "AccessTokenSource",
    "ACTUAL_RUN_SCHEMA",
    "BACKEND_SCHEMA",
    "BUNDLE_SCHEMA",
    "COLLECTOR_METHOD",
    "HttpResponse",
    "EnvironmentAccessTokenSource",
    "QUERY_RECEIPT_SCHEMA",
    "READ_ONLY_METHODS",
    "ReadOnlyHttpsTransport",
    "ReadOnlyPreflightError",
    "StdlibGoogleJsonReadOnlyTransport",
    "TRANSCRIPT_SCHEMA",
    "canonical_bytes",
    "canonical_sha256",
    "collect_read_only_observations",
    "main",
    "run_actual_read_only_preflight",
    "validate_allowed_request",
    "validate_observation_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
