"""GET-only live GCP collector for the full100 runtime preflight.

The pure runtime preflight normalizer intentionally accepts provider JSON from
its caller.  This adapter is the production provenance boundary: it performs
exactly six fixed GETs, retains each bounded raw response in a sealed receipt,
and passes those responses directly to the producer-owned runtime normalizer.
It has no cloud mutation surface.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.parse
from copy import deepcopy
from typing import Any, Callable, Mapping

from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_runtime_preflight_v2 as runtime_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
    _stdlib_http_request,
)


SCHEMA = "hu_m31_t3_step6d_full100_wave_runtime_gcp_read_receipt_v2"
STATUS = "exact_six_endpoint_runtime_gcp_read_complete"
PROVIDER_SOURCE = "gcp_compute_and_storage_json_get_only_v2"
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
HTTP_TIMEOUT_SECONDS = 60
MAX_RESPONSE_BYTES = 512 * 1024
EXACT_ENDPOINT_COUNT = 6

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


def _fields_url(base: str, fields: str) -> str:
    return f"{base}?{urllib.parse.urlencode({'fields': fields})}"


IMAGE_FAMILY_URL = _fields_url(
    "https://compute.googleapis.com/compute/v1/projects/"
    f"{runtime_v2.IMAGE_PROJECT}/global/images/family/debian-12",
    "architecture,deprecated,family,guestOsFeatures,id,name,selfLink,status,storageLocations",
)
MACHINE_TYPE_URL = _fields_url(
    "https://compute.googleapis.com/compute/v1/projects/"
    f"{runtime_v2.PROJECT}/zones/{runtime_v2.ZONE}/machineTypes/"
    f"{runtime_v2.MACHINE_TYPE}",
    "architecture,deprecated,guestCpus,id,memoryMb,name,selfLink,zone",
)
NETWORK_URL = _fields_url(
    "https://compute.googleapis.com/compute/v1/projects/"
    f"{runtime_v2.PROJECT}/global/networks/{runtime_v2.NETWORK_NAME}",
    "id,name,selfLink",
)
SUBNETWORK_URL = _fields_url(
    "https://compute.googleapis.com/compute/v1/projects/"
    f"{runtime_v2.PROJECT}/regions/{runtime_v2.REGION}/subnetworks/"
    f"{runtime_v2.SUBNETWORK_NAME}",
    "id,name,network,region,selfLink",
)
ROUTER_NAT_URL = _fields_url(
    "https://compute.googleapis.com/compute/v1/projects/"
    f"{runtime_v2.PROJECT}/regions/{runtime_v2.REGION}/routers/"
    f"{runtime_v2.NAT_ROUTER_NAME}",
    "id,name,nats(name,natIpAllocateOption,natIps,sourceSubnetworkIpRangesToNat),"
    "network,region,selfLink",
)
BUCKET_URL = _fields_url(
    "https://storage.googleapis.com/storage/v1/b/"
    f"{urllib.parse.quote(runtime_v2.BUCKET, safe='')}",
    "iamConfiguration(uniformBucketLevelAccess(enabled)),id,location,locationType,"
    "name,projectNumber",
)

ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("image_family", IMAGE_FAMILY_URL),
    ("machine_type", MACHINE_TYPE_URL),
    ("network", NETWORK_URL),
    ("subnetwork", SUBNETWORK_URL),
    ("router_nat", ROUTER_NAT_URL),
    ("bucket", BUCKET_URL),
)

_ROW_KEYS = frozenset(
    {
        "resource",
        "http_method",
        "url",
        "http_status",
        "response_bytes",
        "response_sha256",
        "response_body_base64",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "run_name",
        "execution_identity_sha256",
        "project",
        "region",
        "zone",
        "observed_at_utc",
        "issued_at_utc",
        "request_rows",
        "http_get_count",
        "exact_endpoint_count",
        "provider_source",
        "runtime_preflight_receipt",
        "runtime_preflight_receipt_sha256",
        "environment_token_only",
        "strict_json_body_bound",
        "read_only",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)


class RuntimeGcpReadTransportError(RuntimeError):
    """A GET failed without a provider HTTP response."""


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


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_RESPONSE_BYTES:
        raise RuntimeError(f"{label} response escaped the fixed body bound")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} response was not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} response was not a JSON object")
    if parsed.get("nextPageToken") is not None:
        raise RuntimeError(f"{label} response unexpectedly paginated")
    return parsed


def _headers() -> dict[str, str]:
    token = os.environ.get(TOKEN_ENV)
    if (
        not isinstance(token, str)
        or len(token) < 20
        or any(character.isspace() for character in token)
    ):
        raise PermissionError(
            f"Bearer token must be supplied only through {TOKEN_ENV}"
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _row_from_response(
    *, resource: str, url: str, response: HttpResponse
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(response, HttpResponse):
        raise RuntimeError("runtime GCP requester returned an invalid response")
    if response.status != 200:
        raise RuntimeError(f"runtime GCP GET failed with status {response.status}")
    observation = _strict_json_bytes(response.body, f"runtime {resource}")
    row = {
        "resource": resource,
        "http_method": "GET",
        "url": url,
        "http_status": 200,
        "response_bytes": len(response.body),
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
        "response_body_base64": base64.b64encode(response.body).decode("ascii"),
    }
    return row, observation


def _observations_from_rows(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXACT_ENDPOINT_COUNT:
        raise ValueError("runtime GCP receipt must contain exactly six GET rows")
    observations: dict[str, dict[str, Any]] = {}
    for raw_row, (resource, url) in zip(rows, ENDPOINTS, strict=True):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _ROW_KEYS:
            raise ValueError("runtime GCP request row fields changed")
        row = deepcopy(dict(raw_row))
        encoded = row.get("response_body_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("runtime GCP response body evidence changed")
        try:
            body = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("runtime GCP response body evidence changed") from exc
        if base64.b64encode(body).decode("ascii") != encoded:
            raise ValueError("runtime GCP response body encoding is not canonical")
        if (
            row.get("resource") != resource
            or row.get("http_method") != "GET"
            or row.get("url") != url
            or row.get("http_status") != 200
            or row.get("response_bytes") != len(body)
            or row.get("response_sha256") != hashlib.sha256(body).hexdigest()
        ):
            raise ValueError("runtime GCP exact endpoint/body evidence changed")
        observations[resource] = _strict_json_bytes(body, f"runtime {resource}")
    return observations


def validate_runtime_gcp_read_receipt(
    *,
    wave_plan: Mapping[str, Any],
    value: Mapping[str, Any],
    current_utc: str,
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_KEYS:
        raise ValueError("runtime GCP read receipt fields changed")
    receipt = deepcopy(dict(value))
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("runtime GCP read receipt digest changed")
    observations = _observations_from_rows(receipt.get("request_rows"))
    expected_runtime = runtime_v2.build_runtime_preflight_receipt(
        wave_plan=plan,
        image_observation=observations["image_family"],
        machine_type_observation=observations["machine_type"],
        network_observation=observations["network"],
        subnetwork_observation=observations["subnetwork"],
        cloud_nat_observation=observations["router_nat"],
        bucket_observation=observations["bucket"],
        observed_at_utc=receipt.get("observed_at_utc"),
        current_utc=receipt.get("issued_at_utc"),
    )
    runtime = runtime_v2.validate_runtime_preflight_receipt(
        wave_plan=plan,
        value=receipt.get("runtime_preflight_receipt"),
        current_utc=current_utc,
    )
    if expected_runtime != runtime:
        raise ValueError("runtime preflight does not derive from live GCP responses")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("project") != runtime_v2.PROJECT
        or receipt.get("region") != runtime_v2.REGION
        or receipt.get("zone") != runtime_v2.ZONE
        or receipt.get("observed_at_utc") != runtime["observed_at_utc"]
        or receipt.get("issued_at_utc") != runtime["issued_at_utc"]
        or receipt.get("http_get_count") != EXACT_ENDPOINT_COUNT
        or receipt.get("exact_endpoint_count") != EXACT_ENDPOINT_COUNT
        or receipt.get("provider_source") != PROVIDER_SOURCE
        or receipt.get("runtime_preflight_receipt_sha256")
        != runtime["receipt_sha256"]
        or receipt.get("environment_token_only") is not True
        or receipt.get("strict_json_body_bound") is not True
        or receipt.get("read_only") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("runtime GCP read receipt evidence changed or expired")
    return {**receipt, "receipt_sha256": digest}


class RuntimeGcpReadAdapterV2:
    """One-shot adapter exposing only the fixed six runtime GETs."""

    def __init__(
        self,
        *,
        wave_plan: Mapping[str, Any],
        requester: HttpRequester = _stdlib_http_request,
    ) -> None:
        if not callable(requester):
            raise TypeError("runtime GCP requester must be callable")
        self._plan = wave_v2.validate_wave_plan(wave_plan)
        self._requester = requester
        self._used = False
        self._requested_urls: set[str] = set()

    def _request(
        self, *, method: str, url: str, body: bytes | None = None
    ) -> HttpResponse:
        exact_urls = {endpoint for _, endpoint in ENDPOINTS}
        if method != "GET":
            raise PermissionError("runtime GCP adapter is GET-only")
        if body is not None:
            raise PermissionError("runtime GCP GET cannot carry a request body")
        if url not in exact_urls or url in self._requested_urls:
            raise PermissionError("runtime GCP request escaped the exact endpoint set")
        headers = _headers()  # Local credential validation is not transport loss.
        self._requested_urls.add(url)
        try:
            response = self._requester(
                method, url, headers, None, HTTP_TIMEOUT_SECONDS
            )
        except Exception as exc:
            raise RuntimeGcpReadTransportError(
                "runtime GCP GET transport failed without provider status"
            ) from exc
        if not isinstance(response, HttpResponse):
            raise RuntimeError("runtime GCP requester returned an invalid response")
        if response.status != 200:
            raise RuntimeError(f"runtime GCP GET failed with status {response.status}")
        if not isinstance(response.body, bytes) or not 1 <= len(response.body) <= MAX_RESPONSE_BYTES:
            raise RuntimeError("runtime GCP response escaped the fixed body bound")
        return response

    def read(
        self, *, observed_at_utc: str, current_utc: str
    ) -> dict[str, Any]:
        if self._used:
            raise PermissionError("runtime GCP adapter is one-shot")
        self._used = True
        rows: list[dict[str, Any]] = []
        observations: dict[str, dict[str, Any]] = {}
        for resource, url in ENDPOINTS:
            row, observation = _row_from_response(
                resource=resource,
                url=url,
                response=self._request(method="GET", url=url),
            )
            rows.append(row)
            observations[resource] = observation
        runtime = runtime_v2.build_runtime_preflight_receipt(
            wave_plan=self._plan,
            image_observation=observations["image_family"],
            machine_type_observation=observations["machine_type"],
            network_observation=observations["network"],
            subnetwork_observation=observations["subnetwork"],
            cloud_nat_observation=observations["router_nat"],
            bucket_observation=observations["bucket"],
            observed_at_utc=observed_at_utc,
            current_utc=current_utc,
        )
        core = {
            "schema": SCHEMA,
            "status": STATUS,
            "wave_plan_sha256": self._plan["schedule_sha256"],
            "run_name": self._plan["run_name"],
            "execution_identity_sha256": self._plan["execution_identity_sha256"],
            "project": runtime_v2.PROJECT,
            "region": runtime_v2.REGION,
            "zone": runtime_v2.ZONE,
            "observed_at_utc": observed_at_utc,
            "issued_at_utc": current_utc,
            "request_rows": rows,
            "http_get_count": len(rows),
            "exact_endpoint_count": EXACT_ENDPOINT_COUNT,
            "provider_source": PROVIDER_SOURCE,
            "runtime_preflight_receipt": runtime,
            "runtime_preflight_receipt_sha256": runtime["receipt_sha256"],
            "environment_token_only": True,
            "strict_json_body_bound": True,
            "read_only": True,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        return validate_runtime_gcp_read_receipt(
            wave_plan=self._plan,
            value={**core, "receipt_sha256": canonical_sha256(core)},
            current_utc=current_utc,
        )


__all__ = [
    "BUCKET_URL",
    "ENDPOINTS",
    "EXACT_ENDPOINT_COUNT",
    "HTTP_TIMEOUT_SECONDS",
    "IMAGE_FAMILY_URL",
    "MACHINE_TYPE_URL",
    "MAX_RESPONSE_BYTES",
    "NETWORK_URL",
    "PROVIDER_SOURCE",
    "ROUTER_NAT_URL",
    "RuntimeGcpReadAdapterV2",
    "RuntimeGcpReadTransportError",
    "SCHEMA",
    "STATUS",
    "SUBNETWORK_URL",
    "TOKEN_ENV",
    "canonical_bytes",
    "canonical_sha256",
    "validate_runtime_gcp_read_receipt",
]
