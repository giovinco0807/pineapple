"""GET-only live observations for T3 performance-development v2.

The collector is deliberately narrower than a launcher.  It can only read the
fixed image, machine, quota, inventory, billing, and collision surfaces needed
by the local v2 preflight.  It has no request-body surface and rejects every
HTTP method except GET.  Successful output is still a dry-run observation: it
cannot build a package, create an identity, or launch/delete cloud resources.

The executable CLI uses the already-tested stdlib HTTPS transport and an
environment-provisioned short-lived access token.  Tests inject a fixture
transport.  Access tokens, response bodies, and raw pagination tokens are
never retained in returned evidence or exception messages.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from . import hu_m31_t3_step6d_performance_development_v2_contract as contract_v1
from . import hu_m31_t3_step6d_performance_development_v2_preflight as preflight
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as tested_readonly,
)
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as cloud_defaults


COLLECTION_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_live_readonly_collection_v1"
)
QUERY_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_live_readonly_query_v1"
)

PROJECT = cloud_defaults.DEFAULT_PROJECT
BUCKET = cloud_defaults.DEFAULT_BUCKET
REGION = "asia-northeast1"
ZONE = "asia-northeast1-b"
MACHINE_TYPE = contract_v1.MACHINE_TYPE
IMAGE_PROJECT = "debian-cloud"
IMAGE_FAMILY = "debian-12"
COMPUTE_SERVICE_ID = "6F81-5844-456A"
C4_QUOTA_ID = "CPUS-PER-VM-FAMILY-per-project-region"
C4_QUOTA_METRIC = "compute.googleapis.com/cpus_per_vm_family"
SPOT_QUOTA_METRIC = "PREEMPTIBLE_CPUS"
C4_LEGACY_QUOTA_METRIC = "C4_CPUS"
_AGGREGATED_INVENTORY = {
    "aggregated_instances": (
        "instances",
        "compute#instanceAggregatedList",
    ),
    "aggregated_reservations": (
        "reservations",
        "compute#reservationAggregatedList",
    ),
    "aggregated_node_groups": (
        "nodeGroups",
        "compute#nodeGroupAggregatedList",
    ),
    "aggregated_future_reservations": (
        "futureReservations",
        "compute#futureReservationsAggregatedListResponse",
    ),
}

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
MAX_PAGES = 100
MAX_OBSERVATION_AGE_SECONDS = 300
MAX_FUTURE_SKEW_SECONDS = 5

_SAFE_COMPUTE_NAME = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")
_SAFE_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
_RUN_NAME = re.compile(
    r"regular-hu-m31-c02-perfdev-v2-[a-z0-9][a-z0-9-]{7,47}"
)
_IDENTITY = re.compile(r"perfdev-v2-[a-z0-9][a-z0-9-]{7,47}")
_IMAGE_NAME = re.compile(r"debian-12-bookworm-v[0-9]{8}")
_C4_MACHINE = re.compile(r"c4-(?:standard|highcpu|highmem)-([1-9][0-9]{0,3})")
_DIGITS = re.compile(r"[0-9]+")
_PAGE_TOKEN = re.compile(r"[\x21-\x7e]{1,8192}")
_ALLOWED_HOSTS = frozenset(
    {
        "compute.googleapis.com",
        "cloudquotas.googleapis.com",
        "cloudbilling.googleapis.com",
        "storage.googleapis.com",
    }
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
_FORBIDDEN_NAMESPACE_TERMS = (
    "performance-lock",
    "rearm2",
    "lock-r2",
    contract_v1.REARM2_RUN_ID,
    contract_v1.REARM2_PACKAGE_RUN_NAME,
)


HttpResponse = tested_readonly.HttpResponse
EnvironmentAccessTokenSource = tested_readonly.EnvironmentAccessTokenSource
StdlibGoogleJsonReadOnlyTransport = (
    tested_readonly.StdlibGoogleJsonReadOnlyTransport
)


class AccessTokenSource(Protocol):
    def access_token(self) -> str:
        """Return a short-lived OAuth token without logging it."""


class ReadOnlyHttpsTransport(Protocol):
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


@dataclass(frozen=True)
class _Target:
    project: str
    bucket: str
    run_name: str
    identity_namespace: str
    result_prefix: str
    identity_prefix: str
    instance_names: tuple[str, str]
    disk_names: tuple[str, str]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON response field")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON response value")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cloud response is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("cloud response is not a JSON object")
    return value


def _exact_or_allowed_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    label: str,
) -> None:
    fields = set(value)
    if not required <= fields or not fields <= allowed:
        raise ValueError(f"{label} response fields changed")


def _token(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 8192
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError("access-token source returned an invalid token")
    return value


def _integral(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an exact integer")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} is not an exact integer") from exc
    if (
        not parsed.is_finite()
        or parsed != parsed.to_integral_value()
        or parsed < (1 if positive else 0)
    ):
        raise ValueError(f"{label} is not an exact nonnegative integer")
    return int(parsed)


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError("decimal value is not finite and nonnegative")
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result or "0"


def _target(
    *,
    project: str,
    bucket: str,
    run_name: str,
    identity_namespace: str,
    result_prefix: str,
) -> _Target:
    expected_result_prefix = f"hu-m31-t3/perfdev-v2/{run_name}/"
    identity_prefix = (
        f"hu-m31-t3/perfdev-v2-identities/{identity_namespace}/"
    )
    instance_names = (f"{run_name}-c", f"{run_name}-r")
    if (
        project != PROJECT
        or bucket != BUCKET
        or _SAFE_BUCKET.fullmatch(bucket) is None
        or _RUN_NAME.fullmatch(run_name) is None
        or _IDENTITY.fullmatch(identity_namespace) is None
        or result_prefix != expected_result_prefix
        or any(term in run_name for term in _FORBIDDEN_NAMESPACE_TERMS)
        or any(term in identity_namespace for term in _FORBIDDEN_NAMESPACE_TERMS)
        or any(term in result_prefix for term in _FORBIDDEN_NAMESPACE_TERMS)
        or len(set(instance_names)) != 2
        or any(_SAFE_COMPUTE_NAME.fullmatch(name) is None for name in instance_names)
    ):
        raise ValueError("performance-development v2 live target changed")
    return _Target(
        project=project,
        bucket=bucket,
        run_name=run_name,
        identity_namespace=identity_namespace,
        result_prefix=result_prefix,
        identity_prefix=identity_prefix,
        instance_names=instance_names,
        disk_names=instance_names,
    )


def _base_url(endpoint_id: str, target: _Target) -> str:
    project = quote(target.project, safe="")
    zone = quote(ZONE, safe="")
    values = {
        "image_family": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{IMAGE_PROJECT}/global/images/family/{IMAGE_FAMILY}"
        ),
        "machine_type": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/zones/{zone}/machineTypes/{MACHINE_TYPE}"
        ),
        "cloud_quotas_c4": (
            "https://cloudquotas.googleapis.com/v1/projects/"
            f"{project}/locations/global/services/compute.googleapis.com/"
            f"quotaInfos/{C4_QUOTA_ID}"
        ),
        "compute_region": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/regions/{REGION}"
        ),
        "aggregated_instances": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/aggregated/instances"
        ),
        "aggregated_reservations": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/aggregated/reservations"
        ),
        "aggregated_node_groups": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/aggregated/nodeGroups"
        ),
        "aggregated_future_reservations": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/aggregated/futureReservations"
        ),
        "billing_skus": (
            "https://cloudbilling.googleapis.com/v1/services/"
            f"{COMPUTE_SERVICE_ID}/skus"
        ),
        "result_prefix": (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{quote(target.bucket, safe='')}/o"
        ),
        "identity_prefix": (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{quote(target.bucket, safe='')}/o"
        ),
    }
    for name in target.instance_names:
        values[f"instance:{name}"] = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/zones/{zone}/instances/{quote(name, safe='')}"
        )
    for name in target.disk_names:
        values[f"disk:{name}"] = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/zones/{zone}/disks/{quote(name, safe='')}"
        )
    try:
        return values[endpoint_id]
    except KeyError:
        raise ValueError("unknown read-only endpoint") from None


def _query(
    endpoint_id: str, target: _Target, page_token: str | None
) -> list[tuple[str, str]]:
    if endpoint_id in _AGGREGATED_INVENTORY:
        result = [
            ("maxResults", "500"),
            ("returnPartialSuccess", "false"),
            ("includeAllScopes", "true"),
        ]
    elif endpoint_id == "billing_skus":
        result = [("currencyCode", "USD"), ("pageSize", "5000")]
    elif endpoint_id in {"result_prefix", "identity_prefix"}:
        prefix = (
            target.result_prefix
            if endpoint_id == "result_prefix"
            else target.identity_prefix
        )
        result = [
            ("prefix", prefix),
            ("pageSize", "1000"),
            ("projection", "noAcl"),
        ]
    else:
        if page_token is not None:
            raise ValueError("non-paginated endpoint received a page token")
        return []
    if page_token is not None:
        if _PAGE_TOKEN.fullmatch(page_token) is None:
            raise ValueError("invalid pagination token")
        result.append(("pageToken", page_token))
    return result


def _url(endpoint_id: str, target: _Target, page_token: str | None = None) -> str:
    query = _query(endpoint_id, target, page_token)
    return _base_url(endpoint_id, target) + (
        f"?{urlencode(query)}" if query else ""
    )


def _validate_url(
    endpoint_id: str, target: _Target, url: str, page_token: str | None
) -> None:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in _ALLOWED_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.port not in (None, 443)
        or parts.fragment
        or url != _url(endpoint_id, target, page_token)
    ):
        raise ValueError("read-only request escaped exact GET allowlist")


def _sanitized_url(url: str) -> str:
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "pageToken":
            value = f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
        query.append((key, value))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), "")
    )


def _issue(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    page_index: int = 0,
    page_token: str | None = None,
    allowed_statuses: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any], str | None, dict[str, Any]]:
    url = _url(endpoint_id, target, page_token)
    _validate_url(endpoint_id, target, url, page_token)
    try:
        response = transport.request(
            method="GET",
            url=url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except (OSError, TimeoutError):
        raise RuntimeError(f"GET-only cloud read failed: {endpoint_id}") from None
    if (
        not isinstance(response, HttpResponse)
        or type(response.status_code) is not int
        or not isinstance(response.body, bytes)
        or len(response.body) > MAX_RESPONSE_BYTES
        or not isinstance(response.headers, Mapping)
    ):
        raise ValueError("read-only transport response shape changed")
    if response.status_code not in allowed_statuses:
        raise RuntimeError(
            f"GET-only cloud read returned unexpected status for {endpoint_id}"
        )
    payload = _strict_json(response.body)
    raw_next = payload.pop("nextPageToken", None)
    if raw_next == "":
        raw_next = None
    if raw_next is not None and (
        not isinstance(raw_next, str) or _PAGE_TOKEN.fullmatch(raw_next) is None
    ):
        raise ValueError("cloud response pagination token changed")
    receipt = {
        "schema": QUERY_RECEIPT_SCHEMA,
        "endpoint_id": endpoint_id,
        "method": "GET",
        "sanitized_url": _sanitized_url(url),
        "page_index": page_index,
        "request_page_token_sha256": (
            hashlib.sha256(page_token.encode("utf-8")).hexdigest()
            if page_token is not None
            else None
        ),
        "status_code": response.status_code,
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
        "response_bytes": len(response.body),
        "response_next_page_token_sha256": (
            hashlib.sha256(raw_next.encode("utf-8")).hexdigest()
            if raw_next is not None
            else None
        ),
        "authorization_header_recorded": False,
        "access_token_recorded": False,
        "request_body_present": False,
        "cloud_mutated": False,
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    return response.status_code, payload, raw_next, receipt


def _single(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    allowed_statuses: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    status, payload, next_token, receipt = _issue(
        transport=transport,
        access_token=access_token,
        endpoint_id=endpoint_id,
        target=target,
        allowed_statuses=allowed_statuses,
    )
    if next_token is not None:
        raise ValueError("single-resource response unexpectedly paginated")
    return status, payload, receipt


def _pages(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    collection_field: str,
    allowed_top_fields: frozenset[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    values: list[Any] = []
    receipts: list[dict[str, Any]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for page_index in range(MAX_PAGES):
        _, payload, next_token, receipt = _issue(
            transport=transport,
            access_token=access_token,
            endpoint_id=endpoint_id,
            target=target,
            page_index=page_index,
            page_token=page_token,
        )
        if not set(payload) <= allowed_top_fields:
            raise ValueError(f"{endpoint_id} response contains unknown fields")
        page_values = payload.get(collection_field, [])
        if not isinstance(page_values, list):
            raise ValueError(f"{endpoint_id} collection field changed")
        values.extend(page_values)
        receipts.append(receipt)
        if next_token is None:
            return values, receipts
        token_sha = hashlib.sha256(next_token.encode("utf-8")).hexdigest()
        if token_sha in seen_tokens:
            raise ValueError(f"{endpoint_id} pagination token repeated")
        seen_tokens.add(token_sha)
        page_token = next_token
    raise ValueError(f"{endpoint_id} pagination bound exhausted")


def _mapping_pages(
    *,
    transport: ReadOnlyHttpsTransport,
    access_token: str,
    endpoint_id: str,
    target: _Target,
    collection_field: str,
    allowed_top_fields: frozenset[str],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Collect a paginated API whose collection field is an object map."""

    values: list[Mapping[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for page_index in range(MAX_PAGES):
        _, payload, next_token, receipt = _issue(
            transport=transport,
            access_token=access_token,
            endpoint_id=endpoint_id,
            target=target,
            page_index=page_index,
            page_token=page_token,
        )
        if not set(payload) <= allowed_top_fields:
            raise ValueError(f"{endpoint_id} response contains unknown fields")
        if endpoint_id in _AGGREGATED_INVENTORY:
            collection_name, expected_kind = _AGGREGATED_INVENTORY[endpoint_id]
            expected_id = (
                f"projects/{target.project}/aggregated/{collection_name}"
            )
            expected_self_link = (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{target.project}/aggregated/{collection_name}"
            )
            if (
                not {"kind", "id", "items", "selfLink"} <= set(payload)
                or payload.get("kind") != expected_kind
                or payload.get("id") != expected_id
                or payload.get("selfLink") != expected_self_link
                or payload.get("unreachables", []) != []
            ):
                raise ValueError("aggregated instance inventory proof is incomplete")
        page_values = payload.get(collection_field, {})
        if not isinstance(page_values, Mapping):
            raise ValueError(f"{endpoint_id} mapping field changed")
        values.append(dict(page_values))
        receipts.append(receipt)
        if next_token is None:
            return values, receipts
        token_sha = hashlib.sha256(next_token.encode("utf-8")).hexdigest()
        if token_sha in seen_tokens:
            raise ValueError(f"{endpoint_id} pagination token repeated")
        seen_tokens.add(token_sha)
        page_token = next_token
    raise ValueError(f"{endpoint_id} pagination bound exhausted")


def _image_observation(
    value: Mapping[str, Any], *, observation_id: str, observed_at: str
) -> dict[str, Any]:
    _exact_or_allowed_fields(
        value,
        required=frozenset(
            {"name", "id", "status", "selfLink", "family", "guestOsFeatures"}
        ),
        allowed=frozenset(
            {
                "kind",
                "id",
                "creationTimestamp",
                "name",
                "description",
                "sourceType",
                "rawDisk",
                "status",
                "archiveSizeBytes",
                "diskSizeGb",
                "licenses",
                "licenseCodes",
                "family",
                "selfLink",
                "labels",
                "labelFingerprint",
                "guestOsFeatures",
                "shieldedInstanceInitialState",
                "storageLocations",
                "architecture",
                "enableConfidentialCompute",
                "deprecated",
                "satisfiesPzi",
                "satisfiesPzs",
            }
        ),
        label="image get-from-family",
    )
    name = value.get("name")
    image_id = value.get("id")
    raw_guest_features = value.get("guestOsFeatures")
    if (
        not isinstance(raw_guest_features, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"type"}
            or not isinstance(row.get("type"), str)
            or not row["type"]
            or row["type"] != row["type"].upper()
            for row in raw_guest_features
        )
    ):
        raise ValueError("Debian image guest OS feature surface changed")
    guest_os_features = sorted(row["type"] for row in raw_guest_features)
    self_link = (
        "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
        f"global/images/{name}"
    )
    if (
        not isinstance(name, str)
        or _IMAGE_NAME.fullmatch(name) is None
        or not isinstance(image_id, str)
        or _DIGITS.fullmatch(image_id) is None
        or value.get("family") != IMAGE_FAMILY
        or value.get("status") != "READY"
        or value.get("selfLink") != self_link
        or value.get("deprecated") is not None
        or value.get("architecture", "X86_64") != "X86_64"
        or len(guest_os_features) != len(set(guest_os_features))
        or "GVNIC" not in guest_os_features
        or (
            "labels" in value
            and (
                not isinstance(value["labels"], Mapping)
                or any(
                    not isinstance(key, str) or not isinstance(item, str)
                    for key, item in value["labels"].items()
                )
            )
        )
        or (
            "licenseCodes" in value
            and (
                not isinstance(value["licenseCodes"], list)
                or any(
                    not isinstance(code, str) or _DIGITS.fullmatch(code) is None
                    for code in value["licenseCodes"]
                )
            )
        )
    ):
        raise ValueError("Debian 12 get-from-family image is not active and READY")
    observation = {
        "schema": preflight.IMAGE_OBSERVATION_SCHEMA,
        "observation_id": observation_id,
        "observed_at_utc": observed_at,
        "project": IMAGE_PROJECT,
        "name": name,
        "id": image_id,
        "selfLink": self_link,
        "status": "READY",
        "deprecation": {"state": "ACTIVE", "replacement": None},
        "guest_os_features": guest_os_features,
        "read_only": True,
    }
    return preflight.validate_image_observation(observation)


def _machine(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_or_allowed_fields(
        value,
        required=frozenset({"name", "guestCpus", "memoryMb", "zone", "selfLink"}),
        allowed=frozenset(
            {
                "kind",
                "id",
                "creationTimestamp",
                "name",
                "description",
                "guestCpus",
                "memoryMb",
                "imageSpaceGb",
                "maximumPersistentDisks",
                "maximumPersistentDisksSizeGb",
                "architecture",
                "zone",
                "selfLink",
                "isSharedCpu",
                "accelerators",
                "deprecated",
            }
        ),
        label="machine type",
    )
    expected_self_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
        f"{ZONE}/machineTypes/{MACHINE_TYPE}"
    )
    zone_values = {
        ZONE,
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}",
    }
    if (
        value.get("name") != MACHINE_TYPE
        or value.get("guestCpus") != 16
        or value.get("memoryMb") != 61_440
        or value.get("architecture") != "X86_64"
        or value.get("zone") not in zone_values
        or value.get("selfLink") != expected_self_link
        or value.get("deprecated") is not None
    ):
        raise ValueError("c4-standard-16 machine shape changed")
    return {"name": MACHINE_TYPE, "guest_cpus": 16, "memory_mb": 61_440}


def _c4_limit(value: Mapping[str, Any]) -> int:
    _exact_or_allowed_fields(
        value,
        required=frozenset(
            {
                "name",
                "quotaId",
                "metric",
                "service",
                "isPrecise",
                "containerType",
                "metricUnit",
                "dimensions",
                "dimensionsInfos",
            }
        ),
        allowed=frozenset(
            {
                "name",
                "quotaId",
                "metric",
                "service",
                "isPrecise",
                "containerType",
                "metricUnit",
                "dimensions",
                "dimensionsInfos",
                "metricDisplayName",
                "quotaDisplayName",
                "quotaIncreaseEligibility",
                "displayName",
                "description",
                "refreshInterval",
            }
        ),
        label="Cloud Quotas C4",
    )
    name = value.get("name")
    if (
        not isinstance(name, str)
        or re.fullmatch(
            r"projects/[0-9]+/locations/global/services/compute\.googleapis\.com/"
            rf"quotaInfos/{C4_QUOTA_ID}",
            name,
        )
        is None
        or value.get("quotaId") != C4_QUOTA_ID
        or value.get("metric") != C4_QUOTA_METRIC
        or value.get("service") != "compute.googleapis.com"
        or value.get("isPrecise") is not True
        or value.get("containerType") != "PROJECT"
        or value.get("metricUnit") != "1"
        or value.get("dimensions") != ["region", "vm_family"]
        or not isinstance(value.get("dimensionsInfos"), list)
        or (
            "quotaDisplayName" in value
            and (
                not isinstance(value["quotaDisplayName"], str)
                or not value["quotaDisplayName"]
            )
        )
        or (
            "quotaIncreaseEligibility" in value
            and (
                not isinstance(value["quotaIncreaseEligibility"], Mapping)
                or set(value["quotaIncreaseEligibility"]) != {"isEligible"}
                or type(value["quotaIncreaseEligibility"].get("isEligible"))
                is not bool
            )
        )
    ):
        raise ValueError("Cloud Quotas C4 identity or dimensions changed")
    matches = [
        row
        for row in value["dimensionsInfos"]
        if isinstance(row, Mapping)
        and row.get("dimensions") == {"region": REGION, "vm_family": "C4"}
    ]
    if len(matches) != 1:
        raise ValueError("Cloud Quotas Tokyo C4 dimension is ambiguous or absent")
    match = matches[0]
    if (
        match.get("applicableLocations") != [REGION]
        or not isinstance(match.get("details"), Mapping)
    ):
        raise ValueError("Cloud Quotas Tokyo C4 dimension details changed")
    raw_limit = match["details"].get("value")
    if not isinstance(raw_limit, str) or _DIGITS.fullmatch(raw_limit) is None:
        raise ValueError("Cloud Quotas C4 limit is not an exact string int64")
    return _integral(raw_limit, "Cloud Quotas C4 limit", positive=True)


def _inventory_usage(pages: Sequence[Any]) -> tuple[int, dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_names: set[tuple[str, str]] = set()
    active_c4: list[dict[str, Any]] = []
    for raw_page in pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("aggregated instance page items changed")
        for scope, raw_scope in raw_page.items():
            if (
                not isinstance(scope, str)
                or (
                    scope != "global"
                    and re.fullmatch(r"(?:zones|regions)/[a-z0-9-]+", scope)
                    is None
                )
                or not isinstance(raw_scope, Mapping)
                or not set(raw_scope) <= {"instances", "warning"}
            ):
                raise ValueError("aggregated instance scope changed")
            instances = raw_scope.get("instances", [])
            if not isinstance(instances, list):
                raise ValueError("aggregated instance collection changed")
            if "warning" in raw_scope and instances:
                raise ValueError("aggregated instance warning conflicts with items")
            if "warning" in raw_scope:
                _validate_no_results_warning(raw_scope["warning"])
            if not scope.startswith("zones/"):
                if instances or "warning" not in raw_scope:
                    raise ValueError("non-zonal aggregated instance scope changed")
                continue
            zone = scope.split("/", 1)[1]
            expected_zone_suffix = f"/zones/{zone}"
            for raw in instances:
                if not isinstance(raw, Mapping):
                    raise ValueError("aggregated instance is not an object")
                name = raw.get("name")
                instance_id = raw.get("id")
                machine_link = raw.get("machineType")
                status = raw.get("status")
                zone_link = raw.get("zone")
                if (
                    not isinstance(name, str)
                    or _SAFE_COMPUTE_NAME.fullmatch(name) is None
                    or not isinstance(instance_id, str)
                    or _DIGITS.fullmatch(instance_id) is None
                    or not isinstance(machine_link, str)
                    or not isinstance(zone_link, str)
                    or not zone_link.endswith(expected_zone_suffix)
                    or status not in _INSTANCE_STATUSES
                ):
                    raise ValueError("aggregated instance identity changed")
                identity = (zone, name)
                if instance_id in seen_ids or identity in seen_names:
                    raise ValueError("aggregated instance inventory is duplicated")
                seen_ids.add(instance_id)
                seen_names.add(identity)
                machine_name = machine_link.rsplit("/", 1)[-1]
                region = zone.rsplit("-", 1)[0]
                if (
                    region == REGION
                    and machine_name.startswith("c4-")
                    and status != "TERMINATED"
                ):
                    match = _C4_MACHINE.fullmatch(machine_name)
                    if match is None:
                        raise ValueError("live Tokyo C4 machine cannot be exactly sized")
                    active_c4.append(
                        {
                            "id": instance_id,
                            "zone": zone,
                            "name": name,
                            "machine_type": machine_name,
                            "status": status,
                            "vcpus": int(match.group(1)),
                        }
                    )
    active_c4.sort(key=lambda row: (row["zone"], row["name"], row["id"]))
    usage = sum(row["vcpus"] for row in active_c4)
    proof = {
        "scope": f"{PROJECT}:{REGION}:C4:nonterminal-instances",
        "instance_count": len(active_c4),
        "usage_vcpus": usage,
        "instances_sha256": _sha256(active_c4),
        "duplicate_instance_count": 0,
        "all_pages_exhausted": True,
    }
    return usage, proof


def _validate_no_results_warning(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"code", "message", "data"}
        or value.get("code") != "NO_RESULTS_ON_PAGE"
        or not isinstance(value.get("message"), str)
        or not value["message"]
        or not isinstance(value.get("data"), list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"key", "value"}
            or not isinstance(row.get("key"), str)
            or not isinstance(row.get("value"), str)
            for row in value["data"]
        )
    ):
        raise ValueError("aggregated inventory warning changed")


def _require_empty_supporting_inventory(
    pages: Sequence[Any], *, resource_collection: str
) -> dict[str, Any]:
    scopes: set[str] = set()
    for raw_page in pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("supporting compute inventory page changed")
        for scope, raw_scope in raw_page.items():
            if (
                not isinstance(scope, str)
                or (
                    scope != "global"
                    and re.fullmatch(r"(?:zones|regions)/[a-z0-9-]+", scope)
                    is None
                )
                or not isinstance(raw_scope, Mapping)
                or not set(raw_scope) <= {resource_collection, "warning"}
            ):
                raise ValueError("supporting compute inventory scope changed")
            resources = raw_scope.get(resource_collection, [])
            if not isinstance(resources, list) or resources:
                raise ValueError(
                    f"nonempty {resource_collection} inventory prevents exact C4 usage proof"
                )
            if "warning" in raw_scope:
                _validate_no_results_warning(raw_scope["warning"])
            scopes.add(scope)
    return {
        "resource_collection": resource_collection,
        "resource_count": 0,
        "scope_count": len(scopes),
        "scope_names_sha256": _sha256(sorted(scopes)),
        "all_pages_exhausted": True,
    }


def _region_quotas(value: Mapping[str, Any], *, c4_usage: int) -> tuple[int, int]:
    _exact_or_allowed_fields(
        value,
        required=frozenset({"name", "selfLink", "quotas"}),
        allowed=frozenset(
            {
                "kind",
                "id",
                "creationTimestamp",
                "name",
                "description",
                "status",
                "zones",
                "quotas",
                "selfLink",
                "supportsPzs",
                "deprecated",
            }
        ),
        label="Compute region",
    )
    expected_self_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/{REGION}"
    )
    if (
        value.get("name") != REGION
        or value.get("selfLink") != expected_self_link
        or not isinstance(value.get("quotas"), list)
    ):
        raise ValueError("Compute region identity changed")
    found: dict[str, tuple[int, int]] = {}
    for raw in value["quotas"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Compute region quota entry changed")
        metric = raw.get("metric")
        if metric not in {SPOT_QUOTA_METRIC, C4_LEGACY_QUOTA_METRIC}:
            continue
        if metric in found:
            raise ValueError("Compute region quota metric is duplicated")
        limit = _integral(raw.get("limit"), f"{metric} limit")
        usage = _integral(raw.get("usage"), f"{metric} usage")
        if usage > limit:
            raise ValueError("Compute region quota usage exceeds limit")
        found[metric] = (limit, usage)
    if SPOT_QUOTA_METRIC not in found:
        raise ValueError("Compute region PREEMPTIBLE_CPUS quota is absent")
    if (
        C4_LEGACY_QUOTA_METRIC in found
        and found[C4_LEGACY_QUOTA_METRIC][1] != c4_usage
    ):
        raise ValueError("legacy C4 usage disagrees with live inventory")
    return found[SPOT_QUOTA_METRIC]


def _money(value: Mapping[str, Any]) -> Decimal:
    if not isinstance(value, Mapping) or value.get("currencyCode") != "USD":
        raise ValueError("Billing SKU money currency changed")
    units_raw = value.get("units", "0")
    nanos_raw = value.get("nanos", 0)
    units = _integral(units_raw, "Billing SKU money units")
    nanos = _integral(nanos_raw, "Billing SKU money nanos")
    if nanos >= 1_000_000_000:
        raise ValueError("Billing SKU nanos are out of range")
    return Decimal(units) + Decimal(nanos) / Decimal(1_000_000_000)


def _sku_unit_price(info: Mapping[str, Any], expected_unit: str) -> Decimal:
    expression = info.get("pricingExpression")
    if not isinstance(expression, Mapping):
        raise ValueError("Billing SKU pricing expression changed")
    if (
        expression.get("usageUnit") != expected_unit
        or Decimal(str(expression.get("displayQuantity", 1))) != Decimal(1)
        or not isinstance(expression.get("tieredRates"), list)
        or len(expression["tieredRates"]) != 1
    ):
        raise ValueError("Billing SKU pricing unit or tiers changed")
    rate = expression["tieredRates"][0]
    if (
        not isinstance(rate, Mapping)
        or Decimal(str(rate.get("startUsageAmount", 0))) != Decimal(0)
        or not isinstance(rate.get("unitPrice"), Mapping)
    ):
        raise ValueError("Billing SKU zero-based rate changed")
    return _money(rate["unitPrice"])


def _price(
    skus: Sequence[Any], *, observed_at_unix_seconds: int
) -> tuple[Decimal, dict[str, Any]]:
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    matches: dict[str, list[tuple[Mapping[str, Any], Decimal, int]]] = {
        "CPU": [],
        "RAM": [],
    }
    for raw in skus:
        if not isinstance(raw, Mapping):
            raise ValueError("Billing SKU is not an object")
        _exact_or_allowed_fields(
            raw,
            required=frozenset(
                {"name", "skuId", "description", "category", "serviceRegions", "pricingInfo"}
            ),
            allowed=frozenset(
                {
                    "name",
                    "skuId",
                    "description",
                    "category",
                    "serviceRegions",
                    "serviceProviderName",
                    "pricingInfo",
                    "geoTaxonomy",
                }
            ),
            label="Billing SKU",
        )
        name = raw.get("name")
        sku_id = raw.get("skuId")
        if (
            not isinstance(name, str)
            or not name.startswith(f"services/{COMPUTE_SERVICE_ID}/skus/")
            or not isinstance(sku_id, str)
            or not sku_id
            or name in seen_names
            or sku_id in seen_ids
        ):
            raise ValueError("Billing SKU identity is missing or duplicated")
        seen_names.add(name)
        seen_ids.add(sku_id)
        category = raw.get("category")
        if not isinstance(category, Mapping):
            raise ValueError("Billing SKU category changed")
        group = category.get("resourceGroup")
        if (
            group not in matches
            or category.get("serviceDisplayName") != "Compute Engine"
            or category.get("resourceFamily") != "Compute"
            or category.get("usageType") != "Preemptible"
            or raw.get("serviceProviderName") != "Google"
            or raw.get("serviceRegions") != [REGION]
        ):
            continue
        description = raw.get("description")
        if (
            not isinstance(description, str)
            or re.search(r"(?<![a-z0-9])c4(?![a-z0-9])", description.casefold())
            is None
            or (group == "CPU" and "core" not in description.casefold())
            or (
                group == "RAM"
                and not any(
                    token in description.casefold() for token in ("ram", "memory")
                )
            )
        ):
            continue
        pricing = raw.get("pricingInfo")
        if not isinstance(pricing, list) or not pricing:
            raise ValueError("Billing SKU pricing history is absent")
        applicable: list[tuple[int, Mapping[str, Any]]] = []
        seen_effective: set[int] = set()
        for info in pricing:
            if not isinstance(info, Mapping):
                raise ValueError("Billing SKU pricing record changed")
            effective = info.get("effectiveTime")
            if not isinstance(effective, str) or not effective.endswith("Z"):
                raise ValueError("Billing SKU effective time changed")
            try:
                effective_unix = int(
                    datetime.fromisoformat(effective[:-1] + "+00:00").timestamp()
                )
            except (OverflowError, ValueError) as exc:
                raise ValueError("Billing SKU effective time is invalid") from exc
            if effective_unix < 0 or effective_unix in seen_effective:
                raise ValueError("Billing SKU effective time is duplicated")
            seen_effective.add(effective_unix)
            if effective_unix <= observed_at_unix_seconds:
                applicable.append((effective_unix, info))
        if not applicable:
            raise ValueError("Billing SKU has no price active at observation time")
        effective_unix, selected = max(applicable, key=lambda row: row[0])
        unit = "h" if group == "CPU" else "GiBy.h"
        matches[group].append(
            (raw, _sku_unit_price(selected, unit), effective_unix)
        )
    if len(matches["CPU"]) != 1 or len(matches["RAM"]) != 1:
        raise ValueError("exact Tokyo C4 Spot CPU/RAM SKUs are ambiguous")
    cpu, cpu_unit, cpu_effective = matches["CPU"][0]
    ram, ram_unit, ram_effective = matches["RAM"][0]
    total = Decimal(16) * cpu_unit + Decimal(60) * ram_unit
    evidence = {
        "currency": "USD",
        "region": REGION,
        "provisioning_model": "SPOT",
        "cpu_sku_id": cpu["skuId"],
        "cpu_unit_usd_per_core_hour": _canonical_decimal(cpu_unit),
        "cpu_effective_at_unix_seconds": cpu_effective,
        "ram_sku_id": ram["skuId"],
        "ram_unit_usd_per_giby_hour": _canonical_decimal(ram_unit),
        "ram_effective_at_unix_seconds": ram_effective,
        "formula": "16*cpu_spot_per_core_hour+60*ram_spot_per_giby_hour",
        "vm_hour_usd": _canonical_decimal(total),
    }
    return total, evidence


def _absence_payload(value: Mapping[str, Any]) -> None:
    if set(value) != {"error"} or not isinstance(value.get("error"), Mapping):
        raise ValueError("Compute GET404 response shape changed")
    if value["error"].get("code") != 404:
        raise ValueError("Compute GET404 response code changed")


def _transport_provenance(
    transport: ReadOnlyHttpsTransport, *, require_live_transport: bool
) -> dict[str, Any]:
    backend_id = getattr(transport, "backend_id", None)
    fixture_only = getattr(transport, "fixture_only", None)
    external = getattr(transport, "external_cloud_read_performed", None)
    if (
        not isinstance(backend_id, str)
        or not backend_id
        or type(fixture_only) is not bool
        or type(external) is not bool
        or (fixture_only and external)
    ):
        raise ValueError("read-only transport provenance changed")
    exact_live = type(transport) is StdlibGoogleJsonReadOnlyTransport
    if require_live_transport and (
        not exact_live or fixture_only is not False or external is not True
    ):
        raise ValueError("live collection requires the exact GET-only stdlib transport")
    return {
        "backend_id": backend_id,
        "fixture_only": fixture_only,
        "external_cloud_read_performed": external,
        "exact_stdlib_get_only_transport": exact_live,
    }


def collect_read_only_observations(
    *,
    project: str,
    bucket: str,
    run_name: str,
    identity_namespace: str,
    result_prefix: str,
    transport: ReadOnlyHttpsTransport,
    token_source: AccessTokenSource,
    observed_at_unix_seconds: int,
    now_unix_seconds: int | None = None,
    require_live_transport: bool = True,
) -> dict[str, Any]:
    """Collect and normalize all live evidence without mutation authority."""

    if type(observed_at_unix_seconds) is not int or observed_at_unix_seconds < 0:
        raise ValueError("observation timestamp changed")
    target = _target(
        project=project,
        bucket=bucket,
        run_name=run_name,
        identity_namespace=identity_namespace,
        result_prefix=result_prefix,
    )
    transport_evidence = _transport_provenance(
        transport, require_live_transport=require_live_transport
    )
    access_token = _token(token_source.access_token())
    receipts: list[dict[str, Any]] = []

    _, image_raw, receipt = _single(
        transport=transport,
        access_token=access_token,
        endpoint_id="image_family",
        target=target,
    )
    receipts.append(receipt)
    _, machine_raw, receipt = _single(
        transport=transport,
        access_token=access_token,
        endpoint_id="machine_type",
        target=target,
    )
    receipts.append(receipt)
    machine = _machine(machine_raw)
    _, quota_raw, receipt = _single(
        transport=transport,
        access_token=access_token,
        endpoint_id="cloud_quotas_c4",
        target=target,
    )
    receipts.append(receipt)
    c4_limit = _c4_limit(quota_raw)

    inventory_pages, inventory_receipts = _mapping_pages(
        transport=transport,
        access_token=access_token,
        endpoint_id="aggregated_instances",
        target=target,
        collection_field="items",
        allowed_top_fields=frozenset({"kind", "id", "items", "selfLink", "unreachables"}),
    )
    receipts.extend(inventory_receipts)
    c4_usage, inventory_proof = _inventory_usage(inventory_pages)
    supporting_inventory_proofs: list[dict[str, Any]] = []
    for endpoint in (
        "aggregated_reservations",
        "aggregated_node_groups",
        "aggregated_future_reservations",
    ):
        resource_collection = _AGGREGATED_INVENTORY[endpoint][0]
        supporting_pages, supporting_receipts = _mapping_pages(
            transport=transport,
            access_token=access_token,
            endpoint_id=endpoint,
            target=target,
            collection_field="items",
            allowed_top_fields=frozenset(
                {"kind", "id", "items", "selfLink", "unreachables", "etag"}
            ),
        )
        receipts.extend(supporting_receipts)
        supporting_inventory_proofs.append(
            _require_empty_supporting_inventory(
                supporting_pages, resource_collection=resource_collection
            )
        )
    if c4_usage > c4_limit:
        raise ValueError("live C4 inventory usage exceeds Cloud Quotas limit")

    _, region_raw, receipt = _single(
        transport=transport,
        access_token=access_token,
        endpoint_id="compute_region",
        target=target,
    )
    receipts.append(receipt)
    spot_limit, spot_usage = _region_quotas(region_raw, c4_usage=c4_usage)

    for kind, names in (
        ("instance", target.instance_names),
        ("disk", target.disk_names),
    ):
        for name in names:
            endpoint = f"{kind}:{name}"
            status, payload, receipt = _single(
                transport=transport,
                access_token=access_token,
                endpoint_id=endpoint,
                target=target,
                allowed_statuses=frozenset({200, 404}),
            )
            receipts.append(receipt)
            if status != 404:
                raise ValueError(f"fresh run collision at exact planned {kind}")
            _absence_payload(payload)

    for endpoint in ("result_prefix", "identity_prefix"):
        objects, prefix_receipts = _pages(
            transport=transport,
            access_token=access_token,
            endpoint_id=endpoint,
            target=target,
            collection_field="items",
            allowed_top_fields=frozenset({"kind", "items"}),
        )
        receipts.extend(prefix_receipts)
        if objects:
            raise ValueError(f"fresh namespace collision at {endpoint}")

    skus, billing_receipts = _pages(
        transport=transport,
        access_token=access_token,
        endpoint_id="billing_skus",
        target=target,
        collection_field="skus",
        allowed_top_fields=frozenset({"skus"}),
    )
    receipts.extend(billing_receipts)
    spot_price, pricing_evidence = _price(
        skus, observed_at_unix_seconds=observed_at_unix_seconds
    )

    checked_now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    if (
        type(checked_now) is not int
        or checked_now < 0
        or observed_at_unix_seconds - checked_now > MAX_FUTURE_SKEW_SECONDS
        or checked_now - observed_at_unix_seconds > MAX_OBSERVATION_AGE_SECONDS
    ):
        raise ValueError("live read-only observations are stale")
    observed_at = datetime.fromtimestamp(
        observed_at_unix_seconds, tz=UTC
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence_digest = _sha256(receipts)
    image_observation = _image_observation(
        image_raw,
        observation_id=f"perfdev-v2-image-{evidence_digest[:16]}",
        observed_at=observed_at,
    )
    runtime_observation = {
        "schema": preflight.RUNTIME_OBSERVATION_SCHEMA,
        "observation_id": f"perfdev-v2-runtime-{evidence_digest[16:32]}",
        "observed_at_utc": observed_at,
        "project": target.project,
        "region": REGION,
        "zone": ZONE,
        "machine_type": machine,
        "spot_price": {
            "machine_type": MACHINE_TYPE,
            "provisioning_model": "SPOT",
            "currency": "USD",
            "unit": "vm_hour",
            "value": _canonical_decimal(spot_price),
        },
        "quota": {
            "c4_cpus": {"limit": c4_limit, "usage": c4_usage},
            "spot_cpus": {"limit": spot_limit, "usage": spot_usage},
        },
        "namespace": {
            "run_name": target.run_name,
            "identity_namespace": target.identity_namespace,
            "result_prefix": target.result_prefix,
            "run_name_collision_count": 0,
            "identity_collision_count": 0,
            "result_prefix_collision_count": 0,
            "inventory_read_only": True,
        },
        "read_only": True,
        "cloud_mutated": False,
    }
    runtime_observation = preflight.validate_runtime_observation(
        runtime_observation
    )
    receipt_digests = [receipt["receipt_sha256"] for receipt in receipts]
    collection = {
        "schema": COLLECTION_SCHEMA,
        "status": "live_get_only_observations_ready_local_dry_run_only",
        "observed_at_unix_seconds": observed_at_unix_seconds,
        "observed_at_utc": observed_at,
        "freshness_checked_at_unix_seconds": checked_now,
        "maximum_age_seconds": MAX_OBSERVATION_AGE_SECONDS,
        "target": {
            "project": target.project,
            "bucket": target.bucket,
            "region": REGION,
            "zone": ZONE,
            "machine_type": MACHINE_TYPE,
            "run_name": target.run_name,
            "identity_namespace": target.identity_namespace,
            "result_prefix": target.result_prefix,
            "identity_prefix": target.identity_prefix,
            "instance_names": list(target.instance_names),
            "disk_names": list(target.disk_names),
        },
        "image_observation": image_observation,
        "runtime_observation": runtime_observation,
        "capacity_evidence": {
            "cloud_quotas_c4_dimension": {"region": REGION, "vm_family": "C4"},
            "cloud_quotas_c4_limit": c4_limit,
            "live_c4_inventory_usage": c4_usage,
            "live_inventory_proof": inventory_proof,
            "empty_supporting_inventory_proofs": supporting_inventory_proofs,
            "c4_usage_proof_policy": (
                "nonterminal_target_region_c4_instance_vcpus_with_empty_"
                "reservations_nodeGroups_futureReservations"
            ),
            "regional_spot_limit": spot_limit,
            "regional_spot_usage": spot_usage,
        },
        "pricing_evidence": pricing_evidence,
        "query_receipts": receipts,
        "query_receipt_digests": receipt_digests,
        "query_receipts_sha256": _sha256(receipts),
        "query_count": len(receipts),
        "all_query_methods": ["GET"],
        "all_pagination_exhausted": True,
        "transport": transport_evidence,
        "access_token_recorded": False,
        "response_bodies_recorded": False,
        "raw_pagination_tokens_recorded": False,
        "request_bodies_present": False,
        "gcloud_invoked": False,
        "cloud_mutated": False,
        "package_built": False,
        "cloud_executable": False,
        "launch_authorized": False,
        "instances_created": False,
        "current_profile_changed": False,
    }
    collection["collection_sha256"] = _sha256(collection)
    return collection


__all__ = [
    "BUCKET",
    "COLLECTION_SCHEMA",
    "EnvironmentAccessTokenSource",
    "HttpResponse",
    "PROJECT",
    "QUERY_RECEIPT_SCHEMA",
    "StdlibGoogleJsonReadOnlyTransport",
    "collect_read_only_observations",
]
