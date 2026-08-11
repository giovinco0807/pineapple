"""Fresh read-only collectors for the Step12b pre-insert gate.

The builders in ``external_preflight_gate_v2`` validate provider bodies but do
not perform network access.  This module obtains those bodies through fixed
GET-only Google JSON API surfaces and immediately feeds them to the builders.
It deliberately has no POST, PUT, PATCH, or DELETE method.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as readonly_preflight,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_preflight_gate_v2
    as external_preflight,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_cloud_adapters_v2
    as live_cloud,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_live_preflight_collectors_v2"
)
MAX_PAGES = 100
PAGE_SIZE = 500
REQUEST_TIMEOUT_SECONDS = 60


class LivePreflightCollectionError(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code

    def __str__(self) -> str:
        return self.code


def _token(value: Any) -> str:
    return live_cloud._checked_token(value)


def _json(
    response: rest_iam.HttpResponse,
    *,
    operation: str,
    expected_status: int = 200,
) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise LivePreflightCollectionError(
            f"{operation}_failed", status_code=response.status_code
        )
    return live_cloud._json_object(response, operation=operation)


def _get(
    *,
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
    url: str,
    operation: str,
) -> rest_iam.HttpResponse:
    return http_client.request(
        method="GET",
        url=url,
        headers={
            "Authorization": f"Bearer {_token(token_source.access_token())}",
            "Accept": "application/json",
        },
        body=None,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


def _gs_prefix_name(uri: str) -> str:
    parsed = urllib.parse.urlsplit(uri)
    if (
        parsed.scheme != "gs"
        or parsed.netloc != transport.BUCKET
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("direct-v2 stage prefix changed")
    return urllib.parse.unquote(parsed.path[1:]).rstrip("/") + "/"


def collect_direct_v2_prefix_empty_receipt(
    deployment_contract: Mapping[str, Any],
    *,
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
    observed_at_unix_seconds: int,
) -> dict[str, Any]:
    stage_prefix = deployment_contract.get("remote_layout", {}).get(
        "stage_prefix"
    )
    if not isinstance(stage_prefix, str):
        raise ValueError("direct-v2 stage prefix is missing")
    prefix = _gs_prefix_name(stage_prefix)
    pages: list[dict[str, Any]] = []
    page_token: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_PAGES):
        query = {
            "prefix": prefix,
            "pageSize": str(PAGE_SIZE),
            "fields": "items(name,generation,size),nextPageToken",
        }
        if page_token is not None:
            query["pageToken"] = page_token
        response = _get(
            http_client=http_client,
            token_source=token_source,
            url=(
                "https://storage.googleapis.com/storage/v1/b/"
                f"{urllib.parse.quote(transport.BUCKET, safe='')}/o?"
                f"{urllib.parse.urlencode(query)}"
            ),
            operation="direct_v2_prefix_list",
        )
        page = _json(response, operation="direct_v2_prefix_list")
        if not set(page).issubset({"items", "nextPageToken"}):
            raise LivePreflightCollectionError(
                "direct_v2_prefix_list_response_changed"
            )
        pages.append(page)
        if page.get("items", []):
            raise FileExistsError("direct-v2 stage prefix is not empty")
        next_token = page.get("nextPageToken")
        if next_token is None:
            break
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen
        ):
            raise LivePreflightCollectionError(
                "direct_v2_prefix_pagination_changed"
            )
        seen.add(next_token)
        page_token = next_token
    else:
        raise LivePreflightCollectionError(
            "direct_v2_prefix_page_bound_exhausted"
        )
    return external_preflight.build_direct_v2_prefix_empty_receipt(
        deployment_contract,
        observed_at_unix_seconds=observed_at_unix_seconds,
        provider_pages=pages,
    )


def _compute_url(kind: str, name: str | None = None) -> str:
    base = (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{transport.PROJECT}/zones/{transport.ZONE}/{kind}"
    )
    return base if name is None else (
        f"{base}/{urllib.parse.quote(name, safe='')}"
    )


def _collect_operation_target_links(
    *,
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
) -> list[str]:
    links: list[str] = []
    page_token: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_PAGES):
        query = {"maxResults": str(PAGE_SIZE)}
        if page_token is not None:
            query["pageToken"] = page_token
        response = _get(
            http_client=http_client,
            token_source=token_source,
            url=(
                f"{_compute_url('operations')}?"
                f"{urllib.parse.urlencode(query)}"
            ),
            operation="zone_operation_history_list",
        )
        page = _json(response, operation="zone_operation_history_list")
        if not set(page).issuperset({"items"}) and "items" in page:
            raise LivePreflightCollectionError(
                "zone_operation_history_response_changed"
            )
        rows = page.get("items", [])
        if not isinstance(rows, list):
            raise LivePreflightCollectionError(
                "zone_operation_history_response_changed"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise LivePreflightCollectionError(
                    "zone_operation_history_response_changed"
                )
            target_link = row.get("targetLink")
            if target_link is not None:
                if not isinstance(target_link, str):
                    raise LivePreflightCollectionError(
                        "zone_operation_history_response_changed"
                    )
                links.append(target_link)
        next_token = page.get("nextPageToken")
        if next_token is None:
            return links
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen
        ):
            raise LivePreflightCollectionError(
                "zone_operation_history_pagination_changed"
            )
        seen.add(next_token)
        page_token = next_token
    raise LivePreflightCollectionError(
        "zone_operation_history_page_bound_exhausted"
    )


def collect_compute_absence_receipt(
    deployment_contract: Mapping[str, Any],
    *,
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
    observed_at_unix_seconds: int,
) -> dict[str, Any]:
    names = [
        row.get("instance_name")
        for row in deployment_contract.get("instances", [])
        if isinstance(row, Mapping)
    ]
    if (
        len(names) != 2
        or len(set(names)) != 2
        or any(not isinstance(name, str) for name in names)
    ):
        raise ValueError("exact pair instance names changed")
    readbacks: dict[str, list[dict[str, Any]]] = {
        "instances": [],
        "disks": [],
    }
    for kind in ("instances", "disks"):
        for name in names:
            response = _get(
                http_client=http_client,
                token_source=token_source,
                url=_compute_url(kind, name),
                operation=f"{kind}_absence_get",
            )
            if response.status_code != 404:
                raise FileExistsError(f"Step12b {kind}/{name} is not absent")
            readbacks[kind].append(
                {"name": name, "http_status": 404}
            )
    target_links = _collect_operation_target_links(
        http_client=http_client, token_source=token_source
    )
    history: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        suffixes = (
            f"/instances/{name}",
            f"/disks/{name}",
        )
        matches = [
            link for link in target_links if link.endswith(suffixes)
        ]
        if matches:
            raise FileExistsError(
                f"Step12b operation/name history is nonzero: {name}"
            )
        history[name] = []
    return external_preflight.build_compute_absence_receipt(
        deployment_contract,
        observed_at_unix_seconds=observed_at_unix_seconds,
        instance_get_readbacks=readbacks["instances"],
        disk_get_readbacks=readbacks["disks"],
        operation_name_history=history,
    )


def _read_json_url(
    *,
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
    url: str,
    operation: str,
) -> dict[str, Any]:
    return _json(
        _get(
            http_client=http_client,
            token_source=token_source,
            url=url,
            operation=operation,
        ),
        operation=operation,
    )


def _collect_enabled_services(
    *,
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_PAGES):
        query = {"filter": "state:ENABLED", "pageSize": "200"}
        if token is not None:
            query["pageToken"] = token
        page = _read_json_url(
            http_client=http_client,
            token_source=token_source,
            url=(
                "https://serviceusage.googleapis.com/v1/projects/"
                f"{transport.PROJECT}/services?"
                f"{urllib.parse.urlencode(query)}"
            ),
            operation="enabled_services_list",
        )
        services = page.get("services", [])
        if (
            not isinstance(services, list)
            or any(not isinstance(row, dict) for row in services)
        ):
            raise LivePreflightCollectionError(
                "enabled_services_response_changed"
            )
        rows.extend(services)
        next_token = page.get("nextPageToken")
        if next_token is None:
            return rows
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen
        ):
            raise LivePreflightCollectionError(
                "enabled_services_pagination_changed"
            )
        seen.add(next_token)
        token = next_token
    raise LivePreflightCollectionError(
        "enabled_services_page_bound_exhausted"
    )


def _capacity_observations(
    deployment_contract: Mapping[str, Any],
    *,
    token_source: live_cloud.AccessTokenSource,
    observed_at_unix_seconds: int,
) -> dict[str, Any]:
    target = external_preflight._capacity_target(
        dict(deployment_contract)
    )
    active_transport = (
        readonly_preflight.StdlibGoogleJsonReadOnlyTransport()
    )
    access_token = readonly_preflight._validate_token(
        token_source.access_token()
    )
    endpoint_ids = (
        "cloud_quotas_global_cpu",
        "cloud_quotas_c4_cpu",
        *readonly_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS,
    )
    observations: dict[str, Any] = {}
    for endpoint_id in endpoint_ids:
        observation, _ = readonly_preflight._single_query(
            transport=active_transport,
            access_token=access_token,
            endpoint_id=endpoint_id,
            target=target,
            retrieved_at_unix_seconds=observed_at_unix_seconds,
            valid_until_unix_seconds=(
                observed_at_unix_seconds
                + external_preflight.OBSERVATION_MAX_AGE_SECONDS
            ),
        )
        observations[endpoint_id] = observation
    return observations


def collect_live_readback_receipt(
    deployment_contract: Mapping[str, Any],
    *,
    phase2_iam_plan: Mapping[str, Any],
    http_client: live_cloud.HttpsClient,
    token_source: live_cloud.AccessTokenSource,
    observed_at_unix_seconds: int,
) -> dict[str, Any]:
    project = transport.PROJECT
    zone = transport.ZONE
    region = zone.rsplit("-", 1)[0]
    machine = _read_json_url(
        http_client=http_client,
        token_source=token_source,
        url=(
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/zones/{zone}/machineTypes/"
            f"{external_preflight.deployment_v2.ACTUAL_MACHINE_TYPE}"
        ),
        operation="machine_type_get",
    )
    region_readback = _read_json_url(
        http_client=http_client,
        token_source=token_source,
        url=(
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/regions/{region}"
        ),
        operation="region_get",
    )
    router = _read_json_url(
        http_client=http_client,
        token_source=token_source,
        url=(
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{project}/regions/{region}/routers/"
            f"{external_preflight.NAT_ROUTER_NAME}"
        ),
        operation="nat_router_get",
    )
    custom_roles: dict[str, dict[str, Any]] = {}
    requirements = phase2_iam_plan.get(
        "custom_role_readback_contract", {}
    ).get("requirements", [])
    if not isinstance(requirements, list):
        raise ValueError("custom-role readback requirements changed")
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            raise ValueError("custom-role readback requirement changed")
        purpose = requirement.get("purpose")
        role_name = requirement.get("name")
        if (
            not isinstance(purpose, str)
            or not isinstance(role_name, str)
            or not role_name.startswith(f"projects/{project}/roles/")
            or purpose in custom_roles
        ):
            raise ValueError("custom-role readback requirement changed")
        role_id = role_name.rsplit("/", 1)[-1]
        custom_roles[purpose] = _read_json_url(
            http_client=http_client,
            token_source=token_source,
            url=(
                "https://iam.googleapis.com/v1/projects/"
                f"{project}/roles/{urllib.parse.quote(role_id, safe='')}"
            ),
            operation=f"custom_role_get_{purpose}",
        )
    return external_preflight.build_live_readback_receipt(
        deployment_contract,
        phase2_iam_plan=phase2_iam_plan,
        observed_at_unix_seconds=observed_at_unix_seconds,
        machine_type_readback=machine,
        region_readback=region_readback,
        authoritative_capacity_observations=_capacity_observations(
            deployment_contract,
            token_source=token_source,
            observed_at_unix_seconds=observed_at_unix_seconds,
        ),
        nat_router_readback=router,
        enabled_service_readbacks=_collect_enabled_services(
            http_client=http_client, token_source=token_source
        ),
        custom_role_readbacks=custom_roles,
    )


__all__ = [
    "LivePreflightCollectionError",
    "SCHEMA",
    "collect_compute_absence_receipt",
    "collect_direct_v2_prefix_empty_receipt",
    "collect_live_readback_receipt",
]
