from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_live_readonly as live,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_preflight as preflight,
)


OBSERVED_AT = 1_784_680_000
RUN_NAME = "regular-hu-m31-c02-perfdev-v2-fresh001"
IDENTITY = "perfdev-v2-fresh001"
RESULT_PREFIX = f"hu-m31-t3/perfdev-v2/{RUN_NAME}/"


class _TokenSource:
    def access_token(self) -> str:
        return "fixture-access-token-must-not-be-recorded"


def _body(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sku(
    *, sku_id: str, group: str, description: str, nanos: int
) -> dict[str, object]:
    return {
        "name": f"services/{live.COMPUTE_SERVICE_ID}/skus/{sku_id}",
        "skuId": sku_id,
            "description": description,
            "serviceProviderName": "Google",
            "category": {
            "serviceDisplayName": "Compute Engine",
            "resourceFamily": "Compute",
            "resourceGroup": group,
            "usageType": "Preemptible",
        },
        "serviceRegions": [live.REGION],
        "pricingInfo": [
            {
                "effectiveTime": "2026-01-01T00:00:00Z",
                "pricingExpression": {
                    "usageUnit": "h" if group == "CPU" else "GiBy.h",
                    "displayQuantity": 1,
                    "tieredRates": [
                        {
                            "startUsageAmount": 0,
                            "unitPrice": {
                                "currencyCode": "USD",
                                "units": "0",
                                "nanos": nanos,
                            },
                        }
                    ],
                },
            }
        ],
    }


class _FakeTransport:
    backend_id = "fixture-perfdev-v2-readonly"
    fixture_only = True
    external_cloud_read_performed = False

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.repeat_billing_token = False
        self.unknown_billing_field = False
        self.ambiguous_cpu_sku = False
        self.instance_collision = False
        self.result_collision = False
        self.identity_collision = False
        self.reservation_collision = False
        self.duplicate_image_json = False
        self.image_name = "debian-12-bookworm-v20260721"
        self.c4_limit = "128"
        self.spot_limit = 468
        self.spot_usage = 32
        self.cpu_nanos = 23_290_000

    def _response(
        self, value: object, *, status: int = 200
    ) -> live.HttpResponse:
        return live.HttpResponse(status_code=status, body=_body(value), headers={})

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> live.HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "timeout": timeout_seconds,
            }
        )
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        page_token = query.get("pageToken", [None])[0]

        if "/images/family/debian-12" in parts.path:
            if self.duplicate_image_json:
                return live.HttpResponse(
                    status_code=200,
                    body=(
                        b'{"name":"a","name":"b","id":"1",'
                        b'"family":"debian-12","status":"READY",'
                        b'"selfLink":"x"}'
                    ),
                    headers={},
                )
            return self._response(
                {
                    "kind": "compute#image",
                    "id": "9021508813201755912",
                    "name": self.image_name,
                    "family": "debian-12",
                    "status": "READY",
                    "architecture": "X86_64",
                    "guestOsFeatures": [
                        {"type": "UEFI_COMPATIBLE"},
                        {"type": "GVNIC"},
                        {"type": "VIRTIO_SCSI_MULTIQUEUE"},
                    ],
                    "labels": {"build": "debian-cloud"},
                    "licenseCodes": ["1000205"],
                    "selfLink": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"debian-cloud/global/images/{self.image_name}"
                    ),
                }
            )
        if "/machineTypes/c4-standard-16" in parts.path:
            return self._response(
                {
                    "kind": "compute#machineType",
                    "id": "16001",
                    "name": "c4-standard-16",
                    "guestCpus": 16,
                    "memoryMb": 61_440,
                    "architecture": "X86_64",
                    "zone": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{live.PROJECT}/zones/{live.ZONE}"
                    ),
                    "selfLink": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{live.PROJECT}/zones/{live.ZONE}/machineTypes/"
                        "c4-standard-16"
                    ),
                }
            )
        if "/quotaInfos/" in parts.path:
            return self._response(
                {
                    "name": (
                        "projects/123456789/locations/global/services/"
                        "compute.googleapis.com/quotaInfos/"
                        f"{live.C4_QUOTA_ID}"
                    ),
                    "quotaId": live.C4_QUOTA_ID,
                    "metric": live.C4_QUOTA_METRIC,
                    "service": "compute.googleapis.com",
                    "isPrecise": True,
                    "containerType": "PROJECT",
                    "metricUnit": "1",
                    "quotaDisplayName": "C4 CPUs per VM family",
                    "quotaIncreaseEligibility": {"isEligible": True},
                    "dimensions": ["region", "vm_family"],
                    "dimensionsInfos": [
                        {
                            "dimensions": {
                                "region": live.REGION,
                                "vm_family": "C4",
                            },
                            "applicableLocations": [live.REGION],
                            "details": {"value": self.c4_limit},
                        }
                    ],
                }
            )
        if "/aggregated/instances" in parts.path:
            base = {
                "kind": "compute#instanceAggregatedList",
                "id": f"projects/{live.PROJECT}/aggregated/instances",
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{live.PROJECT}/aggregated/instances"
                ),
            }
            if page_token is None:
                return self._response(
                    {
                        **base,
                        "items": {
                            "global": {
                                "warning": {
                                    "code": "NO_RESULTS_ON_PAGE",
                                    "message": "No results",
                                    "data": [{"key": "scope", "value": "global"}],
                                }
                            },
                            f"regions/{live.REGION}": {
                                "warning": {
                                    "code": "NO_RESULTS_ON_PAGE",
                                    "message": "No results",
                                    "data": [{"key": "scope", "value": live.REGION}],
                                }
                            },
                            "zones/us-central1-a": {
                                "instances": [
                                    {
                                        "name": "unrelated-standard-vm",
                                        "id": "3001",
                                        "zone": (
                                            "https://www.googleapis.com/compute/v1/"
                                            f"projects/{live.PROJECT}/zones/us-central1-a"
                                        ),
                                        "machineType": (
                                            "https://www.googleapis.com/compute/v1/"
                                            f"projects/{live.PROJECT}/zones/us-central1-a/"
                                            "machineTypes/n2-standard-16"
                                        ),
                                        "status": "RUNNING",
                                    }
                                ]
                            }
                        },
                        "nextPageToken": "inventory-next",
                    }
                )
            assert page_token == "inventory-next"
            return self._response(
                {**base, "items": {f"zones/{live.ZONE}": {}}}
            )
        for collection, kind in (
            ("reservations", "compute#reservationAggregatedList"),
            ("nodeGroups", "compute#nodeGroupAggregatedList"),
            (
                "futureReservations",
                "compute#futureReservationsAggregatedListResponse",
            ),
        ):
            if f"/aggregated/{collection}" in parts.path:
                scope_value = {}
                if collection == "reservations" and self.reservation_collision:
                    scope_value = {"reservations": [{"name": "unpriced-reservation"}]}
                return self._response(
                    {
                        "kind": kind,
                        "id": (
                            f"projects/{live.PROJECT}/aggregated/{collection}"
                        ),
                        "selfLink": (
                            "https://www.googleapis.com/compute/v1/projects/"
                            f"{live.PROJECT}/aggregated/{collection}"
                        ),
                        "items": {f"zones/{live.ZONE}": scope_value},
                        **({"etag": "fixture-etag"} if collection == "futureReservations" else {}),
                    }
                )
        if f"/regions/{live.REGION}" in parts.path:
            return self._response(
                {
                    "kind": "compute#region",
                    "id": "4001",
                    "name": live.REGION,
                    "selfLink": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{live.PROJECT}/regions/{live.REGION}"
                    ),
                    "quotas": [
                        {
                            "metric": live.SPOT_QUOTA_METRIC,
                            "limit": self.spot_limit,
                            "usage": self.spot_usage,
                        }
                    ],
                }
            )
        if "/instances/" in parts.path or "/disks/" in parts.path:
            if self.instance_collision and "/instances/" in parts.path:
                return self._response({"kind": "compute#instance", "name": "taken"})
            return self._response(
                {
                    "error": {
                        "code": 404,
                        "message": "not found",
                        "errors": [],
                        "status": "NOT_FOUND",
                    }
                },
                status=404,
            )
        if parts.hostname == "storage.googleapis.com":
            prefix = query["prefix"][0]
            if "perfdev-v2-identities" in prefix:
                items = [{"name": prefix + "taken"}] if self.identity_collision else []
            else:
                items = [{"name": prefix + "taken"}] if self.result_collision else []
            return self._response({"kind": "storage#objects", "items": items})
        if parts.hostname == "cloudbilling.googleapis.com":
            if page_token is None:
                payload: dict[str, object] = {
                    "skus": [
                        _sku(
                            sku_id="c4-cpu-tokyo",
                            group="CPU",
                            description="Spot C4 instance core running in Tokyo",
                            nanos=self.cpu_nanos,
                        )
                    ],
                    "nextPageToken": "billing-next",
                }
                if self.unknown_billing_field:
                    payload["unknown"] = True
                return self._response(payload)
            assert page_token == "billing-next"
            skus = [
                _sku(
                    sku_id="c4-ram-tokyo",
                    group="RAM",
                    description="Spot C4 instance RAM running in Tokyo",
                    nanos=2_647_000,
                )
            ]
            if self.ambiguous_cpu_sku:
                skus.append(
                    _sku(
                        sku_id="c4-cpu-tokyo-second",
                        group="CPU",
                        description="Spot C4 instance core second Tokyo SKU",
                        nanos=self.cpu_nanos,
                    )
                )
            payload = {"skus": skus}
            if self.repeat_billing_token:
                payload["nextPageToken"] = "billing-next"
            return self._response(payload)
        raise AssertionError(f"unexpected fixture URL: {url}")


def _collect(transport: _FakeTransport) -> dict[str, object]:
    return live.collect_read_only_observations(
        project=live.PROJECT,
        bucket=live.BUCKET,
        run_name=RUN_NAME,
        identity_namespace=IDENTITY,
        result_prefix=RESULT_PREFIX,
        transport=transport,
        token_source=_TokenSource(),
        observed_at_unix_seconds=OBSERVED_AT,
        now_unix_seconds=OBSERVED_AT,
        require_live_transport=False,
    )


def test_paginated_get_only_collection_builds_exact_existing_observations() -> None:
    transport = _FakeTransport()
    collection = _collect(transport)

    image = preflight.validate_image_observation(collection["image_observation"])
    runtime = preflight.validate_runtime_observation(
        collection["runtime_observation"]
    )
    dry_run = preflight.build_dry_run_receipt(
        image_observation=image, runtime_observation=runtime
    )

    assert image["name"] == "debian-12-bookworm-v20260721"
    assert runtime["spot_price"]["value"] == "0.53146"
    assert runtime["quota"] == {
        "c4_cpus": {"limit": 128, "usage": 0},
        "spot_cpus": {"limit": 468, "usage": 32},
    }
    assert runtime["namespace"]["result_prefix"] == (
        f"hu-m31-t3/perfdev-v2/{RUN_NAME}/"
    )
    assert collection["capacity_evidence"]["live_inventory_proof"][
        "all_pages_exhausted"
    ] is True
    assert collection["query_count"] == 17
    assert collection["all_query_methods"] == ["GET"]
    assert dry_run["launch_authorized"] is False
    assert all(call["method"] == "GET" for call in transport.calls)
    assert all(
        call["headers"]["Authorization"]
        == "Bearer fixture-access-token-must-not-be-recorded"
        for call in transport.calls
    )
    serialized = json.dumps(collection, sort_keys=True)
    assert "fixture-access-token-must-not-be-recorded" not in serialized
    assert "inventory-next" not in serialized
    assert "billing-next" not in serialized
    assert collection["request_bodies_present"] is False
    assert collection["cloud_mutated"] is False
    assert collection["launch_authorized"] is False


def test_fixture_transport_cannot_be_misreported_as_live() -> None:
    with pytest.raises(ValueError, match="exact GET-only stdlib transport"):
        live.collect_read_only_observations(
            project=live.PROJECT,
            bucket=live.BUCKET,
            run_name=RUN_NAME,
            identity_namespace=IDENTITY,
            result_prefix=RESULT_PREFIX,
            transport=_FakeTransport(),
            token_source=_TokenSource(),
            observed_at_unix_seconds=OBSERVED_AT,
            now_unix_seconds=OBSERVED_AT,
        )


def test_duplicate_json_field_fails_closed() -> None:
    transport = _FakeTransport()
    transport.duplicate_image_json = True
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _collect(transport)


def test_unknown_response_field_fails_closed() -> None:
    transport = _FakeTransport()
    transport.unknown_billing_field = True
    with pytest.raises(ValueError, match="unknown fields"):
        _collect(transport)


def test_machine_architecture_drift_fails_closed() -> None:
    with pytest.raises(ValueError, match="machine shape changed"):
        live._machine(
            {
                "name": "c4-standard-16",
                "guestCpus": 16,
                "memoryMb": 61_440,
                "architecture": "ARM64",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{live.PROJECT}/zones/{live.ZONE}"
                ),
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{live.PROJECT}/zones/{live.ZONE}/machineTypes/c4-standard-16"
                ),
            }
        )


def test_image_without_gvnic_guest_feature_fails_closed() -> None:
    name = "debian-12-bookworm-v20260721"
    with pytest.raises(ValueError, match="active and READY"):
        live._image_observation(
            {
                "name": name,
                "id": "9021508813201755912",
                "family": "debian-12",
                "status": "READY",
                "architecture": "X86_64",
                "guestOsFeatures": [{"type": "UEFI_COMPATIBLE"}],
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"debian-cloud/global/images/{name}"
                ),
            },
            observation_id="image-gvnic-negative-001",
            observed_at="2026-07-22T15:00:00Z",
        )


def test_non_zonal_instance_scope_requires_exact_no_results_warning() -> None:
    with pytest.raises(ValueError, match="inventory warning changed"):
        live._inventory_usage(
            [
                {
                    f"regions/{live.REGION}": {
                        "warning": {
                            "code": "PARTIAL_FAILURE",
                            "message": "incomplete",
                            "data": [],
                        }
                    }
                }
            ]
        )


def test_repeated_pagination_token_fails_closed() -> None:
    transport = _FakeTransport()
    transport.repeat_billing_token = True
    with pytest.raises(ValueError, match="pagination token repeated"):
        _collect(transport)


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("instance_collision", "planned instance"),
        ("result_collision", "result_prefix"),
        ("identity_collision", "identity_prefix"),
    ],
)
def test_exact_run_and_namespace_collisions_fail_closed(
    attribute: str, message: str
) -> None:
    transport = _FakeTransport()
    setattr(transport, attribute, True)
    with pytest.raises(ValueError, match=message):
        _collect(transport)


def test_ambiguous_tokyo_c4_spot_sku_fails_closed() -> None:
    transport = _FakeTransport()
    transport.ambiguous_cpu_sku = True
    with pytest.raises(ValueError, match="SKUs are ambiguous"):
        _collect(transport)


def test_nonempty_supporting_inventory_prevents_understated_c4_usage() -> None:
    transport = _FakeTransport()
    transport.reservation_collision = True
    with pytest.raises(ValueError, match="nonempty reservations inventory"):
        _collect(transport)


def test_stale_observation_fails_after_get_only_collection() -> None:
    transport = _FakeTransport()
    with pytest.raises(ValueError, match="stale"):
        live.collect_read_only_observations(
            project=live.PROJECT,
            bucket=live.BUCKET,
            run_name=RUN_NAME,
            identity_namespace=IDENTITY,
            result_prefix=RESULT_PREFIX,
            transport=transport,
            token_source=_TokenSource(),
            observed_at_unix_seconds=OBSERVED_AT,
            now_unix_seconds=OBSERVED_AT + 301,
            require_live_transport=False,
        )
    assert transport.calls
    assert all(call["method"] == "GET" for call in transport.calls)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("c4_limit", "16", "topology gate failed"),
        ("spot_usage", 450, "topology gate failed"),
        ("cpu_nanos", 50_000_000, "topology gate failed"),
        ("image_name", "debian-11-bullseye-v20260721", "get-from-family"),
    ],
)
def test_tampered_image_price_or_capacity_fails_existing_preflight_gate(
    field: str, value: object, message: str
) -> None:
    transport = _FakeTransport()
    setattr(transport, field, value)
    with pytest.raises(ValueError, match=message):
        _collect(transport)


def test_fresh_names_reject_lock_rearm2_and_oversized_compute_names() -> None:
    for run_name in (
        "regular-hu-m31-c02-perfdev-v2-rearm2bad",
        "regular-hu-m31-c02-perfdev-v2-" + "x" * 48,
    ):
        with pytest.raises(ValueError, match="live target changed"):
            live.collect_read_only_observations(
                project=live.PROJECT,
                bucket=live.BUCKET,
                run_name=run_name,
                identity_namespace=IDENTITY,
                result_prefix=f"hu-m31-t3/perfdev-v2/{run_name}/",
                transport=_FakeTransport(),
                token_source=_TokenSource(),
                observed_at_unix_seconds=OBSERVED_AT,
                now_unix_seconds=OBSERVED_AT,
                require_live_transport=False,
            )


def test_caller_supplied_result_prefix_must_match_run_name_exactly() -> None:
    with pytest.raises(ValueError, match="live target changed"):
        live.collect_read_only_observations(
            project=live.PROJECT,
            bucket=live.BUCKET,
            run_name=RUN_NAME,
            identity_namespace=IDENTITY,
            result_prefix="hu-m31-t3/perfdev-v2/different-run/",
            transport=_FakeTransport(),
            token_source=_TokenSource(),
            observed_at_unix_seconds=OBSERVED_AT,
            now_unix_seconds=OBSERVED_AT,
            require_live_transport=False,
        )
