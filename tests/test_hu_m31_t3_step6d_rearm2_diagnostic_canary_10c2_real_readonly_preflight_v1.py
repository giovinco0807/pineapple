from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_receiver_preflight as receiver,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as cloud_package,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport_contract,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as diagnostic_plan,
)


NOW = 2_100_000_000
SECRET = "unit-secret-token-that-must-never-be-recorded"
PROJECT_NUMBER = "123456789012"
PreflightFixture = tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]


class FakeTokenSource:
    def access_token(self) -> str:
        return SECRET


def _money(nanos: int) -> dict[str, Any]:
    return {"currencyCode": "USD", "units": "0", "nanos": nanos}


def _sku(
    *,
    group: str,
    sku_id: str,
    region: str = subject.DEFAULT_REGION,
) -> dict[str, Any]:
    is_cpu = group == "CPU"
    return {
        "name": (
            f"services/{subject.COMPUTE_ENGINE_SERVICE_ID}/skus/{sku_id}"
        ),
        "skuId": sku_id,
        "description": (
            "Spot Preemptible C4 Instance Core running in Tokyo"
            if is_cpu
            else "Spot Preemptible C4 Instance RAM running in Tokyo"
        ),
        "category": {
            "serviceDisplayName": "Compute Engine",
            "resourceFamily": "Compute",
            "resourceGroup": group,
            "usageType": "Preemptible",
        },
        "serviceRegions": [region],
        "pricingInfo": [
            {
                "effectiveTime": "2036-07-18T00:00:00Z",
                "pricingExpression": {
                    "usageUnit": "h" if is_cpu else "GiBy.h",
                    "displayQuantity": 1,
                    "tieredRates": [
                        {
                            "startUsageAmount": 0,
                            "unitPrice": _money(
                                10_000_000 if is_cpu else 1_000_000
                            ),
                        }
                    ],
                }
            }
        ],
    }


class FakeGoogleJsonTransport:
    backend_id = "offline-google-json-fake-v1"
    fixture_only = True
    external_cloud_read_performed = False

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        preview: Mapping[str, Any],
        fault: str | None = None,
        wrong_region: bool = False,
        ambiguous_cpu: bool = False,
        instance_status: int = 404,
        service_account_status: int = 200,
        missing_bucket_role: bool = False,
        bucket_iam_mode: str = "least_privilege",
        image_id: str = transport_contract.IMAGE_ID,
        image_deprecated: bool = False,
        global_cpu_limit: int = 400,
        c4_cpu_limit: int = 355,
        regional_spot_usage: int = 0,
        standard_spot_instance_count: int = 0,
        standard_spot_machine_type: str = "n2-standard-16",
        standard_instance_status: str = "RUNNING",
        instance_record_fault: str | None = None,
        inventory_fault: str | None = None,
    ) -> None:
        self.plan = plan
        self.preview = preview
        self.fault = fault
        self.wrong_region = wrong_region
        self.ambiguous_cpu = ambiguous_cpu
        self.instance_status = instance_status
        self.service_account_status = service_account_status
        self.bucket_iam_mode = (
            "missing" if missing_bucket_role else bucket_iam_mode
        )
        self.image_id = image_id
        self.image_deprecated = image_deprecated
        self.global_cpu_limit = global_cpu_limit
        self.c4_cpu_limit = c4_cpu_limit
        self.regional_spot_usage = regional_spot_usage
        self.standard_spot_instance_count = standard_spot_instance_count
        self.standard_spot_machine_type = standard_spot_machine_type
        self.standard_instance_status = standard_instance_status
        self.instance_record_fault = instance_record_fault
        self.inventory_fault = inventory_fault
        self.calls: list[dict[str, Any]] = []
        self.package_prefix = plan["requirements"]["prefix"][
            "package_prefix"
        ].split(
            f"gs://{adapter.DEFAULT_BUCKET}/", 1
        )[1] + "/"
        self.stage_prefix = plan["requirements"]["prefix"]["stage_prefix"].split(
            f"gs://{adapter.DEFAULT_BUCKET}/", 1
        )[1] + "/"
        expected = plan["requirements"]["prefix"]["expected_package_objects"]
        self.package_items = [
            {
                "bucket": adapter.DEFAULT_BUCKET,
                "name": row["uri"].split(
                    f"gs://{adapter.DEFAULT_BUCKET}/", 1
                )[1],
                "size": str(row["bytes"]),
                "generation": str(index + 100),
                "metageneration": "1",
                "etag": f"etag-{index}",
                "crc32c": f"crc32c-{index}",
                "updated": "2036-07-18T00:00:00Z",
                "metadata": {"sha256": row["sha256"]},
            }
            for index, row in enumerate(expected)
        ]

    def _bucket_iam_bindings(self) -> list[dict[str, Any]]:
        member = (
            "serviceAccount:"
            f"{transport_contract.WORKER_SERVICE_ACCOUNT}"
        )
        exact_view = subject._storage_prefix_condition(
            title=subject.STORAGE_VIEW_CONDITION_TITLE,
            bucket=adapter.DEFAULT_BUCKET,
            prefix=f"{transport_contract.DIRECT_NAMESPACE}/",
        )
        exact_create = subject._storage_prefix_condition(
            title=subject.STORAGE_CREATE_CONDITION_TITLE,
            bucket=adapter.DEFAULT_BUCKET,
            prefix=self.stage_prefix,
        )
        mode = self.bucket_iam_mode
        if mode == "missing":
            return []
        if mode == "object_admin_only":
            return [
                {
                    "role": "roles/storage.objectAdmin",
                    "members": [member],
                }
            ]
        if mode == "unconditional_viewer":
            exact_view = None
        elif mode == "wrong_condition":
            exact_view = {
                **exact_view,
                "expression": exact_view["expression"].replace(
                    f"{transport_contract.DIRECT_NAMESPACE}/",
                    f"{transport_contract.DIRECT_NAMESPACE}-wrong/",
                ),
            }
        elif mode == "broad_condition":
            exact_create = subject._storage_prefix_condition(
                title=subject.STORAGE_CREATE_CONDITION_TITLE,
                bucket=adapter.DEFAULT_BUCKET,
                prefix=f"{transport_contract.DIRECT_NAMESPACE}/",
            )
        elif mode != "least_privilege":
            raise AssertionError(f"unknown fake IAM mode: {mode}")
        bindings: list[dict[str, Any]] = [
            {
                "role": subject.STORAGE_OBJECT_VIEWER_ROLE,
                "members": [member],
            },
            {
                "role": subject.STORAGE_OBJECT_CREATOR_ROLE,
                "members": [member],
                "condition": exact_create,
            },
        ]
        if exact_view is not None:
            bindings[0]["condition"] = exact_view
        return bindings

    @staticmethod
    def _response(status: int, value: Mapping[str, Any]) -> subject.HttpResponse:
        return subject.HttpResponse(
            status_code=status,
            body=subject.canonical_bytes(value),
            headers={"Content-Type": "application/json"},
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> subject.HttpResponse:
        assert method == "GET"
        assert headers["Authorization"] == f"Bearer {SECRET}"
        assert headers["Accept"] == "application/json"
        assert timeout_seconds == subject.REQUEST_TIMEOUT_SECONDS
        self.calls.append({"method": method, "url": url})
        if self.fault == "timeout" and len(self.calls) == 1:
            raise TimeoutError("fake timeout containing no token")
        if self.fault == "403" and len(self.calls) == 1:
            return self._response(
                403,
                {"error": {"code": 403, "status": "PERMISSION_DENIED"}},
            )
        if self.fault == "bad_json" and len(self.calls) == 1:
            return subject.HttpResponse(
                status_code=200,
                body=b"{not-json",
                headers={"Content-Type": "application/json"},
            )

        parts = urlsplit(url)
        path = unquote(parts.path)
        query = parse_qs(parts.query)
        host = parts.hostname
        project = adapter.DEFAULT_PROJECT

        if host == "cloudresourcemanager.googleapis.com":
            return self._response(
                200,
                {
                    "projectId": project,
                    "projectNumber": PROJECT_NUMBER,
                    "lifecycleState": "ACTIVE",
                },
            )
        if (
            host == "compute.googleapis.com"
            and path == f"/compute/v1/projects/{project}"
        ):
            return self._response(
                200,
                {
                    "name": project,
                    "id": "999999999",
                    # Global allocation quotas intentionally no longer appear
                    # here; the production collector must use Cloud Quotas.
                    "quotas": [],
                },
            )
        if host == "cloudquotas.googleapis.com":
            quota_id = path.rsplit("/", 1)[-1]
            assert quota_id in {
                subject.GLOBAL_CPU_QUOTA_ID,
                subject.C4_CPU_QUOTA_ID,
            }
            if quota_id == subject.C4_CPU_QUOTA_ID:
                return self._response(
                    200,
                    {
                        "name": (
                            f"projects/{PROJECT_NUMBER}/locations/global/"
                            "services/compute.googleapis.com/quotaInfos/"
                            f"{subject.C4_CPU_QUOTA_ID}"
                        ),
                        "quotaId": subject.C4_CPU_QUOTA_ID,
                        "metric": subject.C4_CPU_QUOTA_METRIC,
                        "service": "compute.googleapis.com",
                        "isPrecise": True,
                        "containerType": "PROJECT",
                        "metricUnit": "1",
                        "dimensions": ["region", "vm_family"],
                        "dimensionsInfos": [
                            {
                                "dimensions": {
                                    "region": subject.DEFAULT_REGION,
                                    "vm_family": (
                                        subject.C4_CPU_QUOTA_VM_FAMILY
                                    ),
                                },
                                "details": {
                                    "value": str(self.c4_cpu_limit)
                                },
                                "applicableLocations": [
                                    subject.DEFAULT_REGION
                                ],
                            }
                        ],
                    },
                )
            return self._response(
                200,
                {
                    "name": (
                        f"projects/{PROJECT_NUMBER}/locations/global/services/"
                        "compute.googleapis.com/quotaInfos/"
                        f"{subject.GLOBAL_CPU_QUOTA_ID}"
                    ),
                    "quotaId": subject.GLOBAL_CPU_QUOTA_ID,
                    "metric": subject.GLOBAL_CPU_QUOTA_METRIC,
                    "service": "compute.googleapis.com",
                    "isPrecise": True,
                    "containerType": "PROJECT",
                    "metricUnit": "1",
                    "dimensionsInfos": [
                        {
                            "details": {"value": str(self.global_cpu_limit)},
                            "applicableLocations": ["global"],
                        }
                    ],
                },
            )
        if (
            host == "compute.googleapis.com"
            and f"/compute/v1/projects/{project}/aggregated/" in path
        ):
            resource_collection = path.rsplit("/", 1)[-1]
            assert resource_collection in set(
                subject.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS.values()
            )
            assert query == {
                "maxResults": [
                    str(subject.COMPUTE_AGGREGATED_MAX_RESULTS)
                ],
                "returnPartialSuccess": ["false"],
                "includeAllScopes": ["true"],
            }
            scopes = [
                "global",
                f"regions/{subject.DEFAULT_REGION}",
                f"zones/{adapter.DEFAULT_ZONE}",
            ]
            items: dict[str, Any] = {
                scope: {
                    "warning": {
                        "code": "NO_RESULTS_ON_PAGE",
                        "message": f"No results for the scope '{scope}'.",
                        "data": [{"key": "scope", "value": scope}],
                    }
                }
                for scope in scopes
            }
            if (
                self.inventory_fault == "nonempty"
                and resource_collection == "instances"
            ):
                items[scopes[-1]] = {"instances": [{"name": "unexpected-vm"}]}
            elif (
                self.standard_spot_instance_count
                and resource_collection == "instances"
            ):
                zone = adapter.DEFAULT_ZONE
                zone_link = (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{project}/zones/{zone}"
                )
                instances = [
                    {
                        "kind": "compute#instance",
                        "id": str(1000 + index),
                        "name": f"known-spot-{index:02d}",
                        "zone": zone_link,
                        "machineType": (
                            f"{zone_link}/machineTypes/"
                            f"{self.standard_spot_machine_type}"
                        ),
                        "status": self.standard_instance_status,
                        "selfLink": (
                            f"{zone_link}/instances/"
                            f"known-spot-{index:02d}"
                        ),
                        "scheduling": {
                            "preemptible": True,
                            "provisioningModel": "SPOT",
                        },
                    }
                    for index in range(
                        self.standard_spot_instance_count
                    )
                ]
                if self.instance_record_fault == "reservation":
                    instances[0]["reservationConsumptionInfo"] = {
                        "consumptionType": "SPECIFIC_RESERVATION"
                    }
                elif self.instance_record_fault == "wrong_machine_project":
                    instances[0]["machineType"] = instances[0][
                        "machineType"
                    ].replace(project, "other-project")
                elif self.instance_record_fault == "duplicate":
                    instances.append(deepcopy(instances[0]))
                items[f"zones/{zone}"] = {"instances": instances}
            elif (
                self.inventory_fault == "unknown_warning"
                and resource_collection == "instances"
            ):
                items[scopes[-1]]["warning"]["code"] = "UNKNOWN"
            endpoint_id = next(
                key
                for key, value in (
                    subject.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS.items()
                )
                if value == resource_collection
            )
            payload: dict[str, Any] = {
                "kind": subject.GLOBAL_CPU_EMPTY_INVENTORY_KINDS[endpoint_id],
                "id": f"projects/{project}/aggregated/{resource_collection}",
                "items": items,
                "selfLink": (
                    f"https://www.googleapis.com/compute/v1/projects/"
                    f"{project}/aggregated/{resource_collection}"
                ),
            }
            if resource_collection == "futureReservations":
                payload["etag"] = "fake-nonempty-etag"
            if (
                self.inventory_fault == "unreachable"
                and resource_collection == "instances"
            ):
                payload["unreachables"] = [adapter.DEFAULT_ZONE]
            if (
                self.inventory_fault == "pagination"
                and resource_collection == "instances"
            ):
                payload["nextPageToken"] = "unexpected-second-page"
            return self._response(200, payload)
        if host == "storage.googleapis.com" and path.endswith(
            f"/b/{adapter.DEFAULT_BUCKET}"
        ):
            return self._response(
                200,
                {
                    "name": adapter.DEFAULT_BUCKET,
                    "projectNumber": PROJECT_NUMBER,
                    "location": "ASIA-NORTHEAST1",
                    "locationType": "region",
                    "storageClass": "STANDARD",
                    "metageneration": "7",
                    "iamConfiguration": {
                        "uniformBucketLevelAccess": {"enabled": True}
                    },
                },
            )
        if host == "storage.googleapis.com" and path.endswith(
            f"/b/{adapter.DEFAULT_BUCKET}/iam"
        ):
            assert query == {"optionsRequestedPolicyVersion": ["3"]}
            return self._response(
                200,
                {
                    "version": 3,
                    "etag": "unit-etag",
                    "bindings": self._bucket_iam_bindings(),
                },
            )
        if host == "iam.googleapis.com":
            email = transport_contract.WORKER_SERVICE_ACCOUNT
            assert path.endswith(f"/serviceAccounts/{email}")
            if self.service_account_status != 200:
                return self._response(
                    self.service_account_status,
                    {
                        "error": {
                            "code": self.service_account_status,
                            "status": "NOT_FOUND",
                        }
                    },
                )
            return self._response(
                200,
                {
                    "name": f"projects/{project}/serviceAccounts/{email}",
                    "projectId": project,
                    "uniqueId": "987654321",
                    "email": email,
                    "oauth2ClientId": "987654321",
                },
            )
        if host == "compute.googleapis.com" and "/global/images/" in path:
            image: dict[str, Any] = {
                "name": subject.IMAGE_NAME,
                "id": self.image_id,
                "status": "READY",
                "selfLink": transport_contract.IMAGE_SELF_LINK,
                "architecture": "X86_64",
            }
            if self.image_deprecated:
                image["deprecated"] = {
                    "state": "DEPRECATED",
                    "replacement": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{subject.IMAGE_PROJECT}/global/images/"
                        "debian-12-bookworm-v-next"
                    ),
                }
            return self._response(
                200,
                image,
            )
        if host == "compute.googleapis.com" and "/machineTypes/" in path:
            return self._response(
                200,
                {
                    "name": subject.MACHINE_TYPE,
                    "guestCpus": 16,
                    "memoryMb": 61440,
                    "zone": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{project}/zones/{adapter.DEFAULT_ZONE}"
                    ),
                    "selfLink": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{project}/zones/{adapter.DEFAULT_ZONE}/machineTypes/"
                        f"{subject.MACHINE_TYPE}"
                    ),
                },
            )
        if host == "compute.googleapis.com" and "/regions/" in path:
            return self._response(
                200,
                {
                    "name": subject.DEFAULT_REGION,
                    "selfLink": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{project}/regions/{subject.DEFAULT_REGION}"
                    ),
                    "quotas": [
                        {
                            "metric": "PREEMPTIBLE_CPUS",
                            "limit": 500,
                            "usage": self.regional_spot_usage,
                        },
                    ],
                },
            )
        if host == "compute.googleapis.com" and "/instances/" in path:
            if self.instance_status == 404:
                return self._response(
                    404,
                    {
                        "error": {
                            "code": 404,
                            "status": "NOT_FOUND",
                            "message": "not found",
                        }
                    },
                )
            return self._response(
                200, {"name": path.rsplit("/", 1)[-1], "status": "RUNNING"}
            )
        if host == "storage.googleapis.com" and path.endswith(
            f"/b/{adapter.DEFAULT_BUCKET}/o"
        ):
            prefix = query["prefix"][0]
            token = query.get("pageToken", [None])[0]
            if self.fault == "incomplete_pagination":
                next_value = f"never-terminal-{len(self.calls)}"
                return self._response(
                    200, {"items": [], "nextPageToken": next_value}
                )
            if prefix == self.package_prefix:
                if token is None:
                    midpoint = max(1, len(self.package_items) // 2)
                    return self._response(
                        200,
                        {
                            "items": self.package_items[:midpoint],
                            "nextPageToken": "package-page-2",
                        },
                    )
                assert token == "package-page-2"
                return self._response(
                    200, {"items": self.package_items[len(self.package_items) // 2 :]}
                )
            assert prefix == self.stage_prefix
            if token is None:
                return self._response(
                    200, {"items": [], "nextPageToken": "stage-page-2"}
                )
            assert token == "stage-page-2"
            return self._response(200, {"items": []})
        if host == "cloudbilling.googleapis.com":
            token = query.get("pageToken", [None])[0]
            region = (
                "us-central1" if self.wrong_region else subject.DEFAULT_REGION
            )
            cpu = _sku(group="CPU", sku_id="cpu-1", region=region)
            ram = _sku(group="RAM", sku_id="ram-1", region=region)
            if token is None:
                skus = [cpu]
                if self.ambiguous_cpu:
                    duplicate = deepcopy(cpu)
                    duplicate["name"] = (
                        f"services/{subject.COMPUTE_ENGINE_SERVICE_ID}/skus/cpu-2"
                    )
                    duplicate["skuId"] = "cpu-2"
                    skus.append(duplicate)
                return self._response(
                    200, {"skus": skus, "nextPageToken": "sku-page-2"}
                )
            assert token == "sku-page-2"
            return self._response(
                200,
                {"skus": [ram], "nextPageToken": ""},
            )
        raise AssertionError(f"unexpected fake endpoint: {url}")


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("10c2-real-readonly-v1") / "package"
    cloud_package.build_package(output_dir=target)
    return target


@pytest.fixture
def preflight(
    package: Path,
) -> PreflightFixture:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=diagnostic_plan.STAGE1_ID,
    )
    outer = receiver.build_outer_package_manifest(
        package_dir=package,
        offline_wheel_record={
            "path": (
                f"wheels/{transport_contract.EXPECTED_NUMPY_WHEEL_FILENAME}"
            ),
            "uri_suffix": (
                f"wheels/{transport_contract.EXPECTED_NUMPY_WHEEL_FILENAME}"
            ),
            "sha256": transport_contract.EXPECTED_NUMPY_WHEEL_SHA256,
            "bytes": transport_contract.EXPECTED_NUMPY_WHEEL_BYTES,
            "mode": "0644",
            "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
        },
    )
    direct = receiver.build_direct_stage_identity(
        preview,
        outer_package_manifest=outer,
    )
    plan = receiver.build_read_only_preflight_plan(
        preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
    )
    return preview, plan, outer, direct


def _collect(
    preflight: PreflightFixture,
    **transport_kwargs: Any,
) -> tuple[dict[str, Any], FakeGoogleJsonTransport]:
    preview, plan, outer, direct = preflight
    transport = FakeGoogleJsonTransport(
        plan=plan, preview=preview, **transport_kwargs
    )
    bundle = subject.collect_read_only_observations(
        plan=plan,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        transport=transport,
        token_source=FakeTokenSource(),
        retrieved_at_unix_seconds=NOW,
    )
    return bundle, transport


def _collect_through_concrete_stdlib_type(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake: FakeGoogleJsonTransport | None = None,
) -> tuple[dict[str, Any], FakeGoogleJsonTransport]:
    preview, plan, outer, direct = preflight
    delegate = fake or FakeGoogleJsonTransport(plan=plan, preview=preview)

    def request(
        _self: subject.StdlibGoogleJsonReadOnlyTransport,
        **kwargs: Any,
    ) -> subject.HttpResponse:
        return delegate.request(**kwargs)

    monkeypatch.setattr(
        subject.StdlibGoogleJsonReadOnlyTransport,
        "request",
        request,
    )
    concrete = subject.StdlibGoogleJsonReadOnlyTransport()
    bundle = subject.collect_read_only_observations(
        plan=plan,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        transport=concrete,
        token_source=FakeTokenSource(),
        retrieved_at_unix_seconds=NOW,
    )
    return bundle, delegate


def test_full_paginated_bundle_is_hash_bound_secret_free_and_receiver_consumable(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight
    bundle, transport = _collect(preflight)
    assert bundle["page_tokens_exhausted"] is True
    assert bundle["facts"]["machine"]["guest_cpus"] == 16
    assert bundle["facts"]["machine"]["memory_gib"] == "60"
    assert bundle["facts"]["regional_capacity"]["available_vcpu"] == 355
    c4 = bundle["facts"]["regional_capacity"]["quota_metrics"]["C4_CPUS"]
    assert c4["quota_id"] == subject.C4_CPU_QUOTA_ID
    assert c4["cloud_quotas_metric"] == subject.C4_CPU_QUOTA_METRIC
    assert c4["dimensions"] == {
        "region": subject.DEFAULT_REGION,
        "vm_family": "C4",
    }
    assert (
        bundle["facts"]["spot_price"]["hourly_price_usd_decimal"] == "0.220"
    )
    assert (
        bundle["receiver_observations"]["prefix"]["package_inventory_state"]
        == "exact_complete_immutable_reuse"
    )
    assert len(bundle["facts"]["package_inventory"]) == len(
        plan["requirements"]["prefix"]["expected_package_objects"]
    )
    assert all(
        type(row["generation"]) is int
        for row in bundle["facts"]["package_inventory"]
    )
    assert len(transport.calls) >= 14
    rendered = json.dumps(bundle, sort_keys=True)
    assert SECRET not in rendered
    assert "package-page-2" not in rendered
    assert "stage-page-2" not in rendered
    assert "sku-page-2" not in rendered
    assert bundle["launch_authorized"] is False
    assert bundle["launch_ready"] is False
    assert bundle["cloud_mutation_performed"] is False
    bucket_iam = bundle["facts"]["bucket_and_iam"]
    assert bucket_iam["required_worker_storage_permissions"] == [
        "storage.objects.get",
        "storage.objects.create",
    ]
    assert {
        row["role"]
        for row in bucket_iam["qualifying_conditioned_bindings"]
    } == {
        subject.STORAGE_OBJECT_VIEWER_ROLE,
        subject.STORAGE_OBJECT_CREATOR_ROLE,
    }
    assert "roles/storage.objectAdmin" not in bucket_iam[
        "service_account_roles"
    ]
    validated = subject.validate_observation_bundle(
        bundle,
        plan=plan,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        evaluation_unix_seconds=NOW + 1,
    )
    assert validated == bundle
    receiver_result = receiver.evaluate_read_only_preflight(
        plan,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=bundle["receiver_observations"]["prefix"],
        instance_observation=bundle["receiver_observations"]["instances"],
        capacity_observation=bundle["receiver_observations"]["capacity"],
        price_observation=bundle["receiver_observations"]["spot_price"],
        evaluation_unix_seconds=NOW + 1,
    )
    assert receiver_result["preflight_passed"] is False
    assert receiver_result["launch_authorized"] is False
    assert "prefix_external_observations_missing" in receiver_result["failures"]


def test_caller_provenance_booleans_cannot_promote_injected_transport(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight

    class SpoofedExternalTransport(FakeGoogleJsonTransport):
        fixture_only = False
        external_cloud_read_performed = True

    spoofed = SpoofedExternalTransport(plan=plan, preview=preview)
    bundle = subject.collect_read_only_observations(
        plan=plan,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        transport=spoofed,
        token_source=FakeTokenSource(),
        retrieved_at_unix_seconds=NOW,
    )
    assert bundle["external_cloud_read_performed"] is False
    assert bundle["concrete_stdlib_transport_used"] is False
    assert bundle["read_only_observation_passed"] is False
    assert bundle["receiver_observations"]["prefix"]["query_performed"] is False


def test_concrete_stdlib_get_only_observation_passes_separate_gate(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, delegate = _collect_through_concrete_stdlib_type(
        preflight, monkeypatch
    )
    assert bundle["concrete_stdlib_transport_used"] is True
    assert bundle["read_only_observation_passed"] is True
    assert bundle["status"] == (
        "read_only_observation_passed_launch_still_unauthorized"
    )
    assert bundle["launch_authorized"] is False
    assert bundle["launch_ready"] is False
    assert bundle["launch_permission_readiness"][
        "compute_instances_delete_permission_verified"
    ] is False
    assert bundle["launch_permission_readiness"][
        "worker_oauth_scope_sufficiency_verified"
    ] is False
    assert {row["method"] for row in delegate.calls} == {"GET"}


def test_both_attempt_instance_names_are_checked_by_get404(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, _plan, _outer, direct = preflight
    bundle, delegate = _collect_through_concrete_stdlib_type(
        preflight, monkeypatch
    )
    expected = [
        row["instance_name"] for row in direct["inputs"]["attempt_layout"]
    ]
    assert {
        row["instance_name"]
        for row in bundle["facts"]["expected_instance_absence"]
    } == set(expected)
    instance_calls = [
        unquote(urlsplit(row["url"]).path).rsplit("/", 1)[-1]
        for row in delegate.calls
        if "/instances/" in urlsplit(row["url"]).path
    ]
    assert instance_calls == expected
    assert {
        row["attempt_index"] for row in direct["inputs"]["attempt_layout"]
    } == {0, 1}
    assert len(expected) == 2 * len(preview["jobs"])


def test_sparse_package_subset_is_normalized_to_manifest_order(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(plan=plan, preview=preview)
    chosen = list(reversed(fake.package_items[::2]))
    fake.package_items = chosen
    bundle, _ = _collect_through_concrete_stdlib_type(
        preflight, monkeypatch, fake=fake
    )
    prefix = bundle["receiver_observations"]["prefix"]
    expected = plan["requirements"]["prefix"]["expected_package_objects"]
    chosen_uris = {
        f"gs://{adapter.DEFAULT_BUCKET}/{row['name']}" for row in chosen
    }
    assert prefix["package_inventory_state"] == (
        "exact_expected_subset_provisioning_required"
    )
    assert [
        row["uri"] for row in prefix["package_object_identities"]
    ] == [row["uri"] for row in expected if row["uri"] in chosen_uris]
    assert bundle["read_only_observation_passed"] is True


def test_gcs_package_unknown_or_content_mismatch_is_rejected(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    mismatched = FakeGoogleJsonTransport(plan=plan, preview=preview)
    mismatched.package_items[0]["metadata"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="mismatched object"):
        _collect_through_concrete_stdlib_type(
            preflight, monkeypatch, fake=mismatched
        )


def test_global_cpu_quota_is_part_of_capacity_minimum(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        global_cpu_limit=10,
    )
    bundle, _ = _collect_through_concrete_stdlib_type(
        preflight, monkeypatch, fake=fake
    )
    capacity = bundle["facts"]["regional_capacity"]
    assert capacity["available_vcpu"] == 10
    assert capacity["project_global_quota_metric"]["quota_id"] == (
        subject.GLOBAL_CPU_QUOTA_ID
    )
    assert capacity["project_global_quota_metric"]["metric"] == (
        "compute.googleapis.com/cpus_all_regions"
    )
    assert capacity["project_global_quota_metric"]["usage"] == "0"
    assert len(
        capacity["project_global_quota_metric"]["usage_inventory_proof"]
    ) == 4
    assert (
        capacity["provider_metric_mapping"]
        == receiver.CAPACITY_PROVIDER_METRIC_MAPPING
    )
    assert bundle["receiver_contract_result"]["observation_contract_passed"] is False
    assert "launch_capacity_insufficient" in bundle[
        "receiver_contract_result"
    ]["failures"]
    assert bundle["read_only_observation_passed"] is False


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("nonempty", "instance identity changed"),
        ("unknown_warning", "empty-scope warning changed"),
        ("unreachable", "unreachable scopes"),
    ],
)
def test_global_cpu_inventory_rejects_malformed_or_incomplete_responses(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    message: str,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        inventory_fault=fault,
    )
    with pytest.raises(ValueError, match=message):
        _collect_through_concrete_stdlib_type(
            preflight,
            monkeypatch,
            fake=fake,
        )


def test_global_cpu_inventory_pagination_fails_closed(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        inventory_fault="pagination",
    )
    with pytest.raises(
        subject.ReadOnlyPreflightError,
        match="unexpected_pagination_on_single_resource",
    ):
        _collect_through_concrete_stdlib_type(
            preflight,
            monkeypatch,
            fake=fake,
        )


def test_regional_spot_usage_cannot_be_below_observed_spot_instances(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        regional_spot_usage=0,
        standard_spot_instance_count=1,
    )
    with pytest.raises(ValueError, match="below observed Spot instances"):
        _collect_through_concrete_stdlib_type(
            preflight,
            monkeypatch,
            fake=fake,
        )


def test_nonempty_standard_instances_reduce_global_headroom_conservatively(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        global_cpu_limit=400,
        regional_spot_usage=32,
        standard_spot_instance_count=2,
    )
    bundle, delegate = _collect_through_concrete_stdlib_type(
        preflight,
        monkeypatch,
        fake=fake,
    )
    global_quota = bundle["facts"]["regional_capacity"][
        "project_global_quota_metric"
    ]
    assert global_quota["usage"] == "32"
    assert global_quota["available"] == "368"
    assert global_quota["usage_kind"] == "conservative_upper_bound"
    assert global_quota["available_kind"] == "conservative_lower_bound"
    instance_proof = next(
        row
        for row in global_quota["usage_inventory_proof"]
        if row["resource_collection"] == "instances"
    )
    assert instance_proof["resource_count"] == 2
    assert instance_proof["global_vcpu_usage_upper_bound"] == 32
    assert instance_proof["target_region_spot_vcpu_observed"] == 32
    assert "resource_identities" not in instance_proof
    machine_type_calls = [
        row["url"]
        for row in delegate.calls
        if "/machineTypes/" in urlsplit(row["url"]).path
    ]
    assert len(machine_type_calls) == 1
    assert machine_type_calls[0].endswith(
        f"/machineTypes/{subject.MACHINE_TYPE}"
    )
    assert all("n2-standard-16" not in url for url in machine_type_calls)
    assert bundle["read_only_observation_passed"] is True


def test_target_region_c4_instances_reduce_c4_headroom(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        c4_cpu_limit=24,
        regional_spot_usage=16,
        standard_spot_instance_count=1,
        standard_spot_machine_type="c4-standard-16",
    )
    bundle, _ = _collect_through_concrete_stdlib_type(
        preflight,
        monkeypatch,
        fake=fake,
    )
    c4 = bundle["facts"]["regional_capacity"]["quota_metrics"]["C4_CPUS"]
    assert c4["usage"] == "16"
    assert c4["available"] == "8"
    assert bundle["facts"]["regional_capacity"]["available_vcpu"] == 8
    assert bundle["read_only_observation_passed"] is False
    assert "launch_capacity_insufficient" in bundle[
        "receiver_contract_result"
    ]["failures"]


@pytest.mark.parametrize(
    "machine_type",
    [
        "e2-micro",
        "n2-highcpu-16",
        "n2-highmem-16",
        "custom-16-65536",
        "c3d-standard-180-lssd",
        "n2-standard-016",
    ],
)
def test_instance_machine_type_requires_unambiguous_standard_vcpu_suffix(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
    machine_type: str,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        regional_spot_usage=16,
        standard_spot_instance_count=1,
        standard_spot_machine_type=machine_type,
    )
    with pytest.raises(ValueError, match="cannot be conservatively sized"):
        _collect_through_concrete_stdlib_type(
            preflight,
            monkeypatch,
            fake=fake,
        )


@pytest.mark.parametrize(
    ("status", "expected_usage"),
    [
        ("RUNNING", "16"),
        ("TERMINATED", "16"),
        ("SUSPENDED", "16"),
    ],
)
def test_all_known_instance_statuses_are_conservatively_counted(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_usage: str,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        regional_spot_usage=16,
        standard_spot_instance_count=1,
        standard_instance_status=status,
    )
    bundle, _ = _collect_through_concrete_stdlib_type(
        preflight,
        monkeypatch,
        fake=fake,
    )
    assert bundle["facts"]["regional_capacity"][
        "project_global_quota_metric"
    ]["usage"] == expected_usage


@pytest.mark.parametrize(
    ("status", "fault", "message"),
    [
        ("UNKNOWN", None, "instance identity changed"),
        ("RUNNING", "reservation", "instance identity changed"),
        ("RUNNING", "wrong_machine_project", "machine type changed"),
        ("RUNNING", "duplicate", "instance is duplicated"),
    ],
)
def test_ambiguous_or_duplicate_instance_inventory_fails_closed(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    fault: str | None,
    message: str,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        regional_spot_usage=16,
        standard_spot_instance_count=1,
        standard_instance_status=status,
        instance_record_fault=fault,
    )
    with pytest.raises(ValueError, match=message):
        _collect_through_concrete_stdlib_type(
            preflight,
            monkeypatch,
            fake=fake,
        )


@pytest.mark.parametrize("invalid_limit", [0, -1, 1.5, "1e3"])
def test_cloud_quota_values_require_positive_string_int64(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
    invalid_limit: Any,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        global_cpu_limit=invalid_limit,
    )
    with pytest.raises(ValueError, match="positive string int64"):
        _collect_through_concrete_stdlib_type(
            preflight,
            monkeypatch,
            fake=fake,
        )


def test_exact_spot_price_uses_cpu_and_ram_preemptible_skus(
    preflight: PreflightFixture,
) -> None:
    bundle, _ = _collect(preflight)
    price = bundle["facts"]["spot_price"]
    assert price["cpu_sku"]["resource_group"] == "CPU"
    assert price["cpu_sku"]["usage_type"] == "Preemptible"
    assert price["ram_sku"]["resource_group"] == "RAM"
    assert price["ram_sku"]["service_region"] == "asia-northeast1"
    assert price["hourly_price_usd_per_vm"] == pytest.approx(0.22)
    assert (
        price["official_source_url"]
        == "https://cloud.google.com/spot-vms/pricing"
    )
    assert price["catalog_source_url"].startswith(
        "https://cloudbilling.googleapis.com/v1/services/"
    )


def test_billing_time_series_selects_latest_applicable_price_per_sku(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight
    target = subject._target_from_plan(
        plan,
        preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
    )
    cpu = _sku(group="CPU", sku_id="cpu-history")
    ram = _sku(group="RAM", sku_id="ram-history")

    def history(
        sku: dict[str, Any],
        *,
        current_time: str,
        current_nanos: int,
        old_nanos: int,
    ) -> None:
        current = deepcopy(sku["pricingInfo"][0])
        current["effectiveTime"] = current_time
        current["pricingExpression"]["tieredRates"][0]["unitPrice"] = _money(
            current_nanos
        )
        old = deepcopy(current)
        old["effectiveTime"] = "2035-01-01T00:00:00Z"
        old["pricingExpression"]["tieredRates"][0]["unitPrice"] = _money(
            old_nanos
        )
        future = deepcopy(current)
        future["effectiveTime"] = "2037-01-01T00:00:00Z"
        future["pricingExpression"]["tieredRates"][0]["unitPrice"] = _money(
            999_000_000
        )
        sku["pricingInfo"] = [future, old, current]

    history(
        cpu,
        current_time="2036-07-17T00:00:00Z",
        current_nanos=10_000_000,
        old_nanos=5_000_000,
    )
    history(
        ram,
        current_time="2036-07-16T00:00:00Z",
        current_nanos=1_000_000,
        old_nanos=500_000,
    )
    price = subject._price_facts(
        [cpu, ram],
        target=target,
        machine={"guest_cpus": 16, "memory_mb": 61440},
        retrieved_at_unix_seconds=NOW,
    )
    assert price["hourly_price_usd_decimal"] == "0.220"
    assert price["cpu_sku"]["effective_at_unix_seconds"] != price["ram_sku"][
        "effective_at_unix_seconds"
    ]
    assert price["sku_effective_at_unix_seconds"] == max(
        price["cpu_sku"]["effective_at_unix_seconds"],
        price["ram_sku"]["effective_at_unix_seconds"],
    )
    assert price["official_source_url"] == subject.OFFICIAL_PRICE_SOURCE_URL


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    [
        ("403", "readonly_https_unexpected_status"),
        ("timeout", "readonly_https_transport_failure"),
        ("bad_json", "readonly_https_invalid_json"),
    ],
)
def test_http_failures_are_closed_with_secret_free_receipts(
    preflight: PreflightFixture,
    fault: str,
    expected_code: str,
) -> None:
    preview, plan, outer, direct = preflight
    transport = FakeGoogleJsonTransport(
        plan=plan, preview=preview, fault=fault
    )
    with pytest.raises(subject.ReadOnlyPreflightError) as exc_info:
        subject.collect_read_only_observations(
            plan=plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            transport=transport,
            token_source=FakeTokenSource(),
            retrieved_at_unix_seconds=NOW,
        )
    assert exc_info.value.code == expected_code
    rendered = json.dumps(exc_info.value.evidence, sort_keys=True)
    assert SECRET not in rendered
    assert exc_info.value.evidence["launch_authorized"] is False
    receipt = exc_info.value.evidence["receipts"][-1]
    assert receipt["method"] == "GET"
    assert receipt["authorization_header_recorded"] is False
    assert receipt["access_token_recorded"] is False


def test_incomplete_pagination_is_fatal(
    preflight: PreflightFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, outer, direct = preflight
    monkeypatch.setattr(subject, "MAX_PAGES", 2)
    transport = FakeGoogleJsonTransport(
        plan=plan, preview=preview, fault="incomplete_pagination"
    )
    with pytest.raises(
        subject.ReadOnlyPreflightError,
        match="readonly_https_pagination_not_exhausted",
    ) as exc_info:
        subject.collect_read_only_observations(
            plan=plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            transport=transport,
            token_source=FakeTokenSource(),
            retrieved_at_unix_seconds=NOW,
        )
    rendered = json.dumps(exc_info.value.evidence, sort_keys=True)
    assert "never-terminal-" not in rendered
    assert exc_info.value.evidence["launch_ready"] is False


def test_ambiguous_cpu_sku_fails_closed(
    preflight: PreflightFixture,
) -> None:
    with pytest.raises(ValueError, match="CPU/RAM SKUs are ambiguous"):
        _collect(preflight, ambiguous_cpu=True)


def test_wrong_billing_region_fails_closed(
    preflight: PreflightFixture,
) -> None:
    with pytest.raises(ValueError, match="CPU/RAM SKUs are ambiguous"):
        _collect(preflight, wrong_region=True)


def test_expected_instance_must_have_get404_evidence(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight
    transport = FakeGoogleJsonTransport(
        plan=plan, preview=preview, instance_status=200
    )
    with pytest.raises(
        subject.ReadOnlyPreflightError,
        match="readonly_https_unexpected_status",
    ):
        subject.collect_read_only_observations(
            plan=plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            transport=transport,
            token_source=FakeTokenSource(),
            retrieved_at_unix_seconds=NOW,
        )


def test_dedicated_service_account_is_exact_and_absence_is_no_go(
    preflight: PreflightFixture,
) -> None:
    bundle, transport = _collect(preflight)
    iam_calls = [
        row["url"]
        for row in transport.calls
        if urlsplit(row["url"]).hostname == "iam.googleapis.com"
    ]
    assert len(iam_calls) == 1
    assert (
        transport_contract.WORKER_SERVICE_ACCOUNT.replace("@", "%40")
        in iam_calls[0]
    )
    assert f"{PROJECT_NUMBER}-compute" not in iam_calls[0]
    assert (
        bundle["facts"]["service_account"]["email"]
        == transport_contract.WORKER_SERVICE_ACCOUNT
    )
    preview, plan, outer, direct = preflight
    absent = FakeGoogleJsonTransport(
        plan=plan,
        preview=preview,
        service_account_status=404,
    )
    with pytest.raises(
        subject.ReadOnlyPreflightError,
        match="readonly_https_unexpected_status",
    ):
        subject.collect_read_only_observations(
            plan=plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            transport=absent,
            token_source=FakeTokenSource(),
            retrieved_at_unix_seconds=NOW,
        )


@pytest.mark.parametrize(
    "bucket_iam_mode",
    [
        "missing",
        "unconditional_viewer",
        "wrong_condition",
        "broad_condition",
        "object_admin_only",
    ],
)
def test_non_least_privilege_bucket_iam_fails_closed(
    preflight: PreflightFixture,
    bucket_iam_mode: str,
) -> None:
    with pytest.raises(ValueError, match="exact conditioned least-privilege"):
        _collect(preflight, bucket_iam_mode=bucket_iam_mode)


def test_wrong_image_id_fails_closed(
    preflight: PreflightFixture,
) -> None:
    with pytest.raises(ValueError, match="pinned Compute image"):
        _collect(preflight, image_id="999")


def test_exact_pinned_deprecated_ready_image_remains_observable(
    preflight: PreflightFixture,
) -> None:
    bundle, _ = _collect(preflight, image_deprecated=True)
    image = bundle["facts"]["image"]
    assert image["deprecated"] is True
    assert image["deprecation_state"] == "DEPRECATED"
    assert image["launch_warning_expected"] is True
    assert image["replacement"].endswith("/debian-12-bookworm-v-next")
    assert bundle["launch_authorized"] is False
    assert bundle["launch_ready"] is False


def test_stale_bundle_is_rejected(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight
    bundle, _ = _collect(preflight)
    with pytest.raises(ValueError, match="stale or from the future"):
        subject.validate_observation_bundle(
            bundle,
            plan=plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            evaluation_unix_seconds=NOW
            + receiver.PREFIX_OBSERVATION_MAX_AGE_SECONDS
            + 1,
        )


def test_actual_entrypoint_writes_secret_free_exclusive_no_go_receipt(
    preflight: PreflightFixture,
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    transport = FakeGoogleJsonTransport(plan=plan, preview=preview)
    expected_wheel = {
        "path": f"wheels/{transport_contract.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "uri_suffix": (
            f"wheels/{transport_contract.EXPECTED_NUMPY_WHEEL_FILENAME}"
        ),
        "sha256": transport_contract.EXPECTED_NUMPY_WHEEL_SHA256,
        "bytes": transport_contract.EXPECTED_NUMPY_WHEEL_BYTES,
        "mode": "0644",
        "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
    }
    monkeypatch.setattr(
        subject.transport_contract,
        "build_offline_wheel_record",
        lambda _path: expected_wheel,
    )
    output = tmp_path / "readonly-receipt.json"
    artifact = subject.run_actual_read_only_preflight(
        package_dir=package,
        pinned_wheel_path=tmp_path / "not-read-by-fake.whl",
        stage_id=diagnostic_plan.STAGE1_ID,
        output_path=output,
        token_source=FakeTokenSource(),
        transport=transport,
        retrieved_at_unix_seconds=NOW,
    )
    assert artifact["status"] == "no_go_read_only_preflight"
    assert artifact["launch_authorized"] is False
    assert artifact["launch_ready"] is False
    assert artifact["collector_gcloud_used"] is False
    assert artifact["collector_subprocess_used"] is False
    assert artifact["credential_provenance"][
        "credential_acquisition_gcloud_used"
    ] is None
    assert artifact["read_only_observation_passed"] is False
    assert transport.calls == []
    raw = output.read_text(encoding="utf-8")
    assert SECRET not in raw
    assert json.loads(raw) == artifact
    with pytest.raises(FileExistsError):
        subject.run_actual_read_only_preflight(
            package_dir=package,
            pinned_wheel_path=tmp_path / "not-read-by-fake.whl",
            stage_id=diagnostic_plan.STAGE1_ID,
            output_path=output,
            token_source=FakeTokenSource(),
            transport=transport,
            retrieved_at_unix_seconds=NOW,
        )


def test_actual_entrypoint_accepts_only_concrete_stdlib_observation_gate(
    preflight: PreflightFixture,
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, plan, _outer, _direct = preflight
    fake = FakeGoogleJsonTransport(plan=plan, preview=preview)

    def request(
        _self: subject.StdlibGoogleJsonReadOnlyTransport,
        **kwargs: Any,
    ) -> subject.HttpResponse:
        return fake.request(**kwargs)

    monkeypatch.setattr(
        subject.StdlibGoogleJsonReadOnlyTransport,
        "request",
        request,
    )
    monkeypatch.setattr(
        subject.transport_contract,
        "build_offline_wheel_record",
        lambda _path: {
            "path": (
                f"wheels/{transport_contract.EXPECTED_NUMPY_WHEEL_FILENAME}"
            ),
            "uri_suffix": (
                f"wheels/{transport_contract.EXPECTED_NUMPY_WHEEL_FILENAME}"
            ),
            "sha256": transport_contract.EXPECTED_NUMPY_WHEEL_SHA256,
            "bytes": transport_contract.EXPECTED_NUMPY_WHEEL_BYTES,
            "mode": "0644",
            "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
        },
    )
    output = tmp_path / "concrete-readonly-receipt.json"
    artifact = subject.run_actual_read_only_preflight(
        package_dir=package,
        pinned_wheel_path=tmp_path / "synthetic-wheel-record.whl",
        stage_id=diagnostic_plan.STAGE1_ID,
        output_path=output,
        token_source=FakeTokenSource(),
        transport=subject.StdlibGoogleJsonReadOnlyTransport(),
        retrieved_at_unix_seconds=NOW,
    )
    assert artifact["status"] == (
        "pass_read_only_preflight_launch_still_unauthorized"
    )
    assert artifact["read_only_observation_passed"] is True
    assert artifact["launch_permission_ready"] is False
    assert artifact["launch_authorized"] is False
    assert artifact["launch_ready"] is False
    assert artifact["credential_provenance"][
        "credential_acquisition_method"
    ] == "external_unknown"
    assert artifact["credential_provenance"][
        "credential_acquisition_gcloud_used"
    ] is None
    assert SECRET not in output.read_text(encoding="utf-8")


def test_environment_token_source_never_accepts_missing_or_malformed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNIT_GOOGLE_TOKEN", raising=False)
    source = subject.EnvironmentAccessTokenSource("UNIT_GOOGLE_TOKEN")
    with pytest.raises(ValueError, match="environment variable is absent"):
        source.access_token()
    monkeypatch.setenv("UNIT_GOOGLE_TOKEN", "bad token with spaces")
    with pytest.raises(ValueError, match="invalid token"):
        source.access_token()


def test_allowlist_rejects_mutation_and_wrong_host(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight
    target = subject._target_from_plan(
        plan,
        preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
    )
    url = subject._request_url("crm_project", target)
    with pytest.raises(ValueError, match="only HTTPS GET/HEAD"):
        subject.validate_allowed_request(
            method="POST",
            url=url,
            endpoint_id="crm_project",
            target=target,
        )
    with pytest.raises(ValueError, match="allowlist"):
        subject.validate_allowed_request(
            method="GET",
            url=url.replace(
                "cloudresourcemanager.googleapis.com", "example.com"
            ),
            endpoint_id="crm_project",
            target=target,
        )


def test_bundle_hash_tamper_is_rejected(
    preflight: PreflightFixture,
) -> None:
    preview, plan, outer, direct = preflight
    bundle, _ = _collect(preflight)
    changed = deepcopy(bundle)
    changed["facts"]["regional_capacity"]["available_vcpu"] += 1
    with pytest.raises(ValueError, match="bundle hash changed"):
        subject.validate_observation_bundle(
            changed,
            plan=plan,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
            evaluation_unix_seconds=NOW + 1,
        )


def test_concrete_readonly_transport_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject.urllib.request,
        "getproxies",
        lambda: (_ for _ in ()).throw(
            AssertionError("environment proxy discovery is forbidden")
        ),
    )
    transport = subject.StdlibGoogleJsonReadOnlyTransport()
    # ProxyHandler({}) suppresses urllib's default environment-derived handler;
    # with an empty mapping urllib intentionally registers no proxy methods.
    assert not any(
        isinstance(handler, subject.urllib.request.ProxyHandler)
        for handler in transport._opener.handlers
    )
    assert any(
        isinstance(handler, subject._NoRedirectHandler)
        for handler in transport._opener.handlers
    )
