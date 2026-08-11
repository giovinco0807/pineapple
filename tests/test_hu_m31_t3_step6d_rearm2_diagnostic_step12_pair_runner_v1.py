from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_v1.py"
)
INSTANCE_NAMES = [
    "r2d-10c2-s2-candidate-01-a0-29e3c6f8",
    "r2d-10c2-s2-reference-01-a0-29e3c6f8",
]


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_step12_pair_runner_v1", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def local_bundle(
    runner: ModuleType,
) -> tuple[
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    return runner._build_local_bundle(
        issued_at_unix_seconds=1_800_000_000
    )


def _family_quota(*, limit: str = "24") -> dict[str, Any]:
    return {
        "name": (
            "projects/783381566570/locations/global/services/"
            "compute.googleapis.com/quotaInfos/"
            "CPUS-PER-VM-FAMILY-per-project-region"
        ),
        "metric": "compute.googleapis.com/cpus_per_vm_family",
        "quotaId": "CPUS-PER-VM-FAMILY-per-project-region",
        "service": "compute.googleapis.com",
        "isPrecise": True,
        "containerType": "PROJECT",
        "metricUnit": "1",
        "dimensions": ["region", "vm_family"],
        "dimensionsInfos": [
            {
                "dimensions": {
                    "region": "asia-northeast1",
                    "vm_family": "C4",
                },
                "details": {"value": limit},
                "applicableLocations": ["asia-northeast1"],
            }
        ],
    }


def _global_quota(*, limit: str = "64") -> dict[str, Any]:
    return {
        "name": (
            "projects/783381566570/locations/global/services/"
            "compute.googleapis.com/quotaInfos/"
            "CPUS-ALL-REGIONS-per-project"
        ),
        "metric": "compute.googleapis.com/cpus_all_regions",
        "quotaId": "CPUS-ALL-REGIONS-per-project",
        "service": "compute.googleapis.com",
        "isPrecise": True,
        "containerType": "PROJECT",
        "metricUnit": "1",
        "dimensionsInfos": [
            {
                "details": {"value": limit},
                "applicableLocations": ["global"],
            }
        ],
    }


_INVENTORY_COLLECTIONS = {
    "compute_aggregated_instances": (
        "instances",
        "compute#instanceAggregatedList",
    ),
    "compute_aggregated_reservations": (
        "reservations",
        "compute#reservationAggregatedList",
    ),
    "compute_aggregated_node_groups": (
        "nodeGroups",
        "compute#nodeGroupAggregatedList",
    ),
    "compute_aggregated_future_reservations": (
        "futureReservations",
        "compute#futureReservationsAggregatedListResponse",
    ),
}


def _empty_inventory(endpoint_id: str) -> dict[str, Any]:
    collection, kind = _INVENTORY_COLLECTIONS[endpoint_id]
    scopes = [
        "global",
        "regions/asia-northeast1",
        "zones/asia-northeast1-b",
    ]
    value: dict[str, Any] = {
        "kind": kind,
        "id": f"projects/ofc-solver-485418/aggregated/{collection}",
        "items": {
            scope: {
                "warning": {
                    "code": "NO_RESULTS_ON_PAGE",
                    "message": f"No results for scope {scope}",
                    "data": [{"key": "scope", "value": scope}],
                }
            }
            for scope in scopes
        },
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"ofc-solver-485418/aggregated/{collection}"
        ),
    }
    if collection == "futureReservations":
        value["etag"] = "fixture-etag"
    return value


def _capacity_observations(
    *,
    c4_limit: str = "24",
    global_limit: str = "64",
) -> dict[str, Any]:
    return {
        "cloud_quotas_global_cpu": _global_quota(limit=global_limit),
        "cloud_quotas_c4_cpu": _family_quota(limit=c4_limit),
        **{
            endpoint_id: _empty_inventory(endpoint_id)
            for endpoint_id in _INVENTORY_COLLECTIONS
        },
    }


def _instance(
    name: str,
    *,
    status: str = "RUNNING",
    machine_type: str = "c4-standard-8",
) -> dict[str, Any]:
    zone_link = (
        "https://www.googleapis.com/compute/v1/projects/"
        "ofc-solver-485418/zones/asia-northeast1-b"
    )
    return {
        "kind": "compute#instance",
        "id": str(abs(hash((name, status, machine_type))) + 1),
        "name": name,
        "zone": zone_link,
        "machineType": f"{zone_link}/machineTypes/{machine_type}",
        "status": status,
        "selfLink": f"{zone_link}/instances/{name}",
        "scheduling": {
            "preemptible": True,
            "provisioningModel": "SPOT",
        },
    }


class _FakeStep11Runner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _gcloud_json(self, arguments: list[str]) -> Any:
        self.calls.append(arguments)
        if arguments[:3] == ["compute", "machine-types", "describe"]:
            return {"name": "c4-standard-8", "guestCpus": 8}
        if arguments[:3] == ["compute", "regions", "describe"]:
            return {
                "name": "asia-northeast1",
                "status": "UP",
                "quotas": [
                    {"metric": "CPUS", "limit": 355.0, "usage": 0.0},
                    {
                        "metric": "PREEMPTIBLE_CPUS",
                        "limit": 468.0,
                        "usage": 0.0,
                    },
                ],
            }
        raise AssertionError(f"unexpected GET-only gcloud call: {arguments}")


def _gate() -> dict[str, Any]:
    return {
        "instance_contract": {"authorized_instance_names": INSTANCE_NAMES},
        "capacity_contract": {
            "requested_c4_vcpu": 16,
            "quota_metrics": [
                "CPUS",
                "CPUS_ALL_REGIONS",
                "CPUS_PER_VM_FAMILY_C4",
                "PREEMPTIBLE_CPUS",
            ],
            "authoritative_inventory_endpoints": list(
                _INVENTORY_COLLECTIONS
            ),
            "authoritative_inventory_complete_required": True,
            "authoritative_inventory_same_scope_set_required": True,
            "authoritative_inventory_unreachable_count_required": 0,
            "authoritative_inventory_page_tokens_exhausted_required": True,
            "authoritative_noninstance_inventory_empty_required": True,
            "authoritative_transcript_endpoint_digest_map_required": True,
        },
    }


def test_authoritative_capacity_requires_exact_global_and_family_quota(
    runner: ModuleType,
) -> None:
    facts = runner._authoritative_capacity_facts(
        _capacity_observations(), gate=_gate()
    )
    assert facts["global_cpu"]["limit"] == "64"
    assert facts["regional_c4"]["limit"] == "24"
    assert len(facts["inventory_facts"]) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_global_dimensions",
        "duplicate_global_dimensions",
        "wrong_region",
        "wrong_family",
        "wrong_metric",
        "nonpositive_limit",
    ],
)
def test_authoritative_capacity_rejects_ambiguous_or_wrong_quota_rows(
    runner: ModuleType,
    mutation: str,
) -> None:
    observations = _capacity_observations()
    family = observations["cloud_quotas_c4_cpu"]
    row = family["dimensionsInfos"][0]
    if mutation == "missing_global_dimensions":
        observations["cloud_quotas_global_cpu"]["dimensionsInfos"] = []
    elif mutation == "duplicate_global_dimensions":
        global_rows = observations["cloud_quotas_global_cpu"][
            "dimensionsInfos"
        ]
        global_rows.append(copy.deepcopy(global_rows[0]))
    elif mutation == "wrong_region":
        row["dimensions"]["region"] = "us-central1"
    elif mutation == "wrong_family":
        row["dimensions"]["vm_family"] = "N2"
    elif mutation == "wrong_metric":
        family["metric"] = "compute.googleapis.com/cpus"
    elif mutation == "nonpositive_limit":
        row["details"]["value"] = "0"
    with pytest.raises(ValueError):
        runner._authoritative_capacity_facts(
            observations,
            gate=_gate(),
        )


def test_collect_fresh_capacity_accepts_exact_live_schema(
    runner: ModuleType,
) -> None:
    source = _FakeStep11Runner()
    receipt = runner._collect_fresh_capacity(
        source,
        gate=_gate(),
        collected_at_unix_seconds=1_800_000_000,
        authoritative_observations=_capacity_observations(),
    )
    assert receipt["schema"] == runner.CAPACITY_SCHEMA
    assert receipt["requested_c4_vcpu"] == 16
    assert receipt["available_vcpu"] == 24
    assert receipt["quota"]["C4_CPUS_PER_VM_FAMILY"] == {
        "limit": 24,
        "usage": 0,
        "available": 24,
        "limit_source": (
            "cloudquotas.googleapis.com/v1/quotaInfos.get"
        ),
        "usage_source": (
            "compute.googleapis.com/compute/v1/aggregated:"
            "conservative-target-region-c4-instance-vcpu-upper-bound-"
            "with-empty-reservations-nodeGroups-futureReservations"
        ),
    }
    assert receipt["quota"]["CPUS_ALL_REGIONS"]["available"] == 64
    assert receipt["global_cpu_inventory_resource_counts"] == {
        "instances": 0,
        "reservations": 0,
        "nodeGroups": 0,
        "futureReservations": 0,
    }
    transcript = receipt[
        "authoritative_capacity_transcript_evidence"
    ]
    assert transcript["endpoint_count"] == 6
    assert set(transcript["sha256_by_endpoint"]) == set(
        _capacity_observations()
    )
    assert len(transcript["aggregate_sha256"]) == 64
    assert receipt["target_name_collisions"] == []
    assert receipt["nonterminated_c4_interference"] == []
    assert receipt["collected_via_get_only"] is True
    assert receipt["cloud_mutation_performed"] is False
    assert len(source.calls) == 2


def test_collect_fresh_capacity_rejects_insufficient_c4_limit(
    runner: ModuleType,
) -> None:
    source = _FakeStep11Runner()
    with pytest.raises(RuntimeError, match="insufficient"):
        runner._collect_fresh_capacity(
            source,
            gate=_gate(),
            collected_at_unix_seconds=1_800_000_000,
            authoritative_observations=_capacity_observations(
                c4_limit="8"
            ),
        )


def test_collect_fresh_capacity_rejects_insufficient_global_limit(
    runner: ModuleType,
) -> None:
    with pytest.raises(RuntimeError, match="insufficient"):
        runner._collect_fresh_capacity(
            _FakeStep11Runner(),
            gate=_gate(),
            collected_at_unix_seconds=1_800_000_000,
            authoritative_observations=_capacity_observations(
                global_limit="8"
            ),
        )


def test_collect_fresh_capacity_rejects_regional_c4_interference(
    runner: ModuleType,
) -> None:
    observations = _capacity_observations()
    observations["compute_aggregated_instances"]["items"][
        "zones/asia-northeast1-b"
    ] = {"instances": [_instance("unrelated-live-c4")]}
    with pytest.raises(RuntimeError, match="not isolated"):
        runner._collect_fresh_capacity(
            _FakeStep11Runner(),
            gate=_gate(),
            collected_at_unix_seconds=1_800_000_000,
            authoritative_observations=observations,
        )


def test_collect_fresh_capacity_rejects_target_name_collision(
    runner: ModuleType,
) -> None:
    observations = _capacity_observations()
    observations["compute_aggregated_instances"]["items"][
        "zones/asia-northeast1-b"
    ] = {
        "instances": [
            _instance(INSTANCE_NAMES[0], status="TERMINATED")
        ]
    }
    with pytest.raises(RuntimeError, match="not isolated"):
        runner._collect_fresh_capacity(
            _FakeStep11Runner(),
            gate=_gate(),
            collected_at_unix_seconds=1_800_000_000,
            authoritative_observations=observations,
        )


@pytest.mark.parametrize(
    "endpoint_id",
    [
        "compute_aggregated_reservations",
        "compute_aggregated_node_groups",
        "compute_aggregated_future_reservations",
    ],
)
def test_authoritative_capacity_rejects_nonempty_noninstance_inventory(
    runner: ModuleType,
    endpoint_id: str,
) -> None:
    observations = _capacity_observations()
    collection = _INVENTORY_COLLECTIONS[endpoint_id][0]
    observations[endpoint_id]["items"]["zones/asia-northeast1-b"] = {
        collection: [{"name": "unsupported-capacity-holder"}]
    }
    with pytest.raises(ValueError):
        runner._authoritative_capacity_facts(
            observations, gate=_gate()
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_endpoint",
        "next_page_token",
        "unreachable_scope",
        "scope_set_mismatch",
        "reservation_consumption",
        "nonstandard_machine",
    ],
)
def test_authoritative_capacity_rejects_incomplete_or_unsized_inventory(
    runner: ModuleType,
    mutation: str,
) -> None:
    observations = _capacity_observations()
    instances = observations["compute_aggregated_instances"]
    if mutation == "missing_endpoint":
        observations.pop("compute_aggregated_future_reservations")
    elif mutation == "next_page_token":
        instances["nextPageToken"] = "unexpected-page"
    elif mutation == "unreachable_scope":
        instances["unreachables"] = ["zones/asia-northeast1-b"]
    elif mutation == "scope_set_mismatch":
        observations["compute_aggregated_reservations"]["items"].pop(
            "regions/asia-northeast1"
        )
    else:
        record = _instance("unsupported-instance")
        if mutation == "reservation_consumption":
            record["reservationConsumptionInfo"] = {
                "consumptionType": "SPECIFIC_RESERVATION"
            }
        else:
            zone_link = record["zone"]
            record["machineType"] = (
                f"{zone_link}/machineTypes/e2-custom-3-4096"
            )
        instances["items"]["zones/asia-northeast1-b"] = {
            "instances": [record]
        }
    with pytest.raises(ValueError):
        runner._authoritative_capacity_facts(
            observations, gate=_gate()
        )


def test_active_non_c4_instance_consumes_global_cpu_quota(
    runner: ModuleType,
) -> None:
    observations = _capacity_observations()
    observations["compute_aggregated_instances"]["items"][
        "zones/asia-northeast1-b"
    ] = {
        "instances": [
            _instance(
                "active-n2-instance", machine_type="n2-standard-8"
            )
        ]
    }
    receipt = runner._collect_fresh_capacity(
        _FakeStep11Runner(),
        gate=_gate(),
        collected_at_unix_seconds=1_800_000_000,
        authoritative_observations=observations,
    )
    assert receipt["quota"]["CPUS_ALL_REGIONS"] == {
        "limit": 64,
        "usage": 8,
        "available": 56,
        "limit_source": (
            "cloudquotas.googleapis.com/v1/quotaInfos.get"
        ),
        "usage_source": (
            "compute.googleapis.com/compute/v1/aggregated:"
            "conservative-standard-instance-vcpu-upper-bound-with-"
            "empty-reservations-nodeGroups-futureReservations"
        ),
    }
    assert receipt["quota"]["C4_CPUS_PER_VM_FAMILY"]["usage"] == 0


def _provider_transcripts(
    runner: ModuleType,
    observations: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        runner.real_preflight._transcript(
            endpoint_id=endpoint_id,
            receipts=[{"page_tokens_exhausted": True}],
            normalized_observation=observation,
        )
        for endpoint_id, observation in observations.items()
    ]


def test_capacity_transcript_evidence_binds_endpoint_count_and_digest(
    runner: ModuleType,
) -> None:
    observations = _capacity_observations()
    transcripts = _provider_transcripts(runner, observations)
    evidence = runner._capacity_transcript_digest_evidence(
        observations, transcripts, external_read=True
    )
    assert evidence["endpoint_count"] == 6
    assert set(evidence["sha256_by_endpoint"]) == set(observations)

    with pytest.raises(ValueError, match="endpoint set"):
        runner._capacity_transcript_digest_evidence(
            observations, transcripts[:-1], external_read=True
        )

    bad = copy.deepcopy(transcripts)
    bad[0]["transcript_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="transcript evidence"):
        runner._capacity_transcript_digest_evidence(
            observations, bad, external_read=True
        )


def _authorization_specs(runner: ModuleType) -> list[dict[str, Any]]:
    targets = list(runner.rest_iam.PolicyTarget)
    return [
        {
            "target": targets[index % len(targets)],
            "role": f"projects/p/roles/exactRole{index}",
            "member": f"serviceAccount:principal-{index}@example.test",
            "condition": {
                "title": f"exact-condition-{index}",
                "expression": f"request.time < timestamp('2030-01-{index + 1:02d}T00:00:00Z')",
            },
        }
        for index in range(10)
    ]


def _authorization_live(
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "target": row["target"].value,
            "role": row["role"],
            "members": [row["member"]],
            "all_members": [row["member"]],
            "condition": copy.deepcopy(row["condition"]),
        }
        for row in reversed(specs)
    ]


def test_authorization_readback_requires_exact_canonical_binding_set(
    runner: ModuleType,
) -> None:
    specs = _authorization_specs(runner)
    live = _authorization_live(specs)
    digest = runner._exact_authorization_readback_sha256(specs, live)
    assert len(digest) == 64

    role_swap = copy.deepcopy(live)
    role_swap[0]["role"] = role_swap[1]["role"]
    with pytest.raises(RuntimeError, match="exact target/role/member"):
        runner._exact_authorization_readback_sha256(specs, role_swap)

    condition_swap = copy.deepcopy(live)
    condition_swap[0]["condition"] = copy.deepcopy(
        condition_swap[1]["condition"]
    )
    with pytest.raises(RuntimeError, match="exact target/role/member"):
        runner._exact_authorization_readback_sha256(
            specs, condition_swap
        )


def test_authorization_readback_rejects_shared_or_extra_member_binding(
    runner: ModuleType,
) -> None:
    specs = _authorization_specs(runner)
    live = _authorization_live(specs)
    live[0]["all_members"].append(
        "serviceAccount:unrelated@example.test"
    )
    with pytest.raises(RuntimeError, match="one exact binding"):
        runner._exact_authorization_readback_sha256(specs, live)


def test_prelaunch_compute_empty_rechecks_both_instances_and_disks(
    runner: ModuleType,
) -> None:
    class _Client:
        def __init__(self, disk_collision: bool = False) -> None:
            self.disk_collision = disk_collision
            self.calls: list[str] = []

        def request(self, *, method: str, url: str) -> Any:
            assert method == "GET"
            self.calls.append(url)
            status = (
                200
                if self.disk_collision
                and "/disks/" in url
                and url.endswith(INSTANCE_NAMES[0])
                else 404
            )
            return runner.controller.HttpResponse(
                status=status, headers={}, body=b""
            )

    clean = _Client()
    receipt = runner._prelaunch_exact_compute_empty(
        clean, names=INSTANCE_NAMES
    )
    assert receipt["provider_get_404_count"] == 4
    assert len(clean.calls) == 4
    assert sum("/instances/" in url for url in clean.calls) == 2
    assert sum("/disks/" in url for url in clean.calls) == 2

    with pytest.raises(FileExistsError, match="disks"):
        runner._prelaunch_exact_compute_empty(
            _Client(disk_collision=True), names=INSTANCE_NAMES
        )


def test_final_preinsert_timing_gate_rejects_get_elapsed_staleness(
    runner: ModuleType,
) -> None:
    gate = {
        "authorization_window": {
            "expires_at_unix_seconds": 10_000
        }
    }
    capacity = {"collected_at_unix_seconds": 1_000}
    prefix = {"observed_at_unix_seconds": 1_000}
    assert (
        runner._require_launch_evidence_and_authorization_fresh(
            gate=gate,
            capacity=capacity,
            prefix=prefix,
            now_unix_seconds=1_300,
        )
        == 8_700
    )
    with pytest.raises(RuntimeError, match="became stale"):
        runner._require_launch_evidence_and_authorization_fresh(
            gate=gate,
            capacity=capacity,
            prefix=prefix,
            now_unix_seconds=1_301,
        )


def test_final_preinsert_timing_gate_rejects_lost_cleanup_margin(
    runner: ModuleType,
) -> None:
    minimum = runner.step11_cloud.MIN_EXECUTION_REMAINING_SECONDS
    now = 2_000
    gate = {
        "authorization_window": {
            "expires_at_unix_seconds": now + minimum - 1
        }
    }
    with pytest.raises(RuntimeError, match="cleanup margin"):
        runner._require_launch_evidence_and_authorization_fresh(
            gate=gate,
            capacity={"collected_at_unix_seconds": now},
            prefix={"observed_at_unix_seconds": now},
            now_unix_seconds=now,
        )


def _observed_custom_roles(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "name": value["name"],
            "stage": value["stage"],
            "includedPermissions": list(value["permissions"]),
            "etag": f"etag-{key}",
        }
        for key, value in gate["iam_contract"]["custom_roles"].items()
    }


def _router(gate: dict[str, Any]) -> dict[str, Any]:
    capacity = gate["capacity_contract"]
    return {
        "name": capacity["nat_router_resource"].rsplit("/", 1)[-1],
        "region": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{capacity['project']}/regions/{capacity['region']}"
        ),
        "network": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{capacity['project']}/global/networks/default"
        ),
        "nats": [
            {
                "name": capacity["nat_name"],
                "sourceSubnetworkIpRangesToNat": (
                    capacity["nat_source_subnetwork_ip_ranges"]
                ),
                "natIpAllocateOption": "AUTO_ONLY",
            }
        ],
    }


def _enabled_services(runner: ModuleType) -> list[dict[str, Any]]:
    return [
        {"state": "ENABLED", "config": {"name": name}}
        for name in sorted(runner.REQUIRED_ENABLED_SERVICES)
    ]


def test_live_custom_roles_require_exact_included_permissions(
    runner: ModuleType,
    local_bundle: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _signer, _candidate, _reference, _pair, gate, _recovery = local_bundle
    observed = _observed_custom_roles(gate)
    checked = runner._validate_live_custom_roles(
        gate=gate, observed_roles=observed
    )
    assert len(checked) == 6
    changed = copy.deepcopy(observed)
    changed["controller_vm_launch"]["includedPermissions"].append(
        "compute.instances.list"
    )
    with pytest.raises(ValueError, match="custom role drifted"):
        runner._validate_live_custom_roles(
            gate=gate, observed_roles=changed
        )


def test_nat_and_enabled_services_are_exact_get_only_prerequisites(
    runner: ModuleType,
    local_bundle: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _signer, _candidate, _reference, _pair, gate, _recovery = local_bundle
    nat = runner._validate_nat_router(
        gate=gate, observed_router=_router(gate)
    )
    assert nat["nat_name"] == gate["capacity_contract"]["nat_name"]
    services = runner._validate_enabled_services(
        _enabled_services(runner)
    )
    assert services["required_services_enabled"] is True

    wrong_nat = _router(gate)
    wrong_nat["nats"][0]["sourceSubnetworkIpRangesToNat"] = (
        "LIST_OF_SUBNETWORKS"
    )
    with pytest.raises(ValueError, match="NAT configuration"):
        runner._validate_nat_router(
            gate=gate, observed_router=wrong_nat
        )
    with pytest.raises(ValueError, match="not enabled"):
        runner._validate_enabled_services(
            _enabled_services(runner)[:-1]
        )


def test_step11_exception_explicitly_does_not_authorize_step12(
    runner: ModuleType,
) -> None:
    checked = runner._validate_step11_does_not_authorize_step12()
    assert checked["exception_authorizes_step12"] is False
    source = runner._read_json(
        runner.STEP11_ROOT / "shared_project_exception.json"
    )
    source["exception_authorizes_step12"] = True
    unsigned = dict(source)
    unsigned.pop("exception_sha256")
    source["exception_sha256"] = runner.controller.canonical_sha256(
        unsigned
    )
    with pytest.raises(ValueError, match="non-authorization"):
        runner._validate_step11_does_not_authorize_step12(source)


def test_step12_bounded_exception_records_unproven_strict_iam_and_no_retry(
    runner: ModuleType,
    local_bundle: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _signer, candidate, _reference, pair, gate, _recovery = local_bundle
    prefix = runner._seal(
        {
            "stage_prefix": candidate["remote_layout"]["stage_prefix"],
            "object_count": 0,
        }
    )
    package = runner._seal(
        {
            "outer_package_identity_sha256": candidate[
                "outer_package_manifest"
            ]["outer_package_identity_sha256"],
        }
    )
    capacity = runner._seal(
        {
            "target_instance_names": sorted(INSTANCE_NAMES),
            "available_vcpu": 24,
        }
    )
    step11 = runner._validate_step11_does_not_authorize_step12()
    security = runner._seal(
        {
            "gate_plan_sha256": gate["plan_sha256"],
            "step11_non_authorization": step11,
        }
    )
    receipt = runner._build_step12_bounded_exception(
        pair=pair,
        gate=gate,
        prefix_empty=prefix,
        package_readback=package,
        capacity=capacity,
        security_evidence=security,
    )
    checked = runner._validate_step12_bounded_exception(
        receipt,
        pair=pair,
        gate=gate,
        prefix_empty=prefix,
        package_readback=package,
        capacity=capacity,
        security_evidence=security,
    )
    assert checked["bounded_exception_authorizes_exact_step12_pair"] is True
    assert checked["strict_iam_gate_passed"] is False
    assert checked["strict_effective_iam_simulation_passed"] is False
    assert checked["ancestor_deny_policy_effectiveness_proven"] is False
    assert checked["shared_worker_identity_residual_risk"] is True
    assert checked["exception_authorizes_retry"] is False
    assert checked["exception_authorizes_attempt1"] is False
    assert checked["performance_lock_evidence"] is False
    assert checked["training_eligible"] is False


def test_cleanup_never_deletes_compute_when_iam_zero_is_unproven(
    runner: ModuleType,
    local_bundle: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _signer, _candidate, _reference, _pair, gate, _recovery = local_bundle

    class _Compute:
        calls: list[dict[str, Any]] = []

        def request(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise AssertionError("compute cleanup ran before IAM zero")

    compute = _Compute()
    monkeypatch.setattr(runner, "_ACTIVE_GATE_PLAN", gate)
    monkeypatch.setattr(runner, "_ACTIVE_OUTPUT_ROOT", None)
    monkeypatch.setattr(runner, "_REST_ADMIN", object())
    monkeypatch.setattr(runner, "_USER_COMPUTE_CLIENT", compute)
    monkeypatch.setattr(runner, "_CLOUD_MUTATION_STARTED", True)
    monkeypatch.setattr(runner, "_assert_profile_unchanged", lambda: None)

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("IAM zero unproven")

    monkeypatch.setattr(runner, "_clear_pair_bindings_with_retry", _fail)
    with pytest.raises(BaseExceptionGroup):
        runner._final_cleanup()
    assert compute.calls == []


def test_pre_mutation_gate_failure_never_starts_cleanup_mutation(
    runner: ModuleType,
    local_bundle: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _signer, _candidate, _reference, _pair, gate, _recovery = local_bundle

    class _NeverCalled:
        calls = 0

        def __getattr__(self, name: str) -> Any:
            self.calls += 1
            raise AssertionError(f"pre-mutation cleanup called {name}")

    admin = _NeverCalled()
    compute = _NeverCalled()
    monkeypatch.setattr(runner, "_ACTIVE_GATE_PLAN", gate)
    monkeypatch.setattr(runner, "_ACTIVE_OUTPUT_ROOT", None)
    monkeypatch.setattr(runner, "_REST_ADMIN", admin)
    monkeypatch.setattr(runner, "_USER_COMPUTE_CLIENT", compute)
    monkeypatch.setattr(runner, "_CLOUD_MUTATION_STARTED", False)
    monkeypatch.setattr(runner, "_assert_profile_unchanged", lambda: None)
    receipt = runner._final_cleanup()
    assert receipt["cloud_mutation_performed"] is False
    assert receipt["iam_revoke_attempted"] is False
    assert receipt["compute_delete_attempted"] is False
    assert admin.calls == 0
    assert compute.calls == 0


def test_default_runner_dry_run_performs_no_cloud_action(
    runner: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert runner.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == runner.RUNNER_SCHEMA
    assert output["instance_names"] == INSTANCE_NAMES
    assert output["actual_machine_type"] == "c4-standard-8"
    assert output["requested_c4_vcpu"] == 16
    assert output["binding_count"] == 10
    assert output["cloud_api_calls_performed"] is False
    assert output["cloud_mutation_performed"] is False
    assert output["launch_authorized"] is False
    assert output["current_profile_changed"] is False
