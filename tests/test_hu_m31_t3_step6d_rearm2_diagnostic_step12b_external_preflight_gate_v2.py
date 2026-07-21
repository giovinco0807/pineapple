from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as payload_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as step11_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_content_v2
    as bootstrap_source_content,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_authorization_v2
    as external_auth,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_preflight_gate_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
RUN_NONCE = "a1" * 32
AUTH_NONCE = "b2" * 32
ISSUED = 1_900_000_000
NOW = ISSUED + 400
EXPIRES = ISSUED + 3_600


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


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


def _machine() -> dict[str, Any]:
    return {
        "name": "c4-standard-8",
        "guestCpus": 8,
        "memoryMb": subject.deployment_v2.ACTUAL_MEMORY_MB,
        "zone": "asia-northeast1-b",
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/"
            "ofc-solver-485418/zones/asia-northeast1-b/"
            "machineTypes/c4-standard-8"
        ),
    }


def _region() -> dict[str, Any]:
    return {
        "name": "asia-northeast1",
        "status": "UP",
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/"
            "ofc-solver-485418/regions/asia-northeast1"
        ),
        "quotas": [
            {"metric": "CPUS", "limit": 355.0, "usage": 0.0},
            {
                "metric": "PREEMPTIBLE_CPUS",
                "limit": 468.0,
                "usage": 0.0,
            },
        ],
    }


def _nat() -> dict[str, Any]:
    return {
        "name": subject.NAT_ROUTER_NAME,
        "region": (
            "https://www.googleapis.com/compute/v1/projects/"
            "ofc-solver-485418/regions/asia-northeast1"
        ),
        "network": (
            "https://www.googleapis.com/compute/v1/projects/"
            "ofc-solver-485418/global/networks/default"
        ),
        "nats": [
            {
                "name": subject.NAT_NAME,
                "sourceSubnetworkIpRangesToNat": (
                    subject.NAT_SOURCE_SUBNETWORK_IP_RANGES
                ),
                "natIpAllocateOption": subject.NAT_IP_ALLOCATE_OPTION,
            }
        ],
    }


def _services() -> list[dict[str, Any]]:
    return [
        {"state": "ENABLED", "config": {"name": name}}
        for name in sorted(subject.REQUIRED_ENABLED_SERVICES)
    ]


def _role_provider_readbacks(
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        row["purpose"]: {
            "name": row["name"],
            "stage": row["stage"],
            "includedPermissions": list(row["included_permissions"]),
            "deleted": False,
        }
        for row in plan["custom_role_readback_contract"]["requirements"]
    }


def _phase2_role_readbacks(
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "stage": row["stage"],
            "included_permissions": list(row["included_permissions"]),
            "deleted": False,
            "get_status": 200,
            "readback_complete": True,
        }
        for row in plan["custom_role_readback_contract"]["requirements"]
    ]


def _phase2_binding_readbacks(
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            key: copy.deepcopy(row[key])
            for key in (
                "purpose",
                "target",
                "resource",
                "role",
                "member",
                "condition",
                "binding_sha256",
            )
        }
        | {"member_occurrences": 1, "readback_complete": True}
        for row in [
            *plan["phase2_bindings"]["controller"],
            *plan["phase2_bindings"]["worker"],
        ]
    ]


def _source_provision(
    plan: dict[str, Any],
    *,
    deployment: dict[str, Any],
    payloads: list[dict[str, Any]],
    public_record: dict[str, Any],
) -> bootstrap_source.ValidatedBootstrapSourceProvision:
    records = [
        {
            key: copy.deepcopy(row[key])
            for key in ("kind", "path", "uri", "bytes", "sha256")
        }
        | {
            "generation": 1_900_000_000_000_000 + position,
            "created": True,
            "readback_verified": True,
        }
        for position, row in enumerate(plan["objects"], start=1)
    ]
    generations = {
        row["uri"]: row["generation"] for row in records
    }
    body = {
        "schema": bootstrap_source.SOURCE_PROVISION_RECEIPT_SCHEMA,
        "status": (
            "fresh_direct_v2_bootstrap_sources_created_and_read_back"
        ),
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_prefix": plan["source_prefix"],
        "prefix_empty_before_upload": True,
        "prefix_empty_observation_sha256": (
            bootstrap_source.canonical_sha256([])
        ),
        "if_generation_match": bootstrap_source.IF_GENERATION_MATCH,
        "object_count": bootstrap_source.SOURCE_OBJECT_COUNT,
        "records": records,
        "records_sha256": bootstrap_source.canonical_sha256(records),
        "source_generations": generations,
        "source_generations_sha256": (
            bootstrap_source.canonical_sha256(generations)
        ),
        "all_objects_created_once": True,
        "all_generation_bound": True,
        "all_bytes_and_sha256_read_back": True,
        "old_package_write_count": 0,
        "old_result_write_count": 0,
        "direct_v2_result_write_count": 0,
        "cloud_mutation_performed": True,
        "current_profile_changed": False,
    }
    receipt = {
        **body,
        "receipt_sha256": bootstrap_source.canonical_sha256(body),
    }
    return bootstrap_source.validate_bootstrap_source_provision_receipt(
        receipt,
        source_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
    )


def _controller_create_receipt(
    deployment: dict[str, Any],
) -> dict[str, Any]:
    controller = deployment["controller_service_account"]
    body = {
        "schema": controller_sa.SCHEMA,
        "status": "run_scoped_controller_service_account_created",
        "project": controller["project"],
        "account_id": controller["account_id"],
        "email": controller["email"],
        "provider": {
            "name": (
                f"projects/{controller['project']}/serviceAccounts/"
                f"{controller['email']}"
            ),
            "project_id": controller["project"],
            "unique_id": "123456789012345678901",
            "email": controller["email"],
            "disabled": False,
        },
        "create_call_count": 1,
        "create_retry_count": 0,
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    return {**body, "receipt_sha256": subject.canonical_sha256(body)}


class _TokenSource:
    expire_time = "2030-03-17T18:46:40Z"

    def access_token(self) -> str:
        return "not-serialized-test-token"


_ZERO_READBACK = {
    "zero_observed": True,
    "poll_count": 1,
    "stale_present_count": 0,
    "observation_sha256s": ["3" * 64],
    "observation_sha256s_sha256": token_barrier.canonical_sha256(
        ["3" * 64]
    ),
    "sleep_delays_seconds": [],
    "elapsed_seconds": 0.5,
    "maximum_seconds": token_barrier.MAX_PROPAGATION_SECONDS,
    "failure_reason": None,
    "second_add_performed": False,
    "controller_token_reminted": False,
}


def _token_barrier_outcome(
    deployment: dict[str, Any],
) -> token_barrier.TokenBarrierOutcome:
    body = {
        "schema": token_barrier.SCHEMA,
        "status": token_barrier.SUCCESS_STATUS,
        "controller_service_account": deployment[
            "controller_service_account"
        ]["email"],
        "run_scoped_controller_required": True,
        "binding_purpose": token_barrier.TOKEN_CREATOR_PURPOSE,
        "binding_role": token_barrier.TOKEN_CREATOR_ROLE,
        "binding_member": token_barrier.DEFAULT_INITIATING_PRINCIPAL,
        "binding_condition_sha256": subject.canonical_sha256(
            {"condition": "test"}
        ),
        "binding_add_attempts": 1,
        "binding_add_outcome": "confirmed_changed",
        "binding_add_readback_sha256": subject.canonical_sha256(
            [{"present": True}]
        ),
        "token_attempt_count": 1,
        "failed_token_attempt_count": 0,
        "token_attempts": [],
        "started_at_unix_seconds": NOW - 70,
        "finished_at_unix_seconds": NOW - 60,
        "elapsed_seconds": 10.0,
        "maximum_propagation_seconds": (
            token_barrier.MAX_PROPAGATION_SECONDS
        ),
        "token_lifetime_seconds": token_barrier.TOKEN_LIFETIME_SECONDS,
        "token_expire_time": _TokenSource.expire_time,
        "fixed_nonrefreshing_controller_credential": True,
        "controller_token_remint_forbidden": True,
        "token_creator_revoke_attempts": 1,
        "token_creator_revoke_changed": True,
        "token_creator_revoke_readback_sha256": (
            subject.canonical_sha256([])
        ),
        "token_creator_revoke_zero_readback": _ZERO_READBACK,
        "token_creator_revoke_zero_readback_evidence_sha256": (
            token_barrier.canonical_sha256(_ZERO_READBACK)
        ),
        "token_creator_live_after_barrier": False,
        "phase2_started": False,
        "phase2_must_exclude_token_creator": True,
        "run_scoped_controller_delete_after_pair_claim_required": True,
        "vm_insert_attempt_count": 0,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "response_body_stored": False,
        "cloud_mutation_performed": True,
        "current_profile_changed": False,
    }
    receipt = {
        **body,
        "receipt_sha256": token_barrier.canonical_sha256(body),
    }
    return token_barrier.TokenBarrierOutcome(_TokenSource(), receipt)


@pytest.fixture(scope="module")
def context() -> dict[str, Any]:
    signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    public_record = dict(signer.public_record)
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"], done_records=[done]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    source_bytes = subject.load_expected_source_bytes()
    runtime_source_files = {
        path: source_bytes[logical_name].decode("utf-8")
        for path, logical_name in (
            subject.BOOTSTRAP_RUNTIME_SOURCE_LOGICAL_NAMES.items()
        )
    }
    content_binding = (
        bootstrap_source_content.build_bootstrap_source_content_binding(
            runtime_source_files=runtime_source_files,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
        )
    )
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
        bootstrap_source_content_binding=content_binding,
    )
    source_plan = bootstrap_source.build_bootstrap_source_plan(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
        runtime_source_files=runtime_source_files,
    )
    source_provision = _source_provision(
        source_plan,
        deployment=deployment,
        payloads=payloads,
        public_record=public_record,
    )
    role_manifests = {
        job_id: bootstrap_source.build_role_bootstrap_manifest(
            source_plan=source_plan,
            validated_provision=source_provision,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_record,
            run_nonce=RUN_NONCE,
            external_job_id=job_id,
        )
        for job_id in deployment["selected_job_ids"]
    }
    plan = phase2_iam.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=EXPIRES,
    )
    phase2_readback = (
        phase2_iam.build_step12b_phase2_iam_readback_receipt(
            plan,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_record,
            run_nonce=RUN_NONCE,
            observed_at_unix_seconds=NOW - 50,
            custom_role_readbacks=_phase2_role_readbacks(plan),
            binding_readbacks=_phase2_binding_readbacks(plan),
            unexpected_targeted_bindings=[],
        )
    )
    names = [row["instance_name"] for row in deployment["instances"]]
    prefix = subject.build_direct_v2_prefix_empty_receipt(
        deployment,
        observed_at_unix_seconds=NOW - 40,
        provider_pages=[{}],
    )
    compute = subject.build_compute_absence_receipt(
        deployment,
        observed_at_unix_seconds=NOW - 30,
        instance_get_readbacks=[
            {"name": name, "http_status": 404} for name in names
        ],
        disk_get_readbacks=[
            {"name": name, "http_status": 404} for name in names
        ],
        operation_name_history={name: [] for name in names},
    )
    live = subject.build_live_readback_receipt(
        deployment,
        phase2_iam_plan=plan,
        observed_at_unix_seconds=NOW - 20,
        machine_type_readback=_machine(),
        region_readback=_region(),
        authoritative_capacity_observations=_capacity_observations(),
        nat_router_readback=_nat(),
        enabled_service_readbacks=_services(),
        custom_role_readbacks=_role_provider_readbacks(plan),
    )
    return {
        "signer": signer,
        "public_record": public_record,
        "payloads": payloads,
        "deployment": deployment,
        "plan": plan,
        "phase2_readback": phase2_readback,
        "controller_create": _controller_create_receipt(deployment),
        "token_barrier_outcome": _token_barrier_outcome(deployment),
        "source_plan": source_plan,
        "source_provision": source_provision,
        "role_manifests": role_manifests,
        "package": _read_json(
            STEP11_ROOT / "package_provision_receipt.json"
        ),
        "prefix": prefix,
        "compute": compute,
        "live": live,
        "source_bytes": source_bytes,
        "profile_bytes": (
            REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
        ).read_bytes(),
    }


def _gate_kwargs(context: dict[str, Any]) -> dict[str, Any]:
    candidate, reference = context["payloads"]
    return {
        "deployment_contract": context["deployment"],
        "candidate_payload_contract": candidate,
        "reference_payload_contract": reference,
        "controller_public_key_record": context["public_record"],
        "run_nonce": RUN_NONCE,
        "package_provision_receipt": context["package"],
        "direct_v2_prefix_empty_receipt": context["prefix"],
        "compute_absence_receipt": context["compute"],
        "live_readback_receipt": context["live"],
        "phase2_iam_plan": context["plan"],
        "phase2_iam_readback_receipt": context["phase2_readback"],
        "controller_service_account_create_receipt": context[
            "controller_create"
        ],
        "token_barrier_outcome": context["token_barrier_outcome"],
        "bootstrap_source_plan": context["source_plan"],
        "validated_bootstrap_source_provision": context[
            "source_provision"
        ],
        "role_bootstrap_manifests": context["role_manifests"],
        "source_bytes": context["source_bytes"],
        "current_profile_bytes": context["profile_bytes"],
        "now_unix_seconds": NOW,
    }


def test_real_gate_mints_opaque_capability_consumable_by_auth(
    context: dict[str, Any],
) -> None:
    before = subject.canonical_sha256(
        {"profile": context["profile_bytes"].hex()}
    )
    kwargs = _gate_kwargs(context)
    gate = subject.build_external_preflight_gate_receipt(**kwargs)
    assert (
        subject.validate_external_preflight_gate_receipt(
            gate, **kwargs
        )
        == gate
    )
    capability = subject.mint_validated_external_preflight(**kwargs)
    assert gate["schema"] == subject.SCHEMA
    assert gate["passed"] is True
    assert gate["launch_authorized"] is False
    assert gate["cloud_mutation_performed_before_gate"] is True
    assert gate["cloud_mutation_performed_by_gate"] is False
    assert gate["current_profile_changed"] is False
    assert isinstance(capability, external_auth.ValidatedExternalPreflight)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(capability)

    candidate, reference = context["payloads"]
    authorization = external_auth.build_external_authorization(
        deployment_contract=context["deployment"],
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=context["public_record"],
        run_nonce=RUN_NONCE,
        external_job_id=context["deployment"]["selected_job_ids"][0],
        package_generations=context["package"]["package_generations"],
        validated_external_preflight=capability,
        issued_unix_seconds=NOW,
        expires_unix_seconds=NOW + 600,
        signer=context["signer"],
        nonce=AUTH_NONCE,
    )
    assert (
        authorization["external_preflight_receipt_sha256"]
        == capability._receipt["external_preflight_receipt_sha256"]
    )
    source_binding = authorization["bootstrap_source_binding"]
    assert source_binding["source_object_count"] == 3
    assert source_binding["role_manifest_object_count"] == 2
    assert len(source_binding["role_source_generations"]) == 2
    assert all(
        "reference_payload_contract.json" not in uri
        for uri in source_binding["role_source_generations"]
    )
    assert [
        row["kind"] for row in source_binding["role_manifest_objects"]
    ] == [
        "shared_runtime_source_bundle",
        "candidate_role_payload_contract",
    ]
    assert source_binding["source_generations_sha256"] == (
        context["source_provision"].generations_sha256
    )
    assert (
        "generation_pinned_bootstrap_source_download"
        in authorization["allowed_operations"]
    )
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        context["public_record"]
    )
    assert (
        external_auth.validate_external_authorization_role_runtime(
            authorization,
            deployment_contract=context["deployment"],
            selected_payload_contract=candidate,
            controller_public_key_record=context["public_record"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            package_generations=context["package"]["package_generations"],
            external_preflight_receipt=capability._receipt,
            verifier=verifier,
            now_unix_seconds=NOW + 1,
        )
        == authorization
    )
    after = subject.canonical_sha256(
        {"profile": (
            REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
        ).read_bytes().hex()}
    )
    assert after == before


def test_exact_package_generations_sha_and_bytes_are_required(
    context: dict[str, Any],
) -> None:
    changed = copy.deepcopy(context["package"])
    changed["records"][0]["generation"] += 1
    changed["package_generations"][
        changed["records"][0]["uri"]
    ] = changed["records"][0]["generation"]
    changed["records_sha256"] = subject.canonical_sha256(
        changed["records"]
    )
    changed["package_generations_sha256"] = subject.canonical_sha256(
        changed["package_generations"]
    )
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = subject.canonical_sha256(changed)
    kwargs = _gate_kwargs(context)
    kwargs["package_provision_receipt"] = changed
    with pytest.raises(ValueError, match="immutable package"):
        subject.build_external_preflight_gate_receipt(**kwargs)


def test_nonempty_prefix_and_compute_history_fail_closed(
    context: dict[str, Any],
) -> None:
    deployment = context["deployment"]
    with pytest.raises(FileExistsError, match="not empty"):
        subject.build_direct_v2_prefix_empty_receipt(
            deployment,
            observed_at_unix_seconds=NOW,
            provider_pages=[{"items": [{"name": "unexpected"}]}],
        )
    names = [row["instance_name"] for row in deployment["instances"]]
    with pytest.raises(FileExistsError, match="history"):
        subject.build_compute_absence_receipt(
            deployment,
            observed_at_unix_seconds=NOW,
            instance_get_readbacks=[
                {"name": name, "http_status": 404} for name in names
            ],
            disk_get_readbacks=[
                {"name": name, "http_status": 404} for name in names
            ],
            operation_name_history={
                names[0]: [{"operation": "old-insert"}],
                names[1]: [],
            },
        )


@pytest.mark.parametrize(
    "mutation",
    ["capacity", "nat", "services", "custom_role"],
)
def test_live_provider_drift_fails_closed(
    context: dict[str, Any], mutation: str
) -> None:
    plan = context["plan"]
    machine = _machine()
    region = _region()
    capacity = _capacity_observations()
    nat = _nat()
    services = _services()
    roles = _role_provider_readbacks(plan)
    if mutation == "capacity":
        capacity["cloud_quotas_c4_cpu"] = _family_quota(limit="8")
    elif mutation == "nat":
        nat["nats"][0]["natIpAllocateOption"] = "MANUAL_ONLY"
    elif mutation == "services":
        services = services[1:]
    else:
        purpose = sorted(roles)[0]
        roles[purpose]["includedPermissions"] = []
    with pytest.raises(ValueError):
        subject.build_live_readback_receipt(
            context["deployment"],
            phase2_iam_plan=plan,
            observed_at_unix_seconds=NOW,
            machine_type_readback=machine,
            region_readback=region,
            authoritative_capacity_observations=capacity,
            nat_router_readback=nat,
            enabled_service_readbacks=services,
            custom_role_readbacks=roles,
        )


def test_actual_c4_memory_is_bound_and_old_32g_fixture_fails_closed(
    context: dict[str, Any],
) -> None:
    assert subject.deployment_v2.ACTUAL_MEMORY_MB == 30_720
    changed = _machine()
    changed["memoryMb"] = 32_768
    with pytest.raises(ValueError, match="machine or region"):
        subject.build_live_readback_receipt(
            context["deployment"],
            phase2_iam_plan=context["plan"],
            observed_at_unix_seconds=NOW,
            machine_type_readback=changed,
            region_readback=_region(),
            authoritative_capacity_observations=_capacity_observations(),
            nat_router_readback=_nat(),
            enabled_service_readbacks=_services(),
            custom_role_readbacks=_role_provider_readbacks(context["plan"]),
        )


def test_stale_observation_fails_closed(
    context: dict[str, Any],
) -> None:
    stale = subject.build_direct_v2_prefix_empty_receipt(
        context["deployment"],
        observed_at_unix_seconds=(
            NOW - subject.OBSERVATION_MAX_AGE_SECONDS - 1
        ),
        provider_pages=[{}],
    )
    kwargs = _gate_kwargs(context)
    kwargs["direct_v2_prefix_empty_receipt"] = stale
    with pytest.raises(ValueError, match="stale"):
        subject.build_external_preflight_gate_receipt(**kwargs)


def test_phase2_body_tamper_fails_closed(
    context: dict[str, Any],
) -> None:
    changed = copy.deepcopy(context["phase2_readback"])
    changed["controller_binding_count"] -= 1
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = subject.canonical_sha256(changed)
    kwargs = _gate_kwargs(context)
    kwargs["phase2_iam_readback_receipt"] = changed
    with pytest.raises(ValueError):
        subject.build_external_preflight_gate_receipt(**kwargs)


def test_bootstrap_source_plan_and_role_manifest_tamper_fail_closed(
    context: dict[str, Any],
) -> None:
    kwargs = _gate_kwargs(context)
    changed_plan = copy.deepcopy(context["source_plan"])
    changed_plan["source_prefix"] += "-tampered"
    changed_plan.pop("source_plan_sha256")
    changed_plan["source_plan_sha256"] = subject.canonical_sha256(
        changed_plan
    )
    kwargs["bootstrap_source_plan"] = changed_plan
    with pytest.raises(ValueError):
        subject.build_external_preflight_gate_receipt(**kwargs)

    kwargs = _gate_kwargs(context)
    changed_manifests = copy.deepcopy(context["role_manifests"])
    job_id = context["deployment"]["selected_job_ids"][0]
    changed_manifests[job_id]["objects"][0]["generation"] += 1
    changed_manifests[job_id]["objects_sha256"] = (
        subject.canonical_sha256(changed_manifests[job_id]["objects"])
    )
    changed_manifests[job_id].pop("role_manifest_sha256")
    changed_manifests[job_id]["role_manifest_sha256"] = (
        subject.canonical_sha256(changed_manifests[job_id])
    )
    kwargs["role_bootstrap_manifests"] = changed_manifests
    with pytest.raises(ValueError):
        subject.build_external_preflight_gate_receipt(**kwargs)


def test_controller_create_and_token_barrier_tamper_fail_closed(
    context: dict[str, Any],
) -> None:
    kwargs = _gate_kwargs(context)
    changed_create = copy.deepcopy(context["controller_create"])
    changed_create["provider"]["disabled"] = True
    changed_create.pop("receipt_sha256")
    changed_create["receipt_sha256"] = subject.canonical_sha256(
        changed_create
    )
    kwargs["controller_service_account_create_receipt"] = changed_create
    with pytest.raises(ValueError):
        subject.build_external_preflight_gate_receipt(**kwargs)

    kwargs = _gate_kwargs(context)
    original = context["token_barrier_outcome"]
    changed_token = copy.deepcopy(original.receipt)
    changed_token["token_creator_live_after_barrier"] = True
    changed_token.pop("receipt_sha256")
    changed_token["receipt_sha256"] = token_barrier.canonical_sha256(
        changed_token
    )
    kwargs["token_barrier_outcome"] = token_barrier.TokenBarrierOutcome(
        original.token_source(),
        changed_token,
    )
    with pytest.raises(ValueError, match="token barrier"):
        subject.build_external_preflight_gate_receipt(**kwargs)


def test_final_readbacks_must_follow_phase2_install(
    context: dict[str, Any],
) -> None:
    early_prefix = subject.build_direct_v2_prefix_empty_receipt(
        context["deployment"],
        observed_at_unix_seconds=NOW - 55,
        provider_pages=[{}],
    )
    kwargs = _gate_kwargs(context)
    kwargs["direct_v2_prefix_empty_receipt"] = early_prefix
    with pytest.raises(ValueError, match="preceded Phase2"):
        subject.build_external_preflight_gate_receipt(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    ["source_tamper", "aa_placeholder", "profile_tamper"],
)
def test_actual_source_and_profile_bytes_not_hash_placeholders(
    context: dict[str, Any], mutation: str
) -> None:
    kwargs = _gate_kwargs(context)
    if mutation == "profile_tamper":
        kwargs["current_profile_bytes"] = context["profile_bytes"] + b"\n"
    else:
        sources = dict(context["source_bytes"])
        sources["external_authorization"] = (
            b"tamper" if mutation == "source_tamper" else b"\xaa" * 32
        )
        kwargs["source_bytes"] = sources
    with pytest.raises(ValueError, match="bytes changed"):
        subject.build_external_preflight_gate_receipt(**kwargs)


def test_hash_only_legacy_builder_remains_fail_closed() -> None:
    with pytest.raises(ValueError, match="owning real gate"):
        external_auth.build_external_preflight_receipt(
            readonly_preflight_receipt_sha256="aa" * 32,
            iam_capacity_gate_receipt_sha256="aa" * 32,
            package_provision_receipt_sha256="aa" * 32,
        )


def test_token_barrier_fixture_matches_real_producer_field_set():
    # Drift-guard: the external preflight gate validates the token barrier
    # receipt with an exact field set. If the real producer adds a field
    # (as the zero-readback fields once were), this test must fail until the
    # fixture AND the validator are updated together - so the stale-allowlist
    # bug that stopped Step12l cannot silently recur.
    class _Admin:
        controller_service_account = (
            "ofc-m31-s2b-fixturecheck@ofc-solver-485418.iam.gserviceaccount.com"
        )

        def __init__(self) -> None:
            self.binding = None

        def add_binding(self, _t, *, role, member, condition):
            self.binding = {"role": role, "members": [member], "condition": dict(condition)}
            return type("M", (), {"changed": True, "attempts": 1})()

        def remove_binding(self, _t, *, role, member, condition):
            self.binding = None
            return type("M", (), {"changed": True, "attempts": 1})()

        def get_policy(self, _t):
            return {"bindings": [] if self.binding is None else [dict(self.binding)]}

    class _Tok:
        expire_time = "2030-01-01T01:00:00Z"

        def access_token(self):
            return "ya29.fixturecheck"

    class _Gen:
        controller_service_account = _Admin.controller_service_account

        def generate_controller_access_token(self, *, lifetime_seconds):
            return _Tok()

    class _Clock:
        def __init__(self):
            self.t = 0.0

        def monotonic(self):
            return self.t

        def unix(self):
            return 1_900_000_000 + int(self.t)

        def sleep(self, s):
            self.t += s

    clock = _Clock()
    outcome = token_barrier.run_token_creator_barrier(
        admin=_Admin(),
        token_generator=_Gen(),
        binding=token_barrier.build_token_creator_binding(
            controller_service_account=_Admin.controller_service_account,
            expires_at_rfc3339="2030-01-01T00:10:00Z",
        ),
        now_monotonic=clock.monotonic,
        now_unix_seconds=clock.unix,
        sleep=clock.sleep,
    )
    real_fields = set(outcome.receipt.keys())
    minimal_deployment = {
        "controller_service_account": {
            "email": _Admin.controller_service_account
        }
    }
    fixture_fields = set(
        _token_barrier_outcome(minimal_deployment).receipt.keys()
    )
    assert fixture_fields == real_fields
