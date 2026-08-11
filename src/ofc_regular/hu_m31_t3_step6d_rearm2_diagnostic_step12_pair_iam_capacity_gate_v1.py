"""Offline IAM/capacity contract for the Step 12 diagnostic VM pair.

This module is deliberately incapable of contacting Google Cloud.  It turns
the separately frozen Step 12 candidate/reference pair contract into one
short-lived, exact-name authorization plan and validates that plan by complete
reconstruction.

The underlying Step 6d runner remains the frozen 1x16-Rayon diagnostic
payload.  Step 12 runs that payload on two c4-standard-8 Spot VMs only as a
lifecycle/correctness canary.  Nothing produced by this gate is performance,
quality, training, or promotion evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as canary_plan,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1
    as pair_contract_module,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12_pair_iam_capacity_gate_plan_v1"
)
STATUS = "offline_exact_pair_attempt0_authorization_plan_ready"

PROJECT = transport.PROJECT
PROJECT_NUMBER = "783381566570"
BUCKET = transport.BUCKET
ZONE = transport.ZONE
REGION = "asia-northeast1"
MACHINE_TYPE = "c4-standard-8"
VCPU_PER_VM = 8
VM_COUNT = 2
TOTAL_REQUESTED_VCPU = VCPU_PER_VM * VM_COUNT
MAX_CONCURRENT_VMS = VM_COUNT
MAX_ATTEMPTS_PER_JOB = 1
AUTHORIZED_ATTEMPT_INDEX = 0
MAX_RUN_DURATION_SECONDS = 4_200
MAX_AUTHORIZATION_WINDOW_SECONDS = 7_200
MIN_AUTHORIZATION_REMAINING_SECONDS = 120
OBSERVATION_MAX_AGE_SECONDS = 300

CONTROLLER_SERVICE_ACCOUNT = (
    f"ofc-m31-t3-controller@{PROJECT}.iam.gserviceaccount.com"
)
WORKER_SERVICE_ACCOUNT = transport.WORKER_SERVICE_ACCOUNT
DEFAULT_CONTROLLER_PRINCIPAL = (
    f"serviceAccount:{CONTROLLER_SERVICE_ACCOUNT}"
)
WORKER_PRINCIPAL = f"serviceAccount:{WORKER_SERVICE_ACCOUNT}"
DEFAULT_INITIATING_PRINCIPAL = "user:giovinco.080807@gmail.com"

REQUIRED_WORKER_OAUTH_SCOPE = (
    "https://www.googleapis.com/auth/cloud-platform"
)
DEFAULT_NETWORK_RESOURCE = f"projects/{PROJECT}/global/networks/default"
DEFAULT_SUBNETWORK_RESOURCE = (
    f"projects/{PROJECT}/regions/{REGION}/subnetworks/default"
)
DEFAULT_NAT_NAME = "ofc-t3-nat-asia-northeast1"

CUSTOM_ROLE_PERMISSIONS = {
    "worker_object_reader": ("storage.objects.get",),
    "worker_result_creator": ("storage.objects.create",),
    "worker_self_delete": ("compute.instances.delete",),
    "controller_vm_launch": (
        "compute.disks.create",
        "compute.instances.create",
        "compute.instances.setMetadata",
        "compute.instances.setServiceAccount",
        "compute.networks.use",
        "compute.subnetworks.use",
    ),
    "controller_instance_lifecycle": (
        "compute.instances.delete",
        "compute.instances.get",
    ),
    "controller_zone_operation_reader": ("compute.zoneOperations.get",),
}

ROLE_IDS = {
    "worker_object_reader": "ofcM31T3ObjectReaderV1",
    "worker_result_creator": "ofcM31T3ResultCreatorV1",
    "worker_self_delete": "ofcM31T3SelfDeleteV1",
    "controller_vm_launch": "ofcM31T3VmLaunchV1",
    "controller_instance_lifecycle": "ofcM31T3VmLifecycleV1",
    "controller_zone_operation_reader": "ofcM31T3ZoneOperationReaderV1",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SERVICE_ACCOUNT_PRINCIPAL = re.compile(
    r"^serviceAccount:"
    r"(?P<name>[a-z][a-z0-9-]{4,28}[a-z0-9])@"
    r"(?P<project>[a-z][a-z0-9-]{4,61}[a-z0-9])"
    r"\.iam\.gserviceaccount\.com$"
)
_USER_PRINCIPAL = re.compile(r"^user:[^@\s]+@[^@\s]+$")

_PLAN_FIELDS = {
    "schema",
    "status",
    "source_pair_contract",
    "authorization_window",
    "principals",
    "iam_contract",
    "instance_contract",
    "capacity_contract",
    "post_claim_revoke_contract",
    "cleanup_contract",
    "gate_semantics",
    "plan_sha256",
}


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


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields changed")


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
        raise ValueError(f"{label} is not an exact bounded integer")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _safe_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe exact resource name")
    return value


def _rfc3339(unix_seconds: int) -> str:
    return (
        datetime.fromtimestamp(unix_seconds, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_gs_prefix(value: Any, *, label: str) -> str:
    marker = f"gs://{BUCKET}/"
    if (
        not isinstance(value, str)
        or not value.startswith(marker)
        or value.endswith("/")
        or "\\" in value
        or ".." in value.split("/")
    ):
        raise ValueError(f"{label} escaped the exact result bucket")
    suffix = value[len(marker) :]
    if not suffix:
        raise ValueError(f"{label} is empty")
    return suffix


def _storage_resource(object_prefix: str) -> str:
    return f"projects/_/buckets/{BUCKET}/objects/{object_prefix}/"


def _prefix_condition(prefixes: Sequence[str], *, expires_at: str) -> str:
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise ValueError("IAM object prefixes are empty or duplicated")
    terms = [
        f'resource.name.startsWith("{_storage_resource(prefix)}")'
        for prefix in prefixes
    ]
    names = terms[0] if len(terms) == 1 else "(" + " || ".join(terms) + ")"
    return f'{names} && request.time < timestamp("{expires_at}")'


def _instance_resource(instance_name: str) -> str:
    return f"projects/{PROJECT}/zones/{ZONE}/instances/{instance_name}"


def _instance_condition(
    instance_names: Sequence[str], *, expires_at: str
) -> str:
    if (
        list(instance_names) != [_safe_name(name, "instance") for name in instance_names]
        or len(instance_names) != VM_COUNT
        or len(set(instance_names)) != VM_COUNT
    ):
        raise ValueError("Step 12 instance-name set changed")
    terms = [
        f'resource.name == "{_instance_resource(name)}"'
        for name in instance_names
    ]
    return (
        'resource.type == "compute.googleapis.com/Instance" && '
        f"({' || '.join(terms)}) && "
        f'request.time < timestamp("{expires_at}")'
    )


def _time_condition(expires_at: str) -> str:
    return f'request.time < timestamp("{expires_at}")'


def _condition(title: str, expression: str) -> dict[str, str]:
    if (
        not isinstance(title, str)
        or not title
        or not isinstance(expression, str)
        or not expression
        or "*" in expression
    ):
        raise ValueError("Step 12 IAM condition is not exact")
    return {"title": title, "expression": expression}


def _custom_roles() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "name": f"projects/{PROJECT}/roles/{ROLE_IDS[key]}",
            "stage": "GA",
            "permissions": sorted(CUSTOM_ROLE_PERMISSIONS[key]),
        }
        for key in sorted(ROLE_IDS)
    }


def _binding(
    *,
    purpose: str,
    target: str,
    resource: str,
    role: str,
    member: str,
    condition: Mapping[str, str],
) -> dict[str, Any]:
    if target not in {
        "project",
        "bucket",
        "worker_service_account",
        "controller_service_account",
    }:
        raise ValueError("Step 12 IAM target changed")
    checked_condition = dict(condition)
    _exact(checked_condition, {"title", "expression"}, "IAM condition")
    return {
        "purpose": purpose,
        "target": target,
        "resource": resource,
        "role": role,
        "member": member,
        "condition": checked_condition,
    }


def _normalize_controller_principal(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("controller principal is not a string")
    match = _SERVICE_ACCOUNT_PRINCIPAL.fullmatch(value)
    if (
        match is None
        or match.group("project") != PROJECT
        or value == WORKER_PRINCIPAL
    ):
        raise ValueError("controller principal is not the dedicated project SA")
    return value


def _normalize_initiating_principal(value: Any) -> str:
    if not isinstance(value, str) or _USER_PRINCIPAL.fullmatch(value) is None:
        raise ValueError("initiating principal must be one exact user")
    return value


def _validate_pair(
    pair_contract: Mapping[str, Any],
    *,
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checked_pair = pair_contract_module.validate_pair_contract(
        pair_contract,
        candidate_transport_contract=candidate_transport_contract,
        reference_transport_contract=reference_transport_contract,
        stage1_recovery_receipt=stage1_recovery_receipt,
    )
    candidate = transport.validate_job_contract(candidate_transport_contract)
    reference = transport.validate_job_contract(reference_transport_contract)
    if (
        candidate["adapter_preview"] != reference["adapter_preview"]
        or candidate["remote_layout"] != reference["remote_layout"]
        or candidate["outer_package_manifest"]
        != reference["outer_package_manifest"]
        or candidate["direct_stage_identity_sha256"]
        != reference["direct_stage_identity_sha256"]
    ):
        raise ValueError("Step 12 transport pair lost its shared stage identity")
    return checked_pair, candidate, reference


def _normalize_source(
    pair_contract: Mapping[str, Any],
    *,
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    pair, candidate, reference = _validate_pair(
        pair_contract,
        candidate_transport_contract=candidate_transport_contract,
        reference_transport_contract=reference_transport_contract,
        stage1_recovery_receipt=stage1_recovery_receipt,
    )
    topology = pair["execution_topology"]
    boundary = pair["authorization_boundary"]
    instances = pair["instances"]
    expected_jobs = list(canary_plan.STAGE2_JOB_IDS)
    if (
        pair["stage_id"] != canary_plan.STAGE2_ID
        or pair["run_name"] != canary_plan.STAGE2_RUN_NAME
        or pair["selected_job_ids"] != expected_jobs
        or pair["source_roles"] != ["candidate", "reference"]
        or pair["vm_count"] != VM_COUNT
        or pair["attempt_index"] != AUTHORIZED_ATTEMPT_INDEX
        or topology["actual_machine_type"] != MACHINE_TYPE
        or topology["actual_vcpus_per_vm"] != VCPU_PER_VM
        or topology["total_actual_vcpus"] != TOTAL_REQUESTED_VCPU
        or topology["max_concurrent_vms"] != MAX_CONCURRENT_VMS
        or topology["attempt_limit_per_job"] != MAX_ATTEMPTS_PER_JOB
        or topology["diagnostic_only"] is not True
        or topology["admissible_as_performance_evidence"] is not False
        or boundary["vm_limit"] != VM_COUNT
        or boundary["attempt_limit_per_job"] != MAX_ATTEMPTS_PER_JOB
        or boundary["attempt1_authorized"] is not False
        or boundary["resume_authorized"] is not False
        or boundary["third_vm_authorized"] is not False
        or pair["current_profile_changed"] is not False
        or len(instances) != VM_COUNT
    ):
        raise ValueError("Step 12 pair authorization topology changed")

    remote_layout = candidate["remote_layout"]
    package_prefix = remote_layout["package_prefix"]
    stage_prefix = remote_layout["stage_prefix"]
    result_prefix = remote_layout["result_prefix"]
    package_object_prefix = _parse_gs_prefix(
        package_prefix, label="package prefix"
    )
    stage_object_prefix = _parse_gs_prefix(stage_prefix, label="stage prefix")
    result_object_prefix = _parse_gs_prefix(
        result_prefix, label="result prefix"
    )
    direct_jobs = {
        row["job_id"]: row for row in remote_layout["jobs"]
    }
    contract_by_job = {
        candidate["metadata_binding"]["job_id"]: candidate,
        reference["metadata_binding"]["job_id"]: reference,
    }
    jobs: list[dict[str, Any]] = []
    for position, pair_instance in enumerate(instances):
        job_id = expected_jobs[position]
        role = ("candidate", "reference")[position]
        direct_job = direct_jobs.get(job_id)
        job_contract = contract_by_job.get(job_id)
        if (
            pair_instance["job_id"] != job_id
            or pair_instance["source_role"] != role
            or pair_instance["attempt_index"] != AUTHORIZED_ATTEMPT_INDEX
            or pair_instance["machine_type"] != MACHINE_TYPE
            or direct_job is None
            or job_contract is None
            or job_contract["metadata_binding"]["source_role"] != role
            or job_contract["metadata_binding"]["attempt_index"]
            != AUTHORIZED_ATTEMPT_INDEX
            or pair_instance["result_prefix"] != direct_job["result_prefix"]
            or pair_instance["tree_prefix"] != direct_job["tree_prefix"]
            or pair_instance["done_uri"] != direct_job["done_uri"]
        ):
            raise ValueError("Step 12 exact job/instance mapping changed")
        job_result_object_prefix = _parse_gs_prefix(
            direct_job["result_prefix"], label=f"{job_id} result prefix"
        )
        tree_object_prefix = _parse_gs_prefix(
            direct_job["tree_prefix"], label=f"{job_id} tree prefix"
        )
        done_object_name = _parse_gs_prefix(
            direct_job["done_uri"], label=f"{job_id} DONE object"
        )
        heartbeat_prefix = (
            f"{result_object_prefix}/progress/jobs/{job_id}/heartbeats"
        )
        if (
            not tree_object_prefix.startswith(job_result_object_prefix + "/")
            or done_object_name
            != f"{job_result_object_prefix}/DONE.envelope.json"
        ):
            raise ValueError("Step 12 direct-v1 result layout changed")
        jobs.append(
            {
                "job_id": job_id,
                "source_role": role,
                "instance_name": _safe_name(
                    pair_instance["instance_name"], "pair instance"
                ),
                "attempt_index": AUTHORIZED_ATTEMPT_INDEX,
                "metadata_binding_sha256": job_contract[
                    "metadata_binding_sha256"
                ],
                "result_prefix": direct_job["result_prefix"],
                "result_object_prefix": job_result_object_prefix,
                "heartbeat_object_prefix": heartbeat_prefix,
                "tree_prefix": direct_job["tree_prefix"],
                "done_uri": direct_job["done_uri"],
            }
        )
    instance_names = [row["instance_name"] for row in jobs]
    if len(set(instance_names)) != VM_COUNT:
        raise ValueError("Step 12 exact instance names collided")

    underlying = pair["underlying_frozen_identity"]
    return {
        "pair_contract_schema": pair["schema"],
        "pair_contract_sha256": canonical_sha256(pair),
        "stage_id": pair["stage_id"],
        "run_name": pair["run_name"],
        "selected_job_ids": expected_jobs,
        "source_roles": ["candidate", "reference"],
        "outer_package_identity_sha256": _sha(
            underlying["outer_package_identity_sha256"],
            "outer package identity",
        ),
        "direct_stage_identity_sha256": _sha(
            underlying["direct_stage_identity_sha256"],
            "direct stage identity",
        ),
        "package_prefix": package_prefix,
        "package_object_prefix": package_object_prefix,
        "stage_prefix": stage_prefix,
        "stage_object_prefix": stage_object_prefix,
        "result_prefix": result_prefix,
        "result_object_prefix": result_object_prefix,
        "jobs": jobs,
        "instance_names": instance_names,
        "project": PROJECT,
        "bucket": BUCKET,
        "zone": ZONE,
        "region": REGION,
        "actual_machine_type": MACHINE_TYPE,
        "actual_vcpu_per_vm": VCPU_PER_VM,
        "total_requested_vcpu": TOTAL_REQUESTED_VCPU,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "current_profile_sha256": _sha(
            pair["current_profile_sha256"], "current profile registry"
        ),
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def _build_plan(
    pair_contract: Mapping[str, Any],
    *,
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
    nat_router_resource: str,
    controller_principal: str,
    initiating_principal: str,
    nat_name: str,
) -> dict[str, Any]:
    source = _normalize_source(
        pair_contract,
        candidate_transport_contract=candidate_transport_contract,
        reference_transport_contract=reference_transport_contract,
        stage1_recovery_receipt=stage1_recovery_receipt,
    )
    issued = _strict_int(issued_at_unix_seconds, "issued time", minimum=1)
    expires = _strict_int(expires_at_unix_seconds, "expiry time", minimum=1)
    if expires <= issued or expires - issued > MAX_AUTHORIZATION_WINDOW_SECONDS:
        raise ValueError("Step 12 authorization window is not short-lived")
    expiry = _rfc3339(expires)
    controller_principal = _normalize_controller_principal(
        controller_principal
    )
    initiating_principal = _normalize_initiating_principal(
        initiating_principal
    )
    nat_name = _safe_name(nat_name, "Cloud NAT name")
    router_prefix = f"projects/{PROJECT}/regions/{REGION}/routers/"
    if (
        not isinstance(nat_router_resource, str)
        or not nat_router_resource.startswith(router_prefix)
        or _SAFE_NAME.fullmatch(nat_router_resource[len(router_prefix) :])
        is None
    ):
        raise ValueError("Step 12 NAT router resource changed")

    instance_names = source["instance_names"]
    exact_instance_condition = _instance_condition(
        instance_names, expires_at=expiry
    )
    time_condition = _time_condition(expiry)
    package_prefixes = [source["package_object_prefix"]]
    result_prefixes: list[str] = []
    for job in source["jobs"]:
        result_prefixes.extend(
            [job["result_object_prefix"], job["heartbeat_object_prefix"]]
        )
    read_condition = _prefix_condition(
        [*package_prefixes, *result_prefixes], expires_at=expiry
    )
    create_condition = _prefix_condition(
        result_prefixes, expires_at=expiry
    )
    roles = _custom_roles()
    project_resource = f"projects/{PROJECT}"
    bucket_resource = f"projects/_/buckets/{BUCKET}"
    worker_sa_resource = (
        f"projects/{PROJECT}/serviceAccounts/{WORKER_SERVICE_ACCOUNT}"
    )
    controller_sa_resource = (
        f"projects/{PROJECT}/serviceAccounts/{CONTROLLER_SERVICE_ACCOUNT}"
    )
    bindings = [
        _binding(
            purpose="worker_self_delete",
            target="project",
            resource=project_resource,
            role=roles["worker_self_delete"]["name"],
            member=WORKER_PRINCIPAL,
            condition=_condition(
                "ofc-m31-step12-exact-pair-self-delete-v1",
                exact_instance_condition,
            ),
        ),
        _binding(
            purpose="controller_vm_launch",
            target="project",
            resource=project_resource,
            role=roles["controller_vm_launch"]["name"],
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12-short-lived-pair-launch-v1",
                time_condition,
            ),
        ),
        _binding(
            purpose="controller_instance_lifecycle",
            target="project",
            resource=project_resource,
            role=roles["controller_instance_lifecycle"]["name"],
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12-exact-pair-lifecycle-v1",
                exact_instance_condition,
            ),
        ),
        _binding(
            purpose="controller_zone_operation_reader",
            target="project",
            resource=project_resource,
            role=roles["controller_zone_operation_reader"]["name"],
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12-short-lived-operation-read-v1",
                time_condition,
            ),
        ),
        _binding(
            purpose="controller_service_usage",
            target="project",
            resource=project_resource,
            role="roles/serviceusage.serviceUsageConsumer",
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12-short-lived-service-usage-v1",
                time_condition,
            ),
        ),
        _binding(
            purpose="controller_package_and_result_reader",
            target="bucket",
            resource=bucket_resource,
            role=roles["worker_object_reader"]["name"],
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12-controller-exact-read-v1",
                read_condition,
            ),
        ),
        _binding(
            purpose="worker_package_and_result_reader",
            target="bucket",
            resource=bucket_resource,
            role=roles["worker_object_reader"]["name"],
            member=WORKER_PRINCIPAL,
            condition=_condition(
                "ofc-m31-step12-worker-exact-read-v1",
                read_condition,
            ),
        ),
        _binding(
            purpose="worker_result_creator",
            target="bucket",
            resource=bucket_resource,
            role=roles["worker_result_creator"]["name"],
            member=WORKER_PRINCIPAL,
            condition=_condition(
                "ofc-m31-step12-worker-exact-create-v1",
                create_condition,
            ),
        ),
        _binding(
            purpose="controller_worker_act_as",
            target="worker_service_account",
            resource=worker_sa_resource,
            role="roles/iam.serviceAccountUser",
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12-short-lived-worker-actas-v1",
                time_condition,
            ),
        ),
        _binding(
            purpose="initiator_controller_token_creator",
            target="controller_service_account",
            resource=controller_sa_resource,
            role="roles/iam.serviceAccountTokenCreator",
            member=initiating_principal,
            condition=_condition(
                "ofc-m31-step12-short-lived-controller-token-v1",
                time_condition,
            ),
        ),
    ]
    purposes = [row["purpose"] for row in bindings]
    if (
        len(bindings) != 10
        or len(set(purposes)) != len(bindings)
        or "controller_package_creator" in purposes
    ):
        raise AssertionError("Step 12 temporary IAM binding set changed")

    allowed_launches = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "instance_name": row["instance_name"],
            "attempt_index": AUTHORIZED_ATTEMPT_INDEX,
            "project": PROJECT,
            "zone": ZONE,
            "machine_type": MACHINE_TYPE,
            "vcpu": VCPU_PER_VM,
            "worker_service_account": WORKER_SERVICE_ACCOUNT,
            "oauth_scopes": [REQUIRED_WORKER_OAUTH_SCOPE],
            "provisioning_model": "SPOT",
            "reservation_affinity": "NO_RESERVATION",
            "max_run_duration_seconds": MAX_RUN_DURATION_SECONDS,
            "automatic_restart": False,
            "on_host_maintenance": "TERMINATE",
            "boot_disk_auto_delete": True,
            "external_access_configs": [],
            "metadata_binding_sha256": row["metadata_binding_sha256"],
        }
        for row in source["jobs"]
    ]
    for request in allowed_launches:
        request["request_contract_sha256"] = canonical_sha256(request)

    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_pair_contract": source,
        "authorization_window": {
            "issued_at_unix_seconds": issued,
            "expires_at_unix_seconds": expires,
            "expires_at_rfc3339": expiry,
            "maximum_window_seconds": MAX_AUTHORIZATION_WINDOW_SECONDS,
            "minimum_remaining_at_gate_seconds": (
                MIN_AUTHORIZATION_REMAINING_SECONDS
            ),
        },
        "principals": {
            "controller_principal": controller_principal,
            "controller_service_account": CONTROLLER_SERVICE_ACCOUNT,
            "worker_principal": WORKER_PRINCIPAL,
            "worker_service_account": WORKER_SERVICE_ACCOUNT,
            "initiating_principal": initiating_principal,
            "dedicated_controller_required": True,
            "worker_identity_shared_by_exact_pair": True,
            "owner_or_editor_execution_forbidden": True,
            "public_principals_forbidden": [
                "allUsers",
                "allAuthenticatedUsers",
            ],
        },
        "iam_contract": {
            "custom_roles": roles,
            "binding_specs": bindings,
            "binding_count": len(bindings),
            "controller_package_create_binding_required": False,
            "immutable_existing_package_reused": True,
            "worker_object_list_forbidden": True,
            "worker_object_delete_forbidden": True,
            "worker_object_overwrite_forbidden": True,
            "object_read_prefixes": [
                source["package_prefix"],
                *[
                    prefix
                    for row in source["jobs"]
                    for prefix in (
                        row["result_prefix"],
                        "gs://"
                        f"{BUCKET}/{row['heartbeat_object_prefix']}",
                    )
                ],
            ],
            "object_create_prefixes": [
                prefix
                for row in source["jobs"]
                for prefix in (
                    row["result_prefix"],
                    f"gs://{BUCKET}/{row['heartbeat_object_prefix']}",
                )
            ],
            "instance_condition": exact_instance_condition,
            "all_bindings_time_bounded": True,
            "project_folder_org_effective_analysis_required": True,
            "basic_role_service_account_act_as_forbidden": True,
        },
        "instance_contract": {
            "allowed_launch_requests": allowed_launches,
            "authorized_instance_names": instance_names,
            "authorized_attempt_index": AUTHORIZED_ATTEMPT_INDEX,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
            "attempt1_authorized": False,
            "resume_authorized": False,
            "third_vm_authorized": False,
            "required_oauth_scopes": [REQUIRED_WORKER_OAUTH_SCOPE],
            "external_access_configs_forbidden": True,
            "spot_required": True,
            "reservation_consumption_forbidden": True,
        },
        "capacity_contract": {
            "project": PROJECT,
            "zone": ZONE,
            "region": REGION,
            "machine_type": MACHINE_TYPE,
            "vcpu_per_vm": VCPU_PER_VM,
            "requested_concurrent_vms": MAX_CONCURRENT_VMS,
            "requested_c4_vcpu": TOTAL_REQUESTED_VCPU,
            "quota_metrics": [
                "CPUS",
                "CPUS_ALL_REGIONS",
                "CPUS_PER_VM_FAMILY_C4",
                "PREEMPTIBLE_CPUS",
            ],
            "authoritative_inventory_endpoints": [
                "compute_aggregated_instances",
                "compute_aggregated_reservations",
                "compute_aggregated_node_groups",
                "compute_aggregated_future_reservations",
            ],
            "authoritative_inventory_complete_required": True,
            "authoritative_inventory_same_scope_set_required": True,
            "authoritative_inventory_unreachable_count_required": 0,
            "authoritative_inventory_page_tokens_exhausted_required": True,
            "authoritative_noninstance_inventory_empty_required": True,
            "authoritative_transcript_endpoint_digest_map_required": True,
            "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
            "all_instance_inventory_statuses_required": True,
            "target_instance_names": instance_names,
            "target_name_collision_forbidden": True,
            "nonterminated_c4_interference_forbidden": True,
            "quota_must_cover_current_usage_plus_requested_vcpu": True,
            "stale_quota_evidence_forbidden": True,
            "spot_stock_only_provable_by_insert": True,
            "network_resource": DEFAULT_NETWORK_RESOURCE,
            "subnetwork_resource": DEFAULT_SUBNETWORK_RESOURCE,
            "external_ip_forbidden": True,
            "nat_name": nat_name,
            "nat_router_resource": nat_router_resource,
            "nat_source_subnetwork_ip_ranges": (
                "ALL_SUBNETWORKS_ALL_IP_RANGES"
            ),
        },
        "post_claim_revoke_contract": {
            "required_claim_count": VM_COUNT,
            "required_claim_instance_names": instance_names,
            "each_claim_requires_provider_instance_id": True,
            "each_claim_requires_metadata_fingerprint_cas": True,
            "each_claim_requires_provider_readback": True,
            "revoke_after_claim_count": VM_COUNT,
            "revoke_after_first_claim_forbidden": True,
            "revoke_before_all_claims_forbidden": True,
            "launch_binding_must_be_removed": True,
            "worker_actas_binding_must_be_removed": True,
            "post_revoke_policy_readback_required": True,
            "additional_insert_after_revoke_forbidden": True,
        },
        "cleanup_contract": {
            "required_on_success": True,
            "required_on_any_failure": True,
            "exact_instance_names": instance_names,
            "exact_disk_names": instance_names,
            "all_instance_delete_attempts_independent": True,
            "all_disk_delete_attempts_independent": True,
            "cleanup_errors_aggregated": True,
            "iam_revoked_before_user_credential_compute_cleanup": True,
            "all_temporary_iam_bindings_removed": True,
            "all_target_policies_read_back": True,
            "final_instance_get_status_required": 404,
            "final_disk_get_status_required": 404,
            "final_targeted_iam_binding_count_required": 0,
            "package_and_results_retained": True,
            "current_profile_sha256_required": source[
                "current_profile_sha256"
            ],
            "current_profile_changed": False,
        },
        "gate_semantics": {
            "local_validation_only": True,
            "cloud_api_calls_performed": False,
            "cloud_mutation_performed": False,
            "launch_authorized": False,
            "passing_gate_proves_spot_stock": False,
            "fresh_get_only_observation_required_before_launch": True,
            "separate_explicit_execution_confirmation_required": True,
            "diagnostic_only": True,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def build_step12_pair_gate_plan(
    pair_contract: Mapping[str, Any],
    *,
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
    nat_router_resource: str,
    controller_principal: str = DEFAULT_CONTROLLER_PRINCIPAL,
    initiating_principal: str = DEFAULT_INITIATING_PRINCIPAL,
    nat_name: str = DEFAULT_NAT_NAME,
) -> dict[str, Any]:
    """Build a sealed local-only Step 12 exact-pair authorization plan."""

    value = _build_plan(
        pair_contract,
        candidate_transport_contract=candidate_transport_contract,
        reference_transport_contract=reference_transport_contract,
        stage1_recovery_receipt=stage1_recovery_receipt,
        issued_at_unix_seconds=issued_at_unix_seconds,
        expires_at_unix_seconds=expires_at_unix_seconds,
        nat_router_resource=nat_router_resource,
        controller_principal=controller_principal,
        initiating_principal=initiating_principal,
        nat_name=nat_name,
    )
    return validate_step12_pair_gate_plan(
        value,
        pair_contract=pair_contract,
        candidate_transport_contract=candidate_transport_contract,
        reference_transport_contract=reference_transport_contract,
        stage1_recovery_receipt=stage1_recovery_receipt,
    )


def validate_step12_pair_gate_plan(
    plan: Mapping[str, Any],
    *,
    pair_contract: Mapping[str, Any],
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless every Step 12 pair safety boundary is unchanged."""

    checked = deepcopy(dict(plan))
    _exact(checked, _PLAN_FIELDS, "Step 12 pair gate plan")
    digest = _sha(checked.pop("plan_sha256"), "Step 12 plan digest")
    if canonical_sha256(checked) != digest:
        raise ValueError("Step 12 pair gate plan digest mismatch")
    if checked["schema"] != SCHEMA or checked["status"] != STATUS:
        raise ValueError("Step 12 pair gate plan schema/status changed")

    window = checked.get("authorization_window")
    capacity = checked.get("capacity_contract")
    principals = checked.get("principals")
    if (
        not isinstance(window, Mapping)
        or not isinstance(capacity, Mapping)
        or not isinstance(principals, Mapping)
    ):
        raise ValueError("Step 12 pair gate plan sections are missing")
    expected = _build_plan(
        pair_contract,
        candidate_transport_contract=candidate_transport_contract,
        reference_transport_contract=reference_transport_contract,
        stage1_recovery_receipt=stage1_recovery_receipt,
        issued_at_unix_seconds=window.get("issued_at_unix_seconds"),
        expires_at_unix_seconds=window.get("expires_at_unix_seconds"),
        nat_router_resource=capacity.get("nat_router_resource"),
        controller_principal=principals.get("controller_principal"),
        initiating_principal=principals.get("initiating_principal"),
        nat_name=capacity.get("nat_name"),
    )
    expected_without_digest = dict(expected)
    expected_digest = expected_without_digest.pop("plan_sha256")
    if checked != expected_without_digest or digest != expected_digest:
        raise ValueError("Step 12 pair gate plan changed")

    instance_contract = checked["instance_contract"]
    revoke = checked["post_claim_revoke_contract"]
    cleanup = checked["cleanup_contract"]
    semantics = checked["gate_semantics"]
    names = checked["source_pair_contract"]["instance_names"]
    if (
        instance_contract["authorized_instance_names"] != names
        or instance_contract["max_concurrent_vms"] != VM_COUNT
        or instance_contract["max_attempts_per_job"]
        != MAX_ATTEMPTS_PER_JOB
        or instance_contract["attempt1_authorized"] is not False
        or capacity["machine_type"] != MACHINE_TYPE
        or capacity["requested_c4_vcpu"] != TOTAL_REQUESTED_VCPU
        or capacity["quota_metrics"]
        != [
            "CPUS",
            "CPUS_ALL_REGIONS",
            "CPUS_PER_VM_FAMILY_C4",
            "PREEMPTIBLE_CPUS",
        ]
        or capacity["authoritative_inventory_endpoints"]
        != [
            "compute_aggregated_instances",
            "compute_aggregated_reservations",
            "compute_aggregated_node_groups",
            "compute_aggregated_future_reservations",
        ]
        or capacity["authoritative_inventory_complete_required"] is not True
        or capacity[
            "authoritative_inventory_same_scope_set_required"
        ]
        is not True
        or capacity[
            "authoritative_inventory_unreachable_count_required"
        ]
        != 0
        or capacity[
            "authoritative_inventory_page_tokens_exhausted_required"
        ]
        is not True
        or capacity[
            "authoritative_noninstance_inventory_empty_required"
        ]
        is not True
        or capacity[
            "authoritative_transcript_endpoint_digest_map_required"
        ]
        is not True
        or revoke["required_claim_instance_names"] != names
        or revoke["revoke_after_claim_count"] != VM_COUNT
        or revoke["revoke_after_first_claim_forbidden"] is not True
        or cleanup["exact_instance_names"] != names
        or cleanup["exact_disk_names"] != names
        or cleanup["final_targeted_iam_binding_count_required"] != 0
        or any(
            semantics[key] is not False
            for key in (
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("Step 12 pair gate safety boundary changed")
    return {**checked, "plan_sha256": digest}


# Short aliases for callers that are already scoped to Step 12.
build_gate_plan = build_step12_pair_gate_plan
validate_gate_plan = validate_step12_pair_gate_plan


__all__ = [
    "AUTHORIZED_ATTEMPT_INDEX",
    "MAX_ATTEMPTS_PER_JOB",
    "MAX_CONCURRENT_VMS",
    "MACHINE_TYPE",
    "SCHEMA",
    "STATUS",
    "TOTAL_REQUESTED_VCPU",
    "VCPU_PER_VM",
    "VM_COUNT",
    "build_gate_plan",
    "build_step12_pair_gate_plan",
    "canonical_bytes",
    "canonical_sha256",
    "validate_gate_plan",
    "validate_step12_pair_gate_plan",
]
