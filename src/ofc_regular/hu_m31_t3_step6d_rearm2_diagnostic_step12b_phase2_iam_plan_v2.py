"""Offline, exact Phase-2 IAM plan for the fresh Step12b VM pair.

Phase 1 (the short-lived TokenCreator bootstrap used to mint one fixed
controller token) is intentionally outside this module.  This module starts
from a fully validated Step12b deployment contract and reconstructs the exact
five controller bindings and three worker bindings needed by attempt 0.

The plan is data only.  It cannot call Google Cloud, mutate an IAM policy,
create a VM, or authorize a launch by itself.  Existing custom roles are
required to match their frozen v1 names and permission sets; role creation or
role modification is forbidden.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_iam_plan_v2"
)
STATUS = "offline_exact_phase2_iam_plan_ready"
PHASE = "phase2_after_fixed_token_before_pair_release"
READBACK_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_iam_exact_readback_receipt_v2"
)

PROJECT = payload_transport.PROJECT
BUCKET = payload_transport.BUCKET
ZONE = payload_transport.ZONE
REGION = ZONE.rsplit("-", 1)[0]
WORKER_SERVICE_ACCOUNT = payload_transport.WORKER_SERVICE_ACCOUNT
WORKER_PRINCIPAL = f"serviceAccount:{WORKER_SERVICE_ACCOUNT}"
REQUIRED_WORKER_OAUTH_SCOPE = (
    "https://www.googleapis.com/auth/cloud-platform"
)

MACHINE_TYPE = deployment_v2.ACTUAL_MACHINE_TYPE
VM_COUNT = deployment_v2.VM_COUNT
ATTEMPT_INDEX = deployment_v2.ATTEMPT_INDEX
MAX_ATTEMPTS = deployment_v2.MAX_ATTEMPTS
MAX_AUTHORIZATION_WINDOW_SECONDS = 7_200
CONTROLLER_BINDING_COUNT = 5
WORKER_BINDING_COUNT = 3

TOKEN_CREATOR_ROLE = "roles/iam.serviceAccountTokenCreator"
SERVICE_USAGE_ROLE = "roles/serviceusage.serviceUsageConsumer"
WORKER_ACT_AS_ROLE = "roles/iam.serviceAccountUser"
OLD_DIRECT_NAMESPACE = payload_transport.DIRECT_NAMESPACE

CUSTOM_ROLE_IDS = {
    "worker_object_reader": "ofcM31T3ObjectReaderV1",
    "worker_result_creator": "ofcM31T3ResultCreatorV1",
    "worker_self_delete": "ofcM31T3SelfDeleteV1",
    "controller_vm_launch": "ofcM31T3VmLaunchV1",
    "controller_instance_lifecycle": "ofcM31T3VmLifecycleV1",
    "controller_zone_operation_reader": "ofcM31T3ZoneOperationReaderV1",
}

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
    "controller_zone_operation_reader": (
        "compute.zoneOperations.get",
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GCE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_PLAN_FIELDS = {
    "schema",
    "status",
    "phase",
    "source_deployment",
    "authorization_window",
    "principals",
    "custom_role_readback_contract",
    "phase2_bindings",
    "controller_release_removal_group",
    "worker_final_cleanup_removal_group",
    "object_boundary",
    "instance_boundary",
    "cleanup_order",
    "capabilities",
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


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(f"{label} is not an exact bounded integer")
    return value


def _rfc3339(unix_seconds: int) -> str:
    return (
        datetime.fromtimestamp(unix_seconds, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _custom_roles() -> list[dict[str, Any]]:
    return [
        {
            "purpose": purpose,
            "name": (
                f"projects/{PROJECT}/roles/{CUSTOM_ROLE_IDS[purpose]}"
            ),
            "stage": "GA",
            "included_permissions": sorted(
                CUSTOM_ROLE_PERMISSIONS[purpose]
            ),
            "existing_role_get_readback_required": True,
            "role_create_authorized": False,
            "role_patch_authorized": False,
        }
        for purpose in sorted(CUSTOM_ROLE_IDS)
    ]


def _role_name(purpose: str) -> str:
    try:
        role_id = CUSTOM_ROLE_IDS[purpose]
    except KeyError as exc:
        raise ValueError("unknown frozen custom-role purpose") from exc
    return f"projects/{PROJECT}/roles/{role_id}"


def _condition(title: str, expression: str) -> dict[str, str]:
    if (
        not isinstance(title, str)
        or not title
        or not isinstance(expression, str)
        or not expression
        or "*" in expression
    ):
        raise ValueError("Phase2 IAM condition is not exact")
    return {"title": title, "expression": expression}


def _time_condition(expires_at_rfc3339: str) -> str:
    return f'request.time < timestamp("{expires_at_rfc3339}")'


def _instance_condition(
    instance_names: Sequence[str],
    *,
    expires_at_rfc3339: str,
) -> str:
    names = list(instance_names)
    if (
        len(names) != VM_COUNT
        or len(set(names)) != VM_COUNT
        or any(
            not isinstance(name, str)
            or _GCE_NAME.fullmatch(name) is None
            for name in names
        )
    ):
        raise ValueError("Phase2 exact instance set changed")
    terms = [
        (
            'resource.name == '
            f'"projects/{PROJECT}/zones/{ZONE}/instances/{name}"'
        )
        for name in names
    ]
    return (
        'resource.type == "compute.googleapis.com/Instance" && '
        f"({' || '.join(terms)}) && "
        f'{_time_condition(expires_at_rfc3339)}'
    )


def _gs_object_prefix(value: Any, label: str) -> str:
    marker = f"gs://{BUCKET}/"
    if (
        not isinstance(value, str)
        or not value.startswith(marker)
        or value.endswith("/")
        or "\\" in value
        or ".." in value.split("/")
    ):
        raise ValueError(f"{label} escaped the exact bucket")
    suffix = value[len(marker) :]
    if not suffix:
        raise ValueError(f"{label} is empty")
    return suffix


def _prefix_condition(
    gs_prefixes: Sequence[str],
    *,
    expires_at_rfc3339: str,
) -> str:
    prefixes = list(gs_prefixes)
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise ValueError("Phase2 storage prefixes changed")
    terms = []
    for position, prefix in enumerate(prefixes):
        suffix = _gs_object_prefix(prefix, f"storage prefix {position}")
        resource = f"projects/_/buckets/{BUCKET}/objects/{suffix}/"
        terms.append(f'resource.name.startsWith("{resource}")')
    expression = (
        terms[0]
        if len(terms) == 1
        else "(" + " || ".join(terms) + ")"
    )
    return (
        f"{expression} && "
        f'{_time_condition(expires_at_rfc3339)}'
    )


def _binding(
    *,
    purpose: str,
    actor_group: str,
    target: str,
    resource: str,
    role: str,
    member: str,
    condition: Mapping[str, str],
) -> dict[str, Any]:
    if actor_group not in {"controller", "worker"}:
        raise ValueError("Phase2 binding actor group changed")
    if target not in {"project", "bucket", "worker_service_account"}:
        raise ValueError("Phase2 binding target changed")
    checked_condition = dict(condition)
    _exact(
        checked_condition,
        {"title", "expression"},
        "Phase2 binding condition",
    )
    body = {
        "purpose": purpose,
        "phase": PHASE,
        "actor_group": actor_group,
        "target": target,
        "resource": resource,
        "role": role,
        "member": member,
        "condition": checked_condition,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def _binding_identities(
    bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    identities = [
        {
            "purpose": row["purpose"],
            "target": row["target"],
            "resource": row["resource"],
            "role": row["role"],
            "principal": row["member"],
            "binding_sha256": row["binding_sha256"],
        }
        for row in bindings
    ]
    identities.sort(
        key=lambda row: (
            row["resource"],
            row["role"],
            row["purpose"],
        )
    )
    pairs = [(row["resource"], row["role"]) for row in identities]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Phase2 binding readback identity collided")
    return identities


def _result_prefixes(
    deployment: Mapping[str, Any],
) -> list[str]:
    prefixes: list[str] = []
    for job in deployment["remote_layout"]["jobs"]:
        result_prefix = job["result_prefix"]
        heartbeats = job["heartbeat_uris"]
        if (
            not isinstance(heartbeats, list)
            or not heartbeats
            or any(
                not isinstance(uri, str) or "/" not in uri
                for uri in heartbeats
            )
        ):
            raise ValueError("Phase2 heartbeat layout changed")
        heartbeat_prefixes = {
            uri.rsplit("/", 1)[0] for uri in heartbeats
        }
        if len(heartbeat_prefixes) != 1:
            raise ValueError("Phase2 heartbeat prefix changed")
        heartbeat_prefix = next(iter(heartbeat_prefixes))
        prefixes.extend([result_prefix, heartbeat_prefix])
    if len(prefixes) != VM_COUNT * 2 or len(prefixes) != len(set(prefixes)):
        raise ValueError("Phase2 result create prefix set changed")
    for prefix in prefixes:
        _gs_object_prefix(prefix, "Phase2 result prefix")
    return prefixes


def _validated_sources(
    deployment_contract: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deployment = deployment_v2.validate_step12b_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    candidate = payload_transport.validate_job_contract(
        candidate_payload_contract
    )
    reference = payload_transport.validate_job_contract(
        reference_payload_contract
    )
    if (
        candidate["remote_layout"] != reference["remote_layout"]
        or candidate["remote_layout"]["package_prefix"]
        != deployment["payload_binding"]["package_prefix"]
        or candidate["remote_layout"]["result_prefix"]
        != reference["remote_layout"]["result_prefix"]
    ):
        raise ValueError("Phase2 immutable payload layout changed")
    return deployment, candidate, reference


def _build_plan(
    deployment_contract: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
) -> dict[str, Any]:
    deployment, candidate, _ = _validated_sources(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    issued = _strict_int(
        issued_at_unix_seconds, "Phase2 issued time", minimum=1
    )
    expires = _strict_int(
        expires_at_unix_seconds, "Phase2 expiry time", minimum=1
    )
    if (
        expires <= issued
        or expires - issued > MAX_AUTHORIZATION_WINDOW_SECONDS
    ):
        raise ValueError("Phase2 authorization window is not short-lived")
    expiry = _rfc3339(expires)

    controller = deployment["controller_service_account"]
    controller_principal = controller["principal"]
    if (
        controller["run_scoped"] is not True
        or controller["legacy_shared_controller_reused"] is not False
        or controller_principal == WORKER_PRINCIPAL
    ):
        raise ValueError("Phase2 controller is not the fresh run-scoped SA")

    instance_names = [
        row["instance_name"] for row in deployment["instances"]
    ]
    instance_condition = _instance_condition(
        instance_names, expires_at_rfc3339=expiry
    )
    time_condition = _time_condition(expiry)
    package_prefix = deployment["payload_binding"]["package_prefix"]
    bootstrap_source_prefix = (
        f"gs://{BUCKET}/{deployment_v2.DIRECT_NAMESPACE}/"
        "bootstrap-sources/"
        f"{deployment['deployment_contract_sha256']}"
    )
    result_prefixes = _result_prefixes(deployment)
    reader_prefixes = [
        package_prefix,
        bootstrap_source_prefix,
        *result_prefixes,
    ]

    old_payload_result_prefix = candidate["remote_layout"]["result_prefix"]
    old_direct_stage_root = (
        f"gs://{BUCKET}/{OLD_DIRECT_NAMESPACE}/stages"
    )
    if (
        any(
            prefix.startswith(old_direct_stage_root + "/")
            for prefix in result_prefixes
        )
        or old_payload_result_prefix in result_prefixes
        or package_prefix in result_prefixes
        or bootstrap_source_prefix in result_prefixes
        or any(
            not prefix.startswith(
                deployment["remote_layout"]["result_prefix"] + "/"
            )
            for prefix in result_prefixes
        )
    ):
        raise ValueError("Phase2 writable result namespace escaped direct-v2")

    project_resource = f"projects/{PROJECT}"
    bucket_resource = f"projects/_/buckets/{BUCKET}"
    worker_sa_resource = (
        f"projects/{PROJECT}/serviceAccounts/{WORKER_SERVICE_ACCOUNT}"
    )

    controller_bindings = [
        _binding(
            purpose="controller_vm_launch",
            actor_group="controller",
            target="project",
            resource=project_resource,
            role=_role_name("controller_vm_launch"),
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12b-phase2-launch-v2",
                time_condition,
            ),
        ),
        _binding(
            purpose="controller_instance_lifecycle",
            actor_group="controller",
            target="project",
            resource=project_resource,
            role=_role_name("controller_instance_lifecycle"),
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12b-phase2-exact-pair-lifecycle-v2",
                instance_condition,
            ),
        ),
        _binding(
            purpose="controller_zone_operation_reader",
            actor_group="controller",
            target="project",
            resource=project_resource,
            role=_role_name("controller_zone_operation_reader"),
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12b-phase2-zone-operation-read-v2",
                time_condition,
            ),
        ),
        _binding(
            purpose="controller_service_usage",
            actor_group="controller",
            target="project",
            resource=project_resource,
            role=SERVICE_USAGE_ROLE,
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12b-phase2-service-usage-v2",
                time_condition,
            ),
        ),
        _binding(
            purpose="controller_worker_act_as",
            actor_group="controller",
            target="worker_service_account",
            resource=worker_sa_resource,
            role=WORKER_ACT_AS_ROLE,
            member=controller_principal,
            condition=_condition(
                "ofc-m31-step12b-phase2-worker-actas-v2",
                time_condition,
            ),
        ),
    ]
    worker_bindings = [
        _binding(
            purpose="worker_self_delete",
            actor_group="worker",
            target="project",
            resource=project_resource,
            role=_role_name("worker_self_delete"),
            member=WORKER_PRINCIPAL,
            condition=_condition(
                "ofc-m31-step12b-phase2-exact-pair-self-delete-v2",
                instance_condition,
            ),
        ),
        _binding(
            purpose="worker_package_and_result_reader",
            actor_group="worker",
            target="bucket",
            resource=bucket_resource,
            role=_role_name("worker_object_reader"),
            member=WORKER_PRINCIPAL,
            condition=_condition(
                "ofc-m31-step12b-phase2-exact-read-v2",
                _prefix_condition(
                    reader_prefixes,
                    expires_at_rfc3339=expiry,
                ),
            ),
        ),
        _binding(
            purpose="worker_result_creator",
            actor_group="worker",
            target="bucket",
            resource=bucket_resource,
            role=_role_name("worker_result_creator"),
            member=WORKER_PRINCIPAL,
            condition=_condition(
                "ofc-m31-step12b-phase2-direct-v2-create-v2",
                _prefix_condition(
                    result_prefixes,
                    expires_at_rfc3339=expiry,
                ),
            ),
        ),
    ]
    if (
        len(controller_bindings) != CONTROLLER_BINDING_COUNT
        or len(worker_bindings) != WORKER_BINDING_COUNT
        or any(
            row["role"] == TOKEN_CREATOR_ROLE
            for row in [*controller_bindings, *worker_bindings]
        )
        or any(
            row["target"] == "bucket" for row in controller_bindings
        )
    ):
        raise AssertionError("Phase2 least-privilege binding set changed")

    controller_identities = _binding_identities(controller_bindings)
    worker_identities = _binding_identities(worker_bindings)
    launch_requests = []
    for job, instance in zip(
        deployment["remote_layout"]["jobs"],
        deployment["instances"],
        strict=True,
    ):
        request = {
            "external_job_id": job["job_id"],
            "inner_job_id": instance["inner_job_id"],
            "source_role": instance["source_role"],
            "instance_name": instance["instance_name"],
            "boot_disk_name": instance["instance_name"],
            "attempt_index": ATTEMPT_INDEX,
            "project": PROJECT,
            "zone": ZONE,
            "machine_type": MACHINE_TYPE,
            "worker_service_account": WORKER_SERVICE_ACCOUNT,
            "oauth_scopes": [REQUIRED_WORKER_OAUTH_SCOPE],
            "provisioning_model": "SPOT",
            "reservation_affinity": "NO_RESERVATION",
            "automatic_restart": False,
            "on_host_maintenance": "TERMINATE",
            "boot_disk_auto_delete": True,
            "external_access_configs": [],
        }
        launch_requests.append(
            {**request, "request_sha256": canonical_sha256(request)}
        )

    source = {
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "run_identity_sha256": deployment["run_identity_sha256"],
        "direct_stage_identity_sha256": deployment[
            "direct_stage_identity_sha256"
        ],
        "payload_binding_sha256": deployment["payload_binding"][
            "payload_binding_sha256"
        ],
        "outer_package_identity_sha256": deployment["payload_binding"][
            "outer_package_identity_sha256"
        ],
        "run_name": deployment["run_name"],
        "stage_id": deployment["stage_id"],
        "selected_job_ids": list(deployment["selected_job_ids"]),
        "source_roles": list(deployment["source_roles"]),
        "attempt_index": ATTEMPT_INDEX,
        "instance_names": instance_names,
        "package_prefix": package_prefix,
        "deployment_result_prefix": deployment["remote_layout"][
            "result_prefix"
        ],
        "controller_service_account": copy.deepcopy(controller),
    }
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "phase": PHASE,
        "source_deployment": source,
        "authorization_window": {
            "issued_at_unix_seconds": issued,
            "expires_at_unix_seconds": expires,
            "expires_at_rfc3339": expiry,
            "maximum_window_seconds": MAX_AUTHORIZATION_WINDOW_SECONDS,
        },
        "principals": {
            "controller_service_account": copy.deepcopy(controller),
            "controller_principal": controller_principal,
            "worker_service_account": WORKER_SERVICE_ACCOUNT,
            "worker_principal": WORKER_PRINCIPAL,
            "run_scoped_controller_required": True,
            "shared_worker_for_exact_pair": True,
            "controller_gcs_access_required": False,
            "user_clients_monitor_results_after_pair_release": True,
        },
        "custom_role_readback_contract": {
            "requirements": _custom_roles(),
            "requirement_count": len(CUSTOM_ROLE_IDS),
            "all_existing_role_get_readbacks_required": True,
            "exact_name_stage_permissions_required": True,
            "role_create_authorized": False,
            "role_patch_authorized": False,
            "role_delete_authorized": False,
        },
        "phase2_bindings": {
            "controller": controller_bindings,
            "worker": worker_bindings,
            "controller_binding_count": CONTROLLER_BINDING_COUNT,
            "worker_binding_count": WORKER_BINDING_COUNT,
            "total_binding_count": (
                CONTROLLER_BINDING_COUNT + WORKER_BINDING_COUNT
            ),
            "phase1_token_creator_binding_included": False,
            "token_creator_role_forbidden": TOKEN_CREATOR_ROLE,
            "phase1_token_barrier_separate": True,
            "all_bindings_time_bounded": True,
            "cloud_mutation_performed": False,
        },
        "controller_release_removal_group": {
            "binding_identities": controller_identities,
            "binding_identities_sha256": canonical_sha256(
                controller_identities
            ),
            "binding_count": CONTROLLER_BINDING_COUNT,
            "all_removed_after_both_claim_readbacks": True,
            "removal_before_first_claim_forbidden": True,
            "removal_after_only_one_claim_forbidden": True,
            "exact_zero_readback_required_before_pair_release": True,
            "controller_service_account_delete_after_zero_required": True,
        },
        "worker_final_cleanup_removal_group": {
            "binding_identities": worker_identities,
            "binding_identities_sha256": canonical_sha256(
                worker_identities
            ),
            "binding_count": WORKER_BINDING_COUNT,
            "remain_until_worker_done_or_failure_cleanup": True,
            "exact_zero_readback_required_at_final_cleanup": True,
        },
        "object_boundary": {
            "immutable_package_read_prefix": package_prefix,
            "fresh_bootstrap_source_read_prefix": (
                bootstrap_source_prefix
            ),
            "fresh_bootstrap_source_object_count": 3,
            "role_bootstrap_source_object_count": 2,
            "direct_v2_result_read_prefixes": result_prefixes,
            "direct_v2_result_create_prefixes": result_prefixes,
            "old_payload_result_prefix": old_payload_result_prefix,
            "old_direct_stage_root": old_direct_stage_root,
            "package_write_authorized": False,
            "bootstrap_source_write_authorized": False,
            "old_payload_result_write_authorized": False,
            "old_direct_namespace_write_authorized": False,
            "controller_object_read_authorized": False,
            "controller_object_create_authorized": False,
            "worker_object_list_authorized": False,
            "worker_object_delete_authorized": False,
            "worker_object_overwrite_authorized": False,
            "worker_create_if_generation_match_required": 0,
            "worker_create_generation_sha_bytes_readback_required": True,
        },
        "instance_boundary": {
            "allowed_launch_requests": launch_requests,
            "authorized_instance_names": instance_names,
            "authorized_disk_names": instance_names,
            "authorized_attempt_index": ATTEMPT_INDEX,
            "max_attempts_per_job": MAX_ATTEMPTS,
            "max_concurrent_vms": VM_COUNT,
            "machine_type": MACHINE_TYPE,
            "vm_count": VM_COUNT,
            "attempt1_authorized": False,
            "resume_authorized": False,
            "third_vm_authorized": False,
            "external_ip_authorized": False,
            "spot_required": True,
            "exact_instance_and_disk_get_404_required_before_insert": True,
        },
        "cleanup_order": {
            "success": [
                "controller_bindings_zero_after_two_claims",
                "controller_service_account_deleted_and_get404",
                "pair_release",
                "worker_bindings_zero_after_done",
                "exact_instance_and_disk_get404",
            ],
            "failure": [
                "controller_bindings_zero",
                "worker_bindings_zero",
                "controller_service_account_deleted_and_get404",
                "user_credential_exact_instance_and_disk_cleanup",
                "exact_instance_and_disk_get404",
            ],
            "dangerous_iam_removed_before_user_compute_cleanup": True,
        },
        "capabilities": {
            "local_validation_only": True,
            "cloud_api_calls_performed": False,
            "cloud_mutation_performed": False,
            "iam_policy_write_authorized": False,
            "vm_create_authorized": False,
            "launch_authorized": False,
            "phase1_token_creator_included": False,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def build_step12b_phase2_iam_plan(
    deployment_contract: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
) -> dict[str, Any]:
    """Build a non-mutating exact Phase-2 plan."""

    return _build_plan(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        issued_at_unix_seconds=issued_at_unix_seconds,
        expires_at_unix_seconds=expires_at_unix_seconds,
    )


def validate_step12b_phase2_iam_plan(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, Any]:
    """Reconstruct and exactly compare every Phase-2 IAM field."""

    checked = copy.deepcopy(dict(value))
    _exact(checked, _PLAN_FIELDS, "Step12b Phase2 IAM plan")
    supplied_sha = _sha(
        checked.pop("plan_sha256", None),
        "Step12b Phase2 IAM plan digest",
    )
    if canonical_sha256(checked) != supplied_sha:
        raise ValueError("Step12b Phase2 IAM plan digest changed")
    if (
        checked["schema"] != SCHEMA
        or checked["status"] != STATUS
        or checked["phase"] != PHASE
    ):
        raise ValueError("Step12b Phase2 IAM schema/status changed")
    window = checked.get("authorization_window")
    if not isinstance(window, Mapping):
        raise ValueError("Step12b Phase2 IAM window is missing")
    expected = _build_plan(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        issued_at_unix_seconds=window.get("issued_at_unix_seconds"),
        expires_at_unix_seconds=window.get("expires_at_unix_seconds"),
    )
    if dict(value) != expected:
        raise ValueError("Step12b Phase2 IAM plan changed")
    return expected


def build_step12b_phase2_iam_readback_receipt(
    plan: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    observed_at_unix_seconds: int,
    custom_role_readbacks: Sequence[Mapping[str, Any]],
    binding_readbacks: Sequence[Mapping[str, Any]],
    unexpected_targeted_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact GET readbacks for frozen roles and all Phase-2 bindings.

    Inputs are normalized observations collected by a separate GET-only
    adapter.  The adapter must enumerate all target policies and report any
    additional binding involving the controller or worker principal through
    ``unexpected_targeted_bindings``.  Any such row fails closed.
    """

    checked_plan = validate_step12b_phase2_iam_plan(
        plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    observed = _strict_int(
        observed_at_unix_seconds,
        "Phase2 IAM readback time",
        minimum=1,
    )
    window = checked_plan["authorization_window"]
    if (
        observed < window["issued_at_unix_seconds"]
        or observed >= window["expires_at_unix_seconds"]
    ):
        raise ValueError("Phase2 IAM readback is outside authorization window")
    if (
        isinstance(unexpected_targeted_bindings, (str, bytes))
        or not isinstance(unexpected_targeted_bindings, Sequence)
        or list(unexpected_targeted_bindings) != []
    ):
        raise ValueError("unexpected targeted Phase2 IAM binding observed")

    expected_roles = {
        row["name"]: row
        for row in checked_plan["custom_role_readback_contract"][
            "requirements"
        ]
    }
    if (
        isinstance(custom_role_readbacks, (str, bytes))
        or not isinstance(custom_role_readbacks, Sequence)
        or len(custom_role_readbacks) != len(expected_roles)
    ):
        raise ValueError("Phase2 custom-role readback count changed")
    normalized_roles = []
    seen_role_names: set[str] = set()
    role_fields = {
        "name",
        "stage",
        "included_permissions",
        "deleted",
        "get_status",
        "readback_complete",
    }
    for raw in custom_role_readbacks:
        row = copy.deepcopy(dict(raw))
        _exact(row, role_fields, "Phase2 custom-role readback")
        name = row["name"]
        expected = expected_roles.get(name)
        if (
            not isinstance(name, str)
            or expected is None
            or name in seen_role_names
            or row["stage"] != expected["stage"]
            or row["included_permissions"]
            != expected["included_permissions"]
            or row["deleted"] is not False
            or row["get_status"] != 200
            or row["readback_complete"] is not True
        ):
            raise ValueError("Phase2 custom-role readback changed")
        seen_role_names.add(name)
        normalized_roles.append(row)
    normalized_roles.sort(key=lambda row: row["name"])
    if seen_role_names != set(expected_roles):
        raise ValueError("Phase2 custom-role readback is incomplete")

    expected_bindings = {
        row["binding_sha256"]: row
        for row in [
            *checked_plan["phase2_bindings"]["controller"],
            *checked_plan["phase2_bindings"]["worker"],
        ]
    }
    if (
        isinstance(binding_readbacks, (str, bytes))
        or not isinstance(binding_readbacks, Sequence)
        or len(binding_readbacks) != len(expected_bindings)
    ):
        raise ValueError("Phase2 binding readback count changed")
    normalized_bindings = []
    seen_binding_shas: set[str] = set()
    binding_fields = {
        "purpose",
        "target",
        "resource",
        "role",
        "member",
        "condition",
        "binding_sha256",
        "member_occurrences",
        "readback_complete",
    }
    for raw in binding_readbacks:
        row = copy.deepcopy(dict(raw))
        _exact(row, binding_fields, "Phase2 binding readback")
        digest = _sha(
            row["binding_sha256"], "Phase2 binding readback digest"
        )
        expected = expected_bindings.get(digest)
        if expected is None or digest in seen_binding_shas:
            raise ValueError("Phase2 binding readback identity changed")
        expected_projection = {
            key: expected[key]
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
        observed_projection = {
            key: row[key] for key in expected_projection
        }
        if (
            observed_projection != expected_projection
            or row["member_occurrences"] != 1
            or row["readback_complete"] is not True
        ):
            raise ValueError("Phase2 binding readback changed")
        seen_binding_shas.add(digest)
        normalized_bindings.append(row)
    normalized_bindings.sort(key=lambda row: row["binding_sha256"])
    if seen_binding_shas != set(expected_bindings):
        raise ValueError("Phase2 binding readback is incomplete")

    receipt = {
        "schema": READBACK_RECEIPT_SCHEMA,
        "plan_sha256": checked_plan["plan_sha256"],
        "deployment_contract_sha256": checked_plan[
            "source_deployment"
        ]["deployment_contract_sha256"],
        "observed_at_unix_seconds": observed,
        "custom_role_readbacks": normalized_roles,
        "custom_role_readbacks_sha256": canonical_sha256(
            normalized_roles
        ),
        "binding_readbacks": normalized_bindings,
        "binding_readbacks_sha256": canonical_sha256(
            normalized_bindings
        ),
        "controller_binding_count": CONTROLLER_BINDING_COUNT,
        "worker_binding_count": WORKER_BINDING_COUNT,
        "unexpected_targeted_bindings": [],
        "all_existing_custom_roles_exact": True,
        "all_phase2_bindings_present_exactly_once": True,
        "get_only_observation": True,
        "cloud_mutation_performed_by_validator": False,
    }
    return {**receipt, "receipt_sha256": canonical_sha256(receipt)}


def validate_step12b_phase2_iam_readback_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, Any]:
    """Rebuild an exact Phase-2 IAM readback receipt."""

    checked = copy.deepcopy(dict(value))
    supplied_sha = _sha(
        checked.pop("receipt_sha256", None),
        "Phase2 IAM readback receipt",
    )
    if canonical_sha256(checked) != supplied_sha:
        raise ValueError("Phase2 IAM readback receipt digest changed")
    rebuilt = build_step12b_phase2_iam_readback_receipt(
        plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        observed_at_unix_seconds=checked.get(
            "observed_at_unix_seconds"
        ),
        custom_role_readbacks=checked.get("custom_role_readbacks", ()),
        binding_readbacks=checked.get("binding_readbacks", ()),
        unexpected_targeted_bindings=checked.get(
            "unexpected_targeted_bindings", ()
        ),
    )
    if dict(value) != rebuilt:
        raise ValueError("Phase2 IAM readback receipt changed")
    return rebuilt


__all__ = [
    "ATTEMPT_INDEX",
    "BUCKET",
    "CONTROLLER_BINDING_COUNT",
    "CUSTOM_ROLE_IDS",
    "CUSTOM_ROLE_PERMISSIONS",
    "MACHINE_TYPE",
    "MAX_ATTEMPTS",
    "MAX_AUTHORIZATION_WINDOW_SECONDS",
    "OLD_DIRECT_NAMESPACE",
    "PHASE",
    "PROJECT",
    "READBACK_RECEIPT_SCHEMA",
    "REQUIRED_WORKER_OAUTH_SCOPE",
    "SCHEMA",
    "SERVICE_USAGE_ROLE",
    "STATUS",
    "TOKEN_CREATOR_ROLE",
    "VM_COUNT",
    "WORKER_ACT_AS_ROLE",
    "WORKER_BINDING_COUNT",
    "WORKER_PRINCIPAL",
    "WORKER_SERVICE_ACCOUNT",
    "ZONE",
    "build_step12b_phase2_iam_plan",
    "build_step12b_phase2_iam_readback_receipt",
    "canonical_sha256",
    "validate_step12b_phase2_iam_plan",
    "validate_step12b_phase2_iam_readback_receipt",
]
