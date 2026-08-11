"""Offline Step11 IAM, bucket-security, and fresh-capacity gate.

This module deliberately has no Google Cloud client, HTTP transport, subprocess
call, credential lookup, or mutation surface.  It converts an already-pinned
10c.2 stage contract into a least-privilege IAM/network/capacity plan and
validates separately collected, GET-only evidence against that plan.

Passing this gate means only that the pre-launch security and quota evidence is
fresh and matches the pinned contract.  It never authorizes an instance insert,
changes an IAM policy, uploads an object, changes an AI profile, or claims that
Spot stock exists.  A separate signed launch authorization is still required.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_iam_capacity_gate_plan_v1"
)
OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_iam_capacity_observation_v1"
)
RESULT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_iam_capacity_gate_result_v1"
)
SOURCE_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_gce_transport_10c2_v1"
)
DIRECT_STAGE_IDENTITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_direct_stage_identity_v1"
)

PROJECT = "ofc-solver-485418"
PROJECT_NUMBER = "783381566570"
BUCKET = "pokerhu-ofc-solver-485418-training"
MACHINE_TYPE = "c4-standard-16"
VCPU_PER_VM = 16
MAX_CONCURRENT_VMS = 1
MAX_ATTEMPTS = 2
STEP11_AUTHORIZED_ATTEMPTS = 1
MAX_RUN_DURATION_SECONDS = 4_200
ZONE = "asia-northeast1-b"
REGION = "asia-northeast1"
DEFAULT_NETWORK_RESOURCE = f"projects/{PROJECT}/global/networks/default"
DEFAULT_SUBNETWORK_RESOURCE = (
    f"projects/{PROJECT}/regions/{REGION}/subnetworks/default"
)
DEFAULT_NAT_NAME = "ofc-t3-nat-asia-northeast1"
DEFAULT_CONTROLLER_PRINCIPAL = (
    f"serviceAccount:ofc-m31-t3-controller@{PROJECT}.iam.gserviceaccount.com"
)
WORKER_SERVICE_ACCOUNT = (
    f"ofc-m31-t3-diagnostic@{PROJECT}.iam.gserviceaccount.com"
)
WORKER_PRINCIPAL = f"serviceAccount:{WORKER_SERVICE_ACCOUNT}"
DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL = (
    f"serviceAccount:{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
)
CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL = (
    f"serviceAccount:{PROJECT_NUMBER}@cloudservices.gserviceaccount.com"
)
COMPUTE_SERVICE_AGENT_PRINCIPAL = (
    f"serviceAccount:service-{PROJECT_NUMBER}"
    "@compute-system.iam.gserviceaccount.com"
)
REQUIRED_WORKER_OAUTH_SCOPE = (
    "https://www.googleapis.com/auth/cloud-platform"
)
OBSERVATION_MAX_AGE_SECONDS = 300
MAX_AUTHORIZATION_WINDOW_SECONDS = 7_200
MIN_AUTHORIZATION_REMAINING_SECONDS = 120
MAX_CLOCK_SKEW_SECONDS = 5

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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SERVICE_ACCOUNT_PRINCIPAL = re.compile(
    r"^serviceAccount:"
    r"(?P<name>[a-z][a-z0-9-]{4,28}[a-z0-9])@"
    r"(?P<project>[a-z][a-z0-9-]{4,28}[a-z0-9])"
    r"\.iam\.gserviceaccount\.com$"
)

_PLAN_FIELDS = {
    "schema",
    "source_contract",
    "authorization_window",
    "principals",
    "iam_contract",
    "bucket_security_contract",
    "instance_contract",
    "network_contract",
    "capacity_contract",
    "gate_semantics",
    "plan_sha256",
}
_SOURCE_FIELDS = {
    "source_schema",
    "direct_stage_identity_sha256",
    "outer_package_identity_sha256",
    "metadata_binding_sha256",
    "project",
    "bucket",
    "zone",
    "region",
    "run_name",
    "job_id",
    "machine_type",
    "worker_service_account",
    "package_prefix",
    "package_object_prefix",
    "stage_prefix",
    "stage_object_prefix",
    "result_prefix",
    "result_object_prefix",
    "instance_names",
    "image_self_link",
}
_AUTHORIZATION_FIELDS = {
    "issued_at_unix_seconds",
    "expires_at_unix_seconds",
    "expires_at_rfc3339",
    "maximum_window_seconds",
    "minimum_remaining_at_gate_seconds",
}
_PRINCIPAL_FIELDS = {
    "controller_principal",
    "worker_principal",
    "worker_service_account",
    "compute_service_agent_principal",
    "forbidden_execution_principals",
    "owner_or_editor_execution_forbidden",
    "collector_may_be_separate_read_only_principal",
}
_IAM_CONTRACT_FIELDS = {
    "custom_roles",
    "worker_bindings",
    "controller_bindings",
    "worker_effective_permissions",
    "controller_effective_permissions",
    "worker_object_read_prefixes",
    "worker_object_create_prefixes",
    "worker_object_list_forbidden",
    "worker_object_delete_forbidden",
    "worker_object_overwrite_forbidden",
    "worker_self_delete_condition",
    "project_folder_org_effective_analysis_required",
    "basic_role_service_account_act_as_forbidden",
    "public_principals_forbidden",
}
_BUCKET_CONTRACT_FIELDS = {
    "bucket",
    "uniform_bucket_level_access_required",
    "public_access_prevention_effective_required",
    "legacy_object_acl_effective_required",
    "ancestor_public_access_prevention_resolution_required",
}
_INSTANCE_CONTRACT_FIELDS = {
    "allowed_launch_requests",
    "required_oauth_scopes",
    "external_access_configs_forbidden",
    "spot_required",
    "reservation_consumption_forbidden",
    "max_concurrent_vms",
    "max_attempts",
}
_NETWORK_CONTRACT_FIELDS = {
    "network_resource",
    "subnetwork_resource",
    "external_ip_forbidden",
    "nat_name",
    "nat_router_resource",
    "nat_source_subnetwork_ip_ranges",
    "subnetwork_nat_coverage_required",
    "compute_subnetworks_use_external_ip_permission_forbidden",
}
_CAPACITY_CONTRACT_FIELDS = {
    "zone",
    "region",
    "machine_type",
    "vcpu_per_vm",
    "requested_concurrent_vms",
    "requested_vcpu",
    "quota_metrics",
    "observation_max_age_seconds",
    "all_instance_inventory_statuses_required",
    "target_instance_names",
    "target_name_collision_forbidden",
    "nonterminated_c4_interference_forbidden",
    "spot_stock_only_provable_by_insert",
}
_OBSERVATION_FIELDS = {
    "schema",
    "plan_sha256",
    "collected_at_unix_seconds",
    "collected_via_get_only",
    "collector_cloud_mutation_performed",
    "execution",
    "resource_hierarchy",
    "effective_iam",
    "bucket_security",
    "instance_request",
    "network",
    "capacity",
    "observation_sha256",
}
_EXECUTION_FIELDS = {
    "initiating_principal",
    "api_bearer_principal",
    "credential_subject_principal",
    "credential_type",
    "controller_basic_roles",
    "controller_is_project_owner",
    "controller_is_project_editor",
    "owner_token_used_as_api_bearer",
}
_HIERARCHY_FIELDS = {
    "project_resource",
    "project_parent",
    "ancestor_resources",
    "resource_hierarchy_complete",
    "allow_policies_fully_explored",
    "deny_policies_fully_explored",
    "unresolved_resources",
}
_EFFECTIVE_IAM_FIELDS = {
    "custom_roles",
    "worker_bindings",
    "controller_bindings",
    "worker_effective_permissions",
    "controller_effective_permissions",
    "unexpected_worker_bindings",
    "unexpected_controller_bindings",
    "conditional_binding_evaluation_errors",
    "denied_required_permissions",
    "public_principals",
    "default_compute_sa_can_act_as_worker",
    "cloud_services_sa_can_act_as_worker",
    "unexpected_service_account_act_as_principals",
    "compute_service_agent_act_as_worker",
}
_BUCKET_FIELDS = {
    "bucket",
    "uniform_bucket_level_access_enabled",
    "public_access_prevention_mode",
    "public_access_prevention_effective",
    "ancestor_public_access_prevention_fully_resolved",
    "legacy_object_acl_effective",
    "public_principals",
}
_INSTANCE_FIELDS = {
    "attempt_index",
    "instance_name",
    "project",
    "zone",
    "machine_type",
    "vcpu",
    "worker_service_account",
    "oauth_scopes",
    "provisioning_model",
    "reservation_affinity",
    "max_run_duration_seconds",
    "automatic_restart",
    "on_host_maintenance",
    "boot_disk_auto_delete",
    "image_self_link",
    "metadata_binding_sha256",
    "access_configs",
}
_NETWORK_FIELDS = {
    "network_resource",
    "subnetwork_resource",
    "instance_has_external_ip",
    "external_access_config_count",
    "nat_name",
    "nat_router_resource",
    "nat_source_subnetwork_ip_ranges",
    "subnetwork_covered_by_nat",
    "nat_fully_configured",
}
_CAPACITY_FIELDS = {
    "zone_status",
    "machine_type",
    "machine_vcpu",
    "requested_concurrent_vms",
    "requested_vcpu",
    "global_cpu_limit",
    "global_cpu_usage",
    "regional_c4_cpu_limit",
    "regional_c4_cpu_usage",
    "regional_spot_cpu_limit",
    "regional_spot_cpu_usage",
    "available_vcpu",
    "quota_capacity_vms",
    "inventory_fully_enumerated",
    "target_name_collisions",
    "nonterminated_c4_instances",
    "unknown_machine_type_instances",
    "spot_stock_proven",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
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
        raise ValueError(f"{label} must be an exact bounded integer")
    return value


def _strict_bool(value: Any, expected: bool | None, label: str) -> bool:
    if type(value) is not bool or (expected is not None and value is not expected):
        raise ValueError(f"{label} must be a strict boolean")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _safe_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe resource name")
    return value


def _rfc3339(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parse_gs_prefix(value: Any, *, bucket: str, label: str) -> str:
    marker = f"gs://{bucket}/"
    if (
        not isinstance(value, str)
        or not value.startswith(marker)
        or value.endswith("/")
        or "//" in value[len(marker) :]
    ):
        raise ValueError(f"{label} is not a canonical gs:// prefix")
    object_prefix = value[len(marker) :]
    if not object_prefix or object_prefix.startswith("/"):
        raise ValueError(f"{label} object prefix is empty")
    return object_prefix


def _storage_resource(bucket: str, object_prefix: str) -> str:
    return f"projects/_/buckets/{bucket}/objects/{object_prefix}/"


def _prefix_condition(*resources: str) -> str:
    terms = [f'resource.name.startsWith("{resource}")' for resource in resources]
    if len(terms) == 1:
        return terms[0]
    return "(" + " || ".join(terms) + ")"


def _instance_resource(project: str, zone: str, name: str) -> str:
    return f"projects/{project}/zones/{zone}/instances/{name}"


def _instance_and_expiry_condition(
    *, project: str, zone: str, instance_names: Sequence[str], expires_at: str
) -> str:
    terms = [
        f'resource.name == "{_instance_resource(project, zone, name)}"'
        for name in instance_names
    ]
    names = terms[0] if len(terms) == 1 else "(" + " || ".join(terms) + ")"
    return (
        'resource.type == "compute.googleapis.com/Instance" && '
        f"{names} && "
        f'request.time < timestamp("{expires_at}")'
    )


def _custom_role(
    project: str, role_id: str, permissions: Sequence[str]
) -> dict[str, Any]:
    return {
        "name": f"projects/{project}/roles/{role_id}",
        "stage": "GA",
        "permissions": sorted(permissions),
    }


def _binding(
    *,
    resource: str,
    role: str,
    member: str,
    condition_title: str | None,
    condition_expression: str | None,
) -> dict[str, Any]:
    if (condition_title is None) != (condition_expression is None):
        raise ValueError("IAM condition title and expression must be paired")
    return {
        "resource": resource,
        "role": role,
        "member": member,
        "condition": (
            None
            if condition_title is None
            else {
                "title": condition_title,
                "expression": condition_expression,
            }
        ),
    }


def _normalize_controller_principal(value: Any, *, project: str) -> str:
    if not isinstance(value, str):
        raise ValueError("controller principal must be a service account")
    forbidden = {
        WORKER_PRINCIPAL,
        DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL,
        CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL,
        COMPUTE_SERVICE_AGENT_PRINCIPAL,
    }
    if value in forbidden:
        raise ValueError("controller principal is not dedicated")
    match = _SERVICE_ACCOUNT_PRINCIPAL.fullmatch(value)
    if match is None or match.group("project") != project:
        raise ValueError("controller principal must be a dedicated project service account")
    if match.group("name").endswith("-compute"):
        raise ValueError("controller principal is not dedicated")
    return value


def _normalize_source_contract(
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    source = deepcopy(dict(source_contract))
    if source.get("schema") != SOURCE_CONTRACT_SCHEMA:
        raise ValueError("source contract schema changed")
    direct = source.get("direct_stage_identity")
    metadata = source.get("metadata_binding")
    if not isinstance(direct, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("source contract lost direct identity or metadata binding")
    if direct.get("schema") != DIRECT_STAGE_IDENTITY_SCHEMA:
        raise ValueError("direct stage identity schema changed")
    identity_body = dict(direct)
    direct_sha = _sha(
        identity_body.pop("direct_stage_identity_sha256", None),
        "direct stage identity sha256",
    )
    if canonical_sha256(identity_body) != direct_sha:
        raise ValueError("direct stage identity digest mismatch")
    if source.get("direct_stage_identity_sha256") != direct_sha:
        raise ValueError("top-level direct stage identity digest mismatch")
    inputs = direct.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("direct stage identity inputs missing")
    if canonical_sha256(inputs) != direct.get("inputs_sha256"):
        raise ValueError("direct stage identity inputs digest mismatch")
    metadata_sha = _sha(
        source.get("metadata_binding_sha256"), "metadata binding sha256"
    )
    if canonical_sha256(metadata) != metadata_sha:
        raise ValueError("metadata binding digest mismatch")

    project = metadata.get("project")
    zone = metadata.get("zone")
    worker = metadata.get("worker_service_account")
    if project != PROJECT or zone != ZONE or worker != WORKER_SERVICE_ACCOUNT:
        raise ValueError("source contract project, zone, or worker changed")
    if (
        inputs.get("machine_type") != MACHINE_TYPE
        or inputs.get("max_attempts") != MAX_ATTEMPTS
        or inputs.get("zone") != zone
        or inputs.get("worker_service_account") != worker
    ):
        raise ValueError("source contract compute shape changed")
    if metadata.get("direct_stage_identity_sha256") != direct_sha:
        raise ValueError("metadata lost direct stage binding")
    if (
        metadata.get("outer_package_identity_sha256")
        != inputs.get("outer_package_identity_sha256")
    ):
        raise ValueError("metadata lost outer package binding")

    package_prefix = metadata.get("package_prefix")
    stage_prefix = metadata.get("stage_prefix")
    result_prefix = metadata.get("result_prefix")
    package_object_prefix = _parse_gs_prefix(
        package_prefix, bucket=BUCKET, label="package prefix"
    )
    stage_object_prefix = _parse_gs_prefix(
        stage_prefix, bucket=BUCKET, label="stage prefix"
    )
    result_object_prefix = _parse_gs_prefix(
        result_prefix, bucket=BUCKET, label="result prefix"
    )
    outer_sha = _sha(
        inputs.get("outer_package_identity_sha256"),
        "outer package identity sha256",
    )
    run_name = _safe_name(inputs.get("run_name"), "run name")
    if package_object_prefix != f"hu-m31-r2diag-direct-v1/packages/{outer_sha}":
        raise ValueError("package prefix is not exact")
    if stage_object_prefix != (
        f"hu-m31-r2diag-direct-v1/stages/{run_name}/{direct_sha}"
    ):
        raise ValueError("stage prefix is not exact")
    if result_object_prefix != f"{stage_object_prefix}/results":
        raise ValueError("result prefix is not exact")

    attempt_layout = inputs.get("attempt_layout")
    if not isinstance(attempt_layout, list) or len(attempt_layout) != MAX_ATTEMPTS:
        raise ValueError("source contract attempt layout changed")
    expected_attempts = list(range(MAX_ATTEMPTS))
    observed_attempts: list[int] = []
    instance_names: list[str] = []
    job_id: str | None = None
    for row in attempt_layout:
        if not isinstance(row, Mapping) or set(row) != {
            "attempt_index",
            "instance_name",
            "job_id",
        }:
            raise ValueError("attempt layout fields changed")
        observed_attempts.append(
            _strict_int(
                row["attempt_index"],
                "attempt index",
                minimum=0,
                maximum=MAX_ATTEMPTS - 1,
            )
        )
        instance_names.append(_safe_name(row["instance_name"], "instance name"))
        if job_id is None:
            job_id = _safe_name(row["job_id"], "job id")
        elif row["job_id"] != job_id:
            raise ValueError("attempt layout crosses jobs")
    if sorted(observed_attempts) != expected_attempts or len(set(instance_names)) != 2:
        raise ValueError("attempt layout is not exact and unique")
    if metadata.get("instance_name") != instance_names[0]:
        raise ValueError("metadata attempt-zero instance changed")

    image = metadata.get("image")
    if not isinstance(image, Mapping) or image.get("self_link") != inputs.get(
        "image_self_link"
    ):
        raise ValueError("pinned image binding changed")
    return {
        "source_schema": source["schema"],
        "direct_stage_identity_sha256": direct_sha,
        "outer_package_identity_sha256": outer_sha,
        "metadata_binding_sha256": metadata_sha,
        "project": project,
        "bucket": BUCKET,
        "zone": zone,
        "region": REGION,
        "run_name": run_name,
        "job_id": job_id,
        "machine_type": MACHINE_TYPE,
        "worker_service_account": worker,
        "package_prefix": package_prefix,
        "package_object_prefix": package_object_prefix,
        "stage_prefix": stage_prefix,
        "stage_object_prefix": stage_object_prefix,
        "result_prefix": result_prefix,
        "result_object_prefix": result_object_prefix,
        "instance_names": instance_names,
        "image_self_link": image["self_link"],
    }


def build_step11_gate_plan(
    source_contract: Mapping[str, Any],
    *,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
    nat_router_resource: str,
    controller_principal: str = DEFAULT_CONTROLLER_PRINCIPAL,
    nat_name: str = DEFAULT_NAT_NAME,
) -> dict[str, Any]:
    """Build an immutable, local-only Step11 least-privilege plan."""

    source = _normalize_source_contract(source_contract)
    issued = _strict_int(issued_at_unix_seconds, "issued time", minimum=1)
    expires = _strict_int(expires_at_unix_seconds, "expiry time", minimum=1)
    if expires <= issued or expires - issued > MAX_AUTHORIZATION_WINDOW_SECONDS:
        raise ValueError("authorization window is not positive and short-lived")
    controller = _normalize_controller_principal(
        controller_principal, project=source["project"]
    )
    nat = _safe_name(nat_name, "Cloud NAT name")
    expected_router_prefix = (
        f"projects/{source['project']}/regions/{source['region']}/routers/"
    )
    if (
        not isinstance(nat_router_resource, str)
        or not nat_router_resource.startswith(expected_router_prefix)
        or _SAFE_NAME.fullmatch(
            nat_router_resource[len(expected_router_prefix) :]
        )
        is None
    ):
        raise ValueError("Cloud NAT router resource is not exact for the region")

    expiry_rfc3339 = _rfc3339(expires)
    package_resource = _storage_resource(
        source["bucket"], source["package_object_prefix"]
    )
    result_resource = _storage_resource(
        source["bucket"], source["result_object_prefix"]
    )
    worker_read_condition = _prefix_condition(package_resource, result_resource)
    worker_create_condition = _prefix_condition(result_resource)
    instance_condition = _instance_and_expiry_condition(
        project=source["project"],
        zone=source["zone"],
        instance_names=source["instance_names"][:STEP11_AUTHORIZED_ATTEMPTS],
        expires_at=expiry_rfc3339,
    )
    time_condition = f'request.time < timestamp("{expiry_rfc3339}")'
    project_resource = f"projects/{source['project']}"
    bucket_resource = f"projects/_/buckets/{source['bucket']}"
    worker_sa_resource = (
        f"projects/{source['project']}/serviceAccounts/"
        f"{source['worker_service_account']}"
    )

    role_ids = {
        "worker_object_reader": "ofcM31T3ObjectReaderV1",
        "worker_result_creator": "ofcM31T3ResultCreatorV1",
        "worker_self_delete": "ofcM31T3SelfDeleteV1",
        "controller_vm_launch": "ofcM31T3VmLaunchV1",
        "controller_instance_lifecycle": "ofcM31T3VmLifecycleV1",
        "controller_zone_operation_reader": "ofcM31T3ZoneOperationReaderV1",
    }
    custom_roles = {
        key: _custom_role(
            source["project"], role_ids[key], CUSTOM_ROLE_PERMISSIONS[key]
        )
        for key in sorted(role_ids)
    }
    worker_bindings = [
        _binding(
            resource=bucket_resource,
            role=custom_roles["worker_object_reader"]["name"],
            member=WORKER_PRINCIPAL,
            condition_title="ofc-m31-t3-exact-package-and-results-read-v1",
            condition_expression=worker_read_condition,
        ),
        _binding(
            resource=bucket_resource,
            role=custom_roles["worker_result_creator"]["name"],
            member=WORKER_PRINCIPAL,
            condition_title="ofc-m31-t3-only-results-create-v1",
            condition_expression=worker_create_condition,
        ),
        _binding(
            resource=project_resource,
            role=custom_roles["worker_self_delete"]["name"],
            member=WORKER_PRINCIPAL,
            condition_title="ofc-m31-t3-exact-self-delete-v1",
            condition_expression=instance_condition,
        ),
    ]
    controller_bindings = [
        _binding(
            resource=project_resource,
            role=custom_roles["controller_vm_launch"]["name"],
            member=controller,
            condition_title="ofc-m31-t3-short-lived-launch-v1",
            condition_expression=time_condition,
        ),
        _binding(
            resource=project_resource,
            role=custom_roles["controller_instance_lifecycle"]["name"],
            member=controller,
            condition_title="ofc-m31-t3-exact-instance-lifecycle-v1",
            condition_expression=instance_condition,
        ),
        _binding(
            resource=project_resource,
            role=custom_roles["controller_zone_operation_reader"]["name"],
            member=controller,
            condition_title="ofc-m31-t3-short-lived-operation-read-v1",
            condition_expression=time_condition,
        ),
        _binding(
            resource=worker_sa_resource,
            role="roles/iam.serviceAccountUser",
            member=controller,
            condition_title="ofc-m31-t3-short-lived-exact-worker-actas-v1",
            condition_expression=time_condition,
        ),
    ]
    controller_permissions = sorted(
        {
            permission
            for key in (
                "controller_vm_launch",
                "controller_instance_lifecycle",
                "controller_zone_operation_reader",
            )
            for permission in CUSTOM_ROLE_PERMISSIONS[key]
        }
        | {"iam.serviceAccounts.actAs"}
    )
    worker_permissions = sorted(
        {
            permission
            for key in (
                "worker_object_reader",
                "worker_result_creator",
                "worker_self_delete",
            )
            for permission in CUSTOM_ROLE_PERMISSIONS[key]
        }
    )

    allowed_requests = []
    for attempt_index, instance_name in enumerate(
        source["instance_names"][:STEP11_AUTHORIZED_ATTEMPTS]
    ):
        request = {
            "attempt_index": attempt_index,
            "instance_name": instance_name,
            "project": source["project"],
            "zone": source["zone"],
            "machine_type": source["machine_type"],
            "vcpu": VCPU_PER_VM,
            "worker_service_account": source["worker_service_account"],
            "oauth_scopes": [REQUIRED_WORKER_OAUTH_SCOPE],
            "provisioning_model": "SPOT",
            "reservation_affinity": "NO_RESERVATION",
            "max_run_duration_seconds": MAX_RUN_DURATION_SECONDS,
            "automatic_restart": False,
            "on_host_maintenance": "TERMINATE",
            "boot_disk_auto_delete": True,
            "image_self_link": source["image_self_link"],
            "metadata_binding_sha256": source["metadata_binding_sha256"],
            "access_configs": [],
        }
        allowed_requests.append(
            {**request, "request_contract_sha256": canonical_sha256(request)}
        )

    plan_body = {
        "schema": PLAN_SCHEMA,
        "source_contract": source,
        "authorization_window": {
            "issued_at_unix_seconds": issued,
            "expires_at_unix_seconds": expires,
            "expires_at_rfc3339": expiry_rfc3339,
            "maximum_window_seconds": MAX_AUTHORIZATION_WINDOW_SECONDS,
            "minimum_remaining_at_gate_seconds": (
                MIN_AUTHORIZATION_REMAINING_SECONDS
            ),
        },
        "principals": {
            "controller_principal": controller,
            "worker_principal": WORKER_PRINCIPAL,
            "worker_service_account": source["worker_service_account"],
            "compute_service_agent_principal": COMPUTE_SERVICE_AGENT_PRINCIPAL,
            "forbidden_execution_principals": [
                DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL,
                CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL,
                WORKER_PRINCIPAL,
            ],
            "owner_or_editor_execution_forbidden": True,
            "collector_may_be_separate_read_only_principal": True,
        },
        "iam_contract": {
            "custom_roles": custom_roles,
            "worker_bindings": worker_bindings,
            "controller_bindings": controller_bindings,
            "worker_effective_permissions": worker_permissions,
            "controller_effective_permissions": controller_permissions,
            "worker_object_read_prefixes": [
                source["package_prefix"] + "/",
                source["result_prefix"] + "/",
            ],
            "worker_object_create_prefixes": [source["result_prefix"] + "/"],
            "worker_object_list_forbidden": True,
            "worker_object_delete_forbidden": True,
            "worker_object_overwrite_forbidden": True,
            "worker_self_delete_condition": instance_condition,
            "project_folder_org_effective_analysis_required": True,
            "basic_role_service_account_act_as_forbidden": True,
            "public_principals_forbidden": [
                "allUsers",
                "allAuthenticatedUsers",
            ],
        },
        "bucket_security_contract": {
            "bucket": source["bucket"],
            "uniform_bucket_level_access_required": True,
            "public_access_prevention_effective_required": True,
            "legacy_object_acl_effective_required": False,
            "ancestor_public_access_prevention_resolution_required": True,
        },
        "instance_contract": {
            "allowed_launch_requests": allowed_requests,
            "required_oauth_scopes": [REQUIRED_WORKER_OAUTH_SCOPE],
            "external_access_configs_forbidden": True,
            "spot_required": True,
            "reservation_consumption_forbidden": True,
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
            "max_attempts": STEP11_AUTHORIZED_ATTEMPTS,
        },
        "network_contract": {
            "network_resource": DEFAULT_NETWORK_RESOURCE,
            "subnetwork_resource": DEFAULT_SUBNETWORK_RESOURCE,
            "external_ip_forbidden": True,
            "nat_name": nat,
            "nat_router_resource": nat_router_resource,
            "nat_source_subnetwork_ip_ranges": (
                "ALL_SUBNETWORKS_ALL_IP_RANGES"
            ),
            "subnetwork_nat_coverage_required": True,
            "compute_subnetworks_use_external_ip_permission_forbidden": True,
        },
        "capacity_contract": {
            "zone": source["zone"],
            "region": source["region"],
            "machine_type": source["machine_type"],
            "vcpu_per_vm": VCPU_PER_VM,
            "requested_concurrent_vms": MAX_CONCURRENT_VMS,
            "requested_vcpu": VCPU_PER_VM * MAX_CONCURRENT_VMS,
            "quota_metrics": [
                "CPUS_ALL_REGIONS",
                "C4_CPUS",
                "PREEMPTIBLE_CPUS",
            ],
            "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
            "all_instance_inventory_statuses_required": True,
            "target_instance_names": source["instance_names"][
                :STEP11_AUTHORIZED_ATTEMPTS
            ],
            "target_name_collision_forbidden": True,
            "nonterminated_c4_interference_forbidden": True,
            "spot_stock_only_provable_by_insert": True,
        },
        "gate_semantics": {
            "local_validation_only": True,
            "cloud_api_calls_performed": False,
            "cloud_mutation_performed": False,
            "profile_or_current_changed": False,
            "existing_artifact_changed": False,
            "passing_gate_authorizes_launch": False,
            "passing_gate_proves_spot_stock": False,
            "separate_signed_launch_authorization_required": True,
        },
    }
    return {**plan_body, "plan_sha256": canonical_sha256(plan_body)}


def validate_step11_gate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate plan integrity and immutable safety semantics."""

    candidate = deepcopy(dict(plan))
    _exact(candidate, _PLAN_FIELDS, "Step11 plan")
    digest = _sha(candidate.pop("plan_sha256"), "Step11 plan sha256")
    if canonical_sha256(candidate) != digest:
        raise ValueError("Step11 plan digest mismatch")
    if candidate["schema"] != PLAN_SCHEMA:
        raise ValueError("Step11 plan schema changed")

    source = candidate.get("source_contract")
    if not isinstance(source, Mapping):
        raise ValueError("Step11 source contract missing")
    _exact(source, _SOURCE_FIELDS, "Step11 normalized source")
    if (
        source["source_schema"] != SOURCE_CONTRACT_SCHEMA
        or source["project"] != PROJECT
        or source["bucket"] != BUCKET
        or source["zone"] != ZONE
        or source["region"] != REGION
        or source["machine_type"] != MACHINE_TYPE
        or source["worker_service_account"] != WORKER_SERVICE_ACCOUNT
    ):
        raise ValueError("Step11 normalized source target changed")
    direct_sha = _sha(
        source["direct_stage_identity_sha256"],
        "normalized direct stage identity sha256",
    )
    outer_sha = _sha(
        source["outer_package_identity_sha256"],
        "normalized outer package identity sha256",
    )
    _sha(
        source["metadata_binding_sha256"],
        "normalized metadata binding sha256",
    )
    _safe_name(source["run_name"], "normalized run name")
    _safe_name(source["job_id"], "normalized job id")
    if source["package_object_prefix"] != (
        f"hu-m31-r2diag-direct-v1/packages/{outer_sha}"
    ):
        raise ValueError("normalized package prefix changed")
    if source["stage_object_prefix"] != (
        f"hu-m31-r2diag-direct-v1/stages/{source['run_name']}/{direct_sha}"
    ):
        raise ValueError("normalized stage prefix changed")
    if source["result_object_prefix"] != (
        f"{source['stage_object_prefix']}/results"
    ):
        raise ValueError("normalized result prefix changed")
    if source["package_prefix"] != (
        f"gs://{BUCKET}/{source['package_object_prefix']}"
    ):
        raise ValueError("normalized package gs prefix changed")
    if source["stage_prefix"] != (
        f"gs://{BUCKET}/{source['stage_object_prefix']}"
    ):
        raise ValueError("normalized stage gs prefix changed")
    if source["result_prefix"] != (
        f"gs://{BUCKET}/{source['result_object_prefix']}"
    ):
        raise ValueError("normalized result gs prefix changed")
    if (
        not isinstance(source["instance_names"], list)
        or len(source["instance_names"]) != MAX_ATTEMPTS
        or len(set(source["instance_names"])) != MAX_ATTEMPTS
    ):
        raise ValueError("normalized instance names changed")
    for name in source["instance_names"]:
        _safe_name(name, "normalized instance name")

    authorization = candidate.get("authorization_window")
    if not isinstance(authorization, Mapping):
        raise ValueError("Step11 authorization window missing")
    _exact(
        authorization,
        _AUTHORIZATION_FIELDS,
        "Step11 authorization window",
    )
    issued = _strict_int(
        authorization["issued_at_unix_seconds"],
        "plan issued time",
        minimum=1,
    )
    expires = _strict_int(
        authorization["expires_at_unix_seconds"],
        "plan expiry time",
        minimum=1,
    )
    if (
        expires <= issued
        or expires - issued > MAX_AUTHORIZATION_WINDOW_SECONDS
        or authorization["expires_at_rfc3339"] != _rfc3339(expires)
        or authorization["maximum_window_seconds"]
        != MAX_AUTHORIZATION_WINDOW_SECONDS
        or authorization["minimum_remaining_at_gate_seconds"]
        != MIN_AUTHORIZATION_REMAINING_SECONDS
    ):
        raise ValueError("Step11 authorization window changed")

    semantics = candidate.get("gate_semantics")
    if not isinstance(semantics, Mapping) or semantics != {
        "local_validation_only": True,
        "cloud_api_calls_performed": False,
        "cloud_mutation_performed": False,
        "profile_or_current_changed": False,
        "existing_artifact_changed": False,
        "passing_gate_authorizes_launch": False,
        "passing_gate_proves_spot_stock": False,
        "separate_signed_launch_authorization_required": True,
    }:
        raise ValueError("Step11 gate semantics changed")
    principals = candidate.get("principals")
    if not isinstance(principals, Mapping):
        raise ValueError("Step11 principals missing")
    _exact(principals, _PRINCIPAL_FIELDS, "Step11 principals")
    controller = _normalize_controller_principal(
        principals["controller_principal"],
        project=source["project"],
    )
    if principals != {
        "controller_principal": controller,
        "worker_principal": WORKER_PRINCIPAL,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "compute_service_agent_principal": COMPUTE_SERVICE_AGENT_PRINCIPAL,
        "forbidden_execution_principals": [
            DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL,
            CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL,
            WORKER_PRINCIPAL,
        ],
        "owner_or_editor_execution_forbidden": True,
        "collector_may_be_separate_read_only_principal": True,
    }:
        raise ValueError("Step11 principal restrictions changed")

    iam = candidate.get("iam_contract")
    if not isinstance(iam, Mapping):
        raise ValueError("Step11 IAM contract missing")
    _exact(iam, _IAM_CONTRACT_FIELDS, "Step11 IAM contract")
    role_ids = {
        "worker_object_reader": "ofcM31T3ObjectReaderV1",
        "worker_result_creator": "ofcM31T3ResultCreatorV1",
        "worker_self_delete": "ofcM31T3SelfDeleteV1",
        "controller_vm_launch": "ofcM31T3VmLaunchV1",
        "controller_instance_lifecycle": "ofcM31T3VmLifecycleV1",
        "controller_zone_operation_reader": "ofcM31T3ZoneOperationReaderV1",
    }
    expected_roles = {
        key: _custom_role(
            PROJECT, role_ids[key], CUSTOM_ROLE_PERMISSIONS[key]
        )
        for key in sorted(role_ids)
    }
    if iam["custom_roles"] != expected_roles:
        raise ValueError("Step11 custom roles are not exact")
    expiry_rfc3339 = authorization["expires_at_rfc3339"]
    package_resource = _storage_resource(
        BUCKET, source["package_object_prefix"]
    )
    result_resource = _storage_resource(
        BUCKET, source["result_object_prefix"]
    )
    instance_condition = _instance_and_expiry_condition(
        project=PROJECT,
        zone=ZONE,
        instance_names=source["instance_names"][:STEP11_AUTHORIZED_ATTEMPTS],
        expires_at=expiry_rfc3339,
    )
    time_condition = f'request.time < timestamp("{expiry_rfc3339}")'
    expected_worker_bindings = [
        _binding(
            resource=f"projects/_/buckets/{BUCKET}",
            role=expected_roles["worker_object_reader"]["name"],
            member=WORKER_PRINCIPAL,
            condition_title="ofc-m31-t3-exact-package-and-results-read-v1",
            condition_expression=_prefix_condition(
                package_resource, result_resource
            ),
        ),
        _binding(
            resource=f"projects/_/buckets/{BUCKET}",
            role=expected_roles["worker_result_creator"]["name"],
            member=WORKER_PRINCIPAL,
            condition_title="ofc-m31-t3-only-results-create-v1",
            condition_expression=_prefix_condition(result_resource),
        ),
        _binding(
            resource=f"projects/{PROJECT}",
            role=expected_roles["worker_self_delete"]["name"],
            member=WORKER_PRINCIPAL,
            condition_title="ofc-m31-t3-exact-self-delete-v1",
            condition_expression=instance_condition,
        ),
    ]
    expected_controller_bindings = [
        _binding(
            resource=f"projects/{PROJECT}",
            role=expected_roles["controller_vm_launch"]["name"],
            member=controller,
            condition_title="ofc-m31-t3-short-lived-launch-v1",
            condition_expression=time_condition,
        ),
        _binding(
            resource=f"projects/{PROJECT}",
            role=expected_roles["controller_instance_lifecycle"]["name"],
            member=controller,
            condition_title="ofc-m31-t3-exact-instance-lifecycle-v1",
            condition_expression=instance_condition,
        ),
        _binding(
            resource=f"projects/{PROJECT}",
            role=expected_roles["controller_zone_operation_reader"]["name"],
            member=controller,
            condition_title="ofc-m31-t3-short-lived-operation-read-v1",
            condition_expression=time_condition,
        ),
        _binding(
            resource=(
                f"projects/{PROJECT}/serviceAccounts/"
                f"{WORKER_SERVICE_ACCOUNT}"
            ),
            role="roles/iam.serviceAccountUser",
            member=controller,
            condition_title="ofc-m31-t3-short-lived-exact-worker-actas-v1",
            condition_expression=time_condition,
        ),
    ]
    expected_worker_permissions = [
        "compute.instances.delete",
        "storage.objects.create",
        "storage.objects.get",
    ]
    expected_controller_permissions = sorted(
        {
            permission
            for key in (
                "controller_vm_launch",
                "controller_instance_lifecycle",
                "controller_zone_operation_reader",
            )
            for permission in CUSTOM_ROLE_PERMISSIONS[key]
        }
        | {"iam.serviceAccounts.actAs"}
    )
    expected_iam_scalars = {
        "worker_bindings": expected_worker_bindings,
        "controller_bindings": expected_controller_bindings,
        "worker_effective_permissions": expected_worker_permissions,
        "controller_effective_permissions": expected_controller_permissions,
        "worker_object_read_prefixes": [
            source["package_prefix"] + "/",
            source["result_prefix"] + "/",
        ],
        "worker_object_create_prefixes": [source["result_prefix"] + "/"],
        "worker_object_list_forbidden": True,
        "worker_object_delete_forbidden": True,
        "worker_object_overwrite_forbidden": True,
        "worker_self_delete_condition": instance_condition,
        "project_folder_org_effective_analysis_required": True,
        "basic_role_service_account_act_as_forbidden": True,
        "public_principals_forbidden": [
            "allUsers",
            "allAuthenticatedUsers",
        ],
    }
    for key, expected in expected_iam_scalars.items():
        if iam[key] != expected:
            raise ValueError(f"Step11 IAM {key} changed")

    bucket_contract = candidate.get("bucket_security_contract")
    if not isinstance(bucket_contract, Mapping):
        raise ValueError("Step11 bucket security contract missing")
    _exact(
        bucket_contract,
        _BUCKET_CONTRACT_FIELDS,
        "Step11 bucket security contract",
    )
    if bucket_contract != {
        "bucket": BUCKET,
        "uniform_bucket_level_access_required": True,
        "public_access_prevention_effective_required": True,
        "legacy_object_acl_effective_required": False,
        "ancestor_public_access_prevention_resolution_required": True,
    }:
        raise ValueError("Step11 bucket security requirements changed")

    instance = candidate.get("instance_contract")
    if not isinstance(instance, Mapping):
        raise ValueError("Step11 instance contract missing")
    _exact(instance, _INSTANCE_CONTRACT_FIELDS, "Step11 instance contract")
    expected_requests = []
    for attempt_index, instance_name in enumerate(
        source["instance_names"][:STEP11_AUTHORIZED_ATTEMPTS]
    ):
        request = {
            "attempt_index": attempt_index,
            "instance_name": instance_name,
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
            "image_self_link": source["image_self_link"],
            "metadata_binding_sha256": source["metadata_binding_sha256"],
            "access_configs": [],
        }
        expected_requests.append(
            {**request, "request_contract_sha256": canonical_sha256(request)}
        )
    if instance != {
        "allowed_launch_requests": expected_requests,
        "required_oauth_scopes": [REQUIRED_WORKER_OAUTH_SCOPE],
        "external_access_configs_forbidden": True,
        "spot_required": True,
        "reservation_consumption_forbidden": True,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "max_attempts": STEP11_AUTHORIZED_ATTEMPTS,
    }:
        raise ValueError("Step11 instance restrictions changed")

    network = candidate.get("network_contract")
    if not isinstance(network, Mapping):
        raise ValueError("Step11 network contract missing")
    _exact(network, _NETWORK_CONTRACT_FIELDS, "Step11 network contract")
    expected_router_prefix = f"projects/{PROJECT}/regions/{REGION}/routers/"
    router = network["nat_router_resource"]
    if (
        not isinstance(router, str)
        or not router.startswith(expected_router_prefix)
        or _SAFE_NAME.fullmatch(router[len(expected_router_prefix) :]) is None
    ):
        raise ValueError("Step11 NAT router changed")
    _safe_name(network["nat_name"], "Step11 NAT name")
    if network != {
        "network_resource": DEFAULT_NETWORK_RESOURCE,
        "subnetwork_resource": DEFAULT_SUBNETWORK_RESOURCE,
        "external_ip_forbidden": True,
        "nat_name": network["nat_name"],
        "nat_router_resource": router,
        "nat_source_subnetwork_ip_ranges": (
            "ALL_SUBNETWORKS_ALL_IP_RANGES"
        ),
        "subnetwork_nat_coverage_required": True,
        "compute_subnetworks_use_external_ip_permission_forbidden": True,
    }:
        raise ValueError("Step11 network restrictions changed")

    capacity = candidate.get("capacity_contract")
    if not isinstance(capacity, Mapping):
        raise ValueError("Step11 capacity contract missing")
    _exact(
        capacity,
        _CAPACITY_CONTRACT_FIELDS,
        "Step11 capacity contract",
    )
    if capacity != {
        "zone": ZONE,
        "region": REGION,
        "machine_type": MACHINE_TYPE,
        "vcpu_per_vm": VCPU_PER_VM,
        "requested_concurrent_vms": MAX_CONCURRENT_VMS,
        "requested_vcpu": VCPU_PER_VM * MAX_CONCURRENT_VMS,
        "quota_metrics": [
            "CPUS_ALL_REGIONS",
            "C4_CPUS",
            "PREEMPTIBLE_CPUS",
        ],
        "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
        "all_instance_inventory_statuses_required": True,
        "target_instance_names": source["instance_names"][
            :STEP11_AUTHORIZED_ATTEMPTS
        ],
        "target_name_collision_forbidden": True,
        "nonterminated_c4_interference_forbidden": True,
        "spot_stock_only_provable_by_insert": True,
    }:
        raise ValueError("Step11 capacity restrictions changed")
    return {**candidate, "plan_sha256": digest}


def seal_step11_gate_observation(
    observation_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach a canonical digest without accepting an existing digest."""

    body = deepcopy(dict(observation_body))
    if "observation_sha256" in body:
        raise ValueError("observation body is already sealed")
    return {**body, "observation_sha256": canonical_sha256(body)}


def _result(
    *,
    plan_sha256: str,
    observation_sha256: str | None,
    evaluated_at_unix_seconds: int,
    failures: Sequence[str],
) -> dict[str, Any]:
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    body = {
        "schema": RESULT_SCHEMA,
        "plan_sha256": plan_sha256,
        "observation_sha256": observation_sha256,
        "evaluated_at_unix_seconds": evaluated_at_unix_seconds,
        "failures": unique_failures,
        "iam_ubla_capacity_gate_passed": passed,
        "step11_prelaunch_ready": passed,
        "spot_stock_proven": False,
        "launch_authorized": False,
        "cloud_mutation_performed": False,
        "profile_or_current_changed": False,
        "status": (
            "prelaunch_evidence_passed_separate_authorization_required"
            if passed
            else "fail_closed"
        ),
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def validate_step11_gate(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    evaluated_at_unix_seconds: int,
) -> dict[str, Any]:
    """Validate GET-only evidence.  Malformed evidence always fails closed."""

    checked_plan = validate_step11_gate_plan(plan)
    now = _strict_int(
        evaluated_at_unix_seconds, "gate evaluation time", minimum=1
    )
    plan_sha = checked_plan["plan_sha256"]
    failures: list[str] = []
    observed = deepcopy(dict(observation))
    observed_sha: str | None = None

    if set(observed) != _OBSERVATION_FIELDS:
        failures.append("observation_fields_changed")
        return _result(
            plan_sha256=plan_sha,
            observation_sha256=None,
            evaluated_at_unix_seconds=now,
            failures=failures,
        )
    raw_observed_sha = observed.pop("observation_sha256")
    if isinstance(raw_observed_sha, str) and _SHA256.fullmatch(raw_observed_sha):
        observed_sha = raw_observed_sha
        if canonical_sha256(observed) != raw_observed_sha:
            failures.append("observation_sha256_mismatch")
    else:
        failures.append("observation_sha256_invalid")
    observed["observation_sha256"] = raw_observed_sha
    if observed.get("schema") != OBSERVATION_SCHEMA:
        failures.append("observation_schema_mismatch")
    if observed.get("plan_sha256") != plan_sha:
        failures.append("observation_plan_mismatch")

    collected = observed.get("collected_at_unix_seconds")
    if type(collected) is not int:
        failures.append("observation_time_invalid")
    else:
        age = now - collected
        if age < -MAX_CLOCK_SKEW_SECONDS:
            failures.append("observation_from_future")
        if age > OBSERVATION_MAX_AGE_SECONDS:
            failures.append("observation_stale")
    if observed.get("collected_via_get_only") is not True:
        failures.append("observation_not_get_only")
    if observed.get("collector_cloud_mutation_performed") is not False:
        failures.append("collector_cloud_mutation_detected")

    expires = checked_plan["authorization_window"][
        "expires_at_unix_seconds"
    ]
    if expires - now < MIN_AUTHORIZATION_REMAINING_SECONDS:
        failures.append("authorization_window_too_close_or_expired")

    execution = observed.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != _EXECUTION_FIELDS:
        failures.append("execution_evidence_fields_changed")
    else:
        controller = checked_plan["principals"]["controller_principal"]
        if execution["api_bearer_principal"] != controller:
            failures.append("api_bearer_is_not_dedicated_controller")
        if execution["credential_subject_principal"] != controller:
            failures.append("credential_subject_is_not_dedicated_controller")
        if execution["credential_type"] not in {
            "service_account_impersonation",
            "workload_identity",
        }:
            failures.append("controller_credential_type_not_allowed")
        if execution["controller_basic_roles"] != []:
            failures.append("controller_has_basic_role")
        if execution["controller_is_project_owner"] is not False:
            failures.append("controller_is_project_owner")
        if execution["controller_is_project_editor"] is not False:
            failures.append("controller_is_project_editor")
        if execution["owner_token_used_as_api_bearer"] is not False:
            failures.append("owner_token_used_as_api_bearer")

    hierarchy = observed.get("resource_hierarchy")
    if not isinstance(hierarchy, Mapping) or set(hierarchy) != _HIERARCHY_FIELDS:
        failures.append("resource_hierarchy_fields_changed")
    else:
        expected_project = f"projects/{checked_plan['source_contract']['project']}"
        if hierarchy["project_resource"] != expected_project:
            failures.append("resource_hierarchy_project_mismatch")
        if hierarchy["project_parent"] is not None:
            failures.append("unexpected_project_parent")
        if hierarchy["ancestor_resources"] != [expected_project]:
            failures.append("resource_hierarchy_not_exact")
        if hierarchy["resource_hierarchy_complete"] is not True:
            failures.append("resource_hierarchy_incomplete")
        if hierarchy["allow_policies_fully_explored"] is not True:
            failures.append("allow_policies_not_fully_explored")
        if hierarchy["deny_policies_fully_explored"] is not True:
            failures.append("deny_policies_not_fully_explored")
        if hierarchy["unresolved_resources"] != []:
            failures.append("resource_hierarchy_has_unresolved_resources")

    iam = observed.get("effective_iam")
    if not isinstance(iam, Mapping) or set(iam) != _EFFECTIVE_IAM_FIELDS:
        failures.append("effective_iam_fields_changed")
    else:
        expected_iam = checked_plan["iam_contract"]
        comparisons = {
            "custom_roles": "custom_roles_mismatch",
            "worker_bindings": "worker_bindings_mismatch",
            "controller_bindings": "controller_bindings_mismatch",
            "worker_effective_permissions": (
                "worker_effective_permissions_not_exact"
            ),
            "controller_effective_permissions": (
                "controller_effective_permissions_not_exact"
            ),
        }
        for key, code in comparisons.items():
            if iam[key] != expected_iam[key]:
                failures.append(code)
        for key, code in {
            "unexpected_worker_bindings": "unexpected_worker_bindings",
            "unexpected_controller_bindings": "unexpected_controller_bindings",
            "conditional_binding_evaluation_errors": (
                "conditional_binding_evaluation_errors"
            ),
            "denied_required_permissions": "required_permission_denied",
            "public_principals": "public_iam_principal_present",
            "unexpected_service_account_act_as_principals": (
                "unexpected_service_account_can_act_as_worker"
            ),
        }.items():
            if iam[key] != []:
                failures.append(code)
        if iam["default_compute_sa_can_act_as_worker"] is not False:
            failures.append("default_compute_sa_can_act_as_worker")
        if iam["cloud_services_sa_can_act_as_worker"] is not False:
            failures.append("cloud_services_sa_can_act_as_worker")
        if iam["compute_service_agent_act_as_worker"] is not True:
            failures.append("compute_service_agent_act_as_worker_unresolved")

    bucket = observed.get("bucket_security")
    if not isinstance(bucket, Mapping) or set(bucket) != _BUCKET_FIELDS:
        failures.append("bucket_security_fields_changed")
    else:
        if bucket["bucket"] != checked_plan["source_contract"]["bucket"]:
            failures.append("bucket_mismatch")
        if bucket["uniform_bucket_level_access_enabled"] is not True:
            failures.append("uniform_bucket_level_access_disabled")
        if bucket["public_access_prevention_mode"] not in {
            "enforced",
            "inherited",
        }:
            failures.append("public_access_prevention_mode_invalid")
        if bucket["public_access_prevention_effective"] is not True:
            failures.append("public_access_prevention_not_effective")
        if (
            bucket["ancestor_public_access_prevention_fully_resolved"]
            is not True
        ):
            failures.append("ancestor_public_access_prevention_unresolved")
        if bucket["legacy_object_acl_effective"] is not False:
            failures.append("legacy_object_acl_effective")
        if bucket["public_principals"] != []:
            failures.append("bucket_public_principal_present")

    instance = observed.get("instance_request")
    if not isinstance(instance, Mapping) or set(instance) != _INSTANCE_FIELDS:
        failures.append("instance_request_fields_changed")
    else:
        allowed = [
            {
                key: value
                for key, value in row.items()
                if key != "request_contract_sha256"
            }
            for row in checked_plan["instance_contract"][
                "allowed_launch_requests"
            ]
        ]
        if dict(instance) not in allowed:
            failures.append("instance_request_not_exact")
        if instance["oauth_scopes"] != [REQUIRED_WORKER_OAUTH_SCOPE]:
            failures.append("worker_oauth_scope_not_exact")
        if instance["access_configs"] != []:
            failures.append("instance_external_access_config_present")

    network = observed.get("network")
    if not isinstance(network, Mapping) or set(network) != _NETWORK_FIELDS:
        failures.append("network_evidence_fields_changed")
    else:
        expected_network = checked_plan["network_contract"]
        for key in (
            "network_resource",
            "subnetwork_resource",
            "nat_name",
            "nat_router_resource",
            "nat_source_subnetwork_ip_ranges",
        ):
            if network[key] != expected_network[key]:
                failures.append(f"network_{key}_mismatch")
        if network["instance_has_external_ip"] is not False:
            failures.append("instance_external_ip_present")
        if network["external_access_config_count"] != 0:
            failures.append("external_access_config_present")
        if network["subnetwork_covered_by_nat"] is not True:
            failures.append("subnetwork_not_covered_by_nat")
        if network["nat_fully_configured"] is not True:
            failures.append("nat_not_fully_configured")

    capacity = observed.get("capacity")
    if not isinstance(capacity, Mapping) or set(capacity) != _CAPACITY_FIELDS:
        failures.append("capacity_evidence_fields_changed")
    else:
        contract = checked_plan["capacity_contract"]
        if capacity["zone_status"] != "UP":
            failures.append("zone_not_up")
        if (
            capacity["machine_type"] != contract["machine_type"]
            or capacity["machine_vcpu"] != contract["vcpu_per_vm"]
        ):
            failures.append("machine_capacity_shape_mismatch")
        if (
            capacity["requested_concurrent_vms"]
            != contract["requested_concurrent_vms"]
            or capacity["requested_vcpu"] != contract["requested_vcpu"]
        ):
            failures.append("requested_capacity_mismatch")
        quota_values: list[int] = []
        quota_valid = True
        for label, limit_key, usage_key in (
            ("global", "global_cpu_limit", "global_cpu_usage"),
            ("regional_c4", "regional_c4_cpu_limit", "regional_c4_cpu_usage"),
            (
                "regional_spot",
                "regional_spot_cpu_limit",
                "regional_spot_cpu_usage",
            ),
        ):
            limit = capacity[limit_key]
            usage = capacity[usage_key]
            if (
                type(limit) is not int
                or type(usage) is not int
                or limit < 0
                or usage < 0
                or usage > limit
            ):
                failures.append(f"{label}_quota_invalid")
                quota_valid = False
            else:
                quota_values.append(limit - usage)
        if quota_valid:
            available = min(quota_values)
            quota_vms = math.floor(available / VCPU_PER_VM)
            if capacity["available_vcpu"] != available:
                failures.append("available_vcpu_not_reconstructed")
            if capacity["quota_capacity_vms"] != quota_vms:
                failures.append("quota_capacity_vms_not_reconstructed")
            if quota_vms < MAX_CONCURRENT_VMS:
                failures.append("fresh_quota_capacity_insufficient")
        if capacity["inventory_fully_enumerated"] is not True:
            failures.append("instance_inventory_incomplete")
        if capacity["target_name_collisions"] != []:
            failures.append("target_instance_name_collision")
        if capacity["nonterminated_c4_instances"] != []:
            failures.append("nonterminated_c4_interference")
        if capacity["unknown_machine_type_instances"] != []:
            failures.append("unknown_machine_type_inventory")
        if capacity["spot_stock_proven"] is not False:
            failures.append("spot_stock_claimed_without_insert")

    return _result(
        plan_sha256=plan_sha,
        observation_sha256=observed_sha,
        evaluated_at_unix_seconds=now,
        failures=failures,
    )


__all__ = [
    "BUCKET",
    "CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL",
    "COMPUTE_SERVICE_AGENT_PRINCIPAL",
    "DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL",
    "DEFAULT_CONTROLLER_PRINCIPAL",
    "DEFAULT_NAT_NAME",
    "DEFAULT_NETWORK_RESOURCE",
    "DEFAULT_SUBNETWORK_RESOURCE",
    "DIRECT_STAGE_IDENTITY_SCHEMA",
    "MAX_AUTHORIZATION_WINDOW_SECONDS",
    "MAX_RUN_DURATION_SECONDS",
    "OBSERVATION_SCHEMA",
    "PLAN_SCHEMA",
    "PROJECT",
    "PROJECT_NUMBER",
    "REGION",
    "REQUIRED_WORKER_OAUTH_SCOPE",
    "RESULT_SCHEMA",
    "SOURCE_CONTRACT_SCHEMA",
    "VCPU_PER_VM",
    "WORKER_PRINCIPAL",
    "WORKER_SERVICE_ACCOUNT",
    "ZONE",
    "build_step11_gate_plan",
    "canonical_sha256",
    "seal_step11_gate_observation",
    "validate_step11_gate",
    "validate_step11_gate_plan",
]
