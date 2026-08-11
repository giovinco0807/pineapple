from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_step11_iam_capacity_gate_v1
    as subject,
)


NOW = 2_100_000_000
EXPIRES = NOW + 3_600
OUTER_SHA = "2" * 64
DIRECT_SHA_SEED = "f" * 64
METADATA_SHA_SEED = "c" * 64
RUN_NAME = "regular-hu-m31-r2diag-s1-20260718-001"
INSTANCE_NAMES = [
    "r2d-10c2-s1-candidate-00-a0-0ccd956a",
    "r2d-10c2-s1-candidate-00-a1-0ccd956a",
]
NAT_ROUTER_RESOURCE = (
    f"projects/{subject.PROJECT}/regions/{subject.REGION}/routers/"
    "ofc-t3-router-asia-northeast1"
)


def _source_contract() -> dict[str, Any]:
    image_self_link = (
        "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
        "global/images/debian-12-bookworm-v20260609"
    )
    inputs = {
        "outer_package_identity_sha256": OUTER_SHA,
        "machine_type": subject.MACHINE_TYPE,
        "max_attempts": 2,
        "zone": subject.ZONE,
        "worker_service_account": subject.WORKER_SERVICE_ACCOUNT,
        "run_name": RUN_NAME,
        "image_self_link": image_self_link,
        "attempt_layout": [
            {
                "attempt_index": index,
                "instance_name": name,
                "job_id": "candidate-shard-00",
            }
            for index, name in enumerate(INSTANCE_NAMES)
        ],
    }
    direct_body = {
        "schema": subject.DIRECT_STAGE_IDENTITY_SCHEMA,
        "outer_package_identity_sha256": OUTER_SHA,
        "preview_stage_identity_sha256": DIRECT_SHA_SEED,
        "inputs": inputs,
        "inputs_sha256": subject.canonical_sha256(inputs),
    }
    direct_sha = subject.canonical_sha256(direct_body)
    direct = {
        **direct_body,
        "direct_stage_identity_sha256": direct_sha,
    }
    package_prefix = (
        f"gs://{subject.BUCKET}/hu-m31-r2diag-direct-v1/packages/{OUTER_SHA}"
    )
    stage_prefix = (
        f"gs://{subject.BUCKET}/hu-m31-r2diag-direct-v1/stages/"
        f"{RUN_NAME}/{direct_sha}"
    )
    metadata = {
        "schema": "fixture-metadata-v1",
        "project": subject.PROJECT,
        "zone": subject.ZONE,
        "worker_service_account": subject.WORKER_SERVICE_ACCOUNT,
        "direct_stage_identity_sha256": direct_sha,
        "outer_package_identity_sha256": OUTER_SHA,
        "package_prefix": package_prefix,
        "stage_prefix": stage_prefix,
        "result_prefix": stage_prefix + "/results",
        "instance_name": INSTANCE_NAMES[0],
        "image": {"self_link": image_self_link},
        "fixture_digest_seed": METADATA_SHA_SEED,
    }
    return {
        "schema": subject.SOURCE_CONTRACT_SCHEMA,
        "direct_stage_identity": direct,
        "direct_stage_identity_sha256": direct_sha,
        "metadata_binding": metadata,
        "metadata_binding_sha256": subject.canonical_sha256(metadata),
    }


def _plan() -> dict[str, Any]:
    return subject.build_step11_gate_plan(
        _source_contract(),
        issued_at_unix_seconds=NOW,
        expires_at_unix_seconds=EXPIRES,
        nat_router_resource=NAT_ROUTER_RESOURCE,
    )


def _observation_body(plan: dict[str, Any]) -> dict[str, Any]:
    launch_request = deepcopy(
        plan["instance_contract"]["allowed_launch_requests"][0]
    )
    launch_request.pop("request_contract_sha256")
    return {
        "schema": subject.OBSERVATION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "collected_at_unix_seconds": NOW + 10,
        "collected_via_get_only": True,
        "collector_cloud_mutation_performed": False,
        "execution": {
            # An Owner may initiate short-lived impersonation, but its token is
            # never the bearer used for launch/control API calls.
            "initiating_principal": "user:human-owner@example.com",
            "api_bearer_principal": subject.DEFAULT_CONTROLLER_PRINCIPAL,
            "credential_subject_principal": (
                subject.DEFAULT_CONTROLLER_PRINCIPAL
            ),
            "credential_type": "service_account_impersonation",
            "controller_basic_roles": [],
            "controller_is_project_owner": False,
            "controller_is_project_editor": False,
            "owner_token_used_as_api_bearer": False,
        },
        "resource_hierarchy": {
            "project_resource": f"projects/{subject.PROJECT}",
            "project_parent": None,
            "ancestor_resources": [f"projects/{subject.PROJECT}"],
            "resource_hierarchy_complete": True,
            "allow_policies_fully_explored": True,
            "deny_policies_fully_explored": True,
            "unresolved_resources": [],
        },
        "effective_iam": {
            "custom_roles": deepcopy(plan["iam_contract"]["custom_roles"]),
            "worker_bindings": deepcopy(
                plan["iam_contract"]["worker_bindings"]
            ),
            "controller_bindings": deepcopy(
                plan["iam_contract"]["controller_bindings"]
            ),
            "worker_effective_permissions": deepcopy(
                plan["iam_contract"]["worker_effective_permissions"]
            ),
            "controller_effective_permissions": deepcopy(
                plan["iam_contract"]["controller_effective_permissions"]
            ),
            "unexpected_worker_bindings": [],
            "unexpected_controller_bindings": [],
            "conditional_binding_evaluation_errors": [],
            "denied_required_permissions": [],
            "public_principals": [],
            "default_compute_sa_can_act_as_worker": False,
            "cloud_services_sa_can_act_as_worker": False,
            "unexpected_service_account_act_as_principals": [],
            "compute_service_agent_act_as_worker": True,
        },
        "bucket_security": {
            "bucket": subject.BUCKET,
            "uniform_bucket_level_access_enabled": True,
            "public_access_prevention_mode": "inherited",
            "public_access_prevention_effective": True,
            "ancestor_public_access_prevention_fully_resolved": True,
            "legacy_object_acl_effective": False,
            "public_principals": [],
        },
        "instance_request": launch_request,
        "network": {
            "network_resource": subject.DEFAULT_NETWORK_RESOURCE,
            "subnetwork_resource": subject.DEFAULT_SUBNETWORK_RESOURCE,
            "instance_has_external_ip": False,
            "external_access_config_count": 0,
            "nat_name": subject.DEFAULT_NAT_NAME,
            "nat_router_resource": NAT_ROUTER_RESOURCE,
            "nat_source_subnetwork_ip_ranges": (
                "ALL_SUBNETWORKS_ALL_IP_RANGES"
            ),
            "subnetwork_covered_by_nat": True,
            "nat_fully_configured": True,
        },
        "capacity": {
            "zone_status": "UP",
            "machine_type": subject.MACHINE_TYPE,
            "machine_vcpu": subject.VCPU_PER_VM,
            "requested_concurrent_vms": 1,
            "requested_vcpu": subject.VCPU_PER_VM,
            "global_cpu_limit": 512,
            "global_cpu_usage": 0,
            "regional_c4_cpu_limit": 24,
            "regional_c4_cpu_usage": 0,
            "regional_spot_cpu_limit": 468,
            "regional_spot_cpu_usage": 0,
            "available_vcpu": 24,
            "quota_capacity_vms": 1,
            "inventory_fully_enumerated": True,
            "target_name_collisions": [],
            "nonterminated_c4_instances": [],
            "unknown_machine_type_instances": [],
            "spot_stock_proven": False,
        },
    }


def _observation(plan: dict[str, Any]) -> dict[str, Any]:
    return subject.seal_step11_gate_observation(_observation_body(plan))


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(value)
    candidate.pop("observation_sha256", None)
    return subject.seal_step11_gate_observation(candidate)


def _validate(
    plan: dict[str, Any], observation: dict[str, Any], *, now: int = NOW + 20
) -> dict[str, Any]:
    return subject.validate_step11_gate(
        plan,
        observation,
        evaluated_at_unix_seconds=now,
    )


def test_single_instance_condition_matches_gcp_readback_normalization() -> None:
    expression = subject._instance_and_expiry_condition(
        project=subject.PROJECT,
        zone=subject.ZONE,
        instance_names=[INSTANCE_NAMES[0]],
        expires_at="2036-07-18T18:45:00Z",
    )

    assert "(resource.name ==" not in expression
    assert expression == (
        'resource.type == "compute.googleapis.com/Instance" && '
        f'resource.name == "projects/{subject.PROJECT}/zones/{subject.ZONE}/'
        f'instances/{INSTANCE_NAMES[0]}" && '
        'request.time < timestamp("2036-07-18T18:45:00Z")'
    )


def test_plan_freezes_exact_least_privilege_and_never_authorizes_launch() -> None:
    plan = _plan()

    assert subject.validate_step11_gate_plan(plan) == plan
    iam = plan["iam_contract"]
    assert iam["worker_effective_permissions"] == [
        "compute.instances.delete",
        "storage.objects.create",
        "storage.objects.get",
    ]
    assert iam["worker_object_read_prefixes"] == [
        plan["source_contract"]["package_prefix"] + "/",
        plan["source_contract"]["result_prefix"] + "/",
    ]
    assert iam["worker_object_create_prefixes"] == [
        plan["source_contract"]["result_prefix"] + "/"
    ]
    assert "storage.objects.list" not in iam["worker_effective_permissions"]
    assert "storage.objects.delete" not in iam["worker_effective_permissions"]
    assert plan["instance_contract"]["required_oauth_scopes"] == [
        subject.REQUIRED_WORKER_OAUTH_SCOPE
    ]
    assert plan["gate_semantics"]["passing_gate_authorizes_launch"] is False
    assert plan["gate_semantics"]["cloud_api_calls_performed"] is False
    assert plan["gate_semantics"]["profile_or_current_changed"] is False


def test_valid_evidence_passes_but_spot_and_launch_remain_unproven() -> None:
    plan = _plan()
    result = _validate(plan, _observation(plan))

    assert result["failures"] == []
    assert result["iam_ubla_capacity_gate_passed"] is True
    assert result["step11_prelaunch_ready"] is True
    assert result["spot_stock_proven"] is False
    assert result["launch_authorized"] is False
    assert result["cloud_mutation_performed"] is False
    assert result["profile_or_current_changed"] is False


def test_plan_is_deterministic() -> None:
    assert _plan() == _plan()


@pytest.mark.parametrize(
    ("controller", "message"),
    [
        ("user:human-owner@example.com", "service account"),
        (subject.WORKER_PRINCIPAL, "not dedicated"),
        (subject.DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL, "not dedicated"),
        (subject.CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL, "not dedicated"),
    ],
)
def test_builder_rejects_non_dedicated_controller(
    controller: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        subject.build_step11_gate_plan(
            _source_contract(),
            issued_at_unix_seconds=NOW,
            expires_at_unix_seconds=EXPIRES,
            nat_router_resource=NAT_ROUTER_RESOURCE,
            controller_principal=controller,
        )


def test_builder_rejects_long_lived_iam_window() -> None:
    with pytest.raises(ValueError, match="short-lived"):
        subject.build_step11_gate_plan(
            _source_contract(),
            issued_at_unix_seconds=NOW,
            expires_at_unix_seconds=(
                NOW + subject.MAX_AUTHORIZATION_WINDOW_SECONDS + 1
            ),
            nat_router_resource=NAT_ROUTER_RESOURCE,
        )


def test_builder_rejects_stage_prefix_drift() -> None:
    source = _source_contract()
    source["metadata_binding"]["stage_prefix"] += "-broader"
    source["metadata_binding_sha256"] = subject.canonical_sha256(
        source["metadata_binding"]
    )

    with pytest.raises(ValueError, match="stage prefix is not exact"):
        subject.build_step11_gate_plan(
            source,
            issued_at_unix_seconds=NOW,
            expires_at_unix_seconds=EXPIRES,
            nat_router_resource=NAT_ROUTER_RESOURCE,
        )


def test_plan_hash_and_semantics_both_fail_closed() -> None:
    plan = _plan()
    plan["gate_semantics"]["passing_gate_authorizes_launch"] = True
    with pytest.raises(ValueError, match="digest mismatch"):
        subject.validate_step11_gate_plan(plan)

    plan_body = deepcopy(plan)
    plan_body.pop("plan_sha256")
    plan["plan_sha256"] = subject.canonical_sha256(plan_body)
    with pytest.raises(ValueError, match="semantics changed"):
        subject.validate_step11_gate_plan(plan)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["iam_contract"]["custom_roles"][
                "worker_self_delete"
            ]["permissions"].append("compute.instances.stop"),
            "custom roles are not exact",
        ),
        (
            lambda value: value["iam_contract"]["worker_bindings"][0][
                "condition"
            ].__setitem__(
                "expression",
                f'resource.name.startsWith("projects/_/buckets/'
                f'{subject.BUCKET}/objects/hu-m31-r2diag-direct-v1/")',
            ),
            "worker_bindings changed",
        ),
        (
            lambda value: value["instance_contract"].__setitem__(
                "required_oauth_scopes",
                [
                    subject.REQUIRED_WORKER_OAUTH_SCOPE,
                    "https://www.googleapis.com/auth/compute",
                ],
            ),
            "instance restrictions changed",
        ),
    ],
)
def test_rehashed_broader_plan_still_fails_semantic_validation(
    mutate: Mutation, message: str
) -> None:
    plan = _plan()
    mutate(plan)
    plan_body = deepcopy(plan)
    plan_body.pop("plan_sha256")
    plan["plan_sha256"] = subject.canonical_sha256(plan_body)

    with pytest.raises(ValueError, match=message):
        subject.validate_step11_gate_plan(plan)


Mutation = Callable[[dict[str, Any]], None]


def _owner_bearer(value: dict[str, Any]) -> None:
    value["execution"]["api_bearer_principal"] = "user:human-owner@example.com"
    value["execution"]["credential_subject_principal"] = (
        "user:human-owner@example.com"
    )
    value["execution"]["owner_token_used_as_api_bearer"] = True


def _broad_worker_reader(value: dict[str, Any]) -> None:
    value["effective_iam"]["worker_bindings"][0]["condition"]["expression"] = (
        f'resource.name.startsWith("projects/_/buckets/{subject.BUCKET}/'
        'objects/hu-m31-r2diag-direct-v1/")'
    )


def _extra_worker_permission(value: dict[str, Any]) -> None:
    value["effective_iam"]["worker_effective_permissions"].append(
        "storage.objects.list"
    )


def _extra_self_delete_permission(value: dict[str, Any]) -> None:
    value["effective_iam"]["custom_roles"]["worker_self_delete"][
        "permissions"
    ].append("compute.instances.stop")


def _scope_drift(value: dict[str, Any]) -> None:
    value["instance_request"]["oauth_scopes"].append(
        "https://www.googleapis.com/auth/compute"
    )


def _public_bucket(value: dict[str, Any]) -> None:
    value["bucket_security"]["public_principals"] = ["allUsers"]


def _public_project(value: dict[str, Any]) -> None:
    value["effective_iam"]["public_principals"] = ["allAuthenticatedUsers"]


def _default_sa_act_as(value: dict[str, Any]) -> None:
    value["effective_iam"]["default_compute_sa_can_act_as_worker"] = True


def _cloud_services_sa_act_as(value: dict[str, Any]) -> None:
    value["effective_iam"]["cloud_services_sa_can_act_as_worker"] = True


def _ubla_disabled(value: dict[str, Any]) -> None:
    value["bucket_security"]["uniform_bucket_level_access_enabled"] = False


def _pap_unresolved(value: dict[str, Any]) -> None:
    value["bucket_security"]["public_access_prevention_effective"] = False
    value["bucket_security"][
        "ancestor_public_access_prevention_fully_resolved"
    ] = False


def _hierarchy_unresolved(value: dict[str, Any]) -> None:
    value["resource_hierarchy"]["allow_policies_fully_explored"] = False
    value["resource_hierarchy"]["unresolved_resources"] = [
        "folders/unknown"
    ]


def _external_ip(value: dict[str, Any]) -> None:
    value["network"]["instance_has_external_ip"] = True
    value["network"]["external_access_config_count"] = 1
    value["instance_request"]["access_configs"] = [{"name": "External NAT"}]


def _nat_missing(value: dict[str, Any]) -> None:
    value["network"]["subnetwork_covered_by_nat"] = False
    value["network"]["nat_fully_configured"] = False


def _quota_insufficient(value: dict[str, Any]) -> None:
    value["capacity"]["regional_c4_cpu_usage"] = 16
    value["capacity"]["available_vcpu"] = 8
    value["capacity"]["quota_capacity_vms"] = 0


def _target_collision(value: dict[str, Any]) -> None:
    value["capacity"]["target_name_collisions"] = [INSTANCE_NAMES[0]]


def _c4_interference(value: dict[str, Any]) -> None:
    value["capacity"]["nonterminated_c4_instances"] = ["unrelated-c4-vm"]


def _zone_down(value: dict[str, Any]) -> None:
    value["capacity"]["zone_status"] = "DOWN"


def _collector_mutated(value: dict[str, Any]) -> None:
    value["collector_cloud_mutation_performed"] = True


@pytest.mark.parametrize(
    ("mutate", "failure"),
    [
        (_owner_bearer, "api_bearer_is_not_dedicated_controller"),
        (_broad_worker_reader, "worker_bindings_mismatch"),
        (_extra_worker_permission, "worker_effective_permissions_not_exact"),
        (_extra_self_delete_permission, "custom_roles_mismatch"),
        (_scope_drift, "worker_oauth_scope_not_exact"),
        (_public_bucket, "bucket_public_principal_present"),
        (_public_project, "public_iam_principal_present"),
        (_default_sa_act_as, "default_compute_sa_can_act_as_worker"),
        (
            _cloud_services_sa_act_as,
            "cloud_services_sa_can_act_as_worker",
        ),
        (_ubla_disabled, "uniform_bucket_level_access_disabled"),
        (_pap_unresolved, "public_access_prevention_not_effective"),
        (_hierarchy_unresolved, "allow_policies_not_fully_explored"),
        (_external_ip, "instance_external_ip_present"),
        (_nat_missing, "subnetwork_not_covered_by_nat"),
        (_quota_insufficient, "fresh_quota_capacity_insufficient"),
        (_target_collision, "target_instance_name_collision"),
        (_c4_interference, "nonterminated_c4_interference"),
        (_zone_down, "zone_not_up"),
        (_collector_mutated, "collector_cloud_mutation_detected"),
    ],
)
def test_security_and_capacity_drift_fail_closed(
    mutate: Mutation, failure: str
) -> None:
    plan = _plan()
    observed = _observation(plan)
    mutate(observed)
    result = _validate(plan, _reseal(observed))

    assert failure in result["failures"]
    assert result["iam_ubla_capacity_gate_passed"] is False
    assert result["launch_authorized"] is False


def test_stale_observation_fails_closed() -> None:
    plan = _plan()
    observed = _observation(plan)
    observed["collected_at_unix_seconds"] = NOW - 301
    result = _validate(plan, _reseal(observed), now=NOW)

    assert "observation_stale" in result["failures"]
    assert result["launch_authorized"] is False


def test_unsealed_tamper_is_detected() -> None:
    plan = _plan()
    observed = _observation(plan)
    observed["capacity"]["regional_c4_cpu_limit"] = 355

    result = _validate(plan, observed)

    assert "observation_sha256_mismatch" in result["failures"]
    assert result["launch_authorized"] is False


def test_unknown_observation_field_is_rejected_before_use() -> None:
    plan = _plan()
    observed = _observation(plan)
    observed["launch_authorized"] = True

    result = _validate(plan, observed)

    assert result["failures"] == ["observation_fields_changed"]
    assert result["launch_authorized"] is False


def test_current_broad_pre_migration_shape_fails_closed() -> None:
    """Mirror the audited broad viewer/creator + default-SA actAs hazards."""

    plan = _plan()
    observed = _observation(plan)
    observed["effective_iam"]["worker_bindings"][0]["role"] = (
        "roles/storage.objectViewer"
    )
    observed["effective_iam"]["worker_bindings"][1]["role"] = (
        "roles/storage.objectCreator"
    )
    observed["effective_iam"]["default_compute_sa_can_act_as_worker"] = True
    observed["effective_iam"][
        "cloud_services_sa_can_act_as_worker"
    ] = True

    result = _validate(plan, _reseal(observed))

    assert "worker_bindings_mismatch" in result["failures"]
    assert "default_compute_sa_can_act_as_worker" in result["failures"]
    assert "cloud_services_sa_can_act_as_worker" in result["failures"]
    assert result["step11_prelaunch_ready"] is False
    assert result["launch_authorized"] is False
