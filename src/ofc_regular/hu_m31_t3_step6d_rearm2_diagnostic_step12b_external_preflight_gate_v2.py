"""Concrete, non-mutating Step 12b external preflight evidence gate.

The signed external authorization module deliberately cannot mint its opaque
preflight capability from caller-supplied hashes.  This module is the owning
gate.  It validates the complete evidence bodies and the exact local source
bytes before handing one non-serializable capability to the authorization
module.

The gate itself calls no cloud API and performs no mutation.  It runs after
the fresh bootstrap-source upload, controller-service-account creation, token
barrier, and Phase-2 IAM install; their exact receipts/capabilities are bound
as upstream mutations.  Final provider observations are collected by a
separate GET-only adapter and supplied as complete JSON bodies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as readonly_preflight,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_authorization_v2
    as external_auth,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_startup_loader_v2
    as startup_loader,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_external_preflight_gate_v2"
)
STATUS = (
    "all_concrete_external_preflight_evidence_validated_"
    "signed_authorization_still_required"
)
DIRECT_PREFIX_EMPTY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_direct_v2_prefix_empty_v2"
)
COMPUTE_ABSENCE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_exact_pair_compute_absence_v2"
)
LIVE_READBACK_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_live_capacity_nat_services_roles_v2"
)

PROJECT_NUMBER = external_auth.EXPECTED_PROJECT_NUMBER
OBSERVATION_MAX_AGE_SECONDS = 300
EXPECTED_PACKAGE_OBJECT_COUNT = 16
EXPECTED_PACKAGE_PROVISION_RECEIPT_SHA256 = (
    "d1994fb4c759d437dfbcd4cddbabb625a85e830ca5d3c6f95750c969972c97d6"
)
REQUIRED_ENABLED_SERVICES = frozenset(
    {
        "compute.googleapis.com",
        "iamcredentials.googleapis.com",
        "serviceusage.googleapis.com",
        "storage.googleapis.com",
    }
)
NAT_ROUTER_NAME = "ofc-t3-nat-router-asia-northeast1"
NAT_NAME = "ofc-t3-nat-asia-northeast1"
NAT_SOURCE_SUBNETWORK_IP_RANGES = "ALL_SUBNETWORKS_ALL_IP_RANGES"
NAT_IP_ALLOCATE_OPTION = "AUTO_ONLY"

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS: Mapping[str, str] = MappingProxyType({
    "alias_bridge": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_step12b_alias_bridge_v2.py"
    ),
    "controller_helper": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1.py"
    ),
    "controller_sa_lifecycle": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_run_scoped_controller_sa_v2.py"
    ),
    "bootstrap_source": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_bootstrap_source_v2.py"
    ),
    "bootstrap_source_content": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_bootstrap_source_content_v2.py"
    ),
    "deployment": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_deployment_contract_v2.py"
    ),
    "external_authorization": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_external_authorization_v2.py"
    ),
    "external_preflight_gate": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_external_preflight_gate_v2.py"
    ),
    "pair_release": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2.py"
    ),
    "payload_transport": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "canary_gce_transport_10c2_v1.py"
    ),
    "phase2_iam_plan": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_phase2_iam_plan_v2.py"
    ),
    "rest_iam_admin": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1.py"
    ),
    "prebootstrap": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_vm_prebootstrap_v2.py"
    ),
    "startup": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_startup_loader_v2.py"
    ),
    "token_barrier": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1.py"
    ),
    "vm_metadata": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2.py"
    ),
    "vm_prebootstrap": (
        "src/ofc_regular/"
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_vm_prebootstrap_v2.py"
    ),
})

BOOTSTRAP_RUNTIME_SOURCE_LOGICAL_NAMES: Mapping[str, str] = MappingProxyType(
    {
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_vm_prebootstrap_v2.py"
        ): "vm_prebootstrap",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_bootstrap_source_content_v2.py"
        ): "bootstrap_source_content",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_bootstrap_source_v2.py"
        ): "bootstrap_source",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_alias_bridge_v2.py"
        ): "alias_bridge",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_pair_release_v2.py"
        ): "pair_release",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_vm_metadata_v2.py"
        ): "vm_metadata",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_external_authorization_v2.py"
        ): "external_authorization",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_deployment_contract_v2.py"
        ): "deployment",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_phase2_iam_plan_v2.py"
        ): "phase2_iam_plan",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_run_scoped_controller_sa_v2.py"
        ): "controller_sa_lifecycle",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_rest_iam_admin_v1.py"
        ): "rest_iam_admin",
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "canary_gce_transport_10c2_v1.py"
        ): "payload_transport",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_FIELD_PARTS = (
    "access_token",
    "authorization_header",
    "private_key",
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
)
_PREFIX_FIELDS = {
    "schema",
    "status",
    "deployment_contract_sha256",
    "direct_stage_identity_sha256",
    "stage_prefix",
    "observed_at_unix_seconds",
    "observation_max_age_seconds",
    "provider_pages",
    "provider_pages_sha256",
    "page_count",
    "object_count",
    "page_tokens_exhausted",
    "collected_via_get_only",
    "cloud_mutation_performed",
    "receipt_sha256",
}
_COMPUTE_FIELDS = {
    "schema",
    "status",
    "deployment_contract_sha256",
    "direct_stage_identity_sha256",
    "project",
    "zone",
    "instance_get_readbacks",
    "disk_get_readbacks",
    "operation_name_history",
    "observed_at_unix_seconds",
    "observation_max_age_seconds",
    "provider_get_404_count",
    "operation_name_history_count",
    "collected_via_get_only",
    "cloud_mutation_performed",
    "receipt_sha256",
}


def canonical_bytes(value: Any) -> bytes:
    return deployment_v2.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return deployment_v2.canonical_sha256(value)


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_bytes(value))


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _integer(
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


def _reject_forbidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string field")
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_FIELD_PARTS):
                raise ValueError(f"{path}.{key} contains a forbidden field")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in body:
        raise ValueError("receipt body is already sealed")
    checked = _json_copy(body)
    _reject_forbidden(checked)
    return {**checked, "receipt_sha256": canonical_sha256(checked)}


def _unseal(
    value: Mapping[str, Any],
    *,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    checked = _json_copy(value)
    _exact(checked, fields, label)
    supplied = _sha(checked.pop("receipt_sha256"), f"{label} digest")
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return {**checked, "receipt_sha256": supplied}


def _fresh(
    observed_at_unix_seconds: Any,
    *,
    now_unix_seconds: int,
    label: str,
) -> None:
    observed = _integer(
        observed_at_unix_seconds,
        f"{label} observation time",
        minimum=1,
    )
    age = now_unix_seconds - observed
    if age < 0 or age > OBSERVATION_MAX_AGE_SECONDS:
        raise ValueError(f"{label} evidence is stale")


def _validated_deployment(
    deployment_contract: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deployment = deployment_v2.validate_deployment_contract(
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
    candidate_inventory = candidate["remote_layout"]["package_inventory"]
    reference_inventory = reference["remote_layout"]["package_inventory"]
    if (
        candidate_inventory != reference_inventory
        or candidate_inventory["records_sha256"]
        != deployment["payload_binding"][
            "package_inventory_records_sha256"
        ]
        or candidate_inventory["package_prefix"]
        != deployment["payload_binding"]["package_prefix"]
    ):
        raise ValueError("immutable package inventory identity changed")
    return deployment, candidate, reference


def build_direct_v2_prefix_empty_receipt(
    deployment_contract: Mapping[str, Any],
    *,
    observed_at_unix_seconds: int,
    provider_pages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a receipt from complete GCS list-response bodies."""

    deployment = dict(deployment_contract)
    stage_prefix = deployment.get("remote_layout", {}).get("stage_prefix")
    expected_marker = (
        f"gs://{payload_transport.BUCKET}/"
        f"{deployment_v2.DIRECT_NAMESPACE}/stages/"
    )
    if (
        not isinstance(stage_prefix, str)
        or not stage_prefix.startswith(expected_marker)
        or stage_prefix.count("/") < 5
    ):
        raise ValueError("direct-v2 stage prefix changed")
    observed = _integer(
        observed_at_unix_seconds,
        "direct-v2 prefix observation time",
        minimum=1,
    )
    if (
        isinstance(provider_pages, (str, bytes))
        or not isinstance(provider_pages, Sequence)
        or not 1 <= len(provider_pages) <= 10
    ):
        raise ValueError("direct-v2 provider page count changed")
    normalized_pages: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for position, raw in enumerate(provider_pages):
        if not isinstance(raw, Mapping):
            raise ValueError("direct-v2 provider page is not an object")
        page = _json_copy(raw)
        if not set(page).issubset({"items", "nextPageToken"}):
            raise ValueError("direct-v2 provider page fields changed")
        items = page.get("items", [])
        token = page.get("nextPageToken")
        if not isinstance(items, list) or items:
            raise FileExistsError("direct-v2 stage prefix is not empty")
        if token is not None and (
            not isinstance(token, str)
            or not token
            or len(token) > 2_048
            or token in seen_tokens
            or position == len(provider_pages) - 1
        ):
            raise ValueError("direct-v2 pagination changed")
        if token is not None:
            seen_tokens.add(token)
        normalized_pages.append(page)
    if normalized_pages[-1].get("nextPageToken") is not None:
        raise ValueError("direct-v2 page tokens were not exhausted")
    body = {
        "schema": DIRECT_PREFIX_EMPTY_SCHEMA,
        "status": "fresh_direct_v2_stage_prefix_exactly_empty",
        "deployment_contract_sha256": deployment.get(
            "deployment_contract_sha256"
        ),
        "direct_stage_identity_sha256": deployment.get(
            "direct_stage_identity_sha256"
        ),
        "stage_prefix": stage_prefix,
        "observed_at_unix_seconds": observed,
        "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
        "provider_pages": normalized_pages,
        "provider_pages_sha256": canonical_sha256(normalized_pages),
        "page_count": len(normalized_pages),
        "object_count": 0,
        "page_tokens_exhausted": True,
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
    }
    return _seal(body)


def validate_direct_v2_prefix_empty_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    checked = _unseal(
        value,
        fields=_PREFIX_FIELDS,
        label="direct-v2 prefix-empty receipt",
    )
    rebuilt = build_direct_v2_prefix_empty_receipt(
        deployment_contract,
        observed_at_unix_seconds=checked["observed_at_unix_seconds"],
        provider_pages=checked["provider_pages"],
    )
    if checked != rebuilt:
        raise ValueError("direct-v2 prefix-empty receipt changed")
    return rebuilt


def _absence_rows(
    values: Sequence[Mapping[str, Any]],
    *,
    expected_names: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != len(expected_names)
    ):
        raise ValueError(f"{label} GET readback count changed")
    by_name: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} GET readback is not an object")
        row = _json_copy(raw)
        _exact(row, {"name", "http_status"}, f"{label} GET readback")
        name = row["name"]
        if (
            not isinstance(name, str)
            or name in by_name
            or row["http_status"] != 404
        ):
            raise FileExistsError(f"{label} is not exactly absent")
        by_name[name] = row
    if set(by_name) != set(expected_names):
        raise ValueError(f"{label} GET names changed")
    return [by_name[name] for name in expected_names]


def build_compute_absence_receipt(
    deployment_contract: Mapping[str, Any],
    *,
    observed_at_unix_seconds: int,
    instance_get_readbacks: Sequence[Mapping[str, Any]],
    disk_get_readbacks: Sequence[Mapping[str, Any]],
    operation_name_history: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Bind the exact pair's instance/disk GET 404 and zero name history."""

    deployment = dict(deployment_contract)
    instances = deployment.get("instances")
    if not isinstance(instances, list) or len(instances) != 2:
        raise ValueError("exact pair instance contract changed")
    names = [row.get("instance_name") for row in instances]
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 2
    ):
        raise ValueError("exact pair instance names changed")
    instance_rows = _absence_rows(
        instance_get_readbacks,
        expected_names=names,
        label="instance",
    )
    disk_rows = _absence_rows(
        disk_get_readbacks,
        expected_names=names,
        label="disk",
    )
    if not isinstance(operation_name_history, Mapping):
        raise ValueError("operation/name history is missing")
    if set(operation_name_history) != set(names):
        raise ValueError("operation/name history names changed")
    normalized_history: dict[str, list[Any]] = {}
    for name in names:
        rows = operation_name_history[name]
        if (
            isinstance(rows, (str, bytes))
            or not isinstance(rows, Sequence)
            or list(rows) != []
        ):
            raise FileExistsError(
                "operation/name history is not exactly zero"
            )
        normalized_history[name] = []
    body = {
        "schema": COMPUTE_ABSENCE_SCHEMA,
        "status": (
            "exact_two_instances_and_disks_get404_operation_name_history0"
        ),
        "deployment_contract_sha256": deployment.get(
            "deployment_contract_sha256"
        ),
        "direct_stage_identity_sha256": deployment.get(
            "direct_stage_identity_sha256"
        ),
        "project": payload_transport.PROJECT,
        "zone": payload_transport.ZONE,
        "instance_get_readbacks": instance_rows,
        "disk_get_readbacks": disk_rows,
        "operation_name_history": normalized_history,
        "observed_at_unix_seconds": _integer(
            observed_at_unix_seconds,
            "compute absence observation time",
            minimum=1,
        ),
        "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
        "provider_get_404_count": 4,
        "operation_name_history_count": 0,
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
    }
    return _seal(body)


def validate_compute_absence_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    checked = _unseal(
        value,
        fields=_COMPUTE_FIELDS,
        label="compute absence receipt",
    )
    rebuilt = build_compute_absence_receipt(
        deployment_contract,
        observed_at_unix_seconds=checked["observed_at_unix_seconds"],
        instance_get_readbacks=checked["instance_get_readbacks"],
        disk_get_readbacks=checked["disk_get_readbacks"],
        operation_name_history=checked["operation_name_history"],
    )
    if checked != rebuilt:
        raise ValueError("compute absence receipt changed")
    return rebuilt


def _capacity_target(
    deployment: Mapping[str, Any],
) -> readonly_preflight._Target:
    names = tuple(
        row["instance_name"] for row in deployment["instances"]
    )
    return readonly_preflight._Target(
        project=payload_transport.PROJECT,
        bucket=payload_transport.BUCKET,
        region=payload_transport.ZONE.rsplit("-", 1)[0],
        zone=payload_transport.ZONE,
        machine_type=deployment_v2.ACTUAL_MACHINE_TYPE,
        package_object_prefix="step12b-capacity-only/package",
        stage_object_prefix="step12b-capacity-only/stage",
        expected_instance_names=names,
        service_account_email=payload_transport.WORKER_SERVICE_ACCOUNT,
        image_project=readonly_preflight.IMAGE_PROJECT,
        image_name=readonly_preflight.IMAGE_NAME,
        image_id=readonly_preflight.IMAGE_ID,
        image_self_link=readonly_preflight.IMAGE_SELF_LINK,
    )


def _capacity_facts(
    observations: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    expected_endpoints = {
        "cloud_quotas_global_cpu",
        "cloud_quotas_c4_cpu",
        *readonly_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS,
    }
    if not isinstance(observations, Mapping) or set(observations) != (
        expected_endpoints
    ):
        raise ValueError("authoritative capacity endpoint set changed")
    target = _capacity_target(deployment)
    inventory_facts = [
        readonly_preflight._empty_compute_inventory_facts(
            observations[endpoint_id],
            endpoint_id=endpoint_id,
            target=target,
        )
        for endpoint_id in (
            readonly_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS
        )
    ]
    global_cpu = readonly_preflight._global_cpu_quota_facts(
        observations["cloud_quotas_global_cpu"],
        inventory_facts,
        target=target,
        project_number=PROJECT_NUMBER,
    )
    c4_cpu = readonly_preflight._regional_c4_quota_facts(
        observations["cloud_quotas_c4_cpu"],
        target=target,
        project_number=PROJECT_NUMBER,
        usage_inventory_proof=global_cpu["usage_inventory_proof"],
    )
    instance_observation = observations["compute_aggregated_instances"]
    instance_records: list[dict[str, Any]] = []
    for scope, scope_value in instance_observation["items"].items():
        rows = scope_value.get("instances")
        if rows is None:
            continue
        instance_records.extend(
            readonly_preflight._instance_inventory_record(
                row,
                scope=scope,
                target=target,
            )
            for row in rows
        )
    target_names = set(target.expected_instance_names)
    collisions = sorted(
        row["name"] for row in instance_records if row["name"] in target_names
    )
    interfering = sorted(
        row["name"]
        for row in instance_records
        if row["target_region_c4"] and row["status"] != "TERMINATED"
    )
    requested = deployment_v2.ACTUAL_VCPUS_PER_VM * deployment_v2.VM_COUNT
    if (
        collisions
        or interfering
        or Decimal(global_cpu["available"]) < requested
        or Decimal(c4_cpu["available"]) < requested
    ):
        raise ValueError("authoritative Step12b capacity is insufficient")
    return {
        "global_cpu": global_cpu,
        "regional_c4": c4_cpu,
        "inventory_facts": inventory_facts,
        "target_name_collisions": collisions,
        "nonterminated_c4_interference": interfering,
        "requested_vcpu": requested,
        "available_vcpu": str(
            min(
                Decimal(global_cpu["available"]),
                Decimal(c4_cpu["available"]),
            )
        ),
    }


def _finite_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{label} is not a nonnegative integer")
    return int(number)


def _machine_region_facts(
    machine: Mapping[str, Any],
    region: Mapping[str, Any],
) -> dict[str, Any]:
    project = payload_transport.PROJECT
    zone = payload_transport.ZONE
    region_name = zone.rsplit("-", 1)[0]
    machine_name = deployment_v2.ACTUAL_MACHINE_TYPE
    expected_machine_suffix = (
        f"/projects/{project}/zones/{zone}/machineTypes/{machine_name}"
    )
    expected_region_suffix = f"/projects/{project}/regions/{region_name}"
    accepted_zone_values = {
        zone,
        (
            f"https://www.googleapis.com/compute/v1/projects/{project}/"
            f"zones/{zone}"
        ),
    }
    if (
        machine.get("name") != machine_name
        or machine.get("guestCpus") != deployment_v2.ACTUAL_VCPUS_PER_VM
        or machine.get("memoryMb") != deployment_v2.ACTUAL_MEMORY_MB
        or machine.get("zone") not in accepted_zone_values
        or not str(machine.get("selfLink", "")).endswith(
            expected_machine_suffix
        )
        or region.get("name") != region_name
        or region.get("status") != "UP"
        or not str(region.get("selfLink", "")).endswith(
            expected_region_suffix
        )
    ):
        raise ValueError("machine or region readback changed")
    quotas = region.get("quotas")
    if not isinstance(quotas, list):
        raise ValueError("regional quota readback is missing")
    wanted = {"CPUS", "PREEMPTIBLE_CPUS"}
    facts: dict[str, dict[str, int]] = {}
    for raw in quotas:
        if not isinstance(raw, Mapping) or raw.get("metric") not in wanted:
            continue
        metric = raw["metric"]
        if metric in facts:
            raise ValueError("regional quota metric is duplicated")
        limit = _finite_nonnegative_int(
            raw.get("limit"), f"{metric} limit"
        )
        usage = _finite_nonnegative_int(
            raw.get("usage"), f"{metric} usage"
        )
        if usage > limit:
            raise ValueError("regional quota usage exceeds limit")
        facts[metric] = {
            "limit": limit,
            "usage": usage,
            "available": limit - usage,
        }
    requested = deployment_v2.ACTUAL_VCPUS_PER_VM * deployment_v2.VM_COUNT
    if set(facts) != wanted or min(
        row["available"] for row in facts.values()
    ) < requested:
        raise ValueError("regional quota does not cover the exact pair")
    return {
        "machine_type": machine_name,
        "guest_cpus": deployment_v2.ACTUAL_VCPUS_PER_VM,
        "memory_mb": deployment_v2.ACTUAL_MEMORY_MB,
        "region": region_name,
        "region_status": "UP",
        "regional_quotas": facts,
    }


def _nat_facts(value: Mapping[str, Any]) -> dict[str, Any]:
    project = payload_transport.PROJECT
    region = payload_transport.ZONE.rsplit("-", 1)[0]
    expected_network_path = (
        f"/compute/v1/projects/{project}/global/networks/default"
    )
    observed_region = value.get("region")
    observed_network = value.get("network")
    nats = value.get("nats")
    matches = (
        [
            row
            for row in nats
            if isinstance(row, Mapping) and row.get("name") == NAT_NAME
        ]
        if isinstance(nats, list)
        else []
    )
    if (
        value.get("name") != NAT_ROUTER_NAME
        or not isinstance(observed_region, str)
        or observed_region.rsplit("/", 1)[-1] != region
        or not isinstance(observed_network, str)
        or urlsplit(observed_network).path != expected_network_path
        or len(matches) != 1
    ):
        raise ValueError("Cloud NAT router identity changed")
    nat = matches[0]
    if (
        nat.get("sourceSubnetworkIpRangesToNat")
        != NAT_SOURCE_SUBNETWORK_IP_RANGES
        or nat.get("natIpAllocateOption") != NAT_IP_ALLOCATE_OPTION
    ):
        raise ValueError("Cloud NAT configuration changed")
    return {
        "router_name": NAT_ROUTER_NAME,
        "region": region,
        "network_path": expected_network_path,
        "nat_name": NAT_NAME,
        "source_subnetwork_ip_ranges": NAT_SOURCE_SUBNETWORK_IP_RANGES,
        "nat_ip_allocate_option": NAT_IP_ALLOCATE_OPTION,
    }


def _service_facts(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("enabled-service readbacks are missing")
    enabled: set[str] = set()
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError("enabled-service readback is not an object")
        config = raw.get("config")
        name = config.get("name") if isinstance(config, Mapping) else None
        if (
            raw.get("state") != "ENABLED"
            or not isinstance(name, str)
            or name in enabled
        ):
            raise ValueError("enabled-service inventory changed")
        enabled.add(name)
    if not REQUIRED_ENABLED_SERVICES.issubset(enabled):
        raise ValueError("required Step12b service is not enabled")
    return {
        "required_services": sorted(REQUIRED_ENABLED_SERVICES),
        "enabled_services": sorted(enabled),
        "missing_required_services": [],
    }


def _custom_role_facts(
    values: Mapping[str, Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise ValueError("custom-role readbacks are missing")
    requirements = {
        row["purpose"]: row
        for row in plan["custom_role_readback_contract"]["requirements"]
    }
    if set(values) != set(requirements):
        raise ValueError("custom-role readback purpose set changed")
    facts: list[dict[str, Any]] = []
    for purpose in sorted(requirements):
        expected = requirements[purpose]
        raw = values[purpose]
        if not isinstance(raw, Mapping):
            raise ValueError("custom-role readback is not an object")
        permissions = raw.get("includedPermissions")
        if (
            raw.get("name") != expected["name"]
            or raw.get("stage") != expected["stage"]
            or raw.get("deleted", False) is not False
            or not isinstance(permissions, list)
            or sorted(permissions) != expected["included_permissions"]
            or len(permissions) != len(set(permissions))
        ):
            raise ValueError(f"custom role changed: {purpose}")
        facts.append(
            {
                "purpose": purpose,
                "name": expected["name"],
                "stage": expected["stage"],
                "included_permissions": sorted(permissions),
                "provider_record_sha256": canonical_sha256(raw),
            }
        )
    return facts


def build_live_readback_receipt(
    deployment_contract: Mapping[str, Any],
    *,
    phase2_iam_plan: Mapping[str, Any],
    observed_at_unix_seconds: int,
    machine_type_readback: Mapping[str, Any],
    region_readback: Mapping[str, Any],
    authoritative_capacity_observations: Mapping[str, Any],
    nat_router_readback: Mapping[str, Any],
    enabled_service_readbacks: Sequence[Mapping[str, Any]],
    custom_role_readbacks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and retain complete GET-only provider readback bodies."""

    deployment = _json_copy(deployment_contract)
    plan = _json_copy(phase2_iam_plan)
    if (
        plan.get("source_deployment", {}).get(
            "deployment_contract_sha256"
        )
        != deployment.get("deployment_contract_sha256")
    ):
        raise ValueError("live readback Phase2 plan binding changed")
    provider = {
        "machine_type": _json_copy(machine_type_readback),
        "region": _json_copy(region_readback),
        "authoritative_capacity": _json_copy(
            authoritative_capacity_observations
        ),
        "nat_router": _json_copy(nat_router_readback),
        "enabled_services": _json_copy(enabled_service_readbacks),
        "custom_roles": _json_copy(custom_role_readbacks),
    }
    facts = {
        "machine_and_region": _machine_region_facts(
            provider["machine_type"], provider["region"]
        ),
        "capacity": _capacity_facts(
            provider["authoritative_capacity"],
            deployment=deployment,
        ),
        "nat": _nat_facts(provider["nat_router"]),
        "services": _service_facts(provider["enabled_services"]),
        "custom_roles": _custom_role_facts(
            provider["custom_roles"], plan=plan
        ),
    }
    body = {
        "schema": LIVE_READBACK_SCHEMA,
        "status": (
            "fresh_capacity_nat_services_and_custom_roles_exact_get_readback"
        ),
        "deployment_contract_sha256": deployment.get(
            "deployment_contract_sha256"
        ),
        "direct_stage_identity_sha256": deployment.get(
            "direct_stage_identity_sha256"
        ),
        "phase2_iam_plan_sha256": plan.get("plan_sha256"),
        "observed_at_unix_seconds": _integer(
            observed_at_unix_seconds,
            "live readback observation time",
            minimum=1,
        ),
        "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
        "provider_readbacks": provider,
        "provider_readbacks_sha256": canonical_sha256(provider),
        "facts": facts,
        "facts_sha256": canonical_sha256(facts),
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return _seal(body)


def validate_live_readback_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    phase2_iam_plan: Mapping[str, Any],
) -> dict[str, Any]:
    checked = _json_copy(value)
    expected_fields = {
        "schema",
        "status",
        "deployment_contract_sha256",
        "direct_stage_identity_sha256",
        "phase2_iam_plan_sha256",
        "observed_at_unix_seconds",
        "observation_max_age_seconds",
        "provider_readbacks",
        "provider_readbacks_sha256",
        "facts",
        "facts_sha256",
        "collected_via_get_only",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
    checked = _unseal(
        checked,
        fields=expected_fields,
        label="live readback receipt",
    )
    provider = checked["provider_readbacks"]
    if not isinstance(provider, Mapping):
        raise ValueError("live provider readbacks are missing")
    _exact(
        provider,
        {
            "machine_type",
            "region",
            "authoritative_capacity",
            "nat_router",
            "enabled_services",
            "custom_roles",
        },
        "live provider readbacks",
    )
    rebuilt = build_live_readback_receipt(
        deployment_contract,
        phase2_iam_plan=phase2_iam_plan,
        observed_at_unix_seconds=checked["observed_at_unix_seconds"],
        machine_type_readback=provider["machine_type"],
        region_readback=provider["region"],
        authoritative_capacity_observations=provider[
            "authoritative_capacity"
        ],
        nat_router_readback=provider["nat_router"],
        enabled_service_readbacks=provider["enabled_services"],
        custom_role_readbacks=provider["custom_roles"],
    )
    if checked != rebuilt:
        raise ValueError("live readback receipt changed")
    return rebuilt


def validate_immutable_package_provision_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the retained immutable package's exact 16 pinned objects."""

    deployment = dict(deployment_contract)
    candidate = payload_transport.validate_job_contract(
        candidate_payload_contract
    )
    reference = payload_transport.validate_job_contract(
        reference_payload_contract
    )
    candidate_inventory = candidate["remote_layout"]["package_inventory"]
    reference_inventory = reference["remote_layout"]["package_inventory"]
    if candidate_inventory != reference_inventory:
        raise ValueError("candidate/reference package inventories differ")
    receipt = _json_copy(value)
    expected_fields = {
        "schema",
        "status",
        "contract_sha256",
        "outer_package_identity_sha256",
        "package_prefix",
        "records",
        "records_sha256",
        "package_generations",
        "package_generations_sha256",
        "object_count",
        "all_generation_bound",
        "all_bytes_and_sha256_read_back",
        "vm_created",
        "claim_created",
        "diagnostic_only",
        "receipt_sha256",
    }
    _exact(receipt, expected_fields, "package provision receipt")
    supplied = _sha(
        receipt.pop("receipt_sha256"),
        "package provision receipt",
    )
    if (
        canonical_sha256(receipt) != supplied
        or supplied != EXPECTED_PACKAGE_PROVISION_RECEIPT_SHA256
    ):
        raise ValueError("immutable package provision receipt changed")
    records = receipt["records"]
    inventory_records = candidate_inventory["records"]
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_PACKAGE_OBJECT_COUNT
        or len(inventory_records) != EXPECTED_PACKAGE_OBJECT_COUNT
    ):
        raise ValueError("immutable package object count changed")
    generations: dict[str, int] = {}
    for observed, expected in zip(records, inventory_records, strict=True):
        if not isinstance(observed, Mapping):
            raise ValueError("package readback record is not an object")
        _exact(
            observed,
            {
                "uri",
                "generation",
                "created",
                "sha256",
                "bytes",
                "crc32c",
                "etag",
            },
            "package readback record",
        )
        generation = _integer(
            observed["generation"], "package generation", minimum=1
        )
        if (
            observed["uri"] != expected["uri"]
            or observed["sha256"] != expected["sha256"]
            or observed["bytes"] != expected["bytes"]
            or type(observed["created"]) is not bool
            or not isinstance(observed["crc32c"], str)
            or not observed["crc32c"]
            or not isinstance(observed["etag"], str)
            or not observed["etag"]
            or observed["uri"] in generations
        ):
            raise ValueError("package generation/SHA/bytes changed")
        generations[observed["uri"]] = generation
    if (
        receipt["schema"]
        != (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_package_provision_v1"
        )
        or receipt["status"]
        != "exact_package_generation_zero_created_or_identical_readback"
        or receipt["outer_package_identity_sha256"]
        != deployment["payload_binding"]["outer_package_identity_sha256"]
        or receipt["package_prefix"]
        != deployment["payload_binding"]["package_prefix"]
        or receipt["records_sha256"] != canonical_sha256(records)
        or receipt["package_generations"] != generations
        or receipt["package_generations_sha256"]
        != canonical_sha256(generations)
        or receipt["object_count"] != EXPECTED_PACKAGE_OBJECT_COUNT
        or receipt["all_generation_bound"] is not True
        or receipt["all_bytes_and_sha256_read_back"] is not True
        or receipt["vm_created"] is not False
        or receipt["claim_created"] is not False
        or receipt["diagnostic_only"] is not True
        or candidate_inventory["records_sha256"]
        != deployment["payload_binding"][
            "package_inventory_records_sha256"
        ]
    ):
        raise ValueError("immutable package provision semantics changed")
    return {**receipt, "receipt_sha256": supplied}


def load_expected_source_bytes() -> dict[str, bytes]:
    """Read the exact controller-side sources named by this gate."""

    values: dict[str, bytes] = {}
    for logical_name, relative in SOURCE_PATHS.items():
        path = _REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"preflight source is not a real file: {relative}")
        values[logical_name] = path.read_bytes()
    return values


def _source_identity(
    source_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    if not isinstance(source_bytes, Mapping) or set(source_bytes) != set(
        SOURCE_PATHS
    ):
        raise ValueError("preflight source byte set changed")
    records: list[dict[str, Any]] = []
    for logical_name in sorted(SOURCE_PATHS):
        relative = SOURCE_PATHS[logical_name]
        path = _REPO_ROOT / relative
        supplied = source_bytes[logical_name]
        if (
            not isinstance(supplied, bytes)
            or not supplied
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError(f"preflight source bytes changed: {logical_name}")
        current = path.read_bytes()
        if supplied != current:
            raise ValueError(f"preflight source bytes changed: {logical_name}")
        digest = hashlib.sha256(supplied).hexdigest()
        if digest == "aa" * 32 or digest == "0" * 64:
            raise ValueError("placeholder preflight source digest rejected")
        records.append(
            {
                "logical_name": logical_name,
                "path": relative,
                "bytes": len(supplied),
                "sha256": digest,
            }
        )
    return {
        "records": records,
        "record_count": len(records),
        "records_sha256": canonical_sha256(records),
        "all_source_bytes_read_and_hashed": True,
        "all_source_bytes_match_workspace": True,
    }


def _current_profile_identity(
    current_profile_bytes: bytes,
    *,
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    path = _REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
    if (
        not isinstance(current_profile_bytes, bytes)
        or not current_profile_bytes
        or not path.is_file()
        or path.is_symlink()
        or current_profile_bytes != path.read_bytes()
    ):
        raise ValueError("current profile bytes changed")
    digest = hashlib.sha256(current_profile_bytes).hexdigest()
    if (
        digest != deployment_v2.EXPECTED_CURRENT_PROFILE_SHA256
        or digest != deployment["current_profile_sha256"]
    ):
        raise ValueError("current profile hash changed")
    return {
        "path": "src/ofc_regular/ai_profiles.py",
        "bytes": len(current_profile_bytes),
        "sha256": digest,
        "bytes_match_workspace": True,
        "current_profile_changed": False,
    }


def _source_sha(
    source_identity: Mapping[str, Any], logical_name: str
) -> str:
    matches = [
        row["sha256"]
        for row in source_identity["records"]
        if row["logical_name"] == logical_name
    ]
    if len(matches) != 1:
        raise AssertionError("source identity lookup changed")
    return matches[0]


def _validate_token_barrier_outcome(
    outcome: token_barrier.TokenBarrierOutcome,
    *,
    deployment: Mapping[str, Any],
    phase2_observed_at_unix_seconds: int,
) -> dict[str, Any]:
    if not isinstance(outcome, token_barrier.TokenBarrierOutcome):
        raise ValueError("validated token barrier outcome is required")
    receipt = _json_copy(outcome.receipt)
    _exact(
        receipt,
        {
            "schema",
            "status",
            "controller_service_account",
            "run_scoped_controller_required",
            "binding_purpose",
            "binding_role",
            "binding_member",
            "binding_condition_sha256",
            "binding_add_attempts",
            "binding_add_outcome",
            "binding_add_readback_sha256",
            "token_attempt_count",
            "failed_token_attempt_count",
            "token_attempts",
            "started_at_unix_seconds",
            "finished_at_unix_seconds",
            "elapsed_seconds",
            "maximum_propagation_seconds",
            "token_lifetime_seconds",
            "token_expire_time",
            "fixed_nonrefreshing_controller_credential",
            "controller_token_remint_forbidden",
            "token_creator_revoke_attempts",
            "token_creator_revoke_changed",
            "token_creator_revoke_readback_sha256",
            "token_creator_revoke_zero_readback",
            "token_creator_revoke_zero_readback_evidence_sha256",
            "token_creator_live_after_barrier",
            "phase2_started",
            "phase2_must_exclude_token_creator",
            "run_scoped_controller_delete_after_pair_claim_required",
            "vm_insert_attempt_count",
            "access_token_stored",
            "authorization_header_stored",
            "response_body_stored",
            "cloud_mutation_performed",
            "current_profile_changed",
            "receipt_sha256",
        },
        "token barrier success receipt",
    )
    supplied_sha = _sha(
        receipt.pop("receipt_sha256", None),
        "token barrier success receipt",
    )
    if token_barrier.canonical_sha256(receipt) != supplied_sha:
        raise ValueError("token barrier success receipt digest changed")
    attempts = receipt["token_attempts"]
    if not isinstance(attempts, list):
        raise ValueError("token barrier attempts changed")
    attempt_fields = {
        "attempt",
        "started_at_unix_seconds",
        "finished_at_unix_seconds",
        "elapsed_seconds",
        "http_status",
        "internal_error_code",
        "google_error_code",
        "google_error_status",
        "google_error_reason",
        "response_body_sha256",
        "response_body_stored",
        "authorization_header_stored",
        "access_token_stored",
    }
    for position, row in enumerate(attempts, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("token barrier attempt changed")
        _exact(row, attempt_fields, "token barrier attempt")
        if (
            row["attempt"] != position
            or type(row["started_at_unix_seconds"]) is not int
            or type(row["finished_at_unix_seconds"]) is not int
            or row["finished_at_unix_seconds"]
            < row["started_at_unix_seconds"]
            or not isinstance(row["elapsed_seconds"], (int, float))
            or isinstance(row["elapsed_seconds"], bool)
            or not 0 <= row["elapsed_seconds"]
            <= token_barrier.MAX_PROPAGATION_SECONDS
            or row["http_status"] != 403
            or row["internal_error_code"]
            != "controller_access_token_generation_failed"
            or row["response_body_stored"] is not False
            or row["authorization_header_stored"] is not False
            or row["access_token_stored"] is not False
        ):
            raise ValueError("token barrier attempt semantics changed")
        _sha(row["response_body_sha256"], "token error response body")
    started = _integer(
        receipt["started_at_unix_seconds"],
        "token barrier start",
        minimum=1,
    )
    finished = _integer(
        receipt["finished_at_unix_seconds"],
        "token barrier finish",
        minimum=started,
        maximum=phase2_observed_at_unix_seconds,
    )
    controller = deployment["controller_service_account"]
    if (
        receipt["schema"] != token_barrier.SCHEMA
        or receipt["status"] != token_barrier.SUCCESS_STATUS
        or receipt["controller_service_account"] != controller["email"]
        or receipt["run_scoped_controller_required"] is not True
        or receipt["binding_purpose"]
        != token_barrier.TOKEN_CREATOR_PURPOSE
        or receipt["binding_role"] != token_barrier.TOKEN_CREATOR_ROLE
        or receipt["binding_member"]
        != token_barrier.DEFAULT_INITIATING_PRINCIPAL
        or type(receipt["binding_add_attempts"]) is not int
        or not 1 <= receipt["binding_add_attempts"] <= 8
        or receipt["binding_add_outcome"] != "confirmed_changed"
        or type(receipt["token_attempt_count"]) is not int
        or not 1 <= receipt["token_attempt_count"] <= 64
        or receipt["failed_token_attempt_count"] != len(attempts)
        or receipt["token_attempt_count"] != len(attempts) + 1
        or not isinstance(receipt["elapsed_seconds"], (int, float))
        or isinstance(receipt["elapsed_seconds"], bool)
        or not 0 <= receipt["elapsed_seconds"]
        <= token_barrier.MAX_PROPAGATION_SECONDS
        or finished < started
        or receipt["maximum_propagation_seconds"]
        != token_barrier.MAX_PROPAGATION_SECONDS
        or receipt["token_lifetime_seconds"]
        != token_barrier.TOKEN_LIFETIME_SECONDS
        or not isinstance(receipt["token_expire_time"], str)
        or not receipt["token_expire_time"]
        or receipt["fixed_nonrefreshing_controller_credential"] is not True
        or receipt["controller_token_remint_forbidden"] is not True
        or type(receipt["token_creator_revoke_attempts"]) is not int
        or not 1 <= receipt["token_creator_revoke_attempts"] <= 8
        or receipt["token_creator_revoke_changed"] is not True
        or receipt["token_creator_live_after_barrier"] is not False
        or receipt["phase2_started"] is not False
        or receipt["phase2_must_exclude_token_creator"] is not True
        or receipt[
            "run_scoped_controller_delete_after_pair_claim_required"
        ]
        is not True
        or receipt["vm_insert_attempt_count"] != 0
        or receipt["access_token_stored"] is not False
        or receipt["authorization_header_stored"] is not False
        or receipt["response_body_stored"] is not False
        or receipt["cloud_mutation_performed"] is not True
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("token barrier success semantics changed")
    for field in (
        "binding_condition_sha256",
        "binding_add_readback_sha256",
        "token_creator_revoke_readback_sha256",
    ):
        _sha(receipt[field], field)
    if (
        receipt["binding_add_readback_sha256"]
        == receipt["token_creator_revoke_readback_sha256"]
    ):
        raise ValueError("token barrier add/revoke readbacks collided")
    return {**receipt, "receipt_sha256": supplied_sha}


def _bootstrap_source_external_binding(
    *,
    deployment: Mapping[str, Any],
    plan: Mapping[str, Any],
    provision: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    role_records = [
        {
            "external_job_id": row["external_job_id"],
            "inner_job_id": row["inner_job_id"],
            "source_role": row["source_role"],
            "object_count": row["object_count"],
            "objects": copy.deepcopy(row["objects"]),
            "role_manifest_sha256": row["role_manifest_sha256"],
            "objects_sha256": row["objects_sha256"],
        }
        for row in manifests
    ]
    body = {
        "schema": external_auth.BOOTSTRAP_SOURCE_BINDING_SCHEMA,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "bootstrap_source_content_binding_sha256": plan[
            "bootstrap_source_content_binding_sha256"
        ],
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_provision_receipt_sha256": provision["receipt_sha256"],
        "source_prefix": provision["source_prefix"],
        "source_generations": copy.deepcopy(
            provision["source_generations"]
        ),
        "source_generations_sha256": provision[
            "source_generations_sha256"
        ],
        "source_object_count": provision["object_count"],
        "role_manifests": role_records,
        "role_manifests_sha256": canonical_sha256(role_records),
    }
    return external_auth._validate_bootstrap_source_binding(
        body,
        deployment=deployment,
    )


def _validated_bootstrap_source(
    *,
    bootstrap_source_plan: Mapping[str, Any],
    validated_bootstrap_source_provision: (
        bootstrap_source.ValidatedBootstrapSourceProvision
    ),
    role_bootstrap_manifests: Mapping[str, Mapping[str, Any]],
    deployment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    source_bytes: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = bootstrap_source.validate_bootstrap_source_plan(
        bootstrap_source_plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    capability = (
        bootstrap_source.require_validated_bootstrap_source_provision(
            validated_bootstrap_source_provision,
            deployment_contract_sha256=deployment[
                "deployment_contract_sha256"
            ],
            source_plan_sha256=plan["source_plan_sha256"],
        )
    )
    provision = capability.receipt()
    if (
        provision["source_prefix"] != plan["source_prefix"]
        or provision["object_count"] != bootstrap_source.SOURCE_OBJECT_COUNT
        or provision["prefix_empty_before_upload"] is not True
        or provision["if_generation_match"]
        != bootstrap_source.IF_GENERATION_MATCH
        or provision["all_objects_created_once"] is not True
        or provision["all_generation_bound"] is not True
        or provision["all_bytes_and_sha256_read_back"] is not True
        or provision["old_package_write_count"] != 0
        or provision["old_result_write_count"] != 0
        or provision["direct_v2_result_write_count"] != 0
        or provision["cloud_mutation_performed"] is not True
        or provision["current_profile_changed"] is not False
    ):
        raise ValueError("bootstrap source provision semantics changed")
    expected_jobs = deployment["selected_job_ids"]
    if (
        not isinstance(role_bootstrap_manifests, Mapping)
        or set(role_bootstrap_manifests) != set(expected_jobs)
    ):
        raise ValueError("role bootstrap manifest set changed")
    manifests = [
        bootstrap_source.validate_role_bootstrap_manifest(
            role_bootstrap_manifests[job_id],
            source_plan=plan,
            validated_provision=capability,
            deployment_contract=deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=job_id,
        )
        for job_id in expected_jobs
    ]
    if (
        [row["source_role"] for row in manifests]
        != list(deployment["source_roles"])
        or len(
            {
                row["role_manifest_sha256"] for row in manifests
            }
        )
        != deployment_v2.VM_COUNT
        or any(
            row["source_provision_receipt_sha256"]
            != capability.provision_receipt_sha256
            for row in manifests
        )
    ):
        raise ValueError("role bootstrap manifest identity changed")

    objects = plan["objects"]
    runtime_bundle = bootstrap_source.validate_runtime_source_bundle(
        objects[0]["content"]
    )
    if set(BOOTSTRAP_RUNTIME_SOURCE_LOGICAL_NAMES) != set(
        bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    ):
        raise AssertionError("bootstrap runtime source map drifted")
    records = {
        row["path"]: row for row in runtime_bundle["records"]
    }
    for path, logical_name in (
        BOOTSTRAP_RUNTIME_SOURCE_LOGICAL_NAMES.items()
    ):
        supplied = source_bytes.get(logical_name)
        row = records.get(path)
        if (
            not isinstance(supplied, bytes)
            or not isinstance(row, Mapping)
            or row["source"].encode("utf-8") != supplied
            or row["bytes"] != len(supplied)
            or row["sha256"] != hashlib.sha256(supplied).hexdigest()
        ):
            raise ValueError(
                f"provisioned runtime source bytes changed: {path}"
            )
    return plan, provision, manifests


def build_external_preflight_gate_receipt(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    package_provision_receipt: Mapping[str, Any],
    direct_v2_prefix_empty_receipt: Mapping[str, Any],
    compute_absence_receipt: Mapping[str, Any],
    live_readback_receipt: Mapping[str, Any],
    phase2_iam_plan: Mapping[str, Any],
    phase2_iam_readback_receipt: Mapping[str, Any],
    controller_service_account_create_receipt: Mapping[str, Any],
    token_barrier_outcome: token_barrier.TokenBarrierOutcome,
    bootstrap_source_plan: Mapping[str, Any],
    validated_bootstrap_source_provision: (
        bootstrap_source.ValidatedBootstrapSourceProvision
    ),
    role_bootstrap_manifests: Mapping[str, Mapping[str, Any]],
    source_bytes: Mapping[str, bytes],
    current_profile_bytes: bytes,
    now_unix_seconds: int,
) -> dict[str, Any]:
    """Validate every concrete body and emit a persistable audit receipt."""

    now = _integer(
        now_unix_seconds, "external preflight evaluation time", minimum=1
    )
    deployment, candidate, reference = _validated_deployment(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    plan = phase2_iam.validate_step12b_phase2_iam_plan(
        phase2_iam_plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    window = plan["authorization_window"]
    if not (
        window["issued_at_unix_seconds"]
        <= now
        < window["expires_at_unix_seconds"]
    ):
        raise ValueError("Phase2 authorization window is not current")
    phase2_readback = (
        phase2_iam.validate_step12b_phase2_iam_readback_receipt(
            phase2_iam_readback_receipt,
            plan=plan,
            deployment_contract=deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
        )
    )
    controller_create = pair_release.validate_controller_create_receipt(
        controller_service_account_create_receipt,
        deployment_contract=deployment,
    )
    token_receipt = _validate_token_barrier_outcome(
        token_barrier_outcome,
        deployment=deployment,
        phase2_observed_at_unix_seconds=phase2_readback[
            "observed_at_unix_seconds"
        ],
    )
    if (
        phase2_readback["controller_binding_count"]
        + phase2_readback["worker_binding_count"]
        != 8
        or phase2_readback[
            "all_phase2_bindings_present_exactly_once"
        ]
        is not True
        or phase2_readback["cloud_mutation_performed_by_validator"]
        is not False
    ):
        raise ValueError("Phase2 IAM exact eight-binding readback changed")
    package = validate_immutable_package_provision_receipt(
        package_provision_receipt,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
    )
    prefix = validate_direct_v2_prefix_empty_receipt(
        direct_v2_prefix_empty_receipt,
        deployment_contract=deployment,
    )
    compute = validate_compute_absence_receipt(
        compute_absence_receipt,
        deployment_contract=deployment,
    )
    live = validate_live_readback_receipt(
        live_readback_receipt,
        deployment_contract=deployment,
        phase2_iam_plan=plan,
    )
    for label, observed in (
        ("direct-v2 prefix", prefix["observed_at_unix_seconds"]),
        ("compute absence", compute["observed_at_unix_seconds"]),
        ("live readback", live["observed_at_unix_seconds"]),
        ("Phase2 IAM readback", phase2_readback["observed_at_unix_seconds"]),
    ):
        _fresh(observed, now_unix_seconds=now, label=label)
    phase2_observed = phase2_readback["observed_at_unix_seconds"]
    if any(
        observed < phase2_observed
        for observed in (
            prefix["observed_at_unix_seconds"],
            compute["observed_at_unix_seconds"],
            live["observed_at_unix_seconds"],
        )
    ):
        raise ValueError(
            "final read-only observations preceded Phase2 IAM readback"
        )
    source_identity = _source_identity(source_bytes)
    (
        checked_source_plan,
        source_provision,
        checked_role_manifests,
    ) = _validated_bootstrap_source(
        bootstrap_source_plan=bootstrap_source_plan,
        validated_bootstrap_source_provision=(
            validated_bootstrap_source_provision
        ),
        role_bootstrap_manifests=role_bootstrap_manifests,
        deployment=deployment,
        candidate=candidate,
        reference=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        source_bytes=source_bytes,
    )
    source_external_binding = _bootstrap_source_external_binding(
        deployment=deployment,
        plan=checked_source_plan,
        provision=source_provision,
        manifests=checked_role_manifests,
    )
    profile_identity = _current_profile_identity(
        current_profile_bytes, deployment=deployment
    )
    evidence = {
        "immutable_package_provision_receipt_sha256": package[
            "receipt_sha256"
        ],
        "direct_v2_prefix_empty_receipt_sha256": prefix["receipt_sha256"],
        "compute_absence_receipt_sha256": compute["receipt_sha256"],
        "live_readback_receipt_sha256": live["receipt_sha256"],
        "phase2_iam_plan_sha256": plan["plan_sha256"],
        "phase2_iam_readback_receipt_sha256": phase2_readback[
            "receipt_sha256"
        ],
        "controller_service_account_create_receipt_sha256": (
            controller_create["receipt_sha256"]
        ),
        "token_barrier_receipt_sha256": token_receipt["receipt_sha256"],
        "bootstrap_source_plan_sha256": checked_source_plan[
            "source_plan_sha256"
        ],
        "bootstrap_source_provision_receipt_sha256": source_provision[
            "receipt_sha256"
        ],
        "bootstrap_source_generations_sha256": source_provision[
            "source_generations_sha256"
        ],
        "role_bootstrap_manifest_sha256s": [
            row["role_manifest_sha256"]
            for row in checked_role_manifests
        ],
    }
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "evaluated_at_unix_seconds": now,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "run_nonce": deployment["run_nonce"],
        "run_identity_sha256": deployment["run_identity_sha256"],
        "direct_stage_identity_sha256": deployment[
            "direct_stage_identity_sha256"
        ],
        "controller_service_account": copy.deepcopy(
            deployment["controller_service_account"]
        ),
        "source_identity": source_identity,
        "source_identity_sha256": canonical_sha256(source_identity),
        "current_profile_identity": profile_identity,
        "bootstrap_source_binding": source_external_binding,
        "evidence_receipts": evidence,
        "evidence_receipts_sha256": canonical_sha256(evidence),
        "checks": {
            "deployment_fully_validated": True,
            "authorization_source_bytes_validated": True,
            "current_profile_bytes_and_hash_validated": True,
            "immutable_package_exact_16_generations_sha_bytes_validated": True,
            "direct_v2_prefix_empty_validated": True,
            "exact_two_vm_disk_get404_validated": True,
            "operation_name_history0_validated": True,
            "capacity_nat_services_custom_roles_validated": True,
            "phase2_iam_plan_and_readback_validated": True,
            "run_scoped_controller_create_receipt_validated": True,
            "token_creator_barrier_and_revoke_validated": True,
            "bootstrap_source_prefix_empty_before_upload_validated": True,
            "bootstrap_source_generation_bytes_sha_readback_validated": True,
            "role_bootstrap_manifests_validated": True,
            "all_source_bytes_and_hashes_validated": True,
            "all_observations_fresh": True,
        },
        "passed": True,
        "signed_authorization_still_required": True,
        "launch_authorized": False,
        "upstream_cloud_mutations": {
            "bootstrap_source_provision_performed": True,
            "run_scoped_controller_service_account_created": True,
            "token_creator_add_and_revoke_performed": True,
            "phase2_iam_bindings_present": True,
            "all_upstream_mutations_explicitly_bound": True,
        },
        "cloud_mutation_performed_before_gate": True,
        "cloud_api_call_performed_by_gate": False,
        "cloud_mutation_performed_by_gate": False,
        "current_profile_changed": False,
    }
    return {
        **body,
        "gate_receipt_sha256": canonical_sha256(body),
    }


def validate_external_preflight_gate_receipt(
    value: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Rebuild the gate receipt from the same concrete evidence bodies."""

    checked = _json_copy(value)
    supplied = _sha(
        checked.pop("gate_receipt_sha256", None),
        "external preflight gate receipt",
    )
    if canonical_sha256(checked) != supplied:
        raise ValueError("external preflight gate receipt digest changed")
    rebuilt = build_external_preflight_gate_receipt(**kwargs)
    if dict(value) != rebuilt:
        raise ValueError("external preflight gate receipt changed")
    return rebuilt


def mint_validated_external_preflight(
    **kwargs: Any,
) -> external_auth.ValidatedExternalPreflight:
    """Mint the auth module's opaque capability after the real gate passes."""

    gate_receipt = build_external_preflight_gate_receipt(**kwargs)
    deployment = dict(kwargs["deployment_contract"])
    source_identity = gate_receipt["source_identity"]
    evidence = gate_receipt["evidence_receipts"]
    auth_receipt = external_auth._build_external_preflight_receipt_record(
        deployment_contract=deployment,
        readonly_preflight_receipt_sha256=gate_receipt[
            "gate_receipt_sha256"
        ],
        iam_capacity_gate_receipt_sha256=evidence[
            "live_readback_receipt_sha256"
        ],
        package_provision_receipt_sha256=evidence[
            "immutable_package_provision_receipt_sha256"
        ],
        deployment_source_sha256=_source_sha(
            source_identity, "deployment"
        ),
        alias_bridge_source_sha256=_source_sha(
            source_identity, "alias_bridge"
        ),
        prebootstrap_source_sha256=_source_sha(
            source_identity, "prebootstrap"
        ),
        startup_source_sha256=startup_loader.startup_loader_sha256(),
        controller_source_sha256=_source_sha(
            source_identity, "controller_helper"
        ),
        bootstrap_source_binding=gate_receipt[
            "bootstrap_source_binding"
        ],
        upstream_cloud_mutation_evidence_bound=True,
    )
    return external_auth._mint_validated_external_preflight_after_gate(
        auth_receipt,
        deployment_contract=deployment,
        gate_validation_seal=external_auth._VALIDATED_PREFLIGHT_SEAL,
    )


__all__ = [
    "COMPUTE_ABSENCE_SCHEMA",
    "DIRECT_PREFIX_EMPTY_SCHEMA",
    "EXPECTED_PACKAGE_OBJECT_COUNT",
    "EXPECTED_PACKAGE_PROVISION_RECEIPT_SHA256",
    "LIVE_READBACK_SCHEMA",
    "NAT_IP_ALLOCATE_OPTION",
    "NAT_NAME",
    "NAT_ROUTER_NAME",
    "NAT_SOURCE_SUBNETWORK_IP_RANGES",
    "OBSERVATION_MAX_AGE_SECONDS",
    "PROJECT_NUMBER",
    "REQUIRED_ENABLED_SERVICES",
    "SCHEMA",
    "SOURCE_PATHS",
    "STATUS",
    "build_compute_absence_receipt",
    "build_direct_v2_prefix_empty_receipt",
    "build_external_preflight_gate_receipt",
    "build_live_readback_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "load_expected_source_bytes",
    "mint_validated_external_preflight",
    "validate_compute_absence_receipt",
    "validate_direct_v2_prefix_empty_receipt",
    "validate_external_preflight_gate_receipt",
    "validate_immutable_package_provision_receipt",
    "validate_live_readback_receipt",
]
