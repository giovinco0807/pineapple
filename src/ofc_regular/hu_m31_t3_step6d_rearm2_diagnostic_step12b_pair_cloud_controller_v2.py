"""Exact attempt-0 Step12b pair controller core.

This module is deliberately client-neutral.  It owns the execution ordering
and exact request bodies, while callers inject authenticated Compute clients,
the already-created controller-SA admin, Phase-2 IAM teardown, and failure
cleanup implementations.  Importing it performs no cloud operation.

The fixed controller credential is usable only through ``controller_client``
and is never read or serialized here.  After both claim CAS readbacks, the
controller IAM bindings and run-scoped controller service account are removed
before a signed pair release is written with the user Compute client.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter
    as legacy_vm,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as step11_launch,
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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_lifecycle_v2
    as phase2_lifecycle,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2
    as vm_metadata,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_exact_pair_cloud_controller_v2"
)
FINAL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_exact_pair_cloud_lifecycle_receipt_v2"
)
FAILURE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_exact_pair_cloud_failure_receipt_v2"
)

PROJECT = payload_transport.PROJECT
PROJECT_NUMBER = external_auth.EXPECTED_PROJECT_NUMBER
ZONE = payload_transport.ZONE
MACHINE_TYPE = deployment_v2.ACTUAL_MACHINE_TYPE
WORKER_SERVICE_ACCOUNT = payload_transport.WORKER_SERVICE_ACCOUNT
OAUTH_SCOPE = payload_transport.REQUIRED_WORKER_OAUTH_SCOPE
VM_COUNT = deployment_v2.VM_COUNT
ATTEMPT_INDEX = deployment_v2.ATTEMPT_INDEX
REQUEST_ID_COUNT = 6
ROLE_BOOTSTRAP_MANIFEST_KEY = getattr(
    vm_metadata,
    "ROLE_BOOTSTRAP_MANIFEST_KEY",
    "ofc-step12b-role-bootstrap-manifest",
)
LEGACY_ROLE_PAYLOAD_CONTRACT_KEY = getattr(
    vm_metadata,
    "ROLE_PAYLOAD_CONTRACT_KEY",
    "ofc-step12b-role-payload-contract",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL = re.compile(r"^[1-9][0-9]{5,31}$")
_FINGERPRINT = re.compile(r"^[A-Za-z0-9_+/=-]{8,256}$")
_RECEIPT_FORBIDDEN_KEYS = {
    "access_token",
    "authorization_header",
    "private_key",
    "raw_response_body",
}


@dataclass(frozen=True)
class ComputeMutationResult:
    status_code: int
    target_name: str
    request_id: str
    operation_id: str | None
    operation_done: bool


@dataclass(frozen=True)
class PreparedRoleLaunch:
    external_job_id: str
    authorization: Mapping[str, Any]
    package_generations: Mapping[str, int]
    external_preflight_receipt: Mapping[str, Any]
    initial_metadata_values: Mapping[str, str]
    startup_script_bytes: bytes
    bootstrap_role_manifest: Mapping[str, Any]
    claim_nonce: str


class ControllerComputeClient(Protocol):
    project: str
    zone: str
    principal: str
    credential_kind: str

    def insert_instance(
        self,
        *,
        instance_name: str,
        request_id: str,
        body: Mapping[str, Any],
    ) -> ComputeMutationResult: ...

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None:
        ...

    def set_metadata(
        self,
        *,
        instance_name: str,
        request_id: str,
        expected_fingerprint: str,
        values: Mapping[str, str],
    ) -> ComputeMutationResult: ...


class UserComputeClient(Protocol):
    project: str
    zone: str
    principal: str
    credential_kind: str

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None:
        ...

    def set_metadata(
        self,
        *,
        instance_name: str,
        request_id: str,
        expected_fingerprint: str,
        values: Mapping[str, str],
    ) -> ComputeMutationResult: ...

    def cleanup_exact_instances_and_disks(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]: ...


class Phase2TeardownAdapter(Protocol):
    def remove_controller_bindings_after_claims(
        self,
        *,
        installed: phase2_lifecycle.Phase2IamInstalledCapability,
        claim_cas_readback_receipts: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def cleanup_phase2_iam_on_failure(
        self, *, failure_evidence_sha256: str
    ) -> Mapping[str, Any]: ...


class ControllerServiceAccountAdmin(Protocol):
    def delete_created_and_wait_absent(
        self, *, created: Any
    ) -> Mapping[str, Any]: ...

    def cleanup_delete_if_present(self) -> Mapping[str, Any]: ...


class MetadataContractAdapter(Protocol):
    def validate_initial(
        self,
        *,
        values: Mapping[str, str],
        startup_script_bytes: bytes,
        role_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def add_claim(
        self,
        *,
        initial_values: Mapping[str, str],
        claim_value: str,
    ) -> tuple[Mapping[str, str], Mapping[str, Any]]: ...

    def add_release(
        self,
        *,
        initial_values: Mapping[str, str],
        claim_value: str,
        release_value: str,
    ) -> tuple[Mapping[str, str], Mapping[str, Any]]: ...


class VmMetadataContractAdapter:
    """Thin adapter over the pure metadata contract."""

    def validate_initial(
        self,
        *,
        values: Mapping[str, str],
        startup_script_bytes: bytes,
        role_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        checked = dict(values)
        startup = startup_script_bytes.decode("utf-8")
        if (
            checked.get(vm_metadata.STARTUP_KEY) != startup
            or checked.get(ROLE_BOOTSTRAP_MANIFEST_KEY)
            != bootstrap_source.canonical_bytes(role_manifest).decode(
                "ascii"
            )
        ):
            raise ValueError("prepared role metadata changed")
        return vm_metadata.validate_initial_metadata_budget(checked)

    def add_claim(
        self,
        *,
        initial_values: Mapping[str, str],
        claim_value: str,
    ) -> tuple[Mapping[str, str], Mapping[str, Any]]:
        receipt = vm_metadata.validate_postclaim_metadata_budget(
            initial_values=initial_values,
            claim_value=claim_value,
        )
        return (
            {
                **dict(initial_values),
                vm_metadata.POSTCREATE_CLAIM_KEY: claim_value,
            },
            receipt,
        )

    def add_release(
        self,
        *,
        initial_values: Mapping[str, str],
        claim_value: str,
        release_value: str,
    ) -> tuple[Mapping[str, str], Mapping[str, Any]]:
        receipt = vm_metadata.validate_postrelease_metadata_budget(
            initial_values=initial_values,
            claim_value=claim_value,
            release_value=release_value,
        )
        return (
            {
                **dict(initial_values),
                vm_metadata.POSTCREATE_CLAIM_KEY: claim_value,
                vm_metadata.PAIR_RELEASE_KEY: release_value,
            },
            receipt,
        )


class ExactPairCloudControllerError(RuntimeError):
    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__("step12b_exact_pair_cloud_controller_failed")
        self.receipt = copy.deepcopy(dict(receipt))

    def __str__(self) -> str:
        return "step12b_exact_pair_cloud_controller_failed"


class _Abort(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return deployment_v2.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return deployment_v2.canonical_sha256(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    _reject_secret_fields(body)
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _reject_secret_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("receipt key changed")
            if key.lower() in _RECEIPT_FORBIDDEN_KEYS:
                raise ValueError(f"secret field forbidden at {path}.{key}")
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{position}]")


def _request_ids(values: Sequence[str]) -> list[str]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != REQUEST_ID_COUNT
    ):
        raise ValueError("exactly six request IDs are required")
    checked = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("request ID changed")
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError):
            raise ValueError("request ID is not canonical UUIDv4") from None
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("request ID is not canonical UUIDv4")
        checked.append(value)
    if len(set(checked)) != REQUEST_ID_COUNT:
        raise ValueError("request IDs collided")
    return checked


def _metadata_map(items: Any) -> dict[str, str]:
    if not isinstance(items, list):
        raise ValueError("provider metadata items changed")
    result: dict[str, str] = {}
    for item in items:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"key", "value"}
            or not isinstance(item["key"], str)
            or not isinstance(item["value"], str)
            or item["key"] in result
        ):
            raise ValueError("provider metadata item changed")
        result[item["key"]] = item["value"]
    return result


def _metadata_items(values: Mapping[str, str]) -> list[dict[str, str]]:
    if not isinstance(values, Mapping):
        raise ValueError("initial metadata changed")
    checked = dict(values)
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or not value
        for key, value in checked.items()
    ):
        raise ValueError("initial metadata value changed")
    return [
        {"key": key, "value": checked[key]} for key in sorted(checked)
    ]


def build_exact_insert_body(
    *,
    instance_name: str,
    initial_metadata_values: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(instance_name, str):
        raise ValueError("instance name changed")
    return {
        "name": instance_name,
        "machineType": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{PROJECT}/zones/{ZONE}/machineTypes/{MACHINE_TYPE}"
        ),
        "canIpForward": False,
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "type": "PERSISTENT",
                "initializeParams": {
                    "diskName": instance_name,
                    "sourceImage": payload_transport.IMAGE_SELF_LINK,
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": step11_launch.NETWORK_SELF_LINK,
                "subnetwork": step11_launch.SUBNETWORK_SELF_LINK,
                "accessConfigs": [],
            }
        ],
        "reservationAffinity": {"consumeReservationType": "NO_RESERVATION"},
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "maxRunDuration": {
                "seconds": str(legacy_vm.MAX_RUNTIME_SECONDS_PER_VM),
                "nanos": 0,
            },
        },
        "serviceAccounts": [
            {
                "email": WORKER_SERVICE_ACCOUNT,
                "scopes": [OAUTH_SCOPE],
            }
        ],
        "metadata": {"items": _metadata_items(initial_metadata_values)},
    }


def _validate_clients(
    controller_client: ControllerComputeClient,
    user_client: UserComputeClient,
    deployment: Mapping[str, Any],
) -> None:
    controller_principal = deployment["controller_service_account"][
        "principal"
    ]
    if (
        getattr(controller_client, "project", None) != PROJECT
        or getattr(controller_client, "zone", None) != ZONE
        or getattr(controller_client, "principal", None)
        != controller_principal
        or getattr(controller_client, "credential_kind", None)
        != "fixed_nonrefreshing_controller"
        or getattr(user_client, "project", None) != PROJECT
        or getattr(user_client, "zone", None) != ZONE
        or not isinstance(getattr(user_client, "principal", None), str)
        or getattr(user_client, "principal", None) == controller_principal
        or getattr(user_client, "credential_kind", None) != "user"
    ):
        raise ValueError("controller/user Compute client identity changed")


def _mutation(
    result: ComputeMutationResult,
    *,
    expected_name: str,
    expected_request_id: str,
    operation: str,
) -> dict[str, Any]:
    if not isinstance(result, ComputeMutationResult):
        raise _Abort(f"{operation}_result_changed")
    if (
        result.target_name != expected_name
        or result.request_id != expected_request_id
    ):
        raise _Abort(f"{operation}_identity_changed")
    if result.status_code == 412:
        raise _Abort(f"{operation}_cas_412")
    if result.status_code not in {200, 202}:
        raise _Abort(f"{operation}_non2xx")
    if (
        result.operation_done is not True
        or not isinstance(result.operation_id, str)
        or not result.operation_id
    ):
        raise _Abort(f"{operation}_operation_not_done")
    return {
        "operation": operation,
        "target_name": expected_name,
        "request_id": expected_request_id,
        "status_code": result.status_code,
        "operation_id": result.operation_id,
        "operation_done": True,
    }


def _provider_identity(
    value: Mapping[str, Any] | None,
    *,
    expected_name: str,
    expected_metadata: Mapping[str, str],
    prior_instance_id: str | None = None,
    prior_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _Abort("provider_instance_missing")
    provider = copy.deepcopy(dict(value))
    instance_id = provider.get("id")
    fingerprint = (
        provider.get("metadata", {}).get("fingerprint")
        if isinstance(provider.get("metadata"), Mapping)
        else None
    )
    self_link = provider.get("selfLink")
    zone = provider.get("zone")
    machine = provider.get("machineType")
    accounts = provider.get("serviceAccounts")
    interfaces = provider.get("networkInterfaces")
    scheduling = provider.get("scheduling")
    disks = provider.get("disks")
    if (
        not isinstance(instance_id, str)
        or _DECIMAL.fullmatch(instance_id) is None
        or (
            prior_instance_id is not None
            and instance_id != prior_instance_id
        )
        or not isinstance(fingerprint, str)
        or _FINGERPRINT.fullmatch(fingerprint) is None
        or (
            prior_fingerprint is not None
            and fingerprint == prior_fingerprint
        )
        or provider.get("name") != expected_name
        or self_link
        != (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{PROJECT}/zones/{ZONE}/instances/{expected_name}"
        )
        or not isinstance(zone, str)
        or zone.rsplit("/", 1)[-1] != ZONE
        or not isinstance(machine, str)
        or machine.rsplit("/", 1)[-1] != MACHINE_TYPE
        or accounts
        != [
            {
                "email": WORKER_SERVICE_ACCOUNT,
                "scopes": [OAUTH_SCOPE],
            }
        ]
        or not isinstance(interfaces, list)
        or len(interfaces) != 1
        or interfaces[0].get("accessConfigs", []) != []
        or not isinstance(scheduling, Mapping)
        or scheduling.get("provisioningModel") != "SPOT"
        or not isinstance(disks, list)
        or len(disks) != 1
        or disks[0].get("boot") is not True
        or disks[0].get("autoDelete") is not True
        or not isinstance(disks[0].get("source"), str)
        or disks[0]["source"].rsplit("/", 1)[-1] != expected_name
        or not isinstance(provider.get("metadata"), Mapping)
        or _metadata_map(provider["metadata"].get("items"))
        != dict(expected_metadata)
    ):
        raise _Abort("provider_identity_or_metadata_changed")
    return {
        "instance_name": expected_name,
        "provider_instance_id": instance_id,
        "metadata_fingerprint": fingerprint,
        "machine_type": MACHINE_TYPE,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "oauth_scopes": [OAUTH_SCOPE],
        "external_ip_count": 0,
        "spot": True,
        "provider_record_sha256": canonical_sha256(provider),
    }


def _validate_installed_capability(
    installed: phase2_lifecycle.Phase2IamInstalledCapability,
    *,
    deployment: Mapping[str, Any],
    plan: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, Any]:
    if (
        not isinstance(
            installed, phase2_lifecycle.Phase2IamInstalledCapability
        )
        or installed.deployment_contract_sha256
        != deployment["deployment_contract_sha256"]
        or installed.phase2_iam_plan_sha256 != plan["plan_sha256"]
    ):
        raise ValueError("Phase2 installed capability changed")
    readback = (
        phase2_lifecycle.get_step12b_phase2_exact_readback_receipt(
            installed,
            phase2_iam_plan=plan,
            deployment_contract=deployment,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
        )
    )
    if (
        readback["receipt_sha256"]
        != installed.exact_readback_receipt_sha256
    ):
        raise ValueError("Phase2 installed readback changed")
    return readback


def _validate_created_capability(
    created: Any,
    *,
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = getattr(created, "receipt", None)
    if not isinstance(receipt, Mapping):
        raise ValueError("controller-SA created capability is missing")
    return pair_release.validate_controller_create_receipt(
        receipt, deployment_contract=deployment
    )


def _validate_roles(
    role_launches: Sequence[PreparedRoleLaunch],
    *,
    deployment: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    source_plan: Mapping[str, Any],
    source_provision: bootstrap_source.ValidatedBootstrapSourceProvision,
    verifier: external_auth.ControllerVerifier,
    metadata_adapter: MetadataContractAdapter,
    now_unix_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        isinstance(role_launches, (str, bytes))
        or not isinstance(role_launches, Sequence)
        or len(role_launches) != VM_COUNT
        or any(not isinstance(row, PreparedRoleLaunch) for row in role_launches)
    ):
        raise ValueError("exactly two prepared role launches are required")
    by_job = {row.external_job_id: row for row in role_launches}
    selected = list(deployment["selected_job_ids"])
    if len(by_job) != VM_COUNT or set(by_job) != set(selected):
        raise ValueError("prepared role launch mapping changed")
    ordered = [by_job[job_id] for job_id in selected]
    checked_rows = []
    insert_bodies = []
    for position, row in enumerate(ordered):
        job_id = selected[position]
        instance = deployment["instances"][position]
        authorization = external_auth.validate_external_authorization(
            row.authorization,
            deployment_contract=deployment,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=job_id,
            package_generations=row.package_generations,
            external_preflight_receipt=row.external_preflight_receipt,
            verifier=verifier,
            now_unix_seconds=now_unix_seconds,
        )
        manifest = bootstrap_source.validate_role_bootstrap_manifest(
            row.bootstrap_role_manifest,
            source_plan=source_plan,
            validated_provision=source_provision,
            deployment_contract=deployment,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=job_id,
        )
        values = copy.deepcopy(dict(row.initial_metadata_values))
        if (
            values.get(vm_metadata.EXTERNAL_AUTHORIZATION_KEY)
            != external_auth.canonical_bytes(authorization).decode("ascii")
            or values.get(vm_metadata.RUN_NONCE_KEY) != run_nonce
            or values.get(vm_metadata.EXTERNAL_JOB_ID_KEY) != job_id
            or values.get(vm_metadata.SOURCE_ROLE_KEY)
            != instance["source_role"]
            or values.get(ROLE_BOOTSTRAP_MANIFEST_KEY)
            != bootstrap_source.canonical_bytes(manifest).decode("ascii")
            or LEGACY_ROLE_PAYLOAD_CONTRACT_KEY in values
        ):
            # The role payload must be generation-pinned in the source
            # manifest, never embedded in metadata.
            raise ValueError("prepared role metadata identity changed")
        metadata_receipt = metadata_adapter.validate_initial(
            values=values,
            startup_script_bytes=row.startup_script_bytes,
            role_manifest=manifest,
        )
        insert_body = build_exact_insert_body(
            instance_name=instance["instance_name"],
            initial_metadata_values=values,
        )
        checked_rows.append(
            {
                "external_job_id": job_id,
                "inner_job_id": instance["inner_job_id"],
                "source_role": instance["source_role"],
                "instance_name": instance["instance_name"],
                "authorization": authorization,
                "authorization_sha256": external_auth.canonical_sha256(
                    authorization
                ),
                "package_generations": dict(row.package_generations),
                "external_preflight_receipt": copy.deepcopy(
                    dict(row.external_preflight_receipt)
                ),
                "initial_metadata_values": values,
                "initial_metadata_receipt_sha256": _sha(
                    metadata_receipt.get(
                        "metadata_budget_receipt_sha256"
                    ),
                    "initial metadata receipt",
                ),
                "bootstrap_role_manifest": manifest,
                "bootstrap_role_manifest_sha256": manifest[
                    "role_manifest_sha256"
                ],
                "claim_nonce": _sha(row.claim_nonce, "claim nonce"),
            }
        )
        insert_bodies.append(insert_body)
    if len({row["claim_nonce"] for row in checked_rows}) != VM_COUNT:
        raise ValueError("claim nonces collided")
    return checked_rows, insert_bodies


def _cleanup_receipt_sha(value: Mapping[str, Any], label: str) -> str:
    receipt = copy.deepcopy(dict(value))
    supplied = receipt.pop("receipt_sha256", None)
    if not isinstance(supplied, str) or _SHA256.fullmatch(supplied) is None:
        # Adapters may use a differently named canonical receipt digest.
        supplied = canonical_sha256(receipt)
    _reject_secret_fields(receipt)
    return supplied


def _failure_cleanup(
    *,
    phase2_teardown: Phase2TeardownAdapter,
    controller_sa_admin: ControllerServiceAccountAdmin,
    user_compute_client: UserComputeClient,
    instance_names: Sequence[str],
    failure_evidence_sha256: str,
) -> list[dict[str, Any]]:
    records = []
    operations = (
        (
            "phase2_iam_revoke",
            lambda: phase2_teardown.cleanup_phase2_iam_on_failure(
                failure_evidence_sha256=failure_evidence_sha256
            ),
        ),
        (
            "controller_service_account_delete",
            controller_sa_admin.cleanup_delete_if_present,
        ),
        (
            "exact_instance_and_disk_cleanup",
            lambda: user_compute_client.cleanup_exact_instances_and_disks(
                instance_names=list(instance_names),
                disk_names=list(instance_names),
            ),
        ),
    )
    for order, (operation, callback) in enumerate(operations, start=1):
        try:
            result = callback()
            if not isinstance(result, Mapping):
                raise ValueError("cleanup callback result changed")
            records.append(
                {
                    "order": order,
                    "operation": operation,
                    "completed": True,
                    "receipt_sha256": _cleanup_receipt_sha(
                        result, operation
                    ),
                }
            )
        except Exception:
            records.append(
                {
                    "order": order,
                    "operation": operation,
                    "completed": False,
                    "sanitized_failure": "cleanup_callback_failed",
                }
            )
    return records


def run_exact_pair_attempt0(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    source_plan: Mapping[str, Any],
    source_provision: bootstrap_source.ValidatedBootstrapSourceProvision,
    role_launches: Sequence[PreparedRoleLaunch],
    phase2_iam_plan: Mapping[str, Any],
    phase2_installed: phase2_lifecycle.Phase2IamInstalledCapability,
    controller_sa_created: Any,
    controller_client: ControllerComputeClient,
    user_compute_client: UserComputeClient,
    phase2_teardown: Phase2TeardownAdapter,
    controller_sa_admin: ControllerServiceAccountAdmin,
    signer: external_auth.ControllerSigner,
    request_ids: Sequence[str],
    now_unix_seconds: int,
    pair_release_issued_unix_seconds: int,
    pair_release_nonce: str,
    metadata_adapter: MetadataContractAdapter | None = None,
) -> dict[str, Any]:
    """Execute exactly candidate+reference attempt0 with no hidden retry."""

    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    plan = phase2_iam.validate_step12b_phase2_iam_plan(
        phase2_iam_plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    validated_source_plan = bootstrap_source.validate_bootstrap_source_plan(
        source_plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    bootstrap_source.require_validated_bootstrap_source_provision(
        source_provision,
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        source_plan_sha256=validated_source_plan["source_plan_sha256"],
    )
    phase2_exact_readback = _validate_installed_capability(
        phase2_installed,
        deployment=deployment,
        plan=plan,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    controller_create_receipt = _validate_created_capability(
        controller_sa_created, deployment=deployment
    )
    _validate_clients(controller_client, user_compute_client, deployment)
    ids = _request_ids(request_ids)
    if (
        type(now_unix_seconds) is not int
        or now_unix_seconds <= 0
        or type(pair_release_issued_unix_seconds) is not int
        or pair_release_issued_unix_seconds <= 0
    ):
        raise ValueError("controller time input changed")
    release_nonce = _sha(pair_release_nonce, "pair release nonce")
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    if dict(signer.public_record) != public_record:
        raise ValueError("controller signer changed")
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        public_record
    )
    adapter = (
        VmMetadataContractAdapter()
        if metadata_adapter is None
        else metadata_adapter
    )
    roles, insert_bodies = _validate_roles(
        role_launches,
        deployment=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=public_record,
        run_nonce=run_nonce,
        source_plan=validated_source_plan,
        source_provision=source_provision,
        verifier=verifier,
        metadata_adapter=adapter,
        now_unix_seconds=now_unix_seconds,
    )
    names = [row["instance_name"] for row in roles]
    if (
        names != [row["instance_name"] for row in deployment["instances"]]
        or len(set(names)) != VM_COUNT
        or any(row["attempt_index"] != ATTEMPT_INDEX for row in deployment["instances"])
    ):
        raise ValueError("attempt0 exact instance set changed")

    event_records: list[dict[str, Any]] = []
    f0: list[dict[str, Any]] = []
    f1: list[dict[str, Any]] = []
    f2: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    claim_receipts: list[dict[str, Any]] = []
    claim_values: list[str] = []
    postclaim_values: list[dict[str, str]] = []
    failure_stage = "before_insert"
    try:
        for position, (name, body, role) in enumerate(
            zip(names, insert_bodies, roles, strict=True)
        ):
            failure_stage = f"insert_{position}"
            event_records.append(
                _mutation(
                    controller_client.insert_instance(
                        instance_name=name,
                        request_id=ids[position],
                        body=body,
                    ),
                    expected_name=name,
                    expected_request_id=ids[position],
                    operation="insert",
                )
            )
            failure_stage = f"provider_f0_{position}"
            observed = _provider_identity(
                controller_client.get_instance(instance_name=name),
                expected_name=name,
                expected_metadata=role["initial_metadata_values"],
            )
            f0.append(observed)
            event_records.append(
                {
                    "operation": "provider_get_f0",
                    "target_name": name,
                    "provider_record_sha256": observed[
                        "provider_record_sha256"
                    ],
                }
            )
        if len({row["provider_instance_id"] for row in f0}) != VM_COUNT:
            raise _Abort("provider_instance_ids_collided")

        for position, (name, role, observed) in enumerate(
            zip(names, roles, f0, strict=True)
        ):
            failure_stage = f"claim_build_{position}"
            claim = external_auth.build_external_worker_claim(
                authorization=role["authorization"],
                deployment_contract=deployment,
                candidate_payload_contract=candidate_payload_contract,
                reference_payload_contract=reference_payload_contract,
                controller_public_key_record=public_record,
                run_nonce=run_nonce,
                external_job_id=role["external_job_id"],
                package_generations=role["package_generations"],
                external_preflight_receipt=role[
                    "external_preflight_receipt"
                ],
                project_number=PROJECT_NUMBER,
                provider_instance_id=observed["provider_instance_id"],
                metadata_fingerprint=observed["metadata_fingerprint"],
                now_unix_seconds=now_unix_seconds,
                signer=signer,
                nonce=role["claim_nonce"],
            )
            claim_value = external_auth.canonical_bytes(claim).decode(
                "ascii"
            )
            updated, budget = adapter.add_claim(
                initial_values=role["initial_metadata_values"],
                claim_value=claim_value,
            )
            updated_values = dict(updated)
            _sha(
                budget.get("metadata_budget_receipt_sha256"),
                "postclaim metadata receipt",
            )
            failure_stage = f"claim_cas_{position}"
            event_records.append(
                _mutation(
                    controller_client.set_metadata(
                        instance_name=name,
                        request_id=ids[2 + position],
                        expected_fingerprint=observed[
                            "metadata_fingerprint"
                        ],
                        values=updated_values,
                    ),
                    expected_name=name,
                    expected_request_id=ids[2 + position],
                    operation="claim_cas",
                )
            )
            failure_stage = f"provider_f1_{position}"
            claimed = _provider_identity(
                controller_client.get_instance(instance_name=name),
                expected_name=name,
                expected_metadata=updated_values,
                prior_instance_id=observed["provider_instance_id"],
                prior_fingerprint=observed["metadata_fingerprint"],
            )
            receipt = pair_release.build_claim_cas_readback_receipt(
                deployment_contract=deployment,
                candidate_payload_contract=candidate_payload_contract,
                reference_payload_contract=reference_payload_contract,
                controller_public_key_record=public_record,
                run_nonce=run_nonce,
                external_job_id=role["external_job_id"],
                authorization=role["authorization"],
                claim=claim,
                package_generations=role["package_generations"],
                external_preflight_receipt=role[
                    "external_preflight_receipt"
                ],
                verifier=verifier,
                now_unix_seconds=now_unix_seconds,
                provider_instance_id=observed["provider_instance_id"],
                controller_pre_cas_metadata_fingerprint=observed[
                    "metadata_fingerprint"
                ],
                controller_post_cas_metadata_fingerprint=claimed[
                    "metadata_fingerprint"
                ],
                claimed_metadata_sha256=canonical_sha256(updated_values),
            )
            claims.append(claim)
            claim_values.append(claim_value)
            postclaim_values.append(updated_values)
            f1.append(claimed)
            claim_receipts.append(receipt)
            event_records.append(
                {
                    "operation": "provider_get_f1_claim_readback",
                    "target_name": name,
                    "claim_receipt_sha256": receipt["receipt_sha256"],
                    "provider_record_sha256": claimed[
                        "provider_record_sha256"
                    ],
                }
            )

        failure_stage = "phase2_controller_remove"
        phase2_zero_receipt = dict(
            phase2_teardown.remove_controller_bindings_after_claims(
                installed=phase2_installed,
                claim_cas_readback_receipts=claim_receipts,
            )
        )
        pair_release.validate_phase2_zero_receipt(
            phase2_zero_receipt,
            deployment_contract=deployment,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=public_record,
            run_nonce=run_nonce,
            phase2_iam_plan=plan,
        )
        event_records.append(
            {
                "operation": "phase2_controller_bindings_zero",
                "receipt_sha256": phase2_zero_receipt["receipt_sha256"],
            }
        )

        failure_stage = "controller_service_account_delete"
        controller_delete_receipt = dict(
            controller_sa_admin.delete_created_and_wait_absent(
                created=controller_sa_created
            )
        )
        pair_release.validate_controller_delete_receipt(
            controller_delete_receipt,
            deployment_contract=deployment,
            controller_create_receipt=controller_create_receipt,
        )
        event_records.append(
            {
                "operation": "controller_service_account_deleted_get404",
                "receipt_sha256": controller_delete_receipt[
                    "receipt_sha256"
                ],
            }
        )

        failure_stage = "pair_release_build"
        release = pair_release.build_pair_release(
            deployment_contract=deployment,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=public_record,
            run_nonce=run_nonce,
            claim_cas_readback_receipts=claim_receipts,
            phase2_iam_plan=plan,
            phase2_zero_receipt=phase2_zero_receipt,
            controller_create_receipt=controller_create_receipt,
            controller_delete_receipt=controller_delete_receipt,
            issued_unix_seconds=pair_release_issued_unix_seconds,
            signer=signer,
            nonce=release_nonce,
        )
        release_value = pair_release.canonical_bytes(release).decode(
            "ascii"
        )
        event_records.append(
            {
                "operation": "signed_pair_release_built",
                "pair_release_sha256": pair_release.canonical_sha256(
                    release
                ),
            }
        )

        for position, (name, role, claim_value, claimed) in enumerate(
            zip(names, roles, claim_values, f1, strict=True)
        ):
            updated, budget = adapter.add_release(
                initial_values=role["initial_metadata_values"],
                claim_value=claim_value,
                release_value=release_value,
            )
            release_values = dict(updated)
            _sha(
                budget.get("metadata_budget_receipt_sha256"),
                "postrelease metadata receipt",
            )
            failure_stage = f"release_cas_{position}"
            event_records.append(
                _mutation(
                    user_compute_client.set_metadata(
                        instance_name=name,
                        request_id=ids[4 + position],
                        expected_fingerprint=claimed[
                            "metadata_fingerprint"
                        ],
                        values=release_values,
                    ),
                    expected_name=name,
                    expected_request_id=ids[4 + position],
                    operation="pair_release_cas",
                )
            )
            failure_stage = f"provider_f2_{position}"
            released = _provider_identity(
                user_compute_client.get_instance(instance_name=name),
                expected_name=name,
                expected_metadata=release_values,
                prior_instance_id=claimed["provider_instance_id"],
                prior_fingerprint=claimed["metadata_fingerprint"],
            )
            f2.append(released)
            event_records.append(
                {
                    "operation": "user_provider_get_f2_release_readback",
                    "target_name": name,
                    "provider_record_sha256": released[
                        "provider_record_sha256"
                    ],
                }
            )
    except Exception as error:
        reason = error.code if isinstance(error, _Abort) else (
            "cloud_controller_stage_failed"
        )
        evidence = canonical_sha256(
            {
                "deployment_contract_sha256": deployment[
                    "deployment_contract_sha256"
                ],
                "failure_stage": failure_stage,
                "failure_reason": reason,
                "event_records_sha256": canonical_sha256(event_records),
            }
        )
        cleanup_records = _failure_cleanup(
            phase2_teardown=phase2_teardown,
            controller_sa_admin=controller_sa_admin,
            user_compute_client=user_compute_client,
            instance_names=names,
            failure_evidence_sha256=evidence,
        )
        raise ExactPairCloudControllerError(
            _seal(
                {
                    "schema": FAILURE_RECEIPT_SCHEMA,
                    "deployment_contract_sha256": deployment[
                        "deployment_contract_sha256"
                    ],
                    "phase2_iam_plan_sha256": plan["plan_sha256"],
                    "phase2_exact_readback_receipt_sha256": (
                        phase2_exact_readback["receipt_sha256"]
                    ),
                    "failure_stage": failure_stage,
                    "failure_reason": reason,
                    "event_records": event_records,
                    "event_records_sha256": canonical_sha256(
                        event_records
                    ),
                    "cleanup_records": cleanup_records,
                    "cleanup_records_sha256": canonical_sha256(
                        cleanup_records
                    ),
                    "cleanup_order": [
                        "phase2_iam_revoke",
                        "controller_service_account_delete",
                        "exact_instance_and_disk_cleanup",
                    ],
                    "automatic_retry_performed": False,
                    "attempt1_authorized": False,
                    "third_vm_authorized": False,
                    "pair_release_completed": False,
                    "access_token_stored": False,
                    "authorization_header_stored": False,
                    "raw_response_body_stored": False,
                    "current_profile_changed": False,
                }
            )
        ) from None

    body = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "status": "exact_pair_attempt0_claimed_and_released",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "phase2_iam_plan_sha256": plan["plan_sha256"],
        "phase2_exact_readback_receipt_sha256": (
            phase2_exact_readback["receipt_sha256"]
        ),
        "bootstrap_source_plan_sha256": validated_source_plan[
            "source_plan_sha256"
        ],
        "bootstrap_source_provision_receipt_sha256": (
            source_provision.provision_receipt_sha256
        ),
        "selected_job_ids": list(deployment["selected_job_ids"]),
        "source_roles": list(deployment["source_roles"]),
        "instance_names": names,
        "disk_names": names,
        "attempt_index": ATTEMPT_INDEX,
        "vm_count": VM_COUNT,
        "machine_type": MACHINE_TYPE,
        "request_ids": ids,
        "request_ids_sha256": canonical_sha256(ids),
        "insert_request_count": 2,
        "claim_cas_request_count": 2,
        "release_cas_request_count": 2,
        "controller_provider_get_count": 4,
        "user_provider_get_count": 2,
        "event_records": event_records,
        "event_records_sha256": canonical_sha256(event_records),
        "provider_instance_ids": [
            row["provider_instance_id"] for row in f2
        ],
        "claim_sha256s": [
            external_auth.canonical_sha256(row) for row in claims
        ],
        "claim_receipt_sha256s": [
            row["receipt_sha256"] for row in claim_receipts
        ],
        "phase2_zero_receipt_sha256": phase2_zero_receipt[
            "receipt_sha256"
        ],
        "controller_create_receipt_sha256": controller_create_receipt[
            "receipt_sha256"
        ],
        "controller_delete_receipt_sha256": controller_delete_receipt[
            "receipt_sha256"
        ],
        "pair_release_sha256": pair_release.canonical_sha256(release),
        "both_claim_provider_readbacks_complete": True,
        "controller_bindings_zero_before_release": True,
        "controller_service_account_get404_before_release": True,
        "user_only_compute_after_controller_teardown": True,
        "both_release_cas_readbacks_complete": True,
        "attempt1_authorized": False,
        "third_vm_authorized": False,
        "automatic_retry_performed": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "raw_response_body_stored": False,
        "current_profile_changed": False,
    }
    return _seal(body)


__all__ = [
    "ComputeMutationResult",
    "ControllerComputeClient",
    "ControllerServiceAccountAdmin",
    "ExactPairCloudControllerError",
    "FAILURE_RECEIPT_SCHEMA",
    "FINAL_RECEIPT_SCHEMA",
    "MetadataContractAdapter",
    "Phase2TeardownAdapter",
    "PreparedRoleLaunch",
    "SCHEMA",
    "UserComputeClient",
    "VmMetadataContractAdapter",
    "build_exact_insert_body",
    "canonical_bytes",
    "canonical_sha256",
    "run_exact_pair_attempt0",
]
