"""Signed Step 12b external authorization and post-create worker claim.

This module is an add-only bridge around the immutable v1 scientific payload.
It signs the fresh external deployment identity; it does not mutate IAM,
create a VM, upload an object, or execute the payload.

The authorization binds one external job alias to its immutable inner job,
the exact deployment and payload digests, one attempt-0 instance, the package
generation snapshot, and an external preflight receipt.  The claim then binds
that authorization to the provider-observed instance identity and metadata
fingerprint.  Both records are verified before a runtime approval is returned.
The preflight receipt itself must be validated by its owning gate module before
its canonical digest is passed here; a caller-supplied digest alone is not
treated as proof that the gate passed.
"""

from __future__ import annotations

import copy
import re
import secrets
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)


AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_external_authorization_v2"
)
CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_external_worker_claim_v2"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_external_preflight_receipt_v2"
)
PREFLIGHT_RECEIPT_STATUS = (
    "passed_typed_preflight_launch_still_requires_signed_authorization"
)
ATTEMPT_INDEX = 0
MAX_AUTHORIZATION_WINDOW_SECONDS = 7_200
MIN_AUTHORIZATION_WINDOW_SECONDS = 60
EXPECTED_PROJECT_NUMBER = "783381566570"
BOOTSTRAP_SOURCE_BINDING_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_source_external_preflight_binding_v2"
)

ALLOWED_OPERATIONS = [
    "metadata_identity_read",
    "bounded_host_prerequisite_install",
    "metadata_token_read",
    "generation_pinned_bootstrap_source_download",
    "generation_pinned_immutable_package_download",
    "local_immutable_payload_execute",
    "generation_match_zero_deployment_result_upload",
    "deployment_result_readback",
    "compute_delete_self_after_done",
    "bounded_safety_shutdown_on_any_worker_failure",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_ID = re.compile(r"^[1-9][0-9]*$")
_METADATA_FINGERPRINT = re.compile(r"^[A-Za-z0-9_+/=-]{8,256}$")
_FORBIDDEN_FIELD_PARTS = (
    "private_key",
    "access_token",
    "authorization_header",
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
)


class ControllerSigner(Protocol):
    public_record: Mapping[str, Any]

    def sign(
        self, *, record_type: str, unsigned: Mapping[str, Any]
    ) -> str: ...


class ControllerVerifier(Protocol):
    key_id: str
    public_key_sha256: str

    def verify(
        self,
        *,
        record_type: str,
        payload: bytes,
        signature: str,
    ) -> bool: ...


@dataclass(frozen=True)
class ExternalRuntimeApproval:
    deployment_contract_sha256: str
    authorization_sha256: str
    claim_sha256: str
    external_job_id: str
    inner_job_id: str
    source_role: str
    instance_name: str
    provider_instance_id: str
    metadata_fingerprint: str
    package_generations: Mapping[str, int]


_VALIDATED_PREFLIGHT_SEAL = object()


@dataclass(frozen=True)
class ValidatedExternalPreflight:
    """Non-serializable capability minted only after the owning real gate."""

    _receipt: Mapping[str, Any]
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _VALIDATED_PREFLIGHT_SEAL:
            raise ValueError(
                "validated preflight capability was not minted by its gate"
            )

    def __reduce__(self) -> Any:
        raise TypeError(
            "validated external preflight capability is not serializable"
        )


def canonical_bytes(value: Any) -> bytes:
    return deployment_v2.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return deployment_v2.canonical_sha256(value)


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
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


def _decimal_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _DECIMAL_ID.fullmatch(value) is None
        or len(value) > 32
    ):
        raise ValueError(f"{label} must be a positive decimal identifier")
    return value


def _metadata_fingerprint(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _METADATA_FINGERPRINT.fullmatch(value) is None
    ):
        raise ValueError("metadata fingerprint changed")
    return value


def _signature(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 8_192
        or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
    ):
        raise ValueError(f"{label} changed")
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


def _deployment_self_digest(
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    deployment = dict(deployment_contract)
    supplied = _sha(
        deployment.pop("deployment_contract_sha256", None),
        "Step12b deployment contract",
    )
    if canonical_sha256(deployment) != supplied:
        raise ValueError("Step12b deployment contract digest changed")
    return dict(deployment_contract)


def _validate_bootstrap_source_binding(
    value: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    binding = dict(value)
    _exact(
        binding,
        {
            "schema",
            "deployment_contract_sha256",
            "bootstrap_source_content_binding_sha256",
            "source_plan_sha256",
            "source_provision_receipt_sha256",
            "source_prefix",
            "source_generations",
            "source_generations_sha256",
            "source_object_count",
            "role_manifests",
            "role_manifests_sha256",
        },
        "bootstrap source preflight binding",
    )
    generations = binding["source_generations"]
    roles = binding["role_manifests"]
    if not isinstance(generations, Mapping) or len(generations) != 3:
        raise ValueError("bootstrap source generations changed")
    expected_prefix = (
        f"gs://{payload_transport.BUCKET}/"
        f"{deployment_v2.DIRECT_NAMESPACE}/bootstrap-sources/"
        f"{deployment['deployment_contract_sha256']}"
    )
    expected_uris = {
        f"{expected_prefix}/runtime_source_bundle.json",
        f"{expected_prefix}/candidate_payload_contract.json",
        f"{expected_prefix}/reference_payload_contract.json",
    }
    for uri, generation in generations.items():
        if (
            not isinstance(uri, str)
            or uri not in expected_uris
            or type(generation) is not int
            or generation <= 0
        ):
            raise ValueError("bootstrap source generation changed")
    if set(generations) != expected_uris:
        raise ValueError("bootstrap source generation set changed")
    if not isinstance(roles, list) or len(roles) != deployment_v2.VM_COUNT:
        raise ValueError("bootstrap source role manifest set changed")
    role_fields = {
        "external_job_id",
        "inner_job_id",
        "source_role",
        "object_count",
        "objects",
        "role_manifest_sha256",
        "objects_sha256",
    }
    for position, row in enumerate(roles):
        if not isinstance(row, Mapping):
            raise ValueError("bootstrap source role manifest changed")
        _exact(row, role_fields, "bootstrap source role manifest")
        objects = row["objects"]
        if (
            row["external_job_id"]
            != deployment["selected_job_ids"][position]
            or row["inner_job_id"]
            != deployment["instances"][position]["inner_job_id"]
            or row["source_role"] != deployment["source_roles"][position]
            or row["object_count"] != 2
            or not isinstance(objects, list)
            or len(objects) != 2
            or row["objects_sha256"] != canonical_sha256(objects)
        ):
            raise ValueError("bootstrap source role manifest mapping changed")
        expected_role_kind = (
            f"{deployment['source_roles'][position]}_role_payload_contract"
        )
        expected_role_path = (
            "candidate_payload_contract.json"
            if position == 0
            else "reference_payload_contract.json"
        )
        if (
            any(not isinstance(record, Mapping) for record in objects)
            or [record.get("kind") for record in objects]
            != [
            "shared_runtime_source_bundle",
            expected_role_kind,
            ]
            or [record.get("path") for record in objects]
            != ["runtime_source_bundle.json", expected_role_path]
        ):
            raise ValueError("bootstrap source role object selection changed")
        object_fields = {
            "kind",
            "path",
            "uri",
            "bytes",
            "sha256",
            "generation",
            "created",
            "readback_verified",
        }
        for record in objects:
            _exact(
                record,
                object_fields,
                "bootstrap source role object",
            )
            if (
                record["uri"] not in generations
                or record["generation"] != generations[record["uri"]]
                or record["uri"]
                != f"{expected_prefix}/{record['path']}"
                or type(record["bytes"]) is not int
                or not 1 <= record["bytes"] <= 1_048_576
                or record["created"] is not True
                or record["readback_verified"] is not True
            ):
                raise ValueError("bootstrap source role object changed")
            _sha(record["sha256"], "bootstrap source role object")
        _sha(row["role_manifest_sha256"], "role bootstrap manifest")
        _sha(row["objects_sha256"], "role bootstrap objects")
    deployment_source_content = deployment.get(
        "bootstrap_source_content_binding"
    )
    if not isinstance(deployment_source_content, Mapping):
        raise ValueError(
            "bootstrap source binding requires deployment content identity"
        )
    if (
        binding["schema"] != BOOTSTRAP_SOURCE_BINDING_SCHEMA
        or binding["deployment_contract_sha256"]
        != deployment["deployment_contract_sha256"]
        or binding["bootstrap_source_content_binding_sha256"]
        != deployment_source_content.get(
            "bootstrap_source_content_binding_sha256"
        )
        or binding["source_prefix"] != expected_prefix
        or binding["source_generations_sha256"]
        != canonical_sha256(generations)
        or binding["source_object_count"] != 3
        or binding["role_manifests_sha256"] != canonical_sha256(roles)
    ):
        raise ValueError("bootstrap source preflight binding changed")
    for field in (
        "bootstrap_source_content_binding_sha256",
        "source_plan_sha256",
        "source_provision_receipt_sha256",
        "source_generations_sha256",
        "role_manifests_sha256",
    ):
        _sha(binding[field], field)
    return {
        **binding,
        "source_generations": dict(generations),
        "role_manifests": [dict(row) for row in roles],
    }


def _role_bootstrap_source_binding(
    preflight_binding: Mapping[str, Any],
    *,
    external_job_id: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in preflight_binding["role_manifests"]
        if row["external_job_id"] == external_job_id
    ]
    if len(matches) != 1:
        raise ValueError("role bootstrap source manifest changed")
    role = matches[0]
    role_generations = {
        row["uri"]: row["generation"] for row in role["objects"]
    }
    return {
        "bootstrap_source_content_binding_sha256": preflight_binding[
            "bootstrap_source_content_binding_sha256"
        ],
        "source_plan_sha256": preflight_binding["source_plan_sha256"],
        "source_provision_receipt_sha256": preflight_binding[
            "source_provision_receipt_sha256"
        ],
        "source_prefix": preflight_binding["source_prefix"],
        "source_generations_sha256": preflight_binding[
            "source_generations_sha256"
        ],
        "source_object_count": preflight_binding["source_object_count"],
        "role_source_generations": role_generations,
        "role_source_generations_sha256": canonical_sha256(
            role_generations
        ),
        "role_manifest_sha256": role["role_manifest_sha256"],
        "role_manifest_object_count": role["object_count"],
        "role_manifest_objects": [
            dict(row) for row in role["objects"]
        ],
        "role_manifest_objects_sha256": role["objects_sha256"],
    }


def _build_external_preflight_receipt_record(
    *,
    deployment_contract: Mapping[str, Any],
    readonly_preflight_receipt_sha256: str,
    iam_capacity_gate_receipt_sha256: str,
    package_provision_receipt_sha256: str,
    deployment_source_sha256: str,
    alias_bridge_source_sha256: str,
    prebootstrap_source_sha256: str,
    startup_source_sha256: str,
    controller_source_sha256: str,
    bootstrap_source_binding: Mapping[str, Any] | None = None,
    upstream_cloud_mutation_evidence_bound: bool = False,
) -> dict[str, Any]:
    """Internal record builder; the real gate must validate all inputs first."""

    deployment = _deployment_self_digest(deployment_contract)
    observation_receipts = {
        "readonly_preflight_receipt_sha256": _sha(
            readonly_preflight_receipt_sha256,
            "read-only preflight receipt",
        ),
        "iam_capacity_gate_receipt_sha256": _sha(
            iam_capacity_gate_receipt_sha256,
            "IAM/capacity gate receipt",
        ),
        "package_provision_receipt_sha256": _sha(
            package_provision_receipt_sha256,
            "package provision receipt",
        ),
    }
    source_hashes = {
        "deployment_source_sha256": _sha(
            deployment_source_sha256, "deployment source"
        ),
        "alias_bridge_source_sha256": _sha(
            alias_bridge_source_sha256, "alias bridge source"
        ),
        "prebootstrap_source_sha256": _sha(
            prebootstrap_source_sha256, "prebootstrap source"
        ),
        "startup_source_sha256": _sha(
            startup_source_sha256, "startup source"
        ),
        "controller_source_sha256": _sha(
            controller_source_sha256, "controller source"
        ),
    }
    checked_bootstrap_source = (
        _validate_bootstrap_source_binding(
            bootstrap_source_binding,
            deployment=deployment,
        )
        if bootstrap_source_binding is not None
        else None
    )
    if type(upstream_cloud_mutation_evidence_bound) is not bool:
        raise ValueError("upstream cloud mutation evidence flag changed")
    body = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "status": PREFLIGHT_RECEIPT_STATUS,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "run_nonce": deployment["run_nonce"],
        "run_identity_sha256": deployment["run_identity_sha256"],
        "direct_stage_identity_sha256": deployment[
            "direct_stage_identity_sha256"
        ],
        "controller_service_account": dict(
            deployment["controller_service_account"]
        ),
        "package_inventory_records_sha256": deployment[
            "payload_binding"
        ]["package_inventory_records_sha256"],
        "current_profile_sha256": deployment["current_profile_sha256"],
        "observation_receipts": observation_receipts,
        "observation_receipts_sha256": canonical_sha256(
            observation_receipts
        ),
        "source_hashes": source_hashes,
        "source_hashes_sha256": canonical_sha256(source_hashes),
        **(
            {"bootstrap_source_binding": checked_bootstrap_source}
            if checked_bootstrap_source is not None
            else {}
        ),
        "checks": {
            "read_only_preflight_validated": True,
            "iam_capacity_gate_validated": True,
            "package_provision_receipt_validated": True,
            "package_generations_validated": True,
            "stage_prefix_empty_validated": True,
            "instance_absence_validated": True,
            "disk_absence_validated": True,
            "source_hashes_validated": True,
            **(
                {"bootstrap_source_binding_validated": True}
                if checked_bootstrap_source is not None
                else {}
            ),
            "upstream_cloud_mutation_evidence_bound": (
                upstream_cloud_mutation_evidence_bound
            ),
        },
        "passed": True,
        "launch_authorized": False,
        "upstream_cloud_mutation_evidence_bound": (
            upstream_cloud_mutation_evidence_bound
        ),
        "cloud_mutation_performed_before_gate": (
            upstream_cloud_mutation_evidence_bound
        ),
        "cloud_mutation_performed_by_gate": False,
        "current_profile_changed": False,
    }
    _reject_forbidden(body)
    return {
        **body,
        "external_preflight_receipt_sha256": canonical_sha256(body),
    }


def build_external_preflight_receipt(**_: Any) -> dict[str, Any]:
    """Fail closed until the real Step12b gate owns capability construction."""

    raise ValueError(
        "external preflight receipt requires the owning real gate validators"
    )


def _mint_validated_external_preflight_after_gate(
    receipt: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    gate_validation_seal: object,
) -> ValidatedExternalPreflight:
    """Internal handoff used only by the owning gate after real validation."""

    if gate_validation_seal is not _VALIDATED_PREFLIGHT_SEAL:
        raise ValueError("real gate validation capability is missing")
    checked = validate_external_preflight_receipt(
        receipt,
        deployment_contract=deployment_contract,
    )
    return ValidatedExternalPreflight(
        _receipt=MappingProxyType(dict(checked)),
        _seal=_VALIDATED_PREFLIGHT_SEAL,
    )


def _validated_preflight_receipt(
    capability: ValidatedExternalPreflight,
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(capability, ValidatedExternalPreflight)
        or capability._seal is not _VALIDATED_PREFLIGHT_SEAL
    ):
        raise ValueError(
            "signed authorization requires validated preflight capability"
        )
    return validate_external_preflight_receipt(
        capability._receipt,
        deployment_contract=deployment_contract,
    )


def get_validated_external_preflight_receipt(
    capability: ValidatedExternalPreflight,
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fresh, fully revalidated copy of an opaque gate receipt."""

    return copy.deepcopy(
        _validated_preflight_receipt(
            capability,
            deployment_contract=deployment_contract,
        )
    )


def validate_external_preflight_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the typed gate/source receipt before signing or trusting it."""

    deployment = _deployment_self_digest(deployment_contract)
    receipt = dict(value)
    has_bootstrap_source = "bootstrap_source_binding" in receipt
    receipt_fields = {
        "schema",
        "status",
        "deployment_contract_sha256",
        "run_nonce",
        "run_identity_sha256",
        "direct_stage_identity_sha256",
        "controller_service_account",
        "package_inventory_records_sha256",
        "current_profile_sha256",
        "observation_receipts",
        "observation_receipts_sha256",
        "source_hashes",
        "source_hashes_sha256",
        "checks",
        "passed",
        "launch_authorized",
        "upstream_cloud_mutation_evidence_bound",
        "cloud_mutation_performed_before_gate",
        "cloud_mutation_performed_by_gate",
        "current_profile_changed",
        "external_preflight_receipt_sha256",
    }
    if has_bootstrap_source:
        receipt_fields.add("bootstrap_source_binding")
    _exact(
        receipt,
        receipt_fields,
        "Step12b external preflight receipt",
    )
    supplied = _sha(
        receipt.pop("external_preflight_receipt_sha256"),
        "external preflight receipt",
    )
    if canonical_sha256(receipt) != supplied:
        raise ValueError("external preflight receipt digest changed")
    observation_receipts = receipt["observation_receipts"]
    source_hashes = receipt["source_hashes"]
    if not isinstance(observation_receipts, Mapping):
        raise ValueError("preflight observation receipts are missing")
    _exact(
        observation_receipts,
        {
            "readonly_preflight_receipt_sha256",
            "iam_capacity_gate_receipt_sha256",
            "package_provision_receipt_sha256",
        },
        "preflight observation receipts",
    )
    for field, sha in observation_receipts.items():
        _sha(sha, field)
    if not isinstance(source_hashes, Mapping):
        raise ValueError("preflight source hashes are missing")
    _exact(
        source_hashes,
        {
            "deployment_source_sha256",
            "alias_bridge_source_sha256",
            "prebootstrap_source_sha256",
            "startup_source_sha256",
            "controller_source_sha256",
        },
        "preflight source hashes",
    )
    for field, sha in source_hashes.items():
        _sha(sha, field)
    checked_bootstrap_source = (
        _validate_bootstrap_source_binding(
            receipt["bootstrap_source_binding"],
            deployment=deployment,
        )
        if has_bootstrap_source
        else None
    )
    expected_checks = {
        "read_only_preflight_validated": True,
        "iam_capacity_gate_validated": True,
        "package_provision_receipt_validated": True,
        "package_generations_validated": True,
        "stage_prefix_empty_validated": True,
        "instance_absence_validated": True,
        "disk_absence_validated": True,
        "source_hashes_validated": True,
        "upstream_cloud_mutation_evidence_bound": receipt[
            "upstream_cloud_mutation_evidence_bound"
        ],
    }
    if checked_bootstrap_source is not None:
        expected_checks["bootstrap_source_binding_validated"] = True
    if (
        receipt["schema"] != PREFLIGHT_RECEIPT_SCHEMA
        or receipt["status"] != PREFLIGHT_RECEIPT_STATUS
        or receipt["deployment_contract_sha256"]
        != deployment["deployment_contract_sha256"]
        or receipt["run_nonce"] != deployment["run_nonce"]
        or receipt["run_identity_sha256"]
        != deployment["run_identity_sha256"]
        or receipt["direct_stage_identity_sha256"]
        != deployment["direct_stage_identity_sha256"]
        or receipt["controller_service_account"]
        != deployment["controller_service_account"]
        or receipt["package_inventory_records_sha256"]
        != deployment["payload_binding"][
            "package_inventory_records_sha256"
        ]
        or receipt["current_profile_sha256"]
        != deployment["current_profile_sha256"]
        or receipt["observation_receipts_sha256"]
        != canonical_sha256(observation_receipts)
        or receipt["source_hashes_sha256"]
        != canonical_sha256(source_hashes)
        or receipt["checks"] != expected_checks
        or receipt["passed"] is not True
        or receipt["launch_authorized"] is not False
        or type(receipt["upstream_cloud_mutation_evidence_bound"])
        is not bool
        or has_bootstrap_source
        is not receipt["upstream_cloud_mutation_evidence_bound"]
        or receipt["cloud_mutation_performed_before_gate"]
        is not receipt["upstream_cloud_mutation_evidence_bound"]
        or receipt["cloud_mutation_performed_by_gate"] is not False
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("typed external preflight receipt changed")
    checked = {**receipt, "external_preflight_receipt_sha256": supplied}
    _reject_forbidden(checked)
    return checked


def _validated_context(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=public_record,
        run_nonce=run_nonce,
    )
    if (
        not isinstance(external_job_id, str)
        or external_job_id not in deployment["selected_job_ids"]
    ):
        raise ValueError("external job alias changed")
    position = deployment["selected_job_ids"].index(external_job_id)
    instance = deployment["instances"][position]
    payload_binding = deployment["payload_binding"]["job_bindings"][position]
    direct_job = deployment["direct_stage_identity"][
        "external_job_layout"
    ][position]
    layout_job = deployment["remote_layout"]["jobs"][position]
    role = deployment["source_roles"][position]
    if (
        instance["job_id"] != external_job_id
        or payload_binding["source_role"] != role
        or direct_job["source_role"] != role
        or layout_job["source_role"] != role
        or instance["source_role"] != role
        or instance["inner_job_id"] != payload_binding["inner_job_id"]
        or direct_job["inner_job_id"] != payload_binding["inner_job_id"]
        or layout_job["inner_job_id"] != payload_binding["inner_job_id"]
        or instance["payload_contract_sha256"]
        != payload_binding["payload_contract_sha256"]
        or direct_job["payload_contract_sha256"]
        != payload_binding["payload_contract_sha256"]
        or instance["attempt_index"] != ATTEMPT_INDEX
        or deployment["attempt_index"] != ATTEMPT_INDEX
    ):
        raise ValueError("external-to-inner job binding changed")
    return deployment, public_record, instance, payload_binding


def _validated_role_context(
    *,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    deployment = deployment_v2.validate_role_runtime_view(
        deployment_contract,
        selected_payload_contract=selected_payload_contract,
        controller_public_key_record=public_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
    )
    position = deployment["selected_job_ids"].index(external_job_id)
    return (
        deployment,
        public_record,
        deployment["instances"][position],
        deployment["payload_binding"]["job_bindings"][position],
    )


def _package_generations(
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    package_generations: Mapping[str, int],
    deployment: Mapping[str, Any],
) -> dict[str, int]:
    candidate = payload_transport.validate_job_contract(
        candidate_payload_contract
    )
    reference = payload_transport.validate_job_contract(
        reference_payload_contract
    )
    candidate_inventory = candidate["remote_layout"]["package_inventory"]
    reference_inventory = reference["remote_layout"]["package_inventory"]
    if (
        reference_inventory != candidate_inventory
        or candidate_inventory["records_sha256"]
        != deployment["payload_binding"][
            "package_inventory_records_sha256"
        ]
    ):
        raise ValueError("immutable package inventory changed")
    expected_uris = [row["uri"] for row in candidate_inventory["records"]]
    if (
        not isinstance(package_generations, Mapping)
        or set(package_generations) != set(expected_uris)
    ):
        raise ValueError("package generation map is incomplete")
    return {
        uri: _integer(
            package_generations[uri],
            "package generation",
            minimum=1,
        )
        for uri in expected_uris
    }


def _role_package_generations(
    *,
    selected_payload_contract: Mapping[str, Any],
    package_generations: Mapping[str, int],
    deployment: Mapping[str, Any],
) -> dict[str, int]:
    selected = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    inventory = selected["remote_layout"]["package_inventory"]
    if inventory["records_sha256"] != deployment["payload_binding"][
        "package_inventory_records_sha256"
    ]:
        raise ValueError("role-local immutable package inventory changed")
    expected_uris = [row["uri"] for row in inventory["records"]]
    if (
        not isinstance(package_generations, Mapping)
        or set(package_generations) != set(expected_uris)
    ):
        raise ValueError("package generation map is incomplete")
    return {
        uri: _integer(
            package_generations[uri],
            "package generation",
            minimum=1,
        )
        for uri in expected_uris
    }


def _require_signer(
    signer: ControllerSigner,
    public_record: Mapping[str, Any],
) -> None:
    signer_record = payload_transport.validate_rsa_public_key_record(
        signer.public_record
    )
    if signer_record != dict(public_record):
        raise ValueError("signer is not pinned by the deployment contract")


def _require_verifier(
    verifier: ControllerVerifier,
    public_record: Mapping[str, Any],
) -> None:
    if (
        getattr(verifier, "key_id", None) != public_record["key_id"]
        or getattr(verifier, "public_key_sha256", None)
        != canonical_sha256(public_record)
    ):
        raise ValueError("verifier is not pinned by the deployment contract")


def _unsigned_authorization(
    *,
    deployment: Mapping[str, Any],
    public_record: Mapping[str, Any],
    instance: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    issued_unix_seconds: int,
    expires_unix_seconds: int,
    nonce: str,
) -> dict[str, Any]:
    bootstrap_preflight_binding = external_preflight_receipt.get(
        "bootstrap_source_binding"
    )
    role_source_binding = (
        _role_bootstrap_source_binding(
            bootstrap_preflight_binding,
            external_job_id=instance["job_id"],
        )
        if isinstance(bootstrap_preflight_binding, Mapping)
        else None
    )
    immutable_source_hashes = {
        "outer_package_identity_sha256": deployment[
            "payload_binding"
        ]["outer_package_identity_sha256"],
        "outer_package_manifest_sha256": deployment[
            "payload_binding"
        ]["outer_package_manifest_sha256"],
        "inner_adapter_preview_sha256": (
            deployment_v2.EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        ),
        "inner_preview_stage_identity_sha256": deployment[
            "payload_binding"
        ]["inner_preview_stage_identity_sha256"],
        "inner_direct_stage_identity_sha256": deployment[
            "payload_binding"
        ]["inner_direct_stage_identity_sha256"],
        "payload_contract_sha256": payload_binding[
            "payload_contract_sha256"
        ],
        "runner_job_manifest_sha256": payload_binding[
            "runner_job_manifest_sha256"
        ],
        "root_records_sha256": payload_binding["root_records_sha256"],
        "tree_paths_sha256": payload_binding["tree_paths_sha256"],
        "package_inventory_records_sha256": deployment[
            "payload_binding"
        ]["package_inventory_records_sha256"],
        **(
            {
                "bootstrap_source_content_binding_sha256": (
                    role_source_binding[
                        "bootstrap_source_content_binding_sha256"
                    ]
                )
            }
            if role_source_binding is not None
            else {}
        ),
    }
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "run_nonce": deployment["run_nonce"],
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
        "payload_contract_sha256": payload_binding[
            "payload_contract_sha256"
        ],
        "payload_contract_sha256s": list(
            deployment["payload_binding"]["payload_contract_sha256s"]
        ),
        "stage_id": deployment["stage_id"],
        "run_name": deployment["run_name"],
        "external_job_id": instance["job_id"],
        "inner_job_id": instance["inner_job_id"],
        "source_role": instance["source_role"],
        "attempt_index": ATTEMPT_INDEX,
        "max_attempts": deployment["authorization_boundary"][
            "attempt_limit_per_job"
        ],
        "instance_name": instance["instance_name"],
        "external_result_layout_sha256": canonical_sha256(
            next(
                row
                for row in deployment["remote_layout"]["jobs"]
                if row["job_id"] == instance["job_id"]
            )
        ),
        "immutable_source_hashes": immutable_source_hashes,
        "immutable_source_hashes_sha256": canonical_sha256(
            immutable_source_hashes
        ),
        "controller_key_id": public_record["key_id"],
        "controller_public_key_sha256": canonical_sha256(public_record),
        "controller_service_account": dict(
            deployment["controller_service_account"]
        ),
        "package_inventory_records_sha256": deployment["payload_binding"][
            "package_inventory_records_sha256"
        ],
        "package_generations_sha256": canonical_sha256(
            package_generations
        ),
        "package_object_count": len(package_generations),
        **(
            {"bootstrap_source_binding": role_source_binding}
            if role_source_binding is not None
            else {}
        ),
        "external_preflight_receipt_schema": external_preflight_receipt[
            "schema"
        ],
        "external_preflight_receipt_sha256": (
            external_preflight_receipt[
                "external_preflight_receipt_sha256"
            ]
        ),
        "preflight_observation_receipts_sha256": (
            external_preflight_receipt[
                "observation_receipts_sha256"
            ]
        ),
        "preflight_source_hashes_sha256": external_preflight_receipt[
            "source_hashes_sha256"
        ],
        "external_preflight_receipt_validation": (
            "typed_exact_receipt_validated"
        ),
        "allowed_operations": list(ALLOWED_OPERATIONS),
        "issued_unix_seconds": issued_unix_seconds,
        "expires_unix_seconds": expires_unix_seconds,
        "nonce": nonce,
    }


def build_external_authorization(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    package_generations: Mapping[str, int],
    validated_external_preflight: ValidatedExternalPreflight,
    issued_unix_seconds: int,
    expires_unix_seconds: int,
    signer: ControllerSigner,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Build one signed attempt-0 authorization without cloud mutation."""

    deployment, public_record, instance, payload_binding = (
        _validated_context(
            deployment_contract=deployment_contract,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=external_job_id,
        )
    )
    _require_signer(signer, public_record)
    generations = _package_generations(
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        package_generations=package_generations,
        deployment=deployment,
    )
    preflight_receipt = _validated_preflight_receipt(
        validated_external_preflight,
        deployment_contract=deployment,
    )
    issued = _integer(
        issued_unix_seconds, "authorization issued", minimum=1
    )
    expires = _integer(
        expires_unix_seconds,
        "authorization expiry",
        minimum=issued + MIN_AUTHORIZATION_WINDOW_SECONDS,
        maximum=issued + MAX_AUTHORIZATION_WINDOW_SECONDS,
    )
    checked_nonce = (
        secrets.token_hex(32)
        if nonce is None
        else _sha(nonce, "authorization nonce")
    )
    if checked_nonce == deployment["run_nonce"]:
        raise ValueError("authorization nonce must differ from run nonce")
    unsigned = _unsigned_authorization(
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        package_generations=generations,
        external_preflight_receipt=preflight_receipt,
        issued_unix_seconds=issued,
        expires_unix_seconds=expires,
        nonce=checked_nonce,
    )
    authorization = {
        **unsigned,
        "signature": signer.sign(
            record_type="authorization", unsigned=unsigned
        ),
    }
    _signature(authorization["signature"], "authorization signature")
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        public_record
    )
    unsigned_bytes = canonical_bytes(unsigned)
    if not verifier.verify(
        record_type="authorization",
        payload=unsigned_bytes,
        signature=authorization["signature"],
    ):
        raise ValueError("new authorization signature did not verify")
    _reject_forbidden(authorization)
    return authorization


def _validate_external_authorization_bound(
    authorization: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
    public_record: Mapping[str, Any],
    instance: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
    generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    verifier: ControllerVerifier,
    now_unix_seconds: int,
) -> dict[str, Any]:
    _require_verifier(verifier, public_record)
    preflight_receipt = validate_external_preflight_receipt(
        external_preflight_receipt,
        deployment_contract=deployment,
    )
    auth = dict(authorization)
    auth_fields = {
        "schema",
        "run_nonce",
        "deployment_contract_sha256",
        "run_identity_sha256",
        "direct_stage_identity_sha256",
        "payload_binding_sha256",
        "payload_contract_sha256",
        "payload_contract_sha256s",
        "stage_id",
        "run_name",
        "external_job_id",
        "inner_job_id",
        "source_role",
        "attempt_index",
        "max_attempts",
        "instance_name",
        "external_result_layout_sha256",
        "immutable_source_hashes",
        "immutable_source_hashes_sha256",
        "controller_key_id",
        "controller_public_key_sha256",
        "controller_service_account",
        "package_inventory_records_sha256",
        "package_generations_sha256",
        "package_object_count",
        "external_preflight_receipt_schema",
        "external_preflight_receipt_sha256",
        "preflight_observation_receipts_sha256",
        "preflight_source_hashes_sha256",
        "external_preflight_receipt_validation",
        "allowed_operations",
        "issued_unix_seconds",
        "expires_unix_seconds",
        "nonce",
        "signature",
    }
    if "bootstrap_source_binding" in preflight_receipt:
        auth_fields.add("bootstrap_source_binding")
    _exact(
        auth,
        auth_fields,
        "Step12b external authorization",
    )
    issued = _integer(
        auth["issued_unix_seconds"],
        "authorization issued",
        minimum=1,
    )
    expires = _integer(
        auth["expires_unix_seconds"],
        "authorization expiry",
        minimum=issued + MIN_AUTHORIZATION_WINDOW_SECONDS,
        maximum=issued + MAX_AUTHORIZATION_WINDOW_SECONDS,
    )
    _integer(
        now_unix_seconds,
        "authorization current time",
        minimum=issued,
        maximum=expires - 1,
    )
    auth_nonce = _sha(auth["nonce"], "authorization nonce")
    if auth_nonce == deployment["run_nonce"]:
        raise ValueError("authorization nonce must differ from run nonce")
    expected_unsigned = _unsigned_authorization(
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        package_generations=generations,
        external_preflight_receipt=preflight_receipt,
        issued_unix_seconds=issued,
        expires_unix_seconds=expires,
        nonce=auth_nonce,
    )
    unsigned = {key: value for key, value in auth.items() if key != "signature"}
    signature = _signature(
        auth["signature"], "authorization signature"
    )
    if unsigned != expected_unsigned:
        raise ValueError("Step12b external authorization identity changed")
    if not verifier.verify(
        record_type="authorization",
        payload=canonical_bytes(unsigned),
        signature=signature,
    ):
        raise ValueError("Step12b authorization trust verification failed")
    _reject_forbidden(auth)
    return auth


def validate_external_authorization(
    authorization: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    verifier: ControllerVerifier,
    now_unix_seconds: int,
) -> dict[str, Any]:
    """Controller-side validation with both payload contracts present."""

    deployment, public_record, instance, payload_binding = (
        _validated_context(
            deployment_contract=deployment_contract,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=external_job_id,
        )
    )
    generations = _package_generations(
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        package_generations=package_generations,
        deployment=deployment,
    )
    return _validate_external_authorization_bound(
        authorization,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        generations=generations,
        external_preflight_receipt=external_preflight_receipt,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )


def validate_external_authorization_role_runtime(
    authorization: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    verifier: ControllerVerifier,
    now_unix_seconds: int,
) -> dict[str, Any]:
    """VM-side validation with only the selected role payload present."""

    deployment, public_record, instance, payload_binding = (
        _validated_role_context(
            deployment_contract=deployment_contract,
            selected_payload_contract=selected_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=external_job_id,
        )
    )
    generations = _role_package_generations(
        selected_payload_contract=selected_payload_contract,
        package_generations=package_generations,
        deployment=deployment,
    )
    return _validate_external_authorization_bound(
        authorization,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        generations=generations,
        external_preflight_receipt=external_preflight_receipt,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )


def _unsigned_claim(
    *,
    authorization: Mapping[str, Any],
    deployment: Mapping[str, Any],
    public_record: Mapping[str, Any],
    instance: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
    project_number: str,
    provider_instance_id: str,
    metadata_fingerprint: str,
    package_generations: Mapping[str, int],
    nonce: str,
) -> dict[str, Any]:
    return {
        "schema": CLAIM_SCHEMA,
        "authorization_sha256": canonical_sha256(authorization),
        "run_nonce": deployment["run_nonce"],
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
        "payload_contract_sha256": payload_binding[
            "payload_contract_sha256"
        ],
        "project": payload_transport.PROJECT,
        "project_number": project_number,
        "zone": payload_transport.ZONE,
        "instance_name": instance["instance_name"],
        "provider_instance_id": provider_instance_id,
        "worker_service_account": payload_transport.WORKER_SERVICE_ACCOUNT,
        "controller_service_account": dict(
            deployment["controller_service_account"]
        ),
        "controller_key_id": public_record["key_id"],
        "controller_public_key_sha256": canonical_sha256(public_record),
        "stage_id": deployment["stage_id"],
        "run_name": deployment["run_name"],
        "external_job_id": instance["job_id"],
        "inner_job_id": instance["inner_job_id"],
        "source_role": instance["source_role"],
        "attempt_index": ATTEMPT_INDEX,
        "max_attempts": deployment["authorization_boundary"][
            "attempt_limit_per_job"
        ],
        "external_result_layout_sha256": canonical_sha256(
            next(
                row
                for row in deployment["remote_layout"]["jobs"]
                if row["job_id"] == instance["job_id"]
            )
        ),
        "metadata_fingerprint": metadata_fingerprint,
        "metadata_fingerprint_semantics": "pre_claim_cas_observation",
        "package_inventory_records_sha256": deployment["payload_binding"][
            "package_inventory_records_sha256"
        ],
        "package_generations": dict(package_generations),
        "package_generations_sha256": canonical_sha256(
            package_generations
        ),
        "package_object_count": len(package_generations),
        "nonce": nonce,
    }


def build_external_worker_claim(
    *,
    authorization: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    project_number: str,
    provider_instance_id: str,
    metadata_fingerprint: str,
    now_unix_seconds: int,
    signer: ControllerSigner,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Build a signed post-create claim for one observed VM identity."""

    deployment, public_record, instance, payload_binding = (
        _validated_context(
            deployment_contract=deployment_contract,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=external_job_id,
        )
    )
    _require_signer(signer, public_record)
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        public_record
    )
    auth = validate_external_authorization(
        authorization,
        deployment_contract=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=public_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
        package_generations=package_generations,
        external_preflight_receipt=external_preflight_receipt,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )
    generations = _package_generations(
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        package_generations=package_generations,
        deployment=deployment,
    )
    claim_nonce = (
        secrets.token_hex(32)
        if nonce is None
        else _sha(nonce, "claim nonce")
    )
    if claim_nonce in (auth["nonce"], deployment["run_nonce"]):
        raise ValueError(
            "claim nonce must differ from authorization and run nonces"
        )
    checked_project_number = _decimal_id(
        project_number, "project number"
    )
    if checked_project_number != EXPECTED_PROJECT_NUMBER:
        raise ValueError("project number changed")
    unsigned = _unsigned_claim(
        authorization=auth,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        project_number=checked_project_number,
        provider_instance_id=_decimal_id(
            provider_instance_id, "provider instance ID"
        ),
        metadata_fingerprint=_metadata_fingerprint(
            metadata_fingerprint
        ),
        package_generations=generations,
        nonce=claim_nonce,
    )
    claim = {
        **unsigned,
        "signature": signer.sign(record_type="claim", unsigned=unsigned),
    }
    _signature(claim["signature"], "worker claim signature")
    if not verifier.verify(
        record_type="claim",
        payload=canonical_bytes(unsigned),
        signature=claim["signature"],
    ):
        raise ValueError("new worker claim signature did not verify")
    _reject_forbidden(claim)
    return claim


def _validate_external_approval_bound(
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    deployment: Mapping[str, Any],
    public_record: Mapping[str, Any],
    instance: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
    generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    observed_project_number: str,
    observed_provider_instance_id: str,
    observed_metadata_fingerprint: str,
    verifier: ControllerVerifier,
    now_unix_seconds: int,
) -> ExternalRuntimeApproval:
    _require_verifier(verifier, public_record)
    auth = _validate_external_authorization_bound(
        authorization,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        generations=generations,
        external_preflight_receipt=external_preflight_receipt,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )
    project_number = _decimal_id(
        observed_project_number, "observed project number"
    )
    if project_number != EXPECTED_PROJECT_NUMBER:
        raise ValueError("observed project number changed")
    provider_instance_id = _decimal_id(
        observed_provider_instance_id, "observed provider instance ID"
    )
    metadata_fingerprint = _metadata_fingerprint(
        observed_metadata_fingerprint
    )
    claimed = dict(claim)
    _exact(
        claimed,
        {
            "schema",
            "authorization_sha256",
            "run_nonce",
            "deployment_contract_sha256",
            "run_identity_sha256",
            "direct_stage_identity_sha256",
            "payload_binding_sha256",
            "payload_contract_sha256",
            "project",
            "project_number",
            "zone",
            "instance_name",
            "provider_instance_id",
            "worker_service_account",
            "controller_service_account",
            "controller_key_id",
            "controller_public_key_sha256",
            "stage_id",
            "run_name",
            "external_job_id",
            "inner_job_id",
            "source_role",
            "attempt_index",
            "max_attempts",
            "external_result_layout_sha256",
            "metadata_fingerprint",
            "metadata_fingerprint_semantics",
            "package_inventory_records_sha256",
            "package_generations",
            "package_generations_sha256",
            "package_object_count",
            "nonce",
            "signature",
        },
        "Step12b external worker claim",
    )
    claim_nonce = _sha(claimed["nonce"], "claim nonce")
    if claim_nonce in (auth["nonce"], deployment["run_nonce"]):
        raise ValueError(
            "claim nonce must differ from authorization and run nonces"
        )
    expected_unsigned = _unsigned_claim(
        authorization=auth,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        project_number=project_number,
        provider_instance_id=provider_instance_id,
        metadata_fingerprint=metadata_fingerprint,
        package_generations=generations,
        nonce=claim_nonce,
    )
    unsigned = {
        key: value for key, value in claimed.items() if key != "signature"
    }
    signature = _signature(claimed["signature"], "worker claim signature")
    if unsigned != expected_unsigned:
        raise ValueError("Step12b worker claim identity changed")
    if not verifier.verify(
        record_type="claim",
        payload=canonical_bytes(unsigned),
        signature=signature,
    ):
        raise ValueError("Step12b worker claim trust verification failed")
    _reject_forbidden(claimed)
    return ExternalRuntimeApproval(
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        authorization_sha256=canonical_sha256(auth),
        claim_sha256=canonical_sha256(claimed),
        external_job_id=instance["job_id"],
        inner_job_id=instance["inner_job_id"],
        source_role=instance["source_role"],
        instance_name=instance["instance_name"],
        provider_instance_id=provider_instance_id,
        metadata_fingerprint=metadata_fingerprint,
        package_generations=MappingProxyType(dict(generations)),
    )


def validate_external_approval(
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    observed_project_number: str,
    observed_provider_instance_id: str,
    observed_metadata_fingerprint: str,
    verifier: ControllerVerifier,
    now_unix_seconds: int,
) -> ExternalRuntimeApproval:
    """Controller-side full-pair validation of the signed approval chain."""

    deployment, public_record, instance, payload_binding = (
        _validated_context(
            deployment_contract=deployment_contract,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=external_job_id,
        )
    )
    generations = _package_generations(
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        package_generations=package_generations,
        deployment=deployment,
    )
    return _validate_external_approval_bound(
        authorization=authorization,
        claim=claim,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        generations=generations,
        external_preflight_receipt=external_preflight_receipt,
        observed_project_number=observed_project_number,
        observed_provider_instance_id=observed_provider_instance_id,
        observed_metadata_fingerprint=observed_metadata_fingerprint,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )


def validate_external_approval_role_runtime(
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    observed_project: str,
    observed_project_number: str,
    observed_zone: str,
    observed_instance_name: str,
    observed_provider_instance_id: str,
    observed_worker_service_account: str,
    observed_oauth_scopes: Sequence[str],
    verifier: ControllerVerifier,
    now_unix_seconds: int,
) -> ExternalRuntimeApproval:
    """VM-side approval validation with one selected payload contract."""

    deployment, public_record, instance, payload_binding = (
        _validated_role_context(
            deployment_contract=deployment_contract,
            selected_payload_contract=selected_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            external_job_id=external_job_id,
        )
    )
    generations = _role_package_generations(
        selected_payload_contract=selected_payload_contract,
        package_generations=package_generations,
        deployment=deployment,
    )
    if (
        observed_project != payload_transport.PROJECT
        or observed_zone != payload_transport.ZONE
        or observed_instance_name != instance["instance_name"]
        or observed_worker_service_account
        != payload_transport.WORKER_SERVICE_ACCOUNT
        or isinstance(observed_oauth_scopes, (str, bytes))
        or not isinstance(observed_oauth_scopes, Sequence)
        or list(observed_oauth_scopes)
        != [payload_transport.REQUIRED_WORKER_OAUTH_SCOPE]
    ):
        raise ValueError("role-runtime provider metadata identity changed")
    return _validate_external_approval_bound(
        authorization=authorization,
        claim=claim,
        deployment=deployment,
        public_record=public_record,
        instance=instance,
        payload_binding=payload_binding,
        generations=generations,
        external_preflight_receipt=external_preflight_receipt,
        observed_project_number=observed_project_number,
        observed_provider_instance_id=observed_provider_instance_id,
        observed_metadata_fingerprint=_metadata_fingerprint(
            claim.get("metadata_fingerprint")
            if isinstance(claim, Mapping)
            else None
        ),
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )


build_controller_authorization = build_external_authorization
build_worker_claim = build_external_worker_claim
validate_controller_approval = validate_external_approval
validate_role_local_external_authorization = (
    validate_external_authorization_role_runtime
)
validate_role_local_external_approval = (
    validate_external_approval_role_runtime
)


__all__ = [
    "ALLOWED_OPERATIONS",
    "ATTEMPT_INDEX",
    "AUTHORIZATION_SCHEMA",
    "BOOTSTRAP_SOURCE_BINDING_SCHEMA",
    "CLAIM_SCHEMA",
    "ControllerSigner",
    "ControllerVerifier",
    "EXPECTED_PROJECT_NUMBER",
    "ExternalRuntimeApproval",
    "MAX_AUTHORIZATION_WINDOW_SECONDS",
    "MIN_AUTHORIZATION_WINDOW_SECONDS",
    "PREFLIGHT_RECEIPT_SCHEMA",
    "PREFLIGHT_RECEIPT_STATUS",
    "ValidatedExternalPreflight",
    "build_controller_authorization",
    "build_external_authorization",
    "build_external_preflight_receipt",
    "build_external_worker_claim",
    "build_worker_claim",
    "canonical_bytes",
    "canonical_sha256",
    "get_validated_external_preflight_receipt",
    "validate_controller_approval",
    "validate_external_approval",
    "validate_external_approval_role_runtime",
    "validate_external_authorization",
    "validate_external_authorization_role_runtime",
    "validate_external_preflight_receipt",
    "validate_role_local_external_approval",
    "validate_role_local_external_authorization",
]
