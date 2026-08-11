"""Signed pair-wide execution release for Step12b.

One worker claim is not enough to run the payload.  The controller signs this
release only after both claim CAS readbacks, exact removal of every phase-2
controller binding, and deletion plus GET-404 readback of the fresh
run-scoped controller service account.  Workers poll a single metadata value
with a bounded timeout and validate the release before entering the alias
bridge.

All provider mutations are outside this module.  Inputs are validated
receipts and injected readers/signers/verifiers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2
    as vm_metadata,
)


CLAIM_CAS_READBACK_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_claim_cas_readback_v2"
)
PHASE2_ZERO_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_controller_bindings_zero_v2"
)
PAIR_RELEASE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_pair_execution_release_v2"
)
ROLE_RUNTIME_APPROVAL_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_role_runtime_approval_v2"
)
PAIR_RELEASE_APPROVAL_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_pair_release_approval_v2"
)

MAX_RELEASE_WAIT_SECONDS = 900
DEFAULT_RELEASE_POLL_SECONDS = 2.0
PAIR_RELEASE_SIGNATURE_RECORD_TYPE = "authorization"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_ID = re.compile(r"^[1-9][0-9]{5,31}$")
_FINGERPRINT = re.compile(r"^[A-Za-z0-9_+/=-]{8,256}$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9_-]+$")
_ROLE_APPROVAL_SEAL = object()
_PAIR_RELEASE_SEAL = object()


class PairReleaseMetadataReader(Protocol):
    def read_pair_release(self) -> str | None: ...


@dataclass(frozen=True)
class ValidatedRoleRuntimeApproval:
    deployment_contract_sha256: str
    authorization_sha256: str
    claim_sha256: str
    external_job_id: str
    inner_job_id: str
    source_role: str
    instance_name: str
    provider_instance_id: str
    controller_preclaim_metadata_fingerprint: str
    package_generations_sha256: str
    vm_identity_receipt_sha256: str
    receipt_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _ROLE_APPROVAL_SEAL:
            raise ValueError(
                "ValidatedRoleRuntimeApproval cannot be forged"
            )


@dataclass(frozen=True)
class PairReleaseApproval:
    deployment_contract_sha256: str
    pair_release_sha256: str
    external_job_id: str
    own_claim_sha256: str
    phase2_iam_plan_sha256: str
    phase2_zero_receipt_sha256: str
    controller_create_receipt_sha256: str
    controller_delete_receipt_sha256: str
    receipt_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PAIR_RELEASE_SEAL:
            raise ValueError("PairReleaseApproval cannot be forged")


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
        raise ValueError(f"{label} must be nonzero lowercase SHA-256")
    return value


def _decimal(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _DECIMAL_ID.fullmatch(value) is None
    ):
        raise ValueError(f"{label} changed")
    return value


def _fingerprint(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _FINGERPRINT.fullmatch(value) is None
    ):
        raise ValueError(f"{label} changed")
    return value


def _nonce_digest(value: Any, label: str) -> str:
    """Hash every nonce in one common domain so collisions remain visible."""

    return canonical_sha256({"nonce": _sha(value, label)})


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_sealed(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    supplied = _sha(checked.pop("receipt_sha256", None), label)
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return dict(value)


def build_claim_cas_readback_receipt(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
    now_unix_seconds: int,
    provider_instance_id: str,
    controller_pre_cas_metadata_fingerprint: str,
    controller_post_cas_metadata_fingerprint: str,
    claimed_metadata_sha256: str,
) -> dict[str, Any]:
    """Seal one controller-side claim CAS readback (F0 -> F1)."""

    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    deployment_sha = deployment["deployment_contract_sha256"]
    selected = deployment.get("selected_job_ids")
    if (
        not isinstance(external_job_id, str)
        or not isinstance(selected, list)
        or selected.count(external_job_id) != 1
    ):
        raise ValueError("claim readback external job changed")
    position = selected.index(external_job_id)
    instance = deployment["instances"][position]
    checked_authorization = dict(authorization)
    checked_claim = dict(claim)
    validated_external = external_auth.validate_external_approval(
        authorization=checked_authorization,
        claim=checked_claim,
        deployment_contract=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
        package_generations=package_generations,
        external_preflight_receipt=external_preflight_receipt,
        observed_project_number=external_auth.EXPECTED_PROJECT_NUMBER,
        observed_provider_instance_id=provider_instance_id,
        observed_metadata_fingerprint=(
            controller_pre_cas_metadata_fingerprint
        ),
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )
    authorization_sha = canonical_sha256(checked_authorization)
    claim_sha = canonical_sha256(checked_claim)
    authorization_nonce_digest = _nonce_digest(
        checked_authorization.get("nonce"),
        "authorization nonce",
    )
    claim_nonce_digest = _nonce_digest(
        checked_claim.get("nonce"),
        "claim nonce",
    )
    run_nonce_digest = _nonce_digest(
        deployment["run_nonce"], "run nonce"
    )
    pre_fingerprint = _fingerprint(
        controller_pre_cas_metadata_fingerprint,
        "controller pre-CAS metadata fingerprint",
    )
    post_fingerprint = _fingerprint(
        controller_post_cas_metadata_fingerprint,
        "controller post-CAS metadata fingerprint",
    )
    if (
        post_fingerprint == pre_fingerprint
        or len(
            {
                run_nonce_digest,
                authorization_nonce_digest,
                claim_nonce_digest,
            }
        )
        != 3
        or validated_external.authorization_sha256
        != authorization_sha
        or validated_external.claim_sha256 != claim_sha
        or validated_external.external_job_id != external_job_id
        or validated_external.provider_instance_id
        != provider_instance_id
        or checked_claim.get("deployment_contract_sha256")
        != deployment_sha
        or checked_claim.get("authorization_sha256")
        != authorization_sha
        or checked_claim.get("external_job_id") != external_job_id
        or checked_claim.get("inner_job_id") != instance["inner_job_id"]
        or checked_claim.get("source_role") != instance["source_role"]
        or checked_claim.get("instance_name") != instance["instance_name"]
        or checked_claim.get("provider_instance_id")
        != provider_instance_id
        or checked_claim.get("metadata_fingerprint") != pre_fingerprint
        or checked_claim.get("metadata_fingerprint_semantics")
        != "pre_claim_cas_observation"
        or checked_claim.get("attempt_index") != 0
        or checked_claim.get("max_attempts") != 1
    ):
        raise ValueError("claim CAS readback binding changed")
    return _sealed(
        {
            "schema": CLAIM_CAS_READBACK_SCHEMA,
            "deployment_contract_sha256": deployment_sha,
            "external_job_id": external_job_id,
            "inner_job_id": instance["inner_job_id"],
            "source_role": instance["source_role"],
            "instance_name": instance["instance_name"],
            "provider_instance_id": _decimal(
                provider_instance_id, "provider instance ID"
            ),
            "authorization_sha256": authorization_sha,
            "claim_sha256": claim_sha,
            "claim_metadata_value_sha256": canonical_sha256(
                checked_claim
            ),
            "claimed_metadata_sha256": _sha(
                claimed_metadata_sha256, "claimed metadata"
            ),
            "controller_pre_cas_metadata_fingerprint": pre_fingerprint,
            "controller_post_cas_metadata_fingerprint": post_fingerprint,
            "run_nonce_digest": run_nonce_digest,
            "authorization_nonce_digest": authorization_nonce_digest,
            "claim_nonce_digest": claim_nonce_digest,
            "run_authorization_claim_nonces_unique": True,
            "claim_present_exactly_once": True,
            "claim_cas_readback_complete": True,
            "worker_observed_metadata_fingerprint": False,
        }
    )


def validate_claim_cas_readback_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _validate_sealed(value, label="claim CAS readback receipt")
    _exact(
        receipt,
        {
            "schema",
            "deployment_contract_sha256",
            "external_job_id",
            "inner_job_id",
            "source_role",
            "instance_name",
            "provider_instance_id",
            "authorization_sha256",
            "claim_sha256",
            "claim_metadata_value_sha256",
            "claimed_metadata_sha256",
            "controller_pre_cas_metadata_fingerprint",
            "controller_post_cas_metadata_fingerprint",
            "run_nonce_digest",
            "authorization_nonce_digest",
            "claim_nonce_digest",
            "run_authorization_claim_nonces_unique",
            "claim_present_exactly_once",
            "claim_cas_readback_complete",
            "worker_observed_metadata_fingerprint",
            "receipt_sha256",
        },
        "claim CAS readback receipt",
    )
    deployment = dict(deployment_contract)
    selected = deployment["selected_job_ids"]
    job_id = receipt["external_job_id"]
    if job_id not in selected:
        raise ValueError("claim readback escaped pair")
    position = selected.index(job_id)
    instance = deployment["instances"][position]
    if (
        receipt["schema"] != CLAIM_CAS_READBACK_SCHEMA
        or receipt["deployment_contract_sha256"]
        != deployment["deployment_contract_sha256"]
        or receipt["inner_job_id"] != instance["inner_job_id"]
        or receipt["source_role"] != instance["source_role"]
        or receipt["instance_name"] != instance["instance_name"]
        or receipt["claim_metadata_value_sha256"]
        != receipt["claim_sha256"]
        or len(
            {
                receipt["run_nonce_digest"],
                receipt["authorization_nonce_digest"],
                receipt["claim_nonce_digest"],
            }
        )
        != 3
        or receipt["run_nonce_digest"]
        != _nonce_digest(deployment["run_nonce"], "run nonce")
        or receipt["run_authorization_claim_nonces_unique"] is not True
        or receipt["controller_pre_cas_metadata_fingerprint"]
        == receipt["controller_post_cas_metadata_fingerprint"]
        or receipt["claim_present_exactly_once"] is not True
        or receipt["claim_cas_readback_complete"] is not True
        or receipt["worker_observed_metadata_fingerprint"] is not False
    ):
        raise ValueError("claim CAS readback receipt changed")
    _decimal(receipt["provider_instance_id"], "provider instance ID")
    for field in (
        "authorization_sha256",
        "claim_sha256",
        "claimed_metadata_sha256",
        "run_nonce_digest",
        "authorization_nonce_digest",
        "claim_nonce_digest",
        "receipt_sha256",
    ):
        _sha(receipt[field], field)
    _fingerprint(
        receipt["controller_pre_cas_metadata_fingerprint"],
        "controller pre-CAS metadata fingerprint",
    )
    _fingerprint(
        receipt["controller_post_cas_metadata_fingerprint"],
        "controller post-CAS metadata fingerprint",
    )
    return receipt


def build_phase2_zero_receipt(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    phase2_iam_plan: Mapping[str, Any],
    binding_readbacks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal plan-derived controller-principal absence readbacks."""

    deployment = dict(deployment_contract)
    controller = deployment["controller_service_account"]
    plan = phase2_iam.validate_step12b_phase2_iam_plan(
        phase2_iam_plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    removal_group = plan["controller_release_removal_group"]
    expected_identities = removal_group["binding_identities"]
    expected_count = removal_group["binding_count"]
    if (
        isinstance(binding_readbacks, (str, bytes))
        or not isinstance(binding_readbacks, Sequence)
        or len(binding_readbacks) != expected_count
    ):
        raise ValueError("phase2 binding readback count changed")
    checked = []
    for raw in binding_readbacks:
        row = dict(raw)
        _exact(
            row,
            {
                "purpose",
                "target",
                "resource",
                "role",
                "principal",
                "binding_sha256",
                "controller_member_count",
                "controller_member_present",
                "readback_complete",
            },
            "phase2 binding readback",
        )
        if (
            row["principal"] != controller["principal"]
            or row["controller_member_count"] != 0
            or row["controller_member_present"] is not False
            or row["readback_complete"] is not True
        ):
            raise ValueError("phase2 binding readback changed")
        checked.append(row)
    checked.sort(
        key=lambda row: (
            row["resource"],
            row["role"],
            row["purpose"],
        )
    )
    observed_identities = [
        {
            key: row[key]
            for key in (
                "purpose",
                "target",
                "resource",
                "role",
                "principal",
                "binding_sha256",
            )
        }
        for row in checked
    ]
    if (
        observed_identities != expected_identities
        or canonical_sha256(observed_identities)
        != removal_group["binding_identities_sha256"]
    ):
        raise ValueError("phase2 binding readback identity changed")
    return _sealed(
        {
            "schema": PHASE2_ZERO_SCHEMA,
            "deployment_contract_sha256": deployment[
                "deployment_contract_sha256"
            ],
            "controller_service_account": dict(controller),
            "phase2_iam_plan_sha256": plan["plan_sha256"],
            "expected_phase2_binding_count": expected_count,
            "controller_binding_identities": expected_identities,
            "controller_binding_identities_sha256": removal_group[
                "binding_identities_sha256"
            ],
            "binding_readbacks": checked,
            "binding_readbacks_sha256": canonical_sha256(checked),
            "controller_binding_count_after_claims": 0,
            "all_phase2_controller_bindings_zero": True,
        }
    )


def validate_phase2_zero_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    phase2_iam_plan: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _validate_sealed(value, label="phase2 zero receipt")
    _exact(
        receipt,
        {
            "schema",
            "deployment_contract_sha256",
            "controller_service_account",
            "phase2_iam_plan_sha256",
            "expected_phase2_binding_count",
            "controller_binding_identities",
            "controller_binding_identities_sha256",
            "binding_readbacks",
            "binding_readbacks_sha256",
            "controller_binding_count_after_claims",
            "all_phase2_controller_bindings_zero",
            "receipt_sha256",
        },
        "phase2 zero receipt",
    )
    rebuilt = build_phase2_zero_receipt(
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        phase2_iam_plan=phase2_iam_plan,
        binding_readbacks=receipt["binding_readbacks"],
    )
    if receipt != rebuilt:
        raise ValueError("phase2 zero receipt changed")
    return receipt


def validate_controller_create_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _validate_sealed(
        value, label="run-scoped controller create receipt"
    )
    controller = deployment_contract["controller_service_account"]
    _exact(
        receipt,
        {
            "schema",
            "status",
            "project",
            "account_id",
            "email",
            "provider",
            "create_call_count",
            "create_retry_count",
            "readback_verified",
            "cloud_mutation_performed",
            "receipt_sha256",
        },
        "run-scoped controller create receipt",
    )
    provider = receipt["provider"]
    if not isinstance(provider, Mapping):
        raise ValueError("run-scoped controller create provider changed")
    _exact(
        provider,
        {"name", "project_id", "unique_id", "email", "disabled"},
        "run-scoped controller create provider",
    )
    expected_name = (
        f"projects/{controller['project']}/serviceAccounts/"
        f"{controller['email']}"
    )
    if (
        receipt["schema"] != controller_sa.SCHEMA
        or receipt["status"]
        != "run_scoped_controller_service_account_created"
        or receipt["project"] != controller["project"]
        or receipt["account_id"] != controller["account_id"]
        or receipt["email"] != controller["email"]
        or provider["name"] != expected_name
        or provider["project_id"] != controller["project"]
        or provider["email"] != controller["email"]
        or provider["disabled"] is not False
        or receipt["create_call_count"] != 1
        or receipt["create_retry_count"] != 0
        or receipt["readback_verified"] is not True
        or receipt["cloud_mutation_performed"] is not True
    ):
        raise ValueError("run-scoped controller create receipt changed")
    _decimal(provider["unique_id"], "controller unique ID")
    return receipt


def validate_controller_delete_receipt(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    controller_create_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    created = validate_controller_create_receipt(
        controller_create_receipt,
        deployment_contract=deployment_contract,
    )
    receipt = _validate_sealed(
        value, label="run-scoped controller delete receipt"
    )
    controller = deployment_contract["controller_service_account"]
    _exact(
        receipt,
        {
            "schema",
            "status",
            "project",
            "account_id",
            "email",
            "provider_unique_id",
            "controller_create_receipt_sha256",
            "delete_call_count",
            "delete_retry_count",
            "absence_get_count",
            "final_get_status",
            "readback_verified",
            "success_path_teardown_evidence",
            "cloud_mutation_performed",
            "receipt_sha256",
        },
        "run-scoped controller delete receipt",
    )
    if (
        receipt["schema"] != controller_sa.SCHEMA
        or receipt["status"]
        != "run_scoped_controller_service_account_deleted_and_absent"
        or receipt["project"] != controller["project"]
        or receipt["account_id"] != controller["account_id"]
        or receipt["email"] != controller["email"]
        or receipt["provider_unique_id"]
        != created["provider"]["unique_id"]
        or receipt["controller_create_receipt_sha256"]
        != created["receipt_sha256"]
        or receipt["delete_call_count"] != 1
        or receipt["delete_retry_count"] != 0
        or type(receipt["absence_get_count"]) is not int
        or receipt["absence_get_count"] < 1
        or receipt["final_get_status"] != 404
        or receipt["readback_verified"] is not True
        or receipt["success_path_teardown_evidence"] is not True
        or receipt["cloud_mutation_performed"] is not True
    ):
        raise ValueError("run-scoped controller delete receipt changed")
    _decimal(receipt["provider_unique_id"], "controller unique ID")
    return receipt


def _require_signer(
    signer: external_auth.ControllerSigner,
    public_record: Mapping[str, Any],
) -> None:
    checked = payload_transport.validate_rsa_public_key_record(
        signer.public_record
    )
    if checked != dict(public_record):
        raise ValueError("pair release signer changed")


def _require_verifier(
    verifier: external_auth.ControllerVerifier,
    public_record: Mapping[str, Any],
) -> None:
    if (
        getattr(verifier, "key_id", None) != public_record["key_id"]
        or getattr(verifier, "public_key_sha256", None)
        != canonical_sha256(public_record)
    ):
        raise ValueError("pair release verifier changed")


def _release_unsigned(
    *,
    deployment: Mapping[str, Any],
    public_record: Mapping[str, Any],
    claim_readbacks: Sequence[Mapping[str, Any]],
    phase2_zero_receipt: Mapping[str, Any],
    controller_create_receipt: Mapping[str, Any],
    controller_delete_receipt: Mapping[str, Any],
    issued_unix_seconds: int,
    nonce: str,
) -> dict[str, Any]:
    summaries = [
        {
            "external_job_id": row["external_job_id"],
            "inner_job_id": row["inner_job_id"],
            "source_role": row["source_role"],
            "instance_name": row["instance_name"],
            "provider_instance_id": row["provider_instance_id"],
            "authorization_sha256": row["authorization_sha256"],
            "claim_sha256": row["claim_sha256"],
            "authorization_nonce_digest": row[
                "authorization_nonce_digest"
            ],
            "claim_nonce_digest": row["claim_nonce_digest"],
            "controller_pre_cas_metadata_fingerprint": row[
                "controller_pre_cas_metadata_fingerprint"
            ],
            "controller_post_cas_metadata_fingerprint": row[
                "controller_post_cas_metadata_fingerprint"
            ],
            "claim_cas_readback_receipt_sha256": row["receipt_sha256"],
        }
        for row in claim_readbacks
    ]
    return {
        "schema": PAIR_RELEASE_SCHEMA,
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
        "controller_key_id": public_record["key_id"],
        "controller_public_key_sha256": canonical_sha256(public_record),
        "signature_record_type": PAIR_RELEASE_SIGNATURE_RECORD_TYPE,
        "controller_service_account": dict(
            deployment["controller_service_account"]
        ),
        "selected_job_ids": list(deployment["selected_job_ids"]),
        "source_roles": list(deployment["source_roles"]),
        "claim_readbacks": summaries,
        "claim_sha256s": [row["claim_sha256"] for row in summaries],
        "authorization_nonce_digests": [
            row["authorization_nonce_digest"] for row in summaries
        ],
        "claim_nonce_digests": [
            row["claim_nonce_digest"] for row in summaries
        ],
        "run_nonce_digest": _nonce_digest(
            deployment["run_nonce"], "run nonce"
        ),
        "release_nonce_digest": _nonce_digest(
            nonce, "pair release nonce"
        ),
        "all_six_nonces_unique": True,
        "claim_cas_readback_receipt_sha256s": [
            row["claim_cas_readback_receipt_sha256"]
            for row in summaries
        ],
        "both_claim_cas_readbacks_complete": True,
        "phase2_zero_receipt_sha256": phase2_zero_receipt[
            "receipt_sha256"
        ],
        "phase2_iam_plan_sha256": phase2_zero_receipt[
            "phase2_iam_plan_sha256"
        ],
        "phase2_controller_binding_identities_sha256": (
            phase2_zero_receipt[
                "controller_binding_identities_sha256"
            ]
        ),
        "phase2_expected_controller_binding_count": (
            phase2_zero_receipt["expected_phase2_binding_count"]
        ),
        "phase2_controller_binding_count": 0,
        "all_phase2_controller_bindings_zero": True,
        "controller_create_receipt_sha256": controller_create_receipt[
            "receipt_sha256"
        ],
        "controller_provider_unique_id": controller_create_receipt[
            "provider"
        ]["unique_id"],
        "controller_delete_receipt_sha256": controller_delete_receipt[
            "receipt_sha256"
        ],
        "controller_service_account_deleted": True,
        "controller_service_account_final_get_status": 404,
        "controller_credential_reuse_permitted": False,
        "attempt_index": 0,
        "max_attempts": 1,
        "issued_unix_seconds": issued_unix_seconds,
        "nonce": nonce,
    }


def build_pair_release(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    claim_cas_readback_receipts: Sequence[Mapping[str, Any]],
    phase2_iam_plan: Mapping[str, Any],
    phase2_zero_receipt: Mapping[str, Any],
    controller_create_receipt: Mapping[str, Any],
    controller_delete_receipt: Mapping[str, Any],
    issued_unix_seconds: int,
    signer: external_auth.ControllerSigner,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Build the signed pair-wide release after all teardown gates."""

    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    _require_signer(signer, public_record)
    if (
        isinstance(claim_cas_readback_receipts, (str, bytes))
        or not isinstance(claim_cas_readback_receipts, Sequence)
        or len(claim_cas_readback_receipts) != 2
    ):
        raise ValueError("pair claim CAS readback count changed")
    by_job = {
        row["external_job_id"]: validate_claim_cas_readback_receipt(
            row, deployment_contract=deployment
        )
        for row in claim_cas_readback_receipts
    }
    if set(by_job) != set(deployment["selected_job_ids"]):
        raise ValueError("pair claim CAS readback coverage changed")
    ordered = [by_job[job] for job in deployment["selected_job_ids"]]
    if (
        len({row["claim_sha256"] for row in ordered}) != 2
        or len({row["provider_instance_id"] for row in ordered}) != 2
        or len({row["receipt_sha256"] for row in ordered}) != 2
    ):
        raise ValueError("pair claim CAS identities collided")
    phase2 = validate_phase2_zero_receipt(
        phase2_zero_receipt,
        deployment_contract=deployment,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=public_record,
        run_nonce=run_nonce,
        phase2_iam_plan=phase2_iam_plan,
    )
    created = validate_controller_create_receipt(
        controller_create_receipt,
        deployment_contract=deployment,
    )
    deleted = validate_controller_delete_receipt(
        controller_delete_receipt,
        deployment_contract=deployment,
        controller_create_receipt=created,
    )
    if type(issued_unix_seconds) is not int or issued_unix_seconds <= 0:
        raise ValueError("pair release issued time changed")
    release_nonce = (
        secrets.token_hex(32)
        if nonce is None
        else _sha(nonce, "pair release nonce")
    )
    signed_nonce_digests = {
        _nonce_digest(deployment["run_nonce"], "run nonce"),
        *(
            row["authorization_nonce_digest"]
            for row in ordered
        ),
        *(row["claim_nonce_digest"] for row in ordered),
    }
    if len(signed_nonce_digests) != 5:
        raise ValueError("run/authorization/claim nonce collision")
    if _nonce_digest(
        release_nonce, "pair release nonce"
    ) in signed_nonce_digests:
        raise ValueError("pair release nonce reused signed nonce")
    unsigned = _release_unsigned(
        deployment=deployment,
        public_record=public_record,
        claim_readbacks=ordered,
        phase2_zero_receipt=phase2,
        controller_create_receipt=created,
        controller_delete_receipt=deleted,
        issued_unix_seconds=issued_unix_seconds,
        nonce=release_nonce,
    )
    release = {
        **unsigned,
        "signature": signer.sign(
            record_type=PAIR_RELEASE_SIGNATURE_RECORD_TYPE,
            unsigned=unsigned,
        ),
    }
    signature = release["signature"]
    if (
        not isinstance(signature, str)
        or not signature
        or len(signature) > 8_192
        or _SIGNATURE.fullmatch(signature) is None
    ):
        raise ValueError("pair release signature changed")
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        public_record
    )
    if not verifier.verify(
        record_type=PAIR_RELEASE_SIGNATURE_RECORD_TYPE,
        payload=canonical_bytes(unsigned),
        signature=signature,
    ):
        raise ValueError("new pair release signature did not verify")
    if len(canonical_bytes(release)) > vm_metadata.MAX_PAIR_RELEASE_VALUE_BYTES:
        raise ValueError("pair release escaped reserved metadata value")
    return release


def _validate_pair_release_bound(
    release: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
    public_record: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
) -> dict[str, Any]:
    _require_verifier(verifier, public_record)
    checked = dict(release)
    _exact(
        checked,
        {
            "schema",
            "deployment_contract_sha256",
            "run_identity_sha256",
            "direct_stage_identity_sha256",
            "payload_binding_sha256",
            "controller_key_id",
            "controller_public_key_sha256",
            "signature_record_type",
            "controller_service_account",
            "selected_job_ids",
            "source_roles",
            "claim_readbacks",
            "claim_sha256s",
            "authorization_nonce_digests",
            "claim_nonce_digests",
            "run_nonce_digest",
            "release_nonce_digest",
            "all_six_nonces_unique",
            "claim_cas_readback_receipt_sha256s",
            "both_claim_cas_readbacks_complete",
            "phase2_zero_receipt_sha256",
            "phase2_iam_plan_sha256",
            "phase2_controller_binding_identities_sha256",
            "phase2_expected_controller_binding_count",
            "phase2_controller_binding_count",
            "all_phase2_controller_bindings_zero",
            "controller_create_receipt_sha256",
            "controller_provider_unique_id",
            "controller_delete_receipt_sha256",
            "controller_service_account_deleted",
            "controller_service_account_final_get_status",
            "controller_credential_reuse_permitted",
            "attempt_index",
            "max_attempts",
            "issued_unix_seconds",
            "nonce",
            "signature",
        },
        "Step12b pair release",
    )
    readbacks = checked["claim_readbacks"]
    authorization_nonce_digests = checked[
        "authorization_nonce_digests"
    ]
    claim_nonce_digests = checked["claim_nonce_digests"]
    if (
        not isinstance(authorization_nonce_digests, list)
        or len(authorization_nonce_digests) != 2
        or not isinstance(claim_nonce_digests, list)
        or len(claim_nonce_digests) != 2
    ):
        raise ValueError("Step12b pair release nonce set changed")
    if (
        checked["schema"] != PAIR_RELEASE_SCHEMA
        or checked["deployment_contract_sha256"]
        != deployment["deployment_contract_sha256"]
        or checked["run_identity_sha256"]
        != deployment["run_identity_sha256"]
        or checked["direct_stage_identity_sha256"]
        != deployment["direct_stage_identity_sha256"]
        or checked["payload_binding_sha256"]
        != deployment["payload_binding"]["payload_binding_sha256"]
        or checked["controller_key_id"] != public_record["key_id"]
        or checked["controller_public_key_sha256"]
        != canonical_sha256(public_record)
        or checked["signature_record_type"]
        != PAIR_RELEASE_SIGNATURE_RECORD_TYPE
        or checked["controller_service_account"]
        != deployment["controller_service_account"]
        or checked["selected_job_ids"]
        != deployment["selected_job_ids"]
        or checked["source_roles"] != deployment["source_roles"]
        or not isinstance(readbacks, list)
        or len(readbacks) != 2
        or [row.get("external_job_id") for row in readbacks]
        != deployment["selected_job_ids"]
        or [row.get("source_role") for row in readbacks]
        != deployment["source_roles"]
        or checked["claim_sha256s"]
        != [row.get("claim_sha256") for row in readbacks]
        or checked["authorization_nonce_digests"]
        != [
            row.get("authorization_nonce_digest")
            for row in readbacks
        ]
        or checked["claim_nonce_digests"]
        != [row.get("claim_nonce_digest") for row in readbacks]
        or checked["run_nonce_digest"]
        != _nonce_digest(deployment["run_nonce"], "run nonce")
        or checked["release_nonce_digest"]
        != _nonce_digest(checked["nonce"], "pair release nonce")
        or len(
            {
                checked["run_nonce_digest"],
                *authorization_nonce_digests,
                *claim_nonce_digests,
                checked["release_nonce_digest"],
            }
        )
        != 6
        or checked["all_six_nonces_unique"] is not True
        or checked["claim_cas_readback_receipt_sha256s"]
        != [
            row.get("claim_cas_readback_receipt_sha256")
            for row in readbacks
        ]
        or len(set(checked["claim_sha256s"])) != 2
        or len(
            {
                row.get("provider_instance_id")
                for row in readbacks
            }
        )
        != 2
        or checked["both_claim_cas_readbacks_complete"] is not True
        or checked["phase2_expected_controller_binding_count"]
        != phase2_iam.CONTROLLER_BINDING_COUNT
        or checked["phase2_controller_binding_count"] != 0
        or checked["all_phase2_controller_bindings_zero"] is not True
        or checked["controller_service_account_deleted"] is not True
        or checked["controller_service_account_final_get_status"] != 404
        or checked["controller_credential_reuse_permitted"] is not False
        or checked["attempt_index"] != 0
        or checked["max_attempts"] != 1
        or type(checked["issued_unix_seconds"]) is not int
        or checked["issued_unix_seconds"] <= 0
    ):
        raise ValueError("Step12b pair release identity changed")
    for position, row in enumerate(readbacks):
        _exact(
            row,
            {
                "external_job_id",
                "inner_job_id",
                "source_role",
                "instance_name",
                "provider_instance_id",
                "authorization_sha256",
                "claim_sha256",
                "authorization_nonce_digest",
                "claim_nonce_digest",
                "controller_pre_cas_metadata_fingerprint",
                "controller_post_cas_metadata_fingerprint",
                "claim_cas_readback_receipt_sha256",
            },
            "pair release claim readback",
        )
        instance = deployment["instances"][position]
        if (
            row["inner_job_id"] != instance["inner_job_id"]
            or row["instance_name"] != instance["instance_name"]
            or row["controller_pre_cas_metadata_fingerprint"]
            == row["controller_post_cas_metadata_fingerprint"]
        ):
            raise ValueError("pair release claim mapping changed")
        _decimal(row["provider_instance_id"], "provider instance ID")
        _fingerprint(
            row["controller_pre_cas_metadata_fingerprint"],
            "controller pre-CAS metadata fingerprint",
        )
        _fingerprint(
            row["controller_post_cas_metadata_fingerprint"],
            "controller post-CAS metadata fingerprint",
        )
        _sha(row["claim_sha256"], "claim")
        _sha(row["authorization_sha256"], "authorization")
        _sha(
            row["authorization_nonce_digest"],
            "authorization nonce digest",
        )
        _sha(row["claim_nonce_digest"], "claim nonce digest")
        _sha(
            row["claim_cas_readback_receipt_sha256"],
            "claim CAS readback receipt",
        )
    _sha(checked["phase2_zero_receipt_sha256"], "phase2 zero receipt")
    _sha(checked["phase2_iam_plan_sha256"], "phase2 IAM plan")
    _sha(
        checked["phase2_controller_binding_identities_sha256"],
        "phase2 controller binding identities",
    )
    _sha(
        checked["controller_create_receipt_sha256"],
        "controller create receipt",
    )
    _decimal(
        checked["controller_provider_unique_id"],
        "controller provider unique ID",
    )
    _sha(
        checked["controller_delete_receipt_sha256"],
        "controller delete receipt",
    )
    _sha(checked["nonce"], "pair release nonce")
    signature = checked["signature"]
    if (
        not isinstance(signature, str)
        or not signature
        or len(signature) > 8_192
        or _SIGNATURE.fullmatch(signature) is None
    ):
        raise ValueError("pair release signature changed")
    unsigned = {
        key: value for key, value in checked.items() if key != "signature"
    }
    if not verifier.verify(
        record_type=PAIR_RELEASE_SIGNATURE_RECORD_TYPE,
        payload=canonical_bytes(unsigned),
        signature=signature,
    ):
        raise ValueError("pair release trust verification failed")
    if len(canonical_bytes(checked)) > vm_metadata.MAX_PAIR_RELEASE_VALUE_BYTES:
        raise ValueError("pair release escaped reserved metadata value")
    return checked


def validate_role_runtime_approval(
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
    observed_identity: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
    now_unix_seconds: int,
) -> ValidatedRoleRuntimeApproval:
    """Mint an opaque approval from signed claim plus guest-visible identity."""

    role_deployment = deployment_v2.validate_role_runtime_view(
        deployment_contract,
        selected_payload_contract=selected_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
    )
    vm_identity = vm_metadata.validate_vm_identity_after_claim(
        role_runtime_deployment=role_deployment,
        external_job_id=external_job_id,
        observed_identity=observed_identity,
    )
    external = external_auth.validate_external_approval_role_runtime(
        authorization=authorization,
        claim=claim,
        deployment_contract=role_deployment,
        selected_payload_contract=selected_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
        package_generations=package_generations,
        external_preflight_receipt=external_preflight_receipt,
        observed_project=vm_identity.project_id,
        observed_project_number=vm_identity.project_number,
        observed_zone=vm_identity.zone,
        observed_instance_name=vm_identity.instance_name,
        observed_provider_instance_id=(
            vm_identity.provider_instance_id
        ),
        observed_worker_service_account=(
            vm_identity.worker_service_account
        ),
        observed_oauth_scopes=[vm_identity.oauth_scope],
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )
    vm_metadata.require_verified_vm_identity(
        vm_identity,
        deployment_contract_sha256=external.deployment_contract_sha256,
        external_job_id=external.external_job_id,
    )
    if (
        external.inner_job_id != vm_identity.inner_job_id
        or external.source_role != vm_identity.source_role
        or external.instance_name != vm_identity.instance_name
        or external.provider_instance_id
        != vm_identity.provider_instance_id
    ):
        raise ValueError("signed approval and guest VM identity diverged")
    vm_receipt = vm_identity.receipt()
    body = {
        "schema": ROLE_RUNTIME_APPROVAL_SCHEMA,
        "deployment_contract_sha256": (
            external.deployment_contract_sha256
        ),
        "authorization_sha256": external.authorization_sha256,
        "claim_sha256": external.claim_sha256,
        "external_job_id": external.external_job_id,
        "inner_job_id": external.inner_job_id,
        "source_role": external.source_role,
        "instance_name": external.instance_name,
        "provider_instance_id": external.provider_instance_id,
        "controller_preclaim_metadata_fingerprint": (
            external.metadata_fingerprint
        ),
        "package_generations_sha256": canonical_sha256(
            dict(external.package_generations)
        ),
        "vm_identity_receipt_sha256": vm_receipt[
            "vm_identity_receipt_sha256"
        ],
    }
    return ValidatedRoleRuntimeApproval(
        deployment_contract_sha256=body[
            "deployment_contract_sha256"
        ],
        authorization_sha256=body["authorization_sha256"],
        claim_sha256=body["claim_sha256"],
        external_job_id=body["external_job_id"],
        inner_job_id=body["inner_job_id"],
        source_role=body["source_role"],
        instance_name=body["instance_name"],
        provider_instance_id=body["provider_instance_id"],
        controller_preclaim_metadata_fingerprint=body[
            "controller_preclaim_metadata_fingerprint"
        ],
        package_generations_sha256=body[
            "package_generations_sha256"
        ],
        vm_identity_receipt_sha256=body[
            "vm_identity_receipt_sha256"
        ],
        receipt_sha256=canonical_sha256(body),
        _seal=_ROLE_APPROVAL_SEAL,
    )


def require_role_runtime_approval(
    value: ValidatedRoleRuntimeApproval,
    *,
    deployment_contract_sha256: str,
    external_job_id: str,
) -> ValidatedRoleRuntimeApproval:
    if (
        not isinstance(value, ValidatedRoleRuntimeApproval)
        or value._seal is not _ROLE_APPROVAL_SEAL
        or value.deployment_contract_sha256
        != deployment_contract_sha256
        or value.external_job_id != external_job_id
    ):
        raise ValueError("role runtime approval capability changed")
    return value


def validate_pair_release_role_runtime(
    release: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    role_approval: ValidatedRoleRuntimeApproval,
    verifier: external_auth.ControllerVerifier,
) -> PairReleaseApproval:
    deployment = deployment_v2.validate_role_runtime_view(
        deployment_contract,
        selected_payload_contract=selected_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
    )
    role = require_role_runtime_approval(
        role_approval,
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        external_job_id=external_job_id,
    )
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    checked = _validate_pair_release_bound(
        release,
        deployment=deployment,
        public_record=public_record,
        verifier=verifier,
    )
    matching = [
        row
        for row in checked["claim_readbacks"]
        if row["external_job_id"] == external_job_id
    ]
    if (
        len(matching) != 1
        or matching[0]["claim_sha256"] != role.claim_sha256
        or matching[0]["inner_job_id"] != role.inner_job_id
        or matching[0]["source_role"] != role.source_role
        or matching[0]["instance_name"] != role.instance_name
        or matching[0]["provider_instance_id"]
        != role.provider_instance_id
        or matching[0]["controller_pre_cas_metadata_fingerprint"]
        != role.controller_preclaim_metadata_fingerprint
    ):
        raise ValueError("pair release does not contain own signed claim")
    release_sha = canonical_sha256(checked)
    body = {
        "schema": PAIR_RELEASE_APPROVAL_SCHEMA,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "pair_release_sha256": release_sha,
        "external_job_id": external_job_id,
        "own_claim_sha256": role.claim_sha256,
        "phase2_iam_plan_sha256": checked[
            "phase2_iam_plan_sha256"
        ],
        "phase2_zero_receipt_sha256": checked[
            "phase2_zero_receipt_sha256"
        ],
        "controller_create_receipt_sha256": checked[
            "controller_create_receipt_sha256"
        ],
        "controller_delete_receipt_sha256": checked[
            "controller_delete_receipt_sha256"
        ],
    }
    return PairReleaseApproval(
        deployment_contract_sha256=body[
            "deployment_contract_sha256"
        ],
        pair_release_sha256=release_sha,
        external_job_id=external_job_id,
        own_claim_sha256=role.claim_sha256,
        phase2_iam_plan_sha256=body[
            "phase2_iam_plan_sha256"
        ],
        phase2_zero_receipt_sha256=body[
            "phase2_zero_receipt_sha256"
        ],
        controller_create_receipt_sha256=body[
            "controller_create_receipt_sha256"
        ],
        controller_delete_receipt_sha256=body[
            "controller_delete_receipt_sha256"
        ],
        receipt_sha256=canonical_sha256(body),
        _seal=_PAIR_RELEASE_SEAL,
    )


def require_pair_release_approval(
    value: PairReleaseApproval,
    *,
    deployment_contract_sha256: str,
    external_job_id: str,
    role_approval: ValidatedRoleRuntimeApproval,
) -> PairReleaseApproval:
    role = require_role_runtime_approval(
        role_approval,
        deployment_contract_sha256=deployment_contract_sha256,
        external_job_id=external_job_id,
    )
    if (
        not isinstance(value, PairReleaseApproval)
        or value._seal is not _PAIR_RELEASE_SEAL
        or value.deployment_contract_sha256
        != deployment_contract_sha256
        or value.external_job_id != external_job_id
        or value.own_claim_sha256 != role.claim_sha256
    ):
        raise ValueError("pair release approval capability changed")
    return value


def wait_for_pair_release(
    *,
    reader: PairReleaseMetadataReader,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    role_approval: ValidatedRoleRuntimeApproval,
    verifier: external_auth.ControllerVerifier,
    timeout_seconds: int = MAX_RELEASE_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_RELEASE_POLL_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PairReleaseApproval:
    """Poll boundedly; no payload work occurs before a valid release."""

    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= MAX_RELEASE_WAIT_SECONDS
        or not math.isfinite(poll_seconds)
        or not 0 < poll_seconds <= 10
    ):
        raise ValueError("pair release wait bound changed")
    started = float(now())
    while True:
        raw = reader.read_pair_release()
        if raw is not None:
            if (
                not isinstance(raw, str)
                or not raw
                or len(raw.encode("utf-8"))
                > vm_metadata.MAX_PAIR_RELEASE_VALUE_BYTES
            ):
                raise ValueError("pair release metadata value changed")
            encoded = raw.encode("ascii")
            try:
                release = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("pair release metadata is not JSON") from error
            if (
                not isinstance(release, dict)
                or canonical_bytes(release) != encoded
            ):
                raise ValueError(
                    "pair release metadata is not canonical JSON"
                )
            return validate_pair_release_role_runtime(
                release,
                deployment_contract=deployment_contract,
                selected_payload_contract=selected_payload_contract,
                controller_public_key_record=controller_public_key_record,
                run_nonce=run_nonce,
                external_job_id=external_job_id,
                role_approval=role_approval,
                verifier=verifier,
            )
        elapsed = float(now()) - started
        if elapsed >= timeout_seconds:
            raise TimeoutError("pair release metadata wait timed out")
        sleep(min(poll_seconds, timeout_seconds - elapsed))


__all__ = [
    "CLAIM_CAS_READBACK_SCHEMA",
    "DEFAULT_RELEASE_POLL_SECONDS",
    "MAX_RELEASE_WAIT_SECONDS",
    "PAIR_RELEASE_APPROVAL_SCHEMA",
    "PAIR_RELEASE_SCHEMA",
    "PAIR_RELEASE_SIGNATURE_RECORD_TYPE",
    "PHASE2_ZERO_SCHEMA",
    "PairReleaseApproval",
    "PairReleaseMetadataReader",
    "ROLE_RUNTIME_APPROVAL_SCHEMA",
    "ValidatedRoleRuntimeApproval",
    "build_claim_cas_readback_receipt",
    "build_pair_release",
    "build_phase2_zero_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "require_pair_release_approval",
    "require_role_runtime_approval",
    "validate_claim_cas_readback_receipt",
    "validate_controller_create_receipt",
    "validate_controller_delete_receipt",
    "validate_pair_release_role_runtime",
    "validate_phase2_zero_receipt",
    "validate_role_runtime_approval",
    "wait_for_pair_release",
]
