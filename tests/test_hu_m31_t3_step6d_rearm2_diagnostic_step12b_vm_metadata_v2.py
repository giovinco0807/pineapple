from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2
    as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP12_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12_pair_v1_actual"
)
RUN_NONCE = "c3" * 32


def _manifest(
    role_runtime: dict[str, Any], external_job_id: str
) -> dict[str, Any]:
    position = role_runtime["selected_job_ids"].index(external_job_id)
    role = role_runtime["source_roles"][position]
    deployment_sha = role_runtime["deployment_contract_sha256"]
    source_prefix = (
        "gs://pokerhu-ofc-solver-485418-training/"
        "hu-m31-r2diag-direct-v2/bootstrap-sources/"
        f"{deployment_sha}"
    )
    role_path = f"{role}_payload_contract.json"
    objects = [
        {
            "kind": "shared_runtime_source_bundle",
            "path": "runtime_source_bundle.json",
            "uri": f"{source_prefix}/runtime_source_bundle.json",
            "bytes": 351_422,
            "sha256": "11" * 32,
            "generation": 1_900_000_000_000_001,
            "created": True,
            "readback_verified": True,
        },
        {
            "kind": f"{role}_role_payload_contract",
            "path": role_path,
            "uri": f"{source_prefix}/{role_path}",
            "bytes": 220_726,
            "sha256": "22" * 32,
            "generation": 1_900_000_000_000_002,
            "created": True,
            "readback_verified": True,
        },
    ]
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_role_bootstrap_manifest_v2"
        ),
        "deployment_contract_sha256": deployment_sha,
        "source_plan_sha256": "33" * 32,
        "source_provision_receipt_sha256": "44" * 32,
        "source_prefix": source_prefix,
        "external_job_id": external_job_id,
        "inner_job_id": role_runtime["instances"][position]["inner_job_id"],
        "source_role": role,
        "object_count": 2,
        "objects": objects,
        "objects_sha256": subject.canonical_sha256(objects),
        "runtime_source_bundle_sha256": "55" * 32,
        "all_generation_bound": True,
        "one_role_payload_only": True,
        "opponent_role_payload_present": False,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
    }
    return {
        **body,
        "role_manifest_sha256": subject.canonical_sha256(body),
    }


def _initial_values() -> dict[str, str]:
    role_runtime, external_job_id = _role_runtime_inputs()
    position = role_runtime["selected_job_ids"].index(external_job_id)
    values = {key: "x" for key in subject.INITIAL_METADATA_KEYS}
    values[subject.BLOCK_PROJECT_SSH_KEYS_KEY] = "true"
    values[subject.RELEASE_STATE_KEY] = subject.PENDING_RELEASE_STATE
    values[subject.DEPLOYMENT_CONTRACT_KEY] = subject.canonical_bytes(
        role_runtime
    ).decode("ascii")
    values[subject.EXTERNAL_JOB_ID_KEY] = external_job_id
    values[subject.SOURCE_ROLE_KEY] = role_runtime["source_roles"][position]
    values[subject.ROLE_BOOTSTRAP_MANIFEST_KEY] = (
        subject.canonical_bytes(
            _manifest(role_runtime, external_job_id)
        ).decode("ascii")
    )
    return values


def _role_runtime_inputs() -> tuple[dict[str, Any], str]:
    read = lambda name: json.loads(
        (STEP12_ROOT / name).read_text(encoding="utf-8")
    )
    candidate = read("candidate_transport_contract.json")
    reference = read("reference_transport_contract.json")
    public_key = read("controller_public_key.json")
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
    )
    external_job_id = deployment["selected_job_ids"][0]
    role_runtime = deployment_v2.validate_role_runtime_view(
        deployment,
        selected_payload_contract=candidate,
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        external_job_id=external_job_id,
    )
    return role_runtime, external_job_id


def _observed_identity(
    role_runtime: dict[str, Any],
    external_job_id: str,
) -> dict[str, Any]:
    position = role_runtime["selected_job_ids"].index(external_job_id)
    instance = role_runtime["instances"][position]
    return {
        "project_id": "ofc-solver-485418",
        "project_number": subject.EXPECTED_PROJECT_NUMBER,
        "zone": "asia-northeast1-b",
        "instance_name": instance["instance_name"],
        "provider_instance_id": "987654321012345678",
        "service_account_email": (
            "ofc-m31-t3-diagnostic@"
            "ofc-solver-485418.iam.gserviceaccount.com"
        ),
        "oauth_scopes": [
            "https://www.googleapis.com/auth/cloud-platform"
        ],
        "external_job_id": external_job_id,
        "source_role": instance["source_role"],
    }


def test_source_bundle_and_initial_postclaim_budget_are_exact() -> None:
    values = _initial_values()
    initial = subject.validate_initial_metadata_budget(values)
    postclaim = subject.validate_postclaim_metadata_budget(
        initial_values=values,
        claim_value='{"signed":"claim"}',
    )
    postrelease = subject.validate_postrelease_metadata_budget(
        initial_values=values,
        claim_value='{"signed":"claim"}',
        release_value='{"signed":"pair-release"}',
    )
    assert initial["postcreate_claim_present"] is False
    assert (
        initial["remaining_after_reserved_claim_and_release_bytes"]
        >= subject.MIN_POSTCLAIM_HEADROOM_BYTES
    )
    assert postclaim["postcreate_claim_present"] is True
    assert (
        postclaim["remaining_after_reserved_release_bytes"]
        >= subject.MIN_POSTCLAIM_HEADROOM_BYTES
    )
    assert postclaim["metadata_key_count"] == (
        initial["metadata_key_count"] + 1
    )
    assert postclaim["role_bootstrap_manifest_sha256"] == (
        initial["role_bootstrap_manifest_sha256"]
    )
    assert postrelease["pair_release_present"] is True
    assert (
        postrelease["remaining_headroom_bytes"]
        >= subject.MIN_POSTCLAIM_HEADROOM_BYTES
    )


def test_initial_metadata_rejects_claim_or_unknown_key() -> None:
    values = _initial_values()
    values[subject.POSTCREATE_CLAIM_KEY] = "{}"
    with pytest.raises(ValueError, match="allowlist"):
        subject.validate_initial_metadata_budget(values)
    values = _initial_values()
    values["unknown"] = "value"
    with pytest.raises(ValueError, match="allowlist"):
        subject.validate_initial_metadata_budget(values)


def test_per_value_and_aggregate_reserved_headroom_fail_closed() -> None:
    values = _initial_values()
    values[subject.STARTUP_KEY] = "x" * (
        subject.MAX_METADATA_VALUE_BYTES + 1
    )
    with pytest.raises(ValueError, match="metadata value"):
        subject.validate_initial_metadata_budget(values)

    values = _initial_values()
    values[subject.STARTUP_KEY] = "x" * 230_000
    values[subject.EXTERNAL_AUTHORIZATION_KEY] = "y" * 230_000
    with pytest.raises(ValueError, match="reserve claim"):
        subject.validate_initial_metadata_budget(values)


def test_role_manifest_tamper_and_noncanonical_json_fail_closed() -> None:
    values = _initial_values()
    changed = json.loads(values[subject.ROLE_BOOTSTRAP_MANIFEST_KEY])
    changed["objects"][0]["generation"] += 1
    values[subject.ROLE_BOOTSTRAP_MANIFEST_KEY] = (
        subject.canonical_bytes(changed).decode("ascii")
    )
    with pytest.raises(ValueError, match="manifest digest"):
        subject.validate_initial_metadata_budget(values)

    values = _initial_values()
    parsed = json.loads(values[subject.ROLE_BOOTSTRAP_MANIFEST_KEY])
    values[subject.ROLE_BOOTSTRAP_MANIFEST_KEY] = json.dumps(
        parsed, indent=2
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        subject.validate_initial_metadata_budget(values)


def test_postclaim_is_exact_add_only_and_bounded() -> None:
    values = _initial_values()
    with pytest.raises(ValueError, match="reserved budget"):
        subject.validate_postclaim_metadata_budget(
            initial_values=values,
            claim_value="z"
            * (subject.MAX_POSTCREATE_CLAIM_VALUE_BYTES + 1),
        )
    changed = dict(values)
    changed[subject.RELEASE_STATE_KEY] = "released"
    with pytest.raises(ValueError, match="claim-release"):
        subject.validate_postclaim_metadata_budget(
            initial_values=changed,
            claim_value="{}",
        )
    with pytest.raises(ValueError, match="reserved budget"):
        subject.validate_postrelease_metadata_budget(
            initial_values=values,
            claim_value="{}",
            release_value="z"
            * (subject.MAX_PAIR_RELEASE_VALUE_BYTES + 1),
        )


def test_worker_vm_identity_uses_only_guest_metadata_fields() -> None:
    role_runtime, external_job_id = _role_runtime_inputs()
    observed = _observed_identity(role_runtime, external_job_id)
    capability = subject.validate_vm_identity_after_claim(
        role_runtime_deployment=role_runtime,
        external_job_id=external_job_id,
        observed_identity=observed,
    )
    checked = subject.require_verified_vm_identity(
        capability,
        deployment_contract_sha256=role_runtime[
            "deployment_contract_sha256"
        ],
        external_job_id=external_job_id,
    )
    receipt = checked.receipt()
    assert (
        receipt["observation_phase"]
        == "worker_guest_metadata_after_claim"
    )
    assert not any("fingerprint" in key for key in receipt)

    changed = dict(observed)
    changed["metadata_fingerprint"] = "controller-only"
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_vm_identity_after_claim(
            role_runtime_deployment=role_runtime,
            external_job_id=external_job_id,
            observed_identity=changed,
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("project_id", "wrong-project"),
        ("project_number", "1"),
        ("zone", "us-central1-a"),
        ("instance_name", "wrong-instance"),
        ("provider_instance_id", "not-decimal"),
        ("service_account_email", "wrong@example.com"),
        ("oauth_scopes", []),
        ("external_job_id", "wrong-job"),
        ("source_role", "reference"),
    ],
)
def test_postclaim_vm_identity_mismatch_fails_closed(
    field: str, changed: Any
) -> None:
    role_runtime, external_job_id = _role_runtime_inputs()
    observed = _observed_identity(role_runtime, external_job_id)
    observed[field] = changed
    with pytest.raises(ValueError, match="provider identity"):
        subject.validate_vm_identity_after_claim(
            role_runtime_deployment=role_runtime,
            external_job_id=external_job_id,
            observed_identity=observed,
        )


def test_verified_vm_identity_cannot_be_forged_or_cross_bound() -> None:
    role_runtime, external_job_id = _role_runtime_inputs()
    observed = _observed_identity(role_runtime, external_job_id)
    capability = subject.validate_vm_identity_after_claim(
        role_runtime_deployment=role_runtime,
        external_job_id=external_job_id,
        observed_identity=observed,
    )
    with pytest.raises(ValueError, match="capability changed"):
        subject.require_verified_vm_identity(
            capability,
            deployment_contract_sha256="0" * 64,
            external_job_id=external_job_id,
        )
    with pytest.raises(ValueError, match="cannot be constructed"):
        subject.VerifiedVmIdentity(
            deployment_contract_sha256=role_runtime[
                "deployment_contract_sha256"
            ],
            external_job_id=external_job_id,
            inner_job_id=capability.inner_job_id,
            source_role=capability.source_role,
            instance_name=capability.instance_name,
            provider_instance_id=capability.provider_instance_id,
            project_id=capability.project_id,
            project_number=capability.project_number,
            zone=capability.zone,
            worker_service_account=capability.worker_service_account,
            oauth_scope=capability.oauth_scope,
            receipt_sha256=capability.receipt_sha256,
            _seal=object(),
        )
