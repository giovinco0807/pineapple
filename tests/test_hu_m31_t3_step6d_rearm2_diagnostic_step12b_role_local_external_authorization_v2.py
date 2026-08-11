from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as payload_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as step11_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_authorization_v2
    as subject,
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
RUN_NONCE = "c3" * 32
ISSUED = 1_800_000_000
NOW = ISSUED + 120
EXPIRES = ISSUED + 3_600
PROVIDER_INSTANCE_ID = "987654321012345678"
METADATA_FINGERPRINT = "abcDEF123_=-"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def inputs() -> dict[str, Any]:
    signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    public_key = dict(signer.public_record)
    stage1 = _read(STEP11_ROOT / "transport_contract.json")
    done = _read(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"], done_records=[done]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public_key,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
    )
    preflight_receipt = subject._build_external_preflight_receipt_record(
        deployment_contract=deployment,
        readonly_preflight_receipt_sha256=hashlib.sha256(
            b"readonly-preflight"
        ).hexdigest(),
        iam_capacity_gate_receipt_sha256=hashlib.sha256(
            b"iam-capacity-gate"
        ).hexdigest(),
        package_provision_receipt_sha256=hashlib.sha256(
            b"package-provision"
        ).hexdigest(),
        deployment_source_sha256=hashlib.sha256(
            b"deployment-source"
        ).hexdigest(),
        alias_bridge_source_sha256=hashlib.sha256(
            b"alias-bridge-source"
        ).hexdigest(),
        prebootstrap_source_sha256=hashlib.sha256(
            b"prebootstrap-source"
        ).hexdigest(),
        startup_source_sha256=hashlib.sha256(
            b"startup-source"
        ).hexdigest(),
        controller_source_sha256=hashlib.sha256(
            b"controller-source"
        ).hexdigest(),
    )
    validated_preflight = (
        subject._mint_validated_external_preflight_after_gate(
            preflight_receipt,
            deployment_contract=deployment,
            gate_validation_seal=subject._VALIDATED_PREFLIGHT_SEAL,
        )
    )
    generations = {
        row["uri"]: 1_784_439_690_000_000 + index
        for index, row in enumerate(
            payloads[0]["remote_layout"]["package_inventory"]["records"],
            start=1,
        )
    }
    authorizations = []
    claims = []
    for index, external_job_id in enumerate(
        deployment["selected_job_ids"]
    ):
        authorization = subject.build_external_authorization(
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            external_job_id=external_job_id,
            package_generations=generations,
            validated_external_preflight=validated_preflight,
            issued_unix_seconds=ISSUED,
            expires_unix_seconds=EXPIRES,
            signer=signer,
            nonce=hashlib.sha256(
                bytes.fromhex("d4" * 32) + bytes([index])
            ).hexdigest(),
        )
        claim = subject.build_external_worker_claim(
            authorization=authorization,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            external_job_id=external_job_id,
            package_generations=generations,
            external_preflight_receipt=preflight_receipt,
            project_number=subject.EXPECTED_PROJECT_NUMBER,
            provider_instance_id=str(
                int(PROVIDER_INSTANCE_ID) + index
            ),
            metadata_fingerprint=METADATA_FINGERPRINT + str(index),
            now_unix_seconds=NOW,
            signer=signer,
            nonce=hashlib.sha256(
                bytes.fromhex("e5" * 32) + bytes([index])
            ).hexdigest(),
        )
        authorizations.append(authorization)
        claims.append(claim)
    return {
        "payloads": payloads,
        "public_key": public_key,
        "generations": generations,
        "deployment": deployment,
        "preflight_receipt": preflight_receipt,
        "verifier": transport.RsaSha256ControllerTrustVerifier(public_key),
        "authorizations": authorizations,
        "claims": claims,
    }


def _role_kwargs(inputs: dict[str, Any], role_index: int) -> dict[str, Any]:
    deployment = inputs["deployment"]
    return {
        "deployment_contract": deployment,
        "selected_payload_contract": inputs["payloads"][role_index],
        "controller_public_key_record": inputs["public_key"],
        "run_nonce": RUN_NONCE,
        "external_job_id": deployment["selected_job_ids"][role_index],
        "package_generations": inputs["generations"],
        "external_preflight_receipt": inputs["preflight_receipt"],
        "verifier": inputs["verifier"],
        "now_unix_seconds": NOW,
    }


@pytest.mark.parametrize("role_index", [0, 1])
def test_role_local_signed_authorization_and_claim_validate(
    inputs: dict[str, Any], role_index: int
) -> None:
    kwargs = _role_kwargs(inputs, role_index)
    auth = subject.validate_role_local_external_authorization(
        inputs["authorizations"][role_index],
        **kwargs,
    )
    approval = subject.validate_role_local_external_approval(
        authorization=auth,
        claim=inputs["claims"][role_index],
        observed_project=transport.PROJECT,
        observed_project_number=subject.EXPECTED_PROJECT_NUMBER,
        observed_zone=transport.ZONE,
        observed_instance_name=inputs["deployment"]["instances"][
            role_index
        ]["instance_name"],
        observed_provider_instance_id=str(
            int(PROVIDER_INSTANCE_ID) + role_index
        ),
        observed_worker_service_account=transport.WORKER_SERVICE_ACCOUNT,
        observed_oauth_scopes=[transport.REQUIRED_WORKER_OAUTH_SCOPE],
        **kwargs,
    )
    assert approval.external_job_id == kwargs["external_job_id"]
    assert approval.source_role == ("candidate", "reference")[role_index]


def test_role_local_authorization_rejects_cross_role_swap(
    inputs: dict[str, Any],
) -> None:
    kwargs = _role_kwargs(inputs, 0)
    kwargs["selected_payload_contract"] = inputs["payloads"][1]
    with pytest.raises(ValueError, match="selected immutable payload"):
        subject.validate_role_local_external_authorization(
            inputs["authorizations"][0],
            **kwargs,
        )


def test_role_local_authorization_rejects_wrong_preflight_or_signature(
    inputs: dict[str, Any],
) -> None:
    kwargs = _role_kwargs(inputs, 0)
    with pytest.raises(ValueError, match="receipt"):
        subject.validate_role_local_external_authorization(
            inputs["authorizations"][0],
            **{
                **kwargs,
                "external_preflight_receipt": {
                    **inputs["preflight_receipt"],
                    "status": "wrong",
                },
            },
        )
    changed = dict(inputs["authorizations"][0])
    replacement = "B" if changed["signature"][0] != "B" else "A"
    changed["signature"] = replacement + changed["signature"][1:]
    with pytest.raises(ValueError, match="trust"):
        subject.validate_role_local_external_authorization(
            changed,
            **kwargs,
        )
