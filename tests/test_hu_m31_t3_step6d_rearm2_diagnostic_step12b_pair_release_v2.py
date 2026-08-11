from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
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
    as external_auth,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
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
RUN_NONCE = "a1" * 32
ISSUED = 1_800_000_000
NOW = ISSUED + 120
EXPIRES = ISSUED + 3_600
PROVIDER_IDS = ("9876543210001", "9876543210002")
F0 = (
    "ControllerPreCasFingerprint000001",
    "ControllerPreCasFingerprint000002",
)
F1 = (
    "ControllerPostCasFingerprint00001",
    "ControllerPostCasFingerprint00002",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _preflight(
    deployment: dict[str, Any],
) -> dict[str, Any]:
    return external_auth._build_external_preflight_receipt_record(
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


def _create_receipt(
    deployment: dict[str, Any],
) -> dict[str, Any]:
    controller = deployment["controller_service_account"]
    provider = {
        "name": (
            f"projects/{controller['project']}/serviceAccounts/"
            f"{controller['email']}"
        ),
        "project_id": controller["project"],
        "unique_id": "123456789012345678901",
        "email": controller["email"],
        "disabled": False,
    }
    body = {
        "schema": controller_sa.SCHEMA,
        "status": "run_scoped_controller_service_account_created",
        "project": controller["project"],
        "account_id": controller["account_id"],
        "email": controller["email"],
        "provider": provider,
        "create_call_count": 1,
        "create_retry_count": 0,
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    return {
        **body,
        "receipt_sha256": controller_sa.canonical_sha256(body),
    }


def _delete_receipt(
    deployment: dict[str, Any],
    created: dict[str, Any],
) -> dict[str, Any]:
    controller = deployment["controller_service_account"]
    body = {
        "schema": controller_sa.SCHEMA,
        "status": (
            "run_scoped_controller_service_account_deleted_and_absent"
        ),
        "project": controller["project"],
        "account_id": controller["account_id"],
        "email": controller["email"],
        "provider_unique_id": created["provider"]["unique_id"],
        "controller_create_receipt_sha256": created[
            "receipt_sha256"
        ],
        "delete_call_count": 1,
        "delete_retry_count": 0,
        "absence_get_count": 2,
        "final_get_status": 404,
        "readback_verified": True,
        "success_path_teardown_evidence": True,
        "cloud_mutation_performed": True,
    }
    return {
        **body,
        "receipt_sha256": controller_sa.canonical_sha256(body),
    }


def _phase2_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **identity,
            "controller_member_count": 0,
            "controller_member_present": False,
            "readback_complete": True,
        }
        for identity in plan[
            "controller_release_removal_group"
        ]["binding_identities"]
    ]


@pytest.fixture(scope="module")
def context() -> dict[str, Any]:
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
        if row["kind"]
        == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        payload_transport.build_job_contract(
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
    preflight = _preflight(deployment)
    validated_preflight = (
        external_auth._mint_validated_external_preflight_after_gate(
            preflight,
            deployment_contract=deployment,
            gate_validation_seal=external_auth._VALIDATED_PREFLIGHT_SEAL,
        )
    )
    generations = {
        row["uri"]: 1_784_439_690_000_000 + index
        for index, row in enumerate(
            payloads[0]["remote_layout"]["package_inventory"]["records"],
            start=1,
        )
    }
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        public_key
    )
    authorizations = []
    claims = []
    approvals = []
    readbacks = []
    for position, external_job_id in enumerate(
        deployment["selected_job_ids"]
    ):
        authorization = external_auth.build_external_authorization(
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
            nonce=f"{position + 3:02x}" * 32,
        )
        claim = external_auth.build_external_worker_claim(
            authorization=authorization,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            external_job_id=external_job_id,
            package_generations=generations,
            external_preflight_receipt=preflight,
            project_number=external_auth.EXPECTED_PROJECT_NUMBER,
            provider_instance_id=PROVIDER_IDS[position],
            metadata_fingerprint=F0[position],
            now_unix_seconds=NOW,
            signer=signer,
            nonce=f"{position + 9:02x}" * 32,
        )
        instance = deployment["instances"][position]
        observed = {
            "project_id": payload_transport.PROJECT,
            "project_number": external_auth.EXPECTED_PROJECT_NUMBER,
            "zone": payload_transport.ZONE,
            "instance_name": instance["instance_name"],
            "provider_instance_id": PROVIDER_IDS[position],
            "service_account_email": (
                payload_transport.WORKER_SERVICE_ACCOUNT
            ),
            "oauth_scopes": [
                payload_transport.REQUIRED_WORKER_OAUTH_SCOPE
            ],
            "external_job_id": external_job_id,
            "source_role": instance["source_role"],
        }
        approval = subject.validate_role_runtime_approval(
            authorization=authorization,
            claim=claim,
            deployment_contract=deployment,
            selected_payload_contract=payloads[position],
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            external_job_id=external_job_id,
            package_generations=generations,
            external_preflight_receipt=preflight,
            observed_identity=observed,
            verifier=verifier,
            now_unix_seconds=NOW,
        )
        readback = subject.build_claim_cas_readback_receipt(
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            external_job_id=external_job_id,
            authorization=authorization,
            claim=claim,
            package_generations=generations,
            external_preflight_receipt=preflight,
            verifier=verifier,
            now_unix_seconds=NOW,
            provider_instance_id=PROVIDER_IDS[position],
            controller_pre_cas_metadata_fingerprint=F0[position],
            controller_post_cas_metadata_fingerprint=F1[position],
            claimed_metadata_sha256=hashlib.sha256(
                f"claimed-{position}".encode("ascii")
            ).hexdigest(),
        )
        authorizations.append(authorization)
        claims.append(claim)
        approvals.append(approval)
        readbacks.append(readback)
    phase2_plan = phase2_iam.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=EXPIRES,
    )
    phase2 = subject.build_phase2_zero_receipt(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        phase2_iam_plan=phase2_plan,
        binding_readbacks=_phase2_rows(phase2_plan),
    )
    created = _create_receipt(deployment)
    deleted = _delete_receipt(deployment, created)
    release = subject.build_pair_release(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        claim_cas_readback_receipts=readbacks,
        phase2_iam_plan=phase2_plan,
        phase2_zero_receipt=phase2,
        controller_create_receipt=created,
        controller_delete_receipt=deleted,
        issued_unix_seconds=NOW + 1,
        signer=signer,
        nonce="f1" * 32,
    )
    return {
        "signer": signer,
        "public_key": public_key,
        "payloads": payloads,
        "deployment": deployment,
        "preflight": preflight,
        "generations": generations,
        "verifier": verifier,
        "authorizations": authorizations,
        "claims": claims,
        "approvals": approvals,
        "readbacks": readbacks,
        "phase2_plan": phase2_plan,
        "phase2": phase2,
        "created": created,
        "deleted": deleted,
        "release": release,
    }


@pytest.mark.parametrize("position", [0, 1])
def test_pair_release_roundtrip_for_both_roles(
    context: dict[str, Any], position: int
) -> None:
    deployment = context["deployment"]
    external_job_id = deployment["selected_job_ids"][position]
    approval = subject.validate_pair_release_role_runtime(
        context["release"],
        deployment_contract=deployment,
        selected_payload_contract=context["payloads"][position],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
        external_job_id=external_job_id,
        role_approval=context["approvals"][position],
        verifier=context["verifier"],
    )
    checked = subject.require_pair_release_approval(
        approval,
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        external_job_id=external_job_id,
        role_approval=context["approvals"][position],
    )
    assert checked.own_claim_sha256 == context["approvals"][
        position
    ].claim_sha256
    assert context["release"]["both_claim_cas_readbacks_complete"] is True
    assert context["release"]["phase2_controller_binding_count"] == 0
    assert (
        context["release"][
            "controller_service_account_final_get_status"
        ]
        == 404
    )
    nonce_digests = {
        context["release"]["run_nonce_digest"],
        *context["release"]["authorization_nonce_digests"],
        *context["release"]["claim_nonce_digests"],
        context["release"]["release_nonce_digest"],
    }
    assert len(nonce_digests) == 6
    assert context["release"]["all_six_nonces_unique"] is True
    assert context["release"]["phase2_expected_controller_binding_count"] == 5
    assert context["release"]["phase2_iam_plan_sha256"] == (
        context["phase2_plan"]["plan_sha256"]
    )


def test_release_requires_two_claims_phase2_zero_and_deleted_404(
    context: dict[str, Any],
) -> None:
    base = {
        "deployment_contract": context["deployment"],
        "candidate_payload_contract": context["payloads"][0],
        "reference_payload_contract": context["payloads"][1],
        "controller_public_key_record": context["public_key"],
        "run_nonce": RUN_NONCE,
        "phase2_iam_plan": context["phase2_plan"],
        "phase2_zero_receipt": context["phase2"],
        "controller_create_receipt": context["created"],
        "controller_delete_receipt": context["deleted"],
        "issued_unix_seconds": NOW + 1,
        "signer": context["signer"],
        "nonce": "f2" * 32,
    }
    with pytest.raises(ValueError, match="count"):
        subject.build_pair_release(
            claim_cas_readback_receipts=context["readbacks"][:1],
            **base,
        )
    bad_phase2 = copy.deepcopy(context["phase2"])
    bad_phase2["binding_readbacks"][0][
        "controller_member_present"
    ] = True
    body = dict(bad_phase2)
    body.pop("receipt_sha256")
    bad_phase2["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="phase2"):
        subject.build_pair_release(
            claim_cas_readback_receipts=context["readbacks"],
            **{**base, "phase2_zero_receipt": bad_phase2},
        )
    bad_delete = copy.deepcopy(context["deleted"])
    bad_delete["final_get_status"] = 200
    body = dict(bad_delete)
    body.pop("receipt_sha256")
    bad_delete["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="delete receipt"):
        subject.build_pair_release(
            claim_cas_readback_receipts=context["readbacks"],
            **{**base, "controller_delete_receipt": bad_delete},
        )


def test_claim_readback_requires_distinct_controller_f0_f1(
    context: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="binding"):
        subject.build_claim_cas_readback_receipt(
            deployment_contract=context["deployment"],
            candidate_payload_contract=context["payloads"][0],
            reference_payload_contract=context["payloads"][1],
            controller_public_key_record=context["public_key"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            authorization=context["authorizations"][0],
            claim=context["claims"][0],
            package_generations=context["generations"],
            external_preflight_receipt=context["preflight"],
            verifier=context["verifier"],
            now_unix_seconds=NOW,
            provider_instance_id=PROVIDER_IDS[0],
            controller_pre_cas_metadata_fingerprint=F0[0],
            controller_post_cas_metadata_fingerprint=F0[0],
            claimed_metadata_sha256=hashlib.sha256(b"claimed").hexdigest(),
        )


def test_claim_readback_validates_original_signed_authorization(
    context: dict[str, Any],
) -> None:
    authorization = copy.deepcopy(context["authorizations"][0])
    authorization["nonce"] = "e1" * 32
    with pytest.raises(ValueError, match="authorization"):
        subject.build_claim_cas_readback_receipt(
            deployment_contract=context["deployment"],
            candidate_payload_contract=context["payloads"][0],
            reference_payload_contract=context["payloads"][1],
            controller_public_key_record=context["public_key"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            authorization=authorization,
            claim=context["claims"][0],
            package_generations=context["generations"],
            external_preflight_receipt=context["preflight"],
            verifier=context["verifier"],
            now_unix_seconds=NOW,
            provider_instance_id=PROVIDER_IDS[0],
            controller_pre_cas_metadata_fingerprint=F0[0],
            controller_post_cas_metadata_fingerprint=F1[0],
            claimed_metadata_sha256=hashlib.sha256(b"claimed").hexdigest(),
        )


def test_pair_release_rejects_cross_role_and_release_nonce_reuse(
    context: dict[str, Any],
) -> None:
    base = {
        "deployment_contract": context["deployment"],
        "candidate_payload_contract": context["payloads"][0],
        "reference_payload_contract": context["payloads"][1],
        "controller_public_key_record": context["public_key"],
        "run_nonce": RUN_NONCE,
        "phase2_iam_plan": context["phase2_plan"],
        "phase2_zero_receipt": context["phase2"],
        "controller_create_receipt": context["created"],
        "controller_delete_receipt": context["deleted"],
        "issued_unix_seconds": NOW + 1,
        "signer": context["signer"],
    }
    collided = copy.deepcopy(context["readbacks"])
    collided[1]["authorization_nonce_digest"] = collided[0][
        "authorization_nonce_digest"
    ]
    body = dict(collided[1])
    body.pop("receipt_sha256")
    collided[1]["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="nonce collision"):
        subject.build_pair_release(
            claim_cas_readback_receipts=collided,
            nonce="f2" * 32,
            **base,
        )

    with pytest.raises(ValueError, match="reused signed nonce"):
        subject.build_pair_release(
            claim_cas_readback_receipts=context["readbacks"],
            nonce=context["authorizations"][0]["nonce"],
            **base,
        )


def test_phase2_zero_is_bound_to_exact_plan_identity_list(
    context: dict[str, Any],
) -> None:
    changed = copy.deepcopy(context["phase2"])
    changed["binding_readbacks"][0]["resource"] += "-forged"
    changed["binding_readbacks_sha256"] = subject.canonical_sha256(
        changed["binding_readbacks"]
    )
    body = dict(changed)
    body.pop("receipt_sha256")
    changed["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="identity"):
        subject.validate_phase2_zero_receipt(
            changed,
            deployment_contract=context["deployment"],
            candidate_payload_contract=context["payloads"][0],
            reference_payload_contract=context["payloads"][1],
            controller_public_key_record=context["public_key"],
            run_nonce=RUN_NONCE,
            phase2_iam_plan=context["phase2_plan"],
        )


def test_pair_release_requires_matching_create_and_success_teardown(
    context: dict[str, Any],
) -> None:
    base = {
        "deployment_contract": context["deployment"],
        "candidate_payload_contract": context["payloads"][0],
        "reference_payload_contract": context["payloads"][1],
        "controller_public_key_record": context["public_key"],
        "run_nonce": RUN_NONCE,
        "claim_cas_readback_receipts": context["readbacks"],
        "phase2_iam_plan": context["phase2_plan"],
        "phase2_zero_receipt": context["phase2"],
        "controller_create_receipt": context["created"],
        "issued_unix_seconds": NOW + 1,
        "signer": context["signer"],
        "nonce": "f3" * 32,
    }
    failure_teardown = copy.deepcopy(context["deleted"])
    failure_teardown["success_path_teardown_evidence"] = False
    body = dict(failure_teardown)
    body.pop("receipt_sha256")
    failure_teardown["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="delete receipt"):
        subject.build_pair_release(
            controller_delete_receipt=failure_teardown,
            **base,
        )

    wrong_create = copy.deepcopy(context["created"])
    wrong_create["provider"]["unique_id"] = "123456789012345678902"
    body = dict(wrong_create)
    body.pop("receipt_sha256")
    wrong_create["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="delete receipt"):
        subject.build_pair_release(
            controller_create_receipt=wrong_create,
            controller_delete_receipt=context["deleted"],
            **{
                key: value
                for key, value in base.items()
                if key != "controller_create_receipt"
            },
        )


def test_signed_release_still_rejects_wrong_own_claim(
    context: dict[str, Any],
) -> None:
    changed = copy.deepcopy(context["release"])
    changed["claim_readbacks"][0]["claim_sha256"] = "a5" * 32
    changed["claim_sha256s"][0] = "a5" * 32
    unsigned = dict(changed)
    unsigned.pop("signature")
    changed["signature"] = context["signer"].sign(
        record_type=subject.PAIR_RELEASE_SIGNATURE_RECORD_TYPE,
        unsigned=unsigned,
    )
    with pytest.raises(ValueError, match="own signed claim"):
        subject.validate_pair_release_role_runtime(
            changed,
            deployment_contract=context["deployment"],
            selected_payload_contract=context["payloads"][0],
            controller_public_key_record=context["public_key"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            role_approval=context["approvals"][0],
            verifier=context["verifier"],
        )


class _Reader:
    def __init__(self, values: list[str | None]) -> None:
        self.values = list(values)
        self.calls = 0

    def read_pair_release(self) -> str | None:
        self.calls += 1
        return self.values.pop(0) if self.values else None


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_bounded_wait_blocks_until_valid_release_and_times_out(
    context: dict[str, Any],
) -> None:
    raw = subject.canonical_bytes(context["release"]).decode("ascii")
    reader = _Reader([None, None, raw])
    clock = _Clock()
    approval = subject.wait_for_pair_release(
        reader=reader,
        deployment_contract=context["deployment"],
        selected_payload_contract=context["payloads"][0],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
        external_job_id=context["deployment"]["selected_job_ids"][0],
        role_approval=context["approvals"][0],
        verifier=context["verifier"],
        timeout_seconds=10,
        poll_seconds=2,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert approval.external_job_id == (
        context["deployment"]["selected_job_ids"][0]
    )
    assert reader.calls == 3

    clock = _Clock()
    with pytest.raises(TimeoutError, match="timed out"):
        subject.wait_for_pair_release(
            reader=_Reader([None, None, None]),
            deployment_contract=context["deployment"],
            selected_payload_contract=context["payloads"][0],
            controller_public_key_record=context["public_key"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            role_approval=context["approvals"][0],
            verifier=context["verifier"],
            timeout_seconds=3,
            poll_seconds=1,
            now=clock.now,
            sleep=clock.sleep,
        )


def test_runtime_and_release_capabilities_cannot_be_forged(
    context: dict[str, Any],
) -> None:
    role = context["approvals"][0]
    with pytest.raises(ValueError, match="cannot be forged"):
        subject.ValidatedRoleRuntimeApproval(
            deployment_contract_sha256=role.deployment_contract_sha256,
            authorization_sha256=role.authorization_sha256,
            claim_sha256=role.claim_sha256,
            external_job_id=role.external_job_id,
            inner_job_id=role.inner_job_id,
            source_role=role.source_role,
            instance_name=role.instance_name,
            provider_instance_id=role.provider_instance_id,
            controller_preclaim_metadata_fingerprint=(
                role.controller_preclaim_metadata_fingerprint
            ),
            package_generations_sha256=(
                role.package_generations_sha256
            ),
            vm_identity_receipt_sha256=(
                role.vm_identity_receipt_sha256
            ),
            receipt_sha256=role.receipt_sha256,
            _seal=object(),
        )
