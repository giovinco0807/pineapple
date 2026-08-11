from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
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
OUTER_SRC = STEP11_ROOT / "local_preflight" / "outer" / "src"
RUN_NONCE = "a1" * 32
SECOND_RUN_NONCE = "b2" * 32
AUTH_NONCE = "c3" * 32
CLAIM_NONCE = "d4" * 32
ISSUED = 1_800_000_000
EXPIRES = ISSUED + 3_600
NOW = ISSUED + 120
PROVIDER_INSTANCE_IDS = (
    "9876543210001",
    "9876543210002",
)
METADATA_FINGERPRINTS = (
    "AbCdEfGhIjKlMnOpQrStUvWxYz012345",
    "ZyXwVuTsRqPoNmLkJiHgFeDcBa543210",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def context() -> dict[str, Any]:
    signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    public_record = dict(signer.public_record)
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"],
        done_records=[done],
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
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
    inventory = payloads[0]["remote_layout"]["package_inventory"]["records"]
    generations = {
        row["uri"]: 1_784_439_690_000_000 + position
        for position, row in enumerate(inventory, start=1)
    }
    return {
        "signer": signer,
        "public_record": public_record,
        "payloads": payloads,
        "deployment": deployment,
        "preflight_receipt": preflight_receipt,
        "validated_preflight": validated_preflight,
        "generations": generations,
    }


def _authorization(
    context: dict[str, Any],
    position: int = 0,
    *,
    nonce: str | None = AUTH_NONCE,
    signer: Any | None = None,
) -> dict[str, Any]:
    candidate, reference = context["payloads"]
    return subject.build_external_authorization(
        deployment_contract=context["deployment"],
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=context["public_record"],
        run_nonce=RUN_NONCE,
        external_job_id=context["deployment"]["selected_job_ids"][position],
        package_generations=context["generations"],
        validated_external_preflight=context["validated_preflight"],
        issued_unix_seconds=ISSUED,
        expires_unix_seconds=EXPIRES,
        signer=context["signer"] if signer is None else signer,
        nonce=nonce,
    )


def _claim(
    context: dict[str, Any],
    authorization: dict[str, Any],
    position: int = 0,
    *,
    nonce: str | None = CLAIM_NONCE,
) -> dict[str, Any]:
    candidate, reference = context["payloads"]
    return subject.build_external_worker_claim(
        authorization=authorization,
        deployment_contract=context["deployment"],
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=context["public_record"],
        run_nonce=RUN_NONCE,
        external_job_id=context["deployment"]["selected_job_ids"][position],
        package_generations=context["generations"],
        external_preflight_receipt=context["preflight_receipt"],
        project_number=subject.EXPECTED_PROJECT_NUMBER,
        provider_instance_id=PROVIDER_INSTANCE_IDS[position],
        metadata_fingerprint=METADATA_FINGERPRINTS[position],
        now_unix_seconds=NOW,
        signer=context["signer"],
        nonce=nonce,
    )


def _validate(
    context: dict[str, Any],
    authorization: dict[str, Any],
    claim: dict[str, Any],
    position: int = 0,
    **overrides: Any,
) -> subject.ExternalRuntimeApproval:
    candidate, reference = context["payloads"]
    inputs = {
        "authorization": authorization,
        "claim": claim,
        "deployment_contract": context["deployment"],
        "candidate_payload_contract": candidate,
        "reference_payload_contract": reference,
        "controller_public_key_record": context["public_record"],
        "run_nonce": RUN_NONCE,
        "external_job_id": context["deployment"]["selected_job_ids"][
            position
        ],
        "package_generations": context["generations"],
        "external_preflight_receipt": context["preflight_receipt"],
        "observed_project_number": subject.EXPECTED_PROJECT_NUMBER,
        "observed_provider_instance_id": PROVIDER_INSTANCE_IDS[position],
        "observed_metadata_fingerprint": METADATA_FINGERPRINTS[position],
        "verifier": payload_transport.RsaSha256ControllerTrustVerifier(
            context["public_record"]
        ),
        "now_unix_seconds": NOW,
    }
    inputs.update(overrides)
    return subject.validate_external_approval(**inputs)


def _validate_role_runtime(
    context: dict[str, Any],
    authorization: dict[str, Any],
    claim: dict[str, Any],
    position: int = 0,
    **overrides: Any,
) -> subject.ExternalRuntimeApproval:
    inputs = {
        "authorization": authorization,
        "claim": claim,
        "deployment_contract": context["deployment"],
        "selected_payload_contract": context["payloads"][position],
        "controller_public_key_record": context["public_record"],
        "run_nonce": RUN_NONCE,
        "external_job_id": context["deployment"]["selected_job_ids"][
            position
        ],
        "package_generations": context["generations"],
        "external_preflight_receipt": context["preflight_receipt"],
        "observed_project": payload_transport.PROJECT,
        "observed_project_number": subject.EXPECTED_PROJECT_NUMBER,
        "observed_zone": payload_transport.ZONE,
        "observed_instance_name": context["deployment"]["instances"][
            position
        ]["instance_name"],
        "observed_provider_instance_id": PROVIDER_INSTANCE_IDS[position],
        "observed_worker_service_account": (
            payload_transport.WORKER_SERVICE_ACCOUNT
        ),
        "observed_oauth_scopes": [
            payload_transport.REQUIRED_WORKER_OAUTH_SCOPE
        ],
        "verifier": payload_transport.RsaSha256ControllerTrustVerifier(
            context["public_record"]
        ),
        "now_unix_seconds": NOW,
    }
    inputs.update(overrides)
    return subject.validate_external_approval_role_runtime(**inputs)


def test_validated_preflight_public_accessor_revalidates_and_copies(
    context: dict[str, Any],
) -> None:
    first = subject.get_validated_external_preflight_receipt(
        context["validated_preflight"],
        deployment_contract=context["deployment"],
    )
    assert first == context["preflight_receipt"]
    first["passed"] = False
    second = subject.get_validated_external_preflight_receipt(
        context["validated_preflight"],
        deployment_contract=context["deployment"],
    )
    assert second == context["preflight_receipt"]
    assert second["passed"] is True


@pytest.mark.parametrize("position", [0, 1])
def test_both_roles_roundtrip_exact_external_identity(
    context: dict[str, Any],
    position: int,
) -> None:
    authorization = _authorization(
        context, position, nonce=f"{position + 3:02x}" * 32
    )
    claim = _claim(
        context, authorization, position, nonce=f"{position + 9:02x}" * 32
    )
    approval = _validate(context, authorization, claim, position)
    instance = context["deployment"]["instances"][position]
    assert approval.external_job_id == instance["job_id"]
    assert approval.inner_job_id == instance["inner_job_id"]
    assert approval.source_role == instance["source_role"]
    assert approval.instance_name == instance["instance_name"]
    assert approval.provider_instance_id == PROVIDER_INSTANCE_IDS[position]
    assert dict(approval.package_generations) == context["generations"]
    assert authorization["attempt_index"] == 0
    assert authorization["max_attempts"] == 1
    assert claim["attempt_index"] == 0
    assert claim["max_attempts"] == 1
    assert claim["metadata_fingerprint_semantics"] == (
        "pre_claim_cas_observation"
    )


def test_authorization_binds_gate_generations_and_external_result_layout(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    deployment = context["deployment"]
    layout = deployment["remote_layout"]["jobs"][0]
    assert authorization["external_preflight_receipt_sha256"] == (
        context["preflight_receipt"][
            "external_preflight_receipt_sha256"
        ]
    )
    assert authorization["external_preflight_receipt_schema"] == (
        subject.PREFLIGHT_RECEIPT_SCHEMA
    )
    assert authorization["external_preflight_receipt_validation"] == (
        "typed_exact_receipt_validated"
    )
    assert authorization["package_generations_sha256"] == (
        subject.canonical_sha256(context["generations"])
    )
    assert authorization["external_result_layout_sha256"] == (
        subject.canonical_sha256(layout)
    )
    source_hashes = authorization["immutable_source_hashes"]
    assert authorization["immutable_source_hashes_sha256"] == (
        subject.canonical_sha256(source_hashes)
    )
    assert source_hashes["outer_package_identity_sha256"] == (
        deployment_v2.EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
    )
    assert source_hashes["inner_adapter_preview_sha256"] == (
        deployment_v2.EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
    )
    assert source_hashes["runner_job_manifest_sha256"] == (
        deployment["payload_binding"]["job_bindings"][0][
            "runner_job_manifest_sha256"
        ]
    )
    assert authorization["payload_contract_sha256"] == (
        deployment["payload_binding"]["job_bindings"][0][
            "payload_contract_sha256"
        ]
    )
    assert authorization["allowed_operations"] == subject.ALLOWED_OPERATIONS
    assert "vm_create" not in authorization["allowed_operations"]
    assert "iam" not in " ".join(authorization["allowed_operations"])


def test_placeholder_hashes_cannot_mint_signed_preflight_authority(
    context: dict[str, Any],
) -> None:
    placeholders = {
        "deployment_contract": context["deployment"],
        "readonly_preflight_receipt_sha256": "aa" * 32,
        "iam_capacity_gate_receipt_sha256": "aa" * 32,
        "package_provision_receipt_sha256": "aa" * 32,
        "deployment_source_sha256": "aa" * 32,
        "alias_bridge_source_sha256": "aa" * 32,
        "prebootstrap_source_sha256": "aa" * 32,
        "startup_source_sha256": "aa" * 32,
        "controller_source_sha256": "aa" * 32,
    }
    with pytest.raises(ValueError, match="real gate validators"):
        subject.build_external_preflight_receipt(**placeholders)
    candidate, reference = context["payloads"]
    with pytest.raises(ValueError, match="validated preflight capability"):
        subject.build_external_authorization(
            deployment_contract=context["deployment"],
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=context["public_record"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            package_generations=context["generations"],
            validated_external_preflight=context["preflight_receipt"],
            issued_unix_seconds=ISSUED,
            expires_unix_seconds=EXPIRES,
            signer=context["signer"],
            nonce=AUTH_NONCE,
        )


def test_wrong_signer_and_wrong_verifier_are_rejected(
    context: dict[str, Any],
) -> None:
    other = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    with pytest.raises(ValueError, match="signer is not pinned"):
        _authorization(context, signer=other)
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    with pytest.raises(ValueError, match="verifier is not pinned"):
        _validate(
            context,
            authorization,
            claim,
            verifier=payload_transport.RsaSha256ControllerTrustVerifier(
                other.public_record
            ),
        )


def test_signature_tamper_and_cross_type_signature_are_rejected(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    changed = copy.deepcopy(authorization)
    changed["signature"] = (
        ("A" if changed["signature"][0] != "A" else "B")
        + changed["signature"][1:]
    )
    with pytest.raises(ValueError, match="trust verification failed"):
        _validate(context, changed, claim)
    changed_claim = copy.deepcopy(claim)
    changed_claim["signature"] = authorization["signature"]
    with pytest.raises(ValueError, match="trust verification failed"):
        _validate(context, authorization, changed_claim)


def test_cross_role_job_and_candidate_auth_reference_claim_replay_fail(
    context: dict[str, Any],
) -> None:
    candidate_auth = _authorization(context, 0)
    candidate_claim = _claim(context, candidate_auth, 0)
    with pytest.raises(ValueError):
        _validate(context, candidate_auth, candidate_claim, 1)

    reference_auth = _authorization(context, 1, nonce="13" * 32)
    reference_claim = _claim(
        context, reference_auth, 1, nonce="14" * 32
    )
    with pytest.raises(ValueError):
        _validate(context, candidate_auth, reference_claim, 0)
    with pytest.raises(ValueError):
        _validate(context, reference_auth, candidate_claim, 1)


def test_old_deployment_and_run_nonce_replay_fail(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    candidate, reference = context["payloads"]
    other_deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=context["public_record"],
        run_nonce=SECOND_RUN_NONCE,
    )
    with pytest.raises(ValueError):
        _validate(
            context,
            authorization,
            claim,
            deployment_contract=other_deployment,
            run_nonce=SECOND_RUN_NONCE,
        )
    with pytest.raises(ValueError):
        _validate(
            context,
            authorization,
            claim,
            run_nonce=SECOND_RUN_NONCE,
        )


@pytest.mark.parametrize("mode", ["changed", "missing", "extra", "bool"])
def test_wrong_package_generations_are_rejected(
    context: dict[str, Any],
    mode: str,
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    generations = dict(context["generations"])
    first = next(iter(generations))
    if mode == "changed":
        generations[first] += 1
    elif mode == "missing":
        generations.pop(first)
    elif mode == "extra":
        generations[f"{first}.extra"] = 1
    else:
        generations[first] = True
    with pytest.raises(ValueError):
        _validate(
            context,
            authorization,
            claim,
            package_generations=generations,
        )


@pytest.mark.parametrize(
    ("now", "message"),
    [
        (ISSUED - 1, "current time"),
        (EXPIRES, "current time"),
        (EXPIRES + 1, "current time"),
    ],
)
def test_authorization_time_window_is_enforced_at_validation(
    context: dict[str, Any],
    now: int,
    message: str,
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    with pytest.raises(ValueError, match=message):
        _validate(
            context,
            authorization,
            claim,
            now_unix_seconds=now,
        )


@pytest.mark.parametrize(
    "expires",
    [ISSUED + 59, ISSUED + 7_201],
)
def test_authorization_window_bounds_are_enforced_at_build(
    context: dict[str, Any],
    expires: int,
) -> None:
    candidate, reference = context["payloads"]
    with pytest.raises(ValueError, match="authorization expiry"):
        subject.build_external_authorization(
            deployment_contract=context["deployment"],
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=context["public_record"],
            run_nonce=RUN_NONCE,
            external_job_id=context["deployment"]["selected_job_ids"][0],
            package_generations=context["generations"],
            validated_external_preflight=context["validated_preflight"],
            issued_unix_seconds=ISSUED,
            expires_unix_seconds=expires,
            signer=context["signer"],
            nonce=AUTH_NONCE,
        )


def test_nonces_are_fresh_distinct_and_not_run_nonce(
    context: dict[str, Any],
) -> None:
    first = _authorization(context, nonce=None)
    second = _authorization(context, nonce=None)
    assert first["nonce"] != second["nonce"]
    assert first["nonce"] != RUN_NONCE
    first_claim = _claim(context, first, nonce=None)
    second_claim = _claim(context, first, nonce=None)
    assert first_claim["nonce"] != second_claim["nonce"]
    assert first_claim["nonce"] not in (first["nonce"], RUN_NONCE)
    with pytest.raises(ValueError, match="claim nonce"):
        _claim(context, first, nonce=first["nonce"])
    with pytest.raises(ValueError, match="authorization nonce"):
        _authorization(context, nonce=RUN_NONCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project", "wrong-project"),
        ("project_number", "111111111111"),
        ("zone", "us-central1-a"),
        ("worker_service_account", "wrong@example.com"),
        ("provider_instance_id", "999999999999"),
        ("metadata_fingerprint", "DifferentFingerprint12345"),
        ("instance_name", "wrong-instance"),
    ],
)
def test_claim_provider_identity_tamper_is_rejected(
    context: dict[str, Any],
    field: str,
    value: str,
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    claim[field] = value
    with pytest.raises(ValueError):
        _validate(context, authorization, claim)


def test_observed_provider_identity_must_match_claim_and_fixed_project(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    with pytest.raises(ValueError):
        _validate(
            context,
            authorization,
            claim,
            observed_provider_instance_id="999999999999",
        )
    with pytest.raises(ValueError):
        _validate(
            context,
            authorization,
            claim,
            observed_metadata_fingerprint="DifferentFingerprint12345",
        )
    with pytest.raises(ValueError, match="project number changed"):
        _validate(
            context,
            authorization,
            claim,
            observed_project_number="111111111111",
        )


def test_resealed_semantic_tamper_still_fails_exact_observation_binding(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    changed = copy.deepcopy(claim)
    changed["provider_instance_id"] = "999999999999"
    unsigned = {
        key: value for key, value in changed.items() if key != "signature"
    }
    changed["signature"] = context["signer"].sign(
        record_type="claim", unsigned=unsigned
    )
    with pytest.raises(ValueError, match="claim identity changed"):
        _validate(context, authorization, changed)


def test_records_never_serialize_private_material_or_perform_cloud_mutation(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    serialized = json.dumps(
        {"authorization": authorization, "claim": claim},
        sort_keys=True,
    ).lower()
    assert "private_key" not in serialized
    assert "access_token" not in serialized
    assert "authorization_header" not in serialized
    assert context["deployment"]["authorization_boundary"][
        "cloud_mutation_performed"
    ] is False
    assert context["deployment"]["current_profile_changed"] is False
    with pytest.raises(TypeError):
        context["signer"].__reduce__()


@pytest.mark.parametrize("position", [0, 1])
def test_role_runtime_approval_needs_only_selected_payload(
    context: dict[str, Any],
    position: int,
) -> None:
    authorization = _authorization(
        context, position, nonce=f"{position + 21:02x}" * 32
    )
    claim = _claim(
        context, authorization, position, nonce=f"{position + 31:02x}" * 32
    )
    full = _validate(context, authorization, claim, position)
    role_local = _validate_role_runtime(
        context, authorization, claim, position
    )
    assert role_local == full
    assert claim["metadata_fingerprint"] == (
        METADATA_FINGERPRINTS[position]
    )
    assert claim["metadata_fingerprint_semantics"] == (
        "pre_claim_cas_observation"
    )


def test_role_runtime_rejects_missing_swapped_and_resealed_role(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    with pytest.raises(ValueError, match="selected payload"):
        _validate_role_runtime(
            context,
            authorization,
            claim,
            selected_payload_contract=None,
        )
    with pytest.raises(ValueError, match="selected immutable payload"):
        _validate_role_runtime(
            context,
            authorization,
            claim,
            selected_payload_contract=context["payloads"][1],
        )
    changed = copy.deepcopy(authorization)
    changed["source_role"] = "reference"
    unsigned = {
        key: value for key, value in changed.items() if key != "signature"
    }
    changed["signature"] = context["signer"].sign(
        record_type="authorization", unsigned=unsigned
    )
    with pytest.raises(ValueError, match="authorization identity changed"):
        _validate_role_runtime(context, changed, claim)


def test_role_runtime_independently_enforces_authorization_expiry(
    context: dict[str, Any],
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    with pytest.raises(ValueError, match="current time"):
        _validate_role_runtime(
            context,
            authorization,
            claim,
            now_unix_seconds=EXPIRES + 1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_project", "wrong-project"),
        ("observed_zone", "us-central1-a"),
        ("observed_instance_name", "wrong-instance"),
        ("observed_worker_service_account", "wrong@example.com"),
        (
            "observed_oauth_scopes",
            ["https://www.googleapis.com/auth/devstorage.read_only"],
        ),
    ],
)
def test_role_runtime_checks_current_provider_metadata_identity(
    context: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    authorization = _authorization(context)
    claim = _claim(context, authorization)
    with pytest.raises(ValueError, match="provider metadata identity"):
        _validate_role_runtime(
            context,
            authorization,
            claim,
            **{field: value},
        )


def test_external_authorization_imports_in_isolated_outer_namespace(
    tmp_path: Path,
) -> None:
    metadata_root = tmp_path / "metadata"
    metadata_package = metadata_root / "ofc_regular"
    metadata_package.mkdir(parents=True)
    module_names = (
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_deployment_contract_v2",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_external_authorization_v2",
    )
    for module_name in module_names:
        shutil.copy2(
            REPO_ROOT / "src" / "ofc_regular" / f"{module_name}.py",
            metadata_package / f"{module_name}.py",
        )
    auth_module = module_names[1]
    script = (
        "import pathlib,sys\n"
        f"metadata={str(metadata_root)!r}\n"
        f"outer={str(OUTER_SRC)!r}\n"
        f"repo={str(REPO_ROOT)!r}\n"
        "kept=[p for p in sys.path if p and "
        "'site-packages' not in p.casefold() and "
        "not pathlib.Path(p).resolve().is_relative_to("
        "pathlib.Path(repo).resolve())]\n"
        "sys.path[:]=[metadata,outer,*kept]\n"
        f"from ofc_regular import {auth_module} as subject\n"
        "assert callable(subject.validate_external_approval)\n"
        "assert callable(subject.validate_external_approval_role_runtime)\n"
        "assert subject.ValidatedExternalPreflight.__module__ == "
        f"'ofc_regular.{auth_module}'\n"
        "assert not any(name.endswith('diagnostic_canary_plan') "
        "for name in sys.modules)\n"
        "assert 'site-packages' not in '\\n'.join(sys.path).casefold()\n"
        "print(subject.AUTHORIZATION_SCHEMA)\n"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == subject.AUTHORIZATION_SCHEMA
