from __future__ import annotations

import copy
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
OUTER_SRC = (
    STEP11_ROOT / "local_preflight" / "outer" / "src"
)
RUN_NONCE = "a1" * 32
SECOND_RUN_NONCE = "b2" * 32


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def controller_public_key_record() -> dict[str, Any]:
    return _read_json(STEP11_ROOT / "controller_public_key.json")


@pytest.fixture(scope="module")
def payload_contracts(
    controller_public_key_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    contracts = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=controller_public_key_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    return contracts[0], contracts[1]


@pytest.fixture(scope="module")
def deployment(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> dict[str, Any]:
    candidate, reference = payload_contracts
    return subject.build_step12b_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )


def test_fresh_external_identity_is_separate_from_immutable_payload(
    deployment: dict[str, Any],
) -> None:
    assert deployment["schema"] == subject.SCHEMA
    assert deployment["status"] == subject.STATUS
    assert deployment["stage_kind"] == subject.STAGE_KIND
    assert deployment["stage_id"] != payload_plan.STAGE2_ID
    assert deployment["run_name"] != payload_plan.STAGE2_RUN_NAME
    assert deployment["selected_job_ids"] != list(payload_plan.STAGE2_JOB_IDS)
    assert deployment["source_roles"] == ["candidate", "reference"]
    assert deployment["attempt_index"] == 0
    assert deployment["vm_count"] == 2

    payload = deployment["payload_binding"]
    assert payload["immutable_outer_package_reused"] is True
    assert (
        payload["outer_package_identity_sha256"]
        == subject.EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
    )
    assert (
        payload["inner_preview_stage_identity_sha256"]
        == subject.EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
    )
    assert (
        payload["inner_direct_stage_identity_sha256"]
        == subject.EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
    )
    assert payload["inner_job_ids"] == list(payload_plan.STAGE2_JOB_IDS)
    assert payload["package_write_authorized"] is False
    assert payload["payload_result_writes_authorized"] is False
    assert payload["payload_contracts_are_local_execution_only"] is True

    direct_sha = deployment["direct_stage_identity_sha256"]
    assert direct_sha != subject.EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
    assert deployment["direct_stage_identity"]["direct_namespace"] == (
        subject.DIRECT_NAMESPACE
    )
    assert deployment["remote_layout"]["stage_prefix"].startswith(
        "gs://pokerhu-ofc-solver-485418-training/"
        "hu-m31-r2diag-direct-v2/stages/"
    )
    assert (
        deployment["remote_layout"]["package_prefix"]
        == payload["package_prefix"]
    )
    assert payload["package_prefix"] not in deployment["remote_layout"][
        "stage_prefix"
    ]


def test_external_aliases_bind_exact_inner_jobs_and_manifests(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    candidate, reference = payload_contracts
    payloads = (candidate, reference)
    aliases = deployment["selected_job_ids"]
    bindings = deployment["payload_binding"]["job_bindings"]
    instances = deployment["instances"]
    assert len(set(aliases)) == 2
    assert [row["inner_job_id"] for row in bindings] == list(
        payload_plan.STAGE2_JOB_IDS
    )
    assert [row["source_role"] for row in bindings] == [
        "candidate",
        "reference",
    ]
    for alias, binding, instance, payload in zip(
        aliases, bindings, instances, payloads, strict=True
    ):
        inner_job = payload["metadata_binding"]["job_id"]
        preview_job = next(
            row
            for row in payload["adapter_preview"]["jobs"]
            if row["job_id"] == inner_job
        )
        assert alias != inner_job
        assert instance["job_id"] == alias
        assert instance["inner_job_id"] == inner_job
        assert binding["inner_job_manifest_path"] == (
            f"inner/jobs/{inner_job}.json"
        )
        assert binding["runner_job_manifest_sha256"] == (
            preview_job["runner_job_manifest"]["sha256"]
        )
        assert binding["work_hand_indices"] == list(
            payload_plan.STAGE2_HAND_INDICES
        )
        assert binding["payload_contract_sha256"] == subject.canonical_sha256(
            payload
        )


def test_run_scoped_controller_is_bound_into_run_and_direct_identities(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    controller = deployment["controller_service_account"]
    run_identity = deployment["run_identity"]
    direct_identity = deployment["direct_stage_identity"]
    tag = deployment["run_identity_seed_sha256"][:12]
    public_key_sha256 = subject.canonical_sha256(
        controller_public_key_record
    )
    assert deployment["controller_key_id"] == (
        controller_public_key_record["key_id"]
    )
    assert deployment["controller_public_key_sha256"] == public_key_sha256
    assert deployment["run_identity_seed"][
        "controller_public_key_sha256"
    ] == public_key_sha256
    assert run_identity["controller_public_key_sha256"] == public_key_sha256
    assert direct_identity["controller_public_key_sha256"] == (
        public_key_sha256
    )
    assert deployment["payload_binding"][
        "controller_public_key_sha256"
    ] == public_key_sha256
    assert controller == {
        "project": payload_transport.PROJECT,
        "account_id": f"ofc-m31-s2b-{tag}",
        "email": (
            f"ofc-m31-s2b-{tag}@{payload_transport.PROJECT}."
            "iam.gserviceaccount.com"
        ),
        "principal": (
            f"serviceAccount:ofc-m31-s2b-{tag}@"
            f"{payload_transport.PROJECT}.iam.gserviceaccount.com"
        ),
        "derived_from_run_tag": tag,
        "run_scoped": True,
        "legacy_shared_controller_reused": False,
    }
    assert len(controller["account_id"]) <= 30
    assert controller["email"] != (
        subject.LEGACY_SHARED_CONTROLLER_SERVICE_ACCOUNT
    )
    assert run_identity["controller_service_account"] == controller
    assert direct_identity["controller_service_account"] == controller

    unsigned_run_identity = dict(run_identity)
    run_identity_sha = unsigned_run_identity.pop("run_identity_sha256")
    assert run_identity_sha == subject.canonical_sha256(
        unsigned_run_identity
    )
    unsigned_direct_identity = dict(direct_identity)
    direct_identity_sha = unsigned_direct_identity.pop(
        "direct_stage_identity_sha256"
    )
    assert direct_identity_sha == subject.canonical_sha256(
        unsigned_direct_identity
    )

    candidate, reference = payload_contracts
    changed = copy.deepcopy(deployment)
    changed["controller_service_account"]["email"] = (
        subject.LEGACY_SHARED_CONTROLLER_SERVICE_ACCOUNT
    )
    unsigned = dict(changed)
    unsigned.pop("deployment_contract_sha256")
    changed["deployment_contract_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_deployment_contract(
            changed,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )


def test_direct_v2_layout_is_unique_complete_and_attempt0_only(
    deployment: dict[str, Any],
) -> None:
    layout = deployment["remote_layout"]
    jobs = layout["jobs"]
    all_uris = [layout["receive_uri"]]
    for job in jobs:
        assert len(job["upload_uris"]) == len(
            payload_plan.STAGE2_HAND_INDICES
        )
        assert len(job["heartbeat_uris"]) == len(
            payload_plan.STAGE2_HAND_INDICES
        )
        assert len(job["tree_object_uris"]) == 23
        assert job["result_prefix"].startswith(layout["result_prefix"] + "/")
        assert job["tree_prefix"].startswith(job["result_prefix"] + "/")
        assert job["done_uri"] == (
            f"{job['result_prefix']}/DONE.envelope.json"
        )
        all_uris.extend(job["upload_uris"])
        all_uris.extend(job["heartbeat_uris"])
        all_uris.extend(job["tree_object_uris"])
        all_uris.append(job["done_uri"])
    assert len(all_uris) == len(set(all_uris))

    names = [row["instance_name"] for row in deployment["instances"]]
    assert len(names) == len(set(names)) == 2
    assert all(name.endswith(deployment["direct_stage_identity_sha256"][:12]) for name in names)
    assert not any("candidate-01" in name or "reference-01" in name for name in names)

    boundary = deployment["authorization_boundary"]
    assert boundary == {
        "attempt_limit_per_job": 1,
        "attempt1_authorized": False,
        "resume_authorized": False,
        "third_vm_authorized": False,
        "package_write_authorized": False,
        "payload_result_write_authorized": False,
        "deployment_result_write_authorized": False,
        "iam_mutation_authorized": False,
        "vm_create_authorized": False,
        "launch_authorized": False,
        "separate_cloud_authorization_required": True,
        "cloud_mutation_performed": False,
    }


def test_contract_is_deterministic_but_a_new_nonce_changes_every_external_id(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    repeated = subject.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )
    other = subject.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=SECOND_RUN_NONCE,
    )
    assert repeated == deployment
    assert other["payload_binding"] == deployment["payload_binding"]
    for field in (
        "run_identity_sha256",
        "stage_id",
        "run_name",
        "selected_job_ids",
        "direct_stage_identity_sha256",
    ):
        assert other[field] != deployment[field]
    assert other["remote_layout"]["stage_prefix"] != deployment[
        "remote_layout"
    ]["stage_prefix"]
    assert [row["instance_name"] for row in other["instances"]] != [
        row["instance_name"] for row in deployment["instances"]
    ]
    assert other["remote_layout"]["package_prefix"] == deployment[
        "remote_layout"
    ]["package_prefix"]


def test_roundtrip_validator_rebuilds_all_fields(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    assert (
        subject.validate_step12b_deployment_contract(
            deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )
        == deployment
    )


def test_resealed_tamper_is_rejected_by_exact_rebuild(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    changed = copy.deepcopy(deployment)
    changed["authorization_boundary"]["launch_authorized"] = True
    unsigned = dict(changed)
    unsigned.pop("deployment_contract_sha256")
    changed["deployment_contract_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_deployment_contract(
            changed,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )


def test_payload_roles_cannot_be_swapped_or_duplicated(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    with pytest.raises(ValueError, match="mapping changed"):
        subject.build_deployment_contract(
            candidate_payload_contract=reference,
            reference_payload_contract=candidate,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )
    with pytest.raises(ValueError, match="mapping changed"):
        subject.build_deployment_contract(
            candidate_payload_contract=candidate,
            reference_payload_contract=candidate,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )


@pytest.mark.parametrize(
    "nonce",
    [
        "0" * 64,
        "A1" * 32,
        "a1" * 31,
        "a1" * 33,
        "not-a-sha256",
    ],
)
def test_run_nonce_must_be_one_nonzero_lowercase_256_bit_value(
    nonce: str,
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    with pytest.raises(ValueError, match="run nonce"):
        subject.build_deployment_contract(
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=nonce,
        )


def test_wrong_nonce_cannot_validate_a_sealed_contract(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_deployment_contract(
            deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=SECOND_RUN_NONCE,
        )


def test_contract_explicitly_remains_offline_and_non_promotable(
    deployment: dict[str, Any],
) -> None:
    assert deployment["diagnostic_only"] is True
    assert deployment["scientific_payload_present"] is False
    assert deployment["current_profile_sha256"] == (
        subject.EXPECTED_CURRENT_PROFILE_SHA256
    )
    assert deployment["current_profile_changed"] is False
    assert deployment["capabilities"] == {
        "immutable_payload_mapping_bound": True,
        "external_deployment_identity_built": True,
        "worker_bridge_implemented": False,
        "cloud_execution_consumable": False,
        "cloud_executable": False,
        "launch_ready": False,
        "current_profile_changed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    assert len(subject.canonical_bytes(deployment)) < 262_144


def test_payload_contract_tamper_is_rejected_before_external_identity_build(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    changed = copy.deepcopy(reference)
    changed["metadata_binding"]["job_id"] = "reference-shard-99"
    changed["metadata_binding_sha256"] = payload_transport.canonical_sha256(
        changed["metadata_binding"]
    )
    with pytest.raises(ValueError):
        subject.build_deployment_contract(
            candidate_payload_contract=candidate,
            reference_payload_contract=changed,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )


def test_resealed_payload_trust_cannot_substitute_an_unprovided_key(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = copy.deepcopy(payload_contracts)
    fake_key_id = "c3" * 32
    fake_public_sha = "d4" * 32
    for contract in (candidate, reference):
        trust = contract["authorization_contract"]
        trust["controller_key_id"] = fake_key_id
        trust["controller_public_key_sha256"] = fake_public_sha
    with pytest.raises(ValueError, match="supplied public trust key"):
        subject.build_deployment_contract(
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )


def test_valid_but_different_public_key_record_is_rejected(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    candidate, reference = payload_contracts
    other_signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    with pytest.raises(ValueError, match="supplied public trust key"):
        subject.build_deployment_contract(
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=other_signer.public_record,
            run_nonce=RUN_NONCE,
        )


def test_resealed_adapter_preview_capability_change_is_rejected(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = copy.deepcopy(payload_contracts)
    for contract in (candidate, reference):
        contract["adapter_preview"]["capabilities"][
            "cloud_executable"
        ] = True
        contract["adapter_preview_sha256"] = subject.canonical_sha256(
            contract["adapter_preview"]
        )
    with pytest.raises(ValueError):
        subject.build_deployment_contract(
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )


def test_deployment_module_imports_in_isolated_outer_namespace(
    tmp_path: Path,
) -> None:
    metadata_root = tmp_path / "metadata"
    metadata_package = metadata_root / "ofc_regular"
    metadata_package.mkdir(parents=True)
    module_name = (
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_deployment_contract_v2"
    )
    shutil.copy2(
        REPO_ROOT / "src" / "ofc_regular" / f"{module_name}.py",
        metadata_package / f"{module_name}.py",
    )
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
        f"from ofc_regular import {module_name} as subject\n"
        "assert subject.INNER_JOB_IDS == "
        "('candidate-shard-01','reference-shard-01')\n"
        "assert not any(name.endswith('diagnostic_canary_plan') "
        "for name in sys.modules)\n"
        "assert 'site-packages' not in '\\n'.join(sys.path).casefold()\n"
        "print(subject.SCHEMA)\n"
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
    assert completed.stdout.strip() == subject.SCHEMA


@pytest.mark.parametrize("position", [0, 1])
def test_role_runtime_view_needs_only_the_selected_payload(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
    position: int,
) -> None:
    checked = subject.validate_role_runtime_view(
        deployment,
        selected_payload_contract=payload_contracts[position],
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        external_job_id=deployment["selected_job_ids"][position],
    )
    assert checked == deployment


def test_role_runtime_view_rejects_missing_swapped_or_resealed_payload(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    with pytest.raises(ValueError, match="selected payload"):
        subject.validate_role_runtime_view(
            deployment,
            selected_payload_contract=None,  # type: ignore[arg-type]
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            external_job_id=deployment["selected_job_ids"][0],
        )
    with pytest.raises(ValueError, match="selected immutable payload"):
        subject.validate_role_runtime_view(
            deployment,
            selected_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            external_job_id=deployment["selected_job_ids"][0],
        )
    changed = copy.deepcopy(deployment)
    changed["instances"][0]["inner_job_id"] = "reference-shard-01"
    unsigned = dict(changed)
    unsigned.pop("deployment_contract_sha256")
    changed["deployment_contract_sha256"] = subject.canonical_sha256(
        unsigned
    )
    with pytest.raises(ValueError, match="runtime layout changed"):
        subject.validate_role_runtime_view(
            changed,
            selected_payload_contract=candidate,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            external_job_id=deployment["selected_job_ids"][0],
        )
