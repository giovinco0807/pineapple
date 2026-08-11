from __future__ import annotations

import copy
import json
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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
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
SECOND_RUN_NONCE = "d4" * 32
ISSUED = 1_900_000_000
EXPIRES = ISSUED + 3_600


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
    return deployment_v2.build_step12b_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )


@pytest.fixture(scope="module")
def plan(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> dict[str, Any]:
    candidate, reference = payload_contracts
    return subject.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=EXPIRES,
    )


def _validate(
    value: dict[str, Any],
    *,
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
    run_nonce: str = RUN_NONCE,
) -> dict[str, Any]:
    candidate, reference = payload_contracts
    return subject.validate_step12b_phase2_iam_plan(
        value,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(value)
    changed.pop("plan_sha256")
    changed["plan_sha256"] = subject.canonical_sha256(changed)
    return changed


def _role_readbacks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "stage": row["stage"],
            "included_permissions": list(row["included_permissions"]),
            "deleted": False,
            "get_status": 200,
            "readback_complete": True,
        }
        for row in plan["custom_role_readback_contract"]["requirements"]
    ]


def _binding_readbacks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: copy.deepcopy(row[key])
            for key in (
                "purpose",
                "target",
                "resource",
                "role",
                "member",
                "condition",
                "binding_sha256",
            )
        }
        | {
            "member_occurrences": 1,
            "readback_complete": True,
        }
        for row in [
            *plan["phase2_bindings"]["controller"],
            *plan["phase2_bindings"]["worker"],
        ]
    ]


def test_plan_validates_by_complete_reconstruction(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    assert (
        _validate(
            plan,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )
        == plan
    )
    assert plan["schema"] == subject.SCHEMA
    assert plan["status"] == subject.STATUS
    assert plan["capabilities"]["cloud_mutation_performed"] is False
    assert plan["capabilities"]["launch_authorized"] is False
    assert plan["capabilities"]["current_profile_changed"] is False


def test_phase1_token_creator_is_not_a_phase2_binding(
    plan: dict[str, Any],
) -> None:
    bindings = plan["phase2_bindings"]
    all_bindings = [*bindings["controller"], *bindings["worker"]]
    assert bindings["phase1_token_creator_binding_included"] is False
    assert bindings["phase1_token_barrier_separate"] is True
    assert bindings["token_creator_role_forbidden"] == (
        subject.TOKEN_CREATOR_ROLE
    )
    assert all(
        row["role"] != subject.TOKEN_CREATOR_ROLE
        for row in all_bindings
    )
    assert all(
        row["target"] != "controller_service_account"
        for row in all_bindings
    )


def test_exact_controller_and_worker_binding_purposes(
    plan: dict[str, Any],
) -> None:
    bindings = plan["phase2_bindings"]
    controller = bindings["controller"]
    worker = bindings["worker"]
    assert len(controller) == subject.CONTROLLER_BINDING_COUNT == 5
    assert len(worker) == subject.WORKER_BINDING_COUNT == 3
    assert {row["purpose"] for row in controller} == {
        "controller_vm_launch",
        "controller_instance_lifecycle",
        "controller_zone_operation_reader",
        "controller_service_usage",
        "controller_worker_act_as",
    }
    assert {row["purpose"] for row in worker} == {
        "worker_self_delete",
        "worker_package_and_result_reader",
        "worker_result_creator",
    }
    assert all(row["target"] != "bucket" for row in controller)
    assert plan["principals"]["controller_gcs_access_required"] is False
    assert plan["principals"][
        "user_clients_monitor_results_after_pair_release"
    ] is True


def test_run_scoped_controller_and_shared_worker_are_exact(
    plan: dict[str, Any],
    deployment: dict[str, Any],
) -> None:
    principals = plan["principals"]
    controller = deployment["controller_service_account"]
    assert principals["controller_service_account"] == controller
    assert principals["controller_principal"] == controller["principal"]
    assert controller["run_scoped"] is True
    assert controller["legacy_shared_controller_reused"] is False
    assert principals["worker_service_account"] == (
        payload_transport.WORKER_SERVICE_ACCOUNT
    )
    assert principals["worker_principal"] == subject.WORKER_PRINCIPAL


def test_existing_custom_roles_are_read_only_and_permission_exact(
    plan: dict[str, Any],
) -> None:
    contract = plan["custom_role_readback_contract"]
    assert contract["requirement_count"] == 6
    assert contract["role_create_authorized"] is False
    assert contract["role_patch_authorized"] is False
    assert contract["role_delete_authorized"] is False
    requirements = {
        row["purpose"]: row for row in contract["requirements"]
    }
    assert set(requirements) == set(subject.CUSTOM_ROLE_IDS)
    for purpose, role_id in subject.CUSTOM_ROLE_IDS.items():
        row = requirements[purpose]
        assert row["name"] == (
            f"projects/{subject.PROJECT}/roles/{role_id}"
        )
        assert row["stage"] == "GA"
        assert row["included_permissions"] == sorted(
            subject.CUSTOM_ROLE_PERMISSIONS[purpose]
        )
        assert row["existing_role_get_readback_required"] is True
        assert row["role_create_authorized"] is False
        assert row["role_patch_authorized"] is False


def test_worker_can_read_package_and_new_results_but_only_create_direct_v2(
    plan: dict[str, Any],
) -> None:
    boundary = plan["object_boundary"]
    package_prefix = boundary["immutable_package_read_prefix"]
    bootstrap_prefix = boundary["fresh_bootstrap_source_read_prefix"]
    create_prefixes = boundary["direct_v2_result_create_prefixes"]
    assert package_prefix.startswith(
        f"gs://{subject.BUCKET}/{payload_transport.DIRECT_NAMESPACE}/packages/"
    )
    assert len(create_prefixes) == 4
    assert bootstrap_prefix.startswith(
        f"gs://{subject.BUCKET}/{deployment_v2.DIRECT_NAMESPACE}/"
        "bootstrap-sources/"
    )
    assert boundary["fresh_bootstrap_source_object_count"] == 3
    assert boundary["role_bootstrap_source_object_count"] == 2
    assert all(
        prefix.startswith(
            f"gs://{subject.BUCKET}/{deployment_v2.DIRECT_NAMESPACE}/"
        )
        for prefix in create_prefixes
    )
    assert package_prefix not in create_prefixes
    assert bootstrap_prefix not in create_prefixes
    assert boundary["old_payload_result_prefix"] not in create_prefixes
    assert all(
        not prefix.startswith(boundary["old_direct_stage_root"] + "/")
        for prefix in create_prefixes
    )
    assert boundary["package_write_authorized"] is False
    assert boundary["bootstrap_source_write_authorized"] is False
    assert boundary["old_payload_result_write_authorized"] is False
    assert boundary["old_direct_namespace_write_authorized"] is False
    assert boundary["worker_object_list_authorized"] is False
    assert boundary["worker_object_delete_authorized"] is False
    assert boundary["worker_object_overwrite_authorized"] is False
    assert boundary["worker_create_if_generation_match_required"] == 0
    assert boundary[
        "worker_create_generation_sha_bytes_readback_required"
    ] is True

    creator = next(
        row
        for row in plan["phase2_bindings"]["worker"]
        if row["purpose"] == "worker_result_creator"
    )
    reader = next(
        row
        for row in plan["phase2_bindings"]["worker"]
        if row["purpose"] == "worker_package_and_result_reader"
    )
    assert bootstrap_prefix.removeprefix(
        f"gs://{subject.BUCKET}/"
    ) in reader["condition"]["expression"]
    assert package_prefix not in creator["condition"]["expression"]
    assert bootstrap_prefix not in creator["condition"]["expression"]
    assert boundary["old_payload_result_prefix"] not in (
        creator["condition"]["expression"]
    )
    for prefix in create_prefixes:
        object_suffix = prefix.removeprefix(
            f"gs://{subject.BUCKET}/"
        )
        assert object_suffix in creator["condition"]["expression"]


def test_exact_attempt0_pair_and_no_third_vm(
    plan: dict[str, Any],
    deployment: dict[str, Any],
) -> None:
    boundary = plan["instance_boundary"]
    names = [row["instance_name"] for row in deployment["instances"]]
    assert boundary["authorized_instance_names"] == names
    assert boundary["authorized_disk_names"] == names
    assert len(boundary["allowed_launch_requests"]) == 2
    assert boundary["vm_count"] == 2
    assert boundary["max_concurrent_vms"] == 2
    assert boundary["authorized_attempt_index"] == 0
    assert boundary["max_attempts_per_job"] == 1
    assert boundary["attempt1_authorized"] is False
    assert boundary["resume_authorized"] is False
    assert boundary["third_vm_authorized"] is False
    for request in boundary["allowed_launch_requests"]:
        assert request["attempt_index"] == 0
        assert request["machine_type"] == "c4-standard-8"
        assert request["provisioning_model"] == "SPOT"
        assert request["external_access_configs"] == []
        assert request["worker_service_account"] == (
            payload_transport.WORKER_SERVICE_ACCOUNT
        )
        body = dict(request)
        digest = body.pop("request_sha256")
        assert subject.canonical_sha256(body) == digest


def test_removal_groups_expose_exact_identities_and_digests(
    plan: dict[str, Any],
) -> None:
    controller = plan["controller_release_removal_group"]
    worker = plan["worker_final_cleanup_removal_group"]
    assert controller["binding_count"] == 5
    assert worker["binding_count"] == 3
    assert controller["binding_identities_sha256"] == (
        subject.canonical_sha256(controller["binding_identities"])
    )
    assert worker["binding_identities_sha256"] == (
        subject.canonical_sha256(worker["binding_identities"])
    )
    assert len(
        {
            (row["resource"], row["role"])
            for row in controller["binding_identities"]
        }
    ) == 5
    assert len(
        {
            (row["resource"], row["role"])
            for row in worker["binding_identities"]
        }
    ) == 3
    assert controller[
        "exact_zero_readback_required_before_pair_release"
    ] is True
    assert worker[
        "exact_zero_readback_required_at_final_cleanup"
    ] is True
    assert plan["cleanup_order"][
        "dangerous_iam_removed_before_user_compute_cleanup"
    ] is True


def test_authorization_window_is_at_most_7200_seconds(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    exact = subject.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=(
            ISSUED + subject.MAX_AUTHORIZATION_WINDOW_SECONDS
        ),
    )
    assert (
        exact["authorization_window"]["expires_at_unix_seconds"]
        - exact["authorization_window"]["issued_at_unix_seconds"]
        == 7_200
    )
    with pytest.raises(ValueError, match="not short-lived"):
        subject.build_step12b_phase2_iam_plan(
            deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            issued_at_unix_seconds=ISSUED,
            expires_at_unix_seconds=ISSUED + 7_201,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["phase2_bindings"]["controller"].append(
                copy.deepcopy(value["phase2_bindings"]["controller"][0])
            ),
            "plan changed",
        ),
        (
            lambda value: value["phase2_bindings"]["controller"][0].update(
                {"role": subject.TOKEN_CREATOR_ROLE}
            ),
            "plan changed",
        ),
        (
            lambda value: value["phase2_bindings"]["worker"][2][
                "condition"
            ].update(
                {
                    "expression": (
                        value["phase2_bindings"]["worker"][2][
                            "condition"
                        ]["expression"]
                        + " || old-result"
                    )
                }
            ),
            "plan changed",
        ),
        (
            lambda value: value["custom_role_readback_contract"][
                "requirements"
            ][0]["included_permissions"].append("storage.objects.list"),
            "plan changed",
        ),
        (
            lambda value: value["instance_boundary"].update(
                {"third_vm_authorized": True}
            ),
            "plan changed",
        ),
        (
            lambda value: value[
                "controller_release_removal_group"
            ]["binding_identities"][0].update(
                {"role": "roles/owner"}
            ),
            "plan changed",
        ),
    ],
)
def test_resealed_tampering_is_rejected_by_reconstruction(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
    mutator: Any,
    message: str,
) -> None:
    changed = copy.deepcopy(plan)
    mutator(changed)
    changed = _reseal(changed)
    with pytest.raises(ValueError, match=message):
        _validate(
            changed,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )


def test_unsealed_tampering_is_rejected_by_digest(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    changed = copy.deepcopy(plan)
    changed["instance_boundary"]["machine_type"] = "c4-standard-16"
    with pytest.raises(ValueError, match="digest changed"):
        _validate(
            changed,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )


def test_tampered_resealed_deployment_is_not_accepted(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    changed_deployment = copy.deepcopy(deployment)
    changed_deployment["instances"][0]["instance_name"] = (
        "r2d-s2b-c-a0-forged000000"
    )
    body = dict(changed_deployment)
    body.pop("deployment_contract_sha256")
    changed_deployment["deployment_contract_sha256"] = (
        deployment_v2.canonical_sha256(body)
    )
    with pytest.raises(ValueError, match="deployment contract changed"):
        _validate(
            plan,
            deployment=changed_deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )


def test_fresh_nonce_changes_controller_instances_and_result_prefix(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
    deployment: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    second_deployment = deployment_v2.build_step12b_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=SECOND_RUN_NONCE,
    )
    second_plan = subject.build_step12b_phase2_iam_plan(
        second_deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=SECOND_RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=EXPIRES,
    )
    assert second_plan["source_deployment"][
        "controller_service_account"
    ] != plan["source_deployment"]["controller_service_account"]
    assert second_plan["instance_boundary"][
        "authorized_instance_names"
    ] != plan["instance_boundary"]["authorized_instance_names"]
    assert second_plan["source_deployment"][
        "deployment_result_prefix"
    ] != plan["source_deployment"]["deployment_result_prefix"]
    assert second_plan["plan_sha256"] != plan["plan_sha256"]


def test_exact_custom_role_and_binding_readbacks_seal_and_validate(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    receipt = subject.build_step12b_phase2_iam_readback_receipt(
        plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        observed_at_unix_seconds=ISSUED + 30,
        custom_role_readbacks=_role_readbacks(plan),
        binding_readbacks=_binding_readbacks(plan),
        unexpected_targeted_bindings=[],
    )
    assert receipt["schema"] == subject.READBACK_RECEIPT_SCHEMA
    assert receipt["controller_binding_count"] == 5
    assert receipt["worker_binding_count"] == 3
    assert receipt["all_existing_custom_roles_exact"] is True
    assert receipt["all_phase2_bindings_present_exactly_once"] is True
    assert receipt["get_only_observation"] is True
    assert receipt["cloud_mutation_performed_by_validator"] is False
    assert subject.validate_step12b_phase2_iam_readback_receipt(
        receipt,
        plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    ) == receipt


def test_custom_role_permission_readback_drift_fails_closed(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    roles = _role_readbacks(plan)
    roles[0]["included_permissions"].append("storage.objects.list")
    with pytest.raises(ValueError, match="custom-role readback changed"):
        subject.build_step12b_phase2_iam_readback_receipt(
            plan,
            deployment_contract=deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            observed_at_unix_seconds=ISSUED + 30,
            custom_role_readbacks=roles,
            binding_readbacks=_binding_readbacks(plan),
            unexpected_targeted_bindings=[],
        )


def test_binding_condition_or_occurrence_readback_drift_fails_closed(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    readbacks = _binding_readbacks(plan)
    readbacks[0]["member_occurrences"] = 2
    with pytest.raises(ValueError, match="binding readback changed"):
        subject.build_step12b_phase2_iam_readback_receipt(
            plan,
            deployment_contract=deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            observed_at_unix_seconds=ISSUED + 30,
            custom_role_readbacks=_role_readbacks(plan),
            binding_readbacks=readbacks,
            unexpected_targeted_bindings=[],
        )
    readbacks = _binding_readbacks(plan)
    readbacks[0]["condition"]["expression"] += " && false"
    with pytest.raises(ValueError, match="binding readback changed"):
        subject.build_step12b_phase2_iam_readback_receipt(
            plan,
            deployment_contract=deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
            observed_at_unix_seconds=ISSUED + 30,
            custom_role_readbacks=_role_readbacks(plan),
            binding_readbacks=readbacks,
            unexpected_targeted_bindings=[],
        )


def test_unexpected_targeted_binding_and_expired_readback_fail_closed(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    candidate, reference = payload_contracts
    kwargs = {
        "deployment_contract": deployment,
        "candidate_payload_contract": candidate,
        "reference_payload_contract": reference,
        "controller_public_key_record": controller_public_key_record,
        "run_nonce": RUN_NONCE,
        "custom_role_readbacks": _role_readbacks(plan),
        "binding_readbacks": _binding_readbacks(plan),
    }
    with pytest.raises(ValueError, match="unexpected targeted"):
        subject.build_step12b_phase2_iam_readback_receipt(
            plan,
            observed_at_unix_seconds=ISSUED + 30,
            unexpected_targeted_bindings=[
                {"role": "roles/owner", "member": "unexpected"}
            ],
            **kwargs,
        )
    with pytest.raises(ValueError, match="outside authorization window"):
        subject.build_step12b_phase2_iam_readback_receipt(
            plan,
            observed_at_unix_seconds=EXPIRES,
            unexpected_targeted_bindings=[],
            **kwargs,
        )
