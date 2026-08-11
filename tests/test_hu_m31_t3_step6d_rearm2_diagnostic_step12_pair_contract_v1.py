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
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as step11_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as step11_launch,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1 as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_iam_capacity_gate_v1
    as gate,
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
EXPECTED_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
EXPECTED_OUTER_IDENTITY = (
    "593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548"
)
EXPECTED_STAGE2_DIRECT_IDENTITY = (
    "b70bcad3f7df8721d544f45fcd4bc513ce27ea6c199f0cb1a5adc749e1d029e4"
)
EXPECTED_INSTANCE_NAMES = [
    "r2d-10c2-s2-candidate-01-a0-29e3c6f8",
    "r2d-10c2-s2-reference-01-a0-29e3c6f8",
]
NAT_ROUTER_RESOURCE = (
    "projects/ofc-solver-485418/regions/asia-northeast1/"
    "routers/ofc-t3-nat-router-asia-northeast1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _role_neutral_roots(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "source_role"}
        for row in job["root_records"]
    ]


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    recovery = _read_json(STEP11_ROOT / "late_done_recovery_receipt.json")
    public_key = _read_json(STEP11_ROOT / "controller_public_key.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"],
        done_records=[done],
    )
    assert (
        step11_controller.canonical_sha256(stage1_receive)
        == recovery["receive_sha256"]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    contracts = []
    for job_id in plan.STAGE2_JOB_IDS:
        contracts.append(
            transport.build_job_contract(
                package_dir=PACKAGE_DIR,
                stage_id=plan.STAGE2_ID,
                job_id=job_id,
                attempt_index=0,
                offline_wheel_record=wheel,
                controller_public_key_record=public_key,
                prerequisite_stage1_preview=stage1["adapter_preview"],
                prerequisite_stage1_receive=stage1_receive,
            )
        )
    return contracts[0], contracts[1], recovery


@pytest.fixture(scope="module")
def pair_contract(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    candidate, reference, recovery = frozen_inputs
    return subject.build_pair_contract(
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        stage1_recovery_receipt=recovery,
    )


@pytest.fixture(scope="module")
def gate_plan(
    pair_contract: dict[str, Any],
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    candidate, reference, recovery = frozen_inputs
    return gate.build_step12_pair_gate_plan(
        pair_contract,
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        stage1_recovery_receipt=recovery,
        issued_at_unix_seconds=1_753_000_000,
        expires_at_unix_seconds=1_753_003_600,
        nat_router_resource=NAT_ROUTER_RESOURCE,
    )


def test_pair_contract_binds_exact_stage2_attempt0_topology(
    pair_contract: dict[str, Any],
) -> None:
    assert pair_contract["schema"] == subject.SCHEMA
    assert pair_contract["status"] == subject.STATUS
    assert pair_contract["stage_id"] == plan.STAGE2_ID
    assert pair_contract["run_name"] == plan.STAGE2_RUN_NAME
    assert pair_contract["selected_job_ids"] == list(plan.STAGE2_JOB_IDS)
    assert pair_contract["source_roles"] == ["candidate", "reference"]
    assert pair_contract["vm_count"] == 2
    assert pair_contract["attempt_index"] == 0

    topology = pair_contract["execution_topology"]
    assert topology == {
        "actual_machine_type": "c4-standard-8",
        "actual_vcpus_per_vm": 8,
        "vm_count": 2,
        "total_actual_vcpus": 16,
        "process_count_per_vm": 1,
        "inner_declared_machine_type": "c4-standard-16",
        "inner_rayon_threads_per_process": 16,
        "cpu_oversubscribed": True,
        "oversubscription_ratio": 2.0,
        "max_concurrent_vms": 2,
        "attempt_limit_per_job": 1,
        "diagnostic_only": True,
        "admissible_as_performance_evidence": False,
    }

    underlying = pair_contract["underlying_frozen_identity"]
    assert underlying["outer_package_identity_sha256"] == EXPECTED_OUTER_IDENTITY
    assert (
        underlying["direct_stage_identity_sha256"]
        == EXPECTED_STAGE2_DIRECT_IDENTITY
    )
    assert underlying["inner_machine_type"] == "c4-standard-16"
    assert underlying["inner_rayon_num_threads"] == "16"
    assert underlying["immutable_fix3_outer_package_reused"] is True
    assert underlying["underlying_direct_identity_changed"] is False


def test_pair_contract_uses_identical_roots_but_isolates_jobs_and_direct_results(
    pair_contract: dict[str, Any],
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    candidate, reference, _recovery = frozen_inputs
    candidate_job = candidate["adapter_preview"]["jobs"][0]
    reference_job = reference["adapter_preview"]["jobs"][1]
    assert candidate_job["work_hand_indices"] == reference_job[
        "work_hand_indices"
    ] == list(plan.STAGE2_HAND_INDICES)
    assert _role_neutral_roots(candidate_job) == _role_neutral_roots(
        reference_job
    )
    assert {
        row["source_role"] for row in candidate_job["root_records"]
    } == {"candidate"}
    assert {
        row["source_role"] for row in reference_job["root_records"]
    } == {"reference"}
    assert candidate_job["source_role"] == "candidate"
    assert reference_job["source_role"] == "reference"
    assert (
        candidate_job["runner_job_manifest"]["sha256"]
        != reference_job["runner_job_manifest"]["sha256"]
    )

    instances = pair_contract["instances"]
    assert [row["job_id"] for row in instances] == list(plan.STAGE2_JOB_IDS)
    assert [row["source_role"] for row in instances] == [
        "candidate",
        "reference",
    ]
    assert [row["instance_name"] for row in instances] == EXPECTED_INSTANCE_NAMES
    assert all(row["attempt_index"] == 0 for row in instances)
    assert all(row["machine_type"] == "c4-standard-8" for row in instances)
    assert len({row["instance_name"] for row in instances}) == 2
    assert len({row["result_prefix"] for row in instances}) == 2
    assert len({row["tree_prefix"] for row in instances}) == 2
    assert len({row["done_uri"] for row in instances}) == 2
    assert all(
        "/hu-m31-r2diag-direct-v1/stages/" in row["done_uri"]
        for row in instances
    )
    assert all(
        "/hu-m31-r2diag-worker-v1/" not in row["done_uri"]
        for row in instances
    )
    assert pair_contract["result_namespace"]["authoritative_source"] == (
        "remote_layout_direct_v1"
    )
    assert pair_contract["result_namespace"][
        "adapter_preview_result_uris_runtime_authoritative"
    ] is False
    assert pair_contract["result_namespace"][
        "entire_stage_prefix_must_be_empty_before_attempt0"
    ] is True
    assert pair_contract["result_namespace"]["unknown_object_is_fatal"] is True
    assert pair_contract["result_namespace"][
        "permission_or_list_error_is_fatal"
    ] is True


def test_pair_contract_preserves_diagnostic_only_evidence_boundary(
    pair_contract: dict[str, Any],
) -> None:
    assert pair_contract["diagnostic_only"] is True
    assert pair_contract["scientific_payload_present"] is False
    assert pair_contract["current_profile_sha256"] == EXPECTED_PROFILE_SHA256
    assert pair_contract["current_profile_changed"] is False
    assert pair_contract["evidence_admissibility"] == {
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    assert pair_contract["authorization_boundary"]["vm_limit"] == 2
    assert pair_contract["authorization_boundary"]["attempt_limit_per_job"] == 1
    assert pair_contract["authorization_boundary"]["attempt1_authorized"] is False
    assert pair_contract["authorization_boundary"]["resume_authorized"] is False
    assert pair_contract["authorization_boundary"]["third_vm_authorized"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("attempt_index",), 1),
        (("vm_count",), 1),
        (("selected_job_ids",), list(reversed(plan.STAGE2_JOB_IDS))),
        (("execution_topology", "actual_machine_type"), "c4-standard-16"),
        (("execution_topology", "inner_rayon_threads_per_process"), 8),
        (("execution_topology", "cpu_oversubscribed"), False),
        (("instances", 0, "instance_name"), EXPECTED_INSTANCE_NAMES[1]),
        (("instances", 0, "machine_type"), "c4-standard-16"),
        (
            ("instances", 0, "done_uri"),
            "gs://example/hu-m31-r2diag-worker-v1/wrong/DONE.envelope.json",
        ),
        (
            ("result_namespace", "authoritative_source"),
            "adapter_preview_worker_v1",
        ),
        (("evidence_admissibility", "performance_lock_evidence"), True),
        (("evidence_admissibility", "training_eligible"), True),
        (("authorization_boundary", "attempt1_authorized"), True),
        (("authorization_boundary", "third_vm_authorized"), True),
        (("current_profile_changed",), True),
    ],
)
def test_pair_contract_validator_fails_closed_on_topology_or_boundary_drift(
    pair_contract: dict[str, Any],
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    candidate, reference, recovery = frozen_inputs
    changed = copy.deepcopy(pair_contract)
    cursor: Any = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError):
        subject.validate_pair_contract(
            changed,
            candidate_transport_contract=candidate,
            reference_transport_contract=reference,
            stage1_recovery_receipt=recovery,
        )


def test_pair_contract_rejects_wrong_order_attempt_or_stage1_proof(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    candidate, reference, recovery = frozen_inputs
    with pytest.raises(ValueError):
        subject.build_pair_contract(
            candidate_transport_contract=reference,
            reference_transport_contract=candidate,
            stage1_recovery_receipt=recovery,
        )

    changed_candidate = copy.deepcopy(candidate)
    changed_candidate["adapter_preview"]["attempt_index"] = 1
    with pytest.raises(ValueError):
        subject.build_pair_contract(
            candidate_transport_contract=changed_candidate,
            reference_transport_contract=reference,
            stage1_recovery_receipt=recovery,
        )

    changed_recovery = copy.deepcopy(recovery)
    changed_recovery["runner_content_validated"] = False
    with pytest.raises(ValueError):
        subject.build_pair_contract(
            candidate_transport_contract=candidate,
            reference_transport_contract=reference,
            stage1_recovery_receipt=changed_recovery,
        )


def test_step11_frozen_machine_and_runtime_contract_are_not_mutated(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    candidate, reference, _recovery = frozen_inputs
    assert transport.MACHINE_TYPE == "c4-standard-16"
    assert step11_launch.MACHINE_TYPE == "c4-standard-16"
    for contract in (candidate, reference):
        assert (
            contract["direct_stage_identity"]["inputs"]["machine_type"]
            == "c4-standard-16"
        )
        assert (
            contract["metadata_binding"]["worker_invocation"][
                "direct_runner_environment"
            ]["RAYON_NUM_THREADS"]
            == "16"
        )
        assert (
            contract["outer_package_manifest"][
                "outer_package_identity_sha256"
            ]
            == EXPECTED_OUTER_IDENTITY
        )


def test_pair_gate_freezes_exact_capacity_attempt_and_revoke_boundary(
    gate_plan: dict[str, Any],
) -> None:
    assert gate_plan["schema"] == gate.SCHEMA
    assert gate_plan["status"] == gate.STATUS
    capacity = gate_plan["capacity_contract"]
    assert capacity["machine_type"] == "c4-standard-8"
    assert capacity["vcpu_per_vm"] == 8
    assert capacity["requested_concurrent_vms"] == 2
    assert capacity["requested_c4_vcpu"] == 16
    assert capacity["quota_metrics"] == [
        "CPUS",
        "CPUS_ALL_REGIONS",
        "CPUS_PER_VM_FAMILY_C4",
        "PREEMPTIBLE_CPUS",
    ]
    assert capacity["authoritative_inventory_endpoints"] == [
        "compute_aggregated_instances",
        "compute_aggregated_reservations",
        "compute_aggregated_node_groups",
        "compute_aggregated_future_reservations",
    ]
    assert capacity["authoritative_inventory_complete_required"] is True
    assert (
        capacity["authoritative_inventory_same_scope_set_required"] is True
    )
    assert (
        capacity["authoritative_inventory_unreachable_count_required"] == 0
    )
    assert (
        capacity[
            "authoritative_inventory_page_tokens_exhausted_required"
        ]
        is True
    )
    assert (
        capacity["authoritative_noninstance_inventory_empty_required"]
        is True
    )
    assert (
        capacity[
            "authoritative_transcript_endpoint_digest_map_required"
        ]
        is True
    )
    assert capacity["target_instance_names"] == EXPECTED_INSTANCE_NAMES
    assert capacity["target_name_collision_forbidden"] is True
    assert capacity["quota_must_cover_current_usage_plus_requested_vcpu"] is True
    assert capacity["stale_quota_evidence_forbidden"] is True

    instance = gate_plan["instance_contract"]
    assert instance["authorized_instance_names"] == EXPECTED_INSTANCE_NAMES
    assert instance["authorized_attempt_index"] == 0
    assert instance["max_concurrent_vms"] == 2
    assert instance["max_attempts_per_job"] == 1
    assert instance["attempt1_authorized"] is False
    assert instance["resume_authorized"] is False
    assert instance["third_vm_authorized"] is False
    assert [row["machine_type"] for row in instance["allowed_launch_requests"]] == [
        "c4-standard-8",
        "c4-standard-8",
    ]

    revoke = gate_plan["post_claim_revoke_contract"]
    assert revoke["required_claim_count"] == 2
    assert revoke["required_claim_instance_names"] == EXPECTED_INSTANCE_NAMES
    assert revoke["revoke_after_claim_count"] == 2
    assert revoke["revoke_after_first_claim_forbidden"] is True
    assert revoke["revoke_before_all_claims_forbidden"] is True
    assert revoke["additional_insert_after_revoke_forbidden"] is True


def test_pair_gate_iam_condition_names_only_the_exact_pair(
    gate_plan: dict[str, Any],
) -> None:
    iam = gate_plan["iam_contract"]
    assert iam["binding_count"] == 10
    assert len(iam["binding_specs"]) == 10
    assert iam["controller_package_create_binding_required"] is False
    expression = iam["instance_condition"]
    for name in EXPECTED_INSTANCE_NAMES:
        full_name = (
            "projects/ofc-solver-485418/zones/asia-northeast1-b/"
            f"instances/{name}"
        )
        assert expression.count(full_name) == 1
    assert "startsWith" not in expression
    assert "r2d-10c2-s2-" not in expression.replace(
        EXPECTED_INSTANCE_NAMES[0], ""
    ).replace(EXPECTED_INSTANCE_NAMES[1], "")
    assert "request.time < timestamp(" in expression


def test_pair_gate_cleanup_is_independent_and_fail_closed(
    gate_plan: dict[str, Any],
) -> None:
    cleanup = gate_plan["cleanup_contract"]
    assert cleanup["required_on_success"] is True
    assert cleanup["required_on_any_failure"] is True
    assert cleanup["exact_instance_names"] == EXPECTED_INSTANCE_NAMES
    assert cleanup["exact_disk_names"] == EXPECTED_INSTANCE_NAMES
    assert cleanup["all_instance_delete_attempts_independent"] is True
    assert cleanup["all_disk_delete_attempts_independent"] is True
    assert cleanup["cleanup_errors_aggregated"] is True
    assert cleanup["iam_revoked_before_user_credential_compute_cleanup"] is True
    assert cleanup["final_instance_get_status_required"] == 404
    assert cleanup["final_disk_get_status_required"] == 404
    assert cleanup["final_targeted_iam_binding_count_required"] == 0
    assert cleanup["current_profile_changed"] is False
    assert gate_plan["gate_semantics"] == {
        "local_validation_only": True,
        "cloud_api_calls_performed": False,
        "cloud_mutation_performed": False,
        "launch_authorized": False,
        "passing_gate_proves_spot_stock": False,
        "fresh_get_only_observation_required_before_launch": True,
        "separate_explicit_execution_confirmation_required": True,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("capacity_contract", "machine_type"), "c4-standard-16"),
        (("capacity_contract", "requested_concurrent_vms"), 1),
        (("capacity_contract", "requested_c4_vcpu"), 8),
        (
            ("capacity_contract", "quota_metrics"),
            ["CPUS", "CPUS_PER_VM_FAMILY_C4", "PREEMPTIBLE_CPUS"],
        ),
        (
            (
                "capacity_contract",
                "authoritative_inventory_complete_required",
            ),
            False,
        ),
        (("instance_contract", "max_attempts_per_job"), 2),
        (("instance_contract", "attempt1_authorized"), True),
        (("post_claim_revoke_contract", "revoke_after_claim_count"), 1),
        (
            ("post_claim_revoke_contract", "revoke_after_first_claim_forbidden"),
            False,
        ),
        (("cleanup_contract", "exact_instance_names"), EXPECTED_INSTANCE_NAMES[:1]),
        (("cleanup_contract", "all_disk_delete_attempts_independent"), False),
        (("cleanup_contract", "final_targeted_iam_binding_count_required"), 1),
        (("gate_semantics", "performance_lock_evidence"), True),
        (("gate_semantics", "launch_authorized"), True),
    ],
)
def test_pair_gate_validator_rejects_security_or_capacity_drift(
    gate_plan: dict[str, Any],
    pair_contract: dict[str, Any],
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    candidate, reference, recovery = frozen_inputs
    changed = copy.deepcopy(gate_plan)
    cursor: Any = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError):
        gate.validate_step12_pair_gate_plan(
            changed,
            pair_contract=pair_contract,
            candidate_transport_contract=candidate,
            reference_transport_contract=reference,
            stage1_recovery_receipt=recovery,
        )
