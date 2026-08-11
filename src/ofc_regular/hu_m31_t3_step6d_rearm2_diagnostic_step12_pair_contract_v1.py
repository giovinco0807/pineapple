"""Offline contract for the Step 12 two-VM diagnostic pair.

Step 12 deliberately reuses the immutable fix3 worker package and its frozen
``1 process x 16 Rayon threads`` runner contract.  The cloud launch topology is
different: one ``c4-standard-8`` VM per source role.  Keeping those two
identities explicit prevents the diagnostic topology override from being
mistaken for performance evidence or from silently rewriting the accepted
worker inputs.

This module performs no cloud query or mutation.  It only validates the two
existing Stage-2 direct-transport contracts and the independently recovered
Stage-1 receipt, then binds their exact attempt-0 identities into one
fail-closed pair contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan


SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1"
STATUS = "offline_pair_contract_separate_cloud_authorization_required"

ACTUAL_MACHINE_TYPE = "c4-standard-8"
ACTUAL_VCPUS_PER_VM = 8
VM_COUNT = 2
PROCESS_COUNT_PER_VM = 1
INNER_MACHINE_TYPE = "c4-standard-16"
INNER_RAYON_THREADS = 16
ATTEMPT_INDEX = 0
ATTEMPT_LIMIT_PER_JOB = 1

EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256 = (
    "593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548"
)
EXPECTED_STAGE1_DIRECT_IDENTITY_SHA256 = (
    "8e27e55dab68ae9697c1c40c80353e8905ad1339423c52d51070f559fbceb44a"
)
EXPECTED_STAGE2_DIRECT_IDENTITY_SHA256 = (
    "b70bcad3f7df8721d544f45fcd4bc513ce27ea6c199f0cb1a5adc749e1d029e4"
)
EXPECTED_CURRENT_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
EXPECTED_STAGE1_RECOVERY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step11_fix3_late_done_recovery_receipt_v1"
)
EXPECTED_STAGE1_RECOVERY_STATUS = (
    "step11_fix3_late_done_received_validated_after_self_delete_race"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIRECT_STAGE_MARKER = "/hu-m31-r2diag-direct-v1/stages/"
_LEGACY_WORKER_MARKER = "/hu-m31-r2diag-worker-v1/"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _validate_stage1_recovery(
    value: Mapping[str, Any],
    *,
    expected_outer_identity: str,
) -> dict[str, Any]:
    receipt = _mapping(value, "Stage1 recovery receipt")
    receipt_sha = _sha(
        receipt.get("receipt_sha256"), "Stage1 recovery receipt SHA-256"
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if canonical_sha256(unsigned) != receipt_sha:
        raise ValueError("Stage1 recovery receipt digest changed")

    targeted = receipt.get("targeted_iam_bindings")
    provider_status = receipt.get("compute_provider_get_status")
    input_artifacts = receipt.get("input_artifact_canonical_sha256")
    done_generation = receipt.get("done_envelope_generation_record")
    tree_generation = receipt.get("tree_done_generation_record")
    if not all(
        isinstance(row, Mapping)
        for row in (
            targeted,
            provider_status,
            input_artifacts,
            done_generation,
            tree_generation,
        )
    ):
        raise ValueError("Stage1 recovery evidence is incomplete")
    expected_targets = {
        "bucket": [],
        "controller_service_account": [],
        "project": [],
        "worker_service_account": [],
    }
    if dict(targeted) != expected_targets:
        raise ValueError("Stage1 temporary IAM bindings are not absent")
    if dict(provider_status) != {"disks": 404, "instances": 404}:
        raise ValueError("Stage1 VM or disk absence is not proven")
    if any(
        not isinstance(name, str)
        or not name
        or _SHA256.fullmatch(str(digest)) is None
        for name, digest in input_artifacts.items()
    ):
        raise ValueError("Stage1 input artifact binding changed")
    for label, record in (
        ("DONE envelope generation", done_generation),
        ("tree DONE generation", tree_generation),
    ):
        if (
            type(record.get("generation")) is not int
            or record["generation"] <= 0
            or type(record.get("metageneration")) is not int
            or record["metageneration"] <= 0
            or type(record.get("bytes")) is not int
            or record["bytes"] <= 0
            or _sha(record.get("sha256"), f"{label} SHA-256")
            != record["sha256"]
            or not isinstance(record.get("uri"), str)
            or _DIRECT_STAGE_MARKER not in record["uri"]
            or _LEGACY_WORKER_MARKER in record["uri"]
        ):
            raise ValueError(f"{label} record changed")

    false_fields = (
        "additional_vm_created",
        "cloud_mutation_performed_by_recovery",
        "current_profile_changed",
        "performance_lock_evidence",
        "promotion_evidence",
        "quality_evidence",
        "training_eligible",
    )
    true_fields = (
        "all_step11_targeted_iam_bindings_absent",
        "diagnostic_only",
        "download_bytes_and_hash_verified",
        "exact_prefix_inventory_verified",
        "generation_pinned_receive_performed",
        "legacy_to_direct_mapping_validated",
        "package_and_results_retained",
        "provider_instance_and_disk_get_404",
        "runner_content_validated",
    )
    if (
        receipt.get("schema") != EXPECTED_STAGE1_RECOVERY_SCHEMA
        or receipt.get("status") != EXPECTED_STAGE1_RECOVERY_STATUS
        or receipt.get("stage_id") != plan.STAGE1_ID
        or receipt.get("run_name") != plan.STAGE1_RUN_NAME
        or receipt.get("job_id") != plan.STAGE1_JOB_IDS[0]
        or receipt.get("source_role") != "candidate"
        or receipt.get("attempt_index") != 0
        or receipt.get("authorized_vm_count") != 1
        or receipt.get("completed_hand_count") != len(plan.STAGE1_HAND_INDICES)
        or receipt.get("completed_hand_indices")
        != list(plan.STAGE1_HAND_INDICES)
        or receipt.get("heartbeat_count") != len(plan.STAGE1_HAND_INDICES)
        or receipt.get("upload_count") != len(plan.STAGE1_HAND_INDICES)
        or receipt.get("outer_package_identity_sha256")
        != expected_outer_identity
        or receipt.get("direct_stage_identity_sha256")
        != EXPECTED_STAGE1_DIRECT_IDENTITY_SHA256
        or receipt.get("policy_registry_sha256")
        != EXPECTED_CURRENT_PROFILE_SHA256
        or any(receipt.get(field) is not False for field in false_fields)
        or any(receipt.get(field) is not True for field in true_fields)
    ):
        raise ValueError("Stage1 recovery prerequisite changed")
    for field in (
        "receive_sha256",
        "materialization_sha256",
        "transport_contract_sha256",
        "package_provision_receipt_sha256",
        "post_claim_revoke_receipt_sha256",
        "final_cloud_cleanup_receipt_sha256",
        "done_envelope_canonical_sha256",
    ):
        _sha(receipt.get(field), f"Stage1 {field}")
    if (
        receipt.get("canonical_direct_done_uri")
        != done_generation.get("uri")
        or receipt.get("canonical_direct_tree_prefix")
        != str(tree_generation.get("uri")).removesuffix("/DONE.json")
    ):
        raise ValueError("Stage1 direct-v1 generation binding changed")
    return receipt


def _validated_pair(
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = transport.validate_job_contract(candidate_transport_contract)
    reference = transport.validate_job_contract(reference_transport_contract)
    contracts = (candidate, reference)
    previews = [row["adapter_preview"] for row in contracts]
    if previews[0] != previews[1]:
        raise ValueError("candidate/reference Stage2 previews differ")
    preview = previews[0]
    if (
        preview.get("stage_id") != plan.STAGE2_ID
        or preview.get("run_name") != plan.STAGE2_RUN_NAME
        or preview.get("selected_job_ids") != list(plan.STAGE2_JOB_IDS)
        or preview.get("vm_count") != VM_COUNT
        or preview.get("attempt_index") != ATTEMPT_INDEX
        or preview.get("max_attempts") != transport.MAX_ATTEMPTS
    ):
        raise ValueError("pair escaped exact Stage2 attempt0")
    jobs = preview.get("jobs")
    root_identities = [
        [
            {
                key: value
                for key, value in root_record.items()
                if key != "source_role"
            }
            for root_record in job["root_records"]
        ]
        for job in jobs
    ] if isinstance(jobs, list) and len(jobs) == VM_COUNT else []
    root_source_roles = [
        [root_record.get("source_role") for root_record in job["root_records"]]
        for job in jobs
    ] if isinstance(jobs, list) and len(jobs) == VM_COUNT else []
    if (
        not isinstance(jobs, list)
        or [row.get("job_id") for row in jobs] != list(plan.STAGE2_JOB_IDS)
        or [row.get("source_role") for row in jobs]
        != ["candidate", "reference"]
        or any(
            row.get("work_hand_indices") != list(plan.STAGE2_HAND_INDICES)
            for row in jobs
        )
        or root_identities[0] != root_identities[1]
        or root_source_roles
        != [
            ["candidate"] * len(plan.STAGE2_HAND_INDICES),
            ["reference"] * len(plan.STAGE2_HAND_INDICES),
        ]
        or jobs[0].get("runner_job_manifest", {}).get("sha256")
        == jobs[1].get("runner_job_manifest", {}).get("sha256")
    ):
        raise ValueError("pair jobs or common roots changed")

    candidate_binding = candidate["metadata_binding"]
    reference_binding = reference["metadata_binding"]
    bindings = (candidate_binding, reference_binding)
    if (
        [row.get("job_id") for row in bindings] != list(plan.STAGE2_JOB_IDS)
        or [row.get("source_role") for row in bindings]
        != ["candidate", "reference"]
        or any(row.get("attempt_index") != ATTEMPT_INDEX for row in bindings)
        or len({row.get("instance_name") for row in bindings}) != VM_COUNT
    ):
        raise ValueError("pair metadata bindings changed")

    direct_sha = candidate.get("direct_stage_identity_sha256")
    outer_sha = candidate.get("outer_package_manifest", {}).get(
        "outer_package_identity_sha256"
    )
    if (
        direct_sha != EXPECTED_STAGE2_DIRECT_IDENTITY_SHA256
        or reference.get("direct_stage_identity_sha256") != direct_sha
        or outer_sha != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or reference.get("outer_package_manifest", {}).get(
            "outer_package_identity_sha256"
        )
        != outer_sha
        or candidate.get("direct_stage_identity")
        != reference.get("direct_stage_identity")
        or candidate.get("remote_layout") != reference.get("remote_layout")
        or candidate.get("outer_package_manifest")
        != reference.get("outer_package_manifest")
    ):
        raise ValueError("pair frozen package or direct identity changed")

    direct_inputs = candidate["direct_stage_identity"].get("inputs")
    layout = candidate["remote_layout"]
    layout_jobs = layout.get("jobs")
    if (
        not isinstance(direct_inputs, Mapping)
        or direct_inputs.get("machine_type") != INNER_MACHINE_TYPE
        or direct_inputs.get("stage_id") != plan.STAGE2_ID
        or direct_inputs.get("run_name") != plan.STAGE2_RUN_NAME
        or not isinstance(layout_jobs, list)
        or [row.get("job_id") for row in layout_jobs]
        != list(plan.STAGE2_JOB_IDS)
        or layout.get("direct_stage_identity_sha256") != direct_sha
    ):
        raise ValueError("pair direct-v1 layout changed")
    for contract, binding, layout_job in zip(
        contracts, bindings, layout_jobs, strict=True
    ):
        environment = binding.get("worker_invocation", {}).get(
            "direct_runner_environment"
        )
        if (
            not isinstance(environment, Mapping)
            or environment.get("RAYON_NUM_THREADS") != str(INNER_RAYON_THREADS)
            or binding.get("job_result_layout") != layout_job
            or contract.get("capabilities", {}).get("current_profile_changed")
            is not False
            or any(
                contract.get("capabilities", {}).get(field) is not False
                for field in (
                    "performance_lock_evidence",
                    "quality_evidence",
                    "training_eligible",
                    "promotion_evidence",
                )
            )
        ):
            raise ValueError("pair frozen runner or evidence boundary changed")
        for field in ("result_prefix", "tree_prefix", "done_uri"):
            uri = layout_job.get(field)
            if (
                not isinstance(uri, str)
                or _DIRECT_STAGE_MARKER not in uri
                or _LEGACY_WORKER_MARKER in uri
            ):
                raise ValueError("pair runtime URI is not authoritative direct-v1")
    if (
        len({row["result_prefix"] for row in layout_jobs}) != VM_COUNT
        or len({row["tree_prefix"] for row in layout_jobs}) != VM_COUNT
        or len({row["done_uri"] for row in layout_jobs}) != VM_COUNT
        or layout.get("stage_prefix_must_be_exactly_empty_before_attempt0")
        is not True
        or layout.get("result_identity_retry_invariant") is not True
    ):
        raise ValueError("pair result namespaces overlap or lost freshness")

    attempt_layout = direct_inputs.get("attempt_layout")
    expected_attempt0 = [
        {
            "job_id": binding["job_id"],
            "attempt_index": ATTEMPT_INDEX,
            "instance_name": binding["instance_name"],
        }
        for binding in bindings
    ]
    if (
        not isinstance(attempt_layout, list)
        or [
            row
            for row in attempt_layout
            if row.get("attempt_index") == ATTEMPT_INDEX
        ]
        != expected_attempt0
    ):
        raise ValueError("pair attempt0 instance layout changed")
    return candidate, reference


def build_pair_contract(
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic, non-authorizing Step 12 pair contract."""

    candidate, reference = _validated_pair(
        candidate_transport_contract, reference_transport_contract
    )
    preview = candidate["adapter_preview"]
    layout = candidate["remote_layout"]
    direct_inputs = candidate["direct_stage_identity"]["inputs"]
    outer_identity = candidate["outer_package_manifest"][
        "outer_package_identity_sha256"
    ]
    recovery = _validate_stage1_recovery(
        stage1_recovery_receipt,
        expected_outer_identity=outer_identity,
    )
    contracts = (candidate, reference)
    layout_jobs = layout["jobs"]

    instances = []
    for contract, layout_job in zip(contracts, layout_jobs, strict=True):
        binding = contract["metadata_binding"]
        adapter_job = next(
            row
            for row in preview["jobs"]
            if row["job_id"] == binding["job_id"]
        )
        instances.append(
            {
                "job_id": binding["job_id"],
                "source_role": binding["source_role"],
                "instance_name": binding["instance_name"],
                "attempt_index": ATTEMPT_INDEX,
                "machine_type": ACTUAL_MACHINE_TYPE,
                "actual_vcpus": ACTUAL_VCPUS_PER_VM,
                "worker_process_count": PROCESS_COUNT_PER_VM,
                "inner_rayon_threads": INNER_RAYON_THREADS,
                "transport_contract_sha256": canonical_sha256(contract),
                "metadata_binding_sha256": contract[
                    "metadata_binding_sha256"
                ],
                "runner_job_manifest_sha256": adapter_job[
                    "runner_job_manifest"
                ]["sha256"],
                "work_hand_indices": list(adapter_job["work_hand_indices"]),
                "root_records_sha256": canonical_sha256(
                    adapter_job["root_records"]
                ),
                "result_prefix": layout_job["result_prefix"],
                "tree_prefix": layout_job["tree_prefix"],
                "done_uri": layout_job["done_uri"],
            }
        )

    topology = {
        "actual_machine_type": ACTUAL_MACHINE_TYPE,
        "actual_vcpus_per_vm": ACTUAL_VCPUS_PER_VM,
        "vm_count": VM_COUNT,
        "total_actual_vcpus": ACTUAL_VCPUS_PER_VM * VM_COUNT,
        "process_count_per_vm": PROCESS_COUNT_PER_VM,
        "inner_declared_machine_type": INNER_MACHINE_TYPE,
        "inner_rayon_threads_per_process": INNER_RAYON_THREADS,
        "cpu_oversubscribed": True,
        "oversubscription_ratio": (
            INNER_RAYON_THREADS / ACTUAL_VCPUS_PER_VM
        ),
        "max_concurrent_vms": VM_COUNT,
        "attempt_limit_per_job": ATTEMPT_LIMIT_PER_JOB,
        "diagnostic_only": True,
        "admissible_as_performance_evidence": False,
    }
    underlying = {
        "outer_package_identity_sha256": outer_identity,
        "outer_package_manifest_sha256": candidate[
            "outer_package_manifest_sha256"
        ],
        "direct_stage_identity_sha256": candidate[
            "direct_stage_identity_sha256"
        ],
        "preview_stage_identity_sha256": preview["stage_identity_sha256"],
        "adapter_preview_sha256": candidate["adapter_preview_sha256"],
        "inner_machine_type": direct_inputs["machine_type"],
        "inner_rayon_num_threads": str(INNER_RAYON_THREADS),
        "immutable_fix3_outer_package_reused": True,
        "underlying_direct_identity_changed": False,
        "actual_compute_shape_is_outer_launch_override": True,
    }
    stage1 = {
        "schema": recovery["schema"],
        "status": recovery["status"],
        "receipt_sha256": recovery["receipt_sha256"],
        "stage_id": recovery["stage_id"],
        "run_name": recovery["run_name"],
        "job_id": recovery["job_id"],
        "outer_package_identity_sha256": recovery[
            "outer_package_identity_sha256"
        ],
        "direct_stage_identity_sha256": recovery[
            "direct_stage_identity_sha256"
        ],
        "receive_sha256": recovery["receive_sha256"],
        "materialization_sha256": recovery["materialization_sha256"],
        "done_envelope_generation": recovery[
            "done_envelope_generation_record"
        ]["generation"],
        "tree_done_generation": recovery[
            "tree_done_generation_record"
        ]["generation"],
        "runner_content_validated": True,
        "generation_pinned_receive_performed": True,
        "provider_instance_and_disk_get_404": True,
        "all_temporary_iam_bindings_absent": True,
        "current_profile_sha256": recovery["policy_registry_sha256"],
        "current_profile_changed": False,
    }
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "stage_id": plan.STAGE2_ID,
        "run_name": plan.STAGE2_RUN_NAME,
        "selected_job_ids": list(plan.STAGE2_JOB_IDS),
        "source_roles": ["candidate", "reference"],
        "vm_count": VM_COUNT,
        "attempt_index": ATTEMPT_INDEX,
        "execution_topology": topology,
        "underlying_frozen_identity": underlying,
        "stage1_prerequisite": stage1,
        "instances": instances,
        "result_namespace": {
            "authoritative_source": "remote_layout_direct_v1",
            "stage_prefix": layout["stage_prefix"],
            "result_prefix": layout["result_prefix"],
            "receive_uri": layout["receive_uri"],
            "direct_stage_identity_sha256": layout[
                "direct_stage_identity_sha256"
            ],
            "adapter_preview_result_uris_runtime_authoritative": False,
            "entire_stage_prefix_must_be_empty_before_attempt0": True,
            "unknown_object_is_fatal": True,
            "permission_or_list_error_is_fatal": True,
            "ordered_done_job_ids": list(plan.STAGE2_JOB_IDS),
        },
        "diagnostic_only": True,
        "scientific_payload_present": False,
        "evidence_admissibility": {
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
        "authorization_boundary": {
            "vm_limit": VM_COUNT,
            "attempt_limit_per_job": ATTEMPT_LIMIT_PER_JOB,
            "attempt1_authorized": False,
            "resume_authorized": False,
            "third_vm_authorized": False,
            "package_write_authorized": False,
            "iam_mutation_authorized": False,
            "vm_create_authorized": False,
            "launch_authorized": False,
            "separate_cloud_authorization_required": True,
            "cloud_mutation_performed": False,
        },
        "current_profile_sha256": EXPECTED_CURRENT_PROFILE_SHA256,
        "current_profile_changed": False,
    }
    return {**body, "pair_contract_sha256": canonical_sha256(body)}


def validate_pair_contract(
    value: Mapping[str, Any],
    *,
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    stage1_recovery_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and exactly compare every pair-contract field."""

    candidate = _mapping(value, "Step12 pair contract")
    supplied_sha = _sha(
        candidate.get("pair_contract_sha256"), "Step12 pair contract SHA-256"
    )
    unsigned = dict(candidate)
    unsigned.pop("pair_contract_sha256", None)
    if canonical_sha256(unsigned) != supplied_sha:
        raise ValueError("Step12 pair contract digest changed")
    expected = build_pair_contract(
        candidate_transport_contract,
        reference_transport_contract,
        stage1_recovery_receipt,
    )
    if candidate != expected:
        raise ValueError("Step12 pair contract changed")
    return expected


__all__ = [
    "ACTUAL_MACHINE_TYPE",
    "ACTUAL_VCPUS_PER_VM",
    "ATTEMPT_INDEX",
    "EXPECTED_CURRENT_PROFILE_SHA256",
    "EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256",
    "EXPECTED_STAGE2_DIRECT_IDENTITY_SHA256",
    "INNER_MACHINE_TYPE",
    "INNER_RAYON_THREADS",
    "SCHEMA",
    "STATUS",
    "VM_COUNT",
    "build_pair_contract",
    "canonical_bytes",
    "canonical_sha256",
    "validate_pair_contract",
]
