#!/usr/bin/env python3
"""Recover and validate the already-published Step11 fix3 result.

This is a read-only cloud recovery path for the single VM that was already
launched.  It cannot create a VM, delete a resource, or change IAM.  Cloud
access is restricted to:

* generation-pinned GCS reads of the frozen result tree,
* GETs for the exact Step11 instance and disk, and
* IAM policy reads for the four frozen Step11 policy targets.

The only writes are a fresh local materialization directory and a distinct
late-recovery receipt beneath the existing Step11 output root.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import (  # noqa: E402
    run_hu_m31_t3_step6d_rearm2_diagnostic_step11_one_vm_v1 as runner,
)


OUTPUT_ROOT = runner.OUTPUT_ROOT
DESTINATION_ROOT = OUTPUT_ROOT / "late_received"
RECEIPT_PATH = OUTPUT_ROOT / "late_done_recovery_receipt.json"
RECOVERY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step11_fix3_late_done_recovery_receipt_v1"
)
RECOVERY_STATUS = (
    "step11_fix3_late_done_received_validated_after_self_delete_race"
)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required audit artifact is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"required audit artifact is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"required audit artifact is not a JSON object: {path}")
    return value


def _validate_embedded_sha(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    candidate = dict(value)
    digest = candidate.pop(field, None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or runner.controller.canonical_sha256(candidate) != digest
    ):
        raise ValueError(f"{label} canonical digest changed")
    return digest


def _validate_local_preflight(
    value: Mapping[str, Any],
    *,
    contract_sha256: str,
) -> None:
    if (
        value.get("schema")
        != "hu_m31_t3_step6d_rearm2_diagnostic_10c2_local_preflight_v1"
        or value.get("status")
        != "local_precontent_and_safe_extract_pass_cloud_not_authorized"
        or value.get("contract_sha256") != contract_sha256
        or value.get("cloud_write_performed") is not False
        or value.get("network_operation_performed") is not False
        or value.get("claim_or_authorization_written") is not False
        or value.get("launch_ready") is not False
    ):
        raise ValueError("local preflight receipt changed")


def _validate_post_claim_receipt(value: Mapping[str, Any]) -> str:
    digest = _validate_embedded_sha(
        value,
        field="receipt_sha256",
        label="post-claim revoke receipt",
    )
    if (
        value.get("schema") != runner.cloud.POST_CLAIM_REVOKE_SCHEMA
        or value.get("status")
        != "launch_and_worker_actas_removed_after_claim"
        or value.get("instance_name") != runner.EXPECTED_INSTANCE_NAME
        or value.get("launch_binding_removed") is not True
        or value.get("worker_actas_binding_removed") is not True
        or value.get("readback_verified") is not True
        or value.get("cloud_mutation_performed") is not True
        or not isinstance(value.get("provider_instance_id"), str)
        or not value["provider_instance_id"].isdigit()
    ):
        raise ValueError("post-claim revoke receipt changed")
    return digest


def _validate_cleanup_receipt(value: Mapping[str, Any]) -> str:
    digest = _validate_embedded_sha(
        value,
        field="receipt_sha256",
        label="final cleanup receipt",
    )
    if (
        value.get("schema")
        != (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_final_cloud_cleanup_v1"
        )
        or value.get("status")
        != "exact_vm_disk_and_all_step11_bindings_absent"
        or value.get("instance_name") != runner.EXPECTED_INSTANCE_NAME
        or value.get("zone") != runner.transport.ZONE
        or value.get("instance_cleanup", {}).get("provider_get_status") != 404
        or value.get("disk_cleanup", {}).get("provider_get_status") != 404
        or value.get("project_targeted_bindings") != []
        or value.get("bucket_targeted_bindings") != []
        or value.get("worker_service_account_targeted_bindings") != []
        or value.get("controller_service_account_targeted_bindings") != []
        or value.get("package_and_results_retained") is not True
        or value.get("current_profile_changed") is not False
    ):
        raise ValueError("final cleanup receipt changed")
    return digest


def _compute_absence(
    client: runner.ActiveUserExactCloudClient,
) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for resource_kind in ("instances", "disks"):
        response = client.request(
            method="GET",
            url=runner._exact_compute_url(resource_kind),
        )
        statuses[resource_kind] = response.status
        if response.status != 404:
            if response.status == 200:
                runner._validate_exact_compute_resource(
                    response,
                    resource_kind=resource_kind,
                )
            raise RuntimeError(
                f"exact Step11 {resource_kind} is not absent: "
                f"HTTP {response.status}"
            )
    return statuses


def _iam_absence(
    admin: runner.rest_iam.Step11RestIamAdmin,
) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = {}
    for target in runner.rest_iam.PolicyTarget:
        policy = admin.get_policy(target)
        matches[target.value] = runner._matching_targeted_bindings(
            policy,
            target=target,
        )
    residual = {
        target: bindings for target, bindings in matches.items() if bindings
    }
    if residual:
        raise RuntimeError(
            "targeted Step11 IAM bindings remain after final cleanup"
        )
    return matches


def _load_and_validate_frozen_inputs() -> dict[str, Any]:
    contract = runner.transport.validate_job_contract(
        _load_json_object(OUTPUT_ROOT / "transport_contract.json")
    )
    contract_sha256 = runner.controller.canonical_sha256(contract)
    if (
        contract["outer_package_manifest"]["outer_package_identity_sha256"]
        != runner.EXPECTED_OUTER_IDENTITY
        or contract["direct_stage_identity_sha256"]
        != runner.EXPECTED_DIRECT_IDENTITY
        or contract["metadata_binding"]["instance_name"]
        != runner.EXPECTED_INSTANCE_NAME
    ):
        raise ValueError("frozen Step11 identity changed")

    public_key = runner.transport.validate_rsa_public_key_record(
        _load_json_object(OUTPUT_ROOT / "controller_public_key.json")
    )
    launch_contract = runner.launch.validate_launch_contract(
        _load_json_object(OUTPUT_ROOT / "launch_contract.json"),
        transport_contract=contract,
        controller_public_key_record=public_key,
        prebootstrap_path=runner.PREBOOTSTRAP_PATH,
        startup_path=runner.STARTUP_PATH,
    )
    gate_plan = runner.iam_gate.validate_step11_gate_plan(
        _load_json_object(OUTPUT_ROOT / "iam_gate_plan.json")
    )
    local_preflight = _load_json_object(
        OUTPUT_ROOT / "local_preflight_receipt.json"
    )
    _validate_local_preflight(
        local_preflight,
        contract_sha256=contract_sha256,
    )

    package_receipt = runner.cloud.validate_package_provision_receipt(
        transport_contract=contract,
        outer_root=OUTPUT_ROOT / "local_preflight" / "outer",
        receipt=_load_json_object(OUTPUT_ROOT / "package_provision_receipt.json"),
    )
    actual_preflight = _load_json_object(
        OUTPUT_ROOT / "actual_readonly_preflight.json"
    )
    actual_preflight = runner.cloud.validate_authoritative_preflight(
        artifact=actual_preflight,
        transport_contract=contract,
        now_unix_seconds=actual_preflight["retrieved_at_unix_seconds"],
    )
    live_iam_evidence = _load_json_object(
        OUTPUT_ROOT / "live_iam_evidence.json"
    )
    _validate_embedded_sha(
        live_iam_evidence,
        field="evidence_sha256",
        label="live IAM evidence",
    )
    shared_exception = runner.cloud.validate_shared_project_exception(
        _load_json_object(OUTPUT_ROOT / "shared_project_exception.json"),
        transport_contract=contract,
        actual_preflight_artifact=actual_preflight,
        gate_plan=gate_plan,
        live_iam_evidence=live_iam_evidence,
        now_unix_seconds=live_iam_evidence["collected_at_unix_seconds"],
    )
    post_claim = _load_json_object(
        OUTPUT_ROOT / "post_claim_revoke_receipt.json"
    )
    _validate_post_claim_receipt(post_claim)
    cleanup = _load_json_object(
        OUTPUT_ROOT / "final_cloud_cleanup_receipt.json"
    )
    _validate_cleanup_receipt(cleanup)

    return {
        "contract": contract,
        "public_key": public_key,
        "launch_contract": launch_contract,
        "gate_plan": gate_plan,
        "local_preflight": local_preflight,
        "package_receipt": package_receipt,
        "actual_preflight": actual_preflight,
        "live_iam_evidence": live_iam_evidence,
        "shared_exception": shared_exception,
        "post_claim": post_claim,
        "cleanup": cleanup,
    }


def main() -> int:
    runner._assert_policy_registry_unchanged()
    if not OUTPUT_ROOT.is_dir() or OUTPUT_ROOT.is_symlink():
        raise ValueError("existing Step11 output root is not a real directory")
    if DESTINATION_ROOT.exists() or DESTINATION_ROOT.is_symlink():
        raise FileExistsError(
            f"late recovery destination is not fresh: {DESTINATION_ROOT}"
        )
    if RECEIPT_PATH.exists() or RECEIPT_PATH.is_symlink():
        raise FileExistsError(
            f"late recovery receipt already exists: {RECEIPT_PATH}"
        )

    frozen = _load_and_validate_frozen_inputs()
    contract = frozen["contract"]
    preview = contract["adapter_preview"]
    if (
        preview["selected_job_ids"] != ["candidate-shard-00"]
        or preview["vm_count"] != 1
        or len(preview["jobs"]) != 1
    ):
        raise ValueError("late recovery escaped the exact one-VM job")

    user_token = runner.ActiveUserToken()
    user_token.access_token()
    read_client = runner.ActiveUserReadOnlyClient(user_token)
    compute_client = runner.ActiveUserExactCloudClient(user_token)
    iam_admin = runner.rest_iam.Step11RestIamAdmin(
        http_client=runner.rest_iam.StdlibJsonHttpsClient(),
        user_token_source=user_token,
        project=runner.transport.PROJECT,
        bucket=runner.transport.BUCKET,
        worker_service_account=runner.transport.WORKER_SERVICE_ACCOUNT,
        controller_service_account=runner.controller.CONTROLLER_SERVICE_ACCOUNT,
    )

    job = preview["jobs"][0]
    direct_layout = (
        runner.cloud.receiver.build_direct_v1_remote_layout(
            preview,
            outer_package_manifest=contract["outer_package_manifest"],
            direct_stage_identity=contract["direct_stage_identity"],
        )
    )
    direct_job = direct_layout["job_layouts"][0]
    transport_direct_job = contract["remote_layout"]["jobs"][0]
    if (
        direct_job["job_id"] != job["job_id"]
        or direct_job["done_envelope_uri"]
        != transport_direct_job["done_uri"]
        or direct_job["tree_prefix"] != transport_direct_job["tree_prefix"]
        or direct_job["legacy_tree_prefix"] != job["tree_prefix"]
        or direct_job["done_envelope_uri"] == job["done_uri"]
    ):
        raise ValueError("canonical legacy-to-direct result mapping changed")
    pinned_done = runner.cloud.read_generation_pinned_object(
        client=read_client,
        uri=direct_job["done_envelope_uri"],
        allow_missing=False,
    )
    assert pinned_done is not None
    done_record, raw_done = pinned_done
    try:
        done_value = json.loads(raw_done.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("generation-pinned DONE envelope is not JSON") from error
    if not isinstance(done_value, dict):
        raise ValueError("generation-pinned DONE envelope is not an object")
    receive = runner.cloud.adapter.build_receive(
        preview,
        done_records=[done_value],
    )

    tree_done_entry = next(
        (
            row
            for row in done_value["tree_object_records"]
            if row["path"] == "DONE.json"
        ),
        None,
    )
    if tree_done_entry is None:
        raise ValueError("DONE envelope omitted the runner tree DONE object")
    direct_tree_done_uri = f"{direct_job['tree_prefix']}/DONE.json"
    if tree_done_entry["uri"] != f"{direct_job['legacy_tree_prefix']}/DONE.json":
        raise ValueError("legacy DONE tree provenance URI changed")
    pinned_tree_done = runner.cloud.read_generation_pinned_object(
        client=read_client,
        uri=direct_tree_done_uri,
        allow_missing=False,
    )
    assert pinned_tree_done is not None
    tree_done_record, raw_tree_done = pinned_tree_done
    if (
        tree_done_record["bytes"] != tree_done_entry["bytes"]
        or tree_done_record["sha256"] != tree_done_entry["sha256"]
    ):
        raise ValueError("generation-pinned runner tree DONE identity changed")

    existing_done = OUTPUT_ROOT / "late_done_envelope.json"
    existing_tree_done = OUTPUT_ROOT / "late_tree_done.json"
    if (
        not existing_done.is_file()
        or existing_done.is_symlink()
        or hashlib.sha256(existing_done.read_bytes()).hexdigest()
        != done_record["sha256"]
        or not existing_tree_done.is_file()
        or existing_tree_done.is_symlink()
        or hashlib.sha256(existing_tree_done.read_bytes()).hexdigest()
        != tree_done_record["sha256"]
    ):
        raise ValueError("previous late-DONE downloads disagree with pinned reads")

    backend = runner.cloud.GenerationPinnedGcsBackend(
        read_client=read_client,
        list_client=read_client,
        allowed_prefixes=[direct_job["tree_prefix"]],
    )
    materialization = (
        runner.cloud.receiver.materialize_and_validate_received_stage(
            preview,
            receive=receive,
            destination_root=DESTINATION_ROOT,
            backend=backend,
            outer_package_manifest=contract["outer_package_manifest"],
            direct_stage_identity=contract["direct_stage_identity"],
        )
    )
    materialization = runner.cloud._validate_cloud_materialization(
        value=materialization,
        transport_contract=contract,
        receive_record=receive,
    )

    materialized_tree_done = (
        DESTINATION_ROOT / "jobs" / job["job_id"] / "DONE.json"
    )
    if (
        not materialized_tree_done.is_file()
        or materialized_tree_done.is_symlink()
        or hashlib.sha256(materialized_tree_done.read_bytes()).hexdigest()
        != tree_done_record["sha256"]
    ):
        raise ValueError("materialized runner tree DONE identity changed")

    compute_status = _compute_absence(compute_client)
    iam_matches = _iam_absence(iam_admin)
    runner._assert_policy_registry_unchanged()

    completed = done_value["completed_hand_indices"]
    if (
        completed != job["work_hand_indices"]
        or len(completed) != 10
        or len(done_value["heartbeat_sha256s"]) != 10
        or len(done_value["upload_sha256s"]) != 10
        or done_value.get("runner_content_validation_deferred_until_real_receive")
        is not True
        or materialization["runner_validate_completed_output_performed"] is not True
    ):
        raise ValueError("late DONE completion evidence changed")

    input_artifacts = {
        name: runner.controller.canonical_sha256(value)
        for name, value in (
            ("transport_contract", frozen["contract"]),
            ("controller_public_key", frozen["public_key"]),
            ("launch_contract", frozen["launch_contract"]),
            ("iam_gate_plan", frozen["gate_plan"]),
            ("local_preflight_receipt", frozen["local_preflight"]),
            ("package_provision_receipt", frozen["package_receipt"]),
            ("actual_readonly_preflight", frozen["actual_preflight"]),
            ("live_iam_evidence", frozen["live_iam_evidence"]),
            ("shared_project_exception", frozen["shared_exception"]),
            ("post_claim_revoke_receipt", frozen["post_claim"]),
            ("final_cloud_cleanup_receipt", frozen["cleanup"]),
        )
    }
    receipt_body = {
        "schema": RECOVERY_SCHEMA,
        "status": RECOVERY_STATUS,
        "diagnostic_only": True,
        "recovery_reason": (
            "controller_polled_legacy_adapter_result_namespace_while_worker_"
            "published_to_canonical_direct_v1_namespace_then_self_deleted"
        ),
        "failure_classification": (
            "controller_result_namespace_drift_with_terminal_state_race"
        ),
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": job["job_id"],
        "source_role": job["source_role"],
        "attempt_index": preview["attempt_index"],
        "authorized_vm_count": 1,
        "additional_vm_created": False,
        "cloud_mutation_performed_by_recovery": False,
        "transport_contract_sha256": runner.controller.canonical_sha256(
            contract
        ),
        "outer_package_identity_sha256": runner.EXPECTED_OUTER_IDENTITY,
        "direct_stage_identity_sha256": runner.EXPECTED_DIRECT_IDENTITY,
        "package_provision_receipt_sha256": frozen["package_receipt"][
            "receipt_sha256"
        ],
        "post_claim_revoke_receipt_sha256": frozen["post_claim"][
            "receipt_sha256"
        ],
        "final_cloud_cleanup_receipt_sha256": frozen["cleanup"][
            "receipt_sha256"
        ],
        "provider_instance_id": frozen["post_claim"]["provider_instance_id"],
        "instance_name": runner.EXPECTED_INSTANCE_NAME,
        "zone": runner.transport.ZONE,
        "done_envelope_generation_record": done_record,
        "tree_done_generation_record": tree_done_record,
        "legacy_adapter_done_uri": job["done_uri"],
        "canonical_direct_done_uri": direct_job["done_envelope_uri"],
        "legacy_adapter_tree_prefix": job["tree_prefix"],
        "canonical_direct_tree_prefix": direct_job["tree_prefix"],
        "legacy_to_direct_mapping_validated": True,
        "done_envelope_canonical_sha256": runner.controller.canonical_sha256(
            done_value
        ),
        "receive_sha256": runner.controller.canonical_sha256(receive),
        "materialization_sha256": runner.controller.canonical_sha256(
            materialization
        ),
        "materialization_destination": str(
            DESTINATION_ROOT.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "generation_pinned_receive_performed": True,
        "exact_prefix_inventory_verified": True,
        "download_bytes_and_hash_verified": True,
        "runner_content_validated": True,
        "completed_hand_indices": completed,
        "completed_hand_count": len(completed),
        "heartbeat_count": len(done_value["heartbeat_sha256s"]),
        "upload_count": len(done_value["upload_sha256s"]),
        "tree_object_count": len(done_value["tree_object_records"]),
        "compute_provider_get_status": compute_status,
        "provider_instance_and_disk_get_404": compute_status
        == {"instances": 404, "disks": 404},
        "targeted_iam_bindings": iam_matches,
        "all_step11_targeted_iam_bindings_absent": all(
            not bindings for bindings in iam_matches.values()
        ),
        "package_and_results_retained": True,
        "policy_registry_sha256": runner.EXPECTED_POLICY_REGISTRY_SHA256,
        "current_profile_changed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "input_artifact_canonical_sha256": input_artifacts,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": runner.controller.canonical_sha256(receipt_body),
    }
    runner.controller.exclusive_write_json(RECEIPT_PATH, receipt)
    artifact_file_sha256 = hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest()
    sys.stdout.write(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "artifact_file_sha256": artifact_file_sha256,
                "done_generation": done_record["generation"],
                "tree_done_generation": tree_done_record["generation"],
                "completed_hand_count": receipt["completed_hand_count"],
                "provider_instance_and_disk_get_404": receipt[
                    "provider_instance_and_disk_get_404"
                ],
                "all_step11_targeted_iam_bindings_absent": receipt[
                    "all_step11_targeted_iam_bindings_absent"
                ],
                "additional_vm_created": False,
                "output": str(RECEIPT_PATH),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
