"""Read-only, fail-closed receiver for the full100 startup canary.

This receiver is deliberately narrower than the production wave receiver.  It
can prove that ``candidate-shard-00/a00`` completed, that its exact twenty data
objects agree with the last-written ``DONE.json``, and that the owned compute
and worker-IAM lifecycle was closed.  It never creates ``ACCEPTED.json``, never
advances the attempt ledger, and never emits merge, training, quality,
promotion, or performance evidence.

The lifecycle proof is replayed from the controller journal through the same
production receiver adapter used by the full wave.  Result validation reuses
the production receiver's single-job ``_validate_done`` implementation.  No
cloud mutation primitive is reachable from this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_production_cleanup_orchestrator_v2 as cleanup_v2
from . import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as production_receiver_v2
from . import hu_m31_t3_step6d_full100_wave_result_gcs_adapter_v2 as gcs_v2
from . import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as result_receiver_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    _stdlib_http_request,
)


RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_startup_canary_receiver_v1"
)
RECEIPT_STATUS = (
    "candidate_shard_00_a00_done_20_objects_cleanup_revalidated_"
    "diagnostic_only"
)
CANARY_JOB_ID = "candidate-shard-00"
CANARY_SOURCE_ROLE = "candidate"
CANARY_ATTEMPT_ID = "a00"
EXPECTED_DATA_OBJECT_COUNT = 20
CURRENT_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

_CLEANUP_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "execution_namespace",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "cleanup_plan_sha256",
        "execution_manifest_sha256",
        "launch_event_sha256",
        "delete_event_sha256",
        "absence_event_sha256",
        "worker_iam_cleanup_event_sha256",
        "closeout_event_sha256",
        "gce_create_receipt_sha256",
        "gce_delete_receipt_sha256",
        "gce_absence_receipt_sha256",
        "worker_iam_cleanup_receipt_sha256",
        "lifecycle_receipt",
        "receiver_request",
        "receiver_request_sha256",
        "all_owned_instances_absent",
        "all_owned_boot_disks_absent",
        "worker_iam_bindings_absent",
        "shared_content_preserved_for_later_executions",
        "content_cleanup_event_sha256",
        "credentials_from_environment_only",
        "additional_create_authorized",
        "current_profile_changed",
        "manifest_sha256",
    }
)

_LIFECYCLE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "controller_context_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "launch_event_sha256",
        "gce_create_receipt_sha256",
        "actual_launch_receipt_sha256",
        "delete_event_sha256",
        "gce_delete_receipt_sha256",
        "absence_event_sha256",
        "gce_absence_receipt_sha256",
        "worker_iam_cleanup_event_sha256",
        "worker_iam_cleanup_receipt_sha256",
        "worker_iam_bindings_absent",
        "content_cleanup_event_sha256",
        "content_cleanup_receipt_sha256",
        "all_owned_instances_absent",
        "all_owned_boot_disks_absent",
        "additional_create_authorized",
        "attested_at_utc",
        "current_profile_changed",
        "receipt_sha256",
    }
)

_LIFECYCLE_PROOF_FIELDS = frozenset(
    {
        "schema",
        "status",
        "controller_context_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "lifecycle_event_sha256",
        "lifecycle_receipt_sha256",
        "lifecycle_attested_at_utc",
        "launch_event_sha256",
        "delete_event_sha256",
        "absence_event_sha256",
        "worker_iam_cleanup_event_sha256",
        "launch_bundle_sha256",
        "gce_create_receipt",
        "actual_launch_receipt",
        "gce_delete_receipt",
        "gce_absence_receipt",
        "worker_iam_cleanup_receipt",
        "selected_instance_mapping",
        "gce_create_rows",
        "actual_launch_rows",
        "selected_instance_count",
        "exact_created_instance_count",
        "exact_uncreated_instance_count",
        "create_classification",
        "actual_launch_receipt_present",
        "journal_event_count",
        "journal_hash_chain_valid",
        "all_producer_receipts_valid",
        "all_owned_instances_absent",
        "all_owned_boot_disks_absent",
        "worker_iam_bindings_absent",
        "additional_create_authorized",
        "current_profile_changed",
        "proof_sha256",
    }
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "execution_scope",
        "diagnostic_only",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "attempt_prefix",
        "done_record",
        "done_identity_sha256",
        "root_digest",
        "data_object_count",
        "data_records",
        "data_records_sha256",
        "cleanup_manifest_sha256",
        "lifecycle_proof_sha256",
        "cleanup_complete",
        "all_owned_instances_absent",
        "all_owned_boot_disks_absent",
        "worker_iam_bindings_absent",
        "hidden_truth_exposed",
        "opponent_private_discards_used",
        "profile_sha256_before",
        "profile_sha256_after",
        "acceptance_object_observed",
        "acceptance_create_authorized",
        "acceptance_create_performed",
        "attempt_ledger_transition_authorized",
        "attempt_ledger_transition_performed",
        "merge_authorized",
        "merge_evidence",
        "scientific_merge_eligible",
        "training_eligible",
        "performance_evidence",
        "performance_evaluation_eligible",
        "quality_evidence",
        "promotion_evidence",
        "provider_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)


def _strict_clone(value: Mapping[str, Any], fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields changed")
    try:
        clone = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError(f"{label} is not strict JSON") from None
    if not isinstance(clone, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{label} is not an object")
    return clone


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"{label} is not a regular JSON file")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{label} is not readable JSON") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _sha256_file(path: str | Path, label: str) -> str:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canary_selected(
    plan: Mapping[str, Any], resume: Mapping[str, Any]
) -> dict[str, Any]:
    wave_v2.validate_startup_canary_plan(plan)
    selected = [
        deepcopy(dict(row))
        for row in resume["selected_attempts"]
        if row.get("job_id") == CANARY_JOB_ID
    ]
    if len(selected) != 1:
        raise ValueError("startup canary candidate-shard-00 selection changed")
    row = selected[0]
    if (
        row.get("source_role") != CANARY_SOURCE_ROLE
        or row.get("attempt_id") != CANARY_ATTEMPT_ID
        or not isinstance(row.get("instance_id"), str)
        or not row["instance_id"]
        or not isinstance(row.get("artifact_prefix"), str)
        or not row["artifact_prefix"].endswith(
            f"/results/jobs/{CANARY_JOB_ID}/attempts/{CANARY_ATTEMPT_ID}"
        )
    ):
        raise ValueError("startup canary selected tuple changed")
    jobs = {
        item["job_id"]: item for item in plan["full100_plan"]["jobs"]
    }
    scientific = jobs.get(CANARY_JOB_ID)
    if (
        scientific is None
        or scientific.get("source_role") != CANARY_SOURCE_ROLE
        or not isinstance(scientific.get("work_hand_indices"), list)
        or len(scientific["work_hand_indices"]) != 10
        or len(set(scientific["work_hand_indices"])) != 10
    ):
        raise ValueError("startup canary scientific shard changed")
    return row


def _validate_bootstrap(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    selected: Mapping[str, Any],
    job_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    # The production helper is single-job safe when supplied a view containing
    # exactly the selected canary.  Its digest and all full-run bindings remain
    # the original plan/ledger/resume values.
    single_resume = deepcopy(dict(resume))
    single_resume["selected_attempts"] = [deepcopy(dict(selected))]
    checked = result_receiver_v2._validate_bootstraps(
        plan=plan,
        ledger=ledger,
        resume=single_resume,
        job_bootstraps=[job_bootstrap],
    )
    return checked[CANARY_JOB_ID]


def _validate_cleanup_manifest_and_replay(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    cleanup_manifest: Mapping[str, Any],
    journal_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate cleanup and replay lifecycle without network access."""

    manifest = _strict_clone(
        cleanup_manifest, _CLEANUP_MANIFEST_FIELDS, "startup canary cleanup manifest"
    )
    digest = manifest.pop("manifest_sha256")
    if (
        not isinstance(digest, str)
        or digest != cleanup_v2.canonical_sha256(manifest)
    ):
        raise ValueError("startup canary cleanup manifest digest changed")
    manifest["manifest_sha256"] = digest
    namespace = production_receiver_v2.execution_namespace(ledger, resume)
    if (
        manifest["schema"] != cleanup_v2.CLEANUP_MANIFEST_SCHEMA
        or manifest["status"] != cleanup_v2.CLEANUP_MANIFEST_STATUS
        or manifest["execution_namespace"] != namespace
        or manifest["run_name"] != plan["run_name"]
        or manifest["execution_identity_sha256"]
        != plan["execution_identity_sha256"]
        or manifest["wave_index"] != resume["resume_wave_index"]
        or manifest["all_owned_instances_absent"] is not True
        or manifest["all_owned_boot_disks_absent"] is not True
        or manifest["worker_iam_bindings_absent"] is not True
        or manifest["shared_content_preserved_for_later_executions"] is not True
        or manifest["content_cleanup_event_sha256"] is not None
        or manifest["credentials_from_environment_only"] is not True
        or manifest["additional_create_authorized"] is not False
        or manifest["current_profile_changed"] is not False
        or result_receiver_v2._exposes_hidden(manifest)
    ):
        raise ValueError("startup canary cleanup safety binding changed")
    cleanup_v2._assert_no_credentials(
        manifest, "startup canary cleanup manifest"
    )

    request = production_receiver_v2.validate_production_receive_request(
        manifest["receiver_request"],
        expected_execution_namespace=namespace,
    )
    exact_journal = production_receiver_v2.resolve_production_receive_journal_dir(
        request=request,
        journal_dir=journal_dir,
        journal_dir_mode=production_receiver_v2.JOURNAL_DIR_MODE_EXACT,
        expected_execution_namespace=namespace,
    )
    lifecycle = _strict_clone(
        manifest["lifecycle_receipt"],
        _LIFECYCLE_RECEIPT_FIELDS,
        "startup canary lifecycle receipt",
    )
    sealed_lifecycle = deepcopy(lifecycle)
    lifecycle_sha = sealed_lifecycle.pop("receipt_sha256", None)
    if (
        lifecycle_sha != controller_v2.canonical_sha256(sealed_lifecycle)
        or lifecycle.get("schema") != controller_v2.LIFECYCLE_SCHEMA
        or lifecycle.get("status")
        != "exact_owned_gce_lifecycle_absence_attested"
        or lifecycle.get("run_name") != plan["run_name"]
        or lifecycle.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or lifecycle.get("wave_plan_sha256") != plan["schedule_sha256"]
        or lifecycle.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or lifecycle.get("resume_plan_sha256") != resume["resume_sha256"]
        or lifecycle.get("all_owned_instances_absent") is not True
        or lifecycle.get("all_owned_boot_disks_absent") is not True
        or lifecycle.get("worker_iam_bindings_absent") is not True
        or lifecycle.get("additional_create_authorized") is not False
        or lifecycle.get("current_profile_changed") is not False
        or manifest["receiver_request_sha256"] != request["request_sha256"]
        or manifest["closeout_event_sha256"]
        != request["closeout_event_sha256"]
        or request["observed_at_utc"] != lifecycle.get("attested_at_utc")
        or manifest["launch_event_sha256"]
        != lifecycle.get("launch_event_sha256")
        or manifest["delete_event_sha256"]
        != lifecycle.get("delete_event_sha256")
        or manifest["absence_event_sha256"]
        != lifecycle.get("absence_event_sha256")
        or manifest["worker_iam_cleanup_event_sha256"]
        != lifecycle.get("worker_iam_cleanup_event_sha256")
        or manifest["gce_create_receipt_sha256"]
        != lifecycle.get("gce_create_receipt_sha256")
        or manifest["gce_delete_receipt_sha256"]
        != lifecycle.get("gce_delete_receipt_sha256")
        or manifest["gce_absence_receipt_sha256"]
        != lifecycle.get("gce_absence_receipt_sha256")
        or manifest["worker_iam_cleanup_receipt_sha256"]
        != lifecycle.get("worker_iam_cleanup_receipt_sha256")
        or manifest["gce_delete_receipt_sha256"]
        != request["gce_delete_receipt"].get("receipt_sha256")
        or manifest["worker_iam_cleanup_receipt_sha256"]
        != request["worker_iam_cleanup_receipt"].get("receipt_sha256")
    ):
        raise ValueError("startup canary lifecycle cleanup binding changed")

    payload, launch_bundle, startup, validate_bundle = (
        production_receiver_v2._validated_material(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            request=request,
        )
    )
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=exact_journal,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        create_journal=False,
    )
    proof_box: dict[str, dict[str, Any]] = {}
    adapter = production_receiver_v2._lifecycle_adapter(
        controller=controller,
        request=payload,
        launch_bundle=launch_bundle,
        startup=startup,
        validate_bundle=validate_bundle,
        proof_box=proof_box,
    )
    raw_proof = adapter.validate(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        receiver_observed_at_utc=payload["observed_at_utc"],
    )
    proof = _strict_clone(
        raw_proof,
        _LIFECYCLE_PROOF_FIELDS,
        "startup canary replayed lifecycle proof",
    )
    sealed_proof = deepcopy(proof)
    proof_sha = sealed_proof.pop("proof_sha256")
    context = {
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
    }
    nested_receipts = (
        ("gce_create_receipt", "gce_create_receipt_sha256"),
        ("gce_delete_receipt", "gce_delete_receipt_sha256"),
        ("gce_absence_receipt", "gce_absence_receipt_sha256"),
        ("worker_iam_cleanup_receipt", "worker_iam_cleanup_receipt_sha256"),
    )
    if (
        proof_sha != controller_v2.canonical_sha256(sealed_proof)
        or proof.get("schema") != controller_v2.LIFECYCLE_PROOF_SCHEMA
        or proof.get("status")
        != "controller_journal_and_all_producer_receipts_revalidated"
        or proof.get("controller_context_sha256")
        != controller_v2.canonical_sha256(context)
        or any(proof.get(key) != value for key, value in context.items())
        or proof.get("lifecycle_event_sha256")
        != manifest["closeout_event_sha256"]
        or proof.get("lifecycle_receipt_sha256") != lifecycle_sha
        or proof.get("lifecycle_attested_at_utc") != request["observed_at_utc"]
        or proof.get("launch_event_sha256") != manifest["launch_event_sha256"]
        or proof.get("delete_event_sha256") != manifest["delete_event_sha256"]
        or proof.get("absence_event_sha256") != manifest["absence_event_sha256"]
        or proof.get("worker_iam_cleanup_event_sha256")
        != manifest["worker_iam_cleanup_event_sha256"]
        or any(
            not isinstance(proof.get(field), Mapping)
            or proof[field].get("receipt_sha256") != manifest[manifest_field]
            for field, manifest_field in nested_receipts
        )
        or proof.get("selected_instance_count") != 1
        or proof.get("exact_created_instance_count") != 1
        or proof.get("exact_uncreated_instance_count") != 0
        or proof.get("create_classification") != "all_selected_created"
        or proof.get("journal_hash_chain_valid") is not True
        or proof.get("all_producer_receipts_valid") is not True
        or proof.get("all_owned_instances_absent") is not True
        or proof.get("all_owned_boot_disks_absent") is not True
        or proof.get("worker_iam_bindings_absent") is not True
        or proof.get("additional_create_authorized") is not False
        or proof.get("current_profile_changed") is not False
    ):
        raise ValueError("startup canary replayed cleanup is incomplete")
    mappings = proof.get("selected_instance_mapping")
    matches = [
        row
        for row in mappings
        if isinstance(row, Mapping)
        and row.get("job_id") == CANARY_JOB_ID
        and row.get("attempt_id") == CANARY_ATTEMPT_ID
    ] if isinstance(mappings, list) else []
    if (
        not isinstance(mappings, list)
        or len(mappings) != 1
        or len(matches) != 1
        or matches[0].get("source_role") != CANARY_SOURCE_ROLE
        or matches[0].get("exact_instance_created") is not True
        or matches[0].get("final_instance_absent") is not True
        or matches[0].get("final_boot_disk_absent") is not True
    ):
        raise ValueError("startup canary lifecycle mapping changed")
    return manifest, proof


def receive_startup_canary(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    job_bootstrap: Mapping[str, Any],
    cleanup_manifest: Mapping[str, Any],
    journal_dir: str | Path,
    profile_path: str | Path,
    store: result_receiver_v2.ResultStore,
) -> dict[str, Any]:
    """Validate one diagnostic canary after exact lifecycle cleanup."""

    profile_before = _sha256_file(profile_path, "current profile registry")
    if profile_before != CURRENT_PROFILE_SHA256:
        raise ValueError("current profile registry changed before canary receive")
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if resume["all_jobs_complete"] is True:
        raise ValueError("startup canary requires an incomplete selected wave")
    selected = _canary_selected(plan, resume)
    bootstrap = _validate_bootstrap(
        plan=plan,
        ledger=ledger,
        resume=resume,
        selected=selected,
        job_bootstrap=job_bootstrap,
    )

    # Prove the VM, disk, and exact worker-IAM bindings absent before taking a
    # result snapshot.  The replay adapter has no GCE requester capable of I/O.
    cleanup, proof = _validate_cleanup_manifest_and_replay(
        plan=plan,
        ledger=ledger,
        resume=resume,
        cleanup_manifest=cleanup_manifest,
        journal_dir=journal_dir,
    )

    prefix = selected["artifact_prefix"]
    job_prefix = prefix.split("/attempts/", 1)[0] + "/"
    listed = store.list_prefix(prefix=job_prefix)
    records: dict[str, dict[str, Any]] = {}
    for raw_record in listed:
        record = result_receiver_v2._record(raw_record)
        if record["path"] in records:
            raise ValueError("startup canary provider snapshot contains duplicates")
        records[record["path"]] = record

    work = list(
        next(
            row["work_hand_indices"]
            for row in plan["full100_plan"]["jobs"]
            if row["job_id"] == CANARY_JOB_ID
        )
    )
    relative, expected_data_paths = result_receiver_v2._expected_attempt_paths(
        prefix=prefix,
        role=CANARY_SOURCE_ROLE,
        work=work,
    )
    del relative
    done_path = f"{prefix}/DONE.json"
    expected_paths = {*expected_data_paths, done_path}
    acceptance_path = plan["artifact_contract"][
        "job_acceptance_path_template"
    ].format(job_id=CANARY_JOB_ID)
    if acceptance_path in records:
        raise ValueError("startup canary must not observe ACCEPTED.json")
    if set(records) != expected_paths or len(expected_data_paths) != EXPECTED_DATA_OBJECT_COUNT:
        raise ValueError(
            "startup canary requires exact DONE plus twenty data objects"
        )

    data_raw = {
        path: result_receiver_v2._read_bound(store, records[path])
        for path in expected_data_paths
    }
    done_raw = result_receiver_v2._read_bound(store, records[done_path])
    done, _ = result_receiver_v2._validate_done(
        raw=done_raw,
        done_record=records[done_path],
        data_records={path: records[path] for path in expected_data_paths},
        data_raw=data_raw,
        plan=plan,
        ledger=ledger,
        resume=resume,
        selected=selected,
        bootstrap=bootstrap,
        work=work,
    )

    if store.read_current(path=acceptance_path, allow_missing=True) is not None:
        raise ValueError("startup canary must not observe ACCEPTED.json")

    profile_after = _sha256_file(profile_path, "current profile registry")
    if profile_after != profile_before:
        raise ValueError("current profile registry changed during canary receive")
    data_records = [records[path] for path in expected_data_paths]
    core = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "execution_scope": wave_v2.STARTUP_CANARY_SCOPE,
        "diagnostic_only": True,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "job_id": CANARY_JOB_ID,
        "source_role": CANARY_SOURCE_ROLE,
        "attempt_id": CANARY_ATTEMPT_ID,
        "instance_id": selected["instance_id"],
        "attempt_prefix": prefix,
        "done_record": records[done_path],
        "done_identity_sha256": done["done_identity_sha256"],
        "root_digest": done["root_digest"],
        "data_object_count": len(data_records),
        "data_records": data_records,
        "data_records_sha256": wave_v2.canonical_sha256(data_records),
        "cleanup_manifest_sha256": cleanup["manifest_sha256"],
        "lifecycle_proof_sha256": proof["proof_sha256"],
        "cleanup_complete": True,
        "all_owned_instances_absent": True,
        "all_owned_boot_disks_absent": True,
        "worker_iam_bindings_absent": True,
        "hidden_truth_exposed": False,
        "opponent_private_discards_used": False,
        "profile_sha256_before": profile_before,
        "profile_sha256_after": profile_after,
        "acceptance_object_observed": False,
        "acceptance_create_authorized": False,
        "acceptance_create_performed": False,
        "attempt_ledger_transition_authorized": False,
        "attempt_ledger_transition_performed": False,
        "merge_authorized": False,
        "merge_evidence": False,
        "scientific_merge_eligible": False,
        "training_eligible": False,
        "performance_evidence": False,
        "performance_evaluation_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "provider_mutation_performed": False,
        "current_profile_changed": False,
    }
    return validate_receipt({**core, "receipt_sha256": wave_v2.canonical_sha256(core)})


def validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _strict_clone(value, _RECEIPT_FIELDS, "startup canary receipt")
    digest = receipt.pop("receipt_sha256")
    if digest != wave_v2.canonical_sha256(receipt):
        raise ValueError("startup canary receipt digest changed")
    receipt["receipt_sha256"] = digest
    forbidden_true = (
        "hidden_truth_exposed",
        "opponent_private_discards_used",
        "acceptance_object_observed",
        "acceptance_create_authorized",
        "acceptance_create_performed",
        "attempt_ledger_transition_authorized",
        "attempt_ledger_transition_performed",
        "merge_authorized",
        "merge_evidence",
        "scientific_merge_eligible",
        "training_eligible",
        "performance_evidence",
        "performance_evaluation_eligible",
        "quality_evidence",
        "promotion_evidence",
        "provider_mutation_performed",
        "current_profile_changed",
    )
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["status"] != RECEIPT_STATUS
        or receipt["execution_scope"] != wave_v2.STARTUP_CANARY_SCOPE
        or receipt["diagnostic_only"] is not True
        or receipt["job_id"] != CANARY_JOB_ID
        or receipt["source_role"] != CANARY_SOURCE_ROLE
        or receipt["attempt_id"] != CANARY_ATTEMPT_ID
        or receipt["data_object_count"] != EXPECTED_DATA_OBJECT_COUNT
        or not isinstance(receipt["data_records"], list)
        or len(receipt["data_records"]) != EXPECTED_DATA_OBJECT_COUNT
        or receipt["data_records_sha256"]
        != wave_v2.canonical_sha256(receipt["data_records"])
        or receipt["all_owned_instances_absent"] is not True
        or receipt["all_owned_boot_disks_absent"] is not True
        or receipt["worker_iam_bindings_absent"] is not True
        or receipt["cleanup_complete"] is not True
        or receipt["profile_sha256_before"] != CURRENT_PROFILE_SHA256
        or receipt["profile_sha256_after"] != CURRENT_PROFILE_SHA256
        or any(receipt[field] is not False for field in forbidden_true)
    ):
        raise ValueError("startup canary receipt safety contract changed")
    return receipt


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != raw:
            raise FileExistsError(
                "startup canary receipt already exists with different bytes"
            ) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-plan", required=True)
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--resume-plan", required=True)
    parser.add_argument("--job-bootstrap", required=True)
    parser.add_argument("--cleanup-manifest", required=True)
    parser.add_argument("--journal-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-cloud-read", action="store_true")
    parser.add_argument("--confirm-run-name", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    requester: Callable[..., Any] = _stdlib_http_request,
) -> int:
    args = _parser().parse_args(argv)
    plan = wave_v2.validate_wave_plan(_read_json(args.wave_plan, "wave plan"))
    ledger = wave_v2.validate_attempt_ledger(
        plan, _read_json(args.attempt_ledger, "attempt ledger")
    )
    resume = wave_v2.validate_resume_plan(
        plan, ledger, _read_json(args.resume_plan, "resume plan")
    )
    if args.allow_cloud_read is not True:
        raise PermissionError("startup canary receive requires explicit cloud read")
    if args.confirm_run_name != plan["run_name"]:
        raise PermissionError("startup canary run confirmation changed")
    store = gcs_v2.GcsResultStoreV2(
        mode="read",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        requester=requester,
    )
    receipt = receive_startup_canary(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        job_bootstrap=_read_json(args.job_bootstrap, "job bootstrap"),
        cleanup_manifest=_read_json(args.cleanup_manifest, "cleanup manifest"),
        journal_dir=args.journal_dir,
        profile_path=args.profile_path,
        store=store,
    )
    _write_once(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "CANARY_ATTEMPT_ID",
    "CANARY_JOB_ID",
    "CANARY_SOURCE_ROLE",
    "CURRENT_PROFILE_SHA256",
    "EXPECTED_DATA_OBJECT_COUNT",
    "RECEIPT_SCHEMA",
    "RECEIPT_STATUS",
    "main",
    "receive_startup_canary",
    "validate_receipt",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
