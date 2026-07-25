from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as controller
from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_scientific_bridge_v2 as subject,
)
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


RUN_NAME = "regular-hu-m31-c02-f100wv2-bridge-001"
CONTENT_SHA = "4" * 64
OUTER_SHA = "5" * 64


def _sha(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _worker_principal(slot_index: int) -> str:
    return f"wave-worker-{slot_index:02d}@ofcsolver.iam.gserviceaccount.com"


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _local(root: Path, object_path: str) -> Path:
    return root.joinpath(*PurePosixPath(object_path).parts)


def _job_metadata(plan: dict) -> dict[str, dict[str, Any]]:
    science = {row["job_id"]: row for row in plan["full100_plan"]["jobs"]}
    result: dict[str, dict[str, Any]] = {}
    for wave_row in plan["waves"]:
        for pair in wave_row["candidate_reference_pairs"]:
            for role in wave.SOURCE_ROLES:
                job_id = pair[f"{role}_job_id"]
                result[job_id] = {
                    **science[job_id],
                    "wave_index": wave_row["wave_index"],
                    "instance_ids": pair[f"{role}_attempt_instance_ids"],
                }
    return result


def _transport_done(
    *,
    plan: dict,
    meta: dict,
    root: Path,
    attempt_id: str,
    launch_attempt_ledger_sha256: str,
    launch_resume_sha256: str,
    launch_observed_transition_digest: str,
    worker_principal: str,
) -> dict:
    role = meta["source_role"]
    work = meta["work_hand_indices"]
    prefix = plan["artifact_contract"]["attempt_path_template"].format(
        job_id=meta["job_id"], attempt_id=attempt_id
    )
    roots: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    runner_source = root / "runner-source" / meta["job_id"]
    contract = plan["full100_plan"]["run_contract"]
    manifest = runner.build_shard_manifest(
        run_contract=contract,
        source_role=role,
        work_hand_indices=work,
    )
    _write(runner_source / "run_contract.json", runner.canonical_bytes(contract))
    _write(runner_source / "shard_manifest.json", runner.canonical_bytes(manifest))
    for hand in work:
        root_relative = f"roots/hand_{hand:03d}.json"
        hand_relative = f"hands/{role}/hand_{hand:03d}.json"
        root_path = _local(root, f"{prefix}/{root_relative}")
        hand_path = _local(root, f"{prefix}/{hand_relative}")
        root_raw = runner.canonical_bytes(
            {"hand_index": hand, "root": "fixed"}
        )
        hand_raw = runner.canonical_bytes(
            {"hand_index": hand, "role": role, "unit": "source"}
        )
        _write(root_path, root_raw)
        _write(hand_path, hand_raw)
        _write(runner_source / root_relative, root_raw)
        _write(runner_source / hand_relative, hand_raw)
        roots.append({"hand_index": hand, "sha256": _sha(root_raw)})
        artifacts.extend(
            (
                {"path": root_relative, "sha256": _sha(root_raw), "bytes": len(root_raw)},
                {"path": hand_relative, "sha256": _sha(hand_raw), "bytes": len(hand_raw)},
            )
        )
    root_digest = wave.canonical_sha256(roots)
    runner_done = runner._build_done(
        output_dir=runner_source, shard_manifest=manifest
    )
    return {
        "schema": "hu_m31_t3_step6d_full100_wave_attempt_done_v2",
        "status": "complete_validated_single_job_attempt",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": launch_attempt_ledger_sha256,
        "resume_sha256": launch_resume_sha256,
        "observed_transition_digest": launch_observed_transition_digest,
        "wave_index": meta["wave_index"],
        "job_id": meta["job_id"],
        "source_role": role,
        "attempt_id": attempt_id,
        "package_sha256": plan["runtime_binding"]["package_sha256"],
        "image_digest": plan["runtime_binding"]["image_digest"],
        "binary_sha256": plan["runtime_binding"]["binary_sha256_by_role"][role],
        "allocation_digest": plan["runtime_binding"]["allocation_digest"],
        "run_contract_digest": plan["run_contract_digest"],
        "root_digest": root_digest,
        "done_identity_sha256": wave.expected_done_identity_sha256(
            plan,
            job_id=meta["job_id"],
            attempt_id=attempt_id,
            root_digest=root_digest,
        ),
        "content_payload_sha256": CONTENT_SHA,
        "outer_manifest_sha256": OUTER_SHA,
        "prelaunch_authorization_sha256": _sha(
            f"prelaunch-{meta['wave_index']}"
        ),
        "worker_principal": worker_principal,
        "work_hand_indices": list(work),
        "artifact_count": 20,
        "artifacts": artifacts,
        "runner_done_sha256": runner.canonical_sha256(runner_done),
        "metadata_hidden_truth_exposed": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


def _transition(
    *,
    plan: dict,
    previous: dict,
    wave_index: int,
    done_by_job: dict[str, dict],
    acceptance_by_job: dict[str, dict],
    attempt_by_job: dict[str, str],
    observed_at_utc: str,
    retry_only: bool = False,
) -> dict:
    histories = {
        row["job_id"]: copy.deepcopy(row) for row in previous["attempt_history"]
    }
    done = copy.deepcopy(previous["done_objects"])
    acceptances = copy.deepcopy(previous["acceptance_records"])
    meta = _job_metadata(plan)
    wave_jobs = plan["waves"][wave_index]["job_ids"]
    selected_jobs = (
        [job_id for job_id in wave_jobs if attempt_by_job[job_id] == "a01"]
        if retry_only
        else wave_jobs
    )
    for job_id in selected_jobs:
        attempt_id = attempt_by_job[job_id]
        if attempt_id == "a01" and not retry_only:
            histories[job_id]["attempts"].append(
                {
                    "attempt_id": "a00",
                    "instance_id": meta[job_id]["instance_ids"]["a00"],
                    "launch_receipt_sha256": _sha(f"launch-{job_id}-a00"),
                    "terminal_status": "failed",
                }
            )
            continue
        histories[job_id]["attempts"].append(
            {
                "attempt_id": attempt_id,
                "instance_id": meta[job_id]["instance_ids"][attempt_id],
                "launch_receipt_sha256": _sha(f"launch-{job_id}-{attempt_id}"),
                "terminal_status": "accepted",
            }
        )
        done.append(done_by_job[job_id])
        acceptances.append(acceptance_by_job[job_id])
    return wave.build_observed_transition(
        plan,
        project_id="ofc-solver-485418",
        zone="asia-northeast1-b",
        observed_at_utc=observed_at_utc,
        previous_transition_digest=previous["transition_digest"],
        attempt_history=[
            histories[job_id] for job_id in plan["coverage"]["job_ids"]
        ],
        done_objects=done,
        acceptance_records=acceptances,
    )


def _validated_lifecycle_chain(plan: dict, ledger: dict) -> dict[str, Any]:
    proofs, receipts = _controller_receiver_sources(plan, ledger)
    return subject.normalize_controller_receiver_lifecycle_chain(
        wave_plan=plan,
        attempt_ledger=ledger,
        controller_proofs=proofs,
        receiver_receipts=receipts,
    )


def _controller_receiver_sources(
    plan: dict,
    final_ledger: dict,
    *,
    created_jobs_by_execution: dict[int, set[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transitions = final_ledger["transitions"]
    proofs: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for transition_index in range(1, len(transitions)):
        execution_index = transition_index - 1
        pre_transitions = transitions[:transition_index]
        pre_ledger = wave.build_attempt_ledger(
            plan,
            transitions=pre_transitions,
            consumed_transition_digests=[
                row["transition_digest"] for row in pre_transitions[:-1]
            ],
        )
        resume = wave.build_resume_plan(plan, attempt_ledger=pre_ledger)
        wave_index = resume["resume_wave_index"]
        post_transitions = transitions[: transition_index + 1]
        post_ledger = wave.build_attempt_ledger(
            plan,
            transitions=post_transitions,
            consumed_transition_digests=[
                row["transition_digest"] for row in post_transitions[:-1]
            ],
        )
        selected_mapping: list[dict[str, Any]] = []
        create_rows: list[dict[str, Any]] = []
        actual_rows: list[dict[str, Any]] = []
        selected_jobs = {
            row["job_id"] for row in resume["selected_attempts"]
        }
        created_jobs = (
            selected_jobs
            if created_jobs_by_execution is None
            else created_jobs_by_execution.get(execution_index, selected_jobs)
        )
        if not created_jobs.issubset(selected_jobs):
            raise ValueError("created fixture jobs escaped selected execution")
        terminal_by_job = {
            row["job_id"]: row["attempts"][-1]
            for row in post_transitions[-1]["attempt_history"]
            if row["job_id"] in selected_jobs
        }
        for slot_index, selected in enumerate(resume["selected_attempts"]):
            job_id = selected["job_id"]
            terminal = terminal_by_job[job_id]
            attempt_id = selected["attempt_id"]
            exact_created = job_id in created_jobs
            actual_launch_present = len(created_jobs) == len(selected_jobs)
            selected_mapping.append(
                {
                    **selected,
                    "launch_receipt_sha256": terminal[
                        "launch_receipt_sha256"
                    ],
                    "exact_instance_created": exact_created,
                    "provider_instance_id": (
                        f"provider-instance-{job_id}-{attempt_id}"
                        if exact_created
                        else None
                    ),
                    "provider_boot_disk_id": (
                        f"provider-disk-{job_id}-{attempt_id}"
                        if exact_created
                        else None
                    ),
                    "gce_spec_sha256": (
                        _sha(f"gce-spec-{job_id}-{attempt_id}")
                        if exact_created
                        else None
                    ),
                    "gce_operation_id": (
                        f"create-operation-{job_id}-{attempt_id}"
                        if exact_created
                        else None
                    ),
                    "actual_launch_operation_id": (
                        f"launch-operation-{job_id}-{attempt_id}"
                        if actual_launch_present
                        else None
                    ),
                    "actual_launch_instance_status": (
                        "RUNNING" if actual_launch_present else None
                    ),
                    "ownership_label": f"owned-{job_id}-{attempt_id}",
                    "final_instance_absent": True,
                    "final_boot_disk_absent": True,
                }
            )
            if exact_created:
                create_rows.append(
                    {
                        "job_id": job_id,
                        "service_account": _worker_principal(slot_index),
                    }
                )
            if actual_launch_present:
                actual_rows.append({"job_id": job_id})
        actual_receipt = (
            {
                "receipt_sha256": _sha(f"actual-launch-{execution_index}"),
                "planned_mapping_receipt_sha256": _sha(
                    f"mapping-{execution_index}"
                ),
                "prelaunch_authorization_sha256": _sha(
                    f"prelaunch-{execution_index}"
                ),
                "rows": actual_rows,
            }
            if len(created_jobs) == len(selected_jobs)
            else None
        )
        create_receipt = {
            "receipt_sha256": _sha(f"gce-create-{execution_index}"),
            "rows": create_rows,
        }
        delete_receipt = {
            "receipt_sha256": _sha(f"gce-delete-{execution_index}")
        }
        absence_receipt = {
            "receipt_sha256": _sha(f"gce-absence-{execution_index}")
        }
        iam_receipt = {
            "receipt_sha256": _sha(f"iam-cleanup-{execution_index}")
        }
        context = {
            "run_name": plan["run_name"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "wave_plan_sha256": plan["schedule_sha256"],
            "attempt_ledger_sha256": pre_ledger["ledger_sha256"],
            "resume_plan_sha256": resume["resume_sha256"],
            "wave_index": wave_index,
        }
        proof_body = {
            "schema": subject._CONTROLLER_LIFECYCLE_PROOF_SCHEMA,
            "status": "controller_journal_and_all_producer_receipts_revalidated",
            "controller_context_sha256": controller.canonical_sha256(context),
            **context,
            "lifecycle_event_sha256": _sha(
                f"lifecycle-event-{execution_index}"
            ),
            "lifecycle_receipt_sha256": _sha(
                f"lifecycle-receipt-{execution_index}"
            ),
            "launch_event_sha256": _sha(f"launch-event-{execution_index}"),
            "delete_event_sha256": _sha(f"delete-event-{execution_index}"),
            "absence_event_sha256": _sha(f"absence-event-{execution_index}"),
            "worker_iam_cleanup_event_sha256": _sha(
                f"iam-event-{execution_index}"
            ),
            "launch_bundle_sha256": _sha(
                f"launch-bundle-{execution_index}"
            ),
            "gce_create_receipt": create_receipt,
            "actual_launch_receipt": actual_receipt,
            "gce_delete_receipt": delete_receipt,
            "gce_absence_receipt": absence_receipt,
            "worker_iam_cleanup_receipt": iam_receipt,
            "selected_instance_mapping": selected_mapping,
            "gce_create_rows": create_rows,
            "actual_launch_rows": actual_rows,
            "selected_instance_count": len(selected_mapping),
            "exact_created_instance_count": len(created_jobs),
            "exact_uncreated_instance_count": (
                len(selected_mapping) - len(created_jobs)
            ),
            "create_classification": (
                "all_selected_created"
                if len(created_jobs) == len(selected_mapping)
                else "no_selected_created"
                if not created_jobs
                else "partial_selected_created"
            ),
            "actual_launch_receipt_present": actual_receipt is not None,
            "lifecycle_attested_at_utc": (
                f"2026-07-22T00:00:{execution_index + 1:02d}Z"
            ),
            "journal_event_count": 10 + execution_index,
            "journal_hash_chain_valid": True,
            "all_producer_receipts_valid": True,
            "all_owned_instances_absent": True,
            "all_owned_boot_disks_absent": True,
            "worker_iam_bindings_absent": True,
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
        proofs.append(
            {
                **proof_body,
                "proof_sha256": controller.canonical_sha256(proof_body),
            }
        )
        absence_sha = absence_receipt["receipt_sha256"]
        attempt_results: list[dict[str, Any]] = []
        for row in selected_mapping:
            shard = int(row["job_id"].rsplit("-", 1)[1])
            role = row["source_role"]
            peer = (
                f"reference-shard-{shard:02d}"
                if role == "candidate"
                else f"candidate-shard-{shard:02d}"
            )
            terminal_status = terminal_by_job[row["job_id"]]["terminal_status"]
            attempt_results.append(
                {
                    "job_id": row["job_id"],
                    "source_role": role,
                    "attempt_id": row["attempt_id"],
                    "instance_id": row["instance_id"],
                    "launch_receipt_sha256": row["launch_receipt_sha256"],
                    "lifecycle_proof_sha256": proofs[-1]["proof_sha256"],
                    "gce_absence_receipt_sha256": absence_receipt[
                        "receipt_sha256"
                    ],
                    "terminal_status": terminal_status,
                    "pair_id": f"paired-shard-{shard:02d}",
                    "peer_job_id": peer,
                    "pair_atomic_outcome": (
                        "accepted_both_exact_created_valid_done"
                        if terminal_status == "accepted"
                        else "failed_pair_retry_required"
                    ),
                    "exact_instance_created": row["exact_instance_created"],
                    "valid_done_observed": (
                        terminal_status == "accepted"
                        and row["exact_instance_created"]
                    ),
                }
            )
        lifecycle_body = {
            "schema": "hu_m31_t3_step6d_full100_wave_transition_lifecycle_binding_v2",
            "observed_transition_digest": post_ledger[
                "latest_transition_digest"
            ],
            "wave_index": wave_index,
            "wave_launch_receipt_sha256": (
                create_receipt["receipt_sha256"]
                if actual_receipt is None
                else actual_receipt["receipt_sha256"]
            ),
            "launch_mapping_sha256": controller.canonical_sha256(
                selected_mapping
            ),
            "accepted_launch_receipt_sha256s": [
                row["launch_receipt_sha256"]
                for row in selected_mapping
                if terminal_by_job[row["job_id"]]["terminal_status"]
                == "accepted"
            ],
            "lifecycle_proof_sha256": proofs[-1]["proof_sha256"],
            "controller_lifecycle_receipt_sha256": proofs[-1][
                "lifecycle_receipt_sha256"
            ],
            "gce_absence_receipt_sha256": absence_sha,
            "worker_iam_cleanup_receipt_sha256": iam_receipt["receipt_sha256"],
            "all_selected_instances_absent": True,
            "all_selected_boot_disks_absent": True,
            "worker_iam_bindings_absent": True,
        }
        receipts.append(
            {
                "run_name": plan["run_name"],
                "execution_identity_sha256": plan["execution_identity_sha256"],
                "wave_plan_sha256": plan["schedule_sha256"],
                "previous_attempt_ledger_sha256": pre_ledger["ledger_sha256"],
                "input_resume_plan_sha256": resume["resume_sha256"],
                "wave_index": wave_index,
                "attempt_results": attempt_results,
                "accepted_job_ids": [
                    row["job_id"]
                    for row in selected_mapping
                    if terminal_by_job[row["job_id"]]["terminal_status"]
                    == "accepted"
                ],
                "failed_job_ids": [
                    row["job_id"]
                    for row in selected_mapping
                    if terminal_by_job[row["job_id"]]["terminal_status"]
                    == "failed"
                ],
                "attempt_ledger": post_ledger,
                "pair_atomicity_enforced": True,
                "validated_lifecycle_proof": proofs[-1],
                "lifecycle_proof_sha256": proofs[-1]["proof_sha256"],
                "gce_absence_receipt_sha256": absence_sha,
                "worker_iam_cleanup_receipt_sha256": iam_receipt[
                    "receipt_sha256"
                ],
                "worker_iam_bindings_absent": True,
                "transition_lifecycle_binding": {
                    **lifecycle_body,
                    "binding_sha256": subject.canonical_sha256(lifecycle_body),
                },
            }
        )
    return proofs, receipts


def _fixture(
    tmp_path: Path,
    *,
    cross_lane_reference_job: str | None = None,
    retry_pair_shard: int | None = None,
    retry_all_wave0: bool = False,
) -> dict[str, Any]:
    if cross_lane_reference_job is not None and (
        retry_pair_shard is not None or retry_all_wave0
    ):
        raise ValueError("fixture retry modes are mutually exclusive")
    if retry_pair_shard is not None and retry_all_wave0:
        raise ValueError("fixture retry modes are mutually exclusive")
    plan = wave.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    accepted_root = (tmp_path / "accepted").resolve()
    accepted_root.mkdir()
    meta = _job_metadata(plan)
    transport: dict[str, dict] = {}
    done_by_job: dict[str, dict] = {}
    acceptance_by_job: dict[str, dict] = {}
    retry_jobs = (
        set(plan["waves"][0]["job_ids"])
        if retry_all_wave0
        else set()
        if retry_pair_shard is None
        else {
            f"candidate-shard-{retry_pair_shard:02d}",
            f"reference-shard-{retry_pair_shard:02d}",
        }
    )
    attempt_by_job = {
        job_id: (
            "a01"
            if job_id == cross_lane_reference_job or job_id in retry_jobs
            else "a00"
        )
        for job_id in plan["coverage"]["job_ids"]
    }
    baseline = wave.build_observed_transition(
        plan,
        project_id="ofc-solver-485418",
        zone="asia-northeast1-b",
        observed_at_utc="2027-01-15T08:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave.empty_attempt_history(plan),
    )
    transitions = [baseline]
    consumed: list[str] = []
    transition_ordinal = 1
    for wave_index in range(3):
        pre_ledger = wave.build_attempt_ledger(
            plan,
            transitions=transitions,
            consumed_transition_digests=consumed,
        )
        resume = wave.build_resume_plan(plan, attempt_ledger=pre_ledger)
        worker_by_job = {
            row["job_id"]: _worker_principal(index)
            for index, row in enumerate(resume["selected_attempts"])
        }
        for job_id in plan["waves"][wave_index]["job_ids"]:
            ordinal = plan["coverage"]["job_ids"].index(job_id)
            job = meta[job_id]
            attempt_id = attempt_by_job[job_id]
            done = _transport_done(
                plan=plan,
                meta=job,
                root=accepted_root,
                attempt_id=attempt_id,
                launch_attempt_ledger_sha256=pre_ledger["ledger_sha256"],
                launch_resume_sha256=resume["resume_sha256"],
                launch_observed_transition_digest=pre_ledger[
                    "latest_transition_digest"
                ],
                worker_principal=worker_by_job[job_id],
            )
            transport[job_id] = done
            done_path = plan["artifact_contract"]["attempt_path_template"].format(
                job_id=job_id, attempt_id=attempt_id
            ) + "/DONE.json"
            done_raw = controller.canonical_bytes(done)
            _write(_local(accepted_root, done_path), done_raw)
            acceptance_path = plan["artifact_contract"][
                "job_acceptance_path_template"
            ].format(job_id=job_id)
            acceptance_raw = runner.canonical_bytes(
                {"job_id": job_id, "attempt_id": attempt_id, "accepted": True}
            )
            _write(_local(accepted_root, acceptance_path), acceptance_raw)
            done_generation = 10_000 + ordinal * 10
            done_by_job[job_id] = {
                "job_id": job_id,
                "source_role": job["source_role"],
                "attempt_id": attempt_id,
                "path": done_path,
                "generation": done_generation,
                "bytes": len(done_raw),
                "sha256": _sha(done_raw),
                "done_identity_sha256": done["done_identity_sha256"],
                "package_sha256": done["package_sha256"],
                "image_digest": done["image_digest"],
                "binary_sha256": done["binary_sha256"],
                "allocation_digest": done["allocation_digest"],
                "root_digest": done["root_digest"],
            }
            acceptance_by_job[job_id] = {
                "job_id": job_id,
                "source_role": job["source_role"],
                "attempt_id": attempt_id,
                "path": acceptance_path,
                "generation": done_generation + 1,
                "bytes": len(acceptance_raw),
                "sha256": _sha(acceptance_raw),
                "done_generation": done_generation,
                "done_sha256": _sha(done_raw),
                "create_only": True,
            }
        consumed.append(transitions[-1]["transition_digest"])
        transitions.append(
            _transition(
                plan=plan,
                previous=transitions[-1],
                wave_index=wave_index,
                done_by_job=done_by_job,
                acceptance_by_job=acceptance_by_job,
                attempt_by_job=attempt_by_job,
                observed_at_utc=(
                    f"2027-01-15T08:00:{transition_ordinal:02d}Z"
                ),
            )
        )
        transition_ordinal += 1
        if any(
            attempt_by_job[job_id] == "a01"
            for job_id in plan["waves"][wave_index]["job_ids"]
        ):
            consumed.append(transitions[-1]["transition_digest"])
            transitions.append(
                _transition(
                    plan=plan,
                    previous=transitions[-1],
                    wave_index=wave_index,
                    done_by_job=done_by_job,
                    acceptance_by_job=acceptance_by_job,
                    attempt_by_job=attempt_by_job,
                    observed_at_utc=(
                        f"2027-01-15T08:00:{transition_ordinal:02d}Z"
                    ),
                    retry_only=True,
                )
            )
            transition_ordinal += 1
    ledger = wave.build_attempt_ledger(
        plan,
        transitions=transitions,
        consumed_transition_digests=consumed,
    )
    expected = wave.expected_artifact_inventory(plan, attempt_ledger=ledger)
    observed_objects: list[dict[str, Any]] = []
    generation = 20_000
    for record in expected["records"]:
        common = {
            "job_id": record["job_id"],
            "source_role": record["source_role"],
            "attempt_id": record["accepted_attempt_id"],
        }
        for kind, hand, object_path in [
            *(
                ("root", hand_index, object_path)
                for hand_index, object_path in zip(
                    record["work_hand_indices"], record["root_paths"], strict=True
                )
            ),
            *(
                ("source_hand", hand_index, object_path)
                for hand_index, object_path in zip(
                    record["work_hand_indices"],
                    record["source_hand_paths"],
                    strict=True,
                )
            ),
        ]:
            generation += 1
            path = _local(accepted_root, object_path)
            observed_objects.append(
                {
                    "path": object_path,
                    **common,
                    "object_kind": kind,
                    "hand_index": hand,
                    "generation": generation,
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path.read_bytes()),
                    "done_identity_sha256": None,
                    "package_sha256": None,
                    "image_digest": None,
                    "binary_sha256": None,
                    "allocation_digest": None,
                    "root_digest": None,
                }
            )
        for kind, object_path in (
            ("done", record["done_path"]),
            ("acceptance", record["acceptance_path"]),
        ):
            row = {
                "path": object_path,
                **common,
                "object_kind": kind,
                "hand_index": None,
                "generation": record[f"{kind}_generation"],
                "bytes": record[f"{kind}_bytes"],
                "sha256": record[f"{kind}_sha256"],
                "done_identity_sha256": None,
                "package_sha256": None,
                "image_digest": None,
                "binary_sha256": None,
                "allocation_digest": None,
                "root_digest": None,
            }
            if kind == "done":
                for key in (
                    "done_identity_sha256", "package_sha256", "image_digest",
                    "binary_sha256", "allocation_digest", "root_digest",
                ):
                    row[key] = record[key]
            observed_objects.append(row)
    observed = wave.build_observed_artifact_inventory(
        plan,
        attempt_ledger=ledger,
        observed_at_utc="2027-01-15T08:01:00Z",
        objects=observed_objects,
    )
    receiver_receipt = (tmp_path / "receiver_receipt.json").resolve()
    _write(receiver_receipt, runner.canonical_bytes({"receiver": "validated"}))
    subject._pair_atomic_acceptance(plan, ledger)
    lifecycle_chain = _validated_lifecycle_chain(plan, ledger)
    snapshot = subject.build_accepted_results_snapshot(
        wave_plan=plan,
        attempt_ledger=ledger,
        accepted_root=accepted_root,
        receiver_receipt_path=receiver_receipt,
        expected_inventory=expected,
        observed_inventory=observed,
        validated_lifecycle_chain=lifecycle_chain,
        expected_startup_sha256=subject.EXPECTED_STARTUP_SHA256,
        content_payload_sha256=CONTENT_SHA,
        outer_manifest_sha256=OUTER_SHA,
    )
    # runner-source is fixture scaffolding, not accepted receiver inventory.
    import shutil

    shutil.rmtree(accepted_root / "runner-source")
    return {
        "plan": plan,
        "ledger": ledger,
        "accepted_root": accepted_root,
        "receiver_receipt": receiver_receipt,
        "snapshot": snapshot,
        "transport": transport,
        "lifecycle_chain": lifecycle_chain,
    }


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _fixture(tmp_path_factory.mktemp("scientific-bridge"))


def _reseal_snapshot(value: dict[str, Any]) -> None:
    value["snapshot_sha256"] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != "snapshot_sha256"}
    )


def _reseal_lifecycle_chain(value: dict[str, Any], wave_index: int) -> None:
    proof = value["wave_proofs"][wave_index]
    proof["proof_sha256"] = subject.canonical_sha256(
        {key: item for key, item in proof.items() if key != "proof_sha256"}
    )
    value["chain_sha256"] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != "chain_sha256"}
    )


def _scientific_summary(
    plan: dict,
    candidate_done_paths: list[Path],
    reference_done_paths: list[Path],
    *,
    first_p95: float = 140.0,
    portable_count: int = 200,
) -> dict[str, Any]:
    all_gates = first_p95 <= 150.0 and portable_count == 200
    gates = {
        "exactly_100_paired_hands": True,
        "exactly_200_roots": True,
        "exactly_100_first_and_100_second": True,
        "exactly_20_hands_per_profile": True,
        "portable_semantic_parity_fraction_one": portable_count == 200,
        "missing_or_censored_roots_zero": True,
        "first_p95_within_150_seconds": first_p95 <= 150.0,
        "first_p99_within_240_seconds": True,
        "first_max_within_240_seconds": True,
        "second_p95_within_5_seconds": True,
        "peak_rss_within_858993459_bytes": True,
    }

    def inputs(paths: list[Path]) -> list[dict[str, str]]:
        return [{"path": str(path.resolve())} for path in paths]

    return {
        "schema": subject.scientific.MERGE_SCHEMA,
        "status": "pass" if all_gates else "no_go",
        "scope": subject.scientific.FULL_SCOPE,
        "candidate_variant": runner.CANDIDATE02_VARIANT,
        "full100_plan": copy.deepcopy(plan["full100_plan"]),
        "run_contract": copy.deepcopy(plan["full100_plan"]["run_contract"]),
        "run_contract_digest": plan["run_contract_digest"],
        "allocation": copy.deepcopy(runner.ALLOCATION),
        "candidate_library_sha256": plan["runtime_binding"][
            "binary_sha256_by_role"
        ]["candidate"],
        "reference_library_sha256": plan["runtime_binding"][
            "binary_sha256_by_role"
        ]["reference"],
        "paired_hand_count": 100,
        "root_count": 200,
        "source_done_inputs": {
            "candidate": inputs(candidate_done_paths),
            "reference": inputs(reference_done_paths),
        },
        "integrity": {"paired_root_parity_count": portable_count},
        "performance": {
            "candidate_by_seat": {
                "first": {
                    "p95_seconds": first_p95,
                    "p99_seconds": 200.0,
                    "max_seconds": 220.0,
                },
                "second": {"p95_seconds": 4.0},
            },
            "peak_source_process_rss_bytes": 800_000_000,
        },
        "gates": gates,
        "all_gates_passed": all_gates,
        "performance_candidate_frozen": all_gates,
        "performance_lock_authorized": all_gates,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def _install_scientific(
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, Any],
    *,
    first_p95: float = 140.0,
    portable_count: int = 200,
) -> None:
    def merge(**kwargs: Any) -> dict[str, Any]:
        return _scientific_summary(
            evidence["plan"],
            list(kwargs["candidate_done_paths"]),
            list(kwargs["reference_done_paths"]),
            first_p95=first_p95,
            portable_count=portable_count,
        )

    monkeypatch.setattr(subject.scientific, "merge_candidate02_full100", merge)


def _lifecycle_adapter(
    evidence: dict[str, Any], *, chain: dict[str, Any] | None = None
) -> subject.CallbackValidatedLifecycleChainAdapterV2:
    selected = copy.deepcopy(chain or evidence["lifecycle_chain"])
    return subject.CallbackValidatedLifecycleChainAdapterV2(
        lifecycle_sources=[{"validated_chain": selected}],
        validator=lambda _plan, _ledger, sources: sources[0]["validated_chain"],
    )


def test_bridge_builds_separate_merge_view_and_write_once_go_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, Any],
) -> None:
    _install_scientific(monkeypatch, evidence)
    accepted_done = next(evidence["accepted_root"].rglob("DONE.json"))
    accepted_before = _sha(accepted_done.read_bytes())
    view = (tmp_path / "merge-view").resolve()
    output = (tmp_path / "gate-receipt.json").resolve()
    receipt = subject.merge_and_write_scientific_gate_receipt(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        accepted_results_adapter=subject.StaticAcceptedResultsAdapterV2(
            evidence["snapshot"]
        ),
        lifecycle_chain_adapter=_lifecycle_adapter(evidence),
        merge_view_root=view,
        receipt_output_path=output,
    )
    assert receipt["status"] == "pass"
    assert receipt["performance_lock_authorized"] is True
    assert receipt["quality_pilot_authorized"] is False
    assert receipt["job_count"] == 20
    assert receipt["root_count"] == 200
    assert receipt["merge_view_manifest"]["candidate_job_count"] == 10
    assert receipt["merge_view_manifest"]["reference_job_count"] == 10
    assert receipt["merge_view_manifest"]["accepted_tree_modified"] is False
    assert len(list(view.glob("jobs/*/DONE.json"))) == 20
    assert len(list(view.glob("jobs/*/run_contract.json"))) == 20
    assert len(list(view.glob("jobs/*/shard_manifest.json"))) == 20
    assert _sha(accepted_done.read_bytes()) == accepted_before
    assert output.read_bytes() == subject.canonical_bytes(receipt)
    assert subject.validate_scientific_gate_receipt_value(receipt) == receipt
    assert subject.validate_scientific_gate_receipt(
        receipt_path=output
    ) == receipt
    with pytest.raises(FileExistsError, match="write-once"):
        subject.merge_and_write_scientific_gate_receipt(
            wave_plan=evidence["plan"],
            attempt_ledger=evidence["ledger"],
            accepted_results_adapter=subject.StaticAcceptedResultsAdapterV2(
                evidence["snapshot"]
            ),
            lifecycle_chain_adapter=_lifecycle_adapter(evidence),
            merge_view_root=tmp_path / "unused-view",
            receipt_output_path=output,
        )


@pytest.mark.parametrize(
    ("first_p95", "portable_count", "failed_gate"),
    [
        (150.0001, 200, "first_p95_within_150_seconds"),
        (140.0, 199, "portable_semantic_parity_fraction_one"),
    ],
)
def test_bridge_records_no_go_at_exact_performance_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, Any],
    first_p95: float,
    portable_count: int,
    failed_gate: str,
) -> None:
    _install_scientific(
        monkeypatch,
        evidence,
        first_p95=first_p95,
        portable_count=portable_count,
    )
    receipt = subject.merge_accepted_results_to_scientific_gate(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        accepted_results_adapter=subject.StaticAcceptedResultsAdapterV2(
            evidence["snapshot"]
        ),
        lifecycle_chain_adapter=_lifecycle_adapter(evidence),
        merge_view_root=(tmp_path / f"view-{failed_gate}").resolve(),
    )
    assert receipt["status"] == "no_go"
    assert receipt["performance_lock_authorized"] is False
    assert receipt["performance_gate"]["gates"][failed_gate] is False


def test_snapshot_revalidates_all_440_objects_and_exact_adapter_digest(
    evidence: dict[str, Any],
) -> None:
    validated = subject.validate_accepted_results_snapshot(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        validated_lifecycle_chain=evidence["lifecycle_chain"],
        value=evidence["snapshot"],
    )
    assert validated.snapshot["accepted_object_count"] == 440
    assert len(validated.observed["objects"]) == 440
    assert len(validated.jobs) == 20

    forged = copy.deepcopy(evidence["snapshot"])
    forged["accepted_job_count"] = 19
    _reseal_snapshot(forged)
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_accepted_results_snapshot(
            wave_plan=evidence["plan"],
            attempt_ledger=evidence["ledger"],
            validated_lifecycle_chain=evidence["lifecycle_chain"],
            value=forged,
        )


def test_candidate_a00_reference_a01_pair_is_rejected_before_merge(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="candidate/reference pair crossed accepted attempt lanes"
    ):
        _fixture(
            tmp_path,
            cross_lane_reference_job="reference-shard-00",
        )


def test_lifecycle_adapter_is_required_and_running_vm_is_rejected_before_merge(
    tmp_path: Path, evidence: dict[str, Any]
) -> None:
    with pytest.raises(TypeError, match="lifecycle-chain adapter"):
        subject.merge_accepted_results_to_scientific_gate(
            wave_plan=evidence["plan"],
            attempt_ledger=evidence["ledger"],
            accepted_results_adapter=subject.StaticAcceptedResultsAdapterV2(
                evidence["snapshot"]
            ),
            lifecycle_chain_adapter=object(),  # type: ignore[arg-type]
            merge_view_root=(tmp_path / "missing-adapter-view").resolve(),
        )

    running = copy.deepcopy(evidence["lifecycle_chain"])
    running["wave_proofs"][0]["all_selected_instances_absent"] = False
    running["wave_proofs"][0]["running_instance_count"] = 1
    _reseal_lifecycle_chain(running, 0)
    with pytest.raises(ValueError, match="not fully quiescent"):
        subject.merge_accepted_results_to_scientific_gate(
            wave_plan=evidence["plan"],
            attempt_ledger=evidence["ledger"],
            accepted_results_adapter=subject.StaticAcceptedResultsAdapterV2(
                evidence["snapshot"]
            ),
            lifecycle_chain_adapter=_lifecycle_adapter(
                evidence, chain=running
            ),
            merge_view_root=(tmp_path / "running-vm-view").resolve(),
        )
    assert not (tmp_path / "running-vm-view").exists()


def test_lifecycle_launch_receipt_and_pair_done_lineage_drift_fail_closed(
    evidence: dict[str, Any]
) -> None:
    forged_chain = copy.deepcopy(evidence["lifecycle_chain"])
    forged_chain["wave_proofs"][0]["selected_attempts"][0][
        "launch_receipt_sha256"
    ] = _sha("foreign-accepted-launch")
    _reseal_lifecycle_chain(forged_chain, 0)
    with pytest.raises(ValueError, match="execution ledger mapping"):
        subject.validate_validated_lifecycle_chain(
            wave_plan=evidence["plan"],
            attempt_ledger=evidence["ledger"],
            value=forged_chain,
        )

    accepted = subject.validate_accepted_results_snapshot(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        validated_lifecycle_chain=evidence["lifecycle_chain"],
        value=evidence["snapshot"],
    )
    jobs = copy.deepcopy(list(accepted.jobs))
    reference = next(
        row
        for row in jobs
        if row["meta"]["job_id"] == "reference-shard-00"
    )
    reference["transport_done"]["prelaunch_authorization_sha256"] = _sha(
        "foreign-pair-authorization"
    )
    with pytest.raises(ValueError, match="launch lineage"):
        subject._pair_launch_lineage_audit(
            plan=evidence["plan"],
            pair_atomic=evidence["snapshot"]["pair_atomic_acceptance"],
            jobs=jobs,
            validated_lifecycle_chain=evidence["lifecycle_chain"],
        )


def test_controller_receiver_adapter_replays_three_producer_chains_and_normalizes(
    evidence: dict[str, Any]
) -> None:
    proofs, receipts = _controller_receiver_sources(
        evidence["plan"], evidence["ledger"]
    )
    calls: list[int] = []

    def callback(index: int):
        def replay(_plan: dict, _ledger: dict) -> dict[str, Any]:
            calls.append(index)
            return copy.deepcopy(proofs[index])

        return replay

    adapter = subject.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=[
            callback(index) for index in range(len(proofs))
        ],
        receiver_receipts=receipts,
        receiver_validator=lambda _plan, receipt: receipt,
    )
    chain = adapter.load_validated_lifecycle_chain(
        wave_plan=evidence["plan"], attempt_ledger=evidence["ledger"]
    )
    assert calls == [0, 1, 2]
    assert chain["wave_proof_count"] == 3
    assert chain["accepted_launch_receipt_count"] == 20
    assert chain["running_instance_count"] == 0
    assert chain["wave_proofs"][0]["launch_mapping_sha256"] == (
        controller.canonical_sha256(proofs[0]["selected_instance_mapping"])
    )
    assert chain["wave_proofs"][0]["launch_mapping_sha256"] != proofs[0][
        "actual_launch_receipt"
    ]["planned_mapping_receipt_sha256"]
    assert subject.validate_validated_lifecycle_chain(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        value=chain,
    ) == chain

    wrong_domain = copy.deepcopy(proofs)
    wrong_body = {
        key: value
        for key, value in wrong_domain[0].items()
        if key != "proof_sha256"
    }
    assert controller.canonical_sha256(wrong_body) != (
        subject.canonical_sha256(wrong_body)
    )
    wrong_domain[0]["proof_sha256"] = subject.canonical_sha256(wrong_body)
    wrong_adapter = subject.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=[
            (
                lambda _plan, _ledger, index=index: copy.deepcopy(
                    wrong_domain[index]
                )
            )
            for index in range(len(wrong_domain))
        ],
        receiver_receipts=receipts,
        receiver_validator=lambda _plan, receipt: receipt,
    )
    with pytest.raises(ValueError, match="controller lifecycle proof digest changed"):
        wrong_adapter.load_validated_lifecycle_chain(
            wave_plan=evidence["plan"], attempt_ledger=evidence["ledger"]
        )

    forged = copy.deepcopy(proofs)
    forged[0]["selected_instance_mapping"][0]["exact_instance_created"] = False
    forged[0]["exact_created_instance_count"] -= 1
    forged[0]["exact_uncreated_instance_count"] += 1
    forged[0]["create_classification"] = "partial_selected_created"
    forged[0]["proof_sha256"] = controller.canonical_sha256(
        {
            key: value
            for key, value in forged[0].items()
            if key != "proof_sha256"
        }
    )
    rejected = subject.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=[
            (lambda _plan, _ledger, index=index: copy.deepcopy(forged[index]))
            for index in range(len(forged))
        ],
        receiver_receipts=receipts,
        receiver_validator=lambda _plan, receipt: receipt,
    )
    with pytest.raises(ValueError, match="authority or coverage"):
        rejected.load_validated_lifecycle_chain(
            wave_plan=evidence["plan"], attempt_ledger=evidence["ledger"]
        )


def test_transport_done_uses_newline_free_canonical_json(tmp_path: Path) -> None:
    value = {"schema": "transport-done-test", "done": True}
    path = tmp_path / "DONE.json"
    newline_free = controller.canonical_bytes(value)
    newline_terminated = subject.canonical_bytes(value)
    assert newline_free != newline_terminated

    path.write_bytes(newline_terminated)
    with pytest.raises(ValueError, match="canonical newline-free JSON"):
        subject._read_transport_done_canonical(path, "transport DONE")

    path.write_bytes(newline_free)
    assert subject._read_transport_done_canonical(
        path, "transport DONE"
    ) == value


def test_retry_execution_chain_requires_every_receipt_once(
    tmp_path: Path,
) -> None:
    retry = _fixture(tmp_path, retry_pair_shard=0)
    proofs, receipts = _controller_receiver_sources(
        retry["plan"], retry["ledger"]
    )

    adapter = subject.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=[
            (
                lambda _plan, _ledger, index=index: copy.deepcopy(
                    proofs[index]
                )
            )
            for index in range(len(proofs))
        ],
        receiver_receipts=receipts,
        receiver_validator=lambda _plan, receipt: receipt,
    )
    chain = adapter.load_validated_lifecycle_chain(
        wave_plan=retry["plan"], attempt_ledger=retry["ledger"]
    )
    assert chain["execution_transition_count"] == 4
    assert chain["wave_indices"] == [0, 0, 1, 2]
    assert [
        (row["accepted_attempt_count"], row["failed_attempt_count"])
        for row in chain["wave_proofs"]
    ] == [(6, 2), (2, 0), (8, 0), (4, 0)]
    assert {
        row["attempt_id"]
        for row in chain["wave_proofs"][1]["selected_attempts"]
    } == {"a01"}

    missing = subject.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=[
            (
                lambda _plan, _ledger, index=index: copy.deepcopy(
                    proofs[index]
                )
            )
            for index in range(len(proofs) - 1)
        ],
        receiver_receipts=receipts[:-1],
        receiver_validator=lambda _plan, receipt: receipt,
    )
    with pytest.raises(ValueError, match="every observed execution receipt"):
        missing.load_validated_lifecycle_chain(
            wave_plan=retry["plan"], attempt_ledger=retry["ledger"]
        )

    duplicate_receipts = copy.deepcopy(receipts)
    duplicate_receipts[1] = copy.deepcopy(duplicate_receipts[0])
    duplicate = subject.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=[
            (
                lambda _plan, _ledger, index=index: copy.deepcopy(
                    proofs[index]
                )
            )
            for index in range(len(proofs))
        ],
        receiver_receipts=duplicate_receipts,
        receiver_validator=lambda _plan, receipt: receipt,
    )
    with pytest.raises(ValueError, match="receiver"):
        duplicate.load_validated_lifecycle_chain(
            wave_plan=retry["plan"], attempt_ledger=retry["ledger"]
        )


def test_partial_create_closeout_is_replayed_without_inventing_done(
    tmp_path: Path,
) -> None:
    retry = _fixture(tmp_path, retry_pair_shard=0)
    failed_jobs = {"candidate-shard-00", "reference-shard-00"}
    first_execution_jobs = set(retry["plan"]["waves"][0]["job_ids"])
    proofs, receipts = _controller_receiver_sources(
        retry["plan"],
        retry["ledger"],
        created_jobs_by_execution={0: first_execution_jobs - failed_jobs},
    )

    chain = subject.normalize_controller_receiver_lifecycle_chain(
        wave_plan=retry["plan"],
        attempt_ledger=retry["ledger"],
        controller_proofs=proofs,
        receiver_receipts=receipts,
    )
    partial = chain["wave_proofs"][0]
    assert partial["exact_created_attempt_count"] == 6
    assert partial["exact_uncreated_attempt_count"] == 2
    assert partial["actual_launch_receipt_present"] is False
    assert partial["actual_launch_receipt_sha256"] is None
    assert partial["prelaunch_authorization_sha256"] is None
    assert partial["wave_launch_receipt_sha256"] == (
        proofs[0]["gce_create_receipt"]["receipt_sha256"]
    )
    uncreated = [
        row for row in partial["selected_attempts"]
        if not row["exact_instance_created"]
    ]
    assert {row["job_id"] for row in uncreated} == failed_jobs
    assert all(row["worker_principal"] is None for row in uncreated)
    assert all(row["valid_done_observed"] is False for row in uncreated)

    forged_receipts = copy.deepcopy(receipts)
    forged = next(
        row for row in forged_receipts[0]["attempt_results"]
        if row["job_id"] in failed_jobs
    )
    forged["valid_done_observed"] = True
    with pytest.raises(ValueError, match="execution/controller proof"):
        subject.normalize_controller_receiver_lifecycle_chain(
            wave_plan=retry["plan"],
            attempt_ledger=retry["ledger"],
            controller_proofs=proofs,
            receiver_receipts=forged_receipts,
        )


def test_zero_create_closeout_can_retry_without_worker_or_done_lineage(
    tmp_path: Path,
) -> None:
    retry = _fixture(tmp_path, retry_all_wave0=True)
    proofs, receipts = _controller_receiver_sources(
        retry["plan"],
        retry["ledger"],
        created_jobs_by_execution={0: set()},
    )

    chain = subject.normalize_controller_receiver_lifecycle_chain(
        wave_plan=retry["plan"],
        attempt_ledger=retry["ledger"],
        controller_proofs=proofs,
        receiver_receipts=receipts,
    )
    zero = chain["wave_proofs"][0]
    retry_success = chain["wave_proofs"][1]
    assert zero["wave_index"] == retry_success["wave_index"] == 0
    assert zero["accepted_attempt_count"] == 0
    assert zero["failed_attempt_count"] == 8
    assert zero["exact_created_attempt_count"] == 0
    assert zero["exact_uncreated_attempt_count"] == 8
    assert all(
        row["worker_principal"] is None
        and row["valid_done_observed"] is False
        and row["terminal_status"] == "failed"
        for row in zero["selected_attempts"]
    )
    assert retry_success["accepted_attempt_count"] == 8
    assert retry_success["exact_created_attempt_count"] == 8


def test_worker_principal_is_job_specific_and_pair_reuse_fails_closed(
    evidence: dict[str, Any],
) -> None:
    accepted = subject.validate_accepted_results_snapshot(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        validated_lifecycle_chain=evidence["lifecycle_chain"],
        value=evidence["snapshot"],
    )
    jobs = copy.deepcopy(list(accepted.jobs))
    principals = [row["transport_done"]["worker_principal"] for row in jobs]
    assert len(set(principals)) == 8
    by_job = {row["meta"]["job_id"]: row for row in jobs}
    first_wave_jobs = evidence["plan"]["waves"][0]["job_ids"]
    second_wave_jobs = evidence["plan"]["waves"][1]["job_ids"]
    assert by_job[first_wave_jobs[0]]["transport_done"]["worker_principal"] == (
        by_job[second_wave_jobs[0]]["transport_done"]["worker_principal"]
    )

    candidate = by_job["candidate-shard-00"]
    reference = by_job["reference-shard-00"]
    reference["transport_done"]["worker_principal"] = candidate[
        "transport_done"
    ]["worker_principal"]
    with pytest.raises(ValueError, match="launch lineage or attempt lane"):
        subject._pair_launch_lineage_audit(
            plan=evidence["plan"],
            pair_atomic=evidence["snapshot"]["pair_atomic_acceptance"],
            jobs=jobs,
            validated_lifecycle_chain=evidence["lifecycle_chain"],
        )


def test_concrete_receiver_receipt_adapter_maps_locked_final_schema(
    tmp_path: Path, evidence: dict[str, Any]
) -> None:
    observed = evidence["snapshot"]["observed_inventory"]
    lifecycle_chain = copy.deepcopy(evidence["lifecycle_chain"])
    proof = lifecycle_chain["wave_proofs"][2]
    embedded_controller_body = {"schema": "validated-controller-fixture", "wave_index": 2}
    embedded_controller_proof = {
        **embedded_controller_body,
        "proof_sha256": controller.canonical_sha256(
            embedded_controller_body
        ),
    }
    proof["controller_validated_proof_sha256"] = embedded_controller_proof[
        "proof_sha256"
    ]
    transition_lifecycle_body = {
        "schema": "hu_m31_t3_step6d_full100_wave_transition_lifecycle_binding_v2",
        "observed_transition_digest": proof["observed_transition_digest"],
        "wave_index": 2,
        "wave_launch_receipt_sha256": proof["wave_launch_receipt_sha256"],
        "launch_mapping_sha256": proof["launch_mapping_sha256"],
        "accepted_launch_receipt_sha256s": [
            row["launch_receipt_sha256"] for row in proof["selected_attempts"]
        ],
        "lifecycle_proof_sha256": embedded_controller_proof["proof_sha256"],
        "controller_lifecycle_receipt_sha256": proof[
            "controller_lifecycle_receipt_sha256"
        ],
        "gce_absence_receipt_sha256": proof[
            "receiver_gce_absence_receipt_sha256"
        ],
        "worker_iam_cleanup_receipt_sha256": proof[
            "worker_iam_cleanup_receipt_sha256"
        ],
        "all_selected_instances_absent": True,
        "all_selected_boot_disks_absent": True,
        "worker_iam_bindings_absent": True,
    }
    transition_lifecycle = {
        **transition_lifecycle_body,
        "binding_sha256": subject.canonical_sha256(transition_lifecycle_body),
    }
    proof["receiver_transition_lifecycle_binding_sha256"] = transition_lifecycle[
        "binding_sha256"
    ]
    proof["proof_sha256"] = subject.canonical_sha256(
        {key: value for key, value in proof.items() if key != "proof_sha256"}
    )
    lifecycle_body = {
        key: value
        for key, value in lifecycle_chain.items()
        if key != "chain_sha256"
    }
    lifecycle_body["wave_proofs"] = lifecycle_chain["wave_proofs"]
    lifecycle_chain["chain_sha256"] = subject.canonical_sha256(
        lifecycle_body
    )
    receiver = {
        "status": "all_jobs_complete_exact_inventory_accepted",
        "attempt_ledger": evidence["ledger"],
        "expected_artifact_inventory": evidence["snapshot"]["expected_inventory"],
        "observed_artifact_inventory": observed,
        "materialized_paths": [row["path"] for row in observed["objects"]],
        "next_resume_plan": {"all_jobs_complete": True},
        "wave_index": 2,
        "attempt_results": [
            {
                "job_id": row["job_id"],
                "launch_receipt_sha256": row["launch_receipt_sha256"],
                "lifecycle_proof_sha256": embedded_controller_proof[
                    "proof_sha256"
                ],
                "gce_absence_receipt_sha256": proof[
                    "receiver_gce_absence_receipt_sha256"
                ],
                "terminal_status": "accepted",
            }
            for row in proof["selected_attempts"]
        ],
        "transition_lifecycle_binding": transition_lifecycle,
        "done_is_only_worker_commit_marker": True,
        "acceptance_create_only": True,
        "pair_atomicity_enforced": True,
        "validated_lifecycle_proof": embedded_controller_proof,
        "lifecycle_proof_sha256": embedded_controller_proof["proof_sha256"],
        "gce_absence_receipt_sha256": proof[
            "receiver_gce_absence_receipt_sha256"
        ],
        "worker_iam_cleanup_receipt_sha256": proof[
            "worker_iam_cleanup_receipt_sha256"
        ],
        "worker_iam_bindings_absent": True,
        "vm_lifecycle_mutation_performed": False,
        "current_profile_changed": False,
        "local_destination": str(evidence["accepted_root"]),
    }
    receipt_path = (tmp_path / "receiver-final.json").resolve()
    receipt_path.write_bytes(subject.canonical_bytes(receiver))
    adapter = subject.ReceiverReceiptAcceptedResultsAdapterV2(
        receiver_receipt=receiver,
        receiver_receipt_path=receipt_path,
        expected_startup_sha256=subject.EXPECTED_STARTUP_SHA256,
        content_payload_sha256=CONTENT_SHA,
        outer_manifest_sha256=OUTER_SHA,
        validator=lambda _plan, value: value,
    )
    snapshot = adapter.load_accepted_results(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        validated_lifecycle_chain=lifecycle_chain,
    )
    validated = subject.validate_accepted_results_snapshot(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        validated_lifecycle_chain=lifecycle_chain,
        value=snapshot,
    )
    assert validated.snapshot["accepted_object_count"] == 440
    assert validated.snapshot["receiver_receipt_sha256"] == _sha(
        receipt_path.read_bytes()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "candidate-shard-09"),
        ("source_role", "reference"),
        ("attempt_id", "a01"),
        ("run_contract_digest", "a" * 64),
        ("binary_sha256", "b" * 64),
        ("image_digest", "sha256:" + "c" * 64),
        ("allocation_digest", "d" * 64),
        ("root_digest", "e" * 64),
    ],
)
def test_transport_done_job_attempt_role_runtime_and_root_drift_fail_closed(
    evidence: dict[str, Any], field: str, value: Any
) -> None:
    accepted = subject.validate_accepted_results_snapshot(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        validated_lifecycle_chain=evidence["lifecycle_chain"],
        value=evidence["snapshot"],
    )
    item = accepted.jobs[0]
    forged = copy.deepcopy(item["transport_done"])
    forged[field] = value
    observed = {row["path"]: row for row in accepted.observed["objects"]}
    with pytest.raises(ValueError, match="binding drifted"):
        subject._validate_transport_done(
            value=forged,
            plan=evidence["plan"],
            meta=item["meta"],
            expected_record=item["record"],
            observed_by_path=observed,
            accepted_attempt_id=item["attempt_id"],
            content_payload_sha256=CONTENT_SHA,
            outer_manifest_sha256=OUTER_SHA,
        )


def test_rng_namespace_audit_is_exact_600_unique_and_rejects_drift(
    evidence: dict[str, Any],
) -> None:
    audit = subject._rng_namespace_audit(evidence["plan"])
    assert audit["seed_count"] == 600
    assert audit["unique_seed_count"] == 600
    assert audit["duplicate_seed_count"] == 0
    assert audit["missing_seed_count"] == 0
    forged = copy.deepcopy(evidence["plan"])
    forged["seed_contract"]["namespace_bases"]["candidate"] = forged[
        "seed_contract"
    ]["namespace_bases"]["hand"]
    with pytest.raises(ValueError, match="RNG namespace"):
        subject._rng_namespace_audit(forged)


def test_startup_transport_root_and_done_identity_match_wave_contract(
    evidence: dict[str, Any],
) -> None:
    startup = Path("scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh")
    assert _sha(startup.read_bytes()) == subject.EXPECTED_STARTUP_SHA256
    assert subject.EXPECTED_STARTUP_SHA256 == (
        "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
    )
    job_id = evidence["plan"]["coverage"]["job_ids"][0]
    done = evidence["transport"][job_id]
    roots = [
        {"hand_index": hand, "sha256": artifact["sha256"]}
        for hand, artifact in zip(
            done["work_hand_indices"], done["artifacts"][::2], strict=True
        )
    ]
    assert done["root_digest"] == wave.canonical_sha256(roots)
    assert done["done_identity_sha256"] == wave.expected_done_identity_sha256(
        evidence["plan"],
        job_id=job_id,
        attempt_id=done["attempt_id"],
        root_digest=done["root_digest"],
    )


def test_superseded_run004_guard_and_current_startup_are_fail_closed(
    evidence: dict[str, Any],
) -> None:
    forged = copy.deepcopy(evidence["plan"])
    forged["schedule_sha256"] = subject.RUN004_PRELAUNCH_TARGET[
        "wave_plan_sha256"
    ]
    with pytest.raises(ValueError, match="superseded run004"):
        subject._enforce_prelaunch_guard(
            forged,
            startup_sha256=subject.EXPECTED_STARTUP_SHA256,
            content_payload_sha256=CONTENT_SHA,
            outer_manifest_sha256=OUTER_SHA,
        )
    with pytest.raises(ValueError, match="startup SHA"):
        subject._enforce_prelaunch_guard(
            evidence["plan"],
            startup_sha256="f" * 64,
            content_payload_sha256=CONTENT_SHA,
            outer_manifest_sha256=OUTER_SHA,
        )

    run005 = copy.deepcopy(evidence["plan"])
    run005["run_name"] = subject.RUN005_PRELAUNCH_TARGET["run_name"]
    run005["schedule_sha256"] = subject.RUN005_PRELAUNCH_TARGET[
        "wave_plan_sha256"
    ]
    run005["execution_identity_sha256"] = subject.RUN005_PRELAUNCH_TARGET[
        "execution_identity_sha256"
    ]
    subject._enforce_prelaunch_guard(
        run005,
        startup_sha256=subject.RUN005_PRELAUNCH_TARGET["startup_sha256"],
        content_payload_sha256=subject.RUN005_PRELAUNCH_TARGET[
            "content_payload_sha256"
        ],
        outer_manifest_sha256=subject.RUN005_PRELAUNCH_TARGET[
            "outer_manifest_sha256"
        ],
    )
    with pytest.raises(ValueError, match="run005 prelaunch"):
        subject._enforce_prelaunch_guard(
            run005,
            startup_sha256=subject.RUN005_PRELAUNCH_TARGET["startup_sha256"],
            content_payload_sha256="a" * 64,
            outer_manifest_sha256=subject.RUN005_PRELAUNCH_TARGET[
                "outer_manifest_sha256"
            ],
        )


def test_receiver_receipt_and_local_extra_file_tamper_fail_closed(
    evidence: dict[str, Any],
) -> None:
    receipt = evidence["receiver_receipt"]
    original = receipt.read_bytes()
    receipt.write_bytes(runner.canonical_bytes({"receiver": "tampered"}))
    try:
        with pytest.raises(ValueError, match="snapshot boundary changed"):
            subject.validate_accepted_results_snapshot(
                wave_plan=evidence["plan"],
                attempt_ledger=evidence["ledger"],
                validated_lifecycle_chain=evidence["lifecycle_chain"],
                value=evidence["snapshot"],
            )
    finally:
        receipt.write_bytes(original)

    artifact_root = _local(
        evidence["accepted_root"],
        evidence["plan"]["artifact_contract"]["prefix"],
    )
    extra = artifact_root / "EXTRA.json"
    extra.write_bytes(b"{}\n")
    try:
        with pytest.raises(ValueError, match="missing or extra paths"):
            subject.validate_accepted_results_snapshot(
                wave_plan=evidence["plan"],
                attempt_ledger=evidence["ledger"],
                validated_lifecycle_chain=evidence["lifecycle_chain"],
                value=evidence["snapshot"],
            )
    finally:
        extra.unlink()


def test_gate_receipt_digest_and_forbidden_authority_are_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, Any],
) -> None:
    _install_scientific(monkeypatch, evidence)
    receipt = subject.merge_accepted_results_to_scientific_gate(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        accepted_results_adapter=subject.StaticAcceptedResultsAdapterV2(
            evidence["snapshot"]
        ),
        lifecycle_chain_adapter=_lifecycle_adapter(evidence),
        merge_view_root=(tmp_path / "view").resolve(),
    )
    forged = copy.deepcopy(receipt)
    forged["training_authorized"] = True
    forged["receipt_sha256"] = subject.canonical_sha256(
        {key: item for key, item in forged.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_scientific_gate_receipt_value(forged)

    running = copy.deepcopy(receipt)
    running["validated_lifecycle_chain"]["wave_proofs"][0][
        "all_selected_instances_absent"
    ] = False
    running["validated_lifecycle_chain"]["wave_proofs"][0][
        "running_instance_count"
    ] = 1
    _reseal_lifecycle_chain(running["validated_lifecycle_chain"], 0)
    running["validated_lifecycle_chain_sha256"] = running[
        "validated_lifecycle_chain"
    ]["chain_sha256"]
    running["receipt_sha256"] = subject.canonical_sha256(
        {key: item for key, item in running.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="not fully quiescent"):
        subject.validate_scientific_gate_receipt_value(running)
