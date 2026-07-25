from __future__ import annotations

import hashlib
import json
import copy
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as launch_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as gce_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-004"
PROJECT = "ofc-solver-485418"
ZONE = "asia-northeast1-b"
PACKAGE_SHA = "2" * 64
IMAGE_DIGEST = "sha256:" + "3" * 64
STARTUP_SHA = "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"


def _sha(label: str | bytes) -> str:
    raw = label if isinstance(label, bytes) else label.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _done_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


class FakeStore:
    def __init__(self) -> None:
        self._objects: dict[str, tuple[dict, bytes]] = {}
        self._next_generation = 1000
        self.create_count = 0
        self.duplicate_path: str | None = None
        self.create_response_generation_delta = 0

    def put(self, path: str, raw: bytes) -> dict:
        self._next_generation += 1
        record = {
            "path": path,
            "generation": self._next_generation,
            "bytes": len(raw),
            "sha256": _sha(raw),
        }
        self._objects[path] = (record, raw)
        return deepcopy(record)

    def remove(self, path: str) -> None:
        self._objects.pop(path)

    def list_prefix(self, *, prefix: str) -> list[dict]:
        rows = [
            deepcopy(record)
            for path, (record, _) in sorted(self._objects.items())
            if path.startswith(prefix)
        ]
        if self.duplicate_path is not None:
            rows.append(deepcopy(self._objects[self.duplicate_path][0]))
        return rows

    def read_current(
        self, *, path: str, allow_missing: bool = False
    ) -> tuple[dict, bytes] | None:
        value = self._objects.get(path)
        if value is None:
            if allow_missing:
                return None
            raise KeyError(path)
        return deepcopy(value[0]), value[1]

    def read_bytes(self, *, path: str, generation: int) -> bytes:
        record, raw = self._objects[path]
        if record["generation"] != generation:
            raise KeyError((path, generation))
        return raw

    def create_only(self, *, path: str, data: bytes) -> dict:
        if path in self._objects:
            raise FileExistsError(path)
        self.create_count += 1
        record = self.put(path, data)
        record["generation"] += self.create_response_generation_delta
        return record


class FakeIamBackend:
    def __init__(self) -> None:
        self.policy = {
            "kind": "storage#policy",
            "resourceId": f"projects/_/buckets/{worker_iam_v2.BUCKET}",
            "version": 3,
            "etag": "BwWInitialEtag==",
            "bindings": [
                {
                    "role": "roles/storage.objectViewer",
                    "members": ["user:unrelated@example.com"],
                }
            ],
            "auditConfigs": [],
        }
        self.set_count = 0

    def get_bucket_policy(self) -> dict:
        return copy.deepcopy(self.policy)

    def set_bucket_policy(self, *, policy: dict) -> dict:
        supplied = copy.deepcopy(dict(policy))
        if supplied["etag"] != self.policy["etag"]:
            raise worker_iam_v2.WorkerIamCasError("fake stale ETag")
        self.set_count += 1
        supplied["etag"] = f"BwWAfterSet{self.set_count}=="
        self.policy = supplied
        return copy.deepcopy(self.policy)


def _worker_iam_chain(
    plan: dict,
    ledger: dict,
    resume: dict,
    bootstraps: list[dict],
    *,
    recovered_cleanup: bool = False,
) -> dict[str, dict]:
    content_sha = bootstraps[0]["content_payload_sha256"]
    manifest_sha = bootstraps[0]["outer_manifest_sha256"]
    content = {
        "immutable_content_prefix": (
            f"{package_v2.CONTENT_PREFIX_ROOT}/{content_sha}"
        ),
        "content_payload_sha256": content_sha,
        "outer_manifest_sha256": manifest_sha,
    }
    iam = worker_iam_v2.build_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=resume["resume_wave_index"],
        **content,
        issued_at_unix_seconds=1_800_000_000,
        service_accounts_by_job={
            row["job_id"]: row["worker_principal"] for row in bootstraps
        },
    )
    backend = FakeIamBackend()
    common = {
        "iam_plan": iam,
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        **content,
    }
    prepare = worker_iam_v2.prepare_worker_iam(**common, backend=backend)
    install = worker_iam_v2.install_worker_iam(
        **common, prepare_receipt=prepare, backend=backend
    )
    readback = worker_iam_v2.readback_worker_iam(
        **common,
        prepare_receipt=prepare,
        install_receipt=install,
        backend=backend,
    )
    cleanup = worker_iam_v2.cleanup_worker_iam(
        **common,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        backend=backend,
    )
    if recovered_cleanup:
        recovered = {
            key: value
            for key, value in cleanup.items()
            if key != "receipt_sha256"
        }
        recovered["status"] = (
            "recovered_exact_wave_binding_absence_after_outcome_ambiguity"
        )
        recovered["recovered_after_outcome_ambiguity"] = True
        recovered["removed_binding_count"] = 0
        recovered["set_attempt_count"] = 0
        recovered["cloud_mutation_performed"] = False
        recovered["source_mutation_outcome"] = "unknown"
        recovered["receipt_sha256"] = worker_iam_v2.canonical_sha256(recovered)
        cleanup = worker_iam_v2.validate_cleanup_receipt(
            **common,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            value=recovered,
        )
    assert backend.set_count == 2
    return {
        "worker_iam_plan": iam,
        "worker_iam_prepare_receipt": prepare,
        "worker_iam_install_receipt": install,
        "worker_iam_readback_receipt": readback,
        "worker_iam_cleanup_receipt": cleanup,
    }


def _sealed_gce(value: dict) -> dict:
    return {**value, "receipt_sha256": gce_v2.canonical_sha256(value)}


def _validated_lifecycle_proof(
    *,
    plan: dict,
    ledger: dict,
    resume: dict,
    terminals: list[dict],
    cleanup: dict,
    observed_at: str,
) -> dict:
    reasons = {row["job_id"]: row["terminal_reason"] for row in terminals}
    context = {
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
    }
    context_sha = controller_v2.canonical_sha256(context)
    launch_event_sha = _sha(f"launch-event-{resume['resume_sha256']}")
    created_selected = [
        row
        for row in resume["selected_attempts"]
        if reasons[row["job_id"]] != "create_missing_before_done"
    ]
    create_rows = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_name": row["instance_id"],
            "provider_instance_id": str(10_000 + index),
            "provider_boot_disk_id": str(20_000 + index),
            "operation_id": str(30_000 + index),
            "spec_sha256": _sha(f"spec-{row['instance_id']}"),
        }
        for index, row in enumerate(created_selected)
    ]
    create_complete = len(create_rows) == len(resume["selected_attempts"])
    create = _sealed_gce(
        {
            "schema": gce_v2.GCE_CREATE_RECEIPT_SCHEMA,
            "status": (
                "exact_selected_gce_create_complete"
                if create_complete
                else "partial_exact_owned_gce_create"
            ),
            "rows": create_rows,
            "created_instance_count": len(create_rows),
            "create_complete": create_complete,
            "observed_at_utc": observed_at,
        }
    )
    delete = _sealed_gce(
        {
            "schema": gce_v2.GCE_DELETE_RECEIPT_SCHEMA,
            "status": "exact_owned_gce_delete_operations_complete",
            "create_receipt_sha256": create["receipt_sha256"],
            "rows": [],
            "observed_at_utc": observed_at,
        }
    )
    names = [row["instance_id"] for row in resume["selected_attempts"]]
    absence = _sealed_gce(
        {
            "schema": gce_v2.GCE_ABSENCE_RECEIPT_SCHEMA,
            "status": "exact_owned_gce_instances_and_disks_absent",
            "create_receipt_sha256": create["receipt_sha256"],
            "delete_receipt_sha256": delete["receipt_sha256"],
            "project": PROJECT,
            "zone": ZONE,
            "checked_instance_count": len(names),
            "absent_instance_names": names,
            "absent_boot_disk_names": names,
            "all_instances_absent": True,
            "all_boot_disks_absent": True,
            "read_only": True,
            "observed_at_utc": observed_at,
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
    )
    actual = None
    actual_rows: list[dict] = []
    if create_complete:
        actual_rows = [
            {
                "job_id": row["job_id"],
                "source_role": row["source_role"],
                "attempt_id": row["attempt_id"],
                "instance_id": row["instance_id"],
                "operation_id": str(40_000 + index),
                "instance_status": "RUNNING",
            }
            for index, row in enumerate(resume["selected_attempts"])
        ]
        actual_core = {
            "schema": "test-normalized-actual-launch-v2",
            "rows": actual_rows,
        }
        actual = {
            **actual_core,
            "receipt_sha256": wave_v2.canonical_sha256(actual_core),
        }
    create_by_job = {row["job_id"]: row for row in create_rows}
    actual_by_job = {row["job_id"]: row for row in actual_rows}
    mappings = []
    for row in resume["selected_attempts"]:
        created = create_by_job.get(row["job_id"])
        launched = actual_by_job.get(row["job_id"])
        exact_created = created is not None
        launch_identity = controller_v2.canonical_sha256(
            {
                "schema": "hu_m31_t3_step6d_full100_attempt_launch_identity_v2",
                "controller_context_sha256": context_sha,
                "launch_event_sha256": launch_event_sha,
                "gce_create_receipt_sha256": create["receipt_sha256"],
                "job_id": row["job_id"],
                "source_role": row["source_role"],
                "attempt_id": row["attempt_id"],
                "instance_id": row["instance_id"],
                "exact_instance_created": exact_created,
            }
        )
        mappings.append(
            {
                **row,
                "launch_receipt_sha256": launch_identity,
                "exact_instance_created": exact_created,
                "provider_instance_id": (
                    None if created is None else created["provider_instance_id"]
                ),
                "provider_boot_disk_id": (
                    None if created is None else created["provider_boot_disk_id"]
                ),
                "gce_spec_sha256": (
                    None if created is None else created["spec_sha256"]
                ),
                "gce_operation_id": (
                    None if created is None else created["operation_id"]
                ),
                "actual_launch_operation_id": (
                    None if launched is None else launched["operation_id"]
                ),
                "actual_launch_instance_status": (
                    None if launched is None else launched["instance_status"]
                ),
                "ownership_label": f"owned-{row['job_id']}",
                "final_instance_absent": True,
                "final_boot_disk_absent": True,
            }
        )
    selected_count = len(mappings)
    created_count = len(create_rows)
    classification = (
        "all_selected_created"
        if created_count == selected_count
        else "no_selected_created"
        if created_count == 0
        else "partial_selected_created"
    )
    core = {
        "schema": controller_v2.LIFECYCLE_PROOF_SCHEMA,
        "status": "controller_journal_and_all_producer_receipts_revalidated",
        "controller_context_sha256": context_sha,
        **context,
        "lifecycle_event_sha256": _sha(f"closeout-{resume['resume_sha256']}"),
        "lifecycle_receipt_sha256": _sha(
            f"lifecycle-receipt-{resume['resume_sha256']}"
        ),
        "launch_event_sha256": launch_event_sha,
        "delete_event_sha256": _sha(f"delete-{resume['resume_sha256']}"),
        "absence_event_sha256": _sha(f"absence-{resume['resume_sha256']}"),
        "worker_iam_cleanup_event_sha256": _sha(
            f"iam-cleanup-{resume['resume_sha256']}"
        ),
        "launch_bundle_sha256": _sha(f"bundle-{resume['resume_sha256']}"),
        "gce_create_receipt": create,
        "actual_launch_receipt": actual,
        "gce_delete_receipt": delete,
        "gce_absence_receipt": absence,
        "worker_iam_cleanup_receipt": cleanup,
        "selected_instance_mapping": mappings,
        "gce_create_rows": create_rows,
        "actual_launch_rows": actual_rows,
        "selected_instance_count": selected_count,
        "exact_created_instance_count": created_count,
        "exact_uncreated_instance_count": selected_count - created_count,
        "create_classification": classification,
        "actual_launch_receipt_present": actual is not None,
        "lifecycle_attested_at_utc": observed_at,
        "journal_event_count": 12,
        "journal_hash_chain_valid": True,
        "all_producer_receipts_valid": True,
        "all_owned_instances_absent": True,
        "all_owned_boot_disks_absent": True,
        "worker_iam_bindings_absent": True,
        "additional_create_authorized": False,
        "current_profile_changed": False,
    }
    return {**core, "proof_sha256": controller_v2.canonical_sha256(core)}


@pytest.fixture(scope="module")
def plan() -> dict:
    return wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt="0123456789abcdef0123456789abcdef",
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )


def _initial(plan: dict) -> tuple[dict, dict]:
    baseline = wave_v2.build_observed_transition(
        plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[baseline])
    return ledger, wave_v2.build_resume_plan(plan, attempt_ledger=ledger)


def _binding(name: str) -> dict:
    return {
        "object_name": f"content/{name}",
        "sha256": _sha(f"binding-{name}"),
        "bytes": 100 + len(name),
    }


def _bootstraps(plan: dict, ledger: dict, resume: dict) -> list[dict]:
    result = []
    accounts = worker_iam_v2.default_service_accounts(plan, resume)
    for selected in resume["selected_attempts"]:
        value = {
            "schema": package_v2.JOB_BOOTSTRAP_SCHEMA,
            "status": "single_job_bootstrap_bound_to_prelaunch_authorization",
            "run_name": plan["run_name"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "wave_plan_sha256": plan["schedule_sha256"],
            "attempt_ledger_sha256": ledger["ledger_sha256"],
            "resume_sha256": resume["resume_sha256"],
            "observed_transition_digest": resume["observed_transition_digest"],
            "wave_index": resume["resume_wave_index"],
            "job_id": selected["job_id"],
            "source_role": selected["source_role"],
            "attempt_id": selected["attempt_id"],
            "instance_name": selected["instance_id"],
            "artifact_prefix": selected["artifact_prefix"],
            "bucket": "ofc-test-bucket",
            "content_prefix": "content/full100-wave-v2",
            "outer_manifest_sha256": _sha("outer-manifest"),
            "content_payload_sha256": _sha("content-payload"),
            "scientific_source": _binding("source.zip"),
            "scientific_manifest": _binding("science-manifest.json"),
            "wheelhouse": _binding("wheelhouse.zip"),
            "wheelhouse_manifest": _binding("wheelhouse-manifest.json"),
            "startup": {
                "object_name": "content/startup.sh",
                "sha256": launch_v2.EXPECTED_STARTUP_SHA256,
                "bytes": 12345,
            },
            "wave_plan": _binding("wave-plan.json"),
            "job_manifest": _binding(f"{selected['job_id']}.json"),
            "prelaunch_authorization_sha256": _sha("prelaunch-authorization"),
            "worker_principal": accounts[selected["job_id"]],
            "one_vm_one_job_one_role": True,
            "additional_create_authorized": False,
            "hidden_truth_exposed": False,
        }
        value["bootstrap_sha256"] = package_v2.canonical_sha256(value)
        result.append(value)
    return result


def _metadata(plan: dict) -> dict[str, dict]:
    return {row["job_id"]: row for row in plan["full100_plan"]["jobs"]}


def _root_raw(hand: int) -> bytes:
    return json.dumps(
        {"schema": "test-root-v1", "hand_index": hand},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _source_raw(job: str, role: str, hand: int) -> bytes:
    return json.dumps(
        {
            "schema": "test-source-hand-v1",
            "job_id": job,
            "source_role": role,
            "hand_index": hand,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _publish_attempt(
    *,
    plan: dict,
    ledger: dict,
    resume: dict,
    bootstrap: dict,
    selected: dict,
    store: FakeStore,
    complete: bool,
    partial_count: int = 1,
) -> dict | None:
    job = selected["job_id"]
    role = selected["source_role"]
    work = list(_metadata(plan)[job]["work_hand_indices"])
    artifacts = []
    roots = []
    published = 0
    for hand in work:
        root_path = f"roots/hand_{hand:03d}.json"
        source_path = f"hands/{role}/hand_{hand:03d}.json"
        root_raw = _root_raw(hand)
        source_raw = _source_raw(job, role, hand)
        roots.append({"hand_index": hand, "sha256": _sha(root_raw)})
        for relative, raw in ((root_path, root_raw), (source_path, source_raw)):
            artifacts.append(
                {"path": relative, "sha256": _sha(raw), "bytes": len(raw)}
            )
            if complete or published < partial_count:
                store.put(f"{selected['artifact_prefix']}/{relative}", raw)
            published += 1
    if not complete:
        return None
    root_digest = wave_v2.canonical_sha256(roots)
    done_identity = wave_v2.expected_done_identity_sha256(
        plan,
        job_id=job,
        attempt_id=selected["attempt_id"],
        root_digest=root_digest,
    )
    done = {
        "schema": subject.DONE_SCHEMA,
        "status": "complete_validated_single_job_attempt",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "observed_transition_digest": resume["observed_transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "job_id": job,
        "source_role": role,
        "attempt_id": selected["attempt_id"],
        "package_sha256": plan["runtime_binding"]["package_sha256"],
        "image_digest": plan["runtime_binding"]["image_digest"],
        "binary_sha256": plan["runtime_binding"]["binary_sha256_by_role"][role],
        "allocation_digest": plan["runtime_binding"]["allocation_digest"],
        "run_contract_digest": plan["run_contract_digest"],
        "root_digest": root_digest,
        "done_identity_sha256": done_identity,
        "content_payload_sha256": bootstrap["content_payload_sha256"],
        "outer_manifest_sha256": bootstrap["outer_manifest_sha256"],
        "prelaunch_authorization_sha256": bootstrap[
            "prelaunch_authorization_sha256"
        ],
        "worker_principal": bootstrap["worker_principal"],
        "work_hand_indices": work,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "runner_done_sha256": _sha(f"runner-DONE-{job}-{selected['attempt_id']}"),
        "metadata_hidden_truth_exposed": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    store.put(f"{selected['artifact_prefix']}/DONE.json", _done_bytes(done))
    return done


def _terminal(selected: dict, reason: str) -> dict:
    return {
        "job_id": selected["job_id"],
        "source_role": selected["source_role"],
        "attempt_id": selected["attempt_id"],
        "instance_id": selected["instance_id"],
        "terminal_reason": reason,
    }


def _publish_wave(
    *,
    plan: dict,
    ledger: dict,
    resume: dict,
    bootstraps: list[dict],
    store: FakeStore,
    failure: tuple[str, str] | None = None,
) -> tuple[list[dict], dict[str, dict | None]]:
    by_job = {row["job_id"]: row for row in bootstraps}
    terminals = []
    done = {}
    for selected in resume["selected_attempts"]:
        reason = (
            failure[1]
            if failure is not None and selected["job_id"] == failure[0]
            else "done_observed"
        )
        done[selected["job_id"]] = _publish_attempt(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstrap=by_job[selected["job_id"]],
            selected=selected,
            store=store,
            complete=reason == "done_observed",
            partial_count=(
                0
                if reason in {"timeout_before_done", "create_missing_before_done"}
                else 1
            ),
        )
        terminals.append(_terminal(selected, reason))
    return terminals, done


def _receive(
    *,
    plan: dict,
    ledger: dict,
    resume: dict,
    bootstraps: list[dict],
    terminals: list[dict],
    store: FakeStore,
    destination: Path,
    observed_at: str,
) -> dict:
    iam = _worker_iam_chain(plan, ledger, resume, bootstraps)
    proof = _validated_lifecycle_proof(
        plan=plan,
        ledger=ledger,
        resume=resume,
        terminals=terminals,
        cleanup=iam["worker_iam_cleanup_receipt"],
        observed_at=observed_at,
    )
    adapter = subject.ValidatedLifecycleProofAdapterV2(lambda: proof)
    return subject.receive_wave_results(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        job_bootstraps=bootstraps,
        terminal_observations=terminals,
        lifecycle_proof_adapter=adapter,
        **iam,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc=observed_at,
        store=store,
        destination=destination,
        readback_source="local_observed_fixture",
    )


def test_startup_done_identity_cross_contract_and_complete_wave(
    tmp_path: Path, plan: dict
) -> None:
    startup = Path("scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh")
    assert _sha(startup.read_bytes()) == STARTUP_SHA
    assert launch_v2.EXPECTED_STARTUP_SHA256 == STARTUP_SHA
    source = startup.read_text(encoding="utf-8")
    assert 'canonical(roots) + b"\\n"' in source
    assert 'canonical(identity_value) + b"\\n"' in source

    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)
    store = FakeStore()
    terminals, done = _publish_wave(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        store=store,
    )
    receipt = _receive(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        terminals=terminals,
        store=store,
        destination=tmp_path / "accepted",
        observed_at="2026-07-22T00:00:01Z",
    )
    assert receipt["status"] == "wave_results_recorded_next_wave_ready"
    assert len(receipt["accepted_job_ids"]) == 8
    assert receipt["failed_job_ids"] == []
    assert receipt["next_resume_plan"]["resume_wave_index"] == 1
    assert store.create_count == 8
    first = resume["selected_attempts"][0]
    first_done = done[first["job_id"]]
    assert first_done is not None
    assert first_done["done_identity_sha256"] == wave_v2.expected_done_identity_sha256(
        plan,
        job_id=first["job_id"],
        attempt_id=first["attempt_id"],
        root_digest=first_done["root_digest"],
    )
    assert receipt["expected_artifact_inventory"] is None
    assert subject.validate_receiver_receipt(plan, receipt) == receipt


def _coherently_replace_cleanup(
    receipt: Mapping[str, Any], cleanup: Mapping[str, Any]
) -> dict[str, Any]:
    bad = deepcopy(dict(receipt))
    bad_cleanup = deepcopy(dict(cleanup))
    cleanup_body = {
        key: value
        for key, value in bad_cleanup.items()
        if key != "receipt_sha256"
    }
    bad_cleanup["receipt_sha256"] = worker_iam_v2.canonical_sha256(
        cleanup_body
    )
    bad["worker_iam_cleanup_receipt"] = bad_cleanup
    bad["worker_iam_cleanup_receipt_sha256"] = bad_cleanup[
        "receipt_sha256"
    ]
    bad_proof = deepcopy(bad["validated_lifecycle_proof"])
    bad_proof["worker_iam_cleanup_receipt"] = bad_cleanup
    proof_body = {
        key: value
        for key, value in bad_proof.items()
        if key != "proof_sha256"
    }
    bad_proof["proof_sha256"] = controller_v2.canonical_sha256(proof_body)
    bad["validated_lifecycle_proof"] = bad_proof
    bad["lifecycle_proof_sha256"] = bad_proof["proof_sha256"]
    for attempt in bad["attempt_results"]:
        attempt["lifecycle_proof_sha256"] = bad_proof["proof_sha256"]
    lifecycle = bad["transition_lifecycle_binding"]
    lifecycle["lifecycle_proof_sha256"] = bad_proof["proof_sha256"]
    lifecycle["worker_iam_cleanup_receipt_sha256"] = bad_cleanup[
        "receipt_sha256"
    ]
    lifecycle_body = {
        key: value
        for key, value in lifecycle.items()
        if key != "binding_sha256"
    }
    lifecycle["binding_sha256"] = wave_v2.canonical_sha256(lifecycle_body)
    receipt_body = {
        key: value for key, value in bad.items() if key != "receipt_sha256"
    }
    bad["receipt_sha256"] = wave_v2.canonical_sha256(receipt_body)
    return bad


def test_recovered_iam_cleanup_is_accepted_but_status_drift_is_rejected(
    tmp_path: Path, plan: dict
) -> None:
    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)
    store = FakeStore()
    terminals, _ = _publish_wave(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        store=store,
    )
    iam = _worker_iam_chain(
        plan,
        ledger,
        resume,
        bootstraps,
        recovered_cleanup=True,
    )
    cleanup = iam["worker_iam_cleanup_receipt"]
    assert cleanup["status"] == (
        "recovered_exact_wave_binding_absence_after_outcome_ambiguity"
    )
    assert cleanup["recovered_after_outcome_ambiguity"] is True
    proof = _validated_lifecycle_proof(
        plan=plan,
        ledger=ledger,
        resume=resume,
        terminals=terminals,
        cleanup=cleanup,
        observed_at="2026-07-22T00:00:01Z",
    )
    receipt = subject.receive_wave_results(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        job_bootstraps=bootstraps,
        terminal_observations=terminals,
        lifecycle_proof_adapter=subject.ValidatedLifecycleProofAdapterV2(
            lambda: proof
        ),
        **iam,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:00:01Z",
        store=store,
        destination=tmp_path / "recovered-cleanup",
        readback_source="local_observed_fixture",
    )
    assert subject.validate_receiver_receipt(plan, receipt) == receipt

    for bad_status, recovered in (
        ("invented_cleanup_status", True),
        ("removed_exact_wave_bindings_post_readback_absent", True),
        (
            "recovered_exact_wave_binding_absence_after_outcome_ambiguity",
            False,
        ),
    ):
        bad_cleanup = deepcopy(cleanup)
        bad_cleanup["status"] = bad_status
        bad_cleanup["recovered_after_outcome_ambiguity"] = recovered
        bad = _coherently_replace_cleanup(receipt, bad_cleanup)
        with pytest.raises(ValueError, match="worker IAM cleanup"):
            subject.validate_receiver_receipt(plan, bad)

    exact_binding_count = 2 * len(resume["selected_attempts"])
    normal_cleanup = deepcopy(cleanup)
    normal_cleanup.update(
        {
            "status": "removed_exact_wave_bindings_post_readback_absent",
            "recovered_after_outcome_ambiguity": False,
            "removed_binding_count": exact_binding_count,
            "set_attempt_count": 1,
            "cloud_mutation_performed": True,
            "source_mutation_outcome": "performed",
        }
    )
    normal_receipt = _coherently_replace_cleanup(receipt, normal_cleanup)
    assert subject.validate_receiver_receipt(plan, normal_receipt) == normal_receipt

    lies = [
        (cleanup, "removed_binding_count", exact_binding_count),
        (cleanup, "set_attempt_count", 1),
        (cleanup, "cloud_mutation_performed", True),
        (cleanup, "source_mutation_outcome", "performed"),
        (normal_cleanup, "removed_binding_count", 0),
        (normal_cleanup, "set_attempt_count", 0),
        (normal_cleanup, "cloud_mutation_performed", False),
        (normal_cleanup, "source_mutation_outcome", "unknown"),
    ]
    for baseline, field, lie in lies:
        bad_cleanup = deepcopy(baseline)
        bad_cleanup[field] = lie
        bad = _coherently_replace_cleanup(receipt, bad_cleanup)
        with pytest.raises(ValueError, match="worker IAM cleanup"):
            subject.validate_receiver_receipt(plan, bad)


@pytest.mark.parametrize(
    "reason",
    [
        "spot_loss_before_done",
        "timeout_before_done",
        "create_missing_before_done",
    ],
)
def test_partial_wave_is_receipted_and_failed_job_returns_to_a01(
    tmp_path: Path, plan: dict, reason: str
) -> None:
    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)
    failed = resume["selected_attempts"][0]["job_id"]
    store = FakeStore()
    terminals, _ = _publish_wave(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        store=store,
        failure=(failed, reason),
    )
    receipt = _receive(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        terminals=terminals,
        store=store,
        destination=tmp_path / reason,
        observed_at="2026-07-22T00:00:01Z",
    )
    assert receipt["status"] == "wave_results_recorded_resume_required"
    failed_pair = [
        resume["selected_attempts"][0]["job_id"],
        resume["selected_attempts"][1]["job_id"],
    ]
    assert receipt["failed_job_ids"] == failed_pair
    assert store.create_count == 6
    assert receipt["nonaccepted_jobs_returned_to_resume"] is True
    pair = plan["waves"][0]["candidate_reference_pairs"][0]
    assert receipt["next_resume_plan"]["selected_attempts"] == [
        {
            **selected,
            "attempt_id": "a01",
            "instance_id": pair[
                f"{selected['source_role']}_attempt_instance_ids"
            ]["a01"],
            "artifact_prefix": selected["artifact_prefix"].replace(
                "/a00", "/a01"
            ),
        }
        for selected in resume["selected_attempts"][:2]
    ]
    result = receipt["attempt_results"][0]
    assert result["terminal_reason"] == reason
    assert result["terminal_status"] == "failed"
    assert result["current_attempt_object_count"] == (
        0
        if reason in {"timeout_before_done", "create_missing_before_done"}
        else 1
    )
    assert result["exact_instance_created"] is (
        reason != "create_missing_before_done"
    )
    peer = receipt["attempt_results"][1]
    assert peer["terminal_status"] == "failed"
    assert peer["valid_done_observed"] is True
    assert result["pair_id"] == peer["pair_id"]
    assert result["pair_atomic_outcome"] == "failed_both_unaccepted_resume"


def test_done_before_all_artifacts_extra_duplicate_and_acceptance_tamper_reject(
    tmp_path: Path, plan: dict
) -> None:
    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)

    def complete_store() -> tuple[FakeStore, list[dict]]:
        store = FakeStore()
        terminals, _ = _publish_wave(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            store=store,
        )
        return store, terminals

    store, terminals = complete_store()
    first = resume["selected_attempts"][0]
    first_hand = _metadata(plan)[first["job_id"]]["work_hand_indices"][0]
    missing = f"{first['artifact_prefix']}/roots/hand_{first_hand:03d}.json"
    store.remove(missing)
    with pytest.raises(ValueError, match="DONE was observed before"):
        _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=tmp_path / "missing",
            observed_at="2026-07-22T00:00:01Z",
        )
    assert store.create_count == 0

    store, terminals = complete_store()
    store.put(f"{first['artifact_prefix']}/unexpected.json", b"{}")
    with pytest.raises(ValueError, match="unknown or extra"):
        _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=tmp_path / "extra",
            observed_at="2026-07-22T00:00:01Z",
        )

    store, terminals = complete_store()
    store.duplicate_path = f"{first['artifact_prefix']}/DONE.json"
    with pytest.raises(ValueError, match="duplicate"):
        _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=tmp_path / "duplicate",
            observed_at="2026-07-22T00:00:01Z",
        )

    store, terminals = complete_store()
    acceptance_path = plan["artifact_contract"][
        "job_acceptance_path_template"
    ].format(job_id=first["job_id"])
    store.put(acceptance_path, b'{"forged":true}\n')
    with pytest.raises(ValueError, match="differs|fields changed|bytes changed"):
        _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=tmp_path / "acceptance-tamper",
            observed_at="2026-07-22T00:00:01Z",
        )

    store, terminals = complete_store()
    store.create_response_generation_delta = 1
    with pytest.raises(ValueError, match="create/readback generation"):
        _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=tmp_path / "acceptance-generation",
            observed_at="2026-07-22T00:00:01Z",
        )


def test_rerun_is_idempotent_local_collision_fails_and_late_a00_done_cannot_win(
    tmp_path: Path, plan: dict
) -> None:
    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)
    store = FakeStore()
    terminals, _ = _publish_wave(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        store=store,
    )
    destination = tmp_path / "idempotent"
    first = _receive(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        terminals=terminals,
        store=store,
        destination=destination,
        observed_at="2026-07-22T00:00:01Z",
    )
    create_count = store.create_count
    second = _receive(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        terminals=terminals,
        store=store,
        destination=destination,
        observed_at="2026-07-22T00:00:01Z",
    )
    assert store.create_count == create_count
    assert second["observed_transition"] == first["observed_transition"]
    assert second["attempt_ledger"] == first["attempt_ledger"]
    assert all(
        row["acceptance_create_performed"] is False
        for row in second["attempt_results"]
    )

    first_path = first["materialized_paths"][0]
    local = destination.joinpath(*first_path.split("/"))
    local.write_bytes(b"tampered")
    with pytest.raises(FileExistsError, match="immutable local result differs"):
        _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=destination,
            observed_at="2026-07-22T00:00:01Z",
        )

    # Build a partial first pair, then let the failed candidate a00 publish
    # DONE after both a01 attempts were selected.  Historical DONE remains
    # evidence only and cannot beat the complete a01 pair.
    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)
    failed = resume["selected_attempts"][0]["job_id"]
    late_store = FakeStore()
    terminals, _ = _publish_wave(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        store=late_store,
        failure=(failed, "spot_loss_before_done"),
    )
    partial = _receive(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        terminals=terminals,
        store=late_store,
        destination=tmp_path / "late",
        observed_at="2026-07-22T00:00:01Z",
    )
    retry_ledger = partial["attempt_ledger"]
    retry_resume = partial["next_resume_plan"]
    retry_bootstraps = _bootstraps(plan, retry_ledger, retry_resume)
    for retry_selected, retry_bootstrap in zip(
        retry_resume["selected_attempts"], retry_bootstraps, strict=True
    ):
        _publish_attempt(
            plan=plan,
            ledger=retry_ledger,
            resume=retry_resume,
            bootstrap=retry_bootstrap,
            selected=retry_selected,
            store=late_store,
            complete=True,
        )
    # Late a00 uses its original bootstrap/identity and now completes.
    original_selected = resume["selected_attempts"][0]
    _publish_attempt(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstrap=bootstraps[0],
        selected=original_selected,
        store=late_store,
        complete=True,
    )
    before = late_store.create_count
    retry_receipt = _receive(
        plan=plan,
        ledger=retry_ledger,
        resume=retry_resume,
        bootstraps=retry_bootstraps,
        terminals=[
            _terminal(selected, "done_observed")
            for selected in retry_resume["selected_attempts"]
        ],
        store=late_store,
        destination=tmp_path / "late",
        observed_at="2026-07-22T00:00:02Z",
    )
    assert late_store.create_count == before + 2
    assert retry_receipt["accepted_job_ids"] == [
        row["job_id"] for row in retry_resume["selected_attempts"]
    ]
    assert all(
        row["attempt_id"] == "a01"
        for row in retry_receipt["observed_transition"]["done_objects"]
        if row["job_id"] in retry_receipt["accepted_job_ids"]
    )
    assert all(
        row["historical_done_object_count"] == 1
        for row in retry_receipt["attempt_results"]
    )


def test_lifecycle_replay_adapter_is_mandatory_one_shot_and_context_bound(
    tmp_path: Path, plan: dict
) -> None:
    ledger, resume = _initial(plan)
    bootstraps = _bootstraps(plan, ledger, resume)
    store = FakeStore()
    terminals, _ = _publish_wave(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        store=store,
    )
    iam = _worker_iam_chain(plan, ledger, resume, bootstraps)
    proof = _validated_lifecycle_proof(
        plan=plan,
        ledger=ledger,
        resume=resume,
        terminals=terminals,
        cleanup=iam["worker_iam_cleanup_receipt"],
        observed_at="2026-07-22T00:00:01Z",
    )
    common = {
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        "job_bootstraps": bootstraps,
        "terminal_observations": terminals,
        **iam,
        "project_id": PROJECT,
        "zone": ZONE,
        "observed_at_utc": "2026-07-22T00:00:01Z",
        "store": store,
        "destination": tmp_path / "adapter",
        "readback_source": "local_observed_fixture",
    }
    with pytest.raises(TypeError, match="lifecycle_proof_adapter"):
        subject.receive_wave_results(**common)
    with pytest.raises(TypeError, match="adapter is required"):
        subject.receive_wave_results(
            **common,
            lifecycle_proof_adapter=proof,
        )
    assert store.create_count == 0

    calls = 0

    def replay() -> dict:
        nonlocal calls
        calls += 1
        return proof

    adapter = subject.ValidatedLifecycleProofAdapterV2(replay)
    receipt = subject.receive_wave_results(
        **common,
        lifecycle_proof_adapter=adapter,
    )
    assert len(receipt["accepted_job_ids"]) == 8
    assert calls == 1
    with pytest.raises(RuntimeError, match="one-shot"):
        adapter.validate(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            receiver_observed_at_utc="2026-07-22T00:00:01Z",
        )
    assert calls == 1

    wrong_context = deepcopy(proof)
    wrong_context["resume_plan_sha256"] = _sha("wrong-resume")
    wrong_context["proof_sha256"] = controller_v2.canonical_sha256(
        {
            key: value
            for key, value in wrong_context.items()
            if key != "proof_sha256"
        }
    )
    with pytest.raises(ValueError, match="context or safety"):
        subject.receive_wave_results(
            **common,
            lifecycle_proof_adapter=subject.ValidatedLifecycleProofAdapterV2(
                lambda: wrong_context
            ),
        )

    future = deepcopy(proof)
    future_absence = future["gce_absence_receipt"]
    future_absence["observed_at_utc"] = "2026-07-22T00:00:02Z"
    future_absence["receipt_sha256"] = gce_v2.canonical_sha256(
        {
            key: value
            for key, value in future_absence.items()
            if key != "receipt_sha256"
        }
    )
    future["proof_sha256"] = controller_v2.canonical_sha256(
        {key: value for key, value in future.items() if key != "proof_sha256"}
    )
    with pytest.raises(ValueError, match="absence binding"):
        subject.receive_wave_results(
            **common,
            lifecycle_proof_adapter=subject.ValidatedLifecycleProofAdapterV2(
                lambda: future
            ),
        )

    future_lifecycle = deepcopy(proof)
    future_lifecycle["lifecycle_attested_at_utc"] = "2026-07-22T00:00:02Z"
    future_lifecycle["proof_sha256"] = controller_v2.canonical_sha256(
        {
            key: value
            for key, value in future_lifecycle.items()
            if key != "proof_sha256"
        }
    )
    with pytest.raises(ValueError, match="context or safety"):
        subject.receive_wave_results(
            **common,
            lifecycle_proof_adapter=subject.ValidatedLifecycleProofAdapterV2(
                lambda: future_lifecycle
            ),
        )

    premature_lifecycle = deepcopy(proof)
    premature_lifecycle["lifecycle_attested_at_utc"] = "2026-07-22T00:00:00Z"
    premature_lifecycle["proof_sha256"] = controller_v2.canonical_sha256(
        {
            key: value
            for key, value in premature_lifecycle.items()
            if key != "proof_sha256"
        }
    )
    with pytest.raises(ValueError, match="absence binding"):
        subject.receive_wave_results(
            **common,
            lifecycle_proof_adapter=subject.ValidatedLifecycleProofAdapterV2(
                lambda: premature_lifecycle
            ),
        )

    wrong_delete_chain = deepcopy(proof)
    wrong_delete_chain["gce_delete_receipt"]["create_receipt_sha256"] = _sha(
        "wrong-create-receipt"
    )
    wrong_delete_chain["gce_delete_receipt"]["receipt_sha256"] = (
        gce_v2.canonical_sha256(
            {
                key: value
                for key, value in wrong_delete_chain[
                    "gce_delete_receipt"
                ].items()
                if key != "receipt_sha256"
            }
        )
    )
    wrong_delete_chain["gce_absence_receipt"]["delete_receipt_sha256"] = (
        wrong_delete_chain["gce_delete_receipt"]["receipt_sha256"]
    )
    wrong_delete_chain["gce_absence_receipt"]["receipt_sha256"] = (
        gce_v2.canonical_sha256(
            {
                key: value
                for key, value in wrong_delete_chain[
                    "gce_absence_receipt"
                ].items()
                if key != "receipt_sha256"
            }
        )
    )
    wrong_delete_chain["proof_sha256"] = controller_v2.canonical_sha256(
        {
            key: value
            for key, value in wrong_delete_chain.items()
            if key != "proof_sha256"
        }
    )
    with pytest.raises(ValueError, match="absence binding"):
        subject.receive_wave_results(
            **common,
            lifecycle_proof_adapter=subject.ValidatedLifecycleProofAdapterV2(
                lambda: wrong_delete_chain
            ),
        )


def test_three_waves_finish_exact_440_inventory_and_local_tree(
    tmp_path: Path, plan: dict
) -> None:
    ledger, resume = _initial(plan)
    store = FakeStore()
    destination = tmp_path / "complete"
    receipt = None
    for second in (1, 2, 3):
        bootstraps = _bootstraps(plan, ledger, resume)
        terminals, _ = _publish_wave(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            store=store,
        )
        receipt = _receive(
            plan=plan,
            ledger=ledger,
            resume=resume,
            bootstraps=bootstraps,
            terminals=terminals,
            store=store,
            destination=destination,
            observed_at=f"2026-07-22T00:00:0{second}Z",
        )
        ledger = receipt["attempt_ledger"]
        resume = receipt["next_resume_plan"]
    assert receipt is not None
    assert receipt["status"] == "all_jobs_complete_exact_inventory_accepted"
    assert resume["all_jobs_complete"] is True
    assert receipt["expected_artifact_inventory"]["object_count"] == 440
    assert receipt["observed_artifact_inventory"]["object_count"] == 440
    assert len(receipt["materialized_paths"]) == 440
    assert len([path for path in destination.rglob("*") if path.is_file()]) == 440
    assert subject.validate_receiver_receipt(plan, receipt) == receipt
