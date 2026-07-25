from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_prelaunch_abort_cleanup_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as receiver_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


RUN = "regular-hu-m31-c02-f100wv2-20260723-prelaunch-abort-test"
NOW = "2026-07-23T03:00:00Z"


def _receipt(kind: str, **values: Any) -> dict[str, Any]:
    core = {"kind": kind, **values, "current_profile_changed": False}
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


def _context(
    run_name: str = RUN,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = wave_v2.build_wave_plan(
        run_name=run_name,
        identity_salt="0123456789abcdef0123456789abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-solver-485418",
        zone="asia-northeast1-b",
        observed_at_utc="2026-07-23T02:59:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return plan, ledger, resume


def _prepared(tmp_path: Path) -> subject.PreparedPrelaunchAbort:
    plan, ledger, resume = _context()
    namespace = receiver_v2.execution_namespace(ledger, resume)
    execution = tmp_path / namespace
    journal_dir = execution / "controller-journal"
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=journal_dir,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
    )
    anchor = controller.record_read_phase(
        phase="launch-bundle-build-r03",
        operation_key=f"{namespace}:launch-bundle-build-r03",
        predecessor_event_sha256=None,
        evidence={},
        output={"launch_authorized": False},
        recorded_at_utc=NOW,
    ).value
    root = tmp_path / "abort" / namespace
    content = {
        "immutable_content_prefix": "hu-m31-t3/full100-wave-v2/content/" + "6" * 64,
        "content_payload_sha256": "6" * 64,
        "outer_manifest_sha256": "7" * 64,
    }
    iam_plan = {
        "plan_sha256": "8" * 64,
        "exact_binding_count": 16,
        "expires_at_utc": "2026-07-23T04:15:00Z",
        "workers": [
            {
                "job_id": row["job_id"],
                "source_role": row["source_role"],
                "service_account": (
                    f"worker-{index:02d}@ofc-solver-485418.iam.gserviceaccount.com"
                ),
            }
            for index, row in enumerate(resume["selected_attempts"])
        ],
    }
    not_before = subject._worker_iam_cleanup_not_before(iam_plan)
    abort_core = {
        "schema": subject.ABORT_PLAN_SCHEMA,
        "status": "prelaunch_abort_plan_ready_cloud_not_mutated",
        "execution_namespace": namespace,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "controller_journal_dir": str(journal_dir.resolve()),
        "anchor_event_sha256": anchor["event_sha256"],
        "controller_event_count_at_plan": 1,
        "selected_instance_ids": [
            row["instance_id"] for row in resume["selected_attempts"]
        ],
        "worker_iam_plan_sha256": iam_plan["plan_sha256"],
        "worker_iam_binding_count": 16,
        "worker_iam_cleanup_not_before_utc": not_before,
        "claim_receipt_sha256": "9" * 64,
        "claim_object_path": "claims/wave.json",
        "stage_receipt_sha256": "a" * 64,
        "content_entry_count": 26,
        "execution_manifest_present": False,
        "gce_create_event_count": 0,
        "claim_delete_authorized": False,
        "service_account_delete_authorized": False,
        "content_delete_authorized": False,
        "cloud_mutation_performed": False,
        "prelaunch_abort_tombstone_required": True,
        "persisted_authorize_request": None,
        "current_profile_changed": False,
    }
    abort_plan = {
        **abort_core,
        "plan_sha256": subject.canonical_sha256(abort_core),
    }
    return subject.PreparedPrelaunchAbort(
        root=root,
        execution_root=execution,
        journal_dir=journal_dir,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        abort_plan=abort_plan,
        content_binding=content,
        stage_plan={},
        preflight_receipt={},
        stage_receipt={"receipt_sha256": "a" * 64, "rows": [{}] * 26},
        claim_receipt={
            "receipt_sha256": "9" * 64,
            "object_path": "claims/wave.json",
            "generation": "123",
            "etag": "claim-etag",
            "readback_sha256": "b" * 64,
            "claim_bytes": 123,
        },
        raw_claim_nonce="nonce",
        iam_plan=iam_plan,
        iam_prepare_receipt={"receipt_sha256": "c" * 64},
        iam_install_receipt={"receipt_sha256": "d" * 64},
        iam_readback_receipt={"receipt_sha256": "e" * 64},
    )


def _patch_receipt_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject.gcp_v2, "validate_read_receipt", lambda **kwargs: dict(kwargs["value"])
    )
    monkeypatch.setattr(
        subject.worker_iam_v2,
        "validate_readback_receipt",
        lambda **kwargs: dict(kwargs["value"]),
    )
    monkeypatch.setattr(
        subject.worker_iam_v2,
        "validate_cleanup_receipt",
        lambda **kwargs: dict(kwargs["value"]),
    )


def _preservation(prepared: subject.PreparedPrelaunchAbort) -> dict[str, Any]:
    service_accounts = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "service_account": row["service_account"],
            "unique_id": str(100000 + index),
            "resource_name": (
                f"projects/ofc-solver-485418/serviceAccounts/{row['service_account']}"
            ),
            "disabled": False,
            "exists": True,
        }
        for index, row in enumerate(prepared.iam_plan["workers"])
    ]
    core = {
        "schema": subject.PRESERVATION_RECEIPT_SCHEMA,
        "status": "claim_and_exact_26_content_objects_preserved",
        "claim_object_path": prepared.claim_receipt["object_path"],
        "claim_generation": prepared.claim_receipt["generation"],
        "claim_etag": prepared.claim_receipt["etag"],
        "claim_sha256": prepared.claim_receipt["readback_sha256"],
        "claim_bytes": prepared.claim_receipt["claim_bytes"],
        "content_prefix": prepared.content_binding["immutable_content_prefix"],
        "content_entry_count": 26,
        "source_stage_receipt_sha256": prepared.stage_receipt["receipt_sha256"],
        "fresh_stage_receipt_sha256": "f" * 64,
        "service_accounts": service_accounts,
        "service_account_count": len(service_accounts),
        "service_accounts_preserved": True,
        "claim_preserved": True,
        "content_preserved": True,
        "claim_delete_performed": False,
        "content_delete_performed": False,
        "cloud_mutation_performed": False,
        "observed_at_utc": NOW,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


def _compute_absence(
    prepared: subject.PreparedPrelaunchAbort, request: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [
        {
            "instance_id": instance_id,
            "instance_absent": True,
            "boot_disk_absent": True,
        }
        for instance_id in request["selected_instance_ids"]
    ]
    core = {
        "schema": subject.COMPUTE_ABSENCE_RECEIPT_SCHEMA,
        "status": "exact_selected_instances_and_boot_disks_absent_get_only",
        "run_name": prepared.wave_plan["run_name"],
        "execution_identity_sha256": prepared.wave_plan["execution_identity_sha256"],
        "wave_plan_sha256": prepared.wave_plan["schedule_sha256"],
        "attempt_ledger_sha256": prepared.attempt_ledger["ledger_sha256"],
        "resume_plan_sha256": prepared.resume_plan["resume_sha256"],
        "project": gcp_v2.PROJECT,
        "zone": gcp_v2.ZONE,
        "selected_instance_ids": list(request["selected_instance_ids"]),
        "rows": rows,
        "selected_instance_count": len(rows),
        "http_get_count": 2 * len(rows),
        "all_selected_instances_absent": True,
        "all_selected_boot_disks_absent": True,
        "read_only": True,
        "cloud_mutation_performed": False,
        "additional_create_authorized": False,
        "observed_at_utc": request["observed_at_utc"],
        "expires_at_utc": request["expires_at_utc"],
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


class FakeDispatcher:
    def __init__(
        self,
        *,
        iam_failure: str | None = None,
        action_hook: Any | None = None,
    ) -> None:
        self.iam_failure = iam_failure
        self.action_hook = action_hook
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.allow_worker_iam_install: list[bool] = []
        self.mutation_calls = 0
        self.put_action_operations: list[str] = []

    def __call__(self, **kwargs: Any) -> Any:
        controller = kwargs["controller"]
        mode = kwargs["mode"]
        request = json.loads(json.dumps(kwargs["request"]))
        self.calls.append((mode, request))
        self.allow_worker_iam_install.append(
            kwargs["allow_worker_iam_install"]
        )
        if mode == "phasea-read":
            receipt = _receipt(
                "phasea",
                all_selected_instances_absent=True,
                all_selected_boot_disks_absent=True,
                all_service_accounts_exist=True,
                selected_vm_count=len(controller.resume_plan["selected_attempts"]),
                service_accounts=[
                    {"job_id": row["job_id"]}
                    for row in controller.resume_plan["selected_attempts"]
                ],
                selected_result_preflight_receipt={
                    "all_selected_artifact_prefixes_absent": True,
                    "all_selected_acceptance_objects_absent": True,
                },
            )
            return controller.record_read_phase(
                phase=mode,
                operation_key=request["operation_key"],
                predecessor_event_sha256=request["predecessor_event_sha256"],
                evidence={},
                output={"gcp_read_receipt": receipt},
                recorded_at_utc=request["observed_at_utc"],
            )
        if mode == "worker-iam-readback":
            receipt = _receipt("iam-readback")
            return controller.record_read_phase(
                phase=mode,
                operation_key=request["operation_key"],
                predecessor_event_sha256=request["predecessor_event_sha256"],
                evidence={},
                output={"worker_iam_readback_receipt": receipt},
                recorded_at_utc=request["observed_at_utc"],
            )
        if mode == "worker-iam-cleanup":
            self.mutation_calls += 1

            if self.iam_failure == "process_loss":
                controller._begin_mutation(
                    phase=mode,
                    operation_key=request["operation_key"],
                    predecessor_event_sha256=request["predecessor_event_sha256"],
                    evidence={},
                    recorded_at_utc=request["observed_at_utc"],
                )
                raise KeyboardInterrupt("simulated process loss after durable intent")

            def action() -> Mapping[str, Any]:
                self.put_action_operations.append(request["operation_key"])
                if self.action_hook is not None:
                    self.action_hook()
                if self.iam_failure == "transport":
                    raise gcp_v2.GcpPhaseATransportError("ambiguous PUT")
                if self.iam_failure == "local_response":
                    raise RuntimeError("provider response normalization mismatch")
                if self.iam_failure == "provider":
                    raise worker_iam_v2.WorkerIamCasError("explicit CAS rejection")
                return _receipt(
                    "iam-cleanup",
                    cleanup_complete=True,
                    remaining_targeted_binding_count=0,
                    post_cleanup_absence_readback=True,
                    removed_binding_count=16,
                    set_attempt_count=1,
                    cloud_mutation_performed=True,
                    recovered_after_outcome_ambiguity=False,
                    source_mutation_outcome="performed",
                    unrelated_policy_fingerprint_sha256="1" * 64,
                )

            return controller.run_mutation_phase(
                phase=mode,
                operation_key=request["operation_key"],
                predecessor_event_sha256=request["predecessor_event_sha256"],
                evidence={},
                action=action,
                validator=lambda value: value,
                started_at_utc=request["observed_at_utc"],
                completed_at_utc=request["observed_at_utc"],
            )
        if mode == "worker-iam-reconcile-cleanup":
            receipt = _receipt(
                "iam-cleanup-reconciled",
                cleanup_complete=True,
                remaining_targeted_binding_count=0,
                post_cleanup_absence_readback=True,
                removed_binding_count=0,
                set_attempt_count=0,
                cloud_mutation_performed=False,
                recovered_after_outcome_ambiguity=True,
                source_mutation_outcome="unknown",
                unrelated_policy_fingerprint_sha256="1" * 64,
            )
            return controller.record_ambiguous_mutation_reconciliation(
                phase=mode,
                operation_key=request["operation_key"],
                source_operation_key=request["source_operation_key"],
                receipt=receipt,
                output_key="receipt",
                recorded_at_utc=request["observed_at_utc"],
            )
        raise AssertionError(f"unexpected mode {mode}")


def _execute(
    prepared: subject.PreparedPrelaunchAbort,
    fake: FakeDispatcher,
    *, clock: datetime | None = None,
    compute_reader: Any | None = None,
    iam_recovery_reader: Any | None = None,
    recovery_only: bool = False,
) -> dict[str, Any]:
    return _runner(
        prepared,
        fake,
        clock=clock,
        compute_reader=compute_reader,
        iam_recovery_reader=iam_recovery_reader,
        recovery_only=recovery_only,
    ).execute()


def _runner(
    prepared: subject.PreparedPrelaunchAbort,
    fake: FakeDispatcher,
    *, clock: datetime | None = None,
    compute_reader: Any | None = None,
    iam_recovery_reader: Any | None = None,
    recovery_only: bool = False,
) -> subject.ProductionPrelaunchAbortCleanupV2:
    return subject.ProductionPrelaunchAbortCleanupV2(
        prepared=prepared,
        allow_cloud_read=True,
        allow_worker_iam_cleanup=not recovery_only,
        confirm_run_name=RUN,
        recovery_only=recovery_only,
        dispatcher=fake,
        compute_absence_reader=compute_reader
        or (lambda request: _compute_absence(prepared, request)),
        iam_intent_recovery_reader=iam_recovery_reader,
        preservation_reader=lambda _request: _preservation(prepared),
        clock=lambda: clock
        or datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
    )


def _crash_after_journal_terminal_before_step_event(
    runner: subject.ProductionPrelaunchAbortCleanupV2,
    *, label: str,
    status: str,
) -> None:
    original = runner._persist_event
    crashed = False

    def persist(**kwargs: Any) -> dict[str, Any]:
        nonlocal crashed
        event = kwargs["event"]
        if (
            not crashed
            and kwargs["label"] == label
            and event.get("status") == status
        ):
            crashed = True
            raise KeyboardInterrupt(
                "simulated crash after source terminal before step event"
            )
        return original(**kwargs)

    runner._persist_event = persist  # type: ignore[method-assign]


def _iam_recovery(
    prepared: subject.PreparedPrelaunchAbort,
    request: Mapping[str, Any],
    *, state: str,
) -> dict[str, Any]:
    cleanup = None
    readback = None
    if state == "exact_desired_after_absent":
        cleanup = _receipt(
            "iam-cleanup-recovered",
            cleanup_complete=True,
            remaining_targeted_binding_count=0,
            post_cleanup_absence_readback=True,
            removed_binding_count=0,
            set_attempt_count=0,
            cloud_mutation_performed=False,
            recovered_after_outcome_ambiguity=True,
            source_mutation_outcome="unknown",
            unrelated_policy_fingerprint_sha256="1" * 64,
        )
    elif state == "exact_before_installed":
        readback = _receipt("iam-readback-recovered-before")
    else:
        raise AssertionError(state)
    core = {
        "schema": subject.IAM_INTENT_RECOVERY_RECEIPT_SCHEMA,
        "status": "pending_cleanup_resolved_by_get_only_state_probe",
        "recovery_operation_key": request["operation_key"],
        "source_operation_key": request["source_operation_key"],
        "source_intent_event_sha256": request["source_intent_event_sha256"],
        "source_readback_receipt_sha256": request["source_readback_receipt"][
            "receipt_sha256"
        ],
        "fresh_compute_absence_event_sha256": request[
            "fresh_compute_absence_event_sha256"
        ],
        "state": state,
        "cleanup_receipt": cleanup,
        "fresh_readback_receipt": readback,
        "http_method": "GET",
        "read_only": True,
        "cloud_mutation_performed": False,
        "observed_at_utc": request["observed_at_utc"],
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


def test_happy_abort_is_idempotent_and_only_mutates_exact_worker_iam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    fake = FakeDispatcher()
    result = _execute(prepared, fake)
    assert [mode for mode, _ in fake.calls] == [
        "worker-iam-readback",
        "worker-iam-cleanup",
    ]
    assert fake.mutation_calls == 1
    assert result["worker_iam_bindings_absent"] is True
    assert result["worker_iam_removed_binding_count"] == 16
    assert result["claim_preserved"] is True
    assert result["service_accounts_preserved"] is True
    assert result["immutable_content_preserved"] is True
    assert result["claim_cleanup_event_sha256"] is None
    assert result["service_account_cleanup_event_sha256"] is None
    assert result["content_cleanup_event_sha256"] is None
    before = len(fake.calls)
    assert _execute(prepared, fake) == result
    assert len(fake.calls) == before


def test_any_create_event_or_execution_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    controller.record_read_phase(
        phase="authorize-launch",
        operation_key=f"{prepared.abort_plan['execution_namespace']}:authorize-launch",
        predecessor_event_sha256=prepared.abort_plan["anchor_event_sha256"],
        evidence={},
        output={"gce_create_receipt": {}},
        recorded_at_utc=NOW,
    )
    with pytest.raises(PermissionError, match="GCE create event"):
        subject.ProductionPrelaunchAbortCleanupV2(
            prepared=prepared,
            allow_cloud_read=True,
            allow_worker_iam_cleanup=True,
            confirm_run_name=RUN,
        )
    prepared.execution_root.joinpath("execution_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="execution manifest"):
        subject._reject_create_authority(prepared.execution_root, [])
    prepared.execution_root.joinpath("execution_manifest.json").unlink()
    create_request = prepared.execution_root / "steps/authorize-launch/request.json"
    create_request.parent.mkdir(parents=True, exist_ok=True)
    create_request.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="GCE create request"):
        subject._reject_create_authority(prepared.execution_root, [])


def test_exact_request_only_authority_requires_explicit_digest_binding(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    stage = controller.record_read_phase(
        phase="stage-content",
        operation_key=f"{prepared.abort_plan['execution_namespace']}:stage-content-test",
        predecessor_event_sha256=prepared.abort_plan["anchor_event_sha256"],
        evidence={},
        output={"stage_receipt": {"stage_complete": True}},
        recorded_at_utc=NOW,
    ).value
    label = "authorize-launch-r00"
    selected = [
        row["instance_id"] for row in prepared.resume_plan["selected_attempts"]
    ]
    bundle_core = {"selected_instance_ids": selected}
    bundle = {
        **bundle_core,
        "bundle_sha256": subject.canonical_sha256(bundle_core),
    }
    request = {
        "operation_key": f"{prepared.abort_plan['execution_namespace']}:{label}",
        "stage_event_sha256": stage["event_sha256"],
        "request_ids": {name: f"request-{index}" for index, name in enumerate(selected)},
        "launch_started_at_utc": NOW,
        "observed_at_utc": NOW,
        "launch_bundle": bundle,
        "launch_validation": {},
        "startup_script_path": "unused-in-structural-test",
    }
    envelope_core = {
        "schema": subject._REQUEST_ENVELOPE_SCHEMA,
        "execution_namespace": prepared.abort_plan["execution_namespace"],
        "label": label,
        "mode": "authorize-launch",
        "request": request,
        "request_sha256": subject.canonical_sha256(request),
        "current_profile_changed": False,
    }
    envelope = {
        **envelope_core,
        "envelope_sha256": subject.canonical_sha256(envelope_core),
    }
    path = prepared.execution_root / "steps" / label / "request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    events = controller.journal.load()
    observed = subject._persisted_request_only_authority(
        prepared.execution_root, events
    )
    assert observed is not None
    binding, persisted_request = observed
    assert persisted_request == request
    with pytest.raises(PermissionError, match="GCE create request"):
        subject._reject_create_authority(prepared.execution_root, events)
    subject._reject_create_authority(
        prepared.execution_root, events, allowed_request_only=binding
    )
    changed = dict(binding)
    changed["request_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="binding changed"):
        subject._reject_create_authority(
            prepared.execution_root, events, allowed_request_only=changed
        )


def test_tombstone_permanently_vetoes_a_stale_launcher(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    abort_controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    stale_launcher = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    tombstone = abort_controller.record_prelaunch_abort_tombstone(
        operation_key=(
            f"{prepared.abort_plan['execution_namespace']}:"
            "prelaunch-abort-tombstone"
        ),
        predecessor_event_sha256=prepared.abort_plan["anchor_event_sha256"],
        abort_plan_sha256=prepared.abort_plan["plan_sha256"],
        recorded_at_utc=NOW,
    )
    with pytest.raises(PermissionError, match="permanently vetoed"):
        stale_launcher._begin_mutation(
            phase="authorize-launch",
            operation_key=(
                f"{prepared.abort_plan['execution_namespace']}:authorize-launch"
            ),
            predecessor_event_sha256=prepared.abort_plan["anchor_event_sha256"],
            evidence={},
            recorded_at_utc=NOW,
        )
    later = abort_controller.record_read_phase(
        phase="post-tombstone-read",
        operation_key=(
            f"{prepared.abort_plan['execution_namespace']}:post-tombstone-read"
        ),
        predecessor_event_sha256=tombstone.value["event_sha256"],
        evidence={},
        output={"read_only": True},
        recorded_at_utc=NOW,
    )
    with pytest.raises(
        controller_v2.JournalTamperError, match="tombstone binding changed"
    ):
        abort_controller.record_prelaunch_abort_tombstone(
            operation_key=(
                f"{prepared.abort_plan['execution_namespace']}:"
                "prelaunch-abort-tombstone"
            ),
            predecessor_event_sha256=later.value["event_sha256"],
            abort_plan_sha256=prepared.abort_plan["plan_sha256"],
            recorded_at_utc=NOW,
        )


def test_abort_and_authorize_launch_have_one_shared_journal_winner(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    abort_controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    launch_controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    barrier = threading.Barrier(2)
    abort_append = abort_controller.journal.append
    launch_append = launch_controller.journal.append

    def append_abort(**kwargs: Any) -> Any:
        if kwargs["phase"] == "prelaunch-abort-tombstone":
            barrier.wait(timeout=5)
        return abort_append(**kwargs)

    def append_launch(**kwargs: Any) -> Any:
        if kwargs["phase"] == "authorize-launch":
            barrier.wait(timeout=5)
        return launch_append(**kwargs)

    abort_controller.journal.append = append_abort  # type: ignore[method-assign]
    launch_controller.journal.append = append_launch  # type: ignore[method-assign]

    def abort() -> Any:
        return abort_controller.record_prelaunch_abort_tombstone(
            operation_key=(
                f"{prepared.abort_plan['execution_namespace']}:"
                "prelaunch-abort-tombstone"
            ),
            predecessor_event_sha256=prepared.abort_plan["anchor_event_sha256"],
            abort_plan_sha256=prepared.abort_plan["plan_sha256"],
            recorded_at_utc=NOW,
        )

    def launch() -> Any:
        return launch_controller._begin_mutation(
            phase="authorize-launch",
            operation_key=(
                f"{prepared.abort_plan['execution_namespace']}:authorize-launch"
            ),
            predecessor_event_sha256=prepared.abort_plan["anchor_event_sha256"],
            evidence={},
            recorded_at_utc=NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(abort), pool.submit(launch)]
        outcomes: list[Any] = []
        failures: list[BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except BaseException as exc:  # exactly one CAS loser is expected
                failures.append(exc)
    assert len(outcomes) == 1, repr(failures)
    assert len(failures) == 1, repr(failures)
    phases = [
        event.value["phase"]
        for event in abort_controller.journal.load()
        if event.value["phase"]
        in {"prelaunch-abort-tombstone", "authorize-launch"}
    ]
    assert phases in [["prelaunch-abort-tombstone"], ["authorize-launch"]]


def test_transport_ambiguity_uses_get_only_reconcile_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    fake = FakeDispatcher(iam_failure="transport")
    result = _execute(prepared, fake)
    assert [mode for mode, _ in fake.calls] == [
        "worker-iam-readback",
        "worker-iam-cleanup",
        "worker-iam-reconcile-cleanup",
    ]
    assert fake.mutation_calls == 1
    assert result["worker_iam_bindings_absent"] is True
    assert result["worker_iam_removed_binding_count"] == 0


def test_provider_failure_never_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    fake = FakeDispatcher(iam_failure="provider")
    with pytest.raises(RuntimeError, match="outcome-ambiguity"):
        _execute(prepared, fake)
    assert [mode for mode, _ in fake.calls] == [
        "worker-iam-readback", "worker-iam-cleanup"
    ]
    assert fake.mutation_calls == 1


def test_historical_local_response_failure_resumes_by_get_only_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    failed = FakeDispatcher(iam_failure="local_response")

    def interrupt_before_reconcile(**kwargs: Any) -> Any:
        if kwargs["mode"] == "worker-iam-reconcile-cleanup":
            raise KeyboardInterrupt("simulate old binary stopping at local failure")
        return failed(**kwargs)

    with pytest.raises(KeyboardInterrupt, match="old binary"):
        subject.execute_prelaunch_abort(
            prepared=prepared,
            allow_cloud_read=True,
            allow_worker_iam_cleanup=True,
            confirm_run_name=RUN,
            dispatcher=interrupt_before_reconcile,
            compute_absence_reader=lambda request: _compute_absence(
                prepared, request
            ),
            preservation_reader=lambda _request: _preservation(prepared),
            clock=lambda: datetime(
                2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc
            ),
        )
    source_event = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/event.json"
        ).read_text(encoding="utf-8")
    )
    assert source_event["status"] == "failed"
    assert source_event["output"]["failure_kind"] == "local_or_response_failure"

    resumed = FakeDispatcher()
    # The source cleanup's compute proof expired at 03:01:01.  A real restart
    # must obtain a new GET-only compute proof while retaining the immutable
    # source cleanup operation and must never issue another cleanup PUT.
    result = _execute(
        prepared,
        resumed,
        clock=datetime(2026, 7, 23, 3, 2, 1, tzinfo=timezone.utc),
        recovery_only=True,
    )
    assert [mode for mode, _ in resumed.calls] == [
        "worker-iam-reconcile-cleanup"
    ]
    assert resumed.mutation_calls == 0
    assert resumed.allow_worker_iam_install == [False]
    assert (
        prepared.root
        / "steps/prelaunch-abort-compute-before-iam-cleanup-00-r01/event.json"
    ).is_file()
    assert result["worker_iam_bindings_absent"] is True
    assert result["worker_iam_removed_binding_count"] == 0


def test_cleanup_boundary_uses_launch_bundle_minimum_remaining_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    assert bundle_v2.IAM_MIN_REMAINING_SECONDS == 4_500
    assert prepared.abort_plan["worker_iam_cleanup_not_before_utc"] == NOW
    just_before = datetime(2026, 7, 23, 2, 59, 59, tzinfo=timezone.utc)
    fake = FakeDispatcher()
    with pytest.raises(PermissionError, match="minimum window"):
        _execute(prepared, fake, clock=just_before)
    assert fake.calls == []
    exact = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
    with pytest.raises(PermissionError, match="minimum window"):
        _execute(prepared, fake, clock=exact)
    after = datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc)
    result = _execute(prepared, fake, clock=after)
    assert result["worker_iam_bindings_absent"] is True


def test_recovery_only_refuses_to_create_a_cleanup_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    fake = FakeDispatcher()
    with pytest.raises(PermissionError, match="intent/failure"):
        _execute(prepared, fake, recovery_only=True)
    assert fake.calls == []
    assert fake.mutation_calls == 0
    assert not (
        prepared.root / "steps/prelaunch-abort-tombstone/event.json"
    ).exists()


def test_cli_has_no_claim_content_service_account_or_gce_delete_capability() -> None:
    options = {action.dest for action in subject._parser()._actions}
    assert {
        "allow_cloud_read", "allow_worker_iam_cleanup", "recovery_only"
    } <= options
    assert {
        "allow_gce_delete", "allow_claim_delete", "allow_content_delete",
        "allow_service_account_delete", "allow_gce_create",
    }.isdisjoint(options)


def test_intent_only_exact_after_completes_by_get_without_put_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    crashing = FakeDispatcher(iam_failure="process_loss")
    with pytest.raises(KeyboardInterrupt, match="process loss"):
        _execute(prepared, crashing)
    assert crashing.mutation_calls == 1
    resumed = FakeDispatcher()
    result = _execute(
        prepared,
        resumed,
        iam_recovery_reader=lambda request: _iam_recovery(
            prepared, request, state="exact_desired_after_absent"
        ),
    )
    assert resumed.mutation_calls == 0
    assert result["worker_iam_bindings_absent"] is True
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    assert controller.inspect()["pending_mutation_operations"] == []


def test_intent_only_exact_before_closes_old_intent_and_uses_new_cas_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    crashing = FakeDispatcher(iam_failure="process_loss")
    with pytest.raises(KeyboardInterrupt):
        _execute(prepared, crashing)
    resumed = FakeDispatcher()
    result = _execute(
        prepared,
        resumed,
        iam_recovery_reader=lambda request: _iam_recovery(
            prepared, request, state="exact_before_installed"
        ),
    )
    assert resumed.mutation_calls == 1
    cleanup_requests = [
        request
        for mode, request in resumed.calls
        if mode == "worker-iam-cleanup"
    ]
    assert len(cleanup_requests) == 1
    assert cleanup_requests[0]["operation_key"].endswith(
        "prelaunch-abort-worker-iam-cleanup-r01"
    )
    assert result["worker_iam_bindings_absent"] is True
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    assert controller.inspect()["pending_mutation_operations"] == []


@pytest.mark.parametrize(
    ("resume_clock", "same_compute_authority"),
    [
        (datetime(2026, 7, 23, 3, 0, 3, tzinfo=timezone.utc), True),
        (datetime(2026, 7, 23, 3, 2, 3, tzinfo=timezone.utc), False),
    ],
)
def test_crash_after_recovery_get_uses_a_new_get_round_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_clock: datetime,
    same_compute_authority: bool,
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    compute_requests: list[dict[str, Any]] = []

    def compute_reader(request: Mapping[str, Any]) -> Mapping[str, Any]:
        compute_requests.append(json.loads(json.dumps(request)))
        return _compute_absence(prepared, request)

    crashing = FakeDispatcher(iam_failure="process_loss")
    with pytest.raises(KeyboardInterrupt, match="process loss"):
        _execute(
            prepared,
            crashing,
            clock=datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
            compute_reader=compute_reader,
        )

    recovery_requests: list[dict[str, Any]] = []

    def recovery_reader(request: Mapping[str, Any]) -> Mapping[str, Any]:
        recovery_requests.append(json.loads(json.dumps(request)))
        return _iam_recovery(
            prepared, request, state="exact_desired_after_absent"
        )

    interrupted = _runner(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 0, 2, tzinfo=timezone.utc),
        compute_reader=compute_reader,
        iam_recovery_reader=recovery_reader,
    )

    def crash_before_source_terminal(**_kwargs: Any) -> Any:
        raise KeyboardInterrupt("simulated crash after recovery GET event")

    interrupted.controller.reconcile_pending_mutation = (  # type: ignore[method-assign]
        crash_before_source_terminal
    )
    with pytest.raises(KeyboardInterrupt, match="after recovery GET"):
        interrupted.execute()
    assert not prepared.root.joinpath("abort_manifest.json").exists()
    first_recovery_event = (
        prepared.root
        / "steps/prelaunch-abort-worker-iam-cleanup-intent-recovery-r00/event.json"
    )
    assert first_recovery_event.is_file()

    resumed = _runner(
        prepared,
        FakeDispatcher(),
        clock=resume_clock,
        compute_reader=compute_reader,
        iam_recovery_reader=recovery_reader,
    )
    result = resumed.execute()
    assert len(recovery_requests) == 2
    assert recovery_requests[0]["operation_key"].endswith(
        "intent-recovery-r00"
    )
    assert recovery_requests[1]["operation_key"].endswith(
        "intent-recovery-r01"
    )
    assert (
        recovery_requests[0]["fresh_compute_absence_event_sha256"]
        == recovery_requests[1]["fresh_compute_absence_event_sha256"]
    ) is same_compute_authority
    second_recovery_event = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup-intent-recovery-r01/event.json"
        ).read_text(encoding="utf-8")
    )
    assert result["worker_iam_intent_recovery_event_sha256"] == (
        second_recovery_event["event_sha256"]
    )
    assert resumed.controller.inspect()["pending_mutation_operations"] == []


def test_direct_cleanup_terminal_replay_restores_its_original_compute_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    runner = _runner(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
    )
    _crash_after_journal_terminal_before_step_event(
        runner,
        label="prelaunch-abort-worker-iam-cleanup",
        status="complete",
    )
    with pytest.raises(KeyboardInterrupt, match="after source terminal"):
        runner.execute()
    cleanup_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    assert not prepared.root.joinpath(
        "steps/prelaunch-abort-worker-iam-cleanup/event.json"
    ).exists()
    result = _execute(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 2, 3, tzinfo=timezone.utc),
    )
    assert result["preflight_event_sha256"] == cleanup_request[
        "predecessor_event_sha256"
    ]


def test_transport_reconcile_terminal_replay_restores_source_compute_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    runner = _runner(
        prepared,
        FakeDispatcher(iam_failure="transport"),
        clock=datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
    )
    _crash_after_journal_terminal_before_step_event(
        runner,
        label="prelaunch-abort-worker-iam-reconcile-cleanup",
        status="complete",
    )
    with pytest.raises(KeyboardInterrupt, match="after source terminal"):
        runner.execute()
    cleanup_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    assert not prepared.root.joinpath(
        "steps/prelaunch-abort-worker-iam-reconcile-cleanup/event.json"
    ).exists()
    result = _execute(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 2, 3, tzinfo=timezone.utc),
    )
    assert result["preflight_event_sha256"] == cleanup_request[
        "predecessor_event_sha256"
    ]


def test_recovered_cleanup_terminal_replay_restores_recovery_compute_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    with pytest.raises(KeyboardInterrupt, match="process loss"):
        _execute(
            prepared,
            FakeDispatcher(iam_failure="process_loss"),
            clock=datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
        )
    runner = _runner(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 0, 2, tzinfo=timezone.utc),
        iam_recovery_reader=lambda request: _iam_recovery(
            prepared, request, state="exact_desired_after_absent"
        ),
    )
    _crash_after_journal_terminal_before_step_event(
        runner,
        label="prelaunch-abort-worker-iam-cleanup",
        status="reconciled-complete",
    )
    with pytest.raises(KeyboardInterrupt, match="after source terminal"):
        runner.execute()
    recovery_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup-intent-recovery-r00/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    result = _execute(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 2, 3, tzinfo=timezone.utc),
        iam_recovery_reader=lambda request: _iam_recovery(
            prepared, request, state="exact_desired_after_absent"
        ),
    )
    assert result["preflight_event_sha256"] == recovery_request[
        "fresh_compute_absence_event_sha256"
    ]


def test_request_only_stale_cleanup_is_abandoned_before_new_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    underlying = FakeDispatcher()

    def crash_before_intent(**kwargs: Any) -> Any:
        if kwargs["mode"] == "worker-iam-cleanup":
            raise KeyboardInterrupt("simulated crash after request before intent")
        return underlying(**kwargs)

    runner = _runner(
        prepared,
        crash_before_intent,  # type: ignore[arg-type]
        clock=datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(KeyboardInterrupt, match="before intent"):
        runner.execute()
    stale_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    assert runner.controller.inspect()["pending_mutation_operations"] == []

    result = _execute(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 2, 3, tzinfo=timezone.utc),
    )
    abandoned = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/event.json"
        ).read_text(encoding="utf-8")
    )
    new_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup-r01/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    assert abandoned["status"] == "failed"
    assert abandoned["mode"] == "read-only"
    assert abandoned["mutation_requested"] is False
    assert abandoned["output"]["failure_kind"] == (
        "stale_request_abandoned_before_intent"
    )
    assert new_request["predecessor_event_sha256"] != stale_request[
        "predecessor_event_sha256"
    ]
    assert result["preflight_event_sha256"] == new_request[
        "predecessor_event_sha256"
    ]
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=prepared.journal_dir,
        wave_plan=prepared.wave_plan,
        attempt_ledger=prepared.attempt_ledger,
        resume_plan=prepared.resume_plan,
        create_journal=False,
    )
    with pytest.raises(
        controller_v2.PendingMutationError,
        match="consumed before abandonment",
    ):
        controller.abandon_worker_iam_cleanup_request_before_intent(
            operation_key=stale_request["operation_key"],
            stale_predecessor_event_sha256=stale_request[
                "predecessor_event_sha256"
            ],
            fresh_compute_absence_event_sha256=new_request[
                "predecessor_event_sha256"
            ],
            recorded_at_utc=NOW,
        )


def test_stale_abandonment_winning_dispatch_race_never_runs_old_put(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    fake = FakeDispatcher()
    runner = _runner(prepared, fake)
    original_dispatch = runner._dispatch
    raced = False

    def dispatch(**kwargs: Any) -> dict[str, Any]:
        nonlocal raced
        if kwargs["mode"] == "worker-iam-cleanup" and not raced:
            raced = True
            request = json.loads(
                (
                    prepared.root
                    / "steps/prelaunch-abort-worker-iam-cleanup/request.json"
                ).read_text(encoding="utf-8")
            )["request"]
            fresh = runner._fresh_compute_absence(
                base_label=(
                    "prelaunch-abort-compute-before-iam-cleanup-99"
                ),
                predecessor=request["predecessor_event_sha256"],
            )
            runner.controller.abandon_worker_iam_cleanup_request_before_intent(
                operation_key=request["operation_key"],
                stale_predecessor_event_sha256=request[
                    "predecessor_event_sha256"
                ],
                fresh_compute_absence_event_sha256=fresh["event_sha256"],
                recorded_at_utc=NOW,
            )
        return original_dispatch(**kwargs)

    runner._dispatch = dispatch  # type: ignore[method-assign]
    result = runner.execute()
    base_operation = (
        f"{prepared.abort_plan['execution_namespace']}:"
        "prelaunch-abort-worker-iam-cleanup"
    )
    assert raced is True
    assert base_operation not in fake.put_action_operations
    assert len(fake.put_action_operations) == 1
    assert fake.put_action_operations[0].endswith(
        "prelaunch-abort-worker-iam-cleanup-r01"
    )
    assert result["worker_iam_bindings_absent"] is True


def test_cleanup_intent_winning_dispatch_race_rejects_stale_abandonment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    fake = FakeDispatcher()
    runner = _runner(prepared, fake)
    original_dispatch = runner._dispatch
    loser: list[BaseException] = []
    raced = False

    def dispatch(**kwargs: Any) -> dict[str, Any]:
        nonlocal raced
        if kwargs["mode"] == "worker-iam-cleanup" and not raced:
            raced = True
            request = json.loads(
                (
                    prepared.root
                    / "steps/prelaunch-abort-worker-iam-cleanup/request.json"
                ).read_text(encoding="utf-8")
            )["request"]
            fresh = runner._fresh_compute_absence(
                base_label=(
                    "prelaunch-abort-compute-before-iam-cleanup-99"
                ),
                predecessor=request["predecessor_event_sha256"],
            )

            def lose_abandonment_race() -> None:
                try:
                    runner.controller.abandon_worker_iam_cleanup_request_before_intent(
                        operation_key=request["operation_key"],
                        stale_predecessor_event_sha256=request[
                            "predecessor_event_sha256"
                        ],
                        fresh_compute_absence_event_sha256=fresh[
                            "event_sha256"
                        ],
                        recorded_at_utc=NOW,
                    )
                except BaseException as exc:
                    loser.append(exc)

            fake.action_hook = lose_abandonment_race
        return original_dispatch(**kwargs)

    runner._dispatch = dispatch  # type: ignore[method-assign]
    result = runner.execute()
    base_operation = (
        f"{prepared.abort_plan['execution_namespace']}:"
        "prelaunch-abort-worker-iam-cleanup"
    )
    assert raced is True
    assert fake.put_action_operations == [base_operation]
    assert len(loser) == 1
    assert isinstance(loser[0], controller_v2.PendingMutationError)
    assert result["worker_iam_bindings_absent"] is True


def test_request_expiring_immediately_before_dispatch_is_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    now = {
        "value": datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc)
    }
    fake = FakeDispatcher()
    runner = subject.ProductionPrelaunchAbortCleanupV2(
        prepared=prepared,
        allow_cloud_read=True,
        allow_worker_iam_cleanup=True,
        confirm_run_name=RUN,
        dispatcher=fake,
        compute_absence_reader=lambda request: _compute_absence(
            prepared, request
        ),
        preservation_reader=lambda _request: _preservation(prepared),
        clock=lambda: now["value"],
    )
    original_validate = runner._validated_cleanup_compute_event
    advanced = False

    def validate_then_expire(event_sha256: str) -> dict[str, Any]:
        nonlocal advanced
        event = original_validate(event_sha256)
        if not advanced:
            advanced = True
            now["value"] = datetime(
                2026, 7, 23, 3, 2, 1, tzinfo=timezone.utc
            )
        return event

    runner._validated_cleanup_compute_event = (  # type: ignore[method-assign]
        validate_then_expire
    )
    result = runner.execute()
    base_operation = (
        f"{prepared.abort_plan['execution_namespace']}:"
        "prelaunch-abort-worker-iam-cleanup"
    )
    base_event = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/event.json"
        ).read_text(encoding="utf-8")
    )
    assert advanced is True
    assert base_event["output"]["failure_kind"] == (
        "stale_request_abandoned_before_intent"
    )
    assert base_operation not in fake.put_action_operations
    assert len(fake.put_action_operations) == 1
    assert fake.put_action_operations[0].endswith(
        "prelaunch-abort-worker-iam-cleanup-r01"
    )
    assert result["worker_iam_bindings_absent"] is True


def test_request_expiring_during_final_source_revalidation_is_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    now = {
        "value": datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc)
    }
    fake = FakeDispatcher()
    runner = subject.ProductionPrelaunchAbortCleanupV2(
        prepared=prepared,
        allow_cloud_read=True,
        allow_worker_iam_cleanup=True,
        confirm_run_name=RUN,
        dispatcher=fake,
        compute_absence_reader=lambda request: _compute_absence(
            prepared, request
        ),
        preservation_reader=lambda _request: _preservation(prepared),
        clock=lambda: now["value"],
    )
    original_revalidate = runner._revalidate_execution_source
    revalidations = 0

    def revalidate_then_expire() -> None:
        nonlocal revalidations
        revalidations += 1
        original_revalidate()
        if revalidations == 3:
            now["value"] = datetime(
                2026, 7, 23, 3, 2, 1, tzinfo=timezone.utc
            )

    runner._revalidate_execution_source = (  # type: ignore[method-assign]
        revalidate_then_expire
    )
    result = runner.execute()
    base_operation = (
        f"{prepared.abort_plan['execution_namespace']}:"
        "prelaunch-abort-worker-iam-cleanup"
    )
    base_event = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup/event.json"
        ).read_text(encoding="utf-8")
    )
    assert revalidations >= 3
    assert base_event["output"]["failure_kind"] == (
        "stale_request_abandoned_before_intent"
    )
    assert base_operation not in fake.put_action_operations
    assert len(fake.put_action_operations) == 1
    assert fake.put_action_operations[0].endswith(
        "prelaunch-abort-worker-iam-cleanup-r01"
    )
    assert result["worker_iam_bindings_absent"] is True


def test_completed_preservation_replay_restores_its_original_postflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    runner = _runner(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc),
    )

    def crash_before_manifest(_value: Mapping[str, Any]) -> dict[str, Any]:
        raise KeyboardInterrupt("simulated crash after preservation completion")

    runner._validate_manifest = crash_before_manifest  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt, match="after preservation"):
        runner.execute()
    assert not prepared.root.joinpath("abort_manifest.json").exists()
    preservation_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-preservation-readback-r00/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    preservation_event = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-preservation-readback-r00/event.json"
        ).read_text(encoding="utf-8")
    )

    result = _execute(
        prepared,
        FakeDispatcher(),
        clock=datetime(2026, 7, 23, 3, 2, 3, tzinfo=timezone.utc),
    )
    assert result["postflight_event_sha256"] == preservation_request[
        "predecessor_event_sha256"
    ]
    assert result["preservation_event_sha256"] == preservation_event[
        "event_sha256"
    ]


def test_stale_compute_absence_event_is_never_reused_as_cleanup_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    _patch_receipt_validators(monkeypatch)
    compute_requests: list[dict[str, Any]] = []

    def compute_reader(request: Mapping[str, Any]) -> Mapping[str, Any]:
        compute_requests.append(json.loads(json.dumps(request)))
        return _compute_absence(prepared, request)

    crashing = FakeDispatcher(iam_failure="process_loss")
    first_clock = datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc)
    with pytest.raises(KeyboardInterrupt):
        _execute(
            prepared, crashing, clock=first_clock, compute_reader=compute_reader
        )
    stale_event_sha = json.loads(
        next(
            prepared.root.glob(
                "steps/prelaunch-abort-compute-before-iam-cleanup-00-r00/event.json"
            )
        ).read_text(encoding="utf-8")
    )["event_sha256"]
    resumed = FakeDispatcher()
    late_clock = datetime(2026, 7, 23, 3, 2, 2, tzinfo=timezone.utc)
    _execute(
        prepared,
        resumed,
        clock=late_clock,
        compute_reader=compute_reader,
        iam_recovery_reader=lambda request: _iam_recovery(
            prepared, request, state="exact_before_installed"
        ),
    )
    recovery_request = json.loads(
        (
            prepared.root
            / "steps/prelaunch-abort-worker-iam-cleanup-intent-recovery-r00/request.json"
        ).read_text(encoding="utf-8")
    )["request"]
    assert recovery_request["fresh_compute_absence_event_sha256"] != stale_event_sha
    assert len(compute_requests) >= 3


def test_default_compute_absence_reader_is_exact_selected_get_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    calls: list[tuple[str, str]] = []

    def requester(
        method: str, url: str, _headers: Mapping[str, str],
        _body: bytes | None, _timeout: int,
    ) -> Any:
        calls.append((method, url))
        return gcp_v2.HttpResponse(status=404, body=b"{}", headers={})

    monkeypatch.setenv(gcp_v2.TOKEN_ENV, "x" * 32)
    runner = subject.ProductionPrelaunchAbortCleanupV2(
        prepared=prepared,
        allow_cloud_read=True,
        allow_worker_iam_cleanup=True,
        confirm_run_name=RUN,
        requester=requester,
    )
    now = datetime(2026, 7, 23, 3, 0, 1, tzinfo=timezone.utc)
    request = {
        "selected_instance_ids": [
            row["instance_id"] for row in prepared.resume_plan["selected_attempts"]
        ],
        "observed_at_utc": subject._render_utc(now),
        "expires_at_utc": subject._render_utc(
            now + subject.timedelta(seconds=subject.COMPUTE_ABSENCE_FRESHNESS_SECONDS)
        ),
    }
    receipt = runner._default_compute_absence_reader(request)
    assert receipt["http_get_count"] == 2 * len(request["selected_instance_ids"])
    assert {method for method, _ in calls} == {"GET"}
    assert len(calls) == receipt["http_get_count"]
    assert all("/instances/" in url or "/disks/" in url for _, url in calls)
    assert all("quota" not in url.lower() and "roles" not in url.lower() for _, url in calls)


def test_load_prepared_rejects_execution_source_input_drift(tmp_path: Path) -> None:
    plan, ledger, resume = _context()
    root = tmp_path / "abort" / "execution-copy"
    source = tmp_path / "execution-source"
    for base, values in (
        (root, (plan, ledger, resume)),
        (source, _context(RUN + "-drift")),
    ):
        for name, value in zip(
            ("wave_plan", "attempt_ledger", "resume_plan"), values, strict=True
        ):
            path = base / "inputs" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="copied inputs"):
        subject._load_prepared(execution_root=source, root=root)
