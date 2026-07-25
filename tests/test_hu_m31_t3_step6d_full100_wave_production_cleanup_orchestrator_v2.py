from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_cleanup_orchestrator_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as receiver_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


RUN = "regular-hu-m31-c02-f100wv2-20260723-cleanup-test"
NOW = "2026-07-23T03:00:00Z"


def _context() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = wave_v2.build_wave_plan(
        run_name=RUN,
        identity_salt="fedcba9876543210fedcba9876543210",
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


def _receipt(label: str, **values: Any) -> dict[str, Any]:
    core = {"kind": label, **values, "current_profile_changed": False}
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


def _worker_iam_receipt(label: str, **values: Any) -> dict[str, Any]:
    core = {
        "schema": worker_iam_v2.CLEANUP_RECEIPT_SCHEMA,
        "kind": label,
        **values,
        "current_profile_changed": False,
    }
    digest = worker_iam_v2.canonical_sha256(core)
    return {**core, "receipt_sha256": digest}


def test_worker_iam_receipt_uses_producer_newline_canonical_seal() -> None:
    receipt = _worker_iam_receipt("iam-cleanup", cleanup_complete=True)
    assert subject._sealed_worker_iam_receipt(
        receipt, "worker IAM cleanup receipt"
    ) == receipt

    no_newline = dict(receipt)
    body = {key: value for key, value in no_newline.items() if key != "receipt_sha256"}
    no_newline["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="digest changed"):
        subject._sealed_worker_iam_receipt(
            no_newline, "worker IAM cleanup receipt"
        )


def _manifest(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    journal_dir: Path,
    launch_event_sha256: str,
    create_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [row["instance_id"] for row in resume["selected_attempts"]]

    def ids() -> dict[str, str]:
        return {name: str(uuid.uuid4()) for name in selected}

    core = {
        "schema": subject.launch_v2.EXECUTION_MANIFEST_SCHEMA,
        "status": subject.launch_v2.EXECUTION_MANIFEST_STATUS,
        "execution_namespace": receiver_v2.execution_namespace(ledger, resume),
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "orchestration_plan_sha256": "4" * 64,
        "controller_journal_dir": str(journal_dir.resolve()),
        "startup_script_path": str(Path("scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh").resolve()),
        "launch_event_sha256": launch_event_sha256,
        "stage_event_sha256": "5" * 64,
        "launch_bundle": {"selected_instance_ids": selected},
        "launch_validation": {},
        "gce_create_receipt": dict(create_receipt),
        "actual_launch_receipt": None,
        "content_binding": {
            "immutable_content_prefix": "hu-m31-t3/full100-wave-v2/content/" + "6" * 64,
            "content_payload_sha256": "6" * 64,
            "outer_manifest_sha256": "7" * 64,
        },
        "content_preflight_receipt": _receipt("content-preflight"),
        "stage_receipt": _receipt("stage", stage_complete=True),
        "source_content_reused": False,
        "gce_create_request_ids": ids(),
        "gce_delete_request_ids": ids(),
        "orphan_disk_delete_request_ids": ids(),
        "worker_iam_plan": _receipt("worker-iam-plan"),
        "worker_iam_prepare_receipt": _receipt("worker-iam-prepare"),
        "worker_iam_install_receipt": _receipt("worker-iam-install"),
        "worker_iam_readback_receipt": _receipt("worker-iam-readback"),
        "receiver_request_ready_after_lifecycle_closeout": True,
        "cleanup_request_ids_persisted": True,
        "credentials_from_environment_only": True,
        "additional_create_authorized": False,
        "current_profile_changed": False,
    }
    return {**core, "manifest_sha256": subject.canonical_sha256(core)}


def _launch_journal(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    controller_v2.Full100WaveControllerV2,
    dict[str, Any],
    dict[str, Any],
]:
    plan, ledger, resume = _context()
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=tmp_path / "journal",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
    )
    create = _receipt("gce-create", create_complete=False, rows=[])
    launch = controller.journal.append(
        phase="reconcile-create",
        mode="read-only",
        status="partial",
        operation_key=f"{receiver_v2.execution_namespace(ledger, resume)}:reconcile-create",
        predecessor_event_sha256=None,
        evidence={},
        output={"gce_create_receipt": create, "actual_launch_receipt": None},
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc=NOW,
    )
    manifest = _manifest(
        plan=plan,
        ledger=ledger,
        resume=resume,
        journal_dir=controller.journal.directory,
        launch_event_sha256=launch.value["event_sha256"],
        create_receipt=create,
    )
    return plan, ledger, resume, controller, manifest, launch.value


def _patch_terminal(
    monkeypatch: pytest.MonkeyPatch,
    launch: Mapping[str, Any],
    create: Mapping[str, Any],
) -> None:
    monkeypatch.setattr(
        subject,
        "_validate_terminal_launch",
        lambda **kwargs: (dict(launch), dict(create)),
    )


def _prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> subject.PreparedCleanup:
    plan, ledger, resume, _controller, manifest, launch = _launch_journal(tmp_path)
    _patch_terminal(monkeypatch, launch, manifest["gce_create_receipt"])
    return subject.prepare_cleanup(
        execution_manifest=manifest,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        cleanup_root=tmp_path / "cleanup",
    )


class FakeCleanupDispatcher:
    def __init__(
        self,
        *,
        gce_ambiguity: bool = False,
        iam_ambiguity: bool = False,
        crash_after_first_delete: bool = False,
    ) -> None:
        self.gce_ambiguity = gce_ambiguity
        self.iam_ambiguity = iam_ambiguity
        self.crash_after_first_delete = crash_after_first_delete
        self.crashed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mutation_appends: list[str] = []

    @staticmethod
    def _existing(controller: Any, operation: str) -> Any | None:
        rows = [
            event
            for event in controller.journal.load()
            if event.value["operation_key"] == operation
            and event.value["status"] in {"complete", "partial", "failed"}
        ]
        return None if not rows else rows[0]

    def __call__(self, **kwargs: Any) -> Any:
        controller = kwargs["controller"]
        mode = kwargs["mode"]
        request = json.loads(json.dumps(kwargs["request"]))
        operation = request["operation_key"]
        existing = self._existing(controller, operation)
        if existing is not None:
            return existing
        self.calls.append((mode, request))
        label = operation.rsplit(":", 1)[-1]
        phase: str
        status = "complete"
        output: dict[str, Any]
        mutation_requested = False
        mutation_outcome = "not_requested"
        predecessor: str | None
        if mode == "cleanup" and request["step"] == "delete-instances":
            phase = "cleanup-delete"
            predecessor = request["reconcile_event_sha256"] or request[
                "launch_event_sha256"
            ]
            mutation_requested = True
            self.mutation_appends.append(label)
            if self.gce_ambiguity and label.endswith("r00"):
                status = "failed"
                mutation_outcome = "unknown"
                output = {"failure_kind": "transport_ambiguity"}
            else:
                mutation_outcome = "performed"
                output = {
                    "gce_delete_receipt": _receipt(
                        "gce-delete", delete_operation_count=1
                    )
                }
        elif mode == "reconcile-delete":
            phase = "reconcile-delete"
            source = self._existing(controller, request["source_operation_key"])
            assert source is not None
            predecessor = source.value["event_sha256"]
            if self.gce_ambiguity:
                status = "partial"
                output = {
                    "gce_delete_reconciliation_receipt": _receipt(
                        "gce-delete-reconcile", orphan_cleanup_required=True
                    )
                }
            else:  # pragma: no cover - reconcile is only requested when enabled.
                output = {
                    "gce_delete_receipt": _receipt("gce-delete-recovered")
                }
        elif mode == "cleanup" and request["step"] == "verify-instance-absence":
            phase = "cleanup-absence"
            predecessor = request["delete_event_sha256"]
            output = {
                "gce_absence_receipt": _receipt(
                    "gce-absence",
                    all_instances_absent=True,
                    all_boot_disks_absent=True,
                )
            }
        elif mode == "worker-iam-cleanup":
            phase = mode
            predecessor = request["predecessor_event_sha256"]
            mutation_requested = True
            self.mutation_appends.append(label)
            if self.iam_ambiguity:
                status = "failed"
                mutation_outcome = "unknown"
                output = {"failure_kind": "transport_ambiguity"}
            else:
                mutation_outcome = "performed"
                output = {
                    "receipt": _worker_iam_receipt(
                        "iam-cleanup", cleanup_complete=True
                    )
                }
        elif mode == "worker-iam-reconcile-cleanup":
            phase = mode
            source = self._existing(controller, request["source_operation_key"])
            assert source is not None
            predecessor = source.value["event_sha256"]
            output = {
                "receipt": _worker_iam_receipt(
                    "iam-cleanup-reconciled", cleanup_complete=True
                )
            }
        elif mode == "cleanup" and request["step"] == "closeout":
            phase = "closeout"
            predecessor = request["absence_event_sha256"]
            output = _receipt(
                "lifecycle",
                all_owned_instances_absent=True,
                all_owned_boot_disks_absent=True,
                worker_iam_bindings_absent=True,
                content_cleanup_event_sha256=None,
                attested_at_utc=request["recorded_at_utc"],
            )
        else:  # pragma: no cover
            raise AssertionError(f"unexpected cleanup mode {mode} {request}")
        event = controller.journal.append(
            phase=phase,
            mode="mutation" if mutation_requested else "read-only",
            status=status,
            operation_key=operation,
            predecessor_event_sha256=predecessor,
            evidence={},
            output=output,
            mutation_requested=mutation_requested,
            mutation_outcome=mutation_outcome,
            recorded_at_utc=NOW,
        )
        if (
            self.crash_after_first_delete
            and label == "cleanup-delete-r00"
            and not self.crashed
        ):
            self.crashed = True
            raise KeyboardInterrupt("simulated process loss after durable terminal")
        return event


def _execute_kwargs(
    prepared: subject.PreparedCleanup,
    fake: FakeCleanupDispatcher,
) -> dict[str, Any]:
    return {
        "prepared": prepared,
        "allow_cloud_read": True,
        "allow_gce_delete": True,
        "allow_worker_iam_cleanup": True,
        "confirm_run_name": RUN,
        "dispatcher": fake,
        "clock": lambda: datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc),
    }


def test_execution_manifest_requires_exact_distinct_persisted_request_ids(
    tmp_path: Path,
) -> None:
    plan, ledger, resume, _controller, manifest, _launch = _launch_journal(tmp_path)
    checked = subject._validate_execution_manifest(
        value=manifest,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
    )
    assert checked["cleanup_request_ids_persisted"] is True
    selected = [row["instance_id"] for row in resume["selected_attempts"]]
    bad = json.loads(json.dumps(manifest))
    bad["gce_delete_request_ids"][selected[0]] = bad[
        "gce_create_request_ids"
    ][selected[0]]
    core = {key: value for key, value in bad.items() if key != "manifest_sha256"}
    bad["manifest_sha256"] = subject.canonical_sha256(core)
    with pytest.raises(ValueError, match="globally distinct"):
        subject._validate_execution_manifest(
            value=bad,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
        )


def test_plan_is_write_once_and_explicitly_preserves_shared_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume, _controller, manifest, launch = _launch_journal(tmp_path)
    _patch_terminal(monkeypatch, launch, manifest["gce_create_receipt"])
    kwargs = {
        "execution_manifest": manifest,
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        "cleanup_root": tmp_path / "cleanup",
    }
    first = subject.prepare_cleanup(**kwargs)
    second = subject.prepare_cleanup(**kwargs)
    assert first.cleanup_plan == second.cleanup_plan
    assert first.cleanup_plan["shared_content_cleanup_authorized"] is False
    assert first.cleanup_plan["shared_content_preserved_for_later_executions"] is True
    assert first.cleanup_plan["cloud_mutation_performed"] is False
    assert not (first.root / "steps").exists()


def test_execute_requires_exact_confirmation_and_all_cleanup_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher()
    common = _execute_kwargs(prepared, fake)
    with pytest.raises(PermissionError, match="exactly match"):
        subject.execute_production_cleanup(**{**common, "confirm_run_name": "wrong"})
    with pytest.raises(PermissionError, match="allow_worker_iam_cleanup"):
        subject.execute_production_cleanup(
            **{**common, "allow_worker_iam_cleanup": False}
        )
    assert fake.calls == []


def test_happy_cleanup_sequence_is_idempotent_and_never_deletes_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher()
    result = subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    assert [mode for mode, _request in fake.calls] == [
        "cleanup",
        "cleanup",
        "worker-iam-cleanup",
        "cleanup",
    ]
    assert [request.get("step") for _mode, request in fake.calls] == [
        "delete-instances",
        "verify-instance-absence",
        None,
        "closeout",
    ]
    first_request = fake.calls[0][1]
    assert first_request["request_ids"] == prepared.execution_manifest[
        "gce_delete_request_ids"
    ]
    assert first_request["orphan_disk_request_ids"] == prepared.execution_manifest[
        "orphan_disk_delete_request_ids"
    ]
    assert result["status"] == subject.CLEANUP_MANIFEST_STATUS
    assert result["shared_content_preserved_for_later_executions"] is True
    assert result["content_cleanup_event_sha256"] is None
    assert result["receiver_request"]["closeout_event_sha256"] == result[
        "closeout_event_sha256"
    ]
    request_path = prepared.root / "receiver_request.json"
    persisted_request = json.loads(request_path.read_text(encoding="utf-8"))
    assert persisted_request == result["receiver_request"]
    assert result["receiver_request_sha256"] == persisted_request[
        "request_sha256"
    ]
    assert receiver_v2.validate_production_receive_request(
        persisted_request,
        expected_execution_namespace=prepared.execution_manifest[
            "execution_namespace"
        ],
        expected_controller_journal_dir=prepared.journal_dir,
    ) == persisted_request
    assert receiver_v2.resolve_production_receive_journal_dir(
        request=persisted_request,
        journal_dir=prepared.journal_dir,
        journal_dir_mode=receiver_v2.JOURNAL_DIR_MODE_EXACT,
        expected_execution_namespace=prepared.execution_manifest[
            "execution_namespace"
        ],
    ) == prepared.journal_dir.resolve()
    before = len(fake.calls)
    assert subject.execute_production_cleanup(**_execute_kwargs(prepared, fake)) == result
    assert len(fake.calls) == before


def test_valid_single_job_resume_cleanup_keeps_exact_one_name_and_request_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_hu_m31_t3_step6d_full100_wave_launch_bundle_v2 import (
        _single_job_context,
    )

    plan, ledger, resume = _single_job_context()
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=tmp_path / "journal",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
    )
    create = _receipt("gce-create", create_complete=False, rows=[])
    launch = controller.journal.append(
        phase="reconcile-create",
        mode="read-only",
        status="partial",
        operation_key=(
            f"{receiver_v2.execution_namespace(ledger, resume)}:reconcile-create"
        ),
        predecessor_event_sha256=None,
        evidence={},
        output={"gce_create_receipt": create, "actual_launch_receipt": None},
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc=NOW,
    )
    manifest = _manifest(
        plan=plan,
        ledger=ledger,
        resume=resume,
        journal_dir=controller.journal.directory,
        launch_event_sha256=launch.value["event_sha256"],
        create_receipt=create,
    )
    _patch_terminal(monkeypatch, launch.value, create)
    prepared = subject.prepare_cleanup(
        execution_manifest=manifest,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        cleanup_root=tmp_path / "cleanup",
    )
    fake = FakeCleanupDispatcher()
    result = subject.execute_production_cleanup(
        prepared=prepared,
        allow_cloud_read=True,
        allow_gce_delete=True,
        allow_worker_iam_cleanup=True,
        confirm_run_name=plan["run_name"],
        dispatcher=fake,
        clock=lambda: datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc),
    )

    selected_name = resume["selected_attempts"][0]["instance_id"]
    assert len(plan["waves"][0]["job_ids"]) == 8
    assert prepared.cleanup_plan["selected_instance_ids"] == [selected_name]
    delete_request = fake.calls[0][1]
    assert list(delete_request["request_ids"]) == [selected_name]
    assert list(delete_request["orphan_disk_request_ids"]) == [selected_name]
    assert [request.get("step") for _mode, request in fake.calls] == [
        "delete-instances",
        "verify-instance-absence",
        None,
        "closeout",
    ]
    assert result["all_owned_instances_absent"] is True
    assert result["all_owned_boot_disks_absent"] is True
    assert result["shared_content_preserved_for_later_executions"] is True


def test_standalone_receiver_request_tamper_fails_before_cleanup_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher()
    subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    path = prepared.root / "receiver_request.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["controller_journal_dir"] = str(prepared.root.resolve())
    path.write_text(json.dumps(value), encoding="utf-8")
    before = len(fake.calls)
    with pytest.raises(ValueError, match="digest|journal"):
        subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    assert len(fake.calls) == before


def test_gce_transport_ambiguity_uses_get_only_then_exact_orphan_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher(gce_ambiguity=True)
    subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    modes = [mode for mode, _request in fake.calls]
    assert modes[:3] == ["cleanup", "reconcile-delete", "cleanup"]
    reconcile = fake.calls[1][1]
    assert reconcile["source_operation_key"].endswith("cleanup-delete-r00")
    orphan = fake.calls[2][1]
    assert orphan["reconcile_event_sha256"] is not None
    assert orphan["request_ids"] == prepared.execution_manifest[
        "gce_delete_request_ids"
    ]
    assert fake.mutation_appends.count("cleanup-delete-r00") == 1
    assert fake.mutation_appends.count("cleanup-delete-r01") == 1


def test_worker_iam_transport_ambiguity_uses_get_only_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher(iam_ambiguity=True)
    result = subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    modes = [mode for mode, _request in fake.calls]
    assert modes == [
        "cleanup",
        "cleanup",
        "worker-iam-cleanup",
        "worker-iam-reconcile-cleanup",
        "cleanup",
    ]
    assert result["worker_iam_bindings_absent"] is True
    assert fake.mutation_appends.count("worker-iam-cleanup") == 1


def test_controller_allows_get_only_reconcile_after_ambiguous_orphan_round(
    tmp_path: Path,
) -> None:
    _plan, _ledger, _resume, controller, manifest, launch = _launch_journal(tmp_path)
    request_ids = manifest["gce_delete_request_ids"]
    primary_intent = controller.journal.append(
        phase="cleanup-delete",
        mode="mutation",
        status="intent",
        operation_key="delete-primary",
        predecessor_event_sha256=launch["event_sha256"],
        evidence={
            "launch_event_sha256": launch["event_sha256"],
            "request_ids_sha256": subject.canonical_sha256(request_ids),
            "orphan_disk_request_ids_sha256": subject.canonical_sha256(
                manifest["orphan_disk_delete_request_ids"]
            ),
            "reconcile_event_sha256": None,
        },
        output=None,
        mutation_requested=True,
        mutation_outcome="pending",
        recorded_at_utc=NOW,
    )
    primary_failed = controller.journal.append(
        phase="cleanup-delete",
        mode="mutation",
        status="failed",
        operation_key="delete-primary",
        predecessor_event_sha256=primary_intent.value["event_sha256"],
        evidence={"intent_event_sha256": primary_intent.value["event_sha256"]},
        output={"failure_kind": "transport_ambiguity"},
        mutation_requested=True,
        mutation_outcome="unknown",
        recorded_at_utc=NOW,
    )
    partial = controller.journal.append(
        phase="reconcile-delete",
        mode="read-only",
        status="partial",
        operation_key="reconcile-primary",
        predecessor_event_sha256=primary_failed.value["event_sha256"],
        evidence={
            "source_operation_key": "delete-primary",
            "source_failure_event_sha256": primary_failed.value["event_sha256"],
            "launch_event_sha256": launch["event_sha256"],
            "request_ids_sha256": subject.canonical_sha256(request_ids),
        },
        output={
            "gce_delete_reconciliation_receipt": {
                "orphan_cleanup_required": True
            }
        },
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc=NOW,
    )
    orphan_intent = controller.journal.append(
        phase="cleanup-delete",
        mode="mutation",
        status="intent",
        operation_key="delete-orphan",
        predecessor_event_sha256=partial.value["event_sha256"],
        evidence={
            "launch_event_sha256": launch["event_sha256"],
            "request_ids_sha256": subject.canonical_sha256(request_ids),
            "orphan_disk_request_ids_sha256": subject.canonical_sha256(
                manifest["orphan_disk_delete_request_ids"]
            ),
            "reconcile_event_sha256": partial.value["event_sha256"],
        },
        output=None,
        mutation_requested=True,
        mutation_outcome="pending",
        recorded_at_utc=NOW,
    )
    controller.journal.append(
        phase="cleanup-delete",
        mode="mutation",
        status="failed",
        operation_key="delete-orphan",
        predecessor_event_sha256=orphan_intent.value["event_sha256"],
        evidence={"intent_event_sha256": orphan_intent.value["event_sha256"]},
        output={"failure_kind": "transport_ambiguity"},
        mutation_requested=True,
        mutation_outcome="unknown",
        recorded_at_utc=NOW,
    )

    class GetOnlyAdapter:
        @staticmethod
        def reconcile_delete(**kwargs: Any) -> Mapping[str, Any]:
            assert kwargs["request_ids"] == request_ids
            return {"orphan_cleanup_required": True}

        @staticmethod
        def validate_delete_reconcile_receipt(
            value: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return dict(value)

    event = controller.reconcile_delete(
        operation_key="reconcile-orphan",
        source_operation_key="delete-orphan",
        launch_event_sha256=launch["event_sha256"],
        adapter=GetOnlyAdapter(),
        request_ids=request_ids,
        observed_at_utc=NOW,
    )
    assert event.value["phase"] == "reconcile-delete"
    assert event.value["status"] == "partial"
    assert event.value["mutation_requested"] is False

    forged_intent = controller.journal.append(
        phase="cleanup-delete",
        mode="mutation",
        status="intent",
        operation_key="delete-forged-orphan",
        predecessor_event_sha256=partial.value["event_sha256"],
        evidence={
            "launch_event_sha256": launch["event_sha256"],
            "request_ids_sha256": subject.canonical_sha256(request_ids),
            "orphan_disk_request_ids_sha256": subject.canonical_sha256(
                manifest["orphan_disk_delete_request_ids"]
            ),
            # The durable intent must name the exact partial predecessor.
            "reconcile_event_sha256": launch["event_sha256"],
        },
        output=None,
        mutation_requested=True,
        mutation_outcome="pending",
        recorded_at_utc=NOW,
    )
    controller.journal.append(
        phase="cleanup-delete",
        mode="mutation",
        status="failed",
        operation_key="delete-forged-orphan",
        predecessor_event_sha256=forged_intent.value["event_sha256"],
        evidence={"intent_event_sha256": forged_intent.value["event_sha256"]},
        output={"failure_kind": "transport_ambiguity"},
        mutation_requested=True,
        mutation_outcome="unknown",
        recorded_at_utc=NOW,
    )
    with pytest.raises(PermissionError, match="transport-ambiguous source"):
        controller.reconcile_delete(
            operation_key="reconcile-forged-orphan",
            source_operation_key="delete-forged-orphan",
            launch_event_sha256=launch["event_sha256"],
            adapter=GetOnlyAdapter(),
            request_ids=request_ids,
            observed_at_utc=NOW,
        )


def test_restart_after_durable_delete_terminal_does_not_repeat_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher(crash_after_first_delete=True)
    with pytest.raises(KeyboardInterrupt, match="process loss"):
        subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    request_path = prepared.root / "steps" / "cleanup-delete-r00" / "request.json"
    request_bytes = request_path.read_bytes()
    result = subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    assert result["all_owned_instances_absent"] is True
    assert fake.mutation_appends.count("cleanup-delete-r00") == 1
    assert request_path.read_bytes() == request_bytes


def test_persisted_request_tamper_fails_before_another_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    fake = FakeCleanupDispatcher(crash_after_first_delete=True)
    with pytest.raises(KeyboardInterrupt):
        subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    path = prepared.root / "steps" / "cleanup-delete-r00" / "request.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["request"]["request_ids"] = {}
    path.write_text(json.dumps(value), encoding="utf-8")
    before = len(fake.calls)
    with pytest.raises(ValueError, match="digest|binding"):
        subject.execute_production_cleanup(**_execute_kwargs(prepared, fake))
    assert len(fake.calls) == before


def test_cli_exposes_no_shared_content_delete_flag() -> None:
    parser = subject._parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--allow-content-delete" not in options
