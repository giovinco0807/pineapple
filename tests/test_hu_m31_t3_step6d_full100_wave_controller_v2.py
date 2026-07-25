from __future__ import annotations

import json
import urllib.parse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2
from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_worker_identity_gcp_adapter_v2 as identity_gcp_v2,
)
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as identity_v2
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


NOW = "2026-07-22T05:00:00Z"
LATER = "2026-07-22T05:00:01Z"


def _controller(tmp_path: Path) -> subject.Full100WaveControllerV2:
    plan = wave_v2.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-20260722-controller",
        identity_salt="0123456789abcdef0123456789abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-solver-485418",
        zone="asia-northeast1-b",
        observed_at_utc="2026-07-22T04:59:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return subject.Full100WaveControllerV2(
        journal_dir=tmp_path / "journal",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
    )


def _receipt(label: str, **values: Any) -> dict[str, Any]:
    core = {"kind": label, **values, "current_profile_changed": False}
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


def _prepare_event(controller: subject.Full100WaveControllerV2) -> subject.JournalEvent:
    output = {
        "schema": subject.PREPARE_SCHEMA,
        "stage_plan_sha256": "4" * 64,
        "content_preflight_receipt_sha256": "5" * 64,
        "current_profile_changed": False,
    }
    return controller.journal.append(
        phase="prepare",
        mode="read-only",
        status="complete",
        operation_key="prepare-01",
        predecessor_event_sha256=None,
        evidence={},
        output=output,
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc=NOW,
    )


def _stage_event(
    controller: subject.Full100WaveControllerV2,
    prepare: subject.JournalEvent,
) -> subject.JournalEvent:
    return controller.journal.append(
        phase="stage-content",
        mode="mutation",
        status="complete",
        operation_key="stage-01",
        predecessor_event_sha256=prepare.value["event_sha256"],
        evidence={},
        output={
            "stage_receipt": _receipt(
                "stage", stage_complete=True, content_payload_sha256="6" * 64
            )
        },
        mutation_requested=True,
        mutation_outcome="performed",
        recorded_at_utc=LATER,
    )


def test_startup_hash_constant_matches_frozen_script() -> None:
    path = Path("scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh")
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
    )


def test_journal_is_canonical_contiguous_and_restart_readable(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    first = _prepare_event(controller)
    inspection = controller.inspect()
    assert inspection["event_count"] == 1
    assert inspection["latest_event_sha256"] == first.value["event_sha256"]
    assert inspection["pending_mutation_operations"] == []
    raw = first.path.read_bytes()
    assert raw == subject.canonical_bytes(json.loads(raw)) + b"\n"

    reopened = subject.ControllerJournal(
        first.path.parent,
        context_sha256=controller.context_sha256,
        create=False,
    )
    assert reopened.load()[0].value == first.value


def test_journal_reads_legacy_phase_suffixed_event_filename(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    event = _prepare_event(controller)
    legacy_path = event.path.with_name(
        f"{event.value['sequence']:06d}-{event.value['phase']}.json"
    )
    event.path.rename(legacy_path)
    reopened = subject.ControllerJournal(
        legacy_path.parent,
        context_sha256=controller.context_sha256,
        create=False,
    )
    loaded = reopened.load()
    assert len(loaded) == 1
    assert loaded[0].path == legacy_path
    assert loaded[0].value == event.value


@pytest.mark.parametrize("tamper", ["bytes", "extra", "gap"])
def test_journal_fails_closed_on_tamper(tmp_path: Path, tamper: str) -> None:
    controller = _controller(tmp_path)
    event = _prepare_event(controller)
    if tamper == "bytes":
        value = deepcopy(event.value)
        value["output"]["changed"] = True
        event.path.write_bytes(subject.canonical_bytes(value) + b"\n")
    elif tamper == "extra":
        (event.path.parent / "notes.txt").write_text("not allowed", encoding="utf-8")
    else:
        event.path.rename(event.path.parent / "000002-prepare.json")
    with pytest.raises(subject.JournalTamperError):
        controller.inspect()


class FakeIdentityAdapter:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def create_missing(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider failed")
        return _receipt("identity-create", **kwargs)

    def validate_create_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        assert value["kind"] == "identity-create"
        return value


def test_setup_identities_requires_explicit_mutation_and_is_idempotent(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    adapter = FakeIdentityAdapter()
    kwargs = dict(
        operation_key="setup-01",
        prepare_event_sha256=prepare.value["event_sha256"],
        adapter=adapter,
        current_time_utc=NOW,
        completed_at_utc=LATER,
    )
    with pytest.raises(PermissionError):
        controller.setup_identities(**kwargs)
    assert adapter.calls == 0

    complete = controller.setup_identities(**kwargs, allow_cloud_mutation=True)
    assert complete.value["status"] == "complete"
    assert complete.value["mutation_outcome"] == "performed"
    assert adapter.calls == 1
    assert controller.setup_identities(
        **kwargs, allow_cloud_mutation=True
    ).value == complete.value
    assert adapter.calls == 1


def test_failed_mutation_is_terminal_and_never_replayed(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    adapter = FakeIdentityAdapter(fail=True)
    with pytest.raises(RuntimeError, match="provider failed"):
        controller.setup_identities(
            operation_key="setup-fail",
            prepare_event_sha256=prepare.value["event_sha256"],
            adapter=adapter,
            current_time_utc=NOW,
            completed_at_utc=LATER,
            allow_cloud_mutation=True,
        )
    assert adapter.calls == 1
    event = controller.setup_identities(
        operation_key="setup-fail",
        prepare_event_sha256=prepare.value["event_sha256"],
        adapter=adapter,
        current_time_utc=NOW,
        completed_at_utc=LATER,
        allow_cloud_mutation=True,
    )
    assert event.value["status"] == "failed"
    assert adapter.calls == 1


def test_dangling_intent_blocks_all_new_mutations_until_reconciled(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    intent, terminal = controller._begin_mutation(
        phase="setup-identities",
        operation_key="dangling-01",
        predecessor_event_sha256=prepare.value["event_sha256"],
        evidence={},
        recorded_at_utc=NOW,
    )
    assert intent is not None and terminal is None
    with pytest.raises(subject.PendingMutationError):
        controller.setup_identities(
            operation_key="setup-02",
            prepare_event_sha256=prepare.value["event_sha256"],
            adapter=FakeIdentityAdapter(),
            current_time_utc=NOW,
            completed_at_utc=LATER,
            allow_cloud_mutation=True,
        )
    reconciled = controller.reconcile_pending_mutation(
        operation_key="dangling-01",
        receipt=_receipt("recovered"),
        validator=lambda value: value,
        complete=False,
        recorded_at_utc=LATER,
    )
    assert reconciled.value["status"] == "reconciled-partial"
    assert controller.inspect()["pending_mutation_operations"] == []


def test_prepare_calls_all_producer_owned_validators_and_records_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path)
    manifest = {
        "manifest_sha256": "1" * 64,
        "run_name": controller.context["run_name"],
    }
    stage_plan = {"plan_sha256": "2" * 64}
    preflight = {"receipt_sha256": "3" * 64}
    identity = {"plan_sha256": "4" * 64}
    runtime = {"receipt_sha256": "5" * 64}
    runtime_gcp = {
        "receipt_sha256": "6" * 64,
        "runtime_preflight_receipt": runtime,
    }
    calls: list[str] = []

    def outer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("outer")
        return manifest

    def stage(m: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
        calls.append("stage")
        assert m is manifest and value is stage_plan
        return stage_plan

    def absent(p: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
        calls.append("preflight")
        assert p is stage_plan and value is preflight
        return preflight

    def ident(**kwargs: Any) -> dict[str, Any]:
        calls.append("identity")
        return identity

    def runtime_check(**kwargs: Any) -> dict[str, Any]:
        calls.append("runtime")
        assert kwargs["current_utc"] == NOW
        return runtime

    def runtime_gcp_check(**kwargs: Any) -> dict[str, Any]:
        calls.append("runtime-gcp")
        assert kwargs["current_utc"] == NOW
        return runtime_gcp

    monkeypatch.setattr(subject.package_v2, "validate_outer_manifest", outer)
    monkeypatch.setattr(subject.content_v2, "validate_content_stage_plan", stage)
    monkeypatch.setattr(subject.content_v2, "validate_preflight_absence_receipt", absent)
    monkeypatch.setattr(subject.identity_v2, "validate_worker_identity_plan", ident)
    monkeypatch.setattr(subject.runtime_v2, "validate_runtime_preflight_receipt", runtime_check)
    monkeypatch.setattr(
        subject.runtime_gcp_v2,
        "validate_runtime_gcp_read_receipt",
        runtime_gcp_check,
    )

    event = controller.prepare(
        operation_key="prepare-real",
        outer_manifest=manifest,
        stage_plan=stage_plan,
        content_preflight_receipt=preflight,
        worker_identity_plan=identity,
        runtime_preflight_receipt=runtime,
        runtime_gcp_read_receipt=runtime_gcp,
        current_time_utc=NOW,
    )
    assert calls == [
        "outer", "stage", "preflight", "identity", "runtime", "runtime-gcp"
    ]
    assert event.value["output"]["runtime_preflight_receipt_sha256"] == "5" * 64
    assert event.value["output"]["runtime_gcp_read_receipt_sha256"] == "6" * 64
    assert event.value["output"]["cloud_mutation_authorized"] is False


def test_stage_partial_receipt_is_durably_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    plan = {"plan_sha256": "4" * 64}
    preflight = {"receipt_sha256": "5" * 64}
    partial = _receipt("partial-stage", stage_complete=False)
    monkeypatch.setattr(subject.content_v2, "_validate_stage_plan_self", lambda value: plan)
    monkeypatch.setattr(
        subject.content_v2, "validate_preflight_absence_receipt", lambda p, value: preflight
    )
    monkeypatch.setattr(
        subject.content_v2,
        "validate_stage_receipt",
        lambda p, f, value: value,
    )

    def fail(**kwargs: Any) -> dict[str, Any]:
        raise subject.content_v2.ContentStageIncompleteError(
            "partial", partial_receipt=partial
        )

    monkeypatch.setattr(subject.content_v2, "execute_content_stage", fail)
    with pytest.raises(subject.content_v2.ContentStageIncompleteError):
        controller.stage_content(
            operation_key="stage-partial",
            prepare_event_sha256=prepare.value["event_sha256"],
            package_dir=tmp_path,
            stage_plan=plan,
            preflight_receipt=preflight,
            backend=object(),  # type: ignore[arg-type]
            observed_at_utc=LATER,
            allow_cloud_mutation=True,
        )
    terminal = controller._terminal_for("stage-partial")
    assert terminal is not None
    assert terminal.value["status"] == "partial"
    assert terminal.value["output"]["stage_receipt"] == partial


def test_three_wave_journals_rebind_one_stage_receipt_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_sha256": "4" * 64}
    preflight = {"receipt_sha256": "5" * 64}
    rows = [
        {
            "object_name": f"immutable/object-{index:02d}",
            "generation": str(1000 + index),
            "sha256": subject.canonical_sha256(["object", index]),
            "bytes": index + 1,
        }
        for index in range(26)
    ]
    source_receipt = _receipt(
        "stage",
        stage_complete=True,
        created_entry_count=26,
        content_payload_sha256="6" * 64,
        rows=rows,
        observed_at_utc=NOW,
    )
    readbacks: list[object] = []

    monkeypatch.setattr(
        subject.content_v2, "_validate_stage_plan_self", lambda value: plan
    )
    monkeypatch.setattr(
        subject.content_v2,
        "validate_preflight_absence_receipt",
        lambda p, value: preflight,
    )
    monkeypatch.setattr(
        subject.content_v2,
        "validate_stage_receipt",
        lambda p, f, value: deepcopy(dict(value)),
    )

    def readback(**kwargs: Any) -> dict[str, Any]:
        readbacks.append(kwargs["backend"])
        assert kwargs["created_rows"] == rows
        return {
            **deepcopy(source_receipt),
            "observed_at_utc": kwargs["observed_at_utc"],
        }

    monkeypatch.setattr(
        subject.content_v2, "build_stage_readback_receipt", readback
    )
    events = []
    for wave_index in range(3):
        controller = _controller(tmp_path / f"wave-{wave_index}")
        prepare = _prepare_event(controller)
        backend = object()
        event = controller.bind_existing_staged_content(
            operation_key=f"bind-existing-wave-{wave_index}",
            prepare_event_sha256=prepare.value["event_sha256"],
            stage_plan=plan,
            preflight_receipt=preflight,
            source_stage_receipt=source_receipt,
            backend=backend,  # type: ignore[arg-type]
            observed_at_utc=f"2026-07-22T05:00:0{wave_index + 1}Z",
        )
        events.append(event.value)
    assert len(readbacks) == 3
    assert all(event["phase"] == "stage-content" for event in events)
    assert all(event["mode"] == "read-only" for event in events)
    assert all(event["mutation_requested"] is False for event in events)
    assert all(event["mutation_outcome"] == "not_requested" for event in events)
    assert all(
        event["output"]["source_stage_receipt_sha256"]
        == source_receipt["receipt_sha256"]
        for event in events
    )
    assert all(
        event["output"]["content_reused_without_mutation"] is True
        for event in events
    )


def test_bridge_maps_exact_gce_rows_to_cloud_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = {
        "rows": [
            {
                "job_id": "candidate-00",
                "source_role": "candidate",
                "attempt_id": "candidate-00-a00",
                "instance_id": "instance-00",
                "ownership_label": "owned-00",
                "machine_type": "c4-standard-16",
                "vcpus": 16,
            }
        ]
    }
    create = {
        "create_complete": True,
        "observed_at_utc": LATER,
        "rows": [
            {
                "job_id": "candidate-00",
                "source_role": "candidate",
                "attempt_id": "candidate-00-a00",
                "instance_name": "instance-00",
                "recovered_after_insert_failure": False,
                "operation_id": "12345",
                "observed_status": "RUNNING",
            }
        ],
    }
    captured: dict[str, Any] = {}

    def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _receipt("actual-launch")

    monkeypatch.setattr(subject.cloud_v2, "build_actual_launch_receipt", build)
    result = subject.Full100WaveControllerV2.bridge_actual_launch_receipt(
        wave_plan={},
        attempt_ledger={},
        resume_plan={},
        launch_bundle={"immutable_content_sha256": "7" * 64},
        gce_create_receipt=create,
        quota_receipt={},
        persistent_claim_receipt={},
        planned_mapping_receipt=planned,
        prelaunch_authorization={},
        launch_started_at_utc=NOW,
    )
    assert result["kind"] == "actual-launch"
    assert captured["readback_source"] == "compute_instances_and_disks_api"
    assert captured["instance_create_readbacks"] == [
        {
            "job_id": "candidate-00",
            "source_role": "candidate",
            "attempt_id": "candidate-00-a00",
            "instance_id": "instance-00",
            "operation_id": "12345",
            "operation_status": "DONE",
            "instance_status": "RUNNING",
            "ownership_label": "owned-00",
            "machine_type": "c4-standard-16",
            "vcpus": 16,
            "instance_readback_complete": True,
        }
    ]


def test_bridge_rejects_recovered_or_partial_create() -> None:
    with pytest.raises(ValueError, match="complete"):
        subject.Full100WaveControllerV2.bridge_actual_launch_receipt(
            wave_plan={},
            attempt_ledger={},
            resume_plan={},
            launch_bundle={"immutable_content_sha256": "7" * 64},
            gce_create_receipt={"create_complete": False, "rows": []},
            quota_receipt={},
            persistent_claim_receipt={},
            planned_mapping_receipt={"rows": []},
            prelaunch_authorization={},
            launch_started_at_utc=NOW,
        )


class FakeDeleteAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def delete_owned(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls += 1
        return _receipt(
            "delete", delete_operation_count=1, observed_at_utc=kwargs["observed_at_utc"]
        )

    def validate_delete_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return value


class FakeAbsenceAdapter:
    def verify_absence(self, **kwargs: Any) -> Mapping[str, Any]:
        return _receipt(
            "absence",
            delete_receipt_sha256="8" * 64,
            observed_at_utc=kwargs["observed_at_utc"],
        )

    def validate_absence_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return value


class FakeDeleteReconcileAdapter:
    def __init__(self, *, partial: bool) -> None:
        self.partial = partial
        self.calls = 0

    def reconcile_delete(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls += 1
        return _receipt(
            "delete-reconcile",
            orphan_cleanup_required=self.partial,
            recovered_delete_receipt=(
                None
                if self.partial
                else _receipt("recovered-delete", delete_operation_count=0)
            ),
        )

    def validate_delete_reconcile_receipt(
        self, value: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return value


@pytest.mark.parametrize("partial", [False, True])
def test_delete_transport_ambiguity_has_get_only_controller_reconciliation(
    tmp_path: Path, partial: bool
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    stage = _stage_event(controller, prepare)
    launch = controller.journal.append(
        phase="authorize-launch",
        mode="mutation",
        status="partial",
        operation_key="launch-delete-reconcile",
        predecessor_event_sha256=stage.value["event_sha256"],
        evidence={},
        output={"gce_create_receipt": _receipt("create")},
        mutation_requested=True,
        mutation_outcome="unknown",
        recorded_at_utc=LATER,
    )

    class AmbiguousDelete:
        def delete_owned(self, **kwargs: Any) -> Mapping[str, Any]:
            raise subject.gce_v2.GcePhaseBTransportError("DELETE response lost")

        def validate_delete_receipt(
            self, value: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return value

    with pytest.raises(subject.gce_v2.GcePhaseBTransportError):
        controller.delete_instances(
            operation_key="delete-response-lost",
            launch_event_sha256=launch.value["event_sha256"],
            adapter=AmbiguousDelete(),
            request_ids={"instance-00": "11111111-1111-4111-8111-111111111111"},
            observed_at_utc="2026-07-22T05:00:02Z",
            allow_cloud_mutation=True,
        )
    failure = controller.journal.load()[-1]
    assert failure.value["output"]["failure_kind"] == "transport_ambiguity"
    adapter = FakeDeleteReconcileAdapter(partial=partial)
    reconciled = controller.reconcile_delete(
        operation_key=f"reconcile-delete-{'partial' if partial else 'complete'}",
        source_operation_key="delete-response-lost",
        launch_event_sha256=launch.value["event_sha256"],
        adapter=adapter,
        request_ids={"instance-00": "11111111-1111-4111-8111-111111111111"},
        observed_at_utc="2026-07-22T05:00:03Z",
    )
    assert adapter.calls == 1
    assert reconciled.value["mode"] == "read-only"
    assert reconciled.value["status"] == ("partial" if partial else "complete")
    if partial:
        followup = controller.delete_instances(
            operation_key="delete-reconciled-orphan",
            launch_event_sha256=launch.value["event_sha256"],
            reconcile_event_sha256=reconciled.value["event_sha256"],
            adapter=FakeDeleteAdapter(),
            request_ids={"instance-00": "22222222-2222-4222-8222-222222222222"},
            observed_at_utc="2026-07-22T05:00:04Z",
            allow_cloud_mutation=True,
        )
        assert followup.value["predecessor_event_sha256"] != reconciled.value[
            "event_sha256"
        ]
        intent = controller.journal.load()[-2]
        assert intent.value["predecessor_event_sha256"] == reconciled.value[
            "event_sha256"
        ]


def test_delete_reconciliation_rejects_request_id_lineage_drift_before_adapter(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    stage = _stage_event(controller, prepare)
    launch = controller.journal.append(
        phase="authorize-launch",
        mode="mutation",
        status="partial",
        operation_key="launch-delete-lineage",
        predecessor_event_sha256=stage.value["event_sha256"],
        evidence={},
        output={"gce_create_receipt": _receipt("create")},
        mutation_requested=True,
        mutation_outcome="unknown",
        recorded_at_utc=LATER,
    )

    class AmbiguousDelete:
        def delete_owned(self, **kwargs: Any) -> Mapping[str, Any]:
            raise subject.gce_v2.GcePhaseBTransportError("DELETE response lost")

        def validate_delete_receipt(
            self, value: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return value

    with pytest.raises(subject.gce_v2.GcePhaseBTransportError):
        controller.delete_instances(
            operation_key="delete-lineage-response-lost",
            launch_event_sha256=launch.value["event_sha256"],
            adapter=AmbiguousDelete(),
            request_ids={
                "instance-00": "11111111-1111-4111-8111-111111111111"
            },
            observed_at_utc="2026-07-22T05:00:02Z",
            allow_cloud_mutation=True,
        )

    adapter = FakeDeleteReconcileAdapter(partial=False)
    with pytest.raises(PermissionError, match="request IDs do not match"):
        controller.reconcile_delete(
            operation_key="reconcile-delete-lineage-drift",
            source_operation_key="delete-lineage-response-lost",
            launch_event_sha256=launch.value["event_sha256"],
            adapter=adapter,
            request_ids={
                "instance-00": "22222222-2222-4222-8222-222222222222"
            },
            observed_at_utc="2026-07-22T05:00:03Z",
        )
    assert adapter.calls == 0


def test_explicit_delete_failure_cannot_use_reconcile_delete(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    stage = _stage_event(controller, prepare)
    launch = controller.journal.append(
        phase="authorize-launch", mode="mutation", status="partial",
        operation_key="launch-delete-explicit",
        predecessor_event_sha256=stage.value["event_sha256"], evidence={},
        output={"gce_create_receipt": _receipt("create")},
        mutation_requested=True, mutation_outcome="unknown", recorded_at_utc=LATER,
    )

    class ExplicitDeleteFailure:
        def delete_owned(self, **kwargs: Any) -> Mapping[str, Any]:
            raise RuntimeError("provider returned 500")

        def validate_delete_receipt(
            self, value: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return value

    with pytest.raises(RuntimeError, match="500"):
        controller.delete_instances(
            operation_key="delete-explicit-failure",
            launch_event_sha256=launch.value["event_sha256"],
            adapter=ExplicitDeleteFailure(),
            request_ids={"instance-00": "11111111-1111-4111-8111-111111111111"},
            observed_at_utc="2026-07-22T05:00:02Z",
            allow_cloud_mutation=True,
        )
    with pytest.raises(PermissionError, match="transport-ambiguous"):
        controller.reconcile_delete(
            operation_key="reconcile-explicit-delete",
            source_operation_key="delete-explicit-failure",
            launch_event_sha256=launch.value["event_sha256"],
            adapter=FakeDeleteReconcileAdapter(partial=False),
            request_ids={"instance-00": "11111111-1111-4111-8111-111111111111"},
            observed_at_utc="2026-07-22T05:00:03Z",
        )


def test_cleanup_delete_absence_and_closeout_are_chained(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    stage = _stage_event(controller, prepare)
    launch = controller.journal.append(
        phase="authorize-launch",
        mode="mutation",
        status="complete",
        operation_key="launch-01",
        predecessor_event_sha256=stage.value["event_sha256"],
        evidence={},
        output={
            "gce_create_receipt": _receipt("create"),
            "actual_launch_receipt": _receipt("actual"),
        },
        mutation_requested=True,
        mutation_outcome="performed",
        recorded_at_utc=LATER,
    )
    deleted = controller.delete_instances(
        operation_key="cleanup-delete-01",
        launch_event_sha256=launch.value["event_sha256"],
        adapter=FakeDeleteAdapter(),
        request_ids={"instance-00": "11111111-1111-4111-8111-111111111111"},
        observed_at_utc="2026-07-22T05:00:02Z",
        allow_cloud_mutation=True,
    )
    absent = controller.verify_instance_absence(
        operation_key="cleanup-absence-01",
        delete_event_sha256=deleted.value["event_sha256"],
        adapter=FakeAbsenceAdapter(),
        observed_at_utc="2026-07-22T05:00:03Z",
    )
    iam_cleanup = controller.journal.append(
        phase="worker-iam-cleanup",
        mode="mutation",
        status="complete",
        operation_key="worker-iam-cleanup-01",
        predecessor_event_sha256=absent.value["event_sha256"],
        evidence={},
        output={"receipt": _receipt("iam-cleanup", cleanup_complete=True)},
        mutation_requested=True,
        mutation_outcome="performed",
        recorded_at_utc="2026-07-22T05:00:04Z",
    )
    closeout = controller.closeout(
        operation_key="closeout-01",
        launch_event_sha256=launch.value["event_sha256"],
        delete_event_sha256=deleted.value["event_sha256"],
        absence_event_sha256=absent.value["event_sha256"],
        worker_iam_cleanup_event_sha256=iam_cleanup.value["event_sha256"],
        recorded_at_utc="2026-07-22T05:00:05Z",
    )
    assert closeout.value["output"]["all_owned_instances_absent"] is True
    assert closeout.value["output"]["additional_create_authorized"] is False
    assert all(
        event.value["current_profile_changed"] is False
        for event in controller.journal.load()
    )


def test_lifecycle_closeout_chain_revalidates_all_producers_and_maps_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    stage = _stage_event(controller, prepare)
    selected = controller.resume_plan["selected_attempts"]
    create_rows = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_name": row["instance_id"],
            "provider_instance_id": str(1000 + index),
            "provider_boot_disk_id": str(2000 + index),
            "spec_sha256": subject.canonical_sha256(["spec", index]),
            "operation_id": str(3000 + index),
        }
        for index, row in enumerate(selected)
    ]
    actual_rows = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "operation_id": str(4000 + index),
            "instance_status": "RUNNING",
        }
        for index, row in enumerate(selected)
    ]
    mapping_rows = [
        {
            **deepcopy(row),
            "ownership_label": f"owned-{index}",
        }
        for index, row in enumerate(selected)
    ]
    create_receipt = _receipt(
        "gce-create", create_complete=True, rows=create_rows
    )
    actual_receipt = _receipt("actual-launch", rows=actual_rows)
    delete_receipt = _receipt("gce-delete")
    absence_receipt = _receipt(
        "gce-absence",
        checked_instance_count=len(selected),
        absent_instance_names=[row["instance_id"] for row in selected],
        absent_boot_disk_names=[row["instance_id"] for row in selected],
        all_instances_absent=True,
        all_boot_disks_absent=True,
        observed_at_utc="2026-07-22T05:00:04Z",
    )
    iam_receipt = _receipt("iam-cleanup", cleanup_complete=True)
    launch = controller.journal.append(
        phase="authorize-launch",
        mode="mutation",
        status="complete",
        operation_key="launch-proof",
        predecessor_event_sha256=stage.value["event_sha256"],
        evidence={},
        output={
            "gce_create_receipt": create_receipt,
            "actual_launch_receipt": actual_receipt,
        },
        mutation_requested=True,
        mutation_outcome="performed",
        recorded_at_utc="2026-07-22T05:00:02Z",
    )
    delete_intent, terminal = controller._begin_mutation(
        phase="cleanup-delete",
        operation_key="delete-proof",
        predecessor_event_sha256=launch.value["event_sha256"],
        evidence={},
        recorded_at_utc="2026-07-22T05:00:03Z",
    )
    assert delete_intent is not None and terminal is None
    deleted = controller._finish_mutation(
        intent=delete_intent,
        status="complete",
        output={"gce_delete_receipt": delete_receipt},
        outcome="performed",
        recorded_at_utc="2026-07-22T05:00:03Z",
    )
    absent = controller.journal.append(
        phase="cleanup-absence",
        mode="read-only",
        status="complete",
        operation_key="absence-proof",
        predecessor_event_sha256=deleted.value["event_sha256"],
        evidence={},
        output={"gce_absence_receipt": absence_receipt},
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc="2026-07-22T05:00:04Z",
    )
    iam = controller.journal.append(
        phase="worker-iam-cleanup",
        mode="mutation",
        status="complete",
        operation_key="iam-proof",
        predecessor_event_sha256=absent.value["event_sha256"],
        evidence={},
        output={"receipt": iam_receipt},
        mutation_requested=True,
        mutation_outcome="performed",
        recorded_at_utc="2026-07-22T05:00:05Z",
    )
    closeout = controller.closeout(
        operation_key="closeout-proof",
        launch_event_sha256=launch.value["event_sha256"],
        delete_event_sha256=deleted.value["event_sha256"],
        absence_event_sha256=absent.value["event_sha256"],
        worker_iam_cleanup_event_sha256=iam.value["event_sha256"],
        recorded_at_utc="2026-07-22T05:00:06Z",
    )
    calls: list[str] = []

    def checked(label: str):
        def validate(adapter: object, value: Mapping[str, Any]) -> dict[str, Any]:
            calls.append(label)
            return deepcopy(dict(value))

        return validate

    monkeypatch.setattr(subject.gce_v2, "validate_create_receipt", checked("create"))
    monkeypatch.setattr(subject.gce_v2, "validate_delete_receipt", checked("delete"))
    monkeypatch.setattr(subject.gce_v2, "validate_absence_receipt", checked("absence"))

    def mapping_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("mapping")
        return deepcopy(dict(args[3]))

    def actual_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("actual")
        return deepcopy(dict(kwargs["value"]))

    def iam_validator(**kwargs: Any) -> dict[str, Any]:
        calls.append("iam")
        return deepcopy(dict(kwargs["value"]))

    monkeypatch.setattr(
        subject.cloud_v2,
        "validate_planned_launch_mapping_receipt",
        mapping_validator,
    )
    monkeypatch.setattr(
        subject.cloud_v2, "validate_actual_launch_receipt", actual_validator
    )
    monkeypatch.setattr(
        subject.worker_iam_v2, "validate_cleanup_receipt", iam_validator
    )
    mapping_receipt = _receipt("mapping", rows=mapping_rows)
    bundle = {
        "bundle_sha256": "7" * 64,
        "immutable_content_sha256": "6" * 64,
    }
    proof = controller.validate_lifecycle_closeout_chain(
        closeout_event_sha256=closeout.value["event_sha256"],
        launch_bundle=bundle,
        launch_bundle_validator=lambda value: value,
        gce_create_adapter=object(),  # type: ignore[arg-type]
        gce_delete_adapter=object(),  # type: ignore[arg-type]
        gce_absence_adapter=object(),  # type: ignore[arg-type]
        quota_receipt={},
        persistent_claim_receipt={},
        planned_mapping_receipt=mapping_receipt,
        prelaunch_authorization={},
        worker_iam_plan={},
        immutable_content_prefix="immutable/prefix",
        content_payload_sha256="6" * 64,
        outer_manifest_sha256="8" * 64,
        worker_iam_prepare_receipt={},
        worker_iam_install_receipt={},
        worker_iam_readback_receipt={},
    )
    assert calls == ["create", "delete", "absence", "mapping", "actual", "iam"]
    assert proof["schema"] == subject.LIFECYCLE_PROOF_SCHEMA
    assert proof["journal_hash_chain_valid"] is True
    assert proof["all_producer_receipts_valid"] is True
    assert proof["lifecycle_receipt_sha256"] == closeout.value["output"][
        "receipt_sha256"
    ]
    assert proof["gce_create_rows"] == create_rows
    assert proof["actual_launch_rows"] == actual_rows
    assert [row["instance_id"] for row in proof["selected_instance_mapping"]] == [
        row["instance_id"] for row in selected
    ]
    assert [
        row["provider_instance_id"] for row in proof["selected_instance_mapping"]
    ] == [row["provider_instance_id"] for row in create_rows]
    assert proof["create_classification"] == "all_selected_created"
    assert proof["exact_created_instance_count"] == len(selected)
    assert all(
        row["exact_instance_created"] is True
        for row in proof["selected_instance_mapping"]
    )
    assert proof["proof_sha256"] == subject.canonical_sha256(
        {key: value for key, value in proof.items() if key != "proof_sha256"}
    )


@pytest.mark.parametrize(
    ("case", "launch_phase", "launch_status", "created_indices", "classification"),
    [
        (
            "candidate_created_reference_absent",
            "authorize-launch",
            "partial",
            [0],
            "partial_selected_created",
        ),
        (
            "both_absent",
            "reconcile-create",
            "complete",
            [],
            "no_selected_created",
        ),
        (
            "response_loss_reconcile_created",
            "reconcile-create",
            "complete",
            list(range(8)),
            "all_selected_created",
        ),
    ],
)
def test_failure_lifecycle_proof_classifies_created_and_absent_selected_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    launch_phase: str,
    launch_status: str,
    created_indices: list[int],
    classification: str,
) -> None:
    controller = _controller(tmp_path / case)
    prepare = _prepare_event(controller)
    stage = _stage_event(controller, prepare)
    selected = controller.resume_plan["selected_attempts"]
    create_rows = [
        {
            "job_id": selected[index]["job_id"],
            "source_role": selected[index]["source_role"],
            "attempt_id": selected[index]["attempt_id"],
            "instance_name": selected[index]["instance_id"],
            "provider_instance_id": str(5000 + index),
            "provider_boot_disk_id": str(6000 + index),
            "spec_sha256": subject.canonical_sha256(["failure-spec", index]),
            "operation_id": None,
        }
        for index in created_indices
    ]
    create_receipt = _receipt(
        "gce-partial-create", create_complete=False, rows=create_rows
    )
    launch = controller.journal.append(
        phase=launch_phase,
        mode="mutation" if launch_phase == "authorize-launch" else "read-only",
        status=launch_status,
        operation_key=f"launch-{case}",
        predecessor_event_sha256=stage.value["event_sha256"],
        evidence={},
        output={
            "gce_create_receipt": create_receipt,
            "actual_launch_receipt": None,
        },
        mutation_requested=launch_phase == "authorize-launch",
        mutation_outcome=(
            "partial" if launch_phase == "authorize-launch" else "not_requested"
        ),
        recorded_at_utc="2026-07-22T05:00:02Z",
    )
    delete_receipt = _receipt("gce-partial-delete")
    delete_intent, terminal = controller._begin_mutation(
        phase="cleanup-delete",
        operation_key=f"delete-{case}",
        predecessor_event_sha256=launch.value["event_sha256"],
        evidence={},
        recorded_at_utc="2026-07-22T05:00:03Z",
    )
    assert delete_intent is not None and terminal is None
    deleted = controller._finish_mutation(
        intent=delete_intent,
        status="complete",
        output={"gce_delete_receipt": delete_receipt},
        outcome="performed" if created_indices else "none",
        recorded_at_utc="2026-07-22T05:00:03Z",
    )
    selected_names = [row["instance_id"] for row in selected]
    absence_receipt = _receipt(
        "gce-all-selected-absence",
        checked_instance_count=len(selected),
        absent_instance_names=selected_names,
        absent_boot_disk_names=selected_names,
        all_instances_absent=True,
        all_boot_disks_absent=True,
        observed_at_utc="2026-07-22T05:00:04Z",
    )
    absent = controller.journal.append(
        phase="cleanup-absence",
        mode="read-only",
        status="complete",
        operation_key=f"absence-{case}",
        predecessor_event_sha256=deleted.value["event_sha256"],
        evidence={},
        output={"gce_absence_receipt": absence_receipt},
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc="2026-07-22T05:00:04Z",
    )
    iam_receipt = _receipt("iam-cleanup", cleanup_complete=True)
    iam = controller.journal.append(
        phase="worker-iam-cleanup",
        mode="mutation",
        status="complete",
        operation_key=f"iam-{case}",
        predecessor_event_sha256=absent.value["event_sha256"],
        evidence={},
        output={"receipt": iam_receipt},
        mutation_requested=True,
        mutation_outcome="performed",
        recorded_at_utc="2026-07-22T05:00:05Z",
    )
    closeout = controller.closeout(
        operation_key=f"closeout-{case}",
        launch_event_sha256=launch.value["event_sha256"],
        delete_event_sha256=deleted.value["event_sha256"],
        absence_event_sha256=absent.value["event_sha256"],
        worker_iam_cleanup_event_sha256=iam.value["event_sha256"],
        recorded_at_utc="2026-07-22T05:00:06Z",
    )
    monkeypatch.setattr(
        subject.gce_v2,
        "validate_create_receipt",
        lambda adapter, value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.gce_v2,
        "validate_delete_receipt",
        lambda adapter, value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.gce_v2,
        "validate_absence_receipt",
        lambda adapter, value: deepcopy(dict(value)),
    )
    mapping_rows = [
        {**deepcopy(row), "ownership_label": f"owned-{index}"}
        for index, row in enumerate(selected)
    ]
    monkeypatch.setattr(
        subject.cloud_v2,
        "validate_planned_launch_mapping_receipt",
        lambda *args, **kwargs: deepcopy(dict(args[3])),
    )
    monkeypatch.setattr(
        subject.cloud_v2,
        "validate_actual_launch_receipt",
        lambda *args, **kwargs: pytest.fail(
            "partial/reconciled failure must not validate an absent actual launch"
        ),
    )
    monkeypatch.setattr(
        subject.worker_iam_v2,
        "validate_cleanup_receipt",
        lambda **kwargs: deepcopy(dict(kwargs["value"])),
    )
    proof = controller.validate_lifecycle_closeout_chain(
        closeout_event_sha256=closeout.value["event_sha256"],
        launch_bundle={
            "bundle_sha256": "7" * 64,
            "immutable_content_sha256": "6" * 64,
        },
        launch_bundle_validator=lambda value: value,
        gce_create_adapter=object(),  # type: ignore[arg-type]
        gce_delete_adapter=object(),  # type: ignore[arg-type]
        gce_absence_adapter=object(),  # type: ignore[arg-type]
        quota_receipt={},
        persistent_claim_receipt={},
        planned_mapping_receipt=_receipt("mapping", rows=mapping_rows),
        prelaunch_authorization={},
        worker_iam_plan={},
        immutable_content_prefix="immutable/prefix",
        content_payload_sha256="6" * 64,
        outer_manifest_sha256="8" * 64,
        worker_iam_prepare_receipt={},
        worker_iam_install_receipt={},
        worker_iam_readback_receipt={},
    )
    assert proof["create_classification"] == classification
    assert proof["actual_launch_receipt"] is None
    assert proof["actual_launch_rows"] == []
    assert proof["exact_created_instance_count"] == len(created_indices)
    assert proof["exact_uncreated_instance_count"] == len(selected) - len(
        created_indices
    )
    assert [
        index
        for index, row in enumerate(proof["selected_instance_mapping"])
        if row["exact_instance_created"]
    ] == created_indices
    assert len(
        {row["launch_receipt_sha256"] for row in proof["selected_instance_mapping"]}
    ) == len(selected)
    assert all(
        row["final_instance_absent"] is True
        and row["final_boot_disk_absent"] is True
        for row in proof["selected_instance_mapping"]
    )
    for index, row in enumerate(proof["selected_instance_mapping"]):
        if index in created_indices:
            assert row["provider_instance_id"] == str(5000 + index)
        else:
            assert row["provider_instance_id"] is None
            assert row["provider_boot_disk_id"] is None


def test_cli_cloud_modes_are_disabled_without_explicit_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path)
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    resume_path = tmp_path / "resume.json"
    request_path = tmp_path / "request.json"
    plan_path.write_text(json.dumps(controller.wave_plan), encoding="utf-8")
    ledger_path.write_text(json.dumps(controller.attempt_ledger), encoding="utf-8")
    resume_path.write_text(json.dumps(controller.resume_plan), encoding="utf-8")
    request_path.write_text("{}", encoding="utf-8")
    argv = [
        "--journal-dir", str(tmp_path / "cli-journal"),
        "--wave-plan", str(plan_path),
        "--attempt-ledger", str(ledger_path),
        "--resume-plan", str(resume_path),
        "--mode", "stage-content",
        "--request", str(request_path),
        "--confirm-run-name", controller.context["run_name"],
    ]
    with pytest.raises(SystemExit, match="requires allow_content_stage"):
        subject.main(argv)


class FakeIdentityHttp:
    TOKEN = "ya29.controller-fake-requester-token"

    def __init__(self, missing: set[int]) -> None:
        self.accounts = {
            identity_v2.ACCOUNT_IDS[index]: self._desired(index)
            for index in range(identity_v2.POOL_SIZE)
            if index not in missing
        }
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _desired(index: int) -> dict[str, Any]:
        account_id = identity_v2.ACCOUNT_IDS[index]
        email = identity_v2.ACCOUNT_EMAILS[index]
        return {
            "name": identity_v2.account_name(email),
            "projectId": identity_v2.PROJECT,
            "uniqueId": str(100_000_000_000_000_000_000 + index),
            "email": email,
            "displayName": f"OFC full100 worker {index:02d}",
            "description": "Fixed unprivileged OFC full100 worker identity",
            "disabled": False,
            "accountId": account_id,
        }

    @staticmethod
    def _account_url(email: str) -> str:
        return (
            "https://iam.googleapis.com/v1/projects/"
            f"{identity_v2.PROJECT}/serviceAccounts/"
            f"{urllib.parse.quote(email, safe='')}"
        )

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        assert headers["Authorization"] == f"Bearer {self.TOKEN}"
        assert timeout_seconds == 60
        self.calls.append((method, url))
        get_urls = {
            self._account_url(email): identity_v2.ACCOUNT_IDS[index]
            for index, email in enumerate(identity_v2.ACCOUNT_EMAILS)
        }
        if method == "GET" and url in get_urls:
            account = self.accounts.get(get_urls[url])
            return HttpResponse(
                404 if account is None else 200,
                b"{}" if account is None else json.dumps(account).encode("utf-8"),
                {},
            )
        accounts_url = (
            "https://iam.googleapis.com/v1/projects/"
            f"{identity_v2.PROJECT}/serviceAccounts"
        )
        if method == "POST" and url == accounts_url:
            payload = json.loads(body or b"")
            account_id = payload["accountId"]
            index = identity_v2.ACCOUNT_IDS.index(account_id)
            assert account_id not in self.accounts
            self.accounts[account_id] = self._desired(index)
            return HttpResponse(
                201, json.dumps(self.accounts[account_id]).encode("utf-8"), {}
            )
        raise AssertionError(f"unexpected fake IAM request: {method} {url}")


def test_request_envelope_builds_official_identity_adapter_with_fake_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(tmp_path)
    prepare = _prepare_event(controller)
    identity_plan = identity_v2.build_worker_identity_plan(
        wave_plan=controller.wave_plan,
        attempt_ledger=controller.attempt_ledger,
        resume_plan=controller.resume_plan,
        wave_index=0,
    )
    setup_plan = identity_v2.build_create_only_setup_plan(
        setup_nonce="controller-fake-setup-20260722-001",
        issued_at_utc="2026-07-22T04:55:00Z",
    )
    provider = FakeIdentityHttp(missing={2, 6})
    monkeypatch.setenv(identity_gcp_v2.TOKEN_ENV, provider.TOKEN)
    read_adapter = identity_gcp_v2.WorkerIdentityGcpAdapterV2(
        mode="read",
        wave_plan=controller.wave_plan,
        attempt_ledger=controller.attempt_ledger,
        resume_plan=controller.resume_plan,
        identity_plan=identity_plan,
        setup_plan=setup_plan,
        requester=provider,
    )
    read = read_adapter.read_pool(
        observed_at_utc="2026-07-22T04:58:00Z",
        expires_at_utc="2026-07-22T05:03:00Z",
    )
    request = {
        "operation_key": "setup-envelope-01",
        "prepare_event_sha256": prepare.value["event_sha256"],
        "identity_plan": identity_plan,
        "setup_plan": setup_plan,
        "read_receipt": read,
        "current_time_utc": NOW,
        "completed_at_utc": LATER,
    }
    event = subject.execute_mode_request(
        controller=controller,
        mode="setup-identities",
        request=request,
        requester=provider,
        allow_identity_create=True,
    )
    receipt = event.value["output"]["identity_create_receipt"]
    assert receipt["created_account_ids"] == [
        identity_v2.ACCOUNT_IDS[2], identity_v2.ACCOUNT_IDS[6]
    ]
    assert event.value["mutation_outcome"] == "performed"
    assert provider.TOKEN not in json.dumps(controller.inspect())
    before = len(provider.calls)
    assert subject.execute_mode_request(
        controller=controller,
        mode="setup-identities",
        request=request,
        requester=provider,
        allow_identity_create=True,
    ).value == event.value
    assert len(provider.calls) == before

    with pytest.raises(ValueError, match="fields changed"):
        subject.execute_mode_request(
            controller=controller,
            mode="setup-identities",
            request={**request, "token": provider.TOKEN},
            requester=provider,
            allow_identity_create=True,
        )


@pytest.mark.parametrize(
    ("source_phase", "reconcile_mode", "producer_name", "extra"),
    [
        (
            "worker-iam-install",
            "worker-iam-reconcile-install",
            "reconcile_install_worker_iam",
            {},
        ),
        (
            "worker-iam-cleanup",
            "worker-iam-reconcile-cleanup",
            "reconcile_cleanup_worker_iam",
            {"install_receipt": {}, "readback_receipt": {}},
        ),
    ],
)
def test_worker_iam_transport_ambiguity_has_get_only_controller_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_phase: str,
    reconcile_mode: str,
    producer_name: str,
    extra: Mapping[str, Any],
) -> None:
    controller = _controller(tmp_path)
    predecessor = _prepare_event(controller)
    source_operation = f"{source_phase}-transport-loss"

    def ambiguous_action() -> Mapping[str, Any]:
        raise gcp_v2.GcpPhaseATransportError("PUT response was lost")

    with pytest.raises(gcp_v2.GcpPhaseATransportError):
        controller.run_mutation_phase(
            phase=source_phase,
            operation_key=source_operation,
            predecessor_event_sha256=predecessor.value["event_sha256"],
            evidence={},
            action=ambiguous_action,
            validator=lambda value: value,
            started_at_utc=NOW,
            completed_at_utc=LATER,
        )
    source_failure = controller.journal.load()[-1]
    assert source_failure.value["output"]["failure_kind"] == "transport_ambiguity"

    recovered = _receipt("worker-iam-recovered", cleanup_complete=True)
    monkeypatch.setattr(subject, "_phase_a_adapter", lambda **kwargs: object())
    monkeypatch.setattr(
        worker_iam_v2,
        producer_name,
        lambda **kwargs: recovered,
    )
    request = {
        "operation_key": f"{reconcile_mode}-01",
        "predecessor_event_sha256": source_failure.value["event_sha256"],
        "source_operation_key": source_operation,
        "content_binding": {
            "immutable_content_prefix": "immutable/test",
            "content_payload_sha256": "1" * 64,
            "outer_manifest_sha256": "2" * 64,
        },
        "iam_plan": {},
        "prepare_receipt": {},
        "observed_at_utc": LATER,
        **extra,
    }
    event = subject.execute_mode_request(
        controller=controller,
        mode=reconcile_mode,
        request=request,
        allow_cloud_read=True,
    )
    assert event.value["mode"] == "read-only"
    assert event.value["predecessor_event_sha256"] == source_failure.value[
        "event_sha256"
    ]
    assert event.value["output"]["receipt"] == recovered


def test_worker_iam_explicit_failure_cannot_use_ambiguity_recovery(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    predecessor = _prepare_event(controller)

    def explicit_failure() -> Mapping[str, Any]:
        raise worker_iam_v2.WorkerIamCasError("provider returned an explicit 412")

    with pytest.raises(worker_iam_v2.WorkerIamCasError):
        controller.run_mutation_phase(
            phase="worker-iam-install",
            operation_key="iam-explicit-cas",
            predecessor_event_sha256=predecessor.value["event_sha256"],
            evidence={},
            action=explicit_failure,
            validator=lambda value: value,
            started_at_utc=NOW,
            completed_at_utc=LATER,
        )
    assert controller.journal.load()[-1].value["output"]["failure_kind"] == (
        "provider_rejected"
    )
    with pytest.raises(PermissionError, match="authorized ambiguous"):
        controller.record_ambiguous_mutation_reconciliation(
            phase="worker-iam-reconcile-install",
            operation_key="iam-explicit-cas-reconcile",
            source_operation_key="iam-explicit-cas",
            receipt=_receipt("forbidden-recovery"),
            output_key="receipt",
            recorded_at_utc=LATER,
        )


def test_worker_iam_cleanup_local_response_failure_allows_get_only_proof(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    predecessor = _prepare_event(controller)
    operation = "iam-cleanup-provider-normalization"

    def local_response_failure() -> Mapping[str, Any]:
        raise RuntimeError("provider canonicalized an exact cleanup response")

    with pytest.raises(RuntimeError, match="canonicalized"):
        controller.run_mutation_phase(
            phase="worker-iam-cleanup",
            operation_key=operation,
            predecessor_event_sha256=predecessor.value["event_sha256"],
            evidence={},
            action=local_response_failure,
            validator=lambda value: value,
            started_at_utc=NOW,
            completed_at_utc=LATER,
        )
    failure = controller.journal.load()[-1]
    assert failure.value["output"]["failure_kind"] == (
        "local_or_response_failure"
    )

    proof = _receipt(
        "exact-cleanup-state-get-only",
        removed_binding_count=0,
        set_attempt_count=0,
        cloud_mutation_performed=False,
        recovered_after_outcome_ambiguity=True,
        source_mutation_outcome="unknown",
    )
    event = controller.record_ambiguous_mutation_reconciliation(
        phase="worker-iam-reconcile-cleanup",
        operation_key="iam-cleanup-provider-normalization-reconcile",
        source_operation_key=operation,
        receipt=proof,
        output_key="receipt",
        recorded_at_utc=LATER,
    )

    assert event.value["mode"] == "read-only"
    assert event.value["mutation_requested"] is False
    assert event.value["mutation_outcome"] == "not_requested"
    assert event.value["predecessor_event_sha256"] == failure.value[
        "event_sha256"
    ]
    assert event.value["evidence"]["source_failure_kind"] == (
        "local_or_response_failure"
    )
    assert event.value["output"]["receipt"] == proof


def test_worker_iam_legacy_unknown_install_has_get_only_recovery(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    predecessor = _prepare_event(controller)
    operation = "iam-legacy-post-put-validation"
    intent, terminal = controller._begin_mutation(
        phase="worker-iam-install",
        operation_key=operation,
        predecessor_event_sha256=predecessor.value["event_sha256"],
        evidence={},
        recorded_at_utc=NOW,
    )
    assert intent is not None and terminal is None
    failure = controller._finish_mutation(
        intent=intent,
        status="failed",
        output={"failure_kind": "provider_rejected_or_local_failure"},
        outcome="unknown",
        recorded_at_utc=LATER,
    )

    event = controller.record_ambiguous_mutation_reconciliation(
        phase="worker-iam-reconcile-install",
        operation_key="iam-legacy-post-put-validation-reconcile",
        source_operation_key=operation,
        receipt=_receipt("exact-get-only-recovery"),
        output_key="receipt",
        recorded_at_utc=LATER,
    )

    assert event.value["mode"] == "read-only"
    assert event.value["predecessor_event_sha256"] == failure.value["event_sha256"]
    assert event.value["evidence"]["source_failure_kind"] == (
        "provider_rejected_or_local_failure"
    )
