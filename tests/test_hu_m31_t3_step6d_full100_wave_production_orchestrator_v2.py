from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_production_orchestrator_v2 as subject,
)


RUN = "regular-hu-m31-c02-f100wv2-20260723-orchestrator-test"
STARTUP = Path("scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh")


def _context() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = wave_v2.build_wave_plan(
        run_name=RUN,
        identity_salt="0123456789abcdef0123456789abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-solver-485418",
        zone="asia-northeast1-b",
        observed_at_utc="2026-07-23T01:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return plan, ledger, resume


def _patch_package(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer = {
        "manifest_sha256": "4" * 64,
        "content_payload_sha256": "5" * 64,
        "content_prefix": "immutable/full100/" + "5" * 64,
        "run_name": plan["run_name"],
    }
    stage = {
        "plan_sha256": "6" * 64,
        "content_payload_sha256": outer["content_payload_sha256"],
        "content_prefix": outer["content_prefix"],
    }
    monkeypatch.setattr(
        subject.package_v2,
        "validate_outer_package",
        lambda *args, **kwargs: dict(outer),
    )
    monkeypatch.setattr(
        subject.content_v2,
        "build_content_stage_plan",
        lambda *args, **kwargs: dict(stage),
    )
    return outer, stage


def _clock() -> datetime:
    return datetime(2026, 7, 23, 1, 0, 1, tzinfo=timezone.utc)


def test_plan_is_generic_write_once_and_performs_no_cloud_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume = _context()
    _patch_package(monkeypatch, plan)
    kwargs = {
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        "package_dir": tmp_path / "package",
        "control_root": tmp_path / "control",
        "startup_script_path": STARTUP,
        "clock": _clock,
    }
    first = subject.prepare_execution(**kwargs)
    second = subject.prepare_execution(**kwargs)
    assert first.orchestration_plan == second.orchestration_plan
    assert first.namespace.startswith("execution-000-")
    assert first.orchestration_plan["run_name"] == RUN
    assert first.orchestration_plan["dry_run_cloud_mutation_performed"] is False
    assert first.orchestration_plan["cloud_started"] is False
    assert first.orchestration_plan["credentials_from_environment_only"] is True
    assert first.orchestration_plan["current_profile_changed"] is False
    assert [row["mode"] for row in first.orchestration_plan["steps"]][2] == (
        "runtime-gcp-read"
    )
    assert (first.root / "orchestration_plan.json").is_file()
    assert (first.root / "inputs" / "attempt_ledger.json").is_file()
    assert (first.root / "static" / "content_stage_plan.json").is_file()
    assert not first.journal_dir.exists()
    material = json.loads(
        (first.root / "private" / "random_material.json").read_text(
            encoding="utf-8"
        )
    )
    assert uuid.UUID(material["claim_nonce"]).version == 4
    assert set(material["gce_create_request_ids"]) == {
        row["instance_id"] for row in resume["selected_attempts"]
    }
    all_ids = [
        item
        for field in (
            "gce_create_request_ids",
            "gce_delete_request_ids",
            "orphan_disk_delete_request_ids",
        )
        for item in material[field].values()
    ]
    assert len(all_ids) == len(set(all_ids))


def test_write_once_random_material_conflict_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume = _context()
    _patch_package(monkeypatch, plan)
    prepared = subject.prepare_execution(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        package_dir=tmp_path / "package",
        control_root=tmp_path / "control",
        startup_script_path=STARTUP,
        clock=_clock,
    )
    path = prepared.root / "private" / "random_material.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["claim_nonce"] = str(uuid.uuid4())
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        subject.prepare_execution(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            package_dir=tmp_path / "package",
            control_root=tmp_path / "control",
            startup_script_path=STARTUP,
            clock=_clock,
        )


def test_execute_gate_rejects_before_controller_or_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume = _context()
    _patch_package(monkeypatch, plan)
    prepared = subject.prepare_execution(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        package_dir=tmp_path / "package",
        control_root=tmp_path / "control",
        startup_script_path=STARTUP,
        clock=_clock,
    )
    calls: list[str] = []

    def dispatcher(**kwargs: Any) -> Any:
        calls.append(kwargs["mode"])
        raise AssertionError("dispatcher must not run")

    common = {
        "prepared": prepared,
        "allow_cloud_read": True,
        "allow_identity_create": True,
        "allow_content_stage": True,
        "allow_claim_create": True,
        "allow_worker_iam_install": True,
        "allow_launch_authorization": True,
        "allow_gce_create": True,
        "dispatcher": dispatcher,
        "clock": _clock,
    }
    with pytest.raises(PermissionError, match="exactly match"):
        subject.execute_production_wave(
            **common, confirm_run_name="wrong-run"
        )
    with pytest.raises(PermissionError, match="allow flags"):
        subject.execute_production_wave(
            **{
                **common,
                "allow_claim_create": False,
                "confirm_run_name": RUN,
            }
        )
    assert calls == []
    assert not prepared.journal_dir.exists()


def test_cli_plan_is_a_real_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan, ledger, resume = _context()
    _patch_package(monkeypatch, plan)
    paths = {}
    for name, value in (
        ("wave", plan),
        ("ledger", ledger),
        ("resume", resume),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    assert subject.main(
        [
            "--mode",
            "plan",
            "--wave-plan",
            str(paths["wave"]),
            "--attempt-ledger",
            str(paths["ledger"]),
            "--resume-plan",
            str(paths["resume"]),
            "--package-dir",
            str(tmp_path / "package"),
            "--control-root",
            str(tmp_path / "control"),
            "--startup-script",
            str(STARTUP),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == subject.ORCHESTRATION_PLAN_SCHEMA
    assert output["dry_run_cloud_mutation_performed"] is False
    assert output["cloud_started"] is False


def _receipt(label: str, **values: Any) -> dict[str, Any]:
    core = {"kind": label, **values, "current_profile_changed": False}
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 23, 1, 0, 10, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        from datetime import timedelta

        self.value += timedelta(seconds=seconds)

    def render(self, seconds: int = 0) -> str:
        from datetime import timedelta

        return (self.value + timedelta(seconds=seconds)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )


class FakeDispatcher:
    def __init__(
        self,
        *,
        clock: MutableClock,
        initial_identities_exist: bool = True,
        expire_after_iam_readback: bool = False,
        identity_setup_partial_once: bool = False,
        iam_install_ambiguous_once: bool = False,
        launch_response_loss_once: bool = False,
        stage_partial_once: bool = False,
        short_freshness: bool = False,
        phasea_failure_once: bool = False,
        provider_failure_once: bool = False,
        advance_after_phasea_seconds: int = 0,
        advance_after_worker_iam_plan_seconds: int = 0,
        advance_after_iam_readback_seconds: int = 0,
        initial_worker_iam_lifetime_seconds: int = 5_400,
    ) -> None:
        self.clock = clock
        self.initial_identities_exist = initial_identities_exist
        self.expire_after_iam_readback = expire_after_iam_readback
        self.identity_setup_partial_once = identity_setup_partial_once
        self.iam_install_ambiguous_once = iam_install_ambiguous_once
        self.launch_response_loss_once = launch_response_loss_once
        self.stage_partial_once = stage_partial_once
        self.short_freshness = short_freshness
        self.phasea_failure_once = phasea_failure_once
        self.provider_failure_once = provider_failure_once
        self.advance_after_phasea_seconds = advance_after_phasea_seconds
        self.advance_after_worker_iam_plan_seconds = (
            advance_after_worker_iam_plan_seconds
        )
        self.advance_after_iam_readback_seconds = (
            advance_after_iam_readback_seconds
        )
        self.initial_worker_iam_lifetime_seconds = (
            initial_worker_iam_lifetime_seconds
        )
        self.recovery_identity_reads_remaining = (
            1 if identity_setup_partial_once else 0
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.events_by_operation: dict[str, dict[str, Any]] = {}

    def __call__(self, **kwargs: Any) -> Any:
        mode = kwargs["mode"]
        request = json.loads(json.dumps(kwargs["request"]))
        operation = request["operation_key"]
        cached = self.events_by_operation.get(operation)
        if cached is not None:
            return type("CachedFakeEvent", (), {"value": cached})()
        self.calls.append((mode, request))
        label = operation.split(":", 1)[1]
        output: dict[str, Any]
        event_status = "complete"
        if mode == "identity-read":
            initial_missing = (
                label == "identity-read-initial"
                and not self.initial_identities_exist
            )
            recovery_missing = (
                label.startswith("identity-read-recovery-")
                and self.recovery_identity_reads_remaining > 0
            )
            if recovery_missing:
                self.recovery_identity_reads_remaining -= 1
            missing = initial_missing or recovery_missing
            inventory = _receipt(
                "identity-inventory",
                observed_at_utc=request["observed_at_utc"],
            )
            output = {
                "identity_read_receipt": _receipt(
                    "identity-read",
                    all_accounts_exist=not missing,
                    missing_account_ids=(
                        ["ofc-f100-worker-00"] if missing else []
                    ),
                    inventory_receipt=None if missing else inventory,
                )
            }
        elif mode == "content-prefix-preflight":
            output = {
                "content_preflight_receipt": _receipt("content-preflight")
            }
        elif mode == "runtime-gcp-read":
            runtime = _receipt(
                "runtime-preflight",
                issued_at_utc=request["current_utc"],
                expires_at_utc=self.clock.render(
                    10 if self.short_freshness else 300
                ),
            )
            output = {
                "runtime_gcp_read_receipt": _receipt(
                    "runtime-gcp-read",
                    runtime_preflight_receipt=runtime,
                ),
                "runtime_preflight_receipt": runtime,
            }
        elif mode == "prepare":
            output = _receipt("prepare")
        elif mode == "setup-identities":
            if self.identity_setup_partial_once:
                self.identity_setup_partial_once = False
                event_status = "partial"
                output = {
                    "identity_create_receipt": None,
                    "recovery_read_receipt": _receipt(
                        "identity-recovery-read"
                    ),
                }
            else:
                output = {
                    "identity_create_receipt": _receipt("identity-create")
                }
        elif mode in {"stage-content", "bind-existing-staged-content"}:
            if self.stage_partial_once:
                self.stage_partial_once = False
                event_status = "partial"
                stage_complete = False
                created_entry_count = 25
            else:
                stage_complete = True
                created_entry_count = 26
            output = {
                "stage_receipt": _receipt(
                    "stage",
                    stage_complete=stage_complete,
                    created_entry_count=created_entry_count,
                    content_payload_sha256="5" * 64,
                )
            }
            if mode == "bind-existing-staged-content":
                output["content_reused_without_mutation"] = True
        elif mode == "identity-actas":
            output = {
                "worker_identity_act_as_receipt": _receipt(
                    "identity-actas", tested_at_utc=request["tested_at_utc"]
                )
            }
        elif mode == "project-iam-scan":
            output = {
                "project_iam_scan_receipt": _receipt(
                    "project-scan", observed_at_utc=request["observed_at_utc"]
                )
            }
        elif mode == "worker-iam-plan":
            lifetime = (
                self.initial_worker_iam_lifetime_seconds
                if label == "worker-iam-plan"
                else 5_400
            )
            output = {
                "worker_iam_plan": _receipt(
                    "worker-iam-plan",
                    plan_sha256=subject.canonical_sha256(
                        {"operation_label": label}
                    ),
                    issued_at_utc=self.clock.render(),
                    expires_at_utc=self.clock.render(lifetime),
                )
            }
            if self.advance_after_worker_iam_plan_seconds:
                self.clock.advance(self.advance_after_worker_iam_plan_seconds)
                self.advance_after_worker_iam_plan_seconds = 0
        elif mode == "phasea-read":
            if self.phasea_failure_once:
                self.phasea_failure_once = False
                raise RuntimeError("simulated phase A read failure")
            output = {
                "gcp_read_receipt": _receipt(
                    "gcp-read",
                    observed_at_utc=request["observed_at_utc"],
                    quota_receipt=_receipt(
                        "quota",
                        observed_at_utc=request["observed_at_utc"],
                        expires_at_utc=request["expires_at_utc"],
                    ),
                    planned_mapping_receipt=_receipt(
                        "mapping",
                        observed_at_utc=request["observed_at_utc"],
                        expires_at_utc=request["expires_at_utc"],
                    ),
                )
            }
            if self.advance_after_phasea_seconds:
                self.clock.advance(self.advance_after_phasea_seconds)
        elif mode == "provider-actas-check":
            if self.provider_failure_once:
                self.provider_failure_once = False
                raise RuntimeError("simulated provider actAs failure")
            output = {
                "service_account_actas_receipt": _receipt(
                    "provider-actas",
                    checked_at_utc=request["checked_at_utc"],
                    expires_at_utc=request["expires_at_utc"],
                )
            }
        elif mode == "persistent-claim":
            output = {"receipt": _receipt("persistent-claim")}
        elif mode == "worker-iam-prepare":
            output = {
                "worker_iam_prepare_receipt": _receipt("worker-iam-prepare")
            }
        elif mode == "worker-iam-install":
            if self.iam_install_ambiguous_once:
                self.iam_install_ambiguous_once = False
                event_status = "failed"
                output = {"failure_kind": "transport_ambiguity"}
            else:
                output = {"receipt": _receipt("worker-iam-install")}
        elif mode == "worker-iam-reconcile-install":
            output = {"receipt": _receipt("worker-iam-install-reconciled")}
        elif mode == "worker-iam-readback":
            output = {
                "worker_iam_readback_receipt": _receipt("worker-iam-readback")
            }
            if self.expire_after_iam_readback:
                self.clock.advance(290)
                self.expire_after_iam_readback = False
            if self.advance_after_iam_readback_seconds:
                self.clock.advance(self.advance_after_iam_readback_seconds)
                self.advance_after_iam_readback_seconds = 0
        elif mode == "prelaunch-authorization":
            output = {
                "prelaunch_authorization": _receipt(
                    "prelaunch-authorization",
                    expires_at_utc=request["expires_at_utc"],
                )
            }
        elif mode == "launch-bundle-build":
            output = {
                "launch_bundle": {
                    "bundle_sha256": subject.canonical_sha256(request),
                    "immutable_content_sha256": "5" * 64,
                    "worker_iam_plan_sha256": request["launch_validation"][
                        "worker_iam_plan"
                    ]["plan_sha256"],
                }
            }
        elif mode == "authorize-launch":
            if self.launch_response_loss_once:
                self.launch_response_loss_once = False
                event_status = "failed"
                output = {"failure_kind": "transport_ambiguity"}
            else:
                output = {
                    "gce_create_receipt": _receipt(
                        "gce-create", create_complete=True, rows=[]
                    ),
                    "actual_launch_receipt": _receipt("actual-launch"),
                }
        elif mode == "reconcile-create":
            output = {
                "gce_create_receipt": _receipt(
                    "gce-create-reconciled", create_complete=False, rows=[]
                )
            }
        else:  # pragma: no cover - makes new unplanned modes fail loudly.
            raise AssertionError(f"unexpected orchestrator mode {mode}")
        mutating = mode in {
            "setup-identities",
            "stage-content",
            "persistent-claim",
            "worker-iam-install",
            "authorize-launch",
        }
        if not mutating:
            mutation_outcome = "not_requested"
        elif event_status == "complete":
            mutation_outcome = "performed"
        elif event_status == "partial":
            mutation_outcome = "partial"
        else:
            mutation_outcome = "unknown"
        event = kwargs["controller"].journal.append(
            phase=(
                "stage-content"
                if mode == "bind-existing-staged-content"
                else mode
            ),
            mode="mutation" if mutating else "read-only",
            status=event_status,
            operation_key=operation,
            predecessor_event_sha256=(
                request.get("predecessor_event_sha256")
                or request.get("prepare_event_sha256")
                or request.get("stage_event_sha256")
            ),
            evidence={},
            output=output,
            mutation_requested=mutating,
            mutation_outcome=mutation_outcome,
            recorded_at_utc=self.clock.render(),
        ).value
        self.events_by_operation[operation] = event
        return type("FakeEvent", (), {"value": event})()


class CrashPersistingDispatcher(FakeDispatcher):
    """Persist one controller mutation terminal, then simulate process loss."""

    def __init__(self, *, clock: MutableClock, crash_mode: str, **kwargs: Any) -> None:
        super().__init__(clock=clock, **kwargs)
        self.crash_mode = crash_mode
        self.crashed = False

    def __call__(self, **kwargs: Any) -> Any:
        mode = kwargs["mode"]
        request = kwargs["request"]
        controller = kwargs["controller"]
        if mode == self.crash_mode:
            terminals = [
                event
                for event in controller.journal.load()
                if event.value["operation_key"] == request["operation_key"]
                and event.value["status"] in {"partial", "failed", "complete"}
            ]
            if terminals:
                return type("RecoveredEvent", (), {"value": terminals[0].value})()
            if not self.crashed:
                self.crashed = True
                predecessor = (
                    request.get("predecessor_event_sha256")
                    or request.get("prepare_event_sha256")
                    or request.get("stage_event_sha256")
                )
                intent = controller.journal.append(
                    phase=mode,
                    mode="mutation",
                    status="intent",
                    operation_key=request["operation_key"],
                    predecessor_event_sha256=predecessor,
                    evidence={},
                    output=None,
                    mutation_requested=True,
                    mutation_outcome="pending",
                    recorded_at_utc=self.clock.render(),
                )
                if mode == "setup-identities":
                    status = "partial"
                    outcome = "partial"
                    output = {
                        "identity_create_receipt": None,
                        "recovery_read_receipt": _receipt(
                            "persisted-identity-recovery"
                        ),
                    }
                else:
                    status = "failed"
                    outcome = "unknown"
                    output = {"failure_kind": "transport_ambiguity"}
                controller.journal.append(
                    phase=mode,
                    mode="mutation",
                    status=status,
                    operation_key=request["operation_key"],
                    predecessor_event_sha256=intent.value["event_sha256"],
                    evidence={"intent_event_sha256": intent.value["event_sha256"]},
                    output=output,
                    mutation_requested=True,
                    mutation_outcome=outcome,
                    recorded_at_utc=self.clock.render(),
                )
                raise RuntimeError(f"simulated process loss after {mode}")
        return super().__call__(**kwargs)


class CrashAfterCompleteDispatcher(FakeDispatcher):
    """Complete one controller operation, then lose the local response."""

    def __init__(self, *, clock: MutableClock, crash_mode: str, **kwargs: Any) -> None:
        super().__init__(clock=clock, **kwargs)
        self.crash_mode = crash_mode
        self.crashed = False

    def __call__(self, **kwargs: Any) -> Any:
        event = super().__call__(**kwargs)
        if kwargs["mode"] == self.crash_mode and not self.crashed:
            self.crashed = True
            raise RuntimeError(f"simulated response loss after {self.crash_mode}")
        return event


def _prepared_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: MutableClock
) -> subject.PreparedExecution:
    plan, ledger, resume = _context()
    _patch_package(monkeypatch, plan)
    return subject.prepare_execution(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        package_dir=tmp_path / "package",
        control_root=tmp_path / "control",
        startup_script_path=STARTUP,
        clock=clock,
    )


def _execute_kwargs(
    prepared: subject.PreparedExecution,
    dispatcher: FakeDispatcher,
    clock: MutableClock,
) -> dict[str, Any]:
    return {
        "prepared": prepared,
        "allow_cloud_read": True,
        "allow_identity_create": True,
        "allow_content_stage": True,
        "allow_claim_create": True,
        "allow_worker_iam_install": True,
        "allow_launch_authorization": True,
        "allow_gce_create": True,
        "confirm_run_name": RUN,
        "dispatcher": dispatcher,
        "clock": clock,
    }


def _persisted_request(
    prepared: subject.PreparedExecution, label: str
) -> dict[str, Any]:
    envelope = json.loads(
        (prepared.root / "steps" / label / "request.json").read_text(
            encoding="utf-8"
        )
    )
    return envelope["request"]


def test_execute_composes_full_chain_and_emits_receiver_cleanup_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock)
    manifest = subject.execute_production_wave(
        **_execute_kwargs(prepared, fake, clock)
    )
    modes = [mode for mode, _ in fake.calls]
    assert modes == [
        "identity-read",
        "content-prefix-preflight",
        "runtime-gcp-read",
        "prepare",
        "stage-content",
        "runtime-gcp-read",
        "identity-read",
        "identity-actas",
        "project-iam-scan",
        "worker-iam-plan",
        "phasea-read",
        "provider-actas-check",
        "persistent-claim",
        "worker-iam-prepare",
        "worker-iam-install",
        "worker-iam-readback",
        "prelaunch-authorization",
        "launch-bundle-build",
        "authorize-launch",
    ]
    assert manifest["schema"] == subject.EXECUTION_MANIFEST_SCHEMA
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    assert manifest["receiver_request_ready_after_lifecycle_closeout"] is True
    assert manifest["cleanup_request_ids_persisted"] is True
    assert manifest["credentials_from_environment_only"] is True
    assert manifest["additional_create_authorized"] is False
    assert manifest["current_profile_changed"] is False
    assert (prepared.root / "execution_manifest.json").is_file()
    assert len(list((prepared.root / "steps").glob("*/request.json"))) == len(
        modes
    )

    calls = len(fake.calls)
    again = subject.execute_production_wave(
        **_execute_kwargs(prepared, fake, clock)
    )
    assert again == manifest
    assert len(fake.calls) == calls


def test_valid_single_job_resume_uses_same_production_chain_and_exact_request_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This helper derives the one-job state from a genuine eight-job wave by
    # accepting the other seven; it is not a hand-trimmed resume payload.
    from test_hu_m31_t3_step6d_full100_wave_launch_bundle_v2 import (
        _single_job_context,
    )

    plan, ledger, resume = _single_job_context()
    _patch_package(monkeypatch, plan)
    clock = MutableClock()
    prepared = subject.prepare_execution(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        package_dir=tmp_path / "package",
        control_root=tmp_path / "control",
        startup_script_path=STARTUP,
        clock=clock,
    )
    fake = FakeDispatcher(clock=clock)
    manifest = subject.execute_production_wave(
        prepared=prepared,
        allow_cloud_read=True,
        allow_identity_create=True,
        allow_content_stage=True,
        allow_claim_create=True,
        allow_worker_iam_install=True,
        allow_launch_authorization=True,
        allow_gce_create=True,
        confirm_run_name=plan["run_name"],
        dispatcher=fake,
        clock=clock,
    )

    selected = resume["selected_attempts"][0]
    assert len(plan["waves"][0]["job_ids"]) == 8
    assert prepared.identity_plan["selected_count"] == 1
    assert prepared.identity_plan["selected_workers"][0]["job_id"] == selected[
        "job_id"
    ]
    for field in (
        "gce_create_request_ids",
        "gce_delete_request_ids",
        "orphan_disk_delete_request_ids",
    ):
        assert list(prepared.random_material[field]) == [selected["instance_id"]]
        assert list(manifest[field]) == [selected["instance_id"]]
    assert [mode for mode, _ in fake.calls].count("authorize-launch") == 1
    assert manifest["wave_index"] == 0
    assert manifest["additional_create_authorized"] is False


def test_missing_identity_uses_create_only_after_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, initial_identities_exist=False)
    subject.execute_production_wave(**_execute_kwargs(prepared, fake, clock))
    modes = [mode for mode, _ in fake.calls]
    assert modes.index("prepare") < modes.index("setup-identities")
    assert modes.index("setup-identities") < modes.index("stage-content")
    assert modes.count("setup-identities") == 1


def test_expired_five_minute_reads_refresh_without_replaying_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, expire_after_iam_readback=True)
    subject.execute_production_wave(**_execute_kwargs(prepared, fake, clock))
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("runtime-gcp-read") == 3  # bootstrap, r00, r01
    assert modes.count("phasea-read") == 2
    assert modes.count("persistent-claim") == 1
    assert modes.count("worker-iam-install") == 1
    assert modes.count("authorize-launch") == 1
    launch_index = modes.index("authorize-launch")
    second_phasea = [i for i, mode in enumerate(modes) if mode == "phasea-read"][1]
    assert second_phasea < launch_index


def test_provider_actas_expiry_is_clamped_to_phasea_read_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(
        clock=clock,
        advance_after_phasea_seconds=1,
    )
    subject.execute_production_wave(**_execute_kwargs(prepared, fake, clock))
    phasea = next(request for mode, request in fake.calls if mode == "phasea-read")
    provider = next(
        request for mode, request in fake.calls if mode == "provider-actas-check"
    )
    assert provider["checked_at_utc"] > phasea["observed_at_utc"]
    assert provider["expires_at_utc"] == phasea["expires_at_utc"]


def test_restart_skips_expired_unresolved_phasea_epoch_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, phasea_failure_once=True)
    kwargs = _execute_kwargs(prepared, fake, clock)

    with pytest.raises(RuntimeError, match="simulated phase A read failure"):
        subject.execute_production_wave(**kwargs)
    first_modes = [mode for mode, _ in fake.calls]
    assert "persistent-claim" not in first_modes
    assert "worker-iam-install" not in first_modes
    assert "authorize-launch" not in first_modes

    clock.advance(subject.FRESHNESS_SECONDS + 1)
    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    labels = [
        request["operation_key"].split(":", 1)[1]
        for _, request in fake.calls
    ]
    assert labels.count("phasea-read-r00") == 2
    assert labels.count("provider-actas-r00") == 0
    assert labels.count("phasea-read-r01") == 1
    assert labels.count("provider-actas-r01") == 1
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("persistent-claim") == 1
    assert modes.count("worker-iam-install") == 1
    assert modes.count("authorize-launch") == 1


def test_restart_quarantines_expired_unresolved_provider_request_and_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, provider_failure_once=True)
    kwargs = _execute_kwargs(prepared, fake, clock)

    with pytest.raises(RuntimeError, match="simulated provider actAs failure"):
        subject.execute_production_wave(**kwargs)
    provider_request = (
        prepared.root / "steps" / "provider-actas-r00" / "request.json"
    )
    assert provider_request.is_file()
    assert not (
        prepared.root / "steps" / "provider-actas-r00" / "event.json"
    ).exists()
    first_modes = [mode for mode, _ in fake.calls]
    assert not set(first_modes).intersection(
        {"persistent-claim", "worker-iam-install", "authorize-launch"}
    )

    clock.advance(subject.FRESHNESS_SECONDS + 1)
    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    labels = [
        request["operation_key"].split(":", 1)[1]
        for _, request in fake.calls
    ]
    assert labels.count("provider-actas-r00") == 1
    assert labels.count("runtime-gcp-fresh-r01") == 1
    assert labels.count("identity-read-fresh-r01") == 1
    assert labels.count("identity-actas-r01") == 1
    assert labels.count("project-iam-scan-r01") == 1
    assert labels.count("worker-iam-plan-r01") == 1
    assert labels.count("phasea-read-r01") == 1
    assert labels.count("provider-actas-r01") == 1
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("persistent-claim") == 1
    assert modes.count("worker-iam-install") == 1
    assert modes.count("authorize-launch") == 1


def test_preclaim_requires_entire_fresh_chain_not_only_phasea_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(
        clock=clock,
        advance_after_worker_iam_plan_seconds=subject.FRESHNESS_SECONDS + 1,
    )
    subject.execute_production_wave(**_execute_kwargs(prepared, fake, clock))
    labels = [
        request["operation_key"].split(":", 1)[1]
        for _, request in fake.calls
    ]
    assert labels.count("provider-actas-r00") == 1
    assert labels.count("runtime-gcp-fresh-r01") == 1
    assert labels.count("worker-iam-plan-r01") == 1
    assert labels.count("provider-actas-r01") == 1
    claim_index = labels.index("persistent-claim")
    assert labels.index("provider-actas-r01") < claim_index


def test_expired_unresolved_provider_request_is_validated_before_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, provider_failure_once=True)
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="simulated provider actAs failure"):
        subject.execute_production_wave(**kwargs)

    path = prepared.root / "steps" / "provider-actas-r00" / "request.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["request"]["content_binding"]["content_payload_sha256"] = "9" * 64
    envelope["request_sha256"] = subject.canonical_sha256(envelope["request"])
    envelope_core = dict(envelope)
    envelope_core.pop("envelope_sha256")
    envelope["envelope_sha256"] = subject.canonical_sha256(envelope_core)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    calls_before = len(fake.calls)
    clock.advance(subject.FRESHNESS_SECONDS + 1)

    with pytest.raises(
        ValueError, match="persisted provider actAs request binding changed"
    ):
        subject.execute_production_wave(**kwargs)
    assert len(fake.calls) == calls_before


def test_launch_requires_worker_iam_to_cover_watchdog_and_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(
        clock=clock,
        advance_after_iam_readback_seconds=(
            5_401 - subject.WORKER_IAM_MIN_REMAINING_SECONDS
        ),
    )
    with pytest.raises(
        subject.WorkerIamRefreshRequired, match="4200-second watchdog"
    ):
        subject.execute_production_wave(
            **_execute_kwargs(prepared, fake, clock)
        )
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("persistent-claim") == 1
    assert modes.count("worker-iam-install") == 1
    assert modes.count("authorize-launch") == 0
    calls_before_restart = len(fake.calls)
    with pytest.raises(
        subject.WorkerIamRefreshRequired, match="4200-second watchdog"
    ):
        subject.execute_production_wave(
            **_execute_kwargs(prepared, fake, clock)
        )
    assert len(fake.calls) == calls_before_restart


def test_preclaim_replaces_short_r00_worker_iam_with_fresh_epoch_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(
        clock=clock,
        initial_worker_iam_lifetime_seconds=(
            subject.WORKER_IAM_MIN_REMAINING_SECONDS - 1
        ),
    )
    manifest = subject.execute_production_wave(
        **_execute_kwargs(prepared, fake, clock)
    )
    requests = {
        request["operation_key"].split(":", 1)[1]: request
        for _, request in fake.calls
    }
    old_plan = next(
        event["output"]["worker_iam_plan"]
        for event in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in prepared.journal_dir.glob("*.json")
        )
        if event["operation_key"].endswith(":worker-iam-plan")
    )
    new_plan = manifest["worker_iam_plan"]
    assert old_plan["plan_sha256"] != new_plan["plan_sha256"]
    assert requests["phasea-read-r01"]["iam_plan"]["plan_sha256"] == (
        new_plan["plan_sha256"]
    )
    assert requests["provider-actas-r01"]["iam_plan"]["plan_sha256"] == (
        new_plan["plan_sha256"]
    )
    assert requests["worker-iam-prepare"]["iam_plan"]["plan_sha256"] == (
        new_plan["plan_sha256"]
    )
    assert requests["worker-iam-install"]["iam_plan"]["plan_sha256"] == (
        new_plan["plan_sha256"]
    )
    assert manifest["launch_validation"]["worker_iam_plan"][
        "plan_sha256"
    ] == new_plan["plan_sha256"]
    assert manifest["launch_bundle"]["worker_iam_plan_sha256"] == (
        new_plan["plan_sha256"]
    )


def test_worker_iam_exact_minimum_remaining_boundary_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(
        clock=clock,
        initial_worker_iam_lifetime_seconds=(
            subject.WORKER_IAM_MIN_REMAINING_SECONDS
        ),
    )
    manifest = subject.execute_production_wave(
        **_execute_kwargs(prepared, fake, clock)
    )
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    labels = [
        request["operation_key"].split(":", 1)[1]
        for _, request in fake.calls
    ]
    assert "worker-iam-plan-r01" not in labels


def test_claim_restart_refreshes_reads_but_freezes_worker_iam_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = CrashAfterCompleteDispatcher(
        clock=clock,
        crash_mode="persistent-claim",
    )
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="response loss after persistent-claim"):
        subject.execute_production_wave(**kwargs)
    assert [mode for mode, _ in fake.calls].count("persistent-claim") == 1

    clock.advance(subject.FRESHNESS_SECONDS + 1)
    manifest = subject.execute_production_wave(**kwargs)
    labels = [
        request["operation_key"].split(":", 1)[1]
        for _, request in fake.calls
    ]
    modes = [mode for mode, _ in fake.calls]
    assert "worker-iam-plan-r01" not in labels
    assert labels.count("runtime-gcp-fresh-r01") == 1
    assert labels.count("provider-actas-r01") == 1
    assert modes.count("persistent-claim") == 1
    assert modes.count("worker-iam-install") == 1
    assert modes.count("authorize-launch") == 1
    r00_plan = _persisted_request(prepared, "provider-actas-r00")["iam_plan"]
    r01_plan = _persisted_request(prepared, "provider-actas-r01")["iam_plan"]
    assert r01_plan["plan_sha256"] == r00_plan["plan_sha256"]
    assert manifest["worker_iam_plan"]["plan_sha256"] == r00_plan[
        "plan_sha256"
    ]


def test_stale_provider_event_without_request_fails_before_epoch_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock)
    kwargs = _execute_kwargs(prepared, fake, clock)
    original_dispatch = subject.ProductionOrchestratorV2._dispatch

    def stop_before_claim(self: Any, *, label: str, **call_kwargs: Any) -> Any:
        if label == "persistent-claim":
            raise RuntimeError("simulated stop before claim")
        return original_dispatch(self, label=label, **call_kwargs)

    monkeypatch.setattr(
        subject.ProductionOrchestratorV2,
        "_dispatch",
        stop_before_claim,
    )
    with pytest.raises(RuntimeError, match="stop before claim"):
        subject.execute_production_wave(**kwargs)
    provider_dir = prepared.root / "steps" / "provider-actas-r00"
    assert (provider_dir / "event.json").is_file()
    (provider_dir / "request.json").unlink()
    calls_before = len(fake.calls)
    clock.advance(subject.FRESHNESS_SECONDS + 1)
    monkeypatch.setattr(
        subject.ProductionOrchestratorV2,
        "_dispatch",
        original_dispatch,
    )

    with pytest.raises(
        ValueError, match="provider-actas-r00 event exists without its request"
    ):
        subject.execute_production_wave(**kwargs)
    assert len(fake.calls) == calls_before


def test_later_execution_binds_source_stage_without_content_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume = _context()
    _, stage_plan = _patch_package(monkeypatch, plan)
    preflight = _receipt("source-preflight")
    stage = _receipt(
        "source-stage",
        stage_complete=True,
        created_entry_count=26,
        content_payload_sha256="5" * 64,
    )
    monkeypatch.setattr(
        subject.content_v2,
        "validate_preflight_absence_receipt",
        lambda supplied_plan, value: dict(value),
    )
    monkeypatch.setattr(
        subject.content_v2,
        "validate_stage_receipt",
        lambda supplied_plan, supplied_preflight, value: dict(value),
    )
    source_core = {
        "schema": subject.SOURCE_CONTENT_SCHEMA,
        "content_preflight_receipt": preflight,
        "source_stage_receipt": stage,
        "current_profile_changed": False,
    }
    source = {
        **source_core,
        "handoff_sha256": subject.canonical_sha256(source_core),
    }
    clock = MutableClock()
    prepared = subject.prepare_execution(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        package_dir=tmp_path / "package",
        control_root=tmp_path / "control",
        startup_script_path=STARTUP,
        source_content_handoff=source,
        clock=clock,
    )
    assert prepared.stage_plan == stage_plan
    fake = FakeDispatcher(clock=clock)
    kwargs = {
        **_execute_kwargs(prepared, fake, clock),
        "allow_content_stage": False,
    }
    original = _crash_once_immediately_after_stage(monkeypatch)
    with pytest.raises(RuntimeError, match="after stage completion"):
        subject.execute_production_wave(**kwargs)
    calls_before = len(fake.calls)
    monkeypatch.setattr(subject.ProductionOrchestratorV2, "_fresh_chain", original)
    subject.execute_production_wave(**kwargs)
    modes = [mode for mode, _ in fake.calls]
    assert "content-prefix-preflight" not in modes
    assert "stage-content" not in modes
    assert modes.count("bind-existing-staged-content") == 1
    assert not any(
        mode == "bind-existing-staged-content"
        for mode, _ in fake.calls[calls_before:]
    )


def test_identity_partial_uses_fresh_read_then_new_create_missing_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(
        clock=clock,
        initial_identities_exist=False,
        identity_setup_partial_once=True,
    )
    subject.execute_production_wave(**_execute_kwargs(prepared, fake, clock))
    labels = [request["operation_key"].split(":", 1)[1] for _, request in fake.calls]
    assert labels.count("setup-identities") == 1
    assert labels.count("identity-read-recovery-01") == 1
    assert labels.count("setup-identities-recovery-01") == 1
    assert labels.index("setup-identities") < labels.index(
        "identity-read-recovery-01"
    ) < labels.index("setup-identities-recovery-01")


def test_worker_iam_transport_ambiguity_uses_get_only_reconcile_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, iam_install_ambiguous_once=True)
    manifest = subject.execute_production_wave(
        **_execute_kwargs(prepared, fake, clock)
    )
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("worker-iam-install") == 1
    assert modes.count("worker-iam-reconcile-install") == 1
    assert modes.index("worker-iam-install") < modes.index(
        "worker-iam-reconcile-install"
    ) < modes.index("worker-iam-readback")
    assert manifest["worker_iam_install_receipt"]["kind"] == (
        "worker-iam-install-reconciled"
    )


def test_gce_create_response_loss_uses_get_only_reconcile_create_no_repost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, launch_response_loss_once=True)
    manifest = subject.execute_production_wave(
        **_execute_kwargs(prepared, fake, clock)
    )
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("authorize-launch") == 1
    assert modes.count("reconcile-create") == 1
    assert modes.index("authorize-launch") < modes.index("reconcile-create")
    assert manifest["gce_create_receipt"]["kind"] == "gce-create-reconciled"
    assert manifest["actual_launch_receipt"] is None


def test_stage_partial_stops_before_fresh_reads_claim_iam_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, stage_partial_once=True)
    with pytest.raises(RuntimeError, match="staged content is not complete"):
        subject.execute_production_wave(**_execute_kwargs(prepared, fake, clock))
    modes = [mode for mode, _ in fake.calls]
    assert modes[-1] == "stage-content"
    assert "persistent-claim" not in modes
    assert "worker-iam-prepare" not in modes
    assert "worker-iam-install" not in modes
    assert "authorize-launch" not in modes
    assert not any(
        request["operation_key"].endswith("runtime-gcp-fresh-r00")
        for _, request in fake.calls
    )


def _crash_once_immediately_after_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    original = subject.ProductionOrchestratorV2._fresh_chain
    armed = True

    def crash(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal armed
        if armed:
            armed = False
            raise RuntimeError("simulated process loss after stage completion")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(subject.ProductionOrchestratorV2, "_fresh_chain", crash)
    return original


def test_restart_after_stage_replays_resolved_events_and_dispatches_only_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, initial_identities_exist=False)
    kwargs = _execute_kwargs(prepared, fake, clock)
    original = _crash_once_immediately_after_stage(monkeypatch)

    with pytest.raises(RuntimeError, match="after stage completion"):
        subject.execute_production_wave(**kwargs)
    before = list(fake.calls)
    before_modes = [mode for mode, _ in before]
    assert before_modes.count("content-prefix-preflight") == 1
    assert before_modes.count("setup-identities") == 1
    assert before_modes.count("stage-content") == 1

    monkeypatch.setattr(subject.ProductionOrchestratorV2, "_fresh_chain", original)
    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    after_modes = [mode for mode, _ in fake.calls]
    before_labels = [
        request["operation_key"].split(":", 1)[1] for _, request in before
    ]
    after_labels = [
        request["operation_key"].split(":", 1)[1]
        for _, request in fake.calls
    ]
    for resolved_label in (
        "identity-read-initial",
        "content-prefix-preflight",
        "runtime-gcp-bootstrap",
        "prepare",
        "setup-identities",
        "stage-content",
    ):
        assert after_labels.count(resolved_label) == before_labels.count(
            resolved_label
        )
    assert after_modes.count("runtime-gcp-read") == (
        before_modes.count("runtime-gcp-read") + 1
    )
    resumed_modes = [mode for mode, _ in fake.calls[len(before) :]]
    assert resumed_modes[0] == "runtime-gcp-read"
    assert not set(resumed_modes).intersection(
        {"content-prefix-preflight", "setup-identities", "stage-content"}
    )


def test_restart_rejects_validly_resealed_local_event_that_differs_from_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock)
    kwargs = _execute_kwargs(prepared, fake, clock)
    original = _crash_once_immediately_after_stage(monkeypatch)
    with pytest.raises(RuntimeError, match="after stage completion"):
        subject.execute_production_wave(**kwargs)
    monkeypatch.setattr(subject.ProductionOrchestratorV2, "_fresh_chain", original)
    calls_before = len(fake.calls)

    event_path = prepared.root / "steps" / "stage-content" / "event.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    old_sha = event["event_sha256"]
    event["recorded_at_utc"] = "2026-07-23T01:00:11Z"
    event_core = dict(event)
    del event_core["event_sha256"]
    event["event_sha256"] = subject.canonical_sha256(event_core)
    event_path.write_text(json.dumps(event), encoding="utf-8")
    (prepared.root / "checkpoints" / f"{old_sha}.json").unlink()

    with pytest.raises(
        subject.controller_v2.JournalTamperError,
        match="no exact controller journal match",
    ):
        subject.execute_production_wave(**kwargs)
    assert len(fake.calls) == calls_before


def test_restart_rejects_orphan_local_event_missing_from_controller_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock)
    kwargs = _execute_kwargs(prepared, fake, clock)
    original = _crash_once_immediately_after_stage(monkeypatch)
    with pytest.raises(RuntimeError, match="after stage completion"):
        subject.execute_production_wave(**kwargs)
    monkeypatch.setattr(subject.ProductionOrchestratorV2, "_fresh_chain", original)
    calls_before = len(fake.calls)

    stage_event = json.loads(
        (
            prepared.root / "steps" / "stage-content" / "event.json"
        ).read_text(encoding="utf-8")
    )
    journal_paths = sorted(prepared.journal_dir.glob("*.json"))
    assert json.loads(journal_paths[-1].read_text(encoding="utf-8"))[
        "event_sha256"
    ] == stage_event["event_sha256"]
    journal_paths[-1].unlink()

    with pytest.raises(
        subject.controller_v2.JournalTamperError,
        match="no exact controller journal match",
    ):
        subject.execute_production_wave(**kwargs)
    assert len(fake.calls) == calls_before


def test_read_refresh_epochs_are_globally_bounded_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock, short_freshness=True)
    kwargs = _execute_kwargs(prepared, fake, clock)
    for _ in range(2):
        with pytest.raises(
            subject.FreshnessRefreshRequired,
            match="no launch was attempted",
        ):
            subject.execute_production_wave(**kwargs)
    refresh_labels = sorted(
        path.parent.name
        for path in (prepared.root / "steps").glob(
            "runtime-gcp-fresh-r*/request.json"
        )
    )
    assert refresh_labels == [
        "runtime-gcp-fresh-r00",
        "runtime-gcp-fresh-r01",
        "runtime-gcp-fresh-r02",
        "runtime-gcp-fresh-r03",
    ]
    assert not (prepared.root / "steps" / "runtime-gcp-fresh-r04").exists()
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("persistent-claim") == 0
    assert modes.count("worker-iam-install") == 0
    assert modes.count("authorize-launch") == 0


def test_restart_after_persisted_identity_partial_reads_then_new_create_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = CrashPersistingDispatcher(
        clock=clock,
        crash_mode="setup-identities",
        initial_identities_exist=False,
        identity_setup_partial_once=True,
    )
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        subject.execute_production_wave(**kwargs)

    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    steps = prepared.root / "steps"
    assert (steps / "setup-identities" / "request.json").is_file()
    assert (steps / "identity-read-recovery-01" / "request.json").is_file()
    assert (steps / "setup-identities-recovery-01" / "request.json").is_file()
    assert len(list(steps.glob("setup-identities/request.json"))) == 1


def test_restart_after_persisted_iam_ambiguity_reconciles_without_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = CrashPersistingDispatcher(
        clock=clock,
        crash_mode="worker-iam-install",
    )
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        subject.execute_production_wave(**kwargs)

    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["worker_iam_install_receipt"]["kind"] == (
        "worker-iam-install-reconciled"
    )
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("worker-iam-reconcile-install") == 1
    readback_request = _persisted_request(prepared, "worker-iam-readback")
    reconcile_event = json.loads(
        (
            prepared.root
            / "steps"
            / "worker-iam-reconcile-install"
            / "event.json"
        ).read_text(encoding="utf-8")
    )
    assert readback_request["predecessor_event_sha256"] == (
        reconcile_event["event_sha256"]
    )
    assert len(
        list((prepared.root / "steps").glob("worker-iam-install/request.json"))
    ) == 1


def test_restart_after_persisted_gce_response_loss_only_reconciles_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = CrashPersistingDispatcher(
        clock=clock,
        crash_mode="authorize-launch",
    )
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        subject.execute_production_wave(**kwargs)

    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["gce_create_receipt"]["kind"] == "gce-create-reconciled"
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("authorize-launch") == 0
    assert modes.count("reconcile-create") == 1
    assert len(
        list((prepared.root / "steps").glob("authorize-launch-r*/request.json"))
    ) == 1


def test_restart_after_persisted_claim_ambiguity_fails_before_iam_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = CrashPersistingDispatcher(
        clock=clock,
        crash_mode="persistent-claim",
    )
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        subject.execute_production_wave(**kwargs)

    with pytest.raises(RuntimeError, match="lacks receipt"):
        subject.execute_production_wave(**kwargs)
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("persistent-claim") == 0
    assert modes.count("worker-iam-prepare") == 0
    assert modes.count("worker-iam-install") == 0
    assert modes.count("authorize-launch") == 0


def test_tampered_launch_request_envelope_blocks_terminal_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = CrashPersistingDispatcher(
        clock=clock,
        crash_mode="authorize-launch",
    )
    kwargs = _execute_kwargs(prepared, fake, clock)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        subject.execute_production_wave(**kwargs)

    paths = list(
        (prepared.root / "steps").glob("authorize-launch-r*/request.json")
    )
    assert len(paths) == 1
    envelope = json.loads(paths[0].read_text(encoding="utf-8"))
    envelope["request"]["observed_at_utc"] = "2026-07-23T09:09:09Z"
    paths[0].write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="request envelope digest changed"):
        subject.execute_production_wave(**kwargs)
    modes = [mode for mode, _ in fake.calls]
    assert modes.count("reconcile-create") == 0


def test_restart_repairs_fully_written_event_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock)
    kwargs = _execute_kwargs(prepared, fake, clock)
    real_write_once = subject._write_once_json
    crashed = False

    def crash_before_first_checkpoint(
        path: Path, value: dict[str, Any]
    ) -> None:
        nonlocal crashed
        if not crashed and path.parent == prepared.root / "checkpoints":
            crashed = True
            raise RuntimeError("simulated loss after event fsync")
        real_write_once(path, value)

    monkeypatch.setattr(
        subject, "_write_once_json", crash_before_first_checkpoint
    )
    with pytest.raises(RuntimeError, match="after event fsync"):
        subject.execute_production_wave(**kwargs)
    event_path = (
        prepared.root / "steps" / "identity-read-initial" / "event.json"
    )
    assert event_path.is_file()
    event = json.loads(event_path.read_text(encoding="utf-8"))
    checkpoint_path = (
        prepared.root / "checkpoints" / f"{event['event_sha256']}.json"
    )
    assert not checkpoint_path.exists()

    monkeypatch.setattr(subject, "_write_once_json", real_write_once)
    manifest = subject.execute_production_wave(**kwargs)
    assert manifest["status"] == subject.EXECUTION_MANIFEST_STATUS
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert set(checkpoint) == {
        "schema",
        "execution_namespace",
        "label",
        "mode",
        "request_sha256",
        "event_sha256",
        "current_profile_changed",
        "checkpoint_sha256",
    }
    checkpoint_core = dict(checkpoint)
    checkpoint_sha = checkpoint_core.pop("checkpoint_sha256")
    assert checkpoint_sha == subject.canonical_sha256(checkpoint_core)


def test_checkpoint_with_extra_field_is_rejected_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = MutableClock()
    prepared = _prepared_for_execute(tmp_path, monkeypatch, clock)
    fake = FakeDispatcher(clock=clock)
    kwargs = _execute_kwargs(prepared, fake, clock)
    real_write_once = subject._write_once_json
    crashed = False

    def crash_before_first_checkpoint(
        path: Path, value: dict[str, Any]
    ) -> None:
        nonlocal crashed
        if not crashed and path.parent == prepared.root / "checkpoints":
            crashed = True
            raise RuntimeError("simulated checkpoint gap")
        real_write_once(path, value)

    monkeypatch.setattr(
        subject, "_write_once_json", crash_before_first_checkpoint
    )
    with pytest.raises(RuntimeError, match="checkpoint gap"):
        subject.execute_production_wave(**kwargs)
    monkeypatch.setattr(subject, "_write_once_json", real_write_once)

    event = json.loads(
        (
            prepared.root
            / "steps"
            / "identity-read-initial"
            / "event.json"
        ).read_text(encoding="utf-8")
    )
    request = _persisted_request(prepared, "identity-read-initial")
    checkpoint_core = {
        "schema": subject.CHECKPOINT_SCHEMA,
        "execution_namespace": prepared.namespace,
        "label": "identity-read-initial",
        "mode": "identity-read",
        "request_sha256": subject.canonical_sha256(request),
        "event_sha256": event["event_sha256"],
        "current_profile_changed": False,
    }
    checkpoint = {
        **checkpoint_core,
        "checkpoint_sha256": subject.canonical_sha256(checkpoint_core),
        "unexpected": True,
    }
    checkpoint_path = (
        prepared.root / "checkpoints" / f"{event['event_sha256']}.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint binding changed"):
        subject.execute_production_wave(**kwargs)


def test_write_once_publication_failure_never_exposes_partial_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.json"

    def fail_publication(source: Path, destination: Path) -> None:
        raise RuntimeError("simulated atomic publication failure")

    monkeypatch.setattr(subject.os, "link", fail_publication)
    with pytest.raises(RuntimeError, match="atomic publication failure"):
        subject._write_once_json(target, {"complete": True})
    assert not target.exists()
    assert list(tmp_path.glob(".*.publish-tmp")) == []
