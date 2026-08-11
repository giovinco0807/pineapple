#!/usr/bin/env python3
"""Operate the production-safe M3.1 fresh-quality GCP 8+7 lifecycle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ofc_regular import hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1 as bridge
from ofc_regular import hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as provider


TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
SALT_ENV = "OFC_M31_FQ_IDENTITY_SALT"
NONCE_ENV = "OFC_M31_FQ_OPERATION_NONCE"
LAUNCH_SENTINEL = "CREATE_EXACT_FRESH_QUALITY_WAVE"
CLEANUP_SENTINEL = "DELETE_EXACT_FRESH_QUALITY_WAVE"


def _load(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw not in (
        bridge.canonical_bytes(value),
        bridge.canonical_bytes(value) + b"\n",
    ):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(bridge.canonical_bytes(value))


def _current_token() -> str:
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        raise PermissionError(f"{TOKEN_ENV} is required in memory")
    return token


def _transport() -> provider.GcpQualityRestAdapter:
    # A wave polls for longer than an access token lives. Re-reading the same
    # environment variable lets a caller refresh it in place without the token
    # ever being written down, which the contract forbids.
    return provider.GcpQualityRestAdapter(
        access_token=_current_token(), token_provider=_current_token
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _common_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    return (
        _load(args.bridge_plan, "fresh-quality bridge plan"),
        _load(args.ledger, "fresh-quality attempt ledger"),
        _load(args.resume, "fresh-quality resume plan"),
        _load(args.wave_request, "fresh-quality wave request"),
    )


def _confirm_run(plan: Mapping[str, Any], confirmation: str) -> None:
    if confirmation != plan["run_name"]:
        raise PermissionError("exact fresh-quality run-name confirmation is required")


def _journal(
    directory: Path,
    *,
    wave_request: Mapping[str, Any],
    create: bool = False,
):
    journal = bridge.open_controller_journal(
        directory,
        wave_request=wave_request,
        create=create,
    )
    journal.load()
    return journal


def _journal_append(
    journal,
    *,
    phase: str,
    mode: str,
    status: str,
    operation_key: str,
    evidence: Mapping[str, Any],
    output: Any,
    mutation_requested: bool,
    mutation_outcome: str,
    predecessor_event_sha256: str | None = None,
) -> dict[str, Any]:
    events = journal.load()
    head = None if not events else events[-1].value["event_sha256"]
    event = journal.append(
        phase=phase,
        mode=mode,
        status=status,
        operation_key=operation_key,
        predecessor_event_sha256=predecessor_event_sha256,
        evidence=evidence,
        output=output,
        mutation_requested=mutation_requested,
        mutation_outcome=mutation_outcome,
        recorded_at_utc=_utc_now(),
        expected_previous_event_sha256=head,
    )
    return event.value


def _phase(request: Mapping[str, Any]) -> str:
    selected = request.get("selected_attempts")
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            "fresh-quality wave request must select at least one attempt"
        )
    wave_indices: set[int] = set()
    for row in selected:
        if not isinstance(row, Mapping):
            raise ValueError(
                "fresh-quality selected attempt must be an object"
            )
        wave_index = row.get("wave_index")
        if (
            not isinstance(wave_index, int)
            or isinstance(wave_index, bool)
            or wave_index < 0
        ):
            raise ValueError(
                "fresh-quality selected attempt wave index is invalid"
            )
        wave_indices.add(wave_index)
    if len(wave_indices) != 1:
        raise ValueError(
            "fresh-quality selected attempts span multiple waves"
        )
    return f"quality-wave-{next(iter(wave_indices))}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--launch-manifest", type=Path, required=True)
    prepare.add_argument("--staging-directory", type=Path, required=True)
    prepare.add_argument("--performance-receipt", type=Path, required=True)
    prepare.add_argument("--profile-registry", type=Path, required=True)
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--project", required=True)
    prepare.add_argument("--region", required=True)
    prepare.add_argument("--zone", required=True)
    prepare.add_argument("--bucket", required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--bridge-plan", type=Path, required=True)
    plan.add_argument("--ledger", type=Path, required=True)
    plan.add_argument("--image-self-link", required=True)
    plan.add_argument("--image-id", required=True)
    plan.add_argument("--guest-os-feature", action="append", required=True)
    plan.add_argument("--worker-service-account", action="append", required=True)
    plan.add_argument("--output-directory", type=Path, required=True)
    plan.add_argument("--journal-directory", type=Path, required=True)

    for name in ("execute", "poll", "cleanup"):
        command = sub.add_parser(name)
        command.add_argument("--provider-plan", type=Path, required=True)
        command.add_argument("--bridge-plan", type=Path, required=True)
        command.add_argument("--ledger", type=Path, required=True)
        command.add_argument("--resume", type=Path, required=True)
        command.add_argument("--wave-request", type=Path, required=True)
        command.add_argument("--journal-directory", type=Path, required=True)
        if name == "poll":
            command.add_argument("--launch-receipt", type=Path, required=True)
        if name == "cleanup":
            command.add_argument("--launch-receipt", type=Path)
            command.add_argument(
                "--abort-without-launch-receipt",
                action="store_true",
                help=(
                    "scan and remove only exact selected VM/disk/IAM identities "
                    "after a lost launch receipt"
                ),
            )
        if name in {"execute", "cleanup"}:
            command.add_argument("--confirm-run-name", required=True)
            command.add_argument("--execute-cloud-mutations", required=True)
        command.add_argument("--output", type=Path, required=True)

    receive = sub.add_parser("receive")
    receive.add_argument("--bridge-plan", type=Path, required=True)
    receive.add_argument("--ledger", type=Path, required=True)
    receive.add_argument("--resume", type=Path, required=True)
    receive.add_argument("--wave-request", type=Path, required=True)
    receive.add_argument("--lifecycle-receipt", type=Path, required=True)
    receive.add_argument("--output-directory", type=Path, required=True)
    receive.add_argument("--accepted-results-directory", type=Path, required=True)
    receive.add_argument("--ledger-output", type=Path, required=True)
    receive.add_argument("--journal-directory", type=Path, required=True)

    final = sub.add_parser("finalize")
    final.add_argument("--bridge-plan", type=Path, required=True)
    final.add_argument("--ledger", type=Path, required=True)
    final.add_argument("--accepted-results-directory", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        salt = os.environ.get(SALT_ENV, "")
        if not salt:
            raise PermissionError(f"{SALT_ENV} is required and is never persisted")
        if args.output_directory.exists():
            raise FileExistsError("fresh-quality prepare output is create-only")
        args.output_directory.mkdir(parents=True, exist_ok=False)
        plan = bridge.build_gcp_plan(
            launch_manifest_path=args.launch_manifest,
            staging_directory=args.staging_directory,
            performance_receipt_path=args.performance_receipt,
            profile_registry_path=args.profile_registry,
            run_name=args.run_name,
            identity_salt=salt,
            project=args.project,
            region=args.region,
            zone=args.zone,
            bucket=args.bucket,
        )
        ledger = bridge.initial_attempt_ledger(plan)
        _write(args.output_directory / "bridge_plan.json", plan)
        _write(args.output_directory / "attempt_ledger.json", ledger)
        print(
            bridge.canonical_bytes(
                {
                    "status": "prepared",
                    "bridge_plan_sha256": plan["plan_sha256"],
                    "ledger_sha256": ledger["ledger_sha256"],
                    "cloud_mutated": False,
                    "current_profile_changed": False,
                }
            ).decode("ascii")
        )
        return 0

    if args.command == "plan":
        plan = _load(args.bridge_plan, "fresh-quality bridge plan")
        ledger = _load(args.ledger, "fresh-quality attempt ledger")
        resume = bridge.build_resume_plan(plan, ledger)
        request = bridge.build_wave_request(plan, ledger, resume)
        provider_plan = provider.build_provider_plan(
            bridge_plan=plan,
            ledger=ledger,
            resume=resume,
            wave_request=request,
            image_self_link=args.image_self_link,
            image_id=args.image_id,
            guest_os_features=args.guest_os_feature,
            worker_service_accounts=args.worker_service_account,
        )
        if args.output_directory.exists():
            raise FileExistsError("fresh-quality wave plan output is create-only")
        args.output_directory.mkdir(parents=True, exist_ok=False)
        _write(args.output_directory / "resume_plan.json", resume)
        _write(args.output_directory / "wave_request.json", request)
        _write(args.output_directory / "provider_plan.json", provider_plan)
        journal = _journal(
            args.journal_directory,
            wave_request=request,
            create=True,
        )
        _journal_append(
            journal,
            phase=_phase(request),
            mode="plan",
            status="complete",
            operation_key=f"{_phase(request)}-plan",
            evidence={
                "plan_sha256": plan["plan_sha256"],
                "ledger_sha256": ledger["ledger_sha256"],
                "resume_sha256": resume["resume_sha256"],
                "request_sha256": request["request_sha256"],
            },
            output={
                "provider_plan_sha256": provider_plan["provider_plan_sha256"],
                "selected_job_count": len(resume["selected_attempts"]),
                "cloud_mutated": False,
                "current_profile_changed": False,
            },
            mutation_requested=False,
            mutation_outcome="not_requested",
        )
        print(
            bridge.canonical_bytes(
                {
                    "status": "wave_planned_not_authorized",
                    "resume_sha256": resume["resume_sha256"],
                    "request_sha256": request["request_sha256"],
                    "provider_plan_sha256": provider_plan["provider_plan_sha256"],
                    "selected_job_count": len(resume["selected_attempts"]),
                    "cloud_mutated": False,
                    "current_profile_changed": False,
                }
            ).decode("ascii")
        )
        return 0

    if args.command == "finalize":
        receipt = bridge.write_final_receipt(
            args.output,
            plan=_load(args.bridge_plan, "fresh-quality bridge plan"),
            ledger=_load(args.ledger, "fresh-quality attempt ledger"),
            accepted_results_directory=args.accepted_results_directory,
        )
        print(bridge.canonical_bytes(receipt).decode("ascii"))
        return 0 if receipt["quality_pilot_passed"] else 2

    if args.command == "receive":
        plan, ledger, resume, request = _common_inputs(args)
        lifecycle = _load(args.lifecycle_receipt, "fresh-quality lifecycle receipt")
        receipt = provider.receive_wave(
            bridge_plan=plan,
            ledger=ledger,
            resume=resume,
            wave_request=request,
            lifecycle_receipt=lifecycle,
            transport=_transport(),
            output_directory=args.output_directory,
            accepted_results_directory=args.accepted_results_directory,
        )
        _write(args.ledger_output, receipt["output_ledger"])
        journal = _journal(
            args.journal_directory,
            wave_request=request,
        )
        _journal_append(
            journal,
            phase=_phase(request),
            mode="receive",
            status="complete",
            operation_key=(
                f"{_phase(request)}-receive-{len(journal.load()) + 1:06d}"
            ),
            evidence={
                "request_sha256": request["request_sha256"],
                "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
            },
            output=receipt,
            mutation_requested=False,
            mutation_outcome="not_requested",
        )
        print(bridge.canonical_bytes(receipt).decode("ascii"))
        return 0

    provider_plan = _load(args.provider_plan, "fresh-quality provider plan")
    plan, ledger, resume, request = _common_inputs(args)
    transport = _transport()
    if args.command == "execute":
        _confirm_run(plan, args.confirm_run_name)
        if args.execute_cloud_mutations != LAUNCH_SENTINEL:
            raise PermissionError(
                f"execute requires --execute-cloud-mutations {LAUNCH_SENTINEL}"
            )
        nonce = os.environ.get(NONCE_ENV, "")
        if not nonce:
            raise PermissionError(f"{NONCE_ENV} is required and is never persisted")
        journal = _journal(
            args.journal_directory,
            wave_request=request,
        )
        operation_key = f"{_phase(request)}-execute"
        if any(
            event.value["operation_key"] == operation_key
            and event.value["mutation_requested"] is True
            for event in journal.load()
        ):
            raise bridge.controller_v2.PendingMutationError(
                "fresh-quality execute intent already exists; cleanup/reconcile only"
            )
        intent = _journal_append(
            journal,
            phase=_phase(request),
            mode="execute",
            status="pending",
            operation_key=operation_key,
            evidence={
                "provider_plan_sha256": provider_plan["provider_plan_sha256"],
                "request_sha256": request["request_sha256"],
            },
            output={"cloud_mutated": False, "current_profile_changed": False},
            mutation_requested=True,
            mutation_outcome="pending",
        )
        try:
            receipt = provider.execute_wave(
                provider_plan=provider_plan,
                bridge_plan=plan,
                ledger=ledger,
                resume=resume,
                wave_request=request,
                raw_nonce=nonce,
                transport=transport,
                now_unix_seconds=int(time.time()),
                observed_at_utc=_utc_now(),
            )
        except Exception as error:
            _journal_append(
                journal,
                phase=_phase(request),
                mode="execute",
                status="failed",
                operation_key=operation_key,
                predecessor_event_sha256=intent["event_sha256"],
                evidence={
                    "provider_plan_sha256": provider_plan["provider_plan_sha256"],
                    "request_sha256": request["request_sha256"],
                },
                output={
                    "error_type": type(error).__name__,
                    "cloud_state": "unknown",
                    "current_profile_changed": False,
                },
                mutation_requested=True,
                mutation_outcome="unknown",
            )
            raise
        _journal_append(
            journal,
            phase=_phase(request),
            mode="execute",
            status="complete" if receipt["create_complete"] else "partial",
            operation_key=operation_key,
            predecessor_event_sha256=intent["event_sha256"],
            evidence={
                "provider_plan_sha256": provider_plan["provider_plan_sha256"],
                "request_sha256": request["request_sha256"],
            },
            output=receipt,
            mutation_requested=True,
            mutation_outcome=(
                "performed" if receipt["create_complete"] else "partial"
            ),
        )
        _write(args.output, receipt)
        print(bridge.canonical_bytes(receipt).decode("ascii"))
        return 0 if receipt["create_complete"] else 2

    if args.command == "poll":
        launch = _load(args.launch_receipt, "fresh-quality launch receipt")
        receipt = provider.poll_wave(
            provider_plan=provider_plan,
            bridge_plan=plan,
            ledger=ledger,
            resume=resume,
            wave_request=request,
            launch_receipt=launch,
            transport=transport,
        )
        _write(args.output, receipt)
        journal = _journal(
            args.journal_directory,
            wave_request=request,
        )
        _journal_append(
            journal,
            phase=_phase(request),
            mode="poll",
            status="complete",
            operation_key=f"{_phase(request)}-poll-{len(journal.load()) + 1:06d}",
            evidence={
                "request_sha256": request["request_sha256"],
                "launch_receipt_sha256": launch["receipt_sha256"],
            },
            output=receipt,
            mutation_requested=False,
            mutation_outcome="not_requested",
        )
        print(bridge.canonical_bytes(receipt).decode("ascii"))
        return 0 if receipt["done_count"] == receipt["selected_job_count"] else 3

    _confirm_run(plan, args.confirm_run_name)
    if args.execute_cloud_mutations != CLEANUP_SENTINEL:
        raise PermissionError(
            f"cleanup requires --execute-cloud-mutations {CLEANUP_SENTINEL}"
        )
    if args.launch_receipt is None and not args.abort_without_launch_receipt:
        raise PermissionError(
            "cleanup without a receipt requires --abort-without-launch-receipt"
        )
    if args.launch_receipt is not None and args.abort_without_launch_receipt:
        raise ValueError("cleanup launch receipt and abort mode are mutually exclusive")
    launch = (
        None
        if args.launch_receipt is None
        else _load(args.launch_receipt, "fresh-quality launch receipt")
    )
    journal = _journal(
        args.journal_directory,
        wave_request=request,
    )
    cleanup_number = (
        sum(
            event.value["mode"] == "cleanup"
            and event.value["mutation_requested"] is True
            for event in journal.load()
        )
        + 1
    )
    operation_key = f"{_phase(request)}-cleanup-{cleanup_number:02d}"
    intent = _journal_append(
        journal,
        phase=_phase(request),
        mode="cleanup",
        status="pending",
        operation_key=operation_key,
        evidence={
            "provider_plan_sha256": provider_plan["provider_plan_sha256"],
            "request_sha256": request["request_sha256"],
            "launch_receipt_sha256": (
                None if launch is None else launch["receipt_sha256"]
            ),
            "abort_without_launch_receipt": launch is None,
        },
        output={"cloud_mutated": False, "current_profile_changed": False},
        mutation_requested=True,
        mutation_outcome="pending",
    )
    try:
        lifecycle = provider.cleanup_wave(
            provider_plan=provider_plan,
            bridge_plan=plan,
            ledger=ledger,
            resume=resume,
            wave_request=request,
            launch_receipt=launch,
            transport=transport,
        )
    except Exception as error:
        _journal_append(
            journal,
            phase=_phase(request),
            mode="cleanup",
            status="failed",
            operation_key=operation_key,
            predecessor_event_sha256=intent["event_sha256"],
            evidence={
                "provider_plan_sha256": provider_plan["provider_plan_sha256"],
                "request_sha256": request["request_sha256"],
            },
            output={
                "error_type": type(error).__name__,
                "cloud_state": "unknown",
                "current_profile_changed": False,
            },
            mutation_requested=True,
            mutation_outcome="unknown",
        )
        raise
    _journal_append(
        journal,
        phase=_phase(request),
        mode="cleanup",
        status="complete",
        operation_key=operation_key,
        predecessor_event_sha256=intent["event_sha256"],
        evidence={
            "provider_plan_sha256": provider_plan["provider_plan_sha256"],
            "request_sha256": request["request_sha256"],
        },
        output=lifecycle,
        mutation_requested=True,
        mutation_outcome="performed",
    )
    _write(args.output, lifecycle)
    print(bridge.canonical_bytes(lifecycle).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"fresh-quality GCP lifecycle failed closed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
