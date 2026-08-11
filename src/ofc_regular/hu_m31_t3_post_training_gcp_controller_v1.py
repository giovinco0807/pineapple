"""Persistent fail-closed controller for M3.1 post-training Spot fanout.

The controller is deliberately explicit: ``plan`` is local-only, every cloud
phase requires a distinct sentinel, and a lifecycle is accepted only after
exact cleanup plus local scientific receive.  OAuth bearer bytes and launch
nonces are environment-only and never written into a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_post_training_gcp_v1 as core
from . import hu_m31_t3_abr_teacher_v1 as abr_teacher
from . import hu_m31_t3_post_dataset_controller_v1 as post_dataset
from . import hu_m31_t3_post_training_runtime_v1 as runtime_bundle


EVENT_SCHEMA = "hu_m31_t3_post_training_controller_event_v1"
OAUTH_PREFLIGHT_SCHEMA = "hu_m31_t3_post_training_oauth_preflight_v1"
_ACCEPTED = re.compile(r"^([0-9]{4})-([0-9a-f]{64})\.json$")


def _write_once(path: Path, value: Mapping[str, Any]) -> Path:
    return core._write_once(path, value)  # type: ignore[attr-defined]


def _read(path: str | Path, label: str) -> dict[str, Any]:
    return core._read_canonical(path, label)  # type: ignore[attr-defined]


def _root(path: str | Path, *, create: bool) -> Path:
    root = Path(path).resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("post-training controller root is unsafe")
    return root


def _token(raw: str | None) -> str:
    if raw is None or raw != raw.strip():
        raise PermissionError(
            f"{core.CONTROLLER_TOKEN_ENV} must contain a UUIDv4"
        )
    try:
        parsed = uuid.UUID(raw)
    except ValueError as exc:
        raise PermissionError("controller token must be UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw:
        raise PermissionError("controller token must be canonical UUIDv4")
    return raw


def _append_event(
    root: Path,
    *,
    phase: str,
    operation_key: str,
    payload: Mapping[str, Any],
    cloud_mutated: bool,
) -> dict[str, Any]:
    directory = root / "events"
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(directory.glob("*.json"))
    previous = None
    if paths:
        previous_value = _read(paths[-1], "controller event")
        previous = previous_value["event_sha256"]
    core_value = {
        "schema": EVENT_SCHEMA,
        "sequence": len(paths),
        "phase": phase,
        "operation_key": operation_key,
        "previous_event_sha256": previous,
        "payload": dict(payload),
        "cloud_mutated": cloud_mutated,
        "current_profile_changed": False,
    }
    event = {
        **core_value,
        "event_sha256": core.canonical_sha256(core_value),
    }
    _write_once(directory / f"{len(paths):06d}.json", event)
    return event


def prepare_controller(
    *,
    cloud_plan: Mapping[str, Any],
    output_root: str | Path,
    raw_controller_token: str,
) -> dict[str, Any]:
    plan = core.validate_cloud_plan(cloud_plan, replay_sources=True)
    token = _token(raw_controller_token)
    root = _root(output_root, create=True)
    _write_once(root / "cloud_plan.json", plan)
    identity = {
        "schema": core.CONTROLLER_SCHEMA,
        "status": "prepared_cloud_not_started",
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "run_name": plan["contract"]["run_name"],
        "workload": plan["contract"]["workload"],
        "job_count": plan["contract"]["job_count"],
        "controller_token_sha256": hashlib.sha256(
            token.encode("ascii")
        ).hexdigest(),
        "controller_token_environment": core.CONTROLLER_TOKEN_ENV,
        "phase_sentinel_environment": core.PHASE_SENTINEL_ENV,
        "oauth_token_environment": core.OAUTH_TOKEN_ENV,
        "oauth_expiry_environment": core.OAUTH_EXPIRES_ENV,
        "oauth_min_remaining_seconds": core.MIN_OAUTH_TTL_SECONDS,
        "one_wave_at_a_time": True,
        "current_profile_changed": False,
    }
    contract = {
        **identity,
        "controller_sha256": core.canonical_sha256(identity),
    }
    _write_once(root / "controller.json", contract)
    _append_event(
        root,
        phase="prepare",
        operation_key="prepare",
        payload={
            "controller_sha256": contract["controller_sha256"],
            "cloud_plan_sha256": plan["cloud_plan_sha256"],
            "cloud_execution_started": False,
        },
        cloud_mutated=False,
    )
    return contract


def _load(
    output_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = _root(output_root, create=False)
    contract = _read(root / "controller.json", "controller contract")
    plan = core.validate_cloud_plan(
        _read(root / "cloud_plan.json", "cloud plan"),
        replay_sources=True,
    )
    if (
        contract.get("schema") != core.CONTROLLER_SCHEMA
        or contract.get("controller_sha256")
        != core._self_digest(contract, "controller_sha256")  # type: ignore[attr-defined]
        or contract.get("cloud_plan_sha256") != plan["cloud_plan_sha256"]
        or contract.get("current_profile_changed") is not False
    ):
        raise ValueError("post-training controller contract changed")
    raw = _token(os.environ.get(core.CONTROLLER_TOKEN_ENV))
    if hashlib.sha256(raw.encode("ascii")).hexdigest() != contract[
        "controller_token_sha256"
    ]:
        raise PermissionError("post-training controller token changed")
    return root, contract, plan


def _accepted(root: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    directory = root / "accepted"
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("accepted lifecycle directory is unsafe")
    result = []
    for index, path in enumerate(sorted(directory.iterdir())):
        match = _ACCEPTED.fullmatch(path.name)
        if (
            match is None
            or int(match.group(1)) != index
            or path.is_symlink()
            or not path.is_file()
            or core.sha256_file(path) != match.group(2)
        ):
            raise ValueError("accepted lifecycle chain changed")
        result.append(
            core.validate_lifecycle(
                _read(path, "accepted lifecycle"), cloud_plan=plan
            )
        )
    return result


@dataclass(frozen=True)
class Context:
    root: Path
    controller: Mapping[str, Any]
    plan: Mapping[str, Any]
    wave: Mapping[str, Any]
    directory: Path


def plan_next_wave(output_root: str | Path) -> Context | dict[str, Any]:
    root, controller, plan = _load(output_root)
    accepted = _accepted(root, plan)
    wave = core.build_wave(plan, accepted_lifecycles=accepted)
    if wave["status"] == "complete":
        return wave
    directory = (
        root
        / "waves"
        / f"w{int(wave['wave_index']):03d}-{wave['wave_sha256'][:16]}"
    )
    _write_once(directory / "wave.json", wave)
    _append_event(
        root,
        phase="plan",
        operation_key=f"plan:{wave['wave_sha256']}",
        payload={
            "wave_sha256": wave["wave_sha256"],
            "selected_count": wave["selected_count"],
        },
        cloud_mutated=False,
    )
    return Context(root, controller, plan, wave, directory)


def _context(output_root: str | Path) -> Context:
    value = plan_next_wave(output_root)
    if not isinstance(value, Context):
        raise PermissionError("post-training fanout is already complete")
    return value


def phase_sentinel(output_root: str | Path, phase: str) -> str:
    context = _context(output_root)
    return core.expected_phase_sentinel(context.plan, context.wave, phase)


def _require_phase(
    context: Context,
    *,
    phase: str,
    confirm_run_name: str,
    allowed: bool,
) -> None:
    if not allowed:
        raise PermissionError(f"{phase} requires its explicit allow flag")
    if confirm_run_name != context.controller["run_name"]:
        raise PermissionError("confirmed post-training run name changed")
    expected = core.expected_phase_sentinel(
        context.plan, context.wave, phase
    )
    if os.environ.get(core.PHASE_SENTINEL_ENV) != expected:
        raise PermissionError(
            f"{core.PHASE_SENTINEL_ENV} must exactly equal {expected}"
        )


def oauth_preflight(
    *,
    expires_at_unix: int,
    now_unix: int | None = None,
) -> dict[str, Any]:
    now = int(time.time()) if now_unix is None else int(now_unix)
    remaining = int(expires_at_unix) - now
    if remaining < core.MIN_OAUTH_TTL_SECONDS:
        raise PermissionError("fresh OAuth TTL is below 2700 seconds")
    identity = {
        "schema": OAUTH_PREFLIGHT_SCHEMA,
        "status": "pass_redacted_oauth_ttl",
        "remaining_seconds": remaining,
        "minimum_seconds": core.MIN_OAUTH_TTL_SECONDS,
        "token_recorded": False,
        "token_hash_recorded": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {
        **identity,
        "receipt_sha256": core.canonical_sha256(identity),
    }


def _live_cloud() -> core.GcpRestAdapter:
    token = os.environ.get(core.OAUTH_TOKEN_ENV)
    expiry = os.environ.get(core.OAUTH_EXPIRES_ENV)
    if (
        token is None
        or token != token.strip()
        or len(token) < 20
        or any(character.isspace() for character in token)
        or expiry is None
        or not expiry.isdigit()
    ):
        raise PermissionError(
            "fresh OAuth token and numeric expiry must be environment-only"
        )
    oauth_preflight(expires_at_unix=int(expiry))
    return core.GcpRestAdapter(access_token=token)


def execute_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    raw_nonce: str,
    cloud: core.CloudTransport | None = None,
    now_unix_seconds: int | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="execute",
        confirm_run_name=confirm_run_name,
        allowed=allow_cloud_mutation,
    )
    target = context.directory / "launch.json"
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            "execute is one-shot; use poll/cleanup, never retry launch"
        )
    _write_once(
        context.directory / "MUTATION_INTENT.json",
        {
            "schema": "hu_m31_t3_post_training_mutation_intent_v1",
            "wave_sha256": context.wave["wave_sha256"],
            "phase": "execute",
            "cloud_mutation_may_have_started": True,
            "current_profile_changed": False,
        },
    )
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    transport = _live_cloud() if cloud is None else cloud
    receipt = core.execute_wave(
        cloud_plan=context.plan,
        wave=context.wave,
        transport=transport,
        raw_nonce=raw_nonce,
        now_unix_seconds=now,
        observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        sleep=sleep,
    )
    _write_once(target, receipt)
    _append_event(
        context.root,
        phase="execute",
        operation_key=f"execute:{context.wave['wave_sha256']}",
        payload={
            "launch_receipt_sha256": receipt["receipt_sha256"],
            "created_instance_count": receipt["created_instance_count"],
        },
        cloud_mutated=True,
    )
    return receipt


def poll_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: core.CloudTransport | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="poll",
        confirm_run_name=confirm_run_name,
        allowed=allow_cloud_read,
    )
    receipt = core.poll_wave(
        cloud_plan=context.plan,
        wave=context.wave,
        transport=_live_cloud() if cloud is None else cloud,
    )
    _write_once(
        context.directory
        / f"poll-{len(list(context.directory.glob('poll-*.json'))):04d}.json",
        receipt,
    )
    return receipt


def cleanup_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    allow_incomplete: bool,
    recover_missing_launch: bool = False,
    cloud: core.CloudTransport | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    context = _context(output_root)
    phase = "cleanup-incomplete" if allow_incomplete else "cleanup"
    _require_phase(
        context,
        phase=phase,
        confirm_run_name=confirm_run_name,
        allowed=allow_cloud_mutation,
    )
    target = context.directory / "lifecycle.json"
    if target.exists() or target.is_symlink():
        return core.validate_lifecycle(
            _read(target, "lifecycle"), cloud_plan=context.plan
        )
    transport = _live_cloud() if cloud is None else cloud
    launch_path = context.directory / "launch.json"
    if launch_path.exists():
        receipt = core.cleanup_wave(
            cloud_plan=context.plan,
            wave=context.wave,
            launch=_read(launch_path, "launch receipt"),
            transport=transport,
            allow_incomplete=allow_incomplete,
            sleep=sleep,
        )
    elif recover_missing_launch and allow_incomplete:
        receipt = core.abort_wave(
            cloud_plan=context.plan,
            wave=context.wave,
            transport=transport,
            sleep=sleep,
        )
    else:
        raise PermissionError(
            "launch receipt missing; explicit recovery cleanup is required"
        )
    _write_once(target, receipt)
    _append_event(
        context.root,
        phase=phase,
        operation_key=f"cleanup:{context.wave['wave_sha256']}",
        payload={"lifecycle_sha256": receipt["receipt_sha256"]},
        cloud_mutated=True,
    )
    return receipt


def receive_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: core.CloudTransport | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="receive",
        confirm_run_name=confirm_run_name,
        allowed=allow_cloud_read,
    )
    lifecycle = _read(context.directory / "lifecycle.json", "lifecycle")
    receipt = core.receive_lifecycle(
        cloud_plan=context.plan,
        lifecycle=lifecycle,
        transport=_live_cloud() if cloud is None else cloud,
        output_root=context.root / "received",
    )
    _write_once(context.directory / "receive.json", receipt)
    _append_event(
        context.root,
        phase="receive",
        operation_key=f"receive:{context.wave['wave_sha256']}",
        payload={"receive_sha256": receipt["receipt_sha256"]},
        cloud_mutated=False,
    )
    return receipt


def accept_next_wave(output_root: str | Path) -> dict[str, Any]:
    context = _context(output_root)
    lifecycle = core.validate_lifecycle(
        _read(context.directory / "lifecycle.json", "lifecycle"),
        cloud_plan=context.plan,
    )
    receive = _read(context.directory / "receive.json", "receive receipt")
    if (
        receive.get("schema") != core.RECEIVE_SCHEMA
        or receive.get("receipt_sha256")
        != core._self_digest(receive, "receipt_sha256")  # type: ignore[attr-defined]
        or receive.get("cloud_plan_sha256")
        != context.plan["cloud_plan_sha256"]
        or receive.get("lifecycle_sha256") != lifecycle["receipt_sha256"]
        or receive.get("received_file_count")
        != len(receive.get("received", []))
        or receive.get("source_replayed") is not True
        or receive.get("cloud_mutated") is not False
        or receive.get("current_profile_changed") is not False
    ):
        raise PermissionError("scientific receive must precede acceptance")
    for record in receive["received"]:
        path = Path(str(record["local_path"]))
        if (
            path.is_symlink()
            or not path.is_file()
            or core.sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise PermissionError("received file changed before acceptance")
    accepted = _accepted(context.root, context.plan)
    raw = core.canonical_bytes(lifecycle)
    path = (
        context.root
        / "accepted"
        / f"{len(accepted):04d}-{hashlib.sha256(raw).hexdigest()}.json"
    )
    _write_once(path, lifecycle)
    _append_event(
        context.root,
        phase="accept",
        operation_key=f"accept:{lifecycle['receipt_sha256']}",
        payload={"accepted_lifecycle_count": len(accepted) + 1},
        cloud_mutated=False,
    )
    next_value = plan_next_wave(context.root)
    return {
        "status": (
            "complete" if not isinstance(next_value, Context) else "wave_ready"
        ),
        "accepted_lifecycle_count": len(accepted) + 1,
        "next_selected_count": (
            0
            if not isinstance(next_value, Context)
            else next_value.wave["selected_count"]
        ),
        "cloud_mutated": False,
        "current_profile_changed": False,
    }


def controller_status(output_root: str | Path) -> dict[str, Any]:
    root, _controller, plan = _load(output_root)
    accepted = _accepted(root, plan)
    completed = {
        row["job_id"]
        for lifecycle in accepted
        for row in lifecycle["rows"]
        if row["status"] == "complete"
    }
    failed_attempts = sum(
        row["status"] == "failed"
        for lifecycle in accepted
        for row in lifecycle["rows"]
    )
    return {
        "status": (
            "complete"
            if len(completed) == plan["contract"]["job_count"]
            else "incomplete"
        ),
        "workload": plan["contract"]["workload"],
        "job_count": plan["contract"]["job_count"],
        "completed_job_count": len(completed),
        "pending_job_count": plan["contract"]["job_count"] - len(completed),
        "failed_attempt_count": failed_attempts,
        "accepted_lifecycle_count": len(accepted),
        "max_concurrent_vms": core.MAX_CONCURRENT_VMS,
        "machine_type": core.MACHINE_TYPE,
        "current_profile_changed": False,
    }


def finalize_received(output_root: str | Path) -> dict[str, Any]:
    """Merge only after every exact job has an accepted lifecycle."""

    root, _controller, plan = _load(output_root)
    status = controller_status(root)
    if status["status"] != "complete":
        raise PermissionError("post-training fanout is incomplete")
    contract = plan["contract"]
    sources = core._sources_by_kind(contract)  # type: ignore[attr-defined]
    if contract["workload"].startswith("abr_"):
        run = root / "received" / "final_abr_run"
        pairs = run / abr_teacher.PAIR_DIRECTORY
        pairs.mkdir(parents=True, exist_ok=True)
        plan_value = abr_teacher.validate_plan(
            _read(sources["abr_plan"]["source_path"], "ABR plan")
        )
        _write_once(run / abr_teacher.PLAN_FILE, plan_value)
        received_pairs = root / "received" / "pairs"
        evidence = []
        for index in plan_value["pair_indices"]:
            source = received_pairs / f"pair-{int(index):06d}.json"
            value = abr_teacher.validate_pair_evidence(
                _read(source, "received ABR pair"),
                plan=plan_value,
                pair_index=int(index),
            )
            _write_once(pairs / source.name, value)
            evidence.append(value)
        done = abr_teacher._done_value(  # type: ignore[attr-defined]
            plan=plan_value, evidence=evidence
        )
        _write_once(run / abr_teacher.DONE_FILE, done)
        finalized = abr_teacher.finalize_run(run)
        return {
            "status": finalized["status"],
            "workload": contract["workload"],
            "pair_count": plan_value["pair_count"],
            "teacher_receipt_file_sha256": core.sha256_file(
                run / abr_teacher.TEACHER_RECEIPT_FILE
            ),
            "raw_examples_file_sha256": core.sha256_file(
                run / abr_teacher.RAW_EXAMPLES_FILE
            ),
            "synthetic_values_used": False,
            "current_profile_changed": False,
        }
    promotion_plan = _read(
        sources["promotion_plan"]["source_path"], "promotion plan"
    )
    execution_plan = _read(
        sources["execution_plan"]["source_path"], "execution plan"
    )
    progress = post_dataset.inspect_evaluation_progress(
        promotion_plan=promotion_plan,
        execution_plan=execution_plan,
        shard_directory=root / "received" / "shards",
    )
    if (
        progress["status"] != "complete"
        or progress["complete_work_item_count"] != 260
        or progress["accepted_row_count"] != 13_000
    ):
        raise PermissionError("locked-promotion receive grid is incomplete")
    return {
        "status": "complete_260_ready_for_locked_closeout",
        "workload": contract["workload"],
        "progress_sha256": progress["progress_sha256"],
        "work_item_count": 260,
        "row_count": 13_000,
        "current_profile_changed": False,
    }


def _filesystem_type(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if os.name == "nt":
        return "ntfs"
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return "unknown"
    candidates: list[tuple[int, str]] = []
    text = mounts.read_text(encoding="utf-8")
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = fields[1].replace("\\040", " ")
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        candidates.append((len(mount), fields[2]))
    return max(candidates)[1] if candidates else "unknown"


def run_local_smoke(
    *,
    contract_path: str | Path,
    staging_root: str | Path,
    output_root: str | Path,
    receipt_path: str | Path,
    require_host_target: bool = True,
) -> dict[str, Any]:
    """Run job zero with real code; ext4/xfs is mandatory for authorization."""

    contract = core.validate_contract(
        _read(contract_path, "workload contract"), replay_sources=True
    )
    staging_type = _filesystem_type(staging_root)
    output_type = _filesystem_type(output_root)
    if staging_type not in {"ext4", "xfs"} or output_type != staging_type:
        raise PermissionError(
            "local smoke staging/output must share an ext4 or xfs filesystem"
        )
    staged = core.stage_local_sources(contract, staging_root)
    first = contract["jobs"][0]["job_id"]
    manifest = core.run_worker_job(
        contract=contract,
        job_id=first,
        staging_root=staged,
        output_root=output_root,
        require_host_target=require_host_target,
    )
    receipt = core.build_local_smoke_receipt(
        contract=contract,
        output_root=output_root,
        manifest=manifest,
        filesystem_type=staging_type,
        production_executor=True,
    )
    _write_once(Path(receipt_path), receipt)
    return receipt


def _provider_config(path: str | Path) -> dict[str, Any]:
    value = _read(path, "post-training provider config")
    if set(value) != {
        "schema",
        "image_self_link",
        "image_id",
        "guest_os_features",
        "worker_service_accounts",
        "current_profile_changed",
    } or value.get("schema") != "hu_m31_t3_post_training_provider_config_v1":
        raise ValueError("post-training provider config changed")
    if value.get("current_profile_changed") is not False:
        raise ValueError("provider config cannot change current")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package-abr-runtime")
    package.add_argument("--repository-root", required=True)
    package.add_argument("--models-root", required=True)
    package.add_argument("--raw-wheelhouse", required=True)
    package.add_argument("--raw-wheelhouse-manifest", required=True)
    package.add_argument("--accepted-library", required=True)
    package.add_argument("--diagnostic-library", required=True)
    package.add_argument(
        "--expected-diagnostic-library-sha256", required=True
    )
    package.add_argument("--output-root", required=True)
    abr = sub.add_parser("build-abr-contract")
    abr.add_argument("--run-name", required=True)
    abr.add_argument("--bucket", required=True)
    abr.add_argument("--mode", choices=("pilot50", "production250"), required=True)
    abr.add_argument("--abr-plan", required=True)
    abr.add_argument("--runtime-bundle-manifest", required=True)
    abr.add_argument("--runtime-archive", required=True)
    abr.add_argument("--wheelhouse-archive", required=True)
    abr.add_argument("--accepted-library", required=True)
    abr.add_argument("--diagnostic-library", required=True)
    abr.add_argument("--output", required=True)
    locked = sub.add_parser("build-locked-contract")
    locked.add_argument("--run-name", required=True)
    locked.add_argument("--bucket", required=True)
    locked.add_argument("--promotion-plan", required=True)
    locked.add_argument("--execution-plan", required=True)
    locked.add_argument("--runtime-archive", required=True)
    locked.add_argument("--wheelhouse-archive", required=True)
    locked.add_argument("--closure-package", required=True)
    locked.add_argument("--compatibility-threshold-lock", required=True)
    locked.add_argument("--policy-registry", required=True)
    locked.add_argument("--abr-bundle-archive", required=True)
    locked.add_argument("--expected-closure-sha256", required=True)
    locked.add_argument("--expected-abr-bundle-archive-sha256", required=True)
    locked.add_argument("--expected-abr-bundle-file-sha256", required=True)
    locked.add_argument(
        "--expected-abr-production-build-receipt-sha256", required=True
    )
    locked.add_argument("--output", required=True)
    smoke = sub.add_parser("local-smoke")
    smoke.add_argument("--contract", required=True)
    smoke.add_argument("--staging-root", required=True)
    smoke.add_argument("--output-root", required=True)
    smoke.add_argument("--receipt", required=True)
    authorize = sub.add_parser("authorize-cloud-plan")
    authorize.add_argument("--contract", required=True)
    authorize.add_argument("--smoke-receipt", required=True)
    authorize.add_argument("--provider-config", required=True)
    authorize.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare-controller")
    prepare.add_argument("--cloud-plan", required=True)
    prepare.add_argument("--output-root", required=True)
    execute = sub.add_parser("execute")
    poll = sub.add_parser("poll")
    cleanup = sub.add_parser("cleanup")
    receive = sub.add_parser("receive")
    for command in (execute, poll, cleanup, receive):
        command.add_argument("--output-root", required=True)
        command.add_argument("--confirm-run-name", required=True)
    execute.add_argument("--allow-cloud-mutation", action="store_true")
    cleanup.add_argument("--allow-cloud-mutation", action="store_true")
    cleanup.add_argument("--allow-incomplete", action="store_true")
    cleanup.add_argument("--recover-missing-launch", action="store_true")
    poll.add_argument("--allow-cloud-read", action="store_true")
    receive.add_argument("--allow-cloud-read", action="store_true")
    for name in ("status", "plan", "sentinel", "accept", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--output-root", required=True)
        if name == "sentinel":
            command.add_argument("--phase", required=True)
    oauth = sub.add_parser("oauth-preflight")
    oauth.add_argument("--expires-at-unix", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "package-abr-runtime":
            resolution = runtime_bundle.package_runtime_bundle(
                repository_root=args.repository_root,
                models_root=args.models_root,
                raw_wheelhouse_path=args.raw_wheelhouse,
                raw_wheelhouse_manifest_path=args.raw_wheelhouse_manifest,
                accepted_library_path=args.accepted_library,
                diagnostic_library_path=args.diagnostic_library,
                expected_diagnostic_library_sha256=(
                    args.expected_diagnostic_library_sha256
                ),
                output_root=args.output_root,
            )
            result = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in resolution.items()
            }
        elif args.command == "build-abr-contract":
            result = core.build_abr_contract(
                run_name=args.run_name,
                bucket=args.bucket,
                mode=args.mode,
                abr_plan_path=args.abr_plan,
                runtime_bundle_manifest_path=args.runtime_bundle_manifest,
                runtime_archive_path=args.runtime_archive,
                wheelhouse_archive_path=args.wheelhouse_archive,
                accepted_library_path=args.accepted_library,
                diagnostic_library_path=args.diagnostic_library,
            )
            _write_once(Path(args.output), result)
        elif args.command == "build-locked-contract":
            result = core.build_locked_promotion_contract(
                run_name=args.run_name,
                bucket=args.bucket,
                promotion_plan_path=args.promotion_plan,
                execution_plan_path=args.execution_plan,
                runtime_archive_path=args.runtime_archive,
                wheelhouse_archive_path=args.wheelhouse_archive,
                closure_package_path=args.closure_package,
                compatibility_threshold_lock_path=(
                    args.compatibility_threshold_lock
                ),
                policy_registry_path=args.policy_registry,
                abr_bundle_archive_path=args.abr_bundle_archive,
                expected_closure_sha256=args.expected_closure_sha256,
                expected_abr_bundle_archive_sha256=(
                    args.expected_abr_bundle_archive_sha256
                ),
                expected_abr_bundle_file_sha256=(
                    args.expected_abr_bundle_file_sha256
                ),
                expected_abr_production_build_receipt_sha256=(
                    args.expected_abr_production_build_receipt_sha256
                ),
            )
            _write_once(Path(args.output), result)
        elif args.command == "local-smoke":
            result = run_local_smoke(
                contract_path=args.contract,
                staging_root=args.staging_root,
                output_root=args.output_root,
                receipt_path=args.receipt,
            )
        elif args.command == "authorize-cloud-plan":
            config = _provider_config(args.provider_config)
            result = core.build_cloud_plan(
                contract_path=args.contract,
                smoke_receipt_path=args.smoke_receipt,
                image_self_link=config["image_self_link"],
                image_id=config["image_id"],
                guest_os_features=config["guest_os_features"],
                worker_service_accounts=config["worker_service_accounts"],
            )
            _write_once(Path(args.output), result)
        elif args.command == "prepare-controller":
            result = prepare_controller(
                cloud_plan=core.validate_cloud_plan(
                    _read(args.cloud_plan, "cloud plan"), replay_sources=True
                ),
                output_root=args.output_root,
                raw_controller_token=_token(
                    os.environ.get(core.CONTROLLER_TOKEN_ENV)
                ),
            )
        elif args.command == "execute":
            nonce = os.environ.get("OFC_M31_POST_TRAINING_LAUNCH_NONCE")
            if nonce is None:
                raise PermissionError(
                    "OFC_M31_POST_TRAINING_LAUNCH_NONCE is required"
                )
            result = execute_next_wave(
                args.output_root,
                confirm_run_name=args.confirm_run_name,
                allow_cloud_mutation=args.allow_cloud_mutation,
                raw_nonce=nonce,
            )
        elif args.command == "poll":
            result = poll_next_wave(
                args.output_root,
                confirm_run_name=args.confirm_run_name,
                allow_cloud_read=args.allow_cloud_read,
            )
        elif args.command == "cleanup":
            result = cleanup_next_wave(
                args.output_root,
                confirm_run_name=args.confirm_run_name,
                allow_cloud_mutation=args.allow_cloud_mutation,
                allow_incomplete=args.allow_incomplete,
                recover_missing_launch=args.recover_missing_launch,
            )
        elif args.command == "receive":
            result = receive_next_wave(
                args.output_root,
                confirm_run_name=args.confirm_run_name,
                allow_cloud_read=args.allow_cloud_read,
            )
        elif args.command == "status":
            result = controller_status(args.output_root)
        elif args.command == "plan":
            value = plan_next_wave(args.output_root)
            result = value if isinstance(value, dict) else dict(value.wave)
        elif args.command == "sentinel":
            result = {"sentinel": phase_sentinel(args.output_root, args.phase)}
        elif args.command == "accept":
            result = accept_next_wave(args.output_root)
        elif args.command == "finalize":
            result = finalize_received(args.output_root)
        elif args.command == "oauth-preflight":
            result = oauth_preflight(expires_at_unix=args.expires_at_unix)
        else:  # pragma: no cover
            raise ValueError("unknown command")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(core.canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
