"""Persistent controller for the M3.1 parallel-20 transport.

The controller is cloud-neutral until an explicit phase method is called with
the exact run name, allow flag, phase sentinel, controller token, and in-memory
OAuth bearer token.  It stores no bearer token and never creates service
accounts.  Accepted lifecycle receipts are immutable and source-replayed to
derive retry/preemption state.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_gcp_provider_v2 as provider
from . import hu_m31_t3_dataset_gcp_transport_v2 as bridge


CONTROLLER_SCHEMA = "hu_m31_t3_dataset_gcp_controller_v2"
PROVIDER_CONFIG_SCHEMA = "hu_m31_t3_dataset_gcp_provider_config_v2"
CONTROLLER_TOKEN_ENV = "OFC_M31_DATASET_V2_CONTROLLER_TOKEN"
OAUTH_TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
PHASE_SENTINEL_ENV = "OFC_M31_DATASET_V2_PHASE_SENTINEL"
LAUNCH_NONCE_ENV = "OFC_M31_DATASET_V2_LAUNCH_NONCE"

_CONTRACT_FILE = "controller-contract.json"
_PLAN_FILE = "transport-plan.json"
_CONFIG_FILE = "provider-config.json"
_RELOCATION_FILE = "source-ext4-relocation.json"
_ACCEPTED_DIRECTORY = "accepted-lifecycles"
_WAVES_DIRECTORY = "waves"


def canonical_bytes(value: Any) -> bytes:
    return bridge.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return bridge.canonical_sha256(value)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    raw = canonical_bytes(value)
    if target.exists() or target.is_symlink():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise FileExistsError(f"immutable controller artifact conflicts: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _root(path: str | Path, *, create: bool) -> Path:
    root = Path(path).resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("parallel20 controller root is unsafe")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise ValueError("parallel20 controller root is missing")
    return root


def _token(raw: str | None) -> str:
    if raw is None or raw != raw.strip():
        raise PermissionError("parallel20 controller token is missing")
    try:
        value = uuid.UUID(raw)
    except ValueError as exc:
        raise PermissionError("parallel20 controller token must be UUIDv4") from exc
    if value.version != 4 or str(value) != raw:
        raise PermissionError("parallel20 controller token must be canonical UUIDv4")
    return raw


def _provider_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(value))
    accounts = config.get("worker_service_accounts")
    if (
        config.get("schema") != PROVIDER_CONFIG_SCHEMA
        or not isinstance(config.get("image_self_link"), str)
        or not isinstance(config.get("image_id"), str)
        or not isinstance(config.get("guest_os_features"), list)
        or not isinstance(accounts, list)
        or len(accounts) != bridge.MAX_CONCURRENT_VMS
        or len(set(accounts)) != len(accounts)
        or config.get("service_accounts_preexisting") is not True
        or config.get("service_account_creation_authorized") is not False
        or config.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 provider config changed")
    return config


def initialize_controller(
    *,
    transport_plan: Mapping[str, Any],
    source_relocation_receipt: Mapping[str, Any],
    provider_config: Mapping[str, Any],
    local_shard_root: str | Path,
    output_root: str | Path,
    raw_controller_token: str,
) -> dict[str, Any]:
    plan = bridge.validate_transport_plan(transport_plan, replay_sources=True)
    relocation = bridge.validate_source_relocation_receipt(
        source_relocation_receipt
    )
    config = _provider_config(provider_config)
    token = _token(raw_controller_token)
    root = _root(output_root, create=True)
    local = Path(local_shard_root).resolve()
    if local.exists() and (local.is_symlink() or not local.is_dir()):
        raise ValueError("parallel20 local shard root is unsafe")
    local.mkdir(parents=True, exist_ok=True)
    if (
        plan["source_ext4_relocation_receipt_sha256"]
        != relocation["receipt_sha256"]
    ):
        raise ValueError("controller relocation receipt differs from plan")
    core = {
        "schema": CONTROLLER_SCHEMA,
        "status": "parallel20_controller_ready_cloud_not_started",
        "run_name": plan["run_name"],
        "transport_plan_sha256": plan["plan_sha256"],
        "source_ext4_relocation_receipt_sha256": relocation["receipt_sha256"],
        "provider_config_sha256": canonical_sha256(config),
        "local_shard_root": str(local),
        "controller_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "controller_token_environment": CONTROLLER_TOKEN_ENV,
        "oauth_token_environment": OAUTH_TOKEN_ENV,
        "phase_sentinel_environment": PHASE_SENTINEL_ENV,
        "launch_nonce_environment": LAUNCH_NONCE_ENV,
        "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
        "max_concurrent_vcpus": bridge.MAX_CONCURRENT_VCPUS,
        "wave_count": bridge.WAVE_COUNT,
        "service_accounts_created": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    contract = {**core, "contract_sha256": canonical_sha256(core)}
    _write_once(root / _PLAN_FILE, plan)
    _write_once(root / _RELOCATION_FILE, relocation)
    _write_once(root / _CONFIG_FILE, config)
    _write_once(root / _CONTRACT_FILE, contract)
    (root / _ACCEPTED_DIRECTORY).mkdir(exist_ok=True)
    (root / _WAVES_DIRECTORY).mkdir(exist_ok=True)
    return contract


def _load_controller(
    output_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _root(output_root, create=False)
    contract = _read_canonical(root / _CONTRACT_FILE, "parallel20 controller")
    plan = bridge.validate_transport_plan(
        _read_canonical(root / _PLAN_FILE, "parallel20 transport plan"),
        replay_sources=True,
    )
    relocation = bridge.validate_source_relocation_receipt(
        _read_canonical(root / _RELOCATION_FILE, "parallel20 relocation receipt")
    )
    config = _provider_config(
        _read_canonical(root / _CONFIG_FILE, "parallel20 provider config")
    )
    if (
        contract.get("contract_sha256") != _self_digest(contract, "contract_sha256")
        or contract.get("schema") != CONTROLLER_SCHEMA
        or contract.get("transport_plan_sha256") != plan["plan_sha256"]
        or contract.get("source_ext4_relocation_receipt_sha256")
        != relocation["receipt_sha256"]
        or contract.get("provider_config_sha256") != canonical_sha256(config)
        or contract.get("max_concurrent_vms") != bridge.MAX_CONCURRENT_VMS
        or contract.get("service_accounts_created") is not False
        or contract.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 controller contract changed")
    return root, contract, plan, relocation, config


def _require_controller_token(contract: Mapping[str, Any]) -> str:
    token = _token(os.environ.get(CONTROLLER_TOKEN_ENV))
    if hashlib.sha256(token.encode("ascii")).hexdigest() != contract[
        "controller_token_sha256"
    ]:
        raise PermissionError("parallel20 controller token differs")
    return token


def _accepted_paths(root: Path) -> list[Path]:
    directory = root / _ACCEPTED_DIRECTORY
    paths = sorted(directory.glob("*.json"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("parallel20 accepted lifecycle directory is unsafe")
    return paths


@dataclass(frozen=True)
class WaveContext:
    root: Path
    directory: Path
    contract: dict[str, Any]
    transport_plan: dict[str, Any]
    ledger: dict[str, Any]
    resume: dict[str, Any]
    wave_request: dict[str, Any]
    provider_plan: dict[str, Any]


def plan_next_wave(output_root: str | Path) -> WaveContext | dict[str, Any]:
    root, contract, plan, _relocation, config = _load_controller(output_root)
    ledger = bridge.build_attempt_ledger(
        plan, lifecycle_receipt_paths=_accepted_paths(root)
    )
    resume = bridge.build_resume_plan(plan, ledger)
    if resume["status"] != "wave_ready":
        return {
            "status": resume["status"],
            "resume": resume,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
    request = bridge.build_wave_request(plan, ledger, resume)
    provider_plan = provider.build_provider_plan(
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        image_self_link=config["image_self_link"],
        image_id=config["image_id"],
        guest_os_features=config["guest_os_features"],
        worker_service_accounts=config["worker_service_accounts"],
        local_shard_root=contract["local_shard_root"],
    )
    directory = root / _WAVES_DIRECTORY / (
        f"wave-{request['wave_index']:02d}-{request['request_sha256'][:16]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    _write_once(directory / "ledger.json", ledger)
    _write_once(directory / "resume.json", resume)
    _write_once(directory / "wave-request.json", request)
    _write_once(directory / "provider-plan.json", provider_plan)
    return WaveContext(
        root=root,
        directory=directory,
        contract=contract,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        provider_plan=provider_plan,
    )


def _context(output_root: str | Path) -> WaveContext:
    value = plan_next_wave(output_root)
    if not isinstance(value, WaveContext):
        raise PermissionError(f"parallel20 next wave is not ready: {value['status']}")
    return value


def expected_phase_sentinel(context: WaveContext, phase: str) -> str:
    if phase not in {
        "preflight",
        "stage",
        "install-iam",
        "execute",
        "poll",
        "cleanup",
        "receive",
    }:
        raise ValueError("parallel20 phase is unknown")
    return (
        f"{phase.upper()}_M31_DATASET_V2:"
        f"{context.contract['run_name']}:"
        f"{context.wave_request['request_sha256']}"
    )


def _require_phase(
    context: WaveContext,
    *,
    phase: str,
    confirm_run_name: str,
    allow: bool,
) -> None:
    _require_controller_token(context.contract)
    if not allow:
        raise PermissionError(f"parallel20 {phase} requires explicit allow flag")
    if confirm_run_name != context.contract["run_name"]:
        raise PermissionError("parallel20 confirmed run name differs")
    if os.environ.get(PHASE_SENTINEL_ENV) != expected_phase_sentinel(context, phase):
        raise PermissionError("parallel20 phase sentinel differs")


def _live_transport() -> provider.GcpParallelDatasetRestAdapter:
    token = os.environ.get(OAUTH_TOKEN_ENV)
    if (
        token is None
        or token != token.strip()
        or len(token) < 20
        or any(character.isspace() for character in token)
    ):
        raise PermissionError(f"{OAUTH_TOKEN_ENV} must contain one bearer token")
    return provider.GcpParallelDatasetRestAdapter(access_token=token)


def readonly_preflight_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="preflight",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_read,
    )
    transport = cloud or _live_transport()
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    observed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    receipt = provider.build_readonly_preflight(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        transport=transport,
        observed_at_utc=observed,
    )
    _write_once(context.directory / "readonly-preflight.json", receipt)
    return receipt


def _receipt(context: WaveContext, filename: str, label: str) -> dict[str, Any]:
    return _read_canonical(context.directory / filename, label)


def _require_preflight_receipt(context: WaveContext) -> dict[str, Any]:
    receipt = _receipt(
        context, "readonly-preflight.json", "parallel20 readonly preflight"
    )
    return provider.validate_readonly_preflight(
        receipt, provider_plan=context.provider_plan
    )


def stage_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="stage",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_mutation,
    )
    _require_preflight_receipt(context)
    receipt = provider.stage_content(
        provider_plan=context.provider_plan,
        transport=cloud or _live_transport(),
    )
    _write_once(context.directory / "stage-receipt.json", receipt)
    return receipt


def install_iam_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="install-iam",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_mutation,
    )
    _require_preflight_receipt(context)
    receipt = provider.install_worker_iam(
        provider_plan=context.provider_plan,
        transport=cloud or _live_transport(),
        now_unix_seconds=(
            int(time.time()) if now_unix_seconds is None else now_unix_seconds
        ),
    )
    _write_once(context.directory / "iam-receipt.json", receipt)
    return receipt


def execute_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    raw_nonce: str,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
    now_unix_seconds: int | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="execute",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_mutation,
    )
    preflight = _require_preflight_receipt(context)
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    observed_epoch = calendar.timegm(
        time.strptime(preflight["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    )
    if now - observed_epoch > 300 or now < observed_epoch:
        raise PermissionError("parallel20 OAuth/quota preflight is older than 5 minutes")
    result = provider.execute_wave_atomic(
        provider_plan=context.provider_plan,
        preflight_receipt=preflight,
        stage_receipt=_receipt(
            context, "stage-receipt.json", "parallel20 stage receipt"
        ),
        iam_receipt=_receipt(
            context, "iam-receipt.json", "parallel20 IAM receipt"
        ),
        transport=cloud or _live_transport(),
        raw_nonce=raw_nonce,
        sleep=sleep,
    )
    _write_once(context.directory / "launch-receipt.json", result)
    if result["status"] == "partial_launch_cleaned_no_go":
        rows = [
            {
                "shard_id": selected["shard_id"],
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "status": "failed",
                "completed_pair_count": 0,
                "checkpoint": None,
                "heartbeat": None,
                "owned_compute_absent": True,
                "worker_iam_removed": True,
            }
            for selected in context.wave_request["selected_attempts"]
        ]
        lifecycle = bridge.build_lifecycle_receipt(
            plan=context.transport_plan,
            ledger=context.ledger,
            resume=context.resume,
            wave_request=context.wave_request,
            attempt_rows=rows,
        )
        _write_once(context.directory / "lifecycle-receipt.json", lifecycle)
    return result


def poll_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="poll",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_read,
    )
    result = provider.poll_wave(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        transport=cloud or _live_transport(),
    )
    # Poll snapshots are mutable observations, not accepted science.  Keep
    # each content-addressed snapshot instead of overwriting a latest file.
    _write_once(
        context.directory / f"poll-{result['receipt_sha256']}.json", result
    )
    return result


def cleanup_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="cleanup",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_mutation,
    )
    transport = cloud or _live_transport()
    result = provider.cleanup_wave(
        provider_plan=context.provider_plan,
        stage_receipt=_receipt(
            context, "stage-receipt.json", "parallel20 stage receipt"
        ),
        iam_receipt=_receipt(
            context, "iam-receipt.json", "parallel20 IAM receipt"
        ),
        transport=transport,
        sleep=sleep,
    )
    _write_once(context.directory / "cleanup-receipt.json", result)
    lifecycle = provider.build_lifecycle_after_cleanup(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        cleanup_receipt=result,
        transport=transport,
    )
    _write_once(context.directory / "lifecycle-receipt.json", lifecycle)
    return result


def receive_next(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: provider.ParallelDatasetCloudTransport | None = None,
) -> dict[str, Any]:
    context = _context(output_root)
    _require_phase(
        context,
        phase="receive",
        confirm_run_name=confirm_run_name,
        allow=allow_cloud_read,
    )
    lifecycle = _receipt(
        context, "lifecycle-receipt.json", "parallel20 lifecycle receipt"
    )
    transport = cloud or _live_transport()
    mirror = provider.mirror_lifecycle_objects(
        transport_plan=context.transport_plan,
        lifecycle_receipt=lifecycle,
        bucket=context.provider_plan["bucket"],
        output_root=context.directory / "object-mirror",
        transport=transport,
    )
    _write_once(context.directory / "mirror-receipt.json", mirror)
    received = bridge.receive_wave(
        plan=context.transport_plan,
        lifecycle_receipt=lifecycle,
        object_root=mirror["object_root"],
        local_root=context.contract["local_shard_root"],
    )
    _write_once(context.directory / "receive-receipt.json", received)
    return received


def accept_next(output_root: str | Path) -> dict[str, Any]:
    context = _context(output_root)
    lifecycle_path = context.directory / "lifecycle-receipt.json"
    lifecycle = _read_canonical(
        lifecycle_path, "parallel20 lifecycle receipt"
    )
    received = _receipt(
        context, "receive-receipt.json", "parallel20 receive receipt"
    )
    if (
        received.get("schema") != bridge.RECEIVE_SCHEMA
        or received.get("plan_sha256")
        != context.transport_plan["plan_sha256"]
        or received.get("lifecycle_receipt_sha256")
        != lifecycle["receipt_sha256"]
        or received.get("received_count")
        != context.wave_request["selected_count"]
        or received.get("source_replayed_v1_shard_bytes") is not True
        or received.get("current_profile_changed") is not False
    ):
        raise PermissionError("parallel20 receive receipt does not open acceptance")
    return accept_lifecycle_receipt(
        output_root, lifecycle_receipt_path=lifecycle_path
    )


def accept_lifecycle_receipt(
    output_root: str | Path, *, lifecycle_receipt_path: str | Path
) -> dict[str, Any]:
    context = _context(output_root)
    lifecycle = bridge.validate_lifecycle_receipt(
        _read_canonical(lifecycle_receipt_path, "parallel20 lifecycle receipt"),
        plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
    )
    target = (
        context.root
        / _ACCEPTED_DIRECTORY
        / f"{len(_accepted_paths(context.root)):04d}-{lifecycle['receipt_sha256']}.json"
    )
    _write_once(target, lifecycle)
    next_value = plan_next_wave(output_root)
    return {
        "status": "parallel20_lifecycle_accepted",
        "receipt_sha256": lifecycle["receipt_sha256"],
        "accepted_path": str(target),
        "next_status": (
            "wave_ready" if isinstance(next_value, WaveContext) else next_value["status"]
        ),
        "current_profile_changed": False,
    }


def controller_status(output_root: str | Path) -> dict[str, Any]:
    root, contract, plan, relocation, config = _load_controller(output_root)
    ledger = bridge.build_attempt_ledger(
        plan, lifecycle_receipt_paths=_accepted_paths(root)
    )
    resume = bridge.build_resume_plan(plan, ledger)
    return {
        "schema": "hu_m31_t3_dataset_gcp_controller_status_v2",
        "run_name": contract["run_name"],
        "transport_plan_sha256": plan["plan_sha256"],
        "scientific_v1_plan_sha256": plan["scientific_v1_lock"][
            "transport_v1_plan_sha256"
        ],
        "source_ext4_relocation_receipt_sha256": relocation["receipt_sha256"],
        "worker_service_account_count": len(config["worker_service_accounts"]),
        "accepted_lifecycle_count": len(_accepted_paths(root)),
        "complete_shard_count": ledger["complete_shard_count"],
        "pending_shard_count": ledger["pending_shard_count"],
        "exhausted_shard_count": ledger["exhausted_shard_count"],
        "resume_status": resume["status"],
        "resume_wave_index": resume["resume_wave_index"],
        "next_selected_count": resume["selected_count"],
        "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
        "max_concurrent_vcpus": bridge.MAX_CONCURRENT_VCPUS,
        "wave_count": bridge.WAVE_COUNT,
        "cloud_mutated": False,
        "service_accounts_created": False,
        "current_profile_changed": False,
    }


def _print(value: Any) -> None:
    if isinstance(value, WaveContext):
        value = {
            "wave_request": value.wave_request,
            "provider_plan": value.provider_plan,
            "directory": str(value.directory),
        }
    print(canonical_bytes(value).decode("ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 immutable dataset parallel-20 GCP controller"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--transport-plan", required=True)
    prepare.add_argument("--source-relocation-receipt", required=True)
    prepare.add_argument("--provider-config", required=True)
    prepare.add_argument("--local-shard-root", required=True)
    prepare.add_argument("--output-root", required=True)
    for name in ("plan-next", "status"):
        command = sub.add_parser(name)
        command.add_argument("--output-root", required=True)
    for name in (
        "preflight-next",
        "stage-next",
        "install-iam-next",
        "poll-next",
        "cleanup-next",
        "receive-next",
    ):
        command = sub.add_parser(name)
        command.add_argument("--output-root", required=True)
        command.add_argument("--confirm-run-name", required=True)
        command.add_argument(
            (
                "--allow-cloud-read"
                if name in {"preflight-next", "poll-next", "receive-next"}
                else "--allow-cloud-mutation"
            ),
            action="store_true",
        )
    execute = sub.add_parser("execute-next")
    execute.add_argument("--output-root", required=True)
    execute.add_argument("--confirm-run-name", required=True)
    execute.add_argument("--allow-cloud-mutation", action="store_true")
    accept = sub.add_parser("accept-lifecycle")
    accept.add_argument("--output-root", required=True)
    accept.add_argument("--lifecycle-receipt", required=True)
    accept_next_parser = sub.add_parser("accept-next")
    accept_next_parser.add_argument("--output-root", required=True)
    sentinel = sub.add_parser("sentinel")
    sentinel.add_argument("--output-root", required=True)
    sentinel.add_argument(
        "--phase",
        required=True,
        choices=(
            "preflight",
            "stage",
            "install-iam",
            "execute",
            "poll",
            "cleanup",
            "receive",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            _print(
                initialize_controller(
                    transport_plan=_read_canonical(
                        args.transport_plan, "parallel20 transport plan"
                    ),
                    source_relocation_receipt=_read_canonical(
                        args.source_relocation_receipt,
                        "parallel20 source relocation receipt",
                    ),
                    provider_config=_read_canonical(
                        args.provider_config, "parallel20 provider config"
                    ),
                    local_shard_root=args.local_shard_root,
                    output_root=args.output_root,
                    raw_controller_token=_token(
                        os.environ.get(CONTROLLER_TOKEN_ENV)
                    ),
                )
            )
        elif args.command == "plan-next":
            _print(plan_next_wave(args.output_root))
        elif args.command == "status":
            _print(controller_status(args.output_root))
        elif args.command == "preflight-next":
            _print(
                readonly_preflight_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_read=args.allow_cloud_read,
                )
            )
        elif args.command == "stage-next":
            _print(
                stage_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_mutation=args.allow_cloud_mutation,
                )
            )
        elif args.command == "install-iam-next":
            _print(
                install_iam_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_mutation=args.allow_cloud_mutation,
                )
            )
        elif args.command == "execute-next":
            _print(
                execute_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_mutation=args.allow_cloud_mutation,
                    raw_nonce=_token(os.environ.get(LAUNCH_NONCE_ENV)),
                )
            )
        elif args.command == "poll-next":
            _print(
                poll_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_read=args.allow_cloud_read,
                )
            )
        elif args.command == "cleanup-next":
            _print(
                cleanup_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_mutation=args.allow_cloud_mutation,
                )
            )
        elif args.command == "receive-next":
            _print(
                receive_next(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_read=args.allow_cloud_read,
                )
            )
        elif args.command == "accept-lifecycle":
            _print(
                accept_lifecycle_receipt(
                    args.output_root,
                    lifecycle_receipt_path=args.lifecycle_receipt,
                )
            )
        elif args.command == "accept-next":
            _print(accept_next(args.output_root))
        elif args.command == "sentinel":
            context = _context(args.output_root)
            _print(
                {
                    "environment": PHASE_SENTINEL_ENV,
                    "phase": args.phase,
                    "sentinel": expected_phase_sentinel(context, args.phase),
                }
            )
        else:  # pragma: no cover
            raise RuntimeError("unknown parallel20 controller command")
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_SCHEMA",
    "CONTROLLER_TOKEN_ENV",
    "OAUTH_TOKEN_ENV",
    "PHASE_SENTINEL_ENV",
    "LAUNCH_NONCE_ENV",
    "PROVIDER_CONFIG_SCHEMA",
    "WaveContext",
    "accept_lifecycle_receipt",
    "accept_next",
    "cleanup_next",
    "controller_status",
    "execute_next",
    "expected_phase_sentinel",
    "initialize_controller",
    "install_iam_next",
    "main",
    "plan_next_wave",
    "poll_next",
    "receive_next",
    "readonly_preflight_next",
    "stage_next",
]
