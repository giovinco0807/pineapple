"""Resumable one-wave controller for the M3.1 9,000-pair GCP dataset.

The transport and provider modules intentionally expose small, injected
operations.  This module supplies the missing production entry point without
weakening those contracts:

* ``prepare`` source-replays the qualified fresh-quality gate and completed
  25-pair smoke shard before it can persist a transport plan;
* ``plan-next`` derives only the earliest incomplete wave from accepted
  lifecycle receipts and never selects more than eight C4 workers;
* cloud reads and mutations require a hash-bound controller token, an exact
  phase sentinel, an exact run-name confirmation, and an explicit allow flag;
* stage, IAM, launch, poll, cleanup, mirror, receive, acceptance, and final
  merge receipts are create-only;
* a lifecycle receipt becomes part of the attempt ledger only after its
  checkpoint files have been independently received into the local shard root;
* ``finalize`` is local-only and writes the exact 360-shard map and merge.

OAuth credentials are accepted only from ``GOOGLE_OAUTH_ACCESS_TOKEN`` and
are never serialized.  Importing this module, preparing a controller, planning
a wave, accepting a received wave, or finalizing a dataset makes no cloud
request.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_gcp_provider_v1 as provider
from . import hu_m31_t3_dataset_gcp_transport_v1 as bridge
from . import hu_m31_t3_dataset_portable_worker_v1 as portable


CONTROLLER_CONTRACT_SCHEMA = "hu_m31_t3_dataset_gcp_controller_contract_v1"
PROVIDER_CONFIG_SCHEMA = "hu_m31_t3_dataset_gcp_provider_config_v1"
WAVE_CONTEXT_SCHEMA = "hu_m31_t3_dataset_gcp_wave_context_v1"
JOURNAL_EVENT_SCHEMA = "hu_m31_t3_dataset_gcp_controller_event_v1"
ACCEPTANCE_SCHEMA = "hu_m31_t3_dataset_gcp_wave_acceptance_v1"
FINAL_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_final_receipt_v1"

CONTROLLER_TOKEN_ENV = "OFC_M31_DATASET_CONTROLLER_TOKEN"
PHASE_SENTINEL_ENV = "OFC_M31_DATASET_PHASE_SENTINEL"
OAUTH_TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PHASE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_ACCEPTED_NAME = re.compile(r"^(\d{6})-([0-9a-f]{64})\.json$")
_POLL_NAME = re.compile(r"^(\d{6})-([0-9a-f]{64})\.json$")


def canonical_bytes(value: Any) -> bytes:
    return bridge.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return bridge.canonical_sha256(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_dataset_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("dataset plan must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("dataset plan is not canonical M3.1 JSON") from exc
    if not isinstance(value, dict) or raw != dataset.canonical_bytes(value):
        raise ValueError("dataset plan is not canonical M3.1 JSON")
    return dataset.validate_dataset_plan(value)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes(value)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_replay(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.exists() or path.is_symlink():
        if _read_canonical(path, label) != dict(value):
            raise FileExistsError(f"{label} conflicts with immutable artifact")
        return
    _write_once(path, value)


def _controller_root(path: str | Path, *, create: bool) -> Path:
    root = Path(path).resolve()
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("controller root is unsafe")
    elif create:
        root.mkdir(parents=True)
    else:
        raise ValueError("controller root does not exist")
    return root


def _token_value(raw: str | None) -> str:
    if raw is None or raw != raw.strip():
        raise PermissionError(
            f"{CONTROLLER_TOKEN_ENV} must be a canonical UUIDv4"
        )
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise PermissionError(
            f"{CONTROLLER_TOKEN_ENV} must be a canonical UUIDv4"
        ) from exc
    if parsed.version != 4 or str(parsed) != raw:
        raise PermissionError(
            f"{CONTROLLER_TOKEN_ENV} must be a canonical UUIDv4"
        )
    return raw


def _provider_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(value))
    required = {
        "schema",
        "image_self_link",
        "image_id",
        "guest_os_features",
        "worker_service_accounts",
        "current_profile_changed",
    }
    if (
        set(config) != required
        or config.get("schema") != PROVIDER_CONFIG_SCHEMA
        or not isinstance(config.get("image_self_link"), str)
        or not isinstance(config.get("image_id"), str)
        or not isinstance(config.get("guest_os_features"), list)
        or not all(
            isinstance(item, str) for item in config["guest_os_features"]
        )
        or not isinstance(config.get("worker_service_accounts"), list)
        or len(config["worker_service_accounts"]) != bridge.MAX_CONCURRENT_VMS
        or not all(
            isinstance(item, str)
            for item in config["worker_service_accounts"]
        )
        or config.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset provider config contract changed")
    return config


def _load_provider_config(path: str | Path) -> dict[str, Any]:
    return _provider_config(_read_canonical(path, "dataset provider config"))


def _validate_contract(
    value: Mapping[str, Any], *, transport_plan: Mapping[str, Any]
) -> dict[str, Any]:
    contract = deepcopy(dict(value))
    required = {
        "schema",
        "status",
        "transport_plan_sha256",
        "execution_identity_sha256",
        "run_name",
        "bucket",
        "provider_config",
        "local_shard_root",
        "controller_token_sha256",
        "controller_token_environment",
        "phase_sentinel_environment",
        "oauth_token_environment",
        "credentials_from_environment_only",
        "wave_count",
        "max_concurrent_vms",
        "machine_type",
        "quality_and_smoke_source_replayed",
        "full_fanout_authorized_after_smoke_only",
        "current_profile_changed",
        "contract_sha256",
    }
    local_root = Path(str(contract.get("local_shard_root", "")))
    if (
        set(contract) != required
        or contract.get("contract_sha256")
        != _self_digest(contract, "contract_sha256")
        or contract.get("schema") != CONTROLLER_CONTRACT_SCHEMA
        or contract.get("status") != "prepared_cloud_not_started"
        or contract.get("transport_plan_sha256")
        != transport_plan["plan_sha256"]
        or contract.get("execution_identity_sha256")
        != transport_plan["execution_identity_sha256"]
        or contract.get("run_name") != transport_plan["run_name"]
        or contract.get("bucket") != transport_plan["bucket"]
        or _SHA.fullmatch(str(contract.get("controller_token_sha256", "")))
        is None
        or not local_root.is_absolute()
        or contract.get("controller_token_environment") != CONTROLLER_TOKEN_ENV
        or contract.get("phase_sentinel_environment") != PHASE_SENTINEL_ENV
        or contract.get("oauth_token_environment") != OAUTH_TOKEN_ENV
        or contract.get("credentials_from_environment_only") is not True
        or contract.get("wave_count") != bridge.WAVE_COUNT
        or contract.get("max_concurrent_vms") != bridge.MAX_CONCURRENT_VMS
        or contract.get("machine_type") != bridge.MACHINE_TYPE
        or contract.get("quality_and_smoke_source_replayed") is not True
        or contract.get("full_fanout_authorized_after_smoke_only") is not True
        or contract.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset controller contract changed")
    contract["provider_config"] = _provider_config(contract["provider_config"])
    return contract


def _load_controller(
    output_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = _controller_root(output_root, create=False)
    transport_plan = bridge.validate_transport_plan(
        _read_canonical(root / "transport_plan.json", "dataset transport plan"),
        replay_sources=True,
    )
    contract = _validate_contract(
        _read_canonical(
            root / "controller_contract.json", "dataset controller contract"
        ),
        transport_plan=transport_plan,
    )
    return root, contract, transport_plan


def _require_controller_token(contract: Mapping[str, Any]) -> str:
    raw = _token_value(os.environ.get(CONTROLLER_TOKEN_ENV))
    if _sha256(raw.encode("ascii")) != contract["controller_token_sha256"]:
        raise PermissionError("dataset controller token does not match its hash")
    return raw


def _event_files(root: Path) -> list[Path]:
    journal = root / "journal"
    if not journal.exists():
        return []
    if journal.is_symlink() or not journal.is_dir():
        raise ValueError("dataset controller journal is unsafe")
    paths = sorted(journal.glob("*.json"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("dataset controller journal contains an unsafe entry")
    if len(paths) != len(list(journal.iterdir())):
        raise ValueError("dataset controller journal contains an unknown entry")
    return paths


def _load_events(root: Path) -> list[dict[str, Any]]:
    events = []
    predecessor = None
    seen_operations: set[str] = set()
    required = {
        "schema",
        "sequence",
        "phase",
        "operation_key",
        "predecessor_event_sha256",
        "payload",
        "payload_sha256",
        "cloud_mutated",
        "recorded_at_utc",
        "current_profile_changed",
        "event_sha256",
    }
    for sequence, path in enumerate(_event_files(root)):
        value = _read_canonical(path, "dataset controller event")
        if (
            set(value) != required
            or value.get("schema") != JOURNAL_EVENT_SCHEMA
            or value.get("sequence") != sequence
            or value.get("predecessor_event_sha256") != predecessor
            or value.get("payload_sha256")
            != canonical_sha256(value.get("payload"))
            or value.get("event_sha256")
            != _self_digest(value, "event_sha256")
            or not isinstance(value.get("operation_key"), str)
            or value["operation_key"] in seen_operations
            or _PHASE.fullmatch(str(value.get("phase", ""))) is None
            or not isinstance(value.get("cloud_mutated"), bool)
            or value.get("current_profile_changed") is not False
        ):
            raise ValueError("dataset controller journal chain changed")
        expected_name = (
            f"{sequence:06d}-{value['phase']}-{value['event_sha256'][:16]}.json"
        )
        if path.name != expected_name:
            raise ValueError("dataset controller event filename changed")
        events.append(value)
        predecessor = value["event_sha256"]
        seen_operations.add(value["operation_key"])
    return events


def _append_event(
    root: Path,
    *,
    phase: str,
    operation_key: str,
    payload: Mapping[str, Any],
    cloud_mutated: bool,
) -> dict[str, Any]:
    if _PHASE.fullmatch(phase) is None or not operation_key:
        raise ValueError("dataset controller event identity is invalid")
    events = _load_events(root)
    for value in events:
        if value["operation_key"] == operation_key:
            if (
                value["phase"] != phase
                or value["payload"] != dict(payload)
                or value["cloud_mutated"] is not cloud_mutated
            ):
                raise FileExistsError("dataset controller operation conflicts")
            return value
    core = {
        "schema": JOURNAL_EVENT_SCHEMA,
        "sequence": len(events),
        "phase": phase,
        "operation_key": operation_key,
        "predecessor_event_sha256": (
            events[-1]["event_sha256"] if events else None
        ),
        "payload": deepcopy(dict(payload)),
        "payload_sha256": canonical_sha256(payload),
        "cloud_mutated": cloud_mutated,
        "recorded_at_utc": _utc_now(),
        "current_profile_changed": False,
    }
    value = {**core, "event_sha256": canonical_sha256(core)}
    path = (
        root
        / "journal"
        / f"{len(events):06d}-{phase}-{value['event_sha256'][:16]}.json"
    )
    _write_once(path, value)
    return value


def initialize_controller_from_transport_plan(
    *,
    transport_plan: Mapping[str, Any],
    provider_config: Mapping[str, Any],
    local_shard_root: str | Path,
    output_root: str | Path,
    raw_controller_token: str,
) -> dict[str, Any]:
    """Persist a cloud-neutral controller around an already qualified plan."""

    token = _token_value(raw_controller_token)
    plan = bridge.validate_transport_plan(transport_plan, replay_sources=True)
    if (
        plan["quality_and_smoke_source_replayed"] is not True
        or plan["precompleted_smoke_shard_id"] != dataset.SMOKE_SHARD_ID
        or plan["cloud_launch_authorized"] is not True
        or plan["wave_count"] != bridge.WAVE_COUNT
        or plan["machine_contract"]["max_concurrent_vms"]
        != bridge.MAX_CONCURRENT_VMS
    ):
        raise PermissionError(
            "dataset controller requires qualified fresh quality plus smoke"
        )
    config = _provider_config(provider_config)
    local_root = Path(local_shard_root).resolve()
    if local_root.is_symlink() or (local_root.exists() and not local_root.is_dir()):
        raise ValueError("local dataset shard root is unsafe")
    root = _controller_root(output_root, create=True)
    _write_or_replay(root / "transport_plan.json", plan, "transport plan")
    core = {
        "schema": CONTROLLER_CONTRACT_SCHEMA,
        "status": "prepared_cloud_not_started",
        "transport_plan_sha256": plan["plan_sha256"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "run_name": plan["run_name"],
        "bucket": plan["bucket"],
        "provider_config": config,
        "local_shard_root": str(local_root),
        "controller_token_sha256": _sha256(token.encode("ascii")),
        "controller_token_environment": CONTROLLER_TOKEN_ENV,
        "phase_sentinel_environment": PHASE_SENTINEL_ENV,
        "oauth_token_environment": OAUTH_TOKEN_ENV,
        "credentials_from_environment_only": True,
        "wave_count": bridge.WAVE_COUNT,
        "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
        "machine_type": bridge.MACHINE_TYPE,
        "quality_and_smoke_source_replayed": True,
        "full_fanout_authorized_after_smoke_only": True,
        "current_profile_changed": False,
    }
    contract = {**core, "contract_sha256": canonical_sha256(core)}
    _write_or_replay(
        root / "controller_contract.json", contract, "controller contract"
    )
    _append_event(
        root,
        phase="prepare",
        operation_key="prepare",
        payload={
            "contract_sha256": contract["contract_sha256"],
            "transport_plan_sha256": plan["plan_sha256"],
            "cloud_launch_authorized": True,
            "cloud_execution_started": False,
        },
        cloud_mutated=False,
    )
    return contract


def prepare_controller(
    *,
    run_name: str,
    bucket: str,
    dataset_plan_path: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_gate_path: str | Path,
    smoke_shard_directory: str | Path,
    smoke_shard_archive_path: str | Path,
    runtime_archive_path: str | Path,
    wheelhouse_archive_path: str | Path,
    candidate_library_path: str | Path,
    provider_config_path: str | Path,
    local_shard_root: str | Path,
    output_root: str | Path,
    raw_controller_token: str,
) -> dict[str, Any]:
    """Build the gate-bound portable authorization and 45-wave plan locally."""

    token = _token_value(raw_controller_token)
    root = _controller_root(output_root, create=True)
    plan = _read_dataset_plan(dataset_plan_path)
    portable_value = portable.build_portable_fanout_authorization(
        plan=plan,
        fresh_quality_gate_path=fresh_quality_gate_path,
        smoke_gate_receipt_path=smoke_gate_path,
        smoke_shard_directory=smoke_shard_directory,
    )
    portable_path = root / "portable_authorization.json"
    _write_or_replay(
        portable_path, portable_value, "portable fanout authorization"
    )
    transport_plan = bridge.build_transport_plan(
        run_name=run_name,
        bucket=bucket,
        dataset_plan_path=dataset_plan_path,
        fresh_quality_gate_path=fresh_quality_gate_path,
        smoke_gate_path=smoke_gate_path,
        portable_authorization_path=portable_path,
        smoke_shard_directory=smoke_shard_directory,
        smoke_shard_archive_path=smoke_shard_archive_path,
        runtime_archive_path=runtime_archive_path,
        wheelhouse_archive_path=wheelhouse_archive_path,
        candidate_library_path=candidate_library_path,
    )
    return initialize_controller_from_transport_plan(
        transport_plan=transport_plan,
        provider_config=_load_provider_config(provider_config_path),
        local_shard_root=local_shard_root,
        output_root=root,
        raw_controller_token=token,
    )


def _accepted_paths(root: Path) -> list[Path]:
    directory = root / "accepted"
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("accepted lifecycle directory is unsafe")
    paths = sorted(directory.iterdir())
    for index, path in enumerate(paths):
        match = _ACCEPTED_NAME.fullmatch(path.name)
        if (
            match is None
            or int(match.group(1)) != index
            or path.is_symlink()
            or not path.is_file()
            or bridge.sha256_file(path) != match.group(2)
        ):
            raise ValueError("accepted lifecycle chain changed")
    return paths


@dataclass(frozen=True)
class WaveContext:
    root: Path
    directory: Path
    contract: Mapping[str, Any]
    transport_plan: Mapping[str, Any]
    ledger: Mapping[str, Any]
    resume: Mapping[str, Any]
    wave_request: Mapping[str, Any]
    provider_plan: Mapping[str, Any]
    context: Mapping[str, Any]


def _context_value(
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    request: Mapping[str, Any],
    provider_plan: Mapping[str, Any],
    accepted_count: int,
) -> dict[str, Any]:
    core = {
        "schema": WAVE_CONTEXT_SCHEMA,
        "status": "next_wave_planned_cloud_not_started",
        "contract_sha256": contract["contract_sha256"],
        "transport_plan_sha256": plan["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "request_sha256": request["request_sha256"],
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "wave_index": request["wave_index"],
        "selected_count": request["selected_count"],
        "accepted_lifecycle_count": accepted_count,
        "machine_type": bridge.MACHINE_TYPE,
        "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
        "earliest_incomplete_wave_only": True,
        "quality_and_smoke_source_replayed": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "context_sha256": canonical_sha256(core)}


def plan_next_wave(output_root: str | Path) -> WaveContext | dict[str, Any]:
    """Source-replay accepted receipts and persist the next immutable context."""

    root, contract, plan = _load_controller(output_root)
    _require_controller_token(contract)
    accepted = _accepted_paths(root)
    ledger = bridge.build_attempt_ledger(
        plan, lifecycle_receipt_paths=accepted
    )
    resume = bridge.build_resume_plan(plan, ledger)
    if resume["status"] != "wave_ready":
        result = {
            "status": resume["status"],
            "ledger_sha256": ledger["ledger_sha256"],
            "resume_sha256": resume["resume_sha256"],
            "complete_shard_count": ledger["complete_shard_count"],
            "exhausted_shard_count": ledger["exhausted_shard_count"],
            "accepted_lifecycle_count": len(accepted),
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        _append_event(
            root,
            phase="plan-complete" if resume["status"] == "complete" else "no-go",
            operation_key=f"plan-terminal:{resume['resume_sha256']}",
            payload=result,
            cloud_mutated=False,
        )
        return result
    request = bridge.build_wave_request(plan, ledger, resume)
    config = contract["provider_config"]
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
    if (
        request["selected_count"] > bridge.MAX_CONCURRENT_VMS
        or provider_plan["runtime_contract"]["machine_type"]
        != bridge.MACHINE_TYPE
    ):
        raise PermissionError("next dataset wave exceeds the frozen C4 bound")
    directory = (
        root
        / "waves"
        / f"w{request['wave_index']:02d}-{request['request_sha256'][:16]}"
    )
    context = _context_value(
        contract=contract,
        plan=plan,
        ledger=ledger,
        resume=resume,
        request=request,
        provider_plan=provider_plan,
        accepted_count=len(accepted),
    )
    for filename, value, label in (
        ("ledger.json", ledger, "wave ledger"),
        ("resume.json", resume, "wave resume plan"),
        ("wave_request.json", request, "wave request"),
        ("provider_plan.json", provider_plan, "provider plan"),
        ("context.json", context, "wave context"),
    ):
        _write_or_replay(directory / filename, value, label)
    _append_event(
        root,
        phase="plan-wave",
        operation_key=f"plan-wave:{request['request_sha256']}",
        payload={
            "context_sha256": context["context_sha256"],
            "request_sha256": request["request_sha256"],
            "wave_index": request["wave_index"],
            "selected_count": request["selected_count"],
        },
        cloud_mutated=False,
    )
    return WaveContext(
        root=root,
        directory=directory,
        contract=contract,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        provider_plan=provider_plan,
        context=context,
    )


def _current_context(output_root: str | Path) -> WaveContext:
    value = plan_next_wave(output_root)
    if not isinstance(value, WaveContext):
        raise PermissionError(f"dataset has no runnable wave: {value['status']}")
    return value


def expected_phase_sentinel(context: WaveContext, phase: str) -> str:
    if _PHASE.fullmatch(phase) is None:
        raise ValueError("invalid dataset controller phase")
    return (
        f"m31-dataset:{phase}:{context.contract['run_name']}:"
        f"{context.wave_request['request_sha256']}"
    )


def _require_cloud_phase(
    context: WaveContext,
    *,
    phase: str,
    confirm_run_name: str,
    allow_cloud: bool,
) -> str:
    raw = _require_controller_token(context.contract)
    expected = expected_phase_sentinel(context, phase)
    if not allow_cloud:
        raise PermissionError(f"{phase} requires its explicit cloud allow flag")
    if confirm_run_name != context.contract["run_name"]:
        raise PermissionError("confirmed dataset run name does not match")
    if os.environ.get(PHASE_SENTINEL_ENV) != expected:
        raise PermissionError(
            f"{PHASE_SENTINEL_ENV} must exactly equal the planned phase sentinel"
        )
    return raw


def _live_transport() -> provider.GcpDatasetRestAdapter:
    token = os.environ.get(OAUTH_TOKEN_ENV)
    if (
        token is None
        or token != token.strip()
        or len(token) < 20
        or any(character.isspace() for character in token)
    ):
        raise PermissionError(
            f"{OAUTH_TOKEN_ENV} must contain one in-memory OAuth bearer token"
        )
    return provider.GcpDatasetRestAdapter(access_token=token)


def _wave_receipt(
    context: WaveContext, filename: str, label: str
) -> dict[str, Any] | None:
    path = context.directory / filename
    if not path.exists() and not path.is_symlink():
        return None
    return _read_canonical(path, label)


def _record_execute_event(
    context: WaveContext, launch: Mapping[str, Any]
) -> None:
    _append_event(
        context.root,
        phase="execute",
        operation_key=f"execute:{context.wave_request['request_sha256']}",
        payload={
            "request_sha256": context.wave_request["request_sha256"],
            "stage_receipt_sha256": launch["stage_receipt"]["receipt_sha256"],
            "iam_receipt_sha256": launch["iam_receipt"]["receipt_sha256"],
            "launch_receipt_sha256": launch["receipt_sha256"],
            "created_instance_count": launch["created_instance_count"],
            "at_most_eight_c4": launch["at_most_eight_c4"],
        },
        cloud_mutated=True,
    )


def execute_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    cloud: provider.DatasetCloudTransport | None = None,
    now_unix_seconds: int | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    context = _current_context(output_root)
    raw_token = _require_cloud_phase(
        context,
        phase="execute",
        confirm_run_name=confirm_run_name,
        allow_cloud=allow_cloud_mutation,
    )
    launch_existing = _wave_receipt(
        context, "launch_receipt.json", "dataset launch receipt"
    )
    if launch_existing is not None:
        launch_existing = provider.validate_launch_receipt(
            launch_existing, provider_plan=context.provider_plan
        )
        _record_execute_event(context, launch_existing)
        return launch_existing
    transport = cloud or _live_transport()
    stage = _wave_receipt(
        context, "stage_receipt.json", "dataset stage receipt"
    )
    if stage is None:
        stage = provider.stage_content(
            provider_plan=context.provider_plan,
            transport_plan=context.transport_plan,
            ledger=context.ledger,
            resume=context.resume,
            wave_request=context.wave_request,
            transport=transport,
        )
        _write_once(context.directory / "stage_receipt.json", stage)
    else:
        stage = provider.validate_stage_receipt(
            stage, provider_plan=context.provider_plan
        )
    iam = _wave_receipt(context, "iam_receipt.json", "dataset IAM receipt")
    if iam is None:
        iam = provider.install_worker_iam(
            provider_plan=context.provider_plan,
            transport_plan=context.transport_plan,
            ledger=context.ledger,
            resume=context.resume,
            wave_request=context.wave_request,
            transport=transport,
            now_unix_seconds=(
                int(time.time())
                if now_unix_seconds is None
                else int(now_unix_seconds)
            ),
        )
        _write_once(context.directory / "iam_receipt.json", iam)
    else:
        iam = provider.validate_iam_receipt(
            iam, provider_plan=context.provider_plan
        )
    nonce_raw = hashlib.sha256(
        (
            raw_token
            + "\0"
            + context.wave_request["request_sha256"]
        ).encode("ascii")
    ).hexdigest()
    raw_nonce = str(uuid.UUID(nonce_raw[:32], version=4))
    observed = dt.datetime.fromtimestamp(
        int(time.time()) if now_unix_seconds is None else int(now_unix_seconds),
        tz=dt.timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    launch = provider.execute_wave(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        stage_receipt=stage,
        iam_receipt=iam,
        transport=transport,
        raw_nonce=raw_nonce,
        observed_at_utc=observed,
        sleep=sleep,
    )
    _write_once(context.directory / "launch_receipt.json", launch)
    _record_execute_event(context, launch)
    return launch


def poll_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: provider.DatasetCloudTransport | None = None,
) -> dict[str, Any]:
    context = _current_context(output_root)
    _require_cloud_phase(
        context,
        phase="poll",
        confirm_run_name=confirm_run_name,
        allow_cloud=allow_cloud_read,
    )
    launch = _wave_receipt(
        context, "launch_receipt.json", "dataset launch receipt"
    )
    if launch is None:
        raise PermissionError("dataset wave cannot poll before launch")
    provider.validate_launch_receipt(launch, provider_plan=context.provider_plan)
    receipt = provider.poll_wave(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        transport=cloud or _live_transport(),
    )
    polls = context.directory / "poll"
    existing = (
        sorted(polls.iterdir())
        if polls.exists() and not polls.is_symlink()
        else []
    )
    for index, path in enumerate(existing):
        match = _POLL_NAME.fullmatch(path.name)
        if (
            match is None
            or int(match.group(1)) != index
            or path.is_symlink()
            or not path.is_file()
            or bridge.sha256_file(path) != match.group(2)
        ):
            raise ValueError("dataset poll journal changed")
    path = (
        polls
        / f"{len(existing):06d}-{_sha256(canonical_bytes(receipt))}.json"
    )
    _write_once(path, receipt)
    _append_event(
        context.root,
        phase="poll",
        operation_key=(
            f"poll:{context.wave_request['request_sha256']}:"
            f"{len(existing):06d}"
        ),
        payload={
            "request_sha256": context.wave_request["request_sha256"],
            "poll_receipt_sha256": receipt["receipt_sha256"],
            "complete_count": receipt["complete_count"],
            "selected_shard_count": receipt["selected_shard_count"],
        },
        cloud_mutated=False,
    )
    return receipt


def _latest_poll(context: WaveContext) -> dict[str, Any]:
    directory = context.directory / "poll"
    if not directory.is_dir() or directory.is_symlink():
        raise PermissionError("dataset cleanup requires a prior poll")
    paths = sorted(directory.iterdir())
    if not paths:
        raise PermissionError("dataset cleanup requires a prior poll")
    value = _read_canonical(paths[-1], "latest dataset poll")
    return provider.validate_poll_receipt(
        value, provider_plan=context.provider_plan
    )


def cleanup_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    allow_incomplete_cleanup: bool = False,
    cloud: provider.DatasetCloudTransport | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    context = _current_context(output_root)
    phase = "cleanup-incomplete" if allow_incomplete_cleanup else "cleanup"
    _require_cloud_phase(
        context,
        phase=phase,
        confirm_run_name=confirm_run_name,
        allow_cloud=allow_cloud_mutation,
    )
    lifecycle_existing = _wave_receipt(
        context, "lifecycle_receipt.json", "dataset lifecycle receipt"
    )
    if lifecycle_existing is not None:
        lifecycle_existing = bridge.validate_lifecycle_receipt(
            lifecycle_existing,
            plan=context.transport_plan,
            ledger=context.ledger,
            resume=context.resume,
            wave_request=context.wave_request,
        )
        _record_cleanup_event(context, lifecycle_existing, phase=phase)
        return lifecycle_existing
    launch = _wave_receipt(
        context, "launch_receipt.json", "dataset launch receipt"
    )
    if launch is None:
        raise PermissionError("dataset cleanup cannot run before launch")
    latest = _latest_poll(context)
    still_running_incomplete = [
        row["shard_id"]
        for row in latest["rows"]
        if row["instance_present"] and not row["complete"]
    ]
    if still_running_incomplete and not allow_incomplete_cleanup:
        raise PermissionError(
            "incomplete running workers require explicit incomplete cleanup"
        )
    lifecycle = provider.cleanup_wave(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        launch_receipt=launch,
        transport=cloud or _live_transport(),
        sleep=sleep,
    )
    _write_once(context.directory / "lifecycle_receipt.json", lifecycle)
    _record_cleanup_event(context, lifecycle, phase=phase)
    return lifecycle


def _record_cleanup_event(
    context: WaveContext, lifecycle: Mapping[str, Any], *, phase: str
) -> None:
    _append_event(
        context.root,
        phase=phase,
        operation_key=f"cleanup:{context.wave_request['request_sha256']}",
        payload={
            "request_sha256": context.wave_request["request_sha256"],
            "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
            "ready_count": sum(
                row["status"] == "ready"
                for row in lifecycle["attempt_rows"]
            ),
            "checkpointed_count": sum(
                row["status"] == "checkpointed"
                for row in lifecycle["attempt_rows"]
            ),
            "failed_count": sum(
                row["status"] == "failed"
                for row in lifecycle["attempt_rows"]
            ),
        },
        cloud_mutated=True,
    )


def _record_receive_event(
    context: WaveContext, receipt: Mapping[str, Any]
) -> None:
    _append_event(
        context.root,
        phase="receive",
        operation_key=f"receive:{context.wave_request['request_sha256']}",
        payload={
            "request_sha256": context.wave_request["request_sha256"],
            "lifecycle_receipt_sha256": receipt[
                "lifecycle_receipt_sha256"
            ],
            "receive_receipt_sha256": receipt["receipt_sha256"],
            "complete_count": receipt["complete_count"],
            "partial_count": receipt["partial_count"],
            "failed_count": receipt["failed_count"],
        },
        cloud_mutated=False,
    )


def receive_next_wave(
    output_root: str | Path,
    *,
    confirm_run_name: str,
    allow_cloud_read: bool,
    cloud: provider.DatasetCloudTransport | None = None,
) -> dict[str, Any]:
    context = _current_context(output_root)
    _require_cloud_phase(
        context,
        phase="receive",
        confirm_run_name=confirm_run_name,
        allow_cloud=allow_cloud_read,
    )
    existing = _wave_receipt(
        context, "receive_receipt.json", "dataset receive receipt"
    )
    if existing is not None:
        expected_lifecycle = _wave_receipt(
            context, "lifecycle_receipt.json", "dataset lifecycle receipt"
        )
        if (
            expected_lifecycle is None
            or existing.get("lifecycle_receipt_sha256")
            != expected_lifecycle["receipt_sha256"]
            or existing.get("source_replayed") is not True
        ):
            raise ValueError("stored dataset receive receipt changed")
        _record_receive_event(context, existing)
        return existing
    lifecycle = _wave_receipt(
        context, "lifecycle_receipt.json", "dataset lifecycle receipt"
    )
    if lifecycle is None:
        raise PermissionError("dataset receive cannot run before exact cleanup")
    lifecycle = bridge.validate_lifecycle_receipt(
        lifecycle,
        plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
    )
    transport = cloud or _live_transport()
    mirror = provider.mirror_lifecycle_objects(
        lifecycle_receipt=lifecycle,
        bucket=context.transport_plan["bucket"],
        output_root=context.directory / "object_mirror",
        transport=transport,
    )
    _write_or_replay(
        context.directory / "mirror_receipt.json",
        mirror,
        "dataset mirror receipt",
    )
    receipt = bridge.receive_wave(
        plan=context.transport_plan,
        lifecycle_receipt=lifecycle,
        object_root=context.directory / "object_mirror",
        local_root=context.contract["local_shard_root"],
    )
    _write_once(context.directory / "receive_receipt.json", receipt)
    _record_receive_event(context, receipt)
    return receipt


def accept_next_wave(output_root: str | Path) -> dict[str, Any]:
    """Commit a lifecycle to the ledger only after source-replayed receive."""

    context = _current_context(output_root)
    _require_controller_token(context.contract)
    lifecycle = _wave_receipt(
        context, "lifecycle_receipt.json", "dataset lifecycle receipt"
    )
    received = _wave_receipt(
        context, "receive_receipt.json", "dataset receive receipt"
    )
    if lifecycle is None or received is None:
        raise PermissionError(
            "dataset wave requires cleanup plus receive before acceptance"
        )
    lifecycle = bridge.validate_lifecycle_receipt(
        lifecycle,
        plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
    )
    if (
        received.get("schema") != bridge.RECEIVE_SCHEMA
        or received.get("plan_sha256") != context.transport_plan["plan_sha256"]
        or received.get("lifecycle_receipt_sha256")
        != lifecycle["receipt_sha256"]
        or received.get("received_count")
        != lifecycle["selected_attempt_count"]
        or received.get("source_replayed") is not True
        or received.get("cloud_mutated") is not False
        or received.get("current_profile_changed") is not False
        or received.get("receipt_sha256")
        != _self_digest(received, "receipt_sha256")
    ):
        raise ValueError("dataset receive receipt cannot be accepted")
    accepted = _accepted_paths(context.root)
    core = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "received_lifecycle_committed_to_attempt_ledger",
        "transport_plan_sha256": context.transport_plan["plan_sha256"],
        "request_sha256": context.wave_request["request_sha256"],
        "wave_index": context.wave_request["wave_index"],
        "accepted_ordinal": len(accepted),
        "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
        "receive_receipt_sha256": received["receipt_sha256"],
        "all_checkpoint_files_source_replayed": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    acceptance = {**core, "acceptance_sha256": canonical_sha256(core)}
    _write_or_replay(
        context.directory / "acceptance_receipt.json",
        acceptance,
        "dataset wave acceptance",
    )
    # Journal the receive-bound acceptance before publishing the accepted
    # lifecycle commit marker.  A crash between these two create-only writes
    # leaves the same wave current, so a retry replays the event and completes
    # the commit without advancing the ledger early.
    _append_event(
        context.root,
        phase="accept",
        operation_key=f"accept:{context.wave_request['request_sha256']}",
        payload=acceptance,
        cloud_mutated=False,
    )
    target = (
        context.root
        / "accepted"
        / (
            f"{len(accepted):06d}-"
            f"{bridge.sha256_file(context.directory / 'lifecycle_receipt.json')}.json"
        )
    )
    _write_or_replay(target, lifecycle, "accepted lifecycle receipt")
    next_state = plan_next_wave(context.root)
    return {
        **acceptance,
        "next_status": (
            "wave_ready"
            if isinstance(next_state, WaveContext)
            else next_state["status"]
        ),
        "next_wave_index": (
            next_state.wave_request["wave_index"]
            if isinstance(next_state, WaveContext)
            else None
        ),
        "next_selected_count": (
            next_state.wave_request["selected_count"]
            if isinstance(next_state, WaveContext)
            else 0
        ),
    }


def controller_status(output_root: str | Path) -> dict[str, Any]:
    root, contract, plan = _load_controller(output_root)
    accepted = _accepted_paths(root)
    ledger = bridge.build_attempt_ledger(
        plan, lifecycle_receipt_paths=accepted
    )
    resume = bridge.build_resume_plan(plan, ledger)
    events = _load_events(root)
    return {
        "schema": "hu_m31_t3_dataset_gcp_controller_status_v1",
        "run_name": contract["run_name"],
        "transport_plan_sha256": plan["plan_sha256"],
        "accepted_lifecycle_count": len(accepted),
        "journal_event_count": len(events),
        "complete_shard_count": ledger["complete_shard_count"],
        "pending_shard_count": ledger["pending_shard_count"],
        "exhausted_shard_count": ledger["exhausted_shard_count"],
        "resume_status": resume["status"],
        "resume_wave_index": resume["resume_wave_index"],
        "selected_count": resume["selected_count"],
        "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
        "machine_type": bridge.MACHINE_TYPE,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }


def finalize_dataset(output_root: str | Path) -> dict[str, Any]:
    """Write the exact local 360-shard map and training-ready merge."""

    root, contract, plan = _load_controller(output_root)
    _require_controller_token(contract)
    accepted = _accepted_paths(root)
    ledger = bridge.build_attempt_ledger(
        plan, lifecycle_receipt_paths=accepted
    )
    resume = bridge.build_resume_plan(plan, ledger)
    if (
        resume["status"] != "complete"
        or ledger["complete_shard_count"] != bridge.CLOUD_SHARD_COUNT
        or ledger["exhausted_shard_count"] != 0
    ):
        raise PermissionError("dataset cannot finalize before all cloud shards")
    local_root = Path(contract["local_shard_root"]).resolve()
    smoke_root = Path(plan["source_paths"]["smoke_shard_directory"]).resolve()
    shard_directories = {
        str(row["shard_id"]): (
            smoke_root
            if row["shard_id"] == dataset.SMOKE_SHARD_ID
            else local_root / str(row["shard_id"])
        )
        for row in dataset.build_dataset_plan()["shards"]
    }
    shard_map = {
        key: str(path.resolve()) for key, path in shard_directories.items()
    }
    final = root / "final"
    shard_map_path = final / "shard_map.json"
    _write_or_replay(shard_map_path, shard_map, "dataset shard map")
    merge = dataset.build_merge_manifest(
        plan=dataset.build_dataset_plan(),
        shard_directories=shard_directories,
    )
    merge_path = final / "dataset_merge.json"
    if merge_path.exists() or merge_path.is_symlink():
        raw = merge_path.read_bytes()
        if (
            merge_path.is_symlink()
            or raw != dataset.canonical_bytes(merge)
            or dataset.validate_merge_manifest(
                json.loads(raw),
                plan=dataset.build_dataset_plan(),
                shard_directories=shard_directories,
            )
            != merge
        ):
            raise FileExistsError("dataset merge conflicts with immutable output")
    else:
        dataset.write_merge_manifest(
            plan=dataset.build_dataset_plan(),
            shard_directories=shard_directories,
            output_path=merge_path,
        )
    core = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "status": "complete_source_replayed_9000_paired_dataset",
        "transport_plan_sha256": plan["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "accepted_lifecycle_count": len(accepted),
        "cloud_shard_count": bridge.CLOUD_SHARD_COUNT,
        "total_shard_count": len(shard_map),
        "paired_hand_count": dataset.TOTAL_PAIRED_HANDS,
        "root_count": dataset.TOTAL_ROOTS,
        "shard_map_path": str(shard_map_path.resolve()),
        "shard_map_sha256": bridge.sha256_file(shard_map_path),
        "merge_path": str(merge_path.resolve()),
        "merge_file_sha256": bridge.sha256_file(merge_path),
        "merge_identity_sha256": dataset.canonical_sha256(merge),
        "training_eligible": merge["dataset_ready_for_training"],
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _write_or_replay(
        final / "FINAL_RECEIPT.json", receipt, "dataset final receipt"
    )
    _append_event(
        root,
        phase="finalize",
        operation_key="finalize",
        payload=receipt,
        cloud_mutated=False,
    )
    return receipt


def _print(value: Any) -> None:
    if isinstance(value, WaveContext):
        value = {
            "status": value.context["status"],
            "directory": str(value.directory),
            "context": value.context,
            "expected_phase_sentinels": {
                phase: expected_phase_sentinel(value, phase)
                for phase in (
                    "execute",
                    "poll",
                    "cleanup",
                    "cleanup-incomplete",
                    "receive",
                )
            },
        }
    print(json.dumps(value, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable M3.1 9,000-pair one-wave GCP controller"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--bucket", required=True)
    prepare.add_argument("--dataset-plan", required=True)
    prepare.add_argument("--fresh-quality-gate", required=True)
    prepare.add_argument("--smoke-gate", required=True)
    prepare.add_argument("--smoke-shard-directory", required=True)
    prepare.add_argument("--smoke-shard-archive", required=True)
    prepare.add_argument("--runtime-archive", required=True)
    prepare.add_argument("--wheelhouse-archive", required=True)
    prepare.add_argument("--candidate-library", required=True)
    prepare.add_argument("--provider-config", required=True)
    prepare.add_argument("--local-shard-root", required=True)
    prepare.add_argument("--output-root", required=True)

    for name in ("plan-next", "status", "accept-next", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--output-root", required=True)

    for name in ("execute-next", "poll-next", "cleanup-next", "receive-next"):
        command = sub.add_parser(name)
        command.add_argument("--output-root", required=True)
        command.add_argument("--confirm-run-name", required=True)
        if name in {"execute-next", "cleanup-next"}:
            command.add_argument("--allow-cloud-mutation", action="store_true")
        else:
            command.add_argument("--allow-cloud-read", action="store_true")
        if name == "cleanup-next":
            command.add_argument(
                "--allow-incomplete-cleanup", action="store_true"
            )
    sentinel = sub.add_parser("sentinel")
    sentinel.add_argument("--output-root", required=True)
    sentinel.add_argument(
        "--phase",
        required=True,
        choices=(
            "execute",
            "poll",
            "cleanup",
            "cleanup-incomplete",
            "receive",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            _print(
                prepare_controller(
                    run_name=args.run_name,
                    bucket=args.bucket,
                    dataset_plan_path=args.dataset_plan,
                    fresh_quality_gate_path=args.fresh_quality_gate,
                    smoke_gate_path=args.smoke_gate,
                    smoke_shard_directory=args.smoke_shard_directory,
                    smoke_shard_archive_path=args.smoke_shard_archive,
                    runtime_archive_path=args.runtime_archive,
                    wheelhouse_archive_path=args.wheelhouse_archive,
                    candidate_library_path=args.candidate_library,
                    provider_config_path=args.provider_config,
                    local_shard_root=args.local_shard_root,
                    output_root=args.output_root,
                    raw_controller_token=os.environ.get(CONTROLLER_TOKEN_ENV),
                )
            )
        elif args.command == "plan-next":
            _print(plan_next_wave(args.output_root))
        elif args.command == "status":
            _print(controller_status(args.output_root))
        elif args.command == "execute-next":
            _print(
                execute_next_wave(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_mutation=args.allow_cloud_mutation,
                )
            )
        elif args.command == "poll-next":
            _print(
                poll_next_wave(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_read=args.allow_cloud_read,
                )
            )
        elif args.command == "cleanup-next":
            _print(
                cleanup_next_wave(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_mutation=args.allow_cloud_mutation,
                    allow_incomplete_cleanup=args.allow_incomplete_cleanup,
                )
            )
        elif args.command == "receive-next":
            _print(
                receive_next_wave(
                    args.output_root,
                    confirm_run_name=args.confirm_run_name,
                    allow_cloud_read=args.allow_cloud_read,
                )
            )
        elif args.command == "accept-next":
            _print(accept_next_wave(args.output_root))
        elif args.command == "finalize":
            _print(finalize_dataset(args.output_root))
        elif args.command == "sentinel":
            _print(
                {
                    "phase": args.phase,
                    "sentinel": expected_phase_sentinel(
                        _current_context(args.output_root), args.phase
                    ),
                    "environment": PHASE_SENTINEL_ENV,
                }
            )
        else:  # pragma: no cover
            raise RuntimeError("unknown controller command")
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "CONTROLLER_CONTRACT_SCHEMA",
    "CONTROLLER_TOKEN_ENV",
    "FINAL_RECEIPT_SCHEMA",
    "JOURNAL_EVENT_SCHEMA",
    "OAUTH_TOKEN_ENV",
    "PHASE_SENTINEL_ENV",
    "PROVIDER_CONFIG_SCHEMA",
    "WAVE_CONTEXT_SCHEMA",
    "WaveContext",
    "accept_next_wave",
    "canonical_bytes",
    "canonical_sha256",
    "cleanup_next_wave",
    "controller_status",
    "execute_next_wave",
    "expected_phase_sentinel",
    "finalize_dataset",
    "initialize_controller_from_transport_plan",
    "main",
    "plan_next_wave",
    "poll_next_wave",
    "prepare_controller",
    "receive_next_wave",
]
