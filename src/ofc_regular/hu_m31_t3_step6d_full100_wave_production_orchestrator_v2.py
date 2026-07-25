"""Generic, restart-safe production orchestration for full100 wave-v2.

This module removes the manual JSON handoff between the existing controller
modes.  It deliberately owns no cloud transport and no mutation primitive:
every provider action is still dispatched through
``hu_m31_t3_step6d_full100_wave_controller_v2.execute_mode_request``.  Local
requests, random request IDs/nonces, controller events, and the final
receiver/cleanup handoff are immutable, namespaced artifacts.

``plan`` validates and writes only local artifacts.  ``execute`` is a
separate boundary and requires every relevant allow flag plus an exact run
name confirmation.  Credentials are never accepted as arguments or JSON;
the producer adapters read ``GOOGLE_OAUTH_ACCESS_TOKEN`` themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_content_gcp_adapter_v2 as content_gcp_v2
from . import hu_m31_t3_step6d_full100_wave_content_stage_v2 as content_v2
from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as receiver_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as identity_v2


ORCHESTRATION_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_orchestration_plan_v2"
)
RANDOM_MATERIAL_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_random_material_v2"
)
SOURCE_CONTENT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_source_content_handoff_v2"
)
EXECUTION_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_execution_manifest_v2"
)
EXECUTION_MANIFEST_STATUS = "launch_terminal_receiver_cleanup_handoff_ready"
REQUEST_ENVELOPE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_orchestration_request_envelope_v2"
)
CHECKPOINT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_orchestration_checkpoint_v2"
)

FRESHNESS_SECONDS = 300
PRELAUNCH_AUTHORIZATION_SECONDS = 120
MIN_LAUNCH_FRESHNESS_MARGIN_SECONDS = 15
MAX_READ_REFRESH_EPOCHS = 4
WORKER_IAM_MIN_REMAINING_SECONDS = bundle_v2.IAM_MIN_REMAINING_SECONDS

_EVENT_PHASE_BY_DISPATCH_MODE = {
    # Reuse is a read-only dispatch mode, but the controller deliberately
    # records the bound receipt as the execution's stage-content event.
    "bind-existing-staged-content": "stage-content",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)
_FORBIDDEN_CREDENTIAL_KEYS = frozenset(
    {
        "token",
        "access_token",
        "oauth_token",
        "bearer_token",
        "authorization_header",
        "credentials",
        "credential",
        "service_account_key",
        "private_key",
    }
)

Clock = Callable[[], datetime]
Dispatcher = Callable[..., Any]


class FreshnessRefreshRequired(RuntimeError):
    """The current read-only launch evidence must be refreshed."""


class WorkerIamRefreshRequired(FreshnessRefreshRequired):
    """Worker IAM cannot cover the bounded VM watchdog and launch margin."""


class LaunchAlreadyTerminal(RuntimeError):
    """A prior launch terminal forbids a second create operation key."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strict_clone(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    return datetime.strptime(_utc(value, label), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _render_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("clock must return a timezone-aware UTC datetime")
    return value.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _system_clock() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"{label} is not a plain JSON file")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return _strict_clone(value, label)


def _json_line(value: Mapping[str, Any]) -> bytes:
    return canonical_bytes(value) + b"\n"


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _json_line(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("write-once artifact path cannot be a symlink")
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.publish-tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"write-once artifact {path.name} is unreadable"
                ) from exc
            if existing != raw:
                raise FileExistsError(
                    f"write-once artifact {path.name} already contains different bytes"
                ) from None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _assert_no_credentials(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string key")
            if key.lower() in _FORBIDDEN_CREDENTIAL_KEYS:
                raise PermissionError(
                    f"{label} contains forbidden credential field {key!r}"
                )
            _assert_no_credentials(item, label)
    elif isinstance(value, list):
        for item in value:
            _assert_no_credentials(item, label)


def _uuid4(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} is not a UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} is not a canonical UUIDv4")
    return value


def _new_uuid4() -> str:
    return str(uuid.uuid4())


def _validate_random_material(
    value: Mapping[str, Any],
    *,
    namespace: str,
    selected_instance_ids: Sequence[str],
) -> dict[str, Any]:
    expected = {
        "schema",
        "execution_namespace",
        "setup_nonce",
        "claim_nonce",
        "gce_create_request_ids",
        "gce_delete_request_ids",
        "orphan_disk_delete_request_ids",
        "current_profile_changed",
        "material_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("random material fields changed")
    body = _strict_clone(value, "random material")
    digest = body.pop("material_sha256")
    if digest != canonical_sha256(body):
        raise ValueError("random material digest changed")
    body["material_sha256"] = digest
    if (
        body["schema"] != RANDOM_MATERIAL_SCHEMA
        or body["execution_namespace"] != namespace
        or body["current_profile_changed"] is not False
        or not isinstance(body["setup_nonce"], str)
        or not body["setup_nonce"].startswith("setup-")
        or len(body["setup_nonce"]) != 42
    ):
        raise ValueError("random material contract changed")
    _uuid4(body["setup_nonce"].removeprefix("setup-"), "setup nonce entropy")
    _uuid4(body["claim_nonce"], "persistent claim nonce")
    names = list(selected_instance_ids)
    all_ids: list[str] = []
    for field in (
        "gce_create_request_ids",
        "gce_delete_request_ids",
        "orphan_disk_delete_request_ids",
    ):
        mapping = body[field]
        if not isinstance(mapping, Mapping) or set(mapping) != set(names):
            raise ValueError(f"{field} does not exactly cover selected instances")
        checked = {
            name: _uuid4(mapping[name], f"{field} {name}") for name in names
        }
        body[field] = checked
        all_ids.extend(checked.values())
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("request IDs are duplicated across create and cleanup")
    return body


def _load_or_create_random_material(
    path: Path,
    *,
    namespace: str,
    selected_instance_ids: Sequence[str],
) -> dict[str, Any]:
    if path.exists():
        return _validate_random_material(
            _read_json(path, "random material"),
            namespace=namespace,
            selected_instance_ids=selected_instance_ids,
        )
    names = list(selected_instance_ids)
    core: dict[str, Any] = {
        "schema": RANDOM_MATERIAL_SCHEMA,
        "execution_namespace": namespace,
        "setup_nonce": f"setup-{_new_uuid4()}",
        "claim_nonce": _new_uuid4(),
        "gce_create_request_ids": {name: _new_uuid4() for name in names},
        "gce_delete_request_ids": {name: _new_uuid4() for name in names},
        "orphan_disk_delete_request_ids": {
            name: _new_uuid4() for name in names
        },
        "current_profile_changed": False,
    }
    value = {**core, "material_sha256": canonical_sha256(core)}
    checked = _validate_random_material(
        value, namespace=namespace, selected_instance_ids=names
    )
    _write_once_json(path, checked)
    return checked


def _validate_source_content(
    value: Mapping[str, Any], *, stage_plan: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "schema",
        "content_preflight_receipt",
        "source_stage_receipt",
        "current_profile_changed",
        "handoff_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("source content handoff fields changed")
    payload = _strict_clone(value, "source content handoff")
    digest = payload.pop("handoff_sha256")
    if digest != canonical_sha256(payload):
        raise ValueError("source content handoff digest changed")
    payload["handoff_sha256"] = digest
    preflight = content_v2.validate_preflight_absence_receipt(
        stage_plan, payload["content_preflight_receipt"]
    )
    stage = content_v2.validate_stage_receipt(
        stage_plan, preflight, payload["source_stage_receipt"]
    )
    if (
        payload["schema"] != SOURCE_CONTENT_SCHEMA
        or payload["current_profile_changed"] is not False
        or stage["stage_complete"] is not True
        or stage["created_entry_count"] != 26
    ):
        raise ValueError("source content handoff is not a complete immutable stage")
    payload["content_preflight_receipt"] = preflight
    payload["source_stage_receipt"] = stage
    return payload


def build_source_content_handoff(
    *,
    stage_plan: Mapping[str, Any],
    content_preflight_receipt: Mapping[str, Any],
    source_stage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema": SOURCE_CONTENT_SCHEMA,
        "content_preflight_receipt": _strict_clone(
            content_preflight_receipt, "content preflight receipt"
        ),
        "source_stage_receipt": _strict_clone(
            source_stage_receipt, "source stage receipt"
        ),
        "current_profile_changed": False,
    }
    return _validate_source_content(
        {**core, "handoff_sha256": canonical_sha256(core)},
        stage_plan=stage_plan,
    )


@dataclass(frozen=True)
class PreparedExecution:
    root: Path
    journal_dir: Path
    package_dir: Path
    startup_script_path: Path
    wave_plan: dict[str, Any]
    attempt_ledger: dict[str, Any]
    resume_plan: dict[str, Any]
    outer_manifest: dict[str, Any]
    stage_plan: dict[str, Any]
    identity_plan: dict[str, Any]
    setup_plan: dict[str, Any]
    random_material: dict[str, Any]
    source_content: dict[str, Any] | None
    orchestration_plan: dict[str, Any]

    @property
    def namespace(self) -> str:
        return self.orchestration_plan["execution_namespace"]


def _static_step_contract(source_reuse: bool) -> list[dict[str, Any]]:
    return [
        {"ordinal": 0, "mode": "identity-read", "kind": "read-only", "allow_flag": "allow_cloud_read"},
        {"ordinal": 1, "mode": "content-prefix-preflight" if not source_reuse else "bind-source-preflight", "kind": "read-only", "allow_flag": "allow_cloud_read" if not source_reuse else None},
        {"ordinal": 2, "mode": "runtime-gcp-read", "kind": "read-only", "allow_flag": "allow_cloud_read"},
        {"ordinal": 3, "mode": "prepare", "kind": "local", "allow_flag": None},
        {"ordinal": 4, "mode": "setup-identities-if-missing", "kind": "conditional-mutation", "allow_flag": "allow_identity_create"},
        {"ordinal": 5, "mode": "bind-existing-staged-content" if source_reuse else "stage-content", "kind": "read-only" if source_reuse else "mutation", "allow_flag": "allow_cloud_read" if source_reuse else "allow_content_stage"},
        {"ordinal": 6, "mode": "fresh-read-chain", "kind": "refreshable-read-only", "allow_flag": "allow_cloud_read"},
        {"ordinal": 7, "mode": "persistent-claim", "kind": "mutation", "allow_flag": "allow_claim_create"},
        {"ordinal": 8, "mode": "worker-iam-install-chain", "kind": "read-mutation-read", "allow_flag": "allow_worker_iam_install"},
        {"ordinal": 9, "mode": "prelaunch-authorization", "kind": "local-authorization", "allow_flag": "allow_launch_authorization"},
        {"ordinal": 10, "mode": "launch-bundle-build", "kind": "local-authorization", "allow_flag": "allow_launch_authorization"},
        {"ordinal": 11, "mode": "authorize-launch", "kind": "mutation", "allow_flag": "allow_gce_create"},
    ]


def prepare_execution(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    package_dir: str | Path,
    control_root: str | Path,
    startup_script_path: str | Path | None,
    source_content_handoff: Mapping[str, Any] | None = None,
    clock: Clock = _system_clock,
) -> PreparedExecution:
    """Validate inputs and persist the local-only execution plan."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    descriptor = science_registry.descriptor_for_wave_plan(plan)
    expected_startup_sha256 = science_registry.resolve_startup_sha256(plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if resume["all_jobs_complete"] is True or not resume["selected_attempts"]:
        raise PermissionError("completed or empty resume plan has no execution")
    namespace = receiver_v2.execution_namespace(ledger, resume)
    base = Path(control_root)
    if base.exists() and (not base.is_dir() or base.is_symlink()):
        raise ValueError("control root must be a real directory")
    root = base / namespace
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ValueError("execution namespace must be a real directory")
    root.mkdir(parents=True, exist_ok=True)

    package_root = Path(package_dir).resolve()
    startup = (
        descriptor.resolved_startup_path()
        if startup_script_path is None
        else Path(startup_script_path).resolve()
    )
    if not startup.is_file() or startup.is_symlink():
        raise ValueError("startup script path is not a plain file")
    if hashlib.sha256(startup.read_bytes()).hexdigest() != expected_startup_sha256:
        raise ValueError("startup script differs from the frozen wave-v2 hash")
    outer = package_v2.validate_outer_package(
        package_root,
        plan,
        expected_startup_sha256=expected_startup_sha256,
    )
    stage_plan = content_v2.build_content_stage_plan(
        package_dir=package_root,
        wave_plan=plan,
        expected_startup_sha256=expected_startup_sha256,
        bucket=content_gcp_v2.BUCKET,
    )
    identity_plan = identity_v2.build_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=resume["resume_wave_index"],
    )
    selected_names = [row["instance_id"] for row in resume["selected_attempts"]]
    random_material = _load_or_create_random_material(
        root / "private" / "random_material.json",
        namespace=namespace,
        selected_instance_ids=selected_names,
    )
    setup_path = root / "static" / "worker_identity_setup_plan.json"
    if setup_path.exists():
        setup_plan = identity_v2.validate_create_only_setup_plan(
            _read_json(setup_path, "worker identity setup plan")
        )
    else:
        setup_plan = identity_v2.build_create_only_setup_plan(
            setup_nonce=random_material["setup_nonce"],
            issued_at_utc=_render_utc(clock()),
        )
    source = (
        None
        if source_content_handoff is None
        else _validate_source_content(
            source_content_handoff, stage_plan=stage_plan
        )
    )
    core = {
        "schema": ORCHESTRATION_PLAN_SCHEMA,
        "status": "validated_local_plan_cloud_not_started",
        "execution_namespace": namespace,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "transition_ordinal": len(ledger["transitions"]) - 1,
        "wave_index": resume["resume_wave_index"],
        "outer_manifest_sha256": outer["manifest_sha256"],
        "content_payload_sha256": outer["content_payload_sha256"],
        "stage_plan_sha256": stage_plan["plan_sha256"],
        "worker_identity_plan_sha256": identity_plan["plan_sha256"],
        "worker_identity_setup_plan_sha256": setup_plan["plan_sha256"],
        "random_material_sha256": random_material["material_sha256"],
        "source_content_handoff_sha256": (
            None if source is None else source["handoff_sha256"]
        ),
        "source_content_reuse": source is not None,
        "steps": _static_step_contract(source is not None),
        "freshness_seconds": FRESHNESS_SECONDS,
        "max_read_refresh_epochs": MAX_READ_REFRESH_EPOCHS,
        "credentials_from_environment_only": True,
        "dry_run_cloud_mutation_performed": False,
        "cloud_started": False,
        "current_profile_changed": False,
    }
    orchestration_plan = {**core, "plan_sha256": canonical_sha256(core)}
    _assert_no_credentials(orchestration_plan, "orchestration plan")

    snapshots = {
        root / "inputs" / "wave_plan.json": plan,
        root / "inputs" / "attempt_ledger.json": ledger,
        root / "inputs" / "resume_plan.json": resume,
        root / "static" / "outer_manifest.json": outer,
        root / "static" / "content_stage_plan.json": stage_plan,
        root / "static" / "worker_identity_plan.json": identity_plan,
        setup_path: setup_plan,
        root / "orchestration_plan.json": orchestration_plan,
    }
    if source is not None:
        snapshots[root / "static" / "source_content_handoff.json"] = source
    for path, value in snapshots.items():
        _write_once_json(path, value)
    return PreparedExecution(
        root=root,
        journal_dir=root / "controller-journal",
        package_dir=package_root,
        startup_script_path=startup,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        outer_manifest=outer,
        stage_plan=stage_plan,
        identity_plan=identity_plan,
        setup_plan=setup_plan,
        random_material=random_material,
        source_content=source,
        orchestration_plan=orchestration_plan,
    )


@dataclass(frozen=True)
class _EventView:
    value: dict[str, Any]


def _event_value(event: Any) -> dict[str, Any]:
    value = getattr(event, "value", None)
    if not isinstance(value, Mapping):
        raise TypeError("controller dispatcher returned no event value")
    checked = _strict_clone(value, "controller event")
    digest = _sha(checked.get("event_sha256"), "controller event")
    core = dict(checked)
    del core["event_sha256"]
    if digest != canonical_sha256(core):
        raise ValueError("controller event digest changed")
    return checked


def _output(event: _EventView, key: str) -> dict[str, Any]:
    output = event.value.get("output")
    value = output.get(key) if isinstance(output, Mapping) else None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"controller event lacks {key}")
    return _strict_clone(value, key)


def _mutation_receipt(event: _EventView) -> dict[str, Any]:
    return _output(event, "receipt")


@dataclass
class _FreshEvidence:
    epoch: int
    runtime_gcp_read_receipt: dict[str, Any]
    runtime_preflight_receipt: dict[str, Any]
    inventory_receipt: dict[str, Any]
    identity_act_as_receipt: dict[str, Any]
    project_iam_scan_receipt: dict[str, Any]
    worker_iam_plan: dict[str, Any]
    gcp_read_receipt: dict[str, Any]
    provider_actas_receipt: dict[str, Any]
    predecessor_event: _EventView


class ProductionOrchestratorV2:
    """Compose official controller modes without duplicating their mutations."""

    def __init__(
        self,
        *,
        prepared: PreparedExecution,
        allow_cloud_read: bool,
        allow_identity_create: bool,
        allow_content_stage: bool,
        allow_claim_create: bool,
        allow_worker_iam_install: bool,
        allow_launch_authorization: bool,
        allow_gce_create: bool,
        confirm_run_name: str,
        dispatcher: Dispatcher = controller_v2.execute_mode_request,
        requester: Callable[..., Any] | None = None,
        clock: Clock = _system_clock,
    ) -> None:
        if confirm_run_name != prepared.wave_plan["run_name"]:
            raise PermissionError(
                "confirm_run_name must exactly match the validated wave"
            )
        required = {
            "allow_cloud_read": allow_cloud_read,
            "allow_claim_create": allow_claim_create,
            "allow_worker_iam_install": allow_worker_iam_install,
            "allow_launch_authorization": allow_launch_authorization,
            "allow_gce_create": allow_gce_create,
        }
        if prepared.source_content is None:
            required["allow_content_stage"] = allow_content_stage
        if any(value is not True for value in required.values()):
            missing = sorted(key for key, value in required.items() if value is not True)
            raise PermissionError(
                "execute requires explicit allow flags: " + ", ".join(missing)
            )
        if not callable(dispatcher) or not callable(clock):
            raise TypeError("dispatcher and clock must be callable")
        self.prepared = prepared
        self.clock = clock
        self.dispatcher = dispatcher
        self.requester = requester
        self.flags = {
            "allow_cloud_read": allow_cloud_read,
            "allow_identity_create": allow_identity_create,
            "allow_content_stage": allow_content_stage,
            "allow_gce_create": allow_gce_create,
            "allow_gce_delete": False,
            "allow_content_delete": False,
            "allow_claim_create": allow_claim_create,
            "allow_worker_iam_install": allow_worker_iam_install,
            "allow_launch_authorization": allow_launch_authorization,
        }
        self.controller = controller_v2.Full100WaveControllerV2(
            journal_dir=prepared.journal_dir,
            wave_plan=prepared.wave_plan,
            attempt_ledger=prepared.attempt_ledger,
            resume_plan=prepared.resume_plan,
        )

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("clock did not return datetime")
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("clock must return timezone-aware UTC")
        return value.replace(microsecond=0)

    def _op(self, label: str) -> str:
        value = f"{self.prepared.namespace}:{label}"
        if len(value) > 128:
            raise ValueError("orchestration operation key is too long")
        return value

    def _request_path(self, label: str) -> Path:
        return self.prepared.root / "steps" / label / "request.json"

    def _request_envelope(
        self, *, label: str, mode: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        core = {
            "schema": REQUEST_ENVELOPE_SCHEMA,
            "execution_namespace": self.prepared.namespace,
            "label": label,
            "mode": mode,
            "request": _strict_clone(request, f"{label} request"),
            "request_sha256": canonical_sha256(request),
            "current_profile_changed": False,
        }
        return {**core, "envelope_sha256": canonical_sha256(core)}

    def _checkpoint_payload(
        self,
        *,
        label: str,
        mode: str,
        request: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        core = {
            "schema": CHECKPOINT_SCHEMA,
            "execution_namespace": self.prepared.namespace,
            "label": label,
            "mode": mode,
            "request_sha256": canonical_sha256(request),
            "event_sha256": event["event_sha256"],
            "current_profile_changed": False,
        }
        return {**core, "checkpoint_sha256": canonical_sha256(core)}

    def _ensure_checkpoint(
        self,
        *,
        label: str,
        mode: str,
        request: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        expected = self._checkpoint_payload(
            label=label, mode=mode, request=request, event=event
        )
        expected_path = (
            self.prepared.root
            / "checkpoints"
            / f"{event['event_sha256']}.json"
        )
        labelled_paths = []
        for path in (self.prepared.root / "checkpoints").glob("*.json"):
            value = _read_json(path, "orchestration checkpoint")
            if value.get("label") == label:
                labelled_paths.append(path)
        if any(path != expected_path for path in labelled_paths):
            raise ValueError(f"{label} has an unexpected checkpoint path")
        if expected_path.exists():
            actual = _read_json(expected_path, f"{label} checkpoint")
            if actual != expected:
                raise ValueError(f"{label} checkpoint binding changed")
            return
        _write_once_json(expected_path, expected)

    def _validated_persisted_event(
        self,
        *,
        label: str,
        mode: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        event_path = self.prepared.root / "steps" / label / "event.json"
        persisted = _event_value(
            type(
                "PersistedEvent",
                (),
                {"value": _read_json(event_path, f"{label} event")},
            )()
        )
        if persisted.get("operation_key") != request["operation_key"]:
            raise ValueError(f"{label} event operation key changed")
        expected_phase = _EVENT_PHASE_BY_DISPATCH_MODE.get(mode, mode)
        if persisted.get("phase") != expected_phase:
            raise ValueError(f"{label} event phase changed")
        matching = [
            event.value
            for event in self.controller.journal.load()
            if event.value.get("event_sha256") == persisted["event_sha256"]
        ]
        if len(matching) != 1 or matching[0] != persisted:
            raise controller_v2.JournalTamperError(
                f"{label} persisted event has no exact controller journal match"
            )
        self._ensure_checkpoint(
            label=label,
            mode=mode,
            request=request,
            event=persisted,
        )
        return persisted

    def _load_bound_request(
        self, *, label: str, expected_mode: str | None = None
    ) -> dict[str, Any]:
        envelope = _read_json(
            self._request_path(label), f"{label} request envelope"
        )
        expected_fields = {
            "schema",
            "execution_namespace",
            "label",
            "mode",
            "request",
            "request_sha256",
            "current_profile_changed",
            "envelope_sha256",
        }
        if set(envelope) != expected_fields:
            raise ValueError(f"{label} request envelope fields changed")
        envelope_core = dict(envelope)
        envelope_sha = _sha(
            envelope_core.pop("envelope_sha256"), f"{label} request envelope"
        )
        if envelope_sha != canonical_sha256(envelope_core):
            raise ValueError(f"{label} request envelope digest changed")
        if (
            envelope_core["schema"] != REQUEST_ENVELOPE_SCHEMA
            or envelope_core["execution_namespace"] != self.prepared.namespace
            or envelope_core["label"] != label
            or envelope_core["current_profile_changed"] is not False
            or (
                expected_mode is not None
                and envelope_core["mode"] != expected_mode
            )
        ):
            raise ValueError(f"{label} request envelope context changed")
        request_value = envelope_core["request"]
        if not isinstance(request_value, Mapping):
            raise ValueError(f"{label} request envelope payload changed")
        request = _strict_clone(request_value, f"{label} request")
        request_sha = canonical_sha256(request)
        if envelope_core["request_sha256"] != request_sha:
            raise ValueError(f"{label} persisted request digest changed")
        if request.get("operation_key") != self._op(label):
            raise ValueError(f"{label} request operation key changed")
        _assert_no_credentials(request, f"{label} request")

        event_path = self.prepared.root / "steps" / label / "event.json"
        if event_path.exists():
            self._validated_persisted_event(
                label=label,
                mode=envelope_core["mode"],
                request=request,
            )
        else:
            for checkpoint_path in (
                self.prepared.root / "checkpoints"
            ).glob("*.json"):
                checkpoint = _read_json(
                    checkpoint_path, "orchestration checkpoint"
                )
                if checkpoint.get("label") == label:
                    raise ValueError(
                        f"{label} checkpoint exists without its event"
                    )
        return request

    def _materialize_request(
        self,
        label: str,
        mode: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        path = self._request_path(label)
        if path.exists():
            return self._load_bound_request(label=label, expected_mode=mode)
        request = _strict_clone(builder(), f"{label} request")
        _assert_no_credentials(request, f"{label} request")
        if request.get("operation_key") != self._op(label):
            raise ValueError(f"{label} request operation key changed")
        envelope = self._request_envelope(
            label=label, mode=mode, request=request
        )
        _write_once_json(path, envelope)
        return self._load_bound_request(label=label, expected_mode=mode)

    def _dispatch(
        self,
        *,
        label: str,
        mode: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> _EventView:
        event_path = self.prepared.root / "steps" / label / "event.json"
        if event_path.exists() and not self._request_path(label).exists():
            raise ValueError(f"{label} event exists without its request")
        request = self._materialize_request(label, mode, builder)
        if event_path.exists():
            persisted = self._validated_persisted_event(
                label=label, mode=mode, request=request
            )
            return _EventView(persisted)
        kwargs: dict[str, Any] = {
            "controller": self.controller,
            "mode": mode,
            "request": request,
            **self.flags,
        }
        if self.requester is not None:
            kwargs["requester"] = self.requester
        value = _event_value(self.dispatcher(**kwargs))
        if value.get("operation_key") != request["operation_key"]:
            raise RuntimeError("controller event operation key differs from request")
        _assert_no_credentials(value, f"{label} event")
        _write_once_json(event_path, value)
        self._ensure_checkpoint(
            label=label, mode=mode, request=request, event=value
        )
        return _EventView(value)

    @staticmethod
    def _predecessor(event: _EventView | None) -> str | None:
        return None if event is None else event.value["event_sha256"]

    def _read_window(self) -> tuple[str, str]:
        now = self._now()
        return _render_utc(now), _render_utc(now + timedelta(seconds=FRESHNESS_SECONDS))

    def _runtime_read(self, label: str, predecessor: _EventView | None) -> _EventView:
        def build() -> Mapping[str, Any]:
            now = _render_utc(self._now())
            return {
                "operation_key": self._op(label),
                "predecessor_event_sha256": self._predecessor(predecessor),
                "observed_at_utc": now,
                "current_utc": now,
            }

        return self._dispatch(label=label, mode="runtime-gcp-read", builder=build)

    def _identity_read(self, label: str, predecessor: _EventView | None) -> _EventView:
        def build() -> Mapping[str, Any]:
            observed, expires = self._read_window()
            return {
                "operation_key": self._op(label),
                "predecessor_event_sha256": self._predecessor(predecessor),
                "identity_plan": self.prepared.identity_plan,
                "setup_plan": self.prepared.setup_plan,
                "observed_at_utc": observed,
                "expires_at_utc": expires,
            }

        return self._dispatch(label=label, mode="identity-read", builder=build)

    def _fresh_chain(
        self,
        *,
        epoch: int,
        predecessor: _EventView,
        worker_iam_plan: Mapping[str, Any] | None,
    ) -> _FreshEvidence:
        suffix = f"r{epoch:02d}"
        runtime_event = self._runtime_read(
            f"runtime-gcp-fresh-{suffix}", predecessor
        )
        runtime_gcp = _output(runtime_event, "runtime_gcp_read_receipt")
        runtime = _output(runtime_event, "runtime_preflight_receipt")
        identity_event = self._identity_read(
            f"identity-read-fresh-{suffix}", runtime_event
        )
        identity_read = _output(identity_event, "identity_read_receipt")
        if identity_read.get("all_accounts_exist") is not True:
            raise RuntimeError("fresh worker identity inventory is incomplete")
        inventory = identity_read.get("inventory_receipt")
        if not isinstance(inventory, Mapping):
            raise RuntimeError("fresh identity read lacks inventory receipt")
        inventory = _strict_clone(inventory, "identity inventory")

        actas_label = f"identity-actas-{suffix}"
        actas_event = self._dispatch(
            label=actas_label,
            mode="identity-actas",
            builder=lambda: {
                "operation_key": self._op(actas_label),
                "predecessor_event_sha256": identity_event.value["event_sha256"],
                "identity_plan": self.prepared.identity_plan,
                "inventory_receipt": inventory,
                "tested_at_utc": _render_utc(self._now()),
            },
        )
        identity_actas = _output(actas_event, "worker_identity_act_as_receipt")
        scan_label = f"project-iam-scan-{suffix}"
        scan_event = self._dispatch(
            label=scan_label,
            mode="project-iam-scan",
            builder=lambda: {
                "operation_key": self._op(scan_label),
                "predecessor_event_sha256": actas_event.value["event_sha256"],
                "identity_plan": self.prepared.identity_plan,
                "inventory_receipt": inventory,
                "observed_at_utc": _render_utc(self._now()),
            },
        )
        project_scan = _output(scan_event, "project_iam_scan_receipt")
        content = self._content_binding()
        if worker_iam_plan is None:
            iam_label = (
                "worker-iam-plan"
                if epoch == 0
                else f"worker-iam-plan-{suffix}"
            )
            service_accounts = {
                row["job_id"]: row["service_account_email"]
                for row in self.prepared.identity_plan["selected_workers"]
            }
            iam_event = self._dispatch(
                label=iam_label,
                mode="worker-iam-plan",
                builder=lambda: {
                    "operation_key": self._op(iam_label),
                    "predecessor_event_sha256": scan_event.value["event_sha256"],
                    "content_binding": content,
                    "issued_at_unix_seconds": int(self._now().timestamp()),
                    "service_accounts_by_job": service_accounts,
                },
            )
            iam_plan = _output(iam_event, "worker_iam_plan")
            phasea_predecessor = iam_event
        else:
            iam_plan = _strict_clone(worker_iam_plan, "worker IAM plan")
            phasea_predecessor = scan_event
        phasea_label = f"phasea-read-{suffix}"

        def phasea_request() -> Mapping[str, Any]:
            observed, expires = self._read_window()
            return {
                "operation_key": self._op(phasea_label),
                "predecessor_event_sha256": phasea_predecessor.value["event_sha256"],
                "content_binding": content,
                "iam_plan": iam_plan,
                "observed_at_utc": observed,
                "expires_at_utc": expires,
            }

        phasea_event = self._dispatch(
            label=phasea_label, mode="phasea-read", builder=phasea_request
        )
        gcp_read = _output(phasea_event, "gcp_read_receipt")
        provider_label = f"provider-actas-{suffix}"
        provider_request_path = self._request_path(provider_label)
        provider_event_path = (
            self.prepared.root / "steps" / provider_label / "event.json"
        )
        if provider_event_path.exists() and not provider_request_path.exists():
            raise ValueError(
                f"{provider_label} event exists without its request"
            )
        if provider_request_path.exists():
            persisted_provider = self._load_bound_request(
                label=provider_label,
                expected_mode="provider-actas-check",
            )
            expected_provider_fields = {
                "operation_key",
                "predecessor_event_sha256",
                "content_binding",
                "iam_plan",
                "gcp_read_receipt",
                "checked_at_utc",
                "expires_at_utc",
            }
            if (
                set(persisted_provider) != expected_provider_fields
                or persisted_provider["predecessor_event_sha256"]
                != phasea_event.value["event_sha256"]
                or persisted_provider["content_binding"] != content
                or persisted_provider["iam_plan"] != iam_plan
                or persisted_provider["gcp_read_receipt"] != gcp_read
            ):
                raise ValueError("persisted provider actAs request binding changed")
            _parse_utc(
                persisted_provider["checked_at_utc"],
                "persisted provider actAs checked time",
            )
            _parse_utc(
                persisted_provider["expires_at_utc"],
                "persisted provider actAs expiry",
            )
        read_observed = max(
            _parse_utc(
                gcp_read[name]["observed_at_utc"], f"{name} observed time"
            )
            for name in ("quota_receipt", "planned_mapping_receipt")
        )
        read_expires = min(
            _parse_utc(
                gcp_read[name]["expires_at_utc"], f"{name} expiry"
            )
            for name in ("quota_receipt", "planned_mapping_receipt")
        )
        if self._now() >= read_expires:
            raise FreshnessRefreshRequired(
                "phase A read expired before provider actAs check"
            )

        def provider_request() -> Mapping[str, Any]:
            checked_at = self._now()
            expires_at = min(
                checked_at + timedelta(seconds=FRESHNESS_SECONDS),
                read_expires,
            )
            if checked_at < read_observed or expires_at <= checked_at:
                raise FreshnessRefreshRequired(
                    "phase A read cannot cover provider actAs check"
                )
            return {
                "operation_key": self._op(provider_label),
                "predecessor_event_sha256": phasea_event.value["event_sha256"],
                "content_binding": content,
                "iam_plan": iam_plan,
                "gcp_read_receipt": gcp_read,
                "checked_at_utc": _render_utc(checked_at),
                "expires_at_utc": _render_utc(expires_at),
            }

        try:
            provider_event = self._dispatch(
                label=provider_label,
                mode="provider-actas-check",
                builder=provider_request,
            )
        except PermissionError as exc:
            if str(exc) not in {
                "actAs check is outside the GCP read window",
                "service-account actAs receipt is outside its read window",
            }:
                raise
            raise FreshnessRefreshRequired(
                "persisted provider actAs request is outside its phase A read window"
            ) from exc
        provider_actas = _output(
            provider_event, "service_account_actas_receipt"
        )
        return _FreshEvidence(
            epoch=epoch,
            runtime_gcp_read_receipt=runtime_gcp,
            runtime_preflight_receipt=runtime,
            inventory_receipt=inventory,
            identity_act_as_receipt=identity_actas,
            project_iam_scan_receipt=project_scan,
            worker_iam_plan=iam_plan,
            gcp_read_receipt=gcp_read,
            provider_actas_receipt=provider_actas,
            predecessor_event=provider_event,
        )

    def _content_binding(self) -> dict[str, str]:
        return {
            "immutable_content_prefix": self.prepared.outer_manifest["content_prefix"],
            "content_payload_sha256": self.prepared.outer_manifest[
                "content_payload_sha256"
            ],
            "outer_manifest_sha256": self.prepared.outer_manifest[
                "manifest_sha256"
            ],
        }

    def _claim_bound_epoch_and_iam_plan(
        self,
    ) -> tuple[int, dict[str, Any]] | None:
        """Bind restarts to the exact pre-claim plan; never replan after POST."""

        claim_label = "persistent-claim"
        if not self._request_path(claim_label).exists():
            return None
        claim = self._load_bound_request(
            label=claim_label,
            expected_mode="persistent-claim",
        )
        predecessor = claim.get("predecessor_event_sha256")
        matches: list[tuple[str, dict[str, Any]]] = []
        for path in (self.prepared.root / "steps").glob(
            "provider-actas-r*/event.json"
        ):
            event = _read_json(path, "claim predecessor provider event")
            if event.get("event_sha256") == predecessor:
                matches.append((path.parent.name, event))
        if len(matches) != 1:
            raise ValueError(
                "persistent claim has no unique provider actAs predecessor"
            )
        provider_label, _ = matches[0]
        matched = re.fullmatch(r"provider-actas-r([0-9]{2})", provider_label)
        if matched is None:
            raise ValueError("persistent claim provider epoch changed")
        provider = self._load_bound_request(
            label=provider_label,
            expected_mode="provider-actas-check",
        )
        iam_plan = provider.get("iam_plan")
        if not isinstance(iam_plan, Mapping):
            raise ValueError("persistent claim provider lacks worker IAM plan")
        return int(matched.group(1)), _strict_clone(
            iam_plan, "claim-bound worker IAM plan"
        )

    def _refreshable_deadline(
        self,
        fresh: _FreshEvidence,
        *,
        authorization: Mapping[str, Any] | None = None,
    ) -> datetime:
        read = fresh.gcp_read_receipt
        deadlines = [
            _parse_utc(
                fresh.runtime_preflight_receipt["expires_at_utc"],
                "runtime preflight expiry",
            ),
            _parse_utc(read["quota_receipt"]["expires_at_utc"], "quota expiry"),
            _parse_utc(
                read["planned_mapping_receipt"]["expires_at_utc"],
                "mapping expiry",
            ),
            _parse_utc(
                fresh.provider_actas_receipt["expires_at_utc"],
                "provider actAs expiry",
            ),
            _parse_utc(
                fresh.inventory_receipt["observed_at_utc"],
                "identity inventory time",
            )
            + timedelta(seconds=FRESHNESS_SECONDS),
            _parse_utc(
                fresh.identity_act_as_receipt["tested_at_utc"],
                "identity actAs time",
            )
            + timedelta(seconds=FRESHNESS_SECONDS),
            _parse_utc(
                fresh.project_iam_scan_receipt["observed_at_utc"],
                "project IAM scan time",
            )
            + timedelta(seconds=FRESHNESS_SECONDS),
        ]
        if authorization is not None:
            deadlines.append(
                _parse_utc(authorization["expires_at_utc"], "authorization expiry")
            )
        return min(deadlines)

    def _require_iam_live(self, iam_plan: Mapping[str, Any]) -> None:
        deadline = _parse_utc(iam_plan["expires_at_utc"], "worker IAM expiry")
        if deadline < self._now() + timedelta(
            seconds=WORKER_IAM_MIN_REMAINING_SECONDS
        ):
            raise WorkerIamRefreshRequired(
                "worker IAM does not cover the 4200-second watchdog plus "
                "provision/upload margin; cleanup/replan is required and launch "
                "is forbidden"
            )

    def _fresh_enough(
        self,
        fresh: _FreshEvidence,
        *,
        authorization: Mapping[str, Any] | None = None,
    ) -> bool:
        self._require_iam_live(fresh.worker_iam_plan)
        return self._refreshable_deadline(
            fresh, authorization=authorization
        ) > self._now() + timedelta(
            seconds=MIN_LAUNCH_FRESHNESS_MARGIN_SECONDS
        )

    def _launch_validation(
        self,
        *,
        fresh: _FreshEvidence,
        claim: Mapping[str, Any],
        iam_prepare: Mapping[str, Any],
        iam_install: Mapping[str, Any],
        iam_readback: Mapping[str, Any],
        authorization: Mapping[str, Any],
        current_time_utc: str,
    ) -> dict[str, Any]:
        self._require_iam_live(fresh.worker_iam_plan)
        read = fresh.gcp_read_receipt
        return {
            "outer_manifest": self.prepared.outer_manifest,
            "quota_receipt": read["quota_receipt"],
            "persistent_claim_receipt": claim,
            "planned_mapping_receipt": read["planned_mapping_receipt"],
            "prelaunch_authorization": _strict_clone(
                authorization, "prelaunch authorization"
            ),
            "raw_claim_nonce": self.prepared.random_material["claim_nonce"],
            "current_time_utc": current_time_utc,
            "runtime_preflight_receipt": fresh.runtime_preflight_receipt,
            "runtime_gcp_read_receipt": fresh.runtime_gcp_read_receipt,
            "gcp_read_receipt": read,
            "service_account_actas_receipt": fresh.provider_actas_receipt,
            "worker_identity_plan": self.prepared.identity_plan,
            "worker_identity_inventory_receipt": fresh.inventory_receipt,
            "worker_identity_act_as_receipt": fresh.identity_act_as_receipt,
            "project_iam_scan_receipt": fresh.project_iam_scan_receipt,
            "worker_iam_plan": fresh.worker_iam_plan,
            "worker_iam_prepare_receipt": _strict_clone(
                iam_prepare, "worker IAM prepare receipt"
            ),
            "worker_iam_install_receipt": _strict_clone(
                iam_install, "worker IAM install receipt"
            ),
            "worker_iam_readback_receipt": _strict_clone(
                iam_readback, "worker IAM readback receipt"
            ),
        }

    def _existing_launch_terminal(self) -> _EventView | None:
        matches = [
            _EventView(_strict_clone(event.value, "launch terminal"))
            for event in self.controller.journal.load()
            if event.value.get("phase") == "authorize-launch"
            and event.value.get("status") in {
                "complete",
                "partial",
                "failed",
                "reconciled-complete",
                "reconciled-partial",
            }
        ]
        if len(matches) > 1:
            raise LaunchAlreadyTerminal("multiple launch terminals are forbidden")
        return None if not matches else matches[0]

    def _existing_reconcile_create_terminal(self) -> _EventView | None:
        matches = [
            _EventView(_strict_clone(event.value, "create reconciliation"))
            for event in self.controller.journal.load()
            if event.value.get("phase") == "reconcile-create"
            and event.value.get("status") in {
                "complete",
                "partial",
                "failed",
                "reconciled-complete",
                "reconciled-partial",
            }
        ]
        if len(matches) > 1:
            raise LaunchAlreadyTerminal(
                "multiple create reconciliation terminals are forbidden"
            )
        return None if not matches else matches[0]

    def _find_request_for_operation(self, operation_key: str) -> dict[str, Any]:
        matches = []
        for path in (self.prepared.root / "steps").glob("*/request.json"):
            value = self._load_bound_request(label=path.parent.name)
            if value.get("operation_key") == operation_key:
                matches.append(value)
        if len(matches) != 1:
            raise RuntimeError("launch terminal has no unique persisted request")
        return matches[0]

    def _find_stage_material(
        self, stage_event_sha256: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        stage_events = [
            event
            for event in self.controller.journal.load()
            if event.value.get("event_sha256") == stage_event_sha256
        ]
        if len(stage_events) > 1:
            raise RuntimeError("launch stage event is ambiguous")
        if stage_events:
            stage_event = _EventView(
                _strict_clone(stage_events[0].value, "stage event")
            )
        else:
            persisted = []
            for path in (self.prepared.root / "steps").glob("*/event.json"):
                value = _read_json(path, "persisted stage event")
                if value.get("event_sha256") == stage_event_sha256:
                    persisted.append(value)
            if len(persisted) != 1:
                raise RuntimeError("launch stage event is missing")
            stage_event = _EventView(persisted[0])
        stage_receipt = _output(stage_event, "stage_receipt")
        request = self._find_request_for_operation(
            stage_event.value["operation_key"]
        )
        preflight = request.get("preflight_receipt")
        if not isinstance(preflight, Mapping):
            raise RuntimeError("stage request lacks content preflight receipt")
        return _strict_clone(preflight, "content preflight"), stage_receipt

    def _manifest_from_terminal(self, terminal: _EventView) -> dict[str, Any]:
        if terminal.value["status"] == "failed":
            raise LaunchAlreadyTerminal(
                "prior launch failed with an unknown outcome; retry is forbidden"
            )
        request = self._find_request_for_operation(terminal.value["operation_key"])
        if terminal.value.get("phase") == "reconcile-create":
            source_operation = request.get("source_launch_operation_key")
            if not isinstance(source_operation, str):
                raise RuntimeError(
                    "create reconciliation lacks its source launch operation"
                )
            request = self._find_request_for_operation(source_operation)
        launch_validation = request.get("launch_validation")
        launch_bundle = request.get("launch_bundle")
        if not isinstance(launch_validation, Mapping) or not isinstance(
            launch_bundle, Mapping
        ):
            raise RuntimeError("launch request lacks validation material")
        preflight, stage_receipt = self._find_stage_material(
            request["stage_event_sha256"]
        )
        output = terminal.value.get("output")
        if not isinstance(output, Mapping) or not isinstance(
            output.get("gce_create_receipt"), Mapping
        ):
            raise RuntimeError("launch terminal lacks an owned create receipt")
        core = {
            "schema": EXECUTION_MANIFEST_SCHEMA,
            "status": EXECUTION_MANIFEST_STATUS,
            "execution_namespace": self.prepared.namespace,
            "run_name": self.prepared.wave_plan["run_name"],
            "execution_identity_sha256": self.prepared.wave_plan[
                "execution_identity_sha256"
            ],
            "wave_plan_sha256": self.prepared.wave_plan["schedule_sha256"],
            "attempt_ledger_sha256": self.prepared.attempt_ledger[
                "ledger_sha256"
            ],
            "resume_plan_sha256": self.prepared.resume_plan["resume_sha256"],
            "wave_index": self.prepared.resume_plan["resume_wave_index"],
            "orchestration_plan_sha256": self.prepared.orchestration_plan[
                "plan_sha256"
            ],
            "controller_journal_dir": str(self.prepared.journal_dir.resolve()),
            "startup_script_path": str(self.prepared.startup_script_path),
            "launch_event_sha256": terminal.value["event_sha256"],
            "stage_event_sha256": request["stage_event_sha256"],
            "launch_bundle": _strict_clone(launch_bundle, "launch bundle"),
            "launch_validation": _strict_clone(
                launch_validation, "launch validation"
            ),
            "gce_create_receipt": _strict_clone(
                output["gce_create_receipt"], "GCE create receipt"
            ),
            "actual_launch_receipt": (
                None
                if output.get("actual_launch_receipt") is None
                else _strict_clone(
                    output["actual_launch_receipt"], "actual launch receipt"
                )
            ),
            "content_binding": self._content_binding(),
            "content_preflight_receipt": preflight,
            "stage_receipt": stage_receipt,
            "source_content_reused": self.prepared.source_content is not None,
            "gce_create_request_ids": self.prepared.random_material[
                "gce_create_request_ids"
            ],
            "gce_delete_request_ids": self.prepared.random_material[
                "gce_delete_request_ids"
            ],
            "orphan_disk_delete_request_ids": self.prepared.random_material[
                "orphan_disk_delete_request_ids"
            ],
            "worker_iam_plan": launch_validation["worker_iam_plan"],
            "worker_iam_prepare_receipt": launch_validation[
                "worker_iam_prepare_receipt"
            ],
            "worker_iam_install_receipt": launch_validation[
                "worker_iam_install_receipt"
            ],
            "worker_iam_readback_receipt": launch_validation[
                "worker_iam_readback_receipt"
            ],
            "receiver_request_ready_after_lifecycle_closeout": True,
            "cleanup_request_ids_persisted": True,
            "credentials_from_environment_only": True,
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
        _assert_no_credentials(core, "execution manifest")
        return {**core, "manifest_sha256": canonical_sha256(core)}

    def _reconcile_launch_terminal(self, terminal: _EventView) -> dict[str, Any]:
        if terminal.value.get("phase") != "authorize-launch":
            raise ValueError("create reconciliation source phase changed")
        source_request = self._find_request_for_operation(
            terminal.value["operation_key"]
        )
        if terminal.value["status"] == "complete":
            manifest = self._manifest_from_terminal(terminal)
        else:
            source_label = terminal.value["operation_key"].rsplit(":", 1)[-1]
            label = f"reconcile-create-{source_label}"
            reconciled = self._dispatch(
                label=label,
                mode="reconcile-create",
                builder=lambda: {
                    "operation_key": self._op(label),
                    "source_launch_operation_key": terminal.value[
                        "operation_key"
                    ],
                    "request_ids": self.prepared.random_material[
                        "gce_create_request_ids"
                    ],
                    "observed_at_utc": _render_utc(self._now()),
                    "launch_bundle": source_request["launch_bundle"],
                    "launch_validation": source_request["launch_validation"],
                    "startup_script_path": str(
                        self.prepared.startup_script_path
                    ),
                },
            )
            if reconciled.value.get("status") not in {
                "complete",
                "partial",
                "reconciled-complete",
                "reconciled-partial",
            }:
                raise LaunchAlreadyTerminal(
                    "create reconciliation did not produce a terminal receipt"
                )
            manifest = self._manifest_from_terminal(reconciled)
        final_path = self.prepared.root / "execution_manifest.json"
        _write_once_json(final_path, manifest)
        return manifest

    def _identity_create_until_resolved(
        self,
        *,
        initial_read: Mapping[str, Any],
        prepare_event: _EventView,
    ) -> None:
        if initial_read.get("all_accounts_exist") is True:
            return
        if self.flags["allow_identity_create"] is not True:
            raise PermissionError(
                "worker identities are missing; allow_identity_create is required"
            )
        read_receipt = _strict_clone(initial_read, "initial identity read")
        predecessor = prepare_event
        for attempt in range(identity_v2.POOL_SIZE + 1):
            label = (
                "setup-identities"
                if attempt == 0
                else f"setup-identities-recovery-{attempt:02d}"
            )
            setup_event = self._dispatch(
                label=label,
                mode="setup-identities",
                builder=lambda label=label, read_receipt=read_receipt: {
                    "operation_key": self._op(label),
                    "prepare_event_sha256": prepare_event.value[
                        "event_sha256"
                    ],
                    "identity_plan": self.prepared.identity_plan,
                    "setup_plan": self.prepared.setup_plan,
                    "read_receipt": read_receipt,
                    "current_time_utc": _render_utc(self._now()),
                    "completed_at_utc": _render_utc(self._now()),
                },
            )
            if setup_event.value.get("status") == "complete":
                return
            if setup_event.value.get("status") != "partial":
                raise RuntimeError(
                    "identity setup failed without a safe partial recovery receipt"
                )
            read_label = f"identity-read-recovery-{attempt + 1:02d}"
            read_event = self._identity_read(read_label, setup_event)
            read_receipt = _output(read_event, "identity_read_receipt")
            if read_receipt.get("all_accounts_exist") is True:
                return
            predecessor = read_event
        del predecessor
        raise RuntimeError("identity setup recovery exceeded the fixed pool bound")

    def _worker_iam_install_or_reconcile(
        self,
        *,
        install_event: _EventView,
        content: Mapping[str, Any],
        iam_plan: Mapping[str, Any],
        prepare_receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], _EventView]:
        if install_event.value.get("status") == "complete":
            return _mutation_receipt(install_event), install_event
        output = install_event.value.get("output")
        failure_kind = (
            output.get("failure_kind") if isinstance(output, Mapping) else None
        )
        if (
            install_event.value.get("status") != "failed"
            or failure_kind not in {
                "transport_ambiguity",
                "local_or_response_failure",
                # Compatibility for journals written before the controller
                # distinguished an explicit CAS rejection from post-PUT local
                # response validation.  The reconciliation dispatch is GET-only
                # and must prove exact installed bindings before it can return.
                "provider_rejected_or_local_failure",
            }
        ):
            raise RuntimeError(
                "worker IAM install failed without GET-only reconciliation authority"
            )
        label = "worker-iam-reconcile-install"
        reconciled = self._dispatch(
            label=label,
            mode="worker-iam-reconcile-install",
            builder=lambda: {
                "operation_key": self._op(label),
                "predecessor_event_sha256": install_event.value[
                    "event_sha256"
                ],
                "source_operation_key": install_event.value["operation_key"],
                "content_binding": content,
                "iam_plan": iam_plan,
                "observed_at_utc": _render_utc(self._now()),
                "prepare_receipt": prepare_receipt,
            },
        )
        return _output(reconciled, "receipt"), reconciled

    def _validate_existing_manifest(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = _strict_clone(value, "execution manifest")
        digest = payload.pop("manifest_sha256", None)
        if digest != canonical_sha256(payload):
            raise ValueError("execution manifest digest changed")
        payload["manifest_sha256"] = digest
        if (
            payload.get("schema") != EXECUTION_MANIFEST_SCHEMA
            or payload.get("status") != EXECUTION_MANIFEST_STATUS
            or payload.get("execution_namespace") != self.prepared.namespace
            or payload.get("wave_plan_sha256")
            != self.prepared.wave_plan["schedule_sha256"]
            or payload.get("attempt_ledger_sha256")
            != self.prepared.attempt_ledger["ledger_sha256"]
            or payload.get("resume_plan_sha256")
            != self.prepared.resume_plan["resume_sha256"]
            or payload.get("current_profile_changed") is not False
        ):
            raise ValueError("execution manifest binding changed")
        _assert_no_credentials(payload, "execution manifest")
        return payload

    def execute(self) -> dict[str, Any]:
        final_path = self.prepared.root / "execution_manifest.json"
        if final_path.exists():
            return self._validate_existing_manifest(
                _read_json(final_path, "execution manifest")
            )
        terminal = self._existing_launch_terminal()
        if terminal is not None:
            return self._reconcile_launch_terminal(terminal)
        reconciled = self._existing_reconcile_create_terminal()
        if reconciled is not None:
            recovered = self._manifest_from_terminal(reconciled)
            _write_once_json(final_path, recovered)
            return recovered

        initial_identity = self._identity_read("identity-read-initial", None)
        initial_read = _output(initial_identity, "identity_read_receipt")
        if self.prepared.source_content is None:
            preflight_label = "content-prefix-preflight"
            preflight_event = self._dispatch(
                label=preflight_label,
                mode="content-prefix-preflight",
                builder=lambda: {
                    "operation_key": self._op(preflight_label),
                    "predecessor_event_sha256": initial_identity.value[
                        "event_sha256"
                    ],
                    "package_dir": str(self.prepared.package_dir),
                    "stage_plan": self.prepared.stage_plan,
                    "observed_at_utc": _render_utc(self._now()),
                },
            )
            preflight = _output(preflight_event, "content_preflight_receipt")
            runtime_predecessor = preflight_event
        else:
            preflight = self.prepared.source_content[
                "content_preflight_receipt"
            ]
            runtime_predecessor = initial_identity
        bootstrap_runtime_event = self._runtime_read(
            "runtime-gcp-bootstrap", runtime_predecessor
        )
        bootstrap_runtime_gcp = _output(
            bootstrap_runtime_event, "runtime_gcp_read_receipt"
        )
        bootstrap_runtime = _output(
            bootstrap_runtime_event, "runtime_preflight_receipt"
        )
        prepare_label = "prepare"
        prepare_event = self._dispatch(
            label=prepare_label,
            mode="prepare",
            builder=lambda: {
                "operation_key": self._op(prepare_label),
                "outer_manifest": self.prepared.outer_manifest,
                "stage_plan": self.prepared.stage_plan,
                "content_preflight_receipt": preflight,
                "worker_identity_plan": self.prepared.identity_plan,
                "runtime_preflight_receipt": bootstrap_runtime,
                "runtime_gcp_read_receipt": bootstrap_runtime_gcp,
                "current_time_utc": _render_utc(self._now()),
            },
        )
        self._identity_create_until_resolved(
            initial_read=initial_read, prepare_event=prepare_event
        )
        if self.prepared.source_content is None:
            stage_label = "stage-content"
            stage_event = self._dispatch(
                label=stage_label,
                mode="stage-content",
                builder=lambda: {
                    "operation_key": self._op(stage_label),
                    "prepare_event_sha256": prepare_event.value["event_sha256"],
                    "package_dir": str(self.prepared.package_dir),
                    "stage_plan": self.prepared.stage_plan,
                    "preflight_receipt": preflight,
                    "observed_at_utc": _render_utc(self._now()),
                },
            )
        else:
            stage_label = "bind-existing-staged-content"
            source_stage = self.prepared.source_content["source_stage_receipt"]
            stage_event = self._dispatch(
                label=stage_label,
                mode="bind-existing-staged-content",
                builder=lambda: {
                    "operation_key": self._op(stage_label),
                    "prepare_event_sha256": prepare_event.value["event_sha256"],
                    "stage_plan": self.prepared.stage_plan,
                    "preflight_receipt": preflight,
                    "source_stage_receipt": source_stage,
                    "observed_at_utc": _render_utc(self._now()),
                },
            )

        stage_output = stage_event.value.get("output")
        stage_receipt = (
            stage_output.get("stage_receipt")
            if isinstance(stage_output, Mapping)
            else None
        )
        if (
            stage_event.value.get("status") != "complete"
            or not isinstance(stage_receipt, Mapping)
            or stage_receipt.get("stage_complete") is not True
            or stage_receipt.get("created_entry_count") != 26
        ):
            raise RuntimeError(
                "staged content is not complete; claim/IAM/launch forbidden"
            )

        fresh: _FreshEvidence | None = None
        claim_binding = self._claim_bound_epoch_and_iam_plan()
        if claim_binding is None:
            initial_epochs: Sequence[int] = range(MAX_READ_REFRESH_EPOCHS)
            frozen_claim_iam_plan: Mapping[str, Any] | None = None
        else:
            claim_epoch, claim_iam_plan = claim_binding
            self._require_iam_live(claim_iam_plan)
            initial_epochs = range(claim_epoch, MAX_READ_REFRESH_EPOCHS)
            frozen_claim_iam_plan = claim_iam_plan
        for initial_epoch in initial_epochs:
            try:
                candidate = self._fresh_chain(
                    epoch=initial_epoch,
                    predecessor=stage_event,
                    worker_iam_plan=frozen_claim_iam_plan,
                )
                if not self._fresh_enough(candidate):
                    continue
            except FreshnessRefreshRequired:
                continue
            fresh = candidate
            break
        if fresh is None:
            raise FreshnessRefreshRequired(
                "fresh pre-claim evidence could not be secured; "
                "no launch was attempted and no mutation was performed"
            )
        content = self._content_binding()
        claim_label = "persistent-claim"
        claim_event = self._dispatch(
            label=claim_label,
            mode="persistent-claim",
            builder=lambda: {
                "operation_key": self._op(claim_label),
                "predecessor_event_sha256": fresh.predecessor_event.value[
                    "event_sha256"
                ],
                "content_binding": content,
                "claim_nonce": self.prepared.random_material["claim_nonce"],
                "claimed_at_utc": _render_utc(self._now()),
            },
        )
        claim = _mutation_receipt(claim_event)
        iam_prepare_label = "worker-iam-prepare"
        iam_prepare_event = self._dispatch(
            label=iam_prepare_label,
            mode="worker-iam-prepare",
            builder=lambda: {
                "operation_key": self._op(iam_prepare_label),
                "predecessor_event_sha256": claim_event.value["event_sha256"],
                "content_binding": content,
                "iam_plan": fresh.worker_iam_plan,
                "observed_at_utc": _render_utc(self._now()),
            },
        )
        iam_prepare = _output(iam_prepare_event, "worker_iam_prepare_receipt")
        iam_install_label = "worker-iam-install"
        iam_install_event = self._dispatch(
            label=iam_install_label,
            mode="worker-iam-install",
            builder=lambda: {
                "operation_key": self._op(iam_install_label),
                "predecessor_event_sha256": iam_prepare_event.value[
                    "event_sha256"
                ],
                "content_binding": content,
                "iam_plan": fresh.worker_iam_plan,
                "observed_at_utc": _render_utc(self._now()),
                "prepare_receipt": iam_prepare,
            },
        )
        iam_install, iam_install_predecessor = (
            self._worker_iam_install_or_reconcile(
                install_event=iam_install_event,
                content=content,
                iam_plan=fresh.worker_iam_plan,
                prepare_receipt=iam_prepare,
            )
        )
        iam_readback_label = "worker-iam-readback"
        iam_readback_event = self._dispatch(
            label=iam_readback_label,
            mode="worker-iam-readback",
            builder=lambda: {
                "operation_key": self._op(iam_readback_label),
                "predecessor_event_sha256": iam_install_predecessor.value[
                    "event_sha256"
                ],
                "content_binding": content,
                "iam_plan": fresh.worker_iam_plan,
                "observed_at_utc": _render_utc(self._now()),
                "prepare_receipt": iam_prepare,
                "install_receipt": iam_install,
            },
        )
        iam_readback = _output(
            iam_readback_event, "worker_iam_readback_receipt"
        )

        launch_event: _EventView | None = None
        for epoch in range(fresh.epoch, MAX_READ_REFRESH_EPOCHS):
            if epoch > fresh.epoch:
                fresh = self._fresh_chain(
                    epoch=epoch,
                    predecessor=iam_readback_event,
                    worker_iam_plan=fresh.worker_iam_plan,
                )
            if not self._fresh_enough(fresh):
                continue
            deadline = self._refreshable_deadline(fresh)
            auth_label = f"prelaunch-authorization-r{epoch:02d}"

            def auth_request() -> Mapping[str, Any]:
                now = self._now()
                expires = min(
                    now + timedelta(seconds=PRELAUNCH_AUTHORIZATION_SECONDS),
                    deadline,
                )
                if expires <= now:
                    raise FreshnessRefreshRequired(
                        "read evidence expired before prelaunch authorization"
                    )
                read = fresh.gcp_read_receipt
                return {
                    "operation_key": self._op(auth_label),
                    "predecessor_event_sha256": fresh.predecessor_event.value[
                        "event_sha256"
                    ],
                    "immutable_content_sha256": content[
                        "content_payload_sha256"
                    ],
                    "quota_receipt": read["quota_receipt"],
                    "persistent_claim_receipt": claim,
                    "planned_mapping_receipt": read[
                        "planned_mapping_receipt"
                    ],
                    "raw_claim_nonce": self.prepared.random_material[
                        "claim_nonce"
                    ],
                    "authorized_at_utc": _render_utc(now),
                    "expires_at_utc": _render_utc(expires),
                }

            try:
                auth_event = self._dispatch(
                    label=auth_label,
                    mode="prelaunch-authorization",
                    builder=auth_request,
                )
            except FreshnessRefreshRequired:
                continue
            authorization = _output(auth_event, "prelaunch_authorization")
            if not self._fresh_enough(fresh, authorization=authorization):
                continue
            bundle_now = _render_utc(self._now())
            validation = self._launch_validation(
                fresh=fresh,
                claim=claim,
                iam_prepare=iam_prepare,
                iam_install=iam_install,
                iam_readback=iam_readback,
                authorization=authorization,
                current_time_utc=bundle_now,
            )
            bundle_label = f"launch-bundle-build-r{epoch:02d}"
            bundle_event = self._dispatch(
                label=bundle_label,
                mode="launch-bundle-build",
                builder=lambda: {
                    "operation_key": self._op(bundle_label),
                    "predecessor_event_sha256": auth_event.value[
                        "event_sha256"
                    ],
                    "launch_validation": validation,
                },
            )
            launch_bundle = _output(bundle_event, "launch_bundle")
            if not self._fresh_enough(fresh, authorization=authorization):
                continue
            if self._existing_launch_terminal() is not None:
                raise LaunchAlreadyTerminal(
                    "launch became terminal before create dispatch"
                )
            launch_label = f"authorize-launch-r{epoch:02d}"
            launch_event = self._dispatch(
                label=launch_label,
                mode="authorize-launch",
                builder=lambda: {
                    "operation_key": self._op(launch_label),
                    "stage_event_sha256": stage_event.value["event_sha256"],
                    "request_ids": self.prepared.random_material[
                        "gce_create_request_ids"
                    ],
                    "launch_started_at_utc": _render_utc(self._now()),
                    "observed_at_utc": _render_utc(self._now()),
                    "launch_bundle": launch_bundle,
                    "launch_validation": validation,
                    "startup_script_path": str(
                        self.prepared.startup_script_path
                    ),
                },
            )
            if launch_event.value.get("status") != "complete":
                return self._reconcile_launch_terminal(launch_event)
            break
        if launch_event is None:
            raise FreshnessRefreshRequired(
                "fresh read evidence could not be secured; no launch was attempted"
            )
        manifest = self._manifest_from_terminal(launch_event)
        _write_once_json(final_path, manifest)
        return manifest


def execute_production_wave(
    *,
    prepared: PreparedExecution,
    allow_cloud_read: bool,
    allow_identity_create: bool,
    allow_content_stage: bool,
    allow_claim_create: bool,
    allow_worker_iam_install: bool,
    allow_launch_authorization: bool,
    allow_gce_create: bool,
    confirm_run_name: str,
    dispatcher: Dispatcher = controller_v2.execute_mode_request,
    requester: Callable[..., Any] | None = None,
    clock: Clock = _system_clock,
) -> dict[str, Any]:
    return ProductionOrchestratorV2(
        prepared=prepared,
        allow_cloud_read=allow_cloud_read,
        allow_identity_create=allow_identity_create,
        allow_content_stage=allow_content_stage,
        allow_claim_create=allow_claim_create,
        allow_worker_iam_install=allow_worker_iam_install,
        allow_launch_authorization=allow_launch_authorization,
        allow_gce_create=allow_gce_create,
        confirm_run_name=confirm_run_name,
        dispatcher=dispatcher,
        requester=requester,
        clock=clock,
    ).execute()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "execute"), default="plan")
    parser.add_argument("--wave-plan", required=True)
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--resume-plan", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument(
        "--startup-script",
        default=None,
    )
    parser.add_argument("--source-content-handoff")
    parser.add_argument("--allow-cloud-read", action="store_true")
    parser.add_argument("--allow-identity-create", action="store_true")
    parser.add_argument("--allow-content-stage", action="store_true")
    parser.add_argument("--allow-claim-create", action="store_true")
    parser.add_argument("--allow-worker-iam-install", action="store_true")
    parser.add_argument("--allow-launch-authorization", action="store_true")
    parser.add_argument("--allow-gce-create", action="store_true")
    parser.add_argument("--confirm-run-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    wave_plan = _read_json(args.wave_plan, "wave plan")
    source = (
        None
        if args.source_content_handoff is None
        else _read_json(args.source_content_handoff, "source content handoff")
    )
    prepared = prepare_execution(
        wave_plan=wave_plan,
        attempt_ledger=_read_json(args.attempt_ledger, "attempt ledger"),
        resume_plan=_read_json(args.resume_plan, "resume plan"),
        package_dir=args.package_dir,
        control_root=args.control_root,
        startup_script_path=args.startup_script,
        source_content_handoff=source,
    )
    if args.mode == "plan":
        print(json.dumps(prepared.orchestration_plan, sort_keys=True, indent=2))
        return 0
    if args.confirm_run_name is None:
        raise PermissionError("execute requires --confirm-run-name")
    manifest = execute_production_wave(
        prepared=prepared,
        allow_cloud_read=args.allow_cloud_read,
        allow_identity_create=args.allow_identity_create,
        allow_content_stage=args.allow_content_stage,
        allow_claim_create=args.allow_claim_create,
        allow_worker_iam_install=args.allow_worker_iam_install,
        allow_launch_authorization=args.allow_launch_authorization,
        allow_gce_create=args.allow_gce_create,
        confirm_run_name=args.confirm_run_name,
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


__all__ = [
    "EXECUTION_MANIFEST_SCHEMA",
    "EXECUTION_MANIFEST_STATUS",
    "FreshnessRefreshRequired",
    "LaunchAlreadyTerminal",
    "ORCHESTRATION_PLAN_SCHEMA",
    "PreparedExecution",
    "ProductionOrchestratorV2",
    "RANDOM_MATERIAL_SCHEMA",
    "SOURCE_CONTENT_SCHEMA",
    "build_source_content_handoff",
    "canonical_bytes",
    "canonical_sha256",
    "execute_production_wave",
    "main",
    "prepare_execution",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
