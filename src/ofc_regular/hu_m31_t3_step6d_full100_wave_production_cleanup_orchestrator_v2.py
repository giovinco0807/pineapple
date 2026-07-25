"""Restart-safe production cleanup composition for one full100 wave-v2.

The launch orchestrator deliberately stops at a terminal create
classification.  This module resumes from that immutable execution manifest
and composes only the existing controller cleanup modes:

* exact owned instance/boot-disk deletion;
* GET-only reconciliation after an ambiguous DELETE response;
* exact absence verification;
* exact wave-scoped worker IAM removal (with GET-only reconciliation);
* lifecycle closeout and a ready-to-use production receiver request.

No cloud transport or mutation primitive is implemented here.  Every provider
operation is dispatched through ``controller_v2.execute_mode_request``.  The
immutable content prefix is intentionally retained because later executions
reuse it.  Credentials are accepted only by the underlying adapters through
``GOOGLE_OAUTH_ACCESS_TOKEN``; they are forbidden in all persisted artifacts.
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

from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as gce_v2
from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_production_orchestrator_v2 as launch_v2
from . import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as receiver_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


CLEANUP_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_cleanup_plan_v2"
)
CLEANUP_REQUEST_ENVELOPE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_cleanup_request_envelope_v2"
)
CLEANUP_CHECKPOINT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_cleanup_checkpoint_v2"
)
CLEANUP_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_cleanup_manifest_v2"
)
CLEANUP_MANIFEST_STATUS = (
    "exact_owned_compute_absent_worker_iam_removed_receiver_handoff_ready"
)

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
_EXECUTION_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "execution_namespace",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "orchestration_plan_sha256",
        "controller_journal_dir",
        "startup_script_path",
        "launch_event_sha256",
        "stage_event_sha256",
        "launch_bundle",
        "launch_validation",
        "gce_create_receipt",
        "actual_launch_receipt",
        "content_binding",
        "content_preflight_receipt",
        "stage_receipt",
        "source_content_reused",
        "gce_create_request_ids",
        "gce_delete_request_ids",
        "orphan_disk_delete_request_ids",
        "worker_iam_plan",
        "worker_iam_prepare_receipt",
        "worker_iam_install_receipt",
        "worker_iam_readback_receipt",
        "receiver_request_ready_after_lifecycle_closeout",
        "cleanup_request_ids_persisted",
        "credentials_from_environment_only",
        "additional_create_authorized",
        "current_profile_changed",
        "manifest_sha256",
    }
)
_CLEANUP_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "execution_namespace",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "cleanup_plan_sha256",
        "execution_manifest_sha256",
        "launch_event_sha256",
        "delete_event_sha256",
        "absence_event_sha256",
        "worker_iam_cleanup_event_sha256",
        "closeout_event_sha256",
        "gce_create_receipt_sha256",
        "gce_delete_receipt_sha256",
        "gce_absence_receipt_sha256",
        "worker_iam_cleanup_receipt_sha256",
        "lifecycle_receipt",
        "receiver_request",
        "receiver_request_sha256",
        "all_owned_instances_absent",
        "all_owned_boot_disks_absent",
        "worker_iam_bindings_absent",
        "shared_content_preserved_for_later_executions",
        "content_cleanup_event_sha256",
        "credentials_from_environment_only",
        "additional_create_authorized",
        "current_profile_changed",
        "manifest_sha256",
    }
)

Clock = Callable[[], datetime]
Dispatcher = Callable[..., Any]


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


def _clone(value: Any, label: str) -> Any:
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
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    return value


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
    return _clone(value, label)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("write-once artifact path cannot be a symlink")
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"write-once artifact {path.name} is unreadable") from exc
        if existing != raw:
            raise FileExistsError(
                f"write-once artifact {path.name} contains different bytes"
            ) from None


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


def _event_value(value: Any) -> dict[str, Any]:
    raw = value.value if hasattr(value, "value") else value
    if not isinstance(raw, Mapping):
        raise ValueError("controller event is not an object")
    event = _clone(raw, "controller event")
    digest = _sha(event.get("event_sha256"), "controller event")
    core = dict(event)
    core.pop("event_sha256")
    if canonical_sha256(core) != digest:
        raise ValueError("controller event digest changed")
    if event.get("current_profile_changed") is not False:
        raise ValueError("controller event changed current profile")
    return event


def _sealed_receipt(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    receipt = _clone(value, label)
    digest = _sha(receipt.pop("receipt_sha256", None), label)
    if digest != canonical_sha256(receipt):
        raise ValueError(f"{label} digest changed")
    receipt["receipt_sha256"] = digest
    return receipt


def _sealed_worker_iam_receipt(value: Any, label: str) -> dict[str, Any]:
    """Validate the worker-IAM module's newline-terminated canonical seal."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    receipt = _clone(value, label)
    digest = _sha(receipt.pop("receipt_sha256", None), label)
    expected = worker_iam_v2.canonical_sha256(receipt)
    if (
        receipt.get("schema") != worker_iam_v2.CLEANUP_RECEIPT_SCHEMA
        or digest != expected
    ):
        raise ValueError(f"{label} digest changed")
    receipt["receipt_sha256"] = digest
    return receipt


def _no_network(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise PermissionError("cleanup launch classification never performs network I/O")


@dataclass(frozen=True)
class PreparedCleanup:
    root: Path
    journal_dir: Path
    wave_plan: dict[str, Any]
    attempt_ledger: dict[str, Any]
    resume_plan: dict[str, Any]
    execution_manifest: dict[str, Any]
    cleanup_plan: dict[str, Any]
    launch_event: dict[str, Any]
    gce_create_receipt: dict[str, Any]


def _validate_execution_manifest(
    *,
    value: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EXECUTION_MANIFEST_FIELDS:
        raise ValueError("execution manifest fields changed")
    manifest = _clone(value, "execution manifest")
    digest = _sha(manifest.pop("manifest_sha256"), "execution manifest")
    if digest != canonical_sha256(manifest):
        raise ValueError("execution manifest digest changed")
    manifest["manifest_sha256"] = digest
    expected = {
        "run_name": wave_plan["run_name"],
        "execution_identity_sha256": wave_plan["execution_identity_sha256"],
        "wave_plan_sha256": wave_plan["schedule_sha256"],
        "attempt_ledger_sha256": attempt_ledger["ledger_sha256"],
        "resume_plan_sha256": resume_plan["resume_sha256"],
        "wave_index": resume_plan["resume_wave_index"],
    }
    if (
        manifest["schema"] != launch_v2.EXECUTION_MANIFEST_SCHEMA
        or manifest["status"] != launch_v2.EXECUTION_MANIFEST_STATUS
        or any(manifest.get(key) != expected_value for key, expected_value in expected.items())
        or manifest["receiver_request_ready_after_lifecycle_closeout"] is not True
        or manifest["cleanup_request_ids_persisted"] is not True
        or manifest["credentials_from_environment_only"] is not True
        or manifest["additional_create_authorized"] is not False
        or manifest["current_profile_changed"] is not False
    ):
        raise ValueError("execution manifest binding changed")
    expected_namespace = receiver_v2.execution_namespace(attempt_ledger, resume_plan)
    if manifest["execution_namespace"] != expected_namespace:
        raise ValueError("execution manifest namespace changed")
    selected_names = [row["instance_id"] for row in resume_plan["selected_attempts"]]
    all_request_ids: list[str] = []
    for field in (
        "gce_create_request_ids",
        "gce_delete_request_ids",
        "orphan_disk_delete_request_ids",
    ):
        mapping = manifest[field]
        if not isinstance(mapping, Mapping) or set(mapping) != set(selected_names):
            raise ValueError(f"{field} does not exactly cover selected instances")
        for request_id in mapping.values():
            try:
                parsed = uuid.UUID(request_id)
            except (ValueError, AttributeError, TypeError) as exc:
                raise ValueError(f"{field} contains an invalid requestId") from exc
            if parsed.version != 4 or str(parsed) != request_id:
                raise ValueError(f"{field} contains a non-canonical UUIDv4")
            all_request_ids.append(request_id)
    if len(all_request_ids) != len(set(all_request_ids)):
        raise ValueError("create/delete requestIds are not globally distinct")
    if manifest["launch_bundle"].get("selected_instance_ids") != selected_names:
        raise ValueError("launch bundle selected instance order changed")
    _assert_no_credentials(manifest, "execution manifest")
    return manifest


def _validate_terminal_launch(
    *,
    controller: controller_v2.Full100WaveControllerV2,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = controller.journal.load()
    matches = [
        event
        for event in events
        if event.value["event_sha256"] == manifest["launch_event_sha256"]
    ]
    if len(matches) != 1:
        raise PermissionError("execution manifest launch event is absent or duplicated")
    event = _event_value(matches[0])
    output = event.get("output")
    if (
        event.get("phase") not in {"authorize-launch", "reconcile-create"}
        or event.get("status") not in {"complete", "partial"}
        or not isinstance(output, Mapping)
        or output.get("gce_create_receipt") != manifest["gce_create_receipt"]
        or output.get("actual_launch_receipt") != manifest["actual_launch_receipt"]
    ):
        raise ValueError("execution manifest is not an exact terminal create classification")

    startup_path = Path(manifest["startup_script_path"])
    if not startup_path.is_file() or startup_path.is_symlink():
        raise ValueError("startup script path is not a plain file")
    startup = startup_path.read_bytes()
    expected_startup_sha256 = science_registry.resolve_startup_sha256(
        controller.wave_plan
    )
    if hashlib.sha256(startup).hexdigest() != expected_startup_sha256:
        raise ValueError("startup script bytes differ from frozen launch hash")
    validation = manifest["launch_validation"]
    content = manifest["content_binding"]
    outer = validation.get("outer_manifest")
    if (
        not isinstance(content, Mapping)
        or set(content)
        != {
            "immutable_content_prefix",
            "content_payload_sha256",
            "outer_manifest_sha256",
        }
        or not isinstance(outer, Mapping)
        or content["immutable_content_prefix"] != outer.get("content_prefix")
        or content["content_payload_sha256"]
        != outer.get("content_payload_sha256")
        or content["outer_manifest_sha256"] != outer.get("manifest_sha256")
        or manifest["worker_iam_plan"] != validation.get("worker_iam_plan")
        or manifest["worker_iam_prepare_receipt"]
        != validation.get("worker_iam_prepare_receipt")
        or manifest["worker_iam_install_receipt"]
        != validation.get("worker_iam_install_receipt")
        or manifest["worker_iam_readback_receipt"]
        != validation.get("worker_iam_readback_receipt")
    ):
        raise ValueError("execution manifest content/IAM launch binding changed")

    def validate_bundle(value: Mapping[str, Any]) -> Mapping[str, Any]:
        return bundle_v2.validate_launch_bundle(
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            expected_startup_sha256=expected_startup_sha256,
            value=value,
            **validation,
        )

    bundle = deepcopy(dict(validate_bundle(manifest["launch_bundle"])))
    image = validation["runtime_preflight_receipt"]["image"]
    adapter = gce_v2.GceWavePhaseBAdapter(
        mode="create",
        launch_bundle=bundle,
        launch_bundle_validator=validate_bundle,
        active_image_self_link=image["self_link"],
        active_image_identity_sha256=image["image_identity_sha256"],
        expected_image_digest=controller.wave_plan["runtime_binding"]["image_digest"],
        startup_script_bytes=startup,
        expected_startup_sha256=expected_startup_sha256,
        requester=_no_network,
    )
    create = adapter.validate_create_receipt(manifest["gce_create_receipt"])
    if create != manifest["gce_create_receipt"]:
        raise ValueError("normalized GCE create receipt changed")
    return event, create


def prepare_cleanup(
    *,
    execution_manifest: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    cleanup_root: str | Path,
) -> PreparedCleanup:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    manifest = _validate_execution_manifest(
        value=execution_manifest,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
    )
    journal_dir = Path(manifest["controller_journal_dir"])
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=journal_dir,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        create_journal=False,
    )
    launch_event, create_receipt = _validate_terminal_launch(
        controller=controller, manifest=manifest
    )
    namespace = manifest["execution_namespace"]
    base = Path(cleanup_root)
    root = base if base.name == namespace else base / namespace
    core = {
        "schema": CLEANUP_PLAN_SCHEMA,
        "execution_namespace": namespace,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "execution_manifest_sha256": manifest["manifest_sha256"],
        "launch_event_sha256": launch_event["event_sha256"],
        "gce_create_receipt_sha256": create_receipt["receipt_sha256"],
        "selected_instance_ids": [
            row["instance_id"] for row in resume["selected_attempts"]
        ],
        "gce_delete_request_ids_sha256": canonical_sha256(
            manifest["gce_delete_request_ids"]
        ),
        "orphan_disk_delete_request_ids_sha256": canonical_sha256(
            manifest["orphan_disk_delete_request_ids"]
        ),
        "shared_content_cleanup_authorized": False,
        "shared_content_preserved_for_later_executions": True,
        "credentials_from_environment_only": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    cleanup_plan = {**core, "plan_sha256": canonical_sha256(core)}
    _assert_no_credentials(cleanup_plan, "cleanup plan")
    for name, value in (
        ("execution_manifest", manifest),
        ("wave_plan", plan),
        ("attempt_ledger", ledger),
        ("resume_plan", resume),
    ):
        _write_once_json(root / "inputs" / f"{name}.json", value)
    _write_once_json(root / "cleanup_plan.json", cleanup_plan)
    return PreparedCleanup(
        root=root,
        journal_dir=journal_dir,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        execution_manifest=manifest,
        cleanup_plan=cleanup_plan,
        launch_event=launch_event,
        gce_create_receipt=create_receipt,
    )


class ProductionCleanupOrchestratorV2:
    def __init__(
        self,
        *,
        prepared: PreparedCleanup,
        allow_cloud_read: bool,
        allow_gce_delete: bool,
        allow_worker_iam_cleanup: bool,
        confirm_run_name: str,
        dispatcher: Dispatcher = controller_v2.execute_mode_request,
        requester: Callable[..., Any] | None = None,
        clock: Clock = _system_clock,
    ) -> None:
        if confirm_run_name != prepared.wave_plan["run_name"]:
            raise PermissionError("confirm_run_name must exactly match the validated wave")
        required = {
            "allow_cloud_read": allow_cloud_read,
            "allow_gce_delete": allow_gce_delete,
            "allow_worker_iam_cleanup": allow_worker_iam_cleanup,
        }
        if any(value is not True for value in required.values()):
            missing = sorted(key for key, value in required.items() if value is not True)
            raise PermissionError(
                "cleanup requires explicit allow flags: " + ", ".join(missing)
            )
        if not callable(dispatcher) or not callable(clock):
            raise TypeError("dispatcher and clock must be callable")
        self.prepared = prepared
        self.dispatcher = dispatcher
        self.requester = requester
        self.clock = clock
        self.controller = controller_v2.Full100WaveControllerV2(
            journal_dir=prepared.journal_dir,
            wave_plan=prepared.wave_plan,
            attempt_ledger=prepared.attempt_ledger,
            resume_plan=prepared.resume_plan,
            create_journal=False,
        )
        # Reclassify from journal bytes immediately before mutation authority.
        launch, create = _validate_terminal_launch(
            controller=self.controller,
            manifest=prepared.execution_manifest,
        )
        if launch != prepared.launch_event or create != prepared.gce_create_receipt:
            raise ValueError("terminal create classification changed after planning")
        self.flags = {
            "allow_cloud_read": True,
            "allow_identity_create": False,
            "allow_content_stage": False,
            "allow_gce_create": False,
            "allow_gce_delete": True,
            "allow_content_delete": False,
            "allow_claim_create": False,
            # The existing controller flag covers both IAM install and cleanup.
            "allow_worker_iam_install": True,
            "allow_launch_authorization": False,
        }

    def _now(self) -> str:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("clock did not return datetime")
        return _render_utc(value)

    def _op(self, label: str) -> str:
        operation = f"{self.prepared.execution_manifest['execution_namespace']}:{label}"
        if len(operation) > 128:
            raise ValueError("cleanup operation key is too long")
        return operation

    def _request_path(self, label: str) -> Path:
        return self.prepared.root / "steps" / label / "request.json"

    def _load_request(self, label: str, expected_mode: str) -> dict[str, Any]:
        envelope = _read_json(self._request_path(label), f"{label} request")
        expected = {
            "schema",
            "execution_namespace",
            "label",
            "mode",
            "request",
            "request_sha256",
            "current_profile_changed",
            "envelope_sha256",
        }
        if set(envelope) != expected:
            raise ValueError(f"{label} request envelope fields changed")
        digest = envelope.pop("envelope_sha256")
        if _sha(digest, f"{label} envelope") != canonical_sha256(envelope):
            raise ValueError(f"{label} request envelope digest changed")
        request = envelope["request"]
        if (
            envelope["schema"] != CLEANUP_REQUEST_ENVELOPE_SCHEMA
            or envelope["execution_namespace"]
            != self.prepared.execution_manifest["execution_namespace"]
            or envelope["label"] != label
            or envelope["mode"] != expected_mode
            or envelope["request_sha256"] != canonical_sha256(request)
            or envelope["current_profile_changed"] is not False
            or request.get("operation_key") != self._op(label)
        ):
            raise ValueError(f"{label} request binding changed")
        _assert_no_credentials(request, f"{label} request")
        return request

    def _materialize_request(
        self,
        *,
        label: str,
        mode: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        path = self._request_path(label)
        if path.exists():
            return self._load_request(label, mode)
        request = _clone(builder(), f"{label} request")
        if request.get("operation_key") != self._op(label):
            raise ValueError(f"{label} request operation key changed")
        _assert_no_credentials(request, f"{label} request")
        core = {
            "schema": CLEANUP_REQUEST_ENVELOPE_SCHEMA,
            "execution_namespace": self.prepared.execution_manifest[
                "execution_namespace"
            ],
            "label": label,
            "mode": mode,
            "request": request,
            "request_sha256": canonical_sha256(request),
            "current_profile_changed": False,
        }
        _write_once_json(path, {**core, "envelope_sha256": canonical_sha256(core)})
        return self._load_request(label, mode)

    def _terminal_for(self, operation_key: str) -> dict[str, Any] | None:
        matches = [
            event
            for event in self.controller.journal.load()
            if event.value["operation_key"] == operation_key
            and event.value["status"] in {
                "complete", "partial", "failed",
                "reconciled-complete", "reconciled-partial",
            }
        ]
        if len(matches) > 1:
            raise ValueError("controller operation has multiple terminal events")
        return None if not matches else _event_value(matches[0])

    def _journal_event(self, event_sha256: str) -> dict[str, Any]:
        matches = [
            event
            for event in self.controller.journal.load()
            if event.value["event_sha256"] == event_sha256
        ]
        if len(matches) != 1:
            raise ValueError("persisted step event is absent from controller journal")
        return _event_value(matches[0])

    def _validate_checkpoint(
        self,
        *,
        label: str,
        mode: str,
        request: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        checkpoint = _read_json(
            self.prepared.root
            / "checkpoints"
            / f"{event['event_sha256']}.json",
            f"{label} checkpoint",
        )
        expected = {
            "schema": CLEANUP_CHECKPOINT_SCHEMA,
            "execution_namespace": self.prepared.execution_manifest[
                "execution_namespace"
            ],
            "label": label,
            "mode": mode,
            "request_sha256": canonical_sha256(request),
            "event_sha256": event["event_sha256"],
            "current_profile_changed": False,
        }
        if checkpoint != expected:
            raise ValueError(f"{label} checkpoint binding changed")

    def _dispatch(
        self,
        *,
        label: str,
        mode: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        request = self._materialize_request(label=label, mode=mode, builder=builder)
        event_path = self.prepared.root / "steps" / label / "event.json"
        if event_path.exists():
            event = _event_value(_read_json(event_path, f"{label} event"))
            if (
                event.get("operation_key") != request["operation_key"]
                or event != self._journal_event(event["event_sha256"])
            ):
                raise ValueError(f"{label} persisted event binding changed")
            self._validate_checkpoint(
                label=label, mode=mode, request=request, event=event
            )
            return event
        kwargs: dict[str, Any] = {
            "controller": self.controller,
            "mode": mode,
            "request": request,
            **self.flags,
        }
        if self.requester is not None:
            kwargs["requester"] = self.requester
        try:
            raw = self.dispatcher(**kwargs)
        except Exception:
            recovered = self._terminal_for(request["operation_key"])
            if recovered is None:
                raise
            event = recovered
        else:
            event = _event_value(raw)
        if event.get("operation_key") != request["operation_key"]:
            raise ValueError("controller event operation key differs from request")
        if event != self._journal_event(event["event_sha256"]):
            raise ValueError("controller dispatcher event differs from journal")
        _assert_no_credentials(event, f"{label} event")
        _write_once_json(event_path, event)
        checkpoint = {
            "schema": CLEANUP_CHECKPOINT_SCHEMA,
            "execution_namespace": self.prepared.execution_manifest[
                "execution_namespace"
            ],
            "label": label,
            "mode": mode,
            "request_sha256": canonical_sha256(request),
            "event_sha256": event["event_sha256"],
            "current_profile_changed": False,
        }
        _write_once_json(
            self.prepared.root / "checkpoints" / f"{event['event_sha256']}.json",
            checkpoint,
        )
        self._validate_checkpoint(
            label=label, mode=mode, request=request, event=event
        )
        return event

    def _launch_common(self) -> dict[str, Any]:
        manifest = self.prepared.execution_manifest
        return {
            "launch_bundle": manifest["launch_bundle"],
            "launch_validation": manifest["launch_validation"],
            "startup_script_path": manifest["startup_script_path"],
        }

    @staticmethod
    def _transport_ambiguous(event: Mapping[str, Any]) -> bool:
        output = event.get("output")
        return (
            event.get("status") == "failed"
            and isinstance(output, Mapping)
            and output.get("failure_kind") == "transport_ambiguity"
        )

    def _delete_until_classified(self) -> dict[str, Any]:
        manifest = self.prepared.execution_manifest
        launch_sha = manifest["launch_event_sha256"]
        partial_reconcile_sha: str | None = None
        # One primary attempt plus one possible orphan-disk attempt per selected
        # name is a strict upper bound on independently interrupted resources.
        max_rounds = len(self.prepared.resume_plan["selected_attempts"]) + 1
        for round_index in range(max_rounds):
            label = f"cleanup-delete-r{round_index:02d}"
            delete = self._dispatch(
                label=label,
                mode="cleanup",
                builder=lambda label=label, partial_reconcile_sha=partial_reconcile_sha: {
                    **self._launch_common(),
                    "step": "delete-instances",
                    "operation_key": self._op(label),
                    "launch_event_sha256": launch_sha,
                    "request_ids": manifest["gce_delete_request_ids"],
                    "orphan_disk_request_ids": manifest[
                        "orphan_disk_delete_request_ids"
                    ],
                    "reconcile_event_sha256": partial_reconcile_sha,
                    "observed_at_utc": self._now(),
                },
            )
            if delete.get("status") == "complete":
                receipt = delete.get("output", {}).get("gce_delete_receipt")
                _sealed_receipt(receipt, "GCE delete receipt")
                return delete
            if not self._transport_ambiguous(delete):
                raise RuntimeError("GCE cleanup failed without reconciliation authority")
            reconcile_label = f"reconcile-delete-r{round_index:02d}"
            reconciled = self._dispatch(
                label=reconcile_label,
                mode="reconcile-delete",
                builder=lambda reconcile_label=reconcile_label, label=label: {
                    **self._launch_common(),
                    "operation_key": self._op(reconcile_label),
                    "source_operation_key": self._op(label),
                    "launch_event_sha256": launch_sha,
                    "request_ids": manifest["gce_delete_request_ids"],
                    "observed_at_utc": self._now(),
                },
            )
            if reconciled.get("status") == "complete":
                receipt = reconciled.get("output", {}).get("gce_delete_receipt")
                _sealed_receipt(receipt, "reconciled GCE delete receipt")
                return reconciled
            reconcile_receipt = reconciled.get("output", {}).get(
                "gce_delete_reconciliation_receipt"
            )
            if (
                reconciled.get("status") != "partial"
                or not isinstance(reconcile_receipt, Mapping)
                or reconcile_receipt.get("orphan_cleanup_required") is not True
            ):
                raise RuntimeError("GCE delete ambiguity was not safely classified")
            partial_reconcile_sha = reconciled["event_sha256"]
        raise RuntimeError("GCE orphan cleanup exceeded the selected-resource bound")

    def _verify_absence(self, delete: Mapping[str, Any]) -> dict[str, Any]:
        label = "cleanup-verify-absence"
        event = self._dispatch(
            label=label,
            mode="cleanup",
            builder=lambda: {
                **self._launch_common(),
                "step": "verify-instance-absence",
                "operation_key": self._op(label),
                "launch_event_sha256": self.prepared.execution_manifest[
                    "launch_event_sha256"
                ],
                "delete_event_sha256": delete["event_sha256"],
                "observed_at_utc": self._now(),
            },
        )
        receipt = event.get("output", {}).get("gce_absence_receipt")
        checked = _sealed_receipt(receipt, "GCE absence receipt")
        if (
            event.get("status") != "complete"
            or checked.get("all_instances_absent") is not True
            or checked.get("all_boot_disks_absent") is not True
        ):
            raise RuntimeError("exact GCE instance/disk absence is not proven")
        return event

    def _cleanup_worker_iam(self, absence: Mapping[str, Any]) -> dict[str, Any]:
        manifest = self.prepared.execution_manifest
        content = manifest["content_binding"]
        common = {
            "content_binding": content,
            "iam_plan": manifest["worker_iam_plan"],
            "prepare_receipt": manifest["worker_iam_prepare_receipt"],
            "install_receipt": manifest["worker_iam_install_receipt"],
            "readback_receipt": manifest["worker_iam_readback_receipt"],
        }
        label = "worker-iam-cleanup"
        cleanup = self._dispatch(
            label=label,
            mode="worker-iam-cleanup",
            builder=lambda: {
                "operation_key": self._op(label),
                "predecessor_event_sha256": absence["event_sha256"],
                **common,
                "observed_at_utc": self._now(),
            },
        )
        if cleanup.get("status") == "complete":
            receipt = _sealed_worker_iam_receipt(
                cleanup.get("output", {}).get("receipt"),
                "worker IAM cleanup receipt",
            )
            if receipt.get("cleanup_complete") is not True:
                raise RuntimeError("worker IAM cleanup is incomplete")
            return cleanup
        if not self._transport_ambiguous(cleanup):
            raise RuntimeError("worker IAM cleanup failed without reconciliation authority")
        reconcile_label = "worker-iam-reconcile-cleanup"
        reconciled = self._dispatch(
            label=reconcile_label,
            mode="worker-iam-reconcile-cleanup",
            builder=lambda: {
                "operation_key": self._op(reconcile_label),
                "predecessor_event_sha256": cleanup["event_sha256"],
                "source_operation_key": self._op(label),
                **common,
                "observed_at_utc": self._now(),
            },
        )
        receipt = _sealed_worker_iam_receipt(
            reconciled.get("output", {}).get("receipt"),
            "reconciled worker IAM cleanup receipt",
        )
        if reconciled.get("status") != "complete" or receipt.get(
            "cleanup_complete"
        ) is not True:
            raise RuntimeError("worker IAM cleanup ambiguity is unresolved")
        return reconciled

    def _closeout(
        self,
        *,
        delete: Mapping[str, Any],
        absence: Mapping[str, Any],
        iam_cleanup: Mapping[str, Any],
    ) -> dict[str, Any]:
        label = "lifecycle-closeout"
        event = self._dispatch(
            label=label,
            mode="cleanup",
            builder=lambda: {
                "step": "closeout",
                "operation_key": self._op(label),
                "launch_event_sha256": self.prepared.execution_manifest[
                    "launch_event_sha256"
                ],
                "delete_event_sha256": delete["event_sha256"],
                "absence_event_sha256": absence["event_sha256"],
                "worker_iam_cleanup_event_sha256": iam_cleanup[
                    "event_sha256"
                ],
                "content_cleanup_event_sha256": None,
                "recorded_at_utc": self._now(),
            },
        )
        lifecycle = _sealed_receipt(event.get("output"), "lifecycle receipt")
        if (
            event.get("status") != "complete"
            or lifecycle.get("all_owned_instances_absent") is not True
            or lifecycle.get("all_owned_boot_disks_absent") is not True
            or lifecycle.get("worker_iam_bindings_absent") is not True
            or lifecycle.get("content_cleanup_event_sha256") is not None
        ):
            raise RuntimeError("lifecycle closeout is incomplete")
        return event

    def _validate_existing_manifest(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _CLEANUP_MANIFEST_FIELDS:
            raise ValueError("cleanup manifest fields changed")
        manifest = _clone(value, "cleanup manifest")
        digest = manifest.pop("manifest_sha256", None)
        if _sha(digest, "cleanup manifest") != canonical_sha256(manifest):
            raise ValueError("cleanup manifest digest changed")
        manifest["manifest_sha256"] = digest
        if (
            manifest.get("schema") != CLEANUP_MANIFEST_SCHEMA
            or manifest.get("status") != CLEANUP_MANIFEST_STATUS
            or manifest.get("cleanup_plan_sha256")
            != self.prepared.cleanup_plan["plan_sha256"]
            or manifest.get("execution_manifest_sha256")
            != self.prepared.execution_manifest["manifest_sha256"]
            or manifest.get("shared_content_preserved_for_later_executions")
            is not True
            or manifest.get("content_cleanup_event_sha256") is not None
            or manifest.get("credentials_from_environment_only") is not True
            or manifest.get("additional_create_authorized") is not False
            or manifest.get("current_profile_changed") is not False
        ):
            raise ValueError("cleanup manifest binding changed")
        closeout_sha = manifest.get("closeout_event_sha256")
        matches = [
            event
            for event in self.controller.journal.load()
            if event.value["event_sha256"] == closeout_sha
            and event.value["phase"] == "closeout"
            and event.value["status"] == "complete"
        ]
        if len(matches) != 1 or matches[0].value["output"] != manifest[
            "lifecycle_receipt"
        ]:
            raise ValueError("cleanup manifest closeout event changed")
        receiver = receiver_v2.validate_production_receive_request(
            manifest.get("receiver_request"),
            expected_execution_namespace=self.prepared.execution_manifest[
                "execution_namespace"
            ],
            expected_controller_journal_dir=self.prepared.journal_dir,
        )
        receiver_path = self.prepared.root / "receiver_request.json"
        persisted_receiver = receiver_v2.validate_production_receive_request(
            _read_json(receiver_path, "standalone receiver request"),
            expected_execution_namespace=self.prepared.execution_manifest[
                "execution_namespace"
            ],
            expected_controller_journal_dir=self.prepared.journal_dir,
        )
        lifecycle = manifest["lifecycle_receipt"]
        if (
            receiver != persisted_receiver
            or manifest.get("receiver_request_sha256")
            != receiver["request_sha256"]
            or not isinstance(lifecycle, Mapping)
            or not isinstance(receiver.get("gce_delete_receipt"), Mapping)
            or not isinstance(
                receiver.get("worker_iam_cleanup_receipt"), Mapping
            )
            or receiver["closeout_event_sha256"] != closeout_sha
            or receiver["launch_bundle"]
            != self.prepared.execution_manifest["launch_bundle"]
            or receiver["launch_validation"]
            != self.prepared.execution_manifest["launch_validation"]
            or receiver["startup_script_path"]
            != self.prepared.execution_manifest["startup_script_path"]
            or receiver["gce_create_receipt"] != self.prepared.gce_create_receipt
            or receiver["content_binding"]
            != self.prepared.execution_manifest["content_binding"]
            or receiver["observed_at_utc"] != lifecycle.get("attested_at_utc")
            or receiver["gce_delete_receipt"].get("receipt_sha256")
            != manifest["gce_delete_receipt_sha256"]
            or receiver["worker_iam_cleanup_receipt"].get("receipt_sha256")
            != manifest["worker_iam_cleanup_receipt_sha256"]
        ):
            raise ValueError("cleanup receiver handoff binding changed")
        _assert_no_credentials(manifest, "cleanup manifest")
        return manifest

    def execute(self) -> dict[str, Any]:
        final_path = self.prepared.root / "cleanup_manifest.json"
        if final_path.exists():
            return self._validate_existing_manifest(
                _read_json(final_path, "cleanup manifest")
            )
        delete = self._delete_until_classified()
        absence = self._verify_absence(delete)
        iam_cleanup = self._cleanup_worker_iam(absence)
        closeout = self._closeout(
            delete=delete, absence=absence, iam_cleanup=iam_cleanup
        )
        delete_receipt = _sealed_receipt(
            delete["output"]["gce_delete_receipt"], "GCE delete receipt"
        )
        absence_receipt = _sealed_receipt(
            absence["output"]["gce_absence_receipt"], "GCE absence receipt"
        )
        iam_receipt = _sealed_worker_iam_receipt(
            iam_cleanup["output"]["receipt"], "worker IAM cleanup receipt"
        )
        lifecycle = _sealed_receipt(closeout["output"], "lifecycle receipt")
        receiver_request = receiver_v2.build_production_receive_request(
            execution_namespace=self.prepared.execution_manifest[
                "execution_namespace"
            ],
            controller_journal_dir=self.prepared.journal_dir,
            payload={
                "closeout_event_sha256": closeout["event_sha256"],
                "launch_bundle": self.prepared.execution_manifest["launch_bundle"],
                "launch_validation": self.prepared.execution_manifest[
                    "launch_validation"
                ],
                "startup_script_path": self.prepared.execution_manifest[
                    "startup_script_path"
                ],
                "gce_create_receipt": self.prepared.gce_create_receipt,
                "gce_delete_receipt": delete_receipt,
                "worker_iam_cleanup_receipt": iam_receipt,
                "content_binding": self.prepared.execution_manifest[
                    "content_binding"
                ],
                "observed_at_utc": lifecycle["attested_at_utc"],
            },
        )
        _write_once_json(
            self.prepared.root / "receiver_request.json", receiver_request
        )
        core = {
            "schema": CLEANUP_MANIFEST_SCHEMA,
            "status": CLEANUP_MANIFEST_STATUS,
            "execution_namespace": self.prepared.execution_manifest[
                "execution_namespace"
            ],
            "run_name": self.prepared.wave_plan["run_name"],
            "execution_identity_sha256": self.prepared.wave_plan[
                "execution_identity_sha256"
            ],
            "wave_index": self.prepared.resume_plan["resume_wave_index"],
            "cleanup_plan_sha256": self.prepared.cleanup_plan["plan_sha256"],
            "execution_manifest_sha256": self.prepared.execution_manifest[
                "manifest_sha256"
            ],
            "launch_event_sha256": self.prepared.execution_manifest[
                "launch_event_sha256"
            ],
            "delete_event_sha256": delete["event_sha256"],
            "absence_event_sha256": absence["event_sha256"],
            "worker_iam_cleanup_event_sha256": iam_cleanup["event_sha256"],
            "closeout_event_sha256": closeout["event_sha256"],
            "gce_create_receipt_sha256": self.prepared.gce_create_receipt[
                "receipt_sha256"
            ],
            "gce_delete_receipt_sha256": delete_receipt["receipt_sha256"],
            "gce_absence_receipt_sha256": absence_receipt["receipt_sha256"],
            "worker_iam_cleanup_receipt_sha256": iam_receipt["receipt_sha256"],
            "lifecycle_receipt": lifecycle,
            "receiver_request": receiver_request,
            "receiver_request_sha256": receiver_request["request_sha256"],
            "all_owned_instances_absent": True,
            "all_owned_boot_disks_absent": True,
            "worker_iam_bindings_absent": True,
            "shared_content_preserved_for_later_executions": True,
            "content_cleanup_event_sha256": None,
            "credentials_from_environment_only": True,
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
        _assert_no_credentials(core, "cleanup manifest")
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        _write_once_json(final_path, manifest)
        return self._validate_existing_manifest(manifest)


def execute_production_cleanup(
    *,
    prepared: PreparedCleanup,
    allow_cloud_read: bool,
    allow_gce_delete: bool,
    allow_worker_iam_cleanup: bool,
    confirm_run_name: str,
    dispatcher: Dispatcher = controller_v2.execute_mode_request,
    requester: Callable[..., Any] | None = None,
    clock: Clock = _system_clock,
) -> dict[str, Any]:
    return ProductionCleanupOrchestratorV2(
        prepared=prepared,
        allow_cloud_read=allow_cloud_read,
        allow_gce_delete=allow_gce_delete,
        allow_worker_iam_cleanup=allow_worker_iam_cleanup,
        confirm_run_name=confirm_run_name,
        dispatcher=dispatcher,
        requester=requester,
        clock=clock,
    ).execute()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "execute"), default="plan")
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--wave-plan", required=True)
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--resume-plan", required=True)
    parser.add_argument("--cleanup-root", required=True)
    parser.add_argument("--allow-cloud-read", action="store_true")
    parser.add_argument("--allow-gce-delete", action="store_true")
    parser.add_argument("--allow-worker-iam-cleanup", action="store_true")
    parser.add_argument("--confirm-run-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare_cleanup(
        execution_manifest=_read_json(args.execution_manifest, "execution manifest"),
        wave_plan=_read_json(args.wave_plan, "wave plan"),
        attempt_ledger=_read_json(args.attempt_ledger, "attempt ledger"),
        resume_plan=_read_json(args.resume_plan, "resume plan"),
        cleanup_root=args.cleanup_root,
    )
    if args.mode == "plan":
        print(json.dumps(prepared.cleanup_plan, sort_keys=True, indent=2))
        return 0
    result = execute_production_cleanup(
        prepared=prepared,
        allow_cloud_read=args.allow_cloud_read,
        allow_gce_delete=args.allow_gce_delete,
        allow_worker_iam_cleanup=args.allow_worker_iam_cleanup,
        confirm_run_name=args.confirm_run_name or "",
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


__all__ = [
    "CLEANUP_MANIFEST_SCHEMA",
    "CLEANUP_MANIFEST_STATUS",
    "CLEANUP_PLAN_SCHEMA",
    "PreparedCleanup",
    "ProductionCleanupOrchestratorV2",
    "canonical_bytes",
    "canonical_sha256",
    "execute_production_cleanup",
    "main",
    "prepare_cleanup",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
