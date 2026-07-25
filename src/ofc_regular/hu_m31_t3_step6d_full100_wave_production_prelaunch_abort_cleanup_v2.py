"""Restart-safe abort cleanup for a full100 wave that never reached GCE create.

This is deliberately separate from the normal production cleanup orchestrator.
The latter requires an immutable execution manifest and a terminal GCE create
classification.  This module accepts only the opposite state: no execution
manifest and no ``authorize-launch``/``reconcile-create`` controller event.

The only cloud mutation authorized here is removal of the exact 16 wave-scoped
worker bucket-IAM bindings.  The persistent claim, fixed worker service
accounts, and immutable 26-object content prefix are retained.  Selected GCE
instance and boot-disk absence is observed before and after IAM cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_cloud_v2 as cloud_v2
from . import hu_m31_t3_step6d_full100_wave_content_gcp_adapter_v2 as content_gcp_v2
from . import hu_m31_t3_step6d_full100_wave_content_stage_v2 as content_v2
from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_v2
from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as launch_bundle_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2
from . import hu_m31_t3_step6d_full100_wave_production_cleanup_orchestrator_v2 as cleanup_v2


LEGACY_ABORT_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_prelaunch_abort_plan_v2"
ABORT_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_prelaunch_abort_plan_v3"
ABORT_REQUEST_SCHEMA = "hu_m31_t3_step6d_full100_wave_prelaunch_abort_request_v2"
ABORT_CHECKPOINT_SCHEMA = "hu_m31_t3_step6d_full100_wave_prelaunch_abort_checkpoint_v2"
ABORT_MANIFEST_SCHEMA = "hu_m31_t3_step6d_full100_wave_prelaunch_abort_manifest_v2"
ABORT_MANIFEST_STATUS = (
    "prelaunch_compute_absent_worker_iam_removed_preserved_resources_verified"
)
PRESERVATION_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_prelaunch_abort_preservation_receipt_v2"
)
COMPUTE_ABSENCE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_prelaunch_abort_compute_absence_receipt_v2"
)
IAM_INTENT_RECOVERY_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_prelaunch_abort_iam_intent_recovery_receipt_v2"
)
COMPUTE_ABSENCE_FRESHNESS_SECONDS = 60
MAX_RECOVERY_ROUNDS = 4

_LEGACY_ABORT_PLAN_KEYS = frozenset(
    {
        "schema", "status", "execution_namespace", "run_name",
        "execution_identity_sha256", "wave_plan_sha256",
        "attempt_ledger_sha256", "resume_plan_sha256", "wave_index",
        "controller_journal_dir", "anchor_event_sha256",
        "controller_event_count_at_plan", "selected_instance_ids",
        "worker_iam_plan_sha256", "worker_iam_binding_count",
        "worker_iam_cleanup_not_before_utc", "claim_receipt_sha256",
        "claim_object_path", "stage_receipt_sha256", "content_entry_count",
        "execution_manifest_present", "gce_create_event_count",
        "claim_delete_authorized", "service_account_delete_authorized",
        "content_delete_authorized", "cloud_mutation_performed",
        "prelaunch_abort_tombstone_required",
        "current_profile_changed", "plan_sha256",
    }
)
_ABORT_PLAN_KEYS = _LEGACY_ABORT_PLAN_KEYS | frozenset(
    {"persisted_authorize_request"}
)

_REQUEST_ENVELOPE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_orchestration_request_envelope_v2"
)
_REQUEST_ENVELOPE_KEYS = frozenset(
    {
        "schema", "execution_namespace", "label", "mode", "request",
        "request_sha256", "current_profile_changed", "envelope_sha256",
    }
)
_AUTHORIZE_REQUEST_KEYS = frozenset(
    {
        "operation_key", "stage_event_sha256", "request_ids",
        "launch_started_at_utc", "observed_at_utc", "launch_bundle",
        "launch_validation", "startup_script_path",
    }
)
_AUTHORIZE_REQUEST_LABEL = re.compile(r"authorize-launch-r[0-9]{2}")
_PERSISTED_AUTHORIZE_REQUEST_KEYS = frozenset(
    {
        "schema", "status", "label", "operation_key", "envelope_sha256",
        "request_sha256", "launch_bundle_sha256", "stage_event_sha256",
        "request_only", "current_profile_changed",
    }
)

_ABORT_MANIFEST_KEYS = frozenset(
    {
        "schema", "status", "execution_namespace", "run_name",
        "execution_identity_sha256", "wave_index", "abort_plan_sha256",
        "anchor_event_sha256", "tombstone_event_sha256",
        "preflight_event_sha256",
        "worker_iam_readback_event_sha256",
        "worker_iam_intent_recovery_event_sha256",
        "worker_iam_cleanup_event_sha256", "postflight_event_sha256",
        "preservation_event_sha256", "preflight_receipt_sha256",
        "worker_iam_cleanup_receipt_sha256", "postflight_receipt_sha256",
        "preservation_receipt_sha256", "all_selected_instances_absent_pre",
        "all_selected_boot_disks_absent_pre",
        "all_selected_instances_absent_post",
        "all_selected_boot_disks_absent_post", "worker_iam_bindings_absent",
        "worker_iam_removed_binding_count",
        "unrelated_policy_fingerprint_sha256", "claim_preserved",
        "claim_object_path", "claim_generation", "service_accounts_preserved",
        "service_account_count", "immutable_content_preserved",
        "immutable_content_prefix", "immutable_content_entry_count",
        "claim_cleanup_event_sha256", "service_account_cleanup_event_sha256",
        "content_cleanup_event_sha256", "old_execution_relaunch_authorized",
        "additional_create_authorized", "credentials_from_environment_only",
        "current_profile_changed", "manifest_sha256",
    }
)

Clock = Callable[[], datetime]
Dispatcher = Callable[..., Any]
PreservationReader = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ComputeAbsenceReader = Callable[[Mapping[str, Any]], Mapping[str, Any]]
IamIntentRecoveryReader = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def canonical_bytes(value: Any) -> bytes:
    return cleanup_v2.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return cleanup_v2.canonical_sha256(value)


def _clone(value: Any, label: str) -> Any:
    return cleanup_v2._clone(value, label)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    return cleanup_v2._read_json(path, label)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    cleanup_v2._write_once_json(path, value)


def _assert_no_credentials(value: Any, label: str) -> None:
    cleanup_v2._assert_no_credentials(value, label)


def _event_value(value: Any) -> dict[str, Any]:
    return cleanup_v2._event_value(value)


def _sha(value: Any, label: str) -> str:
    return cleanup_v2._sha(value, label)


def _render_utc(value: datetime) -> str:
    return cleanup_v2._render_utc(value)


def _system_clock() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_utc(value: str) -> datetime:
    cleanup_v2._utc(value, "UTC timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _worker_iam_cleanup_not_before(iam_plan: Mapping[str, Any]) -> str:
    return _render_utc(
        _parse_utc(iam_plan["expires_at_utc"])
        - timedelta(seconds=launch_bundle_v2.IAM_MIN_REMAINING_SECONDS)
    )


def _sealed(value: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    payload = _clone(value, "sealed receipt")
    digest = payload.pop(field, None)
    if _sha(digest, field) != canonical_sha256(payload):
        raise ValueError("sealed receipt digest changed")
    payload[field] = digest
    return payload


def _single_event(
    events: Sequence[Any],
    *,
    phases: set[str],
    status: set[str] = {"complete", "reconciled-complete"},
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    label: str,
) -> dict[str, Any]:
    rows = []
    for raw in events:
        event = _event_value(raw)
        if event["phase"] not in phases or event["status"] not in status:
            continue
        if predicate is not None and not predicate(event):
            continue
        rows.append(event)
    if len(rows) != 1:
        raise ValueError(f"{label} must have exactly one matching controller event")
    return rows[0]


def _persisted_request_only_authority(
    execution_root: Path, events: Sequence[Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    steps = execution_root / "steps"
    paths = (
        []
        if not steps.is_dir()
        else sorted(
            path
            for path in steps.glob("authorize-launch*/request.json")
            if path.is_file()
        )
    )
    if not paths:
        return None
    if len(paths) != 1:
        raise PermissionError(
            "prelaunch abort requires exactly one persisted request-only launch"
        )
    path = paths[0]
    label = path.parent.name
    if (
        path.is_symlink()
        or path.parent.is_symlink()
        or _AUTHORIZE_REQUEST_LABEL.fullmatch(label) is None
        or (path.parent / "event.json").exists()
    ):
        raise PermissionError(
            "prelaunch abort refuses a persisted GCE create request that is not "
            "strictly request-only"
        )
    envelope = _read_json(path, "persisted authorize request envelope")
    if set(envelope) != _REQUEST_ENVELOPE_KEYS:
        raise ValueError("persisted authorize request envelope fields changed")
    envelope_core = _clone(envelope, "persisted authorize request envelope")
    envelope_digest = envelope_core.pop("envelope_sha256", None)
    if _sha(envelope_digest, "authorize request envelope") != canonical_sha256(
        envelope_core
    ):
        raise ValueError("persisted authorize request envelope digest changed")
    request = envelope_core.get("request")
    if not isinstance(request, Mapping) or set(request) != _AUTHORIZE_REQUEST_KEYS:
        raise ValueError("persisted authorize request fields changed")
    request = _clone(request, "persisted authorize request")
    request_digest = canonical_sha256(request)
    operation_key = f"{execution_root.name}:{label}"
    if (
        envelope_core.get("schema") != _REQUEST_ENVELOPE_SCHEMA
        or envelope_core.get("execution_namespace") != execution_root.name
        or envelope_core.get("label") != label
        or envelope_core.get("mode") != "authorize-launch"
        or envelope_core.get("request_sha256") != request_digest
        or envelope_core.get("current_profile_changed") is not False
        or request.get("operation_key") != operation_key
    ):
        raise ValueError("persisted authorize request context changed")
    checkpoints = execution_root / "checkpoints"
    if checkpoints.is_dir():
        for checkpoint_path in checkpoints.glob("*.json"):
            checkpoint = _read_json(
                checkpoint_path, "persisted orchestration checkpoint"
            )
            if checkpoint.get("label") == label:
                raise PermissionError(
                    "persisted authorize request unexpectedly has a checkpoint"
                )
    stage_sha = request.get("stage_event_sha256")
    stage_matches = [
        _event_value(event)
        for event in events
        if _event_value(event).get("event_sha256") == stage_sha
    ]
    if (
        len(stage_matches) != 1
        or stage_matches[0].get("phase") != "stage-content"
        or stage_matches[0].get("status") != "complete"
    ):
        raise PermissionError("persisted authorize request stage event changed")
    launch_bundle = request.get("launch_bundle")
    request_ids = request.get("request_ids")
    if not isinstance(launch_bundle, Mapping) or not isinstance(request_ids, Mapping):
        raise ValueError("persisted authorize request launch material changed")
    bundle = _clone(launch_bundle, "persisted authorize launch bundle")
    bundle_digest = bundle.pop("bundle_sha256", None)
    selected_instances = bundle.get("selected_instance_ids")
    if (
        _sha(bundle_digest, "persisted launch bundle") != canonical_sha256(bundle)
        or not isinstance(selected_instances, list)
        or set(request_ids) != set(selected_instances)
    ):
        raise ValueError("persisted authorize request bundle binding changed")
    binding = {
        "schema": "hu_m31_t3_step6d_full100_wave_request_only_authority_v1",
        "status": "persisted_authorize_request_before_controller_intent",
        "label": label,
        "operation_key": operation_key,
        "envelope_sha256": envelope_digest,
        "request_sha256": request_digest,
        "launch_bundle_sha256": bundle_digest,
        "stage_event_sha256": stage_sha,
        "request_only": True,
        "current_profile_changed": False,
    }
    return binding, request


def _reject_create_authority(
    execution_root: Path,
    events: Sequence[Any],
    *,
    allowed_request_only: Mapping[str, Any] | None = None,
) -> None:
    if (execution_root / "execution_manifest.json").exists():
        raise PermissionError("prelaunch abort refuses an execution manifest")
    create = [
        _event_value(event)
        for event in events
        if _event_value(event)["phase"] in {"authorize-launch", "reconcile-create"}
    ]
    if create:
        raise PermissionError("prelaunch abort refuses any GCE create event")
    observed = _persisted_request_only_authority(execution_root, events)
    if observed is None:
        if allowed_request_only is not None:
            raise PermissionError("bound request-only launch artifact is absent")
        return
    binding, _ = observed
    if allowed_request_only is None:
        raise PermissionError("prelaunch abort refuses a persisted GCE create request")
    allowed = _clone(allowed_request_only, "allowed request-only authority")
    if set(allowed) != _PERSISTED_AUTHORIZE_REQUEST_KEYS or allowed != binding:
        raise PermissionError("persisted request-only launch binding changed")


@dataclass(frozen=True)
class PreparedPrelaunchAbort:
    root: Path
    execution_root: Path
    journal_dir: Path
    wave_plan: dict[str, Any]
    attempt_ledger: dict[str, Any]
    resume_plan: dict[str, Any]
    abort_plan: dict[str, Any]
    content_binding: dict[str, str]
    stage_plan: dict[str, Any]
    preflight_receipt: dict[str, Any]
    stage_receipt: dict[str, Any]
    claim_receipt: dict[str, Any]
    raw_claim_nonce: str
    iam_plan: dict[str, Any]
    iam_prepare_receipt: dict[str, Any]
    iam_install_receipt: dict[str, Any]
    iam_readback_receipt: dict[str, Any]


def _validate_persisted_plan(
    *, value: Mapping[str, Any], plan: Mapping[str, Any], ledger: Mapping[str, Any],
    resume: Mapping[str, Any], namespace: str,
) -> dict[str, Any]:
    payload = _clone(value, "prelaunch abort plan")
    schema = payload.get("schema")
    expected_keys = (
        _LEGACY_ABORT_PLAN_KEYS
        if schema == LEGACY_ABORT_PLAN_SCHEMA
        else _ABORT_PLAN_KEYS
    )
    if set(payload) != expected_keys:
        raise ValueError("prelaunch abort plan fields changed")
    digest = payload.pop("plan_sha256", None)
    if _sha(digest, "abort plan") != canonical_sha256(payload):
        raise ValueError("prelaunch abort plan digest changed")
    payload["plan_sha256"] = digest
    if (
        schema not in {LEGACY_ABORT_PLAN_SCHEMA, ABORT_PLAN_SCHEMA}
        or payload.get("status") != "prelaunch_abort_plan_ready_cloud_not_mutated"
        or payload.get("execution_namespace") != namespace
        or payload.get("run_name") != plan["run_name"]
        or payload.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or payload.get("wave_plan_sha256") != plan["schedule_sha256"]
        or payload.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or payload.get("resume_plan_sha256") != resume["resume_sha256"]
        or payload.get("wave_index") != resume["resume_wave_index"]
        or payload.get("execution_manifest_present") is not False
        or payload.get("gce_create_event_count") != 0
        or payload.get("claim_delete_authorized") is not False
        or payload.get("service_account_delete_authorized") is not False
        or payload.get("content_delete_authorized") is not False
        or payload.get("cloud_mutation_performed") is not False
        or payload.get("prelaunch_abort_tombstone_required") is not True
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("prelaunch abort plan binding changed")
    if schema == ABORT_PLAN_SCHEMA:
        request_binding = payload.get("persisted_authorize_request")
        if request_binding is not None and (
            not isinstance(request_binding, Mapping)
            or set(request_binding) != _PERSISTED_AUTHORIZE_REQUEST_KEYS
            or request_binding.get("schema")
            != "hu_m31_t3_step6d_full100_wave_request_only_authority_v1"
            or request_binding.get("status")
            != "persisted_authorize_request_before_controller_intent"
            or request_binding.get("operation_key")
            != f"{namespace}:{request_binding.get('label')}"
            or request_binding.get("request_only") is not True
            or request_binding.get("current_profile_changed") is not False
        ):
            raise ValueError("prelaunch abort request-only authority binding changed")
    return payload


def _load_prepared(
    *, execution_root: Path, root: Path,
) -> PreparedPrelaunchAbort:
    plan = wave_v2.validate_wave_plan(_read_json(root / "inputs/wave_plan.json", "wave plan"))
    ledger = wave_v2.validate_attempt_ledger(
        plan, _read_json(root / "inputs/attempt_ledger.json", "attempt ledger")
    )
    resume = wave_v2.validate_resume_plan(
        plan, ledger, _read_json(root / "inputs/resume_plan.json", "resume plan")
    )
    source_plan = wave_v2.validate_wave_plan(
        _read_json(execution_root / "inputs/wave_plan.json", "source wave plan")
    )
    source_ledger = wave_v2.validate_attempt_ledger(
        source_plan,
        _read_json(execution_root / "inputs/attempt_ledger.json", "source attempt ledger"),
    )
    source_resume = wave_v2.validate_resume_plan(
        source_plan,
        source_ledger,
        _read_json(execution_root / "inputs/resume_plan.json", "source resume plan"),
    )
    if (source_plan, source_ledger, source_resume) != (plan, ledger, resume):
        raise ValueError("prelaunch abort copied inputs differ from execution source")
    namespace = root.name
    abort_plan = _validate_persisted_plan(
        value=_read_json(root / "abort_plan.json", "abort plan"),
        plan=plan, ledger=ledger, resume=resume, namespace=namespace,
    )
    journal_dir = Path(abort_plan["controller_journal_dir"])
    expected_journal_dir = (execution_root / "controller-journal").resolve()
    if journal_dir.resolve() != expected_journal_dir:
        raise ValueError("prelaunch abort controller journal escaped execution root")
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=journal_dir, wave_plan=plan, attempt_ledger=ledger,
        resume_plan=resume, create_journal=False,
    )
    events = controller.journal.load()
    request_binding = abort_plan.get("persisted_authorize_request")
    _reject_create_authority(
        execution_root, events, allowed_request_only=request_binding
    )
    request_only = _persisted_request_only_authority(execution_root, events)
    if request_only is not None:
        observed_binding, request = request_only
        if observed_binding != request_binding:
            raise ValueError("prelaunch abort copied authorize request changed")
        controller_v2._launch_material(controller, request)
    if not any(
        event.value["event_sha256"] == abort_plan["anchor_event_sha256"]
        for event in events
    ):
        raise ValueError("prelaunch abort anchor event is absent")
    content = _read_json(root / "inputs/content_binding.json", "content binding")
    stage_plan = _read_json(root / "inputs/stage_plan.json", "stage plan")
    preflight = _read_json(root / "inputs/preflight_receipt.json", "content preflight")
    stage = _read_json(root / "inputs/stage_receipt.json", "content stage")
    claim = _read_json(root / "inputs/claim_receipt.json", "claim receipt")
    random_material = _read_json(root / "private/random_material.json", "random material")
    iam_plan = _read_json(root / "inputs/iam_plan.json", "IAM plan")
    iam_prepare = _read_json(root / "inputs/iam_prepare_receipt.json", "IAM prepare")
    iam_install = _read_json(root / "inputs/iam_install_receipt.json", "IAM install")
    iam_readback = _read_json(root / "inputs/iam_readback_receipt.json", "IAM readback")
    source_outer = _read_json(
        execution_root / "static/outer_manifest.json", "source outer manifest"
    )
    source_stage_plan = content_v2.validate_content_stage_plan(
        source_outer,
        _read_json(
            execution_root / "static/content_stage_plan.json", "source stage plan"
        ),
    )
    source_random_material = _read_json(
        execution_root / "private/random_material.json", "source random material"
    )
    if source_stage_plan != stage_plan or source_random_material != random_material:
        raise ValueError("prelaunch abort copied static/private source changed")
    if set(content) != {
        "immutable_content_prefix", "content_payload_sha256", "outer_manifest_sha256"
    }:
        raise ValueError("abort content binding changed")
    preflight = content_v2.validate_preflight_absence_receipt(stage_plan, preflight)
    stage = content_v2.validate_stage_receipt(stage_plan, preflight, stage)
    if stage.get("created_entry_count") != 26 or stage.get("stage_complete") is not True:
        raise ValueError("abort content stage is incomplete")
    iam_plan = worker_iam_v2.validate_worker_iam_plan(
        wave_plan=plan, attempt_ledger=ledger, resume_plan=resume,
        immutable_content_prefix=content["immutable_content_prefix"],
        content_payload_sha256=content["content_payload_sha256"],
        outer_manifest_sha256=content["outer_manifest_sha256"], value=iam_plan,
    )
    iam_prepare = worker_iam_v2.validate_prepare_receipt(
        iam_plan=iam_plan, wave_plan=plan, attempt_ledger=ledger,
        resume_plan=resume, value=iam_prepare, **content,
    )
    iam_install = worker_iam_v2.validate_install_receipt(
        iam_plan=iam_plan, wave_plan=plan, attempt_ledger=ledger,
        resume_plan=resume, prepare_receipt=iam_prepare, value=iam_install,
        **content,
    )
    iam_readback = worker_iam_v2.validate_readback_receipt(
        iam_plan=iam_plan, wave_plan=plan, attempt_ledger=ledger,
        resume_plan=resume, prepare_receipt=iam_prepare,
        install_receipt=iam_install, value=iam_readback, **content,
    )
    nonce = random_material.get("claim_nonce")
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("abort claim nonce is absent")
    claim = cloud_v2.validate_persistent_atomic_launch_claim_receipt(
        plan, ledger, resume, claim,
        immutable_content_sha256=content["content_payload_sha256"],
        raw_claim_nonce=nonce,
    )
    selected_ids = [row["instance_id"] for row in resume["selected_attempts"]]
    if (
        abort_plan["controller_event_count_at_plan"] < 1
        or abort_plan["selected_instance_ids"] != selected_ids
        or abort_plan["worker_iam_plan_sha256"] != iam_plan["plan_sha256"]
        or abort_plan["worker_iam_binding_count"] != iam_plan["exact_binding_count"]
        or abort_plan["worker_iam_cleanup_not_before_utc"]
        != _worker_iam_cleanup_not_before(iam_plan)
        or abort_plan["claim_receipt_sha256"] != claim["receipt_sha256"]
        or abort_plan["claim_object_path"] != claim["object_path"]
        or abort_plan["stage_receipt_sha256"] != stage["receipt_sha256"]
        or abort_plan["content_entry_count"] != 26
    ):
        raise ValueError("prelaunch abort persisted inputs changed")
    return PreparedPrelaunchAbort(
        root=root, execution_root=execution_root, journal_dir=journal_dir,
        wave_plan=plan, attempt_ledger=ledger, resume_plan=resume,
        abort_plan=abort_plan, content_binding=content, stage_plan=stage_plan,
        preflight_receipt=preflight, stage_receipt=stage,
        claim_receipt=claim, raw_claim_nonce=nonce, iam_plan=iam_plan,
        iam_prepare_receipt=iam_prepare, iam_install_receipt=iam_install,
        iam_readback_receipt=iam_readback,
    )


def prepare_prelaunch_abort(
    *,
    execution_root: str | Path,
    abort_root: str | Path,
    allow_persisted_authorize_request_only: bool = False,
) -> PreparedPrelaunchAbort:
    execution = Path(execution_root).resolve()
    if not execution.is_dir() or execution.is_symlink():
        raise ValueError("execution_root must be a plain directory")
    plan = wave_v2.validate_wave_plan(
        _read_json(execution / "inputs/wave_plan.json", "wave plan")
    )
    ledger = wave_v2.validate_attempt_ledger(
        plan, _read_json(execution / "inputs/attempt_ledger.json", "attempt ledger")
    )
    resume = wave_v2.validate_resume_plan(
        plan, ledger, _read_json(execution / "inputs/resume_plan.json", "resume plan")
    )
    orchestration = _read_json(
        execution / "orchestration_plan.json", "orchestration plan"
    )
    orchestration_core = _clone(orchestration, "orchestration plan")
    orchestration_digest = orchestration_core.pop("plan_sha256", None)
    if _sha(orchestration_digest, "orchestration plan") != canonical_sha256(
        orchestration_core
    ):
        raise ValueError("execution root orchestration plan digest changed")
    namespace = orchestration.get("execution_namespace")
    if (
        not isinstance(namespace, str)
        or orchestration.get("status")
        != "validated_local_plan_cloud_not_started"
        or execution.name != namespace
        or orchestration.get("run_name") != plan["run_name"]
        or orchestration.get("wave_plan_sha256") != plan["schedule_sha256"]
        or orchestration.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or orchestration.get("resume_plan_sha256") != resume["resume_sha256"]
        or orchestration.get("cloud_started") is not False
        or orchestration.get("dry_run_cloud_mutation_performed") is not False
        or orchestration.get("current_profile_changed") is not False
    ):
        raise ValueError("execution root orchestration binding changed")
    base = Path(abort_root).resolve()
    root = base if base.name == namespace else base / namespace
    if (root / "abort_plan.json").exists():
        return _load_prepared(execution_root=execution, root=root)
    journal_dir = execution / "controller-journal"
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=journal_dir, wave_plan=plan, attempt_ledger=ledger,
        resume_plan=resume, create_journal=False,
    )
    events = controller.journal.load()
    if not events:
        raise PermissionError("prelaunch abort requires a non-empty controller journal")
    request_only = _persisted_request_only_authority(execution, events)
    request_binding = None if request_only is None else request_only[0]
    if request_binding is not None and allow_persisted_authorize_request_only is not True:
        raise PermissionError(
            "request-only prelaunch abort requires explicit authorization"
        )
    _reject_create_authority(
        execution, events, allowed_request_only=request_binding
    )
    if request_only is not None:
        controller_v2._launch_material(controller, request_only[1])
    if controller.inspect()["pending_mutation_operations"]:
        raise PermissionError("prelaunch abort refuses unresolved mutation intents")
    if any(
        _event_value(event)["phase"] == "prelaunch-abort-tombstone"
        for event in events
    ):
        raise PermissionError(
            "prelaunch abort tombstone exists without its persisted abort plan"
        )

    outer = _read_json(execution / "static/outer_manifest.json", "outer manifest")
    stage_plan = content_v2.validate_content_stage_plan(
        outer, _read_json(execution / "static/content_stage_plan.json", "stage plan")
    )
    content = {
        "immutable_content_prefix": stage_plan["content_prefix"],
        "content_payload_sha256": stage_plan["content_payload_sha256"],
        "outer_manifest_sha256": stage_plan["outer_manifest_sha256"],
    }
    preflight_event = _single_event(
        events, phases={"content-prefix-preflight"}, label="content preflight"
    )
    preflight = content_v2.validate_preflight_absence_receipt(
        stage_plan, preflight_event["output"]["content_preflight_receipt"]
    )
    stage_event = _single_event(events, phases={"stage-content"}, label="content stage")
    stage = content_v2.validate_stage_receipt(
        stage_plan, preflight, stage_event["output"]["stage_receipt"]
    )
    if stage.get("stage_complete") is not True or stage.get("created_entry_count") != 26:
        raise ValueError("prelaunch abort requires the exact 26-object stage receipt")
    claim_event = _single_event(events, phases={"persistent-claim"}, label="claim")
    claim = claim_event["output"]["receipt"]
    random_material = _read_json(
        execution / "private/random_material.json", "random material"
    )
    nonce = random_material.get("claim_nonce")
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("claim nonce is absent")
    claim = cloud_v2.validate_persistent_atomic_launch_claim_receipt(
        plan, ledger, resume, claim,
        immutable_content_sha256=content["content_payload_sha256"],
        raw_claim_nonce=nonce,
    )
    iam_prepare_event = _single_event(
        events, phases={"worker-iam-prepare"}, label="worker IAM prepare"
    )
    iam_prepare = iam_prepare_event["output"]["worker_iam_prepare_receipt"]
    iam_sha = iam_prepare["iam_plan_sha256"]
    iam_plan_event = _single_event(
        events, phases={"worker-iam-plan"},
        predicate=lambda event: event.get("output", {}).get(
            "worker_iam_plan", {}
        ).get("plan_sha256") == iam_sha,
        label="installed worker IAM plan",
    )
    iam_plan = iam_plan_event["output"]["worker_iam_plan"]
    install_event = _single_event(
        events, phases={"worker-iam-install", "worker-iam-reconcile-install"},
        predicate=lambda event: isinstance(event.get("output"), Mapping)
        and isinstance(event["output"].get("receipt"), Mapping),
        label="worker IAM install",
    )
    iam_install = install_event["output"]["receipt"]
    readback_event = _single_event(
        events, phases={"worker-iam-readback"},
        predicate=lambda event: event.get("output", {}).get(
            "worker_iam_readback_receipt", {}
        ).get("install_receipt_sha256") == iam_install.get("receipt_sha256"),
        label="worker IAM readback",
    )
    iam_readback = readback_event["output"]["worker_iam_readback_receipt"]
    # Persist first, then use the same validators as restart loading.
    anchor = _event_value(events[-1])
    core = {
        "schema": ABORT_PLAN_SCHEMA,
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
        "controller_event_count_at_plan": len(events),
        "selected_instance_ids": [
            row["instance_id"] for row in resume["selected_attempts"]
        ],
        "worker_iam_plan_sha256": iam_sha,
        "worker_iam_binding_count": iam_plan["exact_binding_count"],
        "worker_iam_cleanup_not_before_utc": _worker_iam_cleanup_not_before(iam_plan),
        "claim_receipt_sha256": claim["receipt_sha256"],
        "claim_object_path": claim["object_path"],
        "stage_receipt_sha256": stage["receipt_sha256"],
        "content_entry_count": 26,
        "execution_manifest_present": False,
        "gce_create_event_count": 0,
        "claim_delete_authorized": False,
        "service_account_delete_authorized": False,
        "content_delete_authorized": False,
        "cloud_mutation_performed": False,
        "prelaunch_abort_tombstone_required": True,
        "persisted_authorize_request": request_binding,
        "current_profile_changed": False,
    }
    abort_plan = {**core, "plan_sha256": canonical_sha256(core)}
    files = {
        "wave_plan": plan, "attempt_ledger": ledger, "resume_plan": resume,
        "content_binding": content, "stage_plan": stage_plan,
        "preflight_receipt": preflight, "stage_receipt": stage,
        "claim_receipt": claim, "iam_plan": iam_plan,
        "iam_prepare_receipt": iam_prepare, "iam_install_receipt": iam_install,
        "iam_readback_receipt": iam_readback,
    }
    for name, value in files.items():
        _write_once_json(root / "inputs" / f"{name}.json", value)
    _write_once_json(root / "private/random_material.json", random_material)
    _write_once_json(root / "abort_plan.json", abort_plan)
    return _load_prepared(execution_root=execution, root=root)


class ProductionPrelaunchAbortCleanupV2:
    def __init__(
        self, *, prepared: PreparedPrelaunchAbort, allow_cloud_read: bool,
        allow_worker_iam_cleanup: bool, confirm_run_name: str,
        recovery_only: bool = False,
        dispatcher: Dispatcher = controller_v2.execute_mode_request,
        requester: Callable[..., Any] | None = None,
        compute_absence_reader: ComputeAbsenceReader | None = None,
        iam_intent_recovery_reader: IamIntentRecoveryReader | None = None,
        preservation_reader: PreservationReader | None = None,
        clock: Clock = _system_clock,
    ) -> None:
        if confirm_run_name != prepared.wave_plan["run_name"]:
            raise PermissionError("confirm_run_name must exactly match the validated wave")
        if allow_cloud_read is not True or (
            recovery_only is not True and allow_worker_iam_cleanup is not True
        ):
            raise PermissionError(
                "prelaunch abort requires allow_cloud_read and, outside "
                "recovery-only mode, allow_worker_iam_cleanup"
            )
        if not callable(dispatcher) or not callable(clock):
            raise TypeError("dispatcher and clock must be callable")
        self.prepared = prepared
        self.dispatcher = dispatcher
        self.requester = requester
        self.compute_absence_reader = compute_absence_reader
        self.iam_intent_recovery_reader = iam_intent_recovery_reader
        self.preservation_reader = preservation_reader
        self.clock = clock
        self.recovery_only = recovery_only is True
        self.controller = controller_v2.Full100WaveControllerV2(
            journal_dir=prepared.journal_dir, wave_plan=prepared.wave_plan,
            attempt_ledger=prepared.attempt_ledger,
            resume_plan=prepared.resume_plan, create_journal=False,
        )
        _reject_create_authority(
            prepared.execution_root,
            self.controller.journal.load(),
            allowed_request_only=prepared.abort_plan.get(
                "persisted_authorize_request"
            ),
        )
        self.flags = {
            "allow_cloud_read": True,
            "allow_identity_create": False,
            "allow_content_stage": False,
            "allow_gce_create": False,
            "allow_gce_delete": False,
            "allow_content_delete": False,
            "allow_claim_create": False,
            "allow_worker_iam_install": not self.recovery_only,
            "allow_launch_authorization": False,
        }

    def _now_dt(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise TypeError("clock must return timezone-aware datetime")
        return value.astimezone(timezone.utc).replace(microsecond=0)

    def _revalidate_execution_source(self) -> None:
        # Public preparation always persists this file.  The guard permits only
        # deliberately in-memory unit fixtures to omit it.
        if not (self.prepared.root / "abort_plan.json").exists():
            return
        fresh = _load_prepared(
            execution_root=self.prepared.execution_root,
            root=self.prepared.root,
        )
        if (
            fresh.abort_plan != self.prepared.abort_plan
            or fresh.wave_plan != self.prepared.wave_plan
            or fresh.attempt_ledger != self.prepared.attempt_ledger
            or fresh.resume_plan != self.prepared.resume_plan
            or fresh.stage_plan != self.prepared.stage_plan
            or fresh.claim_receipt != self.prepared.claim_receipt
            or fresh.iam_plan != self.prepared.iam_plan
        ):
            raise ValueError("prelaunch abort execution source binding changed")

    def _op(self, label: str) -> str:
        value = f"{self.prepared.abort_plan['execution_namespace']}:{label}"
        if len(value) > 128:
            raise ValueError("prelaunch abort operation key is too long")
        return value

    def _request_path(self, label: str) -> Path:
        return self.prepared.root / "steps" / label / "request.json"

    def _materialize_request(
        self, *, label: str, mode: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        path = self._request_path(label)
        if path.exists():
            envelope = _read_json(path, f"{label} request")
            digest = envelope.pop("envelope_sha256", None)
            if _sha(digest, "request envelope") != canonical_sha256(envelope):
                raise ValueError(f"{label} request envelope digest changed")
            request = envelope.get("request")
            if (
                envelope.get("schema") != ABORT_REQUEST_SCHEMA
                or envelope.get("execution_namespace")
                != self.prepared.abort_plan["execution_namespace"]
                or envelope.get("label") != label or envelope.get("mode") != mode
                or envelope.get("request_sha256") != canonical_sha256(request)
                or envelope.get("current_profile_changed") is not False
                or request.get("operation_key") != self._op(label)
            ):
                raise ValueError(f"{label} request binding changed")
            _assert_no_credentials(request, f"{label} request")
            return request
        request = _clone(builder(), f"{label} request")
        if request.get("operation_key") != self._op(label):
            raise ValueError(f"{label} request operation key changed")
        _assert_no_credentials(request, f"{label} request")
        core = {
            "schema": ABORT_REQUEST_SCHEMA,
            "execution_namespace": self.prepared.abort_plan["execution_namespace"],
            "label": label, "mode": mode, "request": request,
            "request_sha256": canonical_sha256(request),
            "current_profile_changed": False,
        }
        _write_once_json(path, {**core, "envelope_sha256": canonical_sha256(core)})
        return self._materialize_request(label=label, mode=mode, builder=builder)

    def _journal_event(self, sha: str) -> dict[str, Any]:
        rows = [
            _event_value(event) for event in self.controller.journal.load()
            if event.value["event_sha256"] == sha
        ]
        if len(rows) != 1:
            raise ValueError("persisted abort event is absent from controller journal")
        return rows[0]

    def _terminal_for(self, operation_key: str) -> dict[str, Any] | None:
        rows = [
            _event_value(event) for event in self.controller.journal.load()
            if event.value["operation_key"] == operation_key
            and event.value["status"] in {
                "complete", "partial", "failed", "reconciled-complete",
                "reconciled-partial",
            }
        ]
        if len(rows) > 1:
            raise ValueError("abort operation has multiple terminal events")
        return None if not rows else rows[0]

    def _persist_event(
        self, *, label: str, mode: str, request: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        checked = _event_value(event)
        if (
            checked["operation_key"] != request["operation_key"]
            or checked != self._journal_event(checked["event_sha256"])
        ):
            raise ValueError("abort dispatcher event differs from journal")
        event_path = self.prepared.root / "steps" / label / "event.json"
        _write_once_json(event_path, checked)
        checkpoint = {
            "schema": ABORT_CHECKPOINT_SCHEMA,
            "execution_namespace": self.prepared.abort_plan["execution_namespace"],
            "label": label, "mode": mode,
            "request_sha256": canonical_sha256(request),
            "event_sha256": checked["event_sha256"],
            "current_profile_changed": False,
        }
        checkpoint_path = (
            self.prepared.root / "checkpoints" / f"{checked['event_sha256']}.json"
        )
        _write_once_json(checkpoint_path, checkpoint)
        if _read_json(checkpoint_path, "abort checkpoint") != checkpoint:
            raise ValueError("abort checkpoint binding changed")
        return checked

    def _dispatch(
        self, *, label: str, mode: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        event_path = self.prepared.root / "steps" / label / "event.json"
        if (
            self.recovery_only
            and mode == "worker-iam-cleanup"
            and not event_path.is_file()
        ):
            raise PermissionError(
                "recovery-only mode forbids a new worker IAM cleanup request"
            )
        request = self._materialize_request(label=label, mode=mode, builder=builder)
        if event_path.exists():
            event = _read_json(event_path, f"{label} event")
            return self._persist_event(
                label=label, mode=mode, request=request, event=event
            )
        if self.recovery_only:
            if mode not in {
                "worker-iam-readback", "worker-iam-reconcile-cleanup"
            }:
                raise PermissionError(
                    "recovery-only mode forbids this dispatcher capability"
                )
            if mode == "worker-iam-readback" and not event_path.is_file():
                raise PermissionError(
                    "recovery-only mode requires the historical IAM readback"
                )
        kwargs = {
            "controller": self.controller, "mode": mode, "request": request,
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
        return self._persist_event(
            label=label, mode=mode, request=request, event=event
        )

    def _validate_recovery_only_source(self) -> None:
        """Require one immutable outcome-ambiguous cleanup source operation."""

        if not self.recovery_only:
            return
        label = "prelaunch-abort-worker-iam-cleanup"
        operation_key = self._op(label)
        rows = [
            _event_value(event)
            for event in self.controller.journal.load()
            if event.value["operation_key"] == operation_key
        ]
        intents = [event for event in rows if event["status"] == "intent"]
        failures = [event for event in rows if event["status"] == "failed"]
        if len(rows) != 2 or len(intents) != 1 or len(failures) != 1:
            raise PermissionError(
                "recovery-only mode requires exactly one cleanup intent/failure"
            )
        intent, failure = intents[0], failures[0]
        if (
            intent.get("phase") != "worker-iam-cleanup"
            or intent.get("mode") != "mutation"
            or intent.get("mutation_requested") is not True
            or intent.get("mutation_outcome") != "pending"
            or failure.get("phase") != "worker-iam-cleanup"
            or failure.get("mode") != "mutation"
            or failure.get("predecessor_event_sha256")
            != intent.get("event_sha256")
            or failure.get("sequence") != intent.get("sequence") + 1
            or failure.get("mutation_requested") is not True
            or failure.get("mutation_outcome") != "unknown"
            or failure.get("output", {}).get("failure_kind")
            not in {"transport_ambiguity", "local_or_response_failure"}
        ):
            raise PermissionError(
                "recovery-only cleanup source is not outcome-ambiguous"
            )
        request, persisted_failure = self._validate_persisted_step(
            label=label,
            mode="worker-iam-cleanup",
            event_sha256=failure["event_sha256"],
        )
        if (
            request.get("operation_key") != operation_key
            or persisted_failure != failure
        ):
            raise PermissionError("recovery-only cleanup source binding changed")

    def _record_tombstone(self) -> dict[str, Any]:
        label = "prelaunch-abort-tombstone"
        mode = "prelaunch-abort-tombstone"
        request = self._materialize_request(
            label=label,
            mode=mode,
            builder=lambda: {
                "operation_key": self._op(label),
                "predecessor_event_sha256": self.prepared.abort_plan[
                    "anchor_event_sha256"
                ],
                "abort_plan_sha256": self.prepared.abort_plan["plan_sha256"],
                "recorded_at_utc": _render_utc(self._now_dt()),
            },
        )
        event_path = self.prepared.root / "steps" / label / "event.json"
        if event_path.exists():
            event = self._persist_event(
                label=label,
                mode=mode,
                request=request,
                event=_read_json(event_path, "prelaunch abort tombstone event"),
            )
        else:
            raw = self.controller.record_prelaunch_abort_tombstone(
                operation_key=request["operation_key"],
                predecessor_event_sha256=request["predecessor_event_sha256"],
                abort_plan_sha256=request["abort_plan_sha256"],
                recorded_at_utc=request["recorded_at_utc"],
            )
            event = self._persist_event(
                label=label,
                mode=mode,
                request=request,
                event=_event_value(raw),
            )
        output = event.get("output")
        if (
            event.get("phase") != "prelaunch-abort-tombstone"
            or event.get("status") != "complete"
            or event.get("mode") != "read-only"
            or event.get("mutation_requested") is not False
            or event.get("mutation_outcome") != "not_requested"
            or event.get("predecessor_event_sha256")
            != request["predecessor_event_sha256"]
            or not isinstance(output, Mapping)
            or output.get("schema")
            != "hu_m31_t3_step6d_full100_wave_prelaunch_abort_tombstone_v2"
            or output.get("status")
            != "old_execution_launch_permanently_vetoed"
            or output.get("abort_plan_sha256") != request["abort_plan_sha256"]
            or output.get("run_name") != self.prepared.wave_plan["run_name"]
            or output.get("execution_identity_sha256")
            != self.prepared.wave_plan["execution_identity_sha256"]
            or output.get("old_execution_relaunch_authorized") is not False
            or output.get("additional_create_authorized") is not False
            or output.get("current_profile_changed") is not False
        ):
            raise ValueError("prelaunch abort tombstone binding changed")
        return event

    def _validate_compute_absence_receipt(
        self, value: Mapping[str, Any], request: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = _sealed(value)
        if set(receipt) != {
            "schema", "status", "run_name", "execution_identity_sha256",
            "wave_plan_sha256", "attempt_ledger_sha256", "resume_plan_sha256",
            "project", "zone", "selected_instance_ids", "rows",
            "selected_instance_count", "http_get_count",
            "all_selected_instances_absent", "all_selected_boot_disks_absent",
            "read_only", "cloud_mutation_performed", "additional_create_authorized",
            "observed_at_utc", "expires_at_utc", "current_profile_changed",
            "receipt_sha256",
        }:
            raise ValueError("prelaunch abort compute absence fields changed")
        selected_ids = [
            row["instance_id"] for row in self.prepared.resume_plan["selected_attempts"]
        ]
        expected_rows = [
            {
                "instance_id": instance_id,
                "instance_absent": True,
                "boot_disk_absent": True,
            }
            for instance_id in selected_ids
        ]
        if (
            receipt.get("schema") != COMPUTE_ABSENCE_RECEIPT_SCHEMA
            or receipt.get("status")
            != "exact_selected_instances_and_boot_disks_absent_get_only"
            or receipt.get("run_name") != self.prepared.wave_plan["run_name"]
            or receipt.get("execution_identity_sha256")
            != self.prepared.wave_plan["execution_identity_sha256"]
            or receipt.get("wave_plan_sha256")
            != self.prepared.wave_plan["schedule_sha256"]
            or receipt.get("attempt_ledger_sha256")
            != self.prepared.attempt_ledger["ledger_sha256"]
            or receipt.get("resume_plan_sha256")
            != self.prepared.resume_plan["resume_sha256"]
            or receipt.get("project") != gcp_v2.PROJECT
            or receipt.get("zone") != gcp_v2.ZONE
            or receipt.get("selected_instance_ids") != selected_ids
            or receipt.get("rows") != expected_rows
            or receipt.get("selected_instance_count") != len(selected_ids)
            or receipt.get("http_get_count") != 2 * len(selected_ids)
            or receipt.get("all_selected_instances_absent") is not True
            or receipt.get("all_selected_boot_disks_absent") is not True
            or receipt.get("read_only") is not True
            or receipt.get("cloud_mutation_performed") is not False
            or receipt.get("additional_create_authorized") is not False
            or receipt.get("current_profile_changed") is not False
            or receipt.get("observed_at_utc") != request["observed_at_utc"]
            or receipt.get("expires_at_utc") != request["expires_at_utc"]
        ):
            raise ValueError("prelaunch abort compute absence receipt changed")
        observed = _parse_utc(receipt["observed_at_utc"])
        expires = _parse_utc(receipt["expires_at_utc"])
        if expires - observed != timedelta(seconds=COMPUTE_ABSENCE_FRESHNESS_SECONDS):
            raise ValueError("prelaunch abort compute absence freshness changed")
        return receipt

    def _default_compute_absence_reader(
        self, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        requester = self.requester or controller_v2._stdlib_http_request
        token = os.environ.get(gcp_v2.TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            char.isspace() for char in token
        ):
            raise PermissionError(
                f"Bearer token must be supplied only through {gcp_v2.TOKEN_ENV}"
            )
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        rows = []
        for instance_id in request["selected_instance_ids"]:
            encoded = urllib.parse.quote(instance_id, safe="")
            for resource in ("instances", "disks"):
                url = (
                    "https://compute.googleapis.com/compute/v1/projects/"
                    f"{gcp_v2.PROJECT}/zones/{gcp_v2.ZONE}/{resource}/{encoded}"
                )
                try:
                    response = requester("GET", url, headers, None, 60)
                except Exception as exc:
                    raise gcp_v2.GcpPhaseATransportError(
                        "prelaunch abort compute absence GET transport failed"
                    ) from exc
                if not isinstance(response, gcp_v2.HttpResponse):
                    raise RuntimeError("compute absence requester returned invalid response")
                if response.status == 200:
                    raise FileExistsError(
                        f"selected GCE {resource[:-1]} is not absent"
                    )
                if response.status != 404:
                    raise RuntimeError(
                        f"compute absence GET failed with status {response.status}"
                    )
            rows.append(
                {
                    "instance_id": instance_id,
                    "instance_absent": True,
                    "boot_disk_absent": True,
                }
            )
        core = {
            "schema": COMPUTE_ABSENCE_RECEIPT_SCHEMA,
            "status": "exact_selected_instances_and_boot_disks_absent_get_only",
            "run_name": self.prepared.wave_plan["run_name"],
            "execution_identity_sha256": self.prepared.wave_plan[
                "execution_identity_sha256"
            ],
            "wave_plan_sha256": self.prepared.wave_plan["schedule_sha256"],
            "attempt_ledger_sha256": self.prepared.attempt_ledger["ledger_sha256"],
            "resume_plan_sha256": self.prepared.resume_plan["resume_sha256"],
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
        return {**core, "receipt_sha256": canonical_sha256(core)}

    def _fresh_compute_absence(
        self, *, base_label: str, predecessor: str
    ) -> dict[str, Any]:
        mode = "prelaunch-abort-compute-absence"
        selected_ids = [
            row["instance_id"] for row in self.prepared.resume_plan["selected_attempts"]
        ]
        current_predecessor = predecessor
        for round_index in range(MAX_RECOVERY_ROUNDS):
            label = f"{base_label}-r{round_index:02d}"
            now = self._now_dt()
            request = self._materialize_request(
                label=label,
                mode=mode,
                builder=lambda now=now, label=label, predecessor=current_predecessor: {
                    "operation_key": self._op(label),
                    "predecessor_event_sha256": predecessor,
                    "selected_instance_ids": selected_ids,
                    "observed_at_utc": _render_utc(now),
                    "expires_at_utc": _render_utc(
                        now + timedelta(seconds=COMPUTE_ABSENCE_FRESHNESS_SECONDS)
                    ),
                },
            )
            event_path = self.prepared.root / "steps" / label / "event.json"
            if event_path.exists():
                event = self._persist_event(
                    label=label,
                    mode=mode,
                    request=request,
                    event=_read_json(event_path, f"{label} event"),
                )
                receipt = self._validate_compute_absence_receipt(
                    event.get("output", {}).get("compute_absence_receipt"), request
                )
                if event.get("status") != "complete":
                    raise RuntimeError("compute absence read did not complete")
                if self._now_dt() < _parse_utc(receipt["expires_at_utc"]):
                    return event
                current_predecessor = event["event_sha256"]
                continue
            if self._now_dt() >= _parse_utc(request["expires_at_utc"]):
                continue
            reader = self.compute_absence_reader or self._default_compute_absence_reader
            receipt = self._validate_compute_absence_receipt(reader(request), request)
            raw = self.controller.record_read_phase(
                phase=mode,
                operation_key=request["operation_key"],
                predecessor_event_sha256=request["predecessor_event_sha256"],
                evidence={
                    "selected_instance_ids_sha256": canonical_sha256(selected_ids),
                    "http_get_count": 2 * len(selected_ids),
                },
                output={"compute_absence_receipt": receipt},
                recorded_at_utc=request["observed_at_utc"],
            )
            event = self._persist_event(
                label=label, mode=mode, request=request, event=_event_value(raw)
            )
            if self._now_dt() >= _parse_utc(receipt["expires_at_utc"]):
                current_predecessor = event["event_sha256"]
                continue
            return event
        raise RuntimeError("fresh compute absence rounds exhausted")

    def _fresh_iam_readback(self, predecessor: str) -> dict[str, Any]:
        label = "prelaunch-abort-worker-iam-readback"
        now = _render_utc(self._now_dt())
        event = self._dispatch(
            label=label, mode="worker-iam-readback",
            builder=lambda: {
                "operation_key": self._op(label),
                "predecessor_event_sha256": predecessor,
                "content_binding": self.prepared.content_binding,
                "iam_plan": self.prepared.iam_plan,
                "observed_at_utc": now,
                "prepare_receipt": self.prepared.iam_prepare_receipt,
                "install_receipt": self.prepared.iam_install_receipt,
            },
        )
        receipt = event.get("output", {}).get("worker_iam_readback_receipt")
        worker_iam_v2.validate_readback_receipt(
            iam_plan=self.prepared.iam_plan,
            wave_plan=self.prepared.wave_plan,
            attempt_ledger=self.prepared.attempt_ledger,
            resume_plan=self.prepared.resume_plan,
            prepare_receipt=self.prepared.iam_prepare_receipt,
            install_receipt=self.prepared.iam_install_receipt,
            value=receipt, **self.prepared.content_binding,
        )
        if event.get("status") != "complete":
            raise RuntimeError("fresh worker IAM readback did not complete")
        return event

    def _default_iam_intent_recovery_reader(
        self, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        requester = self.requester or controller_v2._stdlib_http_request
        common = {
            "iam_plan": self.prepared.iam_plan,
            "wave_plan": self.prepared.wave_plan,
            "attempt_ledger": self.prepared.attempt_ledger,
            "resume_plan": self.prepared.resume_plan,
            "prepare_receipt": self.prepared.iam_prepare_receipt,
            "install_receipt": self.prepared.iam_install_receipt,
            **self.prepared.content_binding,
        }
        after_adapter = controller_v2._phase_a_adapter(
            controller=self.controller,
            mode="bucket-iam-reconcile-cleanup",
            content=self.prepared.content_binding,
            requester=requester,
            iam_plan=self.prepared.iam_plan,
            prepare_receipt=self.prepared.iam_prepare_receipt,
            install_receipt=self.prepared.iam_install_receipt,
            readback_receipt=request["source_readback_receipt"],
        )
        cleanup_receipt: dict[str, Any] | None = None
        fresh_readback: dict[str, Any] | None = None
        try:
            cleanup_receipt = worker_iam_v2.reconcile_cleanup_worker_iam(
                **common,
                readback_receipt=request["source_readback_receipt"],
                backend=after_adapter,
            )
            state = "exact_desired_after_absent"
        except ValueError as after_error:
            before_adapter = controller_v2._phase_a_adapter(
                controller=self.controller,
                mode="bucket-iam-readback",
                content=self.prepared.content_binding,
                requester=requester,
                iam_plan=self.prepared.iam_plan,
                prepare_receipt=self.prepared.iam_prepare_receipt,
                install_receipt=self.prepared.iam_install_receipt,
            )
            try:
                fresh_readback = worker_iam_v2.readback_worker_iam(
                    **common, backend=before_adapter
                )
            except Exception as before_error:
                raise RuntimeError(
                    "pending IAM cleanup is neither exact desired-after nor exact before"
                ) from ExceptionGroup(
                    "IAM intent recovery probes failed", [after_error, before_error]
                )
            state = "exact_before_installed"
        core = {
            "schema": IAM_INTENT_RECOVERY_RECEIPT_SCHEMA,
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
            "cleanup_receipt": cleanup_receipt,
            "fresh_readback_receipt": fresh_readback,
            "http_method": "GET",
            "read_only": True,
            "cloud_mutation_performed": False,
            "observed_at_utc": request["observed_at_utc"],
            "current_profile_changed": False,
        }
        return {**core, "receipt_sha256": canonical_sha256(core)}

    def _validate_iam_intent_recovery_receipt(
        self, value: Mapping[str, Any], request: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = _sealed(value)
        if set(receipt) != {
            "schema", "status", "recovery_operation_key", "source_operation_key",
            "source_intent_event_sha256", "source_readback_receipt_sha256",
            "fresh_compute_absence_event_sha256", "state", "cleanup_receipt",
            "fresh_readback_receipt", "http_method", "read_only",
            "cloud_mutation_performed", "observed_at_utc",
            "current_profile_changed", "receipt_sha256",
        }:
            raise ValueError("IAM intent recovery receipt fields changed")
        if (
            receipt.get("schema") != IAM_INTENT_RECOVERY_RECEIPT_SCHEMA
            or receipt.get("status")
            != "pending_cleanup_resolved_by_get_only_state_probe"
            or receipt.get("recovery_operation_key") != request["operation_key"]
            or receipt.get("source_operation_key")
            != request["source_operation_key"]
            or receipt.get("source_intent_event_sha256")
            != request["source_intent_event_sha256"]
            or receipt.get("source_readback_receipt_sha256")
            != request["source_readback_receipt"]["receipt_sha256"]
            or receipt.get("fresh_compute_absence_event_sha256")
            != request["fresh_compute_absence_event_sha256"]
            or receipt.get("state")
            not in {"exact_desired_after_absent", "exact_before_installed"}
            or receipt.get("http_method") != "GET"
            or receipt.get("read_only") is not True
            or receipt.get("cloud_mutation_performed") is not False
            or receipt.get("observed_at_utc") != request["observed_at_utc"]
            or receipt.get("current_profile_changed") is not False
        ):
            raise ValueError("IAM intent recovery receipt changed")
        if receipt["state"] == "exact_desired_after_absent":
            if receipt.get("fresh_readback_receipt") is not None:
                raise ValueError("desired-after IAM recovery contains before receipt")
            worker_iam_v2.validate_cleanup_receipt(
                iam_plan=self.prepared.iam_plan,
                wave_plan=self.prepared.wave_plan,
                attempt_ledger=self.prepared.attempt_ledger,
                resume_plan=self.prepared.resume_plan,
                prepare_receipt=self.prepared.iam_prepare_receipt,
                install_receipt=self.prepared.iam_install_receipt,
                readback_receipt=request["source_readback_receipt"],
                value=receipt.get("cleanup_receipt"),
                **self.prepared.content_binding,
            )
        else:
            if receipt.get("cleanup_receipt") is not None:
                raise ValueError("exact-before IAM recovery contains cleanup receipt")
            worker_iam_v2.validate_readback_receipt(
                iam_plan=self.prepared.iam_plan,
                wave_plan=self.prepared.wave_plan,
                attempt_ledger=self.prepared.attempt_ledger,
                resume_plan=self.prepared.resume_plan,
                prepare_receipt=self.prepared.iam_prepare_receipt,
                install_receipt=self.prepared.iam_install_receipt,
                value=receipt.get("fresh_readback_receipt"),
                **self.prepared.content_binding,
            )
        return receipt

    def _pending_intent(self, operation_key: str) -> Any | None:
        rows = [
            event
            for event in self.controller.journal.load()
            if event.value["operation_key"] == operation_key
        ]
        intents = [event for event in rows if event.value["status"] == "intent"]
        terminals = [
            event
            for event in rows
            if event.value["status"]
            in {"complete", "partial", "failed", "reconciled-complete", "reconciled-partial"}
        ]
        if len(intents) > 1 or len(terminals) > 1:
            raise ValueError("IAM cleanup operation journal multiplicity changed")
        return intents[0] if len(intents) == 1 and not terminals else None

    def _find_recovery_event_for_receipt(
        self, value: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Bind a terminal recovery wrapper back to its immutable GET event."""

        raw = _clone(value, "IAM intent recovery terminal receipt")
        operation_key = raw.get("recovery_operation_key")
        prefix = f"{self.prepared.abort_plan['execution_namespace']}:"
        if not isinstance(operation_key, str) or not operation_key.startswith(prefix):
            raise ValueError("IAM recovery operation binding changed")
        label = operation_key.removeprefix(prefix)
        recovery_prefix = "prelaunch-abort-worker-iam-cleanup"
        if (
            not label.startswith(recovery_prefix)
            or "-intent-recovery-r" not in label
            or not label.rsplit("-intent-recovery-r", 1)[-1].isdigit()
        ):
            raise ValueError("IAM recovery label changed")
        terminal = self._terminal_for(operation_key)
        if terminal is None:
            raise ValueError("IAM recovery read event is absent")
        request, event = self._validate_persisted_step(
            label=label,
            mode="prelaunch-abort-worker-iam-intent-recovery",
            event_sha256=terminal["event_sha256"],
        )
        checked = self._validate_iam_intent_recovery_receipt(raw, request)
        compute_event = self._validated_cleanup_compute_event(
            request["fresh_compute_absence_event_sha256"]
        )
        if (
            event.get("phase")
            != "prelaunch-abort-worker-iam-intent-recovery"
            or event.get("status") != "complete"
            or event.get("mode") != "read-only"
            or event.get("mutation_requested") is not False
            or event.get("mutation_outcome") != "not_requested"
            or event.get("output", {}).get("iam_intent_recovery_receipt")
            != checked
            or event.get("predecessor_event_sha256")
            != request.get("source_intent_event_sha256")
            or compute_event["sequence"] >= event["sequence"]
        ):
            raise ValueError("IAM recovery event binding changed")
        return event, checked

    def _validated_cleanup_compute_event(
        self, event_sha256: str
    ) -> dict[str, Any]:
        """Restore the historical compute proof used by a completed IAM action."""

        compute_event = self._journal_event(event_sha256)
        prefix = f"{self.prepared.abort_plan['execution_namespace']}:"
        compute_label = compute_event["operation_key"].removeprefix(prefix)
        if not compute_label.startswith(
            "prelaunch-abort-compute-before-iam-cleanup-"
        ):
            raise ValueError("IAM recovery compute authority label changed")
        compute_request, compute_event = self._validate_persisted_step(
            label=compute_label,
            mode="prelaunch-abort-compute-absence",
            event_sha256=compute_event["event_sha256"],
        )
        self._validate_compute_absence_receipt(
            compute_event.get("output", {}).get("compute_absence_receipt"),
            compute_request,
        )
        if (
            compute_event.get("phase") != "prelaunch-abort-compute-absence"
            or compute_event.get("status") != "complete"
            or compute_event.get("mode") != "read-only"
        ):
            raise ValueError("historical IAM compute authority changed")
        return compute_event

    def _validate_stale_cleanup_abandonment(
        self,
        *,
        event: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        checked = _event_value(event)
        output = checked.get("output")
        if (
            checked.get("phase") != "worker-iam-cleanup"
            or checked.get("mode") != "read-only"
            or checked.get("status") != "failed"
            or checked.get("operation_key") != request["operation_key"]
            or checked.get("predecessor_event_sha256")
            != request["predecessor_event_sha256"]
            or checked.get("mutation_requested") is not False
            or checked.get("mutation_outcome") != "none"
            or not isinstance(output, Mapping)
            or set(output)
            != {
                "schema", "status", "failure_kind",
                "stale_predecessor_event_sha256",
                "fresh_compute_absence_event_sha256",
                "cloud_mutation_performed", "current_profile_changed",
            }
            or output.get("schema")
            != "hu_m31_t3_step6d_full100_wave_stale_cleanup_request_v2"
            or output.get("status")
            != "stale_request_permanently_abandoned_before_intent"
            or output.get("failure_kind")
            != "stale_request_abandoned_before_intent"
            or output.get("stale_predecessor_event_sha256")
            != request["predecessor_event_sha256"]
            or output.get("cloud_mutation_performed") is not False
            or output.get("current_profile_changed") is not False
        ):
            raise ValueError("stale cleanup request abandonment changed")
        stale = self._validated_cleanup_compute_event(
            request["predecessor_event_sha256"]
        )
        fresh = self._validated_cleanup_compute_event(
            output["fresh_compute_absence_event_sha256"]
        )
        if fresh["sequence"] <= stale["sequence"]:
            raise ValueError("stale cleanup request abandonment order changed")
        return checked

    def _abandon_expired_cleanup_request(
        self,
        *,
        label: str,
        operation_key: str,
        effective_request: Mapping[str, Any],
        absence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume one expired immutable cleanup request before another CAS key."""

        absence_label = absence["operation_key"].rsplit(":", 1)[-1]
        fresh_for_abandonment = self._fresh_compute_absence(
            base_label=absence_label.rsplit("-r", 1)[0],
            predecessor=absence["event_sha256"],
        )
        raw = self.controller.abandon_worker_iam_cleanup_request_before_intent(
            operation_key=operation_key,
            stale_predecessor_event_sha256=effective_request[
                "predecessor_event_sha256"
            ],
            fresh_compute_absence_event_sha256=fresh_for_abandonment[
                "event_sha256"
            ],
            recorded_at_utc=_render_utc(self._now_dt()),
        )
        abandoned = self._persist_event(
            label=label,
            mode="worker-iam-cleanup",
            request=effective_request,
            event=_event_value(raw),
        )
        return self._validate_stale_cleanup_abandonment(
            event=abandoned, request=effective_request
        )

    def _recover_pending_iam_intent(
        self, *, cleanup_label: str, source_readback: Mapping[str, Any],
        fresh_compute_absence_event_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        operation_key = self._op(cleanup_label)
        intent = self._pending_intent(operation_key)
        if intent is None:
            raise ValueError("IAM cleanup pending intent is absent")
        mode = "prelaunch-abort-worker-iam-intent-recovery"
        recovery_label = ""
        request: dict[str, Any] | None = None
        for recovery_round in range(MAX_RECOVERY_ROUNDS):
            candidate = f"{cleanup_label}-intent-recovery-r{recovery_round:02d}"
            request_path = self._request_path(candidate)
            event_path = self.prepared.root / "steps" / candidate / "event.json"
            terminal = self._terminal_for(self._op(candidate))
            if event_path.exists() or terminal is not None:
                if not request_path.is_file():
                    raise ValueError("IAM recovery journal event lacks its request")
                old_request = self._materialize_request(
                    label=candidate,
                    mode=mode,
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("IAM recovery request must exist")
                    ),
                )
                old_event = self._persist_event(
                    label=candidate,
                    mode=mode,
                    request=old_request,
                    event=(
                        _read_json(event_path, f"{candidate} event")
                        if event_path.exists()
                        else terminal
                    ),
                )
                old_receipt = self._validate_iam_intent_recovery_receipt(
                    old_event.get("output", {}).get(
                        "iam_intent_recovery_receipt"
                    ),
                    old_request,
                )
                if (
                    old_event.get("phase") != mode
                    or old_event.get("status") != "complete"
                    or old_request.get("source_operation_key") != operation_key
                    or old_request.get("source_intent_event_sha256")
                    != intent.value["event_sha256"]
                    or old_receipt.get("source_operation_key") != operation_key
                ):
                    raise ValueError("persisted IAM recovery round changed")
                # A completed GET proves only the state at that prior attempt.
                # While the source intent remains unresolved it is never reused,
                # even when its wall-clock TTL has not expired.
                continue
            if request_path.exists():
                # A crash after the immutable request but before the GET must not
                # turn that old request into authority on a later process run.
                self._materialize_request(
                    label=candidate,
                    mode=mode,
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("IAM recovery request must exist")
                    ),
                )
                continue
            candidate_request = self._materialize_request(
                label=candidate,
                mode=mode,
                builder=lambda candidate=candidate: {
                    "operation_key": self._op(candidate),
                    "predecessor_event_sha256": intent.value["event_sha256"],
                    "source_operation_key": operation_key,
                    "source_intent_event_sha256": intent.value["event_sha256"],
                    "source_readback_receipt": source_readback,
                    "fresh_compute_absence_event_sha256": (
                        fresh_compute_absence_event_sha256
                    ),
                    "observed_at_utc": _render_utc(self._now_dt()),
                },
            )
            recovery_label = candidate
            request = candidate_request
            break
        if request is None:
            raise RuntimeError("fresh IAM intent recovery rounds exhausted")
        reader = (
            self.iam_intent_recovery_reader
            or self._default_iam_intent_recovery_reader
        )
        receipt = self._validate_iam_intent_recovery_receipt(
            reader(request), request
        )
        raw = self.controller.record_read_phase(
            phase=mode,
            operation_key=request["operation_key"],
            predecessor_event_sha256=request["predecessor_event_sha256"],
            evidence={
                "source_operation_key": operation_key,
                "source_intent_event_sha256": intent.value["event_sha256"],
                "fresh_compute_absence_event_sha256": (
                    fresh_compute_absence_event_sha256
                ),
                "state": receipt["state"],
            },
            output={"iam_intent_recovery_receipt": receipt},
            recorded_at_utc=request["observed_at_utc"],
        )
        recovery_event = self._persist_event(
            label=recovery_label,
            mode=mode,
            request=request,
            event=_event_value(raw),
        )
        original_request = self._materialize_request(
            label=cleanup_label,
            mode="worker-iam-cleanup",
            builder=lambda: (_ for _ in ()).throw(
                AssertionError("pending cleanup request must already exist")
            ),
        )
        if receipt["state"] == "exact_desired_after_absent":
            terminal = self.controller.reconcile_pending_mutation(
                operation_key=operation_key,
                receipt=receipt,
                validator=lambda value: self._validate_iam_intent_recovery_receipt(
                    value, request
                ),
                complete=True,
                recorded_at_utc=request["observed_at_utc"],
            )
        else:
            terminal = self.controller.resolve_pending_mutation_no_effect(
                operation_key=operation_key,
                receipt=receipt,
                validator=lambda value: self._validate_iam_intent_recovery_receipt(
                    value, request
                ),
                recorded_at_utc=request["observed_at_utc"],
            )
        event = self._persist_event(
            label=cleanup_label,
            mode="worker-iam-cleanup",
            request=original_request,
            event=_event_value(terminal),
        )
        return event, receipt, recovery_event

    def _cleanup_iam(
        self, *, predecessor: str, fresh_readback: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        current_predecessor = predecessor
        current_readback = dict(fresh_readback)
        last_recovery_event: dict[str, Any] | None = None
        base_label = "prelaunch-abort-worker-iam-cleanup"
        for round_index in range(MAX_RECOVERY_ROUNDS):
            absence = self._fresh_compute_absence(
                base_label=(
                    "prelaunch-abort-compute-before-iam-cleanup"
                    f"-{round_index:02d}"
                ),
                predecessor=current_predecessor,
            )
            authority_absence = absence
            absence_receipt = self._validate_compute_absence_receipt(
                absence["output"]["compute_absence_receipt"],
                self._materialize_request(
                    label=absence["operation_key"].rsplit(":", 1)[-1],
                    mode="prelaunch-abort-compute-absence",
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("compute absence request must exist")
                    ),
                ),
            )
            if self._now_dt() >= _parse_utc(absence_receipt["expires_at_utc"]):
                current_predecessor = absence["event_sha256"]
                continue
            _reject_create_authority(
                self.prepared.execution_root,
                self.controller.journal.load(),
                allowed_request_only=self.prepared.abort_plan.get(
                    "persisted_authorize_request"
                ),
            )
            self._revalidate_execution_source()
            if self._now_dt() >= _parse_utc(absence_receipt["expires_at_utc"]):
                current_predecessor = absence["event_sha256"]
                continue
            label = base_label if round_index == 0 else f"{base_label}-r{round_index:02d}"
            common = {
                "content_binding": self.prepared.content_binding,
                "iam_plan": self.prepared.iam_plan,
                "prepare_receipt": self.prepared.iam_prepare_receipt,
                "install_receipt": self.prepared.iam_install_receipt,
                "readback_receipt": current_readback,
            }
            operation_key = self._op(label)
            terminal = self._terminal_for(operation_key)
            terminal_output = (
                terminal.get("output")
                if isinstance(terminal, Mapping)
                else None
            )
            if (
                isinstance(terminal_output, Mapping)
                and terminal_output.get("failure_kind")
                == "stale_request_abandoned_before_intent"
            ):
                abandoned_request = self._materialize_request(
                    label=label,
                    mode="worker-iam-cleanup",
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("abandoned cleanup request must exist")
                    ),
                )
                abandoned = self._persist_event(
                    label=label,
                    mode="worker-iam-cleanup",
                    request=abandoned_request,
                    event=terminal,
                )
                self._validate_stale_cleanup_abandonment(
                    event=abandoned, request=abandoned_request
                )
                current_predecessor = abandoned["event_sha256"]
                continue
            pending = self._pending_intent(operation_key)
            request_path = self._request_path(label)
            if pending is None and terminal is None and request_path.exists():
                request_only = self._materialize_request(
                    label=label,
                    mode="worker-iam-cleanup",
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("request-only cleanup request must exist")
                    ),
                )
                if (
                    request_only["predecessor_event_sha256"]
                    != absence["event_sha256"]
                ):
                    raw_abandoned = (
                        self.controller.abandon_worker_iam_cleanup_request_before_intent(
                            operation_key=operation_key,
                            stale_predecessor_event_sha256=request_only[
                                "predecessor_event_sha256"
                            ],
                            fresh_compute_absence_event_sha256=absence[
                                "event_sha256"
                            ],
                            recorded_at_utc=_render_utc(self._now_dt()),
                        )
                    )
                    abandoned = self._persist_event(
                        label=label,
                        mode="worker-iam-cleanup",
                        request=request_only,
                        event=_event_value(raw_abandoned),
                    )
                    self._validate_stale_cleanup_abandonment(
                        event=abandoned, request=request_only
                    )
                    current_predecessor = abandoned["event_sha256"]
                    continue
            if pending is not None:
                event, recovery, recovery_event = self._recover_pending_iam_intent(
                    cleanup_label=label,
                    source_readback=current_readback,
                    fresh_compute_absence_event_sha256=absence["event_sha256"],
                )
                last_recovery_event = recovery_event
                if recovery["state"] == "exact_desired_after_absent":
                    authority_absence = self._validated_cleanup_compute_event(
                        recovery["fresh_compute_absence_event_sha256"]
                    )
                    receipt = recovery["cleanup_receipt"]
                else:
                    current_readback = recovery["fresh_readback_receipt"]
                    current_predecessor = event["event_sha256"]
                    continue
            else:
                event_path = self.prepared.root / "steps" / label / "event.json"
                if terminal is None and not event_path.is_file():
                    if self.recovery_only:
                        raise PermissionError(
                            "recovery-only mode reached a new IAM cleanup path"
                        )
                    effective_request = self._materialize_request(
                        label=label,
                        mode="worker-iam-cleanup",
                        builder=lambda label=label, common=common,
                        predecessor=absence["event_sha256"]: {
                            "operation_key": self._op(label),
                            "predecessor_event_sha256": predecessor,
                            **common,
                            "observed_at_utc": _render_utc(self._now_dt()),
                        },
                    )
                    if (
                        effective_request["predecessor_event_sha256"]
                        != absence["event_sha256"]
                    ):
                        raw_abandoned = self.controller.abandon_worker_iam_cleanup_request_before_intent(
                                operation_key=operation_key,
                                stale_predecessor_event_sha256=effective_request[
                                    "predecessor_event_sha256"
                                ],
                                fresh_compute_absence_event_sha256=absence[
                                    "event_sha256"
                                ],
                                recorded_at_utc=_render_utc(self._now_dt()),
                            )
                        abandoned = self._persist_event(
                            label=label,
                            mode="worker-iam-cleanup",
                            request=effective_request,
                            event=_event_value(raw_abandoned),
                        )
                        self._validate_stale_cleanup_abandonment(
                            event=abandoned, request=effective_request
                        )
                        current_predecessor = abandoned["event_sha256"]
                        continue
                    effective_absence = self._validated_cleanup_compute_event(
                        effective_request["predecessor_event_sha256"]
                    )
                    effective_absence_receipt = (
                        self._validate_compute_absence_receipt(
                            effective_absence["output"][
                                "compute_absence_receipt"
                            ],
                            self._materialize_request(
                                label=effective_absence["operation_key"].rsplit(
                                    ":", 1
                                )[-1],
                                mode="prelaunch-abort-compute-absence",
                                builder=lambda: (_ for _ in ()).throw(
                                    AssertionError(
                                        "compute absence request must exist"
                                    )
                                ),
                            ),
                        )
                    )
                    if self._now_dt() >= _parse_utc(
                        effective_absence_receipt["expires_at_utc"]
                    ):
                        abandoned = self._abandon_expired_cleanup_request(
                            label=label,
                            operation_key=operation_key,
                            effective_request=effective_request,
                            absence=absence,
                        )
                        current_predecessor = abandoned["event_sha256"]
                        continue
                    _reject_create_authority(
                        self.prepared.execution_root,
                        self.controller.journal.load(),
                        allowed_request_only=self.prepared.abort_plan.get(
                            "persisted_authorize_request"
                        ),
                    )
                    self._revalidate_execution_source()
                    if self._now_dt() >= _parse_utc(
                        effective_absence_receipt["expires_at_utc"]
                    ):
                        abandoned = self._abandon_expired_cleanup_request(
                            label=label,
                            operation_key=operation_key,
                            effective_request=effective_request,
                            absence=absence,
                        )
                        current_predecessor = abandoned["event_sha256"]
                        continue
                event = self._dispatch(
                    label=label,
                    mode="worker-iam-cleanup",
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("IAM cleanup request must exist")
                    ),
                )
                effective_request = self._materialize_request(
                    label=label,
                    mode="worker-iam-cleanup",
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("IAM cleanup request must exist")
                    ),
                )
                current_readback = effective_request["readback_receipt"]
                authority_absence = self._validated_cleanup_compute_event(
                    effective_request["predecessor_event_sha256"]
                )
                if event.get("status") in {"complete", "reconciled-complete"}:
                    receipt = event.get("output", {}).get("receipt")
                    if receipt is None:
                        recovery = event.get("output", {}).get(
                            "reconciled_receipt"
                        )
                        recovery_event, recovery = (
                            self._find_recovery_event_for_receipt(recovery)
                        )
                        last_recovery_event = recovery_event
                        if recovery["state"] != "exact_desired_after_absent":
                            raise ValueError(
                                "reconciled-complete IAM cleanup lacks desired-after proof"
                            )
                        authority_absence = self._validated_cleanup_compute_event(
                            recovery["fresh_compute_absence_event_sha256"]
                        )
                        receipt = recovery["cleanup_receipt"]
                else:
                    output = event.get("output")
                    failure_kind = (
                        output.get("failure_kind")
                        if isinstance(output, Mapping)
                        else None
                    )
                    if (
                        event.get("status") == "failed"
                        and failure_kind
                        == "stale_request_abandoned_before_intent"
                    ):
                        self._validate_stale_cleanup_abandonment(
                            event=event, request=effective_request
                        )
                        current_predecessor = event["event_sha256"]
                        continue
                    if (
                        event.get("status") == "failed"
                        and failure_kind == "process_loss_no_effect_proven"
                    ):
                        recovery_event, recovery = (
                            self._find_recovery_event_for_receipt(
                                output["reconciled_receipt"]
                            )
                        )
                        if recovery["state"] != "exact_before_installed":
                            raise ValueError(
                                "no-effect IAM terminal lacks exact-before proof"
                            )
                        last_recovery_event = recovery_event
                        current_readback = recovery["fresh_readback_receipt"]
                        current_predecessor = event["event_sha256"]
                        continue
                    if (
                        event.get("status") != "failed"
                        or failure_kind not in {
                            "transport_ambiguity", "local_or_response_failure"
                        }
                    ):
                        raise RuntimeError(
                            "worker IAM abort cleanup failed without exact "
                            "outcome-ambiguity reconciliation authority"
                        )
                    reconcile_label = (
                        "prelaunch-abort-worker-iam-reconcile-cleanup"
                        if round_index == 0
                        else "prelaunch-abort-worker-iam-reconcile-cleanup-"
                        f"r{round_index:02d}"
                    )
                    event = self._dispatch(
                        label=reconcile_label,
                        mode="worker-iam-reconcile-cleanup",
                        builder=lambda label=label, reconcile_label=reconcile_label,
                        common={**common, "readback_receipt": current_readback},
                        failed=event["event_sha256"]: {
                            "operation_key": self._op(reconcile_label),
                            "predecessor_event_sha256": failed,
                            "source_operation_key": self._op(label),
                            **common,
                            "observed_at_utc": _render_utc(self._now_dt()),
                        },
                    )
                    receipt = event.get("output", {}).get("receipt")
            checked = worker_iam_v2.validate_cleanup_receipt(
                iam_plan=self.prepared.iam_plan,
                wave_plan=self.prepared.wave_plan,
                attempt_ledger=self.prepared.attempt_ledger,
                resume_plan=self.prepared.resume_plan,
                prepare_receipt=self.prepared.iam_prepare_receipt,
                install_receipt=self.prepared.iam_install_receipt,
                readback_receipt=current_readback,
                value=receipt,
                **self.prepared.content_binding,
            )
            if (
                event.get("status") not in {"complete", "reconciled-complete"}
                or checked.get("cleanup_complete") is not True
                or checked.get("remaining_targeted_binding_count") != 0
                or checked.get("post_cleanup_absence_readback") is not True
            ):
                raise RuntimeError("worker IAM abort cleanup absence is incomplete")
            return event, authority_absence, checked, last_recovery_event
        raise RuntimeError("worker IAM abort cleanup recovery rounds exhausted")

    def _default_preservation_reader(
        self, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self.requester is None:
            requester = controller_v2._stdlib_http_request
        else:
            requester = self.requester
        backend = content_gcp_v2.GcsContentObjectAdapter(
            mode="readback", stage_plan=self.prepared.stage_plan,
            requester=requester,
        )
        fresh_stage = content_v2.build_stage_readback_receipt(
            stage_plan=self.prepared.stage_plan,
            preflight_receipt=self.prepared.preflight_receipt,
            created_rows=self.prepared.stage_receipt["rows"], backend=backend,
            observed_at_utc=request["observed_at_utc"],
        )
        fresh_stage = content_v2.validate_stage_receipt(
            self.prepared.stage_plan, self.prepared.preflight_receipt, fresh_stage
        )
        claim_backend = controller_v2._phase_a_adapter(
            controller=self.controller, mode="claim",
            content=self.prepared.content_binding, requester=requester,
        )
        remote = claim_backend.get_object(
            object_name=self.prepared.claim_receipt["object_path"]
        )
        if (
            remote["object_name"] != self.prepared.claim_receipt["object_path"]
            or
            remote["generation"] != self.prepared.claim_receipt["generation"]
            or remote["etag"] != self.prepared.claim_receipt["etag"]
            or remote["sha256"] != self.prepared.claim_receipt["readback_sha256"]
            or remote["bytes"] != self.prepared.claim_receipt["claim_bytes"]
            or json.loads(remote["payload"].decode("ascii"))
            != self.prepared.claim_receipt["claim_payload"]
            or fresh_stage["rows"] != self.prepared.stage_receipt["rows"]
            or fresh_stage["created_entry_count"] != 26
        ):
            raise RuntimeError("preserved claim or immutable content changed")
        service_accounts = []
        token = os.environ.get(gcp_v2.TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            char.isspace() for char in token
        ):
            raise PermissionError(
                f"Bearer token must be supplied only through {gcp_v2.TOKEN_ENV}"
            )
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        for worker in self.prepared.iam_plan["workers"]:
            email = worker["service_account"]
            encoded = urllib.parse.quote(email, safe="")
            url = (
                "https://iam.googleapis.com/v1/projects/"
                f"{gcp_v2.PROJECT}/serviceAccounts/{encoded}"
            )
            try:
                response = requester("GET", url, headers, None, 60)
            except Exception as exc:
                raise gcp_v2.GcpPhaseATransportError(
                    "prelaunch abort service-account GET transport failed"
                ) from exc
            if not isinstance(response, gcp_v2.HttpResponse) or response.status != 200:
                raise RuntimeError("preserved service-account GET did not return 200")
            try:
                account = json.loads(response.body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("service-account response is not JSON") from exc
            if (
                not isinstance(account, Mapping)
                or account.get("email") != email
                or not isinstance(account.get("uniqueId"), str)
                or not account["uniqueId"].isdigit()
                or account.get("disabled", False) is not False
                or account.get("name")
                not in {
                    f"projects/{gcp_v2.PROJECT}/serviceAccounts/{email}",
                    f"projects/{gcp_v2.PROJECT}/serviceAccounts/{account['uniqueId']}",
                }
            ):
                raise RuntimeError("preserved service-account identity changed")
            service_accounts.append(
                {
                    "job_id": worker["job_id"],
                    "source_role": worker["source_role"],
                    "service_account": email,
                    "unique_id": account["uniqueId"],
                    "resource_name": account["name"],
                    "disabled": False,
                    "exists": True,
                }
            )
        core = {
            "schema": PRESERVATION_RECEIPT_SCHEMA,
            "status": "claim_and_exact_26_content_objects_preserved",
            "claim_object_path": remote["object_name"],
            "claim_generation": remote["generation"],
            "claim_etag": remote["etag"],
            "claim_sha256": remote["sha256"],
            "claim_bytes": remote["bytes"],
            "content_prefix": self.prepared.content_binding[
                "immutable_content_prefix"
            ],
            "content_entry_count": 26,
            "source_stage_receipt_sha256": self.prepared.stage_receipt[
                "receipt_sha256"
            ],
            "fresh_stage_receipt_sha256": fresh_stage["receipt_sha256"],
            "service_accounts": service_accounts,
            "service_account_count": len(service_accounts),
            "service_accounts_preserved": True,
            "claim_preserved": True, "content_preserved": True,
            "claim_delete_performed": False, "content_delete_performed": False,
            "cloud_mutation_performed": False,
            "observed_at_utc": request["observed_at_utc"],
            "current_profile_changed": False,
        }
        return {**core, "receipt_sha256": canonical_sha256(core)}

    def _validate_preservation_receipt(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = _sealed(value)
        if set(receipt) != {
            "schema", "status", "claim_object_path", "claim_generation",
            "claim_etag", "claim_sha256", "claim_bytes", "content_prefix",
            "content_entry_count", "source_stage_receipt_sha256",
            "fresh_stage_receipt_sha256", "service_accounts",
            "service_account_count", "service_accounts_preserved",
            "claim_preserved", "content_preserved", "claim_delete_performed",
            "content_delete_performed", "cloud_mutation_performed",
            "observed_at_utc", "current_profile_changed", "receipt_sha256",
        }:
            raise ValueError("prelaunch abort preservation receipt fields changed")
        if (
            receipt.get("schema") != PRESERVATION_RECEIPT_SCHEMA
            or receipt.get("status")
            != "claim_and_exact_26_content_objects_preserved"
            or receipt.get("claim_preserved") is not True
            or receipt.get("content_preserved") is not True
            or receipt.get("service_accounts_preserved") is not True
            or receipt.get("content_entry_count") != 26
            or receipt.get("claim_delete_performed") is not False
            or receipt.get("content_delete_performed") is not False
            or receipt.get("cloud_mutation_performed") is not False
            or receipt.get("current_profile_changed") is not False
            or receipt.get("claim_object_path")
            != self.prepared.claim_receipt["object_path"]
            or receipt.get("claim_generation")
            != self.prepared.claim_receipt["generation"]
            or receipt.get("claim_etag") != self.prepared.claim_receipt["etag"]
            or receipt.get("claim_sha256")
            != self.prepared.claim_receipt["readback_sha256"]
            or receipt.get("claim_bytes")
            != self.prepared.claim_receipt["claim_bytes"]
            or receipt.get("content_prefix")
            != self.prepared.content_binding["immutable_content_prefix"]
            or receipt.get("source_stage_receipt_sha256")
            != self.prepared.stage_receipt["receipt_sha256"]
        ):
            raise ValueError("prelaunch abort preservation receipt changed")
        expected_accounts = [
            (row["job_id"], row["source_role"], row["service_account"])
            for row in self.prepared.iam_plan["workers"]
        ]
        observed_accounts = receipt.get("service_accounts")
        if (
            not isinstance(observed_accounts, list)
            or receipt.get("service_account_count") != len(expected_accounts)
            or [
                (row.get("job_id"), row.get("source_role"), row.get("service_account"))
                for row in observed_accounts
                if isinstance(row, Mapping)
            ]
            != expected_accounts
            or any(
                not isinstance(row, Mapping)
                or row.get("exists") is not True
                or row.get("disabled") is not False
                or not isinstance(row.get("unique_id"), str)
                or not row["unique_id"].isdigit()
                for row in observed_accounts
            )
        ):
            raise ValueError("prelaunch abort preserved service accounts changed")
        _sha(receipt.get("fresh_stage_receipt_sha256"), "fresh stage receipt")
        return receipt

    def _validated_postflight_compute_event(
        self, event_sha256: str
    ) -> dict[str, Any]:
        event = self._journal_event(event_sha256)
        prefix = f"{self.prepared.abort_plan['execution_namespace']}:"
        label = event["operation_key"].removeprefix(prefix)
        if not label.startswith("prelaunch-abort-compute-postflight-r"):
            raise ValueError("preservation postflight authority label changed")
        request, event = self._validate_persisted_step(
            label=label,
            mode="prelaunch-abort-compute-absence",
            event_sha256=event["event_sha256"],
        )
        self._validate_compute_absence_receipt(
            event.get("output", {}).get("compute_absence_receipt"), request
        )
        if event.get("status") != "complete":
            raise ValueError("preservation postflight authority changed")
        return event

    def _validate_preservation_event(
        self, *, event: Mapping[str, Any], request: Mapping[str, Any]
    ) -> dict[str, Any]:
        checked = _event_value(event)
        receipt = self._validate_preservation_receipt(
            checked.get("output", {}).get("preservation_receipt")
        )
        if (
            checked.get("phase")
            != "prelaunch-abort-preservation-readback"
            or checked.get("mode") != "read-only"
            or checked.get("status") != "complete"
            or checked.get("operation_key") != request["operation_key"]
            or checked.get("predecessor_event_sha256")
            != request["predecessor_event_sha256"]
            or checked.get("mutation_requested") is not False
            or checked.get("mutation_outcome") != "not_requested"
            or checked.get("output", {}).get("preservation_receipt") != receipt
        ):
            raise ValueError("prelaunch abort preservation event changed")
        return checked

    def _preservation_readback(
        self, postflight: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mode = "prelaunch-abort-preservation-readback"
        current_postflight = self._validated_postflight_compute_event(
            postflight["event_sha256"]
        )
        for round_index in range(MAX_RECOVERY_ROUNDS):
            label = f"prelaunch-abort-preservation-readback-r{round_index:02d}"
            request_path = self._request_path(label)
            terminal = self._terminal_for(self._op(label))
            event_path = self.prepared.root / "steps" / label / "event.json"
            if event_path.exists() or terminal is not None:
                if not request_path.is_file():
                    raise ValueError("preservation event lacks its request")
                request = self._materialize_request(
                    label=label,
                    mode=mode,
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("preservation request must exist")
                    ),
                )
                event = self._persist_event(
                    label=label,
                    mode=mode,
                    request=request,
                    event=(
                        _read_json(event_path, f"{label} event")
                        if event_path.exists()
                        else terminal
                    ),
                )
                checked = self._validate_preservation_event(
                    event=event, request=request
                )
                authority = self._validated_postflight_compute_event(
                    request["predecessor_event_sha256"]
                )
                return checked, authority
            if request_path.exists():
                old_request = self._materialize_request(
                    label=label,
                    mode=mode,
                    builder=lambda: (_ for _ in ()).throw(
                        AssertionError("preservation request must exist")
                    ),
                )
                if (
                    old_request["predecessor_event_sha256"]
                    != current_postflight["event_sha256"]
                ):
                    continue
                request = old_request
            else:
                request = self._materialize_request(
                    label=label,
                    mode=mode,
                    builder=lambda label=label: {
                        "operation_key": self._op(label),
                        "predecessor_event_sha256": current_postflight[
                            "event_sha256"
                        ],
                        "claim_receipt_sha256": self.prepared.claim_receipt[
                            "receipt_sha256"
                        ],
                        "stage_receipt_sha256": self.prepared.stage_receipt[
                            "receipt_sha256"
                        ],
                        "observed_at_utc": _render_utc(self._now_dt()),
                    },
                )
            post_receipt = current_postflight["output"][
                "compute_absence_receipt"
            ]
            if self._now_dt() >= _parse_utc(post_receipt["expires_at_utc"]):
                raise RuntimeError(
                    "postflight compute proof expired before preservation read"
                )
            reader = self.preservation_reader or self._default_preservation_reader
            receipt = self._validate_preservation_receipt(reader(request))
            raw = self.controller.record_read_phase(
                phase=mode,
                operation_key=request["operation_key"],
                predecessor_event_sha256=request["predecessor_event_sha256"],
                evidence={
                    "claim_receipt_sha256": request["claim_receipt_sha256"],
                    "stage_receipt_sha256": request["stage_receipt_sha256"],
                },
                output={"preservation_receipt": receipt},
                recorded_at_utc=request["observed_at_utc"],
            )
            event = self._persist_event(
                label=label, mode=mode, request=request, event=_event_value(raw)
            )
            return (
                self._validate_preservation_event(event=event, request=request),
                current_postflight,
            )
        raise RuntimeError("fresh preservation readback rounds exhausted")

    def _validate_persisted_step(
        self, *, label: str, mode: str, event_sha256: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_path = self._request_path(label)
        event_path = self.prepared.root / "steps" / label / "event.json"
        if not request_path.is_file() or not event_path.is_file():
            raise ValueError("prelaunch abort step artifact is missing")
        request = self._materialize_request(
            label=label, mode=mode,
            builder=lambda: (_ for _ in ()).throw(
                AssertionError("persisted request builder must not run")
            ),
        )
        event = self._persist_event(
            label=label, mode=mode, request=request,
            event=_read_json(event_path, f"{label} event"),
        )
        if (
            event["event_sha256"] != event_sha256
            or event["operation_key"] != self._op(label)
        ):
            raise ValueError("prelaunch abort manifest step binding changed")
        return request, event

    def _validate_manifest(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = _clone(value, "prelaunch abort manifest")
        if set(payload) != _ABORT_MANIFEST_KEYS:
            raise ValueError("prelaunch abort manifest fields changed")
        digest = payload.pop("manifest_sha256", None)
        if _sha(digest, "abort manifest") != canonical_sha256(payload):
            raise ValueError("prelaunch abort manifest digest changed")
        payload["manifest_sha256"] = digest
        if (
            payload.get("schema") != ABORT_MANIFEST_SCHEMA
            or payload.get("status") != ABORT_MANIFEST_STATUS
            or payload.get("abort_plan_sha256")
            != self.prepared.abort_plan["plan_sha256"]
            or payload.get("all_selected_instances_absent_pre") is not True
            or payload.get("all_selected_boot_disks_absent_pre") is not True
            or payload.get("all_selected_instances_absent_post") is not True
            or payload.get("all_selected_boot_disks_absent_post") is not True
            or payload.get("worker_iam_bindings_absent") is not True
            or payload.get("claim_preserved") is not True
            or payload.get("service_accounts_preserved") is not True
            or payload.get("immutable_content_preserved") is not True
            or payload.get("claim_cleanup_event_sha256") is not None
            or payload.get("service_account_cleanup_event_sha256") is not None
            or payload.get("content_cleanup_event_sha256") is not None
            or payload.get("old_execution_relaunch_authorized") is not False
            or payload.get("additional_create_authorized") is not False
            or payload.get("credentials_from_environment_only") is not True
            or payload.get("current_profile_changed") is not False
            or payload.get("execution_namespace")
            != self.prepared.abort_plan["execution_namespace"]
            or payload.get("run_name") != self.prepared.wave_plan["run_name"]
            or payload.get("execution_identity_sha256")
            != self.prepared.wave_plan["execution_identity_sha256"]
            or payload.get("wave_index")
            != self.prepared.resume_plan["resume_wave_index"]
            or payload.get("anchor_event_sha256")
            != self.prepared.abort_plan["anchor_event_sha256"]
            or payload.get("worker_iam_removed_binding_count")
            not in {0, self.prepared.iam_plan["exact_binding_count"]}
            or payload.get("claim_object_path")
            != self.prepared.claim_receipt["object_path"]
            or payload.get("claim_generation")
            != self.prepared.claim_receipt["generation"]
            or payload.get("immutable_content_prefix")
            != self.prepared.content_binding["immutable_content_prefix"]
            or payload.get("immutable_content_entry_count") != 26
        ):
            raise ValueError("prelaunch abort manifest binding changed")
        _reject_create_authority(
            self.prepared.execution_root,
            self.controller.journal.load(),
            allowed_request_only=self.prepared.abort_plan.get(
                "persisted_authorize_request"
            ),
        )
        _, tombstone = self._validate_persisted_step(
            label="prelaunch-abort-tombstone",
            mode="prelaunch-abort-tombstone",
            event_sha256=payload["tombstone_event_sha256"],
        )
        tombstone_output = tombstone.get("output", {})
        if (
            tombstone.get("phase") != "prelaunch-abort-tombstone"
            or tombstone.get("status") != "complete"
            or tombstone_output.get("abort_plan_sha256")
            != self.prepared.abort_plan["plan_sha256"]
            or tombstone_output.get("old_execution_relaunch_authorized") is not False
        ):
            raise ValueError("prelaunch abort manifest tombstone changed")
        _, iam_readback = self._validate_persisted_step(
            label="prelaunch-abort-worker-iam-readback",
            mode="worker-iam-readback",
            event_sha256=payload["worker_iam_readback_event_sha256"],
        )
        preflight = self._journal_event(payload["preflight_event_sha256"])
        preflight_label = preflight["operation_key"].rsplit(":", 1)[-1]
        if not preflight_label.startswith(
            "prelaunch-abort-compute-before-iam-cleanup-"
        ):
            raise ValueError("prelaunch abort pre-cleanup absence label changed")
        pre_request, preflight = self._validate_persisted_step(
            label=preflight_label,
            mode="prelaunch-abort-compute-absence",
            event_sha256=preflight["event_sha256"],
        )
        cleanup_event = self._journal_event(
            payload["worker_iam_cleanup_event_sha256"]
        )
        cleanup_label = cleanup_event["operation_key"].rsplit(":", 1)[-1]
        primary_base = "prelaunch-abort-worker-iam-cleanup"
        reconcile_base = "prelaunch-abort-worker-iam-reconcile-cleanup"
        recovery_event: dict[str, Any] | None = None
        recovery_receipt: dict[str, Any] | None = None
        recovery_event_sha = payload["worker_iam_intent_recovery_event_sha256"]
        if recovery_event_sha is not None:
            recovery_journal_event = self._journal_event(
                _sha(recovery_event_sha, "IAM intent recovery event")
            )
            raw_recovery = recovery_journal_event.get("output", {}).get(
                "iam_intent_recovery_receipt"
            )
            recovery_event, recovery_receipt = (
                self._find_recovery_event_for_receipt(raw_recovery)
            )
            if recovery_event["event_sha256"] != recovery_event_sha:
                raise ValueError("manifest IAM recovery event binding changed")
            source_terminal = self._terminal_for(
                recovery_receipt["source_operation_key"]
            )
            if source_terminal is None:
                raise ValueError("IAM recovery source terminal is absent")
            source_label = recovery_receipt["source_operation_key"].rsplit(
                ":", 1
            )[-1]
            _, source_terminal = self._validate_persisted_step(
                label=source_label,
                mode="worker-iam-cleanup",
                event_sha256=source_terminal["event_sha256"],
            )
            if (
                source_terminal.get("output", {}).get("reconciled_receipt")
                != recovery_receipt
                or (
                    recovery_receipt["state"] == "exact_desired_after_absent"
                    and source_terminal.get("status") != "reconciled-complete"
                )
                or (
                    recovery_receipt["state"] == "exact_before_installed"
                    and (
                        source_terminal.get("status") != "failed"
                        or source_terminal.get("output", {}).get("failure_kind")
                        != "process_loss_no_effect_proven"
                    )
                )
            ):
                raise ValueError("IAM recovery source terminal binding changed")
        if cleanup_label == primary_base or cleanup_label.startswith(
            f"{primary_base}-r"
        ):
            primary_label = cleanup_label
            primary_request, iam_cleanup = self._validate_persisted_step(
                label=primary_label,
                mode="worker-iam-cleanup",
                event_sha256=cleanup_event["event_sha256"],
            )
            if iam_cleanup["status"] == "reconciled-complete":
                if (
                    recovery_event is None
                    or recovery_receipt is None
                    or recovery_receipt["state"]
                    != "exact_desired_after_absent"
                    or recovery_receipt["source_operation_key"]
                    != self._op(primary_label)
                    or recovery_receipt["fresh_compute_absence_event_sha256"]
                    != preflight["event_sha256"]
                    or iam_cleanup.get("output", {}).get("reconciled_receipt")
                    != recovery_receipt
                ):
                    raise ValueError("pending IAM desired-after recovery changed")
                cleanup_receipt_value = recovery_receipt["cleanup_receipt"]
            elif iam_cleanup["status"] == "complete":
                cleanup_receipt_value = iam_cleanup.get("output", {}).get("receipt")
                if primary_request["predecessor_event_sha256"] != preflight[
                    "event_sha256"
                ]:
                    raise ValueError(
                        "IAM cleanup lacks the fresh compute absence proof"
                    )
                if recovery_receipt is not None and (
                    recovery_receipt["state"] != "exact_before_installed"
                    or primary_request["readback_receipt"]
                    != recovery_receipt["fresh_readback_receipt"]
                ):
                    raise ValueError("prior exact-before IAM recovery changed")
            else:
                raise ValueError("prelaunch abort IAM cleanup status changed")
        elif cleanup_label == reconcile_base or cleanup_label.startswith(
            f"{reconcile_base}-r"
        ):
            suffix = cleanup_label.removeprefix(reconcile_base)
            primary_label = primary_base + suffix
            primary_terminal = self._terminal_for(self._op(primary_label))
            if primary_terminal is None:
                raise ValueError("prelaunch abort IAM reconcile source is absent")
            primary_request, primary_failure = self._validate_persisted_step(
                label=primary_label,
                mode="worker-iam-cleanup",
                event_sha256=primary_terminal["event_sha256"],
            )
            if (
                primary_failure["status"] != "failed"
                or primary_failure.get("output", {}).get("failure_kind")
                not in {"transport_ambiguity", "local_or_response_failure"}
            ):
                raise ValueError("prelaunch abort IAM reconciliation source changed")
            _, iam_cleanup = self._validate_persisted_step(
                label=cleanup_label,
                mode="worker-iam-reconcile-cleanup",
                event_sha256=cleanup_event["event_sha256"],
            )
            if (
                primary_request["predecessor_event_sha256"]
                != preflight["event_sha256"]
                or iam_cleanup["predecessor_event_sha256"]
                != primary_failure["event_sha256"]
            ):
                raise ValueError("prelaunch abort IAM reconciliation chain changed")
            if recovery_receipt is not None and (
                recovery_receipt["state"] != "exact_before_installed"
                or primary_request["readback_receipt"]
                != recovery_receipt["fresh_readback_receipt"]
            ):
                raise ValueError("prior exact-before IAM recovery changed")
            cleanup_receipt_value = iam_cleanup["output"]["receipt"]
        else:
            raise ValueError("prelaunch abort IAM cleanup operation changed")
        if (
            recovery_receipt is None
            and primary_request["readback_receipt"]
            != iam_readback["output"]["worker_iam_readback_receipt"]
        ):
            raise ValueError("manifest omitted an IAM recovery authority")
        postflight = self._journal_event(payload["postflight_event_sha256"])
        postflight_label = postflight["operation_key"].rsplit(":", 1)[-1]
        if not postflight_label.startswith("prelaunch-abort-compute-postflight-r"):
            raise ValueError("prelaunch abort post-cleanup absence label changed")
        post_request, postflight = self._validate_persisted_step(
            label=postflight_label,
            mode="prelaunch-abort-compute-absence",
            event_sha256=payload["postflight_event_sha256"],
        )
        preservation = self._journal_event(payload["preservation_event_sha256"])
        preservation_label = preservation["operation_key"].rsplit(":", 1)[-1]
        if not preservation_label.startswith(
            "prelaunch-abort-preservation-readback-r"
        ):
            raise ValueError("prelaunch abort preservation label changed")
        preservation_request, preservation = self._validate_persisted_step(
            label=preservation_label,
            mode="prelaunch-abort-preservation-readback",
            event_sha256=payload["preservation_event_sha256"],
        )
        self._validate_preservation_event(
            event=preservation, request=preservation_request
        )
        if (
            preflight["status"] != "complete"
            or iam_readback["status"] != "complete"
            or iam_readback["predecessor_event_sha256"]
            != tombstone["event_sha256"]
            or iam_cleanup["status"] not in {"complete", "reconciled-complete"}
            or postflight["status"] != "complete"
            or postflight["predecessor_event_sha256"]
            != iam_cleanup["event_sha256"]
            or preservation["status"] != "complete"
            or preservation["predecessor_event_sha256"]
            != postflight["event_sha256"]
        ):
            raise ValueError("prelaunch abort event chain changed")
        pre_receipt = self._validate_compute_absence_receipt(
            preflight["output"]["compute_absence_receipt"], pre_request
        )
        worker_iam_v2.validate_readback_receipt(
            iam_plan=self.prepared.iam_plan,
            wave_plan=self.prepared.wave_plan,
            attempt_ledger=self.prepared.attempt_ledger,
            resume_plan=self.prepared.resume_plan,
            prepare_receipt=self.prepared.iam_prepare_receipt,
            install_receipt=self.prepared.iam_install_receipt,
            value=iam_readback["output"]["worker_iam_readback_receipt"],
            **self.prepared.content_binding,
        )
        effective_readback = primary_request["readback_receipt"]
        cleanup_receipt = worker_iam_v2.validate_cleanup_receipt(
            iam_plan=self.prepared.iam_plan,
            wave_plan=self.prepared.wave_plan,
            attempt_ledger=self.prepared.attempt_ledger,
            resume_plan=self.prepared.resume_plan,
            prepare_receipt=self.prepared.iam_prepare_receipt,
            install_receipt=self.prepared.iam_install_receipt,
            readback_receipt=effective_readback,
            value=cleanup_receipt_value,
            **self.prepared.content_binding,
        )
        post_receipt = self._validate_compute_absence_receipt(
            postflight["output"]["compute_absence_receipt"], post_request
        )
        preservation_receipt = self._validate_preservation_receipt(
            preservation["output"]["preservation_receipt"]
        )
        if (
            payload["preflight_receipt_sha256"]
            != pre_receipt["receipt_sha256"]
            or payload["worker_iam_cleanup_receipt_sha256"]
            != cleanup_receipt["receipt_sha256"]
            or payload["postflight_receipt_sha256"]
            != post_receipt["receipt_sha256"]
            or payload["preservation_receipt_sha256"]
            != preservation_receipt["receipt_sha256"]
            or payload["unrelated_policy_fingerprint_sha256"]
            != cleanup_receipt["unrelated_policy_fingerprint_sha256"]
            or payload["worker_iam_removed_binding_count"]
            != cleanup_receipt["removed_binding_count"]
            or payload["service_account_count"]
            != preservation_receipt["service_account_count"]
        ):
            raise ValueError("prelaunch abort manifest receipt binding changed")
        _assert_no_credentials(payload, "prelaunch abort manifest")
        return payload

    def execute(self) -> dict[str, Any]:
        self._revalidate_execution_source()
        final_path = self.prepared.root / "abort_manifest.json"
        if final_path.exists():
            return self._validate_manifest(
                _read_json(final_path, "prelaunch abort manifest")
            )
        now = self._now_dt()
        not_before = _parse_utc(
            self.prepared.abort_plan["worker_iam_cleanup_not_before_utc"]
        )
        if now <= not_before:
            raise PermissionError("worker IAM minimum window has not expired")
        _reject_create_authority(
            self.prepared.execution_root,
            self.controller.journal.load(),
            allowed_request_only=self.prepared.abort_plan.get(
                "persisted_authorize_request"
            ),
        )
        self._validate_recovery_only_source()
        tombstone = self._record_tombstone()
        _reject_create_authority(
            self.prepared.execution_root,
            self.controller.journal.load(),
            allowed_request_only=self.prepared.abort_plan.get(
                "persisted_authorize_request"
            ),
        )
        iam_readback = self._fresh_iam_readback(tombstone["event_sha256"])
        fresh_readback = iam_readback["output"]["worker_iam_readback_receipt"]
        iam_cleanup, preflight, cleanup_receipt, recovery_event = self._cleanup_iam(
            predecessor=iam_readback["event_sha256"],
            fresh_readback=fresh_readback,
        )
        postflight = self._fresh_compute_absence(
            base_label="prelaunch-abort-compute-postflight",
            predecessor=iam_cleanup["event_sha256"],
        )
        preservation, postflight = self._preservation_readback(postflight)
        preservation_receipt = _sealed(
            preservation["output"]["preservation_receipt"]
        )
        pre = preflight["output"]["compute_absence_receipt"]
        post = postflight["output"]["compute_absence_receipt"]
        core = {
            "schema": ABORT_MANIFEST_SCHEMA,
            "status": ABORT_MANIFEST_STATUS,
            "execution_namespace": self.prepared.abort_plan["execution_namespace"],
            "run_name": self.prepared.wave_plan["run_name"],
            "execution_identity_sha256": self.prepared.wave_plan[
                "execution_identity_sha256"
            ],
            "wave_index": self.prepared.resume_plan["resume_wave_index"],
            "abort_plan_sha256": self.prepared.abort_plan["plan_sha256"],
            "anchor_event_sha256": self.prepared.abort_plan["anchor_event_sha256"],
            "tombstone_event_sha256": tombstone["event_sha256"],
            "preflight_event_sha256": preflight["event_sha256"],
            "worker_iam_readback_event_sha256": iam_readback["event_sha256"],
            "worker_iam_intent_recovery_event_sha256": (
                None if recovery_event is None else recovery_event["event_sha256"]
            ),
            "worker_iam_cleanup_event_sha256": iam_cleanup["event_sha256"],
            "postflight_event_sha256": postflight["event_sha256"],
            "preservation_event_sha256": preservation["event_sha256"],
            "preflight_receipt_sha256": pre["receipt_sha256"],
            "worker_iam_cleanup_receipt_sha256": cleanup_receipt["receipt_sha256"],
            "postflight_receipt_sha256": post["receipt_sha256"],
            "preservation_receipt_sha256": preservation_receipt[
                "receipt_sha256"
            ],
            "all_selected_instances_absent_pre": True,
            "all_selected_boot_disks_absent_pre": True,
            "all_selected_instances_absent_post": True,
            "all_selected_boot_disks_absent_post": True,
            "worker_iam_bindings_absent": True,
            "worker_iam_removed_binding_count": cleanup_receipt[
                "removed_binding_count"
            ],
            "unrelated_policy_fingerprint_sha256": cleanup_receipt[
                "unrelated_policy_fingerprint_sha256"
            ],
            "claim_preserved": True,
            "claim_object_path": self.prepared.claim_receipt["object_path"],
            "claim_generation": self.prepared.claim_receipt["generation"],
            "service_accounts_preserved": True,
            "service_account_count": preservation_receipt[
                "service_account_count"
            ],
            "immutable_content_preserved": True,
            "immutable_content_prefix": self.prepared.content_binding[
                "immutable_content_prefix"
            ],
            "immutable_content_entry_count": 26,
            "claim_cleanup_event_sha256": None,
            "service_account_cleanup_event_sha256": None,
            "content_cleanup_event_sha256": None,
            "old_execution_relaunch_authorized": False,
            "additional_create_authorized": False,
            "credentials_from_environment_only": True,
            "current_profile_changed": False,
        }
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        checked_manifest = self._validate_manifest(manifest)
        _write_once_json(final_path, checked_manifest)
        return checked_manifest


def execute_prelaunch_abort(
    *, prepared: PreparedPrelaunchAbort, allow_cloud_read: bool,
    allow_worker_iam_cleanup: bool, confirm_run_name: str,
    recovery_only: bool = False,
    dispatcher: Dispatcher = controller_v2.execute_mode_request,
    requester: Callable[..., Any] | None = None,
    compute_absence_reader: ComputeAbsenceReader | None = None,
    iam_intent_recovery_reader: IamIntentRecoveryReader | None = None,
    preservation_reader: PreservationReader | None = None,
    clock: Clock = _system_clock,
) -> dict[str, Any]:
    return ProductionPrelaunchAbortCleanupV2(
        prepared=prepared, allow_cloud_read=allow_cloud_read,
        allow_worker_iam_cleanup=allow_worker_iam_cleanup,
        confirm_run_name=confirm_run_name, recovery_only=recovery_only,
        dispatcher=dispatcher,
        requester=requester, compute_absence_reader=compute_absence_reader,
        iam_intent_recovery_reader=iam_intent_recovery_reader,
        preservation_reader=preservation_reader,
        clock=clock,
    ).execute()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "execute"), default="plan")
    parser.add_argument("--execution-root", required=True)
    parser.add_argument("--abort-root", required=True)
    parser.add_argument("--allow-cloud-read", action="store_true")
    parser.add_argument("--allow-worker-iam-cleanup", action="store_true")
    parser.add_argument(
        "--allow-persisted-authorize-request-only", action="store_true"
    )
    parser.add_argument("--recovery-only", action="store_true")
    parser.add_argument("--confirm-run-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare_prelaunch_abort(
        execution_root=args.execution_root,
        abort_root=args.abort_root,
        allow_persisted_authorize_request_only=(
            args.allow_persisted_authorize_request_only
        ),
    )
    if args.mode == "plan":
        print(json.dumps(prepared.abort_plan, sort_keys=True, indent=2))
        return 0
    result = execute_prelaunch_abort(
        prepared=prepared, allow_cloud_read=args.allow_cloud_read,
        allow_worker_iam_cleanup=args.allow_worker_iam_cleanup,
        confirm_run_name=args.confirm_run_name or "",
        recovery_only=args.recovery_only,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


__all__ = [
    "ABORT_MANIFEST_SCHEMA", "ABORT_MANIFEST_STATUS", "ABORT_PLAN_SCHEMA",
    "PreparedPrelaunchAbort", "ProductionPrelaunchAbortCleanupV2",
    "execute_prelaunch_abort", "prepare_prelaunch_abort", "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
