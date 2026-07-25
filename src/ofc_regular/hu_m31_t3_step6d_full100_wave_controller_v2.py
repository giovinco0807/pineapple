"""Restart-safe top-level controller for one full100 wave-v2 lifecycle.

The controller is deliberately narrower than the cloud adapters it composes:

* local inspection and ``prepare`` are the default;
* every cloud mutation needs an explicit caller opt-in;
* a write-once intent is durable before a mutating adapter is called;
* an unresolved intent can only be reconciled or cleaned up, never replayed;
* all producer-owned receipts are revalidated at their module boundary.

The journal is an authorization/audit boundary, not a source of new cloud
authority.  It never changes an AI profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_full100_wave_cloud_v2 as cloud_v2
from . import hu_m31_t3_step6d_full100_wave_content_gcp_adapter_v2 as content_gcp_v2
from . import hu_m31_t3_step6d_full100_wave_content_stage_v2 as content_v2
from . import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as gce_v2
from . import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_v2
from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_runtime_preflight_v2 as runtime_v2
from . import hu_m31_t3_step6d_full100_wave_runtime_gcp_adapter_v2 as runtime_gcp_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import (
    hu_m31_t3_step6d_full100_wave_worker_identity_gcp_adapter_v2
    as identity_gcp_v2,
)
from . import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as identity_v2
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    _stdlib_http_request,
)


EVENT_SCHEMA = "hu_m31_t3_step6d_full100_wave_controller_event_v2"
PREPARE_SCHEMA = "hu_m31_t3_step6d_full100_wave_controller_prepare_receipt_v2"
LIFECYCLE_SCHEMA = "hu_m31_t3_step6d_full100_wave_controller_lifecycle_receipt_v2"
LIFECYCLE_PROOF_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_controller_lifecycle_validated_proof_v2"
)
CONTROLLER_VERSION = 2

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE = re.compile(r"^[a-z0-9](?:[a-z0-9._:-]{0,126}[a-z0-9])?$")
_EVENT_FILE = re.compile(r"^(?P<sequence>[0-9]{6})-(?P<phase>[a-z0-9-]+)\.json$")
_UTC = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)

_EVENT_KEYS = frozenset(
    {
        "schema", "controller_version", "controller_context_sha256", "sequence",
        "phase", "mode", "status", "operation_key", "previous_event_sha256",
        "predecessor_event_sha256", "evidence_sha256", "evidence", "output_sha256",
        "output", "mutation_requested", "mutation_outcome", "retry_authorized",
        "recorded_at_utc", "current_profile_changed", "event_sha256",
    }
)
_MUTATION_OUTCOMES = frozenset(
    {"not_requested", "pending", "none", "performed", "partial", "unknown"}
)
_APPEND_HEAD_UNSET = object()
_TERMINAL_STATUSES = frozenset(
    {"complete", "partial", "failed", "reconciled-complete", "reconciled-partial"}
)


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


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None or value == "0" * 64:
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be RFC3339 UTC seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    return value


def _safe(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe controller identifier")
    return value


def _json_clone(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _object_sha(value: Mapping[str, Any], *preferred: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("controller evidence must be an object")
    for field in preferred:
        candidate = value.get(field)
        if isinstance(candidate, str) and _SHA.fullmatch(candidate):
            return _sha(candidate, f"evidence {field}")
    return canonical_sha256(value)


def _seal(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = _json_clone(dict(core), "receipt")
    return {**payload, field: canonical_sha256(payload)}


class PendingMutationError(RuntimeError):
    """A durable intent has no terminal receipt and must not be replayed."""


class JournalTamperError(RuntimeError):
    """The immutable controller journal is missing, forked, or modified."""


@dataclass(frozen=True)
class JournalEvent:
    path: Path
    value: dict[str, Any]


class CreateMissingAdapter(Protocol):
    def create_missing(
        self, *, current_time_utc: str, completed_at_utc: str
    ) -> Mapping[str, Any]: ...

    def validate_create_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GceCreateAdapter(Protocol):
    def create_selected(
        self,
        *,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
        prior_create_receipt: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def validate_create_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GceStatusAdapter(Protocol):
    def read_status(self, *, observed_at_utc: str) -> Mapping[str, Any]: ...

    def validate_status_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GceDeleteAdapter(Protocol):
    def delete_owned(
        self, *, request_ids: Mapping[str, Any],
        orphan_disk_request_ids: Mapping[str, Any] | None,
        observed_at_utc: str
    ) -> Mapping[str, Any]: ...

    def validate_delete_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GceDeleteReconcileAdapter(Protocol):
    def reconcile_delete(
        self, *, request_ids: Mapping[str, Any], observed_at_utc: str
    ) -> Mapping[str, Any]: ...

    def validate_delete_reconcile_receipt(
        self, value: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class GceAbsenceAdapter(Protocol):
    def verify_absence(self, *, observed_at_utc: str) -> Mapping[str, Any]: ...

    def validate_absence_receipt(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ControllerJournal:
    """Strict append-only JSON journal with a SHA-256 hash chain."""

    def __init__(
        self, directory: str | Path, *, context_sha256: str, create: bool = True
    ) -> None:
        self.directory = Path(directory)
        self.context_sha256 = _sha(context_sha256, "controller context")
        if self.directory.exists():
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise JournalTamperError("controller journal is not a plain directory")
        elif create:
            self.directory.mkdir(parents=True, exist_ok=False)
        else:
            raise FileNotFoundError(self.directory)

    @staticmethod
    def _validate_event(
        value: Mapping[str, Any], *, context_sha256: str, expected_sequence: int,
        expected_previous: str | None,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise JournalTamperError("controller event is not an object")
        payload = deepcopy(dict(value))
        if set(payload) != _EVENT_KEYS:
            raise JournalTamperError("controller event fields changed")
        supplied = payload.pop("event_sha256", None)
        if supplied != canonical_sha256(payload):
            raise JournalTamperError("controller event digest changed")
        output = payload.get("output")
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping) or payload.get("evidence_sha256") != canonical_sha256(evidence):
            raise JournalTamperError("controller event evidence digest changed")
        if payload.get("output_sha256") != canonical_sha256(output):
            raise JournalTamperError("controller event output digest changed")
        if (
            payload.get("schema") != EVENT_SCHEMA
            or payload.get("controller_version") != CONTROLLER_VERSION
            or payload.get("controller_context_sha256") != context_sha256
            or payload.get("sequence") != expected_sequence
            or payload.get("previous_event_sha256") != expected_previous
            or payload.get("mutation_outcome") not in _MUTATION_OUTCOMES
            or payload.get("current_profile_changed") is not False
            or payload.get("retry_authorized") is not False
        ):
            raise JournalTamperError("controller event chain or safety flags changed")
        _safe(payload.get("phase"), "event phase")
        _safe(payload.get("mode"), "event mode")
        _safe(payload.get("status"), "event status")
        _safe(payload.get("operation_key"), "event operation key")
        predecessor = payload.get("predecessor_event_sha256")
        if predecessor is not None:
            _sha(predecessor, "predecessor event")
        _utc(payload.get("recorded_at_utc"), "event time")
        mutation_requested = payload.get("mutation_requested")
        if not isinstance(mutation_requested, bool):
            raise JournalTamperError("controller mutation flag changed")
        outcome = payload["mutation_outcome"]
        if mutation_requested is False and outcome not in {"not_requested", "none"}:
            raise JournalTamperError("read-only event claims a mutation outcome")
        if mutation_requested is True and outcome == "not_requested":
            raise JournalTamperError("mutation event claims no request")
        return {**payload, "event_sha256": supplied}

    def load(self) -> list[JournalEvent]:
        entries = list(self.directory.iterdir())
        parsed: list[tuple[int, Path, re.Match[str]]] = []
        for path in entries:
            match = _EVENT_FILE.fullmatch(path.name)
            if not path.is_file() or path.is_symlink() or match is None:
                raise JournalTamperError(f"unexpected controller journal entry: {path.name}")
            parsed.append((int(match.group("sequence")), path, match))
        parsed.sort(key=lambda row: row[0])
        events: list[JournalEvent] = []
        previous: str | None = None
        for expected, (sequence, path, match) in enumerate(parsed, start=1):
            if sequence != expected:
                raise JournalTamperError("controller journal sequence has a gap or fork")
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise JournalTamperError("controller journal event is unreadable") from exc
            if raw != canonical_bytes(value) + b"\n":
                raise JournalTamperError("controller event is not canonical JSON")
            checked = self._validate_event(
                value,
                context_sha256=self.context_sha256,
                expected_sequence=expected,
                expected_previous=previous,
            )
            if match.group("phase") not in {checked["phase"], "event"}:
                raise JournalTamperError("controller event filename phase changed")
            previous = checked["event_sha256"]
            events.append(JournalEvent(path=path, value=checked))
        return events

    def append(
        self,
        *,
        phase: str,
        mode: str,
        status: str,
        operation_key: str,
        predecessor_event_sha256: str | None,
        evidence: Mapping[str, Any],
        output: Any,
        mutation_requested: bool,
        mutation_outcome: str,
        recorded_at_utc: str,
        expected_previous_event_sha256: str | None | object = _APPEND_HEAD_UNSET,
    ) -> JournalEvent:
        events = self.load()
        if expected_previous_event_sha256 is not _APPEND_HEAD_UNSET:
            expected_head = (
                None
                if expected_previous_event_sha256 is None
                else _sha(
                    expected_previous_event_sha256,
                    "expected controller journal head",
                )
            )
            actual_head = (
                None if not events else events[-1].value["event_sha256"]
            )
            if actual_head != expected_head:
                raise JournalTamperError(
                    "controller journal head changed before linearized append"
                )
        sequence = len(events) + 1
        safe_phase = _safe(phase, "phase")
        _safe(mode, "mode")
        _safe(status, "status")
        _safe(operation_key, "operation key")
        if predecessor_event_sha256 is not None:
            _sha(predecessor_event_sha256, "predecessor event")
        if mutation_outcome not in _MUTATION_OUTCOMES:
            raise ValueError("controller mutation outcome changed")
        cloned_evidence = _json_clone(dict(evidence), "event evidence")
        cloned_output = _json_clone(output, "event output")
        core = {
            "schema": EVENT_SCHEMA,
            "controller_version": CONTROLLER_VERSION,
            "controller_context_sha256": self.context_sha256,
            "sequence": sequence,
            "phase": safe_phase,
            "mode": mode,
            "status": status,
            "operation_key": operation_key,
            "previous_event_sha256": (
                None if not events else events[-1].value["event_sha256"]
            ),
            "predecessor_event_sha256": predecessor_event_sha256,
            "evidence_sha256": canonical_sha256(cloned_evidence),
            "evidence": cloned_evidence,
            "output_sha256": canonical_sha256(cloned_output),
            "output": cloned_output,
            "mutation_requested": mutation_requested,
            "mutation_outcome": mutation_outcome,
            "retry_authorized": False,
            "recorded_at_utc": _utc(recorded_at_utc, "event time"),
            "current_profile_changed": False,
        }
        value = {**core, "event_sha256": canonical_sha256(core)}
        # Every writer contends for the same next-sequence pathname.  The old
        # phase-suffixed names remain readable, while new appends use the common
        # ``-event`` suffix as the create-exclusive linearization point.
        path = self.directory / f"{sequence:06d}-event.json"
        try:
            with path.open("xb") as handle:
                handle.write(canonical_bytes(value) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise JournalTamperError("concurrent controller journal append detected") from exc
        try:
            directory_fd = os.open(self.directory, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
        matches = [
            event
            for event in self.load()
            if event.path == path
            and event.value["sequence"] == sequence
            and event.value["event_sha256"] == value["event_sha256"]
        ]
        if len(matches) != 1:
            raise JournalTamperError(
                "appended controller event cannot be uniquely reloaded"
            )
        return matches[0]


class Full100WaveControllerV2:
    """Receipt-chained lifecycle controller.  It owns no credentials."""

    def __init__(
        self,
        *,
        journal_dir: str | Path,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        resume_plan: Mapping[str, Any],
        create_journal: bool = True,
    ) -> None:
        self.wave_plan = wave_v2.validate_wave_plan(wave_plan)
        self.attempt_ledger = wave_v2.validate_attempt_ledger(
            self.wave_plan, attempt_ledger
        )
        self.resume_plan = wave_v2.validate_resume_plan(
            self.wave_plan, self.attempt_ledger, resume_plan
        )
        self.context = {
            "run_name": self.wave_plan["run_name"],
            "execution_identity_sha256": self.wave_plan["execution_identity_sha256"],
            "wave_plan_sha256": self.wave_plan["schedule_sha256"],
            "attempt_ledger_sha256": self.attempt_ledger["ledger_sha256"],
            "resume_plan_sha256": self.resume_plan["resume_sha256"],
            "wave_index": self.resume_plan["resume_wave_index"],
        }
        self.context_sha256 = canonical_sha256(self.context)
        self.journal = ControllerJournal(
            journal_dir, context_sha256=self.context_sha256, create=create_journal
        )

    def inspect(self) -> dict[str, Any]:
        events = self.journal.load()
        pending = self._pending_operations(events)
        return {
            "schema": "hu_m31_t3_step6d_full100_wave_controller_inspection_v2",
            "controller_context_sha256": self.context_sha256,
            **deepcopy(self.context),
            "event_count": len(events),
            "latest_event_sha256": None if not events else events[-1].value["event_sha256"],
            "pending_mutation_operations": pending,
            "cloud_mutation_performed": False,
            "current_profile_changed": False,
        }

    @staticmethod
    def _pending_operations(events: Sequence[JournalEvent]) -> list[str]:
        intents = {
            event.value["operation_key"]
            for event in events
            if event.value["status"] == "intent"
        }
        terminal = {
            event.value["operation_key"]
            for event in events
            if event.value["status"] in _TERMINAL_STATUSES
        }
        return sorted(intents - terminal)

    def _event(self, event_sha256: str) -> JournalEvent:
        wanted = _sha(event_sha256, "controller event")
        matches = [
            event for event in self.journal.load()
            if event.value["event_sha256"] == wanted
        ]
        if len(matches) != 1:
            raise PermissionError("required predecessor event is absent or duplicated")
        return matches[0]

    def _terminal_for(self, operation_key: str) -> JournalEvent | None:
        matches = [
            event for event in self.journal.load()
            if event.value["operation_key"] == operation_key
            and event.value["status"] in _TERMINAL_STATUSES
        ]
        if len(matches) > 1:
            raise JournalTamperError("controller operation has multiple terminal events")
        return None if not matches else matches[0]

    def _begin_mutation(
        self,
        *,
        phase: str,
        operation_key: str,
        predecessor_event_sha256: str,
        evidence: Mapping[str, Any],
        recorded_at_utc: str,
    ) -> tuple[JournalEvent | None, JournalEvent | None]:
        events = self.journal.load()
        if phase == "authorize-launch" and any(
            event.value["phase"] == "prelaunch-abort-tombstone"
            and event.value["status"] == "complete"
            for event in events
        ):
            raise PermissionError(
                "authorize-launch is permanently vetoed by prelaunch abort"
            )
        terminals = [
            event
            for event in events
            if event.value["operation_key"] == operation_key
            and event.value["status"] in _TERMINAL_STATUSES
        ]
        if len(terminals) > 1:
            raise JournalTamperError("controller operation has multiple terminal events")
        if terminals:
            return None, terminals[0]
        pending = self._pending_operations(events)
        if operation_key in pending:
            raise PendingMutationError(
                "mutation intent has no terminal receipt; reconcile or clean up before retry"
            )
        if pending:
            raise PendingMutationError(
                "another mutation intent is unresolved; controller is fail-closed"
            )
        predecessor = self._event(predecessor_event_sha256)
        intent = self.journal.append(
            phase=phase,
            mode="mutation",
            status="intent",
            operation_key=operation_key,
            predecessor_event_sha256=predecessor.value["event_sha256"],
            evidence=evidence,
            output=None,
            mutation_requested=True,
            mutation_outcome="pending",
            recorded_at_utc=recorded_at_utc,
            expected_previous_event_sha256=(
                None if not events else events[-1].value["event_sha256"]
            ),
        )
        return intent, None

    def record_prelaunch_abort_tombstone(
        self,
        *,
        operation_key: str,
        predecessor_event_sha256: str,
        abort_plan_sha256: str,
        recorded_at_utc: str,
    ) -> JournalEvent:
        """Linearize a permanent old-execution launch veto in the shared journal.

        The journal's create-exclusive next-sequence file is the linearization
        point.  A concurrent ``authorize-launch`` intent and this tombstone race
        for that same file: exactly one can win, and the loser fails closed.
        """

        plan_sha = _sha(abort_plan_sha256, "prelaunch abort plan")
        predecessor = self._event(predecessor_event_sha256)
        events = self.journal.load()
        tombstones = [
            event
            for event in events
            if event.value["phase"] == "prelaunch-abort-tombstone"
            and event.value["status"] == "complete"
        ]
        if len(tombstones) > 1:
            raise JournalTamperError("multiple prelaunch abort tombstones exist")
        if tombstones:
            event = tombstones[0]
            output = event.value.get("output")
            if (
                event.value["operation_key"] != operation_key
                or event.value["predecessor_event_sha256"]
                != predecessor.value["event_sha256"]
                or event.value["phase"] != "prelaunch-abort-tombstone"
                or event.value["mode"] != "read-only"
                or event.value["status"] != "complete"
                or event.value["mutation_requested"] is not False
                or event.value["mutation_outcome"] != "not_requested"
                or event.value.get("evidence")
                != {"abort_plan_sha256": plan_sha}
                or not isinstance(output, Mapping)
                or set(output)
                != {
                    "schema", "status", "abort_plan_sha256", "run_name",
                    "execution_identity_sha256",
                    "old_execution_relaunch_authorized",
                    "additional_create_authorized", "current_profile_changed",
                }
                or output.get("schema")
                != "hu_m31_t3_step6d_full100_wave_prelaunch_abort_tombstone_v2"
                or output.get("status")
                != "old_execution_launch_permanently_vetoed"
                or output.get("abort_plan_sha256") != plan_sha
                or output.get("run_name") != self.wave_plan["run_name"]
                or output.get("execution_identity_sha256")
                != self.wave_plan["execution_identity_sha256"]
                or output.get("old_execution_relaunch_authorized") is not False
                or output.get("additional_create_authorized") is not False
                or output.get("current_profile_changed") is not False
            ):
                raise JournalTamperError("prelaunch abort tombstone binding changed")
            return event
        if any(
            event.value["phase"] in {"authorize-launch", "reconcile-create"}
            for event in events
        ):
            raise PermissionError(
                "prelaunch abort tombstone lost to a GCE create operation"
            )
        if self._pending_operations(events):
            raise PendingMutationError(
                "prelaunch abort tombstone refuses an unresolved mutation"
            )
        return self.journal.append(
            phase="prelaunch-abort-tombstone",
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=predecessor.value["event_sha256"],
            evidence={"abort_plan_sha256": plan_sha},
            output={
                "schema": "hu_m31_t3_step6d_full100_wave_prelaunch_abort_tombstone_v2",
                "status": "old_execution_launch_permanently_vetoed",
                "abort_plan_sha256": plan_sha,
                "run_name": self.wave_plan["run_name"],
                "execution_identity_sha256": self.wave_plan[
                    "execution_identity_sha256"
                ],
                "old_execution_relaunch_authorized": False,
                "additional_create_authorized": False,
                "current_profile_changed": False,
            },
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=recorded_at_utc,
            expected_previous_event_sha256=(
                None if not events else events[-1].value["event_sha256"]
            ),
        )

    def abandon_worker_iam_cleanup_request_before_intent(
        self,
        *,
        operation_key: str,
        stale_predecessor_event_sha256: str,
        fresh_compute_absence_event_sha256: str,
        recorded_at_utc: str,
    ) -> JournalEvent:
        """Permanently consume a stale cleanup request before any PUT intent.

        The common next-sequence journal file linearizes this marker against a
        concurrent mutation intent.  Whichever append loses must reload and
        fail closed; an abandoned operation key can never execute an action.
        """

        stale = self._event(stale_predecessor_event_sha256)
        fresh = self._event(fresh_compute_absence_event_sha256)
        events = self.journal.load()
        rows = [
            event
            for event in events
            if event.value["operation_key"] == operation_key
        ]
        if rows:
            if len(rows) != 1:
                raise JournalTamperError(
                    "stale cleanup request operation journal multiplicity changed"
                )
            event = rows[0]
            output = event.value.get("output")
            if (
                event.value["phase"] != "worker-iam-cleanup"
                or event.value["mode"] != "read-only"
                or event.value["status"] != "failed"
                or event.value["predecessor_event_sha256"]
                != stale.value["event_sha256"]
                or event.value["mutation_requested"] is not False
                or event.value["mutation_outcome"] != "none"
                or event.value.get("evidence")
                != {
                    "stale_predecessor_event_sha256": stale.value[
                        "event_sha256"
                    ],
                    "fresh_compute_absence_event_sha256": fresh.value[
                        "event_sha256"
                    ],
                }
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
                != stale.value["event_sha256"]
                or output.get("fresh_compute_absence_event_sha256")
                != fresh.value["event_sha256"]
                or output.get("cloud_mutation_performed") is not False
                or output.get("current_profile_changed") is not False
            ):
                raise PendingMutationError(
                    "cleanup operation key was consumed before abandonment"
                )
            return event
        if self._pending_operations(events):
            raise PendingMutationError(
                "stale request abandonment refuses an unresolved mutation"
            )
        return self.journal.append(
            phase="worker-iam-cleanup",
            mode="read-only",
            status="failed",
            operation_key=operation_key,
            predecessor_event_sha256=stale.value["event_sha256"],
            evidence={
                "stale_predecessor_event_sha256": stale.value["event_sha256"],
                "fresh_compute_absence_event_sha256": fresh.value["event_sha256"],
            },
            output={
                "schema": (
                    "hu_m31_t3_step6d_full100_wave_stale_cleanup_request_v2"
                ),
                "status": "stale_request_permanently_abandoned_before_intent",
                "failure_kind": "stale_request_abandoned_before_intent",
                "stale_predecessor_event_sha256": stale.value["event_sha256"],
                "fresh_compute_absence_event_sha256": fresh.value["event_sha256"],
                "cloud_mutation_performed": False,
                "current_profile_changed": False,
            },
            mutation_requested=False,
            mutation_outcome="none",
            recorded_at_utc=recorded_at_utc,
            expected_previous_event_sha256=(
                None if not events else events[-1].value["event_sha256"]
            ),
        )

    def _finish_mutation(
        self,
        *,
        intent: JournalEvent,
        status: str,
        output: Any,
        outcome: str,
        recorded_at_utc: str,
    ) -> JournalEvent:
        return self.journal.append(
            phase=intent.value["phase"],
            mode="mutation",
            status=status,
            operation_key=intent.value["operation_key"],
            predecessor_event_sha256=intent.value["event_sha256"],
            evidence={"intent_event_sha256": intent.value["event_sha256"]},
            output=output,
            mutation_requested=True,
            mutation_outcome=outcome,
            recorded_at_utc=recorded_at_utc,
        )

    def record_read_phase(
        self,
        *,
        phase: str,
        operation_key: str,
        predecessor_event_sha256: str | None,
        evidence: Mapping[str, Any],
        output: Mapping[str, Any],
        recorded_at_utc: str,
    ) -> JournalEvent:
        """Append one idempotent pure/read-only producer receipt."""

        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != phase:
                raise PermissionError("operation key was consumed by another phase")
            return prior
        if predecessor_event_sha256 is not None:
            self._event(predecessor_event_sha256)
        return self.journal.append(
            phase=phase,
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=predecessor_event_sha256,
            evidence=evidence,
            output=output,
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=recorded_at_utc,
        )

    def run_mutation_phase(
        self,
        *,
        phase: str,
        operation_key: str,
        predecessor_event_sha256: str,
        evidence: Mapping[str, Any],
        action: Callable[[], Mapping[str, Any]],
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        started_at_utc: str,
        completed_at_utc: str,
    ) -> JournalEvent:
        """Run one existing mutation primitive under durable intent/terminal events."""

        intent, terminal = self._begin_mutation(
            phase=phase,
            operation_key=operation_key,
            predecessor_event_sha256=predecessor_event_sha256,
            evidence=evidence,
            recorded_at_utc=started_at_utc,
        )
        if terminal is not None:
            return terminal
        assert intent is not None
        try:
            receipt = deepcopy(dict(validator(action())))
        except Exception as exc:
            failure_kind = (
                "transport_ambiguity"
                if isinstance(exc, gcp_v2.GcpPhaseATransportError)
                else "provider_rejected"
                if isinstance(exc, worker_iam_v2.WorkerIamCasError)
                else "local_or_response_failure"
                if phase in {"worker-iam-install", "worker-iam-cleanup"}
                else "provider_rejected_or_local_failure"
            )
            self._finish_mutation(
                intent=intent,
                status="failed",
                output={"failure_kind": failure_kind},
                outcome="unknown",
                recorded_at_utc=completed_at_utc,
            )
            raise
        return self._finish_mutation(
            intent=intent,
            status="complete",
            output={"receipt": receipt},
            outcome="performed",
            recorded_at_utc=completed_at_utc,
        )

    def record_ambiguous_mutation_reconciliation(
        self,
        *,
        phase: str,
        operation_key: str,
        source_operation_key: str,
        receipt: Mapping[str, Any],
        output_key: str,
        recorded_at_utc: str,
    ) -> JournalEvent:
        """Attach a GET-only recovery proof to one outcome-ambiguous PUT.

        Older controller journals used ``provider_rejected_or_local_failure``
        for both an explicit CAS rejection and a local validation failure after
        a successful PUT.  Install recovery may accept that legacy value: the
        producer has already performed an exact GET-only proof that all planned
        bindings are installed and that the unrelated-policy fingerprint still
        matches the pre-PUT receipt.  Cleanup also accepts the narrower
        ``local_or_response_failure`` outcome ambiguity: the GET-only producer
        must prove the desired cleanup state by exact target absence and
        unchanged unrelated policy.  This proof does not claim that the source
        PUT ran; it only closes the ambiguous outcome without authorizing a
        second PUT.
        """

        if phase not in {
            "worker-iam-reconcile-install",
            "worker-iam-reconcile-cleanup",
        }:
            raise ValueError("ambiguous mutation reconciliation phase changed")
        source_phase = phase.removeprefix("worker-iam-reconcile-")
        source_phase = f"worker-iam-{source_phase}"
        source = [
            event
            for event in self.journal.load()
            if event.value["operation_key"] == source_operation_key
            and event.value["phase"] == source_phase
        ]
        intents = [event for event in source if event.value["status"] == "intent"]
        failures = [event for event in source if event.value["status"] == "failed"]
        failure_kind = (
            failures[0].value.get("output", {}).get("failure_kind")
            if len(failures) == 1
            else None
        )
        allowed_failure_kinds = (
            {"transport_ambiguity", "local_or_response_failure",
             "provider_rejected_or_local_failure"}
            if phase == "worker-iam-reconcile-install"
            else {"transport_ambiguity", "local_or_response_failure"}
        )
        if (
            len(intents) != 1
            or len(failures) != 1
            or failures[0].value["predecessor_event_sha256"]
            != intents[0].value["event_sha256"]
            or failures[0].value["mutation_outcome"] != "unknown"
            or failure_kind not in allowed_failure_kinds
        ):
            raise PermissionError(
                "IAM reconciliation requires exactly one "
                + (
                    "authorized ambiguous source"
                    if phase == "worker-iam-reconcile-install"
                    else "transport/local-response outcome-ambiguous source"
                )
            )
        checked = deepcopy(dict(receipt))
        return self.record_read_phase(
            phase=phase,
            operation_key=operation_key,
            predecessor_event_sha256=failures[0].value["event_sha256"],
            evidence={
                "source_operation_key": source_operation_key,
                "source_intent_event_sha256": intents[0].value["event_sha256"],
                "source_failure_event_sha256": failures[0].value["event_sha256"],
                "source_failure_kind": failure_kind,
                "recovered_receipt_sha256": _object_sha(
                    checked, "receipt_sha256"
                ),
            },
            output={output_key: checked},
            recorded_at_utc=recorded_at_utc,
        )

    def prepare(
        self,
        *,
        operation_key: str,
        outer_manifest: Mapping[str, Any],
        stage_plan: Mapping[str, Any],
        content_preflight_receipt: Mapping[str, Any],
        worker_identity_plan: Mapping[str, Any],
        runtime_preflight_receipt: Mapping[str, Any],
        runtime_gcp_read_receipt: Mapping[str, Any],
        current_time_utc: str,
        pool_read_receipt: Mapping[str, Any] | None = None,
        pool_read_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> JournalEvent:
        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "prepare":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        manifest = package_v2.validate_outer_manifest(
            self.wave_plan,
            outer_manifest,
            expected_startup_sha256=science_registry.resolve_startup_sha256(
                self.wave_plan
            ),
        )
        content_plan = content_v2.validate_content_stage_plan(manifest, stage_plan)
        preflight = content_v2.validate_preflight_absence_receipt(
            content_plan, content_preflight_receipt
        )
        identity = identity_v2.validate_worker_identity_plan(
            wave_plan=self.wave_plan,
            attempt_ledger=self.attempt_ledger,
            resume_plan=self.resume_plan,
            value=worker_identity_plan,
        )
        runtime = runtime_v2.validate_runtime_preflight_receipt(
            wave_plan=self.wave_plan,
            value=runtime_preflight_receipt,
            current_utc=current_time_utc,
        )
        runtime_gcp = runtime_gcp_v2.validate_runtime_gcp_read_receipt(
            wave_plan=self.wave_plan,
            value=runtime_gcp_read_receipt,
            current_utc=current_time_utc,
        )
        if runtime_gcp["runtime_preflight_receipt"] != runtime:
            raise ValueError("prepare runtime preflight lacks live GCP provenance")
        if (pool_read_receipt is None) != (pool_read_validator is None):
            raise ValueError("pool read receipt and validator must be supplied together")
        pool_read = (
            None
            if pool_read_receipt is None
            else deepcopy(dict(pool_read_validator(pool_read_receipt)))  # type: ignore[misc]
        )
        hashes = {
            "outer_manifest_sha256": manifest["manifest_sha256"],
            "stage_plan_sha256": content_plan["plan_sha256"],
            "content_preflight_receipt_sha256": preflight["receipt_sha256"],
            "worker_identity_plan_sha256": identity["plan_sha256"],
            "runtime_preflight_receipt_sha256": runtime["receipt_sha256"],
            "runtime_gcp_read_receipt_sha256": runtime_gcp["receipt_sha256"],
            "pool_read_receipt_sha256": (
                None if pool_read is None else _object_sha(pool_read, "receipt_sha256")
            ),
        }
        receipt_core = {
            "schema": PREPARE_SCHEMA,
            "status": "validated_read_only_inputs_ready_no_cloud_mutation",
            "controller_context_sha256": self.context_sha256,
            **deepcopy(self.context),
            **hashes,
            "validated_at_utc": _utc(current_time_utc, "prepare current time"),
            "cloud_mutation_authorized": False,
            "read_only": True,
            "current_profile_changed": False,
        }
        receipt = _seal(receipt_core, "receipt_sha256")
        return self.journal.append(
            phase="prepare",
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=None,
            evidence=hashes,
            output=receipt,
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=current_time_utc,
        )

    def setup_identities(
        self,
        *,
        operation_key: str,
        prepare_event_sha256: str,
        adapter: CreateMissingAdapter,
        current_time_utc: str,
        completed_at_utc: str,
        allow_cloud_mutation: bool = False,
    ) -> JournalEvent:
        if allow_cloud_mutation is not True:
            raise PermissionError("setup-identities requires explicit cloud mutation opt-in")
        prepare = self._event(prepare_event_sha256)
        if prepare.value["phase"] != "prepare" or prepare.value["status"] != "complete":
            raise PermissionError("setup-identities requires an exact completed prepare event")
        intent, terminal = self._begin_mutation(
            phase="setup-identities",
            operation_key=operation_key,
            predecessor_event_sha256=prepare_event_sha256,
            evidence={"prepare_event_sha256": prepare_event_sha256},
            recorded_at_utc=current_time_utc,
        )
        if terminal is not None:
            return terminal
        assert intent is not None
        try:
            raw = adapter.create_missing(
                current_time_utc=current_time_utc,
                completed_at_utc=completed_at_utc,
            )
            receipt = deepcopy(dict(adapter.validate_create_receipt(raw)))
        except identity_gcp_v2.WorkerIdentityCreateIncompleteError as exc:
            self._finish_mutation(
                intent=intent,
                status="partial",
                output={
                    "identity_create_receipt": None,
                    "recovery_read_receipt": exc.recovery_read_receipt,
                },
                outcome="partial",
                recorded_at_utc=completed_at_utc,
            )
            raise
        except Exception:
            self._finish_mutation(
                intent=intent,
                status="failed",
                output=None,
                outcome="unknown",
                recorded_at_utc=completed_at_utc,
            )
            raise
        return self._finish_mutation(
            intent=intent,
            status="complete",
            output={"identity_create_receipt": receipt},
            outcome="performed",
            recorded_at_utc=completed_at_utc,
        )

    def stage_content(
        self,
        *,
        operation_key: str,
        prepare_event_sha256: str,
        package_dir: str | Path,
        stage_plan: Mapping[str, Any],
        preflight_receipt: Mapping[str, Any],
        backend: content_v2.ContentObjectBackend,
        observed_at_utc: str,
        allow_cloud_mutation: bool = False,
    ) -> JournalEvent:
        if allow_cloud_mutation is not True:
            raise PermissionError("stage-content requires explicit cloud mutation opt-in")
        prepare = self._event(prepare_event_sha256)
        prepared = prepare.value["output"]
        plan = content_v2._validate_stage_plan_self(stage_plan)
        preflight = content_v2.validate_preflight_absence_receipt(plan, preflight_receipt)
        if (
            prepare.value["phase"] != "prepare"
            or prepared.get("stage_plan_sha256") != plan["plan_sha256"]
            or prepared.get("content_preflight_receipt_sha256")
            != preflight["receipt_sha256"]
        ):
            raise PermissionError("stage-content evidence differs from prepare")
        intent, terminal = self._begin_mutation(
            phase="stage-content",
            operation_key=operation_key,
            predecessor_event_sha256=prepare_event_sha256,
            evidence={
                "prepare_event_sha256": prepare_event_sha256,
                "stage_plan_sha256": plan["plan_sha256"],
                "preflight_receipt_sha256": preflight["receipt_sha256"],
            },
            recorded_at_utc=observed_at_utc,
        )
        if terminal is not None:
            return terminal
        assert intent is not None
        try:
            raw = content_v2.execute_content_stage(
                package_dir=package_dir,
                wave_plan=self.wave_plan,
                expected_startup_sha256=science_registry.resolve_startup_sha256(
                    self.wave_plan
                ),
                stage_plan=plan,
                preflight_receipt=preflight,
                backend=backend,
                observed_at_utc=observed_at_utc,
            )
            receipt = content_v2.validate_stage_receipt(plan, preflight, raw)
        except content_v2.ContentStageIncompleteError as exc:
            partial = exc.partial_receipt
            checked = (
                None
                if partial is None
                else content_v2.validate_stage_receipt(plan, preflight, partial)
            )
            self._finish_mutation(
                intent=intent,
                status="partial" if checked is not None else "failed",
                output=None if checked is None else {"stage_receipt": checked},
                outcome="partial" if checked is not None else "unknown",
                recorded_at_utc=observed_at_utc,
            )
            raise
        except Exception:
            self._finish_mutation(
                intent=intent,
                status="failed",
                output=None,
                outcome="unknown",
                recorded_at_utc=observed_at_utc,
            )
            raise
        if receipt.get("stage_complete") is not True:
            raise RuntimeError("content stage returned a non-complete receipt")
        return self._finish_mutation(
            intent=intent,
            status="complete",
            output={"stage_receipt": receipt},
            outcome="performed",
            recorded_at_utc=observed_at_utc,
        )

    def bind_existing_staged_content(
        self,
        *,
        operation_key: str,
        prepare_event_sha256: str,
        stage_plan: Mapping[str, Any],
        preflight_receipt: Mapping[str, Any],
        source_stage_receipt: Mapping[str, Any],
        backend: content_v2.ContentObjectBackend,
        observed_at_utc: str,
    ) -> JournalEvent:
        """Bind a fresh exact-generation readback into this wave journal.

        Immutable run content is uploaded once.  Later wave controller
        contexts may only reuse it after all 26 source-owned generations have
        been observed again.  This path is intentionally GET-only and records
        no mutation intent, so it cannot accidentally restage the create-only
        prefix.
        """

        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "stage-content":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        prepare = self._event(prepare_event_sha256)
        prepared = prepare.value["output"]
        plan = content_v2._validate_stage_plan_self(stage_plan)
        preflight = content_v2.validate_preflight_absence_receipt(
            plan, preflight_receipt
        )
        source = content_v2.validate_stage_receipt(
            plan, preflight, source_stage_receipt
        )
        if (
            prepare.value["phase"] != "prepare"
            or prepare.value["status"] != "complete"
            or prepared.get("stage_plan_sha256") != plan["plan_sha256"]
            or prepared.get("content_preflight_receipt_sha256")
            != preflight["receipt_sha256"]
            or source.get("stage_complete") is not True
            or source.get("created_entry_count") != 26
        ):
            raise PermissionError("existing staged content differs from prepare")
        fresh = content_v2.build_stage_readback_receipt(
            stage_plan=plan,
            preflight_receipt=preflight,
            created_rows=source["rows"],
            backend=backend,
            observed_at_utc=observed_at_utc,
        )
        receipt = content_v2.validate_stage_receipt(plan, preflight, fresh)
        if (
            receipt.get("stage_complete") is not True
            or receipt.get("created_entry_count") != 26
            or receipt.get("rows") != source.get("rows")
        ):
            raise RuntimeError("existing staged content generation inventory changed")
        return self.journal.append(
            phase="stage-content",
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=prepare_event_sha256,
            evidence={
                "prepare_event_sha256": prepare_event_sha256,
                "stage_plan_sha256": plan["plan_sha256"],
                "preflight_receipt_sha256": preflight["receipt_sha256"],
                "source_stage_receipt_sha256": source["receipt_sha256"],
                "fresh_stage_receipt_sha256": receipt["receipt_sha256"],
            },
            output={
                "stage_receipt": receipt,
                "source_stage_receipt_sha256": source["receipt_sha256"],
                "content_reused_without_mutation": True,
            },
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=observed_at_utc,
        )

    def reconcile_pending_mutation(
        self,
        *,
        operation_key: str,
        receipt: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        complete: bool,
        recorded_at_utc: str,
    ) -> JournalEvent:
        events = self.journal.load()
        intents = [
            event for event in events
            if event.value["operation_key"] == operation_key
            and event.value["status"] == "intent"
        ]
        if len(intents) != 1 or self._terminal_for(operation_key) is not None:
            raise PermissionError("operation is not one unresolved mutation intent")
        checked = deepcopy(dict(validator(receipt)))
        return self._finish_mutation(
            intent=intents[0],
            status="reconciled-complete" if complete else "reconciled-partial",
            output={"reconciled_receipt": checked},
            outcome="performed" if complete else "partial",
            recorded_at_utc=recorded_at_utc,
        )

    def resolve_pending_mutation_no_effect(
        self,
        *,
        operation_key: str,
        receipt: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        recorded_at_utc: str,
    ) -> JournalEvent:
        """Close one interrupted intent only after exact pre-state is re-proven.

        This is intentionally distinct from partial reconciliation: the caller's
        GET-only validator must prove that the authorized mutation had no effect,
        after which a new operation key may safely make one fresh CAS attempt.
        """

        events = self.journal.load()
        intents = [
            event
            for event in events
            if event.value["operation_key"] == operation_key
            and event.value["status"] == "intent"
        ]
        if len(intents) != 1 or self._terminal_for(operation_key) is not None:
            raise PermissionError("operation is not one unresolved mutation intent")
        checked = deepcopy(dict(validator(receipt)))
        return self._finish_mutation(
            intent=intents[0],
            status="failed",
            output={
                "failure_kind": "process_loss_no_effect_proven",
                "reconciled_receipt": checked,
            },
            outcome="none",
            recorded_at_utc=recorded_at_utc,
        )

    def reconcile_launch_create(
        self,
        *,
        operation_key: str,
        source_launch_operation_key: str,
        adapter: Any,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
    ) -> JournalEvent:
        """Record a POST-free all-selected-name scan after interrupted launch."""

        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "reconcile-create":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        source_events = [
            event for event in self.journal.load()
            if event.value["operation_key"] == source_launch_operation_key
            and event.value["phase"] == "authorize-launch"
        ]
        intents = [event for event in source_events if event.value["status"] == "intent"]
        terminals = [
            event for event in source_events if event.value["status"] in _TERMINAL_STATUSES
        ]
        if len(intents) != 1 or len(terminals) > 1:
            raise PermissionError("launch reconciliation source is missing or ambiguous")
        if terminals and terminals[0].value["status"] == "complete":
            raise PermissionError("complete launch must not enter create reconciliation")
        predecessor = terminals[0] if terminals else intents[0]
        raw = adapter.reconcile_create(
            request_ids=request_ids, observed_at_utc=observed_at_utc
        )
        receipt = deepcopy(dict(adapter.validate_create_receipt(raw)))
        return self.record_read_phase(
            phase="reconcile-create",
            operation_key=operation_key,
            predecessor_event_sha256=predecessor.value["event_sha256"],
            evidence={
                "source_launch_intent_sha256": intents[0].value["event_sha256"],
                "request_ids_sha256": canonical_sha256(request_ids),
            },
            output={"gce_create_receipt": receipt},
            recorded_at_utc=observed_at_utc,
        )

    @staticmethod
    def bridge_actual_launch_receipt(
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        resume_plan: Mapping[str, Any],
        launch_bundle: Mapping[str, Any],
        gce_create_receipt: Mapping[str, Any],
        quota_receipt: Mapping[str, Any],
        persistent_claim_receipt: Mapping[str, Any],
        planned_mapping_receipt: Mapping[str, Any],
        prelaunch_authorization: Mapping[str, Any],
        launch_started_at_utc: str,
    ) -> dict[str, Any]:
        rows = gce_create_receipt.get("rows")
        planned = planned_mapping_receipt.get("rows")
        if (
            gce_create_receipt.get("create_complete") is not True
            or not isinstance(rows, list)
            or not isinstance(planned, list)
            or len(rows) != len(planned)
        ):
            raise ValueError("only a complete exact GCE create can bridge to cloud launch")
        observations: list[dict[str, Any]] = []
        for mapping, created in zip(planned, rows, strict=True):
            if not isinstance(mapping, Mapping) or not isinstance(created, Mapping):
                raise ValueError("launch bridge row is not an object")
            if (
                created.get("job_id") != mapping.get("job_id")
                or created.get("source_role") != mapping.get("source_role")
                or created.get("attempt_id") != mapping.get("attempt_id")
                or created.get("instance_name") != mapping.get("instance_id")
                or created.get("recovered_after_insert_failure") is not False
                or created.get("operation_id") is None
            ):
                raise ValueError("GCE create escaped the planned mapping")
            observations.append(
                {
                    "job_id": mapping["job_id"],
                    "source_role": mapping["source_role"],
                    "attempt_id": mapping["attempt_id"],
                    "instance_id": mapping["instance_id"],
                    "operation_id": str(created["operation_id"]),
                    "operation_status": "DONE",
                    "instance_status": created["observed_status"],
                    "ownership_label": mapping["ownership_label"],
                    "machine_type": mapping["machine_type"],
                    "vcpus": mapping["vcpus"],
                    "instance_readback_complete": True,
                }
            )
        return cloud_v2.build_actual_launch_receipt(
            wave_plan,
            attempt_ledger,
            resume_plan,
            immutable_content_sha256=launch_bundle["immutable_content_sha256"],
            quota_receipt=quota_receipt,
            persistent_claim_receipt=persistent_claim_receipt,
            planned_mapping_receipt=planned_mapping_receipt,
            prelaunch_authorization=prelaunch_authorization,
            launch_started_at_utc=launch_started_at_utc,
            observed_at_utc=gce_create_receipt["observed_at_utc"],
            instance_create_readbacks=observations,
            readback_source="compute_instances_and_disks_api",
        )

    def authorize_and_launch(
        self,
        *,
        operation_key: str,
        stage_event_sha256: str,
        launch_bundle: Mapping[str, Any],
        launch_bundle_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        gce_adapter: GceCreateAdapter,
        request_ids: Mapping[str, Any],
        launch_started_at_utc: str,
        observed_at_utc: str,
        quota_receipt: Mapping[str, Any],
        persistent_claim_receipt: Mapping[str, Any],
        planned_mapping_receipt: Mapping[str, Any],
        prelaunch_authorization: Mapping[str, Any],
        allow_cloud_mutation: bool = False,
    ) -> JournalEvent:
        if allow_cloud_mutation is not True:
            raise PermissionError("authorize-launch requires explicit cloud mutation opt-in")
        stage = self._event(stage_event_sha256)
        stage_output = stage.value["output"]
        if (
            stage.value["phase"] != "stage-content"
            or stage.value["status"] != "complete"
            or stage_output.get("stage_receipt", {}).get("stage_complete") is not True
        ):
            raise PermissionError("launch requires a complete staged-content event")
        bundle = deepcopy(dict(launch_bundle_validator(launch_bundle)))
        stage_receipt = stage_output["stage_receipt"]
        if bundle.get("immutable_content_sha256") != stage_receipt.get("content_payload_sha256"):
            raise PermissionError("launch bundle content differs from staged content")
        intent, terminal = self._begin_mutation(
            phase="authorize-launch",
            operation_key=operation_key,
            predecessor_event_sha256=stage_event_sha256,
            evidence={
                "stage_event_sha256": stage_event_sha256,
                "stage_receipt_sha256": stage_receipt["receipt_sha256"],
                "launch_bundle_sha256": bundle["bundle_sha256"],
                "request_ids_sha256": canonical_sha256(request_ids),
            },
            recorded_at_utc=launch_started_at_utc,
        )
        if terminal is not None:
            return terminal
        assert intent is not None
        try:
            raw = gce_adapter.create_selected(
                request_ids=request_ids,
                observed_at_utc=observed_at_utc,
                prior_create_receipt=None,
            )
            create_receipt = deepcopy(dict(gce_adapter.validate_create_receipt(raw)))
        except gce_v2.GceCreateIncompleteError as exc:
            partial = deepcopy(dict(gce_adapter.validate_create_receipt(exc.partial_receipt)))
            self._finish_mutation(
                intent=intent,
                status="partial",
                output={"gce_create_receipt": partial, "actual_launch_receipt": None},
                outcome="partial",
                recorded_at_utc=observed_at_utc,
            )
            raise
        except Exception:
            self._finish_mutation(
                intent=intent,
                status="failed",
                output=None,
                outcome="unknown",
                recorded_at_utc=observed_at_utc,
            )
            raise
        try:
            actual = self.bridge_actual_launch_receipt(
                wave_plan=self.wave_plan,
                attempt_ledger=self.attempt_ledger,
                resume_plan=self.resume_plan,
                launch_bundle=bundle,
                gce_create_receipt=create_receipt,
                quota_receipt=quota_receipt,
                persistent_claim_receipt=persistent_claim_receipt,
                planned_mapping_receipt=planned_mapping_receipt,
                prelaunch_authorization=prelaunch_authorization,
                launch_started_at_utc=launch_started_at_utc,
            )
        except Exception:
            self._finish_mutation(
                intent=intent,
                status="partial",
                output={"gce_create_receipt": create_receipt, "actual_launch_receipt": None},
                outcome="partial",
                recorded_at_utc=observed_at_utc,
            )
            raise
        return self._finish_mutation(
            intent=intent,
            status="complete",
            output={"gce_create_receipt": create_receipt, "actual_launch_receipt": actual},
            outcome="performed",
            recorded_at_utc=observed_at_utc,
        )

    def status(
        self,
        *,
        operation_key: str,
        launch_event_sha256: str,
        adapter: GceStatusAdapter,
        observed_at_utc: str,
    ) -> JournalEvent:
        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "status":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        launch = self._event(launch_event_sha256)
        if launch.value["phase"] != "authorize-launch" or launch.value["status"] not in {
            "complete", "partial"
        }:
            raise PermissionError("status requires an exact launch receipt event")
        raw = adapter.read_status(observed_at_utc=observed_at_utc)
        receipt = deepcopy(dict(adapter.validate_status_receipt(raw)))
        return self.journal.append(
            phase="status",
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=launch_event_sha256,
            evidence={
                "launch_event_sha256": launch_event_sha256,
                "create_receipt_sha256": receipt["create_receipt_sha256"],
            },
            output={"gce_status_receipt": receipt},
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=observed_at_utc,
        )

    def delete_instances(
        self,
        *,
        operation_key: str,
        launch_event_sha256: str,
        adapter: GceDeleteAdapter,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
        orphan_disk_request_ids: Mapping[str, Any] | None = None,
        reconcile_event_sha256: str | None = None,
        allow_cloud_mutation: bool = False,
    ) -> JournalEvent:
        if allow_cloud_mutation is not True:
            raise PermissionError("cleanup delete requires explicit cloud mutation opt-in")
        launch = self._event(launch_event_sha256)
        if (
            launch.value["phase"] not in {"authorize-launch", "reconcile-create"}
            or launch.value["status"] not in {"complete", "partial"}
            or not isinstance(
                launch.value.get("output", {}).get("gce_create_receipt"), Mapping
            )
        ):
            raise PermissionError("cleanup requires a complete or partial owned create receipt")
        predecessor = launch_event_sha256
        if reconcile_event_sha256 is not None:
            reconciled = self._event(reconcile_event_sha256)
            reconcile_receipt = reconciled.value.get("output", {}).get(
                "gce_delete_reconciliation_receipt"
            )
            if (
                reconciled.value["phase"] != "reconcile-delete"
                or reconciled.value["status"] != "partial"
                or not isinstance(reconcile_receipt, Mapping)
                or reconcile_receipt.get("orphan_cleanup_required") is not True
                or reconciled.value.get("evidence", {}).get("launch_event_sha256")
                != launch_event_sha256
            ):
                raise PermissionError("orphan cleanup requires exact partial reconciliation")
            predecessor = reconcile_event_sha256
        intent, terminal = self._begin_mutation(
            phase="cleanup-delete",
            operation_key=operation_key,
            predecessor_event_sha256=predecessor,
            evidence={
                "launch_event_sha256": launch_event_sha256,
                "request_ids_sha256": canonical_sha256(request_ids),
                "orphan_disk_request_ids_sha256": (
                    None
                    if orphan_disk_request_ids is None
                    else canonical_sha256(orphan_disk_request_ids)
                ),
                "reconcile_event_sha256": reconcile_event_sha256,
            },
            recorded_at_utc=observed_at_utc,
        )
        if terminal is not None:
            return terminal
        assert intent is not None
        try:
            raw = adapter.delete_owned(
                request_ids=request_ids,
                orphan_disk_request_ids=orphan_disk_request_ids,
                observed_at_utc=observed_at_utc,
            )
            receipt = deepcopy(dict(adapter.validate_delete_receipt(raw)))
        except Exception as exc:
            self._finish_mutation(
                intent=intent,
                status="failed",
                output={
                    "failure_kind": (
                        "transport_ambiguity"
                        if isinstance(exc, gce_v2.GcePhaseBTransportError)
                        else "provider_rejected_or_local_failure"
                    )
                },
                outcome="unknown",
                recorded_at_utc=observed_at_utc,
            )
            raise
        return self._finish_mutation(
            intent=intent,
            status="complete",
            output={"gce_delete_receipt": receipt},
            outcome="performed" if receipt.get("delete_operation_count", 0) else "none",
            recorded_at_utc=observed_at_utc,
        )

    def reconcile_delete(
        self,
        *,
        operation_key: str,
        source_operation_key: str,
        launch_event_sha256: str,
        adapter: GceDeleteReconcileAdapter,
        request_ids: Mapping[str, Any],
        observed_at_utc: str,
    ) -> JournalEvent:
        """Record one GET-only classification of an ambiguous instance DELETE."""

        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "reconcile-delete":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        source = [
            event for event in self.journal.load()
            if event.value["operation_key"] == source_operation_key
            and event.value["phase"] == "cleanup-delete"
        ]
        intents = [event for event in source if event.value["status"] == "intent"]
        failures = [event for event in source if event.value["status"] == "failed"]
        source_predecessor_ok = False
        if len(intents) == 1:
            source_predecessor = intents[0].value["predecessor_event_sha256"]
            source_predecessor_ok = source_predecessor == launch_event_sha256
            if not source_predecessor_ok:
                try:
                    prior_reconcile = self._event(source_predecessor)
                except PermissionError:
                    prior_reconcile = None
                prior_output = (
                    None if prior_reconcile is None
                    else prior_reconcile.value.get("output")
                )
                prior_receipt = (
                    prior_output.get("gce_delete_reconciliation_receipt")
                    if isinstance(prior_output, Mapping)
                    else None
                )
                source_predecessor_ok = (
                    prior_reconcile is not None
                    and prior_reconcile.value["phase"] == "reconcile-delete"
                    and prior_reconcile.value["status"] == "partial"
                    and prior_reconcile.value.get("evidence", {}).get(
                        "launch_event_sha256"
                    ) == launch_event_sha256
                    and isinstance(prior_receipt, Mapping)
                    and prior_receipt.get("orphan_cleanup_required") is True
                    and intents[0].value.get("evidence", {}).get(
                        "reconcile_event_sha256"
                    ) == prior_reconcile.value["event_sha256"]
                )
        if (
            len(intents) != 1
            or len(failures) != 1
            or not source_predecessor_ok
            or failures[0].value["predecessor_event_sha256"]
            != intents[0].value["event_sha256"]
            or failures[0].value["mutation_outcome"] != "unknown"
            or failures[0].value.get("output", {}).get("failure_kind")
            != "transport_ambiguity"
        ):
            raise PermissionError(
                "delete reconciliation requires one transport-ambiguous source"
            )
        if (
            canonical_sha256(request_ids)
            != intents[0].value.get("evidence", {}).get("request_ids_sha256")
        ):
            raise PermissionError(
                "delete reconciliation request IDs do not match cleanup-delete intent"
            )
        raw = adapter.reconcile_delete(
            request_ids=request_ids, observed_at_utc=observed_at_utc
        )
        receipt = deepcopy(dict(adapter.validate_delete_reconcile_receipt(raw)))
        partial = receipt.get("orphan_cleanup_required") is True
        output: dict[str, Any] = {
            "gce_delete_reconciliation_receipt": receipt,
        }
        if not partial:
            recovered = receipt.get("recovered_delete_receipt")
            if not isinstance(recovered, Mapping):
                raise ValueError("complete delete reconciliation lacks delete receipt")
            output["gce_delete_receipt"] = deepcopy(dict(recovered))
        return self.journal.append(
            phase="reconcile-delete",
            mode="read-only",
            status="partial" if partial else "complete",
            operation_key=operation_key,
            predecessor_event_sha256=failures[0].value["event_sha256"],
            evidence={
                "source_operation_key": source_operation_key,
                "source_failure_event_sha256": failures[0].value["event_sha256"],
                "launch_event_sha256": launch_event_sha256,
                "request_ids_sha256": canonical_sha256(request_ids),
            },
            output=output,
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=observed_at_utc,
        )

    def verify_instance_absence(
        self,
        *,
        operation_key: str,
        delete_event_sha256: str,
        adapter: GceAbsenceAdapter,
        observed_at_utc: str,
    ) -> JournalEvent:
        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "cleanup-absence":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        delete = self._event(delete_event_sha256)
        if (
            delete.value["phase"] not in {"cleanup-delete", "reconcile-delete"}
            or delete.value["status"] != "complete"
        ):
            raise PermissionError("absence verification requires exact cleanup-delete evidence")
        raw = adapter.verify_absence(observed_at_utc=observed_at_utc)
        receipt = deepcopy(dict(adapter.validate_absence_receipt(raw)))
        return self.journal.append(
            phase="cleanup-absence",
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=delete_event_sha256,
            evidence={
                "delete_event_sha256": delete_event_sha256,
                "delete_receipt_sha256": receipt["delete_receipt_sha256"],
            },
            output={"gce_absence_receipt": receipt},
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=observed_at_utc,
        )

    def cleanup_content(
        self,
        *,
        operation_key: str,
        stage_event_sha256: str,
        stage_plan: Mapping[str, Any],
        preflight_receipt: Mapping[str, Any],
        backend: content_v2.ContentObjectBackend,
        observed_at_utc: str,
        allow_cloud_mutation: bool = False,
    ) -> JournalEvent:
        if allow_cloud_mutation is not True:
            raise PermissionError("content cleanup requires explicit cloud mutation opt-in")
        stage = self._event(stage_event_sha256)
        stage_receipt = stage.value.get("output", {}).get("stage_receipt")
        if stage.value["phase"] != "stage-content" or not isinstance(stage_receipt, Mapping):
            raise PermissionError("content cleanup requires owned stage generations")
        plan = content_v2._validate_stage_plan_self(stage_plan)
        preflight = content_v2.validate_preflight_absence_receipt(plan, preflight_receipt)
        owned = content_v2.validate_stage_receipt(plan, preflight, stage_receipt)
        intent, terminal = self._begin_mutation(
            phase="cleanup-content",
            operation_key=operation_key,
            predecessor_event_sha256=stage_event_sha256,
            evidence={
                "stage_event_sha256": stage_event_sha256,
                "stage_receipt_sha256": owned["receipt_sha256"],
            },
            recorded_at_utc=observed_at_utc,
        )
        if terminal is not None:
            return terminal
        assert intent is not None
        try:
            raw = content_v2.cleanup_staged_content(
                stage_plan=plan,
                preflight_receipt=preflight,
                stage_receipt=owned,
                backend=backend,
                observed_at_utc=observed_at_utc,
            )
            receipt = content_v2.validate_cleanup_receipt(plan, preflight, owned, raw)
        except Exception:
            self._finish_mutation(
                intent=intent,
                status="failed",
                output=None,
                outcome="unknown",
                recorded_at_utc=observed_at_utc,
            )
            raise
        return self._finish_mutation(
            intent=intent,
            status="complete",
            output={"content_cleanup_receipt": receipt},
            outcome="performed" if receipt.get("delete_attempt_count", 0) else "none",
            recorded_at_utc=observed_at_utc,
        )

    def closeout(
        self,
        *,
        operation_key: str,
        launch_event_sha256: str,
        delete_event_sha256: str,
        absence_event_sha256: str,
        worker_iam_cleanup_event_sha256: str,
        recorded_at_utc: str,
        content_cleanup_event_sha256: str | None = None,
    ) -> JournalEvent:
        prior = self._terminal_for(operation_key)
        if prior is not None:
            if prior.value["phase"] != "closeout":
                raise PermissionError("operation key was consumed by another phase")
            return prior
        launch = self._event(launch_event_sha256)
        delete = self._event(delete_event_sha256)
        absence = self._event(absence_event_sha256)
        iam_cleanup = self._event(worker_iam_cleanup_event_sha256)
        if (
            launch.value["phase"] not in {"authorize-launch", "reconcile-create"}
            or launch.value["status"] not in {"complete", "partial"}
            or delete.value["phase"] not in {"cleanup-delete", "reconcile-delete"}
            or absence.value["phase"] != "cleanup-absence"
            or absence.value["predecessor_event_sha256"] != delete_event_sha256
            or iam_cleanup.value["phase"] not in {
                "worker-iam-cleanup", "worker-iam-reconcile-cleanup"
            }
            or iam_cleanup.value["status"] != "complete"
            or iam_cleanup.value.get("output", {}).get("receipt", {}).get(
                "cleanup_complete"
            ) is not True
        ):
            raise PermissionError("lifecycle closeout chain is incomplete")
        content_event = (
            None if content_cleanup_event_sha256 is None
            else self._event(content_cleanup_event_sha256)
        )
        if content_event is not None and content_event.value["phase"] != "cleanup-content":
            raise PermissionError("content cleanup event has the wrong phase")
        launch_output = launch.value["output"]
        delete_output = delete.value["output"]
        absence_output = absence.value["output"]
        if not isinstance(launch_output.get("gce_create_receipt"), Mapping):
            raise PermissionError("lifecycle closeout requires an exact create classification")
        actual_launch = launch_output.get("actual_launch_receipt")
        if actual_launch is not None and not isinstance(actual_launch, Mapping):
            raise PermissionError("lifecycle actual launch receipt shape changed")
        core = {
            "schema": LIFECYCLE_SCHEMA,
            "status": "exact_owned_gce_lifecycle_absence_attested",
            "controller_context_sha256": self.context_sha256,
            **deepcopy(self.context),
            "launch_event_sha256": launch_event_sha256,
            "gce_create_receipt_sha256": launch_output["gce_create_receipt"]["receipt_sha256"],
            "actual_launch_receipt_sha256": (
                None
                if actual_launch is None
                else actual_launch["receipt_sha256"]
            ),
            "delete_event_sha256": delete_event_sha256,
            "gce_delete_receipt_sha256": delete_output["gce_delete_receipt"]["receipt_sha256"],
            "absence_event_sha256": absence_event_sha256,
            "gce_absence_receipt_sha256": absence_output["gce_absence_receipt"]["receipt_sha256"],
            "worker_iam_cleanup_event_sha256": worker_iam_cleanup_event_sha256,
            "worker_iam_cleanup_receipt_sha256": iam_cleanup.value["output"]["receipt"][
                "receipt_sha256"
            ],
            "worker_iam_bindings_absent": True,
            "content_cleanup_event_sha256": content_cleanup_event_sha256,
            "content_cleanup_receipt_sha256": (
                None
                if content_event is None
                else content_event.value["output"]["content_cleanup_receipt"]["receipt_sha256"]
            ),
            "all_owned_instances_absent": True,
            "all_owned_boot_disks_absent": True,
            "additional_create_authorized": False,
            "attested_at_utc": _utc(recorded_at_utc, "closeout time"),
            "current_profile_changed": False,
        }
        receipt = _seal(core, "receipt_sha256")
        return self.journal.append(
            phase="closeout",
            mode="read-only",
            status="complete",
            operation_key=operation_key,
            predecessor_event_sha256=absence_event_sha256,
            evidence={
                "launch_event_sha256": launch_event_sha256,
                "delete_event_sha256": delete_event_sha256,
                "absence_event_sha256": absence_event_sha256,
                "worker_iam_cleanup_event_sha256": worker_iam_cleanup_event_sha256,
                "content_cleanup_event_sha256": content_cleanup_event_sha256,
            },
            output=receipt,
            mutation_requested=False,
            mutation_outcome="not_requested",
            recorded_at_utc=recorded_at_utc,
        )

    def validate_lifecycle_closeout_chain(
        self,
        *,
        closeout_event_sha256: str,
        launch_bundle: Mapping[str, Any],
        launch_bundle_validator: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ],
        gce_create_adapter: gce_v2.GceWavePhaseBAdapter,
        gce_delete_adapter: gce_v2.GceWavePhaseBAdapter,
        gce_absence_adapter: gce_v2.GceWavePhaseBAdapter,
        quota_receipt: Mapping[str, Any],
        persistent_claim_receipt: Mapping[str, Any],
        planned_mapping_receipt: Mapping[str, Any],
        prelaunch_authorization: Mapping[str, Any],
        worker_iam_plan: Mapping[str, Any],
        immutable_content_prefix: str,
        content_payload_sha256: str,
        outer_manifest_sha256: str,
        worker_iam_prepare_receipt: Mapping[str, Any],
        worker_iam_install_receipt: Mapping[str, Any],
        worker_iam_readback_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Revalidate one closeout from producer receipts and journal bytes.

        The lifecycle receipt alone intentionally grants no authority.  This
        method re-reads the complete write-once journal (thereby rechecking the
        contiguous event hash chain), resolves every referenced event, and
        invokes each producer-owned receipt validator again.  The returned
        proof is suitable for the receiver boundary; it contains normalized
        selected-instance lineage rather than only hashes.
        """

        if not callable(launch_bundle_validator):
            raise TypeError("launch bundle validator is not callable")
        # A single load validates filenames, canonical bytes, event digests,
        # previous-event links, context binding, and sequence contiguity.
        events = self.journal.load()
        by_sha = {event.value["event_sha256"]: event for event in events}
        if len(by_sha) != len(events):
            raise JournalTamperError("controller journal event digest was reused")

        closeout_sha = _sha(closeout_event_sha256, "closeout event")
        closeout = by_sha.get(closeout_sha)
        if closeout is None:
            raise PermissionError("lifecycle closeout event is absent")
        lifecycle_raw = closeout.value.get("output")
        if not isinstance(lifecycle_raw, Mapping):
            raise ValueError("lifecycle closeout output is not a receipt")
        lifecycle = deepcopy(dict(lifecycle_raw))
        lifecycle_digest = lifecycle.pop("receipt_sha256", None)
        if lifecycle_digest != canonical_sha256(lifecycle):
            raise ValueError("lifecycle closeout receipt digest changed")
        lifecycle = {
            **lifecycle,
            "receipt_sha256": _sha(lifecycle_digest, "lifecycle receipt"),
        }
        lifecycle_attested_at = _utc(
            lifecycle.get("attested_at_utc"), "lifecycle attested time"
        )
        if (
            closeout.value["phase"] != "closeout"
            or closeout.value["mode"] != "read-only"
            or closeout.value["status"] != "complete"
            or closeout.value["mutation_requested"] is not False
            or closeout.value["mutation_outcome"] != "not_requested"
            or lifecycle.get("schema") != LIFECYCLE_SCHEMA
            or lifecycle.get("controller_context_sha256") != self.context_sha256
            or any(lifecycle.get(key) != value for key, value in self.context.items())
            or lifecycle.get("current_profile_changed") is not False
            or closeout.value["recorded_at_utc"] != lifecycle_attested_at
        ):
            raise ValueError("lifecycle closeout controller binding changed")

        def referenced(field: str, phase: str | frozenset[str]) -> JournalEvent:
            event_sha = _sha(lifecycle.get(field), field)
            event = by_sha.get(event_sha)
            expected_phases = frozenset({phase}) if isinstance(phase, str) else phase
            if event is None or event.value["phase"] not in expected_phases:
                raise ValueError(f"lifecycle {field} event lineage changed")
            return event

        launch = referenced(
            "launch_event_sha256", frozenset({"authorize-launch", "reconcile-create"})
        )
        deleted = referenced(
            "delete_event_sha256", frozenset({"cleanup-delete", "reconcile-delete"})
        )
        absent = referenced("absence_event_sha256", "cleanup-absence")
        iam_event = referenced(
            "worker_iam_cleanup_event_sha256",
            frozenset({"worker-iam-cleanup", "worker-iam-reconcile-cleanup"}),
        )
        delete_predecessor = by_sha.get(
            deleted.value["predecessor_event_sha256"]
        )
        delete_intent = (
            delete_predecessor
            if deleted.value["phase"] == "cleanup-delete"
            else None
        )
        reconcile_failure = (
            delete_predecessor
            if deleted.value["phase"] == "reconcile-delete"
            else None
        )
        reconcile_intent = (
            None
            if reconcile_failure is None
            else by_sha.get(reconcile_failure.value["predecessor_event_sha256"])
        )
        delete_intent_source = (
            None
            if delete_intent is None
            else by_sha.get(delete_intent.value["predecessor_event_sha256"])
        )
        reconcile_intent_source = (
            None
            if reconcile_intent is None
            else by_sha.get(reconcile_intent.value["predecessor_event_sha256"])
        )

        def valid_orphan_reconcile_predecessor(
            source: JournalEvent | None, intent: JournalEvent | None,
        ) -> bool:
            if source is None or intent is None:
                return False
            output = source.value.get("output")
            receipt = (
                output.get("gce_delete_reconciliation_receipt")
                if isinstance(output, Mapping)
                else None
            )
            return (
                source.value["phase"] == "reconcile-delete"
                and source.value["status"] == "partial"
                and source.value.get("evidence", {}).get(
                    "launch_event_sha256"
                ) == launch.value["event_sha256"]
                and intent.value.get("evidence", {}).get(
                    "reconcile_event_sha256"
                ) == source.value["event_sha256"]
                and isinstance(receipt, Mapping)
                and receipt.get("orphan_cleanup_required") is True
            )
        if (
            launch.value["status"] not in {"complete", "partial"}
            or deleted.value["status"] != "complete"
            or absent.value["status"] != "complete"
            or iam_event.value["status"] != "complete"
            or (
                deleted.value["phase"] == "cleanup-delete"
                and (
                    delete_intent is None
                    or delete_intent.value["phase"] != "cleanup-delete"
                    or delete_intent.value["status"] != "intent"
                    or (
                        delete_intent.value["predecessor_event_sha256"]
                        != launch.value["event_sha256"]
                        and not valid_orphan_reconcile_predecessor(
                            delete_intent_source, delete_intent
                        )
                    )
                )
            )
            or (
                deleted.value["phase"] == "reconcile-delete"
                and (
                    reconcile_failure is None
                    or reconcile_failure.value["phase"] != "cleanup-delete"
                    or reconcile_failure.value["status"] != "failed"
                    or reconcile_failure.value.get("output", {}).get("failure_kind")
                    != "transport_ambiguity"
                    or reconcile_intent is None
                    or reconcile_intent.value["phase"] != "cleanup-delete"
                    or reconcile_intent.value["status"] != "intent"
                    or (
                        reconcile_intent.value["predecessor_event_sha256"]
                        != launch.value["event_sha256"]
                        and not valid_orphan_reconcile_predecessor(
                            reconcile_intent_source, reconcile_intent
                        )
                    )
                )
            )
            or absent.value["predecessor_event_sha256"]
            != deleted.value["event_sha256"]
            or closeout.value["predecessor_event_sha256"]
            != absent.value["event_sha256"]
            or iam_event.value["sequence"] >= closeout.value["sequence"]
            or iam_event.value["recorded_at_utc"] > lifecycle_attested_at
            or absent.value["recorded_at_utc"] > lifecycle_attested_at
        ):
            raise ValueError("lifecycle event predecessor chain changed")

        launch_output = launch.value.get("output")
        delete_output = deleted.value.get("output")
        absence_output = absent.value.get("output")
        iam_output = iam_event.value.get("output")
        if not all(
            isinstance(value, Mapping)
            for value in (launch_output, delete_output, absence_output, iam_output)
        ):
            raise ValueError("lifecycle event output is missing")
        create_raw = launch_output.get("gce_create_receipt")
        actual_raw = launch_output.get("actual_launch_receipt")
        delete_raw = delete_output.get("gce_delete_receipt")
        absence_raw = absence_output.get("gce_absence_receipt")
        iam_raw = iam_output.get("receipt")
        if not all(
            isinstance(value, Mapping)
            for value in (create_raw, delete_raw, absence_raw, iam_raw)
        ):
            raise ValueError("lifecycle producer receipt is missing")
        complete_authorized_launch = (
            launch.value["phase"] == "authorize-launch"
            and launch.value["status"] == "complete"
        )
        if complete_authorized_launch is not isinstance(actual_raw, Mapping):
            raise ValueError("actual launch presence differs from launch outcome")
        if actual_raw is not None and not isinstance(actual_raw, Mapping):
            raise ValueError("actual launch receipt shape changed")

        bundle = deepcopy(dict(launch_bundle_validator(launch_bundle)))
        create_receipt = gce_v2.validate_create_receipt(
            gce_create_adapter, create_raw  # type: ignore[arg-type]
        )
        delete_receipt = gce_v2.validate_delete_receipt(
            gce_delete_adapter, delete_raw  # type: ignore[arg-type]
        )
        absence_receipt = gce_v2.validate_absence_receipt(
            gce_absence_adapter, absence_raw  # type: ignore[arg-type]
        )
        mapping = cloud_v2.validate_planned_launch_mapping_receipt(
            self.wave_plan,
            self.attempt_ledger,
            self.resume_plan,
            planned_mapping_receipt,
            immutable_content_sha256=bundle["immutable_content_sha256"],
        )
        actual_receipt = (
            None
            if actual_raw is None
            else cloud_v2.validate_actual_launch_receipt(
                self.wave_plan,
                self.attempt_ledger,
                self.resume_plan,
                immutable_content_sha256=bundle["immutable_content_sha256"],
                quota_receipt=quota_receipt,
                persistent_claim_receipt=persistent_claim_receipt,
                planned_mapping_receipt=mapping,
                prelaunch_authorization=prelaunch_authorization,
                value=actual_raw,
            )
        )
        iam_receipt = worker_iam_v2.validate_cleanup_receipt(
            iam_plan=worker_iam_plan,
            wave_plan=self.wave_plan,
            attempt_ledger=self.attempt_ledger,
            resume_plan=self.resume_plan,
            immutable_content_prefix=immutable_content_prefix,
            content_payload_sha256=content_payload_sha256,
            outer_manifest_sha256=outer_manifest_sha256,
            prepare_receipt=worker_iam_prepare_receipt,
            install_receipt=worker_iam_install_receipt,
            readback_receipt=worker_iam_readback_receipt,
            value=iam_raw,  # type: ignore[arg-type]
        )
        absence_observed_at = _utc(
            absence_receipt.get("observed_at_utc"), "GCE absence observed time"
        )

        if (
            absence_receipt.get("all_instances_absent") is not True
            or absence_receipt.get("all_boot_disks_absent") is not True
            or absence_observed_at > lifecycle_attested_at
            or iam_receipt.get("cleanup_complete") is not True
            or lifecycle.get("gce_create_receipt_sha256")
            != create_receipt["receipt_sha256"]
            or lifecycle.get("actual_launch_receipt_sha256")
            != (
                None
                if actual_receipt is None
                else actual_receipt["receipt_sha256"]
            )
            or lifecycle.get("gce_delete_receipt_sha256")
            != delete_receipt["receipt_sha256"]
            or lifecycle.get("gce_absence_receipt_sha256")
            != absence_receipt["receipt_sha256"]
            or lifecycle.get("worker_iam_cleanup_receipt_sha256")
            != iam_receipt["receipt_sha256"]
            or lifecycle.get("worker_iam_bindings_absent") is not True
            or lifecycle.get("all_owned_instances_absent") is not True
            or lifecycle.get("all_owned_boot_disks_absent") is not True
            or lifecycle.get("additional_create_authorized") is not False
        ):
            raise ValueError("lifecycle receipt differs from producer receipts")

        selected = self.resume_plan["selected_attempts"]
        planned_by_job = {row["job_id"]: row for row in mapping["rows"]}
        create_by_job = {row["job_id"]: row for row in create_receipt["rows"]}
        actual_by_job = (
            {}
            if actual_receipt is None
            else {row["job_id"]: row for row in actual_receipt["rows"]}
        )
        selected_jobs = {row["job_id"] for row in selected}
        if (
            set(planned_by_job) != selected_jobs
            or not set(create_by_job).issubset(selected_jobs)
            or (actual_receipt is not None and set(actual_by_job) != selected_jobs)
            or len(create_by_job) != len(create_receipt["rows"])
        ):
            raise ValueError("lifecycle selected job coverage changed")
        selected_names = [row["instance_id"] for row in selected]
        if (
            absence_receipt.get("absent_instance_names") != selected_names
            or absence_receipt.get("absent_boot_disk_names") != selected_names
            or absence_receipt.get("checked_instance_count") != len(selected_names)
        ):
            raise ValueError("lifecycle final selected-name absence changed")
        selected_mapping: list[dict[str, Any]] = []
        for row in selected:
            job_id = row["job_id"]
            planned = planned_by_job[job_id]
            created = create_by_job.get(job_id)
            actual = actual_by_job.get(job_id)
            if planned["instance_id"] != row["instance_id"]:
                raise ValueError("lifecycle selected instance mapping changed")
            if created is not None and (
                created["source_role"] != row["source_role"]
                or created["attempt_id"] != row["attempt_id"]
                or created["instance_name"] != row["instance_id"]
            ):
                raise ValueError("lifecycle exact-created subset changed")
            if actual is not None and (
                actual["source_role"] != row["source_role"]
                or actual["attempt_id"] != row["attempt_id"]
                or actual["instance_id"] != row["instance_id"]
            ):
                raise ValueError("lifecycle actual launch mapping changed")
            exact_created = created is not None
            attempt_launch_identity = canonical_sha256(
                {
                    "schema": "hu_m31_t3_step6d_full100_attempt_launch_identity_v2",
                    "controller_context_sha256": self.context_sha256,
                    "launch_event_sha256": launch.value["event_sha256"],
                    "gce_create_receipt_sha256": create_receipt["receipt_sha256"],
                    "job_id": job_id,
                    "source_role": row["source_role"],
                    "attempt_id": row["attempt_id"],
                    "instance_id": row["instance_id"],
                    "exact_instance_created": exact_created,
                }
            )
            selected_mapping.append(
                {
                    "job_id": job_id,
                    "source_role": row["source_role"],
                    "attempt_id": row["attempt_id"],
                    "instance_id": row["instance_id"],
                    "artifact_prefix": row["artifact_prefix"],
                    "launch_receipt_sha256": attempt_launch_identity,
                    "exact_instance_created": exact_created,
                    "provider_instance_id": (
                        None if created is None else created["provider_instance_id"]
                    ),
                    "provider_boot_disk_id": (
                        None if created is None else created["provider_boot_disk_id"]
                    ),
                    "gce_spec_sha256": (
                        None if created is None else created["spec_sha256"]
                    ),
                    "gce_operation_id": (
                        None if created is None else created["operation_id"]
                    ),
                    "actual_launch_operation_id": (
                        None if actual is None else actual["operation_id"]
                    ),
                    "actual_launch_instance_status": (
                        None if actual is None else actual["instance_status"]
                    ),
                    "ownership_label": planned["ownership_label"],
                    "final_instance_absent": True,
                    "final_boot_disk_absent": True,
                }
            )

        created_count = len(create_by_job)
        selected_count = len(selected)
        classification = (
            "all_selected_created"
            if created_count == selected_count
            else "no_selected_created"
            if created_count == 0
            else "partial_selected_created"
        )

        proof_core = {
            "schema": LIFECYCLE_PROOF_SCHEMA,
            "status": "controller_journal_and_all_producer_receipts_revalidated",
            "controller_context_sha256": self.context_sha256,
            **deepcopy(self.context),
            "lifecycle_event_sha256": closeout.value["event_sha256"],
            "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
            "lifecycle_attested_at_utc": lifecycle_attested_at,
            "launch_event_sha256": launch.value["event_sha256"],
            "delete_event_sha256": deleted.value["event_sha256"],
            "absence_event_sha256": absent.value["event_sha256"],
            "worker_iam_cleanup_event_sha256": iam_event.value["event_sha256"],
            "launch_bundle_sha256": bundle["bundle_sha256"],
            "gce_create_receipt": create_receipt,
            "actual_launch_receipt": actual_receipt,
            "gce_delete_receipt": delete_receipt,
            "gce_absence_receipt": absence_receipt,
            "worker_iam_cleanup_receipt": iam_receipt,
            "selected_instance_mapping": selected_mapping,
            "gce_create_rows": deepcopy(create_receipt["rows"]),
            "actual_launch_rows": (
                [] if actual_receipt is None else deepcopy(actual_receipt["rows"])
            ),
            "selected_instance_count": selected_count,
            "exact_created_instance_count": created_count,
            "exact_uncreated_instance_count": selected_count - created_count,
            "create_classification": classification,
            "actual_launch_receipt_present": actual_receipt is not None,
            "journal_event_count": len(events),
            "journal_hash_chain_valid": True,
            "all_producer_receipts_valid": True,
            "all_owned_instances_absent": True,
            "all_owned_boot_disks_absent": True,
            "worker_iam_bindings_absent": True,
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
        return _seal(proof_core, "proof_sha256")


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


_LAUNCH_VALIDATION_KEYS = frozenset(
    {
        "outer_manifest", "quota_receipt", "persistent_claim_receipt",
        "planned_mapping_receipt", "prelaunch_authorization", "raw_claim_nonce",
        "current_time_utc", "runtime_preflight_receipt", "gcp_read_receipt",
        "runtime_gcp_read_receipt",
        "service_account_actas_receipt", "worker_identity_plan",
        "worker_identity_inventory_receipt", "worker_identity_act_as_receipt",
        "project_iam_scan_receipt", "worker_iam_plan",
        "worker_iam_prepare_receipt", "worker_iam_install_receipt",
        "worker_iam_readback_receipt",
    }
)


def _exact_request(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields changed")
    # Strict cloning also prevents credential-like custom objects from crossing
    # the controller boundary.  Mode schemas intentionally have no token key.
    return _json_clone(dict(value), label)


def _launch_material(
    controller: Full100WaveControllerV2,
    request: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Callable[[Mapping[str, Any]], Mapping[str, Any]],
    dict[str, Any],
    bytes,
]:
    validation_raw = request.get("launch_validation")
    validation = _exact_request(
        validation_raw, _LAUNCH_VALIDATION_KEYS, "launch validation evidence"  # type: ignore[arg-type]
    )

    def validator(value: Mapping[str, Any]) -> Mapping[str, Any]:
        return bundle_v2.validate_launch_bundle(
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            expected_startup_sha256=science_registry.resolve_startup_sha256(
                controller.wave_plan
            ),
            value=value,
            **validation,
        )

    launch_bundle = deepcopy(dict(validator(request.get("launch_bundle"))))  # type: ignore[arg-type]
    runtime = runtime_v2.validate_runtime_preflight_receipt(
        wave_plan=controller.wave_plan,
        value=validation["runtime_preflight_receipt"],
        current_utc=validation["current_time_utc"],
    )
    startup_path = Path(str(request.get("startup_script_path", "")))
    if not startup_path.is_file() or startup_path.is_symlink():
        raise ValueError("startup script path is not a plain existing file")
    startup_bytes = startup_path.read_bytes()
    expected_startup_sha256 = science_registry.resolve_startup_sha256(
        controller.wave_plan
    )
    if hashlib.sha256(startup_bytes).hexdigest() != expected_startup_sha256:
        raise ValueError("startup script bytes differ from frozen launch hash")
    image = runtime["image"]
    return launch_bundle, validator, validation, startup_bytes


def _gce_adapter(
    *,
    controller: Full100WaveControllerV2,
    mode: str,
    request: Mapping[str, Any],
    requester: Callable[..., Any],
    create_receipt: Mapping[str, Any] | None = None,
    delete_receipt: Mapping[str, Any] | None = None,
) -> tuple[gce_v2.GceWavePhaseBAdapter, dict[str, Any], dict[str, Any]]:
    launch_bundle, validator, validation, startup_bytes = _launch_material(
        controller, request
    )
    image = validation["runtime_preflight_receipt"]["image"]
    adapter = gce_v2.GceWavePhaseBAdapter(
        mode=mode,
        launch_bundle=launch_bundle,
        launch_bundle_validator=validator,
        active_image_self_link=image["self_link"],
        active_image_identity_sha256=image["image_identity_sha256"],
        expected_image_digest=controller.wave_plan["runtime_binding"]["image_digest"],
        startup_script_bytes=startup_bytes,
        expected_startup_sha256=science_registry.resolve_startup_sha256(
            controller.wave_plan
        ),
        requester=requester,
        create_receipt=create_receipt,
        delete_receipt=delete_receipt,
    )
    return adapter, launch_bundle, validation


def _launch_create_receipt(
    controller: Full100WaveControllerV2, event_sha256: str
) -> dict[str, Any]:
    event = controller._event(event_sha256)
    value = event.value.get("output")
    receipt = value.get("gce_create_receipt") if isinstance(value, Mapping) else None
    if (
        event.value["phase"] not in {"authorize-launch", "reconcile-create"}
        or event.value["status"] not in {"complete", "partial"}
        or not isinstance(receipt, Mapping)
    ):
        raise PermissionError("launch event lacks an exact owned GCE create receipt")
    return deepcopy(dict(receipt))


def _delete_receipt(
    controller: Full100WaveControllerV2, event_sha256: str
) -> dict[str, Any]:
    event = controller._event(event_sha256)
    value = event.value.get("output")
    receipt = value.get("gce_delete_receipt") if isinstance(value, Mapping) else None
    if (
        event.value["phase"] not in {"cleanup-delete", "reconcile-delete"}
        or event.value["status"] != "complete"
        or not isinstance(receipt, Mapping)
    ):
        raise PermissionError("cleanup event lacks an exact GCE delete receipt")
    return deepcopy(dict(receipt))


_CONTENT_BINDING_KEYS = frozenset(
    {"immutable_content_prefix", "content_payload_sha256", "outer_manifest_sha256"}
)


def _content_binding(value: Any) -> dict[str, str]:
    checked = _exact_request(value, _CONTENT_BINDING_KEYS, "content binding")
    for key in _CONTENT_BINDING_KEYS:
        if not isinstance(checked[key], str):
            raise ValueError("content binding value is not a string")
    return checked  # type: ignore[return-value]


def _phase_a_adapter(
    *,
    controller: Full100WaveControllerV2,
    mode: str,
    content: Mapping[str, str],
    requester: Callable[..., Any],
    iam_plan: Mapping[str, Any] | None = None,
    gcp_read_receipt: Mapping[str, Any] | None = None,
    prepare_receipt: Mapping[str, Any] | None = None,
    install_receipt: Mapping[str, Any] | None = None,
    readback_receipt: Mapping[str, Any] | None = None,
) -> gcp_v2.GcpWavePhaseAAdapter:
    return gcp_v2.GcpWavePhaseAAdapter(
        mode=mode,
        project=gcp_v2.PROJECT,
        region=gcp_v2.REGION,
        zone=gcp_v2.ZONE,
        bucket=gcp_v2.BUCKET,
        wave_plan=controller.wave_plan,
        attempt_ledger=controller.attempt_ledger,
        resume_plan=controller.resume_plan,
        immutable_content_prefix=content["immutable_content_prefix"],
        content_payload_sha256=content["content_payload_sha256"],
        outer_manifest_sha256=content["outer_manifest_sha256"],
        iam_plan=iam_plan,
        gcp_read_receipt=gcp_read_receipt,
        prepare_receipt=prepare_receipt,
        install_receipt=install_receipt,
        readback_receipt=readback_receipt,
        requester=requester,
    )


def execute_mode_request(
    *,
    controller: Full100WaveControllerV2,
    mode: str,
    request: Mapping[str, Any],
    requester: Callable[..., Any] = _stdlib_http_request,
    allow_cloud_read: bool = False,
    allow_identity_create: bool = False,
    allow_content_stage: bool = False,
    allow_gce_create: bool = False,
    allow_gce_delete: bool = False,
    allow_content_delete: bool = False,
    allow_claim_create: bool = False,
    allow_worker_iam_install: bool = False,
    allow_launch_authorization: bool = False,
) -> JournalEvent:
    """Dispatch one exact request envelope through official v2 adapters.

    The injected requester is used only by tests or by the caller-selected live
    boundary.  Credentials are still read by each adapter from the environment
    for every HTTP request and are never accepted in this envelope.
    """

    if not callable(requester):
        raise TypeError("controller HTTP requester is not callable")
    if mode == "runtime-preflight":
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256",
                    "image_observation", "machine_type_observation",
                    "network_observation", "subnetwork_observation",
                    "cloud_nat_observation", "bucket_observation",
                    "observed_at_utc", "current_utc",
                }
            ),
            "runtime preflight request",
        )
        receipt = runtime_v2.build_runtime_preflight_receipt(
            wave_plan=controller.wave_plan,
            image_observation=value["image_observation"],
            machine_type_observation=value["machine_type_observation"],
            network_observation=value["network_observation"],
            subnetwork_observation=value["subnetwork_observation"],
            cloud_nat_observation=value["cloud_nat_observation"],
            bucket_observation=value["bucket_observation"],
            observed_at_utc=value["observed_at_utc"],
            current_utc=value["current_utc"],
        )
        return controller.record_read_phase(
            phase="runtime-preflight",
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"runtime_preflight_receipt_sha256": receipt["receipt_sha256"]},
            output={"runtime_preflight_receipt": receipt},
            recorded_at_utc=value["current_utc"],
        )

    if mode == "runtime-gcp-read":
        if allow_cloud_read is not True:
            raise PermissionError("runtime-gcp-read requires allow_cloud_read")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key",
                    "predecessor_event_sha256",
                    "observed_at_utc",
                    "current_utc",
                }
            ),
            "runtime GCP read request",
        )
        adapter = runtime_gcp_v2.RuntimeGcpReadAdapterV2(
            wave_plan=controller.wave_plan,
            requester=requester,
        )
        receipt = adapter.read(
            observed_at_utc=value["observed_at_utc"],
            current_utc=value["current_utc"],
        )
        runtime = receipt["runtime_preflight_receipt"]
        return controller.record_read_phase(
            phase="runtime-gcp-read",
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={
                "runtime_gcp_read_receipt_sha256": receipt["receipt_sha256"],
                "runtime_preflight_receipt_sha256": runtime["receipt_sha256"],
            },
            output={
                "runtime_gcp_read_receipt": receipt,
                "runtime_preflight_receipt": runtime,
            },
            recorded_at_utc=value["current_utc"],
        )

    if mode == "identity-read":
        if allow_cloud_read is not True:
            raise PermissionError("identity-read requires allow_cloud_read")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "identity_plan",
                    "setup_plan", "observed_at_utc", "expires_at_utc",
                }
            ),
            "identity-read request",
        )
        adapter = identity_gcp_v2.WorkerIdentityGcpAdapterV2(
            mode="read",
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            identity_plan=value["identity_plan"],
            setup_plan=value["setup_plan"],
            requester=requester,
        )
        receipt = adapter.read_pool(
            observed_at_utc=value["observed_at_utc"],
            expires_at_utc=value["expires_at_utc"],
        )
        return controller.record_read_phase(
            phase="identity-read",
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"identity_plan_sha256": value["identity_plan"]["plan_sha256"]},
            output={"identity_read_receipt": receipt},
            recorded_at_utc=value["observed_at_utc"],
        )

    if mode in {"identity-actas", "project-iam-scan"}:
        if allow_cloud_read is not True:
            raise PermissionError(f"{mode} requires allow_cloud_read")
        time_field = "tested_at_utc" if mode == "identity-actas" else "observed_at_utc"
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "identity_plan",
                    "inventory_receipt", time_field,
                }
            ),
            f"{mode} request",
        )
        adapter = identity_gcp_v2.WorkerIdentityGcpAdapterV2(
            mode="actas-check" if mode == "identity-actas" else "project-iam-scan",
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            identity_plan=value["identity_plan"],
            inventory_receipt=value["inventory_receipt"],
            requester=requester,
        )
        receipt = (
            adapter.check_service_account_act_as(tested_at_utc=value[time_field])
            if mode == "identity-actas"
            else adapter.scan_project_iam(observed_at_utc=value[time_field])
        )
        output_key = (
            "worker_identity_act_as_receipt"
            if mode == "identity-actas"
            else "project_iam_scan_receipt"
        )
        return controller.record_read_phase(
            phase=mode,
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={
                "identity_plan_sha256": value["identity_plan"]["plan_sha256"],
                "inventory_receipt_sha256": value["inventory_receipt"]["receipt_sha256"],
            },
            output={output_key: receipt},
            recorded_at_utc=value[time_field],
        )

    if mode == "content-prefix-preflight":
        if allow_cloud_read is not True:
            raise PermissionError("content-prefix-preflight requires allow_cloud_read")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "package_dir",
                    "stage_plan", "observed_at_utc",
                }
            ),
            "content prefix preflight request",
        )
        backend = content_gcp_v2.GcsContentObjectAdapter(
            mode="preflight", stage_plan=value["stage_plan"], requester=requester
        )
        receipt = content_v2.build_preflight_absence_receipt(
            package_dir=value["package_dir"],
            wave_plan=controller.wave_plan,
            expected_startup_sha256=science_registry.resolve_startup_sha256(
                controller.wave_plan
            ),
            stage_plan=value["stage_plan"],
            backend=backend,
            observed_at_utc=value["observed_at_utc"],
        )
        return controller.record_read_phase(
            phase="content-prefix-preflight",
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"stage_plan_sha256": value["stage_plan"]["plan_sha256"]},
            output={"content_preflight_receipt": receipt},
            recorded_at_utc=value["observed_at_utc"],
        )

    if mode == "worker-iam-plan":
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "content_binding",
                    "issued_at_unix_seconds", "service_accounts_by_job",
                }
            ),
            "worker IAM plan request",
        )
        content = _content_binding(value["content_binding"])
        receipt = worker_iam_v2.build_worker_iam_plan(
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            wave_index=controller.resume_plan["resume_wave_index"],
            issued_at_unix_seconds=value["issued_at_unix_seconds"],
            service_accounts_by_job=value["service_accounts_by_job"],
            **content,
        )
        return controller.record_read_phase(
            phase="worker-iam-plan",
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"content_binding_sha256": canonical_sha256(content)},
            output={"worker_iam_plan": receipt},
            recorded_at_utc=receipt["issued_at_utc"],
        )

    if mode in {"phasea-read", "provider-actas-check"}:
        if allow_cloud_read is not True:
            raise PermissionError(f"{mode} requires allow_cloud_read")
        extra = (
            {"observed_at_utc", "expires_at_utc"}
            if mode == "phasea-read"
            else {"gcp_read_receipt", "checked_at_utc", "expires_at_utc"}
        )
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "content_binding",
                    "iam_plan", *extra,
                }
            ),
            f"{mode} request",
        )
        content = _content_binding(value["content_binding"])
        adapter = _phase_a_adapter(
            controller=controller,
            mode="read" if mode == "phasea-read" else "actas-check",
            content=content,
            requester=requester,
            iam_plan=value["iam_plan"],
            gcp_read_receipt=value.get("gcp_read_receipt"),
        )
        receipt = (
            adapter.read_prelaunch(
                observed_at_utc=value["observed_at_utc"],
                expires_at_utc=value["expires_at_utc"],
            )
            if mode == "phasea-read"
            else adapter.check_service_account_act_as(
                checked_at_utc=value["checked_at_utc"],
                expires_at_utc=value["expires_at_utc"],
            )
        )
        output_key = (
            "gcp_read_receipt" if mode == "phasea-read"
            else "service_account_actas_receipt"
        )
        observed = (
            value["observed_at_utc"] if mode == "phasea-read" else value["checked_at_utc"]
        )
        return controller.record_read_phase(
            phase=mode,
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"worker_iam_plan_sha256": value["iam_plan"]["plan_sha256"]},
            output={output_key: receipt},
            recorded_at_utc=observed,
        )

    if mode == "persistent-claim":
        if allow_claim_create is not True:
            raise PermissionError("persistent-claim requires allow_claim_create")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "content_binding",
                    "claim_nonce", "claimed_at_utc",
                }
            ),
            "persistent claim request",
        )
        content = _content_binding(value["content_binding"])
        adapter = _phase_a_adapter(
            controller=controller,
            mode="claim",
            content=content,
            requester=requester,
        )

        def create_claim() -> Mapping[str, Any]:
            return cloud_v2.create_persistent_atomic_launch_claim(
                controller.wave_plan,
                controller.attempt_ledger,
                controller.resume_plan,
                immutable_content_sha256=content["content_payload_sha256"],
                project_id=gcp_v2.PROJECT,
                zone=gcp_v2.ZONE,
                claim_nonce=value["claim_nonce"],
                claimed_at_utc=value["claimed_at_utc"],
                backend=adapter,
            )

        return controller.run_mutation_phase(
            phase="persistent-claim",
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={
                "content_binding_sha256": canonical_sha256(content),
                "claim_nonce_sha256": hashlib.sha256(
                    value["claim_nonce"].encode("ascii")
                ).hexdigest(),
            },
            action=create_claim,
            validator=lambda receipt: cloud_v2.validate_persistent_atomic_launch_claim_receipt(
                controller.wave_plan,
                controller.attempt_ledger,
                controller.resume_plan,
                receipt,
                immutable_content_sha256=content["content_payload_sha256"],
                raw_claim_nonce=value["claim_nonce"],
            ),
            started_at_utc=value["claimed_at_utc"],
            completed_at_utc=value["claimed_at_utc"],
        )

    if mode in {
        "worker-iam-prepare", "worker-iam-install", "worker-iam-readback",
        "worker-iam-cleanup", "worker-iam-reconcile-install",
        "worker-iam-reconcile-cleanup",
    }:
        mutating = mode in {"worker-iam-install", "worker-iam-cleanup"}
        if mutating and allow_worker_iam_install is not True:
            raise PermissionError(f"{mode} requires allow_worker_iam_install")
        if not mutating and allow_cloud_read is not True:
            raise PermissionError(f"{mode} requires allow_cloud_read")
        extra: set[str] = set()
        if mode in {
            "worker-iam-install", "worker-iam-readback", "worker-iam-cleanup",
            "worker-iam-reconcile-install", "worker-iam-reconcile-cleanup",
        }:
            extra.add("prepare_receipt")
        if mode in {
            "worker-iam-readback", "worker-iam-cleanup",
            "worker-iam-reconcile-cleanup",
        }:
            extra.add("install_receipt")
        if mode in {"worker-iam-cleanup", "worker-iam-reconcile-cleanup"}:
            extra.add("readback_receipt")
        if mode in {
            "worker-iam-reconcile-install", "worker-iam-reconcile-cleanup"
        }:
            extra.add("source_operation_key")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256", "content_binding",
                    "iam_plan", "observed_at_utc", *extra,
                }
            ),
            f"{mode} request",
        )
        content = _content_binding(value["content_binding"])
        adapter_mode = {
            "worker-iam-prepare": "bucket-iam-prepare",
            "worker-iam-install": "bucket-iam-install",
            "worker-iam-readback": "bucket-iam-readback",
            "worker-iam-cleanup": "bucket-iam-cleanup",
            "worker-iam-reconcile-install": "bucket-iam-reconcile-install",
            "worker-iam-reconcile-cleanup": "bucket-iam-reconcile-cleanup",
        }[mode]
        adapter = _phase_a_adapter(
            controller=controller,
            mode=adapter_mode,
            content=content,
            requester=requester,
            iam_plan=value["iam_plan"],
            prepare_receipt=value.get("prepare_receipt"),
            install_receipt=value.get("install_receipt"),
            readback_receipt=value.get("readback_receipt"),
        )
        common = {
            "iam_plan": value["iam_plan"],
            "wave_plan": controller.wave_plan,
            "attempt_ledger": controller.attempt_ledger,
            "resume_plan": controller.resume_plan,
            **content,
            "backend": adapter,
        }
        if mode == "worker-iam-prepare":
            receipt = worker_iam_v2.prepare_worker_iam(**common)
            return controller.record_read_phase(
                phase=mode,
                operation_key=value["operation_key"],
                predecessor_event_sha256=value["predecessor_event_sha256"],
                evidence={"worker_iam_plan_sha256": value["iam_plan"]["plan_sha256"]},
                output={"worker_iam_prepare_receipt": receipt},
                recorded_at_utc=value["observed_at_utc"],
            )
        if mode == "worker-iam-readback":
            receipt = worker_iam_v2.readback_worker_iam(
                **common,
                prepare_receipt=value["prepare_receipt"],
                install_receipt=value["install_receipt"],
            )
            return controller.record_read_phase(
                phase=mode,
                operation_key=value["operation_key"],
                predecessor_event_sha256=value["predecessor_event_sha256"],
                evidence={
                    "install_receipt_sha256": value["install_receipt"]["receipt_sha256"]
                },
                output={"worker_iam_readback_receipt": receipt},
                recorded_at_utc=value["observed_at_utc"],
            )
        if mode == "worker-iam-reconcile-install":
            receipt = worker_iam_v2.reconcile_install_worker_iam(
                **common, prepare_receipt=value["prepare_receipt"]
            )
            return controller.record_ambiguous_mutation_reconciliation(
                phase=mode,
                operation_key=value["operation_key"],
                source_operation_key=value["source_operation_key"],
                receipt=receipt,
                output_key="receipt",
                recorded_at_utc=value["observed_at_utc"],
            )
        if mode == "worker-iam-reconcile-cleanup":
            receipt = worker_iam_v2.reconcile_cleanup_worker_iam(
                **common,
                prepare_receipt=value["prepare_receipt"],
                install_receipt=value["install_receipt"],
                readback_receipt=value["readback_receipt"],
            )
            return controller.record_ambiguous_mutation_reconciliation(
                phase=mode,
                operation_key=value["operation_key"],
                source_operation_key=value["source_operation_key"],
                receipt=receipt,
                output_key="receipt",
                recorded_at_utc=value["observed_at_utc"],
            )
        if mode == "worker-iam-install":
            action = lambda: worker_iam_v2.install_worker_iam(
                **common, prepare_receipt=value["prepare_receipt"]
            )
            validator = lambda receipt: worker_iam_v2.validate_install_receipt(
                iam_plan=value["iam_plan"],
                wave_plan=controller.wave_plan,
                attempt_ledger=controller.attempt_ledger,
                resume_plan=controller.resume_plan,
                prepare_receipt=value["prepare_receipt"],
                value=receipt,
                **content,
            )
        else:
            action = lambda: worker_iam_v2.cleanup_worker_iam(
                **common,
                prepare_receipt=value["prepare_receipt"],
                install_receipt=value["install_receipt"],
                readback_receipt=value["readback_receipt"],
            )
            validator = lambda receipt: worker_iam_v2.validate_cleanup_receipt(
                iam_plan=value["iam_plan"],
                wave_plan=controller.wave_plan,
                attempt_ledger=controller.attempt_ledger,
                resume_plan=controller.resume_plan,
                prepare_receipt=value["prepare_receipt"],
                install_receipt=value["install_receipt"],
                readback_receipt=value["readback_receipt"],
                value=receipt,
                **content,
            )
        return controller.run_mutation_phase(
            phase=mode,
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"worker_iam_plan_sha256": value["iam_plan"]["plan_sha256"]},
            action=action,
            validator=validator,
            started_at_utc=value["observed_at_utc"],
            completed_at_utc=value["observed_at_utc"],
        )

    if mode == "prelaunch-authorization":
        if allow_launch_authorization is not True:
            raise PermissionError(
                "prelaunch-authorization requires allow_launch_authorization"
            )
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "predecessor_event_sha256",
                    "immutable_content_sha256", "quota_receipt",
                    "persistent_claim_receipt", "planned_mapping_receipt",
                    "raw_claim_nonce", "authorized_at_utc", "expires_at_utc",
                }
            ),
            "prelaunch authorization request",
        )
        receipt = cloud_v2.build_prelaunch_authorization(
            controller.wave_plan,
            controller.attempt_ledger,
            controller.resume_plan,
            immutable_content_sha256=value["immutable_content_sha256"],
            quota_receipt=value["quota_receipt"],
            persistent_claim_receipt=value["persistent_claim_receipt"],
            planned_mapping_receipt=value["planned_mapping_receipt"],
            raw_claim_nonce=value["raw_claim_nonce"],
            authorized_at_utc=value["authorized_at_utc"],
            expires_at_utc=value["expires_at_utc"],
            explicit_launch_authorized=True,
        )
        return controller.record_read_phase(
            phase=mode,
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={
                "quota_receipt_sha256": value["quota_receipt"]["receipt_sha256"],
                "claim_receipt_sha256": value["persistent_claim_receipt"]["receipt_sha256"],
                "mapping_receipt_sha256": value["planned_mapping_receipt"]["receipt_sha256"],
            },
            output={"prelaunch_authorization": receipt},
            recorded_at_utc=value["authorized_at_utc"],
        )

    if mode == "launch-bundle-build":
        if allow_launch_authorization is not True:
            raise PermissionError("launch-bundle-build requires allow_launch_authorization")
        value = _exact_request(
            request,
            frozenset(
                {"operation_key", "predecessor_event_sha256", "launch_validation"}
            ),
            "launch bundle build request",
        )
        validation = _exact_request(
            value["launch_validation"],
            _LAUNCH_VALIDATION_KEYS,
            "launch validation evidence",
        )
        receipt = bundle_v2.build_launch_bundle(
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            expected_startup_sha256=science_registry.resolve_startup_sha256(
                controller.wave_plan
            ),
            **validation,
        )
        receipt = bundle_v2.validate_launch_bundle(
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            expected_startup_sha256=science_registry.resolve_startup_sha256(
                controller.wave_plan
            ),
            value=receipt,
            **validation,
        )
        return controller.record_read_phase(
            phase=mode,
            operation_key=value["operation_key"],
            predecessor_event_sha256=value["predecessor_event_sha256"],
            evidence={"launch_evidence_sha256": canonical_sha256(validation)},
            output={"launch_bundle": receipt},
            recorded_at_utc=validation["current_time_utc"],
        )
    if mode == "prepare":
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "outer_manifest", "stage_plan",
                    "content_preflight_receipt", "worker_identity_plan",
                    "runtime_preflight_receipt", "runtime_gcp_read_receipt",
                    "current_time_utc",
                }
            ),
            "prepare request",
        )
        return controller.prepare(**value)

    if mode == "setup-identities":
        if allow_identity_create is not True:
            raise PermissionError("setup-identities requires allow_identity_create")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "prepare_event_sha256", "identity_plan",
                    "setup_plan", "read_receipt", "current_time_utc",
                    "completed_at_utc",
                }
            ),
            "setup-identities request",
        )
        adapter = identity_gcp_v2.WorkerIdentityGcpAdapterV2(
            mode="create-missing",
            wave_plan=controller.wave_plan,
            attempt_ledger=controller.attempt_ledger,
            resume_plan=controller.resume_plan,
            identity_plan=value["identity_plan"],
            setup_plan=value["setup_plan"],
            read_receipt=value["read_receipt"],
            requester=requester,
        )
        return controller.setup_identities(
            operation_key=value["operation_key"],
            prepare_event_sha256=value["prepare_event_sha256"],
            adapter=adapter,
            current_time_utc=value["current_time_utc"],
            completed_at_utc=value["completed_at_utc"],
            allow_cloud_mutation=True,
        )

    if mode == "stage-content":
        if allow_content_stage is not True:
            raise PermissionError("stage-content requires allow_content_stage")
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "prepare_event_sha256", "package_dir",
                    "stage_plan", "preflight_receipt", "observed_at_utc",
                }
            ),
            "stage-content request",
        )
        backend = content_gcp_v2.GcsContentObjectAdapter(
            mode="stage", stage_plan=value["stage_plan"], requester=requester
        )
        return controller.stage_content(
            operation_key=value["operation_key"],
            prepare_event_sha256=value["prepare_event_sha256"],
            package_dir=value["package_dir"],
            stage_plan=value["stage_plan"],
            preflight_receipt=value["preflight_receipt"],
            backend=backend,
            observed_at_utc=value["observed_at_utc"],
            allow_cloud_mutation=True,
        )

    if mode == "bind-existing-staged-content":
        if allow_cloud_read is not True:
            raise PermissionError(
                "bind-existing-staged-content requires allow_cloud_read"
            )
        value = _exact_request(
            request,
            frozenset(
                {
                    "operation_key", "prepare_event_sha256", "stage_plan",
                    "preflight_receipt", "source_stage_receipt",
                    "observed_at_utc",
                }
            ),
            "bind-existing-staged-content request",
        )
        backend = content_gcp_v2.GcsContentObjectAdapter(
            mode="readback", stage_plan=value["stage_plan"], requester=requester
        )
        return controller.bind_existing_staged_content(
            operation_key=value["operation_key"],
            prepare_event_sha256=value["prepare_event_sha256"],
            stage_plan=value["stage_plan"],
            preflight_receipt=value["preflight_receipt"],
            source_stage_receipt=value["source_stage_receipt"],
            backend=backend,
            observed_at_utc=value["observed_at_utc"],
        )

    launch_common = frozenset(
        {"launch_bundle", "launch_validation", "startup_script_path"}
    )
    if mode == "authorize-launch":
        if allow_gce_create is not True:
            raise PermissionError("authorize-launch requires allow_gce_create")
        value = _exact_request(
            request,
            launch_common
            | frozenset(
                {
                    "operation_key", "stage_event_sha256", "request_ids",
                    "launch_started_at_utc", "observed_at_utc",
                }
            ),
            "authorize-launch request",
        )
        adapter, launch_bundle, validation = _gce_adapter(
            controller=controller,
            mode="create",
            request=value,
            requester=requester,
        )
        return controller.authorize_and_launch(
            operation_key=value["operation_key"],
            stage_event_sha256=value["stage_event_sha256"],
            launch_bundle=launch_bundle,
            launch_bundle_validator=lambda candidate: bundle_v2.validate_launch_bundle(
                wave_plan=controller.wave_plan,
                attempt_ledger=controller.attempt_ledger,
                resume_plan=controller.resume_plan,
                expected_startup_sha256=science_registry.resolve_startup_sha256(
                    controller.wave_plan
                ),
                value=candidate,
                **validation,
            ),
            gce_adapter=adapter,
            request_ids=value["request_ids"],
            launch_started_at_utc=value["launch_started_at_utc"],
            observed_at_utc=value["observed_at_utc"],
            quota_receipt=validation["quota_receipt"],
            persistent_claim_receipt=validation["persistent_claim_receipt"],
            planned_mapping_receipt=validation["planned_mapping_receipt"],
            prelaunch_authorization=validation["prelaunch_authorization"],
            allow_cloud_mutation=True,
        )

    if mode == "reconcile-create":
        if allow_cloud_read is not True:
            raise PermissionError("reconcile-create requires allow_cloud_read")
        value = _exact_request(
            request,
            launch_common
            | frozenset(
                {
                    "operation_key", "source_launch_operation_key", "request_ids",
                    "observed_at_utc",
                }
            ),
            "reconcile-create request",
        )
        source = [
            event for event in controller.journal.load()
            if event.value["operation_key"] == value["source_launch_operation_key"]
            and event.value["phase"] == "authorize-launch"
            and event.value["status"] in _TERMINAL_STATUSES
        ]
        if len(source) > 1:
            raise PermissionError("launch reconciliation terminal is ambiguous")
        known = None
        if source:
            output = source[0].value.get("output")
            candidate = (
                output.get("gce_create_receipt") if isinstance(output, Mapping) else None
            )
            if isinstance(candidate, Mapping):
                known = candidate
        adapter, _, _ = _gce_adapter(
            controller=controller,
            mode="reconcile-create",
            request=value,
            requester=requester,
            create_receipt=known,
        )
        return controller.reconcile_launch_create(
            operation_key=value["operation_key"],
            source_launch_operation_key=value["source_launch_operation_key"],
            adapter=adapter,
            request_ids=value["request_ids"],
            observed_at_utc=value["observed_at_utc"],
        )

    if mode == "reconcile-delete":
        if allow_cloud_read is not True:
            raise PermissionError("reconcile-delete requires allow_cloud_read")
        value = _exact_request(
            request,
            launch_common
            | frozenset(
                {
                    "operation_key", "source_operation_key",
                    "launch_event_sha256", "request_ids", "observed_at_utc",
                }
            ),
            "reconcile-delete request",
        )
        create_receipt = _launch_create_receipt(
            controller, value["launch_event_sha256"]
        )
        adapter, _, _ = _gce_adapter(
            controller=controller,
            mode="reconcile-delete",
            request=value,
            requester=requester,
            create_receipt=create_receipt,
        )
        return controller.reconcile_delete(
            operation_key=value["operation_key"],
            source_operation_key=value["source_operation_key"],
            launch_event_sha256=value["launch_event_sha256"],
            adapter=adapter,
            request_ids=value["request_ids"],
            observed_at_utc=value["observed_at_utc"],
        )

    if mode == "status":
        if allow_cloud_read is not True:
            raise PermissionError("status requires allow_cloud_read")
        value = _exact_request(
            request,
            launch_common
            | frozenset(
                {"operation_key", "launch_event_sha256", "observed_at_utc"}
            ),
            "status request",
        )
        create_receipt = _launch_create_receipt(
            controller, value["launch_event_sha256"]
        )
        adapter, _, _ = _gce_adapter(
            controller=controller,
            mode="read-status",
            request=value,
            requester=requester,
            create_receipt=create_receipt,
        )
        return controller.status(
            operation_key=value["operation_key"],
            launch_event_sha256=value["launch_event_sha256"],
            adapter=adapter,
            observed_at_utc=value["observed_at_utc"],
        )

    if mode != "cleanup":
        raise ValueError("controller mode changed")
    if not isinstance(request, Mapping) or not isinstance(request.get("step"), str):
        raise ValueError("cleanup request step is missing")
    step = request["step"]
    if step == "delete-instances":
        if allow_gce_delete is not True:
            raise PermissionError("cleanup delete requires allow_gce_delete")
        value = _exact_request(
            request,
            launch_common
            | frozenset(
                {
                    "step", "operation_key", "launch_event_sha256",
                    "request_ids", "orphan_disk_request_ids",
                    "reconcile_event_sha256", "observed_at_utc",
                }
            ),
            "cleanup delete request",
        )
        create_receipt = _launch_create_receipt(
            controller, value["launch_event_sha256"]
        )
        adapter, _, _ = _gce_adapter(
            controller=controller,
            mode="delete",
            request=value,
            requester=requester,
            create_receipt=create_receipt,
        )
        return controller.delete_instances(
            operation_key=value["operation_key"],
            launch_event_sha256=value["launch_event_sha256"],
            adapter=adapter,
            request_ids=value["request_ids"],
            orphan_disk_request_ids=value["orphan_disk_request_ids"],
            reconcile_event_sha256=value["reconcile_event_sha256"],
            observed_at_utc=value["observed_at_utc"],
            allow_cloud_mutation=True,
        )
    if step == "verify-instance-absence":
        if allow_cloud_read is not True:
            raise PermissionError("cleanup absence requires allow_cloud_read")
        value = _exact_request(
            request,
            launch_common
            | frozenset(
                {
                    "step", "operation_key", "launch_event_sha256",
                    "delete_event_sha256", "observed_at_utc",
                }
            ),
            "cleanup absence request",
        )
        create_receipt = _launch_create_receipt(
            controller, value["launch_event_sha256"]
        )
        deleted = _delete_receipt(controller, value["delete_event_sha256"])
        adapter, _, _ = _gce_adapter(
            controller=controller,
            mode="absence",
            request=value,
            requester=requester,
            create_receipt=create_receipt,
            delete_receipt=deleted,
        )
        return controller.verify_instance_absence(
            operation_key=value["operation_key"],
            delete_event_sha256=value["delete_event_sha256"],
            adapter=adapter,
            observed_at_utc=value["observed_at_utc"],
        )
    if step == "delete-content":
        if allow_content_delete is not True:
            raise PermissionError("content cleanup requires allow_content_delete")
        value = _exact_request(
            request,
            frozenset(
                {
                    "step", "operation_key", "stage_event_sha256", "stage_plan",
                    "preflight_receipt", "observed_at_utc",
                }
            ),
            "content cleanup request",
        )
        backend = content_gcp_v2.GcsContentObjectAdapter(
            mode="cleanup", stage_plan=value["stage_plan"], requester=requester
        )
        return controller.cleanup_content(
            operation_key=value["operation_key"],
            stage_event_sha256=value["stage_event_sha256"],
            stage_plan=value["stage_plan"],
            preflight_receipt=value["preflight_receipt"],
            backend=backend,
            observed_at_utc=value["observed_at_utc"],
            allow_cloud_mutation=True,
        )
    if step == "closeout":
        value = _exact_request(
            request,
            frozenset(
                {
                    "step", "operation_key", "launch_event_sha256",
                    "delete_event_sha256", "absence_event_sha256",
                    "worker_iam_cleanup_event_sha256",
                    "content_cleanup_event_sha256", "recorded_at_utc",
                }
            ),
            "cleanup closeout request",
        )
        return controller.closeout(
            operation_key=value["operation_key"],
            launch_event_sha256=value["launch_event_sha256"],
            delete_event_sha256=value["delete_event_sha256"],
            absence_event_sha256=value["absence_event_sha256"],
            worker_iam_cleanup_event_sha256=value[
                "worker_iam_cleanup_event_sha256"
            ],
            content_cleanup_event_sha256=value["content_cleanup_event_sha256"],
            recorded_at_utc=value["recorded_at_utc"],
        )
    raise ValueError("cleanup request step changed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-dir", required=True)
    parser.add_argument("--wave-plan", required=True)
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--resume-plan", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "inspect", "runtime-preflight", "runtime-gcp-read", "identity-read",
            "setup-identities",
            "identity-actas", "project-iam-scan", "content-prefix-preflight",
            "prepare", "stage-content", "bind-existing-staged-content",
            "worker-iam-plan", "phasea-read",
            "provider-actas-check", "persistent-claim", "worker-iam-prepare",
            "worker-iam-install", "worker-iam-readback", "worker-iam-cleanup",
            "worker-iam-reconcile-install", "worker-iam-reconcile-cleanup",
            "prelaunch-authorization", "launch-bundle-build", "authorize-launch",
            "reconcile-create", "reconcile-delete",
            "status", "cleanup",
        ),
        default="inspect",
    )
    parser.add_argument("--request", help="strict JSON mode request")
    parser.add_argument("--allow-cloud-read", action="store_true")
    parser.add_argument("--allow-identity-create", action="store_true")
    parser.add_argument("--allow-content-stage", action="store_true")
    parser.add_argument("--allow-gce-create", action="store_true")
    parser.add_argument("--allow-gce-delete", action="store_true")
    parser.add_argument("--allow-content-delete", action="store_true")
    parser.add_argument("--allow-claim-create", action="store_true")
    parser.add_argument("--allow-worker-iam-install", action="store_true")
    parser.add_argument("--allow-launch-authorization", action="store_true")
    parser.add_argument("--confirm-run-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    wave_plan = _read_json(args.wave_plan, "wave plan")
    ledger = _read_json(args.attempt_ledger, "attempt ledger")
    resume = _read_json(args.resume_plan, "resume plan")
    controller = Full100WaveControllerV2(
        journal_dir=args.journal_dir,
        wave_plan=wave_plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        create_journal=args.mode != "inspect",
    )
    if args.mode == "inspect":
        print(json.dumps(controller.inspect(), sort_keys=True, indent=2))
        return 0
    if args.request is None:
        raise SystemExit("--request is required outside inspect mode")
    request = _read_json(args.request, "controller mode request")
    network_mode = args.mode not in {"prepare", "runtime-preflight", "worker-iam-plan"}
    if network_mode and args.confirm_run_name != controller.context["run_name"]:
        raise SystemExit("--confirm-run-name must exactly match the validated wave")
    try:
        event = execute_mode_request(
            controller=controller,
            mode=args.mode,
            request=request,
            requester=_stdlib_http_request,
            allow_cloud_read=args.allow_cloud_read,
            allow_identity_create=args.allow_identity_create,
            allow_content_stage=args.allow_content_stage,
            allow_gce_create=args.allow_gce_create,
            allow_gce_delete=args.allow_gce_delete,
            allow_content_delete=args.allow_content_delete,
            allow_claim_create=args.allow_claim_create,
            allow_worker_iam_install=args.allow_worker_iam_install,
            allow_launch_authorization=args.allow_launch_authorization,
        )
    except PermissionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(event.value, sort_keys=True, indent=2))
    return 0


__all__ = [
    "CONTROLLER_VERSION", "ControllerJournal", "EVENT_SCHEMA",
    "Full100WaveControllerV2", "JournalEvent", "JournalTamperError",
    "LIFECYCLE_PROOF_SCHEMA", "LIFECYCLE_SCHEMA", "PREPARE_SCHEMA",
    "PendingMutationError",
    "canonical_bytes", "canonical_sha256", "execute_mode_request", "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
