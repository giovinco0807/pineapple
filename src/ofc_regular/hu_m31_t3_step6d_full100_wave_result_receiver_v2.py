"""Fail-closed receiver for full100 wave-v2 attempt results.

The guest startup publishes only attempt-specific roots, source hands, and a
canonical ``DONE.json``.  ``DONE.json`` is the sole worker commit marker.  This
module validates a complete, generation-pinned provider snapshot before a
controller-owned ``ACCEPTED.json`` is created.  It has no cloud implementation;
callers provide a small store protocol, which keeps the contract testable
offline and prevents this module from authorizing VM operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as launch_v2
from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as gce_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_result_receiver_v2"
ACCEPTANCE_SCHEMA = "hu_m31_t3_step6d_full100_wave_job_accepted_v2"
DONE_SCHEMA = "hu_m31_t3_step6d_full100_wave_attempt_done_v2"

TERMINAL_REASONS = frozenset(
    {
        "done_observed",
        "spot_loss_before_done",
        "timeout_before_done",
        "worker_failed_before_done",
        "create_missing_before_done",
    }
)
FAILURE_REASONS = TERMINAL_REASONS - {"done_observed"}

_IAM_CLEANUP_NORMAL_STATUS = (
    "removed_exact_wave_bindings_post_readback_absent"
)
_IAM_CLEANUP_RECOVERED_STATUS = (
    "recovered_exact_wave_binding_absence_after_outcome_ambiguity"
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_ZONE = re.compile(r"^[a-z]+-[a-z]+[0-9]-[a-z]$")
_SERVICE_ACCOUNT = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,61}[a-z0-9]"
    r"\.iam\.gserviceaccount\.com$"
)

_OBJECT_RECORD_KEYS = frozenset({"path", "generation", "bytes", "sha256"})
_TERMINAL_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "terminal_reason",
    }
)
_BOOTSTRAP_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_sha256",
        "observed_transition_digest",
        "wave_index",
        "job_id",
        "source_role",
        "attempt_id",
        "instance_name",
        "artifact_prefix",
        "bucket",
        "content_prefix",
        "outer_manifest_sha256",
        "content_payload_sha256",
        "scientific_source",
        "scientific_manifest",
        "wheelhouse",
        "wheelhouse_manifest",
        "startup",
        "wave_plan",
        "job_manifest",
        "prelaunch_authorization_sha256",
        "worker_principal",
        "one_vm_one_job_one_role",
        "additional_create_authorized",
        "hidden_truth_exposed",
        "bootstrap_sha256",
    }
)
_OBJECT_BINDING_KEYS = frozenset({"object_name", "sha256", "bytes"})
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_sha256",
        "observed_transition_digest",
        "wave_index",
        "job_id",
        "source_role",
        "attempt_id",
        "package_sha256",
        "image_digest",
        "binary_sha256",
        "allocation_digest",
        "run_contract_digest",
        "root_digest",
        "done_identity_sha256",
        "content_payload_sha256",
        "outer_manifest_sha256",
        "prelaunch_authorization_sha256",
        "worker_principal",
        "work_hand_indices",
        "artifact_count",
        "artifacts",
        "runner_done_sha256",
        "metadata_hidden_truth_exposed",
        "opponent_private_discards_used",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_DONE_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_ACCEPTANCE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "observed_transition_digest",
        "wave_index",
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "launch_receipt_sha256",
        "attempt_prefix",
        "done_path",
        "done_generation",
        "done_bytes",
        "done_sha256",
        "done_identity_sha256",
        "artifact_count",
        "artifact_records_sha256",
        "done_observed_before_acceptance",
        "create_only",
        "current_profile_changed",
        "acceptance_identity_sha256",
    }
)
_ATTEMPT_RESULT_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "launch_receipt_sha256",
        "wave_launch_receipt_sha256",
        "launch_mapping_sha256",
        "lifecycle_proof_sha256",
        "gce_absence_receipt_sha256",
        "exact_instance_created",
        "terminal_reason",
        "terminal_status",
        "pair_id",
        "peer_job_id",
        "pair_atomic_outcome",
        "valid_done_observed",
        "job_snapshot_object_count",
        "current_attempt_object_count",
        "historical_partial_object_count",
        "historical_done_object_count",
        "job_snapshot_sha256",
        "done_generation",
        "done_sha256",
        "acceptance_generation",
        "acceptance_sha256",
        "acceptance_create_performed",
        "local_materialized",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "previous_attempt_ledger_sha256",
        "input_resume_plan_sha256",
        "project_id",
        "zone",
        "observed_at_utc",
        "readback_source",
        "wave_index",
        "selected_attempt_count",
        "attempt_results",
        "accepted_job_ids",
        "failed_job_ids",
        "observed_transition",
        "attempt_ledger",
        "next_resume_plan",
        "resume_blocker",
        "expected_artifact_inventory",
        "observed_artifact_inventory",
        "local_destination",
        "materialized_paths",
        "done_is_only_worker_commit_marker",
        "acceptance_create_only",
        "pair_atomicity_enforced",
        "validated_lifecycle_proof",
        "lifecycle_proof_sha256",
        "gce_absence_receipt_sha256",
        "worker_iam_cleanup_receipt",
        "worker_iam_cleanup_receipt_sha256",
        "worker_iam_bindings_absent",
        "transition_lifecycle_binding",
        "nonaccepted_jobs_returned_to_resume",
        "result_store_protocol_used",
        "vm_lifecycle_mutation_performed",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "receipt_sha256",
    }
)

_TRANSITION_LIFECYCLE_BINDING_KEYS = frozenset(
    {
        "schema",
        "observed_transition_digest",
        "wave_index",
        "wave_launch_receipt_sha256",
        "launch_mapping_sha256",
        "accepted_launch_receipt_sha256s",
        "lifecycle_proof_sha256",
        "controller_lifecycle_receipt_sha256",
        "gce_absence_receipt_sha256",
        "worker_iam_cleanup_receipt_sha256",
        "all_selected_instances_absent",
        "all_selected_boot_disks_absent",
        "worker_iam_bindings_absent",
        "binding_sha256",
    }
)

_LIFECYCLE_PROOF_KEYS = frozenset(
    {
        "schema",
        "status",
        "controller_context_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "lifecycle_event_sha256",
        "lifecycle_receipt_sha256",
        "launch_event_sha256",
        "delete_event_sha256",
        "absence_event_sha256",
        "worker_iam_cleanup_event_sha256",
        "launch_bundle_sha256",
        "gce_create_receipt",
        "actual_launch_receipt",
        "gce_delete_receipt",
        "gce_absence_receipt",
        "worker_iam_cleanup_receipt",
        "selected_instance_mapping",
        "gce_create_rows",
        "actual_launch_rows",
        "selected_instance_count",
        "exact_created_instance_count",
        "exact_uncreated_instance_count",
        "create_classification",
        "actual_launch_receipt_present",
        "lifecycle_attested_at_utc",
        "journal_event_count",
        "journal_hash_chain_valid",
        "all_producer_receipts_valid",
        "all_owned_instances_absent",
        "all_owned_boot_disks_absent",
        "worker_iam_bindings_absent",
        "additional_create_authorized",
        "current_profile_changed",
        "proof_sha256",
    }
)

_LIFECYCLE_MAPPING_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "artifact_prefix",
        "launch_receipt_sha256",
        "exact_instance_created",
        "provider_instance_id",
        "provider_boot_disk_id",
        "gce_spec_sha256",
        "gce_operation_id",
        "actual_launch_operation_id",
        "actual_launch_instance_status",
        "ownership_label",
        "final_instance_absent",
        "final_boot_disk_absent",
    }
)

_FORBIDDEN_HIDDEN_KEYS = frozenset(
    {
        "opponent_private_discards",
        "opponent_hidden_discards",
        "realized_deck_tail",
        "hidden_truth",
        "unknown_deck_order",
    }
)


class ResultStore(Protocol):
    """Minimal generation-pinned object-store boundary used by the receiver."""

    def list_prefix(self, *, prefix: str) -> Sequence[Mapping[str, Any]]: ...

    def read_current(
        self, *, path: str, allow_missing: bool = False
    ) -> tuple[Mapping[str, Any], bytes] | None: ...

    def read_bytes(self, *, path: str, generation: int) -> bytes: ...

    def create_only(self, *, path: str, data: bytes) -> Mapping[str, Any]: ...


class ValidatedLifecycleProofAdapterV2:
    """One-shot boundary around controller journal replay validation.

    Production callers bind ``replay_callback`` to
    ``Full100WaveControllerV2.validate_lifecycle_closeout_chain`` with its
    complete producer-evidence arguments.  The receiver never accepts a raw
    lifecycle dictionary as a substitute.
    """

    def __init__(self, replay_callback: Callable[[], Mapping[str, Any]]) -> None:
        if not callable(replay_callback):
            raise TypeError("lifecycle replay callback is not callable")
        self._replay_callback = replay_callback
        self._used = False

    def validate(
        self,
        *,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        resume_plan: Mapping[str, Any],
        receiver_observed_at_utc: str,
    ) -> dict[str, Any]:
        if self._used:
            raise RuntimeError("lifecycle replay adapter is one-shot")
        self._used = True
        raw = self._replay_callback()
        if not isinstance(raw, Mapping):
            raise TypeError("lifecycle replay callback did not return a proof")
        return validate_lifecycle_proof(
            wave_plan,
            attempt_ledger,
            resume_plan,
            raw,
            receiver_observed_at_utc=receiver_observed_at_utc,
        )


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} changed")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} changed")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ValueError(f"{label} changed") from None
    if parsed.tzinfo is None or parsed.microsecond != 0:
        raise ValueError(f"{label} changed")
    return parsed.astimezone(timezone.utc)


def _worker_iam_cleanup_semantics_are_exact(
    cleanup: Mapping[str, Any], *, exact_binding_count: int
) -> bool:
    """Mirror all producer cleanup/recovery attribution coupling."""

    recovered = cleanup.get("recovered_after_outcome_ambiguity")
    if type(recovered) is not bool or type(exact_binding_count) is not int:
        return False
    expected = (
        (
            _IAM_CLEANUP_RECOVERED_STATUS,
            0,
            0,
            False,
            "unknown",
        )
        if recovered is True
        else (
            _IAM_CLEANUP_NORMAL_STATUS,
            exact_binding_count,
            1,
            True,
            "performed",
        )
    )
    actual = (
        cleanup.get("status"),
        cleanup.get("removed_binding_count"),
        cleanup.get("set_attempt_count"),
        cleanup.get("cloud_mutation_performed"),
        cleanup.get("source_mutation_outcome"),
    )
    return actual == expected


def validate_lifecycle_proof(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    receiver_observed_at_utc: str,
) -> dict[str, Any]:
    """Validate the normalized output of controller journal replay.

    The proof is not accepted as a caller-built lifecycle assertion: its exact
    schema is the output of ``Full100WaveControllerV2`` replay, includes the
    producer receipts, and binds every selected name to a unique derived
    per-attempt launch identity.
    """

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if not isinstance(value, Mapping):
        raise ValueError("validated lifecycle proof is not an object")
    proof = deepcopy(dict(value))
    _exact_keys(proof, _LIFECYCLE_PROOF_KEYS, "validated lifecycle proof")
    digest = proof.pop("proof_sha256", None)
    if digest != controller_v2.canonical_sha256(proof):
        raise ValueError("validated lifecycle proof digest changed")
    proof["proof_sha256"] = _require_sha(digest, "lifecycle proof")
    context = {
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
    }
    if (
        proof["schema"] != controller_v2.LIFECYCLE_PROOF_SCHEMA
        or proof["status"]
        != "controller_journal_and_all_producer_receipts_revalidated"
        or any(proof[key] != expected for key, expected in context.items())
        or proof["controller_context_sha256"]
        != controller_v2.canonical_sha256(context)
        or type(proof["journal_event_count"]) is not int
        or proof["journal_event_count"] <= 0
        or proof["journal_hash_chain_valid"] is not True
        or proof["all_producer_receipts_valid"] is not True
        or proof["all_owned_instances_absent"] is not True
        or proof["all_owned_boot_disks_absent"] is not True
        or proof["worker_iam_bindings_absent"] is not True
        or proof["additional_create_authorized"] is not False
        or proof["current_profile_changed"] is not False
        or _parse_utc(
            proof["lifecycle_attested_at_utc"], "lifecycle attested time"
        )
        > _parse_utc(receiver_observed_at_utc, "receiver observed time")
    ):
        raise ValueError("validated lifecycle proof context or safety changed")
    for key in (
        "lifecycle_event_sha256",
        "lifecycle_receipt_sha256",
        "launch_event_sha256",
        "delete_event_sha256",
        "absence_event_sha256",
        "worker_iam_cleanup_event_sha256",
        "launch_bundle_sha256",
    ):
        _require_sha(proof[key], f"lifecycle {key}")

    selected = resume["selected_attempts"]
    mappings = proof.get("selected_instance_mapping")
    if not isinstance(mappings, list) or len(mappings) != len(selected):
        raise ValueError("lifecycle selected mapping cardinality changed")
    launch_receipts: set[str] = set()
    created_jobs: list[str] = []
    checked_mappings: list[dict[str, Any]] = []
    for expected, raw in zip(selected, mappings, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("lifecycle selected mapping is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _LIFECYCLE_MAPPING_KEYS, "lifecycle selected mapping")
        if any(
            row[key] != expected[key]
            for key in (
                "job_id",
                "source_role",
                "attempt_id",
                "instance_id",
                "artifact_prefix",
            )
        ):
            raise ValueError("lifecycle selected mapping escaped resume plan")
        launch_receipt = _require_sha(
            row["launch_receipt_sha256"], "per-attempt launch identity"
        )
        if launch_receipt in launch_receipts:
            raise ValueError("lifecycle per-attempt launch identity was reused")
        launch_receipts.add(launch_receipt)
        if (
            type(row["exact_instance_created"]) is not bool
            or row["final_instance_absent"] is not True
            or row["final_boot_disk_absent"] is not True
        ):
            raise ValueError("lifecycle selected mapping is not finally absent")
        provider_fields = (
            "provider_instance_id",
            "provider_boot_disk_id",
            "gce_spec_sha256",
            "gce_operation_id",
        )
        if row["exact_instance_created"] is True:
            if any(row[key] is None for key in provider_fields):
                raise ValueError("exact-created lifecycle provider lineage is missing")
            if not all(
                isinstance(row[key], str) and row[key]
                for key in (
                    "provider_instance_id",
                    "provider_boot_disk_id",
                    "gce_operation_id",
                )
            ):
                raise ValueError("exact-created lifecycle provider identity changed")
            _require_sha(row["gce_spec_sha256"], "GCE instance spec")
            created_jobs.append(row["job_id"])
        elif any(row[key] is not None for key in provider_fields):
            raise ValueError("uncreated lifecycle row carries provider identity")
        checked_mappings.append(row)

    selected_count = len(selected)
    created_count = len(created_jobs)
    classification = (
        "all_selected_created"
        if created_count == selected_count
        else "no_selected_created"
        if created_count == 0
        else "partial_selected_created"
    )
    if (
        proof["selected_instance_count"] != selected_count
        or proof["exact_created_instance_count"] != created_count
        or proof["exact_uncreated_instance_count"] != selected_count - created_count
        or proof["create_classification"] != classification
    ):
        raise ValueError("lifecycle exact-create classification changed")

    create = proof.get("gce_create_receipt")
    absence = proof.get("gce_absence_receipt")
    deleted = proof.get("gce_delete_receipt")
    if not all(isinstance(row, Mapping) for row in (create, absence, deleted)):
        raise ValueError("lifecycle GCE producer receipt is missing")
    create = deepcopy(dict(create))
    absence = deepcopy(dict(absence))
    deleted = deepcopy(dict(deleted))
    for label, receipt in (
        ("create", create),
        ("delete", deleted),
        ("absence", absence),
    ):
        unsigned = dict(receipt)
        receipt_sha = unsigned.pop("receipt_sha256", None)
        if receipt_sha != gce_v2.canonical_sha256(unsigned):
            raise ValueError(f"lifecycle GCE {label} receipt digest changed")
    if (
        create.get("schema") != gce_v2.GCE_CREATE_RECEIPT_SCHEMA
        or create.get("created_instance_count") != created_count
        or create.get("create_complete") is not (created_count == selected_count)
        or create.get("rows") != proof["gce_create_rows"]
        or [row.get("job_id") for row in create.get("rows", [])]
        != created_jobs
        or deleted.get("schema") != gce_v2.GCE_DELETE_RECEIPT_SCHEMA
        or deleted.get("create_receipt_sha256") != create["receipt_sha256"]
        or absence.get("schema") != gce_v2.GCE_ABSENCE_RECEIPT_SCHEMA
        or absence.get("create_receipt_sha256") != create["receipt_sha256"]
        or absence.get("delete_receipt_sha256") != deleted["receipt_sha256"]
        or absence.get("checked_instance_count") != selected_count
        or absence.get("absent_instance_names")
        != [row["instance_id"] for row in selected]
        or absence.get("absent_boot_disk_names")
        != [row["instance_id"] for row in selected]
        or absence.get("all_instances_absent") is not True
        or absence.get("all_boot_disks_absent") is not True
        or absence.get("read_only") is not True
        or absence.get("additional_create_authorized") is not False
        or absence.get("current_profile_changed") is not False
        or _parse_utc(absence.get("observed_at_utc"), "GCE absence time")
        > _parse_utc(receiver_observed_at_utc, "receiver observed time")
        or _parse_utc(
            proof["lifecycle_attested_at_utc"], "lifecycle attested time"
        )
        < _parse_utc(absence.get("observed_at_utc"), "GCE absence time")
    ):
        raise ValueError("lifecycle GCE create/delete/absence binding changed")

    actual = proof.get("actual_launch_receipt")
    actual_present = actual is not None
    if actual_present:
        if not isinstance(actual, Mapping):
            raise ValueError("lifecycle actual launch receipt shape changed")
        actual = deepcopy(dict(actual))
        unsigned = dict(actual)
        actual_sha = unsigned.pop("receipt_sha256", None)
        if (
            actual_sha != wave_v2.canonical_sha256(unsigned)
            or actual.get("rows") != proof["actual_launch_rows"]
            or created_count != selected_count
            or any(
                row["actual_launch_operation_id"] is None
                or row["actual_launch_instance_status"] is None
                for row in checked_mappings
            )
        ):
            raise ValueError("lifecycle actual launch receipt binding changed")
    elif proof["actual_launch_rows"] != [] or any(
        row["actual_launch_operation_id"] is not None
        or row["actual_launch_instance_status"] is not None
        for row in checked_mappings
    ):
        raise ValueError("partial lifecycle unexpectedly carries actual launch")
    if proof["actual_launch_receipt_present"] is not actual_present:
        raise ValueError("lifecycle actual launch presence classification changed")

    cleanup = proof.get("worker_iam_cleanup_receipt")
    if not isinstance(cleanup, Mapping):
        raise ValueError("lifecycle worker IAM cleanup receipt is missing")
    cleanup = deepcopy(dict(cleanup))
    cleanup_unsigned = dict(cleanup)
    cleanup_sha = cleanup_unsigned.pop("receipt_sha256", None)
    if (
        cleanup_sha != worker_iam_v2.canonical_sha256(cleanup_unsigned)
        or cleanup.get("schema") != worker_iam_v2.CLEANUP_RECEIPT_SCHEMA
        or not _worker_iam_cleanup_semantics_are_exact(
            cleanup,
            exact_binding_count=2 * len(resume["selected_attempts"]),
        )
        or cleanup.get("remaining_targeted_binding_count") != 0
        or cleanup.get("post_cleanup_absence_readback") is not True
        or cleanup.get("cleanup_complete") is not True
    ):
        raise ValueError("lifecycle worker IAM cleanup receipt changed")
    proof["selected_instance_mapping"] = checked_mappings
    return proof


def _canonical_done_bytes(value: Any) -> bytes:
    """Match the startup transport: canonical ASCII JSON without trailing LF."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _safe_path(value: Any, label: str = "object path") -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} changed")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} changed")
    return path.as_posix()


def _record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("provider object record is not an object")
    row = deepcopy(dict(value))
    _exact_keys(row, _OBJECT_RECORD_KEYS, "provider object record")
    row["path"] = _safe_path(row["path"])
    if (
        type(row["generation"]) is not int
        or row["generation"] <= 0
        or type(row["bytes"]) is not int
        or row["bytes"] <= 0
    ):
        raise ValueError("provider object generation or size changed")
    _require_sha(row["sha256"], "provider object SHA-256")
    return row


def _read_bound(store: ResultStore, record: Mapping[str, Any]) -> bytes:
    raw = store.read_bytes(
        path=record["path"], generation=int(record["generation"])
    )
    if not isinstance(raw, bytes):
        raise ValueError("generation-pinned object bytes changed")
    if (
        len(raw) != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("generation-pinned object bytes changed")
    return raw


def _read_current_bound(
    store: ResultStore, *, path: str, allow_missing: bool
) -> tuple[dict[str, Any], bytes] | None:
    observed = store.read_current(path=path, allow_missing=allow_missing)
    if observed is None:
        if allow_missing:
            return None
        raise ValueError("required provider object is missing")
    if not isinstance(observed, tuple) or len(observed) != 2:
        raise ValueError("provider current-object readback changed")
    record = _record(observed[0])
    raw = observed[1]
    if record["path"] != path or not isinstance(raw, bytes):
        raise ValueError("provider current-object identity changed")
    if (
        len(raw) != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("provider current-object readback bytes changed")
    return record, raw


def _json(raw: bytes, label: str, *, canonical_done: bool = False) -> dict[str, Any]:
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError(f"{label} bytes changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{label} JSON changed") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON changed")
    if canonical_done and _canonical_done_bytes(value) != raw:
        raise ValueError(f"{label} canonical bytes changed")
    return value


def _exposes_hidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in _FORBIDDEN_HIDDEN_KEYS or _exposes_hidden(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_exposes_hidden(item) for item in value)
    return False


def _job_metadata(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["job_id"]: {
            "source_role": row["source_role"],
            "shard_index": row["shard_index"],
            "work_hand_indices": list(row["work_hand_indices"]),
        }
        for row in plan["full100_plan"]["jobs"]
    }


def _history_by_job(ledger: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["job_id"]: deepcopy(row["attempts"])
        for row in ledger["transitions"][-1]["attempt_history"]
    }


def _validate_bootstraps(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    job_bootstraps: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_startup_sha256 = science_registry.resolve_startup_sha256(plan)
    selected = resume["selected_attempts"]
    if not isinstance(job_bootstraps, Sequence) or isinstance(
        job_bootstraps, (str, bytes)
    ):
        raise ValueError("job bootstrap inventory is missing")
    if len(job_bootstraps) != len(selected):
        raise ValueError("job bootstrap inventory cardinality changed")
    result: dict[str, dict[str, Any]] = {}
    for expected, raw in zip(selected, job_bootstraps, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("job bootstrap is not an object")
        value = deepcopy(dict(raw))
        _exact_keys(value, _BOOTSTRAP_KEYS, "job bootstrap")
        digest = value.pop("bootstrap_sha256", None)
        if digest != package_v2.canonical_sha256(value):
            raise ValueError("job bootstrap digest changed")
        value["bootstrap_sha256"] = digest
        for field in (
            "scientific_source",
            "scientific_manifest",
            "wheelhouse",
            "wheelhouse_manifest",
            "startup",
            "wave_plan",
            "job_manifest",
        ):
            binding = value.get(field)
            if not isinstance(binding, Mapping):
                raise ValueError("job bootstrap object binding is missing")
            _exact_keys(binding, _OBJECT_BINDING_KEYS, "job bootstrap object binding")
            _safe_path(binding.get("object_name"), "job bootstrap object name")
            _require_sha(binding.get("sha256"), "job bootstrap object SHA-256")
            if type(binding.get("bytes")) is not int or binding["bytes"] <= 0:
                raise ValueError("job bootstrap object bytes changed")
        if (
            value["schema"] != package_v2.JOB_BOOTSTRAP_SCHEMA
            or value["status"]
            != "single_job_bootstrap_bound_to_prelaunch_authorization"
            or value["run_name"] != plan["run_name"]
            or value["execution_identity_sha256"]
            != plan["execution_identity_sha256"]
            or value["wave_plan_sha256"] != plan["schedule_sha256"]
            or value["attempt_ledger_sha256"] != ledger["ledger_sha256"]
            or value["resume_sha256"] != resume["resume_sha256"]
            or value["observed_transition_digest"]
            != resume["observed_transition_digest"]
            or value["wave_index"] != resume["resume_wave_index"]
            or value["job_id"] != expected["job_id"]
            or value["source_role"] != expected["source_role"]
            or value["attempt_id"] != expected["attempt_id"]
            or value["instance_name"] != expected["instance_id"]
            or value["artifact_prefix"] != expected["artifact_prefix"]
            or value["startup"]["sha256"] != expected_startup_sha256
            or _SHA.fullmatch(str(value["outer_manifest_sha256"])) is None
            or _SHA.fullmatch(str(value["content_payload_sha256"])) is None
            or _SHA.fullmatch(str(value["prelaunch_authorization_sha256"])) is None
            or _SERVICE_ACCOUNT.fullmatch(str(value["worker_principal"])) is None
            or value["one_vm_one_job_one_role"] is not True
            or value["additional_create_authorized"] is not False
            or value["hidden_truth_exposed"] is not False
        ):
            raise ValueError("job bootstrap boundary changed")
        result[value["job_id"]] = value
    if list(result) != [row["job_id"] for row in selected]:
        raise ValueError("job bootstrap order or identity changed")
    return result


def _validate_terminal_observations(
    resume: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    lifecycle_proof: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    selected = resume["selected_attempts"]
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("terminal attempt observations are missing")
    if len(observations) != len(selected):
        raise ValueError("terminal attempt observation cardinality changed")
    result: dict[str, dict[str, Any]] = {}
    mappings = lifecycle_proof["selected_instance_mapping"]
    wave_launch_receipt = (
        lifecycle_proof["gce_create_receipt"]["receipt_sha256"]
        if lifecycle_proof["actual_launch_receipt"] is None
        else lifecycle_proof["actual_launch_receipt"]["receipt_sha256"]
    )
    launch_mapping = controller_v2.canonical_sha256(mappings)
    absence_sha = lifecycle_proof["gce_absence_receipt"]["receipt_sha256"]
    for expected, mapping, raw in zip(
        selected, mappings, observations, strict=True
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("terminal attempt observation is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _TERMINAL_KEYS, "terminal attempt observation")
        if any(
            row[key] != expected[key]
            for key in ("job_id", "source_role", "attempt_id", "instance_id")
        ):
            raise ValueError("terminal attempt escaped the selected resume plan")
        if (
            row["terminal_reason"] not in TERMINAL_REASONS
            or (
                row["terminal_reason"] == "create_missing_before_done"
            )
            is not (mapping["exact_instance_created"] is False)
        ):
            raise ValueError(
                "terminal attempt is not bound to exact lifecycle proof lineage"
            )
        result[row["job_id"]] = {
            **row,
            "launch_receipt_sha256": mapping["launch_receipt_sha256"],
            "wave_launch_receipt_sha256": wave_launch_receipt,
            "launch_mapping_sha256": launch_mapping,
            "lifecycle_proof_sha256": lifecycle_proof["proof_sha256"],
            "gce_absence_receipt_sha256": absence_sha,
            "exact_instance_created": mapping["exact_instance_created"],
        }
    return result


def _selected_candidate_reference_pairs(
    plan: Mapping[str, Any], resume: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return complete selected pairs and reject half-pair or lane drift."""

    wave = plan["waves"][resume["resume_wave_index"]]
    selected_by_job = {
        row["job_id"]: row for row in resume["selected_attempts"]
    }
    pairs: list[dict[str, Any]] = []
    flattened: list[str] = []
    for pair in wave["candidate_reference_pairs"]:
        candidate_job = pair["candidate_job_id"]
        reference_job = pair["reference_job_id"]
        present = {
            job
            for job in (candidate_job, reference_job)
            if job in selected_by_job
        }
        if not present:
            continue
        if present != {candidate_job, reference_job}:
            raise ValueError("resume selection contains a candidate/reference half-pair")
        candidate = selected_by_job[candidate_job]
        reference = selected_by_job[reference_job]
        attempt_id = candidate["attempt_id"]
        if reference["attempt_id"] != attempt_id:
            raise ValueError("candidate/reference pair escaped the same attempt lane")
        if (
            candidate["instance_id"]
            != pair["candidate_attempt_instance_ids"][attempt_id]
            or reference["instance_id"]
            != pair["reference_attempt_instance_ids"][attempt_id]
            or candidate["instance_id"] == reference["instance_id"]
        ):
            raise ValueError("candidate/reference pair instance lineage changed")
        pairs.append(
            {
                "pair_id": pair["pair_id"],
                "attempt_id": attempt_id,
                "candidate_job_id": candidate_job,
                "reference_job_id": reference_job,
            }
        )
        flattened.extend((candidate_job, reference_job))
    if flattened != [row["job_id"] for row in resume["selected_attempts"]]:
        raise ValueError("selected attempt order is not complete pair order")
    return pairs


def _validate_worker_iam_cleanup(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    bootstraps: Mapping[str, Mapping[str, Any]],
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
    worker_iam_cleanup_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the full IAM chain before any transition can be emitted."""

    selected_jobs = [row["job_id"] for row in resume["selected_attempts"]]
    content_hashes = {
        bootstraps[job]["content_payload_sha256"] for job in selected_jobs
    }
    manifest_hashes = {
        bootstraps[job]["outer_manifest_sha256"] for job in selected_jobs
    }
    if len(content_hashes) != 1 or len(manifest_hashes) != 1:
        raise ValueError("worker bootstrap content lineage changed within the wave")
    return worker_iam_v2.validate_cleanup_receipt(
        iam_plan=worker_iam_plan,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=worker_iam_plan.get("immutable_content_prefix"),
        content_payload_sha256=next(iter(content_hashes)),
        outer_manifest_sha256=next(iter(manifest_hashes)),
        prepare_receipt=worker_iam_prepare_receipt,
        install_receipt=worker_iam_install_receipt,
        readback_receipt=worker_iam_readback_receipt,
        value=worker_iam_cleanup_receipt,
    )


def _expected_attempt_paths(
    *, prefix: str, role: str, work: Sequence[int]
) -> tuple[list[str], list[str]]:
    relative: list[str] = []
    absolute: list[str] = []
    for hand in work:
        for path in (
            f"roots/hand_{hand:03d}.json",
            f"hands/{role}/hand_{hand:03d}.json",
        ):
            relative.append(path)
            absolute.append(f"{prefix}/{path}")
    return relative, absolute


def _validate_done(
    *,
    raw: bytes,
    done_record: Mapping[str, Any],
    data_records: Mapping[str, Mapping[str, Any]],
    data_raw: Mapping[str, bytes],
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    selected: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    work: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    done = _json(raw, "transport DONE", canonical_done=True)
    _exact_keys(done, _DONE_KEYS, "transport DONE")
    role = selected["source_role"]
    relative, absolute = _expected_attempt_paths(
        prefix=selected["artifact_prefix"], role=role, work=work
    )
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(relative):
        raise ValueError("transport DONE artifact inventory changed")
    for rel, path, raw_row in zip(relative, absolute, artifacts, strict=True):
        if not isinstance(raw_row, Mapping):
            raise ValueError("transport DONE artifact is not an object")
        row = dict(raw_row)
        _exact_keys(row, _DONE_ARTIFACT_KEYS, "transport DONE artifact")
        provider = data_records[path]
        if (
            row["path"] != rel
            or row["sha256"] != provider["sha256"]
            or row["bytes"] != provider["bytes"]
        ):
            raise ValueError("transport DONE artifact disagrees with provider")
    roots = []
    for hand in work:
        path = f"{selected['artifact_prefix']}/roots/hand_{hand:03d}.json"
        roots.append({"hand_index": hand, "sha256": data_records[path]["sha256"]})
    root_digest = wave_v2.canonical_sha256(roots)
    expected_identity = wave_v2.expected_done_identity_sha256(
        plan,
        job_id=selected["job_id"],
        attempt_id=selected["attempt_id"],
        root_digest=root_digest,
    )
    binding = plan["runtime_binding"]
    if (
        done["schema"] != DONE_SCHEMA
        or done["status"] != "complete_validated_single_job_attempt"
        or done["run_name"] != plan["run_name"]
        or done["execution_identity_sha256"]
        != plan["execution_identity_sha256"]
        or done["wave_plan_sha256"] != plan["schedule_sha256"]
        or done["attempt_ledger_sha256"] != ledger["ledger_sha256"]
        or done["resume_sha256"] != resume["resume_sha256"]
        or done["observed_transition_digest"]
        != resume["observed_transition_digest"]
        or done["wave_index"] != resume["resume_wave_index"]
        or done["job_id"] != selected["job_id"]
        or done["source_role"] != role
        or done["attempt_id"] != selected["attempt_id"]
        or done["package_sha256"] != binding["package_sha256"]
        or done["image_digest"] != binding["image_digest"]
        or done["binary_sha256"] != binding["binary_sha256_by_role"][role]
        or done["allocation_digest"] != binding["allocation_digest"]
        or done["run_contract_digest"] != plan["run_contract_digest"]
        or done["root_digest"] != root_digest
        or done["done_identity_sha256"] != expected_identity
        or done["content_payload_sha256"] != bootstrap["content_payload_sha256"]
        or done["outer_manifest_sha256"] != bootstrap["outer_manifest_sha256"]
        or done["prelaunch_authorization_sha256"]
        != bootstrap["prelaunch_authorization_sha256"]
        or done["worker_principal"] != bootstrap["worker_principal"]
        or done["work_hand_indices"] != list(work)
        or done["artifact_count"] != len(relative)
        or _SHA.fullmatch(str(done["runner_done_sha256"])) is None
        or done["metadata_hidden_truth_exposed"] is not False
        or done["opponent_private_discards_used"] is not False
        or done["training_eligible"] is not False
        or done["quality_evidence"] is not False
        or done["promotion_evidence"] is not False
        or done["current_profile_changed"] is not False
        or _exposes_hidden(done)
    ):
        raise ValueError("transport DONE identity or scientific binding changed")
    for path, artifact_raw in data_raw.items():
        value = _json(artifact_raw, f"result artifact {path}")
        if _exposes_hidden(value):
            raise ValueError("result artifact exposed hidden truth")
    transition_done = {
        "job_id": selected["job_id"],
        "source_role": role,
        "attempt_id": selected["attempt_id"],
        "path": done_record["path"],
        "generation": done_record["generation"],
        "bytes": done_record["bytes"],
        "sha256": done_record["sha256"],
        "done_identity_sha256": done["done_identity_sha256"],
        "package_sha256": done["package_sha256"],
        "image_digest": done["image_digest"],
        "binary_sha256": done["binary_sha256"],
        "allocation_digest": done["allocation_digest"],
        "root_digest": done["root_digest"],
    }
    return done, transition_done


def _acceptance_value(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    selected: Mapping[str, Any],
    terminal: Mapping[str, Any],
    done: Mapping[str, Any],
    done_record: Mapping[str, Any],
    artifact_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "done_last_generation_pinned_attempt_accepted",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "observed_transition_digest": resume["observed_transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "job_id": selected["job_id"],
        "source_role": selected["source_role"],
        "attempt_id": selected["attempt_id"],
        "instance_id": selected["instance_id"],
        "launch_receipt_sha256": terminal["launch_receipt_sha256"],
        "attempt_prefix": selected["artifact_prefix"],
        "done_path": done_record["path"],
        "done_generation": done_record["generation"],
        "done_bytes": done_record["bytes"],
        "done_sha256": done_record["sha256"],
        "done_identity_sha256": done["done_identity_sha256"],
        "artifact_count": len(artifact_records),
        "artifact_records_sha256": wave_v2.canonical_sha256(
            list(artifact_records)
        ),
        "done_observed_before_acceptance": True,
        "create_only": True,
        "current_profile_changed": False,
    }
    return {
        **core,
        "acceptance_identity_sha256": wave_v2.canonical_sha256(core),
    }


def _validate_acceptance_value(
    value: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("ACCEPTED object is not an object")
    checked = deepcopy(dict(value))
    _exact_keys(checked, _ACCEPTANCE_KEYS, "ACCEPTED object")
    identity = checked.pop("acceptance_identity_sha256", None)
    if identity != wave_v2.canonical_sha256(checked):
        raise ValueError("ACCEPTED object identity changed")
    checked["acceptance_identity_sha256"] = identity
    if checked != expected:
        raise ValueError("existing ACCEPTED object differs from exact attempt")
    return checked


def validate_acceptance_payload_context(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    path: str,
    payload: bytes,
) -> dict[str, Any]:
    """Validate an ACCEPTED payload before any provider create is attempted."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    path = _safe_path(path, "ACCEPTED path")
    value = _json(payload, "ACCEPTED create payload")
    _exact_keys(value, _ACCEPTANCE_KEYS, "ACCEPTED create payload")
    if wave_v2.canonical_bytes(value) != payload:
        raise ValueError("ACCEPTED create payload canonical bytes changed")
    identity = value.pop("acceptance_identity_sha256", None)
    if identity != wave_v2.canonical_sha256(value):
        raise ValueError("ACCEPTED create payload identity changed")
    value["acceptance_identity_sha256"] = identity
    selected = [
        row
        for row in resume["selected_attempts"]
        if row["job_id"] == value.get("job_id")
    ]
    if len(selected) != 1:
        raise ValueError("ACCEPTED create job is outside selected attempts")
    row = selected[0]
    expected_path = plan["artifact_contract"][
        "job_acceptance_path_template"
    ].format(job_id=row["job_id"])
    if (
        path != expected_path
        or value["schema"] != ACCEPTANCE_SCHEMA
        or value["status"] != "done_last_generation_pinned_attempt_accepted"
        or value["run_name"] != plan["run_name"]
        or value["execution_identity_sha256"]
        != plan["execution_identity_sha256"]
        or value["wave_plan_sha256"] != plan["schedule_sha256"]
        or value["attempt_ledger_sha256"] != ledger["ledger_sha256"]
        or value["resume_plan_sha256"] != resume["resume_sha256"]
        or value["observed_transition_digest"]
        != resume["observed_transition_digest"]
        or value["wave_index"] != resume["resume_wave_index"]
        or value["source_role"] != row["source_role"]
        or value["attempt_id"] != row["attempt_id"]
        or value["instance_id"] != row["instance_id"]
        or value["attempt_prefix"] != row["artifact_prefix"]
        or value["done_path"] != f"{row['artifact_prefix']}/DONE.json"
        or value["done_observed_before_acceptance"] is not True
        or value["create_only"] is not True
        or value["current_profile_changed"] is not False
    ):
        raise ValueError("ACCEPTED create payload escaped exact resume context")
    for key in (
        "launch_receipt_sha256",
        "done_sha256",
        "done_identity_sha256",
        "artifact_records_sha256",
    ):
        _require_sha(value[key], f"ACCEPTED {key}")
    for key in ("done_generation", "done_bytes", "artifact_count"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"ACCEPTED {key} changed")
    return value


def _create_or_read_acceptance(
    *,
    store: ResultStore,
    path: str,
    value: Mapping[str, Any],
    expected_existing_record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bytes, bool]:
    raw = wave_v2.canonical_bytes(value)
    current = _read_current_bound(store, path=path, allow_missing=True)
    created = False
    create_record: dict[str, Any] | None = None
    if current is None:
        create_record = _record(store.create_only(path=path, data=raw))
        if create_record["path"] != path:
            raise ValueError("ACCEPTED create response path changed")
        created = True
    readback = _read_current_bound(store, path=path, allow_missing=False)
    assert readback is not None
    record, observed_raw = readback
    if expected_existing_record is not None and record != expected_existing_record:
        raise ValueError("existing ACCEPTED generation changed after snapshot")
    if create_record is not None and record != create_record:
        raise ValueError("ACCEPTED create/readback generation changed")
    if observed_raw != raw:
        raise ValueError("ACCEPTED create/readback bytes changed")
    parsed = _json(observed_raw, "ACCEPTED object")
    _validate_acceptance_value(parsed, value)
    return record, observed_raw, created


def _local_target(root: Path, object_path: str) -> Path:
    parts = PurePosixPath(_safe_path(object_path)).parts
    target = root.joinpath(*parts)
    root_text = str(root)
    target_text = str(target.resolve(strict=False))
    if os.path.commonpath((root_text, target_text)) != root_text:
        raise ValueError("local receive path escaped destination")
    return target


def _ensure_safe_directory(path: Path, root: Path) -> None:
    if path == root:
        return
    _ensure_safe_directory(path.parent, root)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("local receive directory is unsafe")
    else:
        path.mkdir()
        os.chmod(path, 0o700)


def _write_immutable_bytes(root: Path, object_path: str, raw: bytes) -> bool:
    target = _local_target(root, object_path)
    _ensure_safe_directory(target.parent, root)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise FileExistsError(f"immutable local result differs: {target}")
        return False
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
                raise FileExistsError(f"immutable local result differs: {target}")
        if target.read_bytes() != raw:
            raise FileExistsError(f"immutable local result differs: {target}")
        os.chmod(target, 0o600)
        # POSIX permits an fsync of the containing directory.  Windows does
        # not expose directories through ``os.open``; the file itself was
        # already flushed above and the hard-link operation is write-once.
        if os.name != "nt":
            directory = os.open(
                target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_destination(destination: str | Path) -> Path:
    requested = Path(destination)
    if requested.exists():
        if requested.is_symlink() or not requested.is_dir():
            raise ValueError("local receive destination is unsafe")
    else:
        parent = requested.parent.resolve(strict=True)
        if requested.parent.is_symlink() or not parent.is_dir():
            raise ValueError("local receive destination parent is unsafe")
        requested.mkdir()
        os.chmod(requested, 0o700)
    return requested.resolve(strict=True)


def _final_inventory(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    observed_at_utc: str,
    store: ResultStore,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    expected = wave_v2.expected_artifact_inventory(plan, attempt_ledger=ledger)
    objects: list[dict[str, Any]] = []
    raw_by_path: dict[str, bytes] = {}
    for expected_record in expected["records"]:
        common = {
            "job_id": expected_record["job_id"],
            "source_role": expected_record["source_role"],
            "attempt_id": expected_record["accepted_attempt_id"],
        }
        for hand, path in zip(
            expected_record["work_hand_indices"],
            expected_record["root_paths"],
            strict=True,
        ):
            current = _read_current_bound(store, path=path, allow_missing=False)
            assert current is not None
            record, raw = current
            raw_by_path[path] = raw
            if _exposes_hidden(_json(raw, f"final root {path}")):
                raise ValueError("final root exposed hidden truth")
            objects.append(
                {
                    "path": path,
                    **common,
                    "object_kind": "root",
                    "hand_index": hand,
                    **{key: record[key] for key in ("generation", "bytes", "sha256")},
                    "done_identity_sha256": None,
                    "package_sha256": None,
                    "image_digest": None,
                    "binary_sha256": None,
                    "allocation_digest": None,
                    "root_digest": None,
                }
            )
        for hand, path in zip(
            expected_record["work_hand_indices"],
            expected_record["source_hand_paths"],
            strict=True,
        ):
            current = _read_current_bound(store, path=path, allow_missing=False)
            assert current is not None
            record, raw = current
            raw_by_path[path] = raw
            if _exposes_hidden(_json(raw, f"final source hand {path}")):
                raise ValueError("final source hand exposed hidden truth")
            objects.append(
                {
                    "path": path,
                    **common,
                    "object_kind": "source_hand",
                    "hand_index": hand,
                    **{key: record[key] for key in ("generation", "bytes", "sha256")},
                    "done_identity_sha256": None,
                    "package_sha256": None,
                    "image_digest": None,
                    "binary_sha256": None,
                    "allocation_digest": None,
                    "root_digest": None,
                }
            )
        done_path = expected_record["done_path"]
        done_current = _read_current_bound(
            store, path=done_path, allow_missing=False
        )
        assert done_current is not None
        done_record, done_raw = done_current
        raw_by_path[done_path] = done_raw
        done = _json(done_raw, "final transport DONE", canonical_done=True)
        _exact_keys(done, _DONE_KEYS, "final transport DONE")
        if any(
            done.get(key) != expected_record[key]
            for key in (
                "done_identity_sha256",
                "package_sha256",
                "image_digest",
                "binary_sha256",
                "allocation_digest",
                "root_digest",
            )
        ):
            raise ValueError("final transport DONE binding changed")
        objects.append(
            {
                "path": done_path,
                **common,
                "object_kind": "done",
                "hand_index": None,
                **{
                    key: done_record[key]
                    for key in ("generation", "bytes", "sha256")
                },
                "done_identity_sha256": done["done_identity_sha256"],
                "package_sha256": done["package_sha256"],
                "image_digest": done["image_digest"],
                "binary_sha256": done["binary_sha256"],
                "allocation_digest": done["allocation_digest"],
                "root_digest": done["root_digest"],
            }
        )
        acceptance_path = expected_record["acceptance_path"]
        acceptance_current = _read_current_bound(
            store, path=acceptance_path, allow_missing=False
        )
        assert acceptance_current is not None
        acceptance_record, acceptance_raw = acceptance_current
        raw_by_path[acceptance_path] = acceptance_raw
        acceptance = _json(acceptance_raw, "final ACCEPTED object")
        _exact_keys(acceptance, _ACCEPTANCE_KEYS, "final ACCEPTED object")
        if wave_v2.canonical_bytes(acceptance) != acceptance_raw:
            raise ValueError("final ACCEPTED canonical bytes changed")
        unsigned = dict(acceptance)
        identity = unsigned.pop("acceptance_identity_sha256")
        if (
            identity != wave_v2.canonical_sha256(unsigned)
            or acceptance["job_id"] != expected_record["job_id"]
            or acceptance["source_role"] != expected_record["source_role"]
            or acceptance["attempt_id"]
            != expected_record["accepted_attempt_id"]
            or acceptance["done_generation"] != expected_record["done_generation"]
            or acceptance["done_sha256"] != expected_record["done_sha256"]
            or acceptance["done_identity_sha256"]
            != expected_record["done_identity_sha256"]
            or acceptance["create_only"] is not True
        ):
            raise ValueError("final ACCEPTED binding changed")
        objects.append(
            {
                "path": acceptance_path,
                **common,
                "object_kind": "acceptance",
                "hand_index": None,
                **{
                    key: acceptance_record[key]
                    for key in ("generation", "bytes", "sha256")
                },
                "done_identity_sha256": None,
                "package_sha256": None,
                "image_digest": None,
                "binary_sha256": None,
                "allocation_digest": None,
                "root_digest": None,
            }
        )
    observed = wave_v2.build_observed_artifact_inventory(
        plan,
        attempt_ledger=ledger,
        observed_at_utc=observed_at_utc,
        objects=objects,
    )
    return expected, observed, raw_by_path


def receive_wave_results(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    job_bootstraps: Sequence[Mapping[str, Any]],
    terminal_observations: Sequence[Mapping[str, Any]],
    lifecycle_proof_adapter: ValidatedLifecycleProofAdapterV2,
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
    worker_iam_cleanup_receipt: Mapping[str, Any],
    project_id: str,
    zone: str,
    observed_at_utc: str,
    store: ResultStore,
    destination: str | Path,
    readback_source: str = "gcloud_readback",
) -> dict[str, Any]:
    """Receive one quiescent wave and derive its next local-only transition."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if resume["all_jobs_complete"] is True or not resume["selected_attempts"]:
        raise ValueError("result receiver requires one selected incomplete wave")
    if (
        not isinstance(project_id, str)
        or _PROJECT.fullmatch(project_id) is None
        or not isinstance(zone, str)
        or _ZONE.fullmatch(zone) is None
        or readback_source not in {"local_observed_fixture", "gcloud_readback"}
    ):
        raise ValueError("result receiver observation scope changed")
    bootstraps = _validate_bootstraps(
        plan=plan,
        ledger=ledger,
        resume=resume,
        job_bootstraps=job_bootstraps,
    )
    selected_pairs = _selected_candidate_reference_pairs(plan, resume)
    if type(lifecycle_proof_adapter) is not ValidatedLifecycleProofAdapterV2:
        raise TypeError("validated lifecycle proof adapter is required")
    lifecycle_proof = lifecycle_proof_adapter.validate(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        receiver_observed_at_utc=observed_at_utc,
    )
    cleanup = _validate_worker_iam_cleanup(
        plan=plan,
        ledger=ledger,
        resume=resume,
        bootstraps=bootstraps,
        worker_iam_plan=worker_iam_plan,
        worker_iam_prepare_receipt=worker_iam_prepare_receipt,
        worker_iam_install_receipt=worker_iam_install_receipt,
        worker_iam_readback_receipt=worker_iam_readback_receipt,
        worker_iam_cleanup_receipt=worker_iam_cleanup_receipt,
    )
    if (
        cleanup != lifecycle_proof["worker_iam_cleanup_receipt"]
        or lifecycle_proof["gce_absence_receipt"].get("project") != project_id
        or lifecycle_proof["gce_absence_receipt"].get("zone") != zone
    ):
        raise ValueError("lifecycle proof differs from receiver lifecycle inputs")
    terminals = _validate_terminal_observations(
        resume, terminal_observations, lifecycle_proof
    )
    metadata = _job_metadata(plan)
    previous = ledger["transitions"][-1]
    histories = _history_by_job(ledger)

    prepared: list[dict[str, Any]] = []
    for selected in resume["selected_attempts"]:
        job = selected["job_id"]
        role = selected["source_role"]
        attempt = selected["attempt_id"]
        terminal = terminals[job]
        work = metadata[job]["work_hand_indices"]
        job_prefix = selected["artifact_prefix"].split("/attempts/", 1)[0] + "/"
        listed = [_record(row) for row in store.list_prefix(prefix=job_prefix)]
        by_path = {row["path"]: row for row in listed}
        if len(by_path) != len(listed):
            raise ValueError("provider job snapshot contains a duplicate path")

        relative, data_paths = _expected_attempt_paths(
            prefix=selected["artifact_prefix"], role=role, work=work
        )
        del relative
        done_path = f"{selected['artifact_prefix']}/DONE.json"
        acceptance_path = plan["artifact_contract"][
            "job_acceptance_path_template"
        ].format(job_id=job)
        allowed = {*data_paths, done_path, acceptance_path}
        historical_partial_paths: set[str] = set()
        historical_done_paths: set[str] = set()
        for historical in histories[job]:
            if historical["terminal_status"] != "failed":
                raise ValueError("selected job already has a non-failed history")
            prefix = plan["artifact_contract"]["attempt_path_template"].format(
                job_id=job, attempt_id=historical["attempt_id"]
            )
            _, old_paths = _expected_attempt_paths(
                prefix=prefix, role=role, work=work
            )
            historical_partial_paths.update(old_paths)
            old_done = f"{prefix}/DONE.json"
            historical_done_paths.add(old_done)
        allowed.update(historical_partial_paths)
        allowed.update(historical_done_paths)
        unknown = set(by_path) - allowed
        if unknown:
            raise ValueError("provider job snapshot contains an unknown or extra object")

        done_record = by_path.get(done_path)
        acceptance_preexists = acceptance_path in by_path
        if done_record is None:
            if terminal["terminal_reason"] not in FAILURE_REASONS:
                raise ValueError("DONE is missing for a claimed complete attempt")
            if acceptance_preexists:
                raise ValueError("ACCEPTED exists before DONE")
            current_records = [by_path[path] for path in data_paths if path in by_path]
            for record in current_records:
                artifact_raw = _read_bound(store, record)
                if _exposes_hidden(_json(artifact_raw, "partial result artifact")):
                    raise ValueError("partial result artifact exposed hidden truth")
            prepared.append(
                {
                    "selected": selected,
                    "terminal": terminal,
                    "listed": listed,
                    "current_records": current_records,
                    "historical_count": len(set(by_path) & historical_partial_paths),
                    "historical_done_count": len(
                        set(by_path) & historical_done_paths
                    ),
                    "done": None,
                    "transition_done": None,
                    "raw_by_path": {},
                    "acceptance_path": acceptance_path,
                    "acceptance_record": by_path.get(acceptance_path),
                    "job_prefix": job_prefix,
                }
            )
            continue
        if terminal["terminal_reason"] != "done_observed":
            raise ValueError("DONE conflicts with pre-DONE terminal classification")
        if set(data_paths) - set(by_path):
            raise ValueError("DONE was observed before every artifact")
        current_records = [by_path[path] for path in data_paths]
        data_raw = {row["path"]: _read_bound(store, row) for row in current_records}
        done_raw = _read_bound(store, done_record)
        done, transition_done = _validate_done(
            raw=done_raw,
            done_record=done_record,
            data_records={row["path"]: row for row in current_records},
            data_raw=data_raw,
            plan=plan,
            ledger=ledger,
            resume=resume,
            selected=selected,
            bootstrap=bootstraps[job],
            work=work,
        )
        prepared.append(
            {
                "selected": selected,
                "terminal": terminal,
                "listed": listed,
                "current_records": current_records,
                "historical_count": len(set(by_path) & historical_partial_paths),
                "historical_done_count": len(set(by_path) & historical_done_paths),
                "done": done,
                "transition_done": transition_done,
                "raw_by_path": {**data_raw, done_path: done_raw},
                "acceptance_path": acceptance_path,
                "acceptance_record": by_path.get(acceptance_path),
                "job_prefix": job_prefix,
            }
        )

    # Re-read every selected job prefix after the full wave validation and
    # before the first controller write.  A late old-attempt DONE, an added
    # object, or a generation change therefore cannot race into ACCEPTED.
    for item in prepared:
        relisted = [
            _record(row)
            for row in store.list_prefix(prefix=item["job_prefix"])
        ]
        if len({row["path"] for row in relisted}) != len(relisted):
            raise ValueError("provider job re-read contains a duplicate path")
        if sorted(relisted, key=lambda row: row["path"]) != sorted(
            item["listed"], key=lambda row: row["path"]
        ):
            raise ValueError("provider job snapshot changed before ACCEPTED")

    # Candidate/reference is the scientific unit.  A lone valid DONE is kept
    # as immutable evidence, but cannot create ACCEPTED or enter the ledger.
    # Both sides must be exact-created in the same selected attempt lane and
    # must carry valid DONE under the common launch mapping.
    prepared_by_job = {
        item["selected"]["job_id"]: item for item in prepared
    }
    for pair in selected_pairs:
        candidate = prepared_by_job[pair["candidate_job_id"]]
        reference = prepared_by_job[pair["reference_job_id"]]
        pair_accepted = all(
            item["terminal"]["exact_instance_created"] is True
            and item["done"] is not None
            for item in (candidate, reference)
        )
        if not pair_accepted and any(
            item["acceptance_record"] is not None
            for item in (candidate, reference)
        ):
            raise ValueError("ACCEPTED exists for an incomplete candidate/reference pair")
        outcome = (
            "accepted_both_exact_created_valid_done"
            if pair_accepted
            else "failed_both_unaccepted_resume"
        )
        for item, peer in ((candidate, reference), (reference, candidate)):
            item["pair_id"] = pair["pair_id"]
            item["peer_job_id"] = peer["selected"]["job_id"]
            item["pair_accepted"] = pair_accepted
            item["pair_atomic_outcome"] = outcome

    # Every selected job has now been checked.  Only now may controller-owned
    # ACCEPTED markers be created.  Existing identical markers make a rerun
    # idempotent; any differing marker fails closed.
    acceptance_rows: dict[str, dict[str, Any]] = {}
    acceptance_raw: dict[str, bytes] = {}
    acceptance_created: dict[str, bool] = {}
    for item in prepared:
        if not item["pair_accepted"]:
            continue
        selected = item["selected"]
        done_record = item["transition_done"]
        acceptance_value = _acceptance_value(
            plan=plan,
            ledger=ledger,
            resume=resume,
            selected=selected,
            terminal=item["terminal"],
            done=item["done"],
            done_record=done_record,
            artifact_records=[*item["current_records"], done_record],
        )
        record, raw, created = _create_or_read_acceptance(
            store=store,
            path=item["acceptance_path"],
            value=acceptance_value,
            expected_existing_record=item["acceptance_record"],
        )
        job = selected["job_id"]
        acceptance_rows[job] = {
            "job_id": job,
            "source_role": selected["source_role"],
            "attempt_id": selected["attempt_id"],
            "path": record["path"],
            "generation": record["generation"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "done_generation": done_record["generation"],
            "done_sha256": done_record["sha256"],
            "create_only": True,
        }
        acceptance_raw[record["path"]] = raw
        acceptance_created[job] = created

    prior_done = deepcopy(previous["done_objects"])
    prior_acceptance = deepcopy(previous["acceptance_records"])
    attempt_results: list[dict[str, Any]] = []
    accepted_jobs: list[str] = []
    failed_jobs: list[str] = []
    for item in prepared:
        selected = item["selected"]
        terminal = item["terminal"]
        job = selected["job_id"]
        accepted = item["pair_accepted"]
        histories[job].append(
            {
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "launch_receipt_sha256": terminal["launch_receipt_sha256"],
                "terminal_status": "accepted" if accepted else "failed",
            }
        )
        if accepted:
            accepted_jobs.append(job)
            prior_done.append(item["transition_done"])
            prior_acceptance.append(acceptance_rows[job])
        else:
            failed_jobs.append(job)
        # Controller ACCEPTED may already exist on an idempotent rerun.  It is
        # intentionally excluded from the worker-output snapshot so the same
        # immutable attempt yields the same snapshot digest before and after
        # the controller marker was created.
        snapshot = sorted(
            [
                row
                for row in item["listed"]
                if row["path"] != item["acceptance_path"]
            ],
            key=lambda row: row["path"],
        )
        attempt_results.append(
            {
                "job_id": job,
                "source_role": selected["source_role"],
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "launch_receipt_sha256": terminal["launch_receipt_sha256"],
                "wave_launch_receipt_sha256": terminal[
                    "wave_launch_receipt_sha256"
                ],
                "launch_mapping_sha256": terminal["launch_mapping_sha256"],
                "lifecycle_proof_sha256": terminal[
                    "lifecycle_proof_sha256"
                ],
                "gce_absence_receipt_sha256": terminal[
                    "gce_absence_receipt_sha256"
                ],
                "exact_instance_created": terminal["exact_instance_created"],
                "terminal_reason": terminal["terminal_reason"],
                "terminal_status": "accepted" if accepted else "failed",
                "pair_id": item["pair_id"],
                "peer_job_id": item["peer_job_id"],
                "pair_atomic_outcome": item["pair_atomic_outcome"],
                "valid_done_observed": item["done"] is not None,
                "job_snapshot_object_count": len(snapshot),
                "current_attempt_object_count": len(item["current_records"])
                + (1 if item["done"] is not None else 0),
                "historical_partial_object_count": item["historical_count"],
                "historical_done_object_count": item[
                    "historical_done_count"
                ],
                "job_snapshot_sha256": wave_v2.canonical_sha256(snapshot),
                "done_generation": (
                    item["transition_done"]["generation"]
                    if item["done"] is not None
                    else None
                ),
                "done_sha256": (
                    item["transition_done"]["sha256"]
                    if item["done"] is not None
                    else None
                ),
                "acceptance_generation": (
                    acceptance_rows[job]["generation"] if accepted else None
                ),
                "acceptance_sha256": (
                    acceptance_rows[job]["sha256"] if accepted else None
                ),
                "acceptance_create_performed": (
                    acceptance_created[job] if accepted else False
                ),
                "local_materialized": accepted,
            }
        )

    transition = wave_v2.build_observed_transition(
        plan,
        project_id=project_id,
        zone=zone,
        observed_at_utc=observed_at_utc,
        previous_transition_digest=previous["transition_digest"],
        attempt_history=[
            {
                "job_id": job,
                "source_role": metadata[job]["source_role"],
                "attempts": histories[job],
            }
            for job in plan["coverage"]["job_ids"]
        ],
        owned_vms=(),
        done_objects=prior_done,
        acceptance_records=prior_acceptance,
        readback_source=readback_source,
    )
    consumed = [
        *ledger["consumed_transition_digests"],
        previous["transition_digest"],
    ]
    next_ledger = wave_v2.build_attempt_ledger(
        plan,
        transitions=[*ledger["transitions"], transition],
        consumed_transition_digests=consumed,
    )
    try:
        next_resume = wave_v2.build_resume_plan(
            plan, attempt_ledger=next_ledger
        )
        resume_blocker = None
    except ValueError as exc:
        if "exhausted its two-attempt budget" not in str(exc):
            raise
        next_resume = None
        resume_blocker = "two_attempt_budget_exhausted_fail_closed"

    expected_inventory: dict[str, Any] | None = None
    observed_inventory: dict[str, Any] | None = None
    final_raw: dict[str, bytes] = {}
    all_complete = (
        next_resume is not None and next_resume["all_jobs_complete"] is True
    )
    if all_complete:
        expected_inventory, observed_inventory, final_raw = _final_inventory(
            plan=plan,
            ledger=next_ledger,
            observed_at_utc=observed_at_utc,
            store=store,
        )

    root = _prepare_destination(destination)
    materialize: dict[str, bytes] = {}
    if all_complete:
        materialize.update(final_raw)
    else:
        for item in prepared:
            if not item["pair_accepted"]:
                continue
            materialize.update(item["raw_by_path"])
            materialize.update(
                {
                    item["acceptance_path"]: acceptance_raw[
                        item["acceptance_path"]
                    ]
                }
            )
    done_paths = {row["path"] for row in transition["done_objects"]}
    acceptance_paths = {row["path"] for row in transition["acceptance_records"]}
    data_paths = sorted(set(materialize) - done_paths - acceptance_paths)
    ordered_paths = [
        *data_paths,
        *sorted(set(materialize) & done_paths),
        *sorted(set(materialize) & acceptance_paths),
    ]
    for path in ordered_paths:
        _write_immutable_bytes(root, path, materialize[path])

    if resume_blocker is not None:
        status = "attempt_budget_exhausted_fail_closed"
    elif all_complete:
        status = "all_jobs_complete_exact_inventory_accepted"
    elif failed_jobs:
        status = "wave_results_recorded_resume_required"
    else:
        status = "wave_results_recorded_next_wave_ready"
    nonaccepted_rescheduled = False
    if failed_jobs and next_resume is not None:
        selected_next = {row["job_id"] for row in next_resume["selected_attempts"]}
        nonaccepted_rescheduled = set(failed_jobs).issubset(selected_next)
        if not nonaccepted_rescheduled:
            raise ValueError("non-accepted job did not return to resume selection")
    lifecycle_body: dict[str, Any] = {
        "schema": (
            "hu_m31_t3_step6d_full100_wave_transition_lifecycle_binding_v2"
        ),
        "observed_transition_digest": transition["transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "wave_launch_receipt_sha256": attempt_results[0][
            "wave_launch_receipt_sha256"
        ],
        "launch_mapping_sha256": attempt_results[0]["launch_mapping_sha256"],
        "accepted_launch_receipt_sha256s": [
            row["launch_receipt_sha256"]
            for row in attempt_results
            if row["terminal_status"] == "accepted"
        ],
        "lifecycle_proof_sha256": lifecycle_proof["proof_sha256"],
        "controller_lifecycle_receipt_sha256": lifecycle_proof[
            "lifecycle_receipt_sha256"
        ],
        "gce_absence_receipt_sha256": lifecycle_proof[
            "gce_absence_receipt"
        ]["receipt_sha256"],
        "worker_iam_cleanup_receipt_sha256": cleanup["receipt_sha256"],
        "all_selected_instances_absent": True,
        "all_selected_boot_disks_absent": True,
        "worker_iam_bindings_absent": True,
    }
    transition_lifecycle_binding = {
        **lifecycle_body,
        "binding_sha256": wave_v2.canonical_sha256(lifecycle_body),
    }
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "previous_attempt_ledger_sha256": ledger["ledger_sha256"],
        "input_resume_plan_sha256": resume["resume_sha256"],
        "project_id": project_id,
        "zone": zone,
        "observed_at_utc": observed_at_utc,
        "readback_source": readback_source,
        "wave_index": resume["resume_wave_index"],
        "selected_attempt_count": len(resume["selected_attempts"]),
        "attempt_results": attempt_results,
        "accepted_job_ids": accepted_jobs,
        "failed_job_ids": failed_jobs,
        "observed_transition": transition,
        "attempt_ledger": next_ledger,
        "next_resume_plan": next_resume,
        "resume_blocker": resume_blocker,
        "expected_artifact_inventory": expected_inventory,
        "observed_artifact_inventory": observed_inventory,
        "local_destination": str(root),
        "materialized_paths": ordered_paths,
        "done_is_only_worker_commit_marker": True,
        "acceptance_create_only": True,
        "pair_atomicity_enforced": True,
        "validated_lifecycle_proof": lifecycle_proof,
        "lifecycle_proof_sha256": lifecycle_proof["proof_sha256"],
        "gce_absence_receipt_sha256": lifecycle_proof[
            "gce_absence_receipt"
        ]["receipt_sha256"],
        "worker_iam_cleanup_receipt": cleanup,
        "worker_iam_cleanup_receipt_sha256": cleanup["receipt_sha256"],
        "worker_iam_bindings_absent": True,
        "transition_lifecycle_binding": transition_lifecycle_binding,
        "nonaccepted_jobs_returned_to_resume": nonaccepted_rescheduled,
        "result_store_protocol_used": True,
        "vm_lifecycle_mutation_performed": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    receipt = {**body, "receipt_sha256": wave_v2.canonical_sha256(body)}
    return validate_receiver_receipt(plan, receipt)


def validate_receiver_receipt(
    wave_plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    if not isinstance(value, Mapping):
        raise ValueError("receiver receipt is not an object")
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _RECEIPT_KEYS, "receiver receipt")
    digest = receipt.pop("receipt_sha256", None)
    if digest != wave_v2.canonical_sha256(receipt):
        raise ValueError("receiver receipt digest changed")
    receipt["receipt_sha256"] = digest
    attempts = receipt.get("attempt_results")
    if not isinstance(attempts, list):
        raise ValueError("receiver attempt results are missing")
    for row in attempts:
        if not isinstance(row, Mapping):
            raise ValueError("receiver attempt result is not an object")
        _exact_keys(row, _ATTEMPT_RESULT_KEYS, "receiver attempt result")
    accepted_attempts = [
        row for row in attempts if row["terminal_status"] == "accepted"
    ]
    failed_attempts = [
        row for row in attempts if row["terminal_status"] == "failed"
    ]
    if len(attempts) % 2:
        raise ValueError("receiver attempt results contain a half-pair")
    pair_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in attempts:
        pair_rows.setdefault(row["pair_id"], []).append(row)
        if (
            row["terminal_status"] not in {"accepted", "failed"}
            or type(row["exact_instance_created"]) is not bool
            or type(row["valid_done_observed"]) is not bool
            or row["lifecycle_proof_sha256"]
            != receipt["lifecycle_proof_sha256"]
            or row["gce_absence_receipt_sha256"]
            != receipt["gce_absence_receipt_sha256"]
            or (row["done_generation"] is None)
            is not (row["done_sha256"] is None)
            or row["valid_done_observed"]
            is not (
                row["done_generation"] is not None
                and row["done_sha256"] is not None
            )
        ):
            raise ValueError("receiver attempt result evidence changed")
        if row["terminal_status"] == "accepted":
            if (
                row["pair_atomic_outcome"]
                != "accepted_both_exact_created_valid_done"
                or row["exact_instance_created"] is not True
                or row["valid_done_observed"] is not True
                or row["acceptance_generation"] is None
                or row["acceptance_sha256"] is None
                or row["local_materialized"] is not True
            ):
                raise ValueError("accepted pair evidence changed")
        elif (
            row["pair_atomic_outcome"] != "failed_both_unaccepted_resume"
            or row["acceptance_generation"] is not None
            or row["acceptance_sha256"] is not None
            or row["acceptance_create_performed"] is not False
            or row["local_materialized"] is not False
        ):
            raise ValueError("failed pair evidence changed")
    for pair_id, rows in pair_rows.items():
        if (
            len(rows) != 2
            or {row["source_role"] for row in rows}
            != {"candidate", "reference"}
            or {row["peer_job_id"] for row in rows}
            != {row["job_id"] for row in rows}
            or any(row["peer_job_id"] == row["job_id"] for row in rows)
            or len({row["attempt_id"] for row in rows}) != 1
            or len({row["terminal_status"] for row in rows}) != 1
            or len({row["pair_atomic_outcome"] for row in rows}) != 1
            or len({row["wave_launch_receipt_sha256"] for row in rows}) != 1
            or len({row["launch_mapping_sha256"] for row in rows}) != 1
        ):
            raise ValueError(f"receiver pair {pair_id!r} is not atomic")
    transition = wave_v2.validate_observed_transition(
        plan, receipt["observed_transition"]
    )
    ledger = wave_v2.validate_attempt_ledger(plan, receipt["attempt_ledger"])
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["run_name"] != plan["run_name"]
        or receipt["execution_identity_sha256"]
        != plan["execution_identity_sha256"]
        or receipt["wave_plan_sha256"] != plan["schedule_sha256"]
        or receipt["selected_attempt_count"] != len(attempts)
        or ledger["transitions"][-1] != transition
        or receipt["done_is_only_worker_commit_marker"] is not True
        or receipt["acceptance_create_only"] is not True
        or receipt["pair_atomicity_enforced"] is not True
        or receipt["worker_iam_bindings_absent"] is not True
        or receipt["result_store_protocol_used"] is not True
        or receipt["vm_lifecycle_mutation_performed"] is not False
        or receipt["training_eligible"] is not False
        or receipt["quality_evidence"] is not False
        or receipt["promotion_evidence"] is not False
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("receiver receipt contract changed")
    if receipt["accepted_job_ids"] != [row["job_id"] for row in accepted_attempts]:
        raise ValueError("receiver accepted job order changed")
    if receipt["failed_job_ids"] != [row["job_id"] for row in failed_attempts]:
        raise ValueError("receiver failed job order changed")

    transitions = ledger["transitions"]
    consumed = ledger["consumed_transition_digests"]
    if len(transitions) < 2 or not consumed:
        raise ValueError("receiver ledger lacks its prior lifecycle context")
    prior_ledger = wave_v2.build_attempt_ledger(
        plan,
        transitions=transitions[:-1],
        consumed_transition_digests=consumed[:-1],
    )
    prior_resume = wave_v2.build_resume_plan(
        plan, attempt_ledger=prior_ledger
    )
    if (
        prior_ledger["ledger_sha256"]
        != receipt["previous_attempt_ledger_sha256"]
        or prior_resume["resume_sha256"] != receipt["input_resume_plan_sha256"]
    ):
        raise ValueError("receiver prior lifecycle context changed")
    proof = validate_lifecycle_proof(
        plan,
        prior_ledger,
        prior_resume,
        receipt["validated_lifecycle_proof"],
        receiver_observed_at_utc=receipt["observed_at_utc"],
    )
    proof_by_job = {
        row["job_id"]: row for row in proof["selected_instance_mapping"]
    }
    if (
        proof["proof_sha256"] != receipt["lifecycle_proof_sha256"]
        or proof["gce_absence_receipt"]["receipt_sha256"]
        != receipt["gce_absence_receipt_sha256"]
        or proof["gce_absence_receipt"].get("project")
        != receipt["project_id"]
        or proof["gce_absence_receipt"].get("zone") != receipt["zone"]
        or set(proof_by_job) != {row["job_id"] for row in attempts}
        or any(
            proof_by_job[row["job_id"]]["launch_receipt_sha256"]
            != row["launch_receipt_sha256"]
            or proof_by_job[row["job_id"]]["exact_instance_created"]
            is not row["exact_instance_created"]
            for row in attempts
        )
    ):
        raise ValueError("receiver validated lifecycle proof binding changed")

    cleanup = receipt["worker_iam_cleanup_receipt"]
    if not isinstance(cleanup, Mapping):
        raise ValueError("receiver worker IAM cleanup receipt is missing")
    cleanup_unsigned = dict(cleanup)
    cleanup_digest = cleanup_unsigned.pop("receipt_sha256", None)
    if (
        cleanup_digest != worker_iam_v2.canonical_sha256(cleanup_unsigned)
        or cleanup_digest != receipt["worker_iam_cleanup_receipt_sha256"]
        or cleanup != proof["worker_iam_cleanup_receipt"]
        or cleanup.get("schema") != worker_iam_v2.CLEANUP_RECEIPT_SCHEMA
        or not _worker_iam_cleanup_semantics_are_exact(
            cleanup,
            exact_binding_count=2 * len(prior_resume["selected_attempts"]),
        )
        or cleanup.get("remaining_targeted_binding_count") != 0
        or cleanup.get("post_cleanup_absence_readback") is not True
        or cleanup.get("cleanup_complete") is not True
    ):
        raise ValueError("receiver worker IAM cleanup binding changed")

    lifecycle = receipt["transition_lifecycle_binding"]
    if not isinstance(lifecycle, Mapping):
        raise ValueError("receiver transition lifecycle binding is missing")
    _exact_keys(
        lifecycle,
        _TRANSITION_LIFECYCLE_BINDING_KEYS,
        "receiver transition lifecycle binding",
    )
    lifecycle_unsigned = dict(lifecycle)
    lifecycle_digest = lifecycle_unsigned.pop("binding_sha256", None)
    accepted_launches = [
        row["launch_receipt_sha256"] for row in accepted_attempts
    ]
    if (
        lifecycle_digest != wave_v2.canonical_sha256(lifecycle_unsigned)
        or lifecycle["schema"]
        != "hu_m31_t3_step6d_full100_wave_transition_lifecycle_binding_v2"
        or lifecycle["observed_transition_digest"]
        != transition["transition_digest"]
        or lifecycle["wave_index"] != receipt["wave_index"]
        or lifecycle["wave_launch_receipt_sha256"]
        != attempts[0]["wave_launch_receipt_sha256"]
        or lifecycle["launch_mapping_sha256"]
        != attempts[0]["launch_mapping_sha256"]
        or lifecycle["accepted_launch_receipt_sha256s"] != accepted_launches
        or lifecycle["lifecycle_proof_sha256"] != proof["proof_sha256"]
        or lifecycle["controller_lifecycle_receipt_sha256"]
        != proof["lifecycle_receipt_sha256"]
        or lifecycle["gce_absence_receipt_sha256"]
        != proof["gce_absence_receipt"]["receipt_sha256"]
        or lifecycle["worker_iam_cleanup_receipt_sha256"] != cleanup_digest
        or lifecycle["all_selected_instances_absent"] is not True
        or lifecycle["all_selected_boot_disks_absent"] is not True
        or lifecycle["worker_iam_bindings_absent"] is not True
    ):
        raise ValueError("receiver transition lifecycle binding changed")
    next_resume = receipt["next_resume_plan"]
    if next_resume is None:
        if (
            receipt["status"] != "attempt_budget_exhausted_fail_closed"
            or receipt["resume_blocker"]
            != "two_attempt_budget_exhausted_fail_closed"
        ):
            raise ValueError("receiver exhausted-resume state changed")
    else:
        wave_v2.validate_resume_plan(plan, ledger, next_resume)
        if receipt["resume_blocker"] is not None:
            raise ValueError("receiver resume blocker changed")
    expected = receipt["expected_artifact_inventory"]
    observed = receipt["observed_artifact_inventory"]
    if expected is None or observed is None:
        if expected is not None or observed is not None:
            raise ValueError("receiver final inventory pair is incomplete")
    else:
        wave_v2.validate_artifact_inventory(plan, ledger, expected)
        wave_v2.validate_observed_artifact_inventory(plan, ledger, observed)
    return receipt


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "DONE_SCHEMA",
    "FAILURE_REASONS",
    "RECEIPT_SCHEMA",
    "ResultStore",
    "TERMINAL_REASONS",
    "ValidatedLifecycleProofAdapterV2",
    "receive_wave_results",
    "validate_acceptance_payload_context",
    "validate_lifecycle_proof",
    "validate_receiver_receipt",
]
