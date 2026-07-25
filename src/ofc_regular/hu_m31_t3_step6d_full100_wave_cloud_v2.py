"""Fail-closed evidence contracts for the full-100 8+8+4 cloud transport.

This module deliberately contains no GCP implementation.  Cloud observations
are supplied as data and the only stateful operation is an atomic create made
through an injected backend.  The backend boundary makes the one-shot claim
testable without credentials or network access.

The lifecycle is split deliberately:

* prelaunch: live quota + persistent claim + exact planned mapping;
* launch: exact create/readback for the authorized mapping;
* cleanup: delete and absence readback for exactly those owned instances;
* attestation: binds the complete chain and grants no further create right.

No function changes an AI profile.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2


LIVE_QUOTA_HEADROOM_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_live_quota_headroom_v1"
)
LAUNCH_CLAIM_PAYLOAD_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_launch_claim_payload_v1"
)
PERSISTENT_LAUNCH_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_persistent_launch_claim_v1"
)
PLANNED_LAUNCH_MAPPING_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_planned_launch_mapping_v1"
)
PRELAUNCH_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_prelaunch_authorization_v1"
)
ACTUAL_LAUNCH_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_actual_launch_receipt_v1"
)
CLEANUP_ABSENCE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_cleanup_absence_receipt_v1"
)
LIFECYCLE_ATTESTATION_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_lifecycle_attestation_v1"
)

MAX_LIVE_EVIDENCE_AGE_SECONDS = 300
MAX_PRELAUNCH_AUTHORIZATION_SECONDS = 180
QUOTA_METRICS = (
    "c4_family_vcpus",
    "spot_vcpus",
    "global_vcpus",
)
QUOTA_READBACK_SOURCES = frozenset(
    {
        "cloud_quotas_api_and_compute_inventory",
        "fake_backend_fixture",
    }
)
INSTANCE_READBACK_SOURCES = frozenset(
    {
        "compute_instances_and_disks_api",
        "fake_backend_fixture",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_SAFE_ZONE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$")
_UTC_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,255}$")
_ETAG = re.compile(r"^[\x21-\x7e]{1,512}$")


class AtomicCreateBackend(Protocol):
    """Minimal backend needed to prove a persistent create-only claim."""

    def put_if_absent(
        self, *, object_name: str, payload: bytes
    ) -> Mapping[str, Any]: ...

    def get_object(self, *, object_name: str) -> Mapping[str, Any]: ...


def canonical_bytes(value: Any) -> bytes:
    return wave_v2.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return wave_v2.canonical_sha256(value)


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _parse_utc_seconds(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{label} must be UTC with whole seconds")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return parsed


def _seconds_between(earlier: str, later: str) -> int:
    return int(
        (
            _parse_utc_seconds(later, "later time")
            - _parse_utc_seconds(earlier, "earlier time")
        ).total_seconds()
    )


def _require_scope(project_id: Any, zone: Any) -> tuple[str, str]:
    if not isinstance(project_id, str) or _SAFE_PROJECT.fullmatch(project_id) is None:
        raise ValueError("project ID changed")
    if not isinstance(zone, str) or _SAFE_ZONE.fullmatch(zone) is None:
        raise ValueError("zone changed")
    return project_id, zone


def _require_uuid4(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUIDv4")
    return value


def _require_etag(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ETAG.fullmatch(value) is None:
        raise ValueError(f"{label} is missing or malformed")
    return value


def _receipt_with_digest(
    value: Mapping[str, Any], *, digest_field: str
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[digest_field] = canonical_sha256(result)
    return result


def _pop_and_validate_digest(
    value: Mapping[str, Any], *, digest_field: str, label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    payload = deepcopy(dict(value))
    digest = payload.pop(digest_field, None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{label} digest changed")
    return payload, digest


def _context(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    selected = resume["selected_attempts"]
    if (
        resume["all_jobs_complete"] is not False
        or resume["resume_wave_index"] is None
        or not selected
        or len(selected) > wave_v2.MAX_CONCURRENT_VMS
    ):
        raise ValueError("wave launch context has no bounded incomplete wave")
    wave = plan["waves"][resume["resume_wave_index"]]
    if any(row["job_id"] not in wave["job_ids"] for row in selected):
        raise ValueError("selected attempt escaped the earliest incomplete wave")
    return plan, ledger, resume


def _selected_sha(resume: Mapping[str, Any]) -> str:
    return canonical_sha256(resume["selected_attempts"])


def _base_binding(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    immutable_content_sha256: str,
) -> dict[str, Any]:
    content_sha = _require_sha(
        immutable_content_sha256, "immutable outer content digest"
    )
    return {
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "immutable_content_sha256": content_sha,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_index": resume["resume_wave_index"],
        "selected_attempts_sha256": _selected_sha(resume),
        "selected_vm_count": len(resume["selected_attempts"]),
    }


def _validate_base_binding(
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    immutable_content_sha256: str,
    *,
    label: str,
) -> None:
    expected = _base_binding(plan, ledger, resume, immutable_content_sha256)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label} wave or attempt binding changed")


_QUOTA_METRIC_KEYS = frozenset(
    {"metric", "limit_vcpus", "usage_vcpus", "available_vcpus", "readback_complete"}
)
_QUOTA_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "machine_type",
        "vcpus_per_vm",
        "required_vcpus",
        "quota_metrics",
        "minimum_available_vcpus",
        "quota_sufficient",
        "observed_at_utc",
        "expires_at_utc",
        "readback_source",
        "read_only",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)


def build_live_quota_headroom_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    project_id: str,
    zone: str,
    observed_at_utc: str,
    expires_at_utc: str,
    quota_metrics: Sequence[Mapping[str, Any]],
    readback_source: str,
) -> dict[str, Any]:
    """Build a read-only quota receipt; insufficient headroom remains diagnostic."""

    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    project_id, zone = _require_scope(project_id, zone)
    rows = [deepcopy(dict(row)) for row in quota_metrics]
    required = len(resume["selected_attempts"]) * wave_v2.VCPUS_PER_VM
    available = [row.get("available_vcpus") for row in rows]
    available_are_ints = all(
        isinstance(value, int) and not isinstance(value, bool) for value in available
    )
    minimum_available = (
        min(available) if available and available_are_ints else -1
    )
    sufficient = (
        len(rows) == len(QUOTA_METRICS)
        and available_are_ints
        and minimum_available >= required
    )
    core = {
        "schema": LIVE_QUOTA_HEADROOM_SCHEMA,
        "status": (
            "live_quota_headroom_sufficient"
            if sufficient
            else "live_quota_headroom_insufficient"
        ),
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": project_id,
        "zone": zone,
        "machine_type": plan["quota_contract"]["machine_type"],
        "vcpus_per_vm": plan["quota_contract"]["vcpus_per_vm"],
        "required_vcpus": required,
        "quota_metrics": rows,
        "minimum_available_vcpus": minimum_available,
        "quota_sufficient": sufficient,
        "observed_at_utc": observed_at_utc,
        "expires_at_utc": expires_at_utc,
        "readback_source": readback_source,
        "read_only": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        _receipt_with_digest(core, digest_field="receipt_sha256"),
        immutable_content_sha256=immutable_content_sha256,
    )


def validate_live_quota_headroom_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    now_utc: str | None = None,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    payload, digest = _pop_and_validate_digest(
        value, digest_field="receipt_sha256", label="live quota receipt"
    )
    _exact_keys(payload, _QUOTA_KEYS - {"receipt_sha256"}, "live quota receipt")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="live quota receipt",
    )
    _require_scope(payload.get("project_id"), payload.get("zone"))
    rows = payload.get("quota_metrics")
    if not isinstance(rows, list) or len(rows) != len(QUOTA_METRICS):
        raise ValueError("live quota receipt metric set changed")
    checked: list[dict[str, Any]] = []
    for expected_metric, raw in zip(QUOTA_METRICS, rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("quota metric is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _QUOTA_METRIC_KEYS, "quota metric")
        limit = _require_nonnegative_int(row.get("limit_vcpus"), "quota limit")
        usage = _require_nonnegative_int(row.get("usage_vcpus"), "quota usage")
        if (
            row.get("metric") != expected_metric
            or usage > limit
            or row.get("available_vcpus") != limit - usage
            or row.get("readback_complete") is not True
        ):
            raise ValueError("quota metric identity or arithmetic changed")
        checked.append(row)
    required = len(resume["selected_attempts"]) * wave_v2.VCPUS_PER_VM
    minimum = min(row["available_vcpus"] for row in checked)
    sufficient = minimum >= required
    if (
        payload.get("schema") != LIVE_QUOTA_HEADROOM_SCHEMA
        or payload.get("status")
        != (
            "live_quota_headroom_sufficient"
            if sufficient
            else "live_quota_headroom_insufficient"
        )
        or payload.get("machine_type") != plan["quota_contract"]["machine_type"]
        or payload.get("vcpus_per_vm") != wave_v2.VCPUS_PER_VM
        or payload.get("required_vcpus") != required
        or payload.get("minimum_available_vcpus") != minimum
        or payload.get("quota_sufficient") is not sufficient
        or payload.get("readback_source") not in QUOTA_READBACK_SOURCES
        or payload.get("read_only") is not True
        or payload.get("cloud_mutation_performed") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("live quota receipt contract changed")
    lifetime = _seconds_between(payload["observed_at_utc"], payload["expires_at_utc"])
    if lifetime <= 0 or lifetime > MAX_LIVE_EVIDENCE_AGE_SECONDS:
        raise ValueError("live quota receipt lifetime changed")
    if now_utc is not None and not (
        _parse_utc_seconds(payload["observed_at_utc"], "quota observed time")
        <= _parse_utc_seconds(now_utc, "current time")
        <= _parse_utc_seconds(payload["expires_at_utc"], "quota expiry")
    ):
        raise PermissionError("live quota receipt is stale or future-dated")
    payload["quota_metrics"] = checked
    return {**payload, "receipt_sha256": digest}


_CLAIM_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "claim_nonce_sha256",
        "claimed_at_utc",
        "create_only_precondition_generation",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "object_path",
        "claim_payload",
        "claim_payload_sha256",
        "claim_bytes",
        "generation",
        "etag",
        "readback_generation",
        "readback_etag",
        "readback_sha256",
        "readback_bytes",
        "create_only_precondition_generation",
        "create_only",
        "existing_object_accepted",
        "backend_readback_complete",
        "current_profile_changed",
        "receipt_sha256",
    }
)


def _claim_path(plan: Mapping[str, Any], resume: Mapping[str, Any]) -> str:
    wave_id = plan["waves"][resume["resume_wave_index"]]["wave_id"]
    return (
        f"{plan['artifact_contract']['prefix']}/control/waves/{wave_id}/claims/"
        f"{resume['resume_sha256']}.json"
    )


def build_launch_claim_payload(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    project_id: str,
    zone: str,
    claim_nonce: str,
    claimed_at_utc: str,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    project_id, zone = _require_scope(project_id, zone)
    nonce = _require_uuid4(claim_nonce, "claim nonce")
    _parse_utc_seconds(claimed_at_utc, "claim time")
    return {
        "schema": LAUNCH_CLAIM_PAYLOAD_SCHEMA,
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": project_id,
        "zone": zone,
        "claim_nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "claimed_at_utc": claimed_at_utc,
        "create_only_precondition_generation": 0,
    }


def create_persistent_atomic_launch_claim(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    project_id: str,
    zone: str,
    claim_nonce: str,
    claimed_at_utc: str,
    backend: AtomicCreateBackend,
) -> dict[str, Any]:
    """Atomically claim one resume plan; an existing identical object is failure."""

    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    claim = build_launch_claim_payload(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        project_id=project_id,
        zone=zone,
        claim_nonce=claim_nonce,
        claimed_at_utc=claimed_at_utc,
    )
    raw = canonical_bytes(claim)
    object_path = _claim_path(plan, resume)
    created = dict(backend.put_if_absent(object_name=object_path, payload=raw))
    expected_create_keys = {
        "created", "object_name", "generation", "etag", "sha256", "bytes"
    }
    if set(created) != expected_create_keys or created.get("created") is not True:
        raise ValueError("atomic claim backend did not prove a new object")
    generation = created.get("generation")
    if not isinstance(generation, str) or not generation.isdigit() or int(generation) <= 0:
        raise ValueError("atomic claim generation is invalid")
    etag = _require_etag(created.get("etag"), "atomic claim etag")
    expected_sha = hashlib.sha256(raw).hexdigest()
    if (
        created.get("object_name") != object_path
        or created.get("sha256") != expected_sha
        or created.get("bytes") != len(raw)
    ):
        raise ValueError("atomic claim create receipt changed")
    readback = dict(backend.get_object(object_name=object_path))
    expected_read_keys = {
        "object_name", "generation", "etag", "sha256", "bytes", "payload"
    }
    if set(readback) != expected_read_keys or not isinstance(readback.get("payload"), bytes):
        raise ValueError("atomic claim readback is incomplete")
    if (
        readback["object_name"] != object_path
        or readback["generation"] != generation
        or readback["etag"] != etag
        or readback["sha256"] != expected_sha
        or readback["bytes"] != len(raw)
        or readback["payload"] != raw
    ):
        raise ValueError("atomic claim generation, etag, or bytes changed on readback")
    core = {
        "schema": PERSISTENT_LAUNCH_CLAIM_SCHEMA,
        "status": "persistent_atomic_create_only_claim_confirmed",
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": project_id,
        "zone": zone,
        "object_path": object_path,
        "claim_payload": claim,
        "claim_payload_sha256": expected_sha,
        "claim_bytes": len(raw),
        "generation": generation,
        "etag": etag,
        "readback_generation": readback["generation"],
        "readback_etag": readback["etag"],
        "readback_sha256": readback["sha256"],
        "readback_bytes": readback["bytes"],
        "create_only_precondition_generation": 0,
        "create_only": True,
        "existing_object_accepted": False,
        "backend_readback_complete": True,
        "current_profile_changed": False,
    }
    return validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        _receipt_with_digest(core, digest_field="receipt_sha256"),
        immutable_content_sha256=immutable_content_sha256,
    )


def validate_persistent_atomic_launch_claim_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    raw_claim_nonce: str | None = None,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    payload, digest = _pop_and_validate_digest(
        value, digest_field="receipt_sha256", label="persistent launch claim"
    )
    _exact_keys(payload, _CLAIM_KEYS - {"receipt_sha256"}, "persistent launch claim")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="persistent launch claim",
    )
    _require_scope(payload.get("project_id"), payload.get("zone"))
    claim = payload.get("claim_payload")
    if not isinstance(claim, Mapping):
        raise ValueError("persistent launch claim payload is missing")
    claim = deepcopy(dict(claim))
    _exact_keys(claim, _CLAIM_PAYLOAD_KEYS, "launch claim payload")
    _validate_base_binding(
        claim,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="launch claim payload",
    )
    _require_scope(claim.get("project_id"), claim.get("zone"))
    _require_sha(claim.get("claim_nonce_sha256"), "claim nonce digest")
    _parse_utc_seconds(claim.get("claimed_at_utc"), "claim time")
    raw = canonical_bytes(claim)
    claim_sha = hashlib.sha256(raw).hexdigest()
    generation = payload.get("generation")
    if not isinstance(generation, str) or not generation.isdigit() or int(generation) <= 0:
        raise ValueError("persistent launch claim generation is invalid")
    _require_etag(payload.get("etag"), "persistent launch claim etag")
    if (
        payload.get("schema") != PERSISTENT_LAUNCH_CLAIM_SCHEMA
        or payload.get("status") != "persistent_atomic_create_only_claim_confirmed"
        or claim.get("schema") != LAUNCH_CLAIM_PAYLOAD_SCHEMA
        or claim.get("project_id") != payload.get("project_id")
        or claim.get("zone") != payload.get("zone")
        or claim.get("create_only_precondition_generation") != 0
        or payload.get("object_path") != _claim_path(plan, resume)
        or payload.get("claim_payload_sha256") != claim_sha
        or payload.get("claim_bytes") != len(raw)
        or payload.get("readback_generation") != generation
        or payload.get("readback_etag") != payload.get("etag")
        or payload.get("readback_sha256") != claim_sha
        or payload.get("readback_bytes") != len(raw)
        or payload.get("create_only_precondition_generation") != 0
        or payload.get("create_only") is not True
        or payload.get("existing_object_accepted") is not False
        or payload.get("backend_readback_complete") is not True
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("persistent launch claim contract changed")
    if raw_claim_nonce is not None:
        nonce = _require_uuid4(raw_claim_nonce, "claim nonce")
        if hashlib.sha256(nonce.encode("ascii")).hexdigest() != claim["claim_nonce_sha256"]:
            raise ValueError("raw claim nonce does not match persistent claim")
    payload["claim_payload"] = claim
    return {**payload, "receipt_sha256": digest}


_MAPPING_ROW_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "artifact_prefix",
        "machine_type",
        "vcpus",
        "ownership_label",
        "instance_absent",
        "boot_disk_absent",
        "readback_complete",
    }
)
_MAPPING_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "observed_at_utc",
        "expires_at_utc",
        "readback_source",
        "rows",
        "mapping_sha256",
        "one_vm_one_job_one_role",
        "all_selected_instances_absent",
        "all_selected_boot_disks_absent",
        "unlisted_instance_authorized",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_ABSENCE_OBSERVATION_KEYS = frozenset(
    {"instance_id", "instance_absent", "boot_disk_absent", "readback_complete"}
)


def _ownership_label(plan: Mapping[str, Any]) -> str:
    return f"f100wv2-{plan['execution_identity_sha256'][:12]}"


def build_planned_launch_mapping_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    project_id: str,
    zone: str,
    observed_at_utc: str,
    expires_at_utc: str,
    instance_absence_readbacks: Sequence[Mapping[str, Any]],
    readback_source: str,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    project_id, zone = _require_scope(project_id, zone)
    observations: dict[str, dict[str, Any]] = {}
    for raw in instance_absence_readbacks:
        if not isinstance(raw, Mapping):
            raise ValueError("instance absence readback is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _ABSENCE_OBSERVATION_KEYS, "instance absence readback")
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in observations:
            raise ValueError("instance absence readback is duplicated")
        observations[instance_id] = row
    expected_ids = [row["instance_id"] for row in resume["selected_attempts"]]
    if set(observations) != set(expected_ids) or len(observations) != len(expected_ids):
        raise ValueError("instance absence readback does not exactly cover selected attempts")
    rows: list[dict[str, Any]] = []
    for selected in resume["selected_attempts"]:
        observed = observations[selected["instance_id"]]
        if (
            observed.get("instance_absent") is not True
            or observed.get("boot_disk_absent") is not True
            or observed.get("readback_complete") is not True
        ):
            raise ValueError("selected instance or boot disk is not absent")
        rows.append(
            {
                **deepcopy(selected),
                "machine_type": plan["quota_contract"]["machine_type"],
                "vcpus": wave_v2.VCPUS_PER_VM,
                "ownership_label": _ownership_label(plan),
                "instance_absent": True,
                "boot_disk_absent": True,
                "readback_complete": True,
            }
        )
    core = {
        "schema": PLANNED_LAUNCH_MAPPING_SCHEMA,
        "status": "exact_selected_mapping_absence_confirmed",
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": project_id,
        "zone": zone,
        "observed_at_utc": observed_at_utc,
        "expires_at_utc": expires_at_utc,
        "readback_source": readback_source,
        "rows": rows,
        "mapping_sha256": canonical_sha256(rows),
        "one_vm_one_job_one_role": True,
        "all_selected_instances_absent": True,
        "all_selected_boot_disks_absent": True,
        "unlisted_instance_authorized": False,
        "current_profile_changed": False,
    }
    return validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        _receipt_with_digest(core, digest_field="receipt_sha256"),
        immutable_content_sha256=immutable_content_sha256,
    )


def validate_planned_launch_mapping_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    now_utc: str | None = None,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    payload, digest = _pop_and_validate_digest(
        value, digest_field="receipt_sha256", label="planned launch mapping"
    )
    _exact_keys(payload, _MAPPING_KEYS - {"receipt_sha256"}, "planned launch mapping")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="planned launch mapping",
    )
    _require_scope(payload.get("project_id"), payload.get("zone"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(resume["selected_attempts"]):
        raise ValueError("planned launch mapping cardinality changed")
    checked: list[dict[str, Any]] = []
    for selected, raw in zip(resume["selected_attempts"], rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("planned mapping row is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _MAPPING_ROW_KEYS, "planned mapping row")
        expected = {
            **selected,
            "machine_type": plan["quota_contract"]["machine_type"],
            "vcpus": wave_v2.VCPUS_PER_VM,
            "ownership_label": _ownership_label(plan),
            "instance_absent": True,
            "boot_disk_absent": True,
            "readback_complete": True,
        }
        if row != expected:
            raise ValueError("planned mapping is not one VM per selected job and role")
        checked.append(row)
    if len({row["instance_id"] for row in checked}) != len(checked):
        raise ValueError("planned mapping reuses an instance")
    if len({row["job_id"] for row in checked}) != len(checked):
        raise ValueError("planned mapping reuses a job")
    if (
        payload.get("schema") != PLANNED_LAUNCH_MAPPING_SCHEMA
        or payload.get("status") != "exact_selected_mapping_absence_confirmed"
        or payload.get("readback_source") not in INSTANCE_READBACK_SOURCES
        or payload.get("mapping_sha256") != canonical_sha256(checked)
        or payload.get("one_vm_one_job_one_role") is not True
        or payload.get("all_selected_instances_absent") is not True
        or payload.get("all_selected_boot_disks_absent") is not True
        or payload.get("unlisted_instance_authorized") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("planned launch mapping contract changed")
    lifetime = _seconds_between(payload["observed_at_utc"], payload["expires_at_utc"])
    if lifetime <= 0 or lifetime > MAX_LIVE_EVIDENCE_AGE_SECONDS:
        raise ValueError("planned launch mapping lifetime changed")
    if now_utc is not None and not (
        _parse_utc_seconds(payload["observed_at_utc"], "mapping observed time")
        <= _parse_utc_seconds(now_utc, "current time")
        <= _parse_utc_seconds(payload["expires_at_utc"], "mapping expiry")
    ):
        raise PermissionError("planned launch mapping is stale or future-dated")
    payload["rows"] = checked
    return {**payload, "receipt_sha256": digest}


_AUTH_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "quota_receipt_sha256",
        "persistent_claim_receipt_sha256",
        "planned_mapping_receipt_sha256",
        "claim_object_path",
        "claim_generation",
        "claim_etag",
        "claim_nonce_sha256",
        "authorized_instance_ids",
        "authorized_job_ids",
        "authorized_attempt_ids",
        "authorized_create_count",
        "authorized_vcpus",
        "authorized_at_utc",
        "expires_at_utc",
        "explicit_launch_authorized",
        "exact_selected_create_authorized",
        "one_shot",
        "reuse_authorized",
        "unlisted_instance_create_authorized",
        "worker_iam_evidence_required_before_transport_create",
        "worker_iam_evidence_embedded",
        "current_profile_changed",
        "authorization_sha256",
    }
)


def build_prelaunch_authorization(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    raw_claim_nonce: str,
    authorized_at_utc: str,
    expires_at_utc: str,
    explicit_launch_authorized: bool,
) -> dict[str, Any]:
    if explicit_launch_authorized is not True:
        raise PermissionError("explicit wave launch authorization is required")
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    quota = validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        quota_receipt,
        immutable_content_sha256=immutable_content_sha256,
        now_utc=authorized_at_utc,
    )
    claim = validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        persistent_claim_receipt,
        immutable_content_sha256=immutable_content_sha256,
        raw_claim_nonce=raw_claim_nonce,
    )
    mapping = validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=immutable_content_sha256,
        now_utc=authorized_at_utc,
    )
    if quota["quota_sufficient"] is not True:
        raise PermissionError("live quota headroom is insufficient")
    if (
        quota["project_id"] != claim["project_id"]
        or quota["project_id"] != mapping["project_id"]
        or quota["zone"] != claim["zone"]
        or quota["zone"] != mapping["zone"]
    ):
        raise ValueError("prelaunch evidence cloud scope changed")
    auth_lifetime = _seconds_between(authorized_at_utc, expires_at_utc)
    if auth_lifetime <= 0 or auth_lifetime > MAX_PRELAUNCH_AUTHORIZATION_SECONDS:
        raise ValueError("prelaunch authorization lifetime changed")
    auth_expiry = _parse_utc_seconds(expires_at_utc, "authorization expiry")
    if auth_expiry > min(
        _parse_utc_seconds(quota["expires_at_utc"], "quota expiry"),
        _parse_utc_seconds(mapping["expires_at_utc"], "mapping expiry"),
    ):
        raise PermissionError("prelaunch authorization outlives live evidence")
    selected = resume["selected_attempts"]
    claim_nonce_sha = claim["claim_payload"]["claim_nonce_sha256"]
    core = {
        "schema": PRELAUNCH_AUTHORIZATION_SCHEMA,
        "status": "exact_selected_wave_create_authorized_once",
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": quota["project_id"],
        "zone": quota["zone"],
        "quota_receipt_sha256": quota["receipt_sha256"],
        "persistent_claim_receipt_sha256": claim["receipt_sha256"],
        "planned_mapping_receipt_sha256": mapping["receipt_sha256"],
        "claim_object_path": claim["object_path"],
        "claim_generation": claim["generation"],
        "claim_etag": claim["etag"],
        "claim_nonce_sha256": claim_nonce_sha,
        "authorized_instance_ids": [row["instance_id"] for row in selected],
        "authorized_job_ids": [row["job_id"] for row in selected],
        "authorized_attempt_ids": [row["attempt_id"] for row in selected],
        "authorized_create_count": len(selected),
        "authorized_vcpus": len(selected) * wave_v2.VCPUS_PER_VM,
        "authorized_at_utc": authorized_at_utc,
        "expires_at_utc": expires_at_utc,
        "explicit_launch_authorized": True,
        "exact_selected_create_authorized": True,
        "one_shot": True,
        "reuse_authorized": False,
        "unlisted_instance_create_authorized": False,
        "worker_iam_evidence_required_before_transport_create": True,
        "worker_iam_evidence_embedded": False,
        "current_profile_changed": False,
    }
    return validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        value=_receipt_with_digest(core, digest_field="authorization_sha256"),
        raw_claim_nonce=raw_claim_nonce,
        now_utc=authorized_at_utc,
    )


def validate_prelaunch_authorization(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
    raw_claim_nonce: str | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    quota = validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        quota_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    claim = validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        persistent_claim_receipt,
        immutable_content_sha256=immutable_content_sha256,
        raw_claim_nonce=raw_claim_nonce,
    )
    mapping = validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    payload, digest = _pop_and_validate_digest(
        value, digest_field="authorization_sha256", label="prelaunch authorization"
    )
    _exact_keys(payload, _AUTH_KEYS - {"authorization_sha256"}, "prelaunch authorization")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="prelaunch authorization",
    )
    selected = resume["selected_attempts"]
    expected_instances = [row["instance_id"] for row in selected]
    expected_jobs = [row["job_id"] for row in selected]
    expected_attempts = [row["attempt_id"] for row in selected]
    if (
        quota["quota_sufficient"] is not True
        or payload.get("schema") != PRELAUNCH_AUTHORIZATION_SCHEMA
        or payload.get("status") != "exact_selected_wave_create_authorized_once"
        or payload.get("project_id") != quota["project_id"]
        or payload.get("project_id") != claim["project_id"]
        or payload.get("project_id") != mapping["project_id"]
        or payload.get("zone") != quota["zone"]
        or payload.get("zone") != claim["zone"]
        or payload.get("zone") != mapping["zone"]
        or payload.get("quota_receipt_sha256") != quota["receipt_sha256"]
        or payload.get("persistent_claim_receipt_sha256") != claim["receipt_sha256"]
        or payload.get("planned_mapping_receipt_sha256") != mapping["receipt_sha256"]
        or payload.get("claim_object_path") != claim["object_path"]
        or payload.get("claim_generation") != claim["generation"]
        or payload.get("claim_etag") != claim["etag"]
        or payload.get("claim_nonce_sha256")
        != claim["claim_payload"]["claim_nonce_sha256"]
        or payload.get("authorized_instance_ids") != expected_instances
        or payload.get("authorized_job_ids") != expected_jobs
        or payload.get("authorized_attempt_ids") != expected_attempts
        or payload.get("authorized_create_count") != len(selected)
        or payload.get("authorized_vcpus") != len(selected) * wave_v2.VCPUS_PER_VM
        or payload.get("explicit_launch_authorized") is not True
        or payload.get("exact_selected_create_authorized") is not True
        or payload.get("one_shot") is not True
        or payload.get("reuse_authorized") is not False
        or payload.get("unlisted_instance_create_authorized") is not False
        or payload.get("worker_iam_evidence_required_before_transport_create") is not True
        or payload.get("worker_iam_evidence_embedded") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("prelaunch authorization evidence chain changed")
    lifetime = _seconds_between(payload["authorized_at_utc"], payload["expires_at_utc"])
    if lifetime <= 0 or lifetime > MAX_PRELAUNCH_AUTHORIZATION_SECONDS:
        raise ValueError("prelaunch authorization lifetime changed")
    authorized_time = _parse_utc_seconds(
        payload["authorized_at_utc"], "authorization time"
    )
    if not (
        _parse_utc_seconds(quota["observed_at_utc"], "quota observed time")
        <= authorized_time
        <= _parse_utc_seconds(quota["expires_at_utc"], "quota expiry")
        and _parse_utc_seconds(mapping["observed_at_utc"], "mapping observed time")
        <= authorized_time
        <= _parse_utc_seconds(mapping["expires_at_utc"], "mapping expiry")
        and _parse_utc_seconds(
            claim["claim_payload"]["claimed_at_utc"], "claim time"
        )
        <= authorized_time
    ):
        raise PermissionError("prelaunch evidence was not live before authorization")
    if _parse_utc_seconds(payload["expires_at_utc"], "authorization expiry") > min(
        _parse_utc_seconds(quota["expires_at_utc"], "quota expiry"),
        _parse_utc_seconds(mapping["expires_at_utc"], "mapping expiry"),
    ):
        raise PermissionError("prelaunch authorization outlives live evidence")
    if now_utc is not None and not (
        _parse_utc_seconds(payload["authorized_at_utc"], "authorization time")
        <= _parse_utc_seconds(now_utc, "current time")
        <= _parse_utc_seconds(payload["expires_at_utc"], "authorization expiry")
    ):
        raise PermissionError("prelaunch authorization is stale or future-dated")
    return {**payload, "authorization_sha256": digest}


_LAUNCH_ROW_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "artifact_prefix",
        "operation_id",
        "operation_status",
        "instance_status",
        "ownership_label",
        "machine_type",
        "vcpus",
        "instance_readback_complete",
    }
)
_LAUNCH_OBSERVATION_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "operation_id",
        "operation_status",
        "instance_status",
        "ownership_label",
        "machine_type",
        "vcpus",
        "instance_readback_complete",
    }
)
_LAUNCH_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "prelaunch_authorization_sha256",
        "planned_mapping_receipt_sha256",
        "launch_started_at_utc",
        "observed_at_utc",
        "readback_source",
        "rows",
        "created_instance_count",
        "one_vm_one_job_one_role",
        "all_create_operations_done",
        "all_instances_read_back",
        "additional_create_authorized",
        "current_profile_changed",
        "receipt_sha256",
    }
)


def build_actual_launch_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    launch_started_at_utc: str,
    observed_at_utc: str,
    instance_create_readbacks: Sequence[Mapping[str, Any]],
    readback_source: str,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    auth = validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        value=prelaunch_authorization,
        now_utc=launch_started_at_utc,
    )
    mapping = validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    if _seconds_between(launch_started_at_utc, observed_at_utc) < 0:
        raise ValueError("launch readback predates launch")
    observations = [deepcopy(dict(row)) for row in instance_create_readbacks]
    if len(observations) != len(mapping["rows"]):
        raise ValueError("actual launch readback cardinality changed")
    rows: list[dict[str, Any]] = []
    for planned, observed in zip(mapping["rows"], observations, strict=True):
        _exact_keys(observed, _LAUNCH_OBSERVATION_KEYS, "launch observation")
        expected_identity = {
            key: planned[key]
            for key in (
                "job_id", "source_role", "attempt_id", "instance_id",
                "ownership_label", "machine_type", "vcpus",
            )
        }
        if any(observed.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("actual launch escaped the planned job/role mapping")
        if (
            observed.get("operation_status") != "DONE"
            or observed.get("instance_status")
            not in {"PROVISIONING", "STAGING", "RUNNING", "TERMINATED"}
            or observed.get("instance_readback_complete") is not True
            or _OPERATION_ID.fullmatch(str(observed.get("operation_id"))) is None
        ):
            raise ValueError("actual launch operation or instance readback is incomplete")
        rows.append({**observed, "artifact_prefix": planned["artifact_prefix"]})
    core = {
        "schema": ACTUAL_LAUNCH_RECEIPT_SCHEMA,
        "status": "exact_authorized_wave_launch_read_back",
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": auth["project_id"],
        "zone": auth["zone"],
        "prelaunch_authorization_sha256": auth["authorization_sha256"],
        "planned_mapping_receipt_sha256": mapping["receipt_sha256"],
        "launch_started_at_utc": launch_started_at_utc,
        "observed_at_utc": observed_at_utc,
        "readback_source": readback_source,
        "rows": rows,
        "created_instance_count": len(rows),
        "one_vm_one_job_one_role": True,
        "all_create_operations_done": True,
        "all_instances_read_back": True,
        "additional_create_authorized": False,
        "current_profile_changed": False,
    }
    return validate_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=auth,
        value=_receipt_with_digest(core, digest_field="receipt_sha256"),
    )


def validate_actual_launch_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    auth = validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        value=prelaunch_authorization,
    )
    mapping = validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    payload, digest = _pop_and_validate_digest(
        value, digest_field="receipt_sha256", label="actual launch receipt"
    )
    _exact_keys(payload, _LAUNCH_KEYS - {"receipt_sha256"}, "actual launch receipt")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="actual launch receipt",
    )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(mapping["rows"]):
        raise ValueError("actual launch receipt cardinality changed")
    checked: list[dict[str, Any]] = []
    for planned, raw in zip(mapping["rows"], rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("actual launch row is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _LAUNCH_ROW_KEYS, "actual launch row")
        expected = {
            key: planned[key]
            for key in (
                "job_id", "source_role", "attempt_id", "instance_id",
                "artifact_prefix", "ownership_label", "machine_type", "vcpus",
            )
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("actual launch row changed job, role, or instance")
        if (
            row.get("operation_status") != "DONE"
            or row.get("instance_status")
            not in {"PROVISIONING", "STAGING", "RUNNING", "TERMINATED"}
            or row.get("instance_readback_complete") is not True
            or _OPERATION_ID.fullmatch(str(row.get("operation_id"))) is None
        ):
            raise ValueError("actual launch row is incomplete")
        checked.append(row)
    if (
        len({row["instance_id"] for row in checked}) != len(checked)
        or len({row["job_id"] for row in checked}) != len(checked)
        or len({row["operation_id"] for row in checked}) != len(checked)
    ):
        raise ValueError("actual launch is not one operation and VM per job")
    if (
        payload.get("schema") != ACTUAL_LAUNCH_RECEIPT_SCHEMA
        or payload.get("status") != "exact_authorized_wave_launch_read_back"
        or payload.get("project_id") != auth["project_id"]
        or payload.get("zone") != auth["zone"]
        or payload.get("prelaunch_authorization_sha256")
        != auth["authorization_sha256"]
        or payload.get("planned_mapping_receipt_sha256")
        != mapping["receipt_sha256"]
        or payload.get("readback_source") not in INSTANCE_READBACK_SOURCES
        or payload.get("created_instance_count") != len(checked)
        or payload.get("one_vm_one_job_one_role") is not True
        or payload.get("all_create_operations_done") is not True
        or payload.get("all_instances_read_back") is not True
        or payload.get("additional_create_authorized") is not False
        or payload.get("current_profile_changed") is not False
        or not (
            _parse_utc_seconds(auth["authorized_at_utc"], "authorization time")
            <= _parse_utc_seconds(payload["launch_started_at_utc"], "launch time")
            <= _parse_utc_seconds(auth["expires_at_utc"], "authorization expiry")
        )
        or _seconds_between(payload["launch_started_at_utc"], payload["observed_at_utc"])
        < 0
    ):
        raise ValueError("actual launch receipt contract changed")
    payload["rows"] = checked
    return {**payload, "receipt_sha256": digest}


_CLEANUP_ROW_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "ownership_label",
        "delete_operation_id",
        "delete_operation_status",
        "instance_absent",
        "boot_disk_absent",
        "absence_readback_complete",
        "absence_readback_attempt_count",
    }
)
_CLEANUP_OBSERVATION_KEYS = frozenset(
    {
        "instance_id",
        "delete_operation_id",
        "delete_operation_status",
        "instance_absent",
        "boot_disk_absent",
        "absence_readback_complete",
        "absence_readback_attempt_count",
    }
)
_CLEANUP_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "actual_launch_receipt_sha256",
        "owned_namespace_root",
        "owned_root_verified",
        "parent_directory_fsync_completed",
        "observed_at_utc",
        "readback_source",
        "rows",
        "deleted_instance_count",
        "exact_owned_only",
        "all_instances_absent",
        "all_boot_disks_absent",
        "wildcard_delete_used",
        "unrelated_instance_touched",
        "additional_create_authorized",
        "current_profile_changed",
        "receipt_sha256",
    }
)


def build_cleanup_absence_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    actual_launch_receipt: Mapping[str, Any],
    observed_at_utc: str,
    cleanup_readbacks: Sequence[Mapping[str, Any]],
    readback_source: str,
    owned_root_verified: bool,
    parent_directory_fsync_completed: bool,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    launch = validate_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        value=actual_launch_receipt,
    )
    if owned_root_verified is not True or parent_directory_fsync_completed is not True:
        raise PermissionError("owned root and parent-directory fsync are required")
    observations: dict[str, dict[str, Any]] = {}
    for raw in cleanup_readbacks:
        if not isinstance(raw, Mapping):
            raise ValueError("cleanup readback is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _CLEANUP_OBSERVATION_KEYS, "cleanup readback")
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in observations:
            raise ValueError("cleanup readback is duplicated")
        observations[instance_id] = row
    expected_ids = [row["instance_id"] for row in launch["rows"]]
    if set(observations) != set(expected_ids) or len(observations) != len(expected_ids):
        raise ValueError("cleanup does not exactly cover launched owned instances")
    rows: list[dict[str, Any]] = []
    for launched in launch["rows"]:
        observed = observations[launched["instance_id"]]
        if (
            observed.get("delete_operation_status") != "DONE"
            or observed.get("instance_absent") is not True
            or observed.get("boot_disk_absent") is not True
            or observed.get("absence_readback_complete") is not True
            or _OPERATION_ID.fullmatch(str(observed.get("delete_operation_id"))) is None
        ):
            raise ValueError("cleanup delete or absence readback is incomplete")
        _require_positive_int(
            observed.get("absence_readback_attempt_count"),
            "cleanup absence readback attempts",
        )
        rows.append(
            {
                "job_id": launched["job_id"],
                "source_role": launched["source_role"],
                "attempt_id": launched["attempt_id"],
                "instance_id": launched["instance_id"],
                "ownership_label": launched["ownership_label"],
                **{key: value for key, value in observed.items() if key != "instance_id"},
            }
        )
    core = {
        "schema": CLEANUP_ABSENCE_RECEIPT_SCHEMA,
        "status": "exact_owned_cleanup_and_absence_confirmed",
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": launch["project_id"],
        "zone": launch["zone"],
        "actual_launch_receipt_sha256": launch["receipt_sha256"],
        "owned_namespace_root": plan["artifact_contract"]["prefix"],
        "owned_root_verified": True,
        "parent_directory_fsync_completed": True,
        "observed_at_utc": observed_at_utc,
        "readback_source": readback_source,
        "rows": rows,
        "deleted_instance_count": len(rows),
        "exact_owned_only": True,
        "all_instances_absent": True,
        "all_boot_disks_absent": True,
        "wildcard_delete_used": False,
        "unrelated_instance_touched": False,
        "additional_create_authorized": False,
        "current_profile_changed": False,
    }
    return validate_cleanup_absence_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        actual_launch_receipt=launch,
        value=_receipt_with_digest(core, digest_field="receipt_sha256"),
    )


def validate_cleanup_absence_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    actual_launch_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    launch = validate_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        value=actual_launch_receipt,
    )
    payload, digest = _pop_and_validate_digest(
        value, digest_field="receipt_sha256", label="cleanup absence receipt"
    )
    _exact_keys(payload, _CLEANUP_KEYS - {"receipt_sha256"}, "cleanup absence receipt")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="cleanup absence receipt",
    )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(launch["rows"]):
        raise ValueError("cleanup absence receipt cardinality changed")
    checked: list[dict[str, Any]] = []
    for launched, raw in zip(launch["rows"], rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("cleanup row is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _CLEANUP_ROW_KEYS, "cleanup row")
        expected_identity = {
            key: launched[key]
            for key in (
                "job_id", "source_role", "attempt_id", "instance_id", "ownership_label"
            )
        }
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("cleanup row escaped exact launched ownership")
        if (
            row.get("delete_operation_status") != "DONE"
            or row.get("instance_absent") is not True
            or row.get("boot_disk_absent") is not True
            or row.get("absence_readback_complete") is not True
            or _OPERATION_ID.fullmatch(str(row.get("delete_operation_id"))) is None
        ):
            raise ValueError("cleanup row is incomplete")
        _require_positive_int(
            row.get("absence_readback_attempt_count"),
            "cleanup absence readback attempts",
        )
        checked.append(row)
    if len({row["delete_operation_id"] for row in checked}) != len(checked):
        raise ValueError("cleanup delete operation was reused across instances")
    if (
        payload.get("schema") != CLEANUP_ABSENCE_RECEIPT_SCHEMA
        or payload.get("status") != "exact_owned_cleanup_and_absence_confirmed"
        or payload.get("project_id") != launch["project_id"]
        or payload.get("zone") != launch["zone"]
        or payload.get("actual_launch_receipt_sha256") != launch["receipt_sha256"]
        or payload.get("owned_namespace_root") != plan["artifact_contract"]["prefix"]
        or payload.get("owned_root_verified") is not True
        or payload.get("parent_directory_fsync_completed") is not True
        or payload.get("readback_source") not in INSTANCE_READBACK_SOURCES
        or payload.get("deleted_instance_count") != len(checked)
        or payload.get("exact_owned_only") is not True
        or payload.get("all_instances_absent") is not True
        or payload.get("all_boot_disks_absent") is not True
        or payload.get("wildcard_delete_used") is not False
        or payload.get("unrelated_instance_touched") is not False
        or payload.get("additional_create_authorized") is not False
        or payload.get("current_profile_changed") is not False
        or _parse_utc_seconds(payload["observed_at_utc"], "cleanup observed time")
        < _parse_utc_seconds(launch["observed_at_utc"], "launch observed time")
    ):
        raise ValueError("cleanup absence receipt contract changed")
    payload["rows"] = checked
    return {**payload, "receipt_sha256": digest}


_ATTESTATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "quota_receipt_sha256",
        "persistent_claim_receipt_sha256",
        "planned_mapping_receipt_sha256",
        "prelaunch_authorization_sha256",
        "actual_launch_receipt_sha256",
        "cleanup_absence_receipt_sha256",
        "attested_at_utc",
        "prelaunch_chain_complete",
        "actual_launch_exact",
        "cleanup_absence_complete",
        "lifecycle_complete",
        "additional_create_authorized",
        "next_wave_requires_fresh_evidence",
        "current_profile_changed",
        "attestation_sha256",
    }
)


def build_lifecycle_attestation(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    actual_launch_receipt: Mapping[str, Any],
    cleanup_absence_receipt: Mapping[str, Any],
    attested_at_utc: str,
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    quota = validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        quota_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    claim = validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        persistent_claim_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    mapping = validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    auth = validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        value=prelaunch_authorization,
    )
    launch = validate_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=auth,
        value=actual_launch_receipt,
    )
    cleanup = validate_cleanup_absence_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=auth,
        actual_launch_receipt=launch,
        value=cleanup_absence_receipt,
    )
    if _parse_utc_seconds(attested_at_utc, "attestation time") < _parse_utc_seconds(
        cleanup["observed_at_utc"], "cleanup observed time"
    ):
        raise ValueError("lifecycle attestation predates cleanup")
    core = {
        "schema": LIFECYCLE_ATTESTATION_SCHEMA,
        "status": "wave_lifecycle_complete_no_further_create_right",
        **_base_binding(plan, ledger, resume, immutable_content_sha256),
        "project_id": quota["project_id"],
        "zone": quota["zone"],
        "quota_receipt_sha256": quota["receipt_sha256"],
        "persistent_claim_receipt_sha256": claim["receipt_sha256"],
        "planned_mapping_receipt_sha256": mapping["receipt_sha256"],
        "prelaunch_authorization_sha256": auth["authorization_sha256"],
        "actual_launch_receipt_sha256": launch["receipt_sha256"],
        "cleanup_absence_receipt_sha256": cleanup["receipt_sha256"],
        "attested_at_utc": attested_at_utc,
        "prelaunch_chain_complete": True,
        "actual_launch_exact": True,
        "cleanup_absence_complete": True,
        "lifecycle_complete": True,
        "additional_create_authorized": False,
        "next_wave_requires_fresh_evidence": True,
        "current_profile_changed": False,
    }
    return validate_lifecycle_attestation(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=auth,
        actual_launch_receipt=launch,
        cleanup_absence_receipt=cleanup,
        value=_receipt_with_digest(core, digest_field="attestation_sha256"),
    )


def validate_lifecycle_attestation(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    *,
    immutable_content_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    actual_launch_receipt: Mapping[str, Any],
    cleanup_absence_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan, ledger, resume = _context(wave_plan, attempt_ledger, resume_plan)
    quota = validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        quota_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    claim = validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        persistent_claim_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    mapping = validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=immutable_content_sha256,
    )
    auth = validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        value=prelaunch_authorization,
    )
    launch = validate_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=auth,
        value=actual_launch_receipt,
    )
    cleanup = validate_cleanup_absence_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=immutable_content_sha256,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=auth,
        actual_launch_receipt=launch,
        value=cleanup_absence_receipt,
    )
    payload, digest = _pop_and_validate_digest(
        value, digest_field="attestation_sha256", label="lifecycle attestation"
    )
    _exact_keys(payload, _ATTESTATION_KEYS - {"attestation_sha256"}, "lifecycle attestation")
    _validate_base_binding(
        payload,
        plan,
        ledger,
        resume,
        immutable_content_sha256,
        label="lifecycle attestation",
    )
    if (
        payload.get("schema") != LIFECYCLE_ATTESTATION_SCHEMA
        or payload.get("status") != "wave_lifecycle_complete_no_further_create_right"
        or payload.get("project_id") != quota["project_id"]
        or payload.get("zone") != quota["zone"]
        or payload.get("quota_receipt_sha256") != quota["receipt_sha256"]
        or payload.get("persistent_claim_receipt_sha256") != claim["receipt_sha256"]
        or payload.get("planned_mapping_receipt_sha256") != mapping["receipt_sha256"]
        or payload.get("prelaunch_authorization_sha256") != auth["authorization_sha256"]
        or payload.get("actual_launch_receipt_sha256") != launch["receipt_sha256"]
        or payload.get("cleanup_absence_receipt_sha256") != cleanup["receipt_sha256"]
        or payload.get("prelaunch_chain_complete") is not True
        or payload.get("actual_launch_exact") is not True
        or payload.get("cleanup_absence_complete") is not True
        or payload.get("lifecycle_complete") is not True
        or payload.get("additional_create_authorized") is not False
        or payload.get("next_wave_requires_fresh_evidence") is not True
        or payload.get("current_profile_changed") is not False
        or _parse_utc_seconds(payload["attested_at_utc"], "attestation time")
        < _parse_utc_seconds(cleanup["observed_at_utc"], "cleanup observed time")
    ):
        raise ValueError("lifecycle attestation chain changed")
    return {**payload, "attestation_sha256": digest}


# Explicit aliases for controller code that uses receipt-oriented names.
build_launch_authorization_receipt = build_prelaunch_authorization
validate_launch_authorization_receipt = validate_prelaunch_authorization
build_launch_receipt = build_actual_launch_receipt
validate_launch_receipt = validate_actual_launch_receipt


__all__ = [
    "ACTUAL_LAUNCH_RECEIPT_SCHEMA",
    "AtomicCreateBackend",
    "CLEANUP_ABSENCE_RECEIPT_SCHEMA",
    "LIFECYCLE_ATTESTATION_SCHEMA",
    "LIVE_QUOTA_HEADROOM_SCHEMA",
    "PERSISTENT_LAUNCH_CLAIM_SCHEMA",
    "PLANNED_LAUNCH_MAPPING_SCHEMA",
    "PRELAUNCH_AUTHORIZATION_SCHEMA",
    "build_actual_launch_receipt",
    "build_cleanup_absence_receipt",
    "build_launch_authorization_receipt",
    "build_launch_claim_payload",
    "build_launch_receipt",
    "build_lifecycle_attestation",
    "build_live_quota_headroom_receipt",
    "build_planned_launch_mapping_receipt",
    "build_prelaunch_authorization",
    "canonical_bytes",
    "canonical_sha256",
    "create_persistent_atomic_launch_claim",
    "validate_actual_launch_receipt",
    "validate_cleanup_absence_receipt",
    "validate_launch_authorization_receipt",
    "validate_launch_receipt",
    "validate_lifecycle_attestation",
    "validate_live_quota_headroom_receipt",
    "validate_persistent_atomic_launch_claim_receipt",
    "validate_planned_launch_mapping_receipt",
    "validate_prelaunch_authorization",
]
