"""CAS-safe bucket IAM contract for one Candidate02 full-100 wave.

This module contains no HTTP, gcloud, credential, service-account creation, or
profile mutation code.  A caller supplies a narrow bucket-policy backend.  The
plan is limited to the attempts selected by a validated 8+8+4 wave resume
state, at most eight distinct service accounts and exactly two conditional
bindings per worker.  Every mutation is a single compare-and-swap attempt and
is followed by an exact readback; retries are intentionally the caller's
problem and require fresh evidence.

Each reader condition is exactly the run control prefix OR the immutable,
content-addressed outer-package prefix.  Each creator condition is exactly one
selected attempt prefix.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from . import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as worker_identity


IAM_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_worker_iam_plan_v2"
PREPARE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_iam_prepare_receipt_v2"
)
INSTALL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_iam_install_receipt_v2"
)
READBACK_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_iam_readback_receipt_v2"
)
CLEANUP_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_iam_cleanup_receipt_v3"
)

PROJECT = worker_identity.PROJECT
BUCKET = "pokerhu-ofc-solver-485418-training"
READER_ROLE = f"projects/{PROJECT}/roles/ofcM31T3ObjectReaderV1"
CREATOR_ROLE = f"projects/{PROJECT}/roles/ofcM31T3ResultCreatorV1"
ROLE_PERMISSIONS = {
    READER_ROLE: ("storage.objects.get",),
    CREATOR_ROLE: ("storage.objects.create",),
}
CONDITION_LIFETIME_SECONDS = 5_400
MAX_CONDITION_LIFETIME_SECONDS = 5_400
MAX_WORKERS_PER_WAVE = 8
MAX_BINDINGS_PER_WAVE = 16
IMMUTABLE_CONTENT_PREFIX_ROOT = "hu-m31-t3/full100-wave-v2/content"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}\.iam\.gserviceaccount\.com$"
)
_POLICY_KEYS = frozenset(
    {"version", "etag", "bindings", "auditConfigs", "kind", "resourceId"}
)
_BINDING_KEYS = frozenset({"role", "members", "condition"})
_CONDITION_KEYS = frozenset({"title", "description", "expression", "location"})
_REQUIRED_CONDITION_KEYS = frozenset({"title", "expression"})
_PLAN_KEYS = frozenset(
    {
        "schema", "status", "project", "bucket", "run_name",
        "execution_identity_sha256", "wave_plan_schedule_sha256", "wave_index",
        "attempt_ledger_sha256", "resume_plan_sha256", "issued_at_utc",
        "immutable_content_prefix", "content_payload_sha256",
        "outer_manifest_sha256", "expires_at_utc",
        "condition_lifetime_seconds", "reader_control_prefix",
        "reader_immutable_content_prefix",
        "workers", "expected_bindings", "worker_count", "exact_binding_count",
        "bucket_iam_only", "service_account_creation_managed",
        "service_account_act_as_binding_managed", "object_list_required",
        "single_cas_attempt_only", "post_mutation_exact_readback_required",
        "cloud_mutated", "current_profile_changed", "plan_sha256",
    }
)
_WORKER_KEYS = frozenset(
    {
        "job_id", "source_role", "attempt_id", "vm_instance_id",
        "service_account", "principal", "reader_control_prefix",
        "reader_immutable_content_prefix", "creator_prefix",
    }
)
_PREPARE_KEYS = frozenset(
    {
        "schema", "status", "iam_plan_sha256", "wave_plan_schedule_sha256",
        "wave_index", "attempt_ledger_sha256", "resume_plan_sha256",
        "immutable_content_prefix", "content_payload_sha256",
        "outer_manifest_sha256",
        "pre_policy_fingerprint_sha256", "pre_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256", "unrelated_binding_count",
        "targeted_binding_count", "prepared", "set_attempt_count",
        "cloud_mutation_performed", "current_profile_changed", "receipt_sha256",
    }
)
_INSTALL_KEYS = frozenset(
    {
        "schema", "status", "iam_plan_sha256", "wave_plan_schedule_sha256",
        "wave_index", "attempt_ledger_sha256", "resume_plan_sha256",
        "immutable_content_prefix", "content_payload_sha256",
        "outer_manifest_sha256",
        "prepare_receipt_sha256", "expected_binding_count",
        "installed_binding_count", "post_policy_fingerprint_sha256",
        "post_policy_etag_sha256", "unrelated_policy_fingerprint_sha256",
        "install_complete", "set_attempt_count", "cloud_mutation_performed",
        "recovered_after_transport_ambiguity", "current_profile_changed",
        "receipt_sha256",
    }
)
_READBACK_KEYS = frozenset(
    {
        "schema", "status", "iam_plan_sha256", "wave_plan_schedule_sha256",
        "wave_index", "attempt_ledger_sha256", "resume_plan_sha256",
        "immutable_content_prefix", "content_payload_sha256",
        "outer_manifest_sha256",
        "install_receipt_sha256", "observed_policy_fingerprint_sha256",
        "observed_policy_etag_sha256", "observed_binding_count",
        "unrelated_policy_fingerprint_sha256", "exact_readback_complete",
        "set_attempt_count", "cloud_mutation_performed", "current_profile_changed",
        "receipt_sha256",
    }
)
_CLEANUP_KEYS = frozenset(
    {
        "schema", "status", "iam_plan_sha256", "wave_plan_schedule_sha256",
        "wave_index", "attempt_ledger_sha256", "resume_plan_sha256",
        "immutable_content_prefix", "content_payload_sha256",
        "outer_manifest_sha256",
        "install_receipt_sha256", "readback_receipt_sha256",
        "removed_binding_count", "remaining_targeted_binding_count",
        "post_policy_fingerprint_sha256", "post_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256", "post_cleanup_absence_readback",
        "cleanup_complete", "set_attempt_count", "cloud_mutation_performed",
        "recovered_after_outcome_ambiguity", "source_mutation_outcome",
        "current_profile_changed",
        "receipt_sha256",
    }
)


class BucketIamBackend(Protocol):
    """Narrow CAS backend; implementations must not retry ``set`` internally."""

    def get_bucket_policy(self) -> Mapping[str, Any]: ...

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]: ...


class WorkerIamCasError(RuntimeError):
    """Stable error a backend may use for a failed policy ETag CAS."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    if digest_field in body:
        raise ValueError("immutable IAM value was already sealed")
    return {**body, digest_field: canonical_sha256(body)}


def _validate_seal(
    value: Mapping[str, Any], *, keys: frozenset[str], digest_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    checked = copy.deepcopy(dict(value))
    if set(checked) != keys:
        raise ValueError(f"{label} fields changed")
    supplied = _sha(checked.pop(digest_field, None), f"{label} digest")
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return copy.deepcopy(dict(value))


def _rfc3339(unix_seconds: int) -> str:
    if isinstance(unix_seconds, bool) or not isinstance(unix_seconds, int):
        raise ValueError("IAM timestamp must be an integer")
    return (
        datetime.fromtimestamp(unix_seconds, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _unix_seconds(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} is not canonical UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not canonical UTC RFC3339") from exc
    seconds = int(parsed.timestamp())
    if parsed.microsecond or parsed.tzinfo != timezone.utc or _rfc3339(seconds) != value:
        raise ValueError(f"{label} is not canonical UTC RFC3339")
    return seconds


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable worker IAM artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable worker IAM artifact exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def default_service_accounts(
    wave_plan: Mapping[str, Any], resume_plan: Mapping[str, Any]
) -> dict[str, str]:
    # Validate the plan here as the legacy function contract promises.  The
    # identity pool itself is deliberately independent of run identity; only
    # the already validated resume order chooses slots 00 through 07.
    wave.validate_wave_plan(wave_plan)
    return worker_identity.service_accounts_for_resume(resume_plan)


def _validated_content_binding(
    *,
    immutable_content_prefix: Any,
    content_payload_sha256: Any,
    outer_manifest_sha256: Any,
) -> tuple[str, str, str]:
    payload_sha = _sha(content_payload_sha256, "outer content payload")
    manifest_sha = _sha(outer_manifest_sha256, "outer manifest")
    expected_prefix = f"{IMMUTABLE_CONTENT_PREFIX_ROOT}/{payload_sha}"
    if immutable_content_prefix != expected_prefix:
        raise ValueError(
            "immutable content prefix must be the exact content-addressed prefix"
        )
    return expected_prefix, payload_sha, manifest_sha


def _validated_condition_prefix(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith("/")
        or value.startswith("/")
        or "//" in value
        or ".." in value
        or "\\" in value
        or '"' in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} is not an exact safe object prefix")
    return value


def _condition(
    *, purpose: str, prefixes: Sequence[str], plan_identity: str, worker_ordinal: int,
    expires_at_utc: str,
) -> dict[str, str]:
    if purpose not in {"reader", "creator"}:
        raise ValueError("IAM condition purpose changed")
    expected_count = 3 if purpose == "reader" else 1
    if len(prefixes) != expected_count or len(set(prefixes)) != expected_count:
        raise ValueError("IAM condition prefix cardinality changed")
    checked_prefixes = [
        _validated_condition_prefix(prefix, f"{purpose} condition prefix")
        for prefix in prefixes
    ]
    clauses = [
        f'resource.name.startsWith("projects/_/buckets/{BUCKET}/objects/{prefix}")'
        for prefix in checked_prefixes
    ]
    resource_expression = (
        f"({' || '.join(clauses)})" if purpose == "reader" else clauses[0]
    )
    return {
        "title": (
            f"ofc-f100-wiam-{purpose[0]}-{plan_identity[:12]}-{worker_ordinal:02d}"
        ),
        "description": (
            f"plan_identity={plan_identity};purpose={purpose};"
            f"worker_ordinal={worker_ordinal};expires_at_utc={expires_at_utc}"
        ),
        "expression": (
            f'{resource_expression} && request.time < timestamp("{expires_at_utc}")'
        ),
    }


def _validated_inputs(
    wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any], wave_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = wave.validate_wave_plan(wave_plan)
    ledger = wave.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave.validate_resume_plan(plan, ledger, resume_plan)
    if (
        isinstance(wave_index, bool)
        or not isinstance(wave_index, int)
        or wave_index not in range(len(plan["waves"]))
        or resume["resume_wave_index"] != wave_index
        or resume["all_jobs_complete"] is not False
        or not 1 <= len(resume["selected_attempts"]) <= MAX_WORKERS_PER_WAVE
        or any(
            selected["job_id"] not in plan["waves"][wave_index]["job_ids"]
            for selected in resume["selected_attempts"]
        )
    ):
        raise ValueError("IAM wave index does not match the observed resume selection")
    return plan, ledger, resume


def _build_plan_core(
    *, plan: Mapping[str, Any], ledger: Mapping[str, Any],
    resume: Mapping[str, Any], wave_index: int,
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    service_accounts_by_job: Mapping[str, str], issued_at_unix_seconds: int,
) -> dict[str, Any]:
    content_prefix, content_sha, manifest_sha = _validated_content_binding(
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
    )
    issued = _rfc3339(issued_at_unix_seconds)
    expires = _rfc3339(issued_at_unix_seconds + CONDITION_LIFETIME_SECONDS)
    selected = resume["selected_attempts"]
    expected_jobs = [row["job_id"] for row in selected]
    if set(service_accounts_by_job) != set(expected_jobs):
        raise ValueError("service-account map must exactly cover selected wave jobs")
    accounts = list(service_accounts_by_job.values())
    if (
        len(accounts) != len(set(accounts))
        or any(
            not isinstance(account, str)
            or _SERVICE_ACCOUNT.fullmatch(account) is None
            for account in accounts
        )
    ):
        raise ValueError("wave worker service accounts are invalid or duplicated")
    plan_identity = canonical_sha256(
        {
            "schema": "hu_m31_t3_step6d_full100_wave_worker_iam_identity_v1",
            "schedule_sha256": plan["schedule_sha256"],
            "wave_index": wave_index,
            "attempt_ledger_sha256": ledger["ledger_sha256"],
            "resume_plan_sha256": resume["resume_sha256"],
            "immutable_content_prefix": content_prefix,
            "content_payload_sha256": content_sha,
            "outer_manifest_sha256": manifest_sha,
            "service_accounts_by_job": dict(service_accounts_by_job),
            "issued_at_utc": issued,
        }
    )
    control_prefix = f"{plan['artifact_contract']['prefix']}/control/"
    content_reader_prefix = f"{content_prefix}/"
    _validated_condition_prefix(control_prefix, "reader control prefix")
    _validated_condition_prefix(content_reader_prefix, "reader content prefix")
    workers: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for ordinal, selected_attempt in enumerate(selected):
        job = selected_attempt["job_id"]
        service_account = service_accounts_by_job[job]
        principal = f"serviceAccount:{service_account}"
        worker = {
            "job_id": job,
            "source_role": selected_attempt["source_role"],
            "attempt_id": selected_attempt["attempt_id"],
            "vm_instance_id": selected_attempt["instance_id"],
            "service_account": service_account,
            "principal": principal,
            "reader_control_prefix": control_prefix,
            "reader_immutable_content_prefix": content_reader_prefix,
            "creator_prefix": f"{selected_attempt['artifact_prefix']}/",
        }
        workers.append(worker)
        bindings.extend(
            [
                {
                    "role": READER_ROLE,
                    "members": [principal],
                    "condition": _condition(
                        purpose="reader",
                        prefixes=(
                            control_prefix,
                            content_reader_prefix,
                            worker["creator_prefix"],
                        ),
                        plan_identity=plan_identity, worker_ordinal=ordinal,
                        expires_at_utc=expires,
                    ),
                },
                {
                    "role": CREATOR_ROLE,
                    "members": [principal],
                    "condition": _condition(
                        purpose="creator", prefixes=(worker["creator_prefix"],),
                        plan_identity=plan_identity, worker_ordinal=ordinal,
                        expires_at_utc=expires,
                    ),
                },
            ]
        )
    if (
        len(bindings) != 2 * len(workers)
        or len(bindings) > MAX_BINDINGS_PER_WAVE
    ):
        raise ValueError("worker IAM binding count exceeds the sixteen-binding cap")
    body: dict[str, Any] = {
        "schema": IAM_PLAN_SCHEMA,
        "status": "immutable_wave_iam_plan_cloud_not_mutated",
        "project": PROJECT, "bucket": BUCKET, "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_schedule_sha256": plan["schedule_sha256"],
        "wave_index": wave_index,
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "immutable_content_prefix": content_prefix,
        "content_payload_sha256": content_sha,
        "outer_manifest_sha256": manifest_sha,
        "issued_at_utc": issued, "expires_at_utc": expires,
        "condition_lifetime_seconds": CONDITION_LIFETIME_SECONDS,
        "reader_control_prefix": control_prefix,
        "reader_immutable_content_prefix": content_reader_prefix,
        "workers": workers, "expected_bindings": bindings,
        "worker_count": len(workers), "exact_binding_count": len(bindings),
        "bucket_iam_only": True, "service_account_creation_managed": False,
        "service_account_act_as_binding_managed": False,
        "object_list_required": False, "single_cas_attempt_only": True,
        "post_mutation_exact_readback_required": True,
        "cloud_mutated": False, "current_profile_changed": False,
    }
    return _seal(body, "plan_sha256")


def build_worker_iam_plan(
    *, wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any], wave_index: int,
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    issued_at_unix_seconds: int,
    service_accounts_by_job: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    plan, ledger, resume = _validated_inputs(
        wave_plan, attempt_ledger, resume_plan, wave_index
    )
    accounts = (
        default_service_accounts(plan, resume)
        if service_accounts_by_job is None
        else copy.deepcopy(dict(service_accounts_by_job))
    )
    return validate_worker_iam_plan(
        wave_plan=plan, attempt_ledger=ledger, resume_plan=resume,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=_build_plan_core(
            plan=plan, ledger=ledger, resume=resume, wave_index=wave_index,
            immutable_content_prefix=immutable_content_prefix,
            content_payload_sha256=content_payload_sha256,
            outer_manifest_sha256=outer_manifest_sha256,
            service_accounts_by_job=accounts,
            issued_at_unix_seconds=issued_at_unix_seconds,
        ),
    )


def validate_worker_iam_plan(
    *, wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any], immutable_content_prefix: str,
    content_payload_sha256: str, outer_manifest_sha256: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _validate_seal(
        value, keys=_PLAN_KEYS, digest_field="plan_sha256", label="worker IAM plan"
    )
    plan, ledger, resume = _validated_inputs(
        wave_plan, attempt_ledger, resume_plan, payload.get("wave_index")
    )
    content_prefix, content_sha, manifest_sha = _validated_content_binding(
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
    )
    if (
        payload["schema"] != IAM_PLAN_SCHEMA
        or payload["project"] != PROJECT or payload["bucket"] != BUCKET
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"]
        != plan["execution_identity_sha256"]
        or payload["wave_plan_schedule_sha256"] != plan["schedule_sha256"]
        or payload["attempt_ledger_sha256"] != ledger["ledger_sha256"]
        or payload["resume_plan_sha256"] != resume["resume_sha256"]
        or payload["immutable_content_prefix"] != content_prefix
        or payload["content_payload_sha256"] != content_sha
        or payload["outer_manifest_sha256"] != manifest_sha
        or payload["condition_lifetime_seconds"] != CONDITION_LIFETIME_SECONDS
        or CONDITION_LIFETIME_SECONDS > MAX_CONDITION_LIFETIME_SECONDS
        or payload["cloud_mutated"] is not False
        or payload["current_profile_changed"] is not False
    ):
        raise ValueError("worker IAM plan evidence or safety boundary changed")
    issued = _unix_seconds(payload["issued_at_utc"], "IAM issued_at_utc")
    expires = _unix_seconds(payload["expires_at_utc"], "IAM expires_at_utc")
    if expires - issued != CONDITION_LIFETIME_SECONDS:
        raise ValueError("worker IAM condition lifetime changed")
    workers = payload["workers"]
    bindings = payload["expected_bindings"]
    if (
        not isinstance(workers, list)
        or not isinstance(bindings, list)
        or payload["worker_count"] != len(workers)
        or payload["exact_binding_count"] != len(bindings)
        or len(bindings) != 2 * len(workers)
        or len(bindings) > MAX_BINDINGS_PER_WAVE
    ):
        raise ValueError("worker IAM worker records are missing")
    for worker in workers:
        if not isinstance(worker, Mapping) or set(worker) != _WORKER_KEYS:
            raise ValueError("worker IAM worker record fields changed")
    accounts = {row["job_id"]: row["service_account"] for row in workers}
    rebuilt = _build_plan_core(
        plan=plan, ledger=ledger, resume=resume,
        wave_index=payload["wave_index"], service_accounts_by_job=accounts,
        immutable_content_prefix=content_prefix,
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest_sha,
        issued_at_unix_seconds=issued,
    )
    if payload != rebuilt:
        raise ValueError("worker IAM plan no longer derives from wave evidence")
    return payload


def _validated_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not set(value).issubset(_POLICY_KEYS):
        raise ValueError("bucket IAM policy fields changed")
    policy = copy.deepcopy(dict(value))
    if not {"version", "etag", "bindings"}.issubset(policy):
        raise ValueError("bucket IAM policy header is incomplete")
    policy.setdefault("auditConfigs", [])
    version = policy["version"]
    etag = policy["etag"]
    bindings = policy["bindings"]
    if (
        isinstance(version, bool) or not isinstance(version, int)
        or version not in {0, 1, 3} or not isinstance(etag, str) or not etag
        or not isinstance(bindings, list)
        or not isinstance(policy["auditConfigs"], list)
        or policy.get("kind", "storage#policy") != "storage#policy"
        or policy.get("resourceId", f"projects/_/buckets/{BUCKET}")
        != f"projects/_/buckets/{BUCKET}"
    ):
        raise ValueError("bucket IAM policy header changed")
    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    for raw in bindings:
        if not isinstance(raw, Mapping) or set(raw) not in (
            {"role", "members"}, {"role", "members", "condition"}
        ):
            raise ValueError("bucket IAM binding fields changed")
        role = raw["role"]
        members = raw["members"]
        if (
            not isinstance(role, str) or not role or not isinstance(members, list)
            or not members or len(members) != len(set(members))
            or any(not isinstance(member, str) or not member for member in members)
        ):
            raise ValueError("bucket IAM binding values changed")
        row: dict[str, Any] = {"role": role, "members": list(members)}
        if "condition" in raw:
            condition = raw["condition"]
            if (
                not isinstance(condition, Mapping)
                or not set(condition).issubset(_CONDITION_KEYS)
                or not _REQUIRED_CONDITION_KEYS.issubset(condition)
                or any(not isinstance(condition[key], str) or not condition[key]
                       for key in condition)
                or version != 3
            ):
                raise ValueError("bucket IAM condition changed")
            row["condition"] = copy.deepcopy(dict(condition))
        identity = _binding_identity(row)
        if identity in identities:
            raise ValueError("duplicate bucket IAM role/condition binding")
        identities.add(identity)
        normalized.append(row)
    policy["bindings"] = normalized
    return policy


def _binding_identity(binding: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {"role": binding["role"], "condition": binding.get("condition")}
    )


def _sorted_bindings(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for raw in value:
        row = copy.deepcopy(dict(raw))
        row["members"] = sorted(row["members"])
        rows.append(row)
    return sorted(rows, key=canonical_sha256)


def _policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "version": policy["version"],
            "bindings": _sorted_bindings(policy["bindings"]),
            "auditConfigs": policy["auditConfigs"],
            "kind": policy.get("kind"), "resourceId": policy.get("resourceId"),
        }
    )


def _expected_bindings(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(list(plan["expected_bindings"]))


def _target_principals(plan: Mapping[str, Any]) -> set[str]:
    return {row["principal"] for row in plan["workers"]}


def _collision_sets(plan: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    expected = _expected_bindings(plan)
    return (
        {row["condition"]["title"] for row in expected},
        {row["condition"]["expression"] for row in expected},
    )


def _is_exact(binding: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return dict(binding) == dict(expected)


def _assert_initial_absent(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    principals = _target_principals(plan)
    titles, expressions = _collision_sets(plan)
    expected = _expected_bindings(plan)
    for binding in policy["bindings"]:
        condition = binding.get("condition", {})
        if (
            principals.intersection(binding["members"])
            or condition.get("title") in titles
            or condition.get("expression") in expressions
            or any(_is_exact(binding, row) for row in expected)
        ):
            raise ValueError("stale, duplicate, or drifted wave worker binding exists")


def _assert_installed(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    expected = _expected_bindings(plan)
    counts = [0] * len(expected)
    principals = _target_principals(plan)
    titles, expressions = _collision_sets(plan)
    for binding in policy["bindings"]:
        matches = [_is_exact(binding, row) for row in expected]
        for index, matched in enumerate(matches):
            counts[index] += int(matched)
        condition = binding.get("condition", {})
        targeted = bool(principals.intersection(binding["members"]))
        collides = (
            condition.get("title") in titles
            or condition.get("expression") in expressions
        )
        if (targeted or collides) and not any(matches):
            raise ValueError("installed wave worker binding has an extra or drift")
    if counts != [1] * len(expected):
        raise ValueError("partial add or duplicate wave worker binding detected")


def _assert_absent(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    principals = _target_principals(plan)
    titles, expressions = _collision_sets(plan)
    for binding in policy["bindings"]:
        condition = binding.get("condition", {})
        if (
            principals.intersection(binding["members"])
            or condition.get("title") in titles
            or condition.get("expression") in expressions
        ):
            raise ValueError("post-cleanup worker binding absence is incomplete")


def _unrelated_view(
    policy: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    expected_ids = {_binding_identity(row) for row in _expected_bindings(plan)}
    return {
        "bindings": _sorted_bindings(
            [row for row in policy["bindings"] if _binding_identity(row) not in expected_ids]
        ),
        "auditConfigs": copy.deepcopy(policy["auditConfigs"]),
        "kind": policy.get("kind"), "resourceId": policy.get("resourceId"),
    }


def _unrelated_fingerprint(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    return canonical_sha256(_unrelated_view(policy, plan))


def _receipt_evidence(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "iam_plan_sha256": plan["plan_sha256"],
        "wave_plan_schedule_sha256": plan["wave_plan_schedule_sha256"],
        "wave_index": plan["wave_index"],
        "attempt_ledger_sha256": plan["attempt_ledger_sha256"],
        "resume_plan_sha256": plan["resume_plan_sha256"],
        "immutable_content_prefix": plan["immutable_content_prefix"],
        "content_payload_sha256": plan["content_payload_sha256"],
        "outer_manifest_sha256": plan["outer_manifest_sha256"],
    }


def _validate_receipt_evidence(receipt: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if any(receipt.get(key) != value for key, value in _receipt_evidence(plan).items()):
        raise ValueError("worker IAM receipt belongs to the wrong wave evidence")


def prepare_worker_iam(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    backend: BucketIamBackend,
) -> dict[str, Any]:
    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    policy = _validated_policy(backend.get_bucket_policy())
    _assert_initial_absent(policy, plan)
    unrelated = _unrelated_view(policy, plan)
    return _seal(
        {
            "schema": PREPARE_RECEIPT_SCHEMA,
            "status": "prepared_exact_absence_readback_no_mutation",
            **_receipt_evidence(plan),
            "pre_policy_fingerprint_sha256": _policy_fingerprint(policy),
            "pre_policy_etag_sha256": hashlib.sha256(policy["etag"].encode()).hexdigest(),
            "unrelated_policy_fingerprint_sha256": canonical_sha256(unrelated),
            "unrelated_binding_count": len(unrelated["bindings"]),
            "targeted_binding_count": 0, "prepared": True,
            "set_attempt_count": 0, "cloud_mutation_performed": False,
            "current_profile_changed": False,
        },
        "receipt_sha256",
    )


def _validate_prepare(value: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _validate_seal(
        value, keys=_PREPARE_KEYS, digest_field="receipt_sha256",
        label="worker IAM prepare receipt",
    )
    _validate_receipt_evidence(receipt, plan)
    if (
        receipt["schema"] != PREPARE_RECEIPT_SCHEMA
        or receipt["status"] != "prepared_exact_absence_readback_no_mutation"
        or receipt["targeted_binding_count"] != 0 or receipt["prepared"] is not True
        or receipt["set_attempt_count"] != 0
        or receipt["cloud_mutation_performed"] is not False
        or receipt["current_profile_changed"] is not False
        or isinstance(receipt["unrelated_binding_count"], bool)
        or not isinstance(receipt["unrelated_binding_count"], int)
        or receipt["unrelated_binding_count"] < 0
    ):
        raise ValueError("worker IAM prepare receipt changed")
    for key in (
        "pre_policy_fingerprint_sha256", "pre_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
    ):
        _sha(receipt[key], f"prepare {key}")
    return receipt


def _mutation_policy(
    policy: Mapping[str, Any], plan: Mapping[str, Any], *, install: bool
) -> dict[str, Any]:
    result = copy.deepcopy(dict(policy))
    expected = _expected_bindings(plan)
    if install:
        _assert_initial_absent(policy, plan)
        result["bindings"] = [*result["bindings"], *expected]
    else:
        _assert_installed(policy, plan)
        expected_ids = {_binding_identity(row) for row in expected}
        result["bindings"] = [
            row for row in result["bindings"]
            if _binding_identity(row) not in expected_ids
        ]
    result["version"] = 3
    return result


def install_worker_iam(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    prepare_receipt: Mapping[str, Any], backend: BucketIamBackend,
) -> dict[str, Any]:
    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    current = _validated_policy(backend.get_bucket_policy())
    _assert_initial_absent(current, plan)
    if (
        _policy_fingerprint(current) != prepare["pre_policy_fingerprint_sha256"]
        or hashlib.sha256(current["etag"].encode()).hexdigest()
        != prepare["pre_policy_etag_sha256"]
        or _unrelated_fingerprint(current, plan)
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("bucket IAM policy or ETag changed after prepare")
    desired = _mutation_policy(current, plan, install=True)
    # Exactly one CAS attempt.  No catch/retry is permitted here.
    backend.set_bucket_policy(policy=desired)
    readback = _validated_policy(backend.get_bucket_policy())
    _assert_installed(readback, plan)
    unrelated = _unrelated_fingerprint(readback, plan)
    if unrelated != prepare["unrelated_policy_fingerprint_sha256"]:
        raise ValueError("unrelated bucket IAM policy changed during install")
    return _seal(
        {
            "schema": INSTALL_RECEIPT_SCHEMA,
            "status": "installed_exact_wave_bindings_readback_validated",
            **_receipt_evidence(plan),
            "prepare_receipt_sha256": prepare["receipt_sha256"],
            "expected_binding_count": plan["exact_binding_count"],
            "installed_binding_count": plan["exact_binding_count"],
            "post_policy_fingerprint_sha256": _policy_fingerprint(readback),
            "post_policy_etag_sha256": hashlib.sha256(readback["etag"].encode()).hexdigest(),
            "unrelated_policy_fingerprint_sha256": unrelated,
            "install_complete": True, "set_attempt_count": 1,
            "cloud_mutation_performed": True, "current_profile_changed": False,
            "recovered_after_transport_ambiguity": False,
        },
        "receipt_sha256",
    )


def _validate_install(
    value: Mapping[str, Any], plan: Mapping[str, Any], prepare: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _validate_seal(
        value, keys=_INSTALL_KEYS, digest_field="receipt_sha256",
        label="worker IAM install receipt",
    )
    _validate_receipt_evidence(receipt, plan)
    if (
        receipt["schema"] != INSTALL_RECEIPT_SCHEMA
        or receipt["status"]
        != (
            "recovered_exact_wave_bindings_after_transport_ambiguity"
            if receipt["recovered_after_transport_ambiguity"] is True
            else "installed_exact_wave_bindings_readback_validated"
        )
        or receipt["prepare_receipt_sha256"] != prepare["receipt_sha256"]
        or receipt["expected_binding_count"] != plan["exact_binding_count"]
        or receipt["installed_binding_count"] != plan["exact_binding_count"]
        or receipt["install_complete"] is not True
        or receipt["set_attempt_count"] != 1
        or receipt["cloud_mutation_performed"] is not True
        or type(receipt["recovered_after_transport_ambiguity"]) is not bool
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("worker IAM install receipt changed")
    for key in (
        "post_policy_fingerprint_sha256", "post_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
    ):
        _sha(receipt[key], f"install {key}")
    return receipt


def reconcile_install_worker_iam(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str, prepare_receipt: Mapping[str, Any],
    backend: BucketIamBackend,
) -> dict[str, Any]:
    """GET-only proof that an ambiguous install PUT committed exactly once."""

    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    policy = _validated_policy(backend.get_bucket_policy())
    _assert_installed(policy, plan)
    unrelated = _unrelated_fingerprint(policy, plan)
    if unrelated != prepare["unrelated_policy_fingerprint_sha256"]:
        raise ValueError("unrelated bucket IAM policy changed during install recovery")
    return _validate_install(
        _seal(
            {
                "schema": INSTALL_RECEIPT_SCHEMA,
                "status": "recovered_exact_wave_bindings_after_transport_ambiguity",
                **_receipt_evidence(plan),
                "prepare_receipt_sha256": prepare["receipt_sha256"],
                "expected_binding_count": plan["exact_binding_count"],
                "installed_binding_count": plan["exact_binding_count"],
                "post_policy_fingerprint_sha256": _policy_fingerprint(policy),
                "post_policy_etag_sha256": hashlib.sha256(
                    policy["etag"].encode()
                ).hexdigest(),
                "unrelated_policy_fingerprint_sha256": unrelated,
                "install_complete": True,
                "set_attempt_count": 1,
                "cloud_mutation_performed": True,
                "recovered_after_transport_ambiguity": True,
                "current_profile_changed": False,
            },
            "receipt_sha256",
        ),
        plan,
        prepare,
    )


def readback_worker_iam(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    prepare_receipt: Mapping[str, Any], install_receipt: Mapping[str, Any],
    backend: BucketIamBackend,
) -> dict[str, Any]:
    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    install = _validate_install(install_receipt, plan, prepare)
    policy = _validated_policy(backend.get_bucket_policy())
    _assert_installed(policy, plan)
    fingerprint = _policy_fingerprint(policy)
    etag_sha = hashlib.sha256(policy["etag"].encode()).hexdigest()
    unrelated = _unrelated_fingerprint(policy, plan)
    if (
        fingerprint != install["post_policy_fingerprint_sha256"]
        or etag_sha != install["post_policy_etag_sha256"]
        or unrelated != install["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("installed IAM readback drifted after install receipt")
    return _seal(
        {
            "schema": READBACK_RECEIPT_SCHEMA,
            "status": "exact_wave_bindings_fresh_readback_validated",
            **_receipt_evidence(plan),
            "install_receipt_sha256": install["receipt_sha256"],
            "observed_policy_fingerprint_sha256": fingerprint,
            "observed_policy_etag_sha256": etag_sha,
            "observed_binding_count": plan["exact_binding_count"],
            "unrelated_policy_fingerprint_sha256": unrelated,
            "exact_readback_complete": True, "set_attempt_count": 0,
            "cloud_mutation_performed": False, "current_profile_changed": False,
        },
        "receipt_sha256",
    )


def _validate_readback(
    value: Mapping[str, Any], plan: Mapping[str, Any], install: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _validate_seal(
        value, keys=_READBACK_KEYS, digest_field="receipt_sha256",
        label="worker IAM readback receipt",
    )
    _validate_receipt_evidence(receipt, plan)
    if (
        receipt["schema"] != READBACK_RECEIPT_SCHEMA
        or receipt["status"] != "exact_wave_bindings_fresh_readback_validated"
        or receipt["install_receipt_sha256"] != install["receipt_sha256"]
        or receipt["observed_binding_count"] != plan["exact_binding_count"]
        or receipt["exact_readback_complete"] is not True
        or receipt["set_attempt_count"] != 0
        or receipt["cloud_mutation_performed"] is not False
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("worker IAM readback receipt changed")
    for key in (
        "observed_policy_fingerprint_sha256", "observed_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
    ):
        _sha(receipt[key], f"readback {key}")
    return receipt


def validate_prepare_receipt(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str, value: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate an immutable prepare receipt without a cloud backend."""

    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    return _validate_prepare(value, plan)


def validate_install_receipt(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str, prepare_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate an immutable install receipt without a cloud backend."""

    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    return _validate_install(value, plan, prepare)


def validate_readback_receipt(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str, prepare_receipt: Mapping[str, Any],
    install_receipt: Mapping[str, Any], value: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate a fresh-readback receipt without a cloud backend."""

    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    install = _validate_install(install_receipt, plan, prepare)
    return _validate_readback(value, plan, install)


def cleanup_worker_iam(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    prepare_receipt: Mapping[str, Any], install_receipt: Mapping[str, Any],
    readback_receipt: Mapping[str, Any], backend: BucketIamBackend,
) -> dict[str, Any]:
    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    install = _validate_install(install_receipt, plan, prepare)
    readback = _validate_readback(readback_receipt, plan, install)
    current = _validated_policy(backend.get_bucket_policy())
    _assert_installed(current, plan)
    if (
        _policy_fingerprint(current) != readback["observed_policy_fingerprint_sha256"]
        or hashlib.sha256(current["etag"].encode()).hexdigest()
        != readback["observed_policy_etag_sha256"]
        or _unrelated_fingerprint(current, plan)
        != readback["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("bucket IAM policy or ETag changed before cleanup")
    desired = _mutation_policy(current, plan, install=False)
    # Exactly one removal CAS attempt.  No catch/retry is permitted here.
    backend.set_bucket_policy(policy=desired)
    post = _validated_policy(backend.get_bucket_policy())
    _assert_absent(post, plan)
    unrelated = _unrelated_fingerprint(post, plan)
    if unrelated != prepare["unrelated_policy_fingerprint_sha256"]:
        raise ValueError("unrelated bucket IAM policy changed during cleanup")
    return _seal(
        {
            "schema": CLEANUP_RECEIPT_SCHEMA,
            "status": "removed_exact_wave_bindings_post_readback_absent",
            **_receipt_evidence(plan),
            "install_receipt_sha256": install["receipt_sha256"],
            "readback_receipt_sha256": readback["receipt_sha256"],
            "removed_binding_count": plan["exact_binding_count"],
            "remaining_targeted_binding_count": 0,
            "post_policy_fingerprint_sha256": _policy_fingerprint(post),
            "post_policy_etag_sha256": hashlib.sha256(post["etag"].encode()).hexdigest(),
            "unrelated_policy_fingerprint_sha256": unrelated,
            "post_cleanup_absence_readback": True, "cleanup_complete": True,
            "set_attempt_count": 1, "cloud_mutation_performed": True,
            "recovered_after_outcome_ambiguity": False,
            "source_mutation_outcome": "performed",
            "current_profile_changed": False,
        },
        "receipt_sha256",
    )


def validate_cleanup_receipt(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str,
    prepare_receipt: Mapping[str, Any], install_receipt: Mapping[str, Any],
    readback_receipt: Mapping[str, Any], value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    install = _validate_install(install_receipt, plan, prepare)
    readback = _validate_readback(readback_receipt, plan, install)
    receipt = _validate_seal(
        value, keys=_CLEANUP_KEYS, digest_field="receipt_sha256",
        label="worker IAM cleanup receipt",
    )
    _validate_receipt_evidence(receipt, plan)
    recovered = receipt["recovered_after_outcome_ambiguity"]
    if (
        receipt["schema"] != CLEANUP_RECEIPT_SCHEMA
        or receipt["status"]
        != (
            "recovered_exact_wave_binding_absence_after_outcome_ambiguity"
            if recovered is True
            else "removed_exact_wave_bindings_post_readback_absent"
        )
        or receipt["install_receipt_sha256"] != install["receipt_sha256"]
        or receipt["readback_receipt_sha256"] != readback["receipt_sha256"]
        or receipt["removed_binding_count"]
        != (0 if recovered is True else plan["exact_binding_count"])
        or receipt["remaining_targeted_binding_count"] != 0
        or receipt["post_cleanup_absence_readback"] is not True
        or receipt["cleanup_complete"] is not True
        or receipt["set_attempt_count"] != (0 if recovered is True else 1)
        or receipt["cloud_mutation_performed"] is not (recovered is not True)
        or type(recovered) is not bool
        or receipt["source_mutation_outcome"]
        != ("unknown" if recovered is True else "performed")
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("worker IAM cleanup receipt changed")
    for key in (
        "post_policy_fingerprint_sha256", "post_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
    ):
        _sha(receipt[key], f"cleanup {key}")
    return receipt


def reconcile_cleanup_worker_iam(
    *, iam_plan: Mapping[str, Any], wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any],
    immutable_content_prefix: str, content_payload_sha256: str,
    outer_manifest_sha256: str, prepare_receipt: Mapping[str, Any],
    install_receipt: Mapping[str, Any], readback_receipt: Mapping[str, Any],
    backend: BucketIamBackend,
) -> dict[str, Any]:
    """GET-only proof of the exact desired state after an ambiguous cleanup."""

    plan = validate_worker_iam_plan(
        wave_plan=wave_plan, attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        value=iam_plan,
    )
    prepare = _validate_prepare(prepare_receipt, plan)
    install = _validate_install(install_receipt, plan, prepare)
    readback = _validate_readback(readback_receipt, plan, install)
    policy = _validated_policy(backend.get_bucket_policy())
    _assert_absent(policy, plan)
    unrelated = _unrelated_fingerprint(policy, plan)
    if (
        unrelated != prepare["unrelated_policy_fingerprint_sha256"]
        or unrelated != readback["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("unrelated bucket IAM policy changed during cleanup recovery")
    return validate_cleanup_receipt(
        iam_plan=plan,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        value=_seal(
            {
                "schema": CLEANUP_RECEIPT_SCHEMA,
                "status": (
                    "recovered_exact_wave_binding_absence_after_outcome_ambiguity"
                ),
                **_receipt_evidence(plan),
                "install_receipt_sha256": install["receipt_sha256"],
                "readback_receipt_sha256": readback["receipt_sha256"],
                "removed_binding_count": 0,
                "remaining_targeted_binding_count": 0,
                "post_policy_fingerprint_sha256": _policy_fingerprint(policy),
                "post_policy_etag_sha256": hashlib.sha256(
                    policy["etag"].encode()
                ).hexdigest(),
                "unrelated_policy_fingerprint_sha256": unrelated,
                "post_cleanup_absence_readback": True,
                "cleanup_complete": True,
                "set_attempt_count": 0,
                "cloud_mutation_performed": False,
                "recovered_after_outcome_ambiguity": True,
                "source_mutation_outcome": "unknown",
                "current_profile_changed": False,
            },
            "receipt_sha256",
        ),
    )


__all__ = [
    "BUCKET", "BucketIamBackend", "CLEANUP_RECEIPT_SCHEMA", "CREATOR_ROLE",
    "IAM_PLAN_SCHEMA", "IMMUTABLE_CONTENT_PREFIX_ROOT",
    "INSTALL_RECEIPT_SCHEMA", "MAX_BINDINGS_PER_WAVE",
    "MAX_WORKERS_PER_WAVE",
    "PREPARE_RECEIPT_SCHEMA", "PROJECT", "READER_ROLE", "READBACK_RECEIPT_SCHEMA",
    "ROLE_PERMISSIONS", "WorkerIamCasError", "build_worker_iam_plan",
    "canonical_bytes", "canonical_sha256", "cleanup_worker_iam",
    "default_service_accounts", "install_worker_iam", "prepare_worker_iam",
    "readback_worker_iam", "reconcile_cleanup_worker_iam",
    "reconcile_install_worker_iam", "validate_cleanup_receipt",
    "validate_install_receipt", "validate_prepare_receipt",
    "validate_readback_receipt", "validate_worker_iam_plan", "write_json_once",
]
