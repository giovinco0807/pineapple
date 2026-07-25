"""Fixed worker-identity contract for Candidate02 full-100 waves.

The full-100 transport reuses exactly eight deliberately unprivileged service
accounts.  A wave maps its selected attempts, in their already validated
resume order, to slots 00 through 07.  This avoids creating twenty identities
per run and keeps bucket access in the separate, expiring bucket-IAM contract.

This module is intentionally pure: it performs no HTTP, credential, IAM, or
profile mutation.  Provider observations are supplied by a caller and sealed
into self-contained receipts.  The optional setup artifacts describe a
one-time create-only bootstrap, but do not execute it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave


PROJECT = "ofc-solver-485418"
POOL_SIZE = 8
ACCOUNT_IDS = tuple(f"ofc-f100-worker-{index:02d}" for index in range(POOL_SIZE))
ACCOUNT_EMAILS = tuple(
    f"{account_id}@{PROJECT}.iam.gserviceaccount.com"
    for account_id in ACCOUNT_IDS
)
ACT_AS_PERMISSION = "iam.serviceAccounts.actAs"

IDENTITY_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_worker_identity_plan_v2"
INVENTORY_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_inventory_receipt_v2"
)
ACT_AS_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_act_as_receipt_v2"
)
PROJECT_IAM_SCAN_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_project_iam_scan_receipt_v2"
)
SETUP_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_create_only_setup_plan_v2"
)
SETUP_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_create_only_setup_receipt_v2"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNIQUE_ID = re.compile(r"^[1-9][0-9]{5,31}$")
_NONCE = re.compile(r"^[a-z0-9][a-z0-9-]{15,63}$")
_ROLE = re.compile(r"^(?:roles/[A-Za-z0-9_.]+|projects/[a-z0-9-]+/roles/[A-Za-z0-9_.]+)$")

_PLAN_KEYS = frozenset(
    {
        "schema", "status", "project", "run_name",
        "execution_identity_sha256", "wave_plan_schedule_sha256", "wave_index",
        "attempt_ledger_sha256", "resume_plan_sha256", "pool_account_ids",
        "pool_account_emails", "pool_size", "selected_workers", "selected_count",
        "mapping_basis", "fixed_reusable_pool", "run_specific_account_creation",
        "bucket_iam_managed_here", "cloud_mutated", "current_profile_changed",
        "plan_sha256",
    }
)
_SELECTED_KEYS = frozenset(
    {
        "worker_slot", "job_id", "source_role", "attempt_id", "vm_instance_id",
        "account_id", "service_account_email", "service_account_name",
    }
)
_ACCOUNT_OBSERVATION_KEYS = frozenset(
    {
        "account_id", "email", "name", "project_id", "unique_id", "disabled",
        "exists",
    }
)
_INVENTORY_KEYS = frozenset(
    {
        "schema", "status", "identity_plan_sha256", "project", "observed_at_utc",
        "provider_source", "rows", "pool_account_count", "selected_account_count",
        "all_pool_accounts_exist", "all_selected_accounts_exist", "all_enabled",
        "unique_ids_unique", "read_only_observation", "cloud_mutated",
        "current_profile_changed", "receipt_sha256",
    }
)
_ACT_AS_OBSERVATION_KEYS = frozenset(
    {
        "account_id", "email", "name", "http_method", "requested_permissions",
        "granted_permissions",
    }
)
_ACT_AS_KEYS = frozenset(
    {
        "schema", "status", "identity_plan_sha256", "inventory_receipt_sha256",
        "project", "tested_at_utc", "provider_source", "rows", "selected_count",
        "required_permission", "every_selected_account_tested",
        "every_selected_account_grants_exact_act_as", "test_iam_permissions_only",
        "cloud_mutated", "current_profile_changed", "receipt_sha256",
    }
)
_POLICY_REQUIRED_KEYS = frozenset({"version", "etag", "bindings"})
_POLICY_KEYS = _POLICY_REQUIRED_KEYS | {"auditConfigs"}
_BINDING_KEYS = frozenset({"role", "members", "condition"})
_CONDITION_KEYS = frozenset({"title", "description", "expression", "location"})
_PROJECT_SCAN_KEYS = frozenset(
    {
        "schema", "status", "identity_plan_sha256", "inventory_receipt_sha256",
        "project", "observed_at_utc", "provider_source", "scanned_policy",
        "scanned_policy_sha256", "rows", "pool_account_count",
        "pool_role_membership_count", "all_pool_accounts_have_zero_project_roles",
        "bucket_iam_outside_scope", "read_only_observation", "cloud_mutated",
        "current_profile_changed", "receipt_sha256",
    }
)
_SCAN_ROW_KEYS = frozenset(
    {"account_id", "email", "principal", "role_memberships", "membership_count"}
)
_SETUP_PLAN_KEYS = frozenset(
    {
        "schema", "status", "project", "setup_nonce", "issued_at_utc", "accounts",
        "account_count", "create_only", "all_accounts_must_be_absent_before_create",
        "existing_account_acceptance_allowed", "run_specific", "bucket_iam_managed_here",
        "cloud_mutated", "current_profile_changed", "plan_sha256",
    }
)
_SETUP_ACCOUNT_KEYS = frozenset(
    {"account_id", "email", "name", "display_name", "description"}
)
_ABSENCE_OBSERVATION_KEYS = frozenset(
    {"account_id", "email", "name", "http_method", "http_status", "exists"}
)
_CREATE_OBSERVATION_KEYS = frozenset(
    {
        "account_id", "email", "name", "project_id", "unique_id", "disabled",
        "http_method", "http_status", "created", "already_existed",
    }
)
_SETUP_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "setup_plan_sha256", "project", "completed_at_utc",
        "provider_source", "absence_rows", "creation_rows", "created_account_count",
        "all_precreate_reads_absent", "all_creates_were_post", "all_accounts_newly_created",
        "existing_account_accepted", "create_only_contract_complete",
        "provider_mutation_observed", "module_executed_cloud_mutation",
        "current_profile_changed", "receipt_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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
        raise ValueError("immutable worker identity value was already sealed")
    return {**body, digest_field: canonical_sha256(body)}


def _validated_seal(
    value: Mapping[str, Any], *, keys: frozenset[str], digest_field: str, label: str
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


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} is not canonical UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not canonical UTC RFC3339") from exc
    canonical = parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    canonical = canonical.replace("+00:00", "Z")
    if parsed.tzinfo != timezone.utc or parsed.microsecond or canonical != value:
        raise ValueError(f"{label} is not canonical UTC RFC3339")
    return value


def _provider_source(value: Any) -> str:
    allowed = {
        "iam.googleapis.com/v1",
        "cloudresourcemanager.googleapis.com/v1",
        "fake-provider-test-v1",
    }
    if value not in allowed:
        raise ValueError("worker identity provider source changed")
    return str(value)


def account_name(email: str) -> str:
    return f"projects/{PROJECT}/serviceAccounts/{email}"


def fixed_pool_accounts() -> list[dict[str, str]]:
    return [
        {
            "account_id": account_id,
            "email": email,
            "name": account_name(email),
        }
        for account_id, email in zip(ACCOUNT_IDS, ACCOUNT_EMAILS, strict=True)
    ]


def _context(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    wave_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    plan = wave.validate_wave_plan(wave_plan)
    ledger = wave.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave.validate_resume_plan(plan, ledger, resume_plan)
    expected_wave = resume["resume_wave_index"]
    if (
        expected_wave is None
        or resume["all_jobs_complete"] is not False
        or not 1 <= len(resume["selected_attempts"]) <= POOL_SIZE
        or (wave_index is not None and wave_index != expected_wave)
        or any(
            row["job_id"] not in plan["waves"][expected_wave]["job_ids"]
            for row in resume["selected_attempts"]
        )
    ):
        raise ValueError("worker identity wave does not match resume selection")
    return plan, ledger, resume, expected_wave


def service_accounts_for_resume(resume_plan: Mapping[str, Any]) -> dict[str, str]:
    selected = resume_plan.get("selected_attempts")
    if not isinstance(selected, list) or not 1 <= len(selected) <= POOL_SIZE:
        raise ValueError("resume selection cannot be mapped to fixed worker pool")
    result: dict[str, str] = {}
    for index, row in enumerate(selected):
        if not isinstance(row, Mapping) or not isinstance(row.get("job_id"), str):
            raise ValueError("resume selection contains a malformed job")
        job_id = row["job_id"]
        if job_id in result:
            raise ValueError("resume selection contains a duplicate job")
        result[job_id] = ACCOUNT_EMAILS[index]
    return result


def _build_identity_plan_core(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_index: int,
) -> dict[str, Any]:
    selected_workers: list[dict[str, Any]] = []
    for index, selected in enumerate(resume["selected_attempts"]):
        email = ACCOUNT_EMAILS[index]
        selected_workers.append(
            {
                "worker_slot": index,
                "job_id": selected["job_id"],
                "source_role": selected["source_role"],
                "attempt_id": selected["attempt_id"],
                "vm_instance_id": selected["instance_id"],
                "account_id": ACCOUNT_IDS[index],
                "service_account_email": email,
                "service_account_name": account_name(email),
            }
        )
    body = {
        "schema": IDENTITY_PLAN_SCHEMA,
        "status": "fixed_reusable_worker_pool_mapped_cloud_not_mutated",
        "project": PROJECT,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_schedule_sha256": plan["schedule_sha256"],
        "wave_index": wave_index,
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "pool_account_ids": list(ACCOUNT_IDS),
        "pool_account_emails": list(ACCOUNT_EMAILS),
        "pool_size": POOL_SIZE,
        "selected_workers": selected_workers,
        "selected_count": len(selected_workers),
        "mapping_basis": "validated_resume_selected_attempts_order",
        "fixed_reusable_pool": True,
        "run_specific_account_creation": False,
        "bucket_iam_managed_here": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return _seal(body, "plan_sha256")


def build_worker_identity_plan(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    wave_index: int,
) -> dict[str, Any]:
    plan, ledger, resume, expected_wave = _context(
        wave_plan, attempt_ledger, resume_plan, wave_index
    )
    return validate_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=_build_identity_plan_core(plan, ledger, resume, expected_wave),
    )


def validate_worker_identity_plan(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan, ledger, resume, expected_wave = _context(
        wave_plan, attempt_ledger, resume_plan
    )
    checked = _validated_seal(
        value, keys=_PLAN_KEYS, digest_field="plan_sha256", label="identity plan"
    )
    workers = checked.get("selected_workers")
    if not isinstance(workers, list):
        raise ValueError("identity plan selected workers changed")
    for row in workers:
        if not isinstance(row, Mapping) or set(row) != _SELECTED_KEYS:
            raise ValueError("identity plan selected worker fields changed")
    expected = _build_identity_plan_core(plan, ledger, resume, expected_wave)
    if checked != expected:
        raise ValueError("identity plan is not the deterministic fixed-pool mapping")
    return checked


def _validated_account_rows(
    observations: Sequence[Mapping[str, Any]], *, require_exists: bool
) -> list[dict[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("account observations must be a sequence")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ValueError("account observation is not an object")
        row = copy.deepcopy(dict(raw))
        if set(row) != _ACCOUNT_OBSERVATION_KEYS:
            raise ValueError("account observation fields changed")
        account_id = row.get("account_id")
        if account_id in by_id or account_id not in ACCOUNT_IDS:
            raise ValueError("account observation identity is duplicated or unknown")
        index = ACCOUNT_IDS.index(account_id)
        email = ACCOUNT_EMAILS[index]
        if (
            row.get("email") != email
            or row.get("name") != account_name(email)
            or row.get("project_id") != PROJECT
            or not isinstance(row.get("exists"), bool)
            or (require_exists and row.get("exists") is not True)
            or not isinstance(row.get("disabled"), bool)
            or (require_exists and row.get("disabled") is not False)
            or _UNIQUE_ID.fullmatch(str(row.get("unique_id"))) is None
        ):
            raise ValueError("account observation does not prove the fixed enabled identity")
        by_id[account_id] = row
    if set(by_id) != set(ACCOUNT_IDS):
        raise ValueError("account inventory must exactly cover all eight pool accounts")
    rows = [by_id[account_id] for account_id in ACCOUNT_IDS]
    if len({row["unique_id"] for row in rows}) != POOL_SIZE:
        raise ValueError("service-account unique IDs are not unique")
    return rows


def build_inventory_receipt(
    *,
    identity_plan: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    observed_accounts: Sequence[Mapping[str, Any]],
    observed_at_utc: str,
    provider_source: str,
) -> dict[str, Any]:
    identity = validate_worker_identity_plan(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=identity_plan,
    )
    rows = _validated_account_rows(observed_accounts, require_exists=True)
    _timestamp(observed_at_utc, "inventory observation time")
    source = _provider_source(provider_source)
    body = {
        "schema": INVENTORY_RECEIPT_SCHEMA,
        "status": "all_fixed_worker_identities_exist_enabled_and_unique",
        "identity_plan_sha256": identity["plan_sha256"],
        "project": PROJECT,
        "observed_at_utc": observed_at_utc,
        "provider_source": source,
        "rows": rows,
        "pool_account_count": POOL_SIZE,
        "selected_account_count": identity["selected_count"],
        "all_pool_accounts_exist": True,
        "all_selected_accounts_exist": True,
        "all_enabled": True,
        "unique_ids_unique": True,
        "read_only_observation": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return validate_inventory_receipt(
        identity_plan=identity,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=_seal(body, "receipt_sha256"),
    )


def validate_inventory_receipt(
    *,
    identity_plan: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    identity = validate_worker_identity_plan(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=identity_plan,
    )
    checked = _validated_seal(
        value,
        keys=_INVENTORY_KEYS,
        digest_field="receipt_sha256",
        label="identity inventory receipt",
    )
    rows = _validated_account_rows(checked.get("rows"), require_exists=True)
    if (
        checked.get("schema") != INVENTORY_RECEIPT_SCHEMA
        or checked.get("status") != "all_fixed_worker_identities_exist_enabled_and_unique"
        or checked.get("identity_plan_sha256") != identity["plan_sha256"]
        or checked.get("project") != PROJECT
        or checked.get("provider_source") != _provider_source(checked.get("provider_source"))
        or checked.get("pool_account_count") != POOL_SIZE
        or checked.get("selected_account_count") != identity["selected_count"]
        or checked.get("all_pool_accounts_exist") is not True
        or checked.get("all_selected_accounts_exist") is not True
        or checked.get("all_enabled") is not True
        or checked.get("unique_ids_unique") is not True
        or checked.get("read_only_observation") is not True
        or checked.get("cloud_mutated") is not False
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("identity inventory receipt contract changed")
    _timestamp(checked.get("observed_at_utc"), "inventory observation time")
    if checked["rows"] != rows:
        raise ValueError("identity inventory order changed")
    return checked


def _validated_act_as_rows(
    observations: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("actAs observations must be a sequence")
    by_id: dict[str, dict[str, Any]] = {}
    expected = {row["account_id"]: row for row in identity["selected_workers"]}
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ValueError("actAs observation is not an object")
        row = copy.deepcopy(dict(raw))
        if set(row) != _ACT_AS_OBSERVATION_KEYS:
            raise ValueError("actAs observation fields changed")
        account_id = row.get("account_id")
        selected = expected.get(account_id)
        if selected is None or account_id in by_id:
            raise ValueError("actAs observation is extra, missing, or duplicated")
        if (
            row.get("email") != selected["service_account_email"]
            or row.get("name") != selected["service_account_name"]
            or row.get("http_method") != "POST"
            or row.get("requested_permissions") != [ACT_AS_PERMISSION]
            or row.get("granted_permissions") != [ACT_AS_PERMISSION]
        ):
            raise PermissionError("testIamPermissions did not grant exactly actAs")
        by_id[account_id] = row
    if set(by_id) != set(expected):
        raise PermissionError("testIamPermissions did not cover every selected account")
    return [by_id[row["account_id"]] for row in identity["selected_workers"]]


def build_act_as_receipt(
    *,
    identity_plan: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    test_iam_permissions_observations: Sequence[Mapping[str, Any]],
    tested_at_utc: str,
    provider_source: str,
) -> dict[str, Any]:
    identity = validate_worker_identity_plan(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=identity_plan,
    )
    inventory = validate_inventory_receipt(
        identity_plan=identity,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=inventory_receipt,
    )
    rows = _validated_act_as_rows(test_iam_permissions_observations, identity)
    _timestamp(tested_at_utc, "actAs test time")
    source = _provider_source(provider_source)
    body = {
        "schema": ACT_AS_RECEIPT_SCHEMA,
        "status": "every_selected_worker_grants_exact_act_as",
        "identity_plan_sha256": identity["plan_sha256"],
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "project": PROJECT,
        "tested_at_utc": tested_at_utc,
        "provider_source": source,
        "rows": rows,
        "selected_count": identity["selected_count"],
        "required_permission": ACT_AS_PERMISSION,
        "every_selected_account_tested": True,
        "every_selected_account_grants_exact_act_as": True,
        "test_iam_permissions_only": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return validate_act_as_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=_seal(body, "receipt_sha256"),
    )


def validate_act_as_receipt(
    *,
    identity_plan: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    identity = validate_worker_identity_plan(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=identity_plan,
    )
    inventory = validate_inventory_receipt(
        identity_plan=identity,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=inventory_receipt,
    )
    checked = _validated_seal(
        value, keys=_ACT_AS_KEYS, digest_field="receipt_sha256", label="actAs receipt"
    )
    rows = _validated_act_as_rows(checked.get("rows"), identity)
    if (
        checked.get("schema") != ACT_AS_RECEIPT_SCHEMA
        or checked.get("status") != "every_selected_worker_grants_exact_act_as"
        or checked.get("identity_plan_sha256") != identity["plan_sha256"]
        or checked.get("inventory_receipt_sha256") != inventory["receipt_sha256"]
        or checked.get("project") != PROJECT
        or checked.get("provider_source") != _provider_source(checked.get("provider_source"))
        or checked.get("rows") != rows
        or checked.get("selected_count") != identity["selected_count"]
        or checked.get("required_permission") != ACT_AS_PERMISSION
        or checked.get("every_selected_account_tested") is not True
        or checked.get("every_selected_account_grants_exact_act_as") is not True
        or checked.get("test_iam_permissions_only") is not True
        or checked.get("cloud_mutated") is not False
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("actAs receipt contract changed")
    _timestamp(checked.get("tested_at_utc"), "actAs test time")
    return checked


def _normalize_project_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) not in (
        _POLICY_REQUIRED_KEYS,
        _POLICY_KEYS,
    ):
        raise ValueError("project IAM policy fields changed")
    policy = copy.deepcopy(dict(value))
    # Cloud Resource Manager omits empty repeated fields from its JSON
    # response.  An absent auditConfigs therefore has the same canonical
    # meaning as an explicitly empty list; no other top-level omission is
    # accepted.
    policy.setdefault("auditConfigs", [])
    if (
        isinstance(policy.get("version"), bool)
        or not isinstance(policy.get("version"), int)
        or policy["version"] not in {1, 3}
        or not isinstance(policy.get("etag"), str)
        or not policy["etag"]
        or not isinstance(policy.get("bindings"), list)
        or not isinstance(policy.get("auditConfigs"), list)
    ):
        raise ValueError("project IAM policy is malformed")
    for binding in policy["bindings"]:
        if not isinstance(binding, Mapping):
            raise ValueError("project IAM binding is malformed")
        keys = set(binding)
        if keys not in ({"role", "members"}, _BINDING_KEYS):
            raise ValueError("project IAM binding fields changed")
        if (
            _ROLE.fullmatch(str(binding.get("role"))) is None
            or not isinstance(binding.get("members"), list)
            or len(binding["members"]) != len(set(binding["members"]))
            or any(not isinstance(member, str) or not member for member in binding["members"])
        ):
            raise ValueError("project IAM binding is malformed")
        if "condition" in binding:
            condition = binding["condition"]
            if (
                not isinstance(condition, Mapping)
                or not {"title", "expression"}.issubset(condition)
                or not set(condition).issubset(_CONDITION_KEYS)
                or any(not isinstance(item, str) for item in condition.values())
            ):
                raise ValueError("project IAM condition is malformed")
    return policy


def _scan_project_memberships(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize_project_policy(policy)
    rows: list[dict[str, Any]] = []
    for account_id, email in zip(ACCOUNT_IDS, ACCOUNT_EMAILS, strict=True):
        memberships: list[str] = []
        for binding in normalized["bindings"]:
            if any(email in member for member in binding["members"]):
                memberships.append(binding["role"])
        if memberships:
            raise PermissionError(
                f"fixed worker identity {account_id} has project-level role membership"
            )
        rows.append(
            {
                "account_id": account_id,
                "email": email,
                "principal": f"serviceAccount:{email}",
                "role_memberships": [],
                "membership_count": 0,
            }
        )
    return rows


def build_project_iam_scan_receipt(
    *,
    identity_plan: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    project_iam_policy: Mapping[str, Any],
    observed_at_utc: str,
    provider_source: str,
) -> dict[str, Any]:
    identity = validate_worker_identity_plan(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=identity_plan,
    )
    inventory = validate_inventory_receipt(
        identity_plan=identity,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=inventory_receipt,
    )
    policy = _normalize_project_policy(project_iam_policy)
    rows = _scan_project_memberships(policy)
    _timestamp(observed_at_utc, "project IAM observation time")
    source = _provider_source(provider_source)
    body = {
        "schema": PROJECT_IAM_SCAN_RECEIPT_SCHEMA,
        "status": "fixed_worker_pool_has_zero_project_role_memberships",
        "identity_plan_sha256": identity["plan_sha256"],
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "project": PROJECT,
        "observed_at_utc": observed_at_utc,
        "provider_source": source,
        "scanned_policy": policy,
        "scanned_policy_sha256": canonical_sha256(policy),
        "rows": rows,
        "pool_account_count": POOL_SIZE,
        "pool_role_membership_count": 0,
        "all_pool_accounts_have_zero_project_roles": True,
        "bucket_iam_outside_scope": True,
        "read_only_observation": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return validate_project_iam_scan_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=_seal(body, "receipt_sha256"),
    )


def validate_project_iam_scan_receipt(
    *,
    identity_plan: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    identity = validate_worker_identity_plan(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=identity_plan,
    )
    inventory = validate_inventory_receipt(
        identity_plan=identity,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        value=inventory_receipt,
    )
    checked = _validated_seal(
        value,
        keys=_PROJECT_SCAN_KEYS,
        digest_field="receipt_sha256",
        label="project IAM scan receipt",
    )
    policy = _normalize_project_policy(checked.get("scanned_policy"))
    rows = _scan_project_memberships(policy)
    if (
        checked.get("schema") != PROJECT_IAM_SCAN_RECEIPT_SCHEMA
        or checked.get("status") != "fixed_worker_pool_has_zero_project_role_memberships"
        or checked.get("identity_plan_sha256") != identity["plan_sha256"]
        or checked.get("inventory_receipt_sha256") != inventory["receipt_sha256"]
        or checked.get("project") != PROJECT
        or checked.get("provider_source") != _provider_source(checked.get("provider_source"))
        or checked.get("scanned_policy_sha256") != canonical_sha256(policy)
        or checked.get("rows") != rows
        or checked.get("pool_account_count") != POOL_SIZE
        or checked.get("pool_role_membership_count") != 0
        or checked.get("all_pool_accounts_have_zero_project_roles") is not True
        or checked.get("bucket_iam_outside_scope") is not True
        or checked.get("read_only_observation") is not True
        or checked.get("cloud_mutated") is not False
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("project IAM scan receipt contract changed")
    _timestamp(checked.get("observed_at_utc"), "project IAM observation time")
    return checked


def _setup_accounts() -> list[dict[str, str]]:
    return [
        {
            **row,
            "display_name": f"OFC full100 worker {index:02d}",
            "description": "Fixed unprivileged OFC full100 worker identity",
        }
        for index, row in enumerate(fixed_pool_accounts())
    ]


def build_create_only_setup_plan(
    *, setup_nonce: str, issued_at_utc: str
) -> dict[str, Any]:
    if not isinstance(setup_nonce, str) or _NONCE.fullmatch(setup_nonce) is None:
        raise ValueError("setup nonce is not a stable one-time identifier")
    _timestamp(issued_at_utc, "setup issue time")
    body = {
        "schema": SETUP_PLAN_SCHEMA,
        "status": "one_time_fixed_pool_create_only_setup_planned",
        "project": PROJECT,
        "setup_nonce": setup_nonce,
        "issued_at_utc": issued_at_utc,
        "accounts": _setup_accounts(),
        "account_count": POOL_SIZE,
        "create_only": True,
        "all_accounts_must_be_absent_before_create": True,
        "existing_account_acceptance_allowed": False,
        "run_specific": False,
        "bucket_iam_managed_here": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return validate_create_only_setup_plan(_seal(body, "plan_sha256"))


def validate_create_only_setup_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    checked = _validated_seal(
        value,
        keys=_SETUP_PLAN_KEYS,
        digest_field="plan_sha256",
        label="create-only setup plan",
    )
    accounts = checked.get("accounts")
    if (
        not isinstance(accounts, list)
        or any(not isinstance(row, Mapping) or set(row) != _SETUP_ACCOUNT_KEYS for row in accounts)
        or accounts != _setup_accounts()
        or checked.get("schema") != SETUP_PLAN_SCHEMA
        or checked.get("status") != "one_time_fixed_pool_create_only_setup_planned"
        or checked.get("project") != PROJECT
        or not isinstance(checked.get("setup_nonce"), str)
        or _NONCE.fullmatch(checked["setup_nonce"]) is None
        or checked.get("account_count") != POOL_SIZE
        or checked.get("create_only") is not True
        or checked.get("all_accounts_must_be_absent_before_create") is not True
        or checked.get("existing_account_acceptance_allowed") is not False
        or checked.get("run_specific") is not False
        or checked.get("bucket_iam_managed_here") is not False
        or checked.get("cloud_mutated") is not False
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("create-only setup plan contract changed")
    _timestamp(checked.get("issued_at_utc"), "setup issue time")
    return checked


def _validated_setup_absence_rows(
    observations: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("setup absence observations must be a sequence")
    by_id: dict[str, dict[str, Any]] = {}
    expected = {row["account_id"]: row for row in plan["accounts"]}
    for raw in observations:
        if not isinstance(raw, Mapping) or set(raw) != _ABSENCE_OBSERVATION_KEYS:
            raise ValueError("setup absence observation fields changed")
        row = copy.deepcopy(dict(raw))
        desired = expected.get(row.get("account_id"))
        if desired is None or row["account_id"] in by_id:
            raise ValueError("setup absence observation is extra or duplicated")
        if (
            row.get("email") != desired["email"]
            or row.get("name") != desired["name"]
            or row.get("http_method") != "GET"
            or row.get("http_status") != 404
            or row.get("exists") is not False
        ):
            raise FileExistsError("create-only setup did not prove prior absence")
        by_id[row["account_id"]] = row
    if set(by_id) != set(expected):
        raise ValueError("setup absence observations do not cover the fixed pool")
    return [by_id[row["account_id"]] for row in plan["accounts"]]


def _validated_setup_creation_rows(
    observations: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("setup creation observations must be a sequence")
    by_id: dict[str, dict[str, Any]] = {}
    expected = {row["account_id"]: row for row in plan["accounts"]}
    for raw in observations:
        if not isinstance(raw, Mapping) or set(raw) != _CREATE_OBSERVATION_KEYS:
            raise ValueError("setup creation observation fields changed")
        row = copy.deepcopy(dict(raw))
        desired = expected.get(row.get("account_id"))
        if desired is None or row["account_id"] in by_id:
            raise ValueError("setup creation observation is extra or duplicated")
        if (
            row.get("email") != desired["email"]
            or row.get("name") != desired["name"]
            or row.get("project_id") != PROJECT
            or _UNIQUE_ID.fullmatch(str(row.get("unique_id"))) is None
            or row.get("disabled") is not False
            or row.get("http_method") != "POST"
            or row.get("http_status") not in {200, 201}
            or row.get("created") is not True
            or row.get("already_existed") is not False
        ):
            raise FileExistsError("create-only setup did not newly create the exact account")
        by_id[row["account_id"]] = row
    if set(by_id) != set(expected):
        raise ValueError("setup creation observations do not cover the fixed pool")
    rows = [by_id[row["account_id"]] for row in plan["accounts"]]
    if len({row["unique_id"] for row in rows}) != POOL_SIZE:
        raise ValueError("created service-account unique IDs are not unique")
    return rows


def build_create_only_setup_receipt(
    *,
    setup_plan: Mapping[str, Any],
    absence_observations: Sequence[Mapping[str, Any]],
    creation_observations: Sequence[Mapping[str, Any]],
    completed_at_utc: str,
    provider_source: str,
) -> dict[str, Any]:
    plan = validate_create_only_setup_plan(setup_plan)
    absence = _validated_setup_absence_rows(absence_observations, plan)
    creation = _validated_setup_creation_rows(creation_observations, plan)
    _timestamp(completed_at_utc, "setup completion time")
    if completed_at_utc < plan["issued_at_utc"]:
        raise ValueError("setup receipt predates its plan")
    source = _provider_source(provider_source)
    body = {
        "schema": SETUP_RECEIPT_SCHEMA,
        "status": "fixed_worker_pool_created_once_without_existing_acceptance",
        "setup_plan_sha256": plan["plan_sha256"],
        "project": PROJECT,
        "completed_at_utc": completed_at_utc,
        "provider_source": source,
        "absence_rows": absence,
        "creation_rows": creation,
        "created_account_count": POOL_SIZE,
        "all_precreate_reads_absent": True,
        "all_creates_were_post": True,
        "all_accounts_newly_created": True,
        "existing_account_accepted": False,
        "create_only_contract_complete": True,
        "provider_mutation_observed": True,
        "module_executed_cloud_mutation": False,
        "current_profile_changed": False,
    }
    return validate_create_only_setup_receipt(
        setup_plan=plan, value=_seal(body, "receipt_sha256")
    )


def validate_create_only_setup_receipt(
    *, setup_plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_create_only_setup_plan(setup_plan)
    checked = _validated_seal(
        value,
        keys=_SETUP_RECEIPT_KEYS,
        digest_field="receipt_sha256",
        label="create-only setup receipt",
    )
    absence = _validated_setup_absence_rows(checked.get("absence_rows"), plan)
    creation = _validated_setup_creation_rows(checked.get("creation_rows"), plan)
    if (
        checked.get("schema") != SETUP_RECEIPT_SCHEMA
        or checked.get("status") != "fixed_worker_pool_created_once_without_existing_acceptance"
        or checked.get("setup_plan_sha256") != plan["plan_sha256"]
        or checked.get("project") != PROJECT
        or checked.get("provider_source") != _provider_source(checked.get("provider_source"))
        or checked.get("absence_rows") != absence
        or checked.get("creation_rows") != creation
        or checked.get("created_account_count") != POOL_SIZE
        or checked.get("all_precreate_reads_absent") is not True
        or checked.get("all_creates_were_post") is not True
        or checked.get("all_accounts_newly_created") is not True
        or checked.get("existing_account_accepted") is not False
        or checked.get("create_only_contract_complete") is not True
        or checked.get("provider_mutation_observed") is not True
        or checked.get("module_executed_cloud_mutation") is not False
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("create-only setup receipt contract changed")
    _timestamp(checked.get("completed_at_utc"), "setup completion time")
    if checked["completed_at_utc"] < plan["issued_at_utc"]:
        raise ValueError("setup receipt predates its plan")
    return checked


__all__ = [
    "ACCOUNT_EMAILS",
    "ACCOUNT_IDS",
    "ACT_AS_PERMISSION",
    "ACT_AS_RECEIPT_SCHEMA",
    "IDENTITY_PLAN_SCHEMA",
    "INVENTORY_RECEIPT_SCHEMA",
    "POOL_SIZE",
    "PROJECT",
    "PROJECT_IAM_SCAN_RECEIPT_SCHEMA",
    "SETUP_PLAN_SCHEMA",
    "SETUP_RECEIPT_SCHEMA",
    "account_name",
    "build_act_as_receipt",
    "build_create_only_setup_plan",
    "build_create_only_setup_receipt",
    "build_inventory_receipt",
    "build_project_iam_scan_receipt",
    "build_worker_identity_plan",
    "canonical_bytes",
    "canonical_sha256",
    "fixed_pool_accounts",
    "service_accounts_for_resume",
    "validate_act_as_receipt",
    "validate_create_only_setup_plan",
    "validate_create_only_setup_receipt",
    "validate_inventory_receipt",
    "validate_project_iam_scan_receipt",
    "validate_worker_identity_plan",
]
