"""Injectable GCP adapter for the fixed eight-account full100 worker pool.

No live client is imported.  The caller must inject HTTP and a bearer token is
read only from the environment.  The adapter can read identities, create only
accounts proven absent by a short-lived read receipt, test exact actAs, and
scan project IAM.  It has no role-write or delete surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as identity_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


POOL_READ_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_gcp_read_receipt_v2"
)
POOL_CREATE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_worker_identity_gcp_create_receipt_v2"
)

PROJECT = identity_v2.PROJECT
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
MAX_READ_RECEIPT_LIFETIME_SECONDS = 300
_MODES = frozenset({"read", "create-missing", "actas-check", "project-iam-scan"})
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UNIQUE_ID = re.compile(r"^[1-9][0-9]{5,31}$")
_UTC = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]

_READ_ROW_KEYS = frozenset(
    {
        "account_id", "email", "name", "http_method", "http_status",
        "exists", "project_id", "unique_id", "disabled", "display_name",
        "description",
    }
)
_READ_KEYS = frozenset(
    {
        "schema", "status", "identity_plan_sha256", "setup_plan_sha256",
        "project", "observed_at_utc", "expires_at_utc", "rows",
        "pool_account_count", "existing_account_count", "missing_account_count",
        "missing_account_ids", "all_accounts_exist", "all_accounts_missing",
        "create_missing_authorized", "inventory_receipt",
        "inventory_receipt_sha256", "provider_source", "http_get_count",
        "read_only", "cloud_mutated", "current_profile_changed",
        "receipt_sha256",
    }
)
_CREATE_KEYS = frozenset(
    {
        "schema", "status", "identity_plan_sha256", "setup_plan_sha256",
        "read_receipt_sha256", "project", "completed_at_utc",
        "created_account_ids", "created_account_count",
        "preexisting_account_count", "creation_rows", "setup_receipt",
        "setup_receipt_sha256", "inventory_receipt",
        "inventory_receipt_sha256", "all_pool_accounts_exist_after_create",
        "all_created_accounts_read_back_exact", "create_only_missing_accounts",
        "existing_account_adopted_from_collision", "roles_added",
        "accounts_deleted", "provider_source", "http_post_count",
        "http_get_count", "cloud_mutated", "current_profile_changed",
        "receipt_sha256",
    }
)
_CREATION_ROW_KEYS = frozenset(
    {
        "account_id", "email", "name", "project_id", "unique_id",
        "disabled", "http_method", "http_status", "created",
        "already_existed",
    }
)


class WorkerIdentityCreateIncompleteError(RuntimeError):
    """Create response was ambiguous; a fresh fixed-pool read is attached."""

    def __init__(
        self, message: str, *, recovery_read_receipt: Mapping[str, Any]
    ) -> None:
        super().__init__(message)
        self.recovery_read_receipt = deepcopy(dict(recovery_read_receipt))


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


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> float:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not UTC")
    return parsed.timestamp()


def _seal(core: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(core))
    if "receipt_sha256" in body:
        raise ValueError("worker identity GCP receipt was already sealed")
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _pop_receipt(
    value: Mapping[str, Any], *, keys: frozenset[str], label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    body = deepcopy(dict(value))
    _exact_keys(body, keys, label)
    digest = body.pop("receipt_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError(f"{label} digest changed")
    return body, _require_sha(digest, f"{label} digest")


def _account_url(email: str) -> str:
    return (
        f"https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts/"
        f"{urllib.parse.quote(email, safe='')}"
    )


def _accounts_url() -> str:
    return f"https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts"


def _actas_url(name: str) -> str:
    return f"https://iam.googleapis.com/v1/{name}:testIamPermissions"


def _project_policy_url() -> str:
    return (
        "https://cloudresourcemanager.googleapis.com/v1/projects/"
        f"{PROJECT}:getIamPolicy"
    )


def _expected_setup_accounts(
    setup_plan: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    if setup_plan is None:
        return [
            {
                **row,
                "display_name": f"OFC full100 worker {index:02d}",
                "description": "Fixed unprivileged OFC full100 worker identity",
            }
            for index, row in enumerate(identity_v2.fixed_pool_accounts())
        ]
    plan = identity_v2.validate_create_only_setup_plan(setup_plan)
    return deepcopy(plan["accounts"])


def _provider_account(
    raw: Mapping[str, Any], expected: Mapping[str, str]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("IAM service account response is not an object")
    unique_id = str(raw.get("uniqueId", ""))
    disabled = raw.get("disabled", False)
    if (
        raw.get("name") != expected["name"]
        or raw.get("projectId") != PROJECT
        or raw.get("email") != expected["email"]
        or _UNIQUE_ID.fullmatch(unique_id) is None
        or disabled is not False
        or raw.get("displayName") != expected["display_name"]
        or raw.get("description") != expected["description"]
    ):
        raise RuntimeError("IAM service account identity/configuration drifted")
    return {
        "account_id": expected["account_id"],
        "email": expected["email"],
        "name": expected["name"],
        "project_id": PROJECT,
        "unique_id": unique_id,
        "disabled": False,
        "exists": True,
    }


class WorkerIdentityGcpAdapterV2:
    """Mode-separated fixed-pool IAM adapter."""

    def __init__(
        self,
        *,
        mode: str,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        resume_plan: Mapping[str, Any],
        identity_plan: Mapping[str, Any],
        requester: HttpRequester,
        setup_plan: Mapping[str, Any] | None = None,
        read_receipt: Mapping[str, Any] | None = None,
        inventory_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError("worker identity GCP adapter mode is invalid")
        if not callable(requester):
            raise ValueError("worker identity GCP adapter requires injected HTTP")
        self.mode = mode
        self._requester = requester
        self._wave_plan = deepcopy(dict(wave_plan))
        self._attempt_ledger = deepcopy(dict(attempt_ledger))
        self._resume_plan = deepcopy(dict(resume_plan))
        self._identity = identity_v2.validate_worker_identity_plan(
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            value=identity_plan,
        )
        self._setup_plan = (
            None
            if setup_plan is None
            else identity_v2.validate_create_only_setup_plan(setup_plan)
        )
        self._accounts = _expected_setup_accounts(self._setup_plan)
        self._used = False
        self._http_get_count = 0
        self._http_post_count = 0
        self._read_receipt: dict[str, Any] | None = None
        self._inventory: dict[str, Any] | None = None
        if mode == "read":
            if read_receipt is not None or inventory_receipt is not None:
                raise PermissionError("read mode cannot accept prior pool evidence")
        elif mode == "create-missing":
            if self._setup_plan is None or read_receipt is None:
                raise PermissionError("create-missing requires setup plan and fresh read receipt")
            if inventory_receipt is not None:
                raise PermissionError("create-missing cannot adopt an existing inventory receipt")
            self._read_receipt = self.validate_read_receipt(read_receipt)
        else:
            if read_receipt is not None or inventory_receipt is None:
                raise PermissionError(f"{mode} requires only a validated inventory receipt")
            self._inventory = identity_v2.validate_inventory_receipt(
                identity_plan=self._identity,
                wave_plan=self._wave_plan,
                attempt_ledger=self._attempt_ledger,
                resume_plan=self._resume_plan,
                value=inventory_receipt,
            )

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        token = os.environ.get(TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            character.isspace() for character in token
        ):
            raise PermissionError(f"Bearer token must be supplied only through {TOKEN_ENV}")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def _http(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        allowed_statuses: Sequence[int] = (200,),
    ) -> HttpResponse:
        allowed = {
            "read": {"GET"},
            "create-missing": {"GET", "POST"},
            "actas-check": {"POST"},
            "project-iam-scan": {"POST"},
        }[self.mode]
        if method not in allowed:
            raise PermissionError("HTTP method escaped worker identity adapter mode")
        account_urls = {_account_url(row["email"]) for row in self._accounts}
        selected_actas_urls = {
            _actas_url(row["service_account_name"])
            for row in self._identity["selected_workers"]
        }
        exact_request_allowed = {
            "read": method == "GET" and url in account_urls,
            "create-missing": (
                (method == "GET" and url in account_urls)
                or (method == "POST" and url == _accounts_url())
            ),
            "actas-check": method == "POST" and url in selected_actas_urls,
            "project-iam-scan": (
                method == "POST" and url == _project_policy_url()
            ),
        }[self.mode]
        if not exact_request_allowed:
            raise PermissionError("worker identity request escaped exact fixed-pool URL set")
        if method == "GET":
            self._http_get_count += 1
        else:
            self._http_post_count += 1
        response = self._requester(
            method,
            url,
            self._headers(content_type=content_type),
            body,
            60,
        )
        if response.status in {409, 412}:
            raise RuntimeError(
                f"worker identity create collision/precondition status {response.status}"
            )
        if response.status not in allowed_statuses:
            raise RuntimeError(
                f"worker identity {method} failed with status {response.status}"
            )
        return response

    @staticmethod
    def _json(response: HttpResponse, label: str) -> dict[str, Any]:
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} response was not JSON") from exc
        if not isinstance(value, dict) or value.get("nextPageToken") is not None:
            raise RuntimeError(f"{label} response shape changed")
        return value

    def _read_all_accounts(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        receipt_rows: list[dict[str, Any]] = []
        inventory_rows: list[dict[str, Any]] = []
        for expected in self._accounts:
            response = self._http(
                method="GET",
                url=_account_url(expected["email"]),
                allowed_statuses=(200, 404),
            )
            if response.status == 404:
                receipt_rows.append(
                    {
                        "account_id": expected["account_id"],
                        "email": expected["email"],
                        "name": expected["name"],
                        "http_method": "GET",
                        "http_status": 404,
                        "exists": False,
                        "project_id": None,
                        "unique_id": None,
                        "disabled": None,
                        "display_name": None,
                        "description": None,
                    }
                )
                continue
            observed = _provider_account(
                self._json(response, "IAM service account GET"), expected
            )
            inventory_rows.append(observed)
            receipt_rows.append(
                {
                    "account_id": expected["account_id"],
                    "email": expected["email"],
                    "name": expected["name"],
                    "http_method": "GET",
                    "http_status": 200,
                    "exists": True,
                    "project_id": observed["project_id"],
                    "unique_id": observed["unique_id"],
                    "disabled": observed["disabled"],
                    "display_name": expected["display_name"],
                    "description": expected["description"],
                }
            )
        return receipt_rows, inventory_rows

    def _validate_read_rows(
        self, rows: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        if not isinstance(rows, list) or len(rows) != identity_v2.POOL_SIZE:
            raise ValueError("worker pool read rows must exactly cover eight accounts")
        checked: list[dict[str, Any]] = []
        inventory: list[dict[str, Any]] = []
        missing: list[str] = []
        for expected, raw in zip(self._accounts, rows, strict=True):
            if not isinstance(raw, Mapping):
                raise ValueError("worker pool read row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(row, _READ_ROW_KEYS, "worker pool read row")
            if (
                row.get("account_id") != expected["account_id"]
                or row.get("email") != expected["email"]
                or row.get("name") != expected["name"]
                or row.get("http_method") != "GET"
            ):
                raise ValueError("worker pool read row identity changed")
            if row.get("http_status") == 404 and row.get("exists") is False:
                if any(
                    row.get(field) is not None
                    for field in (
                        "project_id", "unique_id", "disabled",
                        "display_name", "description",
                    )
                ):
                    raise ValueError("missing worker pool row contains adopted identity data")
                missing.append(expected["account_id"])
            elif row.get("http_status") == 200 and row.get("exists") is True:
                if (
                    row.get("project_id") != PROJECT
                    or _UNIQUE_ID.fullmatch(str(row.get("unique_id"))) is None
                    or row.get("disabled") is not False
                    or row.get("display_name") != expected["display_name"]
                    or row.get("description") != expected["description"]
                ):
                    raise ValueError("existing worker pool row configuration changed")
                inventory.append(
                    {
                        "account_id": expected["account_id"],
                        "email": expected["email"],
                        "name": expected["name"],
                        "project_id": PROJECT,
                        "unique_id": str(row["unique_id"]),
                        "disabled": False,
                        "exists": True,
                    }
                )
            else:
                raise ValueError("worker pool read row status changed")
            checked.append(row)
        existing_ids = [row["unique_id"] for row in inventory]
        if len(existing_ids) != len(set(existing_ids)):
            raise ValueError("existing worker pool unique IDs are duplicated")
        return checked, inventory, missing

    def validate_read_receipt(
        self,
        value: Mapping[str, Any],
        *,
        current_time_utc: str | None = None,
    ) -> dict[str, Any]:
        body, digest = _pop_receipt(
            value, keys=_READ_KEYS, label="worker pool GCP read receipt"
        )
        rows, inventory_rows, missing = self._validate_read_rows(body.get("rows"))
        all_exist = not missing
        all_missing = len(missing) == identity_v2.POOL_SIZE
        inventory = body.get("inventory_receipt")
        if all_exist:
            if not isinstance(inventory, Mapping):
                raise ValueError("all-existing worker pool read lacks inventory receipt")
            validated_inventory = identity_v2.validate_inventory_receipt(
                identity_plan=self._identity,
                wave_plan=self._wave_plan,
                attempt_ledger=self._attempt_ledger,
                resume_plan=self._resume_plan,
                value=inventory,
            )
            if validated_inventory["rows"] != inventory_rows:
                raise ValueError("worker pool read and inventory observations differ")
        else:
            if inventory is not None:
                raise ValueError("incomplete worker pool cannot carry inventory receipt")
            validated_inventory = None
        observed = _timestamp(body.get("observed_at_utc"), "pool read time")
        expires = _timestamp(body.get("expires_at_utc"), "pool read expiry")
        if not 0 < expires - observed <= MAX_READ_RECEIPT_LIFETIME_SECONDS:
            raise ValueError("worker pool read receipt lifetime changed")
        if current_time_utc is not None:
            now = _timestamp(current_time_utc, "pool create current time")
            if now < observed or now > expires:
                raise PermissionError("worker pool read receipt is stale or future-dated")
        expected_setup_sha = (
            None if self._setup_plan is None else self._setup_plan["plan_sha256"]
        )
        if (
            body.get("schema") != POOL_READ_RECEIPT_SCHEMA
            or body.get("status")
            != ("fixed_worker_pool_complete_read_only" if all_exist else "fixed_worker_pool_missing_accounts_observed")
            or body.get("identity_plan_sha256") != self._identity["plan_sha256"]
            or body.get("setup_plan_sha256") != expected_setup_sha
            or body.get("project") != PROJECT
            or body.get("pool_account_count") != identity_v2.POOL_SIZE
            or body.get("existing_account_count") != len(inventory_rows)
            or body.get("missing_account_count") != len(missing)
            or body.get("missing_account_ids") != missing
            or body.get("all_accounts_exist") is not all_exist
            or body.get("all_accounts_missing") is not all_missing
            or body.get("create_missing_authorized") is not (not all_exist)
            or body.get("inventory_receipt_sha256")
            != (None if validated_inventory is None else validated_inventory["receipt_sha256"])
            or body.get("provider_source") != "iam.googleapis.com/v1"
            or body.get("http_get_count") != identity_v2.POOL_SIZE
            or body.get("read_only") is not True
            or body.get("cloud_mutated") is not False
            or body.get("current_profile_changed") is not False
        ):
            raise ValueError("worker pool GCP read receipt contract changed")
        body["rows"] = rows
        body["inventory_receipt"] = validated_inventory
        return {**body, "receipt_sha256": digest}

    def read_pool(
        self, *, observed_at_utc: str, expires_at_utc: str
    ) -> dict[str, Any]:
        if self.mode != "read" or self._used:
            raise PermissionError("worker pool read mode is unavailable or consumed")
        self._used = True
        rows, inventory_rows = self._read_all_accounts()
        return self._build_read_receipt(
            rows=rows,
            inventory_rows=inventory_rows,
            observed_at_utc=observed_at_utc,
            expires_at_utc=expires_at_utc,
        )

    def _build_read_receipt(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        inventory_rows: Sequence[Mapping[str, Any]],
        observed_at_utc: str,
        expires_at_utc: str,
    ) -> dict[str, Any]:
        checked_rows = [deepcopy(dict(row)) for row in rows]
        checked_inventory = [deepcopy(dict(row)) for row in inventory_rows]
        missing = [
            row["account_id"] for row in checked_rows if row["exists"] is False
        ]
        inventory = None
        if not missing:
            inventory = identity_v2.build_inventory_receipt(
                identity_plan=self._identity,
                wave_plan=self._wave_plan,
                attempt_ledger=self._attempt_ledger,
                resume_plan=self._resume_plan,
                observed_accounts=checked_inventory,
                observed_at_utc=observed_at_utc,
                provider_source="iam.googleapis.com/v1",
            )
        body = {
            "schema": POOL_READ_RECEIPT_SCHEMA,
            "status": (
                "fixed_worker_pool_complete_read_only"
                if not missing
                else "fixed_worker_pool_missing_accounts_observed"
            ),
            "identity_plan_sha256": self._identity["plan_sha256"],
            "setup_plan_sha256": (
                None if self._setup_plan is None else self._setup_plan["plan_sha256"]
            ),
            "project": PROJECT,
            "observed_at_utc": observed_at_utc,
            "expires_at_utc": expires_at_utc,
            "rows": checked_rows,
            "pool_account_count": identity_v2.POOL_SIZE,
            "existing_account_count": len(checked_inventory),
            "missing_account_count": len(missing),
            "missing_account_ids": missing,
            "all_accounts_exist": not missing,
            "all_accounts_missing": len(missing) == identity_v2.POOL_SIZE,
            "create_missing_authorized": bool(missing),
            "inventory_receipt": inventory,
            "inventory_receipt_sha256": (
                None if inventory is None else inventory["receipt_sha256"]
            ),
            "provider_source": "iam.googleapis.com/v1",
            # This receipt represents this one complete fixed-pool scan.  The
            # enclosing create receipt separately records cumulative GETs.
            "http_get_count": identity_v2.POOL_SIZE,
            "read_only": True,
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
        return self.validate_read_receipt(_seal(body))

    def _validate_creation_rows(self, rows: Any) -> list[dict[str, Any]]:
        if self._read_receipt is None:
            raise PermissionError("worker pool creation rows require read evidence")
        missing_ids = self._read_receipt["missing_account_ids"]
        if not isinstance(rows, list) or len(rows) != len(missing_ids):
            raise ValueError("worker pool creation rows do not cover missing accounts")
        expected = {row["account_id"]: row for row in self._accounts}
        result: list[dict[str, Any]] = []
        unique_ids: set[str] = set()
        for account_id, raw in zip(missing_ids, rows, strict=True):
            if not isinstance(raw, Mapping):
                raise ValueError("worker pool creation row is not an object")
            row = deepcopy(dict(raw))
            _exact_keys(row, _CREATION_ROW_KEYS, "worker pool creation row")
            desired = expected[account_id]
            unique_id = str(row.get("unique_id", ""))
            if (
                row.get("account_id") != account_id
                or row.get("email") != desired["email"]
                or row.get("name") != desired["name"]
                or row.get("project_id") != PROJECT
                or _UNIQUE_ID.fullmatch(unique_id) is None
                or row.get("disabled") is not False
                or row.get("http_method") != "POST"
                or row.get("http_status") not in {200, 201}
                or row.get("created") is not True
                or row.get("already_existed") is not False
                or unique_id in unique_ids
            ):
                raise FileExistsError(
                    "worker pool creation did not prove a new exact missing account"
                )
            unique_ids.add(unique_id)
            row["unique_id"] = unique_id
            result.append(row)
        return result

    def validate_create_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self._read_receipt is None or self._setup_plan is None:
            raise PermissionError("worker pool create validation requires setup/read evidence")
        body, digest = _pop_receipt(
            value, keys=_CREATE_KEYS, label="worker pool GCP create receipt"
        )
        creation = self._validate_creation_rows(body.get("creation_rows"))
        inventory_value = body.get("inventory_receipt")
        if not isinstance(inventory_value, Mapping):
            raise ValueError("worker pool create receipt lacks final inventory")
        inventory = identity_v2.validate_inventory_receipt(
            identity_plan=self._identity,
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            value=inventory_value,
        )
        all_missing = self._read_receipt["all_accounts_missing"] is True
        setup_value = body.get("setup_receipt")
        if all_missing:
            if not isinstance(setup_value, Mapping):
                raise ValueError("all-new worker pool create lacks setup receipt")
            setup = identity_v2.validate_create_only_setup_receipt(
                setup_plan=self._setup_plan, value=setup_value
            )
        else:
            if setup_value is not None:
                raise ValueError("partial worker pool create cannot claim one-time setup")
            setup = None
        completed = _timestamp(body.get("completed_at_utc"), "pool create completion")
        if completed < _timestamp(
            self._read_receipt["observed_at_utc"], "pool read time"
        ):
            raise ValueError("worker pool create receipt predates read evidence")
        missing_ids = self._read_receipt["missing_account_ids"]
        if (
            body.get("schema") != POOL_CREATE_RECEIPT_SCHEMA
            or body.get("status")
            != ("all_eight_fixed_worker_accounts_created_once" if all_missing else "missing_fixed_worker_accounts_created")
            or body.get("identity_plan_sha256") != self._identity["plan_sha256"]
            or body.get("setup_plan_sha256") != self._setup_plan["plan_sha256"]
            or body.get("read_receipt_sha256")
            != self._read_receipt["receipt_sha256"]
            or body.get("project") != PROJECT
            or body.get("created_account_ids") != missing_ids
            or body.get("created_account_count") != len(missing_ids)
            or body.get("preexisting_account_count")
            != identity_v2.POOL_SIZE - len(missing_ids)
            or body.get("setup_receipt_sha256")
            != (None if setup is None else setup["receipt_sha256"])
            or body.get("inventory_receipt_sha256") != inventory["receipt_sha256"]
            or body.get("all_pool_accounts_exist_after_create") is not True
            or body.get("all_created_accounts_read_back_exact") is not True
            or body.get("create_only_missing_accounts") is not True
            or body.get("existing_account_adopted_from_collision") is not False
            or body.get("roles_added") is not False
            or body.get("accounts_deleted") is not False
            or body.get("provider_source") != "iam.googleapis.com/v1"
            or body.get("http_post_count") != len(missing_ids)
            or body.get("http_get_count")
            != identity_v2.POOL_SIZE * len(missing_ids)
            or body.get("cloud_mutated") is not True
            or body.get("current_profile_changed") is not False
        ):
            raise ValueError("worker pool GCP create receipt contract changed")
        body["creation_rows"] = creation
        body["setup_receipt"] = setup
        body["inventory_receipt"] = inventory
        return {**body, "receipt_sha256": digest}

    def create_missing(
        self, *, current_time_utc: str, completed_at_utc: str
    ) -> dict[str, Any]:
        if (
            self.mode != "create-missing"
            or self._used
            or self._read_receipt is None
            or self._setup_plan is None
        ):
            raise PermissionError("worker pool create-missing mode is unavailable or consumed")
        self._used = True
        self._read_receipt = self.validate_read_receipt(
            self._read_receipt, current_time_utc=current_time_utc
        )
        current_time = _timestamp(current_time_utc, "pool create current time")
        completed_time = _timestamp(completed_at_utc, "pool create completion")
        if completed_time < current_time:
            raise ValueError("worker pool create completion predates execution time")
        missing_ids = self._read_receipt["missing_account_ids"]
        if not missing_ids:
            raise PermissionError(
                "complete worker pool must be accepted only through read-only inventory"
            )
        expected = {row["account_id"]: row for row in self._accounts}
        creation: list[dict[str, Any]] = []
        final_inventory_rows: list[dict[str, Any]] = []
        for account_id in missing_ids:
            desired = expected[account_id]
            request_body = {
                "accountId": desired["account_id"],
                "serviceAccount": {
                    "displayName": desired["display_name"],
                    "description": desired["description"],
                },
            }
            try:
                response = self._http(
                    method="POST",
                    url=_accounts_url(),
                    body=canonical_bytes(request_body),
                    content_type="application/json",
                    allowed_statuses=(200, 201),
                )
            except Exception as exc:
                if "collision/precondition" in str(exc):
                    raise
                recovery_rows, recovery_inventory = self._read_all_accounts()
                observed = datetime.fromtimestamp(
                    _timestamp(completed_at_utc, "pool recovery time"),
                    tz=timezone.utc,
                )
                expires = observed.replace(microsecond=0).timestamp() + (
                    MAX_READ_RECEIPT_LIFETIME_SECONDS
                )
                expires_at = datetime.fromtimestamp(
                    expires, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                recovery = self._build_read_receipt(
                    rows=recovery_rows,
                    inventory_rows=recovery_inventory,
                    observed_at_utc=completed_at_utc,
                    expires_at_utc=expires_at,
                )
                raise WorkerIdentityCreateIncompleteError(
                    "worker identity create response was ambiguous",
                    recovery_read_receipt=recovery,
                ) from exc
            observed = _provider_account(
                self._json(response, "IAM service account create"), desired
            )
            creation.append(
                {
                    "account_id": desired["account_id"],
                    "email": desired["email"],
                    "name": desired["name"],
                    "project_id": PROJECT,
                    "unique_id": observed["unique_id"],
                    "disabled": False,
                    "http_method": "POST",
                    "http_status": response.status,
                    "created": True,
                    "already_existed": False,
                }
            )
            _, final_inventory_rows = self._read_all_accounts()
            if len(final_inventory_rows) < (
                identity_v2.POOL_SIZE - len(missing_ids) + len(creation)
            ):
                raise RuntimeError("worker pool post-create readback lost an exact account")
        if len(final_inventory_rows) != identity_v2.POOL_SIZE:
            raise RuntimeError("worker pool final readback is incomplete after create")
        inventory = identity_v2.build_inventory_receipt(
            identity_plan=self._identity,
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            observed_accounts=final_inventory_rows,
            observed_at_utc=completed_at_utc,
            provider_source="iam.googleapis.com/v1",
        )
        all_missing = self._read_receipt["all_accounts_missing"] is True
        setup = None
        if all_missing:
            absence = [
                {
                    "account_id": row["account_id"],
                    "email": row["email"],
                    "name": row["name"],
                    "http_method": "GET",
                    "http_status": 404,
                    "exists": False,
                }
                for row in self._read_receipt["rows"]
            ]
            setup = identity_v2.build_create_only_setup_receipt(
                setup_plan=self._setup_plan,
                absence_observations=absence,
                creation_observations=creation,
                completed_at_utc=completed_at_utc,
                provider_source="iam.googleapis.com/v1",
            )
        body = {
            "schema": POOL_CREATE_RECEIPT_SCHEMA,
            "status": (
                "all_eight_fixed_worker_accounts_created_once"
                if all_missing
                else "missing_fixed_worker_accounts_created"
            ),
            "identity_plan_sha256": self._identity["plan_sha256"],
            "setup_plan_sha256": self._setup_plan["plan_sha256"],
            "read_receipt_sha256": self._read_receipt["receipt_sha256"],
            "project": PROJECT,
            "completed_at_utc": completed_at_utc,
            "created_account_ids": list(missing_ids),
            "created_account_count": len(missing_ids),
            "preexisting_account_count": identity_v2.POOL_SIZE - len(missing_ids),
            "creation_rows": creation,
            "setup_receipt": setup,
            "setup_receipt_sha256": None if setup is None else setup["receipt_sha256"],
            "inventory_receipt": inventory,
            "inventory_receipt_sha256": inventory["receipt_sha256"],
            "all_pool_accounts_exist_after_create": True,
            "all_created_accounts_read_back_exact": True,
            "create_only_missing_accounts": True,
            "existing_account_adopted_from_collision": False,
            "roles_added": False,
            "accounts_deleted": False,
            "provider_source": "iam.googleapis.com/v1",
            "http_post_count": self._http_post_count,
            "http_get_count": self._http_get_count,
            "cloud_mutated": True,
            "current_profile_changed": False,
        }
        return self.validate_create_receipt(_seal(body))

    def validate_act_as_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self._inventory is None:
            raise PermissionError("actAs validation requires fixed-pool inventory")
        return identity_v2.validate_act_as_receipt(
            identity_plan=self._identity,
            inventory_receipt=self._inventory,
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            value=value,
        )

    def check_service_account_act_as(
        self, *, tested_at_utc: str
    ) -> dict[str, Any]:
        """Test exactly actAs for every selected fixed-pool account."""

        if self.mode != "actas-check" or self._used or self._inventory is None:
            raise PermissionError("worker pool actAs-check mode is unavailable or consumed")
        _timestamp(tested_at_utc, "actAs test time")
        self._used = True
        request_body = canonical_bytes(
            {"permissions": [identity_v2.ACT_AS_PERMISSION]}
        )
        observations: list[dict[str, Any]] = []
        for worker in self._identity["selected_workers"]:
            response = self._http(
                method="POST",
                url=_actas_url(worker["service_account_name"]),
                body=request_body,
                content_type="application/json",
            )
            provider = self._json(response, "service-account testIamPermissions")
            if set(provider) != {"permissions"}:
                raise RuntimeError("testIamPermissions response fields changed")
            granted = provider.get("permissions")
            if granted != [identity_v2.ACT_AS_PERMISSION]:
                raise PermissionError(
                    "selected worker does not grant exactly iam.serviceAccounts.actAs"
                )
            observations.append(
                {
                    "account_id": worker["account_id"],
                    "email": worker["service_account_email"],
                    "name": worker["service_account_name"],
                    "http_method": "POST",
                    "requested_permissions": [identity_v2.ACT_AS_PERMISSION],
                    "granted_permissions": [identity_v2.ACT_AS_PERMISSION],
                }
            )
        if self._http_post_count != self._identity["selected_count"]:
            raise RuntimeError("actAs request count changed")
        receipt = identity_v2.build_act_as_receipt(
            identity_plan=self._identity,
            inventory_receipt=self._inventory,
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            test_iam_permissions_observations=observations,
            tested_at_utc=tested_at_utc,
            provider_source="iam.googleapis.com/v1",
        )
        return self.validate_act_as_receipt(receipt)

    def validate_project_iam_scan_receipt(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self._inventory is None:
            raise PermissionError("project IAM validation requires fixed-pool inventory")
        return identity_v2.validate_project_iam_scan_receipt(
            identity_plan=self._identity,
            inventory_receipt=self._inventory,
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            value=value,
        )

    def scan_project_iam(self, *, observed_at_utc: str) -> dict[str, Any]:
        """Read project IAM once and prove the fixed pool has no project roles."""

        if (
            self.mode != "project-iam-scan"
            or self._used
            or self._inventory is None
        ):
            raise PermissionError(
                "worker pool project-iam-scan mode is unavailable or consumed"
            )
        _timestamp(observed_at_utc, "project IAM observation time")
        self._used = True
        response = self._http(
            method="POST",
            url=_project_policy_url(),
            body=canonical_bytes({"options": {"requestedPolicyVersion": 3}}),
            content_type="application/json",
        )
        policy = self._json(response, "project getIamPolicy")
        if self._http_post_count != 1:
            raise RuntimeError("project IAM request count changed")
        receipt = identity_v2.build_project_iam_scan_receipt(
            identity_plan=self._identity,
            inventory_receipt=self._inventory,
            wave_plan=self._wave_plan,
            attempt_ledger=self._attempt_ledger,
            resume_plan=self._resume_plan,
            project_iam_policy=policy,
            observed_at_utc=observed_at_utc,
            provider_source="cloudresourcemanager.googleapis.com/v1",
        )
        return self.validate_project_iam_scan_receipt(receipt)


__all__ = [
    "MAX_READ_RECEIPT_LIFETIME_SECONDS",
    "POOL_CREATE_RECEIPT_SCHEMA",
    "POOL_READ_RECEIPT_SCHEMA",
    "PROJECT",
    "TOKEN_ENV",
    "WorkerIdentityGcpAdapterV2",
    "WorkerIdentityCreateIncompleteError",
    "canonical_bytes",
    "canonical_sha256",
]
