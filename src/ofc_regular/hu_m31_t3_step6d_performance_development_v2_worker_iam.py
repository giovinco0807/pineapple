"""Run-scoped bucket-IAM lifecycle for the perfdev-v2 T3 workers.

This module owns exactly two conditional bucket-IAM bindings.  It has no VM,
GCS-object, service-account, role-mutation, retry, or ambient-credential
surface.  The real adapter reads one user-provided OAuth token from the fixed
``GOOGLE_OAUTH_ACCESS_TOKEN`` environment variable and permits only the two
custom-role GET endpoints, read-only project get-IAM-policy POST, and bucket
get/set-IAM-policy GET/PUT endpoints.

All lifecycle receipts are secret-free and immutable when written through
``write_json_once``.  ETags are used for the one-shot compare-and-swap but only
their SHA-256 digests are recorded.  Any stale binding, duplicate, condition
drift, role drift, ETag drift, or unrelated-policy drift fails closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from . import (
    hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 as cloud,
)


IAM_PLAN_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_worker_iam_plan_v1"
AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_explicit_authorization_v1"
)
EXPIRED_EXACT2_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_expired_exact2_authorization_v1"
)
PREPARE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_prepare_receipt_v1"
)
INSTALL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_install_receipt_v1"
)
READBACK_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_readback_receipt_v1"
)
CLEANUP_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_perfdev_v2_worker_iam_cleanup_receipt_v1"
)

PROJECT = "ofc-solver-485418"
BUCKET = "pokerhu-ofc-solver-485418-training"
WORKER_SERVICE_ACCOUNT = (
    "ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com"
)
WORKER_PRINCIPAL = f"serviceAccount:{WORKER_SERVICE_ACCOUNT}"
READER_ROLE = f"projects/{PROJECT}/roles/ofcM31T3ObjectReaderV1"
CREATOR_ROLE = f"projects/{PROJECT}/roles/ofcM31T3ResultCreatorV1"
ROLE_PERMISSIONS = {
    READER_ROLE: ("storage.objects.get",),
    CREATOR_ROLE: ("storage.objects.create",),
}

TOKEN_ENVIRONMENT_VARIABLE = "GOOGLE_OAUTH_ACCESS_TOKEN"
NONCE_ENVIRONMENT_VARIABLE = "OFC_M31_T3_PERFDEV_V2_IAM_NONCE"
EXPIRED_CLEANUP_NONCE_ENVIRONMENT_VARIABLE = (
    "OFC_M31_T3_PERFDEV_V2_IAM_EXPIRED_CLEANUP_NONCE"
)
REQUEST_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
CONDITION_LIFETIME_SECONDS = 5_400
MAX_CONDITION_LIFETIME_SECONDS = 5_400
EXPIRED_EXACT2_GRACE_SECONDS = 600
EXPIRED_EXACT2_AUTHORIZATION_LIFETIME_SECONDS = 300

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE_ETAG = re.compile(r"^[A-Za-z0-9_+/=-]{1,512}$")
_BEARER = re.compile(r"^[\x21-\x7e]{16,8192}$")
_POLICY_KEYS = frozenset(
    {"version", "etag", "bindings", "auditConfigs", "kind", "resourceId"}
)
_BINDING_KEYS = frozenset({"role", "members", "condition"})
_CONDITION_KEYS = frozenset({"title", "description", "expression", "location"})
_ROLE_KEYS = frozenset(
    {"name", "title", "description", "includedPermissions", "stage", "etag", "deleted"}
)
_ALLOWED_HOSTS = frozenset(
    {
        "iam.googleapis.com",
        "storage.googleapis.com",
        "cloudresourcemanager.googleapis.com",
    }
)


class WorkerIamError(RuntimeError):
    """Secret-free, stable failure returned by the real REST adapter."""

    def __init__(self, code: str, *, operation: str, status_code: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.operation = operation
        self.status_code = status_code

    def __str__(self) -> str:
        return self.code


class WorkerIamTransport(Protocol):
    def get_role(self, *, role_name: str) -> Mapping[str, Any]: ...

    def get_bucket_policy(self) -> Mapping[str, Any]: ...

    def get_project_policy(self) -> Mapping[str, Any]: ...

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str]


HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


def _stdlib_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: int,
) -> HttpResponse:
    parsed = urlsplit(url)
    if (
        method not in {"GET", "POST", "PUT"}
        or parsed.scheme != "https"
        or parsed.hostname not in _ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("worker IAM request escaped the HTTPS allowlist")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirectHandler()
    )
    request = urllib.request.Request(
        url=url, headers=dict(headers), data=body, method=method
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            return HttpResponse(int(response.status), raw, dict(response.headers.items()))
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_RESPONSE_BYTES + 1)
        return HttpResponse(
            int(exc.code), raw, dict(exc.headers.items()) if exc.headers else {}
        )
    except (urllib.error.URLError, TimeoutError, OSError):
        raise WorkerIamError(
            "worker_iam_https_transport_failed", operation="https_request"
        ) from None


class GcpBucketIamTransport:
    """Fixed-surface REST adapter; importing/constructing it performs no I/O."""

    _POLICY_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/iam"
    _GET_POLICY_URL = f"{_POLICY_URL}?optionsRequestedPolicyVersion=3"
    _SET_POLICY_URL = _POLICY_URL
    _GET_PROJECT_POLICY_URL = (
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{PROJECT}:getIamPolicy"
    )
    _ROLE_URLS = {
        role: f"https://iam.googleapis.com/v1/{role}" for role in ROLE_PERMISSIONS
    }

    def __init__(
        self,
        *,
        project: str = PROJECT,
        bucket: str = BUCKET,
        requester: HttpRequester = _stdlib_request,
    ) -> None:
        if project != PROJECT or bucket != BUCKET:
            raise ValueError("worker IAM adapter project or bucket changed")
        self._requester = requester

    @staticmethod
    def _token() -> str:
        value = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
        if value is None or _BEARER.fullmatch(value) is None:
            raise WorkerIamError(
                "worker_iam_oauth_token_missing_or_invalid",
                operation="read_token_environment",
            )
        return value

    def _json(
        self,
        *,
        method: str,
        url: str,
        body: Mapping[str, Any] | None,
        operation: str,
    ) -> dict[str, Any]:
        if method not in {"GET", "POST", "PUT"} or url not in {
            *self._ROLE_URLS.values(),
            self._GET_POLICY_URL,
            self._SET_POLICY_URL,
            self._GET_PROJECT_POLICY_URL,
        }:
            raise ValueError("worker IAM REST surface changed")
        payload = None if body is None else canonical_bytes(dict(body))
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token()}",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        try:
            response = self._requester(
                method, url, headers, payload, REQUEST_TIMEOUT_SECONDS
            )
        except WorkerIamError:
            raise
        except BaseException:
            raise WorkerIamError(
                "worker_iam_https_transport_failed", operation=operation
            ) from None
        if response.status_code != 200:
            raise WorkerIamError(
                "worker_iam_rest_request_rejected",
                operation=operation,
                status_code=response.status_code,
            )
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise WorkerIamError(
                "worker_iam_rest_response_too_large", operation=operation
            )
        try:
            value = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise WorkerIamError(
                "worker_iam_rest_response_invalid_json", operation=operation
            ) from None
        if not isinstance(value, dict):
            raise WorkerIamError(
                "worker_iam_rest_response_not_object", operation=operation
            )
        return value

    def get_role(self, *, role_name: str) -> Mapping[str, Any]:
        if role_name not in self._ROLE_URLS:
            raise ValueError("worker IAM role escaped exact custom-role set")
        return self._json(
            method="GET",
            url=self._ROLE_URLS[role_name],
            body=None,
            operation=f"get_role:{role_name}",
        )

    def get_bucket_policy(self) -> Mapping[str, Any]:
        return self._json(
            method="GET",
            url=self._GET_POLICY_URL,
            body=None,
            operation="get_bucket_iam_policy",
        )

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._json(
            method="PUT",
            url=self._SET_POLICY_URL,
            body=copy.deepcopy(dict(policy)),
            operation="set_bucket_iam_policy",
        )

    def get_project_policy(self) -> Mapping[str, Any]:
        return self._json(
            method="POST",
            url=self._GET_PROJECT_POLICY_URL,
            body={"options": {"requestedPolicyVersion": 3}},
            operation="get_project_iam_policy_read_only",
        )


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_bytes(dict(value)))
        handle.flush()
        os.fsync(handle.fileno())


def read_canonical_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rfc3339(unix_seconds: int) -> str:
    if isinstance(unix_seconds, bool) or not isinstance(unix_seconds, int):
        raise ValueError("worker IAM timestamp is not an integer")
    return (
        datetime.fromtimestamp(unix_seconds, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _unix_seconds(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} is not exact UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not exact UTC RFC3339") from exc
    if parsed.microsecond or parsed.tzinfo is None:
        raise ValueError(f"{label} is not exact UTC RFC3339")
    seconds = int(parsed.timestamp())
    if _rfc3339(seconds) != value:
        raise ValueError(f"{label} is not canonical UTC RFC3339")
    return seconds


def _seal(value: Mapping[str, Any], digest_field: str = "receipt_sha256") -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    if digest_field in body:
        raise ValueError("sealed value already contains its digest")
    return {**body, digest_field: canonical_sha256(body)}


def _validate_seal(
    value: Mapping[str, Any], *, label: str, digest_field: str = "receipt_sha256"
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    supplied = _sha(checked.pop(digest_field, None), f"{label} digest")
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return dict(value)


def _validated_execution_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = cloud.validate_execution_plan(value, require_fresh_receipt=False)
    if (
        plan.get("project") != PROJECT
        or plan.get("worker_identity", {}).get("service_account")
        != WORKER_SERVICE_ACCOUNT
    ):
        raise ValueError("execution plan worker IAM boundary changed")
    return plan


def _condition(
    *, purpose: str, prefix: str, execution_plan_sha256: str, nonce_sha256: str,
    expires_at_utc: str,
) -> dict[str, str]:
    if purpose not in {"reader", "creator"}:
        raise ValueError("worker IAM condition purpose changed")
    return {
        "title": (
            f"ofc-m31-pdv2-{purpose}-{execution_plan_sha256[:12]}-"
            f"{nonce_sha256[:12]}"
        ),
        "description": (
            f"execution_plan_sha256={execution_plan_sha256};"
            f"one_shot_nonce_sha256={nonce_sha256};purpose={purpose};"
            f"expires_at_utc={expires_at_utc}"
        ),
        "expression": (
            f'resource.name.startsWith("projects/_/buckets/{BUCKET}/objects/{prefix}") '
            f'&& request.time < timestamp("{expires_at_utc}")'
        ),
    }


def build_iam_plan(
    *, execution_plan: Mapping[str, Any], one_shot_nonce_sha256: str,
    issued_at_unix_seconds: int,
) -> dict[str, Any]:
    plan = _validated_execution_plan(execution_plan)
    nonce_sha = _sha(one_shot_nonce_sha256, "one-shot nonce")
    if isinstance(issued_at_unix_seconds, bool) or not isinstance(
        issued_at_unix_seconds, int
    ):
        raise ValueError("worker IAM plan issuance time changed")
    issued_at_utc = _rfc3339(issued_at_unix_seconds)
    expires_at_utc = _rfc3339(
        issued_at_unix_seconds + CONDITION_LIFETIME_SECONDS
    )
    execution_sha = canonical_sha256(plan)
    result_prefix = plan["result_prefix"]
    bindings = [
        {
            "role": READER_ROLE,
            "members": [WORKER_PRINCIPAL],
            "condition": _condition(
                purpose="reader",
                prefix=f"{result_prefix}control/",
                execution_plan_sha256=execution_sha,
                nonce_sha256=nonce_sha,
                expires_at_utc=expires_at_utc,
            ),
        },
        {
            "role": CREATOR_ROLE,
            "members": [WORKER_PRINCIPAL],
            "condition": _condition(
                purpose="creator",
                prefix=result_prefix,
                execution_plan_sha256=execution_sha,
                nonce_sha256=nonce_sha,
                expires_at_utc=expires_at_utc,
            ),
        },
    ]
    body = {
        "schema": IAM_PLAN_SCHEMA,
        "status": "planned_not_prepared_not_authorized",
        "project": PROJECT,
        "bucket": BUCKET,
        "worker_service_account": WORKER_SERVICE_ACCOUNT,
        "worker_principal": WORKER_PRINCIPAL,
        "run_name": plan["run_name"],
        "result_prefix": result_prefix,
        "execution_plan": copy.deepcopy(plan),
        "execution_plan_sha256": execution_sha,
        "one_shot_nonce_sha256": nonce_sha,
        "issued_at_utc": issued_at_utc,
        "expires_at_utc": expires_at_utc,
        "condition_lifetime_seconds": CONDITION_LIFETIME_SECONDS,
        "role_contracts": [
            {"role": role, "included_permissions": list(ROLE_PERMISSIONS[role])}
            for role in (READER_ROLE, CREATOR_ROLE)
        ],
        "expected_bindings": bindings,
        "exact_binding_count": 2,
        "bucket_iam_only": True,
        "service_account_act_as_binding_managed": False,
        "object_list_required": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def validate_iam_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(dict(value))
    supplied = _sha(plan.pop("plan_sha256", None), "worker IAM plan")
    expected_fields = {
        "schema",
        "status",
        "project",
        "bucket",
        "worker_service_account",
        "worker_principal",
        "run_name",
        "result_prefix",
        "execution_plan",
        "execution_plan_sha256",
        "one_shot_nonce_sha256",
        "issued_at_utc",
        "expires_at_utc",
        "condition_lifetime_seconds",
        "role_contracts",
        "expected_bindings",
        "exact_binding_count",
        "bucket_iam_only",
        "service_account_act_as_binding_managed",
        "object_list_required",
        "cloud_mutated",
        "current_profile_changed",
    }
    if set(plan) != expected_fields or canonical_sha256(plan) != supplied:
        raise ValueError("worker IAM plan fields or digest changed")
    execution = _validated_execution_plan(plan["execution_plan"])
    issued = _unix_seconds(plan.get("issued_at_utc"), "worker IAM issued_at_utc")
    expires = _unix_seconds(plan.get("expires_at_utc"), "worker IAM expires_at_utc")
    if (
        expires - issued != CONDITION_LIFETIME_SECONDS
        or plan.get("condition_lifetime_seconds") != CONDITION_LIFETIME_SECONDS
        or CONDITION_LIFETIME_SECONDS > MAX_CONDITION_LIFETIME_SECONDS
    ):
        raise ValueError("worker IAM bounded expiry changed")
    rebuilt = build_iam_plan(
        execution_plan=execution,
        one_shot_nonce_sha256=_sha(
            plan.get("one_shot_nonce_sha256"), "worker IAM plan nonce"
        ),
        issued_at_unix_seconds=issued,
    )
    if rebuilt != value:
        raise ValueError("worker IAM plan no longer derives from execution plan")
    return dict(value)


def build_explicit_authorization(
    *, iam_plan: Mapping[str, Any], raw_nonce: str, operation: str
) -> dict[str, Any]:
    plan = validate_iam_plan(iam_plan)
    if operation not in {"apply", "cleanup"}:
        raise ValueError("worker IAM authorization operation changed")
    if not isinstance(raw_nonce, str) or len(raw_nonce) < 16:
        raise ValueError("raw one-shot nonce is invalid")
    nonce_sha = _hash_text(raw_nonce)
    if nonce_sha != plan["one_shot_nonce_sha256"]:
        raise ValueError("raw one-shot nonce does not match IAM plan")
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "operation": operation,
            "explicit_authorization": True,
            "iam_plan_sha256": plan["plan_sha256"],
            "execution_plan_sha256": plan["execution_plan_sha256"],
            "one_shot_nonce_sha256": nonce_sha,
            "project": PROJECT,
            "run_name": plan["run_name"],
            "current_profile_changed": False,
        },
        "authorization_sha256",
    )


def _validate_authorization(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], raw_nonce: str, operation: str
) -> dict[str, Any]:
    auth = _validate_seal(
        value, label="worker IAM authorization", digest_field="authorization_sha256"
    )
    expected = build_explicit_authorization(
        iam_plan=plan, raw_nonce=raw_nonce, operation=operation
    )
    if auth != expected:
        raise ValueError("worker IAM explicit authorization binding changed")
    return auth


def build_expired_exact2_cleanup_authorization(
    *, iam_plan: Mapping[str, Any], raw_cleanup_nonce: str,
    issued_at_unix_seconds: int,
) -> dict[str, Any]:
    """Authorize one short-lived cleanup after the exact2 condition is inert.

    This intentionally uses a new cleanup nonce rather than the launch nonce.
    The stale exact2 bindings therefore remain recoverable even if the launch
    controller and its raw nonce are gone.  This authorization never claims
    that zero VMs were created; it authorizes only removal of the two exact,
    already-expired bucket bindings.
    """

    plan = validate_iam_plan(iam_plan)
    try:
        parsed_nonce = uuid.UUID(raw_cleanup_nonce)
    except (AttributeError, ValueError) as exc:
        raise ValueError("expired exact2 cleanup nonce is not canonical UUIDv4") from exc
    if parsed_nonce.version != 4 or str(parsed_nonce) != raw_cleanup_nonce:
        raise ValueError("expired exact2 cleanup nonce is not canonical UUIDv4")
    cleanup_nonce_sha = _hash_text(raw_cleanup_nonce)
    if cleanup_nonce_sha == plan["one_shot_nonce_sha256"]:
        raise ValueError("expired exact2 cleanup nonce reused the launch nonce")
    if isinstance(issued_at_unix_seconds, bool) or not isinstance(
        issued_at_unix_seconds, int
    ):
        raise ValueError("expired exact2 authorization issuance time changed")
    condition_expires = _unix_seconds(
        plan["expires_at_utc"], "worker IAM expires_at_utc"
    )
    cleanup_not_before = condition_expires + EXPIRED_EXACT2_GRACE_SECONDS
    if issued_at_unix_seconds < cleanup_not_before:
        raise ValueError("expired exact2 authorization precedes expiry grace")
    return _seal(
        {
            "schema": EXPIRED_EXACT2_AUTHORIZATION_SCHEMA,
            "operation": "cleanup-expired-exact2",
            "explicit_authorization": True,
            "iam_plan_sha256": plan["plan_sha256"],
            "execution_plan_sha256": plan["execution_plan_sha256"],
            "one_shot_nonce_sha256": plan["one_shot_nonce_sha256"],
            "cleanup_nonce_sha256": cleanup_nonce_sha,
            "project": PROJECT,
            "run_name": plan["run_name"],
            "condition_expires_at_unix_seconds": condition_expires,
            "required_grace_seconds": EXPIRED_EXACT2_GRACE_SECONDS,
            "cleanup_not_before_unix_seconds": cleanup_not_before,
            "issued_at_unix_seconds": issued_at_unix_seconds,
            "valid_until_unix_seconds": (
                issued_at_unix_seconds
                + EXPIRED_EXACT2_AUTHORIZATION_LIFETIME_SECONDS
            ),
            "removal_scope": "exact_two_plan_owned_bucket_bindings_only",
            "zero_created_claimed": False,
            "current_profile_changed": False,
        },
        "authorization_sha256",
    )


def _validate_expired_exact2_cleanup_authorization(
    value: Mapping[str, Any], *, plan: Mapping[str, Any],
    raw_cleanup_nonce: str, now_unix_seconds: int | None,
) -> tuple[dict[str, Any], int]:
    auth = _validate_seal(
        value,
        label="expired exact2 cleanup authorization",
        digest_field="authorization_sha256",
    )
    issued = auth.get("issued_at_unix_seconds")
    if isinstance(issued, bool) or not isinstance(issued, int):
        raise ValueError("expired exact2 authorization issuance time changed")
    if auth.get("cleanup_nonce_sha256") == plan["one_shot_nonce_sha256"]:
        raise ValueError("expired exact2 cleanup nonce reused the launch nonce")
    expected = build_expired_exact2_cleanup_authorization(
        iam_plan=plan,
        raw_cleanup_nonce=raw_cleanup_nonce,
        issued_at_unix_seconds=issued,
    )
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    if isinstance(now, bool) or not isinstance(now, int):
        raise ValueError("expired exact2 cleanup observation time changed")
    if (
        auth != expected
        or now < auth["cleanup_not_before_unix_seconds"]
        or now < auth["issued_at_unix_seconds"]
        or now > auth["valid_until_unix_seconds"]
    ):
        raise ValueError("expired exact2 cleanup authorization is not live")
    return auth, now


def _validated_role(value: Mapping[str, Any], *, role_name: str) -> dict[str, Any]:
    raw = copy.deepcopy(dict(value))
    if not set(raw).issubset(_ROLE_KEYS) or not {
        "name",
        "includedPermissions",
        "stage",
        "etag",
    }.issubset(raw):
        raise ValueError("worker IAM custom role fields changed")
    title = raw.get("title")
    description = raw.get("description")
    permissions = raw.get("includedPermissions")
    etag = raw.get("etag")
    if (
        raw.get("name") != role_name
        or raw.get("stage") != "GA"
        or raw.get("deleted", False) is not False
        or not isinstance(title, (str, type(None)))
        or not isinstance(description, (str, type(None)))
        or not isinstance(permissions, list)
        or permissions != list(ROLE_PERMISSIONS[role_name])
        or not isinstance(etag, str)
        or _ROLE_ETAG.fullmatch(etag) is None
    ):
        raise ValueError("worker IAM custom role contract changed")
    normalized = {
        "name": role_name,
        "title": title,
        "description": description,
        "included_permissions": list(permissions),
        "stage": "GA",
        "deleted": False,
    }
    return {
        "role_name": role_name,
        "included_permissions": list(permissions),
        "stage": "GA",
        "deleted": False,
        "role_etag_sha256": _hash_text(etag),
        "role_fingerprint_sha256": canonical_sha256(normalized),
    }


def _validate_condition(value: Mapping[str, Any]) -> dict[str, str]:
    condition = copy.deepcopy(dict(value))
    if (
        not set(condition).issubset(_CONDITION_KEYS)
        or not {"title", "expression"}.issubset(condition)
        or any(not isinstance(item, str) for item in condition.values())
        or not condition["title"]
        or not condition["expression"]
    ):
        raise ValueError("bucket IAM condition fields changed")
    return condition


def _validated_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    policy = copy.deepcopy(dict(value))
    if not set(policy).issubset(_POLICY_KEYS):
        raise ValueError("bucket IAM policy fields changed")
    version = policy.get("version", 0)
    etag = policy.get("etag")
    bindings = policy.get("bindings", [])
    audit_configs = policy.get("auditConfigs", [])
    kind = policy.get("kind")
    resource_id = policy.get("resourceId")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in {0, 1, 3}
        or not isinstance(etag, str)
        or not etag
        or not isinstance(bindings, list)
        or not isinstance(audit_configs, list)
        or (kind is not None and kind != "storage#policy")
        or (
            resource_id is not None
            and resource_id != f"projects/_/buckets/{BUCKET}"
        )
    ):
        raise ValueError("bucket IAM policy header changed")
    seen_identities: set[str] = set()
    normalized_bindings: list[dict[str, Any]] = []
    for raw in bindings:
        if not isinstance(raw, Mapping) or not set(raw).issubset(_BINDING_KEYS):
            raise ValueError("bucket IAM binding fields changed")
        if set(raw) not in ({"role", "members"}, {"role", "members", "condition"}):
            raise ValueError("bucket IAM binding shape changed")
        role = raw.get("role")
        members = raw.get("members")
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(members, list)
            or not members
            or any(not isinstance(member, str) or not member for member in members)
            or len(set(members)) != len(members)
        ):
            raise ValueError("bucket IAM binding values changed")
        normalized: dict[str, Any] = {"role": role, "members": list(members)}
        if "condition" in raw:
            if not isinstance(raw["condition"], Mapping):
                raise ValueError("bucket IAM condition changed")
            normalized["condition"] = _validate_condition(raw["condition"])
            if version != 3:
                raise ValueError("conditional bucket IAM policy is not version 3")
        identity = canonical_sha256(
            {"role": role, "condition": normalized.get("condition")}
        )
        if identity in seen_identities:
            raise ValueError("duplicate bucket IAM role/condition binding")
        seen_identities.add(identity)
        normalized_bindings.append(normalized)
    result = {
        "version": version,
        "etag": etag,
        "bindings": normalized_bindings,
        "auditConfigs": copy.deepcopy(audit_configs),
    }
    if kind is not None:
        result["kind"] = kind
    if resource_id is not None:
        result["resourceId"] = resource_id
    return result


def _binding_identity(binding: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {"role": binding["role"], "condition": binding.get("condition")}
    )


def _sorted_bindings(bindings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in bindings:
        item = copy.deepcopy(dict(row))
        item["members"] = sorted(item["members"])
        normalized.append(item)
    return sorted(normalized, key=canonical_sha256)


def _policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "version": policy["version"],
            "bindings": _sorted_bindings(policy["bindings"]),
            "auditConfigs": policy["auditConfigs"],
            "kind": policy.get("kind"),
            "resourceId": policy.get("resourceId"),
        }
    )


def _expected_bindings(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(list(plan["expected_bindings"]))


def _is_exact(binding: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return dict(binding) == dict(expected)


def _assert_initial_zero(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    expected = _expected_bindings(plan)
    titles = {row["condition"]["title"] for row in expected}
    expressions = {row["condition"]["expression"] for row in expected}
    for binding in policy["bindings"]:
        condition = binding.get("condition", {})
        if WORKER_PRINCIPAL in binding["members"]:
            raise ValueError("stale or drifted worker custom-role binding exists")
        if (
            condition.get("title") in titles
            or condition.get("expression") in expressions
            or any(_is_exact(binding, row) for row in expected)
        ):
            raise ValueError("duplicate or drifted worker IAM condition exists")


def _assert_installed(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    expected = _expected_bindings(plan)
    exact_counts = [0, 0]
    titles = {row["condition"]["title"] for row in expected}
    expressions = {row["condition"]["expression"] for row in expected}
    for binding in policy["bindings"]:
        matches = [_is_exact(binding, row) for row in expected]
        for index, match in enumerate(matches):
            exact_counts[index] += int(match)
        condition = binding.get("condition", {})
        targeted = WORKER_PRINCIPAL in binding["members"]
        collides = (
            condition.get("title") in titles
            or condition.get("expression") in expressions
        )
        if (targeted or collides) and not any(matches):
            raise ValueError("installed worker IAM binding or condition drifted")
    if exact_counts != [1, 1]:
        raise ValueError("exact worker IAM binding count changed")


def _unrelated_policy_view(
    policy: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    expected_ids = {_binding_identity(row) for row in _expected_bindings(plan)}
    rows = [
        row
        for row in policy["bindings"]
        if _binding_identity(row) not in expected_ids
    ]
    return {
        "bindings": _sorted_bindings(rows),
        "auditConfigs": copy.deepcopy(policy["auditConfigs"]),
        "kind": policy.get("kind"),
        "resourceId": policy.get("resourceId"),
    }


def _unrelated_fingerprint(policy: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    return canonical_sha256(_unrelated_policy_view(policy, plan))


def _role_readbacks(transport: WorkerIamTransport) -> list[dict[str, Any]]:
    return [
        _validated_role(transport.get_role(role_name=role), role_name=role)
        for role in (READER_ROLE, CREATOR_ROLE)
    ]


def _project_worker_zero_readback(
    transport: WorkerIamTransport,
) -> dict[str, Any]:
    policy = _validated_policy(transport.get_project_policy())
    direct = [
        row for row in policy["bindings"] if WORKER_PRINCIPAL in row["members"]
    ]
    if direct:
        raise ValueError("dedicated worker has a direct project IAM binding")
    return {
        "read_only": True,
        "worker_principal": WORKER_PRINCIPAL,
        "direct_worker_binding_count": 0,
        "project_policy_fingerprint_sha256": _policy_fingerprint(policy),
        "project_policy_etag_sha256": _hash_text(policy["etag"]),
    }


def _validate_project_worker_zero_readback(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("project worker-zero readback changed")
    checked = dict(value)
    if (
        set(checked)
        != {
            "read_only",
            "worker_principal",
            "direct_worker_binding_count",
            "project_policy_fingerprint_sha256",
            "project_policy_etag_sha256",
        }
        or checked.get("read_only") is not True
        or checked.get("worker_principal") != WORKER_PRINCIPAL
        or checked.get("direct_worker_binding_count") != 0
    ):
        raise ValueError("project worker-zero readback changed")
    _sha(checked.get("project_policy_fingerprint_sha256"), "project policy")
    _sha(checked.get("project_policy_etag_sha256"), "project policy ETag")
    return checked


def _validate_role_readbacks(
    value: Any, *, transport: WorkerIamTransport | None = None
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("worker IAM role readback count changed")
    checked = copy.deepcopy(value)
    if [row.get("role_name") for row in checked if isinstance(row, Mapping)] != [
        READER_ROLE,
        CREATOR_ROLE,
    ]:
        raise ValueError("worker IAM role readback order changed")
    if transport is not None and _role_readbacks(transport) != checked:
        raise ValueError("worker IAM custom role readback drifted")
    return checked


def _require_live_window(
    plan: Mapping[str, Any], *, now_unix_seconds: int | None
) -> int:
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    if isinstance(now, bool) or not isinstance(now, int):
        raise ValueError("worker IAM live-window time changed")
    issued = _unix_seconds(plan["issued_at_utc"], "worker IAM issued_at_utc")
    expires = _unix_seconds(plan["expires_at_utc"], "worker IAM expires_at_utc")
    vm_ttl = plan["execution_plan"]["limits"]["vm_ttl_seconds"]
    if (
        isinstance(vm_ttl, bool)
        or not isinstance(vm_ttl, int)
        or vm_ttl <= 0
        or now < issued
        or expires - now < vm_ttl + 300
        or expires - issued > MAX_CONDITION_LIFETIME_SECONDS
    ):
        raise ValueError("worker IAM condition window is not launch-safe")
    return now


def prepare_worker_iam(
    *, iam_plan: Mapping[str, Any], transport: WorkerIamTransport,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    plan = validate_iam_plan(iam_plan)
    _require_live_window(plan, now_unix_seconds=now_unix_seconds)
    roles = _role_readbacks(transport)
    project_zero = _project_worker_zero_readback(transport)
    policy = _validated_policy(transport.get_bucket_policy())
    _assert_initial_zero(policy, plan)
    unrelated = _unrelated_policy_view(policy, plan)
    return _seal(
        {
            "schema": PREPARE_RECEIPT_SCHEMA,
            "status": "prepared_exact_zero_readback_no_mutation",
            "iam_plan_sha256": plan["plan_sha256"],
            "execution_plan_sha256": plan["execution_plan_sha256"],
            "one_shot_nonce_sha256": plan["one_shot_nonce_sha256"],
            "project": PROJECT,
            "run_name": plan["run_name"],
            "role_readbacks": roles,
            "project_worker_zero_readback": project_zero,
            "pre_bucket_policy_fingerprint_sha256": _policy_fingerprint(policy),
            "pre_bucket_policy_etag_sha256": _hash_text(policy["etag"]),
            "unrelated_policy_fingerprint_sha256": canonical_sha256(unrelated),
            "unrelated_binding_count": len(unrelated["bindings"]),
            "targeted_binding_count": 0,
            "prepared": True,
            "cloud_mutation_performed": False,
            "current_profile_changed": False,
        }
    )


def _validate_prepare_receipt(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _validate_seal(value, label="worker IAM prepare receipt")
    expected_fields = {
        "schema",
        "status",
        "iam_plan_sha256",
        "execution_plan_sha256",
        "one_shot_nonce_sha256",
        "project",
        "run_name",
        "role_readbacks",
        "project_worker_zero_readback",
        "pre_bucket_policy_fingerprint_sha256",
        "pre_bucket_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
        "unrelated_binding_count",
        "targeted_binding_count",
        "prepared",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != PREPARE_RECEIPT_SCHEMA
        or receipt.get("status") != "prepared_exact_zero_readback_no_mutation"
        or receipt.get("iam_plan_sha256") != plan["plan_sha256"]
        or receipt.get("execution_plan_sha256") != plan["execution_plan_sha256"]
        or receipt.get("one_shot_nonce_sha256") != plan["one_shot_nonce_sha256"]
        or receipt.get("project") != PROJECT
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("targeted_binding_count") != 0
        or receipt.get("prepared") is not True
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("worker IAM prepare receipt contract changed")
    _validate_role_readbacks(receipt["role_readbacks"])
    _validate_project_worker_zero_readback(receipt["project_worker_zero_readback"])
    _sha(receipt.get("pre_bucket_policy_fingerprint_sha256"), "prepare policy")
    _sha(receipt.get("pre_bucket_policy_etag_sha256"), "prepare policy ETag")
    _sha(receipt.get("unrelated_policy_fingerprint_sha256"), "prepare unrelated policy")
    if (
        isinstance(receipt.get("unrelated_binding_count"), bool)
        or not isinstance(receipt.get("unrelated_binding_count"), int)
        or receipt["unrelated_binding_count"] < 0
    ):
        raise ValueError("worker IAM unrelated binding count changed")
    return receipt


def _mutation_policy(
    policy: Mapping[str, Any], plan: Mapping[str, Any], *, install: bool
) -> dict[str, Any]:
    result = copy.deepcopy(dict(policy))
    expected = _expected_bindings(plan)
    if install:
        _assert_initial_zero(policy, plan)
        result["bindings"] = [*result["bindings"], *expected]
    else:
        _assert_installed(policy, plan)
        expected_ids = {_binding_identity(row) for row in expected}
        result["bindings"] = [
            row
            for row in result["bindings"]
            if _binding_identity(row) not in expected_ids
        ]
    result["version"] = 3
    return result


def apply_worker_iam(
    *,
    iam_plan: Mapping[str, Any],
    prepare_receipt: Mapping[str, Any],
    authorization: Mapping[str, Any],
    raw_nonce: str,
    transport: WorkerIamTransport,
    execute: bool = False,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    plan = validate_iam_plan(iam_plan)
    _require_live_window(plan, now_unix_seconds=now_unix_seconds)
    prepare = _validate_prepare_receipt(prepare_receipt, plan=plan)
    auth = _validate_authorization(
        authorization, plan=plan, raw_nonce=raw_nonce, operation="apply"
    )
    current_roles = _role_readbacks(transport)
    if current_roles != prepare["role_readbacks"]:
        raise ValueError("worker IAM custom roles changed after prepare")
    project_zero = _project_worker_zero_readback(transport)
    current = _validated_policy(transport.get_bucket_policy())
    _assert_initial_zero(current, plan)
    if (
        _policy_fingerprint(current)
        != prepare["pre_bucket_policy_fingerprint_sha256"]
        or _hash_text(current["etag"])
        != prepare["pre_bucket_policy_etag_sha256"]
        or _unrelated_fingerprint(current, plan)
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("bucket IAM policy or ETag changed after prepare")
    desired = _mutation_policy(current, plan, install=True)
    if not execute:
        return _seal(
            {
                "schema": INSTALL_RECEIPT_SCHEMA,
                "status": "dry_run_validated_no_mutation",
                "iam_plan_sha256": plan["plan_sha256"],
                "execution_plan_sha256": plan["execution_plan_sha256"],
                "one_shot_nonce_sha256": plan["one_shot_nonce_sha256"],
                "prepare_receipt_sha256": prepare["receipt_sha256"],
                "authorization_sha256": auth["authorization_sha256"],
                "project_worker_zero_readback": project_zero,
                "expected_binding_count": 2,
                "installed_binding_count": 0,
                "install_complete": False,
                "set_attempt_count": 0,
                "cloud_mutation_performed": False,
                "current_profile_changed": False,
            }
        )
    transport.set_bucket_policy(policy=desired)
    readback = _validated_policy(transport.get_bucket_policy())
    _assert_installed(readback, plan)
    if (
        _unrelated_fingerprint(readback, plan)
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("unrelated bucket IAM bindings changed during install")
    return _seal(
        {
            "schema": INSTALL_RECEIPT_SCHEMA,
            "status": "installed_exact_two_bindings_readback_validated",
            "iam_plan_sha256": plan["plan_sha256"],
            "execution_plan_sha256": plan["execution_plan_sha256"],
            "one_shot_nonce_sha256": plan["one_shot_nonce_sha256"],
            "prepare_receipt_sha256": prepare["receipt_sha256"],
            "authorization_sha256": auth["authorization_sha256"],
            "project_worker_zero_readback": project_zero,
            "expected_binding_count": 2,
            "installed_binding_count": 2,
            "install_complete": True,
            "set_attempt_count": 1,
            "post_bucket_policy_fingerprint_sha256": _policy_fingerprint(readback),
            "post_bucket_policy_etag_sha256": _hash_text(readback["etag"]),
            "unrelated_policy_fingerprint_sha256": _unrelated_fingerprint(
                readback, plan
            ),
            "cloud_mutation_performed": True,
            "current_profile_changed": False,
        }
    )


def _validate_install_receipt(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], prepare: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _validate_seal(value, label="worker IAM install receipt")
    required = {
        "schema",
        "status",
        "iam_plan_sha256",
        "execution_plan_sha256",
        "one_shot_nonce_sha256",
        "prepare_receipt_sha256",
        "authorization_sha256",
        "project_worker_zero_readback",
        "expected_binding_count",
        "installed_binding_count",
        "install_complete",
        "set_attempt_count",
        "post_bucket_policy_fingerprint_sha256",
        "post_bucket_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
    if (
        set(receipt) != required
        or receipt.get("schema") != INSTALL_RECEIPT_SCHEMA
        or receipt.get("status")
        != "installed_exact_two_bindings_readback_validated"
        or receipt.get("iam_plan_sha256") != plan["plan_sha256"]
        or receipt.get("execution_plan_sha256") != plan["execution_plan_sha256"]
        or receipt.get("one_shot_nonce_sha256") != plan["one_shot_nonce_sha256"]
        or receipt.get("prepare_receipt_sha256") != prepare["receipt_sha256"]
        or receipt.get("expected_binding_count") != 2
        or receipt.get("installed_binding_count") != 2
        or receipt.get("install_complete") is not True
        or receipt.get("set_attempt_count") != 1
        or receipt.get("cloud_mutation_performed") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("worker IAM install receipt contract changed")
    for field in (
        "authorization_sha256",
        "post_bucket_policy_fingerprint_sha256",
        "post_bucket_policy_etag_sha256",
        "unrelated_policy_fingerprint_sha256",
    ):
        _sha(receipt.get(field), f"install {field}")
    _validate_project_worker_zero_readback(receipt["project_worker_zero_readback"])
    if (
        receipt["unrelated_policy_fingerprint_sha256"]
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("worker IAM install unrelated policy binding changed")
    return receipt


def readback_worker_iam(
    *,
    iam_plan: Mapping[str, Any],
    prepare_receipt: Mapping[str, Any],
    install_receipt: Mapping[str, Any],
    transport: WorkerIamTransport,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    plan = validate_iam_plan(iam_plan)
    observed_unix_seconds = _require_live_window(
        plan, now_unix_seconds=now_unix_seconds
    )
    prepare = _validate_prepare_receipt(prepare_receipt, plan=plan)
    install = _validate_install_receipt(
        install_receipt, plan=plan, prepare=prepare
    )
    roles = _role_readbacks(transport)
    if roles != prepare["role_readbacks"]:
        raise ValueError("worker IAM custom roles changed after install")
    project_zero = _project_worker_zero_readback(transport)
    policy = _validated_policy(transport.get_bucket_policy())
    _assert_installed(policy, plan)
    if (
        _unrelated_fingerprint(policy, plan)
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("worker IAM readback found unrelated policy drift")
    launch_preflight = {
        "read_only": True,
        "service_account": WORKER_SERVICE_ACCOUNT,
        "bucket": BUCKET,
        "required_reader_prefix": f"{plan['result_prefix']}control/",
        "required_creator_prefix": plan["result_prefix"],
        "exact_conditional_binding_count": 2,
        "reader_binding_count": 1,
        "creator_binding_count": 1,
        "excess_worker_binding_count": 0,
        "iam_expiry_unix_seconds": _unix_seconds(
            plan["expires_at_utc"], "worker IAM expires_at_utc"
        ),
        "object_get_allowed": True,
        "object_create_allowed": True,
        "object_list_required": False,
        "exact_prefix_condition": True,
        "all_required_permissions_present": True,
    }
    return _seal(
        {
            "schema": READBACK_RECEIPT_SCHEMA,
            "status": "launch_preflight_exact_two_bindings_validated",
            "iam_plan_sha256": plan["plan_sha256"],
            "execution_plan_sha256": plan["execution_plan_sha256"],
            "one_shot_nonce_sha256": plan["one_shot_nonce_sha256"],
            "prepare_receipt_sha256": prepare["receipt_sha256"],
            "install_receipt_sha256": install["receipt_sha256"],
            "role_readbacks": roles,
            "project_worker_zero_readback": project_zero,
            "bucket_policy_fingerprint_sha256": _policy_fingerprint(policy),
            "bucket_policy_etag_sha256": _hash_text(policy["etag"]),
            "unrelated_policy_fingerprint_sha256": _unrelated_fingerprint(
                policy, plan
            ),
            "exact_binding_count": 2,
            "observed_unix_seconds": observed_unix_seconds,
            "launch_preflight": launch_preflight,
            "cloud_mutation_performed": False,
            "current_profile_changed": False,
        }
    )


def _validate_readback_receipt(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], prepare: Mapping[str, Any], install: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _validate_seal(value, label="worker IAM readback receipt")
    launch = receipt.get("launch_preflight")
    if (
        receipt.get("schema") != READBACK_RECEIPT_SCHEMA
        or receipt.get("status")
        != "launch_preflight_exact_two_bindings_validated"
        or receipt.get("iam_plan_sha256") != plan["plan_sha256"]
        or receipt.get("execution_plan_sha256") != plan["execution_plan_sha256"]
        or receipt.get("one_shot_nonce_sha256") != plan["one_shot_nonce_sha256"]
        or receipt.get("prepare_receipt_sha256") != prepare["receipt_sha256"]
        or receipt.get("install_receipt_sha256") != install["receipt_sha256"]
        or receipt.get("exact_binding_count") != 2
        or isinstance(receipt.get("observed_unix_seconds"), bool)
        or not isinstance(receipt.get("observed_unix_seconds"), int)
        or not (
            _unix_seconds(plan["issued_at_utc"], "worker IAM issued_at_utc")
            <= receipt["observed_unix_seconds"]
            <= _unix_seconds(plan["expires_at_utc"], "worker IAM expires_at_utc")
            - plan["execution_plan"]["limits"]["vm_ttl_seconds"]
            - 300
        )
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
        or not isinstance(launch, Mapping)
        or set(launch)
        != {
            "read_only",
            "service_account",
            "bucket",
            "required_reader_prefix",
            "required_creator_prefix",
            "exact_conditional_binding_count",
            "reader_binding_count",
            "creator_binding_count",
            "excess_worker_binding_count",
            "iam_expiry_unix_seconds",
            "object_get_allowed",
            "object_create_allowed",
            "object_list_required",
            "exact_prefix_condition",
            "all_required_permissions_present",
        }
        or launch.get("service_account") != WORKER_SERVICE_ACCOUNT
        or launch.get("bucket") != BUCKET
        or launch.get("required_reader_prefix")
        != f"{plan['result_prefix']}control/"
        or launch.get("required_creator_prefix") != plan["result_prefix"]
        or launch.get("exact_conditional_binding_count") != 2
        or launch.get("reader_binding_count") != 1
        or launch.get("creator_binding_count") != 1
        or launch.get("excess_worker_binding_count") != 0
        or launch.get("iam_expiry_unix_seconds")
        != _unix_seconds(plan["expires_at_utc"], "worker IAM expires_at_utc")
        or any(
            launch.get(field) is not expected
            for field, expected in (
                ("read_only", True),
                ("object_get_allowed", True),
                ("object_create_allowed", True),
                ("object_list_required", False),
                ("exact_prefix_condition", True),
                ("all_required_permissions_present", True),
            )
        )
    ):
        raise ValueError("worker IAM launch readback receipt contract changed")
    if set(receipt) != {
        "schema", "status", "iam_plan_sha256", "execution_plan_sha256",
        "one_shot_nonce_sha256", "prepare_receipt_sha256", "install_receipt_sha256",
        "role_readbacks", "project_worker_zero_readback",
        "bucket_policy_fingerprint_sha256",
        "bucket_policy_etag_sha256", "unrelated_policy_fingerprint_sha256",
        "exact_binding_count", "observed_unix_seconds", "launch_preflight",
        "cloud_mutation_performed",
        "current_profile_changed", "receipt_sha256",
    }:
        raise ValueError("worker IAM readback receipt fields changed")
    _validate_role_readbacks(receipt["role_readbacks"])
    _validate_project_worker_zero_readback(receipt["project_worker_zero_readback"])
    return receipt


def _cleanup_evidence(
    *, plan: Mapping[str, Any],
    collection_receipt: Mapping[str, Any] | None,
    partial_launch_receipt: Mapping[str, Any] | None,
    owned_launch_failure_closeout_receipt: Mapping[str, Any] | None,
    expired_exact2: bool,
    expired_exact2_authorization: Mapping[str, Any] | None,
) -> tuple[str, str]:
    supplied = sum(
        bool(value)
        for value in (
            collection_receipt is not None,
            partial_launch_receipt is not None,
            owned_launch_failure_closeout_receipt is not None,
            expired_exact2,
        )
    )
    if supplied != 1:
        raise ValueError("cleanup requires exactly one terminal evidence path")
    if collection_receipt is not None:
        checked = cloud._validate_collection_receipt(collection_receipt)
        if (
            checked.get("run_name") != plan["run_name"]
            or checked.get("execution_plan_sha256") != plan["execution_plan_sha256"]
        ):
            raise ValueError("collection receipt belongs to another IAM plan")
        return "complete_validated_collection", checked["receipt_content_sha256"]
    if partial_launch_receipt is not None:
        checked = cloud._validate_partial_launch_receipt(
            partial_launch_receipt, execution_plan=plan["execution_plan"]
        )
        return "owned_partial_launch", checked["receipt_content_sha256"]
    if expired_exact2:
        if expired_exact2_authorization is None:
            raise ValueError("expired exact2 authorization is absent")
        return (
            "expired_exact2_condition_inert",
            expired_exact2_authorization["authorization_sha256"],
        )

    assert owned_launch_failure_closeout_receipt is not None
    validator = getattr(
        cloud, "_validate_owned_launch_failure_closeout_receipt", None
    )
    if not callable(validator):
        raise RuntimeError("owned launch failure closeout validator is unavailable")
    checked = validator(
        owned_launch_failure_closeout_receipt,
        execution_plan=plan["execution_plan"],
    )
    if (
        not isinstance(checked, Mapping)
        or checked.get("run_name") != plan["run_name"]
        or checked.get("execution_plan_sha256")
        != plan["execution_plan_sha256"]
        or checked.get("operator_abort") is not True
        or checked.get("scientific_result_claimed") is not False
        or checked.get("all_owned_instances_deleted") is not True
        or _SHA256.fullmatch(str(checked.get("receipt_content_sha256"))) is None
    ):
        raise ValueError("owned launch failure closeout evidence changed")
    return (
        "owned_launch_failure_closeout",
        checked["receipt_content_sha256"],
    )


def cleanup_worker_iam(
    *, iam_plan: Mapping[str, Any], prepare_receipt: Mapping[str, Any],
    install_receipt: Mapping[str, Any], readback_receipt: Mapping[str, Any],
    authorization: Mapping[str, Any], raw_nonce: str | None,
    transport: WorkerIamTransport,
    collection_receipt: Mapping[str, Any] | None = None,
    partial_launch_receipt: Mapping[str, Any] | None = None,
    owned_launch_failure_closeout_receipt: Mapping[str, Any] | None = None,
    expired_exact2: bool = False,
    raw_expired_cleanup_nonce: str | None = None,
    execute: bool = False,
    now_unix_seconds: int | None = None,
) -> dict[str, Any]:
    plan = validate_iam_plan(iam_plan)
    if not isinstance(expired_exact2, bool):
        raise ValueError("expired exact2 cleanup selector changed")
    if execute and expired_exact2 and now_unix_seconds is not None:
        raise ValueError(
            "executed expired exact2 cleanup cannot use an injected clock"
        )
    prepare = _validate_prepare_receipt(prepare_receipt, plan=plan)
    install = _validate_install_receipt(install_receipt, plan=plan, prepare=prepare)
    readback = _validate_readback_receipt(
        readback_receipt, plan=plan, prepare=prepare, install=install
    )
    expired_auth: dict[str, Any] | None = None
    expired_observed_unix_seconds: int | None = None
    if expired_exact2:
        if raw_nonce is not None or raw_expired_cleanup_nonce is None:
            raise ValueError(
                "expired exact2 cleanup requires only its fresh cleanup nonce"
            )
        expired_auth, expired_observed_unix_seconds = (
            _validate_expired_exact2_cleanup_authorization(
                authorization,
                plan=plan,
                raw_cleanup_nonce=raw_expired_cleanup_nonce,
                now_unix_seconds=now_unix_seconds,
            )
        )
        auth = expired_auth
    else:
        if raw_nonce is None or raw_expired_cleanup_nonce is not None:
            raise ValueError("ordinary cleanup requires only the launch nonce")
        auth = _validate_authorization(
            authorization, plan=plan, raw_nonce=raw_nonce, operation="cleanup"
        )
    evidence_kind, evidence_sha = _cleanup_evidence(
        plan=plan,
        collection_receipt=collection_receipt,
        partial_launch_receipt=partial_launch_receipt,
        owned_launch_failure_closeout_receipt=(
            owned_launch_failure_closeout_receipt
        ),
        expired_exact2=expired_exact2,
        expired_exact2_authorization=expired_auth,
    )
    roles = _role_readbacks(transport)
    if roles != prepare["role_readbacks"] or roles != readback["role_readbacks"]:
        raise ValueError("worker IAM custom roles changed before cleanup")
    project_zero_before = _project_worker_zero_readback(transport)
    current = _validated_policy(transport.get_bucket_policy())
    _assert_installed(current, plan)
    if (
        _unrelated_fingerprint(current, plan)
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("unrelated bucket IAM policy changed before cleanup")
    desired = _mutation_policy(current, plan, install=False)
    common = {
        "schema": CLEANUP_RECEIPT_SCHEMA,
        "iam_plan_sha256": plan["plan_sha256"],
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "one_shot_nonce_sha256": plan["one_shot_nonce_sha256"],
        "prepare_receipt_sha256": prepare["receipt_sha256"],
        "install_receipt_sha256": install["receipt_sha256"],
        "readback_receipt_sha256": readback["receipt_sha256"],
        "authorization_sha256": auth["authorization_sha256"],
        "terminal_evidence_kind": evidence_kind,
        "terminal_evidence_sha256": evidence_sha,
        "project_worker_zero_readback_before": project_zero_before,
        "removed_binding_count": 0 if not execute else 2,
        "remaining_targeted_binding_count": 2 if not execute else 0,
        "set_attempt_count": 0 if not execute else 1,
        "cloud_mutation_performed": bool(execute),
        "current_profile_changed": False,
    }
    if expired_exact2:
        assert expired_auth is not None
        assert expired_observed_unix_seconds is not None
        common.update(
            {
                "expired_exact2_cleanup": True,
                "condition_expires_at_unix_seconds": expired_auth[
                    "condition_expires_at_unix_seconds"
                ],
                "required_grace_seconds": EXPIRED_EXACT2_GRACE_SECONDS,
                "cleanup_not_before_unix_seconds": expired_auth[
                    "cleanup_not_before_unix_seconds"
                ],
                "cleanup_observed_unix_seconds": expired_observed_unix_seconds,
                "condition_effective_during_cleanup": False,
                "zero_created_claimed": False,
            }
        )
    if not execute:
        return _seal({**common, "status": "dry_run_validated_no_mutation"})
    if expired_exact2:
        assert raw_expired_cleanup_nonce is not None
        # Network readbacks may consume most of the five-minute authorization
        # window.  Re-read the non-injectable live clock immediately before the
        # one IAM mutation so an expired authorization cannot be stretched.
        _validate_expired_exact2_cleanup_authorization(
            authorization,
            plan=plan,
            raw_cleanup_nonce=raw_expired_cleanup_nonce,
            now_unix_seconds=None,
        )
    transport.set_bucket_policy(policy=desired)
    post = _validated_policy(transport.get_bucket_policy())
    _assert_initial_zero(post, plan)
    project_zero_after = _project_worker_zero_readback(transport)
    if (
        _unrelated_fingerprint(post, plan)
        != prepare["unrelated_policy_fingerprint_sha256"]
    ):
        raise ValueError("unrelated bucket IAM policy changed during cleanup")
    return _seal(
        {
            **common,
            "status": "removed_exact_two_bindings_post_readback_zero",
            "post_bucket_policy_fingerprint_sha256": _policy_fingerprint(post),
            "post_bucket_policy_etag_sha256": _hash_text(post["etag"]),
            "project_worker_zero_readback_after": project_zero_after,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--execution-plan", required=True)
    plan.add_argument("--nonce-sha256", required=True)
    plan.add_argument("--issued-at-unix-seconds", required=True, type=int)
    plan.add_argument("--output", required=True)

    authorize = sub.add_parser("authorize")
    authorize.add_argument("--iam-plan", required=True)
    authorize.add_argument(
        "--operation",
        required=True,
        choices=("apply", "cleanup", "cleanup-expired-exact2"),
    )
    authorize.add_argument("--output", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--iam-plan", required=True)
    prepare.add_argument("--output", required=True)

    apply = sub.add_parser("apply")
    apply.add_argument("--iam-plan", required=True)
    apply.add_argument("--prepare", required=True)
    apply.add_argument("--authorization", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--execute", action="store_true")

    readback = sub.add_parser("readback")
    readback.add_argument("--iam-plan", required=True)
    readback.add_argument("--prepare", required=True)
    readback.add_argument("--install", required=True)
    readback.add_argument("--output", required=True)

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--iam-plan", required=True)
    cleanup.add_argument("--prepare", required=True)
    cleanup.add_argument("--install", required=True)
    cleanup.add_argument("--readback", required=True)
    cleanup.add_argument("--authorization", required=True)
    evidence = cleanup.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--collection")
    evidence.add_argument("--partial-launch")
    evidence.add_argument("--owned-launch-failure-closeout")
    evidence.add_argument("--expired-exact2", action="store_true")
    cleanup.add_argument("--output", required=True)
    cleanup.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        value = build_iam_plan(
            execution_plan=read_canonical_json(args.execution_plan, "execution plan"),
            one_shot_nonce_sha256=args.nonce_sha256,
            issued_at_unix_seconds=args.issued_at_unix_seconds,
        )
        write_json_once(args.output, value)
        return 0
    iam_plan = read_canonical_json(args.iam_plan, "worker IAM plan")
    if args.command == "authorize":
        if args.operation == "cleanup-expired-exact2":
            cleanup_nonce = os.environ.get(
                EXPIRED_CLEANUP_NONCE_ENVIRONMENT_VARIABLE
            )
            if cleanup_nonce is None:
                raise ValueError(
                    "expired exact2 cleanup nonce environment variable is absent"
                )
            value = build_expired_exact2_cleanup_authorization(
                iam_plan=iam_plan,
                raw_cleanup_nonce=cleanup_nonce,
                issued_at_unix_seconds=int(time.time()),
            )
        else:
            raw_nonce = os.environ.get(NONCE_ENVIRONMENT_VARIABLE)
            if raw_nonce is None:
                raise ValueError(
                    "worker IAM raw nonce environment variable is absent"
                )
            value = build_explicit_authorization(
                iam_plan=iam_plan,
                raw_nonce=raw_nonce,
                operation=args.operation,
            )
        write_json_once(args.output, value)
        return 0
    transport = GcpBucketIamTransport()
    if args.command == "prepare":
        value = prepare_worker_iam(iam_plan=iam_plan, transport=transport)
    elif args.command == "apply":
        raw_nonce = os.environ.get(NONCE_ENVIRONMENT_VARIABLE)
        if raw_nonce is None:
            raise ValueError("worker IAM raw nonce environment variable is absent")
        value = apply_worker_iam(
            iam_plan=iam_plan,
            prepare_receipt=read_canonical_json(args.prepare, "prepare receipt"),
            authorization=read_canonical_json(args.authorization, "apply authorization"),
            raw_nonce=raw_nonce,
            transport=transport,
            execute=args.execute,
        )
    elif args.command == "readback":
        value = readback_worker_iam(
            iam_plan=iam_plan,
            prepare_receipt=read_canonical_json(args.prepare, "prepare receipt"),
            install_receipt=read_canonical_json(args.install, "install receipt"),
            transport=transport,
        )
    else:
        expired_exact2 = bool(args.expired_exact2)
        if expired_exact2:
            raw_nonce = None
            raw_expired_cleanup_nonce = os.environ.get(
                EXPIRED_CLEANUP_NONCE_ENVIRONMENT_VARIABLE
            )
            if raw_expired_cleanup_nonce is None:
                raise ValueError(
                    "expired exact2 cleanup nonce environment variable is absent"
                )
        else:
            raw_nonce = os.environ.get(NONCE_ENVIRONMENT_VARIABLE)
            raw_expired_cleanup_nonce = None
            if raw_nonce is None:
                raise ValueError(
                    "worker IAM raw nonce environment variable is absent"
                )
        value = cleanup_worker_iam(
            iam_plan=iam_plan,
            prepare_receipt=read_canonical_json(args.prepare, "prepare receipt"),
            install_receipt=read_canonical_json(args.install, "install receipt"),
            readback_receipt=read_canonical_json(args.readback, "readback receipt"),
            authorization=read_canonical_json(args.authorization, "cleanup authorization"),
            raw_nonce=raw_nonce,
            transport=transport,
            collection_receipt=(
                read_canonical_json(args.collection, "collection receipt")
                if args.collection
                else None
            ),
            partial_launch_receipt=(
                read_canonical_json(args.partial_launch, "partial launch receipt")
                if args.partial_launch
                else None
            ),
            owned_launch_failure_closeout_receipt=(
                read_canonical_json(
                    args.owned_launch_failure_closeout,
                    "owned launch failure closeout receipt",
                )
                if args.owned_launch_failure_closeout
                else None
            ),
            expired_exact2=expired_exact2,
            raw_expired_cleanup_nonce=raw_expired_cleanup_nonce,
            execute=args.execute,
        )
    write_json_once(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
