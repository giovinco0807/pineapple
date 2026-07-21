"""Narrow REST IAM administration for the Step 11 diagnostic launch.

The helper deliberately supports only four policy resources fixed at
construction time: the project, result bucket, worker service account, and
controller service account.  Credentials and HTTPS are injected so unit tests
remain offline.  Access tokens are used only to build an Authorization header;
they are never placed in a receipt, error, log, or serializable result.

Policy reads always request IAM policy version 3.  Google may still return a
version 0/1 policy when it has no conditional bindings; mutations upgrade that
policy to version 3 and retain its ETag.  A 409/412 set conflict causes a
bounded get/mutate/set retry.  Every other unexpected response fails closed.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import quote, urlsplit


SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_MUTATION_ATTEMPTS = 4
# Bounded transport retry for the idempotent set_iam_policy read-modify-write.
# A transport failure (no HTTP response) means the request never completed, so
# re-reading the current policy and re-applying the exact member mutation is
# safe - unlike a create, which cannot be blindly retried. The budget is small
# because observed transport blips are brief and random (a canary completed 6
# of 8 mutations before one failed).
_TRANSPORT_RETRY_CODE = "iam_https_transport_failed"
MAX_SET_TRANSPORT_RETRIES = 5
_SET_TRANSPORT_BACKOFF_SECONDS = (2.0, 4.0, 8.0, 8.0, 8.0)
GOOGLE_CLOUD_PLATFORM_SCOPE = (
    "https://www.googleapis.com/auth/cloud-platform"
)

_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
_SERVICE_ACCOUNT = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]@"
    r"[a-z][a-z0-9-]{4,61}[a-z0-9]\.iam\.gserviceaccount\.com$"
)
_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_ALLOWED_HTTP_METHODS = frozenset({"GET", "POST", "PUT"})
_ALLOWED_HOSTS = frozenset(
    {
        "cloudresourcemanager.googleapis.com",
        "storage.googleapis.com",
        "iam.googleapis.com",
        "iamcredentials.googleapis.com",
    }
)
_RETRYABLE_SET_STATUSES = frozenset({409, 412})
_POLICY_VERSIONS = frozenset({0, 1, 3})
_CONDITION_FIELDS = frozenset(
    {"title", "description", "expression", "location"}
)


class AccessTokenSource(Protocol):
    """Caller-owned user credential source."""

    def access_token(self) -> str:
        """Return a bearer token without logging or serializing it."""


@dataclass(frozen=True)
class HttpResponse:
    """Minimal injected HTTP response."""

    status_code: int
    body: bytes
    headers: Mapping[str, str]


class JsonHttpsClient(Protocol):
    """Injected HTTPS client; implementations must not log headers or bodies."""

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        """Perform one request."""


class PolicyTarget(str, Enum):
    PROJECT = "project"
    BUCKET = "bucket"
    WORKER_SERVICE_ACCOUNT = "worker_service_account"
    CONTROLLER_SERVICE_ACCOUNT = "controller_service_account"


@dataclass(frozen=True)
class PolicyMutationResult:
    """Non-secret result of one exact policy mutation."""

    target: PolicyTarget
    changed: bool
    attempts: int
    policy: Mapping[str, Any]


class GeneratedControllerAccessToken:
    """Redacted access-token source returned by IAM Credentials.

    It intentionally has no dataclass/dict serialization surface.  Callers
    must explicitly invoke :meth:`access_token` when passing it to another
    authenticated client.
    """

    __slots__ = ("__token", "expire_time")

    def __init__(self, token: str, expire_time: str) -> None:
        self.__token = _validate_token(token)
        self.expire_time = _validate_expire_time(expire_time)

    def access_token(self) -> str:
        return self.__token

    def __repr__(self) -> str:
        return (
            "<GeneratedControllerAccessToken redacted "
            f"expire_time={self.expire_time!r}>"
        )

    __str__ = __repr__


class RestIamAdminError(RuntimeError):
    """Secret-free, fail-closed REST IAM error."""

    def __init__(
        self,
        code: str,
        *,
        operation: str,
        status_code: int | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.operation = operation
        self.status_code = status_code
        self.attempts = attempts

    def __str__(self) -> str:
        return self.code


class EnvironmentUserAccessTokenSource:
    """Read a user-provisioned token from one environment variable."""

    def __init__(
        self,
        environment_variable: str = "GOOGLE_OAUTH_ACCESS_TOKEN",
    ) -> None:
        if (
            not isinstance(environment_variable, str)
            or _ENVIRONMENT_VARIABLE.fullmatch(environment_variable) is None
        ):
            raise ValueError("token environment-variable name is invalid")
        self._environment_variable = environment_variable

    def access_token(self) -> str:
        token = os.environ.get(self._environment_variable)
        if token is None:
            raise ValueError("required user access-token variable is absent")
        return _validate_token(token)


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


class StdlibJsonHttpsClient:
    """Small HTTPS implementation with redirects and ambient proxies disabled."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        parsed = urlsplit(url)
        if (
            method not in _ALLOWED_HTTP_METHODS
            or parsed.scheme != "https"
            or parsed.hostname not in _ALLOWED_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("REST IAM request escaped the HTTPS allowlist")
        request = urllib.request.Request(
            url=url,
            headers=dict(headers),
            data=body,
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                return HttpResponse(
                    status_code=int(response.status),
                    body=raw,
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            return HttpResponse(
                status_code=int(exc.code),
                body=raw,
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except (urllib.error.URLError, TimeoutError, OSError):
            raise OSError("REST IAM HTTPS request failed") from None


@dataclass(frozen=True)
class _PolicyEndpoint:
    get_method: str
    get_url: str
    get_body: Mapping[str, Any] | None
    set_method: str
    set_url: str
    storage_policy_body: bool


class Step11RestIamAdmin:
    """ETag-safe IAM helper bound to the exact Step 11 resources."""

    def __init__(
        self,
        *,
        http_client: JsonHttpsClient,
        user_token_source: AccessTokenSource,
        project: str,
        bucket: str,
        worker_service_account: str,
        controller_service_account: str,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        max_mutation_attempts: int = MAX_MUTATION_ATTEMPTS,
        retry_base_seconds: float = 0.1,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http_client = http_client
        self._user_token_source = user_token_source
        self.project = _validate_project(project)
        self.bucket = _validate_bucket(bucket)
        self.worker_service_account = _validate_service_account(
            worker_service_account
        )
        self.controller_service_account = _validate_service_account(
            controller_service_account
        )
        if self.worker_service_account == self.controller_service_account:
            raise ValueError("worker and controller service accounts must differ")
        expected_service_account_suffix = (
            f"@{self.project}.iam.gserviceaccount.com"
        )
        if (
            not self.worker_service_account.endswith(
                expected_service_account_suffix
            )
            or not self.controller_service_account.endswith(
                expected_service_account_suffix
            )
        ):
            raise ValueError(
                "service accounts must belong to the configured project"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 120
        ):
            raise ValueError("REST IAM timeout is invalid")
        if (
            isinstance(max_mutation_attempts, bool)
            or not isinstance(max_mutation_attempts, int)
            or not 1 <= max_mutation_attempts <= 8
        ):
            raise ValueError("REST IAM retry bound is invalid")
        if (
            isinstance(retry_base_seconds, bool)
            or not isinstance(retry_base_seconds, (int, float))
            or not 0 <= float(retry_base_seconds) <= 2
        ):
            raise ValueError("REST IAM retry delay is invalid")
        if not callable(sleeper):
            raise ValueError("REST IAM sleeper is invalid")
        self._timeout_seconds = timeout_seconds
        self._max_mutation_attempts = max_mutation_attempts
        self._retry_base_seconds = float(retry_base_seconds)
        self._sleeper = sleeper

    def get_policy(self, target: PolicyTarget | str) -> Mapping[str, Any]:
        """Read one exact policy, requesting version 3 and requiring an ETag."""

        checked_target = _validate_target(target)
        endpoint = self._endpoint(checked_target)
        body = (
            None
            if endpoint.get_body is None
            else _canonical_json_bytes(endpoint.get_body)
        )
        response = self._request(
            operation=f"get_{checked_target.value}_policy",
            method=endpoint.get_method,
            url=endpoint.get_url,
            body=body,
        )
        if response.status_code != 200:
            raise RestIamAdminError(
                "iam_policy_get_failed",
                operation=f"get_{checked_target.value}_policy",
                status_code=response.status_code,
            )
        return _validate_policy(
            _decode_json_object(
                response.body,
                operation=f"get_{checked_target.value}_policy",
            )
        )

    def add_binding(
        self,
        target: PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
    ) -> PolicyMutationResult:
        """Add a member only to the exact role+condition binding."""

        return self._mutate_binding(
            target,
            role=role,
            member=member,
            condition=condition,
            add=True,
        )

    def remove_binding(
        self,
        target: PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
    ) -> PolicyMutationResult:
        """Remove a member only from the exact role+condition binding."""

        return self._mutate_binding(
            target,
            role=role,
            member=member,
            condition=condition,
            add=False,
        )

    def generate_controller_access_token(
        self,
        *,
        lifetime_seconds: int = 3600,
    ) -> GeneratedControllerAccessToken:
        """Generate a cloud-platform token for the exact controller SA."""

        if (
            isinstance(lifetime_seconds, bool)
            or not isinstance(lifetime_seconds, int)
            or not 1 <= lifetime_seconds <= 3600
        ):
            raise ValueError("controller token lifetime is invalid")
        email = quote(self.controller_service_account, safe="")
        url = (
            "https://iamcredentials.googleapis.com/v1/projects/-/"
            f"serviceAccounts/{email}:generateAccessToken"
        )
        response = self._request(
            operation="generate_controller_access_token",
            method="POST",
            url=url,
            body=_canonical_json_bytes(
                {
                    "scope": [GOOGLE_CLOUD_PLATFORM_SCOPE],
                    "lifetime": f"{lifetime_seconds}s",
                }
            ),
        )
        if response.status_code != 200:
            raise RestIamAdminError(
                "controller_access_token_generation_failed",
                operation="generate_controller_access_token",
                status_code=response.status_code,
            )
        payload = _decode_json_object(
            response.body,
            operation="generate_controller_access_token",
        )
        if set(payload) != {"accessToken", "expireTime"}:
            raise RestIamAdminError(
                "controller_access_token_response_changed",
                operation="generate_controller_access_token",
            )
        try:
            return GeneratedControllerAccessToken(
                payload["accessToken"],
                payload["expireTime"],
            )
        except (TypeError, ValueError):
            raise RestIamAdminError(
                "controller_access_token_response_changed",
                operation="generate_controller_access_token",
            ) from None

    def _mutate_binding(
        self,
        target: PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
        add: bool,
    ) -> PolicyMutationResult:
        checked_target = _validate_target(target)
        checked_role = _validate_role(role)
        checked_member = _validate_member(member)
        checked_condition = _validate_condition(condition)
        operation = (
            f"{'add' if add else 'remove'}_"
            f"{checked_target.value}_policy_binding"
        )
        for attempt in range(1, self._max_mutation_attempts + 1):
            current = self._with_set_transport_retry(
                lambda: self.get_policy(checked_target),
                operation=operation,
            )
            mutated, changed = _apply_exact_binding_mutation(
                current,
                role=checked_role,
                member=checked_member,
                condition=checked_condition,
                add=add,
            )
            if not changed:
                return PolicyMutationResult(
                    target=checked_target,
                    changed=False,
                    attempts=attempt,
                    policy=current,
                )
            response = self._with_set_transport_retry(
                lambda: self._set_policy(
                    checked_target,
                    mutated,
                    operation=operation,
                ),
                operation=operation,
            )
            if response.status_code in _RETRYABLE_SET_STATUSES:
                if attempt >= self._max_mutation_attempts:
                    raise RestIamAdminError(
                        "iam_policy_conflict_retry_exhausted",
                        operation=operation,
                        status_code=response.status_code,
                        attempts=attempt,
                    )
                delay = min(
                    1.0,
                    self._retry_base_seconds * (2 ** (attempt - 1)),
                )
                self._sleeper(delay)
                continue
            if response.status_code != 200:
                raise RestIamAdminError(
                    "iam_policy_set_failed",
                    operation=operation,
                    status_code=response.status_code,
                    attempts=attempt,
                )
            accepted = _validate_policy(
                _decode_json_object(response.body, operation=operation)
            )
            present = _has_exact_member(
                accepted,
                role=checked_role,
                member=checked_member,
                condition=checked_condition,
            )
            if present != add:
                raise RestIamAdminError(
                    "iam_policy_set_readback_mismatch",
                    operation=operation,
                    attempts=attempt,
                )
            return PolicyMutationResult(
                target=checked_target,
                changed=True,
                attempts=attempt,
                policy=accepted,
            )
        raise AssertionError("bounded IAM mutation loop escaped")

    def _with_set_transport_retry(
        self,
        call: Callable[[], Any],
        *,
        operation: str,
    ) -> Any:
        """Retry an idempotent read/set on a bare transport failure only.

        A transport failure carries no HTTP response, so the request never
        completed; re-reading and re-applying the exact read-modify-write is
        safe. HTTP-status errors (4xx/5xx) and every other error are never
        retried here. This is used only for the idempotent policy get/set
        inside a mutation loop - never for a create.
        """

        for failed in range(MAX_SET_TRANSPORT_RETRIES + 1):
            try:
                return call()
            except RestIamAdminError as error:
                if (
                    error.code != _TRANSPORT_RETRY_CODE
                    or failed == MAX_SET_TRANSPORT_RETRIES
                ):
                    raise
                self._sleeper(_SET_TRANSPORT_BACKOFF_SECONDS[failed])
        raise AssertionError("bounded set transport retry escaped")

    def _set_policy(
        self,
        target: PolicyTarget,
        policy: Mapping[str, Any],
        *,
        operation: str,
    ) -> HttpResponse:
        endpoint = self._endpoint(target)
        if endpoint.storage_policy_body:
            payload: Mapping[str, Any] = policy
        else:
            payload = {
                "policy": policy,
                # The common SetIamPolicy contract's supported/default mask is
                # bindings+etag.  Policy.version remains 3 in the policy body
                # so conditional bindings are interpreted safely.
                "updateMask": "bindings,etag",
            }
        return self._request(
            operation=operation,
            method=endpoint.set_method,
            url=endpoint.set_url,
            body=_canonical_json_bytes(payload),
        )

    def _request(
        self,
        *,
        operation: str,
        method: str,
        url: str,
        body: bytes | None,
    ) -> HttpResponse:
        try:
            token = _validate_token(self._user_token_source.access_token())
        except Exception:
            raise RestIamAdminError(
                "user_access_token_unavailable",
                operation=operation,
            ) from None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": f"regular-ofc/{SCHEMA}",
        }
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        try:
            response = self._http_client.request(
                method=method,
                url=url,
                headers=headers,
                body=body,
                timeout_seconds=self._timeout_seconds,
            )
        except Exception:
            raise RestIamAdminError(
                "iam_https_transport_failed",
                operation=operation,
            ) from None
        if (
            not isinstance(response, HttpResponse)
            or isinstance(response.status_code, bool)
            or not isinstance(response.status_code, int)
            or not 100 <= response.status_code <= 599
            or not isinstance(response.body, bytes)
            or len(response.body) > MAX_RESPONSE_BYTES
        ):
            raise RestIamAdminError(
                "iam_https_response_invalid",
                operation=operation,
            )
        return response

    def _endpoint(self, target: PolicyTarget) -> _PolicyEndpoint:
        if target is PolicyTarget.PROJECT:
            resource = quote(self.project, safe="")
            base = (
                "https://cloudresourcemanager.googleapis.com/v1/"
                f"projects/{resource}"
            )
            return _PolicyEndpoint(
                get_method="POST",
                get_url=f"{base}:getIamPolicy",
                get_body={"options": {"requestedPolicyVersion": 3}},
                set_method="POST",
                set_url=f"{base}:setIamPolicy",
                storage_policy_body=False,
            )
        if target is PolicyTarget.BUCKET:
            resource = quote(self.bucket, safe="")
            base = (
                "https://storage.googleapis.com/storage/v1/b/"
                f"{resource}/iam"
            )
            versioned = f"{base}?optionsRequestedPolicyVersion=3"
            return _PolicyEndpoint(
                get_method="GET",
                get_url=versioned,
                get_body=None,
                set_method="PUT",
                set_url=base,
                storage_policy_body=True,
            )
        if target is PolicyTarget.WORKER_SERVICE_ACCOUNT:
            service_account = self.worker_service_account
        elif target is PolicyTarget.CONTROLLER_SERVICE_ACCOUNT:
            service_account = self.controller_service_account
        else:
            raise AssertionError("unknown fixed IAM policy target")
        project = quote(self.project, safe="")
        resource = quote(service_account, safe="")
        base = (
            "https://iam.googleapis.com/v1/projects/"
            f"{project}/serviceAccounts/{resource}"
        )
        return _PolicyEndpoint(
            get_method="POST",
            get_url=(
                f"{base}:getIamPolicy?"
                "options.requestedPolicyVersion=3"
            ),
            get_body=None,
            set_method="POST",
            set_url=f"{base}:setIamPolicy",
            storage_policy_body=False,
        )


def _validate_project(value: Any) -> str:
    if not isinstance(value, str) or _PROJECT.fullmatch(value) is None:
        raise ValueError("project is invalid")
    return value


def _validate_bucket(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _BUCKET.fullmatch(value) is None
        or ".." in value
        or ".-" in value
        or "-." in value
    ):
        raise ValueError("bucket is invalid")
    return value


def _validate_service_account(value: Any) -> str:
    if not isinstance(value, str) or _SERVICE_ACCOUNT.fullmatch(value) is None:
        raise ValueError("service account is invalid")
    return value


def _validate_target(value: PolicyTarget | str) -> PolicyTarget:
    try:
        return PolicyTarget(value)
    except (TypeError, ValueError):
        raise ValueError("IAM policy target escaped the fixed set") from None


def _validate_token(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 16_384
        or any(character.isspace() for character in value)
    ):
        raise ValueError("access token shape is invalid")
    return value


def _validate_expire_time(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith("Z")
        or len(value) > 64
    ):
        raise ValueError("access-token expiration is invalid")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ValueError("access-token expiration is invalid") from None
    return value


def _validate_role(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(character.isspace() for character in value)
        or not (
            value.startswith("roles/")
            or re.fullmatch(
                r"projects/[a-z][a-z0-9-]{4,61}[a-z0-9]/roles/"
                r"[A-Za-z0-9_.]{1,64}",
                value,
            )
        )
    ):
        raise ValueError("IAM role is invalid")
    return value


def _validate_member(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(character) < 33 for character in value)
    ):
        raise ValueError("IAM member is invalid")
    return value


def _validate_condition(
    value: Mapping[str, Any] | None,
) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - _CONDITION_FIELDS:
        raise ValueError("IAM condition shape is invalid")
    if not {"title", "expression"}.issubset(value):
        raise ValueError("IAM condition lacks title or expression")
    checked: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 4096
            or any(ord(character) < 32 and character not in "\t\n\r" for character in item)
        ):
            raise ValueError("IAM condition field is invalid")
        checked[key] = item
    return checked


def _validate_policy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RestIamAdminError(
            "iam_policy_shape_invalid",
            operation="validate_policy",
        )
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError):
        raise RestIamAdminError(
            "iam_policy_shape_invalid",
            operation="validate_policy",
        ) from None
    etag = copied.get("etag")
    if (
        not isinstance(etag, str)
        or not etag
        or len(etag) > 1024
        or any(character.isspace() for character in etag)
    ):
        raise RestIamAdminError(
            "iam_policy_etag_missing",
            operation="validate_policy",
        )
    version = copied.get("version", 0)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in _POLICY_VERSIONS
    ):
        raise RestIamAdminError(
            "iam_policy_version_invalid",
            operation="validate_policy",
        )
    bindings = copied.get("bindings", [])
    if not isinstance(bindings, list):
        raise RestIamAdminError(
            "iam_policy_bindings_invalid",
            operation="validate_policy",
        )
    seen: set[tuple[str, str | None]] = set()
    has_conditional_binding = False
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) - {"role", "members", "condition"}
        ):
            raise RestIamAdminError(
                "iam_policy_binding_invalid",
                operation="validate_policy",
            )
        try:
            role = _validate_role(binding.get("role"))
            members = binding.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError
            checked_members = [_validate_member(member) for member in members]
            if len(set(checked_members)) != len(checked_members):
                raise ValueError
            condition = _validate_condition(binding.get("condition"))
            has_conditional_binding = (
                has_conditional_binding or condition is not None
            )
        except ValueError:
            raise RestIamAdminError(
                "iam_policy_binding_invalid",
                operation="validate_policy",
            ) from None
        identity = (role, _condition_identity(condition))
        if identity in seen:
            raise RestIamAdminError(
                "iam_policy_binding_ambiguous",
                operation="validate_policy",
            )
        seen.add(identity)
    if has_conditional_binding and version != 3:
        raise RestIamAdminError(
            "iam_conditional_policy_requires_version_3",
            operation="validate_policy",
        )
    return copied


def _apply_exact_binding_mutation(
    policy: Mapping[str, Any],
    *,
    role: str,
    member: str,
    condition: Mapping[str, str] | None,
    add: bool,
) -> tuple[Mapping[str, Any], bool]:
    copied = json.loads(json.dumps(policy, ensure_ascii=False, allow_nan=False))
    bindings = copied.setdefault("bindings", [])
    target_identity = (role, _condition_identity(condition))
    matching: dict[str, Any] | None = None
    for binding in bindings:
        identity = (
            binding["role"],
            _condition_identity(binding.get("condition")),
        )
        if identity == target_identity:
            matching = binding
            break
    if add:
        if matching is not None and member in matching["members"]:
            return copied, False
        if matching is None:
            matching = {"role": role, "members": [member]}
            if condition is not None:
                matching["condition"] = dict(condition)
            bindings.append(matching)
        else:
            matching["members"].append(member)
    else:
        if matching is None or member not in matching["members"]:
            return copied, False
        matching["members"].remove(member)
        if not matching["members"]:
            bindings.remove(matching)
    copied["version"] = 3
    return _validate_policy(copied), True


def _has_exact_member(
    policy: Mapping[str, Any],
    *,
    role: str,
    member: str,
    condition: Mapping[str, str] | None,
) -> bool:
    target_identity = (role, _condition_identity(condition))
    for binding in policy.get("bindings", []):
        identity = (
            binding["role"],
            _condition_identity(binding.get("condition")),
        )
        if identity == target_identity:
            return member in binding["members"]
    return False


def _condition_identity(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return _canonical_json_bytes(value).decode("utf-8")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_json_object(body: bytes, *, operation: str) -> Mapping[str, Any]:
    if len(body) > MAX_RESPONSE_BYTES:
        raise RestIamAdminError(
            "iam_json_response_too_large",
            operation=operation,
        )
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RestIamAdminError(
            "iam_json_response_invalid",
            operation=operation,
        ) from None
    if not isinstance(value, Mapping):
        raise RestIamAdminError(
            "iam_json_response_invalid",
            operation=operation,
        )
    return value
