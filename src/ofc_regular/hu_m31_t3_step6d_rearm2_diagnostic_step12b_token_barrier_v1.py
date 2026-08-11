"""Secret-free two-phase IAM propagation barrier for Step 12b.

Phase 1 grants only ``roles/iam.serviceAccountTokenCreator`` on the run-scoped
controller service account.  A bounded generator retries HTTP 403 for at most
eight minutes.  On success the TokenCreator binding is removed and read back
before Phase 2 can install any launch permission.

This module is cloud-client neutral.  Callers inject the narrow IAM admin,
token generator, clocks, and sleeper.  It performs no cloud operation at
import time and cannot launch a VM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_controller_token_barrier_receipt_v1"
)
SUCCESS_STATUS = (
    "controller_token_minted_and_token_creator_revoked_before_phase2"
)
FAILURE_STATUS = "controller_token_barrier_failed_before_phase2"

TOKEN_CREATOR_PURPOSE = "initiator_controller_token_creator"
TOKEN_CREATOR_ROLE = "roles/iam.serviceAccountTokenCreator"
TOKEN_CREATOR_TARGET = rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT
DEFAULT_INITIATING_PRINCIPAL = "user:giovinco.080807@gmail.com"

MAX_PROPAGATION_SECONDS = 480
TOKEN_LIFETIME_SECONDS = 3_600
RETRY_DELAYS_SECONDS = (1, 2, 4, 8, 15, 30)
REVOKE_READBACK_DELAYS_SECONDS = (1, 2, 4, 8, 15, 30)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ERROR_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_INTERNAL_ERROR = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SERVICE_ACCOUNT = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]@"
    r"[a-z][a-z0-9-]{4,28}[a-z0-9]\.iam\.gserviceaccount\.com$"
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


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    return {**body, "receipt_sha256": canonical_sha256(body)}


class IamAdmin(Protocol):
    controller_service_account: str

    def add_binding(
        self,
        target: rest_iam.PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
    ) -> Any: ...

    def remove_binding(
        self,
        target: rest_iam.PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
    ) -> Any: ...

    def get_policy(
        self, target: rest_iam.PolicyTarget | str
    ) -> Mapping[str, Any]: ...


class ControllerTokenSource(Protocol):
    expire_time: str

    def access_token(self) -> str: ...


class ControllerTokenGenerator(Protocol):
    controller_service_account: str

    def generate_controller_access_token(
        self, *, lifetime_seconds: int
    ) -> ControllerTokenSource: ...


class TokenGenerationHttpError(RuntimeError):
    """A sanitized HTTP failure; raw response bytes are never retained."""

    def __init__(
        self,
        *,
        status_code: int,
        response_body: bytes,
        error_code: int | None,
        error_status: str | None,
        error_reason: str | None,
        internal_code: str = "controller_access_token_generation_failed",
    ) -> None:
        super().__init__(internal_code)
        if (
            type(status_code) is not int
            or not 400 <= status_code <= 599
            or not isinstance(response_body, bytes)
            or len(response_body) > 4 * 1024 * 1024
            or (
                error_code is not None
                and type(error_code) is not int
            )
            or (
                error_status is not None
                and _ERROR_TOKEN.fullmatch(error_status) is None
            )
            or (
                error_reason is not None
                and _ERROR_TOKEN.fullmatch(error_reason) is None
            )
            or _INTERNAL_ERROR.fullmatch(internal_code) is None
        ):
            raise ValueError("token HTTP error observation is invalid")
        self.status_code = status_code
        self.response_body_sha256 = hashlib.sha256(response_body).hexdigest()
        self.error_code = error_code
        self.error_status = error_status
        self.error_reason = error_reason
        self.internal_code = internal_code

    def __str__(self) -> str:
        return self.internal_code


class TokenBarrierFailure(RuntimeError):
    """Fail-closed barrier error carrying only a non-secret sealed receipt."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__(FAILURE_STATUS)
        self.receipt = copy.deepcopy(dict(receipt))

    def __str__(self) -> str:
        return FAILURE_STATUS


class TokenBarrierOutcome:
    """Success result whose credential has no serialization surface."""

    __slots__ = ("__token_source", "receipt")

    def __init__(
        self,
        token_source: ControllerTokenSource,
        receipt: Mapping[str, Any],
    ) -> None:
        self.__token_source = token_source
        self.receipt = copy.deepcopy(dict(receipt))

    def token_source(self) -> ControllerTokenSource:
        return self.__token_source

    def __repr__(self) -> str:
        return (
            "<TokenBarrierOutcome token=redacted "
            f"receipt_sha256={self.receipt.get('receipt_sha256')!r}>"
        )


class _FrozenControllerTokenSource:
    """One already-minted bearer token; this object cannot refresh it."""

    __slots__ = ("__access_token", "expire_time")

    def __init__(self, access_token: str, expire_time: str) -> None:
        self.__access_token = access_token
        self.expire_time = expire_time

    def access_token(self) -> str:
        return self.__access_token

    def __repr__(self) -> str:
        return "<_FrozenControllerTokenSource token=redacted>"


@dataclass(frozen=True)
class TokenCreatorBinding:
    target: rest_iam.PolicyTarget
    role: str
    member: str
    condition: Mapping[str, Any]
    controller_service_account: str
    purpose: str = TOKEN_CREATOR_PURPOSE


def build_token_creator_binding(
    *,
    controller_service_account: str,
    expires_at_rfc3339: str,
    initiating_principal: str = DEFAULT_INITIATING_PRINCIPAL,
) -> TokenCreatorBinding:
    if (
        not isinstance(expires_at_rfc3339, str)
        or not expires_at_rfc3339.endswith("Z")
        or len(expires_at_rfc3339) != 20
        or initiating_principal != DEFAULT_INITIATING_PRINCIPAL
        or not isinstance(controller_service_account, str)
        or _SERVICE_ACCOUNT.fullmatch(controller_service_account) is None
    ):
        raise ValueError("Step12b TokenCreator binding input changed")
    return TokenCreatorBinding(
        target=TOKEN_CREATOR_TARGET,
        role=TOKEN_CREATOR_ROLE,
        member=initiating_principal,
        controller_service_account=controller_service_account,
        condition={
            "title": "ofc-m31-step12b-token-barrier-v1",
            "description": (
                "Run-scoped Step12b token mint; revoked before Phase 2."
            ),
            "expression": (
                f'request.time < timestamp("{expires_at_rfc3339}")'
            ),
        },
    )


def _targeted_member_bindings(
    policy: Mapping[str, Any], *, member: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bindings = (
        policy["bindings"] if "bindings" in policy else []
    )
    if not isinstance(bindings, list):
        raise ValueError("controller service-account policy changed")
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise ValueError("controller service-account binding changed")
        members = raw.get("members")
        if not isinstance(members, list):
            raise ValueError("controller service-account members changed")
        if member in members:
            result.append(copy.deepcopy(dict(raw)))
    return result


def _require_binding_readback(
    admin: IamAdmin,
    binding: TokenCreatorBinding,
    *,
    present: bool,
) -> str:
    policy = admin.get_policy(binding.target)
    targeted = _targeted_member_bindings(policy, member=binding.member)
    expected = [
        {
            "role": binding.role,
            "members": [binding.member],
            "condition": dict(binding.condition),
        }
    ]
    if targeted != (expected if present else []):
        raise RuntimeError("Step12b TokenCreator readback changed")
    return canonical_sha256(targeted)


def _poll_revoke_zero_readback(
    admin: IamAdmin,
    binding: TokenCreatorBinding,
    *,
    now_monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    maximum_seconds: int,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Poll only the exact stale-present state until zero is observed."""

    started = float(now_monotonic())
    expected_present = [
        {
            "role": binding.role,
            "members": [binding.member],
            "condition": dict(binding.condition),
        }
    ]
    observation_sha256s: list[str] = []
    delays: list[float] = []
    poll_count = 0
    stale_present_count = 0
    failure_reason: str | None = None
    zero_sha256: str | None = None
    while True:
        poll_count += 1
        try:
            policy = admin.get_policy(binding.target)
            targeted = _targeted_member_bindings(
                policy, member=binding.member
            )
        except Exception:
            failure_reason = "token_creator_revoke_readback_failed"
            break
        observation_sha256s.append(canonical_sha256(targeted))
        if targeted == []:
            zero_sha256 = canonical_sha256(targeted)
            break
        if targeted != expected_present:
            failure_reason = "token_creator_revoke_readback_changed"
            break
        stale_present_count += 1
        elapsed = float(now_monotonic()) - started
        if elapsed >= maximum_seconds:
            failure_reason = (
                "token_creator_revoke_zero_readback_timeout"
            )
            break
        delay = float(
            REVOKE_READBACK_DELAYS_SECONDS[
                min(
                    stale_present_count - 1,
                    len(REVOKE_READBACK_DELAYS_SECONDS) - 1,
                )
            ]
        )
        delay = min(delay, maximum_seconds - elapsed)
        if delay <= 0:
            failure_reason = (
                "token_creator_revoke_zero_readback_timeout"
            )
            break
        delays.append(delay)
        sleep(delay)
    elapsed = float(now_monotonic()) - started
    evidence = {
        "poll_count": poll_count,
        "stale_present_count": stale_present_count,
        "observation_sha256s": observation_sha256s,
        "observation_sha256s_sha256": canonical_sha256(
            observation_sha256s
        ),
        "sleep_delays_seconds": delays,
        "elapsed_seconds": round(elapsed, 6),
        "maximum_seconds": maximum_seconds,
        "zero_observed": zero_sha256 is not None,
        "failure_reason": failure_reason,
        "second_add_performed": False,
        "controller_token_reminted": False,
    }
    return zero_sha256, evidence, failure_reason


def _mutation_changed(result: Any, label: str) -> int:
    changed = getattr(result, "changed", None)
    attempts = getattr(result, "attempts", None)
    if (
        changed is not True
        or type(attempts) is not int
        or not 1 <= attempts <= 8
    ):
        raise RuntimeError(f"Step12b {label} was not one fresh mutation")
    return attempts


def _error_record(
    error: TokenGenerationHttpError,
    *,
    attempt: int,
    started_at_unix_seconds: int,
    finished_at_unix_seconds: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "started_at_unix_seconds": started_at_unix_seconds,
        "finished_at_unix_seconds": finished_at_unix_seconds,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "http_status": error.status_code,
        "internal_error_code": error.internal_code,
        "google_error_code": error.error_code,
        "google_error_status": error.error_status,
        "google_error_reason": error.error_reason,
        "response_body_sha256": error.response_body_sha256,
        "response_body_stored": False,
        "authorization_header_stored": False,
        "access_token_stored": False,
    }


def _failure_receipt(
    *,
    reason: str,
    binding: TokenCreatorBinding,
    attempts: Sequence[Mapping[str, Any]],
    started_at_unix_seconds: int,
    finished_at_unix_seconds: int,
    elapsed_seconds: float,
    binding_add_attempts: int | None,
    add_readback_sha256: str | None,
    revoke_attempts: int | None,
    revoke_readback_sha256: str | None,
    token_attempt_count: int,
    cloud_mutation_attempted: bool,
    cloud_mutation_confirmed: bool,
    binding_add_outcome: str,
    token_creator_revoke_changed: bool | None,
    revoke_zero_readback_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal(
        {
            "schema": SCHEMA,
            "status": FAILURE_STATUS,
            "failure_reason": reason,
            "controller_service_account": (
                binding.controller_service_account
            ),
            "run_scoped_controller_required": True,
            "binding_purpose": binding.purpose,
            "binding_role": binding.role,
            "binding_member": binding.member,
            "binding_condition_sha256": canonical_sha256(
                dict(binding.condition)
            ),
            "binding_add_attempts": binding_add_attempts,
            "binding_add_outcome": binding_add_outcome,
            "binding_add_readback_sha256": add_readback_sha256,
            "token_attempt_count": token_attempt_count,
            "failed_http_token_attempt_count": len(attempts),
            "token_attempts": [copy.deepcopy(dict(row)) for row in attempts],
            "started_at_unix_seconds": started_at_unix_seconds,
            "finished_at_unix_seconds": finished_at_unix_seconds,
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "maximum_propagation_seconds": MAX_PROPAGATION_SECONDS,
            "token_creator_revoke_attempts": revoke_attempts,
            "token_creator_revoke_changed": (
                token_creator_revoke_changed
            ),
            "token_creator_revoke_readback_sha256": (
                revoke_readback_sha256
            ),
            "token_creator_revoke_zero_readback": copy.deepcopy(
                dict(revoke_zero_readback_evidence)
            ),
            "token_creator_revoke_zero_readback_evidence_sha256": (
                canonical_sha256(dict(revoke_zero_readback_evidence))
            ),
            "unknown_add_recovered_by_confirmed_revoke": (
                binding_add_outcome == "unknown_after_request"
                and token_creator_revoke_changed is True
            ),
            "phase2_started": False,
            "vm_insert_attempt_count": 0,
            "access_token_stored": False,
            "authorization_header_stored": False,
            "response_body_stored": False,
            "cloud_mutation_attempted": cloud_mutation_attempted,
            "cloud_mutation_performed": cloud_mutation_confirmed,
            "current_profile_changed": False,
        }
    )


def run_token_creator_barrier(
    *,
    admin: IamAdmin,
    token_generator: ControllerTokenGenerator,
    binding: TokenCreatorBinding,
    now_monotonic: Callable[[], float],
    now_unix_seconds: Callable[[], int],
    sleep: Callable[[float], None],
    maximum_propagation_seconds: int = MAX_PROPAGATION_SECONDS,
) -> TokenBarrierOutcome:
    """Mint one run-scoped token or fail before Phase 2/VM insertion."""

    if (
        binding.target is not TOKEN_CREATOR_TARGET
        or binding.role != TOKEN_CREATOR_ROLE
        or binding.member != DEFAULT_INITIATING_PRINCIPAL
        or admin.controller_service_account
        != binding.controller_service_account
        or token_generator.controller_service_account
        != binding.controller_service_account
        or maximum_propagation_seconds != MAX_PROPAGATION_SECONDS
    ):
        raise ValueError("Step12b token barrier contract changed")
    started_monotonic = float(now_monotonic())
    started_unix = int(now_unix_seconds())
    failures: list[dict[str, Any]] = []
    token_source: ControllerTokenSource | None = None
    attempt = 0
    failure_reason: str | None = None
    add_call_started = False
    add_result_returned = False
    add_mutation_confirmed = False
    add_attempts: int | None = None
    add_readback: str | None = None
    revoke_attempts: int | None = None
    revoke_readback: str | None = None
    revoke_verified = False
    revoke_changed: bool | None = None
    revoke_zero_readback_evidence: dict[str, Any] = {
        "poll_count": 0,
        "stale_present_count": 0,
        "observation_sha256s": [],
        "observation_sha256s_sha256": canonical_sha256([]),
        "sleep_delays_seconds": [],
        "elapsed_seconds": 0.0,
        "maximum_seconds": maximum_propagation_seconds,
        "zero_observed": False,
        "failure_reason": "token_creator_revoke_not_started",
        "second_add_performed": False,
        "controller_token_reminted": False,
    }
    try:
        add_call_started = True
        result = admin.add_binding(
            binding.target,
            role=binding.role,
            member=binding.member,
            condition=binding.condition,
        )
        add_result_returned = True
        add_attempts = _mutation_changed(result, "TokenCreator add")
        add_mutation_confirmed = True
        add_readback = _require_binding_readback(
            admin, binding, present=True
        )
        while True:
            elapsed_before_attempt = (
                float(now_monotonic()) - started_monotonic
            )
            if (
                attempt > 0
                and elapsed_before_attempt
                >= maximum_propagation_seconds
            ):
                failure_reason = "token_creator_propagation_timeout"
                break
            if attempt >= 64:
                failure_reason = "token_creator_attempt_bound_exhausted"
                break
            attempt += 1
            attempt_started_monotonic = float(now_monotonic())
            attempt_started_unix = int(now_unix_seconds())
            try:
                candidate = token_generator.generate_controller_access_token(
                    lifetime_seconds=TOKEN_LIFETIME_SECONDS
                )
                token = candidate.access_token()
                if (
                    not isinstance(token, str)
                    or not token
                    or len(token) > 16_384
                    or any(character.isspace() for character in token)
                    or not isinstance(candidate.expire_time, str)
                ):
                    raise RuntimeError(
                        "Step12b generated controller token shape changed"
                    )
                token_source = _FrozenControllerTokenSource(
                    token, candidate.expire_time
                )
                break
            except TokenGenerationHttpError as error:
                finished_monotonic = float(now_monotonic())
                failures.append(
                    _error_record(
                        error,
                        attempt=attempt,
                        started_at_unix_seconds=attempt_started_unix,
                        finished_at_unix_seconds=int(now_unix_seconds()),
                        elapsed_seconds=(
                            finished_monotonic - attempt_started_monotonic
                        ),
                    )
                )
                if error.status_code != 403:
                    failure_reason = "non_403_token_generation_failure"
                    break
                elapsed = finished_monotonic - started_monotonic
                if elapsed >= maximum_propagation_seconds:
                    failure_reason = "token_creator_propagation_timeout"
                    break
                delay = RETRY_DELAYS_SECONDS[
                    min(len(RETRY_DELAYS_SECONDS) - 1, attempt - 1)
                ]
                delay = min(
                    float(delay),
                    maximum_propagation_seconds - elapsed,
                )
                if delay <= 0:
                    failure_reason = "token_creator_propagation_timeout"
                    break
                sleep(delay)
            except Exception:
                failure_reason = "token_generator_unclassified_failure"
                break
    except Exception:
        if failure_reason is None:
            failure_reason = "token_creator_add_or_readback_failure"
    finally:
        if add_call_started:
            try:
                remove_result = admin.remove_binding(
                    binding.target,
                    role=binding.role,
                    member=binding.member,
                    condition=binding.condition,
                )
                changed = getattr(remove_result, "changed", None)
                attempts = getattr(remove_result, "attempts", None)
                if (
                    type(changed) is not bool
                    or type(attempts) is not int
                    or not 1 <= attempts <= 8
                    or (
                        add_mutation_confirmed
                        and changed is not True
                    )
                ):
                    raise RuntimeError(
                        "Step12b TokenCreator revoke result changed"
                    )
                revoke_attempts = attempts
                revoke_changed = changed
                (
                    revoke_readback,
                    revoke_zero_readback_evidence,
                    revoke_readback_failure,
                ) = _poll_revoke_zero_readback(
                    admin,
                    binding,
                    now_monotonic=now_monotonic,
                    sleep=sleep,
                    maximum_seconds=maximum_propagation_seconds,
                )
                if revoke_readback_failure is None:
                    revoke_verified = True
                elif revoke_readback_failure == (
                    "token_creator_revoke_zero_readback_timeout"
                ):
                    failure_reason = revoke_readback_failure
                else:
                    failure_reason = (
                        "token_creator_revoke_verification_failure"
                    )
            except Exception:
                revoke_zero_readback_evidence = {
                    **revoke_zero_readback_evidence,
                    "failure_reason": (
                        "token_creator_revoke_mutation_or_poll_failed"
                    ),
                }
                failure_reason = (
                    "token_creator_revoke_verification_failure"
                )

    finished_monotonic = float(now_monotonic())
    finished_unix = int(now_unix_seconds())
    elapsed = finished_monotonic - started_monotonic
    if (
        token_source is None
        or failure_reason is not None
        or not revoke_verified
    ):
        raise TokenBarrierFailure(
            _failure_receipt(
                reason=failure_reason or "token_generation_failed",
                binding=binding,
                attempts=failures,
                started_at_unix_seconds=started_unix,
                finished_at_unix_seconds=finished_unix,
                elapsed_seconds=elapsed,
                binding_add_attempts=add_attempts,
                add_readback_sha256=add_readback,
                revoke_attempts=revoke_attempts,
                revoke_readback_sha256=revoke_readback,
                token_attempt_count=attempt,
                cloud_mutation_attempted=add_call_started,
                cloud_mutation_confirmed=(
                    add_mutation_confirmed or revoke_changed is True
                ),
                binding_add_outcome=(
                    "confirmed_changed"
                    if add_mutation_confirmed
                    else (
                        "returned_without_confirmed_change"
                        if add_result_returned
                        else "unknown_after_request"
                    )
                ),
                token_creator_revoke_changed=revoke_changed,
                revoke_zero_readback_evidence=(
                    revoke_zero_readback_evidence
                ),
            )
        )
    body = {
        "schema": SCHEMA,
        "status": SUCCESS_STATUS,
        "controller_service_account": (
            binding.controller_service_account
        ),
        "run_scoped_controller_required": True,
        "binding_purpose": binding.purpose,
        "binding_role": binding.role,
        "binding_member": binding.member,
        "binding_condition_sha256": canonical_sha256(
            dict(binding.condition)
        ),
        "binding_add_attempts": add_attempts,
        "binding_add_outcome": "confirmed_changed",
        "binding_add_readback_sha256": add_readback,
        "token_attempt_count": attempt,
        "failed_token_attempt_count": len(failures),
        "token_attempts": failures,
        "started_at_unix_seconds": started_unix,
        "finished_at_unix_seconds": finished_unix,
        "elapsed_seconds": round(float(elapsed), 6),
        "maximum_propagation_seconds": MAX_PROPAGATION_SECONDS,
        "token_lifetime_seconds": TOKEN_LIFETIME_SECONDS,
        "token_expire_time": token_source.expire_time,
        "fixed_nonrefreshing_controller_credential": True,
        "controller_token_remint_forbidden": True,
        "token_creator_revoke_attempts": revoke_attempts,
        "token_creator_revoke_changed": True,
        "token_creator_revoke_readback_sha256": revoke_readback,
        "token_creator_revoke_zero_readback": (
            revoke_zero_readback_evidence
        ),
        "token_creator_revoke_zero_readback_evidence_sha256": (
            canonical_sha256(revoke_zero_readback_evidence)
        ),
        "token_creator_live_after_barrier": False,
        "phase2_started": False,
        "phase2_must_exclude_token_creator": True,
        "run_scoped_controller_delete_after_pair_claim_required": True,
        "vm_insert_attempt_count": 0,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "response_body_stored": False,
        "cloud_mutation_performed": True,
        "current_profile_changed": False,
    }
    return TokenBarrierOutcome(token_source, _seal(body))


__all__ = [
    "MAX_PROPAGATION_SECONDS",
    "REVOKE_READBACK_DELAYS_SECONDS",
    "SCHEMA",
    "SUCCESS_STATUS",
    "TokenBarrierFailure",
    "TokenBarrierOutcome",
    "TokenCreatorBinding",
    "TokenGenerationHttpError",
    "build_token_creator_binding",
    "canonical_sha256",
    "run_token_creator_barrier",
]
