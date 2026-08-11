from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as subject,
)


CONTROLLER = (
    "ofc-m31-s2b-controller-001@"
    "ofc-solver-485418.iam.gserviceaccount.com"
)
SECRET = "ya29.step12b-secret-controller-token"


@dataclass
class _Mutation:
    changed: bool
    attempts: int = 1


class _Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def unix(self) -> int:
        return 1_800_000_000 + int(self.elapsed)

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class _Admin:
    controller_service_account = CONTROLLER

    def __init__(
        self,
        *,
        add_readback_missing: bool = False,
        add_raises_after_mutation: bool = False,
        revoke_raises: bool = False,
        revoke_readback_present: bool = False,
        revoke_stale_reads: int = 0,
        revoke_stale_forever: bool = False,
        omit_empty_bindings: bool = False,
        null_empty_bindings: bool = False,
    ) -> None:
        self.add_readback_missing = add_readback_missing
        self.add_raises_after_mutation = add_raises_after_mutation
        self.revoke_raises = revoke_raises
        self.revoke_readback_present = revoke_readback_present
        self.revoke_stale_reads = revoke_stale_reads
        self.revoke_stale_forever = revoke_stale_forever
        self.omit_empty_bindings = omit_empty_bindings
        self.null_empty_bindings = null_empty_bindings
        self.binding: dict[str, Any] | None = None
        self.removed_binding: dict[str, Any] | None = None
        self.events: list[str] = []
        self.get_count = 0
        self.revoke_get_count = 0

    def add_binding(
        self,
        _target: Any,
        *,
        role: str,
        member: str,
        condition: dict[str, Any],
    ) -> _Mutation:
        self.events.append("add")
        self.binding = {
            "role": role,
            "members": [member],
            "condition": dict(condition),
        }
        if self.add_raises_after_mutation:
            raise RuntimeError("simulated unknown add outcome")
        return _Mutation(True)

    def remove_binding(
        self,
        _target: Any,
        *,
        role: str,
        member: str,
        condition: dict[str, Any],
    ) -> _Mutation:
        self.events.append("remove")
        if self.revoke_raises:
            raise RuntimeError("redacted fake revoke failure")
        assert self.binding == {
            "role": role,
            "members": [member],
            "condition": dict(condition),
        }
        self.removed_binding = dict(self.binding)
        self.binding = None
        return _Mutation(True)

    def get_policy(self, _target: Any) -> dict[str, Any]:
        self.get_count += 1
        self.events.append("get")
        if self.get_count == 1 and self.add_readback_missing:
            return {"bindings": []}
        if self.get_count > 1 and self.revoke_readback_present:
            assert self.binding is None
            return {
                "bindings": [
                    {
                        "role": subject.TOKEN_CREATOR_ROLE,
                        "members": [
                            subject.DEFAULT_INITIATING_PRINCIPAL
                        ],
                        "condition": {
                            "title": "unexpected-live-binding"
                        },
                    }
                ]
            }
        if self.get_count > 1:
            self.revoke_get_count += 1
            if (
                self.revoke_stale_forever
                or self.revoke_get_count <= self.revoke_stale_reads
            ):
                assert self.binding is None
                assert self.removed_binding is not None
                return {"bindings": [dict(self.removed_binding)]}
        if self.binding is None and self.omit_empty_bindings:
            return {"version": 3, "etag": "fixture"}
        if self.binding is None and self.null_empty_bindings:
            return {"bindings": None}
        return {
            "bindings": []
            if self.binding is None
            else [dict(self.binding)]
        }


class _Token:
    expire_time = "2030-01-01T01:00:00Z"

    def __init__(self) -> None:
        self.read_count = 0

    def access_token(self) -> str:
        self.read_count += 1
        return SECRET


class _Generator:
    controller_service_account = CONTROLLER

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.calls = 0
        self.last_token: _Token | None = None

    def generate_controller_access_token(
        self, *, lifetime_seconds: int
    ) -> _Token:
        assert lifetime_seconds == 3_600
        self.calls += 1
        result = self.results.pop(0) if self.results else _Token()
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, _Token)
        self.last_token = result
        return result


class _Always403(_Generator):
    def generate_controller_access_token(
        self, *, lifetime_seconds: int
    ) -> _Token:
        assert lifetime_seconds == 3_600
        self.calls += 1
        raise _http_error(403, body=f"denied-{self.calls}".encode())


def _http_error(
    status: int,
    *,
    body: bytes = b'{"error":"redacted"}',
) -> subject.TokenGenerationHttpError:
    return subject.TokenGenerationHttpError(
        status_code=status,
        response_body=body,
        error_code=403 if status == 403 else status,
        error_status="PERMISSION_DENIED"
        if status == 403
        else "INTERNAL",
        error_reason="IAM_PERMISSION_DENIED"
        if status == 403
        else "BACKEND_ERROR",
    )


def _binding() -> subject.TokenCreatorBinding:
    return subject.build_token_creator_binding(
        controller_service_account=CONTROLLER,
        expires_at_rfc3339="2030-01-01T00:10:00Z",
    )


def _run(
    admin: _Admin,
    generator: _Generator,
    clock: _Clock,
) -> subject.TokenBarrierOutcome:
    return subject.run_token_creator_barrier(
        admin=admin,
        token_generator=generator,
        binding=_binding(),
        now_monotonic=clock.monotonic,
        now_unix_seconds=clock.unix,
        sleep=clock.sleep,
    )


def test_403_only_retry_then_success_revokes_before_phase2() -> None:
    admin = _Admin()
    generator = _Generator(
        [_http_error(403, body=b"first"), _http_error(403, body=b"second")]
    )
    clock = _Clock()
    outcome = _run(admin, generator, clock)
    assert admin.events == ["add", "get", "remove", "get"]
    assert generator.calls == 3
    assert outcome.receipt["token_attempt_count"] == 3
    assert outcome.receipt["failed_token_attempt_count"] == 2
    assert outcome.receipt["token_creator_live_after_barrier"] is False
    assert outcome.receipt["phase2_started"] is False
    assert outcome.receipt["vm_insert_attempt_count"] == 0
    assert outcome.receipt[
        "fixed_nonrefreshing_controller_credential"
    ] is True
    assert generator.last_token is not None
    assert generator.last_token.read_count == 1
    assert outcome.token_source().access_token() == SECRET
    assert outcome.token_source().access_token() == SECRET
    assert generator.last_token.read_count == 1

    encoded = json.dumps(outcome.receipt, sort_keys=True)
    assert SECRET not in encoded
    assert SECRET not in repr(outcome)
    assert SECRET not in repr(outcome.token_source())
    assert not hasattr(outcome, "__dict__")
    assert all(
        row["response_body_stored"] is False
        and len(row["response_body_sha256"]) == 64
        for row in outcome.receipt["token_attempts"]
    )


def test_revoke_zero_readback_polls_exact_stale_then_absent_without_remint() -> None:
    admin = _Admin(revoke_stale_reads=2)
    generator = _Generator()
    clock = _Clock()
    outcome = _run(admin, generator, clock)
    evidence = outcome.receipt[
        "token_creator_revoke_zero_readback"
    ]
    assert admin.events.count("add") == 1
    assert admin.events.count("remove") == 1
    assert admin.events.count("get") == 4
    assert generator.calls == 1
    assert clock.elapsed == 3
    assert evidence["poll_count"] == 3
    assert evidence["stale_present_count"] == 2
    assert evidence["sleep_delays_seconds"] == [1.0, 2.0]
    assert evidence["zero_observed"] is True
    assert evidence["failure_reason"] is None
    assert evidence["second_add_performed"] is False
    assert evidence["controller_token_reminted"] is False
    assert outcome.receipt["phase2_started"] is False
    assert outcome.receipt["vm_insert_attempt_count"] == 0
    assert generator.last_token is not None
    assert generator.last_token.read_count == 1
    assert outcome.token_source().access_token() == SECRET
    assert generator.last_token.read_count == 1


def test_revoke_zero_accepts_provider_policy_with_missing_bindings_key() -> None:
    admin = _Admin(omit_empty_bindings=True)
    generator = _Generator()
    outcome = _run(admin, generator, _Clock())
    evidence = outcome.receipt[
        "token_creator_revoke_zero_readback"
    ]
    assert generator.calls == 1
    assert admin.events == ["add", "get", "remove", "get"]
    assert evidence["poll_count"] == 1
    assert evidence["stale_present_count"] == 0
    assert evidence["zero_observed"] is True
    assert evidence["controller_token_reminted"] is False
    assert outcome.receipt["vm_insert_attempt_count"] == 0


def test_revoke_zero_rejects_null_bindings_as_malformed() -> None:
    admin = _Admin(null_empty_bindings=True)
    generator = _Generator()
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, _Clock())
    receipt = captured.value.receipt
    evidence = receipt["token_creator_revoke_zero_readback"]
    assert receipt["failure_reason"] == (
        "token_creator_revoke_verification_failure"
    )
    assert evidence["zero_observed"] is False
    assert evidence["failure_reason"] == (
        "token_creator_revoke_readback_failed"
    )
    assert generator.calls == 1
    assert admin.events.count("add") == 1
    assert admin.events.count("remove") == 1
    assert receipt["phase2_started"] is False
    assert receipt["vm_insert_attempt_count"] == 0


def test_revoke_zero_readback_timeout_discards_fixed_token_without_readd() -> None:
    admin = _Admin(revoke_stale_forever=True)
    generator = _Generator()
    clock = _Clock()
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, clock)
    receipt = captured.value.receipt
    evidence = receipt["token_creator_revoke_zero_readback"]
    assert receipt["failure_reason"] == (
        "token_creator_revoke_zero_readback_timeout"
    )
    assert clock.elapsed == subject.MAX_PROPAGATION_SECONDS
    assert admin.events.count("add") == 1
    assert admin.events.count("remove") == 1
    assert generator.calls == 1
    assert receipt["token_attempt_count"] == 1
    assert evidence["stale_present_count"] == evidence["poll_count"]
    assert evidence["zero_observed"] is False
    assert evidence["failure_reason"] == (
        "token_creator_revoke_zero_readback_timeout"
    )
    assert evidence["second_add_performed"] is False
    assert evidence["controller_token_reminted"] is False
    assert receipt["phase2_started"] is False
    assert receipt["vm_insert_attempt_count"] == 0
    assert SECRET not in json.dumps(receipt)


def test_non403_fails_immediately_and_never_starts_phase2() -> None:
    admin = _Admin()
    generator = _Generator([_http_error(500)])
    clock = _Clock()
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, clock)
    receipt = captured.value.receipt
    assert generator.calls == 1
    assert clock.elapsed == 0
    assert receipt["failure_reason"] == (
        "non_403_token_generation_failure"
    )
    assert receipt["phase2_started"] is False
    assert receipt["vm_insert_attempt_count"] == 0
    assert admin.binding is None


def test_403_retry_is_bounded_to_exact_eight_minutes() -> None:
    admin = _Admin()
    generator = _Always403()
    clock = _Clock()
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, clock)
    receipt = captured.value.receipt
    assert receipt["failure_reason"] == (
        "token_creator_propagation_timeout"
    )
    assert clock.elapsed == subject.MAX_PROPAGATION_SECONDS
    assert receipt["elapsed_seconds"] == subject.MAX_PROPAGATION_SECONDS
    assert receipt["token_attempt_count"] == generator.calls
    assert generator.calls < 64
    assert all(
        row["http_status"] == 403
        for row in receipt["token_attempts"]
    )
    assert admin.binding is None


def test_add_readback_failure_still_revokes_exact_binding() -> None:
    admin = _Admin(add_readback_missing=True)
    generator = _Generator()
    clock = _Clock()
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, clock)
    assert generator.calls == 0
    assert admin.events == ["add", "get", "remove", "get"]
    assert admin.binding is None
    assert captured.value.receipt["failure_reason"] == (
        "token_creator_add_or_readback_failure"
    )


def test_unknown_add_outcome_reports_confirmed_cleanup_mutation() -> None:
    admin = _Admin(add_raises_after_mutation=True)
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, _Generator(), _Clock())
    receipt = captured.value.receipt
    assert admin.events == ["add", "remove", "get"]
    assert receipt["binding_add_outcome"] == "unknown_after_request"
    assert receipt["token_creator_revoke_changed"] is True
    assert receipt["unknown_add_recovered_by_confirmed_revoke"] is True
    assert receipt["cloud_mutation_attempted"] is True
    assert receipt["cloud_mutation_performed"] is True
    assert receipt["phase2_started"] is False
    assert receipt["vm_insert_attempt_count"] == 0


@pytest.mark.parametrize(
    "admin",
    [
        _Admin(revoke_raises=True),
        _Admin(revoke_readback_present=True),
    ],
)
def test_revoke_or_revoke_readback_failure_discards_token(
    admin: _Admin,
) -> None:
    generator = _Generator()
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, _Clock())
    assert captured.value.receipt["failure_reason"] == (
        "token_creator_revoke_verification_failure"
    )
    assert captured.value.receipt["phase2_started"] is False
    assert SECRET not in json.dumps(captured.value.receipt)


def test_unclassified_generator_failure_is_not_retried() -> None:
    admin = _Admin()
    generator = _Generator([RuntimeError("secret diagnostic")])
    with pytest.raises(subject.TokenBarrierFailure) as captured:
        _run(admin, generator, _Clock())
    assert generator.calls == 1
    assert captured.value.receipt["failure_reason"] == (
        "token_generator_unclassified_failure"
    )
    assert "secret diagnostic" not in json.dumps(captured.value.receipt)


def test_authority_mismatch_fails_before_any_cloud_call() -> None:
    admin = _Admin()
    admin.controller_service_account = (
        "other-controller-00001@"
        "ofc-solver-485418.iam.gserviceaccount.com"
    )
    generator = _Generator()
    with pytest.raises(ValueError, match="contract changed"):
        _run(admin, generator, _Clock())
    assert admin.events == []
    assert generator.calls == 0


def test_http_error_rejects_unsafe_structured_tokens() -> None:
    with pytest.raises(ValueError, match="invalid"):
        subject.TokenGenerationHttpError(
            status_code=403,
            response_body=b"redacted",
            error_code=403,
            error_status="permission denied",
            error_reason="IAM_PERMISSION_DENIED",
        )
