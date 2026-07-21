from __future__ import annotations

from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12e_readonly_retry_v1
    as step12e_retry,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12g_readonly_retry_v2
    as step12g_retry,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12h_readonly_retry_v3
    as subject,
)


def _transport_error() -> rest_iam.RestIamAdminError:
    return rest_iam.RestIamAdminError(
        "iam_https_transport_failed", operation="get_project_policy"
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Inner:
    controller_service_account = "inner-controller@example.iam"

    def __init__(self, *, get_failures: int = 0) -> None:
        self.get_failures = get_failures
        self.get_calls = 0
        self.mutation_calls = 0

    def get_policy(self, _target: Any) -> dict[str, Any]:
        self.get_calls += 1
        if self.get_failures > 0:
            self.get_failures -= 1
            raise _transport_error()
        return {"bindings": []}

    def add_binding(self, *_args: Any, **_kwargs: Any) -> Any:
        self.mutation_calls += 1
        raise _transport_error()

    def remove_binding(self, *_args: Any, **_kwargs: Any) -> Any:
        self.mutation_calls += 1
        raise _transport_error()


def _wrapper(inner: _Inner) -> tuple[subject.ReadOnlyRetryIamAdminV3, _Clock]:
    clock = _Clock()
    wrapper = subject.ReadOnlyRetryIamAdminV3(
        inner, sleep=clock.sleep, monotonic=clock.monotonic
    )
    return wrapper, clock


def test_survives_more_failures_than_the_v2_fixed_budget() -> None:
    # Step12g v2 died after 6 attempts (5 retries). The deadline strategy
    # must ride out many more transport failures while under 480s.
    inner = _Inner(get_failures=10)
    wrapper, clock = _wrapper(inner)
    assert wrapper.get_policy("project") == {"bindings": []}
    assert inner.get_calls == 11
    assert len(wrapper.retry_events) == 10
    assert clock.now < subject.MAX_TOTAL_RETRY_SECONDS
    assert all(row["mutation"] is False for row in wrapper.retry_events)
    # backoff is capped at 60s after the ramp
    assert wrapper.retry_events[-1]["sleep_seconds"] == 60.0


def test_stops_at_the_480s_deadline() -> None:
    inner = _Inner(get_failures=10_000)
    wrapper, clock = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.get_policy("project")
    # never sleeps past the deadline
    assert clock.now < subject.MAX_TOTAL_RETRY_SECONDS
    total_planned = sum(
        row["sleep_seconds"] for row in wrapper.retry_events
    )
    assert total_planned < subject.MAX_TOTAL_RETRY_SECONDS


def test_deadline_matches_token_barrier_propagation_bound() -> None:
    from ofc_regular import (
        hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
        as token_barrier,
    )

    assert subject.MAX_TOTAL_RETRY_SECONDS == 480
    assert (
        subject.MAX_TOTAL_RETRY_SECONDS
        == token_barrier.MAX_PROPAGATION_SECONDS
    )


def test_mutations_are_never_retried() -> None:
    inner = _Inner()
    wrapper, clock = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.add_binding("t", role="r", member="m", condition={})
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.remove_binding("t", role="r", member="m", condition={})
    assert inner.mutation_calls == 2
    assert clock.now == 0.0
    assert wrapper.retry_events == []


def test_non_transport_error_is_never_retried() -> None:
    class _OtherInner(_Inner):
        def get_policy(self, _target: Any) -> dict[str, Any]:
            self.get_calls += 1
            raise rest_iam.RestIamAdminError(
                "iam_https_response_invalid",
                operation="get_project_policy",
            )

    inner = _OtherInner()
    wrapper, clock = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="response_invalid"):
        wrapper.get_policy("project")
    assert inner.get_calls == 1
    assert clock.now == 0.0


def test_nesting_is_rejected_across_all_wrapper_generations() -> None:
    inner = _Inner()
    v3, _ = _wrapper(inner)
    with pytest.raises(TypeError, match="nested"):
        subject.ReadOnlyRetryIamAdminV3(v3)
    v2 = step12g_retry.ReadOnlyRetryIamAdminV2(inner)
    with pytest.raises(TypeError, match="nested"):
        subject.ReadOnlyRetryIamAdminV3(v2)
    v1 = step12e_retry.ReadOnlyRetryIamAdmin(inner)
    with pytest.raises(TypeError, match="nested"):
        subject.ReadOnlyRetryIamAdminV3(v1)


def test_attributes_delegate_to_inner_admin() -> None:
    inner = _Inner()
    wrapper, _ = _wrapper(inner)
    assert (
        wrapper.controller_service_account
        == "inner-controller@example.iam"
    )
