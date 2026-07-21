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
    as subject,
)


def _transport_error() -> rest_iam.RestIamAdminError:
    return rest_iam.RestIamAdminError(
        "iam_https_transport_failed", operation="get_project_policy"
    )


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


def _wrapper(
    inner: _Inner,
) -> tuple[subject.ReadOnlyRetryIamAdminV2, list[float]]:
    sleeps: list[float] = []
    wrapper = subject.ReadOnlyRetryIamAdminV2(inner, sleep=sleeps.append)
    return wrapper, sleeps


def test_widened_budget_survives_a_sustained_burst() -> None:
    # Step12f died after 2 retries; the v2 budget must ride out 5 failures.
    inner = _Inner(get_failures=5)
    wrapper, sleeps = _wrapper(inner)
    assert wrapper.get_policy("project") == {"bindings": []}
    assert inner.get_calls == 6
    assert sleeps == [2.0, 8.0, 15.0, 30.0, 60.0]
    assert len(wrapper.retry_events) == 5


def test_budget_is_exactly_six_read_attempts() -> None:
    inner = _Inner(get_failures=6)
    wrapper, sleeps = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.get_policy("project")
    assert inner.get_calls == 6
    assert sleeps == [2.0, 8.0, 15.0, 30.0, 60.0]


def test_mutations_are_never_retried() -> None:
    inner = _Inner()
    wrapper, sleeps = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.add_binding("t", role="r", member="m", condition={})
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.remove_binding("t", role="r", member="m", condition={})
    assert inner.mutation_calls == 2
    assert sleeps == []
    assert wrapper.retry_events == []


def test_nesting_is_rejected_across_wrapper_generations() -> None:
    inner = _Inner()
    v2, _ = _wrapper(inner)
    with pytest.raises(TypeError, match="nested"):
        subject.ReadOnlyRetryIamAdminV2(v2)
    v1 = step12e_retry.ReadOnlyRetryIamAdmin(inner)
    with pytest.raises(TypeError, match="nested"):
        subject.ReadOnlyRetryIamAdminV2(v1)


def test_attributes_delegate_to_inner_admin() -> None:
    inner = _Inner()
    wrapper, _ = _wrapper(inner)
    assert (
        wrapper.controller_service_account
        == "inner-controller@example.iam"
    )


def test_preregistered_contract_constants() -> None:
    assert subject.RETRYABLE_CODE == "iam_https_transport_failed"
    assert subject.MAX_READ_ATTEMPTS == 6
    assert subject.BACKOFF_DELAYS_SECONDS == (2.0, 8.0, 15.0, 30.0, 60.0)
    assert sum(subject.BACKOFF_DELAYS_SECONDS) == 115.0
