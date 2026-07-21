from __future__ import annotations

from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12e_readonly_retry_v1
    as subject,
)


def _transport_error() -> rest_iam.RestIamAdminError:
    return rest_iam.RestIamAdminError(
        "iam_https_transport_failed", operation="get_project_policy"
    )


class _Inner:
    controller_service_account = "inner-controller@example.iam"

    def __init__(
        self,
        *,
        get_failures: list[BaseException] | None = None,
        mutation_failures: list[BaseException] | None = None,
    ) -> None:
        self.get_failures = list(get_failures or [])
        self.mutation_failures = list(mutation_failures or [])
        self.get_calls = 0
        self.add_calls = 0
        self.remove_calls = 0

    def get_policy(self, _target: Any) -> dict[str, Any]:
        self.get_calls += 1
        if self.get_failures:
            raise self.get_failures.pop(0)
        return {"bindings": []}

    def add_binding(self, *_args: Any, **_kwargs: Any) -> Any:
        self.add_calls += 1
        if self.mutation_failures:
            raise self.mutation_failures.pop(0)
        return "added"

    def remove_binding(self, *_args: Any, **_kwargs: Any) -> Any:
        self.remove_calls += 1
        if self.mutation_failures:
            raise self.mutation_failures.pop(0)
        return "removed"


def _wrapper(inner: _Inner) -> tuple[subject.ReadOnlyRetryIamAdmin, list[float]]:
    sleeps: list[float] = []
    wrapper = subject.ReadOnlyRetryIamAdmin(inner, sleep=sleeps.append)
    return wrapper, sleeps


def test_transport_failed_get_is_retried_with_exact_backoff() -> None:
    inner = _Inner(get_failures=[_transport_error(), _transport_error()])
    wrapper, sleeps = _wrapper(inner)
    assert wrapper.get_policy("project") == {"bindings": []}
    assert inner.get_calls == 3
    assert sleeps == [2.0, 8.0]
    assert [row["failed_attempt"] for row in wrapper.retry_events] == [1, 2]
    assert all(row["mutation"] is False for row in wrapper.retry_events)


def test_retry_budget_is_exactly_three_read_attempts() -> None:
    inner = _Inner(
        get_failures=[_transport_error() for _ in range(3)]
    )
    wrapper, sleeps = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.get_policy("project")
    assert inner.get_calls == 3
    assert sleeps == [2.0, 8.0]


def test_non_transport_iam_error_is_never_retried() -> None:
    inner = _Inner(
        get_failures=[
            rest_iam.RestIamAdminError(
                "iam_https_response_invalid",
                operation="get_project_policy",
            )
        ]
    )
    wrapper, sleeps = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="response_invalid"):
        wrapper.get_policy("project")
    assert inner.get_calls == 1
    assert sleeps == []


def test_non_iam_exception_is_never_retried() -> None:
    inner = _Inner(get_failures=[RuntimeError("unrelated")])
    wrapper, sleeps = _wrapper(inner)
    with pytest.raises(RuntimeError, match="unrelated"):
        wrapper.get_policy("project")
    assert inner.get_calls == 1
    assert sleeps == []


def test_mutations_are_never_retried_even_on_transport_failure() -> None:
    inner = _Inner(mutation_failures=[_transport_error(), _transport_error()])
    wrapper, sleeps = _wrapper(inner)
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.add_binding("t", role="r", member="m", condition={})
    with pytest.raises(rest_iam.RestIamAdminError, match="transport"):
        wrapper.remove_binding("t", role="r", member="m", condition={})
    assert inner.add_calls == 1
    assert inner.remove_calls == 1
    assert sleeps == []
    assert wrapper.retry_events == []


def test_attributes_delegate_to_inner_admin() -> None:
    inner = _Inner()
    wrapper, _ = _wrapper(inner)
    assert (
        wrapper.controller_service_account
        == "inner-controller@example.iam"
    )


def test_wrapper_nesting_is_rejected() -> None:
    inner = _Inner()
    wrapper, _ = _wrapper(inner)
    with pytest.raises(TypeError, match="nested"):
        subject.ReadOnlyRetryIamAdmin(wrapper)


def test_preregistered_contract_constants() -> None:
    assert subject.RETRYABLE_CODE == "iam_https_transport_failed"
    assert subject.MAX_READ_ATTEMPTS == 3
    assert subject.BACKOFF_DELAYS_SECONDS == (2.0, 8.0)
