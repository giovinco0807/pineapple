"""Bounded idempotent retry for read-only IAM policy GETs (Step12e).

Step12d attempt0 died before its first IAM mutation because one read-only
``get_policy`` call hit a transient ``iam_https_transport_failed``.  This
wrapper retries ONLY that case: idempotent policy reads failing at the
HTTPS transport layer.  Mutations (``add_binding``/``remove_binding``) are
never retried, HTTP-level errors (4xx/5xx classified by the adapter under
other codes) are never retried, and the attempt budget is fixed and small.
The frozen step11 adapter is not modified.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)


SCHEMA = "hu_m31_t3_step6d_step12e_readonly_retry_v1"
RETRYABLE_CODE = "iam_https_transport_failed"
MAX_READ_ATTEMPTS = 3
BACKOFF_DELAYS_SECONDS = (2.0, 8.0)


class ReadOnlyRetryIamAdmin:
    """Delegating IAM admin that retries only transport-failed policy reads."""

    def __init__(
        self,
        inner: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(inner, ReadOnlyRetryIamAdmin):
            raise TypeError("read-only retry wrapper must not be nested")
        if len(BACKOFF_DELAYS_SECONDS) != MAX_READ_ATTEMPTS - 1:
            raise AssertionError("retry budget and backoff table diverged")
        self._inner = inner
        self._sleep = sleep
        self.retry_events: list[dict[str, Any]] = []

    def get_policy(self, target: Any) -> Any:
        for attempt in range(1, MAX_READ_ATTEMPTS + 1):
            try:
                return self._inner.get_policy(target)
            except rest_iam.RestIamAdminError as error:
                if (
                    error.code != RETRYABLE_CODE
                    or attempt == MAX_READ_ATTEMPTS
                ):
                    raise
                delay = BACKOFF_DELAYS_SECONDS[attempt - 1]
                self.retry_events.append(
                    {
                        "schema": SCHEMA,
                        "operation": "get_policy",
                        "error_code": error.code,
                        "failed_attempt": attempt,
                        "max_attempts": MAX_READ_ATTEMPTS,
                        "sleep_seconds": delay,
                        "mutation": False,
                    }
                )
                self._sleep(delay)
        raise AssertionError("unreachable retry loop exit")

    def add_binding(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.add_binding(*args, **kwargs)

    def remove_binding(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.remove_binding(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


__all__ = [
    "BACKOFF_DELAYS_SECONDS",
    "MAX_READ_ATTEMPTS",
    "RETRYABLE_CODE",
    "ReadOnlyRetryIamAdmin",
    "SCHEMA",
]
