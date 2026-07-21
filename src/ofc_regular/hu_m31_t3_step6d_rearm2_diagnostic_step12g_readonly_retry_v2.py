"""Extended bounded retry for read-only IAM policy GETs (Step12g).

Step12f attempt0 survived one transport burst with the Step12e budget
(3 attempts, 10s of sleeps) but a later read-only GET exhausted it during
a sustained burst: six attempts consumed ~179s at ~26s per failure,
consistent with episodic USB Wi-Fi adapter stalls of a few minutes.  A
4-minute dual-stack probe immediately afterwards was 47/47 clean on both
IPv4 and IPv6, so no per-family transport change is justified.

Step12g therefore only widens the read budget: 6 attempts with
2/8/15/30/60s backoff (115s of sleeps, several minutes of wall clock
including the failed connect attempts themselves), bounded and
preregistered, mirroring the token barrier's 480s propagation precedent.
Mutations are never retried.  The frozen step11 adapter and the terminal
Step12e wrapper module are not modified.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12e_readonly_retry_v1
    as step12e_retry,
)


SCHEMA = "hu_m31_t3_step6d_step12g_readonly_retry_v2"
RETRYABLE_CODE = "iam_https_transport_failed"
MAX_READ_ATTEMPTS = 6
BACKOFF_DELAYS_SECONDS = (2.0, 8.0, 15.0, 30.0, 60.0)


class ReadOnlyRetryIamAdminV2:
    """Delegating IAM admin that retries only transport-failed policy reads."""

    def __init__(
        self,
        inner: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(
            inner,
            (ReadOnlyRetryIamAdminV2, step12e_retry.ReadOnlyRetryIamAdmin),
        ):
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
    "ReadOnlyRetryIamAdminV2",
    "SCHEMA",
]
