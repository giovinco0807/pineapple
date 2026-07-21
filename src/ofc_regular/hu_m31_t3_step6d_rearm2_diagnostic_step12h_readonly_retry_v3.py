"""Deadline-based bounded retry for read-only IAM policy GETs (Step12h).

Step12g attempt0 died when two separate Phase2 initial-zero reads each
exhausted the Step12g v2 fixed budget (6 attempts, 115s of sleeps): the
local USB Wi-Fi transport outage outlasted even the widened budget.  A
fixed attempt count keeps losing to variable-length outages, so Step12h
switches to a deadline: idempotent policy reads retry until a total wall
bound elapses, capping individual backoff at 60s.

The deadline is 480s, the exact propagation bound the frozen token
barrier already tolerates (``MAX_PROPAGATION_SECONDS``); reusing it keeps
the retry contract principled and precedented rather than an arbitrary
larger number.  Mutations are never retried.  The frozen step11 adapter
and the terminal Step12e/Step12g wrapper modules are not modified.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12e_readonly_retry_v1
    as step12e_retry,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12g_readonly_retry_v2
    as step12g_retry,
)


SCHEMA = "hu_m31_t3_step6d_step12h_readonly_retry_v3"
RETRYABLE_CODE = "iam_https_transport_failed"
MAX_TOTAL_RETRY_SECONDS = token_barrier.MAX_PROPAGATION_SECONDS  # 480
BACKOFF_DELAYS_SECONDS = (2.0, 8.0, 15.0, 30.0, 60.0, 60.0, 60.0, 60.0)
_NESTABLE = (
    step12e_retry.ReadOnlyRetryIamAdmin,
    step12g_retry.ReadOnlyRetryIamAdminV2,
)


class ReadOnlyRetryIamAdminV3:
    """Delegating IAM admin: retry transport-failed reads until a deadline."""

    def __init__(
        self,
        inner: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(inner, (ReadOnlyRetryIamAdminV3, *_NESTABLE)):
            raise TypeError("read-only retry wrapper must not be nested")
        if MAX_TOTAL_RETRY_SECONDS != 480:
            raise AssertionError("token barrier propagation bound changed")
        self._inner = inner
        self._sleep = sleep
        self._monotonic = monotonic
        self.retry_events: list[dict[str, Any]] = []

    def _backoff(self, retry_index: int) -> float:
        table = BACKOFF_DELAYS_SECONDS
        return table[min(retry_index, len(table) - 1)]

    def get_policy(self, target: Any) -> Any:
        started = self._monotonic()
        failed = 0
        while True:
            try:
                return self._inner.get_policy(target)
            except rest_iam.RestIamAdminError as error:
                if error.code != RETRYABLE_CODE:
                    raise
                elapsed = self._monotonic() - started
                delay = self._backoff(failed)
                if elapsed + delay >= MAX_TOTAL_RETRY_SECONDS:
                    raise
                failed += 1
                self.retry_events.append(
                    {
                        "schema": SCHEMA,
                        "operation": "get_policy",
                        "error_code": error.code,
                        "failed_attempt": failed,
                        "elapsed_seconds_before_sleep": elapsed,
                        "deadline_seconds": float(MAX_TOTAL_RETRY_SECONDS),
                        "sleep_seconds": delay,
                        "mutation": False,
                    }
                )
                self._sleep(delay)

    def add_binding(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.add_binding(*args, **kwargs)

    def remove_binding(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.remove_binding(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


__all__ = [
    "BACKOFF_DELAYS_SECONDS",
    "MAX_TOTAL_RETRY_SECONDS",
    "RETRYABLE_CODE",
    "ReadOnlyRetryIamAdminV3",
    "SCHEMA",
]
