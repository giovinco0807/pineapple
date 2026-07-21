"""Bounded create-readback polling for the run-scoped controller SA (Step12f).

Step12e attempt0 died at ``controller_service_account_create``: the create
POST returned 200 but the immediate readback GET did not yet see the new
account (IAM eventual consistency).  The frozen Step12b admin already
polls for ABSENCE after delete but performed a single immediate GET after
create.  This subclass replicates the frozen ``create`` contract exactly -
one POST, absence capability consumed before the POST, never a second
create attempt, identical sealed receipt shape - and only adds a bounded
poll while the readback is still 404/None.  A readback that RETURNS a
record different from the create response still fails closed immediately.
The frozen module is not modified.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as sa_v2,
)


SCHEMA = "hu_m31_t3_step6d_step12f_sa_create_poll_v1"
MAX_CREATE_READBACK_ATTEMPTS = 8
CREATE_READBACK_DELAYS_SECONDS = (2.0, 4.0, 8.0, 8.0, 8.0, 15.0, 15.0)
EXHAUSTED_CODE = "controller_service_account_create_readback_unavailable"


class CreateReadbackPollingControllerAdmin(
    sa_v2.RunScopedControllerServiceAccountAdmin
):
    """Frozen admin plus a bounded 404-only poll on the create readback."""

    def __init__(
        self,
        *args: Any,
        readback_sleep: Callable[[float], None] = time.sleep,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            len(CREATE_READBACK_DELAYS_SECONDS)
            != MAX_CREATE_READBACK_ATTEMPTS - 1
        ):
            raise AssertionError(
                "create-readback budget and delay table diverged"
            )
        self._readback_sleep = readback_sleep
        self.create_readback_poll_events: list[dict[str, Any]] = []

    def create(self, *, absence: Any) -> Any:
        if (
            not isinstance(absence, sa_v2._FreshAbsenceCapability)
            or absence._owner is not self._capability_owner
            or absence._consumed
            or absence.identity != self.identity
            or absence.receipt.get("status")
            != "run_scoped_controller_service_account_absent"
            or absence.receipt.get("get_status") != 404
        ):
            raise PermissionError(
                "fresh controller service-account absence proof is missing"
            )
        # Consume before the POST, exactly as the frozen admin does.  An
        # unknown transport outcome must never become a second create.
        absence._consumed = True

        response = self._request(
            method="POST",
            url=self._collection_url,
            body={
                "accountId": self.identity.account_id,
                "serviceAccount": {
                    "displayName": "OFC M31 Step12b run-scoped controller",
                    "description": (
                        "Fresh attempt0 controller; deleted before pair release."
                    ),
                },
            },
        )
        if response.status_code != 200:
            raise sa_v2.RunScopedControllerServiceAccountError(
                "controller_service_account_create_failed",
                status_code=response.status_code,
            )
        created = self._validate_provider_record(
            self._json(response, "controller_service_account_create")
        )
        readback: dict[str, Any] | None = None
        for attempt in range(1, MAX_CREATE_READBACK_ATTEMPTS + 1):
            readback = self.get()
            if readback is not None:
                break
            if attempt == MAX_CREATE_READBACK_ATTEMPTS:
                raise sa_v2.RunScopedControllerServiceAccountError(
                    EXHAUSTED_CODE
                )
            delay = CREATE_READBACK_DELAYS_SECONDS[attempt - 1]
            self.create_readback_poll_events.append(
                {
                    "schema": SCHEMA,
                    "operation": "controller_service_account_create_readback",
                    "observed": "get_404_not_yet_visible",
                    "failed_attempt": attempt,
                    "max_attempts": MAX_CREATE_READBACK_ATTEMPTS,
                    "sleep_seconds": delay,
                    "second_create_performed": False,
                }
            )
            self._readback_sleep(delay)
        if readback != created:
            raise sa_v2.RunScopedControllerServiceAccountError(
                "controller_service_account_create_readback_changed"
            )
        receipt = sa_v2._seal(
            {
                "schema": sa_v2.SCHEMA,
                "status": "run_scoped_controller_service_account_created",
                "project": self.identity.project,
                "account_id": self.identity.account_id,
                "email": self.identity.email,
                "provider": created,
                "create_call_count": 1,
                "create_retry_count": 0,
                "readback_verified": True,
                "cloud_mutation_performed": True,
            }
        )
        return sa_v2._CreatedControllerCapability(
            owner=self._capability_owner,
            identity=self.identity,
            provider=created,
            receipt=receipt,
        )


__all__ = [
    "CREATE_READBACK_DELAYS_SECONDS",
    "CreateReadbackPollingControllerAdmin",
    "EXHAUSTED_CODE",
    "MAX_CREATE_READBACK_ATTEMPTS",
    "SCHEMA",
]
