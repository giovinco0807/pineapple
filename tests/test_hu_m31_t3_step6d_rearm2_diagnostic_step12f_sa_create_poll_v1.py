from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as sa_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12f_sa_create_poll_v1
    as subject,
)


IDENTITY_RECORD = {
    "project": "ofc-solver-485418",
    "account_id": "ofc-m31-s2b-a1b2c3d4e5f6",
    "email": (
        "ofc-m31-s2b-a1b2c3d4e5f6@"
        "ofc-solver-485418.iam.gserviceaccount.com"
    ),
    "principal": (
        "serviceAccount:ofc-m31-s2b-a1b2c3d4e5f6@"
        "ofc-solver-485418.iam.gserviceaccount.com"
    ),
    "derived_from_run_tag": "a1b2c3d4e5f6",
    "run_scoped": True,
    "legacy_shared_controller_reused": False,
}


class _Token:
    def access_token(self) -> str:
        return "user-token"


def _response(status: int, body: Mapping[str, Any]) -> rest_iam.HttpResponse:
    return rest_iam.HttpResponse(
        status_code=status,
        body=json.dumps(body).encode("utf-8"),
        headers={},
    )


@dataclass
class _FakeHttp:
    """Create succeeds; the next N readback GETs 404 (propagation lag)."""

    identity: sa_v2.ControllerServiceAccountIdentity
    create_visibility_delay_gets: int = 0
    drift_unique_id_on_readback: bool = False

    def __post_init__(self) -> None:
        self.present = False
        self.get_count = 0
        self.post_count = 0

    def _provider(self, *, unique_id: str = "123456789012345678901") -> dict[str, Any]:
        return {
            "name": (
                f"projects/{self.identity.project}/serviceAccounts/"
                f"{self.identity.email}"
            ),
            "projectId": self.identity.project,
            "uniqueId": unique_id,
            "email": self.identity.email,
            "displayName": "ignored",
        }

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> rest_iam.HttpResponse:
        if method == "GET":
            self.get_count += 1
            if not self.present:
                return _response(404, {"error": {"code": 404}})
            if self.create_visibility_delay_gets > 0:
                self.create_visibility_delay_gets -= 1
                return _response(404, {"error": {"code": 404}})
            if self.drift_unique_id_on_readback:
                return _response(
                    200, self._provider(unique_id="999999999999999999999")
                )
            return _response(200, self._provider())
        if method == "POST":
            self.post_count += 1
            self.present = True
            return _response(200, self._provider())
        raise AssertionError(method)


def _admin(
    **fake_kwargs: Any,
) -> tuple[subject.CreateReadbackPollingControllerAdmin, _FakeHttp, list[float]]:
    identity = sa_v2.validate_controller_identity(IDENTITY_RECORD)
    http = _FakeHttp(identity, **fake_kwargs)
    sleeps: list[float] = []
    admin = subject.CreateReadbackPollingControllerAdmin(
        http_client=http,
        user_token_source=_Token(),
        identity=identity,
        readback_sleep=sleeps.append,
    )
    return admin, http, sleeps


def test_immediate_visibility_needs_no_poll() -> None:
    admin, http, sleeps = _admin()
    created = admin.create(absence=admin.require_absent())
    assert created.receipt["status"] == (
        "run_scoped_controller_service_account_created"
    )
    assert created.receipt["readback_verified"] is True
    assert created.receipt["create_call_count"] == 1
    assert http.post_count == 1
    assert sleeps == []
    assert admin.create_readback_poll_events == []


def test_delayed_visibility_is_polled_without_second_create() -> None:
    admin, http, sleeps = _admin(create_visibility_delay_gets=3)
    created = admin.create(absence=admin.require_absent())
    assert created.receipt["readback_verified"] is True
    assert http.post_count == 1
    assert sleeps == list(subject.CREATE_READBACK_DELAYS_SECONDS[:3])
    assert len(admin.create_readback_poll_events) == 3
    assert all(
        row["second_create_performed"] is False
        for row in admin.create_readback_poll_events
    )


def test_receipt_shape_matches_frozen_admin_exactly() -> None:
    admin, _, _ = _admin(create_visibility_delay_gets=2)
    created = admin.create(absence=admin.require_absent())
    frozen_fields = {
        "schema",
        "status",
        "project",
        "account_id",
        "email",
        "provider",
        "create_call_count",
        "create_retry_count",
        "readback_verified",
        "cloud_mutation_performed",
        "receipt_sha256",
    }
    assert set(created.receipt) == frozen_fields
    assert created.receipt["schema"] == sa_v2.SCHEMA


def test_budget_exhaustion_fails_closed_without_second_create() -> None:
    admin, http, sleeps = _admin(
        create_visibility_delay_gets=subject.MAX_CREATE_READBACK_ATTEMPTS + 5
    )
    with pytest.raises(
        sa_v2.RunScopedControllerServiceAccountError,
        match="readback_unavailable",
    ):
        admin.create(absence=admin.require_absent())
    assert http.post_count == 1
    assert sleeps == list(subject.CREATE_READBACK_DELAYS_SECONDS)


def test_drifted_readback_record_fails_immediately_without_poll() -> None:
    admin, http, sleeps = _admin(drift_unique_id_on_readback=True)
    with pytest.raises(
        sa_v2.RunScopedControllerServiceAccountError,
        match="readback_changed",
    ):
        admin.create(absence=admin.require_absent())
    assert http.post_count == 1
    assert sleeps == []


def test_absence_capability_rules_are_preserved() -> None:
    admin, _, _ = _admin()
    other, _, _ = _admin()
    foreign = other.require_absent()
    with pytest.raises(PermissionError, match="absence proof"):
        admin.create(absence=foreign)
    own = admin.require_absent()
    admin.create(absence=own)
    with pytest.raises(PermissionError, match="absence proof"):
        admin.create(absence=own)


def test_preregistered_poll_contract_constants() -> None:
    assert subject.MAX_CREATE_READBACK_ATTEMPTS == 8
    assert subject.CREATE_READBACK_DELAYS_SECONDS == (
        2.0,
        4.0,
        8.0,
        8.0,
        8.0,
        15.0,
        15.0,
    )
    assert sum(subject.CREATE_READBACK_DELAYS_SECONDS) == 60.0
