from __future__ import annotations

import copy
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


@dataclass
class _FakeHttp:
    identity: subject.ControllerServiceAccountIdentity
    present: bool = False
    delete_visibility_gets: int = 0
    create_status: int = 200
    create_unknown_outcome: bool = False

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.delete_called = False

    def _provider(self) -> dict[str, Any]:
        return {
            "name": (
                f"projects/{self.identity.project}/serviceAccounts/"
                f"{self.identity.email}"
            ),
            "projectId": self.identity.project,
            "uniqueId": "123456789012345678901",
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
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        assert headers["Authorization"] == "Bearer user-token"
        assert timeout_seconds == 60
        if method == "GET":
            if self.delete_called and self.delete_visibility_gets > 0:
                self.delete_visibility_gets -= 1
                return _response(200, self._provider())
            return (
                _response(200, self._provider())
                if self.present and not self.delete_called
                else _response(404, {"error": {"code": 404}})
            )
        if method == "POST":
            assert not self.present
            payload = json.loads((body or b"").decode("ascii"))
            assert payload["accountId"] == self.identity.account_id
            if self.create_unknown_outcome:
                self.present = True
                raise OSError("simulated transport loss after create")
            if self.create_status != 200:
                return _response(
                    self.create_status,
                    {"error": {"code": self.create_status}},
                )
            self.present = True
            return _response(200, self._provider())
        if method == "DELETE":
            assert self.present
            self.delete_called = True
            return _response(200, {})
        raise AssertionError(method)


def _response(
    status: int, body: Mapping[str, Any]
) -> rest_iam.HttpResponse:
    return rest_iam.HttpResponse(
        status_code=status,
        body=json.dumps(body).encode("utf-8"),
        headers={},
    )


def _admin(
    *,
    present: bool = False,
    delete_visibility_gets: int = 0,
    create_status: int = 200,
    create_unknown_outcome: bool = False,
) -> tuple[subject.RunScopedControllerServiceAccountAdmin, _FakeHttp]:
    identity = subject.validate_controller_identity(IDENTITY_RECORD)
    http = _FakeHttp(
        identity,
        present=present,
        delete_visibility_gets=delete_visibility_gets,
        create_status=create_status,
        create_unknown_outcome=create_unknown_outcome,
    )
    return (
        subject.RunScopedControllerServiceAccountAdmin(
            http_client=http,
            user_token_source=_Token(),
            identity=identity,
        ),
        http,
    )


def test_identity_rejects_legacy_and_shape_drift() -> None:
    checked = subject.validate_controller_identity(IDENTITY_RECORD)
    assert checked.account_id == "ofc-m31-s2b-a1b2c3d4e5f6"

    legacy = copy.deepcopy(IDENTITY_RECORD)
    legacy["account_id"] = "ofc-m31-t3-controller"
    legacy["email"] = subject.LEGACY_CONTROLLER_SERVICE_ACCOUNT
    legacy["principal"] = f"serviceAccount:{legacy['email']}"
    with pytest.raises(ValueError):
        subject.validate_controller_identity(legacy)

    extra = copy.deepcopy(IDENTITY_RECORD)
    extra["unknown"] = True
    with pytest.raises(ValueError):
        subject.validate_controller_identity(extra)

    mismatched = copy.deepcopy(IDENTITY_RECORD)
    mismatched["derived_from_run_tag"] = "000000000000"
    with pytest.raises(ValueError):
        subject.validate_controller_identity(mismatched)


def test_absent_create_readback_and_no_mutation_retry() -> None:
    admin, http = _admin()
    absent = admin.require_absent()
    assert absent.receipt["get_status"] == 404
    created = admin.create(absence=absent)
    assert created.receipt["create_call_count"] == 1
    assert created.receipt["create_retry_count"] == 0
    assert created.receipt["readback_verified"] is True
    assert [row["method"] for row in http.calls] == ["GET", "POST", "GET"]
    assert all(
        "user-token"
        not in json.dumps(
            {key: value for key, value in receipt.items() if key != "x"}
        )
        for receipt in (absent.receipt, created.receipt)
    )


def test_direct_create_without_matching_absence_capability_is_blocked() -> None:
    admin, http = _admin()
    other, _ = _admin()
    foreign = other.require_absent()
    with pytest.raises(PermissionError):
        admin.create(absence=foreign)
    assert http.calls == []


def test_require_absent_fails_closed_if_identity_exists() -> None:
    admin, _ = _admin(present=True)
    with pytest.raises(
        subject.RunScopedControllerServiceAccountError,
        match="controller_service_account_freshness_failed",
    ):
        admin.require_absent()


def test_delete_once_then_get_only_polling() -> None:
    admin, http = _admin()
    created = admin.create(absence=admin.require_absent())
    http.delete_visibility_gets = 2
    ticks = iter([0.0, 0.0, 1.0, 2.0])
    sleeps: list[float] = []
    receipt = admin.delete_created_and_wait_absent(
        created=created,
        now=lambda: next(ticks),
        sleep=sleeps.append,
    )
    assert receipt["delete_call_count"] == 1
    assert receipt["delete_retry_count"] == 0
    assert receipt["absence_get_count"] == 3
    assert receipt["final_get_status"] == 404
    assert receipt["success_path_teardown_evidence"] is True
    assert (
        receipt["controller_create_receipt_sha256"]
        == created.receipt["receipt_sha256"]
    )
    assert [row["method"] for row in http.calls].count("DELETE") == 1
    assert sleeps == [2.0, 2.0]


def test_absent_delete_is_idempotent_without_mutation() -> None:
    admin, http = _admin()
    receipt = admin.cleanup_delete_if_present()
    assert receipt["delete_call_count"] == 0
    assert receipt["controller_create_receipt_sha256"] is None
    assert receipt["cloud_mutation_performed"] is False
    assert [row["method"] for row in http.calls] == ["GET"]


def test_created_identity_must_still_exist_for_success_teardown() -> None:
    admin, http = _admin()
    created = admin.create(absence=admin.require_absent())
    http.present = False
    with pytest.raises(
        subject.RunScopedControllerServiceAccountError,
        match="disappeared_before_teardown",
    ):
        admin.delete_created_and_wait_absent(created=created)
    assert [row["method"] for row in http.calls].count("DELETE") == 0


def test_create_409_is_not_retried() -> None:
    admin, http = _admin(create_status=409)
    absence = admin.require_absent()
    with pytest.raises(
        subject.RunScopedControllerServiceAccountError,
        match="controller_service_account_create_failed",
    ):
        admin.create(absence=absence)
    assert [row["method"] for row in http.calls].count("POST") == 1


def test_unknown_create_outcome_is_removed_by_failure_cleanup() -> None:
    admin, http = _admin(create_unknown_outcome=True)
    absence = admin.require_absent()
    with pytest.raises(OSError):
        admin.create(absence=absence)
    with pytest.raises(PermissionError):
        admin.create(absence=absence)
    cleanup = admin.cleanup_delete_if_present()
    assert cleanup["delete_call_count"] == 1
    assert cleanup["success_path_teardown_evidence"] is False
    assert [row["method"] for row in http.calls].count("POST") == 1
    assert [row["method"] for row in http.calls].count("DELETE") == 1


def test_absence_capability_is_one_shot() -> None:
    admin, http = _admin(create_status=409)
    absence = admin.require_absent()
    with pytest.raises(subject.RunScopedControllerServiceAccountError):
        admin.create(absence=absence)
    with pytest.raises(PermissionError):
        admin.create(absence=absence)
    assert [row["method"] for row in http.calls].count("POST") == 1
