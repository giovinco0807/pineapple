from __future__ import annotations

import copy
import json
import urllib.parse
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_worker_identity_gcp_adapter_v2 as subject,
)
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as identity
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


OBSERVED = "2027-01-15T08:00:00Z"
EXPIRES = "2027-01-15T08:05:00Z"
TOKEN = "ya29.fixed-test-token-without-whitespace"


def _evidence() -> tuple[dict, dict, dict, dict, dict]:
    plan = wave.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-identitygcp-001",
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave.build_observed_transition(
        plan,
        project_id=identity.PROJECT,
        zone="asia-northeast1-b",
        observed_at_utc="2027-01-15T07:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave.empty_attempt_history(plan),
    )
    ledger = wave.build_attempt_ledger(plan, transitions=[transition])
    resume = wave.build_resume_plan(plan, attempt_ledger=ledger)
    identity_plan = identity.build_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=0,
    )
    setup = identity.build_create_only_setup_plan(
        setup_nonce="fixed-pool-setup-20270115-001",
        issued_at_utc="2027-01-15T07:30:00Z",
    )
    return plan, ledger, resume, identity_plan, setup


def _desired(index: int) -> dict[str, Any]:
    account_id = identity.ACCOUNT_IDS[index]
    email = identity.ACCOUNT_EMAILS[index]
    return {
        "name": identity.account_name(email),
        "projectId": identity.PROJECT,
        "uniqueId": str(100_000_000_000_000_000_000 + index),
        "email": email,
        "displayName": f"OFC full100 worker {index:02d}",
        "description": "Fixed unprivileged OFC full100 worker identity",
        "disabled": False,
        "accountId": account_id,
    }


def _account_url(email: str) -> str:
    return (
        "https://iam.googleapis.com/v1/projects/"
        f"{identity.PROJECT}/serviceAccounts/"
        f"{urllib.parse.quote(email, safe='')}"
    )


def _actas_url(name: str) -> str:
    return f"https://iam.googleapis.com/v1/{name}:testIamPermissions"


class FakeIamProvider:
    def __init__(self, *, missing: set[int] | None = None) -> None:
        missing = set() if missing is None else set(missing)
        self.accounts = {
            identity.ACCOUNT_IDS[index]: _desired(index)
            for index in range(identity.POOL_SIZE)
            if index not in missing
        }
        self.calls: list[dict[str, Any]] = []
        self.create_status: int | None = None
        self.drop_after_create_commit = False
        self.actas_response: dict[str, Any] = {
            "permissions": [identity.ACT_AS_PERMISSION]
        }
        self.policy: dict[str, Any] = {
            "version": 3,
            "etag": "BwYFixedPoolAudit==",
            "bindings": [
                {
                    "role": "roles/viewer",
                    "members": ["user:unrelated@example.com"],
                }
            ],
            "auditConfigs": [],
        }

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert headers["Accept"] == "application/json"
        assert timeout_seconds == 60

        by_get_url = {
            _account_url(row["email"]): account_id
            for account_id, row in {
                account_id: _desired(index)
                for index, account_id in enumerate(identity.ACCOUNT_IDS)
            }.items()
        }
        if method == "GET" and url in by_get_url:
            assert body is None
            account = self.accounts.get(by_get_url[url])
            if account is None:
                return HttpResponse(404, b"{}", {})
            return HttpResponse(200, json.dumps(account).encode("utf-8"), {})

        accounts_url = (
            "https://iam.googleapis.com/v1/projects/"
            f"{identity.PROJECT}/serviceAccounts"
        )
        if method == "POST" and url == accounts_url:
            assert headers["Content-Type"] == "application/json"
            payload = json.loads(body or b"")
            account_id = payload["accountId"]
            index = identity.ACCOUNT_IDS.index(account_id)
            assert payload == {
                "accountId": account_id,
                "serviceAccount": {
                    "displayName": f"OFC full100 worker {index:02d}",
                    "description": "Fixed unprivileged OFC full100 worker identity",
                },
            }
            if self.create_status is not None:
                return HttpResponse(self.create_status, b"{}", {})
            if account_id in self.accounts:
                return HttpResponse(409, b"{}", {})
            self.accounts[account_id] = _desired(index)
            if self.drop_after_create_commit:
                self.drop_after_create_commit = False
                raise ConnectionError("fake response lost after IAM commit")
            return HttpResponse(
                201, json.dumps(self.accounts[account_id]).encode("utf-8"), {}
            )

        actas_urls = {
            _actas_url(identity.account_name(email))
            for email in identity.ACCOUNT_EMAILS
        }
        if method == "POST" and url in actas_urls:
            assert headers["Content-Type"] == "application/json"
            assert json.loads(body or b"") == {
                "permissions": [identity.ACT_AS_PERMISSION]
            }
            return HttpResponse(
                200, json.dumps(self.actas_response).encode("utf-8"), {}
            )

        policy_url = (
            "https://cloudresourcemanager.googleapis.com/v1/projects/"
            f"{identity.PROJECT}:getIamPolicy"
        )
        if method == "POST" and url == policy_url:
            assert headers["Content-Type"] == "application/json"
            assert json.loads(body or b"") == {
                "options": {"requestedPolicyVersion": 3}
            }
            return HttpResponse(200, json.dumps(self.policy).encode("utf-8"), {})

        raise AssertionError(f"unexpected IAM request: {method} {url}")


def _adapter(
    evidence: tuple[dict, dict, dict, dict, dict],
    provider: FakeIamProvider,
    *,
    mode: str,
    setup_plan: dict | None = None,
    read_receipt: dict | None = None,
    inventory_receipt: dict | None = None,
) -> subject.WorkerIdentityGcpAdapterV2:
    plan, ledger, resume, identity_plan, _ = evidence
    return subject.WorkerIdentityGcpAdapterV2(
        mode=mode,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        identity_plan=identity_plan,
        requester=provider,
        setup_plan=setup_plan,
        read_receipt=read_receipt,
        inventory_receipt=inventory_receipt,
    )


def _read(
    evidence: tuple[dict, dict, dict, dict, dict],
    provider: FakeIamProvider,
    *,
    with_setup: bool = True,
) -> dict:
    setup = evidence[4] if with_setup else None
    return _adapter(
        evidence, provider, mode="read", setup_plan=setup
    ).read_pool(observed_at_utc=OBSERVED, expires_at_utc=EXPIRES)


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN)


def test_existing_pool_read_is_exact_read_only_inventory_and_not_recreated() -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    receipt = _read(evidence, provider)

    assert receipt["all_accounts_exist"] is True
    assert receipt["inventory_receipt"] is not None
    assert receipt["http_get_count"] == identity.POOL_SIZE
    assert receipt["cloud_mutated"] is False
    assert [(row["method"], row["url"]) for row in provider.calls] == [
        ("GET", _account_url(email)) for email in identity.ACCOUNT_EMAILS
    ]
    assert TOKEN not in json.dumps(receipt)

    create = _adapter(
        evidence,
        provider,
        mode="create-missing",
        setup_plan=evidence[4],
        read_receipt=receipt,
    )
    with pytest.raises(PermissionError, match="read-only inventory"):
        create.create_missing(
            current_time_utc="2027-01-15T08:01:00Z",
            completed_at_utc="2027-01-15T08:02:00Z",
        )
    assert len(provider.calls) == identity.POOL_SIZE


def test_partial_pool_creates_only_freshly_proven_missing_then_reads_all() -> None:
    evidence = _evidence()
    provider = FakeIamProvider(missing={2, 6})
    read = _read(evidence, provider)
    assert read["missing_account_ids"] == [
        identity.ACCOUNT_IDS[2],
        identity.ACCOUNT_IDS[6],
    ]

    receipt = _adapter(
        evidence,
        provider,
        mode="create-missing",
        setup_plan=evidence[4],
        read_receipt=read,
    ).create_missing(
        current_time_utc="2027-01-15T08:01:00Z",
        completed_at_utc="2027-01-15T08:02:00Z",
    )

    assert receipt["created_account_ids"] == read["missing_account_ids"]
    assert receipt["created_account_count"] == 2
    assert receipt["preexisting_account_count"] == 6
    assert receipt["setup_receipt"] is None
    assert receipt["inventory_receipt"] is not None
    assert receipt["http_post_count"] == 2
    assert receipt["http_get_count"] == 16
    create_calls = [row for row in provider.calls if row["method"] == "POST"]
    assert [json.loads(row["body"])["accountId"] for row in create_calls] == [
        identity.ACCOUNT_IDS[2],
        identity.ACCOUNT_IDS[6],
    ]

    before = len(provider.calls)
    repeat = _read(evidence, provider)
    assert repeat["all_accounts_exist"] is True
    assert all(row["method"] == "GET" for row in provider.calls[before:])


def test_create_response_loss_returns_fresh_missing_only_recovery_read() -> None:
    evidence = _evidence()
    provider = FakeIamProvider(missing={2, 6})
    initial = _read(evidence, provider)
    provider.drop_after_create_commit = True
    with pytest.raises(subject.WorkerIdentityCreateIncompleteError) as caught:
        _adapter(
            evidence,
            provider,
            mode="create-missing",
            setup_plan=evidence[4],
            read_receipt=initial,
        ).create_missing(
            current_time_utc="2027-01-15T08:01:00Z",
            completed_at_utc="2027-01-15T08:02:00Z",
        )
    recovered = caught.value.recovery_read_receipt
    assert recovered["missing_account_ids"] == [identity.ACCOUNT_IDS[6]]
    assert recovered["existing_account_count"] == 7
    assert recovered["http_get_count"] == identity.POOL_SIZE

    completed = _adapter(
        evidence,
        provider,
        mode="create-missing",
        setup_plan=evidence[4],
        read_receipt=recovered,
    ).create_missing(
        current_time_utc="2027-01-15T08:02:30Z",
        completed_at_utc="2027-01-15T08:03:00Z",
    )
    assert completed["created_account_ids"] == [identity.ACCOUNT_IDS[6]]
    assert completed["inventory_receipt"] is not None


def test_all_missing_pool_produces_one_time_setup_and_inventory_receipts() -> None:
    evidence = _evidence()
    provider = FakeIamProvider(missing=set(range(identity.POOL_SIZE)))
    read = _read(evidence, provider)
    assert read["all_accounts_missing"] is True

    receipt = _adapter(
        evidence,
        provider,
        mode="create-missing",
        setup_plan=evidence[4],
        read_receipt=read,
    ).create_missing(
        current_time_utc="2027-01-15T08:01:00Z",
        completed_at_utc="2027-01-15T08:02:00Z",
    )
    assert receipt["status"] == "all_eight_fixed_worker_accounts_created_once"
    assert receipt["created_account_count"] == identity.POOL_SIZE
    assert receipt["setup_receipt"]["all_accounts_newly_created"] is True
    assert receipt["inventory_receipt"]["all_enabled"] is True
    assert receipt["roles_added"] is False
    assert receipt["accounts_deleted"] is False


@pytest.mark.parametrize(
    ("current", "completed", "match"),
    [
        ("2027-01-15T07:59:59Z", "2027-01-15T08:02:00Z", "future-dated"),
        ("2027-01-15T08:05:01Z", "2027-01-15T08:06:00Z", "stale"),
        ("2027-01-15T08:01:00Z", "2027-01-15T08:00:59Z", "predates"),
    ],
)
def test_create_requires_fresh_read_and_monotonic_execution_time(
    current: str, completed: str, match: str
) -> None:
    evidence = _evidence()
    provider = FakeIamProvider(missing={0})
    read = _read(evidence, provider)
    before = len(provider.calls)
    create = _adapter(
        evidence,
        provider,
        mode="create-missing",
        setup_plan=evidence[4],
        read_receipt=read,
    )
    with pytest.raises((PermissionError, ValueError), match=match):
        create.create_missing(
            current_time_utc=current,
            completed_at_utc=completed,
        )
    assert len(provider.calls) == before


@pytest.mark.parametrize("status", [409, 412])
def test_create_collision_or_precondition_is_never_adopted(status: int) -> None:
    evidence = _evidence()
    provider = FakeIamProvider(missing={4})
    read = _read(evidence, provider)
    provider.create_status = status
    before = len(provider.calls)
    with pytest.raises(RuntimeError, match="collision/precondition"):
        _adapter(
            evidence,
            provider,
            mode="create-missing",
            setup_plan=evidence[4],
            read_receipt=read,
        ).create_missing(
            current_time_utc="2027-01-15T08:01:00Z",
            completed_at_utc="2027-01-15T08:02:00Z",
        )
    assert len(provider.calls) == before + 1
    assert identity.ACCOUNT_IDS[4] not in provider.accounts


@pytest.mark.parametrize("drift", ["display", "disabled", "project", "duplicate_uid"])
def test_read_fails_closed_on_provider_identity_or_configuration_drift(
    drift: str,
) -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    if drift == "display":
        provider.accounts[identity.ACCOUNT_IDS[1]]["displayName"] = "changed"
    elif drift == "disabled":
        provider.accounts[identity.ACCOUNT_IDS[1]]["disabled"] = True
    elif drift == "project":
        provider.accounts[identity.ACCOUNT_IDS[1]]["projectId"] = "other-project"
    else:
        provider.accounts[identity.ACCOUNT_IDS[1]]["uniqueId"] = provider.accounts[
            identity.ACCOUNT_IDS[0]
        ]["uniqueId"]
    with pytest.raises((RuntimeError, ValueError)):
        _read(evidence, provider)


def test_resealed_read_receipt_cannot_change_missing_authority() -> None:
    evidence = _evidence()
    provider = FakeIamProvider(missing={3})
    receipt = _read(evidence, provider)
    forged = copy.deepcopy(receipt)
    forged["missing_account_ids"] = [identity.ACCOUNT_IDS[4]]
    core = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = subject.canonical_sha256(core)
    with pytest.raises(ValueError, match="contract changed"):
        _adapter(
            evidence,
            provider,
            mode="create-missing",
            setup_plan=evidence[4],
            read_receipt=forged,
        )


def test_actas_check_posts_exact_permission_for_every_selected_account() -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    inventory = _read(evidence, provider)["inventory_receipt"]
    provider.calls.clear()
    receipt = _adapter(
        evidence,
        provider,
        mode="actas-check",
        inventory_receipt=inventory,
    ).check_service_account_act_as(tested_at_utc="2027-01-15T08:01:00Z")

    assert receipt["every_selected_account_grants_exact_act_as"] is True
    assert len(provider.calls) == identity.POOL_SIZE
    assert [(row["method"], row["url"]) for row in provider.calls] == [
        ("POST", _actas_url(identity.account_name(email)))
        for email in identity.ACCOUNT_EMAILS
    ]
    assert all(
        json.loads(row["body"])
        == {"permissions": [identity.ACT_AS_PERMISSION]}
        for row in provider.calls
    )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"permissions": []},
        {"permissions": [identity.ACT_AS_PERMISSION, "iam.serviceAccounts.get"]},
        {"permissions": [identity.ACT_AS_PERMISSION], "unexpected": True},
    ],
)
def test_actas_check_rejects_missing_extra_or_shape_drift(
    response: dict[str, Any],
) -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    inventory = _read(evidence, provider)["inventory_receipt"]
    provider.actas_response = response
    with pytest.raises((PermissionError, RuntimeError)):
        _adapter(
            evidence,
            provider,
            mode="actas-check",
            inventory_receipt=inventory,
        ).check_service_account_act_as(tested_at_utc="2027-01-15T08:01:00Z")


def test_project_iam_scan_is_one_exact_read_and_proves_zero_pool_roles() -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    inventory = _read(evidence, provider)["inventory_receipt"]
    provider.calls.clear()
    receipt = _adapter(
        evidence,
        provider,
        mode="project-iam-scan",
        inventory_receipt=inventory,
    ).scan_project_iam(observed_at_utc="2027-01-15T08:01:00Z")

    assert receipt["all_pool_accounts_have_zero_project_roles"] is True
    assert receipt["pool_role_membership_count"] == 0
    assert receipt["read_only_observation"] is True
    assert len(provider.calls) == 1
    assert provider.calls[0]["method"] == "POST"
    assert provider.calls[0]["url"].endswith(
        f"projects/{identity.PROJECT}:getIamPolicy"
    )
    assert json.loads(provider.calls[0]["body"]) == {
        "options": {"requestedPolicyVersion": 3}
    }


def test_project_iam_scan_accepts_live_response_with_omitted_empty_audit_configs() -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    inventory = _read(evidence, provider)["inventory_receipt"]
    del provider.policy["auditConfigs"]
    provider.calls.clear()

    receipt = _adapter(
        evidence,
        provider,
        mode="project-iam-scan",
        inventory_receipt=inventory,
    ).scan_project_iam(observed_at_utc="2027-01-15T08:01:00Z")

    assert receipt["scanned_policy"]["auditConfigs"] == []
    assert set(receipt["scanned_policy"]) == {
        "version", "etag", "bindings", "auditConfigs"
    }
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "policy",
    [
        {"version": 3, "etag": "BwYFixedPoolAudit==", "bindings": [], "unknown": True},
        {"version": 3, "etag": "BwYFixedPoolAudit==", "bindings": [{}]},
    ],
)
def test_project_iam_scan_adapter_rejects_live_shape_drift_or_malformed_binding(
    policy: dict[str, Any],
) -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    inventory = _read(evidence, provider)["inventory_receipt"]
    provider.policy = policy

    with pytest.raises(ValueError, match="fields changed|malformed"):
        _adapter(
            evidence,
            provider,
            mode="project-iam-scan",
            inventory_receipt=inventory,
        ).scan_project_iam(observed_at_utc="2027-01-15T08:01:00Z")


@pytest.mark.parametrize("deleted_prefix", ["", "deleted:"])
def test_project_iam_scan_rejects_direct_or_deleted_pool_membership(
    deleted_prefix: str,
) -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    inventory = _read(evidence, provider)["inventory_receipt"]
    provider.policy["bindings"].append(
        {
            "role": "roles/viewer",
            "members": [
                f"{deleted_prefix}serviceAccount:{identity.ACCOUNT_EMAILS[0]}"
            ],
        }
    )
    with pytest.raises(PermissionError, match="project-level role"):
        _adapter(
            evidence,
            provider,
            mode="project-iam-scan",
            inventory_receipt=inventory,
        ).scan_project_iam(observed_at_utc="2027-01-15T08:01:00Z")


def test_token_and_mode_evidence_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    provider = FakeIamProvider()
    monkeypatch.delenv(subject.TOKEN_ENV)
    with pytest.raises(PermissionError, match=subject.TOKEN_ENV):
        _read(evidence, provider)
    assert provider.calls == []

    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN)
    with pytest.raises(PermissionError, match="requires setup plan"):
        _adapter(evidence, provider, mode="create-missing")
    with pytest.raises(PermissionError, match="validated inventory"):
        _adapter(evidence, provider, mode="actas-check")
