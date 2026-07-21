from __future__ import annotations

import dataclasses
import json
from collections import deque
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1 as subject,
)


PROJECT = "ofc-solver-485418"
BUCKET = "pokerhu-ofc-solver-485418-training"
WORKER = (
    "ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com"
)
CONTROLLER = (
    "ofc-m31-t3-controller@ofc-solver-485418.iam.gserviceaccount.com"
)
TOKEN = "unit-user-token-must-never-be-serialized"
ROLE = f"projects/{PROJECT}/roles/ofcM31T3VmLaunchV1"
MEMBER = f"serviceAccount:{CONTROLLER}"
CONDITION = {
    "title": "ofc-m31-step11-time-v12-fix3",
    "description": "bounded unit condition",
    "expression": 'request.time < timestamp("2026-07-19T02:00:00Z")',
}
OTHER_CONDITION = {
    "title": "other-condition",
    "expression": 'request.time < timestamp("2026-07-20T02:00:00Z")',
}


class TokenSource:
    def access_token(self) -> str:
        return TOKEN


class FakeHttp:
    def __init__(self, responses: list[subject.HttpResponse]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> subject.HttpResponse:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.responses.popleft()


def response(status: int, payload: Mapping[str, Any]) -> subject.HttpResponse:
    return subject.HttpResponse(
        status_code=status,
        body=json.dumps(payload).encode("utf-8"),
        headers={},
    )


def policy(
    *,
    etag: str = "BwUnitEtag",
    version: int = 3,
    bindings: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "etag": etag,
        "bindings": [] if bindings is None else bindings,
    }


def client(fake: FakeHttp, **kwargs: Any) -> subject.Step11RestIamAdmin:
    return subject.Step11RestIamAdmin(
        http_client=fake,
        user_token_source=TokenSource(),
        project=PROJECT,
        bucket=BUCKET,
        worker_service_account=WORKER,
        controller_service_account=CONTROLLER,
        retry_base_seconds=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("target", "method", "host_fragment", "body"),
    [
        (
            subject.PolicyTarget.PROJECT,
            "POST",
            "cloudresourcemanager.googleapis.com",
            {"options": {"requestedPolicyVersion": 3}},
        ),
        (
            subject.PolicyTarget.BUCKET,
            "GET",
            "storage.googleapis.com",
            None,
        ),
        (
            subject.PolicyTarget.WORKER_SERVICE_ACCOUNT,
            "POST",
            "iam.googleapis.com",
            None,
        ),
        (
            subject.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT,
            "POST",
            "iam.googleapis.com",
            None,
        ),
    ],
)
def test_get_policy_uses_exact_v3_endpoint(
    target: subject.PolicyTarget,
    method: str,
    host_fragment: str,
    body: Mapping[str, Any] | None,
) -> None:
    fake = FakeHttp([response(200, policy())])
    actual = client(fake).get_policy(target)
    assert actual == policy()
    request = fake.requests[0]
    assert request["method"] == method
    assert host_fragment in request["url"]
    if target is subject.PolicyTarget.BUCKET:
        assert request["url"].endswith(
            "/iam?optionsRequestedPolicyVersion=3"
        )
    elif target in {
        subject.PolicyTarget.WORKER_SERVICE_ACCOUNT,
        subject.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT,
    }:
        assert f"/projects/{PROJECT}/serviceAccounts/" in request["url"]
        assert "/projects/-/serviceAccounts/" not in request["url"]
        assert request["url"].endswith(
            ":getIamPolicy?options.requestedPolicyVersion=3"
        )
    else:
        assert request["url"].endswith(":getIamPolicy")
    assert (
        None
        if request["body"] is None
        else json.loads(request["body"])
    ) == body
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_add_binding_updates_only_exact_condition_and_keeps_etag() -> None:
    existing = policy(
        bindings=[
            {
                "role": ROLE,
                "members": ["user:other@example.com"],
                "condition": OTHER_CONDITION,
            }
        ],
    )
    accepted = policy(
        etag="BwAccepted",
        bindings=[
            *existing["bindings"],
            {
                "role": ROLE,
                "members": [MEMBER],
                "condition": CONDITION,
            },
        ],
    )
    fake = FakeHttp([response(200, existing), response(200, accepted)])
    result = client(fake).add_binding(
        subject.PolicyTarget.PROJECT,
        role=ROLE,
        member=MEMBER,
        condition=CONDITION,
    )
    assert result.changed is True
    assert result.attempts == 1
    set_request = fake.requests[1]
    assert set_request["url"].endswith(":setIamPolicy")
    set_payload = json.loads(set_request["body"])
    assert set_payload["updateMask"] == "bindings,etag"
    assert set_payload["policy"]["etag"] == "BwUnitEtag"
    assert set_payload["policy"]["version"] == 3
    assert set_payload["policy"]["bindings"][0]["condition"] == OTHER_CONDITION
    assert set_payload["policy"]["bindings"][1]["condition"] == CONDITION


def test_idempotent_add_does_not_issue_set() -> None:
    existing = policy(
        bindings=[
            {
                "role": ROLE,
                "members": [MEMBER],
                "condition": CONDITION,
            }
        ]
    )
    fake = FakeHttp([response(200, existing)])
    result = client(fake).add_binding(
        "project", role=ROLE, member=MEMBER, condition=CONDITION
    )
    assert result.changed is False
    assert len(fake.requests) == 1


def test_remove_binding_is_exact_and_drops_only_empty_match() -> None:
    existing = policy(
        bindings=[
            {
                "role": ROLE,
                "members": [MEMBER],
                "condition": OTHER_CONDITION,
            },
            {
                "role": ROLE,
                "members": [MEMBER],
                "condition": CONDITION,
            },
        ]
    )
    accepted = policy(
        etag="BwAccepted",
        bindings=[existing["bindings"][0]],
    )
    fake = FakeHttp([response(200, existing), response(200, accepted)])
    result = client(fake).remove_binding(
        subject.PolicyTarget.BUCKET,
        role=ROLE,
        member=MEMBER,
        condition=CONDITION,
    )
    assert result.changed is True
    request = fake.requests[1]
    assert request["method"] == "PUT"
    assert request["url"].endswith("/iam")
    assert "optionsRequestedPolicyVersion" not in request["url"]
    sent_policy = json.loads(request["body"])
    assert sent_policy["bindings"] == [existing["bindings"][0]]
    assert sent_policy["etag"] == "BwUnitEtag"


@pytest.mark.parametrize("conflict_status", [409, 412])
def test_set_conflict_refetches_latest_etag_then_succeeds(
    conflict_status: int,
) -> None:
    initial = policy(etag="BwFirst")
    latest = policy(etag="BwSecond")
    accepted = policy(
        etag="BwAccepted",
        bindings=[
            {
                "role": ROLE,
                "members": [MEMBER],
                "condition": CONDITION,
            }
        ],
    )
    fake = FakeHttp(
        [
            response(200, initial),
            response(conflict_status, {"error": "conflict"}),
            response(200, latest),
            response(200, accepted),
        ]
    )
    result = client(fake).add_binding(
        "worker_service_account",
        role=ROLE,
        member=MEMBER,
        condition=CONDITION,
    )
    assert result.changed is True
    assert result.attempts == 2
    assert json.loads(fake.requests[1]["body"])["policy"]["etag"] == "BwFirst"
    assert json.loads(fake.requests[3]["body"])["policy"]["etag"] == "BwSecond"


def test_conflict_retry_is_bounded_and_secret_free() -> None:
    fake = FakeHttp(
        [
            response(200, policy(etag="Bw1")),
            response(412, {"error": TOKEN}),
            response(200, policy(etag="Bw2")),
            response(412, {"error": TOKEN}),
        ]
    )
    with pytest.raises(subject.RestIamAdminError) as caught:
        client(fake, max_mutation_attempts=2).add_binding(
            "controller_service_account",
            role=ROLE,
            member=MEMBER,
            condition=CONDITION,
        )
    assert caught.value.code == "iam_policy_conflict_retry_exhausted"
    assert caught.value.attempts == 2
    assert TOKEN not in str(caught.value)
    assert TOKEN not in repr(caught.value)
    assert len(fake.requests) == 4


def test_nonretry_error_and_malformed_policy_fail_closed_without_body() -> None:
    fake = FakeHttp([response(403, {"error": TOKEN})])
    with pytest.raises(subject.RestIamAdminError) as caught:
        client(fake).get_policy("project")
    assert caught.value.code == "iam_policy_get_failed"
    assert caught.value.status_code == 403
    assert TOKEN not in str(caught.value)
    malformed = FakeHttp(
        [response(200, {"version": 3, "bindings": []})]
    )
    with pytest.raises(subject.RestIamAdminError) as malformed_caught:
        client(malformed).get_policy("project")
    assert malformed_caught.value.code == "iam_policy_etag_missing"


def test_duplicate_role_condition_binding_fails_closed() -> None:
    binding = {
        "role": ROLE,
        "members": [MEMBER],
        "condition": CONDITION,
    }
    fake = FakeHttp(
        [response(200, policy(bindings=[binding, dict(binding)]))]
    )
    with pytest.raises(subject.RestIamAdminError) as caught:
        client(fake).get_policy("project")
    assert caught.value.code == "iam_policy_binding_ambiguous"


def test_conditional_policy_requires_version_3() -> None:
    binding = {
        "role": ROLE,
        "members": [MEMBER],
        "condition": CONDITION,
    }
    fake = FakeHttp(
        [response(200, policy(version=1, bindings=[binding]))]
    )
    with pytest.raises(subject.RestIamAdminError) as caught:
        client(fake).get_policy("project")
    assert caught.value.code == "iam_conditional_policy_requires_version_3"


@pytest.mark.parametrize(
    "binding",
    [
        {"role": ROLE, "members": []},
        {"role": ROLE, "members": [MEMBER], "unexpected": True},
    ],
)
def test_invalid_binding_shape_fails_closed(
    binding: Mapping[str, Any],
) -> None:
    fake = FakeHttp([response(200, policy(bindings=[binding]))])
    with pytest.raises(subject.RestIamAdminError) as caught:
        client(fake).get_policy("project")
    assert caught.value.code == "iam_policy_binding_invalid"


def test_generate_token_is_exact_controller_scoped_and_not_serializable() -> None:
    generated = "generated-controller-token-never-serialize"
    fake = FakeHttp(
        [
            response(
                200,
                {
                    "accessToken": generated,
                    "expireTime": "2026-07-19T02:00:00Z",
                },
            )
        ]
    )
    token = client(fake).generate_controller_access_token(
        lifetime_seconds=1800
    )
    request = fake.requests[0]
    assert request["method"] == "POST"
    assert (
        "iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
        in request["url"]
    )
    assert CONTROLLER.replace("@", "%40") in request["url"]
    assert request["url"].endswith(":generateAccessToken")
    assert json.loads(request["body"]) == {
        "scope": [subject.GOOGLE_CLOUD_PLATFORM_SCOPE],
        "lifetime": "1800s",
    }
    assert token.access_token() == generated
    assert generated not in repr(token)
    assert generated not in str(token)
    with pytest.raises(TypeError):
        dataclasses.asdict(token)  # type: ignore[arg-type]
    assert not hasattr(token, "__dict__")


def test_token_source_failure_and_transport_failure_are_sanitized() -> None:
    class BrokenToken:
        def access_token(self) -> str:
            raise RuntimeError(TOKEN)

    fake = FakeHttp([])
    admin = subject.Step11RestIamAdmin(
        http_client=fake,
        user_token_source=BrokenToken(),
        project=PROJECT,
        bucket=BUCKET,
        worker_service_account=WORKER,
        controller_service_account=CONTROLLER,
    )
    with pytest.raises(subject.RestIamAdminError) as caught:
        admin.get_policy("project")
    assert caught.value.code == "user_access_token_unavailable"
    assert TOKEN not in str(caught.value)

    class BrokenHttp:
        def request(self, **kwargs: Any) -> subject.HttpResponse:
            raise RuntimeError(TOKEN)

    transport_admin = subject.Step11RestIamAdmin(
        http_client=BrokenHttp(),
        user_token_source=TokenSource(),
        project=PROJECT,
        bucket=BUCKET,
        worker_service_account=WORKER,
        controller_service_account=CONTROLLER,
    )
    with pytest.raises(subject.RestIamAdminError) as transport_caught:
        transport_admin.get_policy("project")
    assert transport_caught.value.code == "iam_https_transport_failed"
    assert TOKEN not in str(transport_caught.value)


class _RaisingHttp:
    """Fake transport where an Exception item raises (a transport failure)."""

    def __init__(self, items: list[Any]) -> None:
        self.items = deque(items)
        self.calls = 0

    def request(self, **_kwargs: Any) -> subject.HttpResponse:
        self.calls += 1
        item = self.items.popleft()
        if isinstance(item, Exception):
            raise item
        return item


def _sleepless(_seconds: float) -> None:
    return None


def test_set_transport_failure_is_retried_then_succeeds() -> None:
    accepted = policy(
        etag="BwAccepted",
        bindings=[{"role": ROLE, "members": [MEMBER], "condition": CONDITION}],
    )
    fake = _RaisingHttp(
        [
            response(200, policy(etag="BwFirst")),  # initial get
            OSError("simulated transport drop"),     # set: transport failure
            response(200, accepted),                 # set retry: success
        ]
    )
    admin = subject.Step11RestIamAdmin(
        http_client=fake,
        user_token_source=TokenSource(),
        project=PROJECT,
        bucket=BUCKET,
        worker_service_account=WORKER,
        controller_service_account=CONTROLLER,
        retry_base_seconds=0,
        sleeper=_sleepless,
    )
    result = admin.add_binding(
        "worker_service_account", role=ROLE, member=MEMBER, condition=CONDITION
    )
    assert result.changed is True
    assert fake.calls == 3  # get, failed set, retried set


def test_set_transport_retry_is_bounded_and_reraises() -> None:
    # initial get + a set that fails transport on every one of the bounded
    # retries -> the bare transport error is re-raised, never a create.
    items: list[Any] = [response(200, policy(etag="Bw"))]
    items += [OSError("drop")] * (subject.MAX_SET_TRANSPORT_RETRIES + 1)
    fake = _RaisingHttp(items)
    admin = subject.Step11RestIamAdmin(
        http_client=fake,
        user_token_source=TokenSource(),
        project=PROJECT,
        bucket=BUCKET,
        worker_service_account=WORKER,
        controller_service_account=CONTROLLER,
        retry_base_seconds=0,
        sleeper=_sleepless,
    )
    with pytest.raises(subject.RestIamAdminError) as caught:
        admin.add_binding(
            "worker_service_account",
            role=ROLE,
            member=MEMBER,
            condition=CONDITION,
        )
    assert caught.value.code == "iam_https_transport_failed"
    # 1 get + (MAX_SET_TRANSPORT_RETRIES + 1) set attempts
    assert fake.calls == 1 + (subject.MAX_SET_TRANSPORT_RETRIES + 1)


def test_http_status_error_on_set_is_not_transport_retried() -> None:
    # A 403 (HTTP response present) must fail closed immediately, not be
    # retried as a transport failure.
    fake = _RaisingHttp(
        [
            response(200, policy(etag="Bw")),
            response(403, {"error": TOKEN}),
        ]
    )
    admin = subject.Step11RestIamAdmin(
        http_client=fake,
        user_token_source=TokenSource(),
        project=PROJECT,
        bucket=BUCKET,
        worker_service_account=WORKER,
        controller_service_account=CONTROLLER,
        retry_base_seconds=0,
        sleeper=_sleepless,
    )
    with pytest.raises(subject.RestIamAdminError) as caught:
        admin.add_binding(
            "worker_service_account",
            role=ROLE,
            member=MEMBER,
            condition=CONDITION,
        )
    assert caught.value.code == "iam_policy_set_failed"
    assert fake.calls == 2  # get + one set, no transport retry
    assert TOKEN not in str(caught.value)
