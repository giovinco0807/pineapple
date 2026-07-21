from __future__ import annotations

import json
import subprocess
import uuid
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_cloud_adapters_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)


TOKEN = "test-token-" + "x" * 64
NAMES = ("ofc-m31-s2b-candidate-a0", "ofc-m31-s2b-reference-a0")


class _TokenSource:
    def access_token(self) -> str:
        return TOKEN


class _Http:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.calls: list[dict[str, Any]] = []

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
        return self.callback(method, url, body)


def _response(status: int, value: Any) -> rest_iam.HttpResponse:
    body = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True).encode()
    )
    return rest_iam.HttpResponse(status, body, {})


def test_gcloud_user_token_source_is_redacted_and_exact() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, TOKEN + "\n", "")

    source = subject.GcloudUserAccessTokenSource(runner=runner)
    assert source.access_token() == TOKEN
    assert calls == [
        [
            "gcloud.cmd",
            "auth",
            "print-access-token",
            f"--project={transport.PROJECT}",
            "--quiet",
        ]
    ]
    assert TOKEN not in repr(source)


def test_token_generator_converts_403_without_retaining_body() -> None:
    raw = {
        "error": {
            "code": 403,
            "status": "PERMISSION_DENIED",
            "message": "secret detail must not escape",
            "details": [{"reason": "IAM_PERMISSION_DENIED"}],
        }
    }
    http = _Http(lambda *_: _response(403, raw))
    generator = subject.IamCredentialsControllerTokenGenerator(
        controller_service_account=(
            f"ofc-m31-s2b-abcdef123456@{transport.PROJECT}."
            "iam.gserviceaccount.com"
        ),
        http_client=http,
        user_token_source=_TokenSource(),
    )
    with pytest.raises(
        token_barrier.TokenGenerationHttpError
    ) as raised:
        generator.generate_controller_access_token(
            lifetime_seconds=token_barrier.TOKEN_LIFETIME_SECONDS
        )
    error = raised.value
    assert error.status_code == 403
    assert error.error_status == "PERMISSION_DENIED"
    assert error.error_reason == "IAM_PERMISSION_DENIED"
    assert "secret detail" not in str(error)
    assert TOKEN not in repr(error)


def test_bootstrap_source_store_is_generation_pinned_and_exact() -> None:
    prefix = (
        f"gs://{transport.BUCKET}/hu-m31-r2diag-direct-v2/"
        f"bootstrap-sources/{'a' * 64}"
    )
    uris = [
        f"{prefix}/runtime_source_bundle.json",
        f"{prefix}/candidate_payload_contract.json",
        f"{prefix}/reference_payload_contract.json",
    ]
    content = b'{"schema":"test"}'

    def callback(method: str, url: str, body: bytes | None) -> Any:
        if "/storage/v1/" in url and method == "GET" and "prefix=" in url:
            return _response(200, {})
        if "/upload/storage/v1/" in url:
            assert method == "POST"
            assert body == content
            return _response(
                200,
                {
                    "bucket": transport.BUCKET,
                    "name": uris[0].split(f"{transport.BUCKET}/", 1)[1],
                    "generation": "1900200000000001",
                    "size": str(len(content)),
                },
            )
        if "/download/storage/v1/" in url:
            assert method == "GET"
            assert "generation=1900200000000001" in url
            return _response(200, content)
        raise AssertionError((method, url))

    http = _Http(callback)
    store = subject.GenerationPinnedBootstrapSourceStore(
        source_plan={
            "source_prefix": prefix,
            "objects": [{"uri": uri} for uri in uris],
        },
        http_client=http,
        user_token_source=_TokenSource(),
    )
    assert store.list_objects(prefix=prefix) == []
    receipt = store.conditional_create(
        uri=uris[0], content=content, if_generation_match=0
    )
    assert receipt["generation"] == 1_900_200_000_000_001
    assert receipt["created"] is True
    assert (
        store.generation_pinned_get(
            uri=uris[0], generation=receipt["generation"]
        )
        == content
    )
    assert len(http.calls) == 3
    assert all(
        call["headers"]["Authorization"] == f"Bearer {TOKEN}"
        for call in http.calls
    )
    with pytest.raises(ValueError):
        store.conditional_create(
            uri=f"{prefix}/fourth.json",
            content=content,
            if_generation_match=0,
        )


def test_compute_insert_is_one_shot_and_metadata_412_is_not_retried() -> None:
    insert_body = {"name": NAMES[0]}

    def callback(method: str, url: str, body: bytes | None) -> Any:
        if method == "POST" and url.split("?", 1)[0].endswith(
            "/instances"
        ):
            assert json.loads(body or b"{}") == insert_body
            return _response(
                202, {"name": "insert-op-1", "status": "DONE"}
            )
        if method == "POST" and "/setMetadata?" in url:
            return _response(412, {"error": {"code": 412}})
        raise AssertionError((method, url))

    http = _Http(callback)
    client = subject.ExactPairComputeClient(
        token_source=_TokenSource(),
        instance_names=NAMES,
        principal=(
            f"serviceAccount:ofc-m31-s2b-abcdef123456@"
            f"{transport.PROJECT}.iam.gserviceaccount.com"
        ),
        credential_kind="fixed_nonrefreshing_controller",
        http_client=http,
        allow_insert=True,
        sleeper=lambda _: None,
    )
    insert_id = str(uuid.uuid4())
    inserted = client.insert_instance(
        instance_name=NAMES[0],
        request_id=insert_id,
        body=insert_body,
    )
    assert inserted.operation_done is True
    assert inserted.operation_id == "insert-op-1"
    cas_id = str(uuid.uuid4())
    cas = client.set_metadata(
        instance_name=NAMES[0],
        request_id=cas_id,
        expected_fingerprint="fingerprint",
        values={"a": "b"},
    )
    assert cas.status_code == 412
    assert cas.operation_done is False
    assert [call["method"] for call in http.calls] == ["POST", "POST"]
    assert TOKEN not in repr(client)


def test_result_store_lists_then_reads_exact_generation() -> None:
    prefixes = [
        (
            f"gs://{transport.BUCKET}/hu-m31-r2diag-direct-v2/"
            f"stages/{'b' * 64}/results/{role}"
        )
        for role in ("candidate", "reference")
    ]
    prefixes.extend(
        (
            f"gs://{transport.BUCKET}/hu-m31-r2diag-direct-v2/"
            f"stages/{'b' * 64}/results/progress/{role}"
        )
        for role in ("candidate", "reference")
    )
    uri = f"{prefixes[0]}/DONE.envelope.json"
    name = uri.split(f"{transport.BUCKET}/", 1)[1]
    raw = b'{"done":true}'
    item = {
        "bucket": transport.BUCKET,
        "name": name,
        "generation": "1900200000000002",
        "metageneration": "1",
        "size": str(len(raw)),
        "crc32c": "AAAAAA==",
        "etag": "etag-value",
    }

    def callback(method: str, url: str, body: bytes | None) -> Any:
        assert method == "GET"
        assert body is None
        if "/download/storage/v1/" in url:
            assert "generation=1900200000000002" in url
            return _response(200, raw)
        if "/o?" in url and "prefix=" in url:
            return _response(200, {"items": [item]})
        if "/o/" in url and "fields=" in url:
            return _response(200, item)
        raise AssertionError(url)

    http = _Http(callback)
    store = subject.GenerationPinnedResultStore(
        result_prefixes=prefixes,
        http_client=http,
        user_token_source=_TokenSource(),
    )
    records = store.list_prefix(prefix=prefixes[0])
    assert len(records) == 1
    assert records[0]["uri"] == uri
    assert records[0]["generation"] == 1_900_200_000_000_002
    assert (
        store.read_bytes(
            uri=uri, generation=records[0]["generation"]
        )
        == raw
    )
    # list + metadata + media; cached read_bytes performs no fourth call.
    assert len(http.calls) == 3
    with pytest.raises(ValueError):
        store.read_current(
            uri=f"gs://{transport.BUCKET}/outside/DONE.json"
        )


def test_user_cleanup_accepts_only_exact_pair_and_404_readback() -> None:
    def callback(method: str, url: str, body: bytes | None) -> Any:
        assert method == "GET"
        assert body is None
        return _response(404, {"error": {"code": 404}})

    http = _Http(callback)
    client = subject.ExactPairComputeClient(
        token_source=_TokenSource(),
        instance_names=NAMES,
        principal="user:giovinco.080807@gmail.com",
        credential_kind="user",
        http_client=http,
        allow_insert=False,
    )
    receipt = client.cleanup_exact_instances_and_disks(
        instance_names=NAMES, disk_names=NAMES
    )
    assert receipt["exact_cleanup_complete"] is True
    assert receipt["all_four_targets_independently_processed"] is True
    assert receipt["all_four_targets_get404_verified"] is True
    assert receipt["instance_final_statuses"] == [404, 404]
    assert receipt["disk_final_statuses"] == [404, 404]
    assert len(http.calls) == 8
    with pytest.raises(ValueError):
        client.cleanup_exact_instances_and_disks(
            instance_names=(NAMES[0], "other"),
            disk_names=NAMES,
        )


def test_compute_cleanup_continues_all_targets_after_first_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http = _Http(
        lambda method, url, body: (
            _response(404, {"error": {"code": 404}})
            if method == "GET" and body is None
            else (_ for _ in ()).throw(AssertionError((method, url)))
        )
    )
    client = subject.ExactPairComputeClient(
        token_source=_TokenSource(),
        instance_names=NAMES,
        principal="user:giovinco.080807@gmail.com",
        credential_kind="user",
        http_client=http,
        allow_insert=False,
    )
    cleanup_calls: list[tuple[str, str]] = []

    def fail_first(*, kind: str, name: str) -> dict[str, Any]:
        cleanup_calls.append((kind, name))
        if len(cleanup_calls) == 1:
            raise OSError("unknown first cleanup outcome")
        return {
            "kind": kind,
            "name": name,
            "delete_call_count": 0,
            "already_absent": True,
            "final_get_status": 404,
        }

    monkeypatch.setattr(client, "_delete_exact", fail_first)
    receipt = client.cleanup_exact_instances_and_disks(
        instance_names=NAMES,
        disk_names=NAMES,
    )
    assert cleanup_calls == [
        ("instances", NAMES[0]),
        ("instances", NAMES[1]),
        ("disks", NAMES[0]),
        ("disks", NAMES[1]),
    ]
    assert receipt["cleanup_call_failure_count"] == 1
    assert receipt["all_four_targets_get404_verified"] is True
    assert len(http.calls) == 4


def test_compute_final_absence_readback_attempts_all_four_before_failing() -> None:
    statuses = iter((500, 404, 200, 404))

    def callback(method: str, url: str, body: bytes | None) -> Any:
        assert method == "GET"
        assert body is None
        status = next(statuses)
        return _response(status, {"status": status, "url": url})

    http = _Http(callback)
    client = subject.ExactPairComputeClient(
        token_source=_TokenSource(),
        instance_names=NAMES,
        principal="user:giovinco.080807@gmail.com",
        credential_kind="user",
        http_client=http,
        allow_insert=False,
    )
    with pytest.raises(
        subject.LiveCloudAdapterError,
        match="exact_pair_four_target_get404_not_proven",
    ):
        client.verify_exact_instances_and_disks_absent(
            instance_names=NAMES,
            disk_names=NAMES,
        )
    assert len(http.calls) == 4


def test_live_client_allowlist_covers_every_step11_target_host():
    # Regression: the live StdlibCloudHttpsClient formerly omitted
    # cloudresourcemanager.googleapis.com, so every project-level IAM read
    # (get_policy("project")) was rejected instantly and mislabeled as
    # iam_https_transport_failed. step11 validates URLs against its own host
    # allowlist and then hands the request to the injected live client, so
    # the live client must allow at least every host step11 permits.
    assert rest_iam._ALLOWED_HOSTS <= subject._ALLOWED_HOSTS
    assert "cloudresourcemanager.googleapis.com" in subject._ALLOWED_HOSTS


def test_live_client_still_rejects_a_non_allowlisted_host():
    # Negative control (no network): a host outside the allowlist is
    # rejected before any request is attempted, so the allowlist guard
    # itself still works after adding the project IAM host.
    client = subject.StdlibCloudHttpsClient()
    with pytest.raises(ValueError, match="escaped HTTPS allowlist"):
        client.request(
            method="POST",
            url="https://evil.example.com/v1/x",
            headers={"Authorization": "Bearer x"},
            body=b"{}",
            timeout_seconds=1,
        )
