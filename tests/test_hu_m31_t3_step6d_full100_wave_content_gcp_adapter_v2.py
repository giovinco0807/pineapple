from __future__ import annotations

import copy
import hashlib
import json
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_content_gcp_adapter_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_content_stage_v2 as content
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


TOKEN_A = "token-a-abcdefghijklmnopqrstuvwxyz"
TOKEN_B = "token-b-abcdefghijklmnopqrstuvwxyz"
CONTENT_SHA = "4" * 64
PREFIX = f"{package_v2.CONTENT_PREFIX_ROOT}/{CONTENT_SHA}"


def _payload(index: int) -> bytes:
    return f"content-object-{index:02d}\n".encode("ascii")


def _expected_multipart(object_name: str, payload: bytes, sha256: str) -> tuple[bytes, str]:
    boundary = f"ofc-f100-{sha256}"
    marker = f"--{boundary}".encode("ascii")
    metadata = json.dumps(
        {"metadata": {"sha256": sha256}, "name": object_name},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return (
        b"".join(
            (
                marker,
                b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",
                metadata,
                b"\r\n",
                marker,
                b"\r\nContent-Type: application/octet-stream\r\n\r\n",
                payload,
                b"\r\n",
                marker,
                b"--\r\n",
            )
        ),
        boundary,
    )


def _plan() -> dict:
    entries = []
    for index in range(26):
        payload = _payload(index)
        relative = f"payload/file-{index:02d}.bin"
        entries.append(
            {
                "relative_path": relative,
                "object_name": f"{PREFIX}/{relative}",
                "kind": "test_payload",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    body = {
        "schema": content.CONTENT_STAGE_PLAN_SCHEMA,
        "status": "validated_outer_package_content_stage_not_started",
        "run_name": "regular-hu-m31-c02-f100wv2-contentgcp-001",
        "execution_identity_sha256": "1" * 64,
        "wave_plan_sha256": "2" * 64,
        "outer_manifest_sha256": "3" * 64,
        "content_payload_sha256": CONTENT_SHA,
        "bucket": subject.BUCKET,
        "content_prefix": PREFIX,
        "entries": entries,
        "entry_count": 26,
        "content_create_only": True,
        "if_generation_match": 0,
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }
    body["plan_sha256"] = content.canonical_sha256(body)
    return content._validate_stage_plan_self(body)


class FakeGcs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], bytes | None, int]] = []
        self.objects: dict[str, dict[str, Any]] = {}
        self.next_generation = 1000
        self.page_size = 1000
        self.upload_status: int | None = None
        self.upload_response_fault: str | None = None
        self.drop_after_upload_commit = False
        self.metadata_404_after_upload = False
        self.ignore_generation_query = False
        self.replacement_after_delete = False
        self.provider_secret = "provider-secret-must-not-escape"

    def add_object(
        self,
        name: str,
        payload: bytes,
        *,
        generation: str | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        self.next_generation += 1
        row = {
            "name": name,
            "generation": generation or str(self.next_generation),
            "size": str(len(payload)),
            "metadata": {"sha256": sha256 or hashlib.sha256(payload).hexdigest()},
            "payload": payload,
        }
        self.objects[name] = row
        return row

    @staticmethod
    def _response(status: int, value: Any = None) -> HttpResponse:
        if value is None:
            body = b""
        elif isinstance(value, bytes):
            body = value
        else:
            body = json.dumps(value, separators=(",", ":")).encode("ascii")
        return HttpResponse(status=status, body=body, headers={})

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: int,
    ) -> HttpResponse:
        self.calls.append((method, url, dict(headers), body, timeout))
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        bucket = urllib.parse.quote(subject.BUCKET, safe="")
        list_path = f"/storage/v1/b/{bucket}/o"
        upload_path = f"/upload/storage/v1/b/{bucket}/o"
        object_path = f"/storage/v1/b/{bucket}/o/"

        if method == "GET" and parsed.path == list_path:
            assert query["prefix"] == [PREFIX + "/"]
            assert query["maxResults"] == ["1000"]
            names = sorted(
                name for name in self.objects if name.startswith(PREFIX + "/")
            )
            start = int(query.get("pageToken", ["0"])[0])
            selected = names[start : start + self.page_size]
            value: dict[str, Any] = {
                "items": [
                    {
                        key: copy.deepcopy(self.objects[name][key])
                        for key in ("name", "generation", "size", "metadata")
                    }
                    for name in selected
                ]
            }
            if start + self.page_size < len(names):
                value["nextPageToken"] = str(start + self.page_size)
            return self._response(200, value)

        if method == "POST" and parsed.path == upload_path:
            assert query["uploadType"] == ["multipart"]
            assert query["ifGenerationMatch"] == ["0"]
            assert "name" not in query
            assert body is not None
            content_type = headers["Content-Type"]
            assert content_type.startswith("multipart/related; boundary=")
            boundary = content_type.split("=", 1)[1]
            marker = f"--{boundary}".encode("ascii")
            metadata_prefix = (
                marker
                + b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
            )
            assert body.startswith(metadata_prefix)
            media_separator = (
                b"\r\n"
                + marker
                + b"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            metadata_raw, media_and_tail = body[len(metadata_prefix) :].split(
                media_separator, 1
            )
            suffix = b"\r\n" + marker + b"--\r\n"
            assert media_and_tail.endswith(suffix)
            media = media_and_tail[: -len(suffix)]
            metadata = json.loads(metadata_raw)
            assert set(metadata) == {"metadata", "name"}
            assert set(metadata["metadata"]) == {"sha256"}
            name = metadata["name"]
            sha256 = metadata["metadata"]["sha256"]
            assert headers["Content-Length"] == str(len(body))
            if self.upload_status in {409, 412} or name in self.objects:
                return self._response(
                    self.upload_status or 412, {"error": self.provider_secret}
                )
            row = self.add_object(
                name,
                media,
                sha256=sha256,
            )
            if self.drop_after_upload_commit:
                self.drop_after_upload_commit = False
                raise ConnectionError("fake connection lost after GCS commit")
            response = {
                key: copy.deepcopy(row[key])
                for key in ("name", "generation", "size", "metadata")
            }
            if self.upload_response_fault == "wrong_sha":
                response["metadata"]["sha256"] = "f" * 64
            elif self.upload_response_fault == "missing_metadata":
                response.pop("metadata")
            return self._response(200, response)

        if parsed.path.startswith(object_path):
            name = urllib.parse.unquote(parsed.path[len(object_path) :])
            row = self.objects.get(name)
            generation = query.get("generation", [None])[0]
            if method == "GET":
                if self.metadata_404_after_upload:
                    self.metadata_404_after_upload = False
                    return self._response(404, {"error": self.provider_secret})
                if row is None or (
                    generation is not None
                    and row["generation"] != generation
                    and not self.ignore_generation_query
                ):
                    return self._response(404, {"error": self.provider_secret})
                return self._response(
                    200,
                    {
                        key: copy.deepcopy(row[key])
                        for key in ("name", "generation", "size", "metadata")
                    },
                )
            if method == "DELETE":
                expected = query.get("ifGenerationMatch", [None])[0]
                if row is None:
                    return self._response(404, {"error": self.provider_secret})
                if row["generation"] != expected:
                    return self._response(412, {"error": self.provider_secret})
                del self.objects[name]
                if self.replacement_after_delete:
                    self.add_object(name, row["payload"] + b"replacement")
                return self._response(204)
        raise AssertionError(f"unexpected request {method} {url}")


def _adapter(fake: FakeGcs, *, mode: str) -> subject.GcsContentObjectAdapter:
    return subject.GcsContentObjectAdapter(
        mode=mode, stage_plan=_plan(), requester=fake
    )


def _write_entry(tmp_path: Path, index: int = 0) -> tuple[dict, Path]:
    entry = _plan()["entries"][index]
    path = tmp_path / f"file-{index:02d}.bin"
    path.write_bytes(_payload(index))
    return entry, path


def test_complete_pagination_normalizes_metadata_rows_and_reads_token_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeGcs()
    plan = _plan()
    for index in range(3):
        entry = plan["entries"][index]
        fake.add_object(entry["object_name"], _payload(index))
    fake.page_size = 1
    adapter = _adapter(fake, mode="preflight")
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    listing = adapter.list_prefix(bucket=subject.BUCKET, prefix=PREFIX)
    assert listing["complete"] is True
    assert [row["object_name"] for row in listing["objects"]] == sorted(
        row["object_name"] for row in plan["entries"][:3]
    )
    assert len(fake.calls) == 3
    assert all(call[0] == "GET" and call[3] is None for call in fake.calls)
    assert all(call[2]["Authorization"] == f"Bearer {TOKEN_A}" for call in fake.calls)
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_B)
    adapter.list_prefix(bucket=subject.BUCKET, prefix=PREFIX)
    assert fake.calls[-1][2]["Authorization"] == f"Bearer {TOKEN_B}"


def test_readback_mode_is_get_only_for_existing_immutable_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    plan = _plan()
    for index, entry in enumerate(plan["entries"]):
        fake.add_object(entry["object_name"], _payload(index))
    adapter = _adapter(fake, mode="readback")
    listing = adapter.list_prefix(bucket=subject.BUCKET, prefix=PREFIX)
    assert len(listing["objects"]) == 26
    assert [call[0] for call in fake.calls] == ["GET"]
    entry, path = _write_entry(tmp_path)
    with pytest.raises(PermissionError, match="stage mode"):
        adapter.create_object_from_file(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            source_path=str(path),
            sha256=entry["sha256"],
            bytes=entry["bytes"],
            if_generation_match=0,
        )
    with pytest.raises(PermissionError, match="cleanup mode"):
        adapter.delete_object(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            if_generation_match="1001",
        )
    assert [call[0] for call in fake.calls] == ["GET"]


def test_create_multipart_upload_is_exact_create_only_and_metadata_read_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    adapter = _adapter(fake, mode="stage")
    entry, path = _write_entry(tmp_path)
    result = adapter.create_object_from_file(
        bucket=subject.BUCKET,
        object_name=entry["object_name"],
        source_path=str(path),
        sha256=entry["sha256"],
        bytes=entry["bytes"],
        if_generation_match=0,
    )
    assert result == {
        "bucket": subject.BUCKET,
        "object_name": entry["object_name"],
        "generation": result["generation"],
        "sha256": entry["sha256"],
        "bytes": entry["bytes"],
        "created": True,
        "if_generation_match": 0,
    }
    assert [call[0] for call in fake.calls] == ["POST", "GET"]
    upload = fake.calls[0]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(upload[1]).query)
    assert query["uploadType"] == ["multipart"]
    assert query["ifGenerationMatch"] == ["0"]
    assert "name" not in query
    expected_body, boundary = _expected_multipart(
        entry["object_name"], _payload(0), entry["sha256"]
    )
    assert upload[3] == expected_body
    assert upload[2]["Content-Type"] == (
        f"multipart/related; boundary={boundary}"
    )
    assert upload[2]["Content-Length"] == str(len(expected_body))
    assert "X-Goog-Meta-Sha256" not in upload[2]
    assert upload[2]["Authorization"] == f"Bearer {TOKEN_A}"
    assert upload[4] == 600


def test_server_commit_then_connection_drop_recovers_exact_metadata_get_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    fake.drop_after_upload_commit = True
    entry, path = _write_entry(tmp_path)
    result = _adapter(fake, mode="stage").create_object_from_file(
        bucket=subject.BUCKET,
        object_name=entry["object_name"],
        source_path=str(path),
        sha256=entry["sha256"],
        bytes=entry["bytes"],
        if_generation_match=0,
    )
    assert result["created"] is True
    assert result["sha256"] == entry["sha256"]
    assert result["bytes"] == entry["bytes"]
    assert [call[0] for call in fake.calls] == ["POST", "GET"]
    assert all(call[4] == 600 for call in fake.calls)


@pytest.mark.parametrize("status", [409, 412])
def test_preexisting_upload_is_never_adopted_even_when_bytes_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    fake.upload_status = status
    entry, path = _write_entry(tmp_path)
    fake.add_object(entry["object_name"], _payload(0))
    with pytest.raises(FileExistsError, match="never"):
        _adapter(fake, mode="stage").create_object_from_file(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            source_path=str(path),
            sha256=entry["sha256"],
            bytes=entry["bytes"],
            if_generation_match=0,
        )
    assert [call[0] for call in fake.calls] == ["POST"]
    assert TOKEN_A not in str(fake.calls[-1][1])


@pytest.mark.parametrize("fault", ["wrong_sha", "missing_metadata", "get_404"])
def test_partial_upload_never_becomes_success_and_remains_visible_for_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    if fault == "get_404":
        fake.metadata_404_after_upload = True
    else:
        fake.upload_response_fault = fault
    entry, path = _write_entry(tmp_path)
    adapter = _adapter(fake, mode="stage")
    with pytest.raises(RuntimeError):
        adapter.create_object_from_file(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            source_path=str(path),
            sha256=entry["sha256"],
            bytes=entry["bytes"],
            if_generation_match=0,
        )
    listing = adapter.list_prefix(bucket=subject.BUCKET, prefix=PREFIX)
    assert [row["object_name"] for row in listing["objects"]] == [
        entry["object_name"]
    ]


def test_generation_drift_and_post_delete_replacement_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan = _plan()
    entry = plan["entries"][0]
    fake = FakeGcs()
    row = fake.add_object(entry["object_name"], _payload(0), generation="2001")
    adapter = _adapter(fake, mode="cleanup")
    with pytest.raises(RuntimeError, match="404"):
        adapter.delete_object(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            if_generation_match="1999",
        )
    fake.replacement_after_delete = True
    with pytest.raises(RuntimeError, match="absence"):
        adapter.delete_object(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            if_generation_match=row["generation"],
        )


def test_extra_object_and_arbitrary_scope_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    fake.add_object(f"{PREFIX}/unlisted.bin", b"extra")
    adapter = _adapter(fake, mode="preflight")
    with pytest.raises(RuntimeError, match="extra object"):
        adapter.list_prefix(bucket=subject.BUCKET, prefix=PREFIX)
    with pytest.raises(ValueError, match="bucket or prefix"):
        adapter.list_prefix(bucket=subject.BUCKET, prefix=PREFIX + "/other")
    with pytest.raises(ValueError, match="bucket"):
        adapter.list_prefix(bucket="other-bucket", prefix=PREFIX)


def _preflight(plan: dict) -> dict:
    core = {
        "schema": content.CONTENT_PREFLIGHT_SCHEMA,
        "status": "exact_content_prefix_absence_confirmed_create_only",
        **content._base_binding(plan),
        "expected_entry_count": 26,
        "expected_objects_sha256": content.canonical_sha256(
            [row["object_name"] for row in plan["entries"]]
        ),
        "observed_object_count": 0,
        "observed_objects": [],
        "listing_complete": True,
        "all_objects_absent": True,
        "preexisting_same_bytes_accepted": False,
        "create_authorized": True,
        "observed_at_utc": "2027-01-20T10:00:00Z",
        "current_profile_changed": False,
    }
    return content.validate_preflight_absence_receipt(
        plan, content._with_digest(core)
    )


def _partial_stage(plan: dict, preflight: dict, row: Mapping[str, Any]) -> dict:
    core = {
        "schema": content.CONTENT_STAGE_RECEIPT_SCHEMA,
        "status": "partial_content_stage_owned_generations_readback_verified",
        **content._base_binding(plan),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "expected_entry_count": 26,
        "created_entry_count": 1,
        "rows": [dict(row)],
        "listing_complete": True,
        "stage_complete": False,
        "content_create_only": True,
        "if_generation_match": 0,
        "preexisting_object_accepted": False,
        "observed_at_utc": "2027-01-20T10:01:00Z",
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }
    return content.validate_stage_receipt(
        plan, preflight, content._with_digest(core)
    )


def test_cleanup_retry_treats_confirmed_absence_as_already_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan = _plan()
    entry = plan["entries"][0]
    fake = FakeGcs()
    remote = fake.add_object(entry["object_name"], _payload(0))
    stage_row = {
        "object_name": entry["object_name"],
        "generation": remote["generation"],
        "sha256": entry["sha256"],
        "bytes": entry["bytes"],
    }
    preflight = _preflight(plan)
    stage = _partial_stage(plan, preflight, stage_row)
    adapter = _adapter(fake, mode="cleanup")
    first = content.cleanup_staged_content(
        stage_plan=plan,
        preflight_receipt=preflight,
        stage_receipt=stage,
        backend=adapter,
        observed_at_utc="2027-01-20T10:02:00Z",
    )
    assert first["delete_attempt_count"] == 1
    assert first["already_absent_count"] == 0
    second = content.cleanup_staged_content(
        stage_plan=plan,
        preflight_receipt=preflight,
        stage_receipt=stage,
        backend=adapter,
        observed_at_utc="2027-01-20T10:03:00Z",
    )
    assert second["delete_attempt_count"] == 0
    assert second["already_absent_count"] == 1
    assert second["all_objects_absent"] is True


def test_modes_token_and_provider_body_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeGcs()
    preflight = _adapter(fake, mode="preflight")
    with pytest.raises(PermissionError, match=subject.TOKEN_ENV):
        preflight.list_prefix(bucket=subject.BUCKET, prefix=PREFIX)
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    entry, path = _write_entry(tmp_path)
    with pytest.raises(PermissionError, match="stage mode"):
        preflight.create_object_from_file(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            source_path=str(path),
            sha256=entry["sha256"],
            bytes=entry["bytes"],
            if_generation_match=0,
        )
    fake.upload_status = 412
    with pytest.raises(FileExistsError) as error:
        _adapter(fake, mode="stage").create_object_from_file(
            bucket=subject.BUCKET,
            object_name=entry["object_name"],
            source_path=str(path),
            sha256=entry["sha256"],
            bytes=entry["bytes"],
            if_generation_match=0,
        )
    assert TOKEN_A not in str(error.value)
    assert fake.provider_secret not in str(error.value)
