"""Injectable GCS JSON adapter for full100 wave-v2 content staging.

The adapter implements exactly ``ContentObjectBackend`` and is confined to a
validated 26-object content-stage plan.  Its modes expose disjoint mutation
surfaces.  Credentials are read from ``GOOGLE_OAUTH_ACCESS_TOKEN`` for every
request and are never retained or included in errors.

No live call occurs on construction.  Tests inject a requester; production may
explicitly use the stdlib requester supplied by the existing cloud transport.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_content_stage_v2 as content_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
    _stdlib_http_request,
)


PROJECT = "ofc-solver-485418"
BUCKET = "pokerhu-ofc-solver-485418-training"
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
MODES = frozenset({"preflight", "readback", "stage", "cleanup"})
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_LIST_PAGES = 100
HTTP_TIMEOUT_SECONDS = 600
UPLOAD_RECOVERY_READ_ATTEMPTS = 3

_SHA = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[1-9][0-9]*$")

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


class ContentTransportError(RuntimeError):
    """The HTTP requester failed without a provider status response."""


def _strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    if len(value) > MAX_RESPONSE_BYTES:
        raise RuntimeError(f"{label} response exceeded the fixed body limit")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        parsed = json.loads(value, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} response was not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} response was not a JSON object")
    return parsed


def _positive_decimal(value: Any, label: str) -> str:
    rendered = str(value)
    if isinstance(value, bool) or _GENERATION.fullmatch(rendered) is None:
        raise RuntimeError(f"{label} was not a positive decimal")
    return rendered


def _nonnegative_size(value: Any, label: str) -> int:
    rendered = str(value)
    if isinstance(value, bool) or not rendered.isdigit():
        raise RuntimeError(f"{label} was not a nonnegative decimal")
    result = int(rendered)
    if result > MAX_UPLOAD_BYTES:
        raise RuntimeError(f"{label} exceeded the fixed body limit")
    return result


def _quoted(value: str) -> str:
    return urllib.parse.quote(value, safe="")


class GcsContentObjectAdapter:
    """Mode-separated, plan-bound implementation of ``ContentObjectBackend``."""

    def __init__(
        self,
        *,
        mode: str,
        stage_plan: Mapping[str, Any],
        requester: HttpRequester = _stdlib_http_request,
    ) -> None:
        if mode not in MODES:
            raise ValueError("content GCS adapter mode changed")
        if not callable(requester):
            raise TypeError("content GCS requester must be callable")
        # The producer-owned validator re-derives the sealed plan, including
        # all 26 exact object names, hashes, sizes, and the content prefix.
        plan = content_v2._validate_stage_plan_self(stage_plan)
        if plan["bucket"] != BUCKET:
            raise ValueError("content GCS adapter escaped the fixed bucket")
        self.mode = mode
        self._plan = deepcopy(plan)
        self._bucket = BUCKET
        self._prefix = plan["content_prefix"]
        self._entries = {
            row["object_name"]: deepcopy(dict(row)) for row in plan["entries"]
        }
        if len(self._entries) != 26:
            raise ValueError("content GCS adapter inventory changed")
        self._requester = requester

    @property
    def plan_sha256(self) -> str:
        return self._plan["plan_sha256"]

    def _check_scope(self, *, bucket: str, object_name: str | None = None) -> None:
        if bucket != self._bucket:
            raise ValueError("content GCS request escaped the fixed bucket")
        if object_name is not None and object_name not in self._entries:
            raise ValueError("content GCS request escaped the allowed object inventory")

    def _check_prefix(self, *, bucket: str, prefix: str) -> None:
        if bucket != self._bucket or prefix != self._prefix:
            raise ValueError("content GCS listing escaped the exact bucket or prefix")

    @staticmethod
    def _headers(
        *, content_type: str | None = None, content_length: int | None = None
    ) -> dict[str, str]:
        token = os.environ.get(TOKEN_ENV)
        if (
            not isinstance(token, str)
            or len(token) < 20
            or any(character.isspace() for character in token)
        ):
            raise PermissionError(
                f"Bearer token must be supplied only through {TOKEN_ENV}"
            )
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        if content_length is not None:
            if (
                isinstance(content_length, bool)
                or not isinstance(content_length, int)
                or content_length < 0
                or content_length > MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
            ):
                raise ValueError("content request length escaped the fixed bound")
            headers["Content-Length"] = str(content_length)
        return headers

    def _request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        allowed_statuses: Sequence[int] = (200,),
    ) -> HttpResponse:
        allowed_methods = {
            "preflight": {"GET"},
            "readback": {"GET"},
            "stage": {"GET", "POST"},
            "cleanup": {"GET", "DELETE"},
        }[self.mode]
        if method not in allowed_methods:
            raise PermissionError("HTTP method escaped the content GCS adapter mode")
        if (
            body is not None
            and len(body) > MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
        ):
            raise ValueError("content upload exceeded the fixed body limit")
        headers = self._headers(
            content_type=content_type,
            content_length=None if body is None else len(body),
        )
        try:
            response = self._requester(
                method,
                url,
                headers,
                body,
                HTTP_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise ContentTransportError(
                f"content GCS {method} transport failed without provider status"
            ) from exc
        if not isinstance(response, HttpResponse):
            raise RuntimeError("content GCS requester returned an invalid response")
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise RuntimeError("content GCS response exceeded the fixed body limit")
        if response.status not in allowed_statuses:
            # Never copy provider response bodies into errors; they may contain
            # request IDs or credential-adjacent diagnostics.
            raise RuntimeError(
                f"content GCS {method} failed with status {response.status}"
            )
        return response

    def _metadata_url(self, object_name: str, *, generation: str | None = None) -> str:
        self._check_scope(bucket=self._bucket, object_name=object_name)
        query = {
            "fields": "name,generation,size,metadata",
        }
        if generation is not None:
            query["generation"] = generation
        return (
            f"https://storage.googleapis.com/storage/v1/b/{_quoted(self._bucket)}/o/"
            f"{_quoted(object_name)}?{urllib.parse.urlencode(query)}"
        )

    def _metadata_row(
        self,
        value: Mapping[str, Any],
        *,
        expected_name: str | None = None,
        allow_unknown: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError("content GCS metadata was not an object")
        name = value.get("name")
        if not isinstance(name, str) or not name.startswith(f"{self._prefix}/"):
            raise RuntimeError("content GCS metadata escaped the exact prefix")
        if expected_name is not None and name != expected_name:
            raise RuntimeError("content GCS metadata name changed")
        frozen = self._entries.get(name)
        if frozen is None and not allow_unknown:
            raise RuntimeError("content GCS metadata contains an extra object")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != {"sha256"}:
            raise RuntimeError("content GCS custom SHA-256 metadata changed")
        sha256 = metadata.get("sha256")
        if not isinstance(sha256, str) or _SHA.fullmatch(sha256) is None:
            raise RuntimeError("content GCS custom SHA-256 metadata changed")
        size = _nonnegative_size(value.get("size"), "content GCS object size")
        generation = _positive_decimal(
            value.get("generation"), "content GCS object generation"
        )
        if frozen is not None and (
            sha256 != frozen["sha256"] or size != frozen["bytes"]
        ):
            raise RuntimeError("content GCS metadata hash or size changed")
        return {
            "bucket": self._bucket,
            "object_name": name,
            "generation": generation,
            "sha256": sha256,
            "bytes": size,
        }

    def _get_metadata(
        self,
        object_name: str,
        *,
        generation: str | None = None,
        allow_absent: bool,
    ) -> dict[str, Any] | None:
        response = self._request(
            method="GET",
            url=self._metadata_url(object_name, generation=generation),
            allowed_statuses=(200, 404),
        )
        if response.status == 404:
            if allow_absent:
                return None
            raise RuntimeError("content GCS metadata readback returned status 404")
        return self._metadata_row(
            _strict_json_bytes(response.body, "content GCS metadata"),
            expected_name=object_name,
        )

    @staticmethod
    def _multipart_upload_body(
        *, object_name: str, payload: bytes, sha256: str
    ) -> tuple[bytes, str]:
        if not isinstance(object_name, str) or not object_name:
            raise ValueError("content multipart object name changed")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("content multipart payload changed")
        if _SHA.fullmatch(sha256) is None:
            raise ValueError("content multipart SHA-256 changed")
        boundary = f"ofc-f100-{sha256}"
        marker = f"--{boundary}".encode("ascii")
        if marker in payload:
            raise ValueError("content payload collides with deterministic multipart boundary")
        metadata = json.dumps(
            {"metadata": {"sha256": sha256}, "name": object_name},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        body = b"".join(
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
        )
        if len(body) - len(payload) > MAX_MULTIPART_OVERHEAD_BYTES:
            raise ValueError("content multipart overhead escaped the fixed bound")
        return body, boundary

    def list_prefix(self, *, bucket: str, prefix: str) -> Mapping[str, Any]:
        self._check_prefix(bucket=bucket, prefix=prefix)
        objects: list[dict[str, Any]] = []
        names: set[str] = set()
        tokens: set[str] = set()
        page_token: str | None = None
        for _ in range(MAX_LIST_PAGES):
            query = {
                "prefix": f"{self._prefix}/",
                "maxResults": "1000",
                "fields": "items(name,generation,size,metadata),nextPageToken",
            }
            if page_token is not None:
                query["pageToken"] = page_token
            url = (
                f"https://storage.googleapis.com/storage/v1/b/{_quoted(self._bucket)}/o?"
                f"{urllib.parse.urlencode(query)}"
            )
            response = self._request(method="GET", url=url)
            value = _strict_json_bytes(response.body, "content GCS list")
            if not set(value).issubset({"items", "nextPageToken"}):
                raise RuntimeError("content GCS list response fields changed")
            items = value.get("items", [])
            if not isinstance(items, list):
                raise RuntimeError("content GCS list items changed")
            for raw in items:
                row = self._metadata_row(raw)
                if row["object_name"] in names:
                    raise RuntimeError("content GCS list returned a duplicate object")
                names.add(row["object_name"])
                objects.append(row)
            token = value.get("nextPageToken")
            if token is None:
                objects.sort(key=lambda row: row["object_name"])
                return {
                    "bucket": self._bucket,
                    "prefix": self._prefix,
                    "complete": True,
                    "objects": objects,
                }
            if not isinstance(token, str) or not token or token in tokens:
                raise RuntimeError("content GCS list pagination token changed or repeated")
            tokens.add(token)
            page_token = token
        raise RuntimeError("content GCS list exceeded the bounded pagination limit")

    def create_object_from_file(
        self,
        *,
        bucket: str,
        object_name: str,
        source_path: str,
        sha256: str,
        bytes: int,
        if_generation_match: int,
    ) -> Mapping[str, Any]:
        if self.mode != "stage":
            raise PermissionError("content creation requires stage mode")
        self._check_scope(bucket=bucket, object_name=object_name)
        frozen = self._entries[object_name]
        if (
            sha256 != frozen["sha256"]
            or bytes != frozen["bytes"]
            or if_generation_match != 0
        ):
            raise ValueError("content creation escaped the frozen object contract")
        source = Path(source_path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("content upload source is missing or unsafe")
        size = source.stat().st_size
        if size != bytes or size <= 0 or size > MAX_UPLOAD_BYTES:
            raise ValueError("content upload source size changed")
        payload = source.read_bytes()
        if len(payload) != bytes or hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("content upload source hash changed")
        multipart_body, boundary = self._multipart_upload_body(
            object_name=object_name, payload=payload, sha256=sha256
        )
        query = {
            "uploadType": "multipart",
            "ifGenerationMatch": "0",
            "fields": "name,generation,size,metadata",
        }
        url = (
            f"https://storage.googleapis.com/upload/storage/v1/b/"
            f"{_quoted(self._bucket)}/o?{urllib.parse.urlencode(query)}"
        )
        try:
            response = self._request(
                method="POST",
                url=url,
                body=multipart_body,
                content_type=f"multipart/related; boundary={boundary}",
                allowed_statuses=(200, 201, 409, 412),
            )
        except ContentTransportError:
            recovered: dict[str, Any] | None = None
            for _ in range(UPLOAD_RECOVERY_READ_ATTEMPTS):
                recovered = self._get_metadata(object_name, allow_absent=True)
                if recovered is not None:
                    break
            if recovered is None:
                raise
            return {
                **recovered,
                "created": True,
                "if_generation_match": 0,
            }
        if response.status in {409, 412}:
            # Existing objects are never read or adopted as upload success.
            raise FileExistsError(
                "content GCS existing object is never adopted after create-only rejection"
            )
        uploaded = self._metadata_row(
            _strict_json_bytes(response.body, "content GCS multipart upload"),
            expected_name=object_name,
        )
        if uploaded["generation"] == "0":
            raise RuntimeError("content GCS upload returned generation zero")
        readback = self._get_metadata(
            object_name,
            generation=uploaded["generation"],
            allow_absent=False,
        )
        assert readback is not None
        if readback != uploaded:
            raise RuntimeError("content GCS upload metadata readback changed")
        return {
            **readback,
            "created": True,
            "if_generation_match": 0,
        }

    def delete_object(
        self,
        *,
        bucket: str,
        object_name: str,
        if_generation_match: str,
    ) -> Mapping[str, Any]:
        if self.mode != "cleanup":
            raise PermissionError("content deletion requires cleanup mode")
        self._check_scope(bucket=bucket, object_name=object_name)
        generation = _positive_decimal(
            if_generation_match, "content delete generation"
        )
        before = self._get_metadata(
            object_name, generation=generation, allow_absent=False
        )
        assert before is not None
        if before["generation"] != generation:
            raise RuntimeError("content delete generation readback drifted")
        query = urllib.parse.urlencode({"ifGenerationMatch": generation})
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{_quoted(self._bucket)}/o/"
            f"{_quoted(object_name)}?{query}"
        )
        response = self._request(
            method="DELETE",
            url=url,
            allowed_statuses=(204, 404, 409, 412),
        )
        if response.status in {409, 412}:
            raise RuntimeError("content GCS exact-generation delete precondition failed")
        if response.status == 404:
            raise RuntimeError("content GCS owned generation disappeared before delete")
        try:
            remaining = self._get_metadata(object_name, allow_absent=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "content GCS exact absence confirmation failed"
            ) from exc
        if remaining is not None:
            raise RuntimeError("content GCS exact absence confirmation failed")
        return {
            "bucket": self._bucket,
            "object_name": object_name,
            "generation": generation,
            "deleted": True,
            "if_generation_match": generation,
        }


# Concise alias used by orchestration code.
ContentGcpAdapter = GcsContentObjectAdapter


__all__ = [
    "BUCKET",
    "ContentGcpAdapter",
    "ContentTransportError",
    "GcsContentObjectAdapter",
    "HTTP_TIMEOUT_SECONDS",
    "MAX_LIST_PAGES",
    "MAX_RESPONSE_BYTES",
    "MAX_UPLOAD_BYTES",
    "MODES",
    "PROJECT",
    "UPLOAD_RECOVERY_READ_ATTEMPTS",
    "TOKEN_ENV",
]
