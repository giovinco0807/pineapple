"""Official GCS JSON adapter for the full100 wave-v2 result receiver.

``read`` mode exposes only generation-pinned result GET/list operations.
``accept`` mode adds one mutation: create the exact selected job's
``ACCEPTED.json`` with ``ifGenerationMatch=0`` and then read it back exactly.
There is no GCE, IAM, delete, overwrite, or arbitrary-prefix surface.

Bearer credentials are obtained from ``GOOGLE_OAUTH_ACCESS_TOKEN`` for every
HTTP request.  Tokens and provider response bodies are never returned or
interpolated into errors/receipts.  HTTP is injectable for offline tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as receiver_v2
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
    _stdlib_http_request,
)


POLL_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gcs_result_poll_v2"
ACCEPT_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gcs_accept_v2"
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
BUCKET = worker_iam_v2.BUCKET
MODES = frozenset({"read", "accept"})
MAX_LIST_PAGES = 100
MAX_OBJECT_BYTES = 64 * 1024 * 1024

_SHA = re.compile(r"^[0-9a-f]{64}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]$")
_ATTEMPT = re.compile(r"^a0[01]$")
_HAND_PATH = re.compile(r"^hand_([0-9]{3})\.json$")

_OBJECT_RECORD_KEYS = frozenset({"path", "generation", "bytes", "sha256"})
_POLL_KEYS = frozenset(
    {
        "schema",
        "status",
        "mode",
        "bucket",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "selected_job_prefixes",
        "records",
        "record_count",
        "records_sha256",
        "list_page_count",
        "http_get_count",
        "http_post_count",
        "pagination_complete",
        "generation_pinned_media_read",
        "read_only",
        "result_store_protocol_only",
        "vm_lifecycle_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_ACCEPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "mode",
        "bucket",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "job_id",
        "source_role",
        "attempt_id",
        "path",
        "payload_bytes",
        "payload_sha256",
        "record",
        "create_only_precondition_generation",
        "provider_create_confirmed",
        "transport_ambiguity_reconciled",
        "exact_generation_readback",
        "exact_payload_readback",
        "http_get_count",
        "http_post_count",
        "accept_mutation_mode",
        "overwrite_authorized",
        "delete_authorized",
        "vm_lifecycle_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


class _HttpAmbiguity(RuntimeError):
    pass


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(value))
    return {**body, "receipt_sha256": wave_v2.canonical_sha256(body)}


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _safe_object_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("GCS result object name changed")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("GCS result object name changed")
    return path.as_posix()


def _positive_int_string(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.isdigit() or int(value) <= 0:
        raise RuntimeError(f"GCS {label} changed")
    return int(value)


def _json_response(response: HttpResponse, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"GCS {label} response was not JSON") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"GCS {label} response was not an object")
    return value


class GcsResultStoreV2:
    """Mode-separated, exact-prefix implementation of receiver ``ResultStore``."""

    def __init__(
        self,
        *,
        mode: str,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        resume_plan: Mapping[str, Any],
        bucket: str = BUCKET,
        requester: HttpRequester = _stdlib_http_request,
        timeout_seconds: int = 60,
    ) -> None:
        if mode not in MODES:
            raise ValueError("GCS result-store mode changed")
        if (
            not isinstance(bucket, str)
            or _BUCKET.fullmatch(bucket) is None
            or bucket != BUCKET
        ):
            raise ValueError("GCS result-store bucket escaped fixed scope")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 600
        ):
            raise ValueError("GCS result-store timeout escaped fixed range")
        self.mode = mode
        self.plan = wave_v2.validate_wave_plan(wave_plan)
        self.ledger = wave_v2.validate_attempt_ledger(
            self.plan, attempt_ledger
        )
        self.resume = wave_v2.validate_resume_plan(
            self.plan, self.ledger, resume_plan
        )
        if self.resume["all_jobs_complete"] is True:
            raise ValueError("GCS result store requires one selected wave")
        self.bucket = bucket
        self._requester = requester
        self._timeout_seconds = timeout_seconds
        self._cache: dict[tuple[str, int], bytes] = {}
        self._http_get_count = 0
        self._http_post_count = 0
        self._list_page_count = 0
        self._last_create_confirmed = False
        self._last_create_reconciled = False

        metadata = {
            row["job_id"]: row for row in self.plan["full100_plan"]["jobs"]
        }
        self._jobs = metadata
        template = self.plan["artifact_contract"][
            "attempt_path_template"
        ]
        self._job_prefixes = {
            job: template.format(job_id=job, attempt_id="a00").split(
                "/attempts/", 1
            )[0]
            + "/"
            for job in self.plan["coverage"]["job_ids"]
        }
        self._selected = {
            row["job_id"]: deepcopy(row)
            for row in self.resume["selected_attempts"]
        }
        self._selected_prefixes = [
            self._job_prefixes[row["job_id"]]
            for row in self.resume["selected_attempts"]
        ]

    def _token_headers(
        self, *, content_type: str | None = None
    ) -> dict[str, str]:
        token = os.environ.get(TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            character.isspace() for character in token
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
        return headers

    def _request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> HttpResponse:
        allowed = {"GET"} if self.mode == "read" else {"GET", "POST"}
        if method not in allowed:
            raise PermissionError("HTTP method escaped GCS result-store mode")
        if method == "GET":
            self._http_get_count += 1
        else:
            self._http_post_count += 1
        # Credential validation is a local precondition failure, not an
        # ambiguous provider mutation outcome.  Keep it outside the transport
        # exception boundary so a missing token cannot enter reconciliation.
        headers = self._token_headers(content_type=content_type)
        try:
            response = self._requester(
                method,
                url,
                headers,
                body,
                self._timeout_seconds,
            )
        except Exception:
            raise _HttpAmbiguity(
                f"GCS {method} transport outcome was ambiguous"
            ) from None
        if (
            not isinstance(response, HttpResponse)
            or type(response.status) is not int
            or not isinstance(response.body, bytes)
            or not isinstance(response.headers, Mapping)
        ):
            raise _HttpAmbiguity(
                f"GCS {method} transport response shape was ambiguous"
            )
        return response

    def _get(
        self, *, url: str, allowed_statuses: Sequence[int] = (200,)
    ) -> HttpResponse:
        try:
            response = self._request(method="GET", url=url)
        except _HttpAmbiguity:
            raise RuntimeError("GCS GET failed without trusted readback") from None
        if response.status not in allowed_statuses:
            raise RuntimeError(f"GCS GET failed with status {response.status}")
        return response

    def _bucket_base(self) -> str:
        return (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}"
        )

    def _metadata_url(self, path: str) -> str:
        query = urllib.parse.urlencode({"fields": "name,generation,size"})
        return (
            f"{self._bucket_base()}/o/"
            f"{urllib.parse.quote(path, safe='')}?{query}"
        )

    def _media_url(self, path: str, generation: int) -> str:
        query = urllib.parse.urlencode(
            {"alt": "media", "ifGenerationMatch": str(generation)}
        )
        return (
            f"{self._bucket_base()}/o/"
            f"{urllib.parse.quote(path, safe='')}?{query}"
        )

    def _validate_result_path(self, path: str) -> tuple[str, str, str | None]:
        path = _safe_object_name(path)
        matches = [
            (job, prefix)
            for job, prefix in self._job_prefixes.items()
            if path.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError("GCS result path escaped exact planned jobs")
        job, prefix = matches[0]
        suffix = path[len(prefix) :]
        if suffix == "ACCEPTED.json":
            return job, "acceptance", None
        parts = suffix.split("/")
        if len(parts) < 3 or parts[0] != "attempts" or _ATTEMPT.fullmatch(parts[1]) is None:
            raise ValueError("GCS result path escaped exact attempt layout")
        attempt = parts[1]
        tail = parts[2:]
        if tail == ["DONE.json"]:
            return job, "done", attempt
        work = set(self._jobs[job]["work_hand_indices"])
        if len(tail) == 2 and tail[0] == "roots":
            match = _HAND_PATH.fullmatch(tail[1])
            if match is not None and int(match.group(1)) in work:
                return job, "root", attempt
        if (
            len(tail) == 3
            and tail[0] == "hands"
            and tail[1] == self._jobs[job]["source_role"]
        ):
            match = _HAND_PATH.fullmatch(tail[2])
            if match is not None and int(match.group(1)) in work:
                return job, "source_hand", attempt
        raise ValueError("GCS result path escaped exact artifact layout")

    def _validate_list_prefix(self, prefix: str) -> str:
        prefix = _safe_object_name(prefix.rstrip("/")) + "/"
        if prefix not in self._selected_prefixes:
            raise ValueError("GCS list prefix escaped selected job prefixes")
        return prefix

    @staticmethod
    def _metadata_record(value: Mapping[str, Any], *, expected_path: str | None = None) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError("GCS object metadata was not an object")
        row = dict(value)
        if set(row) != {"name", "generation", "size"}:
            raise RuntimeError("GCS object metadata fields changed")
        path = _safe_object_name(row["name"])
        if expected_path is not None and path != expected_path:
            raise RuntimeError("GCS object metadata path changed")
        generation = _positive_int_string(row["generation"], "generation")
        size = _positive_int_string(row["size"], "object size")
        if size > MAX_OBJECT_BYTES:
            raise RuntimeError("GCS result object exceeded fixed size limit")
        return {"path": path, "generation": generation, "bytes": size}

    def read_bytes(self, *, path: str, generation: int) -> bytes:
        self._validate_result_path(path)
        if type(generation) is not int or generation <= 0:
            raise ValueError("GCS generation changed")
        cached = self._cache.get((path, generation))
        if cached is not None:
            return cached
        response = self._get(url=self._media_url(path, generation))
        if not response.body or len(response.body) > MAX_OBJECT_BYTES:
            raise RuntimeError("GCS generation-pinned media bytes changed")
        self._cache[(path, generation)] = response.body
        return response.body

    def read_current(
        self, *, path: str, allow_missing: bool = False
    ) -> tuple[Mapping[str, Any], bytes] | None:
        self._validate_result_path(path)
        response = self._get(
            url=self._metadata_url(path), allowed_statuses=(200, 404)
        )
        if response.status == 404:
            if allow_missing:
                return None
            raise ValueError("required GCS result object is missing")
        metadata = self._metadata_record(
            _json_response(response, "object metadata"), expected_path=path
        )
        raw = self.read_bytes(path=path, generation=metadata["generation"])
        if len(raw) != metadata["bytes"]:
            raise RuntimeError("GCS metadata/media size changed")
        record = {
            **metadata,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return record, raw

    def list_prefix(self, *, prefix: str) -> Sequence[Mapping[str, Any]]:
        prefix = self._validate_list_prefix(prefix)
        records: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        page_token: str | None = None
        seen_tokens: set[str] = set()
        for _ in range(MAX_LIST_PAGES):
            parameters = {
                "prefix": prefix,
                "fields": "items(name,generation,size),nextPageToken",
            }
            if page_token is not None:
                parameters["pageToken"] = page_token
            url = f"{self._bucket_base()}/o?{urllib.parse.urlencode(parameters)}"
            response = self._get(url=url)
            value = _json_response(response, "object list")
            if set(value) - {"items", "nextPageToken"}:
                raise RuntimeError("GCS object-list fields changed")
            items = value.get("items", [])
            if not isinstance(items, list):
                raise RuntimeError("GCS object-list items changed")
            self._list_page_count += 1
            for item in items:
                metadata = self._metadata_record(item)
                path = metadata["path"]
                self._validate_result_path(path)
                if not path.startswith(prefix) or path in seen_paths:
                    raise RuntimeError("GCS object-list duplicate or prefix changed")
                seen_paths.add(path)
                raw = self.read_bytes(
                    path=path, generation=metadata["generation"]
                )
                if len(raw) != metadata["bytes"]:
                    raise RuntimeError("GCS listed metadata/media size changed")
                records.append(
                    {**metadata, "sha256": hashlib.sha256(raw).hexdigest()}
                )
            token = value.get("nextPageToken")
            if token is None:
                return records
            if (
                not isinstance(token, str)
                or not token
                or len(token) > 4096
                or token in seen_tokens
            ):
                raise RuntimeError("GCS object-list page token changed")
            seen_tokens.add(token)
            page_token = token
        raise RuntimeError("GCS object-list exceeded fixed pagination bound")

    def _reconcile_acceptance(self, *, path: str, data: bytes) -> dict[str, Any]:
        try:
            current = self.read_current(path=path, allow_missing=True)
        except Exception:
            raise RuntimeError(
                "GCS ACCEPTED create outcome is ambiguous and exact readback failed"
            ) from None
        if current is None or current[1] != data:
            raise RuntimeError(
                "GCS ACCEPTED create outcome is ambiguous and exact payload was not observed"
            )
        self._last_create_confirmed = False
        self._last_create_reconciled = True
        return dict(current[0])

    def create_only(self, *, path: str, data: bytes) -> Mapping[str, Any]:
        if self.mode != "accept":
            raise PermissionError("ACCEPTED creation requires accept mode")
        job, kind, _ = self._validate_result_path(path)
        if kind != "acceptance" or job not in self._selected:
            raise ValueError("ACCEPTED create path escaped selected jobs")
        receiver_v2.validate_acceptance_payload_context(
            self.plan,
            self.ledger,
            self.resume,
            path=path,
            payload=data,
        )
        parameters = {
            "fields": "name,generation,size",
            "uploadType": "media",
            "ifGenerationMatch": "0",
            "name": path,
        }
        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o?"
            f"{urllib.parse.urlencode(parameters)}"
        )
        try:
            response = self._request(
                method="POST",
                url=url,
                body=data,
                content_type="application/json",
            )
        except _HttpAmbiguity:
            return self._reconcile_acceptance(path=path, data=data)
        if response.status == 412:
            return self._reconcile_acceptance(path=path, data=data)
        if response.status != 200:
            # A concrete provider response is not a transport-ambiguity case.
            # In particular, never reinterpret an explicit 5xx body as a
            # successful create even if an object happens to be readable.
            raise RuntimeError(
                f"GCS ACCEPTED create failed with status {response.status}"
            )
        created = self._metadata_record(
            _json_response(response, "ACCEPTED create"), expected_path=path
        )
        try:
            current = self.read_current(path=path, allow_missing=False)
        except Exception:
            raise RuntimeError(
                "GCS ACCEPTED create succeeded but exact readback failed"
            ) from None
        assert current is not None
        record, observed = current
        if (
            observed != data
            or record["generation"] != created["generation"]
            or record["bytes"] != created["bytes"]
        ):
            raise RuntimeError("GCS ACCEPTED create/readback identity changed")
        self._last_create_confirmed = True
        self._last_create_reconciled = False
        return dict(record)

    def poll_selected(self) -> dict[str, Any]:
        if self.mode != "read":
            raise PermissionError("selected result poll requires read mode")
        start_get = self._http_get_count
        start_post = self._http_post_count
        start_pages = self._list_page_count
        records: list[dict[str, Any]] = []
        for prefix in self._selected_prefixes:
            records.extend(self.list_prefix(prefix=prefix))
        records = sorted(records, key=lambda row: row["path"])
        if len({row["path"] for row in records}) != len(records):
            raise RuntimeError("GCS selected result poll contains duplicate paths")
        body = {
            "schema": POLL_RECEIPT_SCHEMA,
            "status": "selected_result_prefixes_generation_pinned",
            "mode": "read",
            "bucket": self.bucket,
            "run_name": self.plan["run_name"],
            "execution_identity_sha256": self.plan[
                "execution_identity_sha256"
            ],
            "wave_plan_sha256": self.plan["schedule_sha256"],
            "attempt_ledger_sha256": self.ledger["ledger_sha256"],
            "resume_plan_sha256": self.resume["resume_sha256"],
            "wave_index": self.resume["resume_wave_index"],
            "selected_job_prefixes": list(self._selected_prefixes),
            "records": records,
            "record_count": len(records),
            "records_sha256": wave_v2.canonical_sha256(records),
            "list_page_count": self._list_page_count - start_pages,
            "http_get_count": self._http_get_count - start_get,
            "http_post_count": self._http_post_count - start_post,
            "pagination_complete": True,
            "generation_pinned_media_read": True,
            "read_only": True,
            "result_store_protocol_only": True,
            "vm_lifecycle_mutation_performed": False,
            "current_profile_changed": False,
        }
        return validate_poll_receipt(
            self.plan, self.ledger, self.resume, _seal(body)
        )

    def accept_and_readback(self, *, path: str, data: bytes) -> dict[str, Any]:
        if self.mode != "accept":
            raise PermissionError("ACCEPTED mutation receipt requires accept mode")
        value = receiver_v2.validate_acceptance_payload_context(
            self.plan,
            self.ledger,
            self.resume,
            path=path,
            payload=data,
        )
        start_get = self._http_get_count
        start_post = self._http_post_count
        record = dict(self.create_only(path=path, data=data))
        current = self.read_current(path=path, allow_missing=False)
        assert current is not None
        if current[0] != record or current[1] != data:
            raise RuntimeError("GCS ACCEPTED final readback changed")
        body = {
            "schema": ACCEPT_RECEIPT_SCHEMA,
            "status": "selected_acceptance_create_only_exact_readback",
            "mode": "accept",
            "bucket": self.bucket,
            "run_name": self.plan["run_name"],
            "execution_identity_sha256": self.plan[
                "execution_identity_sha256"
            ],
            "wave_plan_sha256": self.plan["schedule_sha256"],
            "attempt_ledger_sha256": self.ledger["ledger_sha256"],
            "resume_plan_sha256": self.resume["resume_sha256"],
            "wave_index": self.resume["resume_wave_index"],
            "job_id": value["job_id"],
            "source_role": value["source_role"],
            "attempt_id": value["attempt_id"],
            "path": path,
            "payload_bytes": len(data),
            "payload_sha256": hashlib.sha256(data).hexdigest(),
            "record": record,
            "create_only_precondition_generation": 0,
            "provider_create_confirmed": self._last_create_confirmed,
            "transport_ambiguity_reconciled": self._last_create_reconciled,
            "exact_generation_readback": True,
            "exact_payload_readback": True,
            "http_get_count": self._http_get_count - start_get,
            "http_post_count": self._http_post_count - start_post,
            "accept_mutation_mode": True,
            "overwrite_authorized": False,
            "delete_authorized": False,
            "vm_lifecycle_mutation_performed": False,
            "current_profile_changed": False,
        }
        return validate_accept_receipt(
            self.plan, self.ledger, self.resume, _seal(body)
        )


def _validate_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("GCS receipt object record is not an object")
    row = deepcopy(dict(value))
    _exact_keys(row, _OBJECT_RECORD_KEYS, "GCS receipt object record")
    _safe_object_name(row["path"])
    if (
        type(row["generation"]) is not int
        or row["generation"] <= 0
        or type(row["bytes"]) is not int
        or row["bytes"] <= 0
        or not isinstance(row["sha256"], str)
        or _SHA.fullmatch(row["sha256"]) is None
    ):
        raise ValueError("GCS receipt object metadata changed")
    return row


def _validate_receipt_context(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    if (
        payload.get("bucket") != BUCKET
        or payload.get("run_name") != plan["run_name"]
        or payload.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or payload.get("wave_plan_sha256") != plan["schedule_sha256"]
        or payload.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or payload.get("resume_plan_sha256") != resume["resume_sha256"]
        or payload.get("wave_index") != resume["resume_wave_index"]
        or payload.get("vm_lifecycle_mutation_performed") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("GCS result receipt context changed")


def validate_poll_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if not isinstance(value, Mapping):
        raise ValueError("GCS poll receipt is not an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _POLL_KEYS, "GCS poll receipt")
    digest = payload.pop("receipt_sha256", None)
    if digest != wave_v2.canonical_sha256(payload):
        raise ValueError("GCS poll receipt digest changed")
    payload["receipt_sha256"] = digest
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("GCS poll receipt records are missing")
    checked = [_validate_record(row) for row in records]
    prefixes = [
        plan["artifact_contract"]["attempt_path_template"]
        .format(job_id=row["job_id"], attempt_id="a00")
        .split("/attempts/", 1)[0]
        + "/"
        for row in resume["selected_attempts"]
    ]
    paths = [row["path"] for row in checked]
    _validate_receipt_context(plan, ledger, resume, payload)
    if (
        payload["schema"] != POLL_RECEIPT_SCHEMA
        or payload["status"]
        != "selected_result_prefixes_generation_pinned"
        or payload["mode"] != "read"
        or payload["selected_job_prefixes"] != prefixes
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or any(
            not any(path.startswith(prefix) for prefix in prefixes)
            for path in paths
        )
        or payload["record_count"] != len(checked)
        or payload["records_sha256"] != wave_v2.canonical_sha256(checked)
        or type(payload["list_page_count"]) is not int
        or payload["list_page_count"] < len(prefixes)
        or payload["list_page_count"] > len(prefixes) * MAX_LIST_PAGES
        or type(payload["http_get_count"]) is not int
        or payload["http_get_count"] < payload["list_page_count"]
        or payload["http_post_count"] != 0
        or payload["pagination_complete"] is not True
        or payload["generation_pinned_media_read"] is not True
        or payload["read_only"] is not True
        or payload["result_store_protocol_only"] is not True
    ):
        raise ValueError("GCS poll receipt semantics changed")
    return payload


def validate_accept_receipt(
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if not isinstance(value, Mapping):
        raise ValueError("GCS ACCEPTED receipt is not an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _ACCEPT_KEYS, "GCS ACCEPTED receipt")
    digest = payload.pop("receipt_sha256", None)
    if digest != wave_v2.canonical_sha256(payload):
        raise ValueError("GCS ACCEPTED receipt digest changed")
    payload["receipt_sha256"] = digest
    record = _validate_record(payload["record"])
    selected = [
        row
        for row in resume["selected_attempts"]
        if row["job_id"] == payload.get("job_id")
    ]
    _validate_receipt_context(plan, ledger, resume, payload)
    expected_path = (
        None
        if len(selected) != 1
        else plan["artifact_contract"]["job_acceptance_path_template"].format(
            job_id=selected[0]["job_id"]
        )
    )
    if (
        len(selected) != 1
        or payload["schema"] != ACCEPT_RECEIPT_SCHEMA
        or payload["status"]
        != "selected_acceptance_create_only_exact_readback"
        or payload["mode"] != "accept"
        or payload["source_role"] != selected[0]["source_role"]
        or payload["attempt_id"] != selected[0]["attempt_id"]
        or payload["path"] != expected_path
        or payload["path"] != record["path"]
        or payload["payload_bytes"] != record["bytes"]
        or payload["payload_sha256"] != record["sha256"]
        or payload["create_only_precondition_generation"] != 0
        or type(payload["provider_create_confirmed"]) is not bool
        or type(payload["transport_ambiguity_reconciled"]) is not bool
        or payload["provider_create_confirmed"]
        is payload["transport_ambiguity_reconciled"]
        or payload["exact_generation_readback"] is not True
        or payload["exact_payload_readback"] is not True
        or type(payload["http_get_count"]) is not int
        or payload["http_get_count"] < 2
        or payload["http_post_count"] != 1
        or payload["accept_mutation_mode"] is not True
        or payload["overwrite_authorized"] is not False
        or payload["delete_authorized"] is not False
    ):
        raise ValueError("GCS ACCEPTED receipt semantics changed")
    return payload


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{label} JSON could not be loaded") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON is not an object")
    return value


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wave-plan", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--resume-plan", type=Path, required=True)
    parser.add_argument("--bucket", default=BUCKET)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    poll = commands.add_parser("poll", help="GET-only selected-prefix poll")
    _add_context_arguments(poll)
    accept = commands.add_parser(
        "accept", help="create one exact selected ACCEPTED object"
    )
    _add_context_arguments(accept)
    accept.add_argument("--object-path", required=True)
    accept.add_argument("--payload", type=Path, required=True)
    accept.add_argument("--authorize-accept-create", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _load_json(args.wave_plan, "wave plan")
    ledger = _load_json(args.attempt_ledger, "attempt ledger")
    resume = _load_json(args.resume_plan, "resume plan")
    if args.command == "poll":
        store = GcsResultStoreV2(
            mode="read",
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            bucket=args.bucket,
        )
        receipt = store.poll_selected()
    else:
        if args.authorize_accept_create is not True:
            raise PermissionError(
                "accept CLI requires explicit --authorize-accept-create"
            )
        try:
            payload = args.payload.read_bytes()
        except OSError:
            raise ValueError("ACCEPTED payload could not be loaded") from None
        store = GcsResultStoreV2(
            mode="accept",
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            bucket=args.bucket,
        )
        receipt = store.accept_and_readback(
            path=args.object_path, data=payload
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACCEPT_RECEIPT_SCHEMA",
    "BUCKET",
    "GcsResultStoreV2",
    "MAX_LIST_PAGES",
    "MODES",
    "POLL_RECEIPT_SCHEMA",
    "TOKEN_ENV",
    "main",
    "validate_accept_receipt",
    "validate_poll_receipt",
]
