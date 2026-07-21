"""Narrow live-cloud adapters for the Step12b exact pair.

The pure Step12b contracts intentionally do not know how credentials or HTTP
are obtained.  This module supplies the smallest live implementations needed
by the one candidate + one reference attempt-0 run:

* a redacted user-token source backed by ``gcloud auth print-access-token``;
* a sanitized IAM Credentials token generator;
* a generation-pinned, three-object GCS bootstrap store; and
* exact-name Compute clients with one-shot mutations and operation readback.

Every adapter is bound at construction time to the fresh Step12b identities.
It has no broad list/delete surface and never returns an access token in a
receipt or exception.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_cloud_controller_v2
    as pair_controller,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_live_cloud_adapters_v2"
)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60
OPERATION_WAIT_SECONDS = 600
ABSENCE_WAIT_SECONDS = 600
POLL_SECONDS = 2.0

_SAFE_NAME = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_TOKEN = re.compile(r"^[^\s]{20,16384}$")
_GENERATION = re.compile(r"^[1-9][0-9]{0,31}$")
_OPERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ALLOWED_HOSTS = frozenset(
    {
        "cloudresourcemanager.googleapis.com",
        "compute.googleapis.com",
        "iam.googleapis.com",
        "iamcredentials.googleapis.com",
        "serviceusage.googleapis.com",
        "storage.googleapis.com",
    }
)


class AccessTokenSource(Protocol):
    def access_token(self) -> str: ...


class HttpsClient(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> rest_iam.HttpResponse: ...


class LiveCloudAdapterError(RuntimeError):
    """Secret-free, stable live-adapter failure."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.operation = operation

    def __str__(self) -> str:
        return self.code


def _checked_token(value: Any) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise LiveCloudAdapterError("access_token_shape_changed")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_object(
    response: rest_iam.HttpResponse,
    *,
    operation: str,
) -> dict[str, Any]:
    if (
        not isinstance(response, rest_iam.HttpResponse)
        or not isinstance(response.body, bytes)
        or len(response.body) > MAX_RESPONSE_BYTES
    ):
        raise LiveCloudAdapterError(
            "cloud_response_shape_changed", operation=operation
        )
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LiveCloudAdapterError(
            "cloud_json_response_changed",
            status_code=response.status_code,
            operation=operation,
        ) from None
    if not isinstance(value, dict):
        raise LiveCloudAdapterError(
            "cloud_json_response_changed",
            status_code=response.status_code,
            operation=operation,
        )
    return value


class GcloudUserAccessTokenSource:
    """Obtain the active user's token without exposing it in ``repr``."""

    __slots__ = ("_gcloud", "_project", "_runner")

    def __init__(
        self,
        *,
        project: str = payload_transport.PROJECT,
        gcloud_executable: str = "gcloud.cmd",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if (
            project != payload_transport.PROJECT
            or not isinstance(gcloud_executable, str)
            or not gcloud_executable
            or not callable(runner)
        ):
            raise ValueError("gcloud user-token source identity changed")
        self._gcloud = gcloud_executable
        self._project = project
        self._runner = runner

    def access_token(self) -> str:
        try:
            completed = self._runner(
                [
                    self._gcloud,
                    "auth",
                    "print-access-token",
                    f"--project={self._project}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except Exception:
            raise LiveCloudAdapterError(
                "gcloud_user_access_token_unavailable"
            ) from None
        if (
            not isinstance(completed, subprocess.CompletedProcess)
            or completed.returncode != 0
            or not isinstance(completed.stdout, str)
        ):
            raise LiveCloudAdapterError(
                "gcloud_user_access_token_unavailable"
            )
        return _checked_token(completed.stdout.strip())

    def __repr__(self) -> str:
        return (
            "<GcloudUserAccessTokenSource "
            f"project={self._project!r} token=redacted>"
        )


class StdlibCloudHttpsClient:
    """HTTPS transport with redirects and ambient proxies disabled."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> rest_iam.HttpResponse:
        parsed = urllib.parse.urlsplit(url)
        if (
            method not in {"GET", "POST", "PUT", "DELETE"}
            or parsed.scheme != "https"
            or parsed.hostname not in _ALLOWED_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= 120
        ):
            raise ValueError("live cloud request escaped HTTPS allowlist")
        request = urllib.request.Request(
            url=url,
            headers=dict(headers),
            data=body,
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                return rest_iam.HttpResponse(
                    status_code=int(response.status),
                    body=raw,
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as error:
            raw = error.read(MAX_RESPONSE_BYTES + 1)
            return rest_iam.HttpResponse(
                status_code=int(error.code),
                body=raw,
                headers=(
                    dict(error.headers.items()) if error.headers else {}
                ),
            )
        except (urllib.error.URLError, TimeoutError, OSError):
            raise LiveCloudAdapterError(
                "live_cloud_https_transport_failed"
            ) from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class IamCredentialsControllerTokenGenerator:
    """Mint one token and translate HTTP failures for the bounded barrier."""

    __slots__ = ("controller_service_account", "_http", "_tokens")

    def __init__(
        self,
        *,
        controller_service_account: str,
        http_client: HttpsClient,
        user_token_source: AccessTokenSource,
    ) -> None:
        if (
            not isinstance(controller_service_account, str)
            or not controller_service_account.endswith(
                f"@{payload_transport.PROJECT}.iam.gserviceaccount.com"
            )
        ):
            raise ValueError("controller service account changed")
        self.controller_service_account = controller_service_account
        self._http = http_client
        self._tokens = user_token_source

    def generate_controller_access_token(
        self, *, lifetime_seconds: int
    ) -> rest_iam.GeneratedControllerAccessToken:
        if lifetime_seconds != token_barrier.TOKEN_LIFETIME_SECONDS:
            raise ValueError("controller token lifetime changed")
        email = urllib.parse.quote(
            self.controller_service_account, safe=""
        )
        response = self._http.request(
            method="POST",
            url=(
                "https://iamcredentials.googleapis.com/v1/projects/-/"
                f"serviceAccounts/{email}:generateAccessToken"
            ),
            headers={
                "Authorization": (
                    f"Bearer {_checked_token(self._tokens.access_token())}"
                ),
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body=_canonical_bytes(
                {
                    "scope": [rest_iam.GOOGLE_CLOUD_PLATFORM_SCOPE],
                    "lifetime": f"{lifetime_seconds}s",
                }
            ),
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            error_code: int | None = None
            error_status: str | None = None
            error_reason: str | None = None
            try:
                payload = json.loads(response.body.decode("utf-8"))
                error = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error, dict):
                    code = error.get("code")
                    status = error.get("status")
                    if type(code) is int:
                        error_code = code
                    if (
                        isinstance(status, str)
                        and re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", status)
                    ):
                        error_status = status
                    details = error.get("details")
                    if isinstance(details, list):
                        for detail in details:
                            reason = (
                                detail.get("reason")
                                if isinstance(detail, dict)
                                else None
                            )
                            if (
                                isinstance(reason, str)
                                and re.fullmatch(
                                    r"[A-Z][A-Z0-9_]{0,127}", reason
                                )
                            ):
                                error_reason = reason
                                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            raise token_barrier.TokenGenerationHttpError(
                status_code=response.status_code,
                response_body=response.body,
                error_code=error_code,
                error_status=error_status,
                error_reason=error_reason,
            )
        payload = _json_object(
            response, operation="generate_controller_access_token"
        )
        if set(payload) != {"accessToken", "expireTime"}:
            raise LiveCloudAdapterError(
                "controller_access_token_response_changed"
            )
        return rest_iam.GeneratedControllerAccessToken(
            payload["accessToken"], payload["expireTime"]
        )


def _gs_parts(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(uri)
    if (
        parsed.scheme != "gs"
        or parsed.netloc != payload_transport.BUCKET
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("bootstrap source GCS URI changed")
    name = urllib.parse.unquote(parsed.path[1:])
    if not name or name.startswith("/") or ".." in name.split("/"):
        raise ValueError("bootstrap source object name changed")
    return parsed.netloc, name


class GenerationPinnedBootstrapSourceStore:
    """Exact three-object GCS store bound to one source plan."""

    __slots__ = (
        "_http",
        "_tokens",
        "_prefix",
        "_prefix_name",
        "_objects",
    )

    def __init__(
        self,
        *,
        source_plan: Mapping[str, Any],
        http_client: HttpsClient,
        user_token_source: AccessTokenSource,
    ) -> None:
        prefix = source_plan.get("source_prefix")
        objects = source_plan.get("objects")
        if (
            not isinstance(prefix, str)
            or not isinstance(objects, list)
            or len(objects) != bootstrap_source.SOURCE_OBJECT_COUNT
        ):
            raise ValueError("bootstrap source plan surface changed")
        _, prefix_name = _gs_parts(prefix)
        uris = {
            row.get("uri")
            for row in objects
            if isinstance(row, Mapping)
        }
        if (
            len(uris) != bootstrap_source.SOURCE_OBJECT_COUNT
            or any(not isinstance(uri, str) for uri in uris)
            or any(
                not uri.startswith(prefix.rstrip("/") + "/")
                for uri in uris
            )
        ):
            raise ValueError("bootstrap source object allowlist changed")
        self._http = http_client
        self._tokens = user_token_source
        self._prefix = prefix
        self._prefix_name = prefix_name.rstrip("/") + "/"
        self._objects = frozenset(uris)

    def _headers(self, *, json_content: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": (
                f"Bearer {_checked_token(self._tokens.access_token())}"
            ),
            "Accept": "application/json",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def list_objects(self, *, prefix: str) -> list[str]:
        if prefix != self._prefix:
            raise ValueError("bootstrap source list prefix changed")
        token: str | None = None
        names: list[str] = []
        for _ in range(10):
            query = {
                "prefix": self._prefix_name,
                "fields": "items(name),nextPageToken",
            }
            if token is not None:
                query["pageToken"] = token
            url = (
                "https://storage.googleapis.com/storage/v1/b/"
                f"{urllib.parse.quote(payload_transport.BUCKET, safe='')}/o?"
                f"{urllib.parse.urlencode(query)}"
            )
            response = self._http.request(
                method="GET",
                url=url,
                headers=self._headers(),
                body=None,
                timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                raise LiveCloudAdapterError(
                    "bootstrap_source_list_failed",
                    status_code=response.status_code,
                )
            payload = _json_object(
                response, operation="bootstrap_source_list"
            )
            if not set(payload).issubset({"items", "nextPageToken"}):
                raise LiveCloudAdapterError(
                    "bootstrap_source_list_response_changed"
                )
            for row in payload.get("items", []):
                name = row.get("name") if isinstance(row, Mapping) else None
                if not isinstance(name, str):
                    raise LiveCloudAdapterError(
                        "bootstrap_source_list_response_changed"
                    )
                names.append(
                    f"gs://{payload_transport.BUCKET}/{name}"
                )
            next_token = payload.get("nextPageToken")
            if next_token is None:
                return names
            if (
                not isinstance(next_token, str)
                or not next_token
                or next_token == token
            ):
                raise LiveCloudAdapterError(
                    "bootstrap_source_list_pagination_changed"
                )
            token = next_token
        raise LiveCloudAdapterError(
            "bootstrap_source_list_page_bound_exhausted"
        )

    def conditional_create(
        self,
        *,
        uri: str,
        content: bytes,
        if_generation_match: int,
    ) -> Mapping[str, Any]:
        if (
            uri not in self._objects
            or if_generation_match != 0
            or not isinstance(content, bytes)
            or not content
        ):
            raise ValueError("bootstrap source conditional create changed")
        bucket, name = _gs_parts(uri)
        query = urllib.parse.urlencode(
            {
                "uploadType": "media",
                "ifGenerationMatch": "0",
                "name": name,
            }
        )
        response = self._http.request(
            method="POST",
            url=(
                "https://storage.googleapis.com/upload/storage/v1/b/"
                f"{urllib.parse.quote(bucket, safe='')}/o?{query}"
            ),
            headers=self._headers(json_content=True),
            body=content,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise LiveCloudAdapterError(
                "bootstrap_source_conditional_create_failed",
                status_code=response.status_code,
            )
        payload = _json_object(
            response, operation="bootstrap_source_conditional_create"
        )
        generation = payload.get("generation")
        if (
            payload.get("bucket") != bucket
            or payload.get("name") != name
            or not isinstance(generation, str)
            or _GENERATION.fullmatch(generation) is None
            or str(payload.get("size")) != str(len(content))
        ):
            raise LiveCloudAdapterError(
                "bootstrap_source_create_readback_changed"
            )
        return {
            "uri": uri,
            "generation": int(generation),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "created": True,
        }

    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes:
        if (
            uri not in self._objects
            or type(generation) is not int
            or generation <= 0
        ):
            raise ValueError("bootstrap source generation read changed")
        bucket, name = _gs_parts(uri)
        query = urllib.parse.urlencode(
            {"alt": "media", "generation": str(generation)}
        )
        response = self._http.request(
            method="GET",
            url=(
                "https://storage.googleapis.com/download/storage/v1/b/"
                f"{urllib.parse.quote(bucket, safe='')}/o/"
                f"{urllib.parse.quote(name, safe='')}?{query}"
            ),
            headers={
                "Authorization": (
                    f"Bearer {_checked_token(self._tokens.access_token())}"
                )
            },
            body=None,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise LiveCloudAdapterError(
                "bootstrap_source_generation_read_failed",
                status_code=response.status_code,
            )
        if (
            not isinstance(response.body, bytes)
            or len(response.body) > bootstrap_source.MAX_SOURCE_OBJECT_BYTES
        ):
            raise LiveCloudAdapterError(
                "bootstrap_source_generation_read_changed"
            )
        return response.body


class GenerationPinnedResultStore:
    """Read-only receiver for two job and two heartbeat prefixes."""

    __slots__ = ("_http", "_tokens", "_prefixes", "_cache")

    def __init__(
        self,
        *,
        result_prefixes: Sequence[str],
        http_client: HttpsClient,
        user_token_source: AccessTokenSource,
    ) -> None:
        prefixes = tuple(prefix.rstrip("/") for prefix in result_prefixes)
        if (
            len(prefixes) != 4
            or len(set(prefixes)) != 4
            or any(_gs_parts(prefix)[0] != payload_transport.BUCKET for prefix in prefixes)
        ):
            raise ValueError("exact result/heartbeat-prefix allowlist changed")
        self._http = http_client
        self._tokens = user_token_source
        self._prefixes = prefixes
        self._cache: dict[tuple[str, int], bytes] = {}

    def _allowed(self, uri: str) -> bool:
        return any(
            uri == prefix or uri.startswith(prefix + "/")
            for prefix in self._prefixes
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": (
                f"Bearer {_checked_token(self._tokens.access_token())}"
            ),
            "Accept": "application/json",
        }

    def read_current(
        self, *, uri: str, allow_missing: bool = False
    ) -> tuple[dict[str, Any], bytes] | None:
        if not self._allowed(uri):
            raise ValueError("result read escaped exact prefixes")
        bucket, name = _gs_parts(uri)
        fields = "bucket,name,generation,metageneration,size,crc32c,etag"
        metadata = self._http.request(
            method="GET",
            url=(
                "https://storage.googleapis.com/storage/v1/b/"
                f"{urllib.parse.quote(bucket, safe='')}/o/"
                f"{urllib.parse.quote(name, safe='')}?fields={fields}"
            ),
            headers=self._headers(),
            body=None,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        if allow_missing and metadata.status_code == 404:
            return None
        if metadata.status_code != 200:
            raise LiveCloudAdapterError(
                "result_metadata_read_failed",
                status_code=metadata.status_code,
            )
        value = _json_object(metadata, operation="result_metadata_read")
        generation = value.get("generation")
        metageneration = value.get("metageneration")
        size = value.get("size")
        if (
            value.get("bucket") != bucket
            or value.get("name") != name
            or not isinstance(generation, str)
            or _GENERATION.fullmatch(generation) is None
            or not isinstance(metageneration, str)
            or _GENERATION.fullmatch(metageneration) is None
            or not isinstance(size, str)
            or not size.isdigit()
            or int(size) <= 0
            or not isinstance(value.get("crc32c"), str)
            or not value["crc32c"]
            or not isinstance(value.get("etag"), str)
            or not value["etag"]
        ):
            raise LiveCloudAdapterError(
                "result_metadata_identity_changed"
            )
        numeric_generation = int(generation)
        raw_response = self._http.request(
            method="GET",
            url=(
                "https://storage.googleapis.com/download/storage/v1/b/"
                f"{urllib.parse.quote(bucket, safe='')}/o/"
                f"{urllib.parse.quote(name, safe='')}?"
                f"{urllib.parse.urlencode({'alt': 'media', 'generation': generation})}"
            ),
            headers={
                "Authorization": (
                    f"Bearer {_checked_token(self._tokens.access_token())}"
                )
            },
            body=None,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        if (
            raw_response.status_code != 200
            or not isinstance(raw_response.body, bytes)
            or len(raw_response.body) != int(size)
        ):
            raise LiveCloudAdapterError(
                "result_generation_read_changed",
                status_code=raw_response.status_code,
            )
        record = {
            "uri": uri,
            "generation": numeric_generation,
            "metageneration": int(metageneration),
            "bytes": len(raw_response.body),
            "sha256": hashlib.sha256(raw_response.body).hexdigest(),
            "crc32c": value["crc32c"],
            "etag": value["etag"],
        }
        self._cache[(uri, numeric_generation)] = raw_response.body
        return record, raw_response.body

    def list_prefix(self, *, prefix: str) -> list[dict[str, Any]]:
        normalized = prefix.rstrip("/")
        if normalized not in self._prefixes:
            raise ValueError("result list escaped exact prefixes")
        bucket, name = _gs_parts(normalized)
        page_token: str | None = None
        seen: set[str] = set()
        listed: list[dict[str, Any]] = []
        for _ in range(100):
            query = {
                "prefix": name.rstrip("/") + "/",
                "fields": (
                    "items(bucket,name,generation,metageneration,size,"
                    "crc32c,etag),nextPageToken"
                ),
            }
            if page_token is not None:
                query["pageToken"] = page_token
            response = self._http.request(
                method="GET",
                url=(
                    "https://storage.googleapis.com/storage/v1/b/"
                    f"{urllib.parse.quote(bucket, safe='')}/o?"
                    f"{urllib.parse.urlencode(query)}"
                ),
                headers=self._headers(),
                body=None,
                timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                raise LiveCloudAdapterError(
                    "result_prefix_list_failed",
                    status_code=response.status_code,
                )
            payload = _json_object(
                response, operation="result_prefix_list"
            )
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise LiveCloudAdapterError(
                    "result_prefix_list_response_changed"
                )
            for item in items:
                uri = (
                    f"gs://{item.get('bucket')}/{item.get('name')}"
                    if isinstance(item, Mapping)
                    else ""
                )
                generation = (
                    item.get("generation")
                    if isinstance(item, Mapping)
                    else None
                )
                if (
                    not self._allowed(uri)
                    or not isinstance(generation, str)
                    or _GENERATION.fullmatch(generation) is None
                ):
                    raise LiveCloudAdapterError(
                        "result_prefix_list_identity_changed"
                    )
                pinned = self.read_current(uri=uri)
                assert pinned is not None
                record, _ = pinned
                if (
                    record["generation"] != int(generation)
                    or str(record["metageneration"])
                    != item.get("metageneration")
                    or str(record["bytes"]) != item.get("size")
                    or record["crc32c"] != item.get("crc32c")
                    or record["etag"] != item.get("etag")
                ):
                    raise LiveCloudAdapterError(
                        "result_list_and_generation_disagree"
                    )
                listed.append(record)
            next_token = payload.get("nextPageToken")
            if next_token is None:
                break
            if (
                not isinstance(next_token, str)
                or not next_token
                or next_token in seen
            ):
                raise LiveCloudAdapterError(
                    "result_prefix_list_pagination_changed"
                )
            seen.add(next_token)
            page_token = next_token
        else:
            raise LiveCloudAdapterError(
                "result_prefix_list_page_bound_exhausted"
            )
        if len({row["uri"] for row in listed}) != len(listed):
            raise LiveCloudAdapterError(
                "result_prefix_list_duplicate_object"
            )
        return listed

    def read_bytes(self, *, uri: str, generation: int) -> bytes:
        if (
            not self._allowed(uri)
            or type(generation) is not int
            or generation <= 0
        ):
            raise ValueError("result generation read escaped exact prefixes")
        cached = self._cache.get((uri, generation))
        if cached is not None:
            return cached
        pinned = self.read_current(uri=uri)
        assert pinned is not None
        record, raw = pinned
        if record["generation"] != generation:
            raise LiveCloudAdapterError(
                "result_generation_changed_between_reads"
            )
        return raw


def _request_id(value: str) -> str:
    if not isinstance(value, str) or _UUID4.fullmatch(value) is None:
        raise ValueError("Compute request ID changed")
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise ValueError("Compute request ID changed") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("Compute request ID changed")
    return value


class ExactPairComputeClient:
    """One-shot Compute adapter restricted to the fresh exact pair."""

    def __init__(
        self,
        *,
        token_source: AccessTokenSource,
        instance_names: Sequence[str],
        principal: str,
        credential_kind: str,
        http_client: HttpsClient,
        allow_insert: bool,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        names = tuple(instance_names)
        if (
            len(names) != 2
            or len(set(names)) != 2
            or any(_SAFE_NAME.fullmatch(name) is None for name in names)
            or not isinstance(principal, str)
            or not principal
            or credential_kind
            not in {"fixed_nonrefreshing_controller", "user"}
            or allow_insert != (
                credential_kind == "fixed_nonrefreshing_controller"
            )
        ):
            raise ValueError("exact pair Compute identity changed")
        self.project = payload_transport.PROJECT
        self.zone = payload_transport.ZONE
        self.principal = principal
        self.credential_kind = credential_kind
        self._tokens = token_source
        self._names = frozenset(names)
        self._http = http_client
        self._allow_insert = allow_insert
        self._sleep = sleeper
        self._monotonic = monotonic

    def _url(self, suffix: str, query: Mapping[str, str] | None = None) -> str:
        base = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/zones/{self.zone}/{suffix}"
        )
        return base if query is None else (
            f"{base}?{urllib.parse.urlencode(query)}"
        )

    def _request(
        self,
        *,
        method: str,
        suffix: str,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, str] | None = None,
    ) -> rest_iam.HttpResponse:
        return self._http.request(
            method=method,
            url=self._url(suffix, query),
            headers={
                "Authorization": (
                    f"Bearer {_checked_token(self._tokens.access_token())}"
                ),
                "Accept": "application/json",
                **(
                    {"Content-Type": "application/json"}
                    if body is not None
                    else {}
                ),
            },
            body=_canonical_bytes(body) if body is not None else None,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )

    def _wait_operation(
        self,
        response: rest_iam.HttpResponse,
        *,
        operation: str,
    ) -> str:
        if response.status_code not in {200, 202}:
            raise LiveCloudAdapterError(
                f"{operation}_failed",
                status_code=response.status_code,
                operation=operation,
            )
        record = _json_object(response, operation=operation)
        name = record.get("name")
        if not isinstance(name, str) or _OPERATION.fullmatch(name) is None:
            raise LiveCloudAdapterError(
                f"{operation}_operation_changed", operation=operation
            )
        deadline = self._monotonic() + OPERATION_WAIT_SECONDS
        while record.get("status") != "DONE":
            if self._monotonic() >= deadline:
                raise LiveCloudAdapterError(
                    f"{operation}_operation_timeout", operation=operation
                )
            self._sleep(POLL_SECONDS)
            polled = self._request(
                method="GET", suffix=f"operations/{name}"
            )
            if polled.status_code != 200:
                raise LiveCloudAdapterError(
                    f"{operation}_operation_read_failed",
                    status_code=polled.status_code,
                    operation=operation,
                )
            record = _json_object(
                polled, operation=f"{operation}_operation_read"
            )
            if record.get("name") != name:
                raise LiveCloudAdapterError(
                    f"{operation}_operation_identity_changed",
                    operation=operation,
                )
        if record.get("error") is not None:
            raise LiveCloudAdapterError(
                f"{operation}_operation_error", operation=operation
            )
        return name

    def insert_instance(
        self,
        *,
        instance_name: str,
        request_id: str,
        body: Mapping[str, Any],
    ) -> pair_controller.ComputeMutationResult:
        if (
            not self._allow_insert
            or instance_name not in self._names
            or body.get("name") != instance_name
        ):
            raise ValueError("exact pair insert escaped allowlist")
        checked_request_id = _request_id(request_id)
        response = self._request(
            method="POST",
            suffix="instances",
            body=body,
            query={"requestId": checked_request_id},
        )
        operation_id = self._wait_operation(
            response, operation="instance_insert"
        )
        return pair_controller.ComputeMutationResult(
            status_code=response.status_code,
            target_name=instance_name,
            request_id=checked_request_id,
            operation_id=operation_id,
            operation_done=True,
        )

    def get_instance(
        self, *, instance_name: str
    ) -> Mapping[str, Any] | None:
        if instance_name not in self._names:
            raise ValueError("exact pair instance GET escaped allowlist")
        response = self._request(
            method="GET", suffix=f"instances/{instance_name}"
        )
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise LiveCloudAdapterError(
                "instance_get_failed", status_code=response.status_code
            )
        return _json_object(response, operation="instance_get")

    def set_metadata(
        self,
        *,
        instance_name: str,
        request_id: str,
        expected_fingerprint: str,
        values: Mapping[str, str],
    ) -> pair_controller.ComputeMutationResult:
        if (
            instance_name not in self._names
            or not isinstance(expected_fingerprint, str)
            or not expected_fingerprint
            or not isinstance(values, Mapping)
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                for key, value in values.items()
            )
        ):
            raise ValueError("exact pair metadata CAS escaped allowlist")
        checked_request_id = _request_id(request_id)
        response = self._request(
            method="POST",
            suffix=f"instances/{instance_name}/setMetadata",
            body={
                "fingerprint": expected_fingerprint,
                "items": [
                    {"key": key, "value": values[key]}
                    for key in sorted(values)
                ],
            },
            query={"requestId": checked_request_id},
        )
        if response.status_code == 412:
            return pair_controller.ComputeMutationResult(
                status_code=412,
                target_name=instance_name,
                request_id=checked_request_id,
                operation_id=None,
                operation_done=False,
            )
        operation_id = self._wait_operation(
            response, operation="instance_set_metadata"
        )
        return pair_controller.ComputeMutationResult(
            status_code=response.status_code,
            target_name=instance_name,
            request_id=checked_request_id,
            operation_id=operation_id,
            operation_done=True,
        )

    def _wait_absent(self, *, kind: str, name: str) -> int:
        deadline = self._monotonic() + ABSENCE_WAIT_SECONDS
        while True:
            response = self._request(
                method="GET", suffix=f"{kind}/{name}"
            )
            if response.status_code == 404:
                return 404
            if response.status_code != 200:
                raise LiveCloudAdapterError(
                    f"{kind}_absence_read_failed",
                    status_code=response.status_code,
                )
            if self._monotonic() >= deadline:
                raise LiveCloudAdapterError(f"{kind}_absence_timeout")
            self._sleep(POLL_SECONDS)

    def _delete_exact(self, *, kind: str, name: str) -> dict[str, Any]:
        initial = self._request(method="GET", suffix=f"{kind}/{name}")
        if initial.status_code == 404:
            return {
                "kind": kind,
                "name": name,
                "delete_call_count": 0,
                "already_absent": True,
                "final_get_status": 404,
            }
        if initial.status_code != 200:
            raise LiveCloudAdapterError(
                f"{kind}_cleanup_get_failed",
                status_code=initial.status_code,
            )
        request_id = str(uuid.uuid4())
        response = self._request(
            method="DELETE",
            suffix=f"{kind}/{name}",
            query={"requestId": request_id},
        )
        if response.status_code == 404:
            return {
                "kind": kind,
                "name": name,
                "delete_call_count": 1,
                "already_absent": True,
                "final_get_status": 404,
            }
        operation_id = self._wait_operation(
            response, operation=f"{kind}_delete"
        )
        return {
            "kind": kind,
            "name": name,
            "delete_call_count": 1,
            "already_absent": False,
            "request_id": request_id,
            "operation_id": operation_id,
            "final_get_status": self._wait_absent(
                kind=kind, name=name
            ),
        }

    def cleanup_exact_instances_and_disks(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]:
        if (
            self.credential_kind != "user"
            or set(instance_names) != self._names
            or set(disk_names) != self._names
            or len(instance_names) != 2
            or len(disk_names) != 2
        ):
            raise ValueError("exact pair cleanup surface changed")
        cleanup_records = []
        for kind, names in (
            ("instances", instance_names),
            ("disks", disk_names),
        ):
            for name in names:
                try:
                    cleanup_records.append(
                        {
                            **self._delete_exact(kind=kind, name=name),
                            "cleanup_call_completed": True,
                        }
                    )
                except Exception:
                    # Never let one target prevent cleanup of the other exact
                    # VM/disk targets.  The independent four-target GET pass
                    # below is the authority for the final state.
                    cleanup_records.append(
                        {
                            "kind": kind,
                            "name": name,
                            "cleanup_call_completed": False,
                            "sanitized_failure": (
                                "exact_target_cleanup_failed"
                            ),
                        }
                    )
        verification = self.verify_exact_instances_and_disks_absent(
            instance_names=instance_names,
            disk_names=disk_names,
        )
        body = {
            "schema": SCHEMA,
            "status": "exact_pair_instances_and_disks_absent",
            "cleanup_records": cleanup_records,
            "cleanup_records_sha256": hashlib.sha256(
                _canonical_bytes(cleanup_records)
            ).hexdigest(),
            "cleanup_call_failure_count": sum(
                row["cleanup_call_completed"] is False
                for row in cleanup_records
            ),
            "absence_verification_receipt_sha256": verification[
                "receipt_sha256"
            ],
            "instance_final_statuses": [404, 404],
            "disk_final_statuses": [404, 404],
            "all_four_targets_independently_processed": True,
            "all_four_targets_get404_verified": True,
            "exact_cleanup_complete": True,
        }
        return {
            **body,
            "receipt_sha256": hashlib.sha256(
                _canonical_bytes(body)
            ).hexdigest(),
        }

    def verify_exact_instances_and_disks_absent(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]:
        """Independently GET all exact targets and require four 404s."""

        if (
            self.credential_kind != "user"
            or set(instance_names) != self._names
            or set(disk_names) != self._names
            or len(instance_names) != 2
            or len(disk_names) != 2
        ):
            raise ValueError("exact pair absence readback surface changed")
        records = []
        for kind, names in (
            ("instances", instance_names),
            ("disks", disk_names),
        ):
            for name in names:
                status: int | None = None
                try:
                    response = self._request(
                        method="GET", suffix=f"{kind}/{name}"
                    )
                    status = response.status_code
                    if status not in {200, 404}:
                        raise LiveCloudAdapterError(
                            f"{kind}_final_absence_read_failed",
                            status_code=status,
                        )
                except Exception:
                    records.append(
                        {
                            "kind": kind,
                            "name": name,
                            "get_status": status,
                            "get_completed": False,
                            "sanitized_failure": (
                                "exact_target_absence_read_failed"
                            ),
                        }
                    )
                    continue
                records.append(
                    {
                        "kind": kind,
                        "name": name,
                        "get_status": status,
                        "get_completed": True,
                    }
                )
        if (
            len(records) != 4
            or any(
                row["get_completed"] is not True
                or row["get_status"] != 404
                for row in records
            )
        ):
            raise LiveCloudAdapterError(
                "exact_pair_four_target_get404_not_proven",
                operation="verify_exact_instances_and_disks_absent",
            )
        body = {
            "schema": SCHEMA,
            "status": "exact_pair_four_target_get404_verified",
            "records": records,
            "records_sha256": hashlib.sha256(
                _canonical_bytes(records)
            ).hexdigest(),
            "instance_final_statuses": [404, 404],
            "disk_final_statuses": [404, 404],
            "readback_count": 4,
            "all_four_targets_get404_verified": True,
        }
        return {
            **body,
            "receipt_sha256": hashlib.sha256(
                _canonical_bytes(body)
            ).hexdigest(),
        }


__all__ = [
    "ExactPairComputeClient",
    "GenerationPinnedBootstrapSourceStore",
    "GenerationPinnedResultStore",
    "GcloudUserAccessTokenSource",
    "IamCredentialsControllerTokenGenerator",
    "LiveCloudAdapterError",
    "SCHEMA",
    "StdlibCloudHttpsClient",
]
