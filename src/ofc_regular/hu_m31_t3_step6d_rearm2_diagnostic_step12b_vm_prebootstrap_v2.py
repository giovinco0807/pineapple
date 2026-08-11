"""Role-local Step12b VM prebootstrap and metadata execution gate.

The VM receives a small generation-pinned role manifest in metadata.  A
stdlib-only startup loader verifies the signed manifest, downloads the shared
runtime bundle plus exactly one role payload, and only then imports this
module.  This module validates that downloaded role payload against the
metadata/authentication chain, waits boundedly for the post-create claim,
verifies guest-visible identity, waits boundedly for the signed pair-wide
release, and only then enters the direct-v2 alias bridge.

Imports and construction are side-effect free.  Metadata reads, sleeps,
package downloads, result writes, and self-delete are injected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_alias_bridge_v2
    as alias_bridge,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_authorization_v2
    as external_auth,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2
    as vm_metadata,
)


PREBOOTSTRAP_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_vm_prebootstrap_execution_v2"
)
MAX_CLAIM_WAIT_SECONDS = 900
DEFAULT_CLAIM_POLL_SECONDS = 2.0
METADATA_REQUEST_TIMEOUT_SECONDS = 5
MAX_METADATA_RESPONSE_BYTES = vm_metadata.MAX_TOTAL_METADATA_BYTES
MAX_WORKER_HTTP_RESPONSE_BYTES = 32 * 1024 * 1024
METADATA_BASE_URL = "http://metadata.google.internal/computeMetadata/v1"
DOWNLOAD_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_download_receipt_v2"
)
DOWNLOADED_ENTRYPOINT_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_downloaded_entrypoint_receipt_v2"
)
DEFAULT_LIVE_WORK_ROOT = Path("/var/lib/ofc-step12b")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METADATA_PATH = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+$")


class MetadataTextClient(Protocol):
    def read_text(self, path: str) -> str | None: ...


class VmMetadataReader(pair_release.PairReleaseMetadataReader, Protocol):
    def read_initial_metadata_values(self) -> Mapping[str, str]: ...

    def read_postcreate_claim(self) -> str | None: ...

    def read_guest_identity(self) -> Mapping[str, Any]: ...

    def read_worker_access_token(self) -> str: ...


@dataclass(frozen=True)
class _ParsedInitialMetadata:
    values: dict[str, str]
    deployment: dict[str, Any]
    selected_payload: dict[str, Any]
    public_key: dict[str, Any]
    authorization: dict[str, Any]
    package_generations: dict[str, int]
    external_preflight_receipt: dict[str, Any]
    role_bootstrap_manifest: dict[str, Any]
    run_nonce: str
    external_job_id: str
    source_role: str
    initial_budget_receipt: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    return vm_metadata.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return vm_metadata.canonical_sha256(value)


def _canonical_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} metadata value changed")
    try:
        encoded = raw.encode("ascii")
        value = json.loads(encoded)
    except (UnicodeEncodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} metadata is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != encoded:
        raise ValueError(f"{label} metadata is not canonical JSON")
    return value


def _checked_now(
    now_unix_seconds: int | Callable[[], int],
) -> int:
    value = (
        now_unix_seconds()
        if callable(now_unix_seconds)
        else now_unix_seconds
    )
    if type(value) is not int or value <= 0:
        raise ValueError("VM prebootstrap wall clock changed")
    return value


def _parse_initial_metadata(
    values: Mapping[str, str],
    *,
    selected_payload_contract: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
    now_unix_seconds: int,
) -> _ParsedInitialMetadata:
    checked_values = copy.deepcopy(dict(values))
    budget = vm_metadata.validate_initial_metadata_budget(checked_values)
    deployment = _canonical_object(
        checked_values[vm_metadata.DEPLOYMENT_CONTRACT_KEY],
        "deployment contract",
    )
    selected_payload = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    public_key = _canonical_object(
        checked_values[vm_metadata.CONTROLLER_PUBLIC_KEY],
        "controller public key",
    )
    authorization = _canonical_object(
        checked_values[vm_metadata.EXTERNAL_AUTHORIZATION_KEY],
        "external authorization",
    )
    package_generations_raw = _canonical_object(
        checked_values[vm_metadata.PACKAGE_GENERATIONS_KEY],
        "package generations",
    )
    external_preflight_receipt = _canonical_object(
        checked_values[vm_metadata.EXTERNAL_PREFLIGHT_RECEIPT_KEY],
        "external preflight receipt",
    )
    role_bootstrap_manifest = (
        bootstrap_source.validate_role_bootstrap_manifest_envelope(
        _canonical_object(
                checked_values[
                    vm_metadata.ROLE_BOOTSTRAP_MANIFEST_KEY
                ],
                "role bootstrap manifest",
            ),
            expected_external_job_id=checked_values[
                vm_metadata.EXTERNAL_JOB_ID_KEY
            ],
            expected_source_role=checked_values[
                vm_metadata.SOURCE_ROLE_KEY
            ],
        )
    )
    run_nonce = checked_values[vm_metadata.RUN_NONCE_KEY]
    external_job_id = checked_values[vm_metadata.EXTERNAL_JOB_ID_KEY]
    source_role = checked_values[vm_metadata.SOURCE_ROLE_KEY]
    deployment = deployment_v2.validate_role_runtime_view(
        deployment,
        selected_payload_contract=selected_payload,
        controller_public_key_record=public_key,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
    )
    position = deployment["selected_job_ids"].index(external_job_id)
    if (
        source_role != deployment["source_roles"][position]
        or selected_payload["metadata_binding"]["source_role"]
        != source_role
        or role_bootstrap_manifest["deployment_contract_sha256"]
        != deployment["deployment_contract_sha256"]
        or role_bootstrap_manifest["inner_job_id"]
        != deployment["instances"][position]["inner_job_id"]
    ):
        raise ValueError("role-local metadata source role changed")
    selected_payload_raw = canonical_bytes(selected_payload)
    payload_object = role_bootstrap_manifest["objects"][1]
    if (
        payload_object["bytes"] != len(selected_payload_raw)
        or payload_object["sha256"]
        != hashlib.sha256(selected_payload_raw).hexdigest()
    ):
        raise ValueError("downloaded role payload escaped role manifest")
    package_generations = {
        str(key): value for key, value in package_generations_raw.items()
    }
    if package_generations != package_generations_raw:
        raise ValueError("package generation key changed")
    checked_authorization = (
        external_auth.validate_external_authorization_role_runtime(
        authorization,
        deployment_contract=deployment,
        selected_payload_contract=selected_payload,
        controller_public_key_record=public_key,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
        package_generations=package_generations,
        external_preflight_receipt=external_preflight_receipt,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
        )
    )
    startup_sha256 = hashlib.sha256(
        checked_values[vm_metadata.STARTUP_KEY].encode("utf-8")
    ).hexdigest()
    if (
        external_preflight_receipt.get("source_hashes", {}).get(
            "startup_source_sha256"
        )
        != startup_sha256
    ):
        raise ValueError(
            "startup script escaped validated external preflight"
        )
    source_binding = checked_authorization.get("bootstrap_source_binding")
    manifest_objects = role_bootstrap_manifest["objects"]
    role_generations = {
        row["uri"]: row["generation"] for row in manifest_objects
    }
    if (
        not isinstance(source_binding, Mapping)
        or source_binding["source_plan_sha256"]
        != role_bootstrap_manifest["source_plan_sha256"]
        or source_binding["source_provision_receipt_sha256"]
        != role_bootstrap_manifest["source_provision_receipt_sha256"]
        or source_binding["source_prefix"]
        != role_bootstrap_manifest["source_prefix"]
        or source_binding["role_manifest_sha256"]
        != role_bootstrap_manifest["role_manifest_sha256"]
        or source_binding["role_manifest_objects_sha256"]
        != role_bootstrap_manifest["objects_sha256"]
        or source_binding["role_manifest_object_count"] != 2
        or source_binding["role_manifest_objects"] != manifest_objects
        or source_binding["role_source_generations"]
        != role_generations
        or source_binding["role_source_generations_sha256"]
        != canonical_sha256(role_generations)
    ):
        raise ValueError(
            "signed authorization bootstrap source binding changed"
        )
    return _ParsedInitialMetadata(
        values=checked_values,
        deployment=deployment,
        selected_payload=selected_payload,
        public_key=public_key,
        authorization=authorization,
        package_generations=package_generations,
        external_preflight_receipt=external_preflight_receipt,
        role_bootstrap_manifest=role_bootstrap_manifest,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
        source_role=source_role,
        initial_budget_receipt=budget,
    )


def build_role_initial_metadata(
    *,
    startup_script: str,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    authorization: Mapping[str, Any],
    package_generations: Mapping[str, int],
    external_preflight_receipt: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
    role_bootstrap_manifest: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
    now_unix_seconds: int,
) -> dict[str, str]:
    """Build the exact role-local insert metadata allowlist."""

    role_deployment = deployment_v2.validate_role_runtime_view(
        deployment_contract,
        selected_payload_contract=selected_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
    )
    position = role_deployment["selected_job_ids"].index(external_job_id)
    source_role = role_deployment["source_roles"][position]
    manifest = bootstrap_source.validate_role_bootstrap_manifest_envelope(
        role_bootstrap_manifest,
        expected_deployment_contract_sha256=role_deployment[
            "deployment_contract_sha256"
        ],
        expected_external_job_id=external_job_id,
        expected_source_role=source_role,
    )
    values = {
        vm_metadata.STARTUP_KEY: startup_script,
        vm_metadata.BLOCK_PROJECT_SSH_KEYS_KEY: "true",
        vm_metadata.RELEASE_STATE_KEY: vm_metadata.PENDING_RELEASE_STATE,
        vm_metadata.DEPLOYMENT_CONTRACT_KEY: canonical_bytes(
            role_deployment
        ).decode("ascii"),
        vm_metadata.ROLE_BOOTSTRAP_MANIFEST_KEY: canonical_bytes(
            manifest
        ).decode("ascii"),
        vm_metadata.CONTROLLER_PUBLIC_KEY: canonical_bytes(
            dict(controller_public_key_record)
        ).decode("ascii"),
        vm_metadata.EXTERNAL_AUTHORIZATION_KEY: canonical_bytes(
            dict(authorization)
        ).decode("ascii"),
        vm_metadata.PACKAGE_GENERATIONS_KEY: canonical_bytes(
            dict(package_generations)
        ).decode("ascii"),
        vm_metadata.EXTERNAL_PREFLIGHT_RECEIPT_KEY: canonical_bytes(
            dict(external_preflight_receipt)
        ).decode("ascii"),
        vm_metadata.RUN_NONCE_KEY: run_nonce,
        vm_metadata.EXTERNAL_JOB_ID_KEY: external_job_id,
        vm_metadata.SOURCE_ROLE_KEY: source_role,
    }
    _parse_initial_metadata(
        values,
        selected_payload_contract=selected_payload_contract,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )
    return values


def validate_role_initial_metadata(
    values: Mapping[str, str],
    *,
    selected_payload_contract: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
    now_unix_seconds: int,
) -> dict[str, Any]:
    """Parse and fully validate an actual role-local metadata snapshot."""

    parsed = _parse_initial_metadata(
        values,
        selected_payload_contract=selected_payload_contract,
        verifier=verifier,
        now_unix_seconds=now_unix_seconds,
    )
    body = {
        "schema": PREBOOTSTRAP_RECEIPT_SCHEMA,
        "phase": "role_local_initial_metadata_validated",
        "deployment_contract_sha256": parsed.deployment[
            "deployment_contract_sha256"
        ],
        "external_job_id": parsed.external_job_id,
        "source_role": parsed.source_role,
        "authorization_sha256": canonical_sha256(parsed.authorization),
        "package_generations_sha256": canonical_sha256(
            parsed.package_generations
        ),
        "role_bootstrap_manifest_sha256": (
            parsed.role_bootstrap_manifest[
                "role_manifest_sha256"
            ]
        ),
        "runtime_source_bundle_sha256": (
            parsed.role_bootstrap_manifest[
                "runtime_source_bundle_sha256"
            ]
        ),
        "initial_metadata_sha256": canonical_sha256(parsed.values),
        "initial_metadata_budget_receipt_sha256": (
            parsed.initial_budget_receipt[
                "metadata_budget_receipt_sha256"
            ]
        ),
        "one_role_payload_only": True,
        "postcreate_claim_present": False,
        "cloud_mutation_performed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


class UrllibGceMetadataTextClient:
    """Minimal read-only GCE metadata transport with redirect rejection."""

    def read_text(self, path: str) -> str | None:
        if (
            not isinstance(path, str)
            or _METADATA_PATH.fullmatch(path) is None
            or ".." in path.split("/")
        ):
            raise ValueError("GCE metadata path changed")
        url = METADATA_BASE_URL + path
        request = urllib.request.Request(
            url,
            headers={"Metadata-Flavor": "Google"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=METADATA_REQUEST_TIMEOUT_SECONDS
            ) as response:
                if (
                    response.status != 200
                    or response.geturl() != url
                    or response.headers.get("Metadata-Flavor") != "Google"
                ):
                    raise RuntimeError("GCE metadata response changed")
                raw = response.read(MAX_METADATA_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise RuntimeError("GCE metadata read failed") from error
        if len(raw) > MAX_METADATA_RESPONSE_BYTES:
            raise ValueError("GCE metadata response escaped size bound")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("GCE metadata response is not UTF-8") from error


class GceGuestMetadataReader:
    """Read the exact custom attributes and guest-visible VM identity."""

    def __init__(self, client: MetadataTextClient) -> None:
        self._client = client

    def _required(self, path: str) -> str:
        value = self._client.read_text(path)
        if not isinstance(value, str) or not value:
            raise ValueError(f"required GCE metadata missing: {path}")
        return value

    def _attribute(self, key: str) -> str | None:
        if key not in vm_metadata.POSTRELEASE_METADATA_KEYS:
            raise ValueError("GCE custom metadata key escaped allowlist")
        return self._client.read_text(f"/instance/attributes/{key}")

    def read_initial_metadata_values(self) -> Mapping[str, str]:
        return {
            key: self._required(f"/instance/attributes/{key}")
            for key in sorted(vm_metadata.INITIAL_METADATA_KEYS)
        }

    def read_postcreate_claim(self) -> str | None:
        return self._attribute(vm_metadata.POSTCREATE_CLAIM_KEY)

    def read_pair_release(self) -> str | None:
        return self._attribute(vm_metadata.PAIR_RELEASE_KEY)

    def read_guest_identity(self) -> Mapping[str, Any]:
        zone_path = self._required("/instance/zone")
        zone = zone_path.rsplit("/", 1)[-1]
        scopes = self._required(
            "/instance/service-accounts/default/scopes"
        ).splitlines()
        return {
            "project_id": self._required("/project/project-id"),
            "project_number": self._required(
                "/project/numeric-project-id"
            ),
            "zone": zone,
            "instance_name": self._required("/instance/name"),
            "provider_instance_id": self._required("/instance/id"),
            "service_account_email": self._required(
                "/instance/service-accounts/default/email"
            ),
            "oauth_scopes": scopes,
            "external_job_id": self._required(
                f"/instance/attributes/{vm_metadata.EXTERNAL_JOB_ID_KEY}"
            ),
            "source_role": self._required(
                f"/instance/attributes/{vm_metadata.SOURCE_ROLE_KEY}"
            ),
        }

    def read_worker_access_token(self) -> str:
        raw = self._required(
            "/instance/service-accounts/default/token"
        )
        try:
            token = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                "worker metadata OAuth token is not JSON"
            ) from error
        if (
            not isinstance(token, dict)
            or set(token) != {"access_token", "expires_in", "token_type"}
            or not isinstance(token["access_token"], str)
            or not token["access_token"]
            or any(character.isspace() for character in token["access_token"])
            or len(token["access_token"]) > 8_192
            or token["token_type"] != "Bearer"
            or type(token["expires_in"]) is not int
            or token["expires_in"] < 60
        ):
            raise ValueError("worker metadata OAuth token shape changed")
        return token["access_token"]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


@dataclass(frozen=True)
class _WorkerApiResponse:
    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes


class _WorkerHttpsClient:
    """Small no-proxy/no-redirect HTTPS client for exact worker routes."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        access_token: str,
        body: bytes | None = None,
        content_type: str | None = None,
        max_response_bytes: int = MAX_WORKER_HTTP_RESPONSE_BYTES,
    ) -> _WorkerApiResponse:
        if (
            method not in {"GET", "POST", "DELETE"}
            or not isinstance(url, str)
            or not url.startswith("https://")
            or not isinstance(access_token, str)
            or not access_token
            or type(max_response_bytes) is not int
            or not 1 <= max_response_bytes <= MAX_WORKER_HTTP_RESPONSE_BYTES
        ):
            raise ValueError("worker HTTPS request changed")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=METADATA_REQUEST_TIMEOUT_SECONDS
            ) as response:
                raw = response.read(max_response_bytes + 1)
                observed = _WorkerApiResponse(
                    status=response.status,
                    url=response.geturl(),
                    headers=dict(response.headers.items()),
                    body=raw,
                )
        except urllib.error.HTTPError as error:
            raw = error.read(max_response_bytes + 1)
            observed = _WorkerApiResponse(
                status=error.code,
                url=error.geturl(),
                headers=dict(error.headers.items()),
                body=raw,
            )
        if observed.url != url:
            raise RuntimeError("worker HTTPS redirect was attempted")
        if len(observed.body) > max_response_bytes:
            raise ValueError("worker HTTPS response escaped size bound")
        return observed


def _gs_parts(uri: Any) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("worker GCS URI changed")
    bucket_and_name = uri[5:]
    bucket, separator, name = bucket_and_name.partition("/")
    if (
        separator != "/"
        or bucket != payload_transport.BUCKET
        or not name
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("worker GCS URI escaped bucket")
    return bucket, name


def _gcs_generation_get_url(uri: str, generation: int) -> str:
    bucket, name = _gs_parts(uri)
    if type(generation) is not int or generation <= 0:
        raise ValueError("worker GCS generation changed")
    query = urllib.parse.urlencode(
        {"alt": "media", "generation": str(generation)}
    )
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?{query}"
    )


def _gcs_upload_url(uri: str) -> str:
    bucket, name = _gs_parts(uri)
    query = urllib.parse.urlencode(
        {
            "uploadType": "media",
            "ifGenerationMatch": "0",
            "name": name,
        }
    )
    return (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o?{query}"
    )


class LiveGenerationPinnedPackageReader:
    def __init__(
        self,
        *,
        metadata_reader: GceGuestMetadataReader,
        selected_payload_contract: Mapping[str, Any],
        http_client: _WorkerHttpsClient,
    ) -> None:
        payload = payload_transport.validate_job_contract(
            selected_payload_contract
        )
        records = payload["remote_layout"]["package_inventory"]["records"]
        if len(records) != 16:
            raise ValueError("live immutable package inventory changed")
        self._allowed = {
            row["uri"]: {
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in records
        }
        self._metadata = metadata_reader
        self._http = http_client

    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes:
        expected = self._allowed.get(uri)
        if expected is None:
            raise ValueError("live package GET escaped inventory")
        response = self._http.request(
            method="GET",
            url=_gcs_generation_get_url(uri, generation),
            access_token=self._metadata.read_worker_access_token(),
            max_response_bytes=min(
                MAX_WORKER_HTTP_RESPONSE_BYTES,
                expected["bytes"],
            ),
        )
        if (
            response.status != 200
            or len(response.body) != expected["bytes"]
            or hashlib.sha256(response.body).hexdigest()
            != expected["sha256"]
        ):
            raise RuntimeError(
                "live generation-pinned package readback changed"
            )
        return response.body


class LiveDirectV2ResultWriter:
    def __init__(
        self,
        *,
        metadata_reader: GceGuestMetadataReader,
        deployment_contract: Mapping[str, Any],
        external_job_id: str,
        http_client: _WorkerHttpsClient,
    ) -> None:
        jobs = [
            row
            for row in deployment_contract["remote_layout"]["jobs"]
            if row["job_id"] == external_job_id
        ]
        if len(jobs) != 1:
            raise ValueError("live result writer job changed")
        job = jobs[0]
        allowed = [
            *job["tree_object_uris"],
            *job["upload_uris"],
            *job["heartbeat_uris"],
            job["done_uri"],
        ]
        if (
            len(allowed) != alias_bridge.RESULT_OBJECT_COUNT
            or len(set(allowed)) != len(allowed)
        ):
            raise ValueError("live direct-v2 result allowlist changed")
        self._allowed = frozenset(allowed)
        self._metadata = metadata_reader
        self._http = http_client

    def conditional_create(
        self, *, uri: str, content: bytes
    ) -> Mapping[str, Any]:
        if (
            uri not in self._allowed
            or not isinstance(content, bytes)
            or not content
            or len(content) > MAX_WORKER_HTTP_RESPONSE_BYTES
        ):
            raise ValueError("live direct-v2 result create changed")
        response = self._http.request(
            method="POST",
            url=_gcs_upload_url(uri),
            access_token=self._metadata.read_worker_access_token(),
            body=content,
            content_type="application/octet-stream",
            max_response_bytes=1 << 20,
        )
        if response.status not in {200, 201}:
            raise RuntimeError(
                f"live direct-v2 create returned HTTP {response.status}"
            )
        try:
            created = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "live direct-v2 create response is not JSON"
            ) from error
        bucket, name = _gs_parts(uri)
        generation_raw = (
            created.get("generation")
            if isinstance(created, Mapping)
            else None
        )
        if (
            not isinstance(generation_raw, str)
            or not generation_raw.isdigit()
            or int(generation_raw) <= 0
            or created.get("bucket") != bucket
            or created.get("name") != name
            or str(created.get("size")) != str(len(content))
        ):
            raise RuntimeError(
                "live direct-v2 create response changed"
            )
        generation = int(generation_raw)
        readback = self._http.request(
            method="GET",
            url=_gcs_generation_get_url(uri, generation),
            access_token=self._metadata.read_worker_access_token(),
            max_response_bytes=len(content),
        )
        digest = hashlib.sha256(content).hexdigest()
        if (
            readback.status != 200
            or readback.body != content
            or hashlib.sha256(readback.body).hexdigest() != digest
        ):
            raise RuntimeError("live direct-v2 result readback changed")
        return {
            "uri": uri,
            "generation": generation,
            "sha256": digest,
            "bytes": len(content),
            "created": True,
        }


class LiveDownloadedPayloadRuntime:
    def validate_and_run(
        self,
        *,
        payload_contract: Mapping[str, Any],
        deployment_contract: Mapping[str, Any],
        execution_job_id: str,
        outer_root: Path,
        work_root: Path,
    ) -> tuple[Path, Mapping[str, Any]]:
        payload = payload_transport.validate_job_contract(payload_contract)
        output = payload_transport.run_downloaded_worker(
            contract=payload,
            outer_root=outer_root,
            work_root=work_root,
        )
        payload_transport.validate_completed_output_with_worker_venv(output)
        return output, alias_bridge.build_payload_run_receipt(
            deployment_contract=deployment_contract,
            execution_job_id=execution_job_id,
        )


class LiveExactSelfDeleter:
    def __init__(
        self,
        *,
        metadata_reader: GceGuestMetadataReader,
        deployment_contract: Mapping[str, Any],
        external_job_id: str,
        http_client: _WorkerHttpsClient,
    ) -> None:
        matches = [
            row
            for row in deployment_contract["instances"]
            if row["job_id"] == external_job_id
        ]
        jobs = [
            row
            for row in deployment_contract["remote_layout"]["jobs"]
            if row["job_id"] == external_job_id
        ]
        if len(matches) != 1 or len(jobs) != 1:
            raise ValueError("live self-delete job changed")
        self._instance_name = matches[0]["instance_name"]
        self._done_uri = jobs[0]["done_uri"]
        self._metadata = metadata_reader
        self._http = http_client

    def request_exact_self_delete(
        self,
        *,
        project: str,
        zone: str,
        instance_name: str,
        done_readback: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if (
            project != payload_transport.PROJECT
            or zone != payload_transport.ZONE
            or instance_name != self._instance_name
            or done_readback.get("uri") != self._done_uri
            or type(done_readback.get("generation")) is not int
            or done_readback["generation"] <= 0
            or not isinstance(done_readback.get("sha256"), str)
            or _SHA256.fullmatch(done_readback["sha256"]) is None
        ):
            raise ValueError("live exact self-delete prerequisite changed")
        url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{urllib.parse.quote(project, safe='')}/zones/"
            f"{urllib.parse.quote(zone, safe='')}/instances/"
            f"{urllib.parse.quote(instance_name, safe='')}"
        )
        response = self._http.request(
            method="DELETE",
            url=url,
            access_token=self._metadata.read_worker_access_token(),
            max_response_bytes=1 << 20,
        )
        if response.status not in {200, 202}:
            raise RuntimeError(
                f"live self-delete returned HTTP {response.status}"
            )
        return {
            "project": project,
            "zone": zone,
            "instance_name": instance_name,
            "delete_requested": True,
            "done_readback_sha256": canonical_sha256(done_readback),
            "response_sha256": hashlib.sha256(response.body).hexdigest(),
        }


def validate_bootstrap_download_receipt(
    value: Mapping[str, Any],
    *,
    role_bootstrap_manifest: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the loader handoff before constructing any live adapter."""

    manifest = bootstrap_source.validate_role_bootstrap_manifest_envelope(
        role_bootstrap_manifest
    )
    payload = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    receipt = copy.deepcopy(dict(value))
    expected_fields = {
        "schema",
        "deployment_contract_sha256",
        "source_plan_sha256",
        "source_provision_receipt_sha256",
        "role_manifest_sha256",
        "external_job_id",
        "inner_job_id",
        "source_role",
        "object_count",
        "records",
        "records_sha256",
        "runtime_source_bundle_sha256",
        "runtime_source_materialization_receipt_sha256",
        "payload_contract_sha256",
        "generation_pinned_get_only",
        "opponent_role_payload_download_count",
        "downloaded_before_full_prebootstrap_import",
        "repo_import_permitted",
        "site_packages_import_permitted",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("bootstrap download receipt fields changed")
    supplied = receipt.pop("receipt_sha256", None)
    if (
        not isinstance(supplied, str)
        or _SHA256.fullmatch(supplied) is None
        or canonical_sha256(receipt) != supplied
    ):
        raise ValueError("bootstrap download receipt digest changed")
    records = receipt["records"]
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("bootstrap download receipt record count changed")
    for manifest_row, downloaded in zip(
        manifest["objects"], records, strict=True
    ):
        if (
            not isinstance(downloaded, Mapping)
            or set(downloaded)
            != {
                "kind",
                "path",
                "uri",
                "bytes",
                "sha256",
                "generation",
                "downloaded",
                "readback_verified",
            }
            or any(
                downloaded[key] != manifest_row[key]
                for key in (
                    "kind",
                    "path",
                    "uri",
                    "bytes",
                    "sha256",
                    "generation",
                )
            )
            or downloaded["downloaded"] is not True
            or downloaded["readback_verified"] is not True
        ):
            raise ValueError("bootstrap download receipt record changed")
    payload_raw = canonical_bytes(payload)
    if (
        receipt["schema"] != DOWNLOAD_RECEIPT_SCHEMA
        or receipt["deployment_contract_sha256"]
        != manifest["deployment_contract_sha256"]
        or receipt["source_plan_sha256"]
        != manifest["source_plan_sha256"]
        or receipt["source_provision_receipt_sha256"]
        != manifest["source_provision_receipt_sha256"]
        or receipt["role_manifest_sha256"]
        != manifest["role_manifest_sha256"]
        or receipt["external_job_id"] != manifest["external_job_id"]
        or receipt["inner_job_id"] != manifest["inner_job_id"]
        or receipt["source_role"] != manifest["source_role"]
        or receipt["object_count"] != 2
        or receipt["records_sha256"] != canonical_sha256(records)
        or receipt["runtime_source_bundle_sha256"]
        != manifest["runtime_source_bundle_sha256"]
        or receipt["payload_contract_sha256"]
        != hashlib.sha256(payload_raw).hexdigest()
        or receipt["payload_contract_sha256"]
        != manifest["objects"][1]["sha256"]
        or receipt["generation_pinned_get_only"] is not True
        or receipt["opponent_role_payload_download_count"] != 0
        or receipt["downloaded_before_full_prebootstrap_import"] is not True
        or receipt["repo_import_permitted"] is not False
        or receipt["site_packages_import_permitted"] is not False
    ):
        raise ValueError("bootstrap download receipt changed")
    materialization_sha = receipt[
        "runtime_source_materialization_receipt_sha256"
    ]
    if (
        not isinstance(materialization_sha, str)
        or _SHA256.fullmatch(materialization_sha) is None
        or materialization_sha == "0" * 64
    ):
        raise ValueError("runtime source materialization receipt changed")
    return {**receipt, "receipt_sha256": supplied}


def run_downloaded_entrypoint(
    manifest: Mapping[str, Any],
    payload_contract: Mapping[str, Any],
    download_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the complete live VM path after the stdlib loader handoff."""

    checked_manifest = (
        bootstrap_source.validate_role_bootstrap_manifest_envelope(manifest)
    )
    checked_payload = payload_transport.validate_job_contract(
        payload_contract
    )
    checked_download = validate_bootstrap_download_receipt(
        download_receipt,
        role_bootstrap_manifest=checked_manifest,
        selected_payload_contract=checked_payload,
    )
    metadata_reader = GceGuestMetadataReader(
        UrllibGceMetadataTextClient()
    )
    initial_values = dict(
        metadata_reader.read_initial_metadata_values()
    )
    public_key = _canonical_object(
        initial_values[vm_metadata.CONTROLLER_PUBLIC_KEY],
        "controller public key",
    )
    verifier = payload_transport.RsaSha256ControllerTrustVerifier(
        public_key
    )
    parsed = _parse_initial_metadata(
        initial_values,
        selected_payload_contract=checked_payload,
        verifier=verifier,
        now_unix_seconds=int(time.time()),
    )
    if (
        parsed.role_bootstrap_manifest != checked_manifest
        or parsed.deployment["deployment_contract_sha256"]
        != checked_download["deployment_contract_sha256"]
    ):
        raise ValueError("downloaded entrypoint metadata handoff changed")

    live_parent = DEFAULT_LIVE_WORK_ROOT
    if live_parent.exists():
        if not live_parent.is_dir() or live_parent.is_symlink():
            raise ValueError("live worker root parent changed")
    else:
        if not live_parent.parent.is_dir() or live_parent.parent.is_symlink():
            raise ValueError("live worker root parent is unavailable")
        live_parent.mkdir(mode=0o700)
    os.chmod(live_parent, 0o700)
    bridge_root = live_parent / (
        "bridge-"
        f"{parsed.deployment['deployment_contract_sha256'][:16]}-"
        f"{parsed.source_role}"
    )
    http_client = _WorkerHttpsClient()
    package_reader = LiveGenerationPinnedPackageReader(
        metadata_reader=metadata_reader,
        selected_payload_contract=checked_payload,
        http_client=http_client,
    )
    result_writer = LiveDirectV2ResultWriter(
        metadata_reader=metadata_reader,
        deployment_contract=parsed.deployment,
        external_job_id=parsed.external_job_id,
        http_client=http_client,
    )
    self_deleter = LiveExactSelfDeleter(
        metadata_reader=metadata_reader,
        deployment_contract=parsed.deployment,
        external_job_id=parsed.external_job_id,
        http_client=http_client,
    )
    execution = execute_vm_prebootstrap(
        metadata_reader=metadata_reader,
        selected_payload_contract=checked_payload,
        verifier=verifier,
        package_reader=package_reader,
        payload_runtime=LiveDownloadedPayloadRuntime(),
        result_writer=result_writer,
        self_deleter=self_deleter,
        fresh_root=str(bridge_root),
    )
    body = {
        "schema": DOWNLOADED_ENTRYPOINT_RECEIPT_SCHEMA,
        "status": "downloaded_runtime_completed_direct_v2_lifecycle",
        "deployment_contract_sha256": parsed.deployment[
            "deployment_contract_sha256"
        ],
        "external_job_id": parsed.external_job_id,
        "source_role": parsed.source_role,
        "role_manifest_sha256": checked_manifest[
            "role_manifest_sha256"
        ],
        "bootstrap_download_receipt_sha256": checked_download[
            "receipt_sha256"
        ],
        "prebootstrap_execution_receipt_sha256": execution[
            "receipt_sha256"
        ],
        "generation_pinned_source_download_complete": True,
        "pair_release_before_package_download": True,
        "old_result_write_count": 0,
        "done_readback_before_self_delete": True,
        "current_profile_changed": False,
        "execution": execution,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _wait_for_claim(
    *,
    reader: VmMetadataReader,
    initial_values: Mapping[str, str],
    timeout_seconds: int,
    poll_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= MAX_CLAIM_WAIT_SECONDS
        or not math.isfinite(poll_seconds)
        or not 0 < poll_seconds <= 10
    ):
        raise ValueError("post-create claim wait bound changed")
    started = float(now())
    while True:
        raw = reader.read_postcreate_claim()
        if raw is not None:
            receipt = vm_metadata.validate_postclaim_metadata_budget(
                initial_values=initial_values,
                claim_value=raw,
            )
            return raw, _canonical_object(raw, "post-create claim"), receipt
        elapsed = float(now()) - started
        if elapsed >= timeout_seconds:
            raise TimeoutError("post-create claim metadata wait timed out")
        sleep(min(poll_seconds, timeout_seconds - elapsed))


class _RecordingReleaseReader:
    def __init__(self, inner: VmMetadataReader) -> None:
        self._inner = inner
        self.release_raw: str | None = None

    def read_pair_release(self) -> str | None:
        raw = self._inner.read_pair_release()
        if raw is not None:
            self.release_raw = raw
        return raw


def execute_vm_prebootstrap(
    *,
    metadata_reader: VmMetadataReader,
    selected_payload_contract: Mapping[str, Any],
    verifier: external_auth.ControllerVerifier,
    package_reader: alias_bridge.ImmutablePackageReader,
    payload_runtime: alias_bridge.PayloadRuntime,
    result_writer: alias_bridge.ConditionalResultWriter,
    self_deleter: alias_bridge.ExactSelfDeleter,
    fresh_root: str,
    wall_clock: int | Callable[[], int] = lambda: int(time.time()),
    claim_timeout_seconds: int = MAX_CLAIM_WAIT_SECONDS,
    claim_poll_seconds: float = DEFAULT_CLAIM_POLL_SECONDS,
    release_timeout_seconds: int = pair_release.MAX_RELEASE_WAIT_SECONDS,
    release_poll_seconds: float = pair_release.DEFAULT_RELEASE_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute the VM path; payload work is impossible before pair release."""

    initial_values = dict(metadata_reader.read_initial_metadata_values())
    initial = _parse_initial_metadata(
        initial_values,
        selected_payload_contract=selected_payload_contract,
        verifier=verifier,
        now_unix_seconds=_checked_now(wall_clock),
    )
    claim_raw, claim, claim_budget = _wait_for_claim(
        reader=metadata_reader,
        initial_values=initial_values,
        timeout_seconds=claim_timeout_seconds,
        poll_seconds=claim_poll_seconds,
        now=monotonic,
        sleep=sleep,
    )
    observed_identity = dict(metadata_reader.read_guest_identity())
    role_approval = pair_release.validate_role_runtime_approval(
        authorization=initial.authorization,
        claim=claim,
        deployment_contract=initial.deployment,
        selected_payload_contract=initial.selected_payload,
        controller_public_key_record=initial.public_key,
        run_nonce=initial.run_nonce,
        external_job_id=initial.external_job_id,
        package_generations=initial.package_generations,
        external_preflight_receipt=initial.external_preflight_receipt,
        observed_identity=observed_identity,
        verifier=verifier,
        now_unix_seconds=_checked_now(wall_clock),
    )
    recording_reader = _RecordingReleaseReader(metadata_reader)
    release_approval = pair_release.wait_for_pair_release(
        reader=recording_reader,
        deployment_contract=initial.deployment,
        selected_payload_contract=initial.selected_payload,
        controller_public_key_record=initial.public_key,
        run_nonce=initial.run_nonce,
        external_job_id=initial.external_job_id,
        role_approval=role_approval,
        verifier=verifier,
        timeout_seconds=release_timeout_seconds,
        poll_seconds=release_poll_seconds,
        now=monotonic,
        sleep=sleep,
    )
    if recording_reader.release_raw is None:
        raise AssertionError("validated pair release raw value was not recorded")
    release_budget = vm_metadata.validate_postrelease_metadata_budget(
        initial_values=initial_values,
        claim_value=claim_raw,
        release_value=recording_reader.release_raw,
    )
    bridge_receipt = alias_bridge.execute_alias_bridge(
        deployment_contract=initial.deployment,
        selected_payload_contract=initial.selected_payload,
        controller_public_key_record=initial.public_key,
        run_nonce=initial.run_nonce,
        package_generations=initial.package_generations,
        execution_job_id=initial.external_job_id,
        role_runtime_approval=role_approval,
        pair_release_approval=release_approval,
        package_reader=package_reader,
        payload_runtime=payload_runtime,
        result_writer=result_writer,
        self_deleter=self_deleter,
        fresh_root=fresh_root,
    )
    body = {
        "schema": PREBOOTSTRAP_RECEIPT_SCHEMA,
        "status": "pair_release_validated_then_direct_v2_bridge_complete",
        "deployment_contract_sha256": initial.deployment[
            "deployment_contract_sha256"
        ],
        "external_job_id": initial.external_job_id,
        "source_role": initial.source_role,
        "initial_metadata_sha256": canonical_sha256(initial_values),
        "initial_metadata_budget_receipt_sha256": (
            initial.initial_budget_receipt[
                "metadata_budget_receipt_sha256"
            ]
        ),
        "postclaim_metadata_budget_receipt_sha256": claim_budget[
            "metadata_budget_receipt_sha256"
        ],
        "postrelease_metadata_budget_receipt_sha256": release_budget[
            "metadata_budget_receipt_sha256"
        ],
        "role_runtime_approval_receipt_sha256": (
            role_approval.receipt_sha256
        ),
        "pair_release_approval_receipt_sha256": (
            release_approval.receipt_sha256
        ),
        "bridge_execution_receipt_sha256": bridge_receipt[
            "receipt_sha256"
        ],
        "execution_order": [
            "initial_metadata_validated",
            "postcreate_claim_validated",
            "guest_identity_validated",
            "pair_release_validated",
            "alias_bridge_executed",
        ],
        "payload_started_after_pair_release": True,
        "one_role_payload_only": True,
        "opponent_role_payload_present": False,
        "worker_observed_controller_metadata_fingerprint": False,
        "current_profile_changed": False,
        "bridge_receipt": bridge_receipt,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


__all__ = [
    "DEFAULT_CLAIM_POLL_SECONDS",
    "DEFAULT_LIVE_WORK_ROOT",
    "DOWNLOADED_ENTRYPOINT_RECEIPT_SCHEMA",
    "DOWNLOAD_RECEIPT_SCHEMA",
    "GceGuestMetadataReader",
    "LiveDirectV2ResultWriter",
    "LiveDownloadedPayloadRuntime",
    "LiveExactSelfDeleter",
    "LiveGenerationPinnedPackageReader",
    "MAX_CLAIM_WAIT_SECONDS",
    "MAX_METADATA_RESPONSE_BYTES",
    "METADATA_BASE_URL",
    "METADATA_REQUEST_TIMEOUT_SECONDS",
    "MetadataTextClient",
    "PREBOOTSTRAP_RECEIPT_SCHEMA",
    "UrllibGceMetadataTextClient",
    "VmMetadataReader",
    "build_role_initial_metadata",
    "canonical_bytes",
    "canonical_sha256",
    "execute_vm_prebootstrap",
    "run_downloaded_entrypoint",
    "validate_bootstrap_download_receipt",
    "validate_role_initial_metadata",
]
