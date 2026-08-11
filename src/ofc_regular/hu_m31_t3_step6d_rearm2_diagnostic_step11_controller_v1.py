"""One-VM Step 11 controller primitives and exact Google JSON API clients.

This module is deliberately separate from the worker package.  The controller
may sign the one diagnostic authorization and post-create claim, but the
private key is held only by the live local process.  It is never serialized,
uploaded, attached to the VM, or included in a receipt.

Cloud mutations are not performed by importing this module.  Callers must
present a validated launch contract, an explicit one-VM authorization, and a
dedicated controller credential.  All object creates use generation-match
zero and every accepted create is generation-pinned and byte-read back.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)


PUBLIC_KEY_FILE_SCHEMA = transport.RSA_PUBLIC_KEY_SCHEMA
PACKAGE_PROVISION_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_package_provision_v1"
)
STEP11_AUTHORIZATION_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_authorization_receipt_v1"
)
STEP11_CLAIM_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_claim_receipt_v1"
)
STEP11_FINAL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_final_receipt_v1"
)
CONTROLLER_SERVICE_ACCOUNT = (
    f"ofc-m31-t3-controller@{transport.PROJECT}.iam.gserviceaccount.com"
)
CLAIM_METADATA_KEY = "ofc-step11-controller-claim"
CONTRACT_METADATA_KEY = "ofc-step11-transport-contract"
AUTHORIZATION_METADATA_KEY = "ofc-step11-controller-authorization"
PUBLIC_KEY_METADATA_KEY = "ofc-step11-controller-public-key"
PREBOOTSTRAP_METADATA_KEY = "ofc-step11-prebootstrap"
STARTUP_METADATA_KEY = "startup-script"
CONTROLLER_ALLOWED_METADATA_KEYS = frozenset(
    {
        CLAIM_METADATA_KEY,
        CONTRACT_METADATA_KEY,
        AUTHORIZATION_METADATA_KEY,
        PUBLIC_KEY_METADATA_KEY,
        PREBOOTSTRAP_METADATA_KEY,
        STARTUP_METADATA_KEY,
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GCE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_ALLOWED_OPERATIONS = [
    "metadata_identity_read",
    "metadata_token_read",
    "generation_pinned_package_download",
    "generation_match_zero_result_upload",
    "result_readback",
    "compute_delete_self_after_done",
    "bounded_safety_shutdown_on_any_worker_failure",
]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero SHA-256")
    return value


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{label} is not an exact bounded integer")
    return value


def _safe_relative(value: Any) -> str:
    path = PurePosixPath(value) if isinstance(value, str) else None
    if (
        not isinstance(value, str)
        or not value
        or path is None
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("package object path escaped outer root")
    return value


@dataclass(frozen=True)
class EphemeralControllerKey:
    """A non-serializable handle plus its public-only canonical record."""

    _private_key: rsa.RSAPrivateKey
    public_record: Mapping[str, Any]

    def __reduce__(self) -> Any:
        raise TypeError("ephemeral controller private key cannot be serialized")

    def sign(self, *, record_type: str, unsigned: Mapping[str, Any]) -> str:
        if record_type not in ("authorization", "claim"):
            raise ValueError("controller record type changed")
        signature = self._private_key.sign(
            record_type.encode("ascii") + b"\0" + canonical_bytes(unsigned),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")


def generate_ephemeral_controller_key(
    *, key_size: int = 3_072
) -> EphemeralControllerKey:
    if key_size not in (2_048, 3_072, 4_096):
        raise ValueError("controller RSA key size is not approved")
    private_key = rsa.generate_private_key(
        public_exponent=65_537, key_size=key_size
    )
    numbers = private_key.public_key().public_numbers()
    modulus_hex = format(numbers.n, "x")
    if len(modulus_hex) % 2:
        modulus_hex = "0" + modulus_hex
    public_record = transport.build_rsa_public_key_record(
        modulus_hex=modulus_hex,
        exponent=numbers.e,
    )
    return EphemeralControllerKey(
        _private_key=private_key,
        public_record=public_record,
    )


def build_controller_authorization(
    *,
    contract: Mapping[str, Any],
    external_preflight_receipt_sha256: str,
    issued_unix_seconds: int,
    expires_unix_seconds: int,
    nonce: str | None = None,
    signer: EphemeralControllerKey,
) -> dict[str, Any]:
    checked = transport.validate_job_contract(contract)
    trust = checked["authorization_contract"]
    public_record = transport.validate_rsa_public_key_record(
        signer.public_record
    )
    if (
        trust["controller_key_id"] != public_record["key_id"]
        or trust["controller_public_key_sha256"]
        != canonical_sha256(public_record)
    ):
        raise ValueError("ephemeral signer is not pinned by contract")
    issued = _integer(
        issued_unix_seconds, "authorization issued", minimum=1
    )
    expires = _integer(
        expires_unix_seconds,
        "authorization expiry",
        minimum=issued + 60,
        maximum=issued + 7_200,
    )
    binding = checked["metadata_binding"]
    unsigned = {
        "schema": transport.AUTHORIZATION_SCHEMA,
        "contract_sha256": canonical_sha256(checked),
        "metadata_binding_sha256": checked["metadata_binding_sha256"],
        "direct_stage_identity_sha256": checked[
            "direct_stage_identity_sha256"
        ],
        "job_id": binding["job_id"],
        "attempt_index": binding["attempt_index"],
        "instance_name": binding["instance_name"],
        "controller_key_id": public_record["key_id"],
        "external_preflight_receipt_sha256": _sha(
            external_preflight_receipt_sha256,
            "external preflight receipt",
        ),
        "allowed_operations": list(_ALLOWED_OPERATIONS),
        "issued_unix_seconds": issued,
        "expires_unix_seconds": expires,
        "nonce": (
            secrets.token_hex(32)
            if nonce is None
            else _sha(nonce, "authorization nonce")
        ),
    }
    return {
        **unsigned,
        "signature": signer.sign(
            record_type="authorization", unsigned=unsigned
        ),
    }


def build_worker_claim(
    *,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    project_number: str,
    instance_id: str,
    package_generations: Mapping[str, int],
    nonce: str | None = None,
    signer: EphemeralControllerKey,
) -> dict[str, Any]:
    checked = transport.validate_job_contract(contract)
    auth = dict(authorization)
    binding = checked["metadata_binding"]
    expected_uris = [
        row["uri"]
        for row in checked["remote_layout"]["package_inventory"]["records"]
    ]
    if (
        not isinstance(package_generations, Mapping)
        or set(package_generations) != set(expected_uris)
    ):
        raise ValueError("package generation map is incomplete")
    generations = {
        uri: _integer(
            package_generations[uri], "package generation", minimum=1
        )
        for uri in expected_uris
    }
    if (
        not isinstance(project_number, str)
        or not project_number.isdigit()
        or not isinstance(instance_id, str)
        or not instance_id.isdigit()
    ):
        raise ValueError("project number or instance ID changed")
    unsigned = {
        "schema": transport.CLAIM_SCHEMA,
        "authorization_sha256": canonical_sha256(auth),
        "contract_sha256": canonical_sha256(checked),
        "project": transport.PROJECT,
        "project_number": project_number,
        "zone": transport.ZONE,
        "instance_name": binding["instance_name"],
        "instance_id": instance_id,
        "worker_service_account": transport.WORKER_SERVICE_ACCOUNT,
        "controller_key_id": checked["authorization_contract"][
            "controller_key_id"
        ],
        "stage_id": binding["stage_id"],
        "job_id": binding["job_id"],
        "attempt_index": binding["attempt_index"],
        "package_generations": generations,
        "nonce": (
            secrets.token_hex(32)
            if nonce is None
            else _sha(nonce, "claim nonce")
        ),
    }
    claim = {
        **unsigned,
        "signature": signer.sign(record_type="claim", unsigned=unsigned),
    }
    transport.validate_controller_approval(
        contract=checked,
        authorization=auth,
        claim=claim,
        verifier=transport.RsaSha256ControllerTrustVerifier(
            signer.public_record
        ),
        now_unix_seconds=auth["issued_unix_seconds"],
    )
    return claim


def build_package_provision_plan(
    *, contract: Mapping[str, Any], outer_root: str | Path
) -> dict[str, Any]:
    checked = transport.validate_job_contract(contract)
    root = Path(outer_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("outer package root must be an existing real directory")
    records: list[dict[str, Any]] = []
    for remote in checked["remote_layout"]["package_inventory"]["records"]:
        relative = _safe_relative(remote["path"])
        local = root / PurePosixPath(relative)
        if (
            not local.is_file()
            or local.is_symlink()
            or local.stat().st_size != remote["bytes"]
            or hashlib.sha256(local.read_bytes()).hexdigest()
            != remote["sha256"]
        ):
            raise ValueError(f"outer package object changed: {relative}")
        records.append(
            {
                "position": len(records),
                "path": relative,
                "uri": remote["uri"],
                "sha256": remote["sha256"],
                "bytes": remote["bytes"],
                "if_generation_match": 0,
                "generation_pinned_readback_required": True,
            }
        )
    return {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_step11_"
            "package_provision_plan_v1"
        ),
        "contract_sha256": canonical_sha256(checked),
        "outer_package_identity_sha256": checked["outer_package_manifest"][
            "outer_package_identity_sha256"
        ],
        "package_prefix": checked["remote_layout"]["package_prefix"],
        "records": records,
        "records_sha256": canonical_sha256(records),
        "object_count": len(records),
        "ordered_generation_match_zero": True,
        "readback_before_launch_required": True,
        "claim_not_created": True,
        "vm_not_created": True,
    }


def build_package_provision_receipt(
    *,
    plan: Mapping[str, Any],
    readbacks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != len(readbacks):
        raise ValueError("package provision readback count changed")
    normalized: list[dict[str, Any]] = []
    for expected, raw in zip(records, readbacks, strict=True):
        row = dict(raw)
        _exact(
            row,
            {
                "uri",
                "generation",
                "created",
                "sha256",
                "bytes",
                "crc32c",
                "etag",
            },
            "package provision readback",
        )
        if (
            row["uri"] != expected["uri"]
            or row["sha256"] != expected["sha256"]
            or row["bytes"] != expected["bytes"]
            or type(row["created"]) is not bool
            or not isinstance(row["crc32c"], str)
            or not row["crc32c"]
            or not isinstance(row["etag"], str)
            or not row["etag"]
        ):
            raise ValueError("package provision content identity changed")
        _integer(row["generation"], "package generation", minimum=1)
        normalized.append(row)
    generations = {
        row["uri"]: row["generation"] for row in normalized
    }
    receipt = {
        "schema": PACKAGE_PROVISION_RECEIPT_SCHEMA,
        "status": "exact_package_generation_zero_created_or_identical_readback",
        "contract_sha256": plan["contract_sha256"],
        "outer_package_identity_sha256": plan[
            "outer_package_identity_sha256"
        ],
        "package_prefix": plan["package_prefix"],
        "records": normalized,
        "records_sha256": canonical_sha256(normalized),
        "package_generations": generations,
        "package_generations_sha256": canonical_sha256(generations),
        "object_count": len(normalized),
        "all_generation_bound": True,
        "all_bytes_and_sha256_read_back": True,
        "vm_created": False,
        "claim_created": False,
        "diagnostic_only": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


class AccessTokenSource(Protocol):
    controller_principal: str

    def access_token(self) -> str: ...


class GcloudImpersonatedTokenSource:
    """Fetch short-lived tokens without persisting service-account keys."""

    def __init__(
        self,
        service_account: str = CONTROLLER_SERVICE_ACCOUNT,
    ) -> None:
        if service_account != CONTROLLER_SERVICE_ACCOUNT:
            raise ValueError("controller principal changed")
        self.controller_principal = service_account

    def access_token(self) -> str:
        completed = subprocess.run(
            [
                "gcloud",
                "auth",
                "print-access-token",
                f"--impersonate-service-account={self.controller_principal}",
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        token = completed.stdout.strip()
        if (
            not token
            or len(token) > 16_384
            or any(character.isspace() for character in token)
        ):
            raise RuntimeError("controller access-token shape changed")
        return token


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
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


class GoogleJsonClient:
    """No-proxy, no-redirect authenticated JSON/media client."""

    def __init__(self, token_source: AccessTokenSource) -> None:
        if (
            getattr(token_source, "controller_principal", None)
            != CONTROLLER_SERVICE_ACCOUNT
        ):
            raise ValueError("mutating client requires dedicated controller")
        self._token_source = token_source
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirectHandler()
        )

    @staticmethod
    def _allowed_url(url: str) -> bool:
        return (
            url.startswith("https://storage.googleapis.com/")
            or url.startswith(
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{transport.PROJECT}/"
            )
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> HttpResponse:
        if (
            method not in {"GET", "POST", "DELETE"}
            or not self._allowed_url(url)
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= 120
        ):
            raise ValueError("controller HTTP request escaped bounded surface")
        token = self._token_source.access_token()
        headers = {"Authorization": f"Bearer {token}"}
        if body is not None:
            multipart_match = re.fullmatch(
                r"multipart/related; boundary=(ofc-s11-[0-9a-f]{48})",
                content_type or "",
            )
            multipart_valid = False
            if multipart_match is not None:
                boundary = multipart_match.group(1).encode("ascii")
                delimiter = b"--" + boundary
                multipart_valid = (
                    len(boundary) <= 70
                    and body.startswith(delimiter + b"\r\n")
                    and body.endswith(delimiter + b"--\r\n")
                    and body.count(delimiter) == 3
                )
            if (
                method != "POST"
                or (
                    content_type
                    not in ("application/json", "application/octet-stream")
                    and not multipart_valid
                )
            ):
                raise ValueError("controller request body changed")
            headers["Content-Type"] = content_type
        elif content_type is not None:
            raise ValueError("content type without request body")
        request = urllib.request.Request(
            url=url, data=body, headers=headers, method=method
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                return HttpResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            return HttpResponse(
                status=int(error.code),
                headers=(
                    dict(error.headers.items()) if error.headers else {}
                ),
                body=error.read(),
            )


def _gs_parts(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("object URI must use gs://")
    bucket, separator, name = uri[5:].partition("/")
    if not separator or not bucket or not name:
        raise ValueError("object URI is incomplete")
    return bucket, name


def _storage_metadata_url(uri: str) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?fields=bucket,name,generation,"
        "metageneration,size,crc32c,etag,metadata"
    )


def _storage_media_url(uri: str, generation: int) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?alt=media&generation={generation}"
    )


def _storage_upload_url(uri: str) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o?uploadType=multipart&"
        f"ifGenerationMatch=0&name={urllib.parse.quote(name, safe='')}"
    )


def _storage_multipart_upload(
    *, uri: str, content: bytes, sha256: str
) -> tuple[bytes, str]:
    _bucket, name = _gs_parts(uri)
    digest = _sha(sha256, "package object SHA-256")
    boundary = f"ofc-s11-{digest[:48]}"
    boundary_bytes = boundary.encode("ascii")
    if len(boundary_bytes) > 70:
        raise AssertionError("multipart boundary exceeds MIME limit")
    if boundary_bytes in content:
        raise ValueError("package object collides with multipart boundary")
    metadata = canonical_bytes(
        {"name": name, "metadata": {"sha256": digest}}
    )
    body = b"".join(
        (
            b"--",
            boundary_bytes,
            b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",
            metadata,
            b"\r\n--",
            boundary_bytes,
            b"\r\nContent-Type: application/octet-stream\r\n\r\n",
            content,
            b"\r\n--",
            boundary_bytes,
            b"--\r\n",
        )
    )
    return body, f"multipart/related; boundary={boundary}"


def conditional_create_and_readback(
    *,
    client: GoogleJsonClient,
    uri: str,
    content: bytes,
) -> dict[str, Any]:
    if not isinstance(content, bytes) or not content:
        raise ValueError("package object content is empty")
    digest = sha256_bytes(content)
    upload_body, upload_content_type = _storage_multipart_upload(
        uri=uri, content=content, sha256=digest
    )
    response = client.request(
        method="POST",
        url=_storage_upload_url(uri),
        body=upload_body,
        content_type=upload_content_type,
    )
    created = response.status in (200, 201)
    metadata_response = response
    if response.status == 412:
        metadata_response = client.request(
            method="GET", url=_storage_metadata_url(uri)
        )
    elif not created:
        raise RuntimeError(
            f"conditional package create returned HTTP {response.status}"
        )
    if metadata_response.status not in (200, 201):
        raise RuntimeError(
            "package metadata readback returned "
            f"HTTP {metadata_response.status}"
        )
    try:
        metadata = json.loads(metadata_response.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("package metadata response is not JSON") from error
    generation_raw = metadata.get("generation")
    size_raw = metadata.get("size")
    if (
        not isinstance(generation_raw, str)
        or not generation_raw.isdigit()
        or int(generation_raw) <= 0
        or not isinstance(size_raw, str)
        or not size_raw.isdigit()
        or int(size_raw) != len(content)
        or not isinstance(metadata.get("crc32c"), str)
        or not metadata["crc32c"]
        or not isinstance(metadata.get("etag"), str)
        or not metadata["etag"]
        or not isinstance(metadata.get("metadata"), Mapping)
        or metadata["metadata"].get("sha256") != digest
    ):
        raise RuntimeError("package object metadata shape changed")
    generation = int(generation_raw)
    readback = client.request(
        method="GET", url=_storage_media_url(uri, generation)
    )
    if (
        readback.status != 200
        or readback.body != content
        or sha256_bytes(readback.body) != digest
    ):
        raise FileExistsError("package object readback differs")
    return {
        "uri": uri,
        "generation": generation,
        "created": created,
        "sha256": digest,
        "bytes": len(content),
        "crc32c": metadata["crc32c"],
        "etag": metadata["etag"],
    }


def provision_package(
    *,
    client: GoogleJsonClient,
    plan: Mapping[str, Any],
    outer_root: str | Path,
) -> dict[str, Any]:
    root = Path(outer_root)
    readbacks = []
    for record in plan["records"]:
        local = root / PurePosixPath(record["path"])
        raw = local.read_bytes()
        if (
            len(raw) != record["bytes"]
            or sha256_bytes(raw) != record["sha256"]
        ):
            raise ValueError("package object changed after planning")
        readbacks.append(
            conditional_create_and_readback(
                client=client, uri=record["uri"], content=raw
            )
        )
    return build_package_provision_receipt(
        plan=plan, readbacks=readbacks
    )


def exclusive_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("controller artifact path must be fresh")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        __import__("os").fsync(handle.fileno())


__all__ = [
    "AUTHORIZATION_METADATA_KEY",
    "CLAIM_METADATA_KEY",
    "CONTROLLER_SERVICE_ACCOUNT",
    "CONTRACT_METADATA_KEY",
    "EphemeralControllerKey",
    "GcloudImpersonatedTokenSource",
    "GoogleJsonClient",
    "PREBOOTSTRAP_METADATA_KEY",
    "PUBLIC_KEY_METADATA_KEY",
    "STARTUP_METADATA_KEY",
    "build_controller_authorization",
    "build_package_provision_plan",
    "build_package_provision_receipt",
    "build_worker_claim",
    "canonical_bytes",
    "canonical_sha256",
    "conditional_create_and_readback",
    "exclusive_write_json",
    "generate_ephemeral_controller_key",
    "provision_package",
]
