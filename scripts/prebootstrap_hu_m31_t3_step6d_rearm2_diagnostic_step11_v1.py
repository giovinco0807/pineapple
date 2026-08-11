#!/usr/bin/env python3
"""Minimal, standard-library-only Step 11 trust bootstrap.

This file is supplied as immutable instance metadata.  It validates the
controller public key, authorization, post-create claim, exact VM identity,
OAuth scope, and content-addressed package generations before downloading the
larger worker-side validator.  The downloaded transport performs the complete
contract validation again before its first result write.

The controller private key, user credentials, access tokens, and opponent
private information are never accepted as inputs or written to disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_gce_transport_10c2_v1"
)
AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_controller_authorization_v1"
)
CLAIM_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_worker_claim_v1"
PUBLIC_KEY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_controller_rsa_public_key_v1"
)
SIGNATURE_ALGORITHM = "RSASSA-PKCS1-v1_5-SHA256"
PROJECT = "ofc-solver-485418"
ZONE = "asia-northeast1-b"
WORKER_SERVICE_ACCOUNT = (
    "ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com"
)
REQUIRED_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
STAGE_ID = "stage1_lifecycle_one_candidate_vm"
JOB_ID = "candidate-shard-00"
ATTEMPT_INDEX = 0
METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
MAX_DOWNLOAD_BYTES = 2_000_000
MAX_PACKAGE_RECORDS = 32
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_HEX = re.compile(r"^[0-9a-f]+$")
_SAFE_METADATA_KEY = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,62})$")
_BOOTSTRAP_PATHS = (
    "scripts/bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_v1.sh",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_plan.py",
    "src/ofc_regular/"
    "hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter.py",
)
_ALLOWED_OPERATIONS = [
    "metadata_identity_read",
    "metadata_token_read",
    "generation_pinned_package_download",
    "generation_match_zero_result_upload",
    "result_readback",
    "compute_delete_self_after_done",
    "bounded_safety_shutdown_on_any_worker_failure",
]
_DIGEST_INFO_PREFIX = bytes.fromhex(
    "3031300d060960864801650304020105000420"
)


def _mark_phase(phase: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase) is None:
        raise ValueError("diagnostic phase changed")
    print(
        f"OFC_STEP11_PREBOOTSTRAP phase={phase}",
        file=sys.stderr,
        flush=True,
    )


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


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size <= 1
        or path.stat().st_size > 1_000_000
    ):
        raise ValueError(f"{label} file identity changed")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ValueError(f"{label} is not exact canonical JSON")
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
        raise ValueError("package path escaped bootstrap root")
    return value


def _validate_public_key(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    _exact(
        record,
        {
            "schema",
            "algorithm",
            "exponent",
            "modulus_hex",
            "key_id",
            "private_key_present",
        },
        "controller public key",
    )
    modulus_hex = record["modulus_hex"]
    exponent = record["exponent"]
    if (
        record["schema"] != PUBLIC_KEY_SCHEMA
        or record["algorithm"] != SIGNATURE_ALGORITHM
        or record["private_key_present"] is not False
        or not isinstance(modulus_hex, str)
        or _LOWER_HEX.fullmatch(modulus_hex) is None
        or len(modulus_hex) % 2
        or type(exponent) is not int
        or exponent < 3
        or exponent >= 1 << 31
        or exponent % 2 == 0
    ):
        raise ValueError("controller public-key identity changed")
    modulus = int(modulus_hex, 16)
    if not 2_048 <= modulus.bit_length() <= 4_096 or modulus % 2 == 0:
        raise ValueError("controller RSA modulus changed")
    identity = {
        "algorithm": SIGNATURE_ALGORITHM,
        "exponent": exponent,
        "modulus_hex": modulus_hex,
    }
    if record["key_id"] != canonical_sha256(identity):
        raise ValueError("controller public-key ID changed")
    return record


def _verify_signature(
    *,
    public_key: Mapping[str, Any],
    record_type: str,
    payload: bytes,
    signature: Any,
) -> None:
    if (
        record_type not in ("authorization", "claim")
        or not isinstance(signature, str)
        or not signature
        or "=" in signature
        or re.fullmatch(r"[A-Za-z0-9_-]+", signature) is None
    ):
        raise ValueError("controller signature encoding changed")
    modulus = int(public_key["modulus_hex"], 16)
    exponent = public_key["exponent"]
    width = (modulus.bit_length() + 7) // 8
    try:
        raw = base64.b64decode(
            signature + "=" * (-len(signature) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as error:
        raise ValueError("controller signature is not base64url") from error
    if len(raw) != width or int.from_bytes(raw, "big") >= modulus:
        raise ValueError("controller signature width changed")
    encoded = pow(int.from_bytes(raw, "big"), exponent, modulus).to_bytes(
        width, "big"
    )
    digest_info = _DIGEST_INFO_PREFIX + hashlib.sha256(
        record_type.encode("ascii") + b"\0" + payload
    ).digest()
    padding_length = width - len(digest_info) - 3
    expected = (
        b"\x00\x01"
        + b"\xff" * padding_length
        + b"\x00"
        + digest_info
    )
    if padding_length < 8 or not __import__("hmac").compare_digest(
        encoded, expected
    ):
        raise ValueError("controller signature verification failed")


def _metadata_get(path: str) -> bytes:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("metadata path changed")
    request = urllib.request.Request(
        METADATA_ROOT + path,
        headers={"Metadata-Flavor": "Google"},
        method="GET",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirectHandler()
    )
    try:
        with opener.open(request, timeout=10) as response:
            raw = response.read(1_000_001)
            if (
                response.status != 200
                or response.headers.get("Metadata-Flavor") != "Google"
                or len(raw) > 1_000_000
            ):
                raise RuntimeError("metadata response identity changed")
            return raw
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"metadata returned HTTP {int(error.code)}"
        ) from None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
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


def _validate_contract_anchor(
    contract: Mapping[str, Any], public_key: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("transport contract schema changed")
    binding = contract.get("metadata_binding")
    trust = contract.get("authorization_contract")
    outer = contract.get("outer_package_manifest")
    remote = contract.get("remote_layout")
    capabilities = contract.get("capabilities")
    if not all(
        isinstance(value, Mapping)
        for value in (binding, trust, outer, remote, capabilities)
    ):
        raise ValueError("transport contract anchor is incomplete")
    if (
        contract.get("metadata_binding_sha256") != canonical_sha256(binding)
        or contract.get("outer_package_manifest_sha256")
        != canonical_sha256(outer)
        or binding.get("project") != PROJECT
        or binding.get("zone") != ZONE
        or binding.get("worker_service_account") != WORKER_SERVICE_ACCOUNT
        or binding.get("stage_id") != STAGE_ID
        or binding.get("job_id") != JOB_ID
        or binding.get("attempt_index") != ATTEMPT_INDEX
        or trust.get("signature_algorithm") != SIGNATURE_ALGORITHM
        or trust.get("controller_key_id") != public_key["key_id"]
        or trust.get("controller_public_key_sha256")
        != canonical_sha256(public_key)
        or outer.get("complete_for_direct_v1") is not True
        or remote.get("package_and_stage_prefix_disjoint") is not True
        or capabilities.get("cloud_executable") is not False
        or capabilities.get("launch_ready") is not False
        or capabilities.get("current_profile_changed") is not False
    ):
        raise ValueError("transport contract anchor changed")
    records = remote.get("package_inventory", {}).get("records")
    outer_paths = [row.get("path") for row in outer.get("objects", [])]
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= MAX_PACKAGE_RECORDS
        or [row.get("path") for row in records]
        != ["outer-manifest.json", *outer_paths]
    ):
        raise ValueError("package inventory changed")
    checked_records: list[dict[str, Any]] = []
    package_prefix = remote.get("package_prefix")
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("package record is not an object")
        path = _safe_relative(raw.get("path"))
        uri = raw.get("uri")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(package_prefix, str)
            or uri != f"{package_prefix}/{path}"
            or _sha(digest, "package record SHA-256") != digest
            or type(size) is not int
            or not 1 <= size <= 20_000_000
        ):
            raise ValueError("package record identity changed")
        checked_records.append(
            {"path": path, "uri": uri, "bytes": size, "sha256": digest}
        )
    return dict(binding), checked_records


def validate_controller_records(
    *,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    public_key: Mapping[str, Any],
    now_unix_seconds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    """Validate all signed records before requesting a metadata OAuth token."""

    key = _validate_public_key(public_key)
    binding, package_records = _validate_contract_anchor(contract, key)
    contract_sha = canonical_sha256(contract)
    auth = dict(authorization)
    _exact(
        auth,
        {
            "schema",
            "contract_sha256",
            "metadata_binding_sha256",
            "direct_stage_identity_sha256",
            "job_id",
            "attempt_index",
            "instance_name",
            "controller_key_id",
            "external_preflight_receipt_sha256",
            "allowed_operations",
            "issued_unix_seconds",
            "expires_unix_seconds",
            "nonce",
            "signature",
        },
        "controller authorization",
    )
    issued = _integer(auth["issued_unix_seconds"], "authorization issued", minimum=1)
    expires = _integer(
        auth["expires_unix_seconds"],
        "authorization expiry",
        minimum=issued + 1,
    )
    _integer(
        now_unix_seconds,
        "current time",
        minimum=issued,
        maximum=expires,
    )
    if (
        auth["schema"] != AUTHORIZATION_SCHEMA
        or auth["contract_sha256"] != contract_sha
        or auth["metadata_binding_sha256"]
        != contract["metadata_binding_sha256"]
        or auth["direct_stage_identity_sha256"]
        != contract["direct_stage_identity_sha256"]
        or auth["job_id"] != JOB_ID
        or auth["attempt_index"] != ATTEMPT_INDEX
        or auth["instance_name"] != binding["instance_name"]
        or auth["controller_key_id"] != key["key_id"]
        or _sha(
            auth["external_preflight_receipt_sha256"],
            "external preflight receipt",
        )
        != auth["external_preflight_receipt_sha256"]
        or auth["allowed_operations"] != _ALLOWED_OPERATIONS
        or _sha(auth["nonce"], "authorization nonce") != auth["nonce"]
    ):
        raise ValueError("controller authorization identity changed")
    unsigned_auth = {
        name: value for name, value in auth.items() if name != "signature"
    }
    _verify_signature(
        public_key=key,
        record_type="authorization",
        payload=canonical_bytes(unsigned_auth),
        signature=auth["signature"],
    )
    auth_sha = canonical_sha256(auth)

    signed_claim = dict(claim)
    _exact(
        signed_claim,
        {
            "schema",
            "authorization_sha256",
            "contract_sha256",
            "project",
            "project_number",
            "zone",
            "instance_name",
            "instance_id",
            "worker_service_account",
            "controller_key_id",
            "stage_id",
            "job_id",
            "attempt_index",
            "package_generations",
            "nonce",
            "signature",
        },
        "worker claim",
    )
    generations = signed_claim["package_generations"]
    expected_uris = [row["uri"] for row in package_records]
    if not isinstance(generations, Mapping) or set(generations) != set(
        expected_uris
    ):
        raise ValueError("worker claim package generations changed")
    checked_generations = {
        uri: _integer(generations[uri], "package generation", minimum=1)
        for uri in expected_uris
    }
    if (
        signed_claim["schema"] != CLAIM_SCHEMA
        or signed_claim["authorization_sha256"] != auth_sha
        or signed_claim["contract_sha256"] != contract_sha
        or signed_claim["project"] != PROJECT
        or not isinstance(signed_claim["project_number"], str)
        or not signed_claim["project_number"].isdigit()
        or signed_claim["zone"] != ZONE
        or signed_claim["instance_name"] != binding["instance_name"]
        or not isinstance(signed_claim["instance_id"], str)
        or not signed_claim["instance_id"].isdigit()
        or signed_claim["worker_service_account"] != WORKER_SERVICE_ACCOUNT
        or signed_claim["controller_key_id"] != key["key_id"]
        or signed_claim["stage_id"] != STAGE_ID
        or signed_claim["job_id"] != JOB_ID
        or signed_claim["attempt_index"] != ATTEMPT_INDEX
        or _sha(signed_claim["nonce"], "claim nonce") != signed_claim["nonce"]
    ):
        raise ValueError("worker claim identity changed")
    unsigned_claim = {
        name: value
        for name, value in signed_claim.items()
        if name != "signature"
    }
    _verify_signature(
        public_key=key,
        record_type="claim",
        payload=canonical_bytes(unsigned_claim),
        signature=signed_claim["signature"],
    )
    expected_identity = {
        "/project/project-id": PROJECT,
        "/instance/id": signed_claim["instance_id"],
        "/instance/name": binding["instance_name"],
        "/instance/zone": (
            f"projects/{signed_claim['project_number']}/zones/{ZONE}"
        ),
        "/instance/service-accounts/default/email": WORKER_SERVICE_ACCOUNT,
    }
    for path, wanted in expected_identity.items():
        if _metadata_get(path).decode("utf-8") != wanted:
            raise RuntimeError(f"metadata identity mismatch at {path}")
    scope_lines = _metadata_get(
        "/instance/service-accounts/default/scopes"
    ).decode("utf-8").splitlines()
    if scope_lines != [REQUIRED_SCOPE]:
        raise RuntimeError("metadata OAuth scope is not exactly cloud-platform")
    metadata_values = contract.get("metadata_values")
    if not isinstance(metadata_values, Mapping):
        raise ValueError("contract metadata values changed")
    for key_name, wanted in metadata_values.items():
        if (
            not isinstance(key_name, str)
            or _SAFE_METADATA_KEY.fullmatch(key_name) is None
            or not isinstance(wanted, str)
        ):
            raise ValueError("contract metadata attribute changed")
        observed = _metadata_get(
            "/instance/attributes/"
            + urllib.parse.quote(key_name, safe="")
        ).decode("utf-8")
        if observed != wanted:
            raise RuntimeError(f"custom metadata mismatch at {key_name}")
    return binding, package_records, checked_generations


def _metadata_token() -> str:
    raw = _metadata_get("/instance/service-accounts/default/token")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("metadata OAuth token is not JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"access_token", "expires_in", "token_type"}
        or not isinstance(value["access_token"], str)
        or not value["access_token"]
        or value["token_type"] != "Bearer"
        or type(value["expires_in"]) is not int
        or value["expires_in"] < 60
    ):
        raise RuntimeError("metadata OAuth token shape changed")
    return value["access_token"]


def _media_url(uri: str, generation: int) -> str:
    if not uri.startswith("gs://"):
        raise ValueError("package URI is not gs://")
    bucket, separator, name = uri[5:].partition("/")
    if not separator or not bucket or not name:
        raise ValueError("package URI is incomplete")
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?alt=media&generation={generation}"
    )


def download_bootstrap_closure(
    *,
    package_records: Sequence[Mapping[str, Any]],
    package_generations: Mapping[str, int],
    token: str,
    destination: Path,
) -> list[dict[str, Any]]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("bootstrap destination must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    by_path = {row["path"]: row for row in package_records}
    if not set(_BOOTSTRAP_PATHS).issubset(by_path):
        raise ValueError("package lacks bootstrap closure")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirectHandler()
    )
    downloaded: list[dict[str, Any]] = []
    for relative in _BOOTSTRAP_PATHS:
        record = by_path[relative]
        generation = package_generations[record["uri"]]
        if record["bytes"] > MAX_DOWNLOAD_BYTES:
            raise ValueError("bootstrap object exceeds bounded size")
        request = urllib.request.Request(
            _media_url(record["uri"], generation),
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with opener.open(request, timeout=30) as response:
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"bootstrap package GET returned HTTP {int(error.code)}"
            ) from None
        if (
            response.status != 200
            or len(raw) != record["bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("bootstrap object identity changed")
        target = destination / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if relative.startswith("scripts/"):
            target.chmod(0o700)
        downloaded.append(
            {
                "path": relative,
                "generation": generation,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
        )
    return downloaded


def run(
    *,
    contract_path: Path,
    authorization_path: Path,
    claim_path: Path,
    public_key_path: Path,
    bootstrap_root: Path,
    fresh_root: Path,
    now_unix_seconds: int | None = None,
) -> int:
    _mark_phase("read_controller_records")
    contract = _read_canonical(contract_path, "transport contract")
    authorization = _read_canonical(
        authorization_path, "controller authorization"
    )
    claim = _read_canonical(claim_path, "worker claim")
    public_key = _read_canonical(public_key_path, "controller public key")
    _mark_phase("validate_controller_records")
    _binding, package_records, generations = validate_controller_records(
        contract=contract,
        authorization=authorization,
        claim=claim,
        public_key=public_key,
        now_unix_seconds=(
            int(time.time())
            if now_unix_seconds is None
            else now_unix_seconds
        ),
    )
    _mark_phase("metadata_token")
    token = _metadata_token()
    _mark_phase("download_bootstrap_closure")
    download_bootstrap_closure(
        package_records=package_records,
        package_generations=generations,
        token=token,
        destination=bootstrap_root,
    )
    _mark_phase("start_inner_transport")
    environment = dict(os.environ)
    environment["OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER"] = "1"
    completed = subprocess.run(
        [
            str(
                bootstrap_root
                / "scripts"
                / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_v1.sh"
            ),
            "authorized-worker",
            str(contract_path),
            str(authorization_path),
            str(claim_path),
            str(public_key_path),
            str(fresh_root),
        ],
        cwd=bootstrap_root,
        env=environment,
        check=False,
    )
    _mark_phase(
        "inner_transport_exit_zero"
        if completed.returncode == 0
        else "inner_transport_exit_nonzero"
    )
    return int(completed.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--controller-public-key", type=Path, required=True)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run(
        contract_path=args.contract,
        authorization_path=args.authorization,
        claim_path=args.claim,
        public_key_path=args.controller_public_key,
        bootstrap_root=args.bootstrap_root,
        fresh_root=args.fresh_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
