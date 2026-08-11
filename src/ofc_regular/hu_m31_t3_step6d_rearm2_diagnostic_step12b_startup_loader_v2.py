"""Build the stdlib-only Step12b VM startup loader.

The generated source is placed directly in the GCE ``startup-script``
metadata value.  It has no repository or site-packages dependency.  Before
the first worker-token request it validates the deployment self digest, the
public RSA trust anchor, the signed authorization, and the exact role-local
generation-pinned source manifest.
"""

from __future__ import annotations

import hashlib

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)


STARTUP_LOADER_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_stdlib_startup_loader_v2"
)
VM_PREBOOTSTRAP_MODULE = (
    "ofc_regular.hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_vm_prebootstrap_v2"
)

_TEMPLATE = r'''#!/usr/bin/python3
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

LOADER_SCHEMA = "__LOADER_SCHEMA__"
MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_role_bootstrap_manifest_v2"
)
AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_external_authorization_v2"
)
PUBLIC_KEY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_controller_rsa_public_key_v1"
)
PUBLIC_KEY_ALGORITHM = "RSASSA-PKCS1-v1_5-SHA256"
RUNTIME_BUNDLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_runtime_source_bundle_v2"
)
DOWNLOAD_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_download_receipt_v2"
)
PROJECT = "ofc-solver-485418"
BUCKET = "pokerhu-ofc-solver-485418-training"
DIRECT_NAMESPACE = "hu-m31-r2diag-direct-v2"
SOURCE_NAMESPACE = DIRECT_NAMESPACE + "/bootstrap-sources"
METADATA_BASE = "http://metadata.google.internal/computeMetadata/v1"
FAILURE_MARKER_PREFIX = "OFC_STEP12N_WORKER_FAILURE_V1 "
FAILURE_MARKER_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_worker_failure_marker_v1"
)
FAILURE_HOLD_SECONDS = 600
CURRENT_STAGE = "startup_loader_entry"
RUNTIME_OBJECT_PATH = "runtime_source_bundle.json"
CANDIDATE_OBJECT_PATH = "candidate_payload_contract.json"
REFERENCE_OBJECT_PATH = "reference_payload_contract.json"
REQUIRED_SOURCE_PATHS = __REQUIRED_PATHS__
VM_PREBOOTSTRAP_MODULE = "__VM_PREBOOTSTRAP_MODULE__"
RUNTIME_ROOT = Path("/opt/ofc-step12b-runtime")
MAX_SOURCE_OBJECT_BYTES = 1_048_576
MAX_METADATA_BYTES = 524_288
REQUEST_TIMEOUT_SECONDS = 10
ALLOWED_OPERATIONS = [
    "metadata_identity_read",
    "bounded_host_prerequisite_install",
    "metadata_token_read",
    "generation_pinned_bootstrap_source_download",
    "generation_pinned_immutable_package_download",
    "local_immutable_payload_execute",
    "generation_match_zero_deployment_result_upload",
    "deployment_result_readback",
    "compute_delete_self_after_done",
    "bounded_safety_shutdown_on_any_worker_failure",
]
AUTHORIZATION_FIELDS = {
    "schema",
    "run_nonce",
    "deployment_contract_sha256",
    "run_identity_sha256",
    "direct_stage_identity_sha256",
    "payload_binding_sha256",
    "payload_contract_sha256",
    "payload_contract_sha256s",
    "stage_id",
    "run_name",
    "external_job_id",
    "inner_job_id",
    "source_role",
    "attempt_index",
    "max_attempts",
    "instance_name",
    "external_result_layout_sha256",
    "immutable_source_hashes",
    "immutable_source_hashes_sha256",
    "controller_key_id",
    "controller_public_key_sha256",
    "controller_service_account",
    "package_inventory_records_sha256",
    "package_generations_sha256",
    "package_object_count",
    "bootstrap_source_binding",
    "external_preflight_receipt_schema",
    "external_preflight_receipt_sha256",
    "preflight_observation_receipts_sha256",
    "preflight_source_hashes_sha256",
    "external_preflight_receipt_validation",
    "allowed_operations",
    "issued_unix_seconds",
    "expires_unix_seconds",
    "nonce",
    "signature",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9_-]+$")
_SOURCE_PATH = re.compile(r"^ofc_regular/[a-z0-9_]+\.py$")
_DIGEST_INFO_PREFIX = bytes.fromhex(
    "3031300d060960864801650304020105000420"
)


def canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(label + " fields changed")


def checked_sha(value, label):
    if (
        not isinstance(value, str)
        or _SHA.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(label + " digest changed")
    return value


def canonical_object(raw, label):
    if isinstance(raw, str):
        try:
            encoded = raw.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(label + " is not ASCII JSON") from error
    elif isinstance(raw, bytes):
        encoded = raw
    else:
        raise ValueError(label + " is not JSON bytes")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " is not JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != encoded:
        raise ValueError(label + " is not canonical JSON")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_metadata_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirect()
)
_https_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirect()
)


def read_url(opener, request, max_bytes):
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read(max_bytes + 1)
            status = response.status
            final_url = response.geturl()
            headers = response.headers
    except urllib.error.HTTPError as error:
        raw = error.read(max_bytes + 1)
        status = error.code
        final_url = error.geturl()
        headers = error.headers
    if final_url != request.full_url:
        raise RuntimeError("redirect was attempted")
    if len(raw) > max_bytes:
        raise ValueError("HTTP response escaped size bound")
    return status, headers, raw


def metadata_text(path):
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or ".." in path.split("/")
    ):
        raise ValueError("metadata path changed")
    url = METADATA_BASE + path
    request = urllib.request.Request(
        url, headers={"Metadata-Flavor": "Google"}, method="GET"
    )
    status, headers, raw = read_url(
        _metadata_opener, request, MAX_METADATA_BYTES
    )
    if status != 200 or headers.get("Metadata-Flavor") != "Google":
        raise RuntimeError("metadata response changed")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("metadata response is not UTF-8") from error


def metadata_attribute(key):
    return metadata_text(
        "/instance/attributes/" + urllib.parse.quote(key, safe="")
    )


def worker_access_token():
    token = json.loads(
        metadata_text("/instance/service-accounts/default/token")
    )
    if (
        not isinstance(token, dict)
        or set(token) != {"access_token", "expires_in", "token_type"}
        or not isinstance(token["access_token"], str)
        or not token["access_token"]
        or any(character.isspace() for character in token["access_token"])
        or len(token["access_token"]) > 8192
        or token["token_type"] != "Bearer"
        or type(token["expires_in"]) is not int
        or token["expires_in"] < 60
    ):
        raise ValueError("metadata OAuth token changed")
    return token["access_token"]


def validate_public_key(value):
    exact(
        value,
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
    modulus_hex = value["modulus_hex"]
    exponent = value["exponent"]
    if (
        value["schema"] != PUBLIC_KEY_SCHEMA
        or value["algorithm"] != PUBLIC_KEY_ALGORITHM
        or value["private_key_present"] is not False
        or not isinstance(modulus_hex, str)
        or re.fullmatch(r"[0-9a-f]+", modulus_hex) is None
        or len(modulus_hex) % 2
        or type(exponent) is not int
        or exponent < 3
        or exponent > (1 << 31) - 1
        or exponent % 2 == 0
    ):
        raise ValueError("controller public key changed")
    modulus = int(modulus_hex, 16)
    identity = {
        "algorithm": PUBLIC_KEY_ALGORITHM,
        "exponent": exponent,
        "modulus_hex": modulus_hex,
    }
    if (
        not 2048 <= modulus.bit_length() <= 4096
        or modulus % 2 == 0
        or value["key_id"] != canonical_sha256(identity)
    ):
        raise ValueError("controller public key identity changed")
    return value


def verify_signature(public_key, record_type, unsigned, signature):
    if (
        record_type not in {"authorization", "claim"}
        or not isinstance(signature, str)
        or not signature
        or _SIGNATURE.fullmatch(signature) is None
        or "=" in signature
    ):
        return False
    try:
        raw = base64.b64decode(
            signature + "=" * (-len(signature) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError):
        return False
    modulus = int(public_key["modulus_hex"], 16)
    width = (modulus.bit_length() + 7) // 8
    if len(raw) != width:
        return False
    number = int.from_bytes(raw, "big")
    if number >= modulus:
        return False
    decoded = pow(number, public_key["exponent"], modulus).to_bytes(
        width, "big"
    )
    digest = hashlib.sha256(
        record_type.encode("ascii")
        + b"\0"
        + canonical_bytes(unsigned)
    ).digest()
    digest_info = _DIGEST_INFO_PREFIX + digest
    padding_length = width - len(digest_info) - 3
    expected = (
        b"\x00\x01"
        + b"\xff" * padding_length
        + b"\x00"
        + digest_info
    )
    return padding_length >= 8 and decoded == expected


def validate_manifest(value, deployment_sha, external_job_id, source_role):
    manifest = dict(value)
    exact(
        manifest,
        {
            "schema",
            "deployment_contract_sha256",
            "source_plan_sha256",
            "source_provision_receipt_sha256",
            "source_prefix",
            "external_job_id",
            "inner_job_id",
            "source_role",
            "object_count",
            "objects",
            "objects_sha256",
            "runtime_source_bundle_sha256",
            "all_generation_bound",
            "one_role_payload_only",
            "opponent_role_payload_present",
            "repo_import_permitted",
            "site_packages_import_permitted",
            "role_manifest_sha256",
        },
        "role bootstrap manifest",
    )
    supplied = checked_sha(
        manifest.pop("role_manifest_sha256"), "role manifest"
    )
    if canonical_sha256(manifest) != supplied:
        raise ValueError("role manifest digest changed")
    prefix = (
        "gs://" + BUCKET + "/" + SOURCE_NAMESPACE + "/" + deployment_sha
    )
    objects = manifest["objects"]
    role_path = (
        CANDIDATE_OBJECT_PATH
        if source_role == "candidate"
        else REFERENCE_OBJECT_PATH
    )
    expected = [
        ("shared_runtime_source_bundle", RUNTIME_OBJECT_PATH),
        (source_role + "_role_payload_contract", role_path),
    ]
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["deployment_contract_sha256"] != deployment_sha
        or manifest["source_prefix"] != prefix
        or manifest["external_job_id"] != external_job_id
        or manifest["source_role"] != source_role
        or not isinstance(manifest["inner_job_id"], str)
        or not manifest["inner_job_id"]
        or manifest["object_count"] != 2
        or not isinstance(objects, list)
        or len(objects) != 2
        or manifest["objects_sha256"] != canonical_sha256(objects)
        or manifest["all_generation_bound"] is not True
        or manifest["one_role_payload_only"] is not True
        or manifest["opponent_role_payload_present"] is not False
        or manifest["repo_import_permitted"] is not False
        or manifest["site_packages_import_permitted"] is not False
    ):
        raise ValueError("role manifest boundary changed")
    checked_sha(manifest["source_plan_sha256"], "source plan")
    checked_sha(
        manifest["source_provision_receipt_sha256"],
        "source provision receipt",
    )
    checked_sha(
        manifest["runtime_source_bundle_sha256"], "runtime bundle"
    )
    for row, expected_identity in zip(objects, expected):
        exact(
            row,
            {
                "kind",
                "path",
                "uri",
                "bytes",
                "sha256",
                "generation",
                "created",
                "readback_verified",
            },
            "manifest object",
        )
        kind, path = expected_identity
        if (
            row["kind"] != kind
            or row["path"] != path
            or row["uri"] != prefix + "/" + path
            or type(row["bytes"]) is not int
            or not 1 <= row["bytes"] <= MAX_SOURCE_OBJECT_BYTES
            or type(row["generation"]) is not int
            or row["generation"] <= 0
            or row["created"] is not True
            or row["readback_verified"] is not True
        ):
            raise ValueError("manifest object changed")
        checked_sha(row["sha256"], "manifest object")
    return {**manifest, "role_manifest_sha256": supplied}


def validate_deployment_and_authorization(
    deployment, manifest, public_key, authorization, run_nonce,
    external_job_id, source_role
):
    deployment_body = dict(deployment)
    deployment_sha = checked_sha(
        deployment_body.pop("deployment_contract_sha256"),
        "deployment",
    )
    if canonical_sha256(deployment_body) != deployment_sha:
        raise ValueError("deployment digest changed")
    if (
        deployment.get("run_nonce") != run_nonce
        or deployment.get("controller_key_id") != public_key["key_id"]
        or deployment.get("controller_public_key_sha256")
        != canonical_sha256(public_key)
        or external_job_id not in deployment.get("selected_job_ids", [])
    ):
        raise ValueError("deployment role identity changed")
    position = deployment["selected_job_ids"].index(external_job_id)
    if deployment["source_roles"][position] != source_role:
        raise ValueError("deployment source role changed")
    checked_manifest = validate_manifest(
        manifest, deployment_sha, external_job_id, source_role
    )
    exact(
        authorization,
        AUTHORIZATION_FIELDS,
        "signed external authorization",
    )
    auth = dict(authorization)
    signature = auth.pop("signature", None)
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("deployment_contract_sha256") != deployment_sha
        or authorization.get("run_nonce") != run_nonce
        or authorization.get("external_job_id") != external_job_id
        or authorization.get("inner_job_id")
        != checked_manifest["inner_job_id"]
        or authorization.get("source_role") != source_role
        or authorization.get("controller_key_id") != public_key["key_id"]
        or authorization.get("controller_public_key_sha256")
        != canonical_sha256(public_key)
        or type(authorization.get("issued_unix_seconds")) is not int
        or type(authorization.get("expires_unix_seconds")) is not int
        or not (
            authorization["issued_unix_seconds"]
            <= int(time.time())
            < authorization["expires_unix_seconds"]
        )
        or authorization.get("allowed_operations") != ALLOWED_OPERATIONS
        or not verify_signature(public_key, "authorization", auth, signature)
    ):
        raise ValueError("signed authorization changed")
    binding = authorization.get("bootstrap_source_binding")
    if not isinstance(binding, dict):
        raise ValueError("signed source binding is missing")
    exact(
        binding,
        {
            "bootstrap_source_content_binding_sha256",
            "source_plan_sha256",
            "source_provision_receipt_sha256",
            "source_prefix",
            "source_generations_sha256",
            "source_object_count",
            "role_source_generations",
            "role_source_generations_sha256",
            "role_manifest_sha256",
            "role_manifest_object_count",
            "role_manifest_objects",
            "role_manifest_objects_sha256",
        },
        "signed role source binding",
    )
    source_content = deployment.get("bootstrap_source_content_binding")
    if not isinstance(source_content, dict):
        raise ValueError("deployment source content binding is missing")
    if (
        binding.get("bootstrap_source_content_binding_sha256")
        != source_content.get(
            "bootstrap_source_content_binding_sha256"
        )
        or binding.get("source_plan_sha256")
        != checked_manifest["source_plan_sha256"]
        or binding.get("source_provision_receipt_sha256")
        != checked_manifest["source_provision_receipt_sha256"]
        or binding.get("source_prefix")
        != checked_manifest["source_prefix"]
        or binding.get("role_manifest_sha256")
        != checked_manifest["role_manifest_sha256"]
        or binding.get("role_manifest_objects_sha256")
        != checked_manifest["objects_sha256"]
        or binding.get("role_manifest_object_count") != 2
        or binding.get("role_manifest_objects")
        != checked_manifest["objects"]
        or binding.get("role_source_generations")
        != {
            row["uri"]: row["generation"]
            for row in checked_manifest["objects"]
        }
        or binding.get("role_source_generations_sha256")
        != canonical_sha256(
            {
                row["uri"]: row["generation"]
                for row in checked_manifest["objects"]
            }
        )
    ):
        raise ValueError("signed source binding changed")
    runtime_summary = source_content.get("runtime_source_bundle", {})
    role_summaries = source_content.get("role_payload_contracts", [])
    if (
        runtime_summary.get("bytes")
        != checked_manifest["objects"][0]["bytes"]
        or runtime_summary.get("sha256")
        != checked_manifest["objects"][0]["sha256"]
        or runtime_summary.get("runtime_source_bundle_sha256")
        != checked_manifest["runtime_source_bundle_sha256"]
        or not isinstance(role_summaries, list)
        or len(role_summaries) != 2
        or role_summaries[position].get("bytes")
        != checked_manifest["objects"][1]["bytes"]
        or role_summaries[position].get("sha256")
        != checked_manifest["objects"][1]["sha256"]
        or role_summaries[position].get("source_role") != source_role
        or role_summaries[position].get("inner_job_id")
        != checked_manifest["inner_job_id"]
    ):
        raise ValueError("manifest escaped deployment content binding")
    return checked_manifest


def generation_pinned_get(row, token):
    uri = row["uri"]
    prefix = "gs://" + BUCKET + "/"
    if not uri.startswith(prefix):
        raise ValueError("source object escaped bucket")
    name = uri[len(prefix):]
    query = urllib.parse.urlencode(
        {"alt": "media", "generation": str(row["generation"])}
    )
    url = (
        "https://storage.googleapis.com/download/storage/v1/b/"
        + urllib.parse.quote(BUCKET, safe="")
        + "/o/"
        + urllib.parse.quote(name, safe="")
        + "?"
        + query
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer " + token},
        method="GET",
    )
    status, _headers, raw = read_url(
        _https_opener, request, row["bytes"]
    )
    if (
        status != 200
        or len(raw) != row["bytes"]
        or hashlib.sha256(raw).hexdigest() != row["sha256"]
    ):
        raise RuntimeError("generation-pinned source download changed")
    return raw


def validate_runtime_bundle(raw, expected_bundle_sha):
    bundle = canonical_object(raw, "runtime source bundle")
    body = dict(bundle)
    supplied = checked_sha(
        body.pop("runtime_source_bundle_sha256"), "runtime source bundle"
    )
    records = bundle.get("records")
    if (
        bundle.get("schema") != RUNTIME_BUNDLE_SCHEMA
        or canonical_sha256(body) != supplied
        or supplied != expected_bundle_sha
        or bundle.get("file_count") != len(REQUIRED_SOURCE_PATHS)
        or bundle.get("paths") != list(REQUIRED_SOURCE_PATHS)
        or not isinstance(records, list)
        or len(records) != len(REQUIRED_SOURCE_PATHS)
        or bundle.get("records_sha256") != canonical_sha256(records)
        or bundle.get("regular_files_only") is not True
        or bundle.get("symlink_count") != 0
        or bundle.get("repo_import_permitted") is not False
        or bundle.get("site_packages_import_permitted") is not False
        or bundle.get("private_material_present") is not False
    ):
        raise ValueError("runtime source bundle changed")
    for row, expected_path in zip(records, REQUIRED_SOURCE_PATHS):
        exact(
            row,
            {"path", "kind", "mode", "bytes", "sha256", "source"},
            "runtime source record",
        )
        source = row["source"]
        if (
            row["path"] != expected_path
            or _SOURCE_PATH.fullmatch(expected_path) is None
            or row["kind"] != "regular_file"
            or row["mode"] != "0644"
            or not isinstance(source, str)
            or not source
            or "\x00" in source
            or "\r" in source
            or row["bytes"] != len(source.encode("utf-8"))
            or row["sha256"]
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
        ):
            raise ValueError("runtime source record changed")
        compile(source, expected_path, "exec", dont_inherit=True)
    return bundle


def materialize_runtime(bundle, deployment_sha, source_role):
    if RUNTIME_ROOT.exists():
        if not RUNTIME_ROOT.is_dir() or RUNTIME_ROOT.is_symlink():
            raise ValueError("runtime root changed")
    else:
        if not RUNTIME_ROOT.parent.is_dir() or RUNTIME_ROOT.parent.is_symlink():
            raise ValueError("runtime root parent changed")
        RUNTIME_ROOT.mkdir(mode=0o700)
    os.chmod(RUNTIME_ROOT, 0o700)
    destination = RUNTIME_ROOT / (
        deployment_sha[:16] + "-" + source_role
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("runtime destination must be fresh")
    destination.mkdir(mode=0o700)
    resolved = destination.resolve()
    written = []
    for row in bundle["records"]:
        relative = PurePosixPath(row["path"])
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            target.exists()
            or target.is_symlink()
            or target.parent.is_symlink()
            or resolved not in target.resolve().parents
        ):
            raise ValueError("runtime source path escaped destination")
        raw = row["source"].encode("utf-8")
        with target.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o644)
        written.append(
            {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
    body = {
        "schema": RUNTIME_BUNDLE_SCHEMA,
        "status": "runtime_source_bundle_materialized",
        "runtime_source_bundle_sha256": bundle[
            "runtime_source_bundle_sha256"
        ],
        "file_count": len(written),
        "records": written,
        "records_sha256": canonical_sha256(written),
        "symlink_count": 0,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
    }
    return destination, {**body, "receipt_sha256": canonical_sha256(body)}


def build_download_receipt(manifest, materialization, payload):
    records = [
        {
            "kind": row["kind"],
            "path": row["path"],
            "uri": row["uri"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "generation": row["generation"],
            "downloaded": True,
            "readback_verified": True,
        }
        for row in manifest["objects"]
    ]
    body = {
        "schema": DOWNLOAD_RECEIPT_SCHEMA,
        "deployment_contract_sha256": manifest[
            "deployment_contract_sha256"
        ],
        "source_plan_sha256": manifest["source_plan_sha256"],
        "source_provision_receipt_sha256": manifest[
            "source_provision_receipt_sha256"
        ],
        "role_manifest_sha256": manifest["role_manifest_sha256"],
        "external_job_id": manifest["external_job_id"],
        "inner_job_id": manifest["inner_job_id"],
        "source_role": manifest["source_role"],
        "object_count": 2,
        "records": records,
        "records_sha256": canonical_sha256(records),
        "runtime_source_bundle_sha256": manifest[
            "runtime_source_bundle_sha256"
        ],
        "runtime_source_materialization_receipt_sha256": materialization[
            "receipt_sha256"
        ],
        "payload_contract_sha256": hashlib.sha256(
            canonical_bytes(payload)
        ).hexdigest(),
        "generation_pinned_get_only": True,
        "opponent_role_payload_download_count": 0,
        "downloaded_before_full_prebootstrap_import": True,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def ensure_host_prerequisites():
    probe = subprocess.run(
        [sys.executable, "-c", "import ensurepip,venv"],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if probe.returncode == 0:
        return
    subprocess.run(
        [
            "/usr/bin/timeout",
            "300",
            "/usr/bin/env",
            "DEBIAN_FRONTEND=noninteractive",
            "/usr/bin/apt-get",
            "update",
            "-o",
            "Acquire::Retries=2",
        ],
        check=True,
        capture_output=True,
        timeout=330,
    )
    subprocess.run(
        [
            "/usr/bin/timeout",
            "300",
            "/usr/bin/env",
            "DEBIAN_FRONTEND=noninteractive",
            "/usr/bin/apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "ca-certificates",
            "python3",
            "python3-venv",
        ],
        check=True,
        capture_output=True,
        timeout=330,
    )
    subprocess.run(
        [sys.executable, "-c", "import ensurepip,venv"],
        check=True,
        capture_output=True,
        timeout=30,
    )


def main():
    global CURRENT_STAGE
    CURRENT_STAGE = "deployment_metadata_read"
    deployment = canonical_object(
        metadata_attribute("ofc-step12b-deployment-contract"),
        "deployment contract",
    )
    CURRENT_STAGE = "role_manifest_metadata_read"
    manifest = canonical_object(
        metadata_attribute("ofc-step12b-role-bootstrap-manifest"),
        "role bootstrap manifest",
    )
    CURRENT_STAGE = "public_key_metadata_read"
    public_key = validate_public_key(
        canonical_object(
            metadata_attribute("ofc-step12b-controller-public-key"),
            "controller public key",
        )
    )
    CURRENT_STAGE = "authorization_metadata_read"
    authorization = canonical_object(
        metadata_attribute("ofc-step12b-external-authorization"),
        "external authorization",
    )
    CURRENT_STAGE = "identity_metadata_read"
    run_nonce = metadata_attribute("ofc-step12b-run-nonce")
    external_job_id = metadata_attribute("ofc-step12b-external-job-id")
    source_role = metadata_attribute("ofc-step12b-source-role")
    CURRENT_STAGE = "deployment_authorization_validate"
    checked_manifest = validate_deployment_and_authorization(
        deployment,
        manifest,
        public_key,
        authorization,
        run_nonce,
        external_job_id,
        source_role,
    )
    CURRENT_STAGE = "host_prerequisite_install"
    ensure_host_prerequisites()
    CURRENT_STAGE = "worker_access_token_read"
    token = worker_access_token()
    CURRENT_STAGE = "runtime_source_download"
    runtime_raw = generation_pinned_get(
        checked_manifest["objects"][0], token
    )
    CURRENT_STAGE = "payload_contract_download"
    payload_raw = generation_pinned_get(
        checked_manifest["objects"][1], token
    )
    del token
    CURRENT_STAGE = "runtime_source_validate"
    runtime_bundle = validate_runtime_bundle(
        runtime_raw,
        checked_manifest["runtime_source_bundle_sha256"],
    )
    CURRENT_STAGE = "payload_contract_validate"
    payload = canonical_object(payload_raw, "selected payload contract")
    if hashlib.sha256(payload_raw).hexdigest() != checked_manifest[
        "objects"
    ][1]["sha256"]:
        raise ValueError("selected payload escaped manifest")
    CURRENT_STAGE = "runtime_source_materialize"
    destination, materialization = materialize_runtime(
        runtime_bundle,
        checked_manifest["deployment_contract_sha256"],
        checked_manifest["source_role"],
    )
    CURRENT_STAGE = "download_receipt_build"
    receipt = build_download_receipt(
        checked_manifest, materialization, payload
    )
    sys.path.insert(0, str(destination))
    CURRENT_STAGE = "vm_prebootstrap_import"
    module = importlib.import_module(VM_PREBOOTSTRAP_MODULE)
    CURRENT_STAGE = "downloaded_worker_execute"
    result = module.run_downloaded_entrypoint(
        checked_manifest, payload, receipt
    )
    CURRENT_STAGE = "downloaded_worker_complete"
    print(canonical_bytes(result).decode("ascii"), flush=True)
    return 0


def self_test():
    if not sys.flags.isolated or "site" in sys.modules:
        raise RuntimeError("startup loader is not isolated")
    for entry in sys.path:
        lowered = str(entry).lower()
        if "site-packages" in lowered:
            raise RuntimeError("startup loader saw site-packages")
    probe = {"loader": LOADER_SCHEMA, "paths": list(REQUIRED_SOURCE_PATHS)}
    if json.loads(canonical_bytes(probe)) != probe:
        raise RuntimeError("startup loader canonical JSON self-test failed")
    print(
        canonical_bytes(
            {
                "schema": LOADER_SCHEMA,
                "status": "stdlib_isolated_self_test_passed",
                "required_source_file_count": len(REQUIRED_SOURCE_PATHS),
                "repo_imported": False,
                "site_packages_imported": False,
                "network_called": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    if sys.argv[1:]:
        raise ValueError("startup loader arguments changed")
    try:
        exit_code = main()
    except BaseException as error:
        traceback.print_exc()
        marker = {
            "schema": FAILURE_MARKER_SCHEMA,
            "status": "worker_failed_before_done",
            "stage": CURRENT_STAGE,
            "exception_type": type(error).__name__,
        }
        print(
            FAILURE_MARKER_PREFIX
            + json.dumps(
                marker,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            flush=True,
        )
        # Keep the failed Spot VM alive long enough for the controller to
        # read the sanitized marker from serial port 1.  The previous
        # immediate shutdown combined with instanceTerminationAction=DELETE
        # erased the only useful failure evidence.  The bounded shutdown
        # below remains a last-resort cost guard if controller cleanup fails.
        time.sleep(FAILURE_HOLD_SECONDS)
        try:
            subprocess.run(
                ["/sbin/shutdown", "-h", "now"],
                check=False,
                timeout=120,
                capture_output=True,
            )
        except BaseException:
            traceback.print_exc()
        raise
    raise SystemExit(exit_code)
'''


def build_startup_loader_source() -> str:
    """Return one deterministic metadata-safe Python startup script."""

    source = (
        _TEMPLATE.replace("__LOADER_SCHEMA__", STARTUP_LOADER_SCHEMA)
        .replace(
            "__REQUIRED_PATHS__",
            repr(tuple(sorted(bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS))),
        )
        .replace("__VM_PREBOOTSTRAP_MODULE__", VM_PREBOOTSTRAP_MODULE)
    )
    if (
        "\r" in source
        or "\x00" in source
        or not source.startswith("#!/usr/bin/python3\n")
    ):
        raise AssertionError("startup loader source changed")
    compile(source, "<step12b-startup-loader-v2>", "exec", dont_inherit=True)
    encoded = source.encode("utf-8")
    if len(encoded) > 262_144:
        raise ValueError("startup loader escaped metadata value limit")
    return source


def startup_loader_sha256() -> str:
    return hashlib.sha256(
        build_startup_loader_source().encode("utf-8")
    ).hexdigest()


__all__ = [
    "STARTUP_LOADER_SCHEMA",
    "VM_PREBOOTSTRAP_MODULE",
    "build_startup_loader_source",
    "startup_loader_sha256",
]
