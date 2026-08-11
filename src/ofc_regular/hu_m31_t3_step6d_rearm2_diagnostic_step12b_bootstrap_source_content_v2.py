"""Content-only bootstrap-source identity used before deployment hashing.

This module deliberately knows nothing about a deployment URI or deployment
digest.  It can therefore be consumed by the deployment builder without a
digest cycle.  Final source prefixes and generations are bound later.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)


RUNTIME_SOURCE_BUNDLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_runtime_source_bundle_v2"
)
CONTENT_BINDING_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_source_content_binding_v2"
)
PREFIX_DERIVATION_RULE = (
    "gs_bucket_direct_v2_bootstrap_sources_slash_deployment_sha256"
)
RUNTIME_SOURCE_OBJECT_PATH = "runtime_source_bundle.json"
CANDIDATE_PAYLOAD_OBJECT_PATH = "candidate_payload_contract.json"
REFERENCE_PAYLOAD_OBJECT_PATH = "reference_payload_contract.json"
MAX_RUNTIME_SOURCE_FILES = 16
MAX_SOURCE_OBJECT_BYTES = 1_048_576

REQUIRED_RUNTIME_SOURCE_PATHS = frozenset(
    {
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "canary_gce_transport_10c2_v1.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_vm_prebootstrap_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_bootstrap_source_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_bootstrap_source_content_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_alias_bridge_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_pair_release_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_vm_metadata_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_external_authorization_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_deployment_contract_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_phase2_iam_plan_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_run_scoped_controller_sa_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_rest_iam_admin_v1.py"
        ),
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATH = re.compile(r"^ofc_regular/[a-z0-9_]+\.py$")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def build_runtime_source_bundle(
    runtime_source_files: Mapping[str, str],
) -> dict[str, Any]:
    if (
        not isinstance(runtime_source_files, Mapping)
        or set(runtime_source_files) != REQUIRED_RUNTIME_SOURCE_PATHS
        or len(runtime_source_files) > MAX_RUNTIME_SOURCE_FILES
    ):
        raise ValueError("bootstrap runtime source closure changed")
    records = []
    for path in sorted(runtime_source_files):
        source = runtime_source_files[path]
        if (
            _SOURCE_PATH.fullmatch(path) is None
            or ".." in path.split("/")
            or not isinstance(source, str)
            or not source
            or "\x00" in source
            or "\r" in source
        ):
            raise ValueError("bootstrap runtime source changed")
        try:
            compile(source, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as error:
            raise ValueError("bootstrap runtime source did not compile") from error
        raw = source.encode("utf-8")
        records.append(
            {
                "path": path,
                "kind": "regular_file",
                "mode": "0644",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": source,
            }
        )
    body = {
        "schema": RUNTIME_SOURCE_BUNDLE_SCHEMA,
        "file_count": len(records),
        "paths": [row["path"] for row in records],
        "records": records,
        "records_sha256": canonical_sha256(records),
        "regular_files_only": True,
        "symlink_count": 0,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
        "private_material_present": False,
    }
    result = {
        **body,
        "runtime_source_bundle_sha256": canonical_sha256(body),
    }
    if len(canonical_bytes(result)) > MAX_SOURCE_OBJECT_BYTES:
        raise ValueError("runtime source bundle escaped source object limit")
    return result


def validate_runtime_source_bundle(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = copy.deepcopy(dict(value))
    records = bundle.get("records")
    if not isinstance(records, list):
        raise ValueError("runtime source bundle records changed")
    sources = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("runtime source bundle record changed")
        path = row.get("path")
        source = row.get("source")
        if not isinstance(path, str) or not isinstance(source, str):
            raise ValueError("runtime source bundle record changed")
        sources[path] = source
    expected = build_runtime_source_bundle(sources)
    if bundle != expected:
        raise ValueError("runtime source bundle changed")
    return expected


def _role_summary(
    *,
    kind: str,
    path: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    checked = payload_transport.validate_job_contract(payload)
    raw = canonical_bytes(checked)
    return {
        "kind": kind,
        "path": path,
        "source_role": checked["metadata_binding"]["source_role"],
        "inner_job_id": checked["metadata_binding"]["job_id"],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_bootstrap_source_content_binding(
    *,
    runtime_source_files: Mapping[str, str],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = build_runtime_source_bundle(runtime_source_files)
    bundle_raw = canonical_bytes(bundle)
    role_payloads = [
        _role_summary(
            kind="candidate_role_payload_contract",
            path=CANDIDATE_PAYLOAD_OBJECT_PATH,
            payload=candidate_payload_contract,
        ),
        _role_summary(
            kind="reference_role_payload_contract",
            path=REFERENCE_PAYLOAD_OBJECT_PATH,
            payload=reference_payload_contract,
        ),
    ]
    if [row["source_role"] for row in role_payloads] != [
        "candidate",
        "reference",
    ]:
        raise ValueError("bootstrap role payload mapping changed")
    runtime_summary = {
        "kind": "shared_runtime_source_bundle",
        "path": RUNTIME_SOURCE_OBJECT_PATH,
        "bytes": len(bundle_raw),
        "sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "runtime_source_bundle_sha256": bundle[
            "runtime_source_bundle_sha256"
        ],
        "records_sha256": bundle["records_sha256"],
        "file_count": bundle["file_count"],
    }
    content_summaries = [runtime_summary, *role_payloads]
    body = {
        "schema": CONTENT_BINDING_SCHEMA,
        "prefix_derivation_rule": PREFIX_DERIVATION_RULE,
        "source_object_count": 3,
        "source_object_paths": [
            RUNTIME_SOURCE_OBJECT_PATH,
            CANDIDATE_PAYLOAD_OBJECT_PATH,
            REFERENCE_PAYLOAD_OBJECT_PATH,
        ],
        "runtime_source_bundle": runtime_summary,
        "role_payload_contracts": role_payloads,
        "role_payload_contracts_sha256": canonical_sha256(role_payloads),
        "content_summaries_sha256": canonical_sha256(content_summaries),
        "canonical_json_objects": True,
        "regular_source_files_only": True,
        "symlink_count": 0,
        "final_source_prefix_present": False,
        "generation_present": False,
    }
    return {
        **body,
        "bootstrap_source_content_binding_sha256": canonical_sha256(body),
    }


def validate_bootstrap_source_content_binding(
    value: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
) -> dict[str, Any]:
    binding = copy.deepcopy(dict(value))
    _exact(
        binding,
        {
            "schema",
            "prefix_derivation_rule",
            "source_object_count",
            "source_object_paths",
            "runtime_source_bundle",
            "role_payload_contracts",
            "role_payload_contracts_sha256",
            "content_summaries_sha256",
            "canonical_json_objects",
            "regular_source_files_only",
            "symlink_count",
            "final_source_prefix_present",
            "generation_present",
            "bootstrap_source_content_binding_sha256",
        },
        "bootstrap source content binding",
    )
    supplied = _sha(
        binding.pop("bootstrap_source_content_binding_sha256", None),
        "bootstrap source content binding",
    )
    if canonical_sha256(binding) != supplied:
        raise ValueError("bootstrap source content binding digest changed")
    runtime = binding["runtime_source_bundle"]
    _exact(
        runtime,
        {
            "kind",
            "path",
            "bytes",
            "sha256",
            "runtime_source_bundle_sha256",
            "records_sha256",
            "file_count",
        },
        "runtime source content summary",
    )
    roles = [
        _role_summary(
            kind="candidate_role_payload_contract",
            path=CANDIDATE_PAYLOAD_OBJECT_PATH,
            payload=candidate_payload_contract,
        ),
        _role_summary(
            kind="reference_role_payload_contract",
            path=REFERENCE_PAYLOAD_OBJECT_PATH,
            payload=reference_payload_contract,
        ),
    ]
    if (
        binding["schema"] != CONTENT_BINDING_SCHEMA
        or binding["prefix_derivation_rule"] != PREFIX_DERIVATION_RULE
        or binding["source_object_count"] != 3
        or binding["source_object_paths"]
        != [
            RUNTIME_SOURCE_OBJECT_PATH,
            CANDIDATE_PAYLOAD_OBJECT_PATH,
            REFERENCE_PAYLOAD_OBJECT_PATH,
        ]
        or runtime["kind"] != "shared_runtime_source_bundle"
        or runtime["path"] != RUNTIME_SOURCE_OBJECT_PATH
        or type(runtime["bytes"]) is not int
        or not 1 <= runtime["bytes"] <= MAX_SOURCE_OBJECT_BYTES
        or type(runtime["file_count"]) is not int
        or runtime["file_count"] != len(REQUIRED_RUNTIME_SOURCE_PATHS)
        or any(
            _sha(runtime[field], field) != runtime[field]
            for field in (
                "sha256",
                "runtime_source_bundle_sha256",
                "records_sha256",
            )
        )
        or binding["role_payload_contracts"] != roles
        or binding["role_payload_contracts_sha256"]
        != canonical_sha256(roles)
        or binding["content_summaries_sha256"]
        != canonical_sha256([runtime, *roles])
        or binding["canonical_json_objects"] is not True
        or binding["regular_source_files_only"] is not True
        or binding["symlink_count"] != 0
        or binding["final_source_prefix_present"] is not False
        or binding["generation_present"] is not False
    ):
        raise ValueError("bootstrap source content binding changed")
    return {**binding, "bootstrap_source_content_binding_sha256": supplied}


def validate_runtime_bundle_against_content_binding(
    runtime_source_bundle: Mapping[str, Any],
    *,
    content_binding: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = validate_runtime_source_bundle(runtime_source_bundle)
    binding = validate_bootstrap_source_content_binding(
        content_binding,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
    )
    raw = canonical_bytes(bundle)
    summary = binding["runtime_source_bundle"]
    if (
        summary["bytes"] != len(raw)
        or summary["sha256"] != hashlib.sha256(raw).hexdigest()
        or summary["runtime_source_bundle_sha256"]
        != bundle["runtime_source_bundle_sha256"]
        or summary["records_sha256"] != bundle["records_sha256"]
        or summary["file_count"] != bundle["file_count"]
    ):
        raise ValueError("runtime source bundle escaped content binding")
    return bundle


__all__ = [
    "CANDIDATE_PAYLOAD_OBJECT_PATH",
    "CONTENT_BINDING_SCHEMA",
    "PREFIX_DERIVATION_RULE",
    "REFERENCE_PAYLOAD_OBJECT_PATH",
    "REQUIRED_RUNTIME_SOURCE_PATHS",
    "RUNTIME_SOURCE_BUNDLE_SCHEMA",
    "RUNTIME_SOURCE_OBJECT_PATH",
    "build_bootstrap_source_content_binding",
    "build_runtime_source_bundle",
    "canonical_bytes",
    "canonical_sha256",
    "validate_bootstrap_source_content_binding",
    "validate_runtime_bundle_against_content_binding",
    "validate_runtime_source_bundle",
]
