"""Pure Step12b role-local VM metadata and source-manifest limits.

This module performs no metadata-server, Compute, IAM, object-store, or
process operation.  It freezes the exact role-local insert allowlist and the
Compute Engine size budget that a later controller and VM prebootstrap must
both enforce.  The post-create claim is forbidden at insert and has an
explicit reserved budget.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)


SOURCE_BUNDLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_vm_runtime_source_bundle_v2"
)
METADATA_BUDGET_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_vm_metadata_budget_receipt_v2"
)
VM_IDENTITY_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_verified_vm_identity_v2"
)

MAX_METADATA_VALUE_BYTES = 262_144
MAX_TOTAL_METADATA_BYTES = 524_288
MIN_POSTCLAIM_HEADROOM_BYTES = 32_768
MAX_POSTCREATE_CLAIM_VALUE_BYTES = 32_768
MAX_PAIR_RELEASE_VALUE_BYTES = 32_768
EXPECTED_PROJECT_NUMBER = "783381566570"

STARTUP_KEY = "startup-script"
BLOCK_PROJECT_SSH_KEYS_KEY = "block-project-ssh-keys"
RELEASE_STATE_KEY = "ofc-step12b-claim-release-state"
DEPLOYMENT_CONTRACT_KEY = "ofc-step12b-deployment-contract"
ROLE_PAYLOAD_CONTRACT_KEY = "ofc-step12b-role-payload-contract"
ROLE_BOOTSTRAP_MANIFEST_KEY = "ofc-step12b-role-bootstrap-manifest"
CONTROLLER_PUBLIC_KEY = "ofc-step12b-controller-public-key"
EXTERNAL_AUTHORIZATION_KEY = "ofc-step12b-external-authorization"
PACKAGE_GENERATIONS_KEY = "ofc-step12b-package-generations"
EXTERNAL_PREFLIGHT_RECEIPT_KEY = (
    "ofc-step12b-external-preflight-receipt"
)
RUN_NONCE_KEY = "ofc-step12b-run-nonce"
EXTERNAL_JOB_ID_KEY = "ofc-step12b-external-job-id"
SOURCE_ROLE_KEY = "ofc-step12b-source-role"
RUNTIME_SOURCE_BUNDLE_KEY = "ofc-step12b-runtime-source-bundle"
POSTCREATE_CLAIM_KEY = "ofc-step12b-external-worker-claim"
PAIR_RELEASE_KEY = "ofc-step12b-pair-release"

PENDING_RELEASE_STATE = "pending-post-create-claim"

INITIAL_METADATA_KEYS = frozenset(
    {
        STARTUP_KEY,
        BLOCK_PROJECT_SSH_KEYS_KEY,
        RELEASE_STATE_KEY,
        DEPLOYMENT_CONTRACT_KEY,
        ROLE_BOOTSTRAP_MANIFEST_KEY,
        CONTROLLER_PUBLIC_KEY,
        EXTERNAL_AUTHORIZATION_KEY,
        PACKAGE_GENERATIONS_KEY,
        EXTERNAL_PREFLIGHT_RECEIPT_KEY,
        RUN_NONCE_KEY,
        EXTERNAL_JOB_ID_KEY,
        SOURCE_ROLE_KEY,
    }
)
POSTCLAIM_METADATA_KEYS = frozenset(
    {*INITIAL_METADATA_KEYS, POSTCREATE_CLAIM_KEY}
)
POSTRELEASE_METADATA_KEYS = frozenset(
    {*POSTCLAIM_METADATA_KEYS, PAIR_RELEASE_KEY}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_ID = re.compile(r"^[1-9][0-9]*$")
_MODULE_PATH = re.compile(r"^[a-z0-9_./-]+\.py$")
_FORBIDDEN_PATH_PARTS = (
    "private",
    "secret",
    "credential",
    "token",
    "site-packages",
    "__pycache__",
)
_VM_IDENTITY_SEAL = object()


@dataclass(frozen=True)
class VerifiedVmIdentity:
    """Opaque proof of provider identity observed after claim CAS."""

    deployment_contract_sha256: str
    external_job_id: str
    inner_job_id: str
    source_role: str
    instance_name: str
    provider_instance_id: str
    project_id: str
    project_number: str
    zone: str
    worker_service_account: str
    oauth_scope: str
    receipt_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VM_IDENTITY_SEAL:
            raise ValueError(
                "VerifiedVmIdentity cannot be constructed externally"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": VM_IDENTITY_RECEIPT_SCHEMA,
            "observation_phase": "worker_guest_metadata_after_claim",
            "deployment_contract_sha256": (
                self.deployment_contract_sha256
            ),
            "external_job_id": self.external_job_id,
            "inner_job_id": self.inner_job_id,
            "source_role": self.source_role,
            "instance_name": self.instance_name,
            "provider_instance_id": self.provider_instance_id,
            "project_id": self.project_id,
            "project_number": self.project_number,
            "zone": self.zone,
            "worker_service_account": self.worker_service_account,
            "oauth_scopes": [self.oauth_scope],
            "vm_identity_receipt_sha256": self.receipt_sha256,
        }


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


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed")


def _safe_source_path(value: Any) -> str:
    path = PurePosixPath(value) if isinstance(value, str) else None
    lowered = value.lower() if isinstance(value, str) else ""
    if (
        not isinstance(value, str)
        or not value
        or path is None
        or path.is_absolute()
        or path.as_posix() != value
        or _MODULE_PATH.fullmatch(value) is None
        or any(part in ("", ".", "..") for part in path.parts)
        or any(part in lowered for part in _FORBIDDEN_PATH_PARTS)
    ):
        raise ValueError("runtime source path escaped its allowlist")
    return value


def _source_text(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or value.startswith(("\ufeff", "#!"))
    ):
        raise ValueError("runtime source must be nonempty UTF-8 Python text")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_METADATA_VALUE_BYTES:
        raise ValueError("one runtime source escaped metadata value limit")
    compile(value, "<step12b-runtime-source>", "exec")
    return value


def build_runtime_source_bundle(
    source_files: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic, self-hashed Python source bundle."""

    if (
        not isinstance(source_files, Mapping)
        or not source_files
        or len(source_files) > 16
    ):
        raise ValueError("runtime source bundle count changed")
    records = []
    for raw_path in sorted(source_files):
        path = _safe_source_path(raw_path)
        source = _source_text(source_files[raw_path])
        raw = source.encode("utf-8")
        records.append(
            {
                "path": path,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": source,
            }
        )
    body = {
        "schema": SOURCE_BUNDLE_SCHEMA,
        "file_count": len(records),
        "paths": [row["path"] for row in records],
        "records": records,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
        "private_material_present": False,
    }
    result = {
        **body,
        "runtime_source_bundle_sha256": canonical_sha256(body),
    }
    if len(canonical_bytes(result)) > MAX_METADATA_VALUE_BYTES:
        raise ValueError("runtime source bundle escaped one metadata value")
    return result


def validate_runtime_source_bundle(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = dict(value)
    _exact(
        bundle,
        {
            "schema",
            "file_count",
            "paths",
            "records",
            "repo_import_permitted",
            "site_packages_import_permitted",
            "private_material_present",
            "runtime_source_bundle_sha256",
        },
        "Step12b runtime source bundle",
    )
    records = bundle["records"]
    if (
        bundle["schema"] != SOURCE_BUNDLE_SCHEMA
        or type(bundle["file_count"]) is not int
        or not 1 <= bundle["file_count"] <= 16
        or not isinstance(records, list)
        or len(records) != bundle["file_count"]
        or bundle["repo_import_permitted"] is not False
        or bundle["site_packages_import_permitted"] is not False
        or bundle["private_material_present"] is not False
    ):
        raise ValueError("Step12b runtime source bundle boundary changed")
    expected_sources: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("runtime source record is not a mapping")
        _exact(
            record,
            {"path", "bytes", "sha256", "source"},
            "runtime source record",
        )
        path = _safe_source_path(record["path"])
        source = _source_text(record["source"])
        raw = source.encode("utf-8")
        if (
            type(record["bytes"]) is not int
            or record["bytes"] != len(raw)
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or record["sha256"] != hashlib.sha256(raw).hexdigest()
            or path in expected_sources
        ):
            raise ValueError("runtime source record identity changed")
        expected_sources[path] = source
    expected = build_runtime_source_bundle(expected_sources)
    if bundle != expected:
        raise ValueError("Step12b runtime source bundle changed")
    return expected


def _metadata_total_bytes(values: Mapping[str, str]) -> int:
    return sum(
        len(key.encode("utf-8")) + len(value.encode("utf-8"))
        for key, value in values.items()
    )


def _checked_metadata_values(
    values: Mapping[str, str],
    *,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(values, Mapping) or set(values) != expected_keys:
        raise ValueError(f"{label} metadata allowlist changed")
    checked: dict[str, str] = {}
    for key in sorted(values):
        value = values[key]
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            or "\x00" in key
            or "\x00" in value
            or len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES
        ):
            raise ValueError(f"{label} metadata value changed: {key!r}")
        checked[key] = value
    return checked


def validate_initial_metadata_budget(
    values: Mapping[str, str],
) -> dict[str, Any]:
    """Validate insert metadata while reserving a bounded future claim."""

    checked = _checked_metadata_values(
        values,
        expected_keys=INITIAL_METADATA_KEYS,
        label="initial",
    )
    if (
        POSTCREATE_CLAIM_KEY in checked
        or checked[BLOCK_PROJECT_SSH_KEYS_KEY] != "true"
        or checked[RELEASE_STATE_KEY] != PENDING_RELEASE_STATE
    ):
        raise ValueError("initial claim-release boundary changed")
    deployment_raw = checked[DEPLOYMENT_CONTRACT_KEY].encode("ascii")
    manifest_raw = checked[ROLE_BOOTSTRAP_MANIFEST_KEY].encode("ascii")
    try:
        parsed_deployment = json.loads(deployment_raw)
        parsed_manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "deployment or role bootstrap manifest metadata is not JSON"
        ) from error
    if (
        not isinstance(parsed_deployment, dict)
        or not isinstance(parsed_manifest, dict)
        or canonical_bytes(parsed_deployment) != deployment_raw
        or canonical_bytes(parsed_manifest) != manifest_raw
    ):
        raise ValueError(
            "deployment or role bootstrap manifest metadata is not "
            "canonical JSON"
        )
    deployment_without_sha = dict(parsed_deployment)
    deployment_sha = deployment_without_sha.pop(
        "deployment_contract_sha256", None
    )
    if (
        not isinstance(deployment_sha, str)
        or _SHA256.fullmatch(deployment_sha) is None
        or canonical_sha256(deployment_without_sha) != deployment_sha
    ):
        raise ValueError("deployment contract metadata digest changed")
    manifest = bootstrap_source.validate_role_bootstrap_manifest_envelope(
        parsed_manifest,
        expected_deployment_contract_sha256=deployment_sha,
        expected_external_job_id=checked[EXTERNAL_JOB_ID_KEY],
        expected_source_role=checked[SOURCE_ROLE_KEY],
    )
    initial_bytes = _metadata_total_bytes(checked)
    reserved_claim_bytes = (
        len(POSTCREATE_CLAIM_KEY.encode("utf-8"))
        + MAX_POSTCREATE_CLAIM_VALUE_BYTES
    )
    reserved_release_bytes = (
        len(PAIR_RELEASE_KEY.encode("utf-8"))
        + MAX_PAIR_RELEASE_VALUE_BYTES
    )
    required_free = (
        reserved_claim_bytes
        + reserved_release_bytes
        + MIN_POSTCLAIM_HEADROOM_BYTES
    )
    if (
        initial_bytes > MAX_TOTAL_METADATA_BYTES
        or MAX_TOTAL_METADATA_BYTES - initial_bytes < required_free
    ):
        raise ValueError(
            "initial metadata cannot reserve claim and required headroom"
        )
    body = {
        "schema": METADATA_BUDGET_RECEIPT_SCHEMA,
        "phase": "initial_insert_claim_absent",
        "metadata_key_count": len(checked),
        "metadata_keys": sorted(checked),
        "metadata_total_bytes": initial_bytes,
        "metadata_max_value_bytes": max(
            len(value.encode("utf-8")) for value in checked.values()
        ),
        "postcreate_claim_present": False,
        "postcreate_claim_reserved_bytes": reserved_claim_bytes,
        "pair_release_reserved_bytes": reserved_release_bytes,
        "remaining_after_reserved_claim_and_release_bytes": (
            MAX_TOTAL_METADATA_BYTES
            - initial_bytes
            - reserved_claim_bytes
            - reserved_release_bytes
        ),
        "required_postclaim_headroom_bytes": (
            MIN_POSTCLAIM_HEADROOM_BYTES
        ),
        "role_bootstrap_manifest_sha256": manifest[
            "role_manifest_sha256"
        ],
        "source_plan_sha256": manifest["source_plan_sha256"],
        "source_provision_receipt_sha256": manifest[
            "source_provision_receipt_sha256"
        ],
        "runtime_source_bundle_sha256": manifest[
            "runtime_source_bundle_sha256"
        ],
        "per_value_limit_bytes": MAX_METADATA_VALUE_BYTES,
        "aggregate_limit_bytes": MAX_TOTAL_METADATA_BYTES,
    }
    return {
        **body,
        "metadata_budget_receipt_sha256": canonical_sha256(body),
    }


def validate_postclaim_metadata_budget(
    *,
    initial_values: Mapping[str, str],
    claim_value: str,
) -> dict[str, Any]:
    """Validate exact add-only claim metadata and 32 KiB final headroom."""

    initial_receipt = validate_initial_metadata_budget(initial_values)
    if (
        not isinstance(claim_value, str)
        or not claim_value
        or "\x00" in claim_value
        or len(claim_value.encode("utf-8"))
        > MAX_POSTCREATE_CLAIM_VALUE_BYTES
    ):
        raise ValueError("post-create claim value escaped reserved budget")
    updated = {
        **dict(initial_values),
        POSTCREATE_CLAIM_KEY: claim_value,
    }
    checked = _checked_metadata_values(
        updated,
        expected_keys=POSTCLAIM_METADATA_KEYS,
        label="post-claim",
    )
    if any(
        checked[key] != value for key, value in initial_values.items()
    ):
        raise ValueError("post-create claim changed initial metadata")
    total_bytes = _metadata_total_bytes(checked)
    headroom = MAX_TOTAL_METADATA_BYTES - total_bytes
    reserved_release_bytes = (
        len(PAIR_RELEASE_KEY.encode("utf-8"))
        + MAX_PAIR_RELEASE_VALUE_BYTES
    )
    if (
        total_bytes > MAX_TOTAL_METADATA_BYTES
        or headroom
        < reserved_release_bytes + MIN_POSTCLAIM_HEADROOM_BYTES
    ):
        raise ValueError(
            "post-claim metadata cannot reserve release and headroom"
        )
    body = {
        "schema": METADATA_BUDGET_RECEIPT_SCHEMA,
        "phase": "postcreate_claim_exact_add_only",
        "metadata_key_count": len(checked),
        "metadata_keys": sorted(checked),
        "metadata_total_bytes": total_bytes,
        "metadata_max_value_bytes": max(
            len(value.encode("utf-8")) for value in checked.values()
        ),
        "postcreate_claim_present": True,
        "postcreate_claim_bytes": len(claim_value.encode("utf-8")),
        "pair_release_reserved_bytes": reserved_release_bytes,
        "remaining_before_reserved_release_bytes": headroom,
        "remaining_after_reserved_release_bytes": (
            headroom - reserved_release_bytes
        ),
        "required_postclaim_headroom_bytes": (
            MIN_POSTCLAIM_HEADROOM_BYTES
        ),
        "initial_metadata_budget_receipt_sha256": initial_receipt[
            "metadata_budget_receipt_sha256"
        ],
        "role_bootstrap_manifest_sha256": initial_receipt[
            "role_bootstrap_manifest_sha256"
        ],
        "source_plan_sha256": initial_receipt["source_plan_sha256"],
        "source_provision_receipt_sha256": initial_receipt[
            "source_provision_receipt_sha256"
        ],
        "runtime_source_bundle_sha256": initial_receipt[
            "runtime_source_bundle_sha256"
        ],
        "per_value_limit_bytes": MAX_METADATA_VALUE_BYTES,
        "aggregate_limit_bytes": MAX_TOTAL_METADATA_BYTES,
    }
    return {
        **body,
        "metadata_budget_receipt_sha256": canonical_sha256(body),
    }


def validate_postrelease_metadata_budget(
    *,
    initial_values: Mapping[str, str],
    claim_value: str,
    release_value: str,
) -> dict[str, Any]:
    """Validate the exact second add-only CAS and final 32 KiB headroom."""

    claim_receipt = validate_postclaim_metadata_budget(
        initial_values=initial_values,
        claim_value=claim_value,
    )
    if (
        not isinstance(release_value, str)
        or not release_value
        or "\x00" in release_value
        or len(release_value.encode("utf-8"))
        > MAX_PAIR_RELEASE_VALUE_BYTES
    ):
        raise ValueError("pair release value escaped reserved budget")
    updated = {
        **dict(initial_values),
        POSTCREATE_CLAIM_KEY: claim_value,
        PAIR_RELEASE_KEY: release_value,
    }
    checked = _checked_metadata_values(
        updated,
        expected_keys=POSTRELEASE_METADATA_KEYS,
        label="post-release",
    )
    if any(
        checked[key] != value for key, value in initial_values.items()
    ) or checked[POSTCREATE_CLAIM_KEY] != claim_value:
        raise ValueError("pair release changed pre-release metadata")
    total_bytes = _metadata_total_bytes(checked)
    headroom = MAX_TOTAL_METADATA_BYTES - total_bytes
    if (
        total_bytes > MAX_TOTAL_METADATA_BYTES
        or headroom < MIN_POSTCLAIM_HEADROOM_BYTES
    ):
        raise ValueError("post-release metadata headroom changed")
    body = {
        "schema": METADATA_BUDGET_RECEIPT_SCHEMA,
        "phase": "pair_release_exact_second_add_only",
        "metadata_key_count": len(checked),
        "metadata_keys": sorted(checked),
        "metadata_total_bytes": total_bytes,
        "metadata_max_value_bytes": max(
            len(value.encode("utf-8")) for value in checked.values()
        ),
        "postcreate_claim_present": True,
        "pair_release_present": True,
        "postcreate_claim_bytes": len(claim_value.encode("utf-8")),
        "pair_release_bytes": len(release_value.encode("utf-8")),
        "remaining_headroom_bytes": headroom,
        "required_postclaim_headroom_bytes": (
            MIN_POSTCLAIM_HEADROOM_BYTES
        ),
        "postclaim_metadata_budget_receipt_sha256": claim_receipt[
            "metadata_budget_receipt_sha256"
        ],
        "role_bootstrap_manifest_sha256": claim_receipt[
            "role_bootstrap_manifest_sha256"
        ],
        "source_plan_sha256": claim_receipt["source_plan_sha256"],
        "source_provision_receipt_sha256": claim_receipt[
            "source_provision_receipt_sha256"
        ],
        "runtime_source_bundle_sha256": claim_receipt[
            "runtime_source_bundle_sha256"
        ],
        "per_value_limit_bytes": MAX_METADATA_VALUE_BYTES,
        "aggregate_limit_bytes": MAX_TOTAL_METADATA_BYTES,
    }
    return {
        **body,
        "metadata_budget_receipt_sha256": canonical_sha256(body),
    }


def validate_vm_identity_after_claim(
    *,
    role_runtime_deployment: Mapping[str, Any],
    external_job_id: str,
    observed_identity: Mapping[str, Any],
) -> VerifiedVmIdentity:
    """Validate only fields exposed by the GCE guest metadata server.

    Metadata fingerprints are deliberately absent.  Both pre-CAS F0 and
    post-CAS F1 belong to controller-side claim/readback/teardown evidence.
    """

    deployment = dict(role_runtime_deployment)
    supplied_deployment_sha = deployment.pop(
        "deployment_contract_sha256", None
    )
    if (
        not isinstance(supplied_deployment_sha, str)
        or _SHA256.fullmatch(supplied_deployment_sha) is None
        or canonical_sha256(deployment) != supplied_deployment_sha
    ):
        raise ValueError("role-runtime deployment self digest changed")
    selected_job_ids = deployment.get("selected_job_ids")
    source_roles = deployment.get("source_roles")
    instances = deployment.get("instances")
    if (
        not isinstance(external_job_id, str)
        or not isinstance(selected_job_ids, list)
        or selected_job_ids.count(external_job_id) != 1
        or not isinstance(source_roles, list)
        or not isinstance(instances, list)
        or len(selected_job_ids) != len(source_roles)
        or len(selected_job_ids) != len(instances)
    ):
        raise ValueError("role-runtime VM mapping changed")
    position = selected_job_ids.index(external_job_id)
    instance = instances[position]
    if not isinstance(instance, Mapping):
        raise ValueError("role-runtime instance mapping changed")
    observed = dict(observed_identity)
    _exact(
        observed,
        {
            "project_id",
            "project_number",
            "zone",
            "instance_name",
            "provider_instance_id",
            "service_account_email",
            "oauth_scopes",
            "external_job_id",
            "source_role",
        },
        "post-claim VM identity observation",
    )
    provider_instance_id = observed["provider_instance_id"]
    expected_role = source_roles[position]
    if (
        observed["project_id"] != payload_transport.PROJECT
        or observed["project_number"] != EXPECTED_PROJECT_NUMBER
        or observed["zone"] != payload_transport.ZONE
        or observed["instance_name"] != instance.get("instance_name")
        or observed["external_job_id"] != external_job_id
        or observed["source_role"] != expected_role
        or instance.get("job_id") != external_job_id
        or instance.get("source_role") != expected_role
        or instance.get("attempt_index") != 0
        or observed["service_account_email"]
        != payload_transport.WORKER_SERVICE_ACCOUNT
        or observed["oauth_scopes"]
        != [payload_transport.REQUIRED_WORKER_OAUTH_SCOPE]
        or not isinstance(provider_instance_id, str)
        or _DECIMAL_ID.fullmatch(provider_instance_id) is None
        or len(provider_instance_id) > 32
    ):
        raise ValueError("post-claim VM provider identity changed")
    body = {
        "schema": VM_IDENTITY_RECEIPT_SCHEMA,
        "observation_phase": "worker_guest_metadata_after_claim",
        "deployment_contract_sha256": supplied_deployment_sha,
        "external_job_id": external_job_id,
        "inner_job_id": instance["inner_job_id"],
        "source_role": expected_role,
        "instance_name": instance["instance_name"],
        "provider_instance_id": provider_instance_id,
        "project_id": payload_transport.PROJECT,
        "project_number": EXPECTED_PROJECT_NUMBER,
        "zone": payload_transport.ZONE,
        "worker_service_account": (
            payload_transport.WORKER_SERVICE_ACCOUNT
        ),
        "oauth_scopes": [
            payload_transport.REQUIRED_WORKER_OAUTH_SCOPE
        ],
    }
    return VerifiedVmIdentity(
        deployment_contract_sha256=supplied_deployment_sha,
        external_job_id=external_job_id,
        inner_job_id=instance["inner_job_id"],
        source_role=expected_role,
        instance_name=instance["instance_name"],
        provider_instance_id=provider_instance_id,
        project_id=payload_transport.PROJECT,
        project_number=EXPECTED_PROJECT_NUMBER,
        zone=payload_transport.ZONE,
        worker_service_account=(
            payload_transport.WORKER_SERVICE_ACCOUNT
        ),
        oauth_scope=payload_transport.REQUIRED_WORKER_OAUTH_SCOPE,
        receipt_sha256=canonical_sha256(body),
        _seal=_VM_IDENTITY_SEAL,
    )


def require_verified_vm_identity(
    value: VerifiedVmIdentity,
    *,
    deployment_contract_sha256: str,
    external_job_id: str,
) -> VerifiedVmIdentity:
    if (
        not isinstance(value, VerifiedVmIdentity)
        or value._seal is not _VM_IDENTITY_SEAL
        or value.deployment_contract_sha256
        != deployment_contract_sha256
        or value.external_job_id != external_job_id
    ):
        raise ValueError("verified VM identity capability changed")
    receipt = value.receipt()
    supplied = receipt.pop("vm_identity_receipt_sha256")
    if canonical_sha256(receipt) != supplied:
        raise ValueError("verified VM identity receipt changed")
    return value


__all__ = [
    "BLOCK_PROJECT_SSH_KEYS_KEY",
    "CONTROLLER_PUBLIC_KEY",
    "DEPLOYMENT_CONTRACT_KEY",
    "EXTERNAL_AUTHORIZATION_KEY",
    "EXTERNAL_JOB_ID_KEY",
    "EXTERNAL_PREFLIGHT_RECEIPT_KEY",
    "EXPECTED_PROJECT_NUMBER",
    "INITIAL_METADATA_KEYS",
    "MAX_METADATA_VALUE_BYTES",
    "MAX_PAIR_RELEASE_VALUE_BYTES",
    "MAX_POSTCREATE_CLAIM_VALUE_BYTES",
    "MAX_TOTAL_METADATA_BYTES",
    "METADATA_BUDGET_RECEIPT_SCHEMA",
    "MIN_POSTCLAIM_HEADROOM_BYTES",
    "PACKAGE_GENERATIONS_KEY",
    "PAIR_RELEASE_KEY",
    "PENDING_RELEASE_STATE",
    "POSTCLAIM_METADATA_KEYS",
    "POSTCREATE_CLAIM_KEY",
    "POSTRELEASE_METADATA_KEYS",
    "RELEASE_STATE_KEY",
    "ROLE_BOOTSTRAP_MANIFEST_KEY",
    "ROLE_PAYLOAD_CONTRACT_KEY",
    "RUN_NONCE_KEY",
    "RUNTIME_SOURCE_BUNDLE_KEY",
    "SOURCE_ROLE_KEY",
    "SOURCE_BUNDLE_SCHEMA",
    "STARTUP_KEY",
    "VM_IDENTITY_RECEIPT_SCHEMA",
    "VerifiedVmIdentity",
    "build_runtime_source_bundle",
    "canonical_bytes",
    "canonical_sha256",
    "validate_initial_metadata_budget",
    "validate_postclaim_metadata_budget",
    "validate_postrelease_metadata_budget",
    "require_verified_vm_identity",
    "validate_runtime_source_bundle",
    "validate_vm_identity_after_claim",
]
