"""Offline Step 12b deployment identity over the immutable v1 payload.

The accepted diagnostic outer package is deliberately not rebuilt.  Its v1
Stage-2 job ids remain payload-local identities.  This module creates a
separate, fresh deployment identity with:

* a nonce-bound external run and stage;
* external candidate/reference job aliases;
* a collision-separated ``direct-v2`` result namespace;
* instance names derived from the new direct-stage identity; and
* exact bindings back to both validated v1 payload contracts.

The resulting object is not a cloud execution contract.  In particular it
does not authorize an insert, IAM mutation, package write, result write,
resume, or attempt 1.  A later worker bridge must keep the external aliases
separate from the immutable inner job ids.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_deployment_contract_v2"
)
STATUS = "offline_fresh_deployment_identity_external_authorization_required"
DIRECT_STAGE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_direct_stage_identity_v2"
)
RUN_IDENTITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_run_identity_v2"
)
ROLE_LOCAL_RUNTIME_VIEW_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_role_local_runtime_view_v2"
)
BOOTSTRAP_SOURCE_CONTENT_BINDING_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_source_content_binding_v2"
)
BOOTSTRAP_SOURCE_PREFIX_DERIVATION_RULE = (
    "gs_bucket_direct_v2_bootstrap_sources_slash_deployment_sha256"
)
BOOTSTRAP_RUNTIME_SOURCE_OBJECT_PATH = "runtime_source_bundle.json"
BOOTSTRAP_CANDIDATE_PAYLOAD_OBJECT_PATH = (
    "candidate_payload_contract.json"
)
BOOTSTRAP_REFERENCE_PAYLOAD_OBJECT_PATH = (
    "reference_payload_contract.json"
)
BOOTSTRAP_SOURCE_OBJECT_MAX_BYTES = 1_048_576
BOOTSTRAP_RUNTIME_SOURCE_FILE_COUNT = 12

DIRECT_NAMESPACE = "hu-m31-r2diag-direct-v2"
STAGE_KIND = "stage2_candidate_reference_pair"
ATTEMPT_INDEX = 0
MAX_ATTEMPTS = 1
VM_COUNT = 2
ACTUAL_MACHINE_TYPE = "c4-standard-8"
ACTUAL_VCPUS_PER_VM = 8
ACTUAL_MEMORY_MB = 30_720
INNER_MACHINE_TYPE = "c4-standard-16"
INNER_RAYON_THREADS = 16
INNER_STAGE_ID = "stage2_candidate_reference_pair"
INNER_RUN_NAME = "regular-hu-m31-r2diag-s2-20260718-001"
INNER_JOB_IDS = ("candidate-shard-01", "reference-shard-01")
INNER_HAND_INDICES = (5, 6, 35, 39, 47, 53, 76, 83, 87, 89)

EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256 = (
    "593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548"
)
EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256 = (
    "c75ef12069fef0c62514b2c8625c098ab6c4819bc74edbd93e8b1a4e082411aa"
)
EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256 = (
    "29e3c6f827f8e59b9cc86d232fe3d329b1563b991557d739567e1c049cb27879"
)
EXPECTED_INNER_ADAPTER_PREVIEW_SHA256 = (
    "0cab33b34cf3800797f81b437c9d78f64191c01536c633f38155190b5aafd4f6"
)
EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256 = (
    "b70bcad3f7df8721d544f45fcd4bc513ce27ea6c199f0cb1a5adc749e1d029e4"
)
EXPECTED_CURRENT_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
LEGACY_SHARED_CONTROLLER_SERVICE_ACCOUNT = (
    "ofc-m31-t3-controller@ofc-solver-485418.iam.gserviceaccount.com"
)
EXPECTED_INNER_PREVIEW_CAPABILITIES = {
    "cloud_executable": False,
    "launch_ready": False,
    "cloud_launch_authorized": False,
    "gcloud_invocation_authorized": False,
    "subprocess_invocation_authorized": False,
    "claim_write_authorized": False,
    "authorization_write_authorized": False,
    "object_write_authorized": False,
    "vm_create_authorized": False,
    "remote_query_performed": False,
    "production_all20_launcher_reused": False,
    "current_profile_changed": False,
    "performance_lock_evidence": False,
    "quality_evidence": False,
    "training_eligible": False,
    "promotion_evidence": False,
}

_ROLES = ("candidate", "reference")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GCE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SAFE_ALIAS = re.compile(r"^[a-z](?:[-a-z0-9]{0,62})$")
_RUN_TAG = re.compile(r"^[0-9a-f]{12}$")
_SERVICE_ACCOUNT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_FORBIDDEN_FIELD_PARTS = (
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
    "private_key",
    "access_token",
    "authorization_header",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed")


def _reject_forbidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string field")
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_FIELD_PARTS):
                raise ValueError(f"{path}.{key} contains a forbidden field")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _safe_tree_path(value: Any) -> str:
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
        raise ValueError("payload tree path escaped the deployment root")
    return value


def _role_neutral_roots(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    roots = job.get("root_records")
    if not isinstance(roots, list):
        raise ValueError("payload root records are missing")
    return [
        {key: child for key, child in row.items() if key != "source_role"}
        for row in roots
    ]


def _validated_payloads(
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    public_record_sha256 = canonical_sha256(public_record)
    candidate = payload_transport.validate_job_contract(
        candidate_payload_contract
    )
    reference = payload_transport.validate_job_contract(
        reference_payload_contract
    )
    contracts = (candidate, reference)
    preview = candidate["adapter_preview"]
    layout = candidate["remote_layout"]
    expected_inner_jobs = list(INNER_JOB_IDS)
    jobs = preview.get("jobs")
    bindings = [row["metadata_binding"] for row in contracts]
    if (
        reference["adapter_preview"] != preview
        or reference["remote_layout"] != layout
        or reference["outer_package_manifest"]
        != candidate["outer_package_manifest"]
        or reference["direct_stage_identity"]
        != candidate["direct_stage_identity"]
        or preview.get("stage_id") != INNER_STAGE_ID
        or preview.get("run_name") != INNER_RUN_NAME
        or preview.get("selected_job_ids") != expected_inner_jobs
        or preview.get("attempt_index") != ATTEMPT_INDEX
        or preview.get("stage_identity_sha256")
        != EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
        or candidate.get("adapter_preview_sha256")
        != EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        or reference.get("adapter_preview_sha256")
        != EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        or canonical_sha256(preview)
        != EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        or preview.get("capabilities")
        != EXPECTED_INNER_PREVIEW_CAPABILITIES
        or not isinstance(jobs, list)
        or [row.get("job_id") for row in jobs] != expected_inner_jobs
        or [row.get("source_role") for row in jobs] != list(_ROLES)
        or any(
            row.get("work_hand_indices")
            != list(INNER_HAND_INDICES)
            for row in jobs
        )
        or [row.get("job_id") for row in bindings] != expected_inner_jobs
        or [row.get("source_role") for row in bindings] != list(_ROLES)
        or any(row.get("attempt_index") != ATTEMPT_INDEX for row in bindings)
    ):
        raise ValueError("immutable payload Stage-2 mapping changed")
    if _role_neutral_roots(jobs[0]) != _role_neutral_roots(jobs[1]):
        raise ValueError("candidate/reference payload roots diverged")

    outer = candidate["outer_package_manifest"]
    inventory = layout.get("package_inventory")
    if (
        outer.get("outer_package_identity_sha256")
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or candidate.get("outer_package_manifest_sha256")
        != EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        or reference.get("outer_package_manifest_sha256")
        != EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        or candidate.get("direct_stage_identity_sha256")
        != EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
        or reference.get("direct_stage_identity_sha256")
        != EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
        or not isinstance(inventory, Mapping)
        or inventory.get("outer_package_identity_sha256")
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or layout.get("package_prefix")
        != (
            f"gs://{payload_transport.BUCKET}/"
            f"{payload_transport.DIRECT_NAMESPACE}/packages/"
            f"{EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256}"
        )
    ):
        raise ValueError("immutable outer payload identity changed")

    key_ids = [
        row.get("authorization_contract", {}).get("controller_key_id")
        for row in contracts
    ]
    public_key_hashes = [
        row.get("authorization_contract", {}).get(
            "controller_public_key_sha256"
        )
        for row in contracts
    ]
    if (
        len(set(key_ids)) != 1
        or len(set(public_key_hashes)) != 1
        or key_ids[0] != public_record["key_id"]
        or public_key_hashes[0] != public_record_sha256
    ):
        raise ValueError(
            "payload contracts are not pinned to the supplied public trust key"
        )
    _sha(key_ids[0], "payload controller key id")
    _sha(public_key_hashes[0], "payload public key record")

    for contract in contracts:
        capabilities = contract.get("capabilities")
        if (
            not isinstance(capabilities, Mapping)
            or capabilities.get("current_profile_changed") is not False
            or capabilities.get("cloud_executable") is not False
            or capabilities.get("launch_ready") is not False
            or capabilities.get("cloud_launch_authorized") is not False
            or any(
                capabilities.get(field) is not False
                for field in (
                    "performance_lock_evidence",
                    "quality_evidence",
                    "training_eligible",
                    "promotion_evidence",
                )
            )
        ):
            raise ValueError("payload capability boundary changed")
    return candidate, reference, public_record


def _identity_seed(
    *,
    run_nonce: str,
    controller_key_id: str,
    controller_public_key_sha256: str,
    payload_contract_sha256s: Sequence[str],
    bootstrap_source_content_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seed = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_run_identity_seed_v2"
        ),
        "run_nonce": run_nonce,
        "controller_key_id": controller_key_id,
        "controller_public_key_sha256": _sha(
            controller_public_key_sha256,
            "controller public key record",
        ),
        "outer_package_identity_sha256": (
            EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        ),
        "inner_preview_stage_identity_sha256": (
            EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
        ),
        "payload_contract_sha256s": list(payload_contract_sha256s),
    }
    if bootstrap_source_content_binding is not None:
        seed["bootstrap_source_content_binding"] = copy.deepcopy(
            dict(bootstrap_source_content_binding)
        )
    return seed


def _validate_role_bootstrap_source_content_binding(
    value: Mapping[str, Any],
    *,
    selected_payload_contract: Mapping[str, Any],
    position: int,
) -> dict[str, Any]:
    """Validate a signed content binding from one role's limited view.

    The controller validates the complete binding against both payloads before
    signing.  A worker can still validate the complete digest structure and its
    own exact payload summary without receiving the opponent role's payload.
    """

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
        "role-local bootstrap source content binding",
    )
    supplied_sha = _sha(
        binding.pop("bootstrap_source_content_binding_sha256", None),
        "bootstrap source content binding",
    )
    if canonical_sha256(binding) != supplied_sha:
        raise ValueError("bootstrap source content binding digest changed")

    runtime = binding["runtime_source_bundle"]
    roles = binding["role_payload_contracts"]
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime source content summary changed")
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
    if not isinstance(roles, list) or len(roles) != VM_COUNT:
        raise ValueError("bootstrap role payload summaries changed")
    role_fields = {
        "kind",
        "path",
        "source_role",
        "inner_job_id",
        "bytes",
        "sha256",
    }
    for row in roles:
        if not isinstance(row, Mapping):
            raise ValueError("bootstrap role payload summary changed")
        _exact(row, role_fields, "bootstrap role payload summary")
        if (
            type(row["bytes"]) is not int
            or not 1 <= row["bytes"] <= BOOTSTRAP_SOURCE_OBJECT_MAX_BYTES
        ):
            raise ValueError("bootstrap role payload byte count changed")
        _sha(row["sha256"], "bootstrap role payload")

    selected_payload = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    selected_raw = canonical_bytes(selected_payload)
    expected_selected = {
        "kind": f"{_ROLES[position]}_role_payload_contract",
        "path": (
            BOOTSTRAP_CANDIDATE_PAYLOAD_OBJECT_PATH
            if position == 0
            else BOOTSTRAP_REFERENCE_PAYLOAD_OBJECT_PATH
        ),
        "source_role": _ROLES[position],
        "inner_job_id": INNER_JOB_IDS[position],
        "bytes": len(selected_raw),
        "sha256": hashlib.sha256(selected_raw).hexdigest(),
    }
    if (
        binding["schema"] != BOOTSTRAP_SOURCE_CONTENT_BINDING_SCHEMA
        or binding["prefix_derivation_rule"]
        != BOOTSTRAP_SOURCE_PREFIX_DERIVATION_RULE
        or binding["source_object_count"] != 3
        or binding["source_object_paths"]
        != [
            BOOTSTRAP_RUNTIME_SOURCE_OBJECT_PATH,
            BOOTSTRAP_CANDIDATE_PAYLOAD_OBJECT_PATH,
            BOOTSTRAP_REFERENCE_PAYLOAD_OBJECT_PATH,
        ]
        or runtime["kind"] != "shared_runtime_source_bundle"
        or runtime["path"]
        != BOOTSTRAP_RUNTIME_SOURCE_OBJECT_PATH
        or type(runtime["bytes"]) is not int
        or not 1
        <= runtime["bytes"]
        <= BOOTSTRAP_SOURCE_OBJECT_MAX_BYTES
        or type(runtime["file_count"]) is not int
        or runtime["file_count"]
        != BOOTSTRAP_RUNTIME_SOURCE_FILE_COUNT
        or any(
            _sha(runtime[field], field) != runtime[field]
            for field in (
                "sha256",
                "runtime_source_bundle_sha256",
                "records_sha256",
            )
        )
        or [row["source_role"] for row in roles] != list(_ROLES)
        or [row["inner_job_id"] for row in roles] != list(INNER_JOB_IDS)
        or [row["kind"] for row in roles]
        != [
            "candidate_role_payload_contract",
            "reference_role_payload_contract",
        ]
        or [row["path"] for row in roles]
        != [
            BOOTSTRAP_CANDIDATE_PAYLOAD_OBJECT_PATH,
            BOOTSTRAP_REFERENCE_PAYLOAD_OBJECT_PATH,
        ]
        or roles[position] != expected_selected
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
    return {**binding, "bootstrap_source_content_binding_sha256": supplied_sha}


def _validate_bootstrap_source_content_binding(
    value: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_checked = _validate_role_bootstrap_source_content_binding(
        value,
        selected_payload_contract=candidate_payload_contract,
        position=0,
    )
    reference_checked = _validate_role_bootstrap_source_content_binding(
        value,
        selected_payload_contract=reference_payload_contract,
        position=1,
    )
    if candidate_checked != reference_checked:
        raise ValueError("bootstrap source content binding changed")
    return candidate_checked


def _external_names(run_identity_sha256: str) -> dict[str, Any]:
    tag = _sha(run_identity_sha256, "run identity")[:12]
    run_name = f"regular-hu-m31-r2diag-s2b-{tag}"
    stage_id = f"stage2_candidate_reference_pair_s2b_{tag}"
    job_ids = [f"{role}-s2b-{tag}" for role in _ROLES]
    if (
        run_name == INNER_RUN_NAME
        or stage_id == INNER_STAGE_ID
        or any(
            _SAFE_ALIAS.fullmatch(job_id) is None
            or job_id in INNER_JOB_IDS
            for job_id in job_ids
        )
    ):
        raise ValueError("fresh external aliases collided with payload ids")
    return {
        "tag": tag,
        "run_name": run_name,
        "stage_id": stage_id,
        "job_ids": job_ids,
    }


def _run_scoped_controller_service_account(
    run_tag: str,
) -> dict[str, Any]:
    if not isinstance(run_tag, str) or _RUN_TAG.fullmatch(run_tag) is None:
        raise ValueError("Step12b run tag changed")
    account_id = f"ofc-m31-s2b-{run_tag}"
    email = (
        f"{account_id}@{payload_transport.PROJECT}.iam.gserviceaccount.com"
    )
    if (
        _SERVICE_ACCOUNT_ID.fullmatch(account_id) is None
        or len(account_id) > 30
        or email == LEGACY_SHARED_CONTROLLER_SERVICE_ACCOUNT
    ):
        raise ValueError("run-scoped controller service account changed")
    return {
        "project": payload_transport.PROJECT,
        "account_id": account_id,
        "email": email,
        "principal": f"serviceAccount:{email}",
        "derived_from_run_tag": run_tag,
        "run_scoped": True,
        "legacy_shared_controller_reused": False,
    }


def _run_identity(
    *,
    seed: Mapping[str, Any],
    seed_sha256: str,
    names: Mapping[str, Any],
    controller_service_account: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": RUN_IDENTITY_SCHEMA,
        "run_identity_seed_sha256": _sha(
            seed_sha256, "run identity seed"
        ),
        "run_nonce": seed["run_nonce"],
        "run_tag": names["tag"],
        "controller_key_id": seed["controller_key_id"],
        "controller_public_key_sha256": seed[
            "controller_public_key_sha256"
        ],
        "controller_service_account": dict(controller_service_account),
        "outer_package_identity_sha256": (
            EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        ),
        "payload_contract_sha256s": list(
            seed["payload_contract_sha256s"]
        ),
        "stage_kind": STAGE_KIND,
        "stage_id": names["stage_id"],
        "run_name": names["run_name"],
        "selected_job_ids": list(names["job_ids"]),
        "attempt_index": ATTEMPT_INDEX,
        "max_attempts": MAX_ATTEMPTS,
    }
    return {
        **body,
        "run_identity_sha256": canonical_sha256(body),
    }


def _payload_binding(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    controller_key_id: str,
    controller_public_key_sha256: str,
) -> dict[str, Any]:
    contracts = (candidate, reference)
    preview = candidate["adapter_preview"]
    layout = candidate["remote_layout"]
    contract_shas = [canonical_sha256(row) for row in contracts]
    job_bindings = []
    for role, contract, job in zip(
        _ROLES, contracts, preview["jobs"], strict=True
    ):
        tree_paths = [
            _safe_tree_path(row.get("path"))
            for row in job["tree_object_manifest"]
        ]
        if len(tree_paths) != len(set(tree_paths)):
            raise ValueError("payload tree paths collided")
        job_bindings.append(
            {
                "source_role": role,
                "inner_job_id": job["job_id"],
                "inner_job_manifest_path": (
                    f"inner/jobs/{job['job_id']}.json"
                ),
                "runner_job_manifest_sha256": job[
                    "runner_job_manifest"
                ]["sha256"],
                "payload_contract_sha256": canonical_sha256(contract),
                "work_hand_indices": list(job["work_hand_indices"]),
                "root_records_sha256": canonical_sha256(job["root_records"]),
                "tree_paths": tree_paths,
                "tree_paths_sha256": canonical_sha256(tree_paths),
            }
        )
    package_inventory = layout["package_inventory"]
    binding = {
        "immutable_outer_package_reused": True,
        "outer_package_identity_sha256": (
            EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        ),
        "outer_package_manifest_sha256": (
            EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        ),
        "controller_key_id": _sha(
            controller_key_id, "payload controller key id"
        ),
        "controller_public_key_sha256": _sha(
            controller_public_key_sha256,
            "payload public key record",
        ),
        "package_prefix": layout["package_prefix"],
        "package_inventory_records_sha256": package_inventory[
            "records_sha256"
        ],
        "package_object_count": len(package_inventory["records"]),
        "inner_stage_id": preview["stage_id"],
        "inner_run_name": preview["run_name"],
        "inner_preview_stage_identity_sha256": (
            EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
        ),
        "inner_direct_stage_identity_sha256": (
            EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
        ),
        "inner_job_ids": list(INNER_JOB_IDS),
        "payload_contract_sha256s": contract_shas,
        "job_bindings": job_bindings,
        "payload_contracts_are_local_execution_only": True,
        "payload_result_prefixes_are_provenance_only": True,
        "payload_result_writes_authorized": False,
        "package_write_authorized": False,
    }
    return {
        **binding,
        "payload_binding_sha256": canonical_sha256(binding),
    }


def _direct_identity(
    *,
    run_identity_sha256: str,
    controller_key_id: str,
    controller_public_key_sha256: str,
    controller_service_account: Mapping[str, Any],
    names: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
) -> dict[str, Any]:
    external_jobs = [
        {
            "job_id": external_job_id,
            "inner_job_id": inner["inner_job_id"],
            "source_role": inner["source_role"],
            "runner_job_manifest_sha256": inner[
                "runner_job_manifest_sha256"
            ],
            "payload_contract_sha256": inner["payload_contract_sha256"],
            "work_hand_indices": list(inner["work_hand_indices"]),
            "tree_paths_sha256": inner["tree_paths_sha256"],
        }
        for external_job_id, inner in zip(
            names["job_ids"],
            payload_binding["job_bindings"],
            strict=True,
        )
    ]
    body = {
        "schema": DIRECT_STAGE_SCHEMA,
        "direct_namespace": DIRECT_NAMESPACE,
        "run_identity_sha256": run_identity_sha256,
        "controller_key_id": controller_key_id,
        "controller_public_key_sha256": _sha(
            controller_public_key_sha256,
            "controller public key record",
        ),
        "controller_service_account": dict(controller_service_account),
        "payload_binding_sha256": payload_binding[
            "payload_binding_sha256"
        ],
        "outer_package_identity_sha256": (
            EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        ),
        "stage_kind": STAGE_KIND,
        "stage_id": names["stage_id"],
        "run_name": names["run_name"],
        "external_job_layout": external_jobs,
        "attempt_index": ATTEMPT_INDEX,
        "max_attempts": MAX_ATTEMPTS,
        "actual_machine_type": ACTUAL_MACHINE_TYPE,
        "actual_vcpus_per_vm": ACTUAL_VCPUS_PER_VM,
        "vm_count": VM_COUNT,
        "inner_machine_type": INNER_MACHINE_TYPE,
        "inner_rayon_threads": INNER_RAYON_THREADS,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    return {
        **body,
        "direct_stage_identity_sha256": canonical_sha256(body),
    }


def _instance_name(
    *, role: str, direct_stage_identity_sha256: str
) -> str:
    role_tag = {"candidate": "c", "reference": "r"}.get(role)
    if role_tag is None:
        raise ValueError("deployment role changed")
    name = (
        f"r2d-s2b-{role_tag}-a0-"
        f"{_sha(direct_stage_identity_sha256, 'direct identity')[:12]}"
    )
    if _GCE_NAME.fullmatch(name) is None or len(name) > 63:
        raise ValueError("deployment instance name is not GCE-safe")
    return name


def _remote_layout(
    *,
    names: Mapping[str, Any],
    direct_identity: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
) -> dict[str, Any]:
    direct_sha = direct_identity["direct_stage_identity_sha256"]
    base_prefix = (
        f"gs://{payload_transport.BUCKET}/{DIRECT_NAMESPACE}"
    )
    stage_prefix = (
        f"{base_prefix}/stages/{names['run_name']}/{direct_sha}"
    )
    result_prefix = f"{stage_prefix}/results"
    jobs = []
    for external_job_id, inner in zip(
        names["job_ids"], payload_binding["job_bindings"], strict=True
    ):
        job_prefix = f"{result_prefix}/jobs/{external_job_id}"
        tree_prefix = f"{job_prefix}/tree"
        jobs.append(
            {
                "job_id": external_job_id,
                "inner_job_id": inner["inner_job_id"],
                "source_role": inner["source_role"],
                "result_prefix": job_prefix,
                "tree_prefix": tree_prefix,
                "upload_uris": [
                    f"{job_prefix}/uploads/hand_{index:03d}.json"
                    for index in inner["work_hand_indices"]
                ],
                "heartbeat_uris": [
                    f"{result_prefix}/progress/jobs/{external_job_id}/"
                    f"heartbeats/{sequence:06d}.json"
                    for sequence in range(
                        1, len(inner["work_hand_indices"]) + 1
                    )
                ],
                "done_uri": f"{job_prefix}/DONE.envelope.json",
                "tree_object_uris": [
                    f"{tree_prefix}/{path}" for path in inner["tree_paths"]
                ],
            }
        )
    layout = {
        "base_prefix": base_prefix,
        "package_prefix": payload_binding["package_prefix"],
        "stage_prefix": stage_prefix,
        "result_prefix": result_prefix,
        "attempt_control_prefix": (
            f"{stage_prefix}/control/attempt-{ATTEMPT_INDEX}"
        ),
        "receive_uri": (
            f"{result_prefix}/received/{names['stage_id']}.json"
        ),
        "direct_stage_identity_sha256": direct_sha,
        "jobs": jobs,
        "package_and_stage_prefix_disjoint": True,
        "payload_and_deployment_result_prefixes_disjoint": True,
        "stage_prefix_must_be_exactly_empty_before_attempt0": True,
        "attempt_control_prefix_must_be_exactly_empty": True,
        "unknown_object_is_fatal": True,
        "result_identity_retry_invariant": False,
    }
    all_uris = [layout["receive_uri"]]
    for job in jobs:
        all_uris.extend(job["upload_uris"])
        all_uris.extend(job["heartbeat_uris"])
        all_uris.append(job["done_uri"])
        all_uris.extend(job["tree_object_uris"])
    old_result_prefix = (
        f"gs://{payload_transport.BUCKET}/"
        f"{payload_transport.DIRECT_NAMESPACE}/stages/"
    )
    if (
        payload_binding["package_prefix"].startswith(stage_prefix + "/")
        or stage_prefix.startswith(payload_binding["package_prefix"] + "/")
        or stage_prefix.startswith(old_result_prefix)
        or len(all_uris) != len(set(all_uris))
    ):
        raise ValueError("deployment namespace collided")
    return layout


def _instances(
    *,
    names: Mapping[str, Any],
    payload_binding: Mapping[str, Any],
    direct_stage_identity_sha256: str,
) -> list[dict[str, Any]]:
    instances = [
        {
            "job_id": external_job_id,
            "inner_job_id": inner["inner_job_id"],
            "source_role": role,
            "instance_name": _instance_name(
                role=role,
                direct_stage_identity_sha256=direct_stage_identity_sha256,
            ),
            "attempt_index": ATTEMPT_INDEX,
            "machine_type": ACTUAL_MACHINE_TYPE,
            "actual_vcpus": ACTUAL_VCPUS_PER_VM,
            "payload_contract_sha256": inner[
                "payload_contract_sha256"
            ],
            "runner_job_manifest_sha256": inner[
                "runner_job_manifest_sha256"
            ],
            "work_hand_indices": list(inner["work_hand_indices"]),
        }
        for role, external_job_id, inner in zip(
            _ROLES,
            names["job_ids"],
            payload_binding["job_bindings"],
            strict=True,
        )
    ]
    if len({row["instance_name"] for row in instances}) != VM_COUNT:
        raise ValueError("deployment instance names collided")
    return instances


def build_step12b_deployment_contract(
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    bootstrap_source_content_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic, non-authorizing Step 12b pair identity."""

    nonce = _sha(run_nonce, "Step12b run nonce")
    candidate, reference, public_record = _validated_payloads(
        candidate_payload_contract,
        reference_payload_contract,
        controller_public_key_record,
    )
    contracts = (candidate, reference)
    controller_key_id = public_record["key_id"]
    controller_public_key_sha256 = canonical_sha256(public_record)
    payload_contract_shas = [canonical_sha256(row) for row in contracts]
    checked_source_content = (
        _validate_bootstrap_source_content_binding(
            bootstrap_source_content_binding,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )
        if bootstrap_source_content_binding is not None
        else None
    )
    seed = _identity_seed(
        run_nonce=nonce,
        controller_key_id=controller_key_id,
        controller_public_key_sha256=controller_public_key_sha256,
        payload_contract_sha256s=payload_contract_shas,
        bootstrap_source_content_binding=checked_source_content,
    )
    seed_sha = canonical_sha256(seed)
    names = _external_names(seed_sha)
    controller_service_account = _run_scoped_controller_service_account(
        names["tag"]
    )
    run_identity = _run_identity(
        seed=seed,
        seed_sha256=seed_sha,
        names=names,
        controller_service_account=controller_service_account,
    )
    run_identity_sha = run_identity["run_identity_sha256"]
    payload_binding = _payload_binding(
        candidate,
        reference,
        controller_key_id=controller_key_id,
        controller_public_key_sha256=controller_public_key_sha256,
    )
    direct_identity = _direct_identity(
        run_identity_sha256=run_identity_sha,
        controller_key_id=controller_key_id,
        controller_public_key_sha256=controller_public_key_sha256,
        controller_service_account=controller_service_account,
        names=names,
        payload_binding=payload_binding,
    )
    direct_sha = direct_identity["direct_stage_identity_sha256"]
    if direct_sha == EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256:
        raise ValueError("deployment direct identity reused the payload identity")
    remote_layout = _remote_layout(
        names=names,
        direct_identity=direct_identity,
        payload_binding=payload_binding,
    )
    instances = _instances(
        names=names,
        payload_binding=payload_binding,
        direct_stage_identity_sha256=direct_sha,
    )

    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "deployment_version": "step12b-v2",
        "run_nonce": nonce,
        "run_identity_seed": seed,
        "run_identity_seed_sha256": seed_sha,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha,
        "controller_key_id": controller_key_id,
        "controller_public_key_sha256": controller_public_key_sha256,
        "controller_service_account": controller_service_account,
        "stage_kind": STAGE_KIND,
        "stage_id": names["stage_id"],
        "run_name": names["run_name"],
        "selected_job_ids": list(names["job_ids"]),
        "source_roles": list(_ROLES),
        "attempt_index": ATTEMPT_INDEX,
        "vm_count": VM_COUNT,
        "payload_binding": payload_binding,
        **(
            {
                "bootstrap_source_content_binding": copy.deepcopy(
                    checked_source_content
                )
            }
            if checked_source_content is not None
            else {}
        ),
        "direct_stage_identity": direct_identity,
        "direct_stage_identity_sha256": direct_sha,
        "remote_layout": remote_layout,
        "instances": instances,
        "freshness_boundary": {
            "nonce_reuse_forbidden": True,
            "external_ids_differ_from_inner_payload_ids": True,
            "direct_v2_namespace_required": True,
            "entire_stage_prefix_must_be_empty_before_attempt0": True,
            "exact_instance_and_disk_get_404_required_before_insert": True,
            "freshness_observation_performed": False,
        },
        "authorization_boundary": {
            "attempt_limit_per_job": MAX_ATTEMPTS,
            "attempt1_authorized": False,
            "resume_authorized": False,
            "third_vm_authorized": False,
            "package_write_authorized": False,
            "payload_result_write_authorized": False,
            "deployment_result_write_authorized": False,
            "iam_mutation_authorized": False,
            "vm_create_authorized": False,
            "launch_authorized": False,
            "separate_cloud_authorization_required": True,
            "cloud_mutation_performed": False,
        },
        "capabilities": {
            "immutable_payload_mapping_bound": True,
            "external_deployment_identity_built": True,
            **(
                {"bootstrap_source_content_bound": True}
                if checked_source_content is not None
                else {}
            ),
            "worker_bridge_implemented": False,
            "cloud_execution_consumable": False,
            "cloud_executable": False,
            "launch_ready": False,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
        "diagnostic_only": True,
        "scientific_payload_present": False,
        "current_profile_sha256": EXPECTED_CURRENT_PROFILE_SHA256,
        "current_profile_changed": False,
    }
    _reject_forbidden(body)
    return {
        **body,
        "deployment_contract_sha256": canonical_sha256(body),
    }


def validate_step12b_deployment_contract(
    value: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    bootstrap_source_content_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild and exactly compare every deployment-contract field."""

    candidate = copy.deepcopy(dict(value))
    supplied_sha = _sha(
        candidate.pop("deployment_contract_sha256", None),
        "Step12b deployment contract digest",
    )
    if canonical_sha256(candidate) != supplied_sha:
        raise ValueError("Step12b deployment contract digest changed")
    embedded_source_content = candidate.get(
        "bootstrap_source_content_binding"
    )
    if (
        bootstrap_source_content_binding is not None
        and embedded_source_content != bootstrap_source_content_binding
    ):
        raise ValueError("Step12b bootstrap source content binding changed")
    expected = build_step12b_deployment_contract(
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        bootstrap_source_content_binding=embedded_source_content,
    )
    if dict(value) != expected:
        raise ValueError("Step12b deployment contract changed")
    return expected


def validate_role_runtime_view(
    value: Mapping[str, Any],
    *,
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
) -> dict[str, Any]:
    """Validate one VM's role-local view without the other payload contract.

    The controller must use :func:`validate_step12b_deployment_contract`
    before signing because it has both payload contracts.  A VM receives the
    complete signed deployment contract but only its selected immutable
    payload.  This validator recomputes all deployment-internal digests and
    layouts, then proves the selected payload's exact digest and inner mapping.
    Trust in the unobserved role is supplied by the later controller signature,
    not by inventing or reconstructing the missing payload.
    """

    if not isinstance(selected_payload_contract, Mapping):
        raise ValueError("selected payload contract is required")
    deployment = copy.deepcopy(dict(value))
    supplied_sha = _sha(
        deployment.pop("deployment_contract_sha256", None),
        "Step12b deployment contract digest",
    )
    if canonical_sha256(deployment) != supplied_sha:
        raise ValueError("Step12b deployment contract digest changed")
    source_content_binding = deployment.get(
        "bootstrap_source_content_binding"
    )
    deployment_fields = {
        "schema",
        "status",
        "deployment_version",
        "run_nonce",
        "run_identity_seed",
        "run_identity_seed_sha256",
        "run_identity",
        "run_identity_sha256",
        "controller_key_id",
        "controller_public_key_sha256",
        "controller_service_account",
        "stage_kind",
        "stage_id",
        "run_name",
        "selected_job_ids",
        "source_roles",
        "attempt_index",
        "vm_count",
        "payload_binding",
        "direct_stage_identity",
        "direct_stage_identity_sha256",
        "remote_layout",
        "instances",
        "freshness_boundary",
        "authorization_boundary",
        "capabilities",
        "diagnostic_only",
        "scientific_payload_present",
        "current_profile_sha256",
        "current_profile_changed",
    }
    if source_content_binding is not None:
        deployment_fields.add("bootstrap_source_content_binding")
    _exact(
        deployment,
        deployment_fields,
        "Step12b role-runtime deployment",
    )
    nonce = _sha(run_nonce, "Step12b run nonce")
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    public_sha = canonical_sha256(public_record)
    if (
        deployment["schema"] != SCHEMA
        or deployment["status"] != STATUS
        or deployment["deployment_version"] != "step12b-v2"
        or deployment["run_nonce"] != nonce
        or deployment["controller_key_id"] != public_record["key_id"]
        or deployment["controller_public_key_sha256"] != public_sha
        or deployment["stage_kind"] != STAGE_KIND
        or deployment["source_roles"] != list(_ROLES)
        or deployment["attempt_index"] != ATTEMPT_INDEX
        or deployment["vm_count"] != VM_COUNT
        or not isinstance(external_job_id, str)
        or external_job_id not in deployment["selected_job_ids"]
    ):
        raise ValueError("Step12b role-runtime top-level identity changed")

    seed = deployment["run_identity_seed"]
    if not isinstance(seed, Mapping):
        raise ValueError("Step12b run identity seed is missing")
    seed_fields = {
        "schema",
        "run_nonce",
        "controller_key_id",
        "controller_public_key_sha256",
        "outer_package_identity_sha256",
        "inner_preview_stage_identity_sha256",
        "payload_contract_sha256s",
    }
    if source_content_binding is not None:
        seed_fields.add("bootstrap_source_content_binding")
    _exact(
        seed,
        seed_fields,
        "Step12b run identity seed",
    )
    payload_shas = seed["payload_contract_sha256s"]
    if (
        seed["schema"]
        != (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_run_identity_seed_v2"
        )
        or seed["run_nonce"] != nonce
        or seed["controller_key_id"] != public_record["key_id"]
        or seed["controller_public_key_sha256"] != public_sha
        or seed["outer_package_identity_sha256"]
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or seed["inner_preview_stage_identity_sha256"]
        != EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
        or not isinstance(payload_shas, list)
        or len(payload_shas) != VM_COUNT
        or len(set(payload_shas)) != VM_COUNT
    ):
        raise ValueError("Step12b run identity seed changed")
    for position, sha in enumerate(payload_shas):
        _sha(sha, f"payload contract {position}")
    seed_sha = canonical_sha256(seed)
    if deployment["run_identity_seed_sha256"] != seed_sha:
        raise ValueError("Step12b run identity seed digest changed")
    names = _external_names(seed_sha)
    controller_service_account = _run_scoped_controller_service_account(
        names["tag"]
    )
    if (
        deployment["stage_id"] != names["stage_id"]
        or deployment["run_name"] != names["run_name"]
        or deployment["selected_job_ids"] != names["job_ids"]
        or deployment["controller_service_account"]
        != controller_service_account
    ):
        raise ValueError("Step12b external names changed")
    expected_run_identity = _run_identity(
        seed=seed,
        seed_sha256=seed_sha,
        names=names,
        controller_service_account=controller_service_account,
    )
    if (
        deployment["run_identity"] != expected_run_identity
        or deployment["run_identity_sha256"]
        != expected_run_identity["run_identity_sha256"]
    ):
        raise ValueError("Step12b run identity changed")

    payload_binding = deployment["payload_binding"]
    if not isinstance(payload_binding, Mapping):
        raise ValueError("Step12b payload binding is missing")
    _exact(
        payload_binding,
        {
            "immutable_outer_package_reused",
            "outer_package_identity_sha256",
            "outer_package_manifest_sha256",
            "controller_key_id",
            "controller_public_key_sha256",
            "package_prefix",
            "package_inventory_records_sha256",
            "package_object_count",
            "inner_stage_id",
            "inner_run_name",
            "inner_preview_stage_identity_sha256",
            "inner_direct_stage_identity_sha256",
            "inner_job_ids",
            "payload_contract_sha256s",
            "job_bindings",
            "payload_contracts_are_local_execution_only",
            "payload_result_prefixes_are_provenance_only",
            "payload_result_writes_authorized",
            "package_write_authorized",
            "payload_binding_sha256",
        },
        "Step12b payload binding",
    )
    binding_body = dict(payload_binding)
    binding_sha = _sha(
        binding_body.pop("payload_binding_sha256", None),
        "payload binding",
    )
    job_bindings = payload_binding["job_bindings"]
    if (
        canonical_sha256(binding_body) != binding_sha
        or payload_binding["payload_contract_sha256s"] != payload_shas
        or payload_binding["immutable_outer_package_reused"] is not True
        or payload_binding["outer_package_identity_sha256"]
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or payload_binding["outer_package_manifest_sha256"]
        != EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        or payload_binding["controller_key_id"] != public_record["key_id"]
        or payload_binding["controller_public_key_sha256"] != public_sha
        or payload_binding["inner_stage_id"] != INNER_STAGE_ID
        or payload_binding["inner_run_name"] != INNER_RUN_NAME
        or payload_binding["inner_preview_stage_identity_sha256"]
        != EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
        or payload_binding["inner_direct_stage_identity_sha256"]
        != EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
        or payload_binding["inner_job_ids"] != list(INNER_JOB_IDS)
        or not isinstance(job_bindings, list)
        or len(job_bindings) != VM_COUNT
        or [
            row.get("source_role")
            for row in job_bindings
            if isinstance(row, Mapping)
        ]
        != list(_ROLES)
        or [
            row.get("inner_job_id")
            for row in job_bindings
            if isinstance(row, Mapping)
        ]
        != list(INNER_JOB_IDS)
        or payload_binding[
            "payload_contracts_are_local_execution_only"
        ]
        is not True
        or payload_binding["payload_result_prefixes_are_provenance_only"]
        is not True
        or payload_binding["payload_result_writes_authorized"] is not False
        or payload_binding["package_write_authorized"] is not False
    ):
        raise ValueError("Step12b payload binding changed")

    position = names["job_ids"].index(external_job_id)
    role = _ROLES[position]
    selected = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    selected_sha = canonical_sha256(selected)
    if source_content_binding is not None:
        checked_source_content = (
            _validate_role_bootstrap_source_content_binding(
                source_content_binding,
                selected_payload_contract=selected,
                position=position,
            )
        )
        if (
            seed.get("bootstrap_source_content_binding")
            != checked_source_content
        ):
            raise ValueError(
                "Step12b seed bootstrap source content binding changed"
            )
    selected_binding = job_bindings[position]
    preview = selected["adapter_preview"]
    layout = selected["remote_layout"]
    preview_jobs = preview.get("jobs")
    trust = selected["authorization_contract"]
    capabilities = selected["capabilities"]
    if (
        selected_sha != payload_shas[position]
        or selected_binding["payload_contract_sha256"] != selected_sha
        or selected["adapter_preview_sha256"]
        != EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        or canonical_sha256(preview)
        != EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        or preview.get("capabilities")
        != EXPECTED_INNER_PREVIEW_CAPABILITIES
        or preview.get("stage_id") != INNER_STAGE_ID
        or preview.get("run_name") != INNER_RUN_NAME
        or preview.get("selected_job_ids") != list(INNER_JOB_IDS)
        or preview.get("stage_identity_sha256")
        != EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256
        or not isinstance(preview_jobs, list)
        or [row.get("job_id") for row in preview_jobs]
        != list(INNER_JOB_IDS)
        or [row.get("source_role") for row in preview_jobs]
        != list(_ROLES)
        or selected["outer_package_manifest_sha256"]
        != EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        or selected["outer_package_manifest"].get(
            "outer_package_identity_sha256"
        )
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or selected["direct_stage_identity_sha256"]
        != EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
        or trust.get("controller_key_id") != public_record["key_id"]
        or trust.get("controller_public_key_sha256") != public_sha
        or capabilities.get("cloud_executable") is not False
        or capabilities.get("launch_ready") is not False
        or capabilities.get("cloud_launch_authorized") is not False
        or capabilities.get("current_profile_changed") is not False
        or selected["metadata_binding"].get("job_id")
        != INNER_JOB_IDS[position]
        or selected["metadata_binding"].get("source_role") != role
        or selected["metadata_binding"].get("attempt_index")
        != ATTEMPT_INDEX
    ):
        raise ValueError("selected immutable payload identity changed")
    selected_preview_job = preview_jobs[position]
    tree_paths = [
        _safe_tree_path(row.get("path"))
        for row in selected_preview_job["tree_object_manifest"]
    ]
    inventory = layout["package_inventory"]
    expected_selected_binding = {
        "source_role": role,
        "inner_job_id": INNER_JOB_IDS[position],
        "inner_job_manifest_path": (
            f"inner/jobs/{INNER_JOB_IDS[position]}.json"
        ),
        "runner_job_manifest_sha256": selected_preview_job[
            "runner_job_manifest"
        ]["sha256"],
        "payload_contract_sha256": selected_sha,
        "work_hand_indices": list(INNER_HAND_INDICES),
        "root_records_sha256": canonical_sha256(
            selected_preview_job["root_records"]
        ),
        "tree_paths": tree_paths,
        "tree_paths_sha256": canonical_sha256(tree_paths),
    }
    if (
        selected_binding != expected_selected_binding
        or inventory.get("records_sha256")
        != payload_binding["package_inventory_records_sha256"]
        or len(inventory.get("records", []))
        != payload_binding["package_object_count"]
        or layout.get("package_prefix")
        != payload_binding["package_prefix"]
    ):
        raise ValueError("selected payload source binding changed")

    expected_direct = _direct_identity(
        run_identity_sha256=expected_run_identity[
            "run_identity_sha256"
        ],
        controller_key_id=public_record["key_id"],
        controller_public_key_sha256=public_sha,
        controller_service_account=controller_service_account,
        names=names,
        payload_binding=payload_binding,
    )
    if (
        deployment["direct_stage_identity"] != expected_direct
        or deployment["direct_stage_identity_sha256"]
        != expected_direct["direct_stage_identity_sha256"]
    ):
        raise ValueError("Step12b direct-stage identity changed")
    expected_layout = _remote_layout(
        names=names,
        direct_identity=expected_direct,
        payload_binding=payload_binding,
    )
    expected_instances = _instances(
        names=names,
        payload_binding=payload_binding,
        direct_stage_identity_sha256=expected_direct[
            "direct_stage_identity_sha256"
        ],
    )
    if (
        deployment["remote_layout"] != expected_layout
        or deployment["instances"] != expected_instances
    ):
        raise ValueError("Step12b runtime layout changed")
    if deployment["freshness_boundary"] != {
        "nonce_reuse_forbidden": True,
        "external_ids_differ_from_inner_payload_ids": True,
        "direct_v2_namespace_required": True,
        "entire_stage_prefix_must_be_empty_before_attempt0": True,
        "exact_instance_and_disk_get_404_required_before_insert": True,
        "freshness_observation_performed": False,
    }:
        raise ValueError("Step12b freshness boundary changed")
    if deployment["authorization_boundary"] != {
        "attempt_limit_per_job": MAX_ATTEMPTS,
        "attempt1_authorized": False,
        "resume_authorized": False,
        "third_vm_authorized": False,
        "package_write_authorized": False,
        "payload_result_write_authorized": False,
        "deployment_result_write_authorized": False,
        "iam_mutation_authorized": False,
        "vm_create_authorized": False,
        "launch_authorized": False,
        "separate_cloud_authorization_required": True,
        "cloud_mutation_performed": False,
    }:
        raise ValueError("Step12b authorization boundary changed")
    expected_capabilities = {
        "immutable_payload_mapping_bound": True,
        "external_deployment_identity_built": True,
        "worker_bridge_implemented": False,
        "cloud_execution_consumable": False,
        "cloud_executable": False,
        "launch_ready": False,
        "current_profile_changed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    if source_content_binding is not None:
        expected_capabilities["bootstrap_source_content_bound"] = True
    if deployment["capabilities"] != expected_capabilities:
        raise ValueError("Step12b capability boundary changed")
    if (
        deployment["diagnostic_only"] is not True
        or deployment["scientific_payload_present"] is not False
        or deployment["current_profile_sha256"]
        != EXPECTED_CURRENT_PROFILE_SHA256
        or deployment["current_profile_changed"] is not False
    ):
        raise ValueError("Step12b diagnostic boundary changed")
    _reject_forbidden(deployment)
    return {**deployment, "deployment_contract_sha256": supplied_sha}


def _validate_embedded_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    supplied = _sha(checked.pop(digest_field, None), label)
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} changed")
    return dict(value)


def validate_role_local_runtime_view(
    deployment_contract: Mapping[str, Any],
    *,
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    external_job_id: str,
) -> dict[str, Any]:
    """Validate the signed-runtime inputs available to exactly one role.

    This deliberately does not claim to rebuild or validate the unselected
    payload contract.  The controller must perform the full two-payload
    ``validate_deployment_contract`` check before signing the deployment.
    The VM then combines this role-local structural view with the separately
    verified external authorization signature.
    """

    deployment = _validate_embedded_digest(
        deployment_contract,
        digest_field="deployment_contract_sha256",
        label="role-local deployment contract digest",
    )
    source_content_binding = deployment.get(
        "bootstrap_source_content_binding"
    )
    deployment_fields = {
        "schema",
        "status",
        "deployment_version",
        "run_nonce",
        "run_identity_seed",
        "run_identity_seed_sha256",
        "run_identity",
        "run_identity_sha256",
        "controller_key_id",
        "controller_public_key_sha256",
        "controller_service_account",
        "stage_kind",
        "stage_id",
        "run_name",
        "selected_job_ids",
        "source_roles",
        "attempt_index",
        "vm_count",
        "payload_binding",
        "direct_stage_identity",
        "direct_stage_identity_sha256",
        "remote_layout",
        "instances",
        "freshness_boundary",
        "authorization_boundary",
        "capabilities",
        "diagnostic_only",
        "scientific_payload_present",
        "current_profile_sha256",
        "current_profile_changed",
        "deployment_contract_sha256",
    }
    if source_content_binding is not None:
        deployment_fields.add("bootstrap_source_content_binding")
    _exact(
        deployment,
        deployment_fields,
        "role-local deployment contract",
    )
    public_record = payload_transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    public_record_sha256 = canonical_sha256(public_record)
    if not isinstance(selected_payload_contract, Mapping):
        raise ValueError("selected payload contract is missing")
    selected_payload = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    selected_payload_sha256 = canonical_sha256(selected_payload)

    seed = deployment["run_identity_seed"]
    if (
        deployment["schema"] != SCHEMA
        or deployment["status"] != STATUS
        or deployment["deployment_version"] != "step12b-v2"
        or deployment["stage_kind"] != STAGE_KIND
        or deployment["attempt_index"] != ATTEMPT_INDEX
        or deployment["vm_count"] != VM_COUNT
        or deployment["source_roles"] != list(_ROLES)
        or deployment["diagnostic_only"] is not True
        or deployment["scientific_payload_present"] is not False
        or deployment["current_profile_sha256"]
        != EXPECTED_CURRENT_PROFILE_SHA256
        or deployment["current_profile_changed"] is not False
        or deployment["controller_key_id"] != public_record["key_id"]
        or deployment["controller_public_key_sha256"]
        != public_record_sha256
        or not isinstance(seed, Mapping)
        or canonical_sha256(seed)
        != deployment["run_identity_seed_sha256"]
    ):
        raise ValueError("role-local deployment authority changed")

    run_identity = _validate_embedded_digest(
        deployment["run_identity"],
        digest_field="run_identity_sha256",
        label="role-local run identity digest",
    )
    payload_binding = _validate_embedded_digest(
        deployment["payload_binding"],
        digest_field="payload_binding_sha256",
        label="role-local payload binding digest",
    )
    direct_identity = _validate_embedded_digest(
        deployment["direct_stage_identity"],
        digest_field="direct_stage_identity_sha256",
        label="role-local direct-stage identity digest",
    )
    if (
        run_identity["run_identity_sha256"]
        != deployment["run_identity_sha256"]
        or direct_identity["direct_stage_identity_sha256"]
        != deployment["direct_stage_identity_sha256"]
        or direct_identity["run_identity_sha256"]
        != deployment["run_identity_sha256"]
        or direct_identity["payload_binding_sha256"]
        != payload_binding["payload_binding_sha256"]
        or deployment["remote_layout"]["direct_stage_identity_sha256"]
        != deployment["direct_stage_identity_sha256"]
        or run_identity["controller_key_id"] != public_record["key_id"]
        or direct_identity["controller_key_id"] != public_record["key_id"]
        or payload_binding["controller_key_id"] != public_record["key_id"]
        or run_identity["controller_public_key_sha256"]
        != public_record_sha256
        or direct_identity["controller_public_key_sha256"]
        != public_record_sha256
        or payload_binding["controller_public_key_sha256"]
        != public_record_sha256
    ):
        raise ValueError("role-local signed identity chain changed")

    selected_job_ids = deployment["selected_job_ids"]
    if (
        not isinstance(external_job_id, str)
        or not isinstance(selected_job_ids, list)
        or selected_job_ids.count(external_job_id) != 1
    ):
        raise ValueError("role-local external job selection changed")
    position = selected_job_ids.index(external_job_id)
    checked_source_content = None
    if source_content_binding is not None:
        checked_source_content = (
            _validate_role_bootstrap_source_content_binding(
                source_content_binding,
                selected_payload_contract=selected_payload,
                position=position,
            )
        )
        if (
            seed.get("bootstrap_source_content_binding")
            != checked_source_content
            or deployment["capabilities"].get(
                "bootstrap_source_content_bound"
            )
            is not True
        ):
            raise ValueError(
                "role-local bootstrap source content binding changed"
            )
    elif "bootstrap_source_content_binding" in seed:
        raise ValueError(
            "role-local bootstrap source content binding presence changed"
        )
    parallel = (
        deployment["source_roles"],
        deployment["instances"],
        payload_binding["job_bindings"],
        direct_identity["external_job_layout"],
        deployment["remote_layout"]["jobs"],
    )
    if any(not isinstance(rows, list) or len(rows) != VM_COUNT for rows in parallel):
        raise ValueError("role-local two-role mapping shape changed")
    role = deployment["source_roles"][position]
    instance = deployment["instances"][position]
    binding = payload_binding["job_bindings"][position]
    direct_job = direct_identity["external_job_layout"][position]
    layout_job = deployment["remote_layout"]["jobs"][position]
    inner_job_id = binding["inner_job_id"]
    payload_binding_record = selected_payload["metadata_binding"]
    preview_jobs = selected_payload["adapter_preview"]["jobs"]
    payload_job = next(
        (
            row
            for row in preview_jobs
            if row.get("job_id") == inner_job_id
        ),
        None,
    )
    common = (instance, direct_job, layout_job)
    if (
        role not in _ROLES
        or inner_job_id != INNER_JOB_IDS[position]
        or binding["source_role"] != role
        or payload_binding_record["job_id"] != inner_job_id
        or payload_binding_record["source_role"] != role
        or payload_binding_record["attempt_index"] != ATTEMPT_INDEX
        or payload_job is None
        or payload_job.get("source_role") != role
        or payload_job.get("work_hand_indices")
        != list(INNER_HAND_INDICES)
        or any(row.get("job_id") != external_job_id for row in common)
        or any(row.get("inner_job_id") != inner_job_id for row in common)
        or any(row.get("source_role") != role for row in common)
        or any(
            row.get("payload_contract_sha256")
            != selected_payload_sha256
            for row in (instance, direct_job)
        )
        or binding["payload_contract_sha256"]
        != selected_payload_sha256
        or payload_binding["payload_contract_sha256s"][position]
        != selected_payload_sha256
        or run_identity["payload_contract_sha256s"][position]
        != selected_payload_sha256
        or seed["payload_contract_sha256s"][position]
        != selected_payload_sha256
    ):
        raise ValueError("role-local external-to-inner payload mapping changed")

    trust = selected_payload["authorization_contract"]
    inventory = selected_payload["remote_layout"]["package_inventory"]
    source_records = [
        dict(row)
        for row in inventory["records"]
        if row["path"].startswith(("scripts/", "src/"))
    ]
    payload_runner_records = (
        [
            row
            for row in payload_job.get("output_control_records", [])
            if row.get("path") == "shard_manifest.json"
        ]
        if isinstance(payload_job, Mapping)
        else []
    )
    if (
        trust.get("controller_key_id") != public_record["key_id"]
        or trust.get("controller_public_key_sha256")
        != public_record_sha256
        or selected_payload["outer_package_manifest"][
            "outer_package_identity_sha256"
        ]
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or selected_payload["outer_package_manifest_sha256"]
        != EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        or selected_payload["adapter_preview_sha256"]
        != EXPECTED_INNER_ADAPTER_PREVIEW_SHA256
        or selected_payload["direct_stage_identity_sha256"]
        != EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256
        or inventory["outer_package_identity_sha256"]
        != EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        or inventory["records_sha256"]
        != payload_binding["package_inventory_records_sha256"]
        or len(inventory["records"]) != payload_binding["package_object_count"]
        or inventory["package_prefix"] != payload_binding["package_prefix"]
        or deployment["remote_layout"]["package_prefix"]
        != payload_binding["package_prefix"]
        or binding["runner_job_manifest_sha256"]
        != instance["runner_job_manifest_sha256"]
        or binding["runner_job_manifest_sha256"]
        != direct_job["runner_job_manifest_sha256"]
        or len(payload_runner_records) != 1
        or binding["runner_job_manifest_sha256"]
        != payload_runner_records[0].get("sha256")
        or not source_records
    ):
        raise ValueError("role-local package or source identity changed")

    body = {
        "schema": ROLE_LOCAL_RUNTIME_VIEW_SCHEMA,
        "validation_scope": (
            "selected_role_only_after_controller_full_deployment_validation"
        ),
        "unselected_payload_reconstructed": False,
        "unselected_payload_validated_on_vm": False,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "run_identity_sha256": deployment["run_identity_sha256"],
        "direct_stage_identity_sha256": deployment[
            "direct_stage_identity_sha256"
        ],
        "controller_key_id": public_record["key_id"],
        "controller_public_key_sha256": public_record_sha256,
        "external_job_id": external_job_id,
        "inner_job_id": inner_job_id,
        "source_role": role,
        "instance_name": instance["instance_name"],
        "attempt_index": ATTEMPT_INDEX,
        "selected_payload_contract_sha256": selected_payload_sha256,
        "selected_payload_metadata_binding_sha256": selected_payload[
            "metadata_binding_sha256"
        ],
        "selected_payload_authorization_binding_sha256": canonical_sha256(
            trust
        ),
        "outer_package_identity_sha256": (
            EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256
        ),
        "outer_package_manifest_sha256": (
            EXPECTED_OUTER_PACKAGE_MANIFEST_SHA256
        ),
        "package_inventory_records_sha256": inventory["records_sha256"],
        "package_object_count": len(inventory["records"]),
        "package_source_records_sha256": canonical_sha256(source_records),
        "runner_job_manifest_sha256": binding[
            "runner_job_manifest_sha256"
        ],
        "root_records_sha256": binding["root_records_sha256"],
        "tree_paths_sha256": binding["tree_paths_sha256"],
        **(
            {
                "bootstrap_source_content_binding_sha256": (
                    checked_source_content[
                        "bootstrap_source_content_binding_sha256"
                    ]
                )
            }
            if checked_source_content is not None
            else {}
        ),
    }
    _reject_forbidden(body)
    return {
        **body,
        "role_local_runtime_view_sha256": canonical_sha256(body),
    }


build_deployment_contract = build_step12b_deployment_contract
validate_deployment_contract = validate_step12b_deployment_contract


__all__ = [
    "ACTUAL_MACHINE_TYPE",
    "ATTEMPT_INDEX",
    "DIRECT_NAMESPACE",
    "EXPECTED_CURRENT_PROFILE_SHA256",
    "EXPECTED_INNER_ADAPTER_PREVIEW_SHA256",
    "EXPECTED_INNER_DIRECT_STAGE_IDENTITY_SHA256",
    "EXPECTED_INNER_PREVIEW_STAGE_IDENTITY_SHA256",
    "EXPECTED_OUTER_PACKAGE_IDENTITY_SHA256",
    "INNER_HAND_INDICES",
    "INNER_JOB_IDS",
    "INNER_MACHINE_TYPE",
    "INNER_RAYON_THREADS",
    "INNER_RUN_NAME",
    "INNER_STAGE_ID",
    "LEGACY_SHARED_CONTROLLER_SERVICE_ACCOUNT",
    "MAX_ATTEMPTS",
    "ROLE_LOCAL_RUNTIME_VIEW_SCHEMA",
    "RUN_IDENTITY_SCHEMA",
    "SCHEMA",
    "STATUS",
    "VM_COUNT",
    "build_deployment_contract",
    "build_step12b_deployment_contract",
    "canonical_bytes",
    "canonical_sha256",
    "validate_deployment_contract",
    "validate_role_runtime_view",
    "validate_role_local_runtime_view",
    "validate_step12b_deployment_contract",
]
