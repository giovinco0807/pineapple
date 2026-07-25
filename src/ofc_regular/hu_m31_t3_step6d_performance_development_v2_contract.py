"""Frozen, development-only contract for the next T3 tail performance probe.

This module deliberately stops before package construction or cloud execution.
It rehashes the accepted plan, merge summary, and validation receipt, then
binds their stored Candidate02 binary/root/seed digests.  It does *not* claim
that the current binary or root files have been rehashed.  That byte-level gate
is mandatory in the future package/launch slice.  The next diagnostic is fixed
to the ten preregistered tail hands.  Performance-lock rearm2 roots, packages,
seeds, and namespaces are recorded only as a denylist; none of them may be
consumed by this development run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import hu_m31_t3_step6d_candidate02_full100_plan as full100
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as selector


CONTRACT_SCHEMA = "hu_m31_t3_step6d_performance_development_v2_contract_v2"
CONTRACT_STATUS = "local_development_tail_contract_ready_cloud_not_executable"
CONTRACT_SCOPE = "candidate02_repeatable_performance_development_tail_only"

MACHINE_TYPE = "c4-standard-16"
GUEST_VCPUS = 16
WORKERS_PER_SOURCE = 1
RAYON_THREADS_PER_WORKER = 16
SOURCE_ROLES = ("candidate", "reference")
FULL100_PLAN_TAIL_HAND_INDICES = tuple(runner.TAIL_HAND_INDICES)
TAIL_HAND_INDICES = tuple(runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES)
TAIL_HEAVY_HAND_INDICES = tuple(runner.CANDIDATE02_TAIL_V2_HEAVY_HAND_INDICES)
TAIL_RANDOM_HAND_INDICES = tuple(runner.CANDIDATE02_TAIL_V2_RANDOM_HAND_INDICES)
TAIL_PRIOR_EXPOSED_HAND_INDICES = tuple(
    runner.CANDIDATE02_TAIL_V2_PRIOR_EXPOSED_HAND_INDICES
)
MIN_REQUIRED_C4_VCPUS = 32
MAX_SPOT_PRICE_USD_PER_VM_HOUR = "0.57"

FULL100_PLAN_SHA256 = (
    "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758"
)
ACCEPTED_SUMMARY_SHA256 = (
    "f587df6d037313e2111a5cbc3d474370106f6a341ef4ec130cee7516192e668f"
)
ACCEPTED_VALIDATION_SHA256 = (
    "da252b3cabfc361d89de2f2fe08931fe61302287b69b82bf9f812a4432baf3eb"
)
FULL_RUN_CONTRACT_DIGEST = (
    "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
)
TAIL_RUN_CONTRACT_DIGEST = (
    "b5d3114d0857723809ec85cef957921acafa67e22756aef050fa1c8ef8f79bf6"
)
TAIL_RUN_CONTRACT_SCHEMA = runner.CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA
TAIL_RUN_CONTRACT_VARIANT = runner.CANDIDATE02_TAIL_V2_VARIANT
TAIL_SELECTION_MANIFEST_SHA256 = (
    runner.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
)
TAIL_SELECTION_MANIFEST_BYTES = 1523
CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
DEVELOPMENT_ROOT_SET_SHA256 = (
    "0aacb1b7f9b3c45a7ca51d9e58218e55ef3cc29159e2d431757806be076e6796"
)
DEVELOPMENT_ROOT_TOPOLOGY_SHA256 = (
    "779e6a6d2ce6d84ccb31df6e02c48efdb7833ef06dab0625aeeab551d3d81633"
)
DEVELOPMENT_SEED_SET_SHA256 = (
    "173cd8d27fe918cab4552b89bbf5fd7929a3e0bd6955929bd5d4cc7d299f0c16"
)

# These values identify the consumed performance-lock rearm2 evidence.  They
# are frozen here only to reject reuse; the development contract never reads
# the external D: artifacts that originally contained them.
REARM2_GLOBAL_CLAIM_SHA256 = (
    "e7f53bc050319f46059bd315e4c9649dc1534bf2c91b87b7fbc547eb1a1309d0"
)
REARM2_PRECONTENT_PLAN_SHA256 = (
    "8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5"
)
REARM2_RUN_CONTRACT_DIGEST = (
    "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5"
)
REARM2_RUN_ID = "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v3"
REARM2_SCHEDULE = "performance_lock_recovery_v3"
REARM2_ROOT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_root_v3"
)
REARM2_ROOT_MATERIALIZATION_SHA256 = (
    "88e535478f892e7d981295cb43fab915b9ecdee89b451b851587c9747535e724"
)
REARM2_ROOT_SEAL_SHA256 = (
    "a25b2d67247c92407d1fd44ba2193d8a200800cc4c93bc004f548be4d866f4de"
)
REARM2_AGGREGATE_ROOT_SHA256 = (
    "bb0cf7fb538eab401f10464a555aa2d9b3bebf870444ecf6c0b7a42190da883b"
)
REARM2_ROOT_TOPOLOGY_SHA256 = (
    "e95e9c7f58503fa7e4b4d830e9969009d9f4ea057f878915c05a21796b50750b"
)
REARM2_PACKAGE_MANIFEST_SHA256 = (
    "92c7977f8a701c83afb53f06ca4ee96e46ba67c891c5769babdc7c70614c05cd"
)
REARM2_PACKAGE_READY_SHA256 = (
    "23024cb1819c681f89f1b3782459709abeb846ad00a673cc57dfbeec9f0357c8"
)
REARM2_PACKAGE_SOURCE_SHA256 = (
    "8aa762cd31c61b33ec8ee984786f723a9b2372b4aa673cfad20e32ed6187efea"
)
REARM2_PACKAGE_SMOKE_SHA256 = (
    "d194f1235cdeb5f74ae96d203c84b381be58c2280aca7833cc55e335395acb77"
)
REARM2_SEED_SET_SHA256 = (
    "e1e24394feb0de5ace140cdcd5f6dd87738059a88025789e69a1a991351927a8"
)
REARM2_SEED_MIN = 710_108_071_901
REARM2_SEED_MAX = 715_207_072_198
REARM2_NAMESPACE_BASES = {
    "hand": 710_108_071_901,
    "behavior": 711_108_071_901,
    "candidate": 712_108_071_901,
    "evaluation": 713_108_071_901,
    "child": 714_108_071_901,
    "confirmation": 715_108_071_901,
}
REARM2_PACKAGE_RUN_NAME = "regular-hu-m31-c02-lock-r2-20260718-001"

OLD_DEPRECATED_IMAGE = {
    "project": "debian-cloud",
    "name": "debian-12-bookworm-v20260609",
    "id": "1449487925682397051",
    "self_link": (
        "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
        "global/images/debian-12-bookworm-v20260609"
    ),
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL100_PLAN_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "full100_plan_v1.json"
)
DEFAULT_ACCEPTED_MERGE_DIR = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/full100_merge/"
    "regular-hu-m31-c02-full100-dev-20260717-002"
)
DEFAULT_TAIL_SELECTION_MANIFEST_PATH = (
    _REPO_ROOT / "configs/hu_joint_policy_m31_t3_candidate02_tail_v2_selection.json"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def validate_tail_selection_manifest(
    path: str | Path = DEFAULT_TAIL_SELECTION_MANIFEST_PATH,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if (
        sha256_file(target) != TAIL_SELECTION_MANIFEST_SHA256
        or target.stat().st_size != TAIL_SELECTION_MANIFEST_BYTES
    ):
        raise ValueError("candidate02 tail-v2 selection manifest bytes changed")
    manifest = selector.validate_selection_manifest(
        _read_canonical(target, "candidate02 tail-v2 selection manifest")
    )
    if (
        manifest["tail_hand_indices"] != list(TAIL_HAND_INDICES)
        or manifest["heavy_hand_indices"] != list(TAIL_HEAVY_HAND_INDICES)
        or manifest["random_hand_indices"] != list(TAIL_RANDOM_HAND_INDICES)
        or manifest["prior_runtime_exposed_hand_indices"]
        != list(TAIL_PRIOR_EXPOSED_HAND_INDICES)
    ):
        raise ValueError("candidate02 tail-v2 contract/selection mismatch")
    return manifest


def build_tail_run_contract() -> dict[str, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        workers=WORKERS_PER_SOURCE,
        rayon_threads_per_worker=RAYON_THREADS_PER_WORKER,
        variant=TAIL_RUN_CONTRACT_VARIANT,
    )
    return validate_tail_run_contract(contract)


def validate_tail_run_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = runner.validate_run_contract(value)
    if (
        contract.get("schema") != TAIL_RUN_CONTRACT_SCHEMA
        or runner.contract_variant(contract) != TAIL_RUN_CONTRACT_VARIANT
        or runner.canonical_sha256(contract) != TAIL_RUN_CONTRACT_DIGEST
        or contract.get("candidate_library_sha256") != CANDIDATE_LIBRARY_SHA256
        or contract.get("reference_library_sha256") != REFERENCE_LIBRARY_SHA256
        or contract.get("contract_hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or contract.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or contract.get("selection_manifest_sha256")
        != TAIL_SELECTION_MANIFEST_SHA256
        or contract.get("seed_contract", {}).get("seed_set_sha256")
        != DEVELOPMENT_SEED_SET_SHA256
    ):
        raise ValueError("candidate02 tail-v2 execution contract changed")
    return contract


def _accepted_provenance(
    *,
    full100_plan_path: str | Path,
    accepted_summary_path: str | Path,
    accepted_validation_path: str | Path,
) -> dict[str, Any]:
    plan_path = Path(full100_plan_path).resolve()
    summary_path = Path(accepted_summary_path).resolve()
    validation_path = Path(accepted_validation_path).resolve()
    if sha256_file(plan_path) != FULL100_PLAN_SHA256:
        raise ValueError("accepted Candidate02 full100 plan hash changed")
    if sha256_file(summary_path) != ACCEPTED_SUMMARY_SHA256:
        raise ValueError("accepted performance-development summary hash changed")
    if sha256_file(validation_path) != ACCEPTED_VALIDATION_SHA256:
        raise ValueError("accepted performance-development validation hash changed")

    plan = full100.validate_full100_plan(
        _read_canonical(plan_path, "Candidate02 full100 plan")
    )
    summary = _read_canonical(summary_path, "accepted development summary")
    validation = _read_canonical(
        validation_path, "accepted development validation"
    )
    run_contract = runner.validate_run_contract(plan["run_contract"])
    if (
        full100.canonical_sha256(plan) != FULL100_PLAN_SHA256
        or plan.get("run_contract_digest") != FULL_RUN_CONTRACT_DIGEST
        or runner.canonical_sha256(run_contract) != FULL_RUN_CONTRACT_DIGEST
        or run_contract.get("candidate_library_sha256")
        != CANDIDATE_LIBRARY_SHA256
        or run_contract.get("reference_library_sha256")
        != REFERENCE_LIBRARY_SHA256
        or run_contract.get("tail_hand_indices")
        != list(FULL100_PLAN_TAIL_HAND_INDICES)
        or run_contract.get("seed_contract", {}).get("seed_set_sha256")
        != DEVELOPMENT_SEED_SET_SHA256
        or plan.get("root_set", {}).get("all100_root_sha256")
        != DEVELOPMENT_ROOT_SET_SHA256
        or plan.get("root_set", {}).get("topology_sha256")
        != DEVELOPMENT_ROOT_TOPOLOGY_SHA256
    ):
        raise ValueError("accepted Candidate02 full100 provenance changed")
    if (
        summary.get("schema")
        != "hu_m31_t3_step6d_full100_received_merge_v1"
        or summary.get("status") != "pass"
        or summary.get("all_gates_passed") is not True
        or summary.get("performance_candidate_frozen") is not True
        or not isinstance(summary.get("scientific_merge"), Mapping)
        or summary["scientific_merge"].get("run_contract_digest")
        != FULL_RUN_CONTRACT_DIGEST
        or validation.get("schema")
        != "hu_m31_t3_step6d_full100_received_merge_validation_v1"
        or validation.get("status") != "pass"
        or validation.get("all_gates_passed") is not True
        or validation.get("summary_sha256") != ACCEPTED_SUMMARY_SHA256
        or validation.get("run_contract_digest") != FULL_RUN_CONTRACT_DIGEST
        or validation.get("paired_hand_count") != 100
        or validation.get("root_count") != 200
    ):
        raise ValueError("accepted performance-development result boundary changed")
    return {
        "full100_plan_sha256": FULL100_PLAN_SHA256,
        "accepted_summary_sha256": ACCEPTED_SUMMARY_SHA256,
        "accepted_validation_sha256": ACCEPTED_VALIDATION_SHA256,
        "full100_plan_run_contract_digest": FULL_RUN_CONTRACT_DIGEST,
        "candidate_library_sha256": CANDIDATE_LIBRARY_SHA256,
        "reference_library_sha256": REFERENCE_LIBRARY_SHA256,
        "root_set_sha256": DEVELOPMENT_ROOT_SET_SHA256,
        "root_topology_sha256": DEVELOPMENT_ROOT_TOPOLOGY_SHA256,
        "seed_set_sha256": DEVELOPMENT_SEED_SET_SHA256,
        "files_rehashed_now": [
            "full100_plan",
            "accepted_merge_summary",
            "accepted_merge_validation",
        ],
        "candidate_library_file_rehashed_now": False,
        "reference_library_file_rehashed_now": False,
        "root_files_rehashed_now": False,
        "seed_material_recomputed_now": False,
        "binary_root_seed_values_are_stored_digest_bindings": True,
        "historical_result_used_for_candidate_freeze_only": True,
        "historical_timing_reused_as_new_evidence": False,
    }


def _rearm2_denylist() -> dict[str, Any]:
    return {
        "policy": "deny_all_performance_lock_rearm2_material_in_development_v2",
        "claims_and_contracts": {
            "global_claim_sha256": REARM2_GLOBAL_CLAIM_SHA256,
            "precontent_plan_sha256": REARM2_PRECONTENT_PLAN_SHA256,
            "run_contract_digest": REARM2_RUN_CONTRACT_DIGEST,
            "run_id": REARM2_RUN_ID,
            "schedule": REARM2_SCHEDULE,
            "root_schema": REARM2_ROOT_SCHEMA,
            "package_run_name": REARM2_PACKAGE_RUN_NAME,
        },
        "roots": {
            "materialization_sha256": REARM2_ROOT_MATERIALIZATION_SHA256,
            "seal_sha256": REARM2_ROOT_SEAL_SHA256,
            "aggregate_root_sha256": REARM2_AGGREGATE_ROOT_SHA256,
            "root_topology_sha256": REARM2_ROOT_TOPOLOGY_SHA256,
        },
        "packages": {
            "manifest_sha256": REARM2_PACKAGE_MANIFEST_SHA256,
            "ready_sha256": REARM2_PACKAGE_READY_SHA256,
            "source_sha256": REARM2_PACKAGE_SOURCE_SHA256,
            "preauthorize_smoke_sha256": REARM2_PACKAGE_SMOKE_SHA256,
        },
        "seeds": {
            "seed_set_sha256": REARM2_SEED_SET_SHA256,
            "seed_min": REARM2_SEED_MIN,
            "seed_max": REARM2_SEED_MAX,
            "namespace_bases": dict(REARM2_NAMESPACE_BASES),
        },
        "root_reuse_allowed": False,
        "package_reuse_allowed": False,
        "seed_reuse_allowed": False,
        "namespace_reuse_allowed": False,
    }


def _build_contract(
    *,
    full100_plan_path: str | Path,
    accepted_summary_path: str | Path,
    accepted_validation_path: str | Path,
    tail_selection_manifest_path: str | Path,
) -> dict[str, Any]:
    provenance = _accepted_provenance(
        full100_plan_path=full100_plan_path,
        accepted_summary_path=accepted_summary_path,
        accepted_validation_path=accepted_validation_path,
    )
    selection = validate_tail_selection_manifest(tail_selection_manifest_path)
    tail_run_contract = build_tail_run_contract()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "scope": CONTRACT_SCOPE,
        "accepted_provenance": provenance,
        "tail_execution": {
            "amendment": (
                "replace_mixed_geometry_profile_tail_with_existing_"
                "candidate02_tail_v2_for_performance_only_remeasurement"
            ),
            "run_contract": tail_run_contract,
            "run_contract_digest": TAIL_RUN_CONTRACT_DIGEST,
            "run_contract_schema": TAIL_RUN_CONTRACT_SCHEMA,
            "candidate_variant": TAIL_RUN_CONTRACT_VARIANT,
            "selection_manifest_sha256": TAIL_SELECTION_MANIFEST_SHA256,
            "selection_manifest_bytes": TAIL_SELECTION_MANIFEST_BYTES,
            "selection_manifest_schema": selection["schema"],
            "prior_runtime_exposed_hand_indices": list(
                TAIL_PRIOR_EXPOSED_HAND_INDICES
            ),
            "prior_mixed_geometry_artifacts_qualification_allowed": False,
            "performance_only_remeasurement": True,
            "quality_or_holdout_claim_allowed": False,
        },
        "work": {
            "contract_hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "work_hand_indices": list(TAIL_HAND_INDICES),
            "source_roles": list(SOURCE_ROLES),
            "candidate_reference_same_roots": True,
            "candidate_reference_separate_processes": True,
            "candidate_reference_separate_instances": True,
            "candidate_selection_mc_reused_for_evaluation": False,
        },
        "topology": {
            "machine_type": MACHINE_TYPE,
            "guest_vcpus_per_instance": GUEST_VCPUS,
            "instances": 2,
            "instances_per_source_role": 1,
            "workers_per_source_process": WORKERS_PER_SOURCE,
            "rayon_threads_per_worker": RAYON_THREADS_PER_WORKER,
            "omp_threads": 1,
            "m3_batch_threads": 1,
            "execution_mode": "one_source_one_process_scalar_1x16",
            "minimum_required_c4_vcpus": MIN_REQUIRED_C4_VCPUS,
        },
        "tail_gates": {
            "portable_parity_required": "10/10",
            "peak_rss_bytes_max": 858_993_459,
            "heavy_first_median_seconds_max": 135,
            "heavy_first_max_seconds_max": 145,
            "candidate_reference_geomean_speedup_min": "1.55",
            "second_seat_max_seconds_max": 5,
        },
        "image_policy": {
            "caller_supplied_read_only_observation_required": True,
            "default_image_allowed": False,
            "automatic_replacement_allowed": False,
            "old_deprecated_image_denied": dict(OLD_DEPRECATED_IMAGE),
        },
        "runtime_policy": {
            "caller_supplied_read_only_price_and_quota_required": True,
            "spot_price_usd_per_vm_hour_max": MAX_SPOT_PRICE_USD_PER_VM_HOUR,
            "minimum_available_c4_vcpus": MIN_REQUIRED_C4_VCPUS,
            "minimum_available_spot_vcpus": MIN_REQUIRED_C4_VCPUS,
            "fresh_run_namespace_required": True,
            "fresh_identity_namespace_required": True,
            "fresh_result_prefix_required": True,
        },
        "future_package_launch_requirements": {
            "candidate_source_library_rehash_required": True,
            "candidate_packaged_library_rehash_required": True,
            "candidate_source_and_packaged_bytes_must_match": True,
            "reference_source_library_rehash_required": True,
            "reference_packaged_library_rehash_required": True,
            "reference_source_and_packaged_bytes_must_match": True,
            "all_100_source_root_files_rehash_required": True,
            "all_100_packaged_root_files_rehash_required": True,
            "source_and_packaged_root_bytes_must_match": True,
            "source_and_packaged_root_aggregate_must_equal_bound_digest": True,
            "tail_v2_selection_source_and_packaged_bytes_must_match": True,
            "tail_v2_selection_sha256_must_equal_bound_digest": True,
            "tail_execution_contract_must_not_reuse_full100_plan_contract": True,
            "development_seed_schedule_recompute_required": True,
            "rearm2_root_package_seed_overlap_must_be_zero": True,
            "image_price_quota_inventory_recollection_required": True,
            "prepackage_and_postpackage_checks_required": True,
            "launch_slice_must_remain_fail_closed_until_all_pass": True,
        },
        "performance_lock_rearm2_denylist": _rearm2_denylist(),
        "cloud_executable": False,
        "launch_authorized": False,
        "package_authorized": False,
        "cloud_mutated": False,
        "instances_created": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def build_contract(
    *,
    full100_plan_path: str | Path = DEFAULT_FULL100_PLAN_PATH,
    accepted_summary_path: str | Path = DEFAULT_ACCEPTED_MERGE_DIR / "summary.json",
    accepted_validation_path: str | Path = (
        DEFAULT_ACCEPTED_MERGE_DIR / "validation.json"
    ),
    tail_selection_manifest_path: str | Path = DEFAULT_TAIL_SELECTION_MANIFEST_PATH,
) -> dict[str, Any]:
    value = _build_contract(
        full100_plan_path=full100_plan_path,
        accepted_summary_path=accepted_summary_path,
        accepted_validation_path=accepted_validation_path,
        tail_selection_manifest_path=tail_selection_manifest_path,
    )
    return validate_contract(
        value,
        full100_plan_path=full100_plan_path,
        accepted_summary_path=accepted_summary_path,
        accepted_validation_path=accepted_validation_path,
        tail_selection_manifest_path=tail_selection_manifest_path,
    )


def validate_contract(
    value: Mapping[str, Any],
    *,
    full100_plan_path: str | Path = DEFAULT_FULL100_PLAN_PATH,
    accepted_summary_path: str | Path = DEFAULT_ACCEPTED_MERGE_DIR / "summary.json",
    accepted_validation_path: str | Path = (
        DEFAULT_ACCEPTED_MERGE_DIR / "validation.json"
    ),
    tail_selection_manifest_path: str | Path = DEFAULT_TAIL_SELECTION_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = dict(value)
    expected = _build_contract(
        full100_plan_path=full100_plan_path,
        accepted_summary_path=accepted_summary_path,
        accepted_validation_path=accepted_validation_path,
        tail_selection_manifest_path=tail_selection_manifest_path,
    )
    if payload != expected:
        raise ValueError("performance-development v2 contract changed")
    return payload


__all__ = [
    "ACCEPTED_SUMMARY_SHA256",
    "ACCEPTED_VALIDATION_SHA256",
    "CANDIDATE_LIBRARY_SHA256",
    "CONTRACT_SCHEMA",
    "DEVELOPMENT_ROOT_SET_SHA256",
    "DEFAULT_TAIL_SELECTION_MANIFEST_PATH",
    "FULL100_PLAN_SHA256",
    "FULL100_PLAN_TAIL_HAND_INDICES",
    "FULL_RUN_CONTRACT_DIGEST",
    "MAX_SPOT_PRICE_USD_PER_VM_HOUR",
    "MIN_REQUIRED_C4_VCPUS",
    "REFERENCE_LIBRARY_SHA256",
    "TAIL_HAND_INDICES",
    "TAIL_HEAVY_HAND_INDICES",
    "TAIL_PRIOR_EXPOSED_HAND_INDICES",
    "TAIL_RANDOM_HAND_INDICES",
    "TAIL_RUN_CONTRACT_DIGEST",
    "TAIL_RUN_CONTRACT_SCHEMA",
    "TAIL_RUN_CONTRACT_VARIANT",
    "TAIL_SELECTION_MANIFEST_BYTES",
    "TAIL_SELECTION_MANIFEST_SHA256",
    "build_contract",
    "build_tail_run_contract",
    "canonical_bytes",
    "canonical_sha256",
    "sha256_file",
    "validate_contract",
    "validate_tail_run_contract",
    "validate_tail_selection_manifest",
]
