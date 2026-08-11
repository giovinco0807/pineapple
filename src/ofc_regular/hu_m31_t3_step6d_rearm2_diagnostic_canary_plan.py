"""Freeze a local-only diagnostic canary contract for the rearm2 lifecycle.

The accepted performance-lock launcher is intentionally all-20.  It must not
be weakened to make a one-VM smoke convenient.  This module therefore freezes
two *separate* diagnostic stages against the already accepted performance-
development roots:

* stage 1: one candidate VM/job;
* stage 2: one candidate and one reference VM/job over the same roots.

This is a contract generator and validator only.  It does not build a cloud
package, write a claim or authorization, invoke gcloud, or make either stage
cloud-capable.  A new versioned startup/package/receive lifecycle is required
before these stages may be launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_full100_plan as full100_plan
from . import hu_m31_t3_step6d_full100_spot_v1 as full100_transport


CONTRACT_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_canary_contract_v1"
CONTRACT_STATUS = "local_fail_closed_contract_ready_cloud_not_authorized"
CONTRACT_SCOPE = "diagnostic_lifecycle_only_no_performance_lock_quality_or_training"

DEVELOPMENT_PLAN_RELATIVE = (
    "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "full100_plan_v1.json"
)
DEVELOPMENT_ROOTS_RELATIVE = (
    "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "tail_reselection_v2/roots"
)
CURRENT_PROFILE_RELATIVE = "src/ofc_regular/ai_profiles.py"
FROZEN_CONTRACT_RELATIVE = (
    "configs/hu_joint_policy_m31_t3_step6d_"
    "rearm2_diagnostic_canary_v1.json"
)

DEFAULT_REARM2_PACKAGE_DIR = Path(
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-rearm2-20260718-001/package"
)
DEFAULT_REARM2_GLOBAL_SPOT_CLAIM = Path(
    "outputs/hu_joint_policy/m31_t3_step6d/"
    "GLOBAL_PERFORMANCE_LOCK_REARM2_SPOT_CLAIM.json"
)

EXPECTED_DEVELOPMENT_PLAN_SHA256 = (
    "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758"
)
EXPECTED_DEVELOPMENT_RUN_CONTRACT_DIGEST = (
    "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
)
EXPECTED_DEVELOPMENT_ALL100_ROOT_SHA256 = (
    "0aacb1b7f9b3c45a7ca51d9e58218e55ef3cc29159e2d431757806be076e6796"
)
EXPECTED_DEVELOPMENT_TOPOLOGY_SHA256 = (
    "779e6a6d2ce6d84ccb31df6e02c48efdb7833ef06dab0625aeeab551d3d81633"
)

EXPECTED_REARM2_MANIFEST_SHA256 = (
    "92c7977f8a701c83afb53f06ca4ee96e46ba67c891c5769babdc7c70614c05cd"
)
EXPECTED_REARM2_SOURCE_SHA256 = (
    "8aa762cd31c61b33ec8ee984786f723a9b2372b4aa673cfad20e32ed6187efea"
)
EXPECTED_REARM2_STARTUP_SHA256 = (
    "9a2fab31412bac1a9a422a2194a93035b262598c9ac1cb17c27b9963481f728d"
)
EXPECTED_REARM2_RUN_CONTRACT_DIGEST = (
    "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5"
)
EXPECTED_REARM2_RUN_NAME = "regular-hu-m31-c02-lock-r2-20260718-001"

EXPECTED_CANDIDATE_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
EXPECTED_REFERENCE_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
EXPECTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
EXPECTED_CURRENT_PROFILE_FILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

STAGE1_ID = "stage1_lifecycle_one_candidate_vm"
STAGE2_ID = "stage2_candidate_reference_pair"
STAGE1_RUN_NAME = "regular-hu-m31-r2diag-s1-20260718-001"
STAGE2_RUN_NAME = "regular-hu-m31-r2diag-s2-20260718-001"
STAGE1_JOB_IDS = ("candidate-shard-00",)
STAGE2_JOB_IDS = ("candidate-shard-01", "reference-shard-01")
STAGE1_HAND_INDICES = (0, 10, 13, 43, 49, 62, 66, 81, 82, 99)
STAGE2_HAND_INDICES = (5, 6, 35, 39, 47, 53, 76, 83, 87, 89)

STAGE1_CLAIM_NAME = "rearm2_diagnostic_stage1_claim_v1.json"
STAGE1_RESULT_NAME = "rearm2_diagnostic_stage1_result_v1.json"
STAGE2_CLAIM_NAME = "rearm2_diagnostic_stage2_claim_v1.json"
STAGE2_RESULT_NAME = "rearm2_diagnostic_stage2_result_v1.json"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT_PLAN = _REPO_ROOT / DEVELOPMENT_PLAN_RELATIVE
DEFAULT_DEVELOPMENT_ROOTS = _REPO_ROOT / DEVELOPMENT_ROOTS_RELATIVE
DEFAULT_CURRENT_PROFILE = _REPO_ROOT / CURRENT_PROFILE_RELATIVE
DEFAULT_FROZEN_CONTRACT = _REPO_ROOT / FROZEN_CONTRACT_RELATIVE

_SHA256 = frozenset("0123456789abcdef")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "status",
        "scope",
        "development_root_source",
        "rearm2_production_anchor",
        "reused_native_artifacts",
        "stages",
        "identity_guards",
        "authorization",
        "implementation_boundary",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "all20_launch_authorized",
        "cloud_launch_authorized",
        "current_profile_changed",
        "gcloud_invocation_authorized",
        "named_profile_added",
        "performance_lock_authorized",
        "performance_lock_evidence",
        "production_fanout_authorized",
        "promotion_authorized",
        "promotion_evidence",
        "quality_authorized",
        "quality_evidence",
        "runtime_policy_activated",
        "training_authorized",
        "training_eligible",
    }
)


@dataclass(frozen=True)
class DiagnosticCanaryInputs:
    repository_root: Path = _REPO_ROOT
    development_plan_path: Path = DEFAULT_DEVELOPMENT_PLAN
    development_root_directory: Path = DEFAULT_DEVELOPMENT_ROOTS
    rearm2_package_directory: Path = DEFAULT_REARM2_PACKAGE_DIR
    current_profile_path: Path = DEFAULT_CURRENT_PROFILE
    rearm2_global_spot_claim_path: Path = (
        _REPO_ROOT / DEFAULT_REARM2_GLOBAL_SPOT_CLAIM
    )


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256)
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise FileNotFoundError(f"{label} is missing or unsafe: {source}")
    resolved = source.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _relative_to_repository(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"diagnostic input escaped repository: {resolved}") from exc
    return relative.as_posix()


def _root_records(
    *,
    root_directory: Path,
    repository_root: Path,
    hand_indices: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for hand_index in hand_indices:
        path = root_directory.resolve() / f"hand_{hand_index:03d}.json"
        root = _read_json_object(path, f"development root {hand_index}")
        if (
            root.get("schema")
            != "hu_m31_t3_step6d_candidate02_performance_root_v1"
            or root.get("hand_index") != hand_index
            or root.get("schedule") != "candidate02_performance_development"
            or root.get("training_eligible") is not False
            or root.get("current_profile_resolved") is not False
            or root.get("opponent_private_discards_used") is not False
        ):
            raise ValueError(f"development root contract changed: {hand_index}")
        records.append(
            {
                "hand_index": hand_index,
                "path": _relative_to_repository(path, repository_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _job_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or any(not isinstance(row, Mapping) for row in jobs):
        raise ValueError("development plan jobs changed")
    expected_order = list(full100_transport.authorized_job_ids())
    if [row.get("job_id") for row in jobs] != expected_order:
        raise ValueError("development plan job order changed")
    return {str(row["job_id"]): row for row in jobs}


def _validate_stage_source(
    *,
    jobs: Mapping[str, Mapping[str, Any]],
    selected: Sequence[str],
    hand_indices: Sequence[int],
) -> None:
    bounded = full100_transport._bounded_jobs(selected)
    if tuple(bounded) != tuple(selected):
        raise ValueError("diagnostic stage order changed")
    for identifier in selected:
        row = jobs[identifier]
        expected_role = identifier.split("-", 1)[0]
        if (
            row.get("source_role") != expected_role
            or row.get("work_hand_indices") != list(hand_indices)
        ):
            raise ValueError(f"diagnostic source job changed: {identifier}")


def _stage(
    *,
    stage_id: str,
    run_name: str,
    claim_name: str,
    result_name: str,
    selected_job_ids: Sequence[str],
    hand_indices: Sequence[int],
    root_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roles = [identifier.split("-", 1)[0] for identifier in selected_job_ids]
    return {
        "stage_id": stage_id,
        "status": "planned_local_contract_only",
        "run_name": run_name,
        "claim_name": claim_name,
        "result_name": result_name,
        "selected_job_ids": list(selected_job_ids),
        "source_roles": roles,
        "vm_count": len(selected_job_ids),
        "hand_indices": list(hand_indices),
        "root_records": [dict(record) for record in root_records],
        "root_record_digest": canonical_sha256(list(root_records)),
        "diagnostic_only": True,
        "cloud_capable": False,
        "result_admissible_as_performance_lock_evidence": False,
        "result_admissible_as_quality_evidence": False,
        "result_admissible_as_training_data": False,
        "result_admissible_as_promotion_evidence": False,
    }


def _validate_rearm2_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, "rearm2 production manifest anchor")
    if sha256_file(path) != EXPECTED_REARM2_MANIFEST_SHA256:
        raise ValueError("rearm2 production manifest hash changed")
    candidate = manifest.get("accepted_candidate")
    reference = manifest.get("accepted_reference")
    feature = manifest.get("feature_encoder")
    if (
        manifest.get("schema")
        != "hu_m31_t3_step6d_performance_lock_spot_package_v1"
        or manifest.get("status")
        != "immutable_performance_lock_package_ready_not_authorized"
        or manifest.get("run_name") != EXPECTED_REARM2_RUN_NAME
        or manifest.get("run_contract_digest")
        != EXPECTED_REARM2_RUN_CONTRACT_DIGEST
        or manifest.get("source_sha256") != EXPECTED_REARM2_SOURCE_SHA256
        or manifest.get("startup_sha256") != EXPECTED_REARM2_STARTUP_SHA256
        or not isinstance(candidate, Mapping)
        or candidate.get("sha256") != EXPECTED_CANDIDATE_SHA256
        or not isinstance(reference, Mapping)
        or reference.get("sha256") != EXPECTED_REFERENCE_SHA256
        or not isinstance(feature, Mapping)
        or feature.get("sha256") != EXPECTED_FEATURE_ENCODER_SHA256
        or any(
            manifest.get(field) is not False
            for field in (
                "gcloud_invoked",
                "spot_execution_authorized",
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "m31_complete",
            )
        )
    ):
        raise ValueError("rearm2 production manifest anchor changed")
    return manifest


def _require_production_cloud_absent(inputs: DiagnosticCanaryInputs) -> None:
    package = inputs.rearm2_package_directory.resolve()
    forbidden = (
        package / full100_transport.AUTHORIZATION_NAME,
        package / full100_transport.LAUNCH_CLAIM_NAME,
        package / full100_transport.LAUNCH_RESULT_NAME,
        package / full100_transport.RESUME_CLAIM_NAME,
        package / full100_transport.RESUME_RESULT_NAME,
        inputs.rearm2_global_spot_claim_path.resolve(),
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(
            "diagnostic canary contract must precede production cloud authorization: "
            + ",".join(existing)
        )


def build_diagnostic_canary_contract(
    inputs: DiagnosticCanaryInputs = DiagnosticCanaryInputs(),
) -> dict[str, Any]:
    """Build the deterministic local contract after read-only input validation."""

    repository_root = inputs.repository_root.resolve()
    plan_path = inputs.development_plan_path.resolve()
    roots = inputs.development_root_directory.resolve()
    current_profile = inputs.current_profile_path.resolve()
    rearm2_manifest_path = inputs.rearm2_package_directory.resolve() / "manifest.json"

    if _relative_to_repository(plan_path, repository_root) != DEVELOPMENT_PLAN_RELATIVE:
        raise ValueError("development plan path changed")
    if _relative_to_repository(roots, repository_root) != DEVELOPMENT_ROOTS_RELATIVE:
        raise ValueError("development root path changed")
    if _relative_to_repository(current_profile, repository_root) != CURRENT_PROFILE_RELATIVE:
        raise ValueError("current profile anchor path changed")
    if sha256_file(current_profile) != EXPECTED_CURRENT_PROFILE_FILE_SHA256:
        raise ValueError("current profile file hash changed")
    if inputs.development_root_directory.is_symlink():
        raise ValueError("development root directory must not be a symlink")

    _require_production_cloud_absent(inputs)

    raw_plan = _read_json_object(plan_path, "development full100 plan")
    if sha256_file(plan_path) != EXPECTED_DEVELOPMENT_PLAN_SHA256:
        raise ValueError("development full100 plan hash changed")
    plan = full100_plan.validate_full100_plan(raw_plan)
    root_set = plan["root_set"]
    run_contract = plan["run_contract"]
    if (
        plan["run_contract_digest"] != EXPECTED_DEVELOPMENT_RUN_CONTRACT_DIGEST
        or root_set["all100_root_sha256"]
        != EXPECTED_DEVELOPMENT_ALL100_ROOT_SHA256
        or root_set["topology_sha256"] != EXPECTED_DEVELOPMENT_TOPOLOGY_SHA256
        or run_contract["candidate_library_sha256"] != EXPECTED_CANDIDATE_SHA256
        or run_contract["reference_library_sha256"] != EXPECTED_REFERENCE_SHA256
        or any(
            plan.get(field) is not False
            for field in (
                "spot_package_authorized",
                "cloud_started",
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "m31_complete",
            )
        )
    ):
        raise ValueError("development full100 source is no longer diagnostic-safe")

    jobs = _job_map(plan)
    _validate_stage_source(
        jobs=jobs,
        selected=STAGE1_JOB_IDS,
        hand_indices=STAGE1_HAND_INDICES,
    )
    _validate_stage_source(
        jobs=jobs,
        selected=STAGE2_JOB_IDS,
        hand_indices=STAGE2_HAND_INDICES,
    )
    if set(STAGE1_HAND_INDICES) & set(STAGE2_HAND_INDICES):
        raise AssertionError("diagnostic stage roots overlap")

    stage1_roots = _root_records(
        root_directory=roots,
        repository_root=repository_root,
        hand_indices=STAGE1_HAND_INDICES,
    )
    stage2_roots = _root_records(
        root_directory=roots,
        repository_root=repository_root,
        hand_indices=STAGE2_HAND_INDICES,
    )

    rearm2_manifest = _validate_rearm2_manifest(rearm2_manifest_path)
    production_job_ids = [
        row.get("job_id") for row in rearm2_manifest.get("job_manifests", [])
    ]
    if production_job_ids != list(full100_transport.authorized_job_ids()):
        raise ValueError("rearm2 production job identity changed")

    stages = [
        _stage(
            stage_id=STAGE1_ID,
            run_name=STAGE1_RUN_NAME,
            claim_name=STAGE1_CLAIM_NAME,
            result_name=STAGE1_RESULT_NAME,
            selected_job_ids=STAGE1_JOB_IDS,
            hand_indices=STAGE1_HAND_INDICES,
            root_records=stage1_roots,
        ),
        _stage(
            stage_id=STAGE2_ID,
            run_name=STAGE2_RUN_NAME,
            claim_name=STAGE2_CLAIM_NAME,
            result_name=STAGE2_RESULT_NAME,
            selected_job_ids=STAGE2_JOB_IDS,
            hand_indices=STAGE2_HAND_INDICES,
            root_records=stage2_roots,
        ),
    ]
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "scope": CONTRACT_SCOPE,
        "development_root_source": {
            "classification": (
                "accepted_performance_development_reuse_diagnostic_only"
            ),
            "plan_path": DEVELOPMENT_PLAN_RELATIVE,
            "plan_sha256": EXPECTED_DEVELOPMENT_PLAN_SHA256,
            "root_directory": DEVELOPMENT_ROOTS_RELATIVE,
            "root_schema": root_set["schema"],
            "root_count": root_set["root_count"],
            "all100_root_sha256": EXPECTED_DEVELOPMENT_ALL100_ROOT_SHA256,
            "topology_sha256": EXPECTED_DEVELOPMENT_TOPOLOGY_SHA256,
            "run_contract_digest": EXPECTED_DEVELOPMENT_RUN_CONTRACT_DIGEST,
            "selected_diagnostic_hand_indices": sorted(
                set(STAGE1_HAND_INDICES) | set(STAGE2_HAND_INDICES)
            ),
            "selected_diagnostic_root_digest": canonical_sha256(
                stage1_roots + stage2_roots
            ),
            "rearm2_locked_root_directory_used": False,
            "new_seed_namespace_opened": False,
            "root_content_admissible_as_performance_lock_evidence": False,
            "root_content_admissible_as_quality_evidence": False,
            "root_content_admissible_as_training_data": False,
            "root_content_admissible_as_promotion_evidence": False,
        },
        "rearm2_production_anchor": {
            "manifest_sha256": EXPECTED_REARM2_MANIFEST_SHA256,
            "run_name": EXPECTED_REARM2_RUN_NAME,
            "run_contract_digest": EXPECTED_REARM2_RUN_CONTRACT_DIGEST,
            "source_sha256": EXPECTED_REARM2_SOURCE_SHA256,
            "startup_sha256": EXPECTED_REARM2_STARTUP_SHA256,
            "read_only_anchor": True,
            "source_archive_opened": False,
            "locked_root_member_opened": False,
            "package_mutation_authorized": False,
            "claim_mutation_authorized": False,
            "authorization_mutation_authorized": False,
        },
        "reused_native_artifacts": {
            "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
            "reference_sha256": EXPECTED_REFERENCE_SHA256,
            "feature_encoder_sha256": EXPECTED_FEATURE_ENCODER_SHA256,
            "candidate_reference_use_separate_processes_and_vms": True,
        },
        "stages": stages,
        "identity_guards": {
            "stage_run_names_distinct": True,
            "stage_claim_names_distinct": True,
            "stage_result_names_distinct": True,
            "stage_run_names_distinct_from_rearm2_production": True,
            "stage_claim_names_distinct_from_production_launch_claim": True,
            "stage_result_names_distinct_from_production_launch_result": True,
            "stage1_stage2_root_sets_disjoint": True,
            "stage2_candidate_reference_root_sets_identical": True,
            "max_stage_vm_count": 2,
            "all20_job_set_selected": False,
        },
        "authorization": {
            key: False for key in sorted(_AUTHORIZATION_KEYS)
        },
        "implementation_boundary": {
            "cloud_capable": False,
            "existing_rearm2_launch_reused": False,
            "existing_full100_startup_compatible_with_subset_claim": False,
            "reason": (
                "accepted startup and launch-chain validators require the exact "
                "all20 initial job set"
            ),
            "required_before_any_cloud_canary": [
                "versioned diagnostic-only package and authorization schema",
                "versioned startup verifier for the exact frozen stage job set",
                "stage-specific heartbeat upload DONE and receive validators",
                "write-once stage receipt whose validation gates the next stage",
            ],
            "production_launch_semantics_changed": False,
        },
    }
    return validate_diagnostic_canary_contract(contract)


def validate_diagnostic_canary_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic canary contract must be an object")
    payload = dict(value)
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "diagnostic canary contract")
    if (
        payload.get("schema") != CONTRACT_SCHEMA
        or payload.get("status") != CONTRACT_STATUS
        or payload.get("scope") != CONTRACT_SCOPE
    ):
        raise ValueError("diagnostic canary contract identity changed")

    development = payload.get("development_root_source")
    production = payload.get("rearm2_production_anchor")
    artifacts = payload.get("reused_native_artifacts")
    stages = payload.get("stages")
    guards = payload.get("identity_guards")
    authorization = payload.get("authorization")
    boundary = payload.get("implementation_boundary")
    if not all(
        isinstance(item, Mapping)
        for item in (development, production, artifacts, guards, authorization, boundary)
    ) or not isinstance(stages, list):
        raise ValueError("diagnostic canary contract sections changed")

    if (
        development.get("classification")
        != "accepted_performance_development_reuse_diagnostic_only"
        or development.get("plan_path") != DEVELOPMENT_PLAN_RELATIVE
        or development.get("plan_sha256") != EXPECTED_DEVELOPMENT_PLAN_SHA256
        or development.get("root_directory") != DEVELOPMENT_ROOTS_RELATIVE
        or development.get("root_schema")
        != "hu_m31_t3_step6d_candidate02_performance_root_v1"
        or development.get("root_count") != 100
        or development.get("all100_root_sha256")
        != EXPECTED_DEVELOPMENT_ALL100_ROOT_SHA256
        or development.get("topology_sha256")
        != EXPECTED_DEVELOPMENT_TOPOLOGY_SHA256
        or development.get("run_contract_digest")
        != EXPECTED_DEVELOPMENT_RUN_CONTRACT_DIGEST
        or development.get("selected_diagnostic_hand_indices")
        != sorted(set(STAGE1_HAND_INDICES) | set(STAGE2_HAND_INDICES))
        or not _is_sha256(development.get("selected_diagnostic_root_digest"))
        or any(
            development.get(field) is not False
            for field in (
                "rearm2_locked_root_directory_used",
                "new_seed_namespace_opened",
                "root_content_admissible_as_performance_lock_evidence",
                "root_content_admissible_as_quality_evidence",
                "root_content_admissible_as_training_data",
                "root_content_admissible_as_promotion_evidence",
            )
        )
    ):
        raise ValueError("diagnostic development-root boundary changed")

    if (
        production.get("manifest_sha256") != EXPECTED_REARM2_MANIFEST_SHA256
        or production.get("run_name") != EXPECTED_REARM2_RUN_NAME
        or production.get("run_contract_digest")
        != EXPECTED_REARM2_RUN_CONTRACT_DIGEST
        or production.get("source_sha256") != EXPECTED_REARM2_SOURCE_SHA256
        or production.get("startup_sha256") != EXPECTED_REARM2_STARTUP_SHA256
        or production.get("read_only_anchor") is not True
        or any(
            production.get(field) is not False
            for field in (
                "source_archive_opened",
                "locked_root_member_opened",
                "package_mutation_authorized",
                "claim_mutation_authorized",
                "authorization_mutation_authorized",
            )
        )
    ):
        raise ValueError("rearm2 production read-only boundary changed")

    if artifacts != {
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "feature_encoder_sha256": EXPECTED_FEATURE_ENCODER_SHA256,
        "candidate_reference_use_separate_processes_and_vms": True,
    }:
        raise ValueError("diagnostic native artifact binding changed")

    expected_stage_specs = (
        (
            STAGE1_ID,
            STAGE1_RUN_NAME,
            STAGE1_CLAIM_NAME,
            STAGE1_RESULT_NAME,
            STAGE1_JOB_IDS,
            STAGE1_HAND_INDICES,
            ("candidate",),
            1,
        ),
        (
            STAGE2_ID,
            STAGE2_RUN_NAME,
            STAGE2_CLAIM_NAME,
            STAGE2_RESULT_NAME,
            STAGE2_JOB_IDS,
            STAGE2_HAND_INDICES,
            ("candidate", "reference"),
            2,
        ),
    )
    if len(stages) != len(expected_stage_specs):
        raise ValueError("diagnostic stage count changed")
    stage_root_sets: list[set[int]] = []
    for raw, spec in zip(stages, expected_stage_specs, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("diagnostic stage must be an object")
        (
            stage_id,
            run_name,
            claim_name,
            result_name,
            job_ids,
            hand_indices,
            roles,
            vm_count,
        ) = spec
        records = raw.get("root_records")
        if (
            raw.get("stage_id") != stage_id
            or raw.get("status") != "planned_local_contract_only"
            or raw.get("run_name") != run_name
            or raw.get("claim_name") != claim_name
            or raw.get("result_name") != result_name
            or raw.get("selected_job_ids") != list(job_ids)
            or raw.get("source_roles") != list(roles)
            or raw.get("vm_count") != vm_count
            or raw.get("hand_indices") != list(hand_indices)
            or not isinstance(records, list)
            or [record.get("hand_index") for record in records] != list(hand_indices)
            or any(
                not isinstance(record, Mapping)
                or record.get("path")
                != f"{DEVELOPMENT_ROOTS_RELATIVE}/"
                f"hand_{record.get('hand_index'):03d}.json"
                or not _is_sha256(record.get("sha256"))
                or isinstance(record.get("size_bytes"), bool)
                or not isinstance(record.get("size_bytes"), int)
                or record.get("size_bytes") <= 0
                for record in records
            )
            or raw.get("root_record_digest") != canonical_sha256(records)
            or raw.get("diagnostic_only") is not True
            or raw.get("cloud_capable") is not False
            or any(
                raw.get(field) is not False
                for field in (
                    "result_admissible_as_performance_lock_evidence",
                    "result_admissible_as_quality_evidence",
                    "result_admissible_as_training_data",
                    "result_admissible_as_promotion_evidence",
                )
            )
        ):
            raise ValueError(f"diagnostic stage changed: {stage_id}")
        full100_transport._bounded_jobs(job_ids)
        stage_root_sets.append(set(hand_indices))

    if stage_root_sets[0] & stage_root_sets[1]:
        raise ValueError("diagnostic stage roots overlap")
    combined_records = list(stages[0]["root_records"]) + list(stages[1]["root_records"])
    if development.get("selected_diagnostic_root_digest") != canonical_sha256(
        combined_records
    ):
        raise ValueError("diagnostic combined root digest changed")

    expected_guards = {
        "stage_run_names_distinct": True,
        "stage_claim_names_distinct": True,
        "stage_result_names_distinct": True,
        "stage_run_names_distinct_from_rearm2_production": True,
        "stage_claim_names_distinct_from_production_launch_claim": True,
        "stage_result_names_distinct_from_production_launch_result": True,
        "stage1_stage2_root_sets_disjoint": True,
        "stage2_candidate_reference_root_sets_identical": True,
        "max_stage_vm_count": 2,
        "all20_job_set_selected": False,
    }
    if dict(guards) != expected_guards:
        raise ValueError("diagnostic identity guards changed")
    run_names = [stage["run_name"] for stage in stages]
    claim_names = [stage["claim_name"] for stage in stages]
    result_names = [stage["result_name"] for stage in stages]
    if (
        len(set(run_names)) != 2
        or len(set(claim_names)) != 2
        or len(set(result_names)) != 2
        or EXPECTED_REARM2_RUN_NAME in run_names
        or full100_transport.LAUNCH_CLAIM_NAME in claim_names
        or full100_transport.LAUNCH_RESULT_NAME in result_names
        or max(stage["vm_count"] for stage in stages) != 2
        or any(
            len(stage["selected_job_ids"]) == full100_transport.MAX_LOGICAL_JOBS
            for stage in stages
        )
    ):
        raise ValueError("diagnostic identities are not isolated")

    _require_exact_keys(
        authorization, _AUTHORIZATION_KEYS, "diagnostic authorization boundary"
    )
    if any(value is not False for value in authorization.values()):
        raise ValueError("diagnostic contract authorized a forbidden action")
    if boundary != {
        "cloud_capable": False,
        "existing_rearm2_launch_reused": False,
        "existing_full100_startup_compatible_with_subset_claim": False,
        "reason": (
            "accepted startup and launch-chain validators require the exact "
            "all20 initial job set"
        ),
        "required_before_any_cloud_canary": [
            "versioned diagnostic-only package and authorization schema",
            "versioned startup verifier for the exact frozen stage job set",
            "stage-specific heartbeat upload DONE and receive validators",
            "write-once stage receipt whose validation gates the next stage",
        ],
        "production_launch_semantics_changed": False,
    }:
        raise ValueError("diagnostic implementation boundary changed")
    return payload


def validate_frozen_contract(
    contract_path: str | Path = DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    return validate_diagnostic_canary_contract(
        _read_json_object(Path(contract_path), "frozen diagnostic canary contract")
    )


def audit_frozen_contract(
    *,
    contract_path: str | Path = DEFAULT_FROZEN_CONTRACT,
    inputs: DiagnosticCanaryInputs = DiagnosticCanaryInputs(),
) -> dict[str, Any]:
    frozen = validate_frozen_contract(contract_path)
    rebuilt = build_diagnostic_canary_contract(inputs)
    if frozen != rebuilt:
        raise ValueError("frozen diagnostic canary contract differs from live anchors")
    return {
        "schema": "hu_m31_t3_step6d_rearm2_diagnostic_canary_audit_v1",
        "status": "pass_local_only_cloud_not_authorized",
        "contract_sha256": canonical_sha256(frozen),
        "stage1_job_count": len(frozen["stages"][0]["selected_job_ids"]),
        "stage2_job_count": len(frozen["stages"][1]["selected_job_ids"]),
        "max_stage_vm_count": frozen["identity_guards"]["max_stage_vm_count"],
        "development_roots_only": True,
        "rearm2_package_mutated": False,
        "rearm2_claim_mutated": False,
        "gcloud_invoked": False,
        "all20_launch_authorized": False,
        "training_authorized": False,
        "performance_lock_authorized": False,
        "quality_authorized": False,
        "promotion_authorized": False,
        "current_profile_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    audit = commands.add_parser("audit")
    validate = commands.add_parser("validate")
    for command in (build, audit):
        command.add_argument(
            "--repository-root", type=Path, default=_REPO_ROOT
        )
        command.add_argument(
            "--development-plan", type=Path, default=DEFAULT_DEVELOPMENT_PLAN
        )
        command.add_argument(
            "--development-roots", type=Path, default=DEFAULT_DEVELOPMENT_ROOTS
        )
        command.add_argument(
            "--rearm2-package-dir", type=Path, default=DEFAULT_REARM2_PACKAGE_DIR
        )
        command.add_argument(
            "--current-profile", type=Path, default=DEFAULT_CURRENT_PROFILE
        )
        command.add_argument(
            "--rearm2-global-spot-claim",
            type=Path,
            default=_REPO_ROOT / DEFAULT_REARM2_GLOBAL_SPOT_CLAIM,
        )
    audit.add_argument("--contract", type=Path, default=DEFAULT_FROZEN_CONTRACT)
    validate.add_argument("--contract", type=Path, default=DEFAULT_FROZEN_CONTRACT)
    return parser


def _inputs(args: argparse.Namespace) -> DiagnosticCanaryInputs:
    return DiagnosticCanaryInputs(
        repository_root=args.repository_root,
        development_plan_path=args.development_plan,
        development_root_directory=args.development_roots,
        rearm2_package_directory=args.rearm2_package_dir,
        current_profile_path=args.current_profile,
        rearm2_global_spot_claim_path=args.rearm2_global_spot_claim,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_diagnostic_canary_contract(_inputs(args))
    elif args.command == "audit":
        result = audit_frozen_contract(
            contract_path=args.contract,
            inputs=_inputs(args),
        )
    else:
        result = validate_frozen_contract(args.contract)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_SCOPE",
    "CONTRACT_STATUS",
    "DiagnosticCanaryInputs",
    "STAGE1_HAND_INDICES",
    "STAGE1_ID",
    "STAGE1_JOB_IDS",
    "STAGE2_HAND_INDICES",
    "STAGE2_ID",
    "STAGE2_JOB_IDS",
    "audit_frozen_contract",
    "build_diagnostic_canary_contract",
    "canonical_sha256",
    "main",
    "validate_diagnostic_canary_contract",
    "validate_frozen_contract",
]
