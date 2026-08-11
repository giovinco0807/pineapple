"""Freeze the single fresh Candidate02 performance-lock recovery plan.

The original one-shot lock is terminal: its attempt-0 workers all failed in
pre-content startup validation and its closeout explicitly forbids attempt-1,
reseed, replacement-package, and alternate-package use for that run.  This
module consumes that canonical closeout plus the already accepted development
qualification and creates a *new* one-shot lock contract with the runner's
fresh recovery-v2 seed namespaces.

This is a local pre-content plan only.  It cannot reuse the old roots, open
either root set, invoke cloud services, train, promote, change ``current``, or
activate runtime policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import close_hu_m31_t3_step6d_performance_lock_startup_failure as closeout
from . import hu_m31_t3_step6d_candidate02_performance_lock_plan as v1_plan
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


PLAN_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_rearm1_precontent_plan_v1"
)
PLAN_STATUS = "frozen_rearm1_precontent_plan_fresh_lock_not_opened"
PLAN_DECISION = (
    "terminal_startup_failure_closeout_authorizes_one_fresh_seed_lock_only"
)
PLAN_SCOPE = "performance_lock_rearm1_fresh_roots_only"
ASSIGNMENT_METHOD = "consecutive_ten_hand_arithmetic_rearm1_v1"

CANDIDATE_VARIANT = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
RUN_ID = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID
SCHEDULE = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
ROOT_SCHEMA = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA
DONE_SCHEMA = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA
SHARD_MANIFEST_SCHEMA = (
    runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
)
RECOVERY_RUN_CONTRACT_DIGEST = (
    "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
)
# Compatibility name used by the shared Spot lifecycle.  It intentionally
# points at the recovery contract, never the terminal v1 lock contract.
LOCK_RUN_CONTRACT_DIGEST = RECOVERY_RUN_CONTRACT_DIGEST
STARTUP_FAILURE_RECEIPT_SHA256 = (
    "b529f87a301ebb588d3e965b01bae08ea887ab13bc1f305c72058c7a90bafbd7"
)

# ``compute_precontent_plan_sha256`` is intentionally usable before this
# constant is frozen.  All build/load/write validators, however, require this
# to be a lowercase SHA-256 and compare the complete canonical plan against it.
PRECONTENT_PLAN_SHA256_PLACEHOLDER = "__SET_AFTER_CANONICAL_INCIDENT_RECEIPT__"
PRECONTENT_PLAN_SHA256 = (
    "3b8a4230531f0d81c5320b2b0d051878113ac57a38ad97fdb0b1d89e0f17a886"
)

CANDIDATE_LIBRARY_SHA256 = v1_plan.CANDIDATE_LIBRARY_SHA256
REFERENCE_LIBRARY_SHA256 = v1_plan.REFERENCE_LIBRARY_SHA256
FEATURE_ENCODER_SHA256 = v1_plan.FEATURE_ENCODER_SHA256
CURRENT_PROFILE_REGISTRY_SHA256 = v1_plan.CURRENT_PROFILE_REGISTRY_SHA256
STEP6D_CONTRACT_SHA256 = v1_plan.STEP6D_CONTRACT_SHA256
SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
SHARD_COUNT_PER_ROLE = 10
HANDS_PER_SHARD = 10
HANDS_PER_PROFILE_PER_SHARD = 2
LOGICAL_JOB_COUNT = len(SOURCE_ROLES) * SHARD_COUNT_PER_ROLE

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT_DIR = v1_plan.DEFAULT_DEVELOPMENT_DIR
DEFAULT_STEP6D_CONTRACT_PATH = v1_plan.DEFAULT_STEP6D_CONTRACT_PATH
DEFAULT_CURRENT_REGISTRY_PATH = v1_plan.DEFAULT_CURRENT_REGISTRY_PATH
DEFAULT_INCIDENT_RECEIPT_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm1/"
    "performance_lock_v1_startup_failure_closeout.json"
)
INCIDENT_RECEIPT_REPO_PATH = (
    "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm1/"
    "performance_lock_v1_startup_failure_closeout.json"
)

_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "development_qualification",
        "startup_failure_closeout",
        "old_run_disposition",
        "fresh_lock_authority",
        "candidate_variant",
        "step6d_contract",
        "run_contract",
        "run_contract_digest",
        "source_identity",
        "image",
        "allocation",
        "root_contract",
        "assignment",
        "source_roles",
        "shard_count_per_role",
        "hands_per_shard",
        "logical_job_count",
        "shards",
        "jobs",
        "open_claim_required_before_root_content",
        "root_content_opened",
        "cloud_started",
        "performance_lock_passed",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)
_STARTUP_FAILURE_CLOSEOUT_KEYS = frozenset({"path", "sha256", "receipt"})
_SHARD_KEYS = frozenset({"shard_index", "work_hand_indices", "profile_counts"})
_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "shard_index",
        "work_hand_indices",
        "shard_manifest_sha256",
    }
)
_EXPECTED_CONTENT_ACCESS = {
    "cloud_mutated": False,
    "cloud_query_performed": False,
    "hand_content_opened": False,
    "result_content_opened": False,
    "root_content_opened": False,
    "source_archive_opaque_hash_verified": True,
    "source_archive_opened": False,
    "startup_logs_opened": True,
}
_EXPECTED_DISPOSITION = {
    "alternate_package_authorized": False,
    "alternate_seed_authorized": False,
    "attempt1_authorized": False,
    "attempt1_forbidden_reason": (
        "same_claimed_package_source_startup_and_claim_bytes_deterministically_"
        "repeat_the_precontent_failure"
    ),
    "cloud_execution_authorized": False,
    "current_profile_changed": False,
    "current_run_irrecoverable": True,
    "m31_complete": False,
    "named_profile_added": False,
    "performance_lock_passed": False,
    "quality_pilot_authorized": False,
    "replacement_package_authorized": False,
    "reseed_authorized": False,
    "runtime_policy_activated": False,
    "training_eligible": False,
}
_EXPECTED_INCIDENT = {
    "attempt_index": 0,
    "cause": (
        "windows_normcase_root_claim_paths_were_resolved_to_different_casing_"
        "in_the_spot_claim_then_compared_case_sensitively_on_linux"
    ),
    "deterministic_with_claimed_bytes": True,
    "error": closeout.ERROR_LINE,
    "failed_jobs": 20,
    "phase": "pre_content_package_claim_validation",
    "startup_line": 746,
}
_EXPECTED_COUNTS = {key: 0 for key in closeout.COUNT_KEYS}


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


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
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"rearm1 performance-lock plan already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(
            f"rearm1 performance-lock plan already exists: {target}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _expected_receipt_anchors() -> dict[str, str]:
    return {
        "current_profile_sha256": CURRENT_PROFILE_REGISTRY_SHA256,
        "global_root_claim_sha256": closeout.EXPECTED_HASHES[
            closeout.GLOBAL_ROOT_CLAIM_NAME
        ],
        "global_spot_claim_sha256": closeout.EXPECTED_HASHES[
            closeout.GLOBAL_SPOT_CLAIM_NAME
        ],
        "launch_authorization_sha256": closeout.EXPECTED_HASHES[
            closeout.AUTHORIZATION_NAME
        ],
        "launch_claim_sha256": closeout.EXPECTED_HASHES[
            closeout.LAUNCH_CLAIM_NAME
        ],
        "launch_result_sha256": closeout.EXPECTED_HASHES[
            closeout.LAUNCH_RESULT_NAME
        ],
        "manifest_sha256": closeout.EXPECTED_HASHES[closeout.MANIFEST_NAME],
        "package_ready_sha256": closeout.EXPECTED_HASHES[closeout.READY_NAME],
        "plan_sha256": v1_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": v1_plan.LOCK_RUN_CONTRACT_DIGEST,
        "source_sha256": closeout.EXPECTED_HASHES[closeout.SOURCE_NAME],
        "startup_sha256": closeout.EXPECTED_HASHES[closeout.STARTUP_NAME],
    }


def _validate_incident_receipt_value(
    value: Mapping[str, Any], *, expected_sha256: str
) -> dict[str, Any]:
    receipt = dict(value)
    mismatch = receipt.get("path_case_mismatch")
    startup_logs = receipt.get("startup_logs")
    if (
        canonical_sha256(receipt) != expected_sha256
        or receipt.get("schema") != closeout.RECEIPT_SCHEMA
        or receipt.get("status") != closeout.RECEIPT_STATUS
        or receipt.get("run_name") != closeout.RUN_NAME
        or receipt.get("anchors") != _expected_receipt_anchors()
        or receipt.get("content_access") != _EXPECTED_CONTENT_ACCESS
        or receipt.get("control_plane_counts") != _EXPECTED_COUNTS
        or receipt.get("disposition") != _EXPECTED_DISPOSITION
        or receipt.get("incident") != _EXPECTED_INCIDENT
        or not isinstance(mismatch, Mapping)
        or set(mismatch) != {"global_claim_path", "lock_output_directory"}
        or not isinstance(startup_logs, Mapping)
        or startup_logs.get("count") != LOGICAL_JOB_COUNT
        or startup_logs.get("job_ids") != list(closeout.JOB_IDS)
        or not _is_sha256(startup_logs.get("aggregate_sha256"))
    ):
        raise ValueError("startup-failure closeout contract changed")
    for label in ("global_claim_path", "lock_output_directory"):
        row = mismatch[label]
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"root_claim", "spot_claim", "exact_equal", "casefold_equal"}
            or not isinstance(row.get("root_claim"), str)
            or not isinstance(row.get("spot_claim"), str)
            or row["root_claim"] == row["spot_claim"]
            or row["root_claim"].casefold() != row["spot_claim"].casefold()
            or row.get("exact_equal") is not False
            or row.get("casefold_equal") is not True
        ):
            raise ValueError("startup-failure path-case evidence changed")
    records = startup_logs.get("records")
    if not isinstance(records, list) or len(records) != LOGICAL_JOB_COUNT:
        raise ValueError("startup-failure log evidence changed")
    for job_id, record in zip(closeout.JOB_IDS, records, strict=True):
        if (
            not isinstance(record, Mapping)
            or record.get("job_id") != job_id
            or record.get("attempt_index") != 0
            or record.get("exact_error_count") != 1
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] <= 0
            or not _is_sha256(record.get("sha256"))
        ):
            raise ValueError("startup-failure log record changed")
    return receipt


def load_startup_failure_closeout(
    path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if sha256_file(target) != STARTUP_FAILURE_RECEIPT_SHA256:
        raise ValueError("startup-failure closeout file hash changed")
    receipt = _read_canonical(target, "startup-failure closeout")
    return _validate_incident_receipt_value(
        receipt, expected_sha256=STARTUP_FAILURE_RECEIPT_SHA256
    )


def _arithmetic_shards() -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for shard_index in range(SHARD_COUNT_PER_ROLE):
        work = list(
            range(
                shard_index * HANDS_PER_SHARD,
                (shard_index + 1) * HANDS_PER_SHARD,
            )
        )
        counts = {profile: 0 for profile in M31_T3_BEHAVIOR_PROFILES}
        for index in work:
            profile = runner.candidate02_performance_lock_recovery_schedule_row(
                index
            )["profile"]
            counts[profile] += 1
        if any(value != HANDS_PER_PROFILE_PER_SHARD for value in counts.values()):
            raise AssertionError("arithmetic rearm1 shard is not profile-balanced")
        shards.append(
            {
                "shard_index": shard_index,
                "work_hand_indices": work,
                "profile_counts": counts,
            }
        )
    return shards


def _old_run_disposition() -> dict[str, Any]:
    return {
        "receipt_sha256": STARTUP_FAILURE_RECEIPT_SHA256,
        "old_run_name": closeout.RUN_NAME,
        "old_run_id": runner.CANDIDATE02_PERFORMANCE_LOCK_RUN_ID,
        "old_plan_sha256": v1_plan.PRECONTENT_PLAN_SHA256,
        "old_run_contract_digest": v1_plan.LOCK_RUN_CONTRACT_DIGEST,
        "attempt0_consumed": True,
        "current_run_irrecoverable": True,
        "old_attempt1_authorized": False,
        "old_root_reuse_authorized": False,
        "old_result_reuse_authorized": False,
        "old_seed_reuse_authorized": False,
        "old_package_reuse_authorized": False,
        "old_claim_reuse_authorized": False,
        "old_cloud_execution_authorized": False,
    }


def _fresh_lock_authority() -> dict[str, Any]:
    return {
        "fresh_lock_ordinal": "rearm1",
        "authorized_fresh_lock_count": 1,
        "new_run_id": RUN_ID,
        "new_candidate_variant": CANDIDATE_VARIANT,
        "new_schedule": SCHEDULE,
        "new_seed_set_sha256": (
            runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
        ),
        "performance_lock_v1_seed_overlap_count": 0,
        "fresh_lock_plan_authorized": True,
        "fresh_root_set_required": True,
        "fresh_seed_set_required": True,
        "fresh_global_claim_required_before_root_content": True,
        "fresh_root_content_authorized_before_global_claim": False,
        "fresh_spot_execution_authorized": False,
        "old_attempt1_authorized": False,
        "old_roots_authorized": False,
        "old_seeds_authorized": False,
        "second_fresh_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }


def _build_precontent_plan_payload(
    *,
    development_summary_path: str | Path,
    development_validation_path: str | Path,
    incident_receipt_path: str | Path,
    step6d_contract_path: str | Path,
    current_registry_path: str | Path,
) -> dict[str, Any]:
    # Reuse the accepted v1 producer as the development/current/contract
    # qualification boundary, then replace only the terminal lock identity.
    v1 = v1_plan.build_precontent_plan(
        development_summary_path=development_summary_path,
        development_validation_path=development_validation_path,
        step6d_contract_path=step6d_contract_path,
        current_registry_path=current_registry_path,
    )
    receipt = load_startup_failure_closeout(incident_receipt_path)
    run_contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=CANDIDATE_VARIANT,
    )
    if canonical_sha256(run_contract) != RECOVERY_RUN_CONTRACT_DIGEST:
        raise ValueError("performance-lock rearm1 runner contract digest changed")
    shards = _arithmetic_shards()
    jobs: list[dict[str, Any]] = []
    for role in SOURCE_ROLES:
        for shard in shards:
            manifest = runner.build_shard_manifest(
                run_contract=run_contract,
                source_role=role,
                work_hand_indices=shard["work_hand_indices"],
            )
            jobs.append(
                {
                    "job_id": f"{role}-shard-{shard['shard_index']:02d}",
                    "source_role": role,
                    "shard_index": shard["shard_index"],
                    "work_hand_indices": list(shard["work_hand_indices"]),
                    "shard_manifest_sha256": canonical_sha256(manifest),
                }
            )
    return {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "decision": PLAN_DECISION,
        "scope": PLAN_SCOPE,
        "development_qualification": dict(v1["development_qualification"]),
        "startup_failure_closeout": {
            "path": INCIDENT_RECEIPT_REPO_PATH,
            "sha256": STARTUP_FAILURE_RECEIPT_SHA256,
            "receipt": receipt,
        },
        "old_run_disposition": _old_run_disposition(),
        "fresh_lock_authority": _fresh_lock_authority(),
        "candidate_variant": CANDIDATE_VARIANT,
        "step6d_contract": {
            "path": "configs/hu_joint_policy_m31_t3_step6d_contract.json",
            "sha256": STEP6D_CONTRACT_SHA256,
            "schedule": SCHEDULE,
            "locked_before_content_read": True,
            "rerun_after_content_read_allowed": False,
            "recovery_of_terminal_infrastructure_failure_only": True,
        },
        "run_contract": run_contract,
        "run_contract_digest": RECOVERY_RUN_CONTRACT_DIGEST,
        "source_identity": dict(v1["source_identity"]),
        "image": dict(v1["image"]),
        "allocation": dict(v1["allocation"]),
        "root_contract": {
            "schema": ROOT_SCHEMA,
            "schedule": SCHEDULE,
            "run_id": RUN_ID,
            "recovery_of_run_id": runner.CANDIDATE02_PERFORMANCE_LOCK_RUN_ID,
            "seed_contract": (
                runner.candidate02_performance_lock_recovery_seed_contract()
            ),
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_indices": list(range(200)),
            "paired_hands": 100,
            "roots": 200,
            "fresh_root_set": True,
            "performance_lock_v1_seed_overlap_count": 0,
            "root_content_addressed": False,
            "root_content_hashes_known": False,
        },
        "assignment": {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "hand_index_and_fixed_profile_rotation_only",
            "timing_used": False,
            "topology_used": False,
            "old_root_content_used": False,
            "new_root_content_used": False,
            "teacher_values_used": False,
            "q_values_used": False,
            "ev_used": False,
            "runtime_results_used": False,
        },
        "source_roles": list(SOURCE_ROLES),
        "shard_count_per_role": SHARD_COUNT_PER_ROLE,
        "hands_per_shard": HANDS_PER_SHARD,
        "logical_job_count": LOGICAL_JOB_COUNT,
        "shards": shards,
        "jobs": jobs,
        "open_claim_required_before_root_content": True,
        "root_content_opened": False,
        "cloud_started": False,
        "performance_lock_passed": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def compute_precontent_plan_sha256(
    *,
    development_summary_path: str | Path = DEFAULT_DEVELOPMENT_DIR / "summary.json",
    development_validation_path: str | Path = DEFAULT_DEVELOPMENT_DIR
    / "validation.json",
    incident_receipt_path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
    step6d_contract_path: str | Path = DEFAULT_STEP6D_CONTRACT_PATH,
    current_registry_path: str | Path = DEFAULT_CURRENT_REGISTRY_PATH,
) -> str:
    """Compute the freeze candidate without weakening build/load validation."""

    return canonical_sha256(
        _build_precontent_plan_payload(
            development_summary_path=development_summary_path,
            development_validation_path=development_validation_path,
            incident_receipt_path=incident_receipt_path,
            step6d_contract_path=step6d_contract_path,
            current_registry_path=current_registry_path,
        )
    )


def build_precontent_plan(
    *,
    development_summary_path: str | Path = DEFAULT_DEVELOPMENT_DIR / "summary.json",
    development_validation_path: str | Path = DEFAULT_DEVELOPMENT_DIR
    / "validation.json",
    incident_receipt_path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
    step6d_contract_path: str | Path = DEFAULT_STEP6D_CONTRACT_PATH,
    current_registry_path: str | Path = DEFAULT_CURRENT_REGISTRY_PATH,
) -> dict[str, Any]:
    value = _build_precontent_plan_payload(
        development_summary_path=development_summary_path,
        development_validation_path=development_validation_path,
        incident_receipt_path=incident_receipt_path,
        step6d_contract_path=step6d_contract_path,
        current_registry_path=current_registry_path,
    )
    return validate_precontent_plan(value)


def validate_precontent_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if not _is_sha256(PRECONTENT_PLAN_SHA256):
        raise RuntimeError(
            "PRECONTENT_PLAN_SHA256 is not frozen; use "
            "compute_precontent_plan_sha256 first"
        )
    if set(payload) != _PLAN_KEYS:
        raise ValueError("rearm1 precontent plan fields changed")
    closeout_record = payload.get("startup_failure_closeout")
    shards = payload.get("shards")
    jobs = payload.get("jobs")
    if (
        not isinstance(closeout_record, Mapping)
        or set(closeout_record) != _STARTUP_FAILURE_CLOSEOUT_KEYS
        or not isinstance(closeout_record.get("receipt"), Mapping)
        or not isinstance(shards, list)
        or not isinstance(jobs, list)
    ):
        raise ValueError("rearm1 precontent nested fields changed")
    receipt = _validate_incident_receipt_value(
        closeout_record["receipt"],
        expected_sha256=STARTUP_FAILURE_RECEIPT_SHA256,
    )
    contract = runner.validate_run_contract(payload["run_contract"])
    expected_shards = _arithmetic_shards()
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != PLAN_STATUS
        or payload.get("decision") != PLAN_DECISION
        or payload.get("scope") != PLAN_SCOPE
        or payload.get("development_qualification")
        != {
            "summary_sha256": v1_plan.OFFICIAL_DEVELOPMENT_SUMMARY_SHA256,
            "validation_sha256": v1_plan.OFFICIAL_DEVELOPMENT_VALIDATION_SHA256,
            "scientific_merge_sha256": (
                v1_plan.OFFICIAL_DEVELOPMENT_SCIENTIFIC_MERGE_SHA256
            ),
            "receive_receipt_sha256": v1_plan.OFFICIAL_DEVELOPMENT_RECEIPT_SHA256,
            "run_contract_digest": (
                v1_plan.OFFICIAL_DEVELOPMENT_RUN_CONTRACT_DIGEST
            ),
            "paired_hand_count": 100,
            "root_count": 200,
            "all_gates_passed": True,
            "performance_candidate_frozen": True,
            "performance_lock_authorized": True,
            "quality_pilot_authorized": False,
        }
        or closeout_record.get("path") != INCIDENT_RECEIPT_REPO_PATH
        or closeout_record.get("sha256") != STARTUP_FAILURE_RECEIPT_SHA256
        or canonical_sha256(receipt) != STARTUP_FAILURE_RECEIPT_SHA256
        or payload.get("old_run_disposition") != _old_run_disposition()
        or payload.get("fresh_lock_authority") != _fresh_lock_authority()
        or payload.get("candidate_variant") != CANDIDATE_VARIANT
        or payload.get("step6d_contract")
        != {
            "path": "configs/hu_joint_policy_m31_t3_step6d_contract.json",
            "sha256": STEP6D_CONTRACT_SHA256,
            "schedule": SCHEDULE,
            "locked_before_content_read": True,
            "rerun_after_content_read_allowed": False,
            "recovery_of_terminal_infrastructure_failure_only": True,
        }
        or runner.contract_variant(contract) != CANDIDATE_VARIANT
        or canonical_sha256(contract) != RECOVERY_RUN_CONTRACT_DIGEST
        or payload.get("run_contract_digest") != RECOVERY_RUN_CONTRACT_DIGEST
        or payload.get("source_identity")
        != {
            "candidate_library_sha256": CANDIDATE_LIBRARY_SHA256,
            "reference_library_sha256": REFERENCE_LIBRARY_SHA256,
            "feature_encoder_sha256": FEATURE_ENCODER_SHA256,
            "current_profile_registry_sha256": CURRENT_PROFILE_REGISTRY_SHA256,
        }
        or payload.get("image")
        != {
            "project": "debian-cloud",
            "name": "debian-12-bookworm-v20260609",
            "id": "1449487925682397051",
            "self_link": (
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260609"
            ),
        }
        or payload.get("allocation")
        != {
            "machine_type": "c4-standard-16",
            "process_count": 1,
            "rayon_threads_per_process": 16,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        }
        or payload.get("root_contract")
        != {
            "schema": ROOT_SCHEMA,
            "schedule": SCHEDULE,
            "run_id": RUN_ID,
            "recovery_of_run_id": runner.CANDIDATE02_PERFORMANCE_LOCK_RUN_ID,
            "seed_contract": (
                runner.candidate02_performance_lock_recovery_seed_contract()
            ),
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_indices": list(range(200)),
            "paired_hands": 100,
            "roots": 200,
            "fresh_root_set": True,
            "performance_lock_v1_seed_overlap_count": 0,
            "root_content_addressed": False,
            "root_content_hashes_known": False,
        }
        or payload.get("assignment")
        != {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "hand_index_and_fixed_profile_rotation_only",
            "timing_used": False,
            "topology_used": False,
            "old_root_content_used": False,
            "new_root_content_used": False,
            "teacher_values_used": False,
            "q_values_used": False,
            "ev_used": False,
            "runtime_results_used": False,
        }
        or payload.get("source_roles") != list(SOURCE_ROLES)
        or payload.get("shard_count_per_role") != SHARD_COUNT_PER_ROLE
        or payload.get("hands_per_shard") != HANDS_PER_SHARD
        or payload.get("logical_job_count") != LOGICAL_JOB_COUNT
        or shards != expected_shards
        or len(jobs) != LOGICAL_JOB_COUNT
        or payload.get("open_claim_required_before_root_content") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "root_content_opened",
                "cloud_started",
                "performance_lock_passed",
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
        raise ValueError("rearm1 precontent plan contract changed")
    expected_ids = [
        f"{role}-shard-{index:02d}"
        for role in SOURCE_ROLES
        for index in range(SHARD_COUNT_PER_ROLE)
    ]
    for raw, expected_id in zip(jobs, expected_ids, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _JOB_KEYS:
            raise ValueError("rearm1 job fields changed")
        role = raw.get("source_role")
        shard_index = raw.get("shard_index")
        if (
            raw.get("job_id") != expected_id
            or role not in SOURCE_ROLES
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < SHARD_COUNT_PER_ROLE
            or raw.get("work_hand_indices")
            != expected_shards[shard_index]["work_hand_indices"]
        ):
            raise ValueError("rearm1 job mapping changed")
        manifest = runner.build_shard_manifest(
            run_contract=contract,
            source_role=str(role),
            work_hand_indices=raw["work_hand_indices"],
        )
        if (
            manifest.get("schema") != SHARD_MANIFEST_SCHEMA
            or raw.get("shard_manifest_sha256") != canonical_sha256(manifest)
        ):
            raise ValueError("rearm1 shard manifest digest changed")
    if canonical_sha256(payload) != PRECONTENT_PLAN_SHA256:
        raise ValueError("rearm1 frozen precontent plan digest changed")
    return payload


def load_and_validate_precontent_plan(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if sha256_file(target) != PRECONTENT_PLAN_SHA256:
        raise ValueError("rearm1 precontent plan file hash changed")
    return validate_precontent_plan(_read_canonical(target, "rearm1 precontent plan"))


def write_precontent_plan(
    *,
    output_path: str | Path,
    development_summary_path: str | Path = DEFAULT_DEVELOPMENT_DIR / "summary.json",
    development_validation_path: str | Path = DEFAULT_DEVELOPMENT_DIR
    / "validation.json",
    incident_receipt_path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
) -> dict[str, Any]:
    plan = build_precontent_plan(
        development_summary_path=development_summary_path,
        development_validation_path=development_validation_path,
        incident_receipt_path=incident_receipt_path,
    )
    _write_once(output_path, plan)
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=DEFAULT_DEVELOPMENT_DIR / "summary.json",
    )
    parser.add_argument(
        "--development-validation",
        type=Path,
        default=DEFAULT_DEVELOPMENT_DIR / "validation.json",
    )
    parser.add_argument(
        "--incident-receipt",
        type=Path,
        default=DEFAULT_INCIDENT_RECEIPT_PATH,
    )
    parser.add_argument(
        "--print-freeze-sha256",
        action="store_true",
        help="print the candidate digest without requiring the constant to be frozen",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "development_summary_path": args.development_summary,
        "development_validation_path": args.development_validation,
        "incident_receipt_path": args.incident_receipt,
    }
    if args.print_freeze_sha256:
        print(compute_precontent_plan_sha256(**kwargs))
        return 0
    plan = build_precontent_plan(**kwargs)
    if args.output is not None:
        _write_once(args.output, plan)
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENT_METHOD",
    "CANDIDATE_LIBRARY_SHA256",
    "CANDIDATE_VARIANT",
    "CURRENT_PROFILE_REGISTRY_SHA256",
    "DEFAULT_DEVELOPMENT_DIR",
    "DEFAULT_INCIDENT_RECEIPT_PATH",
    "DONE_SCHEMA",
    "FEATURE_ENCODER_SHA256",
    "HANDS_PER_SHARD",
    "INCIDENT_RECEIPT_REPO_PATH",
    "LOGICAL_JOB_COUNT",
    "LOCK_RUN_CONTRACT_DIGEST",
    "PLAN_DECISION",
    "PLAN_SCHEMA",
    "PLAN_SCOPE",
    "PLAN_STATUS",
    "PRECONTENT_PLAN_SHA256",
    "PRECONTENT_PLAN_SHA256_PLACEHOLDER",
    "RECOVERY_RUN_CONTRACT_DIGEST",
    "REFERENCE_LIBRARY_SHA256",
    "ROOT_SCHEMA",
    "RUN_ID",
    "SCHEDULE",
    "SHARD_COUNT_PER_ROLE",
    "SHARD_MANIFEST_SCHEMA",
    "SOURCE_ROLES",
    "STARTUP_FAILURE_RECEIPT_SHA256",
    "build_precontent_plan",
    "canonical_bytes",
    "canonical_sha256",
    "compute_precontent_plan_sha256",
    "load_and_validate_precontent_plan",
    "load_startup_failure_closeout",
    "sha256_file",
    "validate_precontent_plan",
    "write_precontent_plan",
]
