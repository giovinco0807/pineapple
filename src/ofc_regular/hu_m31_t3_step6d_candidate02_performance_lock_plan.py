"""Freeze the Candidate02 one-shot performance-lock plan before root content.

The plan consumes only the already-completed performance-development Go and
the preregistered ``performance_lock`` seed schedule.  It deliberately does
not stat, hash, open, generate, or inspect any lock root.  The ten work
partitions are arithmetic (consecutive groups of ten hands), so neither
topology, timing, Q values, EV, nor any other result can affect assignment.

This module cannot open the lock, generate roots, invoke cloud services,
train, promote a profile, activate runtime policy, or resolve ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


PLAN_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_lock_precontent_plan_v1"
PLAN_STATUS = "frozen_precontent_plan_lock_not_opened"
PLAN_DECISION = "performance_development_go_freezes_preregistered_one_shot_lock_only"
PLAN_SCOPE = "performance_lock"
ASSIGNMENT_METHOD = "consecutive_ten_hand_arithmetic_v1"

OFFICIAL_DEVELOPMENT_SUMMARY_SHA256 = (
    "f587df6d037313e2111a5cbc3d474370106f6a341ef4ec130cee7516192e668f"
)
OFFICIAL_DEVELOPMENT_VALIDATION_SHA256 = (
    "da252b3cabfc361d89de2f2fe08931fe61302287b69b82bf9f812a4432baf3eb"
)
OFFICIAL_DEVELOPMENT_SCIENTIFIC_MERGE_SHA256 = (
    "29e6b3db5b372f9808b681fb70246368452eecf5c6bbbca2519b2882430369f7"
)
OFFICIAL_DEVELOPMENT_RECEIPT_SHA256 = (
    "2ba1b7434cda012d8a230109e9c37db6d9e9b41587e1138e79790a177df85cf6"
)
OFFICIAL_DEVELOPMENT_RUN_CONTRACT_DIGEST = (
    "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
)
STEP6D_CONTRACT_SHA256 = (
    "1924295b18070432cf3126159311102d9285dba37c498a3ea7666b0d5b777775"
)
CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
CURRENT_PROFILE_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
LOCK_RUN_CONTRACT_DIGEST = (
    "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
)
PRECONTENT_PLAN_SHA256 = (
    "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b"
)

SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
SHARD_COUNT_PER_ROLE = 10
HANDS_PER_SHARD = 10
HANDS_PER_PROFILE_PER_SHARD = 2
LOGICAL_JOB_COUNT = len(SOURCE_ROLES) * SHARD_COUNT_PER_ROLE

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVELOPMENT_DIR = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/full100_merge/"
    "regular-hu-m31-c02-full100-dev-20260717-002"
)
DEFAULT_STEP6D_CONTRACT_PATH = (
    _REPO_ROOT / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
)
DEFAULT_CURRENT_REGISTRY_PATH = _REPO_ROOT / "src/ofc_regular/ai_profiles.py"

_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "development_qualification",
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
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)
_QUALIFICATION_KEYS = frozenset(
    {
        "summary_sha256",
        "validation_sha256",
        "scientific_merge_sha256",
        "receive_receipt_sha256",
        "run_contract_digest",
        "paired_hand_count",
        "root_count",
        "all_gates_passed",
        "performance_candidate_frozen",
        "performance_lock_authorized",
        "quality_pilot_authorized",
    }
)
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


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"performance-lock plan already exists: {target}")
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
            f"performance-lock plan already exists: {target}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_development_qualification(
    *,
    summary_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    summary_target = Path(summary_path).resolve()
    validation_target = Path(validation_path).resolve()
    if (
        sha256_file(summary_target) != OFFICIAL_DEVELOPMENT_SUMMARY_SHA256
        or sha256_file(validation_target) != OFFICIAL_DEVELOPMENT_VALIDATION_SHA256
    ):
        raise ValueError("official performance-development evidence hash changed")
    summary = _read_canonical(summary_target, "performance-development summary")
    validation = _read_canonical(
        validation_target, "performance-development validation"
    )
    scientific = summary.get("scientific_merge")
    received = summary.get("received_directory_validation")
    if (
        summary.get("schema") != "hu_m31_t3_step6d_full100_received_merge_v1"
        or summary.get("status") != "pass"
        or summary.get("decision")
        != "full100_receive_and_performance_go_open_performance_lock_only"
        or summary.get("all_gates_passed") is not True
        or summary.get("performance_candidate_frozen") is not True
        or summary.get("performance_lock_authorized") is not True
        or summary.get("quality_pilot_authorized") is not False
        or summary.get("training_authorized") is not False
        or summary.get("current_profile_changed") is not False
        or summary.get("runtime_policy_activated") is not False
        or summary.get("scientific_merge_sha256")
        != OFFICIAL_DEVELOPMENT_SCIENTIFIC_MERGE_SHA256
        or summary.get("receive_receipt_sha256") != OFFICIAL_DEVELOPMENT_RECEIPT_SHA256
        or not isinstance(scientific, Mapping)
        or scientific.get("schema")
        != "hu_m31_t3_step6d_candidate02_full100_performance_merge_v1"
        or scientific.get("status") != "pass"
        or scientific.get("run_contract_digest")
        != OFFICIAL_DEVELOPMENT_RUN_CONTRACT_DIGEST
        or scientific.get("paired_hand_count") != 100
        or scientific.get("root_count") != 200
        or scientific.get("all_gates_passed") is not True
        or scientific.get("performance_candidate_frozen") is not True
        or scientific.get("performance_lock_authorized") is not True
        or not isinstance(received, Mapping)
        or received.get("root_pairing_validated") is not True
        or received.get("source_isolation_validated") is not True
        or validation.get("schema")
        != "hu_m31_t3_step6d_full100_received_merge_validation_v1"
        or validation.get("status") != "pass"
        or validation.get("summary_sha256") != OFFICIAL_DEVELOPMENT_SUMMARY_SHA256
        or validation.get("run_contract_digest")
        != OFFICIAL_DEVELOPMENT_RUN_CONTRACT_DIGEST
        or validation.get("paired_hand_count") != 100
        or validation.get("root_count") != 200
        or validation.get("all_gates_passed") is not True
        or validation.get("performance_candidate_frozen") is not True
        or validation.get("performance_lock_authorized") is not True
        or validation.get("quality_pilot_authorized") is not False
        or validation.get("training_authorized") is not False
        or validation.get("current_profile_changed") is not False
        or validation.get("runtime_policy_activated") is not False
    ):
        raise ValueError(
            "official performance-development evidence does not authorize lock"
        )
    return {
        "summary_sha256": OFFICIAL_DEVELOPMENT_SUMMARY_SHA256,
        "validation_sha256": OFFICIAL_DEVELOPMENT_VALIDATION_SHA256,
        "scientific_merge_sha256": OFFICIAL_DEVELOPMENT_SCIENTIFIC_MERGE_SHA256,
        "receive_receipt_sha256": OFFICIAL_DEVELOPMENT_RECEIPT_SHA256,
        "run_contract_digest": OFFICIAL_DEVELOPMENT_RUN_CONTRACT_DIGEST,
        "paired_hand_count": 100,
        "root_count": 200,
        "all_gates_passed": True,
        "performance_candidate_frozen": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
    }


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
            profile = runner.candidate02_performance_lock_schedule_row(index)["profile"]
            counts[profile] += 1
        if any(value != HANDS_PER_PROFILE_PER_SHARD for value in counts.values()):
            raise AssertionError("arithmetic lock shard is not profile-balanced")
        shards.append(
            {
                "shard_index": shard_index,
                "work_hand_indices": work,
                "profile_counts": counts,
            }
        )
    return shards


def build_precontent_plan(
    *,
    development_summary_path: str | Path = DEFAULT_DEVELOPMENT_DIR / "summary.json",
    development_validation_path: str | Path = DEFAULT_DEVELOPMENT_DIR
    / "validation.json",
    step6d_contract_path: str | Path = DEFAULT_STEP6D_CONTRACT_PATH,
    current_registry_path: str | Path = DEFAULT_CURRENT_REGISTRY_PATH,
) -> dict[str, Any]:
    qualification = load_development_qualification(
        summary_path=development_summary_path,
        validation_path=development_validation_path,
    )
    if sha256_file(step6d_contract_path) != STEP6D_CONTRACT_SHA256:
        raise ValueError("Step6d contract bytes changed")
    if sha256_file(current_registry_path) != CURRENT_PROFILE_REGISTRY_SHA256:
        raise ValueError("current profile registry changed")
    run_contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    if canonical_sha256(run_contract) != LOCK_RUN_CONTRACT_DIGEST:
        raise ValueError("performance-lock runner contract digest changed")
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
    value = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "decision": PLAN_DECISION,
        "scope": PLAN_SCOPE,
        "development_qualification": qualification,
        "candidate_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
        "step6d_contract": {
            "path": "configs/hu_joint_policy_m31_t3_step6d_contract.json",
            "sha256": STEP6D_CONTRACT_SHA256,
            "schedule": runner.CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
            "locked_before_content_read": True,
            "rerun_after_content_read_allowed": False,
        },
        "run_contract": run_contract,
        "run_contract_digest": LOCK_RUN_CONTRACT_DIGEST,
        "source_identity": {
            "candidate_library_sha256": CANDIDATE_LIBRARY_SHA256,
            "reference_library_sha256": REFERENCE_LIBRARY_SHA256,
            "feature_encoder_sha256": FEATURE_ENCODER_SHA256,
            "current_profile_registry_sha256": CURRENT_PROFILE_REGISTRY_SHA256,
        },
        "image": {
            "project": "debian-cloud",
            "name": "debian-12-bookworm-v20260609",
            "id": "1449487925682397051",
            "self_link": (
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260609"
            ),
        },
        "allocation": {
            "machine_type": "c4-standard-16",
            "process_count": 1,
            "rayon_threads_per_process": 16,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        },
        "root_contract": {
            "schema": runner.CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA,
            "schedule": runner.CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
            "seed_contract": runner.candidate02_performance_lock_seed_contract(),
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_indices": list(range(200)),
            "paired_hands": 100,
            "roots": 200,
            "root_content_addressed": False,
            "root_content_hashes_known": False,
        },
        "assignment": {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "hand_index_and_preregistered_profile_rotation_only",
            "timing_used": False,
            "topology_used": False,
            "root_content_used": False,
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
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    return validate_precontent_plan(value)


def validate_precontent_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if set(payload) != _PLAN_KEYS:
        raise ValueError("performance-lock precontent plan fields changed")
    qualification = payload.get("development_qualification")
    shards = payload.get("shards")
    jobs = payload.get("jobs")
    if (
        not isinstance(qualification, Mapping)
        or set(qualification) != _QUALIFICATION_KEYS
        or not isinstance(shards, list)
        or not isinstance(jobs, list)
    ):
        raise ValueError("performance-lock precontent nested fields changed")
    contract = runner.validate_run_contract(payload["run_contract"])
    expected_shards = _arithmetic_shards()
    expected_qualification = {
        "summary_sha256": OFFICIAL_DEVELOPMENT_SUMMARY_SHA256,
        "validation_sha256": OFFICIAL_DEVELOPMENT_VALIDATION_SHA256,
        "scientific_merge_sha256": OFFICIAL_DEVELOPMENT_SCIENTIFIC_MERGE_SHA256,
        "receive_receipt_sha256": OFFICIAL_DEVELOPMENT_RECEIPT_SHA256,
        "run_contract_digest": OFFICIAL_DEVELOPMENT_RUN_CONTRACT_DIGEST,
        "paired_hand_count": 100,
        "root_count": 200,
        "all_gates_passed": True,
        "performance_candidate_frozen": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
    }
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != PLAN_STATUS
        or payload.get("decision") != PLAN_DECISION
        or payload.get("scope") != PLAN_SCOPE
        or dict(qualification) != expected_qualification
        or payload.get("candidate_variant")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
        or payload.get("step6d_contract")
        != {
            "path": "configs/hu_joint_policy_m31_t3_step6d_contract.json",
            "sha256": STEP6D_CONTRACT_SHA256,
            "schedule": runner.CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
            "locked_before_content_read": True,
            "rerun_after_content_read_allowed": False,
        }
        or runner.contract_variant(contract)
        != runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
        or canonical_sha256(contract) != LOCK_RUN_CONTRACT_DIGEST
        or payload.get("run_contract_digest") != LOCK_RUN_CONTRACT_DIGEST
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
            "schema": runner.CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA,
            "schedule": runner.CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
            "seed_contract": runner.candidate02_performance_lock_seed_contract(),
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_indices": list(range(200)),
            "paired_hands": 100,
            "roots": 200,
            "root_content_addressed": False,
            "root_content_hashes_known": False,
        }
        or payload.get("assignment")
        != {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "hand_index_and_preregistered_profile_rotation_only",
            "timing_used": False,
            "topology_used": False,
            "root_content_used": False,
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
        raise ValueError("performance-lock precontent plan contract changed")
    expected_ids = [
        f"{role}-shard-{index:02d}"
        for role in SOURCE_ROLES
        for index in range(SHARD_COUNT_PER_ROLE)
    ]
    for raw, expected_id in zip(jobs, expected_ids, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _JOB_KEYS:
            raise ValueError("performance-lock job fields changed")
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
            raise ValueError("performance-lock job mapping changed")
        manifest = runner.build_shard_manifest(
            run_contract=contract,
            source_role=str(role),
            work_hand_indices=raw["work_hand_indices"],
        )
        if raw.get("shard_manifest_sha256") != canonical_sha256(manifest):
            raise ValueError("performance-lock shard manifest digest changed")
    if canonical_sha256(payload) != PRECONTENT_PLAN_SHA256:
        raise ValueError("performance-lock frozen precontent plan digest changed")
    return payload


def write_precontent_plan(
    *,
    output_path: str | Path,
    development_summary_path: str | Path = DEFAULT_DEVELOPMENT_DIR / "summary.json",
    development_validation_path: str | Path = DEFAULT_DEVELOPMENT_DIR
    / "validation.json",
) -> dict[str, Any]:
    plan = build_precontent_plan(
        development_summary_path=development_summary_path,
        development_validation_path=development_validation_path,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_precontent_plan(
        development_summary_path=args.development_summary,
        development_validation_path=args.development_validation,
    )
    if args.output is not None:
        _write_once(args.output, plan)
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENT_METHOD",
    "CANDIDATE_LIBRARY_SHA256",
    "CURRENT_PROFILE_REGISTRY_SHA256",
    "DEFAULT_DEVELOPMENT_DIR",
    "FEATURE_ENCODER_SHA256",
    "HANDS_PER_SHARD",
    "LOCK_RUN_CONTRACT_DIGEST",
    "LOGICAL_JOB_COUNT",
    "PLAN_SCHEMA",
    "PLAN_SCOPE",
    "PRECONTENT_PLAN_SHA256",
    "REFERENCE_LIBRARY_SHA256",
    "SHARD_COUNT_PER_ROLE",
    "SOURCE_ROLES",
    "build_precontent_plan",
    "canonical_sha256",
    "load_development_qualification",
    "sha256_file",
    "validate_precontent_plan",
    "write_precontent_plan",
]
