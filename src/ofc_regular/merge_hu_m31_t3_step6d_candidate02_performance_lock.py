"""Scientifically merge the one-shot Candidate02 Step 6d performance lock.

The generic v2 merger remains the source-artifact correctness oracle.  This
wrapper binds that replay to the preregistered performance-lock plan, the
globally persisted open claim, and the independently replayed root seal.  It
then applies the lock-only hard qualification and quality-launch guards.

A passing result authorizes only a fresh quality pilot.  A failing result
finalizes this Candidate02 lock as No-Go; neither rerunning nor reseeding the
opened lock is authorized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_performance_lock_plan as lock_plan
from . import hu_m31_t3_step6d_performance_lock_open as lock_open
from . import hu_m31_t3_step6d_performance_lock_spot_v1 as lock_spot
from . import merge_hu_m31_t3_step6d_performance_v2 as base
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


MERGE_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_lock_scientific_merge_v1"
VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_scientific_validation_v1"
)
SCOPE = "candidate02_one_shot_performance_lock"
GATE_MODE = "hard_qualification_plus_quality_launch_guards"

HARD_FIRST_P95_SECONDS_MAX = 180.0
HARD_FIRST_P99_SECONDS_MAX = 240.0
HARD_FIRST_MAX_SECONDS_MAX = 240.0
HARD_SECOND_P95_SECONDS_MAX = 6.0
HARD_PEAK_RSS_BYTES_MAX = 1_073_741_824

QUALITY_FIRST_P95_SECONDS_MAX = 150.0
QUALITY_SECOND_P95_SECONDS_MAX = 5.0
QUALITY_PEAK_RSS_BYTES_MAX = 858_993_459

PASS_DECISION = "performance_lock_pass_authorize_fresh_quality_pilot_only"
NO_GO_DECISION = "performance_lock_no_go_candidate_finalized_no_rerun_or_reseed"

_OPEN_INPUT_KEYS = frozenset(
    {
        "repository_root",
        "plan_path",
        "lock_output_directory",
        "candidate_library",
        "reference_library",
        "feature_encoder",
        "startup_source",
        "development_summary_path",
        "development_validation_path",
        "development_root_directory",
        "global_claim_path",
    }
)
_RECEIVE_INPUT_KEYS = frozenset({"run_dir", "receive_dir", "project", "bucket"})
_FILE_RECORD_KEYS = frozenset({"path", "sha256", "bytes"})
_LINEAGE_INPUT_KEYS = frozenset(
    {"precontent_plan", "open_claim", "materialization", "root_seal"}
)
_PARTITION_KEYS = frozenset(
    {
        "source_roles",
        "shard_count_per_role",
        "hands_per_shard",
        "logical_job_count",
        "candidate_shard_count",
        "reference_shard_count",
        "candidate_partition_sha256",
        "reference_partition_sha256",
        "candidate_partition_exact",
        "reference_partition_exact",
        "source_roles_isolated",
        "work_coverage_exact",
    }
)
_ROOT_LINEAGE_KEYS = frozenset(
    {
        "root_artifact_count",
        "observation_count",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "root_topology_sha256",
        "observation_fingerprint_sha256",
        "candidate_reference_root_pairing_exact",
        "materialization_exact",
        "seal_exact",
        "development_overlap_count",
    }
)
_RECEIVED_EXECUTION_KEYS = frozenset(
    {
        "receive_receipt",
        "result_open_claim",
        "run_name",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "launch_claim_sha256",
        "launch_result_sha256",
        "resume_claim_sha256",
        "resume_result_sha256",
        "result_open_claim_sha256",
        "precontent_plan_sha256",
        "root_open_claim_sha256",
        "root_seal_sha256",
        "run_contract_digest",
        "logical_job_count",
        "paired_hand_count",
        "root_count",
        "received_job_sha256",
        "one_shot_result_claim_replayed",
        "launch_chain_replayed",
        "receive_directory_replayed",
        "exact_received_done_set_bound",
    }
)
_HARD_GATE_KEYS = frozenset(
    {
        "exactly_100_paired_hands",
        "exactly_200_roots",
        "exactly_100_first_and_100_second",
        "exactly_20_hands_per_profile",
        "missing_or_censored_roots_zero",
        "portable_semantic_parity_fraction_one",
        "first_p95_within_180_seconds",
        "first_p99_within_240_seconds",
        "first_max_within_240_seconds",
        "second_p95_within_6_seconds",
        "peak_rss_within_1073741824_bytes",
    }
)
_QUALITY_GUARD_KEYS = frozenset(
    {
        "first_p95_within_150_seconds",
        "second_p95_within_5_seconds",
        "peak_rss_within_858993459_bytes",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "contract_variant",
        "run_contract",
        "run_contract_digest",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "budget",
        "allocation",
        "source_identity",
        "open_inputs",
        "receive_inputs",
        "lineage_inputs",
        "received_execution",
        "source_done_inputs",
        "paired_artifacts",
        "integrity",
        "performance",
        "generic_merge_sha256",
        "plan_partitions",
        "root_lineage",
        "gate_mode",
        "hard_qualification_gates",
        "hard_qualification_passed",
        "quality_launch_guard_gates",
        "quality_launch_guards_passed",
        "all_gates_passed",
        "performance_lock_qualified",
        "candidate_finalized_no_go",
        "one_shot_lock_consumed",
        "rerun_authorized",
        "reseed_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)
_VALIDATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "summary_sha256",
        "run_contract_digest",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "source_artifacts_replayed",
        "open_claim_replayed",
        "root_seal_replayed",
        "one_shot_receive_replayed",
        "launch_chain_replayed",
        "plan_partitions_recomputed",
        "root_set_recomputed",
        "candidate_reference_root_pairing_exact",
        "hard_qualification_gates",
        "hard_qualification_passed",
        "quality_launch_guard_gates",
        "quality_launch_guards_passed",
        "all_gates_passed",
        "performance_lock_qualified",
        "candidate_finalized_no_go",
        "one_shot_lock_consumed",
        "rerun_authorized",
        "reseed_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)


@dataclass(frozen=True)
class PerformanceLockReceiveInputs:
    """Identity needed to replay the one-shot Spot receive chain."""

    run_dir: Path
    receive_dir: Path
    project: str = lock_spot.DEFAULT_PROJECT
    bucket: str = lock_spot.DEFAULT_BUCKET


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.casefold()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != runner.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _file_record(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    value = _read_canonical(target, label)
    return {
        "path": str(target.resolve()),
        "sha256": lock_open.sha256_file(target),
        "bytes": target.stat().st_size,
        "_value": value,
    }


def _validate_file_record(
    raw: Any, label: str, *, expected_path: str | Path | None = None
) -> Path:
    value = _mapping(raw, label)
    if set(value) != _FILE_RECORD_KEYS:
        raise ValueError(f"{label} fields changed")
    path = Path(str(value.get("path")))
    if expected_path is not None and path.resolve() != Path(expected_path).resolve():
        raise ValueError(f"{label} path changed")
    if lock_open.sha256_file(path) != _require_sha256(
        value.get("sha256"), f"{label} SHA-256"
    ) or path.stat().st_size != _integer(value.get("bytes"), f"{label} bytes"):
        raise ValueError(f"{label} byte/hash identity changed")
    _read_canonical(path, label)
    return path.resolve()


def _serialize_open_inputs(
    inputs: lock_open.PerformanceLockInputs,
) -> dict[str, str]:
    return {
        "repository_root": str(Path(inputs.repository_root).resolve()),
        "plan_path": str(Path(inputs.plan_path).resolve()),
        "lock_output_directory": os.path.normcase(
            os.path.abspath(os.fspath(inputs.lock_output_directory))
        ),
        "candidate_library": str(Path(inputs.candidate_library).resolve()),
        "reference_library": str(Path(inputs.reference_library).resolve()),
        "feature_encoder": str(Path(inputs.feature_encoder).resolve()),
        "startup_source": str(Path(inputs.startup_source).resolve()),
        "development_summary_path": str(
            Path(inputs.development_summary_path).resolve()
        ),
        "development_validation_path": str(
            Path(inputs.development_validation_path).resolve()
        ),
        "development_root_directory": str(
            Path(inputs.development_root_directory).resolve()
        ),
        "global_claim_path": os.path.normcase(
            os.path.abspath(os.fspath(inputs.global_claim_path))
        ),
    }


def _deserialize_open_inputs(raw: Any) -> lock_open.PerformanceLockInputs:
    value = _mapping(raw, "performance-lock open inputs")
    if set(value) != _OPEN_INPUT_KEYS or any(
        not isinstance(value[key], str) or not value[key] for key in _OPEN_INPUT_KEYS
    ):
        raise ValueError("performance-lock open input identity changed")
    return lock_open.PerformanceLockInputs(
        repository_root=Path(value["repository_root"]),
        plan_path=Path(value["plan_path"]),
        lock_output_directory=Path(value["lock_output_directory"]),
        candidate_library=Path(value["candidate_library"]),
        reference_library=Path(value["reference_library"]),
        feature_encoder=Path(value["feature_encoder"]),
        startup_source=Path(value["startup_source"]),
        development_summary_path=Path(value["development_summary_path"]),
        development_validation_path=Path(value["development_validation_path"]),
        development_root_directory=Path(value["development_root_directory"]),
        global_claim_path=Path(value["global_claim_path"]),
    )


def _serialize_receive_inputs(
    inputs: PerformanceLockReceiveInputs,
) -> dict[str, str]:
    if not inputs.project or not inputs.bucket:
        raise ValueError("performance-lock receive project/bucket is required")
    return {
        "run_dir": str(Path(inputs.run_dir).resolve()),
        "receive_dir": str(Path(inputs.receive_dir).resolve()),
        "project": inputs.project,
        "bucket": inputs.bucket,
    }


def _deserialize_receive_inputs(raw: Any) -> PerformanceLockReceiveInputs:
    value = _mapping(raw, "performance-lock receive inputs")
    if set(value) != _RECEIVE_INPUT_KEYS or any(
        not isinstance(value[key], str) or not value[key] for key in _RECEIVE_INPUT_KEYS
    ):
        raise ValueError("performance-lock receive input identity changed")
    return PerformanceLockReceiveInputs(
        run_dir=Path(value["run_dir"]),
        receive_dir=Path(value["receive_dir"]),
        project=value["project"],
        bucket=value["bucket"],
    )


def _validated_lineage(
    inputs: lock_open.PerformanceLockInputs,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    plan_record = _file_record(inputs.plan_path, "performance-lock precontent plan")
    plan = lock_plan.validate_precontent_plan(plan_record.pop("_value"))

    claim_record = _file_record(inputs.global_claim_path, "performance-lock open claim")
    raw_claim = claim_record.pop("_value")
    claim = lock_open.validate_open_claim(inputs)
    if claim != raw_claim:
        raise ValueError("performance-lock open claim replay changed")

    output = Path(inputs.lock_output_directory)
    materialization_path = output / "materialization.json"
    materialization_record = _file_record(
        materialization_path, "performance-lock materialization"
    )
    materialization = materialization_record.pop("_value")

    seal_path = output / "seal.json"
    seal_record = _file_record(seal_path, "performance-lock root seal")
    raw_seal = seal_record.pop("_value")
    seal = lock_open.validate_root_seal(inputs)
    if seal != raw_seal:
        raise ValueError("performance-lock root seal replay changed")

    lineage_inputs = {
        "precontent_plan": plan_record,
        "open_claim": claim_record,
        "materialization": materialization_record,
        "root_seal": seal_record,
    }
    if set(lineage_inputs) != _LINEAGE_INPUT_KEYS:
        raise AssertionError("performance-lock lineage input schema changed")
    return plan, claim, materialization, seal, lineage_inputs


def _validate_claim_plan_lineage(
    *,
    inputs: lock_open.PerformanceLockInputs,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> None:
    plan_path = Path(inputs.plan_path).resolve()
    plan_bytes_sha = lock_open.sha256_file(plan_path)
    precontent = _mapping(claim.get("precontent_plan"), "open-claim precontent plan")
    accepted = _mapping(claim.get("accepted_binaries"), "open-claim accepted binaries")
    source_identity = _mapping(plan.get("source_identity"), "plan source identity")
    restrictions = {
        "timing_used_for_root_selection": False,
        "q_used_for_root_selection": False,
        "ev_used_for_root_selection": False,
        "alternate_seed_allowed": False,
        "reseed_allowed": False,
        "cloud_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "current_profile_resolution_allowed": False,
        "runtime_activation_allowed": False,
        "opponent_private_discards_allowed": False,
    }
    accepted_hashes = {
        role: _mapping(accepted.get(role), f"accepted {role} binary").get("sha256")
        for role in ("candidate", "reference", "feature_encoder")
    }
    expected_hashes = {
        "candidate": source_identity["candidate_library_sha256"],
        "reference": source_identity["reference_library_sha256"],
        "feature_encoder": source_identity["feature_encoder_sha256"],
    }
    if (
        Path(str(precontent.get("path"))).resolve() != plan_path
        or precontent.get("sha256") != plan_bytes_sha
        or precontent.get("canonical_sha256") != lock_plan.canonical_sha256(plan)
        or precontent.get("schema") != lock_plan.PLAN_SCHEMA
        or claim.get("global_claim_path")
        != os.path.normcase(os.path.abspath(os.fspath(inputs.global_claim_path)))
        or claim.get("lock_output_directory")
        != os.path.normcase(os.path.abspath(os.fspath(inputs.lock_output_directory)))
        or claim.get("lock_run_contract") != plan.get("run_contract")
        or claim.get("lock_run_contract_digest") != plan.get("run_contract_digest")
        or claim.get("image") != plan.get("image")
        or claim.get("allocation") != plan.get("allocation")
        or claim.get("seed_contract")
        != _mapping(plan.get("root_contract"), "plan root contract").get(
            "seed_contract"
        )
        or accepted_hashes != expected_hashes
        or _mapping(
            claim.get("ai_profiles_current"), "open-claim current registry"
        ).get("sha256")
        != source_identity["current_profile_registry_sha256"]
        or claim.get("restrictions") != restrictions
    ):
        raise ValueError("performance-lock claim/plan lineage changed")


def _validate_materialization_and_seal(
    *,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> None:
    expected_profiles = {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
    root_hashes = _array(seal.get("root_artifact_sha256"), "root-seal artifact hashes")
    if (
        set(materialization) != lock_open._MATERIALIZATION_KEYS
        or materialization.get("schema") != lock_open.MATERIALIZATION_SCHEMA
        or materialization.get("status")
        != "all_100_lock_roots_materialized_same_identity"
        or materialization.get("global_claim_sha256")
        != lock_open.canonical_sha256(claim)
        or materialization.get("plan_sha256")
        != lock_open.sha256_file(
            Path(str(_mapping(claim["precontent_plan"], "claim plan")["path"]))
        )
        or materialization.get("run_contract_digest") != plan.get("run_contract_digest")
        or materialization.get("hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or materialization.get("root_count") != 100
        or materialization.get("same_identity_resume_only") is not True
        or materialization.get("reseeded") is not False
        or materialization.get("training_eligible") is not False
        or materialization.get("current_profile_changed") is not False
        or set(seal) != lock_open._SEAL_KEYS
        or seal.get("schema") != lock_open.SEAL_SCHEMA
        or seal.get("status")
        != "sealed_100_disjoint_hidden_safe_performance_lock_roots"
        or seal.get("global_claim_sha256") != lock_open.canonical_sha256(claim)
        or seal.get("materialization_sha256")
        != lock_open.canonical_sha256(materialization)
        or seal.get("plan_sha256") != materialization.get("plan_sha256")
        or seal.get("run_contract_digest") != plan.get("run_contract_digest")
        or seal.get("hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or seal.get("root_count") != 100
        or seal.get("observation_count") != 200
        or seal.get("profile_counts") != expected_profiles
        or seal.get("seat_counts") != {"first": 100, "second": 100}
        or root_hashes != materialization.get("root_artifact_sha256")
        or len(root_hashes) != 100
        or len(set(root_hashes)) != 100
        or any(not _is_sha256(value) for value in root_hashes)
        or seal.get("aggregate_root_sha256") != lock_open.canonical_sha256(root_hashes)
        or seal.get("aggregate_root_sha256")
        != materialization.get("aggregate_root_sha256")
        or seal.get("root_artifact_unique") is not True
        or seal.get("observation_fingerprint_unique") is not True
        or seal.get("development_comparison")
        != {
            "development_all100_root_sha256": (
                _mapping(
                    seal.get("development_comparison"),
                    "seal development comparison",
                ).get("development_all100_root_sha256")
            ),
            "development_root_count": 100,
            "lock_fingerprint_overlap_count": 0,
            "lock_root_hash_overlap_count": 0,
            "lock_seed_overlap_count": 0,
        }
        or seal.get("visibility")
        != {
            "runner_validator_replayed_all_roots": True,
            "first_observation_count": 100,
            "second_observation_count": 100,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        }
        or seal.get("selection_inputs")
        != {
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
            "all_100_preregistered_hands_used": True,
        }
        or any(
            seal.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("performance-lock materialization/root-seal lineage changed")
    for field in (
        "root_topology_sha256",
        "observation_fingerprint_sha256",
    ):
        _require_sha256(seal.get(field), f"root-seal {field}")
    _require_sha256(
        _mapping(
            seal.get("development_comparison"),
            "seal development comparison",
        ).get("development_all100_root_sha256"),
        "sealed development all100 root SHA-256",
    )


def _validated_received_execution(
    *,
    inputs: PerformanceLockReceiveInputs,
    plan: Mapping[str, Any],
    lineage_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[Path]], dict[str, Any]]:
    receive_dir = Path(inputs.receive_dir).resolve()
    raw_receipt_record = _file_record(
        receive_dir / "receive_receipt.json",
        "performance-lock receive receipt",
    )
    raw_receipt = raw_receipt_record.pop("_value")
    receipt = lock_spot.validate_received_directory(
        receive_dir=receive_dir,
        run_dir=Path(inputs.run_dir).resolve(),
        project=inputs.project,
        bucket=inputs.bucket,
    )
    if receipt != raw_receipt:
        raise ValueError("performance-lock receive replay changed")
    result_claim_record = _file_record(
        receive_dir / lock_spot.RESULT_OPEN_CLAIM_NAME,
        "performance-lock result-open claim",
    )
    result_claim_record.pop("_value")

    jobs = _array(receipt.get("jobs"), "performance-lock received jobs")
    done_paths: dict[str, list[Path]] = {role: [] for role in runner.SOURCE_ROLES}
    received_job_sha256: list[dict[str, Any]] = []
    for raw in jobs:
        job = _mapping(raw, "performance-lock received job")
        role = job.get("source_role")
        relative = job.get("done_path")
        if role not in runner.SOURCE_ROLES or not isinstance(relative, str):
            raise ValueError("performance-lock received job identity changed")
        fragment = Path(relative)
        if fragment.is_absolute() or ".." in fragment.parts:
            raise ValueError("performance-lock received DONE path escapes receive tree")
        path = (receive_dir / fragment).resolve()
        try:
            path.relative_to(receive_dir)
        except ValueError as error:
            raise ValueError(
                "performance-lock received DONE path escapes receive tree"
            ) from error
        observed_sha = lock_open.sha256_file(path)
        expected_sha = _require_sha256(
            job.get("done_sha256"), "performance-lock received DONE SHA-256"
        )
        if observed_sha != expected_sha:
            raise ValueError("performance-lock received DONE bytes changed")
        done_paths[str(role)].append(path)
        received_job_sha256.append(
            {
                "job_id": job.get("job_id"),
                "source_role": role,
                "work_hand_indices": list(
                    _array(
                        job.get("work_hand_indices"),
                        "received job work indices",
                    )
                ),
                "done_path": relative,
                "done_sha256": expected_sha,
            }
        )
    expected_relative = {
        role: [
            str(_mapping(job, "received job").get("done_path"))
            for job in jobs
            if _mapping(job, "received job").get("source_role") == role
        ]
        for role in runner.SOURCE_ROLES
    }
    if (
        receipt.get("schema") != lock_spot.RECEIVE_SCHEMA
        or receipt.get("status")
        != "exact_performance_lock_source_shards_received_and_validated"
        or receipt.get("precontent_plan_sha256")
        != _mapping(
            lineage_inputs.get("precontent_plan"), "lineage precontent plan"
        ).get("sha256")
        or receipt.get("root_open_claim_sha256")
        != _mapping(lineage_inputs.get("open_claim"), "lineage open claim").get(
            "sha256"
        )
        or receipt.get("root_seal_sha256")
        != _mapping(lineage_inputs.get("root_seal"), "lineage root seal").get("sha256")
        or receipt.get("run_contract_digest") != plan.get("run_contract_digest")
        or receipt.get("source_roles") != list(runner.SOURCE_ROLES)
        or receipt.get("logical_job_count") != lock_plan.LOGICAL_JOB_COUNT
        or receipt.get("paired_hand_count") != 100
        or receipt.get("root_count") != 200
        or len(jobs) != lock_plan.LOGICAL_JOB_COUNT
        or receipt.get("candidate_done_paths") != expected_relative["candidate"]
        or receipt.get("reference_done_paths") != expected_relative["reference"]
        or receipt.get("result_open_claim_sha256") != result_claim_record["sha256"]
        or any(
            receipt.get(field) is not True
            for field in (
                "source_isolation_validated",
                "root_pairing_validated",
                "sealed_root_set_validated",
            )
        )
        or any(
            receipt.get(field) is not False
            for field in (
                "merge_executed",
                "performance_lock_finalized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
        or any(
            len(done_paths[role]) != lock_plan.SHARD_COUNT_PER_ROLE
            for role in runner.SOURCE_ROLES
        )
    ):
        raise ValueError("performance-lock one-shot receive lineage changed")
    value = {
        "receive_receipt": raw_receipt_record,
        "result_open_claim": result_claim_record,
        "run_name": receipt["run_name"],
        "package_manifest_sha256": receipt["package_manifest_sha256"],
        "launch_authorization_sha256": receipt["launch_authorization_sha256"],
        "launch_claim_sha256": receipt["launch_claim_sha256"],
        "launch_result_sha256": receipt["launch_result_sha256"],
        "resume_claim_sha256": receipt["resume_claim_sha256"],
        "resume_result_sha256": receipt["resume_result_sha256"],
        "result_open_claim_sha256": receipt["result_open_claim_sha256"],
        "precontent_plan_sha256": receipt["precontent_plan_sha256"],
        "root_open_claim_sha256": receipt["root_open_claim_sha256"],
        "root_seal_sha256": receipt["root_seal_sha256"],
        "run_contract_digest": receipt["run_contract_digest"],
        "logical_job_count": receipt["logical_job_count"],
        "paired_hand_count": receipt["paired_hand_count"],
        "root_count": receipt["root_count"],
        "received_job_sha256": received_job_sha256,
        "one_shot_result_claim_replayed": True,
        "launch_chain_replayed": True,
        "receive_directory_replayed": True,
        "exact_received_done_set_bound": True,
    }
    if set(value) != _RECEIVED_EXECUTION_KEYS:
        raise AssertionError("performance-lock received execution schema changed")
    return receipt, done_paths, value


def _validate_plan_partitions(
    plan: Mapping[str, Any], generic: Mapping[str, Any]
) -> dict[str, Any]:
    jobs = _array(plan.get("jobs"), "performance-lock plan jobs")
    source_inputs = _mapping(
        generic.get("source_done_inputs"), "generic source DONE inputs"
    )
    partition_rows: dict[str, list[dict[str, Any]]] = {}
    for role in runner.SOURCE_ROLES:
        expected = [
            {
                "work_hand_indices": list(job["work_hand_indices"]),
                "shard_manifest_sha256": job["shard_manifest_sha256"],
            }
            for job in jobs
            if _mapping(job, "performance-lock plan job").get("source_role") == role
        ]
        actual = [
            {
                "work_hand_indices": list(
                    _mapping(raw, f"{role} DONE input")["work_hand_indices"]
                ),
                "shard_manifest_sha256": _mapping(raw, f"{role} DONE input")[
                    "shard_manifest_digest"
                ],
            }
            for raw in _array(source_inputs.get(role), f"{role} DONE inputs")
        ]
        expected.sort(key=lambda row: row["work_hand_indices"])
        actual.sort(key=lambda row: row["work_hand_indices"])
        if actual != expected:
            raise ValueError(f"performance-lock {role} shard partition changed")
        partition_rows[role] = expected
    candidate_work = [
        index
        for row in partition_rows["candidate"]
        for index in row["work_hand_indices"]
    ]
    reference_work = [
        index
        for row in partition_rows["reference"]
        for index in row["work_hand_indices"]
    ]
    if (
        candidate_work != list(runner.CONTRACT_HAND_INDICES)
        or reference_work != candidate_work
    ):
        raise ValueError("performance-lock source work coverage changed")
    value = {
        "source_roles": list(runner.SOURCE_ROLES),
        "shard_count_per_role": lock_plan.SHARD_COUNT_PER_ROLE,
        "hands_per_shard": lock_plan.HANDS_PER_SHARD,
        "logical_job_count": lock_plan.LOGICAL_JOB_COUNT,
        "candidate_shard_count": len(partition_rows["candidate"]),
        "reference_shard_count": len(partition_rows["reference"]),
        "candidate_partition_sha256": lock_open.canonical_sha256(
            partition_rows["candidate"]
        ),
        "reference_partition_sha256": lock_open.canonical_sha256(
            partition_rows["reference"]
        ),
        "candidate_partition_exact": True,
        "reference_partition_exact": True,
        "source_roles_isolated": True,
        "work_coverage_exact": True,
    }
    if set(value) != _PARTITION_KEYS:
        raise AssertionError("performance-lock partition report schema changed")
    return value


def _validate_generic_and_root_pairing(
    *,
    generic: Mapping[str, Any],
    plan: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    paired = _array(generic.get("paired_artifacts"), "paired artifacts")
    ordered = [_mapping(raw, "paired artifact") for raw in paired]
    artifact_hashes = [
        _require_sha256(row.get("root_artifact_sha256"), "paired root SHA-256")
        for row in ordered
    ]
    file_hashes = [
        _require_sha256(row.get("root_file_sha256"), "paired root-file SHA-256")
        for row in ordered
    ]
    seal_hashes = _array(seal.get("root_artifact_sha256"), "root-seal artifact hashes")
    if (
        generic.get("scope") != base.FULL_PERFORMANCE_SCOPE
        or generic.get("run_contract") != plan.get("run_contract")
        or generic.get("run_contract_digest") != plan.get("run_contract_digest")
        or generic.get("hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or generic.get("paired_hand_count") != 100
        or generic.get("root_count") != 200
        or [row.get("hand_index") for row in ordered]
        != list(runner.CONTRACT_HAND_INDICES)
        or any(row.get("paired_seat_parity_count") != 2 for row in ordered)
        or artifact_hashes != file_hashes
        or artifact_hashes != seal_hashes
        or artifact_hashes != materialization.get("root_artifact_sha256")
        or lock_open.canonical_sha256(artifact_hashes)
        != seal.get("aggregate_root_sha256")
    ):
        raise ValueError("performance-lock generic/root pairing changed")
    development = _mapping(
        seal.get("development_comparison"), "seal development comparison"
    )
    overlap_count = sum(
        _integer(
            development.get(field),
            f"seal {field}",
        )
        for field in (
            "lock_fingerprint_overlap_count",
            "lock_root_hash_overlap_count",
            "lock_seed_overlap_count",
        )
    )
    value = {
        "root_artifact_count": len(artifact_hashes),
        "observation_count": generic["root_count"],
        "root_artifact_sha256": artifact_hashes,
        "aggregate_root_sha256": seal["aggregate_root_sha256"],
        "root_topology_sha256": seal["root_topology_sha256"],
        "observation_fingerprint_sha256": seal["observation_fingerprint_sha256"],
        "candidate_reference_root_pairing_exact": True,
        "materialization_exact": True,
        "seal_exact": True,
        "development_overlap_count": overlap_count,
    }
    if set(value) != _ROOT_LINEAGE_KEYS or overlap_count != 0:
        raise ValueError("performance-lock root lineage changed")
    return value


def _gate_groups(
    generic: Mapping[str, Any],
) -> tuple[dict[str, bool], bool, dict[str, bool], bool]:
    integrity = _mapping(generic.get("integrity"), "generic integrity")
    performance = _mapping(generic.get("performance"), "generic performance")
    candidate = _mapping(performance.get("candidate_by_seat"), "candidate performance")
    first = _mapping(candidate.get("first"), "candidate first performance")
    second = _mapping(candidate.get("second"), "candidate second performance")
    paired = _array(generic.get("paired_artifacts"), "paired artifacts")
    profile_counts = {
        profile: sum(
            _mapping(raw, "paired artifact").get("profile") == profile for raw in paired
        )
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    missing_or_censored_zero = (
        generic.get("hand_indices") == list(runner.CONTRACT_HAND_INDICES)
        and integrity.get("candidate_hand_count") == 100
        and integrity.get("reference_hand_count") == 100
        and integrity.get("paired_hand_count") == 100
        and integrity.get("paired_root_count") == 200
        and integrity.get("unique_observation_fingerprint_count") == 200
        and integrity.get("paired_hand_parity_count") == 100
        and integrity.get("paired_root_parity_count") == 200
        and integrity.get("missing_hand_indices") == []
        and integrity.get("duplicate_hand_indices") == []
        and integrity.get("out_of_contract_hand_indices") == []
    )
    portable_fraction = _integer(
        integrity.get("paired_root_parity_count"),
        "paired root parity count",
    ) / _integer(generic.get("root_count"), "generic root count", minimum=1)
    first_p95 = _finite(first.get("p95_seconds"), "candidate first p95")
    first_p99 = _finite(first.get("p99_seconds"), "candidate first p99")
    first_max = _finite(first.get("max_seconds"), "candidate first max")
    second_p95 = _finite(second.get("p95_seconds"), "candidate second p95")
    peak_rss = _integer(
        performance.get("peak_source_process_rss_bytes"),
        "peak source-process RSS",
    )
    hard = {
        "exactly_100_paired_hands": generic.get("paired_hand_count") == 100,
        "exactly_200_roots": generic.get("root_count") == 200,
        "exactly_100_first_and_100_second": first.get("count")
        == second.get("count")
        == 100,
        "exactly_20_hands_per_profile": set(profile_counts.values()) == {20},
        "missing_or_censored_roots_zero": missing_or_censored_zero,
        "portable_semantic_parity_fraction_one": portable_fraction == 1.0,
        "first_p95_within_180_seconds": first_p95 <= HARD_FIRST_P95_SECONDS_MAX,
        "first_p99_within_240_seconds": first_p99 <= HARD_FIRST_P99_SECONDS_MAX,
        "first_max_within_240_seconds": first_max <= HARD_FIRST_MAX_SECONDS_MAX,
        "second_p95_within_6_seconds": second_p95 <= HARD_SECOND_P95_SECONDS_MAX,
        "peak_rss_within_1073741824_bytes": peak_rss <= HARD_PEAK_RSS_BYTES_MAX,
    }
    guards = {
        "first_p95_within_150_seconds": first_p95 <= QUALITY_FIRST_P95_SECONDS_MAX,
        "second_p95_within_5_seconds": second_p95 <= QUALITY_SECOND_P95_SECONDS_MAX,
        "peak_rss_within_858993459_bytes": peak_rss <= QUALITY_PEAK_RSS_BYTES_MAX,
    }
    if set(hard) != _HARD_GATE_KEYS or set(guards) != _QUALITY_GUARD_KEYS:
        raise AssertionError("performance-lock gate schema changed")
    return hard, all(hard.values()), guards, all(guards.values())


def merge_candidate02_performance_lock(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    lock_inputs: lock_open.PerformanceLockInputs,
    receive_inputs: PerformanceLockReceiveInputs,
) -> dict[str, Any]:
    """Replay and merge exactly the preregistered 100-hand lock."""

    plan, claim, materialization, seal, lineage_inputs = _validated_lineage(lock_inputs)
    _validate_claim_plan_lineage(inputs=lock_inputs, plan=plan, claim=claim)
    _validate_materialization_and_seal(
        plan=plan,
        claim=claim,
        materialization=materialization,
        seal=seal,
    )
    _receipt, received_done_paths, received_execution = _validated_received_execution(
        inputs=receive_inputs,
        plan=plan,
        lineage_inputs=lineage_inputs,
    )
    supplied_paths = {
        "candidate": [Path(path).resolve() for path in candidate_done_paths],
        "reference": [Path(path).resolve() for path in reference_done_paths],
    }
    for role in runner.SOURCE_ROLES:
        if len(supplied_paths[role]) != len(set(supplied_paths[role])) or set(
            supplied_paths[role]
        ) != set(received_done_paths[role]):
            raise ValueError(
                f"performance-lock {role} DONE inputs are not the claimed "
                "one-shot receive set"
            )
    generic = base.merge_performance_v2(
        candidate_done_paths=received_done_paths["candidate"],
        reference_done_paths=received_done_paths["reference"],
        scope=base.FULL_PERFORMANCE_SCOPE,
        contract_variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    partitions = _validate_plan_partitions(plan, generic)
    root_lineage = _validate_generic_and_root_pairing(
        generic=generic,
        plan=plan,
        materialization=materialization,
        seal=seal,
    )
    hard, hard_passed, guards, guards_passed = _gate_groups(generic)
    passed = hard_passed and guards_passed
    value = {
        "schema": MERGE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": PASS_DECISION if passed else NO_GO_DECISION,
        "scope": SCOPE,
        "contract_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
        "run_contract": generic["run_contract"],
        "run_contract_digest": generic["run_contract_digest"],
        "hand_indices": list(generic["hand_indices"]),
        "paired_hand_count": generic["paired_hand_count"],
        "root_count": generic["root_count"],
        "budget": dict(generic["budget"]),
        "allocation": dict(generic["allocation"]),
        "source_identity": dict(plan["source_identity"]),
        "open_inputs": _serialize_open_inputs(lock_inputs),
        "receive_inputs": _serialize_receive_inputs(receive_inputs),
        "lineage_inputs": lineage_inputs,
        "received_execution": received_execution,
        "source_done_inputs": generic["source_done_inputs"],
        "paired_artifacts": generic["paired_artifacts"],
        "integrity": generic["integrity"],
        "performance": generic["performance"],
        "generic_merge_sha256": runner.canonical_sha256(generic),
        "plan_partitions": partitions,
        "root_lineage": root_lineage,
        "gate_mode": GATE_MODE,
        "hard_qualification_gates": hard,
        "hard_qualification_passed": hard_passed,
        "quality_launch_guard_gates": guards,
        "quality_launch_guards_passed": guards_passed,
        "all_gates_passed": passed,
        "performance_lock_qualified": passed,
        "candidate_finalized_no_go": not passed,
        "one_shot_lock_consumed": True,
        "rerun_authorized": False,
        "reseed_authorized": False,
        "quality_pilot_authorized": passed,
        "artifact_fanout_authorized": False,
        "training_eligible": False,
        "training_authorized": False,
        "promotion_evidence": False,
        "promotion_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    if set(value) != _SUMMARY_KEYS:
        raise AssertionError("performance-lock scientific merge schema changed")
    return value


def _source_input_paths(summary: Mapping[str, Any], role: str) -> list[Path]:
    # The generic helper rechecks every stored DONE byte hash before returning.
    return base._input_paths(summary, role)


def _lineage_paths(summary: Mapping[str, Any]) -> None:
    inputs = _deserialize_open_inputs(summary.get("open_inputs"))
    receive_inputs = _deserialize_receive_inputs(summary.get("receive_inputs"))
    raw = _mapping(summary.get("lineage_inputs"), "lineage inputs")
    if set(raw) != _LINEAGE_INPUT_KEYS:
        raise ValueError("performance-lock lineage input fields changed")
    expected = {
        "precontent_plan": inputs.plan_path,
        "open_claim": inputs.global_claim_path,
        "materialization": Path(inputs.lock_output_directory) / "materialization.json",
        "root_seal": Path(inputs.lock_output_directory) / "seal.json",
    }
    for label, path in expected.items():
        _validate_file_record(raw.get(label), label, expected_path=path)
    execution = _mapping(summary.get("received_execution"), "received execution")
    if set(execution) != _RECEIVED_EXECUTION_KEYS:
        raise ValueError("performance-lock received execution fields changed")
    _validate_file_record(
        execution.get("receive_receipt"),
        "performance-lock receive receipt",
        expected_path=Path(receive_inputs.receive_dir) / "receive_receipt.json",
    )
    _validate_file_record(
        execution.get("result_open_claim"),
        "performance-lock result-open claim",
        expected_path=Path(receive_inputs.receive_dir)
        / lock_spot.RESULT_OPEN_CLAIM_NAME,
    )


def _validation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    root_lineage = _mapping(summary.get("root_lineage"), "root lineage")
    value = {
        "schema": VALIDATION_SCHEMA,
        "status": summary["status"],
        "decision": summary["decision"],
        "scope": summary["scope"],
        "summary_sha256": runner.canonical_sha256(summary),
        "run_contract_digest": summary["run_contract_digest"],
        "hand_indices": list(summary["hand_indices"]),
        "paired_hand_count": summary["paired_hand_count"],
        "root_count": summary["root_count"],
        "source_artifacts_replayed": True,
        "open_claim_replayed": True,
        "root_seal_replayed": True,
        "one_shot_receive_replayed": True,
        "launch_chain_replayed": True,
        "plan_partitions_recomputed": True,
        "root_set_recomputed": True,
        "candidate_reference_root_pairing_exact": root_lineage[
            "candidate_reference_root_pairing_exact"
        ],
        "hard_qualification_gates": dict(summary["hard_qualification_gates"]),
        "hard_qualification_passed": summary["hard_qualification_passed"],
        "quality_launch_guard_gates": dict(summary["quality_launch_guard_gates"]),
        "quality_launch_guards_passed": summary["quality_launch_guards_passed"],
        "all_gates_passed": summary["all_gates_passed"],
        "performance_lock_qualified": summary["performance_lock_qualified"],
        "candidate_finalized_no_go": summary["candidate_finalized_no_go"],
        "one_shot_lock_consumed": True,
        "rerun_authorized": False,
        "reseed_authorized": False,
        "quality_pilot_authorized": summary["quality_pilot_authorized"],
        "artifact_fanout_authorized": False,
        "training_eligible": False,
        "training_authorized": False,
        "promotion_evidence": False,
        "promotion_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    if set(value) != _VALIDATION_KEYS:
        raise AssertionError("performance-lock validation schema changed")
    return value


def validate_candidate02_performance_lock_merge(
    *, summary_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    """Independently replay every source and lineage input in a stored merge."""

    summary_path = Path(summary_path).resolve()
    summary = _read_canonical(
        summary_path, "Candidate02 performance-lock scientific merge"
    )
    if set(summary) != _SUMMARY_KEYS or summary.get("schema") != MERGE_SCHEMA:
        raise ValueError("performance-lock scientific merge schema changed")
    _lineage_paths(summary)
    lock_inputs = _deserialize_open_inputs(summary["open_inputs"])
    receive_inputs = _deserialize_receive_inputs(summary["receive_inputs"])
    candidate_paths = _source_input_paths(summary, "candidate")
    reference_paths = _source_input_paths(summary, "reference")
    if output_path is not None:
        validation_output = Path(output_path).resolve()
        if validation_output == summary_path:
            raise ValueError("summary and validation outputs must be distinct")
        _validate_output_locations(
            summary_output=summary_path,
            validation_output=validation_output,
            candidate_done_paths=candidate_paths,
            reference_done_paths=reference_paths,
            lock_inputs=lock_inputs,
            receive_inputs=receive_inputs,
        )
    expected = merge_candidate02_performance_lock(
        candidate_done_paths=candidate_paths,
        reference_done_paths=reference_paths,
        lock_inputs=lock_inputs,
        receive_inputs=receive_inputs,
    )
    if summary != expected:
        raise ValueError("performance-lock scientific merge aggregate changed")
    report = _validation_report(summary)
    if output_path is not None:
        base._write_once(validation_output, report)
    return report


def _is_at_or_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_output_locations(
    *,
    summary_output: Path,
    validation_output: Path,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    lock_inputs: lock_open.PerformanceLockInputs,
    receive_inputs: PerformanceLockReceiveInputs,
) -> None:
    forbidden = {
        Path(lock_inputs.lock_output_directory).resolve(),
        Path(receive_inputs.run_dir).resolve(),
        Path(receive_inputs.receive_dir).resolve(),
        *(
            Path(path).resolve().parent
            for path in (*candidate_done_paths, *reference_done_paths)
        ),
    }
    for target in (summary_output, validation_output):
        if any(_is_at_or_below(target, parent) for parent in forbidden):
            raise ValueError(
                "performance-lock merge output must be outside source, "
                "receive, run, and sealed-root trees"
            )


def merge_and_validate_candidate02_performance_lock(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    lock_inputs: lock_open.PerformanceLockInputs,
    receive_inputs: PerformanceLockReceiveInputs,
    summary_output_path: Path,
    validation_output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the merge once, then independently replay it before validation."""

    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    if summary_output == validation_output:
        raise ValueError("summary and validation outputs must be distinct")
    _validate_output_locations(
        summary_output=summary_output,
        validation_output=validation_output,
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        lock_inputs=lock_inputs,
        receive_inputs=receive_inputs,
    )
    if summary_output.exists() or validation_output.exists():
        raise FileExistsError("performance-lock merge outputs are write-once")
    summary = merge_candidate02_performance_lock(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        lock_inputs=lock_inputs,
        receive_inputs=receive_inputs,
    )
    base._write_once(summary_output, summary)
    validation = validate_candidate02_performance_lock_merge(
        summary_path=summary_output,
        output_path=validation_output,
    )
    return summary, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-done", type=Path, action="append", required=True)
    parser.add_argument("--reference-done", type=Path, action="append", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receive-dir", type=Path, required=True)
    parser.add_argument("--project", default=lock_spot.DEFAULT_PROJECT)
    parser.add_argument("--bucket", default=lock_spot.DEFAULT_BUCKET)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=lock_open.DEFAULT_REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--plan", type=Path, default=lock_open.DEFAULT_PRECONTENT_PLAN_PATH
    )
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--feature-encoder", type=Path, required=True)
    parser.add_argument("--startup-source", type=Path, required=True)
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json",
    )
    parser.add_argument(
        "--development-validation",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json",
    )
    parser.add_argument(
        "--development-roots",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_ROOT_DIR,
    )
    parser.add_argument(
        "--global-claim",
        type=Path,
        default=lock_open.DEFAULT_GLOBAL_CLAIM_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = lock_open.PerformanceLockInputs(
        repository_root=args.repository_root,
        plan_path=args.plan,
        lock_output_directory=args.lock_output,
        candidate_library=args.candidate_library,
        reference_library=args.reference_library,
        feature_encoder=args.feature_encoder,
        startup_source=args.startup_source,
        development_summary_path=args.development_summary,
        development_validation_path=args.development_validation,
        development_root_directory=args.development_roots,
        global_claim_path=args.global_claim,
    )
    receive_inputs = PerformanceLockReceiveInputs(
        run_dir=args.run_dir,
        receive_dir=args.receive_dir,
        project=args.project,
        bucket=args.bucket,
    )
    summary, _ = merge_and_validate_candidate02_performance_lock(
        candidate_done_paths=args.candidate_done,
        reference_done_paths=args.reference_done,
        lock_inputs=inputs,
        receive_inputs=receive_inputs,
        summary_output_path=args.summary_output,
        validation_output_path=args.validation_output,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HARD_FIRST_MAX_SECONDS_MAX",
    "HARD_FIRST_P95_SECONDS_MAX",
    "HARD_FIRST_P99_SECONDS_MAX",
    "HARD_PEAK_RSS_BYTES_MAX",
    "HARD_SECOND_P95_SECONDS_MAX",
    "MERGE_SCHEMA",
    "PerformanceLockReceiveInputs",
    "QUALITY_FIRST_P95_SECONDS_MAX",
    "QUALITY_PEAK_RSS_BYTES_MAX",
    "QUALITY_SECOND_P95_SECONDS_MAX",
    "VALIDATION_SCHEMA",
    "main",
    "merge_and_validate_candidate02_performance_lock",
    "merge_candidate02_performance_lock",
    "validate_candidate02_performance_lock_merge",
]
