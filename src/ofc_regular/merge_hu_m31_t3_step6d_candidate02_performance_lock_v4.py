"""Merge and independently replay the run009-authorized performance lock v4.

The generic Step6d merger remains the source-artifact correctness oracle.  This
wrapper binds its fresh candidate/reference DONE replay to the immutable v4
plan, materialization receipt, root seal, and the stricter v4 speed/performance
gate.  It has no cloud client and cannot modify an AI profile.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as lock_plan
from . import hu_m31_t3_step6d_performance_lock_v4_gate as lock_gate
from . import merge_hu_m31_t3_step6d_performance_v2 as performance_v2
from . import run_hu_m31_t3_step6d_performance_v2 as runner


MERGE_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_lock_v4_merge_v1"
SCOPE = "candidate02_one_shot_performance_lock_v4"

_PARTITION_KEYS = frozenset(
    {
        "source_roles",
        "shard_count_per_role",
        "hands_per_shard",
        "logical_job_count",
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
        "root_file_count",
        "paired_hand_count",
        "observation_count",
        "root_hashes",
        "root_hash_aggregate_sha256",
        "observation_fingerprint_count",
        "observation_fingerprint_aggregate_sha256",
        "candidate_reference_root_pairing_exact",
        "materialization_exact",
        "seal_exact",
        "seed_overlap_zero",
        "old_root_or_seed_reuse",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "performance_lock_plan",
        "performance_lock_plan_sha256",
        "materialization_receipt",
        "materialization_receipt_sha256",
        "root_seal",
        "root_seal_sha256",
        "run009_scientific_gate_binding",
        "contract_variant",
        "run_contract",
        "run_contract_digest",
        "generic_merge",
        "generic_merge_sha256",
        "plan_partitions",
        "root_lineage",
        "performance_gate",
        "performance_gate_sha256",
        "all_gates_passed",
        "scientific_performance_gate_passed",
        "transport_lineage_required",
        "transport_lineage_validated",
        "performance_lock_finalized",
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
        "summary_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _validate_lineage(
    *,
    plan_value: Mapping[str, Any],
    materialization_value: Mapping[str, Any],
    root_seal_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = lock_plan.validate_performance_lock_v4_plan(plan_value)
    materialization = lock_plan.validate_materialization_receipt(
        materialization_value
    )
    seal = lock_plan.validate_root_seal(root_seal_value)
    if (
        canonical_sha256(plan) != lock_plan.PLAN_SHA256
        or plan.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
        or materialization.get("plan_sha256") != lock_plan.PLAN_SHA256
        or seal.get("plan_sha256") != lock_plan.PLAN_SHA256
        or materialization.get("run_contract_digest")
        != lock_plan.RUN_CONTRACT_DIGEST
        or seal.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
        or seal.get("claim_sha256") != materialization.get("claim_sha256")
        or seal.get("materialization_receipt_sha256")
        != canonical_sha256(materialization)
        or seal.get("root_output_directory")
        != materialization.get("root_output_directory")
        or seal.get("root_hashes") != materialization.get("root_hashes")
        or seal.get("root_hash_aggregate_sha256")
        != materialization.get("root_hash_aggregate_sha256")
        or seal.get("observation_fingerprint_aggregate_sha256")
        != materialization.get("observation_fingerprint_aggregate_sha256")
        or seal.get("seed_set_sha256") != materialization.get("seed_set_sha256")
        or seal.get("seed_overlap_counts")
        != materialization.get("seed_overlap_counts")
        or any(seal["seed_overlap_counts"].values())
        or seal.get("old_root_or_seed_reuse") is not False
        or materialization.get("old_root_or_seed_reuse") is not False
    ):
        raise ValueError("performance-lock-v4 plan/materialization/seal lineage changed")
    return plan, materialization, seal


def _validate_partitions(
    plan: Mapping[str, Any], generic: Mapping[str, Any]
) -> dict[str, Any]:
    jobs = _array(plan.get("jobs"), "performance-lock-v4 plan jobs")
    source_inputs = _mapping(generic.get("source_done_inputs"), "source DONE inputs")
    partitions: dict[str, list[dict[str, Any]]] = {}
    all_paths: dict[str, set[str]] = {}
    for role in runner.SOURCE_ROLES:
        expected = [
            {
                "work_hand_indices": list(job["work_hand_indices"]),
                "shard_manifest_sha256": job["shard_manifest_sha256"],
            }
            for job in jobs
            if _mapping(job, "performance-lock-v4 plan job").get("source_role")
            == role
        ]
        actual_rows = _array(source_inputs.get(role), f"{role} DONE inputs")
        actual = [
            {
                "work_hand_indices": list(
                    _mapping(row, f"{role} DONE input")["work_hand_indices"]
                ),
                "shard_manifest_sha256": _mapping(
                    row, f"{role} DONE input"
                )["shard_manifest_digest"],
            }
            for row in actual_rows
        ]
        expected.sort(key=lambda row: row["work_hand_indices"])
        actual.sort(key=lambda row: row["work_hand_indices"])
        if expected != actual or len(actual) != lock_plan.SHARD_COUNT_PER_ROLE:
            raise ValueError(f"performance-lock-v4 {role} partition changed")
        partitions[role] = expected
        all_paths[role] = {
            str(_mapping(row, f"{role} DONE input").get("path"))
            for row in actual_rows
        }
        if len(all_paths[role]) != lock_plan.SHARD_COUNT_PER_ROLE:
            raise ValueError(f"performance-lock-v4 {role} DONE path duplicated")
    candidate_work = sorted(
        index
        for row in partitions["candidate"]
        for index in row["work_hand_indices"]
    )
    reference_work = sorted(
        index
        for row in partitions["reference"]
        for index in row["work_hand_indices"]
    )
    if (
        candidate_work != list(runner.CONTRACT_HAND_INDICES)
        or reference_work != candidate_work
        or all_paths["candidate"] & all_paths["reference"]
    ):
        raise ValueError("performance-lock-v4 source isolation/coverage changed")
    value = {
        "source_roles": list(runner.SOURCE_ROLES),
        "shard_count_per_role": lock_plan.SHARD_COUNT_PER_ROLE,
        "hands_per_shard": lock_plan.HANDS_PER_SHARD,
        "logical_job_count": lock_plan.LOGICAL_JOB_COUNT,
        "candidate_partition_sha256": canonical_sha256(partitions["candidate"]),
        "reference_partition_sha256": canonical_sha256(partitions["reference"]),
        "candidate_partition_exact": True,
        "reference_partition_exact": True,
        "source_roles_isolated": True,
        "work_coverage_exact": True,
    }
    if set(value) != _PARTITION_KEYS:
        raise AssertionError("performance-lock-v4 partition schema changed")
    return value


def _validate_root_lineage(
    *, generic: Mapping[str, Any], materialization: Mapping[str, Any], seal: Mapping[str, Any]
) -> dict[str, Any]:
    paired = _array(generic.get("paired_artifacts"), "paired artifacts")
    if len(paired) != 100:
        raise ValueError("performance-lock-v4 root pairing count changed")
    root_hashes: list[str] = []
    fingerprints: list[str] = []
    for expected_hand, raw_hand in enumerate(paired):
        hand = _mapping(raw_hand, "paired artifact")
        if hand.get("hand_index") != expected_hand:
            raise ValueError("performance-lock-v4 paired root ordering changed")
        root_hash = hand.get("root_file_sha256")
        if (
            not isinstance(root_hash, str)
            or root_hash != hand.get("root_artifact_sha256")
        ):
            raise ValueError("performance-lock-v4 candidate/reference root hash changed")
        root_hashes.append(root_hash)
        for raw_row in _array(hand.get("rows"), "paired artifact rows"):
            fingerprint = _mapping(raw_row, "paired artifact row").get(
                "observation_fingerprint"
            )
            if not isinstance(fingerprint, str):
                raise ValueError("performance-lock-v4 observation fingerprint missing")
            fingerprints.append(fingerprint)
    if (
        root_hashes != seal.get("root_hashes")
        or root_hashes != materialization.get("root_hashes")
        or canonical_sha256(root_hashes) != seal.get("root_hash_aggregate_sha256")
        or len(fingerprints) != 200
        or len(set(fingerprints)) != 200
        or canonical_sha256(sorted(fingerprints))
        != seal.get("observation_fingerprint_aggregate_sha256")
    ):
        raise ValueError("performance-lock-v4 sealed root lineage changed")
    value = {
        "root_file_count": 100,
        "paired_hand_count": 100,
        "observation_count": 200,
        "root_hashes": root_hashes,
        "root_hash_aggregate_sha256": seal["root_hash_aggregate_sha256"],
        "observation_fingerprint_count": 200,
        "observation_fingerprint_aggregate_sha256": seal[
            "observation_fingerprint_aggregate_sha256"
        ],
        "candidate_reference_root_pairing_exact": True,
        "materialization_exact": True,
        "seal_exact": True,
        "seed_overlap_zero": True,
        "old_root_or_seed_reuse": False,
    }
    if set(value) != _ROOT_LINEAGE_KEYS:
        raise AssertionError("performance-lock-v4 root-lineage schema changed")
    return value


def merge_candidate02_performance_lock_v4(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    plan_value: Mapping[str, Any],
    materialization_value: Mapping[str, Any],
    root_seal_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay source DONE inputs and return the deterministic one-shot result."""

    plan, materialization, seal = _validate_lineage(
        plan_value=plan_value,
        materialization_value=materialization_value,
        root_seal_value=root_seal_value,
    )
    generic = performance_v2.merge_performance_v2(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        scope=performance_v2.FULL_PERFORMANCE_SCOPE,
        contract_variant=runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
    )
    if (
        generic.get("run_contract") != plan["run_contract"]
        or generic.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
    ):
        raise ValueError("performance-lock-v4 source contract differs from plan")
    partitions = _validate_partitions(plan, generic)
    root_lineage = _validate_root_lineage(
        generic=generic, materialization=materialization, seal=seal
    )
    gate = lock_gate.build_performance_lock_v4_gate(
        generic, run_contract=plan["run_contract"]
    )
    passed = gate["all_gates_passed"] is True
    body = {
        "schema": MERGE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": lock_gate.PASS_DECISION if passed else lock_gate.NO_GO_DECISION,
        "scope": SCOPE,
        "performance_lock_plan": deepcopy(plan),
        "performance_lock_plan_sha256": canonical_sha256(plan),
        "materialization_receipt": deepcopy(materialization),
        "materialization_receipt_sha256": canonical_sha256(materialization),
        "root_seal": deepcopy(seal),
        "root_seal_sha256": canonical_sha256(seal),
        "run009_scientific_gate_binding": deepcopy(plan["run009_scientific_gate"]),
        "contract_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
        "run_contract": deepcopy(plan["run_contract"]),
        "run_contract_digest": lock_plan.RUN_CONTRACT_DIGEST,
        "generic_merge": generic,
        "generic_merge_sha256": canonical_sha256(generic),
        "plan_partitions": partitions,
        "root_lineage": root_lineage,
        "performance_gate": gate,
        "performance_gate_sha256": canonical_sha256(gate),
        "all_gates_passed": passed,
        "scientific_performance_gate_passed": passed,
        "transport_lineage_required": True,
        "transport_lineage_validated": False,
        "performance_lock_finalized": False,
        "performance_lock_qualified": False,
        "candidate_finalized_no_go": False,
        "one_shot_lock_consumed": False,
        "rerun_authorized": False,
        "reseed_authorized": False,
        "quality_pilot_authorized": False,
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
    return validate_candidate02_performance_lock_v4_value(
        {**body, "summary_sha256": canonical_sha256(body)}, replay_sources=False
    )


def _source_paths(generic: Mapping[str, Any], role: str) -> list[Path]:
    return performance_v2._input_paths(generic, role)


def validate_candidate02_performance_lock_v4_value(
    value: Mapping[str, Any], *, replay_sources: bool = True
) -> dict[str, Any]:
    """Validate a stored summary and optionally replay every source DONE file."""

    if not isinstance(value, Mapping):
        raise ValueError("performance-lock-v4 merge must be an object")
    summary = deepcopy(dict(value))
    if set(summary) != _SUMMARY_KEYS:
        raise ValueError("performance-lock-v4 merge fields changed")
    digest = summary.pop("summary_sha256", None)
    if digest != canonical_sha256(summary):
        raise ValueError("performance-lock-v4 merge digest changed")
    summary["summary_sha256"] = digest
    plan, materialization, seal = _validate_lineage(
        plan_value=_mapping(summary.get("performance_lock_plan"), "stored plan"),
        materialization_value=_mapping(
            summary.get("materialization_receipt"), "stored materialization"
        ),
        root_seal_value=_mapping(summary.get("root_seal"), "stored root seal"),
    )
    generic = _mapping(summary.get("generic_merge"), "stored generic merge")
    gate = lock_gate.validate_performance_lock_v4_gate_value(
        _mapping(summary.get("performance_gate"), "stored performance gate"),
        generic=generic,
        run_contract=plan["run_contract"],
    )
    passed = gate["all_gates_passed"] is True
    boundary_false = (
        "rerun_authorized",
        "reseed_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    )
    if (
        summary.get("schema") != MERGE_SCHEMA
        or summary.get("status") != ("pass" if passed else "no_go")
        or summary.get("decision")
        != (lock_gate.PASS_DECISION if passed else lock_gate.NO_GO_DECISION)
        or summary.get("scope") != SCOPE
        or summary.get("performance_lock_plan_sha256") != canonical_sha256(plan)
        or summary.get("materialization_receipt_sha256")
        != canonical_sha256(materialization)
        or summary.get("root_seal_sha256") != canonical_sha256(seal)
        or summary.get("run009_scientific_gate_binding")
        != plan["run009_scientific_gate"]
        or summary.get("contract_variant")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT
        or summary.get("run_contract") != plan["run_contract"]
        or summary.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
        or summary.get("generic_merge_sha256") != canonical_sha256(generic)
        or summary.get("performance_gate_sha256") != canonical_sha256(gate)
        or summary.get("all_gates_passed") is not passed
        or summary.get("scientific_performance_gate_passed") is not passed
        or summary.get("transport_lineage_required") is not True
        or summary.get("transport_lineage_validated") is not False
        or summary.get("performance_lock_finalized") is not False
        or summary.get("performance_lock_qualified") is not False
        or summary.get("candidate_finalized_no_go") is not False
        or summary.get("one_shot_lock_consumed") is not False
        or summary.get("quality_pilot_authorized") is not False
        or any(summary.get(field) is not False for field in boundary_false)
    ):
        raise ValueError("performance-lock-v4 merge boundary changed")
    expected_partitions = _validate_partitions(plan, generic)
    expected_root_lineage = _validate_root_lineage(
        generic=generic, materialization=materialization, seal=seal
    )
    if (
        summary.get("plan_partitions") != expected_partitions
        or summary.get("root_lineage") != expected_root_lineage
    ):
        raise ValueError("performance-lock-v4 derived lineage changed")
    if replay_sources:
        recomputed = merge_candidate02_performance_lock_v4(
            candidate_done_paths=_source_paths(generic, "candidate"),
            reference_done_paths=_source_paths(generic, "reference"),
            plan_value=plan,
            materialization_value=materialization,
            root_seal_value=seal,
        )
        if recomputed != summary:
            raise ValueError("performance-lock-v4 merge differs from source replay")
    return summary


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"performance-lock-v4 output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(
            f"performance-lock-v4 output already exists: {target}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def merge_and_write_candidate02_performance_lock_v4(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    plan_value: Mapping[str, Any],
    materialization_value: Mapping[str, Any],
    root_seal_value: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    summary = merge_candidate02_performance_lock_v4(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        plan_value=plan_value,
        materialization_value=materialization_value,
        root_seal_value=root_seal_value,
    )
    _write_once(output_path, summary)
    stored = json.loads(Path(output_path).read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or canonical_bytes(stored) != Path(output_path).read_bytes():
        raise ValueError("performance-lock-v4 stored output is not canonical")
    return validate_candidate02_performance_lock_v4_value(stored)


__all__ = [
    "MERGE_SCHEMA",
    "SCOPE",
    "canonical_bytes",
    "canonical_sha256",
    "merge_and_write_candidate02_performance_lock_v4",
    "merge_candidate02_performance_lock_v4",
    "validate_candidate02_performance_lock_v4_value",
]
