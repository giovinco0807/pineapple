"""Freeze the local Candidate02 full-100 performance-development job plan.

This module is deliberately one step short of a cloud lifecycle.  It binds the
accepted Candidate02 tail-v2 qualification, the already-frozen 100 Candidate02
roots, the existing source-isolated runner contract, and a deterministic
20-job shard plan.  It does not package, authorize, launch, receive, merge,
train, promote, or resolve ``current``.

The ten work shards are assigned from public root topology only.  Each shard
contains exactly two hands from each behavior profile.  Within each profile,
roots are processed in descending first-seat action-matrix-cell order and
placed on the currently lightest eligible shard, with shard index as the
deterministic tie-break.  Candidate and reference use the same ten partitions
but remain separate source processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import merge_hu_m31_t3_step6d_candidate02_tail_v2 as tail_merger
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as selector
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


PLAN_SCHEMA = "hu_m31_t3_step6d_candidate02_full100_plan_v1"
PLAN_SCOPE = "full_performance_development"
PLAN_STATUS = "local_full100_plan_ready_cloud_not_authorized"
PLAN_DECISION = "tail_v2_go_freezes_candidate02_full100_source_isolated_shards"

CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
TAIL_SUMMARY_SHA256 = "1eb098073ed51efc6868771ac04ecbde965a91dd59bc7a02da547b5555f8253b"
TAIL_VALIDATION_SHA256 = (
    "26a27c730399cb4adbf612d4e510b21462c29078e75dbe427e431115239e5011"
)
TAIL_RUN_CONTRACT_DIGEST = (
    "b5d3114d0857723809ec85cef957921acafa67e22756aef050fa1c8ef8f79bf6"
)
FULL_RUN_CONTRACT_DIGEST = (
    "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
)
FULL100_PLAN_SHA256 = "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758"

SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
SHARD_COUNT_PER_ROLE = 10
HANDS_PER_SHARD = 10
HANDS_PER_PROFILE_PER_SHARD = 2
LOGICAL_JOB_COUNT = len(SOURCE_ROLES) * SHARD_COUNT_PER_ROLE
ASSIGNMENT_METHOD = "profile_stratified_lpt_first_action_matrix_cells_v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT_DIR = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "tail_reselection_v2/roots"
)
DEFAULT_TAIL_MERGE_DIR = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/spot_v2_merge/"
    "regular-hu-m31-c02-tail-v2-20260717-001"
)

_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "tail_qualification",
        "candidate_variant",
        "run_contract",
        "run_contract_digest",
        "root_set",
        "assignment",
        "source_roles",
        "shard_count_per_role",
        "hands_per_shard",
        "logical_job_count",
        "shards",
        "jobs",
        "spot_package_authorized",
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
        "run_contract_digest",
        "candidate_variant",
        "hand_indices",
        "all_gates_passed",
        "full_performance_development_authorized",
    }
)
_ROOT_SET_KEYS = frozenset(
    {
        "schema",
        "hand_indices",
        "root_count",
        "all100_root_sha256",
        "topology_sha256",
    }
)
_ASSIGNMENT_KEYS = frozenset(
    {
        "method",
        "input_scope",
        "weight",
        "tie_break",
        "timing_used",
        "memory_used",
        "teacher_values_used",
        "runtime_results_used",
    }
)
_SHARD_KEYS = frozenset(
    {
        "shard_index",
        "work_hand_indices",
        "profile_counts",
        "topology_weight",
    }
)
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
        raise FileExistsError(f"refusing to overwrite full100 plan: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite full100 plan: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_tail_qualification(
    *,
    summary_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Recompute and bind the official tail-v2 Go before planning full100."""

    summary_target = Path(summary_path).resolve()
    validation_target = Path(validation_path).resolve()
    if (
        sha256_file(summary_target) != TAIL_SUMMARY_SHA256
        or sha256_file(validation_target) != TAIL_VALIDATION_SHA256
    ):
        raise ValueError("Candidate02 tail-v2 qualification artifact hash changed")
    recomputed = tail_merger.validate_candidate02_tail_v2_merge(
        summary_path=summary_target
    )
    stored = _read_canonical(validation_target, "tail-v2 validation")
    if stored != recomputed:
        raise ValueError("Candidate02 tail-v2 validation does not match source replay")
    if (
        stored.get("schema") != tail_merger.VALIDATION_SCHEMA
        or stored.get("status") != "pass"
        or stored.get("decision")
        != (
            "candidate02_tail_v2_qualification_pass_open_"
            "full_performance_development_only"
        )
        or stored.get("candidate_variant") != runner.CANDIDATE02_TAIL_V2_VARIANT
        or stored.get("run_contract_digest") != TAIL_RUN_CONTRACT_DIGEST
        or stored.get("hand_indices")
        != list(runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES)
        or stored.get("all_gates_passed") is not True
        or stored.get("candidate02_tail_v2_qualified") is not True
        or stored.get("full_performance_development_authorized") is not True
        or stored.get("performance_lock_authorized") is not False
        or stored.get("quality_pilot_authorized") is not False
        or stored.get("training_authorized") is not False
        or stored.get("current_profile_changed") is not False
        or stored.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Candidate02 tail-v2 result does not authorize full100")
    return {
        "summary_sha256": TAIL_SUMMARY_SHA256,
        "validation_sha256": TAIL_VALIDATION_SHA256,
        "run_contract_digest": TAIL_RUN_CONTRACT_DIGEST,
        "candidate_variant": runner.CANDIDATE02_TAIL_V2_VARIANT,
        "hand_indices": list(runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES),
        "all_gates_passed": True,
        "full_performance_development_authorized": True,
    }


def _assign_shards(
    topology: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(topology) != len(runner.CONTRACT_HAND_INDICES):
        raise ValueError("Candidate02 full100 topology must contain exactly 100 hands")
    by_hand = {int(row["hand_index"]): dict(row) for row in topology}
    if tuple(sorted(by_hand)) != tuple(runner.CONTRACT_HAND_INDICES):
        raise ValueError("Candidate02 full100 topology coverage changed")

    shards = [
        {
            "shard_index": index,
            "work_hand_indices": [],
            "profile_counts": {profile: 0 for profile in M31_T3_BEHAVIOR_PROFILES},
            "topology_weight": 0,
        }
        for index in range(SHARD_COUNT_PER_ROLE)
    ]
    for profile in M31_T3_BEHAVIOR_PROFILES:
        profile_rows = [row for row in by_hand.values() if row["profile"] == profile]
        if len(profile_rows) != 20:
            raise ValueError("Candidate02 full100 profile coverage changed")
        ordered = sorted(
            profile_rows,
            key=lambda row: (
                -int(row["hero_legal_actions"])
                * int(row["opponent_response_legal_actions"]),
                int(row["hand_index"]),
            ),
        )
        for row in ordered:
            eligible = [
                shard
                for shard in shards
                if shard["profile_counts"][profile] < HANDS_PER_PROFILE_PER_SHARD
                and len(shard["work_hand_indices"]) < HANDS_PER_SHARD
            ]
            if not eligible:
                raise AssertionError("Candidate02 full100 shard assignment exhausted")
            selected = min(
                eligible,
                key=lambda shard: (
                    int(shard["topology_weight"]),
                    int(shard["shard_index"]),
                ),
            )
            selected["work_hand_indices"].append(int(row["hand_index"]))
            selected["profile_counts"][profile] += 1
            selected["topology_weight"] += int(row["hero_legal_actions"]) * int(
                row["opponent_response_legal_actions"]
            )

    for shard in shards:
        shard["work_hand_indices"] = sorted(shard["work_hand_indices"])
    return shards


def build_full100_plan(
    *,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    tail_summary_path: str | Path = DEFAULT_TAIL_MERGE_DIR / "summary.json",
    tail_validation_path: str | Path = DEFAULT_TAIL_MERGE_DIR / "validation.json",
) -> dict[str, Any]:
    qualification = load_tail_qualification(
        summary_path=tail_summary_path,
        validation_path=tail_validation_path,
    )
    roots = selector.load_frozen_roots(root_dir)
    topology = selector.topology_rows(roots)
    run_contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_VARIANT,
    )
    run_contract_digest = runner.canonical_sha256(run_contract)
    if run_contract_digest != FULL_RUN_CONTRACT_DIGEST:
        raise ValueError("Candidate02 full100 runner contract digest changed")
    shards = _assign_shards(topology)
    jobs: list[dict[str, Any]] = []
    for role in SOURCE_ROLES:
        for shard in shards:
            work = list(shard["work_hand_indices"])
            manifest = runner.build_shard_manifest(
                run_contract=run_contract,
                source_role=role,
                work_hand_indices=work,
            )
            index = int(shard["shard_index"])
            jobs.append(
                {
                    "job_id": f"{role}-shard-{index:02d}",
                    "source_role": role,
                    "shard_index": index,
                    "work_hand_indices": work,
                    "shard_manifest_sha256": runner.canonical_sha256(manifest),
                }
            )
    value = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "decision": PLAN_DECISION,
        "scope": PLAN_SCOPE,
        "tail_qualification": qualification,
        "candidate_variant": runner.CANDIDATE02_VARIANT,
        "run_contract": run_contract,
        "run_contract_digest": run_contract_digest,
        "root_set": {
            "schema": selector.ROOT_SCHEMA,
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_count": len(roots),
            "all100_root_sha256": selector.ALL100_ROOT_SHA256,
            "topology_sha256": selector.TOPOLOGY_SHA256,
        },
        "assignment": {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "public_root_topology_only",
            "weight": "first_hero_legal_actions_times_opponent_response_legal_actions",
            "tie_break": "ascending_shard_index_then_ascending_hand_index",
            "timing_used": False,
            "memory_used": False,
            "teacher_values_used": False,
            "runtime_results_used": False,
        },
        "source_roles": list(SOURCE_ROLES),
        "shard_count_per_role": SHARD_COUNT_PER_ROLE,
        "hands_per_shard": HANDS_PER_SHARD,
        "logical_job_count": LOGICAL_JOB_COUNT,
        "shards": shards,
        "jobs": jobs,
        "spot_package_authorized": False,
        "cloud_started": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    return validate_full100_plan(value)


def validate_full100_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if set(payload) != _PLAN_KEYS:
        raise ValueError("Candidate02 full100 plan fields changed")
    qualification = payload.get("tail_qualification")
    root_set = payload.get("root_set")
    assignment = payload.get("assignment")
    shards = payload.get("shards")
    jobs = payload.get("jobs")
    if (
        not isinstance(qualification, Mapping)
        or set(qualification) != _QUALIFICATION_KEYS
        or not isinstance(root_set, Mapping)
        or set(root_set) != _ROOT_SET_KEYS
        or not isinstance(assignment, Mapping)
        or set(assignment) != _ASSIGNMENT_KEYS
        or not isinstance(shards, list)
        or not isinstance(jobs, list)
    ):
        raise ValueError("Candidate02 full100 plan nested fields changed")
    contract = runner.validate_run_contract(payload["run_contract"])
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != PLAN_STATUS
        or payload.get("decision") != PLAN_DECISION
        or payload.get("scope") != PLAN_SCOPE
        or qualification
        != {
            "summary_sha256": TAIL_SUMMARY_SHA256,
            "validation_sha256": TAIL_VALIDATION_SHA256,
            "run_contract_digest": TAIL_RUN_CONTRACT_DIGEST,
            "candidate_variant": runner.CANDIDATE02_TAIL_V2_VARIANT,
            "hand_indices": list(runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES),
            "all_gates_passed": True,
            "full_performance_development_authorized": True,
        }
        or payload.get("candidate_variant") != runner.CANDIDATE02_VARIANT
        or runner.contract_variant(contract) != runner.CANDIDATE02_VARIANT
        or contract.get("candidate_library_sha256") != CANDIDATE_LIBRARY_SHA256
        or contract.get("reference_library_sha256") != REFERENCE_LIBRARY_SHA256
        or payload.get("run_contract_digest") != FULL_RUN_CONTRACT_DIGEST
        or runner.canonical_sha256(contract) != FULL_RUN_CONTRACT_DIGEST
        or root_set
        != {
            "schema": selector.ROOT_SCHEMA,
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_count": 100,
            "all100_root_sha256": selector.ALL100_ROOT_SHA256,
            "topology_sha256": selector.TOPOLOGY_SHA256,
        }
        or assignment
        != {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "public_root_topology_only",
            "weight": (
                "first_hero_legal_actions_times_opponent_response_legal_actions"
            ),
            "tie_break": "ascending_shard_index_then_ascending_hand_index",
            "timing_used": False,
            "memory_used": False,
            "teacher_values_used": False,
            "runtime_results_used": False,
        }
        or payload.get("source_roles") != list(SOURCE_ROLES)
        or payload.get("shard_count_per_role") != SHARD_COUNT_PER_ROLE
        or payload.get("hands_per_shard") != HANDS_PER_SHARD
        or payload.get("logical_job_count") != LOGICAL_JOB_COUNT
        or len(shards) != SHARD_COUNT_PER_ROLE
        or len(jobs) != LOGICAL_JOB_COUNT
        or any(
            payload.get(field) is not False
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
        raise ValueError("Candidate02 full100 plan contract changed")

    covered: list[int] = []
    shard_work: dict[int, list[int]] = {}
    for expected_index, raw in enumerate(shards):
        if not isinstance(raw, Mapping) or set(raw) != _SHARD_KEYS:
            raise ValueError("Candidate02 full100 shard fields changed")
        index = raw.get("shard_index")
        work = raw.get("work_hand_indices")
        counts = raw.get("profile_counts")
        weight = raw.get("topology_weight")
        if (
            index != expected_index
            or not isinstance(work, list)
            or len(work) != HANDS_PER_SHARD
            or work != sorted(work)
            or len(set(work)) != len(work)
            or not isinstance(counts, Mapping)
            or dict(counts)
            != {
                profile: HANDS_PER_PROFILE_PER_SHARD
                for profile in M31_T3_BEHAVIOR_PROFILES
            }
            or isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight <= 0
        ):
            raise ValueError("Candidate02 full100 shard boundary changed")
        covered.extend(work)
        shard_work[expected_index] = list(work)
    if sorted(covered) != list(runner.CONTRACT_HAND_INDICES):
        raise ValueError("Candidate02 full100 shard partition is not exact")

    expected_job_ids = [
        f"{role}-shard-{index:02d}"
        for role in SOURCE_ROLES
        for index in range(SHARD_COUNT_PER_ROLE)
    ]
    for raw, expected_id in zip(jobs, expected_job_ids, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _JOB_KEYS:
            raise ValueError("Candidate02 full100 job fields changed")
        role = raw.get("source_role")
        index = raw.get("shard_index")
        if (
            raw.get("job_id") != expected_id
            or role not in SOURCE_ROLES
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index not in shard_work
            or raw.get("work_hand_indices") != shard_work[index]
        ):
            raise ValueError("Candidate02 full100 job mapping changed")
        manifest = runner.build_shard_manifest(
            run_contract=contract,
            source_role=str(role),
            work_hand_indices=shard_work[index],
        )
        if raw.get("shard_manifest_sha256") != runner.canonical_sha256(manifest):
            raise ValueError("Candidate02 full100 shard manifest digest changed")
    if canonical_sha256(payload) != FULL100_PLAN_SHA256:
        raise ValueError("Candidate02 full100 frozen plan digest changed")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--tail-summary",
        type=Path,
        default=DEFAULT_TAIL_MERGE_DIR / "summary.json",
    )
    parser.add_argument(
        "--tail-validation",
        type=Path,
        default=DEFAULT_TAIL_MERGE_DIR / "validation.json",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_full100_plan(
        root_dir=args.root_dir,
        tail_summary_path=args.tail_summary,
        tail_validation_path=args.tail_validation,
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
    "FULL100_PLAN_SHA256",
    "FULL_RUN_CONTRACT_DIGEST",
    "HANDS_PER_SHARD",
    "LOGICAL_JOB_COUNT",
    "PLAN_SCHEMA",
    "REFERENCE_LIBRARY_SHA256",
    "SHARD_COUNT_PER_ROLE",
    "build_full100_plan",
    "canonical_sha256",
    "load_tail_qualification",
    "main",
    "validate_full100_plan",
]
