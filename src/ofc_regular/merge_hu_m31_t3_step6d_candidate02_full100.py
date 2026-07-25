"""Merge and independently validate Candidate02 full-100 performance evidence.

The generic Step 6d merger owns the exact portable-parity and frozen latency
gates.  This wrapper selects the Candidate02 contract explicitly and adds the
scientific lineage that the generic Candidate01 output cannot express:

* the accepted tail-v2 qualification;
* the frozen topology-only 20-job full100 plan;
* the exact ten shard partitions for each isolated source role; and
* the frozen all-100 root-set digest.

A passing result freezes only the performance candidate and opens the separate
one-shot performance-lock stage.  It cannot authorize quality, training,
runtime activation, a named profile, or ``current``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_full100_plan as full_plan
from . import merge_hu_m31_t3_step6d_performance_v2 as base
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as selector


MERGE_SCHEMA = "hu_m31_t3_step6d_candidate02_full100_performance_merge_v1"
VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_full100_performance_merge_validation_v1"
)
FULL_SCOPE = base.FULL_PERFORMANCE_SCOPE
DEFAULT_PLAN_PATH = (
    full_plan._REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "full100_plan_v1.json"
)

_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "candidate_variant",
        "full100_plan",
        "full100_plan_sha256",
        "tail_qualification",
        "root_set",
        "run_contract",
        "run_contract_digest",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "budget",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "source_done_inputs",
        "paired_artifacts",
        "integrity",
        "performance",
        "gate_mode",
        "gates",
        "all_gates_passed",
        "performance_candidate_frozen",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_authorized",
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
        "candidate_variant",
        "summary_sha256",
        "full100_plan_sha256",
        "tail_summary_sha256",
        "tail_validation_sha256",
        "all100_root_sha256",
        "run_contract_digest",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "source_shard_count",
        "integrity_recomputed_from_source_artifacts",
        "gates",
        "all_gates_passed",
        "performance_candidate_frozen",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)


def _read_plan(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    value = base._read_canonical(target, "Candidate02 full100 plan")
    return full_plan.validate_full100_plan(value)


def _validated_plan(value: Mapping[str, Any] | None, path: Path) -> dict[str, Any]:
    if value is None:
        plan = _read_plan(path)
    else:
        plan = full_plan.validate_full100_plan(value)
    if full_plan.canonical_sha256(plan) != full_plan.FULL100_PLAN_SHA256:
        raise ValueError("Candidate02 full100 plan digest changed")
    return plan


def _expected_partitions(
    plan: Mapping[str, Any], role: str
) -> dict[tuple[int, ...], str]:
    result: dict[tuple[int, ...], str] = {}
    for raw in plan["jobs"]:
        if raw["source_role"] != role:
            continue
        work = tuple(raw["work_hand_indices"])
        if work in result:
            raise ValueError("Candidate02 full100 plan repeats a source partition")
        result[work] = str(raw["shard_manifest_sha256"])
    if len(result) != full_plan.SHARD_COUNT_PER_ROLE:
        raise ValueError("Candidate02 full100 plan source partition count changed")
    return result


def _validate_source_partitions(
    *,
    plan: Mapping[str, Any],
    source_done_inputs: Mapping[str, Any],
) -> None:
    for role in runner.SOURCE_ROLES:
        rows = source_done_inputs.get(role)
        if not isinstance(rows, list):
            raise ValueError("Candidate02 full100 source DONE inputs are missing")
        observed: dict[tuple[int, ...], str] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("Candidate02 full100 source DONE input changed")
            work = tuple(raw.get("work_hand_indices", ()))
            digest = raw.get("shard_manifest_digest")
            if work in observed or not isinstance(digest, str):
                raise ValueError("Candidate02 full100 source partition is duplicated")
            observed[work] = digest
        if observed != _expected_partitions(plan, role):
            raise ValueError(
                f"Candidate02 full100 {role} shard partition/manifest changed"
            )


def _root_set_digest(paired_artifacts: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(paired_artifacts, key=lambda row: int(row["hand_index"]))
    if [int(row["hand_index"]) for row in ordered] != list(
        runner.CONTRACT_HAND_INDICES
    ):
        raise ValueError("Candidate02 full100 paired root coverage changed")
    root_hashes = [str(row["root_file_sha256"]) for row in ordered]
    digest = selector.canonical_sha256(root_hashes)
    if digest != selector.ALL100_ROOT_SHA256:
        raise ValueError("Candidate02 full100 source root set changed")
    return digest


def merge_candidate02_full100(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    plan_path: Path = DEFAULT_PLAN_PATH,
    plan_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the deterministic Candidate02 full100 performance summary."""

    plan = _validated_plan(plan_value, plan_path)
    generic = base.merge_performance_v2(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        scope=FULL_SCOPE,
        contract_variant=runner.CANDIDATE02_VARIANT,
    )
    if (
        generic["scope"] != FULL_SCOPE
        or generic["run_contract_digest"] != full_plan.FULL_RUN_CONTRACT_DIGEST
        or generic["candidate_library_sha256"] != full_plan.CANDIDATE_LIBRARY_SHA256
        or generic["reference_library_sha256"] != full_plan.REFERENCE_LIBRARY_SHA256
        or generic["hand_indices"] != list(runner.CONTRACT_HAND_INDICES)
        or generic["paired_hand_count"] != 100
        or generic["root_count"] != 200
    ):
        raise ValueError("Candidate02 full100 merge identity changed")
    _validate_source_partitions(
        plan=plan,
        source_done_inputs=generic["source_done_inputs"],
    )
    root_digest = _root_set_digest(generic["paired_artifacts"])
    if plan["root_set"]["all100_root_sha256"] != root_digest:
        raise ValueError("Candidate02 full100 plan/source root digest mismatch")

    full_pass = generic["all_gates_passed"] is True
    summary = {
        key: value
        for key, value in generic.items()
        if key not in {"schema", "candidate01_tail_qualified"}
    }
    summary.update(
        {
            "schema": MERGE_SCHEMA,
            "candidate_variant": runner.CANDIDATE02_VARIANT,
            "full100_plan": plan,
            "full100_plan_sha256": full_plan.FULL100_PLAN_SHA256,
            "tail_qualification": dict(plan["tail_qualification"]),
            "root_set": dict(plan["root_set"]),
            "performance_candidate_frozen": full_pass,
            "performance_lock_authorized": full_pass,
            "quality_pilot_authorized": False,
            "artifact_fanout_authorized": False,
            "training_authorized": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
        }
    )
    if set(summary) != _SUMMARY_KEYS:
        raise AssertionError("Candidate02 full100 merge schema implementation changed")
    return summary


def _validation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "status": summary["status"],
        "decision": summary["decision"],
        "scope": summary["scope"],
        "candidate_variant": runner.CANDIDATE02_VARIANT,
        "summary_sha256": runner.canonical_sha256(summary),
        "full100_plan_sha256": summary["full100_plan_sha256"],
        "tail_summary_sha256": summary["tail_qualification"]["summary_sha256"],
        "tail_validation_sha256": summary["tail_qualification"]["validation_sha256"],
        "all100_root_sha256": summary["root_set"]["all100_root_sha256"],
        "run_contract_digest": summary["run_contract_digest"],
        "hand_indices": list(summary["hand_indices"]),
        "paired_hand_count": summary["paired_hand_count"],
        "root_count": summary["root_count"],
        "source_shard_count": sum(
            len(summary["source_done_inputs"][role]) for role in runner.SOURCE_ROLES
        ),
        "integrity_recomputed_from_source_artifacts": True,
        "gates": dict(summary["gates"]),
        "all_gates_passed": summary["all_gates_passed"],
        "performance_candidate_frozen": summary["performance_candidate_frozen"],
        "performance_lock_authorized": summary["performance_lock_authorized"],
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def validate_candidate02_full100_merge(
    *,
    summary_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = base._read_canonical(summary_path, "Candidate02 full100 merge summary")
    if (
        set(summary) != _SUMMARY_KEYS
        or summary.get("schema") != MERGE_SCHEMA
        or summary.get("scope") != FULL_SCOPE
        or summary.get("candidate_variant") != runner.CANDIDATE02_VARIANT
    ):
        raise ValueError("Candidate02 full100 merge summary schema changed")
    expected = merge_candidate02_full100(
        candidate_done_paths=base._input_paths(summary, "candidate"),
        reference_done_paths=base._input_paths(summary, "reference"),
        plan_value=summary["full100_plan"],
    )
    if summary != expected:
        raise ValueError("Candidate02 full100 merge aggregate/tamper mismatch")
    report = _validation_report(summary)
    if set(report) != _VALIDATION_KEYS:
        raise AssertionError("Candidate02 full100 validation schema changed")
    if output_path is not None:
        base._write_once(Path(output_path).resolve(), report)
    return report


def merge_and_validate_candidate02_full100(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    summary_output_path: Path,
    validation_output_path: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    if summary_output == validation_output:
        raise ValueError("Candidate02 full100 summary/validation paths must differ")
    if summary_output.exists() or validation_output.exists():
        raise FileExistsError("Candidate02 full100 merge outputs are write-once")
    summary = merge_candidate02_full100(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
        plan_path=plan_path,
    )
    validation = _validation_report(summary)
    base._write_once(summary_output, summary)
    base._write_once(validation_output, validation)
    return summary, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-done", type=Path, action="append", required=True)
    parser.add_argument("--reference-done", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, validation = merge_and_validate_candidate02_full100(
        candidate_done_paths=args.candidate_done,
        reference_done_paths=args.reference_done,
        plan_path=args.plan,
        summary_output_path=args.summary_output,
        validation_output_path=args.validation_output,
    )
    print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PLAN_PATH",
    "FULL_SCOPE",
    "MERGE_SCHEMA",
    "VALIDATION_SCHEMA",
    "main",
    "merge_and_validate_candidate02_full100",
    "merge_candidate02_full100",
    "validate_candidate02_full100_merge",
]
