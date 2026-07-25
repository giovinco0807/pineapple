"""Bind a validated full100 Spot receive to the Candidate02 scientific merger.

The full100 lifecycle owns package, launch, resume, receipt, and received-file
validation.  The pure merger owns source-artifact parity and performance gates.
This bridge joins those two contracts without weakening either one: it binds
the exact receive directory, run name, and receipt SHA-256 to the scientific
summary and writes immutable summary/validation outputs.

A GO result may open only the separate performance-lock stage.  Quality,
training, artifact fanout, profile activation, and ``current`` remain closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_full100_plan as full_plan
from . import hu_m31_t3_step6d_full100_spot_v1 as lifecycle
from . import merge_hu_m31_t3_step6d_candidate02_full100 as scientific
from . import merge_hu_m31_t3_step6d_performance_v2 as base
from . import run_hu_m31_t3_step6d_performance_v2 as runner


MERGE_SCHEMA = "hu_m31_t3_step6d_full100_received_merge_v1"
VALIDATION_SCHEMA = "hu_m31_t3_step6d_full100_received_merge_validation_v1"
RECEIVED_DIRECTORY_VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_full100_received_directory_validation_v1"
)

_RECEIVED_VALIDATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "receive_receipt_sha256",
        "run_contract_digest",
        "candidate_done_paths",
        "reference_done_paths",
        "paired_hand_count",
        "root_count",
        "source_isolation_validated",
        "root_pairing_validated",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "run_name",
        "receive_dir",
        "receive_receipt_sha256",
        "received_directory_validation",
        "received_directory_validation_sha256",
        "scientific_merge",
        "scientific_merge_sha256",
        "all_gates_passed",
        "merge_executed",
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
        "run_name",
        "summary_sha256",
        "receive_dir",
        "receive_receipt_sha256",
        "received_directory_validation_sha256",
        "scientific_merge_sha256",
        "run_contract_digest",
        "paired_hand_count",
        "root_count",
        "source_shard_count",
        "receipt_and_sources_recomputed",
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


def _validated_done_paths(
    *,
    receive_root: Path,
    validation: Mapping[str, Any],
    role: str,
) -> tuple[Path, ...]:
    raw_paths = validation.get(f"{role}_done_paths")
    if (
        not isinstance(raw_paths, list)
        or len(raw_paths) != full_plan.SHARD_COUNT_PER_ROLE
    ):
        raise ValueError(f"full100 received {role} DONE coverage changed")
    paths: list[Path] = []
    for raw in raw_paths:
        if not isinstance(raw, str):
            raise ValueError(f"full100 received {role} DONE path changed")
        supplied = Path(raw)
        if not supplied.is_absolute() or supplied.is_symlink():
            raise ValueError(f"full100 received {role} DONE path is not absolute/safe")
        resolved = supplied.resolve()
        try:
            resolved.relative_to(receive_root)
        except ValueError as error:
            raise ValueError(
                f"full100 received {role} DONE path escapes receive directory"
            ) from error
        if resolved.name != "DONE.json" or not resolved.is_file():
            raise ValueError(f"full100 received {role} DONE path is missing")
        paths.append(resolved)
    if len(set(paths)) != len(paths):
        raise ValueError(f"full100 received {role} DONE paths are duplicated")
    return tuple(paths)


def _validate_receive(
    receive_dir: str | Path,
    *,
    expected_run_name: str,
) -> tuple[Path, dict[str, Any], tuple[Path, ...], tuple[Path, ...]]:
    original = Path(receive_dir)
    if original.is_symlink():
        raise ValueError("full100 receive directory must not be a symlink")
    receive_root = original.resolve()
    if not receive_root.is_dir():
        raise ValueError(f"full100 receive directory is missing: {receive_root}")
    validation = lifecycle.validate_received_directory(
        receive_root, expected_run_name=expected_run_name
    )
    if (
        not isinstance(validation, dict)
        or set(validation) != _RECEIVED_VALIDATION_KEYS
        or validation.get("schema") != RECEIVED_DIRECTORY_VALIDATION_SCHEMA
        or validation.get("status") != "exact_full100_receive_directory_revalidated"
        or validation.get("run_name") != expected_run_name
        or validation.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or validation.get("paired_hand_count") != 100
        or validation.get("root_count") != 200
        or validation.get("source_isolation_validated") is not True
        or validation.get("root_pairing_validated") is not True
        or any(
            validation.get(field) is not False
            for field in (
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("full100 received-directory validation boundary changed")
    receipt_path = receive_root / "receive_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("full100 receive receipt is missing or unsafe")
    receipt_sha256 = lifecycle.sha256_file(receipt_path)
    if validation.get("receive_receipt_sha256") != receipt_sha256:
        raise ValueError("full100 receive receipt SHA-256 binding changed")
    candidate = _validated_done_paths(
        receive_root=receive_root, validation=validation, role="candidate"
    )
    reference = _validated_done_paths(
        receive_root=receive_root, validation=validation, role="reference"
    )
    if set(candidate) & set(reference):
        raise ValueError("full100 candidate/reference DONE paths overlap")
    return receive_root, validation, candidate, reference


def _summary_done_paths(summary: Mapping[str, Any], role: str) -> set[Path]:
    source_inputs = summary.get("source_done_inputs")
    if not isinstance(source_inputs, Mapping):
        raise ValueError("full100 scientific source inputs changed")
    rows = source_inputs.get(role)
    if not isinstance(rows, list):
        raise ValueError("full100 scientific source inputs changed")
    paths: set[Path] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ValueError("full100 scientific DONE input changed")
        path = Path(str(raw["path"])).resolve()
        if path in paths:
            raise ValueError("full100 scientific DONE input duplicated")
        paths.add(path)
    return paths


def _validate_scientific_summary(
    value: Mapping[str, Any],
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
) -> dict[str, Any]:
    summary = dict(value)
    all_gates = summary.get("all_gates_passed")
    if (
        summary.get("schema") != scientific.MERGE_SCHEMA
        or summary.get("scope") != scientific.FULL_SCOPE
        or summary.get("candidate_variant") != runner.CANDIDATE02_VARIANT
        or summary.get("run_contract_digest") != full_plan.FULL_RUN_CONTRACT_DIGEST
        or summary.get("paired_hand_count") != 100
        or summary.get("root_count") != 200
        or not isinstance(all_gates, bool)
        or summary.get("status") != ("pass" if all_gates else "no_go")
        or summary.get("performance_candidate_frozen") is not all_gates
        or summary.get("performance_lock_authorized") is not all_gates
        or any(
            summary.get(field) is not False
            for field in (
                "quality_pilot_authorized",
                "artifact_fanout_authorized",
                "training_authorized",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "m31_complete",
            )
        )
        or _summary_done_paths(summary, "candidate")
        != {Path(path).resolve() for path in candidate_done_paths}
        or _summary_done_paths(summary, "reference")
        != {Path(path).resolve() for path in reference_done_paths}
    ):
        raise ValueError("full100 scientific merge authorization boundary changed")
    return summary


def _build_summary(
    *,
    receive_dir: str | Path,
    expected_run_name: str,
    plan_path: Path,
    plan_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receive_root, received, candidate, reference = _validate_receive(
        receive_dir, expected_run_name=expected_run_name
    )
    if plan_value is None:
        merged = scientific.merge_candidate02_full100(
            candidate_done_paths=candidate,
            reference_done_paths=reference,
            plan_path=plan_path,
        )
    else:
        merged = scientific.merge_candidate02_full100(
            candidate_done_paths=candidate,
            reference_done_paths=reference,
            plan_value=plan_value,
        )
    scientific_summary = _validate_scientific_summary(
        merged,
        candidate_done_paths=candidate,
        reference_done_paths=reference,
    )
    all_gates = scientific_summary["all_gates_passed"] is True
    summary = {
        "schema": MERGE_SCHEMA,
        "status": "pass" if all_gates else "no_go",
        "decision": (
            "full100_receive_and_performance_go_open_performance_lock_only"
            if all_gates
            else "full100_receive_valid_performance_no_go_lock_not_authorized"
        ),
        "run_name": expected_run_name,
        "receive_dir": str(receive_root),
        "receive_receipt_sha256": received["receive_receipt_sha256"],
        "received_directory_validation": received,
        "received_directory_validation_sha256": runner.canonical_sha256(received),
        "scientific_merge": scientific_summary,
        "scientific_merge_sha256": runner.canonical_sha256(scientific_summary),
        "all_gates_passed": all_gates,
        "merge_executed": True,
        "performance_candidate_frozen": all_gates,
        "performance_lock_authorized": all_gates,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    if set(summary) != _SUMMARY_KEYS:
        raise AssertionError("full100 received merge schema implementation changed")
    return summary


def merge_received_full100(
    *,
    receive_dir: str | Path,
    expected_run_name: str,
    plan_path: Path = scientific.DEFAULT_PLAN_PATH,
) -> dict[str, Any]:
    """Revalidate a full100 receive and return its bound scientific summary."""

    return _build_summary(
        receive_dir=receive_dir,
        expected_run_name=expected_run_name,
        plan_path=Path(plan_path).resolve(),
    )


def _validation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    received = summary["received_directory_validation"]
    scientific_summary = summary["scientific_merge"]
    return {
        "schema": VALIDATION_SCHEMA,
        "status": summary["status"],
        "decision": summary["decision"],
        "run_name": summary["run_name"],
        "summary_sha256": runner.canonical_sha256(summary),
        "receive_dir": summary["receive_dir"],
        "receive_receipt_sha256": summary["receive_receipt_sha256"],
        "received_directory_validation_sha256": summary[
            "received_directory_validation_sha256"
        ],
        "scientific_merge_sha256": summary["scientific_merge_sha256"],
        "run_contract_digest": received["run_contract_digest"],
        "paired_hand_count": received["paired_hand_count"],
        "root_count": received["root_count"],
        "source_shard_count": sum(
            len(scientific_summary["source_done_inputs"][role])
            for role in runner.SOURCE_ROLES
        ),
        "receipt_and_sources_recomputed": True,
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


def validate_received_full100_merge(
    *,
    summary_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = base._read_canonical(summary_path, "full100 received merge summary")
    if (
        set(summary) != _SUMMARY_KEYS
        or summary.get("schema") != MERGE_SCHEMA
        or not isinstance(summary.get("run_name"), str)
        or not isinstance(summary.get("receive_dir"), str)
        or not isinstance(summary.get("scientific_merge"), Mapping)
    ):
        raise ValueError("full100 received merge summary schema changed")
    expected = _build_summary(
        receive_dir=summary["receive_dir"],
        expected_run_name=summary["run_name"],
        plan_path=scientific.DEFAULT_PLAN_PATH,
        plan_value=summary["scientific_merge"].get("full100_plan"),
    )
    if summary != expected:
        raise ValueError("full100 received merge aggregate/tamper mismatch")
    report = _validation_report(summary)
    if set(report) != _VALIDATION_KEYS:
        raise AssertionError("full100 received validation schema changed")
    if output_path is not None:
        base._write_once(Path(output_path).resolve(), report)
    return report


def merge_and_validate_received_full100(
    *,
    receive_dir: str | Path,
    expected_run_name: str,
    summary_output_path: Path,
    validation_output_path: Path,
    plan_path: Path = scientific.DEFAULT_PLAN_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    receive_root = Path(receive_dir).resolve()
    if summary_output == validation_output:
        raise ValueError("full100 received summary/validation paths must differ")
    for output in (summary_output, validation_output):
        if output == receive_root or receive_root in output.parents:
            raise ValueError("full100 merge outputs must be outside the receive tree")
    if summary_output.exists() or validation_output.exists():
        raise FileExistsError("full100 received merge outputs are write-once")
    summary = merge_received_full100(
        receive_dir=receive_root,
        expected_run_name=expected_run_name,
        plan_path=plan_path,
    )
    validation = _validation_report(summary)
    if set(validation) != _VALIDATION_KEYS:
        raise AssertionError("full100 received validation schema changed")
    base._write_once(summary_output, summary)
    base._write_once(validation_output, validation)
    return summary, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receive-dir", type=Path, required=True)
    parser.add_argument("--expected-run-name", required=True)
    parser.add_argument("--plan", type=Path, default=scientific.DEFAULT_PLAN_PATH)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, validation = merge_and_validate_received_full100(
        receive_dir=args.receive_dir,
        expected_run_name=args.expected_run_name,
        plan_path=args.plan,
        summary_output_path=args.summary_output,
        validation_output_path=args.validation_output,
    )
    print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MERGE_SCHEMA",
    "RECEIVED_DIRECTORY_VALIDATION_SCHEMA",
    "VALIDATION_SCHEMA",
    "main",
    "merge_and_validate_received_full100",
    "merge_received_full100",
    "validate_received_full100_merge",
]
