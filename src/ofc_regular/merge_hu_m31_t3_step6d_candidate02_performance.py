"""Merge the fresh candidate02 ten-hand Step 6d performance tail.

This is deliberately a separate scientific output contract from the immutable
candidate01 merger.  It reuses the independently validated source-artifact and
portable-parity primitives, but accepts only the candidate02 run-contract
schema and can authorize only the repeatable 100-hand performance-development
run.  It cannot open performance-lock, quality, training, or runtime activation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import merge_hu_m31_t3_step6d_performance_v2 as base
from . import run_hu_m31_t3_step6d_performance_v2 as runner


MERGE_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_merge_v1"
VALIDATION_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_merge_validation_v1"
TAIL_DIAGNOSTIC_SCOPE = "candidate02_tail_diagnostic"

_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "candidate_variant",
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
        "candidate02_tail_qualified",
        "full_performance_development_authorized",
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
        "run_contract_digest",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "integrity_recomputed_from_source_artifacts",
        "gates",
        "all_gates_passed",
        "candidate02_tail_qualified",
        "full_performance_development_authorized",
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


def _merge_candidate02_tail(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
) -> dict[str, Any]:
    candidate = base._load_role(
        candidate_done_paths,
        "candidate",
        contract_variant=runner.CANDIDATE02_VARIANT,
    )
    reference = base._load_role(
        reference_done_paths,
        "reference",
        contract_variant=runner.CANDIDATE02_VARIANT,
    )
    if (
        candidate.run_contract != reference.run_contract
        or candidate.run_contract_digest != reference.run_contract_digest
    ):
        raise ValueError("candidate02 candidate/reference run contract mismatch")
    candidate_indices = tuple(item.hand_index for item in candidate.hands)
    reference_indices = tuple(item.hand_index for item in reference.hands)
    if (
        candidate_indices != tuple(runner.TAIL_HAND_INDICES)
        or reference_indices != candidate_indices
    ):
        raise ValueError("candidate02 merge requires the exact frozen ten-hand tail")

    paired_artifacts: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for reference_hand, candidate_hand in zip(
        reference.hands, candidate.hands, strict=True
    ):
        paired, rows = base._pair_hand(
            reference_hand,
            candidate_hand,
            scope=base.TAIL_DIAGNOSTIC_SCOPE,
        )
        for row in rows:
            fingerprint = str(row["observation_fingerprint"])
            if fingerprint in fingerprints:
                raise ValueError("duplicate candidate02 observation fingerprint")
            fingerprints.add(fingerprint)
        paired_artifacts.append(paired)
        paired_rows.extend(rows)

    performance = base._performance(paired_rows, paired_artifacts, reference, candidate)
    _old_mode, gates, all_gates = base._build_gates(
        scope=base.TAIL_DIAGNOSTIC_SCOPE,
        hand_indices=candidate_indices,
        paired_artifacts=paired_artifacts,
        paired_rows=paired_rows,
        performance=performance,
    )
    status = "pass" if all_gates else "no_go"
    decision = (
        "candidate02_tail_qualification_pass_open_full_performance_development_only"
        if all_gates
        else "candidate02_tail_no_go_full_performance_development_not_authorized"
    )
    integrity = {
        "candidate_hand_count": len(candidate.hands),
        "reference_hand_count": len(reference.hands),
        "paired_hand_count": len(paired_artifacts),
        "paired_root_count": len(paired_rows),
        "unique_observation_fingerprint_count": len(fingerprints),
        "paired_hand_parity_count": sum(
            item["paired_seat_parity_count"] == 2 for item in paired_artifacts
        ),
        "paired_root_parity_count": sum(
            row["portable_parity"]["portable_payload_exact"] is True
            for row in paired_rows
        ),
        "missing_hand_indices": [],
        "duplicate_hand_indices": [],
        "out_of_contract_hand_indices": [],
    }
    return {
        "schema": MERGE_SCHEMA,
        "status": status,
        "decision": decision,
        "scope": TAIL_DIAGNOSTIC_SCOPE,
        "candidate_variant": runner.CANDIDATE02_VARIANT,
        "run_contract": candidate.run_contract,
        "run_contract_digest": candidate.run_contract_digest,
        "hand_indices": list(candidate_indices),
        "paired_hand_count": len(paired_artifacts),
        "root_count": len(paired_rows),
        "budget": dict(candidate.run_contract["budget"]),
        "allocation": dict(candidate.run_contract["allocation"]),
        "reference_library_sha256": candidate.run_contract["reference_library_sha256"],
        "candidate_library_sha256": candidate.run_contract["candidate_library_sha256"],
        "source_done_inputs": {
            "candidate": list(candidate.done_inputs),
            "reference": list(reference.done_inputs),
        },
        "paired_artifacts": paired_artifacts,
        "integrity": integrity,
        "performance": performance,
        "gate_mode": "tail_candidate02_qualification_only",
        "gates": gates,
        "all_gates_passed": all_gates,
        "candidate02_tail_qualified": all_gates,
        "full_performance_development_authorized": all_gates,
        "performance_candidate_frozen": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def _validation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "status": summary["status"],
        "decision": summary["decision"],
        "scope": summary["scope"],
        "candidate_variant": runner.CANDIDATE02_VARIANT,
        "summary_sha256": runner.canonical_sha256(summary),
        "run_contract_digest": summary["run_contract_digest"],
        "hand_indices": list(summary["hand_indices"]),
        "paired_hand_count": summary["paired_hand_count"],
        "root_count": summary["root_count"],
        "integrity_recomputed_from_source_artifacts": True,
        "gates": dict(summary["gates"]),
        "all_gates_passed": summary["all_gates_passed"],
        "candidate02_tail_qualified": summary["candidate02_tail_qualified"],
        "full_performance_development_authorized": summary[
            "full_performance_development_authorized"
        ],
        "performance_candidate_frozen": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def merge_candidate02_performance(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
) -> dict[str, Any]:
    """Return the deterministic candidate02 tail summary."""

    return _merge_candidate02_tail(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
    )


def validate_candidate02_performance_merge(
    *, summary_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = base._read_canonical(summary_path, "candidate02 merge summary")
    if set(summary) != _SUMMARY_KEYS or summary.get("schema") != MERGE_SCHEMA:
        raise ValueError("candidate02 merge summary schema changed")
    if (
        summary.get("scope") != TAIL_DIAGNOSTIC_SCOPE
        or summary.get("candidate_variant") != runner.CANDIDATE02_VARIANT
    ):
        raise ValueError("candidate02 merge summary scope changed")
    expected = _merge_candidate02_tail(
        candidate_done_paths=base._input_paths(summary, "candidate"),
        reference_done_paths=base._input_paths(summary, "reference"),
    )
    if summary != expected:
        raise ValueError("candidate02 merge aggregate/tamper mismatch")
    report = _validation_report(summary)
    if set(report) != _VALIDATION_KEYS:
        raise AssertionError("candidate02 validation schema implementation changed")
    if output_path is not None:
        base._write_once(Path(output_path).resolve(), report)
    return report


def merge_and_validate_candidate02_performance(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    summary_output_path: Path,
    validation_output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    if summary_output == validation_output:
        raise ValueError("candidate02 summary and validation outputs must differ")
    if summary_output.exists() or validation_output.exists():
        raise FileExistsError("candidate02 merge outputs are write-once")
    summary = _merge_candidate02_tail(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
    )
    validation = _validation_report(summary)
    base._write_once(summary_output, summary)
    base._write_once(validation_output, validation)
    return summary, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-done", type=Path, action="append", required=True)
    parser.add_argument("--reference-done", type=Path, action="append", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, validation = merge_and_validate_candidate02_performance(
        candidate_done_paths=args.candidate_done,
        reference_done_paths=args.reference_done,
        summary_output_path=args.summary_output,
        validation_output_path=args.validation_output,
    )
    print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MERGE_SCHEMA",
    "TAIL_DIAGNOSTIC_SCOPE",
    "VALIDATION_SCHEMA",
    "main",
    "merge_and_validate_candidate02_performance",
    "merge_candidate02_performance",
    "validate_candidate02_performance_merge",
]
