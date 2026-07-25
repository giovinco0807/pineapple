"""Merge the fresh, non-overlapping candidate02 Step 6d tail-v2 probe.

This contract is intentionally separate from both the immutable candidate01
tail merger and the already-exposed candidate02-v1 tail merger.  It accepts
only the candidate02 tail-v2 runner schema and the exact precommitted ten-hand
selection.  Passing it can authorize only the repeatable 100-hand
performance-development run; it is not performance-lock, quality, training,
or runtime-promotion evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import merge_hu_m31_t3_step6d_performance_v2 as base
from . import run_hu_m31_t3_step6d_performance_v2 as runner


MERGE_SCHEMA = "hu_m31_t3_step6d_candidate02_tail_v2_performance_merge_v2"
VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_tail_v2_performance_merge_validation_v2"
)
TAIL_DIAGNOSTIC_SCOPE = "candidate02_tail_v2_diagnostic"

TAIL_HAND_INDICES = (0, 4, 5, 12, 14, 16, 17, 23, 41, 43)
TAIL_HEAVY_HAND_INDICES = (0, 5, 12, 16, 17, 23, 41, 43)
TAIL_RANDOM_HAND_INDICES = (4, 14)
EXPOSED_CANDIDATE02_V1_HAND_INDICES = (2, 6, 7, 9, 13, 20, 21, 29, 33, 50)
SELECTION_MANIFEST_SHA256 = (
    "62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1"
)
SELECTION_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "hu_joint_policy_m31_t3_candidate02_tail_v2_selection.json"
)

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
        "heavy_hand_indices",
        "random_hand_indices",
        "excluded_candidate02_v1_hand_indices",
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
        "candidate02_tail_v2_qualified",
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
        "heavy_hand_indices",
        "random_hand_indices",
        "paired_hand_count",
        "root_count",
        "integrity_recomputed_from_source_artifacts",
        "gates",
        "all_gates_passed",
        "candidate02_tail_v2_qualified",
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


def _assert_frozen_selection() -> None:
    runner_tail = tuple(runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES)
    manifest = base._read_canonical(
        SELECTION_MANIFEST_PATH, "candidate02 tail-v2 selection manifest"
    )
    if (
        runner_tail != TAIL_HAND_INDICES
        or runner.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
        != SELECTION_MANIFEST_SHA256
        or base._sha256(SELECTION_MANIFEST_PATH) != SELECTION_MANIFEST_SHA256
        or manifest.get("schema")
        != "hu_m31_t3_step6d_candidate02_tail_reselection_manifest_v2"
        or manifest.get("contract_hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or manifest.get("tail_hand_indices") != list(TAIL_HAND_INDICES)
        or manifest.get("heavy_hand_indices") != list(TAIL_HEAVY_HAND_INDICES)
        or manifest.get("random_hand_indices") != list(TAIL_RANDOM_HAND_INDICES)
        or manifest.get("prior_runtime_exposed_hand_indices")
        != list(EXPOSED_CANDIDATE02_V1_HAND_INDICES)
        or any(
            manifest.get(field) is not False
            for field in (
                "runtime_results_used",
                "timing_used",
                "teacher_values_used",
                "memory_used",
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
        or len(set(TAIL_HAND_INDICES)) != 10
        or set(TAIL_HEAVY_HAND_INDICES) | set(TAIL_RANDOM_HAND_INDICES)
        != set(TAIL_HAND_INDICES)
        or set(TAIL_HEAVY_HAND_INDICES) & set(TAIL_RANDOM_HAND_INDICES)
        or set(TAIL_HAND_INDICES) & set(EXPOSED_CANDIDATE02_V1_HAND_INDICES)
    ):
        raise ValueError("candidate02 tail-v2 frozen selection changed")


def _performance(
    paired_rows: Sequence[Mapping[str, Any]],
    paired_artifacts: Sequence[Mapping[str, Any]],
    reference: base._RoleArtifacts,
    candidate: base._RoleArtifacts,
) -> dict[str, Any]:
    by_role = {
        role: {
            seat: base._latencies(
                [
                    float(row[f"{role}_solve_wall_seconds"])
                    for row in paired_rows
                    if row["seat"] == seat
                ]
            )
            for seat in ("first", "second")
        }
        for role in ("reference", "candidate")
    }
    all_hands = [*reference.hands, *candidate.hands]
    peak_rss = max(base._memory_peak(artifact.hand) for artifact in all_hands)
    first_speedups = [
        float(row["paired_speedup"]) for row in paired_rows if row["seat"] == "first"
    ]
    heavy_set = set(TAIL_HEAVY_HAND_INDICES)
    heavy_rows = [
        row
        for artifact in paired_artifacts
        if artifact["hand_index"] in heavy_set
        for row in artifact["rows"]
        if row["seat"] == "first"
    ]
    if len(heavy_rows) != len(TAIL_HEAVY_HAND_INDICES):
        raise ValueError("candidate02 tail-v2 heavy first-root coverage changed")
    heavy = {
        "hand_indices": [
            artifact["hand_index"]
            for artifact in paired_artifacts
            if artifact["hand_index"] in heavy_set
        ],
        "candidate_first": base._latencies(
            [float(row["candidate_solve_wall_seconds"]) for row in heavy_rows]
        ),
        "reference_first": base._latencies(
            [float(row["reference_solve_wall_seconds"]) for row in heavy_rows]
        ),
        "paired_speedups": [float(row["paired_speedup"]) for row in heavy_rows],
        "geometric_mean_speedup": base.geometric_mean(
            [float(row["paired_speedup"]) for row in heavy_rows]
        ),
    }
    return {
        "reference_by_seat": by_role["reference"],
        "candidate_by_seat": by_role["candidate"],
        "first_paired_speedups": first_speedups,
        "first_geometric_mean_speedup_diagnostic": base.geometric_mean(first_speedups),
        "tail_heavy_first": heavy,
        "peak_source_process_rss_bytes": peak_rss,
    }


def _build_gates(
    *,
    hand_indices: Sequence[int],
    paired_artifacts: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
) -> tuple[dict[str, bool], bool]:
    heavy = base._mapping(
        performance["tail_heavy_first"], "candidate02 tail-v2 heavy performance"
    )
    candidate = base._mapping(
        performance["candidate_by_seat"], "candidate02 tail-v2 candidate performance"
    )
    peak = int(performance["peak_source_process_rss_bytes"])
    gates = {
        "exact_precommitted_candidate02_tail_v2_indices": tuple(hand_indices)
        == TAIL_HAND_INDICES,
        "candidate02_v1_exposed_indices_excluded": not (
            set(hand_indices) & set(EXPOSED_CANDIDATE02_V1_HAND_INDICES)
        ),
        "paired_semantic_parity_10_of_10": len(paired_artifacts)
        == base.TAIL_PAIRED_PARITY_REQUIRED
        and len(paired_rows) == 2 * base.TAIL_PAIRED_PARITY_REQUIRED
        and all(item["paired_seat_parity_count"] == 2 for item in paired_artifacts)
        and all(
            row["portable_parity"]["portable_payload_exact"] is True
            for row in paired_rows
        ),
        "heavy_21x21_first_roots_8_of_8": sum(
            item["hand_index"] in TAIL_HEAVY_HAND_INDICES
            and item["heavy_21x21"] is True
            for item in paired_artifacts
        )
        == len(TAIL_HEAVY_HAND_INDICES),
        "peak_rss_within_858993459_bytes": peak <= base.TAIL_MAX_PEAK_RSS_BYTES,
        "heavy_candidate_first_median_within_135_seconds": float(
            heavy["candidate_first"]["median_seconds"]
        )
        <= base.TAIL_HEAVY_FIRST_MEDIAN_SECONDS_MAX,
        "heavy_candidate_first_max_within_145_seconds": float(
            heavy["candidate_first"]["max_seconds"]
        )
        <= base.TAIL_HEAVY_FIRST_MAX_SECONDS_MAX,
        "heavy_geometric_mean_speedup_at_least_1_55": float(
            heavy["geometric_mean_speedup"]
        )
        >= base.TAIL_GEOMETRIC_MEAN_SPEEDUP_MIN,
        "candidate_second_max_within_5_seconds": float(
            candidate["second"]["max_seconds"]
        )
        <= base.TAIL_SECOND_MAX_SECONDS_MAX,
    }
    return gates, all(gates.values())


def _merge_candidate02_tail_v2(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
) -> dict[str, Any]:
    _assert_frozen_selection()
    candidate = base._load_role(
        candidate_done_paths,
        "candidate",
        contract_variant=runner.CANDIDATE02_TAIL_V2_VARIANT,
    )
    reference = base._load_role(
        reference_done_paths,
        "reference",
        contract_variant=runner.CANDIDATE02_TAIL_V2_VARIANT,
    )
    if (
        candidate.run_contract != reference.run_contract
        or candidate.run_contract_digest != reference.run_contract_digest
    ):
        raise ValueError(
            "candidate02 tail-v2 candidate/reference run contract mismatch"
        )
    if (
        candidate.run_contract.get("selection_manifest_sha256")
        != SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("candidate02 tail-v2 selection manifest binding changed")
    candidate_indices = tuple(item.hand_index for item in candidate.hands)
    reference_indices = tuple(item.hand_index for item in reference.hands)
    if candidate_indices != TAIL_HAND_INDICES or reference_indices != candidate_indices:
        raise ValueError("candidate02 tail-v2 requires the exact frozen ten-hand tail")

    paired_artifacts: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for reference_hand, candidate_hand in zip(
        reference.hands, candidate.hands, strict=True
    ):
        # Use a distinct non-base-tail scope so candidate01's exposed heavy set
        # cannot silently define candidate02 tail-v2 semantics.
        paired, rows = base._pair_hand(
            reference_hand,
            candidate_hand,
            scope=TAIL_DIAGNOSTIC_SCOPE,
        )
        if (
            paired["hand_index"] in TAIL_HEAVY_HAND_INDICES
            and paired["heavy_21x21"] is not True
        ):
            raise ValueError(
                f"candidate02 tail-v2 heavy hand {paired['hand_index']} is not 21x21"
            )
        for row in rows:
            fingerprint = str(row["observation_fingerprint"])
            if fingerprint in fingerprints:
                raise ValueError(
                    "duplicate candidate02 tail-v2 observation fingerprint"
                )
            fingerprints.add(fingerprint)
        paired_artifacts.append(paired)
        paired_rows.extend(rows)

    performance = _performance(paired_rows, paired_artifacts, reference, candidate)
    gates, all_gates = _build_gates(
        hand_indices=candidate_indices,
        paired_artifacts=paired_artifacts,
        paired_rows=paired_rows,
        performance=performance,
    )
    status = "pass" if all_gates else "no_go"
    decision = (
        "candidate02_tail_v2_qualification_pass_open_full_performance_development_only"
        if all_gates
        else "candidate02_tail_v2_no_go_full_performance_development_not_authorized"
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
        "heavy_21x21_hand_count": sum(
            item["hand_index"] in TAIL_HEAVY_HAND_INDICES
            and item["heavy_21x21"] is True
            for item in paired_artifacts
        ),
        "candidate02_v1_overlap_hand_indices": [],
    }
    return {
        "schema": MERGE_SCHEMA,
        "status": status,
        "decision": decision,
        "scope": TAIL_DIAGNOSTIC_SCOPE,
        "candidate_variant": runner.CANDIDATE02_TAIL_V2_VARIANT,
        "run_contract": candidate.run_contract,
        "run_contract_digest": candidate.run_contract_digest,
        "hand_indices": list(candidate_indices),
        "heavy_hand_indices": list(TAIL_HEAVY_HAND_INDICES),
        "random_hand_indices": list(TAIL_RANDOM_HAND_INDICES),
        "excluded_candidate02_v1_hand_indices": list(
            EXPOSED_CANDIDATE02_V1_HAND_INDICES
        ),
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
        "gate_mode": "tail_candidate02_v2_qualification_only",
        "gates": gates,
        "all_gates_passed": all_gates,
        "candidate02_tail_v2_qualified": all_gates,
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
        "candidate_variant": runner.CANDIDATE02_TAIL_V2_VARIANT,
        "summary_sha256": runner.canonical_sha256(summary),
        "run_contract_digest": summary["run_contract_digest"],
        "hand_indices": list(summary["hand_indices"]),
        "heavy_hand_indices": list(summary["heavy_hand_indices"]),
        "random_hand_indices": list(summary["random_hand_indices"]),
        "paired_hand_count": summary["paired_hand_count"],
        "root_count": summary["root_count"],
        "integrity_recomputed_from_source_artifacts": True,
        "gates": dict(summary["gates"]),
        "all_gates_passed": summary["all_gates_passed"],
        "candidate02_tail_v2_qualified": summary["candidate02_tail_v2_qualified"],
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


def merge_candidate02_tail_v2(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
) -> dict[str, Any]:
    """Return the deterministic candidate02 tail-v2 summary."""

    return _merge_candidate02_tail_v2(
        candidate_done_paths=candidate_done_paths,
        reference_done_paths=reference_done_paths,
    )


def validate_candidate02_tail_v2_merge(
    *, summary_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = base._read_canonical(summary_path, "candidate02 tail-v2 merge summary")
    if set(summary) != _SUMMARY_KEYS or summary.get("schema") != MERGE_SCHEMA:
        raise ValueError("candidate02 tail-v2 merge summary schema changed")
    if (
        summary.get("scope") != TAIL_DIAGNOSTIC_SCOPE
        or summary.get("candidate_variant") != runner.CANDIDATE02_TAIL_V2_VARIANT
        or summary.get("hand_indices") != list(TAIL_HAND_INDICES)
        or summary.get("heavy_hand_indices") != list(TAIL_HEAVY_HAND_INDICES)
        or summary.get("random_hand_indices") != list(TAIL_RANDOM_HAND_INDICES)
        or summary.get("excluded_candidate02_v1_hand_indices")
        != list(EXPOSED_CANDIDATE02_V1_HAND_INDICES)
    ):
        raise ValueError("candidate02 tail-v2 merge summary selection changed")
    expected = _merge_candidate02_tail_v2(
        candidate_done_paths=base._input_paths(summary, "candidate"),
        reference_done_paths=base._input_paths(summary, "reference"),
    )
    if summary != expected:
        raise ValueError("candidate02 tail-v2 merge aggregate/tamper mismatch")
    report = _validation_report(summary)
    if set(report) != _VALIDATION_KEYS:
        raise AssertionError("candidate02 tail-v2 validation schema changed")
    if output_path is not None:
        base._write_once(Path(output_path).resolve(), report)
    return report


def merge_and_validate_candidate02_tail_v2(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    summary_output_path: Path,
    validation_output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_output = Path(summary_output_path).resolve()
    validation_output = Path(validation_output_path).resolve()
    if summary_output == validation_output:
        raise ValueError(
            "candidate02 tail-v2 summary and validation outputs must differ"
        )
    if summary_output.exists() or validation_output.exists():
        raise FileExistsError("candidate02 tail-v2 merge outputs are write-once")
    summary = _merge_candidate02_tail_v2(
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
    summary, validation = merge_and_validate_candidate02_tail_v2(
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
    "EXPOSED_CANDIDATE02_V1_HAND_INDICES",
    "MERGE_SCHEMA",
    "SELECTION_MANIFEST_SHA256",
    "SELECTION_MANIFEST_PATH",
    "TAIL_DIAGNOSTIC_SCOPE",
    "TAIL_HAND_INDICES",
    "TAIL_HEAVY_HAND_INDICES",
    "TAIL_RANDOM_HAND_INDICES",
    "VALIDATION_SCHEMA",
    "main",
    "merge_and_validate_candidate02_tail_v2",
    "merge_candidate02_tail_v2",
    "validate_candidate02_tail_v2_merge",
]
