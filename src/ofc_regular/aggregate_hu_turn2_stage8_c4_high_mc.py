"""Aggregate GCP shards for HU Turn2 Stage8 C4 selected high-MC replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .audit_hu_turn2_stage8_high_mc import (
    calibration_vs_high_mc_rows,
    group_breakdown,
    threshold_sweep_rows,
    write_audit_metrics_summary,
    write_csv,
    write_jsonl,
    write_recommended_next_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Downloaded GCP run directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_result_dirs(input_dir: Path) -> list[Path]:
    roots = [input_dir / "results", input_dir]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("DONE"):
            found.append(path.parent)
    return sorted(set(found))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_summary(path: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    diagnoses = Counter(str(row.get("diagnosis", "unknown")) for row in results)
    false_positive = diagnoses.get("false_positive_gate", 0)
    positive_delta = sum(1 for row in results if safe_float(row.get("high_mc_delta_candidate_vs_baseline")) > 0.0)
    lower95_positive = sum(1 for row in results if safe_float(row.get("high_mc_lower95_candidate_vs_baseline")) > 0.0)
    enough = len(results) >= int(manifest.get("target_replay_states", 50) or 50)
    c4_status = "Conditional-Go" if enough and not failures and false_positive == 0 else "No-Go"
    lines = [
        "# HU Turn2 Stage8 C4 Selected High-MC Replay",
        "",
        "C4 is analysis-only. It does not authorize production training, T1 training, or a 50k teacher run.",
        "",
        f"- run_name: `{manifest.get('run_name', '')}`",
        f"- mc_samples: `{manifest.get('mc_samples', '')}`",
        f"- target_replay_states: `{manifest.get('target_replay_states', '')}`",
        f"- high-MC successes: `{len(results)}`",
        f"- high-MC failures: `{len(failures)}`",
        f"- positive candidate-vs-baseline delta: `{positive_delta}`",
        f"- lower95 positive candidate-vs-baseline delta: `{lower95_positive}`",
        f"- false_positive_gate: `{false_positive}`",
        "",
        "## Diagnosis Counts",
        "",
    ]
    for label, count in diagnoses.most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- C4 selected high-MC replay: `{c4_status}`",
            "- production training: `No-Go`",
            "- T1 training: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- C2/C3 threshold is still evaluation-only until C4 and another larger seat-swap validation are clean.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    result_dirs = collect_result_dirs(args.input_dir)
    if not result_dirs:
        raise SystemExit(f"no C4 shard result dirs found under {args.input_dir}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        results.extend(iter_jsonl(result_dir / "high_mc_audit_results.jsonl") or ())
        failures.extend(iter_jsonl(result_dir / "high_mc_audit_failures.jsonl") or ())
        action_rows.extend(read_csv(result_dir / "high_mc_action_ev_table.csv"))

    write_jsonl(args.output_dir / "high_mc_audit_results.jsonl", results)
    write_jsonl(args.output_dir / "high_mc_audit_failures.jsonl", failures)
    write_csv(args.output_dir / "high_mc_action_ev_table.csv", action_rows)
    write_csv(args.output_dir / "calibration_vs_high_mc.csv", calibration_vs_high_mc_rows(results))
    threshold_rows = threshold_sweep_rows(results)
    write_csv(args.output_dir / "threshold_sweep_high_mc.csv", threshold_rows)
    write_csv(args.output_dir / "diagnosis_breakdown.csv", group_breakdown(results, "diagnosis"))
    write_csv(args.output_dir / "position_breakdown.csv", group_breakdown(results, "seat"))
    write_csv(args.output_dir / "bucket_breakdown.csv", group_breakdown(results, "bucket_group"))
    write_audit_metrics_summary(args.output_dir, results, threshold_rows)
    write_recommended_next_step(args.output_dir / "recommended_next_step.md", results, failures, int(manifest.get("target_replay_states", 50) or 50))
    write_summary(args.output_dir / "c4_high_mc_summary.md", results, failures, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "completed_shards": len(result_dirs),
                "high_mc_successes": len(results),
                "high_mc_failures": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
