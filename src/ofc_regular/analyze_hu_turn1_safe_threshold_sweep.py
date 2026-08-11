"""Analyze stricter safe-selector thresholds inside an already-fired T1 set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = percentile_value / 100.0 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_thresholds(
    fired_rows: list[dict[str, Any]],
    *,
    thresholds: Iterable[float],
    parent_decisions: int,
) -> list[dict[str, Any]]:
    if parent_decisions <= 0:
        raise ValueError("parent_decisions must be positive")
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [
            row
            for row in fired_rows
            if safe_float(row.get("safe_selector_score"), -math.inf) >= threshold
        ]
        deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in selected]
        count = len(deltas)
        mean_delta = sum(deltas) / count if count else 0.0
        if count > 1:
            variance = sum((value - mean_delta) ** 2 for value in deltas) / (count - 1)
            standard_error = math.sqrt(variance / count)
        else:
            standard_error = 0.0
        losses = [max(0.0, -value) for value in deltas]
        fire_rate = count / parent_decisions
        output.append(
            {
                "threshold": float(threshold),
                "fires": count,
                "fire_rate_per_decision": fire_rate,
                "realized_per_fire_delta_mean": mean_delta,
                "realized_per_fire_delta_se": standard_error,
                "realized_per_fire_ci95_low": mean_delta - 1.96 * standard_error,
                "realized_per_fire_ci95_high": mean_delta + 1.96 * standard_error,
                "estimated_ev_per_decision": fire_rate * mean_delta,
                "estimated_ev_per_decision_ci95_low": fire_rate
                * (mean_delta - 1.96 * standard_error),
                "estimated_ev_per_decision_ci95_high": fire_rate
                * (mean_delta + 1.96 * standard_error),
                "loss_count": sum(value < 0.0 for value in deltas),
                "win_count": sum(value > 0.0 for value in deltas),
                "zero_count": sum(value == 0.0 for value in deltas),
                "p90_loss": percentile(losses, 90.0),
                "p95_loss": percentile(losses, 95.0),
                "p99_loss": percentile(losses, 99.0),
                "max_loss": max(losses, default=0.0),
            }
        )
    return output


def read_fired_rows(path: Path) -> tuple[list[dict[str, Any]], int, float]:
    rows: list[dict[str, Any]] = []
    decision_count = 0
    source_thresholds: set[float] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            decision_count += 1
            row = json.loads(line)
            if not row.get("override_fired"):
                continue
            if not row.get("realized_delta_valid"):
                raise ValueError(f"fired row {line_number} has no valid realized delta")
            if row.get("safe_selector_score") in (None, ""):
                raise ValueError(f"fired row {line_number} has no safe selector score")
            source_thresholds.add(safe_float(row.get("safe_selector_threshold")))
            rows.append(row)
    if len(source_thresholds) != 1:
        raise ValueError(f"expected one source threshold; found {sorted(source_thresholds)}")
    return rows, decision_count, next(iter(source_thresholds))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], *, source_threshold: float) -> None:
    lines = [
        "# HU T1 Nested Safe-Threshold Sweep",
        "",
        f"- source runtime threshold: `{source_threshold:g}`",
        "- performance source: `realized_seat_swap_counterfactual`",
        "- scope: stricter nested subsets of actual fired rows only",
        "",
        "| threshold | fires | rate | per-fire | CI95 | EV/decision | p95 | p99 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {threshold:.2f} | {fires} | {rate:.5f} | {mean:+.4f} | "
            "[{low:+.4f},{high:+.4f}] | {ev:+.5f} | {p95:.3f} | {p99:.3f} | {max_loss:.3f} |".format(
                threshold=row["threshold"],
                fires=row["fires"],
                rate=row["fire_rate_per_decision"],
                mean=row["realized_per_fire_delta_mean"],
                low=row["realized_per_fire_ci95_low"],
                high=row["realized_per_fire_ci95_high"],
                ev=row["estimated_ev_per_decision"],
                p95=row["p95_loss"],
                p99=row["p99_loss"],
                max_loss=row["max_loss"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fired_rows, decision_count, source_threshold = read_fired_rows(args.input)
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    if any(threshold < source_threshold for threshold in thresholds):
        raise ValueError("nested sweep thresholds may not be lower than the source runtime threshold")
    rows = summarize_thresholds(fired_rows, thresholds=thresholds, parent_decisions=decision_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "threshold_sweep.csv", rows)
    write_markdown(args.output_dir / "threshold_sweep.md", rows, source_threshold=source_threshold)
    manifest = {
        "schema": "hu_turn1_nested_safe_threshold_sweep_v1",
        "input": str(args.input),
        "decision_count": decision_count,
        "source_fires": len(fired_rows),
        "source_threshold": source_threshold,
        "thresholds": thresholds,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
