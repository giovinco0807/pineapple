"""Sweep HU T3 Stage9 tail-veto probability thresholds.

The input is the scores CSV emitted by train_hu_turn3_stage9_tail_veto.py.
Rows are runtime-fired Stage9/Stage9d overrides with MC replay delta labels.
This script treats ``hard_negative_probability >= threshold`` as vetoed and
reports the kept fired-state EV/tail profile. It is an offline diagnostic; a
threshold still needs full seat-swap validation before production use.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_THRESHOLDS = "0.2,0.3,0.4,0.5,0.6,0.7,0.8"


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_thresholds(value: str) -> list[float]:
    thresholds = [safe_float(part) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return sorted(set(thresholds))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [safe_float(row.get("delta_vs_baseline")) for row in rows]
    losses = [-delta for delta in deltas if delta < 0.0]
    labels = Counter(str(row.get("tail_label", "")) for row in rows)
    delta_sum = sum(deltas)
    return {
        "rows": len(rows),
        "delta_sum": delta_sum,
        "mean_delta": delta_sum / len(rows) if rows else 0.0,
        "median_delta": percentile(deltas, 0.5),
        "hard_negative_count": labels.get("hard_negative", 0),
        "safe_positive_count": labels.get("safe_positive", 0),
        "gray_count": labels.get("gray", 0),
        "negative_count": sum(1 for delta in deltas if delta < 0.0),
        "p90_loss": percentile(losses, 0.90),
        "p95_loss": percentile(losses, 0.95),
        "p99_loss": percentile(losses, 0.99),
        "worst_delta": min(deltas) if deltas else None,
        "best_delta": max(deltas) if deltas else None,
    }


def threshold_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    splits = ["all", *sorted({str(row.get("split", "")) for row in rows})]
    for split in splits:
        split_rows = rows if split == "all" else [row for row in rows if str(row.get("split", "")) == split]
        base = summarize_subset(split_rows)
        for threshold in thresholds:
            kept = [
                row
                for row in split_rows
                if safe_float(row.get("hard_negative_probability")) < threshold
            ]
            vetoed = [row for row in split_rows if row not in kept]
            kept_summary = summarize_subset(kept)
            vetoed_summary = summarize_subset(vetoed)
            output.append(
                {
                    "split": split,
                    "threshold": threshold,
                    "base_rows": base["rows"],
                    "base_delta_sum": base["delta_sum"],
                    "base_mean_delta": base["mean_delta"],
                    "base_hard_negative_count": base["hard_negative_count"],
                    "kept_rows": kept_summary["rows"],
                    "kept_rate": kept_summary["rows"] / base["rows"] if base["rows"] else 0.0,
                    "kept_delta_sum": kept_summary["delta_sum"],
                    "kept_mean_delta": kept_summary["mean_delta"],
                    "kept_hard_negative_count": kept_summary["hard_negative_count"],
                    "kept_negative_count": kept_summary["negative_count"],
                    "kept_p95_loss": kept_summary["p95_loss"],
                    "kept_worst_delta": kept_summary["worst_delta"],
                    "vetoed_rows": vetoed_summary["rows"],
                    "vetoed_delta_sum": vetoed_summary["delta_sum"],
                    "vetoed_mean_delta": vetoed_summary["mean_delta"],
                    "vetoed_hard_negative_count": vetoed_summary["hard_negative_count"],
                    "vetoed_safe_positive_count": vetoed_summary["safe_positive_count"],
                    "vetoed_worst_delta": vetoed_summary["worst_delta"],
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "threshold",
        "base_rows",
        "base_delta_sum",
        "base_mean_delta",
        "base_hard_negative_count",
        "kept_rows",
        "kept_rate",
        "kept_delta_sum",
        "kept_mean_delta",
        "kept_hard_negative_count",
        "kept_negative_count",
        "kept_p95_loss",
        "kept_worst_delta",
        "vetoed_rows",
        "vetoed_delta_sum",
        "vetoed_mean_delta",
        "vetoed_hard_negative_count",
        "vetoed_safe_positive_count",
        "vetoed_worst_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for row in rows if row["split"] == "all"]
    holdout_rows = [row for row in rows if row["split"] == "test"]
    ranked = sorted(
        holdout_rows or all_rows,
        key=lambda row: (
            int(row["kept_hard_negative_count"]),
            -float(row["kept_delta_sum"]),
            int(row["vetoed_safe_positive_count"]),
        ),
    )
    table_rows = ranked[:10]
    lines = [
        "# HU Turn3 Stage9 Tail Veto Threshold Sweep",
        "",
        "Lower `threshold` is stricter: rows with `hard_negative_probability >= threshold` are vetoed.",
        "",
        "## Top Candidate Thresholds",
        "",
        "| split | threshold | kept | kept mean | kept HN | kept worst | vetoed HN | vetoed safe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split"]),
                    fmt(row["threshold"]),
                    fmt(row["kept_rows"]),
                    fmt(row["kept_mean_delta"]),
                    fmt(row["kept_hard_negative_count"]),
                    fmt(row["kept_worst_delta"]),
                    fmt(row["vetoed_hard_negative_count"]),
                    fmt(row["vetoed_safe_positive_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This sweep is label-cache evidence only. Use it to choose a small number of runtime gate thresholds for full paired seat-swap and fired replay validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    rows = read_rows(args.input)
    result = threshold_rows(rows, thresholds=parse_thresholds(args.thresholds))
    write_csv(args.output, result)
    write_summary(args.summary_output, result)


if __name__ == "__main__":
    main()
