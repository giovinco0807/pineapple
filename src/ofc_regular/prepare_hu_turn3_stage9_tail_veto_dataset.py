"""Prepare a Stage9 HU T3 tail-risk veto dataset from fired-state replay.

The input is one or more joint-exact teacher JSONL files produced from
runtime-fired HU T3 states. The output is a flat CSV with runtime-available
features and labels:

- safe_positive: delta_vs_baseline >= positive_threshold
- hard_negative: delta_vs_baseline <= negative_threshold
- gray: otherwise

This script does not train or promote a runtime. It creates an auditable label
cache for the next learned tail-risk veto experiment.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from ofc_regular.analyze_hu_turn3_stage9_tail_veto import _build_rows, _summarize_deltas


FIELDNAMES = [
    "dataset_row_id",
    "source_path",
    "source_name",
    "sample_id",
    "state_id",
    "seed",
    "seat",
    "baseline_index",
    "hu_index",
    "predicted_margin_vs_baseline",
    "reference_margin",
    "model_score",
    "best_score",
    "baseline_score",
    "hu_score",
    "delta_vs_baseline",
    "regret_vs_best",
    "legal_action_count",
    "hu_top_full",
    "hu_middle_full",
    "hu_bottom_full",
    "baseline_top_full",
    "baseline_middle_full",
    "baseline_bottom_full",
    "hu_places_top_count",
    "hu_places_middle_count",
    "hu_places_bottom_count",
    "baseline_places_top_count",
    "baseline_places_middle_count",
    "baseline_places_bottom_count",
    "hu_discard_count",
    "baseline_discard_count",
    "hu_discard_rank_max",
    "baseline_discard_rank_max",
    "hu_placed_rank_sum",
    "baseline_placed_rank_sum",
    "hu_placed_rank_max",
    "baseline_placed_rank_max",
    "hu_fills_more_rows_than_baseline",
    "hu_discards_higher_rank_than_baseline",
    "tail_label",
    "tail_label_id",
    "safe_positive",
    "hard_negative",
    "gray",
    "label_weight",
]


LABEL_IDS = {
    "hard_negative": 0,
    "gray": 1,
    "safe_positive": 2,
}


def _source_name(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if stem.startswith("override_joint_exact"):
        return parent
    return f"{parent}:{stem}"


def _label_row(row: dict[str, Any], *, positive_threshold: float, negative_threshold: float) -> dict[str, Any]:
    delta = float(row["delta_vs_baseline"])
    if delta >= positive_threshold:
        label = "safe_positive"
        weight = 1.0
    elif delta <= negative_threshold:
        label = "hard_negative"
        weight = 3.0
    else:
        label = "gray"
        weight = 0.35
    result = dict(row)
    result.update(
        {
            "tail_label": label,
            "tail_label_id": LABEL_IDS[label],
            "safe_positive": label == "safe_positive",
            "hard_negative": label == "hard_negative",
            "gray": label == "gray",
            "label_weight": weight,
        }
    )
    return result


def build_dataset(
    inputs: list[Path],
    *,
    positive_threshold: float = 0.25,
    negative_threshold: float = -0.25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        for row in _build_rows(path):
            labeled = _label_row(
                row,
                positive_threshold=positive_threshold,
                negative_threshold=negative_threshold,
            )
            labeled["source_path"] = str(path)
            labeled["source_name"] = _source_name(path)
            rows.append(labeled)
    rows.sort(key=lambda r: (str(r.get("source_name")), str(r.get("state_id")), str(r.get("sample_id"))))
    for idx, row in enumerate(rows):
        row["dataset_row_id"] = idx
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in FIELDNAMES})


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    inputs: list[Path],
    positive_threshold: float,
    negative_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter(row["tail_label"] for row in rows)
    seat_counts = Counter(row.get("seat") for row in rows)
    source_counts = Counter(row.get("source_name") for row in rows)
    summary = _summarize_deltas(rows)

    source_lines = []
    for source, count in sorted(source_counts.items()):
        source_rows = [row for row in rows if row.get("source_name") == source]
        source_summary = _summarize_deltas(source_rows)
        source_label_counts = Counter(row["tail_label"] for row in source_rows)
        source_lines.append(
            "| "
            + " | ".join(
                [
                    str(source),
                    str(count),
                    _fmt(source_summary["mean_delta"]),
                    _fmt(source_summary["worst_delta"]),
                    str(source_label_counts.get("safe_positive", 0)),
                    str(source_label_counts.get("gray", 0)),
                    str(source_label_counts.get("hard_negative", 0)),
                ]
            )
            + " |"
        )

    seat_lines = []
    for seat, count in sorted(seat_counts.items()):
        seat_rows = [row for row in rows if row.get("seat") == seat]
        seat_summary = _summarize_deltas(seat_rows)
        seat_lines.append(f"- {seat}: n={count}, mean_delta={_fmt(seat_summary['mean_delta'])}, worst={_fmt(seat_summary['worst_delta'])}")

    text = "\n".join(
        [
            "# HU Turn3 Stage9 Tail Veto Dataset",
            "",
            "## Inputs",
            "",
            *[f"- `{path}`" for path in inputs],
            "",
            "## Label Rule",
            "",
            f"- safe_positive: `delta_vs_baseline >= {positive_threshold}`",
            f"- hard_negative: `delta_vs_baseline <= {negative_threshold}`",
            "- gray: otherwise",
            "",
            "## Overall",
            "",
            f"- rows: {len(rows)}",
            f"- mean delta: {_fmt(summary['mean_delta'])}",
            f"- median delta: {_fmt(summary['median_delta'])}",
            f"- worst delta: {_fmt(summary['worst_delta'])}",
            f"- best delta: {_fmt(summary['best_delta'])}",
            f"- safe_positive: {label_counts.get('safe_positive', 0)}",
            f"- gray: {label_counts.get('gray', 0)}",
            f"- hard_negative: {label_counts.get('hard_negative', 0)}",
            "",
            "## Seat Split",
            "",
            *seat_lines,
            "",
            "## Source Split",
            "",
            "| source | rows | mean delta | worst | safe | gray | hard negative |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *source_lines,
            "",
            "## Interpretation",
            "",
            "This is a label cache for a learned tail-risk veto. It is not production evidence by itself. The current sample is useful for schema, audit, and first smoke training only; it is too small for a production classifier.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--positive-threshold", type=float, default=0.25)
    parser.add_argument("--negative-threshold", type=float, default=-0.25)
    args = parser.parse_args()

    rows = build_dataset(
        args.input,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )
    write_csv(args.output, rows)
    write_summary(
        args.summary_output,
        rows,
        inputs=args.input,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )


if __name__ == "__main__":
    main()
