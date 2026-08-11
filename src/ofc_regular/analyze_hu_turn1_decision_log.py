"""Analyze HU Turn1 selective-override decision logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BUCKETS = (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, math.inf)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_error": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    sorted_values = sorted(values)
    count = len(values)
    mean = sum(values) / count
    median = (
        sorted_values[count // 2]
        if count % 2
        else (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2.0
    )
    variance = sum((value - mean) ** 2 for value in values) / max(count - 1, 1)
    stderr = math.sqrt(variance / count)
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "min": min(values),
        "max": max(values),
        "std_error": stderr,
        "ci95_low": mean - 1.96 * stderr,
        "ci95_high": mean + 1.96 * stderr,
    }


def valid_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("realized_delta_valid")
        and row.get("realized_candidate_seat_delta") not in (None, "")
    ]


def bucket_label(low: float, high: float) -> str:
    if math.isinf(high):
        return f">={low:g}"
    return f"[{low:g},{high:g})"


def build_margin_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fired = [row for row in valid_rows(rows) if row.get("override_fired")]
    output: list[dict[str, Any]] = []
    for low, high in zip(DEFAULT_BUCKETS, DEFAULT_BUCKETS[1:]):
        bucket = [
            row
            for row in fired
            if low <= finite_float(row.get("hu_turn1_predicted_margin")) < high
        ]
        deltas = [finite_float(row.get("realized_candidate_seat_delta")) for row in bucket]
        stats = summarize_values(deltas)
        output.append(
            {
                "margin_bucket": bucket_label(low, high),
                "rows": len(bucket),
                "mean_delta": stats["mean"],
                "median_delta": stats["median"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "loss_count": sum(1 for value in deltas if value < 0),
                "win_count": sum(1 for value in deltas if value > 0),
                "zero_count": sum(1 for value in deltas if value == 0),
            }
        )
    return output


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = valid_rows(rows)
    fired = [row for row in valid if row.get("override_fired")]
    non_fired = [row for row in valid if not row.get("override_fired")]
    fired_deltas = [finite_float(row.get("realized_candidate_seat_delta")) for row in fired]
    non_fired_deltas = [
        finite_float(row.get("realized_candidate_seat_delta")) for row in non_fired
    ]
    fired_margins = [finite_float(row.get("hu_turn1_predicted_margin")) for row in fired]
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "fired_valid_rows": len(fired),
        "non_fired_valid_rows": len(non_fired),
        "override_rate_on_valid": len(fired) / len(valid) if valid else 0.0,
        "fired_delta": summarize_values(fired_deltas),
        "fired_margin": summarize_values(fired_margins),
        "fired_loss_count": sum(1 for value in fired_deltas if value < 0),
        "fired_win_count": sum(1 for value in fired_deltas if value > 0),
        "fired_zero_count": sum(1 for value in fired_deltas if value == 0),
        "non_fired_nonzero_count": sum(1 for value in non_fired_deltas if abs(value) > 1e-9),
        "non_fired_delta_sum": sum(non_fired_deltas),
        "non_fired_delta_max_abs": max((abs(value) for value in non_fired_deltas), default=0.0),
        "no_override_reason_counts": dict(Counter(row.get("no_override_reason", "") for row in rows)),
        "seat_counts": dict(Counter(row.get("seat", "") for row in rows)),
        "fired_seat_counts": dict(Counter(row.get("seat", "") for row in fired)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def labeled_fired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for row in valid_rows(rows):
        if not row.get("override_fired"):
            continue
        delta = finite_float(row.get("realized_candidate_seat_delta"))
        if delta > 0:
            label = "positive"
            label_id = 1
        elif delta < 0:
            label = "hard_negative"
            label_id = 0
        else:
            label = "neutral"
            label_id = -1
        enriched = dict(row)
        enriched["hu_turn1_safe_override_label"] = label
        enriched["hu_turn1_safe_override_label_id"] = label_id
        enriched["hu_turn1_realized_delta"] = delta
        enriched["hu_turn1_prediction_margin_role"] = "gate_diagnostic_only"
        enriched["hu_turn1_performance_metric"] = "realized_fired_decision_delta"
        labeled.append(enriched)
    return labeled


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    fired = summary["fired_delta"]
    lines = [
        "# HU Turn1 Decision Log Analysis",
        "",
        f"- rows: `{summary['rows']}`",
        f"- valid rows: `{summary['valid_rows']}`",
        f"- fired valid rows: `{summary['fired_valid_rows']}`",
        f"- override rate on valid: `{summary['override_rate_on_valid']:.4f}`",
        f"- fired delta mean: `{fired['mean']:.4f}`",
        f"- fired delta 95% CI: `[{fired['ci95_low']:.4f}, {fired['ci95_high']:.4f}]`",
        f"- fired losses/wins/zeros: `{summary['fired_loss_count']} / {summary['fired_win_count']} / {summary['fired_zero_count']}`",
        f"- non-fired nonzero count: `{summary['non_fired_nonzero_count']}`",
        "",
        "Decision note: use fired realized deltas as the performance metric; model",
        "prediction margins are gate diagnostics only.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_file(input_path: Path, output_dir: Path, *, top_losses: int = 50) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    summary = analyze_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_rows = build_margin_bucket_rows(rows)
    write_csv(output_dir / "hu_turn1_margin_buckets.csv", bucket_rows)

    labeled_rows = labeled_fired_rows(rows)
    positive_rows = [
        row for row in labeled_rows if row["hu_turn1_safe_override_label"] == "positive"
    ]
    neutral_rows = [
        row for row in labeled_rows if row["hu_turn1_safe_override_label"] == "neutral"
    ]
    loss_rows = sorted(
        [row for row in labeled_rows if row["hu_turn1_safe_override_label"] == "hard_negative"],
        key=lambda row: finite_float(row.get("realized_candidate_seat_delta")),
    )
    all_fired_count = write_jsonl(
        output_dir / "hu_turn1_fired_labeled_targets.jsonl",
        labeled_rows,
    )
    positive_count = write_jsonl(
        output_dir / "hu_turn1_positive_targets.jsonl",
        positive_rows,
    )
    neutral_count = write_jsonl(
        output_dir / "hu_turn1_neutral_targets.jsonl",
        neutral_rows,
    )
    top_loss_count = write_jsonl(
        output_dir / "hu_turn1_top_losses.jsonl",
        loss_rows[:top_losses],
    )
    hard_negative_count = write_jsonl(
        output_dir / "hu_turn1_hard_negative_targets.jsonl",
        loss_rows,
    )
    summary["artifacts"] = {
        "margin_buckets": str(output_dir / "hu_turn1_margin_buckets.csv"),
        "fired_labeled_targets": str(output_dir / "hu_turn1_fired_labeled_targets.jsonl"),
        "positive_targets": str(output_dir / "hu_turn1_positive_targets.jsonl"),
        "neutral_targets": str(output_dir / "hu_turn1_neutral_targets.jsonl"),
        "top_losses": str(output_dir / "hu_turn1_top_losses.jsonl"),
        "hard_negative_targets": str(output_dir / "hu_turn1_hard_negative_targets.jsonl"),
    }
    summary["fired_labeled_rows_written"] = all_fired_count
    summary["positive_rows_written"] = positive_count
    summary["neutral_rows_written"] = neutral_count
    summary["top_loss_rows_written"] = top_loss_count
    summary["hard_negative_rows_written"] = hard_negative_count

    summary_path = output_dir / "hu_turn1_decision_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "hu_turn1_decision_summary.md", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-losses", type=int, default=50)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = analyze_file(args.input, args.output_dir, top_losses=args.top_losses)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
