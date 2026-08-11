"""Audit Stage8c fire-head predictions as fire selectors.

This differs from risk/veto audits: high ``risk_probability`` means "fire the
candidate" for ``topk_confirm_fire`` models, so selected rows are evaluated by
their realized/replay delta directly. The output is screening evidence for
candidate generation only; it does not approve runtime, P2, production, T1, or
50k teacher generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_fire_selector_audit")
DEFAULT_TOPK = (1, 3, 5, 10, 20, 50, 100)
DEFAULT_THRESHOLDS = (0.70, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", default=",".join(str(value) for value in DEFAULT_TOPK))
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--group-field",
        action="append",
        default=["seat", "candidate_rank_bucket", "recommended_training_use"],
        help="Group field to audit. Repeatable.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        output = float(value)
        return output if math.isfinite(output) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_ints(value: str) -> list[int]:
    output = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not output:
        raise ValueError("at least one topk value is required")
    return output


def parse_floats(value: str) -> list[float]:
    output = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not output:
        raise ValueError("at least one threshold is required")
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_value(row: dict[str, Any]) -> int:
    return 1 if safe_int(row.get("label")) == 1 else 0


def realized_observed(row: dict[str, Any]) -> bool:
    return truthy(row.get("realized_delta_observed"))


def realized_delta(row: dict[str, Any]) -> float:
    return safe_float(row.get("realized_delta"))


def candidate_rank_bucket(row: dict[str, Any]) -> str:
    rank = safe_int(row.get("candidate_ev_rank"), 9999)
    if rank <= 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 5:
        return "rank_4_5"
    return "rank_6_plus"


def group_value(row: dict[str, Any], field: str) -> str:
    if field == "all":
        return "all"
    if field == "candidate_rank_bucket":
        return candidate_rank_bucket(row)
    return str(row.get(field) or "")


def probability(row: dict[str, Any]) -> float:
    return safe_float(row.get("risk_probability"))


def selected_metrics(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(label_value(row) for row in rows)
    selected_positives = sum(label_value(row) for row in selected)
    selected_count = len(selected)
    selected_deltas = [realized_delta(row) for row in selected]
    selected_losses = [max(0.0, -value) for value in selected_deltas if value < 0.0]
    observed_count = sum(1 for row in selected if realized_observed(row))
    delta_sum = float(sum(selected_deltas))
    return {
        "rows": len(rows),
        "positives": positives,
        "selected_rows": selected_count,
        "tp": selected_positives,
        "fp": selected_count - selected_positives,
        "precision": selected_positives / selected_count if selected_count else 0.0,
        "recall": selected_positives / positives if positives else 0.0,
        "min_probability": min((probability(row) for row in selected), default=0.0),
        "selected_realized_delta_sum": delta_sum,
        "selected_realized_delta_mean": delta_sum / selected_count if selected_count else 0.0,
        "estimated_realized_delta_per_row": delta_sum / len(rows) if rows else 0.0,
        "selected_negative_rows": sum(1 for value in selected_deltas if value < 0.0),
        "selected_negative_rate": sum(1 for value in selected_deltas if value < 0.0) / selected_count if selected_count else 0.0,
        "selected_realized_loss_count": len(selected_losses),
        "selected_realized_loss_mean": float(sum(selected_losses)) / len(selected_losses) if selected_losses else 0.0,
        "selected_realized_max_loss": max(selected_losses, default=0.0),
        "selected_observed_realized_delta_count": observed_count,
        "selected_unknown_realized_delta_count": selected_count - observed_count,
    }


def topk_metric_rows(rows: list[dict[str, Any]], *, topk_values: Iterable[int], group_fields: Iterable[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    splits = sorted({str(row.get("split") or "") for row in rows})
    for split in [*splits, "all"]:
        split_rows = rows if split == "all" else [row for row in rows if str(row.get("split") or "") == split]
        for field in ["all", *group_fields]:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in split_rows:
                groups[group_value(row, field)].append(row)
            for value, group_rows in sorted(groups.items()):
                ordered = sorted(group_rows, key=probability, reverse=True)
                for topk in topk_values:
                    selected = ordered[: min(topk, len(ordered))]
                    output.append(
                        {
                            "split": split,
                            "group_field": field,
                            "group_value": value,
                            "selector": "topk",
                            "topk": int(topk),
                            "threshold": "",
                            **selected_metrics(group_rows, selected),
                        }
                    )
    return output


def threshold_metric_rows(
    rows: list[dict[str, Any]], *, thresholds: Iterable[float], group_fields: Iterable[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    splits = sorted({str(row.get("split") or "") for row in rows})
    for split in [*splits, "all"]:
        split_rows = rows if split == "all" else [row for row in rows if str(row.get("split") or "") == split]
        for field in ["all", *group_fields]:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in split_rows:
                groups[group_value(row, field)].append(row)
            for value, group_rows in sorted(groups.items()):
                for threshold in thresholds:
                    selected = [row for row in group_rows if probability(row) >= threshold]
                    output.append(
                        {
                            "split": split,
                            "group_field": field,
                            "group_value": value,
                            "selector": "threshold",
                            "topk": "",
                            "threshold": float(threshold),
                            **selected_metrics(group_rows, selected),
                        }
                    )
    return output


def best_rows(rows: list[dict[str, Any]], *, split: str = "test", min_selected: int = 5) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("split")) == split
        and safe_int(row.get("selected_rows")) >= min_selected
        and str(row.get("group_field")) != "recommended_training_use"
    ]
    candidates.sort(
        key=lambda row: (
            -safe_float(row.get("estimated_realized_delta_per_row")),
            -safe_float(row.get("selected_realized_delta_mean")),
            safe_float(row.get("selected_negative_rate")),
        )
    )
    return candidates


def write_summary(path: Path, *, manifest: dict[str, Any], top_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> None:
    best_top = best_rows(top_rows)[:10]
    best_threshold = best_rows(threshold_rows)[:10]
    lines = [
        "# HU T2 Stage8c Fire Selector Audit",
        "",
        "This evaluates high risk_probability as a fire selector, not as a veto. It is screening evidence only.",
        "",
        f"- predictions: `{manifest['predictions']}`",
        f"- rows: `{manifest['rows']}`",
        f"- production / P2 fixed: `{manifest['production_p2_fixed']}`",
        f"- 50k teacher: `{manifest['teacher_50k']}`",
        f"- T1 training: `{manifest['t1_training']}`",
        "",
        "## Best Test TopK Screens",
        "",
        "| group | value | topk | selected | precision | mean delta | delta/row | neg rate | max loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_top:
        lines.append(
            "| {group} | {value} | {topk} | {selected} | {precision:.3f} | {mean:.4f} | {per_row:.4f} | {neg:.3f} | {loss:.4f} |".format(
                group=row.get("group_field", ""),
                value=row.get("group_value", ""),
                topk=safe_int(row.get("topk")),
                selected=safe_int(row.get("selected_rows")),
                precision=safe_float(row.get("precision")),
                mean=safe_float(row.get("selected_realized_delta_mean")),
                per_row=safe_float(row.get("estimated_realized_delta_per_row")),
                neg=safe_float(row.get("selected_negative_rate")),
                loss=safe_float(row.get("selected_realized_max_loss")),
            )
        )
    lines.extend(
        [
            "",
            "## Best Test Threshold Screens",
            "",
            "| group | value | threshold | selected | precision | mean delta | delta/row | neg rate | max loss |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_threshold:
        lines.append(
            "| {group} | {value} | {threshold:.2f} | {selected} | {precision:.3f} | {mean:.4f} | {per_row:.4f} | {neg:.3f} | {loss:.4f} |".format(
                group=row.get("group_field", ""),
                value=row.get("group_value", ""),
                threshold=safe_float(row.get("threshold")),
                selected=safe_int(row.get("selected_rows")),
                precision=safe_float(row.get("precision")),
                mean=safe_float(row.get("selected_realized_delta_mean")),
                per_row=safe_float(row.get("estimated_realized_delta_per_row")),
                neg=safe_float(row.get("selected_negative_rate")),
                loss=safe_float(row.get("selected_realized_max_loss")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- allowed role: `fire-selector screening and next replay target design`",
            "- runtime gate: `No-Go`",
            "- production / P2 fixed / T1 / 50k: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_csv(args.predictions)
    topk_values = parse_ints(args.topk)
    thresholds = parse_floats(args.thresholds)
    top_rows = topk_metric_rows(rows, topk_values=topk_values, group_fields=args.group_field)
    threshold_rows = threshold_metric_rows(rows, thresholds=thresholds, group_fields=args.group_field)
    manifest = {
        "schema": "hu_turn2_stage8c_fire_selector_audit_v1",
        "predictions": str(args.predictions),
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "topk": topk_values,
        "thresholds": thresholds,
        "group_fields": args.group_field,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fire_selector_topk_metrics.csv", top_rows)
    write_csv(args.output_dir / "fire_selector_threshold_metrics.csv", threshold_rows)
    write_csv(args.output_dir / "fire_selector_best_topk_test.csv", best_rows(top_rows))
    write_csv(args.output_dir / "fire_selector_best_threshold_test.csv", best_rows(threshold_rows))
    (args.output_dir / "fire_selector_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "fire_selector_summary.md", manifest=manifest, top_rows=top_rows, threshold_rows=threshold_rows)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
