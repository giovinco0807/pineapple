"""Analyze Stage8c trajectory components for fired TopK decisions.

This is a diagnostic tool for trajectory-enriched counterfactual loss targets.
It explains realized candidate-vs-baseline deltas using the terminal score
component deltas written by ``evaluate_hu_turn2_stage8b_topk_mc_rerank``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .analyze_hu_turn2_stage8c_risk_target_gap import load_rows, target_paths


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_trajectory_component_analysis")
COMPONENT_FIELDS = {
    "foul": "foul_delta_vs_baseline",
    "line": "line_score_delta_vs_baseline",
    "scoop": "scoop_delta_vs_baseline",
    "royalty": "royalty_delta_vs_baseline",
    "fl": "fl_delta_vs_baseline",
}
DETAIL_FIELDS = {
    "hero_royalty": "hero_royalty_vs_baseline",
    "opponent_royalty": "opponent_royalty_vs_baseline",
    "hero_fl": "hero_fl_value_vs_baseline",
    "opponent_fl": "opponent_fl_value_vs_baseline",
}
DOWNSTREAM_TRAJECTORY_FIELDS = (
    "post_t2_candidate_board",
    "post_t2_baseline_board",
    "t3_decision_summary",
    "final_board_hero",
    "final_board_opponent",
    "hero_foul",
    "opponent_foul",
    "hero_fl_entry",
    "hero_fl_stay",
    "opponent_fl_entry",
    "hero_royalty",
    "opponent_royalty",
    "line_score_delta",
    "scoop_delta",
    "downstream_override_fired",
    "paired_future_delta_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, action="append", default=[])
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def downstream_complete(row: dict[str, Any]) -> bool:
    value = row.get("downstream_trajectory_complete")
    if value not in (None, ""):
        return bool(safe_int(value))
    present_fields = safe_int(row.get("downstream_trajectory_present_fields"), -1)
    total_fields = safe_int(row.get("downstream_trajectory_total_fields"), -1)
    if present_fields >= 0 and total_fields > 0:
        return present_fields == total_fields
    return all(present(row.get(field)) for field in DOWNSTREAM_TRAJECTORY_FIELDS)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def component_value(row: dict[str, Any], component: str) -> float:
    return safe_float(row.get(COMPONENT_FIELDS[component]))


def all_component_values(row: dict[str, Any]) -> dict[str, float]:
    values = {component: component_value(row, component) for component in COMPONENT_FIELDS}
    for component, field in DETAIL_FIELDS.items():
        values[component] = safe_float(row.get(field))
    return values


def primary_loss_component(row: dict[str, Any]) -> str:
    if safe_float(row.get("realized_delta")) >= 0.0:
        return "non_loss"
    values = {component: component_value(row, component) for component in COMPONENT_FIELDS}
    negative = {component: value for component, value in values.items() if value < 0.0}
    if not negative:
        return "unexplained"
    return min(negative.items(), key=lambda item: (item[1], item[0]))[0]


def component_coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {**COMPONENT_FIELDS, **DETAIL_FIELDS, "terminal": "terminal_score_vs_baseline"}
    output = []
    total = len(rows)
    for name, field in fields.items():
        present_count = sum(1 for row in rows if present(row.get(field)))
        output.append(
            {
                "component": name,
                "field": field,
                "present_rows": present_count,
                "missing_rows": total - present_count,
                "present_rate": present_count / total if total else 0.0,
            }
        )
    return output


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    realized_delta = safe_float(row.get("realized_delta"))
    realized_loss = max(0.0, -realized_delta)
    out = {
        "source_log": row.get("source_log", ""),
        "config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed", ""),
        "hand_id": row.get("hand_id", ""),
        "seat": row.get("seat", ""),
        "seat_swap": row.get("seat_swap", ""),
        "candidate_index": row.get("candidate_index", ""),
        "baseline_index": row.get("baseline_index", ""),
        "recommended_training_use": row.get("recommended_training_use", ""),
        "downstream_trajectory_complete": int(downstream_complete(row)),
        "local_replay_bucket": row.get("local_replay_bucket", ""),
        "local_replay_label": row.get("local_replay_label", ""),
        "realized_delta": realized_delta,
        "realized_loss": realized_loss,
        "realized_loss_label": int(realized_delta < 0.0),
        "primary_loss_component": primary_loss_component(row),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "confirm_delta": safe_float(row.get("confirm_delta")),
        "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 9999),
        "terminal_score_vs_baseline": safe_float(row.get("terminal_score_vs_baseline")),
    }
    out.update({field: safe_float(row.get(field)) for field in COMPONENT_FIELDS.values()})
    out.update({field: safe_float(row.get(field)) for field in DETAIL_FIELDS.values()})
    return out


def summarize_subset(group_field: str, group_value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [safe_float(row.get("realized_loss")) for row in rows]
    realized = [safe_float(row.get("realized_delta")) for row in rows]
    out = {
        "group_field": group_field,
        "group_value": group_value,
        "rows": len(rows),
        "loss_rows": sum(safe_float(row.get("realized_delta")) < 0.0 for row in rows),
        "realized_delta_mean": mean(realized),
        "realized_loss_mean": mean(losses),
        "realized_loss_p90": percentile(losses, 90),
        "realized_loss_p95": percentile(losses, 95),
        "realized_loss_max": max(losses, default=0.0),
    }
    for component, field in COMPONENT_FIELDS.items():
        values = [safe_float(row.get(field)) for row in rows]
        out[f"{component}_delta_mean"] = mean(values)
        out[f"{component}_negative_rows"] = sum(value < 0.0 for value in values)
    for component, field in DETAIL_FIELDS.items():
        out[f"{component}_delta_mean"] = mean(safe_float(row.get(field)) for row in rows)
    return out


def breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [summarize_subset("overall", "all", rows)]
    for field in (
        "recommended_training_use",
        "primary_loss_component",
        "seat",
        "config_id",
        "local_replay_bucket",
        "local_replay_label",
    ):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            output.append(
                summarize_subset(field, value, [row for row in rows if str(row.get(field, "")) == value])
            )
    return output


def component_loss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for component in COMPONENT_FIELDS:
        subset = [row for row in rows if row.get("primary_loss_component") == component]
        output.append(summarize_subset("primary_loss_component", component, subset))
    for component in ("unexplained", "non_loss"):
        subset = [row for row in rows if row.get("primary_loss_component") == component]
        output.append(summarize_subset("primary_loss_component", component, subset))
    return output


def metric_rows(
    rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    total_rows: int | None = None,
) -> list[dict[str, Any]]:
    coverage = component_coverage_rows(raw_rows)
    missing_all = sum(1 for row in coverage if safe_int(row.get("present_rows")) == 0)
    component_counts = Counter(str(row.get("primary_loss_component")) for row in rows)
    raw_total = len(raw_rows) if total_rows is None else total_rows
    metrics: list[dict[str, Any]] = [
        {"metric": "input_paths", "value": len(sources)},
        {"metric": "input_rows", "value": sum(safe_int(row.get("rows")) for row in sources)},
        {"metric": "deduped_rows", "value": raw_total},
        {"metric": "analysis_rows", "value": len(rows)},
        {"metric": "trajectory_incomplete_rows_excluded", "value": raw_total - len(rows)},
        {"metric": "loss_rows", "value": sum(safe_float(row.get("realized_delta")) < 0.0 for row in rows)},
        {
            "metric": "mean_realized_delta",
            "value": mean(safe_float(row.get("realized_delta")) for row in rows),
        },
        {
            "metric": "mean_realized_loss",
            "value": mean(safe_float(row.get("realized_loss")) for row in rows),
        },
        {"metric": "component_fields_total", "value": len(coverage)},
        {"metric": "component_fields_missing_everywhere", "value": missing_all},
    ]
    for component, count in sorted(component_counts.items()):
        metrics.append({"metric": f"primary_loss_component.{component}", "value": count})
    return metrics


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def top_component_losses(rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    raw_by_key = {
        (
            str(row.get("source_log", "")),
            str(row.get("config_id", "")),
            str(row.get("hand_seed", "")),
            str(row.get("seat", "")),
            str(row.get("candidate_index", "")),
            str(row.get("baseline_index", "")),
        ): row
        for row in raw_rows
    }
    selected = [row for row in rows if safe_float(row.get("realized_delta")) < 0.0]
    selected.sort(key=lambda row: safe_float(row.get("realized_loss")), reverse=True)
    output = []
    for row in selected[: max(0, top_n)]:
        key = (
            str(row.get("source_log", "")),
            str(row.get("config_id", "")),
            str(row.get("hand_seed", "")),
            str(row.get("seat", "")),
            str(row.get("candidate_index", "")),
            str(row.get("baseline_index", "")),
        )
        raw = raw_by_key.get(key, {})
        output.append(
            {
                **row,
                "component_values": all_component_values(raw),
                "hero_board": raw.get("hero_board"),
                "opponent_board": raw.get("opponent_board"),
                "cards_to_place": raw.get("cards_to_place"),
                "post_t2_candidate_board": raw.get("post_t2_candidate_board"),
                "post_t2_baseline_board": raw.get("post_t2_baseline_board"),
                "candidate_action": raw.get("candidate_action"),
                "baseline_action": raw.get("baseline_action"),
            }
        )
    return output


def write_summary(path: Path, metrics: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> None:
    values = {str(row["metric"]): row["value"] for row in metrics}
    missing = [row["field"] for row in coverage if safe_int(row.get("present_rows")) == 0]
    lines = [
        "# HU T2 Stage8c Trajectory Component Analysis",
        "",
        "This diagnostic explains realized candidate-vs-baseline deltas using terminal score components.",
        "It is not a production, P2, T1, or 50k-teacher approval artifact.",
        "",
        "## Counts",
        "",
        f"- rows: `{values.get('deduped_rows', 0)}`",
        f"- analysis rows: `{values.get('analysis_rows', 0)}`",
        f"- excluded incomplete rows: `{values.get('trajectory_incomplete_rows_excluded', 0)}`",
        f"- loss rows: `{values.get('loss_rows', 0)}`",
        f"- mean realized delta: `{float(values.get('mean_realized_delta', 0.0)):.4f}`",
        f"- component fields missing everywhere: `{values.get('component_fields_missing_everywhere', 0)} / {values.get('component_fields_total', 0)}`",
        "",
        "## Primary Loss Components",
        "",
    ]
    for row in metrics:
        metric = str(row.get("metric"))
        if metric.startswith("primary_loss_component."):
            lines.append(f"- `{metric.removeprefix('primary_loss_component.')}`: `{row.get('value')}`")
    lines.extend(
        [
            "",
            "## Missing Component Fields",
            "",
        ]
    )
    if missing:
        lines.append(f"- `{', '.join(missing)}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Component metrics are computed only from rows with complete downstream trajectory fields.",
            "- Use this after trajectory-enriched risk-data runs to identify whether losses are coming from foul, FL, royalty, line, or scoop swings.",
            "- If component coverage is missing, rerun the TopK evaluator with the trajectory instrumentation before training another risk head.",
            "- Production/P2 fixed status, T1, and 50k teacher generation remain `No-Go` from this artifact alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = target_paths(args.collection_dir, args.input_jsonl)
    raw_rows, sources = load_rows(paths)
    analysis_raw_rows = [row for row in raw_rows if downstream_complete(row)]
    rows = [normalize_row(row) for row in analysis_raw_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = component_coverage_rows(raw_rows)
    metrics = metric_rows(rows, raw_rows, sources, total_rows=len(raw_rows))

    write_csv(args.output_dir / "trajectory_component_metrics.csv", metrics)
    write_csv(args.output_dir / "trajectory_component_coverage.csv", coverage)
    write_csv(args.output_dir / "trajectory_component_breakdown.csv", breakdown_rows(rows))
    write_csv(args.output_dir / "trajectory_component_loss_breakdown.csv", component_loss_rows(rows))
    write_csv(args.output_dir / "trajectory_component_rows.csv", rows)
    write_jsonl(
        args.output_dir / "trajectory_component_top_losses.jsonl",
        top_component_losses(rows, analysis_raw_rows, args.top_n),
    )
    write_summary(args.output_dir / "trajectory_component_summary.md", metrics, coverage)
    manifest = {
        "schema": "hu_turn2_stage8c_trajectory_component_analysis_v1",
        "input_paths": [str(path) for path in paths],
        "output_dir": str(args.output_dir),
        "rows": len(raw_rows),
        "analysis_rows": len(rows),
        "trajectory_incomplete_rows_excluded": len(raw_rows) - len(rows),
        "loss_rows": sum(safe_float(row.get("realized_delta")) < 0.0 for row in rows),
        "component_fields_missing_everywhere": sum(1 for row in coverage if safe_int(row.get("present_rows")) == 0),
        "runtime_risk_integration": False,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    (args.output_dir / "trajectory_component_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
