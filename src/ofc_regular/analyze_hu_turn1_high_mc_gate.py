"""Audit HU Turn1 runtime gates against independent high-MC action values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    placements = sorted((str(card), str(row)) for card, row in action.get("placements", ()) or ())
    discards = sorted(str(card) for card in action.get("discards", ()) or ())
    return json.dumps(
        {"placements": placements, "discards": discards},
        sort_keys=True,
        separators=(",", ":"),
    )


def action_index(action: dict[str, Any]) -> int | None:
    raw = action.get("action_index", action.get("original_index"))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def find_action(
    row: dict[str, Any],
    *,
    index_fields: tuple[str, ...],
    action_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    wanted_index: int | None = None
    for field in index_fields:
        raw = row.get(field)
        if raw is None:
            continue
        try:
            wanted_index = int(raw)
        except (TypeError, ValueError):
            pass
        if wanted_index is not None:
            break
    actions = [action for action in row.get("actions", ()) or () if isinstance(action, dict)]
    if wanted_index is not None:
        for action in actions:
            if action_index(action) == wanted_index:
                return action

    wanted_signature = ""
    for field in action_fields:
        wanted_signature = action_signature(row.get(field))
        if wanted_signature:
            break
    if wanted_signature:
        for action in actions:
            if action_signature(action) == wanted_signature:
                return action
    return None


def action_value(action: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if action is None:
        return None, None
    ev = finite_float(action.get("ev", action.get("score")))
    se = finite_float(action.get("se", action.get("ev_standard_error")))
    return ev, se


def audit_row(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    candidate = find_action(
        row,
        index_fields=("runtime_candidate_action_index", "candidate_action_index"),
        action_fields=("runtime_candidate_action", "hu_turn1_action", "final_action"),
    )
    baseline = find_action(
        row,
        index_fields=("runtime_baseline_action_index", "baseline_action_index", "fallback_action_index"),
        action_fields=("runtime_baseline_action", "baseline_action", "fallback_action"),
    )
    candidate_ev, candidate_se = action_value(candidate)
    baseline_ev, baseline_se = action_value(baseline)
    mapped = candidate_ev is not None and baseline_ev is not None
    delta = candidate_ev - baseline_ev if mapped else None
    delta_se = None
    if candidate_se is not None and baseline_se is not None:
        delta_se = math.sqrt(candidate_se * candidate_se + baseline_se * baseline_se)
    confirm_delta = finite_float(row.get("confirm_delta"))
    confirm_se = finite_float(row.get("confirm_delta_se"))
    confirm_z = (
        confirm_delta / confirm_se
        if confirm_delta is not None and confirm_se is not None and confirm_se > 0.0
        else None
    )
    candidate_score = finite_float(row.get("candidate_score", row.get("runtime_candidate_score")))
    baseline_score = finite_float(row.get("fallback_score", row.get("runtime_baseline_score")))
    return {
        "row_index": row_index,
        "sample_id": row.get("sample_id"),
        "target_id": row.get("target_id"),
        "hand_seed": row.get("hand_seed", row.get("hand_id")),
        "seat": row.get("seat"),
        "state_key": row.get("state_key"),
        "action_mapping_ok": mapped,
        "candidate_action_index": action_index(candidate) if candidate else None,
        "baseline_action_index": action_index(baseline) if baseline else None,
        "candidate_ev": candidate_ev,
        "baseline_ev": baseline_ev,
        "high_mc_delta": delta,
        "candidate_se": candidate_se,
        "baseline_se": baseline_se,
        "high_mc_delta_se_independent": delta_se,
        "high_mc_lcb164": delta - 1.64 * delta_se if delta is not None and delta_se is not None else None,
        "high_mc_lcb196": delta - 1.96 * delta_se if delta is not None and delta_se is not None else None,
        "runtime_realized_delta": finite_float(row.get("realized_delta", row.get("realized_candidate_seat_delta"))),
        "predicted_margin": finite_float(row.get("hu_turn1_predicted_margin", row.get("runtime_predicted_margin"))),
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "model_score_delta": (
            candidate_score - baseline_score
            if candidate_score is not None and baseline_score is not None
            else None
        ),
        "stage_a_delta": finite_float(row.get("stage_a_delta")),
        "stage_a_delta_se": finite_float(row.get("stage_a_delta_se")),
        "confirm_delta": confirm_delta,
        "confirm_delta_se": confirm_se,
        "confirm_z": confirm_z,
        "future_samples": row.get("future_samples"),
    }


def pearson(rows: list[dict[str, Any]], x_name: str, y_name: str = "high_mc_delta") -> float | None:
    pairs = [
        (finite_float(row.get(x_name)), finite_float(row.get(y_name)))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def threshold_metric(
    rows: list[dict[str, Any]],
    *,
    gate: str,
    threshold: float | None,
) -> dict[str, Any]:
    mapped = [row for row in rows if row["action_mapping_ok"]]
    selected = mapped if threshold is None else [
        row for row in mapped
        if finite_float(row.get(gate)) is not None and float(row[gate]) >= threshold
    ]
    deltas = [float(row["high_mc_delta"]) for row in selected]
    losses = [-delta for delta in deltas if delta < 0.0]
    return {
        "gate": gate,
        "threshold": "all" if threshold is None else threshold,
        "fires": len(selected),
        "fire_rate_within_runtime_fires": len(selected) / len(mapped) if mapped else 0.0,
        "mean_high_mc_delta": float(np.mean(deltas)) if deltas else 0.0,
        "median_high_mc_delta": float(median(deltas)) if deltas else 0.0,
        "false_positive_count": sum(delta <= 0.0 for delta in deltas),
        "false_positive_rate": (
            sum(delta <= 0.0 for delta in deltas) / len(deltas) if deltas else 0.0
        ),
        "lcb164_positive_count": sum(float(row["high_mc_lcb164"]) > 0.0 for row in selected),
        "lcb196_positive_count": sum(float(row["high_mc_lcb196"]) > 0.0 for row in selected),
        "p95_loss": float(np.quantile(losses, 0.95)) if losses else 0.0,
        "max_loss": max(losses) if losses else 0.0,
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


def analyze(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audited = [audit_row(row, row_index=index) for index, row in enumerate(rows)]
    mapped = [row for row in audited if row["action_mapping_ok"]]
    deltas = [float(row["high_mc_delta"]) for row in mapped]
    threshold_rows = [threshold_metric(audited, gate="all", threshold=None)]
    grids = {
        "confirm_z": (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
        "confirm_delta": (1.0, 2.0, 3.0, 4.0, 5.0),
        "predicted_margin": (0.0, 0.02, 0.05, 0.10, 0.20, 0.50),
        "model_score_delta": (0.0, 0.02, 0.05, 0.10, 0.20, 0.50),
    }
    for gate, thresholds in grids.items():
        threshold_rows.extend(
            threshold_metric(audited, gate=gate, threshold=threshold) for threshold in thresholds
        )
    correlations = {
        name: pearson(mapped, name)
        for name in (
            "runtime_realized_delta",
            "predicted_margin",
            "model_score_delta",
            "stage_a_delta",
            "stage_a_delta_se",
            "confirm_delta",
            "confirm_delta_se",
            "confirm_z",
        )
    }
    summary = {
        "schema": "hu_turn1_high_mc_gate_audit_v1",
        "rows": len(rows),
        "mapped_rows": len(mapped),
        "mapping_failures": len(rows) - len(mapped),
        "future_sample_counts": sorted(
            {int(value) for value in (row.get("future_samples") for row in mapped) if value is not None}
        ),
        "high_mc_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "high_mc_delta_median": float(median(deltas)) if deltas else 0.0,
        "high_mc_delta_positive_rate": sum(delta > 0.0 for delta in deltas) / len(deltas) if deltas else 0.0,
        "high_mc_delta_gt1_rate": sum(delta >= 1.0 for delta in deltas) / len(deltas) if deltas else 0.0,
        "high_mc_lcb164_positive_rate": (
            sum(float(row["high_mc_lcb164"]) > 0.0 for row in mapped) / len(mapped) if mapped else 0.0
        ),
        "high_mc_lcb196_positive_rate": (
            sum(float(row["high_mc_lcb196"]) > 0.0 for row in mapped) / len(mapped) if mapped else 0.0
        ),
        "correlations_with_high_mc_delta": correlations,
    }
    return audited, threshold_rows, summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    correlations = summary["correlations_with_high_mc_delta"]
    lines = [
        "# HU T1 High-MC Gate Audit",
        "",
        f"- rows: `{summary['rows']}`",
        f"- mapped rows: `{summary['mapped_rows']}`",
        f"- mapping failures: `{summary['mapping_failures']}`",
        f"- mean high-MC delta: `{summary['high_mc_delta_mean']:.4f}`",
        f"- positive rate: `{summary['high_mc_delta_positive_rate']:.4f}`",
        f"- LCB196 positive rate: `{summary['high_mc_lcb196_positive_rate']:.4f}`",
        "",
        "## Correlations",
        "",
        "| runtime signal | Pearson r vs high-MC delta |",
        "|---|---:|",
    ]
    for name, value in correlations.items():
        rendered = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {rendered} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    audited, thresholds, summary = analyze(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "high_mc_rows.csv", audited)
    write_csv(args.output_dir / "runtime_threshold_grid.csv", thresholds)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
