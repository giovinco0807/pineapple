"""Analyze Stage9 HU T3 fired-state tails for simple veto design.

This script consumes joint-exact teacher JSONL produced from runtime-fired T3
states. It does not train or promote a runtime. It reports whether simple
runtime-available fields can remove large negative fired tails while preserving
positive fired-state EV.

The delta definition intentionally matches analyze_hu_turn3_joint_exact_references:
recorded HU candidate selection (`selection.hu_index`) minus recorded baseline
(`selection.baseline_index`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ofc_regular.cards import card_rank


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _action_score_by_index(record: dict[str, Any], index: int | None) -> float | None:
    if index is None:
        return None
    for action in record.get("actions", []):
        if action.get("original_index") == index:
            score = action.get("joint_ev", action.get("score"))
            if score is None:
                return None
            return float(score)
    return None


def _percentile(values: list[float], q: float) -> float | None:
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


def _summarize_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(r["delta_vs_baseline"]) for r in rows]
    losses = [-d for d in deltas if d < 0.0]
    return {
        "count": len(rows),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "median_delta": _percentile(deltas, 0.5),
        "negative_count": sum(1 for d in deltas if d < 0.0),
        "negative_rate": (sum(1 for d in deltas if d < 0.0) / len(deltas)) if deltas else None,
        "hard_negative_le_m025_count": sum(1 for d in deltas if d <= -0.25),
        "p90_loss": _percentile(losses, 0.90),
        "p95_loss": _percentile(losses, 0.95),
        "p99_loss": _percentile(losses, 0.99),
        "worst_delta": min(deltas) if deltas else None,
        "best_delta": max(deltas) if deltas else None,
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_veto_csv(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    thresholds = {
        "predicted_margin_vs_baseline": [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0],
        "reference_margin": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "model_score": [2.0, 4.0, 6.0, 8.0, 10.0],
    }
    base = _summarize_deltas(rows)
    base_count = max(1, len(rows))
    for field, values in thresholds.items():
        for threshold in values:
            kept = [r for r in rows if r.get(field) is not None and float(r[field]) >= threshold]
            vetoed = [r for r in rows if r not in kept]
            if not kept:
                continue
            kept_summary = _summarize_deltas(kept)
            vetoed_summary = _summarize_deltas(vetoed)
            candidates.append(
                {
                    "rule": f"{field}>={threshold}",
                    "field": field,
                    "threshold": threshold,
                    "kept_count": len(kept),
                    "kept_rate": len(kept) / base_count,
                    "vetoed_count": len(vetoed),
                    "kept_mean_delta": kept_summary["mean_delta"],
                    "kept_negative_rate": kept_summary["negative_rate"],
                    "kept_hard_negative_le_m025_count": kept_summary["hard_negative_le_m025_count"],
                    "kept_worst_delta": kept_summary["worst_delta"],
                    "vetoed_mean_delta": vetoed_summary["mean_delta"],
                    "vetoed_hard_negative_le_m025_count": vetoed_summary["hard_negative_le_m025_count"],
                    "base_mean_delta": base["mean_delta"],
                    "base_hard_negative_le_m025_count": base["hard_negative_le_m025_count"],
                }
            )

    candidates.sort(
        key=lambda r: (
            -(float(r["kept_mean_delta"]) if r["kept_mean_delta"] is not None else -999.0),
            int(r["kept_hard_negative_le_m025_count"]),
            -int(r["kept_count"]),
        )
    )
    fieldnames = [
        "rule",
        "kept_count",
        "kept_rate",
        "vetoed_count",
        "kept_mean_delta",
        "kept_negative_rate",
        "kept_hard_negative_le_m025_count",
        "kept_worst_delta",
        "vetoed_mean_delta",
        "vetoed_hard_negative_le_m025_count",
        "base_mean_delta",
        "base_hard_negative_le_m025_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return candidates


def _row_capacity(board: dict[str, list[str]], row: str) -> int:
    return {"top": 3, "middle": 5, "bottom": 5}[row] - len(board.get(row, []))


def _action_fills(record: dict[str, Any], action_index: int | None) -> dict[str, bool]:
    if action_index is None:
        return {"top": False, "middle": False, "bottom": False}
    action = None
    for candidate in record.get("actions", []):
        if candidate.get("original_index") == action_index:
            action = candidate
            break
    if action is None:
        return {"top": False, "middle": False, "bottom": False}
    board = record.get("board", {})
    placements = action.get("placements", [])
    row_counts = Counter(row for _card, row in placements)
    return {
        row: row_counts[row] >= _row_capacity(board, row)
        for row in ("top", "middle", "bottom")
    }


def _find_action(record: dict[str, Any], action_index: int | None) -> dict[str, Any] | None:
    if action_index is None:
        return None
    for action in record.get("actions", []):
        if action.get("original_index") == action_index:
            return action
    return None


def _rank_value(card: Any) -> int:
    if not card:
        return 0
    try:
        return card_rank(str(card))
    except ValueError:
        return 0


def _action_profile(record: dict[str, Any], action_index: int | None) -> dict[str, Any]:
    action = _find_action(record, action_index)
    if action is None:
        return {
            "places_top_count": 0,
            "places_middle_count": 0,
            "places_bottom_count": 0,
            "discard_count": 0,
            "discard_rank_max": 0,
            "placed_rank_sum": 0,
            "placed_rank_max": 0,
            "filled_row_count": 0,
        }
    placements = action.get("placements", []) or []
    row_counts = Counter(str(row) for _card, row in placements)
    placed_ranks = [_rank_value(card) for card, _row in placements]
    discard_ranks = [_rank_value(card) for card in (action.get("discards", []) or [])]
    fills = _action_fills(record, action_index)
    return {
        "places_top_count": row_counts["top"],
        "places_middle_count": row_counts["middle"],
        "places_bottom_count": row_counts["bottom"],
        "discard_count": len(discard_ranks),
        "discard_rank_max": max(discard_ranks) if discard_ranks else 0,
        "placed_rank_sum": sum(placed_ranks),
        "placed_rank_max": max(placed_ranks) if placed_ranks else 0,
        "filled_row_count": sum(1 for full in fills.values() if full),
    }


def _build_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(input_path):
        selection = record.get("selection", {})
        baseline_index = selection.get("baseline_index")
        hu_index = selection.get("hu_index")
        baseline_score = _action_score_by_index(record, baseline_index)
        hu_score = _action_score_by_index(record, hu_index)
        if baseline_score is None or hu_score is None:
            continue
        delta = hu_score - baseline_score
        candidate_fills = _action_fills(record, hu_index)
        baseline_fills = _action_fills(record, baseline_index)
        candidate_profile = _action_profile(record, hu_index)
        baseline_profile = _action_profile(record, baseline_index)
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "state_id": (record.get("source_state") or {}).get("state_id"),
                "seed": (record.get("source_state") or {}).get("seed"),
                "seat": record.get("seat"),
                "baseline_index": baseline_index,
                "hu_index": hu_index,
                "predicted_margin_vs_baseline": selection.get("predicted_margin_vs_baseline"),
                "reference_margin": selection.get("reference_margin"),
                "model_score": selection.get("model_score"),
                "best_score": record.get("best_score"),
                "baseline_score": baseline_score,
                "hu_score": hu_score,
                "delta_vs_baseline": delta,
                "regret_vs_best": float(record.get("best_score", hu_score)) - hu_score,
                "legal_action_count": record.get("legal_action_count"),
                "hu_top_full": candidate_fills["top"],
                "hu_middle_full": candidate_fills["middle"],
                "hu_bottom_full": candidate_fills["bottom"],
                "baseline_top_full": baseline_fills["top"],
                "baseline_middle_full": baseline_fills["middle"],
                "baseline_bottom_full": baseline_fills["bottom"],
                "hu_places_top_count": candidate_profile["places_top_count"],
                "hu_places_middle_count": candidate_profile["places_middle_count"],
                "hu_places_bottom_count": candidate_profile["places_bottom_count"],
                "baseline_places_top_count": baseline_profile["places_top_count"],
                "baseline_places_middle_count": baseline_profile["places_middle_count"],
                "baseline_places_bottom_count": baseline_profile["places_bottom_count"],
                "hu_discard_count": candidate_profile["discard_count"],
                "baseline_discard_count": baseline_profile["discard_count"],
                "hu_discard_rank_max": candidate_profile["discard_rank_max"],
                "baseline_discard_rank_max": baseline_profile["discard_rank_max"],
                "hu_placed_rank_sum": candidate_profile["placed_rank_sum"],
                "baseline_placed_rank_sum": baseline_profile["placed_rank_sum"],
                "hu_placed_rank_max": candidate_profile["placed_rank_max"],
                "baseline_placed_rank_max": baseline_profile["placed_rank_max"],
                "hu_fills_more_rows_than_baseline": candidate_profile["filled_row_count"] > baseline_profile["filled_row_count"],
                "hu_discards_higher_rank_than_baseline": candidate_profile["discard_rank_max"] > baseline_profile["discard_rank_max"],
            }
        )
    return rows


def _write_summary(path: Path, input_path: Path, rows: list[dict[str, Any]], veto_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summarize_deltas(rows)
    seat_counts = Counter(r["seat"] for r in rows)
    seat_lines = []
    for seat in sorted(seat_counts):
        seat_summary = _summarize_deltas([r for r in rows if r["seat"] == seat])
        seat_lines.append(
            f"- {seat}: n={seat_counts[seat]}, mean_delta={_format_float(seat_summary['mean_delta'])}, "
            f"negative_rate={_format_float(seat_summary['negative_rate'])}"
        )
    top_veto = veto_rows[:10]
    veto_lines = [
        "| rule | kept | kept mean | kept neg rate | kept HN | worst | vetoed HN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_veto:
        veto_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rule"]),
                    str(row["kept_count"]),
                    _format_float(row["kept_mean_delta"]),
                    _format_float(row["kept_negative_rate"]),
                    str(row["kept_hard_negative_le_m025_count"]),
                    _format_float(row["kept_worst_delta"]),
                    str(row["vetoed_hard_negative_le_m025_count"]),
                ]
            )
            + " |"
        )

    text = "\n".join(
        [
            "# HU Turn3 Stage9 Tail Veto Analysis",
            "",
            f"Input: `{input_path}`",
            "",
            "## Fired-State Baseline",
            "",
            f"- rows: {summary['count']}",
            f"- mean delta vs baseline: {_format_float(summary['mean_delta'])}",
            f"- median delta: {_format_float(summary['median_delta'])}",
            f"- negative rate: {_format_float(summary['negative_rate'])}",
            f"- hard negatives <= -0.25: {summary['hard_negative_le_m025_count']}",
            f"- p95 loss among negative fires: {_format_float(summary['p95_loss'])}",
            f"- worst delta: {_format_float(summary['worst_delta'])}",
            f"- best delta: {_format_float(summary['best_delta'])}",
            "",
            "## Seat Split",
            "",
            *seat_lines,
            "",
            "## Simple One-Field Veto Candidates",
            "",
            *veto_lines,
            "",
            "## Interpretation",
            "",
            "This is a diagnostic only. Any rule shown here must be re-tested through full seat-swap and fired-state replay before it can be treated as a runtime candidate.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rows-output", required=True, type=Path)
    parser.add_argument("--veto-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    rows = _build_rows(args.input)
    _write_rows_csv(args.rows_output, rows)
    veto_rows = _write_veto_csv(args.veto_output, rows)
    _write_summary(args.summary_output, args.input, rows, veto_rows)


if __name__ == "__main__":
    main()
