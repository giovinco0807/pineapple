"""Aggregate GCP shards from direct HU Fantasyland EV estimation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _find_result_dirs(input_dir: Path) -> list[Path]:
    direct = input_dir / "fl_ev_direct_summary.json"
    if direct.exists():
        return [input_dir]
    results_dir = input_dir / "results"
    if not results_dir.exists():
        results_dir = input_dir
    return sorted(
        path
        for path in results_dir.rglob("*")
        if path.is_dir() and (path / "fl_ev_direct_summary.json").exists()
    )


def aggregate(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    shard_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []

    for result_dir in _find_result_dirs(input_dir):
        summary = json.loads((result_dir / "fl_ev_direct_summary.json").read_text(encoding="utf-8"))
        iterations = _read_csv(result_dir / "fl_ev_direct_iterations.csv")
        final_iteration = iterations[-1] if iterations else {}
        shard = result_dir.name
        solved = _safe_float(final_iteration, "solved", _safe_float(summary, "trials_per_iteration", 0.0))
        final_ev = _safe_float(summary, "final_fl_ev_14", _safe_float(final_iteration, "next_ev", 0.0))
        final_se = _safe_float(summary, "final_std_error", _safe_float(final_iteration, "std_error", 0.0))
        row = {
            "shard": shard,
            "final_fl_ev_14": final_ev,
            "final_std_error": final_se,
            "final_ci95_low": _safe_float(summary, "final_ci95_low", _safe_float(final_iteration, "ci95_low", 0.0)),
            "final_ci95_high": _safe_float(summary, "final_ci95_high", _safe_float(final_iteration, "ci95_high", 0.0)),
            "solved": solved,
            "iterations_completed": int(_safe_float(summary, "iterations_completed", len(iterations))),
            "elapsed_seconds": _safe_float(summary, "elapsed_seconds", 0.0),
            "hero_stay_rate": _safe_float(final_iteration, "hero_stay_rate", 0.0),
            "opponent_fl_entry_rate": _safe_float(final_iteration, "opponent_fl_entry_rate", 0.0),
            "opponent_bust_rate": _safe_float(final_iteration, "opponent_bust_rate", 0.0),
            "hero_avg_royalty": _safe_float(final_iteration, "hero_avg_royalty", 0.0),
            "opponent_avg_royalty": _safe_float(final_iteration, "opponent_avg_royalty", 0.0),
            "path": str(result_dir),
        }
        shard_rows.append(row)
        for iteration in iterations:
            iteration_row = dict(iteration)
            iteration_row["shard"] = shard
            iteration_rows.append(iteration_row)

    total_solved = sum(float(row["solved"]) for row in shard_rows)
    if total_solved > 0:
        weighted_ev = sum(float(row["final_fl_ev_14"]) * float(row["solved"]) for row in shard_rows) / total_solved
        weighted_stay = sum(float(row["hero_stay_rate"]) * float(row["solved"]) for row in shard_rows) / total_solved
        weighted_opp_entry = (
            sum(float(row["opponent_fl_entry_rate"]) * float(row["solved"]) for row in shard_rows) / total_solved
        )
        weighted_hero_royalty = (
            sum(float(row["hero_avg_royalty"]) * float(row["solved"]) for row in shard_rows) / total_solved
        )
        weighted_opp_royalty = (
            sum(float(row["opponent_avg_royalty"]) * float(row["solved"]) for row in shard_rows) / total_solved
        )
        variance = sum(
            (float(row["solved"]) / total_solved) ** 2 * float(row["final_std_error"]) ** 2
            for row in shard_rows
        )
        aggregate_se = math.sqrt(max(variance, 0.0))
    else:
        weighted_ev = weighted_stay = weighted_opp_entry = weighted_hero_royalty = weighted_opp_royalty = 0.0
        aggregate_se = 0.0

    summary = {
        "status": "complete" if shard_rows else "no_results",
        "shards": len(shard_rows),
        "total_solved": total_solved,
        "fl_ev_14_weighted_mean": weighted_ev,
        "std_error": aggregate_se,
        "ci95_low": weighted_ev - 1.96 * aggregate_se,
        "ci95_high": weighted_ev + 1.96 * aggregate_se,
        "hero_stay_rate_weighted": weighted_stay,
        "opponent_fl_entry_rate_weighted": weighted_opp_entry,
        "hero_avg_royalty_weighted": weighted_hero_royalty,
        "opponent_avg_royalty_weighted": weighted_opp_royalty,
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in shard_rows),
        "max_elapsed_seconds": max((float(row["elapsed_seconds"]) for row in shard_rows), default=0.0),
    }
    return summary, shard_rows, iteration_rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Direct HU Fantasyland EV Aggregate",
        "",
        f"- status: `{summary['status']}`",
        f"- shards: `{summary['shards']}`",
        f"- total_solved: `{summary['total_solved']:.0f}`",
        f"- FL EV 14 weighted mean: `{summary['fl_ev_14_weighted_mean']:.6f}`",
        f"- 95% CI: `[{summary['ci95_low']:.6f}, {summary['ci95_high']:.6f}]`",
        f"- std_error: `{summary['std_error']:.6f}`",
        f"- hero stay rate: `{summary['hero_stay_rate_weighted']:.6f}`",
        f"- opponent FL entry rate: `{summary['opponent_fl_entry_rate_weighted']:.6f}`",
        f"- hero avg royalty: `{summary['hero_avg_royalty_weighted']:.6f}`",
        f"- opponent avg royalty: `{summary['opponent_avg_royalty_weighted']:.6f}`",
        "",
        "This is a direct FL-vs-normal HU simulation estimate. It is not the old shortcut formula.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, shard_rows, iteration_rows = aggregate(args.input_dir)
    (args.output_dir / "fl_ev_direct_aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "fl_ev_direct_aggregate_summary.md", summary)
    _write_csv(args.output_dir / "fl_ev_direct_shards.csv", shard_rows)
    _write_csv(args.output_dir / "fl_ev_direct_iterations.csv", iteration_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
