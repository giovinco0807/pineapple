"""Analyze HU Turn3 override trace JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="override trace JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--thresholds",
        default="8,10,12,15,20,25",
        help="Comma-separated predicted-margin thresholds to sweep.",
    )
    parser.add_argument(
        "--paired-seeds",
        type=int,
        required=True,
        help="Number of paired seeds in the trace run. Used for EV/hand approximation.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def analyze_thresholds(
    rows: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
    paired_seeds: int,
) -> list[dict[str, float]]:
    if paired_seeds <= 0:
        raise ValueError("paired_seeds must be positive")

    results: list[dict[str, float]] = []
    for threshold in thresholds:
        deltas: list[float] = []
        terminal_scores: list[float] = []
        for row in rows:
            margin = row.get("predicted_margin")
            if margin is None or float(margin) < threshold:
                continue
            deltas.append(float(row.get("counterfactual_delta_vs_baseline", 0.0)))
            if row.get("candidate_terminal_score") is not None:
                terminal_scores.append(float(row["candidate_terminal_score"]))

        wins = sum(1 for delta in deltas if delta > 1e-9)
        losses = sum(1 for delta in deltas if delta < -1e-9)
        ties = len(deltas) - wins - losses
        delta_sum = sum(deltas)
        terminal_sum = sum(terminal_scores)
        results.append(
            {
                "threshold": float(threshold),
                "overrides": float(len(deltas)),
                "override_rate": float(len(deltas) / (paired_seeds * 2.0)),
                "delta_sum": float(delta_sum),
                "delta_avg": float(delta_sum / len(deltas)) if deltas else 0.0,
                "approx_ev_per_hand": float(delta_sum / 2.0 / paired_seeds),
                "wins": float(wins),
                "losses": float(losses),
                "ties": float(ties),
                "terminal_score_avg": float(terminal_sum / len(terminal_scores))
                if terminal_scores
                else 0.0,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    rows = read_trace_rows(args.input)
    summary = {
        "input": str(args.input),
        "paired_seeds": args.paired_seeds,
        "trace_rows": len(rows),
        "thresholds": analyze_thresholds(
            rows,
            thresholds=thresholds,
            paired_seeds=args.paired_seeds,
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
