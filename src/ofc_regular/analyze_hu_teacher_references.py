"""Analyze HU Turn3 teacher labels against selection/reference actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

from .hu_self_play_teacher_data import annotate_hu_turn3_reference_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--positive-deltas",
        default="0.05,0.10,0.25,0.50,1.00",
        help="Comma-separated delta_best_vs_selection_hu thresholds to count.",
    )
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one threshold")
    return parsed


def iter_samples(path: Path, max_samples: int | None = None) -> Iterable[dict[str, Any]]:
    seen = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_samples is not None and seen >= max_samples:
                break
            seen += 1
            yield json.loads(line)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(values[0])
    if q >= 1.0:
        return float(values[-1])
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": float(len(sorted_values)),
        "mean": float(mean(sorted_values)),
        "median": float(median(sorted_values)),
        "p05": quantile(sorted_values, 0.05),
        "p25": quantile(sorted_values, 0.25),
        "p75": quantile(sorted_values, 0.75),
        "p95": quantile(sorted_values, 0.95),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


def ensure_references(sample: dict[str, Any]) -> dict[str, Any] | None:
    references = sample.get("reference_actions")
    if references is not None:
        return references
    selection = sample.get("selection")
    if selection is None:
        return None
    try:
        annotate_hu_turn3_reference_actions(sample, selection)
    except Exception:
        return None
    return sample.get("reference_actions")


def analyze_samples(samples: Iterable[dict[str, Any]], *, positive_deltas: Sequence[float]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    total = missing_reference = 0
    for sample in samples:
        total += 1
        references = ensure_references(sample)
        if not references or references.get("selection_hu") is None or references.get("baseline") is None:
            missing_reference += 1
            continue
        rows.append(
            {
                "score_gap": float(sample.get("score_gap", 0.0)),
                "predicted_margin_vs_baseline": float(
                    sample.get("selection", {}).get("predicted_margin_vs_baseline", 0.0)
                ),
                "delta_best_vs_selection_hu": float(
                    references.get("delta_best_vs_selection_hu", 0.0)
                ),
                "delta_best_vs_baseline": float(references.get("delta_best_vs_baseline", 0.0)),
                "delta_selection_hu_vs_baseline": float(
                    references.get("delta_selection_hu_vs_baseline", 0.0)
                ),
                "selection_hu_regret": float(references.get("selection_hu_regret", 0.0)),
                "baseline_regret": float(references.get("baseline_regret", 0.0)),
            }
        )

    delta_best_vs_selection = [row["delta_best_vs_selection_hu"] for row in rows]
    delta_best_vs_baseline = [row["delta_best_vs_baseline"] for row in rows]
    delta_selection_vs_baseline = [row["delta_selection_hu_vs_baseline"] for row in rows]
    score_gaps = [row["score_gap"] for row in rows]
    predicted_margins = [row["predicted_margin_vs_baseline"] for row in rows]
    return {
        "samples_read": total,
        "samples_with_references": len(rows),
        "missing_reference": missing_reference,
        "score_gap": summarize_values(score_gaps),
        "predicted_margin_vs_baseline": summarize_values(predicted_margins),
        "delta_best_vs_selection_hu": summarize_values(delta_best_vs_selection),
        "delta_best_vs_baseline": summarize_values(delta_best_vs_baseline),
        "delta_selection_hu_vs_baseline": summarize_values(delta_selection_vs_baseline),
        "positive_delta_counts": [
            {
                "delta_best_vs_selection_hu_min": float(threshold),
                "samples": float(sum(1 for value in delta_best_vs_selection if value >= threshold)),
                "fraction": float(
                    sum(1 for value in delta_best_vs_selection if value >= threshold)
                    / max(len(delta_best_vs_selection), 1)
                ),
            }
            for threshold in positive_deltas
        ],
        "stage3_selection_teacher_net": {
            "positive": float(sum(1 for value in delta_selection_vs_baseline if value > 1e-9)),
            "negative": float(sum(1 for value in delta_selection_vs_baseline if value < -1e-9)),
            "zero": float(sum(1 for value in delta_selection_vs_baseline if abs(value) <= 1e-9)),
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    positive_deltas = parse_float_list(args.positive_deltas)
    summary = {
        "input": str(args.input),
        "max_samples": args.max_samples,
        **analyze_samples(iter_samples(args.input, args.max_samples), positive_deltas=positive_deltas),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
