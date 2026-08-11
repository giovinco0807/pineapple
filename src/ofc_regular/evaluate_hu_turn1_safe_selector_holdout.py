"""Evaluate a fixed HU Turn1 safe selector on an untouched high-MC holdout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .hu_turn1_safe_selector import (
    load_hu_turn1_safe_selector_model,
    score_hu_turn1_safe_selector,
)
from .train_hu_turn1_safe_override_selector import (
    read_jsonl,
    stage10_mc32_candidate_delta,
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else 0.0


def summarize_holdout(
    scores: Iterable[float],
    deltas: Iterable[float],
    *,
    threshold: float,
    accept_delta: float = 1.0,
    parent_runtime_fire_rate: float | None = None,
) -> dict[str, Any]:
    score_array = np.asarray(tuple(scores), dtype=np.float64)
    delta_array = np.asarray(tuple(deltas), dtype=np.float64)
    if score_array.shape != delta_array.shape:
        raise ValueError("scores and deltas must have the same shape")
    if score_array.ndim != 1:
        raise ValueError("scores and deltas must be one-dimensional")
    if not np.all(np.isfinite(score_array)) or not np.all(np.isfinite(delta_array)):
        raise ValueError("scores and deltas must be finite")
    if parent_runtime_fire_rate is not None and not 0.0 <= parent_runtime_fire_rate <= 1.0:
        raise ValueError("parent_runtime_fire_rate must be in [0, 1]")

    fired_mask = score_array >= threshold
    fired = delta_array[fired_mask]
    losses = -fired[fired < 0.0]
    fired_count = int(fired.size)
    mean_delta = float(np.mean(fired)) if fired_count else 0.0
    median_delta = float(np.median(fired)) if fired_count else 0.0
    standard_error = (
        float(np.std(fired, ddof=1) / math.sqrt(fired_count))
        if fired_count > 1
        else 0.0
    )
    selector_fire_rate = float(np.mean(fired_mask)) if score_array.size else 0.0
    estimated_runtime_fire_rate = (
        parent_runtime_fire_rate * selector_fire_rate
        if parent_runtime_fire_rate is not None
        else None
    )

    return {
        "rows": int(score_array.size),
        "threshold": float(threshold),
        "accept_delta": float(accept_delta),
        "fires": fired_count,
        "selector_fire_rate_within_runtime_fires": selector_fire_rate,
        "mean_high_mc_delta": mean_delta,
        "median_high_mc_delta": median_delta,
        "high_mc_delta_standard_error": standard_error,
        "high_mc_delta_ci95_low": mean_delta - 1.96 * standard_error,
        "high_mc_delta_ci95_high": mean_delta + 1.96 * standard_error,
        "positive_count": int(np.sum(fired > 0.0)),
        "accept_positive_count": int(np.sum(fired >= accept_delta)),
        "gray_positive_count": int(np.sum((fired > 0.0) & (fired < accept_delta))),
        "false_positive_count": int(np.sum(fired < 0.0)),
        "false_positive_rate": float(np.mean(fired < 0.0)) if fired_count else 0.0,
        "p90_loss": _quantile(losses, 0.90),
        "p95_loss": _quantile(losses, 0.95),
        "p99_loss": _quantile(losses, 0.99),
        "max_loss": float(np.max(losses)) if losses.size else 0.0,
        "score_mean": float(np.mean(score_array)) if score_array.size else 0.0,
        "score_p50": _quantile(score_array, 0.50),
        "score_p90": _quantile(score_array, 0.90),
        "score_max": float(np.max(score_array)) if score_array.size else 0.0,
        "parent_runtime_fire_rate": parent_runtime_fire_rate,
        "estimated_runtime_fire_rate": estimated_runtime_fire_rate,
        "estimated_ev_per_decision": (
            estimated_runtime_fire_rate * mean_delta
            if estimated_runtime_fire_rate is not None
            else None
        ),
    }


def evaluate_holdout(
    rows: list[dict[str, Any]],
    model: Any,
    *,
    threshold: float,
    accept_delta: float = 1.0,
    parent_runtime_fire_rate: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores: list[float] = []
    deltas: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        score = float(score_hu_turn1_safe_selector(model, row))
        delta = float(stage10_mc32_candidate_delta(row))
        if not math.isfinite(score) or not math.isfinite(delta):
            raise ValueError(f"row {row_index} produced a non-finite score or delta")
        selected = score >= threshold
        scores.append(score)
        deltas.append(delta)
        detail_rows.append(
            {
                "row_index": row_index,
                "hand_seed": row.get("hand_seed", ""),
                "paired_index": row.get("paired_index", ""),
                "seat": row.get("seat", ""),
                "candidate_action_index": row.get(
                    "candidate_action_index", row.get("runtime_candidate_action_index", "")
                ),
                "safe_probability": score,
                "selected": selected,
                "high_mc_delta": delta,
                "label": (
                    "positive"
                    if delta >= accept_delta
                    else "hard_negative"
                    if delta <= 0.0
                    else "gray"
                ),
            }
        )
    summary = summarize_holdout(
        scores,
        deltas,
        threshold=threshold,
        accept_delta=accept_delta,
        parent_runtime_fire_rate=parent_runtime_fire_rate,
    )
    return summary, detail_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    lines = [
        "# HU T1 Safe Selector Fixed Holdout",
        "",
        f"- input: `{metadata['input']}`",
        f"- model: `{metadata['model']}`",
        f"- feature mode: `{metadata['feature_mode']}`",
        f"- threshold: `{summary['threshold']:.3f}`",
        f"- rows: `{summary['rows']}`",
        f"- fires: `{summary['fires']}`",
        f"- selector fire rate within parent fires: `{summary['selector_fire_rate_within_runtime_fires']:.4f}`",
        f"- mean high-MC delta: `{summary['mean_high_mc_delta']:+.4f}`",
        f"- CI95: `[{summary['high_mc_delta_ci95_low']:+.4f}, {summary['high_mc_delta_ci95_high']:+.4f}]`",
        f"- false positives: `{summary['false_positive_count']}` (`{summary['false_positive_rate']:.4f}`)",
        f"- p95 / max loss: `{summary['p95_loss']:.4f}` / `{summary['max_loss']:.4f}`",
    ]
    if summary["estimated_ev_per_decision"] is not None:
        lines.extend(
            [
                f"- estimated runtime fire rate: `{summary['estimated_runtime_fire_rate']:.6f}`",
                f"- estimated EV/decision: `{summary['estimated_ev_per_decision']:+.6f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one locked HU Turn1 safe-selector threshold on a high-MC holdout."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--accept-delta", type=float, default=1.0)
    parser.add_argument("--parent-runtime-fire-rate", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    model = load_hu_turn1_safe_selector_model(args.model)
    summary, detail_rows = evaluate_holdout(
        rows,
        model,
        threshold=args.threshold,
        accept_delta=args.accept_delta,
        parent_runtime_fire_rate=args.parent_runtime_fire_rate,
    )
    metadata = {
        "schema": "hu_turn1_safe_selector_fixed_holdout_v1",
        "input": str(args.input),
        "model": str(args.model),
        "feature_mode": model.get("feature_mode", "") if isinstance(model, dict) else "",
        "threshold_locked_before_holdout": True,
        "threshold_search_performed": False,
    }
    output = {**metadata, **summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "holdout_rows.csv", detail_rows)
    _write_markdown(args.output_dir / "summary.md", summary, metadata)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
