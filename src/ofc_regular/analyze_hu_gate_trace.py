"""Analyze a trained HU Turn3 gate model on override traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hu_turn3_gate_model import load_hu_turn3_gate_model
from .train_hu_turn3_gate import (
    parse_thresholds,
    predict_probabilities,
    read_trace_rows,
    threshold_sweep,
    trace_row_to_training_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--paired-seeds", type=int, required=True)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_trace_rows([args.input], min_abs_delta=args.min_abs_delta)
    if not rows:
        raise SystemExit("no override rows")
    feature_rows = []
    deltas = []
    labels = []
    for row in rows:
        features, label, delta, _weight = trace_row_to_training_row(row)
        feature_rows.append(features)
        labels.append(label)
        deltas.append(delta)
    features = np.vstack(feature_rows).astype(np.float32)
    delta_array = np.asarray(deltas, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    model = load_hu_turn3_gate_model(args.gate_model)
    probabilities = predict_probabilities(model, features)
    selected = probabilities >= 0.5
    summary = {
        "input": str(args.input),
        "gate_model": str(args.gate_model),
        "paired_seeds": args.paired_seeds,
        "trace_rows": len(rows),
        "positive_rows": int(label_array.sum()),
        "negative_rows": int(label_array.size - label_array.sum()),
        "accuracy_at_0_5": float(np.mean((probabilities >= 0.5) == label_array)),
        "selected_delta_sum_at_0_5": float(delta_array[selected].sum(dtype=np.float64)),
        "thresholds": threshold_sweep(
            probabilities,
            delta_array,
            thresholds=parse_thresholds(args.thresholds),
            paired_seeds=args.paired_seeds,
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
