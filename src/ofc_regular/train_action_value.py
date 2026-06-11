"""Train a regular OFC action-value ridge model from teacher JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .turn3_model import (
    evaluate_model,
    read_teacher_samples,
    split_samples,
    train_ridge_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="teacher JSONL")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no teacher samples")
    train_samples, holdout_samples = split_samples(
        samples,
        holdout_fraction=args.holdout,
        seed=args.seed,
    )
    if not train_samples:
        raise SystemExit("no training samples after split")

    model = train_ridge_model(train_samples, l2=args.l2)
    model.save(args.model_output)

    metrics = {
        "input": str(args.input),
        "model_output": str(args.model_output),
        "total_samples": len(samples),
        "train": evaluate_model(model, train_samples),
        "holdout": evaluate_model(model, holdout_samples),
        "l2": args.l2,
        "seed": args.seed,
    }
    print(json.dumps(metrics, indent=2))

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
