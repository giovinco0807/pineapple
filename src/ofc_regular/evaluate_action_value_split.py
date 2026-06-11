"""Evaluate action-value models on a deterministic train/holdout split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .turn3_model import evaluate_model, load_action_value_model, read_teacher_samples, split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True, help="teacher JSONL")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_teacher_samples(args.validation, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no validation samples")
    train_samples, holdout_samples = split_samples(
        samples,
        holdout_fraction=args.holdout,
        seed=args.seed,
    )
    results = []
    for model_path in args.model:
        model = load_action_value_model(model_path)
        results.append(
            {
                "model": str(model_path),
                "validation": str(args.validation),
                "holdout_fraction": args.holdout,
                "seed": args.seed,
                "total_samples": len(samples),
                "train": evaluate_model(model, train_samples),
                "holdout": evaluate_model(model, holdout_samples),
            }
        )
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
