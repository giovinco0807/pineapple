"""Evaluate saved action-value models on a teacher JSONL set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .turn3_model import evaluate_model, load_action_value_model, read_teacher_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_teacher_samples(
        args.validation,
        max_samples=args.max_samples,
        skip_samples=args.skip_samples,
    )
    if not samples:
        raise SystemExit("no validation samples")

    results = []
    for model_path in args.model:
        model = load_action_value_model(model_path)
        metrics = evaluate_model(model, samples)
        results.append(
            {
                "model": str(model_path),
                "validation": str(args.validation),
                **metrics,
            }
        )

    results.sort(key=lambda item: (item["avg_regret"], -item["top1_accuracy"], item["mse"]))
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
