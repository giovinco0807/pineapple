"""Train a HU-aware Turn3 action-value model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from .hu_turn3_model import (
    HuSklearnActionValueModel,
    evaluate_model,
    read_teacher_samples,
    samples_to_matrix,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=("hgb", "extra_trees", "random_forest"), default="hgb")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help=(
            "Per-sample source weight applied to all action rows from that teacher sample. "
            "May be repeated or comma-separated; unspecified sources use 1.0."
        ),
    )
    return parser.parse_args()


def build_estimator(args: argparse.Namespace):
    if args.model_type == "hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
            early_stopping=True,
        )
    if args.model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
    raise ValueError(f"unsupported model type: {args.model_type}")


def parse_source_weights(entries: Sequence[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        for part in entry.split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"source weight must be SOURCE=WEIGHT, got {item!r}")
            source, raw_weight = item.split("=", 1)
            source = source.strip()
            if not source:
                raise ValueError("source weight source must not be empty")
            try:
                weight = float(raw_weight)
            except ValueError as exc:
                raise ValueError(f"invalid source weight for {source!r}: {raw_weight!r}") from exc
            if weight < 0.0:
                raise ValueError(f"source weight for {source!r} must be non-negative")
            weights[source] = weight
    return weights


def build_action_source_weights(
    samples: Sequence[dict],
    source_weights: dict[str, float],
) -> np.ndarray | None:
    if not source_weights:
        return None
    weights: list[float] = []
    for sample in samples:
        weight = source_weights.get(sample_source(sample), 1.0)
        weights.extend([weight] * len(sample.get("actions", ())))
    if not weights:
        raise ValueError("no action rows to weight")
    action_weights = np.asarray(weights, dtype=np.float64)
    if not np.any(action_weights > 0.0):
        raise ValueError("source weights zeroed every training action")
    return action_weights


def sample_source(sample: dict) -> str:
    return str(sample.get("source") or sample.get("label_source") or sample.get("profile") or "unknown")


def source_counts(samples: Sequence[dict]) -> dict[str, int]:
    return dict(Counter(sample_source(sample) for sample in samples))


def weighted_action_summary(weights: np.ndarray | None) -> dict[str, float]:
    if weights is None:
        return {}
    return {
        "train_action_weight_sum": float(weights.sum(dtype=np.float64)),
        "train_action_weight_mean": float(weights.mean(dtype=np.float64)),
        "train_action_weight_min": float(weights.min()),
        "train_action_weight_max": float(weights.max()),
    }


def main() -> None:
    args = parse_args()
    try:
        source_weights = parse_source_weights(args.source_weight)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no HU teacher samples")
    train_samples, holdout_samples = split_samples(
        samples,
        holdout_fraction=args.holdout,
        seed=args.seed,
    )
    features, targets = samples_to_matrix(train_samples)
    train_weights = build_action_source_weights(train_samples, source_weights)
    estimator = build_estimator(args)
    if train_weights is None:
        estimator.fit(features, targets)
    else:
        estimator.fit(features, targets, sample_weight=train_weights)
    model = HuSklearnActionValueModel(estimator=estimator)
    model.save(args.model_output)

    metrics = {
        "input": str(args.input),
        "model_output": str(args.model_output),
        "model_type": f"hu_{args.model_type}",
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
        "train_actions": int(features.shape[0]),
        "source_weights": source_weights,
        "source_counts": source_counts(samples),
        "train_source_counts": source_counts(train_samples),
        "holdout_source_counts": source_counts(holdout_samples),
        **weighted_action_summary(train_weights),
        "seed": args.seed,
        "train": evaluate_model(model, train_samples),
        "holdout": evaluate_model(model, holdout_samples),
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
