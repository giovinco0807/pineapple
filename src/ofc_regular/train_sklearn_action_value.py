"""Train a non-linear regular OFC action-value model with scikit-learn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .turn3_model import (
    SklearnActionValueModel,
    evaluate_model,
    read_teacher_samples,
    samples_to_matrix,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="teacher JSONL")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument(
        "--model-type",
        choices=("hgb", "extra_trees", "random_forest", "mlp"),
        default="hgb",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--hidden-layer-sizes", default="256,128")
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-fit-metrics",
        action="store_true",
        help="Do not evaluate train/holdout after fitting; useful for quick model selection.",
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
            max_bins=args.max_bins,
            early_stopping=args.early_stopping,
            random_state=args.seed,
            verbose=0,
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
    if args.model_type == "mlp":
        hidden_layer_sizes = tuple(
            int(part.strip()) for part in args.hidden_layer_sizes.split(",") if part.strip()
        )
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=args.alpha,
                learning_rate_init=args.learning_rate,
                max_iter=args.max_iter,
                early_stopping=args.early_stopping,
                random_state=args.seed,
                verbose=False,
            ),
        )
    raise ValueError(f"unsupported model type: {args.model_type}")


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

    train_features, train_targets = samples_to_matrix(train_samples)
    estimator = build_estimator(args)
    estimator.fit(train_features, train_targets)
    model = SklearnActionValueModel(estimator=estimator)
    model.save(args.model_output)

    train_metrics = None
    holdout_metrics = None
    if not args.skip_fit_metrics:
        train_metrics = evaluate_model(model, train_samples)
        holdout_metrics = evaluate_model(model, holdout_samples)

    metrics = {
        "input": str(args.input),
        "model_output": str(args.model_output),
        "model_type": args.model_type,
        "total_samples": len(samples),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2": args.l2,
        "max_bins": args.max_bins,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "hidden_layer_sizes": args.hidden_layer_sizes,
        "alpha": args.alpha,
        "early_stopping": args.early_stopping,
    }
    print(json.dumps(metrics, indent=2))

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
