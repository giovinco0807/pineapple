"""Train a HU T0/T1 TopK candidate-generator model.

This is for validation-only T1 search experiments.  It trains a classifier that
scores legal actions by whether the stronger teacher considers them near-best,
then saves it through the existing HU action-value model interface so the T1
TopK confirm runtime can use it as a candidate generator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .hu_turn3_model import (
    DecisionFunctionAsScoreEstimator,
    HuSklearnActionValueModel,
    PredictProbaAsScoreEstimator,
    load_hu_action_value_model,
    read_teacher_samples,
    sample_to_matrix,
    split_samples,
)
from .hu_turn1_training_augmentation import augment_samples_by_suit
from .train_hu_turn3 import parse_source_weights, sample_source, source_counts


def infer_artifact_stage(samples: Sequence[dict[str, Any]]) -> str:
    """Keep reusable trainer artifacts labeled with the street they represent."""
    phases = {str(sample.get("phase", "")) for sample in samples}
    schemas = {str(sample.get("schema", "")) for sample in samples}
    if any(phase.startswith("hu_turn0") for phase in phases) or any(
        schema.startswith("hu_turn0") for schema in schemas
    ):
        return "hu_turn0"
    return "hu_turn1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--validation-input",
        type=Path,
        help=(
            "Optional explicit state-level validation JSONL. When supplied, all --input "
            "samples are used for training and --holdout is not used to split them."
        ),
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument(
        "--holdout-output",
        type=Path,
        help="Optional state-level holdout JSONL used for leak-free coverage analysis.",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        help="Optional existing HU action-value model evaluated on the same train/holdout split.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026062401)
    parser.add_argument(
        "--suit-augmentations",
        type=int,
        default=0,
        help="Non-identity global suit permutations appended to the training split only (0-23).",
    )
    parser.add_argument(
        "--model-type",
        choices=(
            "hgb",
            "extra_trees",
            "random_forest",
            "logistic",
            "hgb_regressor",
            "extra_trees_regressor",
            "random_forest_regressor",
            "pairwise_logistic",
        ),
        default="hgb",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--accept-regret", type=float, default=0.25)
    parser.add_argument("--gray-regret", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--gray-weight", type=float, default=0.75)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-regret", type=float, default=10.0)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument(
        "--classification-target",
        choices=("nearbest", "safe_lcb196"),
        default="nearbest",
        help="Binary action target used by classifier model types.",
    )
    parser.add_argument("--pairwise-min-gap", type=float, default=0.25)
    parser.add_argument("--pairwise-max-pairs-per-sample", type=int, default=256)
    parser.add_argument(
        "--regression-target",
        choices=("absolute_score", "negative_regret", "baseline_delta"),
        default="absolute_score",
        help="Regression target; negative_regret removes the state-level EV offset.",
    )
    parser.add_argument(
        "--delta-se-weight-floor",
        type=float,
        default=0.0,
        help=(
            "When positive, multiply action-row weights by floor^2/(floor^2+delta_se^2). "
            "Intended for noisy baseline-delta Monte Carlo labels."
        ),
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help="Per-sample source weight. May be repeated or comma-separated.",
    )
    return parser.parse_args()


def build_estimator(args: argparse.Namespace) -> Any:
    if args.model_type == "hgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
            early_stopping=True,
        )
    if args.model_type == "hgb_regressor":
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
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
            class_weight=None,
        )
    if args.model_type == "extra_trees_regressor":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
            class_weight=None,
        )
    if args.model_type == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.model_type == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=args.seed,
                class_weight=None,
            ),
        )
    if args.model_type == "pairwise_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=args.seed,
                class_weight=None,
            ),
        )
    raise ValueError(f"unsupported model type: {args.model_type}")


def is_regressor_model_type(model_type: str) -> bool:
    return model_type.endswith("_regressor")


def is_pairwise_model_type(model_type: str) -> bool:
    return model_type == "pairwise_logistic"


def action_regrets(sample: dict[str, Any], targets: np.ndarray) -> np.ndarray:
    if targets.size == 0:
        return targets.astype(np.float64)
    best_score = float(np.max(targets))
    return best_score - targets.astype(np.float64, copy=False)


def labels_and_weights_for_sample(
    sample: dict[str, Any],
    targets: np.ndarray,
    *,
    source_weight: float,
    accept_regret: float,
    gray_regret: float,
    positive_weight: float,
    gray_weight: float,
    negative_weight: float,
    hard_negative_regret: float,
    hard_negative_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regrets = action_regrets(sample, targets)
    labels = (regrets <= accept_regret).astype(np.int64)
    weights = np.full(targets.shape[0], negative_weight * source_weight, dtype=np.float64)
    weights[regrets <= gray_regret] = gray_weight * source_weight
    weights[labels == 1] = positive_weight * source_weight
    weights[regrets >= hard_negative_regret] = hard_negative_weight * source_weight
    return labels, weights, regrets


def safe_lcb_labels_and_weights_for_sample(
    sample: dict[str, Any],
    targets: np.ndarray,
    *,
    source_weight: float,
    positive_weight: float,
    gray_weight: float,
    negative_weight: float,
    hard_negative_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = list(sample.get("actions", ()))
    if len(actions) != targets.shape[0]:
        raise ValueError("safe-LCB classification requires one action payload per feature row")
    deltas = np.asarray(
        [float(action.get("delta_vs_baseline", float("nan"))) for action in actions],
        dtype=np.float64,
    )
    delta_ses = np.asarray(
        [float(action.get("delta_se_vs_baseline", float("nan"))) for action in actions],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(deltas))
        or not np.all(np.isfinite(delta_ses))
        or np.any(delta_ses < 0.0)
    ):
        raise ValueError("safe-LCB classification requires finite delta and non-negative SE")
    labels = (deltas - 1.96 * delta_ses > 0.0).astype(np.int64)
    weights = np.full(targets.shape[0], gray_weight * source_weight, dtype=np.float64)
    weights[deltas <= 0.0] = negative_weight * source_weight
    weights[deltas + 1.96 * delta_ses < 0.0] = hard_negative_weight * source_weight
    weights[labels == 1] = positive_weight * source_weight
    return labels, weights, action_regrets(sample, targets)


def build_training_matrix(
    samples: Sequence[dict[str, Any]],
    *,
    source_weights: dict[str, float],
    args: argparse.Namespace,
    include_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
]:
    features_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    regret_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    row_start = 0
    for sample in samples:
        features, targets = sample_to_matrix(sample)
        source_weight = float(source_weights.get(sample_source(sample), 1.0))
        if str(getattr(args, "classification_target", "nearbest")) == "safe_lcb196":
            labels, weights, regrets = safe_lcb_labels_and_weights_for_sample(
                sample,
                targets,
                source_weight=source_weight,
                positive_weight=float(args.positive_weight),
                gray_weight=float(args.gray_weight),
                negative_weight=float(args.negative_weight),
                hard_negative_weight=float(args.hard_negative_weight),
            )
        else:
            labels, weights, regrets = labels_and_weights_for_sample(
                sample,
                targets,
                source_weight=source_weight,
                accept_regret=float(args.accept_regret),
                gray_regret=float(args.gray_regret),
                positive_weight=float(args.positive_weight),
                gray_weight=float(args.gray_weight),
                negative_weight=float(args.negative_weight),
                hard_negative_regret=float(args.hard_negative_regret),
                hard_negative_weight=float(args.hard_negative_weight),
            )
        delta_se_weight_floor = float(getattr(args, "delta_se_weight_floor", 0.0))
        if delta_se_weight_floor > 0.0:
            actions = list(sample.get("actions", ()))
            if len(actions) != targets.shape[0]:
                raise ValueError("delta-SE weighting requires one action payload per feature row")
            delta_ses = np.asarray(
                [float(action.get("delta_se_vs_baseline", float("nan"))) for action in actions],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(delta_ses)) or np.any(delta_ses < 0.0):
                raise ValueError("delta-SE weighting requires finite non-negative action SEs")
            floor_squared = delta_se_weight_floor**2
            weights *= floor_squared / (floor_squared + np.square(delta_ses))
        features_parts.append(features.astype(np.float32, copy=False))
        label_parts.append(labels)
        weight_parts.append(weights)
        regret_parts.append(regrets)
        score_parts.append(targets.astype(np.float64, copy=False))
        row_end = row_start + features.shape[0]
        offsets.append((row_start, row_end))
        row_start = row_end
    if not features_parts:
        raise ValueError("no training rows")
    result = (
        np.vstack(features_parts).astype(np.float32, copy=False),
        np.concatenate(label_parts).astype(np.int64, copy=False),
        np.concatenate(weight_parts).astype(np.float64, copy=False),
        offsets,
        np.concatenate(regret_parts).astype(np.float64, copy=False),
    )
    if include_scores:
        return (*result, np.concatenate(score_parts).astype(np.float64, copy=False))
    return result


def build_pairwise_training_matrix(
    samples: Sequence[dict[str, Any]],
    *,
    source_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    features_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    total_forward_pairs = 0
    min_gap = float(args.pairwise_min_gap)
    max_pairs = max(1, int(args.pairwise_max_pairs_per_sample))
    for sample in samples:
        features, targets = sample_to_matrix(sample)
        if features.shape[0] < 2:
            continue
        source_weight = float(source_weights.get(sample_source(sample), 1.0))
        pairs: list[tuple[float, int, int]] = []
        for better in range(targets.shape[0]):
            for worse in range(targets.shape[0]):
                gap = float(targets[better] - targets[worse])
                if gap >= min_gap:
                    pairs.append((gap, better, worse))
        if not pairs:
            continue
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = pairs[:max_pairs]
        total_forward_pairs += len(selected)
        diffs: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for gap, better, worse in selected:
            diff = features[better] - features[worse]
            weight = source_weight * min(8.0, max(1.0, gap / 4.0))
            diffs.append(diff)
            labels.append(1)
            weights.append(weight)
            diffs.append(-diff)
            labels.append(0)
            weights.append(weight)
        features_parts.append(np.vstack(diffs).astype(np.float32, copy=False))
        label_parts.append(np.asarray(labels, dtype=np.int64))
        weight_parts.append(np.asarray(weights, dtype=np.float64))
    if not features_parts:
        raise ValueError("no pairwise training rows")
    return (
        np.vstack(features_parts).astype(np.float32, copy=False),
        np.concatenate(label_parts).astype(np.int64, copy=False),
        np.concatenate(weight_parts).astype(np.float64, copy=False),
        total_forward_pairs,
    )


def predict_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_matrix"):
        return np.asarray(estimator.predict_matrix(features.astype(np.float32, copy=False)), dtype=np.float64)
    return PredictProbaAsScoreEstimator(estimator).predict(features.astype(np.float32, copy=False))


def baseline_delta_targets(
    samples: Sequence[dict[str, Any]],
    scores: np.ndarray,
    offsets: Sequence[tuple[int, int]],
) -> np.ndarray:
    targets = scores.astype(np.float64, copy=True)
    for sample, (start, end) in zip(samples, offsets, strict=True):
        baseline_row_index = sample.get("baseline_action_row_index")
        if baseline_row_index is None:
            raise ValueError("baseline_delta regression requires baseline_action_row_index")
        baseline_row_index = int(baseline_row_index)
        if baseline_row_index < 0 or start + baseline_row_index >= end:
            raise ValueError("baseline_action_row_index is out of range")
        targets[start:end] -= scores[start + baseline_row_index]
    return targets


def fit_estimator(estimator: Any, features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> None:
    if hasattr(estimator, "steps"):
        final_step_name = estimator.steps[-1][0]
        estimator.fit(features, labels, **{f"{final_step_name}__sample_weight": weights})
    else:
        estimator.fit(features, labels, sample_weight=weights)


def evaluate_candidate_generator(
    estimator: Any,
    samples: Sequence[dict[str, Any]],
    *,
    topk_values: Sequence[int] = (1, 3, 5, 10, 15, 20, 25),
    accept_regret: float,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    source_rows: dict[str, list[float]] = {}
    totals = {
        "samples": len(samples),
        "actions": 0,
        "positive_actions": 0,
        "top1_regret_sum": 0.0,
    }
    topk_stats = {
        int(k): {
            "teacher_best_in_topk": 0,
            "accepted_action_in_topk": 0,
            "best_regret_sum": 0.0,
        }
        for k in topk_values
    }
    for sample in samples:
        features, targets = sample_to_matrix(sample)
        scores = predict_scores(estimator, features)
        regrets = action_regrets(sample, targets)
        order = np.argsort(-scores, kind="mergesort")
        best_action = int(sample.get("best_action", int(np.argmin(regrets))) or 0)
        accepted = set(int(index) for index in np.flatnonzero(regrets <= accept_regret))
        totals["actions"] += int(features.shape[0])
        totals["positive_actions"] += len(accepted)
        top1_regret = float(regrets[int(order[0])])
        totals["top1_regret_sum"] += top1_regret
        source_rows.setdefault(sample_source(sample), []).append(top1_regret)
        for k in topk_values:
            safe_k = min(int(k), len(order))
            topk = [int(index) for index in order[:safe_k]]
            stats = topk_stats[int(k)]
            if best_action in topk:
                stats["teacher_best_in_topk"] += 1
            if accepted and any(index in accepted for index in topk):
                stats["accepted_action_in_topk"] += 1
            stats["best_regret_sum"] += float(min(regrets[index] for index in topk))
        rows.append({"top1_regret": top1_regret})

    sample_count = max(1, len(samples))
    result: dict[str, Any] = {
        "samples": len(samples),
        "actions": totals["actions"],
        "positive_actions": totals["positive_actions"],
        "positive_action_rate": totals["positive_actions"] / max(1, totals["actions"]),
        "top1_avg_regret": totals["top1_regret_sum"] / sample_count,
    }
    for k, stats in topk_stats.items():
        result[f"top{k}_teacher_best_recall"] = stats["teacher_best_in_topk"] / sample_count
        result[f"top{k}_accepted_action_recall"] = stats["accepted_action_in_topk"] / sample_count
        result[f"top{k}_best_avg_regret"] = stats["best_regret_sum"] / sample_count
    result["source_top1_avg_regret"] = {
        source: float(np.mean(values)) if values else 0.0
        for source, values in sorted(source_rows.items())
    }
    return result


def main() -> None:
    args = parse_args()
    if args.accept_regret < 0.0:
        raise SystemExit("--accept-regret must be non-negative")
    if args.gray_regret < args.accept_regret:
        raise SystemExit("--gray-regret must be >= --accept-regret")
    source_weights = parse_source_weights(args.source_weight)
    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no HU T0/T1 teacher samples")
    artifact_stage = infer_artifact_stage(samples)
    if args.validation_input is not None:
        train_samples_raw = samples
        holdout_samples = read_teacher_samples(args.validation_input)
        if not holdout_samples:
            raise SystemExit("explicit validation input has no HU T0/T1 teacher samples")
        validation_stage = infer_artifact_stage(holdout_samples)
        if validation_stage != artifact_stage:
            raise SystemExit(
                f"training/validation artifact stage mismatch: {artifact_stage} != {validation_stage}"
            )
        split_strategy = "explicit_validation_input"
    else:
        train_samples_raw, holdout_samples = split_samples(
            samples,
            holdout_fraction=args.holdout,
            seed=args.seed,
        )
        split_strategy = "random_state_holdout"
    try:
        train_samples = augment_samples_by_suit(
            train_samples_raw,
            count=int(args.suit_augmentations),
            seed=int(args.seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    train_x, train_y, train_weights, offsets, train_regrets, train_scores = build_training_matrix(
        train_samples,
        source_weights=source_weights,
        args=args,
        include_scores=True,
    )
    if (
        not is_regressor_model_type(args.model_type)
        and not is_pairwise_model_type(args.model_type)
        and len(set(train_y.tolist())) < 2
    ):
        raise SystemExit("training labels must contain both positive and negative action rows")
    estimator = build_estimator(args)
    pairwise_train_pairs = 0
    if is_pairwise_model_type(args.model_type):
        pair_x, pair_y, pair_weights, pairwise_train_pairs = build_pairwise_training_matrix(
            train_samples,
            source_weights=source_weights,
            args=args,
        )
        if len(set(pair_y.tolist())) < 2:
            raise SystemExit("pairwise training labels must contain both classes")
        fit_estimator(estimator, pair_x, pair_y, pair_weights)
        score_estimator = DecisionFunctionAsScoreEstimator(estimator)
        model = HuSklearnActionValueModel(estimator=score_estimator)
    else:
        if is_regressor_model_type(args.model_type):
            if args.regression_target == "negative_regret":
                train_target = -train_regrets
            elif args.regression_target == "baseline_delta":
                train_target = baseline_delta_targets(train_samples, train_scores, offsets)
            else:
                train_target = train_scores
        else:
            train_target = train_y
        fit_estimator(estimator, train_x, train_target, train_weights)
        if is_regressor_model_type(args.model_type):
            score_estimator = estimator
        else:
            score_estimator = PredictProbaAsScoreEstimator(estimator)
        model = HuSklearnActionValueModel(estimator=score_estimator)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)
    if args.holdout_output is not None:
        args.holdout_output.parent.mkdir(parents=True, exist_ok=True)
        with args.holdout_output.open("w", encoding="utf-8") as handle:
            for sample in holdout_samples:
                handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")

    train_eval = evaluate_candidate_generator(score_estimator, train_samples, accept_regret=args.accept_regret)
    holdout_eval = evaluate_candidate_generator(score_estimator, holdout_samples, accept_regret=args.accept_regret)
    baseline_eval: dict[str, Any] | None = None
    if args.baseline_model is not None:
        baseline_model = load_hu_action_value_model(args.baseline_model)
        baseline_eval = {
            "path": str(args.baseline_model),
            "train": evaluate_candidate_generator(baseline_model, train_samples, accept_regret=args.accept_regret),
            "holdout": evaluate_candidate_generator(baseline_model, holdout_samples, accept_regret=args.accept_regret),
        }
    metrics = {
        "schema": f"{artifact_stage}_candidate_generator_training_metrics_v1",
        "artifact_stage": artifact_stage,
        "input": str(args.input),
        "validation_input": str(args.validation_input) if args.validation_input else None,
        "split_strategy": split_strategy,
        "model_output": str(args.model_output),
        "model_type": f"{artifact_stage}_candidate_{args.model_type}",
        "total_samples": len(samples),
        "raw_train_samples": len(train_samples_raw),
        "train_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
        "holdout_output": str(args.holdout_output) if args.holdout_output else None,
        "suit_augmentations": int(args.suit_augmentations),
        "train_actions": int(train_x.shape[0]),
        "train_positive_actions": int(train_y.sum()),
        "train_positive_action_rate": float(train_y.mean()),
        "training_objective": (
            "pairwise_ranking"
            if is_pairwise_model_type(args.model_type)
            else (
                "state_centered_regret_regression"
                if args.regression_target == "negative_regret"
                else "baseline_delta_regression"
                if args.regression_target == "baseline_delta"
                else "score_regression"
            )
            if is_regressor_model_type(args.model_type)
            else (
                "safe_lcb196_classification"
                if args.classification_target == "safe_lcb196"
                else "near_best_classification"
            )
        ),
        "classification_target": args.classification_target,
        "pairwise_min_gap": float(args.pairwise_min_gap),
        "pairwise_max_pairs_per_sample": int(args.pairwise_max_pairs_per_sample),
        "pairwise_train_pairs": int(pairwise_train_pairs),
        "regression_target": args.regression_target,
        "delta_se_weight_floor": float(args.delta_se_weight_floor),
        "train_weight_sum": float(train_weights.sum(dtype=np.float64)),
        "train_regret_mean": float(np.mean(train_regrets)),
        "accept_regret": float(args.accept_regret),
        "gray_regret": float(args.gray_regret),
        "source_weights": source_weights,
        "source_counts": source_counts(samples),
        "train_source_counts": source_counts(train_samples),
        "holdout_source_counts": source_counts(holdout_samples),
        "seed": int(args.seed),
        "train": train_eval,
        "holdout": holdout_eval,
        "baseline_model": baseline_eval,
        "decision": "candidate_generator_only_not_production_runtime",
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
