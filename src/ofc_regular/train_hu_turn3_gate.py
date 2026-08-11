"""Train a gate model that decides whether to accept HU Turn3 overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .action_space import generate_turn_actions
from .action_key import resolve_action_index
from .hu_infoset import actor_observation_from_record
from .hu_turn3_gate_model import (
    GATE_FEATURE_NAMES,
    HuTurn3GateModel,
    decision_gate_features,
)
from .hu_turn3_model import hu_policy_sample
from .state import Board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--holdout-input",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional external trace JSONL used only for holdout metrics. "
            "When present, all --input rows are used for training and --holdout is ignored."
        ),
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--model-type", choices=("logistic", "random_forest"), default="logistic")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument("--positive-delta", type=float, default=1e-9)
    parser.add_argument("--weight-by-abs-delta", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated gate probability thresholds to sweep.",
    )
    parser.add_argument(
        "--paired-seeds",
        type=int,
        default=1000,
        help="Paired seeds represented by the trace, used for approximate EV/hand.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def read_trace_rows(paths: Sequence[Path], *, min_abs_delta: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("override", False):
                    continue
                if abs(float(row.get("counterfactual_delta_vs_baseline", 0.0))) < min_abs_delta:
                    continue
                rows.append(row)
    return rows


def rows_to_arrays(
    rows: Sequence[dict[str, Any]],
    *,
    positive_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []
    deltas = []
    sample_weights = []
    for row in rows:
        features, _label, delta, weight = trace_row_to_training_row(row)
        feature_rows.append(features)
        labels.append(1 if delta > positive_delta else 0)
        deltas.append(delta)
        sample_weights.append(max(weight, 1.0))
    if not feature_rows:
        return (
            np.zeros((0, len(GATE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.vstack(feature_rows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(deltas, dtype=np.float64),
        np.asarray(sample_weights, dtype=np.float64),
    )


def trace_row_to_training_row(row: dict[str, Any]) -> tuple[np.ndarray, int, float, float]:
    board = Board.from_rows(**row["board"])
    opponent_board = Board.from_rows(**row["opponent_board"])
    dealt = tuple(row["dealt"])
    actions = generate_turn_actions(board, dealt)
    observation = actor_observation_from_record(row)
    if (
        observation.hero_board != board
        or observation.opponent_public_board != opponent_board
        or observation.dealt_cards != dealt
        or observation.street != "T3"
    ):
        raise ValueError("T3 gate row disagrees with ActorObservation")
    chosen_index = resolve_action_index(
        actions, payload=row["chosen_action"]
    ).index
    baseline_index = resolve_action_index(
        actions, payload=row["baseline_action"]
    ).index
    sample = hu_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=row.get("seat", "first"),
        to_act_order=row.get("to_act_order", row.get("seat", "first")),
    )
    hu_predictions = np.zeros(len(actions), dtype=np.float64)
    self_predictions = np.zeros(len(actions), dtype=np.float64)
    hu_predictions[chosen_index] = float(row.get("predicted_hu_score", row.get("predicted_margin", 0.0)))
    hu_predictions[baseline_index] = float(row.get("predicted_baseline_score", 0.0))
    self_predictions[chosen_index] = float(row.get("self_model_hu_score", 0.0))
    self_predictions[baseline_index] = float(row.get("self_model_baseline_score", 0.0))
    features = decision_gate_features(
        sample,
        chosen_index=chosen_index,
        baseline_index=baseline_index,
        hu_predictions=hu_predictions,
        self_predictions=self_predictions,
    )
    delta = float(row.get("counterfactual_delta_vs_baseline", 0.0))
    label = 1 if delta > 0.0 else 0
    return features, label, delta, abs(delta)


def action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def generated_action_key(action: Any) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.placements)
    discards = tuple(str(card) for card in action.discards)
    return placements, discards


def split_indices(total: int, *, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= holdout < 1.0:
        raise ValueError("holdout must be in [0, 1)")
    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    holdout_count = int(round(total * holdout))
    return indices[holdout_count:], indices[:holdout_count]


def build_estimator(args: argparse.Namespace):
    if args.model_type == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=args.seed,
            ),
        )
    if args.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    raise ValueError(f"unsupported model type: {args.model_type}")


def predict_probabilities(model: HuTurn3GateModel, features: np.ndarray) -> np.ndarray:
    estimator = model.estimator
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classes = list(getattr(estimator, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        return probabilities[:, positive_index].astype(np.float64)
    return np.asarray(estimator.predict(features), dtype=np.float64)


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    deltas: np.ndarray,
    *,
    thresholds: Sequence[float],
    paired_seeds: int,
) -> dict[str, Any]:
    predictions = probabilities >= 0.5
    accuracy = float(np.mean(predictions == labels)) if labels.size else 0.0
    selected = probabilities >= 0.5
    return {
        "rows": int(labels.size),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(labels.size - labels.sum()),
        "accuracy_at_0_5": accuracy,
        "selected_delta_sum_at_0_5": float(deltas[selected].sum(dtype=np.float64)),
        "thresholds": threshold_sweep(
            probabilities,
            deltas,
            thresholds=thresholds,
            paired_seeds=paired_seeds,
        ),
    }


def threshold_sweep(
    probabilities: np.ndarray,
    deltas: np.ndarray,
    *,
    thresholds: Sequence[float],
    paired_seeds: int,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for threshold in thresholds:
        mask = probabilities >= threshold
        selected = deltas[mask]
        wins = int(np.sum(selected > 1e-9))
        losses = int(np.sum(selected < -1e-9))
        ties = int(selected.size - wins - losses)
        delta_sum = float(selected.sum(dtype=np.float64))
        results.append(
            {
                "gate_probability_threshold": float(threshold),
                "kept_overrides": float(selected.size),
                "delta_sum": delta_sum,
                "delta_avg": float(delta_sum / selected.size) if selected.size else 0.0,
                "approx_ev_per_hand": float(delta_sum / 2.0 / max(paired_seeds, 1)),
                "wins": float(wins),
                "losses": float(losses),
                "ties": float(ties),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    rows = read_trace_rows(args.input, min_abs_delta=args.min_abs_delta)
    if len(rows) < 2:
        raise SystemExit("need at least two override rows")

    features, label_array, delta_array, weight_array = rows_to_arrays(
        rows,
        positive_delta=args.positive_delta,
    )
    if len(set(label_array.tolist())) < 2:
        raise SystemExit("gate training data needs both winning and losing overrides")

    external_holdout_rows: list[dict[str, Any]] = []
    if args.holdout_input:
        external_holdout_rows = read_trace_rows(
            args.holdout_input,
            min_abs_delta=args.min_abs_delta,
        )
        train_idx = np.arange(len(rows))
        holdout_idx = np.zeros(0, dtype=np.int64)
        split_mode = "external_input"
        holdout_features, holdout_labels, holdout_deltas, _holdout_weights = rows_to_arrays(
            external_holdout_rows,
            positive_delta=args.positive_delta,
        )
    else:
        train_idx, holdout_idx = split_indices(len(rows), holdout=args.holdout, seed=args.seed)
        split_mode = "random_row"
        holdout_features = features[holdout_idx]
        holdout_labels = label_array[holdout_idx]
        holdout_deltas = delta_array[holdout_idx]

    estimator = build_estimator(args)
    fit_kwargs = {}
    if args.weight_by_abs_delta:
        if args.model_type == "logistic":
            fit_kwargs["logisticregression__sample_weight"] = weight_array[train_idx]
        else:
            fit_kwargs["sample_weight"] = weight_array[train_idx]
    estimator.fit(features[train_idx], label_array[train_idx], **fit_kwargs)
    model = HuTurn3GateModel(estimator=estimator, feature_names=GATE_FEATURE_NAMES)
    model.save(args.model_output)

    train_probabilities = predict_probabilities(model, features[train_idx])
    holdout_probabilities = (
        predict_probabilities(model, holdout_features)
        if holdout_features.shape[0]
        else np.zeros(0)
    )
    if args.holdout_input and holdout_features.shape[0]:
        all_features = np.vstack([features, holdout_features])
        all_labels = np.concatenate([label_array, holdout_labels])
        all_deltas = np.concatenate([delta_array, holdout_deltas])
    else:
        all_features = features
        all_labels = label_array
        all_deltas = delta_array
    all_probabilities = predict_probabilities(model, all_features)
    metrics = {
        "inputs": [str(path) for path in args.input],
        "holdout_inputs": [str(path) for path in args.holdout_input],
        "model_output": str(args.model_output),
        "model_type": f"hu_turn3_gate_{args.model_type}",
        "feature_names": list(GATE_FEATURE_NAMES),
        "split_mode": split_mode,
        "total_rows": int(all_labels.size),
        "input_rows": len(rows),
        "external_holdout_rows": len(external_holdout_rows),
        "train_rows": int(train_idx.size),
        "holdout_rows": int(holdout_features.shape[0]),
        "seed": args.seed,
        "holdout": args.holdout,
        "min_abs_delta": args.min_abs_delta,
        "positive_delta": args.positive_delta,
        "weight_by_abs_delta": args.weight_by_abs_delta,
        "paired_seeds": args.paired_seeds,
        "all": classification_metrics(
            all_probabilities,
            all_labels,
            all_deltas,
            thresholds=thresholds,
            paired_seeds=args.paired_seeds,
        ),
        "train": classification_metrics(
            train_probabilities,
            label_array[train_idx],
            delta_array[train_idx],
            thresholds=thresholds,
            paired_seeds=args.paired_seeds,
        ),
        "holdout": classification_metrics(
            holdout_probabilities,
            holdout_labels,
            holdout_deltas,
            thresholds=thresholds,
            paired_seeds=args.paired_seeds,
        ),
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
