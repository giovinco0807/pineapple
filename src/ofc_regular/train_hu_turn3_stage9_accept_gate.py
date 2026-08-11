"""Train a runtime-compatible HU T3 Stage9 accept gate from joint-exact replay.

This consumes the same joint-exact MC replay JSONL files used for the Stage9d
tail-veto label cache, but trains a HuTurn3GateModel. Unlike the smoke
tail-veto model, this model returns accept probability and can be passed to the
runtime via --hu-turn3-gate-a with --hu-turn3-min-gate-probability-a.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.action_key import resolve_action_index
from ofc_regular.hu_turn3_gate_model import GATE_FEATURE_NAMES, HuTurn3GateModel, decision_gate_features
from ofc_regular.hu_infoset import actor_observation_from_record
from ofc_regular.hu_turn3_model import hu_policy_sample, load_hu_action_value_model
from ofc_regular.policy import policy_sample
from ofc_regular.state import Board
from ofc_regular.turn3_model import load_action_value_model


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _selection_index(
    record: dict[str, Any], actions: list[Any], name: str
) -> int:
    selection = record.get("selection", {})
    runtime = record.get("runtime_decision", {})
    payload_name = "stage3_action" if name == "baseline" else "stage7_action"
    payload = runtime.get(payload_name)
    return resolve_action_index(
        actions,
        key=selection.get(f"{name}_action_key"),
        payload=payload if isinstance(payload, dict) else None,
        legacy_index=selection.get(f"{name}_index"),
        expected_order_digest=selection.get("legal_action_order_digest"),
    ).index


def record_delta(record: dict[str, Any]) -> float:
    board = Board.from_rows(**record["board"])
    actions = generate_turn_actions(board, tuple(record["dealt"]))
    predictions = _joint_exact_predictions(record, actions)
    return float(
        predictions[_selection_index(record, actions, "hu")]
        - predictions[_selection_index(record, actions, "baseline")]
    )


def safe_predict(model: Any, sample: dict[str, Any], expected_len: int) -> np.ndarray:
    try:
        if hasattr(model, "predict_sample"):
            values = model.predict_sample(sample)
        elif hasattr(model, "predict_action_values"):
            values = model.predict_action_values(sample)
        elif hasattr(model, "predict_values"):
            values = model.predict_values(sample)
        elif hasattr(model, "predict"):
            values = model.predict(sample)
        else:
            values = np.zeros(expected_len, dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        arr = np.zeros(expected_len, dtype=np.float64)
    if arr.shape[0] != expected_len:
        return np.zeros(expected_len, dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _joint_exact_predictions(record: dict[str, Any], actions: list[Any]) -> np.ndarray:
    predictions = np.zeros(len(actions), dtype=np.float64)
    for action in record.get("actions", []):
        idx = resolve_action_index(
            actions,
            key=action.get("canonical_action_key"),
            payload=action,
            legacy_index=action.get("original_index"),
            expected_order_digest=record.get("selection", {}).get(
                "legal_action_order_digest"
            ),
        ).index
        predictions[idx] = float(action.get("joint_ev", action.get("score", 0.0)))
    return predictions


def record_to_training_row(
    record: dict[str, Any],
    *,
    self_model: Any | None,
    hu_model: Any | None = None,
) -> tuple[np.ndarray, int, float, float]:
    board = Board.from_rows(**record["board"])
    opponent_board = Board.from_rows(**record["opponent_board"])
    dealt = tuple(record["dealt"])
    actions = generate_turn_actions(board, dealt)
    observation = actor_observation_from_record(record)
    if (
        observation.hero_board != board
        or observation.opponent_public_board != opponent_board
        or observation.dealt_cards != dealt
        or observation.street != "T3"
    ):
        raise ValueError("Stage9 gate row disagrees with ActorObservation")
    baseline_index = _selection_index(record, actions, "baseline")
    hu_index = _selection_index(record, actions, "hu")

    sample = hu_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=record.get("seat", "first"),
        to_act_order=record.get("to_act_order", record.get("seat", "first")),
    )
    if hu_model is not None:
        hu_predictions = safe_predict(hu_model, sample, len(actions))
    else:
        hu_predictions = _joint_exact_predictions(record, actions)

    if self_model is not None:
        self_sample = policy_sample(board, dealt, actions)
        self_predictions = safe_predict(self_model, self_sample, len(actions))
    else:
        self_predictions = np.zeros(len(actions), dtype=np.float64)

    delta = record_delta(record)
    label = 1 if delta >= 0.25 else 0
    if delta <= -0.25:
        weight = 3.0
    elif delta >= 0.25:
        weight = 1.0
    else:
        weight = 0.35
    features = decision_gate_features(
        sample,
        chosen_index=hu_index,
        baseline_index=baseline_index,
        hu_predictions=hu_predictions,
        self_predictions=self_predictions,
    )
    return features, label, delta, weight


def build_arrays(
    inputs: list[Path],
    *,
    self_model: Any | None,
    hu_model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    deltas: list[float] = []
    weights: list[float] = []
    for path in inputs:
        for record in read_jsonl(path):
            try:
                feature, label, delta, weight = record_to_training_row(
                    record,
                    self_model=self_model,
                    hu_model=hu_model,
                )
            except Exception:
                continue
            features.append(feature)
            labels.append(label)
            deltas.append(delta)
            weights.append(weight)
    if not features:
        return (
            np.zeros((0, len(GATE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(deltas, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def split_indices(total: int, *, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    holdout_count = int(round(total * holdout))
    return indices[holdout_count:], indices[:holdout_count]


def build_estimator(args: argparse.Namespace) -> Any:
    if args.model_type == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=args.seed),
        )
    if args.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    raise ValueError(f"unsupported model_type: {args.model_type}")


def predict_accept(model: HuTurn3GateModel, features: np.ndarray) -> np.ndarray:
    estimator = model.estimator
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classes = list(getattr(estimator, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        return probabilities[:, positive_index].astype(np.float64)
    return np.asarray(estimator.predict(features), dtype=np.float64)


def threshold_metrics(probabilities: np.ndarray, deltas: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        mask = probabilities >= threshold
        selected = deltas[mask]
        selected_labels = labels[mask]
        rows.append(
            {
                "threshold": threshold,
                "kept_rows": int(mask.sum()),
                "kept_delta_sum": float(selected.sum()) if selected.size else 0.0,
                "kept_mean_delta": float(selected.mean()) if selected.size else 0.0,
                "kept_hard_negative_count": int(np.sum(selected < -0.25)) if selected.size else 0,
                "kept_positive_label_count": int(selected_labels.sum()) if selected_labels.size else 0,
                "vetoed_rows": int((~mask).sum()),
            }
        )
    return rows


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument(
        "--hu-turn3-model",
        type=Path,
        help=(
            "Optional Stage9 runtime HU model used to build gate prediction features. "
            "Labels still come from joint-exact MC replay deltas."
        ),
    )
    parser.add_argument("--turn3-model", type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--model-type", choices=("logistic", "random_forest"), default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()

    hu_model = load_hu_action_value_model(args.hu_turn3_model) if args.hu_turn3_model else None
    self_model = load_action_value_model(args.turn3_model) if args.turn3_model else None
    X, y, deltas, weights = build_arrays(args.input, self_model=self_model, hu_model=hu_model)
    if X.shape[0] < 2 or len(set(y.tolist())) < 2:
        raise SystemExit("need at least two rows and both accept/reject labels")
    train_idx, test_idx = split_indices(X.shape[0], holdout=args.holdout, seed=args.seed)
    estimator = build_estimator(args)
    if args.model_type == "logistic":
        estimator.fit(X[train_idx], y[train_idx], logisticregression__sample_weight=weights[train_idx])
    else:
        estimator.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
    model = HuTurn3GateModel(estimator=estimator, feature_names=GATE_FEATURE_NAMES)
    model.save(args.model_output)

    thresholds = parse_thresholds(args.thresholds)
    train_prob = predict_accept(model, X[train_idx])
    test_prob = predict_accept(model, X[test_idx])
    all_prob = predict_accept(model, X)
    metrics = {
        "inputs": [str(path) for path in args.input],
        "model_output": str(args.model_output),
        "hu_turn3_model": str(args.hu_turn3_model) if args.hu_turn3_model else None,
        "hu_prediction_source": "runtime_model" if args.hu_turn3_model else "joint_exact_replay",
        "turn3_model": str(args.turn3_model) if args.turn3_model else None,
        "rows": int(X.shape[0]),
        "positive_labels": int(y.sum()),
        "reject_labels": int(X.shape[0] - y.sum()),
        "train_rows": int(train_idx.size),
        "holdout_rows": int(test_idx.size),
        "feature_names": list(GATE_FEATURE_NAMES),
        "all": threshold_metrics(all_prob, deltas, y, thresholds),
        "train": threshold_metrics(train_prob, deltas[train_idx], y[train_idx], thresholds),
        "holdout": threshold_metrics(test_prob, deltas[test_idx], y[test_idx], thresholds),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
