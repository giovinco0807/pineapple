"""Train a deployable HU T0 safe-override selector from paired MC replay labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cards import card_rank, card_suit
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix


META_FEATURE_NAMES = (
    "meta_predicted_margin",
    "meta_action_count_log1p",
    "meta_candidate_pool_count_log1p",
    "meta_seat_is_second",
)
FEATURE_MODES = (
    "meta_only",
    "compact_delta_plus_meta",
    "compact_candidate_delta_plus_meta",
    "delta_plus_meta",
    "candidate_delta_plus_meta",
)
ROW_COMPACT_DIM = 12
GLOBAL_COMPACT_DIM = 8
ACTION_COMPACT_DIM = 3 * ROW_COMPACT_DIM + GLOBAL_COMPACT_DIM


@dataclass(frozen=True)
class SelectorData:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    deltas: np.ndarray
    label_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    feature_mode: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _action(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError(f"row is missing action JSON: {keys}")


def row_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    candidate = _action(row, "candidate_action", "hu_turn0_action", "runtime_candidate_action")
    baseline = _action(row, "baseline_action", "fallback_action", "runtime_baseline_action")
    hero_board = row.get("hero_board") or row.get("board")
    opponent_board = row.get("opponent_board")
    if not isinstance(hero_board, dict) or not isinstance(opponent_board, dict):
        raise ValueError("row is missing hero/opponent board JSON")
    seat = str(row.get("seat") or "first")
    return {
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn0_0card",
        "seat": seat,
        "to_act_order": seat,
        "board": hero_board,
        "opponent_board": opponent_board,
        "dead_cards": list(row.get("dead_cards") or row.get("visible_dead_cards") or ()),
        "dealt": list(row.get("cards_to_place") or row.get("dealt") or ()),
        "best_action": 0,
        "score_gap": 0.0,
        "actions": [candidate, baseline],
    }


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if row.get(key) is not None:
            return safe_float(row.get(key), default)
    return default


def _action_compact_features(action: dict[str, Any]) -> np.ndarray:
    next_board = action.get("next_board")
    if not isinstance(next_board, dict):
        next_board = {"top": [], "middle": [], "bottom": []}
        for card, row_name in action.get("placements", ()) or ():
            next_board.setdefault(str(row_name), []).append(str(card))
    values: list[float] = []
    rank_pair_cohesion = 0.0
    suit_pair_cohesion = 0.0
    occupied_rows = 0
    paired_rows = 0
    three_suited_rows = 0
    row_counts: list[int] = []
    for row_name, capacity in (("top", 3), ("middle", 5), ("bottom", 5)):
        cards = [str(card) for card in next_board.get(row_name, ()) or ()]
        row_counts.append(len(cards))
        ranks = [card_rank(card) for card in cards]
        suits = [card_suit(card) for card in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        distinct_ranks = len(rank_counts)
        max_rank_count = max(rank_counts.values(), default=0)
        pair_or_better = sum(count >= 2 for count in rank_counts.values())
        trips = sum(count >= 3 for count in rank_counts.values())
        max_suit_count = max(suit_counts.values(), default=0)
        sorted_unique = sorted(rank_counts)
        connected_pairs = sum(
            right - left == 1 for left, right in zip(sorted_unique, sorted_unique[1:])
        )
        rank_pair_cohesion += sum(count * (count - 1) / 2 for count in rank_counts.values())
        suit_pair_cohesion += sum(count * (count - 1) / 2 for count in suit_counts.values())
        occupied_rows += int(bool(cards))
        paired_rows += int(pair_or_better > 0)
        three_suited_rows += int(max_suit_count >= 3)
        values.extend(
            (
                len(cards) / capacity,
                distinct_ranks / capacity,
                max_rank_count / capacity,
                pair_or_better / capacity,
                trips / capacity,
                max_suit_count / capacity,
                len(suit_counts) / 4.0,
                max(ranks, default=0) / 14.0,
                min(ranks, default=0) / 14.0,
                (sum(ranks) / len(ranks) / 14.0) if ranks else 0.0,
                ((max(ranks) - min(ranks)) / 12.0) if ranks else 0.0,
                connected_pairs / max(1, len(sorted_unique) - 1),
            )
        )
    values.extend(
        (
            rank_pair_cohesion / 10.0,
            suit_pair_cohesion / 10.0,
            row_counts[0] / 5.0,
            row_counts[1] / 5.0,
            row_counts[2] / 5.0,
            occupied_rows / 3.0,
            paired_rows / 3.0,
            three_suited_rows / 3.0,
        )
    )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (ACTION_COMPACT_DIM,):
        raise ValueError(f"unexpected T0 compact feature shape: {output.shape}")
    return output


def row_to_feature_vector(row: dict[str, Any], *, feature_mode: str) -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature mode: {feature_mode}")
    predicted_margin = _first_present(
        row,
        ("predicted_margin", "hu_turn0_predicted_margin", "runtime_predicted_margin"),
    )
    action_count = max(0.0, _first_present(row, ("action_count",), 232.0))
    candidate_pool_count = max(
        0.0,
        _first_present(row, ("candidate_pool_count", "candidate_topk"), 60.0),
    )
    meta = np.asarray(
        [
            predicted_margin,
            math.log1p(action_count),
            math.log1p(candidate_pool_count),
            1.0 if str(row.get("seat")) == "second" else 0.0,
        ],
        dtype=np.float32,
    )
    if feature_mode == "meta_only":
        return meta
    sample = row_to_sample(row)
    if feature_mode in {
        "compact_delta_plus_meta",
        "compact_candidate_delta_plus_meta",
    }:
        candidate_compact = _action_compact_features(sample["actions"][0])
        baseline_compact = _action_compact_features(sample["actions"][1])
        compact_delta = candidate_compact - baseline_compact
        if feature_mode == "compact_delta_plus_meta":
            return np.concatenate((compact_delta, meta)).astype(np.float32, copy=False)
        return np.concatenate((candidate_compact, compact_delta, meta)).astype(
            np.float32,
            copy=False,
        )
    features, _targets = sample_to_matrix(sample)
    if features.shape != (2, HU_FEATURE_DIM):
        raise ValueError(f"expected two HU feature rows, got {features.shape}")
    candidate = features[0].astype(np.float32, copy=False)
    baseline = features[1].astype(np.float32, copy=False)
    delta = candidate - baseline
    if feature_mode == "delta_plus_meta":
        return np.concatenate((delta, meta)).astype(np.float32, copy=False)
    return np.concatenate((candidate, delta, meta)).astype(np.float32, copy=False)


def _label_name(row: dict[str, Any]) -> str:
    raw = str(row.get("safe_override_label") or "").strip().lower()
    if raw in {"positive", "negative", "gray"}:
        return raw
    label_id = row.get("safe_override_label_id")
    if label_id is not None:
        return {-1: "gray", 0: "negative", 1: "positive", 2: "positive"}.get(
            int(label_id),
            "",
        )
    return ""


def build_selector_data(
    rows: Iterable[dict[str, Any]],
    *,
    feature_mode: str,
    gray_weight: float = 0.10,
    positive_weight: float = 1.0,
    negative_weight: float = 2.0,
    high_mc_weight: float = 1.0,
    allowed_seats: tuple[str, ...] | None = ("first",),
) -> SelectorData:
    if (
        gray_weight < 0.0
        or positive_weight <= 0.0
        or negative_weight <= 0.0
        or high_mc_weight <= 0.0
    ):
        raise ValueError("selector weights must be non-negative and class weights positive")
    materialized: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    deltas: list[float] = []
    label_names: list[str] = []
    for row in rows:
        if allowed_seats is not None and str(row.get("seat")) not in allowed_seats:
            continue
        label_name = _label_name(row)
        if label_name not in {"positive", "negative", "gray"}:
            continue
        if row.get("action_mapping_verified") is False:
            continue
        if row.get("common_random_futures_verified") is False:
            continue
        features.append(row_to_feature_vector(row, feature_mode=feature_mode))
        labels.append(1 if label_name == "positive" else 0)
        if label_name == "positive":
            weight = positive_weight
        elif label_name == "negative":
            delta = safe_float(row.get("candidate_delta_vs_baseline"))
            weight = negative_weight * min(4.0, max(1.0, abs(min(0.0, delta))))
        else:
            weight = gray_weight
        future_samples = int(
            safe_float(
                row.get("selector_label_future_samples", row.get("future_samples")),
                0.0,
            )
        )
        if future_samples >= 512:
            weight *= high_mc_weight
        weights.append(weight)
        deltas.append(safe_float(row.get("candidate_delta_vs_baseline")))
        label_names.append(label_name)
        materialized.append(row)
    if not features:
        raise ValueError("no usable T0 paired-replay rows")
    if len(set(labels)) < 2:
        raise ValueError("T0 selector data must include positive and non-positive rows")
    return SelectorData(
        features=np.vstack(features).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
        deltas=np.asarray(deltas, dtype=np.float64),
        label_names=tuple(label_names),
        rows=tuple(materialized),
        feature_mode=feature_mode,
    )


def stable_fold_assignments(data: SelectorData, folds: int = 5) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least two")
    output = np.zeros(len(data.rows), dtype=np.int16)
    for label_name in ("positive", "negative", "gray"):
        indices = [index for index, value in enumerate(data.label_names) if value == label_name]
        indices.sort(key=lambda index: _stable_row_key(data.rows[index]))
        for position, index in enumerate(indices):
            output[index] = position % folds
    return output


def _stable_row_key(row: dict[str, Any]) -> str:
    payload = "|".join(
        str(row.get(key, ""))
        for key in (
            "target_id",
            "hand_seed",
            "seat",
            "candidate_action_index",
            "baseline_action_index",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    hit = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            hit += 1
            total += hit / rank
    return float(total / positives)


def _model_factories(random_state: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_balanced": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient": lambda: HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=2.0,
            random_state=random_state,
        ),
    }


def _fit(model: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    if hasattr(model, "steps"):
        model.fit(x, y, **{f"{model.steps[-1][0]}__sample_weight": weights})
    else:
        model.fit(x, y, sample_weight=weights)


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    decision = np.asarray(model.decision_function(x), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-decision))


def threshold_metrics(
    data: SelectorData,
    scores: np.ndarray,
    *,
    model_name: str,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    label_names = np.asarray(data.label_names, dtype=object)
    for threshold in thresholds:
        fired = scores >= threshold
        count = int(np.sum(fired))
        positive = int(np.sum(label_names[fired] == "positive")) if count else 0
        negative = int(np.sum(label_names[fired] == "negative")) if count else 0
        gray = count - positive - negative
        deltas = data.deltas[fired]
        delta_mean = float(np.mean(deltas)) if count else 0.0
        delta_se = (
            float(np.std(deltas, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        )
        losses = np.maximum(-deltas, 0.0) if count else np.zeros(0, dtype=np.float64)
        output.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "fires": count,
                "fire_rate": count / len(data.rows),
                "safe_positive_count": positive,
                "hard_negative_count": negative,
                "gray_count": gray,
                "precision_safe_positive": positive / count if count else 0.0,
                "hard_negative_rate": negative / count if count else 0.0,
                "mean_mc_delta": delta_mean,
                "mc_delta_se": delta_se,
                "mc_delta_ci95_low": delta_mean - 1.96 * delta_se,
                "mc_delta_ci95_high": delta_mean + 1.96 * delta_se,
                "min_mc_delta": float(np.min(deltas)) if count else 0.0,
                "p95_mc_loss": float(np.quantile(losses, 0.95)) if count else 0.0,
                "max_mc_loss": float(np.max(losses)) if count else 0.0,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_selector(
    data: SelectorData,
    *,
    output_dir: Path,
    model_output: Path,
    folds: int = 5,
    random_state: int = 20260712,
) -> dict[str, Any]:
    assignments = stable_fold_assignments(data, folds=folds)
    factories = _model_factories(random_state)
    model_metrics: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    all_oof: dict[str, np.ndarray] = {}
    thresholds = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)

    for model_name, factory in factories.items():
        scores = np.zeros(len(data.rows), dtype=np.float64)
        for fold in range(folds):
            train_mask = assignments != fold
            test_mask = assignments == fold
            if not np.any(test_mask) or len(set(data.labels[train_mask].tolist())) < 2:
                continue
            model = factory()
            _fit(model, data.features[train_mask], data.labels[train_mask], data.weights[train_mask])
            scores[test_mask] = _predict(model, data.features[test_mask])
        all_oof[model_name] = scores
        ap = average_precision(data.labels, scores)
        model_metrics.append(
            {
                "model": model_name,
                "rows": len(data.rows),
                "positives": int(np.sum(data.labels == 1)),
                "average_precision_oof": ap,
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
            }
        )
        threshold_rows.extend(
            threshold_metrics(data, scores, model_name=model_name, thresholds=thresholds)
        )

    if not model_metrics:
        raise ValueError("no T0 selector model could be trained")
    best_name = max(model_metrics, key=lambda row: float(row["average_precision_oof"]))["model"]
    best_scores = all_oof[str(best_name)]
    best_model = factories[str(best_name)]()
    _fit(best_model, data.features, data.labels, data.weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "model_metrics.csv", model_metrics)
    write_csv(output_dir / "threshold_sweep_oof.csv", threshold_rows)
    prediction_rows = []
    for index, row in enumerate(data.rows):
        prediction_rows.append(
            {
                "target_id": row.get("target_id", ""),
                "fold": int(assignments[index]),
                "seat": row.get("seat", ""),
                "source_config_id": row.get("source_config_id", ""),
                "safe_override_label": data.label_names[index],
                "candidate_delta_vs_baseline": float(data.deltas[index]),
                "predicted_margin": _first_present(row, ("predicted_margin", "hu_turn0_predicted_margin")),
                "safe_probability_oof": float(best_scores[index]),
            }
        )
    write_csv(output_dir / "oof_predictions.csv", prediction_rows)

    payload = {
        "model_kind": "hu_turn0_safe_override_selector_sklearn",
        "schema_version": 1,
        "model_name": best_name,
        "feature_mode": data.feature_mode,
        "feature_dim": int(data.features.shape[1]),
        "base_hu_feature_dim": HU_FEATURE_DIM,
        "meta_feature_names": META_FEATURE_NAMES,
        "label_definition": {
            "positive": "paired_mc_candidate_delta_lcb196_gt_0",
            "negative": "paired_mc_candidate_delta_ucb196_lt_0_or_delta_le_minus_0p25",
            "gray": "otherwise_low_weight",
        },
        "estimator": best_model,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "schema": "hu_turn0_safe_override_selector_training_v1",
        "rows": len(data.rows),
        "positive": data.label_names.count("positive"),
        "negative": data.label_names.count("negative"),
        "gray": data.label_names.count("gray"),
        "feature_mode": data.feature_mode,
        "feature_dim": int(data.features.shape[1]),
        "folds": folds,
        "best_model": best_name,
        "best_oof_average_precision": max(
            float(row["average_precision_oof"]) for row in model_metrics
        ),
        "model_output": str(model_output),
        "runtime_status": "not_approved_until_fresh_whole_game_holdout",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="delta_plus_meta")
    parser.add_argument("--gray-weight", type=float, default=0.10)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    parser.add_argument("--high-mc-weight", type=float, default=1.0)
    parser.add_argument(
        "--allowed-seats",
        default="first",
        help="Comma-separated seat list. Use an empty string to keep all seats.",
    )
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = [row for path in args.input for row in read_jsonl(path)]
    allowed_seats = tuple(
        seat.strip() for seat in args.allowed_seats.split(",") if seat.strip()
    ) or None
    if allowed_seats is not None and any(
        seat not in {"first", "second"} for seat in allowed_seats
    ):
        raise SystemExit("--allowed-seats must contain only first and/or second")
    data = build_selector_data(
        rows,
        feature_mode=args.feature_mode,
        gray_weight=args.gray_weight,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        high_mc_weight=args.high_mc_weight,
        allowed_seats=allowed_seats,
    )
    summary = train_selector(
        data,
        output_dir=args.output_dir,
        model_output=args.model_output,
        folds=args.folds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
