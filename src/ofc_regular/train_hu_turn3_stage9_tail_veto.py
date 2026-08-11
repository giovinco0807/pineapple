"""Train a small HU T3 Stage9 tail-risk veto smoke classifier.

This is a research helper, not a production runtime. It trains a classifier
from the flat CSV produced by prepare_hu_turn3_stage9_tail_veto_dataset.py and
reports whether runtime-available scalar features can identify MC replay hard
negative fired states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


FEATURE_COLUMNS = [
    "predicted_margin_vs_baseline",
    "reference_margin",
    "model_score",
    "best_score",
    "legal_action_count",
    "hu_top_full",
    "hu_middle_full",
    "hu_bottom_full",
    "baseline_top_full",
    "baseline_middle_full",
    "baseline_bottom_full",
    "hu_places_top_count",
    "hu_places_middle_count",
    "hu_places_bottom_count",
    "baseline_places_top_count",
    "baseline_places_middle_count",
    "baseline_places_bottom_count",
    "hu_discard_count",
    "baseline_discard_count",
    "hu_discard_rank_max",
    "baseline_discard_rank_max",
    "hu_placed_rank_sum",
    "baseline_placed_rank_sum",
    "hu_placed_rank_max",
    "baseline_placed_rank_max",
    "hu_fills_more_rows_than_baseline",
    "hu_discards_higher_rank_than_baseline",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return 1.0 if text.lower() == "true" else 0.0
    try:
        result = float(text)
    except ValueError:
        return default
    if not math.isfinite(result):
        return default
    return result


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    data = [[_to_float(row.get(col)) for col in FEATURE_COLUMNS] for row in rows]
    return np.asarray(data, dtype=np.float32)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if str(row.get("tail_label")) == "hard_negative" else 0 for row in rows], dtype=np.int64)


def source_split(rows: list[dict[str, Any]], holdout_source: str | None = None) -> tuple[list[int], list[int], str]:
    sources = sorted({str(row.get("source_name", "")) for row in rows})
    if not sources:
        raise ValueError("dataset has no source_name values")
    if holdout_source is None:
        # Prefer holding out the source with the most hard negatives so the
        # smoke test actually evaluates positive examples when possible.
        by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
        for row in rows:
            by_source[str(row.get("source_name", ""))].append(row)
        holdout_source = max(
            sources,
            key=lambda source: (
                sum(1 for row in by_source[source] if str(row.get("tail_label")) == "hard_negative"),
                len(by_source[source]),
            ),
        )
    train_idx = [idx for idx, row in enumerate(rows) if str(row.get("source_name", "")) != holdout_source]
    test_idx = [idx for idx, row in enumerate(rows) if str(row.get("source_name", "")) == holdout_source]
    if not train_idx or not test_idx:
        raise ValueError(f"invalid holdout source split: {holdout_source}")
    return train_idx, test_idx, holdout_source


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(y_true[idx]) == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        wins += float(np.sum(pos > negatives))
        wins += 0.5 * float(np.sum(pos == negatives))
    return wins / total


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = scores >= threshold
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(y_true)),
        "hard_negatives": int(y_true.sum()),
        "average_precision": average_precision(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(rows: list[dict[str, Any]], train_idx: list[int], seed: int) -> ExtraTreesClassifier:
    X = feature_matrix(rows)
    y = labels(rows)
    train_y = y[train_idx]
    class_weight = "balanced" if len(set(train_y.tolist())) > 1 else None
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=seed,
        class_weight=class_weight,
    )
    model.fit(X[train_idx], train_y)
    return model


def predict_hard_negative_probability(model: ExtraTreesClassifier, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=np.float32)
    return probabilities[:, classes.index(1)]


def write_scores(path: Path, rows: list[dict[str, Any]], scores: np.ndarray, split: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_row_id",
        "split",
        "source_name",
        "seed",
        "tail_label",
        "delta_vs_baseline",
        "hard_negative_probability",
        *FEATURE_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, score, split_name in zip(rows, scores, split):
            output = {name: row.get(name) for name in fieldnames}
            output["split"] = split_name
            output["hard_negative_probability"] = float(score)
            writer.writerow(output)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_lines = [
        f"- {name}: {importance:.4f}"
        for name, importance in summary.get("feature_importance", [])
    ]
    train = summary["train_metrics"]
    test = summary["test_metrics"]
    label_counts = summary["label_counts"]
    text = "\n".join(
        [
            "# HU Turn3 Stage9 Tail Veto Smoke Training",
            "",
            "## Dataset",
            "",
            f"- rows: {summary['rows']}",
            f"- holdout source: `{summary['holdout_source']}`",
            f"- safe_positive: {label_counts.get('safe_positive', 0)}",
            f"- gray: {label_counts.get('gray', 0)}",
            f"- hard_negative: {label_counts.get('hard_negative', 0)}",
            "",
            "## Train Metrics",
            "",
            f"- rows: {train['rows']}",
            f"- hard negatives: {train['hard_negatives']}",
            f"- AP: {train['average_precision']:.4f}",
            f"- ROC AUC: {train['roc_auc']:.4f}",
            f"- precision@0.5: {train['precision']:.4f}",
            f"- recall@0.5: {train['recall']:.4f}",
            "",
            "## Holdout Metrics",
            "",
            f"- rows: {test['rows']}",
            f"- hard negatives: {test['hard_negatives']}",
            f"- AP: {test['average_precision']:.4f}",
            f"- ROC AUC: {test['roc_auc']:.4f}",
            f"- precision@0.5: {test['precision']:.4f}",
            f"- recall@0.5: {test['recall']:.4f}",
            "",
            "## Feature Importance",
            "",
            *feature_lines,
            "",
            "## Interpretation",
            "",
            "This is a smoke classifier only. The dataset is too small and source-biased for production use, but this verifies the tail-veto training path and exposes whether runtime scalar features carry any hard-negative signal.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--scores-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--holdout-source", default=None)
    parser.add_argument("--seed", type=int, default=2026061801)
    args = parser.parse_args()

    rows = _read_csv(args.input)
    train_idx, test_idx, holdout_source = source_split(rows, args.holdout_source)
    model = train_model(rows, train_idx, args.seed)
    X = feature_matrix(rows)
    y = labels(rows)
    scores = predict_hard_negative_probability(model, X)
    split_names = ["test" if idx in set(test_idx) else "train" for idx in range(len(rows))]

    train_metrics = metrics(y[train_idx], scores[train_idx])
    test_metrics = metrics(y[test_idx], scores[test_idx])
    importance = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda item: -float(item[1]))
    summary = {
        "rows": len(rows),
        "input": str(args.input),
        "model_output": str(args.model_output),
        "holdout_source": holdout_source,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "label_counts": dict(Counter(row.get("tail_label") for row in rows)),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": [(name, float(value)) for name, value in importance],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "label": "hard_negative_probability",
            "holdout_source": holdout_source,
        },
        args.model_output,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_scores(args.scores_output, rows, scores, split_names)
    write_summary(args.summary_output, summary)


if __name__ == "__main__":
    main()
