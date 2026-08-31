"""Evaluate no-leak arbitration between two T2 meta-rankers.

This script compares a fixed HGB-style score selector against a LightGBM
listwise ranker, then trains small group-level gates that choose one selector
per dealt T2 spot.  Gate inputs are limited to runtime-available prediction
features: model margins, cross-ranks, base-score margins, and group size.
Teacher EV is used only to build the switch label and evaluate the result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_sklearn_meta_ranker import compact_metrics, load_runtime_features


EPS = 1e-9


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    target: str


@dataclass
class DatasetPredictions:
    name: str
    path: Path
    scores: np.ndarray
    base_scores: np.ndarray
    hgb_scores: np.ndarray
    lgbm_scores: np.ndarray
    bounds: list[tuple[int, int]]
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]


def parse_named_path(value: str) -> tuple[str, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("expected name=path")
    return parts[0].strip(), Path(parts[1].strip())


def parse_model(value: str) -> ModelSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--hgb-model must be name=path,target")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--hgb-model must be name=path,target")
    target = items[1].lower()
    if target not in {"score", "residual"}:
        raise ValueError("model target must be score or residual")
    return ModelSpec(name=name, path=Path(items[0]), target=target)


def rank_positions(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def top_margin(values: np.ndarray, order: np.ndarray, k: int) -> float:
    if len(order) <= k:
        return 0.0
    return float(values[order[0]] - values[order[k]])


def z_gap_to_top(values: np.ndarray, top_idx: int, other_idx: int) -> float:
    std = float(np.std(values))
    if std < 1e-6:
        std = 1.0
    return float((values[top_idx] - values[other_idx]) / std)


def build_group_features(
    true_scores: np.ndarray,
    base_scores: np.ndarray,
    hgb_scores: np.ndarray,
    lgbm_scores: np.ndarray,
    bounds: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = [
        "group_size",
        "base_margin_1",
        "base_margin_3",
        "hgb_margin_1",
        "hgb_margin_3",
        "lgbm_margin_1",
        "lgbm_margin_3",
        "margin1_delta_lgbm_minus_hgb",
        "margin3_delta_lgbm_minus_hgb",
        "same_top1",
        "lgbm_top_rank_under_hgb",
        "hgb_top_rank_under_lgbm",
        "hgb_top_base_rank",
        "lgbm_top_base_rank",
        "hgb_top_lgbm_rank_gap",
        "lgbm_top_hgb_rank_gap",
        "hgb_score_z_gap_to_lgbm_top",
        "lgbm_score_z_gap_to_hgb_top",
        "base_score_z_gap_hgb_minus_lgbm",
    ]
    rows: list[list[float]] = []
    labels: list[int] = []
    for start, end in bounds:
        true = np.asarray(true_scores[start:end], dtype=np.float64)
        base = np.asarray(base_scores[start:end], dtype=np.float64)
        hgb = np.asarray(hgb_scores[start:end], dtype=np.float64)
        lgbm = np.asarray(lgbm_scores[start:end], dtype=np.float64)
        if len(true) == 0:
            continue
        base_order = np.argsort(-base)
        hgb_order = np.argsort(-hgb)
        lgbm_order = np.argsort(-lgbm)
        base_ranks = rank_positions(base_order)
        hgb_ranks = rank_positions(hgb_order)
        lgbm_ranks = rank_positions(lgbm_order)
        h_top = int(hgb_order[0])
        l_top = int(lgbm_order[0])
        best = float(np.max(true))
        hgb_regret = best - float(true[h_top])
        lgbm_regret = best - float(true[l_top])
        labels.append(1 if lgbm_regret + EPS < hgb_regret else 0)
        rows.append(
            [
                float(len(true)),
                top_margin(base, base_order, 1),
                top_margin(base, base_order, 2),
                top_margin(hgb, hgb_order, 1),
                top_margin(hgb, hgb_order, 2),
                top_margin(lgbm, lgbm_order, 1),
                top_margin(lgbm, lgbm_order, 2),
                top_margin(lgbm, lgbm_order, 1) - top_margin(hgb, hgb_order, 1),
                top_margin(lgbm, lgbm_order, 2) - top_margin(hgb, hgb_order, 2),
                1.0 if h_top == l_top else 0.0,
                float(hgb_ranks[l_top]),
                float(lgbm_ranks[h_top]),
                float(base_ranks[h_top]),
                float(base_ranks[l_top]),
                float(hgb_ranks[l_top] - 1),
                float(lgbm_ranks[h_top] - 1),
                z_gap_to_top(hgb, h_top, l_top),
                z_gap_to_top(lgbm, l_top, h_top),
                z_gap_to_top(base, h_top, l_top),
            ]
        )
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), feature_names


def load_predictions(
    name: str,
    path: Path,
    hgb_models: dict[str, object],
    hgb_specs: list[ModelSpec],
    hgb_gamma: float,
    lgbm_model: object,
) -> DatasetPredictions:
    x, scores, base_scores, bounds = load_runtime_features(path)
    residuals: list[np.ndarray] = []
    for spec in hgb_specs:
        pred = hgb_models[spec.name].predict(x).astype(np.float32)
        residuals.append(pred if spec.target == "residual" else pred - base_scores)
    hgb_scores = base_scores + float(hgb_gamma) * np.mean(residuals, axis=0).astype(np.float32)
    lgbm_scores = lgbm_model.predict(x).astype(np.float32)
    features, labels, feature_names = build_group_features(
        scores,
        base_scores,
        hgb_scores,
        lgbm_scores,
        bounds,
    )
    return DatasetPredictions(
        name=name,
        path=path,
        scores=scores,
        base_scores=base_scores,
        hgb_scores=hgb_scores.astype(np.float32),
        lgbm_scores=lgbm_scores.astype(np.float32),
        bounds=bounds,
        features=features,
        labels=labels,
        feature_names=feature_names,
    )


def concat_group_features(datasets: list[DatasetPredictions]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([ds.features for ds in datasets], axis=0),
        np.concatenate([ds.labels for ds in datasets], axis=0),
    )


def switched_prediction(ds: DatasetPredictions, switches: np.ndarray) -> np.ndarray:
    if len(switches) != len(ds.bounds):
        raise ValueError(f"{ds.name}: switch count does not match group count")
    pred = np.empty_like(ds.hgb_scores, dtype=np.float32)
    for group_idx, (start, end) in enumerate(ds.bounds):
        pred[start:end] = ds.lgbm_scores[start:end] if switches[group_idx] else ds.hgb_scores[start:end]
    return pred


def concat_for_summary(
    datasets: list[DatasetPredictions],
    pred_by_dataset: dict[str, np.ndarray],
    topks: list[int],
) -> tuple[dict, list[dict]]:
    scores: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    offset = 0
    for ds in datasets:
        scores.append(ds.scores)
        pred_values = pred_by_dataset[ds.name]
        pred.append(pred_values)
        bounds.extend((start + offset, end + offset) for start, end in ds.bounds)
        offset += len(ds.scores)
    return summarize_groups(np.concatenate(scores), np.concatenate(pred), bounds, topks)


def summarize_selector(
    datasets: list[DatasetPredictions],
    topks: list[int],
    selector_name: str,
    pred_fn: Callable[[DatasetPredictions], np.ndarray],
) -> dict:
    pred_by_dataset = {ds.name: pred_fn(ds) for ds in datasets}
    aggregate_metrics, _rows = concat_for_summary(datasets, pred_by_dataset, topks)
    per_dataset = {}
    for ds in datasets:
        metrics, _ = summarize_groups(ds.scores, pred_by_dataset[ds.name], ds.bounds, topks)
        per_dataset[ds.name] = compact_metrics(metrics)
    return {
        "selector": selector_name,
        "aggregate": compact_metrics(aggregate_metrics),
        "datasets": per_dataset,
    }


def switch_oracle(ds: DatasetPredictions) -> np.ndarray:
    switches = np.zeros(len(ds.bounds), dtype=bool)
    for group_idx, (start, end) in enumerate(ds.bounds):
        true = np.asarray(ds.scores[start:end], dtype=np.float64)
        best = float(true.max())
        h_top = int(np.argmax(ds.hgb_scores[start:end]))
        l_top = int(np.argmax(ds.lgbm_scores[start:end]))
        h_regret = best - float(true[h_top])
        l_regret = best - float(true[l_top])
        switches[group_idx] = l_regret + EPS < h_regret
    return switches


def evaluate_switch_rule(
    datasets: list[DatasetPredictions],
    topks: list[int],
    selector_name: str,
    switch_fn: Callable[[DatasetPredictions], np.ndarray],
) -> dict:
    pred_by_dataset = {}
    switch_counts = {}
    for ds in datasets:
        switches = np.asarray(switch_fn(ds), dtype=bool)
        pred_by_dataset[ds.name] = switched_prediction(ds, switches)
        switch_counts[ds.name] = {
            "switches": int(switches.sum()),
            "groups": int(len(switches)),
            "switch_rate": float(switches.mean()) if len(switches) else 0.0,
        }
    aggregate_metrics, _rows = concat_for_summary(datasets, pred_by_dataset, topks)
    per_dataset = {}
    for ds in datasets:
        metrics, _ = summarize_groups(ds.scores, pred_by_dataset[ds.name], ds.bounds, topks)
        per_dataset[ds.name] = compact_metrics(metrics)
    return {
        "selector": selector_name,
        "aggregate": compact_metrics(aggregate_metrics),
        "datasets": per_dataset,
        "switch_counts": switch_counts,
    }


def train_threshold_gates(
    train_sets: list[DatasetPredictions],
    topks: list[int],
    max_gates: int,
) -> list[dict]:
    x_train, _y_train = concat_group_features(train_sets)
    feature_names = train_sets[0].feature_names
    candidates: list[dict] = []
    for feature_idx, feature_name in enumerate(feature_names):
        values = x_train[:, feature_idx]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        quantiles = np.unique(np.quantile(finite, np.linspace(0.05, 0.95, 19)))
        for threshold in quantiles:
            for direction in ("ge", "le"):
                if direction == "ge":
                    switch_fn = lambda ds, i=feature_idx, t=float(threshold): ds.features[:, i] >= t
                else:
                    switch_fn = lambda ds, i=feature_idx, t=float(threshold): ds.features[:, i] <= t
                result = evaluate_switch_rule(
                    train_sets,
                    topks,
                    f"threshold:{feature_name}:{direction}:{float(threshold):.6g}",
                    switch_fn,
                )
                m = result["aggregate"]
                candidates.append(
                    {
                        "feature": feature_name,
                        "feature_index": int(feature_idx),
                        "direction": direction,
                        "threshold": float(threshold),
                        "train_metrics": m,
                        "sort_key": (
                            float(m.get("group_top1", 0.0)),
                            -float(m.get("group_top1_regret", 0.0)),
                            float(m.get("group_top3", 0.0)),
                            -float(m.get("group_top10_rerank_regret", 0.0)),
                        ),
                    }
                )
    candidates.sort(key=lambda item: item["sort_key"], reverse=True)
    return candidates[:max_gates]


def train_classifiers(
    train_sets: list[DatasetPredictions],
    topks: list[int],
) -> list[dict]:
    x_train, y_train = concat_group_features(train_sets)
    if len(np.unique(y_train)) < 2:
        return []
    classifiers: list[tuple[str, object]] = [
        (
            "logreg_balanced_c1",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=20260620,
            ),
        ),
        (
            "logreg_balanced_c03",
            LogisticRegression(
                C=0.3,
                class_weight="balanced",
                max_iter=1000,
                random_state=20260621,
            ),
        ),
        (
            "hgb_classifier_l7",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=180,
                max_leaf_nodes=7,
                l2_regularization=0.05,
                min_samples_leaf=12,
                random_state=20260620,
            ),
        ),
    ]
    trained = []
    for name, clf in classifiers:
        clf.fit(x_train, y_train)
        probabilities = clf.predict_proba(x_train)[:, 1]
        best_threshold = 0.5
        best_key = None
        best_metrics = None
        for threshold in np.unique(np.quantile(probabilities, np.linspace(0.1, 0.9, 17))):
            switch_fn = lambda ds, c=clf, t=float(threshold): c.predict_proba(ds.features)[:, 1] >= t
            result = evaluate_switch_rule(train_sets, topks, f"{name}@{float(threshold):.6g}", switch_fn)
            m = result["aggregate"]
            key = (
                float(m.get("group_top1", 0.0)),
                -float(m.get("group_top1_regret", 0.0)),
                float(m.get("group_top3", 0.0)),
                -float(m.get("group_top10_rerank_regret", 0.0)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
                best_metrics = m
        trained.append(
            {
                "name": name,
                "classifier": clf,
                "threshold": best_threshold,
                "train_metrics": best_metrics,
            }
        )
    return trained


def metrics_table(rows: list[dict]) -> list[str]:
    lines = [
        "| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["aggregate"]
        lines.append(
            f"| {row['selector']} | "
            f"{float(m.get('group_top1', 0.0)):.1%} | "
            f"{float(m.get('group_top3', 0.0)):.1%} | "
            f"{float(m.get('group_top5', 0.0)):.1%} | "
            f"{float(m.get('group_top10', 0.0)):.1%} | "
            f"{float(m.get('group_top20', 0.0)):.1%} | "
            f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top10_rerank_regret', 0.0)):.3f} |"
        )
    return lines


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 Meta Arbitration",
        "",
        "- inputs: states, base scores, and selector prediction margins/ranks",
        "- excluded inputs: teacher EV, teacher rank, route tags, bust/FL labels",
        f"- hgb_gamma: `{summary['hgb_gamma']}`",
        "",
        "## Train Aggregate",
        "",
        *metrics_table(summary["train_rows"][:12]),
        "",
        "## Eval Aggregate",
        "",
        *metrics_table(summary["eval_rows"][:16]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    hgb_specs = [parse_model(value) for value in args.hgb_model]
    train_specs = [parse_named_path(value) for value in args.train_data]
    eval_specs = [parse_named_path(value) for value in args.eval_data]

    hgb_models = {spec.name: joblib.load(spec.path) for spec in hgb_specs}
    lgbm_model = joblib.load(Path(args.lgbm_model))

    train_sets = [
        load_predictions(name, path, hgb_models, hgb_specs, args.hgb_gamma, lgbm_model)
        for name, path in train_specs
    ]
    eval_sets = [
        load_predictions(name, path, hgb_models, hgb_specs, args.hgb_gamma, lgbm_model)
        for name, path in eval_specs
    ]

    base_selectors = [
        ("base_scores", lambda ds: ds.base_scores),
        ("hgb_fixed", lambda ds: ds.hgb_scores),
        ("lgbm_direct", lambda ds: ds.lgbm_scores),
    ]
    train_rows = [
        summarize_selector(train_sets, topks, name, pred_fn) for name, pred_fn in base_selectors
    ]
    eval_rows = [
        summarize_selector(eval_sets, topks, name, pred_fn) for name, pred_fn in base_selectors
    ]
    train_rows.append(evaluate_switch_rule(train_sets, topks, "oracle_pair_hgb_vs_lgbm", switch_oracle))
    eval_rows.append(evaluate_switch_rule(eval_sets, topks, "oracle_pair_hgb_vs_lgbm", switch_oracle))

    threshold_gates = train_threshold_gates(train_sets, topks, args.max_threshold_gates)
    for gate in threshold_gates:
        idx = int(gate["feature_index"])
        threshold = float(gate["threshold"])
        direction = gate["direction"]
        if direction == "ge":
            switch_fn = lambda ds, i=idx, t=threshold: ds.features[:, i] >= t
        else:
            switch_fn = lambda ds, i=idx, t=threshold: ds.features[:, i] <= t
        selector = f"threshold:{gate['feature']}:{direction}:{threshold:.6g}"
        train_rows.append(evaluate_switch_rule(train_sets, topks, selector, switch_fn))
        eval_rows.append(evaluate_switch_rule(eval_sets, topks, selector, switch_fn))

    classifier_specs = train_classifiers(train_sets, topks)
    for spec in classifier_specs:
        clf = spec["classifier"]
        threshold = float(spec["threshold"])
        switch_fn = lambda ds, c=clf, t=threshold: c.predict_proba(ds.features)[:, 1] >= t
        selector = f"{spec['name']}@{threshold:.6g}"
        train_rows.append(evaluate_switch_rule(train_sets, topks, selector, switch_fn))
        eval_rows.append(evaluate_switch_rule(eval_sets, topks, selector, switch_fn))

    def sort_key(row: dict) -> tuple[float, float, float, float]:
        m = row["aggregate"]
        return (
            float(m.get("group_top1", 0.0)),
            -float(m.get("group_top1_regret", 0.0)),
            float(m.get("group_top3", 0.0)),
            -float(m.get("group_top10_rerank_regret", 0.0)),
        )

    train_rows.sort(key=sort_key, reverse=True)
    eval_rows.sort(key=sort_key, reverse=True)
    summary = {
        "train_data": {name: str(path) for name, path in train_specs},
        "eval_data": {name: str(path) for name, path in eval_specs},
        "hgb_models": {
            spec.name: {"path": str(spec.path), "target": spec.target} for spec in hgb_specs
        },
        "hgb_gamma": float(args.hgb_gamma),
        "lgbm_model": str(args.lgbm_model),
        "topks": topks,
        "feature_names": train_sets[0].feature_names if train_sets else [],
        "train_groups": int(sum(len(ds.bounds) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.bounds) for ds in eval_sets)),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
        "note": "Classifier and threshold gates are selected on train_data only; eval_data rows are the external replay.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_dir / "summary.md", summary)
    print(
        json.dumps(
            {
                "summary": str(out_dir / "summary.json"),
                "train_groups": summary["train_groups"],
                "eval_groups": summary["eval_groups"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", action="append", required=True, help="name=path")
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--hgb-model", action="append", required=True, help="name=path,target")
    parser.add_argument("--hgb-gamma", type=float, default=1.15)
    parser.add_argument("--lgbm-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--max-threshold-gates", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
