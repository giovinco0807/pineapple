"""Train a group-level switch gate between a full T2 selector and a union-pool selector.

The gate is meant for Top1 diagnostics: keep a stable full-candidate selector
as the default, and switch to a small union-pool reranker only when runtime
features say the pool candidate is likely better.  Teacher EV is used only for
the switch target and evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_selector_feature_ranker import (
    DataSpec,
    ModelSpec,
    SelectorData,
    compact_metrics,
    load_selector_scores,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
)
from ai.training.train_t2_union_pool_reranker import prediction_for_dataset


@dataclass
class GroupGateRows:
    x: np.ndarray
    y: np.ndarray
    advantage: np.ndarray
    weights: np.ndarray


def rank_positions(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def top_margin(values: np.ndarray, order: np.ndarray, k: int) -> float:
    if len(order) <= k:
        return 0.0
    return float(values[order[0]] - values[order[k]])


def make_model(kind: str, seed: int):
    if kind == "logreg_bal":
        return LogisticRegression(max_iter=1000, class_weight="balanced", C=0.3)
    if kind == "hgb_l15":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            min_samples_leaf=30,
            random_state=seed,
        )
    if kind == "hgb_l31":
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=420,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            min_samples_leaf=24,
            random_state=seed,
        )
    if kind == "extra_d8":
        return ExtraTreesClassifier(
            n_estimators=240,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "hgb_reg_l15":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=360,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            min_samples_leaf=30,
            random_state=seed,
        )
    if kind == "hgb_reg_l31":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=460,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            min_samples_leaf=24,
            random_state=seed,
        )
    if kind == "extra_reg_d8":
        return ExtraTreesRegressor(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.75,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def is_classifier(kind: str) -> bool:
    return kind in {"logreg_bal", "hgb_l15", "hgb_l31", "extra_d8"}


def group_gate_features(
    ds: SelectorData,
    default_pred: np.ndarray,
    pool_pred: np.ndarray,
    positive_advantage: float,
    weight_scale: float,
    max_weight: float,
) -> GroupGateRows:
    rows: list[list[float]] = []
    labels: list[int] = []
    advantages: list[float] = []
    weights: list[float] = []
    for start, end in ds.bounds:
        true = np.asarray(ds.scores[start:end], dtype=np.float64)
        default = np.asarray(default_pred[start:end], dtype=np.float64)
        pool = np.asarray(pool_pred[start:end], dtype=np.float64)
        base = np.asarray(ds.base_scores[start:end], dtype=np.float64)
        default_order = np.argsort(-default)
        pool_order = np.argsort(-pool)
        base_order = np.argsort(-base)
        default_ranks = rank_positions(default_order)
        pool_ranks = rank_positions(pool_order)
        base_ranks = rank_positions(base_order)
        default_top = int(default_order[0])
        pool_top = int(pool_order[0])
        pool_advantage = float(true[pool_top] - true[default_top])
        advantages.append(pool_advantage)
        labels.append(1 if pool_advantage > float(positive_advantage) + 1e-9 else 0)
        weights.append(min(float(max_weight), 1.0 + abs(pool_advantage) * float(weight_scale)))

        selector_votes_pool: list[float] = []
        selector_votes_default: list[float] = []
        selector_ranks_pool: list[float] = []
        selector_ranks_default: list[float] = []
        selector_gaps_pool: list[float] = []
        selector_gaps_default: list[float] = []
        for scores in ds.selector_scores.values():
            local = np.asarray(scores[start:end], dtype=np.float64)
            order = np.argsort(-local)
            ranks = rank_positions(order)
            selector_votes_pool.append(1.0 if int(order[0]) == pool_top else 0.0)
            selector_votes_default.append(1.0 if int(order[0]) == default_top else 0.0)
            selector_ranks_pool.append(float(ranks[pool_top]))
            selector_ranks_default.append(float(ranks[default_top]))
            selector_gaps_pool.append(float(local[order[0]] - local[pool_top]))
            selector_gaps_default.append(float(local[order[0]] - local[default_top]))

        finite_pool = float(np.sum(pool > -1.0e8))
        rows.append(
            [
                float(end - start),
                finite_pool,
                1.0 if default_top == pool_top else 0.0,
                top_margin(default, default_order, 1),
                top_margin(default, default_order, 2),
                top_margin(pool, pool_order, 1),
                top_margin(pool, pool_order, 2),
                top_margin(base, base_order, 1),
                top_margin(base, base_order, 2),
                float(default[default_top] - default[pool_top]),
                float(pool[pool_top] - pool[default_top]),
                float(base[default_top] - base[pool_top]),
                float(default_ranks[pool_top]),
                float(pool_ranks[default_top]),
                float(base_ranks[default_top]),
                float(base_ranks[pool_top]),
                float(np.mean(selector_votes_pool)),
                float(np.mean(selector_votes_default)),
                float(np.min(selector_ranks_pool)),
                float(np.min(selector_ranks_default)),
                float(np.mean(selector_gaps_pool)),
                float(np.mean(selector_gaps_default)),
            ]
        )
    return GroupGateRows(
        x=np.asarray(rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int32),
        advantage=np.asarray(advantages, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
    )


def concat_metrics(
    datasets: list[SelectorData],
    pred_by_dataset: dict[str, np.ndarray],
    topks: list[int],
) -> tuple[dict, dict]:
    scores: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    per_dataset: dict[str, dict] = {}
    offset = 0
    for ds in datasets:
        pred = pred_by_dataset[ds.spec.name]
        metrics, _ = summarize_groups(ds.scores, pred, ds.bounds, topks)
        per_dataset[ds.spec.name] = compact_metrics(metrics)
        scores.append(ds.scores)
        preds.append(pred)
        bounds.extend((start + offset, end + offset) for start, end in ds.bounds)
        offset += len(ds.scores)
    aggregate, _ = summarize_groups(np.concatenate(scores), np.concatenate(preds), bounds, topks)
    return compact_metrics(aggregate), per_dataset


def switch_predictions(
    datasets: list[SelectorData],
    default_by_dataset: dict[str, np.ndarray],
    pool_by_dataset: dict[str, np.ndarray],
    switch_by_dataset: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int, int]:
    pred_by_dataset: dict[str, np.ndarray] = {}
    switches = 0
    groups = 0
    for ds in datasets:
        pred = np.asarray(default_by_dataset[ds.spec.name], dtype=np.float32).copy()
        pool_pred = pool_by_dataset[ds.spec.name]
        group_switches = switch_by_dataset[ds.spec.name]
        for group_idx, (start, end) in enumerate(ds.bounds):
            if bool(group_switches[group_idx]):
                local = pool_pred[start:end]
                pred[start:end] = -1.0e9
                pred[start + int(np.argmax(local))] = float(np.max(local))
                switches += 1
            groups += 1
        pred_by_dataset[ds.spec.name] = pred
    return pred_by_dataset, switches, groups


def boost_predictions(
    datasets: list[SelectorData],
    default_by_dataset: dict[str, np.ndarray],
    pool_by_dataset: dict[str, np.ndarray],
    switch_by_dataset: dict[str, np.ndarray],
    boost: float,
) -> tuple[dict[str, np.ndarray], int, int]:
    pred_by_dataset: dict[str, np.ndarray] = {}
    boosts = 0
    groups = 0
    for ds in datasets:
        pred = np.asarray(default_by_dataset[ds.spec.name], dtype=np.float32).copy()
        pool_pred = pool_by_dataset[ds.spec.name]
        group_switches = switch_by_dataset[ds.spec.name]
        for group_idx, (start, end) in enumerate(ds.bounds):
            if bool(group_switches[group_idx]):
                pool_top = start + int(np.argmax(pool_pred[start:end]))
                pred[pool_top] += float(boost)
                boosts += 1
            groups += 1
        pred_by_dataset[ds.spec.name] = pred
    return pred_by_dataset, boosts, groups


def model_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def load_sets(
    specs: list[DataSpec],
    model_specs: dict[str, ModelSpec],
    selectors,
    lgbm_selectors,
    loaded_models: dict[str, object],
    loaded_lgbm: dict[str, object],
) -> list[SelectorData]:
    return [
        load_selector_scores(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            include_base=True,
        )
        for spec in specs
    ]


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    train_specs = [parse_named_path(value) for value in args.train_data]
    eval_specs = [parse_named_path(value) for value in args.eval_data]
    model_specs = {spec.name: spec for spec in [parse_model(value) for value in args.model]}
    selectors = [parse_selector(value) for value in args.selector]
    lgbm_selectors = [parse_lgbm_selector(value) for value in args.lgbm_selector]
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    pool_model = joblib.load(args.pool_model)

    train_sets = load_sets(train_specs, model_specs, selectors, lgbm_selectors, loaded_models, loaded_lgbm)
    eval_sets = load_sets(eval_specs, model_specs, selectors, lgbm_selectors, loaded_models, loaded_lgbm)
    train_default = {ds.spec.name: ds.selector_scores[args.default_selector] for ds in train_sets}
    eval_default = {ds.spec.name: ds.selector_scores[args.default_selector] for ds in eval_sets}
    train_pool = {
        ds.spec.name: prediction_for_dataset(ds, pool_model, args.pool_model_kind, args.pool_k, args.pool_feature_scope)
        for ds in train_sets
    }
    eval_pool = {
        ds.spec.name: prediction_for_dataset(ds, pool_model, args.pool_model_kind, args.pool_k, args.pool_feature_scope)
        for ds in eval_sets
    }

    train_rows = [
        group_gate_features(
            ds,
            train_default[ds.spec.name],
            train_pool[ds.spec.name],
            args.positive_advantage,
            args.weight_scale,
            args.max_weight,
        )
        for ds in train_sets
    ]
    eval_rows_by_dataset = {
        ds.spec.name: group_gate_features(
            ds,
            eval_default[ds.spec.name],
            eval_pool[ds.spec.name],
            args.positive_advantage,
            args.weight_scale,
            args.max_weight,
        )
        for ds in eval_sets
    }
    x_train = np.concatenate([rows.x for rows in train_rows], axis=0)
    y_train = np.concatenate([rows.y for rows in train_rows], axis=0)
    advantage_train = np.concatenate([rows.advantage for rows in train_rows], axis=0)
    train_weights = np.concatenate([rows.weights for rows in train_rows], axis=0)

    baseline_default, baseline_default_per = concat_metrics(eval_sets, eval_default, topks)
    baseline_pool, baseline_pool_per = concat_metrics(eval_sets, eval_pool, topks)
    rows_out: list[dict] = []
    boost_rows_out: list[dict] = []
    thresholds = [float(part) for part in args.thresholds.split(",") if part.strip()]
    boosts = [float(part) for part in args.boosts.split(",") if part.strip()]
    for model_idx, kind in enumerate(args.model_kind):
        model = make_model(kind, args.seed + model_idx)
        target = y_train if is_classifier(kind) else advantage_train
        model.fit(x_train, target, sample_weight=train_weights)
        joblib.dump(model, out_dir / f"{kind}.joblib")
        for threshold in thresholds:
            switch_by_dataset = {
                name: model_scores(model, rows.x) >= float(threshold)
                for name, rows in eval_rows_by_dataset.items()
            }
            pred_by_dataset, switches, groups = switch_predictions(
                eval_sets,
                eval_default,
                eval_pool,
                switch_by_dataset,
            )
            aggregate, per_dataset = concat_metrics(eval_sets, pred_by_dataset, topks)
            rows_out.append(
                {
                    "model": kind,
                    "threshold": float(threshold),
                    "switches": int(switches),
                    "groups": int(groups),
                    "aggregate": aggregate,
                    "datasets": per_dataset,
                }
            )
            for boost in boosts:
                boosted_pred_by_dataset, boosted, boost_groups = boost_predictions(
                    eval_sets,
                    eval_default,
                    eval_pool,
                    switch_by_dataset,
                    boost=boost,
                )
                boost_aggregate, boost_per_dataset = concat_metrics(eval_sets, boosted_pred_by_dataset, topks)
                boost_rows_out.append(
                    {
                        "model": kind,
                        "threshold": float(threshold),
                        "boost": float(boost),
                        "boosted": int(boosted),
                        "groups": int(boost_groups),
                        "aggregate": boost_aggregate,
                        "datasets": boost_per_dataset,
                    }
                )
    rows_out.sort(
        key=lambda row: (
            float(row["aggregate"].get("group_top1", 0.0)),
            -float(row["aggregate"].get("group_top1_regret", 0.0)),
            float(row["aggregate"].get("group_top3", 0.0)),
        ),
        reverse=True,
    )
    boost_rows_out.sort(
        key=lambda row: (
            float(row["aggregate"].get("group_top1", 0.0)),
            -float(row["aggregate"].get("group_top1_regret", 0.0)),
            float(row["aggregate"].get("group_top3", 0.0)),
        ),
        reverse=True,
    )
    summary = {
        "default_selector": args.default_selector,
        "pool_model": str(args.pool_model),
        "pool_model_kind": args.pool_model_kind,
        "pool_k": int(args.pool_k),
        "pool_feature_scope": args.pool_feature_scope,
        "train_data": {spec.name: str(spec.path) for spec in train_specs},
        "eval_data": {spec.name: str(spec.path) for spec in eval_specs},
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": [args.default_selector] + [selector.name for selector in selectors] + [name for name, _ in lgbm_selectors],
        "model_kinds": args.model_kind,
        "thresholds": thresholds,
        "boosts": boosts,
        "topks": topks,
        "train_groups": int(len(y_train)),
        "train_positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "train_advantage_mean": float(advantage_train.mean()) if len(advantage_train) else 0.0,
        "train_advantage_positive_rate": float(np.mean(advantage_train > 0.0)) if len(advantage_train) else 0.0,
        "positive_advantage": float(args.positive_advantage),
        "weight_scale": float(args.weight_scale),
        "max_weight": float(args.max_weight),
        "train_weight_mean": float(train_weights.mean()) if len(train_weights) else 0.0,
        "train_weight_max": float(train_weights.max()) if len(train_weights) else 0.0,
        "baseline_default": {"aggregate": baseline_default, "datasets": baseline_default_per},
        "baseline_pool": {"aggregate": baseline_pool, "datasets": baseline_pool_per},
        "rows": rows_out,
        "boost_rows": boost_rows_out,
        "elapsed_seconds": time.time() - started,
        "note": "Threshold rows are diagnostic unless the threshold was chosen without looking at the eval set.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", action="append", required=True, help="name=path")
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument("--default-selector", required=True)
    parser.add_argument("--pool-model", type=Path, required=True)
    parser.add_argument("--pool-model-kind", default="hgb_cls_l31")
    parser.add_argument("--pool-k", type=int, default=2)
    parser.add_argument("--pool-feature-scope", choices=("both", "selector", "state"), default="both")
    parser.add_argument("--model-kind", action="append", default=[])
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--boosts", default="")
    parser.add_argument("--positive-advantage", type=float, default=0.0)
    parser.add_argument("--weight-scale", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=8.0)
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.model_kind:
        args.model_kind = ["logreg_bal", "hgb_l15", "hgb_l31", "extra_d8"]
    run(args)


if __name__ == "__main__":
    main()
