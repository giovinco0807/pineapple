"""Train a T2 reranker inside a selector-feature TopK union pool.

This is the pool version of ``train_t2_selector_feature_ranker.py``.  It keeps
the runtime-only selector score/rank/gap/agreement features, also allows
feature-selector models such as ``xyhard`` to contribute pool members, and then
learns to choose one action from the union pool.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRanker
except ImportError:  # pragma: no cover - optional dependency in minimal envs.
    LGBMRanker = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.evaluate_t2_selector_feature_union import FeatureSelectorSpec, feature_model_input, parse_feature_selector
from ai.training.train_t2_selector_feature_ranker import (
    DataSpec,
    INACTIVE_SELECTOR_SCORE,
    MissWeighting,
    ModelSpec,
    SelectorData,
    predict_adapted,
    build_selector_features,
    compact_metrics,
    load_miss_weighting,
    load_selector_scores,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
)


@dataclass
class PoolRows:
    x: np.ndarray
    y_gap: np.ndarray
    y_best: np.ndarray
    weights: np.ndarray
    group_sizes: list[int]


def make_model(kind: str, seed: int):
    if kind == "lgbm_rank_l31":
        if LGBMRanker is None:
            raise ImportError("lightgbm is required for lgbm_rank_l31")
        return LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            n_estimators=520,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.05,
            label_gain=list(range(32)),
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if kind == "lgbm_rank_l63":
        if LGBMRanker is None:
            raise ImportError("lightgbm is required for lgbm_rank_l63")
        return LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            n_estimators=650,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=14,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.07,
            label_gain=list(range(32)),
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if kind == "hgb_gap_l31":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=520,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=18,
            random_state=seed,
        )
    if kind == "hgb_gap_l63":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.03,
            max_iter=620,
            max_leaf_nodes=63,
            l2_regularization=0.04,
            min_samples_leaf=14,
            random_state=seed,
        )
    if kind == "extra_gap_d14":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "hgb_cls_l31":
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=520,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=18,
            random_state=seed,
        )
    if kind == "hgb_cls_l63":
        return HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=620,
            max_leaf_nodes=63,
            l2_regularization=0.04,
            min_samples_leaf=14,
            random_state=seed,
        )
    if kind == "hgb_cls_l63_leaf4":
        return HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=720,
            max_leaf_nodes=63,
            l2_regularization=0.03,
            min_samples_leaf=4,
            random_state=seed,
        )
    if kind == "extra_cls_d14":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def is_classifier(kind: str) -> bool:
    return kind.startswith("hgb_cls") or kind.startswith("extra_cls")


def is_ranker(kind: str) -> bool:
    return kind.startswith("lgbm_rank")


def rank_relevance(y_gap: np.ndarray, group_sizes: list[int], *, max_label: int = 31, gap_scale: float = 4.0) -> np.ndarray:
    labels = np.zeros(len(y_gap), dtype=np.int32)
    offset = 0
    for group_size in group_sizes:
        end = offset + int(group_size)
        local_gap = np.asarray(y_gap[offset:end], dtype=np.float64)
        # y_gap is candidate_score - group_best, so the best candidate has gap 0.
        loss = np.maximum(0.0, -local_gap)
        raw = np.rint(float(max_label) - loss * float(gap_scale)).astype(np.int32)
        labels[offset:end] = np.clip(raw, 0, int(max_label))
        offset = end
    return labels


def enrich_with_feature_selectors(
    ds: SelectorData,
    feature_selectors: list[FeatureSelectorSpec],
    loaded_feature: dict[str, object],
) -> SelectorData:
    selector_scores = dict(ds.selector_scores)
    for spec in feature_selectors:
        pred = predict_adapted(
            loaded_feature[spec.name],
            feature_model_input(ds, spec.scope, loaded_feature[spec.name]),
        )
        selector_scores[spec.name] = np.asarray(pred, dtype=np.float32)
    features, names = build_selector_features(selector_scores, ds.bounds)
    ds.selector_scores = selector_scores
    ds.selector_features = features
    ds.feature_names = names
    return ds


def load_dataset(
    spec: DataSpec,
    model_specs: dict[str, ModelSpec],
    selectors: list,
    lgbm_selectors: list,
    npy_selectors: list[str],
    loaded_models: dict[str, object],
    loaded_lgbm: dict[str, object],
    feature_selectors: list[FeatureSelectorSpec],
    loaded_feature: dict[str, object],
    include_base: bool,
) -> SelectorData:
    ds = load_selector_scores(
        spec,
        model_specs,
        selectors,
        lgbm_selectors,
        loaded_models,
        loaded_lgbm,
        include_base=include_base,
        npy_selectors=npy_selectors,
    )
    return enrich_with_feature_selectors(ds, feature_selectors, loaded_feature)


def union_members(ds: SelectorData, start: int, end: int, pool_k: int) -> list[int]:
    members: set[int] = set()
    for scores in ds.selector_scores.values():
        local = np.asarray(scores[start:end], dtype=np.float64)
        if len(local) == 0 or float(np.max(local)) <= INACTIVE_SELECTOR_SCORE:
            continue
        order = np.argsort(-local)
        active_order = [int(local_idx) for local_idx in order if float(local[int(local_idx)]) > INACTIVE_SELECTOR_SCORE]
        for local_idx in active_order[: min(pool_k, len(active_order))]:
            members.add(start + int(local_idx))
    return sorted(members)


def pool_local_features(ds: SelectorData, members: list[int]) -> np.ndarray:
    """Runtime-only features computed inside the selected union pool.

    ``build_selector_features`` already captures full-action-space ranks and
    agreement.  Top1 failures often happen after the exact-best action has made
    it into a small union pool, so these features restate each selector's rank,
    margin, and vote pattern within that pool only.
    """
    names = list(ds.selector_scores)
    n = len(members)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    idx = np.asarray(members, dtype=np.int64)
    selector_matrix = np.stack([np.asarray(ds.selector_scores[name][idx], dtype=np.float64) for name in names], axis=1)
    rank_frac = np.zeros_like(selector_matrix)
    gap_to_top = np.zeros_like(selector_matrix)
    margin_to_next = np.zeros_like(selector_matrix)
    top1 = np.zeros_like(selector_matrix)
    top3 = np.zeros_like(selector_matrix)
    top5 = np.zeros_like(selector_matrix)
    for col in range(len(names)):
        values = selector_matrix[:, col]
        if float(np.max(values)) <= INACTIVE_SELECTOR_SCORE:
            rank_frac[:, col] = 1.0
            continue
        order = [int(i) for i in np.argsort(-values) if float(values[int(i)]) > INACTIVE_SELECTOR_SCORE]
        ranks = np.full(n, n, dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)
        rank_frac[:, col] = ranks / max(n - 1, 1)
        gap_to_top[:, col] = float(values[order[0]]) - values if order else 0.0
        for row in range(n):
            if n <= 1:
                margin_to_next[row, col] = 0.0
            else:
                others = np.delete(values, row)
                margin_to_next[row, col] = values[row] - float(np.max(others))
        top1[order[:1], col] = 1.0
        top3[order[: min(3, len(order))], col] = 1.0
        top5[order[: min(5, len(order))], col] = 1.0
    score_mean = selector_matrix.mean(axis=1, keepdims=True)
    score_std = selector_matrix.std(axis=1, keepdims=True)
    rank_mean = rank_frac.mean(axis=1, keepdims=True)
    rank_best = rank_frac.min(axis=1, keepdims=True)
    rank_worst = rank_frac.max(axis=1, keepdims=True)
    votes = np.concatenate(
        [
            top1.sum(axis=1, keepdims=True),
            top3.sum(axis=1, keepdims=True),
            top5.sum(axis=1, keepdims=True),
        ],
        axis=1,
    )
    group_size = np.full((n, 1), n / 30.0, dtype=np.float64)
    return np.concatenate(
        [
            selector_matrix,
            rank_frac,
            gap_to_top,
            margin_to_next,
            top1,
            top3,
            top5,
            score_mean,
            score_std,
            rank_mean,
            rank_best,
            rank_worst,
            votes,
            group_size,
        ],
        axis=1,
    ).astype(np.float32)


def feature_matrix(ds: SelectorData, scope: str) -> np.ndarray:
    if scope == "selector":
        return ds.selector_features.astype(np.float32)
    if scope == "state":
        return ds.x.astype(np.float32)
    if scope == "both":
        return np.concatenate([ds.x, ds.selector_features], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature scope: {scope}")


def build_pool_rows(
    datasets: list[SelectorData],
    pool_k: int,
    scope: str,
    positive_weight: float,
    close_weight: float,
    close_ev: float,
    miss_weighting: MissWeighting | None,
    include_pool_local: bool,
) -> PoolRows:
    xs: list[np.ndarray] = []
    y_gap: list[np.ndarray] = []
    y_best: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    group_sizes: list[int] = []
    for ds in datasets:
        base_x = feature_matrix(ds, scope)
        for group_idx, (start, end) in enumerate(ds.bounds):
            members = union_members(ds, start, end, pool_k)
            if not members:
                continue
            idx = np.asarray(members, dtype=np.int64)
            x = base_x[idx]
            if include_pool_local:
                x = np.concatenate([x, pool_local_features(ds, members)], axis=1).astype(np.float32)
            scores = np.asarray(ds.scores[idx], dtype=np.float32)
            best = float(np.max(ds.scores[start:end]))
            gap = scores - best
            label = (np.abs(gap) <= 1e-6).astype(np.float32)
            w = np.ones(len(idx), dtype=np.float32)
            w += label * float(positive_weight)
            w += (np.maximum(0.0, float(close_ev) - np.abs(gap)) / max(float(close_ev), 1e-6)).astype(
                np.float32
            ) * float(close_weight)
            if miss_weighting:
                key = (ds.spec.name, int(group_idx))
                group_multiplier = miss_weighting.group_weights.get(key, 1.0)
                teacher_multiplier = miss_weighting.teacher_weights.get(key, 1.0)
                w *= float(group_multiplier)
                w += label * float(teacher_multiplier)
            xs.append(x)
            y_gap.append(gap.astype(np.float32))
            y_best.append(label)
            weights.append(w)
            group_sizes.append(len(idx))
    if not xs:
        raise ValueError("no pool rows were built")
    return PoolRows(
        x=np.concatenate(xs, axis=0),
        y_gap=np.concatenate(y_gap, axis=0),
        y_best=np.concatenate(y_best, axis=0),
        weights=np.concatenate(weights, axis=0),
        group_sizes=group_sizes,
    )


def predict_dataset(
    ds: SelectorData,
    model,
    kind: str,
    pool_k: int,
    scope: str,
    include_pool_local: bool,
) -> np.ndarray:
    x = feature_matrix(ds, scope)
    pred = np.full(len(ds.scores), -1.0e9, dtype=np.float32)
    for start, end in ds.bounds:
        members = union_members(ds, start, end, pool_k)
        if not members:
            continue
        idx = np.asarray(members, dtype=np.int64)
        local_x = x[idx]
        if include_pool_local:
            local_x = np.concatenate([local_x, pool_local_features(ds, members)], axis=1).astype(np.float32)
        if is_classifier(kind):
            proba = model.predict_proba(local_x)
            values = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            values = model.predict(local_x)
        pred[idx] = np.asarray(values, dtype=np.float32)
    return pred


def evaluate_pred(datasets: list[SelectorData], pred_by_dataset: dict[str, np.ndarray], topks: list[int]) -> tuple[dict, dict]:
    all_scores: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_bounds: list[tuple[int, int]] = []
    per_dataset: dict[str, dict] = {}
    offset = 0
    for ds in datasets:
        pred = pred_by_dataset[ds.spec.name]
        metrics, _rows = summarize_groups(ds.scores, pred, ds.bounds, topks)
        per_dataset[ds.spec.name] = compact_metrics(metrics)
        all_scores.append(ds.scores)
        all_pred.append(pred)
        for start, end in ds.bounds:
            all_bounds.append((start + offset, end + offset))
        offset += len(ds.scores)
    aggregate, _rows = summarize_groups(np.concatenate(all_scores), np.concatenate(all_pred), all_bounds, topks)
    return compact_metrics(aggregate), per_dataset


def union_ceiling(datasets: list[SelectorData], pool_k: int) -> dict[str, Any]:
    losses: list[float] = []
    sizes: list[int] = []
    hits = 0
    groups = 0
    for ds in datasets:
        for start, end in ds.bounds:
            members = union_members(ds, start, end, pool_k)
            best_idx = start + int(np.argmax(ds.scores[start:end]))
            best = float(ds.scores[best_idx])
            pool_best = max(float(ds.scores[idx]) for idx in members)
            hits += int(best_idx in members)
            groups += 1
            losses.append(max(0.0, best - pool_best))
            sizes.append(len(members))
    arr = np.asarray(losses, dtype=np.float64)
    return {
        "groups": int(groups),
        "recall": float(hits / max(groups, 1)),
        "avg_pool": float(np.mean(sizes)) if sizes else 0.0,
        "max_pool": int(max(sizes)) if sizes else 0,
        "mean_ev_loss": float(arr.mean()) if len(arr) else 0.0,
        "max_ev_loss": float(arr.max()) if len(arr) else 0.0,
    }


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    m = row["aggregate"]
    return (
        float(m.get("group_top1", 0.0)),
        -float(m.get("group_top1_regret", 0.0)),
        float(m.get("group_top3", 0.0)),
    )


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
    feature_selectors = [parse_feature_selector(value) for value in args.feature_selector]
    miss_weighting = load_miss_weighting(args)

    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    loaded_feature = {spec.name: joblib.load(spec.path) for spec in feature_selectors}

    train_sets = [
        load_dataset(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            args.npy_selector,
            loaded_models,
            loaded_lgbm,
            feature_selectors,
            loaded_feature,
            include_base=not args.no_base,
        )
        for spec in train_specs
    ]
    eval_sets = [
        load_dataset(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            args.npy_selector,
            loaded_models,
            loaded_lgbm,
            feature_selectors,
            loaded_feature,
            include_base=not args.no_base,
        )
        for spec in eval_specs
    ]

    rows = build_pool_rows(
        train_sets,
        pool_k=args.pool_k,
        scope=args.feature_scope,
        positive_weight=args.positive_weight,
        close_weight=args.close_weight,
        close_ev=args.close_ev,
        miss_weighting=miss_weighting,
        include_pool_local=args.pool_local_features,
    )

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for i, kind in enumerate(args.model_kind):
        model = make_model(kind, args.seed + i)
        target = rows.y_best if is_classifier(kind) else rows.y_gap
        if is_ranker(kind):
            target = rank_relevance(rows.y_gap, rows.group_sizes)
        t0 = time.time()
        if is_ranker(kind):
            model.fit(rows.x, target, group=rows.group_sizes, sample_weight=rows.weights)
        else:
            model.fit(rows.x, target, sample_weight=rows.weights)
        model_path = out_dir / f"{kind}.joblib"
        joblib.dump(model, model_path)
        train_pred = {
            ds.spec.name: predict_dataset(ds, model, kind, args.pool_k, args.feature_scope, args.pool_local_features)
            for ds in train_sets
        }
        eval_pred = {
            ds.spec.name: predict_dataset(ds, model, kind, args.pool_k, args.feature_scope, args.pool_local_features)
            for ds in eval_sets
        }
        train_agg, train_per = evaluate_pred(train_sets, train_pred, topks)
        eval_agg, eval_per = evaluate_pred(eval_sets, eval_pred, topks)
        row = {"model": kind, "path": str(model_path), "fit_seconds": time.time() - t0}
        train_rows.append({**row, "aggregate": train_agg, "datasets": train_per})
        eval_rows.append({**row, "aggregate": eval_agg, "datasets": eval_per})

    train_rows.sort(key=sort_key, reverse=True)
    eval_rows.sort(key=sort_key, reverse=True)
    summary = {
        "pool_k": int(args.pool_k),
        "feature_scope": args.feature_scope,
        "pool_local_features": bool(args.pool_local_features),
        "train_data": {spec.name: str(spec.path) for spec in train_specs},
        "eval_data": {spec.name: str(spec.path) for spec in eval_specs},
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": list(train_sets[0].selector_scores),
        "feature_selectors": {spec.name: {"path": str(spec.path), "scope": spec.scope} for spec in feature_selectors},
        "lgbm_selectors": {name: str(path) for name, path in lgbm_selectors},
        "model_kinds": args.model_kind,
        "topks": topks,
        "train_groups": int(sum(len(ds.bounds) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.bounds) for ds in eval_sets)),
        "train_pool_rows": int(len(rows.y_best)),
        "positive_rate": float(np.mean(rows.y_best)),
        "train_union_ceiling": union_ceiling(train_sets, args.pool_k),
        "eval_union_ceiling": union_ceiling(eval_sets, args.pool_k),
        "miss_weighting": (
            {
                "rows_seen": miss_weighting.rows_seen,
                "rows_used": miss_weighting.rows_used,
                "weighted_groups": len(miss_weighting.group_weights),
                "datasets": list(miss_weighting.datasets),
                "target_ks": list(miss_weighting.target_ks),
                "miss_rows": args.miss_rows,
                "miss_min_ev_loss": args.miss_min_ev_loss,
            }
            if miss_weighting
            else None
        ),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
        "note": "Teacher EV is used only as target/weight. Runtime pool uses selector and feature-selector scores.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", action="append", required=True, help="name=path")
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument(
        "--npy-selector",
        action="append",
        default=[],
        help="Selector score stored as <data_dir>/selector_scores/<name>.npy",
    )
    parser.add_argument("--feature-selector", action="append", default=[], help="name=path,selector|state")
    parser.add_argument("--model-kind", action="append", default=[])
    parser.add_argument("--pool-k", type=int, default=3)
    parser.add_argument("--feature-scope", choices=("both", "selector", "state"), default="both")
    parser.add_argument("--pool-local-features", action="store_true")
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--close-weight", type=float, default=2.0)
    parser.add_argument("--close-ev", type=float, default=1.0)
    parser.add_argument("--miss-rows", action="append", default=[])
    parser.add_argument("--miss-target-ks", default="")
    parser.add_argument("--miss-min-ev-loss", type=float, default=0.0)
    parser.add_argument("--miss-group-weight", type=float, default=8.0)
    parser.add_argument("--miss-teacher-weight", type=float, default=12.0)
    parser.add_argument("--miss-ev-loss-cap", type=float, default=2.0)
    parser.add_argument("--miss-ev-loss-group-scale", type=float, default=2.0)
    parser.add_argument("--miss-ev-loss-teacher-scale", type=float, default=3.0)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.model_kind:
        args.model_kind = ["hgb_cls_l31", "hgb_gap_l31", "extra_cls_d14"]
    run(args)


if __name__ == "__main__":
    main()
