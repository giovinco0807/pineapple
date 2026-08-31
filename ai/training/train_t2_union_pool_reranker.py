"""Train a T2 reranker only inside a runtime union shortlist.

The broader selector-feature rankers score every legal action.  This script is
focused on the product path where several fast selectors propose a TopK union
pool and a final lightweight model chooses one action from that pool.  Inputs
remain runtime-only selector scores/ranks/gaps plus the existing state/action
features; teacher EV is used only as the supervised label.
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
    RandomForestClassifier,
    RandomForestRegressor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_selector_feature_ranker import (
    DataSpec,
    ModelSpec,
    SelectorData,
    SelectorSpec,
    compact_metrics,
    load_selector_scores,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
)


@dataclass
class PoolRows:
    x: np.ndarray
    y_score: np.ndarray
    y_gap: np.ndarray
    y_best: np.ndarray
    weights: np.ndarray
    group_sizes: list[int]


def make_model(kind: str, seed: int):
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
            max_iter=600,
            max_leaf_nodes=63,
            l2_regularization=0.04,
            min_samples_leaf=14,
            random_state=seed,
        )
    if kind == "extra_gap_d12":
        return ExtraTreesRegressor(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "rf_gap_d14":
        return RandomForestRegressor(
            n_estimators=220,
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
    if kind == "extra_cls_d12":
        return ExtraTreesClassifier(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "rf_cls_d14":
        return RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def union_members(ds: SelectorData, start: int, end: int, pool_k: int) -> list[int]:
    members: set[int] = set()
    for scores in ds.selector_scores.values():
        local = np.asarray(scores[start:end], dtype=np.float64)
        order = np.argsort(-local)
        for local_idx in order[: min(pool_k, len(order))]:
            members.add(start + int(local_idx))
    return sorted(members)


def build_feature_matrix(ds: SelectorData, feature_scope: str) -> np.ndarray:
    if feature_scope == "state":
        return ds.x.astype(np.float32)
    if feature_scope == "selector":
        return ds.selector_features.astype(np.float32)
    if feature_scope == "both":
        return np.concatenate([ds.x, ds.selector_features], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature scope: {feature_scope}")


def build_pool_rows(
    datasets: list[SelectorData],
    pool_k: int,
    feature_scope: str,
    positive_weight: float,
    close_weight: float,
    close_ev: float,
) -> PoolRows:
    xs: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    y_gap: list[np.ndarray] = []
    y_best: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    group_sizes: list[int] = []
    for ds in datasets:
        base_x = build_feature_matrix(ds, feature_scope)
        for start, end in ds.bounds:
            members = union_members(ds, start, end, pool_k)
            if not members:
                continue
            idx = np.asarray(members, dtype=np.int64)
            scores = np.asarray(ds.scores[idx], dtype=np.float32)
            best = float(np.max(ds.scores[start:end]))
            gap = scores - best
            label = (np.abs(gap) <= 1e-6).astype(np.float32)
            w = np.ones(len(idx), dtype=np.float32)
            w += label * float(positive_weight)
            w += (np.maximum(0.0, float(close_ev) - np.abs(gap)) / max(float(close_ev), 1e-6)).astype(
                np.float32
            ) * float(close_weight)
            xs.append(base_x[idx])
            y_score.append(scores)
            y_gap.append(gap.astype(np.float32))
            y_best.append(label)
            weights.append(w)
            group_sizes.append(len(idx))
    if not xs:
        raise ValueError("no pool rows were built")
    return PoolRows(
        x=np.concatenate(xs, axis=0),
        y_score=np.concatenate(y_score, axis=0),
        y_gap=np.concatenate(y_gap, axis=0),
        y_best=np.concatenate(y_best, axis=0),
        weights=np.concatenate(weights, axis=0),
        group_sizes=group_sizes,
    )


def prediction_for_dataset(ds: SelectorData, model, kind: str, pool_k: int, feature_scope: str) -> np.ndarray:
    base_x = build_feature_matrix(ds, feature_scope)
    pred = np.full(len(ds.scores), -1.0e9, dtype=np.float32)
    for start, end in ds.bounds:
        members = union_members(ds, start, end, pool_k)
        if not members:
            continue
        idx = np.asarray(members, dtype=np.int64)
        if kind.endswith("_cls_l31") or kind.startswith("extra_cls") or kind.startswith("rf_cls"):
            prob = model.predict_proba(base_x[idx])
            values = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
        else:
            values = model.predict(base_x[idx])
        pred[idx] = np.asarray(values, dtype=np.float32)
    return pred


def evaluate_pred(datasets: list[SelectorData], pred_by_dataset: dict[str, np.ndarray], topks: list[int]) -> tuple[dict, dict]:
    all_scores: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_bounds: list[tuple[int, int]] = []
    offset = 0
    per_dataset: dict[str, dict] = {}
    for ds in datasets:
        pred = pred_by_dataset[ds.spec.name]
        metrics, _rows = summarize_groups(ds.scores, pred, ds.bounds, topks)
        per_dataset[ds.spec.name] = compact_metrics(metrics)
        all_scores.append(ds.scores)
        all_pred.append(pred)
        for start, end in ds.bounds:
            all_bounds.append((start + offset, end + offset))
        offset += len(ds.scores)
    aggregate_metrics, _rows = summarize_groups(
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_pred, axis=0),
        all_bounds,
        topks,
    )
    return compact_metrics(aggregate_metrics), per_dataset


def baseline_union_metrics(datasets: list[SelectorData], pool_k: int) -> dict:
    groups = 0
    hits = 0
    losses: list[float] = []
    pool_sizes: list[int] = []
    for ds in datasets:
        for start, end in ds.bounds:
            members = union_members(ds, start, end, pool_k)
            best_idx = start + int(np.argmax(ds.scores[start:end]))
            best_score = float(ds.scores[best_idx])
            pool_best = max(float(ds.scores[i]) for i in members)
            hits += 1 if best_idx in members else 0
            groups += 1
            losses.append(best_score - pool_best)
            pool_sizes.append(len(members))
    loss_arr = np.asarray(losses, dtype=np.float64)
    size_arr = np.asarray(pool_sizes, dtype=np.float64)
    return {
        "groups": int(groups),
        "recall": hits / max(groups, 1),
        "mean_ev_loss": float(loss_arr.mean()) if len(loss_arr) else 0.0,
        "max_ev_loss": float(loss_arr.max()) if len(loss_arr) else 0.0,
        "avg_pool": float(size_arr.mean()) if len(size_arr) else 0.0,
        "max_pool": int(size_arr.max()) if len(size_arr) else 0,
    }


def sort_key(row: dict) -> tuple[float, float, float]:
    m = row["aggregate"]
    return (
        float(m.get("group_top1", 0.0)),
        -float(m.get("group_top1_regret", 0.0)),
        float(m.get("group_top3", 0.0)),
    )


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 Union-Pool Reranker",
        "",
        f"- pool_k: `{summary['pool_k']}`",
        f"- feature_scope: `{summary['feature_scope']}`",
        f"- train groups: `{summary['train_groups']}`",
        f"- eval groups: `{summary['eval_groups']}`",
        "",
        "## Eval",
        "",
        "| model | Top1 | Reg1 | Top3 | Top5 | Top10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["eval_rows"][:20]:
        m = row["aggregate"]
        lines.append(
            f"| {row['model']} | "
            f"{float(m.get('group_top1', 0.0)):.1%} | "
            f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top3', 0.0)):.1%} | "
            f"{float(m.get('group_top5', 0.0)):.1%} | "
            f"{float(m.get('group_top10', 0.0)):.1%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    train_sets = [
        load_selector_scores(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            include_base=not args.no_base,
        )
        for spec in train_specs
    ]
    eval_sets = [
        load_selector_scores(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            include_base=not args.no_base,
        )
        for spec in eval_specs
    ]
    rows = build_pool_rows(
        train_sets,
        pool_k=args.pool_k,
        feature_scope=args.feature_scope,
        positive_weight=args.positive_weight,
        close_weight=args.close_weight,
        close_ev=args.close_ev,
    )

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for model_idx, kind in enumerate(args.model_kind):
        model = make_model(kind, args.seed + model_idx)
        target = rows.y_best if kind.startswith(("hgb_cls", "extra_cls", "rf_cls")) else rows.y_gap
        model.fit(rows.x, target, sample_weight=rows.weights)
        joblib.dump(model, out_dir / f"{kind}.joblib")

        train_pred_by_dataset = {
            ds.spec.name: prediction_for_dataset(ds, model, kind, args.pool_k, args.feature_scope)
            for ds in train_sets
        }
        eval_pred_by_dataset = {
            ds.spec.name: prediction_for_dataset(ds, model, kind, args.pool_k, args.feature_scope)
            for ds in eval_sets
        }
        train_agg, train_per = evaluate_pred(train_sets, train_pred_by_dataset, topks)
        eval_agg, eval_per = evaluate_pred(eval_sets, eval_pred_by_dataset, topks)
        train_rows.append({"model": kind, "aggregate": train_agg, "datasets": train_per})
        eval_rows.append({"model": kind, "aggregate": eval_agg, "datasets": eval_per})

    train_rows.sort(key=sort_key, reverse=True)
    eval_rows.sort(key=sort_key, reverse=True)
    summary = {
        "pool_k": int(args.pool_k),
        "feature_scope": args.feature_scope,
        "train_data": {spec.name: str(spec.path) for spec in train_specs},
        "eval_data": {spec.name: str(spec.path) for spec in eval_specs},
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": list(train_sets[0].selector_scores),
        "lgbm_selectors": {name: str(path) for name, path in lgbm_selectors},
        "model_kinds": args.model_kind,
        "topks": topks,
        "train_groups": int(sum(len(ds.bounds) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.bounds) for ds in eval_sets)),
        "train_pool_rows": int(len(rows.y_gap)),
        "positive_rate": float(rows.y_best.mean()),
        "positive_weight": float(args.positive_weight),
        "close_weight": float(args.close_weight),
        "close_ev": float(args.close_ev),
        "train_union_ceiling": baseline_union_metrics(train_sets, args.pool_k),
        "eval_union_ceiling": baseline_union_metrics(eval_sets, args.pool_k),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
        "note": "Teacher EV is used only for the target. Non-pool actions receive -inf prediction at evaluation time.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", action="append", required=True, help="name=path")
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument("--model-kind", action="append", default=[])
    parser.add_argument("--pool-k", type=int, default=5)
    parser.add_argument("--feature-scope", choices=("both", "selector", "state"), default="both")
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--close-weight", type=float, default=2.0)
    parser.add_argument("--close-ev", type=float, default=1.0)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.model_kind:
        args.model_kind = ["hgb_gap_l31", "hgb_gap_l63", "extra_gap_d12", "hgb_cls_l31", "extra_cls_d12"]
    if not args.selector and not args.lgbm_selector and args.no_base:
        raise ValueError("At least one selector is required")
    run(args)


if __name__ == "__main__":
    main()
