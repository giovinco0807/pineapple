"""Train a pairwise T2 reranker inside a runtime union shortlist.

The selector-feature rankers score candidates independently.  This diagnostic
model learns "candidate A beats candidate B" inside a small union pool, then
scores a candidate by its average pairwise win probability against the other
pool members.  Teacher EV is used only as the supervised training label.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.evaluate_t2_selector_feature_union import (
    FeatureSelectorSpec,
    feature_model_input,
    parse_feature_selector,
)
from ai.training.train_t2_selector_feature_ranker import (
    build_selector_features,
    compact_metrics,
    load_selector_scores,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
    predict_adapted,
)


@dataclass
class PairRows:
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    pairs: int
    groups: int
    skipped_small_gap: int


def union_members(ds, start: int, end: int, pool_k: int) -> list[int]:
    members: set[int] = set()
    for scores in ds.selector_scores.values():
        local = np.asarray(scores[start:end], dtype=np.float64)
        order = np.argsort(-local)
        for local_idx in order[: min(pool_k, len(order))]:
            members.add(start + int(local_idx))
    return sorted(members)


def enrich_with_feature_selectors(
    ds,
    feature_selectors: list[FeatureSelectorSpec],
    loaded_feature: dict[str, object],
):
    if not feature_selectors:
        return ds
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


def base_feature_matrix(ds, scope: str) -> np.ndarray:
    if scope == "selector":
        return ds.selector_features.astype(np.float32)
    if scope == "state":
        return ds.x.astype(np.float32)
    if scope == "both":
        return np.concatenate([ds.x, ds.selector_features], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature scope: {scope}")


def pool_local_features(ds, members: list[int]) -> np.ndarray:
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
    for col in range(len(names)):
        values = selector_matrix[:, col]
        order = np.argsort(-values)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        rank_frac[:, col] = ranks / max(n - 1, 1)
        gap_to_top[:, col] = float(values[order[0]]) - values
        for row in range(n):
            others = np.delete(values, row)
            margin_to_next[row, col] = values[row] - float(np.max(others)) if len(others) else 0.0
        top1[order[:1], col] = 1.0
        top3[order[: min(3, n)], col] = 1.0
    score_mean = selector_matrix.mean(axis=1, keepdims=True)
    score_std = selector_matrix.std(axis=1, keepdims=True)
    rank_mean = rank_frac.mean(axis=1, keepdims=True)
    rank_best = rank_frac.min(axis=1, keepdims=True)
    rank_worst = rank_frac.max(axis=1, keepdims=True)
    votes = np.concatenate([top1.sum(axis=1, keepdims=True), top3.sum(axis=1, keepdims=True)], axis=1)
    group_size = np.full((n, 1), n / 30.0, dtype=np.float64)
    return np.concatenate(
        [
            selector_matrix,
            rank_frac,
            gap_to_top,
            margin_to_next,
            top1,
            top3,
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


def candidate_features(ds, scope: str) -> np.ndarray:
    base = base_feature_matrix(ds, scope)
    # Pool-local features depend on the selected pool and are appended later.
    return base


def pair_feature(left: np.ndarray, right: np.ndarray, mode: str) -> np.ndarray:
    diff = left - right
    absdiff = np.abs(diff)
    if mode == "diff":
        return np.concatenate([diff, absdiff]).astype(np.float32)
    if mode == "full":
        return np.concatenate([left, right, diff, absdiff]).astype(np.float32)
    raise ValueError(f"unknown pair feature mode: {mode}")


def build_pair_rows(
    datasets,
    *,
    pool_k: int,
    feature_scope: str,
    pair_mode: str,
    min_gap: float,
    gap_cap: float,
    gap_weight: float,
    best_pair_weight: float,
) -> PairRows:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    weights: list[float] = []
    groups = 0
    skipped_small_gap = 0
    for ds in datasets:
        base = candidate_features(ds, feature_scope)
        for start, end in ds.bounds:
            members = union_members(ds, start, end, pool_k)
            if len(members) < 2:
                continue
            groups += 1
            local_extra = pool_local_features(ds, members)
            feature_by_index = {
                member: np.concatenate([base[member], local_extra[pos]], axis=0).astype(np.float32)
                for pos, member in enumerate(members)
            }
            scores = {member: float(ds.scores[member]) for member in members}
            best_member = start + int(np.argmax(ds.scores[start:end]))
            for i, left_idx in enumerate(members):
                for right_idx in members[i + 1 :]:
                    gap = scores[left_idx] - scores[right_idx]
                    abs_gap = abs(gap)
                    if abs_gap < min_gap:
                        skipped_small_gap += 1
                        continue
                    if gap >= 0:
                        winner, loser = left_idx, right_idx
                    else:
                        winner, loser = right_idx, left_idx
                    weight = 1.0 + min(abs_gap, gap_cap) * gap_weight
                    if winner == best_member:
                        weight += best_pair_weight
                    xs.append(pair_feature(feature_by_index[winner], feature_by_index[loser], pair_mode))
                    ys.append(1.0)
                    weights.append(weight)
                    xs.append(pair_feature(feature_by_index[loser], feature_by_index[winner], pair_mode))
                    ys.append(0.0)
                    weights.append(weight)
    if not xs:
        raise ValueError("no pair rows were built")
    return PairRows(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        pairs=len(xs),
        groups=groups,
        skipped_small_gap=skipped_small_gap,
    )


def make_model(kind: str, seed: int):
    if kind == "hgb_l31":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=520,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=24,
            random_state=seed,
        )
    if kind == "hgb_l63":
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=650,
            max_leaf_nodes=63,
            l2_regularization=0.04,
            min_samples_leaf=18,
            random_state=seed,
        )
    if kind == "extra_d12":
        return ExtraTreesClassifier(
            n_estimators=320,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "rf_d14":
        return RandomForestClassifier(
            n_estimators=260,
            max_depth=14,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def win_probability(model, x: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(x)
    if prob.shape[1] == 1:
        return prob[:, 0].astype(np.float32)
    return prob[:, 1].astype(np.float32)


def prediction_for_dataset(ds, model, *, pool_k: int, feature_scope: str, pair_mode: str) -> np.ndarray:
    base = candidate_features(ds, feature_scope)
    pred = np.full(len(ds.scores), -1.0e9, dtype=np.float32)
    for start, end in ds.bounds:
        members = union_members(ds, start, end, pool_k)
        if len(members) == 1:
            pred[members[0]] = 1.0
            continue
        if len(members) < 2:
            continue
        local_extra = pool_local_features(ds, members)
        feature_by_index = {
            member: np.concatenate([base[member], local_extra[pos]], axis=0).astype(np.float32)
            for pos, member in enumerate(members)
        }
        pair_x: list[np.ndarray] = []
        pair_keys: list[tuple[int, int]] = []
        for left in members:
            for right in members:
                if left == right:
                    continue
                pair_x.append(pair_feature(feature_by_index[left], feature_by_index[right], pair_mode))
                pair_keys.append((left, right))
        probs = win_probability(model, np.stack(pair_x).astype(np.float32))
        scores = {member: 0.0 for member in members}
        counts = {member: 0 for member in members}
        for (left, _right), prob in zip(pair_keys, probs, strict=True):
            scores[left] += float(prob)
            counts[left] += 1
        for member in members:
            pred[member] = scores[member] / max(counts[member], 1)
    return pred


def evaluate_predictions(datasets, pred_by_dataset: dict[str, np.ndarray], topks: list[int]) -> tuple[dict, dict]:
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
    metrics, _rows = summarize_groups(
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_pred, axis=0),
        all_bounds,
        topks,
    )
    return compact_metrics(metrics), per_dataset


def union_ceiling(datasets, pool_k: int) -> dict:
    groups = 0
    hits = 0
    losses: list[float] = []
    sizes: list[int] = []
    for ds in datasets:
        for start, end in ds.bounds:
            members = union_members(ds, start, end, pool_k)
            if not members:
                continue
            best_idx = start + int(np.argmax(ds.scores[start:end]))
            best_ev = float(ds.scores[best_idx])
            pool_best = max(float(ds.scores[idx]) for idx in members)
            groups += 1
            hits += int(best_idx in members)
            losses.append(best_ev - pool_best)
            sizes.append(len(members))
    return {
        "groups": groups,
        "recall": hits / max(groups, 1),
        "mean_ev_loss": float(np.mean(losses)) if losses else 0.0,
        "max_ev_loss": float(np.max(losses)) if losses else 0.0,
        "avg_pool": float(np.mean(sizes)) if sizes else 0.0,
        "max_pool": int(np.max(sizes)) if sizes else 0,
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
        "# T2 Pairwise Pool Reranker",
        "",
        f"- pool_k: `{summary['pool_k']}`",
        f"- feature_scope: `{summary['feature_scope']}`",
        f"- pair_mode: `{summary['pair_mode']}`",
        f"- train groups: `{summary['train_groups']}`",
        f"- eval groups: `{summary['eval_groups']}`",
        "",
        "| model | Top1 | Reg1 | Top3 | Top5 | Top10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["eval_rows"]:
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
    feature_selectors = [parse_feature_selector(value) for value in args.feature_selector]
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    loaded_feature = {spec.name: joblib.load(spec.path) for spec in feature_selectors}

    train_sets = [
        enrich_with_feature_selectors(
            load_selector_scores(
                spec,
                model_specs,
                selectors,
                lgbm_selectors,
                loaded_models,
                loaded_lgbm,
                include_base=not args.no_base,
            ),
            feature_selectors,
            loaded_feature,
        )
        for spec in train_specs
    ]
    eval_sets = [
        enrich_with_feature_selectors(
            load_selector_scores(
                spec,
                model_specs,
                selectors,
                lgbm_selectors,
                loaded_models,
                loaded_lgbm,
                include_base=not args.no_base,
            ),
            feature_selectors,
            loaded_feature,
        )
        for spec in eval_specs
    ]
    pair_rows = build_pair_rows(
        train_sets,
        pool_k=args.pool_k,
        feature_scope=args.feature_scope,
        pair_mode=args.pair_mode,
        min_gap=args.min_gap,
        gap_cap=args.gap_cap,
        gap_weight=args.gap_weight,
        best_pair_weight=args.best_pair_weight,
    )

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for idx, kind in enumerate(args.model_kind):
        model = make_model(kind, args.seed + idx)
        model.fit(pair_rows.x, pair_rows.y, sample_weight=pair_rows.weights)
        model_path = out_dir / f"{kind}.joblib"
        joblib.dump(model, model_path)
        train_pred = {
            ds.spec.name: prediction_for_dataset(
                ds,
                model,
                pool_k=args.pool_k,
                feature_scope=args.feature_scope,
                pair_mode=args.pair_mode,
            )
            for ds in train_sets
        }
        eval_pred = {
            ds.spec.name: prediction_for_dataset(
                ds,
                model,
                pool_k=args.pool_k,
                feature_scope=args.feature_scope,
                pair_mode=args.pair_mode,
            )
            for ds in eval_sets
        }
        train_agg, train_per = evaluate_predictions(train_sets, train_pred, topks)
        eval_agg, eval_per = evaluate_predictions(eval_sets, eval_pred, topks)
        train_rows.append({"model": kind, "aggregate": train_agg, "datasets": train_per, "path": str(model_path)})
        eval_rows.append({"model": kind, "aggregate": eval_agg, "datasets": eval_per, "path": str(model_path)})
    train_rows.sort(key=sort_key, reverse=True)
    eval_rows.sort(key=sort_key, reverse=True)

    summary = {
        "pool_k": int(args.pool_k),
        "feature_scope": args.feature_scope,
        "pair_mode": args.pair_mode,
        "min_gap": float(args.min_gap),
        "gap_cap": float(args.gap_cap),
        "gap_weight": float(args.gap_weight),
        "best_pair_weight": float(args.best_pair_weight),
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
        "pair_rows": {
            "rows": int(pair_rows.pairs),
            "groups": int(pair_rows.groups),
            "positive_rate": float(pair_rows.y.mean()),
            "weight_mean": float(pair_rows.weights.mean()),
            "weight_max": float(pair_rows.weights.max()),
            "skipped_small_gap": int(pair_rows.skipped_small_gap),
        },
        "train_union_ceiling": union_ceiling(train_sets, args.pool_k),
        "eval_union_ceiling": union_ceiling(eval_sets, args.pool_k),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
        "note": "Teacher EV is used only to label pair outcomes. Non-pool actions receive -inf prediction.",
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
    parser.add_argument("--feature-selector", action="append", default=[], help="name=path,selector|state")
    parser.add_argument("--model-kind", action="append", default=[])
    parser.add_argument("--pool-k", type=int, default=3)
    parser.add_argument("--feature-scope", choices=("selector", "state", "both"), default="selector")
    parser.add_argument("--pair-mode", choices=("diff", "full"), default="full")
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--gap-cap", type=float, default=4.0)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--best-pair-weight", type=float, default=2.0)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.model_kind:
        args.model_kind = ["hgb_l31", "hgb_l63", "extra_d12"]
    if not args.selector and not args.lgbm_selector and args.no_base:
        raise ValueError("At least one selector is required")
    run(args)


if __name__ == "__main__":
    main()
