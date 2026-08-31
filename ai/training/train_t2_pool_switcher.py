"""Train a T2 pool switcher over model TopK candidates.

This is a diagnostic for improving model-only Top1 without widening the final
answer too far.  A set of source models first proposes a small pool, e.g. the
union of each source model's Top2 candidates.  The switcher then chooses one
candidate from that pool using state, selector, and source-model agreement
features.

The switcher target is the best exact-EV candidate inside the proposed pool.
Evaluation reports both the pool upper bound and the switcher's realized Top1
hit/regret against the full exact labels.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_t2_position_aware_union import (
    classifier_or_regressor_scores,
    load_stack_specs,
    parse_named_path_value,
    repeated_flag_values,
)
from ai.training.train_t2_selector_feature_ranker import (
    DataSpec,
    SelectorData,
    load_selector_scores,
    model_features,
)


@dataclass(frozen=True)
class PoolSpec:
    model_name: str
    k: int


@dataclass
class PoolDataset:
    name: str
    path: str
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    exact_scores: np.ndarray
    bounds: list[tuple[int, int]]
    full_best_scores: np.ndarray
    full_best_locals: np.ndarray
    pool_locals: list[list[int]]


def build_classifier(kind: str, seed: int) -> Any:
    if kind == "hgb_l31":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=420,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=14,
            random_state=seed,
        )
    if kind == "hgb_l63":
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=560,
            max_leaf_nodes=63,
            l2_regularization=0.05,
            min_samples_leaf=12,
            random_state=seed,
        )
    if kind == "extra_trees_d14":
        return ExtraTreesClassifier(
            n_estimators=420,
            max_depth=14,
            min_samples_leaf=1,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "rf_d14":
        return RandomForestClassifier(
            n_estimators=360,
            max_depth=14,
            min_samples_leaf=1,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown switcher classifier kind: {kind}")


def parse_pool_spec(value: str) -> PoolSpec:
    name, sep, raw_k = value.partition(":")
    if not sep or not name.strip() or not raw_k.strip():
        raise ValueError("--pool-source must be model:k")
    return PoolSpec(model_name=name.strip(), k=int(raw_k))


def load_command_data(path: Path, flag: str) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        command = json.load(f)
    return repeated_flag_values(command, flag)


def top_local_indices(values: np.ndarray, start: int, end: int, k: int) -> list[int]:
    kk = min(max(int(k), 0), end - start)
    if kk <= 0:
        return []
    order = np.argsort(-np.asarray(values[start:end], dtype=np.float64))[:kk]
    return [int(i) for i in order]


def group_normalized_features(
    predictions: dict[str, np.ndarray],
    bounds: list[tuple[int, int]],
    model_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in model_names:
        values = np.asarray(predictions[name], dtype=np.float64)
        z = np.zeros(len(values), dtype=np.float32)
        rank_frac = np.zeros(len(values), dtype=np.float32)
        gap = np.zeros(len(values), dtype=np.float32)
        is_top1 = np.zeros(len(values), dtype=np.float32)
        is_top2 = np.zeros(len(values), dtype=np.float32)
        is_top3 = np.zeros(len(values), dtype=np.float32)
        for start, end in bounds:
            local = values[start:end]
            if len(local) == 0:
                continue
            sd = float(local.std())
            if sd >= 1.0e-9:
                z[start:end] = ((local - float(local.mean())) / sd).astype(np.float32)
            order = np.argsort(-local)
            ranks = np.empty(len(local), dtype=np.float64)
            ranks[order] = np.arange(len(local), dtype=np.float64)
            rank_frac[start:end] = (1.0 - ranks / max(len(local) - 1, 1)).astype(np.float32)
            gap[start:end] = (float(local[order[0]]) - local).astype(np.float32)
            is_top1[start + order[:1]] = 1.0
            is_top2[start + order[: min(2, len(order))]] = 1.0
            is_top3[start + order[: min(3, len(order))]] = 1.0
        out[name] = {
            "raw": values.astype(np.float32),
            "z": z,
            "rank_frac": rank_frac,
            "gap_to_top": gap,
            "is_top1": is_top1,
            "is_top2": is_top2,
            "is_top3": is_top3,
        }
    return out


def build_pool_dataset(
    data: SelectorData,
    predictions: dict[str, np.ndarray],
    pool_specs: list[PoolSpec],
    *,
    source_model_names: list[str],
    label_mode: str,
) -> PoolDataset:
    base_features = model_features([data], include_state=True)
    normalized = group_normalized_features(predictions, data.bounds, source_model_names)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    exact_scores: list[float] = []
    bounds: list[tuple[int, int]] = []
    full_best_scores: list[float] = []
    full_best_locals: list[int] = []
    pool_locals_by_group: list[list[int]] = []

    for group_id, (start, end) in enumerate(data.bounds):
        true = np.asarray(data.scores[start:end], dtype=np.float64)
        if len(true) == 0:
            continue
        pool: set[int] = set()
        for spec in pool_specs:
            pool.update(top_local_indices(predictions[spec.model_name], start, end, spec.k))
        if not pool:
            continue
        pool_locals = sorted(pool)
        pool_global = [start + local for local in pool_locals]
        pool_scores = true[pool_locals]
        full_best_local = int(np.argmax(true))
        pool_best_local_index = int(np.argmax(pool_scores))
        pool_best_global = pool_global[pool_best_local_index]
        row_start = len(rows)
        for global_idx, local_idx in zip(pool_global, pool_locals):
            source_chunks: list[float] = []
            for name in source_model_names:
                feats = normalized[name]
                source_chunks.extend(
                    [
                        float(feats["raw"][global_idx]),
                        float(feats["z"][global_idx]),
                        float(feats["rank_frac"][global_idx]),
                        float(feats["gap_to_top"][global_idx]),
                        float(feats["is_top1"][global_idx]),
                        float(feats["is_top2"][global_idx]),
                        float(feats["is_top3"][global_idx]),
                    ]
                )
            top1_votes = sum(float(normalized[name]["is_top1"][global_idx]) for name in source_model_names)
            top2_votes = sum(float(normalized[name]["is_top2"][global_idx]) for name in source_model_names)
            top3_votes = sum(float(normalized[name]["is_top3"][global_idx]) for name in source_model_names)
            pool_rank = pool_locals.index(local_idx) / max(len(pool_locals) - 1, 1)
            extra = np.asarray(
                [
                    float(len(pool_locals)) / 10.0,
                    float(local_idx) / max(end - start - 1, 1),
                    float(pool_rank),
                    top1_votes,
                    top2_votes,
                    top3_votes,
                ],
                dtype=np.float32,
            )
            rows.append(
                np.concatenate(
                    [
                        np.asarray(base_features[global_idx], dtype=np.float32),
                        np.asarray(source_chunks, dtype=np.float32),
                        extra,
                    ]
                )
            )
            if label_mode == "full_best":
                label = int(local_idx == full_best_local)
            else:
                label = int(global_idx == pool_best_global)
            labels.append(label)
            exact_scores.append(float(true[local_idx]))
        row_end = len(rows)
        group_size = max(row_end - row_start, 1)
        group_weight = 1.0 / max(group_size - 1, 1)
        for idx in range(row_start, row_end):
            weights.append(group_weight)
        # Always strongly weight the best candidate available in the pool.
        weights[row_start + pool_best_local_index] = 8.0
        bounds.append((row_start, row_end))
        full_best_scores.append(float(true[full_best_local]))
        full_best_locals.append(full_best_local)
        pool_locals_by_group.append(pool_locals)

    if rows:
        feature_matrix = np.vstack(rows).astype(np.float32)
    else:
        feature_matrix = np.zeros((0, 0), dtype=np.float32)
    return PoolDataset(
        name=data.spec.name,
        path=str(data.spec.path),
        features=feature_matrix,
        labels=np.asarray(labels, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float32),
        exact_scores=np.asarray(exact_scores, dtype=np.float32),
        bounds=bounds,
        full_best_scores=np.asarray(full_best_scores, dtype=np.float32),
        full_best_locals=np.asarray(full_best_locals, dtype=np.int32),
        pool_locals=pool_locals_by_group,
    )


def evaluate_pool(dataset: PoolDataset, pred: np.ndarray, topks: list[int]) -> dict[str, Any]:
    hits = 0
    regret_sum = 0.0
    max_regret = 0.0
    pool_hits = 0
    pool_regret_sum = 0.0
    pool_max_regret = 0.0
    topk_hits = {k: 0 for k in topks}
    topk_regret = {k: 0.0 for k in topks}
    miss_rows: list[dict[str, Any]] = []

    for group_id, (start, end) in enumerate(dataset.bounds):
        local_scores = np.asarray(dataset.exact_scores[start:end], dtype=np.float64)
        local_pred = np.asarray(pred[start:end], dtype=np.float64)
        full_best_score = float(dataset.full_best_scores[group_id])
        pool_best_score = float(local_scores.max())
        pool_loss = max(0.0, full_best_score - pool_best_score)
        pool_hits += int(pool_loss <= 1.0e-9)
        pool_regret_sum += pool_loss
        pool_max_regret = max(pool_max_regret, pool_loss)

        pick = int(np.argmax(local_pred))
        chosen_score = float(local_scores[pick])
        loss = max(0.0, full_best_score - chosen_score)
        hits += int(loss <= 1.0e-9)
        regret_sum += loss
        max_regret = max(max_regret, loss)
        if loss > 1.0e-9:
            miss_rows.append(
                {
                    "group_id": int(group_id),
                    "full_best_local": int(dataset.full_best_locals[group_id]),
                    "chosen_pool_index": int(pick),
                    "chosen_local": int(dataset.pool_locals[group_id][pick]),
                    "ev_loss": float(loss),
                    "pool_locals": [int(v) for v in dataset.pool_locals[group_id]],
                }
            )

        order = np.argsort(-local_pred)
        for k in topks:
            kk = min(k, len(order))
            chosen = order[:kk]
            best_in_topk = float(local_scores[chosen].max()) if kk else float("-inf")
            topk_loss = max(0.0, full_best_score - best_in_topk)
            topk_hits[k] += int(topk_loss <= 1.0e-9)
            topk_regret[k] += topk_loss

    groups = max(len(dataset.bounds), 1)
    metrics: dict[str, Any] = {
        "groups": int(len(dataset.bounds)),
        "samples": int(len(dataset.labels)),
        "pool_upper_hit": pool_hits / groups,
        "pool_upper_regret": pool_regret_sum / groups,
        "pool_upper_max_regret": pool_max_regret,
        "top1": hits / groups,
        "reg1": regret_sum / groups,
        "max_reg1": max_regret,
        "miss_count": len(miss_rows),
        "miss_rows": miss_rows[:20],
    }
    for k in topks:
        metrics[f"top{k}"] = topk_hits[k] / groups
        metrics[f"reg{k}"] = topk_regret[k] / groups
    return metrics


def concat_pool_datasets(datasets: list[PoolDataset]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([ds.features for ds in datasets], axis=0),
        np.concatenate([ds.labels for ds in datasets], axis=0),
        np.concatenate([ds.weights for ds in datasets], axis=0),
    )


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    training_run = Path(args.training_run)
    model_specs, selector_specs, lgbm_selectors, npy_selectors = load_stack_specs(training_run)
    loaded_models = {spec.name: joblib.load(spec.path) for spec in model_specs}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    final_models = {name: joblib.load(path) for name, path in map(parse_named_path_value, args.final_model)}
    source_model_names = list(final_models)

    train_data_values = list(args.train_data)
    eval_data_values = list(args.eval_data)
    if args.train_command_json:
        command_path = Path(args.train_command_json)
        train_data_values.extend(load_command_data(command_path, "--train-data"))
    train_specs = [DataSpec(name, path) for name, path in map(parse_named_path_value, train_data_values)]
    eval_specs = [DataSpec(name, path) for name, path in map(parse_named_path_value, eval_data_values)]
    pool_specs = [parse_pool_spec(value) for value in args.pool_source]
    if not pool_specs:
        pool_specs = [PoolSpec(name, args.default_pool_k) for name in source_model_names]
    topks = parse_topks(args.topks)

    def load_pool(spec: DataSpec) -> PoolDataset:
        data = load_selector_scores(
            spec,
            {model_spec.name: model_spec for model_spec in model_specs},
            selector_specs,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            include_base=not args.no_base,
            npy_selectors=npy_selectors,
        )
        x = model_features([data], include_state=True)
        predictions = {name: classifier_or_regressor_scores(model, x) for name, model in final_models.items()}
        return build_pool_dataset(
            data,
            predictions,
            pool_specs,
            source_model_names=source_model_names,
            label_mode=args.label_mode,
        )

    train_sets = [load_pool(spec) for spec in train_specs]
    eval_sets = [load_pool(spec) for spec in eval_specs]
    x_train, y_train, w_train = concat_pool_datasets(train_sets)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for offset, kind in enumerate(args.classifier):
        model = build_classifier(kind, args.seed + offset)
        model.fit(x_train, y_train, sample_weight=w_train)
        model_path = out_dir / f"pool_switcher_{kind}.joblib"
        joblib.dump(model, model_path)
        model_rows.append({"name": kind, "path": str(model_path)})
        for ds in train_sets:
            pred = classifier_or_regressor_scores(model, ds.features)
            eval_rows.append({"split": "train", "dataset": ds.name, "model": kind, "metrics": evaluate_pool(ds, pred, topks)})
        for ds in eval_sets:
            pred = classifier_or_regressor_scores(model, ds.features)
            eval_rows.append({"split": "eval", "dataset": ds.name, "model": kind, "metrics": evaluate_pool(ds, pred, topks)})

    summary = {
        "training_run": str(training_run),
        "train_data": [{"name": spec.name, "path": str(spec.path)} for spec in train_specs],
        "eval_data": [{"name": spec.name, "path": str(spec.path)} for spec in eval_specs],
        "final_models": {name: str(path) for name, path in map(parse_named_path_value, args.final_model)},
        "pool_specs": [{"model": spec.model_name, "k": spec.k} for spec in pool_specs],
        "label_mode": args.label_mode,
        "classifiers": model_rows,
        "train_groups": int(sum(len(ds.bounds) for ds in train_sets)),
        "train_samples": int(sum(len(ds.labels) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.bounds) for ds in eval_sets)),
        "eval_samples": int(sum(len(ds.labels) for ds in eval_sets)),
        "rows": eval_rows,
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--train-command-json", default="")
    parser.add_argument("--train-data", action="append", default=[], help="name=path")
    parser.add_argument("--eval-data", action="append", default=[], help="name=path")
    parser.add_argument("--final-model", action="append", required=True, help="name=path")
    parser.add_argument("--pool-source", action="append", default=[], help="model:k")
    parser.add_argument("--default-pool-k", type=int, default=2)
    parser.add_argument("--classifier", action="append", default=["hgb_l31", "hgb_l63", "extra_trees_d14", "rf_d14"])
    parser.add_argument("--label-mode", choices=("pool_best", "full_best"), default="pool_best")
    parser.add_argument("--topks", default="1,2,3")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
