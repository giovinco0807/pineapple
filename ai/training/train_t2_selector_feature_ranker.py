"""Train a T2 ranker with runtime selector-agreement features.

The existing no-leak meta rankers use the action state and one base score.  This
script adds only runtime-available information from several independent
selectors: each selector's score, within-group rank, gap to its top candidate,
z-score, and cross-selector agreement features.  Teacher EV is used only as the
supervised target and evaluation label.
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
from lightgbm import LGBMRanker
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
from ai.training.train_t2_sklearn_meta_ranker import compact_metrics, load_runtime_features


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    target: str


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    model_names: tuple[str, ...]
    gamma: float


@dataclass(frozen=True)
class DataSpec:
    name: str
    path: Path


@dataclass
class SelectorData:
    spec: DataSpec
    x: np.ndarray
    scores: np.ndarray
    base_scores: np.ndarray
    bounds: list[tuple[int, int]]
    selector_scores: dict[str, np.ndarray]
    selector_features: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class MissWeighting:
    group_weights: dict[tuple[str, int], float]
    teacher_weights: dict[tuple[str, int], float]
    rows_used: int
    rows_seen: int
    datasets: tuple[str, ...]
    target_ks: tuple[int, ...]


RUNTIME_SUFFIX_FEATURES = 5
INACTIVE_SELECTOR_SCORE = -1.0e7


def adapt_runtime_features(x: np.ndarray, expected: int) -> np.ndarray:
    """Adapt runtime feature matrices after extra state-only features are appended.

    ``load_runtime_features`` builds ``[state, base_score, rank/gap/z/group]``.
    Newer state encodings append board/deck context to ``state``.  Older
    sklearn models still need the old state prefix plus the runtime suffix, not
    a blind prefix slice that would replace base/rank features with new state
    columns.
    """
    expected = int(expected)
    if x.shape[1] == expected:
        return x
    if x.shape[1] > expected:
        extra = x.shape[1] - expected
        if extra > 0 and expected > RUNTIME_SUFFIX_FEATURES and x.shape[1] > RUNTIME_SUFFIX_FEATURES:
            prefix_width = expected - RUNTIME_SUFFIX_FEATURES
            return np.concatenate([x[:, :prefix_width], x[:, -RUNTIME_SUFFIX_FEATURES:]], axis=1)
        return x[:, :expected]
    raise ValueError(f"model expects {expected} features but input has {x.shape[1]}")


def adapt_model_input(model: object, x: np.ndarray) -> np.ndarray:
    """Match newer feature matrices to older sklearn model input widths.

    Some diagnostics append extra runtime-only features after the original
    feature block.  Older sklearn models can still be valid selector sources if
    they read only the prefix they were trained on.
    """
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        return x
    expected = int(n_features)
    return adapt_runtime_features(x, expected)


def predict_adapted(model: object, x: np.ndarray) -> np.ndarray:
    return model.predict(adapt_model_input(model, x)).astype(np.float32)


def parse_named_path(value: str) -> DataSpec:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("expected name=path")
    return DataSpec(name=parts[0].strip(), path=Path(parts[1].strip()))


def parse_model(value: str) -> ModelSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--model must be name=path,target")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--model must be name=path,target")
    target = items[1].lower()
    if target not in {"score", "residual"}:
        raise ValueError("model target must be score or residual")
    return ModelSpec(name=name, path=Path(items[0]), target=target)


def parse_selector(value: str) -> SelectorSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--selector must be name=model_a+model_b,gamma")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--selector must be name=model_a+model_b,gamma")
    model_names = tuple(part.strip() for part in items[0].split("+") if part.strip())
    if not name or not model_names:
        raise ValueError("--selector needs a name and at least one model")
    return SelectorSpec(name=name, model_names=model_names, gamma=float(items[1]))


def parse_lgbm_selector(value: str) -> tuple[str, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("--lgbm-selector must be name=path")
    return parts[0].strip(), Path(parts[1].strip())


def load_selector_scores(
    data_spec: DataSpec,
    model_specs: dict[str, ModelSpec],
    selectors: list[SelectorSpec],
    lgbm_selectors: list[tuple[str, Path]],
    loaded_models: dict[str, object],
    loaded_lgbm: dict[str, object],
    include_base: bool,
    npy_selectors: list[str] | None = None,
) -> SelectorData:
    x, scores, base_scores, bounds = load_runtime_features(data_spec.path)
    residuals: dict[str, np.ndarray] = {}
    for name, spec in model_specs.items():
        pred = predict_adapted(loaded_models[name], x)
        residuals[name] = pred if spec.target == "residual" else pred - base_scores

    selector_scores: dict[str, np.ndarray] = {}
    if include_base:
        selector_scores["base_scores"] = base_scores.astype(np.float32)
    for selector in selectors:
        missing = [name for name in selector.model_names if name not in residuals]
        if missing:
            raise ValueError(f"selector {selector.name} references unknown model(s): {missing}")
        avg_residual = np.mean([residuals[name] for name in selector.model_names], axis=0).astype(np.float32)
        selector_scores[selector.name] = (base_scores + selector.gamma * avg_residual).astype(np.float32)
    for name, _path in lgbm_selectors:
        selector_scores[name] = predict_adapted(loaded_lgbm[name], x)
    for name in npy_selectors or []:
        score_path = data_spec.path / "selector_scores" / f"{name}.npy"
        if not score_path.exists():
            raise FileNotFoundError(f"missing npy selector {name!r} for {data_spec.name}: {score_path}")
        values = np.load(score_path)
        if len(values) != len(scores):
            raise ValueError(
                f"npy selector {name!r} length mismatch for {data_spec.name}: "
                f"{len(values)} != {len(scores)}"
            )
        selector_scores[name] = np.asarray(values, dtype=np.float32)

    features, feature_names = build_selector_features(selector_scores, bounds)
    return SelectorData(
        spec=data_spec,
        x=x,
        scores=scores,
        base_scores=base_scores,
        bounds=bounds,
        selector_scores=selector_scores,
        selector_features=features,
        feature_names=feature_names,
    )


def build_selector_features(
    selector_scores: dict[str, np.ndarray],
    bounds: list[tuple[int, int]],
) -> tuple[np.ndarray, list[str]]:
    names = list(selector_scores)
    n = len(next(iter(selector_scores.values()))) if selector_scores else 0
    per_selector_feature_names: list[str] = []
    for name in names:
        per_selector_feature_names.extend(
            [
                f"{name}_score",
                f"{name}_rank_frac",
                f"{name}_gap_to_top",
                f"{name}_z",
                f"{name}_is_top1",
                f"{name}_is_top3",
                f"{name}_is_top5",
            ]
        )
    feature_names = [
        *per_selector_feature_names,
        "selector_score_mean",
        "selector_score_std",
        "selector_score_max",
        "selector_score_min",
        "selector_score_range",
        "selector_top1_votes",
        "selector_top3_votes",
        "selector_top5_votes",
        "selector_mean_rank_frac",
        "selector_best_rank_frac",
        "selector_worst_rank_frac",
        "group_size_frac",
    ]
    features = np.zeros((n, len(feature_names)), dtype=np.float32)
    for start, end in bounds:
        group_size = max(end - start, 1)
        selector_matrix = np.stack(
            [np.asarray(selector_scores[name][start:end], dtype=np.float64) for name in names],
            axis=1,
        )
        ranks = np.zeros_like(selector_matrix, dtype=np.float64)
        rank_fracs = np.zeros_like(selector_matrix, dtype=np.float64)
        gaps = np.zeros_like(selector_matrix, dtype=np.float64)
        zscores = np.zeros_like(selector_matrix, dtype=np.float64)
        top1 = np.zeros_like(selector_matrix, dtype=np.float64)
        top3 = np.zeros_like(selector_matrix, dtype=np.float64)
        top5 = np.zeros_like(selector_matrix, dtype=np.float64)
        for col in range(len(names)):
            values = selector_matrix[:, col]
            if float(np.max(values)) <= INACTIVE_SELECTOR_SCORE:
                rank_fracs[:, col] = 1.0
                continue
            order = [int(i) for i in np.argsort(-values) if float(values[int(i)]) > INACTIVE_SELECTOR_SCORE]
            local_ranks = np.full(group_size, group_size, dtype=np.float64)
            local_ranks[order] = np.arange(len(order), dtype=np.float64)
            ranks[:, col] = local_ranks
            rank_fracs[:, col] = local_ranks / max(group_size - 1, 1)
            gaps[:, col] = float(values[order[0]]) - values if order else 0.0
            std = float(values.std())
            if std < 1e-6:
                std = 1.0
            zscores[:, col] = (values - float(values.mean())) / std
            top1[order[:1], col] = 1.0
            top3[order[: min(3, len(order))], col] = 1.0
            top5[order[: min(5, len(order))], col] = 1.0

        offset = 0
        group_features: list[np.ndarray] = []
        for col in range(len(names)):
            group_features.extend(
                [
                    selector_matrix[:, col],
                    rank_fracs[:, col],
                    gaps[:, col],
                    zscores[:, col],
                    top1[:, col],
                    top3[:, col],
                    top5[:, col],
                ]
            )
        group_features.extend(
            [
                selector_matrix.mean(axis=1),
                selector_matrix.std(axis=1),
                selector_matrix.max(axis=1),
                selector_matrix.min(axis=1),
                selector_matrix.max(axis=1) - selector_matrix.min(axis=1),
                top1.sum(axis=1),
                top3.sum(axis=1),
                top5.sum(axis=1),
                rank_fracs.mean(axis=1),
                rank_fracs.min(axis=1),
                rank_fracs.max(axis=1),
                np.full(group_size, group_size / 30.0, dtype=np.float64),
            ]
        )
        for values in group_features:
            features[start:end, offset] = np.asarray(values, dtype=np.float32)
            offset += 1
    return features, feature_names


def build_model(kind: str, seed: int):
    if kind == "hgb_l31":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=520,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=20,
            random_state=seed,
        )
    if kind == "hgb_l63":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.03,
            max_iter=620,
            max_leaf_nodes=63,
            l2_regularization=0.05,
            min_samples_leaf=18,
            random_state=seed,
        )
    if kind == "extra_trees_d12":
        return ExtraTreesRegressor(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.6,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "rf_d12":
        return RandomForestRegressor(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.65,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def build_classifier(kind: str, seed: int):
    if kind == "hgb_cls_l31":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=420,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=18,
            random_state=seed,
        )
    if kind == "hgb_cls_l63":
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=520,
            max_leaf_nodes=63,
            l2_regularization=0.05,
            min_samples_leaf=16,
            random_state=seed,
        )
    if kind == "extra_trees_cls_d12":
        return ExtraTreesClassifier(
            n_estimators=320,
            max_depth=12,
            min_samples_leaf=2,
            max_features=0.65,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "rf_cls_d12":
        return RandomForestClassifier(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=2,
            max_features=0.65,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown classifier kind: {kind}")


def group_sizes(bounds: list[tuple[int, int]]) -> list[int]:
    return [int(end - start) for start, end in bounds]


def build_relevance(
    scores: np.ndarray,
    bounds: list[tuple[int, int]],
    mode: str,
    max_label: int,
    gap_scale: float,
) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=np.int32)
    for start, end in bounds:
        local = np.asarray(scores[start:end], dtype=np.float64)
        if len(local) == 0:
            continue
        order = np.argsort(-local)
        if mode == "rank":
            ranks = np.empty(len(local), dtype=np.int32)
            ranks[order] = np.arange(len(local), dtype=np.int32)
            labels[start:end] = np.maximum(0, int(max_label) - ranks)
            continue
        best = float(local[order[0]])
        gaps = np.maximum(0.0, best - local)
        raw = np.rint(float(max_label) - gaps * float(gap_scale)).astype(np.int32)
        labels[start:end] = np.clip(raw, 0, int(max_label))
    return labels


def build_lgbm_ranker(args: argparse.Namespace, seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=args.lgbm_n_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        reg_alpha=args.lgbm_reg_alpha,
        reg_lambda=args.lgbm_reg_lambda,
        label_gain=list(range(args.lgbm_max_label + 1)),
        random_state=seed,
        n_jobs=args.lgbm_n_jobs,
        verbose=-1,
    )


def concat_arrays(datasets: list[SelectorData], attr: str) -> np.ndarray:
    return np.concatenate([getattr(ds, attr) for ds in datasets], axis=0)


def concat_bounds(datasets: list[SelectorData]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    offset = 0
    for ds in datasets:
        bounds.extend((start + offset, end + offset) for start, end in ds.bounds)
        offset += len(ds.scores)
    return bounds


def concat_group_keys(datasets: list[SelectorData]) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for ds in datasets:
        keys.extend((ds.spec.name, group_id) for group_id in range(len(ds.bounds)))
    return keys


def load_miss_weighting(args: argparse.Namespace) -> MissWeighting | None:
    if not args.miss_rows:
        return None
    target_ks = tuple(parse_topks(args.miss_target_ks)) if args.miss_target_ks else ()
    target_k_set = set(target_ks)
    group_weights: dict[tuple[str, int], float] = {}
    teacher_weights: dict[tuple[str, int], float] = {}
    datasets: set[str] = set()
    rows_seen = 0
    rows_used = 0
    for path_value in args.miss_rows:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows_seen += 1
                row = json.loads(line)
                target_k = int(row.get("target_k", -1))
                if target_k_set and target_k not in target_k_set:
                    continue
                ev_loss = float(row.get("ev_loss", 0.0))
                if ev_loss < float(args.miss_min_ev_loss):
                    continue
                dataset = str(row["dataset"])
                group_id = int(row["group_id"])
                key = (dataset, group_id)
                datasets.add(dataset)
                rows_used += 1
                group_multiplier = float(args.miss_group_weight) + min(ev_loss, float(args.miss_ev_loss_cap)) * float(
                    args.miss_ev_loss_group_scale
                )
                teacher_multiplier = float(args.miss_teacher_weight) + min(
                    ev_loss, float(args.miss_ev_loss_cap)
                ) * float(args.miss_ev_loss_teacher_scale)
                group_weights[key] = max(group_weights.get(key, 1.0), group_multiplier)
                teacher_weights[key] = max(teacher_weights.get(key, 1.0), teacher_multiplier)
    return MissWeighting(
        group_weights=group_weights,
        teacher_weights=teacher_weights,
        rows_used=rows_used,
        rows_seen=rows_seen,
        datasets=tuple(sorted(datasets)),
        target_ks=target_ks,
    )


def best_labels_and_weights(
    datasets: list[SelectorData],
    miss_weighting: MissWeighting | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = concat_arrays(datasets, "scores")
    bounds = concat_bounds(datasets)
    group_keys = concat_group_keys(datasets)
    labels = np.zeros(len(scores), dtype=np.int32)
    weights = np.zeros(len(scores), dtype=np.float32)
    for group_idx, (start, end) in enumerate(bounds):
        group = np.asarray(scores[start:end], dtype=np.float64)
        if len(group) == 0:
            continue
        best_local = int(np.argmax(group))
        key = group_keys[group_idx]
        group_multiplier = miss_weighting.group_weights.get(key, 1.0) if miss_weighting else 1.0
        teacher_multiplier = miss_weighting.teacher_weights.get(key, 1.0) if miss_weighting else 1.0
        labels[start + best_local] = 1
        group_size = max(end - start, 1)
        weights[start:end] = group_multiplier / max(group_size - 1, 1)
        weights[start + best_local] = group_multiplier * teacher_multiplier
    return labels, weights


def regression_target_and_weights(
    datasets: list[SelectorData],
    mode: str,
    miss_weighting: MissWeighting | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build a regression target for candidate ordering.

    ``score`` preserves the original EV target.  Other modes are group-relative
    and are meant to put more learning pressure on choosing the best candidate
    inside each T2 spot.
    """
    weighted = mode.endswith("_top_weighted")
    base_mode = mode.removesuffix("_top_weighted")
    if base_mode not in {"score", "centered", "gap_to_best", "rank_frac", "ev_rank_blend"}:
        raise ValueError(f"unknown target mode: {mode}")
    scores = concat_arrays(datasets, "scores")
    bounds = concat_bounds(datasets)
    group_keys = concat_group_keys(datasets)
    target = np.zeros(len(scores), dtype=np.float32)
    weights = np.ones(len(scores), dtype=np.float32)
    for group_idx, (start, end) in enumerate(bounds):
        true = np.asarray(scores[start:end], dtype=np.float64)
        group_size = len(true)
        if group_size == 0:
            continue
        order = np.argsort(-true)
        ranks = np.empty(group_size, dtype=np.float64)
        ranks[order] = np.arange(group_size, dtype=np.float64)
        rank_score = 1.0 - ranks / max(group_size - 1, 1)
        if base_mode == "score":
            values = true
        elif base_mode == "centered":
            values = true - float(true.mean())
        elif base_mode == "gap_to_best":
            values = true - float(true.max())
        elif base_mode == "rank_frac":
            values = rank_score
        else:
            std = float(true.std())
            if std < 1e-6:
                std = 1.0
            values = (true - float(true.mean())) / std + 0.35 * rank_score
        target[start:end] = values.astype(np.float32)
        if weighted:
            group_weights = 0.25 + np.square(rank_score)
            group_weights[order[0]] += 2.0
            group_weights[order[: min(3, group_size)]] += 0.75
        else:
            group_weights = np.ones(group_size, dtype=np.float64)
        if miss_weighting:
            key = group_keys[group_idx]
            group_multiplier = miss_weighting.group_weights.get(key, 1.0)
            teacher_multiplier = miss_weighting.teacher_weights.get(key, 1.0)
            group_weights *= group_multiplier
            group_weights[order[0]] *= teacher_multiplier
        weights[start:end] = group_weights.astype(np.float32)
    return target, weights if weighted or miss_weighting else None


def model_features(datasets: list[SelectorData], include_state: bool) -> np.ndarray:
    selector_features = concat_arrays(datasets, "selector_features")
    if not include_state:
        return selector_features
    states = np.concatenate([ds.x for ds in datasets], axis=0)
    return np.concatenate([states, selector_features], axis=1).astype(np.float32)


def selector_prediction(datasets: list[SelectorData], name: str) -> np.ndarray:
    return np.concatenate([ds.selector_scores[name] for ds in datasets], axis=0)


def evaluate_prediction(
    datasets: list[SelectorData],
    pred: np.ndarray,
    topks: list[int],
) -> dict:
    scores = concat_arrays(datasets, "scores")
    bounds = concat_bounds(datasets)
    metrics, _rows = summarize_groups(scores, pred.astype(np.float32), bounds, topks)
    return compact_metrics(metrics)


def evaluate_per_dataset(datasets: list[SelectorData], pred_by_dataset: dict[str, np.ndarray], topks: list[int]) -> dict:
    out = {}
    for ds in datasets:
        metrics, _rows = summarize_groups(ds.scores, pred_by_dataset[ds.spec.name].astype(np.float32), ds.bounds, topks)
        out[ds.spec.name] = compact_metrics(metrics)
    return out


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 Selector-Feature Ranker",
        "",
        "- inputs: runtime state features plus selector scores/ranks/gaps/agreement",
        "- excluded inputs: teacher EV, teacher rank, route tags, bust/FL labels",
        f"- train groups: `{summary['train_groups']}`",
        f"- eval groups: `{summary['eval_groups']}`",
        "",
        "## Eval",
        "",
        "| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["eval_rows"][:20]:
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
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    train_specs = [parse_named_path(value) for value in args.train_data]
    eval_specs = [parse_named_path(value) for value in args.eval_data]
    miss_weighting = load_miss_weighting(args)
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
            npy_selectors=args.npy_selector,
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
            npy_selectors=args.npy_selector,
        )
        for spec in eval_specs
    ]

    eval_rows: list[dict] = []
    train_rows: list[dict] = []
    predictor_pool: list[dict] = []
    for name in train_sets[0].selector_scores:
        train_pred = selector_prediction(train_sets, name)
        eval_pred = selector_prediction(eval_sets, name)
        train_rows.append({"selector": name, "aggregate": evaluate_prediction(train_sets, train_pred, topks)})
        eval_rows.append({"selector": name, "aggregate": evaluate_prediction(eval_sets, eval_pred, topks)})
        predictor_pool.append({"name": name, "train_pred": train_pred, "eval_pred": eval_pred, "source": "selector"})

    feature_scopes = [False, True]
    if args.feature_scope == "selector":
        feature_scopes = [False]
    elif args.feature_scope == "state":
        feature_scopes = [True]
    for include_state in feature_scopes:
        x_train = model_features(train_sets, include_state=include_state)
        x_eval = model_features(eval_sets, include_state=include_state)
        for target_mode in args.target_mode:
            train_target, train_weight = regression_target_and_weights(
                train_sets,
                target_mode,
                miss_weighting=miss_weighting,
            )
            for offset, kind in enumerate(args.ranker):
                model = build_model(kind, args.seed + offset + (100 if include_state else 0))
                t0 = time.time()
                if train_weight is None:
                    model.fit(x_train, train_target)
                else:
                    model.fit(x_train, train_target, sample_weight=train_weight)
                suffix = "state" if include_state else "selector"
                model_name = f"{kind}_{suffix}_{target_mode}"
                model_path = out_dir / f"{model_name}.joblib"
                joblib.dump(model, model_path)
                train_pred = model.predict(x_train).astype(np.float32)
                eval_pred = model.predict(x_eval).astype(np.float32)
                train_rows.append(
                    {
                        "selector": model_name,
                        "aggregate": evaluate_prediction(train_sets, train_pred, topks),
                        "fit_seconds": time.time() - t0,
                        "model": str(model_path),
                        "target": target_mode,
                    }
                )
                eval_pred_by_dataset = {}
                offset_idx = 0
                for ds in eval_sets:
                    eval_pred_by_dataset[ds.spec.name] = eval_pred[offset_idx : offset_idx + len(ds.scores)]
                    offset_idx += len(ds.scores)
                eval_rows.append(
                    {
                        "selector": model_name,
                        "aggregate": evaluate_prediction(eval_sets, eval_pred, topks),
                        "datasets": evaluate_per_dataset(eval_sets, eval_pred_by_dataset, topks),
                        "model": str(model_path),
                        "target": target_mode,
                    }
                )
                predictor_pool.append(
                    {
                        "name": model_name,
                        "train_pred": train_pred,
                        "eval_pred": eval_pred,
                        "source": "regressor",
                    }
                )
        train_scores = concat_arrays(train_sets, "scores")
        train_bounds = concat_bounds(train_sets)
        for offset, relevance_mode in enumerate(args.lgbm_relevance_mode):
            relevance = build_relevance(
                train_scores,
                train_bounds,
                mode=relevance_mode,
                max_label=args.lgbm_max_label,
                gap_scale=args.lgbm_gap_scale,
            )
            ranker = build_lgbm_ranker(args, args.seed + offset + (500 if include_state else 400))
            t0 = time.time()
            ranker.fit(x_train, relevance, group=group_sizes(train_bounds))
            suffix = "state" if include_state else "selector"
            model_name = f"lgbm_rank_{relevance_mode}_{suffix}"
            model_path = out_dir / f"{model_name}.joblib"
            joblib.dump(ranker, model_path)
            train_pred = ranker.predict(x_train).astype(np.float32)
            eval_pred = ranker.predict(x_eval).astype(np.float32)
            train_rows.append(
                {
                    "selector": model_name,
                    "aggregate": evaluate_prediction(train_sets, train_pred, topks),
                    "fit_seconds": time.time() - t0,
                    "model": str(model_path),
                    "target": f"lgbm_{relevance_mode}",
                }
            )
            eval_pred_by_dataset = {}
            offset_idx = 0
            for ds in eval_sets:
                eval_pred_by_dataset[ds.spec.name] = eval_pred[offset_idx : offset_idx + len(ds.scores)]
                offset_idx += len(ds.scores)
            eval_rows.append(
                {
                    "selector": model_name,
                    "aggregate": evaluate_prediction(eval_sets, eval_pred, topks),
                    "datasets": evaluate_per_dataset(eval_sets, eval_pred_by_dataset, topks),
                    "model": str(model_path),
                    "target": f"lgbm_{relevance_mode}",
                }
            )
            predictor_pool.append(
                {
                    "name": model_name,
                    "train_pred": train_pred,
                    "eval_pred": eval_pred,
                    "source": "lgbm_ranker",
                }
            )
        if not args.skip_classifiers:
            y_best, cls_weight = best_labels_and_weights(train_sets, miss_weighting=miss_weighting)
            for offset, kind in enumerate(args.classifier):
                clf = build_classifier(kind, args.seed + offset + (300 if include_state else 200))
                t0 = time.time()
                clf.fit(x_train, y_best, sample_weight=cls_weight)
                model_name = f"{kind}_{'state' if include_state else 'selector'}"
                model_path = out_dir / f"{model_name}.joblib"
                joblib.dump(clf, model_path)
                train_pred = clf.predict_proba(x_train)[:, 1].astype(np.float32)
                eval_pred = clf.predict_proba(x_eval)[:, 1].astype(np.float32)
                train_rows.append(
                    {
                        "selector": model_name,
                        "aggregate": evaluate_prediction(train_sets, train_pred, topks),
                        "fit_seconds": time.time() - t0,
                        "model": str(model_path),
                        "target": "teacher_best_probability",
                    }
                )
                eval_pred_by_dataset = {}
                offset_idx = 0
                for ds in eval_sets:
                    eval_pred_by_dataset[ds.spec.name] = eval_pred[offset_idx : offset_idx + len(ds.scores)]
                    offset_idx += len(ds.scores)
                eval_rows.append(
                    {
                        "selector": model_name,
                        "aggregate": evaluate_prediction(eval_sets, eval_pred, topks),
                        "datasets": evaluate_per_dataset(eval_sets, eval_pred_by_dataset, topks),
                        "model": str(model_path),
                        "target": "teacher_best_probability",
                    }
                )
                predictor_pool.append(
                    {
                        "name": model_name,
                        "train_pred": train_pred,
                        "eval_pred": eval_pred,
                        "source": "classifier",
                    }
                )

    def sort_key(row: dict) -> tuple[float, float, float, float]:
        m = row["aggregate"]
        return (
            float(m.get("group_top1", 0.0)),
            -float(m.get("group_top1_regret", 0.0)),
            float(m.get("group_top3", 0.0)),
            -float(m.get("group_top10_rerank_regret", 0.0)),
        )

    if args.blend_trained and len(predictor_pool) >= 2:
        blend_candidates: list[dict] = []
        alphas = np.linspace(0.0, 1.0, args.blend_steps + 1)
        for i, left in enumerate(predictor_pool):
            for right in predictor_pool[i + 1 :]:
                for alpha in alphas:
                    train_pred = float(alpha) * left["train_pred"] + (1.0 - float(alpha)) * right["train_pred"]
                    aggregate = evaluate_prediction(train_sets, train_pred.astype(np.float32), topks)
                    row = {
                        "selector": f"blend_train:{float(alpha):.2f}:{left['name']}+{1.0-float(alpha):.2f}:{right['name']}",
                        "aggregate": aggregate,
                        "alpha": float(alpha),
                        "left": left["name"],
                        "right": right["name"],
                    }
                    blend_candidates.append(row)
        blend_candidates.sort(key=sort_key, reverse=True)
        for row in blend_candidates[: args.max_blends]:
            left = next(item for item in predictor_pool if item["name"] == row["left"])
            right = next(item for item in predictor_pool if item["name"] == row["right"])
            alpha = float(row["alpha"])
            train_pred = alpha * left["train_pred"] + (1.0 - alpha) * right["train_pred"]
            eval_pred = alpha * left["eval_pred"] + (1.0 - alpha) * right["eval_pred"]
            train_rows.append({**row, "aggregate": evaluate_prediction(train_sets, train_pred.astype(np.float32), topks)})
            eval_pred_by_dataset = {}
            offset_idx = 0
            for ds in eval_sets:
                eval_pred_by_dataset[ds.spec.name] = eval_pred[offset_idx : offset_idx + len(ds.scores)]
                offset_idx += len(ds.scores)
            eval_rows.append(
                {
                    **row,
                    "aggregate": evaluate_prediction(eval_sets, eval_pred.astype(np.float32), topks),
                    "datasets": evaluate_per_dataset(eval_sets, eval_pred_by_dataset, topks),
                    "selected_on": "train",
                }
            )

    train_rows.sort(key=sort_key, reverse=True)
    eval_rows.sort(key=sort_key, reverse=True)
    summary = {
        "train_data": {spec.name: str(spec.path) for spec in train_specs},
        "eval_data": {spec.name: str(spec.path) for spec in eval_specs},
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": list(train_sets[0].selector_scores),
        "lgbm_selectors": {name: str(path) for name, path in lgbm_selectors},
        "rankers": args.ranker,
        "classifiers": args.classifier,
        "target_modes": args.target_mode,
        "lgbm_relevance_modes": args.lgbm_relevance_mode,
        "feature_scope": args.feature_scope,
        "skip_classifiers": bool(args.skip_classifiers),
        "blend_trained": bool(args.blend_trained),
        "miss_weighting": (
            {
                "rows_seen": miss_weighting.rows_seen,
                "rows_used": miss_weighting.rows_used,
                "weighted_groups": len(miss_weighting.group_weights),
                "datasets": list(miss_weighting.datasets),
                "target_ks": list(miss_weighting.target_ks),
                "miss_rows": args.miss_rows,
                "miss_min_ev_loss": args.miss_min_ev_loss,
                "miss_group_weight": args.miss_group_weight,
                "miss_teacher_weight": args.miss_teacher_weight,
                "miss_ev_loss_cap": args.miss_ev_loss_cap,
                "miss_ev_loss_group_scale": args.miss_ev_loss_group_scale,
                "miss_ev_loss_teacher_scale": args.miss_ev_loss_teacher_scale,
            }
            if miss_weighting
            else None
        ),
        "topks": topks,
        "include_base": not args.no_base,
        "selector_feature_names": train_sets[0].feature_names,
        "train_groups": int(sum(len(ds.bounds) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.bounds) for ds in eval_sets)),
        "train_samples": int(sum(len(ds.scores) for ds in train_sets)),
        "eval_samples": int(sum(len(ds.scores) for ds in eval_sets)),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
        "note": "Teacher EV is the target only. Selector ranks/scores are runtime predictions, not labels.",
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
    parser.add_argument(
        "--npy-selector",
        action="append",
        default=[],
        help="Selector score stored as <data_dir>/selector_scores/<name>.npy",
    )
    parser.add_argument("--ranker", action="append", default=[])
    parser.add_argument("--classifier", action="append", default=[])
    parser.add_argument("--target-mode", action="append", default=[])
    parser.add_argument("--lgbm-relevance-mode", action="append", default=[], choices=("gap", "rank"))
    parser.add_argument("--lgbm-max-label", type=int, default=31)
    parser.add_argument("--lgbm-gap-scale", type=float, default=4.0)
    parser.add_argument("--lgbm-n-estimators", type=int, default=450)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.04)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=20)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.75)
    parser.add_argument("--lgbm-reg-alpha", type=float, default=0.01)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=0.05)
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument("--feature-scope", choices=("both", "selector", "state"), default="both")
    parser.add_argument("--skip-classifiers", action="store_true")
    parser.add_argument("--blend-trained", action="store_true")
    parser.add_argument("--blend-steps", type=int, default=10)
    parser.add_argument("--max-blends", type=int, default=12)
    parser.add_argument("--miss-rows", action="append", default=[], help="JSONL rows from mine_t2_union_pool_misses.py")
    parser.add_argument("--miss-target-ks", default="", help="comma-separated target_k values to use from miss rows")
    parser.add_argument("--miss-min-ev-loss", type=float, default=0.0)
    parser.add_argument("--miss-group-weight", type=float, default=8.0)
    parser.add_argument("--miss-teacher-weight", type=float, default=12.0)
    parser.add_argument("--miss-ev-loss-cap", type=float, default=2.0)
    parser.add_argument("--miss-ev-loss-group-scale", type=float, default=2.0)
    parser.add_argument("--miss-ev-loss-teacher-scale", type=float, default=3.0)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selector and not args.lgbm_selector and not args.npy_selector and args.no_base:
        raise ValueError("At least one selector is required")
    if not args.ranker:
        args.ranker = ["hgb_l31", "hgb_l63", "extra_trees_d12"]
    if not args.classifier:
        args.classifier = ["hgb_cls_l31", "extra_trees_cls_d12"]
    if not args.target_mode:
        args.target_mode = ["score"]
    run(args)


if __name__ == "__main__":
    main()
