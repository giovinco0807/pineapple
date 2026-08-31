"""Train a T2 ranker with explicit source/T3 runtime candidate features.

The dim693 dataset already contains board/action/opportunity features, and
``train_t2_selector_feature_ranker.py`` adds agreement features from multiple
selectors.  This diagnostic adds the T2 source row values that are also
available at runtime before exact refinement: T3 model EV, T3 FL/bust estimates,
source rank, draw counts, and their within-group ranks/gaps.

Teacher EV is used only as the supervised target and evaluation label.
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
from lightgbm import LGBMRanker
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_selector_feature_ranker import (
    ModelSpec,
    SelectorData,
    SelectorSpec,
    compact_metrics,
    load_selector_scores,
    parse_lgbm_selector,
    parse_model,
    parse_selector,
)


SOURCE_RAW_KEYS = [
    "source_score",
    "model_t3_value_score",
    "model_t3_priority_score",
    "model_t3_value_bust",
    "model_t3_value_fl",
    "t3_raw_top1_match_rate",
    "draws",
    "t3_states_scored",
    "rank",
    "source_index",
]


@dataclass(frozen=True)
class SourceDataSpec:
    name: str
    path: Path
    source_path: Path


@dataclass
class SourceFeatureData:
    selector_data: SelectorData
    source_features: np.ndarray
    source_feature_names: list[str]


def parse_source_data(value: str) -> SourceDataSpec:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise ValueError("--train-data/--eval-data must be name=dataset_dir,source_jsonl")
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2 or not items[0] or not items[1]:
        raise ValueError("--train-data/--eval-data must be name=dataset_dir,source_jsonl")
    return SourceDataSpec(parts[0].strip(), Path(items[0]), Path(items[1]))


def canonical_action_key(action: dict[str, Any]) -> str:
    placements = action.get("placements") or []
    norm_placements = sorted(
        [(str(card), str(row)) for card, row in placements],
        key=lambda item: (item[0], item[1]),
    )
    discard = action.get("discard")
    payload = {
        "discard": None if discard in ("", None) else str(discard),
        "placements": norm_placements,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def candidate_action(candidate: dict[str, Any]) -> dict[str, Any]:
    action = candidate.get("action")
    if isinstance(action, dict):
        return action
    return {
        "placements": candidate.get("placements") or [],
        "discard": candidate.get("discard"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_path_from_dataset(data_dir: Path) -> Path:
    meta_path = data_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source = meta.get("source")
    if not source:
        raise ValueError(f"metadata has no source: {meta_path}")
    return Path(source)


def source_candidate_lookup(source_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(source_row.get("candidates") or []):
        action = candidate_action(candidate)
        key = canonical_action_key(action)
        row = dict(candidate)
        row.setdefault("source_index", idx)
        lookup[key] = row
    return lookup


def raw_source_vector(source_candidate: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    values = np.zeros(len(SOURCE_RAW_KEYS), dtype=np.float32)
    if not source_candidate:
        return values, 1.0
    for idx, key in enumerate(SOURCE_RAW_KEYS):
        value = source_candidate.get(key, 0.0)
        if value in (None, ""):
            value = 0.0
        try:
            values[idx] = float(value)
        except (TypeError, ValueError):
            values[idx] = 0.0
    return values, 0.0


def build_source_features(data_dir: Path, source_path: Path, bounds: list[tuple[int, int]]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    teacher_path = teacher_path_from_dataset(data_dir)
    teacher_rows = read_jsonl(teacher_path)
    source_rows = read_jsonl(source_path)

    raw_rows: list[np.ndarray] = []
    missing_flags: list[float] = []
    row_count_mismatch = len(teacher_rows) != len(source_rows)
    missing_candidates = 0

    for row_idx, teacher_row in enumerate(teacher_rows):
        source_row = source_rows[row_idx] if row_idx < len(source_rows) else {}
        lookup = source_candidate_lookup(source_row)
        for candidate in teacher_row.get("candidates") or []:
            key = canonical_action_key(candidate_action(candidate))
            raw, missing = raw_source_vector(lookup.get(key))
            if missing:
                missing_candidates += 1
            raw_rows.append(raw)
            missing_flags.append(missing)

    if not raw_rows:
        raise ValueError(f"no source features built for {data_dir}")

    raw_arr = np.stack(raw_rows).astype(np.float32)
    missing_arr = np.asarray(missing_flags, dtype=np.float32).reshape(-1, 1)
    n = len(raw_arr)
    if bounds and bounds[-1][1] != n:
        raise ValueError(f"source/teacher sample count mismatch for {data_dir}: features={n}, bounds_end={bounds[-1][1]}")

    names: list[str] = []
    names.extend(f"source_{key}" for key in SOURCE_RAW_KEYS)
    names.append("source_missing")
    for key in SOURCE_RAW_KEYS:
        names.extend(
            [
                f"source_{key}_rank_frac",
                f"source_{key}_gap_to_top",
                f"source_{key}_z",
                f"source_{key}_is_top1",
                f"source_{key}_is_top3",
                f"source_{key}_is_top5",
            ]
        )
    names.extend(
        [
            "source_group_size_frac",
            "source_t3_ev_minus_source_score",
            "source_priority_minus_ev",
            "source_fl_minus_bust",
            "source_ev_times_match",
            "source_rank_inverse",
        ]
    )
    features = np.zeros((n, len(names)), dtype=np.float32)
    features[:, : len(SOURCE_RAW_KEYS)] = raw_arr
    offset = len(SOURCE_RAW_KEYS)
    features[:, offset : offset + 1] = missing_arr
    offset += 1

    for start, end in bounds:
        group_size = max(end - start, 1)
        group = raw_arr[start:end].astype(np.float64)
        tie_break = raw_arr[start:end, SOURCE_RAW_KEYS.index("source_index")].astype(np.float64)
        for col in range(group.shape[1]):
            values = group[:, col]
            # For bust/rank/source_index lower is better; all other source keys
            # are scored in descending order.
            descending = SOURCE_RAW_KEYS[col] not in {"model_t3_value_bust", "rank", "source_index"}
            std = float(values.std())
            top1 = np.zeros(group_size, dtype=np.float64)
            top3 = np.zeros(group_size, dtype=np.float64)
            top5 = np.zeros(group_size, dtype=np.float64)
            if std < 1e-9:
                rank_frac = np.full(group_size, 0.5, dtype=np.float64)
                gap = np.zeros(group_size, dtype=np.float64)
                z = np.zeros(group_size, dtype=np.float64)
            else:
                primary = -values if descending else values
                order = np.lexsort((tie_break, primary))
                ranks = np.empty(group_size, dtype=np.float64)
                ranks[order] = np.arange(group_size, dtype=np.float64)
                rank_frac = ranks / max(group_size - 1, 1)
                best_value = float(values[order[0]])
                gap = (best_value - values) if descending else (values - best_value)
                z = (values - float(values.mean())) / std
                top1[order[:1]] = 1.0
                top3[order[: min(3, group_size)]] = 1.0
                top5[order[: min(5, group_size)]] = 1.0
            block = np.stack([rank_frac, gap, z, top1, top3, top5], axis=1)
            features[start:end, offset : offset + 6] = block.astype(np.float32)
            offset += 6
        features[start:end, offset] = group_size / 30.0
        features[start:end, offset + 1] = raw_arr[start:end, 1] - raw_arr[start:end, 0]
        features[start:end, offset + 2] = raw_arr[start:end, 2] - raw_arr[start:end, 1]
        features[start:end, offset + 3] = raw_arr[start:end, 4] - raw_arr[start:end, 3]
        features[start:end, offset + 4] = raw_arr[start:end, 1] * raw_arr[start:end, 5]
        features[start:end, offset + 5] = 1.0 / np.maximum(raw_arr[start:end, 8], 1.0)
        offset -= len(SOURCE_RAW_KEYS) * 6
    # The loop reuses the same source-rank block offset for every group.
    # Advance to the derived feature block for the shape assertion below.
    final_offset = len(SOURCE_RAW_KEYS) + 1 + len(SOURCE_RAW_KEYS) * 6 + 6
    assert final_offset == features.shape[1], (final_offset, features.shape[1])

    meta = {
        "teacher_path": str(teacher_path),
        "source_path": str(source_path),
        "teacher_rows": len(teacher_rows),
        "source_rows": len(source_rows),
        "row_count_mismatch": bool(row_count_mismatch),
        "missing_candidates": int(missing_candidates),
        "feature_dim": int(features.shape[1]),
    }
    return features, names, meta


def build_source_feature_block(data_spec: SourceDataSpec, selector_data: SelectorData) -> SourceFeatureData:
    features, names, meta = build_source_features(data_spec.path, data_spec.source_path, selector_data.bounds)
    selector_data.source_feature_meta = meta  # type: ignore[attr-defined]
    return SourceFeatureData(selector_data=selector_data, source_features=features, source_feature_names=names)


def concat_arrays(datasets: list[SourceFeatureData], attr: str) -> np.ndarray:
    return np.concatenate([getattr(ds.selector_data, attr) for ds in datasets], axis=0)


def concat_source_arrays(datasets: list[SourceFeatureData]) -> np.ndarray:
    return np.concatenate([ds.source_features for ds in datasets], axis=0)


def concat_bounds(datasets: list[SourceFeatureData]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    offset = 0
    for ds in datasets:
        bounds.extend((start + offset, end + offset) for start, end in ds.selector_data.bounds)
        offset += len(ds.selector_data.scores)
    return bounds


def model_features(datasets: list[SourceFeatureData], scope: str) -> np.ndarray:
    states = concat_arrays(datasets, "x")
    selector = concat_arrays(datasets, "selector_features")
    source = concat_source_arrays(datasets)
    if scope == "source":
        return source.astype(np.float32)
    if scope == "selector_source":
        return np.concatenate([selector, source], axis=1).astype(np.float32)
    if scope == "state_source":
        return np.concatenate([states, source], axis=1).astype(np.float32)
    if scope == "all":
        return np.concatenate([states, selector, source], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature scope: {scope}")


def group_best_mask(scores: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float32)
    for start, end in bounds:
        if end <= start:
            continue
        out[start + int(np.argmax(scores[start:end]))] = 1.0
    return out


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
        group = np.asarray(scores[start:end], dtype=np.float64)
        if len(group) == 0:
            continue
        order = np.argsort(-group)
        if mode == "rank":
            ranks = np.empty(len(group), dtype=np.int32)
            ranks[order] = np.arange(len(group), dtype=np.int32)
            labels[start:end] = np.maximum(0, int(max_label) - ranks)
            continue
        best = float(group[order[0]])
        gaps = np.maximum(0.0, best - group)
        raw = np.rint(float(max_label) - gaps * float(gap_scale)).astype(np.int32)
        labels[start:end] = np.clip(raw, 0, int(max_label))
    return labels


def regression_target(scores: np.ndarray, bounds: list[tuple[int, int]], mode: str) -> np.ndarray:
    target = scores.astype(np.float32).copy()
    if mode == "score":
        return target
    if mode not in {"advantage", "top_weighted"}:
        raise ValueError(f"unknown target mode: {mode}")
    for start, end in bounds:
        group = scores[start:end].astype(np.float64)
        if len(group) == 0:
            continue
        if mode == "advantage":
            target[start:end] = (group - float(group.mean())).astype(np.float32)
            continue
        order = np.argsort(-group)
        bonus = np.zeros(len(group), dtype=np.float64)
        bonus[order[:1]] += 0.6
        bonus[order[: min(3, len(group))]] += 0.15
        target[start:end] = (group + bonus).astype(np.float32)
    return target


def sample_weights(scores: np.ndarray, bounds: list[tuple[int, int]], best_weight: float, gap_weight: float, gap_cap: float) -> np.ndarray:
    weights = np.ones(len(scores), dtype=np.float32)
    for start, end in bounds:
        group = scores[start:end].astype(np.float64)
        if len(group) == 0:
            continue
        best = float(group.max())
        gaps = np.clip(best - group, 0.0, gap_cap)
        weights[start:end] += (gaps * gap_weight).astype(np.float32)
        weights[start + int(np.argmax(group))] += float(best_weight)
    return weights


def make_regressor(kind: str, seed: int):
    if kind == "hgb_l31":
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=520,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=24,
            random_state=seed,
        )
    if kind == "hgb_l63":
        return HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=650,
            max_leaf_nodes=63,
            l2_regularization=0.04,
            min_samples_leaf=18,
            random_state=seed,
        )
    if kind == "extra_d12":
        return ExtraTreesRegressor(
            n_estimators=360,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "extra_d16":
        return ExtraTreesRegressor(
            n_estimators=420,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown regressor: {kind}")


def make_classifier(kind: str, seed: int):
    if kind == "hgb_l31":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=520,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=24,
            random_state=seed,
        )
    if kind == "extra_d12":
        return ExtraTreesClassifier(
            n_estimators=360,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.55,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown classifier: {kind}")


def make_lgbm_ranker(args: argparse.Namespace, seed: int) -> LGBMRanker:
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


def classifier_score(model, x: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(x)
    if prob.shape[1] == 1:
        return prob[:, 0].astype(np.float32)
    return prob[:, 1].astype(np.float32)


def evaluate_prediction(datasets: list[SourceFeatureData], pred: np.ndarray, topks: list[int]) -> dict[str, Any]:
    scores = concat_arrays(datasets, "scores")
    bounds = concat_bounds(datasets)
    metrics, _rows = summarize_groups(scores, pred.astype(np.float32), bounds, topks)
    return compact_metrics(metrics)


def selector_prediction(datasets: list[SourceFeatureData], name: str) -> np.ndarray:
    return np.concatenate([ds.selector_data.selector_scores[name] for ds in datasets], axis=0)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T2 Source-Feature Ranker",
        "",
        f"- status: `{summary['status']}`",
        f"- feature scope: `{summary['feature_scope']}`",
        f"- train groups: `{summary['train_groups']}`",
        f"- eval groups: `{summary['eval_groups']}`",
        f"- source feature dim: `{summary['source_feature_dim']}`",
        "",
        "## Eval",
        "",
        "| selector | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["eval_rows"]:
        m = row["aggregate"]
        lines.append(
            f"| {row['selector']} | "
            f"{m.get('top1', 0):.1%} | {m.get('top3', 0):.1%} | "
            f"{m.get('top5', 0):.1%} | {m.get('top10', 0):.1%} | "
            f"{m.get('top20', 0):.1%} | {m.get('reg1', 0):.3f} | {m.get('reg10', 0):.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)

    model_specs = {spec.name: spec for spec in [parse_model(value) for value in args.model]}
    selectors = [parse_selector(value) for value in args.selector]
    lgbm_selectors = [parse_lgbm_selector(value) for value in args.lgbm_selector]
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}

    train_specs = [parse_source_data(value) for value in args.train_data]
    eval_specs = [parse_source_data(value) for value in args.eval_data]

    def load(spec: SourceDataSpec) -> SourceFeatureData:
        selector_data = load_selector_scores(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            include_base=not args.no_base,
        )
        return build_source_feature_block(spec, selector_data)

    train_sets = [load(spec) for spec in train_specs]
    eval_sets = [load(spec) for spec in eval_specs]
    x_train = model_features(train_sets, args.feature_scope)
    x_eval = model_features(eval_sets, args.feature_scope)
    y_train = concat_arrays(train_sets, "scores")
    train_bounds = concat_bounds(train_sets)
    train_weights = sample_weights(y_train, train_bounds, args.best_weight, args.gap_weight, args.gap_cap)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    predictor_pool: list[dict[str, Any]] = []

    for name in train_sets[0].selector_data.selector_scores:
        train_pred = selector_prediction(train_sets, name)
        eval_pred = selector_prediction(eval_sets, name)
        train_rows.append({"selector": name, "aggregate": evaluate_prediction(train_sets, train_pred, topks)})
        eval_metric = evaluate_prediction(eval_sets, eval_pred, topks)
        eval_rows.append({"selector": name, "aggregate": eval_metric})
        predictor_pool.append({"name": name, "eval_pred": eval_pred, "source": "selector", "metrics": eval_metric})

    for target_mode in args.target_mode:
        target = regression_target(y_train, train_bounds, target_mode)
        for kind in args.regressor:
            model = make_regressor(kind, args.seed)
            model.fit(x_train, target, sample_weight=train_weights)
            name = f"{kind}_{target_mode}_{args.feature_scope}"
            path = out_dir / f"{name}.joblib"
            joblib.dump(model, path)
            train_pred = model.predict(x_train).astype(np.float32)
            eval_pred = model.predict(x_eval).astype(np.float32)
            train_rows.append({"selector": name, "aggregate": evaluate_prediction(train_sets, train_pred, topks)})
            eval_metric = evaluate_prediction(eval_sets, eval_pred, topks)
            eval_rows.append({"selector": name, "aggregate": eval_metric, "model": str(path)})
            predictor_pool.append({"name": name, "eval_pred": eval_pred, "source": "regressor", "model": str(path), "metrics": eval_metric})

    best_mask = group_best_mask(y_train, train_bounds)
    cls_weights = train_weights + best_mask * args.class_best_weight
    for kind in args.classifier:
        model = make_classifier(kind, args.seed)
        model.fit(x_train, best_mask, sample_weight=cls_weights)
        name = f"{kind}_bestcls_{args.feature_scope}"
        path = out_dir / f"{name}.joblib"
        joblib.dump(model, path)
        train_pred = classifier_score(model, x_train)
        eval_pred = classifier_score(model, x_eval)
        train_rows.append({"selector": name, "aggregate": evaluate_prediction(train_sets, train_pred, topks)})
        eval_metric = evaluate_prediction(eval_sets, eval_pred, topks)
        eval_rows.append({"selector": name, "aggregate": eval_metric, "model": str(path)})
        predictor_pool.append({"name": name, "eval_pred": eval_pred, "source": "classifier", "model": str(path), "metrics": eval_metric})

    for offset, relevance_mode in enumerate(args.lgbm_relevance_mode):
        relevance = build_relevance(
            y_train,
            train_bounds,
            mode=relevance_mode,
            max_label=args.lgbm_max_label,
            gap_scale=args.lgbm_gap_scale,
        )
        model = make_lgbm_ranker(args, args.seed + 700 + offset)
        model.fit(x_train, relevance, group=group_sizes(train_bounds), sample_weight=train_weights if args.lgbm_use_weights else None)
        name = f"lgbm_rank_{relevance_mode}_{args.feature_scope}"
        path = out_dir / f"{name}.joblib"
        joblib.dump(model, path)
        train_pred = model.predict(x_train).astype(np.float32)
        eval_pred = model.predict(x_eval).astype(np.float32)
        train_rows.append({"selector": name, "aggregate": evaluate_prediction(train_sets, train_pred, topks)})
        eval_metric = evaluate_prediction(eval_sets, eval_pred, topks)
        eval_rows.append(
            {
                "selector": name,
                "aggregate": eval_metric,
                "model": str(path),
                "target": f"lgbm_{relevance_mode}",
                "label_min": int(relevance.min()) if len(relevance) else 0,
                "label_max": int(relevance.max()) if len(relevance) else 0,
            }
        )
        predictor_pool.append({"name": name, "eval_pred": eval_pred, "source": "lgbm_ranker", "model": str(path), "metrics": eval_metric})

    # Simple train-free eval blends: sometimes the classifier picks Top1 but a
    # regressor gives better EV tails.
    blend_rows: list[dict[str, Any]] = []
    for left in predictor_pool:
        for right in predictor_pool:
            if left is right or left["source"] == right["source"] == "selector":
                continue
            for alpha in args.blend_alpha:
                pred = float(alpha) * np.asarray(left["eval_pred"], dtype=np.float32) + (1.0 - float(alpha)) * np.asarray(
                    right["eval_pred"], dtype=np.float32
                )
                name = f"blend:{float(alpha):.2f}:{left['name']}+{1.0-float(alpha):.2f}:{right['name']}"
                metrics = evaluate_prediction(eval_sets, pred, topks)
                blend_rows.append({"selector": name, "aggregate": metrics})
    blend_rows.sort(key=lambda row: (-row["aggregate"].get("top1", 0.0), row["aggregate"].get("reg1", 999.0)))
    eval_rows.extend(blend_rows[: args.keep_blends])

    eval_rows.sort(key=lambda row: (-row["aggregate"].get("top1", 0.0), row["aggregate"].get("reg1", 999.0)))
    train_rows.sort(key=lambda row: (-row["aggregate"].get("top1", 0.0), row["aggregate"].get("reg1", 999.0)))

    source_metas = {
        ds.selector_data.spec.name: getattr(ds.selector_data, "source_feature_meta", {}) for ds in [*train_sets, *eval_sets]
    }
    summary: dict[str, Any] = {
        "status": "diagnostic",
        "script": "ai/training/train_t2_source_feature_ranker.py",
        "feature_scope": args.feature_scope,
        "source_feature_dim": int(train_sets[0].source_features.shape[1]),
        "source_feature_names": train_sets[0].source_feature_names,
        "train_data": [{"name": spec.name, "path": str(spec.path), "source": str(spec.source_path)} for spec in train_specs],
        "eval_data": [{"name": spec.name, "path": str(spec.path), "source": str(spec.source_path)} for spec in eval_specs],
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": [selector.name for selector in selectors],
        "lgbm_selectors": {name: str(path) for name, path in lgbm_selectors},
        "feature_meta": source_metas,
        "train_groups": int(sum(len(ds.selector_data.bounds) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.selector_data.bounds) for ds in eval_sets)),
        "train_samples": int(sum(len(ds.selector_data.scores) for ds in train_sets)),
        "eval_samples": int(sum(len(ds.selector_data.scores) for ds in eval_sets)),
        "regressors": args.regressor,
        "classifiers": args.classifier,
        "lgbm_relevance_modes": args.lgbm_relevance_mode,
        "lgbm_params": {
            "n_estimators": args.lgbm_n_estimators,
            "learning_rate": args.lgbm_learning_rate,
            "num_leaves": args.lgbm_num_leaves,
            "min_child_samples": args.lgbm_min_child_samples,
            "subsample": args.lgbm_subsample,
            "colsample_bytree": args.lgbm_colsample_bytree,
            "reg_alpha": args.lgbm_reg_alpha,
            "reg_lambda": args.lgbm_reg_lambda,
            "max_label": args.lgbm_max_label,
            "gap_scale": args.lgbm_gap_scale,
            "use_weights": args.lgbm_use_weights,
        },
        "target_modes": args.target_mode,
        "train_rows": train_rows[:20],
        "eval_rows": eval_rows[:60],
        "topks": topks,
        "elapsed_seconds": time.time() - t0,
        "decision": "diagnostic only until it improves clean external Top1 without degrading TopK exact-rerank pools",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", summary)
    print(json.dumps(summary["eval_rows"][:10], indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a T2 source-feature diagnostic ranker")
    parser.add_argument("--train-data", action="append", required=True, help="name=dataset_dir,source_jsonl")
    parser.add_argument("--eval-data", action="append", required=True, help="name=dataset_dir,source_jsonl")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--feature-scope", choices=("source", "selector_source", "state_source", "all"), default="all")
    parser.add_argument("--target-mode", action="append", default=[])
    parser.add_argument("--regressor", action="append", default=[])
    parser.add_argument("--classifier", action="append", default=[])
    parser.add_argument("--lgbm-relevance-mode", action="append", default=[], choices=("gap", "rank"))
    parser.add_argument("--lgbm-n-estimators", type=int, default=450)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.035)
    parser.add_argument("--lgbm-num-leaves", type=int, default=63)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=12)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--lgbm-reg-alpha", type=float, default=0.01)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=0.1)
    parser.add_argument("--lgbm-max-label", type=int, default=31)
    parser.add_argument("--lgbm-gap-scale", type=float, default=4.0)
    parser.add_argument("--lgbm-n-jobs", type=int, default=4)
    parser.add_argument("--lgbm-use-weights", action="store_true")
    parser.add_argument("--blend-alpha", type=float, action="append", default=[])
    parser.add_argument("--keep-blends", type=int, default=20)
    parser.add_argument("--best-weight", type=float, default=5.0)
    parser.add_argument("--class-best-weight", type=float, default=10.0)
    parser.add_argument("--gap-weight", type=float, default=0.25)
    parser.add_argument("--gap-cap", type=float, default=8.0)
    parser.add_argument("--topks", default="1,3,5,8,10,13,15,20")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.target_mode:
        args.target_mode = ["score", "advantage", "top_weighted"]
    if not args.regressor:
        args.regressor = ["hgb_l31", "hgb_l63", "extra_d12", "extra_d16"]
    if not args.classifier:
        args.classifier = ["hgb_l31", "extra_d12"]
    if not args.blend_alpha:
        args.blend_alpha = [0.2, 0.35, 0.5, 0.65, 0.8]
    if not args.model and not args.lgbm_selector and args.no_base:
        raise ValueError("At least one selector source is required")
    run(args)


if __name__ == "__main__":
    main()
