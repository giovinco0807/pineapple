"""Evaluate T2 EV selectors with auxiliary FL/bust target blends.

The goal is diagnostic: check whether remaining Top1 EV-loss misses can be
rescued by predicting high-value FL opportunity and bust risk separately, then
adding those runtime-available predictions to an existing EV selector.

Inputs at inference time are intentionally no-leak:

- encoded candidate state
- base action-value score
- base-score group context from ``load_runtime_features``

Teacher EV, FL, FL type, and bust labels are used only as supervised targets
and as evaluation labels.
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
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_sklearn_meta_ranker import compact_metrics, load_runtime_features, sample_count


FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")
FL_TYPE_REWARDS = np.asarray([0.0, 10.7, 29.9, 63.5], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    target: str


@dataclass(frozen=True)
class EvalSpec:
    name: str
    path: Path


@dataclass
class EvalDataset:
    spec: EvalSpec
    x: np.ndarray
    scores: np.ndarray
    base_scores: np.ndarray
    bounds: list[tuple[int, int]]


def parse_named_path(value: str) -> EvalSpec:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("expected name=path")
    return EvalSpec(name=parts[0].strip(), path=Path(parts[1].strip()))


def parse_model(value: str) -> ModelSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--ev-model must be name=path,target")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--ev-model must be name=path,target")
    target = items[1].lower()
    if target not in {"score", "residual"}:
        raise ValueError("model target must be score or residual")
    return ModelSpec(name=name, path=Path(items[0]), target=target)


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def load_aux_targets(data_dir: Path) -> dict[str, np.ndarray]:
    n = sample_count(data_dir)
    fl = np.asarray(np.load(data_dir / "fl.npy", mmap_mode="r")[:n], dtype=np.float32)
    bust = np.asarray(np.load(data_dir / "bust.npy", mmap_mode="r")[:n], dtype=np.float32)
    fl_types = np.asarray(np.load(data_dir / "fl_types.npy", mmap_mode="r")[:n], dtype=np.float32)
    if fl_types.ndim != 2 or fl_types.shape[1] != len(FL_TYPE_KEYS):
        raise ValueError(f"{data_dir}: expected fl_types shape (n, 4), got {fl_types.shape}")
    fl_value = fl_types @ FL_TYPE_REWARDS
    out: dict[str, np.ndarray] = {
        "fl": fl,
        "bust": bust,
        "fl_value": fl_value.astype(np.float32),
    }
    for idx, key in enumerate(FL_TYPE_KEYS):
        out[f"fl_{key}"] = fl_types[:, idx].astype(np.float32)
    out["fl_aa_kk_trips"] = (
        out["fl_aa"] * FL_TYPE_REWARDS[2]
        + out["fl_kk"] * FL_TYPE_REWARDS[1]
        + out["fl_trips"] * FL_TYPE_REWARDS[3]
    ).astype(np.float32)
    return out


def make_aux_model(seed: int, target: str) -> HistGradientBoostingRegressor:
    if target in {"fl_value", "fl_aa_kk_trips"}:
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=220,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            min_samples_leaf=22,
            random_state=seed,
        )
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        l2_regularization=0.02,
        min_samples_leaf=24,
        random_state=seed,
    )


def fit_aux_models(
    x_train: np.ndarray,
    targets: dict[str, np.ndarray],
    target_names: list[str],
    out_dir: Path,
    seed: int,
    sample_weight: np.ndarray | None,
) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for offset, target_name in enumerate(target_names):
        started = time.time()
        model = make_aux_model(seed + offset, target_name)
        y = targets[target_name]
        if sample_weight is not None:
            model.fit(x_train, y, sample_weight=sample_weight)
        else:
            model.fit(x_train, y)
        model_path = out_dir / f"aux_{target_name}.joblib"
        joblib.dump(model, model_path)
        models[target_name] = {
            "path": str(model_path),
            "fit_seconds": time.time() - started,
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
        }
    return models


def load_ev_prediction(
    x: np.ndarray,
    base_scores: np.ndarray,
    specs: list[ModelSpec],
    gamma: float,
) -> tuple[np.ndarray, dict]:
    residuals = []
    metadata = {}
    for spec in specs:
        model = joblib.load(spec.path)
        pred = model.predict(x).astype(np.float32)
        residual = pred if spec.target == "residual" else pred - base_scores
        residuals.append(residual)
        metadata[spec.name] = {"path": str(spec.path), "target": spec.target}
    avg_residual = np.mean(residuals, axis=0).astype(np.float32)
    return (base_scores + float(gamma) * avg_residual).astype(np.float32), metadata


def predict_aux_features(
    x: np.ndarray,
    aux_models: dict[str, dict],
    target_names: list[str],
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for target_name in target_names:
        model = joblib.load(aux_models[target_name]["path"])
        pred = model.predict(x).astype(np.float32)
        if target_name.startswith("fl") or target_name == "bust":
            pred = np.maximum(pred, 0.0)
        preds[target_name] = pred.astype(np.float32)
    return preds


def group_center(values: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    centered = np.empty_like(values, dtype=np.float32)
    for start, end in bounds:
        local = np.asarray(values[start:end], dtype=np.float32)
        centered[start:end] = local - float(np.mean(local))
    return centered


def group_z(values: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    scaled = np.empty_like(values, dtype=np.float32)
    for start, end in bounds:
        local = np.asarray(values[start:end], dtype=np.float32)
        std = float(np.std(local))
        if std < 1e-6:
            std = 1.0
        scaled[start:end] = (local - float(np.mean(local))) / std
    return scaled


def aggregate_eval(
    datasets: list[EvalDataset],
    pred_by_name: dict[str, np.ndarray],
    topks: list[int],
) -> dict:
    all_scores: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_bounds: list[tuple[int, int]] = []
    offset = 0
    for ds in datasets:
        pred = pred_by_name[ds.spec.name]
        all_scores.append(ds.scores)
        all_pred.append(pred)
        all_bounds.extend((start + offset, end + offset) for start, end in ds.bounds)
        offset += len(ds.scores)
    metrics, _ = summarize_groups(np.concatenate(all_scores), np.concatenate(all_pred), all_bounds, topks)
    return compact_metrics(metrics)


def evaluate_recipe(
    datasets: list[EvalDataset],
    pred_by_name: dict[str, np.ndarray],
    topks: list[int],
) -> dict:
    per_dataset = {}
    for ds in datasets:
        metrics, _ = summarize_groups(ds.scores, pred_by_name[ds.spec.name], ds.bounds, topks)
        per_dataset[ds.spec.name] = compact_metrics(metrics)
    return {"aggregate": aggregate_eval(datasets, pred_by_name, topks), "datasets": per_dataset}


def build_recipe_predictions(
    ev_scores: dict[str, np.ndarray],
    aux_preds: dict[str, dict[str, np.ndarray]],
    datasets: list[EvalDataset],
    fl_weights: list[float],
    bust_weights: list[float],
) -> list[dict]:
    rows: list[dict] = []

    rows.append(
        {
            "name": "ev_only",
            "weights": {},
            "predictions": {name: values.copy() for name, values in ev_scores.items()},
        }
    )

    single_terms = ["fl_value", "fl_aa_kk_trips", "fl", "fl_aa", "fl_kk", "fl_trips"]
    for transform_name, transform in (("center", group_center), ("z", group_z)):
        for term in single_terms:
            for weight in fl_weights:
                pred_by_name = {}
                for ds in datasets:
                    aux = transform(aux_preds[ds.spec.name][term], ds.bounds)
                    pred_by_name[ds.spec.name] = (ev_scores[ds.spec.name] + float(weight) * aux).astype(np.float32)
                rows.append(
                    {
                        "name": f"ev_plus_{transform_name}_{term}",
                        "weights": {term: float(weight), "transform": transform_name},
                        "predictions": pred_by_name,
                    }
                )
        for weight in bust_weights:
            pred_by_name = {}
            for ds in datasets:
                aux = transform(aux_preds[ds.spec.name]["bust"], ds.bounds)
                pred_by_name[ds.spec.name] = (ev_scores[ds.spec.name] - float(weight) * aux).astype(np.float32)
            rows.append(
                {
                    "name": f"ev_minus_{transform_name}_bust",
                    "weights": {"bust": float(weight), "transform": transform_name},
                    "predictions": pred_by_name,
                }
            )
        for fl_term in ("fl_value", "fl_aa_kk_trips"):
            for fl_weight in fl_weights:
                for bust_weight in bust_weights:
                    pred_by_name = {}
                    for ds in datasets:
                        fl_aux = transform(aux_preds[ds.spec.name][fl_term], ds.bounds)
                        bust_aux = transform(aux_preds[ds.spec.name]["bust"], ds.bounds)
                        pred_by_name[ds.spec.name] = (
                            ev_scores[ds.spec.name]
                            + float(fl_weight) * fl_aux
                            - float(bust_weight) * bust_aux
                        ).astype(np.float32)
                    rows.append(
                        {
                            "name": f"ev_plus_{transform_name}_{fl_term}_minus_bust",
                            "weights": {
                                fl_term: float(fl_weight),
                                "bust": float(bust_weight),
                                "transform": transform_name,
                            },
                            "predictions": pred_by_name,
                        }
                    )
    return rows


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 EV + Aux Target Blend Eval",
        "",
        f"- train_data: `{summary['train_data']}`",
        f"- eval: `{', '.join(summary['eval_data'].keys())}`",
        f"- ev_gamma: `{summary['ev_gamma']}`",
        f"- aux_targets: `{', '.join(summary['aux_targets'])}`",
        "- inputs: runtime states/base/group-context only",
        "- labels used only as targets/evaluation: EV, FL, FL type, bust",
        "",
        "| recipe | weights | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"][:20]:
        m = row["metrics"]["aggregate"]
        lines.append(
            f"| {row['recipe']} | `{json.dumps(row['weights'], sort_keys=True)}` | "
            f"{float(m.get('group_top1', 0.0)):.1%} | "
            f"{float(m.get('group_top3', 0.0)):.1%} | "
            f"{float(m.get('group_top5', 0.0)):.1%} | "
            f"{float(m.get('group_top10', 0.0)):.1%} | "
            f"{float(m.get('group_top20', 0.0)):.1%} | "
            f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top3_rerank_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top10_rerank_regret', 0.0)):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    ev_specs = [parse_model(value) for value in args.ev_model]
    eval_specs = [parse_named_path(value) for value in args.eval_data]
    fl_weights = parse_float_list(args.fl_weights)
    bust_weights = parse_float_list(args.bust_weights)
    aux_targets = ["fl", "bust", "fl_value", "fl_qq", "fl_kk", "fl_aa", "fl_trips", "fl_aa_kk_trips"]

    train_dir = Path(args.train_data)
    x_train, _scores_train, _base_train, _bounds_train = load_runtime_features(train_dir)
    train_targets = load_aux_targets(train_dir)
    sample_weight = None
    if args.use_sample_weights and (train_dir / "sample_weights.npy").exists():
        sample_weight = np.asarray(np.load(train_dir / "sample_weights.npy", mmap_mode="r")[: len(x_train)], dtype=np.float32)

    aux_models = fit_aux_models(
        x_train=x_train,
        targets=train_targets,
        target_names=aux_targets,
        out_dir=out_dir,
        seed=args.seed,
        sample_weight=sample_weight,
    )

    datasets: list[EvalDataset] = []
    eval_data_summary = {}
    ev_scores: dict[str, np.ndarray] = {}
    aux_preds: dict[str, dict[str, np.ndarray]] = {}
    ev_model_meta: dict | None = None
    predict_started = time.time()
    for spec in eval_specs:
        x, scores, base_scores, bounds = load_runtime_features(spec.path)
        ds = EvalDataset(spec=spec, x=x, scores=scores, base_scores=base_scores, bounds=bounds)
        datasets.append(ds)
        eval_data_summary[spec.name] = {
            "path": str(spec.path),
            "groups": int(len(bounds)),
            "samples": int(len(scores)),
        }
        ev_pred, ev_model_meta = load_ev_prediction(x, base_scores, ev_specs, args.ev_gamma)
        ev_scores[spec.name] = ev_pred
        aux_preds[spec.name] = predict_aux_features(x, aux_models, aux_targets)
    prediction_seconds = time.time() - predict_started

    recipe_rows = []
    for recipe in build_recipe_predictions(ev_scores, aux_preds, datasets, fl_weights, bust_weights):
        metrics = evaluate_recipe(datasets, recipe["predictions"], topks)
        recipe_rows.append(
            {
                "recipe": recipe["name"],
                "weights": recipe["weights"],
                "metrics": metrics,
            }
        )
    recipe_rows.sort(
        key=lambda row: (
            float(row["metrics"]["aggregate"].get("group_top1", 0.0)),
            -float(row["metrics"]["aggregate"].get("group_top1_regret", 0.0)),
            float(row["metrics"]["aggregate"].get("group_top3", 0.0)),
            -float(row["metrics"]["aggregate"].get("group_top10_rerank_regret", 0.0)),
        ),
        reverse=True,
    )

    summary = {
        "train_data": str(train_dir),
        "eval_data": eval_data_summary,
        "ev_models": ev_model_meta,
        "ev_gamma": float(args.ev_gamma),
        "topks": topks,
        "aux_targets": aux_targets,
        "aux_models": aux_models,
        "fl_weights": fl_weights,
        "bust_weights": bust_weights,
        "use_sample_weights": bool(args.use_sample_weights),
        "n_train_samples": int(len(x_train)),
        "n_features": int(x_train.shape[1]),
        "prediction_seconds": prediction_seconds,
        "elapsed_seconds": time.time() - started,
        "rows": recipe_rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--ev-model", action="append", required=True, help="name=path,target")
    parser.add_argument("--ev-gamma", type=float, default=1.15)
    parser.add_argument("--fl-weights", default="0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--bust-weights", default="0.25,0.5,1.0,2.0,3.0,4.0")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
