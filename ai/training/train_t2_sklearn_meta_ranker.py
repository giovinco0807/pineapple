"""Train no-leak sklearn meta rankers for T2 candidate ordering.

Inputs are limited to runtime-available features:

- candidate state vectors
- base action-value scores
- base-score group context such as rank and gap to group best

Teacher EV is used only as the supervised target.  Teacher ranks, route tags,
bust labels, and FL labels are intentionally not used as model inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_action_value_set_reranker import build_group_bounds, load_metadata


def sample_count(data_dir: Path) -> int:
    meta = load_metadata(data_dir)
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    return min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))


def load_runtime_features(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    n = sample_count(data_dir)
    states = np.asarray(np.load(data_dir / "states.npy", mmap_mode="r")[:n], dtype=np.float32)
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n], dtype=np.float32)
    base_scores = np.asarray(np.load(data_dir / "base_scores.npy", mmap_mode="r")[:n], dtype=np.float32)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    group_features = np.zeros((n, 4), dtype=np.float32)
    for start, end in bounds:
        base = base_scores[start:end].astype(np.float64, copy=False)
        order = np.argsort(-base)
        ranks = np.empty(len(base), dtype=np.float32)
        ranks[order] = np.arange(len(base), dtype=np.float32)
        denom = max(len(base) - 1, 1)
        mean = float(base.mean())
        std = float(base.std()) if len(base) > 1 else 1.0
        if std < 1e-6:
            std = 1.0
        group_features[start:end, 0] = ranks / denom
        group_features[start:end, 1] = (float(base.max()) - base).astype(np.float32)
        group_features[start:end, 2] = ((base - mean) / std).astype(np.float32)
        group_features[start:end, 3] = len(base) / 30.0
    x = np.concatenate([states, base_scores.reshape(-1, 1), group_features], axis=1)
    return x, scores, base_scores, bounds


def make_model(kind: str, seed: int):
    if kind == "hgb_l15":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=450,
            max_leaf_nodes=15,
            l2_regularization=0.02,
            min_samples_leaf=24,
            random_state=seed,
        )
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
    if kind == "extra_trees_d10":
        return ExtraTreesRegressor(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=4,
            max_features=0.45,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "groups",
        "samples",
        "group_top1",
        "group_top3",
        "group_top5",
        "group_top10",
        "group_top15",
        "group_top20",
        "group_top1_regret",
        "group_top3_rerank_regret",
        "group_top10_rerank_regret",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def evaluate_predictions(
    scores: np.ndarray,
    base_scores: np.ndarray,
    residual_pred: np.ndarray,
    bounds: list[tuple[int, int]],
    topks: list[int],
    gammas: list[float],
) -> list[dict]:
    rows = []
    base_metrics, _ = summarize_groups(scores, base_scores, bounds, topks)
    rows.append({"selector": "base_scores", "gamma": 0.0, "metrics": compact_metrics(base_metrics)})
    for gamma in gammas:
        pred = base_scores + float(gamma) * residual_pred
        metrics, _ = summarize_groups(scores, pred.astype(np.float32), bounds, topks)
        rows.append({"selector": "meta_residual", "gamma": float(gamma), "metrics": compact_metrics(metrics)})
    rows.sort(
        key=lambda row: (
            float(row["metrics"].get("group_top1", 0.0)),
            -float(row["metrics"].get("group_top1_regret", 0.0)),
            float(row["metrics"].get("group_top3", 0.0)),
            -float(row["metrics"].get("group_top10_rerank_regret", 0.0)),
        ),
        reverse=True,
    )
    return rows


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 No-Leak Sklearn Meta Ranker",
        "",
        f"- train_data: `{summary['train_data']}`",
        f"- eval_data: `{summary['eval_data']}`",
        f"- target: `{summary['target']}`",
        "- inputs: states, base_scores, base rank/gap/z/group-size runtime context",
        "- excluded inputs: candidate_ranks, route_tags, bust labels, FL labels",
        "",
        "| model | gamma | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, rows in summary["eval_rows"].items():
        for row in rows[:8]:
            m = row["metrics"]
            lines.append(
                f"| {model_name}:{row['selector']} | {float(row['gamma']):.2f} | "
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
    train_dir = Path(args.train_data)
    eval_dir = Path(args.eval_data)
    topks = parse_topks(args.topks)
    gammas = [float(part) for part in args.gammas.split(",") if part.strip()]

    x_train, y_train, base_train, _train_bounds = load_runtime_features(train_dir)
    target = y_train - base_train if args.target == "residual" else y_train
    sample_weight = None
    if args.use_sample_weights and (train_dir / "sample_weights.npy").exists():
        sample_weight = np.asarray(np.load(train_dir / "sample_weights.npy", mmap_mode="r")[: len(target)], dtype=np.float32)

    x_eval, y_eval, base_eval, eval_bounds = load_runtime_features(eval_dir)
    eval_rows: dict[str, list[dict]] = {}
    models = {}
    for offset, kind in enumerate(args.model):
        model = make_model(kind, args.seed + offset)
        t0 = time.time()
        if sample_weight is not None:
            model.fit(x_train, target, sample_weight=sample_weight)
        else:
            model.fit(x_train, target)
        model_path = out_dir / f"{kind}.joblib"
        joblib.dump(model, model_path)
        pred = model.predict(x_eval).astype(np.float32)
        residual_pred = pred if args.target == "residual" else pred - base_eval
        eval_rows[kind] = evaluate_predictions(y_eval, base_eval, residual_pred, eval_bounds, topks, gammas)
        models[kind] = {"path": str(model_path), "fit_seconds": time.time() - t0}

    summary = {
        "train_data": str(train_dir),
        "eval_data": str(eval_dir),
        "target": args.target,
        "models": models,
        "gammas": gammas,
        "topks": topks,
        "use_sample_weights": bool(args.use_sample_weights),
        "n_train_samples": int(len(y_train)),
        "n_eval_samples": int(len(y_eval)),
        "n_features": int(x_train.shape[1]),
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", choices=("residual", "score"), default="residual")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--gammas", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.model:
        args.model = ["hgb_l15", "hgb_l31", "extra_trees_d10"]
    run(args)


if __name__ == "__main__":
    main()
