"""Train a no-leak LightGBM listwise ranker for T2 candidate ordering."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from lightgbm import LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_sklearn_meta_ranker import compact_metrics, load_runtime_features


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


def group_zscore(values: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for start, end in bounds:
        local = np.asarray(values[start:end], dtype=np.float64)
        if len(local) == 0:
            continue
        std = float(local.std())
        if std < 1e-6:
            std = 1.0
        out[start:end] = ((local - float(local.mean())) / std).astype(np.float32)
    return out


def evaluate_rows(
    scores: np.ndarray,
    base_scores: np.ndarray,
    rank_scores: np.ndarray,
    bounds: list[tuple[int, int]],
    topks: list[int],
    gammas: list[float],
) -> list[dict]:
    rows: list[dict] = []
    base_metrics, _ = summarize_groups(scores, base_scores, bounds, topks)
    rows.append({"selector": "base_scores", "gamma": 0.0, "metrics": compact_metrics(base_metrics)})
    direct_metrics, _ = summarize_groups(scores, rank_scores.astype(np.float32), bounds, topks)
    rows.append({"selector": "lgbm_rank_direct", "gamma": 1.0, "metrics": compact_metrics(direct_metrics)})
    rank_z = group_zscore(rank_scores.astype(np.float32), bounds)
    for gamma in gammas:
        pred = base_scores + float(gamma) * rank_z
        metrics, _ = summarize_groups(scores, pred.astype(np.float32), bounds, topks)
        rows.append({"selector": "base_plus_rank_z", "gamma": float(gamma), "metrics": compact_metrics(metrics)})
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


def make_ranker(args: argparse.Namespace) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        label_gain=list(range(args.max_label + 1)),
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 LightGBM Ranker",
        "",
        f"- train_data: `{summary['train_data']}`",
        f"- eval_data: `{summary['eval_data']}`",
        f"- relevance_mode: `{summary['relevance']['mode']}`",
        "- inputs: states, base_scores, base rank/gap/z/group-size runtime context",
        "- excluded inputs: candidate_ranks, route_tags, bust labels, FL labels",
        "",
        "| selector | gamma | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["eval_rows"][:12]:
        m = row["metrics"]
        lines.append(
            f"| {row['selector']} | {float(row['gamma']):.2f} | "
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
    gammas = [float(part) for part in args.gammas.split(",") if part.strip()]

    train_dir = Path(args.train_data)
    eval_dir = Path(args.eval_data)
    x_train, y_train, _base_train, train_bounds = load_runtime_features(train_dir)
    x_eval, y_eval, base_eval, eval_bounds = load_runtime_features(eval_dir)
    relevance = build_relevance(
        y_train,
        train_bounds,
        mode=args.relevance_mode,
        max_label=args.max_label,
        gap_scale=args.gap_scale,
    )
    sample_weight = None
    if args.use_sample_weights and (train_dir / "sample_weights.npy").exists():
        sample_weight = np.asarray(
            np.load(train_dir / "sample_weights.npy", mmap_mode="r")[: len(y_train)],
            dtype=np.float32,
        )

    ranker = make_ranker(args)
    fit_started = time.time()
    ranker.fit(
        x_train,
        relevance,
        group=group_sizes(train_bounds),
        sample_weight=sample_weight,
    )
    fit_seconds = time.time() - fit_started
    model_path = out_dir / "lgbm_ranker.joblib"
    joblib.dump(ranker, model_path)

    pred = ranker.predict(x_eval).astype(np.float32)
    rows = evaluate_rows(y_eval, base_eval, pred, eval_bounds, topks, gammas)
    summary = {
        "train_data": str(train_dir),
        "eval_data": str(eval_dir),
        "model": str(model_path),
        "fit_seconds": fit_seconds,
        "elapsed_seconds": time.time() - started,
        "n_train_samples": int(len(y_train)),
        "n_train_groups": int(len(train_bounds)),
        "n_eval_samples": int(len(y_eval)),
        "n_eval_groups": int(len(eval_bounds)),
        "n_features": int(x_train.shape[1]),
        "relevance": {
            "mode": args.relevance_mode,
            "max_label": int(args.max_label),
            "gap_scale": float(args.gap_scale),
            "label_min": int(relevance.min()) if len(relevance) else 0,
            "label_max": int(relevance.max()) if len(relevance) else 0,
        },
        "params": ranker.get_params(),
        "gammas": gammas,
        "topks": topks,
        "use_sample_weights": bool(args.use_sample_weights),
        "eval_rows": rows,
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
    parser.add_argument("--relevance-mode", choices=("gap", "rank"), default="gap")
    parser.add_argument("--max-label", type=int, default=31)
    parser.add_argument("--gap-scale", type=float, default=4.0)
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--reg-alpha", type=float, default=0.01)
    parser.add_argument("--reg-lambda", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--gammas", default="0,0.5,0.75,0.9,1.0,1.15,1.3,1.5")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
