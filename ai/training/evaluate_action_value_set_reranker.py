"""Evaluate a set/listwise action-value reranker on candidate groups."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.train_action_value_set_reranker import (
    CandidateGroupDataset,
    build_group_bounds,
    collate_groups,
    load_metadata,
)
from ai.training.evaluate_action_value_reranker import parse_topks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xx = x.astype(np.float64, copy=True)
    yy = y.astype(np.float64, copy=True)
    xx -= xx.mean()
    yy -= yy.mean()
    denom = math.sqrt(float((xx * xx).sum() * (yy * yy).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((xx * yy).sum() / denom)


@torch.no_grad()
def predict_groups(
    model: ActionValueSetReranker,
    loader: DataLoader,
    bounds: list[tuple[int, int]],
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    pred_scores = np.full(n_samples, np.nan, dtype=np.float32)
    model.eval()
    for batch in loader:
        states = batch["states"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        group_index = batch["group_index"].detach().cpu().numpy()
        pred = model.predict_scores(states, mask).detach().cpu().numpy()
        valid = mask.detach().cpu().numpy().astype(bool)
        for row, group_i in enumerate(group_index):
            start, end = bounds[int(group_i)]
            length = end - start
            pred_scores[start:end] = pred[row, :length][valid[row, :length]]
    return pred_scores


def summarize_groups(
    scores: np.ndarray,
    pred_score: np.ndarray,
    bounds: list[tuple[int, int]],
    topks: list[int],
) -> tuple[dict, list[dict]]:
    hits = {k: 0 for k in topks}
    rerank_regret = {k: 0.0 for k in topks}
    pred_top1_regret = 0.0
    teacher_ranks: list[int] = []
    rows: list[dict] = []

    for group_id, (start, end) in enumerate(bounds):
        idx = np.arange(start, end, dtype=np.int64)
        true = np.asarray(scores[idx], dtype=np.float64)
        pred = np.asarray(pred_score[idx], dtype=np.float64)
        if len(idx) == 0 or bool(np.isnan(pred).any()):
            continue
        true_best_local = int(np.argmax(true))
        ordered = np.argsort(-pred)
        pred_best_local = int(ordered[0])
        true_rank = int(np.where(ordered == true_best_local)[0][0]) + 1
        teacher_ranks.append(true_rank)
        true_best_score = float(true[true_best_local])
        pred_top1_regret += true_best_score - float(true[pred_best_local])
        for k in topks:
            kk = min(k, len(idx))
            if true_rank <= kk:
                hits[k] += 1
            rerank_regret[k] += true_best_score - float(true[ordered[:kk]].max())
        rows.append(
            {
                "group_id": int(group_id),
                "size": int(len(idx)),
                "teacher_rank_by_model": int(true_rank),
                "teacher_best_index": int(idx[true_best_local]),
                "model_best_index": int(idx[pred_best_local]),
                "teacher_best_score": true_best_score,
                "dealt_ev": true_best_score,
                "model_best_true_score": float(true[pred_best_local]),
                "model_best_pred_score": float(pred[pred_best_local]),
                "regret": true_best_score - float(true[pred_best_local]),
                "ev_loss": true_best_score - float(true[pred_best_local]),
                "ordered_indices": [int(idx[i]) for i in ordered],
            }
        )

    n_groups = max(len(rows), 1)
    valid_pred = pred_score[~np.isnan(pred_score)]
    valid_scores = scores[~np.isnan(pred_score)]
    rank_arr = np.asarray(teacher_ranks, dtype=np.float64)
    metrics = {
        "groups": int(len(rows)),
        "samples": int(len(valid_scores)),
        "score_mae": float(np.mean(np.abs(valid_pred - valid_scores))) if len(valid_scores) else 0.0,
        "score_corr": pearson_corr(np.asarray(valid_pred), np.asarray(valid_scores)),
        "group_top1_regret": pred_top1_regret / n_groups,
        "teacher_rank_mean": float(rank_arr.mean()) if len(rank_arr) else 0.0,
        "teacher_rank_p95": float(np.percentile(rank_arr, 95)) if len(rank_arr) else 0.0,
    }
    for k in topks:
        metrics[f"group_top{k}"] = hits[k] / n_groups
        metrics[f"group_top{k}_rerank_regret"] = rerank_regret[k] / n_groups
    return metrics, rows


def write_summary_md(path: Path, summary: dict, topks: list[int]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Action-Value Set Reranker Evaluation",
        "",
        f"- label: `{summary['label']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- data: `{summary['data']}`",
        f"- groups: {metrics['groups']:,}",
        f"- samples: {metrics['samples']:,}",
        "- `dealt_ev` is the best exact EV available after the cards are dealt.",
        "- `rerank regret` is the EV lost by using the model ranking at that K.",
        f"- score MAE/corr: {metrics['score_mae']:.3f} / {metrics['score_corr']:.3f}",
        f"- teacher rank mean/p95: {metrics['teacher_rank_mean']:.2f} / {metrics['teacher_rank_p95']:.1f}",
        "",
        "## Group Recall",
        "",
        "| K | recall | rerank regret |",
        "|---:|---:|---:|",
    ]
    for k in topks:
        lines.append(
            f"| {k} | {metrics[f'group_top{k}']:.1%} | {metrics[f'group_top{k}_rerank_regret']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    start_time = time.time()
    data_dir = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    meta = load_metadata(data_dir)

    states = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    if args.max_groups > 0:
        bounds = bounds[: args.max_groups]
        n_samples = bounds[-1][1] if bounds else 0
        scores = scores[:n_samples]

    group_indices = list(range(len(bounds)))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ActionValueSetReranker.from_checkpoint(args.checkpoint, map_location=device).to(device)
    dataset = CandidateGroupDataset(data_dir, group_indices, bounds, model.score_mean, model.score_std)
    if int(dataset.input_dim) != int(model.input_dim):
        raise ValueError(f"Data input_dim={dataset.input_dim} does not match model input_dim={model.input_dim}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_groups,
    )
    pred_score = predict_groups(model, loader, bounds, n_samples, device)
    metrics, rows = summarize_groups(scores, pred_score, bounds, topks)

    summary = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "data": str(data_dir),
        "topks": topks,
        "metrics": metrics,
        "elapsed_seconds": time.time() - start_time,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_summary_md(out_dir / "summary.md", summary, topks)

    print("Action-value set reranker evaluation")
    print(f"  label: {args.label}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  data: {data_dir}")
    print(f"  groups={metrics['groups']:,} samples={metrics['samples']:,}")
    print(f"  score_mae={metrics['score_mae']:.3f} corr={metrics['score_corr']:.3f}")
    for k in topks:
        print(
            f"  top{k}={metrics[f'group_top{k}']:.1%} "
            f"top{k}_regret={metrics[f'group_top{k}_rerank_regret']:.3f}"
        )
    print(f"  top1_regret={metrics['group_top1_regret']:.3f}")
    print(f"  summary={out_dir / 'summary.md'}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a set/listwise action-value reranker checkpoint")
    parser.add_argument("--data", required=True, help="Candidate-level reranker data directory")
    parser.add_argument("--checkpoint", required=True, help="Set reranker checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--label", default="set-model")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
