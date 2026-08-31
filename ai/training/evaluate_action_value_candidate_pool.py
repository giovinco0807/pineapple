"""Evaluate multi-model candidate pools for action-value rerankers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.evaluate_action_value_reranker import (
    build_group_bounds,
    load_metadata,
    parse_topks,
    predict,
)
from ai.training.train_action_value_set_reranker import CandidateGroupDataset, collate_groups


def evaluate_pools(
    scores: np.ndarray,
    pred_scores: list[np.ndarray],
    bounds: list[tuple[int, int]],
    pool_ks: list[int],
    ev_loss_thresholds: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    thresholds = list(ev_loss_thresholds or [])
    hits = {k: 0 for k in pool_ks}
    regrets = {k: 0.0 for k in pool_ks}
    pool_sizes = {k: [] for k in pool_ks}
    ev_losses = {k: [] for k in pool_ks}
    rows: list[dict] = []

    for group_id, (start, end) in enumerate(bounds):
        idx = np.arange(start, end, dtype=np.int64)
        true = np.asarray(scores[idx], dtype=np.float64)
        true_best_local = int(np.argmax(true))
        true_best_global = int(idx[true_best_local])
        true_best_score = float(true[true_best_local])
        model_orders = [np.argsort(-np.asarray(pred[idx], dtype=np.float64)) for pred in pred_scores]
        row = {
            "group_id": int(group_id),
            "size": int(len(idx)),
            "teacher_best_index": true_best_global,
            "teacher_best_score": true_best_score,
            "dealt_ev": true_best_score,
            "pool_metrics": {},
        }
        for k in pool_ks:
            members: set[int] = set()
            for ordered in model_orders:
                for local_i in ordered[: min(k, len(idx))]:
                    members.add(int(idx[local_i]))
            pool = sorted(members)
            pool_sizes[k].append(len(pool))
            hit = true_best_global in members
            if hit:
                hits[k] += 1
            best_in_pool = max((float(scores[i]) for i in pool), default=float("-inf"))
            regret = true_best_score - best_in_pool
            regrets[k] += regret
            ev_losses[k].append(regret)
            row["pool_metrics"][str(k)] = {
                "pool_size": int(len(pool)),
                "hit": bool(hit),
                "regret": float(regret),
                "ev_loss": float(regret),
                "pool_best_ev": float(best_in_pool),
                "dealt_ev": true_best_score,
                "indices": pool,
            }
        rows.append(row)

    n_groups = max(len(bounds), 1)
    metrics = {
        "groups": int(len(bounds)),
        "samples": int(len(scores)),
        "checkpoints": int(len(pred_scores)),
    }
    for k in pool_ks:
        sizes = np.asarray(pool_sizes[k], dtype=np.float64)
        metrics[f"union_top{k}_recall"] = hits[k] / n_groups
        metrics[f"union_top{k}_rerank_regret"] = regrets[k] / n_groups
        metrics[f"union_top{k}_avg_pool_size"] = float(sizes.mean()) if len(sizes) else 0.0
        metrics[f"union_top{k}_max_pool_size"] = int(sizes.max()) if len(sizes) else 0
        losses = np.asarray(ev_losses[k], dtype=np.float64)
        metrics[f"union_top{k}_ev_loss_mean"] = float(losses.mean()) if len(losses) else 0.0
        metrics[f"union_top{k}_ev_loss_p95"] = float(np.percentile(losses, 95)) if len(losses) else 0.0
        metrics[f"union_top{k}_ev_loss_p99"] = float(np.percentile(losses, 99)) if len(losses) else 0.0
        metrics[f"union_top{k}_ev_loss_max"] = float(losses.max()) if len(losses) else 0.0
        for threshold in thresholds:
            key = str(threshold).replace(".", "p")
            metrics[f"union_top{k}_ev_loss_gt_{key}"] = int((losses > float(threshold)).sum()) if len(losses) else 0
    return metrics, rows


def write_summary_md(path: Path, summary: dict, pool_ks: list[int]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Action-Value Candidate Pool Evaluation",
        "",
        f"- label: `{summary['label']}`",
        f"- checkpoints: {len(summary['checkpoints'])}",
        f"- data: `{summary['data']}`",
        f"- groups: {metrics['groups']:,}",
        f"- samples: {metrics['samples']:,}",
        "- `dealt_ev` is the best exact EV available after the cards are dealt.",
        "- `rerank regret` is the EV lost if exact reranking is limited to that pool.",
        "",
        "## Union Pool Recall",
        "",
        "| per-model K | recall | mean EV loss | p99 EV loss | max EV loss | avg pool | max pool |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in pool_ks:
        lines.append(
            "| "
            f"{k} | "
            f"{metrics[f'union_top{k}_recall']:.1%} | "
            f"{metrics[f'union_top{k}_ev_loss_mean']:.3f} | "
            f"{metrics[f'union_top{k}_ev_loss_p99']:.3f} | "
            f"{metrics[f'union_top{k}_ev_loss_max']:.3f} | "
            f"{metrics[f'union_top{k}_avg_pool_size']:.1f} | "
            f"{metrics[f'union_top{k}_max_pool_size']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def predict_set_model(
    checkpoint: str | Path,
    data_dir: Path,
    bounds: list[tuple[int, int]],
    n_samples: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = ActionValueSetReranker.from_checkpoint(checkpoint, map_location=device).to(device)
    dataset = CandidateGroupDataset(data_dir, list(range(len(bounds))), bounds, model.score_mean, model.score_std)
    if int(dataset.input_dim) != int(model.input_dim):
        raise ValueError(f"Data input_dim={dataset.input_dim} does not match set model input_dim={model.input_dim}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_groups,
    )
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
    del model
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    if bool(np.isnan(pred_scores).any()):
        raise ValueError("Set model prediction did not cover every sample")
    return pred_scores


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    start_time = time.time()
    data_dir = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_ks = parse_topks(args.pool_ks)
    meta = load_metadata(data_dir)

    states = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    states = states[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    if args.max_groups > 0:
        bounds = bounds[: args.max_groups]
        n_samples = bounds[-1][1] if bounds else 0
        states = states[:n_samples]
        scores = scores[:n_samples]
        turns = turns[:n_samples]

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pred_scores: list[np.ndarray] = []
    for checkpoint in args.checkpoints:
        model = ActionValueReranker.from_checkpoint(checkpoint, map_location=device).to(device)
        pred_score, _, _ = predict(model, states, turns, device, args.batch_size)
        pred_scores.append(pred_score)
        del model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
    for checkpoint in args.set_checkpoints:
        pred_scores.append(predict_set_model(checkpoint, data_dir, bounds, n_samples, device, args.set_batch_size))
    if not pred_scores:
        raise ValueError("Provide --checkpoints and/or --set-checkpoints")

    thresholds = [float(part) for part in args.ev_loss_thresholds.split(",") if part.strip()]
    metrics, rows = evaluate_pools(scores, pred_scores, bounds, pool_ks, thresholds)
    summary = {
        "label": args.label,
        "checkpoints": [str(path) for path in args.checkpoints],
        "set_checkpoints": [str(path) for path in args.set_checkpoints],
        "data": str(data_dir),
        "pool_ks": pool_ks,
        "metrics": metrics,
        "elapsed_seconds": time.time() - start_time,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_summary_md(out_dir / "summary.md", summary, pool_ks)

    print("Action-value candidate pool evaluation")
    print(f"  label: {args.label}")
    print(f"  data: {data_dir}")
    print(
        f"  checkpoints={len(args.checkpoints)} set_checkpoints={len(args.set_checkpoints)} "
        f"groups={metrics['groups']:,} samples={metrics['samples']:,}"
    )
    for k in pool_ks:
        print(
            f"  union_top{k}={metrics[f'union_top{k}_recall']:.1%} "
            f"ev_loss={metrics[f'union_top{k}_ev_loss_mean']:.3f} "
            f"p99={metrics[f'union_top{k}_ev_loss_p99']:.3f} "
            f"max={metrics[f'union_top{k}_ev_loss_max']:.3f} "
            f"avg_pool={metrics[f'union_top{k}_avg_pool_size']:.1f} "
            f"max_pool={metrics[f'union_top{k}_max_pool_size']}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate union candidate pools from multiple reranker checkpoints")
    parser.add_argument("--data", required=True, help="Candidate-level reranker data directory")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--set-checkpoints", nargs="*", default=[], help="Set/listwise reranker checkpoint paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--label", default="pool")
    parser.add_argument("--pool-ks", default="1,3,5,10,15")
    parser.add_argument("--ev-loss-thresholds", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--set-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
