"""Analyze action-value reranker Top-K misses by EV gap and regret."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker


def parse_topks(value: str) -> list[int]:
    return sorted({int(part) for part in value.split(",") if part.strip()})


def parse_epsilons(value: str) -> list[float]:
    return sorted({float(part) for part in value.split(",") if part.strip()})


def parse_position(value: str) -> int | None:
    text = value.strip().lower()
    if not text or text in {"all", "mixed"}:
        return None
    if text in {"btn", "button", "second"}:
        return 1
    if text in {"bb", "blind", "first"}:
        return 0
    raise ValueError("--position must be all, btn, or bb")


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_groups(group_ids: np.ndarray, val_frac: float, seed: int) -> set[int]:
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(len(unique_groups) * val_frac))
    return {int(x) for x in unique_groups[:n_val_groups]}


def build_group_bounds(group_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(group_ids) == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(group_ids)):
        if int(group_ids[i]) != int(group_ids[i - 1]):
            bounds.append((start, i))
            start = i
    bounds.append((start, len(group_ids)))
    return bounds


@torch.no_grad()
def predict_scores(
    checkpoint: Path,
    states: np.ndarray,
    turns: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = ActionValueReranker.from_checkpoint(checkpoint, map_location=device).to(device)
    model.eval()
    pred = np.zeros(int(states.shape[0]), dtype=np.float32)
    for start in range(0, int(states.shape[0]), batch_size):
        end = min(int(states.shape[0]), start + batch_size)
        state = torch.from_numpy(np.array(states[start:end], dtype=np.float32, copy=True)).to(device)
        turn = torch.from_numpy(np.asarray(turns[start:end], dtype=np.int64)).to(device)
        out = model.predict_components(state, turn=turn)
        pred[start:end] = out["score"].detach().cpu().numpy()
    return pred


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def summarize(
    scores: np.ndarray,
    pred: np.ndarray,
    group_ids: np.ndarray,
    positions: np.ndarray | None,
    selected_position: int | None,
    val_groups: set[int] | None,
    topks: list[int],
    epsilons: list[float],
    gap_bins: list[float],
) -> dict:
    bounds = build_group_bounds(group_ids)
    top_hits = {k: 0 for k in topks}
    eps_hits = {eps: 0 for eps in epsilons}
    top_regrets = {k: [] for k in topks}
    top1_regrets: list[float] = []
    top3_miss_regrets: list[float] = []
    best_second_gaps: list[float] = []
    teacher_ranks: list[int] = []
    gap_stats = {
        str(limit): {
            "groups": 0,
            "top1_hits": 0,
            "top3_hits": 0,
            "top1_regret_sum": 0.0,
        }
        for limit in gap_bins
    }
    gap_stats[f">{gap_bins[-1]}"] = {
        "groups": 0,
        "top1_hits": 0,
        "top3_hits": 0,
        "top1_regret_sum": 0.0,
    }

    groups = 0
    for start, end in bounds:
        group = int(group_ids[start])
        if val_groups is not None and group not in val_groups:
            continue
        idx = np.arange(start, end, dtype=np.int64)
        if selected_position is not None:
            if positions is None:
                raise FileNotFoundError("positions.npy is required for --position")
            idx = idx[np.asarray(positions[idx], dtype=np.int8) == int(selected_position)]
        if len(idx) == 0:
            continue

        true = np.asarray(scores[idx], dtype=np.float64)
        predicted = np.asarray(pred[idx], dtype=np.float64)
        true_order = np.argsort(-true)
        pred_order = np.argsort(-predicted)
        best_local = int(true_order[0])
        pred_best_local = int(pred_order[0])
        rank = int(np.where(pred_order == best_local)[0][0]) + 1
        teacher_ranks.append(rank)
        groups += 1

        best_score = float(true[best_local])
        pred_best_score = float(true[pred_best_local])
        regret = max(0.0, best_score - pred_best_score)
        top1_regrets.append(regret)
        second_score = float(true[true_order[1]]) if len(true_order) > 1 else best_score
        best_second_gap = max(0.0, best_score - second_score)
        best_second_gaps.append(best_second_gap)

        for k in topks:
            kk = min(k, len(pred_order))
            kept = pred_order[:kk]
            if rank <= kk:
                top_hits[k] += 1
            top_regrets[k].append(max(0.0, best_score - float(true[kept].max())))
        if rank > 3:
            top3_miss_regrets.append(max(0.0, best_score - float(true[pred_order[: min(3, len(pred_order))]].max())))
        for eps in epsilons:
            if regret <= eps:
                eps_hits[eps] += 1

        key = f">{gap_bins[-1]}"
        for limit in gap_bins:
            if best_second_gap <= limit:
                key = str(limit)
                break
        gap_stats[key]["groups"] += 1
        gap_stats[key]["top1_hits"] += int(rank <= 1)
        gap_stats[key]["top3_hits"] += int(rank <= 3)
        gap_stats[key]["top1_regret_sum"] += regret

    denom = max(groups, 1)
    gap_out = {}
    for key, item in gap_stats.items():
        n = int(item["groups"])
        gap_out[key] = {
            "groups": n,
            "top1": float(item["top1_hits"] / n) if n else 0.0,
            "top3": float(item["top3_hits"] / n) if n else 0.0,
            "top1_regret": float(item["top1_regret_sum"] / n) if n else 0.0,
        }

    return {
        "groups": groups,
        "topk": {
            str(k): {
                "recall": float(top_hits[k] / denom),
                "avg_regret": float(np.mean(top_regrets[k])) if top_regrets[k] else 0.0,
                "p95_regret": percentile(top_regrets[k], 95),
                "p99_regret": percentile(top_regrets[k], 99),
            }
            for k in topks
        },
        "top1_epsilon_hit": {str(eps): float(eps_hits[eps] / denom) for eps in epsilons},
        "teacher_rank": {
            "mean": float(np.mean(teacher_ranks)) if teacher_ranks else 0.0,
            "p50": percentile([float(x) for x in teacher_ranks], 50),
            "p90": percentile([float(x) for x in teacher_ranks], 90),
            "p95": percentile([float(x) for x in teacher_ranks], 95),
            "p99": percentile([float(x) for x in teacher_ranks], 99),
        },
        "top1_regret": {
            "mean": float(np.mean(top1_regrets)) if top1_regrets else 0.0,
            "p50": percentile(top1_regrets, 50),
            "p90": percentile(top1_regrets, 90),
            "p95": percentile(top1_regrets, 95),
            "p99": percentile(top1_regrets, 99),
        },
        "top3_miss_regret": {
            "count": len(top3_miss_regrets),
            "mean": float(np.mean(top3_miss_regrets)) if top3_miss_regrets else 0.0,
            "p50": percentile(top3_miss_regrets, 50),
            "p90": percentile(top3_miss_regrets, 90),
            "p95": percentile(top3_miss_regrets, 95),
            "p99": percentile(top3_miss_regrets, 99),
        },
        "best_second_gap": {
            "mean": float(np.mean(best_second_gaps)) if best_second_gaps else 0.0,
            "p50": percentile(best_second_gaps, 50),
            "p90": percentile(best_second_gaps, 90),
            "p95": percentile(best_second_gaps, 95),
            "p99": percentile(best_second_gaps, 99),
        },
        "by_best_second_gap": gap_out,
    }


def analyze(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    meta = load_metadata(data_dir)
    n_samples = int(meta.get("n_samples", np.load(data_dir / "scores.npy", mmap_mode="r").shape[0]))
    states = np.load(data_dir / "states.npy", mmap_mode="r")[:n_samples]
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    positions_path = data_dir / "positions.npy"
    positions = (
        np.asarray(np.load(positions_path, mmap_mode="r")[:n_samples], dtype=np.int8)
        if positions_path.exists()
        else None
    )
    selected_position = parse_position(args.position)
    val_groups = (
        split_groups(group_ids, args.val_frac, args.seed)
        if args.validation_only
        else None
    )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pred = predict_scores(Path(args.checkpoint), states, turns, args.batch_size, device)
    summary = summarize(
        scores,
        pred,
        group_ids,
        positions,
        selected_position,
        val_groups,
        parse_topks(args.topks),
        parse_epsilons(args.epsilons),
        parse_epsilons(args.gap_bins),
    )
    summary.update(
        {
            "data": str(data_dir),
            "checkpoint": str(args.checkpoint),
            "position": args.position,
            "validation_only": bool(args.validation_only),
            "seed": int(args.seed),
            "val_frac": float(args.val_frac),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze Top-K miss distribution for action-value reranker data")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--position", default="all")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--epsilons", default="0,0.01,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--gap-bins", default="0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
