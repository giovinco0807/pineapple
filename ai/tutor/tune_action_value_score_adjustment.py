"""Tune simple score adjustments for an action-value reranker.

The base reranker score can over-rank candidates with misleading bust/FL head
outputs.  This diagnostic precomputes predictions once, then evaluates a small
grid of:

    adjusted = score - bust_weight * bust + fl_weight * fl
               + fl_type_weights dot fl_types

It is intended for local validation before adding any runtime score correction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.evaluate_action_value_dataset import FL_KEYS, group_ranges, load_metadata, predict
from ai.tutor.evaluate_teacher_model import resolve_model_path
from ai.tutor.hybrid_t1t2 import HybridConfig, build_shortlist


TOP_NS = (1, 3, 5, 10, 15)


@dataclass
class TuneStats:
    decisions: int = 0
    candidates: int = 0
    top_hits: dict[int, int] = field(default_factory=lambda: {k: 0 for k in TOP_NS})
    regret_sum: dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    pool_hits: int = 0
    pool_regret_sum: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.decisions, 1)
        return {
            "decisions": self.decisions,
            "candidates": self.candidates,
            "top_recall": {f"top_{k}": self.top_hits[k] / denom for k in TOP_NS},
            "avg_regret": {f"top_{k}": self.regret_sum[k] / denom for k in TOP_NS},
            "pool_recall": self.pool_hits / denom,
            "pool_avg_regret": self.pool_regret_sum / denom,
        }


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def adjusted_score(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    score = np.asarray(preds["score"], dtype=np.float32).copy()
    score -= float(weights["bust"]) * np.asarray(preds["bust"], dtype=np.float32)
    score += float(weights["fl"]) * np.asarray(preds["fl"], dtype=np.float32)
    type_weights = np.asarray(
        [weights["qq"], weights["kk"], weights["aa"], weights["trips"]],
        dtype=np.float32,
    )
    if np.any(type_weights):
        score += np.asarray(preds["fl_types"], dtype=np.float32) @ type_weights
    return score


def evaluate_weights(
    *,
    scores: np.ndarray,
    turns: np.ndarray,
    group_ids: np.ndarray,
    action_indices: np.ndarray,
    preds: dict[str, np.ndarray],
    weights: dict[str, float],
    config: HybridConfig,
) -> dict[str, Any]:
    adjusted = adjusted_score(preds, weights)
    stats = TuneStats()
    for start, end in group_ranges(group_ids):
        if end <= start:
            continue
        local_scores = scores[start:end]
        local_adjusted = adjusted[start:end]
        order_local = np.argsort(-local_adjusted)
        teacher_best_local = int(np.argmax(local_scores))
        teacher_best_score = float(scores[start + teacher_best_local])
        teacher_best_action = int(action_indices[start + teacher_best_local])
        stats.decisions += 1
        stats.candidates += int(end - start)
        rank_by_local = {int(local_idx): int(rank) + 1 for rank, local_idx in enumerate(order_local)}
        for k in TOP_NS:
            kept_local = [int(idx) for idx in order_local[: min(k, len(order_local))]]
            if teacher_best_action in {int(action_indices[start + idx]) for idx in kept_local}:
                stats.top_hits[k] += 1
            best_kept = max(float(scores[start + idx]) for idx in kept_local) if kept_local else float("-inf")
            stats.regret_sum[k] += max(0.0, teacher_best_score - best_kept)

        candidates: list[dict[str, Any]] = []
        for local_idx in range(end - start):
            absolute_idx = start + local_idx
            candidates.append(
                {
                    "action_idx": int(action_indices[absolute_idx]),
                    "model_score": float(adjusted[absolute_idx]),
                    "model_rank": rank_by_local[local_idx],
                    "predicted_bust": float(preds["bust"][absolute_idx]),
                    "predicted_fl": float(preds["fl"][absolute_idx]),
                    "predicted_fl_types": {
                        key: float(preds["fl_types"][absolute_idx, i]) for i, key in enumerate(FL_KEYS)
                    },
                }
            )
        pool = build_shortlist(candidates, config=config)
        pool_actions = {int(item["action_idx"]) for item in pool}
        if teacher_best_action in pool_actions:
            stats.pool_hits += 1
        best_pool_score = max(
            (
                float(scores[start + idx])
                for idx in range(end - start)
                if int(action_indices[start + idx]) in pool_actions
            ),
            default=float("-inf"),
        )
        stats.pool_regret_sum += max(0.0, teacher_best_score - best_pool_score)
    return {"weights": weights, "stats": stats.as_dict()}


def weight_grid(args: argparse.Namespace) -> Iterable[dict[str, float]]:
    keys = ("bust", "fl", "qq", "kk", "aa", "trips")
    values = {
        "bust": parse_float_list(args.bust_weights),
        "fl": parse_float_list(args.fl_weights),
        "qq": parse_float_list(args.fl_qq_weights),
        "kk": parse_float_list(args.fl_kk_weights),
        "aa": parse_float_list(args.fl_aa_weights),
        "trips": parse_float_list(args.fl_trips_weights),
    }
    for combo in itertools.product(*(values[key] for key in keys)):
        yield {key: float(value) for key, value in zip(keys, combo)}


def sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    stats = item["stats"]
    return (
        -float(stats["top_recall"]["top_3"]),
        float(stats["avg_regret"]["top_3"]),
        -float(stats["top_recall"]["top_1"]),
        float(stats["avg_regret"]["top_1"]),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(data_dir)
    states_mm = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(metadata.get("n_samples", states_mm.shape[0])), states_mm.shape[0])
    if args.limit_samples:
        n_samples = min(n_samples, int(args.limit_samples))
    states = states_mm[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int16)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    action_indices = np.asarray(np.load(data_dir / "action_indices.npy", mmap_mode="r")[:n_samples], dtype=np.int64)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = resolve_model_path(args.model)
    model = ActionValueReranker.from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    preds = predict(model, states, turns, device, args.batch_size)
    config = HybridConfig(
        shortlist_k=args.shortlist_k,
        insurance_k=args.insurance_k,
        sync_exact_k=args.sync_exact_k,
        max_pool=args.max_pool,
        enable_sync_refinement=False,
    )
    results = [
        evaluate_weights(
            scores=scores,
            turns=turns,
            group_ids=group_ids,
            action_indices=action_indices,
            preds=preds,
            weights=weights,
            config=config,
        )
        for weights in weight_grid(args)
    ]
    results.sort(key=sort_key)
    report = {
        "data": str(data_dir),
        "model": str(model_path),
        "device": str(device),
        "n_samples": int(n_samples),
        "metadata": metadata,
        "shortlist_k": int(args.shortlist_k),
        "insurance_k": int(args.insurance_k),
        "max_pool": int(args.max_pool),
        "results": results,
        "best_by_top3": results[: min(len(results), int(args.report_top))],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Action-value Score Adjustment Tune",
        "",
        "| rank | bust | fl | qq | kk | aa | trips | top1 | top3 | top5 | top15 | pool | top1 regret | top3 regret |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(report["best_by_top3"], start=1):
        weights = item["weights"]
        stats = item["stats"]
        top = stats["top_recall"]
        regret = stats["avg_regret"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"{weights['bust']:.3f}",
                    f"{weights['fl']:.3f}",
                    f"{weights['qq']:.3f}",
                    f"{weights['kk']:.3f}",
                    f"{weights['aa']:.3f}",
                    f"{weights['trips']:.3f}",
                    f"{top['top_1']:.1%}",
                    f"{top['top_3']:.1%}",
                    f"{top['top_5']:.1%}",
                    f"{top['top_15']:.1%}",
                    f"{stats['pool_recall']:.1%}",
                    f"{regret['top_1']:.3f}",
                    f"{regret['top_3']:.3f}",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune simple reranker score adjustments")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--shortlist-k", type=int, default=15)
    parser.add_argument("--insurance-k", type=int, default=5)
    parser.add_argument("--sync-exact-k", type=int, default=3)
    parser.add_argument("--max-pool", type=int, default=20)
    parser.add_argument("--bust-weights", default="0")
    parser.add_argument("--fl-weights", default="0")
    parser.add_argument("--fl-qq-weights", default="0")
    parser.add_argument("--fl-kk-weights", default="0")
    parser.add_argument("--fl-aa-weights", default="0")
    parser.add_argument("--fl-trips-weights", default="0")
    parser.add_argument("--report-top", type=int, default=20)
    args = parser.parse_args(list(argv) if argv is not None else None)
    evaluate(args)


if __name__ == "__main__":
    main()
