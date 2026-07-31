"""Evaluate linear score blends of two action-value reranker checkpoints.

This is a local diagnostic for deciding whether a hard-mined fine-tune should
replace the current model, be blended with it, or be rejected.  It evaluates
candidate ranking and the normal Top15+insurance pool on the same dataset.
"""
from __future__ import annotations

import argparse
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
class BlendStats:
    decisions: int = 0
    candidates: int = 0
    top_hits: dict[int, int] = field(default_factory=lambda: {k: 0 for k in TOP_NS})
    top_regret_sum: dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    top_regret_max: dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    pool_hits: int = 0
    pool_regret_sum: float = 0.0
    pool_regret_max: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.decisions, 1)
        return {
            "decisions": self.decisions,
            "candidates": self.candidates,
            "avg_candidates": self.candidates / denom,
            "top_recall": {f"top_{k}": self.top_hits[k] / denom for k in TOP_NS},
            "topk_exact_rerank_avg_regret": {
                f"top_{k}": self.top_regret_sum[k] / denom for k in TOP_NS
            },
            "topk_exact_rerank_max_regret": {
                f"top_{k}": self.top_regret_max[k] for k in TOP_NS
            },
            "hybrid_pool_recall": self.pool_hits / denom,
            "hybrid_pool_avg_regret": self.pool_regret_sum / denom,
            "hybrid_pool_max_regret": self.pool_regret_max,
        }


def parse_weights(value: str) -> list[float]:
    weights = [float(part) for part in value.split(",") if part.strip()]
    if not weights:
        raise ValueError("At least one blend weight is required")
    return weights


def load_model(path: str, device: torch.device) -> ActionValueReranker:
    model = ActionValueReranker.from_checkpoint(resolve_model_path(path), map_location=device)
    model.to(device)
    model.eval()
    return model


def blend_array(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    return (1.0 - weight_b) * np.asarray(a, dtype=np.float32) + weight_b * np.asarray(b, dtype=np.float32)


def evaluate_blend(
    *,
    scores: np.ndarray,
    turns: np.ndarray,
    group_ids: np.ndarray,
    action_indices: np.ndarray,
    pred_a: dict[str, np.ndarray],
    pred_b: dict[str, np.ndarray],
    weight_b: float,
    config: HybridConfig,
) -> dict[str, Any]:
    pred_score = blend_array(pred_a["score"], pred_b["score"], weight_b)
    pred_bust = blend_array(pred_a["bust"], pred_b["bust"], weight_b)
    pred_fl = blend_array(pred_a["fl"], pred_b["fl"], weight_b)
    pred_fl_types = blend_array(pred_a["fl_types"], pred_b["fl_types"], weight_b)
    stats = BlendStats()

    for start, end in group_ranges(group_ids):
        if end <= start:
            continue
        local_scores = scores[start:end]
        local_pred = pred_score[start:end]
        order_local = np.argsort(-local_pred)
        teacher_best_local = int(np.argmax(local_scores))
        teacher_best_action = int(action_indices[start + teacher_best_local])
        teacher_best_score = float(scores[start + teacher_best_local])

        stats.decisions += 1
        stats.candidates += int(end - start)
        rank_by_local = {int(local_idx): int(rank) + 1 for rank, local_idx in enumerate(order_local)}

        for k in TOP_NS:
            kept_local = [int(idx) for idx in order_local[: min(k, len(order_local))]]
            kept_actions = {int(action_indices[start + idx]) for idx in kept_local}
            if teacher_best_action in kept_actions:
                stats.top_hits[k] += 1
            best_kept = max(float(scores[start + idx]) for idx in kept_local) if kept_local else float("-inf")
            regret = max(0.0, teacher_best_score - best_kept)
            stats.top_regret_sum[k] += regret
            stats.top_regret_max[k] = max(stats.top_regret_max[k], regret)

        candidates: list[dict[str, Any]] = []
        for local_idx in range(end - start):
            absolute_idx = start + local_idx
            candidates.append(
                {
                    "action_idx": int(action_indices[absolute_idx]),
                    "model_score": float(pred_score[absolute_idx]),
                    "model_rank": rank_by_local[local_idx],
                    "predicted_bust": float(pred_bust[absolute_idx]),
                    "predicted_fl": float(pred_fl[absolute_idx]),
                    "predicted_fl_types": {
                        key: float(pred_fl_types[absolute_idx, i]) for i, key in enumerate(FL_KEYS)
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
        pool_regret = max(0.0, teacher_best_score - best_pool_score)
        stats.pool_regret_sum += pool_regret
        stats.pool_regret_max = max(stats.pool_regret_max, pool_regret)

    return {
        "weight_b": float(weight_b),
        "model_a_weight": float(1.0 - weight_b),
        "model_b_weight": float(weight_b),
        "stats": stats.as_dict(),
    }


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
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)
    pred_a = predict(model_a, states, turns, device, args.batch_size)
    pred_b = predict(model_b, states, turns, device, args.batch_size)
    config = HybridConfig(
        shortlist_k=args.shortlist_k,
        insurance_k=args.insurance_k,
        sync_exact_k=args.sync_exact_k,
        max_pool=args.max_pool,
        enable_sync_refinement=False,
    )

    results = [
        evaluate_blend(
            scores=scores,
            turns=turns,
            group_ids=group_ids,
            action_indices=action_indices,
            pred_a=pred_a,
            pred_b=pred_b,
            weight_b=weight,
            config=config,
        )
        for weight in parse_weights(args.weights)
    ]
    report = {
        "data": str(data_dir),
        "model_a": str(resolve_model_path(args.model_a)),
        "model_b": str(resolve_model_path(args.model_b)),
        "device": str(device),
        "n_samples": int(n_samples),
        "metadata": metadata,
        "shortlist_k": int(args.shortlist_k),
        "insurance_k": int(args.insurance_k),
        "max_pool": int(args.max_pool),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Action-value Score Blend",
        "",
        "| weight_b | top1 | top3 | top5 | top10 | top15 | pool | top1 regret | top3 regret | pool regret |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        stats = item["stats"]
        top = stats["top_recall"]
        regrets = stats["topk_exact_rerank_avg_regret"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{item['weight_b']:.3f}",
                    f"{top['top_1']:.1%}",
                    f"{top['top_3']:.1%}",
                    f"{top['top_5']:.1%}",
                    f"{top['top_10']:.1%}",
                    f"{top['top_15']:.1%}",
                    f"{stats['hybrid_pool_recall']:.1%}",
                    f"{regrets['top_1']:.3f}",
                    f"{regrets['top_3']:.3f}",
                    f"{stats['hybrid_pool_avg_regret']:.3f}",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate score blends of two reranker checkpoints")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--shortlist-k", type=int, default=15)
    parser.add_argument("--insurance-k", type=int, default=5)
    parser.add_argument("--sync-exact-k", type=int, default=3)
    parser.add_argument("--max-pool", type=int, default=20)
    args = parser.parse_args(list(argv) if argv is not None else None)
    evaluate(args)


if __name__ == "__main__":
    main()
