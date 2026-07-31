"""Evaluate a reranker by averaging predictions across suit-permutation blocks.

Input is a candidate-level dataset where each original decision has been expanded
into a fixed-size block of suit permutations, typically 24 records produced by
`augment_teacher_suits.py` and `convert_action_value_teacher.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.evaluate_action_value_dataset import (
    Bucket,
    FL_KEYS,
    group_ranges,
    load_metadata,
    parse_top_ns,
    predict,
    write_markdown,
)
from ai.tutor.evaluate_teacher_model import resolve_model_path


def _slice(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.asarray(arr[start:end])


def _stack_block(values: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([_slice(values, start, end) for start, end in ranges], axis=0)


def _check_block(candidate_ranks: np.ndarray, ranges: list[tuple[int, int]]) -> None:
    lengths = [end - start for start, end in ranges]
    if len(set(lengths)) != 1:
        raise ValueError(f"Suit block has mismatched candidate counts: {lengths}")
    first = _slice(candidate_ranks, *ranges[0])
    for start, end in ranges[1:]:
        ranks = _slice(candidate_ranks, start, end)
        if not np.array_equal(first, ranks):
            raise ValueError("Suit block candidate order differs across permutations")


def spread_indices(n_items: int, n_selected: int) -> list[int]:
    n_items = max(0, int(n_items))
    n_selected = max(0, min(int(n_selected), n_items))
    if n_selected <= 0:
        return []
    if n_selected == 1:
        return [0]
    step = (n_items - 1) / float(n_selected - 1)
    indices = [int(round(i * step)) for i in range(n_selected)]
    seen: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.append(idx)
    for idx in range(n_items):
        if len(seen) >= n_selected:
            break
        if idx not in seen:
            seen.append(idx)
    return seen[:n_selected]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(data_dir)
    top_ns = parse_top_ns(args.top_ns)

    states_mm = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(metadata.get("n_samples", states_mm.shape[0])), states_mm.shape[0])
    if args.limit_samples:
        n_samples = min(n_samples, int(args.limit_samples))

    states = states_mm[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    fl = np.asarray(np.load(data_dir / "fl.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    bust = np.asarray(np.load(data_dir / "bust.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int16)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    candidate_ranks = np.asarray(
        np.load(data_dir / "candidate_ranks.npy", mmap_mode="r")[:n_samples],
        dtype=np.int16,
    )
    fl_types = (
        np.asarray(np.load(data_dir / "fl_types.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
        if (data_dir / "fl_types.npy").exists()
        else np.zeros((n_samples, len(FL_KEYS)), dtype=np.float32)
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = resolve_model_path(args.model)
    model = ActionValueReranker.from_checkpoint(model_path, map_location=device)
    model.to(device)
    preds = predict(model, states, turns, device, args.batch_size)

    ranges_all = group_ranges(group_ids)
    if len(ranges_all) % args.block_size != 0:
        raise ValueError(
            f"Group count {len(ranges_all)} is not divisible by block size {args.block_size}"
        )
    ensemble_offsets = spread_indices(args.block_size, args.ensemble_size)

    overall = Bucket(top_ns=top_ns)
    by_turn: dict[int, Bucket] = {}
    weak: list[dict[str, Any]] = []

    for block_index in range(0, len(ranges_all), args.block_size):
        block = ranges_all[block_index : block_index + args.block_size]
        selected_block = [block[idx] for idx in ensemble_offsets]
        _check_block(candidate_ranks, selected_block)

        true = _stack_block(scores, selected_block).mean(axis=0)
        pred = _stack_block(preds["score"], selected_block).mean(axis=0)
        pred_fl = _stack_block(preds["fl"], selected_block).mean(axis=0)
        pred_bust = _stack_block(preds["bust"], selected_block).mean(axis=0)
        pred_fl_types = _stack_block(preds["fl_types"], selected_block).mean(axis=0)
        true_fl = _stack_block(fl, selected_block).mean(axis=0)
        true_bust = _stack_block(bust, selected_block).mean(axis=0)
        true_fl_types = _stack_block(fl_types, selected_block).mean(axis=0)

        order = np.argsort(-pred)
        teacher_best = int(np.argmax(true))
        pred_best = int(order[0])
        rank = int(np.where(order == teacher_best)[0][0]) + 1
        regret = float(true[teacher_best] - true[pred_best])
        topk_regrets: dict[int, float] = {}
        for k in top_ns:
            kept = order[: min(k, len(order))]
            best_kept = float(np.max(true[kept])) if len(kept) else float(true[pred_best])
            topk_regrets[k] = float(true[teacher_best] - best_kept)

        turn = int(turns[block[0][0]])
        n = int(block[0][1] - block[0][0])
        score_abs = float(np.abs(pred - true).sum())
        fl_abs = float(np.abs(pred_fl - true_fl).sum())
        bust_abs = float(np.abs(pred_bust - true_bust).sum())
        fl_type_abs = {
            key: float(np.abs(pred_fl_types[:, i] - true_fl_types[:, i]).sum())
            for i, key in enumerate(FL_KEYS)
        }

        overall.add_decision(rank, regret, n, topk_regrets)
        overall.add_candidate_errors(score_abs, fl_abs, bust_abs, fl_type_abs)
        bucket = by_turn.get(turn)
        if bucket is None:
            bucket = Bucket(top_ns=top_ns)
            by_turn[turn] = bucket
        bucket.add_decision(rank, regret, n, topk_regrets)
        bucket.add_candidate_errors(score_abs, fl_abs, bust_abs, fl_type_abs)

        variant_ranks = []
        for start, end in selected_block:
            variant_pred = preds["score"][start:end]
            variant_order = np.argsort(-variant_pred)
            variant_ranks.append(int(np.where(variant_order == teacher_best)[0][0]) + 1)

        weak.append(
            {
                "block_id": int(block_index // args.block_size),
                "group_ids": [int(group_ids[start]) for start, _end in selected_block],
                "turn": turn,
                "rank": rank,
                "regret": regret,
                "n_candidates": n,
                "teacher_score": float(true[teacher_best]),
                "predicted_choice_teacher_score": float(true[pred_best]),
                "predicted_score": float(pred[pred_best]),
                "topk_exact_rerank_regret": {str(k): topk_regrets[k] for k in top_ns},
                "variant_rank_min": int(min(variant_ranks)),
                "variant_rank_max": int(max(variant_ranks)),
                "variant_rank_mean": float(np.mean(variant_ranks)),
            }
        )

    weak.sort(key=lambda item: (item["regret"], item["rank"]), reverse=True)
    weak_path = output_dir / "weak_groups.jsonl"
    with weak_path.open("w", encoding="utf-8") as f:
        for item in weak[: args.weak_groups]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "data": str(data_dir),
        "model": str(model_path),
        "device": str(device),
        "mode": "suit_block_ensemble",
        "block_size": int(args.block_size),
        "ensemble_size": int(len(ensemble_offsets)),
        "ensemble_offsets": [int(i) for i in ensemble_offsets],
        "blocks": int(len(ranges_all) // args.block_size),
        "n_samples": int(n_samples),
        "metadata": metadata,
        "top_ns": list(top_ns),
        "overall": overall.as_dict(),
        "by_turn": {str(turn): bucket.as_dict() for turn, bucket in sorted(by_turn.items())},
        "weak_groups_path": str(weak_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "summary.md", report)
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate suit-permutation prediction ensemble blocks")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--weak-groups", type=int, default=100)
    parser.add_argument("--top-ns", default="1,3,5,10,20,24")
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--ensemble-size", type=int, default=24)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
