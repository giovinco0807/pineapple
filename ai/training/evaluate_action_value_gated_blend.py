"""Evaluate a conditional specialist blend for action-value rerankers.

This is for offline validation before changing a runtime ensemble.  It keeps the
base ensemble unchanged on most groups, and mixes a specialist checkpoint only
for groups that match a simple board/dealt feature gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import (
    build_group_bounds,
    load_metadata,
    load_source_records,
    parse_topks,
    parse_weights,
    predict_ensemble,
    summarize_groups,
)
from ai.models.action_value_reranker import encoded_state_gate_mask


def card_rank(card: str) -> str:
    if not card:
        return ""
    if card.startswith("X"):
        return "X"
    return card[0]


def has_joker(cards: Iterable[str]) -> bool:
    return any(str(card).startswith("X") for card in cards or [])


def has_pair_or_joker(cards: Iterable[str]) -> bool:
    cards = list(cards or [])
    if has_joker(cards):
        return True
    ranks = [card_rank(card) for card in cards]
    return len(ranks) != len(set(ranks))


def visible_cards(record: dict) -> list[str]:
    cards: list[str] = []
    for board_key in ("board", "opponent_board"):
        board = record.get(board_key) or {}
        for row in ("top", "middle", "bottom"):
            cards.extend(board.get(row) or [])
    return cards


def board_features(record: dict) -> dict:
    board = record.get("board") or {}
    opp = record.get("opponent_board") or {}
    top = list(board.get("top") or [])
    opp_top = list(opp.get("top") or [])
    dealt = list(record.get("dealt") or [])
    visible = visible_cards(record)
    return {
        "own_top_len": len(top),
        "opp_top_len": len(opp_top),
        "own_top_has_pair_or_joker": has_pair_or_joker(top),
        "opp_top_has_pair_or_joker": has_pair_or_joker(opp_top),
        "visible_joker": has_joker(visible),
        "dealt_joker": has_joker(dealt),
        "own_top_has_ace": any(card_rank(card) == "A" for card in top),
        "opp_top_has_ace": any(card_rank(card) == "A" for card in opp_top),
    }


def gate_match(record: dict, gate: str) -> bool:
    f = board_features(record)
    if gate == "all":
        return True
    if gate == "visible_joker":
        return bool(f["visible_joker"] and not f["dealt_joker"])
    if gate == "top_structured":
        return bool(
            not f["dealt_joker"]
            and (
                f["own_top_len"] >= 2
                or f["own_top_has_pair_or_joker"]
                or f["opp_top_has_pair_or_joker"]
            )
        )
    if gate == "top_single_unpaired":
        return bool(f["own_top_len"] == 1 and not f["own_top_has_pair_or_joker"])
    if gate == "target_like_v1":
        return bool(
            f["visible_joker"]
            and not f["dealt_joker"]
            and (
                f["own_top_len"] >= 2
                or f["own_top_has_pair_or_joker"]
                or f["opp_top_has_pair_or_joker"]
            )
        )
    if gate == "target_like_strict":
        return bool(
            f["visible_joker"]
            and not f["dealt_joker"]
            and f["own_top_len"] >= 2
            and (f["own_top_has_pair_or_joker"] or f["opp_top_has_pair_or_joker"])
        )
    if gate == "ace_or_pair_joker":
        return bool(
            not f["dealt_joker"]
            and (
                f["own_top_has_ace"]
                or f["opp_top_has_ace"]
                or f["own_top_has_pair_or_joker"]
                or f["opp_top_has_pair_or_joker"]
            )
        )
    raise ValueError(f"Unknown gate: {gate}")


def parse_dataset_spec(spec: str) -> tuple[str, Path, str]:
    parts = spec.split("|")
    if len(parts) not in (2, 3):
        raise ValueError("--dataset must be name|data_dir or name|data_dir|source_jsonl")
    name = parts[0].strip()
    data_dir = Path(parts[1].strip())
    source = parts[2].strip() if len(parts) == 3 else ""
    if not name:
        raise ValueError("dataset name is empty")
    return name, data_dir, source


def group_gate_mask(records: list[dict], bounds: list[tuple[int, int]], gate: str) -> np.ndarray:
    mask = np.zeros(len(bounds), dtype=bool)
    for group_id in range(len(bounds)):
        record = records[group_id] if group_id < len(records) else {}
        mask[group_id] = gate_match(record, gate)
    return mask


def apply_group_blend(
    base: np.ndarray,
    specialist: np.ndarray,
    bounds: list[tuple[int, int]],
    gate_mask: np.ndarray,
    specialist_weight: float,
) -> np.ndarray:
    blended = np.asarray(base, dtype=np.float32).copy()
    w = float(specialist_weight)
    for group_id, enabled in enumerate(gate_mask):
        if not enabled:
            continue
        start, end = bounds[group_id]
        blended[start:end] = (1.0 - w) * blended[start:end] + w * specialist[start:end]
    return blended


def apply_sample_blend(
    base: np.ndarray,
    specialist: np.ndarray,
    sample_mask: np.ndarray,
    specialist_weight: float,
) -> np.ndarray:
    blended = np.asarray(base, dtype=np.float32).copy()
    if not bool(sample_mask.any()):
        return blended
    w = float(specialist_weight)
    blended[sample_mask] = (1.0 - w) * blended[sample_mask] + w * specialist[sample_mask]
    return blended


def group_mask_from_sample_mask(sample_mask: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    group_mask = np.zeros(len(bounds), dtype=bool)
    for group_id, (start, end) in enumerate(bounds):
        group_mask[group_id] = bool(sample_mask[start:end].any())
    return group_mask


def evaluate_dataset(
    *,
    name: str,
    data_dir: Path,
    source_jsonl: str,
    base_checkpoints: list[str],
    base_weights: list[float],
    specialist_checkpoint: str,
    specialist_weight: float,
    gate: str,
    gate_source: str,
    topks: list[int],
    device: torch.device,
    batch_size: int,
    bust_full_threshold: float,
) -> dict:
    meta = load_metadata(data_dir)
    if not source_jsonl:
        source_jsonl = str(meta.get("source", ""))
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    states = states[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    bust = np.asarray(np.load(data_dir / "bust.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    fl = np.asarray(np.load(data_dir / "fl.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    records = load_source_records(source_jsonl, len(bounds))

    base_score, base_bust, base_fl = predict_ensemble(
        base_checkpoints,
        base_weights,
        states,
        turns,
        device,
        batch_size,
    )
    specialist_score, specialist_bust, specialist_fl = predict_ensemble(
        [specialist_checkpoint],
        [1.0],
        states,
        turns,
        device,
        batch_size,
    )
    sample_gate_mask = np.zeros(n_samples, dtype=bool)
    if gate_source == "record":
        gate_mask = group_gate_mask(records, bounds, gate)
        pred_score = apply_group_blend(base_score, specialist_score, bounds, gate_mask, specialist_weight)
        pred_bust = apply_group_blend(base_bust, specialist_bust, bounds, gate_mask, specialist_weight)
        pred_fl = apply_group_blend(base_fl, specialist_fl, bounds, gate_mask, specialist_weight)
        for group_id, enabled in enumerate(gate_mask):
            if not enabled:
                continue
            start, end = bounds[group_id]
            sample_gate_mask[start:end] = True
    elif gate_source == "state":
        sample_gate_mask = encoded_state_gate_mask(
            torch.from_numpy(np.array(states, dtype=np.float32, copy=True)),
            gate,
        ).cpu().numpy()
        gate_mask = group_mask_from_sample_mask(sample_gate_mask, bounds)
        pred_score = apply_sample_blend(base_score, specialist_score, sample_gate_mask, specialist_weight)
        pred_bust = apply_sample_blend(base_bust, specialist_bust, sample_gate_mask, specialist_weight)
        pred_fl = apply_sample_blend(base_fl, specialist_fl, sample_gate_mask, specialist_weight)
    else:
        raise ValueError(f"Unknown gate_source: {gate_source}")
    metrics, _ = summarize_groups(
        scores,
        bust,
        fl,
        pred_score,
        pred_bust,
        pred_fl,
        bounds,
        topks,
        bust_full_threshold,
    )
    return {
        "dataset": name,
        "data": str(data_dir),
        "source_jsonl": source_jsonl,
        "gate": gate,
        "gate_source": gate_source,
        "gate_groups": int(gate_mask.sum()),
        "gate_samples": int(sample_gate_mask.sum()),
        "groups": int(len(bounds)),
        "samples": int(n_samples),
        "gate_fraction": float(gate_mask.mean()) if len(gate_mask) else 0.0,
        "specialist_weight": float(specialist_weight),
        "metrics": metrics,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate feature-gated action-value specialist blends")
    parser.add_argument("--dataset", action="append", required=True, help="name|data_dir or name|data_dir|source_jsonl")
    parser.add_argument("--base-checkpoints", nargs="+", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--specialist-checkpoint", required=True)
    parser.add_argument("--specialist-weight", type=float, required=True)
    parser.add_argument("--gate", default="target_like_v1")
    parser.add_argument("--gate-source", choices=["record", "state"], default="record")
    parser.add_argument("--topks", default="1,3,5,10")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bust-full-threshold", type=float, default=0.999)
    args = parser.parse_args(argv)

    start = time.time()
    datasets = [parse_dataset_spec(spec) for spec in args.dataset]
    base_weights = parse_weights(args.base_weights, len(args.base_checkpoints))
    topks = parse_topks(args.topks)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, data_dir, source_jsonl in datasets:
        result = evaluate_dataset(
            name=name,
            data_dir=data_dir,
            source_jsonl=source_jsonl,
            base_checkpoints=list(args.base_checkpoints),
            base_weights=base_weights,
            specialist_checkpoint=args.specialist_checkpoint,
            specialist_weight=args.specialist_weight,
            gate=args.gate,
            gate_source=args.gate_source,
            topks=topks,
            device=device,
            batch_size=args.batch_size,
            bust_full_threshold=args.bust_full_threshold,
        )
        results.append(result)

    summary = {
        "gate": args.gate,
        "gate_source": args.gate_source,
        "specialist_weight": float(args.specialist_weight),
        "base_checkpoints": list(args.base_checkpoints),
        "base_weights": base_weights,
        "specialist_checkpoint": args.specialist_checkpoint,
        "datasets": results,
        "elapsed_seconds": time.time() - start,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "dataset": result["dataset"],
                "gate_groups": result["gate_groups"],
                "gate_samples": result["gate_samples"],
                "groups": result["groups"],
                "samples": result["samples"],
                "top1": metrics.get("group_top1", 0.0),
                "reg1": metrics.get("group_top1_regret", 0.0),
                "top3": metrics.get("group_top3", 0.0),
                "reg3": metrics.get("group_top3_rerank_regret", 0.0),
                "top5": metrics.get("group_top5", 0.0),
                "reg5": metrics.get("group_top5_rerank_regret", 0.0),
                "top10": metrics.get("group_top10", 0.0),
                "reg10": metrics.get("group_top10_rerank_regret", 0.0),
            }
        )
    with (out_dir / "summary_table.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"gated blend gate={args.gate} specialist_weight={args.specialist_weight:g}")
    for row in rows:
        print(
            f"  {row['dataset']}: gate={row['gate_groups']}/{row['groups']} groups "
            f"{row['gate_samples']}/{row['samples']} samples "
            f"top3={row['top3']:.1%} reg3={row['reg3']:.3f} "
            f"top5={row['top5']:.1%} reg5={row['reg5']:.3f} "
            f"top10={row['top10']:.1%} reg10={row['reg10']:.3f}"
        )
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()
