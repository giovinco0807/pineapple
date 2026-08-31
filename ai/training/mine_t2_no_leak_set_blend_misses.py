"""Mine T2 Top1 misses for a no-leak set-score blend.

The inputs are the converted candidate dataset plus the matching teacher JSONL.
Only runtime-available scores are used for selection:

- base action-value ensemble scores
- old 520-dim set-model scores
- new 617-dim set-model scores

The teacher JSONL is used only to describe and export misses for future
training data, not as an inference feature.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_t2_no_leak_set_blends import (
    load_arrays,
    parse_blend,
    predict_or_load,
)


def parse_dataset(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--dataset must be name=dim520,dim617,teacher_jsonl")
    name = parts[0].strip()
    paths = [Path(part.strip()) for part in parts[1].split(",")]
    if len(paths) != 3:
        raise ValueError("--dataset must provide dim520, dim617, and teacher JSONL")
    return name, paths[0], paths[1], paths[2]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def action_summary(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "placements": candidate.get("placements"),
        "discard": candidate.get("discard"),
        "ev": candidate.get("ev", candidate.get("score")),
        "bust_prob": candidate.get("bust_prob", candidate.get("bust_rate")),
        "fl_rate": candidate.get("fl_rate"),
        "fl_type_rates": candidate.get("fl_type_rates"),
        "teacher_rank": candidate.get("teacher_rank"),
        "forced_bust": candidate.get("forced_bust"),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def topk_hit(pred_order: np.ndarray, best_local: int, k: int) -> bool:
    return int(best_local) in {int(x) for x in pred_order[: min(k, len(pred_order))]}


def mine_dataset(
    name: str,
    dim520: Path,
    dim617: Path,
    teacher_path: Path,
    old_checkpoints: list[Path],
    new_checkpoints: list[Path],
    blend_name: str,
    weights: np.ndarray,
    gamma: float,
    device: torch.device,
    batch_size: int,
    topn: int,
) -> tuple[list[dict], dict]:
    scores, group_ids, bounds, base_scores = load_arrays(dim520)
    old1 = predict_or_load(dim520, "old1_520", old_checkpoints[0], device, batch_size)
    old2 = predict_or_load(dim520, "old2_520", old_checkpoints[1], device, batch_size)
    new1 = predict_or_load(dim617, "new1_617", new_checkpoints[0], device, batch_size)
    new2 = predict_or_load(dim617, "new2_617", new_checkpoints[1], device, batch_size)
    blend_scores = weights[0] * old1 + weights[1] * old2 + weights[2] * new1 + weights[3] * new2
    pred_scores = base_scores + gamma * (blend_scores - base_scores)

    teacher_rows = load_jsonl(teacher_path)
    misses: list[dict] = []
    regrets: list[float] = []
    hits = {1: 0, 3: 0, 5: 10, 10: 0, 15: 0, 20: 0}
    hits[5] = 0

    for group_pos, (start, end) in enumerate(bounds):
        gid = int(group_ids[start])
        row = teacher_rows[gid] if gid < len(teacher_rows) else {}
        local_scores = np.asarray(scores[start:end], dtype=np.float32)
        local_pred = np.asarray(pred_scores[start:end], dtype=np.float32)
        pred_order = np.argsort(-local_pred)
        best_local = int(np.argmax(local_scores))
        chosen_local = int(pred_order[0])
        best_ev = float(local_scores[best_local])
        chosen_ev = float(local_scores[chosen_local])
        regret = float(best_ev - chosen_ev)
        regrets.append(regret)
        for k in hits:
            if topk_hit(pred_order, best_local, k):
                hits[k] += 1

        if chosen_local == best_local:
            continue
        candidates = list(row.get("candidates") or [])
        pred_rank_best = int(np.where(pred_order == best_local)[0][0]) + 1
        top_pred = []
        for local_idx in pred_order[: min(topn, len(pred_order))]:
            candidate = candidates[int(local_idx)] if int(local_idx) < len(candidates) else None
            top_pred.append(
                {
                    "sample_index": int(start + int(local_idx)),
                    "candidate_offset": int(local_idx),
                    "pred_score": float(local_pred[int(local_idx)]),
                    "teacher_ev": float(local_scores[int(local_idx)]),
                    "teacher_rank": int(np.where(np.argsort(-local_scores) == int(local_idx))[0][0]) + 1,
                    "action": action_summary(candidate),
                }
            )
        misses.append(
            {
                "data": str(dim520),
                "dataset": name,
                "group_id": gid,
                "group_position": group_pos,
                "position": row.get("position"),
                "is_btn": row.get("is_btn"),
                "turn": row.get("turn"),
                "board": row.get("board"),
                "opponent_board": row.get("opponent_board"),
                "dealt": row.get("dealt"),
                "known_discards": row.get("known_discards"),
                "future_hidden_deals": row.get("future_hidden_deals"),
                "candidate_count": int(end - start),
                "best_candidate_offset": best_local,
                "chosen_candidate_offset": chosen_local,
                "best_ev": best_ev,
                "chosen_ev": chosen_ev,
                "regret": regret,
                "top1_ev_loss": regret,
                "best_pred_rank": pred_rank_best,
                "best_pred_score": float(local_pred[best_local]),
                "chosen_pred_score": float(local_pred[chosen_local]),
                "teacher_best": action_summary(candidates[best_local] if best_local < len(candidates) else None),
                "model_chosen": action_summary(candidates[chosen_local] if chosen_local < len(candidates) else None),
                "top_predictions": top_pred,
                "top_predicted": top_pred,
            }
        )

    groups = len(bounds)
    summary = {
        "dataset": name,
        "groups": groups,
        "samples": int(len(scores)),
        "selector": blend_name,
        "gamma": gamma,
        "weights": [float(x) for x in weights],
        "top1": hits[1] / max(groups, 1),
        "top3": hits[3] / max(groups, 1),
        "top5": hits[5] / max(groups, 1),
        "top10": hits[10] / max(groups, 1),
        "top15": hits[15] / max(groups, 1),
        "top20": hits[20] / max(groups, 1),
        "misses": len(misses),
        "regret_mean": float(statistics.fmean(regrets)) if regrets else 0.0,
        "regret_p90": percentile(regrets, 0.90),
        "regret_p95": percentile(regrets, 0.95),
        "regret_p99": percentile(regrets, 0.99),
        "regret_max": max(regrets) if regrets else 0.0,
        "regret_ge_0p1": sum(x >= 0.1 for x in regrets),
        "regret_ge_0p5": sum(x >= 0.5 for x in regrets),
        "regret_ge_1p0": sum(x >= 1.0 for x in regrets),
        "regret_ge_3p0": sum(x >= 3.0 for x in regrets),
    }
    return misses, summary


def aggregate(summaries: list[dict], all_misses: list[dict]) -> dict:
    groups = sum(int(row["groups"]) for row in summaries)
    samples = sum(int(row["samples"]) for row in summaries)
    regrets = [float(row["top1_ev_loss"]) for row in all_misses]
    weighted = {}
    for key in ["top1", "top3", "top5", "top10", "top15", "top20"]:
        weighted[key] = sum(float(row[key]) * int(row["groups"]) for row in summaries) / max(groups, 1)
    return {
        "groups": groups,
        "samples": samples,
        **weighted,
        "misses": len(all_misses),
        "miss_regret_mean": float(statistics.fmean(regrets)) if regrets else 0.0,
        "miss_regret_p90": percentile(regrets, 0.90),
        "miss_regret_p95": percentile(regrets, 0.95),
        "miss_regret_p99": percentile(regrets, 0.99),
        "miss_regret_max": max(regrets) if regrets else 0.0,
        "miss_regret_ge_0p1": sum(x >= 0.1 for x in regrets),
        "miss_regret_ge_0p5": sum(x >= 0.5 for x in regrets),
        "miss_regret_ge_1p0": sum(x >= 1.0 for x in regrets),
        "miss_regret_ge_3p0": sum(x >= 3.0 for x in regrets),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="name=dim520,dim617,teacher_jsonl")
    parser.add_argument("--old-checkpoints", nargs=2, required=True)
    parser.add_argument("--new-checkpoints", nargs=2, required=True)
    parser.add_argument("--blend", default="old_new_avg=0.25,0.25,0.25,0.25")
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    blend_name, weights = parse_blend(args.blend)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    old_checkpoints = [Path(path) for path in args.old_checkpoints]
    new_checkpoints = [Path(path) for path in args.new_checkpoints]
    datasets = [parse_dataset(spec) for spec in args.dataset]

    all_misses: list[dict] = []
    summaries: list[dict] = []
    for name, dim520, dim617, teacher_path in datasets:
        misses, summary = mine_dataset(
            name,
            dim520,
            dim617,
            teacher_path,
            old_checkpoints,
            new_checkpoints,
            blend_name,
            weights,
            args.gamma,
            device,
            args.batch_size,
            args.topn,
        )
        all_misses.extend(misses)
        summaries.append(summary)

    all_misses.sort(key=lambda row: float(row["top1_ev_loss"]), reverse=True)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_misses:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "selector": blend_name,
        "gamma": args.gamma,
        "weights": [float(x) for x in weights],
        "device": str(device),
        "datasets": summaries,
        "aggregate": aggregate(summaries, all_misses),
        "misses_jsonl": str(out_path),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"summary": str(summary_path), "misses": str(out_path), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
