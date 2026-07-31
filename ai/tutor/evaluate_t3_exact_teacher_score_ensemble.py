"""Evaluate model-only T3 score ensembles against exact teacher JSONL.

Unlike union-pool evaluation, this chooses an action directly from model
predictions, without exact reranking.  It measures the instant-decision path.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.diagnose_t3_model_from_t2_miss import compact_action_key, score_model_t3
from ai.tutor.evaluate_t3_exact_teacher_models import parse_model_pair, parse_topks, position_of, stats
from ai.tutor.exact_late import CardNormalizer, normalize_board


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_model_pairs(values: list[str], device: torch.device) -> dict[str, dict[str, ActionValueReranker]]:
    loaded: dict[str, dict[str, ActionValueReranker]] = {}
    for value in values:
        name, paths = parse_model_pair(value)
        loaded[name] = {
            pos: ActionValueReranker.from_checkpoint(path, map_location=device).to(device).eval()
            for pos, path in paths.items()
        }
    return loaded


def rank_metrics(ranks: list[int], topks: list[int]) -> dict[str, float]:
    n = len(ranks)
    if not n:
        return {f"top{k}": 0.0 for k in topks}
    return {f"top{k}": sum(rank <= k for rank in ranks) / n for k in topks}


def evaluate(
    *,
    rows: list[dict[str, Any]],
    model_args: list[str],
    topks: list[int],
    method: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normalizer = CardNormalizer()
    models = load_model_pairs(model_args, device)
    max_k = max(64, max(topks))

    ranks: list[int] = []
    top1_losses: list[float] = []
    pool_losses = {k: [] for k in topks}
    pool_recalls = {k: 0 for k in topks}
    by_position: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "ranks": [], "top1_losses": [], "pool_losses": {k: [] for k in topks}, "pool_recalls": {k: 0 for k in topks}}
    )
    worst_top1 = None

    for row_i, row in enumerate(rows):
        position = position_of(row)
        board = normalize_board(row["board"], normalizer)
        opponent_board = normalize_board(row.get("opponent_board") or {}, normalizer)

        exact_items = []
        for candidate in row.get("candidates") or []:
            action = {"placements": candidate.get("placements") or [], "discard": candidate.get("discard")}
            exact_items.append((compact_action_key(action), float(candidate.get("ev", 0.0)), action))
        if not exact_items:
            continue
        exact_by_key = {key: (rank, score, action) for rank, (key, score, action) in enumerate(exact_items, start=1)}
        exact_best_key, exact_best_score, exact_best_action = exact_items[0]

        score_sum: dict[Any, float] = {}
        score_count: dict[Any, int] = {}
        actions: dict[Any, dict[str, Any]] = {}
        for pair in models.values():
            model_eval = score_model_t3(
                model=pair[position],
                device=device,
                board=board,
                opponent_board=opponent_board,
                draw=row["dealt"],
                dead=row.get("exclude") or [],
                is_btn=position == "btn",
                batch_size=batch_size,
                top_n=max_k,
            )
            items = list(model_eval.get("top") or [])
            model_scores = np.asarray([float(item.get("model_score", 0.0)) for item in items], dtype=np.float32)
            if method == "score_zmean" and len(model_scores) > 1:
                std = float(model_scores.std())
                if std > 1e-6:
                    model_scores = (model_scores - float(model_scores.mean())) / std
                else:
                    model_scores = model_scores * 0.0
            for rank, item in enumerate(items, start=1):
                key = compact_action_key(item["action"])
                if method in ("score_mean", "score_zmean"):
                    value = float(model_scores[rank - 1])
                elif method == "rank_mean":
                    value = -float(rank)
                elif method == "reciprocal_rank":
                    value = 1.0 / float(rank)
                else:
                    raise ValueError(f"unknown ensemble method: {method}")
                score_sum[key] = score_sum.get(key, 0.0) + value
                score_count[key] = score_count.get(key, 0) + 1
                actions.setdefault(key, item["action"])

        ordered = sorted(
            score_sum,
            key=lambda key: score_sum[key] / max(score_count.get(key, 1), 1),
            reverse=True,
        )
        if not ordered:
            continue
        best_key = ordered[0]
        best_rank, best_exact_score, best_action = exact_by_key.get(best_key, (999, -999.0, actions[best_key]))
        top1_loss = max(0.0, exact_best_score - best_exact_score)
        ranks.append(int(best_rank))
        top1_losses.append(float(top1_loss))

        pos_bucket = by_position[position]
        pos_bucket["rows"] += 1
        pos_bucket["ranks"].append(int(best_rank))
        pos_bucket["top1_losses"].append(float(top1_loss))
        if worst_top1 is None or top1_loss > worst_top1["loss"]:
            worst_top1 = {
                "row": row_i,
                "position": position,
                "loss": float(top1_loss),
                "rank": int(best_rank),
                "dealt": row["dealt"],
                "model_action": best_action,
                "exact_action": exact_best_action,
                "best_score": exact_best_score,
                "model_exact_score": best_exact_score,
            }

        for k in topks:
            pool = ordered[: min(k, len(ordered))]
            pool_scores = [exact_by_key[key][1] for key in pool if key in exact_by_key]
            best_pool_score = max(pool_scores) if pool_scores else -999.0
            loss = max(0.0, exact_best_score - best_pool_score)
            pool_losses[k].append(float(loss))
            pos_bucket["pool_losses"][k].append(float(loss))
            if exact_best_key in set(pool):
                pool_recalls[k] += 1
                pos_bucket["pool_recalls"][k] += 1

    n = len(ranks)
    out = {
        "rows": n,
        "models": [parse_model_pair(value)[0] for value in model_args],
        "method": method,
        "rank_metrics": rank_metrics(ranks, topks),
        "top1_ev_loss": stats(top1_losses),
        "pool": {
            str(k): {
                "recall": pool_recalls[k] / n if n else 0.0,
                "ev_loss": stats(pool_losses[k]),
            }
            for k in topks
        },
        "by_position": {},
        "worst_top1": worst_top1,
    }
    for pos, bucket in by_position.items():
        pos_n = int(bucket["rows"])
        out["by_position"][pos] = {
            "rows": pos_n,
            "rank_metrics": rank_metrics(bucket["ranks"], topks),
            "top1_ev_loss": stats(bucket["top1_losses"]),
            "pool": {
                str(k): {
                    "recall": bucket["pool_recalls"][k] / pos_n if pos_n else 0.0,
                    "ev_loss": stats(bucket["pool_losses"][k]),
                }
                for k in topks
            },
        }
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-only T3 score ensembles")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=bb_path,btn_path")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument(
        "--method",
        choices=("score_mean", "score_zmean", "rank_mean", "reciprocal_rank"),
        default="score_mean",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = list(iter_jsonl(Path(args.teacher)))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    result = {
        "teacher": str(args.teacher),
        "topks": parse_topks(args.topks),
        "ensemble": evaluate(
            rows=rows,
            model_args=args.model,
            topks=parse_topks(args.topks),
            method=args.method,
            device=device,
            batch_size=args.batch_size,
        ),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
