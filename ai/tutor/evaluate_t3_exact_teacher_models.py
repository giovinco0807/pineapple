"""Evaluate T3 models against exact teacher JSONL.

Reports both model Top1 quality and model TopK + exact-rerank EV loss.  The
TopK metrics are the practical pruning signal: if the exact-best action is in
the model pool, exact reranking can recover it.
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
from ai.tutor.exact_late import CardNormalizer, normalize_board


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_topks(value: str) -> list[int]:
    out = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not out or any(k <= 0 for k in out):
        raise argparse.ArgumentTypeError("--topks must contain positive integers")
    return out


def position_of(row: dict[str, Any]) -> str:
    return str(row.get("position") or ("btn" if row.get("is_btn") else "bb")).lower()


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "gt_0p1": 0, "gt_0p5": 0, "gt_1": 0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "gt_0p1": int(sum(value > 0.1 for value in values)),
        "gt_0p5": int(sum(value > 0.5 for value in values)),
        "gt_1": int(sum(value > 1.0 for value in values)),
    }


def pack_rank_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "top15": 0.0, "top20": 0.0}
    n = len(ranks)
    return {
        "top1": sum(rank <= 1 for rank in ranks) / n,
        "top3": sum(rank <= 3 for rank in ranks) / n,
        "top5": sum(rank <= 5 for rank in ranks) / n,
        "top10": sum(rank <= 10 for rank in ranks) / n,
        "top15": sum(rank <= 15 for rank in ranks) / n,
        "top20": sum(rank <= 20 for rank in ranks) / n,
    }


def parse_model_pair(value: str) -> tuple[str, dict[str, str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be name=bb_path,btn_path")
    name, paths = value.split("=", 1)
    parts = [part.strip() for part in paths.split(",") if part.strip()]
    if len(parts) == 1:
        return name, {"bb": parts[0], "btn": parts[0]}
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--model must be name=bb_path,btn_path")
    return name, {"bb": parts[0], "btn": parts[1]}


def load_models(paths: dict[str, str], device: torch.device) -> dict[str, ActionValueReranker]:
    return {
        pos: ActionValueReranker.from_checkpoint(path, map_location=device).to(device).eval()
        for pos, path in paths.items()
    }


def evaluate_model(
    *,
    rows: list[dict[str, Any]],
    model_paths: dict[str, str],
    topks: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normalizer = CardNormalizer()
    models = load_models(model_paths, device)
    max_k = max(topks)
    top1_ranks: list[int] = []
    top1_losses: list[float] = []
    pool_losses = {k: [] for k in topks}
    pool_recalls = {k: 0 for k in topks}
    by_position: dict[str, dict[str, Any]] = defaultdict(lambda: {"ranks": [], "top1_losses": [], "pool_losses": {k: [] for k in topks}, "pool_recalls": {k: 0 for k in topks}, "rows": 0})
    worst_top1 = None
    worst_pool = {k: None for k in topks}

    for row_i, row in enumerate(rows):
        position = position_of(row)
        board = normalize_board(row["board"], normalizer)
        opponent_board = normalize_board(row.get("opponent_board") or {}, normalizer)
        model_eval = score_model_t3(
            model=models[position],
            device=device,
            board=board,
            opponent_board=opponent_board,
            draw=row["dealt"],
            dead=row.get("exclude") or [],
            is_btn=position == "btn",
            batch_size=batch_size,
            top_n=max_k,
        )
        exact_items = []
        for candidate in row.get("candidates") or []:
            action = {"placements": candidate.get("placements") or [], "discard": candidate.get("discard")}
            exact_items.append((compact_action_key(action), float(candidate.get("ev", 0.0)), action))
        if not exact_items:
            continue
        exact_by_key = {key: (rank, score, action) for rank, (key, score, action) in enumerate(exact_items, start=1)}
        exact_best_key, exact_best_score, exact_best_action = exact_items[0]
        top_actions = model_eval.get("top") or []
        top1_key = compact_action_key(top_actions[0]["action"])
        top1_rank, top1_score, top1_action = exact_by_key.get(top1_key, (999, -999.0, top_actions[0]["action"]))
        top1_loss = max(0.0, exact_best_score - top1_score)

        top1_ranks.append(int(top1_rank))
        top1_losses.append(float(top1_loss))
        pos_bucket = by_position[position]
        pos_bucket["rows"] += 1
        pos_bucket["ranks"].append(int(top1_rank))
        pos_bucket["top1_losses"].append(float(top1_loss))
        if worst_top1 is None or top1_loss > worst_top1["loss"]:
            worst_top1 = {
                "row": row_i,
                "position": position,
                "loss": float(top1_loss),
                "rank": int(top1_rank),
                "dealt": row["dealt"],
                "model_action": top1_action,
                "exact_action": exact_best_action,
                "best_score": exact_best_score,
                "model_exact_score": top1_score,
            }

        for k in topks:
            pool = top_actions[: min(k, len(top_actions))]
            pool_scores = []
            pool_keys = set()
            for item in pool:
                key = compact_action_key(item["action"])
                pool_keys.add(key)
                if key in exact_by_key:
                    pool_scores.append(float(exact_by_key[key][1]))
            best_pool_score = max(pool_scores) if pool_scores else -999.0
            loss = max(0.0, exact_best_score - best_pool_score)
            pool_losses[k].append(float(loss))
            pos_bucket["pool_losses"][k].append(float(loss))
            if exact_best_key in pool_keys:
                pool_recalls[k] += 1
                pos_bucket["pool_recalls"][k] += 1
            if worst_pool[k] is None or loss > worst_pool[k]["loss"]:
                worst_pool[k] = {
                    "row": row_i,
                    "position": position,
                    "loss": float(loss),
                    "dealt": row["dealt"],
                    "exact_action": exact_best_action,
                }

    n = len(top1_ranks)
    out = {
        "rows": n,
        "top1_rank_metrics": pack_rank_metrics(top1_ranks),
        "top1_ev_loss": stats(top1_losses),
        "pool": {
            str(k): {
                "recall": pool_recalls[k] / n if n else 0.0,
                "ev_loss": stats(pool_losses[k]),
                "worst": worst_pool[k],
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
            "top1_rank_metrics": pack_rank_metrics(bucket["ranks"]),
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
    parser = argparse.ArgumentParser(description="Evaluate T3 exact teacher JSONL with TopK pool metrics")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=bb_path,btn_path")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = list(iter_jsonl(Path(args.teacher)))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    topks = parse_topks(args.topks)
    result = {
        "teacher": str(args.teacher),
        "topks": topks,
        "models": {},
    }
    for value in args.model:
        name, paths = parse_model_pair(value)
        result["models"][name] = evaluate_model(
            rows=rows,
            model_paths=paths,
            topks=topks,
            device=device,
            batch_size=args.batch_size,
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
