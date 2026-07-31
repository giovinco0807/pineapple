"""Evaluate a union of T3 model TopK pools against exact teacher data.

The practical T3 serving path is model pruning followed by exact reranking.
This reports whether the exact-best action survives the union candidate pool
from several model sources, and how much EV is lost if it does not.
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

from ai.tutor.diagnose_t3_model_from_t2_miss import compact_action_key, score_model_t3
from ai.tutor.evaluate_t3_exact_teacher_models import (
    iter_jsonl,
    load_models,
    parse_model_pair,
    parse_topks,
    position_of,
    stats,
)
from ai.tutor.exact_late import CardNormalizer, normalize_board


def load_group_bounds(data_dir: Path) -> list[tuple[int, int]]:
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r"), dtype=np.int64)
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


def evaluate_union(
    *,
    rows: list[dict[str, Any]],
    model_sets: dict[str, dict[str, Any]],
    topks: list[int],
    device: torch.device,
    batch_size: int,
    group_bounds: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    normalizer = CardNormalizer()
    max_k = max(topks)
    losses = {k: [] for k in topks}
    recalls = {k: 0 for k in topks}
    pool_sizes = {k: [] for k in topks}
    worst = {k: None for k in topks}
    row_details: list[dict[str, Any]] = []
    by_position: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "losses": {k: [] for k in topks},
            "recalls": {k: 0 for k in topks},
            "pool_sizes": {k: [] for k in topks},
        }
    )

    for row_i, row in enumerate(rows):
        position = position_of(row)
        board = normalize_board(row["board"], normalizer)
        opponent_board = normalize_board(row.get("opponent_board") or {}, normalizer)

        exact_items = []
        candidates = list(row.get("candidates") or [])
        for local_i, candidate in enumerate(candidates):
            action = {"placements": candidate.get("placements") or [], "discard": candidate.get("discard")}
            if group_bounds and row_i < len(group_bounds):
                start, end = group_bounds[row_i]
                global_i = start + local_i
                if global_i >= end:
                    global_i = -1
            else:
                global_i = local_i
            exact_items.append((compact_action_key(action), float(candidate.get("ev", 0.0)), action, int(global_i)))
        if not exact_items:
            continue
        exact_by_key = {
            key: (rank, score, action, global_i)
            for rank, (key, score, action, global_i) in enumerate(exact_items, start=1)
        }
        exact_best_key, exact_best_score, exact_best_action, exact_best_index = exact_items[0]

        source_top: dict[str, list[dict[str, Any]]] = {}
        for name, models in model_sets.items():
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
            source_top[name] = list(model_eval.get("top") or [])

        pos_bucket = by_position[position]
        pos_bucket["rows"] += 1
        detail = {
            "group_id": row_i,
            "size": len(exact_items),
            "teacher_best_index": int(exact_best_index),
            "teacher_best_score": float(exact_best_score),
            "dealt_ev": float(exact_best_score),
            "position": position,
            "dealt": row.get("dealt"),
            "pool_metrics": {},
        }
        for k in topks:
            pool: dict[str, dict[str, Any]] = {}
            for source_name, top_actions in source_top.items():
                for item in top_actions[: min(k, len(top_actions))]:
                    key = compact_action_key(item["action"])
                    if key not in pool:
                        pool[key] = {
                            "action": item["action"],
                            "source": source_name,
                            "model_rank": item.get("rank"),
                        }

            pool_scores = [
                float(exact_by_key[key][1])
                for key in pool
                if key in exact_by_key
            ]
            best_pool_score = max(pool_scores) if pool_scores else -999.0
            loss = max(0.0, exact_best_score - best_pool_score)
            pool_size = len(pool)

            losses[k].append(float(loss))
            pool_sizes[k].append(pool_size)
            pos_bucket["losses"][k].append(float(loss))
            pos_bucket["pool_sizes"][k].append(pool_size)
            if exact_best_key in pool:
                recalls[k] += 1
                pos_bucket["recalls"][k] += 1
            if worst[k] is None or loss > worst[k]["loss"]:
                worst[k] = {
                    "row": row_i,
                    "position": position,
                    "loss": float(loss),
                    "dealt": row["dealt"],
                    "exact_action": exact_best_action,
                    "exact_score": exact_best_score,
                    "pool_size": pool_size,
                }
            pool_indices = sorted(
                int(exact_by_key[key][3])
                for key in pool
                if key in exact_by_key and int(exact_by_key[key][3]) >= 0
            )
            detail["pool_metrics"][str(k)] = {
                "pool_size": int(pool_size),
                "hit": bool(exact_best_key in pool),
                "regret": float(loss),
                "ev_loss": float(loss),
                "pool_best_ev": float(best_pool_score),
                "dealt_ev": float(exact_best_score),
                "indices": pool_indices,
            }
        row_details.append(detail)

    n = len(next(iter(losses.values()))) if losses else 0
    out = {
        "rows": n,
        "pool": {},
        "by_position": {},
        "row_details": row_details,
    }
    for k in topks:
        sizes = pool_sizes[k]
        out["pool"][str(k)] = {
            "recall": recalls[k] / n if n else 0.0,
            "ev_loss": stats(losses[k]),
            "pool_size_avg": float(np.mean(sizes)) if sizes else 0.0,
            "pool_size_p95": float(np.percentile(sizes, 95)) if sizes else 0.0,
            "pool_size_max": int(max(sizes)) if sizes else 0,
            "worst": worst[k],
        }

    for pos, bucket in by_position.items():
        pos_n = int(bucket["rows"])
        out["by_position"][pos] = {
            "rows": pos_n,
            "pool": {},
        }
        for k in topks:
            sizes = bucket["pool_sizes"][k]
            out["by_position"][pos]["pool"][str(k)] = {
                "recall": bucket["recalls"][k] / pos_n if pos_n else 0.0,
                "ev_loss": stats(bucket["losses"][k]),
                "pool_size_avg": float(np.mean(sizes)) if sizes else 0.0,
                "pool_size_p95": float(np.percentile(sizes, 95)) if sizes else 0.0,
                "pool_size_max": int(max(sizes)) if sizes else 0,
            }
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate union TopK T3 pools with exact-rerank EV loss")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=bb_path,btn_path")
    parser.add_argument("--topks", default="3,5,10,15,20")
    parser.add_argument("--output", default="")
    parser.add_argument("--rows-output", default="")
    parser.add_argument("--data-dir", default="", help="Optional converted action-value data directory for global sample indices")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = list(iter_jsonl(Path(args.teacher)))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    topks = parse_topks(args.topks)
    model_paths = dict(parse_model_pair(value) for value in args.model)
    model_sets = {name: load_models(paths, device) for name, paths in model_paths.items()}
    group_bounds = load_group_bounds(Path(args.data_dir)) if args.data_dir else None

    result = {
        "teacher": str(args.teacher),
        "topks": topks,
        "models": model_paths,
        "union": evaluate_union(
            rows=rows,
            model_sets=model_sets,
            topks=topks,
            device=device,
            batch_size=args.batch_size,
            group_bounds=group_bounds,
        ),
    }
    row_details = result["union"].pop("row_details")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.rows_output:
        Path(args.rows_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.rows_output).open("w", encoding="utf-8") as handle:
            for detail in row_details:
                handle.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
