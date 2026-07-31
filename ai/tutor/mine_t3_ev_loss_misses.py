"""Mine T3 model TopK misses by exact EV loss and write group weights.

The input teacher JSONL is exact-ranked, one row per T3 position.  For each row,
this script scores all legal T3 actions with a BB/BTN model pair and compares
the model TopK pool against the exact best.  Rows where exact reranking inside
the pool would still lose EV are written to per-K JSONL files.

When --reranker-data is supplied, a group_sample_weights.npy file is also
written.  The action-value trainer already consumes this file for group-ranking
batches, making this a lightweight hard-negative loop without duplicating the
large candidate arrays.
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
from ai.tutor.evaluate_t3_exact_teacher_models import parse_model_pair, parse_topks, stats
from ai.tutor.exact_late import CardNormalizer, normalize_board


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def position_of(row: dict[str, Any]) -> str:
    return str(row.get("position") or ("btn" if row.get("is_btn") else "bb")).lower()


def load_models(paths: dict[str, str], device: torch.device) -> dict[str, ActionValueReranker]:
    return {
        pos: ActionValueReranker.from_checkpoint(path, map_location=device).to(device).eval()
        for pos, path in paths.items()
    }


def parse_thresholds(value: str, topks: list[int], default: float) -> dict[int, float]:
    thresholds = {int(k): float(default) for k in topks}
    if not value.strip():
        return thresholds
    for part in value.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError("--thresholds entries must be K:loss")
        raw_k, raw_loss = part.split(":", 1)
        thresholds[int(raw_k)] = float(raw_loss)
    return thresholds


def mine(args: argparse.Namespace) -> dict[str, Any]:
    teacher = Path(args.teacher)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(teacher))
    topks = parse_topks(args.topks)
    thresholds = parse_thresholds(args.thresholds, topks, args.min_loss)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    _name, model_paths = parse_model_pair(args.model)
    models = load_models(model_paths, device)
    normalizer = CardNormalizer()
    max_k = max(topks)

    misses: dict[int, list[dict[str, Any]]] = {k: [] for k in topks}
    losses: dict[int, list[float]] = {k: [] for k in topks}
    recalls: dict[int, int] = {k: 0 for k in topks}
    by_position: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "losses": {k: [] for k in topks},
            "recalls": {k: 0 for k in topks},
            "miss_counts": {k: 0 for k in topks},
        }
    )
    group_weights = np.ones(len(rows), dtype=np.float32)

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
            batch_size=args.batch_size,
            top_n=max_k,
        )
        exact_items: list[tuple[str, float, dict[str, Any]]] = []
        for candidate in row.get("candidates") or []:
            action = {
                "placements": candidate.get("placements") or [],
                "discard": candidate.get("discard"),
            }
            exact_items.append((compact_action_key(action), float(candidate.get("ev", 0.0)), action))
        if not exact_items:
            continue
        exact_by_key = {
            key: (rank, score, action)
            for rank, (key, score, action) in enumerate(exact_items, start=1)
        }
        exact_best_key, exact_best_score, exact_best_action = exact_items[0]
        top_actions = model_eval.get("top") or []
        pos_bucket = by_position[position]
        pos_bucket["rows"] += 1

        for k in topks:
            pool = top_actions[: min(k, len(top_actions))]
            pool_keys = {compact_action_key(item["action"]) for item in pool}
            pool_exact = [
                exact_by_key[key]
                for key in pool_keys
                if key in exact_by_key
            ]
            best_pool_rank, best_pool_score, best_pool_action = max(
                pool_exact,
                key=lambda item: item[1],
                default=(999, -999.0, None),
            )
            loss = max(0.0, exact_best_score - best_pool_score)
            losses[k].append(float(loss))
            pos_bucket["losses"][k].append(float(loss))
            if exact_best_key in pool_keys:
                recalls[k] += 1
                pos_bucket["recalls"][k] += 1
            if loss >= thresholds.get(k, args.min_loss):
                pos_bucket["miss_counts"][k] += 1
                miss = {
                    "row": row_i,
                    "group_id": row_i,
                    "position": position,
                    "topk": int(k),
                    "loss": float(loss),
                    "exact_best_score": float(exact_best_score),
                    "best_pool_score": float(best_pool_score),
                    "best_pool_exact_rank": int(best_pool_rank),
                    "dealt": row.get("dealt") or [],
                    "board": row.get("board") or {},
                    "opponent_board": row.get("opponent_board") or {},
                    "exclude": row.get("exclude") or [],
                    "known_discards": row.get("known_discards") or [],
                    "exact_action": exact_best_action,
                    "best_pool_action": best_pool_action,
                    "model_top": top_actions[: min(k, len(top_actions))],
                    "source": row.get("source"),
                    "source_line": row.get("source_line"),
                }
                misses[k].append(miss)
                if k == int(args.weight_topk) and loss >= float(args.weight_min_loss):
                    group_weights[row_i] = max(
                        float(group_weights[row_i]),
                        min(
                            float(args.max_group_weight),
                            1.0 + float(loss) * float(args.loss_scale),
                        ),
                    )

    output_paths = {}
    for k in topks:
        path = output_dir / f"misses_top{k}.jsonl"
        misses[k].sort(key=lambda item: float(item["loss"]), reverse=True)
        output_paths[str(k)] = {
            "path": str(path),
            "count": write_jsonl(path, misses[k]),
        }

    weight_path = None
    if args.reranker_data:
        reranker_dir = Path(args.reranker_data)
        reranker_dir.mkdir(parents=True, exist_ok=True)
        weight_path = reranker_dir / "group_sample_weights.npy"
        np.save(weight_path, group_weights)

    n = len(rows)
    summary = {
        "teacher": str(teacher),
        "model": args.model,
        "rows": n,
        "topks": topks,
        "thresholds": thresholds,
        "misses": output_paths,
        "pool": {
            str(k): {
                "recall": recalls[k] / n if n else 0.0,
                "ev_loss": stats(losses[k]),
                "threshold_count": len(misses[k]),
            }
            for k in topks
        },
        "by_position": {
            pos: {
                "rows": bucket["rows"],
                "pool": {
                    str(k): {
                        "recall": bucket["recalls"][k] / bucket["rows"] if bucket["rows"] else 0.0,
                        "ev_loss": stats(bucket["losses"][k]),
                        "threshold_count": bucket["miss_counts"][k],
                    }
                    for k in topks
                },
            }
            for pos, bucket in by_position.items()
        },
        "group_weights": {
            "path": str(weight_path) if weight_path else None,
            "mean": float(group_weights.mean()) if len(group_weights) else 0.0,
            "max": float(group_weights.max()) if len(group_weights) else 0.0,
            "weighted_groups": int(np.sum(group_weights > 1.0)),
        },
    }
    write_jsonl(output_dir / "summary.jsonl", [summary])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mine T3 EV-loss misses from exact teacher data")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model", required=True, help="name=bb_path,btn_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reranker-data", default="")
    parser.add_argument("--topks", default="3,5,8,10,15,20")
    parser.add_argument("--min-loss", type=float, default=0.1)
    parser.add_argument("--thresholds", default="5:0.25,8:0.1,10:0.1,15:0.05,20:0.01")
    parser.add_argument("--weight-topk", type=int, default=10)
    parser.add_argument("--weight-min-loss", type=float, default=0.1)
    parser.add_argument("--loss-scale", type=float, default=2.0)
    parser.add_argument("--max-group-weight", type=float, default=12.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(mine(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
