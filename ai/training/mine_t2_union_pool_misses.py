"""Mine T2 selector-union pool misses for hard-negative training.

The union evaluators write one JSONL row per T2 spot with ``pool_metrics`` for
several per-selector K values.  This helper extracts the spots where the teacher
best action is outside a target pool, preserving EV loss, pool size, selector
membership, and the smaller/larger K context needed to decide whether to add
training data or widen the runtime pool.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pool(row: dict[str, Any], k: int) -> dict[str, Any]:
    metrics = row.get("pool_metrics") or {}
    value = metrics.get(str(k))
    if value is None:
        raise KeyError(f"missing pool_metrics[{k}] for {row.get('dataset')}#{row.get('group_id')}")
    return value


def selector_votes(pool_metrics: dict[str, Any], teacher_best_index: int) -> dict[str, bool]:
    return {
        str(name): teacher_best_index in {int(index) for index in indices}
        for name, indices in (pool_metrics.get("by_selector") or {}).items()
    }


def summarize(rows: list[dict[str, Any]], ks: list[int], thresholds: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": len(rows),
        "datasets": dict(Counter(str(row.get("dataset")) for row in rows)),
    }
    by_dataset: dict[str, dict[str, Any]] = defaultdict(dict)
    for k in ks:
        misses = [row for row in rows if not bool(pool(row, k).get("hit"))]
        losses = [float(pool(row, k).get("ev_loss", 0.0) or 0.0) for row in rows]
        miss_losses = [float(pool(row, k).get("ev_loss", 0.0) or 0.0) for row in misses]
        out[f"top{k}"] = {
            "misses": len(misses),
            "recall": 1.0 - len(misses) / max(len(rows), 1),
            "ev_loss_mean": sum(losses) / max(len(losses), 1),
            "ev_loss_max": max(losses) if losses else 0.0,
            "miss_ev_loss_mean": sum(miss_losses) / max(len(miss_losses), 1),
            "miss_ev_loss_max": max(miss_losses) if miss_losses else 0.0,
            "threshold_counts": {
                str(threshold): sum(1 for loss in losses if loss > threshold)
                for threshold in thresholds
            },
        }
        ds_counts: Counter[str] = Counter(str(row.get("dataset")) for row in misses)
        for dataset, count in ds_counts.items():
            by_dataset[dataset][f"top{k}_misses"] = int(count)
    out["by_dataset"] = dict(sorted(by_dataset.items()))
    return out


def mine(args: argparse.Namespace) -> dict[str, Any]:
    rows = list(iter_jsonl(Path(args.rows)))
    target_ks = [int(part) for part in args.target_ks.split(",") if part.strip()]
    context_ks = [int(part) for part in args.context_ks.split(",") if part.strip()]
    thresholds = [float(part) for part in args.thresholds.split(",") if part.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    miss_rows: list[dict[str, Any]] = []
    for row in rows:
        teacher_best_index = int(row["teacher_best_index"])
        for k in target_ks:
            target_pool = pool(row, k)
            ev_loss = float(target_pool.get("ev_loss", 0.0) or 0.0)
            if bool(target_pool.get("hit")) or ev_loss < float(args.min_ev_loss):
                continue
            item = {
                "dataset": row.get("dataset"),
                "group_id": int(row.get("group_id", -1)),
                "size": int(row.get("size", 0)),
                "target_k": int(k),
                "teacher_best_index": teacher_best_index,
                "teacher_best_score": float(row.get("teacher_best_score", 0.0) or 0.0),
                "ev_loss": ev_loss,
                "pool_size": int(target_pool.get("pool_size", 0) or 0),
                "pool_best_ev": float(target_pool.get("pool_best_ev", 0.0) or 0.0),
                "selector_hit_votes": selector_votes(target_pool, teacher_best_index),
                "context": {},
            }
            for ctx_k in context_ks:
                ctx = pool(row, ctx_k)
                item["context"][str(ctx_k)] = {
                    "hit": bool(ctx.get("hit")),
                    "ev_loss": float(ctx.get("ev_loss", 0.0) or 0.0),
                    "pool_size": int(ctx.get("pool_size", 0) or 0),
                }
            miss_rows.append(item)

    miss_rows.sort(key=lambda item: (float(item["ev_loss"]), int(item["target_k"])), reverse=True)
    misses_path = out_dir / "misses.jsonl"
    with misses_path.open("w", encoding="utf-8") as f:
        for row in miss_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "rows_path": str(Path(args.rows)),
        "target_ks": target_ks,
        "context_ks": context_ks,
        "min_ev_loss": float(args.min_ev_loss),
        "summary": summarize(rows, sorted(set(target_ks + context_ks)), thresholds),
        "misses_written": len(miss_rows),
        "misses_path": str(misses_path),
        "top_misses": miss_rows[: min(len(miss_rows), int(args.preview))],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "misses": len(miss_rows)}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="Union evaluator rows.jsonl")
    parser.add_argument("--target-ks", default="5,8,10")
    parser.add_argument("--context-ks", default="1,3,5,8,10,15,20")
    parser.add_argument("--thresholds", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--min-ev-loss", type=float, default=0.0)
    parser.add_argument("--preview", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mine(args)


if __name__ == "__main__":
    main()
