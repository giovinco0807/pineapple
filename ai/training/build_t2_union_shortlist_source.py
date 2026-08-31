"""Build a T2 source JSONL whose candidates are a selector-union shortlist.

``evaluate_t2_sklearn_candidate_union.py`` writes per-group union membership as
global sample indices.  The Rust T2 exact evaluator can exact-rerank only the
first K source candidates, so this helper rewrites each source row with just the
union members as its candidate list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def candidate_sort_score(candidate: dict[str, Any]) -> float:
    for key in ("model_t3_value_score", "t3_model_ev", "score"):
        value = candidate.get(key)
        if value is not None:
            return float(value)
    return 0.0


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_jsonl(Path(args.source))
    union_rows = load_jsonl(Path(args.union_rows))
    exact_rows = load_jsonl(Path(args.candidate_exact)) if args.candidate_exact else []
    if len(source_rows) != len(union_rows):
        raise ValueError(f"source rows {len(source_rows)} != union rows {len(union_rows)}")
    if exact_rows and len(exact_rows) != len(source_rows):
        raise ValueError(f"candidate exact rows {len(exact_rows)} != source rows {len(source_rows)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool_key = str(args.pool_k)
    global_start = 0
    written = 0
    selected_counts: list[int] = []
    missing_indices = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row_idx, (source, union) in enumerate(zip(source_rows, union_rows, strict=True)):
            candidates = (
                list((exact_rows[row_idx].get("candidates") or []))
                if exact_rows
                else list(source.get("candidates") or [])
            )
            pool = (union.get("pool_metrics") or {}).get(pool_key)
            if pool is None:
                raise ValueError(f"missing pool k={pool_key} in union row {row_idx}")
            global_indices = [int(index) for index in (pool.get("indices") or [])]
            local_indices = [index - global_start for index in global_indices]
            selected: list[dict[str, Any]] = []
            for local_index in local_indices:
                if 0 <= local_index < len(candidates):
                    selected.append(candidates[local_index])
                else:
                    missing_indices += 1
            if args.keep_source_order:
                selected = [candidate for _idx, candidate in sorted(zip(local_indices, selected, strict=True))]
            else:
                selected = sorted(selected, key=candidate_sort_score, reverse=True)
            output_row = dict(source)
            output_row["candidates"] = selected
            output_row["candidate_count"] = len(selected)
            output_row["shortlist_source"] = {
                "kind": "selector_union",
                "pool_k": int(args.pool_k),
                "union_rows": str(Path(args.union_rows)),
                "candidate_exact": str(Path(args.candidate_exact)) if args.candidate_exact else None,
                "original_candidate_count": len(candidates),
                "selected_global_indices": global_indices,
                "selected_local_indices": local_indices,
                "teacher_best_in_pool": bool(pool.get("hit")),
                "teacher_ev_loss_if_reranked": float(pool.get("ev_loss", 0.0) or 0.0),
                "pool_size": int(pool.get("pool_size", len(selected)) or len(selected)),
            }
            out.write(json.dumps(output_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            selected_counts.append(len(selected))
            global_start += len(candidates)

    summary = {
        "source": str(Path(args.source)),
        "union_rows": str(Path(args.union_rows)),
        "candidate_exact": str(Path(args.candidate_exact)) if args.candidate_exact else None,
        "output": str(out_path),
        "pool_k": int(args.pool_k),
        "written": written,
        "missing_indices": missing_indices,
        "total_candidates": int(sum(selected_counts)),
        "avg_candidates": float(sum(selected_counts) / max(len(selected_counts), 1)),
        "max_candidates": int(max(selected_counts) if selected_counts else 0),
        "min_candidates": int(min(selected_counts) if selected_counts else 0),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--union-rows", required=True)
    parser.add_argument("--candidate-exact", help="Optional all-legal exact output whose sorted candidate list matches union indices.")
    parser.add_argument("--pool-k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-source-order", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
