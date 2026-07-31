"""Inspect T1 runtime misses at candidate-row level.

The hybrid evaluator writes one row per decision to ``results.jsonl`` and one
row per candidate to ``selector_rows.jsonl``.  This script joins them so T1
misses can be reviewed as concrete chosen-vs-teacher action pairs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_selector_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        if int(row.get("turn", -1)) == 1:
            grouped[int(row.get("line", 0))].append(row)
    return dict(grouped)


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if bool(result.get("final_top1_hit")):
        return "hit"
    if result.get("teacher_best_in_pool") is False:
        return "pool_miss"
    if result.get("teacher_best_in_sync") is False:
        return "sync_miss"
    if not any(bool(row.get("is_teacher_best")) and bool(row.get("is_refined")) for row in rows):
        return "not_refined"
    return "selection_miss_after_refine"


def pick_row(rows: list[dict[str, Any]], flag: str) -> dict[str, Any] | None:
    for row in rows:
        if bool(row.get(flag)):
            return row
    return None


def pick_refined_top(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    refined = [row for row in rows if bool(row.get("is_refined"))]
    if not refined:
        return None
    return max(
        refined,
        key=lambda row: (
            fnum(row.get("refined_score"), float("-inf")),
            fnum(row.get("model_score"), float("-inf")),
            -int(row.get("model_rank", 999999)),
        ),
    )


def row_view(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = [
        "action_idx",
        "action",
        "output_rank",
        "model_rank",
        "refinement_rank",
        "refined_rank",
        "teacher_rank",
        "teacher_score",
        "model_score",
        "sync_selector_score",
        "final_selector_score",
        "refined_score",
        "predicted_bust",
        "predicted_fl",
        "predicted_aa",
        "predicted_kk",
        "predicted_qq",
        "predicted_trips",
        "is_sync",
        "is_refined",
        "samples",
        "elapsed_ms",
    ]
    return {key: row.get(key) for key in keys}


def delta(chosen: dict[str, Any] | None, teacher: dict[str, Any] | None, key: str) -> float | None:
    if chosen is None or teacher is None:
        return None
    left = fnum(chosen.get(key))
    right = fnum(teacher.get(key))
    if left is None or right is None:
        return None
    return left - right


def selection_failure_type(chosen: dict[str, Any] | None, teacher: dict[str, Any] | None) -> str:
    if chosen is None:
        return "missing_runtime_row"
    if teacher is None:
        return "teacher_best_not_in_rows"
    chosen_refined = fnum(chosen.get("refined_score"))
    teacher_refined = fnum(teacher.get("refined_score"))
    if teacher_refined is None:
        return "teacher_best_not_refined"
    if chosen_refined is not None and chosen_refined >= teacher_refined:
        return "refinement_prefers_chosen"
    chosen_final = fnum(chosen.get("final_selector_score"))
    teacher_final = fnum(teacher.get("final_selector_score"))
    if chosen_final is not None and teacher_final is not None and chosen_final >= teacher_final:
        return "selector_prefers_chosen_despite_lower_refine"
    return "unexpected_selector_choice"


def build_record(result: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = pick_row(rows, "is_runtime_best")
    teacher = pick_row(rows, "is_teacher_best")
    model_top1 = pick_row(rows, "is_model_top1")
    refined_top1 = pick_refined_top(rows)
    reason = classify(result, rows)
    regret = max(
        0.0,
        float(result.get("teacher_best_score", 0.0) or 0.0)
        - float(result.get("final_teacher_score", 0.0) or 0.0),
    )
    return {
        "line": result.get("line"),
        "source": result.get("source"),
        "source_line": result.get("source_line"),
        "reason": reason,
        "selection_failure_type": selection_failure_type(chosen, teacher)
        if reason == "selection_miss_after_refine"
        else "",
        "regret": regret,
        "teacher_margin": result.get("teacher_margin"),
        "teacher_best_score": result.get("teacher_best_score"),
        "final_teacher_score": result.get("final_teacher_score"),
        "model_teacher_score": result.get("model_teacher_score"),
        "elapsed_ms": result.get("elapsed_ms"),
        "exact_evaluated": result.get("exact_evaluated"),
        "teacher_best_in_pool": result.get("teacher_best_in_pool"),
        "teacher_best_in_sync": result.get("teacher_best_in_sync"),
        "board": result.get("board"),
        "opponent_board": result.get("opponent_board"),
        "dealt": result.get("dealt"),
        "teacher_best_action": result.get("teacher_best_action"),
        "model_action": result.get("model_action"),
        "final_action": result.get("final_action"),
        "chosen_row": row_view(chosen),
        "teacher_row": row_view(teacher),
        "model_top1_row": row_view(model_top1),
        "refined_top1_row": row_view(refined_top1),
        "deltas_chosen_minus_teacher": {
            "teacher_score": delta(chosen, teacher, "teacher_score"),
            "model_score": delta(chosen, teacher, "model_score"),
            "sync_selector_score": delta(chosen, teacher, "sync_selector_score"),
            "final_selector_score": delta(chosen, teacher, "final_selector_score"),
            "refined_score": delta(chosen, teacher, "refined_score"),
            "predicted_bust": delta(chosen, teacher, "predicted_bust"),
            "predicted_fl": delta(chosen, teacher, "predicted_fl"),
            "predicted_aa": delta(chosen, teacher, "predicted_aa"),
            "predicted_kk": delta(chosen, teacher, "predicted_kk"),
            "predicted_qq": delta(chosen, teacher, "predicted_qq"),
            "predicted_trips": delta(chosen, teacher, "predicted_trips"),
        },
    }


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    reason_stats: dict[str, dict[str, Any]] = {}
    failure_type_counts: dict[str, int] = defaultdict(int)
    sync_miss_teacher_model_ranks: list[int] = []
    for record in records:
        reason = str(record["reason"])
        regret = float(record["regret"])
        stats = reason_stats.setdefault(reason, {"count": 0, "sum_regret": 0.0, "max_regret": 0.0})
        stats["count"] += 1
        stats["sum_regret"] += regret
        stats["max_regret"] = max(float(stats["max_regret"]), regret)
        if reason == "selection_miss_after_refine":
            failure_type_counts[str(record["selection_failure_type"])] += 1
        if reason == "sync_miss" and record.get("teacher_row"):
            rank = record["teacher_row"].get("model_rank")
            if rank is not None:
                sync_miss_teacher_model_ranks.append(int(rank))
    for stats in reason_stats.values():
        stats["avg_regret"] = stats["sum_regret"] / max(int(stats["count"]), 1)
    return {
        "results": args.results,
        "selector_rows": args.selector_rows,
        "decisions": len(records),
        "reason_stats": dict(sorted(reason_stats.items())),
        "selection_failure_type_counts": dict(sorted(failure_type_counts.items())),
        "sync_miss_teacher_model_rank": {
            "count": len(sync_miss_teacher_model_ranks),
            "min": min(sync_miss_teacher_model_ranks) if sync_miss_teacher_model_ranks else None,
            "max": max(sync_miss_teacher_model_ranks) if sync_miss_teacher_model_ranks else None,
            "ranks": sorted(sync_miss_teacher_model_ranks),
        },
        "top_regret": [
            {
                "line": record["line"],
                "reason": record["reason"],
                "failure_type": record["selection_failure_type"],
                "regret": record["regret"],
                "teacher_margin": record["teacher_margin"],
                "dealt": record["dealt"],
                "teacher_best_action": record["teacher_best_action"],
                "final_action": record["final_action"],
            }
            for record in sorted(records, key=lambda item: float(item["regret"]), reverse=True)[: args.top_n]
            if record["reason"] != "hit"
        ],
    }


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    rows_by_line = group_selector_rows(Path(args.selector_rows))
    miss_details = (
        {int(record.get("line", 0)): record for record in iter_jsonl(Path(args.misses))}
        if args.misses
        else {}
    )
    records = []
    for result in iter_jsonl(Path(args.results)):
        line_no = int(result.get("line", 0))
        enriched = {**miss_details.get(line_no, {}), **result}
        records.append(build_record(enriched, rows_by_line.get(line_no, [])))
    misses = [record for record in records if record["reason"] != "hit"]
    output_pairs = Path(args.output_pairs)
    output_misses = Path(args.output_misses)
    output_summary = Path(args.output_summary)
    output_pairs.parent.mkdir(parents=True, exist_ok=True)
    with output_pairs.open("w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda item: (item["reason"] == "hit", -float(item["regret"]))):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with output_misses.open("w", encoding="utf-8") as f:
        for record in sorted(misses, key=lambda item: -float(item["regret"])):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, args)
    summary["output_pairs"] = str(output_pairs)
    summary["output_misses"] = str(output_misses)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect T1 runtime misses")
    parser.add_argument("--results", required=True)
    parser.add_argument("--misses", default="")
    parser.add_argument("--selector-rows", required=True)
    parser.add_argument("--output-pairs", required=True)
    parser.add_argument("--output-misses", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(inspect(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
