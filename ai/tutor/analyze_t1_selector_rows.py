"""Analyze candidate-level T1 selector rows.

The input is ``selector_rows.jsonl`` produced by
``evaluate_hybrid_refinement_teacher.py --write-selector-rows``.  The report
separates candidate coverage from selection quality, which is necessary before
training a better T1 selector.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("turn", -1)) != 1:
            continue
        grouped[int(row.get("line", 0))].append(row)
    return dict(grouped)


def score(row: dict[str, Any], name: str, default: float = float("-inf")) -> float:
    value = row.get(name)
    return default if value is None else float(value)


def teacher_best_score(group: list[dict[str, Any]]) -> float:
    if not group:
        return 0.0
    return float(group[0].get("teacher_best_score", 0.0) or 0.0)


def pick_model(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    top = [row for row in group if bool(row.get("is_model_top1"))]
    if top:
        return top[0]
    return min(group, key=lambda row: int(row.get("model_rank", 999999)), default=None)


def pick_runtime(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    top = [row for row in group if bool(row.get("is_runtime_best"))]
    return top[0] if top else pick_model(group)


def pick_refined(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    refined = [row for row in group if bool(row.get("is_refined"))]
    if not refined:
        return pick_model(group)
    return max(refined, key=lambda row: (score(row, "refined_score"), score(row, "model_score")))


def pick_sync_model(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    sync = [row for row in group if bool(row.get("is_sync"))]
    if not sync:
        return pick_model(group)
    return max(sync, key=lambda row: score(row, "model_score"))


def pick_pool_model(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(group, key=lambda row: score(row, "model_score"), default=None)


def pick_blend(group: list[dict[str, Any]], weight: float) -> dict[str, Any] | None:
    refined = [row for row in group if bool(row.get("is_refined"))]
    if not refined:
        return pick_model(group)
    return max(
        refined,
        key=lambda row: (
            score(row, "refined_score", 0.0) + float(weight) * score(row, "model_score", 0.0),
            -int(row.get("model_rank", 999999)),
        ),
    )


def evaluate_policy(groups: dict[int, list[dict[str, Any]]], picker: Callable[[list[dict[str, Any]]], dict[str, Any] | None]) -> dict[str, Any]:
    hits = 0
    regret = 0.0
    max_regret = 0.0
    selected = 0
    for group in groups.values():
        row = picker(group)
        if row is None:
            continue
        selected += 1
        if bool(row.get("is_teacher_best")):
            hits += 1
        row_score = row.get("teacher_score")
        row_regret = 0.0 if row_score is None else max(0.0, teacher_best_score(group) - float(row_score))
        regret += row_regret
        max_regret = max(max_regret, row_regret)
    denom = max(len(groups), 1)
    return {
        "decisions": len(groups),
        "selected": selected,
        "top1": hits / denom,
        "hits": hits,
        "avg_regret": regret / denom,
        "max_regret": max_regret,
    }


def coverage(groups: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    pool_hits = 0
    sync_hits = 0
    refined_hits = 0
    for group in groups.values():
        if any(bool(row.get("is_teacher_best")) for row in group):
            pool_hits += 1
        if any(bool(row.get("is_teacher_best")) and bool(row.get("is_sync")) for row in group):
            sync_hits += 1
        if any(bool(row.get("is_teacher_best")) and bool(row.get("is_refined")) for row in group):
            refined_hits += 1
    denom = max(len(groups), 1)
    return {
        "decisions": len(groups),
        "pool_oracle_top1": pool_hits / denom,
        "sync_oracle_top1": sync_hits / denom,
        "refined_oracle_top1": refined_hits / denom,
        "pool_hits": pool_hits,
        "sync_hits": sync_hits,
        "refined_hits": refined_hits,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    groups = group_rows(iter_jsonl(Path(args.input)))
    policies: dict[str, Any] = {
        "model_top1": evaluate_policy(groups, pick_model),
        "runtime_best": evaluate_policy(groups, pick_runtime),
        "refined_score_top1": evaluate_policy(groups, pick_refined),
        "sync_model_score_top1": evaluate_policy(groups, pick_sync_model),
        "pool_model_score_top1": evaluate_policy(groups, pick_pool_model),
    }
    for raw in args.blend_weights.split(","):
        if not raw.strip():
            continue
        weight = float(raw)
        policies[f"refined_plus_{weight:g}x_model"] = evaluate_policy(
            groups,
            lambda group, weight=weight: pick_blend(group, weight),
        )
    best_policy = max(
        policies.items(),
        key=lambda item: (item[1]["top1"], -item[1]["avg_regret"], -item[1]["max_regret"]),
    )
    report = {
        "input": args.input,
        "coverage": coverage(groups),
        "policies": policies,
        "best_policy": {
            "name": best_policy[0],
            **best_policy[1],
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate-level T1 selector rows")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--blend-weights", default="-1,-0.5,-0.25,0,0.25,0.5,1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(analyze(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
