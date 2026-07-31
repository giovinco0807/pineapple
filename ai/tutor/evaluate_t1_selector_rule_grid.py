"""Grid-search simple arbitration rules between two T1 final selectors."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.evaluate_t1_selector_arbitration import (  # noqa: E402
    DEFAULT_DELTA_FEATURES,
    build_decisions,
    iter_json,
    summarize_policy,
)


def parse_set(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--set must be NAME=PATH")
    name, path = raw.split("=", 1)
    return name, Path(path)


def quantile_thresholds(values: list[float], max_thresholds: int) -> list[float]:
    values = sorted(set(float(value) for value in values))
    if len(values) <= max_thresholds:
        return values
    out = set()
    for i in range(max_thresholds):
        idx = round(i * (len(values) - 1) / max(max_thresholds - 1, 1))
        out.add(values[int(idx)])
    return sorted(out)


def build_conditions(decisions_by_set: dict[str, list[Any]], features: list[str], max_thresholds: int) -> list[dict[str, Any]]:
    decisions = [decision for rows in decisions_by_set.values() for decision in rows]
    out: list[dict[str, Any]] = []
    for feature in features:
        values = [float(decision.features.get(feature, 0.0)) for decision in decisions]
        for threshold in quantile_thresholds(values, max_thresholds):
            out.append({"feature": feature, "direction": "ge", "threshold": threshold})
            out.append({"feature": feature, "direction": "le", "threshold": threshold})
    return out


def condition_match(decision: Any, condition: dict[str, Any]) -> bool:
    value = float(decision.features.get(str(condition["feature"]), 0.0))
    threshold = float(condition["threshold"])
    if condition["direction"] == "ge":
        return value >= threshold
    return value <= threshold


def summarize_rule(decisions_by_set: dict[str, list[Any]], conditions: list[dict[str, Any]]) -> dict[str, Any]:
    by_set: dict[str, Any] = {}
    for name, decisions in decisions_by_set.items():
        switches = [
            all(condition_match(decision, condition) for condition in conditions)
            for decision in decisions
        ]
        by_set[name] = summarize_policy(decisions, switches)
    return by_set


def delta(summary: dict[str, Any], field: str) -> float:
    return float(summary[field]) - float(summary[f"current_{field}"])


def rule_score(
    by_set: dict[str, Any],
    *,
    guard_sets: set[str],
    target_sets: set[str],
    regret_tolerance: float,
    top1_tolerance: float,
) -> tuple[bool, tuple[float, float, float, float]]:
    guard_ok = True
    max_guard_regret = 0.0
    min_guard_top1 = 0.0
    for name in guard_sets:
        row = by_set[name]
        regret_delta = float(row["regret_delta_vs_current"])
        top1_delta = float(row["top1_delta_vs_current"])
        max_guard_regret = max(max_guard_regret, regret_delta)
        min_guard_top1 = min(min_guard_top1, top1_delta)
        if regret_delta > regret_tolerance or top1_delta < -top1_tolerance:
            guard_ok = False

    target_top1 = sum(float(by_set[name]["top1_delta_vs_current"]) for name in target_sets)
    target_regret = -sum(float(by_set[name]["regret_delta_vs_current"]) for name in target_sets)
    switch_penalty = -sum(float(row["switch_rate"]) for row in by_set.values()) * 0.01
    return guard_ok, (target_top1, target_regret, -max_guard_regret, min_guard_top1 + switch_penalty)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grid-search T1 selector arbitration rules")
    parser.add_argument("--current-selector", required=True)
    parser.add_argument("--challenger-selector", required=True)
    parser.add_argument("--set", action="append", required=True)
    parser.add_argument("--guard-sets", default="")
    parser.add_argument("--target-sets", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_DELTA_FEATURES))
    parser.add_argument("--max-thresholds", type=int, default=9)
    parser.add_argument("--max-single-candidates", type=int, default=40)
    parser.add_argument("--regret-tolerance", type=float, default=0.0)
    parser.add_argument("--top1-tolerance", type=float, default=0.0)
    parser.add_argument("--min-switches", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    current_selector = iter_json(Path(args.current_selector))
    challenger_selector = iter_json(Path(args.challenger_selector))
    set_paths = dict(parse_set(raw) for raw in args.set)
    guard_sets = {part.strip() for part in args.guard_sets.split(",") if part.strip()}
    target_sets = {part.strip() for part in args.target_sets.split(",") if part.strip()}
    if not guard_sets:
        guard_sets = set(set_paths) - target_sets
    if not target_sets:
        target_sets = set(set_paths) - guard_sets
    features = [part.strip() for part in args.features.split(",") if part.strip()]

    decisions_by_set = {
        name: build_decisions(
            [path],
            current_selector=current_selector,
            challenger_selector=challenger_selector,
            min_teacher_margin=0.0,
        )
        for name, path in set_paths.items()
    }
    conditions = build_conditions(decisions_by_set, features, args.max_thresholds)

    candidates: list[dict[str, Any]] = []
    for condition in conditions:
        by_set = summarize_rule(decisions_by_set, [condition])
        total_switched = sum(int(row["switched"]) for row in by_set.values())
        if total_switched < args.min_switches:
            continue
        ok, score = rule_score(
            by_set,
            guard_sets=guard_sets,
            target_sets=target_sets,
            regret_tolerance=args.regret_tolerance,
            top1_tolerance=args.top1_tolerance,
        )
        candidates.append({"conditions": [condition], "guard_ok": ok, "score": score, "by_set": by_set})

    seed_conditions = [
        row["conditions"][0]
        for row in sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)[
            : max(1, int(args.max_single_candidates))
        ]
    ]
    for first, second in combinations(seed_conditions, 2):
        by_set = summarize_rule(decisions_by_set, [first, second])
        total_switched = sum(int(row["switched"]) for row in by_set.values())
        if total_switched < args.min_switches:
            continue
        ok, score = rule_score(
            by_set,
            guard_sets=guard_sets,
            target_sets=target_sets,
            regret_tolerance=args.regret_tolerance,
            top1_tolerance=args.top1_tolerance,
        )
        candidates.append({"conditions": [first, second], "guard_ok": ok, "score": score, "by_set": by_set})

    sorted_candidates = sorted(
        candidates,
        key=lambda row: (row["guard_ok"], row["score"]),
        reverse=True,
    )
    report = {
        "current_selector": args.current_selector,
        "challenger_selector": args.challenger_selector,
        "sets": {name: str(path) for name, path in set_paths.items()},
        "guard_sets": sorted(guard_sets),
        "target_sets": sorted(target_sets),
        "features": features,
        "top_rules": sorted_candidates[:25],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
