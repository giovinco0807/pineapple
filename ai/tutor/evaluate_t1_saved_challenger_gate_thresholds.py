"""Replay saved T1 final-challenger gate probabilities over runtime results.

This is a lightweight audit for a completed challenger run.  It compares a
baseline runtime result JSONL against a challenger JSONL that already contains
``t1_final_challenger_details.gate_probability``.  For each threshold it keeps
the baseline action unless the saved challenger gate was applied with
probability at least that threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.evaluate_t1_runtime_rule_grid import (  # noqa: E402
    Decision,
    build_decisions,
    parse_set,
    result_score,
    summarize_policy,
)


def gate_probability(row: dict[str, Any]) -> float:
    details = row.get("t1_final_challenger_details") or {}
    if not isinstance(details, dict):
        return float("-inf")
    try:
        return float(details.get("gate_probability", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def saved_gate_switch(decision: Decision, threshold: float) -> bool:
    challenger = decision.challenger
    if not bool(challenger.get("t1_final_challenger_applied")):
        return False
    if int(decision.current.get("best_action_idx", -1)) == int(challenger.get("best_action_idx", -2)):
        return False
    return gate_probability(challenger) >= threshold


def threshold_candidates(decisions_by_set: dict[str, list[Decision]]) -> list[float]:
    probs = {
        gate_probability(decision.challenger)
        for decisions in decisions_by_set.values()
        for decision in decisions
        if bool(decision.challenger.get("t1_final_challenger_applied"))
    }
    probs = {prob for prob in probs if prob != float("-inf")}
    return sorted({0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.000001, *probs})


def summarize_threshold(decisions_by_set: dict[str, list[Decision]], threshold: float) -> dict[str, Any]:
    return {
        name: summarize_policy(decisions, [saved_gate_switch(decision, threshold) for decision in decisions])
        for name, decisions in decisions_by_set.items()
    }


def score_threshold(
    by_set: dict[str, Any],
    *,
    guard_sets: set[str],
    target_sets: set[str],
    regret_tolerance: float,
    top1_tolerance: float,
) -> tuple[bool, tuple[float, float, float, float]]:
    guard_ok = True
    max_guard_regret = float("-inf")
    min_guard_top1 = float("inf")
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
    parser = argparse.ArgumentParser(description="Replay saved T1 challenger gate thresholds")
    parser.add_argument("--set", action="append", required=True, help="NAME=CURRENT_RESULTS::CHALLENGER_RESULTS")
    parser.add_argument("--guard-sets", default="")
    parser.add_argument("--target-sets", default="")
    parser.add_argument("--regret-tolerance", type=float, default=0.0)
    parser.add_argument("--top1-tolerance", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    parsed_sets = [parse_set(raw) for raw in args.set]
    decisions_by_set = {
        name: build_decisions(current, challenger)
        for name, current, challenger in parsed_sets
    }
    set_names = set(decisions_by_set)
    guard_sets = {part.strip() for part in args.guard_sets.split(",") if part.strip()}
    target_sets = {part.strip() for part in args.target_sets.split(",") if part.strip()}
    if not target_sets:
        target_sets = set_names - guard_sets
    if not guard_sets:
        guard_sets = set_names - target_sets

    candidates: list[dict[str, Any]] = []
    for threshold in threshold_candidates(decisions_by_set):
        by_set = summarize_threshold(decisions_by_set, threshold)
        guard_ok, score = score_threshold(
            by_set,
            guard_sets=guard_sets,
            target_sets=target_sets,
            regret_tolerance=args.regret_tolerance,
            top1_tolerance=args.top1_tolerance,
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "guard_ok": guard_ok,
                "score": score,
                "by_set": by_set,
            }
        )
    selected = sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)[0]
    baseline = {
        name: {
            "current": summarize_policy(decisions, [False for _ in decisions]),
            "challenger": summarize_policy(decisions, [True for _ in decisions]),
        }
        for name, decisions in decisions_by_set.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "sets": {name: {"current": str(current), "challenger": str(challenger)} for name, current, challenger in parsed_sets},
        "guard_sets": sorted(guard_sets),
        "target_sets": sorted(target_sets),
        "baseline": baseline,
        "selected_threshold": selected,
        "candidates": candidates,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_threshold": selected}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
