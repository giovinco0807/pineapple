"""Grid-search simple gates between saved T1 selector-row policies.

This is cheaper than rerunning the full hybrid evaluator.  It replays
``selector_rows.jsonl`` files and asks whether a challenger policy such as
``refined_plus_model`` should replace the saved runtime-best action for each
decision.  Promote a gate into runtime only after this analysis is followed by
full runtime validation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

from ai.tutor.analyze_t1_selector_rows import group_rows, iter_jsonl, score


DEFAULT_FEATURES = [
    "same_action",
    "refined_score_delta",
    "model_score_delta",
    "model_raw_score_delta",
    "model_rank_delta",
    "refined_rank_delta",
    "predicted_bust_delta",
    "predicted_fl_delta",
    "predicted_aa_delta",
    "predicted_kk_delta",
    "predicted_qq_delta",
    "predicted_trips_delta",
    "premium_fl_delta",
    "candidate_fl_delta",
    "samples_delta",
    "current_predicted_bust",
    "challenger_predicted_bust",
    "current_predicted_fl",
    "challenger_predicted_fl",
    "current_model_rank",
    "challenger_model_rank",
    "current_refined_rank",
    "challenger_refined_rank",
    "current_is_model_top1",
    "challenger_is_model_top1",
    "current_is_refined",
    "challenger_is_refined",
    "refined_rows",
    "teacher_best_in_sync",
    "teacher_best_is_refined",
]


@dataclass(frozen=True)
class Decision:
    key: int
    current: dict[str, Any]
    challenger: dict[str, Any]
    best_score: float
    features: dict[str, float]


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rank_value(row: dict[str, Any], key: str) -> float:
    return max(fnum(row, key, 999.0), 1.0)


def row_teacher_score(row: dict[str, Any]) -> float:
    return fnum(row, "teacher_score", fnum(row, "teacher_best_score", 0.0))


def teacher_best_score(group: list[dict[str, Any]]) -> float:
    if not group:
        return 0.0
    return fnum(group[0], "teacher_best_score", 0.0)


def premium_fl(row: dict[str, Any]) -> float:
    return fnum(row, "predicted_aa") + fnum(row, "predicted_kk") + fnum(row, "predicted_trips")


def candidate_fl(row: dict[str, Any]) -> float:
    return max(
        fnum(row, "predicted_fl"),
        fnum(row, "predicted_aa"),
        fnum(row, "predicted_kk"),
        fnum(row, "predicted_qq"),
        fnum(row, "predicted_trips"),
    )


def pick_runtime(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    runtime = [row for row in group if bool(row.get("is_runtime_best"))]
    if runtime:
        return runtime[0]
    model = [row for row in group if bool(row.get("is_model_top1"))]
    if model:
        return model[0]
    return min(group, key=lambda row: int(row.get("model_rank", 999999)), default=None)


def pick_model(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    model = [row for row in group if bool(row.get("is_model_top1"))]
    if model:
        return model[0]
    return min(group, key=lambda row: int(row.get("model_rank", 999999)), default=None)


def pick_refined(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    refined = [row for row in group if bool(row.get("is_refined"))]
    if not refined:
        return pick_model(group)
    return max(refined, key=lambda row: (score(row, "refined_score"), score(row, "model_score")))


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


def picker_for(name: str, *, blend_weight: float) -> Callable[[list[dict[str, Any]]], dict[str, Any] | None]:
    if name == "runtime_best":
        return pick_runtime
    if name == "model_top1":
        return pick_model
    if name == "refined_score":
        return pick_refined
    if name == "refined_plus_model":
        return lambda group: pick_blend(group, blend_weight)
    raise ValueError(f"unsupported policy: {name}")


def build_features(group: list[dict[str, Any]], current: dict[str, Any], challenger: dict[str, Any]) -> dict[str, float]:
    teacher_best = next((row for row in group if bool(row.get("is_teacher_best"))), None)
    out = {
        "same_action": 1.0
        if int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
        else 0.0,
        "refined_score_delta": fnum(challenger, "refined_score") - fnum(current, "refined_score"),
        "model_score_delta": fnum(challenger, "model_score") - fnum(current, "model_score"),
        "model_raw_score_delta": fnum(challenger, "model_raw_score") - fnum(current, "model_raw_score"),
        "model_rank_delta": rank_value(current, "model_rank") - rank_value(challenger, "model_rank"),
        "refined_rank_delta": rank_value(current, "refined_rank") - rank_value(challenger, "refined_rank"),
        "predicted_bust_delta": fnum(challenger, "predicted_bust") - fnum(current, "predicted_bust"),
        "predicted_fl_delta": fnum(challenger, "predicted_fl") - fnum(current, "predicted_fl"),
        "predicted_aa_delta": fnum(challenger, "predicted_aa") - fnum(current, "predicted_aa"),
        "predicted_kk_delta": fnum(challenger, "predicted_kk") - fnum(current, "predicted_kk"),
        "predicted_qq_delta": fnum(challenger, "predicted_qq") - fnum(current, "predicted_qq"),
        "predicted_trips_delta": fnum(challenger, "predicted_trips") - fnum(current, "predicted_trips"),
        "premium_fl_delta": premium_fl(challenger) - premium_fl(current),
        "candidate_fl_delta": candidate_fl(challenger) - candidate_fl(current),
        "samples_delta": fnum(challenger, "samples") - fnum(current, "samples"),
        "current_predicted_bust": fnum(current, "predicted_bust"),
        "challenger_predicted_bust": fnum(challenger, "predicted_bust"),
        "current_predicted_fl": fnum(current, "predicted_fl"),
        "challenger_predicted_fl": fnum(challenger, "predicted_fl"),
        "current_model_rank": rank_value(current, "model_rank"),
        "challenger_model_rank": rank_value(challenger, "model_rank"),
        "current_refined_rank": rank_value(current, "refined_rank"),
        "challenger_refined_rank": rank_value(challenger, "refined_rank"),
        "current_is_model_top1": 1.0 if bool(current.get("is_model_top1")) else 0.0,
        "challenger_is_model_top1": 1.0 if bool(challenger.get("is_model_top1")) else 0.0,
        "current_is_refined": 1.0 if bool(current.get("is_refined")) else 0.0,
        "challenger_is_refined": 1.0 if bool(challenger.get("is_refined")) else 0.0,
        "refined_rows": float(sum(1 for row in group if bool(row.get("is_refined")))),
        "teacher_best_in_sync": 1.0
        if any(bool(row.get("is_teacher_best")) and bool(row.get("is_sync")) for row in group)
        else 0.0,
        "teacher_best_is_refined": 1.0
        if teacher_best is not None and bool(teacher_best.get("is_refined"))
        else 0.0,
    }
    return out


def build_decisions(
    path: Path,
    *,
    current_policy: str,
    challenger_policy: str,
    blend_weight: float,
) -> list[Decision]:
    groups = group_rows(iter_jsonl(path))
    current_picker = picker_for(current_policy, blend_weight=blend_weight)
    challenger_picker = picker_for(challenger_policy, blend_weight=blend_weight)
    decisions: list[Decision] = []
    for key, group in sorted(groups.items()):
        current = current_picker(group)
        challenger = challenger_picker(group)
        if current is None or challenger is None:
            continue
        decisions.append(
            Decision(
                key=key,
                current=current,
                challenger=challenger,
                best_score=teacher_best_score(group),
                features=build_features(group, current, challenger),
            )
        )
    return decisions


def summarize_policy(decisions: list[Decision], switches: list[bool]) -> dict[str, Any]:
    if len(decisions) != len(switches):
        raise ValueError("decision/switch length mismatch")
    rows = len(decisions)
    current_hits = 0
    challenger_hits = 0
    hits = 0
    current_regret = 0.0
    challenger_regret = 0.0
    regret = 0.0
    switched = 0
    switch_gain = 0.0
    bad_switch_regret = 0.0
    missed_good_switches = 0
    same_action = 0
    for decision, switch in zip(decisions, switches):
        current_score = row_teacher_score(decision.current)
        challenger_score = row_teacher_score(decision.challenger)
        current_hit = bool(decision.current.get("is_teacher_best"))
        challenger_hit = bool(decision.challenger.get("is_teacher_best"))
        if current_hit:
            current_hits += 1
        if challenger_hit:
            challenger_hits += 1
        if int(decision.current.get("action_idx", -1)) == int(decision.challenger.get("action_idx", -2)):
            same_action += 1
        current_regret += max(0.0, decision.best_score - current_score)
        challenger_regret += max(0.0, decision.best_score - challenger_score)
        if switch:
            switched += 1
            chosen_score = challenger_score
            chosen_hit = challenger_hit
            gain = challenger_score - current_score
            switch_gain += gain
            if gain < 0.0:
                bad_switch_regret += -gain
        else:
            chosen_score = current_score
            chosen_hit = current_hit
            if challenger_score > current_score:
                missed_good_switches += 1
        if chosen_hit:
            hits += 1
        regret += max(0.0, decision.best_score - chosen_score)
    denom = max(rows, 1)
    return {
        "decisions": rows,
        "switched": switched,
        "switch_rate": switched / denom,
        "same_action": same_action,
        "top1": hits / denom,
        "hits": hits,
        "avg_regret": regret / denom,
        "current_top1": current_hits / denom,
        "current_hits": current_hits,
        "current_avg_regret": current_regret / denom,
        "challenger_top1": challenger_hits / denom,
        "challenger_hits": challenger_hits,
        "challenger_avg_regret": challenger_regret / denom,
        "top1_delta_vs_current": (hits - current_hits) / denom,
        "regret_delta_vs_current": regret / denom - current_regret / denom,
        "switch_gain": switch_gain,
        "bad_switch_regret": bad_switch_regret,
        "missed_good_switches": missed_good_switches,
    }


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


def build_conditions(decisions_by_set: dict[str, list[Decision]], features: list[str], max_thresholds: int) -> list[dict[str, Any]]:
    decisions = [decision for rows in decisions_by_set.values() for decision in rows]
    out: list[dict[str, Any]] = []
    for feature in features:
        values = [float(decision.features.get(feature, 0.0)) for decision in decisions]
        for threshold in quantile_thresholds(values, max_thresholds):
            out.append({"feature": feature, "direction": "ge", "threshold": threshold})
            out.append({"feature": feature, "direction": "le", "threshold": threshold})
    return out


def condition_match(decision: Decision, condition: dict[str, Any]) -> bool:
    value = float(decision.features.get(str(condition["feature"]), 0.0))
    threshold = float(condition["threshold"])
    if condition["direction"] == "ge":
        return value >= threshold
    return value <= threshold


def summarize_rule(decisions_by_set: dict[str, list[Decision]], conditions: list[dict[str, Any]]) -> dict[str, Any]:
    by_set: dict[str, Any] = {}
    for name, decisions in decisions_by_set.items():
        switches = [all(condition_match(decision, condition) for condition in conditions) for decision in decisions]
        by_set[name] = summarize_policy(decisions, switches)
    return by_set


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
    parser = argparse.ArgumentParser(description="Grid-search gates between selector-row policies")
    parser.add_argument("--set", action="append", required=True, help="NAME=SELECTOR_ROWS_JSONL")
    parser.add_argument("--current-policy", default="runtime_best")
    parser.add_argument("--challenger-policy", default="refined_plus_model")
    parser.add_argument("--blend-weight", type=float, default=1.0)
    parser.add_argument("--guard-sets", default="")
    parser.add_argument("--target-sets", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--max-thresholds", type=int, default=9)
    parser.add_argument("--max-single-candidates", type=int, default=40)
    parser.add_argument("--regret-tolerance", type=float, default=0.0)
    parser.add_argument("--top1-tolerance", type=float, default=0.0)
    parser.add_argument("--min-switches", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    parsed_sets = [parse_set(raw) for raw in args.set]
    guard_sets = {part.strip() for part in args.guard_sets.split(",") if part.strip()}
    target_sets = {part.strip() for part in args.target_sets.split(",") if part.strip()}
    set_names = {name for name, _path in parsed_sets}
    if not guard_sets:
        guard_sets = set_names - target_sets
    if not target_sets:
        target_sets = set_names - guard_sets
    features = [part.strip() for part in args.features.split(",") if part.strip()]

    decisions_by_set = {
        name: build_decisions(
            path,
            current_policy=args.current_policy,
            challenger_policy=args.challenger_policy,
            blend_weight=args.blend_weight,
        )
        for name, path in parsed_sets
    }
    conditions = build_conditions(decisions_by_set, features, args.max_thresholds)
    baseline = {
        name: {
            "current": summarize_policy(decisions, [False for _ in decisions]),
            "challenger": summarize_policy(decisions, [True for _ in decisions]),
        }
        for name, decisions in decisions_by_set.items()
    }

    candidates: list[dict[str, Any]] = []
    for condition in conditions:
        by_set = summarize_rule(decisions_by_set, [condition])
        total_switched = sum(int(row["switched"]) for row in by_set.values())
        if total_switched < args.min_switches:
            continue
        ok, score_tuple = rule_score(
            by_set,
            guard_sets=guard_sets,
            target_sets=target_sets,
            regret_tolerance=args.regret_tolerance,
            top1_tolerance=args.top1_tolerance,
        )
        candidates.append({"conditions": [condition], "guard_ok": ok, "score": score_tuple, "by_set": by_set})

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
        ok, score_tuple = rule_score(
            by_set,
            guard_sets=guard_sets,
            target_sets=target_sets,
            regret_tolerance=args.regret_tolerance,
            top1_tolerance=args.top1_tolerance,
        )
        candidates.append({"conditions": [first, second], "guard_ok": ok, "score": score_tuple, "by_set": by_set})

    sorted_candidates = sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)
    report = {
        "sets": {name: str(path) for name, path in parsed_sets},
        "current_policy": args.current_policy,
        "challenger_policy": args.challenger_policy,
        "blend_weight": args.blend_weight,
        "guard_sets": sorted(guard_sets),
        "target_sets": sorted(target_sets),
        "features": features,
        "baseline": baseline,
        "candidate_count": len(candidates),
        "top_rules": sorted_candidates[:25],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
