"""Grid-search simple gates between two T1 runtime result sets.

Unlike selector-row replay, this compares completed runtime outputs such as
current final-selector results vs refined_score results.  This keeps override,
rescue, and fallback interactions in the evidence used to select a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


DEFAULT_FEATURES = [
    "same_action",
    "refined_score_delta",
    "refined_override_delta_delta",
    "best_refined_score_delta",
    "model_top1_refined_score_delta",
    "override_gate_probability_delta",
    "current_override_gate_probability",
    "challenger_override_gate_probability",
    "current_model_top1_overridden",
    "challenger_model_top1_overridden",
    "current_override_accepted",
    "challenger_override_accepted",
    "current_refined_delta",
    "challenger_refined_delta",
    "t1_predicted_bust_delta_delta",
    "t1_predicted_fl_delta_delta",
    "t1_predicted_qq_delta_delta",
    "t1_predicted_kk_delta_delta",
    "t1_predicted_aa_delta_delta",
    "t1_premium_fl_delta_sum_delta",
    "t1_model_rank_gap_delta",
    "t1_best_predicted_bust_delta",
    "t1_best_predicted_fl_delta",
    "t1_best_model_score_delta",
    "challenger_arbitration_accepted",
    "challenger_arbitration_current_margin_under_current",
    "challenger_arbitration_refined_score_delta",
    "challenger_arbitration_model_score_delta",
    "challenger_arbitration_predicted_kk_delta",
]


@dataclass(frozen=True)
class Decision:
    key: str
    current: dict[str, Any]
    challenger: dict[str, Any]
    best_score: float
    features: dict[str, float]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fnum(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return float(default)
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def fbool(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
    return 1.0 if bool(row.get(key)) else 0.0


def row_key(row: dict[str, Any], ordinal: int) -> str:
    line = row.get("line")
    if line is not None:
        return str(line)
    return str(ordinal)


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    return {row_key(row, idx): row for idx, row in enumerate(iter_jsonl(path), start=1)}


def result_score(row: dict[str, Any]) -> float:
    return fnum(row, "final_teacher_score", fnum(row, "teacher_best_score", 0.0))


def result_regret(row: dict[str, Any]) -> float:
    return max(0.0, fnum(row, "teacher_best_score") - result_score(row))


def numeric_prefixed_features(prefix: str, row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    features = row.get("t1_override_features") or {}
    if isinstance(features, dict):
        for key, value in features.items():
            if isinstance(value, (int, float)):
                out[f"{prefix}_{key}"] = float(value)
    return out


def numeric_nested_features(prefix: str, row: dict[str, Any], field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    features = row.get(field) or {}
    if not isinstance(features, dict):
        return out
    for key, value in features.items():
        if isinstance(value, bool):
            out[f"{prefix}_{key}"] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[f"{prefix}_{key}"] = float(value)
    return out


def build_features(current: dict[str, Any], challenger: dict[str, Any]) -> dict[str, float]:
    current_override = current.get("t1_override_features") or {}
    challenger_override = challenger.get("t1_override_features") or {}
    out: dict[str, float] = {
        "same_action": 1.0
        if int(current.get("best_action_idx", -1)) == int(challenger.get("best_action_idx", -2))
        else 0.0,
        "refined_score_delta": fnum(challenger, "best_refined_score") - fnum(current, "best_refined_score"),
        "refined_override_delta_delta": fnum(challenger, "refined_override_delta")
        - fnum(current, "refined_override_delta"),
        "best_refined_score_delta": fnum(challenger, "best_refined_score") - fnum(current, "best_refined_score"),
        "model_top1_refined_score_delta": fnum(challenger, "model_top1_refined_score")
        - fnum(current, "model_top1_refined_score"),
        "override_gate_probability_delta": fnum(challenger, "t1_override_gate_probability")
        - fnum(current, "t1_override_gate_probability"),
        "current_override_gate_probability": fnum(current, "t1_override_gate_probability"),
        "challenger_override_gate_probability": fnum(challenger, "t1_override_gate_probability"),
        "current_model_top1_overridden": fbool(current, "model_top1_overridden"),
        "challenger_model_top1_overridden": fbool(challenger, "model_top1_overridden"),
        "current_override_accepted": fbool(current, "t1_override_gate_accepted"),
        "challenger_override_accepted": fbool(challenger, "t1_override_gate_accepted"),
        "current_refined_delta": fnum(current, "refined_override_delta"),
        "challenger_refined_delta": fnum(challenger, "refined_override_delta"),
        "elapsed_ms_delta": fnum(challenger, "elapsed_ms") - fnum(current, "elapsed_ms"),
        "exact_evaluated_delta": fnum(challenger, "exact_evaluated") - fnum(current, "exact_evaluated"),
    }
    current_prefixed = numeric_prefixed_features("current", current)
    challenger_prefixed = numeric_prefixed_features("challenger", challenger)
    out.update(current_prefixed)
    out.update(challenger_prefixed)
    current_arbitration = numeric_nested_features("current_arbitration", current, "t1_arbitration_details")
    challenger_arbitration = numeric_nested_features("challenger_arbitration", challenger, "t1_arbitration_details")
    out.update(current_arbitration)
    out.update(challenger_arbitration)
    for key in sorted(set(current_override) | set(challenger_override)):
        if isinstance(current_override.get(key), (int, float)) or isinstance(challenger_override.get(key), (int, float)):
            out[f"t1_{key}_delta"] = fnum(challenger_override, key) - fnum(current_override, key)
    current_details = current.get("t1_arbitration_details") or {}
    challenger_details = challenger.get("t1_arbitration_details") or {}
    if isinstance(current_details, dict) and isinstance(challenger_details, dict):
        for key in sorted(set(current_details) | set(challenger_details)):
            if isinstance(current_details.get(key), bool) or isinstance(challenger_details.get(key), bool):
                out[f"arbitration_{key}_delta"] = (
                    (1.0 if bool(challenger_details.get(key)) else 0.0)
                    - (1.0 if bool(current_details.get(key)) else 0.0)
                )
            elif isinstance(current_details.get(key), (int, float)) or isinstance(challenger_details.get(key), (int, float)):
                out[f"arbitration_{key}_delta"] = fnum(challenger_details, key) - fnum(current_details, key)
    return out


def build_decisions(current_path: Path, challenger_path: Path) -> list[Decision]:
    current = load_results(current_path)
    challenger = load_results(challenger_path)
    decisions: list[Decision] = []
    for key in sorted(set(current) & set(challenger), key=lambda value: int(value) if value.isdigit() else value):
        c_row = current[key]
        h_row = challenger[key]
        decisions.append(
            Decision(
                key=key,
                current=c_row,
                challenger=h_row,
                best_score=fnum(c_row, "teacher_best_score", fnum(h_row, "teacher_best_score")),
                features=build_features(c_row, h_row),
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
        current_score = result_score(decision.current)
        challenger_score = result_score(decision.challenger)
        current_hit = bool(decision.current.get("final_top1_hit"))
        challenger_hit = bool(decision.challenger.get("final_top1_hit"))
        if current_hit:
            current_hits += 1
        if challenger_hit:
            challenger_hits += 1
        if int(decision.current.get("best_action_idx", -1)) == int(decision.challenger.get("best_action_idx", -2)):
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


def parse_set(raw: str) -> tuple[str, Path, Path]:
    if "=" not in raw or "::" not in raw:
        raise ValueError("--set must be NAME=CURRENT::CHALLENGER")
    name, paths = raw.split("=", 1)
    current, challenger = paths.split("::", 1)
    return name, Path(current), Path(challenger)


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
    parser = argparse.ArgumentParser(description="Grid-search T1 runtime current-vs-challenger gates")
    parser.add_argument("--set", action="append", required=True, help="NAME=CURRENT_RESULTS::CHALLENGER_RESULTS")
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
    set_names = {name for name, _current, _challenger in parsed_sets}
    if not guard_sets:
        guard_sets = set_names - target_sets
    if not target_sets:
        target_sets = set_names - guard_sets
    features = [part.strip() for part in args.features.split(",") if part.strip()]

    decisions_by_set = {
        name: build_decisions(current, challenger)
        for name, current, challenger in parsed_sets
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

    sorted_candidates = sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)
    report = {
        "sets": {
            name: {"current": str(current), "challenger": str(challenger)}
            for name, current, challenger in parsed_sets
        },
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
