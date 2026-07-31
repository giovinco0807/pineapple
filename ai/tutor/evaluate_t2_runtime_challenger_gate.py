"""Grid-search simple gates between two saved T2 runtime result sets.

This audits a completed baseline run against a completed challenger run without
re-running the expensive T2/T3 refinement path.  It is meant for deciding
whether a specialist model is useful enough to justify a real runtime gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.exact_late import action_key  # noqa: E402


DEFAULT_FEATURES = [
    "same_action",
    "position_btn",
    "position_bb",
    "own_top_len",
    "own_middle_len",
    "own_bottom_len",
    "opp_top_len",
    "opp_middle_len",
    "opp_bottom_len",
    "own_top_structured",
    "opp_top_structured",
    "dealt_max_rank",
    "dealt_min_rank",
    "dealt_rank_sum",
    "dealt_pair",
    "dealt_trip",
    "dealt_has_ace",
    "dealt_has_king",
    "dealt_has_queen",
    "visible_joker",
    "dealt_joker",
    "known_discard_count",
    "baseline_best_model_rank",
    "baseline_best_model_score",
    "baseline_best_refined_score",
    "baseline_best_predicted_bust",
    "baseline_best_predicted_fl",
    "baseline_best_predicted_aa",
    "baseline_best_predicted_kk",
    "baseline_best_predicted_qq",
    "baseline_best_predicted_trips",
    "baseline_model_top1_overridden",
    "baseline_top1_rescue_applied",
    "baseline_exact_evaluated",
    "baseline_elapsed_ms",
    "challenger_best_model_rank",
    "challenger_best_model_score",
    "challenger_best_refined_score",
    "challenger_best_predicted_bust",
    "challenger_best_predicted_fl",
    "challenger_best_predicted_aa",
    "challenger_best_predicted_kk",
    "challenger_best_predicted_qq",
    "challenger_best_predicted_trips",
    "challenger_model_top1_overridden",
    "challenger_top1_rescue_applied",
    "challenger_exact_evaluated",
    "challenger_elapsed_ms",
    "model_rank_delta",
    "model_score_delta",
    "refined_score_delta",
    "predicted_bust_delta",
    "predicted_fl_delta",
]


@dataclass(frozen=True)
class Decision:
    set_name: str
    key: str
    baseline: dict[str, Any]
    challenger: dict[str, Any]
    baseline_loss: float
    challenger_loss: float
    features: dict[str, float]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def runtime_paths(raw_path: Path) -> tuple[Path, Path | None]:
    if raw_path.is_dir():
        rows = raw_path / "rows.jsonl"
        results = raw_path / "results.jsonl"
        return rows, results if results.exists() else None
    return raw_path, None


def runtime_row_count(raw_path: Path) -> int:
    rows_path, _results_path = runtime_paths(raw_path)
    return sum(1 for line in rows_path.open("r", encoding="utf-8-sig") if line.strip())


def row_key(row: dict[str, Any], ordinal: int) -> str:
    source_line = row.get("source_line")
    if source_line is not None:
        return str(source_line)
    line = row.get("line")
    if line is not None:
        return str(line)
    return str(ordinal)


def load_runtime(path: Path) -> dict[str, dict[str, Any]]:
    rows_path, results_path = runtime_paths(path)
    rows = {row_key(row, idx): dict(row) for idx, row in enumerate(iter_jsonl(rows_path), start=1)}
    if results_path is None:
        return rows
    full_rows = {row_key(row, idx): row for idx, row in enumerate(iter_jsonl(results_path), start=1)}
    for key, row in rows.items():
        if key in full_rows:
            row["__full"] = full_rows[key]
    return rows


def fnum(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return float(default)
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def fbool(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
    return 1.0 if bool(row.get(key)) else 0.0


def nested(row: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if row is None:
        return {}
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def full_payload(row: dict[str, Any]) -> dict[str, Any]:
    full = row.get("__full")
    if isinstance(full, dict) and isinstance(full.get("payload"), dict):
        return full["payload"]
    source = row.get("source_record")
    if isinstance(source, dict):
        return source
    payload = row.get("payload")
    if isinstance(payload, dict):
        return payload
    return row


def full_result(row: dict[str, Any]) -> dict[str, Any]:
    full = row.get("__full")
    if isinstance(full, dict) and isinstance(full.get("result"), dict):
        return full["result"]
    result = row.get("result")
    if isinstance(result, dict):
        return result
    return row


def chosen_loss(row: dict[str, Any]) -> float | None:
    teacher = row.get("teacher")
    if not isinstance(teacher, dict):
        return None
    if not teacher.get("valid", True):
        return None
    value = teacher.get("chosen_ev_loss")
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def rank_value(card: Any) -> int:
    if not isinstance(card, str) or not card:
        return 0
    rank = card[0].upper()
    if rank == "A":
        return 14
    if rank == "K":
        return 13
    if rank == "Q":
        return 12
    if rank == "J":
        return 11
    if rank == "T":
        return 10
    if rank.isdigit():
        return int(rank)
    return 0


def is_joker(card: Any) -> bool:
    if not isinstance(card, str):
        return False
    lower = card.lower()
    return lower.startswith("x") or "joker" in lower


def board_cards(board: dict[str, Any], row_name: str) -> list[str]:
    cards = board.get(row_name)
    return [str(card) for card in cards] if isinstance(cards, list) else []


def has_duplicate_rank(cards: list[str]) -> bool:
    ranks = [rank_value(card) for card in cards if rank_value(card) > 0]
    return len(set(ranks)) < len(ranks) if ranks else False


def state_features(row: dict[str, Any]) -> dict[str, float]:
    payload = full_payload(row)
    board = payload.get("board") if isinstance(payload.get("board"), dict) else row.get("board")
    opp = (
        payload.get("opponent_board")
        if isinstance(payload.get("opponent_board"), dict)
        else row.get("opponent_board")
    )
    board = board if isinstance(board, dict) else {}
    opp = opp if isinstance(opp, dict) else {}
    dealt = payload.get("dealt") if isinstance(payload.get("dealt"), list) else row.get("dealt")
    dealt_cards = [str(card) for card in dealt] if isinstance(dealt, list) else []
    known_discards = payload.get("known_discards") if isinstance(payload.get("known_discards"), list) else []
    exclude = payload.get("exclude") if isinstance(payload.get("exclude"), list) else []
    visible_cards = [
        *board_cards(board, "top"),
        *board_cards(board, "middle"),
        *board_cards(board, "bottom"),
        *board_cards(opp, "top"),
        *board_cards(opp, "middle"),
        *board_cards(opp, "bottom"),
        *[str(card) for card in known_discards],
        *[str(card) for card in exclude],
    ]
    ranks = [rank_value(card) for card in dealt_cards if rank_value(card) > 0]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    position = str(payload.get("position") or row.get("position") or "").lower()
    is_btn = bool(payload.get("is_btn", position == "btn"))
    return {
        "position_btn": 1.0 if is_btn else 0.0,
        "position_bb": 0.0 if is_btn else 1.0,
        "own_top_len": float(len(board_cards(board, "top"))),
        "own_middle_len": float(len(board_cards(board, "middle"))),
        "own_bottom_len": float(len(board_cards(board, "bottom"))),
        "opp_top_len": float(len(board_cards(opp, "top"))),
        "opp_middle_len": float(len(board_cards(opp, "middle"))),
        "opp_bottom_len": float(len(board_cards(opp, "bottom"))),
        "own_top_structured": 1.0 if has_duplicate_rank(board_cards(board, "top")) else 0.0,
        "opp_top_structured": 1.0 if has_duplicate_rank(board_cards(opp, "top")) else 0.0,
        "dealt_max_rank": float(max(ranks) if ranks else 0),
        "dealt_min_rank": float(min(ranks) if ranks else 0),
        "dealt_rank_sum": float(sum(ranks)),
        "dealt_pair": 1.0 if any(count >= 2 for count in rank_counts.values()) else 0.0,
        "dealt_trip": 1.0 if any(count >= 3 for count in rank_counts.values()) else 0.0,
        "dealt_has_ace": 1.0 if 14 in ranks else 0.0,
        "dealt_has_king": 1.0 if 13 in ranks else 0.0,
        "dealt_has_queen": 1.0 if 12 in ranks else 0.0,
        "dealt_joker": 1.0 if any(is_joker(card) for card in dealt_cards) else 0.0,
        "visible_joker": 1.0 if any(is_joker(card) for card in visible_cards) else 0.0,
        "known_discard_count": float(len(known_discards)),
    }


def selected_action(row: dict[str, Any]) -> str:
    result = full_result(row)
    best = result.get("best")
    if isinstance(best, dict) and isinstance(best.get("action"), dict):
        return action_key(best["action"])
    action = row.get("best_action")
    if isinstance(action, dict):
        return action_key(action)
    return str(row.get("best_action_idx"))


def runtime_features(prefix: str, row: dict[str, Any]) -> dict[str, float]:
    result = full_result(row)
    best = result.get("best") if isinstance(result.get("best"), dict) else {}
    fl_types = best.get("predicted_fl_types") if isinstance(best.get("predicted_fl_types"), dict) else {}
    rescue = nested(result, "t2_model_top1_rescue_details") or nested(row, "t2_model_top1_rescue_details")
    return {
        f"{prefix}_best_model_rank": fnum(best, "model_rank", 999.0),
        f"{prefix}_best_refined_rank": fnum(best, "refined_rank", fnum(best, "refinement_rank", 999.0)),
        f"{prefix}_best_model_score": fnum(best, "model_score", fnum(row, "best_model_score")),
        f"{prefix}_best_refined_score": fnum(best, "refined_score", fnum(row, "best_refined_score")),
        f"{prefix}_best_predicted_bust": fnum(best, "predicted_bust"),
        f"{prefix}_best_predicted_fl": fnum(best, "predicted_fl"),
        f"{prefix}_best_predicted_aa": fnum(fl_types, "aa"),
        f"{prefix}_best_predicted_kk": fnum(fl_types, "kk"),
        f"{prefix}_best_predicted_qq": fnum(fl_types, "qq"),
        f"{prefix}_best_predicted_trips": fnum(fl_types, "trips"),
        f"{prefix}_best_samples": fnum(best, "samples"),
        f"{prefix}_best_forced_bust": fbool(best, "forced_bust"),
        f"{prefix}_model_top1_overridden": fbool(result, "model_top1_overridden"),
        f"{prefix}_top1_rescue_applied": fbool(result, "t2_model_top1_rescue_applied"),
        f"{prefix}_exact_evaluated": fnum(result, "exact_evaluated", fnum(row, "exact_evaluated")),
        f"{prefix}_sync_count": fnum(
            result,
            "sync_refinement_candidate_count",
            fnum(row, "sync_refinement_candidate_count"),
        ),
        f"{prefix}_candidate_pool_size": fnum(result, "candidate_pool_size", fnum(row, "candidate_pool_size")),
        f"{prefix}_elapsed_ms": fnum(result, "elapsed_ms", fnum(row, "elapsed_ms")),
        f"{prefix}_rescue_current_model_rank": fnum(rescue, "current_model_rank", 999.0),
        f"{prefix}_rescue_challenger_predicted_bust": fnum(rescue, "challenger_predicted_bust"),
        f"{prefix}_rescue_challenger_predicted_fl": fnum(rescue, "challenger_predicted_fl"),
        f"{prefix}_rescue_refined_delta": fnum(rescue, "refined_delta"),
    }


def build_features(baseline: dict[str, Any], challenger: dict[str, Any]) -> dict[str, float]:
    out = state_features(baseline)
    out.update(runtime_features("baseline", baseline))
    out.update(runtime_features("challenger", challenger))
    out["same_action"] = 1.0 if selected_action(baseline) == selected_action(challenger) else 0.0
    pairs = [
        ("model_rank_delta", "best_model_rank"),
        ("refined_rank_delta", "best_refined_rank"),
        ("model_score_delta", "best_model_score"),
        ("refined_score_delta", "best_refined_score"),
        ("predicted_bust_delta", "best_predicted_bust"),
        ("predicted_fl_delta", "best_predicted_fl"),
        ("predicted_aa_delta", "best_predicted_aa"),
        ("predicted_kk_delta", "best_predicted_kk"),
        ("predicted_qq_delta", "best_predicted_qq"),
        ("predicted_trips_delta", "best_predicted_trips"),
        ("elapsed_ms_delta", "elapsed_ms"),
        ("exact_evaluated_delta", "exact_evaluated"),
    ]
    for out_key, suffix in pairs:
        out[out_key] = out[f"challenger_{suffix}"] - out[f"baseline_{suffix}"]
    return out


def build_decisions(set_name: str, baseline_path: Path, challenger_path: Path) -> list[Decision]:
    baseline = load_runtime(baseline_path)
    challenger = load_runtime(challenger_path)
    decisions: list[Decision] = []
    for key in sorted(set(baseline) & set(challenger), key=lambda value: int(value) if value.isdigit() else value):
        b_row = baseline[key]
        c_row = challenger[key]
        b_loss = chosen_loss(b_row)
        c_loss = chosen_loss(c_row)
        if b_loss is None or c_loss is None:
            continue
        decisions.append(
            Decision(
                set_name=set_name,
                key=key,
                baseline=b_row,
                challenger=c_row,
                baseline_loss=b_loss,
                challenger_loss=c_loss,
                features=build_features(b_row, c_row),
            )
        )
    return decisions


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = round((len(ordered) - 1) * pct)
    return ordered[int(max(0, min(len(ordered) - 1, idx)))]


def loss_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "p95": None, "p99": None, "max": None, "sum": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "sum": sum(values),
    }


def summarize_policy(decisions: list[Decision], switches: list[bool]) -> dict[str, Any]:
    if len(decisions) != len(switches):
        raise ValueError("decision/switch length mismatch")
    baseline_losses = [decision.baseline_loss for decision in decisions]
    challenger_losses = [decision.challenger_loss for decision in decisions]
    chosen_losses: list[float] = []
    switched = 0
    action_switches = 0
    beneficial = 0
    harmful = 0
    neutral = 0
    missed_good = 0
    switch_gain = 0.0
    bad_switch_loss = 0.0
    for decision, switch in zip(decisions, switches):
        if switch:
            switched += 1
            if decision.features.get("same_action", 0.0) < 0.5:
                action_switches += 1
            chosen_loss = decision.challenger_loss
            gain = decision.baseline_loss - decision.challenger_loss
            switch_gain += gain
            if gain > 1.0e-9:
                beneficial += 1
            elif gain < -1.0e-9:
                harmful += 1
                bad_switch_loss += -gain
            else:
                neutral += 1
        else:
            chosen_loss = decision.baseline_loss
            if decision.challenger_loss + 1.0e-9 < decision.baseline_loss:
                missed_good += 1
        chosen_losses.append(chosen_loss)
    rows = len(decisions)
    denom = max(rows, 1)
    baseline_hits = sum(1 for value in baseline_losses if value <= 1.0e-9)
    challenger_hits = sum(1 for value in challenger_losses if value <= 1.0e-9)
    chosen_hits = sum(1 for value in chosen_losses if value <= 1.0e-9)
    baseline = loss_summary(baseline_losses)
    challenger = loss_summary(challenger_losses)
    gated = loss_summary(chosen_losses)
    return {
        "decisions": rows,
        "switched": switched,
        "action_switches": action_switches,
        "switch_rate": switched / denom,
        "action_switch_rate": action_switches / denom,
        "beneficial_switches": beneficial,
        "harmful_switches": harmful,
        "neutral_switches": neutral,
        "missed_good_switches": missed_good,
        "switch_gain": switch_gain,
        "bad_switch_loss": bad_switch_loss,
        "baseline_hits": baseline_hits,
        "challenger_hits": challenger_hits,
        "hits": chosen_hits,
        "hit_rate": chosen_hits / denom,
        "baseline_hit_rate": baseline_hits / denom,
        "challenger_hit_rate": challenger_hits / denom,
        "hit_delta_vs_baseline": (chosen_hits - baseline_hits) / denom,
        "baseline_loss": baseline,
        "challenger_loss": challenger,
        "chosen_loss": gated,
        "mean_loss_delta_vs_baseline": (
            (gated["mean"] or 0.0) - (baseline["mean"] or 0.0)
        ),
        "sum_loss_delta_vs_baseline": gated["sum"] - baseline["sum"],
        "max_loss_delta_vs_baseline": (
            (gated["max"] or 0.0) - (baseline["max"] or 0.0)
        ),
    }


def parse_set(raw: str) -> tuple[str, Path, Path]:
    if "=" not in raw or "::" not in raw:
        raise ValueError("--set must be NAME=BASELINE::CHALLENGER")
    name, paths = raw.split("=", 1)
    baseline, challenger = paths.split("::", 1)
    return name, Path(baseline), Path(challenger)


def finite_feature_values(decisions: list[Decision], feature: str) -> list[float]:
    out: list[float] = []
    for decision in decisions:
        value = float(decision.features.get(feature, 0.0))
        if math.isfinite(value):
            out.append(value)
    return out


def quantile_thresholds(values: list[float], max_thresholds: int) -> list[float]:
    values = sorted(set(float(value) for value in values if math.isfinite(float(value))))
    if len(values) <= max_thresholds:
        return values
    out = set()
    for i in range(max_thresholds):
        idx = round(i * (len(values) - 1) / max(max_thresholds - 1, 1))
        out.add(values[int(idx)])
    return sorted(out)


def build_conditions(
    decisions_by_set: dict[str, list[Decision]],
    features: list[str],
    max_thresholds: int,
) -> list[dict[str, Any]]:
    decisions = [decision for rows in decisions_by_set.values() for decision in rows]
    out: list[dict[str, Any]] = []
    for feature in features:
        values = finite_feature_values(decisions, feature)
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


def aggregate_by_set(by_set: dict[str, Any]) -> dict[str, Any]:
    rows = sum(int(row["decisions"]) for row in by_set.values())
    switched = sum(int(row["switched"]) for row in by_set.values())
    action_switches = sum(int(row["action_switches"]) for row in by_set.values())
    baseline_sum = sum(float(row["baseline_loss"]["sum"]) for row in by_set.values())
    challenger_sum = sum(float(row["challenger_loss"]["sum"]) for row in by_set.values())
    chosen_sum = sum(float(row["chosen_loss"]["sum"]) for row in by_set.values())
    return {
        "decisions": rows,
        "switched": switched,
        "action_switches": action_switches,
        "switch_rate": switched / max(rows, 1),
        "action_switch_rate": action_switches / max(rows, 1),
        "baseline_loss_sum": baseline_sum,
        "challenger_loss_sum": challenger_sum,
        "chosen_loss_sum": chosen_sum,
        "chosen_sum_delta_vs_baseline": chosen_sum - baseline_sum,
        "challenger_sum_delta_vs_baseline": challenger_sum - baseline_sum,
    }


def rule_score(
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
        regret_delta = float(row["mean_loss_delta_vs_baseline"])
        top1_delta = float(row["hit_delta_vs_baseline"])
        max_guard_regret = max(max_guard_regret, regret_delta)
        min_guard_top1 = min(min_guard_top1, top1_delta)
        if regret_delta > regret_tolerance or top1_delta < -top1_tolerance:
            guard_ok = False
    target_top1 = sum(float(by_set[name]["hit_delta_vs_baseline"]) for name in target_sets)
    target_regret = -sum(float(by_set[name]["mean_loss_delta_vs_baseline"]) for name in target_sets)
    switch_penalty = -sum(float(row["action_switch_rate"]) for row in by_set.values()) * 0.001
    if max_guard_regret == float("-inf"):
        max_guard_regret = 0.0
    if min_guard_top1 == float("inf"):
        min_guard_top1 = 0.0
    return guard_ok, (target_top1, target_regret, -max_guard_regret, min_guard_top1 + switch_penalty)


def decision_row(decision: Decision) -> dict[str, Any]:
    return {
        "set": decision.set_name,
        "key": decision.key,
        "baseline_loss": decision.baseline_loss,
        "challenger_loss": decision.challenger_loss,
        "loss_delta_if_switch": decision.challenger_loss - decision.baseline_loss,
        "baseline_action": selected_action(decision.baseline),
        "challenger_action": selected_action(decision.challenger),
        "features": decision.features,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grid-search T2 runtime baseline-vs-challenger gates")
    parser.add_argument("--set", action="append", required=True, help="NAME=BASELINE_DIR_OR_ROWS::CHALLENGER_DIR_OR_ROWS")
    parser.add_argument("--guard-sets", default="")
    parser.add_argument("--target-sets", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--max-thresholds", type=int, default=11)
    parser.add_argument("--max-single-candidates", type=int, default=80)
    parser.add_argument("--regret-tolerance", type=float, default=0.0)
    parser.add_argument("--top1-tolerance", type=float, default=0.0)
    parser.add_argument("--min-switches", type=int, default=1)
    parser.add_argument("--top-rules", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    parsed_sets = [parse_set(raw) for raw in args.set]
    set_names = {name for name, _baseline, _challenger in parsed_sets}
    guard_sets = {part.strip() for part in args.guard_sets.split(",") if part.strip()}
    target_sets = {part.strip() for part in args.target_sets.split(",") if part.strip()}
    if not target_sets:
        target_sets = set(set_names)
    if not guard_sets:
        guard_sets = set(set_names)
    features = [part.strip() for part in args.features.split(",") if part.strip()]

    decisions_by_set = {
        name: build_decisions(name, baseline, challenger)
        for name, baseline, challenger in parsed_sets
    }
    input_counts = {
        name: {
            "baseline_rows": runtime_row_count(baseline_path),
            "challenger_rows": runtime_row_count(challenger_path),
            "paired_valid_decisions": len(decisions_by_set[name]),
        }
        for name, baseline_path, challenger_path in parsed_sets
    }
    conditions = build_conditions(decisions_by_set, features, args.max_thresholds)

    baseline = {
        name: {
            "baseline": summarize_policy(decisions, [False for _ in decisions]),
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
        candidates.append(
            {
                "conditions": [condition],
                "guard_ok": ok,
                "score": list(score),
                "aggregate": aggregate_by_set(by_set),
                "by_set": by_set,
            }
        )

    seed_conditions = [
        row["conditions"][0]
        for row in sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)[
            : max(1, int(args.max_single_candidates))
        ]
    ]
    for first, second in combinations(seed_conditions, 2):
        if first == second:
            continue
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
        candidates.append(
            {
                "conditions": [first, second],
                "guard_ok": ok,
                "score": list(score),
                "aggregate": aggregate_by_set(by_set),
                "by_set": by_set,
            }
        )

    sorted_candidates = sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "paired_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for decisions in decisions_by_set.values():
            for decision in decisions:
                f.write(json.dumps(decision_row(decision), ensure_ascii=False) + "\n")
    rules_path = output_dir / "rules.jsonl"
    with rules_path.open("w", encoding="utf-8") as f:
        for candidate in sorted_candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    report = {
        "sets": {
            name: {"baseline": str(baseline_path), "challenger": str(challenger_path)}
            for name, baseline_path, challenger_path in parsed_sets
        },
        "guard_sets": sorted(guard_sets),
        "target_sets": sorted(target_sets),
        "features": features,
        "input_counts": input_counts,
        "baseline": baseline,
        "candidate_count": len(candidates),
        "top_rules": sorted_candidates[: max(1, int(args.top_rules))],
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "rules": str(rules_path),
            "paired_rows": str(rows_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {
        "summary": str(summary_path),
        "paired_rows": sum(len(rows) for rows in decisions_by_set.values()),
        "candidate_count": len(candidates),
        "top_rule": sorted_candidates[0] if sorted_candidates else None,
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
