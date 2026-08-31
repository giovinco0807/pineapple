"""Evaluate simple arbitration rules between two T2 runtime modes.

The intended use is comparing the current fast Top3 refinement path against a
challenger such as Top5 + extra samples + model blend.  It uses scored runtime
outputs for ground-truth selected-action regret, and runtime JSONL rows for
features available before a final mode choice.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EPS = 1e-9


@dataclass(frozen=True)
class EvalSet:
    name: str
    current_score: Path
    current_runtime: Path
    challenger_score: Path
    challenger_runtime: Path


@dataclass(frozen=True)
class Decision:
    set_name: str
    row_id: int
    current_regret: float
    challenger_regret: float
    current_score: float
    challenger_score: float
    features: dict[str, float]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def runtime_result(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    if isinstance(result, dict):
        return result
    return row


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
    return 1.0 if row is not None and bool(row.get(key)) else 0.0


def nested_num(row: dict[str, Any] | None, path: tuple[str, ...], default: float = 0.0) -> float:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return float(default)
        value = value.get(key)
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def action_key(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    best = row.get("best") or {}
    action = best.get("action") or {}
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def top_line_len(row: dict[str, Any] | None) -> int:
    best = (row or {}).get("best") or {}
    board = best.get("board") or {}
    cards = board.get("top") or []
    return len(cards) if isinstance(cards, list) else 0


def candidate_gap(row: dict[str, Any], field: str) -> float:
    values = []
    for cand in row.get("candidates") or []:
        value = cand.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    values.sort(reverse=True)
    if len(values) < 2:
        return 0.0
    return values[0] - values[1]


def exact_refined_gap(row: dict[str, Any]) -> float:
    values = []
    for cand in row.get("candidates") or []:
        if cand.get("refinement_source") != "exact_partial":
            continue
        value = cand.get("refined_score")
        if isinstance(value, (int, float)):
            values.append(float(value))
    values.sort(reverse=True)
    if len(values) < 2:
        return 0.0
    return values[0] - values[1]


def build_features(current: dict[str, Any], challenger: dict[str, Any]) -> dict[str, float]:
    c_best = current.get("best") or {}
    h_best = challenger.get("best") or {}
    c_action_idx = int(current.get("best_action_idx", -1))
    h_action_idx = int(challenger.get("best_action_idx", -2))
    c_rank = nested_num(c_best, ("model_rank",), 999.0)
    h_rank = nested_num(h_best, ("model_rank",), 999.0)
    c_refined = nested_num(c_best, ("refined_score",), 0.0)
    h_refined = nested_num(h_best, ("refined_score",), 0.0)
    c_model = nested_num(c_best, ("model_score",), 0.0)
    h_model = nested_num(h_best, ("model_score",), 0.0)
    c_bust = nested_num(c_best, ("predicted_bust",), 0.0)
    h_bust = nested_num(h_best, ("predicted_bust",), 0.0)
    c_fl = nested_num(c_best, ("predicted_fl",), 0.0)
    h_fl = nested_num(h_best, ("predicted_fl",), 0.0)
    return {
        "same_action_idx": 1.0 if c_action_idx == h_action_idx else 0.0,
        "same_action_payload": 1.0 if action_key(current) == action_key(challenger) else 0.0,
        "challenger_switches": 1.0 if c_action_idx != h_action_idx else 0.0,
        "current_model_top1_overridden": fbool(current, "model_top1_overridden"),
        "challenger_model_top1_overridden": fbool(challenger, "model_top1_overridden"),
        "current_refined_delta": fnum(current, "refined_override_delta"),
        "challenger_refined_delta": fnum(challenger, "refined_override_delta"),
        "refined_delta": h_refined - c_refined,
        "model_delta": h_model - c_model,
        "bust_delta": h_bust - c_bust,
        "fl_delta": h_fl - c_fl,
        "rank_delta": h_rank - c_rank,
        "current_best_model_rank": c_rank,
        "challenger_best_model_rank": h_rank,
        "current_best_model_score": c_model,
        "challenger_best_model_score": h_model,
        "current_best_refined_score": c_refined,
        "challenger_best_refined_score": h_refined,
        "current_best_predicted_bust": c_bust,
        "challenger_best_predicted_bust": h_bust,
        "current_best_predicted_fl": c_fl,
        "challenger_best_predicted_fl": h_fl,
        "challenger_refined_minus_model": h_refined - h_model,
        "current_refined_minus_model": c_refined - c_model,
        "refined_minus_model_delta": (h_refined - h_model) - (c_refined - c_model),
        "current_model_score_gap": candidate_gap(current, "model_score"),
        "challenger_model_score_gap": candidate_gap(challenger, "model_score"),
        "current_exact_refined_gap": exact_refined_gap(current),
        "challenger_exact_refined_gap": exact_refined_gap(challenger),
        "exact_refined_gap_delta": exact_refined_gap(challenger) - exact_refined_gap(current),
        "elapsed_ms_delta": fnum(challenger, "elapsed_ms") - fnum(current, "elapsed_ms"),
        "exact_evaluated_delta": fnum(challenger, "exact_evaluated") - fnum(current, "exact_evaluated"),
        "legal_actions": fnum(current, "legal_actions"),
        "candidate_pool_size": fnum(current, "candidate_pool_size"),
        "challenger_candidate_pool_size": fnum(challenger, "candidate_pool_size"),
        "top_line_len": float(top_line_len(current)),
    }


def score_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain rows")
    return rows


def build_decisions(eval_set: EvalSet) -> list[Decision]:
    c_scores = score_rows(eval_set.current_score)
    h_scores = score_rows(eval_set.challenger_score)
    c_runtime = list(iter_jsonl(eval_set.current_runtime))
    h_runtime = list(iter_jsonl(eval_set.challenger_runtime))
    n = min(len(c_scores), len(h_scores), len(c_runtime), len(h_runtime))
    decisions: list[Decision] = []
    for idx in range(n):
        c_score = c_scores[idx]
        h_score = h_scores[idx]
        c_regret_value = c_score.get("chosen_regret")
        h_regret_value = h_score.get("chosen_regret")
        if c_regret_value is None or h_regret_value is None:
            continue
        row_id = int(c_score.get("row_id", idx))
        teacher_best = fnum(c_score, "teacher_best_score", fnum(h_score, "teacher_best_score"))
        c_regret = float(c_regret_value)
        h_regret = float(h_regret_value)
        decisions.append(
            Decision(
                set_name=eval_set.name,
                row_id=row_id,
                current_regret=c_regret,
                challenger_regret=h_regret,
                current_score=teacher_best - c_regret,
                challenger_score=teacher_best - h_regret,
                features=build_features(runtime_result(c_runtime[idx]), runtime_result(h_runtime[idx])),
            )
        )
    return decisions


def summarize(decisions: list[Decision], switches: list[bool]) -> dict[str, Any]:
    if len(decisions) != len(switches):
        raise ValueError("decision/switch length mismatch")
    rows = len(decisions)
    current_regret = sum(d.current_regret for d in decisions)
    challenger_regret = sum(d.challenger_regret for d in decisions)
    oracle_regret = sum(min(d.current_regret, d.challenger_regret) for d in decisions)
    chosen_regret = 0.0
    hits = 0
    current_hits = 0
    challenger_hits = 0
    switched = 0
    good_switches = 0
    bad_switches = 0
    missed_good_switches = 0
    for d, switch in zip(decisions, switches):
        current_hit = d.current_regret <= EPS
        challenger_hit = d.challenger_regret <= EPS
        current_hits += 1 if current_hit else 0
        challenger_hits += 1 if challenger_hit else 0
        if switch:
            switched += 1
            regret = d.challenger_regret
            if d.challenger_regret + EPS < d.current_regret:
                good_switches += 1
            elif d.challenger_regret > d.current_regret + EPS:
                bad_switches += 1
        else:
            regret = d.current_regret
            if d.challenger_regret + EPS < d.current_regret:
                missed_good_switches += 1
        hits += 1 if regret <= EPS else 0
        chosen_regret += regret
    denom = max(rows, 1)
    return {
        "rows": rows,
        "switched": switched,
        "switch_rate": switched / denom,
        "top1": hits / denom,
        "hits": hits,
        "avg_regret": chosen_regret / denom,
        "current_top1": current_hits / denom,
        "current_hits": current_hits,
        "current_avg_regret": current_regret / denom,
        "challenger_top1": challenger_hits / denom,
        "challenger_hits": challenger_hits,
        "challenger_avg_regret": challenger_regret / denom,
        "oracle_top1": sum(1 for d in decisions if min(d.current_regret, d.challenger_regret) <= EPS) / denom,
        "oracle_avg_regret": oracle_regret / denom,
        "top1_delta_vs_current": (hits - current_hits) / denom,
        "regret_delta_vs_current": chosen_regret / denom - current_regret / denom,
        "regret_delta_vs_challenger": chosen_regret / denom - challenger_regret / denom,
        "oracle_gap": chosen_regret / denom - oracle_regret / denom,
        "good_switches": good_switches,
        "bad_switches": bad_switches,
        "missed_good_switches": missed_good_switches,
    }


def threshold_candidates(values: list[float], *, max_thresholds: int) -> list[float]:
    clean = sorted({v for v in values if math.isfinite(v)})
    if not clean:
        return []
    if len(clean) <= max_thresholds:
        return clean
    out = []
    last_index = len(clean) - 1
    for i in range(max_thresholds):
        index = round(i * last_index / max(max_thresholds - 1, 1))
        out.append(clean[index])
    return sorted(set(out))


def switch_by_gate(decision: Decision, gate: dict[str, Any]) -> bool:
    gate_type = gate.get("gate_type")
    if gate_type == "always_current":
        return False
    if gate_type == "always_challenger":
        return True
    if gate_type == "single_feature":
        value = decision.features.get(str(gate["feature"]), 0.0)
        threshold = float(gate["threshold"])
        direction = str(gate["direction"])
        return value >= threshold if direction == ">=" else value <= threshold
    raise ValueError(f"unsupported gate_type: {gate_type}")


def evaluate_gate(decisions: list[Decision], gate: dict[str, Any]) -> dict[str, Any]:
    switches = [switch_by_gate(decision, gate) for decision in decisions]
    out = summarize(decisions, switches)
    for key in ("gate_type", "feature", "direction", "threshold"):
        if key in gate:
            out[key] = gate[key]
    return out


def candidate_gates(decisions: list[Decision], *, max_thresholds: int) -> list[dict[str, Any]]:
    gates = [
        {"gate_type": "always_current"},
        {"gate_type": "always_challenger"},
    ]
    features = sorted({key for decision in decisions for key in decision.features})
    for feature in features:
        values = [decision.features.get(feature, 0.0) for decision in decisions]
        for threshold in threshold_candidates(values, max_thresholds=max_thresholds):
            gates.append(
                {
                    "gate_type": "single_feature",
                    "feature": feature,
                    "direction": ">=",
                    "threshold": threshold,
                }
            )
            gates.append(
                {
                    "gate_type": "single_feature",
                    "feature": feature,
                    "direction": "<=",
                    "threshold": threshold,
                }
            )
    return gates


def choose_best(rows: list[dict[str, Any]], *, min_switches: int = 0) -> dict[str, Any]:
    filtered = [row for row in rows if int(row.get("switched", 0)) >= min_switches]
    if not filtered:
        filtered = rows
    return min(
        filtered,
        key=lambda row: (
            float(row["avg_regret"]),
            -float(row["top1"]),
            abs(float(row.get("switch_rate", 0.0)) - 0.5),
            str(row.get("gate_type", "")),
            str(row.get("feature", "")),
        ),
    )


def parse_eval_set(raw: str) -> EvalSet:
    parts = raw.split("|")
    if len(parts) != 5:
        raise ValueError("--eval must be name|current_score|current_runtime|challenger_score|challenger_runtime")
    return EvalSet(
        name=parts[0],
        current_score=Path(parts[1]),
        current_runtime=Path(parts[2]),
        challenger_score=Path(parts[3]),
        challenger_runtime=Path(parts[4]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate T2 runtime arbitration rules")
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        help="name|current_score|current_runtime|challenger_score|challenger_runtime",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-thresholds", type=int, default=80)
    parser.add_argument("--min-switches", type=int, default=1)
    args = parser.parse_args()

    eval_sets = [parse_eval_set(raw) for raw in args.eval]
    by_set = {eval_set.name: build_decisions(eval_set) for eval_set in eval_sets}
    all_decisions = [decision for decisions in by_set.values() for decision in decisions]

    all_gate_rows = [
        evaluate_gate(all_decisions, gate)
        for gate in candidate_gates(all_decisions, max_thresholds=args.max_thresholds)
    ]
    best_all = choose_best(all_gate_rows, min_switches=args.min_switches)

    per_set = {}
    for name, decisions in by_set.items():
        rows = [
            evaluate_gate(decisions, gate)
            for gate in candidate_gates(decisions, max_thresholds=args.max_thresholds)
        ]
        per_set[name] = {
            "current": evaluate_gate(decisions, {"gate_type": "always_current"}),
            "challenger": evaluate_gate(decisions, {"gate_type": "always_challenger"}),
            "oracle_between_modes": summarize(decisions, [d.challenger_regret < d.current_regret for d in decisions]),
            "best_diagnostic_gate": choose_best(rows, min_switches=args.min_switches),
            "best_all_gate_replay": evaluate_gate(decisions, best_all),
        }

    loso = {}
    names = list(by_set)
    for heldout in names:
        train = [decision for name in names if name != heldout for decision in by_set[name]]
        held = by_set[heldout]
        train_rows = [
            evaluate_gate(train, gate)
            for gate in candidate_gates(train, max_thresholds=args.max_thresholds)
        ]
        best_train = choose_best(train_rows, min_switches=args.min_switches)
        loso[heldout] = {
            "train_sets": [name for name in names if name != heldout],
            "selected_gate": best_train,
            "heldout_replay": evaluate_gate(held, best_train),
            "heldout_current": evaluate_gate(held, {"gate_type": "always_current"}),
            "heldout_challenger": evaluate_gate(held, {"gate_type": "always_challenger"}),
        }

    output = {
        "evals": [raw for raw in args.eval],
        "sets": {name: len(decisions) for name, decisions in by_set.items()},
        "all": {
            "current": evaluate_gate(all_decisions, {"gate_type": "always_current"}),
            "challenger": evaluate_gate(all_decisions, {"gate_type": "always_challenger"}),
            "oracle_between_modes": summarize(
                all_decisions,
                [d.challenger_regret < d.current_regret for d in all_decisions],
            ),
            "best_gate": best_all,
            "top_gate_candidates": sorted(
                all_gate_rows,
                key=lambda row: (float(row["avg_regret"]), -float(row["top1"])),
            )[:20],
        },
        "per_set": per_set,
        "loso": loso,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    current = output["all"]["current"]
    challenger = output["all"]["challenger"]
    best = output["all"]["best_gate"]
    print(
        f"rows={len(all_decisions)} current_reg={current['avg_regret']:.3f} "
        f"challenger_reg={challenger['avg_regret']:.3f} best_reg={best['avg_regret']:.3f} "
        f"best_gate={best.get('gate_type')} {best.get('feature', '')} "
        f"{best.get('direction', '')} {best.get('threshold', '')}"
    )
    for name, fold in loso.items():
        replay = fold["heldout_replay"]
        print(
            f"loso heldout={name} selected_reg={replay['avg_regret']:.3f} "
            f"current_reg={fold['heldout_current']['avg_regret']:.3f} "
            f"challenger_reg={fold['heldout_challenger']['avg_regret']:.3f} "
            f"switches={replay['switched']}"
        )


if __name__ == "__main__":
    main()
