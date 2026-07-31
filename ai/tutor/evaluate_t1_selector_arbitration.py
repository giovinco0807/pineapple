"""Evaluate a gate that arbitrates between two T1 final selectors.

The T1 candidate pool is already strong enough that most misses are final
selection errors.  This script replays saved ``selector_rows.jsonl`` files and
asks whether a challenger selector should replace the current selector for each
decision.  It is intentionally offline-only; promote a gate into runtime only
after it improves held-out regret as well as Top1.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.train_t1_final_selector import (  # noqa: E402
    group_rows,
    row_features,
    selector_score,
    teacher_best_score,
)


DEFAULT_DELTA_FEATURES = [
    "current_margin_under_current",
    "challenger_margin_under_challenger",
    "challenger_minus_current_under_current",
    "challenger_minus_current_under_challenger",
    "selector_conflict_margin",
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
    "current_refined_rank",
    "challenger_refined_rank",
    "current_model_rank",
    "challenger_model_rank",
]


@dataclass(frozen=True)
class Decision:
    key: str
    current: dict[str, Any]
    challenger: dict[str, Any]
    best_score: float
    features: dict[str, float]


def iter_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rank_value(row: dict[str, Any], key: str, default: float = 999.0) -> float:
    return max(fnum(row, key, default), 1.0)


def row_teacher_score(row: dict[str, Any]) -> float:
    return fnum(row, "teacher_score", fnum(row, "teacher_best_score", 0.0))


def choose_by_selector(rows: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
    return max(rows, key=lambda row: selector_score(row, selector))


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


def build_pair_features(
    rows: list[dict[str, Any]],
    current: dict[str, Any],
    challenger: dict[str, Any],
    current_selector: dict[str, Any],
    challenger_selector: dict[str, Any],
) -> dict[str, float]:
    current_under_current = selector_score(current, current_selector)
    challenger_under_current = selector_score(challenger, current_selector)
    current_under_challenger = selector_score(current, challenger_selector)
    challenger_under_challenger = selector_score(challenger, challenger_selector)

    current_features = row_features(current)
    challenger_features = row_features(challenger)
    feature_delta = {
        f"{name}_delta": float(challenger_features.get(name, 0.0))
        - float(current_features.get(name, 0.0))
        for name in sorted(set(current_features) | set(challenger_features))
    }
    out = {
        "current_margin_under_current": current_under_current - challenger_under_current,
        "challenger_margin_under_challenger": challenger_under_challenger - current_under_challenger,
        "challenger_minus_current_under_current": challenger_under_current - current_under_current,
        "challenger_minus_current_under_challenger": challenger_under_challenger - current_under_challenger,
        "selector_conflict_margin": (challenger_under_challenger - current_under_challenger)
        - (current_under_current - challenger_under_current),
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
        "current_refined_rank": rank_value(current, "refined_rank"),
        "challenger_refined_rank": rank_value(challenger, "refined_rank"),
        "current_model_rank": rank_value(current, "model_rank"),
        "challenger_model_rank": rank_value(challenger, "model_rank"),
        "same_action": 1.0
        if int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
        else 0.0,
        "refined_rows": float(len(rows)),
    }
    out.update(feature_delta)
    return out


def build_decisions(
    paths: list[Path],
    *,
    current_selector: dict[str, Any],
    challenger_selector: dict[str, Any],
    min_teacher_margin: float,
) -> list[Decision]:
    grouped = group_rows(
        paths,
        require_teacher_refined=True,
        min_teacher_margin=min_teacher_margin,
    )
    decisions: list[Decision] = []
    for key, rows in sorted(grouped.items()):
        current = choose_by_selector(rows, current_selector)
        challenger = choose_by_selector(rows, challenger_selector)
        decisions.append(
            Decision(
                key=key,
                current=current,
                challenger=challenger,
                best_score=teacher_best_score(rows),
                features=build_pair_features(
                    rows,
                    current,
                    challenger,
                    current_selector,
                    challenger_selector,
                ),
            )
        )
    return decisions


def summarize_policy(decisions: list[Decision], switch_flags: list[bool]) -> dict[str, Any]:
    if len(decisions) != len(switch_flags):
        raise ValueError("decision/switch length mismatch")
    rows = len(decisions)
    top1 = 0
    current_top1 = 0
    challenger_top1 = 0
    oracle_top1 = 0
    same_action = 0
    regret = 0.0
    current_regret = 0.0
    challenger_regret = 0.0
    oracle_regret = 0.0
    switched = 0
    switch_gain = 0.0
    bad_switch_regret = 0.0
    missed_good_switches = 0
    for decision, switch in zip(decisions, switch_flags):
        current_score = row_teacher_score(decision.current)
        challenger_score = row_teacher_score(decision.challenger)
        if int(decision.current.get("action_idx", -1)) == int(decision.challenger.get("action_idx", -2)):
            same_action += 1
        current_hit = bool(decision.current.get("is_teacher_best"))
        challenger_hit = bool(decision.challenger.get("is_teacher_best"))
        if current_hit:
            current_top1 += 1
        if challenger_hit:
            challenger_top1 += 1
        oracle_uses_challenger = challenger_score > current_score
        oracle_hit = challenger_hit if oracle_uses_challenger else current_hit
        if oracle_hit:
            oracle_top1 += 1
        chosen = decision.challenger if switch else decision.current
        chosen_score = challenger_score if switch else current_score
        if bool(chosen.get("is_teacher_best")):
            top1 += 1
        regret += max(0.0, decision.best_score - chosen_score)
        current_regret += max(0.0, decision.best_score - current_score)
        challenger_regret += max(0.0, decision.best_score - challenger_score)
        oracle_score = max(current_score, challenger_score)
        oracle_regret += max(0.0, decision.best_score - oracle_score)
        if switch:
            switched += 1
            gain = challenger_score - current_score
            switch_gain += gain
            if gain < 0.0:
                bad_switch_regret += -gain
        elif challenger_score > current_score:
            missed_good_switches += 1

    denom = max(rows, 1)
    return {
        "decisions": rows,
        "switched": switched,
        "switch_rate": switched / denom,
        "same_action": same_action,
        "top1": top1 / denom,
        "hits": top1,
        "avg_regret": regret / denom,
        "current_top1": current_top1 / denom,
        "current_hits": current_top1,
        "current_avg_regret": current_regret / denom,
        "challenger_top1": challenger_top1 / denom,
        "challenger_hits": challenger_top1,
        "challenger_avg_regret": challenger_regret / denom,
        "oracle_between_top1": oracle_top1 / denom,
        "oracle_between_hits": oracle_top1,
        "oracle_between_avg_regret": oracle_regret / denom,
        "top1_delta_vs_current": (top1 - current_top1) / denom,
        "regret_delta_vs_current": regret / denom - current_regret / denom,
        "switch_gain": switch_gain,
        "bad_switch_regret": bad_switch_regret,
        "missed_good_switches": missed_good_switches,
    }


def threshold_candidates(values: list[float], max_count: int) -> list[float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return [0.0]
    if len(finite) <= max_count:
        return sorted(set(finite))
    thresholds = []
    for idx in range(max_count):
        q = idx / max(max_count - 1, 1)
        pos = min(len(finite) - 1, max(0, int(round(q * (len(finite) - 1)))))
        thresholds.append(finite[pos])
    return sorted(set(thresholds))


def sweep_single_feature(
    decisions: list[Decision],
    feature_names: list[str],
    *,
    max_thresholds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in feature_names:
        values = [float(decision.features.get(name, 0.0)) for decision in decisions]
        for threshold in threshold_candidates(values, max_thresholds):
            for direction in ("ge", "le"):
                switches = [
                    float(decision.features.get(name, 0.0)) >= threshold
                    if direction == "ge"
                    else float(decision.features.get(name, 0.0)) <= threshold
                    for decision in decisions
                ]
                summary = summarize_policy(decisions, switches)
                summary.update(
                    {
                        "gate_type": "single_feature",
                        "feature": name,
                        "direction": direction,
                        "threshold": threshold,
                    }
                )
                rows.append(summary)
    return rows


def standardize(decisions: list[Decision], feature_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = [float(decision.features.get(name, 0.0)) for decision in decisions]
        means[name] = statistics.fmean(values) if values else 0.0
        scales[name] = statistics.pstdev(values) if len(values) > 1 else 1.0
        if scales[name] < 1e-9:
            scales[name] = 1.0
    return means, scales


def train_linear_gate(
    decisions: list[Decision],
    feature_names: list[str],
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    means, scales = standardize(decisions, feature_names)
    xs = []
    ys = []
    weights_by_row = []
    for decision in decisions:
        current_score = row_teacher_score(decision.current)
        challenger_score = row_teacher_score(decision.challenger)
        delta = challenger_score - current_score
        xs.append(
            [
                (float(decision.features.get(name, 0.0)) - means[name]) / scales[name]
                for name in feature_names
            ]
        )
        ys.append(1.0 if delta > 0.0 else 0.0)
        weights_by_row.append(1.0 + min(abs(delta), 5.0))
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    row_weight = torch.tensor(weights_by_row, dtype=torch.float32)
    weights = torch.zeros(len(feature_names), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=lr)
    final_loss = 0.0
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        logits = x.mv(weights) + intercept
        loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
        )
        loss = (loss_vec * row_weight).mean() + float(l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    return {
        "kind": "t1_selector_arbitration_linear_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": feature_names,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "seed": int(seed),
        "final_loss": final_loss,
    }


def gate_probability(decision: Decision, gate: dict[str, Any]) -> float:
    value = float(gate.get("intercept", 0.0))
    means = gate.get("means") or {}
    scales = gate.get("scales") or {}
    for name, weight in zip(gate.get("features") or [], gate.get("weights") or []):
        raw = float(decision.features.get(name, 0.0))
        raw = (raw - float(means.get(name, 0.0))) / max(float(scales.get(name, 1.0)), 1e-9)
        value += float(weight) * raw
    if value >= 0:
        ez = math.exp(-value)
        return 1.0 / (1.0 + ez)
    ez = math.exp(value)
    return ez / (1.0 + ez)


def sweep_linear_gate(decisions: list[Decision], gate: dict[str, Any], step: float) -> list[dict[str, Any]]:
    step = max(float(step), 0.001)
    count = int(round(1.0 / step))
    thresholds = sorted({round(i * step, 10) for i in range(count + 1)})
    out = []
    probabilities = [gate_probability(decision, gate) for decision in decisions]
    for threshold in thresholds:
        summary = summarize_policy(decisions, [probability >= threshold for probability in probabilities])
        summary.update(
            {
                "gate_type": "linear",
                "threshold": threshold,
            }
        )
        out.append(summary)
    return out


def choose_best_gate(rows: list[dict[str, Any]], *, regret_tolerance: float) -> dict[str, Any]:
    viable = [
        row
        for row in rows
        if float(row["regret_delta_vs_current"]) <= float(regret_tolerance)
        and int(row["switched"]) > 0
    ]
    candidates = viable or rows
    return max(
        candidates,
        key=lambda row: (
            float(row["top1"]),
            -float(row["avg_regret"]),
            -float(row["bad_switch_regret"]),
            -float(row["switch_rate"]),
        ),
    )


def apply_gate_from_summary(decisions: list[Decision], summary: dict[str, Any], gate: dict[str, Any] | None) -> dict[str, Any]:
    gate_type = summary.get("gate_type")
    if gate_type == "linear":
        if gate is None:
            raise ValueError("linear gate summary requires gate")
        threshold = float(summary["threshold"])
        switches = [gate_probability(decision, gate) >= threshold for decision in decisions]
        out = summarize_policy(decisions, switches)
        out.update({"gate_type": "linear", "threshold": threshold})
        return out
    if gate_type == "single_feature":
        feature = str(summary["feature"])
        direction = str(summary["direction"])
        threshold = float(summary["threshold"])
        switches = [
            float(decision.features.get(feature, 0.0)) >= threshold
            if direction == "ge"
            else float(decision.features.get(feature, 0.0)) <= threshold
            for decision in decisions
        ]
        out = summarize_policy(decisions, switches)
        out.update(
            {
                "gate_type": "single_feature",
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
            }
        )
        return out
    raise ValueError(f"unsupported gate type: {gate_type}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate current-vs-challenger T1 selector arbitration")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--current-selector", required=True)
    parser.add_argument("--challenger-selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_DELTA_FEATURES))
    parser.add_argument("--extra-features", default="")
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--max-thresholds", type=int, default=31)
    parser.add_argument("--linear-epochs", type=int, default=2000)
    parser.add_argument("--linear-lr", type=float, default=0.03)
    parser.add_argument("--linear-l2", type=float, default=1e-4)
    parser.add_argument("--linear-threshold-step", type=float, default=0.01)
    parser.add_argument("--regret-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    current_selector = iter_json(Path(args.current_selector))
    challenger_selector = iter_json(Path(args.challenger_selector))
    feature_names = [part.strip() for part in args.features.split(",") if part.strip()]
    feature_names.extend(part.strip() for part in args.extra_features.split(",") if part.strip())
    feature_names = list(dict.fromkeys(feature_names))

    train_decisions = build_decisions(
        [Path(path) for path in args.train],
        current_selector=current_selector,
        challenger_selector=challenger_selector,
        min_teacher_margin=args.min_teacher_margin,
    )
    if not train_decisions:
        raise SystemExit("No train decisions found")

    single_feature_rows = sweep_single_feature(
        train_decisions,
        feature_names,
        max_thresholds=args.max_thresholds,
    )
    linear_gate = train_linear_gate(
        train_decisions,
        feature_names,
        epochs=args.linear_epochs,
        lr=args.linear_lr,
        l2=args.linear_l2,
        seed=args.seed,
    )
    linear_rows = sweep_linear_gate(train_decisions, linear_gate, args.linear_threshold_step)
    all_gate_rows = single_feature_rows + linear_rows
    best_train = choose_best_gate(all_gate_rows, regret_tolerance=args.regret_tolerance)

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_inputs": args.train,
        "eval_inputs": args.eval,
        "current_selector": args.current_selector,
        "challenger_selector": args.challenger_selector,
        "min_teacher_margin": float(args.min_teacher_margin),
        "features": feature_names,
        "train_baseline": summarize_policy(train_decisions, [False for _ in train_decisions]),
        "train_challenger": summarize_policy(train_decisions, [True for _ in train_decisions]),
        "train_best_gate": best_train,
        "linear_gate": linear_gate,
        "gate_train_candidates": sorted(
            all_gate_rows,
            key=lambda row: (
                -float(row["top1"]),
                float(row["avg_regret"]),
                float(row["bad_switch_regret"]),
            ),
        )[:25],
        "eval": {},
    }

    for eval_path in args.eval:
        decisions = build_decisions(
            [Path(eval_path)],
            current_selector=current_selector,
            challenger_selector=challenger_selector,
            min_teacher_margin=args.min_teacher_margin,
        )
        replayed_candidates = [
            apply_gate_from_summary(decisions, row, linear_gate)
            for row in all_gate_rows
        ]
        diagnostic_best = choose_best_gate(
            replayed_candidates,
            regret_tolerance=args.regret_tolerance,
        )
        report["eval"][eval_path] = {
            "current": summarize_policy(decisions, [False for _ in decisions]),
            "challenger": summarize_policy(decisions, [True for _ in decisions]),
            "best_train_gate": apply_gate_from_summary(decisions, best_train, linear_gate),
            "diagnostic_best_gate_for_this_eval": diagnostic_best,
            "diagnostic_top_gate_replays": sorted(
                replayed_candidates,
                key=lambda row: (
                    -float(row["top1"]),
                    float(row["avg_regret"]),
                    float(row["bad_switch_regret"]),
                ),
            )[:10],
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
