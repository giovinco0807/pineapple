"""Train an offline logistic gate between two T1 runtime result sets.

Inputs are paired result JSONL files, usually current hard-5s runtime vs a
challenger runtime.  The gate predicts whether the challenger final action
should replace the current final action using runtime-level features from
``evaluate_t1_runtime_rule_grid.py``.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.evaluate_t1_runtime_rule_grid import (  # noqa: E402
    DEFAULT_FEATURES,
    Decision,
    build_decisions,
    parse_set,
    result_score,
    summarize_policy,
)


@dataclass(frozen=True)
class NamedDecision:
    set_name: str
    decision: Decision


def load_named_decisions(raw_sets: list[str]) -> dict[str, list[Decision]]:
    parsed_sets = [parse_set(raw) for raw in raw_sets]
    return {
        name: build_decisions(current, challenger)
        for name, current, challenger in parsed_sets
    }


def flatten(decisions_by_set: dict[str, list[Decision]]) -> list[NamedDecision]:
    return [
        NamedDecision(set_name=name, decision=decision)
        for name, decisions in decisions_by_set.items()
        for decision in decisions
    ]


def standardize(rows: list[NamedDecision], features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in features:
        values = [float(row.decision.features.get(name, 0.0)) for row in rows]
        means[name] = statistics.fmean(values) if values else 0.0
        scales[name] = statistics.pstdev(values) if len(values) > 1 else 1.0
        if scales[name] < 1e-9:
            scales[name] = 1.0
    return means, scales


def vector(row: NamedDecision, features: list[str], means: dict[str, float], scales: dict[str, float]) -> list[float]:
    return [
        (float(row.decision.features.get(name, 0.0)) - float(means[name]))
        / max(float(scales[name]), 1e-9)
        for name in features
    ]


def train_gate(
    rows: list[NamedDecision],
    *,
    features: list[str],
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    weight_cap: float,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    means, scales = standardize(rows, features)
    xs = []
    ys = []
    row_weights = []
    for row in rows:
        current_score = result_score(row.decision.current)
        challenger_score = result_score(row.decision.challenger)
        delta = challenger_score - current_score
        xs.append(vector(row, features, means, scales))
        ys.append(1.0 if delta > 0.0 else 0.0)
        row_weights.append(1.0 + min(abs(delta), weight_cap))
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    weights_by_row = torch.tensor(row_weights, dtype=torch.float32)
    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=lr)
    last_loss = 0.0
    for _epoch in range(max(1, epochs)):
        optimizer.zero_grad()
        logits = x.mv(weights) + intercept
        loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss_vec * weights_by_row).mean() + float(l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
    return {
        "kind": "t1_runtime_challenger_logistic_gate",
        "features": features,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "seed": int(seed),
        "weight_cap": float(weight_cap),
        "final_loss": last_loss,
    }


def gate_probability(gate: dict[str, Any], decision: Decision) -> float:
    z = float(gate.get("intercept", 0.0))
    means = dict(gate.get("means") or {})
    scales = dict(gate.get("scales") or {})
    for name, weight in zip(gate.get("features") or [], gate.get("weights") or []):
        raw = float(decision.features.get(name, 0.0))
        raw = (raw - float(means.get(name, 0.0))) / max(float(scales.get(name, 1.0)), 1e-9)
        z += float(weight) * raw
    return float(torch.sigmoid(torch.tensor(z)).item())


def summarize_gate(decisions_by_set: dict[str, list[Decision]], gate: dict[str, Any], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, decisions in decisions_by_set.items():
        switches = [gate_probability(gate, decision) >= threshold for decision in decisions]
        out[name] = summarize_policy(decisions, switches)
    return out


def score_by_set(
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


def threshold_candidates(rows: list[NamedDecision], gate: dict[str, Any], max_thresholds: int) -> list[float]:
    probabilities = sorted(set(gate_probability(gate, row.decision) for row in rows))
    if not probabilities:
        return [0.5]
    candidates = [0.0, 0.5, 1.0]
    if len(probabilities) <= max_thresholds:
        candidates.extend(probabilities)
    else:
        for idx in range(max_thresholds):
            pos = round(idx * (len(probabilities) - 1) / max(max_thresholds - 1, 1))
            candidates.append(probabilities[int(pos)])
    return sorted(set(float(value) for value in candidates))


def choose_threshold(
    decisions_by_set: dict[str, list[Decision]],
    gate: dict[str, Any],
    *,
    guard_sets: set[str],
    target_sets: set[str],
    regret_tolerance: float,
    top1_tolerance: float,
    max_thresholds: int,
) -> dict[str, Any]:
    rows = flatten(decisions_by_set)
    candidates: list[dict[str, Any]] = []
    for threshold in threshold_candidates(rows, gate, max_thresholds):
        by_set = summarize_gate(decisions_by_set, gate, threshold)
        ok, score = score_by_set(
            by_set,
            guard_sets=guard_sets,
            target_sets=target_sets,
            regret_tolerance=regret_tolerance,
            top1_tolerance=top1_tolerance,
        )
        candidates.append({"threshold": threshold, "guard_ok": ok, "score": score, "by_set": by_set})
    return sorted(candidates, key=lambda row: (row["guard_ok"], row["score"]), reverse=True)[0]


def baseline(decisions_by_set: dict[str, list[Decision]]) -> dict[str, Any]:
    return {
        name: {
            "current": summarize_policy(decisions, [False for _ in decisions]),
            "challenger": summarize_policy(decisions, [True for _ in decisions]),
        }
        for name, decisions in decisions_by_set.items()
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a logistic gate between T1 runtime result sets")
    parser.add_argument("--train-set", action="append", required=True, help="NAME=CURRENT_RESULTS::CHALLENGER_RESULTS")
    parser.add_argument("--eval-set", action="append", default=[], help="NAME=CURRENT_RESULTS::CHALLENGER_RESULTS")
    parser.add_argument("--guard-sets", default="")
    parser.add_argument("--target-sets", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weight-cap", type=float, default=5.0)
    parser.add_argument("--regret-tolerance", type=float, default=0.0)
    parser.add_argument("--top1-tolerance", type=float, default=0.0)
    parser.add_argument("--max-thresholds", type=int, default=101)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    train_by_set = load_named_decisions(args.train_set)
    eval_by_set = load_named_decisions(args.eval_set) if args.eval_set else {}
    features = [part.strip() for part in args.features.split(",") if part.strip()]
    gate = train_gate(
        flatten(train_by_set),
        features=features,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        weight_cap=args.weight_cap,
    )

    all_train_names = set(train_by_set)
    guard_sets = {part.strip() for part in args.guard_sets.split(",") if part.strip()}
    target_sets = {part.strip() for part in args.target_sets.split(",") if part.strip()}
    if not guard_sets:
        guard_sets = all_train_names - target_sets
    if not target_sets:
        target_sets = all_train_names - guard_sets
    selected = choose_threshold(
        train_by_set,
        gate,
        guard_sets=guard_sets,
        target_sets=target_sets,
        regret_tolerance=args.regret_tolerance,
        top1_tolerance=args.top1_tolerance,
        max_thresholds=args.max_thresholds,
    )
    gate["threshold"] = float(selected["threshold"])
    gate["train_sets"] = args.train_set
    gate["eval_sets"] = args.eval_set
    gate["guard_sets"] = sorted(guard_sets)
    gate["target_sets"] = sorted(target_sets)
    report = {
        "gate": gate,
        "train_baseline": baseline(train_by_set),
        "train_selected_threshold": selected,
    }
    if eval_by_set:
        report["eval_baseline"] = baseline(eval_by_set)
        report["eval_selected_threshold"] = summarize_gate(eval_by_set, gate, float(gate["threshold"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
