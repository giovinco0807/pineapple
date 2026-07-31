"""Leave-one-set-out robustness check for T1 selector arbitration gates."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.evaluate_t1_selector_arbitration import (  # noqa: E402
    DEFAULT_DELTA_FEATURES,
    apply_gate_from_summary,
    build_decisions,
    choose_best_gate,
    iter_json,
    summarize_policy,
    sweep_linear_gate,
    sweep_single_feature,
    train_linear_gate,
)


def parse_named_set(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--set must be name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--set name cannot be empty")
    return name, Path(path.strip())


def train_fold_gate(
    *,
    train_paths: list[Path],
    current_selector: dict[str, Any],
    challenger_selector: dict[str, Any],
    feature_names: list[str],
    min_teacher_margin: float,
    max_thresholds: int,
    linear_epochs: int,
    linear_lr: float,
    linear_l2: float,
    linear_threshold_step: float,
    regret_tolerance: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[Any]]:
    train_decisions = build_decisions(
        train_paths,
        current_selector=current_selector,
        challenger_selector=challenger_selector,
        min_teacher_margin=min_teacher_margin,
    )
    if not train_decisions:
        raise ValueError("fold has no train decisions")
    single_feature_rows = sweep_single_feature(
        train_decisions,
        feature_names,
        max_thresholds=max_thresholds,
    )
    linear_gate = train_linear_gate(
        train_decisions,
        feature_names,
        epochs=linear_epochs,
        lr=linear_lr,
        l2=linear_l2,
        seed=seed,
    )
    linear_rows = sweep_linear_gate(train_decisions, linear_gate, linear_threshold_step)
    gate_rows = single_feature_rows + linear_rows
    best_gate = choose_best_gate(gate_rows, regret_tolerance=regret_tolerance)
    train_summary = {
        "current": summarize_policy(train_decisions, [False for _ in train_decisions]),
        "challenger": summarize_policy(train_decisions, [True for _ in train_decisions]),
        "best_gate": best_gate,
    }
    return best_gate, linear_gate, gate_rows, train_decisions


def evaluate_gate(
    *,
    eval_path: Path,
    current_selector: dict[str, Any],
    challenger_selector: dict[str, Any],
    min_teacher_margin: float,
    best_gate: dict[str, Any],
    linear_gate: dict[str, Any],
    gate_rows: list[dict[str, Any]],
    regret_tolerance: float,
) -> dict[str, Any]:
    decisions = build_decisions(
        [eval_path],
        current_selector=current_selector,
        challenger_selector=challenger_selector,
        min_teacher_margin=min_teacher_margin,
    )
    replayed_candidates = [
        apply_gate_from_summary(decisions, row, linear_gate)
        for row in gate_rows
    ]
    return {
        "current": summarize_policy(decisions, [False for _ in decisions]),
        "challenger": summarize_policy(decisions, [True for _ in decisions]),
        "train_selected_gate": apply_gate_from_summary(decisions, best_gate, linear_gate),
        "diagnostic_best_gate_for_this_eval": choose_best_gate(
            replayed_candidates,
            regret_tolerance=regret_tolerance,
        ),
    }


def promotion_summary(folds: dict[str, Any], *, max_regret_delta: float, min_top1_delta: float) -> dict[str, Any]:
    heldout = {
        name: fold["heldout"]["train_selected_gate"]
        for name, fold in folds.items()
    }
    failures = []
    for name, summary in heldout.items():
        regret_delta = float(summary["regret_delta_vs_current"])
        top1_delta = float(summary["top1_delta_vs_current"])
        if regret_delta > max_regret_delta:
            failures.append(
                {
                    "set": name,
                    "reason": "regret_delta",
                    "value": regret_delta,
                    "limit": max_regret_delta,
                }
            )
        if top1_delta < min_top1_delta:
            failures.append(
                {
                    "set": name,
                    "reason": "top1_delta",
                    "value": top1_delta,
                    "limit": min_top1_delta,
                }
            )
    avg_top1_delta = (
        sum(float(summary["top1_delta_vs_current"]) for summary in heldout.values())
        / max(len(heldout), 1)
    )
    avg_regret_delta = (
        sum(float(summary["regret_delta_vs_current"]) for summary in heldout.values())
        / max(len(heldout), 1)
    )
    return {
        "promotion_ready": not failures,
        "max_regret_delta": max_regret_delta,
        "min_top1_delta": min_top1_delta,
        "avg_heldout_top1_delta": avg_top1_delta,
        "avg_heldout_regret_delta": avg_regret_delta,
        "failures": failures,
    }


def next_data_targets(folds: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for name, fold in folds.items():
        current = fold["heldout"]["current"]
        selected = fold["heldout"]["train_selected_gate"]
        diagnostic = fold["heldout"]["diagnostic_best_gate_for_this_eval"]
        selected_regret_delta = float(selected["regret_delta_vs_current"])
        diagnostic_regret_delta = float(diagnostic["regret_delta_vs_current"])
        diagnostic_top1_delta = float(diagnostic["top1_delta_vs_current"])
        gate_gap = diagnostic_regret_delta - selected_regret_delta
        needs_more_data = (
            selected_regret_delta > 0.0
            or float(selected["top1_delta_vs_current"]) <= 0.0
            or int(current["decisions"]) < 50
        )
        if not needs_more_data:
            continue
        targets.append(
            {
                "set": name,
                "decisions": int(current["decisions"]),
                "current_top1": float(current["top1"]),
                "current_avg_regret": float(current["avg_regret"]),
                "selected_gate_top1": float(selected["top1"]),
                "selected_gate_avg_regret": float(selected["avg_regret"]),
                "selected_gate_top1_delta": float(selected["top1_delta_vs_current"]),
                "selected_gate_regret_delta": selected_regret_delta,
                "diagnostic_best_gate_type": diagnostic.get("gate_type"),
                "diagnostic_best_feature": diagnostic.get("feature"),
                "diagnostic_best_direction": diagnostic.get("direction"),
                "diagnostic_best_threshold": diagnostic.get("threshold"),
                "diagnostic_best_top1_delta": diagnostic_top1_delta,
                "diagnostic_best_regret_delta": diagnostic_regret_delta,
                "diagnostic_vs_selected_regret_delta": gate_gap,
                "priority": (
                    "high"
                    if selected_regret_delta > 0.0 or int(current["decisions"]) < 50
                    else "medium"
                ),
                "recommended_action": (
                    "Add more T1 teacher rows matching this heldout distribution, then retrain selector arbitration."
                ),
            }
        )
    return sorted(
        targets,
        key=lambda row: (
            0 if row["priority"] == "high" else 1,
            -float(row["selected_gate_regret_delta"]),
            int(row["decisions"]),
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-set-out T1 selector arbitration checks")
    parser.add_argument("--set", dest="sets", action="append", required=True, type=parse_named_set)
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
    parser.add_argument("--max-heldout-regret-delta", type=float, default=0.0)
    parser.add_argument("--min-heldout-top1-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if len(args.sets) < 2:
        raise SystemExit("Need at least two --set entries")
    set_paths = dict(args.sets)
    current_selector = iter_json(Path(args.current_selector))
    challenger_selector = iter_json(Path(args.challenger_selector))
    feature_names = [part.strip() for part in args.features.split(",") if part.strip()]
    feature_names.extend(part.strip() for part in args.extra_features.split(",") if part.strip())
    feature_names = list(dict.fromkeys(feature_names))

    folds: dict[str, Any] = {}
    for heldout_name, heldout_path in set_paths.items():
        train_paths = [path for name, path in set_paths.items() if name != heldout_name]
        best_gate, linear_gate, gate_rows, train_decisions = train_fold_gate(
            train_paths=train_paths,
            current_selector=current_selector,
            challenger_selector=challenger_selector,
            feature_names=feature_names,
            min_teacher_margin=args.min_teacher_margin,
            max_thresholds=args.max_thresholds,
            linear_epochs=args.linear_epochs,
            linear_lr=args.linear_lr,
            linear_l2=args.linear_l2,
            linear_threshold_step=args.linear_threshold_step,
            regret_tolerance=args.regret_tolerance,
            seed=args.seed,
        )
        folds[heldout_name] = {
            "train_sets": [name for name in set_paths if name != heldout_name],
            "heldout_set": heldout_name,
            "train_decisions": len(train_decisions),
            "train": {
                "current": summarize_policy(train_decisions, [False for _ in train_decisions]),
                "challenger": summarize_policy(train_decisions, [True for _ in train_decisions]),
                "best_gate": best_gate,
            },
            "heldout": evaluate_gate(
                eval_path=heldout_path,
                current_selector=current_selector,
                challenger_selector=challenger_selector,
                min_teacher_margin=args.min_teacher_margin,
                best_gate=best_gate,
                linear_gate=linear_gate,
                gate_rows=gate_rows,
                regret_tolerance=args.regret_tolerance,
            ),
        }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sets": {name: str(path) for name, path in set_paths.items()},
        "current_selector": args.current_selector,
        "challenger_selector": args.challenger_selector,
        "features": feature_names,
        "min_teacher_margin": float(args.min_teacher_margin),
        "regret_tolerance": float(args.regret_tolerance),
        "folds": folds,
        "promotion": promotion_summary(
            folds,
            max_regret_delta=args.max_heldout_regret_delta,
            min_top1_delta=args.min_heldout_top1_delta,
        ),
        "next_data_targets": next_data_targets(folds),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
