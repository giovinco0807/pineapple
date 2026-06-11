"""Apply a self-board regret penalty to HU Turn3 teacher scores."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .hu_turn3_model import read_teacher_samples
from .turn3_model import load_action_value_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-turn3-model", type=Path, default=Path("models/turn3_stage6.pkl"))
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--free-regret", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def self_board_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_set": sample.get("rule_set", "regular"),
        "phase": "turn3_9card",
        "board": sample["board"],
        "dealt": sample["dealt"],
        "best_action": sample.get("best_action", 0),
        "score_gap": sample.get("score_gap", 0.0),
        "actions": sample["actions"],
    }


def penalize_sample(
    sample: dict[str, Any],
    *,
    baseline_model: Any,
    penalty_weight: float,
    free_regret: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    output = deepcopy(sample)
    predictions = baseline_model.predict_sample(self_board_sample(sample))
    best_prediction = float(max(predictions))
    changed_best = 0.0
    penalty_sum = 0.0
    max_penalty = 0.0
    original_best_key = _action_key(max(sample["actions"], key=lambda action: float(action["score"])))

    actions = []
    for action, self_prediction in zip(output["actions"], predictions):
        raw_score = float(action["score"])
        self_regret = max(0.0, best_prediction - float(self_prediction))
        penalty = penalty_weight * max(0.0, self_regret - free_regret)
        action["raw_score"] = raw_score
        action["self_model_score"] = float(self_prediction)
        action["self_regret"] = self_regret
        action["self_regret_penalty"] = penalty
        action["score"] = raw_score - penalty
        penalty_sum += penalty
        max_penalty = max(max_penalty, penalty)
        actions.append(action)

    actions.sort(key=lambda action: float(action["score"]), reverse=True)
    output["actions"] = actions
    output["best_action"] = 0
    output["score_gap"] = (
        float(actions[0]["score"]) - float(actions[1]["score"]) if len(actions) > 1 else 0.0
    )
    output["source"] = output.get("source", "unknown")
    output["self_regret_penalty"] = {
        "baseline_turn3_model": "provided",
        "penalty_weight": penalty_weight,
        "free_regret": free_regret,
    }
    changed_best = 1.0 if _action_key(actions[0]) != original_best_key else 0.0
    return output, {
        "actions": float(len(actions)),
        "penalty_sum": penalty_sum,
        "max_penalty": max_penalty,
        "changed_best": changed_best,
    }


def _action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def main() -> None:
    args = parse_args()
    if args.penalty_weight < 0:
        raise SystemExit("--penalty-weight must be non-negative")
    if args.free_regret < 0:
        raise SystemExit("--free-regret must be non-negative")
    baseline_model = load_action_value_model(args.baseline_turn3_model)
    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no HU teacher samples")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    action_count = 0.0
    penalty_sum = 0.0
    max_penalty = 0.0
    changed_best = 0.0
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            penalized, stats = penalize_sample(
                sample,
                baseline_model=baseline_model,
                penalty_weight=args.penalty_weight,
                free_regret=args.free_regret,
            )
            handle.write(json.dumps(penalized, separators=(",", ":")) + "\n")
            sample_count += 1
            action_count += stats["actions"]
            penalty_sum += stats["penalty_sum"]
            max_penalty = max(max_penalty, stats["max_penalty"])
            changed_best += stats["changed_best"]

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "baseline_turn3_model": str(args.baseline_turn3_model),
        "samples": sample_count,
        "actions": action_count,
        "penalty_weight": args.penalty_weight,
        "free_regret": args.free_regret,
        "avg_penalty_per_action": penalty_sum / max(action_count, 1.0),
        "max_penalty": max_penalty,
        "changed_best_rate": changed_best / max(sample_count, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
