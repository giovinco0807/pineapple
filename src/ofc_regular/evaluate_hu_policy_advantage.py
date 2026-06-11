"""Compare HU Turn3 choices against the self-board Turn3 baseline on teacher EV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .hu_turn3_model import load_hu_action_value_model, read_teacher_samples, sample_to_matrix, split_samples
from .turn3_model import load_action_value_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--hu-model", type=Path, required=True)
    parser.add_argument("--baseline-turn3-model", type=Path, default=Path("models/turn3_stage6.pkl"))
    parser.add_argument("--hu-min-margin", type=float, default=0.0)
    parser.add_argument("--hu-max-self-regret", type=float)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
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


def evaluate_advantage(
    *,
    samples: Sequence[dict[str, Any]],
    hu_model: Any,
    baseline_model: Any,
    hu_min_margin: float,
    hu_max_self_regret: float | None,
) -> dict[str, float]:
    if not samples:
        return _empty_metrics()

    model_score_sum = 0.0
    baseline_score_sum = 0.0
    best_score_sum = 0.0
    override_count = 0
    override_win = 0
    override_loss = 0
    override_tie = 0
    margin_sum = 0.0
    action_count = 0

    for sample in samples:
        scores = [float(action["score"]) for action in sample["actions"]]
        action_count += len(scores)
        best_score = max(scores)
        best_score_sum += best_score

        baseline_idx = int(baseline_model.choose_action_index(self_board_sample(sample)))
        baseline_predictions = baseline_model.predict_sample(self_board_sample(sample))
        hu_features, _targets = sample_to_matrix(sample)
        hu_predictions = hu_model.predict_matrix(hu_features)
        hu_idx = int(hu_predictions.argmax())
        margin = float(hu_predictions[hu_idx] - hu_predictions[baseline_idx])
        self_regret = float(baseline_predictions[baseline_idx] - baseline_predictions[hu_idx])
        chosen_idx = (
            hu_idx
            if margin >= hu_min_margin
            and (hu_max_self_regret is None or self_regret <= hu_max_self_regret)
            else baseline_idx
        )

        model_score = scores[chosen_idx]
        baseline_score = scores[baseline_idx]
        model_score_sum += model_score
        baseline_score_sum += baseline_score
        margin_sum += margin

        if chosen_idx != baseline_idx:
            override_count += 1
            if model_score > baseline_score:
                override_win += 1
            elif model_score < baseline_score:
                override_loss += 1
            else:
                override_tie += 1

    count = float(len(samples))
    return {
        "samples": float(len(samples)),
        "actions": float(action_count),
        "hu_min_margin": float(hu_min_margin),
        "hu_max_self_regret": float(hu_max_self_regret) if hu_max_self_regret is not None else 0.0,
        "model_teacher_ev": model_score_sum / count,
        "baseline_teacher_ev": baseline_score_sum / count,
        "teacher_delta_vs_baseline": (model_score_sum - baseline_score_sum) / count,
        "model_avg_regret": (best_score_sum - model_score_sum) / count,
        "baseline_avg_regret": (best_score_sum - baseline_score_sum) / count,
        "override_rate": override_count / count,
        "override_wins": float(override_win),
        "override_losses": float(override_loss),
        "override_ties": float(override_tie),
        "override_win_rate": override_win / max(override_count, 1),
        "override_loss_rate": override_loss / max(override_count, 1),
        "avg_predicted_margin": margin_sum / count,
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "samples": 0.0,
        "actions": 0.0,
        "hu_min_margin": 0.0,
        "model_teacher_ev": 0.0,
        "baseline_teacher_ev": 0.0,
        "teacher_delta_vs_baseline": 0.0,
        "model_avg_regret": 0.0,
        "baseline_avg_regret": 0.0,
        "override_rate": 0.0,
        "override_wins": 0.0,
        "override_losses": 0.0,
        "override_ties": 0.0,
        "override_win_rate": 0.0,
        "override_loss_rate": 0.0,
        "avg_predicted_margin": 0.0,
    }


def main() -> None:
    args = parse_args()
    samples = read_teacher_samples(args.validation, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no HU validation samples")

    train_samples, holdout_samples = split_samples(samples, holdout_fraction=args.holdout, seed=args.seed)
    hu_model = load_hu_action_value_model(args.hu_model)
    baseline_model = load_action_value_model(args.baseline_turn3_model)
    result = {
        "validation": str(args.validation),
        "hu_model": str(args.hu_model),
        "baseline_turn3_model": str(args.baseline_turn3_model),
        "holdout_fraction": args.holdout,
        "seed": args.seed,
        "total_samples": len(samples),
        "train": evaluate_advantage(
            samples=train_samples,
            hu_model=hu_model,
            baseline_model=baseline_model,
            hu_min_margin=args.hu_min_margin,
            hu_max_self_regret=args.hu_max_self_regret,
        ),
        "holdout": evaluate_advantage(
            samples=holdout_samples,
            hu_model=hu_model,
            baseline_model=baseline_model,
            hu_min_margin=args.hu_min_margin,
            hu_max_self_regret=args.hu_max_self_regret,
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
