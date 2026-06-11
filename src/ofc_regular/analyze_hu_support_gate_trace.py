"""Analyze a support-model gate on HU Turn3 override traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .action_space import generate_turn_actions
from .hu_turn3_model import hu_policy_sample, load_hu_action_value_model
from .state import Board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="override trace JSONL")
    parser.add_argument("--support-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--thresholds",
        default="0,2,4,6,8",
        help="Comma-separated support-margin thresholds to sweep.",
    )
    parser.add_argument(
        "--paired-seeds",
        type=int,
        required=True,
        help="Number of paired seeds in the trace run. Used for EV/hand approximation.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def generated_action_key(action: Any) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.placements)
    discards = tuple(str(card) for card in action.discards)
    return placements, discards


def add_support_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    support_model: Any,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        board = Board.from_rows(**row["board"])
        opponent_board = Board.from_rows(**row["opponent_board"])
        dealt = tuple(row["dealt"])
        actions = generate_turn_actions(board, dealt)
        sample = hu_policy_sample(
            board,
            dealt,
            actions,
            opponent_board=opponent_board,
            dead_cards=row.get("dead_cards", ()),
            seat=row.get("seat", "first"),
            to_act_order=row.get("to_act_order", row.get("seat", "first")),
        )
        predictions = support_model.predict_sample(sample)
        chosen_index = int(row["chosen_index"])
        baseline_index = int(row["baseline_index"])

        if (
            chosen_index >= len(actions)
            or baseline_index >= len(actions)
            or generated_action_key(actions[chosen_index]) != action_key(row["chosen_action"])
            or generated_action_key(actions[baseline_index]) != action_key(row["baseline_action"])
        ):
            action_index_by_key = {generated_action_key(action): index for index, action in enumerate(actions)}
            chosen_index = action_index_by_key[action_key(row["chosen_action"])]
            baseline_index = action_index_by_key[action_key(row["baseline_action"])]

        support_margin = float(predictions[chosen_index]) - float(predictions[baseline_index])
        support_best_index = int(max(range(len(predictions)), key=lambda index: float(predictions[index])))
        enriched.append(
            {
                **row,
                "support_margin": support_margin,
                "support_chosen_score": float(predictions[chosen_index]),
                "support_baseline_score": float(predictions[baseline_index]),
                "support_best_is_chosen": support_best_index == chosen_index,
                "support_best_is_baseline": support_best_index == baseline_index,
            }
        )
    return enriched


def analyze_support_thresholds(
    rows: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
    paired_seeds: int,
) -> list[dict[str, float]]:
    if paired_seeds <= 0:
        raise ValueError("paired_seeds must be positive")

    results: list[dict[str, float]] = []
    for threshold in thresholds:
        selected = [row for row in rows if float(row["support_margin"]) >= threshold]
        deltas = [float(row.get("counterfactual_delta_vs_baseline", 0.0)) for row in selected]
        support_margins = [float(row["support_margin"]) for row in selected]
        wins = sum(1 for delta in deltas if delta > 1e-9)
        losses = sum(1 for delta in deltas if delta < -1e-9)
        ties = len(deltas) - wins - losses
        delta_sum = sum(deltas)
        support_margin_sum = sum(support_margins)
        results.append(
            {
                "support_margin_threshold": float(threshold),
                "kept_overrides": float(len(selected)),
                "kept_override_rate": float(len(selected) / (paired_seeds * 2.0)),
                "delta_sum": float(delta_sum),
                "delta_avg": float(delta_sum / len(deltas)) if deltas else 0.0,
                "approx_ev_per_hand": float(delta_sum / 2.0 / paired_seeds),
                "wins": float(wins),
                "losses": float(losses),
                "ties": float(ties),
                "support_margin_avg": float(support_margin_sum / len(support_margins))
                if support_margins
                else 0.0,
                "support_best_is_chosen": float(
                    sum(1 for row in selected if row.get("support_best_is_chosen"))
                ),
                "support_best_is_baseline": float(
                    sum(1 for row in selected if row.get("support_best_is_baseline"))
                ),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    rows = read_trace_rows(args.input)
    support_model = load_hu_action_value_model(args.support_model)
    enriched_rows = add_support_predictions(rows, support_model=support_model)
    summary = {
        "input": str(args.input),
        "support_model": str(args.support_model),
        "paired_seeds": args.paired_seeds,
        "trace_rows": len(rows),
        "thresholds": analyze_support_thresholds(
            enriched_rows,
            thresholds=thresholds,
            paired_seeds=args.paired_seeds,
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
