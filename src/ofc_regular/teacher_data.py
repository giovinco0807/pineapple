"""Teacher-data generation for regular OFC models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .cards import create_deck
from .state import Board
from .teacher import EvaluatedAction, evaluate_turn_actions, load_fl_ev


def sample_final_turn_state(rng: random.Random) -> tuple[Board, tuple[str, str, str]]:
    """Sample an 11-card board and the final 3-card deal."""
    deck = create_deck(rng=rng)
    slots = ["top"] * 3 + ["middle"] * 5 + ["bottom"] * 5
    occupied = set(rng.sample(range(len(slots)), 11))
    rows = {"top": [], "middle": [], "bottom": []}
    board_cards = iter(deck[:11])
    for index, row in enumerate(slots):
        if index in occupied:
            rows[row].append(next(board_cards))
    board = Board.from_rows(rows["top"], rows["middle"], rows["bottom"])
    dealt = tuple(deck[11:14])
    return board, dealt  # type: ignore[return-value]


def build_final_turn_sample(
    *,
    rng: random.Random,
    sample_id: int,
    fl_ev: dict[int, float],
) -> dict[str, Any]:
    board, dealt = sample_final_turn_state(rng)
    ranked = evaluate_turn_actions(board, dealt, fl_ev=fl_ev)
    if not ranked:
        raise RuntimeError("sampled final-turn state has no legal actions")
    score_gap = ranked[0].score - ranked[1].score if len(ranked) > 1 else 0.0
    return {
        "sample_id": sample_id,
        "rule_set": "regular",
        "phase": "final_turn",
        "board": board_to_json(board),
        "dealt": list(dealt),
        "best_action": 0,
        "score_gap": score_gap,
        "actions": [evaluated_action_to_json(item) for item in ranked],
    }


def write_final_turn_dataset(
    *,
    output: Path,
    samples: int,
    seed: int,
    fl_ev: dict[int, float],
    min_score_gap: float = 0.0,
) -> None:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    max_attempts = max(samples * 1000, 1000)
    with output.open("w", encoding="utf-8") as handle:
        sample_id = 0
        while sample_id < samples:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"only wrote {sample_id} samples after {max_attempts} attempts; "
                    "lower --min-score-gap"
                )
            sample = build_final_turn_sample(rng=rng, sample_id=sample_id, fl_ev=fl_ev)
            if sample["score_gap"] < min_score_gap:
                continue
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            sample_id += 1


def board_to_json(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def evaluated_action_to_json(item: EvaluatedAction) -> dict[str, Any]:
    board_score = item.board_score
    fl_entry = board_score.fl_entry
    return {
        "placements": [list(placement) for placement in item.action.placements],
        "discards": list(item.action.discards),
        "score": item.score,
        "next_board": board_to_json(item.board),
        "busted": board_score.busted,
        "royalty": {
            "top": board_score.top_royalty,
            "middle": board_score.middle_royalty,
            "bottom": board_score.bottom_royalty,
            "total": board_score.total_royalty,
        },
        "fl_entry": {
            "qualifies": fl_entry.qualifies,
            "card_count": fl_entry.card_count,
            "entry_type": fl_entry.entry_type,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--fl-ev-config", type=Path)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    fl_ev = load_fl_ev(args.fl_ev_config)
    write_final_turn_dataset(
        output=args.output,
        samples=args.samples,
        seed=args.seed,
        fl_ev=fl_ev,
        min_score_gap=args.min_score_gap,
    )
    print(f"wrote {args.samples} final-turn samples to {args.output}")


if __name__ == "__main__":
    main()
