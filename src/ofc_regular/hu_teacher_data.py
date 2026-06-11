"""HU-aware stage1 teacher data for regular OFC.

This generator intentionally writes a separate schema from the existing
self-board action-value samples. It starts with Turn3-like hero states against a
fixed complete opponent board, then scores hero actions by sampled final-turn
search using heads-up terminal scoring.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .action_space import generate_turn_actions
from .cards import ALL_CARDS, create_deck, validate_cards
from .policy import action_to_json, board_to_json
from .state import Board
from .teacher import DEFAULT_FL_EV, evaluate_turn_actions


def sample_hu_turn3_state(rng: random.Random) -> tuple[Board, tuple[str, str, str], Board, tuple[str, ...]]:
    deck = create_deck(shuffle=True, rng=rng)
    hero_board = _sample_partial_board(deck[:9], rng, card_count=9)
    dealt = tuple(deck[9:12])
    opponent_board = Board.from_rows(deck[12:15], deck[15:20], deck[20:25])
    used = set(hero_board.all_cards()) | set(dealt) | set(opponent_board.all_cards())
    remaining = tuple(card for card in ALL_CARDS if card not in used)
    validate_cards((*hero_board.all_cards(), *dealt, *opponent_board.all_cards(), *remaining))
    return hero_board, dealt, opponent_board, remaining  # type: ignore[return-value]


def _sample_partial_board(cards: Iterable[str], rng: random.Random, *, card_count: int) -> Board:
    slots = ["top"] * 3 + ["middle"] * 5 + ["bottom"] * 5
    occupied = set(rng.sample(range(len(slots)), card_count))
    rows = {"top": [], "middle": [], "bottom": []}
    card_iter = iter(cards)
    for index, row in enumerate(slots):
        if index in occupied:
            rows[row].append(next(card_iter))
    return Board.from_rows(rows["top"], rows["middle"], rows["bottom"])


def select_future_deals(
    remaining_cards: Iterable[str],
    *,
    future_samples: int,
    rng: random.Random,
) -> tuple[tuple[str, str, str], ...]:
    remaining = tuple(remaining_cards)
    if future_samples < 0:
        raise ValueError("future_samples must be non-negative")
    if future_samples == 0:
        return tuple(combinations(remaining, 3))
    combo_count = len(remaining) * (len(remaining) - 1) * (len(remaining) - 2) // 6
    if future_samples >= combo_count:
        return tuple(combinations(remaining, 3))
    return tuple(tuple(rng.sample(remaining, 3)) for _ in range(future_samples))


def evaluate_hu_turn3_actions(
    board: Board,
    dealt_cards: Iterable[str],
    *,
    opponent_board: Board,
    remaining_cards: Iterable[str],
    future_samples: int,
    rng: random.Random,
) -> list[dict]:
    dealt = tuple(dealt_cards)
    remaining = tuple(remaining_cards)
    validate_cards((*board.all_cards(), *dealt, *opponent_board.all_cards(), *remaining))
    futures = select_future_deals(remaining, future_samples=future_samples, rng=rng)
    evaluated: list[dict] = []
    for action in generate_turn_actions(board, dealt):
        next_board = board.place(action.placements)
        score_sum = 0.0
        future_count = 0
        for future in futures:
            ranked = evaluate_turn_actions(
                next_board,
                future,
                opponent_board=opponent_board,
                fl_ev=DEFAULT_FL_EV,
            )
            if not ranked:
                continue
            score_sum += ranked[0].score
            future_count += 1
        if future_count:
            payload = action_to_json(board, action)
            payload["score"] = score_sum / future_count
            payload["future_count"] = future_count
            evaluated.append(payload)
    evaluated.sort(key=lambda item: item["score"], reverse=True)
    return evaluated


def build_hu_turn3_sample(
    *,
    rng: random.Random,
    sample_id: int,
    future_samples: int,
) -> dict:
    board, dealt, opponent_board, remaining = sample_hu_turn3_state(rng)
    ranked = evaluate_hu_turn3_actions(
        board,
        dealt,
        opponent_board=opponent_board,
        remaining_cards=remaining,
        future_samples=future_samples,
        rng=rng,
    )
    if not ranked:
        raise RuntimeError("sampled HU T3 state has no evaluable actions")
    score_gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
    seat = rng.choice(("first", "second"))
    return {
        "sample_id": sample_id,
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn3_9card",
        "seat": seat,
        "to_act_order": seat,
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent_board),
        "dead_cards": list(opponent_board.all_cards()),
        "dealt": list(dealt),
        "best_action": 0,
        "score_gap": score_gap,
        "actions": ranked,
    }


def write_hu_turn3_dataset(
    *,
    output: Path,
    samples: int,
    seed: int,
    future_samples: int,
    min_score_gap: float,
) -> dict:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    max_attempts = max(samples * 1000, 1000)
    with output.open("w", encoding="utf-8") as handle:
        sample_id = 0
        while sample_id < samples:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"only wrote {sample_id} samples after {max_attempts} attempts")
            sample = build_hu_turn3_sample(
                rng=rng,
                sample_id=sample_id,
                future_samples=future_samples,
            )
            if sample["score_gap"] < min_score_gap:
                continue
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            sample_id += 1
    return {
        "output": str(output),
        "samples": samples,
        "seed": seed,
        "future_samples": future_samples,
        "min_score_gap": min_score_gap,
        "attempts": attempts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-samples", type=int, default=64)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    summary = write_hu_turn3_dataset(
        output=args.output,
        samples=args.samples,
        seed=args.seed,
        future_samples=args.future_samples,
        min_score_gap=args.min_score_gap,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
