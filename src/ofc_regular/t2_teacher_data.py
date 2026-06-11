"""Teacher-data generation for Turn2 7-card states using the Turn3 model."""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .action_space import generate_turn_actions
from .cards import create_deck, validate_cards
from .policy import action_to_json, board_to_json, policy_sample
from .state import Board
from .turn3_model import Turn3RidgeModel, load_action_value_model, predict_sample_maxes


def sample_t2_state(rng: random.Random) -> tuple[Board, tuple[str, str, str], tuple[str, ...]]:
    """Sample a 7-card board and current 3-card deal."""
    deck = create_deck(shuffle=True, rng=rng)
    slots = ["top"] * 3 + ["middle"] * 5 + ["bottom"] * 5
    occupied = set(rng.sample(range(len(slots)), 7))
    rows = {"top": [], "middle": [], "bottom": []}
    board_cards = iter(deck[:7])
    for index, row in enumerate(slots):
        if index in occupied:
            rows[row].append(next(board_cards))
    board = Board.from_rows(rows["top"], rows["middle"], rows["bottom"])
    dealt = tuple(deck[7:10])
    remaining = tuple(deck[10:])
    return board, dealt, remaining  # type: ignore[return-value]


def build_t2_sample(
    *,
    rng: random.Random,
    sample_id: int,
    model: Turn3RidgeModel,
    future_samples: int,
    action_batch_size: int = 1,
) -> dict:
    board, dealt, remaining = sample_t2_state(rng)
    ranked = evaluate_t2_actions(
        board,
        dealt,
        model=model,
        remaining_cards=remaining,
        future_samples=future_samples,
        action_batch_size=action_batch_size,
        rng=rng,
    )
    if not ranked:
        raise RuntimeError("sampled T2 state has no legal actions")
    score_gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
    return {
        "sample_id": sample_id,
        "rule_set": "regular",
        "phase": "turn2_7card",
        "board": board_to_json(board),
        "dealt": list(dealt),
        "best_action": 0,
        "score_gap": score_gap,
        "actions": ranked,
    }


def evaluate_t2_actions(
    board: Board,
    dealt_cards: Iterable[str],
    *,
    model: Turn3RidgeModel,
    remaining_cards: Iterable[str],
    future_samples: int,
    action_batch_size: int = 1,
    rng: random.Random,
) -> list[dict]:
    dealt = tuple(dealt_cards)
    remaining = tuple(remaining_cards)
    validate_cards((*board.all_cards(), *dealt, *remaining))
    actions = generate_turn_actions(board, dealt)
    futures = select_future_deals(remaining, future_samples=future_samples, rng=rng)
    action_payloads: list[dict] = []
    action_turn3_samples: list[list[dict]] = []

    for action in actions:
        next_board = board.place(action.placements)
        turn3_samples: list[dict] = []
        for future in futures:
            turn3_actions = generate_turn_actions(next_board, future)
            if not turn3_actions:
                continue
            turn3_samples.append(policy_sample(next_board, future, turn3_actions))
        if turn3_samples:
            action_payloads.append(action_to_json(board, action))
            action_turn3_samples.append(turn3_samples)

    evaluated: list[dict] = []
    batch_size = len(action_payloads) if action_batch_size <= 0 else action_batch_size
    batch_size = max(1, batch_size)
    for start in range(0, len(action_payloads), batch_size):
        end = start + batch_size
        batched_samples = [
            sample
            for samples in action_turn3_samples[start:end]
            for sample in samples
        ]
        all_scores = predict_sample_maxes(model, batched_samples)
        cursor = 0
        for payload, turn3_samples in zip(action_payloads[start:end], action_turn3_samples[start:end]):
            sample_count = len(turn3_samples)
            scores = all_scores[cursor : cursor + sample_count]
            cursor += sample_count
            payload["score"] = sum(scores) / len(scores)
            payload["future_count"] = len(scores)
            evaluated.append(payload)

    evaluated.sort(key=lambda item: item["score"], reverse=True)
    return evaluated


def select_future_deals(
    remaining_cards: Iterable[str],
    *,
    future_samples: int,
    rng: random.Random,
) -> tuple[tuple[str, str, str], ...]:
    remaining = tuple(remaining_cards)
    if future_samples == 0:
        return tuple(combinations(remaining, 3))
    if future_samples < 0:
        raise ValueError("future_samples must be non-negative")
    if future_samples >= len(remaining) * (len(remaining) - 1) * (len(remaining) - 2) // 6:
        return tuple(combinations(remaining, 3))
    return tuple(tuple(rng.sample(remaining, 3)) for _ in range(future_samples))


def write_t2_dataset(
    *,
    output: Path,
    samples: int,
    seed: int,
    model: Turn3RidgeModel,
    future_samples: int,
    action_batch_size: int,
    min_score_gap: float,
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
            sample = build_t2_sample(
                rng=rng,
                sample_id=sample_id,
                model=model,
                future_samples=future_samples,
                action_batch_size=action_batch_size,
            )
            if sample["score_gap"] < min_score_gap:
                continue
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            sample_id += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn3-model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--future-samples", type=int, default=64)
    parser.add_argument(
        "--action-batch-size",
        type=int,
        default=1,
        help="actions to batch per downstream prediction call; <=0 batches all actions",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    model = load_action_value_model(args.turn3_model)
    write_t2_dataset(
        output=args.output,
        samples=args.samples,
        seed=args.seed,
        model=model,
        future_samples=args.future_samples,
        action_batch_size=args.action_batch_size,
        min_score_gap=args.min_score_gap,
    )
    print(f"wrote {args.samples} T2 samples to {args.output}")


if __name__ == "__main__":
    main()
