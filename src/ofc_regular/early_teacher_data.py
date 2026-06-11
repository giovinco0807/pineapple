"""Teacher-data generation for Turn1 and opening states."""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal

from .action_space import generate_actions, generate_turn_actions
from .cards import create_deck, validate_cards
from .policy import action_to_json, board_to_json, phase_for_card_count, policy_sample
from .state import Board
from .turn3_model import Turn3RidgeModel, load_action_value_model, predict_sample_maxes

Phase = Literal["turn1", "opening"]


def sample_early_state(
    rng: random.Random,
    *,
    phase: Phase,
) -> tuple[Board, tuple[str, ...], tuple[str, ...]]:
    deck = create_deck(shuffle=True, rng=rng)
    if phase == "opening":
        return Board.from_rows(), tuple(deck[:5]), tuple(deck[5:])
    if phase != "turn1":
        raise ValueError(f"unknown phase: {phase}")

    slots = ["top"] * 3 + ["middle"] * 5 + ["bottom"] * 5
    occupied = set(rng.sample(range(len(slots)), 5))
    rows = {"top": [], "middle": [], "bottom": []}
    board_cards = iter(deck[:5])
    for index, row in enumerate(slots):
        if index in occupied:
            rows[row].append(next(board_cards))
    board = Board.from_rows(rows["top"], rows["middle"], rows["bottom"])
    return board, tuple(deck[5:8]), tuple(deck[8:])


def build_early_sample(
    *,
    rng: random.Random,
    sample_id: int,
    phase: Phase,
    downstream_model: Turn3RidgeModel,
    future_samples: int,
    action_batch_size: int = 0,
) -> dict:
    board, dealt, remaining = sample_early_state(rng, phase=phase)
    ranked = evaluate_bootstrap_actions(
        board,
        dealt,
        downstream_model=downstream_model,
        remaining_cards=remaining,
        future_samples=future_samples,
        action_batch_size=action_batch_size,
        rng=rng,
    )
    if not ranked:
        raise RuntimeError(f"sampled {phase} state has no legal actions")
    score_gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
    return {
        "sample_id": sample_id,
        "rule_set": "regular",
        "phase": phase_for_card_count(board.card_count()),
        "board": board_to_json(board),
        "dealt": list(dealt),
        "best_action": 0,
        "score_gap": score_gap,
        "actions": ranked,
    }


def evaluate_bootstrap_actions(
    board: Board,
    dealt_cards: Iterable[str],
    *,
    downstream_model: Turn3RidgeModel,
    remaining_cards: Iterable[str],
    future_samples: int,
    action_batch_size: int = 0,
    rng: random.Random,
) -> list[dict]:
    dealt = tuple(dealt_cards)
    remaining = tuple(remaining_cards)
    validate_cards((*board.all_cards(), *dealt, *remaining))
    actions = generate_actions(board, dealt)
    futures = select_future_deals(remaining, future_samples=future_samples, rng=rng)
    action_payloads: list[dict] = []
    action_next_samples: list[list[dict]] = []

    for action in actions:
        next_board = board.place(action.placements)
        next_samples: list[dict] = []
        for future in futures:
            next_actions = generate_turn_actions(next_board, future)
            if not next_actions:
                continue
            next_samples.append(policy_sample(next_board, future, next_actions))
        if next_samples:
            action_payloads.append(action_to_json(board, action))
            action_next_samples.append(next_samples)

    evaluated: list[dict] = []
    batch_size = len(action_payloads) if action_batch_size <= 0 else action_batch_size
    batch_size = max(1, batch_size)
    for start in range(0, len(action_payloads), batch_size):
        end = start + batch_size
        batched_samples = [
            sample
            for samples in action_next_samples[start:end]
            for sample in samples
        ]
        all_scores = predict_sample_maxes(downstream_model, batched_samples)
        cursor = 0
        for payload, next_samples in zip(action_payloads[start:end], action_next_samples[start:end]):
            sample_count = len(next_samples)
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
    combo_count = len(remaining) * (len(remaining) - 1) * (len(remaining) - 2) // 6
    if future_samples >= combo_count:
        return tuple(combinations(remaining, 3))
    return tuple(tuple(rng.sample(remaining, 3)) for _ in range(future_samples))


def write_early_dataset(
    *,
    output: Path,
    samples: int,
    seed: int,
    phase: Phase,
    downstream_model: Turn3RidgeModel,
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
            sample = build_early_sample(
                rng=rng,
                sample_id=sample_id,
                phase=phase,
                downstream_model=downstream_model,
                future_samples=future_samples,
                action_batch_size=action_batch_size,
            )
            if sample["score_gap"] < min_score_gap:
                continue
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            sample_id += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("turn1", "opening"), required=True)
    parser.add_argument("--downstream-model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--future-samples", type=int, default=64)
    parser.add_argument(
        "--action-batch-size",
        type=int,
        default=0,
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
    downstream_model = load_action_value_model(args.downstream_model)
    write_early_dataset(
        output=args.output,
        samples=args.samples,
        seed=args.seed,
        phase=args.phase,
        downstream_model=downstream_model,
        future_samples=args.future_samples,
        action_batch_size=args.action_batch_size,
        min_score_gap=args.min_score_gap,
    )
    print(f"wrote {args.samples} {args.phase} samples to {args.output}")


if __name__ == "__main__":
    main()
