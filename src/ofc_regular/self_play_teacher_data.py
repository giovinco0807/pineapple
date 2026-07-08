"""Collect teacher samples from the current self-play distribution."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Literal

from .action_space import generate_actions
from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
)
from .cards import ALL_CARDS, create_deck, validate_cards
from .early_teacher_data import evaluate_bootstrap_actions
from .policy import board_to_json, phase_for_card_count
from .state import Board
from .teacher import ExpectedAction, evaluate_two_turn_actions
from .turn3_model import load_action_value_model
from .visibility import HuDiscardTracker

Phase = Literal["opening", "turn1", "turn3"]


def remaining_for_teacher(
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
) -> tuple[str, ...]:
    dealt = tuple(dealt_cards)
    dead = tuple(dead_cards)
    used_cards = (*board.all_cards(), *dealt, *opponent_board.all_cards(), *dead)
    validate_cards(used_cards)
    used = set(used_cards)
    return tuple(card for card in ALL_CARDS if card not in used)


def build_self_play_sample(
    *,
    sample_id: int,
    phase: Phase,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    downstream_model: object,
    future_samples: int,
    action_batch_size: int,
    rng: random.Random,
) -> dict | None:
    dealt = tuple(dealt_cards)
    actions = generate_actions(board, dealt)
    if not actions:
        return None
    if phase == "turn3":
        ranked_t3 = evaluate_self_play_turn3_actions(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent_board,
            dead_cards=dead_cards,
            future_samples=future_samples,
            rng=rng,
        )
        if not ranked_t3:
            return None
        if all(item["non_bust_future_count"] == 0 for item in ranked_t3):
            return None
        score_gap = ranked_t3[0]["score"] - ranked_t3[1]["score"] if len(ranked_t3) > 1 else 0.0
        return {
            "sample_id": sample_id,
            "rule_set": "regular",
            "phase": phase_for_card_count(board.card_count()),
            "board": board_to_json(board),
            "dealt": list(dealt),
            "best_action": 0,
            "score_gap": score_gap,
            "actions": ranked_t3,
        }

    remaining = remaining_for_teacher(board, dealt, opponent_board, dead_cards)
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
        return None
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


def evaluate_self_play_turn3_actions(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    future_samples: int,
    rng: random.Random,
) -> list[dict]:
    max_future_deals = None if future_samples == 0 else future_samples
    ranked = evaluate_two_turn_actions(
        board,
        dealt_cards,
        dead_cards=(*opponent_board.all_cards(), *tuple(dead_cards)),
        max_future_deals=max_future_deals,
        seed=rng.randrange(2**63),
    )
    return [expected_action_to_json(item) for item in ranked]


def expected_action_to_json(item: ExpectedAction) -> dict:
    return {
        "placements": [list(placement) for placement in item.action.placements],
        "discards": list(item.action.discards),
        "score": item.score,
        "future_count": item.future_count,
        "non_bust_future_count": item.non_bust_future_count,
        "next_board": board_to_json(item.board),
    }


def collect_self_play_teacher_data(
    *,
    output: Path,
    phase: Phase,
    samples: int,
    seed: int,
    downstream_model: object,
    policy_bundle: object,
    future_samples: int,
    action_batch_size: int,
    opening_lookahead_samples: int,
    max_hands: int,
    min_score_gap: float,
) -> dict:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    hands = 0
    attempts = 0
    skipped_gap = 0
    with output.open("w", encoding="utf-8") as handle:
        while collected < samples and hands < max_hands:
            hand_seed = seed + hands
            hands += 1
            deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
            cursor = 0
            boards = [Board.from_rows(), Board.from_rows()]
            discards = HuDiscardTracker()
            policies = [
                build_policy(
                    "current",
                    policy_bundle,
                    seed=hand_seed * 2,
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
                build_policy(
                    "current",
                    policy_bundle,
                    seed=hand_seed * 2 + 1,
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
            ]

            for player in (0, 1):
                dealt = tuple(deck[cursor : cursor + 5])
                cursor += 5
                if phase == "opening" and collected < samples:
                    attempts += 1
                    sample = build_self_play_sample(
                        sample_id=collected,
                        phase=phase,
                        board=boards[player],
                        dealt_cards=dealt,
                        opponent_board=boards[1 - player],
                        dead_cards=discards.own_discards(player),
                        downstream_model=downstream_model,
                        future_samples=future_samples,
                        action_batch_size=action_batch_size,
                        rng=rng,
                    )
                    if sample is not None:
                        if sample["score_gap"] >= min_score_gap:
                            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
                            collected += 1
                        else:
                            skipped_gap += 1
                action = policies[player].choose_action(
                    boards[player],
                    dealt,
                    dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                    opponent_board=boards[1 - player],
                )
                boards[player] = boards[player].place(action.placements)
                discards.record(player, action.discards)

            for _round in range(1, 5):
                for player in (0, 1):
                    dealt = tuple(deck[cursor : cursor + 3])
                    cursor += 3
                    should_collect_turn = (
                        (phase == "turn1" and boards[player].card_count() == 5)
                        or (phase == "turn3" and boards[player].card_count() == 9)
                    )
                    if should_collect_turn and collected < samples:
                        attempts += 1
                        sample = build_self_play_sample(
                            sample_id=collected,
                            phase=phase,
                            board=boards[player],
                            dealt_cards=dealt,
                            opponent_board=boards[1 - player],
                            dead_cards=discards.own_discards(player),
                            downstream_model=downstream_model,
                            future_samples=future_samples,
                            action_batch_size=action_batch_size,
                            rng=rng,
                        )
                        if sample is not None:
                            if sample["score_gap"] >= min_score_gap:
                                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
                                collected += 1
                            else:
                                skipped_gap += 1
                    action = policies[player].choose_action(
                        boards[player],
                        dealt,
                        dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                        opponent_board=boards[1 - player],
                    )
                    boards[player] = boards[player].place(action.placements)
                    discards.record(player, action.discards)

    if collected < samples:
        raise RuntimeError(f"collected {collected}/{samples} samples after {hands} hands")
    return {
        "output": str(output),
        "phase": phase,
        "samples": collected,
        "hands": hands,
        "attempts": attempts,
        "skipped_gap": skipped_gap,
        "seed": seed,
        "future_samples": future_samples,
        "min_score_gap": min_score_gap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("opening", "turn1", "turn3"), required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-samples", type=int, default=16)
    parser.add_argument("--action-batch-size", type=int, default=0)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--max-hands", type=int, default=1000000)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--downstream-model", type=Path)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ModelPaths(
        opening=args.opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
    )
    policy_bundle = load_model_bundle(paths, {"current"})
    downstream_path = args.downstream_model
    if downstream_path is None:
        downstream_path = args.turn1_model if args.phase == "opening" else args.turn2_model
    downstream_model = None if args.phase == "turn3" else load_action_value_model(downstream_path)
    summary = collect_self_play_teacher_data(
        output=args.output,
        phase=args.phase,
        samples=args.samples,
        seed=args.seed,
        downstream_model=downstream_model,
        policy_bundle=policy_bundle,
        future_samples=args.future_samples,
        action_batch_size=args.action_batch_size,
        opening_lookahead_samples=args.opening_lookahead_samples,
        max_hands=args.max_hands,
        min_score_gap=args.min_score_gap,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
