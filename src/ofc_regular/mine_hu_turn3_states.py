"""Mine HU-aware Turn3 self-play states before expensive rollout labeling."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
)
from .cards import create_deck
from .hu_self_play_teacher_data import (
    hu_turn3_selection_metadata,
    passes_hu_turn3_selection,
    to_act_order_for,
)
from .hu_turn3_model import load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import board_to_json
from .state import Board


def build_hu_turn3_state_record(
    *,
    state_id: int,
    seed: int,
    hand_seed: int,
    hand_index: int,
    player: int,
    board: Board,
    opponent_board: Board,
    dealt_cards: Iterable[str],
    discarded_cards: Iterable[str],
    hero_seat: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    discarded = tuple(discarded_cards)
    return {
        "rule_set": "regular",
        "schema": "hu_stage1_state",
        "phase": "hu_turn3_9card",
        "source": "self_play_state_mining",
        "state_id": state_id,
        "seed": seed,
        "hand_seed": hand_seed,
        "hand_index": hand_index,
        "player": player,
        "hero_seat": hero_seat,
        "to_act_order": to_act_order_for(board, opponent_board),
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent_board),
        "dealt": list(dealt_cards),
        "discarded_cards": list(discarded),
        "visible_dead_cards": [*opponent_board.all_cards(), *discarded],
        "selection": selection,
    }


def mine_hu_turn3_states(
    *,
    output: Path,
    states: int,
    seed: int,
    policy_bundle: object,
    selection_hu_model: object,
    selection_compare_hu_model: object | None,
    opening_lookahead_samples: int,
    max_hands: int,
    selection_min_margin: float | None,
    selection_max_margin: float | None,
    selection_require_disagreement: bool,
    selection_require_compare_disagreement: bool,
    progress_every: int = 0,
) -> dict[str, Any]:
    if states <= 0:
        raise ValueError("states must be positive")
    if max_hands <= 0:
        raise ValueError("max_hands must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    hands = 0
    attempts = 0
    skipped_selection = 0
    started_at = time.time()

    with output.open("w", encoding="utf-8") as handle:
        while collected < states and hands < max_hands:
            hand_index = hands
            hand_seed = seed + hand_index
            hands += 1
            deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
            cursor = 0
            boards = [Board.from_rows(), Board.from_rows()]
            discarded_cards: list[str] = []
            policies = [
                build_policy(
                    "current",
                    policy_bundle,
                    seed=hand_seed * 2,
                    seat="first",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
                build_policy(
                    "current",
                    policy_bundle,
                    seed=hand_seed * 2 + 1,
                    seat="second",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
            ]

            for player in (0, 1):
                dealt = tuple(deck[cursor : cursor + 5])
                cursor += 5
                action = policies[player].choose_action(
                    boards[player],
                    dealt,
                    dead_cards=(*boards[1 - player].all_cards(), *discarded_cards),
                    opponent_board=boards[1 - player],
                )
                boards[player] = boards[player].place(action.placements)
                discarded_cards.extend(action.discards)

            for _round in range(1, 5):
                for player in (0, 1):
                    dealt = tuple(deck[cursor : cursor + 3])
                    cursor += 3
                    if boards[player].card_count() == 9 and collected < states:
                        attempts += 1
                        hero_seat = "first" if player == 0 else "second"
                        selection = hu_turn3_selection_metadata(
                            board=boards[player],
                            dealt_cards=dealt,
                            opponent_board=boards[1 - player],
                            dead_cards=discarded_cards,
                            hero_seat=hero_seat,
                            baseline_turn3_model=policy_bundle.turn3,
                            selection_hu_model=selection_hu_model,
                            compare_hu_model=selection_compare_hu_model,
                        )
                        if selection is not None and passes_hu_turn3_selection(
                            selection,
                            min_margin=selection_min_margin,
                            max_margin=selection_max_margin,
                            require_disagreement=selection_require_disagreement,
                            require_compare_disagreement=selection_require_compare_disagreement,
                        ):
                            record = build_hu_turn3_state_record(
                                state_id=collected,
                                seed=seed,
                                hand_seed=hand_seed,
                                hand_index=hand_index,
                                player=player,
                                board=boards[player],
                                opponent_board=boards[1 - player],
                                dealt_cards=dealt,
                                discarded_cards=discarded_cards,
                                hero_seat=hero_seat,
                                selection=selection,
                            )
                            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                            handle.flush()
                            collected += 1
                        else:
                            skipped_selection += 1

                    action = policies[player].choose_action(
                        boards[player],
                        dealt,
                        dead_cards=(*boards[1 - player].all_cards(), *discarded_cards),
                        opponent_board=boards[1 - player],
                    )
                    boards[player] = boards[player].place(action.placements)
                    discarded_cards.extend(action.discards)

            if progress_every > 0 and hands % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "states": collected,
                            "hands": hands,
                            "attempts": attempts,
                            "skipped_selection": skipped_selection,
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    if collected < states:
        raise RuntimeError(f"mined {collected}/{states} states after {hands} hands")
    return {
        "output": str(output),
        "states": collected,
        "hands": hands,
        "attempts": attempts,
        "skipped_selection": skipped_selection,
        "seed": seed,
        "selection": {
            "min_margin": selection_min_margin,
            "max_margin": selection_max_margin,
            "require_disagreement": selection_require_disagreement,
            "require_compare_disagreement": selection_require_compare_disagreement,
            "compare_enabled": selection_compare_hu_model is not None,
        },
        "elapsed_seconds": time.time() - started_at,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-hu-turn3-model", type=Path, required=True)
    parser.add_argument("--selection-compare-hu-turn3-model", type=Path)
    parser.add_argument("--selection-min-margin", type=float)
    parser.add_argument("--selection-max-margin", type=float)
    parser.add_argument("--selection-require-disagreement", action="store_true")
    parser.add_argument("--selection-require-compare-disagreement", action="store_true")
    parser.add_argument("--max-hands", type=int, default=1000000)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.selection_min_margin is not None
        and args.selection_max_margin is not None
        and args.selection_min_margin > args.selection_max_margin
    ):
        raise SystemExit("--selection-min-margin must be <= --selection-max-margin")
    if args.selection_require_compare_disagreement and args.selection_compare_hu_turn3_model is None:
        raise SystemExit("--selection-compare-hu-turn3-model is required for compare disagreement")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))
    paths = ModelPaths(
        opening=args.opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
    )
    policy_bundle = load_model_bundle(paths, {"current"})
    selection_hu_model = load_hu_action_value_model(args.selection_hu_turn3_model)
    selection_compare_hu_model = (
        load_hu_action_value_model(args.selection_compare_hu_turn3_model)
        if args.selection_compare_hu_turn3_model is not None
        else None
    )
    with _prediction_thread_context(args.prediction_threads):
        summary = mine_hu_turn3_states(
            output=args.output,
            states=args.states,
            seed=args.seed,
            policy_bundle=policy_bundle,
            selection_hu_model=selection_hu_model,
            selection_compare_hu_model=selection_compare_hu_model,
            opening_lookahead_samples=args.opening_lookahead_samples,
            max_hands=args.max_hands,
            selection_min_margin=args.selection_min_margin,
            selection_max_margin=args.selection_max_margin,
            selection_require_disagreement=args.selection_require_disagreement,
            selection_require_compare_disagreement=args.selection_require_compare_disagreement,
            progress_every=args.progress_every,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
