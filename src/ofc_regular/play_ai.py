"""Smoke-play regular OFC AI policies."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from .cards import create_deck
from .hu_turn3_gate_model import load_hu_turn3_gate_model
from .hu_turn3_model import load_hu_action_value_model
from .policy import RegularAiPolicy, board_to_json
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score
from .turn3_model import load_action_value_model

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional sklearn runtime dependency
    threadpool_limits = None


@dataclass(frozen=True)
class HandResult:
    score_p0: float
    board_p0: Board
    board_p1: Board


def play_hand(
    *,
    seed: int,
    policy_p0: RegularAiPolicy,
    policy_p1: RegularAiPolicy,
    fl_ev: dict[int, float] | None = None,
) -> HandResult:
    rng = random.Random(seed)
    deck = create_deck(shuffle=True, rng=rng)
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    policies = [policy_p0, policy_p1]
    dead_cards: list[str] = []

    for player in (0, 1):
        dealt = deck[cursor : cursor + 5]
        cursor += 5
        action = policies[player].choose_action(
            boards[player],
            dealt,
            dead_cards=(*boards[1 - player].all_cards(), *dead_cards),
            opponent_board=boards[1 - player],
            hand_id=seed,
            game_id=seed,
            decision_seed=seed,
            street="T0",
        )
        boards[player] = boards[player].place(action.placements)
        dead_cards.extend(action.discards)

    for round_index in range(1, 5):
        for player in (0, 1):
            dealt = deck[cursor : cursor + 3]
            cursor += 3
            action = policies[player].choose_action(
                boards[player],
                dealt,
                dead_cards=(*boards[1 - player].all_cards(), *dead_cards),
                opponent_board=boards[1 - player],
                hand_id=seed,
                game_id=seed,
                decision_seed=seed,
                street=f"T{round_index}",
            )
            boards[player] = boards[player].place(action.placements)
            dead_cards.extend(action.discards)

    score, _board_score = terminal_score(
        boards[0],
        boards[1],
        fl_ev=fl_ev or DEFAULT_FL_EV,
    )
    return HandResult(score_p0=score, board_p0=boards[0], board_p1=boards[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-model", type=Path)
    parser.add_argument("--turn1-model", type=Path)
    parser.add_argument("--turn2-model", type=Path)
    parser.add_argument("--turn3-model", type=Path)
    parser.add_argument("--hu-turn3-model", type=Path)
    parser.add_argument("--hu-turn3-reference-model", type=Path)
    parser.add_argument("--hu-turn3-support-model", type=Path)
    parser.add_argument("--hu-turn3-gate-model", type=Path)
    parser.add_argument("--hu-turn3-min-margin", type=float, default=0.0)
    parser.add_argument("--hu-turn3-reference-min-margin", type=float, default=0.0)
    parser.add_argument("--hu-turn3-min-support-margin", type=float, default=0.0)
    parser.add_argument("--hu-turn3-min-gate-probability", type=float, default=0.0)
    parser.add_argument("--hu-turn3-max-self-regret", type=float)
    parser.add_argument("--disable-hu-turn3-stage7", action="store_true")
    parser.add_argument("--hu-turn3-decision-log", type=Path)
    parser.add_argument(
        "--opening-lookahead-samples",
        type=int,
        default=64,
        help=(
            "Number of next 3-card deals sampled when choosing an opening "
            "placement from the Turn1 model. Use 0 to enumerate all next deals."
        ),
    )
    parser.add_argument(
        "--prediction-threads",
        type=int,
        default=1,
        help=(
            "Thread limit for model prediction. Default 1 avoids OpenMP "
            "oversubscription when running games or teacher chunks. Use 0 for "
            "the library default."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _prediction_thread_context(prediction_threads: int):
    if prediction_threads < 0:
        raise ValueError("--prediction-threads must be non-negative")
    if prediction_threads == 0:
        return contextlib.nullcontext()
    os.environ.setdefault("OMP_NUM_THREADS", str(prediction_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(prediction_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(prediction_threads))
    if threadpool_limits is None:
        return contextlib.nullcontext()
    return threadpool_limits(limits=prediction_threads)


def main() -> None:
    args = parse_args()
    prediction_context = _prediction_thread_context(args.prediction_threads)
    opening_model = load_action_value_model(args.opening_model) if args.opening_model else None
    turn1_model = load_action_value_model(args.turn1_model) if args.turn1_model else None
    turn2_model = load_action_value_model(args.turn2_model) if args.turn2_model else None
    turn3_model = load_action_value_model(args.turn3_model) if args.turn3_model else None
    hu_turn3_model = _safe_load_hu_action_value_model(args.hu_turn3_model) if args.hu_turn3_model else None
    hu_turn3_reference_model = (
        _safe_load_hu_action_value_model(args.hu_turn3_reference_model)
        if args.hu_turn3_reference_model
        else None
    )
    hu_turn3_support_model = (
        _safe_load_hu_action_value_model(args.hu_turn3_support_model)
        if args.hu_turn3_support_model
        else None
    )
    hu_turn3_gate_model = (
        _safe_load_hu_turn3_gate_model(args.hu_turn3_gate_model)
        if args.hu_turn3_gate_model
        else None
    )
    hu_turn3_min_margin = args.hu_turn3_min_margin
    hu_turn3_reference_min_margin = args.hu_turn3_reference_min_margin
    if args.hu_turn3_model and hu_turn3_model is None and hu_turn3_reference_model is not None:
        hu_turn3_model = hu_turn3_reference_model
        hu_turn3_reference_model = None
        hu_turn3_min_margin = hu_turn3_reference_min_margin
        hu_turn3_reference_min_margin = 0.0
    results = []
    total = 0.0
    with prediction_context:
        for game in range(args.games):
            hand_seed = args.seed + game
            policy_p0 = RegularAiPolicy(
                opening_model=opening_model,
                turn1_model=turn1_model,
                turn2_model=turn2_model,
                turn3_model=turn3_model,
                hu_turn3_model=hu_turn3_model,
                hu_turn3_reference_model=hu_turn3_reference_model,
                hu_turn3_support_model=hu_turn3_support_model,
                hu_turn3_gate_model=hu_turn3_gate_model,
                hu_turn3_min_margin=hu_turn3_min_margin,
                hu_turn3_reference_min_margin=hu_turn3_reference_min_margin,
                hu_turn3_min_support_margin=args.hu_turn3_min_support_margin,
                hu_turn3_min_gate_probability=args.hu_turn3_min_gate_probability,
                hu_turn3_max_self_regret=args.hu_turn3_max_self_regret,
                hu_turn3_stage7_enabled=not args.disable_hu_turn3_stage7,
                hu_turn3_decision_log_path=args.hu_turn3_decision_log,
                seat="first",
                seed=hand_seed * 2,
                opening_lookahead_samples=args.opening_lookahead_samples,
            )
            policy_p1 = RegularAiPolicy(
                opening_model=opening_model,
                turn1_model=turn1_model,
                turn2_model=turn2_model,
                turn3_model=turn3_model,
                hu_turn3_model=hu_turn3_model,
                hu_turn3_reference_model=hu_turn3_reference_model,
                hu_turn3_support_model=hu_turn3_support_model,
                hu_turn3_gate_model=hu_turn3_gate_model,
                hu_turn3_min_margin=hu_turn3_min_margin,
                hu_turn3_reference_min_margin=hu_turn3_reference_min_margin,
                hu_turn3_min_support_margin=args.hu_turn3_min_support_margin,
                hu_turn3_min_gate_probability=args.hu_turn3_min_gate_probability,
                hu_turn3_max_self_regret=args.hu_turn3_max_self_regret,
                hu_turn3_stage7_enabled=not args.disable_hu_turn3_stage7,
                hu_turn3_decision_log_path=args.hu_turn3_decision_log,
                seat="second",
                seed=hand_seed * 2 + 1,
                opening_lookahead_samples=args.opening_lookahead_samples,
            )
            result = play_hand(seed=hand_seed, policy_p0=policy_p0, policy_p1=policy_p1)
            total += result.score_p0
            results.append(
                {
                    "game": game,
                    "seed": hand_seed,
                    "score_p0": result.score_p0,
                    "board_p0": board_to_json(result.board_p0),
                    "board_p1": board_to_json(result.board_p1),
                }
            )

    summary = {
        "games": args.games,
        "seed": args.seed,
        "opening_model": str(args.opening_model) if args.opening_model else None,
        "turn1_model": str(args.turn1_model) if args.turn1_model else None,
        "turn2_model": str(args.turn2_model) if args.turn2_model else None,
        "turn3_model": str(args.turn3_model) if args.turn3_model else None,
        "hu_turn3_model": str(args.hu_turn3_model) if args.hu_turn3_model else None,
        "hu_turn3_reference_model": str(args.hu_turn3_reference_model)
        if args.hu_turn3_reference_model
        else None,
        "hu_turn3_support_model": str(args.hu_turn3_support_model)
        if args.hu_turn3_support_model
        else None,
        "hu_turn3_gate_model": str(args.hu_turn3_gate_model)
        if args.hu_turn3_gate_model
        else None,
        "hu_turn3_min_margin": hu_turn3_min_margin,
        "hu_turn3_reference_min_margin": hu_turn3_reference_min_margin,
        "hu_turn3_min_support_margin": args.hu_turn3_min_support_margin,
        "hu_turn3_min_gate_probability": args.hu_turn3_min_gate_probability,
        "hu_turn3_max_self_regret": args.hu_turn3_max_self_regret,
        "hu_turn3_stage7_enabled": not args.disable_hu_turn3_stage7,
        "hu_turn3_decision_log": str(args.hu_turn3_decision_log)
        if args.hu_turn3_decision_log
        else None,
        "opening_lookahead_samples": args.opening_lookahead_samples,
        "prediction_threads": args.prediction_threads,
        "avg_score_p0": total / max(args.games, 1),
        "results": results,
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _safe_load_hu_action_value_model(path: Path) -> object | None:
    try:
        return load_hu_action_value_model(path)
    except Exception:
        return None


def _safe_load_hu_turn3_gate_model(path: Path) -> object | None:
    try:
        return load_hu_turn3_gate_model(path)
    except Exception:
        return None


if __name__ == "__main__":
    main()
