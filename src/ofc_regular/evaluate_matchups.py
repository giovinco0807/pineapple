"""Seat-swapped matchup evaluation and hand tracing for regular OFC AI."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .ai_profiles import (
    DEFAULT_OLD_OPENING_MODEL,
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    DEFAULT_HU_TURN3_CANDIDATE_MODEL,
    DEFAULT_HU_TURN3_STAGE7_MODEL,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
    required_profiles,
)
from .cards import create_deck
from .evaluator import BoardScore, score_board
from .play_ai import _prediction_thread_context
from .rules import check_fl_entry
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score

PROFILE_CHOICES = (
    "current",
    "stage3_baseline",
    "stage7_off",
    "stage7_m5_r10",
    "stage7_m4_r10",
    "stage7_m3_r10_experiment",
    "stage7_m3_r12_experiment",
    "old_opening",
    "random_exact_final",
    "late_t2t3",
    "hu_t3_candidate",
)


def board_to_json(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def board_score_to_json(score: BoardScore) -> dict[str, Any]:
    return {
        "busted": score.busted,
        "top_royalty": score.top_royalty,
        "middle_royalty": score.middle_royalty,
        "bottom_royalty": score.bottom_royalty,
        "total_royalty": score.total_royalty,
        "fl_entry": {
            "qualifies": score.fl_entry.qualifies,
            "card_count": score.fl_entry.card_count,
            "entry_type": score.fl_entry.entry_type,
        },
    }


def classify_board(board: Board) -> str:
    """Classify a completed board for FL-aware hand review."""
    score = score_board(board.top, board.middle, board.bottom)
    top_fl = check_fl_entry(board.top)
    if score.busted:
        return "FL狙いバースト" if top_fl.qualifies else "通常バースト"
    return "FL成功" if score.fl_entry.qualifies else "通常完成"


def trace_hand(
    *,
    seed: int,
    profile_p0: str,
    profile_p1: str,
    policy_p0: Any,
    policy_p1: Any,
) -> dict[str, Any]:
    rng = random.Random(seed)
    deck = create_deck(shuffle=True, rng=rng)
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    policies = [policy_p0, policy_p1]
    profiles = [profile_p0, profile_p1]
    turns: list[dict[str, Any]] = []
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
        turns.append(_turn_record("T0", player, profiles[player], dealt, action, boards[player]))

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
            turns.append(
                _turn_record(
                    f"T{round_index}",
                    player,
                    profiles[player],
                    dealt,
                    action,
                    boards[player],
                )
            )

    score_p0, board_score_p0 = terminal_score(boards[0], boards[1], fl_ev=DEFAULT_FL_EV)
    _reverse_score, board_score_p1 = terminal_score(boards[1], boards[0], fl_ev=DEFAULT_FL_EV)
    return {
        "seed": seed,
        "profiles": {"p0": profile_p0, "p1": profile_p1},
        "score_p0": score_p0,
        "final": {"p0": board_to_json(boards[0]), "p1": board_to_json(boards[1])},
        "board_scores": {
            "p0": board_score_to_json(board_score_p0),
            "p1": board_score_to_json(board_score_p1),
        },
        "classifications": {
            "p0": classify_board(boards[0]),
            "p1": classify_board(boards[1]),
        },
        "turns": turns,
    }


def _turn_record(turn: str, player: int, profile: str, dealt: Iterable[str], action: Any, board: Board) -> dict[str, Any]:
    return {
        "turn": turn,
        "player": player,
        "profile": profile,
        "dealt": list(dealt),
        "placements": [list(placement) for placement in action.placements],
        "discards": list(action.discards),
        "board": board_to_json(board),
    }


def summarize_scores(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {
            "avg_score_per_hand_for_a": 0.0,
            "std_error": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    average = sum(scores) / len(scores)
    variance = sum((score - average) ** 2 for score in scores) / max(len(scores) - 1, 1)
    stderr = math.sqrt(variance / len(scores))
    return {
        "avg_score_per_hand_for_a": average,
        "std_error": stderr,
        "ci95_low": average - 1.96 * stderr,
        "ci95_high": average + 1.96 * stderr,
    }


def evaluate_matchup(
    *,
    profile_a: str,
    profile_b: str,
    games: int,
    seed: int,
    bundle: Any,
    opening_lookahead_samples: int,
    trace_output: Path | None = None,
    trace_limit: int = 0,
    progress_every: int = 0,
) -> dict[str, Any]:
    if games <= 0:
        raise ValueError("games must be positive")
    started_at = time.time()
    paired_scores: list[float] = []
    wins = losses = ties = 0
    profile_classes: dict[str, Counter[str]] = defaultdict(Counter)
    trace_count = 0

    trace_context = (
        trace_output.open("w", encoding="utf-8") if trace_output is not None else contextlib.nullcontext(None)
    )
    with trace_context as trace_handle:
        for index in range(games):
            hand_seed = seed + index
            hand_ab = trace_hand(
                seed=hand_seed,
                profile_p0=profile_a,
                profile_p1=profile_b,
                policy_p0=build_policy(
                    profile_a,
                    bundle,
                    seed=hand_seed * 4,
                    seat="first",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
                policy_p1=build_policy(
                    profile_b,
                    bundle,
                    seed=hand_seed * 4 + 1,
                    seat="second",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
            )
            hand_ba = trace_hand(
                seed=hand_seed,
                profile_p0=profile_b,
                profile_p1=profile_a,
                policy_p0=build_policy(
                    profile_b,
                    bundle,
                    seed=hand_seed * 4 + 2,
                    seat="first",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
                policy_p1=build_policy(
                    profile_a,
                    bundle,
                    seed=hand_seed * 4 + 3,
                    seat="second",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
            )
            paired_score = (float(hand_ab["score_p0"]) - float(hand_ba["score_p0"])) / 2.0
            paired_scores.append(paired_score)
            if paired_score > 0:
                wins += 1
            elif paired_score < 0:
                losses += 1
            else:
                ties += 1

            _add_classification_counts(profile_classes, hand_ab)
            _add_classification_counts(profile_classes, hand_ba)
            if trace_handle is not None and (trace_limit <= 0 or trace_count < trace_limit):
                trace_handle.write(json.dumps(hand_ab, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_count += 1
            if trace_handle is not None and (trace_limit <= 0 or trace_count < trace_limit):
                trace_handle.write(json.dumps(hand_ba, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_count += 1

            if progress_every > 0 and (index + 1) % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "paired_seeds": index + 1,
                            "hands": (index + 1) * 2,
                            **summarize_scores(paired_scores),
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    return {
        "profile_a": profile_a,
        "profile_b": profile_b,
        "paired_seeds": games,
        "hands": games * 2,
        "seed": seed,
        **summarize_scores(paired_scores),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "classification_counts": {
            profile: dict(counter) for profile, counter in sorted(profile_classes.items())
        },
        "classification_rates": {
            profile: _classification_rates(counter)
            for profile, counter in sorted(profile_classes.items())
        },
        "trace_output": str(trace_output) if trace_output else None,
        "trace_hands_written": trace_count,
        "elapsed_seconds": time.time() - started_at,
    }


def _add_classification_counts(profile_classes: dict[str, Counter[str]], hand: dict[str, Any]) -> None:
    for player in ("p0", "p1"):
        profile = hand["profiles"][player]
        classification = hand["classifications"][player]
        profile_classes[profile][classification] += 1


def _classification_rates(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-a", choices=PROFILE_CHOICES, default="current")
    parser.add_argument("--profile-b", choices=PROFILE_CHOICES, default="old_opening")
    parser.add_argument("--games", type=int, default=100, help="Paired seat-swap seeds; total hands are 2x games.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--old-opening-model", type=Path, default=DEFAULT_OLD_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-candidate-model", type=Path, default=DEFAULT_HU_TURN3_CANDIDATE_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_MODEL)
    parser.add_argument(
        "--hu-turn3-stage7-reference-model",
        type=Path,
        default=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    )
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-limit", type=int, default=0, help="Maximum traced hands; <=0 writes all hands.")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))
    paths = ModelPaths(
        opening=args.opening_model,
        old_opening=args.old_opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
        hu_turn3_candidate=args.hu_turn3_candidate_model,
        hu_turn3_stage7=args.hu_turn3_stage7_model,
        hu_turn3_stage7_reference=args.hu_turn3_stage7_reference_model,
    )
    bundle = load_model_bundle(paths, required_profiles(args.profile_a, args.profile_b))
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    with _prediction_thread_context(args.prediction_threads):
        summary = evaluate_matchup(
            profile_a=args.profile_a,
            profile_b=args.profile_b,
            games=args.games,
            seed=args.seed,
            bundle=bundle,
            opening_lookahead_samples=args.opening_lookahead_samples,
            trace_output=args.trace_output,
            trace_limit=args.trace_limit,
            progress_every=args.progress_every,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
