"""Estimate HU 14-card Fantasyland EV by direct FL-vs-normal simulation.

This intentionally avoids the shortcut formula

    (FL royalty - opponent_avg_royalty + line_scoop_advantage) / (1 - stay_rate)

and instead plays a hidden-FL hand against the current baseline policy stack.
The fixed point is over the value of another 14-card FL hand:

    V_next = E[terminal score with stay bonus V_current]

For a player already in Fantasyland, the next-FL bonus is awarded only by the
regular FL-stay rule, not by ordinary QQ+ entry on top.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .ai_profiles import DEFAULT_OPENING_MODEL, DEFAULT_TURN1_MODEL, DEFAULT_TURN2_MODEL, DEFAULT_TURN3_MODEL
from .cards import create_deck
from .evaluator import BoardScore, score_board
from .fantasyland import Placement, solve_fantasyland
from .hu_turn3_model import load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy
from .rules import check_fl_stay
from .state import Board
from .teacher import DEFAULT_FL_EV
from .turn3_model import load_action_value_model


DEFAULT_STAGE7_MODEL = Path("models/hu_turn3_stage7_reference_override_cached_rank_wide.pt")
DEFAULT_STAGE3_REFERENCE = Path("models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/fl_ev_direct_hu")


@dataclass(frozen=True)
class DirectFlTrial:
    score: float
    hero_royalty: int
    opponent_royalty: int
    hero_stay: bool
    opponent_fl_entry: bool
    opponent_busted: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026062201)
    parser.add_argument("--initial-ev", type=float, default=DEFAULT_FL_EV[14])
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-reference-model", type=Path, default=DEFAULT_STAGE3_REFERENCE)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument(
        "--baseline-profile",
        choices=("models", "random"),
        default="models",
        help=(
            "Opponent policy stack. 'models' loads the baseline model files. "
            "'random' runs without models (pipeline smoke only; numbers are "
            "not valid for FL EV calibration)."
        ),
    )
    parser.add_argument(
        "--hidden-fl-opponent-board",
        choices=("none", "empty"),
        default="none",
        help=(
            "Opponent policy view of the FL player's hidden board. 'none' disables HU "
            "opponent-board features; 'empty' passes an empty public board."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-trials", action="store_true")
    return parser.parse_args()


def make_baseline_policy(args: argparse.Namespace, *, seed: int) -> RegularAiPolicy:
    if args.baseline_profile == "random":
        return RegularAiPolicy(
            seat="second",
            seed=seed,
            opening_lookahead_samples=args.opening_lookahead_samples,
            fl_ev={14: 0.0},
        )
    return RegularAiPolicy(
        opening_model=load_action_value_model(args.opening_model),
        turn1_model=load_action_value_model(args.turn1_model),
        turn2_model=load_action_value_model(args.turn2_model),
        turn3_model=load_action_value_model(args.turn3_model),
        hu_turn3_model=load_hu_action_value_model(args.hu_turn3_stage7_model),
        hu_turn3_reference_model=load_hu_action_value_model(args.hu_turn3_reference_model),
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        seat="second",
        seed=seed,
        opening_lookahead_samples=args.opening_lookahead_samples,
        fl_ev={14: 0.0},
    )


def board_from_fl_placement(placement: Placement) -> Board:
    return Board.from_rows(placement.top, placement.middle, placement.bottom)


def play_normal_hidden_fl_opponent(
    *,
    deck: list[str],
    cursor: int,
    policy: RegularAiPolicy,
    seed: int,
    hidden_fl_opponent_board: str,
) -> Board:
    board = Board.from_rows()
    own_discards: list[str] = []
    public_fl_board = Board.from_rows() if hidden_fl_opponent_board == "empty" else None

    dealt = tuple(deck[cursor : cursor + 5])
    cursor += 5
    action = policy.choose_action(
        board,
        dealt,
        dead_cards=own_discards,
        opponent_board=public_fl_board,
        hand_id=seed,
        game_id=seed,
        decision_seed=seed,
        street="T0_vs_fl",
    )
    board = board.place(action.placements)
    own_discards.extend(action.discards)

    for turn in range(1, 5):
        dealt = tuple(deck[cursor : cursor + 3])
        cursor += 3
        action = policy.choose_action(
            board,
            dealt,
            dead_cards=own_discards,
            opponent_board=public_fl_board,
            hand_id=seed,
            game_id=seed,
            decision_seed=seed + turn,
            street=f"T{turn}_vs_fl",
        )
        board = board.place(action.placements)
        own_discards.extend(action.discards)

    return board


def score_fl_vs_normal(hero: Board, opponent: Board, *, next_fl_ev: float) -> tuple[float, BoardScore, BoardScore, bool]:
    """Score a current-FL hand against a normal opponent hand.

    Hero's next-FL value is based on FL-stay, not ordinary QQ+ entry. Opponent's
    next-FL value is based on normal FL entry.
    """
    own = score_board(hero.top, hero.middle, hero.bottom)
    opp = score_board(opponent.top, opponent.middle, opponent.bottom)
    own_royalty = 0 if own.busted else own.total_royalty
    opp_royalty = 0 if opp.busted else opp.total_royalty
    hero_stay = False if own.busted else check_fl_stay(hero.top, hero.bottom).qualifies
    own_fl = float(next_fl_ev) if hero_stay else 0.0
    opp_fl = 0.0 if opp.busted else float(next_fl_ev if opp.fl_entry.qualifies else 0.0)

    if own.busted and opp.busted:
        return 0.0, own, opp, hero_stay
    if own.busted:
        return float(-6 - opp_royalty - opp_fl), own, opp, hero_stay
    if opp.busted:
        return float(6 + own_royalty + own_fl), own, opp, hero_stay

    line_total = 0
    for own_value, opp_value in (
        (own.top_value, opp.top_value),
        (own.middle_value, opp.middle_value),
        (own.bottom_value, opp.bottom_value),
    ):
        if own_value > opp_value:
            line_total += 1
        elif own_value < opp_value:
            line_total -= 1

    scoop_bonus = 3 if abs(line_total) == 3 else 0
    score = float(line_total)
    score += scoop_bonus if line_total > 0 else (-scoop_bonus if line_total < 0 else 0)
    score += own_royalty - opp_royalty
    score += own_fl - opp_fl
    return float(score), own, opp, hero_stay


def run_iteration(
    *,
    trials: int,
    seed: int,
    current_ev: float,
    args: argparse.Namespace,
    policy: RegularAiPolicy,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rng = random.Random(seed)
    scores: list[float] = []
    hero_royalties: list[float] = []
    opponent_royalties: list[float] = []
    stays = 0
    opponent_entries = 0
    opponent_busts = 0
    solved = 0
    trial_rows: list[dict[str, object]] = []

    for trial_index in range(trials):
        deck = create_deck(shuffle=True, rng=rng)
        hero_cards = tuple(deck[:14])
        placement = solve_fantasyland(hero_cards, stay_bonus=current_ev)
        if placement is None:
            continue
        hero_board = board_from_fl_placement(placement)
        opponent_policy = policy
        opponent_board = play_normal_hidden_fl_opponent(
            deck=deck,
            cursor=14,
            policy=opponent_policy,
            seed=seed * 1_000_003 + trial_index,
            hidden_fl_opponent_board=args.hidden_fl_opponent_board,
        )
        score, hero_score, opponent_score, hero_stay = score_fl_vs_normal(
            hero_board,
            opponent_board,
            next_fl_ev=current_ev,
        )
        solved += 1
        scores.append(score)
        hero_royalties.append(0.0 if hero_score.busted else float(hero_score.total_royalty))
        opponent_royalties.append(0.0 if opponent_score.busted else float(opponent_score.total_royalty))
        stays += int(hero_stay)
        opponent_entries += int((not opponent_score.busted) and opponent_score.fl_entry.qualifies)
        opponent_busts += int(opponent_score.busted)
        if args.write_trials:
            trial_rows.append(
                {
                    "trial_index": trial_index,
                    "score": score,
                    "hero_royalty": 0 if hero_score.busted else hero_score.total_royalty,
                    "opponent_royalty": 0 if opponent_score.busted else opponent_score.total_royalty,
                    "hero_stay": hero_stay,
                    "opponent_fl_entry": (not opponent_score.busted) and opponent_score.fl_entry.qualifies,
                    "opponent_busted": opponent_score.busted,
                    "hero_top": " ".join(hero_board.top),
                    "hero_middle": " ".join(hero_board.middle),
                    "hero_bottom": " ".join(hero_board.bottom),
                    "opponent_top": " ".join(opponent_board.top),
                    "opponent_middle": " ".join(opponent_board.middle),
                    "opponent_bottom": " ".join(opponent_board.bottom),
                }
            )

    mean = float(np.mean(scores)) if scores else 0.0
    stderr = float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
    summary = {
        "seed": float(seed),
        "current_ev": float(current_ev),
        "next_ev": mean,
        "std_error": stderr,
        "ci95_low": mean - 1.96 * stderr,
        "ci95_high": mean + 1.96 * stderr,
        "trials": float(trials),
        "solved": float(solved),
        "hero_avg_royalty": float(np.mean(hero_royalties)) if hero_royalties else 0.0,
        "opponent_avg_royalty": float(np.mean(opponent_royalties)) if opponent_royalties else 0.0,
        "hero_stay_rate": stays / max(solved, 1),
        "opponent_fl_entry_rate": opponent_entries / max(solved, 1),
        "opponent_bust_rate": opponent_busts / max(solved, 1),
    }
    return summary, trial_rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    with _prediction_thread_context(args.prediction_threads):
        policy = make_baseline_policy(args, seed=args.seed + 17)
        current_ev = float(args.initial_ev)
        iteration_rows: list[dict[str, float]] = []
        all_trial_rows: list[dict[str, object]] = []
        for iteration in range(1, args.iterations + 1):
            summary, trial_rows = run_iteration(
                trials=args.trials,
                seed=args.seed + iteration * 10_000,
                current_ev=current_ev,
                args=args,
                policy=policy,
            )
            summary["iteration"] = float(iteration)
            summary["abs_delta"] = abs(summary["next_ev"] - current_ev)
            iteration_rows.append(summary)
            if args.write_trials:
                for row in trial_rows:
                    row["iteration"] = iteration
                    row["current_ev"] = current_ev
                all_trial_rows.extend(trial_rows)
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "current_ev": round(current_ev, 6),
                        "next_ev": round(summary["next_ev"], 6),
                        "stderr": round(summary["std_error"], 6),
                        "hero_stay_rate": round(summary["hero_stay_rate"], 6),
                        "opponent_fl_entry_rate": round(summary["opponent_fl_entry_rate"], 6),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            next_ev = float(summary["next_ev"])
            if abs(next_ev - current_ev) <= args.tolerance:
                current_ev = next_ev
                break
            current_ev = next_ev

    elapsed = time.perf_counter() - started_at
    final = iteration_rows[-1]
    summary_payload = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "trials_per_iteration": args.trials,
        "iterations_requested": args.iterations,
        "iterations_completed": len(iteration_rows),
        "final_fl_ev_14": final["next_ev"],
        "final_std_error": final["std_error"],
        "final_ci95_low": final["ci95_low"],
        "final_ci95_high": final["ci95_high"],
        "hidden_fl_opponent_board": args.hidden_fl_opponent_board,
        "baseline_profile": args.baseline_profile,
        "model_paths": {
            "opening": str(args.opening_model),
            "turn1": str(args.turn1_model),
            "turn2": str(args.turn2_model),
            "turn3": str(args.turn3_model),
            "hu_turn3_stage7": str(args.hu_turn3_stage7_model),
            "hu_turn3_reference": str(args.hu_turn3_reference_model),
        },
        "notes": [
            "Hero current-FL next bonus uses FL-stay only, not QQ+ entry.",
            "Opponent normal-hand FL entry subtracts the same current fixed-point value.",
            "This is a direct HU simulation estimate, not the shortcut formula.",
        ],
    }

    (args.output_dir / "fl_ev_direct_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "fl_ev_direct_iterations.csv", iteration_rows)
    if args.write_trials:
        write_csv(args.output_dir / "fl_ev_direct_trials.csv", all_trial_rows)

    print(json.dumps(summary_payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
