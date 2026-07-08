"""Trace HU Turn3 overrides and compare them to baseline Turn3 rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .action_space import Action, generate_turn_actions
from .ai_profiles import DEFAULT_OPENING_MODEL, DEFAULT_TURN1_MODEL, DEFAULT_TURN2_MODEL, DEFAULT_TURN3_MODEL
from .cards import create_deck
from .evaluator import score_board
from .hu_turn3_model import hu_policy_sample, load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy, action_to_json, board_to_json, policy_sample
from .rules import check_fl_entry
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score
from .visibility import HuDiscardTracker
from .turn3_model import load_action_value_model


@dataclass
class HandTrace:
    score_p0: float
    boards: list[Board]
    decisions: list[dict[str, Any]]
    candidate_t3_decision_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-model", type=Path, required=True)
    parser.add_argument("--hu-turn3-min-margin", type=float, default=0.0)
    parser.add_argument("--hu-turn3-max-self-regret", type=float)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--record-all", action="store_true")
    parser.add_argument("--decision-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_parts(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "opening_model": load_action_value_model(args.opening_model),
        "turn1_model": load_action_value_model(args.turn1_model),
        "turn2_model": load_action_value_model(args.turn2_model),
        "turn3_model": load_action_value_model(args.turn3_model),
        "opening_lookahead_samples": args.opening_lookahead_samples,
    }


def make_policy(
    parts: dict[str, Any],
    *,
    seed: int,
    seat: str,
    hu_turn3_model: Any | None = None,
    hu_turn3_min_margin: float = 0.0,
    hu_turn3_max_self_regret: float | None = None,
) -> RegularAiPolicy:
    return RegularAiPolicy(
        seed=seed,
        seat=seat,
        hu_turn3_model=hu_turn3_model,
        hu_turn3_min_margin=hu_turn3_min_margin,
        hu_turn3_max_self_regret=hu_turn3_max_self_regret,
        **parts,
    )


def inspect_hu_turn3_decision(
    *,
    policy: RegularAiPolicy,
    board: Board,
    dealt: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str],
) -> tuple[Action, dict[str, Any]]:
    dealt_tuple = tuple(dealt)
    dead_tuple = tuple(dead_cards)
    actions = generate_turn_actions(board, dealt_tuple)
    if not actions:
        raise ValueError("no legal Turn3 actions")
    if policy.hu_turn3_model is None or policy.turn3_model is None:
        action = policy.choose_action(
            board,
            dealt_tuple,
            dead_cards=dead_tuple,
            opponent_board=opponent_board,
        )
        return action, {
            "override": False,
            "chosen_source": "policy",
            "actions_count": len(actions),
        }

    to_act_order = "second" if opponent_board.card_count() > board.card_count() else "first"
    hu_sample = hu_policy_sample(
        board,
        dealt_tuple,
        actions,
        opponent_board=opponent_board,
        dead_cards=dead_tuple,
        seat=policy.seat,
        to_act_order=to_act_order,
    )
    hu_predictions = policy.hu_turn3_model.predict_sample(hu_sample)
    hu_index = int(hu_predictions.argmax())
    baseline_sample = policy_sample(board, dealt_tuple, actions)
    baseline_index = int(policy.turn3_model.choose_action_index(baseline_sample))
    self_predictions = policy.turn3_model.predict_sample(baseline_sample)
    predicted_margin = float(hu_predictions[hu_index] - hu_predictions[baseline_index])
    self_regret = float(self_predictions[baseline_index] - self_predictions[hu_index])
    if (
        predicted_margin >= policy.hu_turn3_min_margin
        and (
            policy.hu_turn3_max_self_regret is None
            or self_regret <= policy.hu_turn3_max_self_regret
        )
    ):
        chosen_index = hu_index
        chosen_source = "hu"
    elif predicted_margin < policy.hu_turn3_min_margin:
        chosen_index = baseline_index
        chosen_source = "baseline_margin_gate"
    else:
        chosen_index = baseline_index
        chosen_source = "baseline_self_regret_gate"

    decision = {
        "override": chosen_index != baseline_index,
        "chosen_source": chosen_source,
        "seat": policy.seat,
        "to_act_order": to_act_order,
        "actions_count": len(actions),
        "hu_index": hu_index,
        "baseline_index": baseline_index,
        "chosen_index": chosen_index,
        "hu_turn3_min_margin": policy.hu_turn3_min_margin,
        "hu_turn3_max_self_regret": policy.hu_turn3_max_self_regret,
        "predicted_margin": predicted_margin,
        "self_regret_vs_baseline": self_regret,
        "predicted_hu_score": float(hu_predictions[hu_index]),
        "predicted_baseline_score": float(hu_predictions[baseline_index]),
        "self_model_hu_score": float(self_predictions[hu_index]),
        "self_model_baseline_score": float(self_predictions[baseline_index]),
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent_board),
        "dealt": list(dealt_tuple),
        "dead_cards": list(dead_tuple),
        "dead_count": len(dead_tuple),
        "top_state": top_state(board.top),
        "opponent_top_state": top_state(opponent_board.top),
        "chosen_action": action_to_json(board, actions[chosen_index]),
        "hu_action": action_to_json(board, actions[hu_index]),
        "baseline_action": action_to_json(board, actions[baseline_index]),
    }
    return actions[chosen_index], decision


def top_state(cards: Iterable[str]) -> dict[str, Any]:
    cards_tuple = tuple(cards)
    counts: dict[str, int] = {}
    for card in cards_tuple:
        counts[card[0]] = counts.get(card[0], 0) + 1
    high_pair = next((rank for rank in ("A", "K", "Q") if counts.get(rank, 0) >= 2), None)
    return {
        "cards": list(cards_tuple),
        "count": len(cards_tuple),
        "open_slots": max(0, 3 - len(cards_tuple)),
        "high_pair": high_pair,
        "pair_ranks": [rank for rank, count in sorted(counts.items()) if count >= 2],
        "fl_complete": check_fl_entry(cards_tuple).qualifies if len(cards_tuple) == 3 else False,
    }


def play_traced_hand(
    *,
    seed: int,
    candidate_player: int,
    profile_names: list[str],
    policies: list[RegularAiPolicy],
    record_all: bool,
) -> HandTrace:
    deck = create_deck(shuffle=True, rng=__import__("random").Random(seed))
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    discards = HuDiscardTracker()
    decisions: list[dict[str, Any]] = []
    candidate_t3_decision_count = 0

    for player in (0, 1):
        dealt = tuple(deck[cursor : cursor + 5])
        cursor += 5
        action = policies[player].choose_action(
            boards[player],
            dealt,
            dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
            opponent_board=boards[1 - player],
        )
        boards[player] = boards[player].place(action.placements)
        discards.record(player, action.discards)

    for round_index in range(1, 5):
        for player in (0, 1):
            dealt = tuple(deck[cursor : cursor + 3])
            cursor += 3
            if player == candidate_player and boards[player].card_count() == 9:
                candidate_t3_decision_count += 1
                action, decision = inspect_hu_turn3_decision(
                    policy=policies[player],
                    board=boards[player],
                    dealt=dealt,
                    opponent_board=boards[1 - player],
                    dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                )
                chosen_score = rollout_from_t3_action(
                    boards=boards,
                    deck=deck,
                    cursor=cursor,
                    current_player=player,
                    round_index=round_index,
                    action=action,
                    policies=policies,
                    discards=discards,
                    candidate_player=candidate_player,
                )
                baseline_action = _action_from_json(decision["baseline_action"])
                baseline_score = rollout_from_t3_action(
                    boards=boards,
                    deck=deck,
                    cursor=cursor,
                    current_player=player,
                    round_index=round_index,
                    action=baseline_action,
                    policies=policies,
                    discards=discards,
                    candidate_player=candidate_player,
                )
                decision.update(
                    {
                        "seed": seed,
                        "turn": f"T{round_index}",
                        "player": player,
                        "profile": profile_names[player],
                        "counterfactual_chosen_score": chosen_score,
                        "counterfactual_baseline_score": baseline_score,
                        "counterfactual_delta_vs_baseline": chosen_score - baseline_score,
                    }
                )
                if record_all or decision["override"]:
                    decisions.append(decision)
            else:
                action = policies[player].choose_action(
                    boards[player],
                    dealt,
                    dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                    opponent_board=boards[1 - player],
                )
            boards[player] = boards[player].place(action.placements)
            discards.record(player, action.discards)

    score_p0, _score = terminal_score(boards[0], boards[1], fl_ev=DEFAULT_FL_EV)
    candidate_score = score_p0 if candidate_player == 0 else -score_p0
    opponent_player = 1 - candidate_player
    for decision in decisions:
        decision.update(
            {
                "candidate_terminal_score": candidate_score,
                "candidate_final": board_to_json(boards[candidate_player]),
                "opponent_final": board_to_json(boards[opponent_player]),
                "candidate_final_score": board_score_json(boards[candidate_player]),
                "opponent_final_score": board_score_json(boards[opponent_player]),
            }
        )
    return HandTrace(
        score_p0=score_p0,
        boards=boards,
        decisions=decisions,
        candidate_t3_decision_count=candidate_t3_decision_count,
    )


def _action_from_json(payload: dict[str, Any]) -> Action:
    return Action(
        placements=tuple((str(card), str(row)) for card, row in payload.get("placements", ())),
        discards=tuple(str(card) for card in payload.get("discards", ())),
    )


def rollout_from_t3_action(
    *,
    boards: list[Board],
    deck: list[str],
    cursor: int,
    current_player: int,
    round_index: int,
    action: Action,
    policies: list[RegularAiPolicy],
    discards: HuDiscardTracker,
    candidate_player: int,
) -> float:
    local_boards = list(boards)
    local_discards = discards.clone()
    local_cursor = cursor
    local_boards[current_player] = local_boards[current_player].place(action.placements)
    local_discards.record(current_player, action.discards)

    for player in range(current_player + 1, 2):
        local_cursor = play_rollout_turn(
            boards=local_boards,
            deck=deck,
            cursor=local_cursor,
            player=player,
            policies=policies,
            discards=local_discards,
        )
    for next_round in range(round_index + 1, 5):
        for player in (0, 1):
            local_cursor = play_rollout_turn(
                boards=local_boards,
                deck=deck,
                cursor=local_cursor,
                player=player,
                policies=policies,
                discards=local_discards,
            )
    score_p0, _score = terminal_score(local_boards[0], local_boards[1], fl_ev=DEFAULT_FL_EV)
    return score_p0 if candidate_player == 0 else -score_p0


def play_rollout_turn(
    *,
    boards: list[Board],
    deck: list[str],
    cursor: int,
    player: int,
    policies: list[RegularAiPolicy],
    discards: HuDiscardTracker,
) -> int:
    if boards[player].card_count() >= 13:
        return cursor
    dealt = tuple(deck[cursor : cursor + 3])
    next_cursor = cursor + 3
    action = policies[player].choose_action(
        boards[player],
        dealt,
        dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
        opponent_board=boards[1 - player],
    )
    boards[player] = boards[player].place(action.placements)
    discards.record(player, action.discards)
    return next_cursor


def board_score_json(board: Board) -> dict[str, Any]:
    score = score_board(board.top, board.middle, board.bottom)
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


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_avg": 0.0,
            f"{prefix}_ci95_low": 0.0,
            f"{prefix}_ci95_high": 0.0,
        }
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / max(len(values) - 1, 1)
    stderr = math.sqrt(variance / len(values))
    return {
        f"{prefix}_count": float(len(values)),
        f"{prefix}_avg": average,
        f"{prefix}_ci95_low": average - 1.96 * stderr,
        f"{prefix}_ci95_high": average + 1.96 * stderr,
    }


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))

    parts = load_parts(args)
    hu_model = load_hu_action_value_model(args.hu_turn3_model)
    paired_scores: list[float] = []
    override_deltas: list[float] = []
    override_terminal_scores: list[float] = []
    decision_count = 0
    candidate_t3_decision_count = 0
    override_count = 0
    override_delta_wins = 0
    override_delta_losses = 0
    started_at = time.time()

    if args.decision_output:
        args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    output_context = (
        args.decision_output.open("w", encoding="utf-8")
        if args.decision_output is not None
        else __import__("contextlib").nullcontext(None)
    )

    with _prediction_thread_context(args.prediction_threads), output_context as decision_handle:
        for index in range(args.games):
            hand_seed = args.seed + index
            candidate_first = play_traced_hand(
                seed=hand_seed,
                candidate_player=0,
                profile_names=[args.candidate_name, args.baseline_name],
                policies=[
                    make_policy(
                        parts,
                        seed=hand_seed * 4,
                        seat="first",
                        hu_turn3_model=hu_model,
                        hu_turn3_min_margin=args.hu_turn3_min_margin,
                        hu_turn3_max_self_regret=args.hu_turn3_max_self_regret,
                    ),
                    make_policy(parts, seed=hand_seed * 4 + 1, seat="second"),
                ],
                record_all=args.record_all,
            )
            candidate_second = play_traced_hand(
                seed=hand_seed,
                candidate_player=1,
                profile_names=[args.baseline_name, args.candidate_name],
                policies=[
                    make_policy(parts, seed=hand_seed * 4 + 2, seat="first"),
                    make_policy(
                        parts,
                        seed=hand_seed * 4 + 3,
                        seat="second",
                        hu_turn3_model=hu_model,
                        hu_turn3_min_margin=args.hu_turn3_min_margin,
                        hu_turn3_max_self_regret=args.hu_turn3_max_self_regret,
                    ),
                ],
                record_all=args.record_all,
            )
            paired_score = (candidate_first.score_p0 - candidate_second.score_p0) / 2.0
            paired_scores.append(paired_score)
            candidate_t3_decision_count += (
                candidate_first.candidate_t3_decision_count
                + candidate_second.candidate_t3_decision_count
            )
            for hand_label, hand in (("candidate_first", candidate_first), ("candidate_second", candidate_second)):
                for decision in hand.decisions:
                    decision["hand_label"] = hand_label
                    decision["paired_score_for_candidate"] = paired_score
                    if decision_handle is not None:
                        decision_handle.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")
                    decision_count += 1
                    if decision["override"]:
                        override_count += 1
                        delta = float(decision["counterfactual_delta_vs_baseline"])
                        override_deltas.append(delta)
                        override_terminal_scores.append(float(decision["candidate_terminal_score"]))
                        if delta > 0:
                            override_delta_wins += 1
                        elif delta < 0:
                            override_delta_losses += 1
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "paired_seeds": index + 1,
                            **summarize(paired_scores, "paired_score"),
                            "override_count": override_count,
                            **summarize(override_deltas, "override_delta"),
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    summary = {
        "games": args.games,
        "hands": args.games * 2,
        "seed": args.seed,
        "candidate_name": args.candidate_name,
        "baseline_name": args.baseline_name,
        "hu_turn3_model": str(args.hu_turn3_model),
        "hu_turn3_min_margin": args.hu_turn3_min_margin,
        "hu_turn3_max_self_regret": args.hu_turn3_max_self_regret,
        "record_all": args.record_all,
        "decision_output": str(args.decision_output) if args.decision_output else None,
        "decision_records_written": decision_count,
        "candidate_t3_decision_count": candidate_t3_decision_count,
        "override_count": override_count,
        "override_rate": override_count / max(candidate_t3_decision_count, 1),
        "override_delta_wins": override_delta_wins,
        "override_delta_losses": override_delta_losses,
        **summarize(paired_scores, "paired_score"),
        **summarize(override_deltas, "override_delta"),
        **summarize(override_terminal_scores, "override_terminal_score"),
        "elapsed_seconds": time.time() - started_at,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
