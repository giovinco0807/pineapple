"""Exact late-turn teacher utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from .action_space import Action, generate_turn_actions
from .cards import ALL_CARDS, validate_cards
from .evaluator import BoardScore, score_board
from .state import Board

DEFAULT_FL_EV = {14: 12.196164}


@dataclass(frozen=True)
class EvaluatedAction:
    action: Action
    board: Board
    score: float
    board_score: BoardScore


@dataclass(frozen=True)
class ExpectedAction:
    action: Action
    board: Board
    score: float
    future_count: int
    non_bust_future_count: int = 0


def load_fl_ev(path: str | Path | None = None) -> dict[int, float]:
    if path is None:
        return dict(DEFAULT_FL_EV)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("fl_ev", {})
    return {int(k): float(v) for k, v in raw.items()}


def terminal_score(
    board: Board,
    opponent_board: Board | None = None,
    fl_ev: dict[int, float] | None = None,
) -> tuple[float, BoardScore]:
    """Score a complete board, optionally against a complete opponent board."""
    fl_ev = fl_ev or DEFAULT_FL_EV
    own = score_board(board.top, board.middle, board.bottom)
    if opponent_board is None:
        score = _standalone_terminal_score(own, fl_ev)
        return score, own

    if set(board.all_cards()) & set(opponent_board.all_cards()):
        raise ValueError("hero and opponent boards overlap")
    opp = score_board(opponent_board.top, opponent_board.middle, opponent_board.bottom)
    score = _heads_up_terminal_score(own, opp, fl_ev)
    return score, own


def evaluate_turn_actions(
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board | None = None,
    fl_ev: dict[int, float] | None = None,
) -> list[EvaluatedAction]:
    """Evaluate every legal normal-turn action exactly."""
    fl_ev = fl_ev or DEFAULT_FL_EV
    evaluated: list[EvaluatedAction] = []
    for action in generate_turn_actions(board, dealt_cards):
        next_board = board.place(action.placements)
        if not next_board.is_complete():
            continue
        score, board_score = terminal_score(next_board, opponent_board, fl_ev)
        evaluated.append(EvaluatedAction(action, next_board, score, board_score))
    evaluated.sort(key=lambda item: item.score, reverse=True)
    return evaluated


def evaluate_two_turn_actions(
    board: Board,
    dealt_cards: Iterable[str],
    *,
    opponent_board: Board | None = None,
    dead_cards: Iterable[str] = (),
    fl_ev: dict[int, float] | None = None,
    future_deals: Iterable[Sequence[str]] | None = None,
    max_future_deals: int | None = None,
    seed: int = 42,
) -> list[ExpectedAction]:
    """Evaluate current actions by expectimax over the next 3-card turn.

    This is intended for 9-card boards: place two cards now, enumerate or sample
    the final 3-card deal, and solve the final turn exactly.
    """
    fl_ev = fl_ev or DEFAULT_FL_EV
    dealt = tuple(dealt_cards)
    if board.card_count() != 9:
        raise ValueError("two-turn evaluation requires a 9-card board")
    if len(dealt) != 3:
        raise ValueError("two-turn evaluation requires exactly three dealt cards")

    current_actions = generate_turn_actions(board, dealt)
    futures = _select_future_deals(
        board=board,
        dealt_cards=dealt,
        opponent_board=opponent_board,
        dead_cards=dead_cards,
        future_deals=future_deals,
        max_future_deals=max_future_deals,
        seed=seed,
    )
    if not futures:
        raise ValueError("no future deals available")

    evaluated: list[ExpectedAction] = []
    for action in current_actions:
        next_board = board.place(action.placements)
        if next_board.is_complete():
            score, board_score = terminal_score(next_board, opponent_board, fl_ev)
            evaluated.append(
                ExpectedAction(
                    action=action,
                    board=next_board,
                    score=score,
                    future_count=1,
                    non_bust_future_count=0 if board_score.busted else 1,
                )
            )
            continue

        score_sum = 0.0
        future_count = 0
        non_bust_future_count = 0
        for future in futures:
            ranked = evaluate_turn_actions(
                next_board,
                future,
                opponent_board=opponent_board,
                fl_ev=fl_ev,
            )
            if not ranked:
                continue
            score_sum += ranked[0].score
            future_count += 1
            if any(not item.board_score.busted for item in ranked):
                non_bust_future_count += 1

        if future_count:
            evaluated.append(
                ExpectedAction(
                    action=action,
                    board=next_board,
                    score=score_sum / future_count,
                    future_count=future_count,
                    non_bust_future_count=non_bust_future_count,
                )
            )

    evaluated.sort(key=lambda item: item.score, reverse=True)
    return evaluated


def _standalone_terminal_score(board_score: BoardScore, fl_ev: dict[int, float]) -> float:
    if board_score.busted:
        return 0.0
    fl_bonus = fl_ev.get(board_score.fl_entry.card_count, 0.0)
    return float(board_score.total_royalty) + fl_bonus


def _heads_up_terminal_score(own: BoardScore, opp: BoardScore, fl_ev: dict[int, float]) -> float:
    own_royalty = 0 if own.busted else own.total_royalty
    opp_royalty = 0 if opp.busted else opp.total_royalty
    own_fl = 0.0 if own.busted else fl_ev.get(own.fl_entry.card_count, 0.0)
    opp_fl = 0.0 if opp.busted else fl_ev.get(opp.fl_entry.card_count, 0.0)
    if own.busted and opp.busted:
        return 0.0
    if own.busted:
        return float(-6 - opp_royalty - opp_fl)
    if opp.busted:
        return float(6 + own_royalty + own_fl)

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
    score = line_total
    score += scoop_bonus if line_total > 0 else (-scoop_bonus if line_total < 0 else 0)
    score += own_royalty - opp_royalty
    score += own_fl
    score -= opp_fl
    return float(score)


def _future_deals(
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board | None,
    dead_cards: Iterable[str],
) -> Iterable[tuple[str, str, str]]:
    dealt = tuple(dealt_cards)
    dead = tuple(dead_cards)
    used_cards = [*board.all_cards(), *dealt, *dead]
    if opponent_board is not None:
        used_cards.extend(opponent_board.all_cards())
    validate_cards(used_cards)
    used = set(used_cards)
    deck = tuple(card for card in ALL_CARDS if card not in used)
    return combinations(deck, 3)


def _select_future_deals(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board | None,
    dead_cards: Iterable[str],
    future_deals: Iterable[Sequence[str]] | None,
    max_future_deals: int | None,
    seed: int,
) -> tuple[Sequence[str], ...]:
    if future_deals is not None:
        return tuple(future_deals)
    all_futures = tuple(_future_deals(board, dealt_cards, opponent_board, dead_cards))
    if max_future_deals is None or len(all_futures) <= max_future_deals:
        return all_futures
    if max_future_deals <= 0:
        raise ValueError("max_future_deals must be positive")
    rng = random.Random(seed)
    return tuple(rng.sample(all_futures, max_future_deals))
