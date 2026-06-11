"""Legal action generation for regular OFC turns."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

from .cards import validate_cards
from .state import Board, ROWS


@dataclass(frozen=True)
class Action:
    placements: tuple[tuple[str, str], ...]
    discards: tuple[str, ...] = ()


def generate_actions(board: Board, dealt_cards: Iterable[str]) -> list[Action]:
    """Generate legal actions for the next deal size."""
    dealt = tuple(dealt_cards)
    if board.card_count() == 0 and len(dealt) == 5:
        return generate_initial_actions(board, dealt)
    return generate_turn_actions(board, dealt)


def generate_initial_actions(board: Board, dealt_cards: Iterable[str]) -> list[Action]:
    """Generate opening actions: place all five cards with no discard."""
    board.validate()
    dealt = tuple(dealt_cards)
    validate_cards(dealt)
    if board.card_count() != 0:
        raise ValueError("opening actions require an empty board")
    if len(dealt) != 5:
        raise ValueError("opening actions require exactly five dealt cards")

    actions: list[Action] = []
    for rows in product(ROWS, repeat=5):
        if not _rows_fit(board, rows):
            continue
        placements = tuple(zip(dealt, rows))
        try:
            board.place(placements)
        except ValueError:
            continue
        actions.append(Action(placements=placements))
    return actions


def generate_turn_actions(board: Board, dealt_cards: Iterable[str]) -> list[Action]:
    """Generate normal-turn actions.

    For a 3-card pineapple turn this places two cards and discards one. If the
    board has only one slot left, it places one and discards the rest.
    """
    board.validate()
    dealt = tuple(dealt_cards)
    validate_cards(dealt)
    if set(dealt) & set(board.all_cards()):
        raise ValueError("dealt cards overlap board cards")

    open_total = 13 - board.card_count()
    if open_total <= 0:
        return []
    place_count = min(2, open_total, len(dealt))
    if place_count <= 0:
        return []

    actions: list[Action] = []
    for place_cards in combinations(dealt, place_count):
        discards = tuple(card for card in dealt if card not in place_cards)
        for rows in product(ROWS, repeat=place_count):
            if not _rows_fit(board, rows):
                continue
            placements = tuple(zip(place_cards, rows))
            try:
                board.place(placements)
            except ValueError:
                continue
            actions.append(Action(placements=placements, discards=discards))
    return actions


def _rows_fit(board: Board, rows: tuple[str, ...]) -> bool:
    for row in ROWS:
        if rows.count(row) > board.open_slots(row):
            return False
    return True
