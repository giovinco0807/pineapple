"""Hand evaluation, royalties, and board scoring for regular OFC Pineapple."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .cards import RANK_VALUE, card_rank, card_suit, validate_cards
from .rules import FantasylandEntry, check_fl_entry

HAND_HIGH = 0
HAND_PAIR = 1
HAND_TWO_PAIR = 2
HAND_TRIPS = 3
HAND_STRAIGHT = 4
HAND_FLUSH = 5
HAND_FULL_HOUSE = 6
HAND_QUADS = 7
HAND_STRAIGHT_FLUSH = 8


@dataclass(frozen=True)
class BoardScore:
    busted: bool
    top_value: tuple[int, tuple[int, ...]]
    middle_value: tuple[int, tuple[int, ...]]
    bottom_value: tuple[int, tuple[int, ...]]
    top_royalty: int
    middle_royalty: int
    bottom_royalty: int
    total_royalty: int
    fl_entry: FantasylandEntry


def _rank_values(cards: Sequence[str]) -> list[int]:
    return sorted((card_rank(card) for card in cards), reverse=True)


def _straight_high(ranks: Iterable[int]) -> int | None:
    unique = set(ranks)
    if {14, 5, 4, 3, 2}.issubset(unique):
        return 5
    for high in range(14, 5, -1):
        if set(range(high - 4, high + 1)).issubset(unique):
            return high
    return None


def evaluate_5_card(cards: Sequence[str]) -> tuple[int, tuple[int, ...]]:
    """Return a comparable 5-card hand value.

    Larger tuples are stronger. The category values match normal poker order,
    with royal flush represented as straight flush with high card Ace.
    """
    if len(cards) != 5:
        raise ValueError(f"5-card evaluation requires 5 cards, got {len(cards)}")
    validate_cards(cards)

    ranks = _rank_values(cards)
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({card_suit(card) for card in cards}) == 1
    straight_high = _straight_high(ranks)

    if flush and straight_high is not None:
        return HAND_STRAIGHT_FLUSH, (straight_high,)

    if groups[0][0] == 4:
        quad_rank = groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad_rank)
        return HAND_QUADS, (quad_rank, kicker)

    if groups[0][0] == 3 and groups[1][0] == 2:
        return HAND_FULL_HOUSE, (groups[0][1], groups[1][1])

    if flush:
        return HAND_FLUSH, tuple(ranks)

    if straight_high is not None:
        return HAND_STRAIGHT, (straight_high,)

    if groups[0][0] == 3:
        trips_rank = groups[0][1]
        kickers = tuple(rank for rank in ranks if rank != trips_rank)
        return HAND_TRIPS, (trips_rank, *kickers)

    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = max(rank for rank in ranks if rank not in pairs)
        return HAND_TWO_PAIR, (pairs[0], pairs[1], kicker)

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = tuple(rank for rank in ranks if rank != pair)
        return HAND_PAIR, (pair, *kickers)

    return HAND_HIGH, tuple(ranks)


def evaluate_3_card(cards: Sequence[str]) -> tuple[int, tuple[int, ...]]:
    """Return a comparable top-row value on the 5-card category scale."""
    if len(cards) != 3:
        raise ValueError(f"3-card evaluation requires 3 cards, got {len(cards)}")
    validate_cards(cards)

    ranks = _rank_values(cards)
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)

    if groups[0][0] == 3:
        return HAND_TRIPS, (groups[0][1],)
    if groups[0][0] == 2:
        pair = groups[0][1]
        kicker = max(rank for rank in ranks if rank != pair)
        return HAND_PAIR, (pair, kicker)
    return HAND_HIGH, tuple(ranks)


def get_top_royalty(cards: Sequence[str]) -> int:
    category, ranks = evaluate_3_card(cards)
    if category == HAND_TRIPS:
        return 10 + (ranks[0] - 2)
    if category == HAND_PAIR and ranks[0] >= RANK_VALUE["6"]:
        return ranks[0] - 5
    return 0


def get_middle_royalty(cards: Sequence[str]) -> int:
    category, ranks = evaluate_5_card(cards)
    if category == HAND_STRAIGHT_FLUSH and ranks[0] == 14:
        return 50
    if category == HAND_STRAIGHT_FLUSH:
        return 30
    if category == HAND_QUADS:
        return 20
    if category == HAND_FULL_HOUSE:
        return 12
    if category == HAND_FLUSH:
        return 8
    if category == HAND_STRAIGHT:
        return 4
    if category == HAND_TRIPS:
        return 2
    return 0


def get_bottom_royalty(cards: Sequence[str]) -> int:
    category, ranks = evaluate_5_card(cards)
    if category == HAND_STRAIGHT_FLUSH and ranks[0] == 14:
        return 25
    if category == HAND_STRAIGHT_FLUSH:
        return 15
    if category == HAND_QUADS:
        return 10
    if category == HAND_FULL_HOUSE:
        return 6
    if category == HAND_FLUSH:
        return 4
    if category == HAND_STRAIGHT:
        return 2
    return 0


def score_board(
    top: Sequence[str],
    middle: Sequence[str],
    bottom: Sequence[str],
) -> BoardScore:
    """Score one complete OFC board."""
    all_cards = [*top, *middle, *bottom]
    validate_cards(all_cards)
    if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
        raise ValueError("board must have 3 top, 5 middle, and 5 bottom cards")

    top_value = evaluate_3_card(top)
    middle_value = evaluate_5_card(middle)
    bottom_value = evaluate_5_card(bottom)
    busted = top_value > middle_value or middle_value > bottom_value

    top_royalty = middle_royalty = bottom_royalty = 0
    fl_entry = FantasylandEntry(False, 0, None)
    if not busted:
        top_royalty = get_top_royalty(top)
        middle_royalty = get_middle_royalty(middle)
        bottom_royalty = get_bottom_royalty(bottom)
        fl_entry = check_fl_entry(top)

    return BoardScore(
        busted=busted,
        top_value=top_value,
        middle_value=middle_value,
        bottom_value=bottom_value,
        top_royalty=top_royalty,
        middle_royalty=middle_royalty,
        bottom_royalty=bottom_royalty,
        total_royalty=top_royalty + middle_royalty + bottom_royalty,
        fl_entry=fl_entry,
    )
