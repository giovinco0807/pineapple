"""Rank-aware FL14 features for cross-row allocation decisions.

The original FL14 actor block deliberately freezes the two tiebreak columns of
an incomplete row at zero.  Completion histograms recover category odds, but
they do not say that a current pair is aces rather than deuces, or that the
kicker competing for middle is a queen rather than a seven.  Those are exactly
the distinctions that dominate the lap-three T2 regret tail.

This module is a suffix, not a change to ``actor_block``.  The old 104 columns
therefore remain byte-for-byte stable for existing models; v2 models append the
six placed-card columns here and use the unique width 110.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from ai.engine.game_engine import evaluate_hand, is_joker
from ai.tutor.t4_first_features import ROW_CAPACITY, _spread, partial_category

RANK_VALUE = {rank: value for value, rank in enumerate("23456789TJQKA", 2)}
ALLOCATION_RANK_SIZE = 6


def _natural_rank(card: str) -> int:
    return RANK_VALUE[card[0]]


def partial_tiebreaks(cards: Sequence[str], capacity: int) -> tuple[float, float]:
    """Two category-aware leading ranks for a placed, possibly-open row.

    Complete rows reuse the evaluator's exact two leading tiebreaks.  For an
    open row, jokers are assigned to the largest natural rank group (highest
    rank breaks a tie), then the ranks are ordered the same way the row's made
    multiplicity category is ordered: pair rank before kicker, high pair before
    low pair, and trips before the remaining kicker.  A joker-only row uses ace
    as the deterministic best current rank; the existing joker-count columns
    retain the uncertainty that this summary intentionally omits.
    """
    if not cards:
        return 0.0, 0.0
    if len(cards) == capacity:
        spread = _spread(evaluate_hand(list(cards), capacity), partial_category(cards, capacity))
        return float(spread[-2]), float(spread[-1])

    counts = Counter(_natural_rank(card) for card in cards if not is_joker(card))
    jokers = sum(1 for card in cards if is_joker(card))
    if counts:
        anchor = max(counts, key=lambda rank: (counts[rank], rank))
        counts[anchor] += jokers
    elif jokers:
        counts[14] = jokers

    category = partial_category(cards, capacity)
    ranks = sorted(counts, reverse=True)

    def ranks_with_at_least(copies: int) -> list[int]:
        return sorted(
            (rank for rank, count in counts.items() if count >= copies),
            reverse=True,
        )

    first = second = 0
    if category == 0:  # high card
        first, *rest = ranks or [0]
        second = rest[0] if rest else 0
    elif category == 1:  # one pair
        pairs = ranks_with_at_least(2)
        first = pairs[0] if pairs else (ranks[0] if ranks else 0)
        second = next((rank for rank in ranks if rank != first), 0)
    elif category == 2:  # two pair
        pairs = ranks_with_at_least(2)
        first = pairs[0] if pairs else 0
        second = pairs[1] if len(pairs) > 1 else 0
    elif category == 3:  # trips
        trips = ranks_with_at_least(3)
        first = trips[0] if trips else (ranks[0] if ranks else 0)
        second = next((rank for rank in ranks if rank != first), 0)
    elif category == 6:  # full house
        trips = ranks_with_at_least(3)
        pairs = ranks_with_at_least(2)
        first = trips[0] if trips else (ranks[0] if ranks else 0)
        second = next((rank for rank in pairs if rank != first), 0)
    elif category == 7:  # quads
        quads = ranks_with_at_least(4)
        first = quads[0] if quads else (ranks[0] if ranks else 0)
        second = next((rank for rank in ranks if rank != first), 0)
    else:
        # Straight/flush categories only occur when complete and returned above.
        first, *rest = ranks or [0]
        second = rest[0] if rest else 0
    return first / 14.0, second / 14.0


def allocation_rank_block(rows: Sequence[Sequence[str]]) -> list[float]:
    """The two leading placed-card ranks for top, middle, and bottom."""
    if len(rows) != 3:
        raise ValueError("an OFC board must have exactly three rows")
    tiebreaks = [
        partial_tiebreaks(rows[index], ROW_CAPACITY[index]) for index in range(3)
    ]
    out = [value for pair in tiebreaks for value in pair]
    assert len(out) == ALLOCATION_RANK_SIZE
    return out


__all__ = [
    "ALLOCATION_RANK_SIZE",
    "allocation_rank_block",
    "partial_tiebreaks",
]
