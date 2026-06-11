"""Fantasyland placement solver for regular 14-card no-joker hands."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Sequence

from .cards import validate_cards
from .evaluator import (
    HAND_QUADS,
    HAND_TRIPS,
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)

@dataclass(frozen=True)
class Placement:
    top: tuple[str, ...]
    middle: tuple[str, ...]
    bottom: tuple[str, ...]
    discards: tuple[str, ...]
    top_royalty: int
    middle_royalty: int
    bottom_royalty: int
    total_royalty: int
    can_stay: bool
    score: float


def _mask_for(indices: Sequence[int]) -> int:
    mask = 0
    for idx in indices:
        mask |= 1 << idx
    return mask


@lru_cache(maxsize=200_000)
def _eval3(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    return evaluate_3_card(cards)


@lru_cache(maxsize=400_000)
def _eval5(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    return evaluate_5_card(cards)


def solve_fantasyland(
    cards: Sequence[str],
    *,
    stay_bonus: float = 100.0,
) -> Placement | None:
    """Find the best legal 13-card placement from a regular 14-card FL deal.

    The default objective mirrors the existing implementation style: maximize
    royalty, with a large bonus for FL stay. For calibrated EV work, pass a
    stay bonus matching the current estimated value of another 14-card FL hand.
    """
    if len(cards) != 14:
        raise ValueError(f"regular Fantasyland expects exactly 14 cards, got {len(cards)}")
    validate_cards(cards)
    cards_tuple = tuple(cards)
    all_mask = (1 << len(cards_tuple)) - 1

    combo3: list[tuple[int, tuple[str, ...], tuple[int, tuple[int, ...]], int, bool]] = []
    top_by_remaining: dict[int, list[tuple[int, tuple[str, ...], tuple[int, tuple[int, ...]], int, bool]]] = {}
    combo5: list[tuple[int, tuple[str, ...], tuple[int, tuple[int, ...]], int, int, bool]] = []

    for idxs in combinations(range(14), 3):
        hand = tuple(cards_tuple[i] for i in idxs)
        value = _eval3(tuple(sorted(hand)))
        entry = (_mask_for(idxs), hand, value, get_top_royalty(hand), value[0] == HAND_TRIPS)
        combo3.append(entry)

    for idxs in combinations(range(14), 4):
        remaining_mask = _mask_for(idxs)
        top_by_remaining[remaining_mask] = [
            top_entry for top_entry in combo3 if top_entry[0] & remaining_mask == top_entry[0]
        ]

    for idxs in combinations(range(14), 5):
        hand = tuple(cards_tuple[i] for i in idxs)
        sorted_hand = tuple(sorted(hand))
        value = _eval5(sorted_hand)
        combo5.append((
            _mask_for(idxs),
            hand,
            value,
            get_middle_royalty(hand),
            get_bottom_royalty(hand),
            value[0] >= HAND_QUADS,
        ))

    best: Placement | None = None
    best_score = float("-inf")

    for bottom_mask, bottom, bottom_value, _middle_as_bottom_royalty, bottom_royalty, bottom_stay in combo5:
        remaining_after_bottom = all_mask ^ bottom_mask
        for middle_mask, middle, middle_value, middle_royalty, _bottom_as_middle_royalty, _middle_stay in combo5:
            if middle_mask & bottom_mask:
                continue
            if middle_value > bottom_value:
                continue
            remaining_after_middle = remaining_after_bottom ^ middle_mask
            for top_mask, top, top_value, top_royalty, top_stay in top_by_remaining[remaining_after_middle]:
                if top_value > middle_value:
                    continue

                discard_mask = remaining_after_middle ^ top_mask
                discards = tuple(
                    cards_tuple[i] for i in range(14) if discard_mask & (1 << i)
                )
                total_royalty = top_royalty + middle_royalty + bottom_royalty
                stay = top_stay or bottom_stay
                score = float(total_royalty) + (float(stay_bonus) if stay else 0.0)

                if score > best_score:
                    best_score = score
                    best = Placement(
                        top=tuple(top),
                        middle=tuple(middle),
                        bottom=tuple(bottom),
                        discards=discards,
                        top_royalty=top_royalty,
                        middle_royalty=middle_royalty,
                        bottom_royalty=bottom_royalty,
                        total_royalty=total_royalty,
                        can_stay=stay,
                        score=score,
                    )
    return best
