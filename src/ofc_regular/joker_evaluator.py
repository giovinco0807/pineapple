"""Canonical Joker board evaluation ported onto the regular evaluator.

Semantics mirror ``ai/engine/game_engine.py``:

- ``evaluate_hand_joker``: exhaustive distinct natural substitutions, strongest
  value wins.  Substitutions exclude only the natural cards already in the same
  row (row-local exclusion, matching the canonical evaluator).
- ``constrain_row``: strongest substitution whose value does not exceed the row
  below; if the row is already within the bound it is kept unchanged
  (jokers included), and if no substitution fits the row is kept so the final
  ordering check marks the board as busted.
- Royalties and Fantasyland chain entry (QQ=14, KK=15, AA=16, trips=17) are
  derived from the resolved row values.

Hand values are this repository's comparable ``(category, ranks)`` tuples; the
ordering is semantically identical to the main track's base-15 encoding, which
the cross-engine parity test verifies.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Sequence

from .cards import ALL_CARDS as NATURAL_CARDS
from .cards import RANK_VALUE
from .evaluator import (
    HAND_PAIR,
    HAND_STRAIGHT_FLUSH,
    HAND_TRIPS,
    evaluate_3_card,
    evaluate_5_card,
)
from .joker_rules import is_joker

HandValue = tuple[int, tuple[int, ...]]

# Same enumeration order as the canonical evaluator: ranks 2..A, suits s,h,d,c.
_SUB_ORDER: tuple[str, ...] = tuple(
    rank + suit for rank in "23456789TJQKA" for suit in "shdc"
)
assert set(_SUB_ORDER) == set(NATURAL_CARDS)


def _evaluate_natural(cards: Sequence[str], expected_count: int) -> HandValue:
    if expected_count == 3:
        return evaluate_3_card(cards)
    return evaluate_5_card(cards)


def available_subs(cards: Sequence[str]) -> list[str]:
    """Natural substitution candidates, excluding this row's natural cards."""
    used = {card for card in cards if not is_joker(card)}
    return [card for card in _SUB_ORDER if card not in used]


@lru_cache(maxsize=200_000)
def _evaluate_joker_key(cards_key: tuple[str, ...], expected_count: int) -> HandValue:
    cards = list(cards_key)
    non_jokers = [card for card in cards if not is_joker(card)]
    joker_count = len(cards) - len(non_jokers)
    best: HandValue | None = None
    for substitution in combinations(available_subs(cards), joker_count):
        value = _evaluate_natural([*non_jokers, *substitution], expected_count)
        if best is None or value > best:
            best = value
    if best is None:
        raise ValueError("joker hand has no legal natural substitution")
    return best


def evaluate_hand_joker(cards: Sequence[str], expected_count: int) -> HandValue:
    """Strongest value of the row across all distinct natural substitutions."""
    if len(cards) != expected_count:
        raise ValueError(
            f"evaluation requires {expected_count} cards, got {len(cards)}"
        )
    if not any(is_joker(card) for card in cards):
        return _evaluate_natural(cards, expected_count)
    key = tuple(sorted("JK" if is_joker(card) else card for card in cards))
    return _evaluate_joker_key(key, expected_count)


def constrain_row(
    cards: Sequence[str],
    ref_value: HandValue,
    expected_count: int,
) -> list[str]:
    """Strongest joker substitution whose value is ``<= ref_value``."""
    value = evaluate_hand_joker(cards, expected_count)
    if value <= ref_value:
        return list(cards)

    non_jokers = [card for card in cards if not is_joker(card)]
    joker_count = len(cards) - len(non_jokers)
    best: list[str] | None = None
    best_value: HandValue | None = None
    for substitution in combinations(available_subs(cards), joker_count):
        test = [*non_jokers, *substitution]
        test_value = _evaluate_natural(test, expected_count)
        if test_value <= ref_value and (
            best_value is None or test_value > best_value
        ):
            best = test
            best_value = test_value
    return best if best is not None else list(cards)


def top_royalty_from_value(value: HandValue) -> int:
    category, ranks = value
    if category == HAND_TRIPS:
        return 10 + (ranks[0] - 2)
    if category == HAND_PAIR and ranks[0] >= RANK_VALUE["6"]:
        return ranks[0] - 5
    return 0


def middle_royalty_from_value(value: HandValue) -> int:
    category, ranks = value
    if category == HAND_STRAIGHT_FLUSH and ranks[0] == 14:
        return 50
    return {8: 30, 7: 20, 6: 12, 5: 8, 4: 4, 3: 2}.get(category, 0)


def bottom_royalty_from_value(value: HandValue) -> int:
    category, ranks = value
    if category == HAND_STRAIGHT_FLUSH and ranks[0] == 14:
        return 25
    return {8: 15, 7: 10, 6: 6, 5: 4, 4: 2}.get(category, 0)


def fl_chain_entry_from_value(value: HandValue) -> tuple[bool, int]:
    """Chain Fantasyland entry: QQ=14, KK=15, AA=16, trips=17."""
    category, ranks = value
    if category == HAND_TRIPS:
        return True, 17
    if category == HAND_PAIR:
        if ranks[0] == 14:
            return True, 16
        if ranks[0] == 13:
            return True, 15
        if ranks[0] == 12:
            return True, 14
    return False, 0


def evaluate_board_with_joker_constraint(
    top: Sequence[str],
    middle: Sequence[str],
    bottom: Sequence[str],
) -> dict:
    """Canonical bottom-up constrained evaluation of one complete board."""
    if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
        raise ValueError("board must have 3 top, 5 middle, and 5 bottom cards")

    bottom_final = list(bottom)
    bottom_value = evaluate_hand_joker(bottom_final, 5)

    middle_final = constrain_row(middle, bottom_value, 5)
    middle_value = evaluate_hand_joker(middle_final, 5)

    top_final = constrain_row(top, middle_value, 3)
    top_value = evaluate_hand_joker(top_final, 3)

    busted = top_value > middle_value or middle_value > bottom_value

    royalties = {"top": 0, "middle": 0, "bottom": 0, "total": 0}
    fl_entry = False
    fl_card_count = 0
    if not busted:
        royalties["top"] = top_royalty_from_value(top_value)
        royalties["middle"] = middle_royalty_from_value(middle_value)
        royalties["bottom"] = bottom_royalty_from_value(bottom_value)
        royalties["total"] = (
            royalties["top"] + royalties["middle"] + royalties["bottom"]
        )
        fl_entry, fl_card_count = fl_chain_entry_from_value(top_value)

    return {
        "busted": busted,
        "top": top_final,
        "middle": middle_final,
        "bottom": bottom_final,
        "values": {
            "top": top_value,
            "middle": middle_value,
            "bottom": bottom_value,
        },
        "royalties": royalties,
        "fl_entry": fl_entry,
        "fl_card_count": fl_card_count,
    }
