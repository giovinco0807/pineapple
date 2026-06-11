"""Regular OFC Pineapple rule definitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .cards import RANK_VALUE, validate_cards


@dataclass(frozen=True)
class RuleSet:
    name: str
    include_jokers: bool
    fl_entry_cards: Mapping[str, int]
    fl_stay_cards: int


@dataclass(frozen=True)
class FantasylandEntry:
    qualifies: bool
    card_count: int
    entry_type: str | None


REGULAR_RULES = RuleSet(
    name="regular",
    include_jokers=False,
    fl_entry_cards={
        "qq": 14,
        "kk": 14,
        "aa": 14,
        "trips": 14,
    },
    fl_stay_cards=14,
)


def fl_entry_type(top_cards: Sequence[str]) -> str | None:
    """Return the regular-rule Fantasyland entry type for a top row."""
    if len(top_cards) != 3:
        return None
    validate_cards(top_cards)
    ranks = [card[0] for card in top_cards]
    counts = Counter(ranks)

    if any(count >= 3 for count in counts.values()):
        return "trips"
    if counts.get("A", 0) >= 2:
        return "aa"
    if counts.get("K", 0) >= 2:
        return "kk"
    if counts.get("Q", 0) >= 2:
        return "qq"
    return None


def check_fl_entry(
    top_cards: Sequence[str],
    rules: RuleSet = REGULAR_RULES,
) -> FantasylandEntry:
    """Check Fantasyland entry.

    Entry conditions match the existing pineapple rule, but regular mode deals
    14 cards for every entry type.
    """
    entry_type = fl_entry_type(top_cards)
    if entry_type is None:
        return FantasylandEntry(False, 0, None)
    return FantasylandEntry(True, int(rules.fl_entry_cards[entry_type]), entry_type)


def check_fl_stay(
    top_cards: Sequence[str],
    bottom_cards: Sequence[str],
    rules: RuleSet = REGULAR_RULES,
) -> FantasylandEntry:
    """Check Fantasyland stay.

    Stay condition is unchanged: trips on top or quads+ on bottom. Regular mode
    always awards the next Fantasyland as 14 cards.
    """
    from .evaluator import HAND_QUADS, HAND_TRIPS, evaluate_3_card, evaluate_5_card

    if len(top_cards) != 3 or len(bottom_cards) != 5:
        return FantasylandEntry(False, 0, None)
    validate_cards([*top_cards, *bottom_cards])

    top_value = evaluate_3_card(top_cards)
    bottom_value = evaluate_5_card(bottom_cards)
    if top_value[0] == HAND_TRIPS:
        return FantasylandEntry(True, rules.fl_stay_cards, "stay_top_trips")
    if bottom_value[0] >= HAND_QUADS:
        return FantasylandEntry(True, rules.fl_stay_cards, "stay_bottom_quads_plus")
    return FantasylandEntry(False, 0, None)
