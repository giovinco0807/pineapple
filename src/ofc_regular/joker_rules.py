"""Joker (54-card) rule definitions for the unified engine.

Port target per the accepted decision record
``ai/docs/engine_unification_decision_20260728.md`` (B-phased): the canonical
Joker ruleset of the main track (`ai/engine`) hosted inside this repository's
rules-parameterized engine.  The 52-card ``REGULAR_RULES`` behavior is
unchanged; everything Joker-specific lives behind the new ruleset.

Canonical contract (must match `ai/engine/game_engine.py`):

- Two physically distinct jokers ``X1`` / ``X2`` on top of the 52 naturals.
- Joker hands take exhaustive, distinct natural-card substitutions; rows are
  resolved bottom-up with each upper row capped by the row below.
- Fantasyland entry is the chain schedule: QQ=14, KK=15, AA=16, trips=17.
"""

from __future__ import annotations

from typing import Iterable

from .cards import ALL_CARDS as NATURAL_CARDS
from .rules import RuleSet

JOKERS: tuple[str, str] = ("X1", "X2")
ALL_CARDS_54: tuple[str, ...] = tuple(NATURAL_CARDS) + JOKERS
CARD_SET_54 = set(ALL_CARDS_54)

JOKER_CHAIN_RULES = RuleSet(
    name="joker_chain",
    include_jokers=True,
    fl_entry_cards={
        "qq": 14,
        "kk": 15,
        "aa": 16,
        "trips": 17,
    },
    fl_stay_cards=14,
)


def is_joker(card: str) -> bool:
    """Match the canonical evaluator, including the legacy ``JK`` label."""
    return card in ("X1", "X2", "JK")


def validate_cards_54(cards: Iterable[str]) -> None:
    seen: set[str] = set()
    for card in cards:
        if card == "JK":
            continue  # legacy label: identity handled by the caller
        if card not in CARD_SET_54:
            raise ValueError(f"invalid joker-rule card: {card!r}")
        if card in seen:
            raise ValueError(f"duplicate card: {card!r}")
        seen.add(card)
