"""Card constants and helpers for the no-joker regular rule set."""

from __future__ import annotations

import random
from typing import Iterable

RANKS = "23456789TJQKA"
SUITS = "hdcs"
RANK_VALUE = {rank: value for value, rank in enumerate(RANKS, start=2)}
VALUE_RANK = {value: rank for rank, value in RANK_VALUE.items()}

ALL_CARDS = tuple(f"{rank}{suit}" for suit in SUITS for rank in RANKS)
CARD_SET = set(ALL_CARDS)


def create_deck(*, shuffle: bool = True, rng: random.Random | None = None) -> list[str]:
    """Return a 52-card deck. Jokers are intentionally absent."""
    deck = list(ALL_CARDS)
    if shuffle:
        (rng or random).shuffle(deck)
    return deck


def card_rank(card: str) -> int:
    validate_cards([card])
    return RANK_VALUE[card[0]]


def card_suit(card: str) -> str:
    validate_cards([card])
    return card[1]


def validate_cards(cards: Iterable[str]) -> None:
    seen: set[str] = set()
    for card in cards:
        if card not in CARD_SET:
            raise ValueError(f"invalid regular-rule card: {card!r}")
        if card in seen:
            raise ValueError(f"duplicate card: {card!r}")
        seen.add(card)
