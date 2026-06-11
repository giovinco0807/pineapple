"""Board state primitives for regular OFC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .cards import validate_cards

ROWS = ("top", "middle", "bottom")
ROW_CAPACITY = {"top": 3, "middle": 5, "bottom": 5}


@dataclass(frozen=True)
class Board:
    top: tuple[str, ...] = field(default_factory=tuple)
    middle: tuple[str, ...] = field(default_factory=tuple)
    bottom: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_rows(
        cls,
        top: Iterable[str] = (),
        middle: Iterable[str] = (),
        bottom: Iterable[str] = (),
    ) -> "Board":
        board = cls(tuple(top), tuple(middle), tuple(bottom))
        board.validate()
        return board

    def all_cards(self) -> tuple[str, ...]:
        return (*self.top, *self.middle, *self.bottom)

    def card_count(self) -> int:
        return len(self.all_cards())

    def is_complete(self) -> bool:
        return self.card_count() == 13 and all(self.open_slots(row) == 0 for row in ROWS)

    def open_slots(self, row: str) -> int:
        if row not in ROW_CAPACITY:
            raise ValueError(f"unknown row: {row}")
        return ROW_CAPACITY[row] - len(getattr(self, row))

    def validate(self) -> None:
        for row in ROWS:
            if len(getattr(self, row)) > ROW_CAPACITY[row]:
                raise ValueError(f"{row} row exceeds capacity")
        validate_cards(self.all_cards())

    def place(self, placements: Iterable[tuple[str, str]]) -> "Board":
        rows = {row: list(getattr(self, row)) for row in ROWS}
        for card, row in placements:
            if row not in ROW_CAPACITY:
                raise ValueError(f"unknown row: {row}")
            rows[row].append(card)
        return Board.from_rows(rows["top"], rows["middle"], rows["bottom"])
