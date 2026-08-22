"""Seating, act order, deal schedule, and decision geometry for 3-max regular OFC.

Every constant here is fixed by ``docs/three_max_rules_contract_20260812.md``.
The heads-up tables in :mod:`ofc_regular.hu_infoset` are deliberately untouched:
3-max is a parallel geometry, never a widening of the HU one.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Mapping, TypeVar

Seat3 = Literal["sb", "bb", "btn"]
Street3 = Literal["T0", "T1", "T2", "T3", "T4"]

SEAT_SB: Seat3 = "sb"
SEAT_BB: Seat3 = "bb"
SEAT_BTN: Seat3 = "btn"

# R2: the button's left neighbour acts first, then the button's right
# neighbour, then the button.  The button therefore always closes the street.
ACT_ORDER: tuple[Seat3, ...] = (SEAT_SB, SEAT_BB, SEAT_BTN)
SEATS: frozenset[str] = frozenset(ACT_ORDER)

# R4: pairs settle in this order.  With finite stacks the order is load-bearing
# (a player emptied by an earlier pair cannot pay a later one); with the
# infinite stacks the AI assumes it is irrelevant.
SETTLEMENT_ORDER: tuple[tuple[Seat3, Seat3], ...] = (
    (SEAT_SB, SEAT_BB),
    (SEAT_SB, SEAT_BTN),
    (SEAT_BB, SEAT_BTN),
)

STREETS: tuple[Street3, ...] = ("T0", "T1", "T2", "T3", "T4")

PLAYER_COUNT = 3
OPENING_DEAL_SIZE = 5
TURN_DEAL_SIZE = 3
DECISIONS_PER_HAND = len(STREETS) * PLAYER_COUNT  # 15
CARDS_DEALT_PER_HAND = PLAYER_COUNT * (OPENING_DEAL_SIZE + 4 * TURN_DEAL_SIZE)  # 51


@dataclass(frozen=True)
class DealSlot:
    """One of the 15 decision points, with its slice of the explicit deck."""

    decision_index: int
    street: Street3
    act_order: int
    seat: Seat3
    offset: int
    size: int

    @property
    def deal_slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


def _build_deal_schedule() -> tuple[DealSlot, ...]:
    slots: list[DealSlot] = []
    offset = 0
    decision_index = 0
    for street in STREETS:
        size = OPENING_DEAL_SIZE if street == "T0" else TURN_DEAL_SIZE
        for act_order, seat in enumerate(ACT_ORDER):
            slots.append(
                DealSlot(
                    decision_index=decision_index,
                    street=street,
                    act_order=act_order,
                    seat=seat,
                    offset=offset,
                    size=size,
                )
            )
            offset += size
            decision_index += 1
    return tuple(slots)


DEAL_SCHEDULE: tuple[DealSlot, ...] = _build_deal_schedule()

# (street, act_order) -> (hero_cards, (opp_next_cards, opp_later_cards),
#                         dealt_cards, hero_private_discards)
#
# Opponent boards are always ordered act-relative: slot 0 is the player who
# acts next after the hero, slot 1 the one after that.  That keeps "who still
# has to answer my move on this street" in a positionally stable place, which
# is what a feature encoder needs.
DECISION_GEOMETRY: dict[tuple[Street3, int], tuple[int, tuple[int, int], int, int]] = {
    ("T0", 0): (0, (0, 0), 5, 0),
    ("T0", 1): (0, (0, 5), 5, 0),
    ("T0", 2): (0, (5, 5), 5, 0),
    ("T1", 0): (5, (5, 5), 3, 0),
    ("T1", 1): (5, (5, 7), 3, 0),
    ("T1", 2): (5, (7, 7), 3, 0),
    ("T2", 0): (7, (7, 7), 3, 1),
    ("T2", 1): (7, (7, 9), 3, 1),
    ("T2", 2): (7, (9, 9), 3, 1),
    ("T3", 0): (9, (9, 9), 3, 2),
    ("T3", 1): (9, (9, 11), 3, 2),
    ("T3", 2): (9, (11, 11), 3, 2),
    ("T4", 0): (11, (11, 11), 3, 3),
    ("T4", 1): (11, (11, 13), 3, 3),
    ("T4", 2): (11, (13, 13), 3, 3),
}


_V = TypeVar("_V")


class SeatMap(MappingABC):
    """An immutable, hashable seat-keyed mapping over exactly the three seats.

    A plain dict inside a frozen dataclass is neither: item assignment silently
    bypasses the owner's validation, and the synthesised ``__hash__`` raises.
    Search code memoises states, so both matter.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, _V] | Iterable[_V]) -> None:
        if isinstance(values, MappingABC):
            missing = set(ACT_ORDER) - set(values)
            if missing:
                raise ValueError(f"missing 3-max seats: {sorted(missing)}")
            unknown = set(values) - set(ACT_ORDER)
            if unknown:
                raise ValueError(f"unknown 3-max seats: {sorted(unknown)}")
            ordered = tuple(values[seat] for seat in ACT_ORDER)
        else:
            ordered = tuple(values)
            if len(ordered) != PLAYER_COUNT:
                raise ValueError(f"3-max seat map needs {PLAYER_COUNT} values")
        object.__setattr__(self, "_values", ordered)

    def __getitem__(self, seat: str) -> _V:
        try:
            return self._values[ACT_ORDER.index(seat)]  # type: ignore[index]
        except ValueError:
            raise KeyError(seat) from None

    def __iter__(self) -> Iterator[str]:
        return iter(ACT_ORDER)

    def __len__(self) -> int:
        return PLAYER_COUNT

    def __hash__(self) -> int:
        return hash((SeatMap, self._values))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SeatMap):
            return self._values == other._values
        if isinstance(other, MappingABC):
            return dict(self) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        body = ", ".join(f"{seat}={self[seat]!r}" for seat in ACT_ORDER)
        return f"SeatMap({body})"

    def replace(self, seat: str, value: _V) -> "SeatMap":
        updated = dict(self)
        if seat not in updated:
            raise KeyError(seat)
        updated[seat] = value
        return SeatMap(updated)


def placement_split(street: Street3) -> tuple[int, int]:
    """Cards placed and discarded on one decision of ``street`` (contract R1).

    T0 places all five with no discard; every later street places two of three
    and discards one.  ``apply`` enforces this: a wrong split leaves the number
    of consumed cards unchanged, so deck conservation alone cannot see it.
    """
    if street not in STREETS:
        raise ValueError(f"unknown 3-max street: {street!r}")
    if street == "T0":
        return (OPENING_DEAL_SIZE, 0)
    return (2, 1)


def act_order_of(seat: Seat3) -> int:
    """Return the within-street act order (0 first .. 2 last) of ``seat``."""
    try:
        return ACT_ORDER.index(seat)
    except ValueError:  # pragma: no cover - defensive
        raise ValueError(f"unknown 3-max seat: {seat!r}") from None


def opponent_seats(seat: Seat3) -> tuple[Seat3, Seat3]:
    """Opponents in act-relative order: next to act after ``seat``, then the other."""
    index = act_order_of(seat)
    return (ACT_ORDER[(index + 1) % PLAYER_COUNT], ACT_ORDER[(index + 2) % PLAYER_COUNT])


def seat_of_player(player: int, button_player: int) -> Seat3:
    """Map a physical seat index (0..2) to its button-relative role.

    R3: the button rotates clockwise every hand, and action moves clockwise, so
    the player one step clockwise from the button (``button_player + 1``) is the
    button's left neighbour and acts first.
    """
    _validate_player(player)
    _validate_player(button_player)
    return ACT_ORDER[(player - button_player - 1) % PLAYER_COUNT]


def player_of_seat(seat: Seat3, button_player: int) -> int:
    """Inverse of :func:`seat_of_player`."""
    _validate_player(button_player)
    return (button_player + 1 + act_order_of(seat)) % PLAYER_COUNT


def rotate_button(button_player: int) -> int:
    """Advance the button one step clockwise (R3)."""
    _validate_player(button_player)
    return (button_player + 1) % PLAYER_COUNT


def geometry_for(street: Street3, act_order: int) -> tuple[int, tuple[int, int], int, int]:
    key = (street, act_order)
    if key not in DECISION_GEOMETRY:
        raise ValueError(f"no 3-max decision geometry for {key!r}")
    return DECISION_GEOMETRY[key]


def _validate_player(player: int) -> None:
    if isinstance(player, bool) or not isinstance(player, int):
        raise TypeError("player index must be an integer")
    if not 0 <= player < PLAYER_COUNT:
        raise ValueError(f"player index out of range: {player}")
