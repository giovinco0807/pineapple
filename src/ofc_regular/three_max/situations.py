"""Which seats are in Fantasyland, and what that reduces the hand to.

A 3-max hand has six hero-perspective situations.  Three of them are the
hero playing a normal board, and those are the ones that need street policies:

===========  ==========================  ==============================================
label        hero / opponents            what the hero's decision problem reduces to
===========  ==========================  ==============================================
``NNN``      normal vs 2 normal          genuinely new: the 3-max chain
``NNF``      normal vs 1 normal + 1 FL   heads-up NORMAL shape (one visible opponent)
``NFF``      normal vs 2 FL              heads-up VS-FL shape (no visible opponent)
``FNN``      FL vs 2 normal              Fantasyland best response to two boards
``FNF``      FL vs 1 normal + 1 FL       BR to the visible board, blind to the other FL
``FFF``      FL vs 2 FL                  the blind Fantasyland solve
===========  ==========================  ==============================================

The reduction is not a loose analogy.  Scoring is a sum of independent pairwise
comparisons (contract R4), so the hero's value decomposes exactly into one term
per opponent.  And a Fantasyland opponent shows nothing until showdown, so it
contributes zero visible cards -- which makes the hero's unseen-card count in
``NNF`` equal, street for street, to heads-up normal play, and in ``NFF`` equal
to heads-up vs-Fantasyland play.  ``reduction_of`` states this and
``tests/test_three_max_situations.py`` pins the numbers.

Two model mismatches survive the reduction and must not be forgotten:

1. In ``NNF`` the Fantasyland player best-responds to BOTH normal boards, since
   its own score is the sum over the two of them.  A heads-up vs-FL model
   assumes a Fantasyland opponent dedicated to punishing the hero alone, so it
   is PESSIMISTIC here -- a compromise response can only score lower against
   the hero than a dedicated one.
2. Also in ``NNF``, the visible normal opponent is itself playing a three-way
   game: its objective includes its own pair against the Fantasyland player, so
   it does not play the heads-up strategy.  The geometry is heads-up, the
   opponent's policy is not.

``NFF`` has neither problem: each Fantasyland player faces exactly one normal
board -- the hero's -- so each is a dedicated best response, and by linearity
of expectation the hero's value is exactly twice the heads-up vs-FL value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Mapping

from .seating import (
    ACT_ORDER,
    OPENING_DEAL_SIZE,
    PLAYER_COUNT,
    STREETS,
    Seat3,
    SeatMap,
    Street3,
    TURN_DEAL_SIZE,
    placement_split,
)

DECK_SIZE = 52
FANTASYLAND_CARDS = 14
FANTASYLAND_PLACED = 13
NORMAL_CARDS_PER_HAND = 17  # 13 placed + 4 discarded

Reduction = Literal[
    "three_max_normal",
    "heads_up_normal_shaped",
    "heads_up_vs_fl_shaped",
    "fantasyland_best_response",
    "fantasyland_blind",
]


@dataclass(frozen=True)
class FantasylandDeal:
    """A Fantasyland seat's 14 cards, dealt before any street (contract R5)."""

    seat: Seat3
    offset: int
    size: int = FANTASYLAND_CARDS

    @property
    def deal_slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


@dataclass(frozen=True)
class StreetSlot:
    """One street decision, indexed among the NORMAL seats only."""

    decision_index: int
    street: Street3
    act_index: int
    seat: Seat3
    offset: int
    size: int

    @property
    def deal_slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


@dataclass(frozen=True)
class TableConfig:
    """Which of the three seats are in Fantasyland this hand."""

    fantasyland: SeatMap

    def __post_init__(self) -> None:
        object.__setattr__(self, "fantasyland", SeatMap(self.fantasyland))

    @classmethod
    def all_normal(cls) -> "TableConfig":
        return cls(SeatMap({seat: False for seat in ACT_ORDER}))

    @classmethod
    def of(cls, *fantasyland_seats: Seat3) -> "TableConfig":
        unknown = set(fantasyland_seats) - set(ACT_ORDER)
        if unknown:
            raise ValueError(f"unknown 3-max seats: {sorted(unknown)}")
        return cls(SeatMap({seat: seat in fantasyland_seats for seat in ACT_ORDER}))

    @property
    def normal_seats(self) -> tuple[Seat3, ...]:
        return tuple(seat for seat in ACT_ORDER if not self.fantasyland[seat])

    @property
    def fantasyland_seats(self) -> tuple[Seat3, ...]:
        return tuple(seat for seat in ACT_ORDER if self.fantasyland[seat])

    @property
    def cards_dealt(self) -> int:
        return (
            len(self.normal_seats) * NORMAL_CARDS_PER_HAND
            + len(self.fantasyland_seats) * FANTASYLAND_CARDS
        )

    @property
    def undealt_cards(self) -> int:
        return DECK_SIZE - self.cards_dealt

    def label(self, hero: Seat3) -> str:
        """Hero-perspective label, e.g. ``NNF``: hero first, then opponents."""
        from .seating import opponent_seats

        letters = ["F" if self.fantasyland[hero] else "N"]
        letters += [
            "F" if self.fantasyland[seat] else "N" for seat in opponent_seats(hero)
        ]
        return "".join(letters)

    def act_index_of(self, seat: Seat3) -> int:
        """Position of ``seat`` among the seats that actually play streets."""
        if self.fantasyland[seat]:
            raise ValueError(f"seat {seat} is in Fantasyland and plays no streets")
        return self.normal_seats.index(seat)


def all_configs() -> Iterator[TableConfig]:
    """Every table configuration, from all-normal to all-Fantasyland."""
    for mask in range(1 << PLAYER_COUNT):
        yield TableConfig.of(
            *(seat for index, seat in enumerate(ACT_ORDER) if mask & (1 << index))
        )


def reduction_of(config: TableConfig, hero: Seat3) -> Reduction:
    """Which existing problem family the hero's decision reduces to."""
    if config.fantasyland[hero]:
        if len(config.normal_seats) == 0:
            return "fantasyland_blind"
        return "fantasyland_best_response"
    normal_opponents = len(config.normal_seats) - 1
    if normal_opponents == 2:
        return "three_max_normal"
    if normal_opponents == 1:
        return "heads_up_normal_shaped"
    return "heads_up_vs_fl_shaped"


def deal_plan(config: TableConfig) -> tuple[tuple[FantasylandDeal, ...], tuple[StreetSlot, ...]]:
    """Fantasyland hands first (R5), then the normal seats' street schedule."""
    offset = 0
    fantasyland: list[FantasylandDeal] = []
    for seat in config.fantasyland_seats:
        fantasyland.append(FantasylandDeal(seat=seat, offset=offset))
        offset += FANTASYLAND_CARDS

    slots: list[StreetSlot] = []
    decision_index = 0
    for street in STREETS:
        size = OPENING_DEAL_SIZE if street == "T0" else TURN_DEAL_SIZE
        for act_index, seat in enumerate(config.normal_seats):
            slots.append(
                StreetSlot(
                    decision_index=decision_index,
                    street=street,
                    act_index=act_index,
                    seat=seat,
                    offset=offset,
                    size=size,
                )
            )
            offset += size
            decision_index += 1

    if offset != config.cards_dealt:
        raise AssertionError(
            f"deal plan consumes {offset} cards, config says {config.cards_dealt}"
        )
    return tuple(fantasyland), tuple(slots)


def street_geometry(
    config: TableConfig, street: Street3, act_index: int
) -> tuple[int, tuple[int, ...], int, int]:
    """``(hero_cards, opponent_visible_counts, dealt, hero_discards)``.

    Opponent counts follow the act-relative order used by the observation, and
    a Fantasyland opponent always contributes 0: its 14 cards stay hidden until
    showdown, which is exactly why ``NNF`` and ``NFF`` inherit the heads-up
    unseen-card arithmetic.
    """
    from .seating import opponent_seats

    normal = config.normal_seats
    if not 0 <= act_index < len(normal):
        raise ValueError(
            f"act_index {act_index} out of range for {len(normal)} normal seats"
        )
    street_index = STREETS.index(street)
    hero = normal[act_index]

    hero_cards = 0 if street_index == 0 else 5 + 2 * (street_index - 1)
    hero_discards = 0 if street_index == 0 else street_index - 1
    dealt = OPENING_DEAL_SIZE if street_index == 0 else TURN_DEAL_SIZE

    # An opponent who already acted this street is ahead by that street's
    # placement count -- five on the opening street, two on every other.
    placed_this_street = placement_split(street)[0]
    counts: list[int] = []
    for seat in opponent_seats(hero):
        if config.fantasyland[seat]:
            counts.append(0)
            continue
        acted_this_street = config.act_index_of(seat) < act_index
        counts.append(hero_cards + (placed_this_street if acted_this_street else 0))
    return hero_cards, tuple(counts), dealt, hero_discards


def hero_unseen_count(
    config: TableConfig, street: Street3, act_index: int
) -> int:
    """Cards the hero cannot see at this decision, after receiving its deal."""
    hero_cards, opponent_counts, dealt, discards = street_geometry(
        config, street, act_index
    )
    return DECK_SIZE - (hero_cards + discards + dealt + sum(opponent_counts))


def street_decision_count(config: TableConfig) -> int:
    return len(config.normal_seats) * len(STREETS)


def situation_summary(config: TableConfig, hero: Seat3) -> Mapping[str, object]:
    """Everything a caller needs to route a hand to the right machinery."""
    return {
        "label": config.label(hero),
        "reduction": reduction_of(config, hero),
        "normal_seats": config.normal_seats,
        "fantasyland_seats": config.fantasyland_seats,
        "cards_dealt": config.cards_dealt,
        "undealt_cards": config.undealt_cards,
        "street_decisions": street_decision_count(config),
        "hero_acts": not config.fantasyland[hero],
        "hero_act_index": (
            None if config.fantasyland[hero] else config.act_index_of(hero)
        ),
    }
