"""Simulator state and the information-set-safe observation for 3-max regular OFC.

``WorldState3`` belongs to the simulator and holds every player's private
discards plus the explicit deck.  ``ThreeMaxObservation`` is the only
card-bearing view a policy, feature encoder, or search root may see: it carries
the hero's own cards, the two public opponent boards, and nothing else.

This is the M0 shape.  It deliberately does not yet mint the engine-facing
``actor_observation_3max_v1`` schema (fingerprints, request envelopes, belief
partitions); that lands with the 3-max engine in M3.  What is fixed here is the
part the contract already decided: the deal schedule, the act-relative
opponent ordering, and the 15-point decision geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..action_space import Action
from ..cards import ALL_CARDS, validate_cards
from ..state import Board
from .seating import (
    ACT_ORDER,
    CARDS_DEALT_PER_HAND,
    DEAL_SCHEDULE,
    DECISIONS_PER_HAND,
    DealSlot,
    Seat3,
    SeatMap,
    Street3,
    act_order_of,
    geometry_for,
    opponent_seats,
    placement_split,
)

DECK_SIZE = len(ALL_CARDS)


@dataclass(frozen=True)
class ThreeMaxObservation:
    """What one actor may see at one of the 15 decision points."""

    hero_board: Board
    opponent_boards: tuple[Board, Board]
    dealt_cards: tuple[str, ...]
    hero_private_discards: tuple[str, ...]
    seat: Seat3
    street: Street3

    def __post_init__(self) -> None:
        object.__setattr__(self, "dealt_cards", tuple(self.dealt_cards))
        object.__setattr__(
            self, "hero_private_discards", tuple(self.hero_private_discards)
        )
        object.__setattr__(self, "opponent_boards", tuple(self.opponent_boards))
        if len(self.opponent_boards) != 2:
            raise ValueError("3-max observation needs exactly two opponent boards")
        self.hero_board.validate()
        for board in self.opponent_boards:
            board.validate()
        expected = geometry_for(self.street, self.act_order)
        hero_cards, opponent_cards, dealt_count, discard_count = expected
        actual = (
            self.hero_board.card_count(),
            tuple(board.card_count() for board in self.opponent_boards),
            len(self.dealt_cards),
            len(self.hero_private_discards),
        )
        if actual != (hero_cards, opponent_cards, dealt_count, discard_count):
            raise ValueError(
                f"3-max geometry violation at {(self.street, self.seat)}: "
                f"expected {expected}, got {actual}"
            )
        validate_cards(self.visible_cards())

    @property
    def act_order(self) -> int:
        return act_order_of(self.seat)

    @property
    def opponent_seats(self) -> tuple[Seat3, Seat3]:
        """Seats of ``opponent_boards``, act-relative: next to act, then the other."""
        return opponent_seats(self.seat)

    def visible_cards(self) -> tuple[str, ...]:
        """Every card this actor can see, including its own private discards."""
        cards: list[str] = [
            *self.hero_board.all_cards(),
            *self.hero_private_discards,
            *self.dealt_cards,
        ]
        for board in self.opponent_boards:
            cards.extend(board.all_cards())
        return tuple(cards)

    def unknown_cards(self) -> tuple[str, ...]:
        """Cards this actor cannot see: opponent discards plus the undealt tail."""
        seen = set(self.visible_cards())
        return tuple(card for card in ALL_CARDS if card not in seen)

    def unknown_card_count(self) -> int:
        return DECK_SIZE - len(self.visible_cards())


@dataclass(frozen=True)
class WorldState3:
    """Full simulator state for one 3-max hand over an explicit deck."""

    deck: tuple[str, ...]
    boards: SeatMap
    private_discards: SeatMap
    decision_index: int = 0

    def __post_init__(self) -> None:
        deck = tuple(self.deck)
        if len(deck) != DECK_SIZE:
            raise ValueError(f"3-max world needs an explicit {DECK_SIZE}-card deck")
        validate_cards(deck)
        object.__setattr__(self, "deck", deck)

        boards = SeatMap(self.boards)
        discards = SeatMap(
            {seat: tuple(self.private_discards[seat]) for seat in ACT_ORDER}
        )
        object.__setattr__(self, "boards", boards)
        object.__setattr__(self, "private_discards", discards)

        if not 0 <= self.decision_index <= DECISIONS_PER_HAND:
            raise ValueError(f"decision_index out of range: {self.decision_index}")

        consumed: list[str] = []
        for seat in ACT_ORDER:
            boards[seat].validate()
            consumed.extend(boards[seat].all_cards())
            consumed.extend(discards[seat])
        validate_cards(consumed)
        dealt_so_far = sum(slot.size for slot in DEAL_SCHEDULE[: self.decision_index])
        if len(consumed) != dealt_so_far:
            raise ValueError(
                f"deck conservation broken: {len(consumed)} cards held after "
                f"{self.decision_index} decisions, schedule says {dealt_so_far}"
            )
        if set(consumed) != set(deck[:dealt_so_far]):
            raise ValueError("players hold cards the deal schedule did not deal")

    @classmethod
    def new_hand(cls, deck: Iterable[str]) -> "WorldState3":
        empty = Board.from_rows()
        return cls(
            deck=tuple(deck),
            boards=SeatMap({seat: empty for seat in ACT_ORDER}),
            private_discards=SeatMap({seat: () for seat in ACT_ORDER}),
            decision_index=0,
        )

    @property
    def is_terminal(self) -> bool:
        return self.decision_index >= DECISIONS_PER_HAND

    @property
    def undealt_cards(self) -> tuple[str, ...]:
        """The deck tail no player ever receives (one card in a full 3-max hand)."""
        return self.deck[CARDS_DEALT_PER_HAND:]

    def current_slot(self) -> DealSlot:
        if self.is_terminal:
            raise ValueError("hand is over; no decision remains")
        return DEAL_SCHEDULE[self.decision_index]

    def dealt_cards(self) -> tuple[str, ...]:
        slot = self.current_slot()
        return self.deck[slot.deal_slice]

    def observe(self) -> ThreeMaxObservation:
        slot = self.current_slot()
        return ThreeMaxObservation(
            hero_board=self.boards[slot.seat],
            opponent_boards=tuple(
                self.boards[seat] for seat in opponent_seats(slot.seat)
            ),
            dealt_cards=self.dealt_cards(),
            hero_private_discards=self.private_discards[slot.seat],
            seat=slot.seat,
            street=slot.street,
        )

    def apply(self, action: Action) -> "WorldState3":
        slot = self.current_slot()
        dealt = self.dealt_cards()
        placed = tuple(card for card, _row in action.placements)
        discards = tuple(action.discards)

        validate_cards((*placed, *discards))
        if sorted((*placed, *discards)) != sorted(dealt):
            raise ValueError(
                f"action at {(slot.street, slot.seat)} must use exactly the dealt "
                f"cards {dealt}, got placements {placed} and discards {discards}"
            )

        # A wrong split (say three placements and no discard) consumes the same
        # cards, so deck conservation cannot see it and the corruption would
        # surface a decision later, blamed on the next seat.
        expected_placed, expected_discarded = placement_split(slot.street)
        if (len(placed), len(discards)) != (expected_placed, expected_discarded):
            raise ValueError(
                f"action at {(slot.street, slot.seat)} must place {expected_placed} "
                f"and discard {expected_discarded}, got {len(placed)} and "
                f"{len(discards)}"
            )

        return WorldState3(
            deck=self.deck,
            boards=self.boards.replace(
                slot.seat, self.boards[slot.seat].place(action.placements)
            ),
            private_discards=self.private_discards.replace(
                slot.seat, (*self.private_discards[slot.seat], *discards)
            ),
            decision_index=self.decision_index + 1,
        )
