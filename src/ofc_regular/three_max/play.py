"""Deterministic 3-max hand and session loops.

A hand is 15 decisions over an explicit 52-card deck (contract R1/R2).  A
session rotates the button clockwise between hands (R3) and, when stacks are
supplied, settles them in the contract's pair order (R4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol

from ..action_space import Action, generate_actions
from ..cards import create_deck
from ..counter_rng import CounterRngKey
from ..state import Board
from .scoring import HandSettlement, settle_hand
from .seating import (
    ACT_ORDER,
    PLAYER_COUNT,
    Seat3,
    Street3,
    rotate_button,
    seat_of_player,
)
from .world import ThreeMaxObservation, WorldState3

RUN_ID = "regular_ofc_3max_hand"


class Policy3(Protocol):
    """A 3-max policy: choose one legal action for one observation.

    ``decision_seed`` is derived from semantic coordinates, never from a legal
    action index, so a policy that randomises stays reproducible and stays
    independent of enumeration order.
    """

    def __call__(
        self, observation: ThreeMaxObservation, decision_seed: int
    ) -> Action:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class HandResult3:
    world: WorldState3
    settlement: HandSettlement

    @property
    def boards(self) -> Mapping[Seat3, Board]:
        return self.world.boards


@dataclass(frozen=True)
class SessionHand:
    hand_index: int
    button_player: int
    seats: Mapping[int, Seat3]
    result: HandResult3
    stacks_after: Mapping[int, float] | None


def decision_seed(
    *,
    base_seed: int,
    decision_index: int,
    seat: Seat3,
    street: Street3,
) -> int:
    """Domain-separate one decision by (hand seed, seat, street, decision index).

    The seat rides in ``stream`` rather than in ``actor`` on purpose: the
    heads-up ``CounterRngKey`` validator accepts integer actors 0 and 1 only,
    and 3-max must not loosen a heads-up validator to fit.
    """
    return CounterRngKey(
        base_seed=base_seed,
        run_id=RUN_ID,
        phase="three_max_hand",
        sample_index=0,
        actor="hero",
        street=street,
        stream=f"three_max_seat:{seat}",
        counter=decision_index,
    ).seed()


def uniform_random_policy(observation: ThreeMaxObservation, seed: int) -> Action:
    """Pick uniformly among legal actions. Smoke-test policy, not a baseline."""
    actions = generate_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise ValueError("no legal action available")
    return actions[random.Random(seed).randrange(len(actions))]


def play_hand(
    *,
    seed: int,
    policies: Mapping[Seat3, Policy3],
    fl_ev_per_pair: Mapping[int, float] | None = None,
    stacks: Mapping[Seat3, float] | None = None,
    deck: Iterable[str] | None = None,
) -> HandResult3:
    """Play one 3-max hand to settlement.

    ``stacks=None`` settles with infinite stacks, which is what the AI assumes.
    """
    missing = set(ACT_ORDER) - set(policies)
    if missing:
        raise ValueError(f"missing 3-max policies for seats: {sorted(missing)}")

    cards = list(deck) if deck is not None else create_deck(
        shuffle=True, rng=random.Random(seed)
    )
    world = WorldState3.new_hand(cards)

    while not world.is_terminal:
        slot = world.current_slot()
        observation = world.observe()
        action = policies[slot.seat](
            observation,
            decision_seed(
                base_seed=seed,
                decision_index=slot.decision_index,
                seat=slot.seat,
                street=slot.street,
            ),
        )
        world = world.apply(action)

    settlement = settle_hand(
        world.boards, fl_ev_per_pair=fl_ev_per_pair, stacks=stacks
    )
    return HandResult3(world=world, settlement=settlement)


def play_session(
    *,
    base_seed: int,
    policies_by_player: Mapping[int, Policy3],
    hands: int,
    button_player: int = 0,
    starting_stacks: Mapping[int, float] | None = None,
    fl_ev_per_pair: Mapping[int, float] | None = None,
) -> list[SessionHand]:
    """Play ``hands`` hands, rotating the button clockwise between them.

    With ``starting_stacks`` the session carries stacks across hands and each
    hand settles under the contract's capping rule; without them every hand
    settles with infinite stacks.
    """
    if hands < 0:
        raise ValueError("hands must be non-negative")
    missing = set(range(PLAYER_COUNT)) - set(policies_by_player)
    if missing:
        raise ValueError(f"missing 3-max policies for players: {sorted(missing)}")

    stacks = (
        {int(player): float(value) for player, value in starting_stacks.items()}
        if starting_stacks is not None
        else None
    )
    if stacks is not None and set(stacks) != set(range(PLAYER_COUNT)):
        raise ValueError("starting_stacks must cover players 0..2")

    played: list[SessionHand] = []
    button = button_player
    for hand_index in range(hands):
        seats = {
            player: seat_of_player(player, button) for player in range(PLAYER_COUNT)
        }
        seat_policies = {seats[player]: policies_by_player[player] for player in seats}
        seat_stacks = (
            {seats[player]: stacks[player] for player in seats}
            if stacks is not None
            else None
        )
        result = play_hand(
            seed=base_seed + hand_index,
            policies=seat_policies,
            fl_ev_per_pair=fl_ev_per_pair,
            stacks=seat_stacks,
        )
        if stacks is not None:
            final = result.settlement.final_stacks
            assert final is not None  # settle_hand returns stacks when given them
            stacks = {player: final[seats[player]] for player in seats}
        played.append(
            SessionHand(
                hand_index=hand_index,
                button_player=button,
                seats=seats,
                result=result,
                stacks_after=dict(stacks) if stacks is not None else None,
            )
        )
        button = rotate_button(button)
    return played


__all__ = [
    "HandResult3",
    "Policy3",
    "SessionHand",
    "decision_seed",
    "play_hand",
    "play_session",
    "uniform_random_policy",
]
