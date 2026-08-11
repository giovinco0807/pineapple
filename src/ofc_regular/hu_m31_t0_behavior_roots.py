"""Generate hidden-discard-safe T0 roots from explicit behavior profiles.

Mirrors `hu_m31_t1_behavior_roots` one street earlier, which makes it the
degenerate case of that module: T0 is the opening street, so there is nothing to
play out before the first seat's root. The first root is the hand as dealt --
two empty boards and five cards -- and the only behavior actually exercised is
the first seat's own T0, which has to be played before the second seat's root
exists at all.

Both roots are `ActorObservation`s, so the opponent's private discards never
enter any model input. At T0 there are none yet in either case: the opening
street places all five dealt cards and discards nothing, which is why both
geometries carry a zero discard count rather than an omitted one.
"""

from __future__ import annotations

import random
from typing import Iterable

from .ai_profiles import ModelBundle, build_policy
from .cards import create_deck
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from .state import Board


def _observation(
    *,
    player: int,
    boards: list[Board],
    private_discards: list[list[str]],
    dealt_cards: Iterable[str],
    street: str,
) -> ActorObservation:
    seat = "first" if player == 0 else "second"
    return ActorObservation(
        hero_board=boards[player],
        opponent_public_board=boards[1 - player],
        dealt_cards=tuple(dealt_cards),
        hero_private_discards=tuple(private_discards[player]),
        seat=seat,
        street=street,
        to_act_order=seat,
    )


def generate_behavior_t0_roots(
    *,
    hand_seed: int,
    behavior_seed: int,
    profile: str,
    bundle: ModelBundle,
) -> tuple[ActorObservation, ActorObservation]:
    """Expose the T0-first root as dealt and the T0-second root that follows it.

    The first seat's T0 is played with one explicit frozen profile, because the
    second seat's root is not defined until it has been: acting second at T0 the
    opponent's five placed cards are the whole of the public information the
    decision turns on.
    """

    if profile not in M31_T3_BEHAVIOR_PROFILES or profile == "current":
        raise ValueError("behavior profile must be explicit and frozen")
    for name, value in (("hand_seed", hand_seed), ("behavior_seed", behavior_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    private_discards: list[list[str]] = [[], []]
    policies = [
        build_policy(
            profile,
            bundle,
            seed=behavior_seed + player,
            seat="first" if player == 0 else "second",
            opening_lookahead_samples=0,
        )
        for player in (0, 1)
    ]

    def deal(count: int) -> tuple[str, ...]:
        nonlocal cursor
        cards = tuple(deck[cursor : cursor + count])
        cursor += count
        if len(cards) != count:
            raise RuntimeError("T0 behavior-root deck was exhausted")
        return cards

    def advance(player: int, dealt: tuple[str, ...], street: str, serial: int) -> None:
        observation = _observation(
            player=player,
            boards=boards,
            private_discards=private_discards,
            dealt_cards=dealt,
            street=street,
        )
        action = policies[player].choose_action_observation(
            observation,
            hand_id=hand_seed,
            game_id=hand_seed,
            decision_seed=behavior_seed + 100 + serial,
        )
        boards[player] = boards[player].place(action.placements)
        private_discards[player].extend(action.discards)

    first_dealt = deal(5)
    first = _observation(
        player=0,
        boards=boards,
        private_discards=private_discards,
        dealt_cards=first_dealt,
        street="T0",
    )
    advance(0, first_dealt, "T0", 0)
    second = _observation(
        player=1,
        boards=boards,
        private_discards=private_discards,
        dealt_cards=deal(5),
        street="T0",
    )
    return first, second


__all__ = ["generate_behavior_t0_roots"]
