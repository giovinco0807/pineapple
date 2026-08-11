"""Generate hidden-discard-safe T3 roots from explicit behavior profiles."""

from __future__ import annotations

import random
from typing import Iterable

from .ai_profiles import ModelBundle, build_policy
from .cards import create_deck
from .hu_infoset import ActorObservation
from .state import Board


M31_T3_BEHAVIOR_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)


def behavior_profile_for_index(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("behavior index must be a nonnegative integer")
    return M31_T3_BEHAVIOR_PROFILES[index % len(M31_T3_BEHAVIOR_PROFILES)]


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


def generate_behavior_t3_roots(
    *,
    hand_seed: int,
    behavior_seed: int,
    profile: str,
    bundle: ModelBundle,
) -> tuple[ActorObservation, ActorObservation]:
    """Play T0-T2 with one explicit profile and expose both sequential T3 roots.

    Each decision receives only the actor observation.  In particular, the
    opponent's private discards never enter ``dead_cards`` or any model input.
    """

    if profile not in M31_T3_BEHAVIOR_PROFILES or profile == "current":
        raise ValueError("M3.1 behavior profile must be explicit and frozen")
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
            raise RuntimeError("M3.1 behavior-root deck was exhausted")
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

    serial = 0
    for player in (0, 1):
        advance(player, deal(5), "T0", serial)
        serial += 1
    for street in ("T1", "T2"):
        for player in (0, 1):
            advance(player, deal(3), street, serial)
            serial += 1

    first_dealt = deal(3)
    first = _observation(
        player=0,
        boards=boards,
        private_discards=private_discards,
        dealt_cards=first_dealt,
        street="T3",
    )
    advance(0, first_dealt, "T3", serial)
    second = _observation(
        player=1,
        boards=boards,
        private_discards=private_discards,
        dealt_cards=deal(3),
        street="T3",
    )
    return first, second


__all__ = [
    "M31_T3_BEHAVIOR_PROFILES",
    "behavior_profile_for_index",
    "generate_behavior_t3_roots",
]
