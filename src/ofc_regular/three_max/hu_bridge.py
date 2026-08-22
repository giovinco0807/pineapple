"""Play 3-max streets with the frozen heads-up models.

The heads-up engine cannot be shown a 3-max table.  ``ActorObservation`` carries
exactly one opponent board, and its geometry table validates every card count,
so a three-handed position cannot be constructed at all -- and the discard field
is not a spare slot for the third player either, because its length is checked
against the same table.  Relaxing either check is the one thing this track must
not do: the heads-up release is frozen and its validators are what keep the two
tracks from contaminating each other.

What CAN be done is show the heads-up model one opponent at a time.  For every
3-max decision there is a heads-up sub-game with identical card counts:

    SB  (acts first)   both opponents level with the hero    -> HU "first"
    BB  (acts second)  the SB is two cards ahead             -> HU "second"
    BTN (acts last)    both opponents are two cards ahead    -> HU "second"

The rule used here is *show the most recent actor* -- the SB for the BB, the BB
for the BTN, and for the SB (who opens, so nobody has acted) the player who
answers next.  That is the opponent whose board says the most about the street
being played.

This is a root generator, not a policy to ship.  It plays each hand as if the
unshown player did not exist, which is wrong in a way that matters: 3-max value
is a sum over BOTH opponents.  The right fix is the pairwise decomposition --
rank with the heads-up teacher once per opponent and add the values -- but that
path costs 20 s at T0 against ``decide``'s 0.09 s, so it is not what builds a
twenty-thousand-root corpus.  What this buys is boards played by a model that
actually won +1.089/hand heads-up, instead of by a Monte-Carlo referee whose
own continuation model fouls 70% of the time.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ..action_space import Action, generate_actions
from ..state import ROWS
from .seating import ACT_ORDER, SEAT_BB, SEAT_BTN, SEAT_SB, STREETS, Seat3
from .world import ThreeMaxObservation

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Which opponent slot (act-relative index) to show, and the heads-up act order
# the resulting sub-game is in.  ``opponent_seats`` returns (next to act, the
# one after), so slot 0 is the next actor and slot 1 the previous one.
_SHOWN_OPPONENT: dict[Seat3, tuple[int, str]] = {
    SEAT_SB: (0, "first"),    # nobody has acted; show the player who answers
    SEAT_BB: (1, "second"),   # the SB acted before us
    SEAT_BTN: (1, "second"),  # the BB acted immediately before us
}


class HeadsUpEngineUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _engine():
    try:
        from trainer import engine_eval
    except Exception as exc:  # pragma: no cover - import wiring
        raise HeadsUpEngineUnavailable(f"trainer.engine_eval unavailable: {exc}") from exc
    if not engine_eval.available():
        raise HeadsUpEngineUnavailable("heads-up engine reports itself unavailable")
    return engine_eval


def available() -> bool:
    try:
        _engine()
    except HeadsUpEngineUnavailable:
        return False
    return True


def _rows(board) -> dict[str, list[str]]:
    return {row: list(getattr(board, row)) for row in ROWS}


def heads_up_view(observation: ThreeMaxObservation) -> dict[str, Any]:
    """The heads-up sub-game this 3-max decision is shown as."""
    slot, order = _SHOWN_OPPONENT[observation.seat]
    shown = observation.opponent_boards[slot]
    return {
        "hero_board": _rows(observation.hero_board),
        "opp_board": _rows(shown),
        "dealt": list(observation.dealt_cards),
        "dead": list(observation.hero_private_discards),
        "turn": STREETS.index(observation.street),
        "position": order,
        "shown_opponent_seat": observation.opponent_seats[slot],
        "hidden_opponent_seat": observation.opponent_seats[1 - slot],
    }


def decide(observation: ThreeMaxObservation, *, precision: str = "fast") -> Action:
    """The action the frozen heads-up model plays in this sub-game."""
    engine = _engine()
    view = heads_up_view(observation)
    response = engine.decide_with_engine(
        hero_board=view["hero_board"],
        opp_board=view["opp_board"],
        dealt=view["dealt"],
        dead=view["dead"],
        turn=view["turn"],
        position=view["position"],
        precision=precision,
    )
    placements = tuple(
        (card, row) for card, row in response["action"]["placements"]
    )
    placed = {card for card, _row in placements}
    discards = tuple(card for card in observation.dealt_cards if card not in placed)
    action = Action(placements=placements, discards=discards)

    legal = generate_actions(observation.hero_board, observation.dealt_cards)
    if action not in legal:
        raise HeadsUpEngineUnavailable(
            f"heads-up engine returned an action that is not legal here: {action}"
        )
    return action


def hu_policy(*, precision: str = "fast"):
    """A Policy3 backed by the frozen heads-up models."""

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        return decide(observation, precision=precision)

    policy.__name__ = f"hu_policy(precision={precision})"
    return policy
