"""Hidden-discard information model for heads-up play.

Standard Pineapple OFC hides each player's discards from the opponent.
During a hand, a player may condition only on:

- both public boards
- their own current hand
- their own accumulated discards

The opponent's accumulated discards are unknown and must be treated as
live cards when sampling futures from that player's perspective.
"""

from __future__ import annotations

from collections.abc import Iterable


class HuDiscardTracker:
    """Tracks per-player discards so each seat sees only its own."""

    def __init__(self) -> None:
        self._discards: tuple[list[str], list[str]] = ([], [])

    def record(self, player: int, discards: Iterable[str]) -> None:
        self._discards[player].extend(discards)

    def clone(self) -> "HuDiscardTracker":
        copy = HuDiscardTracker()
        copy._discards[0].extend(self._discards[0])
        copy._discards[1].extend(self._discards[1])
        return copy

    def own_discards(self, player: int) -> tuple[str, ...]:
        """Discards visible to ``player``: their own only."""
        return tuple(self._discards[player])

    def all_discards(self) -> tuple[str, ...]:
        """Both players' discards. Oracle-only: never feed into a policy."""
        return (*self._discards[0], *self._discards[1])
