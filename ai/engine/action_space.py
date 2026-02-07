"""
OFC Pineapple - Action Space

Enumerates all valid actions for each turn:
  - Turn 0: 5 cards → distribute to top/mid/bot (no discard)
  - Turn 1-8: 3 cards → place 2, discard 1
"""
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .encoding import Board

MAX_ACTIONS = 250  # Upper bound on action count (initial turn can have up to 232)


@dataclass
class Action:
    """A single action: card placements + optional discard."""
    placements: List[Tuple[str, str]]  # [(card, position), ...]
    discard: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        # Compare as sets within each position (order doesn't matter)
        return (self._canonical_placements() == other._canonical_placements()
                and self.discard == other.discard)

    def __hash__(self):
        return hash((self._canonical_placements(), self.discard))

    def _canonical_placements(self) -> Tuple:
        """Canonical form: group by position, sort cards within each."""
        by_pos = {"top": [], "middle": [], "bottom": []}
        for card, pos in self.placements:
            by_pos[pos].append(card)
        return tuple(
            (pos, tuple(sorted(cards)))
            for pos, cards in sorted(by_pos.items())
            if cards
        )


def get_initial_actions(dealt_cards: List[str], board: Board) -> List[Action]:
    """
    Enumerate all valid 5-card placement patterns for turn 0.

    5 cards distributed among top (max 3), middle (max 5), bottom (max 5).
    Returns deduplicated list of Actions.
    """
    assert len(dealt_cards) == 5, f"Turn 0 expects 5 cards, got {len(dealt_cards)}"
    cards = dealt_cards

    top_space = 3 - len(board.top)
    mid_space = 5 - len(board.middle)
    bot_space = 5 - len(board.bottom)

    seen = set()
    actions = []

    for top_n in range(min(top_space, 5) + 1):
        for mid_n in range(min(mid_space, 5 - top_n) + 1):
            bot_n = 5 - top_n - mid_n
            if bot_n < 0 or bot_n > bot_space:
                continue

            # Generate all ways to assign cards to positions
            for perm in itertools.permutations(cards):
                top_cards = sorted(perm[:top_n])
                mid_cards = sorted(perm[top_n:top_n + mid_n])
                bot_cards = sorted(perm[top_n + mid_n:])

                key = (tuple(top_cards), tuple(mid_cards), tuple(bot_cards))
                if key in seen:
                    continue
                seen.add(key)

                placements = (
                    [(c, "top") for c in top_cards]
                    + [(c, "middle") for c in mid_cards]
                    + [(c, "bottom") for c in bot_cards]
                )
                actions.append(Action(placements=placements, discard=None))

    return actions


def get_turn_actions(dealt_cards: List[str], board: Board) -> List[Action]:
    """
    Enumerate all valid actions for turns 1-8.

    3 cards → choose 1 to discard, place remaining 2 in valid positions.
    """
    assert len(dealt_cards) == 3, f"Regular turn expects 3 cards, got {len(dealt_cards)}"

    positions = ["top", "middle", "bottom"]
    limits = {"top": 3, "middle": 5, "bottom": 5}

    actions = []

    for discard_idx in range(3):
        discard = dealt_cards[discard_idx]
        remaining = [dealt_cards[i] for i in range(3) if i != discard_idx]

        for pos0 in positions:
            for pos1 in positions:
                # Check capacity
                counts = {
                    "top": len(board.top),
                    "middle": len(board.middle),
                    "bottom": len(board.bottom),
                }
                counts[pos0] += 1
                if counts[pos0] > limits[pos0]:
                    continue
                counts[pos1] += 1
                if counts[pos1] > limits[pos1]:
                    continue

                action = Action(
                    placements=[(remaining[0], pos0), (remaining[1], pos1)],
                    discard=discard,
                )
                actions.append(action)

    # Deduplicate
    seen = set()
    unique = []
    for a in actions:
        h = hash(a)
        if h not in seen:
            seen.add(h)
            unique.append(a)

    return unique


def create_action_mask(valid_actions: List[Action]) -> "np.ndarray":
    """Create boolean mask of shape (MAX_ACTIONS,) for valid actions."""
    import numpy as np
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    for i in range(min(len(valid_actions), MAX_ACTIONS)):
        mask[i] = True
    return mask


def encode_action(action: Action, valid_actions: List[Action]) -> int:
    """Find the index of an action within the valid action list."""
    for i, a in enumerate(valid_actions):
        if a == action:
            return i
    raise ValueError(f"Action not found in valid actions: {action}")
