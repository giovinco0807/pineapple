"""
OFC Pineapple - Action Space

Enumerates all valid actions for each turn:
  - Turn 0: 5 cards ↁEdistribute to top/mid/bot (no discard)
  - Turn 1-8: 3 cards ↁEplace 2, discard 1
"""
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .encoding import Board

MAX_ACTIONS = 250  # Upper bound on action count (initial turn can have up to 232)
REGULAR_TURN_ACTIONS = 27  # discard slot (3) x row for card0 (3) x row for card1 (3)
POSITIONS = ["top", "middle", "bottom"]
ROW_LIMITS = {"top": 3, "middle": 5, "bottom": 5}


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
    cards = sorted(dealt_cards)

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

    3 cards ↁEchoose 1 to discard, place remaining 2 in valid positions.
    """
    assert len(dealt_cards) == 3, f"Regular turn expects 3 cards, got {len(dealt_cards)}"

    actions = []

    cards = sorted(dealt_cards)

    for discard_idx in range(3):
        discard = cards[discard_idx]
        remaining = [cards[i] for i in range(3) if i != discard_idx]

        for pos0 in POSITIONS:
            for pos1 in POSITIONS:
                # Check capacity
                counts = {
                    "top": len(board.top),
                    "middle": len(board.middle),
                    "bottom": len(board.bottom),
                }
                counts[pos0] += 1
                if counts[pos0] > ROW_LIMITS[pos0]:
                    continue
                counts[pos1] += 1
                if counts[pos1] > ROW_LIMITS[pos1]:
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


def get_semantic_action_index(action: Action, dealt_cards: List[str]) -> int:
    """
    Map T1-T8 action to a stable semantic index [0, 26].
    dealt_cards: list of 3 strings
    """
    assert action.discard is not None, "Discard must be specified for T1-T8"
    cards = sorted(dealt_cards)
    discard_idx = cards.index(action.discard)
    remaining_cards = [c for i, c in enumerate(cards) if i != discard_idx]
    
    c2p = {c: p for c, p in action.placements}
    pos0_idx = POSITIONS.index(c2p[remaining_cards[0]])
    pos1_idx = POSITIONS.index(c2p[remaining_cards[1]])
    
    return discard_idx * 9 + pos0_idx * 3 + pos1_idx


def get_action_from_semantic_index(index: int, dealt_cards: List[str]) -> Action:
    """
    Recover action from semantic index [0, 26].
    """
    discard_idx = index // 9
    rem = index % 9
    pos0_idx = rem // 3
    pos1_idx = rem % 3
    
    cards = sorted(dealt_cards)
    discard = cards[discard_idx]
    remaining = [c for i, c in enumerate(cards) if i != discard_idx]
    
    pos0 = POSITIONS[pos0_idx]
    pos1 = POSITIONS[pos1_idx]
    
    return Action(
        placements=[(remaining[0], pos0), (remaining[1], pos1)],
        discard=discard
    )


def is_turn_action_valid(action: Action, board: Board) -> bool:
    """Return True if a regular-turn action fits the current board."""
    counts = {
        "top": len(board.top),
        "middle": len(board.middle),
        "bottom": len(board.bottom),
    }
    for _card, row in action.placements:
        if row not in ROW_LIMITS:
            return False
        counts[row] += 1
        if counts[row] > ROW_LIMITS[row]:
            return False
    return True


def get_action_from_semantic_index_if_valid(index: int, dealt_cards: List[str], board: Board) -> Optional[Action]:
    """Recover a regular-turn semantic action only when it is legal for board."""
    if not 0 <= index < REGULAR_TURN_ACTIONS:
        return None
    action = get_action_from_semantic_index(index, dealt_cards)
    return action if is_turn_action_valid(action, board) else None


def create_regular_turn_mask(dealt_cards: List[str], board: Board) -> "np.ndarray":
    """Create a 27-slot semantic mask for turns 1-8."""
    import numpy as np
    mask = np.zeros(REGULAR_TURN_ACTIONS, dtype=bool)
    for idx in range(REGULAR_TURN_ACTIONS):
        action = get_action_from_semantic_index(idx, dealt_cards)
        mask[idx] = is_turn_action_valid(action, board)
    return mask


def create_action_mask(valid_actions: List[Action], turn: int = 0, dealt_cards: Optional[List[str]] = None) -> "np.ndarray":
    """Create boolean mask of shape (MAX_ACTIONS,) for valid actions.
    Uses list index for Turn 0, and semantic index for Turns 1-8.
    """
    import numpy as np
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    if turn == 0:
        for i in range(min(len(valid_actions), MAX_ACTIONS)):
            mask[i] = True
    else:
        assert dealt_cards is not None, "dealt_cards must be provided for T1-8"
        for a in valid_actions:
            idx = get_semantic_action_index(a, dealt_cards)
            mask[idx] = True
    return mask


def encode_action(action: Action, valid_actions: List[Action], turn: int = 0, dealt_cards: Optional[List[str]] = None) -> int:
    """Find the index of an action."""
    if turn > 0 and dealt_cards is not None:
        return get_semantic_action_index(action, dealt_cards)
    for i, a in enumerate(valid_actions):
        if a == action:
            return i
    raise ValueError(f"Action not found in valid actions: {action}")
