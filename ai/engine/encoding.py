"""
OFC Pineapple - State Encoding for Neural Networks

Converts game state (Observation) to a 490-dimensional vector:
  - 54 cards × 9 locations = 486 dims
  - 4 meta features = 4 dims
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# Card index mapping
RANKS = "23456789TJQKA"
SUITS = "hdcs"

# All 54 cards in order
ALL_CARDS = [f"{r}{s}" for s in SUITS for r in RANKS] + ["X1", "X2"]
CARD_TO_IDX = {card: i for i, card in enumerate(ALL_CARDS)}


@dataclass
class Board:
    """Player board with top/middle/bottom rows."""
    top: List[str] = field(default_factory=list)
    middle: List[str] = field(default_factory=list)
    bottom: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(
            top=list(d.get("top", [])),
            middle=list(d.get("middle", [])),
            bottom=list(d.get("bottom", [])),
        )

    def to_dict(self) -> dict:
        return {"top": list(self.top), "middle": list(self.middle), "bottom": list(self.bottom)}

    def is_complete(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def card_count(self) -> int:
        return len(self.top) + len(self.middle) + len(self.bottom)

    def all_cards(self) -> List[str]:
        return self.top + self.middle + self.bottom

    def copy(self) -> "Board":
        return Board(top=list(self.top), middle=list(self.middle), bottom=list(self.bottom))


@dataclass
class Observation:
    """Complete game observation for one player at one decision point."""
    board_self: Board
    board_opponent: Board
    dealt_cards: List[str]
    known_discards_self: List[str]
    turn: int
    is_btn: bool
    chips_self: int = 200
    chips_opponent: int = 200

    @property
    def unseen_cards(self) -> List[str]:
        """Cards not visible to this player."""
        seen = set()
        seen.update(self.board_self.all_cards())
        seen.update(self.board_opponent.all_cards())
        seen.update(self.dealt_cards)
        seen.update(self.known_discards_self)
        return [c for c in ALL_CARDS if c not in seen]


def card_to_idx(card: str) -> int:
    """Convert card string to index (0-53)."""
    if card in CARD_TO_IDX:
        return CARD_TO_IDX[card]
    raise ValueError(f"Unknown card: {card}")


# Location indices in the 9-dim one-hot vector
LOC_MY_TOP = 0
LOC_MY_MID = 1
LOC_MY_BOT = 2
LOC_OPP_TOP = 3
LOC_OPP_MID = 4
LOC_OPP_BOT = 5
LOC_IN_HAND = 6
LOC_MY_DISCARD = 7
LOC_UNSEEN = 8

STATE_DIM = 54 * 9 + 4  # 490


def encode_state(obs: Observation) -> np.ndarray:
    """
    Encode game observation as a 490-dimensional float vector.

    Card matrix (54 × 9): one-hot location for each card
    Meta (4): turn, is_btn, chips_self, chips_opponent
    """
    card_matrix = np.zeros((54, 9), dtype=np.float32)

    # Self board
    for card in obs.board_self.top:
        card_matrix[card_to_idx(card)][LOC_MY_TOP] = 1.0
    for card in obs.board_self.middle:
        card_matrix[card_to_idx(card)][LOC_MY_MID] = 1.0
    for card in obs.board_self.bottom:
        card_matrix[card_to_idx(card)][LOC_MY_BOT] = 1.0

    # Opponent board
    for card in obs.board_opponent.top:
        card_matrix[card_to_idx(card)][LOC_OPP_TOP] = 1.0
    for card in obs.board_opponent.middle:
        card_matrix[card_to_idx(card)][LOC_OPP_MID] = 1.0
    for card in obs.board_opponent.bottom:
        card_matrix[card_to_idx(card)][LOC_OPP_BOT] = 1.0

    # Hand
    for card in obs.dealt_cards:
        card_matrix[card_to_idx(card)][LOC_IN_HAND] = 1.0

    # Own discards
    for card in obs.known_discards_self:
        card_matrix[card_to_idx(card)][LOC_MY_DISCARD] = 1.0

    # Unseen
    for card in obs.unseen_cards:
        card_matrix[card_to_idx(card)][LOC_UNSEEN] = 1.0

    # Meta features (normalized)
    meta = np.array([
        obs.turn / 8.0,
        float(obs.is_btn),
        obs.chips_self / 200.0,
        obs.chips_opponent / 200.0,
    ], dtype=np.float32)

    return np.concatenate([card_matrix.flatten(), meta])
