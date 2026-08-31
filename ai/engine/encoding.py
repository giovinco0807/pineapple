"""
OFC Pineapple - State Encoding for Neural Networks

Converts game state (Observation) to a 522-dimensional vector:
  - 54 cards × 9 locations = 486 dims
  - 6 meta features = 6 dims (turn, is_btn, is_fl, opp_is_fl, chips)
  - 30 game-aware features:
    - 6 row slots (self/opp remaining slots)
    - 6 FL features (high cards, pair info, fl_ready)
    - 4 opponent FL features
    - 6 hand rank features (self/opp rows)
    - 2 bust risk
    - 6 draw detection (flush/straight/pair potential)
"""
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional


# Card index mapping
RANKS = "23456789TJQKA"
SUITS = "hdcs"
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}  # 2=0, 3=1, ..., A=12

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
    is_fl: bool = False         # Self is in Fantasyland
    opp_is_fl: bool = False     # Opponent is in Fantasyland
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

# 486 (cards) + 6 (meta) + 30 (game-aware) = 522
STATE_DIM = 54 * 9 + 6 + 30


# ─── Helper functions for feature extraction ─────────────────────────

def _card_rank(card_str: str) -> int:
    """Get rank value (0-12) from card string. Returns -1 for jokers."""
    if card_str.startswith("X"):
        return -1
    return RANK_VALUES.get(card_str[0], -1)


def _card_suit(card_str: str) -> str:
    """Get suit from card string. Returns '' for jokers."""
    if card_str.startswith("X"):
        return ''
    return card_str[1] if len(card_str) > 1 else ''


def _row_rank_numeric(cards: List[str], expected_size: int) -> float:
    """
    Compute a normalized hand rank for a row (0.0-1.0), safe for incomplete hands.
    For incomplete hands, returns a rough estimate based on available cards.
    """
    if not cards:
        return 0.0

    real_cards = [c for c in cards if not c.startswith("X")]
    if not real_cards:
        return 0.0

    ranks = sorted([_card_rank(c) for c in real_cards], reverse=True)
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    # Base score from best grouping (0-9 scale)
    if expected_size == 3:
        if counts[0] >= 3:
            score = 8.0  # trips
        elif counts[0] >= 2:
            score = 4.0 + ranks[0] / 12.0  # pair + rank
        else:
            score = max(ranks) / 12.0  # high card
        return min(score / 9.0, 1.0)
    else:  # 5-card row
        suits = [_card_suit(c) for c in real_cards if _card_suit(c)]
        suit_counts = Counter(suits)
        max_suited = max(suit_counts.values()) if suit_counts else 0

        if counts[0] >= 4:
            score = 7.0  # quads
        elif counts[0] >= 3 and len(counts) > 1 and counts[1] >= 2:
            score = 6.0  # full house
        elif len(real_cards) >= 4 and max_suited >= 4:
            score = 5.5  # flush draw / flush
        elif counts[0] >= 3:
            score = 3.5  # trips
        elif counts[0] >= 2 and len(counts) > 1 and counts[1] >= 2:
            score = 2.5  # two pair
        elif counts[0] >= 2:
            score = 1.5  # pair
        else:
            score = max(ranks) / 12.0 if ranks else 0.0  # high card
        return min(score / 9.0, 1.0)


def _fl_features(top_cards: List[str]) -> List[float]:
    """
    Compute FL-related features from top row (6 dims):
    [has_Q, has_K, has_A, has_pair, pair_rank/14, fl_ready]
    
    Jokers (X1/X2) are treated as wild cards that can form pairs.
    """
    if not top_cards:
        return [0.0] * 6

    ranks = [_card_rank(c) for c in top_cards if _card_rank(c) >= 0]
    n_jokers = sum(1 for c in top_cards if c.startswith('X'))
    if not ranks and n_jokers == 0:
        return [0.0] * 6

    has_Q = float(RANK_VALUES['Q'] in ranks)
    has_K = float(RANK_VALUES['K'] in ranks)
    has_A = float(RANK_VALUES['A'] in ranks)

    rank_counts = Counter(ranks)
    
    # With jokers, each joker can pair with the highest unpaired card
    effective_counts = dict(rank_counts)
    jokers_left = n_jokers
    # First, boost existing cards (highest first) to form pairs
    for r in sorted(effective_counts.keys(), reverse=True):
        if jokers_left <= 0:
            break
        if effective_counts[r] == 1:  # single card, joker makes a pair
            effective_counts[r] = 2
            jokers_left -= 1
    # If jokers still left and no real cards, treat as high (A)
    if jokers_left > 0 and not ranks:
        effective_counts[RANK_VALUES['A']] = min(jokers_left, 2)
        has_A = 1.0
    elif jokers_left > 0 and ranks:
        # Extra jokers boost highest card further
        best_r = max(effective_counts.keys())
        effective_counts[best_r] += jokers_left

    pairs = [(r, c) for r, c in effective_counts.items() if c >= 2]
    has_pair = float(len(pairs) > 0)
    pair_rank = max(r for r, _ in pairs) / 14.0 if pairs else 0.0

    # FL ready: pair of QQ+ in top (including joker-assisted)
    fl_ready = 0.0
    if pairs:
        best_pair = max(r for r, _ in pairs)
        if best_pair >= RANK_VALUES['Q']:  # QQ, KK, AA
            fl_ready = 1.0
    # Also trips QQ+ counts
    trips = [r for r, c in effective_counts.items() if c >= 3]
    if trips and max(trips) >= RANK_VALUES['Q']:
        fl_ready = 1.0

    return [has_Q, has_K, has_A, has_pair, pair_rank, fl_ready]


def _draw_features(mid_cards: List[str], bot_cards: List[str]) -> List[float]:
    """
    Compute draw detection features (6 dims):
    [flush_draw_mid, flush_draw_bot, straight_potential_mid, 
     straight_potential_bot, pair_count_mid, pair_count_bot]
    """
    features = []
    for cards in [mid_cards, bot_cards]:
        real_cards = [c for c in cards if not c.startswith("X")]
        if not real_cards:
            features.extend([0.0, 0.0, 0.0])
            continue

        # Flush draw: max suited count / 5
        suits = [_card_suit(c) for c in real_cards if _card_suit(c)]
        suit_counts = Counter(suits)
        flush_draw = max(suit_counts.values()) / 5.0 if suit_counts else 0.0

        # Straight potential: count consecutive ranks / 5
        ranks = sorted(set(_card_rank(c) for c in real_cards if _card_rank(c) >= 0))
        if len(ranks) >= 2:
            max_consecutive = 1
            current = 1
            for i in range(1, len(ranks)):
                if ranks[i] == ranks[i-1] + 1:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 1
            straight_pot = max_consecutive / 5.0
        else:
            straight_pot = 0.0

        # Pair count
        rank_counts = Counter(_card_rank(c) for c in real_cards if _card_rank(c) >= 0)
        pair_count = sum(1 for c in rank_counts.values() if c >= 2) / 3.0

        features.extend([flush_draw, straight_pot, pair_count])

    return features


def encode_state(obs: Observation, prob_features: np.ndarray = None) -> np.ndarray:
    """
    Encode game observation as a float vector.

    Base (522 dims):
      Card matrix (54 × 9): one-hot location for each card = 486
      Meta (6): turn, is_btn, is_fl, opp_is_fl, chips_self, chips_opponent
      Game-aware (30): row slots, FL features, hand ranks, draws, bust risk

    With prob_features (822 dims):
      Base 522 + prob_engine histogram 300 (3 rows × 100 fine bins)
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

    # Meta features (normalized) — 6 dims
    meta = np.array([
        obs.turn / 4.0,
        float(obs.is_btn),
        float(obs.is_fl),
        float(obs.opp_is_fl),
        obs.chips_self / 200.0,
        obs.chips_opponent / 200.0,
    ], dtype=np.float32)

    # ─── Game-aware features (30 dims) ────────────────────────────

    bs = obs.board_self
    bo = obs.board_opponent

    # Row slots remaining (6 dims)
    row_slots = np.array([
        (3 - len(bs.top)) / 3.0,
        (5 - len(bs.middle)) / 5.0,
        (5 - len(bs.bottom)) / 5.0,
        (3 - len(bo.top)) / 3.0,
        (5 - len(bo.middle)) / 5.0,
        (5 - len(bo.bottom)) / 5.0,
    ], dtype=np.float32)

    # FL features - self (6 dims)
    fl_self = np.array(_fl_features(bs.top), dtype=np.float32)

    # FL features - opponent (4 dims)
    opp_fl = _fl_features(bo.top)
    fl_opp = np.array([opp_fl[0], opp_fl[1], opp_fl[2], opp_fl[5]],
                       dtype=np.float32)  # has_Q, has_K, has_A, fl_ready

    # Hand rank features (6 dims)
    hand_ranks = np.array([
        _row_rank_numeric(bs.top, 3),
        _row_rank_numeric(bs.middle, 5),
        _row_rank_numeric(bs.bottom, 5),
        _row_rank_numeric(bo.top, 3),
        _row_rank_numeric(bo.middle, 5),
        _row_rank_numeric(bo.bottom, 5),
    ], dtype=np.float32)

    # Bust risk (2 dims)
    bust_risk = np.array([
        1.0 if (hand_ranks[0] > hand_ranks[1] and len(bs.top) > 0
                and len(bs.middle) > 0) else 0.0,
        1.0 if (hand_ranks[3] > hand_ranks[4] and len(bo.top) > 0
                and len(bo.middle) > 0) else 0.0,
    ], dtype=np.float32)

    # Draw detection (6 dims)
    draws = np.array(
        _draw_features(bs.middle, bs.bottom),
        dtype=np.float32)

    # ─── Concatenate all ──────────────────────────────────────────
    game_features = np.concatenate([
        row_slots, fl_self, fl_opp, hand_ranks, bust_risk, draws
    ])  # 6 + 6 + 4 + 6 + 2 + 6 = 30

    base = np.concatenate([card_matrix.flatten(), meta, game_features])
    if prob_features is not None:
        return np.concatenate([base, np.asarray(prob_features, dtype=np.float32)])
    return base

