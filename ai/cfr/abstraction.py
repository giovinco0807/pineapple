"""
OFC Pineapple CFR - State Abstraction

Maps concrete game states to abstract bucket IDs to make CFR tractable.

Abstraction layers:
1. Board Abstraction: Map board → pattern ID based on hand categories + draws
2. Hand Abstraction: Map dealt cards → bucket based on high cards, pairs, suits
3. FL Potential: Encode likelihood of FL entry from current board state

Total InfoSet key = turn | board_bucket | hand_bucket | opp_board_bucket | fl_potential
"""
from collections import Counter
from typing import List, Tuple, Optional
from enum import IntEnum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, RANK_VALUES
from ai.engine.game_engine import (
    evaluate_hand, hand_category, _B,
)


class FLPotential(IntEnum):
    """FL entry potential level."""
    NONE = 0        # No FL cards on top
    POSSIBLE = 1    # Has A/K/Q but no pair yet
    LIKELY = 2      # Has A/K pair started (1 card), needs 1 more
    READY = 3       # QQ+ pair complete, FL guaranteed if not busted


# ─── Board Abstraction ──────────────────────────────────────────

def _row_bucket_3(cards: List[str]) -> str:
    """Bucket a 3-card top row.

    Returns a compact string encoding:
    - empty: 'e'
    - high card: 'h{high_rank}'
    - pair: 'p{pair_rank}'
    - trips: 't{trip_rank}'
    - with slot count
    """
    if not cards:
        return "e3"

    real_cards = [c for c in cards if not c.startswith("X")]
    jokers = len(cards) - len(real_cards)
    slots = 3 - len(cards)

    if not real_cards and jokers > 0:
        # All jokers
        if jokers >= 3:
            return "t14"  # Effective trips aces
        elif jokers >= 2:
            return "p14"  # Effective pair aces
        return f"h14s{slots}"

    ranks = sorted([RANK_VALUES.get(c[0], 0) for c in real_cards], reverse=True)
    rank_counts = Counter(ranks)
    best_count = max(rank_counts.values()) if rank_counts else 0

    if best_count + jokers >= 3:
        trip_rank = max(r for r, c in rank_counts.items() if c + jokers >= 3) if best_count >= 2 else ranks[0]
        return f"t{trip_rank}"
    elif best_count + jokers >= 2:
        pair_rank = max(r for r, c in rank_counts.items() if c >= 2) if best_count >= 2 else ranks[0]
        return f"p{pair_rank}s{slots}"
    else:
        return f"h{ranks[0]}s{slots}"


def _row_bucket_5(cards: List[str], row_name: str = "m") -> str:
    """Bucket a 5-card middle or bottom row.

    Encodes hand category and primary rank:
    - 'sf{rank}': straight flush
    - 'q{rank}': quads
    - 'fh{rank}': full house
    - 'fl{rank}': flush
    - 'st{rank}': straight
    - 'tr{rank}': trips
    - '2p{rank}': two pair
    - '1p{rank}': one pair
    - 'hc{rank}': high card
    - 'd{flush_count}f{straight_count}s{slots}': incomplete (draw potential)
    """
    slots = 5 - len(cards)

    if not cards:
        return f"{row_name}e5"

    if len(cards) == 5:
        val = evaluate_hand(cards, 5)
        cat = hand_category(val)
        r1 = (val // (_B ** 4)) % _B
        cat_names = {8: "sf", 7: "q", 6: "fh", 5: "fl", 4: "st",
                     3: "tr", 2: "2p", 1: "1p", 0: "hc"}
        return f"{row_name}{cat_names.get(cat, 'hc')}{r1}"

    # Incomplete row: encode draw potential
    real_cards = [c for c in cards if not c.startswith("X")]
    jokers = len(cards) - len(real_cards)

    if not real_cards:
        return f"{row_name}e{slots}j{jokers}"

    ranks = sorted([RANK_VALUES.get(c[0], 0) for c in real_cards], reverse=True)
    suits = [c[1] for c in real_cards if len(c) > 1 and not c.startswith("X")]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Flush draw level (0-5)
    max_suited = max(suit_counts.values()) if suit_counts else 0
    flush_level = min(max_suited + jokers, 5)

    # Best grouping
    best_count = max(rank_counts.values()) if rank_counts else 0
    effective = best_count + jokers

    # Pair/trips encoding
    if effective >= 4:
        group = "q"
    elif effective >= 3:
        group = "tr"
    elif effective >= 2:
        group = "1p"
    else:
        group = "hc"

    best_rank = ranks[0] if ranks else 0

    return f"{row_name}{group}{best_rank}f{flush_level}s{slots}"


def abstract_board(board: Board) -> str:
    """
    Create abstract bucket ID for a board state.

    Combines top, middle, bottom row abstractions.
    """
    top = _row_bucket_3(board.top)
    mid = _row_bucket_5(board.middle, "m")
    bot = _row_bucket_5(board.bottom, "b")
    return f"{top}/{mid}/{bot}"


# ─── Hand Abstraction ──────────────────────────────────────────

def abstract_hand_t0(cards: List[str]) -> str:
    """
    Abstract a 5-card initial hand.

    Key features:
    - Pair/trips presence and rank
    - Suit distribution (suited count for flush potential)
    - High card content (A, K, Q for FL potential)
    """
    if not cards:
        return "empty"

    real_cards = [c for c in cards if not c.startswith("X")]
    jokers = len(cards) - len(real_cards)

    ranks = sorted([RANK_VALUES.get(c[0], 0) for c in real_cards], reverse=True)
    suits = [c[1] for c in real_cards if len(c) > 1]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Pair structure
    best_count = max(rank_counts.values()) if rank_counts else 0
    pairs = sorted([r for r, c in rank_counts.items() if c >= 2], reverse=True)
    effective = best_count + jokers

    if effective >= 3:
        pair_code = f"t{ranks[0]}"
    elif len(pairs) >= 2:
        pair_code = f"2p{pairs[0]}.{pairs[1]}"
    elif pairs:
        pair_code = f"p{pairs[0]}"
    elif jokers > 0:
        pair_code = f"j{jokers}h{ranks[0] if ranks else 14}"
    else:
        pair_code = f"h{ranks[0]}"

    # Suit concentration (max suited)
    max_suited = max(suit_counts.values()) if suit_counts else 0
    suit_code = f"s{max_suited + jokers}"

    # FL high cards
    fl_cards = sum(1 for r in ranks if r >= 12)  # Q, K, A
    fl_code = f"f{fl_cards + jokers}"

    return f"{pair_code}/{suit_code}/{fl_code}"


def abstract_hand_t1(cards: List[str], board: Board) -> str:
    """
    Abstract a 3-card pineapple hand (turns 1-4).

    Key features:
    - Connection to existing board (pair completion, flush extension)
    - High card content
    - Rank grouping
    """
    if not cards:
        return "empty"

    real_cards = [c for c in cards if not c.startswith("X")]
    jokers = len(cards) - len(real_cards)

    ranks = sorted([RANK_VALUES.get(c[0], 0) for c in real_cards], reverse=True)
    suits = [c[1] for c in real_cards if len(c) > 1]

    # Check pair within dealt cards
    rank_counts = Counter(ranks)
    has_pair = any(c >= 2 for c in rank_counts.values())

    # Check connection to top row (FL completion)
    top_ranks = set()
    for c in board.top:
        if not c.startswith("X"):
            top_ranks.add(RANK_VALUES.get(c[0], 0))
    connecting = any(r in top_ranks for r in ranks)

    # Connection to mid/bot (suit match for flush, rank match for pairs)
    mid_suits = Counter(c[1] for c in board.middle if len(c) > 1 and not c.startswith("X"))
    bot_suits = Counter(c[1] for c in board.bottom if len(c) > 1 and not c.startswith("X"))
    dealt_suits = Counter(suits)

    mid_flush = max((mid_suits.get(s, 0) + dealt_suits.get(s, 0) for s in dealt_suits), default=0) if dealt_suits else 0
    bot_flush = max((bot_suits.get(s, 0) + dealt_suits.get(s, 0) for s in dealt_suits), default=0) if dealt_suits else 0

    # Encode
    high = ranks[0] if ranks else 14
    pair = "P" if has_pair else "N"
    conn = "C" if connecting else "X"
    fl_m = min(mid_flush, 5)
    fl_b = min(bot_flush, 5)

    return f"{pair}{high}{conn}/m{fl_m}b{fl_b}/j{jokers}"


# ─── FL Potential ──────────────────────────────────────────────

def compute_fl_potential(top_cards: List[str]) -> FLPotential:
    """Compute FL entry potential from current top row."""
    if not top_cards:
        return FLPotential.NONE

    real_cards = [c for c in top_cards if not c.startswith("X")]
    jokers = sum(1 for c in top_cards if c.startswith("X"))

    if not real_cards and jokers == 0:
        return FLPotential.NONE

    ranks = [RANK_VALUES.get(c[0], 0) for c in real_cards]
    rank_counts = Counter(ranks)

    # Check for existing pair QQ+
    for r, count in rank_counts.items():
        if (count + jokers >= 2) and r >= 12:  # Q=12, K=13, A=14
            if count + jokers >= 3:
                return FLPotential.READY  # Trips (FL 17 cards)
            return FLPotential.READY

    # Check for single A/K/Q that could pair
    high_count = sum(1 for r in ranks if r >= 12)
    if high_count > 0 or jokers > 0:
        if len(top_cards) >= 2:
            return FLPotential.LIKELY
        return FLPotential.POSSIBLE

    return FLPotential.NONE


# ─── Combined Abstract Info Set ───────────────────────────────

def abstract_info_set(
    turn: int,
    is_btn: bool,
    my_board: Board,
    opp_board: Board,
    hand_cards: List[str],
    my_discards: List[str],
) -> str:
    """
    Generate abstracted information set key.

    This maps the concrete game state to a compact bucket ID
    that CFR uses to index regrets and strategies.
    """
    # Board abstractions
    my_board_abs = abstract_board(my_board)
    opp_board_abs = abstract_board(opp_board)

    # Hand abstraction
    if turn == 0 and len(hand_cards) == 5:
        hand_abs = abstract_hand_t0(hand_cards)
    elif len(hand_cards) == 3:
        hand_abs = abstract_hand_t1(hand_cards, my_board)
    else:
        hand_abs = "empty"

    # FL potential
    fl_pot = compute_fl_potential(my_board.top)

    # Discard count (not content, for abstraction)
    disc_count = len(my_discards)

    return f"t{turn}|b{int(is_btn)}|{my_board_abs}|{opp_board_abs}|{hand_abs}|fl{fl_pot.value}|dc{disc_count}"
