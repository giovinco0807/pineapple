"""Royalty (bonus point) calculation for OFC Pineapple."""
from typing import List
from .card import Card, RANK_VALUES
from .hand_evaluator import (
    evaluate_5_card_hand, evaluate_3_card_hand,
    HandRank, HandRank3
)


# Top hand royalties (3-card hands)
# Pair of 6s through Pair of Aces, and Three of a Kind
TOP_PAIR_ROYALTIES = {
    '6': 1,
    '7': 2,
    '8': 3,
    '9': 4,
    'T': 5,
    'J': 6,
    'Q': 7,
    'K': 8,
    'A': 9,
}

TOP_TRIPS_ROYALTIES = {
    '2': 10,
    '3': 11,
    '4': 12,
    '5': 13,
    '6': 14,
    '7': 15,
    '8': 16,
    '9': 17,
    'T': 18,
    'J': 19,
    'Q': 20,
    'K': 21,
    'A': 22,
}

# Middle hand royalties (5-card hands)
MIDDLE_ROYALTIES = {
    HandRank.THREE_OF_A_KIND: 2,
    HandRank.STRAIGHT: 4,
    HandRank.FLUSH: 8,
    HandRank.FULL_HOUSE: 12,
    HandRank.FOUR_OF_A_KIND: 20,
    HandRank.STRAIGHT_FLUSH: 30,
    HandRank.ROYAL_FLUSH: 50,
}

# Bottom hand royalties (5-card hands)
BOTTOM_ROYALTIES = {
    HandRank.STRAIGHT: 2,
    HandRank.FLUSH: 4,
    HandRank.FULL_HOUSE: 6,
    HandRank.FOUR_OF_A_KIND: 10,
    HandRank.STRAIGHT_FLUSH: 15,
    HandRank.ROYAL_FLUSH: 25,
}


def get_top_royalty(cards: List[Card]) -> int:
    """
    Calculate royalty points for the top hand (3 cards).
    
    Returns:
        Royalty points (0 if no bonus)
    """
    if len(cards) != 3:
        return 0
    
    hand_rank, kickers = evaluate_3_card_hand(cards)
    
    if hand_rank == HandRank3.THREE_OF_A_KIND:
        # Find the rank of the trips
        rank = cards[0].rank  # All cards have same rank for trips
        return TOP_TRIPS_ROYALTIES.get(rank, 10)
    
    if hand_rank == HandRank3.PAIR:
        # Find the pair rank
        from collections import Counter
        rank_counts = Counter(c.rank for c in cards)
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        return TOP_PAIR_ROYALTIES.get(pair_rank, 0)
    
    return 0


def get_middle_royalty(cards: List[Card]) -> int:
    """
    Calculate royalty points for the middle hand (5 cards).
    
    Returns:
        Royalty points (0 if no bonus)
    """
    if len(cards) != 5:
        return 0
    
    hand_rank, _ = evaluate_5_card_hand(cards)
    return MIDDLE_ROYALTIES.get(hand_rank, 0)


def get_bottom_royalty(cards: List[Card]) -> int:
    """
    Calculate royalty points for the bottom hand (5 cards).
    
    Returns:
        Royalty points (0 if no bonus)
    """
    if len(cards) != 5:
        return 0
    
    hand_rank, _ = evaluate_5_card_hand(cards)
    return BOTTOM_ROYALTIES.get(hand_rank, 0)


def get_total_royalties(top: List[Card], middle: List[Card], bottom: List[Card]) -> int:
    """
    Calculate total royalty points for a complete board.
    
    Returns:
        Total royalty points
    """
    return get_top_royalty(top) + get_middle_royalty(middle) + get_bottom_royalty(bottom)


def check_fantasyland_entry(top: List[Card]) -> int:
    """
    Check if a completed top hand qualifies for Fantasyland.
    
    Returns:
        Number of cards to deal in Fantasyland (0 if not qualified):
        - QQ: 14 cards
        - KK: 15 cards
        - AA: 16 cards
        - Trips: 17 cards
    """
    if len(top) != 3:
        return 0
    
    hand_rank, kickers = evaluate_3_card_hand(top)
    
    if hand_rank == HandRank3.THREE_OF_A_KIND:
        return 17
    
    if hand_rank == HandRank3.PAIR:
        from collections import Counter
        rank_counts = Counter(c.rank for c in top)
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        
        if pair_rank == 'Q':
            return 14
        elif pair_rank == 'K':
            return 15
        elif pair_rank == 'A':
            return 16
    
    return 0


def check_fantasyland_stay(top: List[Card], bottom: List[Card]) -> bool:
    """
    Check if a completed board qualifies to stay in Fantasyland.
    
    Conditions:
    - Top has three of a kind, OR
    - Bottom has four of a kind or better
    
    Returns:
        True if qualified to stay in Fantasyland
    """
    if len(top) != 3 or len(bottom) != 5:
        return False
    
    top_rank, _ = evaluate_3_card_hand(top)
    if top_rank == HandRank3.THREE_OF_A_KIND:
        return True
    
    bottom_rank, _ = evaluate_5_card_hand(bottom)
    if bottom_rank >= HandRank.FOUR_OF_A_KIND:
        return True
    
    return False
