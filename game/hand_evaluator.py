"""Hand evaluation for OFC Pineapple poker."""
from collections import Counter
from enum import IntEnum
from typing import List, Tuple, Optional
from .card import Card, RANK_VALUES


class HandRank(IntEnum):
    """Poker hand rankings."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class HandRank3(IntEnum):
    """3-card hand rankings for top row."""
    HIGH_CARD = 0
    PAIR = 1
    THREE_OF_A_KIND = 2


def evaluate_5_card_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
    """
    Evaluate a 5-card poker hand.
    
    Returns:
        Tuple of (HandRank, kickers) where kickers is a list of rank values
        for tiebreaking, ordered from most significant to least.
    """
    if len(cards) != 5:
        raise ValueError(f"Expected 5 cards, got {len(cards)}")
    
    ranks = sorted([c.rank_value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(c.rank_value for c in cards)
    
    # Check for flush
    is_flush = len(set(suits)) == 1
    
    # Check for straight
    is_straight, straight_high = _check_straight(ranks)
    
    # Get counts for pairs, trips, etc.
    counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Royal Flush
    if is_flush and is_straight and straight_high == RANK_VALUES['A']:
        return (HandRank.ROYAL_FLUSH, [straight_high])
    
    # Straight Flush
    if is_flush and is_straight:
        return (HandRank.STRAIGHT_FLUSH, [straight_high])
    
    # Four of a Kind
    if counts[0][1] == 4:
        quad_rank = counts[0][0]
        kicker = counts[1][0]
        return (HandRank.FOUR_OF_A_KIND, [quad_rank, kicker])
    
    # Full House
    if counts[0][1] == 3 and counts[1][1] == 2:
        trips_rank = counts[0][0]
        pair_rank = counts[1][0]
        return (HandRank.FULL_HOUSE, [trips_rank, pair_rank])
    
    # Flush
    if is_flush:
        return (HandRank.FLUSH, ranks)
    
    # Straight
    if is_straight:
        return (HandRank.STRAIGHT, [straight_high])
    
    # Three of a Kind
    if counts[0][1] == 3:
        trips_rank = counts[0][0]
        kickers = sorted([r for r, c in counts if c == 1], reverse=True)
        return (HandRank.THREE_OF_A_KIND, [trips_rank] + kickers)
    
    # Two Pair
    if counts[0][1] == 2 and counts[1][1] == 2:
        high_pair = max(counts[0][0], counts[1][0])
        low_pair = min(counts[0][0], counts[1][0])
        kicker = counts[2][0]
        return (HandRank.TWO_PAIR, [high_pair, low_pair, kicker])
    
    # Pair
    if counts[0][1] == 2:
        pair_rank = counts[0][0]
        kickers = sorted([r for r, c in counts if c == 1], reverse=True)
        return (HandRank.PAIR, [pair_rank] + kickers)
    
    # High Card
    return (HandRank.HIGH_CARD, ranks)


def evaluate_3_card_hand(cards: List[Card]) -> Tuple[HandRank3, List[int]]:
    """
    Evaluate a 3-card hand for the top row.
    
    Returns:
        Tuple of (HandRank3, kickers) where kickers is a list of rank values
        for tiebreaking, ordered from most significant to least.
    """
    if len(cards) != 3:
        raise ValueError(f"Expected 3 cards, got {len(cards)}")
    
    ranks = sorted([c.rank_value for c in cards], reverse=True)
    rank_counts = Counter(c.rank_value for c in cards)
    counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Three of a Kind
    if counts[0][1] == 3:
        return (HandRank3.THREE_OF_A_KIND, [counts[0][0]])
    
    # Pair
    if counts[0][1] == 2:
        pair_rank = counts[0][0]
        kicker = counts[1][0]
        return (HandRank3.PAIR, [pair_rank, kicker])
    
    # High Card
    return (HandRank3.HIGH_CARD, ranks)


def _check_straight(sorted_ranks: List[int]) -> Tuple[bool, int]:
    """
    Check if sorted ranks form a straight.
    
    Returns:
        Tuple of (is_straight, high_card_rank)
    """
    unique_ranks = sorted(set(sorted_ranks), reverse=True)
    if len(unique_ranks) != 5:
        return (False, 0)
    
    # Check for regular straight
    if unique_ranks[0] - unique_ranks[4] == 4:
        return (True, unique_ranks[0])
    
    # Check for A-2-3-4-5 (wheel)
    if unique_ranks == [RANK_VALUES['A'], RANK_VALUES['5'], RANK_VALUES['4'], 
                        RANK_VALUES['3'], RANK_VALUES['2']]:
        return (True, RANK_VALUES['5'])  # 5-high straight
    
    return (False, 0)


def compare_hands_5(hand1: List[Card], hand2: List[Card]) -> int:
    """
    Compare two 5-card hands.
    
    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    rank1, kickers1 = evaluate_5_card_hand(hand1)
    rank2, kickers2 = evaluate_5_card_hand(hand2)
    
    if rank1 > rank2:
        return 1
    if rank1 < rank2:
        return -1
    
    # Same rank, compare kickers
    for k1, k2 in zip(kickers1, kickers2):
        if k1 > k2:
            return 1
        if k1 < k2:
            return -1
    
    return 0


def compare_hands_3(hand1: List[Card], hand2: List[Card]) -> int:
    """
    Compare two 3-card hands.
    
    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    rank1, kickers1 = evaluate_3_card_hand(hand1)
    rank2, kickers2 = evaluate_3_card_hand(hand2)
    
    if rank1 > rank2:
        return 1
    if rank1 < rank2:
        return -1
    
    # Same rank, compare kickers
    for k1, k2 in zip(kickers1, kickers2):
        if k1 > k2:
            return 1
        if k1 < k2:
            return -1
    
    return 0


def hand_rank_name(hand_rank: HandRank) -> str:
    """Get human-readable name for hand rank."""
    names = {
        HandRank.HIGH_CARD: "High Card",
        HandRank.PAIR: "Pair",
        HandRank.TWO_PAIR: "Two Pair",
        HandRank.THREE_OF_A_KIND: "Three of a Kind",
        HandRank.STRAIGHT: "Straight",
        HandRank.FLUSH: "Flush",
        HandRank.FULL_HOUSE: "Full House",
        HandRank.FOUR_OF_A_KIND: "Four of a Kind",
        HandRank.STRAIGHT_FLUSH: "Straight Flush",
        HandRank.ROYAL_FLUSH: "Royal Flush"
    }
    return names.get(hand_rank, "Unknown")


def hand_rank_3_name(hand_rank: HandRank3) -> str:
    """Get human-readable name for 3-card hand rank."""
    names = {
        HandRank3.HIGH_CARD: "High Card",
        HandRank3.PAIR: "Pair",
        HandRank3.THREE_OF_A_KIND: "Three of a Kind"
    }
    return names.get(hand_rank, "Unknown")
