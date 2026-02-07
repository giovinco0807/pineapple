"""
FL Solver v3.2 - Complete Role-Based Exploration.

Exhaustively enumerates ALL possible hands for each role.
Covers all card combinations to ensure 100% accuracy.
"""

import sys
import os
from typing import List, Tuple, Optional, Dict, Set, Generator
from itertools import combinations
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_solver import Card, evaluate_placement, is_valid_placement


@dataclass
class Placement:
    top: List[Card]
    middle: List[Card]
    bottom: List[Card]
    discards: List[Card] = None
    is_bust: bool = False
    royalties: int = 0
    can_stay: bool = False
    score: float = 0.0


# ============================================================
#  Complete Hand Enumeration - ALL combinations for each role
# ============================================================

def find_all_5card_hands(cards: List[Card]) -> Generator[Tuple[List[Card], List[Card], str], None, None]:
    """
    Yield ALL possible 5-card hands with their type.
    Returns (hand, remaining, hand_type).
    """
    n = len(cards)
    if n < 5:
        return
    
    for indices in combinations(range(n), 5):
        hand = [cards[i] for i in indices]
        remaining = [cards[i] for i in range(n) if i not in indices]
        
        # Determine hand type for prioritization
        hand_type = classify_5card_hand(hand)
        yield (hand, remaining, hand_type)


def find_all_3card_hands(cards: List[Card]) -> Generator[Tuple[List[Card], List[Card], str], None, None]:
    """
    Yield ALL possible 3-card hands with their type.
    Returns (hand, remaining, hand_type).
    """
    n = len(cards)
    if n < 3:
        return
    
    for indices in combinations(range(n), 3):
        hand = [cards[i] for i in indices]
        remaining = [cards[i] for i in range(n) if i not in indices]
        
        hand_type = classify_3card_hand(hand)
        yield (hand, remaining, hand_type)


def classify_5card_hand(cards: List[Card]) -> str:
    """Classify a 5-card hand."""
    jokers = [c for c in cards if c.is_joker]
    non_jokers = [c for c in cards if not c.is_joker]
    
    # Count ranks and suits
    rank_counts = {}
    suit_counts = {}
    for c in non_jokers:
        rank_counts[c.rank_value] = rank_counts.get(c.rank_value, 0) + 1
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    
    num_jokers = len(jokers)
    counts = sorted(rank_counts.values(), reverse=True) if rank_counts else []
    
    # Check hands from highest to lowest
    is_flush = any(cnt + num_jokers >= 5 for cnt in suit_counts.values()) or num_jokers >= 5
    is_straight = check_straight_possible(rank_counts, num_jokers)
    
    if is_flush and is_straight:
        return "StraightFlush"
    if counts and counts[0] + num_jokers >= 4:
        return "Quads"
    if len(counts) >= 2 and counts[0] + counts[1] + num_jokers >= 5 and counts[0] >= 2:
        # Check for full house
        if counts[0] >= 3 or (counts[0] == 2 and num_jokers >= 1):
            return "FullHouse"
    if is_flush:
        return "Flush"
    if is_straight:
        return "Straight"
    if counts and counts[0] + num_jokers >= 3:
        return "Trips"
    if len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
        return "TwoPair"
    if counts and counts[0] + num_jokers >= 2:
        return "OnePair"
    return "HighCard"


def classify_3card_hand(cards: List[Card]) -> str:
    """Classify a 3-card hand."""
    jokers = [c for c in cards if c.is_joker]
    non_jokers = [c for c in cards if not c.is_joker]
    
    rank_counts = {}
    for c in non_jokers:
        rank_counts[c.rank_value] = rank_counts.get(c.rank_value, 0) + 1
    
    num_jokers = len(jokers)
    counts = sorted(rank_counts.values(), reverse=True) if rank_counts else []
    
    if counts and counts[0] + num_jokers >= 3:
        return "Trips"
    if counts and counts[0] + num_jokers >= 2:
        return "Pair"
    return "HighCard"


def check_straight_possible(rank_counts: Dict[int, int], jokers: int) -> bool:
    """Check if a straight is possible with given ranks and jokers."""
    if not rank_counts and jokers >= 5:
        return True
    
    ranks = set(rank_counts.keys())
    
    # Try each straight (A-low to A-high)
    straights = [
        {14, 2, 3, 4, 5},  # A-low (wheel)
        {2, 3, 4, 5, 6},
        {3, 4, 5, 6, 7},
        {4, 5, 6, 7, 8},
        {5, 6, 7, 8, 9},
        {6, 7, 8, 9, 10},
        {7, 8, 9, 10, 11},
        {8, 9, 10, 11, 12},
        {9, 10, 11, 12, 13},
        {10, 11, 12, 13, 14},  # Broadway
    ]
    
    for s in straights:
        missing = len(s - ranks)
        if missing <= jokers:
            return True
    
    return False


# ============================================================
#  Role Priority Order (for early termination)
# ============================================================

BOTTOM_PRIORITY = {
    "StraightFlush": 0,  # FL Stay (15-25 points)
    "Quads": 1,          # FL Stay (10 points)
    "FullHouse": 2,      # 6 points
    "Flush": 3,          # 4 points
    "Straight": 4,       # 2 points
    "Trips": 5,
    "TwoPair": 6,
    "OnePair": 7,
    "HighCard": 8,
}

MIDDLE_PRIORITY = {
    "StraightFlush": 0,
    "Quads": 1,          # 20 points (Middle)
    "FullHouse": 2,      # 12 points
    "Flush": 3,          # 8 points
    "Straight": 4,       # 4 points
    "Trips": 5,          # 2 points
    "TwoPair": 6,
    "OnePair": 7,
    "HighCard": 8,
}

TOP_PRIORITY = {
    "Trips": 0,          # FL Stay (10-22 points)
    "Pair": 1,           # 1-9 points
    "HighCard": 2,
}


# ============================================================
#  Role-Based Solver with Complete Coverage
# ============================================================

def solve_role_based_complete(cards: List[Card], verbose: bool = False) -> Optional[Placement]:
    """
    Solve using complete role-based exploration.
    
    Explores all 5-card combinations for Bottom/Middle,
    and all 3-card combinations for Top.
    """
    best = None
    best_score = -1
    checked = 0
    early_exit = False
    
    n = len(cards)
    
    # Pre-compute all valid 5-card and 3-card combinations
    # to avoid repeated enumeration
    
    # Enumerate all Bottom hands
    for bot_idx in combinations(range(n), 5):
        bottom = [cards[i] for i in bot_idx]
        remaining_bot = [cards[i] for i in range(n) if i not in bot_idx]
        
        if len(remaining_bot) < 8:
            continue
        
        # Enumerate all Middle hands from remaining
        for mid_idx in combinations(range(len(remaining_bot)), 5):
            middle = [remaining_bot[i] for i in mid_idx]
            remaining_mid = [remaining_bot[i] for i in range(len(remaining_bot)) if i not in mid_idx]
            
            if len(remaining_mid) < 3:
                continue
            
            # Enumerate all Top hands from remaining
            for top_idx in combinations(range(len(remaining_mid)), 3):
                top = [remaining_mid[i] for i in top_idx]
                discards = [remaining_mid[i] for i in range(len(remaining_mid)) if i not in top_idx]
                
                # Skip if joker in discards
                if any(c.is_joker for c in discards):
                    continue
                
                checked += 1
                
                # Evaluate
                placement = evaluate_placement(top, middle, bottom)
                if not placement.is_bust and placement.score > best_score:
                    best = placement
                    best_score = placement.score
                    placement.discards = discards
    
    if verbose:
        print(f"Complete role-based: checked {checked} combinations")
    
    return best


def solve_fantasyland_v3(cards: List[Card], parallel: bool = False, verbose: bool = True) -> Optional[Placement]:
    """Main entry point for v3 solver."""
    result = solve_role_based_complete(cards, verbose=verbose)
    return result


# ============================================================
#  Test
# ============================================================

if __name__ == "__main__":
    from fl_solver import deal_fantasyland_hand, solve_fantasyland_exhaustive
    
    print("FL Solver v3.2 (Complete Role-Based) Test")
    print("=" * 50)
    
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        hand = deal_fantasyland_hand(14, include_jokers=False)
        print(f"Hand: {' '.join(str(c) for c in hand[:5])} ...")
        
        # v3.2 Complete
        start = time.time()
        result_v3 = solve_fantasyland_v3(hand)
        t_v3 = time.time() - start
        
        # Exhaustive
        start = time.time()
        result_ex = solve_fantasyland_exhaustive(hand)
        t_ex = time.time() - start
        
        score_v3 = result_v3.score if result_v3 else 0
        score_ex = result_ex[0].score if result_ex else 0
        
        print(f"v3.2: {t_v3:.2f}s, score: {score_v3}")
        print(f"Exhaustive: {t_ex:.2f}s, score: {score_ex}")
        print(f"Match: {'YES' if abs(score_v3 - score_ex) < 0.01 else 'NO'}")
