"""
Fast Heuristic Fantasyland Solver (Joker-aware)

Uses strategic patterns to find optimal placements quickly:

Pattern 1: Quads+ on bottom (FL stay)
Pattern 2: Trips on top + Straight+ bottom (FL stay)
Pattern 3: Full House on bottom (high royalties)
Pattern 4: Best bottom first, then fill
"""

import sys
import os
from typing import List, Tuple, Optional
from itertools import combinations
from collections import Counter
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.card import Card, Deck, RANKS, SUITS, has_joker
from game.hand_evaluator import (
    evaluate_3_card_hand, evaluate_5_card_hand,
    compare_hands_5, HandRank, HandRank3
)
from game.royalty import (
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fantasyland_stay
)
from game.joker_optimizer import (
    generate_substitutions, get_available_cards,
    hand_strength_5, hand_strength_3
)

from ai.fl_solver import Placement, cards_to_str, evaluate_placement


def eval_5_with_joker(cards: List[Card], available: Optional[List[Card]] = None) -> Tuple[HandRank, Tuple]:
    """Evaluate 5-card hand, optimizing jokers."""
    if has_joker(cards):
        if available is None:
            available = get_available_cards([c for c in cards if not c.is_joker])
        best_strength = None
        best_cards = None
        for sub in generate_substitutions(cards, available):
            strength = hand_strength_5(sub)
            if best_strength is None or strength > best_strength:
                best_strength = strength
                best_cards = sub
        if best_cards:
            rank, kickers = evaluate_5_card_hand(best_cards)
            return rank, tuple(kickers)
    rank, kickers = evaluate_5_card_hand(cards)
    return rank, tuple(kickers)


def eval_3_with_joker(cards: List[Card], available: Optional[List[Card]] = None) -> Tuple[HandRank3, Tuple]:
    """Evaluate 3-card hand, optimizing jokers."""
    if has_joker(cards):
        if available is None:
            available = get_available_cards([c for c in cards if not c.is_joker])
        best_strength = None
        best_cards = None
        for sub in generate_substitutions(cards, available):
            strength = hand_strength_3(sub)
            if best_strength is None or strength > best_strength:
                best_strength = strength
                best_cards = sub
        if best_cards:
            rank, kickers = evaluate_3_card_hand(best_cards)
            return rank, tuple(kickers)
    rank, kickers = evaluate_3_card_hand(cards)
    return rank, tuple(kickers)


def find_all_5card_hands_joker(cards: List[Card]) -> List[Tuple[List[Card], HandRank, Tuple]]:
    """Pre-compute all 5-card combinations with their optimized ranks."""
    results = []
    for combo in combinations(cards, 5):
        hand = list(combo)
        rank, kickers = eval_5_with_joker(hand)
        strength = (int(rank), kickers)
        results.append((hand, rank, strength))
    
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def find_all_3card_hands_joker(cards: List[Card]) -> List[Tuple[List[Card], HandRank3, Tuple]]:
    """Pre-compute all 3-card combinations, sorted by potential royalty then strength."""
    results = []
    for combo in combinations(cards, 3):
        hand = list(combo)
        rank, kickers = eval_3_with_joker(hand)
        strength = (int(rank), kickers)
        # Calculate royalty for sorting (higher royalty = should try first)
        royalty = get_top_royalty(hand)
        results.append((hand, rank, strength, royalty))
    
    # Sort by: 1. Royalty (descending), 2. Hand strength (descending)
    results.sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    # Return without the royalty (to maintain API)
    return [(h, r, s) for h, r, s, _ in results]


def try_arrangement(top: List[Card], middle: List[Card], bottom: List[Card], 
                   discards: List[Card]) -> Optional[Placement]:
    """Try an arrangement and return Placement if valid (not bust)."""
    if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
        return None
    
    placement = evaluate_placement(top, middle, bottom)
    placement.discards = discards
    
    if placement.is_bust:
        return None
    
    return placement


def solve_fl_fast(hand: List[Card], max_solutions: int = 5) -> List[Placement]:
    """
    Fast FL solver using precomputation and joker optimization.
    """
    solutions = []
    
    # Pre-compute all 5-card and 3-card hands with joker optimization
    all_5card = find_all_5card_hands_joker(hand)
    all_3card = find_all_3card_hands_joker(hand)
    
    checked = set()
    
    # Strategy: Try all strong bottoms (Full House+), find valid arrangements
    for bottom, bot_rank, bot_strength in all_5card:
        # Stop at weaker than Full House (for efficiency)
        if bot_rank < HandRank.TWO_PAIR and len(solutions) > 20:
            break
        
        remaining = [c for c in hand if c not in bottom]
        
        # Try each possible middle
        for mid_idx, (mid_pattern, mid_rank, mid_strength) in enumerate(find_all_5card_hands_joker(remaining)):
            # Must be <= bottom
            if mid_strength > bot_strength:
                continue
            
            top_remaining = [c for c in remaining if c not in mid_pattern]
            
            # Try each possible top
            for top_pattern, top_rank, top_strength in find_all_3card_hands_joker(top_remaining):
                discards = [c for c in top_remaining if c not in top_pattern]
                
                # Create key for deduplication
                key = tuple(sorted((c.rank, c.suit if not c.is_joker else 'J') 
                                   for c in top_pattern + mid_pattern + bottom))
                if key in checked:
                    continue
                checked.add(key)
                
                placement = try_arrangement(top_pattern, mid_pattern, bottom, discards)
                if placement:
                    solutions.append(placement)
                
                # Limit iterations
                if len(solutions) > 200:
                    break
            
            if len(solutions) > 200:
                break
        
        if len(solutions) > 200:
            break
    
    # Sort by score
    solutions.sort(key=lambda p: p.score, reverse=True)
    
    return solutions[:max_solutions]


def main():
    """Test the fast heuristic solver."""
    from ai.fl_solver import deal_fantasyland_hand
    
    print("Testing Fast Heuristic FL Solver (Joker-aware)")
    print("="*60)
    
    for cards in [14]:
        print(f"\n### {cards} cards ###")
        
        for i in range(3):
            hand = deal_fantasyland_hand(cards, include_jokers=True)
            print(f"\nHand: {cards_to_str(hand)}")
            
            start = time.time()
            solutions = solve_fl_fast(hand)
            elapsed = time.time() - start
            
            print(f"Found {len(solutions)} solutions in {elapsed:.2f}s")
            
            if solutions:
                best = solutions[0]
                print(f"Best: {best.royalties}pts, Stay={best.can_stay}")


if __name__ == "__main__":
    main()
