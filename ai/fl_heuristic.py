"""
Heuristic Fantasyland Solver

Uses strategic patterns to find optimal placements quickly:

Pattern 1: Quads+ on bottom (FL stay)
Pattern 2: Strong trips on top (FL stay) 
Pattern 3: Bottom focus - Full House < Flush < Straight
Pattern 4: Strong pair on top, avoid bust
"""

import sys
import os
from typing import List, Tuple, Optional, Set
from itertools import combinations
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.card import Card, Deck, RANKS, SUITS, has_joker
from game.board import Board, Row
from game.hand_evaluator import (
    evaluate_3_card_hand, evaluate_5_card_hand,
    compare_hands_5, HandRank, HandRank3
)
from game.royalty import (
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fantasyland_stay
)
from game.joker_optimizer import optimize_board

from ai.fl_solver import Placement, cards_to_str, evaluate_placement


def find_quads_plus(cards: List[Card]) -> List[List[Card]]:
    """Find all possible Quads+ (4-of-a-kind or better) hands from cards."""
    results = []
    
    # Count ranks (excluding jokers)
    non_jokers = [c for c in cards if not c.is_joker]
    jokers = [c for c in cards if c.is_joker]
    num_jokers = len(jokers)
    
    rank_cards = {}
    for c in non_jokers:
        if c.rank not in rank_cards:
            rank_cards[c.rank] = []
        rank_cards[c.rank].append(c)
    
    # Find quads (need 4 of same rank, or 3 + joker, or 2 + 2 jokers, etc.)
    for rank, rcards in rank_cards.items():
        count = len(rcards)
        jokers_needed = 4 - count
        
        if jokers_needed <= num_jokers:
            # Can make quads
            quad = rcards[:4] if count >= 4 else rcards + jokers[:jokers_needed]
            # Add kicker
            remaining = [c for c in cards if c not in quad]
            if remaining:
                kicker = max(remaining, key=lambda c: RANKS.index(c.rank) if not c.is_joker else 14)
                results.append(quad + [kicker])
    
    return results


def find_trips(cards: List[Card]) -> List[Tuple[List[Card], int]]:
    """
    Find all possible trips (3-of-a-kind) from cards.
    Returns list of (3 cards, rank_value) sorted by rank.
    """
    results = []
    
    non_jokers = [c for c in cards if not c.is_joker]
    jokers = [c for c in cards if c.is_joker]
    num_jokers = len(jokers)
    
    rank_cards = {}
    for c in non_jokers:
        if c.rank not in rank_cards:
            rank_cards[c.rank] = []
        rank_cards[c.rank].append(c)
    
    # Find trips
    for rank, rcards in rank_cards.items():
        count = len(rcards)
        jokers_needed = 3 - count
        
        if jokers_needed <= num_jokers and count >= 1:
            # Can make trips
            trip = rcards[:3] if count >= 3 else rcards[:count] + jokers[:jokers_needed]
            if len(trip) == 3:
                rank_value = RANKS.index(rank)
                results.append((trip, rank_value))
    
    # Sort by rank (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_pairs(cards: List[Card]) -> List[Tuple[List[Card], int]]:
    """
    Find all possible pairs from cards.
    Returns list of (2 cards + kicker, rank_value) sorted by rank.
    """
    results = []
    
    non_jokers = [c for c in cards if not c.is_joker]
    jokers = [c for c in cards if c.is_joker]
    num_jokers = len(jokers)
    
    rank_cards = {}
    for c in non_jokers:
        if c.rank not in rank_cards:
            rank_cards[c.rank] = []
        rank_cards[c.rank].append(c)
    
    # Find pairs (QQ+ for FL entry bonus)
    for rank, rcards in rank_cards.items():
        rank_value = RANKS.index(rank)
        if rank_value < RANKS.index('Q'):  # Only QQ or higher
            continue
            
        count = len(rcards)
        jokers_needed = 2 - count
        
        if jokers_needed <= num_jokers and count >= 1:
            pair = rcards[:2] if count >= 2 else rcards[:1] + jokers[:jokers_needed]
            if len(pair) == 2:
                # Add best kicker
                remaining = [c for c in cards if c not in pair]
                if remaining:
                    kicker = max(remaining, key=lambda c: RANKS.index(c.rank) if not c.is_joker else 14)
                    results.append((pair + [kicker], rank_value))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_best_5card_hand(cards: List[Card], min_rank: HandRank = HandRank.HIGH_CARD) -> Optional[List[Card]]:
    """Find the best 5-card hand from available cards."""
    if len(cards) < 5:
        return None
    
    best_hand = None
    best_strength = None
    
    for combo in combinations(cards, 5):
        hand = list(combo)
        rank, kickers = evaluate_5_card_hand(hand)
        
        if rank < min_rank:
            continue
            
        strength = (int(rank), tuple(kickers))
        if best_strength is None or strength > best_strength:
            best_strength = strength
            best_hand = hand
    
    return best_hand


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


def fill_remaining(cards: List[Card], top_used: int, mid_used: int, bot_used: int) -> List[Tuple[List[Card], List[Card], List[Card], List[Card]]]:
    """
    Generate valid arrangements filling remaining slots.
    Returns list of (top, middle, bottom, discards) tuples.
    """
    results = []
    
    top_need = 3 - top_used
    mid_need = 5 - mid_used
    bot_need = 5 - bot_used
    total_need = top_need + mid_need + bot_need
    
    if len(cards) < total_need:
        return results
    
    # Try combinations for remaining cards
    for top_cards in combinations(cards, top_need) if top_need > 0 else [()]:
        remaining1 = [c for c in cards if c not in top_cards]
        
        for mid_cards in combinations(remaining1, mid_need) if mid_need > 0 else [()]:
            remaining2 = [c for c in remaining1 if c not in mid_cards]
            
            for bot_cards in combinations(remaining2, bot_need) if bot_need > 0 else [()]:
                discards = [c for c in remaining2 if c not in bot_cards]
                results.append((list(top_cards), list(mid_cards), list(bot_cards), discards))
    
    return results


def solve_pattern1_quads(hand: List[Card]) -> List[Placement]:
    """Pattern 1: Try to make Quads+ on bottom for FL stay."""
    solutions = []
    
    # Find all possible quads
    quads_hands = find_quads_plus(hand)
    
    for bottom in quads_hands:
        remaining = [c for c in hand if c not in bottom]
        
        # Try to fill top and middle
        for top_cards, mid_cards, _, discards in fill_remaining(remaining, 0, 0, 0):
            top = top_cards[:3] if len(top_cards) >= 3 else None
            middle = mid_cards[:5] if len(mid_cards) >= 5 else None
            
            if top and middle:
                placement = try_arrangement(top, middle, bottom, discards)
                if placement and placement.can_stay:
                    solutions.append(placement)
    
    return solutions


def solve_pattern2_trips_top(hand: List[Card]) -> List[Placement]:
    """Pattern 2: Try to place strong trips on top for FL stay."""
    solutions = []
    
    # Find all possible trips
    trips = find_trips(hand)
    
    for trip, rank_value in trips:
        remaining = [c for c in hand if c not in trip]
        
        # Need to find strong bottom (Straight+) that beats middle
        for bot_cards in combinations(remaining, 5):
            bottom = list(bot_cards)
            bot_rank, _ = evaluate_5_card_hand(bottom)
            
            # Bottom needs to be Straight or better for FL stay with trips
            if bot_rank < HandRank.STRAIGHT:
                continue
            
            mid_remaining = [c for c in remaining if c not in bottom]
            
            for mid_cards in combinations(mid_remaining, 5):
                middle = list(mid_cards)
                discards = [c for c in mid_remaining if c not in middle]
                
                placement = try_arrangement(trip, middle, bottom, discards)
                if placement and placement.can_stay:
                    solutions.append(placement)
    
    return solutions


def solve_pattern3_strong_bottom(hand: List[Card]) -> List[Placement]:
    """Pattern 3: Focus on strong bottom (priority: Straight > Flush > Full House)."""
    solutions = []
    
    # Try to find best bottom hands in priority order
    for min_rank in [HandRank.STRAIGHT_FLUSH, HandRank.FOUR_OF_A_KIND, HandRank.FULL_HOUSE, 
                     HandRank.FLUSH, HandRank.STRAIGHT]:
        bottom = find_best_5card_hand(hand, min_rank)
        if not bottom:
            continue
        
        remaining = [c for c in hand if c not in bottom]
        
        # Find best middle that's <= bottom
        for mid_cards in combinations(remaining, 5):
            middle = list(mid_cards)
            
            # Check middle <= bottom
            if compare_hands_5(bottom, middle) < 0:
                continue
            
            top_remaining = [c for c in remaining if c not in middle]
            
            for top_cards in combinations(top_remaining, 3):
                top = list(top_cards)
                discards = [c for c in top_remaining if c not in top]
                
                placement = try_arrangement(top, middle, bottom, discards)
                if placement:
                    solutions.append(placement)
        
        # If we found solutions at this rank, stop
        if solutions:
            break
    
    return solutions


def solve_pattern4_pair_top(hand: List[Card]) -> List[Placement]:
    """Pattern 4: Place strong pair (QQ+) on top, maximize royalties."""
    solutions = []
    
    pairs = find_pairs(hand)
    
    for pair_cards, rank_value in pairs:
        top = pair_cards[:3]  # pair + kicker
        remaining = [c for c in hand if c not in top]
        
        # Find best arrangements for middle and bottom
        for bot_cards in combinations(remaining, 5):
            bottom = list(bot_cards)
            
            mid_remaining = [c for c in remaining if c not in bottom]
            
            for mid_cards in combinations(mid_remaining, 5):
                middle = list(mid_cards)
                discards = [c for c in mid_remaining if c not in middle]
                
                placement = try_arrangement(top, middle, bottom, discards)
                if placement:
                    solutions.append(placement)
    
    return solutions


def solve_fantasyland_heuristic(hand: List[Card], max_solutions: int = 10) -> List[Placement]:
    """
    Solve FL using heuristic patterns.
    
    Priority:
    1. Quads+ on bottom (FL stay)
    2. Trips on top (FL stay)
    3. Strong bottom focus
    4. Strong pair on top
    """
    all_solutions = []
    
    # Pattern 1: Quads+ on bottom
    solutions = solve_pattern1_quads(hand)
    all_solutions.extend(solutions)
    
    # Pattern 2: Trips on top
    solutions = solve_pattern2_trips_top(hand)
    all_solutions.extend(solutions)
    
    # Pattern 3: Strong bottom
    solutions = solve_pattern3_strong_bottom(hand)
    all_solutions.extend(solutions)
    
    # Pattern 4: Strong pair top
    solutions = solve_pattern4_pair_top(hand)
    all_solutions.extend(solutions)
    
    # Remove duplicates and sort by score
    unique = {}
    for p in all_solutions:
        key = tuple(sorted((c.rank, c.suit if not c.is_joker else 'J') for c in p.top + p.middle + p.bottom))
        if key not in unique or p.score > unique[key].score:
            unique[key] = p
    
    solutions = list(unique.values())
    solutions.sort(key=lambda p: p.score, reverse=True)
    
    return solutions[:max_solutions]


def main():
    """Test the heuristic solver."""
    import time
    from ai.fl_solver import deal_fantasyland_hand, print_placement
    
    print("Testing Heuristic FL Solver")
    print("="*60)
    
    for cards in [14, 15, 16, 17]:
        print(f"\n### {cards} cards ###")
        
        for i in range(3):
            hand = deal_fantasyland_hand(cards, include_jokers=True)
            print(f"\nHand {i+1}: {cards_to_str(hand)}")
            
            start = time.time()
            solutions = solve_fantasyland_heuristic(hand)
            elapsed = time.time() - start
            
            print(f"Found {len(solutions)} solutions in {elapsed:.2f}s")
            
            if solutions:
                best = solutions[0]
                print(f"Best: Score={best.score:.0f}, Royalties={best.royalties}, Stay={best.can_stay}")
                print(f"  Top:    {cards_to_str(best.top)}")
                print(f"  Middle: {cards_to_str(best.middle)}")
                print(f"  Bottom: {cards_to_str(best.bottom)}")


if __name__ == "__main__":
    main()
