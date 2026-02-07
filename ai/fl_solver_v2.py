"""
Fantasyland Solver - Complete Implementation

Based on user's design specification:
- Pattern A: Bottom fixed (Quads+) for FL stay
- Pattern B: Top fixed (Trips) for FL stay  
- Pattern C: Bottom fixed (Full House-) for non-FL stay
- Pattern D: Top fixed (66+ Pair) for non-FL stay
"""

import sys
import os
from typing import List, Tuple, Optional
from itertools import combinations
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.card import Card, Deck, RANKS, SUITS, has_joker
from game.hand_evaluator import (
    evaluate_3_card_hand, evaluate_5_card_hand,
    compare_hands_5, compare_hands_3, HandRank, HandRank3
)
from game.royalty import (
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fantasyland_stay
)
from game.joker_optimizer import (
    generate_substitutions, get_available_cards,
    hand_strength_5, hand_strength_3, optimize_board
)


@dataclass
class Placement:
    top: List[Card]
    middle: List[Card]
    bottom: List[Card]
    discards: List[Card]
    score: float = 0.0
    royalties: int = 0
    can_stay: bool = False
    is_bust: bool = False
    opt_top: Optional[List[Card]] = None
    opt_middle: Optional[List[Card]] = None
    opt_bottom: Optional[List[Card]] = None


def cards_to_str(cards: List[Card]) -> str:
    """Convert cards to readable string."""
    result = []
    for c in cards:
        if c.is_joker:
            result.append("🃏")
        else:
            suits = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
            result.append(f"{c.rank}{suits.get(c.suit, c.suit)}")
    return " ".join(result)


def evaluate_placement(top: List[Card], middle: List[Card], bottom: List[Card]) -> Placement:
    """Evaluate a complete placement with joker optimization."""
    placement = Placement(
        top=top.copy(),
        middle=middle.copy(), 
        bottom=bottom.copy(),
        discards=[]
    )
    
    # Optimize jokers
    opt_top, opt_middle, opt_bottom, is_bust = optimize_board(top, middle, bottom)
    
    if is_bust or opt_top is None:
        placement.is_bust = True
        placement.score = -1000.0
        return placement
    
    placement.opt_top = opt_top
    placement.opt_middle = opt_middle
    placement.opt_bottom = opt_bottom
    
    # Calculate royalties
    top_roy = get_top_royalty(opt_top)
    mid_roy = get_middle_royalty(opt_middle)
    bot_roy = get_bottom_royalty(opt_bottom)
    placement.royalties = top_roy + mid_roy + bot_roy
    
    # Check FL stay
    placement.can_stay = check_fantasyland_stay(opt_top, opt_bottom)
    
    # Score = FL bonus + royalties
    stay_bonus = 50.0 if placement.can_stay else 0.0
    placement.score = stay_bonus + placement.royalties
    
    return placement


# ============================================
# Helper Functions
# ============================================

def get_rank_counts(cards: List[Card]) -> dict:
    """Count cards by rank."""
    counts = {}
    for c in cards:
        if not c.is_joker:
            if c.rank not in counts:
                counts[c.rank] = []
            counts[c.rank].append(c)
    return counts


def find_fl_stay_bottoms(cards: List[Card]) -> List[Tuple[List[Card], HandRank, Tuple]]:
    """
    Find all bottoms that qualify for FL stay:
    - Quads or better (including Royal Flush, Straight Flush)
    
    Returns list of (hand, rank, strength) sorted by strength.
    """
    results = []
    
    for combo in combinations(cards, 5):
        hand = list(combo)
        
        # Evaluate with joker optimization
        if has_joker(hand):
            available = get_available_cards([c for c in hand if not c.is_joker])
            best_strength = None
            best_rank = None
            for sub in generate_substitutions(hand, available):
                strength = hand_strength_5(sub)
                rank, _ = evaluate_5_card_hand(sub)
                if best_strength is None or strength > best_strength:
                    best_strength = strength
                    best_rank = rank
            if best_rank is None:
                continue
            rank = best_rank
        else:
            rank, _ = evaluate_5_card_hand(hand)
            best_strength = hand_strength_5(hand)
        
        # FL stay requires: Quads, Straight Flush, or Royal Flush
        if rank >= HandRank.FOUR_OF_A_KIND:
            results.append((hand, rank, best_strength))
    
    # Sort by strength (Royal Flush > Straight Flush > Quads)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def find_all_trips(cards: List[Card]) -> List[List[Card]]:
    """Find all possible trips (3-of-a-kind) for top row."""
    results = []
    jokers = [c for c in cards if c.is_joker]
    non_jokers = [c for c in cards if not c.is_joker]
    num_jokers = len(jokers)
    
    rank_counts = get_rank_counts(non_jokers)
    
    for rank, rcards in rank_counts.items():
        count = len(rcards)
        jokers_needed = 3 - count
        
        if jokers_needed <= num_jokers and jokers_needed >= 0 and count >= 1:
            trip = rcards[:min(3, count)] + jokers[:jokers_needed]
            if len(trip) == 3:
                rank_value = RANKS.index(rank)
                results.append((trip, rank_value))
    
    # Sort by rank (AAA > KKK > ...)
    results.sort(key=lambda x: x[1], reverse=True)
    return [trip for trip, _ in results]


def find_pairs_66_plus(cards: List[Card]) -> List[Tuple[List[Card], int]]:
    """Find all pairs 66+ for top row."""
    results = []
    jokers = [c for c in cards if c.is_joker]
    non_jokers = [c for c in cards if not c.is_joker]
    num_jokers = len(jokers)
    
    rank_counts = get_rank_counts(non_jokers)
    
    for rank, rcards in rank_counts.items():
        rank_value = RANKS.index(rank)
        if rank_value < RANKS.index('6'):  # Only 66+
            continue
        
        count = len(rcards)
        jokers_needed = 2 - count
        
        if jokers_needed <= num_jokers and jokers_needed >= 0 and count >= 1:
            pair = rcards[:min(2, count)] + jokers[:jokers_needed]
            if len(pair) == 2:
                results.append((pair, rank_value))
    
    # Sort by rank (AA > KK > ...)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_all_bottoms(cards: List[Card], min_rank: HandRank = HandRank.HIGH_CARD) -> List[Tuple[List[Card], HandRank, Tuple]]:
    """Find all possible bottom hands (sorted by strength)."""
    results = []
    
    for combo in combinations(cards, 5):
        hand = list(combo)
        
        # Evaluate with joker optimization
        if has_joker(hand):
            available = get_available_cards([c for c in hand if not c.is_joker])
            best_strength = None
            best_rank = None
            for sub in generate_substitutions(hand, available):
                strength = hand_strength_5(sub)
                rank, _ = evaluate_5_card_hand(sub)
                if best_strength is None or strength > best_strength:
                    best_strength = strength
                    best_rank = rank
            if best_rank is None:
                continue
            rank = best_rank
        else:
            rank, _ = evaluate_5_card_hand(hand)
            best_strength = hand_strength_5(hand)
        
        if rank >= min_rank:
            results.append((hand, rank, best_strength))
    
    # Sort by strength
    results.sort(key=lambda x: x[2], reverse=True)
    return results

def find_flushes(cards: List[Card]) -> List[List[Card]]:
    """Find all possible flush combinations (5+ cards of same suit)."""
    results = []
    
    # Group by suit
    suits = {'s': [], 'h': [], 'd': [], 'c': []}
    jokers = []
    
    for c in cards:
        if c.is_joker:
            jokers.append(c)
        else:
            suits[c.suit].append(c)
    
    # For each suit with 4+ cards (+ jokers can make flush)
    for suit, suit_cards in suits.items():
        total = len(suit_cards) + len(jokers)
        if total >= 5:
            # Generate all 5-card flush combinations
            if len(suit_cards) >= 5:
                for combo in combinations(suit_cards, 5):
                    results.append(list(combo))
            elif len(suit_cards) == 4 and len(jokers) >= 1:
                for j in jokers[:1]:
                    results.append(suit_cards + [j])
            elif len(suit_cards) == 3 and len(jokers) >= 2:
                results.append(suit_cards + jokers[:2])
    
    return results


def find_straights(cards: List[Card]) -> List[List[Card]]:
    """Find all possible straight combinations."""
    results = []
    jokers = [c for c in cards if c.is_joker]
    non_jokers = [c for c in cards if not c.is_joker]
    num_jokers = len(jokers)
    
    # Group by rank
    rank_cards = {r: [] for r in RANKS}
    for c in non_jokers:
        rank_cards[c.rank].append(c)
    
    # Straight sequences (A can be low or high)
    sequences = [
        ['A', '2', '3', '4', '5'],  # Wheel
        ['2', '3', '4', '5', '6'],
        ['3', '4', '5', '6', '7'],
        ['4', '5', '6', '7', '8'],
        ['5', '6', '7', '8', '9'],
        ['6', '7', '8', '9', 'T'],
        ['7', '8', '9', 'T', 'J'],
        ['8', '9', 'T', 'J', 'Q'],
        ['9', 'T', 'J', 'Q', 'K'],
        ['T', 'J', 'Q', 'K', 'A'],  # Broadway
    ]
    
    for seq in sequences:
        missing = 0
        straight_cards = []
        
        for rank in seq:
            if rank_cards[rank]:
                straight_cards.append(rank_cards[rank][0])
            else:
                missing += 1
        
        if missing <= num_jokers:
            # Can complete straight with jokers
            straight = straight_cards + jokers[:missing]
            if len(straight) == 5:
                results.append(straight)
    
    return results


def make_top_with_pair(cards: List[Card], pair: List[Card]) -> List[Card]:
    """Create top with pair + weakest kicker."""
    remaining = [c for c in cards if c not in pair]
    
    # Choose weakest kicker (to leave strong cards for middle/bottom)
    if remaining:
        kicker = min(remaining, key=lambda c: RANKS.index(c.rank) if not c.is_joker else -1)
        return pair + [kicker]
    return pair


def optimize_bottom_middle(remaining: List[Card], top: List[Card]) -> Tuple[Optional[List[Card]], Optional[List[Card]]]:
    """
    Optimize bottom and middle given fixed top.
    Returns (best_bottom, best_middle) that doesn't bust and maximizes royalty.
    
    Tries ALL combinations of bottom and middle to find the best.
    """
    best_score = -1
    best_bottom = None
    best_middle = None
    
    # Get all possible bottoms sorted by strength
    all_bottoms = find_all_bottoms(remaining, min_rank=HandRank.HIGH_CARD)
    
    for bottom, bot_rank, _ in all_bottoms[:100]:  # Limit to top 100 bottoms
        middle_remaining = [c for c in remaining if c not in bottom]
        
        if len(middle_remaining) < 5:
            continue
        
        # Try ALL middle combinations (not just the first one)
        for mid_combo in combinations(middle_remaining, 5):
            middle = list(mid_combo)
            
            # Evaluate
            placement = evaluate_placement(top, middle, bottom)
            
            if placement.is_bust:
                continue
            
            if placement.royalties > best_score:
                best_score = placement.royalties
                best_bottom = bottom
                best_middle = middle
    
    return best_bottom, best_middle


# ============================================
# Main Solver
# ============================================

def solve_fantasyland(hand: List[Card], max_solutions: int = 5) -> List[Placement]:
    """
    Main FL solver implementing all 4 patterns.
    """
    fl_candidates = []
    non_fl_candidates = []
    
    n = len(hand)
    discard_count = n - 13
    
    # ============================================
    # PHASE 1: FL Stay Patterns
    # ============================================
    
    # Pattern A: Bottom fixed (Quads+, Straight Flush, Royal Flush)
    fl_stay_bottoms = find_fl_stay_bottoms(hand)
    for bottom, bot_rank, _ in fl_stay_bottoms[:30]:  # Try more FL stay bottoms
        remaining = [c for c in hand if c not in bottom]
        
        # Find best top (try 66+ pairs - for AA royalty)
        pairs = find_pairs_66_plus(remaining)
        for pair, _ in pairs:
            # Try all possible kickers and middles to find best royalty
            possible_kickers = [c for c in remaining if c not in pair]
            
            for kicker in possible_kickers:
                top = pair + [kicker]
                remaining_for_mid = [c for c in remaining if c not in top]
                
                if len(remaining_for_mid) < 5:
                    continue
                
                # Try ALL middle combinations to find highest royalty
                for mid_combo in combinations(remaining_for_mid, 5):
                    middle = list(mid_combo)
                    discards = [c for c in remaining_for_mid if c not in middle]
                    
                    placement = evaluate_placement(top, middle, bottom)
                    placement.discards = discards
                    
                    if not placement.is_bust and placement.can_stay:
                        fl_candidates.append(placement)
        
        # Also try trips on top
        trips = find_all_trips(remaining)
        for top in trips[:3]:
            remaining2 = [c for c in remaining if c not in top]
            
            for mid_combo in combinations(remaining2, 5):
                middle = list(mid_combo)
                discards = [c for c in remaining2 if c not in middle]
                
                placement = evaluate_placement(top, middle, bottom)
                placement.discards = discards
                
                if not placement.is_bust and placement.can_stay:
                    fl_candidates.append(placement)
        
        # Also try flushes as middle (for high royalty middle)
        flushes_in_remaining = find_flushes(remaining)
        for flush_mid in flushes_in_remaining[:5]:
            top_remaining = [c for c in remaining if c not in flush_mid]
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                placement = evaluate_placement(top, flush_mid, bottom)
                placement.discards = discards
                
                if not placement.is_bust and placement.can_stay:
                    fl_candidates.append(placement)
        
        # Also try straights as middle
        straights_in_remaining = find_straights(remaining)
        for straight_mid in straights_in_remaining[:5]:
            top_remaining = [c for c in remaining if c not in straight_mid]
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                placement = evaluate_placement(top, straight_mid, bottom)
                placement.discards = discards
                
                if not placement.is_bust and placement.can_stay:
                    fl_candidates.append(placement)
    
    # Pattern B: Top fixed (Trips) - high royalty even without FL stay
    all_trips = find_all_trips(hand)
    for top in all_trips[:10]:
        remaining = [c for c in hand if c not in top]
        
        best_bottom, best_middle = optimize_bottom_middle(remaining, top)
        
        if best_bottom and best_middle:
            discards = [c for c in remaining if c not in best_bottom and c not in best_middle]
            
            placement = evaluate_placement(top, best_middle, best_bottom)
            placement.discards = discards
            
            if not placement.is_bust:
                # Add to appropriate list based on FL stay status
                if placement.can_stay:
                    fl_candidates.append(placement)
                else:
                    non_fl_candidates.append(placement)
    
    # ============================================
    # PHASE 2: Non-FL Stay Patterns  
    # ============================================
    
    # Pattern C: Bottom fixed (Full House or less, but at least Pair)
    all_bottoms = find_all_bottoms(hand, min_rank=HandRank.PAIR)
    # Filter out FL-stay bottoms (already handled in Pattern A)
    non_fl_bottoms = [(b, r, s) for b, r, s in all_bottoms if r < HandRank.FOUR_OF_A_KIND]
    
    for bottom, bot_rank, _ in non_fl_bottoms[:30]:
        remaining = [c for c in hand if c not in bottom]
        
        # Find all 66+ pairs in remaining cards
        pairs = find_pairs_66_plus(remaining)
        
        # Try EACH pair (don't break early - need to find best one)
        for pair, _ in pairs:  # Try all pairs, not just first 5
            top = make_top_with_pair(remaining, pair)
            if len(top) < 3:
                continue
            remaining2 = [c for c in remaining if c not in top]
            
            if len(remaining2) < 5:
                continue
            
            # Try to find valid middle (just need one valid, then add to candidates)
            for mid_combo in combinations(remaining2, 5):
                middle = list(mid_combo)
                discards = [c for c in remaining2 if c not in middle]
                
                placement = evaluate_placement(top, middle, bottom)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
                    break  # Found valid middle for this pair, move to next pair
    
    # Pattern D: Top fixed (any pair with royalty) - kicker chosen LAST
    pairs = find_pairs_66_plus(hand)
    for pair, _ in pairs[:10]:
        # Remaining cards after taking the pair (will split into bottom, middle, kicker, discard)
        remaining_after_pair = [c for c in hand if c not in pair]
        
        # Try all possible bottoms
        for bottom_combo in combinations(remaining_after_pair, 5):
            bottom = list(bottom_combo)
            remaining_after_bottom = [c for c in remaining_after_pair if c not in bottom]
            
            if len(remaining_after_bottom) < 5:
                continue
            
            # Try all possible middles
            for middle_combo in combinations(remaining_after_bottom, 5):
                middle = list(middle_combo)
                remaining_for_top = [c for c in remaining_after_bottom if c not in middle]
                
                # remaining_for_top has cards for kicker and discard
                # Choose strongest as kicker (or weakest - up to strategy)
                if len(remaining_for_top) < 1:
                    continue
                
                # Try each remaining card as kicker
                for kicker in remaining_for_top:
                    top = pair + [kicker]
                    discards = [c for c in remaining_for_top if c != kicker]
                    
                    placement = evaluate_placement(top, middle, bottom)
                    placement.discards = discards
                    
                    if not placement.is_bust:
                        non_fl_candidates.append(placement)
                        break  # Found valid kicker, move to next middle
    
    # Pattern F: Flush exploration (bottom or middle flush)
    flushes = find_flushes(hand)
    for flush in flushes[:20]:  # Limit
        remaining = [c for c in hand if c not in flush]
        
        # Try flush as bottom - explore ALL middle combinations
        for mid_combo in combinations(remaining, 5):
            middle = list(mid_combo)
            top_remaining = [c for c in remaining if c not in middle]
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                placement = evaluate_placement(top, middle, flush)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
                    # Don't break - try all top combinations for this middle
            
            # Limit total candidates to avoid memory issues
            if len(non_fl_candidates) > 200:
                break
        
        # Try flush as middle
        for bot_combo in combinations(remaining, 5):
            bottom = list(bot_combo)
            top_remaining = [c for c in remaining if c not in bottom]
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                placement = evaluate_placement(top, flush, bottom)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
            
            if len(non_fl_candidates) > 300:
                break
    
    # Pattern G: Straight exploration  
    straights = find_straights(hand)
    for straight in straights[:10]:  # Limit
        remaining = [c for c in hand if c not in straight]
        
        # Try straight as bottom
        for mid_combo in combinations(remaining, 5):
            middle = list(mid_combo)
            top_remaining = [c for c in remaining if c not in middle]
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                placement = evaluate_placement(top, middle, straight)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
                    break
            
            if len(non_fl_candidates) > 100:  # Limit candidates
                break
    
    # Pattern H: Flush + Straight combination (bottom flush, middle straight)
    for flush in flushes[:10]:
        remaining = [c for c in hand if c not in flush]
        
        # Check if remaining cards can form straight
        straights_in_remaining = find_straights(remaining)
        for straight in straights_in_remaining[:5]:
            top_remaining = [c for c in remaining if c not in straight]
            
            if len(top_remaining) < 3:
                continue
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                # Bottom = flush, Middle = straight
                placement = evaluate_placement(top, straight, flush)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
    
    # Also try Straight bottom + Flush middle
    for straight in straights[:10]:
        remaining = [c for c in hand if c not in straight]
        
        flushes_in_remaining = find_flushes(remaining)
        for flush in flushes_in_remaining[:5]:
            top_remaining = [c for c in remaining if c not in flush]
            
            if len(top_remaining) < 3:
                continue
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                # Bottom = straight, Middle = flush
                placement = evaluate_placement(top, flush, straight)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
    
    # Pattern I: Double Flush (bottom flush + middle flush of different suit)
    for flush1 in flushes[:10]:
        remaining = [c for c in hand if c not in flush1]
        
        # Find another flush in remaining cards
        flushes_in_remaining = find_flushes(remaining)
        for flush2 in flushes_in_remaining[:5]:
            top_remaining = [c for c in remaining if c not in flush2]
            
            if len(top_remaining) < 3:
                continue
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                # Try both orders: flush1 bottom + flush2 middle, and vice versa
                placement1 = evaluate_placement(top, flush2, flush1)
                placement1.discards = discards
                if not placement1.is_bust:
                    non_fl_candidates.append(placement1)
                
                placement2 = evaluate_placement(top, flush1, flush2)
                placement2.discards = discards
                if not placement2.is_bust:
                    non_fl_candidates.append(placement2)
    
    # Pattern J: Full House bottom + any middle (explore all combinations)
    all_bottoms = find_all_bottoms(hand, min_rank=HandRank.FULL_HOUSE)
    full_houses = [(b, r, s) for b, r, s in all_bottoms if r == HandRank.FULL_HOUSE]
    
    for fh_bottom, _, _ in full_houses[:15]:  # Try more Full Houses
        remaining = [c for c in hand if c not in fh_bottom]
        
        # Try ALL middle combinations (not just Straight/Flush)
        for mid_combo in combinations(remaining, 5):
            middle = list(mid_combo)
            top_remaining = [c for c in remaining if c not in middle]
            
            if len(top_remaining) < 3:
                continue
            
            for top_combo in combinations(top_remaining, 3):
                top = list(top_combo)
                discards = [c for c in top_remaining if c not in top]
                
                placement = evaluate_placement(top, middle, fh_bottom)
                placement.discards = discards
                
                if not placement.is_bust:
                    non_fl_candidates.append(placement)
    
    # Pattern E: Fallback - try all combinations to find any valid placement
    if not fl_candidates and not non_fl_candidates:
        # Simple greedy: best bottom first
        all_b = find_all_bottoms(hand, min_rank=HandRank.HIGH_CARD)
        for bottom, _, _ in all_b[:50]:
            remaining = [c for c in hand if c not in bottom]
            
            for mid_combo in combinations(remaining, 5):
                middle = list(mid_combo)
                top_remaining = [c for c in remaining if c not in middle]
                
                for top_combo in combinations(top_remaining, 3):
                    top = list(top_combo)
                    discards = [c for c in top_remaining if c not in top]
                    
                    placement = evaluate_placement(top, middle, bottom)
                    placement.discards = discards
                    
                    if not placement.is_bust:
                        non_fl_candidates.append(placement)
                        if len(non_fl_candidates) >= 10:
                            break
                
                if len(non_fl_candidates) >= 10:
                    break
            
            if len(non_fl_candidates) >= 10:
                break
    
    # ============================================
    # Final Selection
    # ============================================
    
    # Combine all candidates and sort by royalties (not score which includes FL bonus)
    # This ensures we get the highest royalty solution regardless of FL stay
    all_candidates = fl_candidates + non_fl_candidates
    
    if all_candidates:
        # Sort by royalties first (highest), then by FL stay (prefer stay as tiebreaker)
        all_candidates.sort(key=lambda p: (p.royalties, p.can_stay), reverse=True)
        return all_candidates[:max_solutions]
    
    return []


def main():
    """Test the solver."""
    import random
    import time
    
    print("Testing Complete FL Solver")
    print("="*60)
    
    # Test with seed 27394 (the failing case)
    random.seed(27394)
    deck = Deck(include_jokers=True)
    deck.shuffle()
    hand = deck.deal(14)
    
    print(f"Hand: {cards_to_str(hand)}")
    print()
    
    start = time.time()
    solutions = solve_fantasyland(hand)
    elapsed = time.time() - start
    
    print(f"Found {len(solutions)} solutions in {elapsed:.2f}s")
    print()
    
    for i, sol in enumerate(solutions[:3]):
        print(f"--- Solution #{i+1} (Score: {sol.score:.0f}) ---")
        print(f"  Top:    {cards_to_str(sol.top)}")
        print(f"  Middle: {cards_to_str(sol.middle)}")
        print(f"  Bottom: {cards_to_str(sol.bottom)}")
        print(f"  Royalties: {sol.royalties}, FL Stay: {sol.can_stay}")
        print()


if __name__ == "__main__":
    main()
