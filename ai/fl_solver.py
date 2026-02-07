"""
Fantasyland Solver - Beam Search / Exhaustive Search

Finds optimal card placement for Fantasyland hands.
"""

import sys
import os
from typing import List, Tuple, Optional, Set
from itertools import combinations, permutations
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.card import Card, Deck, RANKS, SUITS
from game.board import Board, Row
from game.hand_evaluator import (
    evaluate_3_card_hand, evaluate_5_card_hand,
    compare_hands_3, compare_hands_5,
    HandRank, HandRank3
)
from game.royalty import (
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fantasyland_stay
)


@dataclass
class Placement:
    """Represents a complete FL placement."""
    top: List[Card]
    middle: List[Card]
    bottom: List[Card]
    discards: List[Card]
    
    is_bust: bool = False
    royalties: int = 0
    can_stay: bool = False
    score: float = 0.0  # Combined evaluation score
    
    # Optimized hands (with jokers replaced)
    opt_top: List[Card] = None
    opt_middle: List[Card] = None
    opt_bottom: List[Card] = None
    
    def __repr__(self):
        return f"Placement(royalties={self.royalties}, stay={self.can_stay}, bust={self.is_bust})"


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


def is_valid_placement_with_jokers(top: List[Card], middle: List[Card], bottom: List[Card]) -> Tuple[bool, List[Card], List[Card], List[Card]]:
    """
    Check if placement is valid, handling jokers.
    
    Returns:
        (is_valid, optimized_top, optimized_middle, optimized_bottom)
    """
    from game.card import has_joker
    from game.joker_optimizer import optimize_board
    
    # If no jokers, use simple validation
    all_cards = top + middle + bottom
    if not has_joker(all_cards):
        is_valid = is_valid_placement_simple(top, middle, bottom)
        return (is_valid, top, middle, bottom)
    
    # Use joker optimizer
    opt_top, opt_middle, opt_bottom, is_bust = optimize_board(top, middle, bottom)
    
    if is_bust or opt_top is None:
        return (False, top, middle, bottom)
    
    return (True, opt_top, opt_middle, opt_bottom)


def is_valid_placement_simple(top: List[Card], middle: List[Card], bottom: List[Card]) -> bool:
    """
    Check if placement is valid (not bust) - no jokers.
    Top <= Middle <= Bottom in strength.
    """
    if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
        return False
    
    # Evaluate hands
    top_rank, top_kickers = evaluate_3_card_hand(top)
    mid_rank, mid_kickers = evaluate_5_card_hand(middle)
    bot_rank, bot_kickers = evaluate_5_card_hand(bottom)
    
    # Bottom must be >= Middle
    if compare_hands_5(bottom, middle) < 0:
        return False
    
    # Middle must be >= Top (need to compare 3-card to 5-card)
    # Special case: 3-card Trips vs 5-card Trips - compare trips ranks
    if top_rank == HandRank3.THREE_OF_A_KIND and mid_rank == HandRank.THREE_OF_A_KIND:
        # Both are trips - compare the trips rank (first element in kickers)
        top_trips_rank = top_kickers[0] if top_kickers else 0
        mid_trips_rank = mid_kickers[0] if mid_kickers else 0
        if top_trips_rank > mid_trips_rank:
            return False  # Top trips > Middle trips = bust
        # If same trips rank, it's valid (can't compare kickers for 3 vs 5)
        return True
    
    top_strength = _get_3card_strength(top_rank, top_kickers)
    mid_strength = _get_5card_strength(mid_rank, mid_kickers)
    
    if top_strength > mid_strength:
        return False
    
    return True


def is_valid_placement(top: List[Card], middle: List[Card], bottom: List[Card]) -> bool:
    """Check if placement is valid (wrapper that handles jokers)."""
    is_valid, _, _, _ = is_valid_placement_with_jokers(top, middle, bottom)
    return is_valid


def _get_3card_strength(rank: HandRank3, kickers: List[int]) -> Tuple:
    """
    Convert 3-card hand to comparable tuple.
    
    OFC comparison rules for Top vs Middle:
    - 3-card High Card < 5-card Pair
    - 3-card Pair < 5-card Two Pair  
    - 3-card Trips > 5-card Two Pair (but < 5-card Trips)
    """
    # Map 3-card ranks to be comparable with 5-card
    # HIGH_CARD=0, PAIR=1, THREE_OF_A_KIND=2
    if rank == HandRank3.THREE_OF_A_KIND:
        # 3-card Trips beats Two Pair but loses to 5-card Trips
        # Place it between Two Pair (2) and Three of a Kind (3)
        return (2.5, tuple(kickers))
    return (int(rank), tuple(kickers))


def _get_5card_strength(rank: HandRank, kickers: List[int]) -> Tuple:
    """
    Convert 5-card hand to comparable tuple.
    
    Must be comparable with 3-card hands.
    """
    # HandRank: HIGH_CARD=0, PAIR=1, TWO_PAIR=2, THREE_OF_A_KIND=3, ...
    return (int(rank), tuple(kickers))


def evaluate_placement(top: List[Card], middle: List[Card], bottom: List[Card]) -> Placement:
    """Evaluate a complete placement, handling jokers."""
    placement = Placement(
        top=top.copy(),
        middle=middle.copy(),
        bottom=bottom.copy(),
        discards=[]
    )
    
    # Check validity and get optimized hands (for jokers)
    is_valid, opt_top, opt_middle, opt_bottom = is_valid_placement_with_jokers(top, middle, bottom)
    placement.is_bust = not is_valid
    
    if placement.is_bust:
        placement.score = -1000.0
        return placement
    
    # Calculate royalties using optimized hands
    top_roy = get_top_royalty(opt_top)
    mid_roy = get_middle_royalty(opt_middle)
    bot_roy = get_bottom_royalty(opt_bottom)
    placement.royalties = top_roy + mid_roy + bot_roy
    
    # Check FL stay using optimized hands
    placement.can_stay = check_fantasyland_stay(opt_top, opt_bottom)
    
    # Store optimized hands for display
    placement.opt_top = opt_top
    placement.opt_middle = opt_middle
    placement.opt_bottom = opt_bottom
    
    # Calculate score
    # Priorities: 1. Stay (huge bonus), 2. Royalties
    stay_bonus = 50.0 if placement.can_stay else 0.0
    placement.score = stay_bonus + placement.royalties
    
    return placement


def solve_fantasyland_exhaustive(hand: List[Card], max_solutions: int = 10, enable_pruning: bool = True) -> List[Placement]:
    """
    Find best placements using exhaustive search with pruning.
    
    For 14 cards: C(14,3) * C(11,5) * C(6,5) = ~1M combinations
    For 17 cards: C(17,3) * C(14,5) * C(9,5) = ~8.5M combinations
    
    Pruning strategies:
    1. Early bust detection: Skip Top selections that can't form valid placements
    2. Best-so-far pruning: Skip branches with no chance of beating current best
    """
    cards = hand.copy()
    n = len(cards)
    
    if n < 13:
        raise ValueError(f"Need at least 13 cards, got {n}")
    
    solutions = []
    checked = 0
    pruned = 0
    best_score = float('-inf')
    
    # Generate all possible placements
    for top_indices in combinations(range(n), 3):
        top = [cards[i] for i in top_indices]
        remaining_after_top = [cards[i] for i in range(n) if i not in top_indices]
        
        # PRUNING 1: Early bust detection
        # Check if any valid Middle can beat this Top
        if enable_pruning and not _can_top_have_valid_middle(top, remaining_after_top):
            pruned += _count_remaining_combinations(len(remaining_after_top))
            continue
        
        # Get Top strength for comparisons
        top_rank, top_kickers = evaluate_3_card_hand(top)
        top_strength = _get_3card_strength(top_rank, top_kickers)
        
        for mid_indices in combinations(range(len(remaining_after_top)), 5):
            middle = [remaining_after_top[i] for i in mid_indices]
            remaining_after_mid = [remaining_after_top[i] for i in range(len(remaining_after_top)) if i not in mid_indices]
            
            # PRUNING 2: Check Middle >= Top
            mid_rank, mid_kickers = evaluate_5_card_hand(middle)
            mid_strength = _get_5card_strength(mid_rank, mid_kickers)
            
            if enable_pruning and top_strength > mid_strength:
                pruned += _count_bottom_combinations(len(remaining_after_mid))
                continue
            
            for bot_indices in combinations(range(len(remaining_after_mid)), 5):
                bottom = [remaining_after_mid[i] for i in bot_indices]
                discards = [remaining_after_mid[i] for i in range(len(remaining_after_mid)) if i not in bot_indices]
                
                checked += 1
                
                # PRUNING 3: Check Bottom >= Middle
                if enable_pruning and compare_hands_5(bottom, middle) < 0:
                    continue
                
                # PRUNING 4: Never discard jokers
                if enable_pruning and any(c.is_joker for c in discards):
                    continue
                
                placement = evaluate_placement(top, middle, bottom)
                placement.discards = discards
                
                if not placement.is_bust:
                    solutions.append(placement)
                    if placement.score > best_score:
                        best_score = placement.score
    
    # Sort by score (descending)
    solutions.sort(key=lambda p: p.score, reverse=True)
    
    print(f"Checked {checked:,} combinations, found {len(solutions)} valid placements (pruned {pruned:,})")
    
    return solutions[:max_solutions]


def _can_top_have_valid_middle(top: List[Card], remaining: List[Card]) -> bool:
    """Check if any 5-card combination from remaining can beat this Top."""
    top_rank, top_kickers = evaluate_3_card_hand(top)
    top_strength = _get_3card_strength(top_rank, top_kickers)
    
    # Quick check: if Top is just high card, almost any Middle will beat it
    if top_rank == HandRank3.HIGH_CARD:
        return True
    
    # Check ALL Middle combinations (C(11,5) = 462 for 14 cards, fast enough)
    for mid_indices in combinations(range(len(remaining)), 5):
        middle = [remaining[i] for i in mid_indices]
        mid_rank, mid_kickers = evaluate_5_card_hand(middle)
        mid_strength = _get_5card_strength(mid_rank, mid_kickers)
        
        if mid_strength >= top_strength:
            return True
    
    return False


def _count_remaining_combinations(remaining_count: int) -> int:
    """Count Middle*Bottom combinations for pruning statistics."""
    # C(remaining, 5) * C(remaining-5, 5)
    from math import comb
    return comb(remaining_count, 5) * comb(remaining_count - 5, 5)


def _count_bottom_combinations(remaining_count: int) -> int:
    """Count Bottom combinations for pruning statistics."""
    from math import comb
    return comb(remaining_count, 5)


def solve_fantasyland_beam(hand: List[Card], beam_width: int = 1000) -> List[Placement]:
    """
    Beam search for faster (but possibly suboptimal) solving.
    
    Strategy: Build placements incrementally, keeping only top beam_width candidates.
    """
    # For FL, exhaustive is often fast enough (< 1s for 14 cards)
    # But for 17 cards, beam search is useful
    
    # Simple approach: sample combinations and evaluate
    import random
    
    cards = hand.copy()
    n = len(cards)
    
    solutions = []
    
    for _ in range(beam_width * 10):
        # Random placement
        shuffled = cards.copy()
        random.shuffle(shuffled)
        
        top = shuffled[:3]
        middle = shuffled[3:8]
        bottom = shuffled[8:13]
        discards = shuffled[13:]
        
        placement = evaluate_placement(top, middle, bottom)
        placement.discards = discards
        
        if not placement.is_bust:
            solutions.append(placement)
    
    # Remove duplicates and sort
    unique = {}
    for p in solutions:
        key = (tuple(sorted((c.rank, c.suit) for c in p.top)),
               tuple(sorted((c.rank, c.suit) for c in p.middle)),
               tuple(sorted((c.rank, c.suit) for c in p.bottom)))
        if key not in unique or p.score > unique[key].score:
            unique[key] = p
    
    solutions = list(unique.values())
    solutions.sort(key=lambda p: p.score, reverse=True)
    
    return solutions[:10]


def deal_fantasyland_hand(num_cards: int = 14, include_jokers: bool = True) -> List[Card]:
    """Deal a random Fantasyland hand."""
    deck = Deck(include_jokers=include_jokers)
    deck.shuffle()
    return deck.deal(num_cards)


def print_placement(placement: Placement, rank: int = 0):
    """Print a placement nicely."""
    from game.hand_evaluator import hand_rank_name, hand_rank_3_name
    from game.card import has_joker
    
    # Use optimized hands if available (for accurate evaluation)
    top = placement.opt_top if placement.opt_top else placement.top
    middle = placement.opt_middle if placement.opt_middle else placement.middle
    bottom = placement.opt_bottom if placement.opt_bottom else placement.bottom
    
    top_rank, _ = evaluate_3_card_hand(top)
    mid_rank, _ = evaluate_5_card_hand(middle)
    bot_rank, _ = evaluate_5_card_hand(bottom)
    
    has_jokers = has_joker(placement.top + placement.middle + placement.bottom)
    
    print(f"\n{'='*60}")
    print(f"Placement #{rank + 1} - Score: {placement.score:.1f}")
    print(f"{'='*60}")
    
    # Show original placement
    print(f"  Top:    {cards_to_str(placement.top):25} [{hand_rank_3_name(top_rank)}]")
    print(f"  Middle: {cards_to_str(placement.middle):25} [{hand_rank_name(mid_rank)}]")
    print(f"  Bottom: {cards_to_str(placement.bottom):25} [{hand_rank_name(bot_rank)}]")
    
    # Show optimized if jokers present
    if has_jokers and placement.opt_top:
        print(f"\n  (Joker optimized to:)")
        print(f"  Top:    {cards_to_str(placement.opt_top)}")
        print(f"  Middle: {cards_to_str(placement.opt_middle)}")
        print(f"  Bottom: {cards_to_str(placement.opt_bottom)}")
    
    print(f"\n  Discards: {cards_to_str(placement.discards)}")
    print(f"  Total Royalties: {placement.royalties}")
    print(f"  Can Stay in FL: {placement.can_stay}")
    print(f"  Bust: {placement.is_bust}")


def main():
    """Test the solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fantasyland Solver")
    parser.add_argument("--cards", type=int, default=14, help="Number of cards (14-17)")
    parser.add_argument("--hands", type=int, default=5, help="Number of hands to solve")
    parser.add_argument("--method", choices=["exhaustive", "beam"], default="exhaustive")
    
    args = parser.parse_args()
    
    print(f"Solving {args.hands} Fantasyland hands with {args.cards} cards each")
    print(f"Method: {args.method}")
    print()
    
    for i in range(args.hands):
        print(f"\n{'#'*60}")
        print(f"# Hand {i+1}")
        print(f"{'#'*60}")
        
        hand = deal_fantasyland_hand(args.cards)
        print(f"Dealt: {cards_to_str(hand)}")
        
        start = time.time()
        
        if args.method == "exhaustive":
            solutions = solve_fantasyland_exhaustive(hand, max_solutions=3)
        else:
            solutions = solve_fantasyland_beam(hand, beam_width=1000)
        
        elapsed = time.time() - start
        print(f"Solved in {elapsed:.2f}s")
        
        if solutions:
            print(f"\nTop {len(solutions)} solutions:")
            for j, sol in enumerate(solutions):
                print_placement(sol, j)
        else:
            print("No valid placements found!")


if __name__ == "__main__":
    main()
