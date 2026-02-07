"""Joker optimization for OFC Pineapple.

This module handles the complex logic of optimizing joker substitutions
to create the best possible hands while avoiding busts.

The optimization order is:
1. Bottom: Find the best possible hand (no constraint)
2. Middle: Find the best hand that is <= Bottom
3. Top: Find the best hand that is <= Middle
"""
from typing import List, Tuple, Optional, Set
from itertools import combinations, product
from .card import Card, RANKS, SUITS, has_joker, count_jokers, get_non_joker_cards


def get_available_cards(used_cards: List[Card]) -> List[Card]:
    """
    Get all standard cards not already used.
    
    Args:
        used_cards: List of cards already on the board (excluding jokers)
    
    Returns:
        List of available cards for joker substitution
    """
    used_set = set((c.rank, c.suit) for c in used_cards if not c.is_joker)
    available = []
    for suit in SUITS:
        for rank in RANKS:
            if (rank, suit) not in used_set:
                available.append(Card(rank, suit))
    return available


def generate_substitutions(
    cards: List[Card],
    available: List[Card]
) -> List[List[Card]]:
    """
    Generate all possible substitutions for jokers in a hand.
    
    Args:
        cards: Hand with potential jokers
        available: Cards available for substitution
    
    Returns:
        List of all possible hands with jokers replaced
    """
    joker_count = count_jokers(cards)
    if joker_count == 0:
        return [cards]
    
    # Get non-joker cards and their positions
    non_jokers = get_non_joker_cards(cards)
    
    # Generate all combinations of substitutions
    results = []
    for subs in combinations(available, joker_count):
        # Create new hand with substitutions
        new_hand = non_jokers + list(subs)
        results.append(new_hand)
    
    return results


def hand_strength_5(cards: List[Card]) -> Tuple:
    """
    Get hand strength for 5-card hand as a comparable tuple.
    Higher tuple = stronger hand.
    """
    from .hand_evaluator import evaluate_5_card_hand
    rank, kickers = evaluate_5_card_hand(cards)
    return (int(rank), tuple(kickers))


def hand_strength_3(cards: List[Card]) -> Tuple:
    """
    Get hand strength for 3-card hand as a comparable tuple.
    Higher tuple = stronger hand.
    """
    from .hand_evaluator import evaluate_3_card_hand
    rank, kickers = evaluate_3_card_hand(cards)
    return (int(rank), tuple(kickers))


def compare_3_vs_5(hand3_strength: Tuple, hand5_strength: Tuple) -> int:
    """
    Compare a 3-card hand strength against a 5-card hand strength.
    
    For OFC rules, a 3-card hand must not beat the 5-card hand.
    
    Returns:
        1 if 3-card is stronger (would bust)
        -1 if 5-card is stronger or equal (valid)
        0 if they're effectively equal
    """
    # 3-card hand types: HIGH_CARD(0), PAIR(1), THREE_OF_A_KIND(2)
    # 5-card hand types: HIGH_CARD(0), PAIR(1), TWO_PAIR(2), THREE_OF_A_KIND(3), ...
    
    hand3_rank = hand3_strength[0]
    hand5_rank = hand5_strength[0]
    
    # Map 3-card ranks to 5-card equivalent for comparison
    # 3-card: 0=high, 1=pair, 2=trips
    # 5-card: 0=high, 1=pair, 2=two pair, 3=trips, ...
    
    if hand3_rank == 2:  # 3-card trips
        # 3-card trips beats 5-card: high card, pair, two pair
        # 3-card trips ties with 5-card trips (compare kickers)
        if hand5_rank < 3:  # Less than 5-card trips
            return 1  # 3-card wins = bust
        elif hand5_rank == 3:  # Both trips
            # Compare the trips rank
            if hand3_strength[1] > hand5_strength[1]:
                return 1
            elif hand3_strength[1] < hand5_strength[1]:
                return -1
            return 0
        else:  # 5-card is straight or better
            return -1
    
    elif hand3_rank == 1:  # 3-card pair
        # 3-card pair only beats 5-card high card
        if hand5_rank == 0:  # 5-card high card
            return 1  # bust
        elif hand5_rank == 1:  # Both pair
            # Compare pair ranks, then kickers
            if hand3_strength[1] > hand5_strength[1]:
                return 1
            elif hand3_strength[1] < hand5_strength[1]:
                return -1
            return 0
        else:  # 5-card is two pair or better
            return -1
    
    else:  # 3-card high card (rank 0)
        if hand5_rank > 0:
            return -1  # 5-card wins
        # Both high card - compare kickers (only first 3)
        hand3_kickers = hand3_strength[1]
        hand5_kickers = hand5_strength[1][:3]  # Only compare top 3
        if hand3_kickers > hand5_kickers:
            return 1
        elif hand3_kickers < hand5_kickers:
            return -1
        return 0


def optimize_bottom(
    bottom_cards: List[Card],
    available: List[Card]
) -> Tuple[List[Card], Tuple]:
    """
    Optimize jokers in bottom row for maximum strength.
    
    Returns:
        (optimized_cards, strength_tuple)
    """
    if not has_joker(bottom_cards):
        strength = hand_strength_5(bottom_cards)
        return (bottom_cards, strength)
    
    substitutions = generate_substitutions(bottom_cards, available)
    
    best_hand = None
    best_strength = None
    
    for hand in substitutions:
        strength = hand_strength_5(hand)
        if best_strength is None or strength > best_strength:
            best_strength = strength
            best_hand = hand
    
    return (best_hand, best_strength)


def optimize_middle(
    middle_cards: List[Card],
    available: List[Card],
    bottom_strength: Tuple
) -> Tuple[Optional[List[Card]], Optional[Tuple]]:
    """
    Optimize jokers in middle row, constrained by bottom.
    Middle must be <= Bottom.
    
    Returns:
        (optimized_cards, strength_tuple) or (None, None) if impossible
    """
    if not has_joker(middle_cards):
        strength = hand_strength_5(middle_cards)
        if strength <= bottom_strength:
            return (middle_cards, strength)
        else:
            return (None, None)  # Would bust, no jokers to adjust
    
    substitutions = generate_substitutions(middle_cards, available)
    
    best_hand = None
    best_strength = None
    
    for hand in substitutions:
        strength = hand_strength_5(hand)
        # Must not exceed bottom
        if strength <= bottom_strength:
            if best_strength is None or strength > best_strength:
                best_strength = strength
                best_hand = hand
    
    return (best_hand, best_strength)


def optimize_top(
    top_cards: List[Card],
    available: List[Card],
    middle_strength: Tuple
) -> Tuple[Optional[List[Card]], Optional[Tuple]]:
    """
    Optimize jokers in top row, constrained by middle.
    Top (3-card) must be <= Middle (5-card).
    
    Returns:
        (optimized_cards, strength_tuple) or (None, None) if impossible
    """
    if not has_joker(top_cards):
        strength = hand_strength_3(top_cards)
        if compare_3_vs_5(strength, middle_strength) <= 0:
            return (top_cards, strength)
        else:
            return (None, None)  # Would bust, no jokers to adjust
    
    substitutions = generate_substitutions(top_cards, available)
    
    best_hand = None
    best_strength = None
    
    for hand in substitutions:
        strength = hand_strength_3(hand)
        # Must not exceed middle
        if compare_3_vs_5(strength, middle_strength) <= 0:
            if best_strength is None or strength > best_strength:
                best_strength = strength
                best_hand = hand
    
    return (best_hand, best_strength)


def optimize_board(
    top: List[Card],
    middle: List[Card],
    bottom: List[Card]
) -> Tuple[Optional[List[Card]], Optional[List[Card]], Optional[List[Card]], bool]:
    """
    Optimize all jokers on the board.
    
    Optimization order:
    1. Bottom: maximize (no constraint)
    2. Middle: maximize while <= bottom
    3. Top: maximize while <= middle
    
    Args:
        top: Top row cards (3)
        middle: Middle row cards (5)
        bottom: Bottom row cards (5)
    
    Returns:
        (opt_top, opt_middle, opt_bottom, is_bust)
        If bust, the optimized hands may be None
    """
    # Get all used non-joker cards
    all_cards = top + middle + bottom
    used_cards = get_non_joker_cards(all_cards)
    available = get_available_cards(used_cards)
    
    # Step 1: Optimize bottom (no constraint)
    opt_bottom, bottom_strength = optimize_bottom(bottom, available)
    
    # Update available cards (remove cards used for bottom jokers)
    if has_joker(bottom):
        available_for_middle = [c for c in available if c not in opt_bottom]
    else:
        available_for_middle = available
    
    # Step 2: Optimize middle (<= bottom)
    opt_middle, middle_strength = optimize_middle(
        middle, available_for_middle, bottom_strength
    )
    
    if opt_middle is None:
        # Cannot avoid bust in middle
        return (None, None, opt_bottom, True)
    
    # Update available cards
    if has_joker(middle):
        available_for_top = [c for c in available_for_middle if c not in opt_middle]
    else:
        available_for_top = available_for_middle
    
    # Step 3: Optimize top (<= middle)
    opt_top, top_strength = optimize_top(
        top, available_for_top, middle_strength
    )
    
    if opt_top is None:
        # Cannot avoid bust in top
        return (None, opt_middle, opt_bottom, True)
    
    return (opt_top, opt_middle, opt_bottom, False)


def get_optimized_hand_for_display(
    original: List[Card],
    optimized: List[Card]
) -> List[dict]:
    """
    Create display info showing original and optimized cards.
    
    Returns list of dicts with 'original', 'optimized', 'is_joker' keys.
    """
    result = []
    opt_idx = 0
    
    for card in original:
        if card.is_joker:
            result.append({
                'original': card.to_dict(),
                'optimized': optimized[opt_idx].to_dict() if opt_idx < len(optimized) else None,
                'is_joker': True
            })
        else:
            result.append({
                'original': card.to_dict(),
                'optimized': card.to_dict(),
                'is_joker': False
            })
        opt_idx += 1
    
    return result
