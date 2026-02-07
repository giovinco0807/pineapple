"""
Training Data Generator for Fantasyland.

Uses the FL solver to generate optimal placements
that can be used as training labels for supervised learning or RL.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.card import Card, Deck, RANKS, SUITS
from ai.fl_solver import (
    deal_fantasyland_hand, 
    solve_fantasyland_beam, 
    solve_fantasyland_exhaustive,
    cards_to_str,
    Placement
)


def card_to_dict(card: Card) -> Dict:
    """Convert card to serializable dict."""
    if card.is_joker:
        return {"rank": "JOKER", "suit": "JOKER", "is_joker": True}
    return {"rank": card.rank, "suit": card.suit, "is_joker": False}


def placement_to_dict(placement: Placement) -> Dict:
    """Convert placement to serializable dict."""
    return {
        "top": [card_to_dict(c) for c in placement.top],
        "middle": [card_to_dict(c) for c in placement.middle],
        "bottom": [card_to_dict(c) for c in placement.bottom],
        "discards": [card_to_dict(c) for c in placement.discards],
        "opt_top": [card_to_dict(c) for c in placement.opt_top] if placement.opt_top else None,
        "opt_middle": [card_to_dict(c) for c in placement.opt_middle] if placement.opt_middle else None,
        "opt_bottom": [card_to_dict(c) for c in placement.opt_bottom] if placement.opt_bottom else None,
        "is_bust": placement.is_bust,
        "royalties": placement.royalties,
        "can_stay": placement.can_stay,
        "score": placement.score
    }


def generate_training_data(
    num_samples: int = 1000,
    cards_per_hand: int = 14,
    include_jokers: bool = True,
    method: str = "beam",
    output_file: str = "ai/data/fl_training_data.json",
    verbose: bool = True
) -> List[Dict]:
    """
    Generate training data for Fantasyland.
    
    Each sample contains:
    - hand: The dealt cards
    - optimal: The best placement found by solver
    - alternatives: Other good placements (for comparison)
    
    Args:
        num_samples: Number of hands to generate
        cards_per_hand: Number of cards (14-17)
        include_jokers: Whether to include jokers in deck
        method: "beam" or "exhaustive"
        output_file: Where to save the data
        verbose: Print progress
    """
    data = []
    
    # Stats
    total_time = 0
    hands_with_stay = 0
    hands_with_bust_only = 0
    total_royalties = 0
    
    start_total = time.time()
    
    for i in range(num_samples):
        # Deal hand
        hand = deal_fantasyland_hand(cards_per_hand, include_jokers=include_jokers)
        
        # Solve
        start = time.time()
        if method == "exhaustive":
            solutions = solve_fantasyland_exhaustive(hand, max_solutions=5)
        else:
            solutions = solve_fantasyland_beam(hand, beam_width=500)
        elapsed = time.time() - start
        total_time += elapsed
        
        # Create sample
        sample = {
            "id": i,
            "hand": [card_to_dict(c) for c in hand],
            "hand_str": cards_to_str(hand),
            "num_cards": len(hand),
            "has_joker": any(c.is_joker for c in hand),
            "solve_time": elapsed
        }
        
        if solutions:
            best = solutions[0]
            sample["optimal"] = placement_to_dict(best)
            sample["optimal_str"] = {
                "top": cards_to_str(best.top),
                "middle": cards_to_str(best.middle),
                "bottom": cards_to_str(best.bottom),
                "discards": cards_to_str(best.discards)
            }
            sample["alternatives"] = [placement_to_dict(s) for s in solutions[1:]]
            sample["num_valid_placements"] = len(solutions)
            
            # Stats
            if best.can_stay:
                hands_with_stay += 1
            total_royalties += best.royalties
        else:
            sample["optimal"] = None
            sample["num_valid_placements"] = 0
            hands_with_bust_only += 1
        
        data.append(sample)
        
        if verbose and (i + 1) % 100 == 0:
            avg_time = total_time / (i + 1)
            print(f"Generated {i + 1}/{num_samples} samples (avg {avg_time:.2f}s per hand)")
    
    # Summary stats
    elapsed_total = time.time() - start_total
    stats = {
        "num_samples": num_samples,
        "cards_per_hand": cards_per_hand,
        "include_jokers": include_jokers,
        "method": method,
        "total_time": elapsed_total,
        "avg_time_per_hand": total_time / num_samples,
        "hands_with_stay": hands_with_stay,
        "stay_rate": hands_with_stay / num_samples,
        "hands_with_bust_only": hands_with_bust_only,
        "avg_royalties": total_royalties / num_samples if num_samples > 0 else 0
    }
    
    output = {
        "stats": stats,
        "data": data
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Data Generation Complete")
        print(f"{'='*60}")
        print(f"  Samples: {num_samples}")
        print(f"  Total time: {elapsed_total:.1f}s")
        print(f"  Avg time per hand: {stats['avg_time_per_hand']:.2f}s")
        print(f"  Stay rate: {stats['stay_rate']*100:.1f}%")
        print(f"  Avg royalties: {stats['avg_royalties']:.1f}")
        print(f"  Bust-only hands: {hands_with_bust_only}")
        print(f"  Saved to: {output_file}")
    
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FL training data")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--cards", type=int, default=14, help="Cards per hand (14-17)")
    parser.add_argument("--method", choices=["beam", "exhaustive"], default="beam")
    parser.add_argument("--output", type=str, default="ai/data/fl_training_data.json")
    parser.add_argument("--no-jokers", action="store_true", help="Exclude jokers")
    
    args = parser.parse_args()
    
    generate_training_data(
        num_samples=args.samples,
        cards_per_hand=args.cards,
        include_jokers=not args.no_jokers,
        method=args.method,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
