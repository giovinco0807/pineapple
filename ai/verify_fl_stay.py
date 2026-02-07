"""
Verify FL Stay data - check if can_stay samples actually have Top Trips or Bottom Quads+.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from game.card import Card
from game.hand_evaluator import evaluate_3_card_hand, evaluate_5_card_hand, HandRank, HandRank3

RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def dict_to_cards(card_dicts):
    """Convert list of card dicts to Card objects."""
    cards = []
    for c in card_dicts:
        if c.get('is_joker') or c.get('rank') == 'JK':
            # Create a Joker card - use special handling
            joker = Card.__new__(Card)
            joker.rank = 'JK'
            joker.suit = ''
            joker.rank_value = 0
            joker.is_joker = True
            cards.append(joker)
        else:
            cards.append(Card(c['rank'], c['suit']))
    return cards


def main():
    data_file = Path(__file__).parent / 'data' / 'fl_rust_14cards_random.jsonl'
    
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(data)}")
    print()
    
    # Stats by joker count
    for jc in [0, 1, 2]:
        samples = [d for d in data if d.get('joker_count', 0) == jc]
        if not samples:
            continue
        
        print(f"=== Joker Count: {jc} ({len(samples)} samples) ===")
        
        top_trips = 0
        bottom_quads = 0
        bottom_sf = 0
        both = 0
        neither_but_stay = 0
        invalid_stay = 0
        
        top_ranks = Counter()
        bottom_ranks = Counter()
        
        for d in samples:
            top_cards = dict_to_cards(d['solution']['top'])
            bottom_cards = dict_to_cards(d['solution']['bottom'])
            
            top_rank, _ = evaluate_3_card_hand(top_cards)
            bottom_rank, _ = evaluate_5_card_hand(bottom_cards)
            
            top_ranks[top_rank.name] += 1
            bottom_ranks[bottom_rank.name] += 1
            
            has_top_trips = (top_rank == HandRank3.THREE_OF_A_KIND)
            has_bottom_quads_plus = (bottom_rank >= HandRank.FOUR_OF_A_KIND)
            
            if has_top_trips:
                top_trips += 1
            if has_bottom_quads_plus:
                if bottom_rank == HandRank.FOUR_OF_A_KIND:
                    bottom_quads += 1
                else:
                    bottom_sf += 1
            if has_top_trips and has_bottom_quads_plus:
                both += 1
            
            can_stay = d.get('can_stay', False)
            should_stay = has_top_trips or has_bottom_quads_plus
            
            if can_stay and not should_stay:
                invalid_stay += 1
                print(f"  INVALID: Sample {d['sample_id']} - can_stay=True but no Trips/Quads+")
                print(f"    Top: {top_rank.name}, Bottom: {bottom_rank.name}")
            elif should_stay and not can_stay:
                print(f"  MISSING: Sample {d['sample_id']} - should stay but can_stay=False")
        
        actual_stay = sum(1 for d in samples if d.get('can_stay'))
        expected_stay = top_trips + bottom_quads + bottom_sf - both
        
        print(f"\n  Top Ranks: {dict(top_ranks)}")
        print(f"  Bottom Ranks: {dict(bottom_ranks)}")
        print()
        print(f"  Top Trips: {top_trips} ({100*top_trips/len(samples):.1f}%)")
        print(f"  Bottom Quads: {bottom_quads} ({100*bottom_quads/len(samples):.1f}%)")
        print(f"  Bottom SF+: {bottom_sf} ({100*bottom_sf/len(samples):.1f}%)")
        print(f"  Both: {both}")
        print()
        print(f"  Expected FL Stay: {expected_stay} ({100*expected_stay/len(samples):.1f}%)")
        print(f"  Actual FL Stay: {actual_stay} ({100*actual_stay/len(samples):.1f}%)")
        print(f"  Invalid Stay: {invalid_stay}")
        print()


if __name__ == '__main__':
    main()
