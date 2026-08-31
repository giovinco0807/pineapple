#!/usr/bin/env python3
"""
Augment T0 training data with suit permutations.

OFC Pineapple has no suit hierarchy (only flush matters = same-suit grouping).
Therefore, any permutation of the 4 suits produces an equivalent game state.

4! = 24 permutations → 24× data augmentation (including identity).

Usage:
  python augment_suit_permutation.py input.jsonl output.jsonl
"""

import json
import sys
from itertools import permutations
from pathlib import Path

SUITS = ['spades', 'hearts', 'diamonds', 'clubs']


def permute_card(card: dict, suit_map: dict) -> dict:
    """Apply suit permutation to a single card."""
    new_card = dict(card)
    suit = card.get('suit', '')
    if suit in suit_map:
        new_card['suit'] = suit_map[suit]
    return new_card


def permute_card_list(cards: list, suit_map: dict) -> list:
    """Apply suit permutation to a list of cards."""
    return [permute_card(c, suit_map) for c in cards]


def permute_sample(sample: dict, suit_map: dict) -> dict:
    """Apply suit permutation to an entire training sample."""
    new = {}
    
    # Hand
    new['hand'] = permute_card_list(sample['hand'], suit_map)
    new['n_cards'] = sample.get('n_cards', 5)
    new['hand_type'] = sample.get('hand_type', '')
    
    # Solution (best placement)
    sol = sample.get('solution', {})
    new['solution'] = {
        'top': permute_card_list(sol.get('top', []), suit_map),
        'mid': permute_card_list(sol.get('mid', []), suit_map),
        'bot': permute_card_list(sol.get('bot', []), suit_map),
    }
    
    # EVs are invariant under suit permutation
    new['best_ev'] = sample.get('best_ev', 0)
    new['n_placements'] = sample.get('n_placements', 0)
    
    # All placements
    new_placements = []
    for p in sample.get('placements', []):
        new_p = {
            'top': permute_card_list(p.get('top', []), suit_map),
            'mid': permute_card_list(p.get('mid', []), suit_map),
            'bot': permute_card_list(p.get('bot', []), suit_map),
            'ev': p.get('ev', 0),
        }
        new_placements.append(new_p)
    new['placements'] = new_placements
    
    return new


def has_joker(sample: dict) -> bool:
    """Check if sample contains a Joker card."""
    for card in sample.get('hand', []):
        if card.get('rank') == 'Joker' or card.get('suit') == 'joker':
            return True
    return False


def get_used_suits(sample: dict) -> set:
    """Get suits actually used in the hand (excluding Joker)."""
    suits = set()
    for card in sample.get('hand', []):
        s = card.get('suit', '')
        if s and s != 'joker':
            suits.add(s)
    return suits


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.jsonl output.jsonl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Generate all 24 suit permutations
    all_perms = list(permutations(SUITS))
    print(f"Suit permutations: {len(all_perms)}")
    
    n_original = 0
    n_augmented = 0
    n_joker_skipped = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            sample = json.loads(line)
            n_original += 1
            
            # For Joker hands, suit permutation is still valid
            # (Joker has suit='joker' which won't be in suit_map)
            
            seen = set()
            
            for perm in all_perms:
                suit_map = {SUITS[i]: perm[i] for i in range(4)}
                
                new_sample = permute_sample(sample, suit_map)
                
                # Deduplicate: create a canonical key from hand
                hand_key = tuple(
                    (c['rank'], c['suit']) for c in new_sample['hand']
                )
                if hand_key in seen:
                    continue
                seen.add(hand_key)
                
                fout.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                n_augmented += 1
    
    ratio = n_augmented / n_original if n_original > 0 else 0
    print(f"Original samples: {n_original}")
    print(f"Augmented samples: {n_augmented}")
    print(f"Augmentation ratio: {ratio:.1f}x")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
