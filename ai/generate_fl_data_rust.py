"""
Generate Fantasyland training data using Rust solver (Single-threaded, reliable).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))
from game.card import Deck

RUST_SOLVER_PATH = Path(__file__).parent / "rust_solver" / "target" / "release" / "fl_solver.exe"

RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
RANK_REV = {v: k for k, v in RANK_MAP.items()}
SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
SUIT_REV = {0: 's', 1: 'h', 2: 'd', 3: 'c'}


def deal_hand(num_cards: int, include_jokers: bool = True) -> List[Dict]:
    """Deal a hand with specified number of cards."""
    deck = Deck(include_jokers=include_jokers)
    deck.shuffle()
    
    cards = []
    while len(cards) < num_cards:
        dealt = deck.deal()
        if dealt is None:
            break
        card = dealt[0] if isinstance(dealt, list) else dealt
        if card.is_joker:
            cards.append({'rank': 0, 'suit': 4, 'str_rank': 'JK', 'str_suit': '', 'is_joker': True})
        else:
            cards.append({
                'rank': card.rank_value, 
                'suit': SUIT_MAP.get(card.suit[0].lower(), 0),
                'str_rank': card.rank,
                'str_suit': card.suit[0].lower(),
                'is_joker': False
            })
    return cards


def rust_to_card_dict(cards):
    """Convert Rust cards (integer rank/suit) to readable format."""
    result = []
    for c in cards:
        rank_val = c['rank']
        suit_val = c['suit']
        
        if suit_val == 4:  # Joker
            result.append({'rank': 'JK', 'suit': '', 'is_joker': True})
        else:
            rank_str = RANK_REV.get(rank_val, str(rank_val))
            suit_str = SUIT_REV.get(suit_val, '?')
            result.append({'rank': rank_str, 'suit': suit_str})
    return result


def solve_one(hand_cards):
    """Solve a single hand."""
    rust_cards = [{'rank': c['rank'], 'suit': c['suit']} for c in hand_cards]
    
    result = subprocess.run(
        [str(RUST_SOLVER_PATH)],
        input=json.dumps({'cards': rust_cards}),
        capture_output=True,
        text=True,
        timeout=120
    )
    
    for line in result.stdout.split('\n'):
        if line.startswith('{'):
            response = json.loads(line)
            if response.get('success') and response.get('placement'):
                return response['placement']
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--cards', type=int, default=14, help='Number of cards per hand (14, 15, 16, or 17)')
    parser.add_argument('--jokers', type=int, default=None, choices=[0, 1, 2], help='Fixed joker count (omit for random)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    include_jokers = args.jokers is None or args.jokers > 0
    
    if args.output is None:
        if args.jokers is not None:
            args.output = f'data/fl_rust_{args.cards}cards_joker{args.jokers}.jsonl'
        else:
            args.output = f'data/fl_rust_{args.cards}cards_random.jsonl'
    
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    joker_str = f"fixed {args.jokers}" if args.jokers is not None else "random"
    print(f"Generating {args.samples} samples with {args.cards} cards ({joker_str} jokers)")
    print(f"Output: {output_path}")
    print(f"Mode: Single-threaded (reliable)")
    
    results = []
    start_time = time.time()
    
    for i in range(args.samples):
        hand_cards = deal_hand(args.cards, include_jokers)
        
        p = solve_one(hand_cards)
        if p:
            hand_json = []
            joker_count = 0
            for c in hand_cards:
                if c['is_joker']:
                    hand_json.append({'rank': 'JK', 'suit': '', 'is_joker': True})
                    joker_count += 1
                else:
                    hand_json.append({'rank': c['str_rank'], 'suit': c['str_suit']})
            
            results.append({
                'sample_id': i,
                'num_cards': args.cards,
                'joker_count': joker_count,
                'hand': hand_json,
                'solution': {
                    'top': rust_to_card_dict(p['top']),
                    'middle': rust_to_card_dict(p['middle']),
                    'bottom': rust_to_card_dict(p['bottom'])
                },
                'reward': p['score'],
                'royalties': p['top_royalty'] + p['middle_royalty'] + p['bottom_royalty'],
                'can_stay': p['can_stay']
            })
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (args.samples - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{args.samples} ({rate:.1f}/s, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - start_time
    
    # Statistics
    if results:
        joker_counts = {}
        stay_count = 0
        total_royalties = 0
        for r in results:
            jc = r.get('joker_count', 0)
            joker_counts[jc] = joker_counts.get(jc, 0) + 1
            if r['can_stay']:
                stay_count += 1
            total_royalties += r['royalties']
        
        print(f"\n=== Statistics ===")
        print(f"Joker distribution: {joker_counts}")
        print(f"FL Stay Rate: {stay_count}/{len(results)} ({100*stay_count/len(results):.1f}%)")
        print(f"Avg Royalties: {total_royalties/len(results):.2f}")
    
    print(f"\nWriting {len(results)} samples...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Done! {len(results)} samples in {elapsed:.1f}s ({len(results)/elapsed:.1f}/s)")


if __name__ == '__main__':
    main()
