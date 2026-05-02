#!/usr/bin/env python3
"""Convert CFR solver JSONL output to training-compatible JSONL format.

Reads the raw CFR output format:
  {"hand": "2s Td 7s 6c 8d", "placements": [{"p": "Top[6c] Mid[2s 7s] Bot[8d Td]", "ev": 19.6}, ...]}

Converts to the policy_v2 training format:
  {"hand": [{"rank": "2", "suit": "spades"}, ...], "solution": {"top": [...], "mid": [...], "bot": [...]}}

Takes the best placement (highest EV) as the ground-truth solution.
"""

import argparse
import json
import re
import sys
from pathlib import Path


SUIT_MAP = {
    's': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs'
}

RANK_MAP = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
    '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
}


def parse_card_token(token: str) -> dict:
    """Parse 'As' -> {'rank': 'A', 'suit': 'spades'}, 'JK' -> {'rank': 'Joker', 'suit': 'joker'}"""
    token = token.strip()
    if token in ('JK', 'Jo', 'Joker'):
        return {'rank': 'Joker', 'suit': 'joker'}
    if len(token) == 2:
        rank_ch = token[0]
        suit_ch = token[1]
        rank = RANK_MAP.get(rank_ch)
        suit = SUIT_MAP.get(suit_ch)
        if rank and suit:
            return {'rank': rank, 'suit': suit}
    raise ValueError(f"Cannot parse card token: '{token}'")


def parse_placement(placement_str: str) -> dict:
    """Parse 'Top[6c] Mid[2s 7s] Bot[8d Td]' into {top: [...], mid: [...], bot: [...]}"""
    result = {}
    for row in ['Top', 'Mid', 'Bot']:
        match = re.search(rf'{row}\[([^\]]*)\]', placement_str)
        if match:
            content = match.group(1).strip()
            if content:
                cards = [parse_card_token(t) for t in content.split()]
            else:
                cards = []
            result[row.lower()] = cards
        else:
            result[row.lower()] = []
    return result


def convert_sample(cfr_data: dict) -> dict:
    """Convert one CFR output record to training format."""
    # Parse hand cards
    hand_str = cfr_data.get('hand', '')
    hand_tokens = hand_str.split()
    hand_cards = [parse_card_token(t) for t in hand_tokens]

    # Get placements sorted by EV (descending)
    placements = cfr_data.get('placements', [])
    if not placements:
        return None

    # Best placement = highest EV
    best = max(placements, key=lambda x: x.get('ev', -999))
    best_ev = best['ev']
    solution = parse_placement(best['p'])

    # Validate: top should have 3 cards for a complete hand, but during T0 it's partial
    # The training data may have varying card counts per row
    total_placed = len(solution['top']) + len(solution['mid']) + len(solution['bot'])
    if total_placed != len(hand_cards):
        return None

    # Build all placements with EVs for richer training signal
    all_placements = []
    for p in placements:
        try:
            parsed = parse_placement(p['p'])
            all_placements.append({
                'top': parsed['top'],
                'mid': parsed['mid'],
                'bot': parsed['bot'],
                'ev': p['ev']
            })
        except:
            continue

    return {
        'hand': hand_cards,
        'n_cards': len(hand_cards),
        'hand_type': cfr_data.get('type', ''),
        'solution': solution,
        'best_ev': best_ev,
        'n_placements': len(all_placements),
        'placements': all_placements
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CFR JSONL to training format")
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='Input JSONL files (worker output)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSONL file for training')
    parser.add_argument('--min-placements', type=int, default=2,
                       help='Minimum number of placements to include sample')
    args = parser.parse_args()

    converted = 0
    skipped = 0
    errors = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as fout:
        for input_file in args.input:
            path = Path(input_file)
            if not path.exists():
                print(f"  SKIP (not found): {input_file}")
                continue

            with open(path, 'r', encoding='utf-8') as fin:
                for line_no, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cfr_data = json.loads(line)
                        sample = convert_sample(cfr_data)
                        if sample and sample['n_placements'] >= args.min_placements:
                            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            converted += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            print(f"  Error in {path.name}:{line_no}: {e}")

    print(f"\n{'='*50}")
    print(f"Conversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Output:    {output_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
