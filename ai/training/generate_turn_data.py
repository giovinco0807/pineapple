#!/usr/bin/env python3
import json
import random
import glob
from pathlib import Path
from typing import List, Dict

def create_deck(include_joker=True) -> List[Dict[str, str]]:
    suits = ['spades', 'hearts', 'diamonds', 'clubs']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    deck = [{'rank': r, 'suit': s} for s in suits for r in ranks]
    if include_joker:
        deck.append({'rank': 'Joker', 'suit': 'joker'})
    return deck

def get_card_key(c: Dict[str, str]) -> str:
    return f"{c['rank']}_{c['suit']}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate T1 inputs from T0 results")
    parser.add_argument("--input-glob", type=str, default="ai/data/phase_f_results/*.jsonl")
    parser.add_argument("--output", type=str, default="ai/data/t1_inputs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    random.seed(args.seed)
    
    files = glob.glob(args.input_glob)
    if not files:
        print(f"No files found matching {args.input_glob}")
        return

    print(f"Found {len(files)} result files. Processing...")
    
    samples_created = 0
    full_deck = create_deck(include_joker=True)
    
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                        
                    data = json.loads(line.strip())
                    placements = data.get('placements', [])
                    if not placements:
                        continue
                        
                    # Pick the best placement
                    placements.sort(key=lambda x: x.get('ev', -999.0), reverse=True)
                    best_placement = placements[0]
                    
                    # Parse 'p' string: "Top[JK Kc] Mid[5c] Bot[7h Ts]"
                    p_str = best_placement.get('p', '')
                    if not p_str:
                        continue
                        
                    parsed_placement = {'top': [], 'mid': [], 'bot': []}
                    import re
                    
                    def parse_row(row_str):
                        cards = []
                        tokens = row_str.split()
                        for t in tokens:
                            if t == '-' or not t: continue
                            if t.lower() in ('jo', 'jk', 'joker'):
                                cards.append({'rank': 'Joker', 'suit': 'joker'})
                            else:
                                if len(t) == 2:
                                    cards.append({'rank': t[0], 'suit': {'s':'spades','h':'hearts','d':'diamonds','c':'clubs'}.get(t[1].lower())})
                        return cards
                        
                    m_top = re.search(r'Top\[(.*?)\]', p_str)
                    m_mid = re.search(r'Mid\[(.*?)\]', p_str)
                    m_bot = re.search(r'Bot\[(.*?)\]', p_str)
                    
                    if m_top: parsed_placement['top'] = parse_row(m_top.group(1))
                    if m_mid: parsed_placement['mid'] = parse_row(m_mid.group(1))
                    if m_bot: parsed_placement['bot'] = parse_row(m_bot.group(1))
                    
                    # Gather known cards
                    known_keys = set()
                    for row in ['top', 'mid', 'bot']:
                        for c in parsed_placement.get(row, []):
                            known_keys.add(get_card_key(c))
                            
                    if len(known_keys) != 5:
                        continue
                        
                    # Remaining deck
                    remaining = [c for c in full_deck if get_card_key(c) not in known_keys]
                    
                    # Draw 3 cards
                    hand = random.sample(remaining, 3)
                    
                    t1_input = {
                        'board': parsed_placement,
                        'hand': hand,
                        'turn': 1,
                        'original_ev': best_placement.get('ev', 0.0)
                    }
                    
                    out_f.write(json.dumps(t1_input) + '\n')
                    samples_created += 1
                    
                    if samples_created >= args.max_samples:
                        break
                        
            if samples_created >= args.max_samples:
                break
                
    print(f"Created {samples_created} T1 samples in {args.output}")

if __name__ == "__main__":
    main()
