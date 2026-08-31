#!/usr/bin/env python3
"""Convert GCP shard format to v3 training JSONL format.

GCP format:  hand="8h Kc 3c Ac Tc", placement.p="Top[Ac] Mid[8h] Bot[Kc 3c Tc]"
v3 format:   hand=[{rank,suit},...], placement={top:[{rank,suit}],mid:...,bot:...}
"""

import json
import glob
import os
import re
import sys


RANK_MAP_REV = {
    '2':'2','3':'3','4':'4','5':'5','6':'6','7':'7',
    '8':'8','9':'9','T':'T','J':'J','Q':'Q','K':'K','A':'A'
}
SUIT_MAP_REV = {'h':'hearts','d':'diamonds','c':'clubs','s':'spades'}


def parse_card_str(s):
    """Parse '8h' → {'rank':'8','suit':'hearts'}"""
    s = s.strip()
    if not s:
        return None
    if s in ('JK', 'Jo', 'Joker', 'X1', 'X2'):
        return {'rank': 'Joker', 'suit': 'joker'}
    if len(s) < 2:
        return None
    rank = s[:-1]
    suit = s[-1]
    if rank not in RANK_MAP_REV or suit not in SUIT_MAP_REV:
        return None
    return {'rank': RANK_MAP_REV[rank], 'suit': SUIT_MAP_REV[suit]}


def parse_hand_str(hand_str):
    """Parse '8h Kc 3c Ac Tc' → list of card dicts"""
    cards = []
    for token in hand_str.strip().split():
        c = parse_card_str(token)
        if c:
            cards.append(c)
    return cards


def parse_placement_str(p_str):
    """Parse 'Top[Ac] Mid[8h] Bot[Kc 3c Tc]' → {top, mid, bot}"""
    result = {'top': [], 'mid': [], 'bot': []}
    # Match Top[...] Mid[...] Bot[...]
    for row_name in ['Top', 'Mid', 'Bot']:
        pattern = rf'{row_name}\[([^\]]*)\]'
        m = re.search(pattern, p_str)
        if m:
            content = m.group(1).strip()
            if content:
                row_key = row_name.lower()
                for token in content.split():
                    c = parse_card_str(token)
                    if c:
                        result[row_key].append(c)
    return result


def convert_shard_sample(sample):
    """Convert a GCP shard sample to v3 format."""
    hand_str = sample.get('hand', '')
    hand = parse_hand_str(hand_str)
    if len(hand) != 5:
        return None

    placements_raw = sample.get('placements', [])
    if not placements_raw:
        return None

    placements = []
    for p_raw in placements_raw:
        p_str = p_raw.get('p', '')
        ev = p_raw.get('ev', 0)
        parsed = parse_placement_str(p_str)
        parsed['ev'] = ev
        placements.append(parsed)

    if not placements:
        return None

    # Sort by EV descending
    placements.sort(key=lambda x: x['ev'], reverse=True)
    best_ev = placements[0]['ev']

    # Build solution from best placement
    best = placements[0]
    solution = {
        'top': best['top'],
        'mid': best['mid'],
        'bot': best['bot'],
    }

    return {
        'hand': hand,
        'n_cards': 5,
        'hand_type': sample.get('type', ''),
        'solution': solution,
        'best_ev': best_ev,
        'n_placements': len(placements),
        'placements': placements,
    }


def convert_filtered_sample(sample):
    """Convert nn_filtered format to v3 format."""
    hand_str = sample.get('hand', '')
    hand = parse_hand_str(hand_str)
    if len(hand) != 5:
        return None

    placements_raw = sample.get('filtered_placements', [])
    if not placements_raw:
        return None

    placements = []
    for p_raw in placements_raw:
        p_str = p_raw.get('p', '')
        ev = p_raw.get('ev', 0)
        parsed = parse_placement_str(p_str)
        parsed['ev'] = ev
        placements.append(parsed)

    if not placements:
        return None

    placements.sort(key=lambda x: x['ev'], reverse=True)
    best_ev = placements[0]['ev']
    best = placements[0]

    return {
        'hand': hand,
        'n_cards': 5,
        'hand_type': '',
        'solution': {'top': best['top'], 'mid': best['mid'], 'bot': best['bot']},
        'best_ev': best_ev,
        'n_placements': len(placements),
        'placements': placements,
    }


def main():
    data_dir = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\gcp_all_data'
    existing = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_v3.jsonl'
    output = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_all.jsonl'

    seen = set()
    total = 0

    # First load existing v3 data
    existing_count = 0
    with open(output, 'w', encoding='utf-8') as fout:
        if os.path.exists(existing):
            with open(existing, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        hand = d.get('hand', [])
                        if len(hand) != 5:
                            continue
                        key = tuple(sorted((c.get('rank',''), c.get('suit','')) for c in hand))
                        if key in seen:
                            continue
                        seen.add(key)
                        fout.write(json.dumps(d, ensure_ascii=False) + '\n')
                        total += 1
                        existing_count += 1
                    except json.JSONDecodeError:
                        pass
            print(f"Existing v3 data: {existing_count} hands")

        # Process shard files
        shard_files = sorted(glob.glob(os.path.join(data_dir, 't0_full_shard_*.jsonl')))
        shard_files += sorted(glob.glob(os.path.join(data_dir, 't0_*_shard_*.jsonl')))
        shard_files += sorted(glob.glob(os.path.join(data_dir, 'results_*.jsonl')))
        shard_files += sorted(glob.glob(os.path.join(data_dir, 'shard_*.jsonl')))
        # Deduplicate file list
        shard_files = sorted(set(shard_files))

        shard_count = 0
        for fp in shard_files:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        converted = convert_shard_sample(d)
                        if not converted:
                            continue
                        hand = converted['hand']
                        key = tuple(sorted((c['rank'], c['suit']) for c in hand))
                        if key in seen:
                            continue
                        seen.add(key)
                        fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
                        total += 1
                        shard_count += 1
                    except json.JSONDecodeError:
                        pass
        print(f"Shard files converted: {shard_count} new hands from {len(shard_files)} files")

        # Skip nn_filtered files - they don't have EV values
        print("Skipping nn_filtered files (no EV data)")

        # Process any remaining .jsonl files
        other_files = set(glob.glob(os.path.join(data_dir, '*.jsonl'))) - set(shard_files)
        other_count = 0
        for fp in sorted(other_files):
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        # Try shard format first
                        converted = convert_shard_sample(d)
                        if not converted:
                            continue
                        hand = converted['hand']
                        key = tuple(sorted((c['rank'], c['suit']) for c in hand))
                        if key in seen:
                            continue
                        seen.add(key)
                        fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
                        total += 1
                        other_count += 1
                    except json.JSONDecodeError:
                        pass
        print(f"Other files converted: {other_count} new hands from {len(other_files)} files")

    print(f"\n=== TOTAL: {total} unique hands ===")
    print(f"Output: {output}")


if __name__ == '__main__':
    main()
