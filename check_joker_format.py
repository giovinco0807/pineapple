import json

with open(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_local_filtered.json') as f:
    data = json.load(f)

jokers = [d for d in data if 'Jo' in d['hand'] or 'JK' in d['hand'] or 'Joker' in d['hand']]
print(f"Joker hands: {len(jokers)}/{len(data)}")
print()
for d in jokers[:10]:
    hand = d['hand']
    fp = d['filtered_placements']
    print(f"  [{d['hand_idx']:>3}] {hand}  (placements: {len(fp)})")
    # Show joker representation in placements
    jo_placements = [p for p in fp if 'Jo' in p or 'JK' in p or 'Joker' in p]
    print(f"        Placements with Jo/JK: {len(jo_placements)}/{len(fp)}")
    if jo_placements:
        print(f"        Example: {jo_placements[0]}")
