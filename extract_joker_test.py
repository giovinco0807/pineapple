import json

with open(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_local_filtered.json') as f:
    data = json.load(f)

# Extract only joker hands
joker_hands = [d for d in data if 'Jo' in d['hand'] or 'JK' in d['hand']][:5]

# Re-index
for i, h in enumerate(joker_hands):
    h['hand_idx'] = i

with open(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_joker_test.json', 'w') as f:
    json.dump(joker_hands, f)

print(f"Saved {len(joker_hands)} joker hands for test")
for h in joker_hands:
    print(f"  [{h['hand_idx']}] {h['hand']}")
