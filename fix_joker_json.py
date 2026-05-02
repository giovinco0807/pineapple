"""Fix existing JSON: replace 'Jo' with 'JK' in hand and placement strings."""
import json
import re

input_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_joker_test.json'
output_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_joker_test_fixed.json'

with open(input_path) as f:
    data = json.load(f)

for entry in data:
    # Fix hand string: replace standalone "Jo" with "JK"
    hand_cards = entry['hand'].split()
    hand_cards = ['JK' if c == 'Jo' else c for c in hand_cards]
    entry['hand'] = ' '.join(hand_cards)
    
    # Fix placement strings
    fixed_placements = []
    for p in entry['filtered_placements']:
        # Replace "Jo" with "JK" when it appears as a standalone card token
        fixed = re.sub(r'\bJo\b', 'JK', p)
        fixed_placements.append(fixed)
    entry['filtered_placements'] = fixed_placements

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Fixed {len(data)} entries")
for h in data:
    print(f"  [{h['hand_idx']}] {h['hand']}")
    print(f"    placements[0]: {h['filtered_placements'][0]}")
