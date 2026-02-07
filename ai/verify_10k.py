"""Verify 10k v2 data."""
import json

data = [json.loads(l) for l in open('ai/data/fl_14cards_v2_10k.jsonl')]
print(f'Total samples: {len(data)}')

# Verify cards
errors = 0
for d in data:
    hand_set = set(d['hand'])
    solution_set = set(d['solution']['top'] + d['solution']['middle'] + d['solution']['bottom'])
    if solution_set - hand_set:
        errors += 1

print(f'Card errors: {errors}')

# Joker distribution
jokers = {0: 0, 1: 0, 2: 0}
for d in data:
    jokers[d['joker_count']] = jokers.get(d['joker_count'], 0) + 1
print(f'Joker dist: {jokers}')

# FL Stay by joker count
for j in [0, 1, 2]:
    subset = [d for d in data if d['joker_count'] == j]
    if subset:
        stay = sum(1 for d in subset if d['can_stay'])
        print(f'Joker {j}: FL Stay {stay}/{len(subset)} ({100*stay/len(subset):.1f}%)')

# Overall stats
stay = sum(1 for d in data if d['can_stay'])
avg_roy = sum(d['royalties'] for d in data) / len(data)
print(f'\nOverall: FL Stay {stay}/{len(data)} ({100*stay/len(data):.1f}%), Avg Royalties: {avg_roy:.2f}')
