"""Verify 15-card data."""
import json

data = [json.loads(l) for l in open('ai/data/fl_15cards_v2_10k.jsonl')]
print(f'Total samples: {len(data)}')

# Verify cards
errors = 0
for d in data:
    hand_set = set(d['hand'])
    solution_set = set(d['solution']['top'] + d['solution']['middle'] + d['solution']['bottom'])
    if solution_set - hand_set:
        errors += 1
print(f'Card errors: {errors}')

# Joker distribution and stats
print('\nJoker stats:')
for j in [0, 1, 2]:
    subset = [x for x in data if x['joker_count'] == j]
    if subset:
        stay = sum(1 for x in subset if x['can_stay'])
        avg_roy = sum(x['royalties'] for x in subset) / len(subset)
        pct = 100 * len(subset) / len(data)
        stay_pct = 100 * stay / len(subset)
        print(f'Joker {j}: {len(subset)} ({pct:.1f}%), FL Stay {stay}/{len(subset)} ({stay_pct:.1f}%), avg roy={avg_roy:.2f}')

# Overall
stay = sum(1 for d in data if d['can_stay'])
avg_roy = sum(d['royalties'] for d in data) / len(data)
print(f'\nOverall: FL Stay {stay}/{len(data)} ({100*stay/len(data):.1f}%), Avg Royalty: {avg_roy:.2f}')
