"""Get avg royalties by joker count."""
import json

data = [json.loads(l) for l in open('ai/data/fl_14cards_v2_10k.jsonl')]

for j in [0, 1, 2]:
    subset = [x for x in data if x['joker_count'] == j]
    avg = sum(x['royalties'] for x in subset) / len(subset)
    stay = sum(1 for x in subset if x['can_stay'])
    print(f"Joker {j}: count={len(subset)}, FL Stay={stay}/{len(subset)} ({100*stay/len(subset):.1f}%), avg royalty={avg:.2f}")
