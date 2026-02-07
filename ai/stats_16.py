"""Get stats for 16-card."""
import json

data = [json.loads(l) for l in open('ai/data/fl_16cards_v2_10k.jsonl')]
print(f'Total: {len(data)}')

for j in [0, 1, 2]:
    subset = [x for x in data if x['joker_count'] == j]
    if subset:
        count = len(subset)
        pct = 100 * count / len(data)
        stay = sum(1 for x in subset if x['can_stay'])
        stay_pct = 100 * stay / count
        avg_roy = sum(x['royalties'] for x in subset) / count
        contrib = pct * stay_pct / 100
        print(f'{j}枚 | {count} | {pct:.1f}% | {stay_pct:.1f}% | {avg_roy:.2f} | {contrib:.2f}%')

total_stay = sum(1 for d in data if d['can_stay'])
total_avg = sum(d['royalties'] for d in data) / len(data)
print(f'\nTotal: {total_stay}/{len(data)} ({100*total_stay/len(data):.2f}%), avg roy={total_avg:.2f}')
