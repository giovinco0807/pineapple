import json
from collections import Counter

counts = {'top':[], 'mid':[], 'bot':[]}
cards_list = []
with open(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_v3.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        sol = d['solution']
        counts['top'].append(len(sol['top']))
        counts['mid'].append(len(sol['mid']))
        counts['bot'].append(len(sol['bot']))
        cards_list.append(d['n_cards'])

for row in ['top','mid','bot']:
    c = Counter(counts[row])
    print(f'{row}: {dict(sorted(c.items()))}')
print(f'Total samples: {len(counts["top"])}')
print(f'n_cards: {dict(sorted(Counter(cards_list).items()))}')
