"""Verify data correctness - check solution cards are in hand."""
import json

data = [json.loads(l) for l in open('ai/data/fl_rust_14cards_random.jsonl')]
print(f'Total samples: {len(data)}')

errors = 0
for d in data[:10]:
    hand_set = set()
    for c in d['hand']:
        if c.get('is_joker'):
            hand_set.add('JK')
        else:
            hand_set.add(f"{c['rank']}{c['suit']}")
    
    solution_set = set()
    for row in ['top', 'middle', 'bottom']:
        for c in d['solution'][row]:
            if c.get('is_joker'):
                solution_set.add('JK')
            else:
                solution_set.add(f"{c['rank']}{c['suit']}")
    
    # Check if solution cards are in hand
    missing = solution_set - hand_set
    if missing:
        print(f"Sample {d['sample_id']}: MISSING {missing}")
        errors += 1
    else:
        print(f"Sample {d['sample_id']}: OK")

print(f"\nErrors: {errors}/{min(10, len(data))}")
