"""Verify Rust-generated data."""
import json

data = [json.loads(l) for l in open('ai/data/test_14cards.jsonl')]
print(f'Total samples: {len(data)}')

errors = 0
for d in data[:20]:
    hand_set = set(d['hand'])
    solution_set = set(d['solution']['top'] + d['solution']['middle'] + d['solution']['bottom'])
    
    missing = solution_set - hand_set
    if missing:
        print(f"Sample {d['sample_id']}: MISSING {missing}")
        errors += 1

print(f"\nErrors in first 20: {errors}")
print(f"\nSample 0:")
print(f"  Hand: {d['hand']}")
print(f"  Top: {data[0]['solution']['top']}")
print(f"  Middle: {data[0]['solution']['middle']}")
print(f"  Bottom: {data[0]['solution']['bottom']}")
