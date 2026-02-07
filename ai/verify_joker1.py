"""Verify joker-1 data correctness."""
import json

data = [json.loads(l) for l in open('ai/data/output.jsonl')]
print(f'Total samples: {len(data)}')
print(f'All joker_count=1: {all(d["joker_count"]==1 for d in data)}')

errors = 0
for d in data:
    hand_set = set(d['hand'])
    solution_set = set(d['solution']['top'] + d['solution']['middle'] + d['solution']['bottom'])
    
    missing = solution_set - hand_set
    if missing:
        print(f"Sample {d['sample_id']}: MISSING {missing}")
        errors += 1

print(f'\nErrors (missing cards): {errors}')
print(f'FL Stay count: {sum(1 for d in data if d["can_stay"])}')

print('\n--- Sample 0 ---')
d = data[0]
print(f'Hand: {d["hand"]}')
print(f'Top: {d["solution"]["top"]}')
print(f'Middle: {d["solution"]["middle"]}')
print(f'Bottom: {d["solution"]["bottom"]}')
print(f'Can Stay: {d["can_stay"]}, Royalties: {d["royalties"]}, Score: {d["reward"]}')

# Check a can_stay sample
stay_samples = [d for d in data if d["can_stay"]]
if stay_samples:
    print('\n--- FL Stay Sample ---')
    d = stay_samples[0]
    print(f'Top: {d["solution"]["top"]}')
    print(f'Bottom: {d["solution"]["bottom"]}')
    print(f'Royalties: {d["royalties"]}')
