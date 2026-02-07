"""Investigate joker-2 cases that are not FL Stay."""
import json

data = [json.loads(l) for l in open('ai/data/fl_14cards_v2_10k.jsonl')]
j2 = [d for d in data if d['joker_count'] == 2]
print(f'Joker 2 total: {len(j2)}')
stay = [d for d in j2 if d['can_stay']]
no_stay = [d for d in j2 if not d['can_stay']]
print(f'Stay: {len(stay)}, No Stay: {len(no_stay)}')

if no_stay:
    print('\nNot staying samples:')
    for d in no_stay[:5]:
        print(f"Hand: {d['hand']}")
        print(f"Top: {d['solution']['top']}")
        print(f"Middle: {d['solution']['middle']}")
        print(f"Bottom: {d['solution']['bottom']}")
        print(f"Score: {d['reward']}, Royalties: {d['royalties']}")
        print()
