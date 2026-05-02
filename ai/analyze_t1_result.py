import json
from collections import defaultdict

with open('test_1hand_top10.jsonl','r',encoding='utf-8') as f:
    data = json.loads(f.readline().strip())

t0_hand = data['t0_hand']
t1_hand = data['t1_hand']
ps = data['placements']

print(f"T0 Hand: {t0_hand}")
print(f"T1 Hand: {t1_hand}")
print(f"Total combos: {len(ps)}")
print()

# Top 30
print("=== Top 30 by EV ===")
for i, p in enumerate(ps[:30]):
    print(f"  {i+1:3d} | T0#{p['t0_idx']:2d} | {p['t0_p']}")
    print(f"      | disc={p['d']} | {p['p']} | EV={p['ev']:.3f}")

# EV range per T0
print("\n=== EV Summary per T0 Placement ===")
by_t0 = defaultdict(list)
t0_desc = {}
for p in ps:
    by_t0[p['t0_idx']].append(p['ev'])
    if p['t0_idx'] not in t0_desc:
        t0_desc[p['t0_idx']] = p['t0_p']

for idx in sorted(by_t0.keys()):
    evs = by_t0[idx]
    print(f"  T0#{idx:2d}: best={max(evs):7.3f}  worst={min(evs):7.3f}  n={len(evs):3d}")
    print(f"         {t0_desc[idx]}")

# Bottom 10
print("\n=== Bottom 10 by EV ===")
for i, p in enumerate(ps[-10:]):
    rank = len(ps) - 10 + i + 1
    print(f"  {rank:3d} | T0#{p['t0_idx']:2d} | disc={p['d']} | {p['p']} | EV={p['ev']:.3f}")
