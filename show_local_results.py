import json

with open(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_local_worker.jsonl') as f:
    lines = f.readlines()

for i, line in enumerate(lines[:5]):
    d = json.loads(line)
    hand = d['hand']
    htype = d['type']
    n_p = d['n_placements']
    placements = d['placements']
    
    print(f'=== Hand {i}: {hand} ({htype}) | {n_p} placements ===')
    print(f'  Top 10 placements:')
    for j, p in enumerate(placements[:10]):
        ev = p["ev"]
        desc = p["p"]
        print(f'    {j+1:>2}. EV={ev:>+7.2f}  {desc}')
    print(f'  ...')
    worst = placements[-3:]
    for j, p in enumerate(worst):
        rank = n_p - len(worst) + j + 1
        ev = p["ev"]
        desc = p["p"]
        print(f'    {rank:>3}. EV={ev:>+7.2f}  {desc}')
    best_ev = placements[0]["ev"]
    worst_ev = placements[-1]["ev"]
    gap = best_ev - worst_ev
    print(f'  EV range: {best_ev:+.2f} ~ {worst_ev:+.2f} (gap={gap:.2f})')
    print()
