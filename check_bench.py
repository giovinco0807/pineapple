import json, os

bench_dir = "d:/ofc_data/bench"
files = sorted([f for f in os.listdir(bench_dir) if f.endswith('.jsonl')])

for fname in files:
    path = os.path.join(bench_dir, fname)
    size = os.path.getsize(path)
    if size == 0:
        print(f"{fname}: EMPTY")
        continue
    
    with open(path) as f:
        lines = f.readlines()
    
    print(f"\n{'='*70}")
    print(f"{fname}: {len(lines)} hands, {size} bytes")
    print(f"{'='*70}")
    
    for line in lines:
        d = json.loads(line)
        hi = d["hand_idx"]
        hand = d["hand"]
        htype = d["type"]
        np = d["n_placements"]
        ns = d["n_samples"]
        nest = d["nesting"]
        ps = d["placements"]
        
        print(f"  Hand #{hi}: {hand} ({htype}, {np}p)")
        print(f"    Config: samples={ns}, nesting={nest}")
        print(f"    Top 5 EVs: ", end="")
        for i, p in enumerate(ps[:5]):
            print(f"{p['ev']:+.3f}", end="  ")
        print()
        print(f"    Best: {ps[0]['p']}  EV={ps[0]['ev']:+.3f}")
        if len(ps) > 1:
            print(f"    2nd:  {ps[1]['p']}  EV={ps[1]['ev']:+.3f}  gap={ps[0]['ev']-ps[1]['ev']:.3f}")
