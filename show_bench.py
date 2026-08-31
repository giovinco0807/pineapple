import json, sys

path = r"d:\ofc_data\bench\bench_n321_s100.jsonl"
with open(path) as f:
    for line in f:
        d = json.loads(line)
        hand = d["hand"]
        htype = d["type"]
        n = d["n_placements"]
        print(f"Hand {d['hand_idx']}: {hand} ({htype}, {n} placements)")
        for i, p in enumerate(d["placements"][:5]):
            marker = " <<<" if i == 0 else ""
            print(f"  #{i+1:>2}  EV:{p['ev']:+8.3f}  {p['p']}{marker}")
        print()
