import json, sys

path = r"d:\ofc_data\t0_local_hifi.jsonl"
with open(path) as f:
    for line in f:
        d = json.loads(line)
        hand = d["hand"]
        htype = d["type"]
        n = d["n_placements"]
        ns = d["n_samples"]
        nest = d["nesting"]
        
        print(f"=" * 70)
        print(f"Hand: {hand}")
        print(f"Type: {htype}  |  Placements: {n}  |  Samples: {ns}  |  Nesting: {nest}")
        print(f"-" * 70)
        
        placements = d["placements"]
        for i, p in enumerate(placements[:25]):
            ev = p["ev"]
            desc = p["p"]
            marker = "  *** BEST ***" if i == 0 else ""
            print(f"  #{i+1:>3}  EV: {ev:+7.3f}  |  {desc}{marker}")
        
        if len(placements) > 25:
            print(f"  ... ({len(placements) - 25} more placements)")
        
        print()
        best_ev = placements[0]["ev"]
        second_ev = placements[1]["ev"]
        worst_ev = placements[-1]["ev"]
        print(f"  Best EV:    {best_ev:+.3f}")
        print(f"  2nd EV:     {second_ev:+.3f}  (gap: {best_ev - second_ev:.3f})")
        print(f"  Worst EV:   {worst_ev:+.3f}  (gap: {best_ev - worst_ev:.3f})")
        print()
