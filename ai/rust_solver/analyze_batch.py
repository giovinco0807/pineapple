import json

with open('t0_test2.jsonl') as f:
    for line in f:
        d = json.loads(line)
        hi = d["hand_idx"]
        print(f"=== Hand #{hi} ===")
        print(f"Cards: {d['hand']}")
        print(f"Type: {d['type']}")
        print(f"Placements: {d['n_placements']} | Samples: {d['n_samples']} | Nesting: {d['nesting']}")
        print()
        placements = d['placements']
        best_ev = placements[0]['ev']
        worst_ev = placements[-1]['ev']
        print(f"EV Range: {best_ev:+.3f} ~ {worst_ev:+.3f} (spread: {best_ev - worst_ev:.3f})")
        print()
        print("--- Top 10 ---")
        for i, p in enumerate(placements[:10]):
            gap = best_ev - p['ev']
            print(f"  #{i+1:>3}  EV:{p['ev']:>+8.3f}  (gap:{gap:>6.3f})  {p['p']}")
        print()
        print("--- Bottom 5 ---")
        for i, p in enumerate(placements[-5:]):
            rank = d['n_placements'] - 5 + i + 1
            print(f"  #{rank:>3}  EV:{p['ev']:>+8.3f}  (gap:{best_ev - p['ev']:>6.3f})  {p['p']}")
        print()
        print("="*70)
        print()
