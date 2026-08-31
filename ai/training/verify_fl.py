"""Deep dive into a specific Turn 0 hand to verify FL rate accuracy."""
import json

filepath = r'D:\ofc_data\mc_teacher\mc_s50_merged.jsonl'

print("Loading Turn 0 hands...")
t0 = []
with open(filepath) as f:
    for line in f:
        r = json.loads(line)
        if r.get('turn') == 0:
            t0.append(r)

print(f"Found {len(t0)} Turn 0 records.\n")

# Find hands where the best action has FL > 15%
for hand in t0:
    best_idx = hand.get('best_idx', 0)
    cands = hand.get('candidates', [])
    if not cands:
        continue
    best = cands[best_idx]
    mc = best.get('mc', {})
    fl = mc.get('fl_rate', 0)
    if fl > 0.15:
        hid = hand['hand_id']
        dealt = hand['dealt']
        print(f"Hand {hid}: dealt={dealt}")
        print(f"Best action: {best.get('placements')}")
        print(f"FL rate: {fl*100:.0f}%  Bust: {mc.get('bust_rate',0)*100:.0f}%")
        print(f"FL types: {mc.get('fl_type_rates',{})}")
        print()

        # Sort all candidates by FL rate descending
        scored = []
        for ci, c in enumerate(cands):
            m = c.get('mc', {})
            scored.append((ci, c.get('placements'), m.get('fl_rate',0), m.get('bust_rate',0), m.get('fl_type_rates',{})))
        scored.sort(key=lambda x: -x[2])

        print("Top 15 candidates by FL rate:")
        for ci, pl, flr, br, flt in scored[:15]:
            top_c = [p[0] for p in pl if p[1]=='top']
            mid_c = [p[0] for p in pl if p[1]=='middle']
            bot_c = [p[0] for p in pl if p[1]=='bottom']
            mark = ' <<<BEST' if ci == best_idx else ''
            qq = flt.get('qq', 0)*100
            kk = flt.get('kk', 0)*100
            aa = flt.get('aa', 0)*100
            trips = flt.get('trips', 0)*100
            print(f"  [{ci:3d}] T:{top_c} M:{mid_c} B:{bot_c}")
            print(f"         FL={flr*100:.0f}% (QQ:{qq:.0f} KK:{kk:.0f} AA:{aa:.0f} Trips:{trips:.0f})  Bust={br*100:.0f}%{mark}")
        
        print("\n" + "="*60)
        break  # Just show 1 hand
