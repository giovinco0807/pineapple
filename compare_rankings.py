import json, glob

# Build original EV map from Phase 1-2 worker files  
orig_data = {}
for f in glob.glob(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\t0_gcs\*.jsonl'):
    for line in open(f, encoding='utf-8'):
        rec = json.loads(line.strip())
        hand = rec['hand']
        evs = {p['p']: p['ev'] for p in rec['placements']}
        sorted_p = sorted(rec['placements'], key=lambda x: x['ev'], reverse=True)
        ranks = {p['p']: i+1 for i, p in enumerate(sorted_p)}
        orig_data[hand] = {'evs': evs, 'ranks': ranks, 'n': len(rec['placements'])}

# Compare with new high-fidelity results
for line in open(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\filtered_test5_out.jsonl'):
    new = json.loads(line.strip())
    hand = new['hand']
    if hand not in orig_data:
        print("Hand " + hand + ": not found in original")
        continue
    
    old = orig_data[hand]
    htype = new['type']
    print("=" * 85)
    print("Hand " + str(new['hand_idx']) + ": " + hand + " (" + htype + ") | orig had " + str(old['n']) + " placements")
    print("{:>3} | {:>8} | {:>8} | {:>4} | {}".format("#", "NewEV", "OldEV", "Old#", "Placement"))
    print("-" * 85)
    moved = 0
    top1_changed = False
    for i, p in enumerate(new['placements'][:10]):
        o_ev = old['evs'].get(p['p'], -999)
        o_rank = old['ranks'].get(p['p'], '?')
        if o_rank != i+1:
            moved += 1
            flag = " <-- was #" + str(o_rank)
        else:
            flag = ""
        if i == 0 and o_rank != 1:
            top1_changed = True
        o_str = "{:.3f}".format(o_ev) if o_ev != -999 else "???"
        print("{:>3} | {:>8.3f} | {:>8} | {:>4} |  {}{}".format(i+1, p['ev'], o_str, str(o_rank), p['p'], flag))
    
    status = "*** TOP-1 CHANGED ***" if top1_changed else "Top-1 same"
    print("  >> Top-10: {} changed rank | {}".format(moved, status))
    print()
