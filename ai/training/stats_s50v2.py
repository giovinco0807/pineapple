"""Statistics for the downloaded s50 v2 data (with QQ EV = 14.0)."""
import json
from collections import Counter

filepath = r'D:\ofc_data\mc_teacher_s50_v2\mc_s50_merged.jsonl'

print("Loading data...")
hands = []
with open(filepath) as f:
    for line in f:
        line = line.strip()
        if line:
            hands.append(json.loads(line))

total = len(hands)
print(f"Total records: {total}")

# Group by turn
turns = Counter(h.get('turn') for h in hands)
print(f"\nRecords by Turn:")
for t in sorted(turns.keys()):
    print(f"  Turn {t}: {turns[t]}")

# Terminal hands (turn == 4)
terminal = [h for h in hands if h.get('turn') == 4]
n_terminal = len(terminal) if terminal else turns.get(4, 0)
n_hands = total // 5 if n_terminal == 0 else n_terminal
print(f"\nTerminal hands: {n_hands}")

# Analyze all Turn 0 hands for FL/Bust stats from the best candidate
print("\n" + "="*50)
print("=== Per-Turn Statistics (from best candidate) ===")
print("="*50)

for turn_num in sorted(turns.keys()):
    turn_hands = [h for h in hands if h.get('turn') == turn_num]
    
    evs = []
    busts = []
    fls = []
    fl_qq = []
    fl_kk = []
    fl_aa = []
    fl_trips = []
    
    for h in turn_hands:
        best_idx = h.get('best_idx', 0)
        cands = h.get('candidates', [])
        if not cands or len(cands) <= best_idx:
            continue
        best = cands[best_idx]
        
        # Try different stat locations
        if 'mc' in best:
            stats = best['mc']
            ev_key = 'mean'
        elif 'ev' in best:
            stats = best
            ev_key = 'ev'
        else:
            stats = {}
            ev_key = 'ev'
        
        ev = stats.get(ev_key, 0.0)
        bust = stats.get('bust_rate', stats.get('bust_prob', 0.0))
        fl = stats.get('fl_rate', 0.0)
        
        evs.append(ev)
        busts.append(bust)
        fls.append(fl)
        
        fl_types = stats.get('fl_type_rates', {})
        if fl_types:
            fl_qq.append(fl_types.get('qq', 0))
            fl_kk.append(fl_types.get('kk', 0))
            fl_aa.append(fl_types.get('aa', 0))
            fl_trips.append(fl_types.get('trips', 0))
    
    if evs:
        avg_ev = sum(evs) / len(evs)
        avg_bust = sum(busts) / len(busts) * 100
        avg_fl = sum(fls) / len(fls) * 100
        print(f"\nTurn {turn_num} ({len(turn_hands)} records):")
        print(f"  Avg EV: {avg_ev:.3f}")
        print(f"  Avg Bust Rate: {avg_bust:.1f}%")
        print(f"  Avg FL Rate: {avg_fl:.1f}%")
        if fl_qq:
            print(f"  FL Breakdown: QQ={sum(fl_qq)/len(fl_qq)*100:.1f}% KK={sum(fl_kk)/len(fl_kk)*100:.1f}% AA={sum(fl_aa)/len(fl_aa)*100:.1f}% Trips={sum(fl_trips)/len(fl_trips)*100:.1f}%")

# Count number of unique hands
unique_hids = set(h.get('hand_id') for h in hands)
print(f"\n{'='*50}")
print(f"Unique Hand IDs: {len(unique_hids)}")
print(f"Total Records: {total}")
print(f"Records per Hand: {total / len(unique_hids):.1f}")
