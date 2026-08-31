"""Final outcome statistics - FL entry breakdown by top row."""
import json
from collections import Counter

filepath = r'D:\ofc_data\mc_teacher_s50_v2\mc_s50_merged.jsonl'

with open(filepath) as f:
    recs = [json.loads(l) for l in f]

final = [r for r in recs if r.get('turn') == -1]
total = len(final)
busted = sum(1 for r in final if r.get('busted'))
fl_entry = sum(1 for r in final if r.get('fl_entry'))
not_busted = total - busted

scores = [r.get('score', 0) for r in final]
royalties = [r.get('royalty', 0) for r in final if not r.get('busted')]
avg_score = sum(scores) / total
avg_royalty = sum(royalties) / len(royalties) if royalties else 0

print(f"Total Hands: {total}")
print(f"Avg Score: {avg_score:.2f}")
print(f"Busted: {busted} ({busted/total*100:.1f}%)")
print(f"Not Busted: {not_busted} ({not_busted/total*100:.1f}%)")
print(f"Avg Royalty (non-busted): {avg_royalty:.2f}")
print(f"FL Entry: {fl_entry} ({fl_entry/total*100:.1f}%)")

# FL breakdown
fl_hands = [r for r in final if r.get('fl_entry')]
fl_types = Counter()

print("\nSample FL hand top rows:")
for r in fl_hands[:5]:
    board = r.get('board', {})
    top = board.get('top', [])
    print(f"  {top}")

for r in fl_hands:
    board = r.get('board', {})
    top = board.get('top', [])
    # Extract ranks, handle various card formats
    ranks = []
    for c in top:
        if len(c) >= 2:
            # Card format could be '2s', 'Td', 'Xj' etc
            rank = c[0] if c[0] != 'X' else c
            if rank != 'X':
                ranks.append(rank)
    
    rank_counts = Counter(ranks)
    most_common = rank_counts.most_common(1)
    if most_common:
        rank, cnt = most_common[0]
        if cnt >= 3:
            fl_types['Trips+'] += 1
        elif cnt == 2:
            if rank == 'Q':
                fl_types['QQ'] += 1
            elif rank == 'K':
                fl_types['KK'] += 1
            elif rank == 'A':
                fl_types['AA'] += 1
            else:
                fl_types[f'{rank}{rank}'] += 1
        else:
            fl_types['Other/Unknown'] += 1
    else:
        fl_types['Empty'] += 1

print(f"\nFL Breakdown ({fl_entry} total FL entries):")
for k in ['QQ', 'KK', 'AA', 'Trips+', 'Other/Unknown', 'Empty']:
    if k in fl_types:
        n = fl_types[k]
        print(f"  {k}: {n} ({n/fl_entry*100:.1f}%)")
