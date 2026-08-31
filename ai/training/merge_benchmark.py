"""Merge benchmark JSONL results from multiple VMs."""
import sys
import json
import glob
import numpy as np
from collections import Counter

def main():
    patterns = sys.argv[1:] if len(sys.argv) > 1 else ['results/result_*.jsonl']

    results = []
    for pattern in patterns:
        for f in sorted(glob.glob(pattern)):
            with open(f) as fp:
                for line in fp:
                    results.append(json.loads(line))

    if not results:
        print("No results found!")
        return

    n = len(results)
    busts = sum(1 for r in results if r['busted'])
    fl_entries = sum(1 for r in results if r['fl_entry'])
    fl_types = Counter(r.get('fl_type') for r in results if r['fl_entry'] and r.get('fl_type'))
    royalties = np.array([r['royalty'] for r in results])
    normal_scores = np.array([r['normal_score'] for r in results])
    total_scores = np.array([r['total_score'] for r in results])
    fl_rounds = sum(r.get('hero_fl_rounds', 0) for r in results)
    game_times = [r.get('game_time', 0) for r in results if r.get('game_time')]

    print("=" * 60)
    print(f"  Merged Results ({n} hands)")
    print("=" * 60)
    print(f"  Bust rate:     {busts/n*100:.1f}%")
    print(f"  FL entry rate: {fl_entries/n*100:.1f}%")
    if fl_entries > 0:
        print(f"    AA:    {fl_types.get('AA',0):3d} ({fl_types.get('AA',0)/fl_entries*100:.0f}%)")
        print(f"    KK:    {fl_types.get('KK',0):3d} ({fl_types.get('KK',0)/fl_entries*100:.0f}%)")
        print(f"    QQ:    {fl_types.get('QQ',0):3d} ({fl_types.get('QQ',0)/fl_entries*100:.0f}%)")
        print(f"    Trips: {fl_types.get('trips',0):3d} ({fl_types.get('trips',0)/fl_entries*100:.0f}%)")
    print(f"  Avg royalty:   {royalties.mean():.2f}")
    print(f"  Normal score:  {normal_scores.mean():+.2f} +/- {normal_scores.std():.2f}")
    se = total_scores.std() / np.sqrt(n)
    print(f"  FL bonus:      {(total_scores - normal_scores).mean():+.2f}")
    print(f"  Total score:   {total_scores.mean():+.2f} +/- {total_scores.std():.2f} (SE={se:.2f})")
    print(f"  Win rate:      {(total_scores > 0).sum()/n*100:.1f}%")
    print(f"  FL rounds:     {fl_rounds}")
    if game_times:
        print(f"  Avg speed:     {np.mean(game_times):.2f}s/hand")
    print("=" * 60)


if __name__ == "__main__":
    main()
