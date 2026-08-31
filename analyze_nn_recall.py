#!/usr/bin/env python3
"""Analyze NN filter recall: how often is the CFR-optimal in the NN's top-K?

For each hand:
1. Load NN-filtered placements (ranked by NN score, top-100)
2. Load CFR-evaluated placements (with EVs)
3. Find the CFR-best placement
4. Check what rank it had in the NN's ordering

Reports recall@K for K = 1, 3, 5, 10, 20, 30, 50, 75, 100
"""

import json
import sys
import os
import glob
from collections import defaultdict


def normalize_placement(p: str) -> str:
    """Normalize a placement string for matching."""
    return p.strip()


def analyze_slice(nn_input_path: str, cfr_output_path: str):
    """Analyze one slice: match NN rankings with CFR EVs."""
    # Load NN input (ordered by NN rank)
    with open(nn_input_path, 'r', encoding='utf-8') as f:
        nn_data = json.load(f)
    
    # Load CFR output
    cfr_hands = []
    with open(cfr_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cfr_hands.append(json.loads(line))
    
    results = []
    
    for cfr_hand in cfr_hands:
        hand_str = cfr_hand['hand']
        cfr_placements = cfr_hand.get('placements', [])
        
        if not cfr_placements:
            continue
        
        # Find matching NN hand
        nn_hand = None
        for nh in nn_data:
            if nh['hand'] == hand_str:
                nn_hand = nh
                break
        
        if nn_hand is None:
            continue
        
        nn_ordered = nn_hand.get('filtered_placements', [])
        n_total = nn_hand.get('n_total_actions', 0)
        
        # Find CFR-best placement (highest EV)
        best_cfr = max(cfr_placements, key=lambda x: x['ev'])
        best_placement = best_cfr['p']
        best_ev = best_cfr['ev']
        
        # Also get 2nd best for EV gap analysis
        sorted_by_ev = sorted(cfr_placements, key=lambda x: x['ev'], reverse=True)
        second_ev = sorted_by_ev[1]['ev'] if len(sorted_by_ev) > 1 else best_ev
        ev_gap = best_ev - second_ev
        
        # Find rank of best placement in NN ordering
        nn_rank = None
        for i, p in enumerate(nn_ordered):
            if normalize_placement(p) == normalize_placement(best_placement):
                nn_rank = i + 1  # 1-indexed
                break
        
        if nn_rank is None:
            # Try fuzzy matching (sort cards within each row)
            def sort_placement(p):
                import re
                result = p
                for row in ['Top', 'Mid', 'Bot']:
                    match = re.search(rf'{row}\[([^\]]*)\]', p)
                    if match:
                        cards = sorted(match.group(1).strip().split())
                        result = result.replace(match.group(0), f'{row}[{" ".join(cards)}]')
                return result
            
            best_sorted = sort_placement(best_placement)
            for i, p in enumerate(nn_ordered):
                if sort_placement(p) == best_sorted:
                    nn_rank = i + 1
                    break
        
        results.append({
            'hand': hand_str,
            'nn_rank': nn_rank,
            'best_ev': best_ev,
            'ev_gap': ev_gap,
            'n_total_actions': n_total,
            'n_cfr_placements': len(cfr_placements),
        })
    
    return results


def main():
    snapshot_dir = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_phase_e_snapshot'
    
    # Download slices if needed
    slices_dir = os.path.join(snapshot_dir, 'slices')
    if not os.path.exists(slices_dir):
        os.makedirs(slices_dir, exist_ok=True)
        print("Downloading NN input slices from GCS...")
        os.system(f'gsutil -m cp gs://ofc-solver-485418/t0_phase_e/slices/*.json "{slices_dir}/"')
    
    # Match slices to workers
    all_results = []
    
    worker_files = sorted(glob.glob(os.path.join(snapshot_dir, 'worker_*.jsonl')))
    slice_files = sorted(glob.glob(os.path.join(slices_dir, 'phase_e_slice_*.json')))
    
    print(f"Worker files: {len(worker_files)}")
    print(f"Slice files: {len(slice_files)}")
    
    for wf in worker_files:
        # Extract worker id from filename: worker_0_s5000000.jsonl -> 0
        basename = os.path.basename(wf)
        parts = basename.replace('worker_', '').split('_')
        worker_id = int(parts[0])
        
        slice_path = os.path.join(slices_dir, f'phase_e_slice_{worker_id:02d}.json')
        if not os.path.exists(slice_path):
            continue
        
        results = analyze_slice(slice_path, wf)
        all_results.extend(results)
    
    # Aggregate
    total = len(all_results)
    if total == 0:
        print("No results to analyze!")
        return
    
    print(f"\n{'='*60}")
    print(f"T0 NN Filter Recall Analysis")
    print(f"Total hands analyzed: {total}")
    print(f"{'='*60}\n")
    
    # Recall@K
    ks = [1, 3, 5, 10, 20, 30, 50, 75, 100]
    print("Recall@K (% of hands where CFR-best is in NN's top-K):")
    print(f"{'K':>6} | {'Recall':>8} | {'Count':>6}")
    print("-" * 30)
    
    for k in ks:
        count = sum(1 for r in all_results if r['nn_rank'] is not None and r['nn_rank'] <= k)
        recall = count / total * 100
        print(f"{k:>6} | {recall:>7.1f}% | {count:>6}/{total}")
    
    # Not found count
    not_found = sum(1 for r in all_results if r['nn_rank'] is None)
    print(f"\nNot found in NN top-100: {not_found}/{total} ({not_found/total*100:.1f}%)")
    
    # Rank distribution
    print(f"\n--- NN Rank Distribution of CFR-best ---")
    found = [r for r in all_results if r['nn_rank'] is not None]
    if found:
        ranks = [r['nn_rank'] for r in found]
        import statistics
        print(f"  Mean rank: {statistics.mean(ranks):.1f}")
        print(f"  Median rank: {statistics.median(ranks):.1f}")
        print(f"  Min: {min(ranks)}, Max: {max(ranks)}")
        
        # Histogram
        buckets = [(1,1), (2,5), (6,10), (11,20), (21,30), (31,50), (51,75), (76,100)]
        print(f"\n  Rank bucket distribution:")
        for lo, hi in buckets:
            cnt = sum(1 for r in ranks if lo <= r <= hi)
            bar = '#' * (cnt * 40 // len(ranks))
            print(f"    {lo:>3}-{hi:>3}: {cnt:>4} ({cnt/len(ranks)*100:5.1f}%) {bar}")
    
    # EV gap analysis: when NN misranks, how big is the EV gap?
    print(f"\n--- EV Gap Analysis ---")
    for k in [10, 20, 50]:
        in_topk = [r for r in all_results if r['nn_rank'] is not None and r['nn_rank'] <= k]
        not_in_topk = [r for r in all_results if r['nn_rank'] is not None and r['nn_rank'] > k]
        if in_topk:
            avg_ev_in = statistics.mean([r['best_ev'] for r in in_topk])
            avg_gap_in = statistics.mean([r['ev_gap'] for r in in_topk])
        if not_in_topk:
            avg_ev_out = statistics.mean([r['best_ev'] for r in not_in_topk])
            avg_gap_out = statistics.mean([r['ev_gap'] for r in not_in_topk])
            print(f"  Top-{k} miss: {len(not_in_topk)} hands, avg best_ev={avg_ev_out:.2f}, avg ev_gap={avg_gap_out:.3f}")


if __name__ == '__main__':
    main()
