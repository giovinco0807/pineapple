#!/usr/bin/env python3
"""Quick analysis of available benchmark data.
Compare configs against the highest-fidelity one as ground truth.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"d:\ofc_data\bench_matrix")

def load_config(name):
    """Load JSONL file, return dict: hand_idx -> list of (placement, ev)"""
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        return {}
    hands = {}
    for line in path.read_text().strip().split('\n'):
        if not line.strip():
            continue
        d = json.loads(line)
        idx = d['hand_idx']
        placements = [(p['p'], p['ev']) for p in d['placements']]
        hands[idx] = placements
    return hands

def compare(gt_name, test_name):
    """Compare test config against ground truth."""
    gt = load_config(gt_name)
    test = load_config(test_name)
    
    if not gt or not test:
        return None
    
    common = set(gt.keys()) & set(test.keys())
    if len(common) < 3:
        return None
    
    top1_agree = 0
    top3_agree = 0
    mae_sum = 0
    mae_count = 0
    spearman_rhos = []
    
    for idx in sorted(common):
        gt_placements = gt[idx]
        test_placements = test[idx]
        
        # Build EV dicts
        gt_ev = {p: ev for p, ev in gt_placements}
        test_ev = {p: ev for p, ev in test_placements}
        
        # Top-1 agreement
        gt_best = gt_placements[0][0] if gt_placements else None
        test_best = test_placements[0][0] if test_placements else None
        if gt_best == test_best:
            top1_agree += 1
        
        # Top-3 agreement
        gt_top3 = {p for p, _ in gt_placements[:3]}
        test_top3 = {p for p, _ in test_placements[:3]}
        if gt_best in test_top3:
            top3_agree += 1
        
        # MAE on common placements
        common_p = set(gt_ev.keys()) & set(test_ev.keys())
        for p in common_p:
            mae_sum += abs(gt_ev[p] - test_ev[p])
            mae_count += 1
        
        # Spearman on common placements
        if len(common_p) >= 5:
            gt_ranked = sorted(common_p, key=lambda p: gt_ev[p], reverse=True)
            test_ranked = sorted(common_p, key=lambda p: test_ev[p], reverse=True)
            gt_rank = {p: i for i, p in enumerate(gt_ranked)}
            test_rank = {p: i for i, p in enumerate(test_ranked)}
            n = len(common_p)
            d_sq = sum((gt_rank[p] - test_rank[p])**2 for p in common_p)
            rho = 1 - 6 * d_sq / (n * (n**2 - 1))
            spearman_rhos.append(rho)
    
    n_hands = len(common)
    result = {
        'n_hands': n_hands,
        'top1': top1_agree / n_hands,
        'top3': top3_agree / n_hands,
        'mae': mae_sum / max(1, mae_count),
        'spearman': sum(spearman_rhos) / max(1, len(spearman_rhos)),
    }
    return result

# Timing data (manual from benchmark output)
timing = {
    'n521_s30': 103.4,
    'n521_s50': 173.1,
    'n521_s100': 346.8,
    'n321_s30': 86.4,  # local
    'n321_s50': 129.0,  # GCP c2d
}

print("=" * 80)
print("OFC Pineapple - T0 Evaluator Benchmark Analysis")
print("=" * 80)
print()

# Available configs
available = []
for f in sorted(DATA_DIR.glob("*.jsonl")):
    lines = sum(1 for _ in f.open())
    if lines >= 5:
        available.append(f.stem)
        print(f"  {f.stem}: {lines} hands")

print()

if len(available) < 2:
    print("Not enough data yet. Need at least 2 complete configs.")
    sys.exit(0)

# Try different ground truths
gt_candidates = ['n521_s100', 'n521_s50', 'n321_s500', 'n321_s200']
gt_name = None
for g in gt_candidates:
    if g in available:
        gt_name = g
        break

if not gt_name:
    gt_name = available[-1]  # Use highest available

print(f"Ground Truth: {gt_name}")
print("-" * 80)
print(f"{'Config':<15} {'Top-1':>7} {'Top-3':>7} {'Spearman':>10} {'MAE':>8} {'sec/h':>7} {'ρ/s':>8}")
print("-" * 80)

for cfg in available:
    if cfg == gt_name:
        t = timing.get(cfg, 0)
        print(f"{cfg:<15} {'(GT)':>7} {'(GT)':>7} {'1.000':>10} {'0.000':>8} {t:>7.1f} {'---':>8}")
        continue
    
    result = compare(gt_name, cfg)
    if result is None:
        continue
    
    t = timing.get(cfg, 0)
    rho_per_sec = result['spearman'] / t if t > 0 else 0
    
    print(f"{cfg:<15} {result['top1']:>7.1%} {result['top3']:>7.1%} {result['spearman']:>10.4f} {result['mae']:>8.3f} {t:>7.1f} {rho_per_sec:>8.5f}")

print("-" * 80)
print()
print("Higher Spearman ρ = better rank correlation with GT")
print("Higher ρ/s = better efficiency (accuracy per second)")
