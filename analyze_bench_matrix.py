"""
Benchmark Matrix Analysis: Compare nesting/samples configs for T0 EV accuracy.

Metrics:
- Top-1 Agreement: Does this config pick the same best placement as ground truth?
- Top-3 Agreement: Does ground truth's best appear in this config's top 3?
- Spearman ρ: Rank correlation of all placement EVs vs ground truth
- EV MAE: Mean absolute error of EVs vs ground truth
- Speed: seconds per hand

Ground Truth: n321_s2000 (or the most expensive available config)
"""

import json, os, sys
from pathlib import Path
from scipy import stats
import numpy as np

BENCH_DIR = Path("d:/ofc_data/bench_matrix")
TIMING_FILE = BENCH_DIR / "timing.csv"

# Priority order for ground truth selection
GT_PRIORITY = ["n321_s2000", "n532_s500", "n321_s1000"]

def load_data(label):
    path = BENCH_DIR / f"{label}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return None
    hands = []
    with open(path) as f:
        for line in f:
            hands.append(json.loads(line))
    return hands

def load_timing():
    timings = {}
    if TIMING_FILE.exists():
        with open(TIMING_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith("config") or not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 6:
                    timings[parts[0]] = {
                        "nesting": parts[1],
                        "samples": int(parts[2]),
                        "total_s": float(parts[4]),
                        "per_hand_s": float(parts[5]),
                    }
    return timings

def placement_ev_dict(hand_data):
    """Build dict: placement_desc -> ev"""
    return {p["p"]: p["ev"] for p in hand_data["placements"]}

def compare_configs(gt_data, test_data):
    """Compare test config against ground truth across all hands."""
    n_hands = min(len(gt_data), len(test_data))
    
    top1_agree = 0
    top3_agree = 0
    rho_list = []
    mae_list = []
    ev_bias_list = []
    
    for i in range(n_hands):
        gt = gt_data[i]
        test = test_data[i]
        
        gt_placements = gt["placements"]
        test_placements = test["placements"]
        
        gt_best = gt_placements[0]["p"]
        test_best = test_placements[0]["p"]
        test_top3 = [p["p"] for p in test_placements[:3]]
        
        # Top-1 agreement
        if gt_best == test_best:
            top1_agree += 1
        
        # Top-3 agreement (does GT's best appear in test's top 3?)
        if gt_best in test_top3:
            top3_agree += 1
        
        # Build EV dicts for matching placements
        gt_dict = placement_ev_dict(gt)
        test_dict = placement_ev_dict(test)
        common_placements = sorted(set(gt_dict.keys()) & set(test_dict.keys()))
        
        if len(common_placements) >= 5:
            gt_evs = [gt_dict[p] for p in common_placements]
            test_evs = [test_dict[p] for p in common_placements]
            
            # Spearman rank correlation
            rho, _ = stats.spearmanr(gt_evs, test_evs)
            rho_list.append(rho)
            
            # MAE
            mae = np.mean(np.abs(np.array(gt_evs) - np.array(test_evs)))
            mae_list.append(mae)
            
            # Bias (systematic over/under estimation)
            bias = np.mean(np.array(test_evs) - np.array(gt_evs))
            ev_bias_list.append(bias)
    
    return {
        "n_hands": n_hands,
        "top1_rate": top1_agree / n_hands,
        "top3_rate": top3_agree / n_hands,
        "rho_mean": np.mean(rho_list) if rho_list else float("nan"),
        "mae_mean": np.mean(mae_list) if mae_list else float("nan"),
        "bias_mean": np.mean(ev_bias_list) if ev_bias_list else float("nan"),
    }

def main():
    # Find all available configs
    configs = sorted([f.stem for f in BENCH_DIR.glob("*.jsonl") if f.stat().st_size > 0])
    
    if not configs:
        print("No benchmark data found in", BENCH_DIR)
        return
    
    print(f"Found {len(configs)} configs: {configs}")
    print()
    
    # Select ground truth
    gt_label = None
    for candidate in GT_PRIORITY:
        if candidate in configs:
            gt_label = candidate
            break
    
    if gt_label is None:
        # Use the config with highest samples
        gt_label = max(configs, key=lambda c: load_data(c)[0]["n_samples"] if load_data(c) else 0)
    
    gt_data = load_data(gt_label)
    if not gt_data:
        print(f"Ground truth {gt_label} has no data!")
        return
    
    print(f"Ground Truth: {gt_label} ({len(gt_data)} hands, samples={gt_data[0]['n_samples']}, nesting={gt_data[0]['nesting']})")
    print()
    
    # Show GT top placements per hand
    print("=" * 80)
    print("Ground Truth Best Placements:")
    print("=" * 80)
    for h in gt_data:
        hi = h["hand_idx"]
        ps = h["placements"]
        print(f"  Hand #{hi}: {h['hand']} ({h['type']})")
        print(f"    #1 {ps[0]['p']}  EV={ps[0]['ev']:+.3f}")
        if len(ps) > 1:
            print(f"    #2 {ps[1]['p']}  EV={ps[1]['ev']:+.3f}  (gap={ps[0]['ev']-ps[1]['ev']:.3f})")
    print()
    
    # Load timing data
    timings = load_timing()
    
    # Compare all configs vs GT
    print("=" * 80)
    print("Comparison vs Ground Truth")
    print("=" * 80)
    print(f"{'Config':<16} {'Samples':>7} {'Top1%':>6} {'Top3%':>6} {'Spear.ρ':>8} {'MAE':>6} {'Bias':>7} {'s/hand':>7} {'Cost':>6}")
    print("-" * 80)
    
    results = []
    for label in configs:
        data = load_data(label)
        if not data:
            continue
        
        if label == gt_label:
            # Self-comparison is trivially perfect
            timing = timings.get(label, {})
            per_hand = timing.get("per_hand_s", float("nan"))
            samples = data[0]["n_samples"]
            print(f"{label:<16} {samples:>7} {'100%':>6} {'100%':>6} {'1.000':>8} {'0.000':>6} {'0.000':>7} {per_hand:>7.1f} {'GT':>6}")
            continue
        
        comp = compare_configs(gt_data, data)
        timing = timings.get(label, {})
        per_hand = timing.get("per_hand_s", float("nan"))
        samples = data[0]["n_samples"]
        
        # "Cost" = relative cost vs cheapest (for easy comparison)
        results.append({
            "label": label,
            "samples": samples,
            "nesting": data[0]["nesting"],
            **comp,
            "per_hand_s": per_hand,
        })
        
        print(f"{label:<16} {samples:>7} {comp['top1_rate']:>5.0%} {comp['top3_rate']:>5.0%} "
              f"{comp['rho_mean']:>8.3f} {comp['mae_mean']:>6.2f} {comp['bias_mean']:>+7.2f} {per_hand:>7.1f}")
    
    print()
    
    # Summary recommendations
    if results:
        # Best accuracy
        best_rho = max(results, key=lambda r: r["rho_mean"] if not np.isnan(r["rho_mean"]) else -1)
        # Best speed
        valid_speed = [r for r in results if not np.isnan(r["per_hand_s"])]
        if valid_speed:
            fastest = min(valid_speed, key=lambda r: r["per_hand_s"])
        else:
            fastest = None
        # Best bang-for-buck: highest rho per second
        for r in results:
            if not np.isnan(r["rho_mean"]) and not np.isnan(r["per_hand_s"]) and r["per_hand_s"] > 0:
                r["efficiency"] = r["rho_mean"] / r["per_hand_s"]
            else:
                r["efficiency"] = 0
        best_efficiency = max(results, key=lambda r: r["efficiency"])
        
        print("=" * 80)
        print("RECOMMENDATIONS:")
        print(f"  Best accuracy:    {best_rho['label']}  (ρ={best_rho['rho_mean']:.3f}, Top1={best_rho['top1_rate']:.0%})")
        if fastest:
            print(f"  Fastest:          {fastest['label']}  ({fastest['per_hand_s']:.1f}s/hand)")
        print(f"  Best efficiency:  {best_efficiency['label']}  (ρ/s = {best_efficiency['efficiency']:.4f})")
        print("=" * 80)
    
    # Per-hand detail: show where configs disagree
    print()
    print("=" * 80)
    print("Per-Hand Top-1 Comparison (all configs):")
    print("=" * 80)
    
    for hi, gt_hand in enumerate(gt_data):
        gt_best = gt_hand["placements"][0]["p"]
        gt_ev = gt_hand["placements"][0]["ev"]
        print(f"\n  Hand #{gt_hand['hand_idx']}: {gt_hand['hand']} ({gt_hand['type']})")
        print(f"  GT Best: {gt_best}  EV={gt_ev:+.3f}")
        
        for label in configs:
            if label == gt_label:
                continue
            data = load_data(label)
            if not data or hi >= len(data):
                continue
            test_best = data[hi]["placements"][0]
            match = "✓" if test_best["p"] == gt_best else "✗"
            gt_rank = "?"
            for j, p in enumerate(data[hi]["placements"]):
                if p["p"] == gt_best:
                    gt_rank = str(j + 1)
                    break
            print(f"    {label:<16} {match} Best={test_best['p']}  EV={test_best['ev']:+.3f}  (GT's best @ rank {gt_rank})")

if __name__ == "__main__":
    main()
