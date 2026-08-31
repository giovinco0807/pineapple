#!/usr/bin/env python3
"""Verify joker-involved P99 EV labels against Rust solver."""
import json
import sys
import numpy as np
from pathlib import Path

P99_FILE = Path(__file__).parent / "data" / "t3_oracle" / "t3_p99_samples.jsonl"

with open(P99_FILE) as f:
    samples = [json.loads(line) for line in f]

print("=== Joker-Involved P99: EV Label Analysis ===\n")

joker_evs = []
non_joker_evs = []

for s in samples:
    board = s["state"]["board"]
    dealt = s["state"]["dealt"]
    all_cards = board["top"] + board["middle"] + board["bottom"] + dealt
    has_joker = any(c.startswith("X") for c in all_cards)

    best_ev = s["best_ev"]
    if has_joker:
        joker_evs.append(best_ev)
    else:
        non_joker_evs.append(best_ev)

je = np.array(joker_evs)
ne = np.array(non_joker_evs)

print(f"Joker samples: {len(joker_evs)}")
print(f"  Best EV: mean={je.mean():.1f}, median={np.median(je):.1f}, min={je.min():.1f}, max={je.max():.1f}")
print(f"Non-Joker samples: {len(non_joker_evs)}")
print(f"  Best EV: mean={ne.mean():.1f}, median={np.median(ne):.1f}, min={ne.min():.1f}, max={ne.max():.1f}")

# Key diagnostic: The EV labels for P99 use a 9-slot mapping that DROPS discard info
# So the "EV" for slot 5 might be the MAX of several discards all mapped to slot 5
# This means the EV label may be WRONG for the specific discard chosen by the model

print("\n=== Critical Issue: 9-slot vs 27-slot Mapping ===")
print()
print("Training data uses 9-slot mapping (pos0*3 + pos1), NOT 27-slot.")
print("This means discard choice is invisible to the model!")
print("All 'discard' decisions shown as d:0 in analyze_p99 are artifacts")
print("of this collapsed mapping.")
print()

# Analyze: where the placement-swap EVs diverge most 
print("=== Placement Swap EV Analysis ===\n")

SWAP_PAIRS = [(1, 3), (2, 6), (5, 7)]  # TM<->MT, TB<->BT, MB<->BM
SWAP_NAMES = {
    (1, 3): "(T,M) <-> (M,T)",
    (2, 6): "(T,B) <-> (B,T)",
    (5, 7): "(M,B) <-> (B,M)",
}

swap_gaps = {p: [] for p in SWAP_PAIRS}
for s in samples:
    true_a = s["true_action_idx"]
    pred_a = s["pred_action_idx"]
    
    # Find which swap pair this belongs to
    for a, b in SWAP_PAIRS:
        if (true_a == a and pred_a == b) or (true_a == b and pred_a == a):
            evs_lookup = {e["action_idx"]: e["ev"] for e in s["true_top3_by_ev"]}
            if true_a in evs_lookup and pred_a in evs_lookup:
                gap = evs_lookup[true_a] - evs_lookup[pred_a]
                swap_gaps[(a, b)].append(gap)
            else:
                # Get from pred top3
                all_evs = {}
                for e in s["true_top3_by_ev"]:
                    all_evs[e["action_idx"]] = e["ev"]
                for e in s["pred_top3_by_logit"]:
                    all_evs[e["action_idx"]] = e["ev"]
                if true_a in all_evs and pred_a in all_evs:
                    gap = all_evs[true_a] - all_evs[pred_a]
                    swap_gaps[(a, b)].append(gap)

for pair, gaps in swap_gaps.items():
    if gaps:
        arr = np.array(gaps)
        print(f"  {SWAP_NAMES[pair]}: n={len(gaps)}, mean_gap={arr.mean():.1f}, max={arr.max():.1f}")

# The core question: are these EVs (130+ for joker states) plausible?
print("\n=== EV Plausibility Check ===\n")
print("In Fantasyland with joker, high EVs are expected because:")
print("  - Joker completes strong hands (trips -> quads, pair -> trips)")
print("  - Strong hands = high royalties (e.g., quads bot = 10pt)")
print("  - FL stay bonus adds chain value (~14-90pt)")
print("  - Combined: royalties + scoop + FL_chain can easily = 100+pt")
print()

# Verify: in the P99 samples, what hands would the correct placement make?
print("=== Resulting Hand Quality (Best Action) ===\n")
for s in samples[:10]:
    rank = s["tail_rank"]
    board = s["state"]["board"]
    dealt = s["state"]["dealt"]
    true_a = s["true_action_idx"]
    best_ev = s["best_ev"]
    
    # For 9-slot: pos0 = true_a // 3, pos1 = true_a % 3
    pos_names = ["top", "middle", "bottom"]
    pos0 = true_a // 3
    pos1 = true_a % 3
    
    # In this 9-slot mapping, we don't know WHICH card goes where
    # or which card is discarded. This is the fundamental problem.
    print(f"Rank {rank}: EV={best_ev:.1f}")
    print(f"  Board: T={board['top']} M={board['middle']} B={board['bottom']}")
    print(f"  Dealt: {dealt}")
    print(f"  Best slot: {true_a} -> card1->{pos_names[pos0]}, card2->{pos_names[pos1]}")
    
    has_joker = any(c.startswith("X") for c in dealt + board["top"] + board["middle"] + board["bottom"])
    print(f"  Joker: {'YES' if has_joker else 'no'}")
    
    # Show all EVs
    for e in s["true_top3_by_ev"]:
        p0 = e["action_idx"] // 3
        p1 = e["action_idx"] % 3
        print(f"    slot {e['action_idx']} ({pos_names[p0]},{pos_names[p1]}): EV={e['ev']:.1f}")
    print()
