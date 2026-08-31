#!/usr/bin/env python3
"""
Quantify the EV distortion caused by 9-slot collapse in joker states.

For each T3 state in the exhaustive data, compare:
  - 9-slot EV: max(EV) across all discards per placement slot
  - 27-slot EV: true per-action (discard+placement) EV

If different discards give wildly different EVs for the same slot,
the 9-slot collapse creates misleading training labels.
"""
import json
import sys
import numpy as np
from pathlib import Path

DATA_FILE = Path("D:/ofc_data/t3_exhaustive/t3_all.jsonl")

# Arrow chars used in Rust output
ROW_MAP = {"T": 0, "M": 1, "B": 2}
POS_NAMES = ["top", "mid", "bot"]

def parse_action_desc(desc):
    """Parse 'd:Ks 8h→T 8s→M' into (discard, [(card, row_idx)])."""
    # Handle unicode arrows
    desc = desc.replace("\u7aca\u6a3d", "→M").replace("\u7aca\u63a2", "→T").replace("\u7aca\u5e95", "→B")
    desc = desc.replace("→", "->")
    
    parts = desc.split()
    discard = parts[0][2:]  # d:Ks -> Ks
    placements = []
    for p in parts[1:]:
        if "->" in p:
            card, row = p.split("->")
            row_idx = ROW_MAP.get(row[0], -1)
            placements.append((card, row_idx))
    return discard, placements

def get_9slot(placements):
    """Convert placements to 9-slot index: pos0*3 + pos1."""
    if len(placements) != 2:
        return -1
    return placements[0][1] * 3 + placements[1][1]


n_states = 0
n_joker_states = 0
n_states_with_distortion = 0
distortions = []
max_distortion_examples = []

with open(DATA_FILE, encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 50000:
            break
        
        data = json.loads(line)
        for cr in data.get("t3_records", []):
            deal = cr.get("deal", [])
            all_board = cr.get("top", []) + cr.get("mid", []) + cr.get("bot", []) + deal
            has_joker = any(c.startswith("X") for c in all_board)
            
            actions = cr.get("actions", [])
            if not actions:
                continue
            
            n_states += 1
            if has_joker:
                n_joker_states += 1
            
            # Group actions by 9-slot
            slot_evs = {}  # slot -> [ev1, ev2, ev3]
            for a in actions:
                try:
                    discard, placements = parse_action_desc(a["desc"])
                    slot = get_9slot(placements)
                    if slot < 0:
                        continue
                    if slot not in slot_evs:
                        slot_evs[slot] = []
                    slot_evs[slot].append(a["ev"])
                except Exception:
                    continue
            
            # Check distortion: for slots with multiple EVs, how much do they vary?
            max_slot_range = 0
            for slot, evs in slot_evs.items():
                if len(evs) > 1:
                    ev_range = max(evs) - min(evs)
                    max_slot_range = max(max_slot_range, ev_range)
            
            if max_slot_range > 1.0:
                n_states_with_distortion += 1
                distortions.append((max_slot_range, has_joker))
                
                if max_slot_range > 50 and len(max_distortion_examples) < 5:
                    max_distortion_examples.append({
                        "top": cr["top"],
                        "mid": cr["mid"],
                        "bot": cr["bot"],
                        "deal": deal,
                        "slot_evs": {str(k): v for k, v in slot_evs.items()},
                        "max_range": max_slot_range,
                        "has_joker": has_joker,
                    })

print(f"=== 9-Slot Collapse Distortion Analysis ===")
print(f"Total T3 states analyzed: {n_states}")
print(f"Joker states: {n_joker_states} ({n_joker_states/max(n_states,1)*100:.1f}%)")
print(f"States with EV distortion (>1pt): {n_states_with_distortion} ({n_states_with_distortion/max(n_states,1)*100:.1f}%)")

if distortions:
    all_d = np.array([d[0] for d in distortions])
    joker_d = np.array([d[0] for d in distortions if d[1]])
    nojoker_d = np.array([d[0] for d in distortions if not d[1]])
    
    print(f"\nDistortion magnitude (all):")
    print(f"  mean={all_d.mean():.1f}, median={np.median(all_d):.1f}, max={all_d.max():.1f}")
    print(f"  >10pt: {(all_d > 10).sum()}")
    print(f"  >50pt: {(all_d > 50).sum()}")
    print(f"  >100pt: {(all_d > 100).sum()}")
    
    if len(joker_d) > 0:
        print(f"\nWith Joker (n={len(joker_d)}):")
        print(f"  mean={joker_d.mean():.1f}, median={np.median(joker_d):.1f}, max={joker_d.max():.1f}")
    
    if len(nojoker_d) > 0:
        print(f"\nWithout Joker (n={len(nojoker_d)}):")
        print(f"  mean={nojoker_d.mean():.1f}, median={np.median(nojoker_d):.1f}, max={nojoker_d.max():.1f}")

print(f"\n=== Extreme Distortion Examples ===")
for ex in max_distortion_examples:
    print(f"\nJoker: {ex['has_joker']} | Max range: {ex['max_range']:.1f}")
    print(f"  Top: {ex['top']}, Mid: {ex['mid']}, Bot: {ex['bot']}")
    print(f"  Dealt: {ex['deal']}")
    for slot_s, evs in ex["slot_evs"].items():
        slot = int(slot_s)
        p0, p1 = POS_NAMES[slot // 3], POS_NAMES[slot % 3]
        ev_strs = [f"{e:.1f}" for e in sorted(evs, reverse=True)]
        print(f"    slot {slot} ({p0},{p1}): EVs = {ev_strs}")
