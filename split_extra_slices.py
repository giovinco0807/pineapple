#!/usr/bin/env python3
"""Split extra 1000 hands into 10 slices and upload to GCS."""
import json

INPUT = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_phase_e_extra.json"
OUTPUT_DIR = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\slices_extra"

with open(INPUT) as f:
    data = json.load(f)

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

n_slices = 10
chunk = len(data) // n_slices  # 100 hands per slice

for i in range(n_slices):
    start = i * chunk
    end = start + chunk
    slice_data = data[start:end]
    # Re-index
    for j, entry in enumerate(slice_data):
        entry["hand_idx"] = j
    
    slice_id = 20 + i  # VMs 20-29
    out_path = os.path.join(OUTPUT_DIR, f"phase_e_slice_{slice_id:02d}.json")
    with open(out_path, "w") as f:
        json.dump(slice_data, f)
    print(f"  Slice {slice_id}: {len(slice_data)} hands → {out_path}")

print(f"\nDone: {n_slices} slices of {chunk} hands each")
