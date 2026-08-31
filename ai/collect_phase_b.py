#!/usr/bin/env python3
"""
Download Phase B T0 data from GCS, merge with Phase 1, convert to ranking format.

Usage:
    # Download and convert all available data
    python ai/collect_phase_b.py
    
    # Skip download, use existing local files
    python ai/collect_phase_b.py --skip-download
    
    # Include Phase 1 data
    python ai/collect_phase_b.py --include-phase1
"""

import json
import subprocess
import argparse
from pathlib import Path
from collections import OrderedDict


def download_gcs(gcs_path: str, local_dir: str) -> bool:
    """Download JSONL files from GCS."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from {gcs_path}...")
    try:
        result = subprocess.run(
            ['gsutil', '-m', 'cp', f'{gcs_path}worker_*.jsonl', local_dir],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  gsutil error: {result.stderr[:200]}")
            return False
        print(f"  Download complete.")
        return True
    except FileNotFoundError:
        print("  ERROR: gsutil not found.")
        return False
    except subprocess.TimeoutExpired:
        print("  ERROR: Download timed out.")
        return False


def merge_jsonl(dirs: list, output_path: str, local_file: str = None) -> int:
    """Merge JSONL files from multiple directories, deduplicating by hand."""
    files = []
    for d in dirs:
        p = Path(d)
        if p.exists():
            files.extend(sorted(p.glob("*.jsonl")))
    
    if local_file:
        lf = Path(local_file)
        if lf.exists() and lf not in files:
            files.append(lf)
    
    print(f"\n  Merging {len(files)} files:")
    for f in files:
        print(f"    {f.name}")
    
    seen = OrderedDict()
    n_total = 0
    n_dupes = 0
    
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                n_total += 1
                hand_key = data.get('hand', '')
                if hand_key in seen:
                    n_dupes += 1
                    continue
                seen[hand_key] = line
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in seen.values():
            f.write(line + '\n')
    
    n_unique = len(seen)
    print(f"\n  Results:")
    print(f"    Total lines:  {n_total}")
    print(f"    Duplicates:   {n_dupes}")
    print(f"    Unique hands: {n_unique}")
    print(f"    Saved to:     {output_path}")
    
    # Stats
    ev_values = []
    hand_types = {}
    for line in seen.values():
        data = json.loads(line)
        placements = data.get('placements', [])
        if placements:
            best_ev = placements[0]['ev']
            ev_values.append(best_ev)
        ht = data.get('type', 'unknown')
        hand_types[ht] = hand_types.get(ht, 0) + 1
    
    if ev_values:
        print(f"\n  EV Stats:")
        print(f"    Mean: {sum(ev_values)/len(ev_values):.2f}")
        print(f"    Min:  {min(ev_values):.2f}")
        print(f"    Max:  {max(ev_values):.2f}")
        fl_count = sum(1 for ev in ev_values if ev >= 7.0)
        print(f"    FL-likely (EV>=7): {fl_count}/{len(ev_values)} ({100*fl_count/len(ev_values):.1f}%)")
    
    print(f"\n  Hand Type Distribution (top 10):")
    for ht, count in sorted(hand_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ht}: {count}")
    
    return n_unique


def main():
    parser = argparse.ArgumentParser(description="Collect Phase B T0 data")
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip GCS download')
    parser.add_argument('--include-phase1', action='store_true', default=True,
                        help='Include Phase 1 data')
    parser.add_argument('--include-local', action='store_true', default=True,
                        help='Include local solver data')
    parser.add_argument('--output', default='d:/ofc_data/t0_merged_all.jsonl',
                        help='Output merged JSONL')
    parser.add_argument('--phase-b-dir', default='d:/ofc_data/t0_phase_b/',
                        help='Phase B local directory')
    parser.add_argument('--phase1-dir', default='d:/ofc_data/t0_training/',
                        help='Phase 1 local directory')
    parser.add_argument('--local-file', default='d:/ofc_data/t0_local_s20/local_s20.jsonl',
                        help='Local solver output file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  T0 Data Collection: Phase B + Phase 1 + Local")
    print("=" * 60)
    
    # Download Phase B data from GCS
    if not args.skip_download:
        download_gcs('gs://ofc-solver-485418/t0_phase_b/', args.phase_b_dir)
    
    # Merge directories
    dirs = [args.phase_b_dir]
    if args.include_phase1:
        dirs.append(args.phase1_dir)
        print(f"  Including Phase 1 data from {args.phase1_dir}")
    
    local_file = args.local_file if args.include_local else None
    
    n = merge_jsonl(dirs, args.output, local_file=local_file)
    
    if n > 0:
        print(f"\n  Next steps:")
        print(f"    1. Convert to BC format:")
        print(f"       python ai/training/convert_t0_cfr.py --input {args.output} --output ai/data/t0_all.jsonl --augment")
        print(f"    2. Convert to ranking NPZ:")
        print(f"       python ai/training/convert_t0_ranking.py --input {args.output} --output ai/data/ranked_t0_all/")
        print(f"    3. Train ranking model:")
        print(f"       python ai/training/train_ranking.py --data ai/data/ranked_t0_all/ --epochs 100 --size tiny")
    
    print()


if __name__ == '__main__':
    main()
