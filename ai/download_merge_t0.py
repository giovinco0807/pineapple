#!/usr/bin/env python3
"""
Download and merge T0 training data from GCS.

Downloads all worker outputs from gs://ofc-solver-485418/t0_training/
and merges them into a single JSONL file, deduplicating by hand content.

Usage:
    python ai/download_merge_t0.py
    python ai/download_merge_t0.py --bucket gs://ofc-solver-485418/t0_training/ --output d:/ofc_data/t0_training/merged.jsonl
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import OrderedDict


def download_from_gcs(gcs_path: str, local_dir: str):
    """Download files from GCS using gsutil."""
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading from {gcs_path} to {local_dir}...")
    try:
        result = subprocess.run(
            ['gsutil', '-m', 'cp', f'{gcs_path}*.jsonl', local_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  gsutil error: {result.stderr}")
            return False
        print(f"  Download complete.")
        return True
    except FileNotFoundError:
        print("  ERROR: gsutil not found. Install Google Cloud SDK.")
        return False
    except subprocess.TimeoutExpired:
        print("  ERROR: Download timed out.")
        return False


def merge_jsonl_files(input_dir: str, output_path: str, include_local: str = None):
    """Merge all JSONL files, deduplicating by hand content."""
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.jsonl"))

    if include_local:
        local = Path(include_local)
        if local.exists() and local not in files:
            files.append(local)

    print(f"\n  Merging {len(files)} files:")
    for f in files:
        print(f"    {f.name}")

    # Deduplicate by hand string (cards dealt)
    seen_hands = OrderedDict()
    n_total = 0
    n_dupes = 0

    for filepath in files:
        with open(filepath, 'r') as f:
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
                if hand_key in seen_hands:
                    n_dupes += 1
                    continue
                seen_hands[hand_key] = line

    # Write merged output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w') as f:
        for line in seen_hands.values():
            f.write(line + '\n')

    n_unique = len(seen_hands)
    print(f"\n  Results:")
    print(f"    Total lines: {n_total}")
    print(f"    Duplicates:  {n_dupes}")
    print(f"    Unique hands: {n_unique}")
    print(f"    Saved to: {output_path}")

    return n_unique


def main():
    parser = argparse.ArgumentParser(description="Download and merge T0 data from GCS")
    parser.add_argument('--bucket', default='gs://ofc-solver-485418/t0_training/',
                        help='GCS bucket path')
    parser.add_argument('--local-dir', default='d:/ofc_data/t0_training/',
                        help='Local directory for downloaded files')
    parser.add_argument('--local-file', default=None,
                        help='Additional local JSONL to include in merge')
    parser.add_argument('--output', default='d:/ofc_data/t0_training/merged.jsonl',
                        help='Output merged JSONL file')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip GCS download, merge existing local files')
    args = parser.parse_args()

    print("=" * 60)
    print("  T0 Training Data: Download & Merge")
    print("=" * 60)

    if not args.skip_download:
        success = download_from_gcs(args.bucket, args.local_dir)
        if not success:
            print("\n  Download failed. Trying to merge existing local files...")

    # Also include local generation file if present
    local_file = args.local_file
    if local_file is None:
        # Check common local output paths
        candidates = [
            'd:/ofc_data/t0_training/t0_n321_s50_500h.jsonl',
            'd:/ofc_data/t0_local.jsonl',
        ]
        for c in candidates:
            if Path(c).exists():
                local_file = c
                break

    n = merge_jsonl_files(args.local_dir, args.output, include_local=local_file)

    if n > 0:
        print(f"\n  Next step: Convert to NPZ:")
        print(f"    python ai/convert_t0_to_npz.py --input {args.output} --output d:/ofc_data/t0_training.npz --stats")
        print(f"\n  Then train:")
        print(f"    python ai/train_value.py --data d:/ofc_data/t0_training.npz --epochs 200 --save ai/models/value_t0")


if __name__ == '__main__':
    main()
