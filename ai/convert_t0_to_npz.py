"""
T0 JSONL → NPZ Converter for Value Network Training

Converts Rust solver's T0 evaluation data (JSONL) into NPZ format
compatible with train_value.py.

Each T0 hand has ~150 unique placements, each with a solver-computed EV.
For each placement, we create a post-placement board state and encode it
as a 522-dim observation vector. This yields ~150 training samples per hand.

Input JSONL format (from Rust solver):
    {"hand_idx": 0, "hand": "Tc 2c 9c 4s As", "placements": [
        {"p": "Top[As] Mid[2c 4s] Bot[Tc 9c]", "ev": 40.377}, ...
    ]}

Output NPZ format (for train_value.py):
    obs:      float32 (N, 522) — encoded board state after placement
    score:    float32 (N,)     — solver EV for this placement
    turn:     int8    (N,)     — always 0 (T0)
    busted:   bool    (N,)     — always False (T0 placements never bust)
    fl_entry: bool    (N,)     — whether top row has FL-qualifying cards

Usage:
    python ai/convert_t0_to_npz.py --input d:/ofc_data/t0_training/*.jsonl --output d:/ofc_data/t0_training.npz
    python ai/convert_t0_to_npz.py --input d:/ofc_data/t0_training/merged.jsonl --output d:/ofc_data/t0_training.npz --top-k 30
"""

import sys
import re
import json
import argparse
import glob
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.engine.encoding import Board, Observation, encode_state, RANK_VALUES


def parse_placement_string(p_str: str):
    """Parse placement string like 'Top[As] Mid[2c 4s] Bot[Tc 9c]'.

    Returns dict: {'top': ['As'], 'middle': ['2c', '4s'], 'bottom': ['Tc', '9c']}
    """
    result = {'top': [], 'middle': [], 'bottom': []}
    row_map = {'Top': 'top', 'Mid': 'middle', 'Bot': 'bottom'}

    pattern = r'(Top|Mid|Bot)\[([^\]]*)\]'
    for match in re.finditer(pattern, p_str):
        row_name = row_map[match.group(1)]
        cards_str = match.group(2).strip()
        if cards_str:
            result[row_name] = cards_str.split()

    return result


def check_fl_potential(top_cards):
    """Check if top row cards have FL entry potential (QQ+).

    Returns True if top contains a pair of Queens or better.
    """
    if not top_cards:
        return False

    ranks = []
    jokers = 0
    for c in top_cards:
        if c in ('X1', 'X2'):
            jokers += 1
        elif len(c) >= 2:
            r = c[:-1]
            if r in RANK_VALUES:
                ranks.append(RANK_VALUES[r])

    rank_counts = Counter(ranks)

    # Check for pairs of QQ, KK, AA (rank values Q=10, K=11, A=12)
    for r, cnt in rank_counts.items():
        if cnt + jokers >= 2 and r >= RANK_VALUES['Q']:
            return True
    # Trips of anything also qualifies for FL
    for r, cnt in rank_counts.items():
        if cnt + jokers >= 3:
            return True

    return False


def convert_hand(hand_data, top_k=0, min_ev=None):
    """Convert one JSONL hand entry to list of (obs_vec, ev, fl_entry) tuples.

    Args:
        hand_data: Parsed JSON dict from one JSONL line
        top_k: If > 0, only include top K placements by EV (reduces noise)
        min_ev: If set, exclude placements below this EV threshold

    Returns:
        List of (obs_vec, ev, fl_entry) tuples
    """
    hand_cards_str = hand_data['hand']
    hand_cards = hand_cards_str.split()
    placements = hand_data['placements']

    if top_k > 0:
        placements = placements[:top_k]  # Already sorted by EV descending

    if min_ev is not None:
        placements = [p for p in placements if p['ev'] >= min_ev]

    results = []
    for p in placements:
        placement = parse_placement_string(p['p'])
        ev = p['ev']

        # Create board with this placement
        board = Board(
            top=list(placement['top']),
            middle=list(placement['middle']),
            bottom=list(placement['bottom']),
        )

        # Create observation (post-placement state)
        obs = Observation(
            board_self=board,
            board_opponent=Board(),  # Empty opponent board
            dealt_cards=[],          # No cards in hand (already placed)
            known_discards_self=[],
            turn=0,
            is_btn=True,             # Doesn't matter much for T0
            is_fl=False,
            opp_is_fl=False,
        )

        obs_vec = encode_state(obs)
        fl_entry = check_fl_potential(placement['top'])
        results.append((obs_vec, ev, fl_entry))

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert T0 JSONL to NPZ for Value Network")
    parser.add_argument('--input', required=True, nargs='+',
                        help='Input JSONL files (supports glob patterns)')
    parser.add_argument('--output', required=True, help='Output NPZ file')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Only include top K placements per hand (0=all)')
    parser.add_argument('--min-ev', type=float, default=None,
                        help='Exclude placements below this EV')
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Max hands to process (0=all)')
    parser.add_argument('--stats', action='store_true',
                        help='Print detailed statistics')
    args = parser.parse_args()

    # Expand glob patterns
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        elif Path(pattern).exists():
            input_files.append(pattern)
    input_files = sorted(set(input_files))

    if not input_files:
        print(f"ERROR: No input files found for: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  T0 JSONL → NPZ Converter")
    print("=" * 60)
    print(f"  Input files: {len(input_files)}")
    for f in input_files:
        print(f"    {f}")
    print(f"  Top-K: {args.top_k or 'all'}")
    if args.min_ev is not None:
        print(f"  Min EV: {args.min_ev}")
    print(f"  Output: {args.output}")
    print()

    all_obs = []
    all_scores = []
    all_fl = []
    n_hands = 0
    n_skipped = 0
    hand_ids = set()

    for filepath in input_files:
        file_hands = 0
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARN: JSON error in {filepath}:{line_num}: {e}")
                    n_skipped += 1
                    continue

                # Deduplicate by hand_idx (across files from different workers)
                hand_key = data.get('hand', '') + str(data.get('hand_idx', ''))
                if hand_key in hand_ids:
                    continue
                hand_ids.add(hand_key)

                if args.max_hands > 0 and n_hands >= args.max_hands:
                    break

                records = convert_hand(data, top_k=args.top_k, min_ev=args.min_ev)
                for obs_vec, ev, fl_entry in records:
                    all_obs.append(obs_vec)
                    all_scores.append(ev)
                    all_fl.append(fl_entry)

                n_hands += 1
                file_hands += 1

        print(f"  {Path(filepath).name}: {file_hands} hands")

    if not all_obs:
        print("ERROR: No data collected!")
        sys.exit(1)

    # Convert to arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    score_array = np.array(all_scores, dtype=np.float32)
    turn_array = np.zeros(len(all_obs), dtype=np.int8)        # Always T0
    busted_array = np.zeros(len(all_obs), dtype=np.bool_)     # T0 never busts
    fl_array = np.array(all_fl, dtype=np.bool_)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=obs_array,
        score=score_array,
        turn=turn_array,
        busted=busted_array,
        fl_entry=fl_array,
    )

    print(f"\n{'=' * 60}")
    print(f"  Conversion Complete")
    print(f"{'=' * 60}")
    print(f"  Hands: {n_hands}")
    print(f"  Samples: {len(obs_array):,}")
    print(f"  Obs shape: {obs_array.shape}")
    print(f"  Avg placements/hand: {len(obs_array)/max(n_hands,1):.1f}")
    print(f"  Score mean: {score_array.mean():.2f}")
    print(f"  Score std:  {score_array.std():.2f}")
    print(f"  Score range: [{score_array.min():.1f}, {score_array.max():.1f}]")
    print(f"  FL potential: {fl_array.sum():,} ({fl_array.mean()*100:.1f}%)")
    print(f"  Saved to: {args.output}")

    if args.stats:
        print(f"\n  --- Score Distribution ---")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            v = np.percentile(score_array, p)
            print(f"    P{p:2d}: {v:7.2f}")

        print(f"\n  --- EV Bins ---")
        bins = [-100, -10, 0, 10, 20, 30, 40, 50, 100]
        hist, _ = np.histogram(score_array, bins=bins)
        for i in range(len(bins) - 1):
            pct = hist[i] / len(score_array) * 100
            print(f"    [{bins[i]:4d}, {bins[i+1]:4d}): {hist[i]:6d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
