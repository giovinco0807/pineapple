#!/usr/bin/env python3
"""
T1 Training Data Generator using Rust engine for fast imperfect-information Monte Carlo.

Pipeline:
1. Python generates N hands of 5 cards (T0).
2. Uses T0 NN to prune 232 placements -> Top 50.
3. For each top T0 placement, sample `n1` random T1 hands (3 cards).
4. Writes all (T0 board, T1 hand) combinations to a temporary JSONL file.
5. Invokes the Rust `cfr_solver turn-batch-imperfect` to evaluate all combinations.
6. Reformats the Rust output into final training data format.

Usage:
    python ai/generate_t1_mc_rust.py --n-hands 1000 --n1 10 --nesting 5,2
"""
import argparse, json, random, time, sys, os, subprocess
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))
from train_t0_placement import T0PlacementNet, encode_card, compute_hand_features
from generate_t1_mc import (
    ALL_CARDS_STR, RANK_VALUES, SUIT_NAMES, ROW_NAMES,
    ALL_T0_PLACEMENTS, T0_PLACEMENT_TENSOR, card_str_to_dict, t0_nn_top50, Board
)

def build_jsonl_tasks(args, start_idx, end_idx):
    """Generate (board, hand) tasks for Rust."""
    tasks = []
    for hand_idx in range(start_idx, end_idx):
        seed = args.seed + hand_idx * 1000
        rng = random.Random(seed)
        deck = list(ALL_CARDS_STR)
        rng.shuffle(deck)
        t0_hand = deck[:5]
        remaining = deck[5:]
        
        top50 = t0_nn_top50(t0_hand)
        if not top50: continue
        
        for t1_deal_idx in range(args.n1):
            rng.shuffle(remaining)
            t1_hand = remaining[:3]
            
            for ci, t0_placement in enumerate(top50):
                board = Board()
                for i, row in enumerate(t0_placement):
                    board.place(t0_hand[i], row)
                
                t0_fmt = ", ".join(f"{t0_hand[i]}->{ROW_NAMES[t0_placement[i]]}" for i in range(5))
                
                tasks.append({
                    "hand_idx": hand_idx,
                    "t1_deal": t1_deal_idx,
                    "turn": 1,
                    "board": {
                        "top": [card_str_to_dict(c) for c in board.top],
                        "mid": [card_str_to_dict(c) for c in board.mid],
                        "bot": [card_str_to_dict(c) for c in board.bot],
                    },
                    "hand": [card_str_to_dict(c) for c in t1_hand],
                    "_meta": {
                        "t0_hand": " ".join(t0_hand),
                        "t1_hand": " ".join(t1_hand),
                        "t0_p": t0_fmt,
                        "t0_idx": ci
                    }
                })
    return tasks

def main():
    ap = argparse.ArgumentParser(description="T1 Data Generator with Rust Engine")
    ap.add_argument('--n-hands', type=int, default=1000)
    ap.add_argument('--n1', type=int, default=10, help="T1 deal samples per T0 hand")
    ap.add_argument('--samples', type=int, default=300, help="Monte Carlo samples per action inside Rust")
    ap.add_argument('--nesting', type=str, default="5,2", help="Nesting levels for T2, T3 (e.g., '5,2')")
    ap.add_argument('--output', type=str, default='t1_mc_rust.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tmp-in', type=str, default='t1_rust_in.jsonl')
    ap.add_argument('--tmp-out', type=str, default='t1_rust_out.jsonl')
    args = ap.parse_args()

    print(f"=== T1 Generator (Rust-Accelerated) ===")
    print(f"  Hands:    {args.n_hands:,}")
    print(f"  T1 deals: {args.n1} per hand")
    print(f"  Nesting:  {args.nesting}")
    print(f"  Output:   {args.output}")
    print(flush=True)

    t0 = time.time()
    
    # 1. Generate tasks and write to temp input
    print("Generating T0/T1 tasks...")
    tasks = build_jsonl_tasks(args, 0, args.n_hands)
    print(f"Total tasks generated: {len(tasks):,}")
    
    with open(args.tmp_in, 'w') as f:
        for t in tasks:
            f.write(json.dumps(t) + '\n')
            
    # 2. Run Rust solver
    rust_dir = Path(__file__).parent.parent / "rust_solver"
    cargo_cmd = [
        "cargo", "run", "--release", "--bin", "cfr_solver", "--",
        "turn-batch-imperfect",
        "--input", os.path.abspath(args.tmp_in),
        "--output", os.path.abspath(args.tmp_out),
        "--samples", str(args.samples),
        "--nesting", args.nesting,
        "--seed", str(args.seed)
    ]
    
    print(f"\nLaunching Rust solver...")
    print(f"Command: {' '.join(cargo_cmd)}")
    
    if os.path.exists(args.tmp_out):
        os.remove(args.tmp_out)
        
    rust_start = time.time()
    # Popen to capture and print output in real-time
    process = subprocess.Popen(cargo_cmd, cwd=rust_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    for line in process.stdout:
        print(f"[Rust] {line}", end="", flush=True)
    process.wait()
    
    if process.returncode != 0:
        print(f"Rust solver failed with code {process.returncode}")
        return

    print(f"Rust execution finished in {time.time() - rust_start:.1f}s")
    
    # 3. Process output and reformat
    print(f"\nFormatting output to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Read the output tasks back, matching them by line number (assuming 1:1)
    written = 0
    with open(args.tmp_out, 'r') as fr, open(args.output, 'w') as fw:
        for in_task, out_line in zip(tasks, fr):
            out_res = json.loads(out_line)
            meta = in_task["_meta"]
            
            # Reformat to match expected training format
            record = {
                'hand_idx': in_task["hand_idx"],
                't1_deal': in_task["t1_deal"],
                'turn': 1,
                't0_hand': meta["t0_hand"],
                't1_hand': meta["t1_hand"],
                't0_p': meta["t0_p"],
                't0_idx': meta["t0_idx"],
                'n_placements': out_res["n_placements"],
                'placements': out_res["placements"]
            }
            fw.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1

    el = time.time() - t0
    print(f"\n=== Done === {el:.0f}s ({el/3600:.1f}h) | {written:,} records | {args.output}")

if __name__ == '__main__':
    main()
