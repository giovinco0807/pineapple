#!/usr/bin/env python3
"""
T2 Dataset Generator (using T3PolicyValueNet)

Generates T2 states and evaluates actions using the trained T3 oracle.
Instead of an exact solver, it takes the expectation over N randomly
sampled T3 deals (Monte Carlo), evaluated by the T3 Value Network.

Usage:
  python generate_t2_data.py --states 50000 --workers 8 --samples 50
"""
import os
import sys
import json
import time
import argparse
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix path for imports
AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))

from engine.encoding import Board, Observation, encode_state, STATE_DIM
from engine.action_space import get_turn_actions, encode_action, Action
from training.train_t3_oracle import T3PolicyValueNet

ACTION_DIM = 27  # Fixed canonical action space


def generate_random_t2_state():
    """Generate a valid pseudo-random T2 state."""
    RANKS = "23456789TJQKA"
    SUITS = "shdc"
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    random.shuffle(deck)
    
    board_top = []
    board_mid = []
    board_bot = []
    
    # T0 (5 cards)
    t0_cards = sorted([deck.pop() for _ in range(5)], 
                      key=lambda c: RANKS.index(c[0]))
    
    strategy = random.choice(["spread", "bot_heavy", "mid_heavy", "random"])
    
    if strategy == "spread":
        board_top.append(t0_cards[4])
        board_mid.extend(t0_cards[2:4])
        board_bot.extend(t0_cards[0:2])
    elif strategy == "bot_heavy":
        board_bot.extend(t0_cards[:3])
        board_mid.extend(t0_cards[3:5])
    elif strategy == "mid_heavy":
        board_mid.extend(t0_cards[:3])
        board_bot.extend(t0_cards[3:5])
    else:
        positions = ["top", "mid", "bot"]
        limits = {"top": 3, "mid": 5, "bot": 5}
        counts = {"top": 0, "mid": 0, "bot": 0}
        for card in t0_cards:
            valid = [p for p in positions if counts[p] < limits[p]]
            pos = random.choice(valid)
            if pos == "top": board_top.append(card)
            elif pos == "mid": board_mid.append(card)
            else: board_bot.append(card)
            counts[pos] += 1
            
    discards = []
    
    # T1 (Draw 3, play 2, discard 1)
    t1_cards = [deck.pop() for _ in range(3)]
    discard_idx = random.randint(0, 2)
    discards.append(t1_cards[discard_idx])
    remaining = [t1_cards[i] for i in range(3) if i != discard_idx]
    
    for card in remaining:
        valid = []
        if len(board_top) < 3: valid.append("top")
        if len(board_mid) < 5: valid.append("mid")
        if len(board_bot) < 5: valid.append("bot")
        
        if not valid:
            break
        pos = random.choice(valid)
        if pos == "top": board_top.append(card)
        elif pos == "mid": board_mid.append(card)
        else: board_bot.append(card)
            
    total = len(board_top) + len(board_mid) + len(board_bot)
    if total != 7:
        return None
        
    # Deal T2 cards
    t2_cards = [deck.pop() for _ in range(3)]
    
    return {
        "board_top": board_top,
        "board_mid": board_mid,
        "board_bot": board_bot,
        "discards": discards,
        "dealt": t2_cards,
        "deck": deck
    }


def save_chunk(results, output_dir, worker_id, chunk_id):
    if not results:
        return
        
    states_np = np.array([r[0] for r in results], dtype=np.float32)
    evs_np = np.array([r[1] for r in results], dtype=np.float32)
    masks_np = np.array([r[2] for r in results], dtype=bool)
    best_evs_np = np.array([r[3] for r in results], dtype=np.float32)
    
    timestamp = int(time.time())
    output_file = Path(output_dir) / f"t2_data_{timestamp}_w{worker_id:02d}_c{chunk_id:04d}.npz"
    np.savez_compressed(
        output_file,
        states=states_np,
        action_evs=evs_np,
        action_masks=masks_np,
        best_evs=best_evs_np,
    )
    print(f"  [Worker {worker_id}] Saved chunk {chunk_id} to {output_file.name} ({len(results)} states)")


def worker_process(args):
    worker_id, num_states, model_path, output_dir, save_interval, num_samples, device_name = args
    
    # Initialize torch device
    device = torch.device(device_name)
    torch.set_grad_enabled(False)
    
    ck = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        state_dim = ck.get('state_dim', STATE_DIM)
        n_actions = ck.get('n_actions', ACTION_DIM)
        hidden = ck.get('hidden', 1024)
        n_blocks = ck.get('n_blocks', 4)
        model = T3PolicyValueNet(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden=hidden,
            n_blocks=n_blocks,
        )
    else:
        state_dim = STATE_DIM
        model = T3PolicyValueNet(state_dim=state_dim)
    if 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    else:
        model.load_state_dict(ck)
    model.eval()
    model.to(device)
    
    results = []
    chunk_id = 0
    total_processed = 0
    
    # Set random seed for worker
    random.seed(int(time.time()) + worker_id * 1000)
    np.random.seed(int(time.time()) + worker_id * 1000)
    
    # Loop
    while total_processed < num_states:
        state = generate_random_t2_state()
        if not state:
            continue
            
        top, mid, bot = state["board_top"], state["board_mid"], state["board_bot"]
        if len(top) > 3 or len(mid) > 5 or len(bot) > 5:
            continue
        if len(top) + len(mid) + len(bot) != 7:
            continue
            
        board_obj = Board(top=top, middle=mid, bottom=bot)
        dealt = state["dealt"]
        
        try:
            valid_actions = get_turn_actions(dealt, board_obj)
        except Exception:
            continue
            
        if not valid_actions:
            continue
            
        # Base observation for the T2 state
        obs = Observation(
            board_self=board_obj,
            board_opponent=Board(),
            dealt_cards=dealt,
            known_discards_self=state["discards"],
            turn=2,
            is_btn=False,
        )
        base_tensor = encode_state(obs)[:state_dim]
        
        # Evaluate each valid action
        action_evs = np.full(ACTION_DIM, -1e9, dtype=np.float32)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        
        deck = state["deck"]
        
        # We can batch T3 evaluations for speed
        batch_obs = []
        batch_indices = []
        
        for action in valid_actions:
            try:
                a_idx = encode_action(action, valid_actions, turn=2, dealt_cards=dealt)
                if not (0 <= a_idx < ACTION_DIM):
                    continue
            except Exception:
                continue
                
            # Apply action
            new_board = Board(
                top=list(board_obj.top),
                middle=list(board_obj.middle),
                bottom=list(board_obj.bottom)
            )
            for card, row in action.placements:
                getattr(new_board, row).append(card)
                
            new_discards = state["discards"] + [action.discard] if action.discard else state["discards"]
            
            # Sample T3 deals
            for _ in range(num_samples):
                t3_dealt = random.sample(deck, 3)
                
                t3_obs = Observation(
                    board_self=new_board,
                    board_opponent=Board(),
                    dealt_cards=t3_dealt,
                    known_discards_self=new_discards,
                    turn=3,
                    is_btn=False
                )
                batch_obs.append(encode_state(t3_obs)[:state_dim])
                batch_indices.append(a_idx)
                
        if not batch_obs:
            continue
            
        # Run inference in chunks to avoid OOM if num_samples is huge
        CHUNK_SIZE = 512
        all_values = []
        
        for i in range(0, len(batch_obs), CHUNK_SIZE):
            chunk = batch_obs[i:i+CHUNK_SIZE]
            t_states = torch.tensor(np.array(chunk, dtype=np.float32), device=device)
            _, v = model(t_states)
            all_values.extend(v.cpu().numpy())
            
        # Aggregate EVs
        ev_sum = {idx: 0.0 for idx in set(batch_indices)}
        ev_count = {idx: 0 for idx in set(batch_indices)}
        
        for v, a_idx in zip(all_values, batch_indices):
            ev_sum[a_idx] += float(v)
            ev_count[a_idx] += 1
            
        for a_idx, count in ev_count.items():
            if count > 0:
                action_evs[a_idx] = ev_sum[a_idx] / count
                mask[a_idx] = True
                
        if not mask.any():
            continue
            
        best_ev = float(np.max(action_evs[mask]))
        results.append((base_tensor, action_evs, mask, best_ev))
        total_processed += 1
        
        # Save chunk
        if len(results) >= save_interval:
            save_chunk(results, output_dir, worker_id, chunk_id)
            chunk_id += 1
            results = []
            
    # Save remaining
    if results:
        save_chunk(results, output_dir, worker_id, chunk_id)
        
    return total_processed


def main():
    parser = argparse.ArgumentParser(description="Generate T2 Dataset with T3 Model")
    parser.add_argument("--states", type=int, default=50000, help="Total states to generate")
    parser.add_argument("--workers", type=int, default=-1, help="Worker processes (-1 = auto)")
    parser.add_argument("--save-interval", type=int, default=2000, help="Save to NPZ every N states per worker")
    parser.add_argument("--output-dir", type=str, default=str(AI_DIR / "data" / "t2_oracle"), help="Output directory")
    parser.add_argument("--model", type=str, default=str(AI_DIR / "data" / "t3_oracle" / "t3_policyvalue_model.pt"), help="T3 Model path")
    parser.add_argument("--samples", type=int, default=30, help="Monte Carlo samples per action")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)
        
    total = args.states
    num_workers = args.workers if args.workers > 0 else min(8, max(1, mp.cpu_count() - 1))
    
    # If CUDA, maybe reduce workers or rely on CUDA multiprocessing support (spawn)
    if args.device == "cuda":
        mp.set_start_method("spawn", force=True)
        # Using multiple workers on a single GPU can cause OOM. Be careful.
        num_workers = min(num_workers, 4)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== T2 Dataset Generator (T3 Oracle) ===")
    print(f"Target: {total} states | Workers: {num_workers} | Interval: {args.save_interval}")
    print(f"Model: {args.model}")
    print(f"MC Samples per action: {args.samples}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")
    
    start_time = time.time()
    
    base = total // num_workers
    remainder = total % num_workers
    pool_args = []
    for i in range(num_workers):
        n = base + (1 if i < remainder else 0)
        if n > 0:
            pool_args.append((i, n, args.model, str(output_dir), args.save_interval, args.samples, args.device))
            
    if num_workers == 1:
        total_generated = sum([worker_process(pool_args[0])])
    else:
        with mp.Pool(num_workers) as pool:
            results = pool.map(worker_process, pool_args)
            total_generated = sum(results)
            
    elapsed = time.time() - start_time
    print(f"\n=== Generation Complete ===")
    print(f"Total Generated: {total_generated}")
    print(f"Time: {elapsed:.1f}s ({total_generated/elapsed:.1f} states/sec)")
    print(f"All chunks saved to: {output_dir}")


if __name__ == "__main__":
    main()
