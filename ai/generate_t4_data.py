#!/usr/bin/env python3
"""
T4 Dataset Generator

Generates T4 game states by simulating the game up to T3 using a heuristic,
and then making the T3 placements using a trained T3PolicyValueNet model.
Then it evaluates the resulting T4 states using the t4_exact_solver (Rust)
and outputs NPZ chunks ready for training.

Usage:
  python generate_t4_data.py --states 50000 --workers 10 --save-interval 1000
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
import tempfile
import numpy as np
import multiprocessing as mp
from pathlib import Path
import torch

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))

from engine.encoding import Board, Observation, encode_state
from engine.action_space import get_turn_actions, encode_action, Action
from training.train_t3_oracle import T3PolicyValueNet

ACTION_DIM = 27  # Fixed canonical action space for T1-T8

EXE_NAME = "t4_exact_solver.exe" if os.name == "nt" else "t4_exact_solver"
EXE_PATH = str(AI_DIR / "rust_solver" / "target" / "release" / EXE_NAME)

GEN_EXE_NAME = "t4_generator.exe" if os.name == "nt" else "t4_generator"
GEN_EXE_PATH = str(AI_DIR / "rust_solver" / "target" / "release" / GEN_EXE_NAME)

RANKS = "23456789TJQKA"
SUITS = "shdc"

def get_deck():
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck

def apply_t0(board_top, board_mid, board_bot, t0_cards):
    t0_cards = sorted(t0_cards, key=lambda c: RANKS.index(c[0]))
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

def apply_tx(board_top, board_mid, board_bot, discards, tx_cards):
    discard_idx = random.randint(0, 2)
    discards.append(tx_cards[discard_idx])
    remaining = [tx_cards[i] for i in range(3) if i != discard_idx]
    
    pairs_50 = [("top", "mid"), ("mid", "top"), ("top", "bot"), ("bot", "top")]
    pairs_other = [("top", "top"), ("mid", "mid"), ("bot", "bot"), ("mid", "bot"), ("bot", "mid")]
    
    caps = {
        "top": 3 - len(board_top),
        "mid": 5 - len(board_mid),
        "bot": 5 - len(board_bot)
    }
    
    def is_valid(p1, p2):
        if p1 == p2:
            return caps[p1] >= 2
        return caps[p1] >= 1 and caps[p2] >= 1

    valid_pairs_50 = [p for p in pairs_50 if is_valid(*p)]
    valid_pairs_other = [p for p in pairs_other if is_valid(*p)]
    
    if random.random() < 0.5 and valid_pairs_50:
        chosen_pair = random.choice(valid_pairs_50)
    elif valid_pairs_other:
        chosen_pair = random.choice(valid_pairs_other)
    elif valid_pairs_50:
        chosen_pair = random.choice(valid_pairs_50)
    else:
        chosen_pair = ("mid", "bot") # safe fallback

    for card, pos in zip(remaining, chosen_pair):
        if pos == "top": board_top.append(card)
        elif pos == "mid": board_mid.append(card)
        else: board_bot.append(card)

def generate_heuristic_t3_boards(deck):
    bb_top, bb_mid, bb_bot = [], [], []
    bb_discards = []
    btn_top, btn_mid, btn_bot = [], [], []
    btn_discards = []
    
    # T0
    apply_t0(bb_top, bb_mid, bb_bot, [deck.pop() for _ in range(5)])
    apply_t0(btn_top, btn_mid, btn_bot, [deck.pop() for _ in range(5)])
    
    # T1
    apply_tx(bb_top, bb_mid, bb_bot, bb_discards, [deck.pop() for _ in range(3)])
    apply_tx(btn_top, btn_mid, btn_bot, btn_discards, [deck.pop() for _ in range(3)])
    
    # T2
    apply_tx(bb_top, bb_mid, bb_bot, bb_discards, [deck.pop() for _ in range(3)])
    apply_tx(btn_top, btn_mid, btn_bot, btn_discards, [deck.pop() for _ in range(3)])
    
    return {
        "bb": {"top": bb_top, "middle": bb_mid, "bottom": bb_bot, "discards": bb_discards},
        "btn": {"top": btn_top, "middle": btn_mid, "bottom": btn_bot, "discards": btn_discards}
    }

def apply_nn_placement(state_dict, dealt_cards, model, device, is_btn, opp_state_dict):
    board_obj = Board(top=state_dict["top"], middle=state_dict["middle"], bottom=state_dict["bottom"])
    opp_board = Board(top=opp_state_dict["top"], middle=opp_state_dict["middle"], bottom=opp_state_dict["bottom"])
    
    valid_actions = get_turn_actions(dealt_cards, board_obj)
    if not valid_actions:
        return False
        
    obs = Observation(
        board_self=board_obj,
        board_opponent=opp_board,
        dealt_cards=dealt_cards,
        known_discards_self=state_dict["discards"],
        turn=3,
        is_btn=is_btn,
    )
    tensor_522 = encode_state(obs)
    
    # Dynamically match model's expected input dimension
    expected_dim = model.input_proj[0].weight.shape[1]
    if expected_dim == 490:
        # The old model expects 490 dims: 486 cards + 4 meta
        tensor = np.concatenate([tensor_522[:486], tensor_522[486:490]])
    else:
        # The new model expects 520 dims (or we just take the first expected_dim elements)
        tensor = tensor_522[:expected_dim]    
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for act in valid_actions:
        idx = encode_action(act, valid_actions, turn=3, dealt_cards=dealt_cards)
        if 0 <= idx < ACTION_DIM:
            mask[idx] = True
            
    with torch.no_grad():
        s_t = torch.from_numpy(tensor).unsqueeze(0).to(device)
        m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        logits, value = model(s_t, m_t)
        
        # Pick action with highest predicted logit
        # Make invalid actions very negative
        logits[~m_t] = -1e9
        best_idx = logits.argmax(dim=-1).item()
        
    best_action = None
    for act in valid_actions:
        if encode_action(act, valid_actions, turn=3, dealt_cards=dealt_cards) == best_idx:
            best_action = act
            break
            
    if best_action is None:
        best_action = random.choice(valid_actions)
        
    state_dict["discards"].append(best_action.discard)
    for card, pos in best_action.placements:
        state_dict[pos].append(card)
        
    return True

def generate_t4_game_state(sample_id, model, device):
    deck = get_deck()
    boards = generate_heuristic_t3_boards(deck)
    
    # BB T3 turn
    bb_dealt = [deck.pop() for _ in range(3)]
    success = apply_nn_placement(boards["bb"], bb_dealt, model, device, is_btn=False, opp_state_dict=boards["btn"])
    if not success: return None
    
    # BTN T3 turn
    btn_dealt = [deck.pop() for _ in range(3)]
    success = apply_nn_placement(boards["btn"], btn_dealt, model, device, is_btn=True, opp_state_dict=boards["bb"])
    if not success: return None
    
    is_btn = random.choice([True, False])
    
    if is_btn:
        # BTN T4 turn: BB already placed 11 cards, BTN places 11 cards
        return {
            "sample_id": sample_id,
            "is_btn": True,
            "bb": boards["bb"],
            "btn": boards["btn"],
            "deck": deck
        }
    else:
        # BB T4 turn: both have 9 cards, but BB acts first, so state is BEFORE BB T3 action?
        # Wait, the heuristic simulated up to T2 (9 cards placed).
        # Oh, if BB acts in T4, we want both to have 11 cards? No, T4 means starting from 11 cards!
        # If we simulated up to T3 end, BB and BTN both have 11 cards.
        return {
            "sample_id": sample_id,
            "is_btn": False,
            "bb": boards["bb"],
            "btn": boards["btn"],
            "deck": deck
        }

def save_chunk(results, output_dir, worker_id, chunk_id):
    if not results:
        return
        
    states_np = np.array([r[0] for r in results], dtype=np.float32)
    evs_np = np.array([r[1] for r in results], dtype=np.float32)
    masks_np = np.array([r[2] for r in results], dtype=bool)
    best_evs_np = np.array([r[3] for r in results], dtype=np.float32)
    
    output_file = Path(output_dir) / f"t4_data_w{worker_id:02d}_c{chunk_id:04d}.npz"
    np.savez_compressed(
        output_file,
        states=states_np,
        action_evs=evs_np,
        action_masks=masks_np,
        best_evs=best_evs_np,
    )
    print(f"  [Worker {worker_id}] Saved chunk {chunk_id} to {output_file.name} ({len(results)} states)", flush=True)

def worker_process(args):
    worker_id, num_forward_states, peeling_states, exe_path, output_dir, save_interval, model_path = args
    
    # 1. Load model in this worker
    device = 'cpu' # Run inference on CPU in workers to avoid GPU memory issues with multiprocessing
    ckpt = torch.load(model_path, map_location=device)
    state_dim = ckpt.get("state_dim", 490)
    hidden = ckpt.get("hidden", 1024)
    n_blocks = ckpt.get("n_blocks", 4)
    model = T3PolicyValueNet(state_dim=state_dim, hidden=hidden, n_blocks=n_blocks).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    total_processed = 0
    chunk_id = 0
    results = []
    
    row_map = {"top": "top", "mid": "middle", "bot": "bottom"}
    
    # First, process peeling states
    all_states_to_process = list(peeling_states)
    
    forward_generated = 0
    
    while True:
        # Generate forward states if we need more
        while len(all_states_to_process) < save_interval and forward_generated < num_forward_states:
            st = generate_t4_game_state(forward_generated + 1000000, model, device) # offset ID
            if st:
                all_states_to_process.append(st)
                forward_generated += 1
                
        if not all_states_to_process:
            break
            
        chunk_size = min(save_interval, len(all_states_to_process))
        states_buffer = all_states_to_process[:chunk_size]
        all_states_to_process = all_states_to_process[chunk_size:]
        
        state_info = {st["sample_id"]: st for st in states_buffer}
        
        # Write to temp JSONL
        tmpdir = os.path.join(AI_DIR, f"tmp_worker_{worker_id}")
        os.makedirs(tmpdir, exist_ok=True)
        
        in_file = os.path.join(tmpdir, "in.jsonl")
        out_file = os.path.join(tmpdir, "out.jsonl")
            
        with open(in_file, "w") as f:
            for st in states_buffer:
                f.write(json.dumps(st) + "\n")
                
        # Run exact solver
        try:
            subprocess.run(
                [exe_path, in_file, out_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"  [Worker {worker_id}] Exact solver failed, skipping batch.", flush=True)
            if e.output:
                print(f"  [Worker {worker_id}] STDOUT: {e.output.decode()}", flush=True)
            if e.stderr:
                print(f"  [Worker {worker_id}] STDERR: {e.stderr.decode()}", flush=True)
            continue
                
        # Read output
        if not os.path.exists(out_file):
            continue
        
        with open(out_file, "r") as f:
            for line in f:
                sol = json.loads(line)
                sid = sol["sample_id"]
                orig_state = state_info[sid]
                
                is_btn = orig_state.get("is_btn", False)
                bb = orig_state["bb"] if not is_btn else orig_state["btn"]
                top, mid, bot = bb["top"], bb["middle"], bb["bottom"]
                board_obj = Board(top=top, middle=mid, bottom=bot)
                opp_bb = orig_state["btn"] if not is_btn else orig_state["bb"]
                opp_board_obj = Board(top=opp_bb["top"], middle=opp_bb["middle"], bottom=opp_bb["bottom"])
                
                bb_drawn = sol["bb_drawn"]
                
                try:
                    valid_actions = get_turn_actions(bb_drawn, board_obj)
                except Exception:
                    continue
                    
                if not valid_actions:
                    continue
                    
                action_evs = np.full(ACTION_DIM, -1e9, dtype=np.float32)
                
                for p in sol.get("placements", []):
                    cards_placed = p["cards_placed"]
                    slots = p["slots"]
                    ev = p["ev"]
                    
                    # Find discard
                    discard = None
                    for c in bb_drawn:
                        if c not in cards_placed:
                            discard = c
                            break
                    
                    if discard is None or len(cards_placed) != 2:
                        continue
                        
                    placements = [(cards_placed[0], row_map[slots[0]]), (cards_placed[1], row_map[slots[1]])]
                    py_action = Action(placements=placements, discard=discard)
                    
                    try:
                        idx = encode_action(py_action, valid_actions, turn=4, dealt_cards=bb_drawn)
                        if 0 <= idx < ACTION_DIM:
                            action_evs[idx] = ev
                    except (ValueError, IndexError):
                        continue
                        
                valid_ev_count = (action_evs > -1e8).sum()
                if valid_ev_count == 0:
                    continue
                    
                obs = Observation(
                    board_self=board_obj,
                    board_opponent=opp_board_obj,
                    dealt_cards=bb_drawn,
                    known_discards_self=bb["discards"],
                    turn=4,
                    is_btn=is_btn,
                )
                tensor = encode_state(obs)
                
                mask = np.zeros(ACTION_DIM, dtype=bool)
                mask[action_evs > -1e8] = True
                
                best_ev = float(sol.get("best_ev", 0.0))
                results.append((tensor, action_evs, mask, best_ev))
                total_processed += 1
                
        # Save chunk
        save_chunk(results, output_dir, worker_id, chunk_id)
        chunk_id += 1
        results = []
        
    return total_processed

def main():
    parser = argparse.ArgumentParser(description="Generate T4 Dataset with T3 NN Placements")
    parser.add_argument("--states", type=int, default=50000, help="Total states to generate")
    parser.add_argument("--workers", type=int, default=-1, help="Worker processes (-1 = auto)")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save to NPZ every N states per worker")
    parser.add_argument("--output-dir", type=str, default=str(AI_DIR / "data/t4_dataset_v1"), help="Output directory")
    parser.add_argument("--model", type=str, default=str(AI_DIR / "data/t3_oracle/t3_policyvalue_model.pt"), help="Path to trained T3 model")
    args = parser.parse_args()
    
    if not os.path.exists(EXE_PATH):
        print(f"ERROR: Rust binary not found at {EXE_PATH}")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)
    
    total = args.states
    num_peeling = total // 2
    num_forward = total - num_peeling
    
    num_workers = args.workers if args.workers > 0 else min(8, max(1, mp.cpu_count() - 2))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== T4 Dataset Generator (Peeling + Forward NN) ===")
    print(f"Target: {total} states ({num_peeling} peeling, {num_forward} forward) | Workers: {num_workers} | Interval: {args.save_interval}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {args.model}")
    start_time = time.time()
    
    # 1. Generate Peeling states
    print(f"Generating {num_peeling} peeling states using Rust generator...")
    subprocess.run([GEN_EXE_PATH, str(num_peeling)], check=True, cwd=str(AI_DIR))
    
    peeling_states = []
    with open(AI_DIR / "t4_game_states.jsonl", "r") as f:
        for line in f:
            peeling_states.append(json.loads(line))
            
    # Clean up the temp file
    os.remove(AI_DIR / "t4_game_states.jsonl")
    
    print(f"Successfully generated {len(peeling_states)} peeling states.")
    
    # 2. Setup workers
    base_peel = len(peeling_states) // num_workers
    rem_peel = len(peeling_states) % num_workers
    
    base_fwd = num_forward // num_workers
    rem_fwd = num_forward % num_workers
    
    pool_args = []
    curr_peel = 0
    for i in range(num_workers):
        p_count = base_peel + (1 if i < rem_peel else 0)
        f_count = base_fwd + (1 if i < rem_fwd else 0)
        
        chunk_peel = peeling_states[curr_peel:curr_peel+p_count]
        curr_peel += p_count
        
        if p_count > 0 or f_count > 0:
            pool_args.append((i, f_count, chunk_peel, EXE_PATH, str(output_dir), args.save_interval, args.model))
    
    if num_workers == 1:
        total_generated = sum([worker_process(pool_args[0])])
    else:
        # Use spawn method for multiprocessing to safely use PyTorch in workers
        mp.set_start_method('spawn', force=True)
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
