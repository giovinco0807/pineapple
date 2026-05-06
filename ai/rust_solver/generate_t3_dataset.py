import os
import sys
import json
import time
import subprocess
import numpy as np
import multiprocessing as mp
from pathlib import Path

# Add paths for encoding and generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ai.engine.encoding import Board, Observation, encode_state
from ai.heuristic_bot.t3_generator import generate_random_t3_state

def worker_process(worker_id, num_states, exe_path):
    """
    Worker process:
    1. Generates `num_states` using heuristic bot.
    2. Runs t3_exact.exe and passes all states via stdin JSON lines.
    3. Reads EV results.
    4. Encodes states to tensors.
    Returns: list of (tensor, ev)
    """
    states = []
    # 1. Generate states
    for _ in range(num_states):
        while True:
            state = generate_random_t3_state()
            if state is not None:
                states.append(state)
                break
                
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    
    # 2. Run Rust solver
    proc = subprocess.Popen(
        [exe_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env
    )
    results = []
    
    # 2. Write and Read line by line to prevent OS pipe buffer deadlock
    for i, state in enumerate(states):
        req = {
            "board_top": state["board_top"],
            "board_mid": state["board_mid"],
            "board_bot": state["board_bot"],
            "discards": state["discards"],
            "dealt": state["dealt"],
            "bust_penalty": -4.0,
            "fl_ev": {"14": 13.0, "15": 40.0, "16": 55.1, "17": 90.7}
        }
        
        with open("debug_t3_state.json", "w") as f:
            f.write(json.dumps(req))
            
        print(f"Writing state {i} to Rust...")
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        
        print("Waiting for Rust output...")
        line = proc.stdout.readline()
        print("Received Rust output.")
        if not line:
            break
            
        res = json.loads(line)
        if "error" in res:
            print(f"Rust solver error: {res['error']}")
            continue
            
        best_ev = res["best_ev"]
        
        # Encode state
        board_obj = Board(
            top=state["board_top"],
            middle=state["board_mid"],
            bottom=state["board_bot"]
        )
        
        obs = Observation(
            board_self=board_obj,
            board_opponent=Board(), # Ignore opponent for God Mode absolute EV
            dealt_cards=state["dealt"],
            known_discards_self=state["discards"],
            turn=3, # T3
            is_btn=False
        )
        
        from ai.engine.action_space import get_turn_actions, encode_action, create_action_mask, Action
        valid_actions = get_turn_actions(state["dealt"], board_obj)
        action_mask = create_action_mask(valid_actions)
        
        row_map = {"T": "top", "M": "middle", "B": "bottom"}
        action_evs = np.full(250, -1e9, dtype=np.float32)
        
        for act in res.get("actions", []):
            desc = act.get("action_desc", "")
            ev = act.get("ev", -4.0)
            
            parts = desc.split()
            discard = None
            placements = []
            
            jokers_in_dealt = [c for c in state["dealt"] if c.startswith("X")]
            joker_idx = 0
            
            for p in parts:
                if p.startswith("d:"):
                    discard = p[2:]
                    if discard == "JK":
                        discard = jokers_in_dealt[joker_idx]
                        joker_idx += 1
                else:
                    if "->" in p:
                        card, row = p.split("->")
                        if card == "JK":
                            card = jokers_in_dealt[joker_idx]
                            joker_idx += 1
                        placements.append((card, row_map[row]))
                        
            py_action = Action(placements=placements, discard=discard)
            try:
                idx = encode_action(py_action, valid_actions)
                action_evs[idx] = ev
            except ValueError:
                if len(jokers_in_dealt) == 2:
                    swapped_placements = [(c if c not in ("X1", "X2") else ("X2" if c == "X1" else "X1"), r) for c, r in placements]
                    swapped_discard = discard if discard not in ("X1", "X2") else ("X2" if discard == "X1" else "X1")
                    py_action2 = Action(placements=swapped_placements, discard=swapped_discard)
                    try:
                        idx = encode_action(py_action2, valid_actions)
                        action_evs[idx] = ev
                    except ValueError:
                        pass
                else:
                    pass
            
        tensor = encode_state(obs)
        results.append((tensor, action_evs, action_mask))
        
    proc.stdin.close()
    proc.wait()
    return results

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate T3 Dataset")
    parser.add_argument("--states", type=int, default=10000, help="Total number of states to generate")
    parser.add_argument("--workers", type=int, default=-1, help="Number of worker processes")
    args = parser.parse_args()
    
    total_states = args.states
    if args.workers > 0:
        num_workers = args.workers
    else:
        num_workers = min(60, mp.cpu_count() - 1)
        if num_workers < 1: num_workers = 1
    
    exe_name = "t3_exact.exe" if os.name == "nt" else "t3_exact"
    exe_path = str(Path(__file__).resolve().parent / "target" / "release" / exe_name)
    if not os.path.exists(exe_path):
        print(f"Error: Could not find {exe_name} at {exe_path}. Please build it first.")
        sys.exit(1)
        
    print(f"Starting generation of {total_states} T3 states using {num_workers} workers...")
    start_time = time.time()
    
    output_dir = Path("data/t3_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_size = 10000
    num_chunks = (total_states + chunk_size - 1) // chunk_size
    states_generated = 0
    
    for chunk_idx in range(num_chunks):
        chunk_states = min(chunk_size, total_states - states_generated)
        states_per_worker = chunk_states // num_workers
        remainder = chunk_states % num_workers
        
        pool_args = []
        for i in range(num_workers):
            n = states_per_worker + (1 if i < remainder else 0)
            if n > 0:
                pool_args.append((i, n, exe_path))
                
        print(f"\n--- Generating Chunk {chunk_idx+1}/{num_chunks} ({chunk_states} states) ---")
        if num_workers == 1:
            results = [worker_process(*pool_args[0])]
        else:
            with mp.Pool(num_workers) as pool:
                results = pool.starmap(worker_process, pool_args)
            
        all_tensors = []
        all_evs = []
        all_masks = []
        for worker_res in results:
            for tensor, action_evs, action_mask in worker_res:
                all_tensors.append(tensor)
                all_evs.append(action_evs)
                all_masks.append(action_mask)
                
        all_tensors = np.array(all_tensors, dtype=np.float32)
        all_evs = np.array(all_evs, dtype=np.float32)
        all_masks = np.array(all_masks, dtype=bool)
        
        np.save(output_dir / f"states_chunk_{chunk_idx}.npy", all_tensors)
        np.save(output_dir / f"action_evs_chunk_{chunk_idx}.npy", all_evs)
        np.save(output_dir / f"action_masks_chunk_{chunk_idx}.npy", all_masks)
        
        states_generated += chunk_states
        
        print(f"Saved chunk {chunk_idx+1} to {output_dir}")
        print("Uploading to GCS...")
        try:
            subprocess.run(
                ["gsutil", "-m", "rsync", "-r", str(output_dir), "gs://ofc-solver-485418/ofc_rl_output/t3_dataset/"],
                check=False
            )
            print("Upload complete.")
        except Exception as ex:
            print(f"Upload failed: {ex}")
            
    print(f"\nGeneration complete in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
