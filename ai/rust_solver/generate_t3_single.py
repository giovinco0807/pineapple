import os
import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path

# Add paths for encoding and generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ai.engine.encoding import Board, Observation, encode_state
from ai.heuristic_bot.t3_generator import generate_random_t3_state

def main():
    total_states = 50
    exe_path = str(Path(__file__).resolve().parent / "target" / "release" / "t3_exact.exe")
    
    print(f"Starting single-threaded generation of {total_states} states...")
    start_time = time.time()
    
    states = []
    # 1. Generate random states
    for _ in range(total_states):
        while True:
            s = generate_random_t3_state()
            if s is not None:
                states.append(s)
                break
                
    print("Generated random states. Now evaluating...")
    
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "4"
    
    proc = subprocess.Popen(
        [exe_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    all_tensors = []
    all_evs = []
    all_actions = []
    all_masks = []
    
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
        
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        
        line = proc.stdout.readline()
        if not line:
            break
            
        res = json.loads(line)
        if "error" in res:
            continue
            
        best_ev = res["best_ev"]
        
        board_obj = Board(
            top=state["board_top"],
            middle=state["board_mid"],
            bottom=state["board_bot"]
        )
        
        obs = Observation(
            board_self=board_obj,
            board_opponent=Board(),
            dealt_cards=state["dealt"],
            known_discards_self=state["discards"],
            turn=3,
            is_btn=False
        )
        
        # Extract Python valid actions
        from ai.engine.action_space import get_turn_actions, encode_action, create_action_mask, Action
        valid_actions = get_turn_actions(state["dealt"], board_obj)
        action_mask = create_action_mask(valid_actions)
        
        # Parse rust action descriptions
        row_map = {"T": "top", "M": "middle", "B": "bottom"}
        
        # Initialize EV array for all actions (MAX_ACTIONS = 250)
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
                # If X1/X2 order is swapped, try swapping them
                if len(jokers_in_dealt) == 2:
                    swapped_placements = [(c if c not in ("X1", "X2") else ("X2" if c == "X1" else "X1"), r) for c, r in placements]
                    swapped_discard = discard if discard not in ("X1", "X2") else ("X2" if discard == "X1" else "X1")
                    py_action2 = Action(placements=swapped_placements, discard=swapped_discard)
                    try:
                        idx = encode_action(py_action2, valid_actions)
                        action_evs[idx] = ev
                    except ValueError:
                        print(f"Warning: Rust action not in Python valid actions: {py_action} or {py_action2}")
                else:
                    print(f"Warning: Rust action not in Python valid actions: {py_action}")
                
        tensor = encode_state(obs)
        all_tensors.append(tensor)
        all_evs.append(action_evs)
        all_masks.append(action_mask)
        
        if (i+1) % 50 == 0:
            print(f"Evaluated {i+1}/{total_states}...")
            
    proc.stdin.close()
    proc.wait()
    
    all_tensors = np.array(all_tensors, dtype=np.float32)
    all_evs = np.array(all_evs, dtype=np.float32)
    all_masks = np.array(all_masks, dtype=bool)
    
    print(f"Generation complete in {time.time() - start_time:.2f}s")
    print(f"Collected {len(all_tensors)} valid states.")
    
    output_dir = Path("data/t3_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "states.npy", all_tensors)
    np.save(output_dir / "action_evs.npy", all_evs)
    np.save(output_dir / "action_masks.npy", all_masks)
    
    print(f"Saved dataset to {output_dir}")

if __name__ == "__main__":
    main()
