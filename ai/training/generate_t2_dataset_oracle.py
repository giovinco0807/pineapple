import sys
import json
import random
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state, STATE_DIM
from ai.engine.action_space import REGULAR_TURN_ACTIONS, get_semantic_action_index, get_turn_actions
from ai.training.train_t3_oracle_v2 import T3PolicyValueNet
from ai.training.generate_bottomup_data import load_patterns_from_json, save_records
from ai.training.generate_t2_oracle_data import load_t3_oracle, apply_placements

MAX_ACTIONS = REGULAR_TURN_ACTIONS

def generate_t2_data_with_oracle(states, model, device, n_samples=30):
    records = []
    t0 = time.time()
    
    for si, state_info in enumerate(states):
        board = state_info["board"]
        deal = state_info["deal"]
        
        actions = get_turn_actions(deal, board)
        if not actions:
            continue
            
        rem_deck = [c for c in ALL_CARDS if c not in (board.top + board.middle + board.bottom + deal)]
        
        action_evs_arr = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
        valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
        
        t3_states = []
        t3_indices = []
        
        if len(actions) == 1:
            idx = get_semantic_action_index(actions[0], deal)
            action_evs_arr[idx] = 0.0
            valid_mask[idx] = True
        else:
            for ai, action in enumerate(actions):
                next_board = apply_placements(board, action.placements)
                disc = [action.discard] if action.discard else []
                
                action_indices = []
                for _ in range(n_samples):
                    t3_deal = random.sample(rem_deck, 3)
                    obs = Observation(
                        board_self=next_board,
                        board_opponent=Board(),
                        dealt_cards=t3_deal,
                        known_discards_self=disc,
                        turn=3,
                        is_btn=True
                    )
                    state_vec = encode_state(obs)
                    state_vec = state_vec[:model.input_proj[0].in_features]
                    t3_states.append(state_vec)
                    action_indices.append(len(t3_states) - 1)
                t3_indices.append(action_indices)
                
            batch_states = torch.FloatTensor(np.array(t3_states)).to(device)
            with torch.no_grad():
                _, values = model(batch_states, masks=None)
                values = values.cpu().numpy()
                
            for ai, action in enumerate(actions):
                ev = np.mean(values[t3_indices[ai]])
                idx = get_semantic_action_index(action, deal)
                action_evs_arr[idx] = ev
                valid_mask[idx] = True

        obs = Observation(
            board_self=board,
            board_opponent=Board(),
            dealt_cards=deal,
            known_discards_self=[],
            turn=2,
            is_btn=True
        )
        state_vec = encode_state(obs)
        best_idx = int(np.argmax(action_evs_arr))

        records.append({
            "state": state_vec.astype(np.float16),
            "action": best_idx,
            "ev": float(action_evs_arr[best_idx]),
            "turn": 2,
            "valid_mask": valid_mask,
            "action_evs": action_evs_arr.astype(np.float16),
        })

        if (si + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (si + 1) / elapsed
            print(f"  Processed {si+1}/{len(states)} states ({rate:.1f} states/s)")

    return records

def main():
    parser = argparse.ArgumentParser(description="Generate T2 Data using T3 Oracle")
    parser.add_argument("--t3-model", required=True, help="Path to t3_policyvalue_v2_best.pt")
    parser.add_argument("--json-dir", default="data/expectimax_results_v3", help="Directory with JSON results")
    parser.add_argument("--save", required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=30, help="T3 deals to sample per T2 action")
    parser.add_argument("--max-states", type=int, default=0, help="Limit number of states")
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-end", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Oracle-based T2 Data Generation ===")
    print(f"  T3 Model: {args.t3_model}")
    print(f"  Samples/Action: {args.n_samples}")
    print(f"  Device: {device}")

    states = load_patterns_from_json(args.json_dir, target_turn=2, file_start=args.file_start, file_end=args.file_end)
    if args.max_states > 0:
        states = states[:args.max_states]
        print(f"  Limited to {len(states)} states")

    print(f"Loading T3 Oracle...")
    model = load_t3_oracle(args.t3_model, device)

    print(f"Generating data...")
    records = generate_t2_data_with_oracle(states, model, device, args.n_samples)

    print(f"Saving records...")
    save_records(records, args.save)
    print(f"=== Generation Complete ===")

if __name__ == "__main__":
    main()
