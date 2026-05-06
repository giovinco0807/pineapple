#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import subprocess
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.models.t3_policy_value import T3PolicyValueNet
from ai.engine.encoding import Board, encode_state
from ai.engine.action_space import get_turn_actions, encode_action, MAX_ACTIONS

def generate_random_t4_state():
    deck = [r+s for r in '23456789TJQKA' for s in 'shcd']
    random.shuffle(deck)
    
    board_top = []
    board_mid = []
    board_bot = []
    
    # Place exactly 9 cards
    for _ in range(9):
        card = deck.pop()
        valid_rows = []
        if len(board_top) < 3: valid_rows.append('top')
        if len(board_mid) < 5: valid_rows.append('mid')
        if len(board_bot) < 5: valid_rows.append('bot')
        
        row = random.choice(valid_rows)
        if row == 'top': board_top.append(card)
        elif row == 'mid': board_mid.append(card)
        else: board_bot.append(card)
        
    dealt = [deck.pop() for _ in range(3)]
    discards = [deck.pop() for _ in range(2)]
    
    board = Board(top=board_top, middle=board_mid, bottom=board_bot)
    return board, dealt, discards

def format_action_for_canonical(action):
    placements = []
    for c, row in action.placements:
        r_str = "T" if row == "top" else "M" if row == "middle" else "B"
        placements.append(f"{c}->{r_str}")
    placements.sort()
    return f"d:{action.discard} " + " ".join(placements)

def evaluate_state(model, device, board, dealt, discards):
    # 1. NN Prediction
    import numpy as np
    from ai.engine.action_space import create_action_mask
    
    # We must use Observation with discards
    from ai.engine.encoding import Observation
    
    state_obj = Observation(
        board_self=board,
        board_opponent=Board(),
        dealt_cards=dealt,
        known_discards_self=discards,
        turn=4,
        is_btn=False
    )
    encoded_state = encode_state(state_obj)
    
    state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0).cpu().numpy()
        
    valid_actions = get_turn_actions(dealt, board)
    
    nn_predictions = []
    for action in valid_actions:
        idx = encode_action(action, valid_actions)
        if idx != -1:
            nn_predictions.append({
                "action": action,
                "canonical": format_action_for_canonical(action),
                "pred_ev": q_values[idx]
            })
            
    # Sort NN predictions by predicted EV
    nn_predictions.sort(key=lambda x: x["pred_ev"], reverse=True)
    
    # 2. Rust Ground Truth
    req = {
        "board_top": board.top,
        "board_mid": board.middle,
        "board_bot": board.bottom,
        "discards": discards,
        "dealt": dealt
    }
    
    rust_bin = os.path.join(os.path.dirname(__file__), "rust_solver", "target", "release", "t3_exact")
    if os.name == 'nt':
        rust_bin += ".exe"
        
    try:
        proc = subprocess.run([rust_bin], input=json.dumps(req).encode('utf-8'), capture_output=True, timeout=30)
        if proc.returncode != 0:
            return None, None
            
        res = json.loads(proc.stdout.decode('utf-8').strip())
        actions = res.get("actions", [])
        
        gt_map = {}
        for a in actions:
            parts = a['action_desc'].split(' ')
            discard = parts[0][2:]
            placements = []
            for p in parts[1:]:
                c, r = p.split('->')
                placements.append(f"{c}->{r}")
            placements.sort()
            canonical = f"d:{discard} " + " ".join(placements)
            gt_map[canonical] = a['ev']
            
        return nn_predictions, gt_map
    except Exception as e:
        return None, None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T3PolicyValueNet().to(device)
    model_path = "ai/models/t3_policy.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    num_samples = 20
    print(f"Generating and Evaluating {num_samples} Random Turn 4 States...\n")
    
    total_regret = 0.0
    exact_matches = 0
    valid_samples = 0
    
    for i in range(num_samples):
        board, dealt, discards = generate_random_t4_state()
        nn_preds, gt_map = evaluate_state(model, device, board, dealt, discards)
        
        if not nn_preds or not gt_map:
            continue
            
        nn_top = nn_preds[0]
        nn_top_canonical = nn_top["canonical"]
        
        # Find GT best
        gt_best_canonical = max(gt_map.keys(), key=lambda k: gt_map[k])
        gt_best_ev = gt_map[gt_best_canonical]
        
        # GT EV of NN's top choice
        nn_top_actual_ev = gt_map.get(nn_top_canonical, -100.0)
        
        regret = gt_best_ev - nn_top_actual_ev
        
        # For display, let's limit regret to 0 if rounding error
        if regret < 1e-4:
            regret = 0.0
            exact_matches += 1
            
        total_regret += regret
        valid_samples += 1
        
        print(f"--- Sample {i+1} ---")
        print(f"Board: T:{board.top} M:{board.middle} B:{board.bottom}")
        print(f"Dealt: {dealt}")
        print(f"GT Best EV : {gt_best_ev:.2f} ({gt_best_canonical})")
        print(f"NN Top EV  : {nn_top_actual_ev:.2f} ({nn_top_canonical}) | Pred: {nn_top['pred_ev']:.2f}")
        print(f"EV Regret  : {regret:.2f} points\n")
        
    if valid_samples > 0:
        avg_regret = total_regret / valid_samples
        match_rate = (exact_matches / valid_samples) * 100
        
        print("==================================================")
        print("                EVALUATION SUMMARY                ")
        print("==================================================")
        print(f"Total States Evaluated : {valid_samples}")
        print(f"Exact Top-1 Matches    : {exact_matches} ({match_rate:.1f}%)")
        print(f"Average EV Regret      : {avg_regret:.3f} points")
        print("==================================================")
        print("Note: 'EV Regret' is the actual points lost by choosing the Model's")
        print("      favorite move instead of the absolute perfect mathematical move.")

if __name__ == "__main__":
    main()
