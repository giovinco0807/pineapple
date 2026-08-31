#!/usr/bin/env python3
"""
T4 Dataset Generator (NN + Heuristic)

Generates T4 game states by simulating the game up to T3 using a heuristic,
and then making the T3 placements using a trained T3PolicyValueNet model.
Outputs a JSONL file in the exact same format as t4_generator.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
import torch
import numpy as np

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))

from engine.encoding import Board, Observation, encode_state
from engine.action_space import get_turn_actions, encode_action, Action
from training.train_t3_oracle import T3PolicyValueNet

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
    
    for card in remaining:
        valid = []
        if len(board_top) < 3: valid.append("top")
        if len(board_mid) < 5: valid.append("mid")
        if len(board_bot) < 5: valid.append("bot")
        
        pos = random.choice(valid)
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
    # The model expects 490 dims: 486 cards + 4 meta (turn, is_btn, is_fl, opp_is_fl)
    tensor = np.concatenate([tensor_522[:486], tensor_522[486:490]])
    
    mask = np.zeros(27, dtype=bool)
    for act in valid_actions:
        idx = encode_action(act, valid_actions, turn=3, dealt_cards=dealt_cards)
        if 0 <= idx < 27:
            mask[idx] = True
            
    with torch.no_grad():
        s_t = torch.from_numpy(tensor).unsqueeze(0).to(device)
        m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        logits, value = model(s_t, m_t)
        
        # Pick action with highest predicted EV
        # wait, is value_head the one predicting EV for the state, and policy_head ranking actions?
        # logits ranking represents the action's desirability. Let's argmax logits.
        best_idx = logits.argmax(dim=-1).item()
        
    # Find the corresponding Action
    best_action = None
    for act in valid_actions:
        if encode_action(act, valid_actions, turn=3, dealt_cards=dealt_cards) == best_idx:
            best_action = act
            break
            
    if best_action is None:
        # Fallback to random if something goes wrong
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
    
    return {
        "sample_id": sample_id,
        "bb": boards["bb"],
        "btn": boards["btn"],
        "deck": deck
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--model", type=str, default="data/t3_oracle/t3_policyvalue_model.pt")
    parser.add_argument("--output", type=str, default="rust_solver/t4_generator/t4_game_states_nn.jsonl")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading T3 Model from {args.model} on {device}...")
    
    # Load Model
    ckpt = torch.load(args.model, map_location=device)
    state_dim = ckpt.get("state_dim", 490)
    hidden = ckpt.get("hidden", 1024)
    n_blocks = ckpt.get("n_blocks", 4)
    
    model = T3PolicyValueNet(state_dim=state_dim, hidden=hidden, n_blocks=n_blocks).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    
    output_path = Path(AI_DIR) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.samples} T4 states using NN T3 placements...", flush=True)
    
    results_generated = 0
    with open(output_path, "w") as f:
        for i in range(args.samples):
            state = generate_t4_game_state(i, model, device)
            if state:
                f.write(json.dumps(state) + "\n")
                results_generated += 1
            if (i+1) % 100 == 0:
                print(f"  Generated {results_generated} / {args.samples}", flush=True)
                
    print(f"Saved {results_generated} states to {output_path}", flush=True)

if __name__ == "__main__":
    main()
