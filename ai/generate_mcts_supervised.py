import sys
import os
import json
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.engine.encoding import ALL_CARDS
from ai.engine.action_space import get_initial_actions, get_turn_actions
from ai.engine.game_engine import Hand
from ai.mcts.mcts import MCTS, MCTSConfig
from ai.models.networks import PolicyNetwork, ValueNetwork

def format_board(board):
    return f"Top[{' '.join(board.top)}] Mid[{' '.join(board.middle)}] Bot[{' '.join(board.bottom)}]"

def format_action(action):
    p_str = ", ".join([f"{c}->{p.capitalize()}" for c, p in action.placements])
    d_str = action.discard if action.discard else ""
    return p_str, d_str

def record_data(hand, seat, turn_num, probs, valid_actions, out_f):
    model_type = "btn" if seat == hand.btn else "bb"
    model_label = f"t{turn_num}_{model_type}"
    
    placements_data = []
    for act_idx, visit_prob in probs.items():
        if act_idx < len(valid_actions):
            p_str, d_str = format_action(valid_actions[act_idx])
            placements_data.append({
                "p": p_str,
                "d": d_str,
                "visit_prob": float(visit_prob)
            })
    
    record = {
        "model": model_label,
        "board": format_board(hand.boards[seat]),
        "hand": " ".join(hand.dealt_cards[seat]),
        "opp_top": " ".join(hand.boards[1-seat].top),
        "opp_mid": " ".join(hand.boards[1-seat].middle),
        "opp_bot": " ".join(hand.boards[1-seat].bottom),
        "dead_cards": " ".join(hand.discards[seat]),
        "placements": placements_data
    }
    out_f.write(json.dumps(record) + "\n")
    out_f.flush()

def generate_data(num_games=100, output_file="ai/data/mcts_supervised_data.jsonl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize random initialized models if pre-trained are not available
    # Actually, we should check if they exist, but for data collection, MCTS will work (just slowly learning if uniform).
    # Ideally, MCTS with PW and random rollout/evaluation is okay if we use enough sims.
    # But usually it's better to load checkpoint. Let's just instantiate.
    policy_net = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    
    mcts_config = MCTSConfig(num_simulations=100, c_puct=1.5)
    mcts = MCTS(policy_net, value_net, mcts_config, device)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_f = open(output_file, 'w', encoding='utf-8')
    
    print(f"Generating {num_games} games of MCTS self-play data...")
    start_time = time.time()
    
    for game_idx in range(num_games):
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = Hand(deck=deck, btn=random.randint(0, 1))
        
        # Turn 0
        for seat in [hand.btn, 1 - hand.btn]:
            obs = hand.get_observation(seat)
            action_idx, probs, valid_actions = mcts.search(obs, {})
            
            record_data(hand, seat, 0, probs, valid_actions, out_f)
            
            if valid_actions and action_idx < len(valid_actions):
                hand.apply_action(seat, valid_actions[action_idx])
                
        # Turns 1 to 8
        for turn_num in range(1, 9):
            if hand.is_hand_complete():
                break
            hand.deal_next_turn()
            
            for seat in [hand.btn, 1 - hand.btn]:
                cards = hand.dealt_cards[seat]
                if not cards: continue
                if hand.boards[seat].is_complete(): continue
                
                obs = hand.get_observation(seat)
                action_idx, probs, valid_actions = mcts.search(obs, {})
                
                # We want to record data for T1, T2, T3
                if turn_num in [1, 2, 3]:
                    record_data(hand, seat, turn_num, probs, valid_actions, out_f)
                
                if valid_actions and action_idx < len(valid_actions):
                    hand.apply_action(seat, valid_actions[action_idx])
                    
        if (game_idx + 1) % 10 == 0:
            print(f"  {game_idx+1}/{num_games} games...")
            
    out_f.close()
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--output", type=str, default="ai/data/mcts_supervised_data.jsonl", help="Output file path")
    args = parser.parse_args()
    
    generate_data(num_games=args.num_games, output_file=args.output)
