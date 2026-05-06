import sys
import numpy as np
from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.heuristic_bot.t0_generator import generate_t0_placements, RANKS, SUITS
from ai.heuristic_bot.t1_generator import generate_t1_actions
from ai.engine.encoding import Observation, Board, encode_state

def generate_random_deck():
    deck = [f"{r}{s}" for r in RANKS for s in SUITS] + ["X1", "X2"]
    random.shuffle(deck)
    return deck

def generate_t2_state_vector():
    deck = generate_random_deck()
    
    # T0 (5 cards)
    t0_cards = deck[:5]
    t0_placements = generate_t0_placements(t0_cards)
    if not t0_placements: return None
    t0_board_dict = random.choice(t0_placements)
    
    # T1 (3 cards)
    t1_cards = deck[5:8]
    t1_actions = generate_t1_actions(t0_board_dict, t1_cards)
    if not t1_actions: return None
    t1_action = random.choice(t1_actions)
    
    # Merge to get T1 final board
    p = t1_action['place']
    d = t1_action['discard'][0]
    
    final_top = t0_board_dict.get('top', []) + p.get('top', [])
    final_mid = t0_board_dict.get('mid', []) + p.get('mid', [])
    final_bot = t0_board_dict.get('bot', []) + p.get('bot', [])
    
    board = Board(top=final_top, middle=final_mid, bottom=final_bot)
    known_discards = [d]
    
    # T2 (Dealt 3 cards)
    t2_cards = deck[8:11]
    
    # Create Observation for the start of T2 decision
    obs = Observation(
        board_self=board,
        board_opponent=Board(),  # Empty for God Mode absolute EV target
        dealt_cards=t2_cards,
        known_discards_self=known_discards,
        turn=2,
        is_btn=True
    )
    
    state_vector = encode_state(obs)
    return state_vector

def main():
    num_samples = 1000
    print(f"Generating {num_samples} T2 state vectors using Heuristic Bot...")
    
    states = []
    
    # Generate samples
    for i in range(num_samples):
        vec = generate_t2_state_vector()
        if vec is not None:
            states.append(vec)
            
        if (i+1) % 100 == 0:
            print(f"Generated {i+1} samples...")
            
    states = np.array(states)
    print(f"Successfully generated {len(states)} valid T2 state vectors.")
    print(f"State tensor shape: {states.shape}")
    
    # Save as NPZ
    output_file = project_root / "ai" / "heuristic_bot" / "t2_states_seed.npz"
    np.savez(output_file, states=states)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
