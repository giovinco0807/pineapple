import sys
import json
import random
import uuid
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ai_player import init_ai, _ai_re1
from ai.engine.encoding import Observation, Board, encode_state
from ai.engine.action_space import get_initial_actions
import torch
import numpy as np

def generate_hands(num_hands=100):
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    suits = ['s','h','d','c']
    deck = [r+s for r in ranks for s in suits]
    
    hands = []
    for _ in range(num_hands):
        hand = random.sample(deck, 5)
        hands.append(hand)
    return hands

def evaluate_hand(cards):
    valid_actions = get_initial_actions(cards, Board())
    if not valid_actions:
        return None
        
    states = []
    placements = []
    for a in valid_actions:
        b = Board()
        for c, p in a.placements:
            getattr(b, p).append(c)
            
        obs = Observation(board_self=b, board_opponent=Board(), dealt_cards=[], known_discards_self=[], turn=1, is_btn=True)
        states.append(encode_state(obs))
        placements.append({
            "top": list(b.top),
            "middle": list(b.middle),
            "bottom": list(b.bottom)
        })
        
    t = torch.FloatTensor(np.array(states))
    vn = _ai_re1.value_net
    mean = _ai_re1.score_mean
    std = _ai_re1.score_std
    
    with torch.no_grad():
        res = vn(t)
        
    values = res['value'].squeeze(-1).numpy()
    values = values * std + mean
    
    results = []
    for i in range(len(valid_actions)):
        results.append({
            "placement": placements[i],
            "ev": float(values[i]),
            "bust_prob": float(res['bust_prob'][i].item()) if 'bust_prob' in res else 0.0,
            "fl_prob": float(res['fl_prob'][i].item()) if 'fl_prob' in res else 0.0,
            "royalty_ev": float(res['royalty_ev'][i].item()) if 'royalty_ev' in res else 0.0
        })
        
    results.sort(key=lambda x: x['ev'], reverse=True)
    return results[:10]  # Store top 10 placements

def main():
    print("Initializing AI models...")
    init_ai()
    print("Generating hands...")
    hands = generate_hands(100)
    
    presets = {}
    
    for i, hand in enumerate(hands):
        print(f"Evaluating hand {i+1}/100: {hand}")
        top_placements = evaluate_hand(hand)
        if top_placements:
            puzzle_id = str(uuid.uuid4())[:8]
            presets[puzzle_id] = {
                "cards": hand,
                "top_placements": top_placements,
                "optimal_ev": top_placements[0]['ev']
            }
            
    output_path = Path(__file__).parent / "training_presets.json"
    with open(output_path, "w") as f:
        json.dump(presets, f)
        
    print(f"Saved {len(presets)} presets to {output_path}")

if __name__ == "__main__":
    main()
