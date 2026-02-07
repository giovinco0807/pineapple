"""Test Policy Network with Solver."""
import sys
sys.path.insert(0, 'ai')
import torch
import time
import numpy as np

from policy_network_v2 import TransformerPolicyNetwork, CARD_DIM
from fl_solver import deal_fantasyland_hand, solve_fantasyland_exhaustive, evaluate_placement

def encode_card(card):
    features = np.zeros(CARD_DIM, dtype=np.float32)
    if card.is_joker:
        features[CARD_DIM - 1] = 1.0
    else:
        rank_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5,
                   8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12}
        rank_idx = rank_map.get(card.rank_value, 0)
        features[rank_idx] = 1.0
        suit_map = {'spades': 0, 'hearts': 1, 'diamonds': 2, 'clubs': 3}
        suit_idx = suit_map.get(card.suit, 0)
        features[13 + suit_idx] = 1.0
    return features

def main():
    # Load model
    print("Loading model...")
    model = TransformerPolicyNetwork(d_model=128, num_layers=3)
    checkpoint = torch.load('ai/models/policy_net_v2.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded, best acc: {checkpoint['val_accuracy']:.4f}")
    
    # Test multiple hands
    n_tests = 5
    policy_times = []
    full_times = []
    score_matches = 0
    
    for i in range(n_tests):
        print(f"\n--- Test {i+1}/{n_tests} ---")
        hand = deal_fantasyland_hand(14, include_jokers=False)
        print(f"Hand: {' '.join(str(c) for c in hand[:5])} ...")
        
        # Encode hand
        hand_features = np.zeros((17, CARD_DIM), dtype=np.float32)
        for j, card in enumerate(hand):
            hand_features[j] = encode_card(card)
        
        # Policy prediction
        start = time.time()
        with torch.no_grad():
            x = torch.from_numpy(hand_features).unsqueeze(0)
            scores = model(x).squeeze(0).numpy()
        
        # Get top candidates
        top_k = 100
        top_indices = np.argsort(scores[:14])[-top_k:]
        
        # Find best placement using policy candidates
        from itertools import combinations
        best_policy = None
        best_policy_score = float('-inf')
        
        for top_idx in combinations(top_indices, 3):
            top = [hand[i] for i in top_idx]
            remaining = [hand[i] for i in range(14) if i not in top_idx]
            
            for mid_idx in combinations(range(11), 5):
                middle = [remaining[i] for i in mid_idx]
                rest = [remaining[i] for i in range(11) if i not in mid_idx]
                bottom = rest[:5]
                
                p = evaluate_placement(top, middle, bottom)
                if not p.is_bust and p.score > best_policy_score:
                    best_policy_score = p.score
                    best_policy = p
        
        t_policy = time.time() - start
        policy_times.append(t_policy)
        
        # Full exhaustive
        start = time.time()
        full = solve_fantasyland_exhaustive(hand)
        t_full = time.time() - start
        full_times.append(t_full)
        
        full_score = full[0].score if full else 0
        policy_score = best_policy_score if best_policy else 0
        
        print(f"Policy: {t_policy:.2f}s, score: {policy_score}")
        print(f"Full:   {t_full:.2f}s, score: {full_score}")
        print(f"Match: {'YES' if abs(policy_score - full_score) < 0.01 else 'NO'}")
        
        if abs(policy_score - full_score) < 0.01:
            score_matches += 1
    
    print(f"\n=== Summary ===")
    print(f"Policy avg time: {np.mean(policy_times):.2f}s")
    print(f"Full avg time:   {np.mean(full_times):.2f}s")
    print(f"Speedup: {np.mean(full_times) / np.mean(policy_times):.1f}x")
    print(f"Score match rate: {score_matches}/{n_tests} ({100*score_matches/n_tests:.0f}%)")

if __name__ == "__main__":
    main()
