import random
import torch
import sys
from pathlib import Path

# Add parent directory to path so we can import ai modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_v4 import load_policy_net, predict_for_hand, analyze_placement

def main():
    device = "cpu"
    model_path = r"c:\Users\Owner\.gemini\antigravity\worktrees\ofc-pineapple\verify-gcp-phase-one-20260501\ai\models\t0_placement_net_v4.pt"
    
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']
    deck = [r + s for r in ranks for s in suits]
    
    print("Loading PolicyNet v4...")
    model = load_policy_net(model_path, device)
    
    for test_idx in range(5):
        hand_cards = random.sample(deck, 5)
        print("=" * 60)
        print(f"Hand {test_idx + 1}: {hand_cards}")
        print("=" * 60)
        
        results = predict_for_hand(model, hand_cards, device)
        
        if not results:
            print("No valid actions found.")
            continue
            
        print("Top 3 Placements:")
        for i, (p_str, prob, idx) in enumerate(results[:3]):
            props = analyze_placement(p_str)
            props_str = " | ".join(props) if props else ""
            print(f"  {i+1}: {p_str} (Prob: {prob:.4f}) [{props_str}]")
        print("")

if __name__ == "__main__":
    main()
