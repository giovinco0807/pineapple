import time
import random
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ai.engine.game_engine import evaluate_hand, get_top_royalty, get_middle_royalty, get_bottom_royalty, check_fl_entry

def evaluate_full_board(board):
    top_val = evaluate_hand(board["top"], 3)
    mid_val = evaluate_hand(board["middle"], 5)
    bot_val = evaluate_hand(board["bottom"], 5)
    
    if top_val > mid_val or mid_val > bot_val:
        return -6  # Bust penalty
        
    top_r = get_top_royalty(board["top"])
    mid_r = get_middle_royalty(board["middle"])
    bot_r = get_bottom_royalty(board["bottom"])
    
    fl, cards = check_fl_entry(board["top"])
    fl_bonus = cards if fl else 0
        
    return top_r + mid_r + bot_r + fl_bonus

def run_t0_ranking():
    print("Generating T0 Ranking using Monte Carlo Rollouts (Python)...")
    start_time = time.time()
    
    t0_cards = ["Ah", "As", "Kd", "Qs", "Jh"]
    print(f"Starting Hand: {t0_cards}")
    
    valid_placements = []
    for p in itertools.product([0, 1, 2], repeat=5):
        top_cards, mid_cards, bot_cards = [], [], []
        for i, row_idx in enumerate(p):
            if row_idx == 0: top_cards.append(t0_cards[i])
            elif row_idx == 1: mid_cards.append(t0_cards[i])
            else: bot_cards.append(t0_cards[i])
            
        if len(top_cards) > 3 or len(mid_cards) > 5 or len(bot_cards) > 5:
            continue
            
        top_cards.sort()
        mid_cards.sort()
        bot_cards.sort()
        
        board = {"top": top_cards, "middle": mid_cards, "bottom": bot_cards, "discards": []}
        if board not in valid_placements:
            valid_placements.append(board)
            
    print(f"Found {len(valid_placements)} unique valid placements.")
    
    full_deck = [r+s for r in "23456789TJQKA" for s in "shdc"]
    remaining_deck = [c for c in full_deck if c not in t0_cards]
    
    N_SAMPLES = 500
    print(f"Running Pure MC Rollouts ({N_SAMPLES} samples per placement)...")
    
    # Generate random scenarios (500 samples of 12 cards)
    scenarios = []
    for _ in range(N_SAMPLES):
        scenarios.append(random.sample(remaining_deck, 12))

    def rollout(current_board, cards_to_deal):
        board = {
            "top": current_board["top"][:],
            "middle": current_board["middle"][:],
            "bottom": current_board["bottom"][:]
        }
        
        # Turn 1 to 4
        deal_idx = 0
        for turn in range(4):
            dealt = cards_to_deal[deal_idx:deal_idx+3]
            deal_idx += 3
            
            # Simple greedy-ish but fast: just place 2 cards in any available slot randomly, discard 1
            random.shuffle(dealt)
            keep = dealt[:2]
            
            for c in keep:
                if len(board["bottom"]) < 5: board["bottom"].append(c)
                elif len(board["middle"]) < 5: board["middle"].append(c)
                elif len(board["top"]) < 3: board["top"].append(c)
                
        return evaluate_full_board(board)

    results = []
    
    for idx, board in enumerate(valid_placements):
        total_ev = 0
        for remaining_cards in scenarios:
            ev = rollout(board, remaining_cards)
            total_ev += ev
        avg_ev = total_ev / N_SAMPLES
        results.append((avg_ev, board))
            
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- TOP 10 PLACEMENTS ---")
    for i in range(10):
        ev, b = results[i]
        top = ",".join(b["top"]) if b["top"] else "-"
        mid = ",".join(b["middle"]) if b["middle"] else "-"
        bot = ",".join(b["bottom"]) if b["bottom"] else "-"
        print(f"Rank {i+1}: EV = {ev:6.3f} | Top: [{top:^11}] Mid: [{mid:^14}] Bot: [{bot:^14}]")
        
    print(f"\nTotal elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    run_t0_ranking()
