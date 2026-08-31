import time
import random

def bench_500_samples():
    print("Starting PIMC 500 samples DFS in Python...")
    start_time = time.time()
    
    # 1. Initialize T0 board
    # We place 2h->Top, 3h->Mid, 4h,5h,6h->Bot
    board = {
        "top": ["2h"],
        "middle": ["3h"],
        "bottom": ["4h", "5h", "6h"],
        "discards": []
    }
    
    # The remaining deck
    full_deck = [
        r+s for r in "23456789TJQKA" for s in "shdc"
    ]
    seen_cards = board["top"] + board["middle"] + board["bottom"]
    remaining_deck = [c for c in full_deck if c not in seen_cards]
    
    leaves_evaluated = 0
    
    def dfs(current_board, turn_idx, deals):
        nonlocal leaves_evaluated
        if turn_idx == 4:
            leaves_evaluated += 1
            return 0 
            
        dealt = deals[turn_idx]
        
        top_cap = 3 - len(current_board["top"])
        mid_cap = 5 - len(current_board["middle"])
        bot_cap = 5 - len(current_board["bottom"])
        
        rows = []
        if top_cap >= 1: rows.append("top")
        if mid_cap >= 1: rows.append("middle")
        if bot_cap >= 1: rows.append("bottom")
        
        placements = []
        for r1 in rows:
            for r2 in rows:
                if r1 == r2:
                    cap = top_cap if r1 == "top" else (mid_cap if r1 == "middle" else bot_cap)
                    if cap < 2:
                        continue
                placements.append((r1, r2))
                
        best_ev = -9999
        
        for discard_idx in range(3):
            c1_idx = (discard_idx + 1) % 3
            c2_idx = (discard_idx + 2) % 3
            
            c1 = dealt[c1_idx]
            c2 = dealt[c2_idx]
            discard_card = dealt[discard_idx]
            
            for r1, r2 in placements:
                next_board = {
                    "top": current_board["top"][:],
                    "middle": current_board["middle"][:],
                    "bottom": current_board["bottom"][:],
                    "discards": current_board["discards"][:]
                }
                next_board[r1].append(c1)
                next_board[r2].append(c2)
                next_board["discards"].append(discard_card)
                
                ev = dfs(next_board, turn_idx + 1, deals)
                if ev > best_ev:
                    best_ev = ev
                    
        return best_ev

    for i in range(500):
        # Sample 12 cards for the 4 turns (T1, T2, T3, T4)
        sampled_cards = random.sample(remaining_deck, 12)
        deals = [
            sampled_cards[0:3],
            sampled_cards[3:6],
            sampled_cards[6:9],
            sampled_cards[9:12]
        ]
        dfs(board, 0, deals)
        
        if (i+1) % 50 == 0:
            elapsed_so_far = time.time() - start_time
            print(f"Processed {i+1}/500 samples... (elapsed: {elapsed_so_far:.2f}s)")
            
    elapsed = time.time() - start_time
    
    print(f"Evaluated {leaves_evaluated} leaf nodes in total.")
    print(f"Total Elapsed time: {elapsed:.4f} seconds")
    print(f"Average time per sample: {elapsed/500:.4f} seconds")

if __name__ == '__main__':
    bench_500_samples()
