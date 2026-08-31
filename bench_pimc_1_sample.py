import time
import itertools

def bench_1_sample():
    # 1. Initialize
    # Override deck to be deterministic
    deck = [
        # T0 cards
        "2h", "3h", "4h", "5h", "6h",
        # T1 cards
        "7h", "8h", "9h",
        # T2 cards
        "Th", "Jh", "Qh",
        # T3 cards
        "Kh", "Ah", "2s",
        # T4 cards
        "3s", "4s", "5s"
    ]
    # We don't actually need to use the engine's deck if we are doing PIMC DFS
    
    # Let's set up the board after T0 manually
    # Suppose we place 2h->Top, 3h->Mid, 4h,5h,6h->Bot
    board = {
        "top": ["2h"],
        "middle": ["3h"],
        "bottom": ["4h", "5h", "6h"],
        "discards": []
    }
    
    # The deterministic future deals
    deals = [
        ["7h", "8h", "9h"], # T1
        ["Th", "Jh", "Qh"], # T2
        ["Kh", "Ah", "2s"], # T3
        ["3s", "4s", "5s"], # T4
    ]
    
    print("Starting PIMC 1 sample DFS in Python...")
    start_time = time.time()
    
    # DFS to find the maximum score
    # We will track the number of leaf nodes evaluated
    leaves_evaluated = 0
    
    def dfs(current_board, turn_idx):
        nonlocal leaves_evaluated
        if turn_idx == 4:
            # Game over, evaluate terminal
            leaves_evaluated += 1
            # We mock the evaluation for speed in this test, or we can use real eval
            return 0 # Mock EV for speed test, but let's actually do something basic to simulate work
            
        # Get dealt cards
        dealt = deals[turn_idx]
        
        # Determine available slots
        top_cap = 3 - len(current_board["top"])
        mid_cap = 5 - len(current_board["middle"])
        bot_cap = 5 - len(current_board["bottom"])
        
        # Valid row choices for placing 2 cards
        rows = []
        if top_cap >= 1: rows.append("top")
        if mid_cap >= 1: rows.append("middle")
        if bot_cap >= 1: rows.append("bottom")
        
        placements = []
        for r1 in rows:
            for r2 in rows:
                # check capacity if same row
                if r1 == r2:
                    cap = top_cap if r1 == "top" else (mid_cap if r1 == "middle" else bot_cap)
                    if cap < 2:
                        continue
                placements.append((r1, r2))
                
        best_ev = -9999
        
        # 3 possible discards
        for discard_idx in range(3):
            c1_idx = (discard_idx + 1) % 3
            c2_idx = (discard_idx + 2) % 3
            
            c1 = dealt[c1_idx]
            c2 = dealt[c2_idx]
            discard_card = dealt[discard_idx]
            
            for r1, r2 in placements:
                # Optimization: if c1 and c2 go to different rows, order matters only if rows are different
                # Actually, order of cards to different rows matters.
                # If they go to the same row, order doesn't matter, we can avoid duplicate state
                
                # Apply placement
                next_board = {
                    "top": current_board["top"][:],
                    "middle": current_board["middle"][:],
                    "bottom": current_board["bottom"][:],
                    "discards": current_board["discards"][:]
                }
                next_board[r1].append(c1)
                next_board[r2].append(c2)
                next_board["discards"].append(discard_card)
                
                ev = dfs(next_board, turn_idx + 1)
                if ev > best_ev:
                    best_ev = ev
                    
        return best_ev

    dfs(board, 0)
    elapsed = time.time() - start_time
    
    print(f"Evaluated {leaves_evaluated} leaf nodes.")
    print(f"Elapsed time: {elapsed:.4f} seconds")

if __name__ == '__main__':
    bench_1_sample()
