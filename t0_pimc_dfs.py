import time
import random
import itertools
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ai.engine.game_engine import evaluate_hand, get_top_royalty, get_middle_royalty, get_bottom_royalty, check_fl_entry

def evaluate_full_board(top, mid, bot):
    top_val = evaluate_hand(top, 3)
    mid_val = evaluate_hand(mid, 5)
    bot_val = evaluate_hand(bot, 5)
    
    if top_val > mid_val or mid_val > bot_val:
        return -6  # Bust penalty
        
    top_r = get_top_royalty(top)
    mid_r = get_middle_royalty(mid)
    bot_r = get_bottom_royalty(bot)
    
    fl, cards = check_fl_entry(top)
    fl_bonus = cards if fl else 0
        
    return top_r + mid_r + bot_r + fl_bonus

def dfs(current_board, turn_idx, deals):
    if turn_idx == 4:
        return evaluate_full_board(current_board["top"], current_board["middle"], current_board["bottom"])
        
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
        
        # Optimize symmetric placements in same row
        seen_placements = set()
        
        for r1, r2 in placements:
            if r1 == r2:
                # Order doesn't matter, avoid duplicate DFS
                p_key = (r1, r2, tuple(sorted([c1, c2])))
            else:
                p_key = (r1, r2, c1, c2)
                
            if p_key in seen_placements:
                continue
            seen_placements.add(p_key)
            
            next_board = {
                "top": current_board["top"][:],
                "middle": current_board["middle"][:],
                "bottom": current_board["bottom"][:]
            }
            next_board[r1].append(c1)
            next_board[r2].append(c2)
            
            ev = dfs(next_board, turn_idx + 1, deals)
            if ev > best_ev:
                best_ev = ev
                
    return best_ev

def evaluate_placement_worker(args):
    board, scenarios = args
    total_ev = 0
    for deals in scenarios:
        total_ev += dfs(board, 0, deals)
    return total_ev / len(scenarios)

def run_t0_pimc_dfs():
    print("Generating T0 Ranking using TRUE PIMC (DFS)...")
    start_time = time.time()
    
    t0_cards = ["Ah", "As", "Kd", "Qs", "Jh"]
    print(f"Starting Hand: {t0_cards}")
    
    # Generate placements
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
            
    # For speed, let's just pick 10 intuitively decent placements rather than 232.
    # We want to see if PIMC correctly ranks them.
    selected_placements = valid_placements[:15] # Just take first 15 for a quick benchmark
    
    full_deck = [r+s for r in "23456789TJQKA" for s in "shdc"]
    remaining_deck = [c for c in full_deck if c not in t0_cards]
    
    N_SAMPLES = 20
    print(f"Running DFS PIMC on {len(selected_placements)} placements with {N_SAMPLES} samples each...")
    
    scenarios = []
    for _ in range(N_SAMPLES):
        sampled = random.sample(remaining_deck, 12)
        deals = [
            sampled[0:3],
            sampled[3:6],
            sampled[6:9],
            sampled[9:12]
        ]
        scenarios.append(deals)

    args_list = [(board, scenarios) for board in selected_placements]
    
    results = []
    with ProcessPoolExecutor() as executor:
        evs = list(executor.map(evaluate_placement_worker, args_list))
        
    for idx, ev in enumerate(evs):
        results.append((ev, selected_placements[idx]))
            
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- DFS PIMC PLACEMENTS RANKING ---")
    for i in range(len(results)):
        ev, b = results[i]
        top = ",".join(b["top"]) if b["top"] else "-"
        mid = ",".join(b["middle"]) if b["middle"] else "-"
        bot = ",".join(b["bottom"]) if b["bottom"] else "-"
        print(f"Rank {i+1}: EV = {ev:6.3f} | Top: [{top:^11}] Mid: [{mid:^14}] Bot: [{bot:^14}]")
        
    print(f"\nTotal elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    run_t0_pimc_dfs()
