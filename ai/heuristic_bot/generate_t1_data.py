import sys
from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.heuristic_bot.t0_generator import generate_t0_placements, RANKS, SUITS
from ai.heuristic_bot.t1_generator import generate_t1_actions

def generate_random_deck():
    deck = [f"{r}{s}" for r in RANKS for s in SUITS] + ["X1", "X2"]
    random.shuffle(deck)
    return deck

def format_board(b):
    t = " ".join(b.get('top', [])).ljust(8)
    m = " ".join(b.get('mid', [])).ljust(15)
    bot = " ".join(b.get('bot', [])).ljust(15)
    return f"Top: {t} | Mid: {m} | Bot: {bot}"

def generate_combined_sample():
    deck = generate_random_deck()
    
    # T0 (5 cards)
    t0_cards = deck[:5]
    t0_placements = generate_t0_placements(t0_cards)
    
    if not t0_placements:
        return
        
    # Randomly select one T0 placement for variety
    t0_board = random.choice(t0_placements)
    
    # T1 (3 cards)
    t1_cards = deck[5:8]
    t1_actions = generate_t1_actions(t0_board, t1_cards)
    
    if not t1_actions:
        return
        
    print(f"=== Combined T0 -> T1 Generation ===")
    print(f"[T0 Hand] {' '.join(t0_cards)}")
    print(f"[T0 Board] {format_board(t0_board)}")
    print(f"[T1 Dealt] {' '.join(t1_cards)}")
    
    for action in t1_actions:
        p = action['place']
        d = action['discard'][0]
        
        # Merge to show final T1 board
        final_top = t0_board.get('top', []) + p.get('top', [])
        final_mid = t0_board.get('mid', []) + p.get('mid', [])
        final_bot = t0_board.get('bot', []) + p.get('bot', [])
        
        final_board = {'top': final_top, 'mid': final_mid, 'bot': final_bot}
        print(f"  -> Discard [{d}] -> Final Board: {format_board(final_board)}")
    print("-" * 60)

if __name__ == "__main__":
    print("Generating 5 random T0 -> T1 sequences...\n")
    for _ in range(5):
        generate_combined_sample()
