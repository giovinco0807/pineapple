import sys
import random
from pathlib import Path

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

def merge_board(board, placement):
    final_top = board.get('top', []) + placement.get('top', [])
    final_mid = board.get('mid', []) + placement.get('mid', [])
    final_bot = board.get('bot', []) + placement.get('bot', [])
    return {'top': final_top, 'mid': final_mid, 'bot': final_bot}

def simulate_full_run():
    deck = generate_random_deck()
    
    # T0
    t0_cards = deck[:5]
    t0_placements = generate_t0_placements(t0_cards)
    if not t0_placements: return
    t0_board = random.choice(t0_placements)
    
    print(f"=== 新しいハンド ===")
    print(f"[T0 Hand] {' '.join(t0_cards)}")
    print(f"[T0 Board (5枚)] {format_board(t0_board)}")
    
    # T1
    t1_cards = deck[5:8]
    t1_actions = generate_t1_actions(t0_board, t1_cards)
    if not t1_actions: return
    t1_action = random.choice(t1_actions)
    t1_board = merge_board(t0_board, t1_action['place'])
    
    print(f"[T1 Dealt] {' '.join(t1_cards)}")
    print(f"  -> Discard [{t1_action['discard'][0]}]")
    print(f"[T1 Board (7枚)] {format_board(t1_board)}")
    
    # T2
    t2_cards = deck[8:11]
    t2_actions = generate_t1_actions(t1_board, t2_cards)
    if not t2_actions: return
    t2_action = random.choice(t2_actions)
    t2_board = merge_board(t1_board, t2_action['place'])
    
    print(f"[T2 Dealt] {' '.join(t2_cards)}")
    print(f"  -> Discard [{t2_action['discard'][0]}]")
    print(f"[T2 Board (9枚)] {format_board(t2_board)}")
    
    # T3
    t3_cards = deck[11:14]
    t3_actions = generate_t1_actions(t2_board, t3_cards)
    if not t3_actions: return
    t3_action = random.choice(t3_actions)
    t3_board = merge_board(t2_board, t3_action['place'])
    
    print(f"[T3 Dealt] {' '.join(t3_cards)}")
    print(f"  -> Discard [{t3_action['discard'][0]}]")
    print(f"[T3 Board (11枚)] {format_board(t3_board)}")
    print("-" * 60)

if __name__ == "__main__":
    for _ in range(3):
        simulate_full_run()
