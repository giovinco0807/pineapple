import random
import json
import os
import sys

# Add parent dir to path to import t0_generator and t1_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heuristic_bot.t0_generator import generate_t0_placements
from heuristic_bot.t1_generator import generate_t1_actions

RANKS = "23456789TJQKA"
SUITS = "shdc"

def create_deck(include_jokers=True):
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    if include_jokers:
        deck.extend(["X1", "X2"])
    return deck

def generate_random_t3_state(include_jokers=True):
    """
    Simulates a game up to T3 using the heuristic bot.
    Returns:
      board: dict with 'top', 'mid', 'bot' (9 cards total)
      dealt: list of 3 cards for T3
      discards: list of discarded cards so far (2 cards)
      deck: remaining deck
    """
    deck = create_deck(include_jokers=include_jokers)
    random.shuffle(deck)
    
    # T0 (5 cards)
    t0_cards = [deck.pop() for _ in range(5)]
    t0_placements = generate_t0_placements(t0_cards)
    if not t0_placements:
        # Fallback (should not happen)
        t0_placements = [{'top': [], 'mid': [], 'bot': t0_cards}]
    board = random.choice(t0_placements)
    discards = []
    
    # T1 (3 cards drawn, 2 placed, 1 discarded)
    t1_cards = [deck.pop() for _ in range(3)]
    t1_actions = generate_t1_actions(board, t1_cards)
    if not t1_actions:
        return None # Invalid state path
    t1_action = random.choice(t1_actions)
    
    board['top'].extend(t1_action['place']['top'])
    board['mid'].extend(t1_action['place']['mid'])
    board['bot'].extend(t1_action['place']['bot'])
    discards.extend(t1_action['discard'])
    
    # T2 (3 cards drawn, 2 placed, 1 discarded)
    t2_cards = [deck.pop() for _ in range(3)]
    t2_actions = generate_t1_actions(board, t2_cards)
    if not t2_actions:
        return None # Invalid state path
    t2_action = random.choice(t2_actions)
    
    board['top'].extend(t2_action['place']['top'])
    board['mid'].extend(t2_action['place']['mid'])
    board['bot'].extend(t2_action['place']['bot'])
    discards.extend(t2_action['discard'])
    
    # T3 (3 cards drawn, but not placed yet)
    t3_cards = [deck.pop() for _ in range(3)]
    
    return {
        "board_top": board['top'],
        "board_mid": board['mid'],
        "board_bot": board['bot'],
        "discards": discards,
        "dealt": t3_cards
    }

def print_state(state):
    print("--- T3 State ---")
    print(f"Top: {' '.join(state['board_top'])}")
    print(f"Mid: {' '.join(state['board_mid'])}")
    print(f"Bot: {' '.join(state['board_bot'])}")
    print(f"Discards: {' '.join(state['discards'])}")
    print(f"Dealt: {' '.join(state['dealt'])}")

if __name__ == "__main__":
    success_count = 0
    for i in range(10):
        state = generate_random_t3_state()
        if state:
            print_state(state)
            success_count += 1
    print(f"\nSuccessfully generated {success_count} T3 states.")
