#!/usr/bin/env python3
"""Generate complete game trajectories for T1-T4 evaluation.

Reads T0 CFR results (best placement per hand), then for each hand:
1. Parse the best T0 placement to get the initial board state
2. Deal random T1-T4 hands (3 cards per turn × 4 turns = 12 cards)
3. Repeat N times per hand to create N trajectories

Output: JSON array of game scenarios for the Rust GameBatch command.
"""

import json
import random
import argparse
import sys
from pathlib import Path

# Card utilities
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']

def make_deck():
    """Create a standard 52-card deck + Joker."""
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    deck.append("Jo")  # Joker
    return deck

def parse_t0_result_line(line):
    """Parse a JSONL line from T0 CFR results.
    
    Expected format:
    {"hand": "6s Ks 9d Jh 2c", "placements": [{"placement": "Top[Ks] Mid[6s 2c] Bot[9d Jh]", "ev": 25.3}, ...]}
    
    Returns (hand_cards, best_board) or None.
    """
    data = json.loads(line)
    hand = data.get("hand", "")
    placements = data.get("placements", [])
    
    if not placements:
        return None
    
    # Best placement is first (sorted by EV descending)
    best = placements[0]
    placement_str = best.get("placement", "")
    
    # Parse placement string: "Top[Ks] Mid[6s 2c] Bot[9d Jh]"
    board = parse_placement(placement_str)
    if board is None:
        return None
    
    hand_cards = hand.split()
    return hand_cards, board, best.get("ev", 0.0)

def parse_placement(s):
    """Parse 'Top[Ks] Mid[6s 2c] Bot[9d Jh]' into board dict."""
    import re
    
    top_match = re.search(r'Top\[([^\]]*)\]', s)
    mid_match = re.search(r'Mid\[([^\]]*)\]', s)
    bot_match = re.search(r'Bot\[([^\]]*)\]', s)
    
    if not all([top_match, mid_match, bot_match]):
        return None
    
    top = [c for c in top_match.group(1).split() if c and c != '-']
    mid = [c for c in mid_match.group(1).split() if c and c != '-']
    bot = [c for c in bot_match.group(1).split() if c and c != '-']
    
    return {"top": top, "mid": mid, "bot": bot}

def generate_trajectories(hand_cards, board, n_trajectories, rng):
    """Generate N game trajectories from a T0 board state.
    
    Each trajectory deals 4 turns of 3 cards each from the remaining deck.
    """
    # Cards already used
    used = set(hand_cards)
    full_deck = make_deck()
    remaining = [c for c in full_deck if c not in used]
    
    trajectories = []
    for _ in range(n_trajectories):
        # Shuffle remaining and deal 12 cards (4 turns × 3)
        rng.shuffle(remaining)
        
        if len(remaining) < 12:
            print(f"Warning: not enough cards remaining ({len(remaining)})", file=sys.stderr)
            continue
        
        turns = []
        for t in range(4):
            turn_hand = remaining[t*3 : t*3 + 3]
            turns.append({
                "turn": t + 1,
                "hand": turn_hand
            })
        
        trajectories.append({
            "t0_board": board,
            "turns": turns
        })
    
    return trajectories

def main():
    parser = argparse.ArgumentParser(description="Generate game trajectories for T1-T4 evaluation")
    parser.add_argument("--input", required=True, help="T0 CFR results JSONL file")
    parser.add_argument("--output", required=True, help="Output JSON file for GameBatch")
    parser.add_argument("--trajectories", type=int, default=10, help="Trajectories per hand (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-hands", type=int, default=0, help="Max hands to process (0=all)")
    parser.add_argument("--shard", type=int, default=-1, help="Shard index for distributed processing")
    parser.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    
    # Read T0 results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    hands = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = parse_t0_result_line(line)
            if result:
                hands.append(result)
    
    print(f"Loaded {len(hands)} hands from {args.input}")
    
    if args.max_hands > 0:
        hands = hands[:args.max_hands]
        print(f"Using first {len(hands)} hands")
    
    # Shard if requested
    if args.shard >= 0 and args.n_shards > 1:
        hands = hands[args.shard::args.n_shards]
        print(f"Shard {args.shard}/{args.n_shards}: {len(hands)} hands")
    
    # Generate trajectories
    games = []
    game_id = 0
    for i, (hand_cards, board, t0_ev) in enumerate(hands):
        trajs = generate_trajectories(hand_cards, board, args.trajectories, rng)
        for traj in trajs:
            traj["game_id"] = game_id
            traj["t0_hand"] = " ".join(hand_cards)
            traj["t0_ev"] = t0_ev
            games.append(traj)
            game_id += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {game_id} games from {i+1} hands...")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(games, f, indent=None)
    
    print(f"\nDone: {len(games)} games written to {args.output}")
    print(f"  Hands: {len(hands)} × {args.trajectories} trajectories")
    print(f"  Total turn evaluations: {len(games) * 4}")

if __name__ == "__main__":
    main()
