"""
T0 PolicyNet Pre-Filter: Generate top-50 placements per hand for Rust evaluation.

Uses the trained T0 BC PolicyNet to rank all valid T0 placements,
selects the top-50 by probability, and outputs them in Rust-compatible format.

Output JSON format (one hand per line):
  {"hand": "Ad 8c 4s 3d 2s", "filtered_placements": ["Top[Ad] Mid[8c 4s] Bot[3d 2s]", ...]}

Usage:
    python ai/training/generate_filtered_t0.py --n-hands 500 --top-k 50 --output ai/data/filtered_t0.json
    python ai/training/generate_filtered_t0.py --n-hands 10 --top-k 50 --output test_filter.json --verbose
"""
import json
import sys
import random
import argparse
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM, ALL_CARDS
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS
from ai.models.networks import PolicyNetwork

# Rust uses "JK" for jokers; Python encoding uses "X1"/"X2"
# Rust card_to_string: rank_to_char(rank) + suit_to_char(suit)
# Python ALL_CARDS: "2h".."Ac" + "X1","X2"

PYTHON_RANKS = "23456789TJQKA"
PYTHON_SUITS = "hdcs"

# Mapping: Python card strings to Rust card strings
def python_to_rust_card(card: str) -> str:
    """Convert Python card notation to Rust notation.
    
    Python: '2h', 'Ts', 'Ah', 'X1', 'X2'
    Rust:   '2h', 'Ts', 'Ah', 'JK', 'JK'
    
    The suit order is the same (h,d,c,s → mapped to Rust's s,h,d,c).
    Actually let's check: Python SUITS = "hdcs", Rust SUIT_CHARS = ['s','h','d','c']
    Python suit indices: h=0, d=1, c=2, s=3
    Rust suit indices:   s=0, h=1, d=2, c=3
    
    But in card strings both use the literal character, so "Ah" is "Ah" in both.
    The only difference is Jokers.
    """
    if card.startswith("X"):
        return "JK"
    return card


def rust_to_python_card(card: str) -> str:
    """Convert Rust card notation to Python notation."""
    if card == "JK":
        return "X1"  # We'll track joker count separately
    return card


def generate_deck_python() -> list:
    """Generate the full 54-card deck in Python notation."""
    deck = []
    for s in PYTHON_SUITS:
        for r in PYTHON_RANKS:
            deck.append(f"{r}{s}")
    deck.append("X1")
    deck.append("X2")
    return deck


def action_to_rust_placement(action, dealt_cards: list) -> str:
    """Convert a Python Action to Rust placement string format.
    
    IMPORTANT: Must iterate dealt_cards in index order (0→4) to match
    Rust's format_placement which iterates hand[0]..hand[4].
    
    Format: "Top[Ks 2h] Mid[6s] Bot[9d Jh]"
    """
    # Build card→position lookup
    card_to_pos = {}
    for card, pos in action.placements:
        card_to_pos[card] = pos
    
    # Iterate in hand order (same as Rust's format_placement)
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card in dealt_cards:
        pos = card_to_pos.get(card)
        if pos:
            rust_card = python_to_rust_card(card)
            by_pos[pos].append(rust_card)
    
    top_str = " ".join(by_pos["top"])
    mid_str = " ".join(by_pos["middle"])
    bot_str = " ".join(by_pos["bottom"])
    
    return f"Top[{top_str}] Mid[{mid_str}] Bot[{bot_str}]"


def load_policy_net(model_path: str, device: str = "cpu") -> PolicyNetwork:
    """Load the trained T0 BC PolicyNet."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Detect input_dim from checkpoint
    first_key = list(checkpoint.keys())[0]
    input_dim = checkpoint[first_key].shape[1] if first_key == "net.0.weight" else STATE_DIM
    
    model = PolicyNetwork(input_dim=input_dim, max_actions=MAX_ACTIONS)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def encode_t0_state(dealt_cards: list) -> np.ndarray:
    """Encode the T0 state (empty board + dealt cards)."""
    obs = Observation(
        board_self=Board(),
        board_opponent=Board(),
        dealt_cards=dealt_cards,
        known_discards_self=[],
        turn=0,
        is_btn=True,
    )
    return encode_state(obs)


def get_top_k_placements(
    model: PolicyNetwork,
    dealt_cards: list,
    top_k: int = 50,
    device: str = "cpu",
) -> list:
    """Get top-K placements for a given hand using PolicyNet.
    
    Returns: list of (placement_string, probability) tuples, sorted by probability desc.
    """
    # 1. Get all valid actions
    board = Board()
    all_actions = get_initial_actions(dealt_cards, board)
    n_actions = len(all_actions)
    
    if n_actions == 0:
        return []
    
    # 2. Encode state
    state = encode_t0_state(dealt_cards)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 3. Create valid mask
    valid_mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool, device=device)
    valid_mask[0, :n_actions] = True
    
    # 4. Forward pass
    with torch.no_grad():
        probs = model(state_tensor, valid_mask)  # (1, MAX_ACTIONS)
    
    probs = probs[0].cpu().numpy()
    
    # 5. Get top-K action indices
    actual_k = min(top_k, n_actions)
    top_indices = np.argsort(probs)[::-1][:actual_k]
    
    # 6. Convert to Rust placement strings
    results = []
    for idx in top_indices:
        if idx < n_actions:
            action = all_actions[idx]
            placement_str = action_to_rust_placement(action, dealt_cards)
            results.append((placement_str, float(probs[idx])))
    
    return results


def generate_random_hand(rng: random.Random) -> list:
    """Generate a random 5-card hand from the 54-card deck."""
    deck = generate_deck_python()
    rng.shuffle(deck)
    return deck[:5]


def hand_to_rust_string(dealt_cards: list) -> str:
    """Convert Python hand to Rust hand string."""
    return " ".join(python_to_rust_card(c) for c in dealt_cards)


def main():
    parser = argparse.ArgumentParser(description="Generate PolicyNet-filtered T0 placements")
    parser.add_argument("--n-hands", type=int, default=500,
                        help="Number of random hands to generate")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of top placements to keep per hand")
    parser.add_argument("--output", type=str, default="ai/data/filtered_t0.json",
                        help="Output JSON file path")
    parser.add_argument("--model", type=str, 
                        default="ai/models/t0_bc/bc_policy_best.pt",
                        help="Path to trained PolicyNet weights")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-hand info")
    args = parser.parse_args()
    
    print(f"=== T0 PolicyNet Pre-Filter ===")
    print(f"Hands: {args.n_hands}")
    print(f"Top-K: {args.top_k}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_policy_net(args.model, device)
    print(f"Model loaded: input_dim={model.net[0].in_features}, "
          f"output_dim={model.net[-1].out_features}")
    print()
    
    # Generate hands and filter
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_placements = 0
    total_filtered = 0
    
    for i in range(args.n_hands):
        hand = generate_random_hand(rng)
        rust_hand = hand_to_rust_string(hand)
        
        # Get all valid actions count
        all_actions = get_initial_actions(hand, Board())
        n_total = len(all_actions)
        
        # Get top-K
        top_placements = get_top_k_placements(model, hand, args.top_k, device)
        n_filtered = len(top_placements)
        
        total_placements += n_total
        total_filtered += n_filtered
        
        record = {
            "hand_idx": i,
            "hand": rust_hand,
            "n_total": n_total,
            "n_filtered": n_filtered,
            "filtered_placements": [p[0] for p in top_placements],
        }
        results.append(record)
        
        if args.verbose or (i + 1) % 50 == 0:
            top_prob = top_placements[0][1] if top_placements else 0
            coverage = sum(p[1] for p in top_placements) if top_placements else 0
            print(f"[{i+1:>4}/{args.n_hands}] {rust_hand} | "
                  f"{n_total} → {n_filtered} placements | "
                  f"Top-1 prob: {top_prob:.4f} | "
                  f"Top-{n_filtered} coverage: {coverage:.4f}")
    
    # Save output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    avg_total = total_placements / args.n_hands
    avg_filtered = total_filtered / args.n_hands
    reduction = (1 - avg_filtered / avg_total) * 100 if avg_total > 0 else 0
    
    print(f"\n=== Summary ===")
    print(f"Hands generated: {args.n_hands}")
    print(f"Avg placements/hand: {avg_total:.1f} → {avg_filtered:.1f} "
          f"({reduction:.1f}% reduction)")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
