"""
T0 PlacementNet v4 Pre-Filter: Generate top-50 placements per hand for Rust.

Uses the T0 PlacementNet v4 (Config F with hand features) to score all 232
valid T0 placements, selects top-50 by likelihood, outputs Rust-compatible format.

Output JSONL (one hand per line):
  {"hand": "Ad 8c 4s 3d 2s", "n_total": 232, "n_filtered": 50, "filtered_placements": [...]}

Usage:
    python ai/training/generate_filtered_t0.py --n-hands 500 --top-k 50
    python ai/training/generate_filtered_t0.py --n-hands 10 --top-k 50 --verbose
"""
import json
import sys
import random
import argparse
from pathlib import Path
from itertools import combinations
from collections import Counter

import torch
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.train_t0_placement import T0PlacementNet, encode_card, compute_hand_features
from ai.train_t0_placement import CARD_DIM, NUM_ROWS, MAX_CARDS


# Card notation
RANKS = "23456789TJQKA"
SUITS = "shdc"


def generate_deck():
    deck = [f"{r}{s}" for s in SUITS for r in RANKS]
    deck += ["X1", "X2"]  # Jokers
    return deck


def card_str_to_dict(card_str):
    """Convert '2h' → {'rank':'2','suit':'hearts'} etc."""
    suit_map = {'s':'spades','h':'hearts','d':'diamonds','c':'clubs'}
    if card_str.startswith("X") or card_str == "JK":
        return {'rank':'Joker','suit':'joker'}
    return {'rank': card_str[0], 'suit': suit_map.get(card_str[1], card_str[1])}


def card_dict_to_rust(card_dict):
    """{'rank':'A','suit':'hearts'} → 'Ah'"""
    suit_map = {'spades':'s','hearts':'h','diamonds':'d','clubs':'c','joker':''}
    if card_dict.get('rank') == 'Joker':
        return 'JK'
    return f"{card_dict['rank']}{suit_map[card_dict['suit']]}"


def enumerate_t0_placements(hand_dicts):
    """Enumerate all valid T0 placements (232 total for 5 cards).
    
    Constraints: top <= 3, mid <= 5, bot <= 5
    T0 has 5 cards, so valid splits sum to 5.
    Returns list of dicts: [{'top': [...], 'mid': [...], 'bot': [...]}, ...]
    """
    placements = []
    cards = list(range(5))
    
    # Enumerate all ways to assign 5 cards to 3 rows
    for t in range(min(4, 6)):  # top: 0-3
        for m in range(6 - t):  # mid: 0-5
            b = 5 - t - m  # bot: remainder
            if b < 0 or b > 5 or m > 5 or t > 3:
                continue
            # Generate all combinations for this split
            for top_cards in combinations(cards, t):
                remaining = [c for c in cards if c not in top_cards]
                for mid_cards in combinations(remaining, m):
                    bot_cards = [c for c in remaining if c not in mid_cards]
                    placements.append({
                        'top': [hand_dicts[i] for i in top_cards],
                        'mid': [hand_dicts[i] for i in mid_cards],
                        'bot': [hand_dicts[i] for i in bot_cards],
                    })
    return placements


def score_placement(model, features_tensor, hand_dicts, placement, device):
    """Score a single placement using the model's per-card log-probabilities."""
    # Map each card to its row in this placement
    card_rows = {}
    for row_name, row_idx in [('top', 0), ('mid', 1), ('bot', 2)]:
        for card in placement[row_name]:
            key = f"{card['rank']}_{card['suit']}"
            card_rows[key] = row_idx
    
    # Sum log-probs for each card's assigned row
    with torch.no_grad():
        logits, _ = model(features_tensor)
        probs = torch.softmax(logits, dim=-1)[0]  # (5, 3)
    
    score = 0.0
    for i, card in enumerate(hand_dicts):
        key = f"{card['rank']}_{card['suit']}"
        row = card_rows.get(key, 2)
        score += torch.log(probs[i, row] + 1e-8).item()
    
    return score


def format_rust_placement(placement):
    """Format placement as Rust string: 'Top[Ks 2h] Mid[6s] Bot[9d Jh]'"""
    parts = []
    for row_name in ['top', 'mid', 'bot']:
        cards_str = " ".join(card_dict_to_rust(c) for c in placement[row_name])
        label = row_name.capitalize() if row_name != 'mid' else 'Mid'
        parts.append(f"{label}[{cards_str}]")
    return " ".join(parts)


def load_model(model_path, device):
    """Load T0 PlacementNet v4."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = T0PlacementNet(
        d_model=config.get('d_model', 128),
        nhead=4,
        num_layers=config.get('num_layers', 4),
        dim_ff=config.get('d_model', 128) * 2,
        dropout=config.get('dropout', 0.2),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"Loaded model: v={config.get('version','?')}, "
          f"d={config.get('d_model',128)}, "
          f"params={config.get('n_params','?')}, "
          f"val_hand_acc={checkpoint.get('val_hand_acc',0):.4f}")
    return model


def filter_hand(model, hand_strs, top_k, device):
    """Filter placements for one hand. Returns (n_total, top_k_placements)."""
    hand_dicts = [card_str_to_dict(c) for c in hand_strs]
    
    # Encode features (base + hand features)
    base_features = np.stack([encode_card(c) for c in hand_dicts])
    hand_features = compute_hand_features(hand_dicts)
    features = np.concatenate([base_features, hand_features], axis=1)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    # Get model predictions once
    with torch.no_grad():
        logits, _ = model(features_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)[0]  # (5, 3)
    
    # Enumerate all placements
    all_placements = enumerate_t0_placements(hand_dicts)
    
    # Score each placement
    scored = []
    for p in all_placements:
        score = 0.0
        for i, card in enumerate(hand_dicts):
            key = f"{card['rank']}_{card['suit']}"
            # Find which row this card is in
            row = 2  # default bot
            for rn, ri in [('top', 0), ('mid', 1), ('bot', 2)]:
                for pc in p[rn]:
                    if pc['rank'] == card['rank'] and pc['suit'] == card['suit']:
                        row = ri
                        break
            score += log_probs[i, row].item()
        scored.append((score, p))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    actual_k = min(top_k, len(scored))
    top_placements = [(format_rust_placement(p), s) for s, p in scored[:actual_k]]
    
    return len(all_placements), top_placements


def main():
    parser = argparse.ArgumentParser(description="T0 PlacementNet v4 Pre-Filter")
    parser.add_argument("--n-hands", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=str, default="ai/data/filtered_t0_v4.jsonl")
    parser.add_argument("--model", type=str, default="ai/models/t0_placement_net_v4.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== T0 PlacementNet v4 Pre-Filter ===")
    print(f"Hands: {args.n_hands}, Top-K: {args.top_k}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(args.model, device)

    rng = random.Random(args.seed)
    deck = generate_deck()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_placements = 0
    total_filtered = 0
    
    with open(output_path, 'w') as fout:
        for i in range(args.n_hands):
            rng.shuffle(deck)
            hand = deck[:5]
            rust_hand = " ".join("JK" if c.startswith("X") else c for c in hand)
            
            n_total, top_placements = filter_hand(model, hand, args.top_k, device)
            n_filtered = len(top_placements)
            
            total_placements += n_total
            total_filtered += n_filtered
            
            record = {
                "hand_idx": i,
                "hand": rust_hand,
                "n_total_actions": n_total,
                "n_selected": n_filtered,
                "filtered_placements": [p[0] for p in top_placements],
            }
            fout.write(json.dumps(record) + '\n')
            
            if args.verbose or (i + 1) % 100 == 0:
                top_score = top_placements[0][1] if top_placements else 0
                print(f"[{i+1:>4}/{args.n_hands}] {rust_hand} | "
                      f"{n_total} → {n_filtered} | top_score={top_score:.3f}")
    
    avg_total = total_placements / args.n_hands
    avg_filtered = total_filtered / args.n_hands
    reduction = (1 - avg_filtered / avg_total) * 100
    
    print(f"\n=== Summary ===")
    print(f"Hands: {args.n_hands}")
    print(f"Avg: {avg_total:.0f} → {avg_filtered:.0f} ({reduction:.1f}% reduction)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
