"""
OFC Pineapple - NN-filtered T0 Hand Generator for Rust Solver

Generates random T0 hands, runs PolicyNet inference to select top-K placements,
and outputs a JSON file for Rust's T0BatchFiltered command to evaluate.

This replaces the Rust 2-pass screening with NN-based pre-filtering.

Usage:
    python ai/generate_filtered_t0.py \
        --model ai/models/t0_ranking_v2/policy.onnx \
        --hands 200 --top-k 100 \
        --output t0_nn_filtered.json

The output JSON is consumed by:
    cfr_solver t0-batch-filtered --input t0_nn_filtered.json --samples 30 --nesting 3,2,2
"""
import sys
import json
import random
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.engine.encoding import (
    Board, Observation, encode_state, STATE_DIM,
    ALL_CARDS, CARD_TO_IDX
)
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS


def python_card_to_rust(card: str) -> str:
    """Convert Python card notation to Rust notation."""
    if card == "X1":
        return "JK"
    if card == "X2":
        return "JK"
    return card


def generate_random_hand(rng: random.Random) -> list:
    """Generate a random 5-card T0 hand."""
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    return deck[:5]


def encode_t0_state(hand: list) -> np.ndarray:
    """Encode a T0 pre-placement state for NN inference."""
    obs = Observation(
        board_self=Board(),
        board_opponent=Board(),
        dealt_cards=hand,
        known_discards_self=[],
        turn=0,
        is_btn=True,
        is_fl=False,
        opp_is_fl=False,
    )
    return encode_state(obs)


def get_valid_mask_and_actions(hand: list):
    """Get valid action mask and action descriptions for a hand."""
    board = Board()
    actions = get_initial_actions(hand, board)
    
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    for i in range(min(len(actions), MAX_ACTIONS)):
        mask[i] = True
    
    return mask, actions


def action_to_rust_desc(action) -> str:
    """Convert Action to Rust's format_placement format: 'Top[...] Mid[...] Bot[...]'
    
    Cards within each row are sorted for canonical matching with Rust.
    """
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card, pos in action.placements:
        rust_card = python_card_to_rust(card)
        by_pos[pos].append(rust_card)
    
    return "Top[{}] Mid[{}] Bot[{}]".format(
        " ".join(sorted(by_pos["top"])),
        " ".join(sorted(by_pos["middle"])),
        " ".join(sorted(by_pos["bottom"])),
    )


def run_nn_filtering(
    model_path: str,
    n_hands: int,
    top_k: int,
    output_path: str,
    seed: int = 42,
):
    """Generate hands, run NN, pick top-K, save for Rust."""
    import onnxruntime as ort
    
    print(f"=== NN-Filtered T0 Generator ===")
    print(f"Model: {model_path}")
    print(f"Hands: {n_hands} | Top-K: {top_k}")
    print(f"Output: {output_path}")
    print()
    
    # Load ONNX model
    sess = ort.InferenceSession(model_path)
    print(f"ONNX model loaded ({Path(model_path).stat().st_size / 1024:.0f} KB)")
    
    rng = random.Random(seed)
    results = []
    
    start = time.time()
    
    for i in range(n_hands):
        hand = generate_random_hand(rng)
        
        # Encode state
        state = encode_t0_state(hand).reshape(1, -1)
        
        # Get valid mask and all actions
        mask, actions = get_valid_mask_and_actions(hand)
        n_actions = len(actions)
        
        # Run NN inference
        probs = sess.run(None, {
            "state": state.astype(np.float32),
            "valid_mask": mask.reshape(1, -1),
        })[0][0]  # (250,)
        
        # Sort by NN score descending, take top-K
        valid_indices = np.where(mask)[0]
        valid_probs = probs[valid_indices]
        sorted_order = np.argsort(-valid_probs)
        
        k = min(top_k, n_actions)
        selected_indices = valid_indices[sorted_order[:k]]
        
        # Convert to Rust format_placement strings
        filtered_descs = []
        for idx in selected_indices:
            desc = action_to_rust_desc(actions[idx])
            filtered_descs.append(desc)
        
        # Hand string in Rust format: "Ad 8c 4s 3d 2s"
        rust_hand_str = " ".join(python_card_to_rust(c) for c in hand)
        
        results.append({
            "hand_idx": i,
            "hand": rust_hand_str,
            "n_total_actions": n_actions,
            "n_selected": len(filtered_descs),
            "filtered_placements": filtered_descs,
        })
        
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n_hands - i - 1) / rate
            print(f"  [{i+1:>4}/{n_hands}] {rust_hand_str} | "
                  f"{n_actions} actions → {len(filtered_descs)} selected | "
                  f"{rate:.0f} hands/s | ETA: {eta:.0f}s")
    
    # Save JSON
    with open(output_path, "w") as f:
        json.dump(results, f)
    
    elapsed = time.time() - start
    print(f"\n=== Complete ===")
    print(f"Generated {n_hands} hands in {elapsed:.1f}s ({n_hands/elapsed:.0f} hands/s)")
    print(f"Output: {output_path} ({Path(output_path).stat().st_size / 1024:.0f} KB)")
    
    # Stats
    total_actions = sum(r["n_total_actions"] for r in results)
    total_selected = sum(r["n_selected"] for r in results)
    print(f"Total actions: {total_actions} → {total_selected} selected "
          f"({100*total_selected/total_actions:.0f}% kept)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN-filtered T0 hand generator")
    parser.add_argument("--model", default="ai/models/t0_ranking_v2/policy.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--hands", type=int, default=200,
                        help="Number of random hands to generate")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top-K placements to keep per hand")
    parser.add_argument("--output", default="t0_nn_filtered.json",
                        help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    run_nn_filtering(args.model, args.hands, args.top_k, args.output, args.seed)
