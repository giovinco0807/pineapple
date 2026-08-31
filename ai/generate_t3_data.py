#!/usr/bin/env python3
"""
T3 Teacher Data Generator (Backward Induction)

Uses the T4 Oracle NN as a "leaf evaluator" to compute action EVs for T3 states.

Two modes:
  - BTN T3: BB has already placed (11 cards). Evaluate each BTN T3 action
    by running T4 Oracle NN on the resulting T4 state. Simple & fast.
  - BB T3:  2-step minimax. For each BB T3 action, enumerate sampled BTN T3
    draws, find BTN's best response via T4 Oracle NN, and average.

Usage:
  python generate_t3_data.py --states 50000 --workers 4 --btn-samples 500
"""

import os
import sys
import time
import random
import argparse
import itertools
import numpy as np
import torch
from pathlib import Path
from itertools import combinations

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))

from engine.encoding import Board, Observation, encode_state, ALL_CARDS
from engine.action_space import (
    get_turn_actions, get_semantic_action_index, get_action_from_semantic_index,
    is_turn_action_valid, create_regular_turn_mask, REGULAR_TURN_ACTIONS
)
from training.train_t4_oracle import T4PolicyValueNet

ACTION_DIM = 27
RANKS = "23456789TJQKA"
SUITS = "shdc"


# ─────────────────────────────────────────────────────────────────────────────
# Board Generation Helpers (reused from generate_t4_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_deck():
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck

def apply_t0(board_top, board_mid, board_bot, t0_cards):
    t0_cards = sorted(t0_cards, key=lambda c: RANKS.index(c[0]))
    strategy = random.choice(["spread", "bot_heavy", "mid_heavy", "random"])
    if strategy == "spread":
        board_top.append(t0_cards[4])
        board_mid.extend(t0_cards[2:4])
        board_bot.extend(t0_cards[0:2])
    elif strategy == "bot_heavy":
        board_bot.extend(t0_cards[:3])
        board_mid.extend(t0_cards[3:5])
    elif strategy == "mid_heavy":
        board_mid.extend(t0_cards[:3])
        board_bot.extend(t0_cards[3:5])
    else:
        positions = ["top", "mid", "bot"]
        limits = {"top": 3, "mid": 5, "bot": 5}
        counts = {"top": 0, "mid": 0, "bot": 0}
        for card in t0_cards:
            valid = [p for p in positions if counts[p] < limits[p]]
            pos = random.choice(valid)
            if pos == "top": board_top.append(card)
            elif pos == "mid": board_mid.append(card)
            else: board_bot.append(card)
            counts[pos] += 1

def apply_tx(board_top, board_mid, board_bot, discards, tx_cards):
    discard_idx = random.randint(0, 2)
    discards.append(tx_cards[discard_idx])
    remaining = [tx_cards[i] for i in range(3) if i != discard_idx]
    
    pairs_50 = [("top", "mid"), ("mid", "top"), ("top", "bot"), ("bot", "top")]
    pairs_other = [("top", "top"), ("mid", "mid"), ("bot", "bot"), ("mid", "bot"), ("bot", "mid")]
    
    caps = {
        "top": 3 - len(board_top),
        "mid": 5 - len(board_mid),
        "bot": 5 - len(board_bot)
    }
    
    def is_valid(p1, p2):
        if p1 == p2:
            return caps[p1] >= 2
        return caps[p1] >= 1 and caps[p2] >= 1

    valid_pairs_50 = [p for p in pairs_50 if is_valid(*p)]
    valid_pairs_other = [p for p in pairs_other if is_valid(*p)]
    
    if random.random() < 0.5 and valid_pairs_50:
        chosen_pair = random.choice(valid_pairs_50)
    elif valid_pairs_other:
        chosen_pair = random.choice(valid_pairs_other)
    elif valid_pairs_50:
        chosen_pair = random.choice(valid_pairs_50)
    else:
        chosen_pair = ("mid", "bot")

    for card, pos in zip(remaining, chosen_pair):
        if pos == "top": board_top.append(card)
        elif pos == "mid": board_mid.append(card)
        else: board_bot.append(card)


def generate_t2_boards(deck):
    """Generate boards after T0+T1+T2 (9 cards each, 2 discards each)."""
    bb_top, bb_mid, bb_bot = [], [], []
    bb_discards = []
    btn_top, btn_mid, btn_bot = [], [], []
    btn_discards = []
    
    # T0: 5 cards each
    apply_t0(bb_top, bb_mid, bb_bot, [deck.pop() for _ in range(5)])
    apply_t0(btn_top, btn_mid, btn_bot, [deck.pop() for _ in range(5)])
    
    # T1: 3 cards each
    apply_tx(bb_top, bb_mid, bb_bot, bb_discards, [deck.pop() for _ in range(3)])
    apply_tx(btn_top, btn_mid, btn_bot, btn_discards, [deck.pop() for _ in range(3)])
    
    # T2: 3 cards each
    apply_tx(bb_top, bb_mid, bb_bot, bb_discards, [deck.pop() for _ in range(3)])
    apply_tx(btn_top, btn_mid, btn_bot, btn_discards, [deck.pop() for _ in range(3)])
    
    return {
        "bb": {"top": bb_top, "middle": bb_mid, "bottom": bb_bot, "discards": bb_discards},
        "btn": {"top": btn_top, "middle": btn_mid, "bottom": btn_bot, "discards": btn_discards}
    }


# ─────────────────────────────────────────────────────────────────────────────
# T4 Oracle NN Leaf Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def load_t4_oracle(model_path, device='cuda'):
    """Load T4 Oracle NN."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", 522)
    hidden = ckpt.get("hidden", 1024)
    n_blocks = ckpt.get("n_blocks", 4)
    model = T4PolicyValueNet(state_dim=state_dim, hidden=hidden, n_blocks=n_blocks).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def evaluate_t4_state_batch(t4_model, states_tensor, masks_tensor, device):
    """
    Evaluate a batch of T4 states using the T4 Oracle NN.
    Returns the best EV for each state (max over valid actions).
    """
    with torch.no_grad():
        logits, values = t4_model(states_tensor, masks_tensor)
        # Use value head as the primary evaluation
        # But also consider: policy-weighted EV could be more accurate
        # For now, use value head directly
        return values


def encode_t4_state(bb_board, btn_board, dealt_cards, bb_discards, is_btn):
    """Encode a T4 state for the T4 Oracle NN."""
    if is_btn:
        self_board = btn_board
        opp_board = bb_board
        discards = []  # BTN discards not tracked separately in this context
    else:
        self_board = bb_board
        opp_board = btn_board
        discards = bb_discards
    
    obs = Observation(
        board_self=self_board,
        board_opponent=opp_board,
        dealt_cards=dealt_cards,
        known_discards_self=discards,
        turn=4,
        is_btn=is_btn,
    )
    return encode_state(obs)


def evaluate_t4_position_bb(t4_model, bb_board, btn_board, remaining_deck, bb_discards, device):
    """
    Evaluate a T4 starting position from BB's perspective.
    BB acts first at T4. We sample a random T4 draw and use the T4 Oracle NN
    to get the value. Over many BTN draws, the noise averages out.
    
    Returns: EV estimate for BB from this position.
    """
    remaining = list(remaining_deck)
    if len(remaining) < 3:
        return 0.0
    
    # Sample one random BB T4 draw
    t4_draw = random.sample(remaining, 3)
    
    state_vec = encode_t4_state(bb_board, btn_board, t4_draw, bb_discards, is_btn=False)
    mask = create_regular_turn_mask(t4_draw, bb_board)
    
    state_t = torch.from_numpy(state_vec).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, value = t4_model(state_t, mask_t)
        # Get best action's EV from logits (masked)
        logits[~mask_t] = float('-inf')
        return value.item()


def evaluate_t4_position_bb_batch(t4_model, bb_board, btn_boards, remaining_decks, bb_discards_list, device, batch_size=1024):
    """
    Batch evaluate multiple T4 positions from BB's perspective.
    Each position has a different BTN board resulting from different BTN T3 actions.
    
    Returns: list of EV estimates.
    """
    all_states = []
    all_masks = []
    valid_indices = []
    
    for i, (btn_board, remaining, bb_disc) in enumerate(zip(btn_boards, remaining_decks, bb_discards_list)):
        remaining_list = list(remaining)
        if len(remaining_list) < 3:
            continue
        t4_draw = random.sample(remaining_list, 3)
        state_vec = encode_t4_state(bb_board, btn_board, t4_draw, bb_disc, is_btn=False)
        mask = create_regular_turn_mask(t4_draw, bb_board)
        all_states.append(state_vec)
        all_masks.append(mask)
        valid_indices.append(i)
    
    if not all_states:
        return [0.0] * len(btn_boards)
    
    results = [0.0] * len(btn_boards)
    
    # Process in batches
    for start in range(0, len(all_states), batch_size):
        end = min(start + batch_size, len(all_states))
        batch_states = torch.from_numpy(np.array(all_states[start:end])).float().to(device)
        batch_masks = torch.from_numpy(np.array(all_masks[start:end])).to(device)
        
        with torch.no_grad():
            logits, values = t4_model(batch_states, batch_masks)
        
        for j, val in enumerate(values.cpu().numpy()):
            results[valid_indices[start + j]] = float(val)
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# T3 Evaluation Core
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_btn_t3(t4_model, bb_board, btn_board, dealt_cards, btn_discards, remaining_deck, device):
    """
    Evaluate all BTN T3 actions.
    BB has already acted at T3 (11 cards). BTN evaluates T3 actions
    by looking at the resulting T4 state via T4 Oracle NN.
    
    Returns: (state_vec_522, action_evs[27], action_mask[27])
    """
    valid_actions = get_turn_actions(dealt_cards, btn_board)
    if not valid_actions:
        return None
    
    # Encode the T3 state for BTN
    obs = Observation(
        board_self=btn_board,
        board_opponent=bb_board,
        dealt_cards=dealt_cards,
        known_discards_self=btn_discards,
        turn=3,
        is_btn=True,
    )
    state_vec = encode_state(obs)
    
    action_evs = np.full(ACTION_DIM, -1e9, dtype=np.float32)
    action_mask = np.zeros(ACTION_DIM, dtype=bool)
    
    # For each BTN T3 action, evaluate the resulting T4 position
    batch_states = []
    batch_masks = []
    action_indices = []
    
    for action in valid_actions:
        idx = get_semantic_action_index(action, dealt_cards)
        action_mask[idx] = True
        
        # Apply action to get T4 board
        btn_board_after = btn_board.copy()
        for card, pos in action.placements:
            if pos == "top": btn_board_after.top.append(card)
            elif pos == "middle": btn_board_after.middle.append(card)
            else: btn_board_after.bottom.append(card)
        
        # Remaining deck after BTN T3 action
        used_cards = set(dealt_cards)
        remaining = [c for c in remaining_deck if c not in used_cards]
        
        if len(remaining) < 3:
            action_evs[idx] = 0.0
            continue
        
        # Sample a random T4 draw for BB (BB acts first at T4)
        t4_draw = random.sample(remaining, 3)
        
        # Encode from BB's T4 perspective (BB sees both boards, makes T4 decision)
        # We evaluate from BB's perspective, BTN's EV = -BB's EV
        t4_state = encode_t4_state(bb_board, btn_board_after, t4_draw, [], is_btn=False)
        t4_mask = create_regular_turn_mask(t4_draw, bb_board)
        
        batch_states.append(t4_state)
        batch_masks.append(t4_mask)
        action_indices.append(idx)
    
    if batch_states:
        states_t = torch.from_numpy(np.array(batch_states)).float().to(device)
        masks_t = torch.from_numpy(np.array(batch_masks)).to(device)
        
        with torch.no_grad():
            logits, values = t4_model(states_t, masks_t)
        
        for j, idx in enumerate(action_indices):
            # BB's value → BTN's EV = -BB's EV
            action_evs[idx] = -values[j].item()
    
    return state_vec, action_evs, action_mask


def evaluate_bb_t3(t4_model, bb_board, btn_board, dealt_cards, bb_discards, btn_discards,
                   remaining_deck, device, btn_samples=500, batch_size=1024):
    """
    Evaluate all BB T3 actions with 2-step minimax.
    
    For each BB T3 action:
      1. BB board → 11 cards
      2. Sample BTN T3 draws from remaining deck
      3. For each BTN draw, find BTN's best T3 response (minimax)
      4. Average over BTN draws = BB T3 action EV
    
    Returns: (state_vec_522, action_evs[27], action_mask[27])
    """
    valid_bb_actions = get_turn_actions(dealt_cards, bb_board)
    if not valid_bb_actions:
        return None
    
    # Encode BB's T3 state
    obs = Observation(
        board_self=bb_board,
        board_opponent=btn_board,
        dealt_cards=dealt_cards,
        known_discards_self=bb_discards,
        turn=3,
        is_btn=False,
    )
    state_vec = encode_state(obs)
    
    action_evs = np.full(ACTION_DIM, -1e9, dtype=np.float32)
    action_mask = np.zeros(ACTION_DIM, dtype=bool)
    
    for bb_action in valid_bb_actions:
        bb_idx = get_semantic_action_index(bb_action, dealt_cards)
        action_mask[bb_idx] = True
        
        # Apply BB T3 action
        bb_board_after = bb_board.copy()
        for card, pos in bb_action.placements:
            if pos == "top": bb_board_after.top.append(card)
            elif pos == "middle": bb_board_after.middle.append(card)
            else: bb_board_after.bottom.append(card)
        
        # Remaining deck after BB T3 (BB drew 3, placed 2, discarded 1)
        bb_used = set(dealt_cards)
        remaining_after_bb = [c for c in remaining_deck if c not in bb_used]
        
        if len(remaining_after_bb) < 3:
            action_evs[bb_idx] = 0.0
            continue
        
        # Sample BTN T3 draws
        all_btn_draws = list(combinations(remaining_after_bb, 3))
        if len(all_btn_draws) > btn_samples:
            btn_draws = random.sample(all_btn_draws, btn_samples)
        else:
            btn_draws = all_btn_draws
        
        # For each BTN draw, evaluate all BTN T3 actions and find best
        bb_ev_per_btn_draw = []
        
        # Collect all (btn_draw, btn_action) pairs for batch evaluation
        all_batch_states = []
        all_batch_masks = []
        draw_action_map = []  # (draw_idx, btn_action_idx)
        
        for draw_idx, btn_draw in enumerate(btn_draws):
            btn_draw_list = list(btn_draw)
            btn_valid_actions = get_turn_actions(btn_draw_list, btn_board)
            
            if not btn_valid_actions:
                continue
            
            for btn_action in btn_valid_actions:
                btn_action_idx = get_semantic_action_index(btn_action, btn_draw_list)
                
                # Apply BTN T3 action
                btn_board_after = btn_board.copy()
                for card, pos in btn_action.placements:
                    if pos == "top": btn_board_after.top.append(card)
                    elif pos == "middle": btn_board_after.middle.append(card)
                    else: btn_board_after.bottom.append(card)
                
                # Remaining after both T3 actions
                btn_used = set(btn_draw_list)
                remaining_after_both = [c for c in remaining_after_bb if c not in btn_used]
                
                if len(remaining_after_both) < 3:
                    continue
                
                # Sample one T4 draw for BB
                t4_draw = random.sample(remaining_after_both, 3)
                
                # Encode from BB's T4 perspective
                t4_state = encode_t4_state(bb_board_after, btn_board_after, t4_draw, bb_discards + [bb_action.discard], is_btn=False)
                t4_mask = create_regular_turn_mask(t4_draw, bb_board_after)
                
                all_batch_states.append(t4_state)
                all_batch_masks.append(t4_mask)
                draw_action_map.append((draw_idx, btn_action_idx))
        
        if not all_batch_states:
            action_evs[bb_idx] = 0.0
            continue
        
        # Batch inference
        all_values = []
        for start in range(0, len(all_batch_states), batch_size):
            end = min(start + batch_size, len(all_batch_states))
            states_t = torch.from_numpy(np.array(all_batch_states[start:end])).float().to(device)
            masks_t = torch.from_numpy(np.array(all_batch_masks[start:end])).to(device)
            
            with torch.no_grad():
                logits, values = t4_model(states_t, masks_t)
            all_values.extend(values.cpu().numpy().tolist())
        
        # Group by BTN draw, find BTN's best response (minimax)
        draw_best_btn_ev = {}  # draw_idx → best BTN EV (= worst for BB)
        for (draw_idx, btn_action_idx), bb_val in zip(draw_action_map, all_values):
            btn_ev = -bb_val  # BTN wants to maximize their own EV
            if draw_idx not in draw_best_btn_ev or btn_ev > draw_best_btn_ev[draw_idx]:
                draw_best_btn_ev[draw_idx] = btn_ev
        
        if draw_best_btn_ev:
            # BB's EV = -average(BTN's best response)
            avg_btn_best = np.mean(list(draw_best_btn_ev.values()))
            action_evs[bb_idx] = -avg_btn_best
        else:
            action_evs[bb_idx] = 0.0
    
    return state_vec, action_evs, action_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_t3_state(t4_model, device, btn_samples=500, is_btn_mode=None):
    """
    Generate a single T3 training sample.
    
    Args:
        t4_model: loaded T4 Oracle NN
        device: torch device
        btn_samples: number of BTN T3 draws to sample (for BB mode)
        is_btn_mode: None=random, True=BTN only, False=BB only
    
    Returns: (state_vec, action_evs, action_mask, is_btn) or None
    """
    deck = get_deck()
    boards = generate_t2_boards(deck)
    
    # Decide who we're generating data for
    if is_btn_mode is None:
        is_btn = random.choice([True, False])
    else:
        is_btn = is_btn_mode
    
    # Draw T3 cards
    # Action order at T3: BB first, then BTN
    
    if is_btn:
        # BTN T3: BB acts first, then BTN
        # First, BB makes T3 decision (heuristic for diversity)
        bb_t3_cards = [deck.pop() for _ in range(3)]
        apply_tx(
            boards["bb"]["top"], boards["bb"]["middle"], boards["bb"]["bottom"],
            boards["bb"]["discards"], bb_t3_cards
        )
        
        # Now BTN gets dealt cards
        btn_t3_cards = [deck.pop() for _ in range(3)]
        
        bb_board = Board(
            top=boards["bb"]["top"],
            middle=boards["bb"]["middle"],
            bottom=boards["bb"]["bottom"]
        )
        btn_board = Board(
            top=boards["btn"]["top"],
            middle=boards["btn"]["middle"],
            bottom=boards["btn"]["bottom"]
        )
        
        result = evaluate_btn_t3(
            t4_model, bb_board, btn_board, btn_t3_cards,
            boards["btn"]["discards"], deck, device
        )
        if result is None:
            return None
        return (*result, True)
    
    else:
        # BB T3: BB acts first (this is what we're evaluating)
        bb_t3_cards = [deck.pop() for _ in range(3)]
        
        bb_board = Board(
            top=boards["bb"]["top"],
            middle=boards["bb"]["middle"],
            bottom=boards["bb"]["bottom"]
        )
        btn_board = Board(
            top=boards["btn"]["top"],
            middle=boards["btn"]["middle"],
            bottom=boards["btn"]["bottom"]
        )
        
        result = evaluate_bb_t3(
            t4_model, bb_board, btn_board, bb_t3_cards,
            boards["bb"]["discards"], boards["btn"]["discards"],
            deck, device, btn_samples=btn_samples
        )
        if result is None:
            return None
        return (*result, False)


def save_chunk(results, output_dir, chunk_name):
    """Save a chunk of results to NPZ."""
    if not results:
        return
    
    states_np = np.array([r[0] for r in results], dtype=np.float32)
    evs_np = np.array([r[1] for r in results], dtype=np.float32)
    masks_np = np.array([r[2] for r in results], dtype=bool)
    is_btn_np = np.array([bool(r[3]) if len(r) > 3 else False for r in results], dtype=bool)

    evs_for_best = evs_np.copy()
    evs_for_best[~masks_np] = -1e9
    actions_np = evs_for_best.argmax(axis=1).astype(np.int32)
    best_evs_np = evs_for_best[np.arange(len(results)), actions_np].astype(np.float32)
    turns_np = np.full(len(results), 3, dtype=np.int16)
    
    # chunk_name is something like 'c0001' or 'w0_c0001'
    if isinstance(chunk_name, int):
        chunk_name = f"c{chunk_name:04d}"
        
    output_file = Path(output_dir) / f"t3_data_{chunk_name}.npz"
    np.savez_compressed(
        output_file,
        states=states_np,
        action_evs=evs_np,
        action_masks=masks_np,
        valid_masks=masks_np,
        actions=actions_np,
        best_evs=best_evs_np,
        rewards=best_evs_np,
        turns=turns_np,
        is_btn=is_btn_np,
    )
    print(f"  Saved chunk {chunk_name} to {output_file.name} ({len(results)} states)", flush=True)


def worker_process(worker_id, args, num_states):
    """Worker process function for multiprocessing."""
    # Seed properly for multiprocessing
    np.random.seed()
    random.seed()
    torch.manual_seed(random.randint(0, 1000000))
    
    # Avoid thread contention when running multiple processes on CPU
    if args.device == "cpu":
        torch.set_num_threads(1)
        
    model_path = Path(AI_DIR) / args.t4_model
    t4_model = load_t4_oracle(str(model_path), device=args.device)
    
    is_btn_mode = None
    if args.mode == "btn":
        is_btn_mode = True
    elif args.mode == "bb":
        is_btn_mode = False
        
    output_dir = Path(AI_DIR) / args.output_dir
    results = []
    chunk_id = 0
    generated = 0
    failed = 0
    
    t_start = time.time()
    
    while generated < num_states:
        result = generate_single_t3_state(
            t4_model, args.device,
            btn_samples=args.btn_samples,
            is_btn_mode=is_btn_mode
        )
        
        if result is None:
            failed += 1
            continue
            
        results.append(result)
        generated += 1
        
        if generated % args.save_interval == 0:
            chunk_name = f"w{worker_id}_{chunk_id:04d}"
            save_chunk(results, output_dir, chunk_name)
            results = []
            chunk_id += 1
            
            elapsed = time.time() - t_start
            rate = generated / elapsed
            print(f"  [Worker {worker_id}] {generated}/{num_states} ({rate:.1f} states/sec)", flush=True)
            
    if results:
        chunk_name = f"w{worker_id}_{chunk_id:04d}"
        save_chunk(results, output_dir, chunk_name)
        
    return generated, failed


def main():
    parser = argparse.ArgumentParser(description="T3 Teacher Data Generator")
    parser.add_argument("--states", type=int, default=10000, help="Number of T3 states to generate")
    parser.add_argument("--btn-samples", type=int, default=500, help="BTN T3 draws to sample (for BB mode)")
    parser.add_argument("--save-interval", type=int, default=500, help="Save every N states")
    parser.add_argument("--output-dir", type=str, default="data/t3_dataset_bi")
    parser.add_argument("--t4-model", type=str, default="data/t4_oracle_v2/t4_policyvalue_best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", type=str, choices=["both", "btn", "bb"], default="both",
                        help="Generate BTN-only, BB-only, or both")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()
    
    output_dir = Path(AI_DIR) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  T3 Teacher Data Generator (Backward Induction)")
    print("=" * 60)
    print(f"  Device: {args.device}")
    print(f"  T4 Model: {args.t4_model}")
    print(f"  States: {args.states:,}")
    print(f"  BTN Samples: {args.btn_samples}")
    print(f"  Mode: {args.mode}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    print()
    
    t_start = time.time()
    
    if args.workers > 1:
        import multiprocessing as mp
        print(f"  Starting {args.workers} workers...")
        
        # Calculate states per worker
        base_states = args.states // args.workers
        extra_states = args.states % args.workers
        worker_states = [base_states + (1 if i < extra_states else 0) for i in range(args.workers)]
        
        # Use spawn method for CUDA/torch compatibility if needed, but for CPU it's safer anyway
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass # already set
            
        with mp.Pool(processes=args.workers) as pool:
            # Starmap to pass multiple arguments
            worker_args = [(i, args, worker_states[i]) for i in range(args.workers)]
            results = pool.starmap(worker_process, worker_args)
            
        total_generated = sum(r[0] for r in results)
        total_failed = sum(r[1] for r in results)
    else:
        # Single process
        print(f"  Loading T4 Oracle from {args.t4_model}...")
        t4_model = load_t4_oracle(str(Path(AI_DIR) / args.t4_model), device=args.device)
        print(f"  T4 Oracle loaded successfully.\n")
        
        is_btn_mode = None
        if args.mode == "btn":
            is_btn_mode = True
        elif args.mode == "bb":
            is_btn_mode = False
        
        results = []
        chunk_id = 0
        total_generated = 0
        total_failed = 0
        
        while total_generated < args.states:
            result = generate_single_t3_state(
                t4_model, args.device,
                btn_samples=args.btn_samples,
                is_btn_mode=is_btn_mode
            )
            
            if result is None:
                total_failed += 1
                continue
            
            results.append(result)
            total_generated += 1
            
            if total_generated % args.save_interval == 0:
                save_chunk(results, output_dir, chunk_id)
                results = []
                chunk_id += 1
                
                elapsed = time.time() - t_start
                rate = total_generated / elapsed
                eta = (args.states - total_generated) / rate if rate > 0 else 0
                print(f"  [{total_generated}/{args.states}] {rate:.1f} states/sec, "
                      f"ETA: {eta/60:.1f}min, failed: {total_failed}", flush=True)
        
        if results:
            save_chunk(results, output_dir, chunk_id)
            
    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  Generation Complete")
    print(f"  Total Generated: {total_generated}")
    print(f"  Failed: {total_failed}")
    print(f"  Total Time: {elapsed:.1f}s ({total_generated/elapsed:.1f} states/sec)")
    print(f"  All chunks saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
