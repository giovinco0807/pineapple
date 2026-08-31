"""
OFC Pineapple - Ranking-based Preprocessor for T0 PolicyNet

Converts all-action JSONL (from convert_t0_allactions.py) into numpy arrays
suitable for ranking/regression training.

Key difference from preprocess_fast.py:
  - Groups records by hand (same dealt_cards = same state)
  - For each hand, creates one state + array of (action_idx, ev) pairs
  - Stores action_evs: (N, MAX_ACTIONS) matrix with EV for each valid action
  - This enables ListNet/ListMLE ranking loss or direct EV regression

Output files:
  states.npy:      (N_hands, STATE_DIM) float32
  action_evs.npy:  (N_hands, MAX_ACTIONS) float32  [EV for each action, -inf for invalid]
  valid_masks.npy: (N_hands, MAX_ACTIONS) bool
  best_actions.npy:(N_hands,) int64  [action index with highest EV]
  metadata.json:   stats

Usage:
    python ai/training/preprocess_ranking.py ai/data/t0_allactions.jsonl --output ai/data/ranked_t0
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS


def rust_to_python_card(card: str) -> str:
    """Convert Rust solver card notation to Python engine notation."""
    if card in ("JK", "Jo"):
        return "X1"
    return card


def rust_to_python_cards(cards: list) -> list:
    return [rust_to_python_card(c) for c in cards]


def action_to_index_t0(placements: list, dealt_cards: list, board=None) -> int:
    """Convert T0 placement action to deterministic index."""
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card, pos in placements:
        by_pos[pos].append(rust_to_python_card(card))
    target = (
        tuple(sorted(by_pos["top"])),
        tuple(sorted(by_pos["middle"])),
        tuple(sorted(by_pos["bottom"])),
    )

    b = Board.from_dict(board) if board else Board()
    all_actions = get_initial_actions(dealt_cards, b)
    for idx, a in enumerate(all_actions):
        a_pos = {"top": [], "middle": [], "bottom": []}
        for card, pos in a.placements:
            a_pos[pos].append(card)
        a_key = (
            tuple(sorted(a_pos["top"])),
            tuple(sorted(a_pos["middle"])),
            tuple(sorted(a_pos["bottom"])),
        )
        if a_key == target:
            return min(idx, MAX_ACTIONS - 1)
    return -1  # not found


def preprocess_ranking(input_path: str, output_dir: str):
    """Group records by hand and build ranking training data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Phase 1: Group records by dealt_cards (= same state)
    print("Phase 1: Reading and grouping records...")
    hand_groups = defaultdict(list)  # key: tuple of dealt_cards -> list of (placements, ev)

    n_lines = 0
    n_skipped = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                data = json.loads(line.strip())
                tl = data["turn_log"]
                dealt_cards = tuple(rust_to_python_cards(tl["dealt_cards"]))
                placements = tl["action"]["placements"]
                ev = data.get("cfr_ev", 0.0)
                # Convert card notation in placements too
                placements_py = [[rust_to_python_card(c), p] for c, p in placements]
                hand_groups[dealt_cards].append((placements_py, ev))
            except Exception:
                n_skipped += 1

    n_hands = len(hand_groups)
    print(f"  {n_lines:,} lines -> {n_hands:,} unique hands ({n_skipped} skipped)")

    # Phase 2: Encode states and build EV matrices
    print("Phase 2: Encoding states and building EV matrices...")

    states_list = []
    action_evs_list = []
    valid_masks_list = []
    best_actions_list = []
    n_actions_list = []

    processed = 0
    for dealt_cards, records in hand_groups.items():
        dealt_list = list(dealt_cards)

        # Encode state (same for all records of this hand)
        obs = Observation(
            board_self=Board(),
            board_opponent=Board(),
            dealt_cards=dealt_list,
            known_discards_self=[],
            turn=0,
            is_btn=True,
        )
        state = encode_state(obs)

        # Get all valid actions
        all_actions = get_initial_actions(dealt_list, Board())
        n_valid = min(len(all_actions), MAX_ACTIONS)

        # Build action index -> EV mapping
        action_evs = np.full(MAX_ACTIONS, -1e9, dtype=np.float32)
        valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
        valid_mask[:n_valid] = True

        matched = 0
        for placements, ev in records:
            idx = action_to_index_t0(placements, dealt_list)
            if 0 <= idx < MAX_ACTIONS:
                action_evs[idx] = ev
                matched += 1

        # Find best action (highest EV)
        best_idx = int(np.argmax(action_evs))

        states_list.append(state)
        action_evs_list.append(action_evs)
        valid_masks_list.append(valid_mask)
        best_actions_list.append(best_idx)
        n_actions_list.append(n_valid)

        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  {processed:,}/{n_hands:,} hands ({elapsed:.1f}s)")

    # Phase 3: Save
    print("Phase 3: Saving numpy files...")

    states = np.array(states_list, dtype=np.float32)
    action_evs = np.array(action_evs_list, dtype=np.float32)
    valid_masks = np.array(valid_masks_list, dtype=bool)
    best_actions = np.array(best_actions_list, dtype=np.int64)

    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "action_evs.npy", action_evs)
    np.save(output_dir / "valid_masks.npy", valid_masks)
    np.save(output_dir / "actions.npy", best_actions)  # compat with existing BC

    # Also save dummy arrays for compatibility with OFCDataset
    np.save(output_dir / "royalties.npy", np.zeros(n_hands, dtype=np.float32))
    np.save(output_dir / "busted.npy", np.zeros(n_hands, dtype=np.float32))
    np.save(output_dir / "fl_entry.npy", np.zeros(n_hands, dtype=np.float32))
    np.save(output_dir / "rewards.npy", np.zeros(n_hands, dtype=np.float32))

    # Compute stats
    ev_per_hand = []
    coverage = []
    for i in range(n_hands):
        valid_evs = action_evs[i][valid_masks[i]]
        real_evs = valid_evs[valid_evs > -1e8]
        if len(real_evs) > 0:
            ev_per_hand.append(float(real_evs.max()))
            coverage.append(len(real_evs) / int(valid_masks[i].sum()))

    metadata = {
        "total_hands": n_hands,
        "total_records": n_lines,
        "skipped": n_skipped,
        "state_dim": STATE_DIM,
        "max_actions": MAX_ACTIONS,
        "avg_valid_actions": float(np.mean(n_actions_list)),
        "avg_ev_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "avg_best_ev": float(np.mean(ev_per_hand)) if ev_per_hand else 0.0,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone! {n_hands:,} hands -> {output_dir}")
    print(f"  State dim:          {STATE_DIM}")
    print(f"  Avg valid actions:  {np.mean(n_actions_list):.0f}")
    print(f"  Avg EV coverage:    {np.mean(coverage)*100:.1f}%")
    print(f"  Avg best EV:        {np.mean(ev_per_hand):.2f}")
    print(f"  Total time:         {total_time:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="ai/data/ranked_t0")
    args = parser.parse_args()
    preprocess_ranking(args.input, args.output)
