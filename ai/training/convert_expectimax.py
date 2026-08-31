"""
Expectimax JSON → BC/VN Training Data Converter

Reads Expectimax result JSON files and produces numpy arrays
compatible with the existing behavior_cloning.py pipeline.

Output:
  states.npy      (N, 520)  encoded board states
  actions.npy     (N,)      best action index (hard label)
  valid_masks.npy (N, 250)  valid action mask
  evs.npy         (N,)      best action EV (for VN training)
  turns.npy       (N,)      turn number (0-4)
  royalties.npy   (N,)      zeros (not available from Expectimax)
  busted.npy      (N,)      zeros
  fl_entry.npy    (N,)      zeros
  rewards.npy     (N,)      best EV (alias for weighted BC)
  action_evs.npy  (N, 250)  per-action EVs (for soft-label training)
  metadata.json

Usage:
    python -m ai.training.convert_expectimax results_gcp/ --output data/expectimax_train
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM
from ai.engine.action_space import Action, get_semantic_action_index

MAX_ACTIONS = 250
ROW_MAP = {"T": "top", "M": "middle", "B": "bottom"}
ARROW = "\u2192"  # → (Unicode arrow, ASCII-safe source)


def parse_t0_action(desc: str):
    """Parse T0 action desc like 'Ah->M Kh->T ...'.

    Returns list of (card, position) tuples.
    """
    placements = []
    for part in desc.split():
        card, row_code = part.split(ARROW)
        placements.append((card, ROW_MAP[row_code]))
    return placements


def parse_turn_action(desc: str):
    """Parse T1-T4 action desc like 'd:Qs 6h->B 6c->B'.

    Returns (discard, [(card, position), (card, position)]).
    """
    parts = desc.split()
    discard = parts[0][2:]  # "d:Qs" -> "Qs"
    placements = []
    for part in parts[1:]:
        card, row_code = part.split(ARROW)
        placements.append((card, ROW_MAP[row_code]))
    return discard, placements


def t0_action_to_index(placements, dealt_cards):
    """Map T0 placements to action index via Python enumeration.

    Uses get_initial_actions for deterministic ordering (same as production).
    """
    from ai.engine.action_space import get_initial_actions

    # Normalize target: sorted cards per position
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card, pos in placements:
        by_pos[pos].append(card)
    target = (
        tuple(sorted(by_pos["top"])),
        tuple(sorted(by_pos["middle"])),
        tuple(sorted(by_pos["bottom"])),
    )

    all_actions = get_initial_actions(dealt_cards, Board())
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
    return 0  # fallback


def turn_action_to_index(discard, placements, dealt_cards):
    """Map T1-T4 action to fixed 27-slot semantic index."""
    return get_semantic_action_index(
        Action(placements=list(placements), discard=discard),
        dealt_cards,
    )


def process_t0(result: dict, records: list):
    """Process T0 data from one Expectimax result.

    Creates one sample with the best T0 action.
    Also stores all 232 action EVs for soft-label training.
    """
    t0_hand = result["t0_hand"]
    best_desc = result["all_actions"][0]["action_desc"]  # sorted by EV desc
    best_ev = result["all_actions"][0]["ev"]

    # Build Observation for T0
    obs = Observation(
        board_self=Board(),
        board_opponent=Board(),
        dealt_cards=t0_hand,
        known_discards_self=[],
        turn=0,
        is_btn=True,
    )

    # Encode state
    state = encode_state(obs)

    # Parse best action and get index
    placements = parse_t0_action(best_desc)
    action_idx = t0_action_to_index(placements, t0_hand)

    # All action EVs (for soft labels)
    action_evs = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
    valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    for ar in result["all_actions"]:
        pl = parse_t0_action(ar["action_desc"])
        idx = t0_action_to_index(pl, t0_hand)
        action_evs[idx] = ar["ev"]
        valid_mask[idx] = True

    records.append({
        "state": state,
        "action": action_idx,
        "ev": best_ev,
        "turn": 0,
        "valid_mask": valid_mask,
        "action_evs": action_evs,
    })


def process_choice_records(result: dict, records: list):
    """Process T1-T4 choice records from one Expectimax result."""
    for cr in result["choice_records"]:
        turn = cr["turn"]
        deal = cr["deal"]

        # Build board state
        board = Board(
            top=list(cr["top"]),
            middle=list(cr["mid"]),
            bottom=list(cr["bot"]),
        )

        # Build Observation
        obs = Observation(
            board_self=board,
            board_opponent=Board(),
            dealt_cards=deal,
            known_discards_self=[],
            turn=turn,
            is_btn=True,
        )

        state = encode_state(obs)

        # Parse best action (first in sorted list)
        best = cr["actions"][0]
        best_ev = best["ev"]
        best_discard, best_placements = parse_turn_action(best["desc"])
        action_idx = turn_action_to_index(best_discard, best_placements, deal)

        # All action EVs
        action_evs = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
        valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for ar in cr["actions"]:
            discard, pl = parse_turn_action(ar["desc"])
            idx = turn_action_to_index(discard, pl, deal)
            action_evs[idx] = ar["ev"]
            valid_mask[idx] = True

        records.append({
            "state": state,
            "action": action_idx,
            "ev": best_ev,
            "turn": turn,
            "valid_mask": valid_mask,
            "action_evs": action_evs,
        })


def convert_expectimax(input_dir: str, output_dir: str, t0_only: bool = False):
    """Convert all Expectimax JSON files to numpy training data."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {input_path}")

    records = []
    n_files = 0
    n_skipped = 0
    start = time.time()

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                result = json.load(f)
        except (json.JSONDecodeError, KeyError):
            n_skipped += 1
            continue

        if "all_actions" not in result or not result["all_actions"]:
            n_skipped += 1
            continue

        # T0 sample
        process_t0(result, records)

        # T1-T4 samples
        if not t0_only and "choice_records" in result:
            process_choice_records(result, records)

        n_files += 1
        if n_files % 50 == 0:
            elapsed = time.time() - start
            print(f"  {n_files}/{len(json_files)} files, "
                  f"{len(records):,} samples, {elapsed:.1f}s")

    if not records:
        print("No records found!")
        return

    elapsed = time.time() - start
    print(f"\nProcessed {n_files} files ({n_skipped} skipped) "
          f"→ {len(records):,} samples in {elapsed:.1f}s")

    # Convert to numpy arrays (float16 for large arrays to save disk)
    N = len(records)
    states = np.zeros((N, STATE_DIM), dtype=np.float16)
    actions = np.zeros(N, dtype=np.int64)
    evs = np.zeros(N, dtype=np.float32)
    turns = np.zeros(N, dtype=np.int64)
    valid_masks = np.zeros((N, MAX_ACTIONS), dtype=bool)
    action_evs = np.full((N, MAX_ACTIONS), -100.0, dtype=np.float16)

    for i, r in enumerate(records):
        states[i] = r["state"]
        actions[i] = r["action"]
        evs[i] = r["ev"]
        turns[i] = r["turn"]
        valid_masks[i] = r["valid_mask"]
        action_evs[i] = r["action_evs"]

    # Save arrays
    np.save(output_path / "states.npy", states)
    np.save(output_path / "actions.npy", actions)
    np.save(output_path / "evs.npy", evs)
    np.save(output_path / "turns.npy", turns)
    np.save(output_path / "valid_masks.npy", valid_masks)
    np.save(output_path / "action_evs.npy", action_evs)

    # BC-compatible dummy arrays (no royalty/bust/fl data from Expectimax)
    np.save(output_path / "royalties.npy", np.zeros(N, dtype=np.float32))
    np.save(output_path / "busted.npy", np.zeros(N, dtype=np.float32))
    np.save(output_path / "fl_entry.npy", np.zeros(N, dtype=np.float32))
    np.save(output_path / "rewards.npy", evs.copy())  # EV as reward weight

    # VN-compatible NPZ (for train_value.py) - use savez_compressed to save disk
    np.savez_compressed(
        output_path / "value_train.npz",
        obs=states,
        score=evs,
        turn=turns,
        busted=np.zeros(N, dtype=np.float32),
        fl_entry=np.zeros(N, dtype=np.float32),
    )
    print(f"  VN NPZ: {output_path / 'value_train.npz'}")

    # Stats
    turn_counts = Counter(turns)
    ev_by_turn = {}
    for t in sorted(turn_counts):
        mask = turns == t
        ev_by_turn[int(t)] = float(np.mean(evs[mask]))

    metadata = {
        "total_samples": N,
        "n_files": n_files,
        "n_skipped": n_skipped,
        "state_dim": STATE_DIM,
        "max_actions": MAX_ACTIONS,
        "turn_counts": {int(k): int(v) for k, v in sorted(turn_counts.items())},
        "avg_ev_by_turn": ev_by_turn,
        "avg_ev": float(np.mean(evs)),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_path}/")
    print(f"  Samples: {N:,}")
    for t in sorted(turn_counts):
        print(f"    T{t}: {turn_counts[t]:,} (avg EV={ev_by_turn[int(t)]:.1f})")
    print(f"  Avg EV: {np.mean(evs):.2f}")

    total_time = time.time() - start
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert Expectimax JSON results to BC/VN training data")
    parser.add_argument("input", help="Directory with Expectimax .json files")
    parser.add_argument("--output", default="data/expectimax_train",
                        help="Output directory for numpy files")
    parser.add_argument("--t0-only", action="store_true",
                        help="Only convert T0 actions (skip T1-T4 choice records)")
    args = parser.parse_args()
    convert_expectimax(args.input, args.output, t0_only=args.t0_only)
