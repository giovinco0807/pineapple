"""
Convert T4 exact solutions JSONL data to oracle training tensors.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action, REGULAR_TURN_ACTIONS, get_semantic_action_index
from ai.engine.encoding import Board, Observation, encode_state

MAX_ACTIONS = REGULAR_TURN_ACTIONS
ROW_MAP = {"top": "top", "mid": "middle", "bot": "bottom"}

def main():
    parser = argparse.ArgumentParser(description="Convert T4 JSONL to fixed 27-slot tensors")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    n_patterns = 0
    n_actions_seen = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cr = json.loads(line)
            n_patterns += 1

            board = Board(
                top=list(cr["bb_top"]),
                middle=list(cr["bb_middle"]),
                bottom=list(cr["bb_bottom"]),
            )
            deal = list(cr["bb_drawn"])

            obs = Observation(
                board_self=board,
                board_opponent=Board(),
                dealt_cards=deal,
                known_discards_self=[],
                turn=8,
                is_btn=False,
            )
            state = encode_state(obs)

            action_evs = np.full(MAX_ACTIONS, -1e9, dtype=np.float32)
            valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)

            for placement in cr["placements"]:
                cards_placed = placement["cards_placed"]
                slots = placement["slots"]
                
                # Determine discard
                discard = None
                for c in deal:
                    if c not in cards_placed:
                        discard = c
                        break
                
                # Sometimes deal has duplicate cards? E.g., multiple Jokers, but here it's 52-card deck without Jokers?
                # Actually, OFC Pineapple uses standard deck without Jokers in most standard setups, unless specified.
                # If there are duplicate Jokers like X1, X2, we might need a more robust check. But exact solver usually has distinct cards.
                
                placements_list = []
                for card, slot in zip(cards_placed, slots):
                    placements_list.append((card, ROW_MAP[slot]))
                    
                action = Action(placements=placements_list, discard=discard)
                try:
                    idx = get_semantic_action_index(action, deal)
                    action_evs[idx] = max(action_evs[idx], float(placement["ev"]))
                    valid_mask[idx] = True
                    n_actions_seen += 1
                except ValueError:
                    print(f"Error parsing action: {action} with deal {deal}")
                    continue

            if not valid_mask.any():
                continue

            best_idx = int(np.argmax(action_evs))
            records.append({
                "state": state,
                "action": best_idx,
                "ev": float(action_evs[best_idx]),
                "turn": 8,
                "valid_mask": valid_mask,
                "action_evs": action_evs,
            })

    n_records = len(records)
    print(f"Patterns: {n_patterns}, T4 records: {n_records}, actions: {n_actions_seen}")
    if n_records == 0:
        print("No records!")
        return

    states = np.array([r["state"] for r in records], dtype=np.float16)
    actions = np.array([r["action"] for r in records], dtype=np.int64)
    valid_masks = np.array([r["valid_mask"] for r in records], dtype=bool)
    action_evs = np.array([r["action_evs"] for r in records], dtype=np.float16)
    rewards = np.array([r["ev"] for r in records], dtype=np.float32)
    busted = np.zeros(n_records, dtype=np.float32)
    fl_entry = np.zeros(n_records, dtype=np.float32)
    royalties = np.zeros(n_records, dtype=np.float32)

    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "valid_masks.npy", valid_masks)
    np.save(output_dir / "action_evs.npy", action_evs)
    np.save(output_dir / "rewards.npy", rewards)
    np.save(output_dir / "busted.npy", busted)
    np.save(output_dir / "fl_entry.npy", fl_entry)
    np.save(output_dir / "royalties.npy", royalties)

    print(f"\nSaved to {output_dir}:")
    print(f"  states:       {states.shape} {states.dtype}")
    print(f"  actions:      {actions.shape}")
    print(f"  action_evs:   {action_evs.shape}")
    print(f"  EV range:     [{rewards.min():.1f}, {rewards.max():.1f}], mean={rewards.mean():.2f}")
    print(f"  Valid mask:   avg {valid_masks.sum(axis=1).mean():.1f} valid actions")

if __name__ == "__main__":
    main()
