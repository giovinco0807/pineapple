"""
Convert T3 bottom-up JSONL data to oracle training tensors.

Regular-turn actions are encoded into fixed semantic slots:
discard index (0..2) * 9 + first remaining card row (0..2) * 3 +
second remaining card row (0..2). This preserves the discard choice and
prevents board-dependent list-index aliasing.
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
ROW_MAP = {"T": "top", "M": "middle", "B": "bottom"}
ARROW = "\u2192"


def parse_turn_action(desc: str):
    """Parse regular-turn action desc like 'd:Qs 6h->T 6c->B'."""
    parts = desc.split()
    if not parts or not parts[0].startswith("d:"):
        raise ValueError(f"Bad action desc: {desc!r}")

    discard = parts[0][2:]
    placements = []
    for part in parts[1:]:
        if "->" in part:
            card, row_code = part.split("->", 1)
        elif ARROW in part:
            card, row_code = part.split(ARROW, 1)
        else:
            raise ValueError(f"Bad placement token {part!r} in {desc!r}")
        if row_code not in ROW_MAP:
            raise ValueError(f"Bad row code {row_code!r} in {desc!r}")
        placements.append((card, ROW_MAP[row_code]))

    if len(placements) != 2:
        raise ValueError(f"Expected exactly 2 placements in {desc!r}")
    return discard, placements


def main():
    parser = argparse.ArgumentParser(description="Convert T3 JSONL to fixed 27-slot tensors")
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
            result = json.loads(line)
            n_patterns += 1

            for cr in result["t3_records"]:
                board = Board(
                    top=list(cr["top"]),
                    middle=list(cr["mid"]),
                    bottom=list(cr["bot"]),
                )
                deal = list(cr["deal"])

                obs = Observation(
                    board_self=board,
                    board_opponent=Board(),
                    dealt_cards=deal,
                    known_discards_self=[],
                    turn=3,
                    is_btn=True,
                )
                state = encode_state(obs)

                action_evs = np.full(MAX_ACTIONS, -1e9, dtype=np.float32)
                valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)

                for ar in cr["actions"]:
                    discard, placements = parse_turn_action(ar["desc"])
                    action = Action(placements=placements, discard=discard)
                    idx = get_semantic_action_index(action, deal)
                    action_evs[idx] = max(action_evs[idx], float(ar["ev"]))
                    valid_mask[idx] = True
                    n_actions_seen += 1

                if not valid_mask.any():
                    continue

                best_idx = int(np.argmax(action_evs))
                records.append({
                    "state": state,
                    "action": best_idx,
                    "ev": float(action_evs[best_idx]),
                    "turn": 3,
                    "valid_mask": valid_mask,
                    "action_evs": action_evs,
                })

    n_records = len(records)
    print(f"Patterns: {n_patterns}, T3 records: {n_records}, actions: {n_actions_seen}")
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
