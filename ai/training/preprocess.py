"""
OFC Pineapple - Data Preprocessing Pipeline

Converts game logs (JSONL) into numpy arrays for training:
  - states.npy       (N, 490)
  - actions.npy      (N,)
  - valid_masks.npy  (N, MAX_ACTIONS)
  - royalties.npy    (N,)
  - busted.npy       (N,)
  - fl_entry.npy     (N,)
"""
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state
from ai.engine.action_space import (
    Action, get_initial_actions, get_turn_actions,
    create_action_mask, encode_action, MAX_ACTIONS,
)


def build_observation(turn_log: dict) -> Observation:
    """Build Observation from a turn log entry."""
    return Observation(
        board_self=Board.from_dict(turn_log["board_self"]),
        board_opponent=Board.from_dict(turn_log.get("board_opponent", {"top":[],"middle":[],"bottom":[]})),
        dealt_cards=turn_log["dealt_cards"],
        known_discards_self=turn_log.get("discards_self", []),
        turn=turn_log["turn"],
        is_btn=turn_log.get("is_btn", False),
        chips_self=turn_log.get("chips_self", 200),
        chips_opponent=turn_log.get("chips_opponent", 200),
    )


def parse_action(action_data: dict) -> Action:
    """Parse action from log format."""
    placements = []
    for p in action_data["placements"]:
        if isinstance(p, (list, tuple)):
            placements.append((p[0], p[1]))
        elif isinstance(p, dict):
            placements.append((p["card"], p["position"]))
    return Action(placements=placements, discard=action_data.get("discard"))


def preprocess_turn(turn_log: dict, hand_result: dict) -> Optional[dict]:
    """
    Convert one turn log entry into a training sample.

    Returns None if the action cannot be encoded (edge case).
    """
    obs = build_observation(turn_log)
    state = encode_state(obs)

    # Get valid actions for this position
    if turn_log["turn"] == 0:
        valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
    else:
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

    # Encode the player's chosen action
    player_action = parse_action(turn_log["action"])
    try:
        action_idx = encode_action(player_action, valid_actions)
    except ValueError:
        return None  # Skip if action not found

    mask = create_action_mask(valid_actions)

    # Labels from hand result
    player = turn_log["player"]
    royalty = hand_result["royalties"][player]["total"]
    busted = float(hand_result["busted"][player])
    fl_entry = float(hand_result["fl_entry"][player])

    return {
        "state": state,
        "action_idx": action_idx,
        "valid_mask": mask,
        "royalty": royalty,
        "busted": busted,
        "fl_entry": fl_entry,
    }


def preprocess_jsonl(input_path: str, output_dir: str):
    """
    Process a JSONL file with turn logs + hand results.

    Expected JSONL format per line:
    {
        "turn_log": { ... },
        "hand_result": { ... }
    }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = []
    actions = []
    masks = []
    royalties = []
    busted_arr = []
    fl_arr = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = preprocess_turn(data["turn_log"], data["hand_result"])
            if result is None:
                skipped += 1
                continue

            states.append(result["state"])
            actions.append(result["action_idx"])
            masks.append(result["valid_mask"])
            royalties.append(result["royalty"])
            busted_arr.append(result["busted"])
            fl_arr.append(result["fl_entry"])

    if not states:
        print("No valid samples found!")
        return

    # Save as numpy
    np.save(output_dir / "states.npy", np.array(states, dtype=np.float32))
    np.save(output_dir / "actions.npy", np.array(actions, dtype=np.int64))
    np.save(output_dir / "valid_masks.npy", np.array(masks, dtype=bool))
    np.save(output_dir / "royalties.npy", np.array(royalties, dtype=np.float32))
    np.save(output_dir / "busted.npy", np.array(busted_arr, dtype=np.float32))
    np.save(output_dir / "fl_entry.npy", np.array(fl_arr, dtype=np.float32))

    # Metadata
    metadata = {
        "total_samples": len(states),
        "skipped": skipped,
        "state_dim": states[0].shape[0],
        "max_actions": MAX_ACTIONS,
        "busted_ratio": float(np.mean(busted_arr)),
        "fl_ratio": float(np.mean(fl_arr)),
        "avg_royalty": float(np.mean(royalties)),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessed {len(states)} samples ({skipped} skipped)")
    print(f"  Bust ratio: {metadata['busted_ratio']:.2%}")
    print(f"  FL ratio:   {metadata['fl_ratio']:.2%}")
    print(f"  Avg royalty: {metadata['avg_royalty']:.1f}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess game logs for training")
    parser.add_argument("input", help="Path to JSONL input file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    args = parser.parse_args()
    preprocess_jsonl(args.input, args.output)
