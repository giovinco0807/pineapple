"""
OFC Pineapple - Scalable Preprocessor (memmap-based)

Handles millions of samples without OOM by using numpy memmap.

Usage:
    python ai/training/preprocess_fast.py data/train_1m.jsonl --output data/processed_1m
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state

MAX_ACTIONS = 250


def action_to_index(action: dict, turn: int) -> int:
    """Convert action to deterministic index (0..MAX_ACTIONS-1).

    Turn 0 (5-card placement): partition-based index.
      5 cards into (top, mid, bot) with constraints top<=3, mid<=5, bot<=5.
      Each partition (t,m,b) gets a contiguous block. Within a block,
      positions are assigned by lexicographic card order.
    Turn 1-8 (2 placements + 1 discard): pos0*3 + pos1 (0-8).
    """
    placements = action["placements"]

    if turn == 0:
        by_pos = {"top": [], "middle": [], "bottom": []}
        for p in placements:
            by_pos[p[1]].append(p[0])
        t, m, b = len(by_pos["top"]), len(by_pos["middle"]), len(by_pos["bottom"])

        # Encode partition as a unique small integer
        # Valid partitions of 5 into (t,m,b) with t<=3, m<=5, b<=5:
        # We enumerate all valid (t,m,b) tuples and assign sequential IDs
        partition_id = _partition_id(t, m, b)

        # Within partition: sort cards in each row, then hash deterministically
        # to a sub-index within a fixed bucket size
        cards_key = (
            tuple(sorted(by_pos["top"])),
            tuple(sorted(by_pos["middle"])),
            tuple(sorted(by_pos["bottom"])),
        )
        # Use a stable hash (sum of ord values) to distribute within partition bucket
        bucket_size = 10  # ~250 / ~24 partitions ≈ 10 slots each
        sub_idx = _stable_hash(cards_key) % bucket_size
        return min(partition_id * bucket_size + sub_idx, MAX_ACTIONS - 1)
    else:
        positions = ["top", "middle", "bottom"]
        pos0 = placements[0][1] if len(placements) > 0 else "top"
        pos1 = placements[1][1] if len(placements) > 1 else "top"
        pos0_idx = positions.index(pos0) if pos0 in positions else 0
        pos1_idx = positions.index(pos1) if pos1 in positions else 0
        return pos0_idx * 3 + pos1_idx


# Pre-compute partition lookup: (t,m,b) -> sequential ID
_PARTITION_MAP = {}
_pid = 0
for _t in range(4):       # top: 0-3
    for _m in range(6):    # mid: 0-5
        _b = 5 - _t - _m
        if 0 <= _b <= 5:
            _PARTITION_MAP[(_t, _m, _b)] = _pid
            _pid += 1


def _partition_id(t: int, m: int, b: int) -> int:
    return _PARTITION_MAP.get((t, m, b), 0)


def _stable_hash(cards_key: tuple) -> int:
    """Deterministic hash based on card characters (no randomization)."""
    h = 0
    for row in cards_key:
        for card in row:
            for ch in card:
                h = h * 31 + ord(ch)
    return abs(h)


def count_lines(path: str) -> int:
    """Fast line count."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def preprocess_fast(input_path: str, output_dir: str):
    """Scalable preprocessing with memmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count lines first
    print("Counting lines...")
    n_lines = count_lines(input_path)
    print(f"  {n_lines:,} lines found")

    # Create memmap files
    states_mm = np.memmap(output_dir / "states.npy.tmp", dtype=np.float32,
                          mode='w+', shape=(n_lines, 490))
    actions_mm = np.memmap(output_dir / "actions.npy.tmp", dtype=np.int64,
                           mode='w+', shape=(n_lines,))
    royalties_mm = np.memmap(output_dir / "royalties.npy.tmp", dtype=np.float32,
                             mode='w+', shape=(n_lines,))
    busted_mm = np.memmap(output_dir / "busted.npy.tmp", dtype=np.float32,
                          mode='w+', shape=(n_lines,))
    fl_mm = np.memmap(output_dir / "fl_entry.npy.tmp", dtype=np.float32,
                      mode='w+', shape=(n_lines,))

    start_time = time.time()
    total = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                tl = data["turn_log"]
                hr = data["hand_result"]

                obs = Observation(
                    board_self=Board.from_dict(tl["board_self"]),
                    board_opponent=Board.from_dict(tl.get("board_opponent",
                        {"top":[],"middle":[],"bottom":[]})),
                    dealt_cards=tl["dealt_cards"],
                    known_discards_self=tl.get("discards_self", []),
                    turn=tl["turn"],
                    is_btn=tl.get("is_btn", False),
                )

                states_mm[total] = encode_state(obs)
                actions_mm[total] = action_to_index(tl["action"], tl["turn"])

                player = tl["player"]
                royalties_mm[total] = hr["royalties"][player]["total"]
                busted_mm[total] = float(hr["busted"][player])
                fl_mm[total] = float(hr["fl_entry"][player])
                total += 1
            except Exception:
                skipped += 1

            if (line_num + 1) % 1000000 == 0:
                elapsed = time.time() - start_time
                rate = (line_num + 1) / elapsed
                print(f"  {line_num+1:,}/{n_lines:,} lines "
                      f"({total:,} samples, {rate:.0f}/s)")

    elapsed = time.time() - start_time
    print(f"\nEncoding done: {total:,} samples in {elapsed:.1f}s")

    # Flush and truncate memmap to actual size
    states_mm.flush()
    actions_mm.flush()
    royalties_mm.flush()
    busted_mm.flush()
    fl_mm.flush()
    del states_mm, actions_mm, royalties_mm, busted_mm, fl_mm

    # Save as proper .npy files (read back + slice)
    print("Saving final numpy files...")

    for name, dtype in [("states", np.float32), ("actions", np.int64),
                         ("royalties", np.float32), ("busted", np.float32),
                         ("fl_entry", np.float32)]:
        tmp = output_dir / f"{name}.npy.tmp"
        if name == "states":
            mm = np.memmap(tmp, dtype=dtype, mode='r', shape=(n_lines, 490))
            np.save(output_dir / f"{name}.npy", mm[:total])
        else:
            mm = np.memmap(tmp, dtype=dtype, mode='r', shape=(n_lines,))
            np.save(output_dir / f"{name}.npy", mm[:total])
        del mm
        tmp.unlink()

    # Create all-true mask (no action enumeration)
    masks = np.ones((total, MAX_ACTIONS), dtype=bool)
    np.save(output_dir / "valid_masks.npy", masks)
    del masks

    bust_rate = float(np.load(output_dir / "busted.npy").mean())
    fl_rate = float(np.load(output_dir / "fl_entry.npy").mean())
    avg_roy = float(np.load(output_dir / "royalties.npy").mean())

    metadata = {
        "total_samples": total,
        "skipped": skipped,
        "state_dim": 490,
        "max_actions": MAX_ACTIONS,
        "busted_ratio": bust_rate,
        "fl_ratio": fl_rate,
        "avg_royalty": avg_roy,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone! {total:,} samples -> {output_dir}")
    print(f"  Bust: {bust_rate:.1%}, FL: {fl_rate:.1%}, Avg royalty: {avg_roy:.1f}")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="data/processed_1m")
    args = parser.parse_args()
    preprocess_fast(args.input, args.output)
