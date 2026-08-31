"""
Split existing Expectimax training data by turn number.

Usage:
    python -m ai.training.split_by_turn \
        --data data/expectimax_train_v4 \
        --out-prefix data/expectimax_turn
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split BC data by turn")
    parser.add_argument("--data", required=True, help="Input data directory")
    parser.add_argument("--out-prefix", required=True,
                        help="Output prefix (e.g. data/expectimax_turn)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    turns = np.load(data_dir / "turns.npy")

    # Files to split
    files = ["states", "actions", "valid_masks", "action_evs",
             "rewards", "busted", "fl_entry", "royalties"]

    arrays = {}
    for f in files:
        p = data_dir / f"{f}.npy"
        if p.exists():
            arrays[f] = np.load(p)
            print(f"  Loaded {f}: {arrays[f].shape} {arrays[f].dtype}")

    for t in range(5):
        mask = turns == t
        n = mask.sum()
        if n == 0:
            continue

        out_dir = Path(f"{args.out_prefix}_T{t}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for f, arr in arrays.items():
            subset = arr[mask]
            np.save(out_dir / f"{f}.npy", subset)

        print(f"  T{t}: {n:,} samples -> {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
