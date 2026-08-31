"""
Merge multiple BC data directories into a single training set.

Usage:
    python -m ai.training.merge_bc_data \
        --dirs data/expectimax_train_v4 data/dagger_bc_v1 \
        --save data/bc_merged_v5 \
        --oversample-last 5
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge BC data directories")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Data directories to merge")
    parser.add_argument("--save", required=True, help="Output directory")
    parser.add_argument("--oversample-last", type=int, default=1,
                        help="Oversample last directory N times (for DAgger data)")
    args = parser.parse_args()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    files = ["states", "actions", "valid_masks", "action_evs",
             "royalties", "busted", "fl_entry", "rewards"]

    parts = {f: [] for f in files}

    for i, d in enumerate(args.dirs):
        d = Path(d)
        n = None
        for f in files:
            p = d / f"{f}.npy"
            if p.exists():
                arr = np.load(p)
                if n is None:
                    n = len(arr)
                repeat = args.oversample_last if i == len(args.dirs) - 1 else 1
                if repeat > 1:
                    if arr.ndim == 1:
                        arr = np.tile(arr, repeat)
                    else:
                        arr = np.tile(arr, (repeat, 1))
                parts[f].append(arr)
                if repeat > 1:
                    print(f"  {d}/{f}.npy: {n} x{repeat} = {len(arr)}")
            else:
                print(f"  WARNING: {p} not found, using zeros")
                # Need to infer shape from other loaded data
                ref = parts[f][-1] if parts[f] else None
                if ref is not None:
                    zeros = np.zeros_like(ref[:n])
                    parts[f].append(zeros)

        total = len(parts["states"][-1]) if parts["states"] else 0
        print(f"  {d}: {n} samples" + (f" (x{args.oversample_last} = {total})"
              if i == len(args.dirs) - 1 and args.oversample_last > 1 else ""))

    # Concatenate and save
    for f in files:
        if parts[f]:
            merged = np.concatenate(parts[f])
            np.save(save_dir / f"{f}.npy", merged)
            print(f"  Saved {f}.npy: {merged.shape} {merged.dtype}")

    # Stats
    states = np.load(save_dir / "states.npy")
    actions = np.load(save_dir / "actions.npy")
    busted = np.load(save_dir / "busted.npy")
    print(f"\n  Total: {len(states)} samples")
    print(f"  Bust > 0.5: {(busted > 0.5).sum()} ({100*(busted > 0.5).mean():.1f}%)")
    print(f"  Saved to: {save_dir}")


if __name__ == "__main__":
    main()
