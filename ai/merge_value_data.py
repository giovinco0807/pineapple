"""
Merge multiple value data NPZ files into one.

Usage:
    python ai/merge_value_data.py data/value_data_v3.npz data/value_data_v4_new.npz --output data/value_data_v4.npz
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge value data NPZ files")
    parser.add_argument('inputs', nargs='+', help='Input NPZ files')
    parser.add_argument('--output', required=True, help='Output NPZ file')
    args = parser.parse_args()

    all_obs = []
    all_scores = []
    all_turns = []

    for path in args.inputs:
        if not Path(path).exists():
            print(f"  SKIP (not found): {path}")
            continue
        d = np.load(path)
        obs = d['obs']
        score = d['score']
        turn = d['turn']
        print(f"  {path}: {len(obs)} samples, "
              f"score mean={score.mean():.2f} std={score.std():.2f} "
              f"range=[{score.min():.1f}, {score.max():.1f}]")
        all_obs.append(obs)
        all_scores.append(score)
        all_turns.append(turn)

    merged_obs = np.concatenate(all_obs, axis=0)
    merged_scores = np.concatenate(all_scores, axis=0)
    merged_turns = np.concatenate(all_turns, axis=0)

    np.savez_compressed(
        args.output,
        obs=merged_obs,
        score=merged_scores,
        turn=merged_turns,
    )

    print(f"\nMerged: {len(merged_obs)} samples -> {args.output}")
    print(f"  Score mean={merged_scores.mean():.2f} std={merged_scores.std():.2f}")
    print(f"  Range=[{merged_scores.min():.1f}, {merged_scores.max():.1f}]")
    print(f"  Turn distribution: {np.bincount(merged_turns.astype(np.int64)).tolist()}")


if __name__ == '__main__':
    main()
