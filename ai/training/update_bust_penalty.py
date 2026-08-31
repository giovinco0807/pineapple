"""
Estimate optimal bust_penalty from self-play game logs.

bust_penalty ≈ E[score | not bust] - E[score | bust]

This represents how much worse busting is compared to not busting,
which is the theoretically optimal penalty for the VN bust_prob head.

Usage:
    python ai/training/update_bust_penalty.py data/selfplay.jsonl
    python ai/training/update_bust_penalty.py data/sp_iter1.jsonl data/sp_iter2.jsonl
"""
import json
import argparse
from pathlib import Path


def aggregate_bust_stats(jsonl_paths):
    """Aggregate bust/non-bust scores from self-play JSONL files."""
    bust_scores = []
    non_bust_scores = []

    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    game = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                result = game["result"]

                # Hero perspective
                if result["hero_busted"]:
                    bust_scores.append(result["total_score"])
                else:
                    non_bust_scores.append(result["total_score"])

                # Opp perspective (score negated)
                if result["opp_busted"]:
                    bust_scores.append(-result["total_score"])
                else:
                    non_bust_scores.append(-result["total_score"])

    return bust_scores, non_bust_scores


def compute_bust_penalty(bust_scores, non_bust_scores):
    """Compute optimal bust_penalty."""
    if not bust_scores or not non_bust_scores:
        return 0.0

    bust_mean = sum(bust_scores) / len(bust_scores)
    non_bust_mean = sum(non_bust_scores) / len(non_bust_scores)
    penalty = non_bust_mean - bust_mean

    bust_std = (sum((s - bust_mean) ** 2 for s in bust_scores) / max(len(bust_scores) - 1, 1)) ** 0.5
    non_bust_std = (sum((s - non_bust_mean) ** 2 for s in non_bust_scores) / max(len(non_bust_scores) - 1, 1)) ** 0.5

    print(f"  Bust games:     n={len(bust_scores):5d}, mean={bust_mean:+.2f}, std={bust_std:.2f}")
    print(f"  Non-bust games: n={len(non_bust_scores):5d}, mean={non_bust_mean:+.2f}, std={non_bust_std:.2f}")
    print(f"  Raw penalty:    {penalty:.2f}")

    return penalty


def main():
    parser = argparse.ArgumentParser(description="Estimate bust_penalty from self-play logs")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    parser.add_argument("--scale", type=float, default=0.8,
                        help="Scale factor for penalty (conservative, default: 0.8)")
    args = parser.parse_args()

    print("Aggregating bust statistics...")
    bust_scores, non_bust_scores = aggregate_bust_stats(args.inputs)

    total = len(bust_scores) + len(non_bust_scores)
    bust_rate = len(bust_scores) / max(total, 1) * 100
    print(f"  Total: {total} games, bust rate: {bust_rate:.1f}%\n")

    raw_penalty = compute_bust_penalty(bust_scores, non_bust_scores)
    scaled_penalty = raw_penalty * args.scale

    print(f"\n  Scaled penalty (x{args.scale}): {scaled_penalty:.2f}")
    print(f"\n  Use: --bust-penalty {scaled_penalty:.1f}")


if __name__ == "__main__":
    main()
