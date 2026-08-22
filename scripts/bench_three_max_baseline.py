"""M1 baseline: measure the 3-max policy ladder on the rotation harness.

    python scripts/bench_three_max_baseline.py --blocks 150

The ladder's rungs differ only in Monte-Carlo compute, so what this measures is
how much a decision is worth when nothing but search budget changes.  Every
matchup runs the rotation self-test first: three copies of one policy must
cancel to exactly 0.0 per block, and no number below is meaningful without it.

Power (measured block standard deviations): resolving a 1.0/hand edge needs
roughly 90-140 blocks; a 0.5/hand edge needs roughly 360-570.  Do not read a
verdict out of a CI that straddles zero -- add blocks instead.

The default cyclic rotation fixes each ordered pair's relative seating order,
which is harmless while the two field policies are identical (as in every
matchup below).  For a genuine A-vs-B-vs-C gate pass --orientations full: all
six seatings, so both relative orders of every pair are played.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.three_max import (  # noqa: E402
    PLAYER_COUNT,
    evaluate_matchup,
    mc_policy,
    self_test,
    uniform_random_policy,
)


def _matchups(sims_hero: int, sims_field: int):
    return [
        (
            "mc vs random",
            {
                0: mc_policy(sims=sims_hero),
                1: uniform_random_policy,
                2: uniform_random_policy,
            },
            {0: f"mc(sims={sims_hero})", 1: "random", 2: "random"},
        ),
        (
            "mc vs cheap mc",
            {
                0: mc_policy(sims=sims_hero),
                1: mc_policy(sims=sims_field),
                2: mc_policy(sims=sims_field),
            },
            {
                0: f"mc(sims={sims_hero})",
                1: f"mc(sims={sims_field})",
                2: f"mc(sims={sims_field})",
            },
        ),
        (
            "cheap mc vs random",
            {
                0: mc_policy(sims=sims_field),
                1: uniform_random_policy,
                2: uniform_random_policy,
            },
            {0: f"mc(sims={sims_field})", 1: "random", 2: "random"},
        ),
    ]


def adjacency_probe(*, blocks: int, base_seed: int, sims_hero: int) -> None:
    """Does sitting immediately after the strong player cost you?

    The rotation cancels the ABSOLUTE seat term, because every player visits
    every seat.  It does not cancel RELATIVE order: player 1 always acts
    immediately after player 0 and player 2 always acts two after, in every
    rotation, by construction of ``seat_for``.  Players 1 and 2 here run the
    same policy, so any gap between them is a positional effect and not a
    policy effect.  Both are measured on the same blocks, so the paired
    difference is far tighter than either mean alone.
    """
    import math

    from ofc_regular.three_max import play_block

    hero = mc_policy(sims=sims_hero)
    policies = {0: hero, 1: uniform_random_policy, 2: uniform_random_policy}
    differences: list[float] = []
    for index in range(blocks):
        block = play_block(seed=base_seed + index, policies_by_player=policies)
        differences.append(block.totals[1] - block.totals[2])

    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    stderr = math.sqrt(variance / len(differences))
    per_hand = mean / 3.0
    half = 1.96 * stderr / 3.0
    print("=== adjacency probe: next-to-strong vs two-from-strong ===")
    print(f"  hero mc(sims={sims_hero}), both others uniform random, {blocks} blocks")
    print(
        f"  paired difference (p1 - p2) {per_hand:+.3f} / hand   "
        f"95% CI [{per_hand - half:+.3f}, {per_hand + half:+.3f}]"
    )
    verdict = "NO measurable adjacency effect" if abs(per_hand) < half else "adjacency effect detected"
    print(f"  {verdict}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=150)
    parser.add_argument("--base-seed", type=int, default=7_100_000)
    parser.add_argument("--sims-hero", type=int, default=32)
    parser.add_argument("--sims-field", type=int, default=4)
    parser.add_argument("--only", type=str, default=None, help="substring of a matchup name")
    parser.add_argument("--adjacency", action="store_true", help="run the adjacency probe only")
    parser.add_argument("--orientations", choices=("cyclic", "full"), default="cyclic")
    args = parser.parse_args()

    if args.adjacency:
        adjacency_probe(
            blocks=args.blocks, base_seed=args.base_seed, sims_hero=args.sims_hero
        )
        return

    print("rotation self-test (identical policies must cancel to exactly 0.0)")
    started = time.time()
    self_test(policy=uniform_random_policy, seeds=range(args.base_seed, args.base_seed + 40))
    self_test(policy=mc_policy(sims=args.sims_field), seeds=range(args.base_seed, args.base_seed + 3))
    print(f"  passed in {time.time() - started:.1f}s\n")

    for name, policies, labels in _matchups(args.sims_hero, args.sims_field):
        if args.only and args.only not in name:
            continue
        print(f"=== {name} ===", flush=True)
        started = time.time()
        summary = evaluate_matchup(
            policies_by_player=policies,
            blocks=args.blocks,
            base_seed=args.base_seed,
            orientations=args.orientations,
            progress_every=max(1, args.blocks // 4),
        )
        elapsed = time.time() - started
        print(summary.describe(labels))
        total = sum(summary.mean_per_hand[p] for p in range(PLAYER_COUNT))
        print(
            f"  zero-sum check {total:+.6f}   "
            f"{elapsed:.1f}s ({elapsed / args.blocks:.2f}s/block)\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
