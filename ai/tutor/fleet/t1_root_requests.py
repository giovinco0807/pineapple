"""Deterministic T1-vs-FL label requests for one fleet shard.

A shard is a contiguous range of absolute root seeds, and a root's deal is a
pure function of its seed, so a shard's work is defined entirely by
`(seed, roots)`.  That is what makes a preempted shard re-runnable without
coordinating with anything: relaunch it with the same pair and it asks for the
same roots.

The deal itself is NOT redefined here.  `generate_t1_vs_fl_teacher.
sample_t1_root` is the dealing logic the T1 teacher has always used, so it is
imported rather than reimplemented -- a second sampler that agreed today would
be a second sampler to keep in step forever.

The one override is `opp_count`.  The sampler draws it from 14..17, but the
`.jfl1` pool is solved at a single width and the labeler refuses a request
whose width disagrees with the pool's, so a pool run pins it -- exactly what
`--force-opp-count` does in the local generator.

Usage:
    python -m ai.tutor.fleet.t1_root_requests --seed 101000000 --roots 20 \
        --out requests.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from ai.tutor.generate_t1_vs_fl_teacher import sample_t1_root


def requests_for(
    *,
    seed: int,
    roots: int,
    opp_count: int = 14,
    t2_samples: int = 32,
    t3_samples: int = 10,
    t4_draw_sample: int = 60,
    pool_opponents: int = 1500,
    truncate_depth: int | None = 0,
    only_ids: set[str] | None = None,
) -> Iterator[dict]:
    """The shard's requests, in seed order, optionally filtered to `only_ids`.

    `truncate_depth=0` stops the playout at the T2 chooser's own value, so the
    T3 chooser is never reached and `t3_samples` / `t4_draw_sample` are inert.
    They are still written: the labeler defaults them if absent, and a label
    file that records the knobs it was produced under is worth more than one
    that leaves them to be re-derived.
    """
    for index in range(roots):
        root_seed = seed + index
        if only_ids is not None and str(root_seed) not in only_ids:
            continue
        root = sample_t1_root(root_seed)
        root["opp_count"] = opp_count
        root["t2_samples"] = t2_samples
        root["t3_samples"] = t3_samples
        root["t4_draw_sample"] = t4_draw_sample
        root["pool_opponents"] = pool_opponents
        if truncate_depth is not None:
            root["truncate_depth"] = truncate_depth
        yield root


def write_requests(path: Path, rows: Iterator[dict]) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            written += 1
    return written


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The knobs a run is defined by, shared with the shard runner."""
    parser.add_argument("--opp-count", type=int, default=14)
    parser.add_argument("--t2-samples", type=int, default=32)
    parser.add_argument("--t3-samples", type=int, default=10)
    parser.add_argument("--t4-draw-sample", type=int, default=60)
    parser.add_argument("--pool-opponents", type=int, default=1500)
    parser.add_argument(
        "--truncate-depth", type=int, default=0,
        help="-1 plays every line out to an eleven-card board",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_arguments(parser)
    args = parser.parse_args()
    count = write_requests(
        args.out,
        requests_for(
            seed=args.seed,
            roots=args.roots,
            opp_count=args.opp_count,
            t2_samples=args.t2_samples,
            t3_samples=args.t3_samples,
            t4_draw_sample=args.t4_draw_sample,
            pool_opponents=args.pool_opponents,
            truncate_depth=None if args.truncate_depth < 0 else args.truncate_depth,
        ),
    )
    print(f"wrote {count} request(s) to {args.out}")


if __name__ == "__main__":
    main()
