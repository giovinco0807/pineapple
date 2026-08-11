#!/usr/bin/env python3
"""Prepare, run/resume, or reconstruct the HU RL replay v2 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ofc_regular.hu_rl_replay_resume_v2 import (
    HuRlReplayResumeV2Error,
    prepare_replay_resume_run,
    run_or_resume_replay,
    validate_replay_resume_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write the immutable run manifest")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--paired-hand-start", type=int, default=0)
    prepare.add_argument("--paired-hands", type=int, required=True)
    prepare.add_argument("--shard-pairs", type=int, required=True)
    prepare.add_argument("--seed-base", type=int, required=True)
    prepare.add_argument("--seed-stride", type=int, required=True)
    prepare.add_argument("--policy-a-id", required=True)
    prepare.add_argument("--policy-b-id", required=True)
    prepare.add_argument("--chunk-width", type=int, default=64)
    prepare.add_argument("--thread-count", type=int, default=8)

    run = subparsers.add_parser("run", help="validate the durable prefix and append shards")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--seed-base", type=int, required=True)
    run.add_argument(
        "--max-new-shards",
        type=int,
        help="Intentional bounded pause after this many newly written shards.",
    )

    validate = subparsers.add_parser("validate", help="reconstruct every durable trajectory")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--seed-base", type=int, required=True)
    validate.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_replay_resume_run(
            args.output_dir,
            run_id=args.run_id,
            paired_hand_start=args.paired_hand_start,
            paired_hand_count=args.paired_hands,
            shard_pair_count=args.shard_pairs,
            seed_base=args.seed_base,
            seed_stride=args.seed_stride,
            policy_a_id=args.policy_a_id,
            policy_b_id=args.policy_b_id,
            chunk_width=args.chunk_width,
            thread_count=args.thread_count,
        )
    elif args.command == "run":
        result = run_or_resume_replay(
            args.output_dir,
            seed_base=args.seed_base,
            max_new_shards=args.max_new_shards,
        )
    else:
        result = validate_replay_resume_run(
            args.output_dir,
            seed_base=args.seed_base,
            require_complete=not args.allow_incomplete,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except HuRlReplayResumeV2Error as error:
        print(f"HU RL replay/resume v2 failed closed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
