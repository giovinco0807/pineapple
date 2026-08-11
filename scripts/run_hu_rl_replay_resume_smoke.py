#!/usr/bin/env python3
"""Run the bounded deterministic replay/resume subprocess smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ofc_regular.hu_rl_replay_resume_smoke import (
    DEFAULT_LANE_COUNT,
    DEFAULT_PREFIX_DECISIONS,
    DEFAULT_RUN_SEED,
    HuRlReplayResumeSmokeError,
    canonical_replay_resume_json,
    run_deterministic_replay_resume_smoke,
    run_resume_worker_stdio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN_SEED)
    parser.add_argument("--lanes", type=int, default=DEFAULT_LANE_COUNT)
    parser.add_argument(
        "--prefix-decisions",
        type=int,
        default=DEFAULT_PREFIX_DECISIONS,
    )
    parser.add_argument("--chunk-width", type=int, default=32)
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        help="Required write-once public receipt path.",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--resume-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume_worker:
        if args.output is not None or args.pretty:
            raise HuRlReplayResumeSmokeError("resume worker rejects public-output options")
        return run_resume_worker_stdio()
    if args.output is None:
        raise HuRlReplayResumeSmokeError("public smoke requires --output")
    if args.output.exists():
        raise HuRlReplayResumeSmokeError("public receipt path already exists")
    receipt = run_deterministic_replay_resume_smoke(
        run_seed=args.seed,
        lane_count=args.lanes,
        prefix_decisions=args.prefix_decisions,
        chunk_width=args.chunk_width,
        thread_count=args.thread_count,
    )
    canonical = canonical_replay_resume_json(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical)
        handle.write("\n")
    if args.pretty:
        print(json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=True))
    else:
        print(canonical)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except HuRlReplayResumeSmokeError as error:
        print(f"replay resume smoke failed closed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
