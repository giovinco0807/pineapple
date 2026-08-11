#!/usr/bin/env python3
"""Run the hash-bound combined-vs-separate packed PyO3 diagnostic locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ofc_regular.hu_rl_native_benchmark import (
    DEFAULT_BENCHMARK_SEED,
    canonical_benchmark_json,
    run_native_batch_mechanics_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--chunk-width", type=int, default=64)
    parser.add_argument("--thread-count", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON receipt; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print stdout; persisted JSON remains canonical and compact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = run_native_batch_mechanics_benchmark(
        seed=args.seed,
        chunk_width=args.chunk_width,
        thread_count=args.thread_count,
    )
    canonical = canonical_benchmark_json(document)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical)
            handle.write("\n")
    if args.pretty:
        print(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
    else:
        print(canonical)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
