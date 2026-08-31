"""Generate the exact fourteen-card deal stream used by `fl_solver::pool::deal`.

The offset makes separately generated/resumed shards content-disjoint while
retaining one documented seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MASK64 = (1 << 64) - 1
RANKS = "23456789TJQKA"
SUITS = "shdc"


def deal(seed: int, index: int, width: int = 14) -> list[tuple[int, int]]:
    deck = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
    deck += [(0, 4), (0, 4)]
    state = ((seed * 0x9E3779B97F4A7C15) + (index * 0xBF58476D1CE4E5B9)) & MASK64
    state ^= 0xD6E8FEB86659FD93

    def next_u64() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    for position in range(len(deck) - 1, 0, -1):
        target = next_u64() % (position + 1)
        deck[position], deck[target] = deck[target], deck[position]
    return deck[:width]


def card_names(cards: list[tuple[int, int]]) -> list[str]:
    names: list[str] = []
    jokers = 0
    for rank, suit in cards:
        if rank == 0:
            jokers += 1
            names.append(f"X{jokers}")
        else:
            names.append(RANKS[rank - 2] + SUITS[suit])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.count < 0:
        raise SystemExit("offset and count must be non-negative")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(args.offset, args.offset + args.count):
            handle.write(
                json.dumps(
                    {
                        "id": str((args.seed + index) & MASK64),
                        "cards": card_names(deal(args.seed, index)),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
