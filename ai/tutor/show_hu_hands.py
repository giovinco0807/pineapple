"""Print traced heads-up hands street by street, both boards side by side.

`show_fl14_hands` prints one seat against a Fantasyland opponent whose board
is face down.  This one is for the HU chain, where both boards are public and
the whole point of the phase is that each seat can see the other -- so a
placement that only makes sense against what the opponent has done is only
legible with the two boards printed together.

Order matters and is preserved: BB acts first on every street, and BTN places
knowing what BB just did.  The `by` column says what chose each placement --
a street evaluator, the exact V4, or the closed form -- because a chain that
mixes learned and exact decisions should never leave you guessing which one
you are reading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = ("top", "mid", "bot")
CAP = (3, 5, 5)
SEATS = ("BB ", "BTN")


def row_text(cards: list[str], capacity: int) -> str:
    filled = " ".join(f"{c:>2}" for c in cards)
    empty = " ".join(" ." for _ in range(capacity - len(cards)))
    return f"{filled} {empty}".strip().ljust(capacity * 3)


def board_lines(board: list[list[str]]) -> list[str]:
    return [
        f"{ROWS[r]} {row_text(board[r], CAP[r])}" for r in range(3)
    ]


def show(hand: dict, verdict_only: bool = False) -> None:
    print(f"{'=' * 72}")
    settle = hand["bb_points"]
    print(f"HAND {hand['hand']}   BB {settle:+d} points")
    latest = [[[], [], []], [[], [], []]]
    for step in hand["steps"]:
        seat = step["seat"]
        latest[seat] = step["board"]
        if verdict_only:
            continue
        discard = step.get("discard")
        toss = f"  discard {discard}" if discard and step["street"] > 0 else ""
        print(
            f"\n  T{step['street']} {SEATS[seat]}  draw {' '.join(step['draw'])}"
            f"{toss}   [{step['by']}]"
        )
        left = board_lines(latest[0])
        right = board_lines(latest[1])
        for index in range(3):
            marker = "<-" if index == 0 else "  "
            print(f"    BB   {left[index]}   |  BTN  {right[index]}")
    print("\n  final")
    left = board_lines(latest[0])
    right = board_lines(latest[1])
    for index in range(3):
        print(f"    BB   {left[index]}   |  BTN  {right[index]}")
    def tag(prefix: str) -> str:
        if hand[f"{prefix}_busted"]:
            return "FOUL"
        bits = [f"royalty {hand[f'{prefix}_royalty']}"]
        if hand[f"{prefix}_entry"] >= 14:
            bits.append(f"FL{hand[f'{prefix}_entry']}")
        return ", ".join(bits)
    print(f"    BB   {tag('bb')}")
    print(f"    BTN  {tag('btn')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=0,
                        help="how many to print; 0 prints all")
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()

    hands = [
        json.loads(line) for line in args.trace.open(encoding="utf-8") if line.strip()
    ]
    if args.hands:
        hands = hands[: args.hands]
    for hand in hands:
        show(hand, verdict_only=args.final_only)
    print(f"\n{'=' * 72}")
    fouls = sum(h["bb_busted"] + h["btn_busted"] for h in hands)
    entries = sum(
        (h["bb_entry"] >= 14 and not h["bb_busted"])
        + (h["btn_entry"] >= 14 and not h["btn_busted"])
        for h in hands
    )
    print(
        f"{len(hands)} hands, {2 * len(hands)} boards: "
        f"{fouls} fouled, {entries} entered Fantasyland"
    )


if __name__ == "__main__":
    main()
