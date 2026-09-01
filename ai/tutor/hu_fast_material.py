"""Turn champion traces into per-slot teaching material for the fast nets.

One traced hand carries seven model decisions; each becomes a sample of
(state as the deciding seat saw it, the move the champion chose).  The label
is the MOVE, never its value: a distilled net that picks a slightly wrong
move costs a candidate comparison something that largely cancels across
candidates, while one that returns a wrong VALUE reorders them -- measured
at 66x on the regular track, and the reason every labeler here lets models
choose and takes numbers from exact scoring.

What the deciding seat can see, and nothing more:
  * its own board and its own discards (it threw them, so it knows them),
  * the opponent's board as far as the opponent has played THIS street --
    BB acts first, so at street s BTN sees BB's street-s board while BB sees
    only BTN's street-(s-1) board,
  * its own three drawn cards.
The opponent's discards are face down and are NOT in the state.

Action index for streets 1-4: the draw is sorted by card name, one of the
three is discarded and the other two are placed, so
    index = discard_position * 9 + row(first kept) * 3 + row(second kept)
which is 27 slots, of which the capacity-violating ones are masked.

Usage:
    python -m ai.tutor.hu_fast_material --traces D:/ofc_data/hu/fast/traces \
        --out D:/ofc_data/hu/fast/material
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
CAP = (3, 5, 5)
SLOTS = ("t1_bb", "t1_btn", "t2_bb", "t2_btn", "t3_bb", "t3_btn",
         "t4_bb", "t4_btn", "t0_bb", "t0_btn")


def card_index(name: str) -> int:
    if name.startswith("X"):
        return 52 + (0 if name == "X1" else 1)
    return SUITS.index(name[1]) * 13 + RANKS.index(name[0])


def action_index(draw: list[str], placed: dict[str, int]) -> int:
    """(discard, row of each kept card) as one of 27 slots.

    `placed` maps a drawn card to the row it went to; the card missing from
    it is the discard.  The draw is sorted first so the same physical
    decision always lands on the same index.
    """
    order = sorted(draw)
    kept = [c for c in order if c in placed]
    if len(kept) != 2:
        raise ValueError(f"expected two placed cards, got {kept} from {draw}")
    discard_pos = next(i for i, c in enumerate(order) if c not in placed)
    return discard_pos * 9 + placed[kept[0]] * 3 + placed[kept[1]]


def legal_mask(draw: list[str], room: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(27, np.bool_)
    for discard_pos in range(3):
        for r1 in range(3):
            for r2 in range(3):
                need = [0, 0, 0]
                need[r1] += 1
                need[r2] += 1
                if all(need[i] <= room[i] for i in range(3)):
                    mask[discard_pos * 9 + r1 * 3 + r2] = True
    return mask


def board_of(rows) -> list[list[str]]:
    return [list(r) for r in rows]


def extract_hand(hand: dict):
    """Every model decision in one traced hand, as (slot, sample)."""
    steps = sorted(hand["steps"], key=lambda s: (s["street"], s["seat"]))
    boards = {0: [[], [], []], 1: [[], [], []]}
    discards = {0: [], 1: []}
    out = []
    for step in steps:
        street, seat = step["street"], step["seat"]
        before = board_of(boards[seat])
        after = board_of(step["board"])
        draw = list(step["draw"])
        opponent = board_of(boards[1 - seat])
        if street > 0:
            placed = {}
            for row, cards in enumerate(after):
                for card in cards:
                    if card in draw and card not in [c for r in before for c in r]:
                        placed[card] = row
            if len(placed) != 2:
                boards[seat] = after
                continue
            thrown = [c for c in draw if c not in placed]
            room = tuple(CAP[i] - len(before[i]) for i in range(3))
            sample = {
                "own": before, "opp": opponent, "draw": draw,
                "own_discards": list(discards[seat]),
                "action": action_index(draw, placed),
                "mask": legal_mask(draw, room).tolist(),
                "by": step["by"],
            }
            discards[seat].extend(thrown)
            out.append((f"t{street}_{'bb' if seat == 0 else 'btn'}", sample))
        boards[seat] = after
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, required=True,
                    help="directory of trace jsonl shards")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    writers = {}
    counts = Counter()
    by_source = Counter()
    hands = 0
    for shard in sorted(args.traces.glob("*.jsonl")):
        for line in shard.open(encoding="utf-8"):
            if not line.strip():
                continue
            hands += 1
            for slot, sample in extract_hand(json.loads(line)):
                if slot not in writers:
                    writers[slot] = (args.out / f"{slot}.jsonl").open(
                        "w", encoding="utf-8")
                writers[slot].write(json.dumps(sample) + "\n")
                counts[slot] += 1
                by_source[sample["by"]] += 1
    for w in writers.values():
        w.close()
    print(f"hands {hands}")
    for slot in sorted(counts):
        print(f"  {slot}: {counts[slot]}")
    print(f"chosen by: {dict(by_source)}")


if __name__ == "__main__":
    main()
