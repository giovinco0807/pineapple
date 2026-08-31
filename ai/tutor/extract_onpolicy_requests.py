"""Street-decision requests from a traced self-play corpus.

The generation-2 lever: roots drawn from the games the CURRENT chain plays,
so every label lands on a state the policy actually visits.  Visibility
follows the owner-confirmed rules -- within a street the second seat (BTN)
sees the first seat's placement, discards stay face down -- so:

    BB  at street k sees the opponent's board after street k-1
    BTN at street k sees the opponent's board after street k (BB placed)

`opp_after` (the opponent's own street-k placement) rides along on BB rows:
the chooser teacher ignores it, but it keeps traced-style relabels and
apples-to-apples comparisons possible from the same corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = {
        (street, seat): (args.out_dir / f"t{street}_{'bb' if seat == 0 else 'btn'}_requests.jsonl"
                         ).open("w", encoding="utf-8", newline="\n")
        for street in range(4) for seat in range(2)
    }
    hands = 0
    for line in args.traces.open(encoding="utf-8"):
        if not line.strip():
            continue
        hand = json.loads(line)
        steps = {(s["street"], s["seat"]): s for s in hand["steps"]}
        boards = {(-1, 0): [[], [], []], (-1, 1): [[], [], []]}
        deads = {0: [], 1: []}
        for street in range(5):
            for seat in range(2):
                step = steps[(street, seat)]
                boards[(street, seat)] = step["board"]
        for street in range(4):
            for seat in range(2):
                step = steps[(street, seat)]
                before = boards[(street - 1, seat)]
                opp = 1 - seat
                opp_street = street - 1 if seat == 0 else street
                opp_board = boards[(opp_street, opp)] if opp_street >= 0 else [[], [], []]
                row = {
                    # `hands`, not hand["hand"]: the per-shard hand ids restart
                    # at 0 in every trace file, and the teacher seeds its
                    # opponent draws from this id -- colliding ids would give
                    # 60 roots the same draw stream (and dedup would eat them).
                    "id": f"{hands}s{street}{'b' if seat == 0 else 'n'}",
                    "board": before,
                    "dead": list(deads[seat]),
                    "draw": step["draw"],
                    "opp_board": opp_board,
                }
                if seat == 0:
                    row["opp_after"] = boards[(street, opp)]
                outs[(street, seat)].write(
                    json.dumps(row, separators=(",", ":")) + "\n"
                )
            for seat in range(2):
                step = steps[(street, seat)]
                if street > 0 and step.get("discard"):
                    deads[seat].append(step["discard"])
        hands += 1
    for handle in outs.values():
        handle.close()
    print(f"{hands} hands -> 8 request files in {args.out_dir}")


if __name__ == "__main__":
    main()
