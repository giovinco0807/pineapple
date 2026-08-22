"""Gate ii phase A: on-policy (T0, second) positions plus the champion's pick.

Production environment only.  Each record: the first seat's board as the
champion actually built it, the second seat's five cards, and what the champion
plays there.  Phase B (worktree) prices the full fan with the elimination
method; phase C compares.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "2")

ROOT = pathlib.Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple")
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ofc_regular.cards import create_deck  # noqa: E402
from trainer.engine_eval import decide_with_engine  # noqa: E402

EMPTY = {"top": [], "middle": [], "bottom": []}


def rows_from(placements) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {"top": [], "middle": [], "bottom": []}
    for card, row in placements:
        rows[row].append(card)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=93_000_000)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    records = []
    for i in range(args.positions):
        seed = args.seed_base + i
        rng = random.Random(seed)
        deck = create_deck(shuffle=True, rng=rng)
        p0_dealt, p1_dealt = deck[0:5], deck[5:10]

        first = decide_with_engine(hero_board=dict(EMPTY), opp_board=dict(EMPTY),
                                   dealt=list(p0_dealt), dead=[], turn=0,
                                   position="first")
        opp_board = rows_from(first["action"]["placements"])

        champ = decide_with_engine(hero_board=dict(EMPTY), opp_board=opp_board,
                                   dealt=list(p1_dealt), dead=[], turn=0,
                                   position="second")
        records.append({
            "index": i,
            "hand_seed": seed,
            "opp_board": opp_board,
            "hand": list(p1_dealt),
            "champ_placements": [[c, r] for c, r in champ["action"]["placements"]],
            "first_evaluator": first["evaluator"],
        })
        print(f"[{i:02d}] opp {opp_board['top']}|{opp_board['middle']}|"
              f"{opp_board['bottom']}  hand {' '.join(p1_dealt)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"\nwrote {len(records)} positions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
