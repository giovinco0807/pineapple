"""T1 gate phase A: on-policy T1 positions for both seats, plus champion picks.

Production environment.  A T1-first position: both T0 placements are the
champion's own, hero holds three fresh cards, opponent shows five.  A T1-second
position additionally has the opponent's T1 already played (two placed, one
discarded face down), so the opponent shows seven and one discard is counted
but not named.
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


def rows_from(placements, base=None) -> dict[str, list[str]]:
    rows = {r: list(base[r]) if base else [] for r in ("top", "middle", "bottom")}
    for card, row in placements:
        rows[row].append(card)
    return rows


def act(hero, opp, dealt, dead, turn, position):
    out = decide_with_engine(hero_board=hero, opp_board=opp, dealt=list(dealt),
                             dead=list(dead), turn=turn, position=position)
    action = out["action"]
    return rows_from(action["placements"], hero), action.get("discard"), action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", choices=["first", "second"], required=True)
    parser.add_argument("--positions", type=int, default=120)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    records = []
    for i in range(args.positions):
        seed = args.seed_base + i
        rng = random.Random(seed)
        deck = create_deck(shuffle=True, rng=rng)

        # T0 both seats, champion play, dealing exactly as play_ai does.
        p0_board, _, _ = act(dict(EMPTY), dict(EMPTY), deck[0:5], [], 0, "first")
        p1_board, _, _ = act(dict(EMPTY), p0_board, deck[5:10], [], 0, "second")

        if args.seat == "first":
            hero_board, opp_board, opp_discards = p0_board, p1_board, 0
            dealt = deck[10:13]
        else:
            # The first seat plays its T1 before the second seat sees the street.
            p0_after, p0_discard, _ = act(p0_board, p1_board, deck[10:13], [],
                                          1, "first")
            hero_board, opp_board, opp_discards = p1_board, p0_after, 1
            dealt = deck[13:16]

        champ_rows, champ_discard, champ_action = act(
            dict(hero_board), opp_board, dealt, [], 1, args.seat)

        records.append({
            "index": i,
            "hand_seed": seed,
            "seat": args.seat,
            "hero_board": hero_board,
            "opp_board": opp_board,
            "opp_discard_count": opp_discards,
            "hand": list(dealt),
            "champ_placements": [[c, r] for c, r in champ_action["placements"]],
            "champ_discard": champ_discard,
        })
        if i % 20 == 0:
            print(f"[{i:03d}] hero {sum(len(v) for v in hero_board.values())}c "
                  f"opp {sum(len(v) for v in opp_board.values())}c "
                  f"dealt {' '.join(dealt)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} T1-{args.seat} positions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
