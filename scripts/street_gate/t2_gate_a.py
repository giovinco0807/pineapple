"""T2 gate phase A: on-policy T2 positions for both seats, plus champion picks.

Same replay as the T1 version, one street deeper.  From T1 on, each seat
carries one private discard per street, so the record now includes the hero's
own discards (visible to the engine as `dead`) and the opponent's discard
count (their cards stay face down).
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

        # T0 and T1 for both seats, champion play, play_ai dealing order.
        p0, _, _ = act(dict(EMPTY), dict(EMPTY), deck[0:5], [], 0, "first")
        p1, _, _ = act(dict(EMPTY), p0, deck[5:10], [], 0, "second")
        p0, p0_d1, _ = act(p0, p1, deck[10:13], [], 1, "first")
        p1, p1_d1, _ = act(p1, p0, deck[13:16], [], 1, "second")
        p0_dead, p1_dead = [p0_d1], [p1_d1]

        if args.seat == "first":
            hero_board, hero_dead = p0, p0_dead
            opp_board, opp_discards = p1, 1
            dealt = deck[16:19]
        else:
            p0, p0_d2, _ = act(p0, p1, deck[16:19], p0_dead, 2, "first")
            p0_dead.append(p0_d2)
            hero_board, hero_dead = p1, p1_dead
            opp_board, opp_discards = p0, 2
            dealt = deck[19:22]

        champ_rows, champ_discard, champ_action = act(
            dict(hero_board), opp_board, dealt, hero_dead, 2, args.seat)

        records.append({
            "index": i,
            "hand_seed": seed,
            "street": "T2",
            "seat": args.seat,
            "hero_board": hero_board,
            "hero_discards": list(hero_dead),
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
    print(f"wrote {len(records)} T2-{args.seat} positions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
