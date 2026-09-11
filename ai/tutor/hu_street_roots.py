"""Sample referee roots for the model-decided street slots from a champion trace.

A street root is one decision the champion actually faced: (hand, street,
seat) in a self-play trace, with everything before it fixed.  The deep-replay
referee (`--hu-deep-replay`) reconstructs the position from the trace line,
so a root record only names the hand and the slot; the hand line itself is
copied into the record so a fleet shard needs nothing but the roots file.

Slots the models decide: T1 and T2 for both seats, T3 for seat 0 (BB).
T3-BTN and T4 are exact and are not sampled.  Seat 0 = BB (acts first each
street), seat 1 = BTN.

Per slot the eval set mirrors the T0 design: 240 hands drawn at random and
240 where the hero holds a joker among the cards visible at that decision
(board + draw), no hand used twice within a slot.  Seed band 9.5e9
(registry: ... 9.1e9 tr3 trace, 9.3e9 fence duel, 9.4e9 BB deals).

    python -m ai.tutor.hu_street_roots --trace D:/ofc_data/hu/tr3_traces_20260904/0000.jsonl \
        --out-dir D:/ofc_data/hu/street_sharp --random 240 --joker 240

A teaching corpus for the same slot is drawn from the hands the eval set did
not use (`--exclude` the eval roots), under its own salt so the draw is not
the eval draw's continuation, and split fit / dev by stratum:

    python -m ai.tutor.hu_street_roots --trace ... --out-dir ... --slots 1:0 \
        --random 350 --joker 350 --dev 50 --exclude roots_t1s0.jsonl --salt 1 --suffix _pilot
    -> roots_t1s0_pilot.jsonl (300 + 300) and roots_t1s0_pilot_dev.jsonl (50 + 50)

Rows: {"id": "tr3/0000/<hand>/t1s0", "hand_index": <line no>, "hand": <hand>, "street": 1,
       "seat": 0, "stratum": "rand"|"joker", "served": "top|mid|bot", "draw": [...],
       "trace": <trace json line as an object>}
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

BAND = 9_500_000_001
SLOTS = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]


def key_of(rows) -> str:
    return "|".join(",".join(sorted(r)) for r in rows)


def visible_cards(step) -> list[str]:
    return [c for row in step["board"] for c in row] + list(step["draw"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--random", type=int, default=240)
    ap.add_argument("--joker", type=int, default=240)
    ap.add_argument("--slots", default="1:0,1:1,2:0,2:1,3:0")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="roots files whose hand_index must not be drawn again (the eval set)")
    ap.add_argument("--salt", type=int, default=0,
                    help="added to the slot's seed; 0 is the eval draw, a corpus takes its own")
    ap.add_argument("--dev", type=int, default=0,
                    help="per stratum, the last N drawn go to a separate *_dev file")
    ap.add_argument("--suffix", default="", help="appended to the output name before .jsonl")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slots = [tuple(int(x) for x in s.split(":")) for s in args.slots.split(",")]
    hands = [json.loads(l) for l in args.trace.open(encoding="utf-8") if l.strip()]
    excluded: set[int] = set()
    for path in args.exclude:
        for line in path.open(encoding="utf-8"):
            if line.strip():
                excluded.add(int(json.loads(line)["hand_index"]))
    for street, seat in slots:
        rng = random.Random(BAND + street * 10 + seat + 1_000 * args.salt)
        cands = []
        for index, hand in enumerate(hands):
            if index in excluded:
                continue
            step = next((s for s in hand["steps"] if s["street"] == street and s["seat"] == seat), None)
            if step is None or step.get("by") != "model":
                continue
            joker = any(c.startswith("X") for c in visible_cards(step))
            cands.append((index, hand, step, joker))
        rand_pool = list(cands)
        rng.shuffle(rand_pool)
        chosen = []
        used = set()
        for index, hand, step, joker in rand_pool:
            if len(chosen) >= args.random:
                break
            chosen.append((index, hand, step, "rand"))
            used.add(index)
        joker_pool = [c for c in cands if c[3] and c[0] not in used]
        rng.shuffle(joker_pool)
        for index, hand, step, joker in joker_pool[:args.joker]:
            chosen.append((index, hand, step, "joker"))
        splits = {"": chosen}
        if args.dev:
            rand_rows = [c for c in chosen if c[3] == "rand"]
            joker_rows = [c for c in chosen if c[3] == "joker"]
            splits = {"": rand_rows[:-args.dev] + joker_rows[:-args.dev],
                      "_dev": rand_rows[-args.dev:] + joker_rows[-args.dev:]}
        for split, rows in splits.items():
            out = args.out_dir / f"roots_t{street}s{seat}{args.suffix}{split}.jsonl"
            with out.open("w", encoding="utf-8", newline="\n") as f:
                for index, hand, step, stratum in rows:
                    f.write(json.dumps({
                        "id": f"tr3/0000/{hand['hand']}/t{street}s{seat}",
                        "hand_index": index, "hand": hand["hand"], "street": street, "seat": seat,
                        "stratum": stratum, "served": key_of(step["board"]), "draw": step["draw"],
                        "trace": hand,
                    }) + "\n")
            print(f"T{street} seat{seat}{split}: {len(rows)} roots "
                  f"({sum(1 for c in rows if c[3] == 'joker')} joker) from {len(cands)} model-decided hands "
                  f"({len(excluded)} excluded) -> {out}")


if __name__ == "__main__":
    main()
