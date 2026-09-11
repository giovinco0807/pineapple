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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--random", type=int, default=240)
    ap.add_argument("--joker", type=int, default=240)
    ap.add_argument("--slots", default="1:0,1:1,2:0,2:1,3:0")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slots = [tuple(int(x) for x in s.split(":")) for s in args.slots.split(",")]
    hands = [json.loads(l) for l in args.trace.open(encoding="utf-8") if l.strip()]
    trace_name = args.trace.parent.name.replace("_traces_", "/")  # tr3/20260904 style tag
    for street, seat in slots:
        rng = random.Random(BAND + street * 10 + seat)
        cands = []
        for index, hand in enumerate(hands):
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
        out = args.out_dir / f"roots_t{street}s{seat}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for n, (index, hand, step, stratum) in enumerate(chosen):
                f.write(json.dumps({
                    "id": f"tr3/0000/{hand['hand']}/t{street}s{seat}",
                    "hand_index": index, "hand": hand["hand"], "street": street, "seat": seat,
                    "stratum": stratum, "served": key_of(step["board"]), "draw": step["draw"],
                    "trace": hand,
                }) + "\n")
        print(f"T{street} seat{seat}: {len(chosen)} roots ({sum(1 for c in chosen if c[3]=='joker')} joker) "
              f"from {len(cands)} model-decided hands -> {out}")


if __name__ == "__main__":
    main()
