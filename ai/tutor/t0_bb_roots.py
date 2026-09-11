"""Sample T0-BB roots: a Big-Blind opening is decided by hero's five alone.

BB acts first at street 0 and sees nothing (docs/t0_bb_opponent_block_20260824.md):
its own board, discards and the opponent's board are all empty, so the
on-policy root distribution is the uniform five-card deal from the 54-card
deck (52 + two jokers).  No trace is needed.  P(at least one joker) = 17.7%.

Seed band 9.4e9 (registry: 220M/310M/600M/700M/810M/850M/860M/880M/900M/
115-139M/2.1e9/3.3e9/4.4e9/5.1e9/5.2e9/6.1e9/6.6e9/7.7e9/8.9e9/9.1e9/9.3e9);
Python's random, not the referee's, but registered so the band stays unique.

    python -m ai.tutor.t0_bb_roots --fit 2000 --eval-random 240 --eval-joker 240 \
        --out-dir D:/ofc_data/hu/t0bb_sharp

Rows: {"id": "bb/fit/0007", "draw": ["Ah","Kd","7c","7s","2h"], "stratum": "rand"|"joker"}
-- the `draw` list is `t0_mine --seat bb`'s request format.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

BAND = 9_400_000_001
RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [r + s for r in RANKS for s in SUITS] + ["X1", "X2"]


def deal(rng: random.Random) -> list[str]:
    return rng.sample(DECK, 5)


def has_joker(cards) -> bool:
    return any(c.startswith("X") for c in cards)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=int, default=2000)
    ap.add_argument("--eval-random", type=int, default=240)
    ap.add_argument("--eval-joker", type=int, default=240)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[frozenset] = set()

    def fresh(rng, want_joker=None):
        while True:
            d = deal(rng)
            if want_joker is not None and has_joker(d) != want_joker:
                continue
            k = frozenset(d)
            if k in seen:
                continue
            seen.add(k)
            return d

    rng = random.Random(BAND)
    fit = [fresh(rng) for _ in range(args.fit)]
    with (args.out_dir / "roots_fit.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, d in enumerate(fit):
            f.write(json.dumps({"id": f"bb/fit/{i:04d}", "draw": d,
                                "stratum": "joker" if has_joker(d) else "rand"}) + "\n")
    rng = random.Random(BAND + 1)
    ev = [(fresh(rng, False), "rand") for _ in range(args.eval_random)]
    ev += [(fresh(rng, True), "joker") for _ in range(args.eval_joker)]
    with (args.out_dir / "roots_eval.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, (d, s) in enumerate(ev):
            f.write(json.dumps({"id": f"bb/eval/{i:04d}", "draw": d, "stratum": s}) + "\n")
    print(f"fit {len(fit)} (joker {sum(has_joker(d) for d in fit)}), eval {len(ev)} "
          f"(random {args.eval_random}, joker {args.eval_joker}); no overlap")


if __name__ == "__main__":
    main()
