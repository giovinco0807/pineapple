"""Ship gate: price 120 against the shipped 96 on classes neither has seen.

Everything measured so far lives on the 1,000 mined classes -- the correction
trained on 903 of them and was graded on the other 97.  A gate that decides
whether to replace `own_lap4/t0.bin` has to ask a different question: on
classes drawn from the same distribution but never mined, never trained on and
never held out, does serving the 120 net win points?

Per class:

  * rank every opening with both nets (`--rollouts 0`, which is the serving
    argmax and costs a fifth of a second);
  * if the two pick the same opening the EV difference is exactly zero and no
    rollouts are spent -- that is not an approximation, the same placement
    played by the same chain is the same hand;
  * if they differ, duel the two picks head to head under common random
    numbers, so the reported delta is a paired statistic and its error bar is
    the error bar of the difference rather than of two means.

Seed band 900,000,000 + fresh_index*10 + batch, where `fresh_index` is the
class's absolute index in the canonical ordering (1000..1299).  Unused
elsewhere: 850/860 are the miner's select and score, 870/880 the race, 890 the
holdout sharpening.  Keying on the absolute index rather than on a position
within a shard is what makes fleet output byte-compatible with a local run.

Checkpointed per class; a killed run resumes from the shards.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def rank_pick(binary: Path, own: str, fl_ev: Path, cards: str, out: Path) -> str:
    """The opening this chain would actually serve for these five cards."""
    done = subprocess.run(
        [str(binary), "--fl-t0-deep", "--t0-cards", cards, "--rollouts", "0",
         "--arm-a-own", own, "--fl-ev-config", str(fl_ev), "--output", str(out)],
        capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"rank failed for {cards}: {done.stderr[-400:]}")
    first = out.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first)["key"]


def duel(binary: Path, own: str, fl_ev: Path, cards: str, keys_file: Path,
         rollouts: int, seed: int, out: Path) -> dict:
    done = subprocess.run(
        [str(binary), "--fl-t0-deep", "--t0-cards", cards,
         "--rollouts", str(rollouts), "--self-play-seed", str(seed),
         "--arm-a-own", own, "--fl-ev-config", str(fl_ev),
         "--t0-keys", str(keys_file), "--output", str(out)],
        capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"duel seed {seed}: {done.stderr[-500:]}")
    return {r["key"]: r["scores"]
            for r in (json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requests", type=Path,
                    default=Path("D:/ofc_data/fl14_t0_requests_fresh300.jsonl"))
    ap.add_argument("--models-dir", type=Path,
                    default=Path("D:/ofc_data/hu/models_ship_20260830/own_lap4"))
    ap.add_argument("--challenger", type=Path,
                    default=Path("D:/ofc_data/hu/fl_evalfix/t0_120.bin"),
                    help="the T0 image under test; T1/T2 stay the shipped ones")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target_gate/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--out-dir", type=Path, default=Path("D:/ofc_data/hu/ship_gate"))
    ap.add_argument("--work", type=Path, default=Path("C:/tmp/ship_gate_work"))
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=256)
    ap.add_argument("--seed-base", type=int, default=900_000_000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--shard-jsonl", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.requests, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["index"])
    stop = len(rows) if args.count is None else min(args.start + args.count, len(rows))
    rows = rows[args.start:stop]
    shards = args.out_dir / "_gate"
    shards.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    base = ",".join(str(args.models_dir / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    # Only the T0 image changes: the gate prices the decision under test, not a
    # different chain.
    chal = ",".join([str(args.challenger)] + [str(args.models_dir / n) for n in ("t1.bin", "t2.bin")])
    started = time.time()
    print(f"{len(rows)} fresh classes [{args.start},{stop})", flush=True)

    def publish() -> None:
        if args.shard_jsonl is None:
            return
        with open(args.shard_jsonl, "w", encoding="utf-8") as handle:
            for r in rows:
                shard = shards / f"{r['id']}.json"
                if shard.exists():
                    handle.write(shard.read_text(encoding="utf-8").strip() + "\n")

    for position, row in enumerate(rows):
        shard = shards / f"{row['id']}.json"
        if shard.exists():
            continue
        cards = ",".join(row["cards"])
        p96 = rank_pick(args.binary, base, args.fl_ev_config, cards,
                        args.work / f"{row['id']}_96.jsonl")
        p120 = rank_pick(args.binary, chal, args.fl_ev_config, cards,
                         args.work / f"{row['id']}_120.jsonl")
        rec = {"id": row["id"], "index": row["index"], "cards": cards,
               "mult": row["multiplicity"], "pick96": p96, "pick120": p120,
               "agree": p96 == p120}
        if p96 == p120:
            # Same placement, same chain, same deals: the difference is zero by
            # identity, not by measurement.
            rec.update(delta=0.0, se=0.0, n=0)
        else:
            keys_file = args.work / f"{row['id']}_keys.txt"
            keys_file.write_text(f"{p96}\n{p120}\n", encoding="utf-8")
            a: list[float] = []
            b: list[float] = []
            for batch in range(args.batches):
                seed = args.seed_base + row["index"] * 10 + batch
                got = duel(args.binary, base, args.fl_ev_config, cards, keys_file,
                           args.rollouts, seed, args.work / f"{row['id']}_b{batch}.jsonl")
                a.extend(got[p96])
                b.extend(got[p120])
            diff = [y - x for x, y in zip(a, b)]
            n = len(diff)
            mean = sum(diff) / n
            var = sum((d - mean) ** 2 for d in diff) / max(n - 1, 1)
            rec.update(delta=mean, se=math.sqrt(var / n), n=n,
                       mean96=sum(a) / n, mean120=sum(b) / n)
        shard.write_text(json.dumps(rec), encoding="utf-8")
        publish()
        elapsed = time.time() - started
        rate = (position + 1) / max(elapsed, 1e-9)
        print(f"[{position + 1}/{len(rows)}] {row['id']} "
              + ("agree" if rec["agree"] else f"delta {rec['delta']:+.3f} +-{rec['se']:.3f}")
              + f" | eta {(len(rows) - position - 1) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    publish()
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
