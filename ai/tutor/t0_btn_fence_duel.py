"""Price the fence widths the mining field never covered, one duel per root.

The mined candidate field was the ranker's top-16, so `--hu-t0-btn-topk` can be
swept offline up to 16 and no further: past that the evaluator's fenced argmax
is an opening no referee ever played.  Racing a fresh field per root would pay
for all ~18 candidates again; this pays for two.

Per root the only question is whether the opening serving picks at the WIDE
fence beats the one it picks at the NARROW fence.  Roots where the two agree
contribute exactly zero to the difference of means and carry no noise, so the
precision of the whole estimate comes from the roots that actually move -- 48
of 955 at K=24, 86 at no fence at all.  Both openings are played out on the
same rollout futures (common random numbers) and the paired per-batch
difference is the statistic, exactly as `t0_mine` scores its disagreements.

Seed band 9.3e9 (registry: 220M/310M/600M/700M/810M/850M/860M/880M/900M/
115-139M/2.1e9/3.3e9/4.4e9/5.1e9/5.2e9/6.1e9/6.6e9/7.7e9/8.9e9/9.1e9).

    python -m ai.tutor.t0_btn_fence_duel --pairs pairs.jsonl --start 0 --count 10 \
        --models-dir D:/ofc_data/hu/models_ship_20260904 --out duels.jsonl

`--pairs` rows are {"id", "index", "cards", "opp_board", "keys": {K: key}}:
the fenced argmax at each width K, deduplicated.  A root contributes one race
over its DISTINCT picks, so every width is priced against every other on the
same futures and the curve past 16 is read off one run.
"""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path

from ai.tutor.t0_mine import run_batch, t_crit

BAND = 9_300_000_001


def duel_root(binary, models, fl_ev, work, row, rollouts, batches, tag):
    keys = sorted(set(row["keys"].values()))
    kf = work / f"{tag}_keys.txt"
    kf.write_text("\n".join(keys) + "\n", encoding="utf-8")
    per = {k: [] for k in keys}
    for b in range(batches):
        seed = BAND + row["index"] * 10 + b
        rows = run_batch(binary, models, fl_ev, row["cards"], kf, rollouts, seed,
                         work / f"{tag}_b{b}.jsonl", seat=1, opp_board=row["opp_board"])
        got = {r["key"]: r["mean"] for r in rows}
        for k in keys:
            if k not in got:
                raise RuntimeError(f"{row['id']}: batch {b} lost {k}")
            per[k].append(got[k])
    base = row["keys"]["16"]
    out = {"id": row["id"], "index": row["index"], "cards": row["cards"],
           "opp_board": row["opp_board"], "keys": row["keys"],
           "batches": batches, "rollouts": rollouts,
           "mean": {k: sum(v) / len(v) for k, v in per.items()}, "vs16": {}}
    for K, key in row["keys"].items():
        diffs = [w - n for n, w in zip(per[base], per[key])]
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
        out["vs16"][K] = {"mean": m, "se": sd / math.sqrt(len(diffs))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10**9)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--binary", type=Path,
                    default=Path(__file__).resolve().parents[2] / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "ai/config/fl_ev.json")
    ap.add_argument("--work", type=Path, default=Path("C:/tmp/fence_duel"))
    ap.add_argument("--rollouts", type=int, default=150)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    rows = rows[args.start:args.start + args.count]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    with args.out.open("a", encoding="utf-8", newline="\n") as sink:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            rec = duel_root(args.binary, args.models_dir, args.fl_ev_config, args.work,
                            row, args.rollouts, args.batches, f"d{args.start + i}")
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            gain = rec["vs16"].get("232", {}).get("mean", 0.0)
            print(f"{rec['id']}  open-vs-16 {gain:+.2f}  "
                  f"({args.start + i + 1}/{len(rows)})", flush=True)


if __name__ == "__main__":
    main()
