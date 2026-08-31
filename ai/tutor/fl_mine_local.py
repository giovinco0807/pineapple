"""Perpetual local miner for hero's T0 when the opponent is in Fantasyland.

The T0-BB miner (t0_mine.py, t0_mine_local.py) referees the opening against a
normal opponent.  This is the other half of the serving surface: the same
mine -> correct-the-ordering loop, run against `own_lap4/t0.bin`'s argmax for
the hands where the opponent sits in Fantasyland.

Two differences from the BB miner, both forced by what serving actually does
on this path:

* **No fence.**  Against a face-down Fantasyland board there is no ranker and
  no policy, so the served opening is the chooser's outright argmax and
  `model_pick` is simply `own_rank` 1.  The candidate field is still wider
  than the pick (top-36 plus one key per top-row-count stratum), because
  nomination has to be able to reach past what the model already likes; one
  root in twenty races the full 232 to keep "is 36 wide enough" measured.
* **The score is hero's own board.**  Per the owner ruling of 2026-08-30 the
  Fantasyland opponent is ignored outright: a rollout is worth hero's
  royalties plus the entry it earns, minus six for a foul.  That is the
  objective `own_lap4` was trained under, so a disagreement here is the
  chain failing its own declared goal rather than failing a game it was
  never shown.
* **Frequency order, not uncertainty order.**  The root pool is the canonical
  1,000-class request file; there is no policy net here whose top1-top2 gap
  could order it, so roots are mined by how often they occur.  Class
  multiplicity is the closest thing to "where the referee-hours pay".

Nomination uses the corrected successive halving: the final means come from
`totals` over EVERY candidate at the running mean it was eliminated on, not
from the surviving field -- the mine4 bug that shipped 971 roots whose
orderings had collapsed to the last two survivors.  Disagreements are
re-scored on fresh seeds so the winner's curse stays out of the margins.

Appends to its own output file; safe to stop and restart (skips recorded ids).

Shardable: `--start`/`--count` cut a contiguous range out of the frequency
order, which every worker rebuilds identically from the same two files.  The
seed bands and the canary schedule key off the ABSOLUTE index in that order,
never off a position within the run -- so two shards can never draw the same
futures for different roots, and a root re-mined later re-mines rather than
resamples.  Defaults are the whole pool, which is what the perpetual local
miner runs.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
from pathlib import Path

D = Path("D:/ofc_data/hu")
REPO = Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")


def run_batch(ctx, cards, keys_file, rollouts, seed, out):
    own = ",".join(str(ctx.models / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    cmd = [str(ctx.binary), "--fl-t0-deep", "--t0-cards", cards,
           "--rollouts", str(rollouts), "--self-play-seed", str(seed),
           "--arm-a-own", own,
           "--fl-ev-config", str(ctx.fl_ev),
           "--output", str(out)]
    if keys_file is not None:
        cmd += ["--t0-keys", str(keys_file)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"batch failed seed {seed}: {done.stderr[-600:]}")
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]


def duel(ctx, cards, keys, rollouts, seeds, tag):
    kf = ctx.work / f"{tag}_keys.txt"
    kf.write_text("\n".join(keys) + "\n", encoding="utf-8")
    batches = []
    for s in seeds:
        rows = run_batch(ctx, cards, kf, rollouts, s, ctx.work / f"{tag}_s{s}.jsonl")
        batches.append({r["key"]: r["mean"] for r in rows})
    overall = {k: sum(b[k] for b in batches) / len(batches) for k in keys}
    return overall, batches


def select_race(ctx, cards, keys, seed_base, tag):
    """Elimination by gap-to-leader, not by fixed survivor counts.

    Owner's rule (2026-08-30): after each round a candidate is dropped only
    when its running mean trails the current leader by more than four PAIRED
    standard errors -- every rollout deals the same future to every candidate
    (CRN), and rounds concatenate, so the leader-vs-candidate differences are
    the measurement, not the raw spread.  Easy roots collapse in one cheap
    round; contested ones keep their rivals into the deep rounds.  4 sigma is
    the racing threshold the street-gate work validated (economical-SE rule).
    """
    field = list(keys)
    scores = {k: [] for k in field}
    for rnd, rollouts in enumerate(ctx.schedule):
        kf = ctx.work / f"{tag}_r{rnd}_keys.txt"
        kf.write_text(chr(10).join(field) + chr(10), encoding="utf-8")
        rows = run_batch(ctx, cards, kf, rollouts, seed_base + rnd,
                         ctx.work / f"{tag}_r{rnd}.jsonl")
        got = {r["key"]: r["scores"] for r in rows}
        for k in field:
            scores[k].extend(got[k])
        means = {k: sum(v) / len(v) for k, v in scores.items()}
        leader = max(field, key=lambda k: means[k])
        keep = [leader]
        for k in field:
            if k == leader:
                continue
            diffs = [a - b for a, b in zip(scores[leader], scores[k])]
            n = len(diffs)
            m = sum(diffs) / n
            sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(n - 1, 1))
            se = sd / math.sqrt(n)
            # Nomination only has to keep the true best alive -- the final
            # verdict re-measures on fresh seeds -- so the race can cut at
            # 3 paired sigmas, and anything 8+ points off the lead with 48+
            # cumulative rollouts is dead regardless: its exact rank is not
            # worth deep rounds (owner speed call, 2026-08-30).
            if m <= 3 * se and not (n >= 48 and m > 8.0):
                keep.append(k)
        field = keep
        if len(field) <= 1:
            break
    means = {k: sum(v) / len(v) for k, v in scores.items()}
    leader = max(field, key=lambda k: means[k])
    return leader, means


def frequency_order(requests, multiplicity):
    """The canonical classes, commonest first.

    Ties break on id so a resumed run reproduces the same order -- and with it
    the same seed band per root, which is what makes a re-run of a root a
    re-run rather than a fresh sample."""
    return sorted(requests, key=lambda r: (-multiplicity.get(r["id"], 0), r["id"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100000,
                    help="roots to process this run (a dry run wants 1)")
    ap.add_argument("--start", type=int, default=0,
                    help="first root index in the frequency order")
    ap.add_argument("--count", type=int, default=None,
                    help="how many indices of that order this shard owns")
    ap.add_argument("--requests", type=Path,
                    default=Path("D:/ofc_data/fl14_t0_requests_v1.jsonl"))
    ap.add_argument("--models-dir", type=Path,
                    default=D / "models_ship_20260830/own_lap4")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target_gate/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--work", type=Path, default=Path("C:/tmp/fl_mine_local"))
    ap.add_argument("--out", type=Path, default=D / "fl_mine1/material.jsonl")
    ap.add_argument("--score-rollouts", type=int, default=150)
    ap.add_argument("--topk", type=int, default=16,
                    help="how far past the pick nomination may reach")
    ap.add_argument("--sel-schedule", default="16,48,128",
                    help="halving rounds as rollouts:survivors")
    args = ap.parse_args()

    class Ctx: pass
    ctx = Ctx()
    ctx.binary, ctx.models, ctx.fl_ev = args.binary, args.models_dir, args.fl_ev_config
    ctx.work = args.work
    ctx.schedule = [int(x) for x in args.sel_schedule.split(",")]
    ctx.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    requests = [json.loads(l) for l in open(args.requests, encoding="utf-8") if l.strip()]
    mult_path = args.requests.with_suffix(".multiplicity.json")
    multiplicity = json.loads(mult_path.read_text(encoding="utf-8")) if mult_path.exists() else {}
    # Announced, because its absence is not an error: without it every class
    # weighs the same and the pool silently re-orders, which on a sharded run
    # means every worker mines roots other than the ones it was given.  A
    # shard log has to be able to say which order it used.
    print(f"multiplicity: {mult_path if multiplicity else 'MISSING -- id order'} "
          f"({len(multiplicity)} classes weighted)", flush=True)
    ordered = frequency_order(requests, multiplicity)

    done_ids = set()
    if args.out.exists():
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    print(f"pool: {len(ordered)} classes, {len(done_ids)} already recorded, "
          f"commonest first", flush=True)

    stop = len(ordered) if args.count is None else min(args.start + args.count, len(ordered))
    out = open(args.out, "a", encoding="utf-8")
    processed = 0
    for i in range(args.start, stop):
        if processed >= args.limit:
            break
        row = ordered[i]
        rid = row["id"]
        if rid in done_ids:
            continue
        cards = ",".join(row["cards"])
        try:
            rank = run_batch(ctx, cards, None, 0, 1, ctx.work / f"{rid}_rank.jsonl")
        except RuntimeError as e:
            print(f"[{i}] {rid} rank FAIL {e}", flush=True)
            continue
        rank.sort(key=lambda r: r["own_rank"])
        # No fence on this path: the served opening is the chooser's argmax.
        model_pick = rank[0]["key"]
        # Field policy (owner, 2026-08-30): net top-36 + strata for speed,
        # but every 20th root races the FULL field as a canary.  The canary
        # records where the winner sat in the net's own ranking, so "is 36
        # wide enough" is continuously measured instead of assumed -- the
        # top-16 field was rejected precisely because the audited net's
        # ranking is untrusted.
        # Keyed on the root's absolute index, not on a position within the
        # run: a restart, and a shard boundary, must not decide which roots
        # are canaries.  Counting recorded rows instead would make every one
        # of 63 shards open with a full-field root, and would drift whenever
        # a root failed and was skipped.
        exhaustive = (i % 20 == 0)
        if exhaustive:
            cands = [r["key"] for r in rank]
        else:
            strata = {}
            for r in rank:
                n_top = len([c for c in r["key"].split("|")[0].split(",") if c])
                strata.setdefault(n_top, r["key"])
            cands = list(dict.fromkeys(
                [r["key"] for r in rank if r["own_rank"] <= 36] + list(strata.values())))
        try:
            ref_pick, overall = select_race(
                ctx, cards, cands, 870_000_000 + i * 10, f"{rid}_sel")
        except RuntimeError as e:
            print(f"[{i}] {rid} sel FAIL {e}", flush=True)
            continue
        rank_of = {r["key"]: r["own_rank"] for r in rank}
        rec = {"id": rid, "cards": cards, "model_pick": model_pick,
               "ref_pick": ref_pick, "sel_means": overall,
               "serving_gen": "own_lap4_t0", "objective": "own_worth",
               "field": ("full" if exhaustive else "top36"),
               "winner_own_rank": rank_of.get(ref_pick)}
        if exhaustive and rank_of.get(ref_pick, 999) > 36:
            print(f"  !! CANARY: winner own_rank {rank_of.get(ref_pick)} > 36 on {rid}",
                  flush=True)
        if ref_pick == model_pick:
            rec.update(verdict="agree", margin=0.0)
        else:
            score_seeds = [880_000_000 + i * 10 + j for j in range(2)]
            try:
                _, sb = duel(ctx, cards, [model_pick, ref_pick], args.score_rollouts,
                             score_seeds, f"{rid}_score")
            except RuntimeError as e:
                print(f"[{i}] {rid} score FAIL {e}", flush=True)
                continue
            diffs = [b[ref_pick] - b[model_pick] for b in sb]
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
            se = sd / math.sqrt(len(diffs))
            lo, hi = m - 1.96 * se, m + 1.96 * se
            rec.update(verdict=("error" if lo > 0 else ("model_better" if hi < 0 else "undecided")),
                       margin=m, ci=[lo, hi])
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        processed += 1
        print(f"[{i}] {rid} {rec['verdict']} margin {rec.get('margin', 0):+.3f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
