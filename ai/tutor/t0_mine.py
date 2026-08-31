"""T0-BB mining worker: referee the serving pick on a slice of roots.

Portable version of the Phase-0 v3 driver (scratchpad/t0_phase0.py) for
fleet workers: every path arrives as an argument, the binary is the staged
Linux build, and the output is one JSONL of verdict records per shard.

Per root: the serving cascade ranks all 232 openings (--rollouts 0, which
reports evaluator score AND ranker rank); the decision under test is the
SERVING pick (evaluator argmax inside the ranker's top-K fence).  The
referee duels a fence-aware candidate set (the fence itself plus the
ranker's best key per top-row-count stratum) on SELECT seeds, and any
disagreement is re-scored on FRESH seeds so the winner's-curse bias stays
out of the margins.  sel_means keeps the full candidate ordering -- the
teaching material for a future policy net, not just the verdict.
"""
from __future__ import annotations
import argparse, json, math, random, subprocess
from pathlib import Path


def run_batch(binary, models_dir, fl_ev, cards, keys_file, rollouts, seed, out, joint=200, topk=4):
    names = ("t0_bb.bin", "t0_btn.bin", "t1_bb.bin", "t1_btn.bin",
             "t2_bb.bin", "t2_btn.bin", "t3_bb.bin", "t3_btn.bin")
    hu = ",".join(str(models_dir / "hu" / n) for n in names)
    rk = ",".join(str(models_dir / "rankers" / n) for n in names)
    own = ",".join(str(models_dir / "own_lap4" / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    cmd = [str(binary), "--hu-match", "--hu-t0-deep", "--t0-cards", cards,
           "--rollouts", str(rollouts), "--self-play-seed", str(seed),
           "--hu-a-models", hu, "--hu-b-models", hu,
           "--hu-a-rankers", rk, "--hu-b-rankers", rk,
           "--hu-topk", str(topk), "--serve-joint-samples", str(joint),
           "--serve-joint-samples-b", str(joint),
           "--arm-a-own", own, "--arm-b-own", own,
           "--fl-ev-config", str(fl_ev), "--output", str(out)]
    # If the staged bundle carries a policy net, serve T0-BB through it (the
    # shipped 20260829 convention); the referee then audits the CURRENT
    # champion instead of the retired ranker fence.
    policy = models_dir / "policy.bin"
    if policy.exists():
        cmd += ["--hu-a-t0-policy", str(policy), "--hu-b-t0-policy", str(policy),
                "--hu-t0-policy-topk", "8"]
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
        rows = run_batch(ctx.binary, ctx.models, ctx.fl_ev, cards, kf, rollouts, s,
                         ctx.work / f"{tag}_s{s}.jsonl", ctx.joint, ctx.topk)
        batches.append({r["key"]: r["mean"] for r in rows})
    overall = {k: sum(b[k] for b in batches) / len(batches) for k in keys}
    return overall, batches


def select_halving(ctx, cards, keys, seed_base, tag):
    """Successive-elimination nomination: wide field, particles follow promise.

    Rounds of (particles, survivors): all@64 -> 8@128 -> 4@256 -> 2@384.
    Every round is one CRN batch (all survivors share its deals) on a fresh
    seed; the running per-candidate mean decides who stays -- the best-arm
    shape the regular track's elimination runner validated.  Same order of
    total particles as the old flat select, but the finalists carry ~4x the
    evidence, which is what kills the winner's curse at nomination time."""
    field = list(keys)
    totals = {k: [0.0, 0] for k in field}
    means = {}
    for rnd, (rollouts, keep) in enumerate([(64, 8), (128, 4), (256, 2), (384, 1)]):
        overall, _ = duel(ctx, cards, field, rollouts, [seed_base + rnd], f"{tag}_r{rnd}")
        for k in field:
            totals[k][0] += overall[k] * rollouts
            totals[k][1] += rollouts
        means = {k: v[0] / v[1] for k, v in totals.items() if v[1]}
        field = sorted(field, key=lambda k: -means[k])[:keep]
        if len(field) <= 1:
            break
    # means over EVERY candidate (running mean at elimination time), not just
    # the finalists: mine4 shipped 971 roots whose orderings had shrunk to the
    # last two survivors because this dict was rebuilt per-round over `field`.
    return field[0], means


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--fl-ev-config", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="matches the local Phase-0 order so slices never overlap")
    ap.add_argument("--sel-rollouts", type=int, default=96)
    ap.add_argument("--score-rollouts", type=int, default=150)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--joint", type=int, default=200)
    args = ap.parse_args()

    roots = [json.loads(l) for l in open(args.requests, encoding="utf-8") if l.strip()]
    # shuffle-seed 0 preserves the file's own order (frequency-ranked pools);
    # nonzero reproduces the legacy shuffled slicing (mine1-3 conventions).
    if args.shuffle_seed:
        random.Random(args.shuffle_seed).shuffle(roots)
    args.work.mkdir(parents=True, exist_ok=True)

    class Ctx: pass
    ctx = Ctx()
    ctx.binary, ctx.models, ctx.fl_ev = args.binary, args.models_dir, args.fl_ev_config
    ctx.work, ctx.joint, ctx.topk = args.work, args.joint, args.topk

    done_ids = set()
    if args.out.exists():
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                done_ids.add(json.loads(l)["id"])
        print(f"resume: {len(done_ids)} roots already recorded", flush=True)
    out = open(args.out, "a", encoding="utf-8")
    for i in range(args.start, min(args.start + args.count, len(roots))):
        row = roots[i]
        if row["id"] in done_ids:
            continue
        cards = ",".join(row["draw"])
        rid = row["id"]
        rank = run_batch(ctx.binary, ctx.models, ctx.fl_ev, cards, None, 0, 1,
                         ctx.work / f"{rid}_rank.jsonl", ctx.joint, ctx.topk)
        ev = lambda r: r["score"]
        pol = (args.models_dir / "policy.bin").exists()
        fence_key, fence_k = ("policy_rank", 8) if pol else ("ranker_rank", args.topk)
        # The serving pick mirrors the REAL serve (fence of 8); the referee's
        # candidate field is wider (top-16 + shape strata) so nomination can
        # reach past the fence -- the owner's top-16 directive.
        fenced = [r for r in rank if r.get(fence_key) is not None and r[fence_key] <= fence_k]
        wide = [r for r in rank if r.get(fence_key) is not None and r[fence_key] <= 16]
        if fenced:
            model_pick = max(fenced, key=ev)["key"]
            shortlist = [r["key"] for r in sorted(wide, key=lambda r: r[fence_key])]
        else:
            model_pick = max(rank, key=ev)["key"]
            shortlist = [model_pick]
        strata = {}
        for r in sorted(rank, key=lambda r: (r.get(fence_key) or 999)):
            n_top = len([c for c in r["key"].split("|")[0].split(",") if c])
            strata.setdefault(n_top, r["key"])
        cands = list(dict.fromkeys(shortlist + list(strata.values())))
        ref_pick, overall = select_halving(ctx, cards, cands, 220_000_000 + i * 10, f"{rid}_sel")
        rec = {"id": rid, "cards": cards, "model_pick": model_pick,
               "ref_pick": ref_pick, "sel_means": overall}
        if ref_pick == model_pick:
            rec.update(verdict="agree", margin=0.0)
        else:
            score_seeds = [310_000_000 + i * 10 + j for j in range(4)]
            so, sb = duel(ctx, cards, [model_pick, ref_pick], args.score_rollouts,
                          score_seeds, f"{rid}_score")
            diffs = [b[ref_pick] - b[model_pick] for b in sb]
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
            se = sd / math.sqrt(len(diffs))
            lo, hi = m - 1.96 * se, m + 1.96 * se
            rec.update(verdict=("error" if lo > 0 else ("model_better" if hi < 0 else "undecided")),
                       margin=m, ci=[lo, hi])
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{i}] {rid} {rec['verdict']} margin {rec.get('margin', 0):+.3f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
