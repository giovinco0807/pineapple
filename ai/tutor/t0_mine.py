"""T0 mining worker: referee the serving pick on a slice of roots (BB or BTN).

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

Seats (--seat):
  bb  (default) requests rows are {"id", "draw": [5 cards]}; the pick under
      test is the audit's own fence argmax (policy fence of 8 when the bundle
      carries policy.bin, else the ranker fence of --topk).
  btn requests rows are the roots_all.jsonl format lifted from champion
      traces: {"id", "btn_cards": [5], "bb_board": "top|mid|bot", "served":
      "top|mid|bot"}.  The pick under test is the SERVED move from the record,
      never the audit's argmax: serving at BTN depends on the joint-sample
      stream and the audit's own "served" matches real play only 39/50.
      policy.bin is a BB-only net and is not applied at BTN (the Rust side
      leaves policy_rank null), so the fence is always the ranker's.
      Continuations are the champion's; --hu-fast-nets is never passed.

Seed bands (registry: 220M/310M/600M/700M/810M/850M/860M/880M/900M/
115-122M/2.1e9/3.3e9/4.4e9/6.6e9/7.7e9/8.888e9-8.903e9):
  seat  selection (per root i, round rnd)     scoring (per root i, batch j)
  bb    220_000_000 + i*10 + rnd              310_000_000 + i*10 + j
  btn   5_100_000_001 + i*10 + rnd            5_200_000_001 + i*10 + j

Verdict CI: paired per-batch differences, mean +- t_{0.975, n-1} * SE
(2.365 at the 8-batch BTN default, 3.182 at 4).  --legacy-z restores the
old fixed 1.96 for comparability with material mined before this change.
"""
from __future__ import annotations
import argparse, json, math, random, subprocess
from pathlib import Path

SEED_BANDS = {  # (selection base, scoring base) -- see the docstring table
    "bb": (220_000_000, 310_000_000),
    "btn": (5_100_000_001, 5_200_000_001),
}


# t_{0.975, df} for df = 1..30; fleet workers (Debian 12) carry no scipy.
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_crit(n, legacy_z=False):
    """Two-sided 97.5% quantile for n paired batches (n-1 degrees of freedom)."""
    if legacy_z:
        return 1.96
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except ImportError:
        return T975[min(n - 1, len(T975)) - 1] if n > 1 else float("inf")


def joker_blind(key):
    """X1 and X2 are the same card: the audit renumbers hero's jokers by
    position, the trace spelled them by deal order."""
    return key.replace("X1", "X").replace("X2", "X")


def run_batch(binary, models_dir, fl_ev, cards, keys_file, rollouts, seed, out, joint=200, topk=4,
              seat=0, opp_board=None):
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
    # BTN: the decision state is the pair (hero's five, BB's placed board).
    # Seat 0 passes nothing so the BB command line stays byte-identical.
    if seat == 1:
        cmd += ["--hu-t0-seat", "1", "--t0-opp-board", opp_board]
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
                         ctx.work / f"{tag}_s{s}.jsonl", ctx.joint, ctx.topk,
                         ctx.seat, ctx.opp_board)
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
    ap.add_argument("--seat", choices=("bb", "btn"), default="bb",
                    help="bb: audit fence argmax on {id,draw} rows; "
                         "btn: served move on roots_all.jsonl rows")
    ap.add_argument("--verdict-batches", type=int, default=None,
                    help="paired scoring batches on disagreement (default 4 at bb, 8 at btn)")
    ap.add_argument("--legacy-z", action="store_true",
                    help="CI half-width 1.96*SE as before 2026-09-02 instead of the t quantile")
    args = ap.parse_args()
    if args.verdict_batches is None:
        args.verdict_batches = 8 if args.seat == "btn" else 4
    seat = 1 if args.seat == "btn" else 0
    sel_base, score_base = SEED_BANDS[args.seat]
    crit = t_crit(args.verdict_batches, args.legacy_z)

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
    ctx.seat, ctx.opp_board = seat, None

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
        cards = ",".join(row["btn_cards"] if seat == 1 else row["draw"])
        rid = row["id"]
        # The id carries a '/' at BTN (trace/hand); keep work files flat.
        wid = rid.replace("/", "_")
        ctx.opp_board = row["bb_board"] if seat == 1 else None
        rank = run_batch(ctx.binary, ctx.models, ctx.fl_ev, cards, None, 0, 1,
                         ctx.work / f"{wid}_rank.jsonl", ctx.joint, ctx.topk,
                         ctx.seat, ctx.opp_board)
        ev = lambda r: r["score"]
        # policy.bin fences the real BB serve (K=8); at BTN it is not applied
        # (Rust leaves policy_rank null) and the ranker fence selects.
        pol = seat == 0 and (args.models_dir / "policy.bin").exists()
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
        extra = {}
        if seat == 1:
            # The decision under test is what the champion actually played
            # (root record), spelled the way the audit spells it (jokers are
            # renumbered by position at BTN), and forced into the field.
            want = joker_blind(row["served"])
            hit = [r for r in rank if joker_blind(r["key"]) == want]
            if not hit:
                raise RuntimeError(f"{rid}: served move {row['served']} not among the audit's openings")
            model_pick = hit[0]["key"]
            extra = {"seat": "btn", "opp_board": ctx.opp_board,
                     "served_model_rank": hit[0]["rank"],
                     "served_ranker_rank": hit[0].get("ranker_rank")}
            shortlist = shortlist + [model_pick]
        cands = list(dict.fromkeys(shortlist + list(strata.values())))
        ref_pick, overall = select_halving(ctx, cards, cands, sel_base + i * 10, f"{wid}_sel")
        rec = {"id": rid, "cards": cards, "model_pick": model_pick,
               "ref_pick": ref_pick, "sel_means": overall, **extra}
        if ref_pick == model_pick:
            rec.update(verdict="agree", margin=0.0)
        else:
            score_seeds = [score_base + i * 10 + j for j in range(args.verdict_batches)]
            so, sb = duel(ctx, cards, [model_pick, ref_pick], args.score_rollouts,
                          score_seeds, f"{wid}_score")
            diffs = [b[ref_pick] - b[model_pick] for b in sb]
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
            se = sd / math.sqrt(len(diffs))
            lo, hi = m - crit * se, m + crit * se
            rec.update(verdict=("error" if lo > 0 else ("model_better" if hi < 0 else "undecided")),
                       margin=m, ci=[lo, hi], batches=len(diffs), t_crit=crit)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{i}] {rid} {rec['verdict']} margin {rec.get('margin', 0):+.3f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
