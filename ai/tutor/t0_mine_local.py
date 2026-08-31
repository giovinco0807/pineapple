"""Perpetual local T0-BB miner: referee the SHIPPED policy serving, hardest first.

The fleet miners refereed the old ranker-fence serving; this grinds against
the current champion (policy v3b top-8 -> evaluator) so every verdict is a
residual error of what actually ships.  Roots are ordered by the policy's
own uncertainty (smallest top1-top2 probability gap first) over the unmined
remainder of the lap-1 request pool, so each referee-hour is spent where the
model is least sure.  Appends to its own output file; safe to stop and
restart (skips recorded ids).
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

from ai.tutor.train_t0_policy import Policy, canonical, feats, action_index

D = Path("D:/ofc_data/hu")
REPO = Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")


def run_batch(binary, models_dir, policy_bin, cards, keys_file, rollouts, seed, out,
              joint=200, topk=8):
    import subprocess
    names = ("t0_bb.bin", "t0_btn.bin", "t1_bb.bin", "t1_btn.bin",
             "t2_bb.bin", "t2_btn.bin", "t3_bb.bin", "t3_btn.bin")
    hu = ",".join(str(models_dir / "hu" / n) for n in names)
    rk = ",".join(str(models_dir / "rankers" / n) for n in names)
    own = ",".join(str(models_dir / "own_lap4" / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    cmd = [str(binary), "--hu-match", "--hu-t0-deep", "--t0-cards", cards,
           "--rollouts", str(rollouts), "--self-play-seed", str(seed),
           "--hu-a-models", hu, "--hu-b-models", hu,
           "--hu-a-rankers", rk, "--hu-b-rankers", rk,
           "--hu-topk", "4",
           "--hu-a-t0-policy", str(policy_bin), "--hu-b-t0-policy", str(policy_bin),
           "--hu-t0-policy-topk", str(topk),
           "--serve-joint-samples", str(joint), "--serve-joint-samples-b", str(joint),
           "--arm-a-own", own, "--arm-b-own", own,
           "--fl-ev-config", str(REPO / "ai/config/fl_ev.json"),
           "--output", str(out)]
    if keys_file is not None:
        cmd += ["--t0-keys", str(keys_file)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"batch failed seed {seed}: {done.stderr[-500:]}")
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]


def duel(ctx, cards, keys, rollouts, seeds, tag):
    kf = ctx.work / f"{tag}_keys.txt"
    kf.write_text("\n".join(keys) + "\n", encoding="utf-8")
    batches = []
    for s in seeds:
        rows = run_batch(ctx.binary, ctx.models, ctx.policy, cards, kf, rollouts, s,
                         ctx.work / f"{tag}_s{s}.jsonl", ctx.joint, ctx.topk)
        batches.append({r["key"]: r["mean"] for r in rows})
    overall = {k: sum(b[k] for b in batches) / len(batches) for k in keys}
    return overall, batches


def uncertainty_order(policy_pt, requests, exclude_ids):
    net = Policy()
    blob = torch.load(policy_pt, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model_state_dict"])
    net.eval()
    scored = []
    with torch.no_grad():
        for r in requests:
            if r["id"] in exclude_ids:
                continue
            canon, _ = canonical(r["draw"])
            probs = torch.softmax(net(torch.tensor(feats(canon)).unsqueeze(0))[0], -1)
            top2 = torch.topk(probs, 2).values
            scored.append((float(top2[0] - top2[1]), r))
    scored.sort(key=lambda t: t[0])
    return [r for _, r in scored]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--policy-bin", type=Path,
                    default=D / "models_ship_20260829/policy.bin")
    ap.add_argument("--policy-pt", type=Path,
                    default=D / "t0_policy_v3b/policy_best.pt")
    ap.add_argument("--models-dir", type=Path, default=D / "models_ship_20260829")
    ap.add_argument("--out", type=Path, default=D / "t0_mine1/material_local.jsonl")
    ap.add_argument("--sel-rollouts", type=int, default=96)
    ap.add_argument("--score-rollouts", type=int, default=150)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    class Ctx: pass
    ctx = Ctx()
    ctx.binary = REPO / "ai/rust_solver/target/release/t4_first_exact.exe"
    ctx.models = args.models_dir
    ctx.policy = args.policy_bin
    ctx.work = Path("C:/tmp/t0_mine_local")
    ctx.work.mkdir(exist_ok=True)
    ctx.joint, ctx.topk = 200, args.topk

    # Root pool: the lap-1 requests not yet in any material file (fleet mining
    # used shuffled indices 2..1991 of the same file; ids are authoritative).
    mined = set()
    for f in (D / "t0_mine1/material.jsonl", args.out):
        if f.exists():
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    mined.add(json.loads(l)["id"])
    requests = [json.loads(l) for l in open(D / "onpol_requests/t0_bb_onpol.jsonl", encoding="utf-8")]
    ordered = uncertainty_order(args.policy_pt, requests, mined)
    print(f"pool: {len(ordered)} unmined roots, hardest-first", flush=True)

    out = open(args.out, "a", encoding="utf-8")
    for done, row in enumerate(ordered[:args.n]):
        cards = ",".join(row["draw"])
        rid = row["id"]
        try:
            rank = run_batch(ctx.binary, ctx.models, ctx.policy, cards, None, 0, 1,
                             ctx.work / f"{rid}_rank.jsonl", ctx.joint, ctx.topk)
        except RuntimeError as e:
            print(f"[{done}] {rid} rank FAIL {e}", flush=True)
            continue
        canon, mp = canonical(row["draw"])
        ev = lambda r: r["score"]
        fenced = [r for r in rank if r.get("policy_rank") is not None and r["policy_rank"] <= args.topk]
        if not fenced:
            print(f"[{done}] {rid} no policy fence -- check flags", flush=True)
            break
        model_pick = max(fenced, key=ev)["key"]
        shortlist = [r["key"] for r in sorted(fenced, key=lambda r: r["policy_rank"])]
        strata = {}
        for r in sorted(rank, key=lambda r: (r.get("policy_rank") or 999)):
            n_top = len([c for c in r["key"].split("|")[0].split(",") if c])
            strata.setdefault(n_top, r["key"])
        cands = list(dict.fromkeys(shortlist + list(strata.values())))
        sel_seeds = [600_000_000 + done * 10 + j for j in range(2)]
        try:
            overall, _ = duel(ctx, cards, cands, args.sel_rollouts, sel_seeds, f"{rid}_sel")
        except RuntimeError as e:
            print(f"[{done}] {rid} sel FAIL {e}", flush=True)
            continue
        ref_pick = max(overall, key=lambda k: overall[k])
        rec = {"id": rid, "cards": cards, "model_pick": model_pick,
               "ref_pick": ref_pick, "sel_means": overall, "serving_gen": "v3b_top8"}
        if ref_pick == model_pick:
            rec.update(verdict="agree", margin=0.0)
        else:
            score_seeds = [700_000_000 + done * 10 + j for j in range(4)]
            try:
                so, sb = duel(ctx, cards, [model_pick, ref_pick], args.score_rollouts,
                              score_seeds, f"{rid}_score")
            except RuntimeError as e:
                print(f"[{done}] {rid} score FAIL {e}", flush=True)
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
        print(f"[{done}] {rid} {rec['verdict']} margin {rec.get('margin', 0):+.3f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
