"""Escalation re-audit: settle the ledger's undecided rows with 4x particles.

Every undecided verdict in the mining material was scored at 600 particles
(CI half-width ~1.8).  This re-duels the stored pair -- the serving pick and
the referee's challenger -- on EIGHT fresh seed batches of 300 deals each
(2,400 particles, half-width ~0.9), most-promising first (largest |margin|).
Rows are updated in place: verdict/margin/ci replaced, the 600-particle
score kept under `pre_escalation`, and `escalated: true` marks the row.
Settled errors become hinge material for the next policy generation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from ai.tutor.t0_mine_local import run_batch

D = Path("D:/ofc_data/hu")
REPO = Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--material", type=Path, default=D / "t0_mine1/material.jsonl")
    ap.add_argument("--min-margin", type=float, default=0.5)
    ap.add_argument("--rollouts", type=int, default=300)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--models-dir", type=Path, default=D / "models_ship_20260829")
    ap.add_argument("--policy-bin", type=Path, default=D / "models_ship_20260829/policy.bin")
    args = ap.parse_args()

    class Ctx: pass
    ctx = Ctx()
    ctx.binary = REPO / "ai/rust_solver/target/release/t4_first_exact.exe"
    ctx.models = args.models_dir
    ctx.policy = args.policy_bin
    ctx.work = Path("C:/tmp/t0_escalate")
    ctx.work.mkdir(exist_ok=True)
    ctx.joint, ctx.topk = 200, 8

    rows = [json.loads(l) for l in open(args.material, encoding="utf-8")]
    todo = sorted(
        (i for i, r in enumerate(rows)
         if r["verdict"] == "undecided" and not r.get("escalated")
         and abs(r.get("margin", 0.0)) >= args.min_margin),
        key=lambda i: -abs(rows[i].get("margin", 0.0)))
    print(f"escalation queue: {len(todo)} undecided rows (|margin| >= {args.min_margin})", flush=True)

    updated = {}

    def rewrite():
        # Merge by id against the CURRENT file contents: a whole-file dump
        # of this process's start-time snapshot silently deleted 1,079 rows
        # another process had appended after we loaded (2026-08-29 incident).
        disk = [json.loads(l) for l in open(args.material, encoding="utf-8")]
        tmp = args.material.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in disk:
                f.write(json.dumps(updated.get(r["id"], r), ensure_ascii=False) + "\n")
        tmp.replace(args.material)

    settled = 0
    for done, i in enumerate(todo):
        r = rows[i]
        keys = [r["model_pick"], r["ref_pick"]]
        kf = ctx.work / f"{r['id']}_keys.txt"
        kf.write_text("\n".join(keys) + "\n", encoding="utf-8")
        batches = []
        try:
            for j in range(args.batches):
                seed = 810_000_000 + done * 20 + j
                out = ctx.work / f"{r['id']}_e{seed}.jsonl"
                res = run_batch(ctx.binary, ctx.models, ctx.policy, r["cards"], kf,
                                args.rollouts, seed, out, ctx.joint, ctx.topk)
                batches.append({x["key"]: x["mean"] for x in res})
        except RuntimeError as e:
            print(f"[{done}] {r['id']} FAIL {e}", flush=True)
            continue
        diffs = [b[r["ref_pick"]] - b[r["model_pick"]] for b in batches]
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))
        se = sd / math.sqrt(len(diffs))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        verdict = "error" if lo > 0 else ("model_better" if hi < 0 else "undecided")
        r["pre_escalation"] = {"margin": r.get("margin"), "ci": r.get("ci")}
        r.update(escalated=True, verdict=verdict, margin=round(m, 4),
                 ci=[round(lo, 4), round(hi, 4)])
        updated[r["id"]] = r
        if verdict != "undecided":
            settled += 1
        if done % 5 == 4 or verdict != "undecided":
            rewrite()
        print(f"[{done}] {r['id']} {verdict} margin {m:+.3f} [{lo:+.2f},{hi:+.2f}]  "
              f"(settled {settled}/{done+1})", flush=True)
    rewrite()
    print(f"DONE: {settled}/{len(todo)} settled", flush=True)


if __name__ == "__main__":
    main()
