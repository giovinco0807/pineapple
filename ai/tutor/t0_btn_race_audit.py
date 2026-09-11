"""Score Button T0 evaluators against the raced (ultra-precise) referee.

`t0_btn_label --race` priced the ranker's top-16 field on 480 btnmine4 roots
with adaptive racing (per-value SE recorded).  This reads each model's
fenced argmax (ranker_rank <= K, encodings from `encode_t0_material`) and
prices it in the race's own units:

  * regret      -- mean over roots of max(field) - field[pick], with the
                   measurement SE of that gap (0 when pick == race best)
  * top1 / <1pt -- pick is the race best / within one point of it
  * vs ship     -- paired per-root difference field[cand] - field[ship];
                   SE across roots (carries both sampling and measurement
                   noise); roots where the picks agree contribute exactly 0

Usage:
    python -m ai.tutor.t0_btn_race_audit --labels D:/ofc_data/hu/t0btn_race480/labels_race.jsonl \
        --ship D:/ofc_data/hu/models_ship_20260904/hu/t0_btn.bin \
        --candidate D:/ofc_data/hu/t0btn_sharp2/model_s20260906/t0_btn.bin --fence 16
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from ai.tutor.t0_btn_sharp_verdict import Scorer
from ai.tutor.t0_btn_sharp_audit import load_meta, D


def picks_for(scores, meta, ids, fence):
    out = {}
    for rid in ids:
        g = meta[rid]; rank = g["ranker"]
        inside = np.flatnonzero((rank > 0) & (rank <= fence)) if fence else np.arange(len(rank))
        out[rid] = g["keys"][inside[int(np.argmax(scores[g["rows"][inside]]))]]
    return out


def price(picks, labels):
    reg, regvar, top1, near, miss = [], [], 0, 0, 0
    for rid, r in labels.items():
        f, se = r["field"], r["field_se"]; k = picks[rid]
        if k not in f:
            miss += 1; continue
        best = max(f, key=f.get)
        gap = f[best] - f[k]
        reg.append(gap); regvar.append(0.0 if k == best else se[best] ** 2 + se[k] ** 2)
        top1 += k == best; near += gap < 1.0
    n = len(reg)
    return dict(n=n, miss=miss, regret=sum(reg) / n, regret_meas_se=math.sqrt(sum(regvar)) / n,
                regret_root_se=float(np.std(reg, ddof=1) / math.sqrt(n)), top1=top1, near=near)


def paired(pa, pb, labels):
    d = [labels[r]["field"][pa[r]] - labels[r]["field"][pb[r]] for r in labels]
    moved = sum(pa[r] != pb[r] for r in labels)
    mv = [labels[r]["field_se"][pa[r]] ** 2 + labels[r]["field_se"][pb[r]] ** 2 for r in labels if pa[r] != pb[r]]
    n = len(d)
    return dict(mean=sum(d) / n, root_se=float(np.std(d, ddof=1) / math.sqrt(n)),
                meas_se=math.sqrt(sum(mv)) / n, moved=moved)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--ship", type=Path, default=D / "models_ship_20260904/hu/t0_btn.bin")
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--enc-dir", type=Path, default=D / "t0_btn_evalfix4")
    ap.add_argument("--fence", type=int, default=16)
    ap.add_argument("--fence-by", choices=("ranker_rank", "policy_rank"), default="ranker_rank",
                    help="the shortlist serving selects from: ranker at Button, policy at BB")
    args = ap.parse_args()
    labels = {}
    for line in args.labels.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                labels[r["id"]] = r
    meta = load_meta(args.enc_dir / "pairs_meta.jsonl", args.fence_by)
    ids = [r for r in labels if r in meta]
    labels = {r: labels[r] for r in ids}
    x = np.load(args.enc_dir / "enc_pairs.npz")["x"]
    ses = [max(r["field_se"].values()) for r in labels.values()]
    bestse = [r["field_se"][max(r["field"], key=r["field"].get)] for r in labels.values()]
    print(f"{len(labels)} raced roots; SE of race best: mean {np.mean(bestse):.3f}, "
          f"worst value SE mean {np.mean(ses):.3f}; fence {args.fence_by}<={args.fence}")
    rows = [("served", {r: labels[r]["served"] for r in labels})]
    models = [("ship", Scorer(str(args.ship)))] + [(None, Scorer(c)) for c in args.candidate]
    for label, sc in models:
        rows.append((label or sc.name, picks_for(sc.scores(x), meta, ids, args.fence)))
    ship_picks = rows[1][1]
    print(f"{'model':<34} {'regret':>7} {'±meas':>6} {'±root':>6} {'top1':>5} {'<1pt':>5} {'unpriced':>8}   vs ship (±root / ±meas) moved")
    for name, picks in rows:
        p = price(picks, labels)
        pr = paired(picks, ship_picks, labels) if name != "ship" else None
        tail = "" if pr is None else f"{pr['mean']:+.3f} (±{pr['root_se']:.3f} / ±{pr['meas_se']:.3f}) {pr['moved']}"
        print(f"{name:<34} {p['regret']:7.3f} {p['regret_meas_se']:6.3f} {p['regret_root_se']:6.3f} "
              f"{p['top1']:5d} {p['near']:5d} {p['miss']:8d}   {tail}")


if __name__ == "__main__":
    main()
