"""Summarise leak-map / pilot records: the headline is the MEAN paired margin.

Zeros (race re-nominated the served move) and negatives (the champion was
right) are included -- that is what makes the estimator unbiased for "how
much switching to this referee's nominee would gain per decision".  The CI
is a bootstrap over roots.  `confirmed` and the tie rate are descriptive.

Usage:
    python -m ai.tutor.hu_leak_summary out.jsonl [null.jsonl]
"""
from __future__ import annotations
import json
import sys
import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    graded = [r for r in rows if "margin" in r]
    errors = [r for r in rows if "error" in r or "skipped" in r]
    return graded, errors


def summarise(label, graded, errors):
    m = np.array([r["margin"] for r in graded], float)
    n = len(m)
    rng = np.random.default_rng(20260902)
    boot = np.array([m[rng.integers(0, n, n)].mean() for _ in range(10000)]) if n else np.array([0.0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ties = sum(1 for r in graded if r.get("challenger") == r.get("served"))
    disputed = n - ties
    conf = sum(1 for r in graded if r.get("confirmed"))
    pos = sum(1 for r in graded if r["margin"] > 0)
    print(f"== {label}: {n} graded, {len(errors)} errored/skipped")
    print(f"   mean margin (leak lower bound)  {m.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  /decision")
    print(f"   race re-nominated served (tie)  {ties}/{n} = {100*ties/max(n,1):.0f}%")
    print(f"   disputed {disputed}: margin>0 {pos}, confirmed {conf} "
          f"({100*conf/max(n,1):.1f}% of all roots)")
    if disputed:
        d = np.array([r["margin"] for r in graded if r.get("challenger") != r.get("served")])
        print(f"   disputed margin mean {d.mean():+.3f}, median {np.median(d):+.3f}, "
              f"max {d.max():+.2f}")
    ranks = [r["served_model_rank"] for r in graded if "served_model_rank" in r]
    if ranks:
        r = np.array(ranks)
        print(f"   served's own-model rank: 1st {np.mean(r==1):.0%}, top-4 {np.mean(r<=4):.0%}, >8 {np.mean(r>8):.0%}")


def main():
    graded, errors = load(sys.argv[1])
    summarise("PILOT", graded, errors)
    if len(sys.argv) > 2:
        ng, ne = load(sys.argv[2])
        summarise("NULL (served vs itself)", ng, ne)
        nm = np.array([r["margin"] for r in ng])
        conf = sum(1 for r in ng if r.get("confirmed"))
        print(f"   -> instrument false-positive rate {conf}/{len(ng)}; null margin SD {nm.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
