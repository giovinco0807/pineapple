"""Pool a gate's shards into one verdict, on the standing criterion.

Reports the same fields the T3 v7 gates reported, so the two generations are
read on one scale. The health checks come first and can veto the verdict: if a
deal where neither chain deviated produced a nonzero pair, the mirror did not
cancel and nothing downstream means anything.
"""
import glob
import json
import math
import pathlib
import statistics
import sys

FLOOR = -0.05


def report(directory, label):
    shards = sorted(glob.glob(str(pathlib.Path(directory) / "shard_*.json")))
    if not shards:
        print(f"{label}: no shards")
        return None
    pairs, divergent, seconds = [], [], []
    swap = None
    for path in shards:
        d = json.loads(pathlib.Path(path).read_text())
        pairs.extend(d["pairs"])
        divergent.extend(d["divergent"])
        seconds.append(d["seconds"])
        swap = swap or d.get("swap_path")

    n = len(pairs)
    mean = statistics.mean(pairs)
    stderr = statistics.stdev(pairs) / math.sqrt(n)
    lo, hi = mean - 1.96 * stderr, mean + 1.96 * stderr
    div_pairs = [p for p, d in zip(pairs, divergent) if d]
    identical = [p for p, d in zip(pairs, divergent) if not d]
    bad = [p for p in identical if p != 0.0]

    print(f"=== {label}")
    print(f"  model            {swap}")
    print(f"  shards / deals   {len(shards)} / {n:,}   games {2*n:,}")
    print(f"  divergence       {sum(divergent)/n:.4f}  ({sum(divergent):,} deals)")
    print(f"  paired mean      {mean:+.4f}  per hand")
    print(f"  stderr / 95% CI  {stderr:.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  mean on divergent{statistics.mean(div_pairs):+.4f}"
          if div_pairs else "  mean on divergent  n/a")
    print(f"  identical deals  {len(identical):,}  nonzero among them {len(bad)}")
    print(f"  longest shard    {max(seconds)/3600:.2f} h   "
          f"pooled {2*n/(max(seconds)/3600):,.0f} games/h")

    if bad:
        print(f"  MIRROR BROKEN: {len(bad)} identical-decision deals did not "
              f"cancel (max {max(abs(p) for p in bad):.6f}) -- NO VERDICT")
        return None
    print("  mirror clean     every identical-decision deal cancelled to 0.0")

    verdict = "ADOPT" if (lo > FLOOR and mean >= 0) else (
        "NO_GAIN" if lo > FLOOR else "REGRESSION")
    print(f"  VERDICT          {verdict}   (rule: lower CI > {FLOOR} and mean >= 0)")
    return verdict


if __name__ == "__main__":
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "/home/wner/ofc-m7/gate_t2_2048")
    for seat in ("first", "second"):
        report(base / seat, f"T2 {seat} seat")
        print()
