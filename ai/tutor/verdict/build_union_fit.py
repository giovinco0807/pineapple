"""Stack lap-1's fit onto lap-2's fit, so "more data" can be tested separately
from "newer data".

lap 2 trained on 87.3% of the roots lap 1 had, and came out slightly worse on
every street.  Two explanations fit that: the corpus is smaller, or the newer
policy's states are harder to learn.  Concatenating the two fit splits and
retraining separates them -- if the gap was size, the union beats both.

Root ids restart at zero in each lap, so lap 2's are shifted past lap 1's;
otherwise two unrelated decisions would be scored as one root's action set.
Only fit and dev are written.  The test splits stay untouched and unpooled,
because they are the ruler this experiment is judged on.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

D = pathlib.Path("D:/ofc_data/hu")


def build(street: str) -> None:
    out = D / f"{street}_enc_both"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("fit", "dev"):
        xs, ys, roots, jokers, base = [], [], [], [], 0
        for tag in ("onpol", "lap2"):
            z = np.load(D / f"{street}_enc_{tag}" / f"{split}.npz")
            xs.append(z["x"]); ys.append(z["y"])
            shifted = z["roots"].astype(np.int64) + base
            roots.append(shifted)
            base = int(shifted.max()) + 1
            if "jokers" in z:
                jokers.append(z["jokers"])
        payload = {"x": np.concatenate(xs), "y": np.concatenate(ys),
                   "roots": np.concatenate(roots)}
        if len(jokers) == len(xs):
            payload["jokers"] = np.concatenate(jokers)
        np.savez(out / f"{split}.npz", **payload)
        print(f"  {street}/{split}: {payload['x'].shape[0]:,} rows, "
              f"{len(np.unique(payload['roots'])):,} roots")


if __name__ == "__main__":
    for street in sys.argv[1:] or ["t3_bb", "t3_btn", "t2_bb", "t2_btn",
                                   "t1_bb", "t1_btn"]:
        build(street)
