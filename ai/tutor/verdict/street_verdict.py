"""Judge lap-1 against lap-2 on one ruler both models are strangers to.

The dev split is not that ruler.  Each model picked its checkpoint on its own
lap's dev, so the diagonal of a cross-evaluation is a training report; and the
two off-diagonals sit on different corpora, so comparing them is two rulers
again -- the exact mistake that has been made three times this session.

The test splits are clean for both: neither model trained on them and neither
selected a checkpoint on them.  Pool lap1-test with lap2-test and both models
face identical rows.  Regret is charged per root, so the bootstrap resamples
ROOTS, not rows -- rows inside a root are one decision.

    python street_verdict.py t3_bb t2_bb t1_bb
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

D = pathlib.Path("D:/ofc_data/hu")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pooled_test(street: str):
    """lap1-test and lap2-test stacked, with root ids kept distinct."""
    xs, ys, roots, base = [], [], [], 0
    for tag in ("onpol", "lap2"):
        z = np.load(D / f"{street}_enc_{tag}" / "test.npz")
        xs.append(z["x"]); ys.append(z["y"])
        roots.append(z["roots"].astype(np.int64) + base)
        base = int(roots[-1].max()) + 1
    x = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=DEV)
    y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=DEV)
    root = np.concatenate(roots)
    order = np.argsort(root, kind="stable")
    groups = [g for g in np.split(order, np.flatnonzero(np.diff(root[order])) + 1)
              if g.size > 1]
    return x, y, groups


def per_root_regret(model_dir: pathlib.Path, x, y, groups) -> np.ndarray:
    blob = torch.load(model_dir / "evaluator_best.pt", map_location=DEV,
                      weights_only=False)
    model = T4FirstEvaluator(blob["input_dim"], tuple(blob["hidden"])).to(DEV)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model((x - blob["input_mean"].to(DEV)) / blob["input_std"].to(DEV))
    cost = np.empty(len(groups))
    for i, group in enumerate(groups):
        pick = group[int(torch.argmax(pred[group]).item())]
        cost[i] = float(y[group].max().item() - y[pick].item())
    return cost


def main() -> None:
    streets = sys.argv[1:] or ["t3_bb", "t3_btn", "t2_bb", "t2_btn", "t1_bb", "t1_btn"]
    rng = np.random.default_rng(20260815)
    print("pooled lap1-test + lap2-test -- no model trained or selected on these\n")
    print(f"{'street':<9}{'lap1':>9}{'lap2':>9}{'both':>9}{'both-lap1':>12}   95% CI")
    for street in streets:
        try:
            x, y, groups = pooled_test(street)
        except FileNotFoundError as missing:
            print(f"{street:<9} skipped ({missing.filename})")
            continue
        cost = {}
        for tag, suffix in (("lap1", "onpol"), ("lap2", "lap2"), ("both", "both")):
            path = D / f"{street}_{suffix}_s20260815"
            if (path / "evaluator_best.pt").exists():
                cost[tag] = per_root_regret(path, x, y, groups)
        cells = "".join(f"{cost[t].mean():>9.4f}" if t in cost else f"{'-':>9}"
                        for t in ("lap1", "lap2", "both"))
        tail = ""
        if "both" in cost:
            diff = cost["both"] - cost["lap1"]
            idx = rng.integers(0, len(diff), size=(4000, len(diff)))
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mark = "" if lo < 0 < hi else ("  both good" if hi < 0 else "  lap1 good")
            tail = f"{diff.mean():>+12.4f}   [{lo:+.4f}, {hi:+.4f}]{mark}"
        print(f"{street:<9}{cells}{tail}")
    print(f"\n{len(groups):,} roots on the last street; bootstrap resamples roots")


if __name__ == "__main__":
    main()
