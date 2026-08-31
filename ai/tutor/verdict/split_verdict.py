"""Ask whether relabelling on-policy bought anything that volume alone would not.

The pooled verdict says lap2-only lost on every street and the union won on
every street -- but the union also had 2.1x the roots, so "more data" explains
all of it.  The claim on-policy makes is narrower and testable: lap 2's roots
are the states the CURRENT champion actually reaches, so a model trained there
should be better THERE, whatever it does elsewhere.

So score both models on each lap's test split separately.  Both splits are
clean for both models -- neither trained nor selected on either.

    lap2-test is the current policy's own distribution.  If lap2-only still
    loses there, relabelling bought nothing and the lap is just a slow way to
    collect rows.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

D = pathlib.Path("D:/ofc_data/hu")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_split(street: str, tag: str):
    z = np.load(D / f"{street}_enc_{tag}" / "test.npz")
    x = torch.tensor(z["x"], dtype=torch.float32, device=DEV)
    y = torch.tensor(z["y"], dtype=torch.float32, device=DEV)
    root = z["roots"].astype(np.int64)
    order = np.argsort(root, kind="stable")
    groups = [g for g in np.split(order, np.flatnonzero(np.diff(root[order])) + 1)
              if g.size > 1]
    return x, y, groups


def regret(model_dir: pathlib.Path, x, y, groups) -> np.ndarray:
    blob = torch.load(model_dir / "evaluator_best.pt", map_location=DEV,
                      weights_only=False)
    model = T4FirstEvaluator(blob["input_dim"], tuple(blob["hidden"])).to(DEV)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model((x - blob["input_mean"].to(DEV)) / blob["input_std"].to(DEV))
    out = np.empty(len(groups))
    for i, group in enumerate(groups):
        pick = group[int(torch.argmax(pred[group]).item())]
        out[i] = float(y[group].max().item() - y[pick].item())
    return out


def main() -> None:
    streets = sys.argv[1:] or ["t3_bb", "t3_btn", "t2_bb", "t2_btn", "t1_bb", "t1_btn"]
    rng = np.random.default_rng(20260815)
    for tag, label in (("onpol", "lap1-test (前世代の分布)"),
                       ("lap2", "lap2-test (現チャンピオンの分布)")):
        print(f"\n=== {label} ===")
        print(f"{'street':<9}{'lap1':>9}{'lap2':>9}{'both':>9}"
              f"{'both-best単独':>15}   95% CI")
        for street in streets:
            x, y, groups = test_split(street, tag)
            a = regret(D / f"{street}_onpol_s20260815", x, y, groups)
            b = regret(D / f"{street}_lap2_s20260815", x, y, groups)
            u = regret(D / f"{street}_both_s20260815", x, y, groups)
            # Compare the union against whichever specialist owns this split --
            # the bar is "as good as the home model", not "better than average".
            home = a if a.mean() <= b.mean() else b
            diff = u - home
            idx = rng.integers(0, len(diff), size=(4000, len(diff)))
            lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
            mark = "" if lo < 0 < hi else ("  both good" if hi < 0 else "  単独が上")
            print(f"{street:<9}{a.mean():>9.4f}{b.mean():>9.4f}{u.mean():>9.4f}"
                  f"{diff.mean():>+15.4f}   [{lo:+.4f}, {hi:+.4f}]{mark}")


if __name__ == "__main__":
    main()
