"""Score existing checkpoints on somebody else's dev split.

The point is to stop comparing numbers that were never comparable.  A trainer
prints the regret its checkpoint reached on *its own* dev set, and this lap's
dev set is drawn from this lap's corpus -- so "8.42 vs 8.08" across laps says
nothing until both models face the same rows.  Three times this session a
"degradation" turned out to be two different rulers held up side by side.

    python cross_eval.py --models A=D:/.../t1_bb_onpol_s20260815 \
                                  B=D:/.../t1_bb_lap2_s20260815 \
                         --devs lap1=D:/.../t1_bb_enc_onpol \
                                lap2=D:/.../t1_bb_enc_lap2

Every model is scored on every dev, printed as a matrix.  A model's own dev is
marked with a dot, because a diagonal number is a training report, not a
comparison.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")

# Imported, not restated: a hand-copied stack drifts from the trainer's and
# then fails on the state_dict key names, which is exactly how this started.
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator


def load_dev(data_dir: pathlib.Path, device: torch.device):
    payload = np.load(data_dir / "dev.npz")
    x = torch.tensor(payload["x"], dtype=torch.float32, device=device)
    y = torch.tensor(payload["y"], dtype=torch.float32, device=device)
    groups = None
    if "roots" in payload:
        roots = payload["roots"]
        order = np.argsort(roots, kind="stable")
        edges = np.flatnonzero(np.diff(roots[order])) + 1
        groups = [g for g in np.split(order, edges) if g.size > 1]
    return x, y, groups


def score(path: pathlib.Path, x, y, groups, device):
    blob = torch.load(path / "evaluator_best.pt", map_location=device, weights_only=False)
    model = T4FirstEvaluator(blob["input_dim"], tuple(blob["hidden"])).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    if x.shape[1] != blob["input_dim"]:
        raise SystemExit(
            f"{path.name}: model wants {blob['input_dim']} dims, dev has {x.shape[1]}"
        )
    normed = (x - blob["input_mean"].to(device)) / blob["input_std"].to(device)
    with torch.no_grad():
        pred = model(normed)
    mae = float((pred - y).abs().mean().item())
    regret = float("nan")
    if groups:
        total = 0.0
        for group in groups:
            pick = group[int(torch.argmax(pred[group]).item())]
            total += float(y[group].max().item() - y[pick].item())
        regret = total / len(groups)
    return regret, mae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="tag=dir")
    parser.add_argument("--devs", nargs="+", required=True, help="tag=encoded-dir")
    parser.add_argument("--metric", choices=["regret", "mae"], default="regret")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = dict(pair.split("=", 1) for pair in args.models)
    devs = dict(pair.split("=", 1) for pair in args.devs)

    loaded = {}
    for tag, path in devs.items():
        x, y, groups = load_dev(pathlib.Path(path), device)
        loaded[tag] = (x, y, groups)
        print(f"dev {tag}: {x.shape[0]:,} rows, "
              f"{len(groups) if groups else 0:,} roots, {x.shape[1]} dims")

    width = max(len(t) for t in models) + 2
    print("\n" + "model".ljust(width) + "".join(f"{t:>12}" for t in devs))
    for mtag, mpath in models.items():
        cells = []
        for dtag, (x, y, groups) in loaded.items():
            regret, mae = score(pathlib.Path(mpath), x, y, groups, device)
            value = regret if args.metric == "regret" else mae
            own = "." if dtag in mtag or mtag in dtag else " "
            cells.append(f"{value:>11.4f}{own}")
        print(mtag.ljust(width) + "".join(cells))
    print("\n. = the model's own dev split (a training report, not a comparison)")


if __name__ == "__main__":
    main()
