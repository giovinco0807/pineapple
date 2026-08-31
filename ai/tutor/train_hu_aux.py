"""Aux-head distillation: the sampling moves from serve time to train time.

The production evaluator reads the joint block -- four hundred sampled
completions per candidate -- as an *input*, which is why serving spends
99.99% of its time sampling.  This trainer makes those sixteen numbers
*targets* instead: the trunk must learn to compute them, the value head reads
the representation that computation shapes, and at serve time nothing is
sampled at all.

Measured at T3-BTN against exact labels, this beats the model that is handed
the true sampled block at serve time (dev regret 0.231 vs 0.292).  Two
reasons it can win from apparent disadvantage: the net converges to the
conditional mean of its noisy 400-sample targets -- the infinite-rollout
joint -- while the production model reads a fresh noisy draw every call; and
the auxiliary gradient regularises the trunk, which is what card-level
inputs alone fatally lacked (pure one-hots memorised: fit 2.1 / dev 9.8).

Inputs are the hybrid rows (432 card one-hots + 191 cheap feature dims);
aux targets are the sixteen joint dims of the 207-dim encoding, row-aligned
by construction and re-verified here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

JOINT = list(range(89, 97)) + list(range(199, 207))


class AuxNet(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 16)

    def forward(self, x):
        h = self.trunk(x)
        return self.value(h).squeeze(-1), self.aux(h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--enc-dir", type=Path, required=True,
                        help="207-dim encoding, source of the aux targets")
    parser.add_argument("--out", type=Path, required=True,
                        help="report path (json)")
    parser.add_argument("--lambdas", default="0,2",
                        help="aux weights to train, comma-separated")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="save each lambda's best net here, already in "
                             "T4FirstEvaluator checkpoint form so the "
                             "existing .bin exporter serves it unchanged.  "
                             "The value head is trained on standardised y, so "
                             "scores are a monotone transform of points -- "
                             "fine for a chooser, wrong for a settlement.")
    parser.add_argument("--metric", choices=["regret", "mae"], default="regret",
                        help="mae for boundary nets: one row per root makes "
                             "every regret group a singleton, identically zero")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(split):
        a = np.load(args.hybrid_dir / f"{split}.npz")
        b = np.load(args.enc_dir / f"{split}.npz")
        if not (np.array_equal(a["roots"], b["roots"])
                and np.allclose(a["y"], b["y"], atol=1e-4)):
            raise SystemExit(f"FATAL: {split} rows do not align")
        return a["x"], a["y"], a["roots"], b["x"][:, JOINT]

    xf, yf, rf, af = load("fit")
    xd, yd, rd, ad = load("dev")
    # Deviations are floored, not epsiloned: near-constant columns exist in
    # both spaces and a millionth-scale divisor turns them into the largest
    # feature in the vector.
    xm, xs = xf.mean(0), np.maximum(xf.std(0), 1e-2)
    ym, ys = float(yf.mean()), float(yf.std())
    am, as_ = af.mean(0), np.maximum(af.std(0), 1e-2)

    X = torch.tensor((xf - xm) / xs).to(device)
    Y = torch.tensor((yf - ym) / ys).to(device)
    A = torch.tensor((af - am) / as_).to(device)
    Xd = torch.tensor((xd - xm) / xs).to(device)

    order = np.argsort(rd, kind="stable")
    groups, start = [], 0
    rs = rd[order]
    for i in range(1, len(rs) + 1):
        if i == len(rs) or rs[i] != rs[i - 1]:
            groups.append(order[start:i])
            start = i
    best_y = np.array([yd[g].max() for g in groups])

    results = {}
    for lam in [float(v) for v in args.lambdas.split(",")]:
        torch.manual_seed(args.seed)
        net = AuxNet(X.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
        best = float("inf")
        best_state = None
        for _epoch in range(args.epochs):
            perm = torch.randperm(len(X), device=device)
            for i in range(0, len(X), 4096):
                b = perm[i:i + 4096]
                v, a = net(X[b])
                loss = nn.functional.mse_loss(v, Y[b])
                if lam:
                    loss = loss + lam * nn.functional.mse_loss(a, A[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = np.concatenate([
                    net(Xd[i:i + 65536])[0].cpu().numpy()
                    for i in range(0, len(Xd), 65536)
                ])
            if args.metric == "regret":
                score = float(np.mean([
                    best_y[k] - yd[g[np.argmax(pred[g])]]
                    for k, g in enumerate(groups)
                ]))
            else:
                score = float(np.abs(pred * ys + ym - yd).mean())
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
        results[str(lam)] = best
        if args.save_dir is not None and best_state is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            mapping = {"trunk.0": "net.0", "trunk.2": "net.2",
                       "trunk.4": "net.4", "value": "net.6"}
            composed = {}
            for key, value in best_state.items():
                head, _, rest = key.partition(".")
                prefix = f"{head}.{rest.split('.')[0]}" if head == "trunk" else head
                if prefix in mapping:
                    tail = key.split(".")[-1]
                    composed[f"{mapping[prefix]}.{tail}"] = value
            torch.save({
                "input_dim": int(X.shape[1]),
                "hidden": (512, 256, 128),
                "model_state_dict": composed,
                "input_mean": torch.tensor(xm, dtype=torch.float32),
                "input_std": torch.tensor(xs, dtype=torch.float32),
            }, args.save_dir / f"evaluator_lam{lam:g}.pt")
            print(f"saved {args.save_dir}/evaluator_lam{lam:g}.pt", flush=True)
        print(f"lambda={lam}: best dev {args.metric} {best:.5f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "ofc_hu_aux_distill/v1",
        "hybrid_dir": str(args.hybrid_dir),
        "seed": args.seed,
        "dev_roots": len(groups),
        "regret_by_lambda": results,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
