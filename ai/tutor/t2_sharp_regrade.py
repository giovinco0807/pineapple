"""Re-grade every T2 checkpoint on the SHARPENED dev set.

The sharpened dev (D:/ofc_data/lap4_t2_own/sharpdev/pass{1,2}) re-prices
every action of every dev root at t3_draws=96 under two independent draw
streams.  Pooled (192 draws) values replace y; the same encoded feature
arrays are reused, so each model is scored on exactly the vector it serves
with, but graded against labels four times less noisy than the originals
(and free of the checkpoint-selection adaptation, since no training ever
saw these values).

Reports per model: sharpened dev regret, and a paired bootstrap CI against
the 110_s14 baseline.  The pass1-vs-pass2 split also gives a per-model
label-noise SE on the regret itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = Path("D:/ofc_data/lap4_t2_own")
SHARP = D / "sharpdev"
LABELS = D / "t2_labels_own_10k.jsonl"

MODELS = {
    "110_s14": (D / "model_s20260814/evaluator_best.pt", D / "encoded_110"),
    "110_s15": (D / "model_s20260815/evaluator_best.pt", D / "encoded_110"),
    "110_s16": (D / "model_s20260816/evaluator_best.pt", D / "encoded_110"),
    "120": (Path("D:/ofc_data/hu/fl_gen2/t2_120/evaluator_best.pt"), D / "encoded_120"),
    "128_k400": (D / "model_128_splitmix/evaluator_best.pt", D / "encoded_128_splitmix"),
    "128_k400_s14": (D / "model_128_splitmix_s20260814/evaluator_best.pt", D / "encoded_128_splitmix"),
    "128_k1024_s14": (D / "model_128_k1024_s20260814/evaluator_best.pt", D / "encoded_128_k1024"),
    "128_k1024_s15": (D / "model_128_k1024_s20260815/evaluator_best.pt", D / "encoded_128_k1024"),
    "128_k1024_s16": (D / "model_128_k1024_s20260816/evaluator_best.pt", D / "encoded_128_k1024"),
    "134_s15": (D / "model_134_s20260815/evaluator_best.pt", D / "encoded_134_splitmix"),
}
BASE = "110_s14"


class T4FirstEvaluator(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers, previous = [], input_dim
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predictions(model_path: Path, enc_dir: Path) -> np.ndarray:
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(ck["input_dim"], tuple(ck["hidden"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    fit_x = np.load(enc_dir / "fit.npz")["x"]
    mean, std = fit_x.mean(0), fit_x.std(0)
    std = np.where(std > 1e-6, std, 1.0)
    dev_x = np.load(enc_dir / "dev.npz")["x"]
    with torch.no_grad():
        return model(torch.from_numpy((dev_x - mean) / std).float()).numpy()


def load_pass(name: str) -> dict[tuple[str, str], float]:
    values = {}
    for line in (SHARP / name / "t2_labels.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        for act in rec["actions"]:
            values[(str(rec["id"]), act["action_key"])] = float(act["value"])
    return values


def main() -> None:
    import hashlib

    def split_of(rid: str) -> str:
        d = hashlib.sha256(f"fl14-t2-teacher-v1/{rid}".encode()).digest()
        b = int.from_bytes(d[:4], "big") % 100
        return "fit" if b < 80 else ("dev" if b < 90 else "test")

    # Row -> (root id, action key) in the encoders' dev order.
    dev_rows = []
    for line in open(LABELS, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = str(rec["id"])
        if split_of(rid) != "dev":
            continue
        for act in rec["actions"]:
            dev_rows.append((rid, act["action_key"]))

    ref = np.load(D / "encoded_128_splitmix/dev.npz")
    roots = ref["roots"]
    assert len(dev_rows) == len(roots), f"{len(dev_rows)} vs {len(roots)}"

    import os
    p1 = load_pass("pass1")
    # Smoke mode: validate the plumbing (row alignment, key matching) on
    # pass1 alone before pass2 exists; the noise SE is meaningless then.
    p2 = p1 if os.environ.get("T2_SMOKE") else load_pass("pass2")
    y1 = np.array([p1[key] for key in dev_rows], np.float64)
    y2 = np.array([p2[key] for key in dev_rows], np.float64)
    y_sharp = (y1 + y2) / 2.0

    order = np.argsort(roots, kind="stable")
    bounds = np.flatnonzero(np.diff(roots[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 2]

    per_root = {}
    for name, paths in MODELS.items():
        if not paths[0].exists():
            print(f"{name}: checkpoint missing, skipped")
            continue
        p = predictions(*paths)
        regret = []
        noise = []
        for g in groups:
            pick = g[int(p[g].argmax())]
            regret.append(y_sharp[g].max() - y_sharp[pick])
            # Same pick graded on each pass: half the difference samples the
            # label noise this root contributes to the model's regret.
            r1 = y1[g].max() - y1[pick]
            r2 = y2[g].max() - y2[pick]
            noise.append((r1 - r2) / 2.0)
        per_root[name] = np.asarray(regret)
        noise = np.asarray(noise)
        print(f"{name}: sharp dev regret {per_root[name].mean():.5f}  "
              f"(label-noise SE {np.sqrt((noise**2).sum())/len(groups):.5f})")

    rng = np.random.default_rng(20260901)
    base = per_root[BASE]
    n = len(base)
    print(f"\npaired vs {BASE} (bootstrap over {n} roots):")
    for name, arr in per_root.items():
        if name == BASE:
            continue
        delta = arr - base
        draws = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(10_000)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        verdict = "WORSE" if lo > 0 else ("BETTER" if hi < 0 else "unresolved")
        print(f"  {name:>14}: {delta.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  {verdict}")


if __name__ == "__main__":
    main()
