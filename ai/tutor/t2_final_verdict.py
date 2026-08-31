"""Final paired verdicts vs 110_s14, exact where it matters.

Per disagreement root the difference d = v(base pick) - v(challenger pick)
decides everything; the best-action term cancels.  Where the exact pass
re-priced BOTH picks (the near-tie roots), d comes from the exact values --
their shared per-root pilot bias cancels in the difference (residual
differential measured <= 0.016/root).  Everywhere else d comes from the
pooled 192-draw labels, whose noise the near-tie cut already showed to be
harmless there (|d| > 3 sigma).

Bootstrap over the 1,022 dev roots gives the sampling CI; the label-noise
SE of the pooled part rides on the pass split as before.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = Path("D:/ofc_data/lap4_t2_own")
SHARP = D / "sharpdev"
LABELS = D / "t2_labels_own_10k.jsonl"
SCRATCH = Path(r"C:\Users\Owner\AppData\Local\Temp\claude\C--Users-Owner--gemini-antigravity-scratch-ofc-pineapple\fbbaae4a-e47e-44b7-ab1b-cde058d5523e\scratchpad")
EXACT = SCRATCH / "exact_out/t2_labels.jsonl"

MODELS = {
    "110_s14": (D / "model_s20260814/evaluator_best.pt", D / "encoded_110"),
    "110_s15": (D / "model_s20260815/evaluator_best.pt", D / "encoded_110"),
    "110_s16": (D / "model_s20260816/evaluator_best.pt", D / "encoded_110"),
    "128_k1024_s14": (D / "model_128_k1024_s20260814/evaluator_best.pt", D / "encoded_128_k1024"),
    "128_k1024_s15": (D / "model_128_k1024_s20260815/evaluator_best.pt", D / "encoded_128_k1024"),
    "128_k1024_s16": (D / "model_128_k1024_s20260816/evaluator_best.pt", D / "encoded_128_k1024"),
    "128_k400": (D / "model_128_splitmix/evaluator_best.pt", D / "encoded_128_splitmix"),
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


def load_jsonl_values(path: Path):
    values = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            for act in rec["actions"]:
                values[(str(rec["id"]), act["action_key"])] = float(act["value"])
    return values


def split_of(rid: str) -> str:
    d = hashlib.sha256(f"fl14-t2-teacher-v1/{rid}".encode()).digest()
    b = int.from_bytes(d[:4], "big") % 100
    return "fit" if b < 80 else ("dev" if b < 90 else "test")


def main() -> None:
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
    ref = np.load(D / "encoded_128_k1024/dev.npz")
    roots = ref["roots"]
    assert len(dev_rows) == len(roots)

    p1 = load_jsonl_values(SHARP / "pass1/t2_labels.jsonl")
    p2 = load_jsonl_values(SHARP / "pass2/t2_labels.jsonl")
    exact = load_jsonl_values(EXACT)
    y1 = np.array([p1[k] for k in dev_rows])
    y2 = np.array([p2[k] for k in dev_rows])
    pooled = (y1 + y2) / 2.0

    order = np.argsort(roots, kind="stable")
    bounds = np.flatnonzero(np.diff(roots[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 2]

    preds = {name: predictions(*paths) for name, paths in MODELS.items()}
    n = len(groups)
    rng = np.random.default_rng(20260901)

    print(f"exact-priced keys: {len(exact)}; dev roots {n}")
    for name in MODELS:
        if name == BASE:
            continue
        deltas = np.zeros(n)
        used_exact = 0
        for i, g in enumerate(groups):
            base_pick = g[int(preds[BASE][g].argmax())]
            pick = g[int(preds[name][g].argmax())]
            if pick == base_pick:
                continue
            kb, kc = dev_rows[base_pick], dev_rows[pick]
            if kb in exact and kc in exact:
                deltas[i] = exact[kb] - exact[kc]
                used_exact += 1
            else:
                deltas[i] = pooled[base_pick] - pooled[pick]
        draws = np.array([deltas[rng.integers(0, n, n)].mean()
                          for _ in range(10_000)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        verdict = "WORSE" if lo > 0 else ("BETTER" if hi < 0 else "unresolved")
        print(f"  {name:>14}: {deltas.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  "
              f"{verdict}  (exact on {used_exact} roots)")


if __name__ == "__main__":
    main()
