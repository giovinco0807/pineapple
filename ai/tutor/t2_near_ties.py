"""Extract the roots the sharpened dev still cannot adjudicate, for the
exact (--t3-draws 0, pilot top-1) re-pricing pass.

A root goes to the exact list when, for any model pair we care about
(each challenger vs the 110_s14 base), the two nets pick different actions
AND the pooled 192-draw value difference of those picks is within
`margin_sigma` of the per-root label noise (measured from the pass1/pass2
split of that root's own pick difference).  Everything else is already
resolved by the 192-draw labels.

Outputs exact_roots.jsonl / exact_keep.jsonl in teach-t2 format; keep holds
the union of disputed pick keys plus the pooled label best (so the exact
pass can also confirm the per-root best for regret levels, not just pairs).
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
SCRATCH = Path(r"C:\Users\Owner\AppData\Local\Temp\claude\C--Users-Owner--gemini-antigravity-scratch-ofc-pineapple\fbbaae4a-e47e-44b7-ab1b-cde058d5523e\scratchpad")

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
MARGIN_SIGMA = 3.0


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


def load_pass(name: str):
    values = {}
    for line in (SHARP / name / "t2_labels.jsonl").open(encoding="utf-8"):
        if line.strip():
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

    dev_rows, recs = [], {}
    for line in open(LABELS, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = str(rec["id"])
        if split_of(rid) != "dev":
            continue
        recs[rid] = rec
        for act in rec["actions"]:
            dev_rows.append((rid, act["action_key"]))

    ref = np.load(D / "encoded_128_k1024/dev.npz")
    roots = ref["roots"]
    assert len(dev_rows) == len(roots)
    p1, p2 = load_pass("pass1"), load_pass("pass2")
    y1 = np.array([p1[k] for k in dev_rows])
    y2 = np.array([p2[k] for k in dev_rows])
    y = (y1 + y2) / 2.0

    order = np.argsort(roots, kind="stable")
    bounds = np.flatnonzero(np.diff(roots[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 2]

    preds = {name: predictions(*paths) for name, paths in MODELS.items()}

    need: dict[str, set[str]] = {}
    reasons = 0
    for g in groups:
        rid = dev_rows[g[0]][0]
        picks = {name: g[int(p[g].argmax())] for name, p in preds.items()}
        base_pick = picks[BASE]
        wanted: set[int] = set()
        for name, pick in picks.items():
            if name == BASE or pick == base_pick:
                continue
            d1 = y1[base_pick] - y1[pick]
            d2 = y2[base_pick] - y2[pick]
            se = abs(d1 - d2) / 2.0 + 1e-9
            if abs((d1 + d2) / 2.0) <= MARGIN_SIGMA * se:
                wanted.add(pick)
                reasons += 1
        if wanted:
            wanted.add(base_pick)
            wanted.add(g[int(y[g].argmax())])
            need[rid] = {dev_rows[i][1] for i in wanted}

    print(f"near-tie roots: {len(need)} (pair-events {reasons}), "
          f"avg keys/root {np.mean([len(v) for v in need.values()]):.1f}")
    with (SCRATCH / "exact_roots.jsonl").open("w", encoding="utf-8") as ro, \
         (SCRATCH / "exact_keep.jsonl").open("w", encoding="utf-8") as ke:
        for rid, keys in need.items():
            rec = recs[rid]
            rows = [p.split(",") if p else [] for p in rec["board"].split("|")]
            ro.write(json.dumps({"id": rid, "rows": rows,
                                 "dead": rec["dead"].split(","),
                                 "draw": rec["draw"].split(",")}) + "\n")
            ke.write(json.dumps({"id": rid, "keep": sorted(keys)}) + "\n")
    print(f"wrote exact_roots.jsonl / exact_keep.jsonl")


if __name__ == "__main__":
    main()
