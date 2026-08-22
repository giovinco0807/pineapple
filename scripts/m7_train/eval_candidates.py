"""Score checkpoint candidates on the three sets that matter before a gate.

(i)  the mined miss positions -- does the candidate now pick the measured best?
(ii) a winner sample -- does it still pick what it already got right?
(iii) the ORIGINAL corpus's holdout only -- a v2-comparable held regret,
      uncontaminated by boost copies.

Position order in the dump is offset order: original 25,000 first, then the
mined 3,600 (offsets 1,000,000+), then boost copies (2,000,000+).  Runs on the
x8 corpus so every candidate sees identical rows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "/home/wner/ofc-m7/ablation/scripts")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ofcdata import MLP, load_corpus  # noqa: E402


def per_position(x, y, groups, model, device):
    scores = np.empty(len(y), dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, len(y), 65536):
            hi = min(lo + 65536, len(y))
            t = torch.from_numpy(x[lo:hi]).to(device)
            scores[lo:hi] = model(t).squeeze(-1).cpu().numpy()
    top1 = 0
    regret = 0.0
    n = 0
    for g in np.unique(groups):
        m = groups == g
        pick = np.argmax(scores[m])
        best = np.argmax(y[m])
        top1 += int(pick == best)
        regret += float(y[m][best] - y[m][pick])
        n += 1
    return n, top1 / n, regret / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--misses", required=True,
                        help="misses.jsonl (fleet offsets)")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="label=path pairs")
    args = parser.parse_args()

    x, y, groups, codes, meta = load_corpus(args.corpus)
    print(f"corpus: {meta['rows']} rows, {meta['positions']} positions")

    starts = np.flatnonzero(np.r_[True, np.diff(groups) != 0])
    pos_group = groups[starts]
    pos_code = codes[starts]

    miss_offsets = sorted(
        json.loads(line)["offset"]
        for line in pathlib.Path(args.misses).read_text(encoding="utf-8").splitlines()
        if line.strip() and "offset" in json.loads(line))
    mined_pos = np.arange(25000, 28600)
    offset_rank = {off: i for i, off in enumerate(sorted(range(3600)))}
    miss_pos = np.array([25000 + offset_rank[o] for o in miss_offsets])
    orig_held_pos = np.flatnonzero(pos_code[:25000] >= 230)
    winner_pos = np.setdiff1d(mined_pos, miss_pos)[:600]

    def subset(pos_indices):
        wanted = np.isin(groups, pos_group[pos_indices])
        return x[wanted], y[wanted], groups[wanted]

    sets = {
        "miss121": subset(miss_pos),
        "winner600": subset(winner_pos),
        "orig_held": subset(orig_held_pos),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for pair in args.checkpoints:
        label, path = pair.split("=", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck.get("model", ck.get("state_dict", ck))
        model = MLP([168, 256, 128, 64, 1]).to(device)
        model.load_state_dict(state if not hasattr(state, "state_dict") else state.state_dict())
        model.eval()
        mean = ck.get("mean", ck.get("x_mean"))
        std = ck.get("std", ck.get("x_std"))
        print(f"\n== {label} ==")
        for name, (sx, sy, sg) in sets.items():
            fx = sx.astype(np.float32)
            if mean is not None:
                fx = (fx - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
            n, top1, regret = per_position(fx, sy, sg, model, device)
            print(f"  {name:10s} n={n:5d}  top1={top1:.4f}  regret={regret:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
