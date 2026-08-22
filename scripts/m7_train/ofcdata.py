"""Shared data loading + metrics for the M7 feature/capacity ablation.

Corpus layout ("labelgen_features_v1"):
  x.bin      f32le, rows x dim
  y.bin      f64le, rows            (teacher label EV for that candidate)
  groups.bin u32le, rows            (position index, contiguous)
  codes.bin  u32le, rows            (STABLE per-position hash bucket 0..255)
  keys.txt   one action key per row

The holdout is code >= split_code. This is a stable hash of the position
skeleton, NOT a random shuffle, so the split is bit-reproducible and identical
across every arm.
"""
import json
import os

import numpy as np
import torch


def load_corpus(d, extra_x=None):
    meta = json.load(open(os.path.join(d, "meta.json")))
    dim = meta["dim"]
    x = np.fromfile(os.path.join(d, "x.bin"), dtype="<f4").reshape(-1, dim)
    y = np.fromfile(os.path.join(d, "y.bin"), dtype="<f8")
    groups = np.fromfile(os.path.join(d, "groups.bin"), dtype="<u4").astype(np.int64)
    codes = np.fromfile(os.path.join(d, "codes.bin"), dtype="<u4").astype(np.int64)
    assert x.shape[0] == y.shape[0] == groups.shape[0] == codes.shape[0], "row mismatch"
    if extra_x is not None:
        assert extra_x.shape[0] == x.shape[0], (
            f"extra feature rows {extra_x.shape[0]} != base rows {x.shape[0]}")
        x = np.concatenate([x, extra_x.astype(np.float32)], axis=1)
    return x, y, groups, codes, meta


def load_npz_corpus(paths, extra_x=None):
    """T3-second lives as sharded .npz with per-shard group numbering.

    Groups are re-based per shard so positions stay distinct after concat.
    """
    xs, ys, gs, cs = [], [], [], []
    off = 0
    for p in paths:
        z = np.load(p)
        g = z["groups"].astype(np.int64)
        xs.append(z["x"])
        ys.append(z["y"].astype(np.float64))
        gs.append(g + off)
        cs.append(z["codes"].astype(np.int64))
        off += int(g.max()) + 1
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    codes = np.concatenate(cs)
    if extra_x is not None:
        assert extra_x.shape[0] == x.shape[0]
        x = np.concatenate([x, extra_x.astype(np.float32)], axis=1)
    return x, y, groups, codes, {"dim": x.shape[1], "rows": x.shape[0], "positions": off}


def pack(x, y, groups, codes, width):
    """Pack rows into [P, width, D] with a boolean validity mask.

    Padding is ZERO in x and -inf-ish in y, but NOTHING may rely on the padding
    value: every max/argmax/comparison in this module masks first. (M11 lesson:
    an unmasked max over 0.0-padded slots silently returns the padding.)
    """
    P = int(groups.max()) + 1
    D = x.shape[1]
    counts = np.bincount(groups, minlength=P)
    assert counts.min() > 0
    assert counts.max() <= width, f"width {width} truncates (max actions {counts.max()})"
    order = np.argsort(groups, kind="stable")
    xs, ys, gs = x[order], y[order], groups[order]
    starts = np.zeros(P, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    slot = np.arange(len(gs)) - starts[gs]

    X = np.zeros((P, width, D), dtype=np.float32)
    Y = np.zeros((P, width), dtype=np.float64)
    M = np.zeros((P, width), dtype=bool)
    X[gs, slot] = xs
    Y[gs, slot] = ys
    M[gs, slot] = True
    pos_code = np.zeros(P, dtype=np.int64)
    pos_code[groups] = codes
    return X, Y, M, pos_code, counts


def split_masks(pos_code, split_code):
    return pos_code < split_code, pos_code >= split_code


NEG = -1e30


# Tie tolerances recovered by exact reproduction of the shipped T1-second
# model's published metadata (see metrics/gate_a.json). They are asymmetric in
# the original code, which is why top1 + strictly_worse != 1.
TOP1_TOL = 1e-3
WORSE_TOL = 1e-2


def metrics(scores, Y, M):
    """Held-out regret against the teacher labels.

    mean_regret == the project's `ev_given_up`
      = mean over positions of [best label EV] - [label EV of the model argmax].

    top1_agreement    : regret <= 1e-3
    top3_agreement    : the TEACHER'S argmax is inside the MODEL'S top-3 scores
                        (recall@3 of the teacher's best action -- not "model's
                        pick is in the teacher's top 3")
    strictly_worse    : regret > 1e-2

    scores/Y/M are [P, W]. EVERY max/argmax/comparison masks padding first --
    an unmasked max silently returns padding (M11 vs-FL lesson).
    """
    s = np.where(M, scores, NEG)
    yv = np.where(M, Y, NEG)
    n = len(s)
    idx = np.arange(n)
    pick = s.argmax(axis=1)
    best = yv.max(axis=1)
    got = yv[idx, pick]
    regret = best - got

    top3idx = np.argsort(-s, axis=1)[:, :3]
    contains_best = np.zeros(n, dtype=bool)
    for j in range(min(3, top3idx.shape[1])):
        contains_best |= yv[idx, top3idx[:, j]] >= best

    mae = float(np.abs(np.where(M, scores - Y, 0.0)).sum() / M.sum())

    return {
        "positions": int(n),
        "mean_regret": float(regret.mean()),
        "mae": mae,
        "regret_se": float(regret.std(ddof=1) / np.sqrt(n)),
        "median_regret": float(np.median(regret)),
        "p95_regret": float(np.percentile(regret, 95)),
        "max_regret": float(regret.max()),
        "top1_agreement": float((regret <= TOP1_TOL).mean()),
        "top3_agreement": float(contains_best.mean()),
        "strictly_worse": float((regret > WORSE_TOL).mean()),
    }


class MLP(torch.nn.Sequential):
    """Bare Sequential -- state_dict keys are '0.weight', '2.weight', ...

    This matches the shipped checkpoints exactly, so a shipped model loads
    without any key remapping.
    """

    def __init__(self, arch):
        layers = []
        for i in range(len(arch) - 1):
            layers.append(torch.nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(torch.nn.ReLU())
        super().__init__(*layers)


def n_params(arch):
    return sum(arch[i] * arch[i + 1] + arch[i + 1] for i in range(len(arch) - 1))
