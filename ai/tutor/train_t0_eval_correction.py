"""Teach the T0-BB serving evaluator the referee's candidate ORDERING.

Why this exists (docs/t0_evaluator_teaching_design_20260830.md): the mining
flywheel proved the policy net already fences in the referee's answer on 22
of 23 repeated errors -- the wrong pick survives because the EVALUATOR argmaxes
to it inside the fence, and policy retraining can never reach that.  Values
can't be transplanted (teacher labels are own-hand EV, referee means are
full-game settlements), but the serving decision only consumes the ordering,
so the ordering is what gets taught.

Recipe mirrors the policy net's proven stage-B: start FROM the shipped weights
(parsed straight out of the .bin image -- the torch run dirs predate the
bundle), anchor every step with a replay of the original corpus under the
original SmoothL1 objective so the 207-dim landscape doesn't drift, and add a
margin-weighted pairwise hinge over each audited root's referee ordering.
Roots are held out by id hash so every version of this model shares the split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = Path("D:/ofc_data/hu")


def load_bin(path: Path):
    """Parse a T4F1 image back into (mean, std, [(W, b), ...])."""
    raw = path.read_bytes()
    assert raw[:4] == b"T4F1"
    version, layers, input_dim = struct.unpack_from("<III", raw, 4)
    assert version == 1
    off = 16
    mean = np.frombuffer(raw, np.float32, input_dim, off); off += 4 * input_dim
    std = np.frombuffer(raw, np.float32, input_dim, off); off += 4 * input_dim
    mats = []
    for _ in range(layers):
        n_in, n_out = struct.unpack_from("<II", raw, off); off += 8
        w = np.frombuffer(raw, np.float32, n_in * n_out, off).reshape(n_out, n_in)
        off += 4 * n_in * n_out
        b = np.frombuffer(raw, np.float32, n_out, off); off += 4 * n_out
        mats.append((w.copy(), b.copy()))
    return mean.copy(), std.copy(), mats


def build_net(mats):
    layers = []
    for i, (w, b) in enumerate(mats):
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(w))
            lin.bias.copy_(torch.tensor(b))
        layers.append(lin)
        if i < len(mats) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def load_pairs(enc_dir: Path):
    feats = {}
    for f in sorted(enc_dir.glob("enc_*.jsonl")):
        if f.name == "enc_requests.jsonl":
            continue
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue  # killed encoders leave one truncated final line
            feats[r["id"]] = np.asarray(r["features"], np.float32)
    roots = {}
    for l in open(enc_dir / "pairs_meta.jsonl", encoding="utf-8"):
        m = json.loads(l)
        if m["id"] not in feats:
            continue
        roots.setdefault(m["root"], []).append(
            dict(x=feats[m["id"]], mean=m["mean"], scored=m["scored"],
                 escalated=m["escalated"], is_serving=m["is_serving"],
                 is_ref=m["is_ref"]))
    # mine4's halving truncation left rows with a 2-candidate ordering; a pair
    # is still a pair, so they stay, but a root needs at least 2 to teach.
    return {k: v for k, v in roots.items() if len(v) >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=D / "models_ship_20260830/hu/t0_bb.bin")
    ap.add_argument("--enc-dir", type=Path, default=D / "t0_evalfix")
    ap.add_argument("--anchor", type=Path, default=D / "t0_bb_enc_both/fit.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--torch-seed", type=int, default=20260830)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.torch_seed)

    mean, std, mats = load_bin(args.bin)
    net = build_net(mats).to(dev)
    mu = torch.tensor(mean, device=dev)
    sig = torch.tensor(std, device=dev)

    def score(x):
        return net((x - mu) / sig).squeeze(-1)

    anchor = np.load(args.anchor)
    # 4.3M x 207 doesn't fit beside the model on an 8GB card; a deterministic
    # half-million-row sample anchors the landscape just as well.
    rng = np.random.default_rng(20260830)
    idx = rng.choice(len(anchor["x"]), size=min(500_000, len(anchor["x"])), replace=False)
    idx.sort()
    ax = torch.tensor(anchor["x"][idx], device=dev)
    ay = torch.tensor(anchor["y"][idx], device=dev)
    print(f"anchor rows {len(ax)} (of {len(anchor['x'])})", flush=True)

    roots = load_pairs(args.enc_dir)
    # Same split family as the corpus encoders: hash of the root id.
    def is_hold(rid):
        return int(hashlib.sha256(f"t0-evalfix-v1/{rid}".encode()).hexdigest()[:4], 16) % 100 < 10
    print(f"roots fit {sum(not is_hold(k) for k in roots)} "
          f"hold {sum(is_hold(k) for k in roots)}", flush=True)

    # The scored (600-particle) and escalated (2,400-particle) duels measured
    # the ONE pair that matters -- serving vs the referee's nominee -- far more
    # precisely than the 64-384-particle select means the listwise term uses.
    # Round 1 ignored this; worse, on model_better roots the select ordering
    # points the WRONG way (the nominee only looked better).  The override
    # teaches that pair from the precise margin, out-weighing the coarse term.
    prec = {}
    for l in open(args.enc_dir.parent / "t0_mine1/material.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "ci" in r:
            prec[r["id"]] = (r["margin"], 3.0 if r.get("escalated") else 2.0)

    def root_tensors(rid, group):
        xs = torch.tensor(np.stack([g["x"] for g in group]), device=dev)
        ms = torch.tensor([g["mean"] for g in group], device=dev)
        w = 1.0 if any(g["scored"] or g["escalated"] for g in group) else 0.5
        ov = None
        if rid in prec:
            si = next((i for i, g in enumerate(group) if g["is_serving"]), None)
            ri = next((i for i, g in enumerate(group) if g["is_ref"]), None)
            if si is not None and ri is not None and si != ri:
                ov = (si, ri, *prec[rid])
        return xs, ms, w, ov

    fit_t = [root_tensors(k, v) for k, v in roots.items() if not is_hold(k)]
    hold_t = [root_tensors(k, v) for k, v in roots.items() if is_hold(k)]

    def eval_hold():
        top1 = 0
        with torch.no_grad():
            for xs, ms, _, _ in hold_t:
                top1 += int(score(xs).argmax()) == int(ms.argmax())
        print(f"  hold top1(審判最善一致) {top1}/{len(hold_t)} = {top1/len(hold_t):.1%}", flush=True)
        return top1 / len(hold_t)

    huber = nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("before:", flush=True)
    eval_hold()
    order = list(range(len(fit_t)))
    g = torch.Generator().manual_seed(args.torch_seed)
    for ep in range(args.epochs):
        perm = torch.randperm(len(fit_t), generator=g).tolist()
        tot = 0.0
        for step, i in enumerate(perm):
            xs, ms, w, ov = fit_t[i]
            s = score(xs)
            # every ordered pair, hinge on the referee's gap, capped at 3
            diff = ms.unsqueeze(0) - ms.unsqueeze(1)
            gap = s.unsqueeze(0) - s.unsqueeze(1)
            mask = diff > 0
            if mask.any():
                pw = torch.clamp(diff[mask] / 2.0, max=3.0)
                loss = args.pair_weight * w * (pw * torch.relu(1.0 - gap[mask])).mean()
            else:
                loss = s.sum() * 0.0
            if ov is not None:
                si, ri, margin, ow = ov
                li = (s[ri] - s[si]) * (1.0 if margin > 0 else -1.0)
                ov_w = ow * min(1.0 + abs(margin) / 4.0, 3.0)
                loss = loss + args.pair_weight * ov_w * torch.relu(1.0 - li)
            bi = torch.randint(0, len(ax), (512,), generator=g).to(dev)
            loss = loss + args.anchor_weight * huber(score(ax[bi]), ay[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"ep{ep+1} loss {tot/len(fit_t):.4f}", flush=True)
        eval_hold()

    args.out.mkdir(parents=True, exist_ok=True)
    linears = [l for l in net if isinstance(l, nn.Linear)]
    ckpt = dict(
        schema="t4-first-evaluator-v1",
        model_state_dict={f"net.{i}": None for i in []},  # replaced below
        input_mean=torch.tensor(mean),
        input_std=torch.tensor(std),
        input_dim=len(mean),
        hidden=[l.out_features for l in linears[:-1]],
    )
    # Rebuild in T4FirstEvaluator's own module so the exporter loads it as-is.
    from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator
    canonical = T4FirstEvaluator(len(mean), tuple(ckpt["hidden"]))
    can_lin = [l for l in canonical.net if isinstance(l, nn.Linear)]
    with torch.no_grad():
        for src, dst in zip(linears, can_lin):
            dst.weight.copy_(src.weight.cpu())
            dst.bias.copy_(src.bias.cpu())
    ckpt["model_state_dict"] = canonical.state_dict()
    torch.save(ckpt, args.out / "evaluator_best.pt")
    print("saved ->", args.out / "evaluator_best.pt", flush=True)


if __name__ == "__main__":
    main()
