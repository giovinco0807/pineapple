"""Train a shipping model on a labelgen corpus, warm-started from the incumbent.

Same frozen protocol as the M7 ablation trainer -- stable-hash holdout on
position code, weighted pairwise ranking + 0.1 regression with masked padding,
Adam cosined to zero, keep the epoch with the lowest HELD-OUT mean regret.

What this adds is the warm start, and the part of it that matters is
RENORMALISATION. The incumbent was fitted against its own corpus's mean/std;
the new corpus has different ones. Loading the weights unchanged would feed the
network inputs on a different scale than it was fitted on. So the first layer is
transformed to compute the same function under the new standardization:

    z_old = (x - m_old)/s_old,  z_new = (x - m_new)/s_new
    z_old = z_new * (s_new/s_old) + (m_new - m_old)/s_old
    W' = W * (s_new/s_old)   (per input column)
    b' = b + W @ ((m_new - m_old)/s_old)

and the transform is then VERIFIED by scoring real rows through both forms and
recording the maximum absolute drift, rather than being trusted.

Usage:
  train_ship.py --corpus <dump-dir> --width N --seat first --street T2
                --warm-start <incumbent.pt> --seed N --out <prefix>
                [--epochs 120] [--batch 512] [--lr 1e-3] [--label-provenance "..."]
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/wner/ofc-m7/ablation/scripts")
from ofcdata import MLP, load_corpus, metrics, n_params, pack, split_masks  # noqa: E402


def sha256_of(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def renormalise(state, arch, old_mean, old_std, new_mean, new_std):
    """Move layer 0 from the incumbent's standardization onto the new one."""
    weight = np.asarray(state["0.weight"], dtype=np.float64)  # [out, in]
    bias = np.asarray(state["0.bias"], dtype=np.float64)
    ratio = new_std / old_std
    shift = (new_mean - old_mean) / old_std
    state = dict(state)
    state["0.weight"] = torch.from_numpy(
        (weight * ratio[None, :]).astype(np.float32))
    state["0.bias"] = torch.from_numpy(
        (bias + weight @ shift).astype(np.float32))
    return state, ratio, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--street", required=True)
    ap.add_argument("--seat", required=True)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--hidden", default="256,128,64")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--split-code", type=int, default=230)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--label-provenance", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x, y, groups, codes, corpus_meta = load_corpus(a.corpus)
    X, Y, M, pos_code, _ = pack(x, y, groups, codes, a.width)
    tr, va = split_masks(pos_code, a.split_code)
    P, W, D = X.shape
    print(f"[{a.street} {a.seat}] {a.corpus}  X={X.shape} "
          f"train={tr.sum()} held={va.sum()} dev={dev}", flush=True)

    trm = M & tr[:, None]
    flat_tr = X[trm]
    mean = flat_tr.mean(axis=0)
    std = flat_tr.std(axis=0)
    std[std < 1e-6] = 1.0

    hidden = [int(h) for h in a.hidden.split(",") if h]
    arch = [D] + hidden + [1]
    model = MLP(arch).to(dev)

    warm = None
    if a.warm_start:
        payload = torch.load(a.warm_start, map_location="cpu", weights_only=False)
        old_arch = list(payload["architecture"])
        if old_arch != arch:
            raise SystemExit(f"warm start arch {old_arch} != {arch}")
        old_mean = np.asarray(payload["mean"], dtype=np.float64)
        old_std = np.asarray(payload["std"], dtype=np.float64)
        state, ratio, shift = renormalise(
            payload["state_dict"], arch, old_mean, old_std,
            mean.astype(np.float64), std.astype(np.float64))

        # Verify rather than trust: score real held-out rows both ways.
        probe_rows = X[M][:4096].astype(np.float64)
        before = MLP(arch)
        before.load_state_dict(payload["state_dict"])
        after = MLP(arch)
        after.load_state_dict(state)
        with torch.no_grad():
            zb = torch.from_numpy(
                ((probe_rows - old_mean) / old_std).astype(np.float32))
            za = torch.from_numpy(
                ((probe_rows - mean) / std).astype(np.float32))
            drift = float((before(zb) - after(za)).abs().max())
        model.load_state_dict(state)
        model.to(dev)
        warm = {
            "checkpoint": str(a.warm_start),
            "checkpoint_sha256": sha256_of(a.warm_start),
            "renormalisation_drift_maxabs": drift,
            "mean_shift_maxabs": float(np.abs(mean - old_mean).max()),
            "std_ratio_range": [float(ratio.min()), float(ratio.max())],
        }
        print(f"  warm start drift {drift:.3e}  mean shift "
              f"{warm['mean_shift_maxabs']:.6f}", flush=True)

    Xs = (X - mean) / std
    Xg = torch.from_numpy(Xs.astype(np.float32)).to(dev)
    Yg = torch.from_numpy(Y.astype(np.float32)).to(dev)
    Mg = torch.from_numpy(M).to(dev)
    tr_idx = torch.from_numpy(np.nonzero(tr)[0]).to(dev)
    va_idx = torch.from_numpy(np.nonzero(va)[0]).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    print(f"  arch={arch} params={n_params(arch):,}", flush=True)

    def score(idx, bs=4096):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                outs.append(model(Xg[b].reshape(-1, D)).reshape(len(b), W))
        model.train()
        return torch.cat(outs)

    def evaluate(idx):
        s = score(idx).cpu().numpy().astype(np.float64)
        ii = idx.cpu().numpy()
        return metrics(s, Y[ii], M[ii])

    best = None
    hist = []
    ntr = len(tr_idx)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(a.seed + 1)
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(ntr, generator=gen).to(dev)
        tot, nb = 0.0, 0
        for i in range(0, ntr, a.batch):
            b = tr_idx[perm[i:i + a.batch]]
            xb, yb, mb = Xg[b], Yg[b], Mg[b]
            s = model(xb.reshape(-1, D)).reshape(len(b), W)
            ds = s.unsqueeze(2) - s.unsqueeze(1)
            dy = yb.unsqueeze(2) - yb.unsqueeze(1)
            pair = mb.unsqueeze(2) & mb.unsqueeze(1) & (dy > 0)
            w = torch.where(pair, dy, torch.zeros_like(dy))
            rank_loss = (w * torch.nn.functional.softplus(-ds)).sum() / w.sum().clamp_min(1e-9)
            se = torch.where(mb, (s - yb) ** 2, torch.zeros_like(s))
            reg_loss = se.sum() / mb.sum().clamp_min(1)
            loss = rank_loss + a.reg_weight * reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        vm = evaluate(va_idx)
        # Key names match the shipped records so the two are comparable.
        hist.append(dict(epoch=ep, train_loss=tot / nb,
                         held_regret=vm["mean_regret"],
                         held_top1=vm["top1_agreement"]))
        if best is None or vm["mean_regret"] < best["val"]["mean_regret"]:
            best = dict(epoch=ep, val=vm,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d} loss={tot/nb:.5f} val_regret={vm['mean_regret']:.6f} "
                  f"best={best['val']['mean_regret']:.6f} @{best['epoch']}", flush=True)

    model.load_state_dict(best["state"])
    train_m, val_m = evaluate(tr_idx), evaluate(va_idx)
    secs = time.time() - t0

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best["state"], mean=mean, std=std,
                    architecture=arch, clamp=0.0), str(out) + ".pt")
    rec = dict(street=a.street, seat=a.seat, corpus=a.corpus,
               corpus_meta=corpus_meta, width=a.width, feature_dim=D,
               arch=arch, params=n_params(arch), seed=a.seed, epochs=a.epochs,
               kept_epoch=best["epoch"], lr=a.lr, batch=a.batch,
               reg_weight=a.reg_weight, clamp=0.0, split_code=a.split_code,
               n_train_positions=int(tr.sum()), n_held_positions=int(va.sum()),
               warm_start=warm, label_provenance=a.label_provenance,
               train=train_m, held_out=val_m, history=hist, seconds=secs,
               built=time.strftime("%Y-%m-%dT%H:%M:%S"))
    pathlib.Path(str(out) + ".json").write_text(
        json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"DONE kept_epoch={best['epoch']} held_regret={val_m['mean_regret']:.6f} "
          f"top1={val_m['top1_agreement']:.4f} ({secs:.1f}s) -> {out}.json", flush=True)


if __name__ == "__main__":
    main()
