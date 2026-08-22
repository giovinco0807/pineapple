"""Fit a T3 evaluator on the M7 v4 relabelled corpus, one seat per run.

The protocol is the frozen M7 Task-0 one the T1/T2/T3 ablation arms ran under --
stable-hash holdout on the position code, masked padding everywhere, weighted
pairwise ranking plus a 0.1 regression anchor, Adam at 1e-3 cosined to zero,
and the epoch with the lowest HELD-OUT mean regret kept. Nothing here is tuned
per seat; the two seats differ only in their corpus.

Three things this adds over the ablation trainer, each because the retrain needs
it and the ablation did not:

* **Warm start that preserves the function.** A gen-1 checkpoint carries its own
  standardisation. Fitting fresh statistics and then loading old weights would
  silently feed the network differently-scaled inputs, so the first layer is
  re-expressed in the new statistics: the composed function is unchanged at
  load, which is what makes "warm start" mean what it says. Verified numerically
  rather than asserted, on real rows, before a single step is taken.
* **Nested subsets.** `--positions N` keeps the first N roots -- the corpus is
  in root order and the roots are i.i.d., so prefixes are unbiased samples and
  4k ⊂ 8k ⊂ 12k ⊂ 18k. The exam does not move: the held-out split is always
  every position of the FULL corpus with code >= split_code.
* **Degenerate accounting.** Positions where every candidate scores identically
  are dead positions, not broken labels, but regret on them is 0 by
  construction. Every held-out table is reported twice, with and without.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ofcdata import MLP, load_corpus, metrics, n_params, pack, split_masks  # noqa: E402


def dead_mask(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Positions whose every legal candidate carries the same label."""
    high = np.where(M, Y, -np.inf).max(axis=1)
    low = np.where(M, Y, np.inf).min(axis=1)
    return (high - low) <= 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="labelgen_features_v1 dump dir")
    ap.add_argument("--seat", required=True, choices=["first", "second"])
    ap.add_argument("--positions", type=int, default=0,
                    help="keep the first N roots (0 = all); the exam never moves")
    ap.add_argument("--init", default="", help="gen-1 checkpoint to warm start from")
    ap.add_argument("--hidden", default="256,128,64")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--split-code", type=int, default=230)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    # Standardised inputs are NOT clamped, and that is a decision with evidence
    # behind it rather than an omission. The engine's weight image (T4M1) stores
    # a mean and a std and nothing else; `Model::predict_with` computes
    # (x - mean) * inverse_std and feeds it straight to the first layer
    # (t4_model.rs:176). A model fitted on clamped inputs would therefore be run
    # unclamped inside the rollout — asked about rows it was never shown. On
    # this corpus 0.045 % of feature entries fall outside ±8 under the
    # incumbent's own statistics (one entry in ~2,200), including a dim reaching
    # |z| = 833, so the clamp is not the no-op it looks like: it also breaks the
    # exact warm-start transfer, which is how it was caught.
    ap.add_argument("--clamp", type=float, default=0.0,
                    help="0 disables; the engine does not clamp, so shipping "
                         "models are fitted unclamped")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label-provenance", default="")
    ap.add_argument("--out", required=True, help="path stem: writes .pt and .json")
    a = ap.parse_args()

    began = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y, groups, codes, meta = load_corpus(a.corpus)
    width = int(np.bincount(groups).max())
    X, Y, M, pos_code, counts = pack(x, y, groups, codes, width)
    del x, y, groups, codes
    P, W, D = X.shape

    train, held = split_masks(pos_code, a.split_code)
    if a.positions:
        keep = np.zeros(P, dtype=bool)
        keep[:a.positions] = True
        train = train & keep          # the subset only ever shrinks TRAINING
    dead = dead_mask(Y, M)

    print(f"[{a.seat}] corpus {a.corpus}")
    print(f"  positions {P:,}  width {W}  dim {D}  device {device}")
    print(f"  train {int(train.sum()):,}   held out {int(held.sum()):,} "
          f"(of which dead {int((dead & held).sum()):,})", flush=True)

    trained_slots = M & train[:, None]
    flat = X[trained_slots]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0            # near-constant dims must not blow up
    del flat

    Xs = (X - mean) / std
    clipped = 0.0
    if a.clamp > 0:
        clipped = float(np.mean(np.abs(Xs[trained_slots]) > a.clamp))
        Xs = np.clip(Xs, -a.clamp, a.clamp)
        print(f"  standardised, clamped at ±{a.clamp:g}: "
              f"{clipped:.4%} of training entries clipped", flush=True)

    hidden = [int(h) for h in a.hidden.split(",") if h]
    arch = [D] + hidden + [1]
    model = MLP(arch).to(device)

    warm = None
    if a.init:
        blob = torch.load(a.init, map_location="cpu", weights_only=False)
        old_mean = np.asarray(blob["mean"], dtype=np.float64)
        old_std = np.asarray(blob["std"], dtype=np.float64)
        if list(blob["architecture"]) != arch:
            raise SystemExit(f"warm start arch {blob['architecture']} != {arch}")
        state = {k: v.clone() for k, v in blob["state_dict"].items()}
        # Re-express layer 0 in the new statistics so the composed function is
        # unchanged: z_old = z_new * (std_new/std_old) + (mean_new-mean_old)/std_old.
        w0 = state["0.weight"].to(torch.float64).numpy()
        b0 = state["0.bias"].to(torch.float64).numpy()
        # A dimension the incumbent floored (std pinned at 1e-6) was CONSTANT in
        # its corpus, so its standardised input was identically zero and the
        # incumbent's function does not depend on it at all -- whatever w0 holds
        # there, the contribution is zero.  Rescaling by std_new/std_old would
        # multiply that meaningless weight by up to 1e6 and, the moment the new
        # corpus gives the dimension any spread, put 1e6 into the first layer.
        # That is the T1 first seat on 2026-08-17: dims 153 and 165 are constant
        # in the gen-1 corpus but carry std 0.0108 in the relabelled one, and
        # training diverged to loss 1.1e6 before settling back.
        # Preserving the function means zeroing those columns, not scaling them;
        # the new corpus teaches them from scratch.
        floored = old_std <= 1e-6
        scale = np.where(floored, 0.0, std / np.where(floored, 1.0, old_std))
        shift = np.where(floored, 0.0, (mean - old_mean) / np.where(floored, 1.0, old_std))
        w0_new = w0 * scale[None, :]
        b0_new = b0 + w0 @ shift
        if floored.any():
            print(f"  warm start: {int(floored.sum())} dimensions were constant "
                  f"for the incumbent and carry no transferable weight; their "
                  f"columns start at zero", flush=True)
        state["0.weight"] = torch.from_numpy(w0_new).to(torch.float32)
        state["0.bias"] = torch.from_numpy(b0_new).to(torch.float32)

        # Prove it on real rows rather than trusting the algebra.
        sample = X[M][:4096]
        old_net = MLP(arch)
        old_net.load_state_dict(blob["state_dict"])
        new_net = MLP(arch)
        new_net.load_state_dict(state)
        with torch.no_grad():
            # On a floored dimension the incumbent's own corpus fed it exactly
            # zero.  Dividing the new corpus's spread by that 1e-6 floor instead
            # would ask the incumbent about inputs of order 1e6 -- rows it never
            # saw and answers nonsense on -- and the comparison would measure
            # that, not the transfer.  Feed it the zero it was trained on.
            zo_raw = (sample - old_mean) / old_std
            zo_raw[:, floored] = 0.0
            zo = torch.from_numpy(zo_raw.astype(np.float32))
            zn = torch.from_numpy(((sample - mean) / std).astype(np.float32))
            if a.clamp > 0:
                zn = zn.clamp(-a.clamp, a.clamp)
                zo = zo.clamp(-a.clamp, a.clamp)
            drift = float((old_net(zo) - new_net(zn)).abs().max())
        model.load_state_dict(state)
        warm = {
            "checkpoint": a.init,
            "checkpoint_sha256": hashlib.sha256(
                pathlib.Path(a.init).read_bytes()).hexdigest(),
            "renormalisation_drift_maxabs": drift,
            "mean_shift_maxabs": float(np.abs(mean - old_mean).max()),
            "std_ratio_range": [float((std / old_std).min()),
                                float((std / old_std).max())],
        }
        print(f"  warm start {a.init}")
        print(f"    function drift after renormalisation {drift:.3e} "
              f"(mean shift max {warm['mean_shift_maxabs']:.3f}, "
              f"std ratio {warm['std_ratio_range'][0]:.3f}…"
              f"{warm['std_ratio_range'][1]:.3f})", flush=True)
        if drift > 1e-2:
            raise SystemExit("warm start did not preserve the function; refusing")

    Xg = torch.from_numpy(Xs.astype(np.float32)).to(device)
    Yg = torch.from_numpy(Y.astype(np.float32)).to(device)
    Mg = torch.from_numpy(M).to(device)
    train_idx = torch.from_numpy(np.nonzero(train)[0]).to(device)
    held_idx = torch.from_numpy(np.nonzero(held)[0]).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=a.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=a.epochs)
    print(f"  arch {arch}  params {n_params(arch):,}", flush=True)

    def score(idx, chunk=4096):
        model.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(idx), chunk):
                block = idx[start:start + chunk]
                outs.append(model(Xg[block].reshape(-1, D)).reshape(len(block), W))
        model.train()
        return torch.cat(outs)

    def evaluate(idx, subset=None):
        raw = score(idx).cpu().numpy().astype(np.float64)
        rows = idx.cpu().numpy()
        if subset is not None:
            take = subset[rows]
            raw, rows = raw[take], rows[take]
        return metrics(raw, Y[rows], M[rows])

    best = None
    history = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(a.seed + 1)
    n_train = len(train_idx)
    for epoch in range(1, a.epochs + 1):
        order = torch.randperm(n_train, generator=generator).to(device)
        total, batches = 0.0, 0
        for start in range(0, n_train, a.batch):
            block = train_idx[order[start:start + a.batch]]
            xb, yb, mb = Xg[block], Yg[block], Mg[block]
            scores = model(xb.reshape(-1, D)).reshape(len(block), W)

            gap = yb.unsqueeze(2) - yb.unsqueeze(1)
            margin = scores.unsqueeze(2) - scores.unsqueeze(1)
            pair = mb.unsqueeze(2) & mb.unsqueeze(1) & (gap > 0)
            weight = torch.where(pair, gap, torch.zeros_like(gap))
            ranking = (weight * torch.nn.functional.softplus(-margin)).sum() \
                / weight.sum().clamp_min(1e-9)
            squared = torch.where(mb, (scores - yb) ** 2, torch.zeros_like(scores))
            loss = ranking + a.reg_weight * squared.sum() / mb.sum().clamp_min(1)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss)
            batches += 1
        schedule.step()

        current = evaluate(held_idx)
        history.append({"epoch": epoch, "train_loss": total / batches,
                        "held_regret": current["mean_regret"],
                        "held_top1": current["top1_agreement"]})
        if best is None or current["mean_regret"] < best["held"]["mean_regret"]:
            best = {"epoch": epoch, "held": current,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep{epoch:4d} loss {total/batches:.5f} "
                  f"held regret {current['mean_regret']:.6f} "
                  f"top1 {current['top1_agreement']:.4f} "
                  f"best {best['held']['mean_regret']:.6f} @ {best['epoch']}",
                  flush=True)

    model.load_state_dict(best["state"])
    live = ~dead
    record = {
        "street": "T3", "seat": a.seat,
        "objective": "weighted pairwise ranking + 0.1 regression, masked",
        "corpus": a.corpus, "corpus_meta": meta,
        "label_provenance": a.label_provenance,
        "positions_kept": a.positions or P, "width": W, "feature_dim": D,
        "arch": arch, "params": n_params(arch),
        "seed": a.seed, "epochs": a.epochs, "kept_epoch": best["epoch"],
        "lr": a.lr, "batch": a.batch, "reg_weight": a.reg_weight,
        "clamp": a.clamp, "clipped_fraction": clipped,
        "split_code": a.split_code,
        "n_train_positions": int(train.sum()),
        "n_held_positions": int(held.sum()),
        "n_held_dead": int((dead & held).sum()),
        "warm_start": warm,
        "train": evaluate(train_idx),
        "held_out": evaluate(held_idx),
        "held_out_live_only": evaluate(held_idx, live),
        "history": history,
        "seconds": time.time() - began,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "mean": mean, "std": std,
                "architecture": arch, "clamp": a.clamp}, str(out) + ".pt")
    pathlib.Path(str(out) + ".json").write_text(json.dumps(record, indent=2),
                                                encoding="utf-8")
    held_metrics = record["held_out"]
    live_metrics = record["held_out_live_only"]
    print(f"[{a.seat}] DONE kept epoch {best['epoch']}  "
          f"held regret {held_metrics['mean_regret']:.6f} "
          f"(live only {live_metrics['mean_regret']:.6f})  "
          f"top1 {held_metrics['top1_agreement']:.4f}  "
          f"top3 {held_metrics['top3_agreement']:.4f}  "
          f"({record['seconds']:.0f}s) -> {out}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
