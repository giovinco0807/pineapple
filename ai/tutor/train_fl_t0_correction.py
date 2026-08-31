"""Teach the vs-FL T0 serving evaluator the referee's candidate ORDERING.

Sibling of `train_t0_eval_correction.py`, which does this for the normal-track
T0-BB evaluator; the recipe is that one's, and the reasons are the same.  The
serving decision on this path is `own_lap4/t0.bin`'s outright argmax over the
232 openings -- no ranker, no policy fence -- so an audited error is always an
EVALUATOR error and there is nothing else to retrain.  Values cannot be
transplanted (the model predicts own-hand EV of a five-card board, the referee
reports the mean own-worth of a played-out hand), but the argmax only consumes
the ordering, so the ordering is what gets taught.

Three terms, and the third is the one that matters:

* **anchor** -- a replay of the original 96-dim corpus under its own SmoothL1
  objective, so the value landscape the chain's T1/T2 search still reads does
  not drift while the ordering moves.
* **pairwise hinge** over each root's `sel_means`, weighted by the referee's
  own gap and capped, so a root the race barely separated cannot shout.
* **precise-pair override** -- for the 619 roots whose (serving, ref) pair was
  re-scored on fresh seeds, that one pair is taught from the scored margin's
  SIGN and magnitude at double weight.  This is what keeps `undecided` and
  `model_better` roots from teaching backwards: their select ordering puts the
  nominee on top (it won the race) while the precise re-score says serving was
  right or the difference is noise.  Without the override those 386 roots
  actively teach the wrong direction.

Roots are held out by id hash so every version of this model shares the split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.tutor.encode_fl_material import load_bin

D = Path("D:/ofc_data/hu")


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


def find_anchor(path: Path) -> Path:
    """The fit npz inside an encoded corpus directory, or the file itself."""
    if path.is_file():
        return path
    for name in ("fit.npz", "train.npz"):
        if (path / name).exists():
            return path / name
    found = sorted(path.glob("*.npz"))
    if not found:
        raise SystemExit(f"no npz under {path}")
    return found[0]


def load_roots(enc_npz: Path, meta_path: Path):
    """Group the encoded placements by root.

    Every opening is kept, not only the audited ones: the ordering terms use
    the audited slice, but the question "what does serving now pick" is about
    the argmax over all of them, and a root that answers it needs its whole
    fan-out in the same tensor.
    """
    x = np.load(enc_npz)["x"]
    roots: dict[str, dict] = {}
    for index, line in enumerate(open(meta_path, encoding="utf-8")):
        m = json.loads(line)
        group = roots.setdefault(m["root"], dict(rows=[], meta=[], verdict=m["verdict"]))
        group["rows"].append(index)
        group["meta"].append(m)
    return x, roots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=D / "models_ship_20260830/own_lap4/t0.bin")
    ap.add_argument("--enc-dir", type=Path, default=D / "fl_evalfix")
    # Overridable so the 96- and 110-dim arms run through one trainer: two
    # trainers would be two places for the loss to drift, and the whole point
    # of the comparison is that only the feature width changed.
    ap.add_argument("--enc-npz", type=Path, default=None,
                    help="defaults to <enc-dir>/enc_pairs.npz")
    ap.add_argument("--meta", type=Path, default=None,
                    help="defaults to <enc-dir>/pairs_meta.jsonl")
    ap.add_argument("--anchor", type=Path, default=Path("D:/ofc_data/lap4_t0_own/encoded_96"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--anchor-rows", type=int, default=500_000)
    ap.add_argument("--undecided-weight", type=float, default=1.0,
                    help="scale on roots the referee could not separate; 1.0 is "
                         "the shipped recipe, 0.0 drops them from the teaching")
    ap.add_argument("--torch-seed", type=int, default=20260830)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.torch_seed)
    print(f"device {dev}, seed {args.torch_seed}", flush=True)

    mean, std, mats = load_bin(args.bin)
    net = build_net(mats).to(dev)
    base = build_net(mats).to(dev)          # the shipped net, frozen, for before/after
    for p in base.parameters():
        p.requires_grad_(False)
    mu = torch.tensor(mean, device=dev)
    sig = torch.tensor(std, device=dev)

    def score(x, model=None):
        # Explicit None test: nn.Sequential is truthy by length, so `model or
        # net` would quietly fall through for an empty module.
        return (net if model is None else model)((x - mu) / sig).squeeze(-1)

    anchor_path = find_anchor(args.anchor)
    anchor = np.load(anchor_path)
    rng = np.random.default_rng(20260830)
    take = min(args.anchor_rows, len(anchor["x"]))
    idx = rng.choice(len(anchor["x"]), size=take, replace=False)
    idx.sort()
    ax = torch.tensor(anchor["x"][idx], device=dev)
    ay = torch.tensor(anchor["y"][idx], device=dev)
    print(f"anchor {anchor_path.name}: {len(ax)} rows (of {len(anchor['x'])})", flush=True)

    x_all, roots = load_roots(args.enc_npz or args.enc_dir / "enc_pairs.npz",
                              args.meta or args.enc_dir / "pairs_meta.jsonl")
    if x_all.shape[1] != len(mean):
        raise SystemExit(f"encoded pairs are {x_all.shape[1]}-dim, model is {len(mean)}")
    xt = torch.tensor(x_all, device=dev)

    def is_hold(rid):
        return int(hashlib.sha256(f"fl-t0-evalfix-v1/{rid}".encode()).hexdigest()[:4], 16) % 100 < 10

    undecided_weight = args.undecided_weight

    def pack(rid, group):
        rows = np.asarray(group["rows"])
        sel = [i for i, m in enumerate(group["meta"]) if m["in_sel"]]
        means = torch.tensor([group["meta"][i]["mean"] for i in sel], device=dev)
        sel_rows = torch.tensor(rows[sel], device=dev)
        all_rows = torch.tensor(rows, device=dev)
        # Coarse-vs-precise: the race means separate the field, the re-scored
        # margin settles the one pair the verdict is about.
        w = 1.0 if any(m["scored"] for m in group["meta"]) else 0.5
        # An `undecided` root is one whose re-score could not tell the two
        # apart, yet its select ordering still names a winner and its margin
        # still has a sign.  Both are noise pointed AWAY from the served pick,
        # and a third of the corpus is this.  The scale exists to measure that,
        # not to be tuned.
        if group["verdict"] == "undecided":
            w *= undecided_weight
        ov = None
        si = next((k for k, i in enumerate(sel) if group["meta"][i]["is_serving"]), None)
        ri = next((k for k, i in enumerate(sel) if group["meta"][i]["is_ref"]), None)
        margin = next((m["margin"] for m in group["meta"] if "margin" in m), None)
        if margin is not None and si is not None and ri is not None and si != ri:
            ov = (si, ri, margin, 2.0)
        return dict(rid=rid, sel=sel_rows, means=means, all=all_rows, w=w, ov=ov,
                    ov_scale=(undecided_weight if group["verdict"] == "undecided" else 1.0),
                    verdict=group["verdict"],
                    keys=[m["key"] for m in group["meta"]],
                    sel_keys=[group["meta"][i]["key"] for i in sel],
                    ref_key=next((m["key"] for m in group["meta"] if m["is_ref"]), None))

    fit_t, hold_t = [], []
    for rid, group in roots.items():
        packed = pack(rid, group)
        (hold_t if is_hold(rid) else fit_t).append(packed)
    print(f"roots fit {len(fit_t)} hold {len(hold_t)} "
          f"({sum(len(r['means']) for r in fit_t + hold_t)} audited placements, "
          f"{len(x_all)} encoded)", flush=True)

    def hold_top1(model=None):
        """審判最善一致: does the net's argmax over the audited field match the
        referee's best mean?"""
        hits = []
        with torch.no_grad():
            for r in hold_t:
                s = score(xt[r["sel"]], model)
                hits.append(int(s.argmax()) == int(r["means"].argmax()))
        return np.asarray(hits)

    verdicts = np.asarray([r["verdict"] for r in hold_t])
    baseline = hold_top1(base)
    wrong0 = ~baseline

    def by_verdict(hits):
        """Per-verdict agreement.  The headline number mixes three different
        questions: `agree` roots are already right and can only be broken,
        `undecided` roots are ones the referee itself could not separate, and
        only `error`/`model_better` roots carry a claim worth moving."""
        return "  ".join(
            f"{v} {int(hits[verdicts == v].sum())}/{int((verdicts == v).sum())}"
            for v in ("error", "model_better", "undecided", "agree")
            if (verdicts == v).any())

    print(f"before: hold top1(審判最善一致) {baseline.sum()}/{len(hold_t)} = "
          f"{baseline.mean():.1%}  ({wrong0.sum()} to recover)", flush=True)
    print(f"        {by_verdict(baseline)}", flush=True)

    huber = nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.torch_seed)
    for ep in range(args.epochs):
        perm = torch.randperm(len(fit_t), generator=g).tolist()
        tot = 0.0
        for i in perm:
            r = fit_t[i]
            s = score(xt[r["sel"]])
            ms = r["means"]
            # every ordered pair, hinge on the referee's gap, capped at 3
            diff = ms.unsqueeze(0) - ms.unsqueeze(1)
            gap = s.unsqueeze(0) - s.unsqueeze(1)
            mask = diff > 0
            if mask.any():
                pw = torch.clamp(diff[mask] / 2.0, max=3.0)
                loss = args.pair_weight * r["w"] * (pw * torch.relu(1.0 - gap[mask])).mean()
            else:
                loss = s.sum() * 0.0
            if r["ov"] is not None and r["ov_scale"]:
                si, ri, margin, ow = r["ov"]
                li = (s[ri] - s[si]) * (1.0 if margin > 0 else -1.0)
                ov_w = ow * min(1.0 + abs(margin) / 4.0, 3.0) * r["ov_scale"]
                loss = loss + args.pair_weight * ov_w * torch.relu(1.0 - li)
            bi = torch.randint(0, len(ax), (512,), generator=g).to(dev)
            loss = loss + args.anchor_weight * huber(score(ax[bi]), ay[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        now = hold_top1()
        recovered = int((now & wrong0).sum())
        broke = int((~now & baseline).sum())
        print(f"ep{ep+1} loss {tot/len(fit_t):.4f}  hold top1 {now.sum()}/{len(hold_t)} = "
              f"{now.mean():.1%}  recovery {recovered}/{int(wrong0.sum())} = "
              f"{recovered/max(int(wrong0.sum()),1):.1%}  broke {broke}", flush=True)
        print(f"        {by_verdict(now)}", flush=True)

    # Unseen-error fix rate: on holdout roots the referee called an ERROR, does
    # the argmax over EVERY opening now land on the referee's nominee, or on a
    # placement the referee measured within 0.5 of its best?  Argmax over the
    # audited field alone would not answer it -- serving reads all 232.
    def error_fix(model=None):
        exact = near = unknown = 0
        errs = [r for r in hold_t if r["verdict"] == "error"]
        for r in errs:
            with torch.no_grad():
                pick = r["keys"][int(score(xt[r["all"]], model).argmax())]
            if pick == r["ref_key"]:
                exact += 1
                near += 1
                continue
            if pick in r["sel_keys"]:
                k = r["sel_keys"].index(pick)
                if float(r["means"][k]) >= float(r["means"].max()) - 0.5:
                    near += 1
            else:
                unknown += 1
        return len(errs), exact, near, unknown

    n_err, e0, n0, u0 = error_fix(base)
    n_err2, e1, n1, u1 = error_fix()
    print(f"unseen-error fix rate on {n_err} holdout error roots (argmax over ALL openings):",
          flush=True)
    print(f"  before: exact-ref {e0}/{n_err} = {e0/max(n_err,1):.1%}   "
          f"within-0.5 {n0}/{n_err} = {n0/max(n_err,1):.1%}   unmeasured pick {u0}", flush=True)
    print(f"  after : exact-ref {e1}/{n_err2} = {e1/max(n_err2,1):.1%}   "
          f"within-0.5 {n1}/{n_err2} = {n1/max(n_err2,1):.1%}   unmeasured pick {u1}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    linears = [l for l in net if isinstance(l, nn.Linear)]
    hidden = [l.out_features for l in linears[:-1]]
    from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator
    canonical = T4FirstEvaluator(len(mean), tuple(hidden))
    can_lin = [l for l in canonical.net if isinstance(l, nn.Linear)]
    with torch.no_grad():
        for src, dst in zip(linears, can_lin):
            dst.weight.copy_(src.weight.cpu())
            dst.bias.copy_(src.bias.cpu())
    torch.save(dict(schema="t4-first-evaluator-v1",
                    model_state_dict=canonical.state_dict(),
                    input_mean=torch.tensor(mean), input_std=torch.tensor(std),
                    input_dim=len(mean), hidden=hidden),
               args.out / "evaluator_best.pt")
    (args.out / "report.json").write_text(json.dumps(dict(
        seed=args.torch_seed, epochs=args.epochs, lr=args.lr,
        hold_roots=len(hold_t), fit_roots=len(fit_t),
        top1_before=float(baseline.mean()), top1_after=float(hold_top1().mean()),
        error_roots=n_err, exact_before=e0, exact_after=e1,
        near_before=n0, near_after=n1), indent=2), encoding="utf-8")
    print("saved ->", args.out / "evaluator_best.pt", flush=True)


if __name__ == "__main__":
    main()
