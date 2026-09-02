"""Teach the T0-BTN serving evaluator the referee's candidate ORDERING.

Button port of `train_fl_t0_correction.py` (itself the improved sibling of
`train_t0_eval_correction.py`, the T0-BB stage that gated +0.105/hand).  The
serving decision at Button is `hu/t0_btn.bin`'s argmax inside the ranker's
top-K fence over the 232 openings, so an audited error whose nominee sits in
the fence is an EVALUATOR error, and the ordering is what gets taught.
Values cannot be transplanted (the model predicts own-hand EV of a five-card
board against a five-card opponent board, the referee reports the mean
own-worth of a played-out hand), but the argmax only consumes the ordering.

Three terms, and the third is the one that matters:

* **anchor** -- a replay of the original 207-dim BTN corpus under its own
  SmoothL1 objective, so the value landscape the chain still reads does not
  drift while the ordering moves.
* **pairwise hinge** over each root's `sel_means`, weighted by the referee's
  own gap and capped, so a root the race barely separated cannot shout.
* **precise-pair override** -- for roots whose (serving, ref) pair was
  re-scored on fresh seeds, that one pair is taught from the scored margin's
  SIGN and magnitude at double weight.  This is what keeps `undecided` and
  `model_better` roots from teaching backwards: their select ordering puts
  the nominee on top (it won the race) while the precise re-score says
  serving was right or the difference is noise.

Roots are held out by id hash under the salt `t0-btn-evalfix-v1/`, so every
version of this model shares the split.  `--torch-seed` has no default: the
8/29 lesson is that unseeded runs of this recipe varied 21-52% on the same
material, so a run without a stated seed is not a measurement.

What is reported (holdout, and per root on request):

* `hold top1` -- argmax over the AUDITED field matches the referee's best.
* `error_fix` -- on holdout `error` roots, argmax over ALL 232 openings lands
  on the nominee / within 0.5 of it / on an UNAUDITED opening (the trap
  policy-v1 fell into: reordering the field while promoting a move nobody
  measured).
* `fence_fix` -- the same, restricted to the ranker fence, which is the
  decision serving actually makes at Button.

The exported `t0_btn.bin` (via `export_t4_first_evaluator`) is reloaded and
re-scored in numpy over every encoded row, so the file that ships is the
file that was measured.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.tutor.encode_fl_material import forward, load_bin
from ai.tutor.train_fl_t0_correction import build_net, find_anchor

D = Path("D:/ofc_data/hu")
DEFAULT_SALT = "t0-btn-evalfix-v1/"


def load_roots(enc_npz: Path, meta_path: Path):
    """Group the encoded placements by root, keeping every opening: the
    ordering terms use the audited slice, but "what does serving now pick" is
    the argmax over all of them (and over the fence), and a root that answers
    it needs its whole fan-out in the same tensor."""
    x = np.load(enc_npz)["x"]
    roots: dict[str, dict] = {}
    for index, line in enumerate(open(meta_path, encoding="utf-8")):
        m = json.loads(line)
        group = roots.setdefault(m["root"], dict(rows=[], meta=[], verdict=m["verdict"]))
        group["rows"].append(index)
        group["meta"].append(m)
    if len(roots) and len(x) != sum(len(g["rows"]) for g in roots.values()):
        raise SystemExit(f"{enc_npz} has {len(x)} rows, {meta_path} describes "
                         f"{sum(len(g['rows']) for g in roots.values())}")
    return x, roots


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=Path, default=D / "models_ship_20260903/hu/t0_btn.bin")
    ap.add_argument("--enc-dir", type=Path, default=D / "t0_btn_evalfix")
    ap.add_argument("--enc-npz", type=Path, default=None, help="defaults to <enc-dir>/enc_pairs.npz")
    ap.add_argument("--meta", type=Path, default=None, help="defaults to <enc-dir>/pairs_meta.jsonl")
    ap.add_argument("--anchor", type=Path, default=D / "t0_btn_enc_both/fit.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--anchor-rows", type=int, default=500_000)
    ap.add_argument("--anchor-seed", type=int, default=20260830,
                    help="the anchor subsample is drawn deterministically from this, "
                         "independent of --torch-seed, so seeds compare on one anchor")
    ap.add_argument("--undecided-weight", type=float, default=1.0,
                    help="scale on roots the referee could not separate; 1.0 is "
                         "the shipped BB recipe, 0.0 drops them from the teaching")
    ap.add_argument("--torch-seed", type=int, required=True,
                    help="required: unseeded runs of this recipe varied 21-52%% (8/29)")
    ap.add_argument("--split-salt", default=DEFAULT_SALT)
    ap.add_argument("--hold-pct", type=int, default=10)
    ap.add_argument("--export-name", default="t0_btn.bin")
    ap.add_argument("--dump-roots", action="store_true",
                    help="write roots.jsonl with every root's before/after picks (fit and hold)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.torch_seed)
    print(f"device {dev}, seed {args.torch_seed}, salt {args.split_salt}", flush=True)

    mean, std, mats = load_bin(args.bin)
    net = build_net(mats).to(dev)
    base = build_net(mats).to(dev)          # the shipped net, frozen, for before/after
    for p in base.parameters():
        p.requires_grad_(False)
    mu = torch.tensor(mean, device=dev)
    sig = torch.tensor(std, device=dev)

    def score(x, model=None):
        # Explicit None test: nn.Sequential is truthy by length.
        return (net if model is None else model)((x - mu) / sig).squeeze(-1)

    anchor_path = find_anchor(args.anchor)
    anchor = np.load(anchor_path)
    if anchor["x"].shape[1] != len(mean):
        raise SystemExit(f"anchor {anchor_path} is {anchor['x'].shape[1]}-dim, model is {len(mean)}")
    rng = np.random.default_rng(args.anchor_seed)
    take = min(args.anchor_rows, len(anchor["x"]))
    idx = rng.choice(len(anchor["x"]), size=take, replace=False)
    idx.sort()
    ax = torch.tensor(anchor["x"][idx], device=dev)
    ay = torch.tensor(anchor["y"][idx], device=dev)
    print(f"anchor {anchor_path}: {len(ax)} rows (of {len(anchor['x'])})", flush=True)
    del anchor

    x_all, roots = load_roots(args.enc_npz or args.enc_dir / "enc_pairs.npz",
                              args.meta or args.enc_dir / "pairs_meta.jsonl")
    if x_all.shape[1] != len(mean):
        raise SystemExit(f"encoded pairs are {x_all.shape[1]}-dim, model is {len(mean)}")
    xt = torch.tensor(x_all, device=dev)

    def is_hold(rid):
        return int(hashlib.sha256(f"{args.split_salt}{rid}".encode()).hexdigest()[:4], 16) % 100 < args.hold_pct

    undecided_weight = args.undecided_weight

    def pack(rid, group):
        rows = np.asarray(group["rows"])
        metas = group["meta"]
        sel = [i for i, m in enumerate(metas) if m["in_sel"]]
        if len(sel) < 2:
            return None
        means = torch.tensor([metas[i]["mean"] for i in sel], device=dev)
        # Coarse-vs-precise: the race means separate the field, the re-scored
        # margin settles the one pair the verdict is about.
        w = 1.0 if any(m["scored"] for m in metas) else 0.5
        # An `undecided` root is one whose re-score could not tell the two
        # apart, yet its select ordering still names a winner and its margin
        # still has a sign.  Both are noise pointed AWAY from the served pick.
        if group["verdict"] == "undecided":
            w *= undecided_weight
        ov = None
        si = next((k for k, i in enumerate(sel) if metas[i]["is_serving"]), None)
        ri = next((k for k, i in enumerate(sel) if metas[i]["is_ref"]), None)
        margin = next((m["margin"] for m in metas if "margin" in m), None)
        if margin is not None and si is not None and ri is not None and si != ri:
            ov = (si, ri, margin, 3.0 if any(m["escalated"] for m in metas) else 2.0)
        fence = [i for i, m in enumerate(metas) if m.get("in_fence")]
        return dict(rid=rid, sel=torch.tensor(rows[sel], device=dev), means=means,
                    all=torch.tensor(rows, device=dev), rows=rows, w=w, ov=ov,
                    ov_scale=(undecided_weight if group["verdict"] == "undecided" else 1.0),
                    verdict=group["verdict"],
                    keys=[m["key"] for m in metas],
                    sel_keys=[metas[i]["key"] for i in sel],
                    fence=fence,
                    served_key=next((m["key"] for m in metas if m["is_serving"]), None),
                    ref_key=next((m["key"] for m in metas if m["is_ref"]), None))

    fit_t, hold_t, thin = [], [], 0
    for rid, group in roots.items():
        packed = pack(rid, group)
        if packed is None:
            thin += 1
            continue
        (hold_t if is_hold(rid) else fit_t).append(packed)
    print(f"roots fit {len(fit_t)} hold {len(hold_t)} (thin {thin}; "
          f"{sum(len(r['means']) for r in fit_t + hold_t)} audited placements, "
          f"{len(x_all)} encoded)", flush=True)
    if not fit_t:
        raise SystemExit("nothing to fit")

    def hold_top1(model=None):
        """審判最善一致: does the net's argmax over the audited field match the
        referee's best mean?"""
        hits = []
        with torch.no_grad():
            for r in hold_t:
                s = score(xt[r["sel"]], model)
                hits.append(int(s.argmax()) == int(r["means"].argmax()))
        return np.asarray(hits, dtype=bool)

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
            if (verdicts == v).any()) or "(no holdout roots)"

    n_hold = max(len(hold_t), 1)
    print(f"before: hold top1(審判最善一致) {baseline.sum()}/{len(hold_t)} = "
          f"{baseline.sum()/n_hold:.1%}  ({wrong0.sum()} to recover)", flush=True)
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
              f"{now.sum()/n_hold:.1%}  recovery {recovered}/{int(wrong0.sum())} = "
              f"{recovered/max(int(wrong0.sum()),1):.1%}  broke {broke}", flush=True)
        print(f"        {by_verdict(now)}", flush=True)

    def picks(r, model=None):
        """(argmax over all 232, argmax inside the serving fence) as keys."""
        with torch.no_grad():
            s = score(xt[r["all"]], model).cpu().numpy()
        whole = r["keys"][int(s.argmax())]
        fenced = r["keys"][max(r["fence"], key=lambda i: float(s[i]))] if r["fence"] else whole
        return whole, fenced

    def fix_rate(model=None, fenced=False):
        """Unseen-error fix rate: on holdout roots the referee called an ERROR,
        does the argmax now land on the referee's nominee, or on a placement
        the referee measured within 0.5 of its best, or on an UNAUDITED
        opening (which is not a fix, it is a new unknown)?"""
        exact = near = unknown = 0
        errs = [r for r in hold_t if r["verdict"] == "error"]
        for r in errs:
            whole, fence = picks(r, model)
            pick = fence if fenced else whole
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

    report = dict(seed=args.torch_seed, epochs=args.epochs, lr=args.lr,
                  anchor=str(anchor_path), anchor_rows=int(len(ax)),
                  undecided_weight=args.undecided_weight, split_salt=args.split_salt,
                  hold_roots=len(hold_t), fit_roots=len(fit_t), thin_roots=thin,
                  top1_before=float(baseline.sum() / n_hold), top1_after=float(hold_top1().sum() / n_hold))
    for label, fenced in (("all232", False), ("fence", True)):
        n_err, e0, n0, u0 = fix_rate(base, fenced)
        _, e1, n1, u1 = fix_rate(None, fenced)
        print(f"unseen-error fix rate on {n_err} holdout error roots (argmax over {label}):", flush=True)
        print(f"  before: exact-ref {e0}/{n_err} = {e0/max(n_err,1):.1%}   "
              f"within-0.5 {n0}/{n_err} = {n0/max(n_err,1):.1%}   unmeasured pick {u0}", flush=True)
        print(f"  after : exact-ref {e1}/{n_err} = {e1/max(n_err,1):.1%}   "
              f"within-0.5 {n1}/{n_err} = {n1/max(n_err,1):.1%}   unmeasured pick {u1}", flush=True)
        report[f"error_fix_{label}"] = dict(error_roots=n_err, exact_before=e0, exact_after=e1,
                                            near_before=n0, near_after=n1,
                                            unmeasured_before=u0, unmeasured_after=u1)

    # Export: canonical module -> checkpoint -> T4F1 image, then the image is
    # read back and every encoded row re-scored in numpy against the torch
    # net, so what is measured below is the file that would ship.
    args.out.mkdir(parents=True, exist_ok=True)
    linears = [l for l in net if isinstance(l, nn.Linear)]
    hidden = [l.out_features for l in linears[:-1]]
    from ai.tutor.export_t4_first_evaluator import export
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
    info = export(args.out / "evaluator_best.pt", args.out / args.export_name)
    print(f"exported {info['path']} ({info['bytes']} bytes, sha256 {info['sha256'][:16]})", flush=True)

    mean2, std2, mats2 = load_bin(args.out / args.export_name)
    after_np = forward(mats2, (x_all - mean2) / std2)
    before_np = forward(mats, (x_all - mean) / std)
    with torch.no_grad():
        torch_after = score(xt).cpu().numpy()
    export_gap = float(np.abs(after_np - torch_after).max())
    print(f"exported bin vs torch net over {len(x_all)} rows: max|diff| {export_gap:.3e}", flush=True)

    # Per-root decisions under the shipped image and the exported one.  The
    # fenced column is the serving decision; the whole-set column is the trap
    # detector.  A root whose fenced pick moved off an `agree` verdict is a
    # regression the holdout numbers above cannot show if the root was fit.
    dump, moved, moved_agree, fence_to_ref = [], 0, 0, 0
    for r in fit_t + hold_t:
        rows = r["rows"]
        b, a = before_np[rows], after_np[rows]
        fence = r["fence"] or list(range(len(rows)))
        b_fence = r["keys"][max(fence, key=lambda i: float(b[i]))]
        a_fence = r["keys"][max(fence, key=lambda i: float(a[i]))]
        entry = dict(root=r["rid"], verdict=r["verdict"], hold=is_hold(r["rid"]),
                     served=r["served_key"], ref=r["ref_key"],
                     before_fence=b_fence, after_fence=a_fence,
                     before_all=r["keys"][int(b.argmax())], after_all=r["keys"][int(a.argmax())],
                     after_all_audited=r["keys"][int(a.argmax())] in r["sel_keys"])
        dump.append(entry)
        if a_fence != b_fence:
            moved += 1
            if r["verdict"] == "agree":
                moved_agree += 1
        if r["verdict"] == "error" and a_fence == r["ref_key"]:
            fence_to_ref += 1
    n_err_all = sum(r["verdict"] == "error" for r in fit_t + hold_t)
    print(f"exported bin, all {len(dump)} roots: fenced pick moved on {moved} "
          f"({moved_agree} of them `agree` roots); error roots now fenced to ref "
          f"{fence_to_ref}/{n_err_all}", flush=True)
    if args.dump_roots or len(dump) <= 20:
        with open(args.out / "roots.jsonl", "w", encoding="utf-8") as handle:
            for entry in dump:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if len(dump) <= 20:
            for entry in dump:
                print(f"  {entry['root']:>10} {entry['verdict']:<12} hold={int(entry['hold'])} "
                      f"fence {entry['before_fence']} -> {entry['after_fence']}"
                      f"{'  (moved)' if entry['before_fence'] != entry['after_fence'] else ''}"
                      f"  ref {entry['ref']}", flush=True)
    report.update(export=info, export_vs_torch_max_abs=export_gap,
                  fenced_moved=moved, fenced_moved_agree=moved_agree,
                  error_roots_all=n_err_all, error_fenced_to_ref=fence_to_ref)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved ->", args.out / "evaluator_best.pt", "and", args.out / args.export_name, flush=True)


if __name__ == "__main__":
    main()
