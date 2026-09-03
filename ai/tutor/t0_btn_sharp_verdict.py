"""Grade Button T0 evaluators on the sharp dev set, the T2 way.

Per dev root the labelled field (~40 openings) carries the referee's mean
from two independent passes (`build_t0_btn_sharp.py`).  A model's charged
regret at a root is label_max - label[the model's argmax]; the mean over
roots is the number, a bootstrap over roots its sampling CI, and the
label-noise SE rides on the pass split: the same pick graded on each pass,
half the difference per root (docs/t2_width128_20260831.md §6.5).

`--compare A B` is the paired verdict.  Per root d = v(A's pick) - v(B's
pick) under the pooled labels: the "true best" term cancels, so only roots
where the two picks differ contribute (the paired-difference identity) and
the bootstrap runs over every dev root with zeros on the agreeing ones.
Positive d means B is worse than A; the noise SE comes from the disputed
roots' pass split.

Two scopes are reported: `field`, the argmax over every labelled opening
(what `--select-on regret` trains against), and `fence<=K`, the argmax over
the ranker's top-K -- the decision serving actually makes at Button.

Scoring reproduces serving.  A T4F1 image is scored in float32 with its OWN
baked scaler; a trainer checkpoint with its stored input_mean/input_std in
torch float32.  Neither is ever recomputed from the arrays: the T2 campaign
lost a day to a numpy recompute that moved regret by 0.015 (§9).  Two
reference rules are accepted anywhere a model path is: `served` (the move
the champion played at that root) and `ranker` (the ranker's top-1).

Usage:
    python -m ai.tutor.t0_btn_sharp_verdict --dev D:/ofc_data/hu/t0_btn_sharp/dev.npz \\
        --model D:/ofc_data/hu/t0_btn_sharp/model/t0_btn.bin
    python -m ai.tutor.t0_btn_sharp_verdict --dev ... --compare <ship.bin> <candidate.bin>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ai.tutor.encode_fl_material import forward, load_bin
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

REFERENCES = ("served", "ranker")
CHUNK = 32_768


class Scorer:
    """One evaluator (T4F1 image or trainer checkpoint) or one reference rule."""

    def __init__(self, spec: str):
        self.spec = spec
        if spec in REFERENCES:
            self.kind = self.name = spec
            return
        path = Path(spec)
        # Every export is called t0_btn.bin; the parent directory tells them apart.
        self.name = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
        if path.read_bytes()[:4] == b"T4F1":
            self.kind = "image"
            self.mean, self.std, self.mats = load_bin(path)
            self.dim = len(self.mean)
        else:
            self.kind = "checkpoint"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.model = T4FirstEvaluator(int(checkpoint["input_dim"]), tuple(checkpoint["hidden"]))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.mean = checkpoint["input_mean"].to(torch.float32)
            self.std = checkpoint["input_std"].to(torch.float32)
            self.dim = int(checkpoint["input_dim"])

    @property
    def is_model(self) -> bool:
        return self.kind in ("image", "checkpoint")

    def scores(self, x: np.ndarray) -> np.ndarray:
        if not self.is_model:
            raise ValueError(f"{self.kind} is a rule, not a scorer")
        if x.shape[1] != self.dim:
            raise SystemExit(f"{self.spec} reads {self.dim} dims, the arrays are {x.shape[1]}-dim")
        if self.kind == "image":
            return forward(self.mats, (x - self.mean) / self.std).astype(np.float32)
        out = np.empty(len(x), np.float32)
        with torch.no_grad():
            for base in range(0, len(x), CHUNK):
                block = torch.from_numpy(x[base:base + CHUNK]).to(torch.float32)
                out[base:base + CHUNK] = self.model((block - self.mean) / self.std).numpy()
        return out


def groups_of(roots: np.ndarray) -> list[np.ndarray]:
    """Row indices per root, in the trainer's own grouping."""
    order = np.argsort(roots, kind="stable")
    bounds = np.flatnonzero(np.diff(roots[order])) + 1
    return [g for g in np.split(order, bounds) if g.size >= 2]


def picks_for(scorer: Scorer, x: np.ndarray, groups: list[np.ndarray], served: np.ndarray,
              ranker_rank: np.ndarray, fence: int) -> np.ndarray:
    """The row each root's decision lands on, inside the fence when one is given."""
    scores = scorer.scores(x) if scorer.is_model else None
    rank = np.where(ranker_rank > 0, ranker_rank, np.iinfo(np.int32).max)
    picks = np.empty(len(groups), np.int64)
    for i, g in enumerate(groups):
        if scorer.kind == "served":
            hit = g[served[g]]
            if hit.size == 0:
                raise SystemExit(f"root group {i} has no served row; the labeler forces it into the field")
            picks[i] = hit[0]
        elif scorer.kind == "ranker":
            picks[i] = g[int(np.argmin(rank[g]))]
        else:
            inside = g[rank[g] <= fence] if fence else g
            if inside.size == 0:
                inside = g
            picks[i] = inside[int(np.argmax(scores[inside]))]
    return picks


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    means = np.empty(draws)
    for base in range(0, draws, 1000):
        index = rng.integers(0, n, (min(1000, draws - base), n))
        means[base:base + len(index)] = values[index].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def grade(picks: np.ndarray, y: np.ndarray, passes, groups: list[np.ndarray]):
    """Per-root charged regret under the pooled labels, top-1 hits, and the
    label-noise SE of the mean regret (None without two passes)."""
    best = np.array([y[g].max() for g in groups])
    regret = best - y[picks]
    top1 = regret <= 0.0
    noise = None
    if passes is not None:
        y1, y2 = passes
        first = np.array([y1[g].max() for g in groups]) - y1[picks]
        second = np.array([y2[g].max() for g in groups]) - y2[picks]
        noise = float(np.sqrt((((first - second) / 2.0) ** 2).sum()) / len(groups))
    return regret, top1, noise


def paired(pick_a: np.ndarray, pick_b: np.ndarray, y: np.ndarray, passes):
    """d = v(A pick) - v(B pick) per root (zero where they agree) and the
    label-noise SE of its mean from the disputed roots' pass split."""
    d = y[pick_a] - y[pick_b]
    differ = pick_a != pick_b
    noise = None
    if passes is not None:
        y1, y2 = passes
        d1 = (y1[pick_a] - y1[pick_b])[differ]
        d2 = (y2[pick_a] - y2[pick_b])[differ]
        noise = float(np.sqrt((((d1 - d2) / 2.0) ** 2).sum()) / len(d))
    return d, differ, noise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True, help="dev.npz from build_t0_btn_sharp")
    ap.add_argument("--model", action="append", default=[],
                    help="T4F1 image, trainer checkpoint, `served` or `ranker`; repeatable")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="paired verdict of B against A (positive = B worse)")
    ap.add_argument("--fence", type=int, default=4,
                    help="ranker top-K for the fenced scope; 0 reports the field scope only")
    ap.add_argument("--no-references", action="store_true",
                    help="skip the served / ranker reference rows")
    ap.add_argument("--boots", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=None, help="write every number as json")
    args = ap.parse_args()
    if not args.model and not args.compare:
        ap.error("give --model and/or --compare")

    payload = np.load(args.dev)
    x = payload["x"]
    y = payload["y"].astype(np.float64)
    roots = payload["roots"]
    served = payload["served"] if "served" in payload else np.zeros(len(y), bool)
    ranker_rank = payload["ranker_rank"].astype(np.int64) if "ranker_rank" in payload \
        else np.zeros(len(y), np.int64)
    passes = None
    if "y1" in payload and "y2" in payload:
        passes = (payload["y1"].astype(np.float64), payload["y2"].astype(np.float64))
    groups = groups_of(roots)
    if not groups:
        raise SystemExit("dev has no root with two or more labelled openings")
    rng = np.random.default_rng(args.seed)
    scopes = [("field", 0)] + ([(f"fence<={args.fence}", args.fence)] if args.fence else [])
    print(f"dev {args.dev}: {len(groups)} roots, {len(y)} rows, "
          f"{'two passes (label-noise SE available)' if passes else 'ONE pass (no label-noise SE)'}",
          flush=True)
    report: dict = dict(dev=str(args.dev), roots=len(groups), rows=int(len(y)),
                        two_pass=passes is not None, fence=args.fence, models={}, compare=None)

    scorers = [Scorer(spec) for spec in args.model]
    if scorers and not args.no_references:
        scorers += [Scorer(spec) for spec in REFERENCES if spec not in args.model]
    if scorers:
        print(f"{'scope':<10} {'model':<34} {'regret':>8} {'95% CI':>20} {'noise SE':>9} "
              f"{'top-1':>7}", flush=True)
    for scope, fence in scopes:
        for scorer in scorers:
            if not scorer.is_model and scope != scopes[0][0]:
                continue  # a rule does not change with the fence
            picks = picks_for(scorer, x, groups, served, ranker_rank, fence)
            regret, top1, noise = grade(picks, y, passes, groups)
            lo, hi = bootstrap_ci(regret, rng, args.boots)
            label = scorer.name + ("" if scorer.is_model else " (reference)")
            print(f"{scope:<10} {label:<34} {regret.mean():8.4f} [{lo:8.4f}, {hi:8.4f}] "
                  f"{(f'{noise:9.4f}' if noise is not None else '        -')} "
                  f"{top1.mean():6.1%}", flush=True)
            report["models"].setdefault(scorer.name, {})[scope] = dict(
                regret=float(regret.mean()), ci=[lo, hi], label_noise_se=noise,
                top1=float(top1.mean()))

    if args.compare:
        base, challenger = (Scorer(spec) for spec in args.compare)
        print(f"\npaired: {challenger.name} vs {base.name} (d = v(A pick) - v(B pick); "
              f"positive = {challenger.name} worse)", flush=True)
        report["compare"] = dict(a=base.name, b=challenger.name, scopes={})
        for scope, fence in scopes:
            pick_a = picks_for(base, x, groups, served, ranker_rank, fence)
            pick_b = picks_for(challenger, x, groups, served, ranker_rank, fence)
            d, differ, noise = paired(pick_a, pick_b, y, passes)
            lo, hi = bootstrap_ci(d, rng, args.boots)
            verdict = "WORSE" if lo > 0 else ("BETTER" if hi < 0 else "unresolved")
            wins, losses = int((d < 0).sum()), int((d > 0).sum())
            print(f"  {scope:<10} d {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}  "
                  f"disputed {int(differ.sum())}/{len(d)} roots (B better {wins}, worse {losses})"
                  + (f"  label-noise SE {noise:.4f}" if noise is not None else ""), flush=True)
            report["compare"]["scopes"][scope] = dict(
                delta=float(d.mean()), ci=[lo, hi], verdict=verdict,
                disputed=int(differ.sum()), b_better=wins, b_worse=losses, label_noise_se=noise)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
