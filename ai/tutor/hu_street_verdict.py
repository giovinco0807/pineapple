"""Price street evaluators on a raced street set: per-decision regret by stratum.

Reads an encoding written by `build_street_sharp` (every legal placement of
every root, in the served vector) and scores one or more evaluators over it.
A model's pick is its argmax over all placements -- the unfenced serve of
contract v2 -- and its regret is the referee's best value minus the value of
that pick.  The paired difference against the shipped bin, with its root
standard error, is the verdict; the served row (the trace's own move, made
under the K=4 fence) is reported beside it as the champion's historical leak.

    python -m ai.tutor.hu_street_verdict --enc-dir D:/ofc_data/hu/street_sharp/enc_t1s0_eval --split eval \\
        --ship D:/ofc_data/hu/models_ship_20260911/hu/t1_bb.bin \\
        --cand D:/ofc_data/hu/street_sharp/model_t1s0_pilot_s1/t1_bb.bin ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.t0_btn_sharp_verdict import Scorer


def load(enc_dir: Path, split: str):
    z = np.load(enc_dir / f"{split}.npz")
    meta = [json.loads(l) for l in (enc_dir / f"{split}_meta.jsonl").open(encoding="utf-8") if l.strip()]
    if len(meta) != len(z["y"]):
        raise SystemExit(f"{split}: {len(meta)} meta rows against {len(z['y'])} encoded rows")
    return z["x"], z["y"].astype(np.float64), meta


def groups(meta) -> dict[int, list[int]]:
    by_root: dict[int, list[int]] = {}
    for i, m in enumerate(meta):
        by_root.setdefault(int(m["root"]), []).append(i)
    return by_root


def picks_of(scores: np.ndarray, by_root) -> dict[int, int]:
    return {r: max(idx, key=lambda i: scores[i]) for r, idx in by_root.items()}


def regrets_of(picks: dict[int, int], y: np.ndarray, by_root) -> dict[int, float]:
    return {r: float(max(y[i] for i in idx) - y[picks[r]]) for r, idx in by_root.items()}


def mean_se(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, np.float64)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc-dir", type=Path, required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--ship", required=True, help="the bundle's bin for this slot (T4F1 image)")
    ap.add_argument("--cand", nargs="*", default=[], help="candidate bins or trainer checkpoints")
    args = ap.parse_args()

    x, y, meta = load(args.enc_dir, args.split)
    by_root = groups(meta)
    stratum_of = {int(m["root"]): m.get("stratum") for m in meta}
    strata = {"all": list(by_root), "rand": [r for r in by_root if stratum_of[r] == "rand"],
              "joker": [r for r in by_root if stratum_of[r] == "joker"]}
    print(f"{args.split}: {len(meta)} rows, {len(by_root)} roots "
          f"(rand {len(strata['rand'])}, joker {len(strata['joker'])}), "
          f"{np.mean([len(v) for v in by_root.values()]):.1f} placements per root")

    served = {}
    for r, idx in by_root.items():
        hit = [i for i in idx if meta[i].get("served")]
        if hit:
            served[r] = hit[0]
    rows = [("served (trace, K=4)", served)] if len(served) == len(by_root) else []
    models = [("ship", Scorer(args.ship))] + [(Path(c).parent.name or Path(c).name, Scorer(c)) for c in args.cand]
    for name, scorer in models:
        rows.append((name, picks_of(scorer.scores(x), by_root)))
    ship_picks = rows[len(rows) - len(models)][1]
    ship_reg = regrets_of(ship_picks, y, by_root)

    width = 34
    head = f"{'model':<28}" + "".join(f"{s + ' (n=' + str(len(strata[s])) + ')':>{width}}" for s in strata)
    print(head + "   [regret ±root SE ; paired vs ship ±root SE ; picks agree]")
    for name, picks in rows:
        reg = regrets_of(picks, y, by_root)
        cells = []
        for s, roots in strata.items():
            m, se = mean_se([reg[r] for r in roots])
            if name == "ship":
                cells.append(f"{m:.3f} ±{se:.3f}")
            else:
                d, dse = mean_se([ship_reg[r] - reg[r] for r in roots])
                agree = sum(picks[r] == ship_picks[r] for r in roots)
                cells.append(f"{m:.3f} ; {d:+.3f} ±{dse:.3f} ; {agree}/{len(roots)}")
        print(f"{name:<28}" + "".join(f"{c:>{width}}" for c in cells))


if __name__ == "__main__":
    main()
