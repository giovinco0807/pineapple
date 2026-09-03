"""Pre-gate audit of a Button T0 evaluator over the 475 mined roots.

The sharp dev set grades the ordering inside the labelled field; this asks
the question the field cannot: what does SERVING pick when the candidate
scores all 232 openings and the ranker fence (top-K) selects?  Material is
`t0_mine --seat btn`'s referee verdicts (btnmine3: agree 236, error 102,
undecided 130, model_better 7); encodings are `encode_t0_material`'s
all-232 vectors (parity 1.1e-5 against the shipped image), so nothing here
re-invokes the solver.

Per candidate and root the fenced argmax is the candidate's best-scored
opening among those with ranker_rank <= K.  Reported against the referee:
  * error->ref   -- on error roots, fenced pick == ref_pick (the nominee)
  * agree kept   -- on agree roots, fenced pick == model_pick: the move the
                    champion played and the referee confirmed, which a
                    retrain must not break
  * unaudited    -- fenced pick outside sel_means on any root: neither
                    confirmed nor refuted (the trap policy-v1 fell into)
  * moved        -- fenced pick differs from the shipped image's
  * race dEV     -- sum over audited picks of sel_means[pick] -
                    sel_means[model_pick]: gain on error roots, damage on
                    agree roots, in the race's own units (noisy, SE ~2 per
                    candidate).  A proxy for reading candidates against each
                    other, not a verdict; the verdict is the mirror gate.

The shipped image is always the first row.  Its agree preservation is below
236/236: at Button `model_pick` is the move played under that hand's own
joint sample stream, and the audit's fenced argmax reproduces it on 382/475
(encode manifest), so the ship row is the floor every candidate is read
against.  The ship row also re-checks parity against the `own_score` column
the encoder published, which catches a wrong scaler before it costs a day.

Usage:
    python -m ai.tutor.t0_btn_sharp_audit --candidate D:/ofc_data/hu/t0_btn_sharp/model/t0_btn.bin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.t0_btn_sharp_verdict import Scorer

D = Path("D:/ofc_data/hu")
VERDICTS = ("error", "agree", "undecided", "model_better")


def load_material(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            rows[rec["id"]] = rec
    return rows


def load_meta(path: Path) -> dict[str, dict]:
    """Row indices, keys and ranker ranks of every encoded opening, per root."""
    roots: dict[str, dict] = {}
    for index, line in enumerate(path.open(encoding="utf-8")):
        if not line.strip():
            continue
        m = json.loads(line)
        group = roots.setdefault(m["root"], dict(rows=[], keys=[], ranker=[], own_score=[]))
        group["rows"].append(index)
        group["keys"].append(m["key"])
        group["ranker"].append(m.get("ranker_rank") or 0)
        group["own_score"].append(m["own_score"])
    for group in roots.values():
        group["rows"] = np.asarray(group["rows"])
        group["ranker"] = np.asarray(group["ranker"])
    return roots


def fenced_picks(scores: np.ndarray, material: dict, meta: dict, fence: int) -> dict[str, str]:
    picks = {}
    for rid in material:
        group = meta[rid]
        rank = group["ranker"]
        inside = np.flatnonzero((rank > 0) & (rank <= fence)) if fence else np.arange(len(rank))
        if inside.size == 0:
            inside = np.arange(len(rank))
        picks[rid] = group["keys"][inside[int(np.argmax(scores[group["rows"][inside]]))]]
    return picks


def audit(picks: dict[str, str], material: dict, ship_picks: dict[str, str] | None) -> dict:
    kept = {v: [0, 0] for v in VERDICTS}
    unaudited = moved = 0
    race = dict(error=0.0, agree=0.0, all=0.0)
    for rid, rec in material.items():
        key = picks[rid]
        verdict = rec["verdict"]
        target = rec["ref_pick"] if verdict == "error" else rec["model_pick"]
        kept[verdict][1] += 1
        kept[verdict][0] += key == target
        sel = rec["sel_means"]
        if key in sel:
            gain = float(sel[key] - sel[rec["model_pick"]])
            race["all"] += gain
            if verdict in race:
                race[verdict] += gain
        else:
            unaudited += 1
        if ship_picks is not None and ship_picks[rid] != key:
            moved += 1
    return dict(kept=kept, unaudited=unaudited, moved=moved, race=race)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", default=[],
                    help="T4F1 image or trainer checkpoint; repeatable")
    ap.add_argument("--ship", type=Path, default=D / "models_ship_20260903/hu/t0_btn.bin")
    ap.add_argument("--material", type=Path, default=D / "t0btn_mine/material_btnmine3.jsonl")
    ap.add_argument("--enc-dir", type=Path, default=D / "t0_btn_evalfix")
    ap.add_argument("--fence", type=int, default=4, help="ranker top-K serving selects from")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write per-root picks of every model as jsonl")
    args = ap.parse_args()

    material = load_material(args.material)
    meta = load_meta(args.enc_dir / "pairs_meta.jsonl")
    missing = [rid for rid in material if rid not in meta]
    if missing:
        raise SystemExit(f"{len(missing)} material roots lack encodings (first {missing[0]})")
    x = np.load(args.enc_dir / "enc_pairs.npz")["x"]
    counts = {v: sum(r["verdict"] == v for r in material.values()) for v in VERDICTS}
    print(f"{len(material)} roots ({', '.join(f'{v} {n}' for v, n in counts.items())}), "
          f"{len(x)} encoded openings, fence ranker<={args.fence}", flush=True)

    models = [("ship", Scorer(str(args.ship)))] + [(None, Scorer(spec)) for spec in args.candidate]
    scored = []
    for label, scorer in models:
        scores = scorer.scores(x)
        if label == "ship":
            published = np.concatenate([meta[rid]["own_score"] for rid in material])
            mine = np.concatenate([scores[meta[rid]["rows"]] for rid in material])
            gap = float(np.abs(published - mine).max())
            print(f"parity: ship scores vs encoder's own_score max|diff| {gap:.3e}"
                  + ("" if gap < 1e-3 else "  <-- NOT the serving scaler/weights"), flush=True)
        scored.append((label or scorer.name, scores))

    # Two scopes.  The fenced one is the serving decision; there every pick
    # is audited by construction (the mined field holds the ranker's top-16,
    # which covers any fence up to 16).  The whole-232 argmax is where a net
    # can escape the field, so `unaudited` only carries information there.
    all_picks: dict[str, dict[str, dict[str, str]]] = {}
    for scope, fence in ((f"fence<={args.fence}", args.fence), ("all232", 0)):
        print(f"\n[{scope}] {'model':<34} {'error->ref':>11} {'agree kept':>11} {'undec kept':>11} "
              f"{'mb kept':>8} {'unaudited':>10} {'moved':>6}   race dEV error / agree / all",
              flush=True)
        ship_picks = None
        for name, scores in scored:
            picks = fenced_picks(scores, material, meta, fence)
            all_picks.setdefault(name, {})[scope] = picks
            result = audit(picks, material, ship_picks)
            if ship_picks is None:
                ship_picks = picks
            kept = result["kept"]
            race = result["race"]
            print(f"{'':<{len(scope) + 3}}{name:<34} {kept['error'][0]:>4}/{kept['error'][1]:<6} "
                  f"{kept['agree'][0]:>4}/{kept['agree'][1]:<6} "
                  f"{kept['undecided'][0]:>4}/{kept['undecided'][1]:<6} "
                  f"{kept['model_better'][0]:>3}/{kept['model_better'][1]:<4} "
                  f"{result['unaudited']:>4}/{len(material):<5} "
                  f"{result['moved']:>6}   "
                  f"{race['error']:+8.1f} / {race['agree']:+8.1f} / {race['all']:+8.1f}", flush=True)

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as handle:
            for rid, rec in material.items():
                handle.write(json.dumps(dict(
                    root=rid, verdict=rec["verdict"], served=rec["model_pick"], ref=rec["ref_pick"],
                    picks={name: {scope: picks[rid] for scope, picks in scopes.items()}
                           for name, scopes in all_picks.items()},
                    audited={name: {scope: picks[rid] in rec["sel_means"]
                                    for scope, picks in scopes.items()}
                             for name, scopes in all_picks.items()}),
                    ensure_ascii=False) + "\n")
        print(f"wrote {args.dump}", flush=True)


if __name__ == "__main__":
    main()
