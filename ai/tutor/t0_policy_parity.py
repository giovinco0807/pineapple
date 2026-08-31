"""Language-parity harness for the served T0-BB policy.

The policy speaks in action *indices*: a number that means "this canonical
card goes to that row".  Rust and Python each build that index from the dealt
five by their own copy of the canonicalisation, and the two copies agreeing
on a handful of hands proves nothing -- the rules that can drift (the suit
tie-break, the joker tail, the position order) only bite on particular hands.
So this compares the full ordering of all 232 openings, per hand, and calls
anything short of exact a failure.

Python side: `ai.tutor.train_t0_policy` itself, not a transcription of it.
Rust side: `--hu-t0-deep --rollouts 0` with `--hu-a-t0-policy`, whose
`policy_rank` column is written by the same code path that serves.

Usage:
    python -m ai.tutor.t0_policy_parity \
        --exe C:/tmp/t0policy_target/release/t4_first_exact.exe \
        --policy D:/ofc_data/hu/t0_policy_v1/policy.bin \
        --models-dir D:/ofc_data/hu/models_ship_20260827 \
        --hands 50
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ai.tutor.train_t0_policy import MASK, Policy, action_index, canonical, feats

ROW_CAPACITY = (3, 5, 5)


def openings(draw):
    """Every distinct arrangement of the opening five, keyed as Rust keys them."""
    seen = {}
    for mask in range(3 ** 5):
        rows = [[], [], []]
        code = mask
        legal = True
        for card in draw:
            row = code % 3
            code //= 3
            rows[row].append(card)
            if len(rows[row]) > ROW_CAPACITY[row]:
                legal = False
                break
        if not legal:
            continue
        key = "|".join(",".join(sorted(row)) for row in rows)
        seen.setdefault(key, rows)
    return seen


def python_ranks(net, draw):
    """key -> (logit, rank) under the training script's own canonicalisation."""
    canon, mapping = canonical(draw)
    with torch.no_grad():
        logits = net(torch.from_numpy(feats(canon)).unsqueeze(0))[0].numpy()
    rows = []
    for key in openings(draw):
        index = action_index(key, canon, mapping)
        if index is None or not MASK[index]:
            raise SystemExit(f"opening {key} maps to no legal action")
        rows.append((float(logits[index]), key))
    # Descending, stable -- the serve path's comparator.
    order = sorted(range(len(rows)), key=lambda i: -rows[i][0])
    return {rows[i][1]: (rows[i][0], place + 1) for place, i in enumerate(order)}


def eight(directory: Path, kind: str) -> str:
    names = [f"t{s}_{seat}" for s in range(4) for seat in ("bb", "btn")]
    return ",".join(str(directory / kind / f"{n}.bin") for n in names)


def rust_ranks(exe, policy, models, fl_ev, draw, topk):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rank.jsonl"
        own = ",".join(str(models / "own_lap4" / f"t{i}.bin") for i in range(3))
        argv = [
            str(exe), "--hu-match", "--hu-t0-deep",
            "--t0-cards", ",".join(draw), "--rollouts", "0", "--self-play-seed", "1",
            "--hu-a-models", eight(models, "hu"), "--hu-b-models", eight(models, "hu"),
            "--hu-a-rankers", eight(models, "rankers"),
            "--hu-b-rankers", eight(models, "rankers"),
            "--hu-topk", "4",
            "--serve-joint-samples", "1", "--serve-joint-samples-b", "1",
            "--arm-a-own", own, "--arm-b-own", own,
            "--hu-a-t0-policy", str(policy), "--hu-t0-policy-topk", str(topk),
            "--fl-ev-config", str(fl_ev), "--output", str(out),
        ]
        done = subprocess.run(argv, capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"rust failed: {done.stderr[-2000:]}")
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["key"]: (r["policy_score"], r["policy_rank"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--policy", type=Path,
                    default=Path("D:/ofc_data/hu/t0_policy_v1/policy.bin"))
    ap.add_argument("--model", type=Path,
                    default=Path("D:/ofc_data/hu/t0_policy_v1/policy_best.pt"))
    ap.add_argument("--models-dir", type=Path,
                    default=Path("D:/ofc_data/hu/models_ship_20260827"))
    ap.add_argument("--fl-ev-config", type=Path,
                    default=Path("ai/config/fl_ev.json"))
    ap.add_argument("--requests", type=Path,
                    default=Path("D:/ofc_data/hu/onpol_requests/t0_bb_onpol.jsonl"))
    ap.add_argument("--hands", type=int, default=50)
    ap.add_argument("--jokers", type=int, default=8,
                    help="how many of the sample must contain a joker")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    draws = [json.loads(l)["draw"]
             for l in args.requests.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    with_joker = [d for d in draws if any(c.startswith("X") for c in d)]
    without = [d for d in draws if not any(c.startswith("X") for c in d)]
    sample = (rng.sample(with_joker, min(args.jokers, len(with_joker)))
              + rng.sample(without, args.hands - min(args.jokers, len(with_joker))))
    rng.shuffle(sample)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    net = Policy()
    net.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    net.eval()

    agreed = 0
    worst_logit = 0.0
    for number, draw in enumerate(sample, 1):
        want = python_ranks(net, draw)
        got = rust_ranks(args.exe, args.policy, args.models_dir,
                         args.fl_ev_config, draw, args.topk)
        if set(want) != set(got):
            print(f"[{number}] {','.join(draw)}: key sets differ "
                  f"({len(want)} python, {len(got)} rust)")
            continue
        bad = [k for k in want if want[k][1] != got[k][1]]
        gap = max(abs(want[k][0] - got[k][0]) for k in want)
        worst_logit = max(worst_logit, gap)
        if bad:
            print(f"[{number}] {','.join(draw)}: {len(bad)}/232 ranks differ, "
                  f"e.g. {bad[0]} python#{want[bad[0]][1]} rust#{got[bad[0]][1]}")
        else:
            agreed += 1
            print(f"[{number}] {','.join(draw)}: 232/232 ok "
                  f"(max |logit gap| {gap:.2e})", flush=True)

    print(f"\nparity: {agreed}/{len(sample)} hands with all 232 ranks identical; "
          f"worst per-action logit gap {worst_logit:.3e}")
    if agreed != len(sample):
        raise SystemExit("the two canonicalisations disagree")


if __name__ == "__main__":
    main()
