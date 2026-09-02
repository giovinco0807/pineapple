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

`--seat btn` does the same for the Button net (`ai.tutor.train_t0_btn_policy`):
the state is (hero five, BB's placed board), fed to Rust as `--hu-t0-seat 1
--t0-opp-board ... --hu-a-t0-btn-policy <bin> --hu-t0-btn-policy-topk K`.
The joint suit canonicalisation and the opponent-side joker counting are the
rules that can drift there, so the sample carries jokers on both sides,
including a hero holding X2 alone (Rust's `normalise_hero` respells it X1 in
every key; the Python side applies the same rename before spelling keys).

Usage:
    python -m ai.tutor.t0_policy_parity \
        --exe C:/tmp/t0policy_target/release/t4_first_exact.exe \
        --policy D:/ofc_data/hu/t0_policy_v1/policy.bin \
        --models-dir D:/ofc_data/hu/models_ship_20260827 \
        --hands 50
    python -m ai.tutor.t0_policy_parity --seat btn \
        --exe C:/tmp/t0policy_target/release/t4_first_exact.exe \
        --policy D:/ofc_data/hu/t0_btn_policy_s1/policy.bin \
        --model  D:/ofc_data/hu/t0_btn_policy_s1/policy_best.pt \
        --hands 50
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ai.tutor.train_t0_policy import MASK, Policy, action_index, canonical, feats
from ai.tutor.train_t0_btn_policy import (
    board_spec, encode_state, policy_from_state, rename_hero_jokers,
)

ROW_CAPACITY = (3, 5, 5)
BTN_FLAG = "--hu-a-t0-btn-policy"
BTN_TOPK_FLAG = "--hu-t0-btn-policy-topk"


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


def python_ranks_btn(net, draw, opp_board):
    """key -> (logit, rank) for the Button net.

    Keys are spelled with hero jokers renamed in deal order, which is how
    Rust's `normalise_hero` spells them in the rows it writes; the encoder's
    own renumbering (sorted original name) is then the identity.
    """
    hero = rename_hero_jokers(draw)
    x, canon, mapping = encode_state(hero, opp_board)
    with torch.no_grad():
        logits = net(torch.from_numpy(x).unsqueeze(0))[0].numpy()
    rows = []
    for key in openings(hero):
        index = action_index(key, canon, mapping)
        if index is None or not MASK[index]:
            raise SystemExit(f"opening {key} maps to no legal action")
        rows.append((float(logits[index]), key))
    order = sorted(range(len(rows)), key=lambda i: -rows[i][0])
    return {rows[i][1]: (rows[i][0], place + 1) for place, i in enumerate(order)}


def eight(directory: Path, kind: str) -> str:
    names = [f"t{s}_{seat}" for s in range(4) for seat in ("bb", "btn")]
    return ",".join(str(directory / kind / f"{n}.bin") for n in names)


def rust_ranks(exe, policy, models, fl_ev, draw, topk, opp_board=None, keys_only=False):
    """policy_score / policy_rank per opening key from the Rust deep path.

    `opp_board` given selects the Button decision state (seat 1) and the
    Button policy flags; absent, the BB flags exactly as before.  `keys_only`
    omits the policy flags (Button): the run then only proves that Rust
    spells the same 232 keys Python does, which a build without the Button
    policy can already do.
    """
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
            "--fl-ev-config", str(fl_ev), "--output", str(out),
        ]
        if opp_board is None:
            argv += ["--hu-a-t0-policy", str(policy), "--hu-t0-policy-topk", str(topk)]
        else:
            argv += ["--hu-t0-seat", "1", "--t0-opp-board", board_spec(opp_board)]
            if not keys_only:
                argv += [BTN_FLAG, str(policy), BTN_TOPK_FLAG, str(topk)]
        done = subprocess.run(argv, capture_output=True, text=True)
        if done.returncode != 0:
            unexpected = re.search(r"unexpected argument '([^']+)'", done.stderr)
            if unexpected:
                raise SystemExit(
                    f"the binary {exe} does not take {unexpected.group(1)}: it predates the "
                    f"Button T0 flags ({BTN_FLAG} / {BTN_TOPK_FLAG}; --hu-t0-seat / "
                    "--t0-opp-board are in the current source).  The harness is ready -- "
                    "rerun against a build that has them.\n" + done.stderr[-600:])
            raise SystemExit(f"rust failed: {done.stderr[-2000:]}")
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    ranks = {r["key"]: (r["policy_score"], r["policy_rank"]) for r in rows}
    if rows and not keys_only and all(v[1] is None for v in ranks.values()):
        raise SystemExit(
            "rust wrote no policy_rank for any opening: the policy was not applied at this "
            f"seat (stderr tail: {done.stderr[-400:]})")
    return ranks


def has_joker(cards) -> bool:
    return any(c.startswith("X") for c in cards)


def sample_btn(rows, hands, jokers, rng):
    """`jokers` states carrying a joker, split between the hero side and the
    opponent side, the rest clean.  Hero-side picks prefer the spellings that
    exercise the rename: one hero holding both jokers, then heroes holding
    X2 alone (BB has X1), then X1 alone."""
    hero_side = [r for r in rows if has_joker(r["draw"])]
    opp_side = [r for r in rows if not has_joker(r["draw"]) and has_joker(sum(r["opp_board"], []))]
    clean = [r for r in rows if not has_joker(r["draw"] + sum(r["opp_board"], []))]

    def take(pool, n, chosen):
        pool = [r for r in pool if r["id"] not in {c["id"] for c in chosen}]
        return rng.sample(pool, max(0, min(n, len(pool))))

    want_hero = jokers // 2
    want_opp = jokers - want_hero
    picked = []
    for bucket, n in (
        ([r for r in hero_side if r["draw"].count("X1") + r["draw"].count("X2") == 2], 1),
        ([r for r in hero_side if "X2" in r["draw"] and "X1" not in r["draw"]], 2),
        (hero_side, want_hero),
    ):
        picked += take(bucket, min(n, want_hero - len(picked)), picked)
    n_hero = len(picked)
    for bucket, n in (
        ([r for r in opp_side if sum(has_joker([c]) for c in sum(r["opp_board"], [])) == 2], 1),
        (opp_side, want_opp),
    ):
        picked += take(bucket, min(n, want_opp - (len(picked) - n_hero)), picked)
    n_opp = len(picked) - n_hero
    picked += take(clean, hands - len(picked), picked)
    rng.shuffle(picked)
    return picked, n_hero, n_opp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--seat", choices=("bb", "btn"), default="bb")
    ap.add_argument("--policy", type=Path, default=None,
                    help="T4F1 image (default by seat: t0_policy_v1 / t0_btn_policy_s1)")
    ap.add_argument("--model", type=Path, default=None,
                    help="the torch checkpoint the image was exported from")
    ap.add_argument("--models-dir", type=Path,
                    default=Path("D:/ofc_data/hu/models_ship_20260827"))
    ap.add_argument("--fl-ev-config", type=Path,
                    default=Path("ai/config/fl_ev.json"))
    ap.add_argument("--requests", type=Path, default=None,
                    help="request rows; BTN rows carry the opp_board (default by seat)")
    ap.add_argument("--hands", type=int, default=50)
    ap.add_argument("--jokers", type=int, default=8,
                    help="how many of the sample must contain a joker (BTN: split hero/opp side)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--tie-tol", type=float, default=1e-5,
                    help="two keys whose Python logits are within this of each other may "
                         "legitimately swap places across f32 accumulation orders; such swaps "
                         "are reported but are not a canonicalisation disagreement")
    ap.add_argument("--keys-only", action="store_true",
                    help="BTN: run Rust without the policy flags and only require the same "
                         "232 keys -- what a build without the Button policy can already prove")
    args = ap.parse_args()
    if args.keys_only and args.seat != "btn":
        ap.error("--keys-only is a BTN mode")
    defaults = {
        "bb": ("D:/ofc_data/hu/t0_policy_v1", "D:/ofc_data/hu/onpol_requests/t0_bb_onpol.jsonl"),
        "btn": ("D:/ofc_data/hu/t0_btn_policy_s1", "D:/ofc_data/hu/onpol2_requests/t0_btn_lap2.jsonl"),
    }[args.seat]
    args.policy = args.policy or Path(defaults[0]) / "policy.bin"
    args.model = args.model or Path(defaults[0]) / "policy_best.pt"
    args.requests = args.requests or Path(defaults[1])

    rows = [json.loads(l) for l in args.requests.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    if args.seat == "bb":
        draws = [r["draw"] for r in rows]
        with_joker = [d for d in draws if has_joker(d)]
        without = [d for d in draws if not has_joker(d)]
        sample = (rng.sample(with_joker, min(args.jokers, len(with_joker)))
                  + rng.sample(without, args.hands - min(args.jokers, len(with_joker))))
        rng.shuffle(sample)
        states = [(d, None) for d in sample]
    else:
        picked, n_hero, n_opp = sample_btn(rows, args.hands, args.jokers, rng)
        states = [(r["draw"], r["opp_board"]) for r in picked]
        print(f"BTN sample: {len(states)} states, jokers on the hero side {n_hero}, "
              f"on the opponent side {n_opp}")

    if not states:
        raise SystemExit(f"nothing to compare: {args.requests} yielded no states for the sample")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    net = Policy() if args.seat == "bb" else policy_from_state(state)
    net.load_state_dict(state)
    net.eval()

    agreed = strict = 0
    tie_swaps = 0
    worst_logit = 0.0
    for number, (draw, opp_board) in enumerate(states, 1):
        label = ",".join(draw) + ("" if opp_board is None else f" vs {board_spec(opp_board)}")
        want = python_ranks(net, draw) if opp_board is None else python_ranks_btn(net, draw, opp_board)
        got = rust_ranks(args.exe, args.policy, args.models_dir,
                         args.fl_ev_config, draw, args.topk, opp_board, args.keys_only)
        if set(want) != set(got):
            only_py = sorted(set(want) - set(got))[:2]
            only_rs = sorted(set(got) - set(want))[:2]
            print(f"[{number}] {label}: key sets differ "
                  f"({len(want)} python, {len(got)} rust; python-only {only_py}, rust-only {only_rs})")
            continue
        if args.keys_only:
            agreed += 1
            strict += 1
            print(f"[{number}] {label}: 232/232 keys spelled identically (ranks not compared)", flush=True)
            continue
        differ = [k for k in want if want[k][1] != got[k][1]]
        # A swapped pair whose Python logits are tied within --tie-tol says
        # nothing about the encoders: f32 sums in a different order put
        # either first, on the Rust side as much as here.
        rust_at = {rank: k for k, (_, rank) in got.items()}
        bad = [k for k in differ
               if abs(want[k][0] - want[rust_at[want[k][1]]][0]) > args.tie_tol]
        gap = max(abs(want[k][0] - got[k][0]) for k in want)
        worst_logit = max(worst_logit, gap)
        if bad:
            print(f"[{number}] {label}: {len(bad)}/232 ranks differ, "
                  f"e.g. {bad[0]} python#{want[bad[0]][1]} rust#{got[bad[0]][1]}")
        elif differ:
            agreed += 1
            tie_swaps += len(differ)
            print(f"[{number}] {label}: 232/232 up to {len(differ)} rank swaps between logits "
                  f"tied within {args.tie_tol:g} (max |logit gap| {gap:.2e})", flush=True)
        else:
            agreed += 1
            strict += 1
            print(f"[{number}] {label}: 232/232 ok "
                  f"(max |logit gap| {gap:.2e})", flush=True)

    print(f"\nparity ({args.seat}): {agreed}/{len(states)} states with all 232 ranks identical "
          f"({strict} strictly, {tie_swaps} rank swaps on the rest between logits tied within "
          f"{args.tie_tol:g}); worst per-action logit gap {worst_logit:.3e}")
    if agreed != len(states):
        raise SystemExit("the two canonicalisations disagree")


if __name__ == "__main__":
    main()
