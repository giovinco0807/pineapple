"""Encode raced street labels into the street evaluator's serving vector.

Input: a roots file from `hu_street_roots` (one traced decision per row: the
hand, the slot, the served board) and its race labels from
`t0_btn_label --street S` (every legal placement priced by own_fl).  Output:
one 207-dim row per (root, placement), fit for `train_t4_first_evaluator
--init-from <bundle>/hu/t{S}_{seat}.bin`.

# The vector is the served one

`--hu-deep-replay --rollouts 0` scores the candidates of a traced decision
with the same encoding `play_hand` acts on: the pool is the deck minus the
pre-decision boards, hero's discards so far and hero's draw (`hu_match::
pool_of`, jokers by count), the opponent tail is the opponent's last placed
board, and the joint block is sampled at the ARM's serve count (200) under
the node seed `replay-rank/{street}/{seat}`.  This module assembles exactly
that vector through the `--joint-outlook` path (`encode_t0_material.
fetch_blocks`), and proves it on a sample of roots: the bundle's own bin,
run in Python over these rows, must reproduce the binary's score column to
float precision -- the same parity gate the T0 corpora passed at 1.1e-5.
A trained model therefore ranks at serve time as it ranked in training.

# Which board is which

Seat 0 (BB) acts first on every street: at (S, 0) the opponent's visible
board is the Button's placement from street S-1.  Seat 1 (BTN) acts after
the BB placed street S, and sees it.  Hero's "before" board is hero's own
placement from street S-1; hero's dead cards are the draws hero did not
place on streets 1..S-1.  All of it is read from the trace line the root
carries, so a corpus needs nothing but the roots and the labels.

Usage:
    python -m ai.tutor.build_street_sharp \\
        --roots D:/ofc_data/hu/street_sharp/roots_t1s0_pilot.jsonl \\
        --labels D:/ofc_data/hu/street_sharp/labels_p1fit.jsonl \\
        --models-dir D:/ofc_data/hu/models_ship_20260911 \\
        --out-dir D:/ofc_data/hu/street_sharp/enc_t1s0_pilot --split fit --parity-roots 30

Writes <out-dir>/<split>.npz (x, y, jokers, roots = hand numbers) and
<out-dir>/<split>_meta.jsonl (id, root, key, stratum, served, y, se, n).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl14_teacher import actor_block, allocation_rank_block, context_block
from ai.tutor.encode_fl_material import forward, load_bin, rows_of
from ai.tutor.encode_t0_material import (
    FEATURE_SIZE, REPO, RUST_NATURALS, SERVE_JOINT_SAMPLES, board_request, fetch_blocks)
from ai.tutor.t0_btn_label import Teacher, joker_blind

SEAT_NAME = {0: "bb", 1: "btn"}


def order(street: int, seat: int) -> int:
    return 2 * street + seat


def street_pool(own_before, opp, dead, draw) -> list[str]:
    """`hu_match::pool_of`: naturals removed by name, jokers by count, the
    unseen jokers appended as X1.. in the Rust deck order."""
    naturals: set[str] = set()
    jokers = 0
    for card in ([c for row in own_before for c in row] + [c for row in opp for c in row]
                 + list(dead) + list(draw)):
        if card.startswith("X"):
            jokers += 1
        elif card in naturals:
            raise AssertionError(f"card {card} appears twice")
        else:
            naturals.add(card)
    if jokers > 2:
        raise AssertionError(f"{jokers} jokers in view; the deck holds two")
    return [c for c in RUST_NATURALS if c not in naturals] + [f"X{i + 1}" for i in range(2 - jokers)]


def decision_of(root: dict) -> dict:
    """The boards the replay encodes for this root's decision."""
    street, seat = int(root["street"]), int(root["seat"])
    target = order(street, seat)
    steps = root["trace"]["steps"]
    own_before: list[list[str]] = [[], [], []]
    opp_rows: list[list[str]] = [[], [], []]
    best_own = best_opp = -1
    dead: list[str] = []
    for step in steps:
        s, t = int(step["street"]), int(step["seat"])
        o = order(s, t)
        if o >= target:
            continue
        if t == seat:
            if o > best_own:
                best_own, own_before = o, [list(r) for r in step["board"]]
            if s > 0:
                placed = [c for row in step["board"] for c in row]
                dead.extend(c for c in step["draw"] if c not in placed)
        else:
            if o > best_opp:
                best_opp, opp_rows = o, [list(r) for r in step["board"]]
    draw = list(root["draw"])
    return {"street": street, "seat": seat, "own_before": own_before, "opp": opp_rows,
            "dead": dead, "draw": draw,
            "pool": street_pool(own_before, opp_rows, dead, draw)}


def toss_of(rows, draw) -> str | None:
    placed = [c for row in rows for c in row]
    return next((c for c in draw if c not in placed), None)


def encode_rows(binary: Path, fl_ev: Path, rows, samples: int, seed: str, batch: int) -> np.ndarray:
    """207 dims per (own rows, opp rows, pool); the opponent half once per
    distinct (opp board, pool)."""
    out = np.empty((len(rows), FEATURE_SIZE), np.float32)
    for base in range(0, len(rows), batch):
        chunk = rows[base:base + batch]
        requests, tails = [], {}
        for index, (own, opp, pool) in enumerate(chunk):
            requests.append(board_request(f"o{index}", own, pool))
            tail_key = (tuple(tuple(r) for r in opp), tuple(pool))
            if tail_key not in tails:
                tails[tail_key] = f"t{len(tails)}"
                requests.append(board_request(tails[tail_key], opp, pool))
        blocks = fetch_blocks(binary, fl_ev, requests, samples, seed)
        for index, (own, opp, pool) in enumerate(chunk):
            own_rowwise, own_joint = blocks[f"o{index}"]
            opp_rowwise, opp_joint = blocks[tails[(tuple(tuple(r) for r in opp), tuple(pool))]]
            own_actor, _ = actor_block(own, pool)
            opp_actor, _ = actor_block(opp, pool)
            vector = (own_actor + [float(v) for v in own_rowwise] + [float(v) for v in own_joint]
                      + context_block(pool) + allocation_rank_block(own)
                      + opp_actor + [float(v) for v in opp_rowwise] + [float(v) for v in opp_joint])
            if len(vector) != FEATURE_SIZE:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            out[base + index] = vector
    return out


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="fit", help="output name: fit, dev or eval")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=None,
                    help="default: the bundle's fl_ev.json")
    ap.add_argument("--joint-samples", type=int, default=SERVE_JOINT_SAMPLES)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--parity-roots", type=int, default=30,
                    help="roots re-scored by the binary (--hu-deep-replay --rollouts 0) and "
                         "compared with the bundle bin run in Python over these rows")
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    ap.add_argument("--work", type=Path, default=Path("C:/tmp/street_build_work"))
    args = ap.parse_args()
    fl_ev = args.fl_ev_config or args.models_dir / "fl_ev.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roots = {r["id"]: r for r in load_rows(args.roots)}
    labels = [r for r in load_rows(args.labels) if "error" not in r and r["id"] in roots]
    skipped = sum(1 for r in load_rows(args.labels) if "error" in r)
    if not labels:
        raise SystemExit("no labelled roots match the roots file")
    slots = {(int(roots[r["id"]]["street"]), int(roots[r["id"]]["seat"])) for r in labels}
    if len(slots) != 1:
        raise SystemExit(f"one slot per corpus; found {sorted(slots)}")
    (street, seat), = slots
    seed = f"replay-rank/{street}/{seat}"
    contracts = {r.get("contract", "pre-contract-stamp") for r in labels}
    scores = {r.get("score") for r in labels}
    print(f"T{street}-{SEAT_NAME[seat].upper()}: {len(labels)} labelled roots ({skipped} error rows skipped), "
          f"contract {sorted(contracts)}, score {sorted(scores)}, node seed {seed}", flush=True)

    rows, meta = [], []
    for record in labels:
        root = roots[record["id"]]
        dec = decision_of(root)
        served_blind = joker_blind(root["served"])
        for key, value in record["field"].items():
            own = rows_of(key)
            if sum(len(r) for r in own) != sum(len(r) for r in dec["own_before"]) + len(dec["draw"]) - 1:
                raise AssertionError(f"{record['id']}: placement {key} does not place two of the draw")
            toss = toss_of(own, dec["draw"])
            if toss is None:
                raise AssertionError(f"{record['id']}: placement {key} tosses nothing")
            rows.append((own, dec["opp"], dec["pool"]))
            meta.append({"id": f"{record['id']}/{key}", "root": int(root["hand"]), "key": key,
                         "street": street, "seat": SEAT_NAME[seat], "stratum": root.get("stratum"),
                         "served": joker_blind(key) == served_blind, "y": float(value),
                         "se": (record.get("field_se") or {}).get(key),
                         "n": (record.get("field_n") or {}).get(key),
                         "jokers": sum(1 for r in own for c in r if c.startswith("X"))})
    started = time.time()
    x = encode_rows(args.binary, fl_ev, rows, args.joint_samples, seed, args.batch)
    print(f"encoded {len(x)} rows over {len(labels)} roots in {time.time() - started:.0f}s", flush=True)

    ship_bin = args.models_dir / "hu" / f"t{street}_{SEAT_NAME[seat]}.bin"
    mean, std, mats = load_bin(ship_bin)
    py_scores = forward(mats, (x - mean) / std)
    if args.parity_roots:
        args.work.mkdir(parents=True, exist_ok=True)
        teacher = Teacher(args.binary, args.models_dir, fl_ev, args.work, seat=seat, street=street)
        gap, compared, argmax_agree = 0.0, 0, 0
        by_root: dict[str, list[int]] = {}
        for index, m in enumerate(meta):
            by_root.setdefault(m["id"].rsplit("/", 1)[0], []).append(index)
        for n, (rid, indices) in enumerate(list(by_root.items())[:args.parity_roots]):
            root = roots[rid]
            teacher.hand_file = args.work / f"hand_{n}.jsonl"
            teacher.hand_file.write_text(json.dumps(dict(root["trace"], hand=0)) + "\n", encoding="utf-8")
            rust = {r["key"]: float(r["mean"])
                    for r in teacher.run(",".join(root["draw"]), "", 0, 0, None, f"parity_{n}")}
            py = {meta[i]["key"]: float(py_scores[i]) for i in indices}
            missing = [k for k in py if k not in rust]
            if missing:
                raise SystemExit(f"{rid}: {len(missing)} keys the binary did not enumerate: {missing[:3]}")
            gap = max(gap, max(abs(py[k] - rust[k]) for k in py))
            compared += len(py)
            argmax_agree += max(py, key=py.get) == max(rust, key=rust.get)
        print(f"parity over {len(list(by_root)[:args.parity_roots])} roots / {compared} rows: "
              f"max |py - rust| {gap:.3e}, argmax agrees {argmax_agree}/{min(len(by_root), args.parity_roots)}",
              flush=True)
        if gap > args.parity_tol:
            raise SystemExit(f"PARITY FAILED (tol {args.parity_tol:g}): the Python vector is not the served one")

    np.savez_compressed(args.out_dir / f"{args.split}.npz",
                        x=x, y=np.asarray([m["y"] for m in meta], np.float32),
                        jokers=np.asarray([m["jokers"] for m in meta], np.int8),
                        roots=np.asarray([m["root"] for m in meta], np.int64))
    with (args.out_dir / f"{args.split}_meta.jsonl").open("w", encoding="utf-8") as handle:
        for m in meta:
            handle.write(json.dumps(m) + "\n")
    manifest = {"schema": "ofc_hu_street_sharp_encoded/v1", "feature_size": FEATURE_SIZE,
                "slot": f"t{street}s{seat}", "node_seed": seed, "joint_samples": args.joint_samples,
                "contracts": sorted(contracts), "score": sorted(str(s) for s in scores),
                "roots": len(labels), "rows": len(meta), "labels": str(args.labels),
                "ship_bin": str(ship_bin)}
    # Per-root regret of the bundle bin's own argmax, priced by the labels: the
    # number the verdict tool reports for "ship", written here for a quick read.
    by_root_idx: dict[int, list[int]] = {}
    for i, m in enumerate(meta):
        by_root_idx.setdefault(m["root"], []).append(i)
    regrets = []
    for r, idx in by_root_idx.items():
        best = max(meta[i]["y"] for i in idx)
        pick = max(idx, key=lambda i: py_scores[i])
        regrets.append(best - meta[pick]["y"])
    manifest["ship_regret"] = float(np.mean(regrets))
    manifest["ship_regret_root_se"] = float(np.std(regrets, ddof=1) / np.sqrt(len(regrets)))
    (args.out_dir / f"{args.split}_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{args.split}: {len(meta)} rows, {len(by_root_idx)} roots -> {args.out_dir}; "
          f"ship bin regret {manifest['ship_regret']:.3f} ±{manifest['ship_regret_root_se']:.3f} per decision",
          flush=True)


if __name__ == "__main__":
    main()
