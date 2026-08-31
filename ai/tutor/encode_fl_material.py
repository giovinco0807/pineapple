"""Encode every audited vs-FL T0 placement into the 96-dim serving vector.

The referee (`fl_mine_local.py`) ranks placements by playing them; the thing
that has to learn that ranking is `own_lap4/t0.bin`, which reads a 96-dim
FL14_RANKER vector -- actor 48 + rowwise 41 + deck 7, no joint block.  This
writes one such vector per (root, candidate) so the correction trainer can
work on arrays instead of re-invoking the solver.

# Why the full opening set, not just the audited candidates

`sel_means` holds the field the race actually ran (top-36 + strata, or all 232
on a canary).  Training only needs those.  But the question "did the
correction move the SERVED pick" is a question about the argmax over every
opening, including the ones nomination never scored -- a net that reorders the
audited field while promoting an unaudited placement has not been fixed, it
has been broken somewhere nobody looked.  So every opening is encoded and the
audited ones are flagged; rows outside the field carry `mean: null`.

# Parity

The candidate list and its serving scores come from the binary's own
`--fl-t0-deep --rollouts 0`, which calls `play_roots::t0_scores` -- the same
function a played hand calls.  Every encoded row therefore has the score the
shipped net gives it, and this module checks its own vectors against that
column.  Agreement to float32 is the proof that the Python encoder and
`playout::encode_for` build the same vector; an argmax check alone would pass
on an encoder that was wrong in a way the argmax happened to survive.

Usage:
    python -m ai.tutor.encode_fl_material            # defaults below
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ai.engine.encoding import ALL_CARDS
from ai.tutor.encode_fl14_teacher import context_block
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask

D = Path("D:/ofc_data/hu")
REPO = Path(__file__).resolve().parents[2]
FEATURE_SIZE = 96


def load_bin(path: Path):
    """Parse a T4F1 image back into (mean, std, [(W, b), ...])."""
    raw = path.read_bytes()
    assert raw[:4] == b"T4F1"
    version, layers, input_dim = struct.unpack_from("<III", raw, 4)
    assert version == 1
    off = 16
    mean = np.frombuffer(raw, np.float32, input_dim, off)
    off += 4 * input_dim
    std = np.frombuffer(raw, np.float32, input_dim, off)
    off += 4 * input_dim
    mats = []
    for _ in range(layers):
        n_in, n_out = struct.unpack_from("<II", raw, off)
        off += 8
        w = np.frombuffer(raw, np.float32, n_in * n_out, off).reshape(n_out, n_in)
        off += 4 * n_in * n_out
        b = np.frombuffer(raw, np.float32, n_out, off)
        off += 4 * n_out
        mats.append((w.copy(), b.copy()))
    return mean.copy(), std.copy(), mats


def forward(mats, x):
    """The image's own arithmetic, in numpy -- only used to check parity."""
    h = x
    for index, (w, b) in enumerate(mats):
        h = h @ w.T + b
        if index < len(mats) - 1:
            h = np.maximum(h, 0.0)
    return h.squeeze(-1)


def rows_of(key: str) -> list[list[str]]:
    rows = [part.split(",") if part else [] for part in key.split("|")]
    if len(rows) != 3:
        raise AssertionError(f"a T0 key has {len(rows)} rows: {key}")
    return rows


def rank_root(binary: Path, models: Path, fl_ev: Path, cards: str, out: Path):
    """Every opening this root's chooser can reach, with its serving score."""
    if not out.exists() or not out.stat().st_size:
        own = ",".join(str(models / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
        done = subprocess.run(
            [str(binary), "--fl-t0-deep", "--t0-cards", cards, "--rollouts", "0",
             "--arm-a-own", own, "--fl-ev-config", str(fl_ev), "--output", str(out)],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"rank failed for {cards}: {done.stderr[-400:]}")
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]


def fetch_rowwise(binary: Path, fl_ev: Path, requests: list[dict]) -> dict:
    """The 41-dim rowwise block per position, from the block binary.

    Batched because the per-call cost is a process launch and a model load,
    not the positions: one call per few thousand boards instead of per board
    is the difference between minutes and most of a day.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                # A T0 board leaves eight slots open, so the joint block would
                # sample; it is not part of the 96-dim vector, so ask for the
                # cheapest one the mode will produce and drop it.
                handle.write(json.dumps(dict(request, samples=1, max_arrangements=1)) + "\n")
        subprocess.run(
            [str(binary), "--input", str(in_path), "--output", str(out_path),
             "--fl-ev-config", str(fl_ev), "--joint-outlook", "--chunk-size", "256"],
            check=True, capture_output=True)
        return {json.loads(l)["id"]: json.loads(l)["rowwise_block"]
                for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}


def encode(binary: Path, fl_ev: Path, boards: list[list[list[str]]], batch: int):
    """The 96-dim vector for each board, in the order given."""
    out = np.empty((len(boards), FEATURE_SIZE), np.float32)
    for base in range(0, len(boards), batch):
        chunk = boards[base:base + batch]
        requests, pools = [], []
        for index, rows in enumerate(chunk):
            flat = [c for row in rows for c in row]
            seen = seen_mask(flat)
            if bin(seen).count("1") != len(flat):
                raise AssertionError(f"a card repeats in {flat}")
            pool = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)]
            pools.append(pool)
            requests.append({"id": str(index),
                             "board": {"top": rows[0], "middle": rows[1], "bottom": rows[2]},
                             "pool": pool})
        blocks = fetch_rowwise(binary, fl_ev, requests)
        for index, rows in enumerate(chunk):
            actor, _categories = actor_block(rows, pools[index])
            vector = actor + [float(v) for v in blocks[str(index)]] + context_block(pools[index])
            if len(vector) != FEATURE_SIZE:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            out[base + index] = vector
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--material", type=Path, default=D / "fl_mine1/material_final.jsonl")
    ap.add_argument("--out-dir", type=Path, default=D / "fl_evalfix")
    ap.add_argument("--models-dir", type=Path, default=D / "models_ship_20260830/own_lap4")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target_gate/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--rank-workers", type=int, default=12)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranks_dir = args.out_dir / "ranks"
    ranks_dir.mkdir(exist_ok=True)
    started = time.time()

    material = [json.loads(l) for l in open(args.material, encoding="utf-8") if l.strip()]
    print(f"material: {len(material)} roots", flush=True)

    # Ranking is one short process per root and the binary's T0 scorer is
    # single-threaded, so the box is filled by running roots side by side.
    def ranked(row):
        return row["id"], rank_root(args.binary, args.models_dir, args.fl_ev_config,
                                    row["cards"], ranks_dir / f"{row['id']}.jsonl")

    with ThreadPoolExecutor(max_workers=args.rank_workers) as pool:
        ranks = dict(pool.map(ranked, material))
    print(f"ranked {len(ranks)} roots in {time.time() - started:.0f}s", flush=True)

    boards, meta = [], []
    for row in material:
        sel = row["sel_means"]
        precise = "ci" in row
        for entry in ranks[row["id"]]:
            key = entry["key"]
            is_serving = key == row["model_pick"]
            is_ref = key == row["ref_pick"]
            record = {
                "id": f"{row['id']}/{key}", "root": row["id"], "key": key,
                "mean": sel.get(key), "in_sel": key in sel,
                "scored": precise, "escalated": False,
                "is_serving": is_serving, "is_ref": is_ref,
                "verdict": row["verdict"], "field": row.get("field"),
                "own_rank": entry["own_rank"], "own_score": entry["score"],
            }
            if precise and (is_serving or is_ref):
                record["margin"] = row["margin"]
                record["ci"] = row["ci"]
            boards.append(rows_of(key))
            meta.append(record)
    print(f"boards: {len(boards)} ({sum(m['in_sel'] for m in meta)} audited)", flush=True)

    x = encode(args.binary, args.fl_ev_config, boards, args.batch)
    print(f"encoded in {time.time() - started:.0f}s", flush=True)

    # Parity: the vectors must reproduce the score the serving path published
    # for the same placement.  Checked on every row, not a sample -- it costs
    # one matmul and it is the only evidence that this file and
    # `playout::encode_for` agree.
    mean, std, mats = load_bin(args.models_dir / "t0.bin")
    py = forward(mats, (x - mean) / std)
    rust = np.asarray([m["own_score"] for m in meta], np.float32)
    delta = np.abs(py - rust)
    agree = tied = split = at = 0
    for row in material:
        n = len(ranks[row["id"]])
        window, keys = py[at:at + n], [m["key"] for m in meta[at:at + n]]
        pick = int(window.argmax())
        served = keys.index(row["model_pick"])
        if keys[pick] == row["model_pick"]:
            agree += 1
        elif abs(float(window[pick]) - float(window[served])) <= 1e-4:
            # This net scores whole groups of openings identically (its
            # features cannot separate, say, which junk row a low card sits
            # in).  Rust breaks such a tie by enumeration order and numpy by
            # array order, so a differing argmax here is a tie, not a
            # disagreement -- what must match is the SCORE.
            tied += 1
        else:
            split += 1
        at += n
    if at != len(meta):
        raise AssertionError(f"parity walked {at} of {len(meta)} rows")
    print(f"PARITY max|py-rust| {delta.max():.3e}  mean {delta.mean():.3e}", flush=True)
    print(f"PARITY argmax==model_pick {agree}/{len(material)} roots "
          f"(+{tied} tie-equal, {split} genuine disagreements)", flush=True)
    # Written before the gate: a parity failure is something to diagnose, and
    # re-encoding 231k boards just to look at it is a waste.  A failed run
    # still leaves arrays nothing should train on, so the exit code, not the
    # presence of the file, is the contract.
    np.savez_compressed(args.out_dir / "enc_pairs.npz", x=x,
                        own_score=rust, py_score=py.astype(np.float32))
    with open(args.out_dir / "pairs_meta.jsonl", "w", encoding="utf-8") as handle:
        for record in meta:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_dir/'enc_pairs.npz'} x{x.shape} and pairs_meta.jsonl "
          f"({time.time() - started:.0f}s total)", flush=True)
    if delta.max() > 1e-3 or split:
        raise SystemExit("PARITY FAILED -- the Python encoder is not the serving encoder")


if __name__ == "__main__":
    main()
