"""Encode T0 positions at the 110-dim FL14_FEATURE width.

The T0 own-hand corpus (`lap4_t0_own`) was only ever encoded at 96 dims --
`encode_fl14_t1.py --street t0`, which has no joint or allocation block.  The
feature-sufficiency experiment needs the same positions at the width the T2
own chooser already serves under, so this reads that script's label shape and
emits `encode_fl14_teacher.py`'s block set:

    actor  48 + rowwise 41 + joint 8 + context 7 + allocation 6 = 110

The order is not a choice.  `playout.rs::encode_for` appends exactly these
blocks in exactly this sequence for a 110-dim model, and serving picks the
encoder by model width alone -- so a vector assembled in any other order
trains a net that reads garbage the moment it ships.  The two routes to this
block set never meet in code, which is why the order is restated here as a
comment rather than trusted to memory.

Two modes, one assembly function, deliberately:

* `corpus` re-encodes the labelled T0 actions into fit/dev/test arrays, under
  the SAME split hash the 96-dim corpus used (`fl14-t0-teacher-v1/<root>`), so
  a net trained here has a dev split comparable to the 0.203 regret the 96-dim
  net reached on its own.
* `pairs` re-encodes the audited referee candidates so the correction stage
  has 110-dim inputs for the same (root, key) rows it used at 96.

Sharing `encode_boards` between them is the point: if the corpus and the
audited pairs were assembled by two functions, the net would be trained on one
vector and corrected on another, and nothing downstream would report it.

# Cost

The joint block samples 400 completions of a T0 board's eight open slots, so
this runs at ~16 boards/s on sixteen cores against ~33/s for the 96-dim
shape.  Measured: the 231k-row corpus is about four hours, the 47k audited
pairs about fifty minutes.

Usage:
    python -m ai.tutor.encode_fl14_t0_110 --mode corpus \
        --labels D:/ofc_data/lap4_t0_own/t0_labels_1k.jsonl \
        --requests D:/ofc_data/lap4_t0_own/requests_1k.jsonl \
        --out-dir D:/ofc_data/lap4_t0_own/encoded_110
    python -m ai.tutor.encode_fl14_t0_110 --mode pairs \
        --meta D:/ofc_data/hu/fl_evalfix/pairs_meta.jsonl \
        --out-dir D:/ofc_data/hu/fl_evalfix
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ai.tutor.cheap_draw_features import cheap_draw_block, cheap_draw_block_v2
from ai.tutor.encode_fl14_teacher import context_block
from ai.tutor.fl14_allocation_features import allocation_rank_block
from ai.tutor.t3_second_features import actor_block

REPO = Path(__file__).resolve().parents[2]

# The deck in `main.rs::all_cards()` order -- suits s,h,d,c, then the jokers.
#
# Not `ai.engine.encoding.ALL_CARDS`, which starts at hearts.  The two hold
# the same 54 cards, and for every unsampled block that is all that matters,
# which is why the 96-dim encoder never had to care.  The joint block does:
# `sampled_subset` draws POOL INDICES, so the same set in a different order is
# a different set of completions and a joint block that no serving encoder
# will ever reproduce.  Order is part of this interface.
DECK = [f"{rank}{suit}" for suit in "shdc" for rank in "23456789TJQKA"] + ["X1", "X2"]
SPLITS = ("fit", "dev", "test")
JOINT_SAMPLES = 400
MAX_ARRANGEMENTS = 32

# 110 = actor 48 + rowwise 41 + joint 8 + context 7 + allocation 6
# 112 = actor 48 + rowwise 41 + context 7 + cheap draw 16
#
# The widths are different encoders, not different sizes of one: 110 samples
# 400 completions per board and 112 counts.  `playout.rs` selects between them
# on model width alone, so the assembly below has to match its branch exactly.
WIDTHS = (110, 112, 120)


def split_of(root: str) -> str:
    """The 96-dim corpus's split rule, restated so the two agree row for row."""
    digest = hashlib.sha256(f"fl14-t0-teacher-v1/{root}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def rows_of(key: str) -> list[list[str]]:
    rows = [part.split(",") if part else [] for part in key.split("|")]
    if len(rows) != 3:
        raise AssertionError(f"a T0 action key has {len(rows)} rows: {key}")
    return rows


def fetch_blocks(binary: Path, fl_ev: Path, requests: list[dict], width: int = 110) -> dict:
    """Rowwise and joint blocks for one batch, from the block binary.

    At width 112 the joint block is not part of the vector, so the cheapest
    setting the mode accepts is asked for and the block discarded -- one call
    either way, and a second mode would be a second thing to keep in step.
    """
    sampled = width == 110
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                payload = dict(request)
                payload["samples"] = JOINT_SAMPLES if sampled else 1
                payload["max_arrangements"] = MAX_ARRANGEMENTS if sampled else 1
                handle.write(json.dumps(payload) + "\n")
        subprocess.run(
            [str(binary), "--input", str(in_path), "--output", str(out_path),
             "--fl-ev-config", str(fl_ev), "--joint-outlook", "--chunk-size", "256"],
            check=True, capture_output=True)
        blocks = {}
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                blocks[payload["id"]] = (payload["rowwise_block"], payload["joint_block"])
    return blocks


def encode_boards(binary: Path, fl_ev: Path, boards, batch: int, label: str,
                  shard_dir: Path | None = None, deadline: float | None = None,
                  width: int = 110):
    """The 110-dim vector for each (rows, seed), in the order given.

    The seed only reaches the completion sampler: every candidate at a
    root draws the SAME completions, so the sampling noise cancels in the
    within-root comparison that both the teacher's gate and the referee's
    ordering are read from.  It is the `seed_scope=root` convention of
    `encode_fl14_teacher.py`, and losing it would put sampling noise straight
    onto the ordering this experiment measures.
    """
    out = np.empty((len(boards), width), np.float32)
    started = time.time()
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    for base in range(0, len(boards), batch):
        chunk = boards[base:base + batch]
        # Checkpointed per batch.  The first run of this encoder was killed at
        # 31% and wrote nothing, because it only saved at the end: three hours
        # of joint sampling thrown away by one signal.  The board list is a
        # deterministic function of the label file, so a batch index names the
        # same rows on every run and a finished shard is simply reused.
        shard = None if shard_dir is None else shard_dir / f"{base:08d}.npy"
        if shard is not None and shard.exists():
            cached = np.load(shard)
            if cached.shape == (len(chunk), width):
                out[base:base + len(chunk)] = cached
                continue
            print(f"  [{label}] shard {shard.name} has shape {cached.shape}, redoing",
                  flush=True)
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(
                f"deadline reached with {base}/{len(boards)} encoded; "
                f"rerun to resume from the shards")
        requests, pools = [], []
        for index, (rows, seed) in enumerate(chunk):
            flat = [c for row in rows for c in row]
            seen = set(flat)
            if len(seen) != len(flat):
                raise AssertionError(f"a card repeats in {flat}")
            pool = [c for c in DECK if c not in seen]
            pools.append(pool)
            requests.append({"id": str(index), "seed": seed,
                             "board": {"top": rows[0], "middle": rows[1], "bottom": rows[2]},
                             "pool": pool})
        blocks = fetch_blocks(binary, fl_ev, requests, width)
        for index, (rows, _seed) in enumerate(chunk):
            rowwise, joint = blocks[str(index)]
            actor, _categories = actor_block(rows, pools[index])
            if width == 110:
                vector = (actor
                          + [float(v) for v in rowwise]
                          + [float(v) for v in joint]
                          + context_block(pools[index])
                          + allocation_rank_block(rows))
            else:
                cheap = cheap_draw_block if width == 112 else cheap_draw_block_v2
                vector = (actor
                          + [float(v) for v in rowwise]
                          + context_block(pools[index])
                          + cheap(rows, pools[index]))
            if len(vector) != width:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            out[base + index] = vector
        if shard is not None:
            np.save(shard, out[base:base + len(chunk)])
        done = min(base + batch, len(boards))
        rate = done / max(time.time() - started, 1e-9)
        print(f"  [{label}] {done}/{len(boards)} {rate:.1f} rows/s "
              f"eta {(len(boards) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    return out


def do_corpus(args):
    positions = {}
    for line in open(args.requests, encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            positions[str(record["id"])] = record
    boards, ys, roots = [], [], []
    for line in open(args.labels, encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        root = str(record["id"])
        if root not in positions:
            raise SystemExit(f"no request for label {root}; wrong requests file?")
        for action in record["actions"]:
            boards.append((rows_of(action["action_key"]), f"root/{root}"))
            ys.append(action["value"])
            roots.append(root)
    print(f"corpus: {len(boards)} actions over {len(set(roots))} roots", flush=True)

    x = encode_boards(args.binary, args.fl_ev_config, boards, args.batch, "corpus",
                      args.out_dir / f"_shards{args.width}", args.deadline, args.width)
    y = np.asarray(ys, np.float32)
    index = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    r = np.asarray([index[v] for v in roots], np.int64)
    j = np.asarray([sum(1 for row in rows for c in row if c in ("X1", "X2"))
                    for rows, _seed in boards], np.int8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = ({"actor": 48, "rowwise": 41, "joint": 8, "context": 7, "allocation": 6}
              if args.width == 110
              else {"actor": 48, "rowwise": 41, "context": 7,
                    "cheap_draw": args.width - 96})
    manifest = {"schema": f"ofc_fl14_t0_teacher/v2_{args.width}",
                "feature_size": args.width, "blocks": blocks,
                "joint_samples": JOINT_SAMPLES if args.width == 110 else 0,
                "max_arrangements": MAX_ARRANGEMENTS if args.width == 110 else 0,
                "joint_seed_scope": "root", "labels": str(args.labels),
                "requests": str(args.requests), "rows_encoded": len(boards),
                "split_rule": "sha256('fl14-t0-teacher-v1/<id>') % 100 -> 80/10/10",
                "splits": {}}
    which = np.asarray([split_of(v) for v in roots])
    for name in SPLITS:
        mask = which == name
        np.savez_compressed(args.out_dir / f"{name}.npz", x=x[mask], y=y[mask],
                            jokers=j[mask], roots=r[mask])
        manifest["splits"][name] = {"rows": int(mask.sum()),
                                    "roots": int(np.unique(r[mask]).size),
                                    "ev_mean": float(y[mask].mean()) if mask.any() else None,
                                    "ev_std": float(y[mask].std()) if mask.any() else None}
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=2), flush=True)


def do_pairs(args):
    meta = []
    for line in open(args.meta, encoding="utf-8"):
        record = json.loads(line)
        # At 112 the whole fan-out is affordable, and the served pick is an
        # argmax over all of it -- an audited-only field would answer a
        # question serving never asks.
        if record["in_sel"] or args.all_rows:
            meta.append(record)
    print(f"pairs: {len(meta)} audited placements over "
          f"{len({m['root'] for m in meta})} roots", flush=True)
    boards = [(rows_of(m["key"]), f"root/{m['root']}") for m in meta]
    x = encode_boards(args.binary, args.fl_ev_config, boards, args.batch, "pairs",
                      args.out_dir / f"_shards_pairs{args.width}", args.deadline, args.width)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / f"enc_pairs_{args.width}.npz", x=x)
    with open(args.out_dir / f"pairs_meta_{args.width}.jsonl", "w", encoding="utf-8") as handle:
        for record in meta:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_dir}/enc_pairs_{args.width}.npz x{x.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["corpus", "pairs"], required=True)
    ap.add_argument("--labels", type=Path,
                    default=Path("D:/ofc_data/lap4_t0_own/t0_labels_1k.jsonl"))
    ap.add_argument("--requests", type=Path,
                    default=Path("D:/ofc_data/lap4_t0_own/requests_1k.jsonl"))
    ap.add_argument("--meta", type=Path,
                    default=Path("D:/ofc_data/hu/fl_evalfix/pairs_meta.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target_gate/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--width", type=int, default=110, choices=WIDTHS)
    ap.add_argument("--all-rows", action="store_true",
                    help="pairs mode: encode the full fan-out, not just audited")
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly after this long; finished shards survive")
    args = ap.parse_args()
    started = time.time()
    args.deadline = None if args.max_minutes is None else started + args.max_minutes * 60
    (do_corpus if args.mode == "corpus" else do_pairs)(args)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
