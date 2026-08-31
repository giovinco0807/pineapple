"""Encode the T2 own corpus at width 128 = 120 (cheap v2) + all 8 joint dims,
drawn by the splitmix sampler at K=400.

The oracle run proved this composition reaches 110-grade dev regret (0.4036);
the splitmix sampler removes the SHA-256 shuffle that was ~95% of the sampled
joint's cost at T2.  This corpus is the honest version of that oracle: the
joint dims come from the sampler serving will actually use, not from the
encoded_110 arrays.

Vector order (must match playout.rs::encode_for's FL14_CHEAP_V2_JOINT_SIZE
branch): actor 48 | rowwise 41 | context 7 | cheap_draw_v2 24 | joint 8 = 128.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")

from ai.tutor.cheap_draw_features import cheap_draw_block_v2
from ai.tutor.encode_fl14_t0_110 import DECK
from ai.tutor.encode_fl14_teacher import JokerNamer, context_block
from ai.tutor.t3_second_features import actor_block

REPO = Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")
WIDTH = 128
SPLITS = ("fit", "dev")


def split_of(record_id: str) -> str:
    digest = hashlib.sha256(f"fl14-t2-teacher-v1/{record_id}".encode()).digest()
    return ("fit" if (b := int.from_bytes(digest[:4], "big") % 100) < 80
            else ("dev" if b < 90 else "test"))


def fetch_blocks(binary: Path, fl_ev: Path, requests: list[dict], samples: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                payload = dict(request)
                payload["samples"] = samples
                payload["max_arrangements"] = 32
                payload["sampler"] = "splitmix"
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path,
                    default=Path("D:/ofc_data/lap4_t2_own/t2_labels_own_10k.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--batch", type=int, default=4000)
    args = ap.parse_args()
    started = time.time()

    boards, ys, ids = [], [], []
    for line in open(args.labels, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = str(rec["id"])
        if split_of(rid) == "test":
            continue  # dev-only experiment; test stays sealed
        for act in rec["actions"]:
            namer = JokerNamer()
            parts = act["action_key"].split("|")
            rows = [namer(p) for p in parts[:3]]
            tail = namer(parts[3]) + namer(rec["dead"])
            flat = [c for row in rows for c in row] + tail
            if len(set(flat)) != len(flat):
                raise AssertionError(f"a card repeats in {rid}: {flat}")
            boards.append((rows, f"t2/{rid}"))
            ys.append(act["value"])
            ids.append(rid)
    print(f"corpus: {len(boards)} actions over {len(set(ids))} records", flush=True)

    out = np.empty((len(boards), WIDTH), np.float32)
    shard_dir = args.out_dir / f"_shards{WIDTH}_k{args.samples}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for base in range(0, len(boards), args.batch):
        chunk = boards[base:base + args.batch]
        shard = shard_dir / f"{base:08d}.npy"
        if shard.exists():
            cached = np.load(shard)
            if cached.shape == (len(chunk), WIDTH):
                out[base:base + len(chunk)] = cached
                continue
        requests, pools = [], []
        for index, (rows, seed) in enumerate(chunk):
            flat = [c for row in rows for c in row]
            pool = [c for c in DECK if c not in set(flat)]
            pools.append(pool)
            requests.append({"id": str(index), "seed": seed,
                             "board": {"top": rows[0], "middle": rows[1], "bottom": rows[2]},
                             "pool": pool})
        blocks = fetch_blocks(args.binary, args.fl_ev_config, requests, args.samples)
        for index, (rows, _seed) in enumerate(chunk):
            rowwise, joint = blocks[str(index)]
            actor, _cats = actor_block(rows, pools[index])
            vector = (actor + [float(v) for v in rowwise] + context_block(pools[index])
                      + cheap_draw_block_v2(rows, pools[index])
                      + [float(v) for v in joint])
            if len(vector) != WIDTH:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            out[base + index] = vector
        np.save(shard, out[base:base + len(chunk)])
        done = min(base + args.batch, len(boards))
        rate = done / max(time.time() - started, 1e-9)
        print(f"  {done}/{len(boards)} {rate:.1f} rows/s "
              f"eta {(len(boards) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    y = np.asarray(ys, np.float32)
    index = {r: i for i, r in enumerate(dict.fromkeys(ids))}
    roots = np.asarray([index[r] for r in ids], np.int64)
    jokers = np.asarray([sum(1 for row in rows for c in row if c.startswith("X"))
                         for rows, _ in boards], np.int8)
    which = np.asarray([split_of(r) for r in ids])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "ofc_fl14_t2_teacher/v5_cheap120_joint8_splitmix",
                "feature_size": WIDTH, "joint_samples": args.samples,
                "joint_sampler": "splitmix", "joint_seed_scope": "root",
                "blocks": {"actor": 48, "rowwise": 41, "context": 7,
                           "cheap_draw": 24, "joint": 8},
                "labels": str(args.labels),
                "split_rule": "sha256('fl14-t2-teacher-v1/<id>') % 100 -> 80/10/10",
                "splits": {}}
    for name in SPLITS:
        mask = which == name
        np.savez_compressed(args.out_dir / f"{name}.npz", x=out[mask], y=y[mask],
                            jokers=jokers[mask], roots=roots[mask])
        manifest["splits"][name] = {"rows": int(mask.sum()),
                                    "roots": int(np.unique(roots[mask]).size)}
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=2), flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
