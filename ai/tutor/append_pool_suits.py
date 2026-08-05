"""Append the pool-suit block to an already re-encoded teacher (109 -> 116).

The joint blocks are already in the v2 chunks, so this touches no solver:
reconstruct each chunk's rows (same alignment machinery as the re-encoder,
actor-block verification included), compute the 7 pool-suit dims in Python,
and write 116-dim chunks.  Minutes, not hours.

Usage:
    python -m ai.tutor.append_pool_suits --street t2 \
        --in-dir D:/ofc_data/t2_vs_fl_teacher_v2 \
        --out-dir D:/ofc_data/t2_vs_fl_teacher_v3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ai.engine.encoding import ALL_CARDS
from ai.tutor.pool_suit_block import pool_suit_block
from ai.tutor.reencode_vs_fl_teacher import SAMPLERS, chunk_rows
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask

SPLITS = ("fit", "dev", "test")
NEW_FEATURE_SIZE = 116


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t0", "t1", "t2"], required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sampler = SAMPLERS[args.street]

    from ai.tutor.generate_t0_vs_fl_teacher import split_of as split_t0
    from ai.tutor.generate_t1_vs_fl_teacher import split_of as split_t1
    from ai.tutor.generate_t2_vs_fl_teacher import split_of as split_t2
    split_of = {"t0": split_t0, "t1": split_t1, "t2": split_t2}[args.street]

    manifest = json.loads((args.in_dir / "manifest.json").read_text(encoding="utf-8"))
    seed = manifest["seed"]
    out_chunk_dir = args.out_dir / "chunks"
    out_chunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    chunk_paths = sorted((args.in_dir / "chunks").glob("chunk_*.npz"))
    for done, chunk_path in enumerate(chunk_paths, start=1):
        out_path = out_chunk_dir / chunk_path.name
        if out_path.exists():
            continue
        stem_number = int(chunk_path.stem.split("_")[1])
        first_root = stem_number if stem_number >= seed else seed + stem_number
        old = np.load(chunk_path)
        rows_expected = sum(old[f"{name}_x"].shape[0] for name in SPLITS)
        per_chunk, index = [], first_root
        while sum(len(r[1]) for r in per_chunk) < rows_expected:
            root = sampler(index)
            per_chunk.append((root, chunk_rows(args.street, root)))
            index += 1
            if index - first_root > 2000:
                raise RuntimeError("row count never matched: alignment bug")
        per_split_rows = {name: [] for name in SPLITS}
        for root, actions in per_chunk:
            bucket = split_of(int(root["id"]))
            for rows_after, dead_after in actions:
                per_split_rows[bucket].append((rows_after, dead_after))

        arrays = {}
        for name in SPLITS:
            old_x = old[f"{name}_x"]
            rows = per_split_rows[name]
            if len(rows) != old_x.shape[0]:
                raise RuntimeError(f"{chunk_path.name} {name}: alignment bug")
            new_x = np.zeros((old_x.shape[0], NEW_FEATURE_SIZE), dtype=np.float32)
            for position, (rows_after, dead_after) in enumerate(rows):
                seen = seen_mask([c for r in rows_after for c in r] + dead_after)
                pool = [
                    card for card in ALL_CARDS
                    if not ((1 << CARD_INDEX[card]) & seen)
                ]
                actor, _categories = actor_block(rows_after, pool)
                if not np.allclose(actor, old_x[position, :48], atol=1e-5):
                    raise RuntimeError(
                        f"{chunk_path.name} {name} row {position}: actor mismatch"
                    )
                new_x[position, :109] = old_x[position, :109]
                new_x[position, 109:116] = np.asarray(
                    pool_suit_block(rows_after, pool), dtype=np.float32
                )
            arrays[f"{name}_x"] = new_x
            arrays[f"{name}_y"] = old[f"{name}_y"]
            arrays[f"{name}_j"] = old[f"{name}_j"]
        temp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(out_path)
        rate = done / (time.time() - started)
        print(f"[{done}/{len(chunk_paths)}] eta {(len(chunk_paths)-done)/rate/60:.1f} min", flush=True)

    chunks = [np.load(path) for path in sorted(out_chunk_dir.glob("chunk_*.npz"))]

    def merged(key, empty_shape, dtype):
        parts = [c[key] for c in chunks if c[key].size]
        return np.concatenate(parts) if parts else np.zeros(empty_shape, dtype=dtype)

    new_manifest = dict(manifest)
    new_manifest["feature_size"] = NEW_FEATURE_SIZE
    new_manifest["schema"] = manifest["schema"] + "+pool_suits_v1"
    new_manifest["splits"] = {}
    for name in SPLITS:
        x = merged(f"{name}_x", (0, NEW_FEATURE_SIZE), np.float32)
        y = merged(f"{name}_y", (0,), np.float32)
        j = merged(f"{name}_j", (0,), np.int8)
        np.savez_compressed(args.out_dir / f"{name}.npz", x=x, y=y, jokers=j)
        new_manifest["splits"][name] = {"rows": int(x.shape[0])}
    (args.out_dir / "manifest.json").write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(new_manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
