"""Merge a fleet job's per-shard feature matrices into one encoded directory.

Each worker encodes only its own slice, so the job's `fit/dev/test` arrive as
one `.npz` per shard per split.  The split assignment is a hash of the hand,
not a position in the file, so concatenating the shards reproduces exactly
the partition a single-process encode would have made -- no hand can land in
two splits, and none is dropped.

The manifest is copied from any shard: every worker ran the same encoder at
the same feature size, and a shard whose feature width disagrees is refused
here rather than discovered as a shape error halfway through training.

Usage:
    python -m ai.tutor.fleet.collect_hu_encoded --run-id t0-r1 --job t0_btn \\
        --out-dir D:/ofc_data/hu/t0_btn_enc
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ai.tutor.fleet.gcs import GCLOUD, listing

BUCKET = "pokerhu-ofc-solver-485418-training"
PREFIX = "hu-street"
SPLITS = ("fit", "dev", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=0,
                        help="shards launched; 0 skips the completeness check")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()

    base = f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/enc/{args.job}"
    names = [n for n in listing(base + "/") if n.endswith(".npz")]
    starts = sorted({n.split("_")[0] for n in names})
    print(f"{base}: {len(names)} objects from {len(starts)} shards")
    for split in SPLITS:
        missing = [s for s in starts if f"{s}_{split}.npz" not in names]
        if missing:
            raise SystemExit(f"FATAL: {split} missing for shards {missing[:10]}")
    if args.shards and len(starts) != args.shards:
        print(f"WARNING: {len(starts)} shards present, {args.shards} launched")

    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        done = subprocess.run(
            [GCLOUD, "storage", "cp", f"{base}/*.npz", str(into)],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise SystemExit(f"FATAL: download failed: {done.stderr.strip()[:400]}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        width = None
        for split in SPLITS:
            parts = [np.load(into / f"{s}_{split}.npz") for s in starts]
            keys = sorted(set.intersection(*[set(p.files) for p in parts]))
            # A shard can legitimately contribute nothing to a split; an empty
            # array from `np.asarray([])` is one-dimensional and would not
            # concatenate against the (rows, features) ones.
            filled = [p for p in parts if p["x"].shape[0]]
            # Shards from two waves of the same job do not share boundaries:
            # a relaunch that splits the work differently covers some roots
            # twice, and create-only publishing keeps whichever name was
            # written first, so both ranges survive.  Nothing downstream would
            # notice -- the duplicate rows are identical and land in the same
            # split -- except that those roots would carry twice the weight.
            seen: set[int] = set()
            masks = []
            for part in filled:
                roots = part["roots"]
                fresh = np.array([r not in seen for r in roots.tolist()])
                seen.update(roots[fresh].tolist())
                masks.append(fresh)
            dropped = sum(int((~m).sum()) for m in masks)
            merged = {
                key: (np.concatenate(
                    [p[key][m] for p, m in zip(filled, masks)], axis=0)
                      if filled else parts[0][key])
                for key in keys
            }
            if dropped:
                print(f"  {split}: dropped {dropped} rows of repeated roots")
            if merged["x"].shape[0]:
                if width is None:
                    width = merged["x"].shape[1]
                elif merged["x"].shape[1] != width:
                    raise SystemExit(
                        f"FATAL: {split} is {merged['x'].shape[1]} wide, "
                        f"another split is {width}"
                    )
            np.savez(args.out_dir / f"{split}.npz", **merged)
            print(f"  {split}: {merged['x'].shape} from {len(filled)} shards")
            # `np.load` on an npz reads lazily and holds the file open, and
            # Windows will not delete an open file -- the temporary directory
            # would fail to clean up and take the exit code with it.
            for part in parts:
                part.close()

    print(f"{args.out_dir}: merged {len(starts)} shards, feature width {width}")
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ofc_hu_teacher_encoded_merged/v1",
                "run_id": args.run_id,
                "job": args.job,
                "shards": len(starts),
                "feature_size": int(width) if width else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
