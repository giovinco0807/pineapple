"""Swap new teacher values into existing encoded datasets, features untouched.

A relabel changes what the actions are worth, not what the boards look like:
the requests are the same 100k rows, the candidate enumeration is the same
code, so every x matrix already on disk -- the 207-dim rollout encoding, the
432-dim card one-hots, the 623-dim hybrid -- is still exactly right.  Only y
moves.  Re-encoding the 207-dim set would cost hours of rollouts to
recompute bytes that cannot have changed.

Row order inside the npz files is the one thing the swap has to know, and it
is reconstructed rather than assumed: the encoders iterated the OLD label
file in order, appending one row per action, bucketing whole roots by
`split_of`.  Replaying that walk gives each row its (root, action_key)
identity, and three checks pin it down before anything is written: the
replayed root sequence must equal the npz `roots` array, the old values must
match the npz `y`, and every (root, action_key) must exist in the new
labels.  A swap that cannot prove all three refuses to run -- misaligned
labels would train a model that looks fine and is wrong everywhere.

Boundary sets (`--boundary`) are one row per root with y = the root's best
action value, so they swap by root id alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.encode_hu_teacher import split_of, stable_id

SPLITS = ("fit", "dev", "test")


def read_labels(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            out[stable_id(record)] = {
                action["action_key"]: float(action["value"])
                for action in record["actions"]
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-labels", type=Path, required=True,
                        help="the label file the encodings were built from")
    parser.add_argument("--new-labels", type=Path, required=True)
    parser.add_argument("--enc-dirs", nargs="+", type=Path, default=[],
                        help="action-level encoded dirs to re-target")
    parser.add_argument("--boundary-dirs", nargs="+", type=Path, default=[],
                        help="per-root dirs whose y is the root's best value")
    parser.add_argument("--suffix", default="_d16")
    args = parser.parse_args()

    new = read_labels(args.new_labels)
    print(f"new labels: {len(new)} roots")

    # Replay the old file to give every encoded row its identity.
    order = {name: {"root": [], "key": [], "y": []} for name in SPLITS}
    with args.old_labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            hand = stable_id(record)
            bucket = order[split_of(str(hand))]
            for action in record["actions"]:
                bucket["root"].append(hand)
                bucket["key"].append(action["action_key"])
                bucket["y"].append(float(action["value"]))

    for enc in args.enc_dirs:
        out_dir = enc.parent / (enc.name + args.suffix)
        out_dir.mkdir(exist_ok=True)
        for split in SPLITS:
            z = np.load(enc / f"{split}.npz")
            walk = order[split]
            if not np.array_equal(
                z["roots"], np.asarray(walk["root"], dtype=np.int64)
            ):
                raise SystemExit(f"FATAL: {enc}/{split}: root walk diverges")
            if z["y"].shape[0] and not np.allclose(
                z["y"], np.asarray(walk["y"], dtype=np.float32), atol=1e-3
            ):
                raise SystemExit(f"FATAL: {enc}/{split}: old values diverge")
            try:
                fresh = np.asarray(
                    [new[r][k] for r, k in zip(walk["root"], walk["key"])],
                    dtype=np.float32,
                )
            except KeyError as missing:
                raise SystemExit(
                    f"FATAL: {enc}/{split}: new labels lack {missing}"
                ) from None
            np.savez(out_dir / f"{split}.npz", x=z["x"], y=fresh,
                     roots=z["roots"], jokers=z["jokers"])
            moved = float(np.abs(fresh - z["y"]).mean()) if len(fresh) else 0.0
            print(f"{out_dir.name}/{split}: {len(fresh)} rows, "
                  f"mean |shift| {moved:.3f}")
            z.close()

    for bnd in args.boundary_dirs:
        out_dir = bnd.parent / (bnd.name + args.suffix)
        out_dir.mkdir(exist_ok=True)
        best = {root: max(values.values()) for root, values in new.items()}
        for split in SPLITS:
            z = np.load(bnd / f"{split}.npz")
            try:
                fresh = np.asarray(
                    [best[int(r)] for r in z["roots"]], dtype=np.float32
                )
            except KeyError as missing:
                raise SystemExit(
                    f"FATAL: {bnd}/{split}: new labels lack root {missing}"
                ) from None
            np.savez(out_dir / f"{split}.npz", x=z["x"], y=fresh,
                     roots=z["roots"], jokers=z["jokers"])
            moved = float(np.abs(fresh - z["y"]).mean()) if len(fresh) else 0.0
            print(f"{out_dir.name}/{split}: {len(fresh)} rows, "
                  f"mean |shift| {moved:.3f}")
            z.close()


if __name__ == "__main__":
    main()
