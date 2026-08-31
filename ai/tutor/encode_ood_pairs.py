"""Encode harvested expansion pairs into V3s training rows.

Same 207-dim PairEncoder bytes the boundary net always reads; what is new is
only which states get encoded -- these are the expansion-distribution pairs
the net used to see only at read time.  Targets arrive separately (the Rust
teacher's per-draw root values, averaged per pair upstream).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.encode_hu_teacher import split_of, stable_id
from ai.tutor.hu_street_teacher import PairEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True,
                        help="jsonl of {id, value}")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    targets = {}
    for line in args.targets.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            targets[r["id"]] = float(r["value"])

    encoder = PairEncoder(args.workspace_root, 400)
    buffers = {n: {"x": [], "y": [], "r": [], "j": []} for n in ("fit", "dev", "test")}
    pending, meta = [], []

    def flush():
        nonlocal pending, meta
        if not pending:
            return
        x = encoder.encode(pending)
        for row, (hand, value, jokers) in zip(x, meta):
            b = buffers[split_of(str(hand))]
            b["x"].append(row); b["y"].append(value)
            b["r"].append(hand); b["j"].append(jokers)
        pending, meta = [], []

    done = 0
    for line in args.pairs.open(encoding="utf-8"):
        if not line.strip():
            continue
        pair = json.loads(line)
        if pair["id"] not in targets:
            continue
        jokers = sum(1 for row in (pair["board"] + pair["opp_board"])
                     for c in row if c.startswith("X"))
        jokers += sum(1 for c in pair["dead"] if c.startswith("X"))
        pending.append((pair["board"], pair["opp_board"], pair["dead"]))
        meta.append((stable_id(pair), targets[pair["id"]], jokers))
        done += 1
        if len(pending) >= args.batch:
            flush()
            if done % 5120 < args.batch:
                print(f"[{done}]", flush=True)
    flush()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, b in buffers.items():
        np.savez(args.out_dir / f"{name}.npz",
                 x=np.asarray(b["x"], dtype=np.float32),
                 y=np.asarray(b["y"], dtype=np.float32),
                 roots=np.asarray(b["r"], dtype=np.int64),
                 jokers=np.asarray(b["j"], dtype=np.int64))
        print(f"{name}: {len(b['y'])} rows")
    print(f"encoded {done} pairs")


if __name__ == "__main__":
    main()
