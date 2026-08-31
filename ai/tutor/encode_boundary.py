"""Encode street-boundary states for the ladder's value nets.

A boundary net V_k answers: at street k's start, what is the first actor's
value of (first-actor board, second-actor board)?  Its training rows come
straight out of the street-k first-seat teacher: each root's board pair is a
boundary state (conditioned on the first actor's draw, which the regression
marginalises), and the target is the root's best action value.

The 207-dim pair encoding is `hu_street_teacher.PairEncoder`, the same bytes
the teacher will feed the net at serve time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ai.tutor.encode_hu_teacher import canonical_jokers
from ai.tutor.hu_street_teacher import PairEncoder


def split_of(hand: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"hu-boundary-v1/{hand}".encode()).digest()[:4], "big"
    ) % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def stable(hand: str) -> int:
    try:
        return int(hand)
    except ValueError:
        return int.from_bytes(hashlib.sha256(hand.encode()).digest()[:8], "big") & (
            (1 << 63) - 1
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="first-seat street teacher output")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    requests_by_id = {}
    with args.requests.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            requests_by_id[row["id"]] = row

    encoder = PairEncoder(args.workspace_root)
    buffers = {n: {"x": [], "y": [], "r": [], "j": []} for n in ("fit", "dev", "test")}
    pending_pairs = []
    pending_meta = []

    def flush():
        nonlocal pending_pairs, pending_meta
        if not pending_pairs:
            return
        x = encoder.encode(pending_pairs)
        for row, (hand, target, jokers) in zip(x, pending_meta):
            bucket = buffers[split_of(str(hand))]
            bucket["x"].append(row)
            bucket["y"].append(target)
            bucket["r"].append(hand)
            bucket["j"].append(jokers)
        pending_pairs, pending_meta = [], []

    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            request = requests_by_id[record["id"]]
            target = max(a["value"] for a in record["actions"])
            opp_rows, used = canonical_jokers(request["opp_board"], 0)
            own_rows, used2 = canonical_jokers(request["board"], used)
            jokers_used = used2
            dead = []
            for card in request["dead"]:
                if card.startswith("X"):
                    jokers_used += 1
                    dead.append(f"X{jokers_used}")
                else:
                    dead.append(card)
            # The boundary state hides the first actor's draw: the pair plus
            # the actor's private dead is everything the net may see.
            pending_pairs.append((own_rows, opp_rows, dead))
            jokers = sum(1 for row in own_rows for c in row if c.startswith("X"))
            pending_meta.append((stable(record["id"]), target, jokers))
            if len(pending_pairs) >= args.batch:
                flush()
    flush()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, bucket in buffers.items():
        np.savez_compressed(
            args.out_dir / f"{name}.npz",
            x=np.asarray(bucket["x"], dtype=np.float32),
            y=np.asarray(bucket["y"], dtype=np.float32),
            roots=np.asarray(bucket["r"], dtype=np.int64),
            jokers=np.asarray(bucket["j"], dtype=np.int8),
        )
        print(name, len(bucket["y"]), "rows")


if __name__ == "__main__":
    main()
