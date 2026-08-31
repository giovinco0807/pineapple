"""Re-encode the T2 own-hand corpus at width 120 (96 prefix + cheap draw v2).

The shipped T2 chooser is 110-dim and its joint block samples 400 completions
per candidate, which costs about 60 ms of the ~64 ms a gen-2 teacher hand
takes.  Width 120 replaces that block with deterministic draw descriptors, so
the chooser stops thinking at serve time and the teacher's inner loop gets its
cost back.

# Why this is the same width as the T0 net, not a new one

`playout.rs` dispatches its encoder on model width alone, so a second meaning
for 120 would be a collision.  There is none: every block in the composition
-- actor, rowwise, deck context, cheap draw -- is a function of (rows, pool)
and nothing else.  A T2 board is simply a fuller one (nine placed against
five) drawn from a smaller pool.  Checked rather than assumed: the Python and
Rust vectors agree to 0.000e+00 on 402 real T2 boards.

# The joker trap this corpus carries

The T2 labeler writes both jokers as `X`, because for evaluation they are the
same card.  An encoder cannot be so relaxed: the pool is the deck minus what
is seen, and two cards both called `X1` would leave `X2` looking available
while hero holds it.  `JokerNamer` hands out X1 then X2 across one record --
and here it must run per ACTION, because a T2 action key repeats the whole
resulting board, so the numbering has to span that key and the record's dead
cards together.

Usage:
    python -m ai.tutor.encode_fl14_t2_120 --out-dir D:/ofc_data/lap4_t2_own/encoded_120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl14_t0_110 import DECK, encode_boards
from ai.tutor.encode_fl14_teacher import JokerNamer

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("fit", "dev", "test")
WIDTH = 120


def split_of(record_id: str) -> str:
    """The split rule `encoded_110` used, restated so dev stays comparable."""
    digest = hashlib.sha256(f"fl14-t2-teacher-v1/{record_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path,
                    default=Path("D:/ofc_data/lap4_t2_own/t2_labels_own_10k.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("D:/ofc_data/lap4_t2_own/encoded_120"))
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target_gate/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    started = time.time()
    args.deadline = None if args.max_minutes is None else started + args.max_minutes * 60

    boards, ys, ids, jokers = [], [], [], []
    for line in open(args.labels, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = str(rec["id"])
        for act in rec["actions"]:
            namer = JokerNamer()
            parts = act["action_key"].split("|")
            if len(parts) != 4:
                raise AssertionError(f"a T2 action key has {len(parts)} parts: {act['action_key']}")
            rows = [namer(p) for p in parts[:3]]
            tail = namer(parts[3]) + namer(rec["dead"])
            flat = [c for row in rows for c in row] + tail
            if len(set(flat)) != len(flat):
                raise AssertionError(f"a card repeats in {rid}: {flat}")
            # The seed only reaches sampled encoders; 120 samples nothing, so
            # it is a constant here and named for provenance rather than use.
            boards.append((rows, f"t2/{rid}"))
            ys.append(act["value"])
            ids.append(rid)
            jokers.append(sum(1 for c in flat if c.startswith("X")))
    print(f"corpus: {len(boards)} actions over {len(set(ids))} records", flush=True)

    x = encode_boards(args.binary, args.fl_ev_config, boards, args.batch, "t2-120",
                      args.out_dir / "_shards120", args.deadline, WIDTH)
    y = np.asarray(ys, np.float32)
    index = {r: i for i, r in enumerate(dict.fromkeys(ids))}
    roots = np.asarray([index[r] for r in ids], np.int64)
    jk = np.asarray(jokers, np.int8)
    which = np.asarray([split_of(r) for r in ids])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "ofc_fl14_t2_teacher/v3_cheapdraw120", "feature_size": WIDTH,
                "blocks": {"actor": 48, "rowwise": 41, "context": 7, "cheap_draw": 24},
                "joint_samples": 0, "labels": str(args.labels),
                "encoder_role": "fl14_cheap_v2",
                "split_rule": "sha256('fl14-t2-teacher-v1/<id>') % 100 -> 80/10/10",
                "rows_encoded": len(boards), "splits": {}}
    for name in SPLITS:
        mask = which == name
        np.savez_compressed(args.out_dir / f"{name}.npz", x=x[mask], y=y[mask],
                            jokers=jk[mask], roots=roots[mask])
        manifest["splits"][name] = {"rows": int(mask.sum()),
                                    "roots": int(np.unique(roots[mask]).size),
                                    "ev_mean": float(y[mask].mean()) if mask.any() else None,
                                    "ev_std": float(y[mask].std()) if mask.any() else None}
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=2), flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
