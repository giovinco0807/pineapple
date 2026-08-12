"""Encode the FL14 T1 labels into training arrays.

# What these labels are

Produced by `t4_first_exact --t1-vs-fl-library` at `truncate_depth: 0`, which
means an action's value is

    mean over 32 sampled T2 draws of
      max over T2 placements of
        the T2 evaluator's own value

The playout stops at the T2 chooser and never reaches an eleven-card board, so
**the Fantasyland pool is never consulted** -- every label carries
`mean_fl_samples: 0.0`.  The best-responding opponent reaches these labels only
through what the T2 evaluator learned from labels that did price against it.

That is a one-step bootstrap, chosen deliberately for a first lap: the
alternative, playing every line out to an exactly-priced terminal, measured
91 s a root against this shape's 3.5 and would have cost about $390 of cloud
time for the same 3,000 roots.  On four positions the two disagreed on every
one, with the truncated shape preferring to build the top row where the exact
one spread into the middle and bottom -- but the exact reference at that budget
does not reproduce its own argmax between seeds, so that comparison sizes the
disagreement and does not settle it.  Settling it is second-lap work.

# Features

    actor   48  hero's own board
    rowwise 41  per-row completion outlook over the unseen pool
    context  7  deck composition
    ----------
             96

No joint block, unlike the T2/T3 encoders.  Not because it is worthless -- it
is worth 0.085 of regret at T2 -- but because this model's job is to choose
moves inside the T0 search, and the joint block is 98% of what a candidate
encode costs.  96 is also the width `playout.rs` already knows as
`FL14_RANKER_SIZE`, so the trained model loads without a Rust change.

# Why the requests file is required

The label record carries `id` and nothing about the position.  The deal is a
pure function of the seed, but re-deriving it here would make these arrays
depend on `sample_t1_root` never changing.  The request file that produced the
labels is read instead, so the position comes from the same bytes the labeler
saw.

Usage:
    python -m ai.tutor.encode_fl14_t1 \
        --labels D:/ofc_data/fl14_t1_labels_v1.jsonl \
        --requests D:/ofc_data/fl14_t1_requests_v1.jsonl \
        --out-dir D:/ofc_data/fl14_t1_encoded_v1
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

from ai.engine.encoding import ALL_CARDS
from ai.tutor.encode_fl14_teacher import context_block
from ai.tutor.solver_paths import _solver_path
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask

SPLITS = ("fit", "dev", "test")
ACTOR_SIZE = 48
ROWWISE_SIZE = 41
CONTEXT_SIZE = 7
FEATURE_SIZE = ACTOR_SIZE + ROWWISE_SIZE + CONTEXT_SIZE  # 96

ROWS = ("top", "middle", "bottom")


def split_of(root: str, street: str) -> str:
    digest = hashlib.sha256(f"fl14-{street}-teacher-v1/{root}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def board_after(
    board: dict | None, action_key: str
) -> tuple[list[list[str]], list[str]]:
    """The board an action reaches, and what it threw away.

    The two streets spell an action differently, because they are different
    moves: a T1 action places two of three drawn cards onto an existing board
    and discards the third, so it is written as placements plus a discard; a T0
    action assigns all five dealt cards and discards nothing, so it is written
    as the board itself.
    """
    if action_key.startswith("{"):
        action = json.loads(action_key)
        rows = [list(board[name]) for name in ROWS]
        for card, row in action["placements"]:
            rows[ROWS.index(row)].append(card)
        return rows, [action["discard"]]
    rows = [part.split(",") if part else [] for part in action_key.split("|")]
    if len(rows) != 3:
        raise AssertionError(f"a T0 action key has {len(rows)} rows: {action_key}")
    return rows, []


def fetch_rowwise(requests: list[dict], workspace_root: Path, solver: str) -> dict:
    """The 41-dim rowwise block per position, from the block binary.

    The joint-outlook mode returns the joint block too; it is dropped here
    rather than not asked for, because asking is one call either way and a
    second mode would be a second thing to keep in step.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.jsonl"
        out_path = Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                payload = dict(request)
                # Six open slots after a T1 placement, so the joint block would
                # have to sample; it is discarded, so ask for the cheapest one
                # the mode will produce.
                payload["samples"] = 1
                payload["max_arrangements"] = 1
                handle.write(json.dumps(payload) + "\n")
        subprocess.run(
            [
                solver,
                "--input", str(in_path),
                "--output", str(out_path),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--joint-outlook", "--chunk-size", "256",
            ],
            check=True,
        )
        blocks = {}
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    blocks[payload["id"]] = payload["rowwise_block"]
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t0", "t1"], default="t1")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4000)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--solver", default=None)
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve(strict=True)
    solver = args.solver or str(_solver_path(workspace_root))
    if not Path(solver).exists():
        raise SystemExit(f"no block binary at {solver}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    positions: dict[str, dict] = {}
    with args.requests.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                positions[str(record["id"])] = record

    # A T0 root is named `t0c-00042`, a T1 root by its seed.  The grouping the
    # gate needs is the identity, not the number, so ids are interned.
    root_index: dict[str, int] = {}
    for key in positions:
        root_index.setdefault(key, len(root_index))

    buffers = {name: {"x": [], "y": [], "j": [], "r": []} for name in SPLITS}
    pending: list[tuple] = []
    processed = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        requests = []
        for position, (rows_after, dead_after, _value, _root) in enumerate(pending):
            all_seen = [c for row in rows_after for c in row] + dead_after
            if bin(seen_mask(all_seen)).count("1") != len(all_seen):
                raise AssertionError(f"a card repeats in {all_seen}")
            seen = seen_mask(all_seen)
            requests.append(
                {
                    "id": str(position),
                    "board": {
                        "top": rows_after[0],
                        "middle": rows_after[1],
                        "bottom": rows_after[2],
                    },
                    "pool": [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)],
                }
            )
        blocks = fetch_rowwise(requests, workspace_root, solver)
        for position, (rows_after, dead_after, value, root) in enumerate(pending):
            seen = seen_mask([c for row in rows_after for c in row] + dead_after)
            pool = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)]
            actor, _categories = actor_block(rows_after, pool)
            vector = (
                actor
                + [float(v) for v in blocks[str(position)]]
                + context_block(pool)
            )
            if len(vector) != FEATURE_SIZE:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            bucket = buffers[split_of(root, args.street)]
            bucket["x"].append(vector)
            bucket["y"].append(value)
            bucket["j"].append(
                sum(1 for row in rows_after for c in row if c in ("X1", "X2"))
            )
            bucket["r"].append(root_index[root])
        pending = []

    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            root = str(record["id"])
            position = positions.get(root)
            if position is None:
                raise SystemExit(f"no request for label {root}; wrong requests file?")
            for action in record["actions"]:
                rows_after, discard = board_after(
                    position.get("board"), action["action_key"]
                )
                pending.append(
                    (
                        rows_after,
                        list(position.get("dead", [])) + discard,
                        action["value"],
                        root,
                    )
                )
                processed += 1
            if len(pending) >= args.batch:
                flush()
                print(f"[{processed}] {processed / (time.time() - started):.0f} rows/s",
                      flush=True)
    flush()

    manifest = {
        "schema": f"ofc_fl14_{args.street}_teacher/v1_bootstrap",
        "feature_size": FEATURE_SIZE,
        "blocks": {"actor": ACTOR_SIZE, "rowwise": ROWWISE_SIZE, "context": CONTEXT_SIZE},
        "labels": str(args.labels),
        "requests": str(args.requests),
        "opponent": "none_directly__t2_evaluator_value_at_truncate_depth_0",
        "rows_encoded": processed,
        "split_rule": f"sha256('fl14-{args.street}-teacher-v1/<id>') % 100 -> 80/10/10",
        "elapsed_seconds": time.time() - started,
        "splits": {},
    }
    for name in SPLITS:
        x = np.asarray(buffers[name]["x"], dtype=np.float32)
        y = np.asarray(buffers[name]["y"], dtype=np.float32)
        j = np.asarray(buffers[name]["j"], dtype=np.int8)
        r = np.asarray(buffers[name]["r"], dtype=np.int64)
        np.savez_compressed(args.out_dir / f"{name}.npz", x=x, y=y, jokers=j, roots=r)
        manifest["splits"][name] = {
            "rows": int(x.shape[0]),
            "roots": int(np.unique(r).size),
            "ev_mean": float(y.mean()) if y.size else None,
            "ev_std": float(y.std()) if y.size else None,
        }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
