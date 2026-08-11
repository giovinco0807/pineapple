"""Encode the FL14 best-response teachers into training arrays.

The labels are produced by `fl_solver teach` and carry the position they
describe -- root, board, dead, draw -- so this reads them without re-deriving
any deal.

# Features

    actor   48  hero's own board: made values, rooms, ordering slack, FL facts
    rowwise 41  per-row completion outlook over the unseen pool
    joint    8  what the rows can achieve simultaneously
    context  7  deck composition
    ----------
            104

Seven context dims, not twelve: at a fixed Fantasyland width the opponent's
card-count one-hot (4) and `fl_ev[width]` (1) are constants, and a constant
column is a zero-variance column for the trainer to divide by.  They come
back when widths 15-17 get pools.

# What is NOT here

Nothing describing how contested each of hero's rows is against the
opponent's reachable values.  A best responder attacks the cheapest row, so
that is plausibly the fact this encoder is missing -- but the method that
found the joint block was to train, gate, and read the worst hands, and the
one time a block was added on a hunch (pool suits) the gate refuted it.  So
it is left out until a gate says otherwise.

Usage:
    python -m ai.tutor.encode_fl14_teacher --street t3 \
        --labels D:/ofc_data/fl14_teacher_v1/t3_labels.jsonl \
        --out-dir D:/ofc_data/fl14_t3_teacher_v1
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
from ai.tutor.solver_paths import _solver_path
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask

SPLITS = ("fit", "dev", "test")
ACTOR_SIZE = 48
ROWWISE_SIZE = 41
JOINT_SIZE = 8
CONTEXT_SIZE = 7
FEATURE_SIZE = ACTOR_SIZE + ROWWISE_SIZE + JOINT_SIZE + CONTEXT_SIZE  # 104


def split_of(root: int, street: str) -> str:
    digest = hashlib.sha256(f"fl14-{street}-teacher-v1/{root}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


class JokerNamer:
    """A comma-joined key back to cards.

    The labeler writes both jokers as `X` -- they are interchangeable for
    evaluation, so it never had to tell them apart.  This encoder does: the
    pool is `ALL_CARDS` minus what hero has seen, and two cards both named
    `X1` would leave `X2` looking available while hero holds it.

    So the allocator is per *record*, not per string.  One instance covers a
    label's board, dead cards and draw together, and hands out `X1` then `X2`
    across all of them.
    """

    def __init__(self) -> None:
        self.used = 0

    def __call__(self, text: str) -> list[str]:
        if not text:
            return []
        out = []
        for name in text.split(","):
            if name == "X":
                self.used += 1
                if self.used > 2:
                    raise AssertionError("a record named three jokers")
                out.append(f"X{self.used}")
            else:
                out.append(name)
        return out

    def rows(self, text: str) -> list[list[str]]:
        return [self(part) for part in text.split("|")]


def context_block(pool_cards: list[str]) -> list[float]:
    """Deck composition, without the dims a fixed width makes constant."""
    counts = {"A": 0, "K": 0, "Q": 0}
    jokers = 0
    for card in pool_cards:
        if card in ("X1", "X2"):
            jokers += 1
        elif card[0] in counts:
            counts[card[0]] += 1
    return [
        jokers / 2.0,
        counts["A"] / 4.0,
        counts["K"] / 4.0,
        counts["Q"] / 4.0,
        len(pool_cards) / 54.0,
        sum(counts.values()) / max(len(pool_cards), 1),
        (2 - jokers) / 2.0,
    ]


def fetch_blocks(
    requests: list[dict], workspace_root: Path, joint_samples: int, solver: str
) -> dict:
    """rowwise (41) and joint (8) from the Rust encoder, in one batch."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.jsonl"
        out_path = Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                payload = dict(request)
                payload["samples"] = joint_samples
                payload["max_arrangements"] = 32
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
                    blocks[payload["id"]] = (
                        payload["rowwise_block"],
                        payload["joint_block"],
                    )
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t3", "t4"], required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    # 0 means enumerate every completion instead of sampling it.  A T3
    # placement leaves two open slots (C(40,2) = 780) and a T4 placement
    # leaves none, so exact is affordable at both streets this encoder
    # serves -- and the labels it pairs with are exact, so the features
    # should not be the sampled half.
    parser.add_argument("--joint-samples", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--solver", default=None, help="override the block binary")
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve(strict=True)
    solver = args.solver or str(_solver_path(workspace_root))
    if not Path(solver).exists():
        raise SystemExit(f"no block binary at {solver}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # `r` is the root each row came from.  Without it the arrays are a
    # regression set and nothing more; the gate this teacher exists for
    # compares actions WITHIN a root, so the grouping has to survive.
    buffers = {name: {"x": [], "y": [], "j": [], "r": []} for name in SPLITS}
    pending: list[tuple] = []
    processed = 0
    dropped = 0

    def flush() -> None:
        """Fetch the Rust blocks for one batch and encode its rows."""
        nonlocal pending
        if not pending:
            return
        requests = []
        for position, (rows_after, dead_after, _value, _root) in enumerate(pending):
            all_seen = [c for row in rows_after for c in row] + dead_after
            # A duplicate card would shrink the mask without shrinking the
            # list -- which is exactly what a joker-naming slip looks like.
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
                    "pool": [
                        card for card in ALL_CARDS
                        if not ((1 << CARD_INDEX[card]) & seen)
                    ],
                }
            )
        blocks = fetch_blocks(requests, workspace_root, args.joint_samples, solver)
        for position, (rows_after, dead_after, value, root) in enumerate(pending):
            rowwise, joint = blocks[str(position)]
            seen = seen_mask([c for row in rows_after for c in row] + dead_after)
            pool = [card for card in ALL_CARDS if not ((1 << CARD_INDEX[card]) & seen)]
            actor, _categories = actor_block(rows_after, pool)
            vector = (
                actor
                + [float(v) for v in rowwise]
                + [float(v) for v in joint]
                + context_block(pool)
            )
            if len(vector) != FEATURE_SIZE:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            bucket = buffers[split_of(root, args.street)]
            bucket["x"].append(vector)
            bucket["y"].append(value)
            bucket["j"].append(
                sum(1 for row in rows_after for card in row if card in ("X1", "X2"))
            )
            bucket["r"].append(root)
        pending = []

    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            root = int(record["root"])
            if args.street == "t3":
                # `action_key` already spells the board this action reaches,
                # plus the card it threw away: top|mid|bot|discard.  Each
                # action re-lists the same cards, so each gets its own namer.
                for action in record["actions"]:
                    namer = JokerNamer()
                    rows_text, discard = action["action_key"].rsplit("|", 1)
                    rows_after = namer.rows(rows_text)
                    dead_after = namer(discard) + namer(record["dead"])
                    pending.append((rows_after, dead_after, action["value"], root))
                    processed += 1
            else:
                # A T4 root's `board` is the T3 action that produced it, in
                # the same top|mid|bot|discard shape, so the T3 discard joins
                # the dead cards here rather than being re-derived.  The
                # actions name cards rather than repeating the board, so one
                # namer serves the whole record.
                namer = JokerNamer()
                *row_text, t3_discard = record["board"].split("|")
                rows_root = [namer(part) for part in row_text]
                dead_root = namer(t3_discard) + namer(record["dead"])
                draw_raw = record["draw"].split(",")
                draw_cards = namer(record["draw"])
                for action in record["actions"]:
                    placed, targets = action["a"].split("@")
                    rows_after = [list(row) for row in rows_root]
                    left = list(zip(draw_raw, draw_cards))
                    legal = True
                    for name, target in zip(placed.split("+"), targets.split(",")):
                        index = next(
                            (i for i, (raw, _) in enumerate(left) if raw == name), None
                        )
                        if index is None:
                            # Labels written before the harvest compared cards
                            # by identity list placements the draw cannot make
                            # -- the two jokers share a rank and a suit, so one
                            # in the draw matched the one still in the deck.
                            # The old filter only ever ADDED actions, so the
                            # remaining ones are exactly the legal set at their
                            # original values, and dropping these is a repair
                            # rather than a resample.
                            legal = False
                            break
                        rows_after[int(target)].append(left.pop(index)[1])
                    if not legal:
                        dropped += 1
                        continue
                    if len(left) != 1:
                        raise AssertionError(f"T4 action kept {3 - len(left)} cards")
                    pending.append(
                        (rows_after, dead_root + [left[0][1]], action["v"], root)
                    )
                    processed += 1
            if len(pending) >= args.batch:
                flush()
                rate = processed / (time.time() - started)
                print(f"[{processed}] {rate:.0f} rows/s", flush=True)
    flush()

    manifest = {
        "schema": f"ofc_fl14_{args.street}_teacher/v1_best_response",
        "feature_size": FEATURE_SIZE,
        "blocks": {"actor": ACTOR_SIZE, "rowwise": ROWWISE_SIZE,
                   "joint": JOINT_SIZE, "context": CONTEXT_SIZE},
        "labels": str(args.labels),
        "opponent": "best_response_over_frontier_pool",
        "rows_encoded": processed,
        "actions_dropped_as_illegal": dropped,
        "split_rule": f"sha256('fl14-{args.street}-teacher-v1/<root>') % 100 -> 80/10/10",
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
