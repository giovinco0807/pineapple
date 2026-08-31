"""Encode the HU T3 teachers: the first opponent-aware feature set.

Own block (110) is the proven FL14 v2 vector of hero's after-board.  The new
opponent block (97) runs the same machinery over the opponent's visible
board -- its made values (actor 48), its per-row completion outlook over the
same unseen pool (rowwise 41), and what its rows can reach together
(joint 8).  Contested-ness is therefore present implicitly: the net sees
both sides' outlooks over one shared deck.  Explicit margin features lost
every dev screen they were tried in on the own-hand track, so they are not
assumed here either; if the worst hands say otherwise later, that is the
method for adding them.

Seats are encoded separately: a BTN decision faces an eleven-card opponent
board (joint exact), a BB decision a nine-card one (joint sampled 400), and
the widths differ, so one model per seat.

Pool discipline: the pool is hero's unseen -- the deck minus hero's board,
dead, draw AND the opponent's visible cards.  Jokers are counted across both
boards, never name-matched (each seat names its own first joker X1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl14_teacher import (
    ALL_CARDS,
    CARD_INDEX,
    actor_block,
    allocation_rank_block,
    context_block,
    fetch_blocks,
    seen_mask,
)
from ai.tutor.solver_paths import _solver_path

OWN_SIZE = 110
OPP_SIZE = 48 + 41 + 8
FEATURE_SIZE = OWN_SIZE + OPP_SIZE  # 207


def stable_id(record: dict) -> int:
    """A decision identity that survives sharding.

    Not `encode_fl14_teacher.stable_root_id`: that one reads
    `record.get("id", record["root"])`, and Python evaluates the default
    eagerly, so a record carrying only `id` raises.  The HU teachers write
    `id` alone.
    """
    raw = str(record.get("id", record.get("root", 0)))
    try:
        return int(raw)
    except ValueError:
        return int.from_bytes(
            hashlib.sha256(raw.encode()).digest()[:8], "big"
        ) & ((1 << 63) - 1)


def split_of(hand: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"hu-t3-teacher-v1/{hand}".encode()).digest()[:4], "big"
    ) % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def canonical_jokers(rows: list[list[str]], start: int) -> tuple[list[list[str]], int]:
    """Rename X* cards to globally unique X1/X2 across both boards."""
    used = start
    out = []
    for row in rows:
        named = []
        for card in row:
            if card.startswith("X"):
                used += 1
                if used > 2:
                    raise ValueError("more than two jokers visible")
                named.append(f"X{used}")
            else:
                named.append(card)
        out.append(named)
    return out, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    solver = str(_solver_path(args.workspace_root))

    requests_by_id: dict[str, dict] = {}
    with args.requests.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            requests_by_id[row["id"]] = row

    buffers = {
        name: {"x": [], "y": [], "j": [], "r": []} for name in ("fit", "dev", "test")
    }
    pending: list[tuple[list[list[str]], list[str], list[list[str]], float, int]] = []
    processed = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        block_requests = []
        # Deduplicated on (board, pool): within a root the opponent's board is
        # fixed and hero's discard takes three values, so the opponent's
        # blocks are three computations instead of one per action.
        seen_keys: dict[tuple, str] = {}
        slot_of: list[tuple[str, str]] = []
        for position, (own_rows, own_dead, opp_rows, _value, _hand) in enumerate(pending):
            all_seen = (
                [c for row in own_rows for c in row]
                + own_dead
                + [c for row in opp_rows for c in row]
            )
            if bin(seen_mask(all_seen)).count("1") != len(all_seen):
                raise AssertionError(f"a card repeats in {all_seen}")
            seen = seen_mask(all_seen)
            pool = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)]
            ids = []
            for rows in (own_rows, opp_rows):
                key = (tuple(tuple(sorted(row)) for row in rows), tuple(pool))
                if key not in seen_keys:
                    block_id = str(len(seen_keys))
                    seen_keys[key] = block_id
                    block_requests.append(
                        {
                            "id": block_id,
                            # Seed rides the board so identical requests stay
                            # identical under a sampled joint block.
                            "seed": block_id,
                            "board": {
                                "top": rows[0],
                                "middle": rows[1],
                                "bottom": rows[2],
                            },
                            "pool": pool,
                        }
                    )
                ids.append(seen_keys[key])
            slot_of.append((ids[0], ids[1]))
        blocks = fetch_blocks(block_requests, args.workspace_root, 400, solver)
        for position, (own_rows, own_dead, opp_rows, value, hand) in enumerate(pending):
            own_id, opp_id = slot_of[position]
            own_rowwise, own_joint = blocks[own_id]
            opp_rowwise, opp_joint = blocks[opp_id]
            all_seen = (
                [c for row in own_rows for c in row]
                + own_dead
                + [c for row in opp_rows for c in row]
            )
            seen = seen_mask(all_seen)
            pool = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)]
            own_actor, _ = actor_block(own_rows, pool)
            opp_actor, _ = actor_block(opp_rows, pool)
            vector = (
                own_actor
                + [float(v) for v in own_rowwise]
                + [float(v) for v in own_joint]
                + context_block(pool)
                + allocation_rank_block(own_rows)
                + opp_actor
                + [float(v) for v in opp_rowwise]
                + [float(v) for v in opp_joint]
            )
            if len(vector) != FEATURE_SIZE:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            bucket = buffers[split_of(str(hand))]
            bucket["x"].append(vector)
            bucket["y"].append(value)
            bucket["j"].append(
                sum(1 for row in own_rows for c in row if c.startswith("X"))
            )
            bucket["r"].append(hand)
        pending = []

    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            request = requests_by_id[record["id"]]
            hand = stable_id(record)
            # The request's own board names restart joker numbering per seat;
            # canonicalise across the pair so the pool count is honest.
            opp_rows, used = canonical_jokers(request["opp_board"], 0)
            for action in record["actions"]:
                *rows_text, discard = action["action_key"].split("|")
                own_rows_raw = [
                    [c for c in part.split(",") if c] for part in rows_text
                ]
                own_rows, _ = canonical_jokers(own_rows_raw, used)
                # T0 places all five cards, so its action key carries an
                # empty discard field.  An empty name is not a card and must
                # not reach the seen mask.
                dead_raw = request["dead"] + ([discard] if discard else [])
                own_dead = []
                jokers_used = used + sum(
                    1 for row in own_rows_raw for c in row if c.startswith("X")
                )
                for card in dead_raw:
                    if card.startswith("X"):
                        jokers_used += 1
                        own_dead.append(f"X{jokers_used}")
                    else:
                        own_dead.append(card)
                pending.append((own_rows, own_dead, opp_rows, action["value"], hand))
                processed += 1
                if len(pending) >= args.batch:
                    flush()
    flush()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "ofc_hu_t3_teacher_encoded/v1",
        "feature_size": FEATURE_SIZE,
        "blocks": {
            "own": "actor48+rowwise41+joint8+context7+alloc6 (fl14 v2)",
            "opp": "actor48+rowwise41+joint8 over the shared pool",
        },
        "labels": str(args.labels),
        "splits": {},
    }
    for name, bucket in buffers.items():
        np.savez_compressed(
            args.out_dir / f"{name}.npz",
            x=np.asarray(bucket["x"], dtype=np.float32),
            y=np.asarray(bucket["y"], dtype=np.float32),
            jokers=np.asarray(bucket["j"], dtype=np.int8),
            roots=np.asarray(bucket["r"], dtype=np.int64),
        )
        manifest["splits"][name] = {
            "rows": len(bucket["y"]),
            "roots": len(set(bucket["r"])),
        }
        print(name, manifest["splits"][name])
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"encoded {processed} actions -> {args.out_dir}")


if __name__ == "__main__":
    main()
