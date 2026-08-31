"""Card-level encoding of a HU street teacher: the distillation screen.

The 207-dim encoder spends 99.99% of its time on sixteen numbers -- the two
joint blocks, four hundred sampled completions each -- and the ablation curve
says those sixteen carry +0.26 to +0.33 of regret at T3/T2.  The question
this encoding exists to answer: can a net recover that from the raw cards,
with no rollouts at all?

The input is eight 54-card multi-hots, 432 dims, and nothing else:

    own top / mid / bot     (the candidate after-board, from the action key)
    opp top / mid / bot     (the visible opponent board)
    own dead                (prior discards plus this action's toss)
    pool                    (everything neither seat shows)

No solver runs.  Every block is a set-membership read of the JSONL, so
encoding is minutes of numpy where the 207-dim pipeline is hours of Rust.
Jokers are counted, never name-matched -- the two seats both call their
first joker X1 -- and within a block the k-th joker present fills the k-th
joker slot, which is consistent because the two jokers are physically
interchangeable.

The split is imported from `encode_hu_teacher`, not reimplemented, so a dev
regret measured here lands on exactly the roots the 207-dim bracket
(with-joint 0.292 / without 0.550 at T3-BTN) was measured on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl14_teacher import ALL_CARDS, CARD_INDEX
from ai.tutor.encode_hu_teacher import split_of, stable_id

BLOCK = 54
BLOCKS = 8  # own rows 3, opp rows 3, own dead, pool
FEATURES = BLOCK * BLOCKS
NATURALS = [c for c in ALL_CARDS if not c.startswith("X")]
JOKER_SLOTS = [CARD_INDEX["X1"], CARD_INDEX["X2"]]


def block_positions(cards: list[str], block: int) -> list[int]:
    """Flat feature positions for one block's multi-hot."""
    base = block * BLOCK
    out = []
    jokers = 0
    for card in cards:
        if card.startswith("X"):
            out.append(base + JOKER_SLOTS[jokers])
            jokers += 1
        else:
            out.append(base + CARD_INDEX[card])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--boundary", action="store_true",
                        help="one row per root: the pre-draw boundary pair, "
                             "y = the root's best action value.  The draw is "
                             "deliberately absent -- a boundary net prices "
                             "the street before it is dealt, which is where "
                             "its irreducible MAE floor comes from.")
    args = parser.parse_args()

    requests_by_id: dict[str, dict] = {}
    with args.requests.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                requests_by_id[row["id"]] = row

    buffers = {name: {"idx": [], "y": [], "r": [], "j": []}
               for name in ("fit", "dev", "test")}
    rows_done = 0
    if args.boundary:
        with args.labels.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                request = requests_by_id[record["id"]]
                hand = stable_id(record)
                bucket = buffers[split_of(str(hand))]
                own_rows = request["board"]
                opp_rows = request["opp_board"]
                dead = request["dead"]
                own_cards = [c for row in own_rows for c in row]
                opp_cards = [c for row in opp_rows for c in row]
                jokers = sum(1 for c in own_cards + opp_cards + dead
                             if c.startswith("X"))
                visible = {c for c in own_cards + opp_cards + dead
                           if not c.startswith("X")}
                pool = [c for c in NATURALS if c not in visible]
                pool += ["X1", "X2"][: 2 - jokers]
                if len(own_cards) + len(dead) + len(opp_cards) + len(pool) != 54:
                    raise AssertionError(
                        f"root {record['id']}: cards do not partition the deck"
                    )
                positions = (
                    block_positions(own_rows[0], 0)
                    + block_positions(own_rows[1], 1)
                    + block_positions(own_rows[2], 2)
                    + block_positions(opp_rows[0], 3)
                    + block_positions(opp_rows[1], 4)
                    + block_positions(opp_rows[2], 5)
                    + block_positions(dead, 6)
                    + block_positions(pool, 7)
                )
                bucket["idx"].append(np.asarray(positions, dtype=np.int16))
                bucket["y"].append(max(a["value"] for a in record["actions"]))
                bucket["r"].append(hand)
                bucket["j"].append(jokers)
                rows_done += 1
        write_out(args.out_dir, buffers, rows_done)
        return
    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            request = requests_by_id[record["id"]]
            hand = stable_id(record)
            bucket = buffers[split_of(str(hand))]

            opp_rows = request["opp_board"]
            opp_cards = [c for row in opp_rows for c in row]
            dead_prior = request["dead"]
            # Jokers by count across everything visible; the seats' names
            # collide and must never be compared.
            seen_jokers = sum(
                1 for c in opp_cards + dead_prior if c.startswith("X")
            )
            opp_positions = (
                block_positions(opp_rows[0], 3)
                + block_positions(opp_rows[1], 4)
                + block_positions(opp_rows[2], 5)
            )

            for action in record["actions"]:
                *rows_text, discard = action["action_key"].split("|")
                own_rows = [
                    [c for c in part.split(",") if c] for part in rows_text
                ]
                own_cards = [c for row in own_rows for c in row]
                dead = dead_prior + ([discard] if discard else [])
                own_jokers = sum(
                    1 for c in own_cards + [discard]
                    if c and c.startswith("X")
                )
                visible = {
                    c for c in own_cards + dead + opp_cards
                    if not c.startswith("X")
                }
                pool = [c for c in NATURALS if c not in visible]
                pool += ["X1", "X2"][: 2 - seen_jokers - own_jokers]
                if len(own_cards) + len(dead) + len(opp_cards) + len(pool) != 54:
                    raise AssertionError(
                        f"root {record['id']}: cards do not partition the deck"
                    )
                positions = (
                    block_positions(own_rows[0], 0)
                    + block_positions(own_rows[1], 1)
                    + block_positions(own_rows[2], 2)
                    + opp_positions
                    + block_positions(dead, 6)
                    + block_positions(pool, 7)
                )
                bucket["idx"].append(np.asarray(positions, dtype=np.int16))
                bucket["y"].append(action["value"])
                bucket["r"].append(hand)
                bucket["j"].append(seen_jokers + own_jokers)
                rows_done += 1
            if rows_done % 200_000 < 25:
                print(f"[{rows_done} rows]", flush=True)

    write_out(args.out_dir, buffers, rows_done)


def write_out(out_dir: Path, buffers: dict, rows_done: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, bucket in buffers.items():
        n = len(bucket["y"])
        x = np.zeros((n, FEATURES), dtype=np.float32)
        if n:
            row_ids = np.concatenate([
                np.full(len(idx), i, dtype=np.int64)
                for i, idx in enumerate(bucket["idx"])
            ])
            col_ids = np.concatenate(bucket["idx"]).astype(np.int64)
            x[row_ids, col_ids] = 1.0
        np.savez(
            out_dir / f"{name}.npz",
            x=x,
            y=np.asarray(bucket["y"], dtype=np.float32),
            roots=np.asarray(bucket["r"], dtype=np.int64),
            jokers=np.asarray(bucket["j"], dtype=np.int64),
        )
        print(f"{name}: {x.shape}", flush=True)
    print(f"encoded {rows_done} rows -> {out_dir}")


if __name__ == "__main__":
    main()
