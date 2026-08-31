"""Shortlist a T2 root's actions with the trained chooser.

`fl_solver teach-t2 --enumerate-only` writes every action key of a root without
pricing any of them; this scores them and writes the top K back for
`--keep-file`.  Pricing five actions instead of twenty-four is where the time
goes -- measured 0.337 s an action against 8 ms to encode one.

The features are built by importing `encode_fl14_teacher`'s own blocks rather
than reimplementing them: a chooser fed a vector its training never saw ranks
something else.  That includes the joint block's seed scope, `root/<id>`, which
is what shares one set of sampled completions across a root's actions and keeps
the sampling noise out of the ordering being read.

What this does **not** do is decide whether K is enough.  That is
`--report-hits` against a fully priced corpus, and it is a property of the
model, not of this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai.tutor.encode_fl14_teacher import (
    ALL_CARDS,
    CARD_INDEX,
    JokerNamer,
    actor_block,
    allocation_rank_block,
    context_block,
    fetch_blocks,
    seen_mask,
    stable_root_id,
)
from ai.tutor.solver_paths import _solver_path
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

ROW_NAMES = ("top", "middle", "bottom")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def named_action(action_key: str, dead_text: str) -> tuple[list[list[str]], list[str]]:
    """The board and dead cards an action leaves, jokers named as X1/X2.

    Verbatim from `encode_fl14_teacher`'s T2/T3 branch, including the fresh
    namer per action: each action re-lists the same cards, so the X1/X2
    assignment has to restart or the pool it implies is wrong.
    """
    namer = JokerNamer()
    rows_text, discard = action_key.rsplit("|", 1)
    rows_after = namer.rows(rows_text)
    dead_after = namer(discard) + namer(dead_text)
    return rows_after, dead_after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True, help="teach-t2 --enumerate-only output")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--joint-samples", type=int, default=400)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--solver", default=None, help="override the block binary")
    args = parser.parse_args()
    solver = args.solver or str(_solver_path(args.workspace_root))
    if not Path(solver).exists():
        raise SystemExit(f"no block binary at {solver}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    width = checkpoint["input_dim"]
    if width not in (104, 110):
        raise SystemExit(f"a T2 chooser is 104 or 110 dims, not {width}")

    records = read_jsonl(args.actions)
    requests: list[dict[str, Any]] = []
    index: list[tuple[int, str, list[list[str]]]] = []
    for position, record in enumerate(records):
        root = stable_root_id(record)
        for key in record["action_keys"]:
            rows_after, dead_after = named_action(key, record["dead"])
            all_seen = [c for row in rows_after for c in row] + dead_after
            if bin(seen_mask(all_seen)).count("1") != len(all_seen):
                raise AssertionError(f"a card repeats in {all_seen}")
            seen = seen_mask(all_seen)
            requests.append(
                {
                    "id": str(len(requests)),
                    # Shared across a root's actions; see the module note.
                    "seed": f"root/{root}",
                    "board": dict(zip(ROW_NAMES, rows_after)),
                    "pool": [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)],
                }
            )
            index.append((position, key, rows_after))

    blocks = fetch_blocks(requests, args.workspace_root, args.joint_samples, solver)

    vectors: list[list[float]] = []
    for slot, (_, _, named) in enumerate(index):
        rowwise, joint = blocks[str(slot)]
        pool = requests[slot]["pool"]
        actor, _ = actor_block(named, pool)
        vector = actor + [float(v) for v in rowwise] + [float(v) for v in joint] + context_block(pool)
        if width == 110:
            vector += allocation_rank_block(named)
        if len(vector) != width:
            raise AssertionError(f"feature width {len(vector)} against model {width}")
        vectors.append(vector)

    features = (torch.tensor(np.array(vectors, dtype=np.float32)) - checkpoint["input_mean"]) / checkpoint[
        "input_std"
    ]
    with torch.no_grad():
        scores = model(features).squeeze(-1).numpy()

    by_root: dict[int, list[tuple[float, str]]] = {}
    for slot, (position, key, _) in enumerate(index):
        by_root.setdefault(position, []).append((float(scores[slot]), key))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept_total = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for position, record in enumerate(records):
            ranked = sorted(by_root[position], key=lambda pair: (-pair[0], pair[1]))
            keep = [key for _, key in ranked[: args.keep]]
            kept_total += len(keep)
            handle.write(
                json.dumps({"id": record["id"], "keep": keep}, separators=(",", ":")) + "\n"
            )
    print(
        json.dumps(
            {
                "roots": len(records),
                "actions_scored": len(index),
                "actions_kept": kept_total,
                "keep": args.keep,
                "model": str(args.model),
                "input_dim": width,
                "joint_samples": args.joint_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
