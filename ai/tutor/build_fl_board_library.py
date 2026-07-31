"""Build a global library of solved Fantasyland boards.

Hands are dealt unconditionally from the full 54-card deck and solved once
with fl_solver (version 2, canonical-correct since the 2026-07-31 rebuild).
Consumers then filter entries whose dealt cards are disjoint from the hero's
seen set; because library hands are uniform over C(54,n), the surviving
entries are uniform over C(unseen,n) -- exactly the conditional distribution
the normal-vs-FL subgame needs.  No per-root solving.

Workers append results in batches, so progress is visible from outside and a
killed run RESUMES by skipping the seeds already present in its shard -- the
lesson of the 96k build, whose workers held hours of results in memory with
nothing on disk when the host application crashed.

Each entry stores the dealt-hand bitmask (for the disjointness test), the
constrained row values, royalty, stay flag and foul flag, all recomputed in
Python from the canonical evaluator as an independent check on the solver.

Usage:
    python -m ai.tutor.build_fl_board_library --hands 100000 --cards 14 \
        --workers 14 --out-dir D:/ofc_data/fl_library_14
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RANKS = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUITS = {"s": 0, "h": 1, "d": 2, "c": 3}
ALL_CARDS = [rank + suit for suit in "shdc" for rank in "23456789TJQKA"] + ["X1", "X2"]
CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
LIBRARY_SCHEMA = "ofc_fl_board_library/v1"
BATCH = 200


def solver_path() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return (
        Path(__file__).resolve().parents[2]
        / "ai" / "rust_solver" / "target" / "release" / f"fl_solver{suffix}"
    )


def encode(card: str) -> dict:
    if card in ("X1", "X2"):
        return {"rank": 0, "suit": 4}
    rank = RANKS.get(card[0], int(card[0]) if card[0].isdigit() else 0)
    return {"rank": rank, "suit": SUITS[card[1]]}


def decode(card: dict, jokers_left: list[str]) -> str:
    if card.get("rank", 0) == 0:
        return jokers_left.pop(0)
    return "23456789TJQKA"[card["rank"] - 2] + "shdc"[card["suit"]]


def _solve_batch(batch_seeds: list[int], cards_per_hand: int) -> tuple[list, int]:
    from ai.engine.game_engine import (
        check_fl_entry,
        evaluate_board_with_joker_constraint,
        evaluate_hand,
        get_bottom_royalty,
        get_middle_royalty,
        get_top_royalty,
        hand_category,
    )

    hands = []
    for seed in batch_seeds:
        rng = random.Random(seed)
        deck = ALL_CARDS[:]
        rng.shuffle(deck)
        hands.append((seed, deck[:cards_per_hand]))

    payload = "\n".join(
        json.dumps({"cards": [encode(card) for card in hand], "version": 2})
        for _seed, hand in hands
    )
    # Process-level parallelism only: the v1 fallback inside the solver is
    # rayon-parallel and oversubscribes the machine otherwise.
    environment = dict(os.environ, RAYON_NUM_THREADS="1")
    result = subprocess.run(
        [str(solver_path())],
        input=payload + "\n",
        capture_output=True,
        text=True,
        env=environment,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(responses) != len(hands):
        raise RuntimeError(f"solver returned {len(responses)} of {len(hands)}")

    rows = []
    fouls = 0
    for (hand_seed, hand), response in zip(hands, responses):
        if not response.get("success"):
            continue
        placement = response["placement"]
        jokers = [card for card in hand if card in ("X1", "X2")]
        board = [
            [decode(card, jokers) for card in placement[key]]
            for key in ("top", "middle", "bottom")
        ]
        evaluation = evaluate_board_with_joker_constraint(*[list(r) for r in board])
        busted = bool(evaluation["busted"])
        if busted:
            fouls += 1
        final_rows = (evaluation["top"], evaluation["middle"], evaluation["bottom"])
        values = [
            evaluate_hand(list(final_rows[0]), 3),
            evaluate_hand(list(final_rows[1]), 5),
            evaluate_hand(list(final_rows[2]), 5),
        ]
        royalty = (
            0
            if busted
            else get_top_royalty(final_rows[0])
            + get_middle_royalty(final_rows[1])
            + get_bottom_royalty(final_rows[2])
        )
        stay = (
            not busted
            and (hand_category(values[0]) == 3 or hand_category(values[2]) >= 7)
        )
        mask = 0
        for card in hand:
            mask |= 1 << CARD_INDEX[card]
        rows.append(
            {
                "seed": hand_seed,
                "mask": mask,
                "board": board,
                "values": values,
                "royalty": royalty,
                "stay": stay,
                "busted": busted,
            }
        )
    return rows, fouls


def build_shard(args: tuple) -> dict:
    """One worker: solve a seed range, appending results in resumable batches."""
    seed_start, count, cards_per_hand, out_path = args
    out_file = Path(out_path)
    done_seeds: set[int] = set()
    if out_file.exists():
        with out_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_seeds.add(json.loads(line)["seed"])

    pending = [
        seed_start + offset
        for offset in range(count)
        if seed_start + offset not in done_seeds
    ]
    written = len(done_seeds)
    fouls_total = 0
    for start in range(0, len(pending), BATCH):
        rows, fouls = _solve_batch(pending[start : start + BATCH], cards_per_hand)
        fouls_total += fouls
        with out_file.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        written += len(rows)
        print(f"  worker@{seed_start}: {written}/{count}", flush=True)
    return {"path": str(out_path), "rows": written, "fouls": fouls_total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--cards", type=int, default=14)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed-base", type=int, default=31_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_worker = (args.hands + args.workers - 1) // args.workers
    tasks = []
    for worker in range(args.workers):
        count = min(per_worker, args.hands - worker * per_worker)
        if count <= 0:
            break
        tasks.append(
            (
                args.seed_base + worker * per_worker,
                count,
                args.cards,
                args.out_dir / f"shard_{worker:02d}.jsonl",
            )
        )
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for outcome in pool.map(build_shard, tasks):
            results.append(outcome)
    manifest = {
        "schema": LIBRARY_SCHEMA,
        "hands": args.hands,
        "cards_per_hand": args.cards,
        "seed_base": args.seed_base,
        "solver": "fl_solver v2 canonical (rebuilt 2026-07-31)",
        "rows": sum(r["rows"] for r in results),
        "fouled_boards": sum(r["fouls"] for r in results),
        "elapsed_seconds": time.time() - started,
        "shards": [r["path"] for r in results],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
