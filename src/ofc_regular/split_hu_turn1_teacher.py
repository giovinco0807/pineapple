"""Create deterministic state-level train and holdout HU Turn1 teacher files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026071704)
    return parser.parse_args()


def state_identity(row: dict[str, Any]) -> str:
    if row.get("state_key"):
        return str(row["state_key"])
    payload = {
        key: row.get(key)
        for key in ("hand_seed", "seat", "board", "opponent_board", "dealt", "visible_dead_cards")
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def holdout_score(row: dict[str, Any], *, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}|{state_identity(row)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_rows(
    rows: Iterable[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_states = 0
    for row in rows:
        identity = state_identity(row)
        destination = "holdout" if holdout_score(row, seed=seed) < holdout_fraction else "train"
        duplicate_states += int(identity in seen)
        seen.add(identity)
        (holdout if destination == "holdout" else train).append(row)
    if not train or not holdout:
        raise ValueError("split produced an empty train or holdout set")
    summary = {
        "schema": "hu_turn1_teacher_split_summary_v1",
        "rows": len(train) + len(holdout),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "actual_holdout_fraction": len(holdout) / (len(train) + len(holdout)),
        "requested_holdout_fraction": holdout_fraction,
        "seed": seed,
        "duplicate_state_rows": duplicate_states,
        "train_seats": dict(sorted(Counter(str(row.get("seat", "unknown")) for row in train).items())),
        "holdout_seats": dict(sorted(Counter(str(row.get("seat", "unknown")) for row in holdout).items())),
    }
    return train, holdout, summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    try:
        train, holdout, summary = split_rows(
            read_jsonl(args.input),
            holdout_fraction=float(args.holdout_fraction),
            seed=int(args.seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_jsonl(args.train_output, train)
    write_jsonl(args.holdout_output, holdout)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
