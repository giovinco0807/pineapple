"""Augment active teacher labels with suit permutations.

OFC has no suit ordering, so a global permutation of h/d/c/s preserves every
candidate's EV, FL, and bust labels.  This script applies that symmetry to
already-labeled teacher JSONL records without re-running the evaluator.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUITS = "hdcs"
RANKS = {str(n) for n in range(2, 10)} | {"T", "J", "Q", "K", "A"}


def is_card(value: str) -> bool:
    return len(value) >= 2 and value[-1] in SUITS and value[:-1] in RANKS


def map_card(value: Any, suit_map: dict[str, str]) -> Any:
    if not isinstance(value, str) or not is_card(value):
        return value
    return value[:-1] + suit_map[value[-1]]


def map_cards(cards: Iterable[Any], suit_map: dict[str, str]) -> list[Any]:
    return [map_card(card, suit_map) for card in cards]


def map_board(board: dict[str, Any], suit_map: dict[str, str]) -> dict[str, Any]:
    return {key: map_cards(value or [], suit_map) for key, value in board.items()}


def map_candidate(candidate: dict[str, Any], suit_map: dict[str, str]) -> dict[str, Any]:
    copied = dict(candidate)
    copied["discard"] = map_card(copied.get("discard"), suit_map)
    copied["placements"] = [
        [map_card(card, suit_map), row]
        for card, row in (copied.get("placements") or [])
    ]
    return copied


def map_record(record: dict[str, Any], suit_map: dict[str, str], augmentation_index: int) -> dict[str, Any]:
    copied = dict(record)
    for key in ("board", "opponent_board", "board_opponent"):
        if isinstance(copied.get(key), dict):
            copied[key] = map_board(copied[key], suit_map)
    for key in ("dealt", "known_discards", "exclude"):
        if isinstance(copied.get(key), list):
            copied[key] = map_cards(copied[key], suit_map)
    copied["candidates"] = [
        map_candidate(candidate, suit_map)
        for candidate in (copied.get("candidates") or [])
    ]
    copied["suit_permutation"] = "".join(suit_map[suit] for suit in SUITS)
    copied["augmentation_index"] = int(augmentation_index)
    return copied


def record_key(record: dict[str, Any]) -> str:
    stable = {
        "turn": record.get("turn"),
        "board": record.get("board"),
        "opponent_board": record.get("opponent_board") or record.get("board_opponent"),
        "dealt": record.get("dealt"),
        "known_discards": record.get("known_discards"),
        "exclude": record.get("exclude"),
        "is_btn": record.get("is_btn"),
        "candidates": [
            {
                "discard": candidate.get("discard"),
                "placements": candidate.get("placements") or [],
            }
            for candidate in (record.get("candidates") or [])
        ],
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def suit_mappings(mode: str) -> list[dict[str, str]]:
    if mode == "identity":
        return [{suit: suit for suit in SUITS}]
    return [dict(zip(SUITS, perm)) for perm in itertools.permutations(SUITS)]


def augment(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mappings = suit_mappings(args.mode)

    stats = Counter()
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            stats["input_records"] += 1
            for idx, suit_map in enumerate(mappings):
                augmented = map_record(record, suit_map, idx)
                key = record_key(augmented)
                if key in seen:
                    stats["duplicates_skipped"] += 1
                    continue
                seen.add(key)
                dst.write(json.dumps(augmented, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats[f"turn_{augmented.get('turn')}"] += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "mode": args.mode,
        "suit_permutations": len(mappings),
        **dict(stats),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Suit-augment active teacher labels")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("all", "identity"), default="all")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(augment(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
