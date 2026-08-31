"""Extract high-value states for active teacher / DAgger regeneration.

The output is a JSONL of decision states that should be sent back through a
deeper teacher.  This script does not impose rules on play; it only finds
states where the current policy likely needs better labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


HIGH_RANKS = {"Q", "K", "A"}


def card_rank(card: str) -> str:
    if str(card).startswith("X"):
        return "X"
    return str(card)[0] if card else ""


def flatten_board(board: dict) -> list[str]:
    return (
        list(board.get("top", []))
        + list(board.get("middle", []))
        + list(board.get("mid", []))
        + list(board.get("bottom", []))
        + list(board.get("bot", []))
    )


def board_for_teacher(board: dict) -> dict:
    return {
        "top": list(board.get("top", [])),
        "mid": list(board.get("middle", [])) + list(board.get("mid", [])),
        "bot": list(board.get("bottom", [])) + list(board.get("bot", [])),
    }


def has_high_route(cards: list[str]) -> bool:
    return any(card_rank(c) in HIGH_RANKS or card_rank(c) == "X" for c in cards)


def top_has_fl_shape(top: list[str]) -> bool:
    ranks = [card_rank(c) for c in top]
    jokers = ranks.count("X")
    counts = Counter(r for r in ranks if r != "X")
    for r in HIGH_RANKS:
        if counts.get(r, 0) + jokers >= 2:
            return True
    return any(count + jokers >= 3 for count in counts.values()) or jokers >= 2


def reasons_for_record(record: dict, turns: set[int]) -> list[str]:
    turn_log = record.get("turn_log", {})
    turn = int(turn_log.get("turn", -1))
    if turn not in turns:
        return []

    player = str(turn_log.get("player", 0))
    hand = record.get("hand_result", {})
    busted = bool(hand.get("busted", {}).get(player, False))
    fl_entry = bool(hand.get("fl_entry", {}).get(player, False))
    board = turn_log.get("board_self", {})
    dealt = list(turn_log.get("dealt_cards", []))
    top = list(board.get("top", []))
    reasons = []

    if turn <= 1 and has_high_route(dealt + top) and not fl_entry:
        reasons.append("missed_early_fl_route")
    if turn <= 2 and has_high_route(dealt + top):
        reasons.append("high_card_route_decision")
    if busted and (top_has_fl_shape(top) or has_high_route(top)):
        reasons.append("fl_shape_but_foul")
    if turn >= 3 and top_has_fl_shape(top):
        reasons.append("late_fl_protection")

    return reasons


def extract(args: argparse.Namespace) -> None:
    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    turns = {int(t) for t in args.turns.split(",") if t.strip()}

    stats = Counter()
    written = 0
    total_lines = 0

    with output_path.open("w", encoding="utf-8") as dst:
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as src:
                for source_line, line in enumerate(src, start=1):
                    total_lines += 1
                    if args.max_lines and total_lines > args.max_lines:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        continue
                    turn_log = record.get("turn_log", {})
                    reasons = reasons_for_record(record, turns)
                    if not reasons:
                        continue

                    player = str(turn_log.get("player", 0))
                    board_opp = turn_log.get("board_opponent", {})
                    discards_self = list(turn_log.get("discards_self", []))
                    excludes = flatten_board(board_opp) + discards_self
                    payload = {
                        "source": str(input_path),
                        "source_line": source_line,
                        "turn": int(turn_log.get("turn", -1)),
                        "board": board_for_teacher(turn_log.get("board_self", {})),
                        "opponent_board": board_for_teacher(board_opp),
                        "dealt": list(turn_log.get("dealt_cards", [])),
                        "known_discards": discards_self,
                        "exclude": excludes,
                        "is_btn": bool(turn_log.get("is_btn", False)),
                        "player": int(turn_log.get("player", 0)),
                        "reasons": reasons,
                        "busted": bool(record.get("hand_result", {}).get("busted", {}).get(player, False)),
                        "fl_entry": bool(record.get("hand_result", {}).get("fl_entry", {}).get(player, False)),
                        "fl_cards": int(record.get("hand_result", {}).get("fl_cards", {}).get(player, 0)),
                        "reward": float(record.get("reward", 0.0)),
                    }
                    dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written += 1
                    for reason in reasons:
                        stats[reason] += 1
                    if args.max_records and written >= args.max_records:
                        break
                if args.max_records and written >= args.max_records:
                    break
                if args.max_lines and total_lines > args.max_lines:
                    break

    summary = {
        "inputs": [str(path) for path in input_paths],
        "output": str(output_path),
        "records": written,
        "turns": sorted(turns),
        "reason_counts": dict(stats),
    }
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Extracted {written:,} active-teacher states -> {output_path}")
    print(f"Reasons: {dict(stats)}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract active teacher target states from self-play JSONL")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="0,1,2,3,4")
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(argv)
    extract(args)


if __name__ == "__main__":
    main()
