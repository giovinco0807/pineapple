"""Build a fixed balanced teacher-evaluation holdout from self-play logs.

The output uses the target schema consumed by ai.training.generate_active_teacher.
It is intended for evaluation only; do not mix these records into training.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _norm_source(path: str | None) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).replace("/", "\\").lower()


def _source_key(path: Path, line_no: int) -> tuple[str, int]:
    return (_norm_source(str(path)), int(line_no))


def _source_key_from_record(record: dict) -> tuple[str, int] | None:
    source = record.get("source")
    line = record.get("source_line")
    if source is None or line is None:
        return None
    try:
        return (_norm_source(str(source)), int(line))
    except (TypeError, ValueError):
        return None


def load_excluded(paths: list[str]) -> set[tuple[str, int]]:
    excluded: set[tuple[str, int]] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = _source_key_from_record(record)
                if key is not None:
                    excluded.add(key)
    return excluded


def _row(board: dict, *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(board.get(name, []) or [])
    return out


def flatten_board(board: dict) -> list[str]:
    return _row(board, "top") + _row(board, "middle", "mid") + _row(board, "bottom", "bot")


def board_for_teacher(board: dict) -> dict:
    return {
        "top": list(board.get("top", [])),
        "mid": _row(board, "middle", "mid"),
        "bot": _row(board, "bottom", "bot"),
    }


def target_from_selfplay(input_path: Path, line_no: int, record: dict, split: str) -> dict | None:
    turn_log = record.get("turn_log") or {}
    turn = int(turn_log.get("turn", -1))
    dealt = list(turn_log.get("dealt_cards") or [])
    if turn < 0 or not dealt:
        return None

    player = str(turn_log.get("player", 0))
    board_opp = turn_log.get("board_opponent") or {}
    discards_self = list(turn_log.get("discards_self") or [])
    excludes = flatten_board(board_opp) + discards_self
    hand = record.get("hand_result") or {}
    return {
        "split": split,
        "source": str(input_path),
        "source_line": int(line_no),
        "turn": turn,
        "board": board_for_teacher(turn_log.get("board_self") or {}),
        "opponent_board": board_for_teacher(board_opp),
        "dealt": dealt,
        "known_discards": discards_self,
        "exclude": excludes,
        "is_btn": bool(turn_log.get("is_btn", False)),
        "player": int(turn_log.get("player", 0)),
        "selection_reason": "balanced_eval_holdout",
        "busted": bool((hand.get("busted") or {}).get(player, False)),
        "fl_entry": bool((hand.get("fl_entry") or {}).get(player, False)),
        "fl_cards": int((hand.get("fl_cards") or {}).get(player, 0)),
        "reward": float(record.get("reward", 0.0)),
    }


def collect_candidates(
    input_paths: list[Path],
    turns: set[int],
    excluded: set[tuple[str, int]],
    split: str,
) -> dict[int, list[dict]]:
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                if _source_key(input_path, line_no) in excluded:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = target_from_selfplay(input_path, line_no, record, split)
                if target is None:
                    continue
                turn = int(target["turn"])
                if turn in turns:
                    by_turn[turn].append(target)
    return by_turn


def build(args: argparse.Namespace) -> dict:
    input_paths = [Path(path) for path in args.inputs]
    turns = {int(t) for t in args.turns.split(",") if t.strip()}
    excluded = load_excluded(args.exclude_jsonl)
    by_turn = collect_candidates(input_paths, turns, excluded, args.split)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    available_counts = {str(turn): len(by_turn.get(turn, [])) for turn in sorted(turns)}
    selected_counts: Counter[int] = Counter()
    for turn in sorted(turns):
        candidates = list(by_turn.get(turn, []))
        rng.shuffle(candidates)
        if len(candidates) < args.per_turn and not args.allow_short:
            raise SystemExit(
                f"Not enough T{turn} candidates: requested {args.per_turn}, available {len(candidates)}. "
                "Use --allow-short to write a partial holdout."
            )
        chosen = candidates[: args.per_turn]
        for i, target in enumerate(chosen):
            target["holdout_id"] = f"{args.split}-T{turn}-{i:05d}"
            target["selection_seed"] = int(args.seed)
        selected.extend(chosen)
        selected_counts[turn] = len(chosen)

    rng.shuffle(selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        for item in selected:
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output),
        "split": args.split,
        "inputs": [str(path) for path in input_paths],
        "exclude_jsonl": list(args.exclude_jsonl),
        "excluded_source_lines": len(excluded),
        "turns": sorted(turns),
        "per_turn_requested": int(args.per_turn),
        "available_by_turn": available_counts,
        "selected_by_turn": {str(turn): int(selected_counts[turn]) for turn in sorted(turns)},
        "records": len(selected),
        "seed": int(args.seed),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build balanced OFC tutor eval holdout targets")
    parser.add_argument("inputs", nargs="+", help="Self-play JSONL inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="0,1,2,3")
    parser.add_argument("--per-turn", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--split", default="eval_holdout_20260523")
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = build(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
