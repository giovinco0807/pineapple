"""Select diverse oracle input rows from an existing teacher JSONL.

This is a lightweight prefilter before expensive capped/exact oracle runs.
Many OFC teacher files contain suit-expanded copies of the same source hand.
For oracle generation we usually want diverse base states first, so this script
keeps at most N rows per source key and writes a compact subset JSONL.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def source_group_key(record: dict[str, Any], fallback_line: int) -> str:
    source = record.get("source")
    source_line = record.get("source_line")
    decision = record.get("source_decision_index")
    if source is not None and source_line is not None:
        return json.dumps(
            {
                "source": str(source),
                "source_line": int(source_line),
                "source_decision_index": None if decision is None else int(decision),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return f"input_line:{fallback_line}"


def card_rank(card: Any) -> str:
    text = str(card)
    if not text:
        return ""
    if text.startswith("X"):
        return "X"
    return text[0]


def rank_group_key(record: dict[str, Any]) -> str:
    board = record.get("board") or {}
    stable = {
        "turn": int(record.get("turn", -1)),
        "top": tuple(card_rank(card) for card in (board.get("top") or [])),
        "mid": tuple(card_rank(card) for card in ((board.get("mid") or []) + (board.get("middle") or []))),
        "bot": tuple(card_rank(card) for card in ((board.get("bot") or []) + (board.get("bottom") or []))),
        "dealt": tuple(sorted(card_rank(card) for card in (record.get("dealt") or []))),
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def load_indices_file(path: str, field: str) -> set[int]:
    if not path:
        return set()
    indices: set[int] = set()
    for _line_no, record in iter_jsonl(Path(path)):
        value = record
        for part in field.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            indices.add(int(value))
    return indices


def load_excluded_keys(paths: list[str], exclude_rank_groups: bool) -> tuple[set[str], set[str]]:
    source_keys: set[str] = set()
    rank_keys: set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude file not found: {path}")
        for line_no, record in iter_jsonl(path):
            source_keys.add(source_group_key(record, line_no))
            if exclude_rank_groups:
                rank_keys.add(rank_group_key(record))
    return source_keys, rank_keys


def select(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    indices = {int(part) for part in args.indices.split(",") if part.strip()}
    indices.update(load_indices_file(args.indices_file, args.index_field))
    excluded_source_keys, excluded_rank_keys = load_excluded_keys(
        args.exclude_files,
        args.exclude_rank_groups,
    )
    group_counts: Counter[str] = Counter()
    rank_group_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []

    for line_no, record in iter_jsonl(Path(args.input)):
        stats["read"] += 1
        input_index = line_no - 1
        if indices and input_index not in indices:
            stats["skipped_index"] += 1
            continue
        turn = int(record.get("turn", -1))
        if turn not in turns:
            stats["skipped_turn"] += 1
            continue
        group_key = source_group_key(record, line_no)
        if group_key in excluded_source_keys:
            stats["skipped_excluded_source_group"] += 1
            continue
        if group_counts[group_key] >= args.max_per_source:
            stats["skipped_source_group_cap"] += 1
            continue
        rank_key = rank_group_key(record)
        if args.exclude_rank_groups and rank_key in excluded_rank_keys:
            stats["skipped_excluded_rank_group"] += 1
            continue
        if args.max_per_rank_group > 0 and rank_group_counts[rank_key] >= args.max_per_rank_group:
            stats["skipped_rank_group_cap"] += 1
            continue
        copied = dict(record)
        copied["oracle_input_source"] = str(args.input)
        copied["oracle_input_line"] = int(line_no)
        copied["oracle_group_key"] = group_key
        copied["oracle_rank_group_key"] = rank_key
        selected.append(copied)
        group_counts[group_key] += 1
        rank_group_counts[rank_key] += 1
        stats["selected"] += 1
        stats[f"turn_{turn}"] += 1
        if args.limit > 0 and len(selected) >= args.limit:
            break

    with output.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(output),
        "turns": sorted(turns),
        "indices": sorted(indices),
        "indices_file": str(args.indices_file),
        "index_field": str(args.index_field),
        "exclude_files": [str(path) for path in args.exclude_files],
        "exclude_rank_groups": bool(args.exclude_rank_groups),
        "excluded_source_groups": len(excluded_source_keys),
        "excluded_rank_groups": len(excluded_rank_keys),
        "limit": int(args.limit),
        "max_per_source": int(args.max_per_source),
        "max_per_rank_group": int(args.max_per_rank_group),
        "selected": len(selected),
        "unique_source_groups": len(group_counts),
        "unique_rank_groups": len(rank_group_counts),
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select diverse rows for oracle generation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="2")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--max-per-rank-group", type=int, default=0)
    parser.add_argument("--indices", default="", help="Comma-separated zero-based input row indices")
    parser.add_argument("--indices-file", default="", help="JSONL whose rows contain indices to select")
    parser.add_argument("--index-field", default="record_index", help="Field path to read from --indices-file")
    parser.add_argument("--exclude-files", nargs="*", default=[], help="JSONL files whose source groups should be skipped")
    parser.add_argument("--exclude-rank-groups", action="store_true", help="Also skip rank-abstracted groups from --exclude-files")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(select(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
