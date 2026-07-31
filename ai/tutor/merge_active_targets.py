"""Merge active-teacher target JSONL files with stable target-key dedupe."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ai.tutor.weak_groups_to_active_targets import target_key


def parse_turns(raw: str) -> set[int]:
    return {int(part) for part in raw.split(",") if part.strip()}


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def merge_reasons(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    reasons: list[str] = []
    seen: set[str] = set()
    for item in list(existing.get("reasons") or []) + list(incoming.get("reasons") or []):
        reason = str(item)
        if reason in seen:
            continue
        seen.add(reason)
        reasons.append(reason)
    if reasons:
        existing["reasons"] = reasons


def merge(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    turns = parse_turns(args.turns) if args.turns else None
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    stats = Counter()

    for raw_path in args.inputs:
        path = Path(raw_path)
        stats["input_files"] += 1
        for line_no, target in iter_jsonl(path):
            stats["read"] += 1
            turn = int(target.get("turn", -1))
            if turns is not None and turn not in turns:
                stats["skipped_turn"] += 1
                continue
            key = target_key(target)
            source_ref = {"path": str(path), "line": line_no}
            if key in seen:
                stats["duplicates"] += 1
                existing = seen[key]
                existing.setdefault("merged_sources", []).append(source_ref)
                existing["merged_duplicate_count"] = int(existing.get("merged_duplicate_count", 0)) + 1
                merge_reasons(existing, target)
                continue
            copied = dict(target)
            copied["merged_sources"] = [source_ref]
            copied["merged_duplicate_count"] = 0
            seen[key] = copied
            order.append(key)
            stats["written"] += 1
            stats[f"turn_{turn}"] += 1

    with output.open("w", encoding="utf-8") as f:
        for key in order:
            f.write(json.dumps(seen[key], ensure_ascii=False) + "\n")

    summary = {
        "inputs": [str(Path(path)) for path in args.inputs],
        "output": str(output),
        "turns": sorted(turns) if turns is not None else None,
        "counts": dict(stats),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate active-teacher target JSONL files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="1,2")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(merge(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
