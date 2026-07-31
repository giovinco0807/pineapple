"""Filter teacher JSONL records to states present in active-target JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ai.tutor.audit_teacher_label_stability import state_key


def load_target_keys(paths: list[Path], turns: set[int]) -> tuple[set[str], Counter]:
    keys: set[str] = set()
    stats = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                stats["targets_read"] += 1
                target = json.loads(line)
                turn = int(target.get("turn", -1))
                if turn not in turns:
                    stats["targets_skipped_turn"] += 1
                    continue
                key = state_key(target)
                if key in keys:
                    stats["target_duplicate_state"] += 1
                keys.add(key)
                stats[f"target_turn_{turn}"] += 1
    return keys, stats


def filter_records(args: argparse.Namespace) -> dict[str, Any]:
    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    target_paths = [Path(path) for path in args.targets]
    target_keys, stats = load_target_keys(target_paths, turns)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with Path(args.teacher).open("r", encoding="utf-8-sig") as src, output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            stats["teacher_read"] += 1
            record = json.loads(line)
            turn = int(record.get("turn", -1))
            if turn not in turns:
                stats["teacher_skipped_turn"] += 1
                continue
            key = state_key(record)
            if key not in target_keys:
                stats["teacher_skipped_not_target"] += 1
                continue
            if key in seen and not args.keep_duplicates:
                stats["teacher_skipped_duplicate_state"] += 1
                continue
            seen.add(key)
            record.setdefault("source_teacher_line", line_no)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            stats[f"teacher_turn_{turn}"] += 1

    summary = {
        "targets": [str(path) for path in target_paths],
        "teacher": str(args.teacher),
        "output": str(output),
        "turns": sorted(turns),
        "target_states": len(target_keys),
        "written": written,
        "keep_duplicates": bool(args.keep_duplicates),
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Filter teacher records to active-target states")
    parser.add_argument("--targets", action="append", required=True, help="Active target JSONL; repeatable")
    parser.add_argument("--teacher", required=True, help="Teacher JSONL with candidates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="0,1,2,3,4")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(filter_records(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
