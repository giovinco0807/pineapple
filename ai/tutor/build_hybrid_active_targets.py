"""Build active teacher targets from hybrid shortlist audit misses.

The audit misses point at hard decision states.  To avoid training directly on
holdout rows, this script reads the source self-play logs around each miss and
extracts nearby T1/T2 decision states.  The result can be sent through
``generate_active_teacher.py --batch-engine`` so Rust ``prob_engine`` does the
expensive data generation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.extract_selfplay_targets import _target_from_selfplay


def iter_misses(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def miss_source_record(miss: dict[str, Any]) -> tuple[Path, int] | None:
    record = miss.get("source_record") or {}
    source = record.get("source") or miss.get("source")
    line = record.get("source_line")
    if line is None:
        line = miss.get("source_line")
    if not source or line is None:
        return None
    return Path(str(source)), int(line)


def collect_windows(args: argparse.Namespace) -> dict[Path, set[int]]:
    windows: dict[Path, set[int]] = defaultdict(set)
    for item in iter_misses(Path(path) for path in args.misses):
        source = miss_source_record(item)
        if source is None:
            continue
        path, center = source
        start = max(1, center - args.window)
        end = center + args.window
        for line_no in range(start, end + 1):
            if not args.include_miss_lines and line_no == center:
                continue
            windows[path].add(line_no)
    return windows


def collect_targets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    windows = collect_windows(args)
    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    stats: Counter[str] = Counter()

    for source_path, wanted_lines in sorted(windows.items(), key=lambda kv: str(kv[0])):
        if not source_path.exists():
            stats["missing_source"] += 1
            continue
        max_line = max(wanted_lines) if wanted_lines else 0
        with source_path.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, start=1):
                if line_no > max_line:
                    break
                if line_no not in wanted_lines:
                    continue
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                target = _target_from_selfplay(source_path, line_no, record)
                if target is None:
                    stats["skipped"] += 1
                    continue
                turn = int(target.get("turn", -1))
                if turn not in turns:
                    stats[f"turn_{turn}_filtered"] += 1
                    continue
                key = (str(source_path), line_no, turn)
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)
                target.setdefault("reasons", [])
                target["reasons"] = list(target["reasons"]) + ["hybrid_pool_miss_neighbor"]
                target["active_source"] = "hybrid_shortlist_miss_neighbor"
                target["neighbor_window"] = int(args.window)
                targets.append(target)
                stats[f"turn_{turn}_available"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(targets)
    if args.max_records > 0:
        targets = targets[: args.max_records]
    stats["written"] = len(targets)
    for target in targets:
        stats[f"turn_{int(target['turn'])}_written"] += 1

    summary = {
        "misses": [str(path) for path in args.misses],
        "source_files": [str(path) for path in sorted(windows)],
        "turns": sorted(turns),
        "window": int(args.window),
        "include_miss_lines": bool(args.include_miss_lines),
        "max_records": int(args.max_records),
        "seed": int(args.seed),
        "counts": dict(stats),
    }
    return targets, summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build active T1/T2 targets from hybrid shortlist misses")
    parser.add_argument("--misses", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="1,2")
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--include-miss-lines", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    targets, summary = collect_targets(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for target in targets:
            f.write(json.dumps(target, ensure_ascii=False) + "\n")
    summary["output"] = str(output)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
