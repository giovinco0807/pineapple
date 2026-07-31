"""Build focused active-teacher targets from source-line neighbors.

This is narrower than ``build_hybrid_active_targets.py``: each seed miss owns a
small nearest-neighbor window, so the next MC1000 relabeling batch covers the
same local pattern family without spending hours on broad random windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.build_hybrid_active_targets import miss_source_record
from ai.tutor.extract_selfplay_targets import _target_from_selfplay
from ai.tutor.weak_groups_to_active_targets import target_key


def parse_turns(raw: str) -> set[int]:
    return {int(part) for part in raw.split(",") if part.strip()}


def iter_seed_targets(paths: Iterable[Path]) -> Iterable[tuple[int, dict[str, Any]]]:
    seed_index = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                seed_index += 1
                yield seed_index, json.loads(line)


def source_key(path: Path) -> str:
    return str(path)


def collect_seed_windows(args: argparse.Namespace) -> tuple[
    list[tuple[int, dict[str, Any], Path, int]],
    dict[Path, set[int]],
    set[tuple[str, int]],
    set[str],
    Counter[str],
]:
    seed_paths = [Path(path) for path in args.seeds]
    seeds: list[tuple[int, dict[str, Any], Path, int]] = []
    windows: dict[Path, set[int]] = defaultdict(set)
    seed_source_lines: set[tuple[str, int]] = set()
    seed_target_keys: set[str] = set()
    stats: Counter[str] = Counter()

    for seed_index, seed in iter_seed_targets(seed_paths):
        source = miss_source_record(seed)
        if source is None:
            stats["seed_missing_source"] += 1
            continue
        source_path, center = source
        seeds.append((seed_index, seed, source_path, center))
        seed_source_lines.add((source_key(source_path), center))
        try:
            seed_target_keys.add(target_key(seed))
        except Exception:
            stats["seed_key_error"] += 1
        start = max(1, center - args.window)
        end = center + args.window
        for line_no in range(start, end + 1):
            windows[source_path].add(line_no)

    return seeds, windows, seed_source_lines, seed_target_keys, stats


def load_window_targets(
    windows: dict[Path, set[int]],
    turns: set[int],
    stats: Counter[str],
) -> dict[Path, dict[int, dict[str, Any]]]:
    by_source: dict[Path, dict[int, dict[str, Any]]] = {}
    for source_path, wanted_lines in sorted(windows.items(), key=lambda kv: str(kv[0])):
        if not source_path.exists():
            stats["missing_source"] += 1
            continue
        max_line = max(wanted_lines) if wanted_lines else 0
        targets_by_line: dict[int, dict[str, Any]] = {}
        with source_path.open("r", encoding="utf-8-sig") as src:
            for line_no, line in enumerate(src, start=1):
                if line_no > max_line:
                    break
                if line_no not in wanted_lines or not line.strip():
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
                targets_by_line[line_no] = target
                stats[f"turn_{turn}_available"] += 1
        by_source[source_path] = targets_by_line
    return by_source


def add_neighbor_metadata(
    target: dict[str, Any],
    seed_index: int,
    seed: dict[str, Any],
    center: int,
    distance: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    copied = dict(target)
    reasons = list(copied.get("reasons", []))
    for reason in (
        "source_neighbor",
        "source_neighbor_top1_miss",
        f"source_neighbor_window_{int(args.window)}",
    ):
        if reason not in reasons:
            reasons.append(reason)
    copied["reasons"] = reasons
    copied["active_source"] = "source_neighbor_top1_miss"
    copied["neighbor_window"] = int(args.window)
    copied["neighbor_max_per_seed"] = int(args.max_per_seed)
    copied["neighbor_seed_index"] = int(seed_index)
    copied["neighbor_seed_source"] = seed.get("source")
    copied["neighbor_seed_source_line"] = int(center)
    copied["neighbor_seed_distance"] = int(distance)
    for name in (
        "teacher_margin",
        "teacher_sims",
        "runtime_override_delta",
        "runtime_model_top1_hit",
        "runtime_final_top1_hit",
        "runtime_teacher_best_score",
    ):
        if name in seed:
            copied[f"neighbor_seed_{name}"] = seed.get(name)
    return copied


def collect_targets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    turns = parse_turns(args.turns)
    seeds, windows, seed_source_lines, seed_target_keys, stats = collect_seed_windows(args)
    window_targets = load_window_targets(windows, turns, stats)
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for seed_index, seed, source_path, center in seeds:
        targets_by_line = window_targets.get(source_path) or {}
        candidate_lines = sorted(
            targets_by_line,
            key=lambda line_no: (abs(line_no - center), line_no),
        )
        written_for_seed = 0
        for line_no in candidate_lines:
            if args.exclude_seeds and (source_key(source_path), line_no) in seed_source_lines:
                stats["excluded_seed_line"] += 1
                continue
            target = targets_by_line[line_no]
            key = target_key(target)
            if args.exclude_seeds and key in seed_target_keys:
                stats["excluded_seed_state"] += 1
                continue
            if key in seen_keys:
                stats["duplicate_target_key"] += 1
                continue
            seen_keys.add(key)
            distance = abs(line_no - center)
            selected.append(add_neighbor_metadata(target, seed_index, seed, center, distance, args))
            written_for_seed += 1
            stats[f"turn_{int(target['turn'])}_written"] += 1
            if args.max_per_seed > 0 and written_for_seed >= args.max_per_seed:
                break
        if written_for_seed == 0:
            stats["seed_no_neighbors_written"] += 1
        stats["seeds_with_neighbor"] += int(written_for_seed > 0)

    if args.max_records > 0:
        selected = selected[: args.max_records]

    stats["written"] = len(selected)
    summary = {
        "seeds": [str(path) for path in args.seeds],
        "source_files": [str(path) for path in sorted(windows, key=lambda p: str(p))],
        "turns": sorted(turns),
        "window": int(args.window),
        "max_per_seed": int(args.max_per_seed),
        "max_records": int(args.max_records),
        "exclude_seeds": bool(args.exclude_seeds),
        "seed_records": len(seeds),
        "counts": dict(stats),
    }
    return selected, summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build nearest source-neighbor active targets")
    parser.add_argument("--seeds", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="1")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--max-per-seed", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--exclude-seeds", action=argparse.BooleanOptionalAction, default=True)
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
