"""Aggregate sharded HU T2 Stage9f profile-canary outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=[],
        help="Completed shard result directory. May be repeated.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Root containing completed shard result subdirectories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_input_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = list(args.input_dir)
    if args.results_root is not None:
        dirs.extend(
            path
            for path in sorted(args.results_root.iterdir())
            if path.is_dir() and (path / "DONE").exists()
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("no completed shard input dirs found")
    return unique


def aggregate_summaries(input_dirs: list[Path]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        summary_path = input_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
        summary["_input_dir"] = str(input_dir)
        summaries.append(summary)

    paired_seeds = sum(int(summary.get("paired_seeds", 0)) for summary in summaries)
    hands = sum(int(summary.get("hands", 0)) for summary in summaries)
    weighted_score_sum = sum(
        float(summary.get("avg_score_per_hand_for_a", 0.0)) * int(summary.get("hands", 0))
        for summary in summaries
    )
    topk_decisions_written = sum(
        int(summary.get("topk_decisions_written", 0)) for summary in summaries
    )
    topk_realized_delta_count = sum(
        int(summary.get("topk_realized_delta_count", 0)) for summary in summaries
    )
    topk_realized_override_count = sum(
        int(summary.get("topk_realized_override_count", 0)) for summary in summaries
    )
    topk_non_fired_nonzero_count = sum(
        int(summary.get("topk_non_fired_nonzero_count", 0)) for summary in summaries
    )
    topk_non_fired_delta_sum = sum(
        float(summary.get("topk_non_fired_delta_sum", 0.0)) for summary in summaries
    )
    topk_non_fired_delta_max_abs = max(
        (float(summary.get("topk_non_fired_delta_max_abs", 0.0)) for summary in summaries),
        default=0.0,
    )
    hu_turn1_decisions_written = sum(
        int(summary.get("hu_turn1_decisions_written", 0)) for summary in summaries
    )
    hu_turn1_realized_delta_count = sum(
        int(summary.get("hu_turn1_realized_delta_count", 0)) for summary in summaries
    )
    hu_turn1_realized_override_count = sum(
        int(summary.get("hu_turn1_realized_override_count", 0)) for summary in summaries
    )
    hu_turn1_non_fired_final_mismatch_count = sum(
        int(summary.get("hu_turn1_non_fired_final_mismatch_count", 0)) for summary in summaries
    )
    realized_delta_sum = sum(
        float(summary.get("topk_realized_override_delta_mean", 0.0))
        * int(summary.get("topk_realized_override_count", 0))
        for summary in summaries
    )
    realized_delta_mean = (
        realized_delta_sum / topk_realized_override_count
        if topk_realized_override_count
        else 0.0
    )

    first = summaries[0]
    return {
        "profile_a": first.get("profile_a", ""),
        "profile_b": first.get("profile_b", ""),
        "paired_seeds": paired_seeds,
        "hands": hands,
        "seed": first.get("seed", ""),
        "avg_score_per_hand_for_a": weighted_score_sum / hands if hands else 0.0,
        "std_error": None,
        "ci95_low": None,
        "ci95_high": None,
        "topk_decisions_written": topk_decisions_written,
        "topk_realized_delta_count": topk_realized_delta_count,
        "topk_realized_override_count": topk_realized_override_count,
        "topk_realized_override_delta_mean": realized_delta_mean,
        "topk_non_fired_nonzero_count": topk_non_fired_nonzero_count,
        "topk_non_fired_delta_sum": topk_non_fired_delta_sum,
        "topk_non_fired_delta_max_abs": topk_non_fired_delta_max_abs,
        "hu_turn1_decisions_written": hu_turn1_decisions_written,
        "hu_turn1_realized_delta_count": hu_turn1_realized_delta_count,
        "hu_turn1_realized_override_count": hu_turn1_realized_override_count,
        "hu_turn1_non_fired_final_mismatch_count": hu_turn1_non_fired_final_mismatch_count,
        "shard_count": len(summaries),
        "shards": summaries,
    }


def merge_decisions(input_dirs: list[Path], output_path: Path, *, file_name: str) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out_handle:
        for input_dir in input_dirs:
            decision_path = input_dir / file_name
            if not decision_path.exists():
                raise FileNotFoundError(decision_path)
            with decision_path.open("rb") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line)
                        count += 1
    return count


def main() -> None:
    args = parse_args()
    input_dirs = discover_input_dirs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged_decisions = args.output_dir / "topk_decisions.jsonl"
    decision_count = merge_decisions(input_dirs, merged_decisions, file_name="topk_decisions.jsonl")
    summary = aggregate_summaries(input_dirs)
    summary["topk_decision_output"] = str(merged_decisions)
    summary["topk_decisions_merged"] = decision_count
    if any((input_dir / "hu_turn1_decisions.jsonl").exists() for input_dir in input_dirs):
        merged_t1_decisions = args.output_dir / "hu_turn1_decisions.jsonl"
        t1_decision_count = merge_decisions(
            input_dirs,
            merged_t1_decisions,
            file_name="hu_turn1_decisions.jsonl",
        )
        summary["hu_turn1_decision_output"] = str(merged_t1_decisions)
        summary["hu_turn1_decisions_merged"] = t1_decision_count
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shard_list_path = args.output_dir / "input_dirs.txt"
    shard_list_path.write_text(
        "\n".join(str(path) for path in input_dirs) + "\n",
        encoding="utf-8",
    )
    for name in ("manifest.json", "shards_manifest.jsonl"):
        source = input_dirs[0].parent.parent / name
        if source.exists():
            shutil.copy2(source, args.output_dir / name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
