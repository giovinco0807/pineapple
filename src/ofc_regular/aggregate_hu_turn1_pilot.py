"""Aggregate HU Turn1 pilot shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_shard_specs(path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                specs.append(json.loads(line))
    return specs


def _sum_duplicate_profile(
    summaries: list[dict[str, Any]], profile_name: str
) -> dict[str, Any]:
    raw = 0
    unique = 0
    repeated = 0
    max_occurrence = 0
    present = 0
    for summary in summaries:
        duplicate_profile = summary.get("duplicate_profile_stats") or {}
        stats = duplicate_profile.get(profile_name) or {}
        if not stats:
            continue
        present += 1
        raw += int(stats.get("raw", 0) or 0)
        unique += int(stats.get("unique", 0) or 0)
        repeated += int(stats.get("repeated", 0) or 0)
        max_occurrence = max(max_occurrence, int(stats.get("max_occurrence", 0) or 0))
    return {
        "summary_count": present,
        "raw": raw,
        "unique_sum": unique,
        "repeated": repeated,
        "duplicate_rate": (repeated / raw) if raw else 0.0,
        "max_occurrence": max_occurrence,
    }


def _globalize_record_ids(
    record: dict[str, Any],
    *,
    shard: int,
    local_row_index: int,
) -> dict[str, Any]:
    """Assign IDs that remain unique after shard aggregation."""

    output = dict(record)
    output["local_sample_id"] = record.get("sample_id", local_row_index)
    output["sample_id"] = shard * 1_000_000 + local_row_index
    output["aggregate_shard"] = shard
    output["aggregate_local_row_index"] = local_row_index
    return output


def aggregate_hu_turn1_pilot(
    *,
    results_root: Path,
    shard_manifest: Path,
    output_dir: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    specs = read_shard_specs(shard_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "hu_turn1_stage1_pilot.jsonl"
    summaries_path = output_dir / "hu_turn1_stage1_pilot_summaries.jsonl"

    missing: list[int] = []
    completed_dirs: list[Path] = []
    for spec in specs:
        result_dir = results_root / str(spec["output_prefix"])
        if not (result_dir / "DONE").exists():
            missing.append(int(spec["shard"]))
        else:
            completed_dirs.append(result_dir)
    if missing and not allow_partial:
        raise RuntimeError(f"Missing HU T1 pilot result shards: {missing}")
    if not completed_dirs:
        raise RuntimeError("No completed HU T1 pilot result shards found")

    records = 0
    summaries: list[dict[str, Any]] = []
    with merged_path.open("w", encoding="utf-8") as merged, summaries_path.open(
        "w", encoding="utf-8"
    ) as summaries_out:
        for result_dir in completed_dirs:
            teacher_path = result_dir / "teacher.jsonl"
            summary_path = result_dir / "summary.json"
            if not teacher_path.exists():
                raise RuntimeError(f"Missing teacher output: {teacher_path}")
            if not summary_path.exists():
                raise RuntimeError(f"Missing summary output: {summary_path}")
            summary = load_json(summary_path)
            summaries.append(summary)
            summaries_out.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
            spec = next(
                item for item in specs if results_root / str(item["output_prefix"]) == result_dir
            )
            shard = int(spec["shard"])
            with teacher_path.open("r", encoding="utf-8") as teacher:
                for local_row_index, line in enumerate(teacher):
                    if line.strip():
                        record = _globalize_record_ids(
                            json.loads(line),
                            shard=shard,
                            local_row_index=local_row_index,
                        )
                        merged.write(
                            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                        records += 1

    elapsed_values = [float(item.get("elapsed_seconds", 0.0) or 0.0) for item in summaries]
    seconds_per_sample = [
        float(item.get("seconds_per_sample", 0.0) or 0.0) for item in summaries
    ]
    action_counts = [float(item.get("mean_action_count", 0.0) or 0.0) for item in summaries]
    topk_decisions = [int(item.get("topk_decisions", 0) or 0) for item in summaries]
    topk_overrides = [int(item.get("topk_overrides", 0) or 0) for item in summaries]
    t2_seconds = [
        float((item.get("profile_stats") or {}).get("choose_action_T2_seconds", 0.0) or 0.0)
        for item in summaries
    ]
    rollout_seconds = [
        float((item.get("profile_stats") or {}).get("rollout_seconds", 0.0) or 0.0)
        for item in summaries
    ]

    aggregate = {
        "schema": "hu_turn1_stage1_pilot_aggregate_v1",
        "records": records,
        "expected_shards": len(specs),
        "completed_shards": len(completed_dirs),
        "missing_shards": missing,
        "allow_partial": allow_partial,
        "elapsed_seconds_sum": sum(elapsed_values),
        "mean_seconds_per_sample": mean(seconds_per_sample) if seconds_per_sample else 0.0,
        "max_seconds_per_sample": max(seconds_per_sample) if seconds_per_sample else 0.0,
        "mean_action_count": mean(action_counts) if action_counts else 0.0,
        "topk_decisions": sum(topk_decisions),
        "topk_overrides": sum(topk_overrides),
        "t2_choose_action_seconds_sum": sum(t2_seconds),
        "rollout_seconds_sum": sum(rollout_seconds),
        "sample_id_globalized": True,
        "duplicate_profile_stats": {
            "t2_state_only": _sum_duplicate_profile(summaries, "t2_state_only"),
            "t2_state_plus_decision_seed": _sum_duplicate_profile(
                summaries, "t2_state_plus_decision_seed"
            ),
        },
        "merged_output": str(merged_path),
        "shard_summaries": str(summaries_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_hu_turn1_pilot(
        results_root=args.results_root,
        shard_manifest=args.shard_manifest,
        output_dir=args.output_dir,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
