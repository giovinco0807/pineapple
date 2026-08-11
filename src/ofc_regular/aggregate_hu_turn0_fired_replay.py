"""Aggregate sharded HU T0 fired-pair replay labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def aggregate_replay(
    *,
    results_root: Path,
    shard_manifest: Path,
    output_dir: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    specs = _read_jsonl(shard_manifest)
    completed: list[tuple[dict[str, Any], Path]] = []
    missing: list[int] = []
    for spec in specs:
        result_dir = results_root / str(spec["output_prefix"])
        if not (result_dir / "DONE").exists():
            missing.append(int(spec["shard"]))
        else:
            completed.append((spec, result_dir))
    if missing and not allow_partial:
        raise RuntimeError(f"missing T0 fired replay shards: {missing}")
    if not completed:
        raise RuntimeError("no completed T0 fired replay shards")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    duplicate_target_ids: list[str] = []
    for spec, result_dir in completed:
        teacher_path = result_dir / "teacher.jsonl"
        summary_path = result_dir / "summary.json"
        if not teacher_path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"incomplete result artifact in {result_dir}")
        shard_rows = _read_jsonl(teacher_path)
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if len(shard_rows) != int(spec["samples"]):
            raise RuntimeError(f"row count mismatch in {result_dir}")
        if int(shard_summary.get("written", -1)) != len(shard_rows):
            raise RuntimeError(f"summary row count mismatch in {result_dir}")
        if int(shard_summary.get("missing", -1)) != 0:
            raise RuntimeError(f"replay summary reports missing rows in {result_dir}")
        for local_index, row in enumerate(shard_rows):
            target_id = str(row.get("target_id") or "")
            if not target_id:
                raise RuntimeError(f"missing target_id in {teacher_path}")
            if target_id in seen_target_ids:
                duplicate_target_ids.append(target_id)
                continue
            seen_target_ids.add(target_id)
            if not bool(row.get("common_random_futures_verified")):
                raise RuntimeError(f"common futures failed for {target_id}")
            if not bool(row.get("action_mapping_verified")):
                raise RuntimeError(f"action mapping failed for {target_id}")
            future_samples = int(row.get("future_samples", -1))
            replay_actions = list(row.get("replay_actions", ()))
            if len(replay_actions) != 2 or any(
                int(action.get("rollout_count", -1)) != future_samples
                for action in replay_actions
            ):
                raise RuntimeError(f"rollout count mismatch for {target_id}")
            for field in (
                "candidate_delta_vs_baseline",
                "candidate_delta_se_vs_baseline",
                "candidate_delta_lcb196",
            ):
                if not math.isfinite(float(row.get(field, float("nan")))):
                    raise RuntimeError(f"non-finite {field} for {target_id}")
            output = dict(row)
            output["aggregate_shard"] = int(spec["shard"])
            output["aggregate_local_row_index"] = local_index
            rows.append(output)
        summaries.append(shard_summary)
    if duplicate_target_ids:
        raise RuntimeError(f"duplicate replay target ids: {len(duplicate_target_ids)}")

    expected_records = sum(int(spec["samples"]) for spec in specs)
    if not allow_partial and len(rows) != expected_records:
        raise RuntimeError(f"replay record count {len(rows)} != {expected_records}")
    rows.sort(key=lambda row: str(row["target_id"]))
    deltas = [float(row["candidate_delta_vs_baseline"]) for row in rows]
    delta_ses = [float(row["candidate_delta_se_vs_baseline"]) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "hu_turn0_fired_pair_replay.jsonl"
    summaries_path = output_dir / "hu_turn0_fired_pair_replay_summaries.jsonl"
    _write_jsonl(merged_path, rows)
    _write_jsonl(summaries_path, summaries)
    result = {
        "schema": "hu_turn0_fired_pair_replay_aggregate_v1",
        "records": len(rows),
        "expected_records": expected_records,
        "expected_shards": len(specs),
        "completed_shards": len(completed),
        "missing_shards": missing,
        "allow_partial": bool(allow_partial),
        "duplicate_target_count": 0,
        "common_random_futures_verified": True,
        "action_mapping_verified": True,
        "future_sample_counts": dict(
            sorted(Counter(str(row["future_samples"]) for row in rows).items())
        ),
        "label_counts": dict(
            sorted(Counter(str(row["safe_override_label"]) for row in rows).items())
        ),
        "seat_counts": dict(sorted(Counter(str(row["seat"]) for row in rows).items())),
        "source_config_counts": dict(
            sorted(Counter(str(row["source_config_id"]) for row in rows).items())
        ),
        "delta_mean": mean(deltas),
        "delta_median": median(deltas),
        "delta_se_mean": mean(delta_ses),
        "lcb196_positive_rate": sum(
            float(row["candidate_delta_lcb196"]) > 0.0 for row in rows
        )
        / len(rows),
        "elapsed_seconds_sum": sum(
            float(summary.get("elapsed_seconds", 0.0)) for summary in summaries
        ),
        "merged_output": str(merged_path),
        "shard_summaries": str(summaries_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = aggregate_replay(
        results_root=args.results_root,
        shard_manifest=args.shard_manifest,
        output_dir=args.output_dir,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
