"""Aggregate and validate HU T0 teacher pilot shards."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_specs(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def aggregate_hu_turn0_pilot(
    *,
    results_root: Path,
    shard_manifest: Path,
    output_dir: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    specs = _read_specs(shard_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "hu_turn0_stage1_pilot.jsonl"
    summaries_path = output_dir / "hu_turn0_stage1_pilot_summaries.jsonl"

    missing: list[int] = []
    completed: list[tuple[dict[str, Any], Path]] = []
    for spec in specs:
        result_dir = results_root / str(spec["output_prefix"])
        if not (result_dir / "DONE").exists():
            missing.append(int(spec["shard"]))
        else:
            completed.append((spec, result_dir))
    if missing and not allow_partial:
        raise RuntimeError(f"Missing HU T0 pilot result shards: {missing}")
    if not completed:
        raise RuntimeError("No completed HU T0 pilot result shards found")

    records = 0
    seat_counts: Counter[str] = Counter()
    baseline_regrets: list[float] = []
    score_gaps: list[float] = []
    legal_counts: list[int] = []
    evaluated_counts: list[int] = []
    non_finite_actions = 0
    wrong_rollout_counts = 0
    common_future_failures = 0
    replay_not_ready = 0
    summaries: list[dict[str, Any]] = []

    with merged_path.open("w", encoding="utf-8") as merged, summaries_path.open(
        "w", encoding="utf-8"
    ) as summaries_out:
        for spec, result_dir in completed:
            teacher_path = result_dir / "teacher.jsonl"
            summary_path = result_dir / "summary.json"
            if not teacher_path.exists() or not summary_path.exists():
                raise RuntimeError(f"Incomplete result directory: {result_dir}")
            shard_summary = _read_json(summary_path)
            summaries.append(shard_summary)
            summaries_out.write(
                json.dumps(shard_summary, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            expected_rollouts = int(shard_summary["future_samples"])
            shard = int(spec["shard"])
            with teacher_path.open("r", encoding="utf-8") as teacher:
                for local_index, line in enumerate(teacher):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["local_sample_id"] = row.get("sample_id", local_index)
                    row["sample_id"] = shard * 1_000_000 + local_index
                    row["aggregate_shard"] = shard
                    row["aggregate_local_row_index"] = local_index
                    merged.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    records += 1
                    seat_counts[str(row.get("seat", "unknown"))] += 1
                    baseline_regrets.append(float(row["delta_best_vs_baseline"]))
                    score_gaps.append(float(row["score_gap"]))
                    legal_counts.append(int(row["total_legal_actions"]))
                    evaluated_counts.append(int(row["evaluated_action_count"]))
                    common_future_failures += int(
                        not bool(row.get("common_random_futures_verified"))
                    )
                    replay_not_ready += int(not bool(row.get("replay_ready")))
                    for action in row.get("actions", ()):
                        if not all(
                            math.isfinite(float(action.get(key, float("nan"))))
                            for key in ("ev", "score", "se")
                        ):
                            non_finite_actions += 1
                        if int(action.get("rollout_count", -1)) != expected_rollouts:
                            wrong_rollout_counts += 1

    seconds_per_sample = [float(item["seconds_per_sample"]) for item in summaries]
    profile_keys = (
        "action_eval_seconds",
        "continuation_decision_seconds",
        "continuation_T1_seconds",
        "continuation_T2_seconds",
        "continuation_T3_seconds",
        "continuation_T4_seconds",
        "opponent_opening_seconds",
        "terminal_score_seconds",
        "raw_rollout_count",
    )
    profile_totals = {
        key: sum(float((item.get("profile_stats") or {}).get(key, 0.0)) for item in summaries)
        for key in profile_keys
    }
    aggregate = {
        "schema": "hu_turn0_stage1_pilot_aggregate_v1",
        "records": records,
        "expected_records": sum(int(spec["samples"]) for spec in specs),
        "expected_shards": len(specs),
        "completed_shards": len(completed),
        "missing_shards": missing,
        "allow_partial": allow_partial,
        "seat_counts": dict(sorted(seat_counts.items())),
        "mean_seconds_per_sample": mean(seconds_per_sample),
        "median_seconds_per_sample": median(seconds_per_sample),
        "p95_seconds_per_sample": _quantile(seconds_per_sample, 0.95),
        "max_seconds_per_sample": max(seconds_per_sample),
        "mean_legal_actions": mean(legal_counts),
        "min_legal_actions": min(legal_counts),
        "max_legal_actions": max(legal_counts),
        "mean_evaluated_actions": mean(evaluated_counts),
        "baseline_regret_mean": mean(baseline_regrets),
        "baseline_regret_median": median(baseline_regrets),
        "baseline_regret_p90": _quantile(baseline_regrets, 0.90),
        "baseline_regret_p95": _quantile(baseline_regrets, 0.95),
        "score_gap_mean": mean(score_gaps),
        "score_gap_median": median(score_gaps),
        "non_finite_actions": non_finite_actions,
        "wrong_rollout_counts": wrong_rollout_counts,
        "common_random_future_failures": common_future_failures,
        "replay_not_ready": replay_not_ready,
        "profile_totals": profile_totals,
        "sample_id_globalized": True,
        "merged_output": str(merged_path),
        "shard_summaries": str(summaries_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = aggregate_hu_turn0_pilot(
        results_root=args.results_root,
        shard_manifest=args.shard_manifest,
        output_dir=args.output_dir,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
