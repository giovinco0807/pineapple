"""Strictly merge disjoint M4 population shards before final acceptance.

Per-shard confidence intervals are never averaged.  Every worker record is
content-checked against its shard summary, the frozen seed grid is reconstructed
exactly once, and the final metrics are recomputed over merged hand-seed
clusters.  This keeps a sharded Spot run statistically identical to one
monolithic invocation of :mod:`evaluate_hu_m4_population`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_hu_m4_population import (
    M4_OPPONENT_PROFILES,
    M4_POPULATION_EVALUATION_SCHEMA,
    summarize_hu_m4_population_records,
)


M43_POPULATION_PLAN_SCHEMA = "hu_m43_population_acceptance_plan_v1"
M4_POPULATION_MERGE_SCHEMA = "hu_m4_population_shard_merge_v1"
REQUIRED_ACTIVATION_GUARDS = {
    "current_profile_changed",
    "runtime_policy_activated",
    "full_replacement_enabled",
    "threshold_changed_after_lock",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summary_for_comparison(summary: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    normalized.pop("runtime_config", None)
    normalized.pop("shard_merge", None)
    normalized.pop("records", None)
    normalized["elapsed_seconds"] = None
    normalized["records_output"] = None
    return normalized


def load_population_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    plan = _read_json(source)
    if plan.get("schema") != M43_POPULATION_PLAN_SCHEMA:
        raise ValueError("M4.3 population plan schema mismatch")
    if plan.get("status") != "frozen_before_population_evaluation":
        raise ValueError("M4.3 population plan is not frozen")
    opponents = plan.get("opponents")
    if opponents != list(M4_OPPONENT_PROFILES):
        raise ValueError("M4.3 population plan opponent set/order changed")
    paired_seeds = plan.get("paired_seeds_per_opponent")
    seed = plan.get("seed")
    stride = plan.get("seed_stride")
    shards = plan.get("shards")
    per_shard = plan.get("paired_seeds_per_shard")
    integers = (paired_seeds, seed, stride, shards, per_shard)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise ValueError("M4.3 population integer schedule is invalid")
    if min(int(value) for value in integers) <= 0:
        raise ValueError("M4.3 population schedule must be positive")
    if int(shards) * int(per_shard) != int(paired_seeds):
        raise ValueError("M4.3 population shard budget does not cover the seed grid")
    if int(plan.get("minimum_valid_overrides", -1)) < 300:
        raise ValueError("M4.3 population plan lowers the 300-override gate")
    guards = plan.get("activation_guards")
    if (
        not isinstance(guards, Mapping)
        or set(guards) != REQUIRED_ACTIVATION_GUARDS
        or any(guards[key] is not False for key in REQUIRED_ACTIVATION_GUARDS)
    ):
        raise ValueError("M4.3 population plan violates activation guards")
    planned_seeds = {
        int(seed) + index * int(stride) for index in range(int(paired_seeds))
    }
    freshness = plan.get("freshness")
    schedules = (
        freshness.get("excluded_population_schedules")
        if isinstance(freshness, Mapping)
        else None
    )
    if (
        not isinstance(schedules, Sequence)
        or isinstance(schedules, (str, bytes))
        or not schedules
    ):
        raise ValueError("M4.3 population plan lacks prior population exclusions")
    for index, schedule in enumerate(schedules):
        if not isinstance(schedule, Mapping):
            raise ValueError(f"M4.3 excluded schedule {index} is invalid")
        prior_seed = schedule.get("seed")
        prior_stride = schedule.get("seed_stride")
        prior_count = schedule.get("paired_seeds")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (prior_seed, prior_stride, prior_count)
        ):
            raise ValueError(f"M4.3 excluded schedule {index} is invalid")
        prior_seeds = {
            int(prior_seed) + offset * int(prior_stride)
            for offset in range(int(prior_count))
        }
        if planned_seeds & prior_seeds:
            raise ValueError("M4.3 population seed schedule overlaps a prior run")
    return plan


def merge_population_shards(
    *,
    plan: Mapping[str, Any],
    shard_evaluations: Sequence[tuple[Path, Mapping[str, Any]]],
    shard_records: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    plan_sha256: str,
    records_output: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if len(shard_evaluations) != len(shard_records):
        raise ValueError("each population shard requires one summary and record file")
    expected_shards = int(plan["shards"])
    if len(shard_evaluations) != expected_shards:
        raise ValueError(
            f"population shard count mismatch: {len(shard_evaluations)} != {expected_shards}"
        )
    opponents = tuple(str(value) for value in plan["opponents"])
    seed = int(plan["seed"])
    stride = int(plan["seed_stride"])
    total_seeds = int(plan["paired_seeds_per_opponent"])
    per_shard = int(plan["paired_seeds_per_shard"])
    runtime_config: dict[str, Any] | None = None
    shards_by_start: dict[int, tuple[Path, Path, Mapping[str, Any], Sequence[Mapping[str, Any]]]] = {}
    provenance: list[dict[str, Any]] = []

    for (evaluation_path, evaluation), (records_path, records) in zip(
        shard_evaluations, shard_records, strict=True
    ):
        if evaluation.get("schema") != M4_POPULATION_EVALUATION_SCHEMA:
            raise ValueError(f"population shard schema mismatch: {evaluation_path}")
        if evaluation.get("opponents") != list(opponents):
            raise ValueError(f"population shard opponent order mismatch: {evaluation_path}")
        if evaluation.get("paired_seat_swap") is not True:
            raise ValueError(f"population shard is not seat-swapped: {evaluation_path}")
        shard_seed = evaluation.get("seed")
        shard_stride = evaluation.get("seed_stride")
        shard_count = evaluation.get("paired_seeds_per_opponent")
        if (
            isinstance(shard_seed, bool)
            or not isinstance(shard_seed, int)
            or shard_stride != stride
            or shard_count != per_shard
        ):
            raise ValueError(f"population shard schedule mismatch: {evaluation_path}")
        offset, remainder = divmod(shard_seed - seed, stride)
        if remainder or offset < 0 or offset % per_shard:
            raise ValueError(f"population shard start is outside frozen grid: {shard_seed}")
        if offset in shards_by_start:
            raise ValueError(f"duplicate population shard offset: {offset}")
        shard_runtime = evaluation.get("runtime_config")
        if not isinstance(shard_runtime, Mapping):
            raise ValueError(f"population shard runtime config missing: {evaluation_path}")
        if (
            shard_runtime.get("current_profile_used") is not False
            or shard_runtime.get("promotion_artifact_contract") is not True
            or shard_runtime.get("diagnostic_legacy") is not False
        ):
            raise ValueError(f"population shard runtime contract is unsafe: {evaluation_path}")
        if runtime_config is None:
            runtime_config = dict(shard_runtime)
        elif _canonical(runtime_config) != _canonical(shard_runtime):
            raise ValueError("population shards used different runtime artifacts or thresholds")

        recomputed = summarize_hu_m4_population_records(
            records,
            opponents=opponents,
            paired_seeds=per_shard,
            seed=shard_seed,
            seed_stride=stride,
            elapsed_seconds=None,
            baseline_profile=str(
                shard_runtime.get("baseline_profile", "stage18_p1")
            ),
        )
        if _canonical(_summary_for_comparison(evaluation)) != _canonical(
            _summary_for_comparison(recomputed)
        ):
            raise ValueError(f"population shard summary/content mismatch: {evaluation_path}")
        shards_by_start[offset] = (
            evaluation_path,
            records_path,
            evaluation,
            records,
        )

    expected_offsets = set(range(0, total_seeds, per_shard))
    if set(shards_by_start) != expected_offsets:
        missing = sorted(expected_offsets - set(shards_by_start))
        extra = sorted(set(shards_by_start) - expected_offsets)
        raise ValueError(f"population shard grid mismatch: missing={missing}, extra={extra}")
    assert runtime_config is not None
    opponent_order = {name: index for index, name in enumerate(opponents)}
    seat_order = {"first": 0, "second": 1}
    merged_records = [
        dict(row)
        for offset in sorted(shards_by_start)
        for row in shards_by_start[offset][3]
    ]
    merged_records.sort(
        key=lambda row: (
            opponent_order[str(row["opponent"])],
            int(row["seed"]),
            seat_order[str(row["seat"])],
        )
    )
    final = summarize_hu_m4_population_records(
        merged_records,
        opponents=opponents,
        paired_seeds=total_seeds,
        seed=seed,
        seed_stride=stride,
        records_output=records_output,
        elapsed_seconds=sum(
            float(shards_by_start[offset][2].get("elapsed_seconds") or 0.0)
            for offset in sorted(shards_by_start)
        ),
        baseline_profile=str(
            runtime_config.get("baseline_profile", "stage18_p1")
        ),
    )
    final["runtime_config"] = {
        **runtime_config,
        "population_plan_sha256": plan_sha256,
        "sharded_evaluation": True,
        "shard_count": expected_shards,
        "final_metrics_recomputed_from_merged_records": True,
    }
    for offset in sorted(shards_by_start):
        evaluation_path, records_path, evaluation, records = shards_by_start[offset]
        provenance.append(
            {
                "offset": offset,
                "seed": int(evaluation["seed"]),
                "paired_seeds": len(records) // (2 * len(opponents)),
                "evaluation_path": str(evaluation_path.resolve()),
                "evaluation_sha256": _sha256(evaluation_path),
                "records_path": str(records_path.resolve()),
                "records_sha256": _sha256(records_path),
                "records": len(records),
            }
        )
    merge_manifest = {
        "schema": M4_POPULATION_MERGE_SCHEMA,
        "status": "complete_content_verified",
        "population_plan_sha256": plan_sha256,
        "seed": seed,
        "seed_stride": stride,
        "paired_seeds_per_opponent": total_seeds,
        "opponents": list(opponents),
        "shards": provenance,
        "merged_records": len(merged_records),
        "current_profile_used": False,
        "metrics_recomputed_from_merged_seed_clusters": True,
    }
    final["shard_merge"] = {
        "schema": M4_POPULATION_MERGE_SCHEMA,
        "population_plan_sha256": plan_sha256,
        "shards": expected_shards,
        "metrics_recomputed_from_merged_seed_clusters": True,
    }
    return final, merge_manifest, merged_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-evaluation", type=Path, action="append", required=True)
    parser.add_argument("--shard-records", type=Path, action="append", required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument(
        "--records-output-label",
        default="records.jsonl",
        help="Portable path label stored in evaluation.json (default: records.jsonl).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_paths = [
        args.plan,
        *args.shard_evaluation,
        *args.shard_records,
    ]
    resolved_inputs = [path.resolve() for path in input_paths]
    outputs = {
        args.records_output.resolve(),
        args.output.resolve(),
        args.merge_manifest_output.resolve(),
    }
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("population merge inputs must be distinct")
    if len(outputs) != 3 or outputs & set(resolved_inputs):
        raise ValueError("population merge outputs must be distinct from all inputs")
    plan = load_population_plan(args.plan)
    plan_sha = _sha256(args.plan)
    shard_evaluations = [
        (path, _read_json(path)) for path in args.shard_evaluation
    ]
    shard_records = [(path, _read_jsonl(path)) for path in args.shard_records]
    final, merge_manifest, merged_records = merge_population_shards(
        plan=plan,
        shard_evaluations=shard_evaluations,
        shard_records=shard_records,
        plan_sha256=plan_sha,
        records_output=args.records_output_label,
    )
    records_text = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in merged_records
    )
    _atomic_text(args.records_output, records_text)
    _atomic_text(
        args.output,
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    merge_manifest["merged_records_sha256"] = _sha256(args.records_output)
    merge_manifest["evaluation_sha256"] = _sha256(args.output)
    _atomic_text(
        args.merge_manifest_output,
        json.dumps(merge_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    print(json.dumps(merge_manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M43_POPULATION_PLAN_SCHEMA",
    "M4_POPULATION_MERGE_SCHEMA",
    "REQUIRED_ACTIVATION_GUARDS",
    "load_population_plan",
    "merge_population_shards",
]
