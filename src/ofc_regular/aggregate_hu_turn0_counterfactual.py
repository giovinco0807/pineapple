"""Aggregate sharded HU T0 whole-game counterfactual evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .evaluate_hu_turn0_counterfactual import summarize_events


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_shards(results_root: Path) -> list[tuple[Path, Path]]:
    shards: list[tuple[Path, Path]] = []
    for events_path in sorted(results_root.rglob("events.jsonl")):
        summary_path = events_path.with_name("summary.json")
        if summary_path.is_file():
            shards.append((events_path, summary_path))
    return shards


def aggregate_shards(
    shards: Sequence[tuple[Path, Path]],
    *,
    expected_paired_seeds_per_config: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not shards:
        raise ValueError("no completed counterfactual shards found")

    events_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata_by_config: dict[str, dict[str, Any]] = {}
    seen_event_ids: set[tuple[str, str]] = set()
    duplicate_event_count = 0
    shard_rows: list[dict[str, Any]] = []

    for events_path, summary_path in shards:
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        config_id = str(shard_summary.get("config_id") or "").strip()
        if not config_id:
            raise ValueError(f"missing config_id in {summary_path}")
        metadata = {
            "candidate_model": shard_summary.get("candidate_model"),
            "candidate_topk": shard_summary.get("candidate_topk"),
            "safe_selector_model": shard_summary.get("safe_selector_model"),
            "safe_selector_threshold_by_seat": shard_summary.get(
                "safe_selector_threshold_by_seat"
            ),
            "min_margin_by_seat": shard_summary.get("min_margin_by_seat"),
            "allowed_seats": shard_summary.get("allowed_seats"),
            "profile": shard_summary.get("profile"),
            "seed_stride": shard_summary.get("seed_stride"),
        }
        if config_id in metadata_by_config and metadata_by_config[config_id] != metadata:
            raise ValueError(f"runtime metadata changed within config {config_id}")
        metadata_by_config[config_id] = metadata

        shard_events = _read_jsonl(events_path)
        for event in shard_events:
            event_config = str(event.get("config_id") or config_id)
            if event_config != config_id:
                raise ValueError(f"event config mismatch in {events_path}")
            event_id = str(event.get("event_id") or "")
            if not event_id:
                raise ValueError(f"missing event_id in {events_path}")
            key = (config_id, event_id)
            if key in seen_event_ids:
                duplicate_event_count += 1
                continue
            seen_event_ids.add(key)
            event["config_id"] = config_id
            events_by_config[config_id].append(event)
        shard_rows.append(
            {
                "config_id": config_id,
                "events_path": str(events_path),
                "summary_path": str(summary_path),
                "paired_seeds": int(shard_summary.get("paired_seeds", 0)),
                "events": len(shard_events),
                "elapsed_seconds": float(shard_summary.get("elapsed_seconds", 0.0)),
            }
        )

    if duplicate_event_count:
        raise ValueError(f"duplicate counterfactual events found: {duplicate_event_count}")

    config_summaries: list[dict[str, Any]] = []
    seed_breakdown: list[dict[str, Any]] = []
    reference_seed_set: set[int] | None = None
    seed_set_mismatch_configs: list[str] = []
    for config_id in sorted(events_by_config):
        events = events_by_config[config_id]
        grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for event in events:
            seed = int(event["seed"])
            seat = str(event["seat"])
            if seat in grouped[seed]:
                raise ValueError(f"duplicate seat event for config={config_id} seed={seed} seat={seat}")
            grouped[seed][seat] = event
        incomplete = {
            seed: sorted(rows)
            for seed, rows in grouped.items()
            if set(rows) != {"first", "second"}
        }
        if incomplete:
            raise ValueError(f"incomplete seat pairs for config {config_id}: {incomplete}")
        seed_set = set(grouped)
        if reference_seed_set is None:
            reference_seed_set = seed_set
        elif seed_set != reference_seed_set:
            seed_set_mismatch_configs.append(config_id)
        if (
            expected_paired_seeds_per_config is not None
            and len(seed_set) != expected_paired_seeds_per_config
        ):
            raise ValueError(
                f"paired seed count mismatch for {config_id}: "
                f"{len(seed_set)} != {expected_paired_seeds_per_config}"
            )

        metrics = summarize_events(events)
        metadata = metadata_by_config[config_id]
        config_summaries.append(
            {
                "config_id": config_id,
                **metrics,
                **metadata,
            }
        )
        for seed, rows in sorted(grouped.items()):
            first = rows["first"]
            second = rows["second"]
            seed_breakdown.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "first_delta": float(first["realized_delta"]),
                    "second_delta": float(second["realized_delta"]),
                    "paired_delta": (
                        float(first["realized_delta"]) + float(second["realized_delta"])
                    )
                    / 2.0,
                    "first_fired": bool(first["override_fired"]),
                    "second_fired": bool(second["override_fired"]),
                }
            )

    ranked = sorted(
        config_summaries,
        key=lambda row: (
            float(row["avg_delta_per_hand"]),
            float(row["ci95_low"]),
            float(row["avg_delta_per_fire"]),
        ),
        reverse=True,
    )
    summary = {
        "schema": "hu_turn0_counterfactual_aggregate_v1",
        "completed_shards": len(shards),
        "configs": len(config_summaries),
        "events": sum(len(rows) for rows in events_by_config.values()),
        "duplicate_event_count": duplicate_event_count,
        "seed_set_mismatch_configs": seed_set_mismatch_configs,
        "expected_paired_seeds_per_config": expected_paired_seeds_per_config,
        "config_results": ranked,
        "best_observed_config": ranked[0]["config_id"] if ranked else None,
        "decision": "evidence_ready_for_completion_audit",
        "primary_metric": "realized_same_seed_whole_game_delta_per_hand",
    }
    merged_events = [
        event
        for config_id in sorted(events_by_config)
        for event in sorted(
            events_by_config[config_id], key=lambda row: (int(row["seed"]), str(row["seat"]))
        )
    ]
    return summary, merged_events, seed_breakdown


def _flat_config_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary["config_results"]:
        rows.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"by_seat", "no_override_reason_counts", "min_margin_by_seat"}
            }
        )
    return rows


def _position_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary["config_results"]:
        for seat in ("first", "second"):
            rows.append({"config_id": item["config_id"], "seat": seat, **item["by_seat"][seat]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-paired-seeds-per-config", type=int)
    args = parser.parse_args()

    summary, events, seed_breakdown = aggregate_shards(
        discover_shards(args.results_root),
        expected_paired_seeds_per_config=args.expected_paired_seeds_per_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "events_merged.jsonl", events)
    _write_csv(args.output_dir / "config_results.csv", _flat_config_rows(summary))
    _write_csv(args.output_dir / "seed_breakdown.csv", seed_breakdown)
    _write_csv(args.output_dir / "position_breakdown.csv", _position_rows(summary))
    failures = sorted(
        (event for event in events if bool(event["override_fired"])),
        key=lambda row: float(row["realized_delta"]),
    )[:30]
    _write_jsonl(args.output_dir / "failure_top30.jsonl", failures)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
