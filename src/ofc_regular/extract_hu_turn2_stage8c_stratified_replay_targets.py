"""Extract stratified Stage8c replay targets from ranker comparison rows.

This tool is for building the next independent replay pool after the high
confidence rows have been mostly exhausted. It samples unknown rows across
ranker-score buckets instead of only taking the very top rows. The output is
replay input only; it does not approve runtime, P2, production, T1, or 50k
teacher generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .extract_hu_turn2_stage8c_fire_veto_replay_targets import has_observed_delta
from .extract_hu_turn2_stage8c_ranker_replay_targets import (
    load_distillation_index,
    row_ranker_role,
)
from .extract_hu_turn2_stage8c_topk_replay_targets import (
    replay_missing_fields,
    row_key,
    safe_float,
    safe_int,
)


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_stratified_replay_targets")
DEFAULT_TARGET_NAME = "stratified_replay_targets.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--ranker-top-rows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    parser.add_argument("--ranker", action="append", default=[], help="Ranker to include. Repeatable. Defaults to all runtime-eligible rankers.")
    parser.add_argument("--split", action="append", default=["test"], help="Split to include. Repeatable. Defaults to test.")
    parser.add_argument("--seat", action="append", default=[], help="Seat to include. Repeatable. Defaults to all seats.")
    parser.add_argument("--buckets", type=int, default=5, help="Number of score-rank buckets per ranker/seat/split.")
    parser.add_argument("--targets-per-bucket", type=int, default=10)
    parser.add_argument("--max-targets-total", type=int, default=0)
    parser.add_argument("--max-rank", type=int, default=0, help="Optional maximum rank to consider before bucketing. 0 means all rows in the top-rows artifact.")
    parser.add_argument("--include-observed", action="store_true", help="Include rows that already have observed/replay delta.")
    parser.add_argument(
        "--allow-non-runtime-rankers",
        action="store_true",
        help="Allow confirm/replay-derived rankers. Default keeps runtime-eligible rankers only.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_rows(
    rows: Iterable[dict[str, str]],
    *,
    rankers: set[str] | None,
    splits: set[str] | None,
    seats: set[str] | None,
    max_rank: int,
    allow_non_runtime_rankers: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    selected: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    selected_by_ranker: Counter[str] = Counter()
    skipped_non_runtime_by_ranker: Counter[str] = Counter()
    for row in rows:
        counters["input_rows"] += 1
        if rankers is not None and row.get("ranker", "") not in rankers:
            counters["skipped_ranker_filter"] += 1
            continue
        if splits is not None and row.get("split", "") not in splits:
            counters["skipped_split_filter"] += 1
            continue
        if seats is not None and row.get("seat", "") not in seats:
            counters["skipped_seat_filter"] += 1
            continue
        if max_rank > 0 and safe_int(row.get("rank"), 10**9) > max_rank:
            counters["skipped_rank_filter"] += 1
            continue
        role = row_ranker_role(row)
        if not allow_non_runtime_rankers and not role["runtime_eligible"]:
            counters["skipped_non_runtime_ranker"] += 1
            skipped_non_runtime_by_ranker[row.get("ranker", "")] += 1
            continue
        out = dict(row)
        out.update(role)
        selected.append(out)
        counters["selected_rows"] += 1
        selected_by_ranker[out.get("ranker", "")] += 1
    return selected, {
        "selection_counts": dict(counters),
        "selected_by_ranker": dict(selected_by_ranker),
        "skipped_non_runtime_by_ranker": dict(skipped_non_runtime_by_ranker),
    }


def _bucket_id(index: int, total: int, buckets: int) -> int:
    if total <= 0 or buckets <= 1:
        return 0
    return min(buckets - 1, int(index * buckets / total))


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("ranker", ""), row.get("split", ""), row.get("seat", ""))


def _sort_key(row: dict[str, str]) -> tuple[float, int, str]:
    return (-safe_float(row.get("ranker_score")), safe_int(row.get("rank"), 10**9), row.get("join_key", ""))


def _build_target(
    *,
    source_row: dict[str, Any],
    ranker_row: dict[str, str],
    selector_rank: int,
    bucket_id: int,
    bucket_size: int,
) -> dict[str, Any]:
    missing = replay_missing_fields(source_row)
    target = dict(source_row)
    role = row_ranker_role(ranker_row)
    target.update(
        {
            "schema": "hu_turn2_stage8c_stratified_replay_target_v1",
            "replay_target_reason": "stratified_prediction_ranker_unknown_delta",
            "selector": "stage8c_stratified_ranker_score_bucket",
            "selector_rank": selector_rank,
            "ranker": ranker_row.get("ranker", ""),
            "ranker_role": ranker_row.get("ranker_role", role["ranker_role"]),
            "runtime_eligible_ranker": role["runtime_eligible"],
            "ranker_role_reason": ranker_row.get("role_reason", role["role_reason"]),
            "ranker_rank": safe_int(ranker_row.get("rank"), -1),
            "ranker_score": safe_float(ranker_row.get("ranker_score")),
            "ranker_score_bucket": bucket_id,
            "ranker_score_bucket_size": bucket_size,
            "ranker_split": ranker_row.get("split", ""),
            "fire_probability": safe_float(ranker_row.get("risk_probability")),
            "policy_delta_prediction": safe_float(ranker_row.get("policy_delta_prediction")),
            "observed_delta_prediction": ranker_row.get("observed_delta_prediction", ""),
            "predicted_delta": safe_float(ranker_row.get("predicted_delta")),
            "confirm_delta": safe_float(ranker_row.get("confirm_delta")),
            "confirm_delta_se": safe_float(ranker_row.get("confirm_delta_se")),
            "candidate_ev_rank": safe_float(ranker_row.get("candidate_ev_rank"), 9999.0),
            "realized_delta_observed": False,
            "observed_performance_claim": "No",
            "replay_ready": not missing,
            "replay_blocker": ",".join(missing),
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
            "note": (
                "Selected by stratified ranker-score bucket for independent replay. "
                "This is pool exploration only; do not treat ranker_score, confirm_delta, "
                "or model predictions as performance evidence."
            ),
        }
    )
    return target


def extract_stratified_targets(
    distillation_index: dict[tuple[str, ...], dict[str, Any]],
    ranker_rows: list[dict[str, str]],
    *,
    buckets: int,
    targets_per_bucket: int,
    include_observed: bool,
    max_targets_total: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ranker_rows:
        grouped[_group_key(row)].append(row)

    targets: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    target_counts_by_bucket: Counter[str] = Counter()
    seen_keys: set[tuple[str, ...]] = set()

    for group_key in sorted(grouped):
        group_rows = sorted(grouped[group_key], key=_sort_key)
        bucketed: dict[int, list[dict[str, str]]] = defaultdict(list)
        for index, row in enumerate(group_rows):
            bucketed[_bucket_id(index, len(group_rows), buckets)].append(row)
        for bucket_id in range(max(1, buckets)):
            bucket_rows = bucketed.get(bucket_id, [])
            selected_in_bucket = 0
            for row in bucket_rows:
                counters["candidate_rows"] += 1
                key = row_key(row)
                source_row = distillation_index.get(key)
                if source_row is None:
                    counters["missing_distillation_row"] += 1
                    audit_rows.append({"status": "missing_distillation_row", "ranker": row.get("ranker", ""), "hand_seed": row.get("hand_seed", "")})
                    continue
                if not include_observed and (has_observed_delta(row) or has_observed_delta(source_row)):
                    counters["skipped_observed_delta"] += 1
                    continue
                if key in seen_keys:
                    counters["skipped_duplicate_target"] += 1
                    continue
                target = _build_target(
                    source_row=source_row,
                    ranker_row=row,
                    selector_rank=len(targets) + 1,
                    bucket_id=bucket_id,
                    bucket_size=len(bucket_rows),
                )
                seen_keys.add(key)
                targets.append(target)
                selected_in_bucket += 1
                counters["targets"] += 1
                counters["replay_ready" if target["replay_ready"] else "replay_blocked"] += 1
                bucket_key = "|".join([group_key[0], group_key[1], group_key[2], str(bucket_id)])
                target_counts_by_bucket[bucket_key] += 1
                audit_rows.append(
                    {
                        "status": "target",
                        "selector_rank": target["selector_rank"],
                        "ranker": target["ranker"],
                        "ranker_split": target["ranker_split"],
                        "seat": target.get("seat", ""),
                        "ranker_score_bucket": bucket_id,
                        "ranker_score_bucket_size": len(bucket_rows),
                        "ranker_rank": target["ranker_rank"],
                        "ranker_score": target["ranker_score"],
                        "fire_probability": target["fire_probability"],
                        "policy_delta_prediction": target["policy_delta_prediction"],
                        "replay_ready": target["replay_ready"],
                        "replay_blocker": target["replay_blocker"],
                        "hand_seed": target.get("hand_seed", ""),
                    }
                )
                if targets_per_bucket > 0 and selected_in_bucket >= targets_per_bucket:
                    counters["stopped_bucket_cap"] += 1
                    break
                if max_targets_total > 0 and len(targets) >= max_targets_total:
                    counters["stopped_total_cap"] += 1
                    return targets, audit_rows, {
                        "counts": dict(counters),
                        "target_counts_by_bucket": dict(target_counts_by_bucket),
                    }
    return targets, audit_rows, {
        "counts": dict(counters),
        "target_counts_by_bucket": dict(target_counts_by_bucket),
    }


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# HU T2 Stage8c Stratified Replay Targets",
        "",
        "These rows are independent replay targets only. They do not approve runtime, production, P2, T1, or 50k teacher generation.",
        "",
        f"- targets: `{manifest.get('counts', {}).get('targets', 0)}`",
        f"- replay ready: `{manifest.get('counts', {}).get('replay_ready', 0)}`",
        f"- skipped observed delta: `{manifest.get('counts', {}).get('skipped_observed_delta', 0)}`",
        f"- buckets: `{manifest.get('buckets', 0)}`",
        f"- targets per bucket: `{manifest.get('targets_per_bucket', 0)}`",
        f"- selected rankers: `{json.dumps(manifest.get('selected_rankers', 'all'), sort_keys=True)}`",
        f"- selected seats: `{json.dumps(manifest.get('selected_seats', 'all'), sort_keys=True)}`",
        f"- selected splits: `{json.dumps(manifest.get('selected_splits', 'all'), sort_keys=True)}`",
        "",
        "## Decision",
        "",
        "- allowed role: `fresh replay target selection`",
        "- runtime gate: `No-Go`",
        "- production / P2 fixed / T1 / 50k: `No-Go`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rankers = set(args.ranker) if args.ranker else None
    splits = set(args.split) if args.split else None
    seats = set(args.seat) if args.seat else None
    distillation_index, distillation_manifest = load_distillation_index(args.distillation_jsonl)
    selected_rows, selection_manifest = _selected_rows(
        read_csv(args.ranker_top_rows_csv),
        rankers=rankers,
        splits=splits,
        seats=seats,
        max_rank=args.max_rank,
        allow_non_runtime_rankers=args.allow_non_runtime_rankers,
    )
    targets, audit_rows, manifest = extract_stratified_targets(
        distillation_index,
        selected_rows,
        buckets=args.buckets,
        targets_per_bucket=args.targets_per_bucket,
        include_observed=args.include_observed,
        max_targets_total=args.max_targets_total,
    )
    manifest.update(
        {
            **distillation_manifest,
            **selection_manifest,
            "ranker_top_rows_csv": str(args.ranker_top_rows_csv),
            "selected_rankers": sorted(rankers) if rankers is not None else "runtime_eligible",
            "selected_splits": sorted(splits) if splits is not None else "all",
            "selected_seats": sorted(seats) if seats is not None else "all",
            "buckets": args.buckets,
            "targets_per_bucket": args.targets_per_bucket,
            "max_targets_total": args.max_targets_total,
            "include_observed": args.include_observed,
            "allow_non_runtime_rankers": args.allow_non_runtime_rankers,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / args.target_name, targets)
    write_csv(args.output_dir / "stratified_replay_target_audit.csv", audit_rows)
    (args.output_dir / "stratified_replay_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "stratified_replay_target_summary.md", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
