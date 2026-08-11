"""Extract replay targets from Stage8c prediction-ranker top rows.

This tool converts a ranker-comparison artifact into replay-ready JSONL rows by
joining top-ranked prediction rows back to the original TopK-confirm
distillation records. It is for selecting rows for additional independent
replay only; it does not make runtime, P2, T1, production, or 50k claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .compare_hu_turn2_stage8c_prediction_rankers import ranker_role
from .extract_hu_turn2_stage8c_fire_veto_replay_targets import has_observed_delta
from .extract_hu_turn2_stage8c_topk_replay_targets import (
    replay_missing_fields,
    row_key,
    safe_float,
)


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_ranker_replay_targets")
DEFAULT_TARGET_NAME = "ranker_replay_targets.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--ranker-top-rows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    parser.add_argument("--ranker", action="append", default=[], help="Ranker to include. Repeatable. Defaults to all.")
    parser.add_argument("--split", action="append", default=["test"], help="Split to include. Repeatable. Defaults to test.")
    parser.add_argument("--seat", action="append", default=[], help="Seat to include, e.g. first or second. Repeatable. Defaults to all.")
    parser.add_argument("--max-rank", type=int, default=50, help="Maximum rank per selected ranker.")
    parser.add_argument("--max-targets-per-ranker", type=int, default=20)
    parser.add_argument("--max-targets-total", type=int, default=0)
    parser.add_argument(
        "--allow-non-runtime-rankers",
        action="store_true",
        help=(
            "Allow replay/diagnostic rankers such as confirm_delta. Default keeps only rankers "
            "classified as deployable runtime inputs so replay target generation cannot "
            "accidentally optimize around non-runtime information."
        ),
    )
    parser.add_argument(
        "--include-observed",
        action="store_true",
        help="Include rows that already have observed/replay delta. Default skips them.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _boolish(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def row_ranker_role(row: dict[str, Any]) -> dict[str, Any]:
    ranker = str(row.get("ranker", ""))
    classified = ranker_role(ranker)
    runtime_eligible = _boolish(row.get("runtime_eligible"), default=None)
    # Ranker-name classification is the hard safety guard. A stale or malformed
    # CSV must not be able to promote confirm/replay-derived rankers.
    effective_runtime_eligible = bool(classified["runtime_eligible"]) and (True if runtime_eligible is None else runtime_eligible)
    return {
        "ranker_role": row.get("ranker_role") or classified["ranker_role"],
        "runtime_eligible": effective_runtime_eligible,
        "role_reason": row.get("role_reason") or classified["role_reason"],
    }


def load_distillation_index(paths: Iterable[Path]) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, Any]]:
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    duplicates = 0
    loaded = 0
    for path in paths:
        for row in read_jsonl(path):
            loaded += 1
            key = row_key(row)
            if key in index:
                duplicates += 1
            index[key] = row
    return index, {"loaded_distillation_rows": loaded, "duplicate_distillation_keys": duplicates}


def selected_top_rows(
    rows: Iterable[dict[str, str]],
    *,
    rankers: set[str] | None,
    splits: set[str] | None,
    max_rank: int,
    seats: set[str] | None = None,
    allow_non_runtime_rankers: bool = False,
) -> list[dict[str, str]]:
    selected, _audit = selected_top_rows_with_audit(
        rows,
        rankers=rankers,
        splits=splits,
        seats=seats,
        max_rank=max_rank,
        allow_non_runtime_rankers=allow_non_runtime_rankers,
    )
    return selected


def selected_top_rows_with_audit(
    rows: Iterable[dict[str, str]],
    *,
    rankers: set[str] | None,
    splits: set[str] | None,
    max_rank: int,
    seats: set[str] | None = None,
    allow_non_runtime_rankers: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    selected: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    skipped_non_runtime_by_ranker: Counter[str] = Counter()
    selected_by_ranker: Counter[str] = Counter()
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
        if safe_int(row.get("rank"), 10**9) > max_rank:
            counters["skipped_rank_filter"] += 1
            continue
        role = row_ranker_role(row)
        if not allow_non_runtime_rankers and not role["runtime_eligible"]:
            counters["skipped_non_runtime_ranker"] += 1
            skipped_non_runtime_by_ranker[str(row.get("ranker", ""))] += 1
            continue
        row = dict(row)
        row.update(role)
        selected.append(row)
        counters["selected_top_rows"] += 1
        selected_by_ranker[str(row.get("ranker", ""))] += 1
    selected.sort(key=lambda row: (row.get("ranker", ""), safe_int(row.get("rank"), 10**9)))
    return selected, {
        "selection_counts": dict(counters),
        "selected_top_rows_by_ranker": dict(selected_by_ranker),
        "skipped_non_runtime_rankers": dict(skipped_non_runtime_by_ranker),
    }


def build_target(
    *,
    source_row: dict[str, Any],
    top_row: dict[str, str],
) -> dict[str, Any]:
    missing = replay_missing_fields(source_row)
    target = dict(source_row)
    target.update(
        {
            "schema": "hu_turn2_stage8c_prediction_ranker_replay_target_v1",
            "replay_target_reason": "prediction_ranker_selected_unknown_delta",
            "ranker": top_row.get("ranker", ""),
            "ranker_role": top_row.get("ranker_role", row_ranker_role(top_row)["ranker_role"]),
            "runtime_eligible_ranker": row_ranker_role(top_row)["runtime_eligible"],
            "ranker_role_reason": top_row.get("role_reason", row_ranker_role(top_row)["role_reason"]),
            "ranker_rank": safe_int(top_row.get("rank"), -1),
            "ranker_score": safe_float(top_row.get("ranker_score")),
            "ranker_split": top_row.get("split", ""),
            "fire_probability": safe_float(top_row.get("risk_probability")),
            "policy_delta_prediction": safe_float(top_row.get("policy_delta_prediction")),
            "observed_delta_prediction": top_row.get("observed_delta_prediction", ""),
            "predicted_delta": safe_float(top_row.get("predicted_delta")),
            "confirm_delta": safe_float(top_row.get("confirm_delta")),
            "confirm_delta_se": safe_float(top_row.get("confirm_delta_se")),
            "candidate_ev_rank": safe_float(top_row.get("candidate_ev_rank"), 9999.0),
            "realized_delta_observed": False,
            "observed_performance_claim": "No",
            "replay_ready": not missing,
            "replay_blocker": ",".join(missing),
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
            "note": (
                "Selected by prediction-ranker comparison for independent replay. "
                "Do not use ranker_score, confirm_delta, or model predictions as performance evidence. "
                "Non-runtime rankers require explicit opt-in."
            ),
        }
    )
    return target


def extract_ranker_targets(
    distillation_rows: dict[tuple[str, ...], dict[str, Any]],
    top_rows: list[dict[str, str]],
    *,
    include_observed: bool,
    max_targets_per_ranker: int,
    max_targets_total: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, ...]] = set()
    emitted_by_ranker: Counter[str] = Counter()
    counters: Counter[str] = Counter()

    for top_row in top_rows:
        counters["selected_top_rows"] += 1
        ranker = top_row.get("ranker", "")
        if max_targets_per_ranker > 0 and emitted_by_ranker[ranker] >= max_targets_per_ranker:
            counters["skipped_ranker_cap"] += 1
            continue
        key = row_key(top_row)
        source_row = distillation_rows.get(key)
        if source_row is None:
            counters["missing_distillation_row"] += 1
            audit_rows.append(
                {
                    "ranker": ranker,
                    "rank": top_row.get("rank", ""),
                    "hand_seed": top_row.get("hand_seed", ""),
                    "status": "missing_distillation_row",
                }
            )
            continue
        if not include_observed and (has_observed_delta(top_row) or has_observed_delta(source_row)):
            counters["skipped_observed_delta"] += 1
            continue
        if key in seen_targets:
            counters["skipped_duplicate_target"] += 1
            continue
        target = build_target(source_row=source_row, top_row=top_row)
        seen_targets.add(key)
        emitted_by_ranker[ranker] += 1
        counters["targets"] += 1
        if target["replay_ready"]:
            counters["replay_ready"] += 1
        else:
            counters["replay_blocked"] += 1
        targets.append(target)
        audit_rows.append(
            {
                "ranker": ranker,
                "ranker_role": target["ranker_role"],
                "runtime_eligible_ranker": target["runtime_eligible_ranker"],
                "rank": top_row.get("rank", ""),
                "hand_seed": top_row.get("hand_seed", ""),
                "recommended_training_use": top_row.get("recommended_training_use", ""),
                "status": "target",
                "replay_ready": target["replay_ready"],
                "replay_blocker": target["replay_blocker"],
                "ranker_score": target["ranker_score"],
                "fire_probability": target["fire_probability"],
                "policy_delta_prediction": target["policy_delta_prediction"],
                "confirm_delta": target["confirm_delta"],
            }
        )
        if max_targets_total > 0 and len(targets) >= max_targets_total:
            counters["stopped_total_cap"] += 1
            break

    manifest = {
        "include_observed": include_observed,
        "max_targets_per_ranker": max_targets_per_ranker,
        "max_targets_total": max_targets_total,
        "counts": dict(counters),
        "targets_by_ranker": dict(emitted_by_ranker),
    }
    return targets, audit_rows, manifest


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# HU T2 Stage8c Ranker Replay Targets",
        "",
        "These rows are replay targets only. They are not runtime, P2, production, T1, or 50k evidence.",
        "",
        f"- targets: `{manifest.get('counts', {}).get('targets', 0)}`",
        f"- replay ready: `{manifest.get('counts', {}).get('replay_ready', 0)}`",
        f"- skipped observed delta: `{manifest.get('counts', {}).get('skipped_observed_delta', 0)}`",
        f"- missing distillation row: `{manifest.get('counts', {}).get('missing_distillation_row', 0)}`",
        f"- targets by ranker: `{json.dumps(manifest.get('targets_by_ranker', {}), sort_keys=True)}`",
        f"- allow non-runtime rankers: `{manifest.get('allow_non_runtime_rankers', False)}`",
        f"- selected seats: `{json.dumps(manifest.get('selected_seats', 'all'), sort_keys=True)}`",
        f"- selected top rows by ranker: `{json.dumps(manifest.get('selected_top_rows_by_ranker', {}), sort_keys=True)}`",
        f"- skipped non-runtime rankers: `{json.dumps(manifest.get('skipped_non_runtime_rankers', {}), sort_keys=True)}`",
        "",
        "## Decision",
        "",
        "- allowed role: `independent replay target selection`",
        "- runtime gate: `No-Go`",
        "- production / P2 fixed: `No-Go`",
        "- 50k teacher: `No-Go`",
        "- T1 training: `No-Go`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rankers = set(args.ranker) if args.ranker else None
    splits = set(args.split) if args.split else None
    seats = set(args.seat) if args.seat else None
    distillation_index, distillation_manifest = load_distillation_index(args.distillation_jsonl)
    top_rows, selection_manifest = selected_top_rows_with_audit(
        read_csv(args.ranker_top_rows_csv),
        rankers=rankers,
        splits=splits,
        seats=seats,
        max_rank=args.max_rank,
        allow_non_runtime_rankers=args.allow_non_runtime_rankers,
    )
    targets, audit_rows, manifest = extract_ranker_targets(
        distillation_index,
        top_rows,
        include_observed=args.include_observed,
        max_targets_per_ranker=args.max_targets_per_ranker,
        max_targets_total=args.max_targets_total,
    )
    manifest.update(
        {
            **distillation_manifest,
            "ranker_top_rows_csv": str(args.ranker_top_rows_csv),
            "selected_rankers": sorted(rankers) if rankers is not None else "all",
            "selected_splits": sorted(splits) if splits is not None else "all",
            "selected_seats": sorted(seats) if seats is not None else "all",
            "max_rank": args.max_rank,
            "allow_non_runtime_rankers": args.allow_non_runtime_rankers,
            **selection_manifest,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / args.target_name, targets)
    write_csv(args.output_dir / "ranker_replay_target_audit.csv", audit_rows)
    (args.output_dir / "ranker_replay_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "ranker_replay_target_summary.md", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
