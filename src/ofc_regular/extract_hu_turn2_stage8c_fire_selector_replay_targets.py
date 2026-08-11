"""Extract replay targets from Stage8c fire-head prediction rows.

The fire-head probability is a deployable model output, but this extractor only
creates independent replay targets. It does not approve runtime, P2,
production, T1, or 50k teacher generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .extract_hu_turn2_stage8c_fire_veto_replay_targets import has_observed_delta
from .extract_hu_turn2_stage8c_topk_replay_targets import replay_missing_fields, row_key, safe_float, safe_int


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_fire_selector_replay_targets")
DEFAULT_TARGET_NAME = "fire_selector_replay_targets.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    parser.add_argument("--split", action="append", default=["test"], help="Split to include. Repeatable. Defaults to test.")
    parser.add_argument("--seat", action="append", default=[], help="Seat to include, e.g. first or second. Repeatable.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum fire-head probability.")
    parser.add_argument("--max-targets", type=int, default=50)
    parser.add_argument("--include-observed", action="store_true", help="Include rows with observed/replay delta.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def load_distillation_index(paths: Iterable[Path]) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, Any]]:
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    loaded = 0
    duplicates = 0
    for path in paths:
        for row in read_jsonl(path):
            loaded += 1
            key = row_key(row)
            if key in index:
                duplicates += 1
            index[key] = row
    return index, {"loaded_distillation_rows": loaded, "duplicate_distillation_keys": duplicates}


def selected_prediction_rows(
    rows: Iterable[dict[str, str]],
    *,
    splits: set[str] | None,
    seats: set[str] | None,
    threshold: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    selected: list[dict[str, str]] = []
    for row in rows:
        counters["input_rows"] += 1
        if splits is not None and str(row.get("split", "")) not in splits:
            counters["skipped_split_filter"] += 1
            continue
        if seats is not None and str(row.get("seat", "")) not in seats:
            counters["skipped_seat_filter"] += 1
            continue
        if safe_float(row.get("risk_probability")) < threshold:
            counters["skipped_threshold"] += 1
            continue
        selected.append(dict(row))
        counters["selected_prediction_rows"] += 1
    selected.sort(key=lambda row: safe_float(row.get("risk_probability")), reverse=True)
    return selected, {"selection_counts": dict(counters)}


def build_target(*, source_row: dict[str, Any], prediction_row: dict[str, str], selector_rank: int) -> dict[str, Any]:
    missing = replay_missing_fields(source_row)
    target = dict(source_row)
    target.update(
        {
            "schema": "hu_turn2_stage8c_fire_selector_replay_target_v1",
            "replay_target_reason": "fire_selector_high_probability_unknown_delta",
            "selector": "stage8c_fire_head_probability",
            "selector_rank": selector_rank,
            "ranker": "fire_selector_probability",
            "ranker_rank": selector_rank,
            "ranker_score": safe_float(prediction_row.get("risk_probability")),
            "fire_probability": safe_float(prediction_row.get("risk_probability")),
            "fire_selector_probability": safe_float(prediction_row.get("risk_probability")),
            "fire_selector_split": prediction_row.get("split", ""),
            "risk_target_group": prediction_row.get("risk_target_group", ""),
            "label_from_existing_cache": prediction_row.get("label", ""),
            "predicted_delta": safe_float(prediction_row.get("predicted_delta")),
            "gate_probability": safe_float(prediction_row.get("gate_probability")),
            "candidate_ev_rank": safe_int(prediction_row.get("candidate_ev_rank"), 9999),
            "realized_delta_observed": False,
            "observed_performance_claim": "No",
            "replay_ready": not missing,
            "replay_blocker": ",".join(missing),
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
            "note": (
                "Selected by fire-head probability for independent replay. "
                "This is target selection only, not runtime or performance evidence."
            ),
        }
    )
    return target


def extract_targets(
    distillation_index: dict[tuple[str, ...], dict[str, Any]],
    prediction_rows: list[dict[str, str]],
    *,
    include_observed: bool,
    max_targets: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()
    for prediction_row in prediction_rows:
        counters["selected_prediction_rows"] += 1
        key = row_key(prediction_row)
        source_row = distillation_index.get(key)
        if source_row is None:
            counters["missing_distillation_row"] += 1
            audit.append({"status": "missing_distillation_row", "hand_seed": prediction_row.get("hand_seed", "")})
            continue
        if not include_observed and (has_observed_delta(prediction_row) or has_observed_delta(source_row)):
            counters["skipped_observed_delta"] += 1
            continue
        if key in seen:
            counters["skipped_duplicate_target"] += 1
            continue
        target = build_target(source_row=source_row, prediction_row=prediction_row, selector_rank=len(targets) + 1)
        seen.add(key)
        targets.append(target)
        counters["targets"] += 1
        counters["replay_ready" if target["replay_ready"] else "replay_blocked"] += 1
        audit.append(
            {
                "status": "target",
                "hand_seed": target.get("hand_seed", ""),
                "seat": target.get("seat", ""),
                "selector_rank": target["selector_rank"],
                "fire_selector_probability": target["fire_selector_probability"],
                "replay_ready": target["replay_ready"],
                "replay_blocker": target["replay_blocker"],
            }
        )
        if max_targets > 0 and len(targets) >= max_targets:
            counters["stopped_max_targets"] += 1
            break
    return targets, audit, {"counts": dict(counters), "max_targets": max_targets, "include_observed": include_observed}


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# HU T2 Stage8c Fire Selector Replay Targets",
        "",
        "These rows are independent replay targets only. They do not approve runtime, production, P2, T1, or 50k teacher generation.",
        "",
        f"- targets: `{manifest.get('counts', {}).get('targets', 0)}`",
        f"- replay ready: `{manifest.get('counts', {}).get('replay_ready', 0)}`",
        f"- skipped observed delta: `{manifest.get('counts', {}).get('skipped_observed_delta', 0)}`",
        f"- selected seats: `{json.dumps(manifest.get('selected_seats', 'all'), sort_keys=True)}`",
        f"- selected splits: `{json.dumps(manifest.get('selected_splits', 'all'), sort_keys=True)}`",
        f"- threshold: `{manifest.get('threshold', 0.0)}`",
        "",
        "## Decision",
        "",
        "- allowed role: `fresh replay target selection`",
        "- runtime gate: `No-Go`",
        "- production / P2 fixed / T1 / 50k: `No-Go`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    splits = set(args.split) if args.split else None
    seats = set(args.seat) if args.seat else None
    distillation_index, distillation_manifest = load_distillation_index(args.distillation_jsonl)
    prediction_rows, selection_manifest = selected_prediction_rows(
        read_csv(args.predictions_csv),
        splits=splits,
        seats=seats,
        threshold=args.threshold,
    )
    targets, audit_rows, manifest = extract_targets(
        distillation_index,
        prediction_rows,
        include_observed=args.include_observed,
        max_targets=args.max_targets,
    )
    manifest.update(
        {
            **distillation_manifest,
            "predictions_csv": str(args.predictions_csv),
            "selected_splits": sorted(splits) if splits is not None else "all",
            "selected_seats": sorted(seats) if seats is not None else "all",
            "threshold": args.threshold,
            **selection_manifest,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / args.target_name, targets)
    write_csv(args.output_dir / "fire_selector_replay_target_audit.csv", audit_rows)
    (args.output_dir / "fire_selector_replay_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "fire_selector_replay_target_summary.md", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
