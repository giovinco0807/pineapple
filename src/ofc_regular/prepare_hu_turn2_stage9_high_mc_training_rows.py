"""Convert Stage9 high-MC replay results into candidate-generator training rows.

The output JSONL is consumed by ``train_hu_turn2_pilot_model`` via
``--candidate-generator-training-jsonl``.  These rows are label-improvement
inputs only.  They are not seat-swap evidence and must not be treated as a
production runtime gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage9_high_mc_training_rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="stage9_high_mc_training_rows.jsonl")
    parser.add_argument("--positive-lcb", choices=("90", "95"), default="90")
    parser.add_argument("--positive-weight", type=float, default=6.0)
    parser.add_argument("--negative-weight", type=float, default=6.0)
    parser.add_argument("--gray-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=8.0)
    parser.add_argument("--include-gray", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_offsets(cache_dir: Path) -> np.ndarray:
    path = cache_dir / "sample_offsets.npy"
    if not path.exists():
        raise FileNotFoundError(f"cache offsets not found: {path}")
    offsets = np.load(path)
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"invalid cache offsets shape: {offsets.shape}")
    return np.asarray(offsets, dtype=np.int64)


def replay_fallback_candidate_id(row: dict[str, str], ordinal: int) -> str:
    state_index = row.get("state_index", f"row{ordinal}")
    candidate = row.get("candidate_action_local_index", "na")
    baseline = row.get("baseline_action_local_index", "na")
    return f"state{state_index}_cand{candidate}_base{baseline}_row{ordinal}"


def candidate_index(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for ordinal, row in enumerate(rows):
        key = str(row.get("candidate_id", ""))
        if key:
            index[key] = row
        index[replay_fallback_candidate_id(row, ordinal)] = row
    return index


def _label_for_result(result: dict[str, str], *, positive_lcb: str) -> tuple[str, int]:
    gain = safe_float(result.get("gain_mean"))
    positive_bound = safe_float(result.get("lower_bound_95" if positive_lcb == "95" else "lower_bound_90"))
    if positive_bound > 0.0:
        return f"high_mc_lcb{positive_lcb}_positive", 1
    if gain < 0.0:
        return "high_mc_hard_negative", 0
    return "high_mc_gray", 1


def _weight_for_result(
    result: dict[str, str],
    *,
    label: str,
    positive_weight: float,
    negative_weight: float,
    gray_weight: float,
    max_weight: float,
) -> float:
    gain = abs(safe_float(result.get("gain_mean")))
    stderr = max(safe_float(result.get("gain_stderr")), 1e-6)
    confidence = min(max_weight, 1.0 + min(gain / stderr, max_weight - 1.0))
    if label.startswith("high_mc_lcb"):
        return min(max_weight, max(positive_weight, confidence))
    if label == "high_mc_hard_negative":
        return min(max_weight, max(negative_weight, confidence))
    return max(0.0, min(max_weight, gray_weight))


def _action_row_index(offsets: np.ndarray, state_index: int, local_index: int) -> int | None:
    if state_index < 0 or state_index + 1 >= offsets.size:
        return None
    start = int(offsets[state_index])
    end = int(offsets[state_index + 1])
    if local_index < 0 or start + local_index >= end:
        return None
    return start + local_index


def build_high_mc_training_rows(
    *,
    offsets: np.ndarray,
    candidates: dict[str, dict[str, str]],
    results: list[dict[str, str]],
    positive_lcb: str,
    positive_weight: float,
    negative_weight: float,
    gray_weight: float,
    max_weight: float,
    include_gray: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for result in results:
        counters["results"] += 1
        if str(result.get("replay_status", "")) != "success":
            skipped["not_success"] += 1
            continue
        candidate_id = str(result.get("candidate_id", ""))
        candidate = candidates.get(candidate_id)
        if candidate is None:
            skipped["missing_candidate"] += 1
            continue
        state_index = safe_int(candidate.get("state_index"), -1)
        local_index = safe_int(candidate.get("candidate_action_local_index"), -1)
        action_row = _action_row_index(offsets, state_index, local_index)
        if action_row is None:
            skipped["invalid_action_index"] += 1
            continue
        label, target = _label_for_result(result, positive_lcb=positive_lcb)
        if label == "high_mc_gray" and not include_gray:
            skipped["gray"] += 1
            continue
        weight = _weight_for_result(
            result,
            label=label,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            gray_weight=gray_weight,
            max_weight=max_weight,
        )
        if weight <= 0.0:
            skipped["non_positive_weight"] += 1
            continue
        labels[label] += 1
        row = {
            "schema": "hu_turn2_stage9_high_mc_training_row_v1",
            "state_index": state_index,
            "action_index": local_index,
            "action_row_index": int(action_row),
            "label": label,
            "target": target,
            "weight": float(weight),
            "target_ev": safe_float(result.get("candidate_ev")),
            "delta_vs_baseline": safe_float(result.get("gain_mean")),
            "gain_stderr": safe_float(result.get("gain_stderr")),
            "lower_bound_90": safe_float(result.get("lower_bound_90")),
            "lower_bound_95": safe_float(result.get("lower_bound_95")),
            "mc_n": safe_int(result.get("mc_n"), 0),
            "split": candidate.get("split", ""),
            "source_reason": candidate.get("reason", ""),
            "candidate_id": candidate_id,
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
            "observed_performance_claim": "No",
        }
        output_rows.append(row)
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "state_index": state_index,
                "action_index": local_index,
                "label": label,
                "weight": float(weight),
                "delta_vs_baseline": row["delta_vs_baseline"],
                "gain_stderr": row["gain_stderr"],
                "lower_bound_90": row["lower_bound_90"],
                "lower_bound_95": row["lower_bound_95"],
                "mc_n": row["mc_n"],
                "split": row["split"],
                "source_reason": row["source_reason"],
            }
        )
    summary = {
        "input_results": counters["results"],
        "output_rows": len(output_rows),
        "label_counts": dict(sorted(labels.items())),
        "skipped": dict(sorted(skipped.items())),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    return output_rows, audit_rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = load_offsets(args.cache_dir)
    candidates = candidate_index(read_csv(args.candidates_csv))
    results = read_csv(args.results_csv)
    rows, audit_rows, summary = build_high_mc_training_rows(
        offsets=offsets,
        candidates=candidates,
        results=results,
        positive_lcb=args.positive_lcb,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        gray_weight=args.gray_weight,
        max_weight=args.max_weight,
        include_gray=args.include_gray,
    )
    output_path = args.output_dir / args.output_name
    write_jsonl(output_path, rows)
    write_csv(args.output_dir / "stage9_high_mc_training_rows_audit.csv", audit_rows)
    manifest = {
        "schema": "hu_turn2_stage9_high_mc_training_rows_v1",
        "cache_dir": str(args.cache_dir),
        "candidates_csv": str(args.candidates_csv),
        "results_csv": str(args.results_csv),
        "output_jsonl": str(output_path),
        "positive_lcb": args.positive_lcb,
        "positive_weight": args.positive_weight,
        "negative_weight": args.negative_weight,
        "gray_weight": args.gray_weight,
        "include_gray": bool(args.include_gray),
        **summary,
    }
    write_json(args.output_dir / "stage9_high_mc_training_rows_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
