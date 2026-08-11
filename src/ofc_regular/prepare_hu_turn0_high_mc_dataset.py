"""Validate, merge, and split high-MC HU T0 teacher rows."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .prepare_hu_turn0_candidate_dataset import (
    _seat_stratified_holdout,
    _write_jsonl,
    canonical_t0_state_key,
)
from .teacher import DEFAULT_FL_EV


REQUIRED_ACTION_FINITE_FIELDS = (
    "ev",
    "score",
    "se",
    "delta_vs_baseline",
    "delta_se_vs_baseline",
    "delta_z_vs_baseline",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["high_mc_input_path"] = str(path)
            row["high_mc_input_line"] = line_number
            rows.append(row)
    return rows


def _validate_row(
    row: dict[str, Any],
    *,
    min_future_samples: int,
    expected_evaluated_actions: int,
) -> None:
    location = f"{row.get('high_mc_input_path')}:{row.get('high_mc_input_line')}"
    if not str(row.get("phase", "")).startswith("hu_turn0"):
        raise ValueError(f"{location} is not a HU T0 row")
    future_samples = int(row.get("future_samples", 0))
    if future_samples < min_future_samples:
        raise ValueError(
            f"{location} future_samples={future_samples} < {min_future_samples}"
        )
    if int(row.get("total_legal_actions", 0)) != 232:
        raise ValueError(f"{location} does not report 232 total legal actions")
    actions = list(row.get("actions", ()))
    if int(row.get("evaluated_action_count", len(actions))) != expected_evaluated_actions:
        raise ValueError(f"{location} evaluated action count mismatch")
    if len(actions) != expected_evaluated_actions:
        raise ValueError(f"{location} action payload count mismatch")
    if not bool(row.get("common_random_futures_verified")):
        raise ValueError(f"{location} common random futures are not verified")
    if not bool(row.get("replay_ready")):
        raise ValueError(f"{location} is not replay-ready")
    expected_continuations = {
        "t1_continuation": "stage18_p1",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
    }
    for field, expected in expected_continuations.items():
        if str(row.get(field)) != expected:
            raise ValueError(f"{location} {field}={row.get(field)!r} != {expected!r}")
    if str(row.get("visibility_model")) != "hidden_discard":
        raise ValueError(f"{location} does not use hidden-discard visibility")
    if str(row.get("discard_visibility")) != "own_private_only":
        raise ValueError(f"{location} exposes opponent private discards")
    fl_ev_payload = row.get("fl_ev") or {}
    actual_fl_ev = float(fl_ev_payload.get("14", fl_ev_payload.get(14, float("nan"))))
    expected_fl_ev = float(DEFAULT_FL_EV[14])
    if not math.isfinite(actual_fl_ev) or abs(actual_fl_ev - expected_fl_ev) > 1e-9:
        raise ValueError(f"{location} FL EV {actual_fl_ev} != {expected_fl_ev}")
    selector = row.get("candidate_selector") or {}
    if str(selector.get("mode")) != "candidate_model_topk" or int(selector.get("topk", 0)) != 60:
        raise ValueError(f"{location} is not Opening Top60 candidate selection")

    digests = {str(action.get("common_random_future_digest")) for action in actions}
    if digests != {str(row.get("common_random_future_digest"))}:
        raise ValueError(f"{location} action future digests differ within the state")
    original_indices: set[int] = set()
    for action in actions:
        original_indices.add(int(action.get("original_index", action.get("action_index"))))
        if int(action.get("rollout_count", -1)) != future_samples:
            raise ValueError(f"{location} action rollout count mismatch")
        for field in REQUIRED_ACTION_FINITE_FIELDS:
            if not math.isfinite(float(action.get(field, float("nan")))):
                raise ValueError(f"{location} action field {field} is not finite")
    if len(original_indices) != len(actions):
        raise ValueError(f"{location} has duplicate original action indices")
    baseline_original_index = int(row["baseline_action_index"])
    best_original_index = int(row["best_action_index"])
    if baseline_original_index not in original_indices:
        raise ValueError(f"{location} baseline action is missing from evaluated candidates")
    if best_original_index not in original_indices:
        raise ValueError(f"{location} best action is missing from evaluated candidates")
    baseline_offsets = [
        offset
        for offset, action in enumerate(actions)
        if int(action.get("original_index", action.get("action_index")))
        == baseline_original_index
    ]
    best_offsets = [
        offset
        for offset, action in enumerate(actions)
        if int(action.get("original_index", action.get("action_index")))
        == best_original_index
    ]
    if len(baseline_offsets) != 1 or len(best_offsets) != 1:
        raise ValueError(f"{location} action index to candidate-row mapping is ambiguous")
    row["baseline_action_row_index"] = baseline_offsets[0]
    row["best_action_row_index"] = best_offsets[0]
    if not math.isfinite(float(row.get("delta_best_vs_baseline", float("nan")))):
        raise ValueError(f"{location} best delta is not finite")
    if not math.isfinite(float(row.get("delta_best_vs_baseline_se", float("nan")))):
        raise ValueError(f"{location} best delta SE is not finite")


def prepare_hu_turn0_high_mc_dataset(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    holdout_fraction: float = 0.20,
    seed: int = 2026105101,
    min_future_samples: int = 32,
    expected_evaluated_actions: int = 60,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one high-MC input is required")
    materialized: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    for path in inputs:
        rows = _read_jsonl(path)
        input_counts[str(path)] = len(rows)
        for row in rows:
            _validate_row(
                row,
                min_future_samples=min_future_samples,
                expected_evaluated_actions=expected_evaluated_actions,
            )
            row["dataset_state_id"] = canonical_t0_state_key(row)
            row["teacher_quality_source"] = (
                f"hu_turn0_terminal_rollout_mc{int(row['future_samples'])}"
            )
            row["source"] = row["teacher_quality_source"]
            materialized.append(row)

    by_key: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in materialized:
        key = str(row["dataset_state_id"])
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row
            continue
        duplicates += 1
        if int(row["future_samples"]) > int(previous["future_samples"]):
            by_key[key] = row
    unique = list(by_key.values())
    train, holdout = _seat_stratified_holdout(
        unique,
        fraction=holdout_fraction,
        seed=seed,
    )
    random.Random(seed).shuffle(train)

    train_ids = {str(row["dataset_state_id"]) for row in train}
    holdout_ids = {str(row["dataset_state_id"]) for row in holdout}
    split_overlap = len(train_ids & holdout_ids)
    if split_overlap:
        raise RuntimeError(f"high-MC state leakage across train/holdout: {split_overlap}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "hu_turn0_high_mc_merged.jsonl"
    train_path = output_dir / "train.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    _write_jsonl(merged_path, unique, split="all")
    _write_jsonl(train_path, train, split="train")
    _write_jsonl(holdout_path, holdout, split="holdout")

    deltas = [float(row["delta_best_vs_baseline"]) for row in unique]
    delta_ses = [float(row["delta_best_vs_baseline_se"]) for row in unique]
    summary = {
        "schema": "hu_turn0_high_mc_dataset_v1",
        "inputs": [str(path) for path in inputs],
        "input_counts": input_counts,
        "input_rows": len(materialized),
        "unique_rows": len(unique),
        "duplicate_states_dropped": duplicates,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_fraction": float(holdout_fraction),
        "seed": int(seed),
        "seat_counts": dict(sorted(Counter(str(row.get("seat")) for row in unique).items())),
        "train_seat_counts": dict(
            sorted(Counter(str(row.get("seat")) for row in train).items())
        ),
        "holdout_seat_counts": dict(
            sorted(Counter(str(row.get("seat")) for row in holdout).items())
        ),
        "min_future_samples": int(min_future_samples),
        "expected_evaluated_actions": int(expected_evaluated_actions),
        "fixed_continuations": {
            "t1": "stage18_p1",
            "t2": "stage9f_p2",
            "t3": "stage7_m5_r10",
        },
        "fl_ev_14": float(DEFAULT_FL_EV[14]),
        "visibility_model": "hidden_discard",
        "candidate_selector": "opening_stage7_top60",
        "split_overlap": split_overlap,
        "best_delta_mean": mean(deltas),
        "best_delta_se_mean": mean(delta_ses),
        "best_delta_positive_rate": sum(delta > 0.0 for delta in deltas) / len(deltas),
        "best_delta_lcb196_positive_rate": sum(
            delta - 1.96 * se > 0.0 for delta, se in zip(deltas, delta_ses, strict=True)
        )
        / len(deltas),
        "paths": {
            "merged": str(merged_path),
            "train": str(train_path),
            "holdout": str(holdout_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026105101)
    parser.add_argument("--min-future-samples", type=int, default=32)
    parser.add_argument("--expected-evaluated-actions", type=int, default=60)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = prepare_hu_turn0_high_mc_dataset(
        inputs=args.input,
        output_dir=args.output_dir,
        holdout_fraction=args.holdout,
        seed=args.seed,
        min_future_samples=args.min_future_samples,
        expected_evaluated_actions=args.expected_evaluated_actions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
