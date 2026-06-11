"""Gate C1d threshold repair for HU Turn2 pilot calibration.

This is an evaluation-only gate. It uses the current C1/C1b/C1c calibration
artifacts to repair threshold candidates, expand high-MC candidate lists, and
decide whether a fixed-threshold C2-small heldout is justified.

It does not start production training, T1 training, or a 50k teacher run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_gate_c1_followup import (
    add_action_fields,
    classify_false_positive,
    enriched_state_rows,
    load_action_original_indices,
    margin_bucket_label,
    state_action_lookup_from_cache_inputs,
)
from .analyze_hu_turn2_gate_c1_large_calibration import (
    baseline_confidence_group,
    sample_source_group,
)
from .analyze_hu_turn2_pilot_calibration import load_model, rows_for_split, write_csv
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device


DEFAULT_CACHE_DIR = Path(
    "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_gate_c1_5k_mc512"
)
DEFAULT_MODEL = Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt")
DEFAULT_C1C_DIR = Path("outputs/hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy")
DEFAULT_OUTPUT_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1d_threshold_repair")
PRIMARY_SPLIT = "test"
ANALYSIS_SPLITS = ("val", "test", "holdout")
GO_MIN_FIRES = 30
CONDITIONAL_MIN_FIRES = 15
MAX_FP_RATE = 0.10
FATAL_LOSS_LIMIT = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--c1c-dir", type=Path, default=DEFAULT_C1C_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--high-mc-limit", type=int, default=120)
    parser.add_argument("--target-min-states", type=int, default=20_000)
    parser.add_argument("--target-max-states", type=int, default=30_000)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_cache_state_count(cache_dir: Path) -> int:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return 0
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata.get("state_count", 0) or 0)


def sample_status(state_count: int, target_min: int, target_max: int) -> str:
    if state_count >= target_min and state_count <= target_max:
        return "target_met"
    if state_count > target_max:
        return "above_target"
    return "additional_sample_not_available"


def source_position_key(row: dict[str, Any]) -> str:
    return f"{sample_source_group(row)}|{row.get('seat', 'unknown')}"


def c1d_strategy_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "best_diagnostic_m2p5_r0_g0p7",
            "threshold_family": "diagnostic_reference",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p64_positive": False,
            "requires_current_lcb_1p96_positive": False,
        },
        {
            "candidate_id": "confidence_lcb196_m2p5_r0_g0p7",
            "threshold_family": "lcb95_conservative",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p96_positive": True,
        },
        {
            "candidate_id": "confidence_lcb164_m2p5_r0_g0p7",
            "threshold_family": "lcb90_medium",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p64_positive": True,
        },
        {
            "candidate_id": "global_conservative_m3_r0_g0p9",
            "threshold_family": "global_conservative_fallback",
            "min_margin": 3.0,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.9,
        },
        {
            "candidate_id": "source_filtered_lcb196",
            "threshold_family": "source_filtered_lcb95",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p96_positive": True,
            "blocked_source_position": [("random_off_policy", "second")],
        },
        {
            "candidate_id": "position_specific_second_strict",
            "threshold_family": "position_specific_lcb95",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p96_positive": True,
            "seat_thresholds": {
                "second": {"min_margin": 3.0, "reference_min_margin": 0.0, "gate_threshold": 0.9}
            },
        },
    ]


def strategy_threshold_for(row: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    seat = str(row.get("seat", "unknown"))
    return dict(strategy.get("seat_thresholds", {}).get(seat, strategy))


def threshold_passes_strategy(row: dict[str, Any], strategy: dict[str, Any]) -> bool:
    if int(row["candidate_is_baseline"]):
        return False
    for source, seat in strategy.get("blocked_source_position", []):
        if sample_source_group(row) == source and str(row.get("seat", "unknown")) == seat:
            return False
    threshold = strategy_threshold_for(row, strategy)
    if safe_float(row["predicted_delta_vs_baseline"]) < safe_float(threshold.get("min_margin")):
        return False
    if safe_float(row["reference_margin_raw"]) < safe_float(threshold.get("reference_min_margin")):
        return False
    if safe_float(row["gate_probability"]) < safe_float(threshold.get("gate_threshold")):
        return False
    if strategy.get("requires_current_lcb_1p64_positive") and safe_float(row["gain_lcb_1p64"]) <= 0.0:
        return False
    if strategy.get("requires_current_lcb_1p96_positive") and safe_float(row["gain_lcb_1p96"]) <= 0.0:
        return False
    return True


def high_mc_key_for_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        safe_int(row.get("state_index")),
        safe_int(row.get("candidate_action_original_index")),
        safe_int(row.get("baseline_action_original_index")),
    )


def high_mc_key_for_result(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        safe_int(row.get("state_index")),
        safe_int(row.get("candidate_original_index")),
        safe_int(row.get("baseline_original_index")),
    )


def load_high_mc_results(c1c_dir: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = read_csv(c1c_dir / "high_mc_recheck_results_mc2048.csv")
    return {high_mc_key_for_result(row): row for row in rows}


def high_mc_fields(row: dict[str, Any], high_mc: dict[tuple[int, int, int], dict[str, Any]]) -> dict[str, Any]:
    result = high_mc.get(high_mc_key_for_row(row))
    if not result:
        return {
            "high_mc_available": 0,
            "high_mc_gain_mean": "",
            "high_mc_lower_bound_90": "",
            "high_mc_lower_bound_95": "",
            "high_mc_false_positive": "",
            "high_mc_lcb95_positive": "",
        }
    return {
        "high_mc_available": 1,
        "high_mc_gain_mean": result.get("gain_mean", ""),
        "high_mc_lower_bound_90": result.get("lower_bound_90", ""),
        "high_mc_lower_bound_95": result.get("lower_bound_95", ""),
        "high_mc_false_positive": result.get("false_positive_after_high_mc", ""),
        "high_mc_lcb95_positive": result.get("lower_bound_95_positive", ""),
    }


def summarize_gains(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [safe_float(row["actual_delta_candidate_vs_baseline"]) for row in rows]
    losses = [max(0.0, -gain) for gain in gains]
    return {
        "teacher_avg_gain": float(np.mean(gains)) if gains else 0.0,
        "teacher_median_gain": float(np.median(gains)) if gains else 0.0,
        "false_positive_count": sum(1 for gain in gains if gain < 0.0),
        "false_positive_rate": sum(1 for gain in gains if gain < 0.0) / max(len(gains), 1),
        "worst_override_loss": max(losses) if losses else 0.0,
        "min_gain": min(gains) if gains else 0.0,
    }


def c2_status_for_metrics(row: dict[str, Any], *, sample_state_count: int) -> tuple[str, str]:
    blockers: list[str] = []
    override_count = safe_int(row["override_count"])
    fp_rate = safe_float(row["false_positive_rate"])
    avg_gain = safe_float(row["teacher_avg_gain"])
    worst_loss = safe_float(row["worst_override_loss"])
    high_mc_available = safe_int(row["high_mc_available_count"])
    high_mc_lcb95 = safe_int(row["high_mc_lcb95_positive_count"])
    source_biased = bool(safe_int(row["source_biased"]))
    position_biased = bool(safe_int(row["position_biased"]))
    if sample_state_count < 20_000:
        blockers.append("additional_20k_30k_teacher_cache_not_available")
    if override_count < CONDITIONAL_MIN_FIRES:
        blockers.append("fires_lt_15")
    elif override_count < GO_MIN_FIRES:
        blockers.append("fires_lt_30")
    if fp_rate > MAX_FP_RATE:
        blockers.append("false_positive_rate_gt_10pct")
    if avg_gain <= 0.0:
        blockers.append("teacher_avg_gain_not_positive")
    if worst_loss > FATAL_LOSS_LIMIT:
        blockers.append("worst_override_loss_above_limit")
    if high_mc_available and high_mc_lcb95 < high_mc_available:
        blockers.append("not_all_high_mc_lcb95_positive")
    if source_biased:
        blockers.append("source_biased")
    if position_biased:
        blockers.append("position_biased")

    if (
        override_count >= GO_MIN_FIRES
        and fp_rate <= MAX_FP_RATE
        and avg_gain > 0.0
        and worst_loss <= FATAL_LOSS_LIMIT
        and not source_biased
        and not position_biased
        and sample_state_count >= 20_000
        and (not high_mc_available or high_mc_lcb95 == high_mc_available)
    ):
        return "go", ";".join(blockers)
    if (
        override_count >= CONDITIONAL_MIN_FIRES
        and fp_rate <= MAX_FP_RATE
        and avg_gain > 0.0
        and worst_loss <= FATAL_LOSS_LIMIT
    ):
        return "conditional_go", ";".join(blockers)
    return "no_go", ";".join(blockers)


def candidate_grid_rows(
    rows: list[dict[str, Any]],
    *,
    high_mc: dict[tuple[int, int, int], dict[str, Any]],
    sample_state_count: int,
    sample_status_value: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in c1d_strategy_specs():
        for split_name in ANALYSIS_SPLITS:
            subset = rows_for_split(rows, split_name)
            fired = [row for row in subset if threshold_passes_strategy(row, strategy)]
            gain_summary = summarize_gains(fired)
            source_counts = Counter(sample_source_group(row) for row in fired)
            seat_counts = Counter(str(row.get("seat", "unknown")) for row in fired)
            source_position_counts = Counter(source_position_key(row) for row in fired)
            high_mc_available = [row for row in fired if high_mc.get(high_mc_key_for_row(row))]
            high_mc_lcb95_positive = 0
            high_mc_fp = 0
            high_mc_gain_values: list[float] = []
            for row in high_mc_available:
                result = high_mc[high_mc_key_for_row(row)]
                high_mc_lcb95_positive += safe_int(result.get("lower_bound_95_positive"))
                high_mc_fp += safe_int(result.get("false_positive_after_high_mc"))
                high_mc_gain_values.append(safe_float(result.get("gain_mean")))
            override_count = len(fired)
            dominant_source_share = max(source_counts.values(), default=0) / max(override_count, 1)
            dominant_position_share = max(seat_counts.values(), default=0) / max(override_count, 1)
            dominant_source_position_share = max(source_position_counts.values(), default=0) / max(override_count, 1)
            metrics = {
                "candidate_id": strategy["candidate_id"],
                "threshold_family": strategy["threshold_family"],
                "split": split_name,
                "sample_status": sample_status_value,
                "sample_state_count": sample_state_count,
                "override_count": override_count,
                "override_rate": override_count / max(len(subset), 1),
                **gain_summary,
                "current_lcb90_positive_count": sum(1 for row in fired if safe_float(row["gain_lcb_1p64"]) > 0.0),
                "current_lcb95_positive_count": sum(1 for row in fired if safe_float(row["gain_lcb_1p96"]) > 0.0),
                "high_mc_available_count": len(high_mc_available),
                "high_mc_false_positive_count": high_mc_fp,
                "high_mc_false_positive_rate": high_mc_fp / max(len(high_mc_available), 1),
                "high_mc_lcb95_positive_count": high_mc_lcb95_positive,
                "high_mc_avg_gain": float(np.mean(high_mc_gain_values)) if high_mc_gain_values else 0.0,
                "source_counts": json.dumps(dict(source_counts), sort_keys=True),
                "position_counts": json.dumps(dict(seat_counts), sort_keys=True),
                "source_position_counts": json.dumps(dict(source_position_counts), sort_keys=True),
                "dominant_source_share": dominant_source_share,
                "dominant_position_share": dominant_position_share,
                "dominant_source_position_share": dominant_source_position_share,
                "source_biased": int(override_count > 0 and dominant_source_share >= 0.80),
                "position_biased": int(override_count > 0 and dominant_position_share >= 0.80),
                "source_position_biased": int(override_count > 0 and dominant_source_position_share >= 0.80),
            }
            status, blockers = c2_status_for_metrics(metrics, sample_state_count=sample_state_count)
            metrics["c2_small_status"] = status
            metrics["c2_small_blockers"] = blockers
            output.append(metrics)
    return output


def source_position_breakdown_rows(
    rows: list[dict[str, Any]],
    *,
    high_mc: dict[tuple[int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in c1d_strategy_specs():
        for split_name in ANALYSIS_SPLITS:
            subset = rows_for_split(rows, split_name)
            fired = [row for row in subset if threshold_passes_strategy(row, strategy)]
            groups = sorted({(sample_source_group(row), str(row.get("seat", "unknown"))) for row in subset})
            for source, seat in groups:
                group_total = [
                    row
                    for row in subset
                    if sample_source_group(row) == source and str(row.get("seat", "unknown")) == seat
                ]
                group_fired = [
                    row
                    for row in fired
                    if sample_source_group(row) == source and str(row.get("seat", "unknown")) == seat
                ]
                gain_summary = summarize_gains(group_fired)
                high_mc_rows = [row for row in group_fired if high_mc.get(high_mc_key_for_row(row))]
                output.append(
                    {
                        "candidate_id": strategy["candidate_id"],
                        "threshold_family": strategy["threshold_family"],
                        "split": split_name,
                        "source_group": source,
                        "position": seat,
                        "evaluated_states": len(group_total),
                        "override_count": len(group_fired),
                        "override_rate": len(group_fired) / max(len(group_total), 1),
                        **gain_summary,
                        "high_mc_available_count": len(high_mc_rows),
                        "high_mc_false_positive_count": sum(
                            safe_int(high_mc[high_mc_key_for_row(row)].get("false_positive_after_high_mc"))
                            for row in high_mc_rows
                        ),
                        "high_mc_lcb95_positive_count": sum(
                            safe_int(high_mc[high_mc_key_for_row(row)].get("lower_bound_95_positive"))
                            for row in high_mc_rows
                        ),
                    }
                )
    return output


def fired_event_rows(
    rows: list[dict[str, Any]],
    *,
    high_mc: dict[tuple[int, int, int], dict[str, Any]],
    action_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in c1d_strategy_specs():
        for row in rows_for_split(rows, PRIMARY_SPLIT):
            if not threshold_passes_strategy(row, strategy):
                continue
            enriched = add_action_fields(row, action_lookup)
            output.append(
                {
                    "candidate_id": strategy["candidate_id"],
                    "threshold_family": strategy["threshold_family"],
                    "state_index": row["state_index"],
                    "sample_id": enriched.get("sample_id", ""),
                    "hand_id": enriched.get("hand_id", ""),
                    "source_group": sample_source_group(row),
                    "run_bucket": row.get("run_bucket", ""),
                    "bucket_group": row.get("bucket_group", ""),
                    "position": row.get("seat", ""),
                    "predicted_delta_vs_baseline": row["predicted_delta_vs_baseline"],
                    "reference_margin_raw": row["reference_margin_raw"],
                    "gate_probability": row["gate_probability"],
                    "actual_delta_candidate_vs_baseline": row["actual_delta_candidate_vs_baseline"],
                    "false_positive_flag": int(safe_float(row["actual_delta_candidate_vs_baseline"]) < 0.0),
                    "candidate_loss": max(0.0, -safe_float(row["actual_delta_candidate_vs_baseline"])),
                    "gain_lcb_1p64": row["gain_lcb_1p64"],
                    "gain_lcb_1p96": row["gain_lcb_1p96"],
                    "candidate_action_local_index": row["candidate_action_local_index"],
                    "baseline_action_local_index": row["baseline_action_local_index"],
                    "teacher_best_action_local_index": row["teacher_best_action_local_index"],
                    "candidate_action_text": enriched.get("candidate_action_text", ""),
                    "baseline_action_text": enriched.get("baseline_action_text", ""),
                    "teacher_best_action_text": enriched.get("teacher_best_action_text", ""),
                    "error_type": classify_false_positive(row)
                    if safe_float(row["actual_delta_candidate_vs_baseline"]) < 0.0
                    else "",
                    **high_mc_fields(row, high_mc),
                }
            )
    return output


def false_positive_autopsy_rows(
    rows: list[dict[str, Any]],
    *,
    high_mc: dict[tuple[int, int, int], dict[str, Any]],
    action_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in fired_event_rows(rows, high_mc=high_mc, action_lookup=action_lookup)
        if int(row["false_positive_flag"])
    ]


def high_mc_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    action_lookup: dict[int, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: dict[tuple[int, int], dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str, priority: int) -> None:
        key = (safe_int(row["state_index"]), safe_int(row["candidate_action_local_index"]))
        current = candidates.get(key)
        if current and safe_int(current["priority"]) <= priority:
            return
        enriched = add_action_fields(row, action_lookup)
        requested = 4096 if reason in {"false_positive", "near_threshold"} else 2048
        candidates[key] = {
            "priority": priority,
            "reason": reason,
            "state_index": row["state_index"],
            "sample_id": enriched.get("sample_id", ""),
            "hand_id": enriched.get("hand_id", ""),
            "split": row["split"],
            "source_group": sample_source_group(row),
            "run_bucket": row.get("run_bucket", ""),
            "bucket_group": row.get("bucket_group", ""),
            "seat": row.get("seat", ""),
            "predicted_delta_vs_baseline": row["predicted_delta_vs_baseline"],
            "reference_margin_raw": row["reference_margin_raw"],
            "gate_probability": row["gate_probability"],
            "teacher_gain_mean_current_mc512": row["actual_delta_candidate_vs_baseline"],
            "teacher_gain_stderr_proxy_current_mc512": row["gain_stderr_proxy"],
            "gain_mean_minus_1p64_stderr_current_mc512": row["gain_lcb_1p64"],
            "gain_mean_minus_1p96_stderr_current_mc512": row["gain_lcb_1p96"],
            "accept_under_current_mc512_lcb_1p64": int(safe_float(row["gain_lcb_1p64"]) > 0.0),
            "accept_under_current_mc512_lcb_1p96": int(safe_float(row["gain_lcb_1p96"]) > 0.0),
            "candidate_action_local_index": row["candidate_action_local_index"],
            "baseline_action_local_index": row["baseline_action_local_index"],
            "teacher_best_action_local_index": row["teacher_best_action_local_index"],
            "candidate_action_original_index": row["candidate_action_original_index"],
            "baseline_action_original_index": row["baseline_action_original_index"],
            "teacher_best_action_original_index": row["teacher_best_action_original_index"],
            "candidate_action_text": enriched.get("candidate_action_text", ""),
            "baseline_action_text": enriched.get("baseline_action_text", ""),
            "teacher_best_action_text": enriched.get("teacher_best_action_text", ""),
            "current_mc_samples": 512,
            "requested_mc_samples": requested,
            "high_mc_status": "planned_or_existing_c1c_reuse",
            "accept_rule": "accept only if high-MC gain_mean - 1.96 * stderr > 0",
        }

    test_rows = rows_for_split(rows, PRIMARY_SPLIT)
    for strategy in c1d_strategy_specs():
        fired = [row for row in test_rows if threshold_passes_strategy(row, strategy)]
        for row in fired:
            add(row, "strategy_override", 3)
            if safe_float(row["actual_delta_candidate_vs_baseline"]) < 0.0:
                add(row, "false_positive", 1)
        for row in sorted(fired, key=lambda item: safe_float(item["actual_delta_candidate_vs_baseline"]), reverse=True)[:20]:
            add(row, "top_gain", 4)

    for row in test_rows:
        if int(row["candidate_is_baseline"]):
            continue
        near_lcb = abs(safe_float(row["gain_lcb_1p96"])) <= 0.20
        near_margin = 2.25 <= safe_float(row["predicted_delta_vs_baseline"]) <= 3.25
        near_gate = 0.60 <= safe_float(row["gate_probability"]) <= 0.95
        if near_lcb and near_margin and near_gate:
            add(row, "near_threshold", 2)

    output = sorted(
        candidates.values(),
        key=lambda item: (
            safe_int(item["priority"]),
            safe_float(item["gain_mean_minus_1p96_stderr_current_mc512"]),
            -safe_float(item["predicted_delta_vs_baseline"]),
        ),
    )
    return output[:limit]


def candidate_shortlist_rows(grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in grid_rows if row["split"] == PRIMARY_SPLIT]
    ordering = {"go": 0, "conditional_go": 1, "no_go": 2}
    test_rows.sort(
        key=lambda row: (
            ordering.get(str(row["c2_small_status"]), 9),
            safe_float(row["false_positive_rate"]),
            -safe_int(row["high_mc_lcb95_positive_count"]),
            -safe_int(row["override_count"]),
            -safe_float(row["teacher_avg_gain"]),
            safe_int(row["source_biased"]),
            safe_int(row["position_biased"]),
        )
    )
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(test_rows[:6], start=1):
        recommended = int(row["c2_small_status"] in {"go", "conditional_go"} and rank <= 3)
        output.append({"rank": rank, "recommended_for_c2_small": recommended, **row})
    return output


def write_go_nogo(path: Path, grid_rows: list[dict[str, Any]], *, sample_state_count: int) -> None:
    shortlist = candidate_shortlist_rows(grid_rows)
    ready = [row for row in shortlist if row["c2_small_status"] in {"go", "conditional_go"}]
    decision = "No-Go"
    if any(row["c2_small_status"] == "go" for row in shortlist):
        decision = "Go"
    elif ready:
        decision = "Conditional Go"
    lines = [
        "# Gate C1d Go/No-Go For C2 Small",
        "",
        f"- C2-small heldout: `{decision}`",
        "- production training: `No-Go`",
        "- T1 training: `No-Go`",
        "- 50k teacher: `No-Go`",
        f"- evaluated C1d cache states: `{sample_state_count}`",
        "",
        "## Candidate Shortlist",
        "",
        "| rank | candidate | status | fires | FP rate | avg gain | blockers |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in shortlist[:6]:
        lines.append(
            f"| {row['rank']} | {row['candidate_id']} | {row['c2_small_status']} | "
            f"{row['override_count']} | {safe_float(row['false_positive_rate']):.3f} | "
            f"{safe_float(row['teacher_avg_gain']):.4f} | {row['c2_small_blockers']} |"
        )
    lines.extend(
        [
            "",
            "C2-small must use fixed thresholds on a separate heldout seed. C1d does not authorize 50k teacher, T1, or production training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    *,
    grid_rows: list[dict[str, Any]],
    high_mc_rows: list[dict[str, Any]],
    sample_state_count: int,
    sample_status_value: str,
    elapsed_seconds: float,
) -> None:
    shortlist = candidate_shortlist_rows(grid_rows)
    best = shortlist[0] if shortlist else {}
    lines = [
        "# HU Turn2 Gate C1d Threshold Repair Summary",
        "",
        f"- evaluated states: `{sample_state_count}`",
        f"- sample status: `{sample_status_value}`",
        f"- high-MC candidates prepared: `{len(high_mc_rows)}`",
        f"- elapsed seconds: `{elapsed_seconds:.2f}`",
        "",
        "## Best Current Candidate",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- candidate: `{best['candidate_id']}`",
                f"- status: `{best['c2_small_status']}`",
                f"- fires: `{best['override_count']}`",
                f"- false positive rate: `{safe_float(best['false_positive_rate']):.4f}`",
                f"- average gain: `{safe_float(best['teacher_avg_gain']):.4f}`",
                f"- blockers: `{best['c2_small_blockers']}`",
            ]
        )
    else:
        lines.append("- no candidates")
    lines.extend(
        [
            "",
            "## Gate Decisions",
            "",
            "- production training: `No-Go`",
            "- T1 training: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- C2-small: `No-Go` unless a listed candidate reaches Go/Conditional Go on separate heldout",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    cache = load_cache(args.cache_dir)
    device = select_device(torch, args.device)
    net, stats, _model_payload = load_model(torch, args.model.resolve(), device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    targets = target_matrix(cache)
    action_original_indices = load_action_original_indices(args.cache_dir, len(cache["features"]))
    rows = enriched_state_rows(cache, predictions, targets, action_original_indices)
    state_count = load_cache_state_count(args.cache_dir) or len(rows)
    status_value = sample_status(state_count, args.target_min_states, args.target_max_states)
    high_mc = load_high_mc_results(args.c1c_dir)

    needed_indices = {int(row["state_index"]) for row in rows_for_split(rows, PRIMARY_SPLIT)}
    action_lookup = state_action_lookup_from_cache_inputs(cache, needed_indices)

    grid_rows = candidate_grid_rows(
        rows,
        high_mc=high_mc,
        sample_state_count=state_count,
        sample_status_value=status_value,
    )
    breakdown = source_position_breakdown_rows(rows, high_mc=high_mc)
    fired_rows = fired_event_rows(rows, high_mc=high_mc, action_lookup=action_lookup)
    false_positive_rows = false_positive_autopsy_rows(rows, high_mc=high_mc, action_lookup=action_lookup)
    high_mc_candidates = high_mc_candidate_rows(rows, action_lookup=action_lookup, limit=args.high_mc_limit)
    shortlist = candidate_shortlist_rows(grid_rows)
    lcb_rows = [
        row
        for row in fired_rows
        if row["candidate_id"]
        in {"confidence_lcb196_m2p5_r0_g0p7", "confidence_lcb164_m2p5_r0_g0p7", "source_filtered_lcb196"}
    ]
    existing_high_mc = read_csv(args.c1c_dir / "high_mc_recheck_results_mc2048.csv")
    c1d_high_mc_keys = {high_mc_key_for_row(row) for row in high_mc_candidates}
    c1d_existing_high_mc = [
        row for row in existing_high_mc if high_mc_key_for_result(row) in c1d_high_mc_keys
    ]

    write_csv(args.output_dir / "c1d_candidate_grid_metrics.csv", grid_rows)
    write_csv(args.output_dir / "c1d_lcb_candidates.csv", lcb_rows)
    write_csv(args.output_dir / "c1d_source_position_breakdown.csv", breakdown)
    write_csv(args.output_dir / "c1d_false_positive_autopsy.csv", false_positive_rows)
    write_csv(args.output_dir / "c1d_high_mc_candidates.csv", high_mc_candidates)
    write_csv(args.output_dir / "c1d_high_mc_results_mc2048.csv", c1d_existing_high_mc)
    write_csv(args.output_dir / "c1d_candidate_shortlist.csv", shortlist)
    write_go_nogo(args.output_dir / "go_nogo_for_c2_small.md", grid_rows, sample_state_count=state_count)
    write_summary(
        args.output_dir / "c1d_summary.md",
        grid_rows=grid_rows,
        high_mc_rows=high_mc_candidates,
        sample_state_count=state_count,
        sample_status_value=status_value,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    manifest = {
        "schema": "hu_turn2_gate_c1d_manifest_v1",
        "cache_dir": str(args.cache_dir),
        "model": str(args.model),
        "c1c_dir": str(args.c1c_dir),
        "output_dir": str(args.output_dir),
        "state_count": state_count,
        "sample_status": status_value,
        "high_mc_candidates": len(high_mc_candidates),
        "c2_small_ready_candidates": sum(1 for row in shortlist if row["c2_small_status"] in {"go", "conditional_go"}),
        "production_training": "No-Go",
        "t1_training": "No-Go",
        "teacher_50k": "No-Go",
    }
    (args.output_dir / "c1d_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
